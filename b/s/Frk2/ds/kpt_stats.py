"""Compute and patch observation.keypoint_3d (56D) stats into a LeRobot v3.0 dataset.

The Franka v2 4D generation wrote the keypoint column into parquet / info.json, but
left episodes_stats.jsonl (and therefore the v3.0 aggregated meta/stats.json) at the
pre-keypoint snapshot. This module fills that gap without rewriting other features.

Stats format matches existing v30 meta: min/max/mean/std + count, no quantiles.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

KEYPOINT_COLUMN = "observation.keypoint_3d"
NUM_KEYPOINTS = 8
KEYPOINT_DIM = 7
TOTAL_DIM = NUM_KEYPOINTS * KEYPOINT_DIM  # 56
STAT_KEYS = ("min", "max", "mean", "std", "count")
VECTOR_STAT_KEYS = ("min", "max", "mean", "std")

DEFAULT_DATASET = Path.home() / "b/Dta/plug_into_socket_franka3_15hz_lerobot_4d_v30"

HF_LIST_F64 = {"feature": {"dtype": "float64", "_type": "Value"}, "_type": "List"}
HF_LIST_I64 = {"feature": {"dtype": "int64", "_type": "Value"}, "_type": "List"}


class KeypointStatsError(ValueError):
    """Raised when the dataset cannot support keypoint stats patching."""


def stack_list_column(values) -> np.ndarray:
    """Stack a parquet list-column into a 2D float64 array."""
    rows = [np.asarray(v, dtype=np.float64) for v in values]
    if not rows:
        raise KeypointStatsError("empty list column")
    arr = np.stack(rows, axis=0)
    return arr


def compute_vector_stats(arr: np.ndarray) -> dict[str, list]:
    """Population (ddof=0) min/max/mean/std, matching LeRobot RunningStats.

    Args:
        arr: shape [N, D] or [N].

    Returns:
        Dict with list-valued min/max/mean/std and count=[N], JSON-serializable.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise KeypointStatsError(f"expected 1D/2D array, got shape {arr.shape}")
    if arr.shape[0] < 1:
        raise KeypointStatsError("cannot compute stats over 0 rows")
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0, ddof=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def stats_to_float64_lists(stats: dict[str, list]) -> dict[str, list]:
    """Ensure JSON-friendly Python lists (ints for count, floats otherwise)."""
    out = {}
    for key in VECTOR_STAT_KEYS:
        out[key] = [float(x) for x in stats[key]]
    out["count"] = [int(stats["count"][0])]
    return out


def insert_after(mapping: dict, after_key: str, new_key: str, new_val: Any) -> dict:
    """Return a new dict with ``new_key`` placed immediately after ``after_key``."""
    if new_key in mapping:
        mapping = {k: v for k, v in mapping.items() if k != new_key}
    out: dict = {}
    inserted = False
    for k, v in mapping.items():
        out[k] = v
        if k == after_key:
            out[new_key] = new_val
            inserted = True
    if not inserted:
        out[new_key] = new_val
    return out


def find_data_parquet_files(dataset: Path) -> list[Path]:
    files = sorted(dataset.glob("data/**/*.parquet"))
    if not files:
        raise KeypointStatsError(f"no data parquet under {dataset}/data")
    return files


def find_episodes_parquet_files(dataset: Path) -> list[Path]:
    files = sorted((dataset / "meta" / "episodes").glob("**/*.parquet"))
    if not files:
        raise KeypointStatsError(f"no episodes parquet under {dataset}/meta/episodes")
    return files


def load_keypoint_frames(dataset: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all ``observation.keypoint_3d`` rows and their episode indices.

    Returns:
        kpts: [N, 56] float64
        episode_index: [N] int64
    """
    parts_k, parts_e = [], []
    for pq_path in find_data_parquet_files(dataset):
        table = pq.read_table(pq_path, columns=[KEYPOINT_COLUMN, "episode_index"])
        if KEYPOINT_COLUMN not in table.column_names:
            raise KeypointStatsError(f"{KEYPOINT_COLUMN} missing in {pq_path}")
        k = stack_list_column(table.column(KEYPOINT_COLUMN).to_pylist())
        if k.ndim != 2 or k.shape[1] != TOTAL_DIM:
            raise KeypointStatsError(
                f"{pq_path}: expected keypoint shape [N, {TOTAL_DIM}], got {k.shape}"
            )
        e = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
        if k.shape[0] != e.shape[0]:
            raise KeypointStatsError(f"{pq_path}: keypoint/episode_index length mismatch")
        parts_k.append(k)
        parts_e.append(e)
    kpts = np.concatenate(parts_k, axis=0)
    ep = np.concatenate(parts_e, axis=0)
    if np.isnan(kpts).any() or np.isinf(kpts).any():
        raise KeypointStatsError("NaN/Inf in observation.keypoint_3d")
    return kpts, ep


def compute_keypoint_stats(
    kpts: np.ndarray, episode_index: np.ndarray
) -> tuple[dict[str, list], dict[int, dict[str, list]]]:
    """Compute global and per-episode keypoint stats.

    Returns:
        (global_stats, {episode_index: stats})
    """
    global_stats = stats_to_float64_lists(compute_vector_stats(kpts))
    per_ep: dict[int, dict[str, list]] = {}
    for ep_id in np.unique(episode_index):
        mask = episode_index == int(ep_id)
        per_ep[int(ep_id)] = stats_to_float64_lists(compute_vector_stats(kpts[mask]))
    return global_stats, per_ep


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def backup_meta_files(dataset: Path, extra: list[Path] | None = None) -> Path:
    """Copy stats-related meta files to ``meta/_backup_kpt_stats_<timestamp>/``."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = dataset / "meta" / f"_backup_kpt_stats_{stamp}"
    dest.mkdir(parents=True, exist_ok=False)
    targets = [
        dataset / "meta" / "stats.json",
        dataset / "meta" / "episodes_stats.jsonl",
        *find_episodes_parquet_files(dataset),
    ]
    if extra:
        targets.extend(extra)
    for src in targets:
        if not src.exists():
            continue
        rel = src.relative_to(dataset / "meta")
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        logger.info("backed up %s -> %s", src, out)
    return dest


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, obj: Any, *, indent: int | None = 4) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)
        f.write("\n")


def patch_stats_json(path: Path, kpt_stats: dict[str, list]) -> None:
    stats = _load_json(path)
    if not isinstance(stats, dict):
        raise KeypointStatsError(f"{path} is not a JSON object")
    if KEYPOINT_COLUMN in stats:
        stats[KEYPOINT_COLUMN] = kpt_stats
        logger.info("updated existing %s in %s", KEYPOINT_COLUMN, path)
    else:
        stats = insert_after(stats, "observation.wrench", KEYPOINT_COLUMN, kpt_stats)
        logger.info("inserted %s into %s", KEYPOINT_COLUMN, path)
    _write_json(path, stats, indent=4)


def patch_episodes_stats_jsonl(
    path: Path, per_ep: dict[int, dict[str, list]]
) -> None:
    if not path.exists():
        raise KeypointStatsError(f"missing {path}")
    lines = path.read_text().splitlines()
    out_lines = []
    seen: set[int] = set()
    for line in lines:
        if not line.strip():
            continue
        obj = json.loads(line)
        ep_id = int(obj["episode_index"])
        if ep_id not in per_ep:
            raise KeypointStatsError(f"{path}: episode {ep_id} has no keypoint frames")
        if "stats" not in obj or not isinstance(obj["stats"], dict):
            raise KeypointStatsError(f"{path}: episode {ep_id} missing stats dict")
        obj["stats"][KEYPOINT_COLUMN] = per_ep[ep_id]
        out_lines.append(json.dumps(obj, ensure_ascii=False))
        seen.add(ep_id)
    missing = sorted(set(per_ep) - seen)
    extra = sorted(seen - set(per_ep))
    if missing or extra:
        raise KeypointStatsError(
            f"{path}: episode set mismatch missing={missing[:5]} extra={extra[:5]}"
        )
    path.write_text("\n".join(out_lines) + "\n")
    logger.info("wrote %d episodes to %s", len(out_lines), path)


def _drop_columns(table: pa.Table, names: set[str]) -> pa.Table:
    keep = [n for n in table.column_names if n not in names]
    return table.select(keep)


def _hf_features_from_metadata(schema: pa.Schema) -> dict | None:
    meta = schema.metadata or {}
    raw = meta.get(b"huggingface")
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
        return payload
    except json.JSONDecodeError:
        return None


def _with_kpt_hf_features(payload: dict) -> dict:
    features = payload.setdefault("info", {}).setdefault("features", {})
    for sk in VECTOR_STAT_KEYS:
        features[f"stats/{KEYPOINT_COLUMN}/{sk}"] = dict(HF_LIST_F64)
    features[f"stats/{KEYPOINT_COLUMN}/count"] = dict(HF_LIST_I64)
    return payload


def patch_episodes_parquet(
    path: Path, per_ep: dict[int, dict[str, list]], episode_ids_in_file: list[int]
) -> None:
    """Append/replace keypoint stats columns without rewriting nested types."""
    table = pq.read_table(path)
    if "episode_index" not in table.column_names:
        raise KeypointStatsError(f"{path} missing episode_index")
    ep_ids = [int(x) for x in table.column("episode_index").to_pylist()]
    if ep_ids != episode_ids_in_file:
        # Allow caller to pass the file's own ids; still require per_ep coverage.
        episode_ids_in_file = ep_ids

    kpt_cols = {f"stats/{KEYPOINT_COLUMN}/{sk}" for sk in STAT_KEYS}
    table = _drop_columns(table, kpt_cols)

    def col_for(stat_key: str) -> list:
        rows = []
        for ep_id in episode_ids_in_file:
            if ep_id not in per_ep:
                raise KeypointStatsError(f"{path}: episode {ep_id} missing keypoint stats")
            rows.append(per_ep[ep_id][stat_key])
        return rows

    new_arrays: dict[str, pa.Array] = {}
    for sk in VECTOR_STAT_KEYS:
        new_arrays[f"stats/{KEYPOINT_COLUMN}/{sk}"] = pa.array(
            col_for(sk), type=pa.list_(pa.float64())
        )
    new_arrays[f"stats/{KEYPOINT_COLUMN}/count"] = pa.array(
        col_for("count"), type=pa.list_(pa.int64())
    )

    names = table.column_names
    insert_at = (
        names.index("meta/episodes/chunk_index")
        if "meta/episodes/chunk_index" in names
        else len(names)
    )

    arrays = []
    fields = []
    for i, name in enumerate(names):
        if i == insert_at:
            for nn, arr in new_arrays.items():
                arrays.append(arr)
                fields.append(pa.field(nn, arr.type))
        arrays.append(table.column(name))
        fields.append(table.schema.field(name))
    if insert_at == len(names):
        for nn, arr in new_arrays.items():
            arrays.append(arr)
            fields.append(pa.field(nn, arr.type))

    new_schema = pa.schema(fields, metadata=table.schema.metadata)
    payload = _hf_features_from_metadata(table.schema)
    if payload is not None:
        payload = _with_kpt_hf_features(payload)
        meta = dict(table.schema.metadata or {})
        meta[b"huggingface"] = json.dumps(payload, separators=(",", ":")).encode()
        new_schema = new_schema.with_metadata(meta)

    new_table = pa.Table.from_arrays(arrays, schema=new_schema)
    pq.write_table(new_table, path)
    logger.info("updated %s with %s stats columns", path, KEYPOINT_COLUMN)


@dataclass
class PatchResult:
    dataset: Path
    n_frames: int
    n_episodes: int
    global_stats: dict[str, list]
    backup_dir: Path | None = None
    data_parquet_sha256_before: dict[str, str] = field(default_factory=dict)
    data_parquet_sha256_after: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False


def patch_dataset(
    dataset: Path,
    *,
    dry_run: bool = False,
    backup: bool = True,
    skip_if_present: bool = False,
) -> PatchResult:
    """Patch v3.0 meta so ``observation.keypoint_3d`` has full norm stats."""
    dataset = Path(dataset).resolve()
    stats_path = dataset / "meta" / "stats.json"
    jsonl_path = dataset / "meta" / "episodes_stats.jsonl"
    info_path = dataset / "meta" / "info.json"

    if not stats_path.exists():
        raise KeypointStatsError(f"missing {stats_path} (not a v3.0 dataset with stats.json?)")
    if not info_path.exists():
        raise KeypointStatsError(f"missing {info_path}")

    info = _load_json(info_path)
    feats = info.get("features", {})
    if KEYPOINT_COLUMN not in feats:
        raise KeypointStatsError(f"{info_path} has no {KEYPOINT_COLUMN} feature")
    shape = feats[KEYPOINT_COLUMN].get("shape")
    if list(shape) != [TOTAL_DIM]:
        raise KeypointStatsError(f"{KEYPOINT_COLUMN} shape {shape} != [{TOTAL_DIM}]")

    existing_stats = _load_json(stats_path)
    if skip_if_present and KEYPOINT_COLUMN in existing_stats:
        logger.info("skip_if_present: %s already in stats.json", KEYPOINT_COLUMN)
        kpts, ep = load_keypoint_frames(dataset)
        global_stats, _ = compute_keypoint_stats(kpts, ep)
        return PatchResult(
            dataset=dataset,
            n_frames=int(kpts.shape[0]),
            n_episodes=int(len(np.unique(ep))),
            global_stats=global_stats,
            dry_run=True,
        )

    kpts, ep = load_keypoint_frames(dataset)
    n_frames = int(kpts.shape[0])
    n_episodes = int(len(np.unique(ep)))
    declared_frames = int(info.get("total_frames", n_frames))
    declared_eps = int(info.get("total_episodes", n_episodes))
    if n_frames != declared_frames:
        raise KeypointStatsError(
            f"parquet frames {n_frames} != info.total_frames {declared_frames}"
        )
    if n_episodes != declared_eps:
        raise KeypointStatsError(
            f"parquet episodes {n_episodes} != info.total_episodes {declared_eps}"
        )

    global_stats, per_ep = compute_keypoint_stats(kpts, ep)
    logger.info(
        "computed %s stats: frames=%d episodes=%d dim=%d max|pos|=%.6f",
        KEYPOINT_COLUMN,
        n_frames,
        n_episodes,
        TOTAL_DIM,
        float(np.abs(kpts.reshape(-1, NUM_KEYPOINTS, KEYPOINT_DIM)[:, :, :3]).max()),
    )

    data_files = find_data_parquet_files(dataset)
    sha_before = {str(p): file_sha256(p) for p in data_files}

    backup_dir = None
    if dry_run:
        logger.info("dry-run: not writing meta files")
        return PatchResult(
            dataset=dataset,
            n_frames=n_frames,
            n_episodes=n_episodes,
            global_stats=global_stats,
            data_parquet_sha256_before=sha_before,
            data_parquet_sha256_after=sha_before,
            dry_run=True,
        )

    if backup:
        backup_dir = backup_meta_files(dataset)

    patch_stats_json(stats_path, global_stats)
    if jsonl_path.exists():
        patch_episodes_stats_jsonl(jsonl_path, per_ep)
    else:
        logger.warning("no %s — skipped jsonl patch", jsonl_path)

    for ep_path in find_episodes_parquet_files(dataset):
        ep_table = pq.read_table(ep_path, columns=["episode_index"])
        ids = [int(x) for x in ep_table.column("episode_index").to_pylist()]
        patch_episodes_parquet(ep_path, per_ep, ids)

    sha_after = {str(p): file_sha256(p) for p in data_files}
    if sha_before != sha_after:
        raise KeypointStatsError("data parquet changed during patch — aborting invariant")

    return PatchResult(
        dataset=dataset,
        n_frames=n_frames,
        n_episodes=n_episodes,
        global_stats=global_stats,
        backup_dir=backup_dir,
        data_parquet_sha256_before=sha_before,
        data_parquet_sha256_after=sha_after,
        dry_run=False,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

ABS_TOL = 1e-9
REL_TOL = 1e-6
# Pre-existing columns were aggregated from v2.1 episode stats; gripper/quat
# std already differed from a full-parquet recompute by ~5e-8 before this patch.
PREEXISTING_ABS_TOL = 1e-6
POS_BOUND = 1.01
QUAT_NORM_TOL = 1e-3


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _close(a, b, abs_tol=ABS_TOL, rel_tol=REL_TOL) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    return np.allclose(a, b, atol=abs_tol, rtol=rel_tol, equal_nan=False)


def verify_dataset(dataset: Path) -> list[CheckResult]:
    """Full acceptance checks for patched v30 keypoint stats."""
    dataset = Path(dataset).resolve()
    results: list[CheckResult] = []

    def add(name: str, ok: bool, detail: str) -> None:
        results.append(CheckResult(name, bool(ok), detail))

    info_path = dataset / "meta" / "info.json"
    stats_path = dataset / "meta" / "stats.json"
    jsonl_path = dataset / "meta" / "episodes_stats.jsonl"

    info = _load_json(info_path)
    feats = info["features"]
    add(
        "info.json declares keypoint_3d [56]",
        KEYPOINT_COLUMN in feats and list(feats[KEYPOINT_COLUMN].get("shape")) == [TOTAL_DIM],
        f"shape={feats.get(KEYPOINT_COLUMN, {}).get('shape')}",
    )

    stats = _load_json(stats_path)
    has = KEYPOINT_COLUMN in stats
    add("stats.json contains observation.keypoint_3d", has, f"keys={list(stats)}")
    if not has:
        return results

    kpt_s = stats[KEYPOINT_COLUMN]
    missing_sub = [k for k in STAT_KEYS if k not in kpt_s]
    add("stats.json kpt has min/max/mean/std/count", not missing_sub, f"missing={missing_sub}")
    dims_ok = all(len(kpt_s[k]) == TOTAL_DIM for k in VECTOR_STAT_KEYS) and len(kpt_s["count"]) == 1
    add(
        "stats.json kpt dims = 56 / count len = 1",
        dims_ok,
        f"mean_len={len(kpt_s.get('mean', []))} count={kpt_s.get('count')}",
    )

    kpts, ep = load_keypoint_frames(dataset)
    rec_global, rec_per_ep = compute_keypoint_stats(kpts, ep)
    add(
        "stats.json kpt count == parquet frames",
        int(kpt_s["count"][0]) == kpts.shape[0],
        f"stats={kpt_s['count'][0]} parquet={kpts.shape[0]}",
    )
    for sk in VECTOR_STAT_KEYS:
        add(
            f"stats.json kpt {sk} matches parquet recompute",
            _close(kpt_s[sk], rec_global[sk]),
            f"max_abs={float(np.max(np.abs(np.asarray(kpt_s[sk]) - np.asarray(rec_global[sk])))):.3e}",
        )

    # Existing numeric features (if present) still match parquet — state/action/force/wrench.
    data_files = find_data_parquet_files(dataset)
    table = pq.read_table(data_files[0]) if len(data_files) == 1 else None
    if table is not None:
        for col in ("observation.state", "action", "observation.force", "observation.wrench"):
            if col not in stats or col not in table.column_names:
                continue
            arr = stack_list_column(table.column(col).to_pylist())
            rec = compute_vector_stats(arr)
            diffs = {
                sk: float(
                    np.max(np.abs(np.asarray(stats[col][sk], dtype=np.float64) - np.asarray(rec[sk])))
                )
                for sk in VECTOR_STAT_KEYS
            }
            ok_all = all(
                _close(stats[col][sk], rec[sk], abs_tol=PREEXISTING_ABS_TOL, rel_tol=REL_TOL)
                for sk in VECTOR_STAT_KEYS
            )
            ok_all = ok_all and int(stats[col]["count"][0]) == arr.shape[0]
            add(
                f"pre-existing {col} stats still match parquet",
                ok_all,
                f"count={stats[col]['count'][0]} vs {arr.shape[0]} max_abs={diffs}",
            )

    backups = sorted((dataset / "meta").glob("_backup_kpt_stats_*"), reverse=True)
    if backups:
        bak_stats_path = backups[0] / "stats.json"
        if bak_stats_path.exists():
            old = _load_json(bak_stats_path)
            mutated = []
            for key in old:
                if key not in stats:
                    mutated.append(f"dropped {key}")
                    continue
                if json.dumps(old[key], sort_keys=True) != json.dumps(stats[key], sort_keys=True):
                    mutated.append(key)
            add(
                "backup stats.json non-kpt keys unchanged",
                not mutated,
                f"backup={backups[0].name} mutated={mutated[:5]}",
            )

    # jsonl coverage
    if jsonl_path.exists():
        n_jsonl = 0
        jsonl_ok = True
        details = []
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                n_jsonl += 1
                ep_id = int(obj["episode_index"])
                if KEYPOINT_COLUMN not in obj.get("stats", {}):
                    jsonl_ok = False
                    details.append(f"ep {ep_id} missing kpt")
                    continue
                js = obj["stats"][KEYPOINT_COLUMN]
                rec = rec_per_ep[ep_id]
                if not all(_close(js[sk], rec[sk]) for sk in VECTOR_STAT_KEYS):
                    jsonl_ok = False
                    details.append(f"ep {ep_id} numeric mismatch")
                if int(js["count"][0]) != int(rec["count"][0]):
                    jsonl_ok = False
                    details.append(f"ep {ep_id} count {js['count']} vs {rec['count']}")
        add(
            "episodes_stats.jsonl has kpt for every episode",
            jsonl_ok and n_jsonl == len(rec_per_ep),
            f"n_jsonl={n_jsonl} n_ep={len(rec_per_ep)} {details[:3]}",
        )
    else:
        add("episodes_stats.jsonl present", False, "file missing")

    # episodes parquet
    ep_files = find_episodes_parquet_files(dataset)
    pq_ok = True
    pq_detail = []
    n_ep_rows = 0
    for ep_path in ep_files:
        et = pq.read_table(ep_path)
        n_ep_rows += et.num_rows
        for sk in STAT_KEYS:
            col = f"stats/{KEYPOINT_COLUMN}/{sk}"
            if col not in et.column_names:
                pq_ok = False
                pq_detail.append(f"missing {col}")
        if not pq_ok:
            continue
        expected_type = pa.list_(pa.float64())
        for sk in VECTOR_STAT_KEYS:
            col = f"stats/{KEYPOINT_COLUMN}/{sk}"
            if et.schema.field(col).type != expected_type:
                pq_ok = False
                pq_detail.append(f"{col} type={et.schema.field(col).type}")
        count_t = et.schema.field(f"stats/{KEYPOINT_COLUMN}/count").type
        if count_t != pa.list_(pa.int64()):
            pq_ok = False
            pq_detail.append(f"count type={count_t}")
        ep_ids = [int(x) for x in et.column("episode_index").to_pylist()]
        for i, ep_id in enumerate(ep_ids):
            rec = rec_per_ep[ep_id]
            for sk in VECTOR_STAT_KEYS:
                val = et.column(f"stats/{KEYPOINT_COLUMN}/{sk}")[i].as_py()
                if not _close(val, rec[sk]):
                    pq_ok = False
                    pq_detail.append(f"ep {ep_id} {sk} mismatch")
                    break
            cnt = et.column(f"stats/{KEYPOINT_COLUMN}/count")[i].as_py()
            if int(cnt[0]) != int(rec["count"][0]):
                pq_ok = False
                pq_detail.append(f"ep {ep_id} count mismatch")
        # nested image stats still list<list<list<double>>>
        img_col = "stats/observation.images.global/min"
        if img_col in et.column_names:
            t = et.schema.field(img_col).type
            if str(t) != "list<element: list<element: list<element: double>>>":
                pq_ok = False
                pq_detail.append(f"image stats type corrupted: {t}")
        payload = _hf_features_from_metadata(et.schema)
        if payload is not None:
            hf_feats = payload.get("info", {}).get("features", {})
            if f"stats/{KEYPOINT_COLUMN}/mean" not in hf_feats:
                pq_ok = False
                pq_detail.append("huggingface schema missing kpt stats")
    add(
        "episodes parquet kpt stats complete and typed",
        pq_ok and n_ep_rows == len(rec_per_ep),
        f"rows={n_ep_rows} {pq_detail[:4]}",
    )

    # geometric sanity on the keypoint values themselves
    k7 = kpts.reshape(-1, NUM_KEYPOINTS, KEYPOINT_DIM)
    pos = k7[:, :, :3]
    quat = k7[:, :, 3:7]
    pos_max = float(np.abs(pos).max())
    add("keypoint |pos| <= 1.01 (R_pad space)", pos_max <= POS_BOUND, f"max|pos|={pos_max:.6f}")
    qn = np.linalg.norm(quat.reshape(-1, 4), axis=1)
    qn_err = float(np.abs(qn - 1.0).max())
    add("quaternion unit norm", qn_err <= QUAT_NORM_TOL, f"max |‖q‖-1|={qn_err:.2e}")
    qw_min = float(quat[:, :, 3].min())
    add("hemisphere qw >= 0", qw_min >= -1e-7, f"qw_min={qw_min:.8f}")

    # weighted aggregation of per-ep mean equals global mean
    counts = np.array([rec_per_ep[e]["count"][0] for e in sorted(rec_per_ep)], dtype=np.float64)
    means = np.stack([rec_per_ep[e]["mean"] for e in sorted(rec_per_ep)])
    agg_mean = (means * counts[:, None]).sum(0) / counts.sum()
    add(
        "episode-weighted mean == global mean",
        _close(agg_mean, rec_global["mean"]),
        f"max_abs={float(np.max(np.abs(agg_mean - np.asarray(rec_global['mean'])))):.3e}",
    )

    add(
        "info.json total_frames == parquet rows",
        int(info["total_frames"]) == kpts.shape[0],
        f"{info['total_frames']} vs {kpts.shape[0]}",
    )
    add(
        "info.json total_episodes == unique episode_index",
        int(info["total_episodes"]) == len(rec_per_ep),
        f"{info['total_episodes']} vs {len(rec_per_ep)}",
    )

    return results
