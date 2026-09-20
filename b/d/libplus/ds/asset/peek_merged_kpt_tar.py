#!/usr/bin/env python3
"""Pull the authoritative keypoint metadata out of the merged_kpt tarball.

`opvla_libero_merged_kpt` is the LeRobot dataset the policy was actually
trained on: it carries `observation.keypoint_3d` plus the `keypoints_meta.json`
that records the R_pad used to normalize it. Only the 1.8 GB archive survives
on this host, so rather than unpacking it we stream it once and keep just the
few members we need, in memory.

This settles two long-running disputes in b/d/libplus/:
  * whether R_pad is 1.8212722539901733 or 2.0945
  * whether the merged dataset is 10 Hz or 20 Hz

and gives `analyze_4d.py` real training keypoints to diff our FK against.

Runs on the base python3 (needs pyarrow); no mujoco required.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TARBALL = Path("/B/Dta/opvla_libero_merged_kpt.tar.gz")
OUT_NPZ = Path(__file__).with_name("merged_kpt_probe.npz")
OUT_JSON = Path(__file__).with_name("merged_kpt_meta.json")

WANTED_META = {
    "meta/keypoints_meta.json",
    "meta/info.json",
}
MAX_PROBE_ROWS = 4000


def _rel(name: str) -> str:
    """Strip the single `libero_merged_kpt/` top-level directory."""
    return name.split("/", 1)[1] if "/" in name else name


def stream_members() -> tuple[dict, bytes | None, bytes | None]:
    meta: dict[str, dict] = {}
    data_parquet: bytes | None = None
    episodes_parquet: bytes | None = None

    with tarfile.open(TARBALL, mode="r|gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            rel = _rel(member.name)
            if rel in WANTED_META:
                meta[rel] = json.loads(tf.extractfile(member).read().decode())
            elif rel.startswith("meta/episodes/") and rel.endswith(".parquet") and episodes_parquet is None:
                episodes_parquet = tf.extractfile(member).read()
            elif rel.startswith("data/") and rel.endswith(".parquet") and data_parquet is None:
                data_parquet = tf.extractfile(member).read()
                break  # data/ is the last group in this archive
    return meta, data_parquet, episodes_parquet


def main() -> int:
    if not TARBALL.exists():
        print(f"missing {TARBALL}")
        return 1

    print(f"streaming {TARBALL} ({TARBALL.stat().st_size / 2**30:.2f} GiB compressed) ...")
    meta, data_parquet, episodes_parquet = stream_members()

    kmeta = meta.get("meta/keypoints_meta.json", {})
    info = meta.get("meta/info.json", {})

    print("\n=== meta/keypoints_meta.json ===")
    for key in (
        "bbox_radius",
        "bbox_margin",
        "global_min_world",
        "global_max_world",
        "normalization",
        "coordinate_system",
        "keypoint_dim_layout",
        "rotation_representation",
        "num_keypoints",
        "keypoint_bodies",
        "qpos_layout",
        "total_frames",
        "mjcf_path",
    ):
        if key in kmeta:
            print(f"  {key:26s} {kmeta[key]}")

    print("\n=== meta/info.json ===")
    for key in ("codebase_version", "robot_type", "fps", "total_episodes", "total_frames", "total_tasks"):
        if key in info:
            print(f"  {key:26s} {info[key]}")
    feats = info.get("features", {})
    print(f"  {'features':26s} {list(feats)}")
    for key in ("observation.state", "observation.keypoint_3d", "action"):
        if key in feats:
            print(f"    {key:24s} shape={feats[key].get('shape')} dtype={feats[key].get('dtype')}")
    kpt_names = feats.get("observation.keypoint_3d", {}).get("names")
    if kpt_names:
        print(f"    keypoint_3d names[:7] {kpt_names[:7]}")

    payload: dict[str, np.ndarray] = {}
    if data_parquet is not None:
        table = pq.read_table(io.BytesIO(data_parquet))
        print(f"\n=== data parquet ({table.num_rows} rows) ===")
        print(f"  columns: {table.column_names}")
        n = min(MAX_PROBE_ROWS, table.num_rows)
        for col, key in (
            ("observation.keypoint_3d", "keypoint_3d"),
            ("observation.state", "state"),
            ("observation.state.joint_position", "joint_position"),
            ("episode_index", "episode_index"),
            ("frame_index", "frame_index"),
            ("task_index", "task_index"),
        ):
            if col not in table.column_names:
                continue
            values = table.column(col).slice(0, n).to_pylist()
            payload[key] = np.asarray(values, dtype=np.float64 if isinstance(values[0], list) else np.int64)
            print(f"  kept {key:16s} {payload[key].shape}")
    else:
        print("\n(no data parquet found in archive)")

    if episodes_parquet is not None:
        table = pq.read_table(io.BytesIO(episodes_parquet))
        print(f"\n=== meta/episodes parquet ({table.num_rows} rows) ===")
        print(f"  columns: {table.column_names}")
        for col in ("tasks", "meta/episodes/length", "length"):
            if col in table.column_names:
                sample = table.column(col).slice(0, 3).to_pylist()
                print(f"  {col}: {sample}")

    OUT_JSON.write_text(json.dumps({"keypoints_meta": kmeta, "info": info}, indent=2))
    print(f"\nwrote {OUT_JSON}")
    if payload:
        np.savez_compressed(OUT_NPZ, **payload)
        print(f"wrote {OUT_NPZ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
