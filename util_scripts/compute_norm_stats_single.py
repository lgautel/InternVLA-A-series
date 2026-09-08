#!/usr/bin/env python

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm

from lerobot.dataset_schemas import get_schema
from lerobot.datasets.utils import write_json
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE

QUANTILE_PERCENTILES = [1, 10, 50, 90, 99]
QUANTILE_KEYS = ["q01", "q10", "q50", "q90", "q99"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics for LeRobot datasets"
    )

    parser.add_argument(
        "--action_mode",
        type=str,
        choices=["abs", "delta"],
        required=True,
        help="Action mode used to compute statistics (abs or delta).",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        required=True,
        help="Chunk size used for delta action computation.",
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="LeRobotDataset repo id.",
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Local dataset root (passed to LeRobotDataset as root=).",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. If not set, use default HF_LEROBOT_HOME path.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional direct output path for stats.json (overrides output_dir layout).",
    )

    parser.add_argument(
        "--quantile_method",
        type=str,
        choices=["none", "histogram", "percentile"],
        default="none",
        help=(
            "Quantile computation for non-video features: "
            "none (default, backward compatible), "
            "histogram (global RunningQuantileStats, 5000 bins), "
            "percentile (global np.percentile, exact)."
        ),
    )

    parser.add_argument(
        "--quantile_zero_std_eps",
        type=float,
        default=1e-12,
        help="Dimensions with std below this get quantiles set to mean (percentile mode).",
    )

    parser.add_argument(
        "--preserve_moments_from",
        type=str,
        default=None,
        help=(
            "If set, keep min/max/mean/std/count from this existing stats.json "
            "and only write quantile fields from --quantile_method."
        ),
    )

    return parser.parse_args()


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None

    def update(self, batch: torch.Tensor) -> None:
        batch = batch.to(torch.float32)

        if batch.ndim == 1:
            batch = batch[:, None]

        if batch.ndim > 1:
            batch = batch.reshape(-1, batch.shape[-1])
        count = batch.shape[0]
        mean = batch.mean(dim=0)
        mean_sq = (batch**2).mean(dim=0)
        min_ = batch.min(dim=0).values
        max_ = batch.max(dim=0).values

        if self._count == 0:
            self._count = count
            self._mean = mean
            self._mean_of_squares = mean_sq
            self._min = min_
            self._max = max_
        else:
            total = self._count + count
            w_old = self._count / total
            w_new = count / total

            self._mean = w_old * self._mean + w_new * mean
            self._mean_of_squares = w_old * self._mean_of_squares + w_new * mean_sq
            self._min = torch.minimum(self._min, min_)
            self._max = torch.maximum(self._max, max_)
            self._count = total

    def get_statistics(self) -> dict:
        if self._count == 0:
            raise ValueError("No data has been added yet.")
        var = self._mean_of_squares - self._mean**2
        std = torch.sqrt(torch.clamp(var, min=0.0))
        return {
            "min": self._min.tolist(),
            "max": self._max.tolist(),
            "mean": self._mean.tolist(),
            "std": std.tolist(),
            "count": [self._count],
        }


class SampleAccumulator:
    """Collect per-key samples for global quantile computation."""

    def __init__(self, keys: list[str]):
        self._buffers: dict[str, list[np.ndarray]] = {key: [] for key in keys}

    def add(self, key: str, batch: torch.Tensor) -> None:
        arr = batch.to(torch.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        else:
            arr = arr.reshape(-1, arr.shape[-1])
        self._buffers[key].append(arr.cpu().numpy())

    def get_array(self, key: str) -> np.ndarray:
        parts = self._buffers[key]
        if not parts:
            raise ValueError(f"No samples collected for key '{key}'.")
        return np.concatenate(parts, axis=0)


class HistogramQuantileAccumulator:
    """Global RunningQuantileStats per key (5000-bin histogram)."""

    def __init__(self, keys: list[str], repo_root: Path | None = None):
        RunningQuantileStats = _load_running_quantile_stats(repo_root)
        self._stats = {key: RunningQuantileStats(num_quantile_bins=5000) for key in keys}

    def add(self, key: str, batch: torch.Tensor) -> None:
        arr = batch.to(torch.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        else:
            arr = arr.reshape(-1, arr.shape[-1])
        self._stats[key].update(arr.cpu().numpy())

    def get_quantiles(self, key: str, vector_length: int) -> dict[str, list]:
        stats = self._stats[key].get_statistics()
        out = {}
        for qk in QUANTILE_KEYS:
            val = np.asarray(stats[qk], dtype=np.float64)
            if vector_length == 1:
                val = val.reshape(1)
            out[qk] = val.tolist()
        return out


def _load_running_quantile_stats(repo_root: Path | None):
    fake_utils = types.ModuleType("lerobot.datasets.utils")
    fake_utils.load_image_as_numpy = lambda *a, **k: None  # noqa: ARG005
    sys.modules.setdefault("lerobot", types.ModuleType("lerobot"))
    sys.modules.setdefault("lerobot.datasets", types.ModuleType("lerobot.datasets"))
    sys.modules["lerobot.datasets.utils"] = fake_utils

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "compute_stats",
        repo_root / "src" / "lerobot" / "datasets" / "compute_stats.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.RunningQuantileStats


def _as_vector(arr: list | np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def compute_percentile_quantiles(
    samples: np.ndarray,
    mean: list | np.ndarray,
    std: list | np.ndarray,
    eps: float,
) -> dict[str, list]:
    mean_v = _as_vector(mean)
    std_v = _as_vector(std)
    dim = samples.shape[1]
    if mean_v.size != dim or std_v.size != dim:
        raise ValueError(
            f"Shape mismatch: samples D={dim}, mean={mean_v.size}, std={std_v.size}"
        )

    out: dict[str, list] = {}
    for qk, pct in zip(QUANTILE_KEYS, QUANTILE_PERCENTILES, strict=True):
        qvals = np.empty(dim, dtype=np.float64)
        for d in range(dim):
            if std_v[d] < eps:
                qvals[d] = mean_v[d]
            else:
                qvals[d] = np.percentile(samples[:, d], pct, method="linear")
        out[qk] = qvals.tolist() if dim > 1 else [float(qvals[0])]
    return out


def _feature_dim(feature_stats: dict) -> int:
    return len(_as_vector(feature_stats["mean"]))


def _merge_quantiles(output_dict: dict, quantiles_by_key: dict[str, dict[str, list]]) -> None:
    for key, qfeat in quantiles_by_key.items():
        for qk in QUANTILE_KEYS:
            output_dict[key][qk] = qfeat[qk]


def _apply_preserved_moments(output_dict: dict, preserved: dict) -> None:
    for key in output_dict:
        if key not in preserved:
            continue
        for field in ("min", "max", "mean", "std", "count"):
            if field in preserved[key]:
                output_dict[key][field] = preserved[key][field]


def _col_to_tensor(series: pd.Series) -> torch.Tensor:
    vals = series.values
    first = vals[0]
    if isinstance(first, (list, np.ndarray)):
        return torch.from_numpy(np.stack(vals).astype(np.float32))
    return torch.from_numpy(np.asarray(vals, dtype=np.float32).copy())


def _to_feature_tensor(ep_df: pd.DataFrame, key: str) -> torch.Tensor:
    val = _col_to_tensor(ep_df[key])
    return val if val.ndim > 1 else val.unsqueeze(-1)


def _load_info_and_keys(dataset_root: Path) -> tuple[dict, list[str], list[str]]:
    with open(dataset_root / "meta" / "info.json") as f:
        info = json.load(f)
    video_keys = [
        k for k, v in info["features"].items() if v["dtype"] in ("video", "image")
    ]
    keys = [k for k in info["features"] if k not in video_keys]
    return info, keys, video_keys


def _visual_stats_source_paths(dataset_root: Path) -> list[Path]:
    return [
        dataset_root / "meta" / "stats.json",
        dataset_root / "meta" / "stats" / "abs" / "stats.json",
    ]


def _load_visual_stats(dataset_root: Path, video_keys: list[str]) -> dict:
    for stats_path in _visual_stats_source_paths(dataset_root):
        if not stats_path.exists():
            continue
        source_stats = json.load(open(stats_path))
        visual = {k: source_stats[k] for k in video_keys if k in source_stats}
        if visual:
            print(f"Loaded visual stats from {stats_path}")
            return visual
    raise FileNotFoundError(
        "Missing visual stats. Expected one of: "
        + ", ".join(str(p) for p in _visual_stats_source_paths(dataset_root))
    )


def _resolve_dataset_root(
    cfg,
    dataset_root: Path | None = None,
    *,
    output_path: Path | None = None,
) -> Path | None:
    if dataset_root is not None:
        return dataset_root
    if cfg.dataset_root:
        return Path(cfg.dataset_root)
    if output_path is not None:
        parts = output_path.parts
        if len(parts) >= 4 and parts[-4:] == ("meta", "stats", "abs", "stats.json"):
            return Path(*parts[:-4])
        if len(parts) >= 3 and parts[-3:] == ("meta", "stats", "stats.json"):
            return Path(*parts[:-3])
    default_root = HF_LEROBOT_HOME / cfg.repo_id
    return default_root if default_root.exists() else None


def _resolve_output_path(cfg, action_mode: str) -> Path:
    if cfg.output_path:
        return Path(cfg.output_path)
    if cfg.output_dir:
        return Path(cfg.output_dir) / action_mode / cfg.repo_id / "stats.json"
    return HF_LEROBOT_HOME / "stats" / action_mode / cfg.repo_id / "stats.json"


def _maybe_sync_meta_stats(action_mode: str, dataset_root: Path | None, output_dict: dict) -> None:
    if action_mode != "abs" or dataset_root is None:
        return
    meta_stats_path = dataset_root / "meta" / "stats.json"
    meta_stats_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_dict, meta_stats_path)
    print(f"synced meta stats: {meta_stats_path}")


def _write_stats_output(
    cfg,
    action_mode: str,
    output_dict: dict,
    dataset_root: Path | None = None,
) -> Path:
    out_path = _resolve_output_path(cfg, action_mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_dict, out_path)
    resolved_root = _resolve_dataset_root(cfg, dataset_root, output_path=out_path)
    _maybe_sync_meta_stats(action_mode, resolved_root, output_dict)
    return out_path


def _iter_episode_slices(dataset_root: Path):
    ep_path = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes = pd.read_parquet(ep_path)
    parquet_files = sorted((dataset_root / "data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    for _, ep in episodes.iterrows():
        from_idx = int(ep["dataset_from_index"])
        to_idx = int(ep["dataset_to_index"])
        yield from_idx, to_idx, df.iloc[from_idx:to_idx]


def compute_norm_stats_from_parquet(cfg, dataset_root: Path, repo_root: Path):
    """Parquet-backed path (no LeRobotDataset / PyAV dependency)."""
    info, keys, video_keys = _load_info_and_keys(dataset_root)
    robot_type = info["robot_type"]
    schema = get_schema(robot_type)
    mask = schema.action_mask
    mapping = schema.feature_mapping

    stats = {key: RunningStats() for key in keys}
    sample_acc = (
        SampleAccumulator(keys) if cfg.quantile_method == "percentile" else None
    )
    hist_acc = (
        HistogramQuantileAccumulator(keys, repo_root=repo_root)
        if cfg.quantile_method == "histogram"
        else None
    )

    action_mode = cfg.action_mode
    chunk_size = cfg.chunk_size
    total_frame = 0
    total_episodes = 0

    for from_idx, to_idx, ep_df in tqdm.tqdm(
        list(_iter_episode_slices(dataset_root)), desc="Computing stats (parquet)"
    ):
        total_episodes += 1
        ep_len = to_idx - from_idx
        total_frame += ep_len
        if ep_len < chunk_size:
            continue

        for key in keys:
            if action_mode == "abs" or key not in mapping[ACTION]:
                val = _col_to_tensor(ep_df[key])
                stats[key].update(val)
                if sample_acc is not None:
                    sample_acc.add(key, val)
                if hist_acc is not None:
                    hist_acc.add(key, val)

        if action_mode == "delta":
            action = torch.cat([_to_feature_tensor(ep_df, k) for k in mapping[ACTION]], dim=-1)
            state = torch.cat([_to_feature_tensor(ep_df, k) for k in mapping[OBS_STATE]], dim=-1)
            truncated_state = state[0 : (ep_len - chunk_size + 1)]
            action_chunk = action.unfold(dimension=0, size=chunk_size, step=1).permute(0, 2, 1)
            delta_action = action_chunk - torch.where(mask, truncated_state, 0)[:, None]
            sid, eid = 0, 0
            for action_key in mapping[ACTION]:
                eid += info["features"][action_key]["shape"][0]
                chunk = delta_action[..., sid:eid]
                stats[action_key].update(chunk)
                if sample_acc is not None:
                    sample_acc.add(action_key, chunk)
                if hist_acc is not None:
                    hist_acc.add(action_key, chunk)
                sid = eid

    output_dict = {key: stats[key].get_statistics() for key in keys}

    if cfg.quantile_method == "percentile":
        assert sample_acc is not None
        quantiles_by_key = {
            key: compute_percentile_quantiles(
                sample_acc.get_array(key),
                output_dict[key]["mean"],
                output_dict[key]["std"],
                cfg.quantile_zero_std_eps,
            )
            for key in keys
        }
        _merge_quantiles(output_dict, quantiles_by_key)
    elif cfg.quantile_method == "histogram":
        assert hist_acc is not None
        quantiles_by_key = {
            key: hist_acc.get_quantiles(key, _feature_dim(output_dict[key])) for key in keys
        }
        _merge_quantiles(output_dict, quantiles_by_key)

    output_dict.update(_load_visual_stats(dataset_root, video_keys))

    if cfg.preserve_moments_from:
        with open(cfg.preserve_moments_from) as f:
            preserved = json.load(f)
        _apply_preserved_moments(output_dict, preserved)
        print(f"Preserved min/max/mean/std/count from {cfg.preserve_moments_from}")

    out_path = _write_stats_output(cfg, action_mode, output_dict, dataset_root)

    print(f"total_episodes: {total_episodes}")
    print(f"total_frame: {total_frame}")
    print(f"output: {out_path}")


def compute_norm_stats(cfg):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    repo_root = Path(__file__).resolve().parents[1]
    root = Path(cfg.dataset_root) if cfg.dataset_root else None

    print(f"---------- compute statistics for dataset: {cfg.repo_id} ----------")
    if root is not None:
        print(f"dataset root: {root}")
    print(f"quantile_method: {cfg.quantile_method}")

    if root is not None and root.exists():
        return compute_norm_stats_from_parquet(cfg, root, repo_root)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(cfg.repo_id, root=root, download_videos=False)

    from_ids = np.asarray(dataset.meta.episodes["dataset_from_index"])
    to_ids = np.asarray(dataset.meta.episodes["dataset_to_index"])
    total_episodes = dataset.num_episodes

    action_mode = cfg.action_mode
    chunk_size = cfg.chunk_size
    robot_type = dataset.meta.robot_type

    schema = get_schema(robot_type)
    mask = schema.action_mask
    mapping = schema.feature_mapping

    keys = list(dataset.meta.features.keys())
    [keys.remove(video_key) for video_key in dataset.meta.video_keys]
    [keys.remove(image_key) for image_key in dataset.meta.image_keys]
    stats = {key: RunningStats() for key in keys}

    sample_acc = (
        SampleAccumulator(keys)
        if cfg.quantile_method == "percentile"
        else None
    )
    hist_acc = (
        HistogramQuantileAccumulator(keys, repo_root=repo_root)
        if cfg.quantile_method == "histogram"
        else None
    )

    total_frame = 0

    for from_idx, to_idx in tqdm.tqdm(
        zip(from_ids, to_ids, strict=True), total=total_episodes, desc="Computing stats"
    ):
        ep_len = to_idx - from_idx
        total_frame += ep_len
        if ep_len < chunk_size:
            continue
        curr_episode = dataset.hf_dataset.select(np.arange(from_idx, to_idx))
        for key in keys:
            if action_mode == "abs" or key not in mapping[ACTION]:
                val = torch.stack(curr_episode[key][:])
                stats[key].update(val)
                if sample_acc is not None:
                    sample_acc.add(key, val)
                if hist_acc is not None:
                    hist_acc.add(key, val)
        if action_mode == "delta":
            action = [torch.stack(curr_episode[key][:]) for key in mapping[ACTION]]
            action = [a if a.ndim > 1 else a[:, None] for a in action]
            action = torch.cat(action, dim=-1)
            state = [torch.stack(curr_episode[key][:]) for key in mapping[OBS_STATE]]
            state = [s if s.ndim > 1 else s[:, None] for s in state]
            state = torch.cat(state, dim=-1)
            truncated_state = state[0 : (ep_len - chunk_size + 1)]
            action_chunk = action.unfold(dimension=0, size=chunk_size, step=1).permute(0, 2, 1)
            delta_action = action_chunk - torch.where(mask, truncated_state, 0)[:, None]
            sid, eid = 0, 0
            for action_key in mapping[ACTION]:
                eid += dataset.meta.features[action_key]["shape"][0]
                chunk = delta_action[..., sid:eid]
                stats[action_key].update(chunk)
                if sample_acc is not None:
                    sample_acc.add(action_key, chunk)
                if hist_acc is not None:
                    hist_acc.add(action_key, chunk)
                sid = eid

    output_dict = {key: stats[key].get_statistics() for key in keys}

    if cfg.quantile_method == "percentile":
        assert sample_acc is not None
        quantiles_by_key = {
            key: compute_percentile_quantiles(
                sample_acc.get_array(key),
                output_dict[key]["mean"],
                output_dict[key]["std"],
                cfg.quantile_zero_std_eps,
            )
            for key in keys
        }
        _merge_quantiles(output_dict, quantiles_by_key)
    elif cfg.quantile_method == "histogram":
        assert hist_acc is not None
        quantiles_by_key = {
            key: hist_acc.get_quantiles(key, _feature_dim(output_dict[key])) for key in keys
        }
        _merge_quantiles(output_dict, quantiles_by_key)

    for key in dataset.meta.video_keys + dataset.meta.image_keys:
        dataset.meta.stats[key]
        visual_stats = dataset.meta.stats[key]
        for stat_key in visual_stats:
            if isinstance(visual_stats[stat_key], np.ndarray):
                visual_stats[stat_key] = visual_stats[stat_key].tolist()
            elif isinstance(visual_stats[stat_key], torch.Tensor):
                visual_stats[stat_key] = visual_stats[stat_key].cpu().numpy().tolist()
        output_dict[key] = visual_stats

    if cfg.preserve_moments_from:
        with open(cfg.preserve_moments_from) as f:
            preserved = json.load(f)
        _apply_preserved_moments(output_dict, preserved)
        print(f"Preserved min/max/mean/std/count from {cfg.preserve_moments_from}")

    dataset_root = _resolve_dataset_root(cfg, root or getattr(dataset, "root", None))
    out_path = _write_stats_output(cfg, action_mode, output_dict, dataset_root)

    print(f"total_frame: {total_frame}")
    print(f"output: {out_path}")


if __name__ == "__main__":
    compute_norm_stats(parse_args())
