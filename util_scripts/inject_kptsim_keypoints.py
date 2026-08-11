"""Inject kptsim 3D keypoint GT into a LeRobot dataset copy (方案 A: voxel coords as-is).

Reads pre-computed SAPIEN FK keypoints from GeoPredict kptsim extraction
(`episode_NNNNNN/keypoints.npy`) and writes them as the `observation.keypoint_3d`
column into a copied LeRobot dataset.  Also embeds a key-remapped norm_stat.json
so the dataset folder is self-contained.

Design doc: b/d/itrnVLA15_GeoP_3dtrj_3cn4_wrmup.md §12

Usage:
    python util_scripts/inject_kptsim_keypoints.py \\
        --source /path/to/stack_bowls_three \\
        --kptsim_dir /path/to/stack_bowls_three_kptsim \\
        --dest /path/to/stack_bowls_three_kptsim_lrb \\
        --norm_stats_path /path/to/robotwin_norm_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KEYPOINT_NAMES: list[str] = [
    "fl_link1", "fl_link2", "fl_link3", "fl_link4", "fl_link5", "fl_link6", "fl_eef_tcp",
    "fr_link1", "fr_link2", "fr_link3", "fr_link4", "fr_link5", "fr_link6", "fr_eef_tcp",
]
NUM_KEYPOINTS = len(KEYPOINT_NAMES)  # 14
KEYPOINT_FEATURE_NAMES: list[str] = [
    f"{name}_{axis}" for name in KEYPOINT_NAMES for axis in ("x", "y", "z")
]

NORM_STATS_KEY_REMAP = {
    "state": "observation.state",
    "actions": "action",
}


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _copy_dataset(source: Path, dest: Path, force: bool) -> None:
    if dest.exists():
        if force:
            logger.info("Removing existing destination %s (--force)", dest)
            shutil.rmtree(dest)
        else:
            raise FileExistsError(
                f"Destination {dest} already exists. Pass --force to overwrite, or pick a new --dest."
            )
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Copying dataset %s -> %s (rsync -a) ...", source, dest)
    subprocess.run(["rsync", "-a", f"{source}/", f"{dest}/"], check=True)


def _load_kptsim_meta(kptsim_dir: Path) -> dict:
    meta_path = kptsim_dir / "keypoints_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = _load_json(meta_path)
    assert meta["K"] == NUM_KEYPOINTS, f"Expected K={NUM_KEYPOINTS}, got {meta['K']}"
    assert meta["keypoint_names"] == KEYPOINT_NAMES, (
        f"Keypoint name mismatch: expected {KEYPOINT_NAMES}, got {meta['keypoint_names']}"
    )
    return meta


def _inject_keypoints_into_parquets(dest: Path, kptsim_dir: Path) -> dict:
    parquet_files = sorted((dest / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dest / 'data'}")

    kptsim_episodes = sorted(kptsim_dir.glob("episode_*/keypoints.npy"))
    if len(parquet_files) != len(kptsim_episodes):
        raise ValueError(
            f"Episode count mismatch: {len(parquet_files)} parquets vs {len(kptsim_episodes)} kptsim episodes"
        )

    all_kpts = []
    total_rows = 0
    for pq_path in parquet_files:
        ep_idx = int(pq_path.stem.split("_")[-1])
        kpt_path = kptsim_dir / f"episode_{ep_idx:06d}" / "keypoints.npy"
        if not kpt_path.exists():
            raise FileNotFoundError(f"Missing {kpt_path}")

        kpts = np.load(kpt_path)  # [T, 42]
        assert kpts.ndim == 2 and kpts.shape[1] == NUM_KEYPOINTS * 3, (
            f"Episode {ep_idx}: expected shape [T, {NUM_KEYPOINTS * 3}], got {kpts.shape}"
        )

        df = pd.read_parquet(pq_path)
        if len(df) != kpts.shape[0]:
            raise ValueError(
                f"Episode {ep_idx}: parquet rows={len(df)} vs kptsim rows={kpts.shape[0]}"
            )

        df["observation.keypoint_3d"] = [row for row in kpts.astype(np.float32)]
        df.to_parquet(pq_path)
        logger.info("Wrote observation.keypoint_3d for %d rows -> %s", len(df), pq_path)
        all_kpts.append(kpts.reshape(-1, NUM_KEYPOINTS, 3))
        total_rows += len(df)

    all_kpts_cat = np.concatenate(all_kpts, axis=0)  # [total, 14, 3]
    stats = {
        "total_rows": total_rows,
        "total_episodes": len(parquet_files),
        "min_xyz": all_kpts_cat.reshape(-1, 3).min(axis=0).tolist(),
        "max_xyz": all_kpts_cat.reshape(-1, 3).max(axis=0).tolist(),
        "mean_xyz": all_kpts_cat.reshape(-1, 3).mean(axis=0).tolist(),
    }
    return stats


def _update_info_json(dest: Path, kptsim_meta: dict, coord_mode: str) -> None:
    info_path = dest / "meta" / "info.json"
    info = _load_json(info_path)
    info["features"]["observation.keypoint_3d"] = {
        "dtype": "float32",
        "shape": [NUM_KEYPOINTS * 3],
        "names": KEYPOINT_FEATURE_NAMES,
        "fps": info.get("fps", 15),
    }
    info["keypoint_coord_mode"] = coord_mode
    info["keypoint_coord_offset"] = kptsim_meta["coord_offset"]
    _write_json(info, info_path)
    logger.info("Updated %s with observation.keypoint_3d feature declaration.", info_path)


def _create_self_contained_stats(norm_stats_path: Path, dest: Path) -> None:
    raw = _load_json(norm_stats_path)
    remapped = {}
    for src_key, dst_key in NORM_STATS_KEY_REMAP.items():
        if src_key not in raw:
            raise KeyError(f"Key '{src_key}' not found in {norm_stats_path}")
        remapped[dst_key] = raw[src_key]

    _write_json(remapped, dest / "norm_stat.json")
    logger.info("Wrote key-remapped norm_stat.json -> %s", dest / "norm_stat.json")

    _write_json(remapped, dest / "meta" / "stats.json")
    logger.info("Wrote meta/stats.json (same content) -> %s", dest / "meta" / "stats.json")


def _copy_provenance(kptsim_dir: Path, dest: Path) -> None:
    src = kptsim_dir / "keypoints_meta.json"
    dst = dest / "meta" / "keypoints_meta.json"
    shutil.copy2(src, dst)
    logger.info("Copied keypoints_meta.json -> %s", dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", type=str, required=True,
        help="Read-only source LeRobot dataset directory.",
    )
    parser.add_argument(
        "--kptsim_dir", type=str, required=True,
        help="kptsim directory with episode_*/keypoints.npy and keypoints_meta.json.",
    )
    parser.add_argument(
        "--dest", type=str, required=True,
        help="Output directory (copy of source + keypoint column + norm stats).",
    )
    parser.add_argument(
        "--norm_stats_path", type=str,
        default="/home/luogang/SRC/Robot/GeoPredict/ckpts/robotwin_norm_stats.json",
        help="Path to GeoPredict norm_stats JSON (state/actions keys).",
    )
    parser.add_argument(
        "--coord_mode", choices=["voxel", "footprint"], default="voxel",
        help="Coordinate mode: voxel (方案 A, default) or footprint (方案 B).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing --dest.")
    parser.add_argument(
        "--skip-copy", action="store_true",
        help="Skip rsync; assume --dest already contains a copy.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    kptsim_dir = Path(args.kptsim_dir)
    norm_stats_path = Path(args.norm_stats_path)

    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    if not kptsim_dir.exists():
        raise FileNotFoundError(f"kptsim directory not found: {kptsim_dir}")
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"Norm stats file not found: {norm_stats_path}")

    # Step 1: Copy dataset
    if not args.skip_copy:
        _copy_dataset(source, dest, args.force)
    else:
        if not dest.exists():
            raise FileNotFoundError(f"--skip-copy given but {dest} does not exist.")
        logger.info("--skip-copy: reusing existing copy at %s", dest)

    # Step 2: Load & validate kptsim metadata
    kptsim_meta = _load_kptsim_meta(kptsim_dir)

    # Step 3: Inject keypoints into parquet files
    stats = _inject_keypoints_into_parquets(dest, kptsim_dir)

    # Step 4: Update meta/info.json
    _update_info_json(dest, kptsim_meta, args.coord_mode)

    # Step 5: Create self-contained norm stats
    _create_self_contained_stats(norm_stats_path, dest)

    # Step 6: Copy keypoints_meta.json for provenance
    _copy_provenance(kptsim_dir, dest)

    # Summary
    logger.info("Done. %d frames across %d episodes injected.", stats["total_rows"], stats["total_episodes"])
    logger.info("  min XYZ: %s", stats["min_xyz"])
    logger.info("  max XYZ: %s", stats["max_xyz"])
    logger.info("  mean XYZ: %s", stats["mean_xyz"])
    logger.info("Dataset is self-contained at: %s", dest)


if __name__ == "__main__":
    main()
