#!/usr/bin/env python3
"""Split a merged LIBERO LeRobot v3 dataset into four training suites.

Episode order matches `port_libero.py` / OpenVLA RLDS conversion order:
  libero_spatial -> libero_object -> libero_goal -> libero_10

Usage:
    export HF_LEROBOT_HOME=/path/to/lerobot
    python util_scripts/split_libero_merged.py \\
        --source /path/to/libero_merged_kpt \\
        --output-dir $HF_LEROBOT_HOME

Creates:
    $HF_LEROBOT_HOME/libero_spatial
    $HF_LEROBOT_HOME/libero_object
    $HF_LEROBOT_HOME/libero_goal
    $HF_LEROBOT_HOME/libero_10
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from lerobot.datasets.dataset_tools import split_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Counts from openvla/modified_libero_rlds (must match conversion order).
SUITE_COUNTS = {
    "libero_spatial": 432,
    "libero_object": 454,
    "libero_goal": 428,
    "libero_10": 379,
}


def build_suite_episode_ranges(total_episodes: int) -> dict[str, list[int]]:
    expected = sum(SUITE_COUNTS.values())
    if total_episodes != expected:
        raise ValueError(
            f"Dataset has {total_episodes} episodes, expected {expected} "
            f"for merged LIBERO ({', '.join(f'{k}={v}' for k, v in SUITE_COUNTS.items())})."
        )

    splits: dict[str, list[int]] = {}
    start = 0
    for suite, count in SUITE_COUNTS.items():
        end = start + count
        splits[suite] = list(range(start, end))
        start = end
    return splits


def patch_robot_type(dataset_root: Path, robot_type: str) -> None:
    info_path = dataset_root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["robot_type"] = robot_type
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info("Patched robot_type=%s in %s", robot_type, info_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Merged LIBERO LeRobot v3 dataset root.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/libero_merged",
        help="Repo id used when loading the merged dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where libero_{spatial,object,goal,10} will be created.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing suite directories under --output-dir before splitting.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for suite_name in SUITE_COUNTS:
        suite_dir = output_dir / suite_name
        if suite_dir.exists():
            if args.force:
                logger.info("Removing existing split %s (--force)", suite_dir)
                shutil.rmtree(suite_dir)
            else:
                raise FileExistsError(
                    f"Split output already exists: {suite_dir}. "
                    "Pass --force to overwrite, or remove it manually."
                )

    dataset = LeRobotDataset(args.repo_id, root=source)
    splits = build_suite_episode_ranges(dataset.meta.total_episodes)
    logger.info(
        "Splitting %d episodes from %s into suites: %s",
        dataset.meta.total_episodes,
        source,
        {k: len(v) for k, v in splits.items()},
    )

    result = split_dataset(dataset, splits, output_dir=output_dir)

    for suite_name, split_ds in result.items():
        patch_robot_type(split_ds.root, suite_name)
        logger.info(
            "Created %s: %d episodes, %d frames at %s",
            suite_name,
            split_ds.meta.total_episodes,
            split_ds.meta.total_frames,
            split_ds.root,
        )

    logger.info("Done. Set DATASET_REPO_ID='libero_spatial libero_object libero_goal libero_10' for training.")


if __name__ == "__main__":
    main()
