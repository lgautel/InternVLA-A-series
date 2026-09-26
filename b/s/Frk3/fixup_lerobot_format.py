"""Fix up dataset format to match LeRobot v3 expectations.

Converts from the fast converter's format to proper LeRobot v3:
  1. Renames data/chunk-000/episode_NNNNNN.parquet → file-NNN.parquet
  2. Renames videos/.../episode_NNNNNN.mp4 → file-NNN.mp4
  3. Converts meta/episodes.jsonl → meta/episodes/chunk-000/file-000.parquet
  4. Updates info.json path templates

Usage:
    python b/s/Frk3/fixup_lerobot_format.py --dataset /path/to/dataset
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def fixup(dataset_path: str | Path):
    root = Path(dataset_path)

    data_dir = root / "data" / "chunk-000"
    old_parquets = sorted(data_dir.glob("episode_*.parquet"))
    n_episodes = len(old_parquets)
    logger.info("Found %d episode parquet files to rename", n_episodes)

    for i, old_pq in enumerate(old_parquets):
        new_name = f"file-{i:03d}.parquet"
        new_path = old_pq.parent / new_name
        if old_pq != new_path:
            old_pq.rename(new_path)
            logger.info("  %s → %s", old_pq.name, new_name)

    video_base = root / "videos"
    for vid_key_dir in sorted(video_base.iterdir()):
        if not vid_key_dir.is_dir():
            continue
        chunk_dir = vid_key_dir / "chunk-000"
        if not chunk_dir.exists():
            continue
        old_vids = sorted(chunk_dir.glob("episode_*.mp4"))
        for i, old_vid in enumerate(old_vids):
            new_name = f"file-{i:03d}.mp4"
            new_path = old_vid.parent / new_name
            if old_vid != new_path:
                old_vid.rename(new_path)
                logger.info("  %s → %s", old_vid.name, new_name)

    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["data_path"] = DEFAULT_DATA_PATH
    info["video_path"] = DEFAULT_VIDEO_PATH
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info("Updated info.json path templates")

    jsonl_path = root / "meta" / "episodes.jsonl"
    if jsonl_path.exists():
        df = pd.read_json(jsonl_path, lines=True)
    else:
        parquets = sorted((root / "data" / "chunk-000").glob("file-*.parquet"))
        records = []
        for i, pq_path in enumerate(parquets):
            pq_df = pd.read_parquet(pq_path)
            records.append({"episode_index": i, "length": len(pq_df)})
        df = pd.DataFrame(records)

    vid_keys = []
    for feat_name, feat_info in info.get("features", {}).items():
        if feat_info.get("dtype") == "video":
            vid_keys.append(feat_name)

    records = []
    for i in range(len(df)):
        row = df.iloc[i]
        ep_record = {
            "episode_index": int(row["episode_index"]),
            "length": int(row["length"]),
            "data/chunk_index": 0,
            "data/file_index": i,
        }
        if "tasks" in row:
            tasks = row["tasks"]
            if isinstance(tasks, list):
                ep_record["tasks"] = tasks
            else:
                ep_record["tasks"] = [str(tasks)]
        else:
            ep_record["tasks"] = ["put cube into box"]

        ep_record["task_index"] = 0

        ep_length = int(row["length"])
        for vid_key in vid_keys:
            ep_record[f"videos/{vid_key}/chunk_index"] = 0
            ep_record[f"videos/{vid_key}/file_index"] = i
            ep_record[f"videos/{vid_key}/from_timestamp"] = 0.0
            ep_record[f"videos/{vid_key}/to_timestamp"] = ep_length / info.get("fps", 30)

        records.append(ep_record)

    ep_df = pd.DataFrame(records)
    ep_parquet_dir = root / "meta" / "episodes" / "chunk-000"
    ep_parquet_dir.mkdir(parents=True, exist_ok=True)
    ep_parquet_path = ep_parquet_dir / "file-000.parquet"
    ep_df.to_parquet(ep_parquet_path, index=False)
    logger.info("Wrote episodes parquet: %s (%d episodes)", ep_parquet_path, len(ep_df))

    if jsonl_path.exists():
        jsonl_path.rename(jsonl_path.with_suffix(".jsonl.bak"))
        logger.info("Backed up %s", jsonl_path)

    tasks_jsonl = root / "meta" / "tasks.jsonl"
    if tasks_jsonl.exists():
        pass

    logger.info("=== Fixup complete: %s ===", root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    fixup(args.dataset)
