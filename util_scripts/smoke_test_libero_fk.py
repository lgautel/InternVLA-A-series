#!/usr/bin/env python3
"""Quick FK sanity check on one LIBERO parquet file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from generate_libero_keypoints import (  # noqa: E402
    JOINT_COLUMN,
    STATE_COLUMN,
    LiberoMujocoFK,
    _build_qpos,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=5)
    args = parser.parse_args()

    parquets = sorted((args.source / "data").rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet under {args.source}/data")

    pq = parquets[0]
    df = pd.read_parquet(pq, columns=[JOINT_COLUMN, STATE_COLUMN])
    qpos = np.stack(
        [_build_qpos(j, s) for j, s in zip(df[JOINT_COLUMN].values, df[STATE_COLUMN].values, strict=True)]
    )
    fk = LiberoMujocoFK(args.xml)
    kpts = fk.compute_batch(qpos[: min(args.max_frames, len(qpos))])

    print(f"SMOKE OK: {pq.name}")
    print(f"  keypoints shape: {kpts.shape}")
    print(f"  sample link1 pos: {kpts[0, 0]}")
    print(f"  sample eef pos:   {kpts[0, -1]}")


if __name__ == "__main__":
    main()
