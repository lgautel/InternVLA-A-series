#!/usr/bin/env python3
"""Quick FK smoke test on one parquet file.

Verifies that the FK engine produces sane keypoints from real joint data.

Usage:
    python b/s/libplus/smoke_test.py \
        --source /path/to/libero_merged_lerobot_v3 \
        --xml evaluation/panda_robosuite_lift.xml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_libero_kpt_v2 import (
    JOINT_COLUMN, STATE_COLUMN, NUM_KEYPOINTS, KEYPOINT_DIM,
    LiftMujocoFK, build_qpos,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=10)
    args = parser.parse_args()

    parquets = sorted((args.source / "data").rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquets under {args.source}/data")

    pq = parquets[0]
    df = pd.read_parquet(pq, columns=[JOINT_COLUMN, STATE_COLUMN])
    print(f"File: {pq.name}, frames: {len(df)}")

    qpos = np.stack([
        build_qpos(j, s)
        for j, s in zip(df[JOINT_COLUMN].values[:args.max_frames],
                        df[STATE_COLUMN].values[:args.max_frames], strict=True)
    ])

    fk = LiftMujocoFK(args.xml)
    kpts = fk.compute_batch(qpos)

    assert kpts.shape == (len(qpos), NUM_KEYPOINTS, KEYPOINT_DIM), \
        f"Bad shape: {kpts.shape}"
    assert not np.any(np.isnan(kpts)), "NaN in keypoints"

    q = kpts[:, :, 3:7].reshape(-1, 4)
    q_norms = np.linalg.norm(q, axis=1)
    assert np.abs(q_norms - 1.0).max() < 0.01, \
        f"Quaternion not unit: max err = {np.abs(q_norms - 1.0).max()}"

    qw = q[:, 3]
    assert (qw >= -1e-6).all(), f"qw < 0: min qw = {qw.min()}"

    print(f"SMOKE OK")
    print(f"  keypoints shape: {kpts.shape}")
    print(f"  link1 pos: {kpts[0, 0, :3]}")
    print(f"  eef pos:   {kpts[0, -1, :3]}")
    print(f"  eef quat:  {kpts[0, -1, 3:]}")
    print(f"  quat norm range: [{q_norms.min():.6f}, {q_norms.max():.6f}]")


if __name__ == "__main__":
    main()
