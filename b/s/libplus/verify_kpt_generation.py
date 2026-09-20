#!/usr/bin/env python3
"""Verify generated keypoints against observation.state EEF position.

The EEF (gripper0_eef / grip_site) world position computed by FK should match
observation.state[0:3] up to the constant base offset between Lift and the
actual arena:

    FK_eef_pos = state[0:3] + (Lift_base - Arena_base)

This script reads the generated keypoints from the destination dataset and
cross-validates against the state column. It also checks:
  - Quaternion unit norm (|q| = 1 within tolerance)
  - Position range within [-1, 1] after R_pad normalization
  - No NaN values
  - Consistent keypoint body count and dimension

Usage:
    python b/s/libplus/verify_kpt_generation.py \
        --dataset /B/Dta/libero_merged_kpt_v2 \
        --max-files 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NUM_KEYPOINTS = 8
KEYPOINT_DIM = 7
EEF_IDX = 7  # last keypoint is EEF

LIFT_BASE = np.array([-0.56, 0.0, 0.912])

ARENA_BASES = {
    "table": np.array([-0.66, 0.0, 0.912]),
    "kitchen_table": np.array([-0.66, 0.0, 0.912]),
    "study_table": np.array([-0.75, 0.0, 0.912]),
    "living_room_table": np.array([-0.51, 0.0, 0.42]),
    "floor": np.array([-0.60, 0.0, 0.0]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=0,
                        help="0 = all files")
    parser.add_argument("--pos-tol", type=float, default=2e-3,
                        help="Position error tolerance (m)")
    parser.add_argument("--quat-tol", type=float, default=1e-3,
                        help="Quaternion norm error tolerance")
    args = parser.parse_args()

    meta_path = args.dataset / "meta" / "keypoints_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    r_pad = meta["bbox_radius"]

    print(f"R_pad = {r_pad}")
    print(f"Bodies: {meta['keypoint_bodies']}")

    parquets = sorted((args.dataset / "data").rglob("*.parquet"))
    if args.max_files > 0:
        parquets = parquets[:args.max_files]
    print(f"Checking {len(parquets)} parquet files...")

    checks = {"nan": 0, "quat_norm_err": 0.0, "pos_oob": 0,
              "total_frames": 0, "eef_pos_err_max": 0.0}

    for pq in parquets:
        df = pd.read_parquet(pq)
        if "observation.keypoint_3d" not in df.columns:
            print(f"  SKIP {pq.name}: no keypoint_3d column")
            continue

        kpts_flat = np.stack(df["observation.keypoint_3d"].values)
        kpts = kpts_flat.reshape(-1, NUM_KEYPOINTS, KEYPOINT_DIM)
        checks["total_frames"] += len(kpts)

        nan_count = int(np.isnan(kpts).sum())
        checks["nan"] += nan_count

        q = kpts[:, :, 3:7].reshape(-1, 4)
        q_norms = np.linalg.norm(q, axis=1)
        q_err = float(np.abs(q_norms - 1.0).max())
        checks["quat_norm_err"] = max(checks["quat_norm_err"], q_err)

        pos = kpts[:, :, :3]
        if (np.abs(pos) > 1.01).any():
            checks["pos_oob"] += 1

        if "observation.state" in df.columns:
            states = np.stack(df["observation.state"].values)
            eef_pos_kpt = kpts[:, EEF_IDX, :3] * r_pad  # de-normalize
            eef_pos_state = states[:, :3]

            diff = eef_pos_kpt - eef_pos_state
            diff_std = diff.std(axis=0)
            checks["eef_pos_err_max"] = max(checks["eef_pos_err_max"],
                                            float(diff_std.max()))

            if float(diff_std.max()) >= args.pos_tol:
                diff_mean = diff.mean(axis=0)
                print(f"  WARN {pq.name}: EEF diff not constant "
                      f"(std={diff_std}, mean={diff_mean})")

    print(f"\n=== Verification Results ===")
    print(f"  Total frames: {checks['total_frames']}")
    print(f"  NaN values: {checks['nan']}")
    print(f"  Quat norm error max: {checks['quat_norm_err']:.6f}")
    print(f"  OOB files: {checks['pos_oob']}")
    print(f"  EEF pos consistency (std max): {checks['eef_pos_err_max']:.6f}")

    ok = (
        checks["nan"] == 0
        and checks["quat_norm_err"] < args.quat_tol
        and checks["pos_oob"] == 0
        and checks["eef_pos_err_max"] < args.pos_tol
    )
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
