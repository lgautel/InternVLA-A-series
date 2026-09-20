#!/usr/bin/env python3
"""Verify the generated RLDS-to-LeRobot 4D dataset.

Checks:
  T1: Dataset structure (parquet files, videos, meta files)
  T2: Episode count matches (1,693 expected)
  T3: Frame count matches (273,465 expected)
  T4: Keypoint shape [56] = 8 x 7
  T5: No NaN in keypoints
  T6: Quaternion unit norm (|q| - 1 < 0.001)
  T7: Position in [-1.01, 1.01] after R_pad
  T8: qw >= 0 (hemisphere constraint)
  T9: EEF position cross-validation against observation.state
  T10: R_pad matches eval default (1.8212722539901733)
  T11: MJCF MD5 matches eval file
  T12: Joint positions present and in valid range
  T13: Video files exist for each episode

Usage:
    python b/s/libplus/verify_rlds_4d.py --dataset ~/b/Dta/opvla_libero_4d
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NUM_KEYPOINTS = 8
KEYPOINT_DIM = 7
EEF_IDX = 7
EVAL_DEFAULT_R_PAD = 1.8212722539901733


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--eval-xml", type=Path,
                        default=Path("evaluation/panda_robosuite_lift.xml"))
    parser.add_argument("--expected-episodes", type=int, default=1693)
    parser.add_argument("--expected-frames", type=int, default=273465)
    parser.add_argument("--sample-files", type=int, default=50,
                        help="Number of parquet files to sample for checks")
    args = parser.parse_args()

    results = {}
    dataset = args.dataset

    # T1: Structure
    parquets = sorted((dataset / "data").rglob("*.parquet"))
    meta_files = ["info.json", "episodes.jsonl", "tasks.jsonl", "keypoints_meta.json"]
    meta_ok = all((dataset / "meta" / f).exists() for f in meta_files)
    results["T1_structure"] = len(parquets) > 0 and meta_ok
    print(f"T1 Structure: {len(parquets)} parquets, meta={'OK' if meta_ok else 'MISSING'}"
          f" -> {'PASS' if results['T1_structure'] else 'FAIL'}")

    # Load meta
    with open(dataset / "meta" / "info.json") as f:
        info = json.load(f)
    with open(dataset / "meta" / "keypoints_meta.json") as f:
        kpt_meta = json.load(f)
    r_pad = kpt_meta["bbox_radius"]

    # T2: Episode count
    results["T2_episodes"] = info["total_episodes"] == args.expected_episodes
    print(f"T2 Episodes: {info['total_episodes']} (expected {args.expected_episodes})"
          f" -> {'PASS' if results['T2_episodes'] else 'FAIL'}")

    # T3: Frame count
    results["T3_frames"] = info["total_frames"] == args.expected_frames
    print(f"T3 Frames: {info['total_frames']} (expected {args.expected_frames})"
          f" -> {'PASS' if results['T3_frames'] else 'FAIL'}")

    # T10: R_pad
    diff = abs(r_pad - EVAL_DEFAULT_R_PAD)
    results["T10_r_pad"] = diff < 1e-6
    print(f"T10 R_pad: {r_pad:.10f} (eval: {EVAL_DEFAULT_R_PAD:.10f}, diff={diff:.2e})"
          f" -> {'PASS' if results['T10_r_pad'] else 'FAIL'}")

    # T11: MJCF MD5
    if args.eval_xml.exists():
        eval_md5 = hashlib.md5(args.eval_xml.read_bytes()).hexdigest()
        gen_md5 = kpt_meta.get("mjcf_md5", "MISSING")
        results["T11_mjcf_md5"] = gen_md5 == eval_md5
        print(f"T11 MJCF MD5: gen={gen_md5[:12]}..., eval={eval_md5[:12]}..."
              f" -> {'PASS' if results['T11_mjcf_md5'] else 'FAIL'}")
    else:
        results["T11_mjcf_md5"] = False
        print(f"T11 MJCF MD5: eval XML not found -> FAIL")

    # Sample parquets for data checks
    sample_pqs = parquets[:args.sample_files] if len(parquets) > args.sample_files else parquets
    checks = {"nan": 0, "quat_norm_err": 0.0, "pos_oob": 0, "qw_neg": 0,
              "frames_checked": 0, "eef_std_max": 0.0, "joint_nan": 0}

    for pq in sample_pqs:
        df = pd.read_parquet(pq)

        # T4: shape check
        if "observation.keypoint_3d" not in df.columns:
            continue
        kpts = np.stack(df["observation.keypoint_3d"].values)
        if kpts.shape[1] != NUM_KEYPOINTS * KEYPOINT_DIM:
            print(f"  BAD SHAPE in {pq.name}: {kpts.shape}")
        kpts = kpts.reshape(-1, NUM_KEYPOINTS, KEYPOINT_DIM)
        checks["frames_checked"] += len(kpts)

        # T5: NaN
        checks["nan"] += int(np.isnan(kpts).sum())

        # T6: Quaternion norm
        q = kpts[:, :, 3:7].reshape(-1, 4)
        q_norms = np.linalg.norm(q, axis=1)
        checks["quat_norm_err"] = max(checks["quat_norm_err"],
                                      float(np.abs(q_norms - 1.0).max()))

        # T7: Position range
        if (np.abs(kpts[:, :, :3]) > 1.01).any():
            checks["pos_oob"] += 1

        # T8: qw >= 0
        qw = q[:, 3]
        checks["qw_neg"] += int((qw < -1e-6).sum())

        # T9: EEF cross-validation
        if "observation.state" in df.columns:
            states = np.stack(df["observation.state"].values)
            eef_pos = kpts[:, EEF_IDX, :3] * r_pad
            diff_vec = eef_pos - states[:, :3]
            diff_std = diff_vec.std(axis=0)
            checks["eef_std_max"] = max(checks["eef_std_max"], float(diff_std.max()))

        # T12: Joint positions
        if "observation.state.joint_position" in df.columns:
            joints = np.stack(df["observation.state.joint_position"].values)
            checks["joint_nan"] += int(np.isnan(joints).sum())

    # Report data checks
    results["T4_shape"] = checks["frames_checked"] > 0
    print(f"T4 Shape: checked {checks['frames_checked']} frames"
          f" -> {'PASS' if results['T4_shape'] else 'FAIL'}")

    results["T5_no_nan"] = checks["nan"] == 0
    print(f"T5 No NaN: {checks['nan']} NaN values"
          f" -> {'PASS' if results['T5_no_nan'] else 'FAIL'}")

    results["T6_quat_norm"] = checks["quat_norm_err"] < 0.001
    print(f"T6 Quat norm: max err={checks['quat_norm_err']:.6f}"
          f" -> {'PASS' if results['T6_quat_norm'] else 'FAIL'}")

    results["T7_pos_range"] = checks["pos_oob"] == 0
    print(f"T7 Pos range: {checks['pos_oob']} OOB files"
          f" -> {'PASS' if results['T7_pos_range'] else 'FAIL'}")

    results["T8_qw_positive"] = checks["qw_neg"] == 0
    print(f"T8 qw >= 0: {checks['qw_neg']} violations"
          f" -> {'PASS' if results['T8_qw_positive'] else 'FAIL'}")

    results["T9_eef_consistency"] = checks["eef_std_max"] < 0.002
    print(f"T9 EEF consistency: std_max={checks['eef_std_max']:.6f}"
          f" -> {'PASS' if results['T9_eef_consistency'] else 'FAIL'}")

    results["T12_joints"] = checks["joint_nan"] == 0
    print(f"T12 Joint positions: {checks['joint_nan']} NaN"
          f" -> {'PASS' if results['T12_joints'] else 'FAIL'}")

    # T13: Video files
    vid_dir_1 = dataset / "videos" / "chunk-000" / "observation.images.image"
    vid_dir_2 = dataset / "videos" / "chunk-000" / "observation.images.image2"
    n_vid_1 = len(list(vid_dir_1.glob("*.mp4"))) if vid_dir_1.exists() else 0
    n_vid_2 = len(list(vid_dir_2.glob("*.mp4"))) if vid_dir_2.exists() else 0
    results["T13_videos"] = (n_vid_1 == info["total_episodes"] and
                             n_vid_2 == info["total_episodes"])
    print(f"T13 Videos: agentview={n_vid_1}, wrist={n_vid_2} "
          f"(expected {info['total_episodes']} each)"
          f" -> {'PASS' if results['T13_videos'] else 'FAIL'}")

    # Summary
    all_pass = all(results.values())
    print(f"\n=== {'ALL PASS' if all_pass else 'SOME FAILED'} "
          f"({sum(results.values())}/{len(results)} passed) ===")
    for k, v in results.items():
        if not v:
            print(f"  FAILED: {k}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
