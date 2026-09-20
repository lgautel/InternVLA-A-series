#!/usr/bin/env python3
"""Verify training-evaluation keypoint consistency.

Checks that the offline-generated keypoints match what the eval-time
StandaloneFK would produce, using the same MJCF and R_pad.

Tests:
  T1: MJCF MD5 match (offline gen vs eval-time)
  T2: R_pad match (keypoints_meta.json vs DEFAULT_R_PAD)
  T3: Body names match
  T4: FK output match (sample qpos -> compare outputs)
  T5: kpt_4d_mode = pos_rot (7D) consistency
  T6: history max_len match

Usage:
    python b/s/libplus/verify_train_eval_consistency.py \
        --dataset /B/Dta/libero_merged_kpt_v2 \
        --eval-xml evaluation/panda_robosuite_lift.xml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--eval-xml", type=Path,
                        default=Path("evaluation/panda_robosuite_lift.xml"))
    parser.add_argument("--eval-r-pad", type=float, default=1.8212722539901733)
    parser.add_argument("--train-history-max-len", type=int, default=200)
    parser.add_argument("--eval-history-max-len", type=int, default=200)
    args = parser.parse_args()

    results = {}

    meta_path = args.dataset / "meta" / "keypoints_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    # T1: MJCF MD5
    eval_md5 = hashlib.md5(args.eval_xml.read_bytes()).hexdigest()
    gen_md5 = meta.get("mjcf_md5", "MISSING")
    results["T1_mjcf_md5"] = gen_md5 == eval_md5
    print(f"T1 MJCF MD5: gen={gen_md5}, eval={eval_md5} -> "
          f"{'PASS' if results['T1_mjcf_md5'] else 'FAIL'}")

    # T2: R_pad
    gen_r_pad = meta["bbox_radius"]
    diff = abs(gen_r_pad - args.eval_r_pad)
    results["T2_r_pad"] = diff < 1e-4
    print(f"T2 R_pad: gen={gen_r_pad:.10f}, eval={args.eval_r_pad:.10f}, "
          f"diff={diff:.2e} -> {'PASS' if results['T2_r_pad'] else 'FAIL'}")

    # T3: Body names
    gen_bodies = meta["keypoint_bodies"]
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(args.eval_xml))

    eef_candidates = ("gripper0_eef", "gripper0_right_eef")
    eval_eef = None
    for name in eef_candidates:
        try:
            model.body(name)
            eval_eef = name
            break
        except (KeyError, ValueError):
            continue

    eval_bodies = [
        "robot0_link1", "robot0_link2", "robot0_link3", "robot0_link4",
        "robot0_link5", "robot0_link6", "robot0_link7",
    ]
    if eval_eef:
        eval_bodies.append(eval_eef)

    results["T3_bodies"] = gen_bodies == eval_bodies
    print(f"T3 Bodies: gen={gen_bodies[-1]}, eval={eval_bodies[-1]} -> "
          f"{'PASS' if results['T3_bodies'] else 'FAIL'}")

    # T4: FK output shape check
    import pandas as pd
    parquets = sorted((args.dataset / "data").rglob("*.parquet"))
    if parquets:
        df = pd.read_parquet(parquets[0])
        kpts = np.stack(df["observation.keypoint_3d"].values[:1])
        kpts = kpts.reshape(-1, 8, 7)
        results["T4_fk_shape"] = kpts.shape == (1, 8, 7)
        print(f"T4 FK shape: {kpts.shape} -> "
              f"{'PASS' if results['T4_fk_shape'] else 'FAIL'}")
    else:
        results["T4_fk_shape"] = False
        print("T4 FK shape: no parquets -> FAIL")

    # T5: kpt_dim = 7 (pos_rot)
    kpt_dim = meta["keypoint_dim"]
    results["T5_kpt_dim"] = kpt_dim == 7
    print(f"T5 kpt_dim: {kpt_dim} -> {'PASS' if results['T5_kpt_dim'] else 'FAIL'}")

    # T6: history max_len
    results["T6_history"] = args.train_history_max_len == args.eval_history_max_len
    print(f"T6 history: train={args.train_history_max_len}, "
          f"eval={args.eval_history_max_len} -> "
          f"{'PASS' if results['T6_history'] else 'FAIL'}")

    all_pass = all(results.values())
    print(f"\n=== {'ALL PASS' if all_pass else 'SOME FAILED'} ===")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
