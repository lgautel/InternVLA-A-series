#!/usr/bin/env python3
"""T1 acceptance: live agentview + wrist vs training orientation.

Requires CLIENT_VENV (mujoco/libero). `--render_backend egl` needs a free
GPU/EGL device; `--render_backend osmesa` renders on CPU and needs libosmesa6.

Usage:
    export LIBERO_HOME=/home/a26113/DATA/LIBERO-plus
    export LIBERO_CONFIG_PATH=<dir containing config.yaml>
    python evaluation/LIBERO2/test_live_orientation.py --render_backend osmesa

Exit 0 = both cameras match training RAW (do not rotate at eval).
The orientation contract is backend-independent, so this must pass on both
backends; run it once per backend when switching (OSMesa acceptance O3).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="T1 live train/eval orientation contract")
    parser.add_argument("--dataset_root", type=str, default="/B/Dta/opvla_libero_merged_kpt")
    parser.add_argument("--num_samples", type=int, default=80)
    parser.add_argument("--min_ratio", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json_out", type=str, default="")
    from evaluation.LIBERO2.render_backend import add_render_backend_cli

    add_render_backend_cli(parser)
    args = parser.parse_args()

    from evaluation.LIBERO2.orientation_contract import (
        DEFAULT_TRAIN_ROOT,
        evaluate_orientation_contract,
        setup_client_render_env,
    )

    backend = setup_client_render_env(backend=args.render_backend)
    train_root = Path(args.dataset_root) / "videos"
    if not train_root.exists():
        train_root = DEFAULT_TRAIN_ROOT

    print("=" * 60)
    print(f"T1 live [{backend}]: agentview + wrist vs training orientation")
    print("=" * 60)
    report = evaluate_orientation_contract(
        n_train=args.num_samples,
        seed=args.seed,
        min_ratio=args.min_ratio,
        train_root=train_root,
    )
    report["render_backend"] = backend
    vis = report.pop("_vis", None)
    del vis  # images not printed
    print(json.dumps({k: v for k, v in report.items() if k != "_vis"}, indent=2, default=str))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
        print("wrote", args.json_out)

    for name, cam in report["cameras"].items():
        tag = "PASS" if cam["matches_raw"] else "FAIL"
        print(
            f"  {name}: {tag}  raw={cam['mse_live_raw_vs_train']}  "
            f"rot180={cam['mse_live_rot180_vs_train']}  "
            f"ratio={cam['ratio_rot_over_raw']}  vote={cam['majority_vote']}"
        )

    overall = "PASS" if report["passed"] else "FAIL"
    print(f"\nOVERALL: {overall}  (training must match LIVE RAW on both cameras)")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
