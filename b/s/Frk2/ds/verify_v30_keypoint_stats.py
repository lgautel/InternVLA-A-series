"""Acceptance checks for observation.keypoint_3d stats on a LeRobot v3.0 dataset.

Usage:
    source /B/VENV/itnvla15rbt20/bin/activate
    python b/s/Frk2/ds/verify_v30_keypoint_stats.py \
        --dataset /home/a26113/b/Dta/plug_into_socket_franka3_15hz_lerobot_4d_v30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DS_DIR = Path(__file__).resolve().parent
if str(_DS_DIR) not in sys.path:
    sys.path.insert(0, str(_DS_DIR))

from kpt_stats import DEFAULT_DATASET, verify_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results = verify_dataset(args.dataset)
    n_ok = sum(1 for r in results if r.ok)
    print(f"\n=== Keypoint stats verification: {args.dataset} ===")
    for r in results:
        mark = "PASS" if r.ok else "FAIL"
        print(f"  [{mark}] {r.name}")
        print(f"         {r.detail}")
    print(f"\nSummary: {n_ok}/{len(results)} PASS")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
