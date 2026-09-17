#!/usr/bin/env python3
"""Check LIBERO-plus evaluation progress and per-suite success rates.

Usage:
    python evaluation/LIBERO-plus2/check_progress.py --eval_dir <EVAL_DIR>
    python evaluation/LIBERO-plus2/check_progress.py  # auto-find latest run
"""
import argparse
import glob
import os
import re
from pathlib import Path

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_TOTAL = {"libero_spatial": 2402, "libero_object": 2518, "libero_goal": 2591, "libero_10": 2519}

TASK_PAT = re.compile(r'-> (\d+)/(\d+) \(running total')


def find_latest_eval_dir():
    base = Path.home() / "b/Ckp"
    # Find all full_* dirs that contain worker_gpu0, pick most recently modified
    candidates = []
    for p in base.rglob("full_2026*/worker_gpu0"):
        parent = p.parent
        try:
            mtime = max(f.stat().st_mtime for f in parent.rglob("client_*.log"))
            candidates.append((mtime, str(parent)))
        except (StopIteration, ValueError):
            candidates.append((p.stat().st_mtime, str(parent)))
    if candidates:
        return sorted(candidates, reverse=True)[0][1]
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", type=str, default=None,
                   help="Eval output directory (auto-detected if omitted)")
    return p.parse_args()


def main():
    args = parse_args()
    eval_dir = args.eval_dir or find_latest_eval_dir()
    if not eval_dir or not os.path.isdir(eval_dir):
        print("ERROR: could not find eval directory. Pass --eval_dir <path>")
        return

    print(f"Eval dir: {eval_dir}\n")

    suite_succ = {s: 0 for s in SUITES}
    suite_done = {s: 0 for s in SUITES}
    suite_crashed = {s: 0 for s in SUITES}
    suite_latest = {s: "" for s in SUITES}

    for suite in SUITES:
        logs = glob.glob(f"{eval_dir}/worker_gpu*/client_{suite}_*.log")
        for log in sorted(logs):
            with open(log, errors="replace") as f:
                for line in f:
                    m = TASK_PAT.search(line)
                    if m:
                        suite_succ[suite] += int(m.group(1))
                        suite_done[suite] += int(m.group(2))
                        # extract timestamp
                        ts = line[line.find("]") - 13 : line.find("]") + 1]
                        suite_latest[suite] = ts
                    if "CRASHED" in line:
                        suite_crashed[suite] += 1

    print(f"{'Suite':<20} {'Done':>8}  {'Total':>8}  {'%Done':>7}  {'Succ':>6}  {'SR%':>7}  {'Crash':>6}  Latest")
    print("-" * 85)
    grand_done = grand_total = grand_succ = grand_crashed = 0
    for suite in SUITES:
        full = SUITE_TOTAL[suite]
        done = suite_done[suite]
        succ = suite_succ[suite]
        crashed = suite_crashed[suite]
        pct_done = 100.0 * done / full if full else 0
        sr = 100.0 * succ / done if done else 0
        print(f"{suite:<20} {done:>8}/{full:<8}  {pct_done:>6.1f}%  {succ:>6}  {sr:>6.2f}%  {crashed:>6}  {suite_latest[suite]}")
        grand_done += done
        grand_total += full
        grand_succ += succ
        grand_crashed += crashed

    print("-" * 85)
    pct_done = 100.0 * grand_done / grand_total if grand_total else 0
    sr = 100.0 * grand_succ / grand_done if grand_done else 0
    print(f"{'TOTAL':<20} {grand_done:>8}/{grand_total:<8}  {pct_done:>6.1f}%  {grand_succ:>6}  {sr:>6.2f}%  {grand_crashed:>6}")


if __name__ == "__main__":
    main()
