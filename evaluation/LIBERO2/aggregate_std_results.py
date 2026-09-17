#!/usr/bin/env python3
"""Aggregate standard-LIBERO per-shard results across a split-topology run.

evaluation/LIBERO2/eval_libero_std.py writes one JSON file per (suite, start,
end) shard: logs/<suite>/std_<start>_to_<end>.json. When six clients each
process one or more shards of a suite (see
run_eval_libero_std_2server_6client_venv.sh), a suite's results end up split
across several files. This script is the standard-LIBERO analogue of
evaluation/LIBERO-plus2/aggregate_results.py: it sums total_episodes /
total_successes per suite and overall, merges the per-task tables, flags any
task that reported an error (crash), and applies a PASS/FAIL threshold on the
overall success rate so the launcher can gate on it directly.

Usage:
    python evaluation/LIBERO2/aggregate_std_results.py --root <eval_log_dir> \
        [--threshold 85]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# Matches TASK_SUITE_MAX_STEPS in evaluation/LIBERO2/eval_libero_std.py.
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]


def _rate(successes: int, total: int) -> float:
    return float(successes) / float(total) if total else 0.0


def aggregate(root_path: str) -> dict:
    per_suite: dict[str, dict] = {}
    grand_episodes = 0
    grand_successes = 0
    crash_tasks: list[dict] = []

    for suite in SUITES:
        suite_dir = os.path.join(root_path, "logs", suite)
        json_files = sorted(glob.glob(os.path.join(suite_dir, "std_*.json")))
        if not json_files:
            continue

        suite_episodes = 0
        suite_successes = 0
        per_task: dict[int, dict] = {}
        for file in json_files:
            with open(file, encoding="utf-8") as f:
                shard = json.load(f)
            suite_episodes += int(shard.get("total_episodes", 0))
            suite_successes += int(shard.get("total_successes", 0))
            for task in shard.get("per_task", []):
                task_id = task.get("task_id")
                if task_id in per_task:
                    # Two shards should never claim the same task_id; surface
                    # it loudly rather than silently double-counting.
                    raise ValueError(
                        f"duplicate task_id={task_id} in {suite} across shard files "
                        f"(found again in {file}); shard boundaries overlap"
                    )
                per_task[task_id] = task
                if task.get("error"):
                    crash_tasks.append(
                        {
                            "suite": suite,
                            "task_id": task_id,
                            "task_desc": task.get("task_desc"),
                            "error": task["error"],
                            "shard_file": file,
                        }
                    )

        per_suite[suite] = {
            "total_episodes": suite_episodes,
            "total_successes": suite_successes,
            "success_rate": _rate(suite_successes, suite_episodes),
            "num_shard_files": len(json_files),
            "num_tasks_reported": len(per_task),
            "per_task": [per_task[k] for k in sorted(per_task)],
        }
        grand_episodes += suite_episodes
        grand_successes += suite_successes

    overall = {
        "total_episodes": grand_episodes,
        "total_successes": grand_successes,
        "success_rate": _rate(grand_successes, grand_episodes),
    }

    return {
        "root": root_path,
        "per_suite": per_suite,
        "overall": overall,
        "crash_tasks": crash_tasks,
        "crash_task_count": len(crash_tasks),
    }


def _print_table(result: dict) -> None:
    print("\n| Suite | Episodes | Successes | SR | Shards | Tasks |")
    print("|-------|---------:|----------:|-----:|-------:|------:|")
    for suite, counts in result["per_suite"].items():
        sr_pct = 100.0 * counts["success_rate"]
        print(
            f"| {suite} | {counts['total_episodes']} | {counts['total_successes']} | "
            f"{sr_pct:.2f}% | {counts['num_shard_files']} | {counts['num_tasks_reported']} |"
        )
    overall = result["overall"]
    print(
        f"| **TOTAL** | **{overall['total_episodes']}** | **{overall['total_successes']}** | "
        f"**{100.0 * overall['success_rate']:.2f}%** | - | - |"
    )
    if result["crash_tasks"]:
        print(f"\nWARNING: {result['crash_task_count']} task(s) reported an error (see crash_tasks):")
        for item in result["crash_tasks"]:
            print(f"  - {item['suite']} task_id={item['task_id']}: {item['error']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate standard LIBERO per-shard results")
    parser.add_argument("--root", required=True, help="eval_log_dir containing logs/{suite}/std_*.json")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Overall SR percent required to PASS (e.g. 50 for smoke, 85 for full). "
        "If omitted, no PASS/FAIL verdict is printed and the exit code is always 0.",
    )
    args = parser.parse_args()

    result = aggregate(args.root)
    out_path = os.path.join(args.root, "overall_std_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    _print_table(result)
    print(f"\nSaved aggregated results to {out_path}")

    if result["overall"]["total_episodes"] == 0:
        print("FAIL: no results collected (check logs/<suite>/std_*.json exist)")
        return 1
    if result["crash_task_count"] > 0:
        print(f"FAIL: {result['crash_task_count']} task(s) reported an error (see crash_tasks above)")
        return 1
    if args.threshold is not None:
        sr_pct = 100.0 * result["overall"]["success_rate"]
        passed = sr_pct >= args.threshold
        print(f"\n{'PASS' if passed else 'FAIL'}: overall SR = {sr_pct:.2f}% (threshold={args.threshold}%)")
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
