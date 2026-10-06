#!/usr/bin/env python3
"""Consolidate failure records from a LIBERO-plus Goal evaluation run.

Scans EVAL_LOG_DIR for per-episode failure JSONs, action NPZs, and failure
videos, then produces:
  1. failures_manifest.json — indexed failure records with category, paths to
     associated action NPZ / video, and a one-liner replay command
  2. replay_commands.sh — executable script containing all replay commands

The evaluation infrastructure (eval_libero_plus.py) saves:
  - failures/{suite}/task{id}_ep{idx}.json  (always, for every failed episode)
  - actions/{suite}/{name}_ep{idx}.npz      (when --save_actions, default ON)
  - videos/{suite}/rollout_{name}_*_failure.mp4  (when --save_failure_videos)

This script reads those artifacts and enriches them with category info from
task_classification.json.

Usage:
    python collect_failures.py \\
        --eval-log-dir /B/Log/4dwvlaLbPlusGol0929/20261001_120000_eval \\
        [--task-classification /B/SRC/LIBERO-plus/libero/libero/benchmark/task_classification.json] \\
        [--suite libero_goal]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_id2category(task_cls_path: str, suite: str) -> dict[int, str]:
    """Map 0-based task_id to perturbation category."""
    with open(task_cls_path, encoding="utf-8") as f:
        mapping = json.load(f)
    if suite not in mapping:
        return {}
    id2cat: dict[int, str] = {}
    for item in mapping[suite]:
        id2cat[int(item["id"]) - 1] = item["category"]
    return id2cat


def collect(eval_log_dir: str, suite: str, task_cls_path: str | None) -> dict:
    root = Path(eval_log_dir)
    failures_dir = root / "failures" / suite
    actions_dir = root / "actions" / suite
    videos_dir = root / "videos" / suite
    logs_dir = root / "logs" / suite

    id2cat: dict[int, str] = {}
    if task_cls_path and Path(task_cls_path).is_file():
        id2cat = _load_id2category(task_cls_path, suite)

    failure_records: list[dict] = []
    if failures_dir.is_dir():
        for fj in sorted(failures_dir.glob("task*_ep*.json")):
            with open(fj) as f:
                rec = json.load(f)
            tid = rec.get("task_id")
            rec["category"] = id2cat.get(tid, "unknown") if tid is not None else "unknown"
            rec["_failure_json"] = str(fj.relative_to(root))

            task_name = rec.get("task_name", "")
            ep = rec.get("episode_idx", 0)
            npz_path = actions_dir / f"{task_name}_ep{ep}.npz"
            if npz_path.exists():
                rec["_action_npz"] = str(npz_path.relative_to(root))
                import numpy as np

                d = np.load(npz_path, allow_pickle=True)
                rec["_action_steps"] = int(d["actions"].shape[0]) if "actions" in d else -1

            vid_path = videos_dir / f"rollout_{task_name}_episode{ep}_failure.mp4"
            if vid_path.exists():
                rec["_failure_video"] = str(vid_path.relative_to(root))

            failure_records.append(rec)

    crash_records: list[dict] = []
    if logs_dir.is_dir():
        for fl in sorted(logs_dir.glob("*_failures.json")):
            with open(fl) as f:
                crashes = json.load(f)
            for crash in crashes:
                crash["_source"] = str(fl.relative_to(root))
                if "task_id" in crash and crash["task_id"] in id2cat:
                    crash.setdefault("category", id2cat[crash["task_id"]])
            crash_records.extend(crashes)

    by_category: dict[str, list[int]] = {}
    for rec in failure_records:
        by_category.setdefault(rec["category"], []).append(rec.get("task_id", -1))

    return {
        "eval_log_dir": str(root),
        "suite": suite,
        "total_failures": len(failure_records),
        "total_crashes": len(crash_records),
        "failures_with_actions": sum(1 for r in failure_records if "_action_npz" in r),
        "failures_with_videos": sum(1 for r in failure_records if "_failure_video" in r),
        "by_category": {
            cat: {"count": len(tids), "task_ids": sorted(set(tids))}
            for cat, tids in sorted(by_category.items())
        },
        "failures": failure_records,
        "crashes": crash_records,
    }


def generate_replay_script(manifest: dict, here: str) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "# Auto-generated replay commands for failed evaluation tasks.",
        f"# Source: {manifest['eval_log_dir']}",
        f"# Total failures: {manifest['total_failures']}",
        f"# Failures with action NPZ: {manifest['failures_with_actions']}",
        "",
        "set -euo pipefail",
        "",
        f'REPLAY_SCRIPT="{here}/replay_failure.sh"',
        f'EVAL_LOG_DIR="{manifest["eval_log_dir"]}"',
        "",
        "# --- Per-task replay commands ---",
        '# Usage: source this file, or run individual lines.',
        '# Each command replays one failed task against a running server.',
        '# Start a server first:',
        '#   CKPT_PATH=<checkpoint> SERVER_GPU=0 bash $REPLAY_SCRIPT --start-server-only',
        "",
    ]
    for rec in manifest["failures"]:
        tid = rec.get("task_id", -1)
        name = rec.get("task_name", f"task{tid}")
        ep = rec.get("episode_idx", 0)
        cat = rec.get("category", "unknown")
        fj = rec.get("_failure_json", "")
        lines.append(f"# [{cat}] Task {tid}: {name} (ep={ep})")
        lines.append(
            f'bash "$REPLAY_SCRIPT" '
            f'--failure-json "$EVAL_LOG_DIR/{fj}" '
            f'--output-dir "$EVAL_LOG_DIR/replay/task{tid}_ep{ep}"'
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolidate LIBERO-plus Goal evaluation failures")
    parser.add_argument("--eval-log-dir", required=True)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument(
        "--task-classification",
        default=None,
        help="Path to task_classification.json (default: $LIBERO_HOME/libero/libero/benchmark/task_classification.json)",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    if args.task_classification is None:
        libero_home = os.environ.get("LIBERO_HOME", "/B/SRC/LIBERO-plus")
        args.task_classification = f"{libero_home}/libero/libero/benchmark/task_classification.json"

    manifest = collect(args.eval_log_dir, args.suite, args.task_classification)

    out_dir = Path(args.eval_log_dir)
    manifest_path = out_dir / "failures_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    replay_path = out_dir / "replay_commands.sh"
    with open(replay_path, "w") as f:
        f.write(generate_replay_script(manifest, str(here)))
    os.chmod(replay_path, 0o755)

    print(f"Manifest:  {manifest_path}")
    print(f"Replay:    {replay_path}")
    print(f"Failures:  {manifest['total_failures']}  (actions: {manifest['failures_with_actions']}, "
          f"videos: {manifest['failures_with_videos']})")
    print(f"Crashes:   {manifest['total_crashes']}")
    if manifest["by_category"]:
        print("By category:")
        for cat, info in sorted(manifest["by_category"].items()):
            print(f"  {cat}: {info['count']} failures ({len(info['task_ids'])} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
