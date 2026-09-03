#!/usr/bin/env python3
"""List RoboTwin 2.0 *source* task directories under CLEAN_ROOT.

Each child folder with meta/info.json is a task. Derived pipeline outputs
(${TASK}_kptsim, _kptsim_lrb, _kptsim_lrbv30) and hidden/old dirs are skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DERIVED_SUFFIXES = ("_lrb3_kptsim", "_kptsim_lrbv30", "_kptsim_lrb", "_kptsim", "_lrb3")


def is_derived_name(name: str) -> bool:
    if name.startswith(".") or name.endswith("_old"):
        return True
    return any(name.endswith(suf) for suf in DERIVED_SUFFIXES)


def is_source_task_dir(path: Path) -> bool:
    if not path.is_dir() or is_derived_name(path.name):
        return False
    return (path / "meta" / "info.json").is_file()


def iter_source_tasks(clean_root: Path) -> list[dict]:
    rows = []
    for child in sorted(p for p in clean_root.iterdir() if p.is_dir()):
        if not is_source_task_dir(child):
            continue
        info = json.loads((child / "meta" / "info.json").read_text())
        rows.append(
            {
                "task": child.name,
                "path": str(child),
                "codebase_version": info.get("codebase_version"),
                "total_episodes": info.get("total_episodes"),
                "total_frames": info.get("total_frames"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--names-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.clean_root)
    if not root.is_dir():
        raise SystemExit(f"CLEAN_ROOT 不是目录: {root}")
    rows = iter_source_tasks(root)
    if args.names_only:
        for row in rows:
            print(row["task"])
        return
    for row in rows:
        print(
            f"{row['task']}\t{row['codebase_version']}\t"
            f"ep={row['total_episodes']}\tframes={row['total_frames']}"
        )


if __name__ == "__main__":
    main()
