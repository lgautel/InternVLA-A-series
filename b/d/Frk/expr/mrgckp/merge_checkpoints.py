#!/usr/bin/env python3
"""Merge two InternVLA-A1.5 checkpoints by weighted averaging of shared parameters.

Usage:
    python merge_checkpoints.py \
        --ckpt-a <path_a> --weight-a 0.3 \
        --ckpt-b <path_b> --weight-b 0.7 \
        --output  <output_path> \
        [--config-from a|b]  # which checkpoint's config/stats/train_config to copy (default: a)
        [--dry-run]          # print stats without writing

Both checkpoint dirs are expected to contain at least model.safetensors.
Config/stats/train_config JSON files are copied from the --config-from side.
Non-floating-point tensors (e.g. int64 grid sizes) are copied from --config-from
side unchanged since averaging integers is meaningless.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def merge(
    path_a: Path,
    path_b: Path,
    weight_a: float,
    weight_b: float,
    output: Path,
    config_from: str = "a",
    dry_run: bool = False,
) -> dict:
    assert abs(weight_a + weight_b - 1.0) < 1e-6, f"Weights must sum to 1.0, got {weight_a + weight_b}"

    st_a = path_a / "model.safetensors"
    st_b = path_b / "model.safetensors"
    assert st_a.is_file(), f"Not found: {st_a}"
    assert st_b.is_file(), f"Not found: {st_b}"

    print(f"Loading A: {st_a}")
    tensors_a = load_file(str(st_a), device="cpu")
    print(f"Loading B: {st_b}")
    tensors_b = load_file(str(st_b), device="cpu")

    keys_a, keys_b = set(tensors_a), set(tensors_b)
    shared = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    report = {
        "keys_a": len(keys_a),
        "keys_b": len(keys_b),
        "shared": len(shared),
        "only_a": len(only_a),
        "only_b": len(only_b),
        "merged_float": 0,
        "copied_nonfloat": 0,
        "shape_mismatches": [],
    }

    print(f"Keys  — A: {report['keys_a']}, B: {report['keys_b']}, shared: {report['shared']}")
    if only_a:
        print(f"  A-only ({len(only_a)}): {only_a[:5]}{'...' if len(only_a) > 5 else ''}")
    if only_b:
        print(f"  B-only ({len(only_b)}): {only_b[:5]}{'...' if len(only_b) > 5 else ''}")

    merged = {}
    for key in shared:
        ta, tb = tensors_a[key], tensors_b[key]
        if ta.shape != tb.shape:
            report["shape_mismatches"].append((key, list(ta.shape), list(tb.shape)))
            print(f"  SKIP shape mismatch: {key} {ta.shape} vs {tb.shape}")
            merged[key] = ta if config_from == "a" else tb
            continue

        if ta.is_floating_point():
            merged[key] = weight_a * ta.float() + weight_b * tb.float()
            merged[key] = merged[key].to(ta.dtype)
            report["merged_float"] += 1
        else:
            merged[key] = ta if config_from == "a" else tb
            report["copied_nonfloat"] += 1

    for key in only_a:
        merged[key] = tensors_a[key]
    for key in only_b:
        merged[key] = tensors_b[key]

    print(f"Merged: {report['merged_float']} float, {report['copied_nonfloat']} non-float")
    if report["shape_mismatches"]:
        print(f"Shape mismatches: {len(report['shape_mismatches'])}")

    if dry_run:
        print("DRY RUN — not writing.")
        return report

    output.mkdir(parents=True, exist_ok=True)

    out_st = output / "model.safetensors"
    print(f"Saving merged weights → {out_st}")
    save_file(merged, str(out_st))
    print(f"  Size: {out_st.stat().st_size / 1e9:.2f} GB")

    src = path_a if config_from == "a" else path_b
    for fname in ("config.json", "stats.json", "train_config.json"):
        src_f = src / fname
        if src_f.is_file():
            shutil.copy2(str(src_f), str(output / fname))
            print(f"  Copied {fname} from {'A' if config_from == 'a' else 'B'}")

    merge_meta = {
        "merge_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_a": str(path_a),
        "checkpoint_b": str(path_b),
        "weight_a": weight_a,
        "weight_b": weight_b,
        "config_from": config_from,
        "shared_keys": report["shared"],
        "merged_float_keys": report["merged_float"],
        "copied_nonfloat_keys": report["copied_nonfloat"],
        "only_a_keys": report["only_a"],
        "only_b_keys": report["only_b"],
    }
    with open(output / "merge_info.json", "w") as f:
        json.dump(merge_meta, f, indent=2)
    print(f"  Wrote merge_info.json")

    print("Done.")
    return report


def main():
    parser = argparse.ArgumentParser(description="Merge two InternVLA-A1.5 checkpoints by weighted average")
    parser.add_argument("--ckpt-a", type=Path, required=True, help="Path to checkpoint A directory")
    parser.add_argument("--weight-a", type=float, required=True, help="Weight for checkpoint A")
    parser.add_argument("--ckpt-b", type=Path, required=True, help="Path to checkpoint B directory")
    parser.add_argument("--weight-b", type=float, required=True, help="Weight for checkpoint B")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for merged checkpoint")
    parser.add_argument("--config-from", choices=["a", "b"], default="a", help="Copy config/stats from A or B")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, don't write")
    args = parser.parse_args()

    report = merge(args.ckpt_a, args.ckpt_b, args.weight_a, args.weight_b, args.output, args.config_from, args.dry_run)

    if report["shape_mismatches"]:
        print(f"\nWARNING: {len(report['shape_mismatches'])} shape mismatches", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
