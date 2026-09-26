#!/usr/bin/env python3
"""
Weighted averaging of two model checkpoints.

Usage:
    python merge_checkpoints.py \\
        --ckpt1 /path/to/checkpoint1 --weight1 0.3 \\
        --ckpt2 /path/to/checkpoint2 --weight2 0.7 \\
        --output /path/to/merged
"""
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any

import torch
import safetensors.torch


def load_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    """Load weights from checkpoint (supports both .safetensors and .bin)."""
    safetensors_file = ckpt_path / "model.safetensors"
    bin_file = ckpt_path / "pytorch_model.bin"

    if safetensors_file.exists():
        print(f"Loading from {safetensors_file}")
        return safetensors.torch.load_file(safetensors_file, device="cpu")
    elif bin_file.exists():
        print(f"Loading from {bin_file}")
        return torch.load(bin_file, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No model weights found in {ckpt_path}")


def save_state_dict(state_dict: Dict[str, torch.Tensor], output_path: Path):
    """Save merged weights as safetensors."""
    output_file = output_path / "model.safetensors"
    print(f"Saving merged weights to {output_file}")
    safetensors.torch.save_file(state_dict, output_file)


def merge_configs(ckpt1: Path, ckpt2: Path, output: Path):
    """Copy config files from ckpt1 (assumes both have compatible configs)."""
    for config_name in ["config.json", "configuration_internvla_a1_5.json"]:
        src = ckpt1 / config_name
        if src.exists():
            dst = output / config_name
            print(f"Copying {config_name}")
            shutil.copy2(src, dst)

    # Copy tokenizer files if present
    for tok_file in ckpt1.glob("tokenizer*"):
        shutil.copy2(tok_file, output / tok_file.name)
    for tok_file in ckpt1.glob("*.json"):
        if "tokenizer" in tok_file.name.lower() or tok_file.name in ["special_tokens_map.json", "vocab.json"]:
            shutil.copy2(tok_file, output / tok_file.name)


def weighted_merge(
    ckpt1_path: Path,
    ckpt2_path: Path,
    weight1: float,
    weight2: float,
    output_path: Path,
):
    """
    Merge two checkpoints with weighted averaging.

    Args:
        ckpt1_path: First checkpoint directory
        ckpt2_path: Second checkpoint directory
        weight1: Weight for first checkpoint
        weight2: Weight for second checkpoint
        output_path: Output directory for merged checkpoint
    """
    assert abs(weight1 + weight2 - 1.0) < 1e-6, f"Weights must sum to 1.0, got {weight1} + {weight2}"

    output_path.mkdir(parents=True, exist_ok=True)

    # Load state dicts
    state1 = load_state_dict(ckpt1_path)
    state2 = load_state_dict(ckpt2_path)

    # Check key consistency
    keys1 = set(state1.keys())
    keys2 = set(state2.keys())
    common = keys1 & keys2
    only1 = keys1 - keys2
    only2 = keys2 - keys1

    print(f"\nCheckpoint 1: {len(keys1)} parameters")
    print(f"Checkpoint 2: {len(keys2)} parameters")
    print(f"Common parameters: {len(common)}")
    if only1:
        print(f"Only in ckpt1: {len(only1)} params (will be scaled by {weight1})")
    if only2:
        print(f"Only in ckpt2: {len(only2)} params (will be scaled by {weight2})")

    # Merge
    merged = {}
    for key in common:
        t1, t2 = state1[key], state2[key]
        if t1.shape != t2.shape:
            print(f"WARNING: Shape mismatch for {key}: {t1.shape} vs {t2.shape}, skipping")
            continue
        merged[key] = weight1 * t1 + weight2 * t2

    # Add unique keys
    for key in only1:
        merged[key] = weight1 * state1[key]
    for key in only2:
        merged[key] = weight2 * state2[key]

    # Save
    save_state_dict(merged, output_path)
    merge_configs(ckpt1_path, ckpt2_path, output_path)

    print(f"\n✓ Merged checkpoint saved to {output_path}")
    print(f"  Total parameters: {len(merged)}")


def main():
    parser = argparse.ArgumentParser(description="Weighted merge of two checkpoints")
    parser.add_argument("--ckpt1", type=Path, required=True, help="First checkpoint path")
    parser.add_argument("--weight1", type=float, required=True, help="Weight for first checkpoint")
    parser.add_argument("--ckpt2", type=Path, required=True, help="Second checkpoint path")
    parser.add_argument("--weight2", type=float, required=True, help="Weight for second checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output path for merged checkpoint")
    args = parser.parse_args()

    weighted_merge(args.ckpt1, args.ckpt2, args.weight1, args.weight2, args.output)


if __name__ == "__main__":
    main()
