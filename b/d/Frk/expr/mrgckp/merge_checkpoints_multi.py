#!/usr/bin/env python3
"""
Weighted averaging of multiple model checkpoints.

Usage:
    python merge_checkpoints_multi.py \\
        --ckpt /path/to/checkpoint1 --weight 0.3 \\
        --ckpt /path/to/checkpoint2 --weight 0.5 \\
        --ckpt /path/to/checkpoint3 --weight 0.2 \\
        --output /path/to/merged
"""
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

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


def merge_configs(ckpt_paths: List[Path], output: Path):
    """Copy config files from first checkpoint (assumes all have compatible configs)."""
    ckpt1 = ckpt_paths[0]
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


def weighted_merge_multi(
    checkpoints: List[Tuple[Path, float]],
    output_path: Path,
):
    """
    Merge multiple checkpoints with weighted averaging.

    Args:
        checkpoints: List of (checkpoint_path, weight) tuples
        output_path: Output directory for merged checkpoint
    """
    # Validate weights sum to 1.0
    total_weight = sum(w for _, w in checkpoints)
    assert abs(total_weight - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total_weight}"

    output_path.mkdir(parents=True, exist_ok=True)

    # Load all state dicts
    print(f"\n{'='*60}")
    print(f"Loading {len(checkpoints)} checkpoints")
    print(f"{'='*60}")
    state_dicts = []
    for i, (ckpt_path, weight) in enumerate(checkpoints, 1):
        print(f"\nCheckpoint {i}: {ckpt_path}")
        print(f"  Weight: {weight}")
        state = load_state_dict(ckpt_path)
        state_dicts.append(state)
        print(f"  Parameters: {len(state)}")

    # Analyze key consistency across all checkpoints
    print(f"\n{'='*60}")
    print("Analyzing parameter consistency")
    print(f"{'='*60}")

    all_keys = [set(sd.keys()) for sd in state_dicts]
    common = set.intersection(*all_keys)

    print(f"\nCommon parameters across all {len(checkpoints)} checkpoints: {len(common)}")

    for i, keys in enumerate(all_keys, 1):
        unique = keys - common
        if unique:
            print(f"  Checkpoint {i} unique: {len(unique)} params")

    # Merge common parameters
    print(f"\n{'='*60}")
    print("Merging parameters")
    print(f"{'='*60}")

    merged = {}
    skipped = 0

    # Merge common parameters with weighted average
    for key in common:
        tensors = [state_dicts[i][key] for i in range(len(checkpoints))]

        # Check shape consistency
        shapes = [t.shape for t in tensors]
        if len(set(shapes)) > 1:
            print(f"WARNING: Shape mismatch for {key}: {shapes}, skipping")
            skipped += 1
            continue

        # Weighted average
        merged[key] = sum(w * state_dicts[i][key] for i, (_, w) in enumerate(checkpoints))

    # Handle unique parameters (scale by their checkpoint's weight)
    for i, (ckpt_path, weight) in enumerate(checkpoints):
        unique_keys = all_keys[i] - common
        for key in unique_keys:
            if key not in merged:  # Avoid overwriting if already added
                merged[key] = weight * state_dicts[i][key]

    print(f"\nMerged parameters: {len(merged)}")
    if skipped > 0:
        print(f"Skipped (shape mismatch): {skipped}")

    # Save
    save_state_dict(merged, output_path)
    merge_configs([ckpt for ckpt, _ in checkpoints], output_path)

    print(f"\n{'='*60}")
    print(f"✓ Merged checkpoint saved to {output_path}")
    print(f"  Total parameters: {len(merged)}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Weighted merge of multiple checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python merge_checkpoints_multi.py \\
        --ckpt /path/to/ckpt1 --weight 0.3 \\
        --ckpt /path/to/ckpt2 --weight 0.5 \\
        --ckpt /path/to/ckpt3 --weight 0.2 \\
        --output /path/to/output
        """
    )
    parser.add_argument("--ckpt", action="append", dest="ckpts", type=Path, required=True,
                       help="Checkpoint path (can be specified multiple times)")
    parser.add_argument("--weight", action="append", dest="weights", type=float, required=True,
                       help="Weight for corresponding checkpoint (must match --ckpt count)")
    parser.add_argument("--output", type=Path, required=True, help="Output path for merged checkpoint")
    args = parser.parse_args()

    if len(args.ckpts) != len(args.weights):
        parser.error(f"Number of checkpoints ({len(args.ckpts)}) must match number of weights ({len(args.weights)})")

    if len(args.ckpts) < 2:
        parser.error("At least 2 checkpoints required for merging")

    checkpoints = list(zip(args.ckpts, args.weights))
    weighted_merge_multi(checkpoints, args.output)


if __name__ == "__main__":
    main()
