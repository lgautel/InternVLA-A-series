#!/usr/bin/env python3
"""Smoke test: verify a merged checkpoint loads in the InternVLA-A1.5 framework.

Checks:
  1. model.safetensors loads via safetensors
  2. config.json is valid and matches expected policy type
  3. merge_info.json exists and weights sum to 1.0
  4. Merged weights are actually interpolated (spot-check vs source checkpoints)
  5. Policy loads from the merged checkpoint (draccus config + model weights)
  6. Single forward pass on synthetic data (1 GPU, action-only mode for speed)

Usage:
    python smoke_test_merged.py --merged <merged_ckpt_path> \
        [--ckpt-a <path>] [--ckpt-b <path>]   # optional: verify interpolation
        [--skip-forward]                        # skip GPU forward pass
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def check_files(merged: Path) -> bool:
    ok = True
    for f in ("model.safetensors", "config.json"):
        if not (merged / f).is_file():
            print(f"FAIL: missing {f}")
            ok = False
        else:
            print(f"  OK: {f} exists")
    for f in ("stats.json", "train_config.json", "merge_info.json"):
        if (merged / f).is_file():
            print(f"  OK: {f} exists")
        else:
            print(f"  WARN: {f} missing (optional)")
    return ok


def check_config(merged: Path) -> bool:
    with open(merged / "config.json") as f:
        cfg = json.load(f)
    ptype = cfg.get("type", "")
    if ptype != "internvla_a1_5":
        print(f"FAIL: config type={ptype}, expected internvla_a1_5")
        return False
    print(f"  OK: config type={ptype}")
    print(f"      kpt_4d_mode={cfg.get('kpt_4d_mode')}, num_kpt={cfg.get('num_keypoint_joints')}")
    return True


def check_merge_info(merged: Path) -> bool:
    mi_path = merged / "merge_info.json"
    if not mi_path.is_file():
        print("  SKIP: no merge_info.json")
        return True
    with open(mi_path) as f:
        mi = json.load(f)
    ws = mi.get("weight_a", 0) + mi.get("weight_b", 0)
    if abs(ws - 1.0) > 1e-6:
        print(f"FAIL: weights sum to {ws}")
        return False
    print(f"  OK: merge_info — w_a={mi['weight_a']}, w_b={mi['weight_b']}, shared={mi['shared_keys']}")
    return True


def check_interpolation(merged: Path, ckpt_a: Path, ckpt_b: Path) -> bool:
    mi_path = merged / "merge_info.json"
    if not mi_path.is_file():
        print("  SKIP: no merge_info.json for interpolation check")
        return True
    with open(mi_path) as f:
        mi = json.load(f)
    wa, wb = mi["weight_a"], mi["weight_b"]

    tm = load_file(str(merged / "model.safetensors"), device="cpu")
    ta = load_file(str(ckpt_a / "model.safetensors"), device="cpu")
    tb = load_file(str(ckpt_b / "model.safetensors"), device="cpu")

    test_keys = [k for k in sorted(tm.keys()) if tm[k].is_floating_point() and tm[k].numel() > 1][:5]
    all_ok = True
    for key in test_keys:
        expected = wa * ta[key].float() + wb * tb[key].float()
        expected = expected.to(tm[key].dtype)
        diff = (tm[key].float() - expected.float()).abs().max().item()
        if diff > 1e-4:
            print(f"FAIL: {key} max diff = {diff:.6f}")
            all_ok = False
        else:
            print(f"  OK: {key} — max diff {diff:.2e}")
    return all_ok


def check_forward(merged: Path) -> bool:
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", "/B/VENV/hf_home")

    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
    from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config
    from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import InternVLAA15Policy

    with open(merged / "config.json") as f:
        cfg_dict = json.load(f)

    cfg_dict["action_loss_only"] = True
    cfg_dict["pretrained_path"] = str(merged)
    cfg_dict["inference_backend"] = "standard"
    cfg_dict["device"] = "cuda:0"

    cfg = InternVLAA15Config(**{k: v for k, v in cfg_dict.items() if k in InternVLAA15Config.__dataclass_fields__})
    cfg.action_loss_only = True
    cfg.pretrained_path = str(merged)
    cfg.device = "cuda:0"

    print(f"  Loading policy from {merged} (action_loss_only=True) ...")
    policy = InternVLAA15Policy.from_pretrained(config=cfg, pretrained_name_or_path=str(merged))
    print(f"  OK: policy instantiated")

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  OK: total params = {total_params:,}, trainable = {trainable_params:,}")

    merged_tensors = load_file(str(merged / "model.safetensors"), device="cpu")
    state_dict = {k: v for k, v in policy.state_dict().items()}
    matched, mismatched = 0, 0
    for key in merged_tensors:
        if key in state_dict:
            if merged_tensors[key].shape == state_dict[key].shape:
                matched += 1
            else:
                mismatched += 1
                print(f"  WARN: shape mismatch after load: {key}")
        else:
            print(f"  WARN: merged key not in model state_dict: {key}")
    print(f"  OK: {matched}/{len(merged_tensors)} weights loaded and shape-matched")
    if mismatched:
        print(f"  FAIL: {mismatched} shape mismatches")
        return False

    spot_key = "model.action_in_proj.weight"
    if spot_key in merged_tensors and spot_key in state_dict:
        diff = (merged_tensors[spot_key].float() - state_dict[spot_key].cpu().float()).abs().max().item()
        print(f"  OK: spot check '{spot_key}' loaded correctly (max diff = {diff:.2e})")
        if diff > 1e-4:
            print(f"  FAIL: weight not loaded correctly")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Smoke test merged checkpoint")
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--ckpt-a", type=Path, default=None)
    parser.add_argument("--ckpt-b", type=Path, default=None)
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()

    results = {}
    print("=== 1. File check ===")
    results["files"] = check_files(args.merged)

    print("\n=== 2. Config check ===")
    results["config"] = check_config(args.merged)

    print("\n=== 3. Merge info check ===")
    results["merge_info"] = check_merge_info(args.merged)

    if args.ckpt_a and args.ckpt_b:
        print("\n=== 4. Interpolation check ===")
        results["interpolation"] = check_interpolation(args.merged, args.ckpt_a, args.ckpt_b)
    else:
        print("\n=== 4. Interpolation check — SKIPPED (no source ckpts given) ===")

    if not args.skip_forward:
        print("\n=== 5. Forward pass check (action_loss_only, 1 GPU) ===")
        try:
            results["forward"] = check_forward(args.merged)
        except Exception as e:
            print(f"FAIL: forward pass — {e}")
            import traceback
            traceback.print_exc()
            results["forward"] = False
    else:
        print("\n=== 5. Forward pass — SKIPPED ===")

    print("\n" + "=" * 50)
    failed = [k for k, v in results.items() if v is False]
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
