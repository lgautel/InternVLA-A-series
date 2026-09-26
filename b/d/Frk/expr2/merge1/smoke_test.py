#!/usr/bin/env python3
"""
Smoke test for merged checkpoint.

Verifies that the merged checkpoint can be loaded by:
1. Transformers AutoModel/AutoConfig (VLM backbone)
2. LeRobot policy loading
3. Forward pass produces valid outputs

Usage:
    python smoke_test.py --checkpoint /path/to/merged/checkpoint
"""
import argparse
import sys
from pathlib import Path

import torch
import numpy as np


def test_transformers_loading(ckpt_path: Path) -> bool:
    """Test 1: Load with Transformers AutoModel."""
    print("\n=== Test 1: Transformers AutoModel/AutoConfig ===")
    try:
        from transformers import AutoConfig, AutoModel

        print(f"Loading config from {ckpt_path}")
        config = AutoConfig.from_pretrained(str(ckpt_path), trust_remote_code=True)
        print(f"✓ Config loaded: {config.__class__.__name__}")

        print(f"Loading model weights")
        model = AutoModel.from_pretrained(
            str(ckpt_path),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        print(f"✓ Model loaded: {model.__class__.__name__}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        del model
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lerobot_policy_loading(ckpt_path: Path) -> bool:
    """Test 2: Load with LeRobot policy factory."""
    print("\n=== Test 2: LeRobot Policy Loading ===")
    try:
        # Import LeRobot internals
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
        from lerobot.policies.factory import make_policy_from_checkpoint
        from lerobot.configs.default import DatasetConfig

        print(f"Loading policy from {ckpt_path}")
        policy = make_policy_from_checkpoint(str(ckpt_path), device="cpu")
        print(f"✓ Policy loaded: {policy.__class__.__name__}")

        # Check key attributes
        if hasattr(policy, "config"):
            print(f"  Policy type: {policy.config.type}")
            print(f"  Action mode: {getattr(policy.config, 'action_mode', 'N/A')}")
            print(f"  Action loss only: {getattr(policy.config, 'action_loss_only', 'N/A')}")

        del policy
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass(ckpt_path: Path) -> bool:
    """Test 3: Run a dummy forward pass."""
    print("\n=== Test 3: Forward Pass ===")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
        from lerobot.policies.factory import make_policy_from_checkpoint

        print("Loading policy for forward test")
        policy = make_policy_from_checkpoint(str(ckpt_path), device="cpu")
        policy.eval()

        # Create dummy batch (minimal viable input)
        batch_size = 2
        dummy_batch = {
            "observation.images.top": torch.rand(batch_size, 3, 224, 224, dtype=torch.float32),
            "observation.state": torch.rand(batch_size, 7, dtype=torch.float32),
            "action": torch.rand(batch_size, 7, dtype=torch.float32),
        }

        # Add chunk indices if policy expects them
        if hasattr(policy.config, "chunk_size"):
            chunk_size = policy.config.chunk_size
            dummy_batch["action_chunk_indices"] = torch.zeros(batch_size, chunk_size, dtype=torch.long)

        print(f"Running forward pass (batch_size={batch_size})")
        with torch.no_grad():
            output = policy.forward(dummy_batch)

        if isinstance(output, tuple):
            loss, output_dict = output
            print(f"✓ Forward pass succeeded")
            print(f"  Loss: {loss.item():.4f}")
            if output_dict:
                print(f"  Output keys: {list(output_dict.keys())}")
        else:
            print(f"✓ Forward pass succeeded")
            print(f"  Output type: {type(output)}")

        del policy, dummy_batch, output
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Smoke test for merged checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to merged checkpoint")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        return 1

    print(f"Testing checkpoint: {args.checkpoint}")

    results = {
        "Transformers loading": test_transformers_loading(args.checkpoint),
        "LeRobot policy loading": test_lerobot_policy_loading(args.checkpoint),
        "Forward pass": test_forward_pass(args.checkpoint),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {test_name}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
