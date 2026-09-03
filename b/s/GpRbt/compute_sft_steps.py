#!/usr/bin/env python3
"""Compute per-task SFT steps and checkpoint schedule from LeRobot info.json.

Training length is driven by configurable total epochs (SFT_EPOCHS), not a fixed
step count. Each task has different total_frames, so steps and save points differ.

effective_batch = n_gpus * per_gpu_batch * n_nodes
steps_per_epoch = ceil(total_frames / effective_batch)
steps = steps_per_epoch * epochs

Checkpoints: every (epochs / 4) epochs, plus always at the final step.
Lerobot: `step % save_freq == 0 or step == steps`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def checkpoint_epochs(total_epochs: int) -> list[int]:
    """Epoch indices (1-based) at which checkpoints are saved, always including total_epochs."""
    if total_epochs <= 0:
        raise ValueError("total_epochs must be > 0")
    every = max(total_epochs // 4, 1)
    epochs: list[int] = []
    e = every
    while e < total_epochs:
        epochs.append(e)
        e += every
    if total_epochs not in epochs:
        epochs.append(total_epochs)
    return epochs


def quarter_epoch_save_freq(
    steps: int, steps_per_epoch: int, epochs: int
) -> tuple[int, int, list[int], list[int]]:
    """Return (save_every_epochs, save_freq, save_steps, save_at_epochs)."""
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be > 0")
    save_every_epochs = max(epochs // 4, 1)
    save_freq = max(steps_per_epoch * save_every_epochs, 1)
    save_at_epochs = checkpoint_epochs(epochs)
    save_steps: list[int] = []
    for ep in save_at_epochs:
        if ep == epochs:
            save_steps.append(steps)
        else:
            save_steps.append(ep * steps_per_epoch)
    # Deduplicate while preserving order (e.g. small epoch counts)
    seen: set[int] = set()
    deduped: list[int] = []
    for s in save_steps:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    if steps not in deduped:
        deduped.append(steps)
    return save_every_epochs, save_freq, deduped, save_at_epochs


def compute_schedule(
    total_frames: int,
    epochs: int,
    n_gpus: int,
    batch_size: int,
    n_nodes: int = 1,
) -> dict:
    effective_batch = n_gpus * batch_size * n_nodes
    if effective_batch <= 0:
        raise ValueError("effective batch size must be > 0")
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    steps_per_epoch = math.ceil(total_frames / effective_batch)
    steps = steps_per_epoch * epochs
    save_every_epochs, save_freq, save_steps, save_at_epochs = quarter_epoch_save_freq(
        steps, steps_per_epoch, epochs
    )
    scheduler_warmup = min(1000, max(50, steps // 10))
    return {
        "total_frames": total_frames,
        "effective_batch": effective_batch,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "steps": steps,
        "save_every_epochs": save_every_epochs,
        "save_at_epochs": save_at_epochs,
        "save_freq": save_freq,
        "save_steps": save_steps,
        "scheduler_warmup_steps": scheduler_warmup,
        "scheduler_decay_steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute SFT steps and checkpoint schedule for one task dataset."
    )
    parser.add_argument("--info", required=True, help="Path to meta/info.json")
    parser.add_argument(
        "--epochs",
        type=int,
        default=76,
        help="Total training epochs (SFT_EPOCHS). Save points scale with this value.",
    )
    parser.add_argument("--n-gpus", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-nodes", type=int, default=1)
    parser.add_argument("--as-exports", action="store_true")
    args = parser.parse_args()

    info = json.loads(Path(args.info).read_text())
    result = compute_schedule(
        total_frames=int(info["total_frames"]),
        epochs=args.epochs,
        n_gpus=args.n_gpus,
        batch_size=args.batch_size,
        n_nodes=args.n_nodes,
    )

    if args.as_exports:
        print(f"SFT_EPOCHS={result['epochs']}")
        print(f"SFT_STEPS={result['steps']}")
        print(f"SFT_SAVE_FREQ={result['save_freq']}")
        print(f"SFT_SAVE_EVERY_EPOCHS={result['save_every_epochs']}")
        print(f"SFT_SAVE_AT_EPOCHS={','.join(str(x) for x in result['save_at_epochs'])}")
        print(f"SFT_SAVE_STEPS={','.join(str(x) for x in result['save_steps'])}")
        print(f"SFT_SCHEDULER_WARMUP={result['scheduler_warmup_steps']}")
        print(f"SFT_STEPS_PER_EPOCH={result['steps_per_epoch']}")
        print(f"SFT_TOTAL_FRAMES={result['total_frames']}")
        print(f"SFT_EFFECTIVE_BATCH={result['effective_batch']}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
