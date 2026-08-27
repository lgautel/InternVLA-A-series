#!/usr/bin/env python3
"""Hold >=90% VRAM and sustain >=80% GPU compute on every visible CUDA device.

Designed for 8x NVIDIA A800-SXM4-80GB. Each GPU runs in its own thread:
1) reserve bf16 GEMM buffers sized for high SM utilization;
2) fill the remaining budget with resident tensors until total VRAM >= target;
3) run continuous GEMM on the reserved buffers.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time

import torch


def _mib(nbytes: int) -> float:
    return nbytes / (1024.0 * 1024.0)


def _gib(nbytes: int) -> float:
    return nbytes / (1024.0 * 1024.0 * 1024.0)


def _fill_bytes(device: torch.device, nbytes: int) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    remaining = nbytes // 4
    chunk = remaining
    while remaining > 0 and chunk > 0:
        try:
            t = torch.empty(chunk, device=device, dtype=torch.float32)
            t.fill_(1.0)
            tensors.append(t)
            remaining -= chunk
            chunk = remaining
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            chunk //= 2
    return tensors


def _allocate_gpu(
    device: torch.device,
    target_vram_ratio: float,
    reserve_mib: int,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, int]:
    torch.cuda.synchronize(device)
    free_b, total_b = torch.cuda.mem_get_info(device)
    reserve_b = reserve_mib * 1024 * 1024
    elem_size = torch.tensor([], dtype=dtype).element_size()
    target_used_b = int(total_b * target_vram_ratio)
    already_used_b = total_b - free_b
    budget_b = max(0, target_used_b - already_used_b - reserve_b)

    # Search large->small GEMM side so we can still meet the VRAM target afterward.
    for n in range(32768, 4095, -256):
        compute_b = 3 * n * n * elem_size
        holder_b = budget_b - compute_b
        if holder_b < 128 * 1024 * 1024:
            continue
        a = b = out = None
        holders: list[torch.Tensor] = []
        try:
            a = torch.randn(n, n, device=device, dtype=dtype)
            b = torch.randn(n, n, device=device, dtype=dtype)
            out = torch.empty(n, n, device=device, dtype=dtype)
            holders = _fill_bytes(device, holder_b)
            return holders, a, b, out, n
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            del a, b, out, holders
            torch.cuda.empty_cache()
            continue

    raise RuntimeError(f"failed to allocate compute+VRAM budget on {device}")


def _gpu_worker(
    device_idx: int,
    target_vram_ratio: float,
    reserve_mib: int,
    stop_event: threading.Event,
    ready_event: threading.Event,
    error_box: list[str],
) -> None:
    try:
        device = torch.device(f"cuda:{device_idx}")
        torch.cuda.set_device(device)

        holders, a, b, out, n = _allocate_gpu(device, target_vram_ratio, reserve_mib, torch.bfloat16)

        for _ in range(8):
            torch.matmul(a, b, out=out)
        torch.cuda.synchronize(device)

        free_b, total_b = torch.cuda.mem_get_info(device)
        used_ratio = (total_b - free_b) / total_b
        print(
            f"[cuda:{device_idx}] {torch.cuda.get_device_name(device_idx)}  "
            f"vram_used={used_ratio * 100:.1f}%  "
            f"allocated={_gib(torch.cuda.memory_allocated(device)):.2f} GiB  "
            f"matmul={n}x{n} bf16  "
            f"free_after={_mib(free_b):.0f} MiB  "
            f"slabs={len(holders)}",
            flush=True,
        )
        ready_event.set()

        while not stop_event.is_set():
            torch.matmul(a, b, out=out)

        torch.cuda.synchronize(device)
        _ = holders
    except Exception as exc:  # noqa: BLE001 - surface worker failures to main
        error_box.append(f"cuda:{device_idx}: {exc}")
        ready_event.set()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill >=90% VRAM and sustain >=80% GPU compute on all visible CUDA GPUs.",
    )
    parser.add_argument(
        "--target-vram-ratio",
        type=float,
        default=0.92,
        help="Target fraction of total VRAM to occupy (default: 0.92).",
    )
    parser.add_argument(
        "--reserve-mib",
        type=int,
        default=512,
        help="Per-GPU headroom reserved for driver/context.",
    )
    parser.add_argument(
        "--hold-sec",
        type=float,
        default=0.0,
        help="Run time in seconds; 0 means until SIGINT/SIGTERM.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available.", file=sys.stderr)
        return 1

    if args.target_vram_ratio < 0.90:
        print("target-vram-ratio must be >= 0.90", file=sys.stderr)
        return 1

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    n = torch.cuda.device_count()
    print(
        f"torch {torch.__version__}  devices={n}  "
        f"target_vram>={args.target_vram_ratio * 100:.0f}%  target_compute>=80%",
        flush=True,
    )

    stop_event = threading.Event()
    error_box: list[str] = []
    workers: list[threading.Thread] = []
    ready_events: list[threading.Event] = []

    def _stop(signum, _frame):
        print(f"received signal {signum}, stopping workers...", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    for i in range(n):
        ready = threading.Event()
        ready_events.append(ready)
        t = threading.Thread(
            target=_gpu_worker,
            args=(i, args.target_vram_ratio, args.reserve_mib, stop_event, ready, error_box),
            name=f"gpu-{i}",
            daemon=True,
        )
        t.start()
        workers.append(t)

    if not all(ev.wait(timeout=180) for ev in ready_events):
        stop_event.set()
        print("timeout waiting for GPU workers", file=sys.stderr)
        return 1

    if error_box:
        stop_event.set()
        for msg in error_box:
            print(f"error: {msg}", file=sys.stderr)
        return 1

    print("VRAM + compute load running. Ctrl+C or SIGTERM to stop.", flush=True)
    t0 = time.time()
    while not stop_event.is_set():
        if args.hold_sec > 0 and (time.time() - t0) >= args.hold_sec:
            stop_event.set()
            break
        time.sleep(1.0)

    for t in workers:
        t.join(timeout=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
