#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import random
import signal
import sys
import threading
import time

import torch


_GRAPH_CAPTURE_LOCK = threading.Lock()


def _mib(nbytes: int) -> float:
    return nbytes / (1024.0 * 1024.0)


def _gib(nbytes: int) -> float:
    return nbytes / (1024.0 * 1024.0 * 1024.0)


def _fill_bytes(device: torch.device, nbytes: int) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    remaining = max(0, nbytes) // 4
    # Several medium-sized slabs allow the target to be reduced later without
    # having to release the whole holder tensor at once.
    chunk = min(remaining, (512 * 1024 * 1024) // 4)
    while remaining > 0 and chunk > 0:
        try:
            t = torch.empty(chunk, device=device, dtype=torch.float32)
            t.fill_(1.0)
            tensors.append(t)
            remaining -= chunk
            chunk = min(remaining, (512 * 1024 * 1024) // 4)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            chunk //= 2
    return tensors


def _resize_holders(
    device: torch.device,
    holders: list[torch.Tensor],
    target_vram_ratio: float,
    reserve_mib: int,
) -> None:
    """Move resident allocations close to a new VRAM target."""
    free_b, total_b = torch.cuda.mem_get_info(device)
    reserve_b = reserve_mib * 1024 * 1024
    desired_used_b = max(0, int(total_b * target_vram_ratio) - reserve_b)
    current_used_b = total_b - free_b

    if current_used_b < desired_used_b:
        holders.extend(_fill_bytes(device, desired_used_b - current_used_b))
        return

    bytes_to_release = current_used_b - desired_used_b
    while holders and bytes_to_release > 0:
        tensor = holders.pop()
        bytes_to_release -= tensor.numel() * tensor.element_size()
        del tensor
    # Returning cached blocks helps nvidia-smi reflect the lower target.
    torch.cuda.empty_cache()


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


def _capture_matmul_graph(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    device: torch.device,
) -> torch.cuda.CUDAGraph | None:
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            for _ in range(4):
                torch.matmul(a, b, out=out)
            torch.cuda.synchronize(device)
            with torch.cuda.graph(graph, stream=stream):
                # Batch several GEMMs per replay to reduce host wakeups and CPU
                # overhead while the compute phase is active.
                for _ in range(4):
                    torch.matmul(a, b, out=out)
    except RuntimeError as exc:
        if "stream is capturing" not in str(exc).lower():
            raise
        # Some CUDA/PyTorch combinations reject graph capture when several
        # devices initialize concurrently. Keep the VRAM holder and fall back
        # to ordinary GEMMs so the utility still provides its intended load.
        torch.cuda.synchronize(device)
        print(f"[cuda:{device.index}] cudagraph unavailable; using regular matmul", flush=True)
        return None
    return graph


def _gpu_worker(
    device_idx: int,
    target_vram_ratio: float,
    reserve_mib: int,
    vram_min_ratio: float,
    vram_max_ratio: float,
    compute_min_ratio: float,
    compute_max_ratio: float,
    vram_change_interval_sec: float,
    compute_change_interval_sec: float,
    compute_cycle_sec: float,
    stop_event: threading.Event,
    ready_event: threading.Event,
    error_box: list[str],
) -> None:
    holders: list[torch.Tensor] = []
    try:
        device = torch.device(f"cuda:{device_idx}")
        torch.cuda.set_device(device)
        rng = random.Random(time.time_ns() ^ (os.getpid() << 16) ^ device_idx)
        free_b, total_b = torch.cuda.mem_get_info(device)
        current_vram_ratio = (total_b - free_b) / total_b
        if vram_min_ratio < vram_max_ratio and target_vram_ratio < current_vram_ratio:
            # If another job already occupies more than the first random
            # target, start at the upper bound so there is still room for the
            # GEMM buffers.  Later cycles can release only this script's slabs.
            target_vram_ratio = vram_max_ratio

        holders, a, b, out, n = _allocate_gpu(
            device, target_vram_ratio, reserve_mib, torch.bfloat16
        )
        # BLOCKING_SYNC makes Event.synchronize() yield the host thread instead
        # of busy-spinning while the GPU finishes the captured GEMMs.
        completion_event = torch.cuda.Event(blocking=True)
        # PyTorch/CUDA graph capture uses process-wide capture state in parts
        # of the runtime, so serialize only this one-time startup operation.
        with _GRAPH_CAPTURE_LOCK:
            graph = _capture_matmul_graph(a, b, out, device)

        free_b, total_b = torch.cuda.mem_get_info(device)
        used_ratio = (total_b - free_b) / total_b
        print(
            f"[cuda:{device_idx}] {torch.cuda.get_device_name(device_idx)}  "
            f"vram_used={used_ratio * 100:.1f}%  "
            f"allocated={_gib(torch.cuda.memory_allocated(device)):.2f} GiB  "
            f"matmul={n}x{n} bf16+cudagraph  "
            f"free_after={_mib(free_b):.0f} MiB  "
            f"slabs={len(holders)}",
            flush=True,
        )
        ready_event.set()

        now = time.monotonic()
        next_vram_change = now + rng.uniform(
            vram_change_interval_sec * 0.90, vram_change_interval_sec * 1.10
        )
        next_compute_change = now + rng.uniform(
            compute_change_interval_sec * 0.90, compute_change_interval_sec * 1.10
        )
        compute_ratio = rng.uniform(compute_min_ratio, compute_max_ratio)
        cycle_start = now
        cycle_end = cycle_start + compute_cycle_sec
        active_until = cycle_start + compute_cycle_sec * compute_ratio

        while not stop_event.is_set():
            now = time.monotonic()
            if now >= next_vram_change:
                target_vram_ratio = rng.uniform(vram_min_ratio, vram_max_ratio)
                _resize_holders(device, holders, target_vram_ratio, reserve_mib)
                next_vram_change = now + rng.uniform(
                    vram_change_interval_sec * 0.90, vram_change_interval_sec * 1.10
                )
                print(
                    f"[cuda:{device_idx}] random_vram_target  "
                    f"vram={target_vram_ratio * 100:.1f}%",
                    flush=True,
                )
                continue

            if now >= next_compute_change:
                compute_ratio = rng.uniform(compute_min_ratio, compute_max_ratio)
                next_compute_change = now + rng.uniform(
                    compute_change_interval_sec * 0.90, compute_change_interval_sec * 1.10
                )
                print(
                    f"[cuda:{device_idx}] random_compute_target  "
                    f"compute={compute_ratio * 100:.1f}%",
                    flush=True,
                )
                continue

            if now >= cycle_end:
                cycle_start = now
                cycle_end = cycle_start + compute_cycle_sec
                active_until = cycle_start + compute_cycle_sec * compute_ratio
                continue

            if now < active_until:
                if graph is None:
                    for _ in range(4):
                        torch.matmul(a, b, out=out)
                else:
                    graph.replay()
                completion_event.record()
                completion_event.synchronize()
            else:
                # Sleeping here releases the Python thread while the GPU is
                # intentionally idle; stop_event keeps signal handling prompt.
                stop_event.wait(
                    timeout=min(cycle_end, next_vram_change, next_compute_change) - now
                )

        torch.cuda.synchronize(device)
        _ = holders
    except Exception as exc:  # noqa: BLE001 - surface worker failures to main
        error_box.append(f"cuda:{device_idx}: {exc}")
        stop_event.set()
        ready_event.set()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Randomly hold 70-90% VRAM and GPU compute with minimal CPU usage.",
    )
    parser.add_argument(
        "--target-vram-ratio",
        type=float,
        default=None,
        help="Legacy fixed VRAM target; omit it to use the random 70-90%% range.",
    )
    parser.add_argument(
        "--vram-min-ratio",
        type=float,
        default=0.70,
        help="Minimum random VRAM target (default: 0.70).",
    )
    parser.add_argument(
        "--vram-max-ratio",
        type=float,
        default=0.90,
        help="Maximum random VRAM target (default: 0.90).",
    )
    parser.add_argument(
        "--compute-min-ratio",
        type=float,
        default=0.70,
        help="Minimum random compute duty cycle (default: 0.70).",
    )
    parser.add_argument(
        "--compute-max-ratio",
        type=float,
        default=0.90,
        help="Maximum random compute duty cycle (default: 0.90).",
    )
    parser.add_argument(
        "--vram-change-interval-min",
        type=float,
        default=15.0,
        help="Average interval between VRAM target changes in minutes (default: 15).",
    )
    parser.add_argument(
        "--compute-change-interval-min",
        type=float,
        default=18.0,
        help="Average interval between compute target changes in minutes (default: 18).",
    )
    parser.add_argument(
        "--compute-cycle-sec",
        type=float,
        default=10.0,
        help="Short duty-cycle window used to approximate compute utilization (default: 10).",
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

    if args.vram_change_interval_min <= 0:
        print("vram-change-interval-min must be > 0", file=sys.stderr)
        return 1
    if args.compute_change_interval_min <= 0:
        print("compute-change-interval-min must be > 0", file=sys.stderr)
        return 1
    if args.compute_cycle_sec <= 0:
        print("compute-cycle-sec must be > 0", file=sys.stderr)
        return 1
    if not 0.70 <= args.vram_min_ratio <= args.vram_max_ratio <= 0.90:
        print("VRAM random range must satisfy 0.70 <= min <= max <= 0.90", file=sys.stderr)
        return 1
    if not 0.70 <= args.compute_min_ratio <= args.compute_max_ratio <= 0.90:
        print("compute random range must satisfy 0.70 <= min <= max <= 0.90", file=sys.stderr)
        return 1
    if args.target_vram_ratio is not None and not 0.70 <= args.target_vram_ratio <= 0.90:
        print("target-vram-ratio must be between 0.70 and 0.90", file=sys.stderr)
        return 1

    if args.target_vram_ratio is not None:
        # Preserve the old option as a genuinely fixed-target compatibility
        # mode; the default path remains randomized.
        args.vram_min_ratio = args.target_vram_ratio
        args.vram_max_ratio = args.target_vram_ratio

    if args.target_vram_ratio is None:
        initial_vram_ratio = random.uniform(args.vram_min_ratio, args.vram_max_ratio)
    else:
        initial_vram_ratio = args.target_vram_ratio

    try:
        os.nice(19)
    except OSError:
        pass

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Another library may have initialized the inter-op pool already.
        pass

    n = torch.cuda.device_count()
    print(
        f"torch {torch.__version__}  devices={n}  "
        f"vram={args.vram_min_ratio * 100:.0f}-{args.vram_max_ratio * 100:.0f}%  "
        f"compute={args.compute_min_ratio * 100:.0f}-{args.compute_max_ratio * 100:.0f}%  "
        f"vram_change~{args.vram_change_interval_min:.0f}min  "
        f"compute_change~{args.compute_change_interval_min:.0f}min  "
        f"initial_vram={initial_vram_ratio * 100:.1f}%  low_cpu=on  processes=1",
        flush=True,
    )

    stop_event = threading.Event()
    error_box: list[str] = []
    workers: list[threading.Thread] = []
    ready_events: list[threading.Event] = []

    def _stop(signum, _frame):
        print(f"received signal {signum}, releasing GPU resources...", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    for i in range(n):
        ready = threading.Event()
        ready_events.append(ready)
        t = threading.Thread(
            target=_gpu_worker,
            args=(
                i,
                initial_vram_ratio,
                args.reserve_mib,
                args.vram_min_ratio,
                args.vram_max_ratio,
                args.compute_min_ratio,
                args.compute_max_ratio,
                args.vram_change_interval_min * 60.0,
                args.compute_change_interval_min * 60.0,
                args.compute_cycle_sec,
                stop_event,
                ready,
                error_box,
            ),
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

    print("GPU VRAM + compute held (low CPU). Ctrl+C or SIGTERM to stop.", flush=True)
    t0 = time.time()
    while not stop_event.wait(timeout=1.0):
        if args.hold_sec > 0 and (time.time() - t0) >= args.hold_sec:
            stop_event.set()
            break

    for t in workers:
        t.join(timeout=30)
    return 1 if error_box else 0


if __name__ == "__main__":
    raise SystemExit(main())
