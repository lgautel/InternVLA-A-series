#!/usr/bin/env python3
"""O5 acceptance: multi-episode render soak, the direct SIGABRT regression test.

The observed failure was: a forked child finishes ~4 episodes of a
`libero_spatial` task, then dies of SIGABRT inside the NVIDIA EGL stack before
it can write its result JSON, losing every episode of that task. fork-per-task
(eval3 B10) isolates tasks from each other but not episodes within a task, so
the loss is a whole task at a time.

This soak reproduces that shape of workload without a policy server: one
process, one env, `--n_episodes` reset+rollout cycles. Under `osmesa` there is
no EGL context, so this class of abort cannot occur; under `egl` this is the
reproducer.

The child is run under a wrapper process so a C-level abort is observed as a
signal rather than taking the harness down with it.

Usage:
    export LIBERO_HOME=/home/a26113/DATA/LIBERO-plus
    export LIBERO_CONFIG_PATH=/tmp/test_libero_config
    python evaluation/LIBERO2/test_osmesa_soak.py --render_backend osmesa \
        --n_episodes 8 --n_steps 40

Exit 0 = all episodes completed and the process exited cleanly (no signal).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _soak_child(backend: str, n_episodes: int, n_steps: int, seed: int, out_path: Path) -> int:
    """Run `n_episodes` reset+rollout cycles in a single process."""
    import numpy as np

    from evaluation.LIBERO2.orientation_contract import (
        UNPERTURBED_BDDL,
        setup_client_render_env,
    )

    resolved = setup_client_render_env(backend=backend)

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = Path(get_libero_path("bddl_files")) / UNPERTURBED_BDDL
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(seed)
    dummy = [0.0] * 6 + [-1.0]

    episodes: list[dict] = []
    for ep in range(n_episodes):
        t0 = time.perf_counter()
        obs = env.reset()
        checksum = 0.0
        for _ in range(n_steps):
            obs, _, _, _ = env.step(dummy)
            # Touch both camera buffers so the render path is actually exercised.
            checksum += float(np.asarray(obs["agentview_image"], dtype=np.float32).mean())
            checksum += float(np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.float32).mean())
        episodes.append(
            {
                "episode": ep,
                "steps": n_steps,
                "seconds": round(time.perf_counter() - t0, 2),
                "checksum": round(checksum, 3),
            }
        )
        # Flush per episode: a mid-soak abort then still leaves evidence of
        # exactly how far the process got.
        out_path.write_text(
            json.dumps(
                {"backend": resolved, "completed": len(episodes), "episodes": episodes}, indent=2
            )
        )

    env.close = lambda: None
    sys.stdout.flush()
    sys.stderr.flush()
    # os._exit so GL teardown cannot abort after a fully successful soak.
    os._exit(0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-episode render soak (SIGABRT regression)")
    parser.add_argument("--_soak_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_out", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--render_backend", type=str, default="osmesa", choices=["auto", "egl", "osmesa"]
    )
    parser.add_argument(
        "--n_episodes", type=int, default=8, help="Must exceed the ~4 episodes seen before abort"
    )
    parser.add_argument("--n_steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json_out", type=str, default="")
    args = parser.parse_args()

    if args._soak_child:
        return _soak_child(
            args.render_backend, args.n_episodes, args.n_steps, args.seed, Path(args._out)
        )

    out = Path(args.json_out or "/tmp/osmesa_soak_progress.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"completed": 0, "episodes": []}))

    print("=" * 68)
    print(
        f"O5 soak: backend={args.render_backend} "
        f"{args.n_episodes} episodes x {args.n_steps} steps"
    )
    print("=" * 68)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_soak_child",
        "--_out",
        str(out),
        "--render_backend",
        args.render_backend,
        "--n_episodes",
        str(args.n_episodes),
        "--n_steps",
        str(args.n_steps),
        "--seed",
        str(args.seed),
    ]
    env = dict(os.environ)
    env.pop("MUJOCO_GL", None)
    env.pop("PYOPENGL_PLATFORM", None)
    env["RENDER_BACKEND"] = args.render_backend
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (env.get("LIBERO_HOME", ""), str(REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)
    wall = time.perf_counter() - t0

    progress = json.loads(out.read_text())
    completed = int(progress.get("completed", 0))

    signaled = proc.returncode < 0
    sig_name = ""
    if signaled:
        try:
            sig_name = signal.Signals(-proc.returncode).name
        except ValueError:
            sig_name = f"signal_{-proc.returncode}"

    passed = (not signaled) and proc.returncode == 0 and completed == args.n_episodes
    result = {
        "backend": args.render_backend,
        "requested_episodes": args.n_episodes,
        "completed_episodes": completed,
        "returncode": proc.returncode,
        "signal": sig_name,
        "wall_seconds": round(wall, 1),
        "seconds_per_episode": round(wall / max(1, completed), 2),
        "episodes": progress.get("episodes", []),
        "passed": bool(passed),
    }

    for ep in result["episodes"]:
        print(f"  episode {ep['episode']}: {ep['steps']} steps in {ep['seconds']}s")
    if signaled:
        print(f"  CRASHED with {sig_name} after {completed}/{args.n_episodes} episodes")
        print(proc.stderr[-1500:])
    elif proc.returncode != 0:
        print(f"  child exited {proc.returncode} after {completed}/{args.n_episodes} episodes")
        print(proc.stderr[-1500:])

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
        print("wrote", args.json_out)

    print(
        f"\nOVERALL: {'PASS' if passed else 'FAIL'}  "
        f"({completed}/{args.n_episodes} episodes, {result['seconds_per_episode']}s each)"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
