#!/usr/bin/env python3
"""Standard LIBERO evaluation with all LIBERO2 fixes.

Changes from evaluation/LIBERO/eval_libero_server_client.py:
  - Import LIBERO2/model2libero_interface.py (B1 gripper fix, keypoints)
  - B10: fork-per-task subprocess isolation
  - B7: delayed imageio import
  - F1: rotate_images=False by default (--rotate_images to opt in)

Usage:
    python evaluation/LIBERO2/eval_libero_std.py \
        --host 127.0.0.1 --port 5784 \
        --task_suite_name libero_spatial \
        --num_trials_per_task 50 \
        --eval_log_dir <output_dir> \
        --gripper_convention libero_native
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import signal as _signal
import sys
import traceback
from pathlib import Path

import numpy as np
from termcolor import colored
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LOGGER = logging.getLogger(__name__)


def _rotate_images_from_args(args: argparse.Namespace) -> bool:
    if getattr(args, "no_rotate_images", False):
        return False
    return bool(getattr(args, "rotate_images", False))


TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# Parent process must not import libero/mujoco (B10: EGL in parent breaks
# eglCreateContext in the forked child). Task counts are fixed by the
# official LIBERO suites; the child still loads benchmark for BDDL/init.
TASK_SUITE_N_TASKS: dict[str, int] = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}


def _get_libero_env(task, resolution: int, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def evaluate_task(task, initial_states, args, client, max_steps, video_dir,
                   task_id=None, failures_dir=None, _partial_results=None):
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    client.bind_env(env)
    n_episodes = min(args.num_trials_per_task, len(initial_states))
    successes: list[bool] = []
    if _partial_results is not None:
        _partial_results["task_desc"] = task_description
    _repro_base = {
        "task_suite": args.task_suite_name,
        "task_id": task_id,
        "task_desc": "",
        "seed": args.seed,
        "replan_steps": args.replan_steps,
        "wait_steps": args.num_steps_wait,
        "rotate_images": _rotate_images_from_args(args),
        "gripper_convention": args.gripper_convention,
    }

    for episode_idx in tqdm(
        range(n_episodes), desc=f"task: {task_description}", leave=False
    ):
        client.reset(task_description)
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])

        replay_images: list[np.ndarray] = []
        done = False
        for t in range(max_steps + args.num_steps_wait):
            # Push the *current* obs into history *before* env.step so his_kpts
            # matches training: Extract3DKeypointTransformFn keeps offsets
            # [-H, ..., -1] in his_kpts and the current frame in kpt_t.
            if t < args.num_steps_wait:
                client.push_keypoint()
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                continue

            action = client.step(obs, task_description)
            replay_images.append(client._maybe_rotate(obs["agentview_image"]))
            client.push_keypoint()
            obs, _, done, _ = env.step(action.tolist())
            if done:
                break

        successes.append(bool(done))
        if _partial_results is not None:
            _partial_results["successes"].append(bool(done))

        if not done:
            # Save reproducibility info for failures (allows video re-generation later)
            if failures_dir is not None:
                failures_dir.mkdir(parents=True, exist_ok=True)
                failure_info = {**_repro_base,
                    "task_desc": task_description,
                    "episode_idx": episode_idx,
                    "initial_state_idx": episode_idx,
                }
                fname = failures_dir / f"task{task_id}_ep{episode_idx}.json"
                with open(fname, "w") as _f:
                    json.dump(failure_info, _f, indent=2)
            # Save failure video when requested
            if getattr(args, "save_failure_videos", False) and replay_images:
                import imageio
                if video_dir is not None:
                    video_dir.mkdir(parents=True, exist_ok=True)
                out = video_dir / f"{task_description}_ep{episode_idx}_failure.mp4"
                imageio.mimwrite(out, [np.asarray(x) for x in replay_images], fps=10)
        elif args.save_videos and replay_images:
            # Legacy: save all videos (success path)
            import imageio
            out = video_dir / f"{task_description}_ep{episode_idx}_success.mp4"
            imageio.mimwrite(out, [np.asarray(x) for x in replay_images], fps=10)

    return successes, task_description


def _run_task_in_subprocess(task_id, args, max_steps, video_dir, result_path, failures_dir=None):
    n_ep = args.num_trials_per_task
    pid = os.fork()
    if pid == 0:
        _partial = {"successes": [], "task_desc": f"task_{task_id}"}

        def _sigabrt_handler(signum, frame):
            n_done = len(_partial["successes"])
            _partial["successes"].extend([False] * (n_ep - n_done))
            _partial["error"] = f"SIGABRT after {n_done}/{n_ep} episodes"
            with open(result_path, "w") as _fh:
                json.dump(_partial, _fh)
            os._exit(1)

        _signal.signal(_signal.SIGABRT, _sigabrt_handler)

        try:
            from evaluation.LIBERO2.model2libero_interface import LiberoModelClient
            from libero.libero import benchmark

            child_client = LiberoModelClient(
                host=args.host,
                port=args.port,
                rotate_images=_rotate_images_from_args(args),
                gripper_convention=args.gripper_convention,
                replan_steps=args.replan_steps,
                enable_keypoints=args.enable_keypoints,
            )

            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[args.task_suite_name]()
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)

            successes, task_desc = evaluate_task(
                task, initial_states, args, child_client, max_steps, video_dir,
                task_id=task_id, failures_dir=failures_dir,
                _partial_results=_partial,
            )
            result = {
                "successes": [bool(s) for s in successes],
                "task_desc": task_desc,
            }
        except Exception as e:
            n_done = len(_partial["successes"])
            result = {
                "successes": _partial["successes"] + [False] * (n_ep - n_done),
                "task_desc": _partial.get("task_desc", f"task_{task_id}"),
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        with open(result_path, "w") as f:
            json.dump(result, f)
        os._exit(0)

    _, status = os.waitpid(pid, 0)

    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)
        os.unlink(result_path)
        return result

    if os.WIFSIGNALED(status):
        sig = os.WTERMSIG(status)
        try:
            sig_name = _signal.Signals(sig).name
        except (ValueError, AttributeError):
            sig_name = f"signal_{sig}"
        error = f"Killed by {sig_name} (signal {sig})"
    else:
        error = f"Exit code {os.WEXITSTATUS(status)}"

    return {"successes": [False] * n_ep, "error": error}


def evaluate_policy(args):
    # Read task count from the local table only — no LIBERO/robosuite/mujoco
    # imports so the parent process never initialises EGL (plus2 pattern).
    if args.task_suite_name not in TASK_SUITE_N_TASKS:
        raise KeyError(
            f"Unknown suite '{args.task_suite_name}'. "
            f"Available: {list(TASK_SUITE_N_TASKS)}"
        )
    n_tasks = TASK_SUITE_N_TASKS[args.task_suite_name]

    start_idx = max(0, args.start_idx) if args.start_idx >= 0 else 0
    end_idx = args.end_idx if args.end_idx >= 0 else n_tasks
    end_idx = min(end_idx, n_tasks)

    max_steps = TASK_SUITE_MAX_STEPS.get(args.task_suite_name, 300)
    LOGGER.info(
        "Suite=%s | tasks [%d, %d) of %d | max_steps=%d | trials/task=%d",
        args.task_suite_name, start_idx, end_idx, n_tasks, max_steps,
        args.num_trials_per_task,
    )

    eval_log_dir = Path(args.eval_log_dir)
    video_dir = eval_log_dir / "videos" / args.task_suite_name
    if args.save_videos or getattr(args, "save_failure_videos", False):
        video_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = eval_log_dir / "failures" / args.task_suite_name
    failures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = eval_log_dir / "logs" / args.task_suite_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    total_successes = 0
    per_task_results = []

    for task_id in tqdm(
        range(start_idx, end_idx), desc=f"{args.task_suite_name}"
    ):
        result_path = str(eval_log_dir / f".task_result_{task_id}.json")
        result = _run_task_in_subprocess(
            task_id, args, max_steps, video_dir, result_path, failures_dir=failures_dir,
        )

        successes = [bool(s) for s in result.get("successes", [False])]
        task_desc = result.get("task_desc", f"task_{task_id}")

        if result.get("error"):
            LOGGER.error(
                "task_id=%d '%s' CRASHED: %s",
                task_id, task_desc, result["error"],
            )

        n_succ = sum(successes)
        total_episodes += len(successes)
        total_successes += n_succ
        sr = n_succ / max(len(successes), 1)
        per_task_results.append({
            "task_id": task_id,
            "task_desc": task_desc,
            "successes": n_succ,
            "total": len(successes),
            "sr": sr,
            "error": result.get("error"),
        })
        LOGGER.info(
            "task_id=%d '%s' -> %d/%d (running SR %.2f%%)",
            task_id, task_desc, n_succ, len(successes),
            100.0 * total_successes / max(total_episodes, 1),
        )

    summary = {
        "task_suite": args.task_suite_name,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_sr": total_successes / max(total_episodes, 1),
        "per_task": per_task_results,
    }

    out_json = logs_dir / f"std_{start_idx}_to_{end_idx}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("Saved results to %s", out_json)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Standard LIBERO eval with LIBERO2 fixes")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5784)
    parser.add_argument(
        "--task_suite_name", type=str, default="libero_spatial",
        choices=list(TASK_SUITE_MAX_STEPS.keys()),
    )
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan_steps", type=int, default=8)
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--eval_log_dir", type=str, default="outputs/sim_eval/libero_std")
    parser.add_argument("--save_videos", action=argparse.BooleanOptionalAction, default=False,
                        help="Save videos for ALL episodes (legacy; wastes disk). Default off.")
    parser.add_argument("--save_failure_videos", action=argparse.BooleanOptionalAction, default=False,
                        help="Save MP4 only for failed episodes. Failure reproduction JSON is always saved.")
    parser.add_argument(
        "--rotate_images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="180 deg flip of agentview + wrist. Default false (raw training frames).",
    )
    parser.add_argument(
        "--no_rotate_images",
        action="store_true",
        help="Deprecated alias forcing rotate_images=False.",
    )
    parser.add_argument(
        "--gripper_convention", type=str, default="libero_native",
        choices=["libero_native", "openvla", "auto"],
    )
    parser.add_argument("--enable_keypoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    np.random.seed(args.seed)
    Path(args.eval_log_dir).mkdir(parents=True, exist_ok=True)

    summary = evaluate_policy(args)
    sr_pct = 100.0 * summary["overall_sr"]
    LOGGER.info(
        "Standard LIBERO eval done. Suite=%s SR=%.2f%% (%d/%d)",
        summary["task_suite"], sr_pct,
        summary["total_successes"], summary["total_episodes"],
    )
    print(colored(
        f"\n{'PASS' if sr_pct >= 50 else 'FAIL'}: {summary['task_suite']} SR = {sr_pct:.2f}%",
        "green" if sr_pct >= 50 else "red",
    ))


if __name__ == "__main__":
    # Backend is chosen before any fork so the child inherits a consistent
    # MUJOCO_GL/PYOPENGL_PLATFORM pair. osmesa removes the EGL SIGABRT class.
    from evaluation.LIBERO2.render_backend import describe, setup_render_env

    _backend = setup_render_env(os.environ.get("RENDER_BACKEND"))
    print(f"[render] backend={_backend} {describe(_backend)}", file=sys.stderr)
    main()
