"""LIBERO-plus2 evaluation script.

Fixes vs LIBERO-plus/eval_libero_plus.py:
  - B6: per-task try-except (single task crash doesn't kill entire shard)
  - B7: NO top-level import imageio (delayed to function body to avoid SIGABRT)
  - B1: uses LIBERO2/model2libero_interface.py (gripper_convention fix)
  - B10: fork-per-task subprocess isolation (MuJoCo EGL can't create two
         contexts in one process without SIGABRT on the second)
  - NEW: keypoint extraction via StandaloneFK + KeypointHistory
  - NEW: --save_actions for action sequence recording
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

# NOTE: LiberoModelClient is imported LAZILY inside the forked child process.
# Top-level import triggers `import mujoco` -> EGL init, which makes fork() unusable.

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


def _get_libero_env(task, resolution: int, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def evaluate_task(
    task,
    initial_states,
    args: argparse.Namespace,
    client: LiberoModelClient,
    max_steps: int,
    video_dir: Path,
    video_tag: str,
    actions_dir: Path | None = None,
    task_id: int | None = None,
    failures_dir: Path | None = None,
    _partial_results: dict | None = None,
):
    """Run all episodes for one LIBERO-plus task; returns (successes, task_desc, action_logs)."""
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    client.bind_env(env)
    n_episodes = min(args.num_trials_per_task, len(initial_states))
    successes: list[bool] = []
    action_logs: list[dict] = []
    if _partial_results is not None:
        _partial_results["task_desc"] = task_description
    _repro_base = {
        "task_suite": args.task_suite_name,
        "task_id": task_id,
        "task_name": video_tag,
        "task_desc": "",
        "seed": args.seed,
        "replan_steps": args.replan_steps,
        "wait_steps": args.num_steps_wait,
        "rotate_images": _rotate_images_from_args(args),
        "gripper_convention": args.gripper_convention,
    }

    for episode_idx in tqdm(range(n_episodes), desc=f"task: {video_tag}", leave=False):
        client.reset(task_description)
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])

        replay_images: list[np.ndarray] = []
        episode_actions: list[np.ndarray] = []
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
            episode_actions.append(action.copy())
            replay_images.append(client._maybe_rotate(obs["agentview_image"]))
            client.push_keypoint()
            obs, _, done, _ = env.step(action.tolist())
            if done:
                break

        successes.append(bool(done))
        if _partial_results is not None:
            _partial_results["successes"].append(bool(done))

        # B7 FIX: delayed import imageio (NOT at top level)
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
                out_path = video_dir / f"rollout_{video_tag}_episode{episode_idx}_failure.mp4"
                imageio.mimwrite(out_path, [np.asarray(x) for x in replay_images], fps=10)
        elif args.save_videos and replay_images:
            # Legacy: save all videos (success path)
            import imageio
            out_path = video_dir / f"rollout_{video_tag}_episode{episode_idx}_success.mp4"
            imageio.mimwrite(out_path, [np.asarray(x) for x in replay_images], fps=10)

        if args.save_actions and episode_actions:
            action_log = {
                "task": video_tag,
                "episode": episode_idx,
                "seed": args.seed,
                "success": bool(done),
                "num_steps": len(episode_actions),
            }
            if actions_dir is not None:
                npz_path = actions_dir / f"{video_tag}_ep{episode_idx}.npz"
                np.savez_compressed(
                    npz_path,
                    actions=np.array(episode_actions, dtype=np.float32),
                    success=bool(done),
                    seed=args.seed,
                    task=video_tag,
                )
            action_logs.append(action_log)

    # Don't close env — runs in forked child that exits via os._exit()
    return successes, task_description, action_logs


def _run_task_in_subprocess(task_id, args, max_steps,
                            video_dir, video_tag, actions_dir, result_path, failures_dir=None):
    """B10 FIX: fork a child for each task to isolate MuJoCo EGL contexts.

    ALL LIBERO/robosuite/mujoco imports happen inside the child so the parent
    process never initialises an EGL context (which would be inherited and
    break eglCreateContext in the child).
    """
    n_ep = args.num_trials_per_task
    pid = os.fork()
    if pid == 0:
        # ─── CHILD PROCESS ───
        _partial = {"successes": [], "task_desc": video_tag}

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
                host=args.host, port=args.port,
                rotate_images=_rotate_images_from_args(args),
                gripper_convention=args.gripper_convention,
                replan_steps=args.replan_steps,
                enable_keypoints=args.enable_keypoints,
            )

            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[args.task_suite_name]()
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)

            successes, task_desc, action_logs = evaluate_task(
                task, initial_states, args, child_client, max_steps,
                video_dir, video_tag, actions_dir=actions_dir,
                task_id=task_id, failures_dir=failures_dir,
                _partial_results=_partial,
            )
            result = {
                "successes": [bool(s) for s in successes],
                "task_desc": task_desc,
                "action_logs": action_logs,
            }
        except Exception as e:
            n_done = len(_partial["successes"])
            result = {
                "successes": _partial["successes"] + [False] * (n_ep - n_done),
                "task_desc": _partial.get("task_desc", video_tag),
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        with open(result_path, "w") as f:
            json.dump(result, f)
        os._exit(0)

    # ─── PARENT PROCESS ───
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


def _load_id2category(task_classification_path: str, suite: str) -> tuple[dict[int, tuple[str, str]], dict[str, dict]]:
    with open(task_classification_path, encoding="utf-8") as f:
        mapping = json.load(f)
    if suite not in mapping:
        raise KeyError(
            f"Suite '{suite}' not in task_classification.json. Available: {list(mapping.keys())}"
        )
    id2category: dict[int, tuple[str, str]] = {}
    disturb_res: dict[str, dict] = {}
    for item in mapping[suite]:
        category = item["category"]
        id2category[int(item["id"])] = (category, item["name"])
        disturb_res.setdefault(category, {"total_count": 0, "success_count": 0})
    return id2category, disturb_res


def evaluate_policy(args: argparse.Namespace) -> dict:
    # Read task metadata from JSON only — no LIBERO/robosuite/mujoco imports
    # so the parent process never initialises EGL.
    id2category, disturb_res = _load_id2category(args.task_classification_path, args.task_suite_name)
    n_tasks_in_suite = max(int(item_id) for item_id in id2category)

    start_idx = max(0, args.start_idx) if args.start_idx >= 0 else 0
    end_idx = args.end_idx if args.end_idx >= 0 else n_tasks_in_suite
    end_idx = min(end_idx, n_tasks_in_suite)
    if start_idx >= end_idx:
        raise ValueError(f"Empty shard: start_idx={start_idx} end_idx={end_idx} (n_tasks={n_tasks_in_suite})")

    max_steps = args.max_steps_override if args.max_steps_override > 0 else TASK_SUITE_MAX_STEPS[args.task_suite_name]

    LOGGER.info(
        "Suite=%s | shard [%d, %d) of %d tasks | max_steps=%d | trials/task=%d",
        args.task_suite_name, start_idx, end_idx, n_tasks_in_suite, max_steps, args.num_trials_per_task,
    )

    eval_log_dir = Path(args.eval_log_dir)
    video_dir = eval_log_dir / "videos" / args.task_suite_name
    if args.save_videos or getattr(args, "save_failure_videos", False):
        video_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = eval_log_dir / "failures" / args.task_suite_name
    failures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = eval_log_dir / "logs" / args.task_suite_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    actions_dir = None
    if args.save_actions:
        actions_dir = eval_log_dir / "actions" / args.task_suite_name
        actions_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    total_successes = 0
    failed_tasks: list[dict] = []

    for task_id in tqdm(range(start_idx, end_idx), desc=f"{args.task_suite_name}[{start_idx}:{end_idx}]"):
        category, clean_name = id2category[task_id + 1]

        if args.categories and category not in args.categories:
            continue

        # B10 FIX: run each task in a forked subprocess to isolate EGL contexts
        result_path = str(eval_log_dir / f".task_result_{task_id}.json")
        result = _run_task_in_subprocess(
            task_id, args, max_steps,
            video_dir, clean_name, actions_dir, result_path, failures_dir=failures_dir,
        )

        successes = [bool(s) for s in result.get("successes", [False])]
        task_desc = result.get("task_desc", clean_name)

        if result.get("error"):
            LOGGER.error(
                "task_id=%d [%s] '%s' CRASHED: %s",
                task_id, category, clean_name, result["error"],
            )
            if result.get("traceback"):
                LOGGER.error("Traceback:\n%s", result["traceback"])
            failed_tasks.append({
                "task_id": task_id,
                "category": category,
                "name": clean_name,
                "error": result["error"],
                "seed": args.seed,
            })

        n_succ = int(sum(successes))
        disturb_res[category]["total_count"] += len(successes)
        disturb_res[category]["success_count"] += n_succ
        total_episodes += len(successes)
        total_successes += n_succ
        LOGGER.info(
            "task_id=%d [%s] '%s' -> %d/%d (running total %d/%d = %.2f%%)",
            task_id, category, task_desc, n_succ, len(successes),
            total_successes, total_episodes, 100.0 * total_successes / max(total_episodes, 1),
        )

    shard_summary = {
        "task_suite": args.task_suite_name,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_success_rate": total_successes / max(total_episodes, 1),
        "per_category": disturb_res,
        "failed_tasks": failed_tasks,
    }

    out_json = logs_dir / f"{start_idx}_to_{end_idx}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(disturb_res, f, indent=2)
    LOGGER.info("Saved per-shard category results to %s", out_json)

    if failed_tasks:
        fail_json = logs_dir / f"{start_idx}_to_{end_idx}_failures.json"
        with open(fail_json, "w", encoding="utf-8") as f:
            json.dump(failed_tasks, f, indent=2)
        LOGGER.warning("Saved %d failed task details to %s", len(failed_tasks), fail_json)

    return shard_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LIBERO-plus2 evaluation with websocket policy server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5694)
    parser.add_argument(
        "--task_suite_name",
        type=str,
        default="libero_goal",
        choices=list(TASK_SUITE_MAX_STEPS.keys()),
    )
    parser.add_argument("--num_trials_per_task", type=int, default=1)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan_steps", type=int, default=8)
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--task_classification_path", type=str, required=True)
    parser.add_argument("--max_steps_override", type=int, default=-1)
    parser.add_argument("--eval_log_dir", type=str, default="outputs/sim_eval/libero_plus")
    parser.add_argument("--save_videos", action=argparse.BooleanOptionalAction, default=False,
                        help="Save videos for ALL episodes (legacy; wastes disk). Default off.")
    parser.add_argument("--save_failure_videos", action=argparse.BooleanOptionalAction, default=False,
                        help="Save MP4 only for failed episodes. Failure reproduction JSON is always saved.")
    parser.add_argument("--save_actions", action=argparse.BooleanOptionalAction, default=False)
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
        "--gripper_convention",
        type=str,
        default="libero_native",
        choices=["libero_native", "openvla", "auto"],
    )
    parser.add_argument("--enable_keypoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--categories", type=str, nargs="*", default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    np.random.seed(args.seed)

    eval_log_dir = Path(args.eval_log_dir)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    summary = evaluate_policy(args)
    LOGGER.info(
        "LIBERO-plus2 eval done. Suite=%s shard=[%d,%d) SR=%.4f (%d/%d)",
        summary["task_suite"],
        summary["start_idx"],
        summary["end_idx"],
        summary["overall_success_rate"],
        summary["total_successes"],
        summary["total_episodes"],
    )
    if summary.get("failed_tasks"):
        LOGGER.warning("%d tasks crashed (see failures JSON)", len(summary["failed_tasks"]))
    LOGGER.info(colored(f"Saved results under {args.eval_log_dir}", "green"))


if __name__ == "__main__":
    # Backend is chosen before any fork so the child inherits a consistent
    # MUJOCO_GL/PYOPENGL_PLATFORM pair. osmesa removes the EGL SIGABRT class.
    from evaluation.LIBERO2.render_backend import describe, setup_render_env

    _backend = setup_render_env(os.environ.get("RENDER_BACKEND"))
    print(f"[render] backend={_backend} {describe(_backend)}", file=sys.stderr)
    main()
