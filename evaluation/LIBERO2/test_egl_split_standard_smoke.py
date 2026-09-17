#!/usr/bin/env python3
"""Real-model multi-episode smoke for the split EGL topology.

This is the direct regression test for the reported failure:

  two model servers: GPU 0/1
  six LIBERO clients: GPU 2..7, three clients per server
  standard LIBERO spatial task 0..5
  five episodes per task in one forked task child

Unlike the LIBERO-plus smoke (one perturbation episode per task), this keeps
five episodes inside each child and therefore exercises the old "SIGABRT after
~4 episodes" failure boundary.  It deliberately uses the standard backend,
because the current optimized backend does not accept the keypoint payload
(`his_kpts`) emitted by the LIBERO2 client.

The script starts real InternVLA-A1.5 servers.  It is intentionally a smoke
gate, not a benchmark: success means all 30 episodes produced task result
JSON, no client was signal-killed, and no log contains SIGABRT/SIGSEGV.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _signal_name(returncode: int) -> str:
    if returncode >= 0:
        return ""
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"signal_{-returncode}"


def _write_config(libero_home: Path, path: Path) -> None:
    bench = libero_home / "libero" / "libero"
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text(
        "\n".join(
            [
                f"benchmark_root: {bench}",
                f"bddl_files: {bench}/bddl_files",
                f"init_states: {bench}/init_files",
                f"datasets: {bench}/../datasets",
                f"assets: {bench}/assets",
            ]
        )
        + "\n"
    )


def _healthcheck(
    python: Path,
    port: int,
    stats_key: str,
    resize_size: int,
    deadline: float,
) -> dict:
    # This parent-side import is websocket-only; it intentionally does not
    # import LIBERO, robosuite, mujoco, or OpenGL.
    sys.path.insert(0, str(REPO_ROOT))
    from evaluation.LIBERO.policy_server.tools.websocket_policy_client import WebsocketClientPolicy

    while True:
        try:
            client = WebsocketClientPolicy(host="127.0.0.1", port=port)
            metadata = client.get_server_metadata()
            client.close()
            assert metadata.get("action_mode") == "joint", metadata
            assert metadata.get("preprocessing_owner") == "server_canonical", metadata
            assert metadata.get("stats_key") == stats_key, metadata
            assert int(metadata.get("resize_size") or 0) == resize_size, metadata
            return metadata
        except (AssertionError, ConnectionError, OSError, TimeoutError):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"server on port {port} did not pass healthcheck")
            time.sleep(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-model standard LIBERO split EGL smoke")
    parser.add_argument(
        "--ckpt_path",
        default=os.environ.get("CKPT_PATH", ""),
        help="LIBERO-compatible checkpoint (stats.json must contain panda)",
    )
    parser.add_argument(
        "--vlm_model_path",
        default=os.environ.get("VLM_MODEL_PATH", ""),
    )
    parser.add_argument(
        "--libero_home",
        default=os.environ.get("LIBERO_HOME", "/home/a26113/DATA/LIBERO-plus"),
    )
    parser.add_argument("--server_venv", default=os.environ.get("SERVER_VENV", "/B/VENV/itnvla15rbt20"))
    parser.add_argument("--client_venv", default=os.environ.get("CLIENT_VENV", "/B/VENV/libero_plus_client"))
    parser.add_argument("--server_gpus", default="0,1")
    parser.add_argument("--client_gpus", default="2,3,4,5,6,7")
    parser.add_argument("--base_port", type=int, default=5894)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--task_count", type=int, default=6)
    parser.add_argument("--server_timeout", type=int, default=300)
    parser.add_argument("--client_timeout", type=int, default=1800)
    parser.add_argument("--json_out", type=Path, default=None)
    args = parser.parse_args()

    if not args.ckpt_path:
        parser.error("--ckpt_path or CKPT_PATH is required")
    server_ids = [int(x) for x in args.server_gpus.split(",")]
    client_ids = [int(x) for x in args.client_gpus.split(",")]
    if len(server_ids) != 2 or len(client_ids) != 6 or len(set(server_ids + client_ids)) != 8:
        parser.error("expected disjoint server_gpus=0,1 and six client_gpus=2,3,4,5,6,7")
    if args.task_count != 6:
        parser.error("this smoke intentionally launches one task per client (task_count=6)")

    ckpt = Path(args.ckpt_path)
    if not (ckpt / "config.json").exists() or not (ckpt / "stats.json").exists():
        parser.error(f"checkpoint missing config.json/stats.json: {ckpt}")
    stats = json.loads((ckpt / "stats.json").read_text())
    if "panda" not in stats:
        parser.error(f"checkpoint stats.json has no panda key: {list(stats)}")

    server_python = Path(args.server_venv) / "bin/python"
    client_python = Path(args.client_venv) / "bin/python"
    libero_home = Path(args.libero_home)
    if not server_python.exists() or not client_python.exists():
        parser.error("server/client venv python not found")

    root = Path(tempfile.mkdtemp(prefix="egl_split_standard_smoke_"))
    config_path = root / "libero_config"
    _write_config(libero_home, config_path)
    (root / "logs").mkdir()
    (root / "gpu_topology.txt").write_text(
        json.dumps(
            {
                "server_gpus": server_ids,
                "client_gpus": client_ids,
                "groups": [
                    {"server_gpu": server_ids[0], "clients": client_ids[:3], "port": args.base_port},
                    {"server_gpu": server_ids[1], "clients": client_ids[3:], "port": args.base_port + 1},
                ],
            },
            indent=2,
        )
    )

    servers: list[subprocess.Popen] = []
    clients: list[subprocess.Popen] = []

    def cleanup() -> None:
        for proc in clients + servers:
            if proc.poll() is None:
                proc.terminate()
        for proc in clients + servers:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

    try:
        for group, gpu in enumerate(server_ids):
            env = dict(os.environ)
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT / 'src'}:"
                    + env.get("PYTHONPATH", ""),
                }
            )
            log = (root / f"server{group}.log").open("w")
            proc = subprocess.Popen(
                [
                    str(server_python),
                    str(REPO_ROOT / "evaluation/LIBERO2/policy_server/server_policy.py"),
                    "--ckpt_path",
                    str(ckpt),
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(args.base_port + group),
                    "--device",
                    "cuda",
                    "--resize_size",
                    "224",
                    "--stats_key",
                    "panda",
                    "--robot_type",
                    "panda",
                    *(["--vlm_model_path", args.vlm_model_path] if args.vlm_model_path else []),
                    "--action_loss_only",
                    "--inference_backend",
                    "standard",
                    "--idle_timeout",
                    "-1",
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            servers.append(proc)

        deadline = time.monotonic() + args.server_timeout
        metadata = [
            _healthcheck(client_python, args.base_port + group, "panda", 224, deadline)
            for group in range(2)
        ]

        for index, gpu in enumerate(client_ids):
            group = index // 3
            env = dict(os.environ)
            env.update(
                {
                    "LIBERO_HOME": str(libero_home),
                    "LIBERO_CONFIG_PATH": str(config_path),
                    "RENDER_BACKEND": "egl",
                    "MUJOCO_GL": "egl",
                    "PYOPENGL_PLATFORM": "egl",
                    "MUJOCO_EGL_DEVICE_ID": str(gpu),
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "__EGL_VENDOR_LIBRARY_DIRS": str(Path(args.client_venv) / "egl_vendor.d"),
                    "LD_LIBRARY_PATH": f"{Path(args.client_venv) / 'lib'}:/usr/local/nvidia/lib64:"
                    + env.get("LD_LIBRARY_PATH", ""),
                    "PYTHONPATH": f"{libero_home}:{REPO_ROOT}:" + env.get("PYTHONPATH", ""),
                }
            )
            log = (root / f"client{index}.log").open("w")
            proc = subprocess.Popen(
                [
                    str(client_python),
                    str(REPO_ROOT / "evaluation/LIBERO2/eval_libero_std.py"),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.base_port + group),
                    "--task_suite_name",
                    "libero_spatial",
                    "--num_trials_per_task",
                    str(args.episodes),
                    "--num_steps_wait",
                    "10",
                    "--seed",
                    "7",
                    "--replan_steps",
                    "8",
                    "--start_idx",
                    str(index),
                    "--end_idx",
                    str(index + 1),
                    "--eval_log_dir",
                    str(root),
                    "--gripper_convention",
                    "libero_native",
                    "--enable_keypoints",
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            clients.append(proc)

        deadline = time.monotonic() + args.client_timeout
        while any(proc.poll() is None for proc in clients):
            if time.monotonic() >= deadline:
                raise TimeoutError("standard multi-episode client smoke timed out")
            time.sleep(2)

        summaries = []
        signal_killed = []
        log_crashes = []
        for index, proc in enumerate(clients):
            summary_path = root / "logs" / "libero_spatial" / f"std_{index}_to_{index + 1}.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            log = (root / f"client{index}.log").read_text()
            summaries.append(
                {
                    "client": index,
                    "gpu": client_ids[index],
                    "server_group": index // 3,
                    "returncode": proc.returncode,
                    "signal": _signal_name(proc.returncode),
                    "summary": summary,
                    "log_has_sigabrt": "SIGABRT" in log or "SIGSEGV" in log,
                    "log_has_crashed": "CRASHED" in log,
                }
            )
            if proc.returncode < 0:
                signal_killed.append(index)
            if "SIGABRT" in log or "SIGSEGV" in log or "CRASHED" in log:
                log_crashes.append(index)

        passed = (
            not signal_killed
            and not log_crashes
            and all(item["returncode"] == 0 for item in summaries)
            and all(item["summary"].get("total_episodes") == args.episodes for item in summaries)
        )
        report = {
            "passed": passed,
            "root": str(root),
            "server_metadata": metadata,
            "server_gpus": server_ids,
            "client_gpus": client_ids,
            "episodes_per_task": args.episodes,
            "total_episodes": args.episodes * len(client_ids),
            "signal_killed_clients": signal_killed,
            "log_crash_clients": log_crashes,
            "clients": summaries,
        }
        output = args.json_out or root / "report.json"
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print(f"OVERALL: {'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1
    finally:
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
