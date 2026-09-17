#!/usr/bin/env python3
"""A/B smoke benchmark for two-server and six-server EGL deployments.

The workload is intentionally identical for both topologies:

  - four standard LIBERO suites;
  - six tasks per suite (24 task definitions);
  - five episodes per task (120 episodes per topology);
  - six EGL clients on GPUs 2..7;
  - the same LIBERO-compatible InternVLA-A1.5 checkpoint.

Topology ``two``:
  GPU 0 -> one server -> clients 2,3,4
  GPU 1 -> one server -> clients 5,6,7

Topology ``six``:
  GPU 0 -> three servers -> clients 2,3,4
  GPU 1 -> three servers -> clients 5,6,7

The six-server topology gives every client a separate WebSocket endpoint while
the three model processes on each model GPU still share that GPU's compute.
The parent process never imports MuJoCo/OpenGL; only worker subprocesses do.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASKS_PER_SUITE_TOTAL = 10
SERVER_GPUS = (0, 1)
CLIENT_GPUS = (2, 3, 4, 5, 6, 7)


def _write_libero_config(libero_home: Path, config_path: Path) -> None:
    bench = libero_home / "libero" / "libero"
    config_path.mkdir(parents=True, exist_ok=True)
    (config_path / "config.yaml").write_text(
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


def _topology(name: str) -> dict[str, Any]:
    if name == "two":
        server_gpu_for_instance = [0, 1]
        client_server = [0, 0, 0, 1, 1, 1]
    elif name == "six":
        server_gpu_for_instance = [0, 0, 0, 1, 1, 1]
        client_server = [0, 1, 2, 3, 4, 5]
    else:
        raise ValueError(f"unsupported topology {name!r}")
    return {
        "name": name,
        "server_gpu_for_instance": server_gpu_for_instance,
        "client_server": client_server,
        "num_servers": len(server_gpu_for_instance),
    }


def _split_ranges(total: int, n_parts: int) -> list[tuple[int, int]]:
    result = []
    base, remainder = divmod(total, n_parts)
    start = 0
    for index in range(n_parts):
        size = base + (1 if index < remainder else 0)
        result.append((start, start + size))
        start += size
    return result


def _assignments(tasks_per_suite: int) -> list[list[tuple[str, int, int]]]:
    assignments: list[list[tuple[str, int, int]]] = [[] for _ in CLIENT_GPUS]
    for suite in SUITES:
        for client_index, (start, end) in enumerate(_split_ranges(tasks_per_suite, len(CLIENT_GPUS))):
            if start < end:
                assignments[client_index].append((suite, start, end))
    return assignments


def _wait_for_port(port: int, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"port {port} did not open before timeout")


def _healthcheck(port: int, expected_stats_key: str = "panda") -> dict[str, Any]:
    # WebSocket-only import.  Do not import LIBERO/MuJoCo in this parent.
    sys.path.insert(0, str(REPO_ROOT))
    from evaluation.LIBERO.policy_server.tools.websocket_policy_client import WebsocketClientPolicy

    client = WebsocketClientPolicy(host="127.0.0.1", port=port)
    metadata = client.get_server_metadata()
    client.close()
    assert metadata.get("action_mode") == "joint", metadata
    assert metadata.get("preprocessing_owner") == "server_canonical", metadata
    assert metadata.get("stats_key") == expected_stats_key, metadata
    assert int(metadata.get("resize_size") or 0) == 224, metadata
    return metadata


def _signal_name(returncode: int) -> str:
    if returncode >= 0:
        return ""
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"signal_{-returncode}"


def _snapshot_gpu(path: Path) -> None:
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
        path.write_text(
            "GPU memory:\n"
            + gpu.stdout
            + "\nCompute processes:\n"
            + apps.stdout
            + (("\n" + gpu.stderr) if gpu.stderr else "")
        )
    except OSError as exc:
        path.write_text(f"nvidia-smi unavailable: {exc}\n")


def _worker_main(args: argparse.Namespace) -> int:
    assignments = json.loads(Path(args.assignments).read_text())
    root = Path(args.eval_log_dir)
    logs = root / "client_logs"
    logs.mkdir(parents=True, exist_ok=True)
    records = []
    status = 0
    started = time.perf_counter()

    for suite, start, end in assignments:
        log_path = logs / f"client{args.client_index}_{suite}_{start}_{end}.log"
        command = [
            str(args.client_python),
            str(REPO_ROOT / "evaluation/LIBERO2/eval_libero_std.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port_base + args.server_index),
            "--task_suite_name",
            suite,
            "--num_trials_per_task",
            str(args.episodes),
            "--num_steps_wait",
            "10",
            "--seed",
            "7",
            "--replan_steps",
            "8",
            "--start_idx",
            str(start),
            "--end_idx",
            str(end),
            "--eval_log_dir",
            str(root),
            "--gripper_convention",
            "libero_native",
            "--enable_keypoints",
        ]
        command_started = time.perf_counter()
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record = {
            "suite": suite,
            "start_idx": start,
            "end_idx": end,
            "returncode": completed.returncode,
            "wall_seconds": round(time.perf_counter() - command_started, 3),
            "log": str(log_path),
        }
        records.append(record)
        if completed.returncode != 0:
            status = 1

    (root / f"client{args.client_index}_worker.json").write_text(
        json.dumps(
            {
                "client_index": args.client_index,
                "client_gpu": args.client_gpu,
                "server_index": args.server_index,
                "server_gpu": args.server_gpu,
                "commands": records,
                "wall_seconds": round(time.perf_counter() - started, 3),
                "returncode": status,
            },
            indent=2,
        )
    )
    return status


def _read_range_summary(root: Path, suite: str, start: int, end: int) -> dict[str, Any]:
    path = root / "logs" / suite / f"std_{start}_to_{end}.json"
    if not path.exists():
        return {"path": str(path), "missing": True}
    summary = json.loads(path.read_text())
    summary["path"] = str(path)
    return summary


def _run_topology(
    topology_name: str,
    args: argparse.Namespace,
    root: Path,
    server_python: Path,
    client_python: Path,
    libero_home: Path,
    config_path: Path,
) -> dict[str, Any]:
    topology = _topology(topology_name)
    topology_root = root / topology_name
    topology_root.mkdir(parents=True, exist_ok=True)
    assignments = _assignments(args.tasks_per_suite)
    assignment_paths = []
    for index, assignment in enumerate(assignments):
        path = topology_root / f"client{index}_assignments.json"
        path.write_text(json.dumps(assignment))
        assignment_paths.append(path)

    servers: list[subprocess.Popen] = []
    clients: list[subprocess.Popen] = []
    server_logs = []
    topology_started = time.perf_counter()

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
        for server_index, server_gpu in enumerate(topology["server_gpu_for_instance"]):
            port = args.base_port + (0 if topology_name == "two" else 100) + server_index
            env = dict(os.environ)
            for key in (
                "MUJOCO_GL",
                "PYOPENGL_PLATFORM",
                "MUJOCO_EGL_DEVICE_ID",
                "__EGL_VENDOR_LIBRARY_DIRS",
            ):
                env.pop(key, None)
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(server_gpu),
                    "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT / 'src'}:"
                    + env.get("PYTHONPATH", ""),
                }
            )
            log_path = topology_root / f"server{server_index}_gpu{server_gpu}.log"
            log = log_path.open("w")
            server_logs.append(log)
            command = [
                str(server_python),
                str(REPO_ROOT / "evaluation/LIBERO2/policy_server/server_policy.py"),
                "--ckpt_path",
                str(args.ckpt_path),
                "--vlm_model_path",
                str(args.vlm_model_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--device",
                "cuda",
                "--resize_size",
                "224",
                "--stats_key",
                "panda",
                "--robot_type",
                "panda",
                "--action_loss_only",
                "--inference_backend",
                "standard",
                "--idle_timeout",
                "-1",
            ]
            servers.append(
                subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )

        ready_deadline = time.monotonic() + args.server_timeout
        for server_index in range(topology["num_servers"]):
            port = args.base_port + (0 if topology_name == "two" else 100) + server_index
            _wait_for_port(port, ready_deadline)
        metadata = [
            _healthcheck(
                args.base_port + (0 if topology_name == "two" else 100) + server_index
            )
            for server_index in range(topology["num_servers"])
        ]
        ready_seconds = time.perf_counter() - topology_started
        _snapshot_gpu(topology_root / "gpu_snapshot_ready.txt")

        client_started = time.perf_counter()
        for client_index, client_gpu in enumerate(CLIENT_GPUS):
            server_index = topology["client_server"][client_index]
            server_gpu = topology["server_gpu_for_instance"][server_index]
            env = dict(os.environ)
            env.update(
                {
                    "LIBERO_HOME": str(libero_home),
                    "LIBERO_CONFIG_PATH": str(config_path),
                    "RENDER_BACKEND": "egl",
                    "MUJOCO_GL": "egl",
                    "PYOPENGL_PLATFORM": "egl",
                    "MUJOCO_EGL_DEVICE_ID": str(client_gpu),
                    "CUDA_VISIBLE_DEVICES": str(client_gpu),
                    "__EGL_VENDOR_LIBRARY_DIRS": str(Path(args.client_venv) / "egl_vendor.d"),
                    "LD_LIBRARY_PATH": f"{Path(args.client_venv) / 'lib'}:/usr/local/nvidia/lib64:"
                    + env.get("LD_LIBRARY_PATH", ""),
                    "PYTHONPATH": f"{libero_home}:{REPO_ROOT}:" + env.get("PYTHONPATH", ""),
                }
            )
            clients.append(
                subprocess.Popen(
                    [
                        str(client_python),
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--assignments",
                        str(assignment_paths[client_index]),
                        "--eval_log_dir",
                        str(topology_root),
                        "--client_index",
                        str(client_index),
                        "--client_gpu",
                        str(client_gpu),
                        "--server_index",
                        str(server_index),
                        "--server_gpu",
                        str(server_gpu),
                        "--port_base",
                        str(args.base_port + (0 if topology_name == "two" else 100)),
                        "--episodes",
                        str(args.episodes),
                        "--client_python",
                        str(client_python),
                    ],
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
            )

        client_deadline = time.monotonic() + args.client_timeout
        while any(proc.poll() is None for proc in clients):
            if time.monotonic() >= client_deadline:
                raise TimeoutError(f"{topology_name} clients exceeded timeout")
            time.sleep(2)
        client_wall = time.perf_counter() - client_started

        expected_tasks = args.tasks_per_suite * len(SUITES)
        expected_episodes = expected_tasks * args.episodes
        summaries = []
        crash_hits = []
        missing = []
        total_episodes = 0
        total_successes = 0
        for client_index, assignment in enumerate(assignments):
            worker_path = topology_root / f"client{client_index}_worker.json"
            worker = json.loads(worker_path.read_text()) if worker_path.exists() else {}
            client_log_paths = [
                Path(item["log"])
                for item in worker.get("commands", [])
                if item.get("log")
            ]
            client_crash = False
            for log_path in client_log_paths:
                if log_path.exists():
                    text = log_path.read_text(errors="replace")
                    if any(token in text for token in ("SIGABRT", "SIGSEGV", "CRASHED", "CUDA out of memory")):
                        client_crash = True
                        crash_hits.append(str(log_path))
            range_summaries = []
            for suite, start, end in assignment:
                summary = _read_range_summary(topology_root, suite, start, end)
                range_summaries.append(summary)
                if summary.get("missing"):
                    missing.append(summary["path"])
                total_episodes += int(summary.get("total_episodes", 0))
                total_successes += int(summary.get("total_successes", 0))
                for task_result in summary.get("per_task", []):
                    if task_result.get("error"):
                        client_crash = True
                        crash_hits.append(f"{suite}:{task_result.get('task_id')}: {task_result['error']}")
            summaries.append(
                {
                    "client_index": client_index,
                    "client_gpu": CLIENT_GPUS[client_index],
                    "server_index": topology["client_server"][client_index],
                    "server_gpu": topology["server_gpu_for_instance"][
                        topology["client_server"][client_index]
                    ],
                    "returncode": clients[client_index].returncode,
                    "signal": _signal_name(clients[client_index].returncode),
                    "worker": worker,
                    "summaries": range_summaries,
                    "crash": client_crash,
                }
            )

        passed = (
            not missing
            and not crash_hits
            and total_episodes == expected_episodes
            and all(item["returncode"] == 0 and not item["signal"] for item in summaries)
        )
        result = {
            "topology": topology_name,
            "server_gpu_for_instance": topology["server_gpu_for_instance"],
            "client_gpus": list(CLIENT_GPUS),
            "tasks_per_suite": args.tasks_per_suite,
            "suites": list(SUITES),
            "task_definitions": expected_tasks,
            "episodes_per_task": args.episodes,
            "expected_episodes": expected_episodes,
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "success_rate": total_successes / max(total_episodes, 1),
            "server_ready_seconds": round(ready_seconds, 3),
            "client_wall_seconds": round(client_wall, 3),
            "end_to_end_seconds": round(time.perf_counter() - topology_started, 3),
            "missing_result_jsons": missing,
            "crash_hits": crash_hits,
            "clients": summaries,
            "server_metadata": metadata,
            "passed": passed,
        }
        (topology_root / "report.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        cleanup()
        for log in server_logs:
            log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B smoke test for 2-server vs 6-server EGL")
    parser.add_argument("--topology", choices=("two", "six", "both"), default="both")
    parser.add_argument("--ckpt_path", default=os.environ.get("CKPT_PATH", ""))
    parser.add_argument("--vlm_model_path", default=os.environ.get("VLM_MODEL_PATH", ""))
    parser.add_argument("--libero_home", default=os.environ.get("LIBERO_HOME", "/home/a26113/DATA/LIBERO-plus"))
    parser.add_argument("--server_venv", default=os.environ.get("SERVER_VENV", "/B/VENV/itnvla15rbt20"))
    parser.add_argument("--client_venv", default=os.environ.get("CLIENT_VENV", "/B/VENV/libero_plus_client"))
    parser.add_argument("--tasks_per_suite", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--base_port", type=int, default=5924)
    parser.add_argument("--server_timeout", type=int, default=600)
    parser.add_argument("--client_timeout", type=int, default=2400)
    parser.add_argument("--json_out", type=Path, default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--assignments", default="", help=argparse.SUPPRESS)
    parser.add_argument("--eval_log_dir", default="", help=argparse.SUPPRESS)
    parser.add_argument("--client_index", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--client_gpu", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--server_index", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--server_gpu", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--port_base", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--client_python", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if not args.assignments or not args.eval_log_dir or args.client_index < 0:
            parser.error("worker arguments are incomplete")
        return _worker_main(args)

    if not args.ckpt_path or not args.vlm_model_path:
        parser.error("--ckpt_path and --vlm_model_path are required")
    if not 1 <= args.tasks_per_suite <= TASKS_PER_SUITE_TOTAL:
        parser.error(f"--tasks_per_suite must be in [1, {TASKS_PER_SUITE_TOTAL}]")
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")

    ckpt = Path(args.ckpt_path)
    if not (ckpt / "config.json").exists() or not (ckpt / "stats.json").exists():
        parser.error(f"checkpoint missing config.json/stats.json: {ckpt}")
    if "panda" not in json.loads((ckpt / "stats.json").read_text()):
        parser.error("checkpoint stats.json must contain the panda key")

    server_python = Path(args.server_venv) / "bin/python"
    client_python = Path(args.client_venv) / "bin/python"
    libero_home = Path(args.libero_home)
    if not server_python.exists() or not client_python.exists():
        parser.error("server/client venv python not found")

    root = Path(tempfile.mkdtemp(prefix="egl_topology_ab_"))
    config_path = root / "libero_config"
    _write_libero_config(libero_home, config_path)
    names = ("two", "six") if args.topology == "both" else (args.topology,)
    results = []
    for name in names:
        results.append(
            _run_topology(
                name,
                args,
                root,
                server_python,
                client_python,
                libero_home,
                config_path,
            )
        )

    comparison = {"root": str(root), "results": results}
    if len(results) == 2:
        two, six = results
        comparison["comparison"] = {
            "six_minus_two_client_wall_seconds": round(
                six["client_wall_seconds"] - two["client_wall_seconds"], 3
            ),
            "six_over_two_client_wall_ratio": round(
                six["client_wall_seconds"] / max(two["client_wall_seconds"], 1e-9), 4
            ),
            "both_passed": bool(two["passed"] and six["passed"]),
            "six_has_fewer_crash_hits": len(six["crash_hits"]) < len(two["crash_hits"]),
        }
    output = args.json_out or root / "comparison.json"
    output.write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))
    print(f"OVERALL: {'PASS' if all(item['passed'] for item in results) else 'FAIL'}")
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
