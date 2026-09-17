#!/usr/bin/env python3
"""Acceptance tests for the standard-LIBERO two/six-server split topology.

This is the standard-LIBERO twin of
evaluation/LIBERO-plus2/test_egl_split_topology.py: it checks
evaluation/LIBERO2/run_eval_libero_std_2server_6client_venv.sh instead of the
LIBERO-plus launcher. The production default is six server instances (three
per GPU, ``SERVER_INSTANCES_PER_GPU=3``); setting it to ``1`` falls back to
the original two-server topology:

    server GPU 0, port P       <- client GPUs 2, 3, 4
    server GPU 1, port P + 1   <- client GPUs 5, 6, 7

The static checks are safe to run anywhere. ``--live`` starts two *mock*
policy servers and six real LIBERO/EGL client processes that each render on
their own physical EGL device and drive `evaluation/LIBERO2/eval_libero_std.py`'s
underlying env against a single un-perturbed BDDL, exercising the exact
renderer/env code path the production launcher uses. This isolates the
renderer/topology from model-inference correctness; the full-model smoke is
`run_eval_libero_std_2server_6client_venv.sh EVAL_MODE=smoke`.

Usage:
    python evaluation/LIBERO2/test_egl_split_std_topology.py

    export LIBERO_HOME=/home/a26113/DATA/LIBERO-plus
    export LIBERO_CONFIG_PATH=/tmp/test_libero_std_config
    python evaluation/LIBERO2/test_egl_split_std_topology.py --live \
        --server_venv /B/VENV/itnvla15rbt20 \
        --client_venv /B/VENV/libero_plus_client
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
LAUNCHER = Path(__file__).with_name("run_eval_libero_std_2server_6client_venv.sh")
AGGREGATOR = Path(__file__).with_name("aggregate_std_results.py")


def _ids(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def validate_topology(server_ids: list[int], client_ids: list[int], clients_per_server: int = 3) -> dict:
    """Return a deterministic topology manifest or raise ValueError."""
    if len(server_ids) != 2:
        raise ValueError(f"expected exactly 2 server GPUs, got {server_ids}")
    if len(client_ids) != 6:
        raise ValueError(f"expected exactly 6 client GPUs, got {client_ids}")
    if clients_per_server != 3:
        raise ValueError(f"expected 3 clients/server, got {clients_per_server}")
    all_ids = server_ids + client_ids
    if len(set(all_ids)) != len(all_ids):
        raise ValueError(f"server/client GPU sets overlap or contain duplicates: {all_ids}")
    groups = []
    for group in range(2):
        members = client_ids[group * clients_per_server : (group + 1) * clients_per_server]
        groups.append(
            {
                "group": group,
                "server_gpu": server_ids[group],
                "port_offset": group,
                "client_gpus": members,
                "shared_gpu_ids": sorted(set(members) & {server_ids[group]}),
            }
        )
    return {"server_gpus": server_ids, "client_gpus": client_ids, "groups": groups}


def check_static() -> bool:
    src = LAUNCHER.read_text()
    checks = {
        "launcher_exists": LAUNCHER.exists(),
        "aggregator_exists": AGGREGATOR.exists(),
        "default_two_server_gpus": 'SERVER_GPU_IDS="${SERVER_GPU_IDS:-0,1}"' in src,
        "default_six_client_gpus": 'CLIENT_GPU_IDS="${CLIENT_GPU_IDS:-2,3,4,5,6,7}"' in src,
        "requires_two_servers": "NUM_SERVERS} -ne 2" in src,
        "requires_six_clients": "NUM_CLIENTS} -ne 6" in src,
        "server_cuda_isolation": 'export CUDA_VISIBLE_DEVICES="${server_gpu}"' in src,
        "client_cuda_isolation": 'export CUDA_VISIBLE_DEVICES="${client_gpu}"' in src,
        "client_egl_physical_id": 'export MUJOCO_EGL_DEVICE_ID="${client_gpu}"' in src,
        "client_forces_egl": "export RENDER_BACKEND=egl" in src,
        "server_ports_are_grouped": 'local port=$((BASE_PORT + group))' in src,
        "three_clients_per_server": "CLIENTS_PER_SERVER" in src,
        "six_server_is_default": 'SERVER_INSTANCES_PER_GPU="${SERVER_INSTANCES_PER_GPU:-3}"' in src
        and 'SERVER_INSTANCES_PER_GPU}" =~ ^[13]$' in src,
        "server_instance_gpu_expansion": "SERVER_INSTANCE_GPU_ARRAY" in src,
        "six_server_ports_supported": "NUM_SERVER_INSTANCES" in src,
        "four_suites_all_tasks_by_default": (
            "libero_spatial libero_object libero_goal libero_10" in src
            and "SUITE_N_TASKS" in src
        ),
        "no_task_classification_dependency": "task_classification" not in src,
        "server_never_initializes_egl": "unset MUJOCO_GL PYOPENGL_PLATFORM" in src,
        "no_osmesa_in_split_launcher": "MUJOCO_GL=osmesa" not in src,
        "standard_backend_default": 'INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"' in src,
        "calls_std_client_script": "evaluation/LIBERO2/eval_libero_std.py" in src,
        "calls_std_aggregator": "evaluation/LIBERO2/aggregate_std_results.py" in src,
        "smoke_and_full_trial_counts": (
            'NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-5}"' in src
            and 'NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"' in src
        ),
    }
    manifest = validate_topology([0, 1], [2, 3, 4, 5, 6, 7])
    checks["default_manifest_disjoint"] = all(
        not group["shared_gpu_ids"] for group in manifest["groups"]
    )
    for name, passed in checks.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print("  topology:", json.dumps(manifest, sort_keys=True))
    return all(checks.values())


def _ensure_config(libero_home: Path, config_path: str | None) -> Path:
    if config_path:
        path = Path(config_path)
        if not (path / "config.yaml").exists():
            raise FileNotFoundError(f"config.yaml missing under {path}")
        return path
    path = Path(tempfile.mkdtemp(prefix="egl_split_std_config_"))
    bench = libero_home / "libero" / "libero"
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
    return path


def _signal_name(returncode: int) -> str:
    if returncode >= 0:
        return ""
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"signal_{-returncode}"


def _client_child(
    client_gpu: int,
    port: int,
    libero_home: str,
    config_path: str,
    episodes: int,
    steps: int,
    out_path: Path,
) -> int:
    """Run actual EGL+LIBERO rendering and mock-server requests in one child."""
    from evaluation.LIBERO2.orientation_contract import UNPERTURBED_BDDL, setup_client_render_env

    setup_client_render_env(
        backend="egl",
        libero_home=libero_home,
        libero_config=config_path,
        device=str(client_gpu),
    )

    from evaluation.LIBERO.policy_server.tools.websocket_policy_client import WebsocketClientPolicy
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    client = WebsocketClientPolicy(host="127.0.0.1", port=port)
    metadata = client.get_server_metadata()
    bddl = Path(get_libero_path("bddl_files")) / UNPERTURBED_BDDL
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(7)
    dummy = [0.0] * 6 + [-1.0]
    episodes_out: list[dict] = []

    from OpenGL import GL

    renderer = GL.glGetString(GL.GL_RENDERER)
    gl_renderer = renderer.decode(errors="replace") if renderer else ""

    for episode in range(episodes):
        obs = env.reset()
        checksum = 0.0
        for step in range(steps):
            response = client.predict_action(
                {
                    "type": "infer",
                    "request_id": f"split-std-smoke-{client_gpu}-{episode}-{step}",
                    "payload": {"examples": [{}], "do_sample": False},
                }
            )
            if not response.get("ok", False):
                raise RuntimeError(f"mock server returned an error: {response}")
            checksum += float(response["data"]["actions"].mean())
            obs, _, _, _ = env.step(dummy)
            checksum += float(obs["agentview_image"].mean())
            checksum += float(obs["robot0_eye_in_hand_image"].mean())
        episodes_out.append({"episode": episode, "steps": steps, "checksum": round(checksum, 4)})
        out_path.write_text(
            json.dumps(
                {
                    "client_gpu": client_gpu,
                    "port": port,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
                    "gl_renderer": gl_renderer,
                    "server_metadata": metadata,
                    "completed": len(episodes_out),
                    "episodes": episodes_out,
                },
                indent=2,
            )
        )

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def run_live(args: argparse.Namespace) -> bool:
    server_ids = _ids(args.server_gpus)
    client_ids = _ids(args.client_gpus)
    manifest = validate_topology(server_ids, client_ids)
    server_venv = Path(args.server_venv)
    client_venv = Path(args.client_venv)
    server_python = server_venv / "bin" / "python"
    client_python = client_venv / "bin" / "python"
    if not server_python.exists():
        raise FileNotFoundError(server_python)
    if not client_python.exists():
        raise FileNotFoundError(client_python)

    libero_home = Path(args.libero_home)
    config_path = _ensure_config(libero_home, args.config_path)
    root = Path(tempfile.mkdtemp(prefix="egl_split_std_live_"))
    server_logs = root / "servers"
    client_logs = root / "clients"
    server_logs.mkdir()
    client_logs.mkdir()
    server_procs: list[subprocess.Popen] = []
    client_procs: list[subprocess.Popen] = []

    def cleanup() -> None:
        for proc in client_procs + server_procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in client_procs + server_procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    try:
        for group, server_gpu in enumerate(server_ids):
            port = args.base_port + group
            env = dict(os.environ)
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(server_gpu),
                    "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT / 'src'}:" + env.get("PYTHONPATH", ""),
                }
            )
            log = (server_logs / f"server{group}.log").open("w")
            proc = subprocess.Popen(
                [
                    str(server_python),
                    str(REPO_ROOT / "evaluation/LIBERO2/policy_server/server_policy.py"),
                    "--mock_policy",
                    "random",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--mock_action_dim",
                    "7",
                    "--mock_chunk_size",
                    "16",
                    "--idle_timeout",
                    "-1",
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            server_procs.append(proc)

        sys.path.insert(0, str(REPO_ROOT))
        from evaluation.LIBERO.policy_server.tools.websocket_policy_client import WebsocketClientPolicy

        deadline = time.monotonic() + args.server_timeout
        for group in range(2):
            while True:
                try:
                    client = WebsocketClientPolicy(host="127.0.0.1", port=args.base_port + group)
                    metadata = client.get_server_metadata()
                    assert metadata["policy_type"] == "mock_random"
                    client.close()
                    break
                except (AssertionError, ConnectionError, OSError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"mock server {group} did not become ready")
                    time.sleep(0.5)

        for client_idx, client_gpu in enumerate(client_ids):
            group = client_idx // 3
            out = client_logs / f"client{client_idx}.json"
            env = dict(os.environ)
            env.update(
                {
                    "LIBERO_HOME": str(libero_home),
                    "LIBERO_CONFIG_PATH": str(config_path),
                    "MUJOCO_GL": "egl",
                    "PYOPENGL_PLATFORM": "egl",
                    "MUJOCO_EGL_DEVICE_ID": str(client_gpu),
                    "CUDA_VISIBLE_DEVICES": str(client_gpu),
                    "RENDER_BACKEND": "egl",
                    "LD_LIBRARY_PATH": f"{client_venv / 'lib'}:/usr/local/nvidia/lib64:"
                    + env.get("LD_LIBRARY_PATH", ""),
                    "PYTHONPATH": f"{libero_home}:{REPO_ROOT}:" + env.get("PYTHONPATH", ""),
                }
            )
            log = (client_logs / f"client{client_idx}.log").open("w")
            proc = subprocess.Popen(
                [
                    str(client_python),
                    str(Path(__file__).resolve()),
                    "--_client",
                    "--client_gpu",
                    str(client_gpu),
                    "--port",
                    str(args.base_port + group),
                    "--libero_home",
                    str(libero_home),
                    "--config_path",
                    str(config_path),
                    "--episodes",
                    str(args.episodes),
                    "--steps",
                    str(args.steps),
                    "--out",
                    str(out),
                ],
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            client_procs.append(proc)

        deadline = time.monotonic() + args.client_timeout
        while any(proc.poll() is None for proc in client_procs):
            if time.monotonic() >= deadline:
                raise TimeoutError("live std client smoke timed out")
            time.sleep(1)

        results = []
        for idx, proc in enumerate(client_procs):
            out = client_logs / f"client{idx}.json"
            result = json.loads(out.read_text()) if out.exists() else {}
            result.update(
                {
                    "client_index": idx,
                    "returncode": proc.returncode,
                    "signal": _signal_name(proc.returncode),
                    "completed_expected": result.get("completed") == args.episodes,
                }
            )
            results.append(result)

        passed = all(
            proc.returncode == 0
            and result.get("completed_expected")
            and result.get("mujoco_egl_device_id") == str(result.get("client_gpu"))
            for proc, result in zip(client_procs, results)
        )
        report = {
            "passed": passed,
            "root": str(root),
            "topology": manifest,
            "episodes_per_client": args.episodes,
            "steps_per_episode": args.steps,
            "clients": results,
        }
        report_path = Path(args.json_out) if args.json_out else root / "report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print(f"OVERALL: {'PASS' if passed else 'FAIL'}")
        return passed
    finally:
        cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description="Standard-LIBERO two/six-server split topology acceptance")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--_client", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--server_gpus", default="0,1")
    parser.add_argument("--client_gpus", default="2,3,4,5,6,7")
    parser.add_argument("--base_port", type=int, default=5894)
    parser.add_argument("--server_venv", default="/B/VENV/itnvla15rbt20")
    parser.add_argument("--client_venv", default="/B/VENV/libero_plus_client")
    parser.add_argument("--libero_home", default=os.environ.get("LIBERO_HOME", "/home/a26113/DATA/LIBERO-plus"))
    parser.add_argument("--config_path", default=os.environ.get("LIBERO_CONFIG_PATH", ""))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--server_timeout", type=int, default=120)
    parser.add_argument("--client_timeout", type=int, default=900)
    parser.add_argument("--client_gpu", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--json_out", type=Path, default=None)
    args = parser.parse_args()

    if args._client:
        if args.client_gpu < 0 or args.port < 0 or args.out is None:
            parser.error("internal client arguments are incomplete")
        return _client_child(
            args.client_gpu,
            args.port,
            args.libero_home,
            args.config_path,
            args.episodes,
            args.steps,
            args.out,
        )

    print("[static] standard-LIBERO two/six-server split topology")
    static_passed = check_static()
    if not args.live:
        print(f"OVERALL: {'PASS' if static_passed else 'FAIL'}")
        return 0 if static_passed else 1
    if not static_passed:
        print("LIVE: skipped because static checks failed")
        return 1
    return 0 if run_live(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
