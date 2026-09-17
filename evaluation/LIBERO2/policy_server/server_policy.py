"""LIBERO2 policy server — uses LIBERO2's backend_factory (with B2 fix + keypoints)."""
from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.LIBERO2.policy_server.backends.backend_factory import build_backend
from evaluation.LIBERO.policy_server.tools.websocket_policy_server import WebsocketPolicyServer


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LIBERO2 policy server for evaluation.")
    parser.add_argument("--ckpt_path", type=str, default="", help="Checkpoint path.")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resize_size", type=int, default=224)
    parser.add_argument("--stats_key", type=str, default="", help="Stats key in stats.json.")
    parser.add_argument("--robot_type", type=str, default="", help="Robot type override.")
    parser.add_argument("--idle_timeout", type=int, default=1800, help="Idle timeout (-1 = never).")
    parser.add_argument("--vlm_model_path", type=str, default="")
    parser.add_argument("--wan_model_path", type=str, default="")
    parser.add_argument("--wan_vae_path", type=str, default="")
    parser.add_argument(
        "--action_loss_only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--inference_backend",
        type=str,
        default="standard",
        choices=("standard", "optimized"),
    )
    parser.add_argument("--mock_policy", type=str, default="", choices=["", "random"])
    parser.add_argument("--mock_action_dim", type=int, default=7)
    parser.add_argument("--mock_chunk_size", type=int, default=16)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(args: argparse.Namespace) -> None:
    stats_key = args.stats_key or None
    robot_type = args.robot_type or None
    mock_policy = args.mock_policy or None

    if not mock_policy and not args.ckpt_path:
        raise ValueError("--ckpt_path is required unless --mock_policy is set")

    backend = build_backend(
        ckpt_path=args.ckpt_path,
        device=args.device,
        stats_key=stats_key,
        robot_type=robot_type,
        resize_size=args.resize_size,
        mock_policy=mock_policy,
        mock_action_dim=args.mock_action_dim,
        mock_chunk_size=args.mock_chunk_size,
        vlm_model_path=args.vlm_model_path or None,
        wan_model_path=args.wan_model_path or None,
        wan_vae_path=args.wan_vae_path or None,
        action_loss_only=args.action_loss_only,
        inference_backend=args.inference_backend,
    )

    metadata = backend.metadata()
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating LIBERO2 server (host: %s, ip: %s)", hostname, local_ip)
    logging.info("Backend metadata: %s", metadata)

    server = WebsocketPolicyServer(
        policy=backend,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
    )
    logging.info("LIBERO2 server running ...")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available() and not args.mock_policy:
        raise RuntimeError("CUDA requested but not available")
    main(args)
