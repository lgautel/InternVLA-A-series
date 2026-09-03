#!/usr/bin/env python
"""InternVLA-A1.5 inference server for R1 Pro real-robot deployment.

WebSocket server matching the openpi bare-dict protocol that EFMNode already speaks.
Loads an InternVLA-A1.5 checkpoint (with optional GeoPredict 16-keypoint path),
preprocesses observations, runs the model, and returns denormalized actions.

Usage:
    python evaluation/R1Pro/inference.py --ckpt-path <checkpoint>

    # With GeoPredict keypoints:
    python evaluation/R1Pro/inference.py --ckpt-path <checkpoint> \
        --kpt-meta-path <keypoints_meta.json>
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.dataset_schemas import get_schema
from lerobot.policies.factory import get_policy_class
from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config
from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
    InternVLAA15ChatProcessorTransformFn,
)
from lerobot.transforms.core import (
    ComposeFieldsTransform,
    NormalizeTransformFn,
    PadStateAndActionTransformFn,
    RemapImageKeyTransformFn,
    ReorderStateActionTransform,
    ResizeImagesWithPadFn,
    UnNormalizeTransformFn,
    compose,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# msgpack numpy support (inline copy from openpi)
# ---------------------------------------------------------------------------

import msgpack  # noqa: E402


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_Packer = functools.partial(msgpack.Packer, default=_pack_array, use_bin_type=True)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array, raw=False)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EFMNODE_STATE_DIM = 23
MODEL_STATE_DIM = 25
MODEL_ACTION_DIM = 19

R1PRO_STATE_FIELDS = [
    ("observation.state.left_arm", 7),
    ("observation.state.right_arm", 7),
    ("observation.state.left_gripper", 1),
    ("observation.state.right_gripper", 1),
    ("observation.state.chassis", 9),
]

R1PRO_IMAGE_KEYS = {
    f"{OBS_IMAGES}.head_rgb": f"{OBS_IMAGES}.image0",
    f"{OBS_IMAGES}.wrist_left_rgb": f"{OBS_IMAGES}.image1",
    f"{OBS_IMAGES}.wrist_right_rgb": f"{OBS_IMAGES}.image2",
}


# ---------------------------------------------------------------------------
# State / Action remapping
# ---------------------------------------------------------------------------


def remap_efmnode_state(state_23d: np.ndarray) -> dict[str, torch.Tensor]:
    """Split EFMNode's 23D flat state into per-field tensors for the transform pipeline.

    EFMNode order: la7 + ra7 + lg1 + rg1 + torso4 + chassis3
    Model fields:  la7 + ra7 + lg1 + rg1 + chassis9 (torso dropped, chassis zero-padded)
    """
    s = np.asarray(state_23d, dtype=np.float32)
    if s.shape[0] != EFMNODE_STATE_DIM:
        raise ValueError(f"Expected {EFMNODE_STATE_DIM}-D state from EFMNode, got {s.shape[0]}")
    chassis_3d = s[20:23]
    chassis_9d = np.zeros(9, dtype=np.float32)
    chassis_9d[:3] = chassis_3d

    return {
        "observation.state.left_arm": torch.from_numpy(s[0:7].copy()),
        "observation.state.right_arm": torch.from_numpy(s[7:14].copy()),
        "observation.state.left_gripper": torch.from_numpy(s[14:15].copy()),
        "observation.state.right_gripper": torch.from_numpy(s[15:16].copy()),
        "observation.state.chassis": torch.from_numpy(chassis_9d),
    }


def remap_action_to_efmnode(action: np.ndarray, pad_torso: bool = True) -> np.ndarray:
    """Convert model's 19D feature-order actions to EFMNode's 23D format.

    Model order:   la7 + ra7 + lg1 + rg1 + chassis3
    EFMNode order: la7 + ra7 + lg1 + rg1 + torso4 + chassis3
    """
    if not pad_torso:
        return action
    if action.ndim == 1:
        out = np.zeros(23, dtype=action.dtype)
        out[0:16] = action[0:16]
        out[20:23] = action[16:19]
        return out
    out = np.zeros((*action.shape[:-1], 23), dtype=action.dtype)
    out[..., 0:16] = action[..., 0:16]
    out[..., 20:23] = action[..., 16:19]
    return out


def compute_inverse_reorder(forward_spec: list[list[int]] | None) -> list[list[int]] | None:
    """Invert a reorder spec by swapping src/dst."""
    if forward_spec is None:
        return None
    return [[dst_s, dst_e, src_s, src_e] for src_s, src_e, dst_s, dst_e in forward_spec]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_policy(args, dtype):
    config = PreTrainedConfig.from_pretrained(args.ckpt_path)
    if not isinstance(config, InternVLAA15Config):
        raise ValueError(f"Checkpoint policy.type must be 'internvla_a1_5', got {config.type!r}.")

    config.action_loss_only = True
    config.inference_backend = args.inference_backend
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(args.ckpt_path, config=config)
    device = torch.device(config.device)
    policy.to(device=device, dtype=dtype)
    policy.eval()
    logger.info("Model loaded: backend=%s, device=%s, dtype=%s, kpt=%s",
                config.inference_backend, device, dtype, config.enable_keypoint_predictor)
    return policy, device, config


def load_stats(stats_path: Path) -> tuple[dict, dict]:
    """Load per-field norm stats. Returns per-field dicts for state and composed dict for action."""
    with open(stats_path) as f:
        raw = json.load(f)

    # Unwrap dataset-level nesting (e.g. {"r1_pro": {field: ...}} → {field: ...})
    if len(raw) == 1 and isinstance(next(iter(raw.values())), dict):
        raw = next(iter(raw.values()))

    state_fields = [k for k, _ in R1PRO_STATE_FIELDS]
    action_fields = [
        "action.left_arm", "action.right_arm",
        "action.left_gripper", "action.right_gripper",
        "action.chassis.velocities",
    ]

    def pick(fk):
        if fk not in raw:
            raise KeyError(f"Stats field '{fk}' not found. Available: {list(raw.keys())}")
        d = {}
        for s in ("mean", "std", "min", "max"):
            if s in raw[fk]:
                d[s] = np.atleast_1d(np.asarray(raw[fk][s], dtype=np.float32))
        return d

    state_stat = {fk: pick(fk) for fk in state_fields}

    action_arrays = {s: [] for s in ("mean", "std", "min", "max")}
    for fk in action_fields:
        p = pick(fk)
        for s in action_arrays:
            if s in p:
                action_arrays[s].append(p[s])
    action_stat = {ACTION: {s: np.concatenate(action_arrays[s]) for s in action_arrays if action_arrays[s]}}

    return state_stat, action_stat


def build_input_transforms(resize_size, state_stat, config):
    schema = get_schema("r1_pro")
    state_keys = list(state_stat.keys())
    return compose([
        ResizeImagesWithPadFn(height=resize_size, width=resize_size, mapping=schema.image_mapping),
        RemapImageKeyTransformFn(mapping=schema.image_mapping),
        NormalizeTransformFn(selected_keys=state_keys, norm_stats=state_stat),
        ComposeFieldsTransform(mapping=schema.feature_mapping),
        InternVLAA15ChatProcessorTransformFn(
            mode="eval",
            tokenize_state=getattr(config, "tokenize_state", True),
            max_state_dim=getattr(config, "max_state_dim", 32),
        ),
        PadStateAndActionTransformFn(
            max_state_dim=getattr(config, "max_state_dim", 32),
            max_action_dim=getattr(config, "max_action_dim", 32),
        ),
        ReorderStateActionTransform(
            state_reorder=schema.state_reorder,
            action_reorder=schema.action_reorder,
        ),
    ])


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------


def build_sample(obs_dict: dict) -> dict:
    """Convert WebSocket observation dict to the sample dict expected by transforms."""
    state_fields = remap_efmnode_state(obs_dict["state"])

    sample = {
        **state_fields,
        "action.left_arm": torch.zeros(50, 7, dtype=torch.float32),
        "action.right_arm": torch.zeros(50, 7, dtype=torch.float32),
        "action.left_gripper": torch.zeros(50, 1, dtype=torch.float32),
        "action.right_gripper": torch.zeros(50, 1, dtype=torch.float32),
        "action.chassis.velocities": torch.zeros(50, 3, dtype=torch.float32),
        "task": obs_dict.get("prompt", ""),
    }

    IMAGE_MAP = {
        "head_rgb": f"{OBS_IMAGES}.head_rgb",
        "left_wrist_rgb": f"{OBS_IMAGES}.wrist_left_rgb",
        "right_wrist_rgb": f"{OBS_IMAGES}.wrist_right_rgb",
    }
    for src_key, obs_key in IMAGE_MAP.items():
        img = obs_dict[src_key]
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        if img.ndim == 3 and img.shape[-1] in (1, 3):
            img = img.permute(2, 0, 1)
        sample[obs_key] = img

    return sample


def to_policy_batch(sample: dict, device: torch.device, dtype: torch.dtype) -> dict:
    batch = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            value = value.unsqueeze(0)
            if value.dtype.is_floating_point:
                value = value.to(device=device, dtype=dtype)
            else:
                value = value.to(device=device)
            batch[key] = value
        else:
            batch[key] = [value]
    return batch


# ---------------------------------------------------------------------------
# GeoPredict Keypoint Tracker
# ---------------------------------------------------------------------------


class R1ProKeypointTracker:
    """Online FK keypoint extraction for GeoPredict inference."""

    def __init__(self, urdf_path: str, meta_path: str, history_max_len: int = 300):
        with open(meta_path) as f:
            meta = json.load(f)

        torso_q = tuple(meta["torso_q"])
        self.coord_offset = np.asarray(meta["coord_offset"], dtype=np.float64)

        if str(REPO_ROOT / "util_scripts") not in sys.path:
            sys.path.insert(0, str(REPO_ROOT / "util_scripts"))
        from generate_r1pro_keypoints import R1ProFKExtractor

        self.fk = R1ProFKExtractor(urdf_path, torso_q=torso_q)
        self.H = history_max_len
        self.his_kpts = np.zeros((self.H, 16, 3), dtype=np.float32)
        self.his_len = 0

        logger.info("KeypointTracker: urdf=%s, torso_q=%s, offset=%s, H=%d",
                     urdf_path, torso_q, self.coord_offset, self.H)

    def update(self, left_arm: np.ndarray, right_arm: np.ndarray):
        kpt_base = self.fk.compute(left_arm.astype(np.float64), right_arm.astype(np.float64))
        kpt_voxel = (kpt_base - self.coord_offset[np.newaxis, :]).astype(np.float32)

        if self.his_len < self.H:
            self.his_kpts[self.his_len] = kpt_voxel
        else:
            self.his_kpts[:-1] = self.his_kpts[1:]
            self.his_kpts[-1] = kpt_voxel
        self.his_len = min(self.his_len + 1, self.H)

    def get_tensors(self, device, dtype):
        his_kpts = torch.from_numpy(self.his_kpts).unsqueeze(0).to(device=device, dtype=dtype)
        his_len = torch.tensor([self.his_len], dtype=torch.long, device=device)
        return his_kpts, his_len

    def reset(self):
        self.his_kpts[:] = 0
        self.his_len = 0


# ---------------------------------------------------------------------------
# Inference Server
# ---------------------------------------------------------------------------


class R1ProInferenceServer:
    def __init__(self, policy, input_transforms, unnormalize_fn,
                 config, device, dtype, inverse_action_reorder,
                 kpt_tracker, pad_torso):
        self.policy = policy
        self.input_transforms = input_transforms
        self.unnormalize_fn = unnormalize_fn
        self.config = config
        self.device = device
        self.dtype = dtype
        self.inverse_action_reorder = inverse_action_reorder
        self.kpt_tracker = kpt_tracker
        self.pad_torso = pad_torso
        self.use_kpt = config.enable_keypoint_predictor and kpt_tracker is not None
        self._call_count = 0

    def predict(self, obs_dict: dict) -> dict:
        t0 = time.monotonic()

        sample = build_sample(obs_dict)
        sample = self.input_transforms(sample)
        batch = to_policy_batch(sample, self.device, self.dtype)

        if self.use_kpt:
            raw_state = np.asarray(obs_dict["state"], dtype=np.float32)
            self.kpt_tracker.update(raw_state[0:7], raw_state[7:14])
            his_kpts, his_len = self.kpt_tracker.get_tensors(self.device, self.dtype)
            batch["observation.his_kpts"] = his_kpts
            batch["observation.his_len"] = his_len

        with torch.no_grad():
            action_pred = self.policy.predict_action_chunk(batch)

        if action_pred.ndim != 3 or action_pred.shape[0] != 1:
            raise ValueError(
                f"Expected predict_action_chunk shape [1, T, D], got {action_pred.shape}"
            )
        action_pred = action_pred[0]
        action_pred = self._reverse_reorder(action_pred)
        action_pred = self.unnormalize_fn({ACTION: action_pred})[ACTION]

        actions_np = action_pred.detach().float().cpu().numpy()
        if self.pad_torso:
            actions_np = remap_action_to_efmnode(actions_np, pad_torso=True)

        infer_ms = (time.monotonic() - t0) * 1000
        self._call_count += 1
        if self._call_count % 10 == 1:
            logger.info("predict #%d: %.1fms, actions shape=%s", self._call_count, infer_ms, actions_np.shape)

        return {
            "actions": actions_np,
            "policy_timing": {"infer_ms": infer_ms},
            "server_timing": {"infer_ms": infer_ms},
        }

    def _reverse_reorder(self, canonical_actions: torch.Tensor) -> torch.Tensor:
        expected_dim = getattr(self.config, "max_action_dim", 32)
        if canonical_actions.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected model output dim {expected_dim}, got {canonical_actions.shape[-1]}"
            )
        if self.inverse_action_reorder is None:
            return canonical_actions[..., :MODEL_ACTION_DIM]
        output = torch.zeros_like(canonical_actions)
        for src_s, src_e, dst_s, dst_e in self.inverse_action_reorder:
            output[..., dst_s:dst_e] = canonical_actions[..., src_s:src_e]
        return output[..., :MODEL_ACTION_DIM]


# ---------------------------------------------------------------------------
# WebSocket server (openpi bare-dict protocol)
# ---------------------------------------------------------------------------


_client_lock = asyncio.Semaphore(1)


async def _handle_client(websocket, server: R1ProInferenceServer):
    if not _client_lock.locked():
        async with _client_lock:
            await _handle_client_inner(websocket, server)
    else:
        logger.warning("Rejected connection from %s: another client is active", websocket.remote_address)
        await websocket.close(1013, "Only one client allowed at a time")


async def _handle_client_inner(websocket, server: R1ProInferenceServer):
    packer = _Packer()
    metadata = {"model": "internvla_a1_5", "version": "r1pro"}
    await websocket.send(packer.pack(metadata))
    logger.info("Client connected from %s", websocket.remote_address)

    try:
        async for raw in websocket:
            obs = _unpackb(raw)
            result = server.predict(obs)
            await websocket.send(packer.pack(result))
    except Exception as e:
        logger.error("Client error: %s", e, exc_info=True)
    finally:
        if server.kpt_tracker:
            server.kpt_tracker.reset()
        logger.info("Client disconnected")


async def _health_handler(path, request_headers):
    if path == "/healthz":
        return (200, [("Content-Type", "text/plain")], b"OK\n")
    return None


async def run_server(server: R1ProInferenceServer, host: str, port: int):
    try:
        import websockets
    except ImportError:
        raise ImportError("pip install websockets")

    handler = functools.partial(_handle_client, server=server)
    async with websockets.serve(handler, host, port, process_request=_health_handler,
                                max_size=None, ping_interval=30, ping_timeout=60):
        logger.info("Listening on ws://%s:%d (healthz at http://%s:%d/healthz)", host, port, host, port)
        await asyncio.Future()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-path", type=str, required=True, help="Checkpoint directory")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--stats-path", type=str, default=None, help="External stats.json path")
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--inference-backend", choices=["standard", "optimized"], default="standard")
    p.add_argument("--kpt-meta-path", type=str, default=None, help="keypoints_meta.json path")
    p.add_argument("--urdf-path", type=str, default=str(REPO_ROOT / "assets" / "r1_pro_with_gripper.urdf"))
    p.add_argument("--no-pad-torso", action="store_true", help="Don't pad torso zeros (19D output)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    ckpt_path = Path(args.ckpt_path)

    policy, device, config = load_policy(args, dtype)

    stats_path = Path(args.stats_path) if args.stats_path else ckpt_path / "stats.json"
    if not stats_path.exists():
        alt = REPO_ROOT / "assets" / "norm_stats" / "abs" / "stats.json"
        if alt.exists():
            stats_path = alt
        else:
            raise FileNotFoundError(f"Stats not found at {stats_path} or {alt}")
    logger.info("Loading stats from %s", stats_path)
    state_stat, action_stat = load_stats(stats_path)

    input_transforms = build_input_transforms(args.resize_size, state_stat, config)

    action_keys = action_stat.get(ACTION, {})
    if "mean" not in action_keys or "std" not in action_keys:
        raise ValueError(
            "stats.json missing 'mean'/'std' for action. "
            "Check if training used min-max normalization instead of mean-std."
        )
    unnormalize_fn = UnNormalizeTransformFn(selected_keys=[ACTION], mode="mean_std", norm_stats=action_stat)

    schema = get_schema("r1_pro")
    inv_reorder = compute_inverse_reorder(schema.action_reorder)

    kpt_tracker = None
    if config.enable_keypoint_predictor:
        meta_path = args.kpt_meta_path
        if meta_path is None:
            for candidate in [ckpt_path / "keypoints_meta.json",
                              ckpt_path.parent / "keypoints_meta.json"]:
                if candidate.exists():
                    meta_path = str(candidate)
                    break
        if meta_path is None:
            logger.warning("Keypoint predictor enabled but no keypoints_meta.json found. Disabling keypoints.")
        else:
            h = getattr(config, "keypoint_history_max_len", 300)
            kpt_tracker = R1ProKeypointTracker(args.urdf_path, meta_path, history_max_len=h)

    server = R1ProInferenceServer(
        policy=policy,
        input_transforms=input_transforms,
        unnormalize_fn=unnormalize_fn,
        config=config,
        device=device,
        dtype=dtype,
        inverse_action_reorder=inv_reorder,
        kpt_tracker=kpt_tracker,
        pad_torso=not args.no_pad_torso,
    )

    asyncio.run(run_server(server, args.host, args.port))


if __name__ == "__main__":
    main()
