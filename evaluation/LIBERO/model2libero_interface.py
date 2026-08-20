from __future__ import annotations

import math
from typing import Any

import numpy as np

from evaluation.LIBERO.policy_server.tools.websocket_policy_client import WebsocketClientPolicy


def _patch_torch_load_for_libero_init_states() -> None:
    """LIBERO init_files are numpy arrays pickled via torch.save.

    PyTorch >= 2.6 defaults ``torch.load(weights_only=True)``, which rejects those
    pickles. Force ``weights_only=False`` unless the caller set it explicitly.
    """
    try:
        import torch
    except ImportError:
        return
    orig = torch.load
    if getattr(orig, "_internvla_libero_patched", False):
        return

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig(*args, **kwargs)

    _load._internvla_libero_patched = True  # type: ignore[attr-defined]
    torch.load = _load


_patch_torch_load_for_libero_init_states()


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32).reshape(-1).copy()
    if q.shape[0] != 4:
        raise ValueError(f"Expected quaternion of length 4, got {q.shape}")
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = math.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * math.acos(q[3])) / den).astype(np.float32)


class LiberoModelClient:
    """Thin LIBERO env-side adapter.

    The websocket server returns already-unnormalized action chunks after
    running the canonical training preprocessing internally. This client only
    handles env-specific work:
      - extract 8-dim state from LIBERO obs (eef_pos[3] + axisangle[3] + gripper_qpos[2])
      - rotate agentview / wrist images 180 deg to match training preprocessing
      - chunk caching
      - optional gripper sign-binarization for LIBERO env's [-1, +1] convention
    """

    def __init__(
        self,
        host: str,
        port: int,
        rotate_images: bool = True,
        binarize_gripper: bool = True,
        replan_steps: int = 8,
    ) -> None:
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.metadata = self.client.get_server_metadata()
        self._validate_server_metadata(self.metadata)

        self.protocol_version = str(self.metadata.get("protocol_version", ""))
        self.chunk_size = int(self.metadata.get("chunk_size", 1))
        self.expected_action_dim = int(self.metadata.get("action_dim", 7))
        self.rotate_images = bool(rotate_images)
        self.binarize_gripper = bool(binarize_gripper)

        replan_steps = int(replan_steps)
        if replan_steps < 1:
            raise ValueError(f"replan_steps must be >= 1, got {replan_steps}")
        if replan_steps > self.chunk_size:
            raise ValueError(
                f"replan_steps ({replan_steps}) must be <= server chunk_size ({self.chunk_size})"
            )
        self.replan_steps = replan_steps

        self._action_low: np.ndarray | None = None
        self._action_high: np.ndarray | None = None
        self._chunk: np.ndarray | None = None
        self._step = 0
        self._task_description: str | None = None

    @staticmethod
    def _parse_version(version: str) -> tuple[int, int]:
        parts = version.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid protocol version '{version}'")
        return int(parts[0]), int(parts[1])

    @classmethod
    def _validate_server_metadata(cls, metadata: dict[str, Any]) -> None:
        version = str(metadata.get("protocol_version", ""))
        major, minor = cls._parse_version(version)
        if (major, minor) < (2, 1):
            raise RuntimeError(
                f"Server protocol_version={version} is too old. Require >=2.1 for canonical preprocessing contract."
            )
        if metadata.get("preprocessing_owner") != "server_canonical":
            raise RuntimeError(
                "Server metadata preprocessing_owner must be 'server_canonical' for train-infer parity."
            )
        if not bool(metadata.get("deterministic_inference_preprocess", False)):
            raise RuntimeError("Server must enable deterministic_inference_preprocess.")

    def reset(self, task_description: str | None = None) -> None:
        self._chunk = None
        self._step = 0
        self._task_description = task_description

    @staticmethod
    def _extract_state(obs: dict[str, Any]) -> np.ndarray:
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
        eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)
        gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
        if eef_pos.shape[0] != 3 or eef_quat.shape[0] != 4 or gripper_qpos.shape[0] != 2:
            raise ValueError(
                "Unexpected LIBERO state shapes: "
                f"eef_pos={eef_pos.shape}, eef_quat={eef_quat.shape}, gripper_qpos={gripper_qpos.shape}"
            )
        axisangle = _quat2axisangle(eef_quat)
        state = np.concatenate([eef_pos, axisangle, gripper_qpos], axis=0)
        return state.astype(np.float32)

    def _maybe_rotate(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if self.rotate_images:
            arr = arr[::-1, ::-1]
        return np.ascontiguousarray(arr)

    def _update_action_space(self, data: dict[str, Any]) -> None:
        space = data.get("action_space", {})
        if not isinstance(space, dict):
            return
        low = space.get("low")
        high = space.get("high")
        if low is None or high is None:
            return
        low_arr = np.asarray(low, dtype=np.float32).reshape(-1)
        high_arr = np.asarray(high, dtype=np.float32).reshape(-1)
        if low_arr.shape != high_arr.shape or low_arr.size == 0:
            return
        self._action_low = low_arr
        self._action_high = high_arr

    def _request_chunk(self, obs: dict[str, Any], lang: str) -> np.ndarray:
        primary = self._maybe_rotate(np.asarray(obs["agentview_image"], dtype=np.uint8))
        wrist = self._maybe_rotate(np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8))
        state = self._extract_state(obs)

        payload = {
            "examples": [
                {
                    "image": [primary, wrist],
                    "lang": lang,
                    "task": lang,
                    "state": state,
                }
            ],
            "do_sample": False,
        }

        response = self.client.predict_action(
            {
                "type": "infer",
                "request_id": f"libero-{self._step}",
                "payload": payload,
            }
        )

        if not response.get("ok", False):
            raise RuntimeError(f"Inference server error: {response}")
        data = response.get("data", {})
        if "actions" not in data:
            raise KeyError(
                f"Server response missing 'data.actions'. protocol={response.get('protocol_version')} keys={list(data.keys())}"
            )
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim != 3:
            raise ValueError(f"Expected actions with ndim=3, got shape={actions.shape}")

        self._update_action_space(data)
        chunk = np.array(actions[0], dtype=np.float32, copy=True)
        if self._action_low is not None and self._action_high is not None:
            dim = min(chunk.shape[-1], self._action_low.shape[-1], self._action_high.shape[-1])
            chunk[:, :dim] = np.clip(chunk[:, :dim], self._action_low[:dim], self._action_high[:dim])
        return chunk.astype(np.float32)

    def step(self, obs: dict[str, Any], lang: str) -> np.ndarray:
        if lang != self._task_description:
            self.reset(lang)
        if self._chunk is None or self._step % self.replan_steps == 0:
            self._chunk = self._request_chunk(obs, lang)

        idx = self._step % self.replan_steps
        action = self._chunk[idx].astype(np.float32)
        if action.shape[-1] >= self.expected_action_dim:
            action = action[: self.expected_action_dim]
        else:
            action = np.pad(action, (0, self.expected_action_dim - action.shape[-1]))
        if self.binarize_gripper and action.shape[-1] >= 7:
            # Dataset uses OpenVLA RLDS convention: action[6] in [0, 1] where
            # 0 = close, 1 = open (verified against gripper qpos in episode 0
            # of libero_goal). The LIBERO env uses +1 = close, -1 = open. So
            # threshold at 0.5 and flip:
            action[6] = 1.0 if action[6] < 0.5 else -1.0
        self._step += 1
        return action.astype(np.float32)
