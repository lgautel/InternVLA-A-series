"""LIBERO2 model-to-env interface — fixes B1 (gripper convention) + adds keypoint support."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from evaluation.LIBERO.policy_server.tools.websocket_policy_client import WebsocketClientPolicy
from evaluation.LIBERO2.keypoint_utils import (
    DEFAULT_R_PAD,
    KeypointExtractor,
    KeypointHistory,
    StandaloneFK,
    _DEFAULT_MJCF_PATH,
)


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
    """LIBERO env-side adapter with B1 fix (gripper convention) and keypoint support.

    Changes from evaluation/LIBERO/model2libero_interface.py:
      - B1 FIX: gripper_convention parameter ('libero_native' / 'openvla' / 'auto')
      - NEW: keypoint extraction and history tracking via StandaloneFK
    """

    def __init__(
        self,
        host: str,
        port: int,
        rotate_images: bool = False,
        gripper_convention: str = "libero_native",
        replan_steps: int = 8,
        enable_keypoints: bool = True,
        kpt_r_pad: float = DEFAULT_R_PAD,
        kpt_history_max_len: int = 200,
        mjcf_path: str | None = None,
    ) -> None:
        from evaluation.LIBERO2.orientation_contract import enforce_rotate_against_contract

        enforce_rotate_against_contract(bool(rotate_images))
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.metadata = self.client.get_server_metadata()
        self._validate_server_metadata(self.metadata)

        self.protocol_version = str(self.metadata.get("protocol_version", ""))
        self.chunk_size = int(self.metadata.get("chunk_size", 1))
        self.expected_action_dim = int(self.metadata.get("action_dim", 7))
        self.rotate_images = bool(rotate_images)
        self.gripper_convention = gripper_convention

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
        self._resolved_convention: str = gripper_convention

        self.enable_keypoints = enable_keypoints
        if enable_keypoints:
            self._fk = StandaloneFK(
                mjcf_path=mjcf_path or _DEFAULT_MJCF_PATH,
                r_pad=kpt_r_pad,
            )
            self._kpt_extractor: KeypointExtractor | None = None
            self._kpt_history = KeypointHistory(max_len=kpt_history_max_len)
        else:
            self._fk = None
            self._kpt_extractor = None
            self._kpt_history = None

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
                f"Server protocol_version={version} is too old. Require >=2.1."
            )
        if metadata.get("preprocessing_owner") != "server_canonical":
            raise RuntimeError(
                "Server metadata preprocessing_owner must be 'server_canonical'."
            )
        if not bool(metadata.get("deterministic_inference_preprocess", False)):
            raise RuntimeError("Server must enable deterministic_inference_preprocess.")

    def bind_env(self, env) -> None:
        """Call once after creating env (before any episode)."""
        if self.enable_keypoints:
            self._kpt_extractor = KeypointExtractor(env, self._fk)

    def reset(self, task_description: str | None = None) -> None:
        self._chunk = None
        self._step = 0
        self._task_description = task_description
        if self._kpt_history is not None:
            self._kpt_history.reset()

    def push_keypoint(self) -> None:
        """Record the *current* env keypoint as past history.

        Must run **before** ``env.step`` (wait dummy or policy action). Training
        ``Extract3DKeypointTransformFn`` puts offsets ``[-H, ..., -1]`` in
        ``his_kpts`` and the current frame in ``kpt_t``; TrackEncoder at eval
        only reads ``his_kpts``, so pushing after ``env.step`` would leak the
        current pose into history.
        """
        if self._kpt_extractor is not None:
            kpt = self._kpt_extractor.extract()
            self._kpt_history.push(kpt)

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
        if self.gripper_convention == "auto":
            if self._action_low is not None and self._action_low[6] < -0.5:
                self._resolved_convention = "libero_native"
            else:
                self._resolved_convention = "openvla"

    def _request_chunk(self, obs: dict[str, Any], lang: str) -> np.ndarray:
        primary = self._maybe_rotate(np.asarray(obs["agentview_image"], dtype=np.uint8))
        wrist = self._maybe_rotate(np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8))
        state = self._extract_state(obs)

        example: dict[str, Any] = {
            "image": [primary, wrist],
            "lang": lang,
            "task": lang,
            "state": state,
        }

        if self.enable_keypoints and self._kpt_history is not None:
            kpt_history, his_len = self._kpt_history.get_history()
            if his_len > 0:
                example["kpt_history"] = kpt_history
                example["his_len"] = his_len

        payload = {
            "examples": [example],
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
        if action.shape[-1] >= 7:
            convention = self._resolved_convention
            if convention == "openvla":
                action[6] = 1.0 if action[6] < 0.5 else -1.0
            else:  # libero_native
                action[6] = 1.0 if action[6] > 0 else -1.0
        self._step += 1
        return action.astype(np.float32)
