"""History window with the same layout as Extract3DKeypointTransformFn.

Training does not store a history tensor. ``LeRobotDataset`` gathers
``observation.keypoint_3d`` at offsets ``[-H, ..., -1, 0, 1, ..., C]`` and the
transform packs the valid past frames at the front of ``his_kpts``, zeros at
the back. ``his_len`` counts frames strictly before the current index.

Eval must reproduce that packing. The current frame is committed only after
the policy request, so it becomes history for the next control step. Wait
steps that are not part of the demonstration must not be committed.
"""

from __future__ import annotations

import numpy as np

from contract import CHUNK_SIZE, HISTORY_MAX_LEN, KEYPOINT_DIM, NUM_KEYPOINTS


class KeypointHistory:
    """Oldest-first, zero-padded at the back. Capacity H."""

    def __init__(self, max_len: int = HISTORY_MAX_LEN, num_kpts: int = NUM_KEYPOINTS, kpt_dim: int = KEYPOINT_DIM):
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")
        self.max_len = max_len
        self.num_kpts = num_kpts
        self.kpt_dim = kpt_dim
        self.reset()

    def reset(self) -> None:
        self._buffer = np.zeros((self.max_len, self.num_kpts, self.kpt_dim), dtype=np.float32)
        self._len = 0

    def push(self, kpt: np.ndarray) -> None:
        frame = np.asarray(kpt, dtype=np.float32)
        if frame.shape != (self.num_kpts, self.kpt_dim):
            raise ValueError(f"expected keypoint {(self.num_kpts, self.kpt_dim)}, got {frame.shape}")
        if self._len < self.max_len:
            self._buffer[self._len] = frame
            self._len += 1
        else:
            self._buffer[:-1] = self._buffer[1:]
            self._buffer[-1] = frame

    @property
    def his_len(self) -> int:
        return self._len

    def get_history(self) -> tuple[np.ndarray, int]:
        return self._buffer.copy(), self._len


def pack_like_training(episode_kpts: np.ndarray, t: int, history_len: int, chunk_size: int = CHUNK_SIZE) -> dict:
    """Slice one stored trajectory the way the training transform does.

    ``episode_kpts`` is ``[T, J, D]`` in stored (already / R_pad) units.
    Out-of-range past offsets are padding and do not enter ``his_kpts``.
    Out-of-range future offsets clamp to the last frame, matching
    ``LeRobotDataset`` query clamping. ``his_len`` does not count those clamps.
    """
    kpts = np.asarray(episode_kpts, dtype=np.float32)
    if kpts.ndim != 3:
        raise ValueError(f"expected [T,J,D], got {kpts.shape}")
    length, joints, dim = kpts.shape
    if t < 0 or t >= length:
        raise IndexError(f"t={t} outside episode length {length}")

    his = np.zeros((history_len, joints, dim), dtype=np.float32)
    # Valid past frames are [max(0, t-H), t).
    start = max(0, t - history_len)
    past = kpts[start:t]
    his_len = past.shape[0]
    if his_len:
        his[:his_len] = past

    future = np.empty((chunk_size, joints, dim), dtype=np.float32)
    for i in range(chunk_size):
        future[i] = kpts[min(length - 1, t + 1 + i)]
    return {
        "his_kpts": his,
        "his_len": his_len,
        "kpt_t": kpts[t].copy(),
        "kpt_future": future,
    }


class GoalKeypointRuntime:
    """Per-episode eval clock. One commit per control step, never during wait.

    Call ``begin_step`` to read the history that training would see at this
    index, send that payload with the image, then ``commit_step`` so the
    current pose becomes history for the next index. ``commit_step`` runs on
    every control step, including steps that reuse an action chunk.
    """

    def __init__(self, fk, r_pad: float, history_len: int = HISTORY_MAX_LEN):
        if r_pad <= 0:
            raise ValueError("r_pad must come from keypoints_meta.json and be positive")
        self.fk = fk
        self.r_pad = float(r_pad)
        self.history = KeypointHistory(max_len=history_len)
        self._pending: np.ndarray | None = None

    def reset(self) -> None:
        self.history.reset()
        self._pending = None

    def begin_step(self, qpos9: np.ndarray) -> dict[str, np.ndarray | int]:
        """History is the past. Current keypoints are computed but not pushed."""
        if self._pending is not None:
            raise RuntimeError("begin_step called twice without commit_step")
        current = self.fk.keypoints(qpos9, self.r_pad)
        self._pending = current
        his, length = self.history.get_history()
        return {"kpt_history": his, "his_len": length, "kpt_t": current}

    def commit_step(self) -> None:
        if self._pending is None:
            raise RuntimeError("commit_step without begin_step")
        self.history.push(self._pending)
        self._pending = None

    def server_example_fields(self) -> dict:
        """Fields the existing LIBERO2 server already reads: kpt_history, his_len."""
        if self._pending is None:
            raise RuntimeError("server fields are only valid between begin_step and commit_step")
        his, length = self.history.get_history()
        fields: dict = {"his_len": int(length)}
        if length > 0:
            fields["kpt_history"] = his
        return fields
