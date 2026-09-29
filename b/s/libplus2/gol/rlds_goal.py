"""Read libero_goal RLDS episodes without TensorFlow.

Reuses the TFRecord decoder already verified on this host. This module does
not modify that decoder.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_READER_DIR = Path("/B/SRC/itvlaGpLibPlus/b/d/libplus/ds/asset")
if str(_READER_DIR) not in sys.path:
    sys.path.insert(0, str(_READER_DIR))

from rlds_reader import _as_bytes_list, _as_float_array, _parse_example, iter_tfrecords  # noqa: E402

DEFAULT_ROOT = Path("/B/Dta/LIBERO/rlds/libero_goal/1.0.0")


@dataclass
class GoalEpisode:
    file_path: str
    perturbation: str
    task_name: str
    instruction: str
    state: np.ndarray       # [T, 8]
    joint_state: np.ndarray  # [T, 7]
    action: np.ndarray       # [T, 7]
    image_jpeg: list[bytes]
    wrist_jpeg: list[bytes]

    @property
    def num_steps(self) -> int:
        return int(self.state.shape[0])

    def qpos9(self) -> np.ndarray:
        return np.concatenate([self.joint_state, self.state[:, 6:8]], axis=1)

    def trajectory_hash(self) -> str:
        blob = self.joint_state.astype("<f4").tobytes() + self.action.astype("<f4").tobytes()
        return hashlib.md5(blob).hexdigest()


def _parse_path(file_path: str) -> tuple[str, str]:
    parts = file_path.split("/")
    try:
        i = parts.index("pro_data")
        pert = parts[i + 1]
        fname = parts[i + 3]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"unexpected file_path: {file_path}") from exc
    task = fname.removesuffix("_demo.hdf5")
    return pert, task


def decode_episode(payload: bytes) -> GoalEpisode:
    raw = _parse_example(payload)
    file_path = _as_bytes_list(raw["episode_metadata/file_path"])[0].decode()
    langs = [b.decode() for b in _as_bytes_list(raw["steps/language_instruction"])]
    if len(set(langs)) != 1:
        raise ValueError(f"instruction changes inside an episode: {file_path}")
    steps = len(langs)
    state = _as_float_array(raw["steps/observation/state"]).reshape(steps, 8)
    joint = _as_float_array(raw["steps/observation/joint_state"]).reshape(steps, 7)
    action = _as_float_array(raw["steps/action"]).reshape(steps, 7)
    pert, task = _parse_path(file_path)
    return GoalEpisode(
        file_path=file_path,
        perturbation=pert,
        task_name=task,
        instruction=langs[0],
        state=state,
        joint_state=joint,
        action=action,
        image_jpeg=_as_bytes_list(raw["steps/observation/image"]),
        wrist_jpeg=_as_bytes_list(raw["steps/observation/wrist_image"]),
    )


def iter_goal_episodes(root: Path = DEFAULT_ROOT, max_episodes: int | None = None):
    seen = 0
    for path in sorted(root.glob("libero_goal-train.tfrecord-*")):
        for payload in iter_tfrecords(path):
            yield decode_episode(payload)
            seen += 1
            if max_episodes is not None and seen >= max_episodes:
                return
