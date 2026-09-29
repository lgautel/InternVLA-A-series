"""Load the R_pad written by generate_goal_4d.py. No silent fallback."""

from __future__ import annotations

import json
from pathlib import Path


def load_r_pad(keypoints_meta: str | Path) -> float:
    path = Path(keypoints_meta)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}. Eval must use the R_pad of the dataset it was trained on. "
            "Do not substitute the Lift-scene constant 1.8212722539901733."
        )
    payload = json.loads(path.read_text())
    if payload.get("coordinate_system") != "libero_goal_table_world":
        raise ValueError(
            f"{path} coordinate_system={payload.get('coordinate_system')!r}; "
            "expected libero_goal_table_world"
        )
    r_pad = float(payload["bbox_radius"])
    if r_pad <= 0:
        raise ValueError(f"bbox_radius must be positive, got {r_pad}")
    return r_pad


def load_goal_contract(path: str | Path) -> dict:
    """Load meta/goal_train_eval_contract.json written by generate_goal_4d.py.

    Pure JSON, no mujoco, so the eval launcher can call it before fork().
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"missing goal contract {p}. Set GOAL4D_CONTRACT to <dataset>/meta/goal_train_eval_contract.json "
            "of the dataset the checkpoint was trained on."
        )
    c = json.loads(p.read_text())
    if c.get("schema") != "goal_train_eval_contract/1":
        raise ValueError(f"{p}: unknown schema {c.get('schema')!r}")
    if c.get("coordinate_system") != "libero_goal_table_world":
        raise ValueError(f"{p}: coordinate_system={c.get('coordinate_system')!r}")
    for key in ("r_pad", "keypoint_history_max_len", "image_orientation", "robot_type", "eval_mjcf_md5"):
        if key not in c:
            raise KeyError(f"{p}: missing {key}")
    if float(c["r_pad"]) <= 0:
        raise ValueError(f"{p}: r_pad must be positive")
    return c
