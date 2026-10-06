"""Load goal_train_eval_contract.json — supports schema v1 and v2."""

from __future__ import annotations

import json
from pathlib import Path


def load_goal_contract(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"missing goal contract {p}. Set GOAL4D_CONTRACT to <dataset>/meta/goal_train_eval_contract.json "
            "of the dataset the checkpoint was trained on."
        )
    c = json.loads(p.read_text())
    schema = c.get("schema")
    if schema not in ("goal_train_eval_contract/1", "goal_train_eval_contract/2"):
        raise ValueError(f"{p}: unknown schema {schema!r}")
    if schema == "goal_train_eval_contract/2":
        for key in ("r_pad", "keypoint_history_max_len", "image_orientation",
                     "robot_type", "eval_mjcf_md5", "state_dim", "stats_key"):
            if key not in c:
                raise KeyError(f"{p}: missing {key}")
    else:
        if c.get("coordinate_system") != "libero_goal_table_world":
            raise ValueError(f"{p}: coordinate_system={c.get('coordinate_system')!r}")
        for key in ("r_pad", "keypoint_history_max_len", "image_orientation",
                     "robot_type", "eval_mjcf_md5"):
            if key not in c:
                raise KeyError(f"{p}: missing {key}")
    if float(c["r_pad"]) <= 0:
        raise ValueError(f"{p}: r_pad must be positive")
    return c
