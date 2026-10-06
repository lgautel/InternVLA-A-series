#!/usr/bin/env python3
"""v2-compatible wrapper around gol/eval_goal_plus.py.

Overrides load_contract to support goal_train_eval_contract/2 schema,
then delegates everything else to the original eval_goal_plus.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_GOL = _HERE.parent / "gol"

sys.path.insert(0, str(_HERE))
import load_contract_v2  # noqa: E402

sys.path.insert(0, str(_GOL))
import load_contract  # noqa: E402

load_contract.load_goal_contract = load_contract_v2.load_goal_contract

sys.argv[0] = str(_GOL / "eval_goal_plus.py")
exec(open(_GOL / "eval_goal_plus.py").read())
