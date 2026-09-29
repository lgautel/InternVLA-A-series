#!/usr/bin/env python3
"""Run evaluation/LIBERO-plus2/eval_libero_plus.py with the Goal 4D client.

    GOAL4D_CONTRACT=<dataset>/meta/goal_train_eval_contract.json \
    python b/s/libplus2/gol/eval_goal_plus.py -- <all arguments of eval_libero_plus.py>

The LIBERO-plus2 script imports LiberoModelClient lazily inside its forked child
because a top-level ``import mujoco`` breaks fork(). This launcher therefore must
not import the client in the parent either. It installs an import hook that swaps
``LiberoModelClient`` for ``GoalLiberoModelClient`` the moment
``evaluation.LIBERO2.model2libero_interface`` finishes importing in the child.
Nothing in evaluation/ is edited.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]
TARGET = "evaluation.LIBERO2.model2libero_interface"
EVAL_SCRIPT = _ROOT / "evaluation" / "LIBERO-plus2" / "eval_libero_plus.py"


class _SwapClientAfterImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)  # one shot, and avoids recursion in find_spec below
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        original = loader.exec_module

        def exec_module(module):
            original(module)
            if str(_HERE) not in sys.path:
                sys.path.insert(0, str(_HERE))
            import goal_client  # imports the (now complete) base module

            module.LiberoModelClient = goal_client.GoalLiberoModelClient

        loader.exec_module = exec_module
        return spec


def install_hook() -> None:
    if not any(isinstance(f, _SwapClientAfterImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _SwapClientAfterImport())


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    contract = os.environ.get("GOAL4D_CONTRACT")
    if not contract:
        print("GOAL4D_CONTRACT is required", file=sys.stderr)
        return 2
    sys.path.insert(0, str(_HERE))
    from load_contract import load_goal_contract  # pure json, parent-safe

    load_goal_contract(contract)  # fail before any GPU work
    for extra in (str(_ROOT), str(_ROOT / "src")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    assert "mujoco" not in sys.modules, "parent imported mujoco before fork"
    install_hook()
    sys.argv = [str(EVAL_SCRIPT), *argv]
    runpy.run_path(str(EVAL_SCRIPT), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
