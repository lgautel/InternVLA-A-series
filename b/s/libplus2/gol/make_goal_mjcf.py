#!/usr/bin/env python3
"""Derive assets/panda_goal_table.xml from the robosuite Lift export.

Why this file exists
--------------------
The eval client already builds its FK from ``StandaloneFK(mjcf_path=..., r_pad=...)``
and reads bodies ``robot0_link1..7`` + ``gripper0_eef`` out of a *full* robosuite
export. ``evaluation/panda_robosuite_lift.xml`` is that export, but

  * its base is the Lift base (-0.56, 0, 0.912), 0.10 m away from the Goal table
    base (-0.66, 0, 0.912), and
  * it references mesh/texture files under ``/B/VENV/libero_plus_client/...`` that
    do not exist on every host, so ``MjModel.from_xml_path`` fails there.

This script keeps the kinematic tree byte-for-byte, drops the ``<asset>`` block and
every mesh/material/texture/mesh-geom reference (they do not affect ``xpos/xquat``),
and rewrites the single ``robot0_base`` position. The result loads anywhere and is
the model the eval runtime shares with the generator's cross-check test.

Usage:  python make_goal_mjcf.py [--check]
        --check  fail if the committed asset differs from a fresh derivation.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from contract import GOAL_BASE_XPOS, GOAL_MJCF, LIFT_MJCF_SOURCE  # noqa: E402

_LIFT_BASE = "-0.56 0 0.912"


def _fmt(vec) -> str:
    return " ".join(f"{float(v):g}" for v in vec)


def derive(source: Path = LIFT_MJCF_SOURCE) -> str:
    text = source.read_text()
    src_md5 = hashlib.md5(text.encode()).hexdigest()

    if text.count(f'<body name="robot0_base" pos="{_LIFT_BASE}"') != 1:
        raise RuntimeError(f"expected exactly one robot0_base at {_LIFT_BASE} in {source}")
    text = text.replace(
        f'<body name="robot0_base" pos="{_LIFT_BASE}"',
        f'<body name="robot0_base" pos="{_fmt(GOAL_BASE_XPOS)}"',
    )
    text = re.sub(r"<asset>.*?</asset>", "", text, flags=re.S)
    text = re.sub(r'\s(?:mesh|material|texture)="[^"]*"', "", text)
    text = re.sub(r'<geom [^>]*type="mesh"[^>]*/>', "", text)
    header = (
        "<!-- Derived by b/s/libplus2/gol/make_goal_mjcf.py. Do not edit by hand.\n"
        f"     source: evaluation/panda_robosuite_lift.xml (md5 {src_md5})\n"
        f"     change: robot0_base pos {_LIFT_BASE} -> {_fmt(GOAL_BASE_XPOS)} (LIBERO Goal table base)\n"
        "     dropped: <asset>, mesh/material/texture attributes, mesh-only geoms -->\n"
    )
    return header + text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    fresh = derive()
    if args.check:
        if not GOAL_MJCF.exists() or GOAL_MJCF.read_text() != fresh:
            print(f"STALE: {GOAL_MJCF} differs from a fresh derivation", file=sys.stderr)
            return 1
        print("OK: panda_goal_table.xml is up to date")
        return 0
    GOAL_MJCF.write_text(fresh)
    print(f"wrote {GOAL_MJCF} ({len(fresh)} bytes, md5 {hashlib.md5(fresh.encode()).hexdigest()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
