#!/usr/bin/env python3
"""Export robosuite Lift MJCF for standalone FK.

Must run in the LIBERO venv (/B/VENV/libero_plus_client/) because robosuite
may attempt GL context init even with has_renderer=False.

Usage:
    /B/VENV/libero_plus_client/bin/python b/s/libplus/export_lift_mjcf.py \
        --output evaluation/panda_robosuite_lift.xml

Outputs:
    1. The Lift scene MJCF XML
    2. A JSON sidecar with body names and model info for verification
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


DEFAULT_OUTPUT = Path("evaluation/panda_robosuite_lift.xml")

EEF_BODY_CANDIDATES = ("gripper0_eef", "gripper0_right_eef")

KEYPOINT_ARM_BODY_NAMES = [
    "robot0_link1", "robot0_link2", "robot0_link3", "robot0_link4",
    "robot0_link5", "robot0_link6", "robot0_link7",
]


def resolve_eef_body_name(model) -> str:
    for name in EEF_BODY_CANDIDATES:
        try:
            model.body(name)
            return name
        except (KeyError, ValueError):
            continue
    raise KeyError(f"No EEF body found (tried {', '.join(EEF_BODY_CANDIDATES)})")


def export_lift_mjcf(output_path: Path) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import robosuite

    env = robosuite.make(
        "Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
    )
    model_xml = env.sim.model.get_xml()
    env.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(model_xml)

    model = mujoco.MjModel.from_xml_path(str(output_path))
    eef_name = resolve_eef_body_name(model)
    body_names = [*KEYPOINT_ARM_BODY_NAMES, eef_name]

    xml_bytes = output_path.read_bytes()
    md5 = hashlib.md5(xml_bytes).hexdigest()

    info = {
        "nq": model.nq, "nv": model.nv, "nbody": model.nbody,
        "eef_body": eef_name,
        "keypoint_bodies": body_names,
        "body_ids": {name: model.body(name).id for name in body_names},
        "md5": md5,
        "robosuite_version": getattr(robosuite, "__version__", "unknown"),
        "mujoco_version": mujoco.__version__,
    }

    sidecar = output_path.with_suffix(".json")
    with open(sidecar, "w") as f:
        json.dump(info, f, indent=2)

    print(f"Exported Lift MJCF to {output_path}")
    print(f"  nq={model.nq}, nv={model.nv}, nbody={model.nbody}")
    print(f"  EEF body: {eef_name}")
    print(f"  MD5: {md5}")
    print(f"  Sidecar: {sidecar}")

    for name in body_names:
        bid = model.body(name).id
        print(f"  {name}: body_id={bid}")

    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_lift_mjcf(args.output)


if __name__ == "__main__":
    main()
