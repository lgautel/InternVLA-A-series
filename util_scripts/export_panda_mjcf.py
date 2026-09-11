#!/usr/bin/env python3
"""Export Franka Panda + gripper MJCF from robosuite for LIBERO FK keypoints.

LIBERO / robosuite use MuJoCo under the hood. Exporting the same scene model
ensures FK keypoints align with the recorded `joint_state` values.

Usage:
    MUJOCO_GL=egl python util_scripts/export_panda_mjcf.py
    MUJOCO_GL=egl python util_scripts/export_panda_mjcf.py --output /tmp/panda_robosuite_full.xml

Requires: robosuite, mujoco
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_OUTPUT = Path("/tmp/panda_robosuite_full.xml")

EEF_BODY_CANDIDATES = ("gripper0_eef", "gripper0_right_eef")

KEYPOINT_BODIES = [
    "robot0_link1",
    "robot0_link2",
    "robot0_link3",
    "robot0_link4",
    "robot0_link5",
    "robot0_link6",
    "robot0_link7",
]


def resolve_eef_body_name(model) -> str:
    for name in EEF_BODY_CANDIDATES:
        try:
            model.body(name)
            return name
        except KeyError:
            continue
    raise KeyError(f"No EEF body found (tried {', '.join(EEF_BODY_CANDIDATES)})")


def export_panda_mjcf(output_path: Path) -> Path:
    os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco
    import robosuite

    env = robosuite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    model_xml = env.sim.model.get_xml()
    env.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(model_xml)

    model = mujoco.MjModel.from_xml_path(str(output_path))
    print(f"Exported Panda MJCF to {output_path}")
    print(f"  nq={model.nq}, nv={model.nv}, nbody={model.nbody}")

    eef_name = resolve_eef_body_name(model)
    for name in [*KEYPOINT_BODIES, eef_name]:
        bid = model.body(name).id
        print(f"  {name}: body_id={bid}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output MJCF path.")
    args = parser.parse_args()
    export_panda_mjcf(args.output)


if __name__ == "__main__":
    main()
