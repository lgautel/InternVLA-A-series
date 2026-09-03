"""Verify R1 Pro URDF FK against dataset ee_pose ground truth.

Loads the R1 Pro URDF, runs Pinocchio FK with joint angles from the dataset,
and compares the computed gripper link positions with the recorded ee_pose.

`observation.state.{left,right}_ee_pose` is expressed in the **torso_link4 body
frame**, not in base_link: at the sample frame below, base-frame FK sits 1.145 m
away from the recorded pose, while `oMf(torso_link4).actInv(TCP)` matches to
0.034 mm. Both arms hang off torso_link4 via fixed joints, so that residual is
constant no matter what the torso joints do.

Two consequences, and the second one is the reason this docstring exists:

1. This check DOES validate the arm chain -- that `left_arm[0:7]` maps to
   `left_arm_joint1..7` in that order, and that the URDF link geometry is right.
2. This check does NOT and CANNOT validate the assumption that the torso sat at
   zero during data collection. Expressing the TCP in torso_link4's own frame
   cancels the torso transform exactly (see --torso-sweep), so every torso pose
   produces the same 0.034 mm. Settle the torso pose some other way; see
   `b/d/r1pro_migration_design.md` risk 9b.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import pinocchio as pin
except ImportError:
    sys.exit("pinocchio not installed. Run: pip install pin")


URDF_PATH = Path(__file__).resolve().parents[1] / "assets" / "r1_pro_with_gripper.urdf"
MESH_DIR = URDF_PATH.parent


def load_model(urdf_path: Path):
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    return model, data


def print_joint_info(model):
    print(f"Model: nq={model.nq}, nv={model.nv}, njoints={model.njoints}")
    print(f"{'idx':>4s}  {'joint name':<35s}  {'type':>4s}  {'nq':>3s}  {'idx_q':>5s}")
    print("-" * 60)
    for i in range(model.njoints):
        j = model.joints[i]
        print(f"{i:4d}  {model.names[i]:<35s}  {j.shortname():>4s}  {j.nq:3d}  {j.idx_q:5d}")


def print_frame_info(model):
    print(f"\nFrames ({model.nframes}):")
    for i in range(model.nframes):
        f = model.frames[i]
        print(f"  {i:3d}  {f.name:<40s}  parent_joint={f.parentJoint}")


def build_q_vector(model, left_arm, right_arm, torso):
    """Map dataset joint angles to Pinocchio q vector.

    Dataset fields:
      - left_arm: [7] (left_arm_joint1..7)
      - right_arm: [7] (right_arm_joint1..7)
      - torso: [4] (torso_joint1..4)

    URDF also has: steer_motor_joint1..3, wheel_motor_joint1..3,
    left/right_gripper_finger_joint1..2 — set to 0.
    """
    q = np.zeros(model.nq, dtype=np.float64)

    joint_map = {}
    for i in range(model.njoints):
        joint_map[model.names[i]] = i

    for k in range(4):
        name = f"torso_joint{k + 1}"
        if name in joint_map:
            j = model.joints[joint_map[name]]
            q[j.idx_q : j.idx_q + j.nq] = torso[k]

    for k in range(7):
        name = f"left_arm_joint{k + 1}"
        if name in joint_map:
            j = model.joints[joint_map[name]]
            q[j.idx_q : j.idx_q + j.nq] = left_arm[k]

    for k in range(7):
        name = f"right_arm_joint{k + 1}"
        if name in joint_map:
            j = model.joints[joint_map[name]]
            q[j.idx_q : j.idx_q + j.nq] = right_arm[k]

    return q


EE_REFERENCE_FRAME = "torso_link4"


def fk_get_frame_position(model, data, q, frame_name):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    frame_id = model.getFrameId(frame_name)
    if frame_id >= model.nframes:
        raise ValueError(f"Frame '{frame_name}' not found in model")
    return data.oMf[frame_id].translation.copy()


def fk_get_position_in_ee_frame(model, data, q, frame_name):
    """TCP expressed in the same frame the dataset's ee_pose uses."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    ref = data.oMf[model.getFrameId(EE_REFERENCE_FRAME)]
    tcp = data.oMf[model.getFrameId(frame_name)].translation
    return ref.actInv(tcp).copy()


def main():
    parser = argparse.ArgumentParser(description="Verify R1 Pro FK against dataset ee_pose")
    parser.add_argument("--urdf", type=str, default=str(URDF_PATH))
    parser.add_argument("--print-model", action="store_true", help="Print joint/frame info and exit")
    parser.add_argument("--torso-sweep", action="store_true",
                        help="Show that ee_pose is invariant to the torso joints, so it cannot "
                             "be used to confirm the torso-at-zero assumption")
    args = parser.parse_args()

    model, data = load_model(Path(args.urdf))

    if args.print_model:
        print_joint_info(model)
        print_frame_info(model)
        return

    print_joint_info(model)
    print()

    # Test data from open0630_mj_clean episode_000000 frame 50
    left_arm = np.array([0., 0., 0., -1.74744678, 0., 0., 0.], dtype=np.float64)
    right_arm = np.array([2.12765954e-04, -2.12765954e-04, 0., -1.74744678, 0., 0., 0.], dtype=np.float64)
    torso = np.array([0., 0., 0., 0.], dtype=np.float64)

    # Dataset ee_pose: [x, y, z, qx, qy, qz, qw]
    left_ee_pose = np.array([0.41170394, 0.251999, 0.06656582, 0., 0., 0., 1.], dtype=np.float64)
    right_ee_pose = np.array([4.1165274e-01, -2.5205132e-01, 6.6478126e-02, 6.8295340e-05, 0., 0., 1.], dtype=np.float64)

    q = build_q_vector(model, left_arm, right_arm, torso)
    print(f"q vector (nq={model.nq}): {q}")
    print()

    left_ee_xyz = left_ee_pose[:3]
    right_ee_xyz = right_ee_pose[:3]

    print(f"=== FK vs dataset ee_pose (reference frame: {EE_REFERENCE_FRAME}) ===")
    errs = {}
    for side, gt in (("left", left_ee_xyz), ("right", right_ee_xyz)):
        link = f"{side}_gripper_link"
        in_base = fk_get_frame_position(model, data, q, link)
        in_ref = fk_get_position_in_ee_frame(model, data, q, link)
        errs[side] = float(np.linalg.norm(in_ref - gt))
        print(f"{side:>5s}  FK in base_link      : {in_base}")
        print(f"{side:>5s}  FK in {EE_REFERENCE_FRAME:<14s}: {in_ref}")
        print(f"{side:>5s}  dataset ee_pose      : {gt}")
        print(f"{side:>5s}  err = {errs[side] * 1000:.3f} mm "
              f"(vs {np.linalg.norm(in_base - gt) * 1000:.1f} mm if compared in base_link)")
        print()

    worst = max(errs.values())
    if worst < 1e-3:
        print(f"PASS: arm chain matches to {worst * 1000:.3f} mm -- joint ordering and "
              f"link geometry are correct.")
        print("      NOTE: this says nothing about the torso pose; run --torso-sweep.")
    elif worst < 0.01:
        print(f"WARN: {worst * 1000:.1f} mm residual -- check gripper_link vs realsense_link")
    else:
        print(f"FAIL: {worst:.3f} m -- likely a joint ordering mismatch between dataset and URDF")

    if args.torso_sweep:
        print(f"\n=== Torso sweep: ee_pose is blind to the torso ===")
        print(f"{'torso [rad]':<26s} {'TCP in base_link':<34s} {'err in ' + EE_REFERENCE_FRAME}")
        for torso_pose in ([0, 0, 0, 0], [0.3, -0.5, 0.2, 0.0], [-0.4, 0.8, -0.3, 0.5]):
            qs = build_q_vector(model, left_arm, right_arm, np.asarray(torso_pose, dtype=float))
            in_base = fk_get_frame_position(model, data, qs, "left_gripper_link")
            in_ref = fk_get_position_in_ee_frame(model, data, qs, "left_gripper_link")
            err = np.linalg.norm(in_ref - left_ee_xyz) * 1000
            print(f"{str(torso_pose):<26s} {str(np.round(in_base, 4)):<34s} {err:8.3f} mm")
        print("\nThe base-frame TCP moves by >100 mm while the error stays at 0.034 mm.")
        print("So matching ee_pose CANNOT confirm the torso sat at zero -- see risk 9b.")

    # Also print all 16 keypoint positions for inspection
    print("\n=== All 16 keypoint positions (base_link-relative, torso at zero) ===")
    left_links = [f"left_arm_link{i}" for i in range(1, 8)] + ["left_gripper_link"]
    right_links = [f"right_arm_link{i}" for i in range(1, 8)] + ["right_gripper_link"]

    for i, name in enumerate(left_links + right_links):
        pos = fk_get_frame_position(model, data, q, name)
        print(f"  [{i:2d}] {name:<25s}  [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")


if __name__ == "__main__":
    main()
