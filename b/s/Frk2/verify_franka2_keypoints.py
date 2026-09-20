"""Post-generation verification for Franka v2 7D keypoints.

10-check framework adapted for the v2 dataset format with unified
observation.state column and FK cross-validation against built-in EE pose.

Usage:
    python b/s/Frk2/verify_franka2_keypoints.py \
        --dataset /B/Dta/plug_into_socket_franka3_15hz_lerobot_4D \
        --urdf b/d/Frk2/fr3v2_1_franka_hand.urdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NUM_KEYPOINTS = 8
KEYPOINT_DIM = 7
TOTAL_DIM = NUM_KEYPOINTS * KEYPOINT_DIM
STATE_COLUMN = "observation.state"
ARM_SLICE = slice(0, 7)
EE_POS_SLICE = slice(8, 11)
EE_QUAT_SLICE = slice(11, 15)


def load_all(dataset: Path):
    files = sorted(dataset.glob("data/**/*.parquet"))
    if not files:
        sys.exit(f"No parquet files under {dataset}/data")
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if "observation.keypoint_3d" not in df.columns:
            sys.exit(f"observation.keypoint_3d not found in {f}")
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    raw = np.stack(full["observation.keypoint_3d"].values)
    kpts = raw.reshape(-1, NUM_KEYPOINTS, KEYPOINT_DIM)
    return kpts, full


def check1_shape(kpts):
    print(f"\n=== Check 1: Shape ===")
    print(f"  Total frames: {kpts.shape[0]}")
    print(f"  Shape per frame: [{kpts.shape[1]}, {kpts.shape[2]}]")
    ok = kpts.shape[1:] == (NUM_KEYPOINTS, KEYPOINT_DIM)
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def check2_position_bounds(kpts):
    print(f"\n=== Check 2: Position bounds ===")
    pos = kpts[:, :, :3]
    pos_max = np.abs(pos).max()
    print(f"  max |position|: {pos_max:.6f} (threshold: 1.01)")
    ok = pos_max <= 1.01
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def check3_quaternion_norm(kpts):
    print(f"\n=== Check 3: Quaternion unit norm ===")
    quat = kpts[:, :, 3:7]
    norms = np.linalg.norm(quat.reshape(-1, 4), axis=1)
    norm_err = np.abs(norms - 1.0)
    print(f"  norm error: mean={norm_err.mean():.2e}, max={norm_err.max():.2e}")
    ok = norm_err.max() <= 0.001
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def check4_hemisphere(kpts):
    print(f"\n=== Check 4: Hemisphere constraint (qw >= 0) ===")
    qw = kpts[:, :, 6]
    qw_min = qw.min()
    violations = (qw < -1e-7).sum()
    print(f"  qw min: {qw_min:.8f}, violations: {violations}")
    ok = violations == 0
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def check5_temporal_smoothness(kpts, full_df):
    print(f"\n=== Check 5: Temporal smoothness (hemisphere-aware) ===")
    episodes = full_df["episode_index"].values
    quat = kpts[:, :, 3:7]
    max_jump = 0.0
    jump_count = 0
    total_transitions = 0
    for ep in np.unique(episodes):
        mask = episodes == ep
        ep_quat = quat[mask]
        if len(ep_quat) < 2:
            continue
        q1 = ep_quat[:-1]
        q2 = ep_quat[1:]
        # hemisphere-aware: min(|q1-q2|, |q1+q2|) handles q = -q
        d_pos = np.linalg.norm(q1 - q2, axis=-1)
        d_neg = np.linalg.norm(q1 + q2, axis=-1)
        diffs = np.minimum(d_pos, d_neg)
        frame_max = diffs.max(axis=1)
        max_jump = max(max_jump, float(frame_max.max()))
        jump_count += int((frame_max > 0.5).sum())
        total_transitions += len(frame_max)
    print(f"  Transitions: {total_transitions}, max jump: {max_jump:.6f}, jumps>0.5: {jump_count}")
    ok = jump_count == 0
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def check6_fk_reproducibility(kpts, full_df, dataset, urdf_path):
    print(f"\n=== Check 6: FK reproducibility ===")
    try:
        import pinocchio  # noqa: F401
    except ImportError:
        print("  [skip] pinocchio not available")
        return True

    meta_path = dataset / "meta" / "keypoints_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    r_pad = meta["bbox_radius"]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_franka2_keypoints import FrankaFKExtractor7D

    extractor = FrankaFKExtractor7D(urdf_path)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(full_df), size=min(20, len(full_df)), replace=False)
    max_err = 0.0
    for idx in indices:
        row = full_df.iloc[idx]
        state = np.array(row[STATE_COLUMN], dtype=np.float64)
        arm = state[ARM_SLICE]
        recomputed = extractor.compute(arm)
        recomputed[:, :3] /= r_pad
        stored = kpts[idx]
        err = np.abs(recomputed - stored).max()
        max_err = max(max_err, err)

    print(f"  Max recomputation error (20 random frames): {max_err:.2e}")
    ok = max_err <= 1e-5
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def check7_statistics(kpts):
    print(f"\n=== Check 7: Per-dimension statistics ===")
    flat = kpts.reshape(-1, KEYPOINT_DIM)
    labels = ["px", "py", "pz", "qx", "qy", "qz", "qw"]
    print(f"  {'dim':>4s}  {'mean':>10s}  {'std':>10s}  {'min':>10s}  {'max':>10s}")
    print("  " + "-" * 50)
    for i, label in enumerate(labels):
        col = flat[:, i]
        print(f"  {label:>4s}  {col.mean():+10.6f}  {col.std():10.6f}  "
              f"{col.min():+10.6f}  {col.max():+10.6f}")


def check8_fk_cross_validation(full_df, dataset, urdf_path):
    print(f"\n=== Check 8: FK cross-validation vs dataset EE pose ===")
    try:
        import pinocchio  # noqa: F401
    except ImportError:
        print("  [skip] pinocchio not available")
        return True

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_franka2_keypoints import FrankaFKExtractor7D

    extractor = FrankaFKExtractor7D(urdf_path)

    rng = np.random.default_rng(123)
    indices = rng.choice(len(full_df), size=min(500, len(full_df)), replace=False)

    pos_errs = []
    rot_errs = []
    for idx in indices:
        row = full_df.iloc[idx]
        state = np.array(row[STATE_COLUMN], dtype=np.float64)
        arm = state[ARM_SLICE]
        kpt = extractor.compute(arm)

        fk_tcp_pos = kpt[-1, :3]
        fk_tcp_quat = kpt[-1, 3:7]  # [qx, qy, qz, qw]

        ds_ee_pos = state[EE_POS_SLICE].astype(np.float32)
        # Dataset stores quat as [qx,qy,qz,qw] despite info.json names
        ds_ee_quat_xyzw = state[EE_QUAT_SLICE].astype(np.float32)

        pos_err = np.linalg.norm(fk_tcp_pos - ds_ee_pos)
        pos_errs.append(pos_err)

        dot = abs(np.dot(fk_tcp_quat, ds_ee_quat_xyzw))
        dot = min(dot, 1.0)
        rot_err = 2 * np.arccos(dot)
        rot_errs.append(rot_err)

    pos_errs = np.array(pos_errs)
    rot_errs = np.array(rot_errs)

    print(f"  Position error: mean={pos_errs.mean()*1000:.3f}mm, "
          f"max={pos_errs.max()*1000:.3f}mm (threshold: 2.0mm)")
    print(f"  Rotation error: mean={np.degrees(rot_errs.mean()):.4f}°, "
          f"max={np.degrees(rot_errs.max()):.4f}° (threshold: 1.0°)")

    ok_pos = pos_errs.max() < 0.002
    ok_rot = np.degrees(rot_errs.max()) < 1.0
    ok = ok_pos and ok_rot
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def check9_meta_completeness(dataset):
    print(f"\n=== Check 9: keypoints_meta.json completeness ===")
    meta_path = dataset / "meta" / "keypoints_meta.json"
    if not meta_path.exists():
        print(f"  FAIL: {meta_path} not found")
        return False
    with open(meta_path) as f:
        meta = json.load(f)

    required = ["bbox_radius", "keypoint_dim", "num_keypoints", "keypoint_links",
                "total_frames", "urdf", "dataset_format_version"]
    missing = [k for k in required if k not in meta]
    if missing:
        print(f"  FAIL: missing keys: {missing}")
        return False
    print(f"  All required keys present. R_pad={meta['bbox_radius']:.6f}, "
          f"keypoints={meta['num_keypoints']}x{meta['keypoint_dim']}D")
    print(f"  PASS")
    return True


def check10_contract(dataset):
    print(f"\n=== Check 10: train_inference_contract.json completeness ===")
    contract_path = dataset / "meta" / "train_inference_contract.json"
    if not contract_path.exists():
        print(f"  FAIL: {contract_path} not found")
        return False
    with open(contract_path) as f:
        contract = json.load(f)

    required = ["version", "dataset_fps", "urdf_sha256", "r_pad",
                "inference_constraints", "keypoint_links"]
    missing = [k for k in required if k not in contract]
    if missing:
        print(f"  FAIL: missing keys: {missing}")
        return False
    print(f"  Contract v{contract['version']}, fps={contract['dataset_fps']}, "
          f"R_pad={contract['r_pad']:.6f}")
    print(f"  PASS")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--urdf", type=str,
                        default=str(Path(__file__).resolve().parents[3] / "b" / "d" / "Frk2" / "fr3v2_1_franka_hand.urdf"))
    args = parser.parse_args()

    dataset = Path(args.dataset)
    print(f"Verifying Franka v2 7D keypoints in: {dataset}")

    kpts, full_df = load_all(dataset)
    ok = check1_shape(kpts)
    ok &= check2_position_bounds(kpts)
    ok &= check3_quaternion_norm(kpts)
    ok &= check4_hemisphere(kpts)
    ok &= check5_temporal_smoothness(kpts, full_df)
    ok &= check6_fk_reproducibility(kpts, full_df, dataset, args.urdf)
    check7_statistics(kpts)
    ok &= check8_fk_cross_validation(full_df, dataset, args.urdf)
    ok &= check9_meta_completeness(dataset)
    ok &= check10_contract(dataset)

    print(f"\n=== Summary: {'ALL PASS' if ok else 'SOME CHECKS FAILED'} ===")
    print(f"  Frames: {len(kpts)}, Keypoints: {NUM_KEYPOINTS} x {KEYPOINT_DIM}D = {TOTAL_DIM}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
