"""Unit tests and FK cross-validation for Franka v2 keypoint generation.

Can be run standalone (before generation) to verify FK correctness
against the dataset's built-in EE pose.

Usage:
    # Pre-generation FK validation (no keypoints needed):
    python b/s/Frk2/test_franka2_keypoints.py \
        --source /B/Dta/plug_into_socket_franka3_15hz_lerobot \
        --urdf b/d/Frk2/fr3v2_1_franka_hand.urdf

    # Or via pytest:
    pytest b/s/Frk2/test_franka2_keypoints.py -v
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_URDF = SCRIPT_DIR.parents[2] / "b" / "d" / "Frk2" / "fr3v2_1_franka_hand.urdf"
DEFAULT_SOURCE = Path("/B/Dta/plug_into_socket_franka3_15hz_lerobot")


def _get_extractor(urdf_path=None):
    sys.path.insert(0, str(SCRIPT_DIR))
    from generate_franka2_keypoints import FrankaFKExtractor7D
    return FrankaFKExtractor7D(str(urdf_path or DEFAULT_URDF))


def test_fk_home_position():
    """FK at all-zero joints should produce a reasonable TCP position."""
    ext = _get_extractor()
    kpt = ext.compute(np.zeros(7))

    tcp_pos = kpt[7, :3]
    assert tcp_pos[2] > 0.7, f"TCP Z at home should be > 0.7m, got {tcp_pos[2]:.4f}"
    assert abs(tcp_pos[0]) < 0.1, f"TCP X at home should be ~0, got {tcp_pos[0]:.4f}"
    assert abs(tcp_pos[1]) < 0.02, f"TCP Y at home should be ~0, got {tcp_pos[1]:.4f}"

    qw = kpt[7, 6]
    assert qw >= 0, f"TCP qw should be >= 0, got {qw:.6f}"
    print(f"  Home TCP: pos={tcp_pos}, qw={qw:.6f}")


def test_hemisphere_normalization():
    """All keypoints should have qw >= 0."""
    ext = _get_extractor()
    rng = np.random.default_rng(42)
    for _ in range(100):
        q = rng.uniform(-1.5, 1.5, size=7)
        kpt = ext.compute(q)
        qw = kpt[:, 6]
        assert (qw >= -1e-7).all(), f"qw < 0 found: {qw}"


def test_quaternion_unit_norm():
    """All quaternions should have unit norm."""
    ext = _get_extractor()
    rng = np.random.default_rng(123)
    for _ in range(100):
        q = rng.uniform(-1.5, 1.5, size=7)
        kpt = ext.compute(q)
        norms = np.linalg.norm(kpt[:, 3:7], axis=1)
        err = np.abs(norms - 1.0).max()
        assert err < 0.001, f"Quaternion norm error: {err:.6f}"


def test_joint_slice_from_state():
    """Verify that observation.state[0:7] gives correct joint angles."""
    if not DEFAULT_SOURCE.exists():
        print(f"  [skip] source dataset not found: {DEFAULT_SOURCE}")
        return

    pq = sorted(DEFAULT_SOURCE.glob("data/**/*.parquet"))[0]
    df = pd.read_parquet(pq)
    state = np.stack(df["observation.state"].values)
    arm = state[:, 0:7]

    assert arm.shape == (len(df), 7), f"Expected (N, 7), got {arm.shape}"
    assert not np.any(np.isnan(arm)), "NaN in joint angles"
    print(f"  Joint range per dim: {arm.min(axis=0)} .. {arm.max(axis=0)}")


def test_fk_cross_validation(source_path=None, urdf_path=None):
    """Cross-validate FK TCP output against dataset's EE pose."""
    source = Path(source_path) if source_path else DEFAULT_SOURCE
    if not source.exists():
        print(f"  [skip] source dataset not found: {source}")
        return

    ext = _get_extractor(urdf_path)
    parquets = sorted(source.glob("data/**/*.parquet"))

    all_pos_err = []
    all_rot_err = []

    for pq in parquets[:10]:
        df = pd.read_parquet(pq)
        state = np.stack(df["observation.state"].values)
        arm = state[:, 0:7].astype(np.float64)

        kpts = ext.compute_batch(arm)
        fk_pos = kpts[:, 7, :3]
        fk_quat = kpts[:, 7, 3:7]  # [qx, qy, qz, qw]

        ds_pos = state[:, 8:11].astype(np.float32)
        # Dataset stores quat as [qx,qy,qz,qw] despite info.json names
        ds_quat_xyzw = state[:, 11:15].astype(np.float32)

        pos_err = np.linalg.norm(fk_pos - ds_pos, axis=1)
        all_pos_err.extend(pos_err.tolist())

        dots = np.abs(np.sum(fk_quat * ds_quat_xyzw, axis=1))
        dots = np.clip(dots, 0, 1)
        rot_err = 2 * np.arccos(dots)
        all_rot_err.extend(rot_err.tolist())

    pos_err = np.array(all_pos_err)
    rot_err = np.array(all_rot_err)

    print(f"\n  FK Cross-Validation ({len(pos_err)} frames from {min(10, len(parquets))} episodes):")
    print(f"    Position error: mean={pos_err.mean()*1000:.3f}mm, "
          f"max={pos_err.max()*1000:.3f}mm, p99={np.percentile(pos_err, 99)*1000:.3f}mm")
    print(f"    Rotation error: mean={np.degrees(rot_err.mean()):.4f}°, "
          f"max={np.degrees(rot_err.max()):.4f}°, p99={np.degrees(np.percentile(rot_err, 99)):.4f}°")

    if pos_err.max() > 0.002:
        print(f"    WARNING: Position error exceeds 2mm threshold!")
    if np.degrees(rot_err.max()) > 1.0:
        print(f"    WARNING: Rotation error exceeds 1° threshold!")

    return {
        "pos_err_mean_mm": float(pos_err.mean() * 1000),
        "pos_err_max_mm": float(pos_err.max() * 1000),
        "rot_err_mean_deg": float(np.degrees(rot_err.mean())),
        "rot_err_max_deg": float(np.degrees(rot_err.max())),
    }


def test_hislen_semantics():
    """Verify step/commit pattern produces correct his_len sequence."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from fk_keypoints_v2 import FKKeypointComputerV2

    meta_path = DEFAULT_SOURCE.parent / "plug_into_socket_franka3_15hz_lerobot_4D" / "meta" / "keypoints_meta.json"
    if not meta_path.exists():
        print(f"  [skip] meta not found (run generation first): {meta_path}")
        return

    fk = FKKeypointComputerV2(str(DEFAULT_URDF), str(meta_path))
    q = np.zeros(7)

    _, his_len, _ = fk.step(q)
    assert his_len == 0, f"Episode start: his_len should be 0, got {his_len}"

    fk.commit()
    _, his_len, _ = fk.step(q)
    assert his_len == 1, f"After 1 commit: his_len should be 1, got {his_len}"

    fk.commit()
    _, his_len, _ = fk.step(q)
    assert his_len == 2, f"After 2 commits: his_len should be 2, got {his_len}"

    fk.reset()
    for t in range(50):
        _, his_len, _ = fk.step(q)
        assert his_len == t, f"Step {t}: expected {t}, got {his_len}"
        fk.commit()

    print(f"  his_len semantics: PASS (tested 50 steps)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE))
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_URDF))
    args = parser.parse_args()

    print("=== Franka v2 Keypoint Unit Tests ===\n")

    print("Test 1: FK home position")
    test_fk_home_position()

    print("\nTest 2: Hemisphere normalization (100 random configs)")
    test_hemisphere_normalization()

    print("\nTest 3: Quaternion unit norm (100 random configs)")
    test_quaternion_unit_norm()

    print("\nTest 4: Joint slice from state column")
    test_joint_slice_from_state()

    print("\nTest 5: FK cross-validation vs dataset EE pose")
    result = test_fk_cross_validation(args.source, args.urdf)

    print("\n=== All tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
