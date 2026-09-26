"""10-check verification for put_cube_into_box keypoint dataset (V3).

Checks:
  1. Shape: observation.keypoint_3d = [56]
  2. Position bounds: |pos| <= 1.01
  3. Quaternion norm: |q| = 1 ± 0.001
  4. Hemisphere: qw >= 0 (keypoints AND state ee_quat2)
  5. Temporal smoothness (hemisphere-aware)
  6. FK reproducibility
  7. Per-dimension statistics
  8. FK cross-validation vs ee_quat2
  9. keypoints_meta.json completeness
  10. train_inference_contract.json completeness

Usage:
    source /B/VENV/itnvla15rbt20/bin/activate
    python b/s/Frk3/verify_cubinbx_keypoints.py \
        --dataset /home/a26113/b/Dta/put_cube_into_box_lrb3_4D \
        --urdf b/d/Frk2/fr3v2_1_franka_hand.urdf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 8
KEYPOINT_DIM = 7
TOTAL_DIM = NUM_KEYPOINTS * KEYPOINT_DIM
KPT_COL = "observation.keypoint_3d"
STATE_COL = "observation.state"


def check_1_shape(all_kpts):
    """Check keypoint shape is [56]."""
    for i, kpt in enumerate(all_kpts):
        if kpt.shape[-1] != TOTAL_DIM:
            return False, f"Frame {i}: shape {kpt.shape}, expected [..., {TOTAL_DIM}]"
    return True, f"All {len(all_kpts)} frames have shape [56]"


def check_2_bounds(all_kpts):
    """Check position bounds |pos| <= 1.01."""
    kpts = np.stack(all_kpts).reshape(-1, NUM_KEYPOINTS, KEYPOINT_DIM)
    pos = kpts[:, :, :3]
    max_abs = np.abs(pos).max()
    ok = max_abs <= 1.01
    return ok, f"max |pos| = {max_abs:.6f} (limit 1.01)"


def check_3_quat_norm(all_kpts):
    """Check quaternion unit norm."""
    kpts = np.stack(all_kpts).reshape(-1, NUM_KEYPOINTS, KEYPOINT_DIM)
    quat = kpts[:, :, 3:7]
    norms = np.linalg.norm(quat.reshape(-1, 4), axis=1)
    max_err = np.abs(norms - 1.0).max()
    ok = max_err <= 0.001
    return ok, f"max |q|-1 error = {max_err:.6e} (limit 0.001)"


def check_4_hemisphere(all_kpts, all_states):
    """Check qw >= 0 for keypoints AND state ee_quat2."""
    kpts = np.stack(all_kpts).reshape(-1, NUM_KEYPOINTS, KEYPOINT_DIM)
    qw_kpt = kpts[:, :, 6]
    qw_kpt_min = qw_kpt.min()

    states = np.stack(all_states)
    qw_state = states[:, 14]
    qw_state_min = qw_state.min()

    ok_kpt = qw_kpt_min >= 0
    ok_state = qw_state_min >= 0
    ok = ok_kpt and ok_state
    return ok, (f"keypoint qw_min={qw_kpt_min:.6f}, "
                f"state ee_quat2 qw_min={qw_state_min:.6f}")


def check_5_temporal_smoothness(all_kpts, episode_boundaries):
    """Hemisphere-aware temporal smoothness."""
    kpts = np.stack(all_kpts).reshape(-1, NUM_KEYPOINTS, KEYPOINT_DIM)
    quat = kpts[:, :, 3:7]

    max_diff = 0.0
    n_jumps = 0
    threshold = 0.5

    for start, end in episode_boundaries:
        if end - start < 2:
            continue
        ep_quat = quat[start:end]
        for t in range(1, len(ep_quat)):
            q1 = ep_quat[t - 1]
            q2 = ep_quat[t]
            d_pos = np.linalg.norm(q1 - q2, axis=-1)
            d_neg = np.linalg.norm(q1 + q2, axis=-1)
            d = np.minimum(d_pos, d_neg).max()
            max_diff = max(max_diff, d)
            if d > threshold:
                n_jumps += 1

    ok = n_jumps == 0
    return ok, f"max hemi-aware diff = {max_diff:.4f}, jumps>{threshold}: {n_jumps}"


def _make_extractor(urdf_path):
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    from generate_cubinbx_keypoints import FrankaFKExtractor7D
    return FrankaFKExtractor7D(urdf_path)


def check_6_fk_reproducibility(all_kpts, all_states, urdf_path, dataset_path, n_samples=200):
    """Recompute FK on random samples and compare."""
    extractor = _make_extractor(urdf_path)
    kpts = np.stack(all_kpts).reshape(-1, NUM_KEYPOINTS, KEYPOINT_DIM)
    states = np.stack(all_states)

    r_pad = _read_r_pad(Path(dataset_path))
    if r_pad is None:
        return False, "Cannot read R_pad from keypoints_meta.json"

    indices = np.random.choice(len(kpts), min(n_samples, len(kpts)), replace=False)
    max_err = 0.0
    for idx in indices:
        arm = states[idx, 0:7]
        recomputed = extractor.compute(arm)
        kpt_stored = kpts[idx].copy()
        kpt_stored[:, :3] *= r_pad
        diff = np.abs(recomputed - kpt_stored).max()
        max_err = max(max_err, diff)

    ok = max_err < 1e-4
    return ok, f"max FK reproducibility error = {max_err:.2e}"


def check_7_stats(all_kpts):
    """Per-dimension statistics."""
    kpts = np.stack(all_kpts).reshape(-1, NUM_KEYPOINTS, KEYPOINT_DIM)
    for j in range(NUM_KEYPOINTS):
        for d in range(KEYPOINT_DIM):
            vals = kpts[:, j, d]
            dim_name = ["px", "py", "pz", "qx", "qy", "qz", "qw"][d]
            logger.info(f"    KP{j} {dim_name}: min={vals.min():.4f} "
                        f"max={vals.max():.4f} mean={vals.mean():.4f} "
                        f"std={vals.std():.4f}")
    return True, "Statistics logged above"


def check_8_cross_val(all_states, urdf_path, n_samples=500):
    """FK cross-validation vs ee_quat2 in state[11:15]."""
    extractor = _make_extractor(urdf_path)

    states = np.stack(all_states)
    indices = np.random.choice(len(states), min(n_samples, len(states)), replace=False)

    pos_errs = []
    rot_errs = []
    for idx in indices:
        arm = states[idx, 0:7]
        kpt = extractor.compute(arm)
        fk_pos = kpt[-1, :3]
        fk_quat = kpt[-1, 3:7]
        ds_pos = states[idx, 8:11]
        ds_quat = states[idx, 11:15]

        pos_errs.append(np.linalg.norm(fk_pos - ds_pos))
        dot = min(abs(np.dot(fk_quat, ds_quat)), 1.0)
        rot_errs.append(2 * np.arccos(dot))

    pos_max = max(pos_errs) * 1000
    rot_max = np.degrees(max(rot_errs))
    ok = pos_max < 2.0 and rot_max < 1.0
    return ok, f"pos_err_max={pos_max:.6f}mm, rot_err_max={rot_max:.6f}°"


def check_9_meta(dataset_path):
    """keypoints_meta.json completeness."""
    meta_path = Path(dataset_path) / "meta" / "keypoints_meta.json"
    if not meta_path.exists():
        return False, f"Missing {meta_path}"
    meta = json.loads(meta_path.read_text())
    required = ["bbox_radius", "keypoint_dim", "num_keypoints",
                 "keypoint_links", "fk_cross_validation"]
    missing = [k for k in required if k not in meta]
    if missing:
        return False, f"Missing keys: {missing}"
    return True, f"All required fields present, R_pad={meta['bbox_radius']}"


def check_10_contract(dataset_path):
    """train_inference_contract.json completeness."""
    path = Path(dataset_path) / "meta" / "train_inference_contract.json"
    if not path.exists():
        return False, f"Missing {path}"
    contract = json.loads(path.read_text())
    required = ["version", "dataset_fps", "r_pad", "keypoint_links",
                 "hemisphere_convention", "his_len_semantics",
                 "ee_quat_source", "ee_quat_convention"]
    missing = [k for k in required if k not in contract]
    if missing:
        return False, f"Missing keys: {missing}"
    return True, f"v{contract['version']}, fps={contract['dataset_fps']}, R_pad={contract['r_pad']}"


def _read_r_pad(dataset_path):
    meta_path = Path(dataset_path) / "meta" / "keypoints_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())["bbox_radius"]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--urdf", required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    if not parquet_files:
        logger.error("No parquet files in %s", data_dir)
        sys.exit(1)

    logger.info("Loading data from %d parquet files...", len(parquet_files))
    all_kpts = []
    all_states = []
    episode_boundaries = []
    offset = 0

    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path)
        if KPT_COL not in df.columns:
            logger.error("Column '%s' not found in %s", KPT_COL, pq_path)
            sys.exit(1)

        kpts = list(df[KPT_COL].values)
        states = list(np.stack(df[STATE_COL].values))
        n = len(kpts)

        ep_indices = df["episode_index"].values if "episode_index" in df.columns else None
        if ep_indices is not None:
            unique_eps = np.unique(ep_indices)
            for ep in unique_eps:
                mask = ep_indices == ep
                first = offset + np.where(mask)[0][0]
                last = offset + np.where(mask)[0][-1] + 1
                episode_boundaries.append((first, last))
        else:
            episode_boundaries.append((offset, offset + n))

        all_kpts.extend(kpts)
        all_states.extend(states)
        offset += n

    logger.info("Total frames: %d, episodes: %d", len(all_kpts), len(episode_boundaries))

    all_kpts_np = [np.array(k, dtype=np.float32) for k in all_kpts]
    all_states_np = [np.array(s, dtype=np.float32) for s in all_states]

    checks = [
        ("Check 1: Shape", lambda: check_1_shape(all_kpts_np)),
        ("Check 2: Position bounds", lambda: check_2_bounds(all_kpts_np)),
        ("Check 3: Quaternion norm", lambda: check_3_quat_norm(all_kpts_np)),
        ("Check 4: Hemisphere", lambda: check_4_hemisphere(all_kpts_np, all_states_np)),
        ("Check 5: Temporal smoothness", lambda: check_5_temporal_smoothness(all_kpts_np, episode_boundaries)),
        ("Check 6: FK reproducibility", lambda: check_6_fk_reproducibility(all_kpts_np, all_states_np, args.urdf, args.dataset)),
        ("Check 7: Per-dim stats", lambda: check_7_stats(all_kpts_np)),
        ("Check 8: FK cross-val vs ee_quat2", lambda: check_8_cross_val(all_states_np, args.urdf)),
        ("Check 9: keypoints_meta.json", lambda: check_9_meta(args.dataset)),
        ("Check 10: contract.json", lambda: check_10_contract(args.dataset)),
    ]

    results = []
    for name, fn in checks:
        logger.info("--- %s ---", name)
        try:
            ok, msg = fn()
            status = "PASS" if ok else "FAIL"
            logger.info("  %s: %s", status, msg)
            results.append((name, status, msg))
        except Exception as e:
            logger.error("  ERROR: %s", e)
            results.append((name, "ERROR", str(e)))

    logger.info("\n=== SUMMARY ===")
    all_pass = True
    for name, status, msg in results:
        marker = "✓" if status == "PASS" else "✗"
        logger.info("  %s %s: %s — %s", marker, name, status, msg)
        if status != "PASS":
            all_pass = False

    if all_pass:
        logger.info("\n🎉 All 10 checks PASSED!")
    else:
        logger.error("\n❌ Some checks FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
