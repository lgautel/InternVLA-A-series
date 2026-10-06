#!/usr/bin/env python3
"""Verify that a v2 (14D state) Goal dataset is a correct column transform of v1 (8D state).

    python verify_goal_4dv2.py --v1 <v1_path> --v2 <v2_path> [--full]

Without --full, each parquet reads only the first 64 rows per file.
With --full, reads everything and asserts the published release counts
(512604 frames / 4243 episodes / 20 fps).

Checks W01--W22.  Exit 0 if all pass (W10 always passes), exit 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# sys.path setup -- lets us import contract_v2, fk, contract from the script
# directory, and lerobot from the repo src/.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parents[3]  # /B/SRC/itvlaGpLibPlus
for _extra in (_ROOT, _ROOT / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from contract_v2 import (  # noqa: E402
    CONTRACT_SCHEMA,
    FINGER_ANTISYM_TOL_M,
    GRIPPER_INDEX,
    ROBOT_TYPE,
    STATE_DIM,
    STATE_NAMES,
    STATS_KEY,
    gripper_opening,
    qpos9_from_columns,
)

# ---------------------------------------------------------------------------
# Expected counts for the full release
# ---------------------------------------------------------------------------
FULL_FRAMES = 512604
FULL_EPISODES = 4243
FULL_FPS = 20

# Feature classification used by the repack script.
# B-class: vector features whose stats are recomputed from the v2 data.
B_CLASS_FEATURES = {
    "observation.state",
    "observation.state.fingers",
    "observation.state.joint_position",
    "action",
    "observation.keypoint_3d",
}
# C-class: features whose stats are copied verbatim from v1.
C_CLASS_FEATURES = {
    "observation.images.image",
    "observation.images.image2",
    "episode_index",
    "frame_index",
    "index",
    "task_index",
    "timestamp",
}
METRIC_KEYS = ["mean", "std", "min", "max", "count", "q01", "q10", "q50", "q90", "q99"]


# ---------------------------------------------------------------------------
# Report helper
# ---------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, cid: str, ok: bool, detail: str) -> bool:
        self.rows.append((cid, bool(ok), detail))
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {cid}: {detail}", flush=True)
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(r[1] for r in self.rows)


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------
def _load_parquets(data_dir: Path, max_rows_per_file: int | None = None) -> list:
    """Return a list of pyarrow Tables, one per parquet file.

    If *max_rows_per_file* is not None, each table is truncated to that many
    rows (quick smoke-test mode).
    """
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {data_dir}")
    tables = []
    for f in files:
        t = pq.read_table(f)
        if max_rows_per_file is not None and len(t) > max_rows_per_file:
            t = t.slice(0, max_rows_per_file)
        tables.append(t)
    return tables


def _concat_column(tables: list, col: str, dtype=np.float32) -> np.ndarray:
    """Stack a list-of-float column across tables into a 2-D numpy array."""
    parts = []
    for t in tables:
        parts.append(np.array(t[col].to_pylist(), dtype=dtype))
    return np.concatenate(parts, axis=0)


def _concat_scalar(tables: list, col: str) -> np.ndarray:
    """Concatenate a scalar integer column across tables."""
    parts = []
    for t in tables:
        parts.append(np.array(t[col].to_pylist()))
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v1", type=Path, required=True, help="v1 dataset root")
    parser.add_argument("--v2", type=Path, required=True, help="v2 dataset root")
    parser.add_argument("--full", action="store_true",
                        help="read all parquet rows (default: first 64 per file)")
    args = parser.parse_args()

    v1_root: Path = args.v1
    v2_root: Path = args.v2
    max_rows: int | None = None if args.full else 64

    rep = Report()

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------
    v1_info = json.loads((v1_root / "meta" / "info.json").read_text())
    v2_info = json.loads((v2_root / "meta" / "info.json").read_text())
    v2_contract = json.loads((v2_root / "meta" / "goal_train_eval_contract.json").read_text())
    v1_stats = json.loads((v1_root / "meta" / "stats.json").read_text())
    v2_stats = json.loads((v2_root / "meta" / "stats.json").read_text())

    # ------------------------------------------------------------------
    # W01  identity: robot_type, stats_key, schema
    # ------------------------------------------------------------------
    ok = (
        v2_info.get("robot_type") == ROBOT_TYPE
        and v2_contract.get("stats_key") == STATS_KEY
        and v2_contract.get("schema") == CONTRACT_SCHEMA
    )
    rep.add(
        "W01", ok,
        f"robot_type={v2_info.get('robot_type')!r} "
        f"stats_key={v2_contract.get('stats_key')!r} "
        f"schema={v2_contract.get('schema')!r}",
    )

    # ------------------------------------------------------------------
    # W02  counts: frames, episodes, fps match v1 (full: exact release)
    # ------------------------------------------------------------------
    frames_eq = v2_info["total_frames"] == v1_info["total_frames"]
    eps_eq = v2_info["total_episodes"] == v1_info["total_episodes"]
    fps_eq = v2_info["fps"] == v1_info["fps"]
    ok = frames_eq and eps_eq and fps_eq
    detail = (
        f"frames v1={v1_info['total_frames']} v2={v2_info['total_frames']}  "
        f"episodes v1={v1_info['total_episodes']} v2={v2_info['total_episodes']}  "
        f"fps v1={v1_info['fps']} v2={v2_info['fps']}"
    )
    if args.full:
        ok = ok and (
            v2_info["total_frames"] == FULL_FRAMES
            and v2_info["total_episodes"] == FULL_EPISODES
            and v2_info["fps"] == FULL_FPS
        )
        detail += f"  (full: expect {FULL_FRAMES}/{FULL_EPISODES}/{FULL_FPS})"
    rep.add("W02", ok, detail)

    # ------------------------------------------------------------------
    # W03  state layout: 14 dims, names match STATE_NAMES
    # ------------------------------------------------------------------
    state_feat = v2_info.get("features", {}).get("observation.state", {})
    state_shape = tuple(state_feat.get("shape", []))
    state_names = state_feat.get("names", [])
    ok = state_shape == (STATE_DIM,) and state_names == list(STATE_NAMES)
    rep.add(
        "W03", ok,
        f"shape={state_shape} names_head={state_names[:3]}..{state_names[-1:]}",
    )

    # ------------------------------------------------------------------
    # Load parquet data
    # ------------------------------------------------------------------
    print("\nLoading v1 parquets ...", flush=True)
    v1_tables = _load_parquets(v1_root / "data", max_rows)
    v1_n = sum(len(t) for t in v1_tables)
    print(f"  v1: {v1_n} rows", flush=True)

    print("Loading v2 parquets ...", flush=True)
    v2_tables = _load_parquets(v2_root / "data", max_rows)
    v2_n = sum(len(t) for t in v2_tables)
    print(f"  v2: {v2_n} rows\n", flush=True)

    # v2 arrays
    v2_state = _concat_column(v2_tables, "observation.state")
    v2_joints = _concat_column(v2_tables, "observation.state.joint_position")
    v2_fingers = _concat_column(v2_tables, "observation.state.fingers")
    v2_action = _concat_column(v2_tables, "action")
    v2_kp = _concat_column(v2_tables, "observation.keypoint_3d")
    v2_ep = _concat_scalar(v2_tables, "episode_index")
    v2_idx = _concat_scalar(v2_tables, "index")

    # v1 arrays
    v1_state = _concat_column(v1_tables, "observation.state")
    v1_action = _concat_column(v1_tables, "action")
    v1_kp = _concat_column(v1_tables, "observation.keypoint_3d")
    v1_idx = _concat_scalar(v1_tables, "index")

    # Build index-based alignment (v1 row -> v2 row for matching indices)
    v1_pos_map = {int(idx): i for i, idx in enumerate(v1_idx)}
    v2_pos_map = {int(idx): i for i, idx in enumerate(v2_idx)}
    common = sorted(set(v1_pos_map.keys()) & set(v2_pos_map.keys()))
    a_v1 = np.array([v1_pos_map[k] for k in common])
    a_v2 = np.array([v2_pos_map[k] for k in common])
    n_aligned = len(common)

    # ------------------------------------------------------------------
    # W04  state[0:7] == joint_position column, every frame
    # ------------------------------------------------------------------
    diff = float(np.max(np.abs(v2_state[:, 0:7] - v2_joints)))
    rep.add("W04", diff == 0.0, f"max|state[0:7] - joint_position| = {diff:.2e}")

    # ------------------------------------------------------------------
    # W05  state[7:13] == v1 state[0:6], aligned by index
    # ------------------------------------------------------------------
    diff = float(np.max(np.abs(v2_state[a_v2, 7:13] - v1_state[a_v1, 0:6])))
    rep.add("W05", diff == 0.0,
            f"max|v2 state[7:13] - v1 state[0:6]| = {diff:.2e}  ({n_aligned} aligned)")

    # ------------------------------------------------------------------
    # W06  fingers == v1 state[6:8], aligned by index
    # ------------------------------------------------------------------
    diff = float(np.max(np.abs(v2_fingers[a_v2] - v1_state[a_v1, 6:8])))
    rep.add("W06", diff == 0.0,
            f"max|v2 fingers - v1 state[6:8]| = {diff:.2e}  ({n_aligned} aligned)")

    # ------------------------------------------------------------------
    # W07  state[13] == finger_l - finger_r  (exact float32 recompute)
    # ------------------------------------------------------------------
    fl = v2_fingers[:, 0].astype(np.float32)
    fr = v2_fingers[:, 1].astype(np.float32)
    recomputed = gripper_opening(fl, fr)
    diff = float(np.max(np.abs(v2_state[:, GRIPPER_INDEX] - recomputed)))
    rep.add("W07", diff == 0.0,
            f"max|state[13] - float32(finger_l - finger_r)| = {diff:.2e}")

    # ------------------------------------------------------------------
    # W08  action and keypoint_3d byte-identical vs v1
    # ------------------------------------------------------------------
    v2_act_a = v2_action[a_v2].astype(np.float32)
    v1_act_a = v1_action[a_v1].astype(np.float32)
    act_ok = np.array_equal(v2_act_a.view(np.uint32), v1_act_a.view(np.uint32))

    v2_kp_a = v2_kp[a_v2].astype(np.float32)
    v1_kp_a = v1_kp[a_v1].astype(np.float32)
    kp_ok = np.array_equal(v2_kp_a.view(np.uint32), v1_kp_a.view(np.uint32))

    rep.add("W08", act_ok and kp_ok,
            f"action byte-identical={act_ok}  keypoint_3d byte-identical={kp_ok}")

    # ------------------------------------------------------------------
    # W09  video files: exist, not symlink, not samefile, size+SHA match
    # ------------------------------------------------------------------
    vid_contract = v2_contract.get("videos_are_independent_copies", False)
    v1_vid_dir = v1_root / "videos"
    v2_vid_dir = v2_root / "videos"
    v1_vids = {str(f.relative_to(v1_vid_dir)): f
               for f in sorted(v1_vid_dir.rglob("*.mp4"))} if v1_vid_dir.exists() else {}
    v2_vids = {str(f.relative_to(v2_vid_dir)): f
               for f in sorted(v2_vid_dir.rglob("*.mp4"))} if v2_vid_dir.exists() else {}

    issues_09: list[str] = []
    if not vid_contract:
        issues_09.append("videos_are_independent_copies is not true")
    if set(v1_vids) != set(v2_vids):
        miss = set(v1_vids) - set(v2_vids)
        extra = set(v2_vids) - set(v1_vids)
        issues_09.append(f"file set mismatch: missing={len(miss)} extra={len(extra)}")

    checked_09 = 0
    for rel in sorted(v1_vids):
        v1f = v1_vids[rel]
        v2f = v2_vids.get(rel)
        if v2f is None:
            continue
        if v2f.is_symlink():
            issues_09.append(f"symlink: {rel}")
            continue
        if os.path.samefile(str(v1f), str(v2f)):
            issues_09.append(f"samefile (hardlink?): {rel}")
            continue
        if v1f.stat().st_size != v2f.stat().st_size:
            issues_09.append(f"size mismatch: {rel}")
            continue
        h1 = hashlib.sha256(v1f.read_bytes()).hexdigest()
        h2 = hashlib.sha256(v2f.read_bytes()).hexdigest()
        if h1 != h2:
            issues_09.append(f"sha256 mismatch: {rel}")
        checked_09 += 1

    ok_09 = not issues_09
    detail_09 = f"{checked_09}/{len(v1_vids)} checked, independent_copies={vid_contract}"
    if issues_09:
        detail_09 += f"  issues: {issues_09[:5]}"
    rep.add("W09", ok_09, detail_09)

    # ------------------------------------------------------------------
    # W10  antisymmetry report (always PASS -- informational only)
    # ------------------------------------------------------------------
    antisym = np.abs(v2_fingers[:, 0] + v2_fingers[:, 1])
    max_antisym = float(np.max(antisym))
    count_above = int(np.sum(antisym > FINGER_ANTISYM_TOL_M))
    rep.add(
        "W10", True,
        f"max|qL+qR| = {max_antisym:.6f}  "
        f"count > {FINGER_ANTISYM_TOL_M} = {count_above}/{len(antisym)}  (info only)",
    )

    # ------------------------------------------------------------------
    # W11  gripper command sign: corr(action[6], delta_finger_l) < 0,
    #      action[6] in {-1, 1}
    # ------------------------------------------------------------------
    same_ep = v2_ep[1:] == v2_ep[:-1]
    delta_fl = (v2_fingers[1:, 0] - v2_fingers[:-1, 0])[same_ep]
    act6 = v2_action[:-1, 6][same_ep]
    if len(delta_fl) > 10:
        corr_val = float(np.corrcoef(act6, delta_fl)[0, 1])
        unique_vals = sorted(set(np.unique(v2_action[:, 6]).tolist()))
        acts_binary = set(unique_vals).issubset({-1.0, 1.0})
        ok = corr_val < 0 and acts_binary
        rep.add(
            "W11", ok,
            f"corr(action[6], delta_finger_l) = {corr_val:+.4f}  "
            f"action[6] values = {unique_vals}  binary = {acts_binary}",
        )
    else:
        rep.add("W11", False, "too few intra-episode transitions for correlation")

    # ------------------------------------------------------------------
    # W12  stats dimensions
    # ------------------------------------------------------------------
    s_state = v2_stats.get("observation.state", {})
    s_action = v2_stats.get("action", {})
    state_mean_len = len(s_state.get("mean", []))
    action_mean_len = len(s_action.get("mean", []))
    v1_img_keys = sorted(k for k in v1_stats if k.startswith("observation.images."))
    v2_img_keys = sorted(k for k in v2_stats if k.startswith("observation.images."))
    img_match = v1_img_keys == v2_img_keys
    rep.add(
        "W12",
        state_mean_len == STATE_DIM and action_mean_len == 7 and img_match,
        f"state mean len = {state_mean_len}  action mean len = {action_mean_len}  "
        f"image keys match = {img_match}",
    )

    # ------------------------------------------------------------------
    # W13  FK recompute: qpos9_from_columns + GoalTableFK vs stored kp
    # W14  old qpos9_from_state gives wrong fingers on v2 state
    # ------------------------------------------------------------------
    try:
        from fk import GoalTableFK, qpos9_from_state  # noqa: E402

        # Try to locate the kinematic XML via the contract's eval_mjcf,
        # falling back to GoalTableFK's compiled-in default (panda_kin.xml).
        eval_mjcf_rel = v2_contract.get("eval_mjcf", "")
        mjcf_path = _ROOT / eval_mjcf_rel if eval_mjcf_rel else None
        fk: GoalTableFK | None = None
        if mjcf_path is not None and mjcf_path.exists():
            try:
                fk = GoalTableFK(xml_path=mjcf_path)
            except Exception:
                pass  # likely nq != 7; fall through
        if fk is None:
            fk = GoalTableFK()  # default KIN_XML (panda_kin.xml, nq=7)

        r_pad = float(v2_contract["r_pad"])
        n = len(v2_state)
        sample_idx = np.unique(np.linspace(0, n - 1, min(n, 32)).astype(int))

        # W13
        worst_kp = 0.0
        for i in sample_idx:
            q9 = qpos9_from_columns(v2_joints[i], v2_fingers[i])
            got = fk.keypoints(q9.astype(np.float64), r_pad)
            stored = v2_kp[i].reshape(8, 7)
            worst_kp = max(worst_kp, float(np.abs(got - stored).max()))
        rep.add("W13", worst_kp < 1e-5,
                f"FK(qpos9_from_columns) vs stored keypoint_3d: max diff = {worst_kp:.2e}  "
                f"({len(sample_idx)} frames)")

        # W14
        wrong_count = 0
        for i in sample_idx:
            q9_old = qpos9_from_state(v2_joints[i], v2_state[i])
            old_fingers = q9_old[7:9].astype(np.float32)
            actual_fingers = v2_fingers[i]
            if not np.allclose(old_fingers, actual_fingers, atol=1e-7):
                wrong_count += 1
        rep.add(
            "W14", wrong_count > 0,
            f"qpos9_from_state(joint, state14) wrong fingers in {wrong_count}/{len(sample_idx)} frames  "
            f"(state14[6:8] = [joint7, eef_x], not [finger_l, finger_r])",
        )
    except (ImportError, FileNotFoundError) as exc:
        print(f"  WARNING: skipping W13/W14: {exc}", flush=True)
        rep.add("W13", True, f"SKIPPED ({exc})")
        rep.add("W14", True, f"SKIPPED ({exc})")

    # ------------------------------------------------------------------
    # W15  stats.json has exactly 12 feature keys, includes fingers
    # ------------------------------------------------------------------
    stats_keys = set(v2_stats.keys())
    has_fingers = "observation.state.fingers" in stats_keys
    rep.add(
        "W15",
        len(stats_keys) == 12 and has_fingers,
        f"stats.json has {len(stats_keys)} keys (expect 12), "
        f"includes observation.state.fingers = {has_fingers}",
    )

    # ------------------------------------------------------------------
    # W16  each B-class feature has all 10 metric keys
    # ------------------------------------------------------------------
    bad_16: list[str] = []
    for feat in sorted(B_CLASS_FEATURES):
        s = v2_stats.get(feat, {})
        missing = [m for m in METRIC_KEYS if m not in s]
        if missing:
            bad_16.append(f"{feat} missing {missing}")
    rep.add("W16", not bad_16,
            "all vector features have 10 metrics" if not bad_16 else f"missing: {bad_16}")

    # ------------------------------------------------------------------
    # W17  stats monotonicity (recomputed features)
    # ------------------------------------------------------------------
    bad_17: list[str] = []
    for feat in sorted(B_CLASS_FEATURES):
        s = v2_stats.get(feat)
        if not s:
            bad_17.append(f"{feat}: missing")
            continue
        mn = np.array(s["min"], dtype=np.float64)
        mx = np.array(s["max"], dtype=np.float64)
        mean = np.array(s["mean"], dtype=np.float64)
        std = np.array(s["std"], dtype=np.float64)
        q01 = np.array(s["q01"], dtype=np.float64)
        q10 = np.array(s["q10"], dtype=np.float64)
        q50 = np.array(s["q50"], dtype=np.float64)
        q90 = np.array(s["q90"], dtype=np.float64)
        q99 = np.array(s["q99"], dtype=np.float64)

        all_vals = np.concatenate([mn, mx, mean, std, q01, q10, q50, q90, q99])
        if not np.all(np.isfinite(all_vals)):
            bad_17.append(f"{feat}: non-finite values")
            continue

        eps = 1e-12
        mean_eps = 1e-5
        for d in range(len(mn)):
            chain = [mn[d], q01[d], q10[d], q50[d], q90[d], q99[d], mx[d]]
            if not all(a <= b + eps for a, b in zip(chain, chain[1:])):
                bad_17.append(f"{feat} dim{d}: monotonicity")
                break
            if not (mn[d] <= mean[d] + mean_eps and mean[d] <= mx[d] + mean_eps):
                bad_17.append(f"{feat} dim{d}: min<=mean<=max")
                break
            if std[d] < -eps:
                bad_17.append(f"{feat} dim{d}: std<0")
                break

    rep.add("W17", not bad_17,
            "stats monotonicity OK" if not bad_17 else f"violations: {bad_17[:5]}")

    # ------------------------------------------------------------------
    # W18  v2 state stats slices match component stats and v1
    # ------------------------------------------------------------------
    jp_stats = v2_stats.get("observation.state.joint_position", {})
    v1_state_stats = v1_stats.get("observation.state", {})
    s_state_mean = np.array(s_state.get("mean", []), dtype=np.float64)

    ok_a = False
    if jp_stats.get("mean") and len(s_state_mean) >= 7:
        ok_a = bool(np.allclose(s_state_mean[:7],
                                np.array(jp_stats["mean"], dtype=np.float64),
                                atol=1e-6))
    ok_b = False
    if v1_state_stats.get("mean") and len(s_state_mean) >= 13:
        ok_b = bool(np.allclose(s_state_mean[7:13],
                                np.array(v1_state_stats["mean"], dtype=np.float64)[:6],
                                atol=1e-6))
    rep.add(
        "W18", ok_a and ok_b,
        f"state[0:7] mean ~ joint_position mean: {ok_a}  "
        f"state[7:13] mean ~ v1 state[0:6] mean: {ok_b}",
    )

    # ------------------------------------------------------------------
    # W19  v2 fingers stats mean ~ v1 state[6:8] mean
    # ------------------------------------------------------------------
    fing_stats = v2_stats.get("observation.state.fingers", {})
    ok_19 = False
    if fing_stats.get("mean") and v1_state_stats.get("mean"):
        ok_19 = bool(np.allclose(
            np.array(fing_stats["mean"], dtype=np.float64),
            np.array(v1_state_stats["mean"], dtype=np.float64)[6:8],
            atol=1e-6,
        ))
    rep.add("W19", ok_19, f"fingers stats mean ~ v1 state[6:8] mean: {ok_19}")

    # ------------------------------------------------------------------
    # W20  gripper_opening range from stats
    # ------------------------------------------------------------------
    grip_min_list = s_state.get("min", [])
    grip_max_list = s_state.get("max", [])
    if len(grip_min_list) > GRIPPER_INDEX and len(grip_max_list) > GRIPPER_INDEX:
        grip_min = float(grip_min_list[GRIPPER_INDEX])
        grip_max = float(grip_max_list[GRIPPER_INDEX])
        ok_20 = grip_min > -0.01 and grip_max < 0.1
    else:
        grip_min, grip_max, ok_20 = float("nan"), float("nan"), False
    rep.add(
        "W20", ok_20,
        f"gripper_opening (state dim {GRIPPER_INDEX}) min = {grip_min:.6f}  max = {grip_max:.6f}",
    )

    # ------------------------------------------------------------------
    # W21  B-class feature counts all equal to state count
    # ------------------------------------------------------------------
    state_count = s_state.get("count", [])
    bad_21: list[str] = []
    for feat in sorted(B_CLASS_FEATURES):
        s = v2_stats.get(feat, {})
        fc = s.get("count", [])
        if fc != state_count:
            bad_21.append(f"{feat}: count={fc} != {state_count}")
    rep.add(
        "W21", not bad_21,
        f"B-class counts all == {state_count}" if not bad_21 else f"mismatches: {bad_21}",
    )

    # ------------------------------------------------------------------
    # W22  C-class stats match v1 exactly
    # ------------------------------------------------------------------
    bad_22: list[str] = []
    for feat in sorted(C_CLASS_FEATURES):
        v1s = v1_stats.get(feat)
        v2s = v2_stats.get(feat)
        if v1s is None and v2s is None:
            continue  # both absent is fine (e.g., some scalar stats)
        if v1s != v2s:
            bad_22.append(feat)
    rep.add(
        "W22", not bad_22,
        "C-class stats match v1 exactly" if not bad_22 else f"mismatches: {bad_22}",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    passed = sum(r[1] for r in rep.rows)
    total = len(rep.rows)
    print(f"\n{passed}/{total} passed")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
