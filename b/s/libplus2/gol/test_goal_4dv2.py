#!/usr/bin/env python3
"""v2 repack acceptance tests for the Goal 14D state layout.

Sections (numbered to match the design doc):
  8.1  Pure functions (gripper_opening, pack_state_v2, qpos9)
  8.2  Small-sample repack (2 episodes, 4 frames each, in temp dir)
  8.3  Schema and delta-transform failure modes
  8.6  Stats computation pure functions
  8.7  Small-sample stats recompute (reuses the mini dataset from 8.2)
  8.8  Meta file integrity (on mini dataset)

Run:
    python -m pytest test_goal_4dv2.py -v
    python test_goal_4dv2.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[3]
for p in (str(_HERE), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from contract_v2 import (
    CONTRACT_SCHEMA,
    FINGER_NAMES,
    GRIPPER_INDEX,
    ROBOT_TYPE,
    STATE_DIM,
    STATE_NAMES,
    gripper_opening,
    pack_state_v2,
    qpos9_from_columns,
)
from repack_goal_4dv2 import compute_vector_stats

PY = sys.executable
REPACK_SCRIPT = str(_HERE / "repack_goal_4dv2.py")

EXPECTED_STAT_METRICS = frozenset(
    ["mean", "std", "min", "max", "count", "q01", "q10", "q50", "q90", "q99"]
)

VECTOR_FEATURES = [
    "observation.state",
    "observation.state.fingers",
    "observation.state.joint_position",
    "action",
    "observation.keypoint_3d",
]


# ============================================================================
# Helper: build a minimal v1 dataset for smoke tests
# ============================================================================

def _build_mini_v1(tmp_root: str) -> tuple[Path, dict]:
    """Create a minimal v1 dataset with 2 episodes x 4 frames = 8 frames.

    Returns (src_path, data_info_dict).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = Path(tmp_root) / "v1_smoke"
    meta = src / "meta"
    meta.mkdir(parents=True)
    episodes_dir = meta / "episodes"
    episodes_dir.mkdir()

    rng = np.random.default_rng(42)
    N = 8  # 2 episodes x 4 frames

    # ---- generate data arrays ----

    # Joint positions (7D), within typical Panda ranges
    joints = rng.uniform(
        [-2.5, -1.5, -2.5, -2.5, -2.5, -0.5, -2.5],
        [2.5, 1.5, 2.5, -0.1, 2.5, 3.5, 2.5],
        size=(N, 7),
    ).astype(np.float32)

    # EEF 6D = position(3) + axis-angle(3)
    eef_pos = rng.uniform([-0.3, -0.3, 0.8], [0.3, 0.3, 1.2], size=(N, 3)).astype(np.float32)
    eef_aa = rng.uniform(-2.5, 2.5, size=(N, 3)).astype(np.float32)

    # Frame 2: set axis-angle magnitude to 3.5 > pi (must survive repack unchanged)
    norm2 = float(np.linalg.norm(eef_aa[2]))
    if norm2 < 1e-8:
        eef_aa[2] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        norm2 = 1.0
    eef_aa[2] = (eef_aa[2] / norm2 * 3.5).astype(np.float32)

    eef6 = np.hstack([eef_pos, eef_aa]).astype(np.float32)

    # Finger positions (antisymmetric within tolerance)
    finger_l = rng.uniform(0.01, 0.04, size=(N,)).astype(np.float32)
    finger_r = -finger_l.copy()

    # v1 state (8D): [eef6, finger_l, finger_r]
    state8 = np.hstack([eef6, finger_l[:, None], finger_r[:, None]]).astype(np.float32)

    # Keypoints (56D = 8 joints x 7 values)
    keypoints = rng.uniform(-1.0, 1.0, size=(N, 56)).astype(np.float32)

    # Actions (7D)
    actions = rng.uniform(-0.1, 0.1, size=(N, 7)).astype(np.float32)

    # Index columns
    episode_index = [0, 0, 0, 0, 1, 1, 1, 1]
    frame_index = [0, 1, 2, 3, 0, 1, 2, 3]
    index_col = list(range(N))
    task_index = [0, 0, 0, 0, 1, 1, 1, 1]
    timestamp = [float(i) * 0.05 for i in range(N)]

    # ---- write parquet ----

    data_dir = src / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    columns = {
        "observation.state": state8.tolist(),
        "observation.state.joint_position": joints.tolist(),
        "observation.keypoint_3d": keypoints.tolist(),
        "action": actions.tolist(),
        "episode_index": episode_index,
        "frame_index": frame_index,
        "index": index_col,
        "task_index": task_index,
        "timestamp": timestamp,
    }
    pq.write_table(pa.table(columns), data_dir / "episode_000000.parquet")

    # ---- write fake video files (1 byte each) ----

    for cam in ["observation.images.image", "observation.images.image2"]:
        cam_dir = src / "videos" / cam
        cam_dir.mkdir(parents=True)
        for ep in range(2):
            (cam_dir / f"episode_{ep:06d}.mp4").write_bytes(b"\x00")

    # ---- compute v1 stats (must be accurate for cross-validation) ----

    v1_stats = {}
    for key, arr in [
        ("observation.state", state8),
        ("observation.state.joint_position", joints),
        ("observation.keypoint_3d", keypoints),
        ("action", actions),
    ]:
        v1_stats[key] = compute_vector_stats(arr.astype(np.float64))

    for key in ["observation.images.image", "observation.images.image2"]:
        dummy = rng.uniform(0.0, 1.0, size=(N, 1))
        v1_stats[key] = compute_vector_stats(dummy)

    for key in ["episode_index", "frame_index", "index", "task_index", "timestamp"]:
        vals = np.array(columns[key], dtype=np.float64).reshape(-1, 1)
        v1_stats[key] = compute_vector_stats(vals)

    with open(meta / "stats.json", "w") as f:
        json.dump(v1_stats, f, indent=2)

    # ---- write v1 info.json ----

    v1_info = {
        "codebase_version": "v3.0",
        "robot_type": "panda",
        "total_episodes": 2,
        "total_frames": N,
        "total_tasks": 2,
        "chunks_size": 1000,
        "fps": 20,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{chunk_idx:03d}/episode_{episode_idx:06d}.parquet",
        "video_path": "videos/{video_key}/episode_{episode_idx:06d}.mp4",
        "data_files_size_in_mb": 0,
        "video_files_size_in_mb": 0,
        "features": {
            "observation.state": {
                "dtype": "float32", "shape": [8], "names": None,
            },
            "observation.state.joint_position": {
                "dtype": "float32", "shape": [7], "names": None,
            },
            "observation.keypoint_3d": {
                "dtype": "float32", "shape": [56], "names": None,
            },
            "observation.images.image": {
                "dtype": "video", "shape": [256, 256, 3], "video_info": {},
            },
            "observation.images.image2": {
                "dtype": "video", "shape": [256, 256, 3], "video_info": {},
            },
            "action": {
                "dtype": "float32", "shape": [7],
                "names": ["dx", "dy", "dz", "dax", "day", "daz", "gripper"],
            },
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with open(meta / "info.json", "w") as f:
        json.dump(v1_info, f, indent=2)

    # ---- write v1 contract ----

    v1_contract = {
        "schema": "goal_train_eval_contract/1",
        "robot_type": "panda",
        "cameras": ["agentview", "wrist"],
        "resize_hw": [256, 256],
        "fps": 20,
        "kpt_4d_mode": "pos_only",
        "num_keypoints": 8,
        "keypoint_history_max_len": 200,
        "chunk_size": 50,
        "r_pad": 1.75,
        "base_xpos_m": [-0.46, 0.0, 0.912],
        "eval_mjcf": "/tmp/fake_eval.xml",
        "eval_mjcf_md5": "d41d8cd98f00b204e9800998ecf8427e",
        "live_eef_tol_m": 0.005,
        "image_orientation": "raw",
        "coordinate_system": "libero_goal_table_world",
    }
    with open(meta / "goal_train_eval_contract.json", "w") as f:
        json.dump(v1_contract, f, indent=2)

    # ---- write per-episode metadata ----

    for ep in range(2):
        with open(episodes_dir / f"episode_{ep:06d}.json", "w") as f:
            json.dump({"episode_index": ep, "task": f"task_{ep}", "length": 4}, f)

    # ---- write tasks.parquet ----

    tasks_tbl = pa.table({
        "task_index": [0, 1],
        "task": ["pick up red block", "put cup on shelf"],
    })
    pq.write_table(tasks_tbl, meta / "tasks.parquet")

    # ---- write goal_episodes.jsonl ----

    with open(meta / "goal_episodes.jsonl", "w") as f:
        for ep in range(2):
            f.write(json.dumps({"episode_index": ep, "goal": f"goal_{ep}"}) + "\n")

    # ---- write keypoints_meta.json ----

    with open(meta / "keypoints_meta.json", "w") as f:
        json.dump({"num_keypoints": 8, "keypoint_dim": 7}, f)

    data_info = {
        "state8": state8,
        "joints": joints,
        "eef6": eef6,
        "finger_l": finger_l,
        "finger_r": finger_r,
        "keypoints": keypoints,
        "actions": actions,
        "v1_info": v1_info,
    }
    return src, data_info


# ===========================================================================
# 8.1  Pure functions
# ===========================================================================


class S81_PureFunctions(unittest.TestCase):
    """gripper_opening, pack_state_v2, qpos9_from_columns, and the old qpos9_from_state."""

    def test_gripper_opening_is_left_minus_right(self):
        g = gripper_opening(np.float32(0.04), np.float32(-0.04))
        self.assertAlmostEqual(float(g), 0.08, places=6)

    def test_gripper_opening_closed_is_near_zero(self):
        g = gripper_opening(np.float32(0.0), np.float32(0.0))
        self.assertAlmostEqual(float(g), 0.0, places=6)

    def test_pack_matches_named_slices(self):
        joint = np.arange(1, 8, dtype=np.float32)
        eef = np.arange(8, 14, dtype=np.float32)
        out = pack_state_v2(joint, eef, np.float32(0.02), np.float32(-0.01))

        np.testing.assert_array_equal(out[0:7], joint)
        np.testing.assert_array_equal(out[7:13], eef)
        self.assertAlmostEqual(float(out[13]), 0.03, places=5)
        self.assertEqual(out.shape, (14,))
        self.assertEqual(out.dtype, np.float32)

    def test_pack_rejects_nan(self):
        joint = np.arange(1, 8, dtype=np.float32)
        joint[3] = np.nan
        eef = np.arange(8, 14, dtype=np.float32)
        with self.assertRaises(ValueError):
            pack_state_v2(joint, eef, np.float32(0.02), np.float32(-0.01))

    def test_qpos9_uses_finger_column(self):
        joint = np.zeros(7, dtype=np.float32)
        fingers = np.array([0.02, -0.02], dtype=np.float32)
        qpos = qpos9_from_columns(joint, fingers)
        # The v2 helper puts the actual finger values into qpos[7:9]
        np.testing.assert_array_equal(qpos[7:9], fingers)
        # In a 14D v2 state, indices [6:8] are NOT the fingers
        eef6 = np.arange(8, 14, dtype=np.float32)
        packed = pack_state_v2(joint, eef6, fingers[0], fingers[1])
        self.assertFalse(
            np.array_equal(qpos[7:9].astype(np.float32), packed[6:8]),
            "packed[6:8] should differ from the actual finger values",
        )

    def test_old_qpos_helper_would_misread_v2(self):
        from fk import qpos9_from_state

        joint = np.ones(7, dtype=np.float32) * 0.5
        eef6 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        fl, fr = np.float32(0.03), np.float32(-0.03)
        state14 = pack_state_v2(joint, eef6, fl, fr)

        # The old helper reads state[6:8] as if they were finger positions
        qpos_old = qpos9_from_state(joint, state14)
        np.testing.assert_array_equal(
            qpos_old[7:9], state14[6:8],
            err_msg="old helper should literally copy state[6:8]",
        )
        # But in v2, state[6]=joint7=0.5 and state[7]=eef_x=0.1, not the fingers
        self.assertFalse(
            np.allclose(qpos_old[7:9], [fl, fr]),
            "old helper would wrongly read eef_x/joint7 as finger positions",
        )


# ===========================================================================
# 8.2 / 8.7 / 8.8  Small-sample repack and its stats / meta checks
# ===========================================================================


class S82_S87_S88_SmokeRepack(unittest.TestCase):
    """Build a mini v1 dataset, repack it, then verify layout, stats, and meta."""

    @classmethod
    def setUpClass(cls):
        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("pyarrow not available")

        cls._tmpdir = tempfile.mkdtemp(prefix="test_goal_4dv2_")
        cls.src, cls.data_info = _build_mini_v1(cls._tmpdir)
        cls.dest = Path(cls._tmpdir) / "v2_smoke"

        result = subprocess.run(
            [PY, REPACK_SCRIPT,
             "--src", str(cls.src),
             "--dest", str(cls.dest),
             "--confirm-full-run"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Repack failed (rc={result.returncode}):\n"
                f"STDOUT:\n{result.stdout[-2000:]}\n"
                f"STDERR:\n{result.stderr[-2000:]}"
            )

        cls.v2_info = json.loads((cls.dest / "meta" / "info.json").read_text())
        cls.v2_stats = json.loads((cls.dest / "meta" / "stats.json").read_text())
        cls.v2_contract = json.loads(
            (cls.dest / "meta" / "goal_train_eval_contract.json").read_text()
        )
        pf = sorted(cls.dest.glob("data/**/*.parquet"))[0]
        cls.v2_table = pq.read_table(pf)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    # ---------------------------------------------------------------
    # 8.2  Repack layout checks
    # ---------------------------------------------------------------

    def test_smoke_repack_layout(self):
        """v2 state is 14D = joint7 + eef6 + gripper_opening, names match STATE_NAMES."""
        import pyarrow.parquet as pq

        d = self.data_info

        # info.json state names
        self.assertEqual(
            self.v2_info["features"]["observation.state"]["names"],
            list(STATE_NAMES),
        )

        # Read v2 parquet
        states_v2 = np.stack(
            self.v2_table["observation.state"].to_pylist()
        ).astype(np.float32)
        fingers_v2 = np.stack(
            self.v2_table["observation.state.fingers"].to_pylist()
        ).astype(np.float32)

        N = len(states_v2)
        self.assertEqual(N, 8)

        for i in range(N):
            # Each frame is 14D
            self.assertEqual(states_v2[i].shape, (STATE_DIM,))

            # state[0:7] == joint_position
            np.testing.assert_array_equal(
                states_v2[i, 0:7], d["joints"][i],
                err_msg=f"frame {i}: state[0:7] != joint",
            )

            # state[7:13] == v1 state[0:6] (eef6, including the 3.5 frame)
            np.testing.assert_array_equal(
                states_v2[i, 7:13], d["eef6"][i],
                err_msg=f"frame {i}: state[7:13] != eef6",
            )

            # state[13] == gripper_opening = finger_l - finger_r
            expected_g = float(d["finger_l"][i]) - float(d["finger_r"][i])
            self.assertAlmostEqual(
                float(states_v2[i, GRIPPER_INDEX]), expected_g, places=6,
                msg=f"frame {i}: gripper opening mismatch",
            )

            # fingers == src state[6:8]
            np.testing.assert_array_equal(
                fingers_v2[i], d["state8"][i, 6:8],
                err_msg=f"frame {i}: fingers != src state[6:8]",
            )

        # The 3.5 axis-angle frame (frame 2) must survive unchanged
        aa_mag = float(np.linalg.norm(states_v2[2, 10:13]))
        self.assertAlmostEqual(aa_mag, 3.5, places=4,
                               msg="axis-angle magnitude >pi must not be normalized")

    def test_smoke_videos_are_independent_copies(self):
        """v2 videos are byte-equal to v1, not hardlinks, not symlinks."""
        v1_video = (
            self.src / "videos" / "observation.images.image" / "episode_000000.mp4"
        )
        v2_video = (
            self.dest / "videos" / "observation.images.image" / "episode_000000.mp4"
        )
        v1_bytes = v1_video.read_bytes()
        v2_bytes = v2_video.read_bytes()

        self.assertEqual(v1_bytes, v2_bytes, "v2 video not byte-equal to v1")
        self.assertFalse(
            os.path.samefile(str(v1_video), str(v2_video)),
            "v2 video is the same file (hardlink) as v1",
        )
        self.assertFalse(v2_video.is_symlink(), "v2 video is a symlink")

        # Delete the v1 file; v2 must still be readable
        v1_video.unlink()
        v2_bytes_after = v2_video.read_bytes()
        self.assertEqual(v2_bytes, v2_bytes_after,
                         "v2 video unreadable after v1 deletion")

    def test_smoke_action_and_keypoints_bitwise(self):
        """action and keypoint_3d columns are bitwise identical after repack."""
        import pyarrow.parquet as pq

        v1_pf = sorted(self.src.glob("data/**/*.parquet"))[0]
        v1_table = pq.read_table(v1_pf)

        for col in ("action", "observation.keypoint_3d"):
            v1_arr = np.stack(v1_table[col].to_pylist()).astype(np.float32)
            v2_arr = np.stack(self.v2_table[col].to_pylist()).astype(np.float32)
            self.assertEqual(
                v1_arr.tobytes(), v2_arr.tobytes(),
                f"{col}: float32 bytes differ between v1 and v2",
            )

    def test_smoke_refuses_existing_dest(self):
        """Second run without --force returns exit code 2."""
        r = subprocess.run(
            [PY, REPACK_SCRIPT,
             "--src", str(self.src),
             "--dest", str(self.dest),
             "--confirm-full-run"],
            capture_output=True, text=True,
        )
        self.assertEqual(r.returncode, 2, f"Expected exit code 2, got {r.returncode}")

    def test_smoke_force_refuses_non_v2(self):
        """--force on a dest with robot_type='panda' is rejected; dir survives."""
        bad_dest = Path(self._tmpdir) / "non_v2_dest"
        bad_dest.mkdir()
        (bad_dest / "meta").mkdir()
        with open(bad_dest / "meta" / "info.json", "w") as f:
            json.dump({"robot_type": "panda"}, f)

        r = subprocess.run(
            [PY, REPACK_SCRIPT,
             "--src", str(self.src),
             "--dest", str(bad_dest),
             "--force", "--confirm-full-run"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertTrue(bad_dest.exists(), "Directory should still exist after refusal")

    def test_full_flag_rejects_max_episodes(self):
        """--confirm-full-run combined with --max-episodes is rejected."""
        r = subprocess.run(
            [PY, REPACK_SCRIPT,
             "--src", str(self.src),
             "--dest", "/tmp/never_created_smoke",
             "--confirm-full-run", "--max-episodes", "1"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(r.returncode, 0)

    # ---------------------------------------------------------------
    # 8.7  Stats recompute on the mini dataset
    # ---------------------------------------------------------------

    def test_smoke_stats_has_all_keys(self):
        self.assertEqual(len(self.v2_stats), 12, list(self.v2_stats.keys()))

    def test_smoke_stats_state_dim_14(self):
        self.assertEqual(len(self.v2_stats["observation.state"]["mean"]), 14)

    def test_smoke_stats_fingers_dim_2(self):
        self.assertEqual(len(self.v2_stats["observation.state.fingers"]["mean"]), 2)

    def test_smoke_stats_action_dim_7(self):
        self.assertEqual(len(self.v2_stats["action"]["mean"]), 7)

    def test_smoke_stats_keypoint_dim_56(self):
        self.assertEqual(len(self.v2_stats["observation.keypoint_3d"]["mean"]), 56)

    def test_smoke_stats_joint_dim_7(self):
        self.assertEqual(
            len(self.v2_stats["observation.state.joint_position"]["mean"]), 7,
        )

    def test_smoke_stats_each_has_10_metrics(self):
        for feat in VECTOR_FEATURES:
            got = set(self.v2_stats[feat].keys())
            self.assertEqual(
                got, EXPECTED_STAT_METRICS,
                f"{feat}: missing or extra metrics: {got ^ EXPECTED_STAT_METRICS}",
            )

    def test_smoke_stats_count_matches_frames(self):
        for feat in VECTOR_FEATURES:
            self.assertEqual(
                self.v2_stats[feat]["count"][0], 8,
                f"{feat}: count should be 8 (2 episodes x 4 frames)",
            )

    def test_smoke_stats_monotonicity(self):
        for feat in VECTOR_FEATURES:
            s = self.v2_stats[feat]
            mn = np.array(s["min"])
            q01 = np.array(s["q01"])
            q10 = np.array(s["q10"])
            q50 = np.array(s["q50"])
            q90 = np.array(s["q90"])
            q99 = np.array(s["q99"])
            mx = np.array(s["max"])
            mean = np.array(s["mean"])
            std = np.array(s["std"])

            for d in range(len(mn)):
                self.assertLessEqual(mn[d], q01[d], f"{feat} dim {d}: min > q01")
                self.assertLessEqual(q01[d], q10[d], f"{feat} dim {d}: q01 > q10")
                self.assertLessEqual(q10[d], q50[d], f"{feat} dim {d}: q10 > q50")
                self.assertLessEqual(q50[d], q90[d], f"{feat} dim {d}: q50 > q90")
                self.assertLessEqual(q90[d], q99[d], f"{feat} dim {d}: q90 > q99")
                self.assertLessEqual(q99[d], mx[d], f"{feat} dim {d}: q99 > max")
                self.assertLessEqual(mn[d], mean[d], f"{feat} dim {d}: min > mean")
                self.assertLessEqual(mean[d], mx[d], f"{feat} dim {d}: mean > max")
                self.assertGreaterEqual(std[d], 0.0, f"{feat} dim {d}: std < 0")

    # ---------------------------------------------------------------
    # 8.8  Meta file integrity
    # ---------------------------------------------------------------

    def test_smoke_meta_all_files_present(self):
        meta = self.dest / "meta"
        for name in [
            "info.json",
            "stats.json",
            "goal_train_eval_contract.json",
            "keypoints_meta.json",
            "goal_episodes.jsonl",
            "tasks.parquet",
        ]:
            self.assertTrue(
                (meta / name).exists(), f"Missing meta file: {name}",
            )
        self.assertTrue(
            (meta / "episodes").is_dir(), "Missing meta/episodes directory",
        )

    def test_smoke_info_robot_type(self):
        self.assertEqual(self.v2_info["robot_type"], ROBOT_TYPE)

    def test_smoke_info_state_shape(self):
        self.assertEqual(
            self.v2_info["features"]["observation.state"]["shape"], [STATE_DIM],
        )

    def test_smoke_info_state_names(self):
        self.assertEqual(
            self.v2_info["features"]["observation.state"]["names"],
            list(STATE_NAMES),
        )

    def test_smoke_info_fingers_feature_exists(self):
        feat = self.v2_info["features"].get("observation.state.fingers")
        self.assertIsNotNone(feat, "Missing observation.state.fingers feature")
        self.assertEqual(feat["shape"], [2])

    def test_smoke_info_action_unchanged(self):
        self.assertEqual(
            self.v2_info["features"]["action"],
            self.data_info["v1_info"]["features"]["action"],
        )

    def test_smoke_contract_schema_2(self):
        self.assertEqual(self.v2_contract["schema"], CONTRACT_SCHEMA)

    def test_smoke_contract_state_dim_14(self):
        self.assertEqual(self.v2_contract["state_dim"], STATE_DIM)


# ===========================================================================
# 8.3  Schema and delta-transform failure modes
# ===========================================================================


class S83_SchemaAndDelta(unittest.TestCase):
    """Schema lookup, delta transform shape mismatch, and abs-mode transform removal."""

    def test_schema_two_cameras_and_prompt_mode(self):
        try:
            from lerobot.dataset_schemas.registry import get_schema
        except ImportError:
            self.skipTest("cannot import dataset schema registry")

        schema = get_schema("libero_goal_4dv2")
        self.assertIn("observation.images.image", schema.image_mapping)
        self.assertIn("observation.images.image2", schema.image_mapping)
        self.assertEqual(schema.action_mode, "end_effector")
        self.assertEqual(
            schema.feature_mapping.get("observation.state"),
            ["observation.state"],
        )
        self.assertIsNone(schema.action_mask_spec)

    def test_delta_transform_rejects_14_vs_7(self):
        try:
            import torch
            from lerobot.transforms.core import DeltaActionTransformFn
        except ImportError:
            self.skipTest("cannot import DeltaActionTransformFn or torch")

        # Simulate a hydrated transform with 14D state and a 7-element mask
        tfn = DeltaActionTransformFn()
        tfn.mapping = {
            "observation.state": ["observation.state"],
            "action": ["action"],
        }
        tfn.mask = torch.tensor([True] * 6 + [False])  # 7 elements vs 14D state

        data = {
            "observation.state": torch.randn(14),
            "action": torch.randn(7),
        }
        with self.assertRaises(RuntimeError):
            tfn(data)

    def test_abs_path_does_not_subtract(self):
        try:
            from lerobot.policies.internvla_a1_5 import InternVLAA15DatasetConfig
            from lerobot.transforms.core import DeltaActionTransformFn
        except ImportError:
            self.skipTest("cannot import InternVLAA15DatasetConfig")

        cfg = InternVLAA15DatasetConfig(repo_id="dummy", action_mode="abs")
        transforms = cfg.data_transforms.inputs
        has_delta = any(isinstance(t, DeltaActionTransformFn) for t in transforms)
        self.assertFalse(
            has_delta,
            "DeltaActionTransformFn should be removed when action_mode='abs'",
        )


# ===========================================================================
# 8.6  Stats computation pure functions
# ===========================================================================


class S86_StatsPureFunctions(unittest.TestCase):
    """compute_vector_stats correctness, edge cases, and rejection guards."""

    def test_compute_vector_stats_basic(self):
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        s = compute_vector_stats(data)
        np.testing.assert_array_equal(s["mean"], [3, 4])
        np.testing.assert_array_equal(s["min"], [1, 2])
        np.testing.assert_array_equal(s["max"], [5, 6])
        np.testing.assert_array_equal(s["count"], [3])
        np.testing.assert_array_equal(s["q50"], [3, 4])

    def test_compute_vector_stats_single_row(self):
        data = np.array([[7, 8]], dtype=np.float64)
        s = compute_vector_stats(data)
        for metric in ("mean", "min", "max", "q50"):
            np.testing.assert_array_equal(
                s[metric], [7, 8], err_msg=f"{metric} mismatch for single row",
            )
        np.testing.assert_array_equal(s["std"], [0, 0])
        np.testing.assert_array_equal(s["count"], [1])

    def test_compute_vector_stats_constant_column(self):
        # Use exactly-representable binary values (3.14 causes np.mean
        # accumulation error that triggers compute_vector_stats internal assert)
        row = np.array([3.125, 2.75], dtype=np.float64)
        data = np.tile(row, (100, 1))
        s = compute_vector_stats(data)
        for metric in ("mean", "min", "max", "q01", "q99"):
            np.testing.assert_allclose(
                s[metric], [3.125, 2.75], atol=1e-10,
                err_msg=f"{metric} differs from constant value",
            )
        np.testing.assert_array_equal(s["std"], [0, 0])

    def test_stats_monotonicity_invariant(self):
        rng = np.random.default_rng(99)
        data = rng.standard_normal((1000, 14))
        s = compute_vector_stats(data)
        mn = np.array(s["min"])
        q01 = np.array(s["q01"])
        q10 = np.array(s["q10"])
        q50 = np.array(s["q50"])
        q90 = np.array(s["q90"])
        q99 = np.array(s["q99"])
        mx = np.array(s["max"])
        mean = np.array(s["mean"])
        std = np.array(s["std"])

        self.assertTrue(np.all(mn <= q01))
        self.assertTrue(np.all(q01 <= q10))
        self.assertTrue(np.all(q10 <= q50))
        self.assertTrue(np.all(q50 <= q90))
        self.assertTrue(np.all(q90 <= q99))
        self.assertTrue(np.all(q99 <= mx))
        self.assertTrue(np.all(mn <= mean))
        self.assertTrue(np.all(mean <= mx))
        self.assertTrue(np.all(std >= 0))

    def test_stats_nan_rejection(self):
        data = np.ones((10, 14), dtype=np.float64)
        data[5, 7] = np.nan
        with self.assertRaises(ValueError):
            compute_vector_stats(data)

    def test_stats_inf_rejection(self):
        data = np.ones((10, 14), dtype=np.float64)
        data[3, 2] = np.inf
        with self.assertRaises(ValueError):
            compute_vector_stats(data)


# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
