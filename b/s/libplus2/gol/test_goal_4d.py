#!/usr/bin/env python3
"""Acceptance tests for the Goal 3D/4D contract.

Run:
    /B/VENV/itnvla15rbt20/bin/python b/s/libplus2/gol/test_goal_4d.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from contract import GOAL_BASE_XPOS, HISTORY_MAX_LEN  # noqa: E402
from fk import (  # noqa: E402
    GoalTableFK,
    axisangle_to_quat_xyzw,
    compute_r_pad,
    qpos9_from_state,
    rotation_error_deg,
    wxyz_to_xyzw_hemisphere,
)
from history import GoalKeypointRuntime, pack_like_training  # noqa: E402
from load_contract import load_r_pad  # noqa: E402
from rlds_goal import DEFAULT_ROOT, iter_goal_episodes  # noqa: E402

LIFT_BASE = np.array([-0.56, 0.0, 0.912])


class FkAgainstGoalRlds(unittest.TestCase):
    """Personal measurement: table base matches state, Lift base does not."""

    @classmethod
    def setUpClass(cls):
        cls.fk = GoalTableFK()
        cls.episodes = []
        for ep in iter_goal_episodes(DEFAULT_ROOT, max_episodes=4):
            cls.episodes.append(ep)
        if len(cls.episodes) < 4:
            raise RuntimeError("expected at least 4 RLDS episodes")

    def test_table_base_matches_eef_position_and_hand_orientation(self):
        worst_pos = 0.0
        worst_rot = 0.0
        for ep in self.episodes:
            qpos = ep.qpos9()
            step = max(1, ep.num_steps // 10)
            for t in range(0, ep.num_steps, step):
                pose = self.fk.world_poses(qpos[t])
                worst_pos = max(worst_pos, float(np.linalg.norm(pose[7, :3] - ep.state[t, :3])))
                q_hand = wxyz_to_xyzw_hemisphere(self.fk.hand_quat_wxyz())
                q_state = axisangle_to_quat_xyzw(ep.state[t, 3:6])
                worst_rot = max(worst_rot, rotation_error_deg(q_hand, q_state))
        self.assertLess(worst_pos, 2e-3, f"eef position error {worst_pos} m")
        self.assertLess(worst_rot, 0.5, f"hand rotation error {worst_rot} deg")

    def test_lift_base_is_ten_centimetres_off(self):
        ep = self.episodes[0]
        pose = self.fk.world_poses(ep.qpos9()[0], base_xpos=LIFT_BASE)
        err = float(np.linalg.norm(pose[7, :3] - ep.state[0, :3]))
        self.assertGreater(err, 0.09)
        self.assertLess(err, 0.11)

    def test_eef_quaternion_is_ninety_degrees_from_state(self):
        ep = self.episodes[0]
        pose = self.fk.world_poses(ep.qpos9()[0])
        q_eef = wxyz_to_xyzw_hemisphere(pose[7, 3:])
        q_state = axisangle_to_quat_xyzw(ep.state[0, 3:6])
        err = rotation_error_deg(q_eef, q_state)
        self.assertAlmostEqual(err, 90.0, delta=0.05)

    def test_fingers_do_not_move_keypoints(self):
        ep = self.episodes[0]
        q = ep.qpos9()[0].copy()
        a = self.fk.world_poses(q)
        q[7:9] = [0.0, 0.0]
        b = self.fk.world_poses(q)
        self.assertLess(float(np.max(np.abs(a - b))), 1e-8)

    def test_default_base_is_the_table(self):
        np.testing.assert_allclose(self.fk.base_xpos, GOAL_BASE_XPOS)

    def test_qpos_builder_rejects_short_vectors(self):
        with self.assertRaises(ValueError):
            qpos9_from_state(np.zeros(6), np.zeros(8))


class HistoryMatchesTraining(unittest.TestCase):
    def test_pack_front_and_zero_tail(self):
        kpts = np.arange(6 * 8 * 7, dtype=np.float32).reshape(6, 8, 7)
        packed = pack_like_training(kpts, t=2, history_len=4, chunk_size=3)
        self.assertEqual(packed["his_len"], 2)
        np.testing.assert_allclose(packed["his_kpts"][0], kpts[0])
        np.testing.assert_allclose(packed["his_kpts"][1], kpts[1])
        self.assertTrue(np.all(packed["his_kpts"][2:] == 0))
        np.testing.assert_allclose(packed["kpt_t"], kpts[2])
        np.testing.assert_allclose(packed["kpt_future"][0], kpts[3])

    def test_future_clamps_at_episode_end(self):
        kpts = np.ones((3, 8, 7), dtype=np.float32)
        kpts[-1] *= 2
        packed = pack_like_training(kpts, t=2, history_len=2, chunk_size=4)
        np.testing.assert_allclose(packed["kpt_future"], np.broadcast_to(kpts[-1], packed["kpt_future"].shape))

    def test_runtime_does_not_leak_current_frame(self):
        fk = GoalTableFK()
        ep = next(iter_goal_episodes(DEFAULT_ROOT, max_episodes=1))
        runtime = GoalKeypointRuntime(fk, r_pad=1.5, history_len=8)
        qpos = ep.qpos9()
        first = runtime.begin_step(qpos[0])
        self.assertEqual(first["his_len"], 0)
        runtime.commit_step()
        second = runtime.begin_step(qpos[1])
        self.assertEqual(second["his_len"], 1)
        np.testing.assert_allclose(second["kpt_history"][0], fk.keypoints(qpos[0], 1.5))
        # current frame is not yet in the history buffer
        self.assertFalse(np.allclose(second["kpt_history"][0], second["kpt_t"]))
        runtime.commit_step()

    def test_wait_steps_must_not_be_committed(self):
        """A 10-step settle prefix is not in the RLDS demonstrations."""
        fk = GoalTableFK()
        ep = next(iter_goal_episodes(DEFAULT_ROOT, max_episodes=1))
        runtime = GoalKeypointRuntime(fk, r_pad=1.5, history_len=HISTORY_MAX_LEN)
        # Correct: first policy request sees an empty history.
        fields = runtime.begin_step(ep.qpos9()[0])
        self.assertEqual(fields["his_len"], 0)
        runtime.commit_step()

    def test_chunk_reuse_still_commits_every_step(self):
        fk = GoalTableFK()
        ep = next(iter_goal_episodes(DEFAULT_ROOT, max_episodes=1))
        runtime = GoalKeypointRuntime(fk, r_pad=1.5, history_len=16)
        qpos = ep.qpos9()
        for t in range(5):
            payload = runtime.begin_step(qpos[t])
            self.assertEqual(payload["his_len"], t)
            # even if the policy does not replan, the step is committed
            runtime.commit_step()
        packed = pack_like_training(fk.batch_keypoints(qpos[:5], 1.5), t=4, history_len=16, chunk_size=1)
        his, length = runtime.history.get_history()
        self.assertEqual(length, 5)
        np.testing.assert_allclose(his[:4], packed["his_kpts"][:4])

    def test_double_begin_is_rejected(self):
        fk = GoalTableFK()
        ep = next(iter_goal_episodes(DEFAULT_ROOT, max_episodes=1))
        runtime = GoalKeypointRuntime(fk, r_pad=1.0, history_len=4)
        runtime.begin_step(ep.qpos9()[0])
        with self.assertRaises(RuntimeError):
            runtime.begin_step(ep.qpos9()[1])

    def test_transform_agrees_on_a_short_window(self):
        import torch
        from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import Extract3DKeypointTransformFn

        torch.manual_seed(0)
        length, h, c, j, d = 6, 4, 2, 8, 7
        episode = torch.randn(length, j * d)
        t = 3
        # Build the stacked window the dataset would hand the transform.
        offsets = list(range(-h, c + 1))
        rows = []
        is_pad = []
        for off in offsets:
            idx = t + off
            if idx < 0 or idx >= length:
                rows.append(episode[max(0, min(length - 1, idx))])
                is_pad.append(True if idx < 0 else idx >= length)
            else:
                rows.append(episode[idx])
                is_pad.append(False)
        # Future clamps are still real samples of the last frame; the transform
        # only treats history is_pad as invalid. Mirror LeRobot: history pads
        # are the negative overflows. Future overflow is a clamped value with
        # is_pad True, but Extract3DKeypointTransformFn does not drop future pads.
        data = {
            "observation.keypoint_3d": torch.stack(rows),
            "observation.keypoint_3d_is_pad": torch.tensor(is_pad),
        }
        out = Extract3DKeypointTransformFn(num_joints=j, history_max_len=h, chunk_size=c, keypoint_dim=d)(data)
        packed = pack_like_training(episode.numpy().reshape(length, j, d), t=t, history_len=h, chunk_size=c)
        self.assertEqual(int(out["observation.his_len"]), packed["his_len"])
        np.testing.assert_allclose(out["observation.his_kpts"].numpy(), packed["his_kpts"], atol=1e-6)
        np.testing.assert_allclose(out["observation.kpt_t"].numpy(), packed["kpt_t"], atol=1e-6)


class ContractFile(unittest.TestCase):
    def test_missing_file_has_no_fallback(self):
        with self.assertRaises(FileNotFoundError):
            load_r_pad("/tmp/does_not_exist_goal_keypoints_meta.json")

    def test_rejects_lift_coordinate_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "keypoints_meta.json"
            path.write_text('{"coordinate_system": "lift", "bbox_radius": 1.8}')
            with self.assertRaises(ValueError):
                load_r_pad(path)

    def test_r_pad_formula_includes_margin_once(self):
        gmin = np.array([-1.0, -0.2, 0.0])
        gmax = np.array([0.5, 0.2, 1.5])
        self.assertAlmostEqual(compute_r_pad(gmin, gmax, 0.15), 1.5 * 1.15)


class GenerateSmoke(unittest.TestCase):
    def test_one_episode_roundtrip(self):
        from generate_goal_4d import main as generate_main

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "ds"
            argv = sys.argv
            sys.argv = [
                "generate_goal_4d.py",
                "--dest",
                str(dest),
                "--max-episodes",
                "1",
                "--no-video",
                "--repo-id",
                "goal4d_smoke",
            ]
            try:
                generate_main()
            finally:
                sys.argv = argv
            meta = json_load(dest / "meta" / "keypoints_meta.json")
            self.assertEqual(meta["coordinate_system"], "libero_goal_table_world")
            self.assertEqual(meta["total_episodes"], 1)
            self.assertGreater(meta["bbox_radius"], 0.5)
            rows = (dest / "meta" / "goal_episodes.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(rows), 1)
            info = json_load(dest / "meta" / "info.json")
            self.assertEqual(info["codebase_version"], "v3.0")
            self.assertEqual(info["fps"], 20)
            self.assertIn("observation.keypoint_3d", info["features"])


def json_load(path: Path):
    import json
    return json.loads(path.read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
