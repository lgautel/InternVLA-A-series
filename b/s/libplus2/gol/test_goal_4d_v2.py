#!/usr/bin/env python3
"""gen2 acceptance tests: the incidents that gen1 did not cover.

Run:
    /B/VENV/itnvla15rbt20/bin/python b/s/libplus2/gol/test_goal_4d_v2.py            # all
    SKIP_SLOW=1 /B/VENV/itnvla15rbt20/bin/python b/s/libplus2/gol/test_goal_4d_v2.py # no video conversion

Groups (see 3d4d_gen2.markdown section 7 for the coverage table):
  A  FK model equivalence           (eval FK == train FK == robosuite export)
  B  Constants agree with the rest of the repo (H, robot_type, schema, orientation contract)
  C  Image orientation offline probes on the RLDS release
  D  Generator guards               (validate_episode, jpeg orientation)
  E  Eval client clock and guards   (fake websocket + fake env, no simulator)
  F  Import hook does not import mujoco in the parent
  G  End-to-end small conversion    (slow; video path, orientation, training window, negative control)
"""

from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[3]
for p in (str(_HERE), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from contract import (  # noqa: E402
    CHUNK_SIZE,
    EPISODE_EEF_TOL_M,
    GOAL_BASE_XPOS,
    GOAL_MJCF,
    HISTORY_MAX_LEN,
    IMAGE_ORIENTATION,
    LIFT_MJCF_SOURCE,
    ROBOT_TYPE,
)
from fk import GoalTableFK, qpos9_from_state  # noqa: E402
from history import pack_like_training  # noqa: E402
from rlds_goal import DEFAULT_ROOT, iter_goal_episodes  # noqa: E402

PY = sys.executable


def _quat_err(a, b):
    return np.minimum(np.abs(a - b).max(-1), np.abs(a + b).max(-1))


# ---------------------------------------------------------------------------
class A_FkEquivalence(unittest.TestCase):
    """The eval FK (StandaloneFK on panda_goal_table.xml) is the generator FK."""

    def test_goal_xml_is_up_to_date_and_self_contained(self):
        out = subprocess.run([PY, str(_HERE / "make_goal_mjcf.py"), "--check"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        # loads from an unrelated cwd, no mesh/texture files needed
        code = (
            "import mujoco,sys;m=mujoco.MjModel.from_xml_path(sys.argv[1]);"
            "assert m.nq>=9;print(m.body('robot0_base').id)"
        )
        r = subprocess.run([PY, "-c", code, str(GOAL_MJCF)], cwd="/tmp", capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_generator_fk_equals_standalone_fk_on_goal_xml(self):
        from evaluation.LIBERO2.keypoint_utils import StandaloneFK

        r_pad = 1.7
        sfk = StandaloneFK(mjcf_path=GOAL_MJCF, r_pad=r_pad)
        gfk = GoalTableFK()
        rng = np.random.default_rng(1)
        lo = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.017, -2.89])
        hi = np.array([2.89, 1.76, 2.89, -0.07, 2.89, 3.75, 2.89])
        worst = 0.0
        for _ in range(400):
            q = np.concatenate([rng.uniform(lo, hi), rng.uniform([0, -0.04], [0.04, 0])])
            worst = max(worst, float(np.abs(sfk.extract(q) - gfk.keypoints(q, r_pad)).max()))
        self.assertLess(worst, 1e-5)

    def test_goal_xml_differs_from_lift_export_by_exactly_the_base_shift(self):
        import mujoco

        text = LIFT_MJCF_SOURCE.read_text()
        text = re.sub(r"<asset>.*?</asset>", "", text, flags=re.S)
        text = re.sub(r'\s(?:mesh|material|texture)="[^"]*"', "", text)
        text = re.sub(r'<geom [^>]*type="mesh"[^>]*/>', "", text)
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as fh:
            fh.write(text)
            lift_path = fh.name
        lift = mujoco.MjModel.from_xml_path(lift_path)
        goal = mujoco.MjModel.from_xml_path(str(GOAL_MJCF))
        dl, dg = mujoco.MjData(lift), mujoco.MjData(goal)
        names = [f"robot0_link{i}" for i in range(1, 8)] + ["gripper0_eef"]
        rng = np.random.default_rng(2)
        for _ in range(50):
            q = np.concatenate([rng.uniform(lift.jnt_range[:7, 0], lift.jnt_range[:7, 1]), rng.uniform([0, -0.04], [0.04, 0])])
            for m, d in ((lift, dl), (goal, dg)):
                d.qpos[:9] = q
                mujoco.mj_forward(m, d)
            for n in names:
                a, b = dl.xpos[lift.body(n).id], dg.xpos[goal.body(n).id]
                np.testing.assert_allclose(b - a, [-0.10, 0, 0], atol=1e-9)
                self.assertLess(float(_quat_err(dl.xquat[lift.body(n).id], dg.xquat[goal.body(n).id])), 1e-9)

    def test_generator_fk_equals_robosuite_export_with_lift_base(self):
        """Independent check of the eef composition: robosuite's own export vs our composed chain."""
        import mujoco

        text = LIFT_MJCF_SOURCE.read_text()
        text = re.sub(r"<asset>.*?</asset>", "", text, flags=re.S)
        text = re.sub(r'\s(?:mesh|material|texture)="[^"]*"', "", text)
        text = re.sub(r'<geom [^>]*type="mesh"[^>]*/>', "", text)
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as fh:
            fh.write(text)
        m = mujoco.MjModel.from_xml_path(fh.name)
        d = mujoco.MjData(m)
        fk = GoalTableFK(base_xpos=np.array([-0.56, 0, 0.912]))
        rng = np.random.default_rng(3)
        ids = [m.body(f"robot0_link{i}").id for i in range(1, 8)] + [m.body("gripper0_eef").id]
        worst_p = worst_q = 0.0
        for _ in range(300):
            q = np.concatenate([rng.uniform(m.jnt_range[:7, 0], m.jnt_range[:7, 1]), rng.uniform([0, -0.04], [0.04, 0])])
            d.qpos[:9] = q
            mujoco.mj_forward(m, d)
            P = fk.world_poses(q)
            for k, i in enumerate(ids):
                worst_p = max(worst_p, float(np.abs(P[k, :3] - d.xpos[i]).max()))
                worst_q = max(worst_q, float(_quat_err(P[k, 3:], d.xquat[i])))
        self.assertLess(worst_p, 1e-6)
        self.assertLess(worst_q, 1e-6)


# ---------------------------------------------------------------------------
class B_ConstantsAgreeWithRepo(unittest.TestCase):
    def test_history_length_matches_client_keypoint_history_and_launch(self):
        from evaluation.LIBERO2.keypoint_utils import KeypointHistory
        from evaluation.LIBERO2.model2libero_interface import LiberoModelClient

        sig = inspect.signature(LiberoModelClient.__init__)
        self.assertEqual(sig.parameters["kpt_history_max_len"].default, HISTORY_MAX_LEN)
        self.assertEqual(inspect.signature(KeypointHistory.__init__).parameters["max_len"].default, HISTORY_MAX_LEN)
        launch = (ROOT / "launch" / "libplus_sft_launch.sh").read_text()
        values = set(re.findall(r"keypoint_history_max_len=(\d+)", launch))
        self.assertEqual(values, {str(HISTORY_MAX_LEN)}, "training launch and goal contract disagree on H")

    def test_chunk_and_replan_fit(self):
        from evaluation.LIBERO2.model2libero_interface import LiberoModelClient

        replan = inspect.signature(LiberoModelClient.__init__).parameters["replan_steps"].default
        self.assertLessEqual(replan, CHUNK_SIZE)

    def test_robot_type_has_a_two_camera_schema(self):
        text = (ROOT / "src/lerobot/dataset_schemas/configs" / f"{ROBOT_TYPE}.yaml").read_text()
        self.assertIn("observation.images.image: observation.images.image0", text)
        self.assertIn("observation.images.image2: observation.images.image1", text)

    def test_image_orientation_matches_global_eval_contract(self):
        from evaluation.LIBERO2.orientation_contract import load_train_eval_contract

        self.assertEqual(load_train_eval_contract()["image_orientation"], IMAGE_ORIENTATION)

    def test_fps_is_a_multiple_the_video_stack_accepts(self):
        from contract import FPS

        self.assertEqual(FPS, 20)


# ---------------------------------------------------------------------------
class C_OrientationOfTheRldsRelease(unittest.TestCase):
    """RLDS frames are rot180 of robosuite raw. Measured, not assumed."""

    @classmethod
    def setUpClass(cls):
        import io as _io

        from PIL import Image

        from orientation_probe import to_gray

        cls.eef, cls.gray_a, cls.gray_w, cls.camp, cls.camr = [], [], [], [], []
        fk = GoalTableFK()
        from orientation_probe import WRIST_POS_IN_HAND, WRIST_QUAT_WXYZ, quat_to_mat

        n = 0
        for ep in iter_goal_episodes(DEFAULT_ROOT, max_episodes=60):
            if ep.perturbation != "language":
                continue
            dec = lambda blobs: np.stack([np.asarray(Image.open(_io.BytesIO(b)).convert("RGB")) for b in blobs])
            cls.eef.append(ep.state[:, :3].astype(np.float64))
            cls.gray_a.append(to_gray(dec(ep.image_jpeg)))
            cls.gray_w.append(to_gray(dec(ep.wrist_jpeg)))
            pos, rot = [], []
            for t in range(ep.num_steps):
                hp, hq = fk.hand_pose(qpos9_from_state(ep.joint_state[t], ep.state[t]))
                rh = quat_to_mat(hq)
                pos.append(hp + rh @ WRIST_POS_IN_HAND)
                rot.append(rh @ quat_to_mat(WRIST_QUAT_WXYZ))
            cls.camp.append(np.asarray(pos))
            cls.camr.append(rot)
            n += 1
            if n >= 6:
                break
        assert n >= 4

    def test_agentview_rlds_is_flipH_of_upright_i_e_rot180_of_raw(self):
        from orientation_probe import agentview_orientation

        r = agentview_orientation(self.eef, self.gray_a)
        self.assertEqual(r["best"], "flipH_RLDS", r)
        self.assertGreater(r["scores"]["flipH_RLDS"]["min_corr"], 0.5)
        # the raw hypothesis must clearly lose: one axis has the wrong sign
        self.assertLess(r["scores"]["flipV_RAW"]["min_corr"], -0.5)

    def test_wrist_rlds_is_flipH_of_upright(self):
        from orientation_probe import wrist_orientation

        r = wrist_orientation(self.camp, self.camr, self.gray_w)
        self.assertEqual(r["best"], "flipH_RLDS", r)
        self.assertGreater(r["scores"]["flipH_RLDS"]["min_corr"], 0.3)

    def test_probe_flips_when_the_frames_are_rotated(self):
        """Negative control: rotating the frames by 180 degrees moves the verdict to the RAW hypothesis."""
        from orientation_probe import agentview_orientation

        rot = [g[:, ::-1, ::-1] for g in self.gray_a]
        r = agentview_orientation(self.eef, rot)
        self.assertEqual(r["best"], "flipV_RAW", r)


# ---------------------------------------------------------------------------
class D_GeneratorGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ep = next(iter_goal_episodes(DEFAULT_ROOT, max_episodes=1))
        cls.fk = GoalTableFK()
        cls.poses = cls.fk.batch_world(cls.ep.qpos9())

    def test_jpeg_orientation_raw_is_rot180_of_rlds(self):
        from PIL import Image

        from generate_goal_4d import _jpeg_to_rgb

        arr = np.zeros((16, 16, 3), dtype=np.uint8)
        arr[0, 0] = 255  # top-left marker
        buf = BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        blob = buf.getvalue()
        rlds = _jpeg_to_rgb(blob, "rlds")
        raw = _jpeg_to_rgb(blob, "raw")
        self.assertTrue(rlds[0, 0].all() and not rlds[-1, -1].any())
        self.assertTrue(raw[-1, -1].all() and not raw[0, 0].any())
        self.assertTrue(raw.flags["C_CONTIGUOUS"])
        with self.assertRaises(ValueError):
            _jpeg_to_rgb(blob, "upright")

    def test_default_orientation_is_the_contract_orientation(self):
        from generate_goal_4d import _jpeg_to_rgb

        self.assertEqual(inspect.signature(_jpeg_to_rgb).parameters["orientation"].default, IMAGE_ORIENTATION)

    def test_validate_accepts_a_real_episode(self):
        from generate_goal_4d import validate_episode

        self.assertLess(validate_episode(self.ep, self.poses), EPISODE_EEF_TOL_M)

    def test_validate_rejects_lift_base(self):
        from generate_goal_4d import EpisodeValidationError, validate_episode

        lift = GoalTableFK(base_xpos=np.array([-0.56, 0, 0.912])).batch_world(self.ep.qpos9())
        with self.assertRaises(EpisodeValidationError) as cm:
            validate_episode(self.ep, lift)
        self.assertIn("base frame", str(cm.exception))

    def test_validate_rejects_nan_and_short_and_joint_limit(self):
        import copy

        from generate_goal_4d import EpisodeValidationError, validate_episode

        bad = copy.copy(self.ep)
        bad.state = self.ep.state.copy()
        bad.state[3, 0] = np.nan
        with self.assertRaises(EpisodeValidationError):
            validate_episode(bad, self.poses)
        bad = copy.copy(self.ep)
        bad.joint_state = self.ep.joint_state.copy()
        bad.joint_state[2, 3] = 0.5  # joint4 must be < -0.07
        with self.assertRaises(EpisodeValidationError):
            validate_episode(bad, self.poses)

    def test_full_run_flag_refuses_test_switches(self):
        for extra in (["--no-video"], ["--max-episodes", "3"], ["--image-orientation", "rlds"]):
            r = subprocess.run(
                [PY, str(_HERE / "generate_goal_4d.py"), "--dest", "/tmp/never_written", "--confirm-full-run", *extra],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(r.returncode, 0, extra)
            self.assertFalse(Path("/tmp/never_written").exists())


# ---------------------------------------------------------------------------
class _FakeWS:
    """Stands in for WebsocketClientPolicy. Records every infer payload."""

    log: list = []

    def __init__(self, host, port):
        pass

    def get_server_metadata(self):
        return {
            "protocol_version": "2.1",
            "preprocessing_owner": "server_canonical",
            "deterministic_inference_preprocess": True,
            "chunk_size": CHUNK_SIZE,
            "action_dim": 7,
        }

    def predict_action(self, request):
        _FakeWS.log.append(request["payload"]["examples"][0])
        return {"ok": True, "data": {"actions": np.zeros((1, CHUNK_SIZE, 7), dtype=np.float32)}}


def _fake_env(qpos_now: np.ndarray):
    data = SimpleNamespace(qpos=np.zeros(16))
    robot = SimpleNamespace(_ref_joint_pos_indexes=np.arange(7), _ref_gripper_joint_pos_indexes=np.array([7, 8]))
    env = SimpleNamespace(robots=[robot], sim=SimpleNamespace(data=data))
    data.qpos[:9] = qpos_now
    return env


def _obs(state_row, eef_shift=0.0):
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    return {
        "agentview_image": img,
        "robot0_eye_in_hand_image": img,
        "robot0_eef_pos": state_row[:3] + np.array([eef_shift, 0, 0]),
        "robot0_eef_quat": np.array([0, 0, 0, 1.0]),
        "robot0_gripper_qpos": state_row[6:8],
    }


class E_EvalClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import evaluation.LIBERO2.model2libero_interface as base

        cls.base = base
        cls._orig_ws = base.WebsocketClientPolicy
        base.WebsocketClientPolicy = _FakeWS
        cls.ep = next(iter_goal_episodes(DEFAULT_ROOT, max_episodes=1))
        cls.tmp = tempfile.TemporaryDirectory()
        import hashlib

        cls.contract = {
            "schema": "goal_train_eval_contract/1",
            "coordinate_system": "libero_goal_table_world",
            "r_pad": 1.75,
            "keypoint_history_max_len": HISTORY_MAX_LEN,
            "image_orientation": "raw",
            "robot_type": "panda",
            "eval_mjcf_md5": hashlib.md5(GOAL_MJCF.read_bytes()).hexdigest(),
            "live_eef_tol_m": 0.005,
        }
        cls.cpath = Path(cls.tmp.name) / "c.json"
        cls.cpath.write_text(json.dumps(cls.contract))

    @classmethod
    def tearDownClass(cls):
        cls.base.WebsocketClientPolicy = cls._orig_ws
        cls.tmp.cleanup()

    def _client(self, **kw):
        from goal_client import GoalLiberoModelClient

        return GoalLiberoModelClient("h", 1, contract_path=self.cpath, replan_steps=1, **kw)

    def _run(self, client, n_wait=10, n_steps=6, shift=0.0):
        """The exact call order of evaluation/LIBERO-plus2/eval_libero_plus.py::evaluate_task."""
        _FakeWS.log = []
        qpos = self.ep.qpos9()
        env = _fake_env(qpos[0])
        client.bind_env(env)
        client.reset("put the bowl on the plate")
        for t in range(n_wait + n_steps):
            k = max(0, t - n_wait)
            # during wait the arm drifts slightly, so wait frames are distinguishable from frame 0
            q = qpos[0] + (0.02 * (n_wait - t) if t < n_wait else 0.0)
            env.sim.data.qpos[:9] = q if t < n_wait else qpos[k]
            obs = _obs(self.ep.state[k], eef_shift=shift)
            if t < n_wait:
                client.push_keypoint()
                continue
            client.step(obs, "put the bowl on the plate")
            client.push_keypoint()

    def test_defaults_come_from_the_contract(self):
        c = self._client()
        self.assertEqual(c._kpt_history.max_len, HISTORY_MAX_LEN)
        self.assertAlmostEqual(c._fk.r_pad, 1.75)
        self.assertIn("panda_goal_table.xml", str(GOAL_MJCF))

    def test_first_request_has_no_history_and_wait_frames_are_not_history(self):
        c = self._client()
        self._run(c, n_wait=10, n_steps=6)
        payloads = _FakeWS.log
        self.assertEqual(len(payloads), 6)
        self.assertNotIn("kpt_history", payloads[0])  # his_len == 0 is not sent
        self.assertEqual([p.get("his_len", 0) for p in payloads], [0, 1, 2, 3, 4, 5])

    def test_history_equals_training_packing(self):
        c = self._client()
        self._run(c, n_wait=10, n_steps=6)
        stored = GoalTableFK().batch_keypoints(self.ep.qpos9(), 1.75)
        for t in (1, 3, 5):
            packed = pack_like_training(stored, t=t, history_len=HISTORY_MAX_LEN, chunk_size=CHUNK_SIZE)
            got = _FakeWS.log[t]
            self.assertEqual(got["his_len"], packed["his_len"])
            np.testing.assert_allclose(got["kpt_history"], packed["his_kpts"], atol=2e-5)

    def test_current_frame_is_never_in_history(self):
        c = self._client()
        self._run(c, n_wait=10, n_steps=4)
        stored = GoalTableFK().batch_keypoints(self.ep.qpos9(), 1.75)
        got = _FakeWS.log[3]
        self.assertFalse(np.allclose(got["kpt_history"][2], stored[3], atol=1e-6))
        np.testing.assert_allclose(got["kpt_history"][2], stored[2], atol=2e-5)

    def test_reset_disarms_and_clears(self):
        c = self._client()
        self._run(c, n_wait=3, n_steps=3)
        self.assertEqual(c.his_len, 3)
        c.reset("put the bowl on the plate")
        self.assertEqual(c.his_len, 0)
        c.push_keypoint()  # wait-step call of the next episode
        self.assertEqual(c.his_len, 0)

    def test_live_fk_guard_aborts_on_a_wrong_base(self):
        from goal_client import GoalFKMismatch

        c = self._client()
        with self.assertRaises(GoalFKMismatch):
            self._run(c, n_wait=2, n_steps=2, shift=0.10)

    def test_live_fk_guard_tracks_error_on_the_right_base(self):
        c = self._client()
        self._run(c, n_wait=2, n_steps=4)
        self.assertLess(c.live_fk_max_err_m, 0.002)
        self.assertEqual(c.live_fk_violations, 0)

    def test_conflicting_overrides_are_rejected(self):
        with self.assertRaises(ValueError):
            self._client(kpt_r_pad=1.8212722539901733)
        with self.assertRaises(ValueError):
            self._client(kpt_history_max_len=128)

    def test_orientation_mismatch_with_global_contract_is_rejected(self):
        bad = dict(self.contract, image_orientation="rlds")
        path = Path(self.tmp.name) / "bad.json"
        path.write_text(json.dumps(bad))
        from goal_client import GoalLiberoModelClient

        with self.assertRaises(RuntimeError) as cm:
            GoalLiberoModelClient("h", 1, contract_path=path)
        self.assertIn("image_orientation", str(cm.exception))

    def test_md5_mismatch_is_rejected(self):
        bad = dict(self.contract, eval_mjcf_md5="0" * 32)
        path = Path(self.tmp.name) / "bad2.json"
        path.write_text(json.dumps(bad))
        from goal_client import GoalLiberoModelClient

        with self.assertRaises(RuntimeError):
            GoalLiberoModelClient("h", 1, contract_path=path)

    def test_missing_contract_has_no_fallback(self):
        from goal_client import GoalLiberoModelClient

        os.environ.pop("GOAL4D_CONTRACT", None)
        with self.assertRaises(RuntimeError):
            GoalLiberoModelClient("h", 1)

    def test_rotate_images_true_is_still_refused(self):
        with self.assertRaises(RuntimeError):
            self._client(rotate_images=True)


# ---------------------------------------------------------------------------
class F_ImportHook(unittest.TestCase):
    def test_parent_stays_mujoco_free_and_child_gets_the_goal_client(self):
        code = f"""
import sys, os
sys.path.insert(0, {str(_HERE)!r}); sys.path.insert(0, {str(ROOT)!r}); sys.path.insert(0, {str(ROOT / 'src')!r})
import eval_goal_plus as g
g.install_hook()
assert 'mujoco' not in sys.modules, 'hook import pulled mujoco into the parent'
import evaluation.LIBERO2.model2libero_interface as m
assert m.LiberoModelClient.__name__ == 'GoalLiberoModelClient', m.LiberoModelClient
print('OK')
"""
        r = subprocess.run([PY, "-c", code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertIn("OK", r.stdout)

    def test_launcher_requires_contract(self):
        env = {k: v for k, v in os.environ.items() if k != "GOAL4D_CONTRACT"}
        r = subprocess.run([PY, str(_HERE / "eval_goal_plus.py"), "--"], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 2)

    def test_eval_script_still_lazy_imports_the_client(self):
        """If someone moves the import to module level, fork() breaks and the hook no longer applies."""
        src = (ROOT / "evaluation/LIBERO-plus2/eval_libero_plus.py").read_text()
        self.assertIsNone(re.search(r"^(?:from|import)\s+\S*model2libero_interface", src, flags=re.M),
                          "client import moved to module level: fork() would break")
        self.assertIn("from evaluation.LIBERO2.model2libero_interface import LiberoModelClient", src)
        self.assertIn("client.push_keypoint()", src)


# ---------------------------------------------------------------------------
@unittest.skipIf(os.environ.get("SKIP_SLOW") == "1", "SKIP_SLOW=1")
class G_EndToEndSmall(unittest.TestCase):
    """13 (raw) and 8 (rlds) episodes through the real LeRobot video writer, then the full verifier."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.raw = Path(cls.tmp.name) / "raw"
        cls.rlds = Path(cls.tmp.name) / "rlds"
        # first 13 episodes hold 4 "language" (unperturbed camera) episodes: indices 3, 6, 8, 12
        for dest, orient, n_eps in ((cls.raw, "raw", 13), (cls.rlds, "rlds", 8)):
            r = subprocess.run(
                [PY, str(_HERE / "generate_goal_4d.py"), "--dest", str(dest), "--max-episodes", str(n_eps),
                 "--image-orientation", orient, "--force"],
                capture_output=True, text=True,
            )
            assert r.returncode == 0, r.stderr[-1500:]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _verify(self, dataset):
        return subprocess.run(
            [PY, str(_HERE / "verify_goal_dataset.py"), "--dataset", str(dataset), "--n-probe", "4"],
            capture_output=True, text=True,
        )

    def test_raw_dataset_passes_every_check(self):
        r = self._verify(self.raw)
        self.assertEqual(r.returncode, 0, r.stdout[-2500:] + r.stderr[-800:])
        self.assertIn("V18.training_window", r.stdout)
        self.assertNotIn("[FAIL]", r.stdout)

    def test_contract_says_panda_raw_h200(self):
        c = json.loads((self.raw / "meta/goal_train_eval_contract.json").read_text())
        self.assertEqual((c["robot_type"], c["image_orientation"], c["keypoint_history_max_len"]), ("panda", "raw", 200))
        info = json.loads((self.raw / "meta/info.json").read_text())
        self.assertEqual(info["robot_type"], "panda")
        self.assertEqual(info["features"]["observation.images.image"]["dtype"], "video")

    def test_negative_control_wrong_orientation_is_caught(self):
        """A dataset written in RLDS orientation but declared raw must FAIL the verifier."""
        broken = Path(self.tmp.name) / "broken"
        import shutil

        shutil.copytree(self.rlds, broken)
        cp = broken / "meta/goal_train_eval_contract.json"
        c = json.loads(cp.read_text())
        c["image_orientation"] = "raw"
        cp.write_text(json.dumps(c))
        r = self._verify(broken)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("[FAIL] V15.pixels_agentview", r.stdout)
        self.assertIn("[FAIL] V16.geometry_agentview", r.stdout)

    def test_rlds_orientation_dataset_is_self_consistent_when_declared_honestly(self):
        r = self._verify(self.rlds)
        # the verifier still fails V04 (contract.image_orientation != global constant) but geometry agrees
        self.assertIn("[PASS] V16.geometry_agentview", r.stdout)
        self.assertIn("[FAIL] V04.contract", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
