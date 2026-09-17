"""Unit tests for evaluation/LIBERO2/keypoint_utils.py.

KeypointHistory needs only numpy. StandaloneFK / resolve_eef_body_name need mujoco
and the Lift MJCF (skip if missing).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.LIBERO2.keypoint_utils import (
    DEFAULT_KEYPOINT_BODIES,
    DEFAULT_R_PAD,
    EEF_BODY_CANDIDATES,
    KEYPOINT_ARM_BODY_NAMES,
    KEYPOINT_DIM,
    KeypointHistory,
    _DEFAULT_MJCF_PATH,
)
from evaluation.LIBERO2.train_eval_extra_contract import (
    first_request_his_len,
    history_excludes_current,
    simulate_eval_history_loop,
)


def _mujoco_available() -> bool:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return False
    return _DEFAULT_MJCF_PATH.exists()


# --------------- KeypointHistory ---------------

class TestKeypointHistory:
    def test_reset_produces_zeros(self):
        h = KeypointHistory(max_len=5, num_kpts=8, kpt_dim=7)
        buf, length = h.get_history()
        assert buf.shape == (5, 8, 7)
        assert length == 0
        np.testing.assert_array_equal(buf, 0)

    def test_push_and_length(self):
        h = KeypointHistory(max_len=5, num_kpts=2, kpt_dim=3)
        frame = np.ones((2, 3), dtype=np.float32)
        h.push(frame)
        buf, length = h.get_history()
        assert length == 1
        np.testing.assert_allclose(buf[0], frame)
        np.testing.assert_array_equal(buf[1:], 0)

    def test_push_fills_in_order(self):
        h = KeypointHistory(max_len=4, num_kpts=1, kpt_dim=1)
        for i in range(3):
            h.push(np.array([[float(i + 1)]], dtype=np.float32))
        buf, length = h.get_history()
        assert length == 3
        assert buf[0, 0, 0] == 1.0
        assert buf[1, 0, 0] == 2.0
        assert buf[2, 0, 0] == 3.0
        assert buf[3, 0, 0] == 0.0

    def test_rolling_buffer_overflow(self):
        h = KeypointHistory(max_len=3, num_kpts=1, kpt_dim=1)
        for i in range(5):
            h.push(np.array([[float(i)]], dtype=np.float32))
        buf, length = h.get_history()
        assert length == 3
        assert buf[0, 0, 0] == 2.0
        assert buf[1, 0, 0] == 3.0
        assert buf[2, 0, 0] == 4.0

    def test_reset_clears_buffer(self):
        h = KeypointHistory(max_len=3, num_kpts=1, kpt_dim=1)
        h.push(np.array([[1.0]], dtype=np.float32))
        h.reset()
        buf, length = h.get_history()
        assert length == 0
        np.testing.assert_array_equal(buf, 0)

    def test_get_history_returns_copy(self):
        h = KeypointHistory(max_len=3, num_kpts=1, kpt_dim=1)
        h.push(np.array([[5.0]], dtype=np.float32))
        buf1, _ = h.get_history()
        buf2, _ = h.get_history()
        buf1[0, 0, 0] = 999.0
        assert buf2[0, 0, 0] == 5.0

    def test_his_len_property(self):
        h = KeypointHistory(max_len=10, num_kpts=1, kpt_dim=1)
        assert h.his_len == 0
        for _ in range(7):
            h.push(np.array([[0.0]], dtype=np.float32))
        assert h.his_len == 7

    def test_his_len_capped_at_max(self):
        h = KeypointHistory(max_len=3, num_kpts=1, kpt_dim=1)
        for _ in range(10):
            h.push(np.array([[0.0]], dtype=np.float32))
        assert h.his_len == 3


class TestEvalHistorySchedule:
    def test_request_excludes_current(self):
        assert history_excludes_current()

    def test_wait_10_gives_his_len_10(self):
        assert first_request_his_len(10) == 10

    def test_current_is_next_id_after_wait(self):
        hist, current = simulate_eval_history_loop(num_wait=10, num_action=1)[0]
        assert hist == list(range(10))
        assert current == 10
        assert current not in hist


class TestEefCandidates:
    def test_candidates_match_generate_script(self):
        assert EEF_BODY_CANDIDATES == ("gripper0_eef", "gripper0_right_eef")

    def test_default_lift_list_has_gripper0_eef(self):
        assert DEFAULT_KEYPOINT_BODIES[-1] == "gripper0_eef"
        assert KEYPOINT_DIM == 7
        assert len(KEYPOINT_ARM_BODY_NAMES) == 7
        assert abs(DEFAULT_R_PAD - 1.8212722539901733) < 1e-10


@pytest.mark.skipif(not _mujoco_available(), reason="mujoco or Lift MJCF missing")
class TestStandaloneFkResolve:
    def test_lift_mjcf_resolves_gripper0_eef(self):
        import mujoco

        from evaluation.LIBERO2.keypoint_utils import StandaloneFK, resolve_eef_body_name

        model = mujoco.MjModel.from_xml_path(str(_DEFAULT_MJCF_PATH))
        assert resolve_eef_body_name(model) == "gripper0_eef"
        fk = StandaloneFK()
        assert fk.body_names[-1] == "gripper0_eef"
        q = np.zeros(9, dtype=np.float64)
        kpts = fk.extract(q)
        assert kpts.shape == (8, 7)
        assert kpts.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
