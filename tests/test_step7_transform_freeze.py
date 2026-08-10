"""Step 7: data transform, freeze logic, and per-module optimizer-group tests."""

import numpy as np
import torch

from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import Extract3DKeypointTransformFn


class TestExtract3DKeypointTransformFn:
    """Test the transform that splits the delta-timestamp-stacked keypoint window into
    his_kpts/his_len/kpt_t/kpt_future/kpt_mask."""

    def test_phase1_no_keypoint_column_zero_fills(self):
        """Phase 1 (no observation.keypoint_3d in the dataset): all outputs zero-filled,
        kpt_mask=False."""
        fn = Extract3DKeypointTransformFn(num_joints=4, history_max_len=6, chunk_size=3)
        data = {"some_other_key": torch.zeros(1)}
        out = fn(data)

        assert out["observation.his_kpts"].shape == (6, 4, 3)
        assert torch.equal(out["observation.his_kpts"], torch.zeros(6, 4, 3))
        assert out["observation.his_len"].item() == 0
        assert out["observation.kpt_t"].shape == (4, 3)
        assert out["observation.kpt_future"].shape == (3, 4, 3)
        assert out["observation.kpt_mask"].item() is False

    def test_phase2_full_history_no_padding(self):
        """When all H history frames are valid (no clamping), his_len == H and his_kpts holds
        the full chronological history at the front."""
        h, j, c = 4, 2, 3
        fn = Extract3DKeypointTransformFn(num_joints=j, history_max_len=h, chunk_size=c)

        total = h + 1 + c
        stacked = torch.arange(total * j * 3, dtype=torch.float32).reshape(total, j * 3)
        data = {
            "observation.keypoint_3d": stacked,
            "observation.keypoint_3d_is_pad": torch.zeros(total, dtype=torch.bool),
        }
        out = fn(data)

        assert out["observation.his_len"].item() == h
        expected_his = stacked[:h].reshape(h, j, 3)
        assert torch.equal(out["observation.his_kpts"], expected_his)
        assert torch.equal(out["observation.kpt_t"], stacked[h].reshape(j, 3))
        assert torch.equal(out["observation.kpt_future"], stacked[h + 1 : h + 1 + c].reshape(c, j, 3))
        assert out["observation.kpt_mask"].item() is True
        # The transform must consume (pop) the raw stacked keys.
        assert "observation.keypoint_3d" not in out
        assert "observation.keypoint_3d_is_pad" not in out

    def test_phase2_partial_history_clamped_at_episode_start(self):
        """When some of the earliest (most-negative-offset) history frames were clamped
        (is_pad=True) because they fall before the episode start, his_len < H and the valid,
        chronologically-ascending tail is packed at the FRONT of his_kpts (matching
        TrackEncoder's `points[i, :length]` convention), with the back zero-padded."""
        h, j, c = 5, 2, 2
        fn = Extract3DKeypointTransformFn(num_joints=j, history_max_len=h, chunk_size=c)

        total = h + 1 + c
        stacked = torch.arange(total * j * 3, dtype=torch.float32).reshape(total, j * 3)
        # First 3 of the H=5 history frames are clamped (i.e. before episode start).
        is_pad = torch.zeros(total, dtype=torch.bool)
        is_pad[:3] = True
        data = {"observation.keypoint_3d": stacked, "observation.keypoint_3d_is_pad": is_pad}
        out = fn(data)

        num_invalid = 3
        expected_his_len = h - num_invalid
        assert out["observation.his_len"].item() == expected_his_len

        his_kpts = out["observation.his_kpts"]
        assert his_kpts.shape == (h, j, 3)
        # Valid tail (originally at history-window positions [num_invalid:h)) should now be at
        # the FRONT.
        expected_valid_tail = stacked[num_invalid:h].reshape(expected_his_len, j, 3)
        assert torch.equal(his_kpts[:expected_his_len], expected_valid_tail)
        # Remainder must be zero-padded.
        assert torch.equal(his_kpts[expected_his_len:], torch.zeros(h - expected_his_len, j, 3))

    def test_numpy_input_is_handled(self):
        """__getitem__ may hand back numpy arrays (pre-collation); the transform must accept them."""
        h, j, c = 2, 2, 2
        fn = Extract3DKeypointTransformFn(num_joints=j, history_max_len=h, chunk_size=c)
        total = h + 1 + c
        stacked_np = np.arange(total * j * 3, dtype=np.float32).reshape(total, j * 3)
        data = {"observation.keypoint_3d": stacked_np}
        out = fn(data)
        assert isinstance(out["observation.his_kpts"], torch.Tensor)
        assert out["observation.kpt_mask"].item() is True


class TestFreezeKeypointModules:
    """Test InternVLAA15.set_requires_grad's freeze_keypoint_modules handling."""

    def test_freeze_keypoint_modules_true(self, tiny_qwen35_checkpoint_dir):
        from tests.conftest import make_tiny_internvla_a15_config

        from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import InternVLAA15

        cfg = make_tiny_internvla_a15_config(tiny_qwen35_checkpoint_dir, freeze_keypoint_modules=True)
        torch.manual_seed(0)
        model = InternVLAA15(cfg)

        kpt_modules = [
            model.track_encoder,
            model.kpt_state_proj,
            model.keypoint_embedding,
            model.keypoint_out_proj,
            model.qwen3_5_with_expert.keypoint_expert,
        ]
        for module in kpt_modules:
            assert not module.training
            for p in module.parameters():
                assert p.requires_grad is False

        # Action expert must remain trainable (freeze_keypoint_modules should not affect it).
        assert any(p.requires_grad for p in model.qwen3_5_with_expert.action_expert.parameters())

    def test_freeze_keypoint_modules_false_by_default(self, tiny_internvla_a15_model):
        model = tiny_internvla_a15_model
        assert any(p.requires_grad for p in model.track_encoder.parameters())
        assert any(p.requires_grad for p in model.qwen3_5_with_expert.keypoint_expert.parameters())


class TestGetOptimParamsLrGrouping:
    """Test InternVLAA15Policy.get_optim_params' per-module LR-scale grouping."""

    def _build_policy(self, tiny_internvla_a15_config):
        from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import InternVLAA15Policy

        policy = InternVLAA15Policy(tiny_internvla_a15_config)
        return policy

    def test_flat_iterator_when_scales_all_default(self, tiny_qwen35_checkpoint_dir):
        """When all LR scales are 1.0 (or keypoint predictor disabled), get_optim_params must
        preserve the exact historical behavior: a flat parameter iterator (not a list of dict
        groups), so existing (non-keypoint) training configs are unaffected."""
        from tests.conftest import make_tiny_internvla_a15_config

        cfg = make_tiny_internvla_a15_config(tiny_qwen35_checkpoint_dir, enable_keypoint_predictor=False)
        policy = self._build_policy(cfg)
        result = policy.get_optim_params()
        assert not isinstance(result, list)

    def test_grouped_when_lr_scales_differ(self, tiny_qwen35_checkpoint_dir):
        from tests.conftest import make_tiny_internvla_a15_config

        cfg = make_tiny_internvla_a15_config(
            tiny_qwen35_checkpoint_dir,
            track_encoder_lr_scale=0.1,
            kpt_expert_lr_scale=0.5,
            optimizer_lr=1e-4,
        )
        policy = self._build_policy(cfg)
        groups = policy.get_optim_params()
        assert isinstance(groups, list)
        assert all("params" in g and "lr" in g for g in groups)

        lr_by_group = {tuple(id(p) for p in g["params"]): g["lr"] for g in groups}
        track_encoder_param_ids = {id(p) for p in policy.model.track_encoder.parameters()}
        # get_optim_params' "kpt_expert" group is everything in `kpt_modules` MINUS
        # track_encoder (which gets its own dedicated group) -- i.e. kpt_state_proj +
        # keypoint_embedding + keypoint_out_proj + qwen3_5_with_expert.keypoint_expert.
        kpt_expert_param_ids = {
            id(p)
            for m in (
                policy.model.kpt_state_proj,
                policy.model.keypoint_embedding,
                policy.model.keypoint_out_proj,
                policy.model.qwen3_5_with_expert.keypoint_expert,
            )
            for p in m.parameters()
        }

        found_track_encoder_group = False
        found_kpt_expert_group = False
        for g in groups:
            group_ids = {id(p) for p in g["params"]}
            if group_ids == track_encoder_param_ids:
                assert abs(g["lr"] - 1e-4 * 0.1) < 1e-12
                found_track_encoder_group = True
            if group_ids == kpt_expert_param_ids:
                assert abs(g["lr"] - 1e-4 * 0.5) < 1e-12
                found_kpt_expert_group = True
        assert found_track_encoder_group
        assert found_kpt_expert_group

    def test_all_trainable_params_are_covered_exactly_once(self, tiny_qwen35_checkpoint_dir):
        from tests.conftest import make_tiny_internvla_a15_config

        cfg = make_tiny_internvla_a15_config(
            tiny_qwen35_checkpoint_dir, track_encoder_lr_scale=0.1, kpt_expert_lr_scale=0.5
        )
        policy = self._build_policy(cfg)
        groups = policy.get_optim_params()

        grouped_ids = []
        for g in groups:
            grouped_ids.extend(id(p) for p in g["params"])
        trainable_ids = [id(p) for p in policy.parameters() if p.requires_grad]

        assert sorted(grouped_ids) == sorted(trainable_ids)
        assert len(grouped_ids) == len(set(grouped_ids))  # no duplicates across groups
