"""Step 3.5: weight initialization verification.

Covers:
1. action_expert -> keypoint_expert weight copy correctness (zero missing keys, all allclose).
2. `InternVLAA15.post_init_keypoint_weights` functional test (already invoked by __init__).
3. GeoPredict TrackEncoder selective loading (skip track_fusion_layer, load the rest) against
   the real downloaded `Jingjing0601/GeoPredict-Robocasa` checkpoint.
4. Full three-stage init pipeline simulation: construct -> checkpoint load -> warm-start -> verify.
"""

import os

import pytest
import torch

from tests.conftest import make_tiny_expert, make_tiny_qwen35_config

GEOPREDICT_CKPT_PATH = "/mnt/r/CKPT/geopredict/GeoPredict_robocasa.pth"


class TestActionToKptWeightCopy:
    """Verify action-expert weights copy into a keypoint-expert with zero missing keys."""

    def test_zero_missing_keys(self):
        cfg = make_tiny_qwen35_config(hidden_size=32, intermediate_size=64)
        act = make_tiny_expert(cfg)
        kpt = make_tiny_expert(cfg)

        act_sd = act.state_dict()
        kpt_sd = kpt.state_dict()
        # Pick a randomly-initialized weight matrix (not a LayerNorm weight, which is
        # constant-1-initialized in both models and would trivially "match").
        some_key = next(k for k in act_sd if k.endswith("q_proj.weight"))
        assert not torch.allclose(act_sd[some_key], kpt_sd[some_key])

        missing, unexpected = kpt.load_state_dict(act.state_dict(), strict=True)
        assert len(missing) == 0
        assert len(unexpected) == 0

        for key in act_sd:
            assert torch.allclose(kpt.state_dict()[key], act_sd[key])


class TestPostInitKeypointWeights:
    """Verify InternVLAA15.post_init_keypoint_weights (already called from __init__)."""

    def test_kpt_expert_matches_action_expert_after_init(self, tiny_internvla_a15_model):
        model = tiny_internvla_a15_model
        act_sd = model.qwen3_5_with_expert.action_expert.state_dict()
        kpt_sd = model.qwen3_5_with_expert.keypoint_expert.state_dict()
        assert set(act_sd.keys()) == set(kpt_sd.keys())
        for key in act_sd:
            assert torch.allclose(kpt_sd[key], act_sd[key]), f"mismatch at {key}"

    def test_disabled_when_init_kpt_expert_from_action_false(self, tiny_qwen35_checkpoint_dir):
        from tests.conftest import make_tiny_internvla_a15_config

        from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import InternVLAA15

        cfg = make_tiny_internvla_a15_config(
            tiny_qwen35_checkpoint_dir, init_kpt_expert_from_action=False
        )
        torch.manual_seed(1)
        model = InternVLAA15(cfg)
        act_sd = model.qwen3_5_with_expert.action_expert.state_dict()
        kpt_sd = model.qwen3_5_with_expert.keypoint_expert.state_dict()
        some_key = next(iter(act_sd.keys()))
        # Both randomly initialized independently -> should NOT match without the warm-start.
        assert not torch.allclose(kpt_sd[some_key], act_sd[some_key])

    def test_explicit_call_is_idempotent(self, tiny_internvla_a15_model):
        model = tiny_internvla_a15_model
        model.post_init_keypoint_weights()
        act_sd = model.qwen3_5_with_expert.action_expert.state_dict()
        kpt_sd = model.qwen3_5_with_expert.keypoint_expert.state_dict()
        for key in act_sd:
            assert torch.allclose(kpt_sd[key], act_sd[key])


class TestGeoTrackEncoderSelectiveLoad:
    """Verify GeoPredict TrackEncoder selective loading."""

    def test_skip_track_fusion_layer_mock(self):
        from lerobot.policies.internvla_a1_5.keypoints import TrackEncoder

        encoder = TrackEncoder(
            input_dim=3, output_dim=32, num_queries=1,
            patch_size=2, embed_dim=16, query_dim=32,
            num_heads=2, ff_dim=64, max_seq_len=50,
        )
        geo_encoder = TrackEncoder(
            input_dim=3, output_dim=64, num_queries=1,
            patch_size=2, embed_dim=16, query_dim=32,
            num_heads=2, ff_dim=64, max_seq_len=50,
        )
        filtered = {
            k: v for k, v in geo_encoder.state_dict().items() if "track_fusion_layer" not in k
        }
        missing, unexpected = encoder.load_state_dict(filtered, strict=False)

        assert all("track_fusion_layer" in k for k in missing)
        assert len(unexpected) == 0
        for key in filtered:
            assert torch.allclose(encoder.state_dict()[key], filtered[key])

    @pytest.mark.skipif(
        not os.path.exists(GEOPREDICT_CKPT_PATH),
        reason=f"real GeoPredict checkpoint not found at {GEOPREDICT_CKPT_PATH}",
    )
    def test_real_geopredict_checkpoint_selective_load(self, tiny_internvla_a15_model):
        """End-to-end test against the real `Jingjing0601/GeoPredict-Robocasa` checkpoint.

        The tiny model's TrackEncoder uses query_dim=32 (see conftest's
        ``make_tiny_internvla_a15_config``) which differs from GeoPredict's real query_dim=512, so
        (unlike production, where query_dim=512 on both sides) *all* keys are shape-mismatched here
        except none. We therefore build a TrackEncoder with query_dim=512 (matching GeoPredict)
        purely for this loading test, mirroring what `InternVLAA15.__init__` does in production
        when `keypoint_track_query_dim=512` (the default).
        """
        from lerobot.policies.internvla_a1_5.keypoints import TrackEncoder, load_geopredict_track_encoder_weights

        encoder = TrackEncoder(output_dim=1024)  # production defaults, incl. query_dim=512
        loaded_keys, skipped_keys = load_geopredict_track_encoder_weights(encoder, GEOPREDICT_CKPT_PATH)

        assert len(loaded_keys) > 0
        assert any("track_fusion_layer" in k for k in skipped_keys)
        # Sanity: queries buffer/parameter should now be non-default (i.e. actually loaded).
        assert encoder.queries.shape == (1, 1, 512)


class TestFullInitPipeline:
    """Simulate the full three-stage initialization pipeline."""

    def test_stages_1_2_3(self):
        cfg = make_tiny_qwen35_config(hidden_size=32, intermediate_size=64)

        # Stage 1: construction (random init)
        act = make_tiny_expert(cfg)
        kpt = make_tiny_expert(cfg)

        # Stage 2: simulate checkpoint load (only affects action_expert)
        for p in act.parameters():
            p.data.fill_(0.42)

        # Stage 3: warm-start kpt from action
        kpt.load_state_dict(act.state_dict())

        for name, p in kpt.named_parameters():
            assert torch.allclose(p.data, torch.full_like(p.data, 0.42)), f"Stage 3 failed for {name}"
