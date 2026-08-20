"""Step 3: keypoint expert construction + embed_kpt_suffix tests."""

import torch

from tests.conftest import make_tiny_expert, make_tiny_qwen35_config


class TestKeypointExpertConfig:
    def test_head_params_from_vlm(self):
        """head_dim/num_heads/num_kv_heads must be inherited from the VLM."""
        from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import KeypointExpertConfig

        vlm_cfg = make_tiny_qwen35_config(
            hidden_size=64, num_attention_heads=2, num_key_value_heads=1, head_dim=32
        )
        kpt_cfg = KeypointExpertConfig(hidden_size=32, intermediate_size=64)
        kpt_cfg.num_attention_heads = vlm_cfg.num_attention_heads
        kpt_cfg.num_key_value_heads = vlm_cfg.num_key_value_heads
        kpt_cfg.head_dim = vlm_cfg.head_dim

        assert kpt_cfg.hidden_size == 32  # freely customizable
        assert kpt_cfg.num_attention_heads == 2  # inherited from VLM
        assert kpt_cfg.head_dim == 32  # inherited from VLM


class TestKeypointExpertModel:
    def test_build_tiny_kpt_expert(self):
        """Verify a tiny keypoint-expert Qwen3_5TextModel can be built."""
        cfg = make_tiny_qwen35_config(hidden_size=32, intermediate_size=64)
        expert = make_tiny_expert(cfg)

        assert expert.embed_tokens is None
        assert len(expert.layers) == 4
        # o_proj: num_heads * head_dim -> hidden_size; num_heads=2, head_dim=32 => attn_out=64
        assert expert.layers[3].self_attn.o_proj.in_features == 64
        assert expert.layers[3].self_attn.o_proj.out_features == 32

    def test_with_expert_model_builds_keypoint_expert(self, tiny_qwen35_checkpoint_dir):
        """InternVLAA15WithExpertModel should construct self.keypoint_expert when a
        KeypointExpertConfig is supplied, with head params inherited from the VLM."""
        from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import (
            ActionExpertConfig,
            InternVLAA15WithExpertModel,
            KeypointExpertConfig,
        )

        model = InternVLAA15WithExpertModel(
            vlm_model_name_or_path=tiny_qwen35_checkpoint_dir,
            action_expert_config=ActionExpertConfig(hidden_size=32, intermediate_size=64),
            keypoint_expert_config=KeypointExpertConfig(hidden_size=32, intermediate_size=64),
            precision="bfloat16",
        )
        assert model.keypoint_expert is not None
        assert model.keypoint_expert.embed_tokens is None
        vlm_text_cfg = model.qwen3_5.config.text_config
        assert model.keypoint_expert.config.num_attention_heads == vlm_text_cfg.num_attention_heads
        assert model.keypoint_expert.config.num_key_value_heads == vlm_text_cfg.num_key_value_heads
        assert model.keypoint_expert.config.head_dim == vlm_text_cfg.head_dim
        assert model.keypoint_expert.config.layer_types == vlm_text_cfg.layer_types

    def test_with_expert_model_no_keypoint_expert_by_default(self, tiny_qwen35_checkpoint_dir):
        """Without a KeypointExpertConfig, self.keypoint_expert must stay None (2-path backward
        compatibility)."""
        from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import (
            ActionExpertConfig,
            InternVLAA15WithExpertModel,
        )

        model = InternVLAA15WithExpertModel(
            vlm_model_name_or_path=tiny_qwen35_checkpoint_dir,
            action_expert_config=ActionExpertConfig(hidden_size=32, intermediate_size=64),
            precision="bfloat16",
        )
        assert model.keypoint_expert is None


class TestEmbedKptSuffix:
    """Test embed_kpt_suffix's output shapes and attention-mask pattern on a real tiny model."""

    def test_output_shape(self, tiny_internvla_a15_model, tiny_internvla_a15_config):
        cfg = tiny_internvla_a15_config
        j = cfg.num_keypoint_joints
        hidden = cfg.kpt_expert_hidden_size
        device = next(tiny_internvla_a15_model.parameters()).device

        bsize = 2
        state = torch.randn(bsize, cfg.max_state_dim, device=device)
        his_kpts = torch.randn(bsize, cfg.keypoint_history_max_len, j, 3, device=device)
        his_len = torch.tensor([cfg.keypoint_history_max_len, 3], device=device)

        kpt_embs, kpt_pad, kpt_att = tiny_internvla_a15_model.embed_kpt_suffix(state, his_kpts, his_len)

        expected_len = 1 + 2 * j
        assert kpt_embs.shape == (bsize, expected_len, hidden)
        assert kpt_pad.shape == (bsize, expected_len)
        assert kpt_att.shape == (bsize, expected_len)
        assert kpt_pad.dtype == torch.bool
        assert kpt_pad.all()  # no padding within the keypoint suffix itself

    def test_att_masks_pattern(self, tiny_internvla_a15_model, tiny_internvla_a15_config):
        """att_masks should be [1, 1, 0...0, 1, 0...0] with exactly 3 block boundaries."""
        cfg = tiny_internvla_a15_config
        j = cfg.num_keypoint_joints
        device = next(tiny_internvla_a15_model.parameters()).device

        state = torch.randn(1, cfg.max_state_dim, device=device)
        _, _, kpt_att = tiny_internvla_a15_model.embed_kpt_suffix(state)

        expected = [1] + [1] + [0] * (j - 1) + [1] + [0] * (j - 1)
        assert kpt_att.shape[1] == 1 + 2 * j
        assert kpt_att[0].tolist() == [float(x) for x in expected]
        assert kpt_att[0].sum().item() == 3

    def test_none_history_defaults_to_zero(self, tiny_internvla_a15_model, tiny_internvla_a15_config):
        """When his_kpts/his_len are None (Phase 1 dataloader path), embed_kpt_suffix should not
        crash and should treat it as zero history."""
        cfg = tiny_internvla_a15_config
        device = next(tiny_internvla_a15_model.parameters()).device
        state = torch.randn(3, cfg.max_state_dim, device=device)

        kpt_embs, kpt_pad, kpt_att = tiny_internvla_a15_model.embed_kpt_suffix(state, None, None)
        assert kpt_embs.shape == (3, 1 + 2 * cfg.num_keypoint_joints, cfg.kpt_expert_hidden_size)
        assert torch.isfinite(kpt_embs.float()).all()

    def test_get_keypoint_token_output_slices_last_j(self, tiny_internvla_a15_model, tiny_internvla_a15_config):
        cfg = tiny_internvla_a15_config
        j = cfg.num_keypoint_joints
        hidden = cfg.kpt_expert_hidden_size
        kpt_out = torch.randn(2, 1 + 2 * j, hidden)
        sliced = tiny_internvla_a15_model.get_keypoint_token_output(kpt_out)
        assert sliced.shape == (2, j, hidden)
        assert torch.equal(sliced, kpt_out[:, -j:])
