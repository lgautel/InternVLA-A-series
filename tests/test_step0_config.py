"""Step 0: verify keypoint-related config fields and the tiny fixtures."""

from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config


class TestKeypointConfigFields:
    """Verify existence and defaults of all newly added config fields."""

    def test_default_enable_keypoint_predictor(self):
        cfg = InternVLAA15Config()
        assert cfg.enable_keypoint_predictor is False
        assert cfg.num_keypoint_joints == 8
        assert cfg.kpt_loss_weight == 1.0

    def test_default_kpt_expert_dims(self):
        cfg = InternVLAA15Config()
        assert cfg.kpt_expert_hidden_size == 1024
        assert cfg.kpt_expert_intermediate_size == 3072

    def test_default_loss_weight_fields(self):
        cfg = InternVLAA15Config()
        assert cfg.action_loss_weight == 10.0
        assert cfg.kpt_loss_weight == 1.0
        assert cfg.kpt_future_loss_weight == 1.0

    def test_default_knowledge_insulation_switches(self):
        cfg = InternVLAA15Config()
        assert cfg.knowledge_insulation_kpt is False
        assert cfg.kpt_to_action_detach is False
        assert cfg.freeze_keypoint_modules is False

    def test_default_soft_ki_fields(self):
        cfg = InternVLAA15Config()
        assert cfg.ki_gradient_scale == 0.0
        assert cfg.ki_kpt_gradient_scale == 0.0

    def test_default_per_module_lr_scales(self):
        cfg = InternVLAA15Config()
        assert cfg.vlm_lr_scale == 1.0
        assert cfg.action_expert_lr_scale == 1.0
        assert cfg.kpt_expert_lr_scale == 1.0
        assert cfg.track_encoder_lr_scale == 1.0

    def test_default_weight_init_fields(self):
        cfg = InternVLAA15Config()
        assert cfg.init_kpt_expert_from_action is True
        assert cfg.geopredict_checkpoint_path is None

    def test_default_track_encoder_params(self):
        cfg = InternVLAA15Config()
        assert cfg.keypoint_track_input_dim == 3
        assert cfg.keypoint_track_patch_size == 4
        assert cfg.keypoint_track_embed_dim == 256
        assert cfg.keypoint_track_query_dim == 512
        assert cfg.keypoint_track_num_heads == 8
        assert cfg.keypoint_track_ff_dim == 1024
        assert cfg.keypoint_history_max_len == 1000

    def test_custom_values(self):
        cfg = InternVLAA15Config()
        cfg.enable_keypoint_predictor = True
        cfg.num_keypoint_joints = 14
        cfg.kpt_expert_hidden_size = 512
        assert cfg.enable_keypoint_predictor is True
        assert cfg.num_keypoint_joints == 14
        assert cfg.kpt_expert_hidden_size == 512

    def test_keypoint_3d_delta_indices_disabled(self):
        cfg = InternVLAA15Config()
        assert cfg.enable_keypoint_predictor is False
        assert cfg.keypoint_3d_delta_indices is None

    def test_keypoint_3d_delta_indices_enabled(self):
        """H+1+C indices spanning [-H, ..., -1, 0, 1, ..., C] (see property docstring)."""
        cfg = InternVLAA15Config(
            enable_keypoint_predictor=True,
            keypoint_history_max_len=5,
            chunk_size=3,
            n_action_steps=3,
        )
        indices = cfg.keypoint_3d_delta_indices
        assert indices == [-5, -4, -3, -2, -1, 0, 1, 2, 3]
        assert len(indices) == 5 + 1 + 3


class TestTinyFixture:
    """Verify the conftest fixtures can build tiny models."""

    def test_tiny_vlm_config_shape(self, tiny_vlm_config):
        assert tiny_vlm_config.hidden_size == 64
        assert tiny_vlm_config.num_attention_heads == 2
        assert tiny_vlm_config.num_key_value_heads == 1
        assert tiny_vlm_config.head_dim == 32
        assert tiny_vlm_config.num_hidden_layers == 4
        assert len(tiny_vlm_config.layer_types) == 4

    def test_tiny_expert_build(self, tiny_expert_config):
        from tests.conftest import make_tiny_expert

        model = make_tiny_expert(tiny_expert_config)
        assert model.embed_tokens is None
        assert len(model.layers) == 4
