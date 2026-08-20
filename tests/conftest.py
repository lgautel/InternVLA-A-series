"""Shared pytest fixtures for the GeoPredict 3D keypoint fusion test suite.

Two tiers of fixtures are provided:

1. **Lightweight tensor-level fixtures** (:func:`make_tiny_qwen35_config`,
   :func:`make_tiny_expert`) -- build a tiny, CPU-friendly ``Qwen3_5TextModel`` (no vocabulary,
   no vision tower) with random weights. Used by tests that only exercise
   ``compute_layer_complete_3path`` / ``make_att_2d_masks`` / expert construction directly,
   without needing a full :class:`InternVLAA15` policy.
2. **Full tiny VLM checkpoint fixtures** (:func:`tiny_qwen35_checkpoint_dir`,
   :func:`tiny_internvla_a15_config`, :func:`tiny_internvla_a15_model`) -- build (once, cached
   on disk) a real (tiny) ``Qwen3_5ForConditionalGeneration`` checkpoint + real Qwen3.5 tokenizer,
   so that :class:`InternVLAA15WithExpertModel`/:class:`InternVLAA15` can be constructed exactly
   as in production (via ``from_pretrained``), just with a handful of layers/small hidden sizes.
   Used for end-to-end integration tests (Step 3/5/6/7).

See ``b/d/itrnVLA15_GeoP_3dtrj_3cn2_rbt2stak3_LOG.md`` §"单元测试基础设施" for the rationale
behind this split and for why the tiny checkpoint is cached under
``/mnt/r/CKPT/qwen35_tiny`` (building a fresh ``Qwen3_5ForConditionalGeneration`` + tokenizer
resize on every test run would dominate test runtime).
"""

from __future__ import annotations

import os

import pytest
import torch

TINY_QWEN35_CKPT_DIR = os.environ.get(
    "TINY_QWEN35_CKPT_DIR", "/mnt/r/CKPT/qwen35_tiny"
)


def make_tiny_qwen35_config(
    hidden_size=64,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=32,
    intermediate_size=128,
    num_hidden_layers=4,
    layer_types=None,
):
    """Build a tiny Qwen3.5 *text-only* HF config for fast CPU unit tests.

    Default: 4 layers, 3x linear_attention + 1x full_attention (mirrors the production
    ``(3+1)xN`` layer_types pattern at a much smaller scale).
    """
    from transformers import CONFIG_MAPPING

    if layer_types is None:
        layer_types = ["linear_attention"] * 3 + ["full_attention"]

    config = CONFIG_MAPPING["qwen3_5_text"]()
    config.hidden_size = hidden_size
    config.num_attention_heads = num_attention_heads
    config.num_key_value_heads = num_key_value_heads
    config.head_dim = head_dim
    config.intermediate_size = intermediate_size
    config.num_hidden_layers = num_hidden_layers
    config.layer_types = layer_types
    config.max_position_embeddings = 1024
    config.vocab_size = 1000
    config.rms_norm_eps = 1e-6
    config.use_cache = False
    config.attn_output_gate = True
    config.linear_conv_kernel_dim = 4
    config.linear_key_head_dim = max(8, head_dim // 2)
    config.linear_value_head_dim = max(8, head_dim // 2)
    config.linear_num_key_heads = num_key_value_heads
    config.linear_num_value_heads = num_key_value_heads
    return config


def make_tiny_expert(hf_config):
    """Build a tiny ``Qwen3_5TextModel`` (random weights, no embedding table) from an HF config."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    model = Qwen3_5TextModel(config=hf_config)
    model.embed_tokens = None
    return model


@pytest.fixture
def tiny_vlm_config():
    """VLM tiny config: hidden=64, heads=2, kv_heads=1, head_dim=32, 4 layers."""
    return make_tiny_qwen35_config(hidden_size=64)


@pytest.fixture
def tiny_expert_config():
    """Expert tiny config: hidden=32, heads=2, kv_heads=1, head_dim=32, 4 layers.

    hidden_size differs from the VLM's, but heads/head_dim must match (see design doc §3.2).
    """
    return make_tiny_qwen35_config(hidden_size=32, intermediate_size=64)


def build_tiny_qwen35_checkpoint(out_dir: str = TINY_QWEN35_CKPT_DIR) -> str:
    """Build (or reuse an existing) tiny random ``Qwen3_5ForConditionalGeneration`` checkpoint.

    The checkpoint uses the **real** ``Qwen/Qwen3.5-2B`` tokenizer (so ``ensure_qwen35_action_tokens``
    and the real FAST-token vocabulary range behave exactly as in production) but a randomly
    initialized model with only 4 decoder layers and hidden_size=64 (vs. the real 2048), so it
    builds/loads in a few seconds and fits comfortably even on CPU-only machines (though these
    tests are run on the project's H200 GPU per the task's venv).

    Idempotent: if ``out_dir/config.json`` already exists, the existing checkpoint is reused.
    """
    marker = os.path.join(out_dir, "config.json")
    if os.path.exists(marker):
        return out_dir

    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import Qwen3_5Tokenizer

    os.makedirs(out_dir, exist_ok=True)

    tok = Qwen3_5Tokenizer.from_pretrained("Qwen/Qwen3.5-2B")
    vocab_size = len(tok)

    text_config = dict(
        model_type="qwen3_5_text",
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_hidden_layers=4,
        layer_types=["full_attention", "linear_attention", "full_attention", "linear_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [3, 3, 2],
            "rope_theta": 10000000,
            "rope_type": "default",
            "partial_rotary_factor": 0.25,
        },
        partial_rotary_factor=0.25,
        tie_word_embeddings=True,
        use_cache=True,
        vocab_size=vocab_size,
        attn_output_gate=True,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        full_attention_interval=4,
        mlp_only_layers=[],
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        mamba_ssm_dtype="float32",
        eos_token_id=tok.eos_token_id,
        pad_token_id=None,
        bos_token_id=None,
        initializer_range=0.02,
        dtype="bfloat16",
    )
    vision_config = dict(
        model_type="qwen3_5",
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        depth=2,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        out_hidden_size=64,
        num_position_embeddings=64,
        hidden_act="gelu_pytorch_tanh",
        deepstack_visual_indexes=[],
        initializer_range=0.02,
    )
    config = Qwen3_5Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=True,
    )

    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(config).to(dtype=torch.bfloat16)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


@pytest.fixture(scope="session")
def tiny_qwen35_checkpoint_dir():
    """Session-scoped path to the (built-once) tiny Qwen3.5 VLM checkpoint directory."""
    return build_tiny_qwen35_checkpoint()


def make_tiny_internvla_a15_config(checkpoint_dir: str, **overrides):
    """Build an :class:`InternVLAA15Config` pointed at the tiny VLM checkpoint.

    ``action_loss_only=True`` is always forced so the (heavy, unrelated-to-this-task) WAN2.2
    video-foresight branch is never constructed -- see CLAUDE.md's "Real-Robot Inference" /
    "optimized backend" section for why ``action_loss_only`` is the documented way to skip it.
    """
    from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config

    kwargs = dict(
        vlm_model_name_or_path=checkpoint_dir,
        action_loss_only=True,
        dtype="bfloat16",
        tokenize_state=True,
        chunk_size=10,
        n_action_steps=10,
        num_learnable_tokens=6,
        action_expert_hidden_size=32,
        action_expert_intermediate_size=64,
        enable_keypoint_predictor=True,
        num_keypoint_joints=14,
        kpt_expert_hidden_size=32,
        kpt_expert_intermediate_size=64,
        keypoint_history_max_len=20,
        keypoint_track_embed_dim=16,
        keypoint_track_query_dim=32,
        keypoint_track_ff_dim=64,
        keypoint_track_num_heads=2,
    )
    kwargs.update(overrides)
    return InternVLAA15Config(**kwargs)


@pytest.fixture
def tiny_internvla_a15_config(tiny_qwen35_checkpoint_dir):
    """Default tiny :class:`InternVLAA15Config` with the keypoint predictor enabled (J=14)."""
    return make_tiny_internvla_a15_config(tiny_qwen35_checkpoint_dir)


@pytest.fixture
def tiny_internvla_a15_model(tiny_internvla_a15_config):
    """A fully constructed tiny :class:`InternVLAA15` (VLM + action expert + keypoint expert)."""
    from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import InternVLAA15

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = InternVLAA15(tiny_internvla_a15_config).to(device)
    return model
