"""Step 6: inference-path tests.

InternVLAA15's inference path does NOT use a dedicated ``compute_layer_suffix_only`` function
(unlike the general design doc's pseudocode) -- instead ``InternVLAA15WithExpertModel._forward_3path``
exposes three *single-path* branches (prefix-only / keypoint-only / action-only), each of which can
read a ``past_key_values`` cache produced by a *different* expert and append its own newly-computed
K/V to it (see ``_forward_3path``'s docstring). ``InternVLAA15.sample_actions``/``denoise_step``
chain these three single-path calls together to reproduce the same joint-attention semantics as the
training-time joint 3-path forward, without recomputing the prefix/keypoint segments at every one of
the ``num_inference_steps`` action-expert calls.

This file tests that chaining directly (Step 6's real analogue of "compute_layer_suffix_only"), plus
a full ``sample_actions`` smoke test on the tiny model.
"""

import torch

from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import (
    ActionExpertConfig,
    InternVLAA15WithExpertModel,
    KeypointExpertConfig,
    make_att_2d_masks,
)


class TestCachedKVThreePathChaining:
    """Test the prefix-only -> keypoint-only -> action-only single-path branches of
    ``_forward_3path``, chained via a shared ``past_key_values`` cache (the actual mechanism
    ``InternVLAA15.sample_actions``/``denoise_step`` rely on for cached-KV inference)."""

    def _build_model(self, tiny_qwen35_checkpoint_dir):
        return InternVLAA15WithExpertModel(
            vlm_model_name_or_path=tiny_qwen35_checkpoint_dir,
            action_expert_config=ActionExpertConfig(hidden_size=32, intermediate_size=64),
            keypoint_expert_config=KeypointExpertConfig(hidden_size=32, intermediate_size=64),
            precision="bfloat16",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def test_prefix_only_returns_cache(self, tiny_qwen35_checkpoint_dir):
        model = self._build_model(tiny_qwen35_checkpoint_dir)
        device = next(model.parameters()).device
        vlm_hidden = model.qwen3_5.config.text_config.hidden_size

        # sample_actions/denoise_step always force `_attn_implementation = "eager"` on the
        # relevant expert before a cached-KV call (see production code) so that the additive
        # float mask is handled directly rather than via SDPA (which requires the mask dtype to
        # exactly match the query's).
        model.qwen3_5.language_model.config._attn_implementation = "eager"

        B, P = 1, 5
        prefix_embs = torch.randn(B, P, vlm_hidden, dtype=torch.bfloat16, device=device)
        pad_masks = torch.ones(B, P, dtype=torch.bool, device=device)
        att_masks = torch.ones(B, P, device=device)
        mask_2d = make_att_2d_masks(pad_masks, att_masks)
        mask_4d = mask_2d.unsqueeze(1).float()
        mask_4d = torch.where(mask_4d.bool(), torch.zeros_like(mask_4d), torch.full_like(mask_4d, -1e9))
        pos_ids = torch.arange(P, device=device).unsqueeze(0).unsqueeze(0).repeat(3, B, 1).float()

        outputs, past_kv = model.forward(
            attention_mask=mask_4d,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None, None],
            use_cache=True,
        )
        assert outputs[0].shape == (B, P, vlm_hidden)
        assert outputs[1] is None and outputs[2] is None
        assert past_kv is not None

    def test_prefix_then_kpt_then_action_cached_chain_shapes(self, tiny_qwen35_checkpoint_dir):
        """Chain all three single-path branches via a shared cache and verify shapes at each stage
        (mirrors what sample_actions/denoise_step do internally)."""
        model = self._build_model(tiny_qwen35_checkpoint_dir)
        device = next(model.parameters()).device
        vlm_hidden = model.qwen3_5.config.text_config.hidden_size

        model.qwen3_5.language_model.config._attn_implementation = "eager"
        model.keypoint_expert.config._attn_implementation = "eager"
        model.action_expert.config._attn_implementation = "eager"

        B, P, K, A = 1, 5, 17, 10

        # Stage 1: prefix-only, populates the cache.
        prefix_embs = torch.randn(B, P, vlm_hidden, dtype=torch.bfloat16, device=device)
        prefix_pad = torch.ones(B, P, dtype=torch.bool, device=device)
        prefix_att = torch.ones(B, P, device=device)
        prefix_mask_2d = make_att_2d_masks(prefix_pad, prefix_att)
        prefix_mask_4d = prefix_mask_2d.unsqueeze(1).float()
        prefix_mask_4d = torch.where(
            prefix_mask_4d.bool(), torch.zeros_like(prefix_mask_4d), torch.full_like(prefix_mask_4d, -1e9)
        )
        prefix_pos_ids = torch.arange(P, device=device).unsqueeze(0).unsqueeze(0).repeat(3, B, 1).float()

        _, past_kv = model.forward(
            attention_mask=prefix_mask_4d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None, None],
            use_cache=True,
        )

        # Stage 2: keypoint-only, attends to [cached prefix, kpt] and appends its own K/V.
        kpt_embs = torch.randn(B, K, 32, dtype=torch.bfloat16, device=device)
        kpt_pad = torch.ones(B, K, dtype=torch.bool, device=device)
        kpt_att = torch.ones(B, K, device=device)
        kpt_att_2d = make_att_2d_masks(kpt_pad, kpt_att)  # [B, K, K]
        prefix_pad_2d = prefix_pad[:, None, :].expand(B, K, P)
        kpt_full_2d = torch.cat([prefix_pad_2d, kpt_att_2d], dim=2)  # [B, K, P+K]
        kpt_full_4d = kpt_full_2d.unsqueeze(1).float()
        kpt_full_4d = torch.where(kpt_full_4d.bool(), torch.zeros_like(kpt_full_4d), torch.full_like(kpt_full_4d, -1e9))
        kpt_pos_ids = (torch.arange(1, K + 1, device=device) + P).unsqueeze(0).unsqueeze(0).repeat(3, B, 1).float()

        kpt_outputs, past_kv = model.forward(
            attention_mask=kpt_full_4d,
            position_ids=kpt_pos_ids,
            past_key_values=past_kv,
            inputs_embeds=[None, kpt_embs, None],
            use_cache=True,
        )
        assert kpt_outputs[1].shape == (B, K, 32)

        # Stage 3: action-only, attends to [cached prefix, cached kpt, action].
        act_embs = torch.randn(B, A, 32, dtype=torch.bfloat16, device=device)
        act_pad = torch.ones(B, A, dtype=torch.bool, device=device)
        act_att = torch.ones(B, A, device=device)
        act_att_2d = make_att_2d_masks(act_pad, act_att)
        prefix_kpt_pad_2d = torch.cat([prefix_pad, kpt_pad], dim=1)[:, None, :].expand(B, A, P + K)
        act_full_2d = torch.cat([prefix_kpt_pad_2d, act_att_2d], dim=2)  # [B, A, P+K+A]
        act_full_4d = act_full_2d.unsqueeze(1).float()
        act_full_4d = torch.where(act_full_4d.bool(), torch.zeros_like(act_full_4d), torch.full_like(act_full_4d, -1e9))
        act_pos_ids = (torch.arange(1, A + 1, device=device) + P + K).unsqueeze(0).unsqueeze(0).repeat(3, B, 1).float()

        act_outputs, _ = model.forward(
            attention_mask=act_full_4d,
            position_ids=act_pos_ids,
            past_key_values=past_kv,
            inputs_embeds=[None, None, act_embs],
            use_cache=False,
        )
        assert act_outputs[2].shape == (B, A, 32)


def _cast_top_level_projections_to_bf16(model):
    """`InternVLAA15`'s own top-level projections (action_in_proj, action_out_proj,
    action_time_mlp_*, kpt_state_proj, keypoint_embedding, keypoint_out_proj,
    learnable_tokens[_in_proj]) are intentionally left in float32 by ``__init__`` (only
    ``qwen3_5_with_expert``'s bf16 casting is handled by ``to_bfloat16_for_selected_params``,
    see design doc §training precision). In production, these layers end up in bf16 anyway once
    a bf16-trained checkpoint is loaded (`_load_as_safetensor`) — but our tiny test model is
    randomly initialized from scratch, never loaded from such a checkpoint. We replicate the
    "loaded from a bf16 checkpoint" end state here so the inference path (`sample_actions` /
    `denoise_step`, which -- like production -- does not itself re-cast `embed_suffix`'s output
    before feeding the bf16 action expert) can be exercised without a dtype mismatch.
    """
    # `action_out_proj`/`keypoint_out_proj` are always fed an explicitly-float32-cast input right
    # before use (see `InternVLAA15.forward`/`denoise_step`: `.to(dtype=torch.float32)` immediately
    # precedes both), so they must stay float32 too.
    keep_float32 = (
        "input_layernorm", "post_attention_layernorm", "model.norm",
        "action_out_proj", "keypoint_out_proj",
    )
    for name, param in model.named_parameters():
        if "qwen3_5_with_expert" in name:
            continue  # already handled by to_bfloat16_for_selected_params
        if any(k in name for k in keep_float32):
            continue
        param.data = param.data.to(dtype=torch.bfloat16)
    # Buffers (e.g. TrackEncoder's TimeEmbedding.pos_embedding sinusoidal table,
    # future_kpt_pos_embed) are not covered by named_parameters() and must be cast separately,
    # or ops like `key + key_pos_emb` inside MultiHeadAttention silently upcast to float32 and
    # then hit a bf16-weight Linear layer with a float32 input.
    for name, buf in model.named_buffers():
        if "qwen3_5_with_expert" in name:
            continue
        model_buf = buf
        if model_buf.dtype.is_floating_point:
            model_buf.data = model_buf.data.to(dtype=torch.bfloat16)


class TestSampleActionsEndToEnd:
    """Full `InternVLAA15.sample_actions` smoke test on the tiny model, with and without the
    keypoint predictor enabled."""

    def _fake_prefix_batch(self, model, bsize, device):
        vis_cfg = model.qwen3_5_with_expert.qwen3_5.config.vision_config
        patch_dim = vis_cfg.in_channels * vis_cfg.temporal_patch_size * vis_cfg.patch_size**2
        grid_t, grid_h, grid_w = 1, 2, 2
        num_patches = grid_t * grid_h * grid_w
        merged = num_patches // (vis_cfg.spatial_merge_size**2)

        image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]] * bsize, device=device)
        pixel_values = torch.randn(bsize, num_patches, patch_dim, device=device, dtype=torch.bfloat16)

        image_token_id = model.qwen3_5_with_expert.qwen3_5.config.image_token_id
        seq_len = merged + 4
        lang_tokens = torch.randint(0, 100, (bsize, seq_len), device=device)
        lang_tokens[:, :merged] = image_token_id
        lang_masks = torch.ones(bsize, seq_len, dtype=torch.bool, device=device)
        return pixel_values, image_grid_thw, lang_tokens, lang_masks

    def test_sample_actions_with_keypoint_predictor(
        self, tiny_internvla_a15_model, tiny_internvla_a15_config
    ):
        model = tiny_internvla_a15_model
        cfg = tiny_internvla_a15_config
        device = next(model.parameters()).device
        bsize = 2

        pixel_values, image_grid_thw, lang_tokens, lang_masks = self._fake_prefix_batch(
            model, bsize, device
        )
        # `sample_actions` derives its working dtype from `state.dtype` (see
        # `dtype = state.dtype` near the top of `sample_actions`) and does NOT separately cast
        # `embed_suffix`'s output before feeding the (bf16) action expert -- so callers must pass
        # state already in the model's compute dtype, exactly as `InternVLAA15Policy.prepare_state`
        # does in production (state tensors flow through in the batch's/model's bf16 dtype).
        state = torch.randn(bsize, cfg.max_state_dim, device=device, dtype=torch.bfloat16)
        his_kpts = torch.randn(
            bsize, cfg.keypoint_history_max_len, cfg.num_keypoint_joints, 3, device=device
        )
        his_len = torch.full((bsize,), cfg.keypoint_history_max_len, device=device, dtype=torch.long)

        _cast_top_level_projections_to_bf16(model)
        model.eval()
        actions = model.sample_actions(
            pixel_values, image_grid_thw, lang_tokens, lang_masks, state,
            num_steps=2, his_kpts=his_kpts, his_len=his_len,
        )
        assert actions.shape == (bsize, cfg.chunk_size, cfg.max_action_dim)
        assert torch.isfinite(actions).all()

    def test_sample_actions_without_keypoint_predictor(self, tiny_qwen35_checkpoint_dir):
        """Regression: the 2-path (no keypoint expert) inference path must still work unmodified."""
        from tests.conftest import make_tiny_internvla_a15_config

        from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import InternVLAA15

        cfg = make_tiny_internvla_a15_config(tiny_qwen35_checkpoint_dir, enable_keypoint_predictor=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(0)
        model = InternVLAA15(cfg).to(device)
        _cast_top_level_projections_to_bf16(model)
        model.eval()

        bsize = 2
        pixel_values, image_grid_thw, lang_tokens, lang_masks = self._fake_prefix_batch(
            model, bsize, device
        )
        state = torch.randn(bsize, cfg.max_state_dim, device=device, dtype=torch.bfloat16)

        actions = model.sample_actions(pixel_values, image_grid_thw, lang_tokens, lang_masks, state, num_steps=2)
        assert actions.shape == (bsize, cfg.chunk_size, cfg.max_action_dim)
        assert torch.isfinite(actions).all()
