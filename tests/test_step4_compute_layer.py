"""Step 4: compute_layer_complete_3path tests.

Uses tiny models (4 layers, hidden=64/32) on CPU/GPU. Tests the core three-path MoT layer
function directly (mocking the VLM as a lightweight namespace object exposing only
``.language_model``, since ``compute_layer_complete_3path`` never touches the VLM's vision
tower / lm_head).
"""

import pytest
import torch

from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import (
    compute_layer_complete_3path,
    make_att_2d_masks,
)
from tests.conftest import make_tiny_expert, make_tiny_qwen35_config


@pytest.fixture
def three_models():
    """Build 3 tiny models: VLM(hidden=64), kpt(hidden=32), act(hidden=32)."""
    vlm_cfg = make_tiny_qwen35_config(hidden_size=64)
    kpt_cfg = make_tiny_qwen35_config(hidden_size=32, intermediate_size=64)
    act_cfg = make_tiny_qwen35_config(hidden_size=32, intermediate_size=64)

    class FakeQwen35:
        pass

    vlm = FakeQwen35()
    vlm.language_model = make_tiny_expert(vlm_cfg)
    vlm.language_model.config = vlm_cfg

    kpt = make_tiny_expert(kpt_cfg)
    act = make_tiny_expert(act_cfg)

    return vlm, kpt, act, vlm_cfg


def _bool_pad(*sizes):
    return torch.ones(*sizes, dtype=torch.bool)


class TestComputeLayerLinear:
    """Test the linear-attention layer (layer_idx=1, per the 3-linear+1-full tiny layout).

    The ``causal_conv1d`` kernel backing Qwen3.5's linear-attention path is CUDA-only (see
    ``flash-linear-attention``/``causal-conv1d`` deps in CLAUDE.md's install steps), so this test
    requires a GPU and is skipped otherwise.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="linear_attention requires CUDA (causal_conv1d)")
    def test_output_shapes(self, three_models):
        vlm, kpt, act, vlm_cfg = three_models
        assert vlm_cfg.layer_types[0] == "linear_attention"
        device = "cuda"
        vlm.language_model = vlm.language_model.to(device)
        kpt = kpt.to(device)
        act = act.to(device)

        B, P, K, A = 1, 4, 17, 10

        prefix_embs = torch.randn(B, P, 64, requires_grad=True, device=device)
        kpt_embs = torch.randn(B, K, 32, requires_grad=True, device=device)
        act_embs = torch.randn(B, A, 32, requires_grad=True, device=device)

        total = P + K + A
        pad_masks = _bool_pad(B, total).to(device)
        att_masks = torch.ones(B, total, device=device)
        mask_2d = make_att_2d_masks(pad_masks, att_masks)
        mask_4d = mask_2d.unsqueeze(1).float()

        pos_ids = torch.arange(total, device=device).unsqueeze(0).unsqueeze(0).repeat(3, B, 1)

        outputs = compute_layer_complete_3path(
            layer_idx=0,  # linear_attention
            inputs_embeds=[prefix_embs, kpt_embs, act_embs],
            attention_mask=mask_4d,
            position_ids=pos_ids,
            qwen3_5=vlm,
            keypoint_expert=kpt,
            action_expert=act,
            prefix_len=P,
            kpt_len=K,
            linear_attn_mask=pad_masks.float(),
        )

        assert len(outputs) == 3
        assert outputs[0].shape == (B, P, 64)
        assert outputs[1].shape == (B, K, 32)
        assert outputs[2].shape == (B, A, 32)


class TestComputeLayerFull:
    """Test the full-attention layer (layer_idx=3) three-path cross-attention."""

    def _build_inputs(self):
        B, P, K, A = 1, 4, 17, 10
        prefix_embs = torch.randn(B, P, 64, requires_grad=True)
        kpt_embs = torch.randn(B, K, 32, requires_grad=True)
        act_embs = torch.randn(B, A, 32, requires_grad=True)

        pad_masks = _bool_pad(B, P + K + A)
        prefix_att = torch.ones(B, P)
        kpt_att = torch.tensor(
            [[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float
        )
        act_att = torch.tensor([[1] + [0] * 4 + [1] + [0] * 4], dtype=torch.float)
        att_masks = torch.cat([prefix_att, kpt_att, act_att], dim=1)

        mask_2d = make_att_2d_masks(pad_masks, att_masks)
        mask_4d = mask_2d.unsqueeze(1).float()
        mask_4d = torch.where(mask_4d.bool(), torch.zeros_like(mask_4d), torch.full_like(mask_4d, -1e9))

        total = P + K + A
        pos_ids = torch.arange(total).unsqueeze(0).unsqueeze(0).repeat(3, B, 1)
        return prefix_embs, kpt_embs, act_embs, mask_4d, pos_ids, P, K, A

    def test_output_shapes(self, three_models):
        vlm, kpt, act, _ = three_models
        prefix_embs, kpt_embs, act_embs, mask_4d, pos_ids, P, K, A = self._build_inputs()

        outputs = compute_layer_complete_3path(
            layer_idx=3,  # full_attention
            inputs_embeds=[prefix_embs, kpt_embs, act_embs],
            attention_mask=mask_4d,
            position_ids=pos_ids,
            qwen3_5=vlm,
            keypoint_expert=kpt,
            action_expert=act,
            prefix_len=P,
            kpt_len=K,
            use_sdpa=False,
        )

        assert len(outputs) == 3
        assert outputs[0].shape == (1, P, 64)
        assert outputs[1].shape == (1, K, 32)
        assert outputs[2].shape == (1, A, 32)

    def test_gradient_flow_no_ki(self, three_models):
        """Without KI, kpt loss should backprop into the VLM."""
        vlm, kpt, act, _ = three_models
        prefix_embs, kpt_embs, act_embs, mask_4d, pos_ids, P, K, A = self._build_inputs()

        outputs = compute_layer_complete_3path(
            layer_idx=3,
            inputs_embeds=[prefix_embs, kpt_embs, act_embs],
            attention_mask=mask_4d,
            position_ids=pos_ids,
            qwen3_5=vlm,
            keypoint_expert=kpt,
            action_expert=act,
            prefix_len=P,
            kpt_len=K,
            knowledge_insulation_kpt=False,
        )

        kpt_loss = outputs[1].sum()
        kpt_loss.backward()
        vlm_k_grad = vlm.language_model.layers[3].self_attn.k_proj.weight.grad
        assert vlm_k_grad is not None and vlm_k_grad.abs().sum() > 0

    def test_gradient_blocked_with_ki_kpt(self, three_models):
        """KI_kpt=True => kpt loss should NOT backprop into the VLM's k_proj."""
        vlm, kpt, act, _ = three_models
        vlm.language_model.zero_grad()
        kpt.zero_grad()
        act.zero_grad()

        prefix_embs, kpt_embs, act_embs, mask_4d, pos_ids, P, K, A = self._build_inputs()

        outputs = compute_layer_complete_3path(
            layer_idx=3,
            inputs_embeds=[prefix_embs, kpt_embs, act_embs],
            attention_mask=mask_4d,
            position_ids=pos_ids,
            qwen3_5=vlm,
            keypoint_expert=kpt,
            action_expert=act,
            prefix_len=P,
            kpt_len=K,
            knowledge_insulation_kpt=True,  # block
        )

        kpt_loss = outputs[1].sum()
        kpt_loss.backward()
        vlm_k_grad = vlm.language_model.layers[3].self_attn.k_proj.weight.grad
        assert vlm_k_grad is None or vlm_k_grad.abs().sum() == 0

    def test_kpt_to_action_detach_blocks_action_to_kpt_grad(self, three_models):
        """kpt_to_action_detach=True => action loss should NOT backprop into the kpt expert."""
        vlm, kpt, act, _ = three_models
        vlm.language_model.zero_grad()
        kpt.zero_grad()
        act.zero_grad()

        prefix_embs, kpt_embs, act_embs, mask_4d, pos_ids, P, K, A = self._build_inputs()

        outputs = compute_layer_complete_3path(
            layer_idx=3,
            inputs_embeds=[prefix_embs, kpt_embs, act_embs],
            attention_mask=mask_4d,
            position_ids=pos_ids,
            qwen3_5=vlm,
            keypoint_expert=kpt,
            action_expert=act,
            prefix_len=P,
            kpt_len=K,
            kpt_to_action_detach=True,
        )

        action_loss = outputs[2].sum()
        action_loss.backward()
        kpt_k_grad = kpt.layers[3].self_attn.k_proj.weight.grad
        assert kpt_k_grad is None or kpt_k_grad.abs().sum() == 0

    def test_kpt_to_action_no_detach_allows_grad(self, three_models):
        """kpt_to_action_detach=False (default) => action loss DOES backprop into the kpt expert."""
        vlm, kpt, act, _ = three_models
        vlm.language_model.zero_grad()
        kpt.zero_grad()
        act.zero_grad()

        prefix_embs, kpt_embs, act_embs, mask_4d, pos_ids, P, K, A = self._build_inputs()

        outputs = compute_layer_complete_3path(
            layer_idx=3,
            inputs_embeds=[prefix_embs, kpt_embs, act_embs],
            attention_mask=mask_4d,
            position_ids=pos_ids,
            qwen3_5=vlm,
            keypoint_expert=kpt,
            action_expert=act,
            prefix_len=P,
            kpt_len=K,
            kpt_to_action_detach=False,
        )

        action_loss = outputs[2].sum()
        action_loss.backward()
        kpt_k_grad = kpt.layers[3].self_attn.k_proj.weight.grad
        assert kpt_k_grad is not None and kpt_k_grad.abs().sum() > 0
