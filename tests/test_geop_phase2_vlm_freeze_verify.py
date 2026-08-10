"""Verify GeoP Phase2 VLM/WAN freeze and loss routing (080718 / 080719).

- 080718: VLM frozen but VQA loss enabled (superseded by 080719 plan).
- 080719: VLM + WAN frozen, no VLM/video loss, all other modules trainable.
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from tests.conftest import make_tiny_internvla_a15_config


def _phase2_080718_overrides(**extra):
    """Policy kwargs aligned with internvla_a15_geop_phase2_finetune_stackb3_080718.sh."""
    base = dict(
        train_expert_only=True,
        freeze_vision_encoder=False,
        knowledge_insulation=True,
        knowledge_insulation_kpt=True,
        ki_gradient_scale=0.0,
        ki_kpt_gradient_scale=0.0,
        enable_vqa_loss=True,
        freeze_learnable_tokens=True,
        freeze_keypoint_modules=False,
        action_loss_only=True,  # skip WAN for unit test speed
        enable_keypoint_predictor=True,
        init_kpt_expert_from_action=False,
    )
    base.update(extra)
    return base


def _phase2_080719_overrides(**extra):
    """Policy kwargs aligned with internvla_a15_geop_phase2_finetune_stackb3_080719.sh."""
    base = dict(
        train_expert_only=True,
        freeze_vision_encoder=False,
        knowledge_insulation=True,
        knowledge_insulation_kpt=True,
        enable_vqa_loss=False,
        freeze_learnable_tokens=False,
        freeze_keypoint_modules=False,
        video_loss_weight=0.0,
        action_loss_only=True,  # skip WAN load in unit tests; video_loss_weight=0 tested separately
        enable_keypoint_predictor=True,
        init_kpt_expert_from_action=False,
        action_loss_weight=10.0,
        kpt_loss_weight=0.1,
        kpt_future_loss_weight=0.1,
    )
    base.update(extra)
    return base


def _fake_prefix_batch(model, bsize, device):
    """Minimal valid prefix tensors (mirrors test_step5_forward_loss.py)."""
    vis_cfg = model.qwen3_5_with_expert.qwen3_5.config.vision_config
    patch_dim = vis_cfg.in_channels * vis_cfg.temporal_patch_size * vis_cfg.patch_size**2
    grid_t, grid_h, grid_w = 1, 2, 2
    num_patches = grid_t * grid_h * grid_w

    image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]] * bsize, device=device)
    pixel_values = torch.randn(bsize, num_patches, patch_dim, device=device, dtype=torch.bfloat16)

    merged_patches_per_image = num_patches // (vis_cfg.spatial_merge_size**2)
    image_token_id = model.qwen3_5_with_expert.qwen3_5.config.image_token_id
    seq_len = merged_patches_per_image + 8
    lang_tokens = torch.randint(0, 100, (bsize, seq_len), device=device)
    lang_tokens[:, :merged_patches_per_image] = image_token_id
    lang_masks = torch.ones(bsize, seq_len, dtype=torch.bool, device=device)
    labels = lang_tokens.clone()
    labels[:, : merged_patches_per_image + 2] = -100
    return pixel_values, image_grid_thw, lang_tokens, lang_masks, labels


def _make_model(checkpoint_dir: str, preset=_phase2_080718_overrides, **overrides):
    from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import InternVLAA15

    cfg = make_tiny_internvla_a15_config(checkpoint_dir, **preset(**overrides))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    return InternVLAA15(cfg).to(device), cfg, device


def _run_model_forward(model, cfg, device, *, with_labels: bool = True):
    bsize = 2
    pixel_values, image_grid_thw, lang_tokens, lang_masks, labels = _fake_prefix_batch(
        model, bsize, device
    )
    state = torch.randn(bsize, cfg.max_state_dim, device=device)
    actions = torch.randn(bsize, cfg.chunk_size, cfg.max_action_dim, device=device)
    noise = model.sample_noise((bsize, cfg.chunk_size, cfg.max_action_dim), device)
    time = model.sample_time(bsize, device)

    j = cfg.num_keypoint_joints
    h = cfg.keypoint_history_max_len
    kpt_kwargs = dict(
        his_kpts=torch.randn(bsize, h, j, 3, device=device),
        his_len=torch.full((bsize,), h, device=device, dtype=torch.long),
        kpt_t=torch.randn(bsize, j, 3, device=device),
        kpt_future=torch.randn(bsize, cfg.chunk_size, j, 3, device=device),
        kpt_mask=torch.ones(bsize, dtype=torch.bool, device=device),
    )

    return model.forward(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        state=state,
        actions=actions,
        noise=noise,
        time=time,
        labels=labels if with_labels else None,
        video_frames=actions.new_zeros(bsize, cfg.num_video_frames + 1, 3, cfg.video_height, cfg.video_width),
        **kpt_kwargs,
    )


def _run_forward_backward(model, cfg, device, *, with_labels: bool = True):
    loss_action, loss_vqa, video_loss, _, _, loss_kpt_cur, loss_kpt_fut = _run_model_forward(
        model, cfg, device, with_labels=with_labels
    )
    vqa_term = cfg.lambda_vqa * loss_vqa.mean() if cfg.enable_vqa_loss else 0.0
    total = (
        cfg.action_loss_weight * loss_action.mean()
        + vqa_term
        + cfg.video_loss_weight * video_loss
        + cfg.kpt_loss_weight * (loss_kpt_cur.mean() + cfg.kpt_future_loss_weight * loss_kpt_fut.mean())
    )
    return total, loss_action.mean(), loss_vqa.mean(), video_loss, loss_kpt_cur.mean()


def _vlm_grad_norm(model) -> float:
    vlm = model.qwen3_5_with_expert.qwen3_5
    total = 0.0
    for p in vlm.parameters():
        if p.grad is not None:
            total += p.grad.norm().item()
    return total


class TestGeoPPhase2VLMFreeze:
    def test_080718_vlm_requires_grad_false(self, tiny_qwen35_checkpoint_dir):
        model, _, _ = _make_model(tiny_qwen35_checkpoint_dir)
        vlm = model.qwen3_5_with_expert.qwen3_5
        assert all(p.requires_grad is False for p in vlm.parameters())
        assert vlm.training is False

    def test_080718_experts_trainable(self, tiny_qwen35_checkpoint_dir):
        model, _, _ = _make_model(tiny_qwen35_checkpoint_dir)
        assert any(p.requires_grad for p in model.qwen3_5_with_expert.action_expert.parameters())
        assert any(p.requires_grad for p in model.qwen3_5_with_expert.keypoint_expert.parameters())
        assert any(p.requires_grad for p in model.track_encoder.parameters())
        assert model.learnable_tokens.requires_grad is False

    def test_080718_backward_vlm_no_grad_experts_have_grad(self, tiny_qwen35_checkpoint_dir):
        model, cfg, device = _make_model(tiny_qwen35_checkpoint_dir)
        model.train()
        model.zero_grad(set_to_none=True)

        total, _, loss_vqa, _, _ = _run_forward_backward(model, cfg, device)
        assert total.item() > 0
        assert loss_vqa.item() > 0
        total.backward()

        vlm = model.qwen3_5_with_expert.qwen3_5
        assert _vlm_grad_norm(model) == 0.0

        assert any(p.grad is not None for p in model.qwen3_5_with_expert.action_expert.parameters())
        assert any(p.grad is not None for p in model.qwen3_5_with_expert.keypoint_expert.parameters())

    def test_unfreeze_vlm_enables_vlm_grad_from_vqa_only(self, tiny_qwen35_checkpoint_dir):
        """train_expert_only=false: VLM grad from loss_vlm, blocked from action via KI."""
        model, cfg, device = _make_model(tiny_qwen35_checkpoint_dir, train_expert_only=False)
        vlm = model.qwen3_5_with_expert.qwen3_5
        assert any(p.requires_grad for p in vlm.parameters())

        model.train()

        # Action-only path: no labels so loss_vqa is zero and not in the graph.
        model.zero_grad(set_to_none=True)
        _, loss_action_only, _, _, _ = _run_forward_backward(model, cfg, device, with_labels=False)
        (cfg.action_loss_weight * loss_action_only).backward()
        assert _vlm_grad_norm(model) == 0.0, (
            "knowledge_insulation=true should block action loss -> VLM gradients"
        )

        model.zero_grad(set_to_none=True)
        _, _, loss_vqa, _, _ = _run_forward_backward(model, cfg, device, with_labels=True)
        (cfg.lambda_vqa * loss_vqa).backward()
        assert _vlm_grad_norm(model) > 0.0


class TestGeoPPhase2080719:
    def test_080719_learnable_tokens_trainable(self, tiny_qwen35_checkpoint_dir):
        model, _, _ = _make_model(tiny_qwen35_checkpoint_dir, preset=_phase2_080719_overrides)
        assert model.learnable_tokens.requires_grad is True

    def test_080719_no_vlm_or_video_loss(self, tiny_qwen35_checkpoint_dir):
        model, cfg, device = _make_model(tiny_qwen35_checkpoint_dir, preset=_phase2_080719_overrides)
        model.eval()
        _, loss_action, loss_vqa, video_loss, loss_kpt_cur = _run_forward_backward(
            model, cfg, device, with_labels=False
        )
        assert loss_vqa.item() == 0.0
        assert video_loss.item() == 0.0
        assert loss_action.item() > 0.0
        assert loss_kpt_cur.item() > 0.0

    def test_080719_skips_compute_video_loss_when_weight_zero(self, tiny_qwen35_checkpoint_dir):
        model, cfg, device = _make_model(
            tiny_qwen35_checkpoint_dir,
            preset=_phase2_080719_overrides,
            video_loss_weight=0.0,
        )
        model.eval()
        with patch.object(model, "_compute_video_loss") as mock_video:
            mock_video.side_effect = AssertionError("_compute_video_loss should not be called")
            _, _, video_loss, _, _ = _run_forward_backward(model, cfg, device, with_labels=False)
            assert video_loss.item() == 0.0
            mock_video.assert_not_called()

    def test_080719_backward_experts_have_grad_vlm_frozen(self, tiny_qwen35_checkpoint_dir):
        model, cfg, device = _make_model(tiny_qwen35_checkpoint_dir, preset=_phase2_080719_overrides)
        model.train()
        model.zero_grad(set_to_none=True)
        total, _, loss_vqa, video_loss, _ = _run_forward_backward(model, cfg, device, with_labels=False)
        assert loss_vqa.item() == 0.0
        assert video_loss.item() == 0.0
        total.backward()
        assert _vlm_grad_norm(model) == 0.0
        assert any(p.grad is not None for p in model.qwen3_5_with_expert.action_expert.parameters())
        assert any(p.grad is not None for p in model.qwen3_5_with_expert.keypoint_expert.parameters())
        assert model.learnable_tokens.grad is not None
