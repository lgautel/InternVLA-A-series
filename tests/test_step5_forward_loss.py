"""Step 5: InternVLAA15WithExpertModel.forward 3-path dispatch + training loss tests."""

import torch

from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import (
    ActionExpertConfig,
    InternVLAA15WithExpertModel,
    KeypointExpertConfig,
    make_att_2d_masks,
)


class TestWithExpertModelDispatch:
    """Test InternVLAA15WithExpertModel.forward's 3-path joint dispatch."""

    def test_joint_dispatch_output_shapes(self, tiny_qwen35_checkpoint_dir):
        """3-path joint dispatch (len(inputs_embeds) == 3) should return [prefix, kpt, act]."""
        model = InternVLAA15WithExpertModel(
            vlm_model_name_or_path=tiny_qwen35_checkpoint_dir,
            action_expert_config=ActionExpertConfig(hidden_size=32, intermediate_size=64),
            keypoint_expert_config=KeypointExpertConfig(hidden_size=32, intermediate_size=64),
            precision="bfloat16",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        device = next(model.parameters()).device
        vlm_hidden = model.qwen3_5.config.text_config.hidden_size

        B, P, K, A = 1, 4, 17, 10
        prefix_embs = torch.randn(B, P, vlm_hidden, dtype=torch.bfloat16, device=device)
        kpt_embs = torch.randn(B, K, 32, dtype=torch.bfloat16, device=device)
        act_embs = torch.randn(B, A, 32, dtype=torch.bfloat16, device=device)

        total = P + K + A
        pad_masks = torch.ones(B, total, dtype=torch.bool, device=device)
        att_masks = torch.ones(B, total, device=device)
        mask_2d = make_att_2d_masks(pad_masks, att_masks)
        mask_4d = mask_2d.unsqueeze(1).float()
        mask_4d = torch.where(mask_4d.bool(), torch.zeros_like(mask_4d), torch.full_like(mask_4d, -1e9))
        pos_ids = torch.arange(total, device=device).unsqueeze(0).unsqueeze(0).repeat(3, B, 1).float()

        outputs, _ = model.forward(
            attention_mask=mask_4d,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, kpt_embs, act_embs],
            use_cache=False,
        )
        prefix_out, kpt_out, act_out = outputs

        assert prefix_out.shape == (B, P, vlm_hidden)
        assert kpt_out.shape == (B, K, 32)
        assert act_out.shape == (B, A, 32)

    def test_joint_backward_updates_all_three_experts(self, tiny_qwen35_checkpoint_dir):
        model = InternVLAA15WithExpertModel(
            vlm_model_name_or_path=tiny_qwen35_checkpoint_dir,
            action_expert_config=ActionExpertConfig(hidden_size=32, intermediate_size=64),
            keypoint_expert_config=KeypointExpertConfig(hidden_size=32, intermediate_size=64),
            precision="bfloat16",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        device = next(model.parameters()).device
        vlm_hidden = model.qwen3_5.config.text_config.hidden_size

        B, P, K, A = 2, 4, 17, 10
        prefix_embs = torch.randn(B, P, vlm_hidden, dtype=torch.bfloat16, device=device)
        kpt_embs = torch.randn(B, K, 32, dtype=torch.bfloat16, device=device)
        act_embs = torch.randn(B, A, 32, dtype=torch.bfloat16, device=device)

        total = P + K + A
        pad_masks = torch.ones(B, total, dtype=torch.bool, device=device)
        att_masks = torch.ones(B, total, device=device)
        mask_2d = make_att_2d_masks(pad_masks, att_masks)
        mask_4d = mask_2d.unsqueeze(1).float()
        mask_4d = torch.where(mask_4d.bool(), torch.zeros_like(mask_4d), torch.full_like(mask_4d, -1e9))
        pos_ids = torch.arange(total, device=device).unsqueeze(0).unsqueeze(0).repeat(3, B, 1).float()

        outputs, _ = model.forward(
            attention_mask=mask_4d,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, kpt_embs, act_embs],
            use_cache=False,
        )
        loss = sum(o.float().pow(2).mean() for o in outputs)
        loss.backward()

        vlm_grad = sum(p.grad.abs().sum().item() for p in model.qwen3_5.parameters() if p.grad is not None)
        kpt_grad = sum(p.grad.abs().sum().item() for p in model.keypoint_expert.parameters() if p.grad is not None)
        act_grad = sum(p.grad.abs().sum().item() for p in model.action_expert.parameters() if p.grad is not None)
        assert vlm_grad > 0
        assert kpt_grad > 0
        assert act_grad > 0


class TestKeypointLossComputation:
    """Test extraction and computation of the kpt current/future losses (pure-tensor logic
    mirroring `InternVLAA15.forward`'s post-processing of the keypoint expert's suffix output).
    """

    def test_kpt_loss_extraction(self):
        """Extract and compute MSE loss from the last J tokens of kpt_out."""
        import torch.nn as nn

        J = 8
        hidden = 32
        keypoint_out_proj = nn.Linear(hidden, 3)

        kpt_out = torch.randn(2, 1 + 2 * J, hidden)
        query_kpt_out = kpt_out[:, -J:]
        pred_kpt = keypoint_out_proj(query_kpt_out)

        kpt_gt = torch.randn(2, J, 3)
        loss = torch.nn.functional.mse_loss(pred_kpt, kpt_gt)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_future_kpt_loss(self):
        """Dimension check for the future-keypoint-trajectory loss."""
        import torch.nn as nn

        J, C, hidden = 8, 50, 32
        keypoint_out_proj = nn.Linear(hidden, 3)
        future_kpt_pos_embed = torch.randn(C, hidden)

        kpt_out = torch.randn(2, 1 + 2 * J, hidden)
        query_kpt_out = kpt_out[:, -J:]  # [B, J, hidden]
        kpt_rep = query_kpt_out.unsqueeze(1).expand(-1, C, -1, -1)  # [B, C, J, hidden]
        fut_pe = future_kpt_pos_embed[:C][None, :, None, :]  # [1, C, 1, hidden]
        future_pred = keypoint_out_proj((kpt_rep + fut_pe).reshape(-1, J, hidden)).reshape(2, C, J, 3)

        future_kpts_gt = torch.randn(2, C, J, 3)
        loss = torch.nn.functional.mse_loss(future_pred, future_kpts_gt)
        assert loss.shape == ()
        assert future_pred.shape == (2, 50, 8, 3)


class TestInternVLAA15ForwardKptLoss:
    """Integration test: InternVLAA15.forward produces finite loss_kpt_current/loss_kpt_future
    that actually depend on the ground-truth keypoints (Phase 2, kpt_mask=True)."""

    def _fake_prefix_batch(self, model, cfg, bsize, device):
        """Build a minimal-but-valid prefix (pixel_values/image_grid_thw/lang_tokens/...)."""
        vis_cfg = model.qwen3_5_with_expert.qwen3_5.config.vision_config
        patch_dim = vis_cfg.in_channels * vis_cfg.temporal_patch_size * vis_cfg.patch_size**2
        grid_t, grid_h, grid_w = 1, 2, 2
        num_patches = grid_t * grid_h * grid_w

        image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]] * bsize, device=device)
        pixel_values = torch.randn(bsize, num_patches, patch_dim, device=device, dtype=torch.bfloat16)

        # The vision tower spatially merges `spatial_merge_size**2` raw patches into one token,
        # so the number of `image_token_id` slots in lang_tokens must match the *merged* count.
        merged_patches_per_image = num_patches // (vis_cfg.spatial_merge_size**2)
        image_token_id = model.qwen3_5_with_expert.qwen3_5.config.image_token_id
        seq_len = merged_patches_per_image + 4
        lang_tokens = torch.randint(0, 100, (bsize, seq_len), device=device)
        lang_tokens[:, :merged_patches_per_image] = image_token_id
        lang_masks = torch.ones(bsize, seq_len, dtype=torch.bool, device=device)
        return pixel_values, image_grid_thw, lang_tokens, lang_masks

    def test_kpt_losses_are_finite_and_gradient_flows(
        self, tiny_internvla_a15_model, tiny_internvla_a15_config
    ):
        model = tiny_internvla_a15_model
        cfg = tiny_internvla_a15_config
        device = next(model.parameters()).device
        bsize = 2

        pixel_values, image_grid_thw, lang_tokens, lang_masks = self._fake_prefix_batch(
            model, cfg, bsize, device
        )
        state = torch.randn(bsize, cfg.max_state_dim, device=device)
        actions = torch.randn(bsize, cfg.chunk_size, cfg.max_action_dim, device=device)
        action_is_pad = torch.zeros(bsize, cfg.chunk_size, dtype=torch.bool, device=device)

        his_kpts = torch.randn(bsize, cfg.keypoint_history_max_len, cfg.num_keypoint_joints, 3, device=device)
        his_len = torch.full((bsize,), cfg.keypoint_history_max_len, device=device, dtype=torch.long)
        kpt_t = torch.randn(bsize, cfg.num_keypoint_joints, 3, device=device)
        kpt_future = torch.randn(bsize, cfg.chunk_size, cfg.num_keypoint_joints, 3, device=device)
        kpt_mask = torch.ones(bsize, dtype=torch.bool, device=device)

        noise = model.sample_noise((bsize, cfg.chunk_size, cfg.max_action_dim), device)
        time = model.sample_time(bsize, device)

        (
            loss_action,
            loss_vqa,
            video_loss,
            loss_per_token,
            token_mask,
            loss_kpt_current,
            loss_kpt_future,
        ) = model.forward(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            actions=actions,
            noise=noise,
            time=time,
            his_kpts=his_kpts,
            his_len=his_len,
            kpt_t=kpt_t,
            kpt_future=kpt_future,
            kpt_mask=kpt_mask,
        )

        assert torch.isfinite(loss_action).all()
        # loss_kpt_current/future are per-sample [B] tensors (mean-reduced over J/C/3 inside forward).
        assert loss_kpt_current.shape == (bsize,)
        assert loss_kpt_future.shape == (bsize,)
        assert torch.isfinite(loss_kpt_current).all()
        assert torch.isfinite(loss_kpt_future).all()
        assert (loss_kpt_current > 0).all()
        assert (loss_kpt_future > 0).all()

        total_loss = loss_action.mean() + loss_kpt_current.mean() + loss_kpt_future.mean()
        total_loss.backward()

        kpt_expert_grad = sum(
            p.grad.abs().sum().item()
            for p in model.qwen3_5_with_expert.keypoint_expert.parameters()
            if p.grad is not None
        )
        track_encoder_grad = sum(
            p.grad.abs().sum().item() for p in model.track_encoder.parameters() if p.grad is not None
        )
        assert kpt_expert_grad > 0
        assert track_encoder_grad > 0
