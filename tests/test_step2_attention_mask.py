"""Step 2: verify that concatenating three-path att_masks and feeding them through
``make_att_2d_masks`` produces a 2D mask satisfying all four three-path attention rules.

No production code is modified/exercised beyond ``make_att_2d_masks``, which is already fully
generic w.r.t. sequence length/composition -- this is a pure-tensor test of the masking
convention used to build the three-path [PREFIX | KPT_SUFFIX | ACT_SUFFIX] attention pattern.
"""

import torch

from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import make_att_2d_masks


class TestThreePathAttentionMask:
    """Verify [PREFIX(P) | KPT_SUFFIX(17) | ACT_SUFFIX(100)] attention rules."""

    def _three_path_mask(self):
        """Build the three-path mask for tokenize_state=True: P=5 (prefix), K=17 (kpt), A=100 (act)."""
        B, P, K, A = 1, 5, 17, 100
        J = 8  # num_keypoint_joints

        # NOTE: pad_masks must be bool (production code always builds them via `.to(torch.bool)`,
        # see e.g. `embed_prefix`/`embed_kpt_suffix`/`embed_suffix`) -- `make_att_2d_masks` combines
        # them with `&`, which requires bool (not float) tensors.
        prefix_pad = torch.ones(B, P, dtype=torch.bool)
        prefix_att = torch.ones(B, P)

        # KPT suffix: state(1) + hist_kpt(J: [1, 0x(J-1)]) + query_kpt(J: [1, 0x(J-1)])
        kpt_pad = torch.ones(B, K, dtype=torch.bool)
        kpt_att_list = [1] + [1] + [0] * (J - 1) + [1] + [0] * (J - 1)
        kpt_att = torch.tensor([kpt_att_list], dtype=torch.float)

        # ACT suffix: learnable(50: [1,0x49]) + action(50: [1,0x49])
        act_pad = torch.ones(B, A, dtype=torch.bool)
        act_att_list = [1] + [0] * 49 + [1] + [0] * 49
        act_att = torch.tensor([act_att_list], dtype=torch.float)

        pad_masks = torch.cat([prefix_pad, kpt_pad, act_pad], dim=1)
        att_masks = torch.cat([prefix_att, kpt_att, act_att], dim=1)
        mask_2d = make_att_2d_masks(pad_masks, att_masks)  # [B, total, total]

        return mask_2d[0], P, K, A

    def test_vlm_self_attention_causal(self):
        mask, p, k, a = self._three_path_mask()
        vlm_block = mask[:p, :p]
        for q in range(p):
            for kk in range(p):
                if kk <= q:
                    assert vlm_block[q, kk], f"VLM q={q} should see k={kk}"
                else:
                    assert not vlm_block[q, kk], f"VLM q={q} should NOT see k={kk}"

    def test_vlm_cannot_attend_kpt(self):
        mask, p, k, a = self._three_path_mask()
        vlm_to_kpt = mask[:p, p : p + k]
        assert not vlm_to_kpt.any(), "VLM should not attend to KPT"

    def test_vlm_cannot_attend_act(self):
        mask, p, k, a = self._three_path_mask()
        vlm_to_act = mask[:p, p + k :]
        assert not vlm_to_act.any(), "VLM should not attend to ACT"

    def test_kpt_can_attend_vlm(self):
        mask, p, k, a = self._three_path_mask()
        kpt_to_vlm = mask[p : p + k, :p]
        assert kpt_to_vlm.all(), "KPT should attend to all VLM tokens"

    def test_kpt_cannot_attend_act(self):
        mask, p, k, a = self._three_path_mask()
        kpt_to_act = mask[p : p + k, p + k :]
        assert not kpt_to_act.any(), "KPT should not attend to ACT"

    def test_act_can_attend_vlm(self):
        mask, p, k, a = self._three_path_mask()
        act_to_vlm = mask[p + k :, :p]
        assert act_to_vlm.all(), "ACT should attend to all VLM tokens"

    def test_act_can_attend_kpt(self):
        mask, p, k, a = self._three_path_mask()
        act_to_kpt = mask[p + k :, p : p + k]
        assert act_to_kpt.all(), "ACT should attend to all KPT tokens"

    def test_kpt_internal_block_causal(self):
        mask, p, k, a = self._three_path_mask()
        kpt_self = mask[p : p + k, p : p + k]
        # query_kpt (positions 9-16) can see hist_kpt (positions 1-8)
        assert kpt_self[9, 1], "query_kpt should see hist_kpt"
        # hist_kpt (position 1) cannot see query_kpt (position 9)
        assert not kpt_self[1, 9], "hist_kpt should NOT see query_kpt"
        # state (position 0) cannot see hist_kpt (position 1)
        assert not kpt_self[0, 1], "state should NOT see hist_kpt"

    def test_tokenize_state_false(self):
        """tokenize_state=False => act_suffix has 101 tokens; core rules unchanged."""
        B, P, K, A = 1, 3, 17, 101
        J = 8

        prefix_att = torch.ones(B, P)
        kpt_att = torch.tensor(
            [[1] + [1] + [0] * (J - 1) + [1] + [0] * (J - 1)], dtype=torch.float
        )
        # act_suffix 101: state(1:[1]) + learnable(50:[1,0x49]) + action(50:[1,0x49])
        act_att = torch.tensor([[1] + [1] + [0] * 49 + [1] + [0] * 49], dtype=torch.float)

        pad = torch.ones(B, P + K + A, dtype=torch.bool)
        att = torch.cat([prefix_att, kpt_att, act_att], dim=1)
        mask_2d = make_att_2d_masks(pad, att)[0]

        assert not mask_2d[:P, P : P + K].any(), "VLM should not attend to KPT"
        assert mask_2d[P : P + K, :P].all(), "KPT should attend to VLM"
        assert not mask_2d[P : P + K, P + K :].any(), "KPT should not attend to ACT"
        assert mask_2d[P + K :, :P].all(), "ACT should attend to VLM"
        assert mask_2d[P + K :, P : P + K].all(), "ACT should attend to KPT"
