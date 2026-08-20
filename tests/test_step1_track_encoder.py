"""Step 1: TrackEncoder standalone tests (ported from GeoPredict, see keypoints.py)."""

import torch


class TestPointPatchEmbedding:
    def test_output_shape(self):
        from lerobot.policies.internvla_a1_5.keypoints import PointPatchEmbedding

        ppe = PointPatchEmbedding(patch_size=4, in_dim=3, embed_dim=256)
        points = torch.randn(2, 20, 8, 3)  # [B, T, num_points, 3]
        lengths = torch.tensor([20, 12])
        patches, patch_lengths = ppe(points, lengths)

        assert patches.shape[0] == 2  # batch
        assert patches.shape[2] == 8  # num_points
        assert patches.shape[3] == 256  # embed_dim
        assert patch_lengths[0] == 5  # 20 / 4
        assert patch_lengths[1] == 3  # ceil(12/4) = 3

    def test_variable_lengths(self):
        from lerobot.policies.internvla_a1_5.keypoints import PointPatchEmbedding

        ppe = PointPatchEmbedding(patch_size=4, in_dim=3, embed_dim=128)
        points = torch.randn(3, 25, 4, 3)
        lengths = torch.tensor([25, 10, 7])
        patches, patch_lengths = ppe(points, lengths)

        assert patches.shape[0] == 3
        assert patches.shape[2] == 4
        assert patch_lengths[0].item() == 7
        assert patch_lengths[1].item() == 3
        assert patch_lengths[2].item() == 2


class TestTrackEncoder:
    def test_output_shape_default(self):
        from lerobot.policies.internvla_a1_5.keypoints import TrackEncoder

        encoder = TrackEncoder(
            input_dim=3, output_dim=1024, num_queries=1,
            patch_size=4, embed_dim=64, query_dim=128,
            num_heads=4, ff_dim=256, max_seq_len=200,
        )
        points = torch.randn(2, 40, 8, 3)  # [B, T, J, 3]
        lengths = torch.tensor([40, 20])
        output = encoder(points, lengths)

        # num_queries=1 => output shape = [B, J * 1, output_dim] = [B, 8, 1024]
        assert output.shape == (2, 8, 1024)

    def test_gradient_flows(self):
        from lerobot.policies.internvla_a1_5.keypoints import TrackEncoder

        encoder = TrackEncoder(
            input_dim=3, output_dim=64, num_queries=1,
            patch_size=4, embed_dim=32, query_dim=64,
            num_heads=2, ff_dim=128, max_seq_len=100,
        )
        points = torch.randn(1, 16, 4, 3, requires_grad=True)
        lengths = torch.tensor([16])
        output = encoder(points, lengths)

        loss = output.sum()
        loss.backward()
        assert points.grad is not None
        assert points.grad.abs().sum() > 0

    def test_single_sample_batch(self):
        from lerobot.policies.internvla_a1_5.keypoints import TrackEncoder

        encoder = TrackEncoder(
            input_dim=3, output_dim=32, num_queries=1,
            patch_size=2, embed_dim=16, query_dim=32,
            num_heads=2, ff_dim=64, max_seq_len=50,
        )
        points = torch.randn(1, 8, 2, 3)
        lengths = torch.tensor([8])
        output = encoder(points, lengths)
        assert output.shape == (1, 2, 32)  # J=2, output_dim=32

    def test_zero_length_history_does_not_crash(self):
        """his_len=0 (Phase 1, no ground-truth keypoints yet) must still produce a valid output."""
        from lerobot.policies.internvla_a1_5.keypoints import TrackEncoder

        encoder = TrackEncoder(
            input_dim=3, output_dim=32, num_queries=1,
            patch_size=2, embed_dim=16, query_dim=32,
            num_heads=2, ff_dim=64, max_seq_len=50,
        )
        points = torch.zeros(2, 20, 4, 3)
        lengths = torch.tensor([0, 0])
        output = encoder(points, lengths)
        assert output.shape == (2, 4, 32)
        assert torch.isfinite(output).all()


class TestSincosPosEmbed:
    def test_shape(self):
        from lerobot.policies.internvla_a1_5.keypoints import get_1d_sincos_pos_embed

        pos = torch.arange(50).float()
        emb = get_1d_sincos_pos_embed(embed_dim=1024, pos=pos)
        assert emb.shape == (50, 1024)

    def test_deterministic(self):
        from lerobot.policies.internvla_a1_5.keypoints import get_1d_sincos_pos_embed

        pos = torch.arange(10).float()
        emb1 = get_1d_sincos_pos_embed(64, pos)
        emb2 = get_1d_sincos_pos_embed(64, pos)
        assert torch.allclose(emb1, emb2)


class TestSelectiveLoading:
    """GeoPredict weight selective-loading tests."""

    def test_load_geopredict_track_encoder_weights_mock(self):
        """Using a mock (in-memory) checkpoint, verify selective loading skips track_fusion_layer."""
        from lerobot.policies.internvla_a1_5.keypoints import TrackEncoder

        # Target encoder: output_dim=32
        encoder = TrackEncoder(
            input_dim=3, output_dim=32, num_queries=1,
            patch_size=2, embed_dim=16, query_dim=32,
            num_heads=2, ff_dim=64, max_seq_len=50,
        )

        # Simulated GeoPredict encoder: output_dim=64 (mismatched)
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

    def test_load_geopredict_track_encoder_weights_from_file(self, tmp_path):
        """End-to-end test of load_geopredict_track_encoder_weights against a real .pth file."""
        from lerobot.policies.internvla_a1_5.keypoints import (
            TrackEncoder,
            load_geopredict_track_encoder_weights,
        )

        geo_encoder = TrackEncoder(
            input_dim=3, output_dim=64, num_queries=1,
            patch_size=2, embed_dim=16, query_dim=32,
            num_heads=2, ff_dim=64, max_seq_len=50,
        )
        # Prefix keys as they'd appear in a real GeoPredict checkpoint state_dict.
        state_dict = {f"keypoint_encoder.{k}": v for k, v in geo_encoder.state_dict().items()}
        state_dict["some_unrelated_module.weight"] = torch.randn(4, 4)

        ckpt_path = tmp_path / "geopredict_mock.pth"
        torch.save(state_dict, ckpt_path)

        encoder = TrackEncoder(
            input_dim=3, output_dim=32, num_queries=1,
            patch_size=2, embed_dim=16, query_dim=32,
            num_heads=2, ff_dim=64, max_seq_len=50,
        )
        loaded_keys, skipped_keys = load_geopredict_track_encoder_weights(encoder, str(ckpt_path))

        assert len(loaded_keys) > 0
        assert any("track_fusion_layer" in k for k in skipped_keys)
        assert all(k.startswith("keypoint_encoder.") for k in loaded_keys + skipped_keys)

        # Loaded sub-modules should now match geo_encoder's weights exactly.
        assert torch.allclose(encoder.queries, geo_encoder.queries)
        assert torch.allclose(
            encoder.point_patch_embed.conv.weight, geo_encoder.point_patch_embed.conv.weight
        )
