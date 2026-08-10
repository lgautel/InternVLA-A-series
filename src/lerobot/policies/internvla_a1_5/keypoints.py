"""3D keypoint trajectory encoder ported from GeoPredict (models/keypoints.py).

This module is a near-verbatim port of ``GeoPredict/models/keypoints.py`` plus the
``get_1d_sincos_pos_embed`` helper from ``GeoPredict/models/geopredict.py``, adapted for use as
the "keypoint expert" auxiliary path in InternVLA-A1.5 (see
``b/d/itrnVLA15_GeoP_3dtrj_3cn2.md`` §4/§16.1 and ``b/d/itrnVLA15_GeoP_3dtrj_3cn2_rbt2stak3.md``
§2.1/§6.2).

Adaptations relative to the original GeoPredict implementation:
    - ``TrackEncoder.output_dim`` default changed from 2048 (Gemma prefix width) to 1024
      (InternVLA-A1.5 action/keypoint expert hidden size).
    - ``dropout`` defaults to 0.0 (GeoPredict used 0.1); InternVLA-A1.5 fine-tuning is
      typically done with small aloha datasets where the extra regularization is not needed
      and it makes unit tests deterministic-friendlier.
    - Added :func:`load_geopredict_track_encoder_weights` for selective (shape-checked)
      weight loading from a GeoPredict checkpoint, since ``track_fusion_layer`` output_dim
      differs (512->2048 in GeoPredict vs 512->1024 here) and therefore cannot be reused.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from einops import rearrange

logger = logging.getLogger(__name__)


def get_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, base: float = 32) -> torch.Tensor:
    """1D sinusoidal position embedding (ported verbatim from GeoPredict/models/geopredict.py).

    Args:
        embed_dim: output embedding dimension, must be even.
        pos: 1D (or reshape-to-1D) tensor of positions, shape ``[L]``.
        base: frequency base (GeoPredict uses 32 by default, 100 for the 50-step action-horizon
            future position embedding).

    Returns:
        Tensor of shape ``[L, embed_dim]``.
    """
    assert embed_dim % 2 == 0

    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / base**omega  # (D/2,)

    pos = pos.reshape(-1)  # (L,)
    out = torch.einsum("m,d->md", pos, omega)  # (L, D/2), outer product

    emb_sin = torch.sin(out)  # (L, D/2)
    emb_cos = torch.cos(out)  # (L, D/2)
    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (L, D)

    return emb


class PointPatchEmbedding(nn.Module):
    """Temporal patchify of a 3D point track via a strided 1D conv (ported from GeoPredict)."""

    def __init__(self, patch_size: int = 4, in_dim: int = 3, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv1d(in_dim, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, points: torch.Tensor, lengths: torch.Tensor):
        # points: (batch_size, time_len, num_points, in_dim)
        # lengths: (batch_size,)
        batch_size, _, num_points, in_dim = points.shape
        patch_size = self.patch_size

        processed_points = []
        updated_lengths = []
        for i in range(batch_size):
            actual_len = lengths[i].item()
            actual_len = max(actual_len, 1)  # avoid degenerate 0-length sequences
            batch_points = points[i, :actual_len]
            if actual_len % patch_size != 0:
                pad_len = patch_size - (actual_len % patch_size)
                padding = batch_points[-1:].repeat(pad_len, 1, 1)
                batch_points = torch.cat([batch_points, padding], dim=0)

            processed_points.append(batch_points)
            updated_lengths.append(batch_points.size(0))

        max_padded_len = max(len(bp) for bp in processed_points)
        final_points = torch.zeros(
            batch_size, max_padded_len, num_points, in_dim, dtype=points.dtype, device=points.device
        )
        for i, batch_points in enumerate(processed_points):
            final_points[i, : batch_points.size(0)] = batch_points

        points = final_points
        lengths = torch.tensor(updated_lengths, dtype=torch.long, device=points.device)

        points_reshaped = rearrange(points, "b t n c -> (b n) c t")
        patches = self.conv(points_reshaped)
        patches = rearrange(patches, "(b n) c t -> b t n c", b=batch_size, n=num_points)
        patch_lengths = lengths // patch_size

        # patches: (batch_size, num_patches, num_points, embed_dim)
        # patch_lengths: (batch_size,), different for every sample in the batch
        return patches, patch_lengths


class TimeEmbedding(nn.Module):
    """Fixed sinusoidal position embedding lookup table (ported from GeoPredict)."""

    def __init__(self, dim: int, max_seq_len: int = 10000, embedding_type: str = "sinusoidal"):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.embedding_type = embedding_type
        self.register_buffer("pos_embedding", self._create_sinusoidal_embeddings(max_seq_len, dim))

    def _create_sinusoidal_embeddings(self, max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.pos_embedding[positions]


class MultiHeadAttention(nn.Module):
    """Cross-attention with a time-embedding added to the keys (ported from GeoPredict)."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        time_embedding_type: str = "sinusoidal",
        max_seq_len: int = 10000,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads

        assert query_dim % num_heads == 0
        self.q_linear = nn.Linear(query_dim, query_dim)
        self.k_linear = nn.Linear(key_dim, query_dim)
        self.v_linear = nn.Linear(key_dim, query_dim)
        self.out_linear = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        self.key_time_embedding = TimeEmbedding(key_dim, max_seq_len, time_embedding_type)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        bs, q_len, k_len = query.size(0), query.size(1), key.size(1)

        key_pos_emb = self.key_time_embedding(key_positions)  # (k_len, key_dim)
        key_pos_emb = key_pos_emb.unsqueeze(0).expand(bs, -1, -1)  # (bs, k_len, key_dim)
        key = key + key_pos_emb

        query_states = self.q_linear(query).view(bs, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_linear(key).view(bs, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_linear(value).view(bs, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / self.scale
        if mask is not None:  # mask: (bs, 1, k_len)
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, q_len, -1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = (
            torch.matmul(attn_weights, value_states)
            .transpose(1, 2)
            .contiguous()
            .view(bs, q_len, self.num_heads * self.head_dim)
        )

        return self.out_linear(out)


class CrossAttentionBlock(nn.Module):
    """Cross-attention + FFN block used inside :class:`TrackEncoder` (ported from GeoPredict)."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.0,
        time_embedding_type: str = "sinusoidal",
        max_seq_len: int = 10000,
    ):
        super().__init__()
        self.norm_cross = nn.LayerNorm(query_dim)
        self.cross_attn = MultiHeadAttention(
            query_dim, key_dim, num_heads, dropout, time_embedding_type=time_embedding_type,
            max_seq_len=max_seq_len,
        )

        self.norm_ffn = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        inputs: torch.Tensor,
        input_mask: torch.Tensor,
        input_positions: torch.Tensor,
    ) -> torch.Tensor:
        # queries: (batch_size, num_queries, query_dim)
        # inputs: (batch_size, seq_len, key_dim)
        # input_mask: (batch_size, seq_len)
        # input_positions: (seq_len,)
        input_mask = input_mask.unsqueeze(1)
        cross_attn_out = self.cross_attn(
            self.norm_cross(queries), inputs, inputs, mask=input_mask, key_positions=input_positions
        )
        queries = queries + cross_attn_out

        ffn_out = self.ffn(self.norm_ffn(queries))
        queries = queries + ffn_out

        return queries


class TrackEncoder(nn.Module):
    """Encodes a variable-length history of J 3D point tracks into J learned tokens.

    Ported from ``GeoPredict/models/keypoints.py::TrackEncoder``. The only functional change is
    the default ``output_dim`` (2048 -> 1024) to match InternVLA-A1.5's action/keypoint expert
    hidden size instead of GeoPredict's Gemma prefix width.
    """

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 1024,
        patch_size: int = 4,
        embed_dim: int = 256,
        query_dim: int = 512,
        num_queries: int = 1,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.0,
        max_seq_len: int = 1000,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, query_dim))
        nn.init.xavier_uniform_(self.queries)

        self.point_patch_embed = PointPatchEmbedding(patch_size=patch_size, in_dim=input_dim, embed_dim=embed_dim)
        self.cross_attention_block = CrossAttentionBlock(
            query_dim, embed_dim, num_heads, ff_dim, dropout, max_seq_len=max_seq_len // patch_size
        )
        self.linear_transform = nn.Sequential(
            nn.Linear(query_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, query_dim),
        )
        self.final_norm = nn.LayerNorm(query_dim)

        self.track_fusion_layer = nn.Linear(query_dim, output_dim)

    def forward(self, points: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: ``[B, T, J, 3]`` history of 3D keypoint positions.
            lengths: ``[B]`` number of valid (non-padding) history frames per sample.

        Returns:
            ``[B, J * num_queries, output_dim]`` per-joint tokens.
        """
        batch_size = points.size(0)
        patches, patch_lengths = self.point_patch_embed(points, lengths)
        num_patches, num_points = patches.size(1), patches.size(2)
        input_positions = torch.arange(num_patches).to(points.device)

        all_point_outputs = []
        for point_idx in range(num_points):
            point_patches = patches[:, :, point_idx, :]
            point_queries = self.queries.expand(batch_size, -1, -1)
            point_mask = torch.arange(num_patches, device=points.device)[None, :] < patch_lengths[:, None]

            point_queries = self.cross_attention_block(point_queries, point_patches, point_mask, input_positions)
            point_queries = self.linear_transform(point_queries)
            all_point_outputs.append(point_queries)

        output = torch.stack(all_point_outputs, dim=1)  # (bs, num_points, q_len, query_dim)
        output = self.final_norm(output)
        output = output.reshape(batch_size, -1, output.size(-1))  # (bs, num_points * q_len, query_dim)
        output = self.track_fusion_layer(output)  # (bs, num_points * q_len, output_dim)

        return output


# Prefix used by GeoPredict checkpoints for the track encoder submodule.
_GEOPREDICT_TRACK_ENCODER_PREFIX = "keypoint_encoder."

# Sub-prefixes (relative to `keypoint_encoder.`) that are architecture-compatible between
# GeoPredict (query_dim=512, output_dim=2048) and InternVLA-A1.5 (query_dim=512, output_dim=1024).
# `track_fusion_layer` is intentionally excluded because its output_dim differs.
_LOADABLE_SUBMODULE_PREFIXES = (
    "queries",
    "point_patch_embed.",
    "cross_attention_block.",
    "linear_transform.",
    "final_norm.",
)


def load_geopredict_track_encoder_weights(
    track_encoder: TrackEncoder, checkpoint_path: str, strict_shape_check: bool = True
) -> tuple[list[str], list[str]]:
    """Selectively load a :class:`TrackEncoder`'s weights from a GeoPredict checkpoint.

    Only sub-modules that are architecture-identical between GeoPredict (``output_dim=2048``)
    and this port (``output_dim=1024`` by default) are loaded: the point-patch embedding, the
    cross-attention block, ``linear_transform`` and ``final_norm``. ``track_fusion_layer`` is
    always skipped because its output dimension (and therefore weight shape) differs.

    Args:
        track_encoder: the (already constructed) :class:`TrackEncoder` instance to load into.
        checkpoint_path: path to a GeoPredict ``.pth`` checkpoint (a plain ``state_dict``, or a
            dict containing one under a ``"state_dict"``/``"model"`` key).
        strict_shape_check: if True, raise if a key that *should* be loadable (i.e. matches one
            of the loadable prefixes) has a mismatched shape, instead of silently skipping it.

    Returns:
        ``(loaded_keys, skipped_keys)`` — the list of keys copied into ``track_encoder``, and the
        list of keys found in the checkpoint under the ``keypoint_encoder.`` prefix that were
        intentionally skipped (e.g. ``track_fusion_layer``).
    """
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw:
        raw = raw["model"]

    dst_state = track_encoder.state_dict()
    new_state = {}
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []

    for full_key, tensor in raw.items():
        if not full_key.startswith(_GEOPREDICT_TRACK_ENCODER_PREFIX):
            continue
        sub_key = full_key[len(_GEOPREDICT_TRACK_ENCODER_PREFIX):]

        is_loadable_prefix = any(sub_key.startswith(p) for p in _LOADABLE_SUBMODULE_PREFIXES)
        if not is_loadable_prefix:
            skipped_keys.append(full_key)
            continue

        if sub_key not in dst_state:
            logger.warning("GeoPredict key %s has no matching TrackEncoder parameter, skipping.", full_key)
            skipped_keys.append(full_key)
            continue

        if tuple(tensor.shape) != tuple(dst_state[sub_key].shape):
            msg = (
                f"Shape mismatch for {full_key}: checkpoint {tuple(tensor.shape)} vs "
                f"TrackEncoder {tuple(dst_state[sub_key].shape)}"
            )
            if strict_shape_check:
                raise RuntimeError(msg)
            logger.warning("%s — skipping.", msg)
            skipped_keys.append(full_key)
            continue

        new_state[sub_key] = tensor
        loaded_keys.append(full_key)

    missing, unexpected = track_encoder.load_state_dict(new_state, strict=False)
    expected_missing = {"track_fusion_layer.weight", "track_fusion_layer.bias"}
    unexpected_missing = set(missing) - expected_missing
    if unexpected_missing:
        raise RuntimeError(f"Unexpected missing TrackEncoder keys after GeoPredict load: {unexpected_missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys returned by load_state_dict: {unexpected}")

    logger.info(
        "load_geopredict_track_encoder_weights: loaded %d keys, skipped %d keys (e.g. track_fusion_layer).",
        len(loaded_keys),
        len(skipped_keys),
    )
    return loaded_keys, skipped_keys
