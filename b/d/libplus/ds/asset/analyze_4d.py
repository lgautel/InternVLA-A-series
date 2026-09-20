#!/usr/bin/env python3
"""4D (position + orientation) keypoint analysis of /B/Dta/opvla_libero.

Runs MuJoCo FK over every frame and answers the questions the report needs:

  * which arena base each episode was actually recorded in (recovered from the
    data, not read off an XML)
  * what R_pad the training pipeline would compute from this data, and how it
    compares to the 1.8212722539901733 hardcoded at eval time
  * how much of the [-1, 1] box each normalization scheme actually uses
  * which of the 8 keypoint bodies carry independent information
  * whether the stored quaternion convention is continuous along a trajectory
  * how much of the H=200 history window is padding

Requires mujoco, so run it with the eval venv:
    /B/VENV/libero_plus_client/bin/python analyze_4d.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from panda_fk import (
    ARENA_BASE_XPOS,
    DEFAULT_R_PAD,
    LIFT_BASE_XPOS,
    PandaFK,
    arena_of,
    wxyz_to_xyzw_hemisphere,
)
from rlds_reader import SUBSETS, iter_episodes

HERE = Path(__file__).parent
OUT_JSON = HERE / "kpt_stats.json"
OUT_NPZ = HERE / "kpt_arrays.npz"

BODY_NAMES = [f"link{i}" for i in range(1, 8)] + ["eef"]
BBOX_MARGIN = 0.15
HISTORY_H = 200
CHUNK_C = 50


def compute_r_pad(lo: np.ndarray, hi: np.ndarray, margin: float = BBOX_MARGIN) -> float:
    """util_scripts/generate_libero_keypoints.py:118 -- isotropic, world origin."""
    return float(np.maximum(np.abs(lo), np.abs(hi)).max() * (1.0 + margin))


def utilisation(lo: np.ndarray, hi: np.ndarray, scale: float | np.ndarray) -> list[float]:
    """Fraction of the [-1, 1] interval each axis occupies after scaling."""
    return [round(float(v), 4) for v in (hi - lo) / scale / 2.0]


def main() -> int:
    fk = PandaFK()

    # world-frame extents, per normalization scheme
    lift_lo = np.full(3, np.inf)
    lift_hi = np.full(3, -np.inf)
    base_lo = np.full(3, np.inf)  # base-relative (arena-invariant by construction)
    base_hi = np.full(3, -np.inf)

    arena_counter: Counter[str] = Counter()
    arena_frames: Counter[str] = Counter()
    measured_offsets: dict[str, list[np.ndarray]] = defaultdict(list)
    offset_residual_max = 0.0

    body_pos_sum = np.zeros((8, 3))
    body_pos_sqsum = np.zeros((8, 3))
    body_pos_lo = np.full((8, 3), np.inf)
    body_pos_hi = np.full((8, 3), -np.inf)
    pair_max_gap = np.zeros((8, 8))

    quat_flips = np.zeros(8, dtype=np.int64)
    quat_step_max = np.zeros(8)
    quat_near_hemisphere_edge = np.zeros(8, dtype=np.int64)
    quat_lo = np.full((8, 4), np.inf)
    quat_hi = np.full((8, 4), -np.inf)

    hist_slots_total = 0
    hist_slots_filled = 0
    per_suite_fill: dict[str, list[int]] = defaultdict(list)
    episodes_reaching_H = 0
    future_clamped_frames = 0

    total_frames = 0
    total_episodes = 0
    kpt_sample: list[np.ndarray] = []

    for subset in SUBSETS:
        for ep in iter_episodes(subset):
            total_episodes += 1
            T = ep.num_steps
            total_frames += T
            arena = arena_of(subset, ep.task_name)
            arena_counter[arena] += 1
            arena_frames[arena] += T

            qpos = ep.qpos9()
            state = ep.state.astype(np.float64)

            # FK once with the base at the origin; both frames are shifts of it.
            poses = fk.batch_world_poses(qpos, base_xpos=np.zeros(3))
            pos_local = poses[:, :, :3]
            quats = poses[:, :, 3:]

            # --- arena base recovered from the data
            offsets = state[:, :3] - pos_local[:, 7, :]
            measured_offsets[arena].append(offsets.mean(axis=0))
            offset_residual_max = max(offset_residual_max, float(offsets.std(axis=0).max()))

            # --- extents in both candidate frames
            pos_lift = pos_local + LIFT_BASE_XPOS
            lift_lo = np.minimum(lift_lo, pos_lift.reshape(-1, 3).min(axis=0))
            lift_hi = np.maximum(lift_hi, pos_lift.reshape(-1, 3).max(axis=0))
            base_lo = np.minimum(base_lo, pos_local.reshape(-1, 3).min(axis=0))
            base_hi = np.maximum(base_hi, pos_local.reshape(-1, 3).max(axis=0))

            # --- per-body statistics, in the Lift frame that training used
            body_pos_sum += pos_lift.sum(axis=0)
            body_pos_sqsum += (pos_lift**2).sum(axis=0)
            body_pos_lo = np.minimum(body_pos_lo, pos_lift.min(axis=0))
            body_pos_hi = np.maximum(body_pos_hi, pos_lift.max(axis=0))
            for i in range(8):
                gap = np.linalg.norm(pos_lift[:, i : i + 1, :] - pos_lift, axis=2).max(axis=0)
                pair_max_gap[i] = np.maximum(pair_max_gap[i], gap)

            # --- quaternion convention as stored
            stored = np.empty((T, 8, 4))
            for t in range(T):
                for i in range(8):
                    stored[t, i] = wxyz_to_xyzw_hemisphere(quats[t, i])
            quat_near_hemisphere_edge += (np.abs(stored[:, :, 3]) < 0.05).sum(axis=0)
            quat_lo = np.minimum(quat_lo, stored.min(axis=0))
            quat_hi = np.maximum(quat_hi, stored.max(axis=0))
            if T > 1:
                step = np.linalg.norm(np.diff(stored, axis=0), axis=2)
                quat_step_max = np.maximum(quat_step_max, step.max(axis=0))
                quat_flips += (step > 1.0).sum(axis=0)

            # --- history window occupancy for Extract3DKeypointTransformFn
            fill = np.minimum(np.arange(T), HISTORY_H)
            hist_slots_total += T * HISTORY_H
            hist_slots_filled += int(fill.sum())
            per_suite_fill[subset].append(int(fill.max()))
            if T > HISTORY_H:
                episodes_reaching_H += 1
            future_clamped_frames += int(min(T, CHUNK_C))

            if total_episodes % 53 == 0:
                kpt_sample.append((pos_lift / DEFAULT_R_PAD)[::5].astype(np.float32))

        print(f"  FK done for {subset:15s} episodes: {total_episodes} frames: {total_frames}")

    # ---------------- normalization schemes ----------------
    r_pad_measured = compute_r_pad(lift_lo, lift_hi)
    r_pad_base = compute_r_pad(base_lo, base_hi)
    base_halfspan = (base_hi - base_lo) / 2.0

    schemes = {
        "A_lift_world_isotropic": {
            "description": "current pipeline: Lift world frame, pos / R_pad",
            "r_pad": r_pad_measured,
            "range_min": (lift_lo / r_pad_measured).round(4).tolist(),
            "range_max": (lift_hi / r_pad_measured).round(4).tolist(),
            "axis_utilisation": utilisation(lift_lo, lift_hi, r_pad_measured),
        },
        "B_base_frame_isotropic": {
            "description": "subtract the robot base first, then one isotropic radius",
            "r_pad": r_pad_base,
            "range_min": (base_lo / r_pad_base).round(4).tolist(),
            "range_max": (base_hi / r_pad_base).round(4).tolist(),
            "axis_utilisation": utilisation(base_lo, base_hi, r_pad_base),
        },
        "C_base_frame_per_axis": {
            "description": "subtract the base, then scale each axis by its own half-span",
            "scale": base_halfspan.round(6).tolist(),
            "centre": ((base_hi + base_lo) / 2.0).round(6).tolist(),
            "axis_utilisation": [1.0, 1.0, 1.0],
        },
    }

    body_mean = body_pos_sum / total_frames
    body_std = np.sqrt(np.maximum(body_pos_sqsum / total_frames - body_mean**2, 0.0))
    static_bodies = [BODY_NAMES[i] for i in range(8) if body_std[i].max() < 1e-6]
    duplicate_pairs = [
        [BODY_NAMES[i], BODY_NAMES[j], round(float(pair_max_gap[i, j]), 9)]
        for i in range(8)
        for j in range(i + 1, 8)
        if pair_max_gap[i, j] < 1e-6
    ]
    moving_position_dims = int(sum(1 for i in range(8) for a in range(3) if body_std[i, a] >= 1e-6))

    # Collapse coincident bodies, then count the axes that still move. This is
    # the number of position channels that actually carry signal in pos_only.
    representative: list[int] = []
    for i in range(8):
        if not any(pair_max_gap[min(i, j), max(i, j)] < 1e-6 for j in representative):
            representative.append(i)
    independent_position_dims = int(
        sum(1 for i in representative for a in range(3) if body_std[i, a] >= 1e-6)
    )

    offsets_out = {}
    for arena, samples in measured_offsets.items():
        block = np.stack(samples)
        offsets_out[arena] = {
            "episodes": arena_counter[arena],
            "frames": arena_frames[arena],
            "frame_share_pct": round(100 * arena_frames[arena] / total_frames, 2),
            "measured_base_xpos": block.mean(axis=0).round(4).tolist(),
            "spread_across_episodes_m": float(block.std(axis=0).max()),
            "reference_from_libero_source": ARENA_BASE_XPOS[arena].tolist(),
            "abs_diff_to_reference_m": float(
                np.abs(block.mean(axis=0) - ARENA_BASE_XPOS[arena]).max()
            ),
        }

    fill_summary = {
        suite: {
            "episodes": len(v),
            "max_history_reached": int(max(v)),
            "episodes_that_can_fill_H": int(sum(1 for x in v if x >= HISTORY_H)),
        }
        for suite, v in per_suite_fill.items()
    }

    report = {
        "frames": total_frames,
        "episodes": total_episodes,
        "fk": {
            "source_xml": str(fk.model.nbody) + " bodies from robosuite robots/panda/robot.xml",
            "keypoint_bodies": BODY_NAMES,
            "qpos_layout": "joint_state[0:7] + state[6:8]",
            "eef_body_is_state_position": True,
            "base_offset_residual_within_episode_m": round(offset_residual_max, 6),
        },
        "arena_bases": offsets_out,
        "r_pad": {
            "recomputed_from_this_dataset": r_pad_measured,
            "hardcoded_at_eval": DEFAULT_R_PAD,
            "abs_diff": abs(r_pad_measured - DEFAULT_R_PAD),
            "relative_diff": abs(r_pad_measured - DEFAULT_R_PAD) / DEFAULT_R_PAD,
            "driver_axis": ["x", "y", "z"][int(np.argmax(np.maximum(np.abs(lift_lo), np.abs(lift_hi))))],
            "global_min_world": lift_lo.round(6).tolist(),
            "global_max_world": lift_hi.round(6).tolist(),
            "margin": BBOX_MARGIN,
        },
        "normalisation_schemes": schemes,
        "body_statistics": {
            name: {
                "mean_xyz": body_mean[i].round(6).tolist(),
                "std_xyz": body_std[i].round(6).tolist(),
                "range_xyz": (body_pos_hi[i] - body_pos_lo[i]).round(6).tolist(),
            }
            for i, name in enumerate(BODY_NAMES)
        },
        "redundancy": {
            "static_bodies": static_bodies,
            "coincident_body_pairs": duplicate_pairs,
            "moving_position_dims_of_24": moving_position_dims,
            "independent_position_dims_of_24": independent_position_dims,
            "independent_bodies": [BODY_NAMES[i] for i in representative],
            "note": (
                "moving = axes with non-zero variance; independent additionally collapses "
                "bodies whose world position coincides at every frame"
            ),
        },
        "quaternion": {
            "convention": "xyzw, hemisphere qw >= 0 (matches keypoints_meta.json)",
            "frames_with_qw_below_0.05": {
                BODY_NAMES[i]: int(quat_near_hemisphere_edge[i]) for i in range(8)
            },
            "sign_flips_between_consecutive_frames": {
                BODY_NAMES[i]: int(quat_flips[i]) for i in range(8)
            },
            "max_l2_step_between_consecutive_frames": {
                BODY_NAMES[i]: round(float(quat_step_max[i]), 4) for i in range(8)
            },
            "total_sign_flips": int(quat_flips.sum()),
            "component_range_xyzw": {
                BODY_NAMES[i]: (quat_hi[i] - quat_lo[i]).round(4).tolist() for i in range(8)
            },
            "static_position_bodies_still_rotate": {
                name: bool((quat_hi[i] - quat_lo[i]).max() > 1e-3)
                for i, name in enumerate(BODY_NAMES)
                if name in static_bodies
            },
        },
        "history_window": {
            "H": HISTORY_H,
            "chunk_C": CHUNK_C,
            "padding_fraction": round(1 - hist_slots_filled / hist_slots_total, 4),
            "episodes_longer_than_H": episodes_reaching_H,
            "episodes_longer_than_H_pct": round(100 * episodes_reaching_H / total_episodes, 2),
            "per_suite": fill_summary,
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2))
    sample = np.concatenate(kpt_sample) if kpt_sample else np.zeros((0, 8, 3), dtype=np.float32)
    np.savez_compressed(
        OUT_NPZ,
        kpt_sample_normalised=sample,
        body_std=body_std,
        body_lo=body_pos_lo,
        body_hi=body_pos_hi,
        lift_lo=lift_lo,
        lift_hi=lift_hi,
        base_lo=base_lo,
        base_hi=base_hi,
        arena_bases=np.stack([np.asarray(v["measured_base_xpos"]) for v in offsets_out.values()]),
        arena_frames=np.asarray([v["frames"] for v in offsets_out.values()]),
    )
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_NPZ}")

    print(f"\nR_pad recomputed = {r_pad_measured!r}  (eval hardcodes {DEFAULT_R_PAD!r})")
    print(f"  abs diff = {report['r_pad']['abs_diff']:.3e}")
    print(f"arena bases: { {k: v['measured_base_xpos'] for k, v in offsets_out.items()} }")
    print(f"scheme A utilisation = {schemes['A_lift_world_isotropic']['axis_utilisation']}")
    print(f"scheme B utilisation = {schemes['B_base_frame_isotropic']['axis_utilisation']}")
    print(f"static bodies = {static_bodies}, coincident pairs = {duplicate_pairs}")
    print(f"moving position dims = {moving_position_dims} / 24")
    print(f"quaternion sign flips = {int(quat_flips.sum())}")
    print(f"history padding fraction = {report['history_window']['padding_fraction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
