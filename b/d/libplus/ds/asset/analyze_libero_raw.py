#!/usr/bin/env python3
"""Full-dataset statistics for /B/Dta/opvla_libero (OpenVLA modified_libero_rlds).

One pass over all 96 shards / 1693 episodes / 273465 frames. Produces
`raw_stats.json` (numbers quoted in the report) and `raw_arrays.npz`
(per-frame arrays the plotting script reuses).

Runs on the base python3: no TensorFlow, no mujoco, no robosuite.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from rlds_reader import SUBSETS, iter_episodes

HERE = Path(__file__).parent
OUT_JSON = HERE / "raw_stats.json"
OUT_NPZ = HERE / "raw_arrays.npz"

# Official LIBERO demo counts before OpenVLA dropped failures and no-ops.
OFFICIAL_DEMOS_PER_SUITE = 500

# OpenVLA's filter, from experiments/robot/libero/regenerate_libero_dataset.py
NOOP_EPS = 1e-4

# Franka Panda joint limits, robosuite robots/panda/robot.xml.
PANDA_JOINT_LIMITS = np.array(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ]
)


class Welford:
    """Streaming mean/std/min/max over a fixed-width vector."""

    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim)
        self.m2 = np.zeros(dim)
        self.lo = np.full(dim, np.inf)
        self.hi = np.full(dim, -np.inf)

    def update(self, block: np.ndarray) -> None:
        block = np.asarray(block, dtype=np.float64)
        self.lo = np.minimum(self.lo, block.min(axis=0))
        self.hi = np.maximum(self.hi, block.max(axis=0))
        for row in block:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (row - self.mean)

    def as_dict(self) -> dict:
        std = np.sqrt(self.m2 / self.n) if self.n > 1 else np.zeros_like(self.mean)
        return {
            "count": int(self.n),
            "min": self.lo.round(6).tolist(),
            "max": self.hi.round(6).tolist(),
            "mean": self.mean.round(6).tolist(),
            "std": std.round(6).tolist(),
        }


def is_noop(action: np.ndarray, prev_action: np.ndarray | None) -> bool:
    """OpenVLA's no-op test: no EEF motion and no change of gripper command."""
    if np.linalg.norm(action[:6]) > NOOP_EPS:
        return False
    if prev_action is None:
        return True
    return bool(action[6] == prev_action[6])


def quantisation(values: np.ndarray, candidates=(350, 700, 1120, 1400, 2800, 5600)) -> dict:
    """Smallest 1/N grid that all values sit on.

    LIBERO demos were teleoperated with a SpaceMouse, whose raw axes are
    mapped by `scale_to_control(x, axis_scale=350.0)`
    (robosuite/devices/spacemouse.py:67), so the actions are discrete.
    """
    nonzero = np.abs(values[np.abs(values) > 1e-9])
    if nonzero.size == 0:
        return {"grid": None}
    for denom in candidates:
        residual = np.abs(nonzero * denom - np.round(nonzero * denom))
        if residual.max() < 1e-3:
            return {
                "grid": denom,
                "step": 1.0 / denom,
                "max_residual": float(residual.max()),
                "levels_used": int(round(nonzero.max() * denom)),
            }
    return {"grid": None, "max_abs": float(nonzero.max())}


def main() -> int:
    action_stats = Welford(7)
    state_stats = Welford(8)
    joint_stats = Welford(7)

    per_task: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"episodes": 0, "frames": 0, "lengths": [], "instructions": set()}
    )
    per_subset: dict[str, dict] = defaultdict(lambda: {"episodes": 0, "frames": 0, "lengths": []})

    gripper_values = Counter()
    action_abs_max = 0.0
    action_quantum_pool: list[np.ndarray] = []
    clipped_frames = 0
    noop_frames = 0
    tiny_motion_frames = 0
    nan_frames = 0

    flag_violations: list[str] = []
    reward_violations: list[str] = []
    discount_violations: list[str] = []

    jpeg_bytes = {"image": [], "wrist_image": []}
    axisangle_norms: list[np.ndarray] = []
    eef_positions: list[np.ndarray] = []
    joint_limit_margin = np.full((7, 2), np.inf)
    over_pi_frames = 0
    episodes_with_aa_jump = 0
    aa_jump_max = 0.0

    total_episodes = 0
    for subset in SUBSETS:
        for ep in iter_episodes(subset):
            total_episodes += 1
            T = ep.num_steps
            action = ep.action.astype(np.float64)
            state = ep.state.astype(np.float64)
            joints = ep.joint_state.astype(np.float64)

            if not (len(action) == len(state) == len(joints) == T):
                flag_violations.append(f"{subset}/{ep.task_name}: ragged T ({T})")

            key = (subset, ep.task_name)
            rec = per_task[key]
            rec["episodes"] += 1
            rec["frames"] += T
            rec["lengths"].append(T)
            rec["instructions"].update(set(ep.language_instructions))

            sub = per_subset[subset]
            sub["episodes"] += 1
            sub["frames"] += T
            sub["lengths"].append(T)

            action_stats.update(action)
            state_stats.update(state)
            joint_stats.update(joints)

            nan_frames += int(
                np.isnan(action).any(axis=1).sum()
                + np.isnan(state).any(axis=1).sum()
                + np.isnan(joints).any(axis=1).sum()
            )

            gripper_values.update(np.round(action[:, 6], 6).tolist())
            action_abs_max = max(action_abs_max, float(np.abs(action[:, :6]).max()))
            clipped_frames += int((np.abs(action[:, :6]) >= 0.9375 - 1e-9).any(axis=1).sum())

            prev = None
            for row in action:
                if is_noop(row, prev):
                    noop_frames += 1
                prev = row

            if T > 1:
                disp = np.linalg.norm(np.diff(state[:, :3], axis=0), axis=1)
                tiny_motion_frames += int((disp < 1e-3).sum())

            first, last, term = ep.flags("is_first"), ep.flags("is_last"), ep.flags("is_terminal")
            expected_first = np.zeros(T, dtype=bool)
            expected_first[0] = True
            expected_last = np.zeros(T, dtype=bool)
            expected_last[-1] = True
            if not (
                np.array_equal(first, expected_first)
                and np.array_equal(last, expected_last)
                and np.array_equal(term, expected_last)
            ):
                flag_violations.append(f"{subset}/{ep.task_name}#{ep.index}")

            reward = ep.reward
            expected_reward = np.zeros(T, dtype=np.float32)
            expected_reward[-1] = 1.0
            if not np.allclose(reward, expected_reward):
                reward_violations.append(f"{subset}/{ep.task_name}#{ep.index}")
            if not np.allclose(ep.discount, 1.0):
                discount_violations.append(f"{subset}/{ep.task_name}#{ep.index}")

            norms = np.linalg.norm(state[:, 3:6], axis=1)
            axisangle_norms.append(norms)
            over_pi_frames += int((norms > np.pi).sum())
            if T > 1:
                # robosuite's quat2axisangle returns 2*acos(qw) without folding
                # qw<0 back into the qw>=0 hemisphere, so the encoding is
                # double-valued and jumps by ~2*pi when qw changes sign.
                jump = np.abs(np.diff(norms))
                aa_jump_max = max(aa_jump_max, float(jump.max()))
                if (jump > 1.0).any():
                    episodes_with_aa_jump += 1
            eef_positions.append(state[::10, :3])
            joint_limit_margin[:, 0] = np.minimum(
                joint_limit_margin[:, 0], (joints - PANDA_JOINT_LIMITS[:, 0]).min(axis=0)
            )
            joint_limit_margin[:, 1] = np.minimum(
                joint_limit_margin[:, 1], (PANDA_JOINT_LIMITS[:, 1] - joints).min(axis=0)
            )

            if total_episodes % 37 == 0:  # sparse sampling keeps the arrays small
                jpeg_bytes["image"].append(ep.jpeg_sizes("image"))
                jpeg_bytes["wrist_image"].append(ep.jpeg_sizes("wrist_image"))
                action_quantum_pool.append(action[:, :6])

        print(f"  parsed {subset:15s} episodes so far: {total_episodes}")

    # -- action quantisation, per channel group
    quant_pool = np.concatenate(action_quantum_pool)
    quant = {
        "position_xyz": quantisation(quant_pool[:, :3].reshape(-1)),
        "rotation_rpy": quantisation(quant_pool[:, 3:6].reshape(-1)),
    }
    pool = np.abs(quant_pool.reshape(-1))
    pool = pool[pool > 1e-9]

    total_frames = sum(v["frames"] for v in per_subset.values())
    all_lengths = np.concatenate([np.asarray(v["lengths"]) for v in per_subset.values()])

    tasks_out = []
    for (subset, task), rec in sorted(per_task.items()):
        lengths = np.asarray(rec["lengths"])
        tasks_out.append(
            {
                "subset": subset,
                "task": task,
                "instruction": sorted(rec["instructions"])[0],
                "num_instructions": len(rec["instructions"]),
                "episodes": rec["episodes"],
                "frames": rec["frames"],
                "frame_share_pct": round(100 * rec["frames"] / total_frames, 3),
                "len_min": int(lengths.min()),
                "len_max": int(lengths.max()),
                "len_mean": round(float(lengths.mean()), 1),
            }
        )

    subsets_out = {}
    for subset, rec in per_subset.items():
        lengths = np.asarray(rec["lengths"])
        subsets_out[subset] = {
            "episodes": rec["episodes"],
            "frames": rec["frames"],
            "num_tasks": sum(1 for s, _ in per_task if s == subset),
            "retention_pct": round(100 * rec["episodes"] / OFFICIAL_DEMOS_PER_SUITE, 2),
            "episode_share_pct": round(100 * rec["episodes"] / total_episodes, 2),
            "frame_share_pct": round(100 * rec["frames"] / total_frames, 2),
            "len_min": int(lengths.min()),
            "len_max": int(lengths.max()),
            "len_mean": round(float(lengths.mean()), 1),
            "len_p50": int(np.percentile(lengths, 50)),
            "len_p95": int(np.percentile(lengths, 95)),
        }

    aa = np.concatenate(axisangle_norms)
    report = {
        "dataset": str(Path("/B/Dta/opvla_libero")),
        "totals": {
            "episodes": total_episodes,
            "frames": total_frames,
            "official_demos": OFFICIAL_DEMOS_PER_SUITE * len(SUBSETS),
            "retention_pct": round(100 * total_episodes / (OFFICIAL_DEMOS_PER_SUITE * len(SUBSETS)), 2),
            "num_tasks": len(per_task),
        },
        "subsets": subsets_out,
        "tasks": tasks_out,
        "episode_length": {
            "min": int(all_lengths.min()),
            "max": int(all_lengths.max()),
            "mean": round(float(all_lengths.mean()), 1),
            "p50": int(np.percentile(all_lengths, 50)),
            "p95": int(np.percentile(all_lengths, 95)),
            "frac_shorter_than_200": round(float((all_lengths < 200).mean()), 4),
        },
        "action": {
            **action_stats.as_dict(),
            "abs_max_first6": round(action_abs_max, 9),
            "quantisation": quant,
            "saturation_origin": (
                "teleoperation, not a controller limit. Both channel groups saturate at "
                "exactly 1050 = 3 * 350 grid levels, and 350 is robosuite's SpaceMouse "
                "axis_scale (robosuite/devices/spacemouse.py:67); LIBERO collected these "
                "demos with --pos-sensitivity 1.5 / --rot-sensitivity 1.0 "
                "(LIBERO/scripts/collect_demonstration.py:237-246). OSC_POSE itself "
                "accepts the full [-1, 1], so |a|>0.9375 is legal but unseen in training."
            ),
            "physical_span_per_step": {
                "note": "action * output_max from robosuite/controllers/config/osc_pose.json",
                "position_max_m": round(0.9375 * 0.05, 6),
                "position_step_m": round(0.05 / 1120, 9),
                "rotation_max_rad": round(0.375 * 0.5, 6),
                "rotation_step_rad": round(0.5 / 2800, 9),
            },
            "frames_at_clip": clipped_frames,
            "frames_at_clip_pct": round(100 * clipped_frames / total_frames, 3),
            "gripper_values": {str(k): int(v) for k, v in sorted(gripper_values.items())},
        },
        "state": {
            **state_stats.as_dict(),
            "layout": "eef_pos(3) + axisangle(3) + gripper_qpos(2)",
            "axisangle_norm": {
                "min": round(float(aa.min()), 6),
                "max": round(float(aa.max()), 6),
                "mean": round(float(aa.mean()), 6),
                "frac_within_0.1rad_of_pi": round(float((np.abs(aa - np.pi) < 0.1).mean()), 4),
            },
            "axisangle_double_cover": {
                "note": (
                    "robosuite quat2axisangle returns 2*acos(qw) in [0, 2*pi] without "
                    "folding qw<0 into the qw>=0 hemisphere, so |axisangle| can exceed pi "
                    "and flips discontinuously when qw crosses 0"
                ),
                "frames_with_norm_over_pi": over_pi_frames,
                "frames_with_norm_over_pi_pct": round(100 * over_pi_frames / total_frames, 3),
                "episodes_with_jump_over_1rad": episodes_with_aa_jump,
                "episodes_with_jump_pct": round(100 * episodes_with_aa_jump / total_episodes, 2),
                "largest_single_step_jump_rad": round(aa_jump_max, 4),
            },
        },
        "joint_state": {
            **joint_stats.as_dict(),
            "limit_margin_min_rad": joint_limit_margin.round(4).tolist(),
            "max_limit_overshoot_rad": round(float(max(0.0, -joint_limit_margin.min())), 4),
            "limit_overshoot_note": (
                "small negative margins are OSC tracking overshoot inside the simulator, "
                "not corrupt data; magnitudes stay near 1e-2 rad"
            ),
        },
        "integrity": {
            "nan_frames": nan_frames,
            "flag_violations": len(flag_violations),
            "reward_violations": len(reward_violations),
            "discount_violations": len(discount_violations),
            "flag_violation_examples": flag_violations[:5],
        },
        "noop_audit": {
            "criterion": "||action[:6]|| <= 1e-4 and gripper unchanged (OpenVLA)",
            "frames_flagged": noop_frames,
            "frames_flagged_pct": round(100 * noop_frames / total_frames, 4),
            "note": "first frame of each episode has no predecessor and counts as no-op by definition",
            "episodes": total_episodes,
            "tiny_eef_motion_frames": tiny_motion_frames,
            "tiny_eef_motion_pct": round(100 * tiny_motion_frames / total_frames, 3),
            "tiny_eef_motion_criterion": "||eef_pos[t+1] - eef_pos[t]|| < 1 mm",
        },
        "images": {
            "declared_shape": [256, 256, 3],
            "encoding": "jpeg",
            "cameras": ["image (agentview)", "wrist_image (robot0_eye_in_hand)"],
            "sampled_episodes": len(jpeg_bytes["image"]),
            "jpeg_bytes_mean": {
                cam: round(float(np.concatenate(v).mean()), 1) for cam, v in jpeg_bytes.items()
            },
            "jpeg_bytes_max": {cam: int(np.concatenate(v).max()) for cam, v in jpeg_bytes.items()},
        },
    }

    OUT_JSON.write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        OUT_NPZ,
        episode_lengths=all_lengths,
        axisangle_norms=aa[::7],
        eef_positions=np.concatenate(eef_positions),
        action_pool=pool,
    )
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_NPZ}")

    t = report["totals"]
    print(f"\nepisodes={t['episodes']} frames={t['frames']} retention={t['retention_pct']}%")
    print(f"action abs max (first 6) = {action_abs_max}  quantisation = {quant}")
    print(f"gripper values = {list(report['action']['gripper_values'])}")
    print(f"no-op frames = {noop_frames} ({report['noop_audit']['frames_flagged_pct']}%)")
    print(f"integrity = {report['integrity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
