#!/usr/bin/env python3
"""Independently compute 3D keypoint and quaternion statistics from RLDS data via FK."""
import json, sys, numpy as np
sys.path.insert(0, '.')
from rlds_reader import SUBSETS, iter_episodes
from panda_fk import PandaFK, LIFT_BASE_XPOS, DEFAULT_R_PAD, wxyz_to_xyzw_hemisphere

fk = PandaFK()

# Accumulators
all_positions = []  # [N, 8, 3] normalized
all_quats = []      # [N, 8, 4] xyzw hemisphere
qw_near_zero = np.zeros(8, dtype=int)  # count of |qw| < 0.05 per body
antipodal_jumps = np.zeros(8, dtype=int)  # count of jumps per body
total_frames = 0
body_pos_stats = [{'min': np.full(3, np.inf), 'max': np.full(3, -np.inf)} for _ in range(8)]
body_qw_ranges = [[] for _ in range(8)]

# Sample every 10th frame to keep memory manageable
SAMPLE_RATE = 10
ep_count = 0

for subset in SUBSETS:
    print(f'Processing {subset}...', file=sys.stderr)
    for ep in iter_episodes(subset):
        ep_count += 1
        qpos = ep.qpos9()
        T = len(qpos)
        total_frames += T

        prev_quats = None
        for t in range(T):
            poses = fk.world_poses(qpos[t])  # [8, 7] in world frame with Lift base
            pos_norm = poses[:, :3] / DEFAULT_R_PAD
            quats_xyzw = np.array([wxyz_to_xyzw_hemisphere(poses[i, 3:]) for i in range(8)])

            # Track qw near zero
            for j in range(8):
                if abs(quats_xyzw[j, 3]) < 0.05:
                    qw_near_zero[j] += 1
                body_qw_ranges[j].append(quats_xyzw[j, 3])

            # Track antipodal jumps within episode
            if prev_quats is not None:
                for j in range(8):
                    dot = abs(np.dot(quats_xyzw[j], prev_quats[j]))
                    if dot < 0.5:  # large jump = antipodal
                        antipodal_jumps[j] += 1
            prev_quats = quats_xyzw.copy()

            # Update position stats
            for j in range(8):
                body_pos_stats[j]['min'] = np.minimum(body_pos_stats[j]['min'], pos_norm[j])
                body_pos_stats[j]['max'] = np.maximum(body_pos_stats[j]['max'], pos_norm[j])

            # Sample for distributions
            if t % SAMPLE_RATE == 0:
                all_positions.append(pos_norm.copy())
                all_quats.append(quats_xyzw.copy())

print(f'Processed {ep_count} episodes, {total_frames} frames', file=sys.stderr)

all_positions = np.array(all_positions)  # [N_sampled, 8, 3]
all_quats = np.array(all_quats)          # [N_sampled, 8, 4]

BODY_NAMES = ['link1','link2','link3','link4','link5','link6','link7','eef']

# Position analysis per body
pos_analysis = {}
for j, name in enumerate(BODY_NAMES):
    pos_j = all_positions[:, j, :]  # [N, 3]
    pos_std = pos_j.std(axis=0)
    pos_range = pos_j.max(axis=0) - pos_j.min(axis=0)
    pos_analysis[name] = {
        'mean': pos_j.mean(axis=0).round(6).tolist(),
        'std': pos_std.round(6).tolist(),
        'range': pos_range.round(6).tolist(),
        'is_static': bool(pos_std.max() < 1e-5),
    }

# Quaternion analysis per body
quat_analysis = {}
for j, name in enumerate(BODY_NAMES):
    qw_vals = np.array(body_qw_ranges[j])
    quat_analysis[name] = {
        'qw_min': float(np.min(qw_vals).round(6)),
        'qw_max': float(np.max(qw_vals).round(6)),
        'qw_mean': float(np.mean(qw_vals).round(6)),
        'qw_near_zero_count': int(qw_near_zero[j]),
        'qw_near_zero_pct': round(100 * qw_near_zero[j] / total_frames, 2),
        'antipodal_jumps': int(antipodal_jumps[j]),
    }

# Check link1/link2 position redundancy
link1_pos = all_positions[:, 0, :]
link2_pos = all_positions[:, 1, :]
link12_max_dist = float(np.linalg.norm(link1_pos - link2_pos, axis=1).max())

# Check link5/link6 position redundancy
link5_pos = all_positions[:, 4, :]
link6_pos = all_positions[:, 5, :]
link56_max_dist = float(np.linalg.norm(link5_pos - link6_pos, axis=1).max())

# Quaternion variation per body (using std of each component)
quat_var = {}
for j, name in enumerate(BODY_NAMES):
    q = all_quats[:, j, :]
    quat_var[name] = {
        'qx_std': float(q[:, 0].std().round(6)),
        'qy_std': float(q[:, 1].std().round(6)),
        'qz_std': float(q[:, 2].std().round(6)),
        'qw_std': float(q[:, 3].std().round(6)),
        'has_rotation_info': bool(q.std(axis=0).max() > 0.01),
    }

report = {
    'total_frames': total_frames,
    'total_episodes': ep_count,
    'sampled_frames': len(all_positions),
    'body_names': BODY_NAMES,
    'position_analysis': pos_analysis,
    'quaternion_analysis': quat_analysis,
    'quaternion_variation': quat_var,
    'redundancy': {
        'link1_link2_max_position_distance': link12_max_dist,
        'link5_link6_max_position_distance': link56_max_dist,
        'link1_is_static': pos_analysis['link1']['is_static'],
        'link2_is_static': pos_analysis['link2']['is_static'],
    },
    'total_antipodal_jumps': int(antipodal_jumps.sum()),
    'r_pad_used': DEFAULT_R_PAD,
}

with open('/tmp/kpt_quat_stats.json', 'w') as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
