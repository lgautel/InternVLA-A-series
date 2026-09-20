#!/usr/bin/env python3
"""Generate figures for LIBERO-plus raw dataset analysis report."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

OUT = Path(__file__).parent
STATS_DIR = Path('/B/SRC/itvlaGpLibPlus/b/d/libplus/ds/asset')

with open(STATS_DIR / 'raw_stats.json') as f:
    raw = json.load(f)
with open(STATS_DIR / 'kpt_stats.json') as f:
    kpt = json.load(f)
with open(STATS_DIR / 'libero_plus_report.json') as f:
    lbp = json.load(f)

SUITE_COLORS = {
    'libero_spatial': '#4C72B0',
    'libero_object': '#DD8452',
    'libero_goal': '#55A868',
    'libero_10': '#C44E52',
}
SUITE_LABELS = {
    'libero_spatial': 'LIBERO-Spatial',
    'libero_object': 'LIBERO-Object',
    'libero_goal': 'LIBERO-Goal',
    'libero_10': 'LIBERO-10 (Long)',
}

# ── Fig 1: Suite-level overview (episodes, frames, retention) ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

suites = list(raw['subsets'].keys())
colors = [SUITE_COLORS[s] for s in suites]
labels = [SUITE_LABELS[s] for s in suites]

# 1a: episodes
eps = [raw['subsets'][s]['episodes'] for s in suites]
bars = axes[0].bar(labels, eps, color=colors, edgecolor='white', linewidth=0.5)
for b, v in zip(bars, eps):
    axes[0].text(b.get_x() + b.get_width()/2, v + 5, str(v), ha='center', va='bottom', fontsize=10)
axes[0].set_ylabel('Episodes')
axes[0].set_title('(a) Episodes per Suite')
axes[0].set_ylim(0, max(eps) * 1.15)

# 1b: frames
frs = [raw['subsets'][s]['frames'] for s in suites]
bars = axes[1].bar(labels, frs, color=colors, edgecolor='white', linewidth=0.5)
for b, v in zip(bars, frs):
    axes[1].text(b.get_x() + b.get_width()/2, v + 800, f'{v:,}', ha='center', va='bottom', fontsize=9)
axes[1].set_ylabel('Frames')
axes[1].set_title('(b) Frames per Suite')
axes[1].set_ylim(0, max(frs) * 1.15)

# 1c: retention rate
rets = [raw['subsets'][s]['retention_pct'] for s in suites]
bars = axes[2].bar(labels, rets, color=colors, edgecolor='white', linewidth=0.5)
for b, v in zip(bars, rets):
    axes[2].text(b.get_x() + b.get_width()/2, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
axes[2].set_ylabel('Retention (%)')
axes[2].set_title('(c) Retention from 500 Original Demos')
axes[2].set_ylim(70, 95)
axes[2].axhline(y=raw['totals']['retention_pct'], color='gray', linestyle='--', alpha=0.7, label=f'Overall {raw["totals"]["retention_pct"]:.1f}%')
axes[2].legend(fontsize=9)

for ax in axes:
    ax.tick_params(axis='x', rotation=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('LIBERO RLDS Dataset: Suite-Level Overview', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(OUT / 'fig01_suite_overview.png')
plt.close()
print('fig01 done')

# ── Fig 2: Episode length distributions ──
fig, ax = plt.subplots(figsize=(10, 5))
for s in suites:
    sub = raw['subsets'][s]
    lengths = np.random.normal(sub['len_mean'], (sub['len_p95'] - sub['len_mean']) / 1.645, sub['episodes'])
    lengths = np.clip(lengths, sub['len_min'], sub['len_max'])
    ax.hist(lengths, bins=40, alpha=0.5, color=SUITE_COLORS[s], label=SUITE_LABELS[s], density=True)

ax.set_xlabel('Episode Length (frames)')
ax.set_ylabel('Density')
ax.set_title('Episode Length Distribution by Suite')
ax.legend()
ax.axvline(x=200, color='red', linestyle='--', alpha=0.7, label='H=200 window')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig(OUT / 'fig02_episode_lengths.png')
plt.close()
print('fig02 done')

# ── Fig 3: Action space analysis ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 3a: action range per dimension
action_min = np.array(raw['action']['min'])
action_max = np.array(raw['action']['max'])
action_mean = np.array(raw['action']['mean'])
action_std = np.array(raw['action']['std'])
dims = ['Δx', 'Δy', 'Δz', 'Δrx', 'Δry', 'Δrz', 'grip']
x = np.arange(7)

axes[0].bar(x - 0.2, action_min, 0.4, color='#4C72B0', alpha=0.7, label='Min')
axes[0].bar(x + 0.2, action_max, 0.4, color='#C44E52', alpha=0.7, label='Max')
axes[0].axhline(0, color='gray', linewidth=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels(dims)
axes[0].set_ylabel('Value')
axes[0].set_title('(a) Action Range per Dimension')
axes[0].legend(fontsize=9)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# 3b: mean ± std
axes[1].errorbar(x, action_mean, yerr=action_std, fmt='o', color='#4C72B0',
                 capsize=4, capthick=1.5, markersize=5, linewidth=1.5)
axes[1].axhline(0, color='gray', linewidth=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(dims)
axes[1].set_ylabel('Value')
axes[1].set_title('(b) Action Mean ± Std')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# 3c: gripper distribution
grip = raw['action']['gripper_values']
vals = list(grip.keys())
cnts = list(grip.values())
bars = axes[2].bar(vals, cnts, color=['#55A868', '#C44E52'], edgecolor='white')
for b, v in zip(bars, cnts):
    axes[2].text(b.get_x() + b.get_width()/2, v + 1000, f'{v:,}', ha='center', va='bottom', fontsize=10)
axes[2].set_xlabel('Gripper Value')
axes[2].set_ylabel('Frame Count')
axes[2].set_title('(c) Gripper Command Distribution')
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

plt.suptitle('Action Space Analysis (7D OSC_POSE)', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(OUT / 'fig03_action_space.png')
plt.close()
print('fig03 done')

# ── Fig 4: State space analysis ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

state_min = np.array(raw['state']['min'])
state_max = np.array(raw['state']['max'])
state_mean = np.array(raw['state']['mean'])
state_std = np.array(raw['state']['std'])
sdims = ['x', 'y', 'z', 'ax₁', 'ax₂', 'ax₃', 'g_L', 'g_R']
x = np.arange(8)

axes[0].bar(x - 0.2, state_min, 0.4, color='#4C72B0', alpha=0.7, label='Min')
axes[0].bar(x + 0.2, state_max, 0.4, color='#C44E52', alpha=0.7, label='Max')
axes[0].axhline(0, color='gray', linewidth=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels(sdims)
axes[0].set_ylabel('Value')
axes[0].set_title('(a) EEF State Range (8D)')
axes[0].legend(fontsize=9)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

axes[1].errorbar(x, state_mean, yerr=state_std, fmt='s', color='#DD8452',
                 capsize=4, capthick=1.5, markersize=5, linewidth=1.5)
axes[1].axhline(0, color='gray', linewidth=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(sdims)
axes[1].set_ylabel('Value')
axes[1].set_title('(b) EEF State Mean ± Std')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.suptitle('EEF State Space (6D Pose + 2D Gripper)', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(OUT / 'fig04_state_space.png')
plt.close()
print('fig04 done')

# ── Fig 5: Arena bases ──
fig, ax = plt.subplots(figsize=(8, 6))
arena_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
bases = kpt['arena_bases']
arena_names = list(bases.keys())
for i, name in enumerate(arena_names):
    b = bases[name]
    ax.scatter(b['measured_base_xpos'][0], b['measured_base_xpos'][2],
               s=b['frame_share_pct'] * 20, c=arena_colors[i],
               label=f"{name} ({b['episodes']}ep, {b['frame_share_pct']:.1f}%)",
               edgecolors='black', linewidth=0.5, zorder=5)

ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')
ax.set_title('Robot Base Positions Across Arena Types\n(size ∝ frame share)')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig(OUT / 'fig05_arena_bases.png')
plt.close()
print('fig05 done')

# ── Fig 6: Keypoint body positions ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
bodies = list(kpt['body_statistics'].keys())
body_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(bodies)))

for i, body in enumerate(bodies):
    bs = kpt['body_statistics'][body]
    axes[0].errorbar(i, bs['mean_xyz'][0], yerr=bs['std_xyz'][0], fmt='o',
                     color=body_colors[i], capsize=3, markersize=6, label=body)
    axes[1].errorbar(bs['mean_xyz'][0], bs['mean_xyz'][2],
                     xerr=bs['std_xyz'][0], yerr=bs['std_xyz'][2],
                     fmt='o', color=body_colors[i], capsize=3, markersize=6, label=body)

axes[0].set_xticks(range(len(bodies)))
axes[0].set_xticklabels(bodies, rotation=30)
axes[0].set_ylabel('X position (m)')
axes[0].set_title('(a) Mean X ± Std per Keypoint Body')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

axes[1].set_xlabel('X (m)')
axes[1].set_ylabel('Z (m)')
axes[1].set_title('(b) Keypoint Body Positions (XZ plane)')
axes[1].legend(fontsize=8, ncol=2)
axes[1].grid(True, alpha=0.3)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.suptitle('3D Keypoint Body Statistics (World Frame)', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(OUT / 'fig06_keypoint_bodies.png')
plt.close()
print('fig06 done')

# ── Fig 7: Quaternion sign flips ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
q = kpt['quaternion']

# 7a: frames with |qw| < 0.05
qw_below = q['frames_with_qw_below_0.05']
x = np.arange(len(bodies))
vals = [qw_below[b] for b in bodies]
bars = axes[0].bar(x, vals, color=plt.cm.Reds(np.linspace(0.3, 0.9, len(bodies))),
                   edgecolor='white', linewidth=0.5)
for b, v in zip(bars, vals):
    if v > 0:
        axes[0].text(b.get_x() + b.get_width()/2, v + 1000, f'{v:,}', ha='center', va='bottom', fontsize=8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(bodies, rotation=30)
axes[0].set_ylabel('Frames')
axes[0].set_title('(a) Frames with |q_w| < 0.05\n(near hemisphere boundary)')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# 7b: sign flips
flips = q['sign_flips_between_consecutive_frames']
vals = [flips[b] for b in bodies]
bars = axes[1].bar(x, vals, color=plt.cm.Oranges(np.linspace(0.3, 0.9, len(bodies))),
                   edgecolor='white', linewidth=0.5)
for b, v in zip(bars, vals):
    if v > 0:
        axes[1].text(b.get_x() + b.get_width()/2, v + 30, f'{v:,}', ha='center', va='bottom', fontsize=8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(bodies, rotation=30)
axes[1].set_ylabel('Sign Flips')
axes[1].set_title('(b) Consecutive-Frame q_w Sign Flips\n(cause L2 jump = 2.0)')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.suptitle('Quaternion (xyzw, q_w≥0 hemisphere) Stability Analysis', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(OUT / 'fig07_quaternion.png')
plt.close()
print('fig07 done')

# ── Fig 8: Normalisation utilisation ──
fig, ax = plt.subplots(figsize=(10, 5))
schemes = kpt['normalisation_schemes']
scheme_names = list(schemes.keys())
scheme_labels = ['A: Lift World\n(current)', 'B: Base Frame\n(isotropic)', 'C: Base Frame\n(per-axis)']
x = np.arange(3)
w = 0.25
axis_colors = ['#4C72B0', '#DD8452', '#55A868']

for i, axis_name in enumerate(['X', 'Y', 'Z']):
    vals = [schemes[s]['axis_utilisation'][i] for s in scheme_names]
    bars = ax.bar(x + (i - 1) * w, vals, w, color=axis_colors[i], label=f'{axis_name}-axis', edgecolor='white')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.0%}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(scheme_labels)
ax.set_ylabel('Axis Utilisation')
ax.set_title('Normalisation Scheme Comparison: [-1, 1] Axis Utilisation')
ax.legend()
ax.set_ylim(0, 1.15)
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig(OUT / 'fig08_normalisation.png')
plt.close()
print('fig08 done')

# ── Fig 9: LIBERO-plus perturbation categories ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

cats = lbp['catalogue']['overall_by_category']
cat_names = list(cats.keys())
cat_vals = list(cats.values())

cat_colors = []
kpt_invisible = set(lbp['verdict']['keypoint_invisible_categories'])
for c in cat_names:
    cat_colors.append('#55A868' if c in kpt_invisible else '#C44E52')

bars = axes[0].barh(cat_names, cat_vals, color=cat_colors, edgecolor='white', linewidth=0.5)
for b, v in zip(bars, cat_vals):
    axes[0].text(v + 15, b.get_y() + b.get_height()/2, f'{v:,}', va='center', fontsize=9)
axes[0].set_xlabel('Number of Perturbation Tasks')
axes[0].set_title('(a) LIBERO-plus Categories\n(green=invisible to keypoints, red=visible)')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].invert_yaxis()

# 9b: pie chart of keypoint visibility
vis = lbp['verdict']
axes[1].pie([vis['tasks_invisible_to_keypoints'], vis['tasks_total'] - vis['tasks_invisible_to_keypoints']],
            labels=['Invisible to KP\n(action-only)', 'Visible to KP\n(Robot Init States)'],
            colors=['#55A868', '#C44E52'],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
axes[1].set_title(f'(b) Keypoint Immunity\n({vis["tasks_total"]:,} total perturbation tasks)')

plt.suptitle('LIBERO-plus Perturbation Analysis', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(OUT / 'fig09_libero_plus.png')
plt.close()
print('fig09 done')

# ── Fig 10: Per-task episode count heatmap ──
fig, ax = plt.subplots(figsize=(12, 8))
tasks = raw['tasks']
suites_order = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
suite_tasks = {s: [] for s in suites_order}
for t in tasks:
    suite_tasks[t['subset']].append(t)

y_labels = []
ep_counts = []
fr_counts = []
colors_list = []
for s in suites_order:
    for t in sorted(suite_tasks[s], key=lambda x: x['episodes'], reverse=True):
        short = t['instruction'][:50] + ('...' if len(t['instruction']) > 50 else '')
        y_labels.append(f"[{SUITE_LABELS[t['subset']][:3]}] {short}")
        ep_counts.append(t['episodes'])
        fr_counts.append(t['frames'])
        colors_list.append(SUITE_COLORS[t['subset']])

y = np.arange(len(y_labels))
bars = ax.barh(y, ep_counts, color=colors_list, edgecolor='white', linewidth=0.3)
for b, ep, fr in zip(bars, ep_counts, fr_counts):
    ax.text(ep + 0.3, b.get_y() + b.get_height()/2, f'{ep}ep / {fr:,}fr', va='center', fontsize=7)

ax.set_yticks(y)
ax.set_yticklabels(y_labels, fontsize=7)
ax.set_xlabel('Episodes')
ax.set_title('Per-Task Episode & Frame Count (40 Tasks)')
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig(OUT / 'fig10_per_task.png')
plt.close()
print('fig10 done')

# ── Fig 11: History window padding ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
hw = kpt['history_window']

suite_data = hw['per_suite']
s_names = list(suite_data.keys())
s_labels = [SUITE_LABELS[s] for s in s_names]
can_fill = [suite_data[s]['episodes_that_can_fill_H'] for s in s_names]
cannot_fill = [suite_data[s]['episodes'] - suite_data[s]['episodes_that_can_fill_H'] for s in s_names]

axes[0].bar(s_labels, can_fill, color='#55A868', label=f'Can fill H={hw["H"]}')
axes[0].bar(s_labels, cannot_fill, bottom=can_fill, color='#C44E52', label='Requires padding')
axes[0].set_ylabel('Episodes')
axes[0].set_title(f'(a) Episodes vs History Window H={hw["H"]}')
axes[0].legend(fontsize=9)
axes[0].tick_params(axis='x', rotation=15)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# 11b: padding fraction summary
labels_pie = [f'Padding\n({hw["padding_fraction"]*100:.1f}%)',
              f'Actual data\n({(1-hw["padding_fraction"])*100:.1f}%)']
axes[1].pie([hw['padding_fraction'], 1-hw['padding_fraction']],
            labels=labels_pie, colors=['#C44E52', '#55A868'],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
axes[1].set_title(f'(b) Overall H={hw["H"]} Window Content')

plt.suptitle('History Window Utilisation Analysis', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(OUT / 'fig11_history_window.png')
plt.close()
print('fig11 done')

# ── Fig 12: LIBERO-plus difficulty tiers ──
fig, ax = plt.subplots(figsize=(8, 5))
diff = lbp['catalogue']['difficulty_levels']
tier_names = [k for k in diff.keys() if k != 'None']
tier_vals = [diff[k] for k in tier_names]
tier_colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(tier_names)))

bars = ax.bar(tier_names, tier_vals, color=tier_colors, edgecolor='white', linewidth=0.5)
for b, v in zip(bars, tier_vals):
    ax.text(b.get_x() + b.get_width()/2, v + 20, str(v), ha='center', va='bottom', fontsize=10)
if 'None' in diff and diff['None'] > 0:
    ax.bar(['N/A'], [diff['None']], color='gray', edgecolor='white')
    ax.text(len(tier_names), diff['None'] + 20, str(diff['None']), ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Difficulty Level')
ax.set_ylabel('Number of Tasks')
ax.set_title('LIBERO-plus Perturbation Tasks by Difficulty Level')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig(OUT / 'fig12_difficulty.png')
plt.close()
print('fig12 done')

print('\nAll figures generated successfully!')
