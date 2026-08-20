"""
Generate a comparison table figure for ELAN4D vs Pri4R vs GeoPredict vs InternVLA-A1.5.
Output: method_comparison.png
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('off')

methods = ['ELAN4D', 'Pri4R', 'GeoPredict', 'InternVLA-A1.5']
dimensions = [
    '4D Signal Source',
    'Tracked Points',
    'Preprocess Cost',
    'Injection Point',
    'VLM Protection',
    'Inference Overhead',
    'LIBERO Overall',
    'OOD Robustness',
]

data = [
    ['FK (proprioception)', 'SpatialTrackerV2', 'Track Encoder+3DGS', 'Frozen WAN2.2-5B'],
    ['Robot joints+EE\n(K=7~14)', 'Robot+Scene\n(N~1024)', 'Robot keypoints\n+ 3D Gaussians', 'Video frames\n(latent space)'],
    ['~1 CPU-min/hr', '~4 GPU-hr/hr', 'Medium (GPU)', 'Medium (GPU)'],
    ['ControlNet\nAction Branch', 'VLM Backbone\n(shared repr)', 'VLM\n(track queries)', 'Foresight Tokens\n(Action Expert)'],
    ['Strong\n(stop-gradient)', 'No explicit\nprotection', 'Weak\n(repr drift)', 'Strong\n(WAN frozen)'],
    ['Zero', 'Zero', 'Extra queries', 'Zero'],
    ['97.0%', '96.3%', '96.6%', 'N/A'],
    ['Strong\n(+4.6 LIBERO-Plus)', 'Medium', 'Weak\n(-6.8 w/ VLM query)', 'Strong'],
]

colors_header = '#1a5276'
colors_row_method = '#2874a6'
colors_good = '#d4efdf'
colors_ok = '#fdebd0'
colors_bad = '#fadbd8'
colors_na = '#f2f3f4'
colors_white = '#ffffff'

quality_map = {
    (0, 0): colors_good, (1, 0): colors_bad, (2, 0): colors_ok, (3, 0): colors_ok,
    (0, 1): colors_ok,   (1, 1): colors_good, (2, 1): colors_good, (3, 1): colors_good,
    (0, 2): colors_good, (1, 2): colors_bad, (2, 2): colors_ok, (3, 2): colors_ok,
    (0, 3): colors_good, (1, 3): colors_ok, (2, 3): colors_bad, (3, 3): colors_good,
    (0, 4): colors_good, (1, 4): colors_bad, (2, 4): colors_bad, (3, 4): colors_good,
    (0, 5): colors_good, (1, 5): colors_good, (2, 5): colors_ok, (3, 5): colors_good,
    (0, 6): colors_good, (1, 6): colors_ok, (2, 6): colors_ok, (3, 6): colors_na,
    (0, 7): colors_good, (1, 7): colors_ok, (2, 7): colors_bad, (3, 7): colors_good,
}

n_rows = len(dimensions) + 1
n_cols = len(methods) + 1
cell_w = 1.0 / n_cols
cell_h = 1.0 / n_rows

for j, m in enumerate(methods):
    x = (j + 1) * cell_w
    rect = plt.Rectangle((x, 1 - cell_h), cell_w, cell_h,
                          facecolor=colors_header, edgecolor='white', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + cell_w / 2, 1 - cell_h / 2, m,
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')

for i, dim in enumerate(dimensions):
    y = 1 - (i + 2) * cell_h
    rect = plt.Rectangle((0, y), cell_w, cell_h,
                          facecolor=colors_row_method, edgecolor='white', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(cell_w / 2, y + cell_h / 2, dim,
            ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')

for i in range(len(dimensions)):
    for j in range(len(methods)):
        x = (j + 1) * cell_w
        y = 1 - (i + 2) * cell_h
        bg = quality_map.get((j, i), colors_white)
        rect = plt.Rectangle((x, y), cell_w, cell_h,
                              facecolor=bg, edgecolor='#d5d8dc', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(x + cell_w / 2, y + cell_h / 2, data[i][j],
                ha='center', va='center', fontsize=7.5, color='#2c3e50')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('4D Supervision Methods for VLA: Comprehensive Comparison',
             fontsize=13, fontweight='bold', pad=12, color='#2c3e50')

legend_items = [
    (colors_good, 'Advantage'),
    (colors_ok, 'Moderate'),
    (colors_bad, 'Limitation'),
    (colors_na, 'N/A'),
]
for idx, (c, label) in enumerate(legend_items):
    x_pos = 0.15 + idx * 0.18
    rect = plt.Rectangle((x_pos, -0.04), 0.025, 0.025,
                          facecolor=c, edgecolor='#7f8c8d', linewidth=0.5,
                          transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(x_pos + 0.035, -0.027, label, transform=ax.transAxes,
            fontsize=8, va='center', color='#2c3e50')

plt.tight_layout()
plt.savefig('d:/SRC/Robot/InternVLA-A-series/b/d/ELAN4D/asset/method_comparison.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved method_comparison.png")
