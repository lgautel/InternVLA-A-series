"""
Draw the token sequence layout and attention mask pattern for the fused model.
Shows how prefix tokens (including new keypoint tokens) and suffix tokens are arranged,
and which groups can attend to which.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 1.5]})

# ============================================================
# Panel 1: Token Sequence Layout
# ============================================================
ax = axes[0]
ax.set_xlim(0, 16)
ax.set_ylim(0, 4)
ax.axis('off')

ax.text(8, 3.7, 'Fused Token Sequence Layout', fontsize=12, fontweight='bold', ha='center')

# PREFIX
ax.text(0.3, 3.1, 'PREFIX (dim=2048)', fontsize=9, fontweight='bold', color='#2E7D32')
bg_pfx = Rectangle((0.2, 1.5), 10.0, 1.5, facecolor='#E8F5E9', edgecolor='#2E7D32',
                    linewidth=2, alpha=0.3)
ax.add_patch(bg_pfx)

# Image tokens
draw_x = 0.4
ax.add_patch(FancyBboxPatch((draw_x, 1.7), 2.5, 1.0, boxstyle="round,pad=0.03",
             facecolor='#A5D6A7', edgecolor='#388E3C', linewidth=1.5))
ax.text(draw_x + 1.25, 2.2, 'Image Tokens\n(variable)', ha='center', va='center', fontsize=7)
ax.text(draw_x + 1.25, 1.55, 'att: [1,1,1,...] (causal)', ha='center', fontsize=5.5, color='#616161')

# Language tokens
draw_x = 3.1
ax.add_patch(FancyBboxPatch((draw_x, 1.7), 2.5, 1.0, boxstyle="round,pad=0.03",
             facecolor='#A5D6A7', edgecolor='#388E3C', linewidth=1.5))
ax.text(draw_x + 1.25, 2.2, 'Language Tokens\n(variable)', ha='center', va='center', fontsize=7)
ax.text(draw_x + 1.25, 1.55, 'att: [1,1,1,...] (causal)', ha='center', fontsize=5.5, color='#616161')

# History KPT tokens (NEW)
draw_x = 5.8
ax.add_patch(FancyBboxPatch((draw_x, 1.7), 1.8, 1.0, boxstyle="round,pad=0.03",
             facecolor='#81C784', edgecolor='#1B5E20', linewidth=2))
ax.text(draw_x + 0.9, 2.2, 'Hist KPT (8)\n(NEW)', ha='center', va='center',
        fontsize=7, fontweight='bold')
ax.text(draw_x + 0.9, 1.55, 'att: [1,0,0,...,0] (bidir)', ha='center', fontsize=5.5, color='#1B5E20')

# Query KPT tokens (NEW)
draw_x = 7.8
ax.add_patch(FancyBboxPatch((draw_x, 1.7), 1.8, 1.0, boxstyle="round,pad=0.03",
             facecolor='#81C784', edgecolor='#1B5E20', linewidth=2))
ax.text(draw_x + 0.9, 2.2, 'Query KPT (8)\n(NEW)', ha='center', va='center',
        fontsize=7, fontweight='bold')
ax.text(draw_x + 0.9, 1.55, 'att: [1,0,0,...,0] (bidir)', ha='center', fontsize=5.5, color='#1B5E20')

# SUFFIX
ax.text(10.8, 3.1, 'SUFFIX (dim=1024)', fontsize=9, fontweight='bold', color='#E65100')
bg_sfx = Rectangle((10.4, 1.5), 5.2, 1.5, facecolor='#FFF3E0', edgecolor='#E65100',
                    linewidth=2, alpha=0.3)
ax.add_patch(bg_sfx)

draw_x = 10.6
ax.add_patch(FancyBboxPatch((draw_x, 1.7), 0.8, 1.0, boxstyle="round,pad=0.03",
             facecolor='#FFCC80', edgecolor='#E65100', linewidth=1.5))
ax.text(draw_x + 0.4, 2.2, 'St\n(1)', ha='center', va='center', fontsize=6)
ax.text(draw_x + 0.4, 1.55, '[1]', ha='center', fontsize=5.5, color='#616161')

draw_x = 11.6
ax.add_patch(FancyBboxPatch((draw_x, 1.7), 1.8, 1.0, boxstyle="round,pad=0.03",
             facecolor='#FFCC80', edgecolor='#E65100', linewidth=1.5))
ax.text(draw_x + 0.9, 2.2, 'Foresight\n(50)', ha='center', va='center', fontsize=6)
ax.text(draw_x + 0.9, 1.55, '[1,0..0] (bidir)', ha='center', fontsize=5.5, color='#616161')

draw_x = 13.6
ax.add_patch(FancyBboxPatch((draw_x, 1.7), 1.8, 1.0, boxstyle="round,pad=0.03",
             facecolor='#FFCC80', edgecolor='#E65100', linewidth=1.5))
ax.text(draw_x + 0.9, 2.2, 'Action+Time\n(50)', ha='center', va='center', fontsize=6)
ax.text(draw_x + 0.9, 1.55, '[1,0..0] (bidir)', ha='center', fontsize=5.5, color='#616161')

# ============================================================
# Panel 2: Attention Pattern Matrix
# ============================================================
ax2 = axes[1]
ax2.set_xlim(0, 16)
ax2.set_ylim(0, 6)
ax2.axis('off')

ax2.text(8, 5.7, 'Block-Causal Attention Pattern', fontsize=12, fontweight='bold', ha='center')

groups = ['Image', 'Language', 'Hist KPT\n(NEW)', 'Query KPT\n(NEW)', 'State', 'Foresight', 'Action']
n = len(groups)

# Attention matrix values:
# 0 = no attention, 1 = causal, 2 = bidirectional, 3 = full
attn = np.array([
    [1, 0, 0, 0, 0, 0, 0],  # Image
    [3, 1, 0, 0, 0, 0, 0],  # Language
    [3, 3, 2, 0, 0, 0, 0],  # Hist KPT
    [3, 3, 3, 2, 0, 0, 0],  # Query KPT
    [3, 3, 3, 3, 1, 0, 0],  # State
    [3, 3, 3, 3, 3, 2, 0],  # Foresight
    [3, 3, 3, 3, 3, 3, 2],  # Action
])

colors_map = {0: '#FFFFFF', 1: '#BBDEFB', 2: '#81D4FA', 3: '#42A5F5'}
labels_map = {0: '', 1: 'Causal', 2: 'Bidir', 3: 'Full'}

cell_size = 0.65
start_x = 3.5
start_y = 0.5

for i in range(n):
    for j in range(n):
        x = start_x + j * cell_size
        y = start_y + (n - 1 - i) * cell_size
        val = attn[i, j]
        edgecolor = '#1B5E20' if (i in [2, 3] or j in [2, 3]) and val > 0 else '#90A4AE'
        lw = 1.5 if edgecolor == '#1B5E20' else 0.5
        rect = Rectangle((x, y), cell_size, cell_size,
                         facecolor=colors_map[val], edgecolor=edgecolor, linewidth=lw)
        ax2.add_patch(rect)
        if val > 0:
            fs = 5.5
            ax2.text(x + cell_size/2, y + cell_size/2, labels_map[val],
                    ha='center', va='center', fontsize=fs,
                    color='#212121' if val < 3 else 'white')

# Row labels (Query)
for i in range(n):
    y = start_y + (n - 1 - i) * cell_size + cell_size / 2
    color = '#1B5E20' if i in [2, 3] else '#424242'
    weight = 'bold' if i in [2, 3] else 'normal'
    ax2.text(start_x - 0.15, y, groups[i], ha='right', va='center',
            fontsize=7, color=color, fontweight=weight)

# Column labels (Key)
for j in range(n):
    x = start_x + j * cell_size + cell_size / 2
    color = '#1B5E20' if j in [2, 3] else '#424242'
    weight = 'bold' if j in [2, 3] else 'normal'
    ax2.text(x, start_y + n * cell_size + 0.15, groups[j],
            ha='center', va='bottom', fontsize=7, rotation=30,
            color=color, fontweight=weight)

ax2.text(start_x - 1.5, start_y + n * cell_size / 2, 'Query\n(row)',
        ha='center', va='center', fontsize=9, fontweight='bold', color='#424242')
ax2.text(start_x + n * cell_size / 2, start_y + n * cell_size + 0.9, 'Key (column)',
        ha='center', va='center', fontsize=9, fontweight='bold', color='#424242')

# Legend
legend_x = start_x + n * cell_size + 1.0
for i, (val, label_full) in enumerate([
    (0, 'No attention'),
    (1, 'Causal (lower triangular)'),
    (2, 'Bidirectional (within group)'),
    (3, 'Full attention (later -> earlier)'),
]):
    y_l = start_y + (n - 1) * cell_size - i * 0.6
    rect = Rectangle((legend_x, y_l), 0.4, 0.4,
                     facecolor=colors_map[val], edgecolor='#90A4AE', linewidth=1)
    ax2.add_patch(rect)
    ax2.text(legend_x + 0.55, y_l + 0.2, label_full, va='center', fontsize=7)

# Key insight annotation
ax2.text(start_x + n * cell_size + 1.0, start_y + 0.8,
        'Key insight: Action tokens (row 7) have\n'
        'FULL attention to Query KPT and Hist KPT\n'
        'tokens, gaining 3D kinematic awareness\n'
        'through the existing cross-attention in\n'
        'compute_layer_complete.',
        fontsize=7, color='#1B5E20', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#1B5E20'))

plt.tight_layout()
plt.savefig('b/d/asset/token_sequence_attention.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: b/d/asset/token_sequence_attention.png")
