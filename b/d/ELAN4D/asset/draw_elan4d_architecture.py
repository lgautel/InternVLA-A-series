"""
Generate the ELAN4D architecture overview diagram showing the data flow
during training and the components that are discarded at inference.
Output: elan4d_architecture.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={'width_ratios': [3, 2]})

ax = axes[0]
ax.set_xlim(-1, 16)
ax.set_ylim(-1, 11)
ax.axis('off')
ax.set_title('Training Phase', fontsize=14, fontweight='bold', color='#2c3e50', pad=10)

def draw_box(ax, xy, w, h, label, color, text_color='white', fontsize=9, alpha=1.0):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor='#2c3e50',
                         linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    ax.text(xy[0] + w/2, xy[1] + h/2, label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=text_color, alpha=alpha)

def arrow(ax, s, e, c='#34495e', lw=2):
    a = FancyArrowPatch(s, e, arrowstyle='->', color=c, linewidth=lw,
                        mutation_scale=15, connectionstyle='arc3,rad=0.0')
    ax.add_patch(a)

draw_box(ax, (0, 8.5), 3.5, 1.5, 'Images $I_t$\n+ Language $L$', '#3498db')
draw_box(ax, (5, 8.5), 3.5, 1.5, 'VLM Backbone\n(PaliGemma)\nFrozen/LoRA', '#95a5a6')
draw_box(ax, (0, 6), 3.5, 1.5, 'Proprioception\n$q_t$, Actions,\nFlow time $t$', '#2ecc71')
draw_box(ax, (5, 6), 3.5, 1.5, 'Action Expert\n(Flow Matching)', '#2980b9')
draw_box(ax, (10.5, 8.5), 4, 1.5, 'Action Decoder\n$\\hat{A}_t \\in \\mathbb{R}^{H \\times 7}$', '#2980b9')

draw_box(ax, (5, 3), 3.5, 1.5, 'Control Branch\n$b_\\psi(\\mathrm{sg}(u_t))$', '#8e44ad')
draw_box(ax, (10.5, 6), 4, 1.5, 'Zero-Init Proj\n$\\oplus$ Residual Fusion\n$\\tilde{u}_t = u_t + \\mathrm{Proj}(C_t)$', '#8e44ad')

draw_box(ax, (0, 0.5), 3.5, 1.5, 'Current Keypoints\n$P_t = \\mathrm{FK}(q_t)$\n$\\in \\mathbb{R}^{K \\times 3}$', '#27ae60')
draw_box(ax, (5, 0.5), 3.5, 1.5, 'Track Decoder\n(Point MLP +\nControl MLP +\nFusion MLP)', '#e67e22')
draw_box(ax, (10.5, 0.5), 4, 1.5, 'Predicted Tracks\n$\\hat{Y}_t \\in \\mathbb{R}^{H \\times K \\times 3}$', '#e67e22')

draw_box(ax, (10.5, 3), 4, 1.5, '$\\mathcal{L} = \\mathcal{L}_{act} + 0.1 \\cdot \\mathcal{L}_{track}$',
         '#e74c3c', fontsize=10)

arrow(ax, (3.5, 9.25), (5, 9.25))
arrow(ax, (8.5, 9.25), (10.5, 9.25))
arrow(ax, (3.5, 6.75), (5, 6.75))
arrow(ax, (8.5, 6.75), (10.5, 6.75))
arrow(ax, (6.75, 6), (6.75, 4.5), c='#8e44ad')
ax.text(7.1, 5.3, 'sg($u_t$)', fontsize=8, color='#e74c3c', fontweight='bold')
arrow(ax, (8.5, 3.75), (10.5, 6.75), c='#8e44ad')
ax.text(9.7, 5.3, '$C_t$', fontsize=9, color='#8e44ad')
arrow(ax, (8.5, 1.25), (10.5, 1.25), c='#e67e22')
arrow(ax, (3.5, 1.25), (5, 1.25), c='#27ae60')
arrow(ax, (6.75, 3), (6.75, 2), c='#8e44ad')
ax.text(7.1, 2.5, '$C_t$', fontsize=9, color='#8e44ad')
arrow(ax, (12.5, 8.5), (12.5, 4.5), c='#27ae60')
ax.text(13.0, 5.5, '$\\mathcal{L}_{act}$', fontsize=10, color='#27ae60')
arrow(ax, (12.5, 2), (12.5, 3), c='#e74c3c')
ax.text(13.0, 2.5, '$\\mathcal{L}_{track}$', fontsize=10, color='#e74c3c')

ax.add_patch(plt.Rectangle((4.2, -0.2), 11.2, 3.3,
             fill=False, edgecolor='#e67e22', linewidth=2, linestyle='--'))
ax.text(9.8, -0.5, 'Discarded at Inference', fontsize=10,
        fontweight='bold', color='#e67e22', ha='center', style='italic')

ax2 = axes[1]
ax2.set_xlim(-1, 11)
ax2.set_ylim(-1, 11)
ax2.axis('off')
ax2.set_title('Inference Phase', fontsize=14, fontweight='bold', color='#2c3e50', pad=10)

draw_box(ax2, (0, 8), 4, 1.5, 'Images $I_t$\n+ Language $L$', '#3498db')
draw_box(ax2, (0, 5.5), 4, 1.5, 'Proprioception\n$q_t$', '#2ecc71')
draw_box(ax2, (5, 8), 4.5, 1.5, 'VLM Backbone\n(PaliGemma)', '#95a5a6')
draw_box(ax2, (5, 5.5), 4.5, 1.5, 'Action Expert\n+ Residual from\nControl Branch', '#2980b9')
draw_box(ax2, (5, 3), 4.5, 1.5, 'Action Decoder', '#2980b9')
draw_box(ax2, (3, 0.5), 5, 1.5, 'Action Chunk\n$A_t \\in \\mathbb{R}^{H \\times 7}$', '#1a5276', fontsize=11)

a = FancyArrowPatch((4, 8.75), (5, 8.75), arrowstyle='->', color='#34495e',
                     linewidth=2, mutation_scale=15)
ax2.add_patch(a)
a = FancyArrowPatch((4, 6.25), (5, 6.25), arrowstyle='->', color='#34495e',
                     linewidth=2, mutation_scale=15)
ax2.add_patch(a)
a = FancyArrowPatch((9.5, 8.75), (9.8, 8.75), arrowstyle='->', color='#34495e',
                     linewidth=2, mutation_scale=15)
ax2.add_patch(a)
ax2.annotate('', xy=(7.25, 5.5), xytext=(7.25, 8),
             arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
ax2.annotate('', xy=(7.25, 3), xytext=(7.25, 5.5),
             arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))
ax2.annotate('', xy=(5.5, 2), xytext=(7.25, 3),
             arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))

ax2.text(5.5, 10, 'No Track Decoder\nNo FK input needed\nSame interface as base VLA',
         fontsize=10, color='#27ae60', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#d5f5e3', edgecolor='#27ae60'))

plt.tight_layout()
plt.savefig('d:/SRC/Robot/InternVLA-A-series/b/d/ELAN4D/asset/elan4d_architecture.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved elan4d_architecture.png")
