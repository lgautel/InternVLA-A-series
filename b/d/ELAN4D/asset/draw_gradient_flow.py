"""
Generate a gradient flow diagram for ELAN4D showing how L_act and L_track
gradients flow through the architecture.
Output: gradient_flow.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(-0.5, 15.5)
ax.set_ylim(-1, 10)
ax.axis('off')

def draw_box(ax, xy, w, h, label, color, text_color='white', fontsize=9):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor='#2c3e50', linewidth=1.5)
    ax.add_patch(box)
    ax.text(xy[0] + w/2, xy[1] + h/2, label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=text_color, wrap=True)

def draw_arrow(ax, start, end, color='#2c3e50', style='->', lw=2, ls='-'):
    arrow = FancyArrowPatch(start, end, arrowstyle=style,
                            color=color, linewidth=lw, linestyle=ls,
                            connectionstyle='arc3,rad=0.0',
                            mutation_scale=15)
    ax.add_patch(arrow)

ax.text(8, 9.5, 'ELAN4D Gradient Flow Analysis', fontsize=16,
        fontweight='bold', ha='center', va='center', color='#2c3e50')

draw_box(ax, (0, 6.5), 3, 1.5, 'VLM Backbone\n(PaliGemma)\n[PROTECTED]', '#95a5a6')
draw_box(ax, (4.5, 6.5), 3, 1.5, 'Action Expert\nMain Pathway', '#2980b9')
draw_box(ax, (4.5, 4), 3, 1.5, 'Control Branch\n$b_\\psi$', '#8e44ad')
draw_box(ax, (9, 6.5), 2.5, 1.5, 'Zero-Init\nProj', '#8e44ad')
draw_box(ax, (12.5, 6.5), 2.5, 1.5, 'Action\nDecoder', '#2980b9')
draw_box(ax, (9, 4), 3, 1.5, 'Track\nDecoder', '#e67e22')
draw_box(ax, (0, 4), 3, 1.5, 'Forward\nKinematics\nFK($q_t$) = $P_t$', '#27ae60')
draw_box(ax, (13, 2), 2, 1, '$\\mathcal{L}_{act}$', '#27ae60', fontsize=11)
draw_box(ax, (9, 1.5), 2.5, 1, '$\\mathcal{L}_{track}$', '#e74c3c', fontsize=11)

draw_arrow(ax, (3, 7.25), (4.5, 7.25), color='#2c3e50')
ax.text(3.75, 7.6, 'prefix\ntokens', fontsize=7, ha='center', color='#2c3e50')

draw_arrow(ax, (6, 6.5), (6, 5.5), color='#2c3e50')
ax.text(6.35, 6.0, '$u_t$', fontsize=9, ha='left', color='#2c3e50')

sg_box = FancyBboxPatch((5.2, 5.6), 1.6, 0.6, boxstyle="round,pad=0.1",
                         facecolor='#e74c3c', edgecolor='#c0392b', linewidth=2)
ax.add_patch(sg_box)
ax.text(6.0, 5.9, 'sg( )', fontsize=9, fontweight='bold',
        ha='center', va='center', color='white')

draw_arrow(ax, (7.5, 4.75), (9, 4.75), color='#2c3e50')
ax.text(8.25, 5.0, '$C_t$', fontsize=9, ha='center', color='#2c3e50')

draw_arrow(ax, (7.5, 4.3), (9, 7.0), color='#8e44ad')
ax.text(8.0, 5.8, '$C_t$', fontsize=8, ha='center', color='#8e44ad')

draw_arrow(ax, (11.5, 7.25), (12.5, 7.25), color='#2c3e50')
ax.text(12.0, 7.6, '$\\tilde{u}_t$', fontsize=9, ha='center', color='#2c3e50')

draw_arrow(ax, (7.5, 7.25), (9, 7.25), color='#2c3e50')
ax.text(8.25, 7.6, '$u_t$', fontsize=9, ha='center', color='#2c3e50')

draw_arrow(ax, (3, 4.75), (9, 4.75), color='#27ae60')
ax.text(6.0, 4.3, '$P_t$', fontsize=9, ha='center', color='#27ae60')

draw_arrow(ax, (14, 6.5), (14, 3), color='#27ae60', lw=2.5)
draw_arrow(ax, (10.5, 4), (10.5, 2.5), color='#e74c3c', lw=2.5)

draw_arrow(ax, (13, 2.5), (12.5, 4), color='#27ae60', lw=2, ls='--', style='->')
ax.text(12.2, 3.2, '$\\partial\\mathcal{L}_{act}$', fontsize=8, color='#27ae60')

draw_arrow(ax, (13.5, 3), (8, 5), color='#27ae60', lw=2, ls='--', style='->')
ax.text(10.5, 3.8, '$\\partial\\mathcal{L}_{act}/\\partial\\psi$', fontsize=8, color='#27ae60')

draw_arrow(ax, (9.5, 2.5), (6, 4), color='#e74c3c', lw=2, ls='--', style='->')
ax.text(7.5, 2.8, '$\\partial\\mathcal{L}_{track}/\\partial\\psi$', fontsize=8, color='#e74c3c')

ax.annotate('BLOCKED\nby sg()',
            xy=(5.8, 5.6), xytext=(3.5, 2.5),
            fontsize=9, fontweight='bold', color='#e74c3c',
            ha='center',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2, ls='--'))
ax.text(3.5, 1.8, '$\\mathcal{L}_{track}$ cannot\nreach VLM or\nAction Expert',
        fontsize=8, ha='center', color='#e74c3c', style='italic')

legend_items = [
    (mpatches.Patch(facecolor='#2980b9'), 'Action Pathway'),
    (mpatches.Patch(facecolor='#8e44ad'), 'Control Branch'),
    (mpatches.Patch(facecolor='#e67e22'), 'Track Decoder (train only)'),
    (mpatches.Patch(facecolor='#95a5a6'), 'VLM (protected)'),
    (mpatches.Patch(facecolor='#27ae60'), '$\\mathcal{L}_{act}$ gradient'),
    (mpatches.Patch(facecolor='#e74c3c'), '$\\mathcal{L}_{track}$ gradient'),
]
ax.legend(handles=[item[0] for item in legend_items],
          labels=[item[1] for item in legend_items],
          loc='lower right', fontsize=8, ncol=2,
          frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('d:/SRC/Robot/InternVLA-A-series/b/d/ELAN4D/asset/gradient_flow.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved gradient_flow.png")
