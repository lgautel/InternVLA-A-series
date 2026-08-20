"""
Draw the fused InternVLA-A1.5 + GeoPredict 3D Keypoint Trajectory Predictor architecture.
Shows the token sequence, dual-model transformer, and all loss branches.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(18, 14))
ax.set_xlim(0, 18)
ax.set_ylim(0, 14)
ax.axis('off')
fig.patch.set_facecolor('white')

C_PREFIX = '#E8F5E9'
C_PREFIX_BORDER = '#2E7D32'
C_SUFFIX = '#FFF3E0'
C_SUFFIX_BORDER = '#E65100'
C_BACKBONE = '#E3F2FD'
C_BACKBONE_BORDER = '#1565C0'
C_LOSS = '#FCE4EC'
C_LOSS_BORDER = '#C62828'
C_NEW = '#C8E6C9'
C_NEW_BORDER = '#1B5E20'
C_FROZEN = '#ECEFF1'
C_FROZEN_BORDER = '#607D8B'

def draw_box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=8, bold=False, alpha=1.0):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color='#424242', style='->', lw=1.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

# === INPUT SECTION ===
ax.text(1.5, 13.5, 'INPUT', fontsize=11, fontweight='bold', color='#616161')
draw_box(ax, 0.3, 12.6, 2.0, 0.7, 'Camera Images\n(up to 3, 224x224)', '#F5F5F5', '#9E9E9E', 7)
draw_box(ax, 2.5, 12.6, 2.0, 0.7, 'Language +\nDiscretized State', '#F5F5F5', '#9E9E9E', 7)
draw_box(ax, 4.7, 12.6, 2.0, 0.7, '3D Keypoint\nHistory [T,8,3]', C_NEW, C_NEW_BORDER, 7, bold=True)
draw_box(ax, 7.7, 12.6, 1.5, 0.7, 'Robot State\n[32]', '#F5F5F5', '#9E9E9E', 7)
draw_box(ax, 9.4, 12.6, 1.5, 0.7, 'Noisy Actions\n[50,32]', '#F5F5F5', '#9E9E9E', 7)
draw_box(ax, 11.1, 12.6, 1.3, 0.7, 'FM Time\nt~Beta', '#F5F5F5', '#9E9E9E', 7)

# === PREFIX SECTION ===
ax.text(0.5, 11.9, 'PREFIX (VLM Backbone, dim=2048)', fontsize=10, fontweight='bold', color=C_PREFIX_BORDER)
box_pfx = FancyBboxPatch((0.2, 9.3), 7.0, 2.5, boxstyle="round,pad=0.1",
                          facecolor=C_PREFIX, edgecolor=C_PREFIX_BORDER, linewidth=2, alpha=0.3)
ax.add_patch(box_pfx)

draw_box(ax, 0.4, 10.5, 1.5, 0.9, 'Qwen3.5 Vision\nEncoder', '#C8E6C9', '#388E3C', 7)
draw_box(ax, 0.4, 9.5, 1.5, 0.7, 'Image Tokens\n(variable)', '#A5D6A7', '#388E3C', 7)
draw_arrow(ax, 1.15, 10.5, 1.15, 10.25)

draw_box(ax, 2.1, 9.5, 1.5, 0.7, 'Language Tokens\n(variable)', '#A5D6A7', '#388E3C', 7)

draw_box(ax, 3.8, 10.5, 1.5, 0.9, 'TrackEncoder\n(NEW)\nPatchEmb+CrossAttn', C_NEW, C_NEW_BORDER, 6.5, bold=True)
draw_box(ax, 3.8, 9.5, 1.5, 0.7, 'Hist KPT Tokens\n[8, 2048] (NEW)', C_NEW, C_NEW_BORDER, 6.5, bold=True)
draw_arrow(ax, 4.55, 10.5, 4.55, 10.25)

draw_box(ax, 5.5, 10.5, 1.5, 0.9, 'KPT Query Emb\n(NEW)\nnn.Emb(8, 2048)', C_NEW, C_NEW_BORDER, 6.5, bold=True)
draw_box(ax, 5.5, 9.5, 1.5, 0.7, 'Query KPT Tokens\n[8, 2048] (NEW)', C_NEW, C_NEW_BORDER, 6.5, bold=True)
draw_arrow(ax, 6.25, 10.5, 6.25, 10.25)

# Arrows from inputs to prefix
draw_arrow(ax, 1.3, 12.6, 1.15, 11.45)
draw_arrow(ax, 3.5, 12.6, 2.85, 10.25)
draw_arrow(ax, 5.7, 12.6, 4.55, 11.45)

# === SUFFIX SECTION ===
ax.text(8.2, 11.9, 'SUFFIX (Action Expert, dim=1024)', fontsize=10, fontweight='bold', color=C_SUFFIX_BORDER)
box_sfx = FancyBboxPatch((7.5, 9.3), 6.5, 2.5, boxstyle="round,pad=0.1",
                          facecolor=C_SUFFIX, edgecolor=C_SUFFIX_BORDER, linewidth=2, alpha=0.3)
ax.add_patch(box_sfx)

draw_box(ax, 7.7, 10.5, 1.3, 0.9, 'state_proj\nLinear(32,1024)', '#FFE0B2', '#E65100', 6.5)
draw_box(ax, 7.7, 9.5, 1.3, 0.7, 'State Token\n[1, 1024]', '#FFCC80', '#E65100', 6.5)
draw_arrow(ax, 8.35, 10.5, 8.35, 10.25)

draw_box(ax, 9.2, 10.5, 1.8, 0.9, 'Learnable Foresight\nnn.Param(50,1024)\n+ in_proj', '#FFE0B2', '#E65100', 6.5)
draw_box(ax, 9.2, 9.5, 1.8, 0.7, 'Foresight Tokens\n[50, 1024]', '#FFCC80', '#E65100', 6.5)
draw_arrow(ax, 10.1, 10.5, 10.1, 10.25)

draw_box(ax, 11.2, 10.5, 2.5, 0.9, 'action_in_proj +\ntime_mlp (SiLU)', '#FFE0B2', '#E65100', 6.5)
draw_box(ax, 11.2, 9.5, 2.5, 0.7, 'Action+Time Tokens\n[50, 1024]', '#FFCC80', '#E65100', 6.5)
draw_arrow(ax, 12.45, 10.5, 12.45, 10.25)

# Arrows from inputs to suffix
draw_arrow(ax, 8.45, 12.6, 8.35, 11.45)
draw_arrow(ax, 10.15, 12.6, 12.45, 11.45)
draw_arrow(ax, 11.75, 12.6, 12.45, 11.45)

# === BACKBONE ===
ax.text(4.5, 8.9, 'Joint Transformer (28 Layers: 18 Linear Attn + 6 Full Attn)',
        fontsize=10, fontweight='bold', color=C_BACKBONE_BORDER)
box_bb = FancyBboxPatch((0.2, 7.3), 17.0, 1.5, boxstyle="round,pad=0.1",
                         facecolor=C_BACKBONE, edgecolor=C_BACKBONE_BORDER, linewidth=2, alpha=0.4)
ax.add_patch(box_bb)

ax.text(4.0, 8.15, 'Mixture-of-Transformers (MoT)', fontsize=9, fontweight='bold', color=C_BACKBONE_BORDER)
ax.text(4.0, 7.7, 'Linear Attn: prefix/suffix run independently\n'
        'Full Attn: suffix Q attends [prefix K/V, suffix K/V];\n'
        '           prefix Q attends prefix K/V only',
        fontsize=7, color='#424242', va='top')

draw_box(ax, 11.5, 7.5, 2.5, 0.6, 'Knowledge Insulation\n(optional detach)', '#BBDEFB', C_BACKBONE_BORDER, 6.5)
draw_box(ax, 14.5, 7.5, 2.5, 0.6, 'prefix_out  suffix_out\n[B,L+16,2048] [B,101,1024]', '#BBDEFB', C_BACKBONE_BORDER, 6.5)

# Arrows into backbone
draw_arrow(ax, 3.5, 9.3, 3.5, 8.85)
draw_arrow(ax, 10.5, 9.3, 10.5, 8.85)

# === LOSS SECTION ===
ax.text(1.0, 6.7, 'LOSS BRANCHES (Training Only)', fontsize=10, fontweight='bold', color=C_LOSS_BORDER)

# L_vqa
draw_box(ax, 0.3, 5.0, 2.5, 1.4, 'L_vqa (weight: lambda)\n\nCross-Entropy via\nlm_head on prefix_out\n(subtask + FAST tokens)',
         C_LOSS, C_LOSS_BORDER, 6.5)
draw_arrow(ax, 1.55, 7.3, 1.55, 6.45)

# L_kpt (NEW)
draw_box(ax, 3.0, 5.0, 3.0, 1.4, 'L_kpt_current + L_kpt_future (NEW)\n(weight: beta = 1.0)\n\nMSE via keypoint_out_proj\non query KPT output tokens\n+ sinusoidal time PE for future',
         C_NEW, C_NEW_BORDER, 6.5, bold=True)
draw_arrow(ax, 4.5, 7.3, 4.5, 6.45)

# L_action
draw_box(ax, 6.2, 5.0, 2.8, 1.4, 'L_action (weight: 10)\n\nFlow Matching MSE via\naction_out_proj on\nsuffix_out[-50:]',
         C_LOSS, C_LOSS_BORDER, 6.5)
draw_arrow(ax, 7.6, 7.3, 7.6, 6.45)

# L_video
draw_box(ax, 9.2, 5.0, 3.5, 1.4, 'L_video (weight: alpha)\n\nVideo FM MSE via\nfrozen WAN2.2 DiT on\nsuffix_out foresight tokens',
         C_LOSS, C_LOSS_BORDER, 6.5)
draw_arrow(ax, 10.95, 7.3, 10.95, 6.45)

# WAN frozen
draw_box(ax, 13.0, 5.0, 2.5, 1.4, 'WAN2.2-5B DiT\n(FROZEN)\n\nProvides gradient via\ncross-attention only',
         C_FROZEN, C_FROZEN_BORDER, 6.5)
draw_arrow(ax, 12.7, 5.7, 13.0, 5.7, color=C_FROZEN_BORDER, style='->')

# === TOTAL LOSS ===
draw_box(ax, 3.5, 3.5, 9.5, 1.0, 'L_total = 10 * L_action + lambda_vqa * L_vqa + alpha * L_video + beta * (L_kpt_current + L_kpt_future)',
         '#FFF9C4', '#F9A825', 9, bold=True)
draw_arrow(ax, 1.55, 5.0, 5.0, 4.55)
draw_arrow(ax, 4.5, 5.0, 6.0, 4.55)
draw_arrow(ax, 7.6, 5.0, 8.0, 4.55)
draw_arrow(ax, 10.95, 5.0, 10.0, 4.55)

# === LEGEND ===
ax.text(14.5, 13.5, 'Legend', fontsize=9, fontweight='bold')
draw_box(ax, 14.3, 12.8, 1.2, 0.5, 'NEW module', C_NEW, C_NEW_BORDER, 7, bold=True)
draw_box(ax, 15.7, 12.8, 1.2, 0.5, 'Existing', '#A5D6A7', '#388E3C', 7)
draw_box(ax, 14.3, 12.2, 1.2, 0.5, 'Frozen', C_FROZEN, C_FROZEN_BORDER, 7)
draw_box(ax, 15.7, 12.2, 1.2, 0.5, 'Suffix', '#FFCC80', '#E65100', 7)

plt.tight_layout()
plt.savefig('b/d/asset/fusion_architecture.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: b/d/asset/fusion_architecture.png")
