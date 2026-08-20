"""
Draw the gradient flow diagram for the fused model.
Shows which losses send gradients to which modules, and the effect of knowledge insulation.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('white')

C_LOSS = '#EF9A9A'
C_HEAD = '#CE93D8'
C_BACKBONE = '#90CAF9'
C_MODULE_TRAIN = '#A5D6A7'
C_MODULE_FROZEN = '#ECEFF1'
C_NEW = '#81C784'
C_BORDER = '#424242'

def draw_box(ax, x, y, w, h, text, facecolor, fontsize=7.5, edgecolor='#424242', lw=1.5, bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=lw)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight)

def draw_grad_arrow(ax, x1, y1, x2, y2, color='#E53935', lw=1.5, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

# Title
ax.text(8, 9.7, 'Gradient Flow: Fused InternVLA-A1.5 + 3D Keypoint Trajectory Predictor',
        fontsize=12, fontweight='bold', ha='center', color='#212121')

# === ROW 1: LOSSES ===
ax.text(0.3, 9.1, 'Losses', fontsize=9, fontweight='bold', color='#B71C1C')
losses = [
    (0.5, 'L_action\n(w=10)', 2.0),
    (2.8, 'L_vqa\n(w=lambda)', 2.0),
    (5.1, 'L_video\n(w=alpha)', 2.0),
    (7.4, 'L_kpt_cur\n(w=beta)', 2.0),
    (9.7, 'L_kpt_fut\n(w=beta)', 2.0),
]
for x, text, w in losses:
    c = C_NEW if 'kpt' in text else C_LOSS
    draw_box(ax, x, 8.4, w, 0.6, text, c, bold='kpt' in text)

# === ROW 2: PROJECTION HEADS ===
ax.text(0.3, 7.7, 'Heads', fontsize=9, fontweight='bold', color='#4A148C')
heads = [
    (0.5, 'action_out_proj\nLinear(1024,32)\ntrainable', 2.0),
    (2.8, 'lm_head\n(Qwen3.5 vocab)\ntrainable', 2.0),
    (5.1, 'learnable_to_wan_proj\nLinear(1024,2048)\nconfigurable', 2.0),
    (7.4, 'keypoint_out_proj\nLinear(2048,3)\ntrainable (NEW)', 2.0),
]
for i, (x, text, w) in enumerate(heads):
    c = C_NEW if i == 3 else C_HEAD
    draw_box(ax, x, 6.8, w, 0.8, text, c, 6.5, bold=i==3)

# Arrows: losses -> heads
draw_grad_arrow(ax, 1.5, 8.4, 1.5, 7.65)
draw_grad_arrow(ax, 3.8, 8.4, 3.8, 7.65)
draw_grad_arrow(ax, 6.1, 8.4, 6.1, 7.65)
draw_grad_arrow(ax, 8.4, 8.4, 8.4, 7.65)
draw_grad_arrow(ax, 10.7, 8.4, 9.2, 7.65)  # L_kpt_fut shares kpt_out_proj

# === ROW 3: TRANSFORMER OUTPUTS ===
ax.text(0.3, 6.15, 'Transformer\nOutputs', fontsize=8, fontweight='bold', color='#0D47A1')
draw_box(ax, 2.0, 5.3, 3.5, 0.7, 'prefix_out\n[B, L+16, 2048]', C_BACKBONE, 7.5)
draw_box(ax, 6.5, 5.3, 3.5, 0.7, 'suffix_out\n[B, 101, 1024]', C_BACKBONE, 7.5)

# Knowledge insulation
draw_box(ax, 10.5, 5.3, 3.5, 0.7, 'Knowledge Insulation\nKI=False: grad flows\nKI=True: detach (no grad)', '#FFF9C4', 6.5)

# Arrows: heads -> outputs
draw_grad_arrow(ax, 1.5, 6.8, 7.5, 6.05)  # action_out_proj <- suffix_out
draw_grad_arrow(ax, 3.8, 6.8, 3.75, 6.05)  # lm_head <- prefix_out
draw_grad_arrow(ax, 6.1, 6.8, 7.0, 6.05)  # wan_proj <- suffix_out
draw_grad_arrow(ax, 8.4, 6.8, 4.5, 6.05)  # kpt_proj <- prefix_out (query kpt positions)

# suffix -> prefix (via cross-attention, KI dependent)
draw_grad_arrow(ax, 6.5, 5.65, 5.5, 5.65, color='#FF6F00', lw=2.0)
ax.text(5.7, 5.15, 'KI=False:\ngrad flows', fontsize=6, color='#FF6F00', ha='center')

# === ROW 4: INPUT MODULES ===
ax.text(0.3, 4.5, 'Input\nModules', fontsize=8, fontweight='bold', color='#1B5E20')

modules = [
    (0.3, 'Vision Encoder\nQwen3.5 ViT\nconfigurable', C_MODULE_TRAIN, False),
    (2.3, 'Text Embed\nQwen3.5\ntrainable', C_MODULE_TRAIN, False),
    (4.3, 'TrackEncoder\n(NEW)\ntrainable', C_NEW, True),
    (6.3, 'KPT Embedding\n(NEW)\ntrainable', C_NEW, True),
    (8.3, 'Action Expert\nalways trainable', C_MODULE_TRAIN, False),
    (10.3, 'Foresight Tok\nconfigurable', C_MODULE_TRAIN, False),
    (12.3, 'action_in/time\ntrainable', C_MODULE_TRAIN, False),
    (14.0, 'WAN DiT\nFROZEN', C_MODULE_FROZEN, False),
]

for x, text, color, bold in modules:
    draw_box(ax, x, 3.3, 1.8, 0.9, text, color, 6.5, bold=bold)

# Arrows: outputs -> modules
draw_grad_arrow(ax, 2.5, 5.3, 1.2, 4.25)
draw_grad_arrow(ax, 3.0, 5.3, 3.2, 4.25)
draw_grad_arrow(ax, 3.75, 5.3, 5.2, 4.25, color='#1B5E20', lw=2.0)
draw_grad_arrow(ax, 4.5, 5.3, 7.2, 4.25, color='#1B5E20', lw=2.0)
draw_grad_arrow(ax, 7.5, 5.3, 9.2, 4.25)
draw_grad_arrow(ax, 8.0, 5.3, 11.2, 4.25)
draw_grad_arrow(ax, 8.5, 5.3, 13.2, 4.25)
draw_grad_arrow(ax, 6.1, 6.8, 14.9, 4.25, color='#607D8B', lw=1.0)

# === LEGEND ===
ax.text(12.5, 9.1, 'Legend', fontsize=9, fontweight='bold')
draw_box(ax, 12.3, 8.4, 1.5, 0.5, 'NEW module', C_NEW, 7, bold=True)
draw_box(ax, 14.0, 8.4, 1.5, 0.5, 'Frozen', C_MODULE_FROZEN, 7)
ax.annotate('', xy=(14.5, 7.5), xytext=(14.5, 7.9),
            arrowprops=dict(arrowstyle='->', color='#E53935', lw=2))
ax.text(14.8, 7.6, 'Gradient', fontsize=7, color='#E53935')
ax.annotate('', xy=(14.5, 7.0), xytext=(14.5, 7.4),
            arrowprops=dict(arrowstyle='->', color='#FF6F00', lw=2))
ax.text(14.8, 7.1, 'KI-dependent', fontsize=7, color='#FF6F00')

# === SUMMARY TABLE ===
ax.text(0.5, 2.5, 'Gradient Path Summary:', fontsize=9, fontweight='bold', color='#212121')
summary = [
    'L_kpt -> keypoint_out_proj -> prefix_out (query KPT) -> all 28 VLM layers -> TrackEncoder, KPT Emb, Vision, Text',
    'L_action -> action_out_proj -> suffix_out -> Expert layers -> (if KI=False) cross-attn -> prefix_out -> all prefix modules',
    'L_vqa -> lm_head -> prefix_out (lang positions) -> all 28 VLM layers -> Vision, Text (NOT keypoint-specific)',
    'L_video -> wan_proj -> suffix_out (foresight) -> Expert -> (if KI=False) -> prefix -> all prefix modules; WAN DiT is frozen',
]
for i, line in enumerate(summary):
    color = '#1B5E20' if i == 0 else '#424242'
    weight = 'bold' if i == 0 else 'normal'
    ax.text(0.5, 2.1 - i * 0.4, line, fontsize=6.5, color=color, fontweight=weight)

plt.tight_layout()
plt.savefig('b/d/asset/gradient_flow.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: b/d/asset/gradient_flow.png")
