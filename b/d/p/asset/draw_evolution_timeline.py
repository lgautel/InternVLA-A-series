"""Draw an evolution timeline of VLA / world-model action policies leading to InternVLA-A1.5.

Output: evolution_timeline.png (English labels only).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13.5, 6.6))
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 6.6)
ax.axis("off")

ax.text(6.75, 6.3, "Evolution of Action Representations in VLA / World-Action Models",
        ha="center", fontsize=14, fontweight="bold")

# Timeline axis
ax.add_patch(FancyArrowPatch((0.4, 3.3), (13.1, 3.3), arrowstyle="-|>",
                              color="#666666", linewidth=1.8, mutation_scale=16))

entries = [
    (1.1, "2023\nRT-2", "Discrete\naction tokens\n(VLM co-fine-tune)", "up"),
    (3.0, "2024\nOpenVLA", "Discrete tokens,\nopen VLM backbone\n(Llama2 + DINOv2/SigLIP)", "down"),
    (4.9, "2024-25\n$\\pi_0$", "Flow-matching\ncontinuous action\nexpert (PaliGemma)", "up"),
    (6.7, "2025\n$\\pi_{0.5}$ / FAST", "Discrete pretrain +\ncontinuous flow expert,\nknowledge insulation", "down"),
    (8.5, "2025\nUniVLA / WorldVLA", "Explicit pixel-level\nfuture-frame prediction\n(shared AR token space)", "up"),
    (10.3, "2026\nInternVLA-A1", "Joint future-frame +\naction targets in one\nunified architecture", "down"),
    (12.4, "2026\nInternVLA-A1.5", "Latent foresight tokens\nquery frozen WAN2.2;\nvideo branch train-only", "up"),
]

for x, title, desc, direction in entries:
    ax.add_patch(plt.Circle((x, 3.3), 0.07, color="#333333", zorder=4))
    if direction == "up":
        box_y0, box_y1 = 3.55, 4.55
        text_y = 4.05
        ax.add_patch(FancyArrowPatch((x, 3.37), (x, 3.5), arrowstyle="-", color="#888888", linewidth=1.2))
    else:
        box_y0, box_y1 = 2.05, 3.05
        text_y = 2.55
        ax.add_patch(FancyArrowPatch((x, 3.23), (x, 3.1), arrowstyle="-", color="#888888", linewidth=1.2))

    fc = "#e3edfb" if direction == "up" else "#e6f4ea"
    if "InternVLA-A1.5" in title:
        fc = "#fff2cc"
    b = FancyBboxPatch((x - 0.95, box_y0), 1.9, box_y1 - box_y0,
                        boxstyle="round,pad=0.05,rounding_size=0.07",
                        linewidth=1.3, edgecolor="#333333", facecolor=fc, zorder=3)
    ax.add_patch(b)
    ax.text(x, box_y1 - 0.22, title, ha="center", va="top", fontsize=9.3, fontweight="bold", zorder=4)
    ax.text(x, text_y - 0.28, desc, ha="center", va="top", fontsize=7.6, zorder=4, linespacing=1.35)

# Axis annotation: action representation dimension
ax.text(0.4, 1.35,
        "Action head:  discrete tokens (AR decoding)  "
        "$\\longrightarrow$  continuous flow-matching expert  "
        "$\\longrightarrow$  + latent world-model supervision",
        ha="left", fontsize=9.6, color="#444444")
ax.text(0.4, 0.85,
        "World-model coupling:  none  $\\longrightarrow$  explicit pixel-level video prediction  "
        "$\\longrightarrow$  compact latent foresight tokens distilled from a frozen pretrained video generator",
        ha="left", fontsize=9.6, color="#444444")

plt.tight_layout()
out_path = __file__.rsplit("/", 1)[0] + "/evolution_timeline.png"
plt.savefig(out_path, dpi=170, bbox_inches="tight")
print("saved", out_path)
