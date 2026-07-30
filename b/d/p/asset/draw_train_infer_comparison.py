"""Draw a training-vs-inference data-flow / latency comparison for InternVLA-A1.5.

Output: train_infer_comparison.png (English labels only).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig = plt.figure(figsize=(13, 7.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.06)
ax_train = fig.add_subplot(gs[0, 0])
ax_infer = fig.add_subplot(gs[0, 1])

for ax in (ax_train, ax_infer):
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 10)
    ax.axis("off")


def box(ax, x, y, w, h, text, fc="#eef2f7", ec="#333333", fontsize=9.3, lw=1.3):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.08",
                        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             zorder=3, linespacing=1.4)


def down_arrow(ax, x, y1, y2, color="#444444"):
    a = FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", color=color,
                         linewidth=1.6, mutation_scale=13, zorder=1)
    ax.add_patch(a)


# ============ Left: training-time pipeline ============
ax_train.set_title("Training-time pipeline", fontsize=13, fontweight="bold", pad=10)
steps_train = [
    ("Batch: images + language\n+ state + GT action chunk\n+ GT future frames + VQA labels", "#fdf3e7"),
    ("embed_prefix: image tower +\ntext embedding -> $H_t$\n(causal attention, Qwen3.5)", "#e3edfb"),
    ("embed_suffix: state + foresight\ntokens $Q^f$ + noisy action $a^\\tau$\n(group-causal / bidirectional)", "#e6f4ea"),
    ("Joint MoT forward (shared full-\nattention + separate linear attn)\n$\\rightarrow$ prefix logits, suffix hidden", "#e6f4ea"),
    ("3 parallel loss heads:\nlm_head CE  |  action_out_proj MSE\n|  WAN DiT MSE (video)", "#f7e6f0"),
    ("Backward pass: gradient flows\ninto VLM + expert + foresight proj.\n(WAN DiT / VAE stay frozen)", "#f0e3f7"),
    ("AdamW step (bf16), grad clip 1.0\n300K (stage1) + 600K (stage2)\n+ 60K (post-train) steps", "#eef2f7"),
]
y = 9.3
for text, fc in steps_train:
    box(ax_train, 0.3, y - 0.9, 5.4, 0.95, text, fc=fc)
    y -= 1.28
    if y > -0.3:
        down_arrow(ax_train, 3.0, y + 0.28, y + 0.05)

ax_train.text(3.0, -0.15, "Latency: not critical (throughput-oriented, multi-GPU)",
              ha="center", fontsize=8.8, style="italic", color="#555555")

# ============ Right: inference-time pipeline ============
ax_infer.set_title("Inference-time pipeline (deployed robot)", fontsize=13, fontweight="bold", pad=10)
steps_infer = [
    ("Single-step observation:\nimages + language + state\n(no GT action / no video)", "#fdf3e7"),
    ("embed_prefix + Qwen3.5 causal\nforward $\\rightarrow$ KV-cache\n(computed once per chunk)", "#e3edfb"),
    ("Sample noise $a^0\\sim\\mathcal{N}(0,I)$\nEuler loop: $K{=}10$ denoise steps\nreusing cached VLM KV", "#e6f4ea"),
    ("Each step: unified expert only\n(SDPA / CUDA-graph replay in\n'optimized' backend)", "#e6f4ea"),
    ("action_out_proj $\\rightarrow$ velocity\n$a^{\\tau+\\Delta\\tau}=a^\\tau+\\Delta\\tau\\,v_\\theta^{act}$", "#f9e0e0"),
    ("Output: action chunk\n(first $n\\_action\\_steps$ executed,\nrest cached in queue)", "#eef2f7"),
    ("WAN DiT / VAE: NOT loaded\n(action_loss_only=True)\n$\\approx$0.1s / step end-to-end", "#f0e3f7"),
]
y = 9.3
for text, fc in steps_infer:
    box(ax_infer, 0.3, y - 0.9, 5.4, 0.95, text, fc=fc)
    y -= 1.28
    if y > -0.3:
        down_arrow(ax_infer, 3.0, y + 0.28, y + 0.05)

ax_infer.text(3.0, -0.15, "Latency-critical: real-time closed-loop control (single RTX 5090)",
              ha="center", fontsize=8.8, style="italic", color="#555555")

fig.suptitle("InternVLA-A1.5: Training vs. Inference Data Flow", fontsize=15, fontweight="bold", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = __file__.rsplit("/", 1)[0] + "/train_infer_comparison.png"
plt.savefig(out_path, dpi=170, bbox_inches="tight")
print("saved", out_path)
