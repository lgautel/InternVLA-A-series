"""Draw a self-contained architecture schematic for InternVLA-A1.5.

Output: architecture_overview.png (English labels only, per repo convention).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13, 8.3))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.axis("off")


def box(x, y, w, h, text, fc="#eef2f7", ec="#333333", fontsize=10, lw=1.4):
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.08",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, zorder=3, linespacing=1.5)
    return b


def arrow(x1, y1, x2, y2, color="#444444", lw=1.6, connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color=color,
                         linewidth=lw, connectionstyle=connectionstyle,
                         zorder=1, mutation_scale=14)
    ax.add_patch(a)


fig.suptitle("InternVLA-A1.5 Architecture (training-time full computation graph)",
             fontsize=14, fontweight="bold", y=0.985)

# ---------------- Inputs (col 1) ----------------
box(0.2, 6.9, 2.1, 0.8, "Multi-view images\n$o_t^{(k)},\\,k=1..K$", fc="#fdf3e7", fontsize=9.3)
box(0.2, 5.9, 2.1, 0.8, "Language instruction $\\ell$\n+ control mode $m$", fc="#fdf3e7", fontsize=9.3)
box(0.2, 4.9, 2.1, 0.8, "Proprioceptive state $q_t$\n(256-bin discretized)", fc="#fdf3e7", fontsize=9.3)

# ---------------- VLM backbone (col 2) ----------------
box(2.7, 4.5, 3.3, 3.2,
    "Qwen3.5-2B VLM backbone\n\n3x Gated DeltaNet\n+ 1x full attention\n(repeated x N blocks)\n\nhidden states $H_t$\n\nlm_head -> subtask text /\nFAST action tokens",
    fc="#e3edfb", fontsize=9.6)

arrow(2.3, 7.3, 2.7, 6.6)
arrow(2.3, 6.3, 2.7, 6.3)
arrow(2.3, 5.3, 2.7, 5.9)

# ---------------- Unified expert (col 3) ----------------
box(6.4, 4.5, 3.3, 3.2,
    "Unified expert (460M params)\n\nshares full-attention layers\nwith VLM; separate\nGated DeltaNet layers\n\nForesight tokens $Q^f$ ($M{=}50$)\nNoisy action tokens $a^\\tau$\n(chunk $H{=}50$)",
    fc="#e6f4ea", fontsize=9.6)
arrow(6.0, 6.1, 6.4, 6.1, connectionstyle="arc3,rad=0.2")
ax.text(6.2, 6.42, "shared full\nattention", fontsize=7.3, ha="center", color="#555555")

# ---------------- L_stage1 under VLM ----------------
box(2.7, 2.95, 3.3, 0.85, "next-token cross-entropy\n$\\mathcal{L}_{stage1}$ (subtask + FAST)",
    fc="#f6e9f9", fontsize=9.3)
arrow(4.35, 4.5, 4.35, 3.8)

# ---------------- Expert -> WAN branch (col 3a) ----------------
box(6.4, 2.95, 1.55, 0.85, "$C_t^f{=}P_{WAN}(Z_t^f)$\nforesight embedding", fc="#fff2cc", fontsize=8.4)
arrow(7.1, 4.5, 7.1, 3.8)
box(6.4, 1.35, 1.55, 1.3, "Frozen WAN2.2-5B\nDiT + VAE\n(cross-attn cond.)\n\n$\\rightarrow \\mathcal{L}_{video}$\n(flow-match MSE)",
    fc="#f0e3f7", fontsize=8.2)
arrow(7.1, 2.95, 7.1, 2.65)

# ---------------- Expert -> action branch (col 3b) ----------------
box(8.15, 2.95, 1.55, 0.85, "action_out_proj\n$v_\\theta^{act}$ velocity", fc="#f9e0e0", fontsize=8.4)
arrow(9.05, 4.5, 9.05, 3.8)
box(8.15, 1.35, 1.55, 1.3, "flow-matching loss\n\n$\\mathcal{L}_{action}$\n(MSE vs.\n$\\epsilon{-}a_{t:t+H}$)",
    fc="#f9e0e0", fontsize=8.4)
arrow(9.05, 2.95, 9.05, 2.65)

# ---------------- Total loss ----------------
box(3.4, 0.2, 5.7, 0.75,
    "$\\mathcal{L}_{total} = \\mathcal{L}_{stage1} + \\alpha\\,\\mathcal{L}_{video} + \\beta\\,\\mathcal{L}_{action}$\n($\\alpha{=}1,\\ \\beta{=}10$)",
    fc="#ffffff", ec="#111111", fontsize=10.3)
arrow(4.35, 2.95, 5.3, 0.95, connectionstyle="arc3,rad=0.25")
arrow(7.1, 1.35, 6.6, 0.95, connectionstyle="arc3,rad=-0.2")
arrow(9.05, 1.35, 7.9, 0.95, connectionstyle="arc3,rad=0.25")

# ---------------- Side panel: inference-time differences ----------------
box(10.1, 0.2, 2.7, 7.1, "", fc="#fbfbfb", ec="#999999", lw=1.2)
ax.text(11.45, 7.0, "Inference-time only", ha="center", fontsize=10.8, fontweight="bold")
ax.text(11.45, 6.3, "Active:\nVLM prefix (KV-cached)\n+ unified expert\n(flow-matching denoise,\nK=10 Euler steps)", ha="center", fontsize=8.6)
ax.text(11.45, 4.85, "Discarded:", ha="center", fontsize=9.6, fontweight="bold", color="#8a2020")
ax.text(11.45, 4.25, "- WAN DiT / VAE\n- video loss branch\n- FAST autoregressive\n  decoding path", ha="center", fontsize=8.4, color="#8a2020")
ax.text(11.45, 2.7, "Output:\naction chunk\n$a_{t:t+H}\\in\\mathbb{R}^{H\\times D}$,\n$H{=}50,\\,D{\\leq}32$", ha="center", fontsize=8.7)
ax.text(11.45, 0.9, "config switch:\ninference_backend=\n'optimized'\naction_loss_only=True", ha="center", fontsize=8.2, color="#444444", style="italic")

plt.tight_layout(rect=[0, 0, 1, 0.965])
out_path = __file__.rsplit("/", 1)[0] + "/architecture_overview.png"
plt.savefig(out_path, dpi=170, bbox_inches="tight")
print("saved", out_path)
