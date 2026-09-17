#!/usr/bin/env python3
"""Figure for eval3_optim3.md section 12: OSMesa vs EGL render backend.

Reads the acceptance JSON produced by the O4/O5 gates and plots three panels:

  (a) per-step latency, the quantity that sets episode wall-clock
  (b) llvmpipe thread scaling, justifying the 4-threads-per-process default
  (c) backend parity vs the 180-degree control, on a log scale

Inputs (same directory):
    osmesa_parity.json   evaluation/LIBERO2/test_backend_parity.py --json_out
    osmesa_soak.json     evaluation/LIBERO2/test_osmesa_soak.py --json_out
    egl_soak.json        ditto, --render_backend egl

Output:
    eval3_optim3_osmesa.png

Usage:
    python b/d/libplus/asset/eval3_optim3_osmesa.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "eval3_optim3_osmesa.png"

EGL_COLOR = "#2E7D32"
OSMESA_COLOR = "#C62828"

# Measured with evaluation/LIBERO2/test_backend_parity.py, LP_NUM_THREADS swept
# by hand on the 64-CPU-quota container (2026-09-16). ms per env.step().
THREAD_SCALING = {1: 212.7, 2: 154.2, 4: 131.2, 8: 126.6, 16: 124.5}
DEFAULT_THREADS = 4
EPISODE_STEPS = 220  # libero_spatial max_steps


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def panel_latency(ax, parity: dict) -> None:
    egl_ms = parity["backends"]["egl"]["meta"]["seconds_per_step"] * 1000
    osm_ms = parity["backends"]["osmesa"]["meta"]["seconds_per_step"] * 1000
    bars = ax.bar(
        ["EGL\n(GPU)", "OSMesa\n(CPU llvmpipe)"],
        [egl_ms, osm_ms],
        color=[EGL_COLOR, OSMESA_COLOR],
        width=0.55,
    )
    for bar, val in zip(bars, [egl_ms, osm_ms]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 4,
            f"{val:.1f} ms",
            ha="center",
            fontweight="bold",
        )
    ax.set_ylabel("ms per env.step()  (physics + 2 cameras)")
    ax.set_ylim(0, osm_ms * 1.28)
    ax.set_title(
        f"(a) Per-step latency\nOSMesa {osm_ms / egl_ms:.1f}x slower; "
        f"{EPISODE_STEPS}-step episode: {egl_ms * EPISODE_STEPS / 1000:.0f}s -> "
        f"{osm_ms * EPISODE_STEPS / 1000:.0f}s",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)


def panel_threads(ax) -> None:
    threads = sorted(THREAD_SCALING)
    values = [THREAD_SCALING[t] for t in threads]
    ax.plot(threads, values, "o-", color=OSMESA_COLOR, linewidth=2, markersize=7)
    ax.axvline(DEFAULT_THREADS, color="#555", linestyle="--", linewidth=1.2)
    ax.annotate(
        f"default\nLP_NUM_THREADS={DEFAULT_THREADS}",
        xy=(DEFAULT_THREADS, THREAD_SCALING[DEFAULT_THREADS]),
        xytext=(5.2, 185),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#555"),
    )
    gain = 100 * (1 - THREAD_SCALING[16] / THREAD_SCALING[DEFAULT_THREADS])
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("LP_NUM_THREADS (llvmpipe rasteriser threads)")
    ax.set_ylabel("ms per env.step()")
    ax.set_title(
        f"(b) Thread scaling saturates at ~4\n4 -> 16 threads buys only {gain:.0f}%; "
        "spend CPUs on more workers",
        fontsize=10,
    )
    ax.grid(alpha=0.3)


def panel_parity(ax, parity: dict) -> None:
    cams = ["agentview", "wrist"]
    same = [parity["cameras"][c]["parity_mse"] for c in cams]
    rot = [parity["cameras"][c]["parity_mse_rot180"] for c in cams]
    x = range(len(cams))
    w = 0.35
    ax.bar([i - w / 2 for i in x], same, w, label="OSMesa vs EGL (same orientation)",
           color="#1565C0")
    ax.bar([i + w / 2 for i in x], rot, w, label="OSMesa rot180 vs EGL (control)",
           color="#9E9E9E")
    for i, (s, r) in enumerate(zip(same, rot)):
        ax.text(i - w / 2, s * 1.35, f"{s:.1f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + w / 2, r * 1.3, f"{r:.0f}", ha="center", fontsize=9)
        ax.text(i, r * 4.0, f"{r / s:.0f}x apart", ha="center", fontsize=10, color="#1565C0",
                fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cams)
    ax.set_ylabel("pixel MSE (log scale)")
    ax.set_ylim(1, max(rot) * 18)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    r_vals = ", ".join(f"{c}: r={parity['cameras'][c]['parity_r']:.4f}" for c in cams)
    ax.set_title(f"(c) Backend parity\n{r_vals}", fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    parity = load("osmesa_parity.json")
    osm_soak = load("osmesa_soak.json")
    egl_soak = load("egl_soak.json")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    panel_latency(axes[0], parity)
    panel_threads(axes[1])
    panel_parity(axes[2], parity)

    fig.suptitle(
        "LIBERO eval render backend: OSMesa (CPU) as the EGL-SIGABRT-free fallback   |   "
        f"soak {osm_soak['completed_episodes']}/{osm_soak['requested_episodes']} osmesa, "
        f"{egl_soak['completed_episodes']}/{egl_soak['requested_episodes']} egl episodes, "
        "no signal",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(OUT, dpi=140)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
