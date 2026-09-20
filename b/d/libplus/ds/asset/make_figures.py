#!/usr/bin/env python3
"""Render every figure used by libero_raw_analyz2.md from the measured artefacts.

Reads the *.json / *.npz written by the analysis scripts and writes PNGs next
to them. All labels are in English on purpose, so the figures stay readable
when the markdown is exported.

    /B/VENV/libero_plus_client/bin/python make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DPI = 130
SUITE_COLORS = {
    "libero_spatial": "#4C72B0",
    "libero_object": "#DD8452",
    "libero_goal": "#55A868",
    "libero_10": "#C44E52",
}
BODY_NAMES = [f"link{i}" for i in range(1, 8)] + ["eef"]


def load(name: str):
    return json.loads((HERE / name).read_text())


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(HERE / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def fig_scale(raw: dict) -> None:
    suites = list(raw["subsets"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    eps = [raw["subsets"][s]["episodes"] for s in suites]
    frames = [raw["subsets"][s]["frames"] for s in suites]
    colors = [SUITE_COLORS[s] for s in suites]

    axes[0].bar(suites, eps, color=colors)
    for i, (e, s) in enumerate(zip(eps, suites)):
        axes[0].text(i, e + 6, f"{e}\n{raw['subsets'][s]['retention_pct']}% kept", ha="center", fontsize=8)
    axes[0].set_title("Episodes per suite (of 500 official demos)")
    axes[0].set_ylabel("episodes")
    axes[0].set_ylim(0, max(eps) * 1.25)

    axes[1].bar(suites, frames, color=colors)
    for i, f in enumerate(frames):
        axes[1].text(i, f + 900, f"{f}", ha="center", fontsize=8)
    axes[1].set_title("Frames per suite")
    axes[1].set_ylabel("frames")
    axes[1].set_ylim(0, max(frames) * 1.2)

    ep_share = [raw["subsets"][s]["episode_share_pct"] for s in suites]
    fr_share = [raw["subsets"][s]["frame_share_pct"] for s in suites]
    x = np.arange(len(suites))
    axes[2].bar(x - 0.2, ep_share, 0.4, label="episode share", color="#8C8C8C")
    axes[2].bar(x + 0.2, fr_share, 0.4, label="frame share", color="#C44E52")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(suites)
    axes[2].axhline(25, ls="--", lw=1, color="k", alpha=0.5)
    axes[2].set_title("Frame-uniform sampling reweights the suites")
    axes[2].set_ylabel("% of dataset")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.tick_params(axis="x", rotation=18, labelsize=8)
    save(fig, "fig01_scale.png")


def fig_lengths(raw: dict, arrays) -> None:
    lengths = arrays["episode_lengths"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    axes[0].hist(lengths, bins=60, color="#4C72B0", alpha=0.85)
    axes[0].axvline(200, color="#C44E52", ls="--", lw=1.6, label="history window H=200")
    axes[0].axvline(float(np.median(lengths)), color="k", ls=":", lw=1.4, label=f"median {int(np.median(lengths))}")
    axes[0].set_xlabel("episode length (frames)")
    axes[0].set_ylabel("episodes")
    axes[0].set_title("Episode length distribution")
    axes[0].legend(fontsize=8)

    h = 200
    t = np.arange(0, 520)
    fill = np.zeros_like(t, dtype=float)
    for i, step in enumerate(t):
        alive = lengths > step
        fill[i] = np.nan if alive.sum() == 0 else min(step, h) / h
    axes[1].plot(t, fill * 100, color="#55A868", lw=2)
    axes[1].axhline(100, color="k", ls=":", lw=1)
    axes[1].axvline(h, color="#C44E52", ls="--", lw=1.4, label="t = H = 200")
    axes[1].set_xlabel("timestep within episode")
    axes[1].set_ylabel("history window occupancy (%)")
    axes[1].set_title(
        f"Only {raw['episode_length']['frac_shorter_than_200'] * 100:.0f}% of episodes "
        "are shorter than H"
    )
    axes[1].legend(fontsize=8)
    save(fig, "fig02_episode_length.png")


def fig_action(raw: dict, arrays) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]

    lo = np.asarray(raw["action"]["min"])
    hi = np.asarray(raw["action"]["max"])
    mean = np.asarray(raw["action"]["mean"])
    std = np.asarray(raw["action"]["std"])
    x = np.arange(7)
    axes[0].vlines(x, lo, hi, color="#8C8C8C", lw=6, alpha=0.5, label="observed range")
    axes[0].errorbar(x, mean, yerr=std, fmt="o", color="#C44E52", capsize=4, label="mean +- std")
    axes[0].axhline(0.9375, color="#4C72B0", ls="--", lw=1.2, label="+-0.9375 teleop saturation")
    axes[0].axhline(-0.9375, color="#4C72B0", ls="--", lw=1.2)
    axes[0].axhline(1.0, color="k", ls=":", lw=1.2, label="+-1 controller limit")
    axes[0].axhline(-1.0, color="k", ls=":", lw=1.2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("Action channels never reach the controller limit")
    axes[0].legend(fontsize=7, loc="lower left")

    pool = arrays["action_pool"]
    axes[1].hist(pool, bins=120, color="#4C72B0", log=True)
    axes[1].axvline(0.9375, color="#C44E52", ls="--", lw=1.4)
    axes[1].set_xlabel("|action| (first 6 dims)")
    axes[1].set_ylabel("count (log)")
    axes[1].set_title(
        f"{raw['action']['frames_at_clip_pct']}% of frames pile up at the 0.9375 rail"
    )

    q = raw["action"]["quantisation"]
    zoom = pool[pool < 0.02]
    axes[2].hist(zoom, bins=200, color="#55A868")
    axes[2].set_xlabel("|action| (zoom near zero)")
    axes[2].set_ylabel("count")
    axes[2].set_title(
        f"Discrete teleop grid: pos 1/{q['position_xyz']['grid']}, rot 1/{q['rotation_rpy']['grid']}"
    )
    save(fig, "fig03_action.png")


def fig_state(raw: dict, arrays) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    eef = arrays["eef_positions"]
    hb = axes[0].hexbin(eef[:, 0], eef[:, 2], gridsize=55, bins="log", cmap="viridis")
    axes[0].set_xlabel("world x (m)")
    axes[0].set_ylabel("world z (m)")
    axes[0].set_title("EEF workspace, live arena frame\n(two z clusters = table vs floor arenas)")
    fig.colorbar(hb, ax=axes[0], label="log count")

    aa = arrays["axisangle_norms"]
    axes[1].hist(aa, bins=120, color="#DD8452")
    axes[1].axvline(np.pi, color="k", ls="--", lw=1.6, label=r"$\pi$")
    axes[1].set_xlabel(r"$\|$axis-angle$\|$ (rad)")
    axes[1].set_ylabel("frames")
    dc = raw["state"]["axisangle_double_cover"]
    axes[1].set_title(
        f"{dc['frames_with_norm_over_pi_pct']}% of frames exceed "
        r"$\pi$" + "\n(robosuite does not fold the hemisphere)"
    )
    axes[1].legend(fontsize=8)

    labels = ["x", "y", "z", "ax", "ay", "az", "grip0", "grip1"]
    lo, hi = np.asarray(raw["state"]["min"]), np.asarray(raw["state"]["max"])
    x = np.arange(8)
    axes[2].vlines(x, lo, hi, color="#4C72B0", lw=7, alpha=0.7)
    axes[2].plot(x, raw["state"]["mean"], "o", color="#C44E52")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_title("state[8] = eef_pos(3) + axisangle(3) + gripper(2)")
    axes[2].axhline(0, color="k", lw=0.6)
    save(fig, "fig04_state.png")


def fig_arena(kpt: dict, kpt_arrays) -> None:
    names = list(kpt["arena_bases"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    measured = np.asarray([kpt["arena_bases"][n]["measured_base_xpos"] for n in names])
    frames = np.asarray([kpt["arena_bases"][n]["frames"] for n in names])
    sizes = 120 + 900 * frames / frames.max()
    axes[0].scatter(measured[:, 0], measured[:, 2], s=sizes, c="#4C72B0", alpha=0.65, edgecolors="k")
    axes[0].scatter([-0.56], [0.912], marker="*", s=420, c="#C44E52", edgecolors="k", zorder=5)
    axes[0].annotate(
        "Lift virtual base (-0.56, 0, 0.912)\nused by ALL keypoints",
        (-0.56, 0.912),
        textcoords="offset points",
        xytext=(14, 18),
        fontsize=8,
        color="#C44E52",
    )
    for n, m in zip(names, measured):
        axes[0].annotate(n, (m[0], m[2]), textcoords="offset points", xytext=(8, -14), fontsize=8)
    axes[0].set_xlabel("base world x (m)")
    axes[0].set_ylabel("base world z (m)")
    axes[0].set_title("Arena bases recovered from the data\n(marker area = frame share)")
    axes[0].grid(alpha=0.25)

    err = [kpt["arena_bases"][n]["abs_diff_to_reference_m"] * 1000 for n in names]
    spread = [kpt["arena_bases"][n]["spread_across_episodes_m"] * 1000 for n in names]
    x = np.arange(len(names))
    axes[1].bar(x - 0.2, err, 0.4, label="|measured - LIBERO source|", color="#4C72B0")
    axes[1].bar(x + 0.2, spread, 0.4, label="spread across episodes", color="#DD8452")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, fontsize=8)
    axes[1].set_ylabel("mm")
    axes[1].set_title("Recovery error stays below 0.1 mm")
    axes[1].legend(fontsize=8)
    save(fig, "fig05_arena_base.png")


def fig_rpad(kpt: dict) -> None:
    schemes = kpt["normalisation_schemes"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    names = ["A: Lift world\n+ isotropic R_pad", "B: base frame\n+ isotropic", "C: base frame\n+ per axis"]
    keys = ["A_lift_world_isotropic", "B_base_frame_isotropic", "C_base_frame_per_axis"]
    width = 0.25
    x = np.arange(3)
    for i, axis in enumerate("xyz"):
        vals = [schemes[k]["axis_utilisation"][i] * 100 for k in keys]
        bars = axes[0].bar(x + (i - 1) * width, vals, width, label=axis)
        for b, v in zip(bars, vals):
            axes[0].text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_ylabel(r"% of $[-1, 1]$ occupied")
    axes[0].set_title("Normalization scheme vs dynamic range used")
    axes[0].set_ylim(0, 115)
    axes[0].legend(title="axis", fontsize=8)

    a = schemes["A_lift_world_isotropic"]
    lo, hi = np.asarray(a["range_min"]), np.asarray(a["range_max"])
    y = np.arange(3)
    axes[1].hlines(y, -1, 1, color="#CCCCCC", lw=14, label="representable range")
    axes[1].hlines(y, lo, hi, color="#C44E52", lw=14, label="actually occupied")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(["x", "y", "z"])
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_xlim(-1.15, 1.15)
    axes[1].set_title(
        f"Scheme A (shipped): R_pad = {kpt['r_pad']['recomputed_from_this_dataset']:.7f}\n"
        "z never goes negative, x/y never reach the rails"
    )
    axes[1].legend(fontsize=8, loc="lower right")
    save(fig, "fig06_normalisation.png")


def fig_redundancy(kpt: dict, kpt_arrays) -> None:
    std = kpt_arrays["body_std"]
    lo, hi = kpt_arrays["body_lo"], kpt_arrays["body_hi"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))

    x = np.arange(8)
    for i, axis in enumerate("xyz"):
        axes[0].bar(x + (i - 1) * 0.26, std[:, i] * 100, 0.26, label=axis)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(BODY_NAMES)
    axes[0].set_ylabel("position std (cm)")
    axes[0].set_title(
        f"link1/link2 are frozen; link5==link6\n"
        f"only {kpt['redundancy']['independent_position_dims_of_24']} of 24 position dims "
        "are independent"
    )
    axes[0].legend(title="axis", fontsize=8)

    span = (hi - lo) * 100
    im = axes[1].imshow(span.T, aspect="auto", cmap="magma")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(BODY_NAMES)
    axes[1].set_yticks(range(3))
    axes[1].set_yticklabels(["x", "y", "z"])
    for i in range(8):
        for j in range(3):
            axes[1].text(i, j, f"{span[i, j]:.0f}", ha="center", va="center",
                         color="white" if span[i, j] < span.max() * 0.6 else "black", fontsize=8)
    axes[1].set_title("Per-body travel range (cm)")
    fig.colorbar(im, ax=axes[1], label="cm")
    save(fig, "fig07_redundancy.png")


def fig_quaternion(kpt: dict) -> None:
    q = kpt["quaternion"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))

    x = np.arange(8)
    near = [q["frames_with_qw_below_0.05"][b] for b in BODY_NAMES]
    axes[0].bar(x, np.asarray(near) / kpt["frames"] * 100, color="#DD8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(BODY_NAMES)
    axes[0].set_ylabel(r"% of frames with $|q_w| < 0.05$")
    axes[0].set_title("link7 and eef sit on the hemisphere branch cut")

    flips = [q["sign_flips_between_consecutive_frames"][b] for b in BODY_NAMES]
    axes[1].bar(x, flips, color="#C44E52")
    for i, f in enumerate(flips):
        if f:
            axes[1].text(i, f + 60, str(f), ha="center", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(BODY_NAMES)
    axes[1].set_ylabel("antipodal jumps")
    axes[1].set_title(
        f"{q['total_sign_flips']} sign flips total\n(an MSE loss reads each as maximal error)"
    )
    save(fig, "fig08_quaternion.png")


def fig_orientation(orient: dict) -> None:
    overlay = np.load(HERE / "orientation_overlay.npz")
    chain = np.load(HERE / "chain_orientation_frames.npz")
    chain_report = load("chain_orientation_report.json")

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))

    # --- top row: the actual pixels at each hop of the chain
    axes[0, 0].imshow(chain["rlds"])
    axes[0, 0].set_title("1. RLDS JPEG, as stored\n(= rot180 of the MuJoCo buffer)", fontsize=9)
    axes[0, 1].imshow(chain["merged"])
    axes[0, 1].set_title(
        f"2. merged_kpt mp4 frame 0\n(= rot180 of panel 1, MSE {chain_report['verdict']['mse']})",
        fontsize=9,
    )
    axes[0, 2].imshow(chain["merged"])
    axes[0, 2].set_title(
        "3. what eval sends with rotate_images=false\n(obs['agentview_image'], OpenGL raw)",
        fontsize=9,
    )
    for ax in axes[0]:
        ax.axis("off")

    # --- bottom row: the two independent measurements plus the projection overlay
    frame = overlay["frame"]
    axes[1, 0].imshow(frame)
    names = orient["overlay"]["variant_order"]
    preds = overlay["predictions"]
    markers = ["o", "s", "^", "D"]
    for (name, pred, m) in zip(names, preds, markers):
        axes[1, 0].plot(pred[1], pred[0], m, ms=11, mfc="none", mew=2.2, label=name.split(" ")[0])
    c = overlay["centroid"]
    axes[1, 0].plot(c[1], c[0], "x", ms=16, mew=3, color="#FF3333", label="observed motion centroid")
    axes[1, 0].set_title(
        f"Projected EEF on an RLDS frame ({orient['overlay']['arena']})", fontsize=9
    )
    axes[1, 0].legend(fontsize=7, loc="lower left")
    axes[1, 0].axis("off")

    scores = orient["variant_scores_px"]
    labels = [k.split(" ")[0] for k in scores]
    vals = [v["median_pixel_error"] for v in scores.values()]
    colors = ["#C44E52" if v == min(vals) else "#8C8C8C" for v in vals]
    axes[1, 1].barh(labels, vals, color=colors)
    for i, v in enumerate(vals):
        axes[1, 1].text(v + 2, i, f"{v:.0f}", va="center", fontsize=8)
    axes[1, 1].set_xlabel("median projection error (px)")
    axes[1, 1].set_title(
        "Geometry test: RLDS = rot180 of MuJoCo buffer\n"
        f"row corr {orient['trajectory_correlation']['row_median']:+.2f}, "
        f"col corr {orient['trajectory_correlation']['col_median']:+.2f}",
        fontsize=9,
    )

    mse = chain_report["mse_rlds_vs_merged"]
    labels2 = list(mse)
    vals2 = [mse[k] for k in labels2]
    colors2 = ["#C44E52" if v == min(vals2) else "#8C8C8C" for v in vals2]
    axes[1, 2].barh(labels2, vals2, color=colors2, log=True)
    for i, v in enumerate(vals2):
        axes[1, 2].text(v * 1.15, i, f"{v:.0f}", va="center", fontsize=8)
    axes[1, 2].set_xlabel("MSE vs merged_kpt mp4 frame 0 (log)")
    axes[1, 2].set_title(
        "Pixel test: merged_kpt = rot180 of RLDS\n"
        f"{chain_report['verdict']['margin_ratio']:.0f}x margin",
        fontsize=9,
    )
    save(fig, "fig09_orientation.png")


def fig_libero_plus(plus: dict) -> None:
    arrays = np.load(HERE / "libero_plus_arrays.npz")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

    cat = plus["catalogue"]["overall_by_category"]
    names = list(cat)
    vals = [cat[n] for n in names]
    colors = ["#C44E52" if plus["channel_reach"].get(n, {}).get("keypoints") else "#4C72B0" for n in names]
    axes[0].barh(names, vals, color=colors)
    axes[0].set_xlabel("tasks")
    axes[0].set_title(
        f"LIBERO-plus catalogue: {100 * plus['verdict']['fraction_of_catalogue_invisible_to_keypoints']:.0f}% "
        "invisible\nto the keypoint channel (blue)",
        fontsize=9,
    )
    axes[0].tick_params(axis="y", labelsize=8)

    tiers = plus["init_qpos_perturbation"]["tiers"]
    tier_keys = list(tiers)
    disp = arrays["eef_disp_m"] * 100
    tier_vals = arrays["tier"]
    data = [disp[tier_vals == float(t)] for t in tier_keys]
    axes[1].boxplot(data, tick_labels=tier_keys, showfliers=False)
    p95 = plus["training_first_frame_distribution"]["eef_radius_p95_m"] * 100
    axes[1].axhline(p95, color="#C44E52", ls="--", lw=1.8,
                    label=f"training start spread p95 = {p95:.1f} cm")
    axes[1].set_xlabel(r"designed $\|\delta q\|$ (rad)")
    axes[1].set_ylabel("EEF displacement (cm)")
    axes[1].set_title("init_qpos perturbation dwarfs the\ntraining start distribution", fontsize=9)
    axes[1].legend(fontsize=8)

    axes[2].hist(arrays["train_eef_radius"] * 100, bins=60, color="#55A868", alpha=0.9,
                 label="training first-frame EEF radius")
    axes[2].axvline(
        float(np.median(disp[tier_vals == 0.1])),
        color="#C44E52", ls="--", lw=1.8,
        label=f"mildest tier median = {np.median(disp[tier_vals == 0.1]):.1f} cm",
    )
    axes[2].set_xlabel("distance from mean start pose (cm)")
    axes[2].set_ylabel("episodes")
    axes[2].set_title("Even tier 0.1 rad starts out of distribution", fontsize=9)
    axes[2].legend(fontsize=8)
    save(fig, "fig10_libero_plus.png")


def main() -> int:
    raw = load("raw_stats.json")
    kpt = load("kpt_stats.json")
    orient = load("orientation_report.json")
    plus = load("libero_plus_report.json")
    arrays = np.load(HERE / "raw_arrays.npz", allow_pickle=True)
    kpt_arrays = np.load(HERE / "kpt_arrays.npz", allow_pickle=True)

    print("rendering figures:")
    fig_scale(raw)
    fig_lengths(raw, arrays)
    fig_action(raw, arrays)
    fig_state(raw, arrays)
    fig_arena(kpt, kpt_arrays)
    fig_rpad(kpt)
    fig_redundancy(kpt, kpt_arrays)
    fig_quaternion(kpt)
    fig_orientation(orient)
    fig_libero_plus(plus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
