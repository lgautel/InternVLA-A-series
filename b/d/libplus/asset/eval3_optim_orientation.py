#!/usr/bin/env python3
"""Figures for eval3_optim.md: the 180-degree train/eval image orientation mismatch.

Outputs (same directory):
  - eval3_optim_orientation.png : 3-panel visual proof + MSE control table
  - eval3_optim_sr.png          : measured SR by suite and by perturbation category

Run with the server venv (has torchvision + matplotlib):
    /B/VENV/itnvla15rbt20/bin/python eval3_optim_orientation.py
"""
from __future__ import annotations

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision.io import read_video

HERE = os.path.dirname(os.path.abspath(__file__))

TRAIN_AV = "/B/Dta/opvla_libero_merged_kpt/videos/observation.images.image/chunk-000/file-000.mp4"
EVAL_GLOB = (
    "/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/"
    "2026_09_12_08_52_49-internvla_a1_5-libplus-sft/eval_libero_plus/"
    "step_026725/full_20260914_164719/videos/libero_spatial/"
    "rollout_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_"
    "and_place_it_on_the_plate_language_*.mp4"
)

# Measured 2026-09-16 via evaluation/LIBERO-plus2/check_progress.py and
# per-category aggregation over worker_gpu*/client_*.log of full_20260915_095437.
SR_SUITE = [
    ("libero_spatial", 172, 2402),
    ("libero_object", 558, 2121),
]
SR_CATEGORY = [
    ("Light Conditions", 170, 585, 96.4),
    ("Background Textures", 129, 506, 98.2),
    ("Objects Layout", 130, 703, 85.2),
    ("Camera Viewpoints", 122, 758, 83.1),
    ("Language Instructions", 104, 731, 86.9),
    ("Sensor Noise", 42, 533, 95.6),
    ("Robot Initial States", 37, 721, 55.1),
]


def frames(path: str, n: int) -> np.ndarray:
    vid, _, _ = read_video(path, start_pts=0, end_pts=8.0, pts_unit="sec", output_format="THWC")
    return vid[:n].numpy().astype(np.float32)


def best_mse(frame: np.ndarray, bank: np.ndarray) -> float:
    return float(min(np.mean((frame - b) ** 2) for b in bank))


def make_orientation_fig() -> None:
    train = frames(TRAIN_AV, 60)
    eval_path = sorted(glob.glob(EVAL_GLOB))[0]
    ev = frames(eval_path, 1)[0]

    as_is = best_mse(ev, train)
    rot180 = best_mse(ev[::-1, ::-1], train)
    ctrl_same = best_mse(train[0], train[1:])
    ctrl_rot = best_mse(train[0][::-1, ::-1], train[1:])

    fig = plt.figure(figsize=(15.5, 5.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.5], wspace=0.30)

    panels = [
        (train[0], "TRAINING frame\n(opvla_libero_merged_kpt)\nraw robosuite orientation"),
        (ev, "EVAL frame actually sent to model\nclient applies raw[::-1, ::-1]\n=> 180 deg rotated"),
        (ev[::-1, ::-1], "EVAL frame rotated back 180 deg\n=> matches training"),
    ]
    for col, (img, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img.astype(np.uint8))
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    rows = [
        ("eval frame as sent  vs training", as_is),
        ("eval frame rot180   vs training", rot180),
        ("control: train vs train (same)", ctrl_same),
        ("control: train rot180 vs train", ctrl_rot),
    ]
    tbl = ax.table(
        cellText=[[k, f"{v:,.0f}"] for k, v in rows],
        colLabels=["pixel MSE comparison", "MSE"],
        cellLoc="left",
        colWidths=[0.78, 0.22],
        loc="center",
        bbox=[0.0, 0.30, 1.0, 0.46],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    ax.set_title("Control experiment", fontsize=10)
    ax.text(
        0.5,
        0.20,
        "The mismatch of the eval input (%.0f)\nequals the 180 deg signature (%.0f)\n=> eval input is rotated, training is not"
        % (as_is, ctrl_rot),
        ha="center",
        va="top",
        fontsize=9,
        transform=ax.transAxes,
    )

    fig.suptitle(
        "Root cause: eval feeds the policy 180-degree-rotated images that training never saw",
        fontsize=12.5,
        y=1.02,
    )
    out = os.path.join(HERE, "eval3_optim_orientation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    print(f"as_is={as_is:.1f} rot180={rot180:.1f} ctrl_same={ctrl_same:.1f} ctrl_rot={ctrl_rot:.1f}")


def make_sr_fig() -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 4.4), gridspec_kw={"width_ratios": [1, 1.7]})

    names = [n for n, _, _ in SR_SUITE]
    srs = [100.0 * s / d for _, s, d in SR_SUITE]
    ax0.bar(names, srs)
    for i, (v, (_, s, d)) in enumerate(zip(srs, SR_SUITE)):
        ax0.text(i, v + 0.7, f"{v:.1f}%\n({s}/{d})", ha="center", fontsize=9)
    ax0.set_ylabel("Success rate (%)")
    ax0.set_ylim(0, 40)
    ax0.set_title("Measured SR by suite (run full_20260915_095437)", fontsize=10)
    ax0.grid(axis="y", alpha=0.3)

    cats = [c for c, _, _, _ in SR_CATEGORY]
    ours = [100.0 * s / d for _, s, d, _ in SR_CATEGORY]
    paper = [p for _, _, _, p in SR_CATEGORY]
    x = np.arange(len(cats))
    w = 0.38
    ax1.bar(x - w / 2, paper, w, label="Paper InternVLA-A1.5 (Table 6)")
    ax1.bar(x + w / 2, ours, w, label="This run (measured)")
    for i, v in enumerate(ours):
        ax1.text(i + w / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace(" ", "\n") for c in cats], fontsize=8)
    ax1.set_ylabel("Success rate (%)")
    ax1.set_ylim(0, 110)
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.set_title(
        "All 7 perturbation categories collapse uniformly\n"
        "-> points to a global preprocessing fault, not a perturbation-specific weakness",
        fontsize=10,
    )
    ax1.grid(axis="y", alpha=0.3)

    out = os.path.join(HERE, "eval3_optim_sr.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    make_orientation_fig()
    make_sr_fig()
