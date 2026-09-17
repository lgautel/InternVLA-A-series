#!/usr/bin/env python3
"""T1 figures: wraps evaluation/LIBERO2/orientation_contract.py

Outputs (same directory):
  eval3_optim_wrist_t1.png
  eval3_optim_wrist_t1.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = Path("/B/SRC/itvlaGpLibPlus")
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))


def main() -> None:
    from evaluation.LIBERO2.orientation_contract import (
        evaluate_orientation_contract,
        setup_client_egl_env,
    )

    setup_client_egl_env()
    report = evaluate_orientation_contract()
    vis = report.pop("_vis")
    serial = {k: v for k, v in report.items()}
    json_path = HERE / "eval3_optim_wrist_t1.json"
    json_path.write_text(json.dumps(serial, indent=2))
    print("wrote", json_path)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.4))
    titles_col = [
        "TRAINING (dataset video)",
        "LIVE raw env obs",
        "LIVE raw[::-1, ::-1]  (old eval default)",
    ]
    rows = []
    for r, name in enumerate(("agentview", "wrist")):
        cam = report["cameras"][name]
        show = [
            vis[name]["best_train"].astype(np.uint8),
            vis[name]["raw"],
            vis[name]["rot"],
        ]
        for c, img in enumerate(show):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles_col[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(name, fontsize=11)
        rows.append((name, cam["mse_live_raw_vs_train"], cam["mse_live_rot180_vs_train"]))
        print(f"\n=== {name} ===")
        print(json.dumps(cam, indent=2))

    fig.suptitle(
        "T1 live EGL: wrist + agentview orientation vs training\n"
        + " | ".join(f"{n}: raw={a:.0f} rot={b:.0f}" for n, a, b in rows),
        fontsize=12,
    )
    fig.tight_layout()
    png = HERE / "eval3_optim_wrist_t1.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print("wrote", png)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
