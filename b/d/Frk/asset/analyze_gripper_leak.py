#!/usr/bin/env python3
"""Quantify the gripper action/state algebraic leak in the Franka plug-into-socket data.

Reproduces every number quoted in ``b/d/Frk/dta_prblm_solv1.markdown``.

The source of truth is the raw HDF5 capture; the 30 Hz LeRobot dataset used for
training is re-derived here with the same nearest-neighbour alignment that
``b/s/Frk/convert_franka_plug_hdf5.py`` performs, so the analysis does not depend
on the (now deleted) ``/B/Dta/plug_into_socket_lrb_4D`` directory.

Usage:
    python b/d/Frk/asset/analyze_gripper_leak.py \
        --hdf5-dir /B/Dta/plug_into_socket_hdf5 \
        --outdir   b/d/Frk/asset
"""

from __future__ import annotations

import argparse
import glob
import os

import h5py
import numpy as np

GRIPPER_MAX = 0.08  # the hard-coded normalisation constant, see b/s/Frk/verify_franka_conversion.py
CHUNK_SIZE = 50  # policy.chunk_size used by both warmup and SFT
N_EXEC_EVAL = 10  # --n-exec default of the real-robot inference server


def load_raw(hdf5_dir: str):
    """Read every episode at the native 100 Hz state rate."""
    out = {k: [] for k in ("w", "a", "jp", "aj", "ei")}
    for i, fp in enumerate(sorted(glob.glob(os.path.join(hdf5_dir, "episode_*.hdf5")))):
        with h5py.File(fp, "r") as f:
            rs = f["robot_state"]
            out["w"].append(rs["gripper_width"][:].reshape(-1))
            out["a"].append(rs["action_gripper"][:].reshape(-1))
            out["jp"].append(rs["joint_positions"][:])
            out["aj"].append(rs["action_joints"][:])
            out["ei"].append(np.full(len(rs["timestamps"]), i))
    return {k: np.concatenate(v) for k, v in out.items()}


def load_aligned_30hz(hdf5_dir: str):
    """Re-derive the 30 Hz LeRobot tensors: camera frames anchor, nearest state index."""
    S, A, EI = [], [], []
    for i, fp in enumerate(sorted(glob.glob(os.path.join(hdf5_dir, "episode_*.hdf5")))):
        with h5py.File(fp, "r") as f:
            rs = f["robot_state"]
            ts = rs["timestamps"][:]
            n = min(len(f["camera_global/timestamps"]), len(f["camera_wrist/timestamps"]))
            cam = f["camera_global/timestamps"][:][:n]
            idx = np.abs(ts[None, :] - cam[:, None]).argmin(axis=1)
            S.append(
                np.column_stack(
                    [rs["joint_positions"][:][idx], rs["gripper_width"][:].reshape(-1)[idx]]
                ).astype(np.float32)
            )
            A.append(
                np.column_stack(
                    [rs["action_joints"][:][idx], rs["action_gripper"][:].reshape(-1)[idx]]
                ).astype(np.float32)
            )
            EI.append(np.full(n, i))
    return np.concatenate(S).astype(np.float64), np.concatenate(A).astype(np.float64), np.concatenate(EI)


def copy_residual(zA, zS, EI, horizons, channel):
    """Residual of predicting z_action[t+h] by copying the current z_state[t].

    The gripper is the *negated* copy because a = 1 - w/0.08 flips the sign after
    mean/std normalisation; the arm joints are a straight copy.
    """
    chunks = []
    for h in horizons:
        per_ep = []
        for ep in np.unique(EI):
            idx = np.flatnonzero(EI == ep)
            tgt = np.minimum(np.arange(len(idx)) + h, len(idx) - 1)  # LeRobot pads with the last frame
            if channel == "gripper":
                per_ep.append(zA[idx[tgt], 7] + zS[idx, 7])
            else:
                per_ep.append((zA[idx[tgt], :7] - zS[idx, :7]).ravel())
        chunks.append(np.concatenate(per_ep))
    return np.concatenate(chunks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5-dir", default="/B/Dta/plug_into_socket_hdf5")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    raw = load_raw(args.hdf5_dir)
    w, a = raw["w"], raw["a"]
    resid = np.abs(a - (1 - w / GRIPPER_MAX))
    print(f"[raw 100Hz] frames={len(w)}  max|a-(1-w/0.08)|={resid.max():.3e}")
    print(f"[raw 100Hz] bit-exact w == 0.08*(1-a) : {(w == GRIPPER_MAX * (1 - a)).mean() * 100:.4f}%")
    print(f"[raw 100Hz] bit-exact a == 1-w/0.08   : {(a == 1 - w / GRIPPER_MAX).mean() * 100:.4f}%")
    print(f"[raw 100Hz] |unique(a)|={len(np.unique(a))}  |unique(w)|={len(np.unique(w))}")

    S, A, EI = load_aligned_30hz(args.hdf5_dir)
    print(f"[30Hz]      frames={len(S)} (conversion log records 66,577)")
    zA = (A - A.mean(0)) / A.std(0)
    zS = (S - S.mean(0)) / S.std(0)

    horizons = np.arange(CHUNK_SIZE)
    r2_g, r2_a = [], []
    for h in horizons:
        eg = copy_residual(zA, zS, EI, [h], "gripper")
        ea = copy_residual(zA, zS, EI, [h], "arm")
        r2_g.append(1 - (eg**2).mean())
        r2_a.append(1 - (ea**2).mean())

    mse_g = (copy_residual(zA, zS, EI, horizons, "gripper") ** 2).mean()
    mse_a = (copy_residual(zA, zS, EI, horizons, "arm") ** 2).mean()
    print(f"[chunk50]   gripper copy-MSE={mse_g:.4f}  arm copy-MSE={mse_a:.4f}")
    print(f"[chunk50]   8-dim action MSE by pure copy = {(7 * mse_a + mse_g) / 8:.4f} (vs 1.0 for mean)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

    ep0 = EI == 0
    t = np.arange(ep0.sum()) / 30.0
    ax[0].plot(t, A[ep0, 7], lw=1.8, color="#c0392b", label=r"$a_t$  (action.gripper)")
    ax[0].plot(t, 1 - S[ep0, 7] / GRIPPER_MAX, lw=1.8, ls="--", color="#2980b9",
               label=r"$1-w_t/0.08$  (from state)")
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("value")
    ax[0].set_title("Episode 0: the two curves are bit-identical\n"
                    rf"$\max|a_t-(1-w_t/0.08)|={resid.max():.2e}$ on all 221,428 raw frames", fontsize=10)
    ax[0].legend(fontsize=9)

    ax[1].plot(horizons, r2_g, lw=2.2, color="#c0392b", marker="o", ms=3, label="gripper (dim 7)")
    ax[1].plot(horizons, r2_a, lw=2.2, color="#16a085", marker="s", ms=3, label="arm (dims 0-6, mean)")
    ax[1].axvspan(0, N_EXEC_EVAL - 1, color="#f39c12", alpha=0.18)
    ax[1].text(N_EXEC_EVAL / 2, 0.30, f"executed\nprefix\n(n_exec={N_EXEC_EVAL})",
               ha="center", fontsize=8.5, color="#a2690a")
    r2_exec = 1 - (copy_residual(zA, zS, EI, range(N_EXEC_EVAL), "gripper") ** 2).mean()
    print(f"[n_exec={N_EXEC_EVAL}] gripper copy-R2 over executed prefix = {r2_exec:.4f}")
    ax[1].set_xlabel("position in action chunk  $h$")
    ax[1].set_ylabel(r"$R^2$ of copying current state")
    ax[1].set_title("The shortcut is perfect exactly where it is executed\n"
                    r"$R^2$ at $h=0$ is 1.0000;  mean over $h<10$ is "
                    f"{r2_exec:.4f}", fontsize=10)
    ax[1].legend(fontsize=9)
    ax[1].set_ylim(0, 1.05)

    labels = ["predict\nconstant mean", "copy current\nstate (arm+grip)", "copy, gripper\ndim only"]
    vals = [1.0, (7 * mse_a + mse_g) / 8, (7 * 1.0 + mse_g) / 8]
    bars = ax[2].bar(labels, vals, color=["#95a5a6", "#c0392b", "#e67e22"])
    for b, v in zip(bars, vals):
        ax[2].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    ax[2].set_ylabel("normalised 8-dim action MSE")
    ax[2].set_title("Loss obtainable with no visual input at all\n"
                    f"pure copy removes {(1 - (7 * mse_a + mse_g) / 8) * 100:.0f}% of the action loss", fontsize=10)
    ax[2].set_ylim(0, 1.15)

    for x in ax:
        x.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(args.outdir, "fig1_gripper_leak.png")
    plt.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
