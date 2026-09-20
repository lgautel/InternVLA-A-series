"""
Visualization script for plug_into_socket_franka3_15hz_lerobot dataset analysis.
Generates figures for ds2_analyz.md report.

Usage:
    python b/d/Frk2/asset/analyze_plug_socket_ds2.py
"""
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import json
import os

DATA_ROOT = "/B/Dta/plug_into_socket_franka3_15hz_lerobot"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

parquet_files = sorted(glob.glob(f"{DATA_ROOT}/data/chunk-000/episode_*.parquet"))

# Load all data
all_states, all_actions, all_wrenches, all_forces = [], [], [], []
ep_lengths = []
for f in parquet_files:
    tbl = pq.read_table(f)
    df = tbl.to_pandas()
    all_states.append(np.stack(df["observation.state"].values))
    all_actions.append(np.stack(df["action"].values))
    all_wrenches.append(np.stack(df["observation.wrench"].values))
    all_forces.append(np.stack(df["observation.force"].values))
    ep_lengths.append(len(df))

states = np.concatenate(all_states)
actions = np.concatenate(all_actions)
wrenches = np.concatenate(all_wrenches)
forces = np.concatenate(all_forces)

with open(f"{DATA_ROOT}/meta/episodes.jsonl") as fh:
    episodes_meta = [json.loads(line) for line in fh]


# ============================================================
# Figure 1: Episode Length Distribution
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))
lengths = [e["length"] for e in episodes_meta]
ax.hist(lengths, bins=20, color="#4C72B0", edgecolor="white", alpha=0.9)
ax.axvline(np.mean(lengths), color="#C44E52", linestyle="--", linewidth=1.5,
           label=f"Mean = {np.mean(lengths):.0f} frames ({np.mean(lengths)/15:.1f}s)")
ax.set_xlabel("Episode Length (frames)")
ax.set_ylabel("Count")
ax.set_title("Episode Length Distribution (100 episodes)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig1_episode_lengths.png", dpi=150)
plt.close()
print("Saved fig1_episode_lengths.png")


# ============================================================
# Figure 2: Action/State Joint Distributions (7 joints + gripper)
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
joint_names = ["Joint 0", "Joint 1", "Joint 2", "Joint 3",
               "Joint 4", "Joint 5", "Joint 6", "Gripper"]
for i, (ax, name) in enumerate(zip(axes.flat, joint_names)):
    if i < 7:
        ax.hist(states[:, i], bins=50, alpha=0.6, color="#4C72B0", label="State", density=True)
        ax.hist(actions[:, i], bins=50, alpha=0.6, color="#C44E52", label="Action", density=True)
    else:
        ax.hist(states[:, 7], bins=50, alpha=0.6, color="#4C72B0", label="State (width)", density=True)
        ax.hist(actions[:, 7], bins=50, alpha=0.6, color="#C44E52", label="Action", density=True)
    ax.set_title(name)
    ax.legend(fontsize=7)
fig.suptitle("Joint State vs Action Distributions", fontsize=14)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig2_joint_distributions.png", dpi=150)
plt.close()
print("Saved fig2_joint_distributions.png")


# ============================================================
# Figure 3: End-Effector 3D Trajectories (sample episodes)
# ============================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
cmap = plt.cm.tab10
sample_eps = list(range(0, 100, 10))
for idx in sample_eps:
    ee = all_states[idx][:, 8:11]
    color = cmap(idx / 100)
    ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color=color, alpha=0.6, linewidth=1.0)
    ax.scatter(ee[0, 0], ee[0, 1], ee[0, 2], color=color, marker="o", s=20)
    ax.scatter(ee[-1, 0], ee[-1, 1], ee[-1, 2], color=color, marker="x", s=30)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("End-Effector Trajectories (every 10th episode)\n○ = start, × = end")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig3_ee_trajectories.png", dpi=150)
plt.close()
print("Saved fig3_ee_trajectories.png")


# ============================================================
# Figure 4: Wrench (Force/Torque) Time Series - Sample Episode
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ep_idx = 0
w = all_wrenches[ep_idx]
t = np.arange(len(w)) / 15.0

axes[0].plot(t, w[:, 0], label="Fx", linewidth=0.8)
axes[0].plot(t, w[:, 1], label="Fy", linewidth=0.8)
axes[0].plot(t, w[:, 2], label="Fz", linewidth=0.8)
axes[0].set_ylabel("Force (N)")
axes[0].legend()
axes[0].set_title(f"Wrench Time Series — Episode 0 ({len(w)} frames)")
axes[0].axhline(0, color="gray", linestyle=":", linewidth=0.5)

axes[1].plot(t, w[:, 3], label="Tx", linewidth=0.8)
axes[1].plot(t, w[:, 4], label="Ty", linewidth=0.8)
axes[1].plot(t, w[:, 5], label="Tz", linewidth=0.8)
axes[1].set_ylabel("Torque (Nm)")
axes[1].set_xlabel("Time (s)")
axes[1].legend()
axes[1].axhline(0, color="gray", linestyle=":", linewidth=0.5)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig4_wrench_timeseries.png", dpi=150)
plt.close()
print("Saved fig4_wrench_timeseries.png")


# ============================================================
# Figure 5: Force Magnitude Heatmap (all episodes)
# ============================================================
max_len = max(ep_lengths)
n_eps = len(all_wrenches)
force_mag_matrix = np.full((n_eps, max_len), np.nan)
for i, w in enumerate(all_wrenches):
    mag = np.sqrt(w[:, 0] ** 2 + w[:, 1] ** 2 + w[:, 2] ** 2)
    force_mag_matrix[i, : len(mag)] = mag

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(
    force_mag_matrix,
    aspect="auto",
    cmap="hot",
    vmin=0,
    vmax=20,
    interpolation="nearest",
)
ax.set_xlabel("Frame Index")
ax.set_ylabel("Episode Index")
ax.set_title("Force Magnitude (N) Across All Episodes")
cbar = fig.colorbar(im, ax=ax, label="||F|| (N)")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig5_force_heatmap.png", dpi=150)
plt.close()
print("Saved fig5_force_heatmap.png")


# ============================================================
# Figure 6: Gripper + EE Z + Force over time (multi-episode overlay)
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ep_idx in range(0, 100, 5):
    s = all_states[ep_idx]
    w = all_wrenches[ep_idx]
    n = len(s)
    t_norm = np.linspace(0, 1, n)

    axes[0].plot(t_norm, s[:, 7], alpha=0.15, color="#4C72B0", linewidth=0.5)
    axes[1].plot(t_norm, s[:, 10], alpha=0.15, color="#55A868", linewidth=0.5)
    fmag = np.sqrt(w[:, 0] ** 2 + w[:, 1] ** 2 + w[:, 2] ** 2)
    axes[2].plot(t_norm, fmag, alpha=0.15, color="#C44E52", linewidth=0.5)

axes[0].set_ylabel("Gripper Width")
axes[0].set_title("Trajectory Profiles (20 episodes overlay, normalized time)")
axes[1].set_ylabel("EE Z (m)")
axes[2].set_ylabel("||F|| (N)")
axes[2].set_xlabel("Normalized Episode Progress")

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig6_trajectory_profiles.png", dpi=150)
plt.close()
print("Saved fig6_trajectory_profiles.png")


# ============================================================
# Figure 7: Starting position scatter (EE XY and XZ)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
start_ee = np.array([all_states[i][0, 8:11] for i in range(n_eps)])
end_ee = np.array([all_states[i][-1, 8:11] for i in range(n_eps)])

axes[0].scatter(start_ee[:, 0], start_ee[:, 1], c="#4C72B0", s=15, alpha=0.7, label="Start")
axes[0].scatter(end_ee[:, 0], end_ee[:, 1], c="#C44E52", s=15, alpha=0.7, label="End")
axes[0].set_xlabel("X (m)")
axes[0].set_ylabel("Y (m)")
axes[0].set_title("EE Start/End Positions (XY plane)")
axes[0].legend()

axes[1].scatter(start_ee[:, 0], start_ee[:, 2], c="#4C72B0", s=15, alpha=0.7, label="Start")
axes[1].scatter(end_ee[:, 0], end_ee[:, 2], c="#C44E52", s=15, alpha=0.7, label="End")
axes[1].set_xlabel("X (m)")
axes[1].set_ylabel("Z (m)")
axes[1].set_title("EE Start/End Positions (XZ plane)")
axes[1].legend()

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig7_start_end_positions.png", dpi=150)
plt.close()
print("Saved fig7_start_end_positions.png")

print("\nAll figures saved to:", OUT_DIR)
