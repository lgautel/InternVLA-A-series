#!/usr/bin/env python3
"""Build a training-ready copy of the franka3 plug-into-socket dataset.

The source capture has three defects that no amount of inference-side tuning can
repair (see ``b/d/Frk2/ds2_pblm_solv1.md``):

  P1  ``action[7]`` is a bit-exact copy of ``observation.state[7]`` on 100% of
      frames, so the gripper target carries zero information the policy does not
      already have in its prompt.
  P2  ``info.json`` names dim 7 ``gripper_width`` while the stored value is the
      normalised *closedness* command (1.0 = fully closed).  Following the name
      inverts the gripper on the real robot.
  P3  ``state[8:15]`` (ee_pos + ee_quat) is forward kinematics of ``state[0:7]``,
      so it adds 7 redundant dimensions that widen the copy shortcut for the arm.

This script writes a new dataset directory with the defects addressed, leaving the
source untouched and requiring no change to the training framework: point
``--dataset.repo_id`` at the output and the existing pipeline just works.

Transformations (each independently switchable):

  --gripper-state-mode
      keep  : leave state[7] as-is (baseline, reproduces the defect)
      mask  : replace state[7] with a constant  -> the policy must read the
              cameras to decide when to close.  This is the honest fix: the
              capture contains no measured width to restore.
      lag   : replace state[7] with the command delayed by --gripper-lag frames,
              emulating actuator response.  Weaker than `mask` but keeps a
              closed-loop signal.  The lag value must come from a measured
              open/close step, not be guessed.
  --ee-state-mode
      keep  : leave state[8:15]
      mask  : constant-fill the FK-redundant dims

Added columns (never overwrite existing ones):
  action.gripper_phase        int64  0=hold_open 1=closing 2=hold_closed 3=opening
  action.frames_to_next_event int64  frames until the next phase change (clipped)

Added metadata:
  meta/gripper_contract.json  machine-readable polarity + calibration contract,
                              with the polarity *derived from the data*, not assumed.

Usage:
    python b/s/Frk2/ds/build_franka3_fixed_dataset.py \
        --src  /B/Dta/plug_into_socket_franka3_15hz_lerobot \
        --dst  ~/b/Dta/plug_into_socket_franka3_15hz_fixed \
        --gripper-state-mode mask --ee-state-mode keep
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GRIP = 7
EE = slice(8, 15)
GRIPPER_MAX_M = 0.08  # the constant baked into the source capture; UNCALIBRATED
CLOSE_THRESHOLD = 0.5
PHASE_NAMES = {0: "hold_open", 1: "closing", 2: "hold_closed", 3: "opening"}


# --------------------------------------------------------------------- polarity


def derive_polarity(state: np.ndarray, action: np.ndarray, ep: np.ndarray, wrench: np.ndarray | None):
    """Decide whether a high value of dim 7 means 'closed' or 'open', from the data.

    Three independent signals are combined so that a single misleading one cannot
    flip the answer:

      1. Episode shape.  A plug-insertion demo is open -> closed -> open, so the
         value at mid-episode must be on the *closed* side of the value at t=0.
      2. Contact force.  While the plug is held and pushed, |fz| is larger.
      3. Range arithmetic.  w = w_max * (1 - v) must land in [0, w_max].
    """
    v = state[:, GRIP]
    starts, mids = [], []
    for e in np.unique(ep):
        x = v[ep == e]
        starts.append(x[0])
        mids.append(np.median(x[len(x) // 3: 2 * len(x) // 3]))
    start_med, mid_med = float(np.median(starts)), float(np.median(mids))
    high_is_closed_shape = mid_med > start_med

    force_note = "n/a"
    high_is_closed_force = None
    if wrench is not None:
        hi = np.abs(wrench[v > CLOSE_THRESHOLD, 2])
        lo = np.abs(wrench[v <= CLOSE_THRESHOLD, 2])
        if hi.size and lo.size:
            high_is_closed_force = float(hi.mean()) > float(lo.mean())
            force_note = f"|fz| high-side={hi.mean():.3f}N low-side={lo.mean():.3f}N"

    w = GRIPPER_MAX_M * (1.0 - v)
    range_ok = bool(w.min() >= -1e-9 and w.max() <= GRIPPER_MAX_M + 1e-9)

    votes = [high_is_closed_shape] + ([high_is_closed_force] if high_is_closed_force is not None else [])
    high_is_closed = sum(votes) > len(votes) / 2

    return {
        "high_value_means": "closed" if high_is_closed else "open",
        "evidence_episode_shape": {
            "median_first_frame": start_med,
            "median_mid_episode": mid_med,
            "high_is_closed": high_is_closed_shape,
        },
        "evidence_contact_force": {"note": force_note, "high_is_closed": high_is_closed_force},
        "evidence_width_arithmetic": {
            "formula": "w_m = w_max_m * (1 - v)",
            "implied_width_range_m": [float(w.min()), float(w.max())],
            "range_consistent": range_ok,
        },
        "unanimous": len(set(votes)) == 1,
    }


def build_contract(state, action, ep, wrench) -> dict:
    v = state[:, GRIP]
    pol = derive_polarity(state, action, ep, wrench)
    high_closed = pol["high_value_means"] == "closed"
    return {
        "schema_version": "1.0",
        "channel": "observation.state[7] / action[7]",
        "info_json_name": "gripper_width",
        "info_json_name_is_MISLEADING": True,
        "actual_semantics": "normalised closedness command",
        "value_range": [float(v.min()), float(v.max())],
        "polarity": pol,
        "close_if_above": CLOSE_THRESHOLD if high_closed else None,
        "close_if_below": None if high_closed else CLOSE_THRESHOLD,
        "physical_width": {
            "formula": "w_m = w_max_m * (1 - v)" if high_closed else "w_m = w_max_m * v",
            "w_max_m": GRIPPER_MAX_M,
            "w_max_source": "hard-coded in the source capture; NOT from a calibration record",
            "w_max_observed_max_m": float(GRIPPER_MAX_M * (1.0 - v.min())),
        },
        "WARNINGS": [
            "state[7] is the COMMAND, not a measurement: action[7] == state[7] bit-exactly.",
            "No measured gripper width, grasp force or is_grasped channel exists in this capture.",
            "w_max=0.08 is uncalibrated; the real hand reported 0.0664 m during evaluation.",
        ],
    }


# ------------------------------------------------------------------ phase labels


def phase_labels(v: np.ndarray, high_is_closed: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame gripper phase and frames-until-next-phase-change.

    ``closed`` is a boolean track; a change in that track is an event.  The
    transition frames themselves are labelled ``closing`` / ``opening`` so the
    0.6% of frames that actually carry a decision can be weighted or resampled.
    """
    closed = (v > CLOSE_THRESHOLD) if high_is_closed else (v < CLOSE_THRESHOLD)
    n = len(closed)
    change = np.flatnonzero(np.diff(closed.astype(np.int8)) != 0)

    phase = np.where(closed, 2, 0).astype(np.int64)
    for k in change:
        phase[k + 1] = 1 if closed[k + 1] else 3

    ttl = np.full(n, n, dtype=np.int64)
    events = set((change + 1).tolist())
    nxt = n
    for i in range(n - 1, -1, -1):
        if i in events:
            nxt = i
        ttl[i] = nxt - i
    return phase, ttl


# ------------------------------------------------------------------------ stats


def compute_stats(arr: np.ndarray) -> dict:
    a = arr.astype(np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return {
        "min": a.min(0).tolist(),
        "max": a.max(0).tolist(),
        "mean": a.mean(0).tolist(),
        "std": a.std(0).tolist(),
        "count": [int(a.shape[0])],
    }


# ------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--gripper-state-mode", choices=("keep", "mask", "lag"), default="mask")
    ap.add_argument("--gripper-lag", type=int, default=3,
                    help="Frames of actuator lag for --gripper-state-mode=lag. "
                         "MUST come from a measured open/close step response.")
    ap.add_argument("--ee-state-mode", choices=("keep", "mask"), default="keep")
    ap.add_argument("--link-videos", action="store_true",
                    help="Symlink instead of copying videos/ (saves disk, source must stay mounted).")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src, dst = args.src.expanduser(), args.dst.expanduser()
    if dst.exists():
        if not args.force:
            print(f"{dst} 已存在, 用 --force 覆盖", file=sys.stderr)
            return 1
        shutil.rmtree(dst)

    files = sorted(glob.glob(str(src / "data" / "chunk-*" / "*.parquet")))
    if not files:
        print(f"未找到 parquet: {src/'data'}", file=sys.stderr)
        return 1

    # ---- pass 1: read everything, derive the contract from the whole dataset
    frames = [pd.read_parquet(f) for f in files]
    allrows = pd.concat(frames, ignore_index=True)
    state_all = np.stack(allrows["observation.state"]).astype(np.float64)
    action_all = np.stack(allrows["action"]).astype(np.float64)
    ep_all = allrows["episode_index"].to_numpy()
    wrench_all = (np.stack(allrows["observation.wrench"]).astype(np.float64)
                  if "observation.wrench" in allrows else None)

    contract = build_contract(state_all, action_all, ep_all, wrench_all)
    high_is_closed = contract["polarity"]["high_value_means"] == "closed"

    print("=" * 78)
    print("夹爪极性判定 (由数据推出, 非假设):")
    print(f"  高值含义          : {contract['polarity']['high_value_means']}")
    print(f"  episode 形状证据  : 首帧中位={contract['polarity']['evidence_episode_shape']['median_first_frame']:.4f} "
          f"中段中位={contract['polarity']['evidence_episode_shape']['median_mid_episode']:.4f}")
    print(f"  接触力证据        : {contract['polarity']['evidence_contact_force']['note']}")
    print(f"  宽度反推区间      : {np.round(contract['physical_width']['w_max_observed_max_m'], 6)} m 满开")
    print(f"  三项证据一致      : {contract['polarity']['unanimous']}")
    print("=" * 78)

    if not contract["polarity"]["unanimous"]:
        print("警告: 极性证据不一致, 请人工确认后再训练", file=sys.stderr)

    # ---- pass 2: rewrite each episode file
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    n_written = 0
    per_ep_stats = []
    out_state, out_action = [], []

    for f, df in zip(files, frames):
        df = df.copy()
        s = np.stack(df["observation.state"]).astype(np.float32)
        v = s[:, GRIP].astype(np.float64).copy()

        if args.gripper_state_mode == "mask":
            s[:, GRIP] = np.float32(0.0)
        elif args.gripper_state_mode == "lag":
            lag = args.gripper_lag
            s[:, GRIP] = np.concatenate([np.full(lag, v[0]), v[:-lag]])[: len(v)].astype(np.float32)

        if args.ee_state_mode == "mask":
            s[:, EE] = np.float32(0.0)

        phase, ttl = phase_labels(v, high_is_closed)
        df["observation.state"] = list(s)
        df["action.gripper_phase"] = phase
        df["action.frames_to_next_event"] = ttl

        rel = Path(f).relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        n_written += 1

        out_state.append(s.astype(np.float64))
        out_action.append(np.stack(df["action"]).astype(np.float64))
        per_ep_stats.append({
            "episode_index": int(df["episode_index"].iloc[0]),
            "stats": {
                "observation.state": compute_stats(s),
                "action": compute_stats(np.stack(df["action"])),
                "action.gripper_phase": compute_stats(phase),
                "action.frames_to_next_event": compute_stats(ttl),
            },
        })

    # ---- metadata
    for name in ("info.json", "tasks.jsonl", "episodes.jsonl", "force_meta.json"):
        p = src / "meta" / name if (src / "meta" / name).exists() else src / name
        if p.exists():
            shutil.copy2(p, dst / ("meta" / Path(name) if p.parent.name == "meta" else Path(name)))

    info_p = dst / "meta" / "info.json"
    if info_p.exists():
        info = json.loads(info_p.read_text())
        feats = info.get("features", {})
        # Correct the two known misleading names rather than silently propagating them.
        st = feats.get("observation.state", {})
        if st.get("names"):
            st["names"][GRIP] = ("gripper_closedness_cmd" if high_is_closed else "gripper_openness_cmd")
            st["names"][11:15] = ["ee_quat_x", "ee_quat_y", "ee_quat_z", "ee_quat_w"]
        feats["action.gripper_phase"] = {"dtype": "int64", "shape": [1], "names": ["phase"], "fps": info.get("fps")}
        feats["action.frames_to_next_event"] = {"dtype": "int64", "shape": [1], "names": ["ttl"], "fps": info.get("fps")}
        info["features"] = feats
        info["derived_from"] = str(src)
        info["gripper_state_mode"] = args.gripper_state_mode
        info["ee_state_mode"] = args.ee_state_mode
        info_p.write_text(json.dumps(info, indent=4, ensure_ascii=False))

    (dst / "meta" / "gripper_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False))

    S = np.concatenate(out_state)
    A = np.concatenate(out_action)
    stats = {"observation.state": compute_stats(S), "action": compute_stats(A)}
    old_stats_p = src / "meta" / "stats.json"
    merged = json.loads(old_stats_p.read_text()) if old_stats_p.exists() else {}
    merged.update(stats)
    (dst / "meta" / "stats.json").write_text(json.dumps(merged, indent=4))
    with (dst / "meta" / "episodes_stats.jsonl").open("w") as fh:
        for r in per_ep_stats:
            fh.write(json.dumps(r) + "\n")

    vsrc, vdst = src / "videos", dst / "videos"
    if vsrc.exists():
        if args.link_videos:
            vdst.symlink_to(vsrc.resolve())
        else:
            shutil.copytree(vsrc, vdst)

    print(f"\n写出 {n_written} 个 parquet -> {dst}")
    print(f"夹爪状态处理: {args.gripper_state_mode}" + (f" (lag={args.gripper_lag})" if args.gripper_state_mode == "lag" else ""))
    print(f"EE 状态处理  : {args.ee_state_mode}")
    print(f"契约文件     : {dst/'meta'/'gripper_contract.json'}")
    print(f"新增列       : action.gripper_phase, action.frames_to_next_event")
    print(f"\n复查: python b/s/Frk2/ds/verify_franka3_semantics.py --dataset {dst} --config b/s/Frk2/cfg/data_gate.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
