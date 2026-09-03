#!/usr/bin/env python3
"""Generic Layer-1 acceptance checks for a kptsim-injected LeRobot v2.1 dataset.

Mirrors the six checks in itrnVLA15_GeoP_3dtrj_3cn4_wrmup1G_scnObj.md §5, but all
paths and episode counts are parameterized so any RoboTwin 2.0 task can run it.

Does NOT fail on large adjacent-TCP jumps: scan_object ep42 (~0.125 m) is a
demonstration artifact, not an extraction bug.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NUM_KEYPOINTS = 14
FLAT_DIM = NUM_KEYPOINTS * 3  # 42
VOXEL_MIN = -0.01
VOXEL_MAX = 1.61
TCP_L, TCP_R = 6, 13
ADJACENT_TCP_WARN_M = 0.15
EPS = 1e-5

KEYPOINT_NAMES = [
    "fl_link1", "fl_link2", "fl_link3", "fl_link4", "fl_link5", "fl_link6", "fl_eef_tcp",
    "fr_link1", "fr_link2", "fr_link3", "fr_link4", "fr_link5", "fr_link6", "fr_eef_tcp",
]


def fail(ok: bool, msg: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    return ok


def parquet_path(root: Path, ep: int) -> Path:
    return root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"


def kptsim_path(kptsim: Path, ep: int) -> Path:
    return kptsim / f"episode_{ep:06d}" / "keypoints.npy"


def check_info(root: Path) -> bool:
    info = json.loads((root / "meta" / "info.json").read_text())
    feat = info.get("features", {}).get("observation.keypoint_3d", {})
    ok = True
    ok &= fail(bool(feat), "info.json 含 observation.keypoint_3d")
    ok &= fail(feat.get("dtype") == "float32", f"dtype={feat.get('dtype')} (expect float32)")
    ok &= fail(feat.get("shape") == [FLAT_DIM], f"shape={feat.get('shape')} (expect [{FLAT_DIM}])")
    names = feat.get("names") or []
    ok &= fail(
        bool(names) and names[0] == "fl_link1_x" and names[-1] == "fr_eef_tcp_z",
        f"names[0]={names[0] if names else None} names[-1]={names[-1] if names else None}",
    )
    ok &= fail(info.get("keypoint_coord_mode") == "voxel", f"coord_mode={info.get('keypoint_coord_mode')}")
    offset = info.get("keypoint_coord_offset") or []
    ok &= fail(len(offset) == 3, f"keypoint_coord_offset len={len(offset)}")
    return ok


def check_alignment(root: Path, kptsim: Path) -> bool:
    info = json.loads((root / "meta" / "info.json").read_text())
    n = int(info["total_episodes"])
    ok = True
    for ep in range(n):
        df = pd.read_parquet(parquet_path(root, ep))
        kpts = np.load(kptsim_path(kptsim, ep))
        if len(df) != kpts.shape[0]:
            ok &= fail(False, f"ep{ep} rows={len(df)} vs kptsim T={kpts.shape[0]}")
            return ok
        parquet_kpt = np.stack(df["observation.keypoint_3d"].tolist())
        if parquet_kpt.shape != kpts.shape:
            ok &= fail(False, f"ep{ep} parquet shape={parquet_kpt.shape} vs npy {kpts.shape}")
            return ok
        diff = float(np.max(np.abs(parquet_kpt.astype(np.float64) - kpts.astype(np.float64))))
        if diff > EPS:
            ok &= fail(False, f"ep{ep} max|parquet-npy|={diff:.2e}")
            return ok
    ok &= fail(True, f"{n}/{n} episode 行数与数值对齐 (atol={EPS})")
    return ok


def check_voxel_range(root: Path) -> bool:
    info = json.loads((root / "meta" / "info.json").read_text())
    n = int(info["total_episodes"])
    chunks = []
    for ep in range(n):
        df = pd.read_parquet(parquet_path(root, ep), columns=["observation.keypoint_3d"])
        chunks.append(np.stack(df["observation.keypoint_3d"].tolist()))
    k = np.concatenate(chunks, axis=0).reshape(-1, 3)
    ok = True
    ok &= fail(bool(np.isfinite(k).all()), "全部关键点有限")
    ok &= fail(float(k.min()) >= VOXEL_MIN, f"min={float(k.min()):.4f} (expect >= {VOXEL_MIN})")
    ok &= fail(float(k.max()) <= VOXEL_MAX, f"max={float(k.max()):.4f} (expect <= {VOXEL_MAX})")
    print(f"  [INFO] voxel xyz min={k.min(0).tolist()} max={k.max(0).tolist()}")
    return ok


def check_norm_stat(root: Path) -> bool:
    path = root / "norm_stat.json"
    ok = fail(path.is_file(), f"存在 {path}")
    if not ok:
        return False
    d = json.loads(path.read_text())
    ok &= fail("observation.state" in d and "action" in d, "键名为 observation.state / action")
    ok &= fail("state" not in d and "actions" not in d, "不含 GeoPredict 原始键 state / actions")
    mean = d.get("observation.state", {}).get("mean", [])
    ok &= fail(len(mean) == 14, f"state.mean dim={len(mean)} (expect 14)")
    stats_path = root / "meta" / "stats.json"
    if stats_path.is_file():
        ok &= fail(json.loads(stats_path.read_text()) == d, "meta/stats.json 与 norm_stat.json 一致")
    return ok


def check_provenance(root: Path, kptsim: Path, task: str) -> bool:
    dest_meta = json.loads((root / "meta" / "keypoints_meta.json").read_text())
    src_meta = json.loads((kptsim / "keypoints_meta.json").read_text())
    ok = True
    ok &= fail(dest_meta.get("K") == NUM_KEYPOINTS, f"dest K={dest_meta.get('K')}")
    ok &= fail(src_meta.get("K") == NUM_KEYPOINTS, f"src K={src_meta.get('K')}")
    ok &= fail(dest_meta.get("keypoint_names") == KEYPOINT_NAMES, "keypoint_names 顺序")
    names = dest_meta.get("keypoint_names") or []
    ok &= fail(len(names) > 6 and names[6] == "fl_eef_tcp", "TCP 为 fl_eef_tcp / fr_eef_tcp")
    a = np.asarray(dest_meta["coord_offset"], dtype=np.float64)
    b = np.asarray(src_meta["coord_offset"], dtype=np.float64)
    ok &= fail(np.allclose(a, b, atol=1e-8), f"coord_offset dest={a.tolist()} src={b.tolist()}")
    ds = str(dest_meta.get("dataset_dir", "")).rstrip("/")
    ok &= fail(ds.endswith(task), f"dataset_dir 以任务名结尾: {ds}")
    print(f"  [INFO] coord_offset={a.tolist()}")
    return ok


def check_original_columns(root: Path) -> bool:
    info = json.loads((root / "meta" / "info.json").read_text())
    df = pd.read_parquet(parquet_path(root, 0))
    required = [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "observation.keypoint_3d",
    ]
    ok = True
    for col in required:
        ok &= fail(col in df.columns, f"parquet 含 {col}")
    state = np.stack(df["observation.state"].tolist())
    ok &= fail(state.shape[1] == 14, f"state dim={state.shape[1]} (expect 14)")
    kpt = np.stack(df["observation.keypoint_3d"].tolist())
    ok &= fail(kpt.shape[1] == FLAT_DIM, f"keypoint_3d dim={kpt.shape[1]} (expect {FLAT_DIM})")
    feats = info.get("features", {})
    ok &= fail("observation.images.cam_high" in feats, "相机键 observation.images.cam_high")
    n = int(info["total_episodes"])
    ok &= fail(n >= 1, f"total_episodes={n}")
    return ok


def warn_tcp_continuity(root: Path) -> None:
    info = json.loads((root / "meta" / "info.json").read_text())
    n = int(info["total_episodes"])
    worst_ep, worst = -1, 0.0
    for ep in range(n):
        df = pd.read_parquet(parquet_path(root, ep), columns=["observation.keypoint_3d"])
        arr = np.stack(df["observation.keypoint_3d"].tolist()).reshape(-1, NUM_KEYPOINTS, 3)
        if arr.shape[0] < 2:
            continue
        d = np.linalg.norm(np.diff(arr[:, [TCP_L, TCP_R], :], axis=0), axis=-1)
        m = float(d.max()) if d.size else 0.0
        if m > worst:
            worst, worst_ep = m, ep
    mark = "WARN" if worst > ADJACENT_TCP_WARN_M else "INFO"
    print(
        f"  [{mark}] 相邻 TCP 最大跳变={worst:.4f} (体素单位) ep{worst_ep}; "
        f">{ADJACENT_TCP_WARN_M} 仅告警, 不判失败 (scan_object ep42 类演示轨迹)"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest-root", required=True)
    parser.add_argument("--kptsim-root", required=True)
    parser.add_argument("--task", required=True, help="RoboTwin task name, e.g. scan_object")
    args = parser.parse_args()
    dest = Path(args.dest_root)
    kptsim = Path(args.kptsim_root)

    print(f"== Layer-1 验收 dest={dest}")
    print(f"== kptsim={kptsim} task={args.task}")
    checks = [
        ("Check 1 info.json feature", lambda: check_info(dest)),
        ("Check 2 行数对齐 + 数值匹配", lambda: check_alignment(dest, kptsim)),
        ("Check 3 体素盒值域", lambda: check_voxel_range(dest)),
        ("Check 4 norm_stat 键名", lambda: check_norm_stat(dest)),
        ("Check 5 溯源 meta", lambda: check_provenance(dest, kptsim, args.task)),
        ("Check 6 原列完整", lambda: check_original_columns(dest)),
    ]
    all_ok = True
    for name, fn in checks:
        print(f"\n-- {name}")
        all_ok &= fn()
    print("\n-- TCP 连续性 (告警, 不纳入 PASS/FAIL)")
    warn_tcp_continuity(dest)
    print("\n" + ("ALL PASS" if all_ok else "FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
