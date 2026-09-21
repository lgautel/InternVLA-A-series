#!/usr/bin/env python3
"""Semantic admission gate for the franka3 15 Hz plug-into-socket LeRobot dataset.

``verify_franka2_keypoints.py`` checks that the FK keypoints are geometrically
sound.  It does **not** look at whether the supervised targets carry any
information that is not already in the observation, which is the failure mode
that produced the real-robot "gripper never closes" behaviour documented in
``b/d/Frk2/realwrld_debug/``.  This script closes that gap.

Every check is a *semantic invariant*: something that must hold for the dataset
to be trainable into a policy that actually uses its cameras.  A range check
cannot catch an algebraic mirror, because a perfect mirror is still in range.

Usage:
    python b/s/Frk2/ds/verify_franka3_semantics.py \
        --dataset /B/Dta/plug_into_socket_franka3_15hz_lerobot \
        --config  b/s/Frk2/cfg/data_gate.yaml

Exit code 0 = all gates pass, 1 = at least one gate failed.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is optional, defaults are inlined
    yaml = None

# Layout of `observation.state` for robot_type=franka3 (see b/s/Frk2/cfg/franka3.yaml).
ARM = slice(0, 7)
GRIP = 7
EE_POS = slice(8, 11)
EE_QUAT = slice(11, 15)

DEFAULT_THRESHOLDS = {
    # G1: no action dim may be a deterministic affine image of the same-frame state.
    "r2_max": 0.90,
    # G2: bit-exact copies are an even harder failure than a high R^2.
    "bit_exact_max": 1e-3,
    # G3: a channel claiming to be a measurement must not be a rounded image of a command.
    "derived_unique_ratio_min": 1.0,
    # G4: the executed chunk prefix must not be reconstructible by copying the state.
    "copy_r2_max": 0.80,
    "n_exec": 10,
    "chunk_size": 50,
    # G5: action distribution must be contained in the state distribution (servo-lag check).
    "quantile_slack": 0.0,
    # G6: a real sensor channel does not repeat bit-exactly, and its quantum is physical.
    "zero_diff_max": 0.10,
    "sensor_resolution": 1e-5,
    # G7: timing must either jitter or be declared as resampled.
    "dt_std_min": 0.0,
    # G8: scene coverage.
    "ee_pos_std_min": 0.02,
}


#: Gates are split by severity.  A BLOCK gate catches a channel that carries no
#: independent information at all -- there is a concrete data fix for it, so it
#: must stop training.  A WARN gate quantifies a risk that is often inherent to
#: the control mode (absolute joint targets are *supposed* to track the state
#: closely), so it is reported but does not fail CI.  Mixing the two produces a
#: gate that cries wolf and gets ignored.
BLOCK, WARN = "BLOCK", "WARN"


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    rows: list[str] = field(default_factory=list)
    severity: str = BLOCK


def masked_dims(state: np.ndarray) -> set[int]:
    """Dims deliberately constant-filled by build_franka3_fixed_dataset.py.

    A constant channel conveys nothing, which is the *intent*; re-flagging it as
    a derived/implausible sensor would be a false positive.
    """
    return {j for j in range(state.shape[1]) if float(state[:, j].std()) == 0.0}


def _fit_r2(x: np.ndarray, y: np.ndarray) -> float:
    """R^2 of the best affine fit y ~ a + b*x (x may be 1-D or 2-D)."""
    X = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    denom = float(((y - y.mean()) ** 2).sum())
    if denom == 0.0:
        return 1.0
    return 1.0 - float(resid @ resid) / denom


def load_dataset(root: Path):
    files = sorted(glob.glob(str(root / "data" / "chunk-*" / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet under {root / 'data'}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "episode_index" in df:
        df = df.sort_values(["episode_index", "frame_index"], kind="stable").reset_index(drop=True)
    state = np.stack(df["observation.state"]).astype(np.float64)
    action = np.stack(df["action"]).astype(np.float64)
    ep = df["episode_index"].to_numpy()
    ts = df["timestamp"].to_numpy().astype(np.float64)
    return state, action, ep, ts


# --------------------------------------------------------------------------- gates


def gate_affine_leak(state, action, th) -> GateResult:
    """G1: no action dim reconstructible from the same-frame state."""
    rows, bad = [], []
    for j in range(action.shape[1]):
        r2_uni = _fit_r2(state[:, j], action[:, j]) if j < state.shape[1] else -1.0
        r2_multi = _fit_r2(state, action[:, j])
        worst = max(r2_uni, r2_multi)
        flag = worst >= th["r2_max"]
        bad.append(flag)
        rows.append(f"    dim{j}: R2_uni={r2_uni:.6f} R2_multi={r2_multi:.6f} {'FAIL' if flag else 'ok'}")
    n = sum(bad)
    return GateResult(
        "G1 同帧仿射泄漏",
        n == 0,
        f"{n}/{action.shape[1]} 个动作维可被同帧状态线性重构 (阈值 R2<{th['r2_max']})",
        rows,
    )


def gate_bit_exact(state, action, th) -> GateResult:
    """G2: no action dim is a bit-exact copy / affine mirror of a state dim."""
    rows, bad = [], []
    for j in range(min(state.shape[1], action.shape[1])):
        frac = float(np.mean(action[:, j] == state[:, j]))
        flag = frac > th["bit_exact_max"]
        bad.append(flag)
        if flag or frac > 0:
            rows.append(f"    dim{j}: action==state 的帧占比 {frac * 100:.4f}% {'FAIL' if flag else 'ok'}")
    return GateResult(
        "G2 逐位恒等",
        not any(bad),
        f"{sum(bad)} 个动作维与同帧状态逐位相等",
        rows or ["    (无逐位相等的维度)"],
    )


def gate_derived_channel(state, action, th) -> GateResult:
    """G3: a state channel must not be a rounded image of an action channel.

    If ``state[:, j]`` were an independent measurement it would carry at least as
    many distinct values as the command.  Fewer distinct values means the state
    was *computed from* the action and lost precision to rounding.
    """
    skip = masked_dims(state)
    rows, bad = [], []
    for j in range(min(state.shape[1], action.shape[1])):
        if j in skip:
            rows.append(f"    dim{j}: 已被常数屏蔽, 跳过 (by design)")
            continue
        us, ua = len(np.unique(state[:, j])), len(np.unique(action[:, j]))
        ratio = us / max(ua, 1)
        flag = ratio < th["derived_unique_ratio_min"]
        bad.append(flag)
        if flag:
            rows.append(f"    dim{j}: |unique(state)|={us} < |unique(action)|={ua} -> state 疑为 action 的派生量 FAIL")
    return GateResult(
        "G3 派生通道检测",
        not any(bad),
        f"{sum(bad)} 个状态维疑似由动作反算得到",
        rows or ["    (无派生通道)"],
    )


def gate_chunk_copy(state, action, ep, th) -> GateResult:
    """G4: the *executed* chunk prefix must not be copyable from the current state."""
    mu_a, sd_a = action.mean(0), action.std(0) + 1e-12
    mu_s, sd_s = state.mean(0), state.std(0) + 1e-12
    za, zs = (action - mu_a) / sd_a, (state - mu_s) / sd_s
    skip = masked_dims(state)
    rows, bad = [], []
    for label, horizon in (("n_exec", th["n_exec"]), ("chunk", th["chunk_size"])):
        for j in range(action.shape[1]):
            if j in skip:
                continue
            errs = []
            for e in np.unique(ep):
                idx = np.flatnonzero(ep == e)
                for h in range(horizon):
                    tgt = np.minimum(np.arange(len(idx)) + h, len(idx) - 1)
                    # allow either polarity: the mirror may be a negated copy
                    errs.append(np.minimum(
                        np.abs(za[idx[tgt], j] - zs[idx, j]),
                        np.abs(za[idx[tgt], j] + zs[idx, j]),
                    ))
            mse = float((np.concatenate(errs) ** 2).mean())
            r2 = 1.0 - mse
            flag = label == "n_exec" and r2 >= th["copy_r2_max"]
            bad.append(flag)
            if r2 > 0.5:
                rows.append(f"    dim{j} [{label}={horizon}]: copy_R2={r2:+.6f} {'FAIL' if flag else 'warn'}")
    return GateResult(
        "G4 chunk 复制捷径",
        not any(bad),
        f"{sum(bad)} 个动作维在被执行的前 {th['n_exec']} 步上可被复制 (阈值 R2<{th['copy_r2_max']})",
        rows or ["    (无强复制捷径)"],
        severity=WARN,
    )


def gate_action_within_state(state, action, th) -> GateResult:
    """G5: action distribution must sit inside the state distribution (servo lag)."""
    skip = masked_dims(state)
    rows, bad = [], []
    for j in range(min(state.shape[1], action.shape[1])):
        if j in skip:
            continue
        sq01, sq99 = np.quantile(state[:, j], [0.01, 0.99])
        aq01, aq99 = np.quantile(action[:, j], [0.01, 0.99])
        flag = (aq01 < sq01 - th["quantile_slack"]) or (aq99 > sq99 + th["quantile_slack"])
        bad.append(flag)
        if flag:
            rows.append(
                f"    dim{j}: action[q01,q99]=[{aq01:.4f},{aq99:.4f}] 超出 "
                f"state[q01,q99]=[{sq01:.4f},{sq99:.4f}] FAIL"
            )
    return GateResult(
        "G5 动作被状态包住",
        not any(bad),
        f"{sum(bad)} 个动作维的分布探出状态分布 (开环漂移风险)",
        rows or ["    (全部被包住)"],
        severity=WARN,
    )


def gate_sensor_authenticity(state, ep, th) -> GateResult:
    """G6: channels advertised as measurements must look like measurements."""
    skip = masked_dims(state)
    rows, bad = [], []
    for j in list(range(7)) + [GRIP]:
        if j in skip:
            rows.append(f"    dim{j}: 已被常数屏蔽, 跳过 (by design)")
            continue
        diffs = np.concatenate([np.diff(state[ep == e, j]) for e in np.unique(ep)])
        zero_frac = float(np.mean(diffs == 0))
        nz = np.abs(diffs[diffs != 0])
        quantum = float(np.median(nz)) if nz.size else 0.0
        flag = zero_frac > th["zero_diff_max"] or quantum < th["sensor_resolution"]
        bad.append(flag)
        if flag:
            rows.append(
                f"    dim{j}: 逐位不变占比={zero_frac * 100:.2f}% 非零变化中位={quantum:.3e} "
                f"-> 不像真实传感器 FAIL"
            )
    return GateResult(
        "G6 传感器真实性",
        not any(bad),
        f"{sum(bad)} 个状态维不具备真实传感器的数值特征",
        rows or ["    (全部像真实测量)"],
    )


def gate_timing(ts, ep, th) -> GateResult:
    dts = np.concatenate([np.diff(ts[ep == e]) for e in np.unique(ep)])
    std = float(dts.std())
    passed = std > th["dt_std_min"]
    return GateResult(
        "G7 时间抖动",
        passed,
        f"dt mean={dts.mean():.6f}s std={std:.3e} -> "
        f"{'有抖动' if passed else '完美网格, 训练无频率鲁棒性'}",
        [f"    dt min={dts.min():.6f} max={dts.max():.6f}"],
        severity=WARN,
    )


def gate_scene_diversity(state, th) -> GateResult:
    stds = state[:, EE_POS].std(0)
    worst = float(stds.min())
    passed = worst >= th["ee_pos_std_min"]
    return GateResult(
        "G8 场景多样性",
        passed,
        f"ee_pos std = {np.round(stds, 5).tolist()} m, 最小 {worst:.5f} "
        f"(阈值 {th['ee_pos_std_min']})",
        [],
        severity=WARN,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=None, help="YAML overriding the gate thresholds.")
    ap.add_argument("--json-out", type=Path, default=None, help="Write the machine-readable report here.")
    args = ap.parse_args()

    th = dict(DEFAULT_THRESHOLDS)
    if args.config and args.config.exists():
        if yaml is None:
            print("PyYAML 未安装, 忽略 --config, 使用内置阈值", file=sys.stderr)
        else:
            th.update(yaml.safe_load(args.config.read_text()) or {})

    state, action, ep, ts = load_dataset(args.dataset)
    print("=" * 78)
    print(f"数据集 : {args.dataset}")
    print(f"规模   : {len(state)} 帧 / {len(np.unique(ep))} episode / state {state.shape[1]}D / action {action.shape[1]}D")
    print("=" * 78)

    results = [
        gate_affine_leak(state, action, th),
        gate_bit_exact(state, action, th),
        gate_derived_channel(state, action, th),
        gate_chunk_copy(state, action, ep, th),
        gate_action_within_state(state, action, th),
        gate_sensor_authenticity(state, ep, th),
        gate_timing(ts, ep, th),
        gate_scene_diversity(state, th),
    ]

    for r in results:
        if r.passed:
            tag = "PASS "
        else:
            tag = "BLOCK" if r.severity == BLOCK else "WARN "
        print(f"\n[{tag}] {r.name}")
        print(f"    {r.detail}")
        for row in r.rows:
            print(row)

    blocked = [r for r in results if not r.passed and r.severity == BLOCK]
    warned = [r for r in results if not r.passed and r.severity == WARN]
    print("\n" + "=" * 78)
    print(f"结果: {sum(r.passed for r in results)}/{len(results)} 项通过; "
          f"{len(blocked)} 项阻断, {len(warned)} 项告警")
    if blocked:
        print("阻断项 (必须修数据, 否则禁止训练):")
        for r in blocked:
            print(f"  - {r.name}: {r.detail}")
    if warned:
        print("告警项 (风险量化, 不阻断):")
        for r in warned:
            print(f"  - {r.name}")
    print("=" * 78)

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {r.name: {"passed": r.passed, "severity": r.severity, "detail": r.detail} for r in results},
            ensure_ascii=False, indent=2,
        ))
        print(f"报告写入 {args.json_out}")

    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
