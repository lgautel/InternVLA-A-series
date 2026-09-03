# 方案 E1 实施手册: elevator0714 数据集 7D 关键点生成与训练适配

> **目标**: 将 `/home/luogang/DATA/elevator0714_lerobot/`（原始数据, 只读）处理为带 7D 关键点 (位置 3D + 四元数姿态 4D) 的训练数据集, 保存到 `/home/luogang/DATA/elevator0714_lerobot_4D/`.
>
> **前置文档**: [dta_3dtrj_E2.md](dta_3dtrj_E2.md) — E1 方案设计原理与架构
>
> **读者假设**: 无需事先了解本代码库. 按照本文的步骤顺序操作即可完成全部数据处理与验收.
>
> **撰写日**: 2026-09-03

---

## 目录

1. [环境准备](#1-环境准备)
2. [原始数据集检查 (只读)](#2-原始数据集检查-只读)
3. [生成脚本开发: generate_r1pro_keypoints_e1.py](#3-生成脚本开发-generate_r1pro_keypoints_e1py)
4. [执行关键点生成](#4-执行关键点生成)
5. [生成后数据验证](#5-生成后数据验证)
6. [norm_stats 统计量生成](#6-norm_stats-统计量生成)
7. [模型侧代码改动](#7-模型侧代码改动)
8. [Smoke Test: 端到端数据流验证](#8-smoke-test-端到端数据流验证)
9. [完整文件清单与 diff](#9-完整文件清单与-diff)
10. [故障排查手册](#10-故障排查手册)

---

## 1. 环境准备

### 1.1 所需环境

本项目使用 conda 环境 `itvlaGp`. 关键点生成脚本依赖 Pinocchio (C++ FK 库).

```bash
# 激活环境
conda activate itvlaGp

# 确认 pinocchio 可用 (版本 >= 2.6)
python -c "import pinocchio as pin; print(f'pinocchio {pin.__version__}, nq test: {pin.buildModelFromUrdf.__doc__[:20]}...')"

# 确认 pandas, numpy, torch 可用
python -c "import pandas, numpy, torch; print(f'pandas={pandas.__version__}, numpy={numpy.__version__}, torch={torch.__version__}')"
```

若 pinocchio 未安装或版本过低:

```bash
conda install -c conda-forge pinocchio -y
```

### 1.2 路径约定

| 变量 | 路径 | 说明 |
|------|------|------|
| `SOURCE` | `/home/luogang/DATA/elevator0714_lerobot` | 原始数据集 (只读, **绝不修改**) |
| `DEST` | `/home/luogang/DATA/elevator0714_lerobot_4D` | 带 7D 关键点的输出数据集 |
| `REPO_ROOT` | `/home/luogang/SRC/Robot/itvlaGp` | 代码库根目录 |
| `URDF` | `$REPO_ROOT/assets/r1_pro_with_gripper.urdf` | R1 Pro URDF 文件 |
| `HF_LEROBOT_HOME` | 由训练框架读取, 需包含 `DEST` | 训练时定位数据集的根路径 |

### 1.3 磁盘空间检查

```bash
# 原始数据集大小
du -sh /home/luogang/DATA/elevator0714_lerobot
# 预期: ~300 MB (100 MB data + 200 MB video)

# 目标目录可用空间 (至少需要 500 MB)
df -h /home/luogang/DATA/
```

输出数据集会比原始数据集稍大 (每帧增加 112 × 4 = 448 字节的关键点数据, 27,145 帧共约 12 MB 增量, parquet 压缩后更小).

---

## 2. 原始数据集检查 (只读)

在生成关键点之前, 先验证原始数据集的完整性和格式.

### 2.1 运行 precheck 脚本

```bash
cd /home/luogang/SRC/Robot/itvlaGp

python util_scripts/precheck_r1pro_dataset.py \
    --dataset /home/luogang/DATA/elevator0714_lerobot
```

**预期输出要点** (若不符则停止, 排查后再继续):

| 检查项 | 预期 | 不符时动作 |
|--------|------|-----------|
| `robot_type` | `r1_pro` | 确认数据集路径正确 |
| `total_episodes` | `100` | — |
| `total_frames` | `27145` | — |
| `fps` | `15` | — |
| `observation.state.left_arm` shape | `[7]` | 数据集 schema 不匹配, 检查 codebase_version |
| `observation.state.right_arm` shape | `[7]` | 同上 |
| right arm std | 非零 (应 >0.1) | 若全零则退回单臂 J=8 |
| `observation.state.torso` std | 全零 | 若非零, 需要重新评估 FK torso_q 设定 |
| `action.chassis.velocities` std | 全零 | 若非零, 需确认底盘运动策略 |

### 2.2 手动抽检 parquet 内容

```bash
python -c "
import pandas as pd, numpy as np

pq = pd.read_parquet('/home/luogang/DATA/elevator0714_lerobot/data/chunk-000/file-001.parquet')
print(f'行数: {len(pq)}')
print(f'列: {list(pq.columns)}')
print()

# 确认 observation.keypoint_3d 尚不存在 (原始数据不含关键点)
assert 'observation.keypoint_3d' not in pq.columns, '原始数据集已包含关键点列, 请确认路径!'
print('✓ 原始数据集不含 observation.keypoint_3d (预期)')

# 检查关节角格式
left = np.stack(pq['observation.state.left_arm'].values)
right = np.stack(pq['observation.state.right_arm'].values)
print(f'left_arm shape: {left.shape}, dtype: {left.dtype}')
print(f'right_arm shape: {right.shape}, dtype: {right.dtype}')
print(f'left_arm[0]: {left[0]}')
print(f'right_arm[0]: {right[0]}')

# 确认无 NaN
assert not np.isnan(left).any(), 'left_arm 含 NaN!'
assert not np.isnan(right).any(), 'right_arm 含 NaN!'
print('✓ 无 NaN')
"
```

---

## 3. 生成脚本开发: generate_r1pro_keypoints_e1.py

### 3.1 脚本概览

新建脚本 `util_scripts/generate_r1pro_keypoints_e1.py`, 基于现有 `generate_r1pro_keypoints.py` 修改. 核心变化:

| 维度 | 原脚本 (方案 E, voxel 归一化) | E1 脚本 (方案 E1, isotropic + 四元数) |
|------|-----|------|
| 每关键点维度 | 3 (仅位置) | **7** (位置 3 + 四元数 4) |
| 归一化方式 | voxel 空间: `kpts - offset`, 值域 [0, 1.6]×[0, 1.6]×[0, 1.0] | **isotropic**: `pos / R_pad`, 值域 ≈[-1, 1]³; 四元数半球归一化 |
| 总维度 | 16×3 = 48 | 16×7 = **112** |
| Pass 1 输出 | `offset` (3D 向量) | `R_pad` (标量) + `global_min/max` |
| meta 记录 | `coord_offset`, `voxel_center`, `voxel_bounds` | `bbox_radius`, `bbox_margin`, `keypoint_dim=7` |

### 3.2 完整脚本

在 `util_scripts/generate_r1pro_keypoints_e1.py` 创建以下内容:

```python
"""Offline FK-based 7D keypoint generation for R1 Pro — E1 scheme (position + quaternion).

Two-pass pipeline:
  Pass 1: FK all frames → collect global position bounding box → compute R_pad
  Pass 2: position / R_pad, quaternion hemisphere-normalize → write parquet

Output: observation.keypoint_3d [112] = 16 keypoints × 7D (px,py,pz,qx,qy,qz,qw).

Position normalization: isotropic division by R_pad = max(|all axis extremes|) × (1 + margin).
Quaternion normalization: hemisphere constraint (qw >= 0).

See dta_3dtrj_E2.md for design rationale.

Usage:
    python util_scripts/generate_r1pro_keypoints_e1.py \
        --source /home/luogang/DATA/elevator0714_lerobot \
        --dest /home/luogang/DATA/elevator0714_lerobot_4D

TORSO CONVENTION: same as generate_r1pro_keypoints.py — see that file's docstring.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 16
KEYPOINT_DIM = 7  # 3 (pos) + 4 (quat xyzw)
BBOX_MARGIN = 0.15  # 15% safety margin for R_pad

KEYPOINT_LINKS: list[str] = [
    "left_arm_link1", "left_arm_link2", "left_arm_link3", "left_arm_link4",
    "left_arm_link5", "left_arm_link6", "left_arm_link7", "left_gripper_link",
    "right_arm_link1", "right_arm_link2", "right_arm_link3", "right_arm_link4",
    "right_arm_link5", "right_arm_link6", "right_arm_link7", "right_gripper_link",
]

LEFT_ARM_JOINTS = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"right_arm_joint{i}" for i in range(1, 8)]
TORSO_JOINTS = [f"torso_joint{i}" for i in range(1, 5)]

TORSO_Q_DEFAULT = (0.0, 0.0, 0.0, 0.0)
TORSO_Q_PHYSICAL = (0.8, -1.4, -0.60, 0.0)

KEYPOINT_FEATURE_NAMES: list[str] = [
    f"{link}_{comp}"
    for link in KEYPOINT_LINKS
    for comp in ("px", "py", "pz", "qx", "qy", "qz", "qw")
]


class R1ProFKExtractorE1:
    """FK extractor producing 7D keypoints: [px, py, pz, qx, qy, qz, qw] per link."""

    def __init__(self, urdf_path: str, torso_q=TORSO_Q_DEFAULT):
        import pinocchio as pin
        self._pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.frame_ids = [self.model.getFrameId(name) for name in KEYPOINT_LINKS]
        for i, fid in enumerate(self.frame_ids):
            if fid >= self.model.nframes:
                raise ValueError(f"Frame '{KEYPOINT_LINKS[i]}' not found in URDF")

        self._left_idx_q = []
        self._right_idx_q = []
        self._torso_idx_q = []
        for joints_list, idx_list, label in [
            (LEFT_ARM_JOINTS, self._left_idx_q, "left_arm"),
            (RIGHT_ARM_JOINTS, self._right_idx_q, "right_arm"),
            (TORSO_JOINTS, self._torso_idx_q, "torso"),
        ]:
            for jname in joints_list:
                jid = self.model.getJointId(jname)
                nq = self.model.joints[jid].nq
                if nq != 1:
                    raise ValueError(
                        f"Joint '{jname}' ({label}) has nq={nq}, expected 1."
                    )
                idx_list.append(self.model.joints[jid].idx_q)

        self.torso_q = np.asarray(torso_q, dtype=np.float64)
        if self.torso_q.shape != (4,):
            raise ValueError(f"torso_q must have 4 entries, got {self.torso_q.shape}")
        self._q_base = pin.neutral(self.model)
        for idx_q, angle in zip(self._torso_idx_q, self.torso_q, strict=True):
            self._q_base[idx_q] = float(angle)

    def compute(self, left_arm: np.ndarray, right_arm: np.ndarray) -> np.ndarray:
        """Compute 16 keypoints as 7D (position + quaternion) in base_link frame.

        Returns: [16, 7] float32 — [px, py, pz, qx, qy, qz, qw] per keypoint.
        Quaternions are hemisphere-normalized (qw >= 0).
        """
        pin = self._pin
        q = self._q_base.copy()
        for idx_q, angle in zip(self._left_idx_q, left_arm, strict=True):
            q[idx_q] = float(angle)
        for idx_q, angle in zip(self._right_idx_q, right_arm, strict=True):
            q[idx_q] = float(angle)

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        keypoints = np.empty((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
        for i, fid in enumerate(self.frame_ids):
            oMf = self.data.oMf[fid]
            # Position: 3D
            keypoints[i, :3] = oMf.translation
            # Orientation: quaternion [qx, qy, qz, qw]
            quat = pin.Quaternion(oMf.rotation)
            raw_q = np.array([quat.x(), quat.y(), quat.z(), quat.w()],
                             dtype=np.float32)
            # Hemisphere normalization: ensure qw >= 0
            if raw_q[3] < 0:
                raw_q = -raw_q
            keypoints[i, 3:7] = raw_q
        return keypoints

    def compute_batch(self, left_arms: np.ndarray, right_arms: np.ndarray) -> np.ndarray:
        """[N, 7], [N, 7] -> [N, 16, 7]."""
        n = left_arms.shape[0]
        out = np.empty((n, NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
        for i in range(n):
            out[i] = self.compute(left_arms[i], right_arms[i])
        return out


def _get_parquet_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")
    return files


def _read_joint_angles(df: pd.DataFrame, pq_path: Path | None = None):
    if len(df) == 0:
        logger.warning("Skipping empty parquet (0 rows): %s", pq_path)
        return None, None
    for col in ("observation.state.left_arm", "observation.state.right_arm"):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in {pq_path}.")
    left = np.stack(df["observation.state.left_arm"].values).astype(np.float64)
    right = np.stack(df["observation.state.right_arm"].values).astype(np.float64)
    for name, arr in [("left_arm", left), ("right_arm", right)]:
        if np.any(np.isnan(arr)):
            raise ValueError(f"NaN in observation.state.{name} in {pq_path}")
    return left, right


def _read_recorded_torso(df: pd.DataFrame, pq_path: Path) -> np.ndarray | None:
    col = "observation.state.torso"
    if col not in df.columns:
        return None
    torso = np.stack(df[col].to_numpy()).astype(np.float64)
    spread = float(np.abs(torso - torso[0]).max())
    if spread > 1e-6:
        raise ValueError(
            f"{col} varies within {pq_path} (max deviation {spread:.6f} rad)."
        )
    return torso[0]


def pass1_compute_bbox(parquet_files: list[Path], extractor: R1ProFKExtractorE1):
    """Pass 1: FK all frames, collect global position min/max (quaternion is self-normalized)."""
    global_min = np.full(3, np.inf, dtype=np.float64)
    global_max = np.full(3, -np.inf, dtype=np.float64)
    total_frames = 0
    recorded_torso = None

    # Also collect quaternion statistics for logging
    qw_min, qw_max = 1.0, 0.0
    quat_norm_err_max = 0.0

    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path)
        left, right = _read_joint_angles(df, pq_path)
        if left is None:
            continue
        torso = _read_recorded_torso(df, pq_path)
        if torso is not None:
            if recorded_torso is not None and not np.allclose(recorded_torso, torso, atol=1e-6):
                raise ValueError(f"Torso pose differs between parquet files")
            recorded_torso = torso

        kpts = extractor.compute_batch(left, right)  # [N, 16, 7]
        pos = kpts[:, :, :3]  # [N, 16, 3]
        quat = kpts[:, :, 3:7]  # [N, 16, 4]

        # Position bounding box
        frame_min = pos.reshape(-1, 3).min(axis=0)
        frame_max = pos.reshape(-1, 3).max(axis=0)
        global_min = np.minimum(global_min, frame_min)
        global_max = np.maximum(global_max, frame_max)

        # Quaternion sanity checks
        qw_vals = quat[:, :, 3]  # qw component
        qw_min = min(qw_min, float(qw_vals.min()))
        qw_max = max(qw_max, float(qw_vals.max()))
        quat_norms = np.linalg.norm(quat.reshape(-1, 4), axis=1)
        quat_norm_err_max = max(quat_norm_err_max, float(np.abs(quat_norms - 1.0).max()))

        total_frames += len(df)
        logger.info("  Pass 1: %s — %d frames (total: %d)", pq_path.name, len(df), total_frames)

    if recorded_torso is not None and not np.allclose(recorded_torso, extractor.torso_q, atol=1e-6):
        logger.warning(
            "Dataset records torso=%s but FK used torso_q=%s. See generate_r1pro_keypoints.py docstring.",
            np.array2string(recorded_torso, precision=4),
            np.array2string(extractor.torso_q, precision=4),
        )

    logger.info("Quaternion stats: qw_min=%.6f, qw_max=%.6f, norm_err_max=%.2e",
                qw_min, qw_max, quat_norm_err_max)
    if qw_min < 0:
        logger.error("FATAL: qw_min=%.6f < 0, hemisphere normalization failed!", qw_min)
        raise RuntimeError("Hemisphere normalization failed")
    if quat_norm_err_max > 0.01:
        logger.error("FATAL: quaternion norm error %.4f > 0.01, FK or Pinocchio issue!", quat_norm_err_max)
        raise RuntimeError("Quaternion norm check failed")

    return global_min.astype(np.float32), global_max.astype(np.float32), total_frames


def compute_r_pad(global_min: np.ndarray, global_max: np.ndarray, margin: float = BBOX_MARGIN) -> float:
    """Isotropic bounding radius with safety margin.

    R = max(|x_min|, x_max, |y_min|, y_max, |z_min|, z_max)
    R_pad = R × (1 + margin)
    """
    abs_extremes = np.maximum(np.abs(global_min), np.abs(global_max))
    R = float(abs_extremes.max())
    R_pad = R * (1.0 + margin)
    return R_pad


def pass2_write_keypoints(parquet_files: list[Path], extractor: R1ProFKExtractorE1,
                          r_pad: float, source_data_dir: Path, dest: Path):
    """Pass 2: FK → normalize → write observation.keypoint_3d [112] into parquet."""
    total_frames = 0
    oob_count = 0
    quat_err_count = 0

    for pq_path in parquet_files:
        rel = pq_path.relative_to(source_data_dir)
        dest_pq = dest / "data" / rel
        df = pd.read_parquet(dest_pq)
        left, right = _read_joint_angles(df, dest_pq)
        if left is None:
            continue

        kpts = extractor.compute_batch(left, right)  # [N, 16, 7]

        # Normalize position: divide by R_pad
        kpts[:, :, :3] /= r_pad

        # Position boundary check
        pos_oob = (np.abs(kpts[:, :, :3]) > 1.01).any()
        if pos_oob:
            oob_count += 1
            logger.warning("  Position OOB in %s: max |pos| = %.4f",
                           dest_pq.name, np.abs(kpts[:, :, :3]).max())

        # Quaternion unit norm check
        quat_norms = np.linalg.norm(kpts[:, :, 3:7].reshape(-1, 4), axis=1)
        quat_err = np.abs(quat_norms - 1.0).max()
        if quat_err > 0.01:
            quat_err_count += 1
            logger.warning("  Quaternion norm error in %s: max = %.6f", dest_pq.name, quat_err)

        # Write: flatten [16, 7] → [112]
        df["observation.keypoint_3d"] = [row.reshape(-1) for row in kpts]
        df.to_parquet(dest_pq)
        total_frames += len(df)
        logger.info("  Pass 2: %s — %d frames written", dest_pq.name, len(df))

    if oob_count:
        logger.warning("%d parquet files had OOB positions. Increase BBOX_MARGIN.", oob_count)
    if quat_err_count:
        logger.warning("%d parquet files had quaternion norm errors.", quat_err_count)
    return total_frames


def _copy_dataset(source: Path, dest: Path, force: bool) -> None:
    if dest.exists():
        if force:
            logger.info("Removing existing %s (--force)", dest)
            shutil.rmtree(dest)
        else:
            raise FileExistsError(f"{dest} exists. Use --force to overwrite.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Copying %s -> %s ...", source, dest)
    subprocess.run(["rsync", "-a", f"{source}/", f"{dest}/"], check=True)


def _update_info_json(dest: Path) -> None:
    info_path = dest / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["features"]["observation.keypoint_3d"] = {
        "dtype": "float32",
        "shape": [NUM_KEYPOINTS * KEYPOINT_DIM],
        "names": KEYPOINT_FEATURE_NAMES,
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info("Updated %s with observation.keypoint_3d [%d].", info_path, NUM_KEYPOINTS * KEYPOINT_DIM)


def _write_meta(dest: Path, r_pad: float, global_min: np.ndarray,
                global_max: np.ndarray, total_frames: int, torso_q: np.ndarray,
                urdf_path: str) -> None:
    meta = {
        "bbox_radius": r_pad,
        "bbox_margin": BBOX_MARGIN,
        "global_min_base_relative": global_min.tolist(),
        "global_max_base_relative": global_max.tolist(),
        "normalization": "base_link_origin_isotropic",
        "keypoint_dim": KEYPOINT_DIM,
        "keypoint_dim_layout": "px,py,pz,qx,qy,qz,qw",
        "rotation_representation": "quaternion_xyzw_hemisphere",
        "rotation_convention": "qw >= 0; negate if qw < 0",
        "num_keypoints": NUM_KEYPOINTS,
        "keypoint_links": KEYPOINT_LINKS,
        "total_frames": total_frames,
        "coordinate_system": (
            "base_link-relative, position divided by bbox_radius, "
            "quaternion hemisphere-normalized"
        ),
        "torso_q": torso_q.tolist(),
        "torso_q_physical": list(TORSO_Q_PHYSICAL),
        "torso_q_note": (
            "torso_q is what FK used. Inference MUST reuse this value. "
            "Do NOT substitute the live torso encoder reading. "
            "See generate_r1pro_keypoints.py docstring for the full explanation."
        ),
        "urdf": str(urdf_path),
    }
    meta_path = dest / "meta" / "keypoints_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote %s", meta_path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Source LeRobot dataset directory (read-only).")
    parser.add_argument("--dest", type=str, required=True,
                        help="Destination for the copy with 7D keypoints.")
    parser.add_argument("--urdf", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "assets" / "r1_pro_with_gripper.urdf"),
                        help="Path to R1 Pro URDF.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite --dest if it exists.")
    parser.add_argument("--skip-copy", action="store_true",
                        help="Reuse existing --dest (e.g. from a partial run).")
    parser.add_argument("--torso-q", type=float, nargs=4, default=list(TORSO_Q_DEFAULT),
                        metavar=("J1", "J2", "J3", "J4"),
                        help="Torso joint angles (rad) baked into FK.")
    parser.add_argument("--bbox-margin", type=float, default=BBOX_MARGIN,
                        help=f"Safety margin for R_pad (default: {BBOX_MARGIN}).")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    global BBOX_MARGIN
    BBOX_MARGIN = args.bbox_margin

    # Step 0: Copy dataset
    if not args.skip_copy:
        _copy_dataset(source, dest, args.force)
    else:
        if not dest.exists():
            raise FileNotFoundError(f"--skip-copy but {dest} does not exist.")
        logger.info("--skip-copy: reusing %s", dest)

    # Step 1: Load URDF and build FK extractor
    logger.info("Loading URDF from %s", args.urdf)
    logger.info("FK torso_q = %s", args.torso_q)
    extractor = R1ProFKExtractorE1(args.urdf, torso_q=args.torso_q)

    source_parquets = _get_parquet_files(source / "data")
    logger.info("Found %d parquet files", len(source_parquets))

    # Step 2: Pass 1 — compute global bounding box (position only)
    logger.info("=== Pass 1: computing global bounding box ===")
    global_min, global_max, total_frames_p1 = pass1_compute_bbox(source_parquets, extractor)
    r_pad = compute_r_pad(global_min, global_max, margin=BBOX_MARGIN)

    logger.info("Global min (base-rel): %s", global_min)
    logger.info("Global max (base-rel): %s", global_max)
    logger.info("R_pad = %.6f m (margin=%.0f%%)", r_pad, BBOX_MARGIN * 100)
    logger.info("Position range after normalization: [%.4f, %.4f]",
                -np.abs(global_min).max() / r_pad, np.abs(global_max).max() / r_pad)

    # Step 3: Pass 2 — normalize and write keypoints
    logger.info("=== Pass 2: writing 7D keypoints ===")
    total_frames_p2 = pass2_write_keypoints(
        source_parquets, extractor, r_pad, source / "data", dest
    )

    # Step 4: Update metadata
    _update_info_json(dest)
    _write_meta(dest, r_pad, global_min, global_max, total_frames_p2,
                extractor.torso_q, args.urdf)

    logger.info("=== DONE ===")
    logger.info("  Frames: %d (Pass 1) / %d (Pass 2)", total_frames_p1, total_frames_p2)
    logger.info("  Output: %s", dest)
    logger.info("  keypoint_dim: %d (= %d keypoints × %d per point)",
                NUM_KEYPOINTS * KEYPOINT_DIM, NUM_KEYPOINTS, KEYPOINT_DIM)
    logger.info("  R_pad: %.6f m", r_pad)


if __name__ == "__main__":
    main()
```

### 3.3 创建脚本文件

```bash
# 确认脚本目录存在
ls /home/luogang/SRC/Robot/itvlaGp/util_scripts/generate_r1pro_keypoints.py

# 将上面的代码写入新文件 (或通过编辑器创建):
# 文件路径: /home/luogang/SRC/Robot/itvlaGp/util_scripts/generate_r1pro_keypoints_e1.py
```

### 3.4 与原脚本的差异对照

| 对比项 | `generate_r1pro_keypoints.py` (原) | `generate_r1pro_keypoints_e1.py` (E1) |
|--------|----------------------------------|---------------------------------------|
| `R1ProFKExtractor.compute()` 返回值 | `[16, 3]` (仅 `.translation`) | `[16, 7]` (`.translation` + `pin.Quaternion(.rotation)` + 半球归一化) |
| `KEYPOINT_DIM` | 不存在 (隐含 3) | `7` |
| `KEYPOINT_FEATURE_NAMES` | `{link}_{x,y,z}` (48 names) | `{link}_{px,py,pz,qx,qy,qz,qw}` (112 names) |
| Pass 1 输出 | `global_min, global_max` → `offset` | `global_min, global_max` → `R_pad` + 四元数健康检查 |
| Pass 2 归一化 | `kpts - offset` (平移) | `pos / R_pad` (缩放) + quat 已在 `compute()` 中半球归一化 |
| `_write_meta()` | `coord_offset`, `voxel_center`, `voxel_bounds` | `bbox_radius`, `bbox_margin`, `keypoint_dim`, `rotation_representation` 等 |
| OOB 检查 | `< VOXEL_MIN` 或 `> VOXEL_MAX` | `|pos| > 1.01` 且 `|quat_norm - 1| > 0.01` |
| `_update_info_json()` shape | `[48]`, dtype=`float64` | `[112]`, dtype=`float32` |

---

## 4. 执行关键点生成

### 4.1 运行命令

```bash
cd /home/luogang/SRC/Robot/itvlaGp

conda activate itvlaGp

python util_scripts/generate_r1pro_keypoints_e1.py \
    --source /home/luogang/DATA/elevator0714_lerobot \
    --dest /home/luogang/DATA/elevator0714_lerobot_4D \
    --urdf assets/r1_pro_with_gripper.urdf \
    --force
```

### 4.2 预期日志

```
[INFO] Copying /home/luogang/DATA/elevator0714_lerobot -> /home/luogang/DATA/elevator0714_lerobot_4D ...
[INFO] Loading URDF from assets/r1_pro_with_gripper.urdf
[INFO] FK torso_q = [0.0, 0.0, 0.0, 0.0]
[INFO] Found 200 parquet files
[INFO] === Pass 1: computing global bounding box ===
[INFO]   Pass 1: file-001.parquet — XXX frames (total: XXX)
...
[INFO] Quaternion stats: qw_min=0.XXXXXX, qw_max=1.000000, norm_err_max=X.XXe-XX
[INFO] Global min (base-rel): [...]
[INFO] Global max (base-rel): [...]
[INFO] R_pad = X.XXXXXX m (margin=15%)
[INFO] === Pass 2: writing 7D keypoints ===
...
[INFO] === DONE ===
[INFO]   Frames: XXXXX (Pass 1) / XXXXX (Pass 2)
[INFO]   Output: /home/luogang/DATA/elevator0714_lerobot_4D
[INFO]   keypoint_dim: 112 (= 16 keypoints × 7 per point)
[INFO]   R_pad: X.XXXXXX m
```

### 4.3 关键检查点

运行后立即检查:

```bash
# 1. keypoints_meta.json 存在且内容正确
cat /home/luogang/DATA/elevator0714_lerobot_4D/meta/keypoints_meta.json | python3 -m json.tool

# 预期字段:
#   "bbox_radius": <正数>,
#   "keypoint_dim": 7,
#   "rotation_representation": "quaternion_xyzw_hemisphere",
#   "num_keypoints": 16

# 2. info.json 包含 observation.keypoint_3d 且 shape=[112]
python3 -c "
import json
with open('/home/luogang/DATA/elevator0714_lerobot_4D/meta/info.json') as f:
    info = json.load(f)
kpt = info['features']['observation.keypoint_3d']
print(f'shape: {kpt[\"shape\"]}')   # 应输出 [112]
print(f'dtype: {kpt[\"dtype\"]}')   # 应输出 float32
print(f'names count: {len(kpt[\"names\"])}')  # 应输出 112
"

# 3. 无 WARNING 日志 (若有则排查)
```

### 4.4 预期耗时

- rsync 拷贝: ~30 秒 (数据约 300 MB)
- Pass 1: ~2 秒 (27k 帧, ~34k fps FK 吞吐量, 四元数开销 <5%)
- Pass 2: ~2 秒 (同上) + parquet I/O ~30 秒
- **总计: 约 1-2 分钟**

---

## 5. 生成后数据验证

### 5.1 验证脚本: verify_e1_keypoints.py

在 `util_scripts/verify_e1_keypoints.py` 创建以下验证脚本:

```python
"""Post-generation verification for E1 7D keypoints.

Checks:
  1. observation.keypoint_3d exists in every parquet and has shape [112]
  2. All positions within [-1.01, 1.01] (after R_pad normalization)
  3. All quaternions are unit vectors (|q| = 1 ± 0.001)
  4. All quaternions satisfy hemisphere constraint (qw >= 0)
  5. Temporal smoothness: frame-to-frame quaternion change < 0.5 (no sign flips)
  6. FK reproducibility: re-compute a random sample and compare with stored values
  7. Print per-dimension statistics (mean, std, min, max)

Usage:
    python util_scripts/verify_e1_keypoints.py \
        --dataset /home/luogang/DATA/elevator0714_lerobot_4D \
        --urdf assets/r1_pro_with_gripper.urdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


NUM_KEYPOINTS = 16
KEYPOINT_DIM = 7
TOTAL_DIM = NUM_KEYPOINTS * KEYPOINT_DIM  # 112


def load_all_keypoints(dataset: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Load all keypoint vectors and return (kpts [N, 16, 7], full_df)."""
    files = sorted(dataset.glob("data/**/*.parquet"))
    if not files:
        sys.exit(f"No parquet files under {dataset}/data")

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if "observation.keypoint_3d" not in df.columns:
            sys.exit(f"observation.keypoint_3d not found in {f}")
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    raw = np.stack(full["observation.keypoint_3d"].values)  # [N, 112]
    kpts = raw.reshape(-1, NUM_KEYPOINTS, KEYPOINT_DIM)  # [N, 16, 7]
    return kpts, full


def check_shape(kpts: np.ndarray):
    print(f"\n=== Check 1: Shape ===")
    print(f"  Total frames: {kpts.shape[0]}")
    print(f"  Shape per frame: [{kpts.shape[1]}, {kpts.shape[2]}] (expect [16, 7])")
    assert kpts.shape[1:] == (NUM_KEYPOINTS, KEYPOINT_DIM), \
        f"Shape mismatch: {kpts.shape[1:]} != ({NUM_KEYPOINTS}, {KEYPOINT_DIM})"
    print("  ✓ PASS")


def check_position_bounds(kpts: np.ndarray):
    print(f"\n=== Check 2: Position bounds ===")
    pos = kpts[:, :, :3]
    pos_max = np.abs(pos).max()
    print(f"  max |position|: {pos_max:.6f} (threshold: 1.01)")
    if pos_max > 1.01:
        print(f"  ✗ FAIL: {pos_max:.6f} > 1.01")
    else:
        print("  ✓ PASS")


def check_quaternion_norm(kpts: np.ndarray):
    print(f"\n=== Check 3: Quaternion unit norm ===")
    quat = kpts[:, :, 3:7]
    norms = np.linalg.norm(quat.reshape(-1, 4), axis=1)
    norm_err = np.abs(norms - 1.0)
    print(f"  norm error: mean={norm_err.mean():.2e}, max={norm_err.max():.2e}")
    if norm_err.max() > 0.001:
        print(f"  ✗ FAIL: max norm error {norm_err.max():.6f} > 0.001")
    else:
        print("  ✓ PASS")


def check_hemisphere(kpts: np.ndarray):
    print(f"\n=== Check 4: Hemisphere constraint (qw >= 0) ===")
    qw = kpts[:, :, 6]  # qw is the 4th quaternion component at index 6 (overall index 3+3=6)
    qw_min = qw.min()
    violations = (qw < -1e-7).sum()
    print(f"  qw min: {qw_min:.8f}")
    print(f"  violations (qw < 0): {violations}")
    if violations > 0:
        print(f"  ✗ FAIL: {violations} frames violate hemisphere constraint")
    else:
        print("  ✓ PASS")


def check_temporal_smoothness(kpts: np.ndarray, full_df: pd.DataFrame):
    print(f"\n=== Check 5: Temporal smoothness (within episodes) ===")
    episodes = full_df["episode_index"].values
    quat = kpts[:, :, 3:7]  # [N, 16, 4]

    max_jump = 0.0
    jump_count = 0
    total_transitions = 0

    for ep in np.unique(episodes):
        mask = episodes == ep
        ep_quat = quat[mask]  # [T, 16, 4]
        if len(ep_quat) < 2:
            continue
        diffs = np.linalg.norm(ep_quat[1:] - ep_quat[:-1], axis=-1)  # [T-1, 16]
        frame_max = diffs.max(axis=1)  # [T-1]
        max_jump = max(max_jump, float(frame_max.max()))
        jump_count += int((frame_max > 0.5).sum())
        total_transitions += len(frame_max)

    print(f"  Total frame transitions: {total_transitions}")
    print(f"  Max quaternion jump: {max_jump:.6f} (threshold: 0.5)")
    print(f"  Jumps > 0.5: {jump_count}")
    if jump_count > 0:
        print(f"  ⚠ WARNING: {jump_count} large quaternion jumps detected")
    else:
        print("  ✓ PASS")


def check_fk_reproducibility(kpts: np.ndarray, full_df: pd.DataFrame,
                             dataset: Path, urdf_path: str):
    print(f"\n=== Check 6: FK reproducibility ===")
    try:
        import pinocchio as pin
    except ImportError:
        print("  [skip] pinocchio not available")
        return

    meta_path = dataset / "meta" / "keypoints_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    r_pad = meta["bbox_radius"]
    torso_q = meta["torso_q"]

    from generate_r1pro_keypoints_e1 import R1ProFKExtractorE1
    extractor = R1ProFKExtractorE1(urdf_path, torso_q=torso_q)

    # Pick 10 random frames
    rng = np.random.default_rng(42)
    indices = rng.choice(len(full_df), size=min(10, len(full_df)), replace=False)
    max_err = 0.0

    for idx in indices:
        row = full_df.iloc[idx]
        left = np.array(row["observation.state.left_arm"], dtype=np.float64)
        right = np.array(row["observation.state.right_arm"], dtype=np.float64)
        recomputed = extractor.compute(left, right)  # [16, 7]
        recomputed[:, :3] /= r_pad
        stored = kpts[idx]  # [16, 7]
        err = np.abs(recomputed - stored).max()
        max_err = max(max_err, err)

    print(f"  Max recomputation error (10 random frames): {max_err:.2e}")
    if max_err > 1e-5:
        print(f"  ✗ FAIL: reproducibility error {max_err:.6f} > 1e-5")
    else:
        print("  ✓ PASS")


def print_statistics(kpts: np.ndarray):
    print(f"\n=== Check 7: Per-dimension statistics ===")
    flat = kpts.reshape(-1, KEYPOINT_DIM)  # [N*16, 7]
    labels = ["px", "py", "pz", "qx", "qy", "qz", "qw"]
    print(f"  {'dim':>4s}  {'mean':>10s}  {'std':>10s}  {'min':>10s}  {'max':>10s}")
    print("  " + "-" * 50)
    for i, label in enumerate(labels):
        col = flat[:, i]
        print(f"  {label:>4s}  {col.mean():+10.6f}  {col.std():10.6f}  {col.min():+10.6f}  {col.max():+10.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--urdf", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "assets" / "r1_pro_with_gripper.urdf"))
    args = parser.parse_args()

    dataset = Path(args.dataset)
    print(f"Verifying E1 keypoints in: {dataset}")

    kpts, full_df = load_all_keypoints(dataset)

    check_shape(kpts)
    check_position_bounds(kpts)
    check_quaternion_norm(kpts)
    check_hemisphere(kpts)
    check_temporal_smoothness(kpts, full_df)
    check_fk_reproducibility(kpts, full_df, dataset, args.urdf)
    print_statistics(kpts)

    print(f"\n=== Summary ===")
    print(f"  Dataset: {dataset}")
    print(f"  Frames: {len(kpts)}")
    print(f"  Keypoints per frame: {NUM_KEYPOINTS} × {KEYPOINT_DIM}D = {TOTAL_DIM}")
    print("  All checks complete.")


if __name__ == "__main__":
    main()
```

### 5.2 运行验证

```bash
cd /home/luogang/SRC/Robot/itvlaGp

python util_scripts/verify_e1_keypoints.py \
    --dataset /home/luogang/DATA/elevator0714_lerobot_4D \
    --urdf assets/r1_pro_with_gripper.urdf
```

### 5.3 验收标准

**全部 7 项检查必须通过**:

| # | 检查 | 通过条件 | 失败时动作 |
|---|------|---------|-----------|
| 1 | Shape | 每帧 `[16, 7]` | 检查生成脚本的 `KEYPOINT_DIM` |
| 2 | Position bounds | `max|pos| ≤ 1.01` | 增大 `--bbox-margin` 后重新生成 |
| 3 | Quaternion norm | `max|‖q‖-1| ≤ 0.001` | 检查 Pinocchio 版本或 FK 代码 |
| 4 | Hemisphere | 全部 `qw ≥ 0` | 检查 `hemisphere_normalize` 逻辑 |
| 5 | Temporal smoothness | 无 jump > 0.5 | 检查半球归一化是否在 FK 之后执行 |
| 6 | FK reproducibility | `max_err ≤ 1e-5` | 检查 Pass 2 是否使用了正确的 `R_pad` |
| 7 | Statistics | 位置分量在 [-1, 1], qw 均 ≥0 | 综合检查 |

---

## 6. norm_stats 统计量生成

训练前需要为带关键点的新数据集生成归一化统计量 (`stats.json`).

### 6.1 准备

新数据集必须对训练框架可见. LeRobot 通过 `HF_LEROBOT_HOME / repo_id` 定位数据集:

```bash
# 方式 A: 设置 HF_LEROBOT_HOME 指向 DATA 目录的父目录
export HF_LEROBOT_HOME=/home/luogang/DATA

# 此时 repo_id = "elevator0714_lerobot_4D"
# 训练框架会在 /home/luogang/DATA/elevator0714_lerobot_4D/ 找到它

# 方式 B: 创建符号链接到 HF_LEROBOT_HOME 下
# ln -s /home/luogang/DATA/elevator0714_lerobot_4D $HF_LEROBOT_HOME/elevator0714_lerobot_4D
```

### 6.2 运行 norm_stats 计算

```bash
cd /home/luogang/SRC/Robot/itvlaGp

export HF_LEROBOT_HOME=/home/luogang/DATA

# abs 模式:
python util_scripts/compute_norm_stats_single.py \
    --repo_id elevator0714_lerobot_4D \
    --action_mode abs \
    --chunk_size 50

# delta 模式 (如果训练用 delta):
python util_scripts/compute_norm_stats_single.py \
    --repo_id elevator0714_lerobot_4D \
    --action_mode delta \
    --chunk_size 50
```

### 6.3 验证 stats.json 包含 keypoint_3d

```bash
python3 -c "
import json
from pathlib import Path

# 找到生成的 stats.json
hf_home = Path('/home/luogang/DATA')
stats_dir = hf_home / 'stats' / 'abs' / 'elevator0714_lerobot_4D'
stats_path = stats_dir / 'stats.json'
if not stats_path.exists():
    # 也可能在数据集内部
    stats_path = hf_home / 'elevator0714_lerobot_4D' / 'meta' / 'stats.json'

print(f'Reading: {stats_path}')
with open(stats_path) as f:
    stats = json.load(f)

# 检查 observation.keypoint_3d 是否存在
if 'observation.keypoint_3d' in stats:
    kpt_stats = stats['observation.keypoint_3d']
    print(f'observation.keypoint_3d stats:')
    print(f'  mean dim: {len(kpt_stats[\"mean\"])}')  # 应为 112
    print(f'  std dim:  {len(kpt_stats[\"std\"])}')    # 应为 112
    print(f'  min dim:  {len(kpt_stats[\"min\"])}')    # 应为 112
    print(f'  max dim:  {len(kpt_stats[\"max\"])}')    # 应为 112

    # 抽查前 7 个维度 (第一个关键点: px,py,pz,qx,qy,qz,qw)
    print(f'  First keypoint (7D):')
    print(f'    mean: {kpt_stats[\"mean\"][:7]}')
    print(f'    std:  {kpt_stats[\"std\"][:7]}')
    print(f'    min:  {kpt_stats[\"min\"][:7]}')
    print(f'    max:  {kpt_stats[\"max\"][:7]}')
else:
    print('✗ observation.keypoint_3d NOT found in stats.json')
    print(f'Available keys: {list(stats.keys())}')
"
```

**预期**: `observation.keypoint_3d` 的 stats 有 112 维, 位置分量的 mean/std 在合理范围内 (mean 接近 0, std < 1), 四元数分量的 min/max 在 [-1, 1].

### 6.4 关键注意

`compute_norm_stats_single.py` 会遍历数据集的所有非 video/image 列来计算统计量. `observation.keypoint_3d` 是一个 112 维的 float32 向量, 它会被自动包含在计算中. **无需修改 norm_stats 脚本**.

---

## 7. 模型侧代码改动

为了让训练管道支持 7D 关键点 (E1), 需要修改 3 个文件, 新增 4 个配置参数. 以下是**逐文件、逐行**的精确改动说明.

### 7.1 文件 1: `configuration_internvla_a1_5.py`

**路径**: `src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py`

#### 改动 1a: `InternVLAA15DatasetConfig` 新增 `keypoint_dim`

**位置**: 第 40 行附近 (在 `keypoint_history_max_len` 之后)

**原代码** (L38-40):
```python
    enable_keypoint_predictor: bool = False
    num_keypoint_joints: int = 8
    keypoint_history_max_len: int = 1000
```

**改为**:
```python
    enable_keypoint_predictor: bool = False
    num_keypoint_joints: int = 8
    keypoint_history_max_len: int = 1000
    keypoint_dim: int = 3  # per-keypoint feature dimension: 3 (pos-only) or 7 (pos + quat)
```

**原因**: `Extract3DKeypointTransformFn` 需要知道每个关键点的维度才能正确 reshape. 该参数从 dataset config 传递到 transform.

#### 改动 1b: `InternVLAA15Config` (policy config) 新增 `keypoint_out_dim` 和 `kpt_rot_loss_weight`

**位置**: 第 474 行附近 (在 `keypoint_track_input_dim` 之后)

**原代码** (L474):
```python
    keypoint_track_input_dim: int = 3
```

**改为**:
```python
    keypoint_track_input_dim: int = 3
    keypoint_out_dim: int = 3  # per-keypoint output dimension for keypoint_out_proj: 3 or 7
    kpt_rot_loss_weight: float = 1.0  # weight of rotation loss relative to position loss
```

**原因**: 
- `keypoint_out_dim` 控制 `keypoint_out_proj = nn.Linear(hidden, keypoint_out_dim)` 的输出维度
- `kpt_rot_loss_weight` 控制旋转 MSE 损失相对于位置 MSE 损失的权重

### 7.2 文件 2: `modeling_internvla_a1_5.py`

**路径**: `src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py`

#### 改动 2a: `keypoint_out_proj` 使用配置参数

**位置**: 第 1020 行

**原代码**:
```python
            self.keypoint_out_proj = nn.Linear(kpt_hidden_size, 3)
```

**改为**:
```python
            self.keypoint_out_proj = nn.Linear(kpt_hidden_size, config.keypoint_out_dim)
```

**原因**: 从硬编码 3 改为可配置. E1 设为 7.

#### 改动 2b: 当前帧 loss 中的零填充维度

**位置**: 第 1957-1961 行

**原代码**:
```python
            if kpt_t is None:
                kpt_t = torch.zeros(B, j, 3, device=actions.device, dtype=torch.float32)
            loss_kpt_current = F.mse_loss(
                pred_kpt_current, kpt_t.to(torch.float32), reduction="none"
            ).mean(dim=(-1, -2))  # [B]
```

**改为**:
```python
            kpt_out_dim = self.config.keypoint_out_dim
            if kpt_t is None:
                kpt_t = torch.zeros(B, j, kpt_out_dim, device=actions.device, dtype=torch.float32)

            if kpt_out_dim > 3:
                # E1: split position and rotation, compute separate losses
                pred_pos = pred_kpt_current[..., :3]
                gt_pos = kpt_t[..., :3].to(torch.float32)
                loss_pos = F.mse_loss(pred_pos, gt_pos, reduction="none").mean(dim=(-1, -2))

                pred_rot = pred_kpt_current[..., 3:kpt_out_dim]
                gt_rot = kpt_t[..., 3:kpt_out_dim].to(torch.float32)
                pred_rot_norm = F.normalize(pred_rot, p=2, dim=-1)
                loss_rot = F.mse_loss(pred_rot_norm, gt_rot, reduction="none").mean(dim=(-1, -2))

                loss_kpt_current = loss_pos + self.config.kpt_rot_loss_weight * loss_rot
            else:
                loss_kpt_current = F.mse_loss(
                    pred_kpt_current, kpt_t.to(torch.float32), reduction="none"
                ).mean(dim=(-1, -2))
```

**原因**: 当 `keypoint_out_dim=7` 时, 位置 ([0:3]) 和旋转 ([3:7]) 分别计算 MSE, 旋转部分先 L2 归一化到单位球面后再计算 MSE. 当 `keypoint_out_dim=3` 时行为与改动前完全一致 (向后兼容).

#### 改动 2c: 未来帧预测的 shape

**位置**: 第 1968-1976 行

**原代码**:
```python
            future_kpt_pred = self.keypoint_out_proj(
                future_kpt_tokens.reshape(B * chunk_size, j, -1)
            ).reshape(B, chunk_size, j, 3)

            if kpt_future is None:
                kpt_future = torch.zeros(B, chunk_size, j, 3, device=actions.device, dtype=torch.float32)
            loss_kpt_future = F.mse_loss(
                future_kpt_pred, kpt_future.to(torch.float32), reduction="none"
            ).mean(dim=(-1, -2, -3))  # [B]
```

**改为**:
```python
            future_kpt_pred = self.keypoint_out_proj(
                future_kpt_tokens.reshape(B * chunk_size, j, -1)
            ).reshape(B, chunk_size, j, kpt_out_dim)

            if kpt_future is None:
                kpt_future = torch.zeros(B, chunk_size, j, kpt_out_dim, device=actions.device, dtype=torch.float32)

            if kpt_out_dim > 3:
                pred_pos_f = future_kpt_pred[..., :3]
                gt_pos_f = kpt_future[..., :3].to(torch.float32)
                loss_pos_f = F.mse_loss(pred_pos_f, gt_pos_f, reduction="none").mean(dim=(-1, -2, -3))

                pred_rot_f = future_kpt_pred[..., 3:kpt_out_dim]
                gt_rot_f = kpt_future[..., 3:kpt_out_dim].to(torch.float32)
                pred_rot_f_norm = F.normalize(pred_rot_f, p=2, dim=-1)
                loss_rot_f = F.mse_loss(pred_rot_f_norm, gt_rot_f, reduction="none").mean(dim=(-1, -2, -3))

                loss_kpt_future = loss_pos_f + self.config.kpt_rot_loss_weight * loss_rot_f
            else:
                loss_kpt_future = F.mse_loss(
                    future_kpt_pred, kpt_future.to(torch.float32), reduction="none"
                ).mean(dim=(-1, -2, -3))
```

**原因**: 未来帧预测与当前帧完全对称, 同样需要分离位置和旋转损失.

#### 改动 2d: loss_dict 增加位置/旋转分项 (可选但推荐)

**位置**: 第 2530-2532 行

**原代码**:
```python
        if self.config.enable_keypoint_predictor:
            loss_dict["loss_kpt_current"] = loss_kpt_cur.item()
            loss_dict["loss_kpt_future"] = loss_kpt_fut.item()
```

**改为**:
```python
        if self.config.enable_keypoint_predictor:
            loss_dict["loss_kpt_current"] = loss_kpt_cur.item()
            loss_dict["loss_kpt_future"] = loss_kpt_fut.item()
            loss_dict["loss_kpt"] = loss_kpt.item()
```

**原因**: 方便在 wandb 中直接观察总的 keypoint loss.

### 7.3 文件 3: `transform_internvla_a1_5.py`

**路径**: `src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py`

#### 改动 3a: `Extract3DKeypointTransformFn` 新增 `keypoint_dim` 字段

**位置**: 第 688-690 行

**原代码**:
```python
    num_joints: int = 8
    history_max_len: int = 1000
    chunk_size: int = 50
```

**改为**:
```python
    num_joints: int = 8
    history_max_len: int = 1000
    chunk_size: int = 50
    keypoint_dim: int = 3  # 3 for position-only, 7 for position + quaternion (E1)
```

#### 改动 3b: `__call__` 中的 reshape 和零填充维度

**位置**: 第 693-707 行和 715-730 行

**原代码** (L693-701):
```python
    def __call__(self, data: DataDict) -> DataDict:
        h, j, c = self.history_max_len, self.num_joints, self.chunk_size
        key = "observation.keypoint_3d"

        if key not in data:
            data["observation.his_kpts"] = torch.zeros(h, j, 3)
            data["observation.his_len"] = torch.tensor(0, dtype=torch.long)
            data["observation.kpt_t"] = torch.zeros(j, 3)
            data["observation.kpt_future"] = torch.zeros(c, j, 3)
            data["observation.kpt_mask"] = torch.tensor(False)
            return data
```

**改为**:
```python
    def __call__(self, data: DataDict) -> DataDict:
        h, j, c, d = self.history_max_len, self.num_joints, self.chunk_size, self.keypoint_dim
        key = "observation.keypoint_3d"

        if key not in data:
            data["observation.his_kpts"] = torch.zeros(h, j, d)
            data["observation.his_len"] = torch.tensor(0, dtype=torch.long)
            data["observation.kpt_t"] = torch.zeros(j, d)
            data["observation.kpt_future"] = torch.zeros(c, j, d)
            data["observation.kpt_mask"] = torch.tensor(False)
            return data
```

**原代码** (L707):
```python
        stacked = stacked.reshape(h + 1 + c, j, 3).float()
```

**改为**:
```python
        stacked = stacked.reshape(h + 1 + c, j, d).float()
```

**原代码** (L720):
```python
        his_kpts = torch.zeros(h, j, 3, dtype=stacked.dtype)
```

**改为**:
```python
        his_kpts = torch.zeros(h, j, d, dtype=stacked.dtype)
```

**原因**: 所有零填充和 reshape 都从硬编码 `3` 改为参数 `d` (= `self.keypoint_dim`). 当 `keypoint_dim=3` 时行为不变 (向后兼容).

#### 改动 3c: 传递 `keypoint_dim` 到 transform 实例

`Extract3DKeypointTransformFn` 是在 transform pipeline 的数据流中被实例化的. 需要确认它的 `keypoint_dim` 参数能被正确设置. 

**搜索构造 `Extract3DKeypointTransformFn` 的位置**:

在 `transform_internvla_a1_5.py` 内搜索 `Extract3DKeypointTransformFn` 的实例化:

```bash
grep -n "Extract3DKeypointTransformFn" src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py
```

它是 dataclass + `register_subclass("extract_3d_keypoint")`, 通过 draccus config 实例化. 在 dataset config 的 `data_transforms.inputs` 列表中或者在构建 transform pipeline 时被添加. 

具体来说, 在训练脚本 (`lerobot_train.py`) 或 dataset 构建代码中, 当 `enable_keypoint_predictor=True` 时, `Extract3DKeypointTransformFn` 会被动态添加到 transform pipeline, 其参数来自 dataset config:

```python
Extract3DKeypointTransformFn(
    num_joints=dataset_config.num_keypoint_joints,
    history_max_len=dataset_config.keypoint_history_max_len,
    chunk_size=policy_config.chunk_size,
    keypoint_dim=dataset_config.keypoint_dim,  # ← E1: 需要传递此参数
)
```

需要找到这个构造位置并添加 `keypoint_dim` 参数. 搜索代码:

```bash
grep -rn "Extract3DKeypointTransformFn\|extract_3d_keypoint" src/lerobot/ --include="*.py"
```

如果该 transform 是通过 draccus 配置文件 (YAML/JSON) 实例化的, 则在配置文件中添加 `keypoint_dim: 7` 即可. 如果是在 Python 代码中手动构造的, 则需要传入 `keypoint_dim=dataset_config.keypoint_dim`.

### 7.4 改动验证: 向后兼容性

所有新增参数的默认值与改动前的行为一致:

| 参数 | 默认值 | 效果 |
|------|--------|------|
| `keypoint_dim = 3` | `3` | reshape 为 `[h+1+c, j, 3]`, 与改动前一致 |
| `keypoint_out_dim = 3` | `3` | `Linear(hidden, 3)`, 与改动前一致 |
| `kpt_rot_loss_weight = 1.0` | `1.0` | 当 `keypoint_out_dim=3` 时不进入旋转 loss 分支, 无影响 |

**不改任何 CLI 参数的情况下, 现有的 3D 关键点训练不受影响.**

### 7.5 E1 训练的 CLI 参数差异

在训练命令中, 相比原方案 E 仅多设 3 个参数:

```bash
# ---- E1 新增 (相对方案 E) ----
--policy.keypoint_track_input_dim=7   # TrackEncoder Conv1d 输入通道
--policy.keypoint_out_dim=7           # keypoint_out_proj 输出维度
--dataset.keypoint_dim=7              # Transform reshape 维度
```

完整训练命令示例:

```bash
accelerate launch --num_processes=2 src/lerobot/scripts/lerobot_train.py \
    --policy.type=internvla_a1_5 \
    --policy.pretrained_path=InternRobotics/InternVLA-A1.5-base \
    --policy.enable_keypoint_predictor=true \
    --policy.num_keypoint_joints=16 \
    --policy.keypoint_track_input_dim=7 \
    --policy.keypoint_out_dim=7 \
    --policy.kpt_rot_loss_weight=1.0 \
    --policy.kpt_loss_weight=10.0 \
    --policy.kpt_future_loss_weight=2.0 \
    --policy.train_expert_only=true \
    --policy.action_loss_only=true \
    --policy.keypoint_history_max_len=200 \
    --dataset.type=internvla_a1_5 \
    --dataset.repo_id=elevator0714_lerobot_4D \
    --dataset.enable_keypoint_predictor=true \
    --dataset.num_keypoint_joints=16 \
    --dataset.keypoint_dim=7 \
    --dataset.action_mode=abs \
    --batch_size=12 \
    --steps=100
```

---

## 8. Smoke Test: 端到端数据流验证

在正式训练前, 先做一个 100 步的 smoke test, 确认整条数据管道无报错.

### 8.1 Smoke test 命令

```bash
cd /home/luogang/SRC/Robot/itvlaGp

export HF_LEROBOT_HOME=/home/luogang/DATA

# 单 GPU smoke test (100 步, 验证数据流)
python src/lerobot/scripts/lerobot_train.py \
    --policy.type=internvla_a1_5 \
    --policy.pretrained_path=InternRobotics/InternVLA-A1.5-base \
    --policy.enable_keypoint_predictor=true \
    --policy.num_keypoint_joints=16 \
    --policy.keypoint_track_input_dim=7 \
    --policy.keypoint_out_dim=7 \
    --policy.kpt_rot_loss_weight=1.0 \
    --policy.kpt_loss_weight=10.0 \
    --policy.train_expert_only=true \
    --policy.action_loss_only=true \
    --policy.keypoint_history_max_len=200 \
    --dataset.type=internvla_a1_5 \
    --dataset.repo_id=elevator0714_lerobot_4D \
    --dataset.enable_keypoint_predictor=true \
    --dataset.num_keypoint_joints=16 \
    --dataset.keypoint_dim=7 \
    --dataset.action_mode=abs \
    --batch_size=2 \
    --steps=100
```

### 8.2 Smoke test 检查项

| 检查项 | 方法 | 预期 | 不符时动作 |
|--------|------|------|-----------|
| 启动无报错 | 观察日志 | 不出现 RuntimeError, shape mismatch | 检查 `keypoint_dim` 传递链路 |
| 数据加载正确 | 日志中的 batch shape | `his_kpts: [B, 200, 16, 7]` | 检查 transform 的 reshape |
| TrackEncoder 无报错 | 第一个 forward 通过 | Conv1d(7, 256) 接受 7 通道输入 | 检查 `keypoint_track_input_dim` |
| Loss 非 NaN | 日志中的 loss 值 | `loss_kpt_current`, `loss_kpt_future` 均有限 | 检查四元数归一化 |
| Loss 下降 | 前 100 步 loss 趋势 | 不要求单调, 但总体下降 | 检查学习率 |

### 8.3 常见错误与解决

| 错误消息 | 原因 | 解决 |
|---------|------|------|
| `RuntimeError: shape mismatch: ... [N, 48] cannot be reshaped to [H+1+C, 16, 7]` | 数据集仍是 48 维 (3D 关键点) 但 config 设了 `keypoint_dim=7` | 确认 `--dataset.repo_id` 指向 `_4D` 数据集 |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied (... x 3 and 7 x ...)` | `keypoint_out_dim` 与数据不匹配 | 确认 `--policy.keypoint_out_dim=7` |
| `KeyError: 'observation.keypoint_3d'` | 数据集未生成关键点 | 重新执行 §4 的关键点生成步骤 |
| `ValueError: operands could not be broadcast together` | `num_keypoint_joints` 与数据维度不匹配 | 确认 `--policy.num_keypoint_joints=16` 和 `--dataset.num_keypoint_joints=16` |

---

## 9. 完整文件清单与 diff

### 9.1 新增文件

| 文件 | 用途 | 行数 |
|------|------|------|
| `util_scripts/generate_r1pro_keypoints_e1.py` | E1 7D 关键点离线生成脚本 | ~280 |
| `util_scripts/verify_e1_keypoints.py` | 生成后验证脚本 (7 项检查) | ~180 |

### 9.2 修改文件

| 文件 | 改动 | 新增行 | 修改行 |
|------|------|--------|--------|
| `src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py` | 新增 `keypoint_dim`, `keypoint_out_dim`, `kpt_rot_loss_weight` | 3 | 0 |
| `src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py` | `keypoint_out_proj` 参数化 + 分离位置/旋转损失 | ~25 | ~10 |
| `src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py` | `keypoint_dim` 参数 + reshape 参数化 | 1 | 5 |

### 9.3 不修改的文件

| 文件 | 原因 |
|------|------|
| `src/lerobot/policies/internvla_a1_5/keypoints.py` | `TrackEncoder.input_dim` 已参数化, **零改动** |
| `util_scripts/compute_norm_stats_single.py` | 自动遍历所有非 video 列, **零改动** |
| `util_scripts/precheck_r1pro_dataset.py` | 检查原始数据集, 与关键点无关, **零改动** |
| `src/lerobot/dataset_schemas/configs/r1_pro.yaml` | schema 定义的是 state/action 字段, keypoint_3d 不在 schema 中, **零改动** |

### 9.4 不修改的文件清单 (数据侧)

| 文件/目录 | 处理方式 |
|----------|---------|
| `SOURCE/data/chunk-000/*.parquet` | rsync 拷贝到 DEST, 然后 Pass 2 写入 `observation.keypoint_3d` 列 |
| `SOURCE/meta/info.json` | rsync 拷贝后更新 (添加 `observation.keypoint_3d` feature) |
| `SOURCE/meta/stats.json` | rsync 拷贝 (原始 stats), 之后重新计算 (§6) |
| `SOURCE/meta/episodes.jsonl` | rsync 拷贝, **不修改** |
| `SOURCE/meta/tasks.jsonl` | rsync 拷贝, **不修改** |
| `SOURCE/videos/` | rsync 拷贝, **不修改** |
| (新增) `DEST/meta/keypoints_meta.json` | 生成脚本创建 |

---

## 10. 故障排查手册

### 10.1 生成阶段

| 症状 | 排查 | 修复 |
|------|------|------|
| `ModuleNotFoundError: No module named 'pinocchio'` | pinocchio 未安装 | `conda install -c conda-forge pinocchio -y` |
| `ValueError: Frame 'left_arm_link1' not found in URDF` | URDF 文件路径错误或 URDF 内容不匹配 | 检查 `--urdf` 参数指向 `assets/r1_pro_with_gripper.urdf` |
| `ValueError: torso varies within ...` | 数据集中 torso 列非全零 | 检查数据集是否正确; 若确认 torso 有实际值则需重新设计 FK 策略 |
| `RuntimeError: Hemisphere normalization failed` | Pass 1 中发现 qw < 0 | 这不应该发生 (compute() 中已做半球归一化), 检查 compute() 逻辑 |
| 生成速度很慢 (>10 分钟) | I/O 瓶颈或 pinocchio 版本问题 | 确认 pinocchio >= 2.6; 检查磁盘 I/O (`iostat`) |
| `FileExistsError: ... exists` | 目标目录已存在 | 添加 `--force` 参数, 或手动删除 |

### 10.2 验证阶段

| 症状 | 排查 | 修复 |
|------|------|------|
| Check 2 失败 (position OOB) | R_pad 计算的安全裕量不足 | 增大 `--bbox-margin` (如 0.2), 重新生成 |
| Check 4 失败 (qw < 0) | 半球归一化逻辑 bug | 检查 `compute()` 中的 `if raw_q[3] < 0: raw_q = -raw_q` |
| Check 5 警告 (large jumps) | 连续帧间四元数跳变 | 可能是 180° 旋转附近的振荡; 检查具体 episode 和帧号, 确认是否物理合理 |
| Check 6 失败 (FK 不可复现) | R_pad 值不匹配或 torso_q 不一致 | 检查 keypoints_meta.json 中的 bbox_radius 和 torso_q |

### 10.3 训练阶段

| 症状 | 排查 | 修复 |
|------|------|------|
| NaN loss | 四元数分母为零或 loss 值爆炸 | 检查 `F.normalize` 是否对零向量安全 (加 eps); 降低学习率 |
| `loss_kpt_current` 不下降 | 权重太小或旋转 loss 主导 | 增大 `kpt_loss_weight`; 调整 `kpt_rot_loss_weight` |
| `loss_kpt_current` 下降但 action loss 不变 | keypoint 与 action 解耦 | 检查 `knowledge_insulation_kpt` 和 `kpt_to_action_detach` 设置 |
| OOM (显存不足) | his_kpts buffer 太大 | 减小 `keypoint_history_max_len` (如 100); 减小 batch_size |

---

## 附录 A: 关键路径的端到端数据流

```
原始数据集 (elevator0714_lerobot)
  └── data/chunk-000/file-*.parquet
       ├── observation.state.left_arm [7]
       └── observation.state.right_arm [7]
                          │
                          ▼
        generate_r1pro_keypoints_e1.py
          ├── Pass 1: FK → global_min/max → R_pad
          └── Pass 2: FK → pos/R_pad + quat_hemisphere → [16, 7] → flatten [112]
                          │
                          ▼
输出数据集 (elevator0714_lerobot_4D)
  ├── data/chunk-000/file-*.parquet
  │    ├── observation.state.left_arm [7]      (原样保留)
  │    ├── observation.state.right_arm [7]     (原样保留)
  │    └── observation.keypoint_3d [112]       ★ 新增
  ├── meta/info.json                          (更新: 添加 keypoint_3d feature)
  ├── meta/keypoints_meta.json                ★ 新增
  ├── meta/stats.json                         (原样保留, 由 compute_norm_stats 更新)
  └── videos/                                 (原样保留)
                          │
                          ▼
        compute_norm_stats_single.py
          └── 输出: HF_LEROBOT_HOME/stats/abs/elevator0714_lerobot_4D/stats.json
                          │
                          ▼
        训练 (lerobot_train.py)
          ├── LeRobotDataset 读取 observation.keypoint_3d [112]
          ├── delta_indices 堆叠 → [(H+1+C) × 112]
          ├── Extract3DKeypointTransformFn reshape → [H+1+C, 16, 7]
          │    ├── his_kpts  [H, 16, 7] → TrackEncoder(input_dim=7) → [B, 16, 1024]
          │    ├── kpt_t     [16, 7]    → loss_kpt_current (pos MSE + rot MSE)
          │    └── kpt_future [C, 16, 7] → loss_kpt_future (pos MSE + rot MSE)
          └── keypoint_out_proj: Linear(1024, 7) → pred [B, 16, 7]
```

## 附录 B: keypoints_meta.json 参考

生成后 `/home/luogang/DATA/elevator0714_lerobot_4D/meta/keypoints_meta.json` 的内容示例:

```json
{
  "bbox_radius": 1.84,
  "bbox_margin": 0.15,
  "global_min_base_relative": [-0.30, -0.45, 0.60],
  "global_max_base_relative": [0.50, 0.45, 1.60],
  "normalization": "base_link_origin_isotropic",
  "keypoint_dim": 7,
  "keypoint_dim_layout": "px,py,pz,qx,qy,qz,qw",
  "rotation_representation": "quaternion_xyzw_hemisphere",
  "rotation_convention": "qw >= 0; negate if qw < 0",
  "num_keypoints": 16,
  "keypoint_links": [
    "left_arm_link1", "left_arm_link2", "left_arm_link3", "left_arm_link4",
    "left_arm_link5", "left_arm_link6", "left_arm_link7", "left_gripper_link",
    "right_arm_link1", "right_arm_link2", "right_arm_link3", "right_arm_link4",
    "right_arm_link5", "right_arm_link6", "right_arm_link7", "right_gripper_link"
  ],
  "total_frames": 27145,
  "coordinate_system": "base_link-relative, position divided by bbox_radius, quaternion hemisphere-normalized",
  "torso_q": [0.0, 0.0, 0.0, 0.0],
  "torso_q_physical": [0.8, -1.4, -0.6, 0.0],
  "torso_q_note": "torso_q is what FK used. Inference MUST reuse this value. ...",
  "urdf": "assets/r1_pro_with_gripper.urdf"
}
```

> 注意: `bbox_radius` 和 `global_min/max` 的具体数值在实际运行后才能确定, 上面是基于关节角范围的估算值.

## 附录 C: 操作检查清单

逐项勾选, 确保无遗漏:

- [ ] **1. 环境准备**: conda activate itvlaGp, pinocchio 可用
- [ ] **2. 预检**: `precheck_r1pro_dataset.py` 通过, 确认 robot_type=r1_pro
- [ ] **3. 脚本创建**: `generate_r1pro_keypoints_e1.py` 已创建在 `util_scripts/`
- [ ] **4. 关键点生成**: 脚本运行完成, 日志无 WARNING
- [ ] **5. 元数据检查**: `keypoints_meta.json` 存在, `keypoint_dim=7`
- [ ] **6. info.json 检查**: `observation.keypoint_3d` shape=[112]
- [ ] **7. 验证脚本创建**: `verify_e1_keypoints.py` 已创建在 `util_scripts/`
- [ ] **8. 7 项验证全部通过**: shape / bounds / norm / hemisphere / smoothness / reproducibility / stats
- [ ] **9. norm_stats 生成**: `compute_norm_stats_single.py` 运行, stats.json 含 112 维 keypoint 统计量
- [ ] **10. 代码改动**: 3 个文件改动完成 (configuration / modeling / transform)
- [ ] **11. Smoke test**: 100 步训练无报错, loss 非 NaN
- [ ] **12. 数据集保持完整**: 原始 SOURCE 未被修改, DEST 的 videos / episodes / tasks 完整
