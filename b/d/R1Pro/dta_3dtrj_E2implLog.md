# E1 实施日志: elevator0714 数据集 7D 关键点生成

> **执行时间**: 2026-09-03
> **执行环境**: conda env `itvlaGp`, GPU server
> **操作者**: Claude Code
> **依据文档**: [dta_3dtrj_E2impl.md](dta_3dtrj_E2impl.md)

---

## 目录

1. [环境准备](#1-环境准备)
2. [原始数据集检查](#2-原始数据集检查)
3. [脚本创建](#3-脚本创建)
4. [关键点生成执行](#4-关键点生成执行)
5. [生成后数据验证](#5-生成后数据验证)
6. [元数据验证](#6-元数据验证)
7. [源数据集完整性确认](#7-源数据集完整性确认)
8. [遇到的错误与修复记录](#8-遇到的错误与修复记录)
9. [最终产出汇总](#9-最终产出汇总)

---

## 1. 环境准备

### 1.1 环境检查

**命令**:

```bash
conda run -n itvlaGp python -c "import pinocchio as pin; print(f'pinocchio {pin.__version__}')"
conda run -n itvlaGp python -c "import pandas, numpy, torch; print(f'pandas={pandas.__version__}, numpy={numpy.__version__}, torch={torch.__version__}')"
```

**结果**:

| 依赖 | 版本 | 状态 |
|------|------|------|
| pinocchio | 4.1.0 | OK |
| pandas | 2.3.3 | OK |
| numpy | 2.2.6 | OK |
| torch | 2.11.0+cu128 | OK |

### 1.2 磁盘空间检查

**命令**:

```bash
du -sh /home/luogang/DATA/elevator0714_lerobot
df -h /home/luogang/DATA/
```

**结果**:

- 原始数据集大小: **1.4 GB** (比手册预期的 300 MB 大, 因为包含 200 个 parquet 文件而非 100 个)
- 可用空间: **131 GB** (充足)

---

## 2. 原始数据集检查

### 2.1 运行 precheck 脚本

**命令**:

```bash
python util_scripts/precheck_r1pro_dataset.py --dataset /home/luogang/DATA/elevator0714_lerobot
```

**关键结果**:

| 检查项 | 预期 | 实际 | 状态 |
|--------|------|------|------|
| `robot_type` | `r1_pro` | `r1_pro` | OK |
| `total_episodes` | 100 | 100 | OK |
| `total_frames` | 27145 | 27145 | OK |
| `fps` | 15 | 15 | OK |
| left_arm shape | [7] | [7] | OK |
| right_arm shape | [7] | [7] | OK |
| right arm std | >0.1 | mean std=0.179 | OK (双臂均活跃) |
| torso std | 全零 | 全零 | OK |
| chassis velocities std | 全零 | 全零 | OK |

**注意**: precheck 报告 `frames total = 54290` (扫描所有 parquet 文件), 这是因为数据集 `data/chunk-000/` 目录下有 200 个 parquet 文件 — 100 个以序号命名 (`file-001.parquet` ~ `file-100.parquet`) 和 100 个以时间戳命名 (`file-150711.parquet` 等). 每一对包含相同 episode 的数据. `info.json` 中的 `total_frames=27145` 只计数 `data_path` 模式 (`file-{file_index:03d}.parquet`) 匹配的文件.

---

## 3. 脚本创建

### 3.1 生成脚本

**文件**: `util_scripts/generate_r1pro_keypoints_e1.py`

基于实施手册 §3.2 创建, 核心功能:

- `R1ProFKExtractorE1` 类: 通过 Pinocchio FK 计算 16 个关键点的 7D 表示 (3D 位置 + 4D 四元数)
- 四元数从 `pin.Quaternion(oMf.rotation)` 获取, 属性 `.x`, `.y`, `.z`, `.w`
- 半球归一化: `if qw < 0: raw_q = -raw_q`
- 双 pass 管道: Pass 1 计算全局 bounding box → R_pad; Pass 2 写入归一化后的 7D 关键点
- 位置归一化: `pos / R_pad` (isotropic)
- 输出: `observation.keypoint_3d [112]` (16 × 7 展平)

### 3.2 验证脚本

**文件**: `util_scripts/verify_e1_keypoints.py`

基于实施手册 §5.1 创建, 包含 7 项检查:

1. Shape: 每帧 [16, 7]
2. Position bounds: |pos| ≤ 1.01
3. Quaternion unit norm: |‖q‖-1| ≤ 0.001
4. Hemisphere constraint: qw ≥ 0
5. Temporal smoothness: quaternion jump < 0.5
6. FK reproducibility: 10 随机帧重算误差 ≤ 1e-5
7. Per-dimension statistics

---

## 4. 关键点生成执行

### 4.1 执行命令

```bash
conda run -n itvlaGp python util_scripts/generate_r1pro_keypoints_e1.py \
    --source /home/luogang/DATA/elevator0714_lerobot \
    --dest /home/luogang/DATA/elevator0714_lerobot_4D \
    --urdf assets/r1_pro_with_gripper.urdf \
    --force
```

### 4.2 执行结果

**Pass 1 (全局 bounding box 计算)**:

| 指标 | 值 |
|------|-----|
| 扫描文件数 | 200 |
| 总帧数 | 54290 |
| Global min (base-rel) | [-0.1663, -0.3240, 1.0981] |
| Global max (base-rel) | [0.4504, 0.4180, 1.4701] |
| R_pad | **1.690601 m** (margin=15%) |
| 归一化后位置范围 | [-0.6495, 0.8696] |
| qw_min | 0.229339 (> 0, 半球约束满足) |
| qw_max | 1.000000 |
| 四元数范数误差最大值 | 5.96e-08 |

**Pass 2 (归一化 + 写入)**:

- 200 个 parquet 文件全部成功写入
- 0 个 OOB 警告
- 0 个四元数范数错误
- 总帧数: 54290 (Pass 1 与 Pass 2 一致)

**元数据写入**:

- `info.json` 更新: 添加 `observation.keypoint_3d` feature (shape=[112], dtype=float32)
- `keypoints_meta.json` 创建: bbox_radius=1.690601, keypoint_dim=7, rotation_representation=quaternion_xyzw_hemisphere

---

## 5. 生成后数据验证

### 5.1 运行验证脚本

**命令**:

```bash
conda run -n itvlaGp python util_scripts/verify_e1_keypoints.py \
    --dataset /home/luogang/DATA/elevator0714_lerobot_4D \
    --urdf assets/r1_pro_with_gripper.urdf
```

### 5.2 检查结果

| # | 检查项 | 结果 | 详细数据 |
|---|--------|------|---------|
| 1 | Shape | **PASS** | 54290 帧, 每帧 [16, 7] |
| 2 | Position bounds | **PASS** | max\|pos\| = 0.869565 (< 1.01) |
| 3 | Quaternion unit norm | **PASS** | mean err=1.04e-08, max err=5.96e-08 (< 0.001) |
| 4 | Hemisphere constraint | **PASS** | qw_min=0.229339, violations=0 |
| 5 | Temporal smoothness | **PASS** | max jump=0.198210 (< 0.5), jumps>0.5: 0 |
| 6 | FK reproducibility | **PASS** | max err=0.00e+00 (完美复现) |
| 7 | Statistics | **PASS** | 见下表 |

**Per-dimension statistics**:

| dim | mean | std | min | max |
|-----|------|-----|-----|-----|
| px | +0.039064 | 0.085291 | -0.098366 | +0.266436 |
| py | +0.018802 | 0.137294 | -0.191627 | +0.247239 |
| pz | +0.771008 | 0.064910 | +0.649513 | +0.869565 |
| qx | -0.010267 | 0.127992 | -0.338812 | +0.362903 |
| qy | -0.526155 | 0.342596 | -0.950635 | +0.139754 |
| qz | -0.003890 | 0.101133 | -0.279332 | +0.379424 |
| qw | +0.730894 | 0.211783 | +0.229339 | +1.000000 |

**统计量分析**:

- **位置分量** (px, py, pz): mean 接近 0 (px, py) 或偏正 (pz ≈ 0.77, 因为手臂在 base_link 上方约 1.1-1.5m 处工作, 除以 R_pad=1.69 后得到 ~0.77), std < 0.14, 范围严格在 [-1, 1] 内
- **四元数分量** (qx, qy, qz, qw): qw 均 > 0 (半球约束满足), qy 均值偏负 (-0.526) 说明手臂主要朝下方/前方; 所有分量在 [-1, 1] 范围内
- **整体**: 数据质量良好, 无异常值

---

## 6. 元数据验证

### 6.1 keypoints_meta.json

```json
{
  "bbox_radius": 1.6906009316444395,
  "bbox_margin": 0.15,
  "global_min_base_relative": [-0.1663, -0.3240, 1.0981],
  "global_max_base_relative": [0.4504, 0.4180, 1.4701],
  "normalization": "base_link_origin_isotropic",
  "keypoint_dim": 7,
  "keypoint_dim_layout": "px,py,pz,qx,qy,qz,qw",
  "rotation_representation": "quaternion_xyzw_hemisphere",
  "rotation_convention": "qw >= 0; negate if qw < 0",
  "num_keypoints": 16,
  "total_frames": 54290,
  "torso_q": [0.0, 0.0, 0.0, 0.0],
  "urdf": "assets/r1_pro_with_gripper.urdf"
}
```

### 6.2 info.json 中的 observation.keypoint_3d

- shape: [112]
- dtype: float32
- names count: 112
- 首个关键点名称: `left_arm_link1_px`, `left_arm_link1_py`, `left_arm_link1_pz`, `left_arm_link1_qx`, `left_arm_link1_qy`, `left_arm_link1_qz`, `left_arm_link1_qw`

---

## 7. 源数据集完整性确认

**命令**:

```bash
python3 -c "
import pandas as pd
pq = pd.read_parquet('/home/luogang/DATA/elevator0714_lerobot/data/chunk-000/file-001.parquet')
assert 'observation.keypoint_3d' not in pq.columns
print('Source dataset untouched')
"
```

**结果**: 源数据集 `/home/luogang/DATA/elevator0714_lerobot/` 未被修改, 不包含 `observation.keypoint_3d` 列.

---

## 8. 遇到的错误与修复记录

### 8.1 Error 1: `SyntaxError: name 'BBOX_MARGIN' is used prior to global declaration`

**发生阶段**: 首次运行生成脚本

**错误原因**: `generate_r1pro_keypoints_e1.py` 中, `BBOX_MARGIN` 作为模块级常量已在函数签名默认值 (`def compute_r_pad(..., margin=BBOX_MARGIN)`) 中使用, 随后在 `main()` 函数内又声明 `global BBOX_MARGIN`. Python 不允许在函数内使用 `global` 声明一个已在该函数外部作为默认参数值使用的名称.

**根因分析**: 实施手册 §3.2 的脚本模板直接使用了 `global BBOX_MARGIN` 来让 CLI 的 `--bbox-margin` 参数覆盖模块级默认值. 但 Python 语法规则要求如果一个名称在函数中被声明为 `global`, 它不能在该名称被声明之前就在同一作用域中被引用 — 而函数签名中的默认参数在函数定义时就已求值.

**修复方案**:

1. 删除 `global BBOX_MARGIN` 声明
2. 改用局部变量 `bbox_margin = args.bbox_margin`
3. 将 `compute_r_pad()` 调用中的 `margin=BBOX_MARGIN` 改为 `margin=bbox_margin`
4. 给 `_write_meta()` 增加 `bbox_margin` 参数, 在 `main()` 中传入 `bbox_margin=bbox_margin`

**修改文件**: `util_scripts/generate_r1pro_keypoints_e1.py`

**具体改动**:

```python
# 改动 1: main() 中
# 原:
#     global BBOX_MARGIN
#     BBOX_MARGIN = args.bbox_margin
# 改为:
    bbox_margin = args.bbox_margin

# 改动 2: main() 中 compute_r_pad 调用
# 原:
#     r_pad = compute_r_pad(global_min, global_max, margin=BBOX_MARGIN)
# 改为:
    r_pad = compute_r_pad(global_min, global_max, margin=bbox_margin)

# 改动 3: main() 中 R_pad 日志
# 原:
#     logger.info("R_pad = %.6f m (margin=%.0f%%)", r_pad, BBOX_MARGIN * 100)
# 改为:
    logger.info("R_pad = %.6f m (margin=%.0f%%)", r_pad, bbox_margin * 100)

# 改动 4: _write_meta 函数签名
# 原:
#     def _write_meta(dest, r_pad, global_min, global_max, total_frames, torso_q, urdf_path):
# 改为:
    def _write_meta(dest, r_pad, global_min, global_max, total_frames, torso_q, urdf_path, bbox_margin=BBOX_MARGIN):

# 改动 5: main() 中 _write_meta 调用
# 原:
#     _write_meta(dest, r_pad, ..., args.urdf)
# 改为:
    _write_meta(dest, r_pad, ..., args.urdf, bbox_margin=bbox_margin)
```

### 8.2 Error 2: `TypeError: 'float' object is not callable`

**发生阶段**: Pass 1 第一帧的 FK 计算

**错误位置**: `generate_r1pro_keypoints_e1.py`, `R1ProFKExtractorE1.compute()` 方法

**错误代码**:

```python
raw_q = np.array([quat.x(), quat.y(), quat.z(), quat.w()], dtype=np.float32)
```

**错误原因**: Pinocchio 4.1.0 中 `pin.Quaternion` 的 `x`, `y`, `z`, `w` 是 **属性 (property)** 而非方法 (method). 实施手册 §3.2 中的代码使用了函数调用语法 `quat.x()`, 但实际上应该是属性访问 `quat.x`.

**根因分析**: 这是 Pinocchio API 版本差异. 早期版本或某些绑定 (如 eigenpy) 中 `Quaternion` 的分量可能是方法, 但 Pinocchio 4.x 版本使用 Python property 暴露这些值.

**修复方案**: 将 `quat.x()` 改为 `quat.x` (去掉括号), 其余三个分量同理.

**修改文件**: `util_scripts/generate_r1pro_keypoints_e1.py`

**具体改动**:

```python
# 原:
raw_q = np.array([quat.x(), quat.y(), quat.z(), quat.w()], dtype=np.float32)
# 改为:
raw_q = np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float32)
```

### 8.3 错误汇总

| # | 错误类型 | 阶段 | 根因 | 修复 | 重试次数 |
|---|---------|------|------|------|---------|
| 1 | SyntaxError | 脚本加载 | Python global 声明语法限制 | 改用局部变量 + 参数传递 | 1 |
| 2 | TypeError | Pass 1 FK 计算 | Pinocchio 4.x 中 Quaternion 分量是属性非方法 | 去掉括号 | 1 |

两个错误均在第一次重试后修复, 第三次运行成功完成全部流程.

---

## 9. 最终产出汇总

### 9.1 新增文件

| 文件 | 路径 | 行数 | 用途 |
|------|------|------|------|
| 生成脚本 | `util_scripts/generate_r1pro_keypoints_e1.py` | ~280 | E1 7D 关键点离线生成 |
| 验证脚本 | `util_scripts/verify_e1_keypoints.py` | ~180 | 生成后 7 项验证 |

### 9.2 输出数据集

| 属性 | 值 |
|------|-----|
| 路径 | `/home/luogang/DATA/elevator0714_lerobot_4D/` |
| 大小 | ~1.5 GB (原始 1.4 GB + 关键点增量) |
| parquet 文件数 | 200 |
| 总帧数 | 54290 |
| 新增列 | `observation.keypoint_3d` [112] (float32) |
| R_pad | 1.690601 m |
| 位置范围 | [-0.8696, 0.8696] |
| 半球约束 | qw ≥ 0.229 (全部满足) |
| 四元数范数误差 | max 5.96e-08 |

### 9.3 验收状态

| 检查项 | 状态 |
|--------|------|
| 7 项验证全部通过 | **PASS** |
| 源数据集未修改 | **PASS** |
| 元数据完整 (info.json + keypoints_meta.json) | **PASS** |
| 零 OOB 警告 | **PASS** |
| 零四元数范数错误 | **PASS** |
| 零时序跳变 | **PASS** |
| FK 完美可复现 (误差=0) | **PASS** |

### 9.4 待完成事项

按照实施手册, 以下步骤尚未执行 (需要用户确认是否继续):

- [ ] §6: norm_stats 统计量生成 (`compute_norm_stats_single.py`)
- [ ] §7: 模型侧代码改动 (3 个文件的配置/模型/transform 修改)
- [ ] §8: Smoke test (100 步端到端训练验证)

这些步骤涉及训练框架代码修改和 GPU 训练, 需要根据用户决定是否立即执行.
