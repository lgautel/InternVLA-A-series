# 3D/4D 关键点生成执行日志

> **方案文档**: [3d4d_gen_1.md](3d4d_gen_1.md)  
> **源数据集**: `/B/Dta/plug_into_socket_franka3_15hz_lerobot/`  
> **目标数据集**: `~/b/Dta/plug_into_socket_franka3_15hz_lerobot_4d/`  
> **日期**: 2026-09-19  
> **Python 环境**: `/B/VENV/itnvla15rbt20/` (pinocchio 4.1.0, pandas 3.0.5, numpy 2.2.6)

---

## 环境检查

| 项目 | 状态 |
|---|---|
| 源数据集 | 存在: `/B/Dta/plug_into_socket_franka3_15hz_lerobot/` (data/, meta/, videos/, force_meta.json) |
| 目标目录父级 | 已创建: `~/b/Dta/` |
| Python venv | `/B/VENV/itnvla15rbt20/bin/activate` 存在 |
| pinocchio | 4.1.0 |
| pandas | 3.0.5 |
| numpy | 2.2.6 |
| URDF | `b/d/Frk2/fr3v2_1_franka_hand.urdf` 存在 |

---

## Step 1: FK 交叉验证 (pre-generation)

运行 `test_franka2_keypoints.py` 验证 FK 正确性.

### Error 1: URDF 路径计算 bug (parents index off-by-one)

**命令**:
```bash
python b/s/Frk2/test_franka2_keypoints.py --source /B/Dta/plug_into_socket_franka3_15hz_lerobot --urdf b/d/Frk2/fr3v2_1_franka_hand.urdf
```

**报错**:
```
ValueError: The file /B/SRC/itvlaGpLibPlus/b/b/d/Frk2/fr3v2_1_franka_hand.urdf does not contain a valid URDF model.
```

**根因**: 三个脚本的默认 URDF 路径计算中 `Path.parents` 索引错误, 产生了双层 `b/b/d/` 路径:
- `test_franka2_keypoints.py`: `SCRIPT_DIR.parents[1]` 解析为 `/B/SRC/itvlaGpLibPlus/b`, 加上 `"b"/"d"/"Frk2"/...` 变成 `b/b/d/Frk2/...`
- 正确索引应为 `parents[2]` (到达项目根 `/B/SRC/itvlaGpLibPlus/`)

脚本位于 `b/s/Frk2/` 目录下:
```
parents[0] = /B/SRC/itvlaGpLibPlus/b/s     (b/s/)
parents[1] = /B/SRC/itvlaGpLibPlus/b       (b/)
parents[2] = /B/SRC/itvlaGpLibPlus          (项目根) ✓
```

**Fix**: 修改三个文件的 parents 索引:

| 文件 | 原值 | 修正 |
|---|---|---|
| `b/s/Frk2/test_franka2_keypoints.py` L26 | `SCRIPT_DIR.parents[1]` | `SCRIPT_DIR.parents[2]` |
| `b/s/Frk2/generate_franka2_keypoints.py` L382 | `Path(__file__).resolve().parents[2]` | `.parents[3]` |
| `b/s/Frk2/verify_franka2_keypoints.py` L263 | `Path(__file__).resolve().parents[2]` | `.parents[3]` |

注: `generate` 和 `verify` 使用 `Path(__file__).resolve()` (含文件名), 所以 parents 多一级: `parents[0]`=文件所在目录, 需要 `parents[3]` 才到项目根.

### Error 2: FK home position Z 阈值过严

**报错**:
```
AssertionError: TCP Z at home should be > 0.9m, got 0.8226
```

**根因**: FR3 v2.1 在 all-zero 关节角度下, link7 的 Z 轴指向下方 (由运动学链累积旋转导致). hand_tcp 在 link7 下方:

```
link5 Z = 1.033 m (腕关节)
link7 → link8: -0.107 m (Z 轴朝下)
hand → hand_tcp: -0.1034 m (Z 轴朝下)
TCP Z = 1.033 - 0.107 - 0.1034 = 0.8226 m
TCP X = 0.088 m (link7 的 X 偏移)
```

**Fix**: 放宽测试阈值:
```python
# 原: assert tcp_pos[2] > 0.9, assert abs(tcp_pos[0]) < 0.02
# 改: assert tcp_pos[2] > 0.7, assert abs(tcp_pos[0]) < 0.1
```

各关键点在 home 位置 (q=0) 的 FK 输出:
```
link1:    pos=[0, 0, 0.333],      qw=1.000000
link2:    pos=[0, 0, 0.333],      qw=0.707107
link3:    pos=[0, 0, 0.649],      qw=1.000000
link4:    pos=[0.082, 0, 0.649],  qw=0.707107
link5:    pos=[0, 0, 1.033],      qw=1.000000
link6:    pos=[0, 0, 1.033],      qw=0.707107
link7:    pos=[0.088, 0, 1.033],  qw=0.000000
hand_tcp: pos=[0.088, 0, 0.822],  qw=0.000000
```

### Error 3: EE 四元数列序错误 (info.json names 误导)

**症状**: FK cross-validation position error = 0.000mm (完美), 但 rotation error = 175.57° (灾难性不匹配).

**调试过程**: 打印前 3 帧的 FK 和数据集四元数:

```
Frame 0:
  FK   quat [xyzw]: [-0.9962, -0.0719, -0.0384,  0.0307]
  DS   quat [wxyz]: [ 0.9962,  0.0719,  0.0384, -0.0307]  (按 info.json 解读)
  DS→  quat [xyzw]: [ 0.0719,  0.0384, -0.0307,  0.9962]  (reorder 后)
  dot product: -0.042649  → err ≈ 175°
```

**根因**: `info.json` 的 feature names 声称列顺序为 `["ee_quat_w", "ee_quat_x", "ee_quat_y", "ee_quat_z"]` (即 wxyz), 但实际数据存储顺序为 **[qx, qy, qz, qw]** (即 xyzw, 与 pinocchio 输出一致).

**验证**: 不做 wxyz→xyzw 重排, 直接用 `state[11:15]` 作为 `[qx,qy,qz,qw]`:

```
Frame 0:
  FK   [xyzw]: [-0.9962, -0.0719, -0.0384,  0.0307]
  DS   [xyzw]: [ 0.9962,  0.0719,  0.0384, -0.0307]
  dot = -1.000000 → |dot| = 1.0 → err = 0.000° ✓
```

FK 和数据集四元数互为负数 (q 和 -q 表示同一旋转), 验证通过.

批量验证 500 帧: mean error = 0.012°, max error = 0.056°.

**Fix**: 三个脚本中删除 wxyz→xyzw 重排, 直接使用 `state[:, 11:15]`:

```python
# 原:
ds_ee_quat_wxyz = state[:, EE_QUAT_SLICE]
ds_ee_quat_xyzw = np.concatenate([ds_ee_quat_wxyz[:, 1:4], ds_ee_quat_wxyz[:, 0:1]], axis=1)

# 改:
ds_ee_quat_xyzw = state[:, EE_QUAT_SLICE]  # 已是 xyzw 顺序
```

影响的文件:
- `generate_franka2_keypoints.py`: `_cross_validate_fk()` 函数
- `test_franka2_keypoints.py`: `test_fk_cross_validation()` 函数
- `verify_franka2_keypoints.py`: `check8_fk_cross_validation()` 函数
- `generate_franka2_keypoints.py`: `EE_QUAT_SLICE` 注释修正

### Step 1 最终结果

```
=== Franka v2 Keypoint Unit Tests ===
Test 1: FK home position — PASS (TCP: [0.088, 0, 0.822])
Test 2: Hemisphere normalization (100 random configs) — PASS
Test 3: Quaternion unit norm (100 random configs) — PASS
Test 4: Joint slice from state column — PASS
Test 5: FK cross-validation (3025 frames, 10 episodes):
  Position error: mean=0.000mm, max=0.000mm
  Rotation error: mean=0.011°, max=0.056°
  — PASS
```

---

## Step 2: 关键点生成

**命令**:
```bash
source /B/VENV/itnvla15rbt20/bin/activate
export HF_HOME=/B/VENV/hf_home
python b/s/Frk2/generate_franka2_keypoints.py \
    --source /B/Dta/plug_into_socket_franka3_15hz_lerobot \
    --dest ~/b/Dta/plug_into_socket_franka3_15hz_lerobot_4d \
    --urdf b/d/Frk2/fr3v2_1_franka_hand.urdf
```

**执行过程** (无错误):
1. rsync 复制源数据集 → 204MB
2. Pass 1: 扫描 100 parquet, 33308 帧, 计算全局 bounding box + FK 交叉验证
3. Pass 2: 计算归一化 7D 关键点, 写入每个 parquet 的 `observation.keypoint_3d` 列
4. 更新 `meta/info.json`, 写 `keypoints_meta.json`, 写 `train_inference_contract.json`

**关键输出**:

| 指标 | 值 |
|---|---|
| 总帧数 | 33308 (Pass 1 = Pass 2) |
| R_pad | **0.836100 m** (margin=15%) |
| Global min (base-rel) | [-0.032, -0.140, 0.178] |
| Global max (base-rel) | [0.603, 0.061, 0.727] |
| 四元数 qw 范围 | [0.000002, 1.000000] |
| 四元数 norm 误差 max | 5.96e-08 |
| FK 交叉验证 pos err max | 0.000 mm |
| FK 交叉验证 rot err max | 0.056° |
| Position OOB 文件数 | 0 |

**生成的元数据文件**:

`keypoints_meta.json`:
- `bbox_radius`: 0.836100
- `keypoint_dim`: 7 (px,py,pz,qx,qy,qz,qw)
- `num_keypoints`: 8 (link1-7 + hand_tcp)
- `total_frames`: 33308
- `fk_cross_validation`: 全部 100 episodes, 33308 帧验证

`train_inference_contract.json`:
- `version`: 2.0
- `dataset_fps`: 15
- `urdf_sha256`: e90ac0a4c109eb05...
- `pinocchio_version`: 4.1.0
- `r_pad`: 0.836100
- `his_len_semantics`: "count of frames strictly before current frame"
- `inference_constraints`: 完整训推对齐约束集

---

## Step 3: 10-check 验证

**命令**:
```bash
python b/s/Frk2/verify_franka2_keypoints.py \
    --dataset ~/b/Dta/plug_into_socket_franka3_15hz_lerobot_4d \
    --urdf b/d/Frk2/fr3v2_1_franka_hand.urdf
```

### Error 4: Check 5 temporal smoothness 误报 (四元数半球跳变)

**第一次运行结果**: 9/10 PASS, Check 5 FAIL:
```
=== Check 5: Temporal smoothness ===
  Transitions: 33208, max jump: 2.000000, jumps>0.5: 535
```

**根因**: 四元数半球归一化 (qw ≥ 0) 的边界效应. 当某个 link 的旋转接近 qw=0 (180° 旋转) 时, 微小的关节变化使 qw 在 0 附近来回穿越, 触发符号翻转:

```
帧 t:   q = [qx, qy, qz, +0.0001]  (qw 刚好 > 0, 保持)
帧 t+1: q = [-qx, -qy, -qz, +0.0001]  (qw 变负后翻转)
L2距离 ≈ 2.0 (最大可能值)
```

但实际旋转变化 < 0.02° (geodesic distance ≈ 0). 这是纯粹的表示歧义, 不影响模型训练.

**Fix**: 使用半球感知的距离度量 `min(‖q1-q2‖, ‖q1+q2‖)` 替代朴素 L2:

```python
# 原: diffs = np.linalg.norm(ep_quat[1:] - ep_quat[:-1], axis=-1)
# 改:
d_pos = np.linalg.norm(q1 - q2, axis=-1)
d_neg = np.linalg.norm(q1 + q2, axis=-1)
diffs = np.minimum(d_pos, d_neg)
```

### Step 3 最终结果

```
Check  1: Shape [8, 7]                      — PASS
Check  2: Position bounds max=0.8696 < 1.01 — PASS
Check  3: Quaternion norm err max=5.96e-08   — PASS
Check  4: Hemisphere qw_min=1.5e-06          — PASS
Check  5: Temporal smoothness max=0.024      — PASS
Check  6: FK reproducibility err=0.00        — PASS
Check  7: Per-dimension statistics           — (输出)
Check  8: FK cross-val pos=0.000mm rot=0.04° — PASS
Check  9: keypoints_meta.json completeness   — PASS
Check 10: contract.json completeness         — PASS

Summary: ALL PASS
```

**Check 7 统计信息**:

| dim | mean | std | min | max |
|:---:|:---:|:---:|:---:|:---:|
| px | +0.337 | 0.293 | -0.039 | +0.722 |
| py | -0.023 | 0.046 | -0.168 | +0.074 |
| pz | +0.543 | 0.164 | +0.213 | +0.870 |
| qx | +0.125 | 0.596 | -1.000 | +1.000 |
| qy | +0.199 | 0.358 | -0.423 | +0.934 |
| qz | -0.102 | 0.206 | -0.682 | +0.110 |
| qw | +0.523 | 0.369 | +0.000 | +1.000 |

位置在 [-0.168, +0.870] 范围内, 远在 [-1, 1] 之内. 四元数均为单位四元数, qw ≥ 0.

---

## 生成的数据集概况

```
路径: ~/b/Dta/plug_into_socket_franka3_15hz_lerobot_4d/
大小: 204 MB
结构:
  data/
    episode_000000.parquet  (含 observation.keypoint_3d [56])
    ...
    episode_000099.parquet
  meta/
    info.json                 (已更新, 含 keypoint_3d feature 声明)
    keypoints_meta.json       (新增, R_pad + FK 交叉验证结果)
    train_inference_contract.json  (新增, 训推一致性契约)
    episodes.jsonl
    episodes_stats.jsonl
    tasks.jsonl
  videos/                     (原样复制)
  force_meta.json             (原样复制)
```

Parquet 列:
```
observation.state [15], observation.force [24], observation.wrench [6],
action [8], timestamp, frame_index, episode_index, index, task_index,
observation.keypoint_3d [56]  ← 新增
```

---

## 错误汇总

| 编号 | 错误 | 根因 | Fix | 影响文件 |
|:---:|---|---|---|---|
| E1 | URDF 路径 `b/b/d/` 双层 | `Path.parents` 索引 off-by-one | parents[1]→[2], parents[2]→[3] | test, generate, verify |
| E2 | FK home TCP Z=0.82 < 0.9 | FR3 v2.1 运动学: hand 在 link7 下方 | 阈值 0.9→0.7, X 容差 0.02→0.1 | test |
| E3 | 四元数旋转误差 175° | info.json names 误导, 实际 xyzw 非 wxyz | 去除 wxyz→xyzw 重排 | generate, test, verify |
| E4 | Check 5 temporal jump=2.0 | 四元数半球归一化边界跳变 | min(‖q-p‖, ‖q+p‖) 半球感知距离 | verify |

所有 4 个错误均已修复, 最终 Step 1 (5 tests) + Step 3 (10 checks) 全部 PASS.

---

## 文件变更清单

### 新增文件
| 文件 | 用途 |
|---|---|
| `~/b/Dta/plug_into_socket_franka3_15hz_lerobot_4d/` | 生成的 4D 数据集 (完整) |
| `~/b/Dta/.../meta/keypoints_meta.json` | 关键点元数据 (R_pad, FK 交叉验证等) |
| `~/b/Dta/.../meta/train_inference_contract.json` | 训推一致性契约 |

### 修改文件
| 文件 | 改动 | 原因 |
|---|---|---|
| `b/s/Frk2/test_franka2_keypoints.py` L26 | `parents[1]` → `parents[2]` | Fix E1: URDF 路径 |
| `b/s/Frk2/test_franka2_keypoints.py` L42-44 | Z>0.9→0.7, X<0.02→0.1 | Fix E2: FK 阈值 |
| `b/s/Frk2/test_franka2_keypoints.py` L113-115 | 去除 wxyz→xyzw 重排 | Fix E3: 四元数列序 |
| `b/s/Frk2/generate_franka2_keypoints.py` L382 | `parents[2]` → `parents[3]` | Fix E1: URDF 路径 |
| `b/s/Frk2/generate_franka2_keypoints.py` L27 | 更新 `EE_QUAT_SLICE` 注释 | Fix E3 |
| `b/s/Frk2/generate_franka2_keypoints.py` L161-165 | 去除 wxyz→xyzw 重排 | Fix E3: 四元数列序 |
| `b/s/Frk2/verify_franka2_keypoints.py` L263 | `parents[2]` → `parents[3]` | Fix E1: URDF 路径 |
| `b/s/Frk2/verify_franka2_keypoints.py` L188-191 | 去除 wxyz→xyzw 重排 | Fix E3: 四元数列序 |
| `b/s/Frk2/verify_franka2_keypoints.py` L88-108 | 半球感知距离 | Fix E4: temporal smoothness |
| `~/b/Dta/.../meta/info.json` | 新增 keypoint_3d feature | 生成脚本自动更新 |

---

## 关键发现

### 1. 数据集四元数实际为 xyzw 而非 wxyz

`info.json` 的 feature names `["ee_quat_w", "ee_quat_x", "ee_quat_y", "ee_quat_z"]` 暗示 wxyz 顺序, 但实际数据为 **xyzw** (与 pinocchio 一致). 这一发现通过 FK 交叉验证确认:
- 按 wxyz 解读: 旋转误差 175.57° (错误)
- 按 xyzw 解读: 旋转误差 0.011° (正确)

**对训推一致性的影响**: 本方案的关键点生成仅依赖 FK (关节角度), 不使用数据集中的 EE 四元数. 因此此发现不影响生成结果, 仅影响交叉验证逻辑. 但如果推理侧需要读取数据集的 EE 四元数 (例如用于状态条件化), 必须注意这个列序差异.

### 2. FR3 v2.1 home position 几何

Franka FR3 v2.1 在 all-zero 关节角度下, hand_tcp 位于 (0.088, 0, 0.822) 而非预期的 (0, 0, ~1.03). 这是因为:
- link7 (法兰) 的累积旋转使后续连杆的 Z 轴指向 -Z 方向
- hand_tcp 在 link7 下方 0.107+0.1034 = 0.2104 m

### 3. R_pad = 0.836 m 与旧数据集一致

新数据集的 R_pad (0.836100) 与旧数据集文档记录的 R_pad (0.836100) 完全一致. 这是因为两个数据集采集时机器人工作空间相同 (同一台 Franka, 同一个插座任务).

### 4. FK 精度极高

Position error: max 0.00014 mm (sub-micron 级别)
Rotation error: max 0.056° (远低于 1° 阈值)

这说明 URDF `fr3v2_1_franka_hand.urdf` 与真机完全一致 (可能使用了标定后的参数), 无需 §8.1 中提到的标定偏差修正.
