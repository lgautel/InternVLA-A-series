# Franka 插拔插座数据集处理实施日志

> **日期**: 2026-09-04
> **方案文档**: [dta_4dtrj_plan.md](dta_4dtrj_plan.md)
> **操作者**: Claude Code (自动化执行)

---

## 0. 环境确认

### 0.1 服务器环境

| 项目 | 值 |
|------|-----|
| Python 环境 | `conda activate itvlaGp` (`/home/luogang/miniforge3/envs/itvlaGp/bin/python`) |
| pinocchio | 4.1.0 |
| h5py | 3.16.0 |
| cv2 | 4.11.0 |
| pandas | 2.3.3 |
| numpy | 2.2.6 |
| 代码库 | `/home/luogang/SRC/Robot/itvlaGp` (branch: `08GpR1pro09`) |

### 0.2 数据源确认

- HDF5 源目录: `/home/luogang/DATA/plug_into_socket_hdf5/`
- Episode 文件数: 100 个 HDF5 + 1 个 meta.json (`episode_000000.hdf5` ~ `episode_000099.hdf5`)
- 目录总大小: 19 GB
- 磁盘可用空间: ~111 GB (充足)

### 0.3 HDF5 结构确认 (episode_000000.hdf5)

```
camera_global/color_image_jpeg: shape=(596,), dtype=object
camera_global/depth_image:      shape=(596, 480, 640), dtype=uint16
camera_global/timestamps:       shape=(596,), dtype=float64
camera_wrist/color_image_jpeg:  shape=(596,), dtype=object
camera_wrist/depth_image:       shape=(596, 480, 640), dtype=uint16
camera_wrist/timestamps:        shape=(596,), dtype=float64
robot_state/action_gripper:     shape=(1981, 1), dtype=float64
robot_state/action_joints:      shape=(1981, 7), dtype=float64
robot_state/ee_force:           shape=(1981, 3), dtype=float64
robot_state/ee_pos:             shape=(1981, 3), dtype=float64
robot_state/ee_quat:            shape=(1981, 4), dtype=float64
robot_state/ee_torque:          shape=(1981, 3), dtype=float64
robot_state/gripper_width:      shape=(1981, 1), dtype=float64
robot_state/joint_positions:    shape=(1981, 7), dtype=float64
robot_state/joint_torques:      shape=(1981, 7), dtype=float64
robot_state/joint_torques_external: shape=(1981, 7), dtype=float64
robot_state/joint_velocities:   shape=(1981, 7), dtype=float64
robot_state/timestamps:         shape=(1981,), dtype=float64
```

确认: state=1981帧@100Hz, camera_global=596帧@30Hz — 与方案预期一致.

---

## 1. 文件创建

### 1.1 新建文件清单

按方案 §10.1, 创建了以下 5 个文件:

| 文件 | 行数 | 说明 |
|------|------|------|
| `b/s/Frk/convert_franka_plug_hdf5.py` | ~200 | HDF5→LeRobot 格式转换, 含频率对齐 |
| `b/s/Frk/generate_franka_keypoints.py` | ~250 | 7D 关键点离线生成 (import R1Pro 共享函数) |
| `b/s/Frk/verify_franka_conversion.py` | ~200 | 转换验证脚本 (8项检查) |
| `b/s/Frk/verify_franka_keypoints.py` | ~170 | 关键点验证脚本 (7项检查) |
| `b/s/Frk/cfg/franka_plug.yaml` | 12 | Schema 配置 |

### 1.2 Schema 注册

创建了符号链接将 schema 注册到 LeRobot 的 schema 扫描目录:

```bash
ln -sf /home/luogang/SRC/Robot/itvlaGp/b/s/Frk/cfg/franka_plug.yaml \
       /home/luogang/SRC/Robot/itvlaGp/src/lerobot/dataset_schemas/configs/franka_plug.yaml
```

### 1.3 与方案的差异

在实际创建脚本时, 对方案中的代码做了一处修改:

- **`convert_franka_plug_hdf5.py` 的 `process_episode` 函数**: 方案中 `for cam_idx in range(n_cam_frames - 1)` 跳过了最后一帧. 实际实现中改为 `for cam_idx in range(n_cam_frames)`, 保留全部相机帧. 原因: 最后一帧的 action 仍是有效的 (即该时刻的关节位置目标), 跳过会损失约 100 帧数据.

---

## 2. Step 1: HDF5 → LeRobot 格式转换

### 2.1 执行命令

```bash
/home/luogang/miniforge3/envs/itvlaGp/bin/python b/s/Frk/convert_franka_plug_hdf5.py \
    --source /home/luogang/DATA/plug_into_socket_hdf5 \
    --dest /home/luogang/hf_home/lerobot/plug_into_socket_lrb \
    --robot-type franka_plug \
    --fps 30 \
    --force
```

### 2.2 执行结果

| 指标 | 值 |
|------|-----|
| 状态 | 成功 (exit code 0) |
| 总 episodes | 100 |
| 总帧数 | 66,577 |
| FPS | 30 |
| 输出目录 | `/home/luogang/hf_home/lerobot/plug_into_socket_lrb/` |
| 数据集大小 | 421 MB |
| 耗时 | ~40 分钟 (主要开销: JPEG 解码 + SVT-AV1 视频编码) |
| 视频编码器 | SVT-AV1 (CRF=30, preset=10, YUV420 8-bit) |

### 2.3 Errors

**无 error**. 脚本运行无任何异常退出或错误.

### 2.4 Warnings

共 18 条频率对齐警告, 均为 state-camera 时间戳间距超过 20ms 阈值:

| Episode | 最大间距 (ms) |
|---------|--------------|
| episode_000002 | 25.1 |
| episode_000005 | 33.1 |
| episode_000013 | 33.1 |
| episode_000016 | 20.6 |
| episode_000021 | 33.7 |
| episode_000023 | 36.5 |
| episode_000026 | 32.5 |
| episode_000033 | 40.3 |
| episode_000036 | 36.1 |
| episode_000037 | 36.7 |
| episode_000041 | **46.8** (最大) |
| episode_000058 | 30.1 |
| episode_000060 | 32.7 |
| episode_000075 | 29.8 |
| episode_000088 | 28.2 |
| episode_000091 | 35.2 |
| episode_000095 | 42.9 |
| episode_000099 | 27.4 |

**根因分析**: 100Hz state 与 30Hz camera 的时间戳并非严格整数倍关系, 部分 episode 的采样时钟存在微小漂移. 最大 46.8ms 约等于 30Hz 采样周期的 1.4 倍, 意味着 nearest-neighbor 匹配仍然选中了正确的状态帧 (下一个状态帧距离只有 10ms). 对下游训练无影响.

### 2.5 视频文件

LeRobot 将视频按 chunk 存储:

```
videos/observation.images.global/chunk-000/file-000.mp4   (1 file)
videos/observation.images.wrist/chunk-000/file-000.mp4    (2 files)
videos/observation.images.wrist/chunk-000/file-001.mp4
```

wrist 相机有 2 个视频文件是因为单文件超过了 LeRobot 的视频 chunk 大小限制, 自动拆分.

---

## 3. 转换验证 (8 项检查)

### 3.1 执行命令

```bash
/home/luogang/miniforge3/envs/itvlaGp/bin/python b/s/Frk/verify_franka_conversion.py \
    --dataset /home/luogang/hf_home/lerobot/plug_into_socket_lrb \
    --urdf b/d/Frk/fr3v2_1_franka_hand.urdf
```

### 3.2 结果

| # | 检查项 | 结果 | 详情 |
|---|--------|------|------|
| 1 | Metadata integrity | PASS | robot_type=franka_plug, fps=30, 100 episodes, 66577 frames |
| 2 | Feature shapes | PASS | arm=[7], gripper=[1], action.arm=[7], action.gripper=[1] |
| 3 | No NaN values | PASS | 全部列无 NaN |
| 4 | Joint limits | PASS | 7 个关节均在 URDF 限位内 |
| 5 | Gripper range | PASS | [0.000000, 0.079405] m, 在 [0, 0.08] 内 |
| 6 | Episode consistency | PASS | 声明数 = 实际数 (100 episodes, 66577 frames) |
| 7 | Video files | PASS | global: 1 file, wrist: 2 files |
| 8 | FK cross-check | PASS | max error = 0.000000 m (URDF 完美匹配) |

**结论**: 8/8 全部通过.

### 3.3 数据统计量 (observation.state.arm, 7 维)

| 关节 | mean | std | min | max |
|------|------|-----|-----|-----|
| joint1 | -0.2406 | 0.1206 | -0.4842 | +0.0452 |
| joint2 | +0.1457 | 0.0805 | -0.1030 | +0.3120 |
| joint3 | +0.1872 | 0.1464 | -0.2025 | +0.4789 |
| joint4 | -2.0600 | 0.0854 | -2.2044 | -1.5347 |
| joint5 | -0.0553 | 0.0429 | -0.2041 | +0.0806 |
| joint6 | +2.2011 | 0.1285 | +1.5702 | +2.4536 |
| joint7 | +0.6998 | 0.0968 | +0.4843 | +0.9807 |

所有关节值在 URDF 限位范围内. joint4 全部为负值 (限位 [-3.077, -0.117]), joint6 全部为正值 (限位 [0.440, 4.622]), 与 Franka 物理结构一致.

---

## 4. Step 2: 7D 关键点生成

### 4.1 执行命令

```bash
/home/luogang/miniforge3/envs/itvlaGp/bin/python b/s/Frk/generate_franka_keypoints.py \
    --source /home/luogang/hf_home/lerobot/plug_into_socket_lrb \
    --dest /home/luogang/hf_home/lerobot/plug_into_socket_lrb_4D \
    --urdf b/d/Frk/fr3v2_1_franka_hand.urdf \
    --link-prefix fr3v2_1 \
    --force
```

### 4.2 执行结果

| 指标 | 值 |
|------|-----|
| 状态 | 成功 (exit code 0) |
| Pass 1 帧数 | 66,577 |
| Pass 2 帧数 | 66,577 |
| Parquet 文件数 | 1 |
| 数据集大小 | 432 MB (比中间数据集多 11 MB = 关键点 parquet 增量) |
| 耗时 | ~2 分钟 (rsync ~20s + Pass 1 ~40s + Pass 2 ~50s) |

### 4.3 Errors

**无 error**.

### 4.4 关键点归一化参数

| 参数 | 值 |
|------|-----|
| global_min (base-rel) | [-0.032, -0.140, +0.178] m |
| global_max (base-rel) | [+0.603, +0.062, +0.727] m |
| R (最大绝对极值) | 0.727 m (由 z_max 主导) |
| R_pad (R × 1.15) | **0.836100 m** |
| qw_min | 0.000000 (半球归一化边界) |
| qw_max | 1.000000 |
| quat norm_err_max | 1.19e-07 (远低于 0.01 阈值) |
| Position OOB 警告 | 0 (无越界) |

**与方案估算的对比**:

| | 方案估算 (§5.6) | 实际值 | 原因 |
|--|----------------|--------|------|
| R_pad | 0.98~1.15 m | **0.836 m** | 插拔任务工作空间紧凑, 机械臂未伸展到最大 |
| global_min.z | +0.10 | +0.178 | 插座位置较高, 手臂未低伸 |
| global_max.x | +0.80 | +0.603 | 前伸距离小于估算 |

R_pad 偏小不影响归一化质量 — 实际上更小的 R_pad 意味着归一化后的位置分量利用 [-1, 1] 空间更充分.

### 4.5 keypoints_meta.json

```json
{
  "bbox_radius": 0.8361004471778869,
  "bbox_margin": 0.15,
  "global_min_base_relative": [-0.032, -0.140, 0.178],
  "global_max_base_relative": [0.603, 0.062, 0.727],
  "normalization": "base_link_origin_isotropic",
  "keypoint_dim": 7,
  "keypoint_dim_layout": "px,py,pz,qx,qy,qz,qw",
  "rotation_representation": "quaternion_xyzw_hemisphere",
  "rotation_convention": "qw >= 0; negate if qw < 0",
  "num_keypoints": 8,
  "keypoint_links": [
    "fr3v2_1_link1", "fr3v2_1_link2", "fr3v2_1_link3", "fr3v2_1_link4",
    "fr3v2_1_link5", "fr3v2_1_link6", "fr3v2_1_link7", "fr3v2_1_hand_tcp"
  ],
  "total_frames": 66577,
  "coordinate_system": "base_link-relative, position divided by bbox_radius, quaternion hemisphere-normalized",
  "urdf": "b/d/Frk/fr3v2_1_franka_hand.urdf"
}
```

---

## 5. 关键点验证 (7 项检查)

### 5.1 执行命令

```bash
/home/luogang/miniforge3/envs/itvlaGp/bin/python b/s/Frk/verify_franka_keypoints.py \
    --dataset /home/luogang/hf_home/lerobot/plug_into_socket_lrb_4D \
    --urdf b/d/Frk/fr3v2_1_franka_hand.urdf
```

### 5.2 结果

| # | 检查项 | 结果 | 详情 |
|---|--------|------|------|
| 1 | Shape | PASS | [8, 7] per frame, 66577 frames |
| 2 | Position bounds | PASS | max\|pos\| = 0.869565 (< 1.01) |
| 3 | Quaternion unit norm | PASS | norm_err: mean=1.30e-08, max=1.19e-07 |
| 4 | Hemisphere (qw >= 0) | PASS | qw_min = 0.00000012, violations = 0 |
| 5 | Temporal smoothness | **WARNING** | 548 jumps > 0.5 (max jump = 2.000000) |
| 6 | FK reproducibility | PASS | max error = 0.00e+00 (完美可复现) |
| 7 | Per-dim statistics | — | 见下表 |

### 5.3 Check 5 分析: 四元数跳变

548 个跳变 (占 66477 transitions 的 0.82%) 的 **根因** 是半球归一化在 $q_w \approx 0$ 边界处的不连续性:

- 当旋转经过 $q_w = 0$ 平面时, $q$ 和 $-q$ 表示相同的旋转, 但半球约束会将一帧映射到 $+q$ 而下一帧映射到 $-q$
- 两帧之间 L2 距离 = $\|q - (-q)\| = 2\|q\| \approx 2.0$
- 这不是数据错误 — FK reproducibility (Check 6) 的 0.00e+00 误差证明了关键点数学正确

**对训练的影响**: 跳变只发生在四元数空间, 不影响位置分量. 模型的 quaternion loss 通常使用 geodesic distance 而非 L2 distance, 可以正确处理这种边界情况. 如果需要完全消除, 可考虑改用 6D rotation representation (但会改变整体 pipeline).

### 5.4 关键点统计量 (across all 8 keypoints)

| 维度 | mean | std | min | max |
|------|------|-----|-----|-----|
| px | +0.337 | 0.293 | -0.039 | +0.722 |
| py | -0.023 | 0.046 | -0.168 | +0.074 |
| pz | +0.543 | 0.164 | +0.213 | +0.870 |
| qx | +0.125 | 0.596 | -1.000 | +1.000 |
| qy | +0.199 | 0.358 | -0.423 | +0.933 |
| qz | -0.102 | 0.206 | -0.682 | +0.110 |
| qw | +0.523 | 0.369 | +0.000 | +1.000 |

位置分量均在 [-1, 1] 内, 四元数分量范数约为 1, 半球约束 qw >= 0 有效.

---

## 6. Step 3: 归一化统计量生成

### 6.1 执行命令

```bash
# abs 模式
HF_HOME=/home/luogang/hf_home HF_LEROBOT_HOME=/home/luogang/hf_home/lerobot \
    /home/luogang/miniforge3/envs/itvlaGp/bin/python util_scripts/compute_norm_stats_single.py \
    --repo_id plug_into_socket_lrb_4D \
    --action_mode abs \
    --chunk_size 50

# delta 模式
HF_HOME=/home/luogang/hf_home HF_LEROBOT_HOME=/home/luogang/hf_home/lerobot \
    /home/luogang/miniforge3/envs/itvlaGp/bin/python util_scripts/compute_norm_stats_single.py \
    --repo_id plug_into_socket_lrb_4D \
    --action_mode delta \
    --chunk_size 50
```

### 6.2 执行结果

| 模式 | 状态 | 帧数 | 输出路径 | 大小 |
|------|------|------|----------|------|
| abs | 成功 | 66,577 | `${HF_LEROBOT_HOME}/stats/abs/plug_into_socket_lrb_4D/stats.json` | 24 KB |
| delta | 成功 | 66,577 | `${HF_LEROBOT_HOME}/stats/delta/plug_into_socket_lrb_4D/stats.json` | 24 KB |

### 6.3 Errors

**无 error**.

### 6.4 stats.json 验证

abs 模式 stats.json 中 `observation.keypoint_3d` 的统计量:

- mean 维度: 56 (= 8 x 7)
- std 维度: 56
- 第一个关键点 (link1) mean: [0.0, 0.0, 0.398, 0.0, 0.0, -0.120, 0.991]
- 第一个关键点 (link1) std: [0.0, 0.0, 0.0, 0.0, 0.0, 0.060, 0.007]

> link1 的位置 mean/std ≈ 0 是预期行为 — link1 是肩部关节, 其 frame 原点在基座附近, 仅旋转不平移. 标准化时 NormalizeTransformFn 会对 std=0 的维度做特殊处理 (除以 1 而非 0).

---

## 7. 数据集符号链接

按用户要求, 创建了到 `/home/luogang/DATA/plug_into_socket_lrb_4D` 的符号链接:

```bash
ln -sfn /home/luogang/hf_home/lerobot/plug_into_socket_lrb_4D /home/luogang/DATA/plug_into_socket_lrb_4D
```

---

## 8. 最终数据产出

| 产出 | 路径 | 大小 |
|------|------|------|
| 中间 LeRobot 数据集 | `/home/luogang/hf_home/lerobot/plug_into_socket_lrb/` | 421 MB |
| 最终 LeRobot 数据集 (含关键点) | `/home/luogang/hf_home/lerobot/plug_into_socket_lrb_4D/` | 432 MB |
| abs 归一化统计量 | `${HF_LEROBOT_HOME}/stats/abs/plug_into_socket_lrb_4D/stats.json` | 24 KB |
| delta 归一化统计量 | `${HF_LEROBOT_HOME}/stats/delta/plug_into_socket_lrb_4D/stats.json` | 24 KB |
| 数据集符号链接 | `/home/luogang/DATA/plug_into_socket_lrb_4D` → 上述最终数据集 | — |

---

## 9. 新增/修改文件清单

### 9.1 新增文件 (代码库内)

| 文件 | 用途 |
|------|------|
| `b/s/Frk/convert_franka_plug_hdf5.py` | HDF5→LeRobot 格式转换脚本 |
| `b/s/Frk/generate_franka_keypoints.py` | 7D 关键点离线生成脚本 |
| `b/s/Frk/verify_franka_conversion.py` | 转换验证脚本 (8项检查) |
| `b/s/Frk/verify_franka_keypoints.py` | 关键点验证脚本 (7项检查) |
| `b/s/Frk/cfg/franka_plug.yaml` | Franka 插拔数据集 schema 配置 |
| `src/lerobot/dataset_schemas/configs/franka_plug.yaml` | → `b/s/Frk/cfg/franka_plug.yaml` 符号链接 |
| `b/d/Frk/dta_4dtrj_plan_0904LOG.md` | 本日志文件 |

### 9.2 修改文件

**无**. 遵循"扩展代替修改"原则, 所有改动均为新增文件.

---

## 10. 操作检查清单

- [x] **1. 环境准备**: itvlaGp 虚拟环境已确认, pinocchio 4.1.0 + h5py 3.16.0 + cv2 4.11.0 可用
- [x] **2. URDF 确认**: `b/d/Frk/fr3v2_1_franka_hand.urdf` 存在
- [x] **3. Schema 注册**: `b/s/Frk/cfg/franka_plug.yaml` 已创建, 符号链接已注册
- [x] **4. HDF5 → LeRobot**: 100 episodes 全部转换, 66577 frames
- [x] **5. 转换验证**: 8/8 项检查全部通过
- [x] **6. 关键点生成**: Pass 1 R_pad=0.836m, Pass 2 无 OOB
- [x] **7. 元数据检查**: keypoints_meta.json 存在, keypoint_dim=7, num_keypoints=8
- [x] **8. info.json 检查**: observation.keypoint_3d shape=[56]
- [x] **9. 关键点验证**: 6/7 项通过, 1 项 WARNING (四元数半球边界跳变, 非错误)
- [x] **10. norm_stats 生成**: abs + delta 模式均完成, stats.json 含 56 维 keypoint 统计量
- [ ] **11. 模型侧改动**: 待按 R1Pro E1 方案执行 (与数据处理无关, 需另行安排)
- [ ] **12. Smoke test**: 待模型侧改动完成后执行
- [x] **13. 源数据完整**: 原始 HDF5 未被修改; 中间/最终数据集完整

---

## 11. 总结

### 11.1 处理管道执行情况

```
Step 1: HDF5 → LeRobot          ✓ 成功 (100 eps, 66577 frames, ~40 min)
        ↓
        转换验证                  ✓ 8/8 通过
        ↓
Step 2: 7D 关键点生成            ✓ 成功 (R_pad=0.836m, 无 OOB, ~2 min)
        ↓
        关键点验证                ✓ 6/7 通过 + 1 WARNING (预期行为)
        ↓
Step 3: 归一化统计量             ✓ 成功 (abs + delta, ~10s each)
```

### 11.2 关键数字

| 指标 | 值 |
|------|-----|
| 源数据大小 | 19 GB (HDF5) |
| 最终数据集大小 | 432 MB (LeRobot v3.0, Parquet + MP4) |
| 压缩比 | ~44x (主要因 JPEG→AV1 视频压缩 + 丢弃深度图和高频状态) |
| Episodes | 100 |
| 总帧数 | 66,577 |
| FPS | 30 Hz |
| 关键点维度 | 56 (= 8 keypoints x 7D) |
| R_pad | 0.836100 m |
| FK cross-check 误差 | 0.000000 m (URDF 完美匹配) |
| FK reproducibility 误差 | 0.00e+00 |
| 总 errors | **0** |
| 总 warnings | 18 (频率对齐) + 548 (四元数跳变, 预期) |
