# 3D/4D Keypoint Generation Execution Log — 2026-09-29

> **方案文档**: `b/d/libplus2/gol/3d4d_gen2.markdown`
> **代码目录**: `b/s/libplus2/gol/`
> **数据源**: `/B/Dta/LIBERO/rlds/libero_goal/1.0.0` (RLDS TFRecord, 256 shards, ~18 GB)
> **输出目标**: `/home/a26113/b/Dta/libero_plus_goal_lrb3_4D/`
> **Python 环境**: `/B/VENV/itnvla15rbt20/` (Python 3.11.9, mujoco 3.13.0, lerobot 1.0.0)

---

## 环境确认

| 项目 | 值 |
|------|-----|
| Python | 3.11.9 |
| mujoco | 3.13.0 |
| lerobot | 1.0.0 |
| RLDS reader | `b/d/libplus/ds/asset/rlds_reader.py` (存在) |
| RLDS 数据 | `/B/Dta/LIBERO/rlds/libero_goal/1.0.0/` (256 shards) |
| 输出目录 | `/home/a26113/b/Dta/libero_plus_goal_lrb3_4D/` |
| 工作目录 | `/B/SRC/itvlaGpLibPlus` |

---

## R0: 离线验收 (accept_goal_4d.sh, S1-S3)

**命令**:
```bash
PY=/B/VENV/itnvla15rbt20/bin/python bash b/s/libplus2/gol/accept_goal_4d.sh
```

**结果**: ✅ 全部通过

| 阶段 | 内容 | 结果 |
|------|------|------|
| S1 | MJCF 派生文件一致性 (`panda_goal_table.xml`) | OK |
| S1b | eval wrapper 恰好两处替换 + 拒绝 `ROTATE_IMAGES=true` | OK |
| S2 | gen1 tests (17 项: FK, history packing, generator, eval clock) | 17/17 OK, 10.9s |
| S3 | gen2 tests (37 项: A-G 组) | 37/37 OK, 344.5s |

**总计**: 54/54 tests passed, 约 6 分钟

**详细结果**:
- A 组 (FK 等价, 4 项): GoalTableFK vs StandaloneFK 数值一致; Goal XML vs Lift XML 仅差底座平移 (0.10m in x)
- B 组 (常数一致, 5 项): H=200 与 launch/client/KeypointHistory 一致; panda.yaml 双相机; fps=20; 朝向匹配全局契约
- C 组 (RLDS 朝向, 3 项): agentview/wrist 均为 flipH (即 rot180 of raw); 负对照通过
- D 组 (生成器护栏, 6 项): raw = rot180(RLDS); validate_episode 正确接受/拒绝; --confirm-full-run 拒绝测试开关
- E 组 (评估客户端, 12 项): 时序、历史、活体 FK 护栏、MD5/朝向/覆盖冲突检测全通过
- F 组 (导入钩子, 3 项): 父进程无 mujoco; 子进程拿到子类; 懒加载验证
- G 组 (端到端, 4 项): raw 数据集全检查通过; 契约验证; 负对照拦截

---

## R1: 全量数据生成

**命令**:
```bash
/B/VENV/itnvla15rbt20/bin/python b/s/libplus2/gol/generate_goal_4d.py \
  --dest /home/a26113/b/Dta/libero_plus_goal_lrb3_4D \
  --confirm-full-run --force
```

### Pass 1 结果

✅ **全部 4,243 episodes 通过验证**

| 指标 | 值 | 阈值/预期 |
|------|-----|-----------|
| R_pad | **1.8212723273** | 计算得出（vs Lift default 1.8212722540, 差 7.3e-8） |
| episodes | 4,243 | 4,243 ✅ |
| frames | 512,604 | 512,604 ✅ |
| worst FK-vs-state EEF err | **7.22e-4 m** (0.722 mm) | < 2.0e-3 m (2 mm) ✅ |
| EpisodeValidationError | 0 | 0 ✅ |

**分析**: R_pad 值与 Lift 场景的默认值（1.8212722540）极为接近，差异仅 7.3e-8。这是因为 Goal 场景底座 (-0.66, 0, 0.912) 与 Lift 场景底座 (-0.56, 0, 0.912) 仅在 x 方向差 0.10m，但 R_pad 由全局包围盒的最大绝对值决定，0.10m 的平移对极端位置的影响很小。FK vs EEF 的最大误差仅 0.722mm，远低于 2mm 阈值，说明底座设置正确且 FK 计算准确。

### Pass 2 结果

✅ **全部 4,243 episodes 写入完成**

**初次运行 (FUSE 目标路径)**:
- 目标: `/home/a26113/b/Dta/libero_plus_goal_lrb3_4D` (FUSE 云存储挂载)
- 速率: ~47 秒/episode, 预计 55 小时 ❌
- **根因**: FUSE 文件系统 (`autel-ai-crater-prod-ew1 fuse`) 对小文件有严重的网络延迟。每个 episode 的视频编码需要写/读/删 ~240 个 PNG 临时文件, 每个文件操作都触发网络往返。
- **操作**: 立即 kill 该进程

**修复运行 (本地 ext4 SSD)**:
- 目标: `/tmp/libero_plus_goal_lrb3_4D` (本地 ext4 SSD, `/dev/md127 12T`)
- 命令:
```bash
/B/VENV/itnvla15rbt20/bin/python b/s/libplus2/gol/generate_goal_4d.py \
  --dest /tmp/libero_plus_goal_lrb3_4D --confirm-full-run --force
```
- 速率: ~3.6 秒/episode (**13x 加速**)
- 结果: 全部 4,243 episodes 成功写入
- 最终输出: `[INFO] wrote /tmp/libero_plus_goal_lrb3_4D`

**数据集结构验证**:

| 项目 | 值 |
|------|-----|
| 总大小 | 3.5 GB |
| 格式 | LeRobot v3.0 |
| robot_type | panda |
| total_episodes | 4,243 |
| total_frames | 512,604 |
| total_tasks | 10 |
| fps | 20 |
| 视频编码 | AV1 (SVT-AV1, CRF 30) |
| 数据文件 | data/chunk-000/ (2 parquet files) |
| 视频文件 | 18 mp4 files (9 per camera, 2 cameras) |
| 元数据 | info.json, stats.json, tasks.parquet, keypoints_meta.json, goal_train_eval_contract.json, goal_episodes.jsonl |
| PNG 残留 | 0 (全部清理) |

**复制到最终目标**:
```bash
cp -r /tmp/libero_plus_goal_lrb3_4D /home/a26113/b/Dta/libero_plus_goal_lrb3_4D
```
- md5sum 全文件比对: ✅ 完全一致 (diff exit 0)

---

## R2: 全量验收 (accept_goal_4d.sh, S1-S6, FULL_CHECK=1)

**命令**:
```bash
DATASET=/home/a26113/b/Dta/libero_plus_goal_lrb3_4D FULL_CHECK=1 PY=/B/VENV/itnvla15rbt20/bin/python \
  bash b/s/libplus2/gol/accept_goal_4d.sh
```

**结果**: ✅ 全部通过

### S1-S3 (离线测试, 同 R0)

| 阶段 | 结果 |
|------|------|
| S1 | MJCF 派生文件一致性 OK |
| S1b | eval wrapper 恰好两处替换 + 拒绝 ROTATE_IMAGES=true OK |
| S2 | gen1 tests 17/17 OK (14.1s) |
| S3 | gen2 tests 37/37 OK (353.7s) |

### S4: Dataset Verifier V01-V18 (--full)

| 验证项 | 结果 | 关键数据 |
|--------|------|----------|
| V01 info | ✅ | codebase=v3.0 fps=20 robot_type=panda |
| V02 features | ✅ | state[8] keypoint_3d[56] action[7] image image2 |
| V03 video_dtype | ✅ | both cameras stored as video |
| V04 contract | ✅ | contract file matches code constants |
| V05 counts | ✅ | frames=512604 sidecar=512604 episodes=4243 |
| V06 finite | ✅ | no NaN/Inf in state, joints, action, keypoints |
| V07 quat | ✅ | max\|norm-1\|=3.51e-07, min qw=3.57e-07 |
| V08 r_pad_range | ✅ | max\|pos/R_pad\|=0.8696 (bound 0.8697); r_pad=1.821272 |
| V09 fk_recompute | ✅ | stored keypoints == FK(joint_position, fingers): max diff 0.00e+00 |
| V10 fk_vs_state | ✅ | \|FK eef - state[0:3]\| max 0.64 mm (tol 2.0 mm) |
| V11 eval_fk | ✅ | StandaloneFK(panda_goal_table.xml) vs stored: max diff 3.58e-07 |
| V12 gripper | ✅ | corr(action[6], d finger_l)=-0.329, range=[-1.00,1.00] |
| V13 stats | ✅ | meta/stats.json has state[8], action[7] with std>1e-6 |
| V14 trajectories | ✅ | 428 unique (joint_state, action) trajectories over 4243 episodes |
| V15 pixels_agentview | ✅ | mse vs RLDS as-is=2985.6, vs rot180=8.0 (contract=raw) |
| V15 pixels_wrist | ✅ | mse vs RLDS as-is=3931.4, vs rot180=10.5 (contract=raw) |
| V16 geometry_agentview | ✅ | best=flipV_RAW corr_col=+0.987 corr_row=+0.944 n=308 |
| V17 geometry_wrist | ✅ | best=flipV_RAW corr_col=+0.572 corr_row=+0.838 n=151 |
| V18 training_window | ✅ | his_len=min(t,H), his_kpts=past frames, zero padded; kpt_t/kpt_future correct |

**19/19 passed**

### S5: Real Training Data Path

6 个采样点 (indices: 0, 102520, 205041, 307561, 410082, 512603) 全部通过:
- his_kpts shape (200, 8, 7) ✅
- kpt_future shape (50, 8, 7) ✅
- action chunk (50, 32) with dims >=7 zero padding ✅
- two camera images tokenised image_grid_thw=(2, 3) ✅
- pixel_values present ✅
- state finite (32,) ✅
- his_len 正确 (0 for t=0, 各 episode 中的 t 步各有正确值) ✅
- zeros after his_len ✅, filled before his_len ✅

### S6: Live-Orientation Judge Self-Test

| 相机 | raw MSE | rot180 MSE | ratio | 投票 | 结果 |
|------|---------|------------|-------|------|------|
| agentview | 114.4 | 6852.7 | 59.9 | 63:0 prefer_raw | ✅ matches_raw |
| wrist | 849.4 | 8649.1 | 10.2 | 63:0 prefer_raw | ✅ matches_raw |

负对照 (pseudo-live rotated 180°): 正确检测为 FAIL ✅

**SELFTEST OK**

---

## 总结

| 里程碑 | 状态 | 耗时 |
|--------|------|------|
| R0: 离线验收 S1-S3 | ✅ 54/54 passed | ~6 min |
| R1 Pass 1: 全量扫描 + R_pad | ✅ 4243 eps, R_pad=1.8212723273 | ~25 min |
| R1 Pass 2: LeRobot v3 写入 | ✅ 4243 eps, 512604 frames | ~4.5 hrs (ext4 SSD) |
| 数据复制 + md5 验证 | ✅ 3.5 GB copied | ~2 min |
| R2: 全量验收 S1-S6 (FULL_CHECK=1) | ✅ 54+19+S5+S6 all passed | ~15 min |

**最终数据集路径**: `/home/a26113/b/Dta/libero_plus_goal_lrb3_4D/`

### 修复记录

| 问题 | 根因 | 修复 | 影响 |
|------|------|------|------|
| Pass 2 极慢 (47s/ep, ETA 55h) | FUSE 云存储对 PNG 临时文件的网络延迟 | 输出重定向到本地 ext4 SSD `/tmp/`, 完成后 cp 到最终路径 | 13x 加速, 4.5h 完成 |

### 未修改的文件

按照约束, 未修改 `evaluation/`, `src/`, `launch/` 下的任何文件。所有生成代码在 `b/s/libplus2/gol/` 目录。

