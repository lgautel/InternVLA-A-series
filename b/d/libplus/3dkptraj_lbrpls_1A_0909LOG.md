# 3dkptraj_lbrpls_1A 数据处理操作日志

> **任务**: 将 `/home/luogang/DATA/libero_plus_rlds/` RLDS数据转换为含3D关键点的 LeRobot v3.0 格式  
> **输出**: `/home/luogang/DATA/libero_plus_lrb3/`  
> **方案**: [`3dkptraj_lbrpls_1A.md`](3dkptraj_lbrpls_1A.md)  
> **日期**: 2026-09-09  
> **环境**: `phantom` conda env (mujoco 3.11.0, tensorflow 2.16.2)

---

## 0. 初始环境检查 (09-09)

### 0.1 磁盘状态

```
Filesystem      Size  Used  Avail  Use%  Mounted on
/dev/root       969G  755G   215G   78%  /
```

**可用空间**: 215 GB，足够输出 ~30 GB 的数据。

### 0.2 已有文件状态

| 文件 | 状态 |
|------|------|
| `/tmp/panda_robosuite_full.xml` | **已存在**（上次会话已导出）|
| `GeoPredict/b/script/kpt_libero/` | 不存在，需新建 |
| `3dkptraj_lbrpls_1A_0909LOG.md` | 本文件，新建 |

### 0.3 操作计划

1. 创建 `GeoPredict/b/script/kpt_libero/` 目录和所有代码文件
2. 运行单元测试 (`test_fk_correctness.py`)
3. 运行小规模端到端测试 (1 shard)
4. 执行全量转换 (`convert_libero_rlds.py`)
5. 运行验证脚本 (`validate_libero.py`)
6. 记录所有 error 及其 fix

---

## 1. 环境准备 (09-09)

### 1.1 创建脚本目录

```bash
mkdir -p GeoPredict/b/script/kpt_libero
touch GeoPredict/b/script/kpt_libero/__init__.py
```

**目的**: 创建 Python 包目录，使各模块可以相互导入。

### 1.2 MJCF 验证 ✅

```
nq=16, nbody=24
Bodies: ['world', 'table', 'robot0_base', 'robot0_link0', 'robot0_link1',
         'robot0_link2', ..., 'robot0_link7', 'robot0_right_hand',
         'gripper0_right_gripper', 'gripper0_eef', 'gripper0_leftfinger', ...]
robot0_link1: id=4
robot0_link7: id=10
gripper0_eef: id=13
```

**结论**: MJCF 结构正确，关键 body 名称已确认。`gripper0_eef` 存在（非 `gripper0_right_eef`）。

---

## 2. 代码文件创建 (09-09)

### 2.1 创建代码文件

**操作**: 在 `GeoPredict/b/script/kpt_libero/` 创建以下文件：

| 文件 | 用途 | 状态 |
|------|------|------|
| `__init__.py` | Python 包标识 | ✅ |
| `config_libero.py` | 配置常量 | ✅ |
| `mujoco_fk.py` | MuJoCo FK 封装 | ✅ |
| `rlds_reader.py` | RLDS TFRecord 解析 | ✅ |
| `convert_libero_rlds.py` | 主转换脚本 | ✅ |
| `validate_libero.py` | 验证脚本 | ✅ |
| `test_fk_correctness.py` | 单元测试 | ✅ |

---

## 3. 单元测试 (09-09)

### 3.1 运行命令

```bash
MUJOCO_GL=egl conda run -n phantom python b/script/kpt_libero/test_fk_correctness.py
```

### 3.2 结果 ✅

```
[Test 1. FK at home position]
  robot0_base pos: [-0.56   0.     0.912]
  gripper0_eef pos: [-0.472   0.      1.7415]
  PASS

[Test 2. FK vs robosuite]
  robosuite EEF: [-0.09060067  0.00731613  1.01471167]
  standalone FK: [-0.09060067  0.00731613  1.01471167]
  Position error: 1.399e-17 m
  PASS

[Test 3. RLDS parse]
  Episode: steps=277, lang='put the white mug on the left plate...'
  qpos range: [-2.7398, 2.7455]
  PASS

[Test 4. FK on RLDS]
  Pos: shape=(277, 8, 3), range=[-0.6111, 1.5702]
  Quat: qw range=[0.0001, 1.0000], norm err max=5.96e-08
  PASS

Tests: 4 passed, 0 failed
```

**结论**: 所有4项单元测试通过，FK 引擎与 robosuite 结果 bit-exact 一致（误差 1.4e-17 m）。

---

## 4. 集成测试 (1 shard) (09-09)

### 4.1 操作

```bash
# 复制 1 个 shard 到测试目录
cp /home/luogang/DATA/libero_plus_rlds/libero_mix/1.0.0/libero_mix-train.tfrecord-00000-of-01024 \
   /tmp/libero_rlds_test/libero_mix/1.0.0/

# 运行转换（跳过视频编码，节省时间）
conda run -n phantom python b/script/kpt_libero/convert_libero_rlds.py \
    --rlds_root /tmp/libero_rlds_test/libero_mix/1.0.0 \
    --output /tmp/libero_lrb3_test \
    --normalization auto_offset --no_video
```

### 4.2 转换结果

```
Phase 1 complete: 14 episodes, 2115 frames
Global pos min: [-0.6134, -0.2991, 0.9088]
Global pos max: [0.2537, 0.3687, 1.5704]
Auto-offset: [-0.9798, -0.7652, 0.7396]
Chunk 1/1: wrote 2115 rows
DONE — Episodes: 14, Tasks: 13
```

### 4.3 验证结果 ✅ (8/8 PASSED)

```
[OK] codebase_version == v3.0
[OK] robot_type == franka
[OK] observation.keypoint_3d present, dim=24
[OK] observation.keypoint_quat present, dim=32
[OK] 'reward' not present
[OK] 'discount' not present
[OK] Frame count: 2115 (matches info.json)
[OK] 14 episodes: all frame indices contiguous
[INFO] Position range [0.1692, 1.2335] (auto-offset)
[OK] Quat norm error: max=5.96e-08
[OK] Hemisphere: 0/16920 violations

Result: 8 PASSED, 0 FAILED
```

**结论**: 集成测试完全通过，开始全量转换。

---

## 5. 全量转换 (09-09)

### 5.1 运行命令

```bash
nohup conda run -n phantom python b/script/kpt_libero/convert_libero_rlds.py \
    --rlds_root /home/luogang/DATA/libero_plus_rlds/libero_mix/1.0.0 \
    --output /home/luogang/DATA/libero_plus_lrb3 \
    --xml_path /tmp/panda_robosuite_full.xml \
    --normalization auto_offset \
    > /tmp/convert_libero_full.log 2>&1 &
```

**开始时间**: 2026-09-09 05:10:20 UTC

**磁盘状态**:
- 输入: 78 GB (只读, RLDS)
- 可用: 215 GB
- 缓存目录: `/tmp/libero_plus_cache/`
- 输出目录: `/home/luogang/DATA/libero_plus_lrb3/`

### 5.2 Phase 1 完成情况

**Phase 1 完成**: 约 05:13 UTC（约3分钟，78 GB TFRecord + FK计算）

```
Global pos min: [-0.771, -0.440, 0.908]
Global pos max: [ 0.276,  0.435, 1.584]
Auto-offset computed from center
Total: 14,347 episodes, ~2.24M frames
Cache size: 78 GB (在 /tmp/libero_plus_cache/)
```

> **性能**: TFRecord解析 + MuJoCo FK，速度约 ~80 episodes/s，3分钟完成所有 14,347 episodes。

### 5.3 Phase 2 进行中（视频编码）

视频编码是最慢的阶段。每个chunk编码约2-4分钟（libx264, 1000 episodes × 2 cameras）。

| 时间 | data chunks | img videos | wrist videos | 总大小 |
|------|-------------|------------|--------------|--------|
| 05:16:45 | 2 | 2 | 1 | 493M |
| 05:18:45 | 2 | 2 | 2 | 729M |
| 05:20:45 | 3 | 3 | 3 | 919M |
| 05:22:45 | 4 | 4 | 3 | 1.2G |
| 05:24:45 | 4 | 4 | 4 | 1.5G |

**视频编码进度**:

| 时间 | data chunks | img videos | wrist videos | 总大小 |
|------|-------------|------------|--------------|--------|
| 05:30:02 | 6 | 6 | 6 | 2.0G |
| 05:35:02 | 8 | 8 | 7 | 2.7G |
| 05:40:02 | 9 | 9 | 9 | 3.2G |
| 05:45:02 | 11 | 11 | 11 | 3.8G |
| 05:50:02 | 13 | 13 | 12 | 4.4G |
| 05:55:02 | 14 | 14 | 14 | 4.9G |
| **06:00:02** | **15** | **15** | **15** | **5.2G** |

**Phase 2+3 完成**: 06:00 UTC（约50分钟视频编码）

### 5.4 最终输出结构

```
/home/luogang/DATA/libero_plus_lrb3/   (5.2 GB total)
├── data/   (829 MB, 15 chunks)
│   ├── chunk-000/file-000.parquet
│   ├── ...
│   └── chunk-014/file-000.parquet
├── meta/
│   ├── info.json
│   ├── tasks.parquet        (40 tasks)
│   ├── stats.json
│   ├── keypoints_meta.json
│   └── episodes/
│       ├── chunk-000/file-000.parquet
│       └── ... (15 chunks)
├── videos/   (4.4 GB)
│   ├── observation.images.image/
│   │   └── chunk-000/ ~ chunk-014/file-000.mp4  (15 videos)
│   └── observation.images.wrist_image/
│       └── chunk-000/ ~ chunk-014/file-000.mp4  (15 videos)
└── norm_stat.json
```

**info.json 关键参数**:
- `codebase_version`: v3.0
- `total_episodes`: 14,347
- `total_frames`: 2,238,036
- `total_tasks`: 40
- `total_chunks`: 15
- `fps`: 20
- features: state[8], joint_state[7], action[7], keypoint_3d[24], keypoint_quat[32], images×2, lang

**整体耗时**: 约50分钟（Phase 1 ~3分钟，Phase 2+3 ~47分钟）

---

## 6. 全量验证 (09-09 06:00 UTC) ✅

### 6.1 运行命令

```bash
conda run -n phantom python b/script/kpt_libero/validate_libero.py \
    --dataset_root /home/luogang/DATA/libero_plus_lrb3
```

### 6.2 验证结果 (8/8 PASSED)

```
[Check 1. info.json schema]
  [OK] codebase_version == v3.0
  [OK] robot_type == franka, fps == 20
  [OK] observation.keypoint_3d in features (dim=24)
  [OK] observation.keypoint_quat in features (dim=32)
  [OK] no reward feature, no discount feature

[Check 2. Parquet columns & dims]
  [OK] All 11 required columns present
  [OK] 'reward' not present, 'discount' not present
  [OK] state=8, joint_state=7, action=7, keypoint_3d=24, keypoint_quat=32

[Check 3. Frame count consistency]
  [OK] Frame count: 2,238,036 (matches info.json)

[Check 4. Episode continuity]
  [OK] 14,347 episodes: all frame indices contiguous

[Check 5. Keypoint positions]
  [INFO] Position range [0.1623, 1.3233] (auto-offset)
  → 关键点位于体素空间合理范围内

[Check 6. Quaternion unit norm]
  [OK] Quat norm error: max=1.19e-07 (远低于阈值 0.01)

[Check 7. Quaternion hemisphere]
  [OK] Hemisphere: 0/17,904,288 violations (所有 qw ≥ 0)

[Check 8. Video files]
  [OK] All 30 video files present (15 chunks × 2 cameras)

Result: 8 PASSED, 0 FAILED
```

### 6.3 关键统计

| 指标 | 值 | 说明 |
|------|-----|------|
| Episodes | 14,347 | 与 RLDS 原始数据完全一致 |
| Frames | 2,238,036 | 与 RLDS 一致 |
| Tasks | 40 | 独立语言指令数 |
| Keypoint pos range | [0.162, 1.323] | auto-offset 后位于体素空间内 |
| Quat norm error | max=1.19e-07 | MuJoCo 高精度，远低于 0.01 阈值 |
| qw<0 violations | 0 / 17,904,288 | 半球约束100%满足 |
| Video files | 30/30 | 15 chunk × 2 cameras |
| 总数据大小 | 5.2 GB | parquet 829M + video 4.4G |

---

## 7. 整体总结

### 7.1 操作时间线

| 时间 (UTC) | 操作 | 结果 |
|-----------|------|------|
| 05:09 | 代码文件创建 | ✅ |
| 05:09 | 单元测试 (4项) | ✅ 4/4 通过 |
| 05:09 | 集成测试 (1 shard) | ✅ 8/8 验证通过 |
| 05:10 | 启动全量转换 | ✅ |
| 05:13 | Phase 1 完成 (FK 14347 episodes) | ✅ |
| 06:00 | Phase 2+3 完成 (parquet + 视频 + 元数据) | ✅ |
| 06:03 | 全量验证 (8项) | ✅ 8/8 通过 |

### 7.2 无 Error 记录

本次数据处理**没有发生任何 Error**。代码在首次运行时即成功完成，全部验证通过。这得益于：
1. 充分的预分析和方案设计（`3dkptraj_lbrpls_1.md` + `3dkptraj_lbrpls_1A.md`）
2. 单元测试先行（4个测试全部通过后才启动全量转换）
3. 集成测试（1 shard 验证通过后才启动全量转换）

### 7.3 输出产物

| 产物 | 路径 | 说明 |
|------|------|------|
| 数据集 | `/home/luogang/DATA/libero_plus_lrb3/` | LeRobot v3.0 格式 |
| 代码 | `GeoPredict/b/script/kpt_libero/` | 7个Python文件 |
| 方案文档 | `GeoPredict/b/d/3dkptraj_lbrpls_1A.md` | 实施方案 |
| 操作日志 | `GeoPredict/b/d/3dkptraj_lbrpls_1A_0909LOG.md` | 本文件 |

---

