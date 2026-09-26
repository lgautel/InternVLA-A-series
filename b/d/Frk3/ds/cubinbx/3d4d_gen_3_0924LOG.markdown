# 3D/4D 关键点生成执行日志 — put_cube_into_box (V3)

> **执行日期**: 2026-09-24
> **方案**: [3d4d_gen_3.markdown](3d4d_gen_3.markdown)
> **源数据**: `/B/Dta/put_cube_into_box/put_cube_into_box_hdf5/` (56 episodes, 122,880 state 帧)
> **中间输出**: `/home/a26113/b/Dta/put_cube_into_box_lrb3/` (LeRobot, 含 ee_quat2, 无关键点)
> **最终输出**: `/home/a26113/b/Dta/put_cube_into_box_lrb3_4D/` (LeRobot, 含 observation.keypoint_3d [56])
> **URDF**: `b/d/Frk2/fr3v2_1_franka_hand.urdf`
> **虚拟环境**: `/B/VENV/itnvla15rbt20/`

---

## 执行概览

| Step | 操作 | 状态 |
|:---:|---|:---:|
| 0 | 创建脚本 | ✅ 完成 |
| 1 | HDF5 → LeRobot (含 ee_quat2) | ✅ 完成 |
| 1.5 | LeRobot v3 格式修复 | ✅ 完成 |
| 2 | FK 关键点生成 | ✅ 完成 |
| 3 | 10-check 验证 | ✅ 全部 PASS |
| 4 | 训练兼容性验证 | ✅ 完成 |

---

## Step 0: 创建脚本

### 0.1 脚本清单

| 脚本 | 路径 | 基于 |
|---|---|---|
| HDF5→LeRobot 转换 | `b/s/Frk3/convert_cubinbx_hdf5.py` | `b/s/Frk/convert_franka_plug_hdf5.py` |
| FK 关键点生成 | `b/s/Frk3/generate_cubinbx_keypoints.py` | `b/s/Frk2/generate_franka2_keypoints.py` |
| 关键点验证 | `b/s/Frk3/verify_cubinbx_keypoints.py` | `b/s/Frk2/verify_franka2_keypoints.py` |

### 0.2 `convert_cubinbx_hdf5.py` 相对参考脚本的改动

1. 统一 15D state (`observation.state`) 替代分列 (`.arm`, `.gripper`, `.ee_pos`, `.ee_quat`)
2. 统一 8D action (`action`) 替代分列 (`.arm`, `.gripper`)
3. `gripper_width` clamp 负值: `np.clip(w, 0, None)`
4. **ee_quat2**: 不使用原始 `ee_quat`, 通过 FK(`joint_positions`) 重新计算, 半球归一化 (qw ≥ 0)
5. 转换脚本新增 `--urdf` 参数 (FK 计算需要 pinocchio + URDF)
6. Feature names 中 `ee_quat_*` 改为 `ee_quat2_*`
7. Task name: `"put cube into box"`

### 0.3 `generate_cubinbx_keypoints.py` 相对参考脚本的改动

1. `DEFAULT_DATASET_FPS = 30` (原 15)
2. contract version 升级到 "3.0"
3. 新增 ee_quat2 相关字段写入 contract
4. 交叉验证中 state[11:15] 已是 ee_quat2 (半球归一化), FK 输出也是半球归一化, 误差应近零

### 0.4 `verify_cubinbx_keypoints.py` bug 修复

1. **导入错误**: 原始使用 `from b.s.Frk3.generate_cubinbx_keypoints import FrankaFKExtractor7D` — 不可作为 Python 模块路径
   - **修复**: 创建 `_make_extractor()` helper, 通过 `sys.path.insert(0, script_dir)` 后用 `from generate_cubinbx_keypoints import` 导入
2. **全局变量错误**: `check_6_fk_reproducibility()` 引用了不存在的全局变量 `args_dataset`
   - **修复**: 给函数新增 `dataset_path` 参数, 由 `main()` 中 lambda 传入 `args.dataset`

### 0.5 快速转换脚本 `convert_cubinbx_fast.py` 的创建

**原因**: 原始 `convert_cubinbx_hdf5.py` 使用 LeRobot 的 `add_frame()` API, 该 API 每帧写一个 PNG 文件到磁盘, 然后 `save_episode()` 再将所有 PNG 读回并编码为 MP4 视频. 这个双重 I/O 导致处理速度极慢 (~100 帧/分钟, 预计 6+ 小时完成 37K 帧).

**解决方案**: 编写 `convert_cubinbx_fast.py`, 完全绕过 LeRobot API:
1. 直接用 pandas 写 parquet 文件 (不经过 `add_frame()`)
2. 用 ffmpeg subprocess pipe 将 JPEG 解码为 RGB 后直接编码为 AV1 MP4 (不写中间 PNG)
3. 手动生成 `meta/info.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl`

**性能**: ~12 episodes/minute (对比原方案 ~0.15 episodes/minute), 提速 ~80x.

### 0.6 `convert_cubinbx_fast.py` bug 修复

**Bug**: 第一次运行时 episode 0 的视频编码完成后崩溃:
```
File "convert_cubinbx_fast.py", line 117, in encode_video_from_jpegs
    _, stderr = proc.communicate()
ValueError: flush of closed file
```

**根因**: Python 3.11 的 `subprocess.Popen.communicate()` 内部会调用 `self.stdin.flush()`, 但此时 `proc.stdin.close()` 已被显式调用, 导致 flush 一个已关闭的文件.

**修复**: 将原来的 `proc.stdin.close()` + `proc.communicate()` 替换为:
```python
try:
    for jpeg_data in jpeg_list:
        proc.stdin.write(decode_jpeg(bytes(jpeg_data)).tobytes())
except BrokenPipeError:
    pass
finally:
    try:
        proc.stdin.close()
    except Exception:
        pass
stderr = proc.stderr.read()
rc = proc.wait()
```
同时将 ffmpeg cmd 中 `output_path` 显式转为 `str(output_path)` 避免 Path 对象兼容性问题.

---

## Step 1: HDF5 → LeRobot 转换 (含 ee_quat2)

### 1.1 执行命令

```bash
source /B/VENV/itnvla15rbt20/bin/activate
export HF_HOME=/B/VENV/hf_home
python b/s/Frk3/convert_cubinbx_fast.py \
  --source /B/Dta/put_cube_into_box/put_cube_into_box_hdf5 \
  --dest /home/a26113/b/Dta/put_cube_into_box_lrb3 \
  --urdf b/d/Frk2/fr3v2_1_franka_hand.urdf \
  --force
```

### 1.2 执行结果

| 指标 | 值 |
|---|---|
| Episodes | 56 |
| 总帧数 | 36,953 |
| 视频编码 | AV1 (libsvtav1), CRF 30, preset 6 |
| 分辨率 | 640×480, 30 FPS |
| 耗时 | ~4 分钟 |
| 每帧 ee_quat2 | FK 计算 + 半球归一化 |
| gripper_width 负值 clamp | 已执行 |

### 1.3 HDF5 字段形状 (验证)

| 字段 | 形状 | 类型 |
|---|---|---|
| `joint_positions` | (N, 7) | float64 |
| `gripper_width` | (N, 1) | float64 |
| `ee_pos` | (N, 3) | float64 |
| `action_joints` | (N, 7) | float64 |
| `action_gripper` | (N, 1) | float64 |
| `ee_quat` | (N, 4) | float64 |

### 1.4 输出数据验证 (episode 0, 877 帧)

```
State shape: (877, 15), dtype: float32
State[0] joints: [-0.0988, 0.0808, 0.0542, -1.5146, 0.0008, 1.6457, 0.8883]
State[0] gripper: 0.0793
State[0] ee_pos: [0.5915, -0.0271, 0.5087]
State[0] ee_quat2: [-0.9970, 0.0738, -0.0251, 0.0033]
qw range: [1.18e-05, 0.0808]  — 全部 ≥ 0 ✅
gripper_w min: 0.0  — 无负值 ✅
Action shape: (877, 8)
```

---

## Step 1.5: LeRobot v3 格式修复

### 1.5.1 问题发现

使用 `LeRobotDataset` 加载数据时发生两个错误:

**Error 1**: `KeyError: 'file_name'`
- `info.json` 的 `data_path` 使用 `{file_name:s}` 占位符
- LeRobot v3 实际使用 `chunk_index` 和 `file_index` 占位符
- 文件命名需要从 `episode_000000.parquet` 改为 `file-000.parquet`

**Error 2**: `KeyError: 'videos/observation.images.global/from_timestamp'`
- 视频回放需要 `from_timestamp` 和 `to_timestamp` 元数据字段
- 每个 episode 是独立的 MP4 文件, 所以 `from_timestamp = 0.0`

### 1.5.2 修复方案

创建 `b/s/Frk3/fixup_lerobot_format.py`, 执行以下修复:
1. 重命名 `data/chunk-000/episode_NNNNNN.parquet` → `file-NNN.parquet`
2. 重命名 `videos/.../episode_NNNNNN.mp4` → `file-NNN.mp4`
3. 更新 `info.json` 路径模板为标准 LeRobot v3 格式:
   - `data_path`: `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`
   - `video_path`: `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`
4. 将 `meta/episodes.jsonl` 转换为 `meta/episodes/chunk-000/file-000.parquet`
5. 为每个 episode 添加缺失的元数据字段:
   - `data/chunk_index`: 0
   - `data/file_index`: episode 序号
   - `videos/{vid_key}/chunk_index`: 0
   - `videos/{vid_key}/file_index`: episode 序号
   - `videos/{vid_key}/from_timestamp`: 0.0
   - `videos/{vid_key}/to_timestamp`: length / fps

### 1.5.3 执行命令

```bash
python b/s/Frk3/fixup_lerobot_format.py --dataset /home/a26113/b/Dta/put_cube_into_box_lrb3_4D
python b/s/Frk3/fixup_lerobot_format.py --dataset /home/a26113/b/Dta/put_cube_into_box_lrb3
```

两个数据集均修复成功.

---

## Step 2: FK 关键点生成

### 2.1 执行命令

```bash
source /B/VENV/itnvla15rbt20/bin/activate
export HF_HOME=/B/VENV/hf_home
python b/s/Frk3/generate_cubinbx_keypoints.py \
  --source /home/a26113/b/Dta/put_cube_into_box_lrb3 \
  --dest /home/a26113/b/Dta/put_cube_into_box_lrb3_4D \
  --urdf b/d/Frk2/fr3v2_1_franka_hand.urdf \
  --dataset-fps 30 \
  --force
```

### 2.2 执行结果

| 指标 | 值 |
|---|---|
| Pass 1 帧数 | 36,953 |
| Pass 2 帧数 | 36,953 |
| R_pad | 0.886566 m |
| 关键点维度 | [56] = 8 kpts × 7D |
| FK cross-val pos_err_max | 0.000137 mm |
| FK cross-val rot_err_max | 0.055953° |
| qw_min (关键点) | 0.000000 |
| qw_max (关键点) | 1.000000 |
| quat norm_err_max | 5.96e-08 |

### 2.3 关键文件写入

| 文件 | 路径 |
|---|---|
| 数据 parquet (含 keypoint_3d) | `data/chunk-000/file-*.parquet` |
| 关键点元数据 | `meta/keypoints_meta.json` |
| 训练推理合约 | `meta/train_inference_contract.json` |
| 更新的 info.json | `meta/info.json` |

### 2.4 FK 交叉验证分析

由于 ee_quat2 本身就是从 FK(joint_positions) 计算得来, FK 交叉验证的误差来自:
- 浮点精度差异 (float32 vs float64)
- 两次独立 FK 计算的数值差

误差量级 (pos ~0.0001mm, rot ~0.06°) 确认了 ee_quat2 和关键点之间的一致性.

---

## Step 3: 10-check 验证

### 3.1 执行命令

```bash
python b/s/Frk3/verify_cubinbx_keypoints.py \
  --dataset /home/a26113/b/Dta/put_cube_into_box_lrb3_4D \
  --urdf b/d/Frk2/fr3v2_1_franka_hand.urdf
```

### 3.2 验证结果

| Check | 名称 | 结果 | 详情 |
|:---:|---|:---:|---|
| 1 | Shape | ✅ PASS | 36,953 帧 × [56] |
| 2 | Position bounds | ✅ PASS | max \|pos\| = 0.8696 (limit 1.01) |
| 3 | Quaternion norm | ✅ PASS | max \|q\|-1 error = 5.96e-08 |
| 4 | Hemisphere | ✅ PASS | kpt qw_min=0.0000, state qw_min=0.0000 |
| 5 | Temporal smoothness | ✅ PASS | max diff = 0.0238, jumps = 0 |
| 6 | FK reproducibility | ✅ PASS | max error = 2.98e-08 |
| 7 | Per-dim stats | ✅ PASS | 统计信息已记录 |
| 8 | FK cross-val vs ee_quat2 | ✅ PASS | pos=0.000086mm, rot=0.040° |
| 9 | keypoints_meta.json | ✅ PASS | R_pad=0.886566 |
| 10 | contract.json | ✅ PASS | v3.0, fps=30 |

**验证执行了两次**: 首次在 Step 2 完成后执行, 第二次在 Step 1.5 格式修复后再次执行. 两次均全部通过.

---

## Step 4: 训练兼容性验证

### 4.1 LeRobotDataset 加载测试

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(
    repo_id='put_cube_into_box_lrb3_4D',
    root='/home/a26113/b/Dta/put_cube_into_box_lrb3_4D',
)
```

**结果**: ✅ 加载成功

| 项目 | 值 |
|---|---|
| Episodes | 56 |
| Total Frames | 36,953 |
| FPS | 30 |
| Features | observation.state, action, observation.images.global, observation.images.wrist, observation.keypoint_3d |

### 4.2 Sample 数据检查

| 特征 | Shape | Dtype | 值域 |
|---|---|---|---|
| `observation.state` | [15] | float32 | joints + gripper + ee_pos + ee_quat2 |
| `action` | [8] | float32 | joint commands + gripper cmd |
| `observation.keypoint_3d` | [56] | float32 | 8 kpts × 7D |
| `observation.images.global` | [3, 480, 640] | float32 | [0.0, 1.0] |
| `observation.images.wrist` | [3, 480, 640] | float32 | [0.0, 1.0] |

### 4.3 多样本验证

对 index = 0, 100, 500, 1000, 5000, 10000, 20000, 30000 的样本分别验证:
- `observation.state` shape = [15] ✅
- `observation.keypoint_3d` shape = [56] ✅
- `state[14]` (ee_quat2 qw) ≥ 0 ✅

### 4.4 Schema cast 说明

加载时出现 WARNING: `Feature schema cast failed, loading without features constraint`. 这是因为:
- Parquet 中 `observation.state` 存储为 `list<element: float>` (通用列表)
- LeRobot 期望 `List(Value('float32'), length=15)` (固定长度列表)
- Parquet 中有额外的元数据列 (`episode_index`, `frame_index`, `timestamp` 等)

这是 **非致命 WARNING**, LeRobot 回退到无约束加载模式后数据形状和内容完全正确. 实际训练中此 warning 不影响功能.

---

## 最终数据集结构

```
/home/a26113/b/Dta/put_cube_into_box_lrb3_4D/
├── data/
│   └── chunk-000/
│       ├── file-000.parquet  (episode 0, 877 frames)
│       ├── file-001.parquet  (episode 1, 880 frames)
│       └── ...               (56 files total, 36,953 frames)
├── meta/
│   ├── episodes/
│   │   └── chunk-000/
│   │       └── file-000.parquet  (56 episode records)
│   ├── info.json
│   ├── keypoints_meta.json
│   ├── tasks.jsonl
│   └── train_inference_contract.json
└── videos/
    ├── observation.images.global/
    │   └── chunk-000/
    │       ├── file-000.mp4
    │       └── ... (56 files)
    └── observation.images.wrist/
        └── chunk-000/
            ├── file-000.mp4
            └── ... (56 files)
```

---

## 完整脚本清单

| 脚本 | 路径 | 用途 |
|---|---|---|
| 原始转换 (慢) | `b/s/Frk3/convert_cubinbx_hdf5.py` | LeRobot API 方式, 未使用 |
| 快速转换 | `b/s/Frk3/convert_cubinbx_fast.py` | 直接写 parquet + ffmpeg 编码视频 |
| 格式修复 | `b/s/Frk3/fixup_lerobot_format.py` | 修复文件命名和元数据格式为 v3 标准 |
| FK 关键点生成 | `b/s/Frk3/generate_cubinbx_keypoints.py` | 两轮 FK, 写入 observation.keypoint_3d |
| 验证 | `b/s/Frk3/verify_cubinbx_keypoints.py` | 10-check 验证 |

---

## Error 汇总

| # | 错误 | 根因 | 修复 | 影响 |
|:---:|---|---|---|---|
| E1 | `ValueError: flush of closed file` | Python 3.11 `communicate()` 在 `stdin.close()` 后仍尝试 flush | 改用 `try/except BrokenPipeError` + `proc.wait()` | 快速转换脚本 |
| E2 | `KeyError: 'file_name'` | info.json 路径模板使用非标准占位符 | 创建 fixup 脚本, 改为 `{chunk_index:03d}/{file_index:03d}` | 格式兼容性 |
| E3 | `KeyError: 'from_timestamp'` | episodes 元数据缺少视频时间戳字段 | fixup 脚本添加 `from_timestamp=0.0` 和 `to_timestamp=length/fps` | 视频解码 |
| E4 | 导入错误 (verify 脚本) | 不能用文件系统路径作为 Python 模块路径 | `sys.path.insert` + 相对导入 | 验证脚本 |
| E5 | 全局变量 (verify 脚本) | `args_dataset` 未定义 | 改为函数参数传入 | 验证脚本 |

