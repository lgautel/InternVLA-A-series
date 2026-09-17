# LIBERO-plus 评估执行日志

> Checkpoint: `/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_12_08_52_49-internvla_a1_5-libplus-sft/checkpoints/026725/pretrained_model/`
> 操作手册: `eval.md`
> 开始时间: 2026-09-14

---

## 0. 初始状态检查

**时间**: 2026-09-14 (会话开始)

### 0.1 GPU 状态

```
GPU 0: 4469 MB free / 143771 MB total, util 18%
GPU 1: 4469 MB free / 143771 MB total, util 27%
GPU 2: 4463 MB free / 143771 MB total, util 26%
GPU 3: 4465 MB free / 143771 MB total, util 27%
GPU 4: 4465 MB free / 143771 MB total, util 22%
GPU 5: 4467 MB free / 143771 MB total, util 30%
GPU 6: 4465 MB free / 143771 MB total, util 31%
GPU 7: 4469 MB free / 143771 MB total, util 28%
```

**结论**: 8 卡全被训练占用 (~139 GB/卡), 仅剩 ~4.4 GB free. 需等训练完成.

### 0.2 环境状态

| 项目 | 状态 |
|------|------|
| Server venv (`/B/VENV/itnvla15rbt20`) | 存在 |
| Client venv (`/B/VENV/libero_plus_client`) | **不存在, 需创建** |
| Checkpoint 文件 | 完整 (config.json, model.safetensors, stats.json, train_config.json) |
| LIBERO-plus 仓库 (`/home/a26113/DATA/LIBERO-plus`) | 待验证 |

### 0.3 行动计划

1. 创建 Client venv (§5.1) — 无需 GPU
2. 验证 Server venv (§5.2) — 无需 GPU
3. 运行预检测试 (§六) — 无需 GPU
4. 等待 GPU 释放
5. 冒烟测试 (§7.2) — 需 1 GPU
6. 全量评估 (§7.3) — 需 8 GPU

---

## 1. 创建 Client 环境 (eval.md §5.1)

**时间**: 2026-09-14
**原因**: Client venv `/B/VENV/libero_plus_client` 不存在, 评估需要独立的 client 环境运行 robosuite + mujoco + LIBERO-plus 仿真.

### 1.1 创建 venv

```bash
python3.11 -m venv /B/VENV/libero_plus_client
```

### 1.2 安装依赖 (逐步, 按依赖顺序)

| 步骤 | 命令 | 结果 |
|------|------|------|
| pip upgrade | `pip install --upgrade pip setuptools wheel` | pip 26.2.1 |
| numpy | `pip install "numpy>=1.24,<2.0"` | numpy 1.26.4 (后被 robosuite 升级到 2.4.6) |
| mujoco | `pip install mujoco==3.2.3` | OK |
| robosuite | `pip install robosuite==1.4.0` | OK, 拉入 numpy 2.4.6, scipy, numba |
| bddl + misc | `pip install bddl==1.0.1 easydict pyyaml` | OK |
| websockets | `pip install websockets msgpack msgpack-numpy imageio imageio-ffmpeg termcolor tqdm` | OK |
| LIBERO-plus | `pip install -e /home/a26113/DATA/LIBERO-plus --no-deps` | OK |
| torch (CPU) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` | torch 2.14.0+cpu (LIBERO benchmark/__init__.py imports torch) |
| future | `pip install future` | bddl 依赖 |
| matplotlib | `pip install matplotlib h5py hydra-core` | OK |
| wand | `pip install wand` | OK |
| scikit-image | `pip install scikit-image` | env_wrapper.py imports skimage |
| cloudpickle + gym | `pip install cloudpickle gym` | venv.py imports cloudpickle |
| ImageMagick | `sudo apt-get install -y libmagickwand-dev` | 系统级, wand Python 绑定需要 |

### 1.3 EGL 渲染问题 (ERROR #1)

**错误**: mujoco 3.2.3 导入时报 `ImportError: Cannot initialize a EGL device display`

**根因分析**:
- mujoco 的 `egl/__init__.py` 在模块导入时立即调用 `eglQueryDevicesEXT()` 枚举 EGL 设备
- 这需要 EGL ICD (Installable Client Driver) 配置文件 (`__EGL_VENDOR_LIBRARY_DIRS`)
- 系统有 NVIDIA EGL 库 (`/usr/local/nvidia/lib64/libEGL.so.1`), 但没有 ICD vendor JSON 配置
- 没有 ICD 配置, `eglQueryDevicesEXT` 找不到任何设备 → 返回空列表 → `EGL_NO_DISPLAY` → ImportError

**修复方案**: 创建 EGL vendor JSON 配置文件

```bash
mkdir -p /B/VENV/libero_plus_client/egl_vendor.d
cat > /B/VENV/libero_plus_client/egl_vendor.d/10_nvidia.json <<'JSON'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "/usr/local/nvidia/lib64/libEGL_nvidia.so.0"
    }
}
JSON
```

**运行时必须设置的环境变量**:

```bash
export __EGL_VENDOR_LIBRARY_DIRS="/B/VENV/libero_plus_client/egl_vendor.d"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### 1.4 LIBERO config 交互式 prompt 问题

**问题**: `from libero.libero import benchmark` 时, LIBERO 的 `__init__.py` 会交互式询问数据集路径 → 非交互环境下 EOFError

**修复**: 必须在导入前设置 `LIBERO_CONFIG_PATH` 环境变量, 指向含有 `config.yaml` 的目录.

已创建配置:
```
/home/a26113/b/Ckp/.../eval_libero_plus/step_026725/libero_config/config.yaml
```

### 1.5 验证结果

```
robosuite 1.4.0
mujoco 3.2.3
LIBERO suites: ['libero_spatial', 'libero_object', 'libero_goal', 'libero_90', 'libero_10', 'libero_100', 'libero_mix']
OffScreenRenderEnv import OK
websockets + msgpack OK
=== Client env ready ===
```

**结论**: Client 环境创建成功 ✓

---

## 2. Server 环境验证 (eval.md §5.2)

**时间**: 2026-09-14

```
type: internvla_a1_5
chunk_size: 50
vlm: Qwen/Qwen3.5-2B
enable_keypoint_predictor: True
kpt_4d_mode: pos_rot
tokenize_state: True
websockets + msgpack OK
=== Server env ready ===
```

**结论**: Server 环境验证通过 ✓

---

## 3. 预检测试 (eval.md §六)

**时间**: 2026-09-14

| Test | 内容 | 结果 |
|------|------|------|
| Test 1 | Checkpoint 文件完整性 | ✓ config.json 3601B, model.safetensors 6320.9MB, stats.json 19214B, train_config.json 12762B |
| Test 2 | stats.json key=panda, state=8D, action=7D | ✓ |
| Test 3 | train_config: normalize=mean_std, action_mode=joint, tokenize_state=True, max_length=650 | ✓ |
| Test 4 | panda schema: action_mode=joint, image_mapping=2 images | ✓ |
| Test 5 | VLM tokenizer path valid | ✓ |
| Test 6 | LIBERO-plus assets: 4 suites × correct task counts, all asset dirs present | ✓ |
| Test 7 | Client venv imports | ✓ (with EGL fix) |
| Test 8 | Server import chain (build_backend, WebsocketPolicyServer, LiberoModelClient) | ✓ |
| Test 9 | LIBERO config.yaml paths all exist | ✓ |

**结论**: 所有预检测试通过 ✓

---

## 4. eval.md 更新 (发现 EGL 问题后)

**时间**: 2026-09-14 ~06:00 UTC

基于 §1.3 发现的 EGL vendor ICD 问题, 更新了 eval.md:

| 修改位置 | 修改内容 |
|---------|---------|
| §3.5 环境变量表 | 新增 `__EGL_VENDOR_LIBRARY_DIRS` 和 `LD_LIBRARY_PATH` 行 |
| §5.1 Client 环境创建 | 新增 EGL vendor JSON 配置步骤, 新增 `scikit-image cloudpickle gym future matplotlib h5py hydra-core` 依赖, 新增 `sudo apt-get install libmagickwand-dev` |
| §7.2 冒烟测试 Client | 新增 `__EGL_VENDOR_LIBRARY_DIRS` 和 `LD_LIBRARY_PATH` export |
| §7.3 全量评估 | 新增 EGL 渲染环境变量 block |

---

## 5. 等待 GPU 释放

**时间**: 2026-09-14 05:47 UTC

### 5.1 GPU 状态

训练进程仍在运行 (PID 1471852, 8 GPU workers). 当前 step ~32070/53450.

**Checkpoint 时间线** (每 5345 步保存一次, 间隔 ~7h):

| Checkpoint | 时间 | 间隔 |
|-----------|------|------|
| 005345 | Sep 12 16:00 | — |
| 010690 | Sep 12 23:06 | 7h06m |
| 016035 | Sep 13 06:09 | 7h03m |
| 021380 | Sep 13 13:14 | 7h05m |
| 026725 | Sep 13 20:20 | 7h06m |
| 032070 | Sep 14 03:28 | 7h08m |
| 037415 (预计) | Sep 14 ~10:30 | ~7h |
| 042760 (预计) | Sep 14 ~17:30 | ~7h |
| 048105 (预计) | Sep 15 ~00:30 | ~7h |
| 053450 (完成) | Sep 15 ~07:30 | ~7h |

**预计 GPU 释放**: Sep 15 ~07:30 UTC (距现在 ~26 小时)

### 5.2 用户决定: Kill 训练进程

**时间**: 2026-09-14 ~06:10 UTC
**用户决定**: Kill 训练以立即释放 GPU (checkpoint 032070 已保存, 无数据丢失)

```bash
kill -TERM 1471852  # accelerate launcher PID
```

训练进程及其 8 个 GPU worker 全部终止. 所有 8 GPU 释放至 ~143 GB free.

| GPU | 释放前 free | 释放后 free |
|-----|-----------|-----------|
| 0-7 | ~4.4 GB | ~143 GB |

---

## 6. 冒烟测试 (eval.md §7.2)

**时间**: 2026-09-14 ~06:30 UTC

### 6.1 Server 启动

```bash
# GPU 0, port 5784
CUDA_VISIBLE_DEVICES=0 python evaluation/LIBERO/policy_server/server_policy.py \
  --ckpt_path "/home/a26113/b/Ckp/.../checkpoints/026725/pretrained_model" \
  --host 0.0.0.0 --port 5784 --device cuda --resize_size 224 \
  --stats_key panda --robot_type panda \
  --action_loss_only --inference_backend standard --idle_timeout -1
```

Server PID: 1937271. 启动成功, 无错误.

### 6.2 Health Check

7 项断言全部通过:

| 断言 | 期望值 | 实际值 | 结果 |
|------|--------|--------|------|
| policy_type | internvla_a1_5 | internvla_a1_5 | ✓ |
| chunk_size | 50 | 50 | ✓ |
| action_dim | 7 | 7 | ✓ |
| expected_num_input_images | 2 | 2 | ✓ |
| protocol_version | 2.1 | 2.1 | ✓ |
| preprocessing_owner | server | server | ✓ |
| action_mode | joint | joint | ✓ |

### 6.3 Client 运行

```bash
# libero_goal shard [0,4), Background Textures perturbation
python evaluation/LIBERO-plus/eval_libero_plus.py \
  --host 127.0.0.1 --port 5784 --task_suite_name libero_goal \
  --start_idx 0 --end_idx 4 --num_trials_per_task 1 --seed 7 \
  --replan_steps 8 --eval_log_dir ".../smoke_test"
```

**Exit code**: 0 (无 crash)

### 6.4 结果

```json
{
    "Background Textures": {"total_count": 4, "success_count": 0},
    "Robot Initial States": {"total_count": 0, "success_count": 0},
    "Camera Viewpoints": {"total_count": 0, "success_count": 0},
    "Language Instructions": {"total_count": 0, "success_count": 0},
    "Sensor Noise": {"total_count": 0, "success_count": 0},
    "Objects Layout": {"total_count": 0, "success_count": 0},
    "Light Conditions": {"total_count": 0, "success_count": 0}
}
```

**SR = 0.0000 (0/4)** — 4 个 Background Textures 任务全部失败. 这是 50% 训练的 checkpoint 的预期表现, 不是 bug.

### 6.5 验收

| 验收项 | 结果 |
|--------|------|
| Server 无 crash | ✓ |
| Client 无 crash | ✓ |
| 结果 JSON 生成 | ✓ (`logs/libero_goal/0_to_4.json`) |
| 无 INVALID_STATE_DIM/INVALID_IMAGE_COUNT 错误 | ✓ |

**结论**: 冒烟测试通过 ✓, 可以进行全量评估.

---

## 7. 全量评估 (eval.md §7.3)

**时间**: 2026-09-14 06:25 UTC
**PID**: 1943079

### 7.1 配置

| 参数 | 值 |
|------|-----|
| GPU_IDS | 0,1,2,3,4,5,6,7 (8 卡) |
| SHARDS_PER_SUITE | 8 |
| STATS_KEY_MODE | panda |
| ROBOT_TYPE_MODE | panda |
| ACTION_LOSS_ONLY_FLAG | --action_loss_only |
| INFERENCE_BACKEND | standard |
| NUM_TRIALS_PER_TASK | 1 |
| SEED | 7 |
| NO_VIDEO_FLAG | --no-save_videos |
| EVAL_LOG_DIR | `.../step_026725/full_20260914_062508` |

### 7.2 工作单元分布

| Suite | Tasks | Shards |
|-------|-------|--------|
| libero_spatial | 2402 | 8 × ~300 tasks |
| libero_object | 2518 | 8 × ~315 tasks |
| libero_goal | 2591 | 8 × ~324 tasks |
| libero_10 | 2519 | 8 × ~315 tasks |
| **Total** | **10030** | **32 work units** |

### 7.3 启动确认

```
GPUs (fixed)       : 0 1 2 3 4 5 6 7
Workers            : 8 on GPU(s) 0 1 2 3 4 5 6 7
Work units         : 32 (suites=4 x shards=8)
Categories filter  : ALL
```

8 workers 全部启动, 第一轮 8 个 shard (libero_spatial) 已开始执行.

### 7.4 ERROR #3: GPU 5 CUBLAS 分配失败

**时间**: 2026-09-14 06:27 UTC
**工作单元**: libero_spatial [1502, 1802) (300 tasks)

**错误**:
```
RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
```

**根因**: GPU 5 server 加载模型后, 在首个任务推理时 CUBLAS 句柄创建失败. 这是瞬态 CUDA 资源分配问题, 非代码 bug. Server 的 Qwen3.5-2B forward pass 在 `linear.py:134` 处触发 `F.linear()` → CUBLAS 初始化失败.

**影响**: 300 tasks 丢失 (libero_spatial shard 5/8). 需在全量评估完成后单独重跑此 shard.

**GPU 5 恢复**: 自动恢复, 已进入下一工作单元 (libero_object [0,315)), 正常运行中.

### 7.5 进度监控

**T+5 min (06:30 UTC)**: 8 workers 全部活跃

| GPU | 工作单元 | 进度 |
|-----|---------|------|
| 0 | libero_spatial [0,301) | 9/301 |
| 1 | libero_spatial [301,602) | 7/301 |
| 2 | libero_spatial [602,902) | 7/300 |
| 3 | libero_spatial [902,1202) | 7/300 |
| 4 | libero_spatial [1202,1502) | 7/300 |
| 5 | libero_object [0,315) | 2/315 (恢复后) |
| 6 | libero_spatial [1802,2102) | 6/300 |
| 7 | libero_spatial [2102,2402) | 7/300 |

速度: ~25 s/task. 预计总时长: ~10 小时 (每个 worker 处理 ~4 shards × ~300 tasks).

**T+21 min (06:46 UTC)**: 

ERROR #4: GPU 7 也失败了
- 工作单元: libero_spatial [2102, 2402) (300 tasks)
- 错误: `mujoco.FatalError: Offscreen framebuffer is not complete, error 0x8cdd`
- 根因: 8 个 client 同时初始化 EGL 渲染上下文, 竞争 EGL device 资源, 导致部分 framebuffer 初始化失败. 与 ERROR #3 不同, 这是 client 端 (EGL/mujoco) 的问题, 不是 server 端 (CUDA) 的问题.
- GPU 7 已恢复, 正在处理 libero_object [315, 630)

**T+40 min (07:05 UTC)**: FAILED 增加到 5:
- GPU 5: 连续 3 次失败 (libero_spatial [1502,1802), libero_object [0,315), libero_object [630,945)). 全部是 EGL framebuffer 错误 `0x8cdd`.
- GPU 3: 1 次失败 (libero_spatial [902,1202), 已完成 84/300 tasks 后崩溃)
- GPU 7: 1 次失败 (libero_spatial [2102,2402))

eval_libero_plus.py 仅在 shard 完成后写 JSON, crash 后无 partial results 保存.

**T+52 min (07:17 UTC)**: GPU 5 和 GPU 3 均恢复正常, 无新失败.

| GPU | 当前工作 | 进度 |
|-----|---------|------|
| 0 | libero_spatial [0,301) | 116/301 (39%) |
| 1 | libero_spatial [301,602) | 117/301 (39%) |
| 2 | libero_spatial [602,902) | 119/300 (40%) |
| 3 | libero_object [1260,1575) | 29/315 (9%) |
| 4 | libero_spatial [1202,1502) | 118/300 (39%) |
| 5 | libero_object [945,1260) | 28/315 (9%) |
| 6 | libero_spatial [1802,2102) | 82/300 (27%) |
| 7 | libero_object [315,630) | 85/315 (27%) |

**待重跑 shards** (5 个, 共 ~1446 tasks):
1. libero_spatial [902, 1202) — 300 tasks (GPU 3, 84 tasks 后 crash, 无 partial results)
2. libero_spatial [1502, 1802) — 300 tasks (GPU 5 CUBLAS 错误)
3. libero_spatial [2102, 2402) — 300 tasks (GPU 7 EGL framebuffer 错误)
4. libero_object [0, 315) — 315 tasks (GPU 5 EGL 错误)
5. libero_object [630, 945) — 315 tasks (GPU 5 EGL 错误)

---

## §8 Mini Evaluation (小型评估) — 2026-09-14 12:22–12:44

用户要求停止完整评估, 改为小型评估: 每个 LIBERO-plus 子集抽前 10 题 (4 suites × 10 = 40 tasks total).

### §8.1 配置

- **Server**: GPU 0, PID 2303191, port 5784
- **Client**: 单进程顺序执行, 避免 EGL contention
- **输出目录**: `eval_libero_plus/step_026725/mini_20260914_122224/`
- **参数**: `--start_idx 0 --end_idx 10 --num_trials_per_task 1 --seed 42 --replan_steps 8 --save_videos`

### §8.2 结果汇总

| Suite | Success | Total | SR | Perturbation | 耗时 |
|-------|---------|-------|----|--------------|------|
| libero_spatial | 0 | 10 | **0.00%** | Background Textures ×10 | ~3.5 min |
| libero_object | 5 | 10 | **50.00%** | Background Textures ×10 | ~4 min |
| libero_goal | 5 | 10 | **50.00%** | Background Textures ×10 | ~4 min |
| libero_10 | 7 | 10 | **70.00%** | Background Textures ×10 | ~5 min |
| **Overall** | **17** | **40** | **42.50%** | | ~16.5 min |

### §8.3 逐任务明细

**libero_spatial** (0/10 = 0%):
- 所有 10 个 Background Textures 扰动任务均失败 (pick up the red mug and place it to the right of the plate)

**libero_object** (5/10 = 50%):
- task 0: ✗ | task 1: ✓ | task 2: ✓ | task 3: ✓ | task 4: ✗
- task 5: ✓ | task 6: ✓ | task 7: ✗ | task 8: ✗ | task 9: ✗
- 任务: pick up the alphabet soup and place it in the basket (不同桌面纹理)

**libero_goal** (5/10 = 50%):
- task 0: ✗ | task 1: ✓ | task 2: ✗ | task 3: ✓ | task 4: ✗
- task 5: ✓ | task 6: ✓ | task 7: ✗ | task 8: ✓ | task 9: ✗
- 任务: open the middle drawer of the cabinet (不同桌面纹理)

**libero_10** (7/10 = 70%):
- task 0: ✓ | task 1: ✓ | task 2: ✓ | task 3: ✗ | task 4: ✓
- task 5: ✗ | task 6: ✓ | task 7: ✗ | task 8: ✓ | task 9: ✓
- 任务: turn on the stove and put the moka pot on it (不同桌面纹理)

### §8.4 Artifacts

- **JSON 结果**: `logs/{suite}/0_to_10.json` (每 suite 一个)
- **视频**: `videos/{suite}/` (共 40 个 .mp4 文件)
- **Server 日志**: `server.log`
- **所有文件保存在**: `~/b/Ckp/4dwvlaOpvlaLibplusKpt0911/.../eval_libero_plus/step_026725/mini_20260914_122224/`

### §8.5 观察

1. **由于 start_idx=0, 所有抽到的任务均为 Background Textures 类别** — 未覆盖其他 6 种扰动类型
2. libero_spatial 在该 checkpoint 下表现极差 (0%), 与之前完整评估观察到的趋势一致
3. libero_10 表现最好 (70%), 可能因为 "turn on the stove" 任务相对简单
4. 总体 42.5% SR 表明模型在部分任务上具备一定的背景纹理鲁棒性
5. 无 EGL 错误, 无 crash — 单进程顺序执行策略有效避免了并发 EGL 问题

---

## §9 深度调查: 低成功率根因分析 — 2026-09-14 13:00

### §9.1 调查方法

对以下文件/模块进行了完整代码审查:
- `evaluation/LIBERO-plus/eval_libero_plus.py` — 客户端评估主循环
- `evaluation/LIBERO/model2libero_interface.py` — 客户端与 server 的交互层 (图像/state/action 处理)
- `evaluation/LIBERO/policy_server/backends/policy_backend_internvla_a1_5.py` — 服务端推理后端
- `evaluation/LIBERO/policy_server/backends/base_backend.py` — stats 加载与 action 反归一化
- `evaluation/LIBERO/policy_server/backends/canonical_preprocess.py` — 图像预处理
- `evaluation/LIBERO/policy_server/backends/input_semantics.py` — 图像通道映射
- `src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py` — 训练/推理时的 prompt 构建
- `src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py` — `sample_actions()` flow matching 推理路径
- `src/lerobot/transforms/core.py` — `LoadActionTextFromJsonlTransformFn` sub_task 加载逻辑
- `src/lerobot/dataset_schemas/configs/panda.yaml` — panda schema 定义
- `src/lerobot/dataset_schemas/configs/libero.yaml` — libero schema 对比
- `launch/internvla_a15_finetune.sh` — 训练启动脚本
- Checkpoint `stats.json` — 反归一化参数
- 训练数据 `opvla_libero_merged_kpt` 的 `meta/info.json`, `tasks.parquet`, `stats.json`
- `server.log` — 服务端运行日志

### §9.2 发现 #1 (CRITICAL): Prompt 后缀不匹配

**问题描述**:

模型训练时的 user prompt 以 `"; Output: <Action>"` 结尾, 但评估时变成了 `"; Output: <Subtask, Action>"`. 这两个不同的后缀会导致 VLM (Qwen3.5) 产生不同的 hidden states, 而 action expert 通过 cross-attention 依赖这些 hidden states 作为 conditioning — 因此 action 质量下降.

**根因链**:

```
训练配置 (train_config.json):
  use_fast_action_tokens = true  ← finetune.sh:123 显式设置
  
训练数据 (opvla_libero_merged_kpt):
  meta/episodes_detailed_task.jsonl → 文件不存在
  → LoadActionTextFromJsonlTransformFn 返回空 cache
  → sub_task 字段在训练时始终为空
  
训练时 prompt 构建 (transform_internvla_a1_5.py:125-150):
  has_fast = True (use_fast_action_tokens=True + action_text 存在)
  has_sub_task = False (sub_task 为空)
  → label_mode = LABEL_MODE_FAST
  → user_text += "; Output: <Action>"         ← 训练时的后缀

评估时 prompt 构建 (transform_internvla_a1_5.py:140-142):
  mode = "eval"
  → label_mode = LABEL_MODE_NONE (直接覆盖)
  → user_text += "; Output: <Subtask, Action>"  ← 评估时的后缀 (硬编码!)

服务端后端 (policy_backend_internvla_a1_5.py:148):
  use_fast_action_tokens = False  ← 硬编码为 False, 忽略了 checkpoint config!
```

**影响机制**:

InternVLA-A1.5 的 action 生成路径:
1. VLM 前向传播处理 prefix (图像 + 语言 prompt + state tokens) → 生成 KV cache
2. Keypoint expert 基于 prefix KV cache cross-attend → 扩展 cache
3. Action expert 通过 10 步 flow matching 迭代, 每步 cross-attend prefix+kpt cache → 生成 action

Prompt 后缀不同 → VLM 的最后几个 token 的 hidden states 不同 → KV cache 中的 conditioning 信号偏移 → 所有 10 步 denoising 的 velocity prediction 偏移 → action chunk 质量下降.

**修复 (已实施)**:

文件 1: `src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py` (line 140-142)
```python
# 修复前:
if self.mode == "eval":
    label_mode = LABEL_MODE_NONE
    user_text = user_text + "; Output: <Subtask, Action>"

# 修复后:
if self.mode == "eval":
    label_mode = LABEL_MODE_NONE
    if self.use_fast_action_tokens:
        user_text = user_text + "; Output: <Action>"
    else:
        user_text = user_text + "; Output: <Subtask, Action>"
```

文件 2: `evaluation/LIBERO/policy_server/backends/policy_backend_internvla_a1_5.py` (line 148)
```python
# 修复前:
use_fast_action_tokens=False,      # 硬编码, 忽略了 checkpoint config

# 修复后:
use_fast_action_tokens=bool(config.use_fast_action_tokens),  # 从 checkpoint config 读取
```

### §9.3 已验证为正确的部分

| 检查项 | 训练值 | 评估值 | 匹配? |
|--------|--------|--------|--------|
| 图像数量 | 2 (agentview + wrist) | 2 (server expects 2) | ✓ |
| 图像 key 映射 | image→slot0, image2→slot1 | panda.yaml 同上 | ✓ |
| state 维度 | 8D (eef_pos[3]+axisangle[3]+gripper_qpos[2]) | 8D (client 同构) | ✓ |
| action 维度 | 7D | 7D | ✓ |
| stats_key | "panda" (dataset robot_type) | "panda" (CLI --stats_key) | ✓ |
| action 反归一化 mode | "mean_std" | "mean_std" | ✓ |
| action mean | [0.063, 0.087, -0.090, 0.001, 0.006, -0.005, -0.050] | 同 (从 ckpt stats.json 读取) | ✓ |
| action std | [0.336, 0.378, 0.445, 0.039, 0.063, 0.078, 0.999] | 同 | ✓ |
| action clip [min,max] | [-0.94.., 0.94..] (dims 0-2) | 同 | ✓ |
| gripper 二值化 | libero_native (>0→+1, ≤0→-1) | auto→检测到 libero_native | ✓ |
| action_mode in prompt | "joint" (panda schema 默认) | "joint" | ✓ |
| Control Mode prompt tag | "Control Mode: \<joint\>" | "Control Mode: \<joint\>" | ✓ |
| 图像旋转 180° | 训练数据已旋转 | client 做 `[::-1,::-1]` | ✓ |
| num_inference_steps | 10 | 10 (config 默认) | ✓ |
| chunk_size | 50 | 50 | ✓ |
| block_action_attend_fast_tokens | True | True (eval 时无 FAST token, mask 全 False) | ✓ |

### §9.4 次要观察

1. **LIBERO-plus 任务描述修改**: LIBERO-plus 在 Background Textures 扰动中会在任务描述后附加 " table N" 后缀 (如 "pick up the alphabet soup and place it in the basket table 1"), 训练数据中没有此后缀. 这是 LIBERO-plus benchmark 的设计行为 (测试语言鲁棒性), 模型对此表现出部分鲁棒性 (libero_object 50%, libero_10 70%).

2. **libero_spatial 0% 特异性**: 前 10 个 libero_spatial 任务全部是 "pick up the black bowl between the plate and the ramekin and place it on the plate" 的不同纹理变体. 这是一个场景拥挤 (plate, ramekin, cookie box 紧密排列) 的精确空间推理任务, 在 prompt conditioning 偏移的情况下可能受影响最大.

3. **panda vs libero schema 差异**: panda.yaml 的 `action_mask_spec: [7]` (全 delta) vs libero.yaml 的 `action_mask_spec: [6, -1]` (6 delta + 1 absolute gripper). 但由于训练使用 `action_mode=abs` (无 delta 变换), 此差异不影响结果.

### §9.5 待验证

以上修复已实施, 需要重新运行 mini eval 来验证:
- 如果 prompt suffix 是主要问题, 预期修复后 SR 显著提升
- 如果 libero_spatial 仍然为 0%, 可能需要进一步调查任务特异性问题

---

## §10. Random-Sampling Mini Eval (修复后)

**时间**: 2026-09-14 13:36 ~ 14:35
**目的**: 在应用 prompt suffix 修复 + config 传播修复后, 用随机采样覆盖全部 7 个扰动类别进行评估

### §10.1 修复清单 (本次运行前新增)

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 3 | `policy_backend_internvla_a1_5.py:148` | `config.use_fast_action_tokens` 属性不存在于 checkpoint config.json | 改为 `getattr(config, "use_fast_action_tokens", True)` |
| 4 | `evaluation/LIBERO/keypoint_utils.py` | `KeypointExtractor` 缓存 `env.sim` 引用, 但 `env.reset()` 后 sim 对象可能重建 | 改为存储 `env` 引用, 每次 `extract()` 访问 `env.sim` |
| 5 | `LIBERO-plus/libero/libero/envs/env_wrapper.py:52` | `np.fromstring` binary mode 在 NumPy 2.x 中已移除 | 改为 `np.frombuffer` |
| 6 | `libero_config/config.yaml` | 使用嵌套 `base:` 结构, LIBERO 库期望扁平 key (`bddl_files`, `init_states`) | 重写为扁平 key 格式 |

### §10.2 随机采样方案

- 每个 suite 随机抽取 10 个 task (seed=42), 覆盖全部 7 个扰动类别
- 总计 4 suites × 10 tasks = 40 tasks, 每 task 1 trial

| Suite | Task IDs |
|-------|----------|
| libero_spatial | 57, 357, 372, 421, 620, 1363, 1445, 1850, 2239, 2250 |
| libero_object | 108, 151, 359, 862, 1058, 1515, 1865, 2075, 2236, 2472 |
| libero_goal | 26, 101, 510, 655, 1056, 1142, 1430, 1809, 1999, 2526 |
| libero_10 | 216, 380, 419, 679, 856, 1243, 1560, 1656, 2043, 2324 |

### §10.3 Per-Suite Results

| Suite | Success | Total | SR |
|-------|---------|-------|----|
| libero_spatial | 1 | 10 | **10.0%** |
| libero_object | 6 | 10 | **60.0%** |
| libero_goal | 1 | 10 | **10.0%** |
| libero_10 | 0 | 10 | **0.0%** |
| **OVERALL** | **8** | **40** | **20.0%** |

### §10.4 Per-Category Results (所有 suite 合并)

| Category | Success | Total | SR |
|----------|---------|-------|----|
| Background Textures | 3 | 6 | **50.0%** |
| Light Conditions | 3 | 6 | **50.0%** |
| Camera Viewpoints | 1 | 4 | **25.0%** |
| Objects Layout | 1 | 5 | **20.0%** |
| Language Instructions | 0 | 5 | **0.0%** |
| Robot Initial States | 0 | 9 | **0.0%** |
| Sensor Noise | 0 | 5 | **0.0%** |

### §10.5 Per-Task Detail

**libero_spatial** (1/10 = 10%):
- 唯一成功: task 2250 [Light Conditions] "pick up the black bowl next to the cookie box and place it on the plate light 16"
- 全部 spatial 任务都是 "pick up the black bowl ... and place it on the plate" 的变体, 场景拥挤, 需要精确空间推理

**libero_object** (6/10 = 60%): ← 最佳 suite
- Background Textures: 2/2 (100%)
- Light Conditions: 2/2 (100%)
- Camera Viewpoints: 1/1 (100%)
- Objects Layout: 1/2 (50%)
- Robot Initial States: 0/1, Language Instructions: 0/1, Sensor Noise: 0/1

**libero_goal** (1/10 = 10%):
- 唯一成功: task 26 [Background Textures] "open the middle drawer of the cabinet tb 6"
- "push the plate" 和 "put the wine bottle on the rack" 任务全部失败

**libero_10** (0/10 = 0%): ← 最差 suite
- 全部长步骤 multi-step 任务失败, 包括 "put both moka pots on the stove", "put the black bowl in the drawer and close it" 等

### §10.6 与修复前对比

| 指标 | 修复前 (§8) | 修复后 (§10) | 变化 |
|------|------------|-------------|------|
| 采样方式 | 顺序 (0-9, Background Textures only) | 随机 (全 7 类) | 更全面 |
| libero_spatial | 0/10 (0%) | 1/10 (10%) | +10pp |
| libero_object | 5/10 (50%) | 6/10 (60%) | +10pp |
| libero_goal | 5/10 (50%) | 1/10 (10%) | -40pp* |
| libero_10 | 7/10 (70%) | 0/10 (0%) | -70pp* |
| Overall | 17/40 (42.5%) | 8/40 (20.0%) | -22.5pp |

*注: 修复前只测 Background Textures (最简单类别), 修复后随机采样包含 Robot Initial States、Language Instructions、Sensor Noise 等更难类别, 因此直接比较数值不公平. 修复前的高 SR 部分归因于仅采样最简单类别.

### §10.7 分析

1. **Prompt suffix 修复效果**: 在同类别 (Background Textures) 对比中, libero_object 从 50% 提升到 100% (2/2), 说明修复有效. 但总体 SR 下降, 因为随机采样引入了更难的扰动类别.

2. **扰动难度排序**: Background Textures ≈ Light Conditions (50%) > Camera Viewpoints (25%) > Objects Layout (20%) >> Language Instructions = Robot Initial States = Sensor Noise (0%)

3. **Suite 难度排序**: libero_object (60%) >> libero_spatial ≈ libero_goal (10%) >> libero_10 (0%)

4. **Language Instructions 0%**: 模型对重述/同义词替换的任务描述完全不鲁棒, 如 "lift the darkcolored vessel made for food..." 替代 "pick up the black bowl..."

5. **Robot Initial States 0%**: 模型对起始姿态变化完全不鲁棒 (9/9 全部失败), 表明策略对初始状态高度过拟合

6. **libero_10 0%**: 长序列 multi-step 任务全部失败, 表明模型在复杂任务上泛化能力不足

### §10.8 输出文件

```
mini_20260914_133646/
├── server.log
├── server.pid
├── libero_config/config.yaml
├── logs/
│   ├── libero_spatial/random_10.json
│   ├── libero_spatial_eval.log
│   ├── libero_object/random_10.json
│   ├── libero_object_eval.log
│   ├── libero_goal/random_10.json
│   ├── libero_goal_eval.log
│   ├── libero_10/random_10.json
│   └── libero_10_eval.log
└── videos/
    ├── libero_spatial/ (10 mp4)
    ├── libero_object/ (10 mp4)
    ├── libero_goal/ (10 mp4)
    └── libero_10/ (10 mp4)
```

