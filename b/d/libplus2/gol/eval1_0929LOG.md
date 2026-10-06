# LIBERO-plus `libero_goal` 评估执行日志

> **EXPR_NAME**: `4dwvlaLbPlusGol0929`
>
> **Checkpoint**: `016020/pretrained_model`
>
> **开始时间**: 2026-10-01

---

## Phase 0: GPU 清理

```
时间: 2026-10-01
操作: kill bigmatrix_multiply_optimization (PID 2076809)
结果: 8× H200 全部释放 (0 MiB used)
```

---

## Phase 1: Client Venv 搭建与 NumPy Patch

```
时间: 2026-10-01
Venv: /B/VENV/libero_plus_client (python3.11)

已安装包:
  - mujoco==3.2.3, robosuite==1.4.0, bddl==1.0.1
  - libero (editable, /B/SRC/LIBERO-plus)
  - websockets==17.1, msgpack==1.2.3, msgpack-numpy==0.4.8
  - scikit-image==0.26.0, cloudpickle==3.1.2, gymnasium==1.3.0
  - torch==2.14.1+cpu
  - termcolor, h5py, Pillow, numba, future, easydict, matplotlib, opencv-python-headless, wand, gym

EGL 配置:
  - /B/VENV/libero_plus_client/egl_vendor.d/10_nvidia.json → /usr/local/nvidia/lib64/libEGL_nvidia.so.0
  - LD_LIBRARY_PATH 需含 /opt/conda/lib (ImageMagick) 和 /usr/local/nvidia/lib64

NumPy 2.x Patches:
  - env_wrapper.py: np.fromstring→np.frombuffer, np.float_→np.float64
  - venv.py: np.fromstring→np.frombuffer

LIBERO Config:
  - /B/VENV/libero_plus_client/libero_config/config.yaml (避免交互式 prompt)

验证: libero_goal benchmark import OK, 2591 tasks
```

---

## Phase 2: Preflight 检查

```
时间: 2026-10-01
命令: bash b/s/libplus2/gol/preflight_goal.sh
结果: 30 passed, 0 failed — PREFLIGHT PASSED

检查项: checkpoint files, contract schema, MJCF MD5, config consistency,
        server venv, client venv, EGL vendor, NumPy patches, task config,
        evaluation scripts, goal extension scripts, VLM weights
```

---

## Phase 3: Test Suite 验收

```
时间: 2026-10-01
命令: bash b/s/libplus2/gol/test_eval_goal.sh
结果: 9 passed, 0 failed — TEST SUITE PASSED

T1: Preflight           ✅ (30/30)
T2: Contract Loader     ✅
T3: KeypointHistory     ✅
T4: pack_like_training  ✅
T5: MJCF FK consistency ✅
T6: DRY_RUN derived     ✅
T7: kpt_hist_len=92     ✅
T8: orientation=raw     ✅
T9: stats.json panda    ✅
```

---

## Phase 4: Smoke Test

### 修复过程中发现的问题与 Fix

#### Issue 1: Client venv 缺少依赖包
```
tqdm, future, easydict, matplotlib, opencv-python-headless, wand, gym
→ 逐一安装
```

#### Issue 2: LIBERO-plus assets 缺失
```
FileNotFoundError: .../scenes/tabletop250/tabletop_table_FabricCashmere001_COL_VAR1_6K.xml
原因: libero/libero/assets/ 在 .gitignore 中, 需从 HuggingFace 下载 assets.zip
操作: wget https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip
     解压到 /B/SRC/LIBERO-plus/libero/libero/assets/ (6.0G)
```

#### Issue 3: torch.load weights_only 默认值变更 (PyTorch 2.6+)
```
_pickle.UnpicklingError: Weights only load failed.
原因: client venv torch==2.14.1+cpu, 默认 weights_only=True
修复: sed -i 's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/g'
     LIBERO-plus/libero/libero/benchmark/__init__.py (2处)
     LIBERO-plus/libero/lifelong/metric.py, evaluate.py, utils.py
```

#### Issue 4: keypoint_history_max_len 不一致 (最关键 Bug)
```
CUDA error: device-side assert triggered (vectorized_gather_kernel: index out of bounds)
根因: goal_client.py 从 contract.json 读取 kpt_history_max_len=200,
     但 checkpoint 训练时用的是 92, 模型 embedding 只支持 92 个位置.
     第 93 步 his_len=93 超出 embedding 范围 → CUDA assert.
修复: goal_client.py 改为优先读取 KPT_HISTORY_MAX_LEN 环境变量 (eval_goal_wrapper.sh 从 checkpoint config 读出 92 并 export)
文件: b/s/libplus2/gol/goal_client.py line 74
```

#### Issue 5: CATEGORIES="Language Instructions" 空格被拆分
```
smoke 模式设 CATEGORIES="Language Instructions", 被 shell word-split 为两个 token,
argparse 收到 ["Language", "Instructions"] 无法匹配 "Language Instructions" → 所有 task 被跳过
影响: 仅 smoke 模式, full 模式不设 CATEGORIES 不受影响
状态: 不修改原脚本, smoke 测试改为直接运行 3 task 验证
```

### Smoke Test 结果

```
时间: 2026-10-01
方式: 手动 3-task 测试 (start_idx=0, end_idx=3, GPU 0)
结果: SR=1.0000 (3/3), 0 failures, 0 crashes
任务: Background Textures 类别 — open_the_middle_drawer_of_the_cabinet (table_1, table_10, table_12)
Action NPZ: 3 个文件已保存
KPT_HISTORY_MAX_LEN: 92 (from checkpoint)
日志: /B/Log/4dwvlaLbPlusGol0929/manual_smoke4_20261001_093420/
```

---

## Phase 5: Full Evaluation

### 启动信息

```
启动时间: 2026-10-01 09:39:20 UTC
EVAL_MODE: full
LOG_DIR: /B/Log/4dwvlaLbPlusGol0929/20261001_093920_eval
KPT_HISTORY_MAX_LEN: 92 (from checkpoint config)

GPU 布局 (8× H200):
  GPU 0: Server PID=2375464, port=5784, shard [0, 324)
  GPU 1: Server PID=2375469, port=5785, shard [324, 648)
  GPU 2: Server PID=2375470, port=5786, shard [648, 972)
  GPU 3: Server PID=2375473, port=5787, shard [972, 1296)
  GPU 4: Server PID=2375481, port=5788, shard [1296, 1620)
  GPU 5: Server PID=2375472, port=5789, shard [1620, 1944)
  GPU 6: Server PID=2375482, port=5790, shard [1944, 2268)
  GPU 7: Server PID=2375484, port=5791, shard [2268, 2591)

总任务数: 2591
每 GPU: ~324 tasks (GPU 7: 323)
max_steps/task: 300, trials/task: 1
EGL backend: egl, 8 concurrent client workers
Keypoints: ENABLED, Rotation: DISABLED (raw)
Stats key: panda, Robot: panda, Gripper: libero_native

Servers 全部在 09:41:04 UTC 准备就绪, 各 shard 同步启动
```

### 各 GPU Shard 完成情况

| GPU | Shard 范围 | SR | 成功/总数 | 完成时间 (UTC) | 耗时 |
|-----|-----------|-----|----------|---------------|------|
| 0 | [0, 324) | 62.96% | 204/324 | 13:09 | 3h28m |
| 1 | [324, 648) | 1.54% | 5/324 | 14:13 | 4h32m |
| 2 | [648, 972) | 17.28% | 56/324 | 13:56 | 4h15m |
| 3 | [972, 1296) | 26.23% | 85/324 | 13:49 | 4h08m |
| 4 | [1296, 1620) | 33.33% | 108/324 | 15:19 | 5h38m |
| 5 | [1620, 1944) | 19.14% | 62/324 | 17:36 | 7h55m |
| 6 | [1944, 2268) | 49.38% | 160/324 | 12:14 | 2h33m |
| 7 | [2268, 2591) | 75.54% | 244/323 | 12:58 | 3h17m |

### 总体结果

```
完成时间: 2026-10-01 17:36:35 UTC
总耗时: 477 min (7h57m)
退出码: 0
```

**Overall SR: 35.7% (924/2591)**

#### 按扰动类别分解

| 类别 | SR (%) | 成功/总数 |
|------|--------|----------|
| Light Conditions | **76.7** | 214/279 |
| Background Textures | **71.5** | 201/281 |
| Objects Layout | **52.9** | 225/425 |
| Language Instructions | **33.7** | 138/410 |
| Sensor Noise | **17.4** | 66/379 |
| Camera Viewpoints | **16.7** | 68/408 |
| Robot Initial States | **2.9** | 12/409 |

#### 失败记录

```
Failure JSONs: 1667
Action NPZs:  1667
Crashes:       0
Manifest:  /B/Log/4dwvlaLbPlusGol0929/20261001_093920_eval/failures_manifest.json
Replay:    /B/Log/4dwvlaLbPlusGol0929/20261001_093920_eval/replay_commands.sh

按类别:
  Background Textures:   80 failures / 281 tasks
  Camera Viewpoints:    340 failures / 408 tasks
  Language Instructions: 272 failures / 410 tasks
  Light Conditions:      65 failures / 279 tasks
  Objects Layout:       200 failures / 425 tasks
  Robot Initial States: 397 failures / 409 tasks
  Sensor Noise:         313 failures / 379 tasks
```

### 观察与分析

1. **Light Conditions (76.7%) 和 Background Textures (71.5%)** 表现最好 — 模型对光照和背景纹理变化有较强鲁棒性
2. **Robot Initial States (2.9%)** 表现极差 — 模型几乎无法应对初始状态扰动，说明对起始姿态泛化能力很弱
3. **Camera Viewpoints (16.7%)** 和 **Sensor Noise (17.4%)** 也很差 — 视角变化和传感器噪声对策略影响严重
4. **Sensor Noise 类别耗时最长** — GPU 4/5 处理该类别，每 task ~200s (其他类别 ~40s)，是因为噪声导致模型预测不稳定、步数用满
5. **0 crashes** — fork-per-task 隔离 (B10) 完全有效，无 EGL SIGABRT

---

## Phase 6: 后处理

```
Archive: /B/Log/4dwvlaLbPlusGol0929/eval_goal_20261001_093920.tar.gz (16M)
复制到: ~/b/Ckp/eval_goal_20261001_093920.tar.gz
Results JSON: /B/Log/4dwvlaLbPlusGol0929/20261001_093920_eval/overall_results.json

bigmatrix_multiply_optimization.py 已启动 (PID 3508516)
```

