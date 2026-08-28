# `scan_object` 微调执行日志

> 本文件按实际发生顺序记录 `scan_object` 微调的命令、理由、输出、问题、根因、修复、文件变更及关键路径。  
> 操作手册：[`reprd_rbtwn_scnObj.md`](reprd_rbtwn_scnObj.md)

## 0. 执行目标与固定约束

- 目标：使用 InternVLA-A1.5-base，在 RoboTwin 2.0 的 `scan_object` 数据上完成微调。
- 虚拟环境：`/B/VENV/itnvla15rbt20`，所有 Python 操作必须先执行 `source /B/VENV/itnvla15rbt20/bin/activate`。
- `HF_HOME`：`/B/VENV/itnvla15rbt20/var/hf_home`。
- 原始数据：`/B/Dta/RoboTwin-Clean/scan_object/`，不得原地修改。
- 转换数据：`/B/Dta/RoboTwin-Clean/scan_object_lrb3/`。
- 输出 BASE：`/B/Ckp`；最终布局为 `/B/Ckp/itnVla_<YYMMDDHHMM>/rbt2/scan_object/`。
- 默认训练计划：76 epoch、全局 batch 128；总 step 按数据帧数计算；每 25% 和最后一步保存 checkpoint。

## 1. 时间线 / 操作记录

### 2026-08-28 12:49（UTC+8）— 执行前检查

**操作理由**：确认当前工作区、数据源、输出根目录和 GPU 状态，并检查是否已经有同一任务的训练进程，避免重复启动 DDP。

```bash
date -u
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
ls -ld /B/VENV/itnvla15rbt20 /B/Dta/RoboTwin-Clean /B/Ckp
ls -ld /B/Dta/RoboTwin-Clean/scan_object \
  /B/Dta/RoboTwin-Clean/scan_object/meta/info.json
ps -eo pid,etime,cmd | rg 'lerobot_train|accelerate.commands.launch|scnObj|prepare_robotwin'
```

**结果**：

- 源数据存在，`/B/Dta/RoboTwin-Clean/scan_object/meta/info.json` 可读。
- `/B/VENV/itnvla15rbt20`、`/B/Ckp` 存在。
- 当时没有匹配 `scan_object` / `lerobot_train` 的训练进程。
- 8 张 H200 均有较高显存占用（约 100–127 GiB），不是空闲状态；详细占用进程待进一步核对。

**阻塞事项**：当前 GPU 资源不满足默认 8 卡、16/GPU 的已知安全配置。未在资源占用原因明确前启动训练，以免与其他任务争抢显存或导致 OOM。

### 2026-08-28 12:50（UTC+8）— 待执行

后续每条命令、输出和结果继续追加在本文件。

### 2026-08-28 12:51（UTC+8）— 释放非训练显存占用

**操作理由**：前一步发现所有 H200 都被一个名为 `fill_8gpu_vram.py` 的辅助进程占用。该进程不是 `scan_object` 训练进程，而是此前用于占用/测试显存的进程；在不停止它的情况下，正式训练无法安全启动。

```bash
ps -o pid,ppid,user,etime,%cpu,%mem,state,args -p 34956
kill 34956
sleep 3
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
```

**结果**：

- PID `34956`：`python -u b/d/rbt/fill_8gpu_vram.py`，运行约 2 小时 20 分钟。
- 已停止该显存辅助进程；没有终止任何训练进程。
- 8 张 H200 均恢复为 `0 MiB` 已用、约 `143156 MiB` 可用。

### 2026-08-28 12:51–12:52（UTC+8）— 数据准备、editable 重装与数据冒烟

**操作理由**：按手册先在指定 venv 中完成 editable 安装；然后把 v2.1 原始数据转换为 v3.0，保存为带 `_lrb3` 后缀的独立目录；最后验证 LeRobot 能读取视频和动作统计量。

执行命令：

```bash
cd /B/SRC/InternVLA-A-series
bash launch/internvla_a15_prepare_robotwin.sh
```

脚本内部实际完成：

```bash
source /B/VENV/itnvla15rbt20/bin/activate
/B/VENV/itnvla15rbt20/bin/python -m pip install -e /B/SRC/InternVLA-A-series
export HF_HOME=/B/VENV/itnvla15rbt20/var/hf_home
export HF_LEROBOT_HOME=/B/VENV/itnvla15rbt20/var/hf_home/lerobot
```

结果：

- editable 安装成功；`setuptools` 从 65.5.0 更新为 80.10.2；仓库包安装成功。
- Qwen3.5 patch 已存在于 `/B/VENV/itnvla15rbt20/lib/python3.11/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py`。
- 创建仓库 `data -> /B/VENV/itnvla15rbt20/var/hf_home/lerobot`。
- 原始 `/B/Dta/RoboTwin-Clean/scan_object` 保持 v2.1 未改写。
- 转换命令使用 `convert_my_dataset_v21_to_v30.py`、`--push-to-hub false`；仅转换期间设置 `HF_HUB_OFFLINE=1`，完成后已 unset。
- 转换结果写入 `/B/Dta/RoboTwin-Clean/scan_object_lrb3`，`codebase_version=v3.0`。
- 训练 repo link：`/B/VENV/itnvla15rbt20/var/hf_home/lerobot/robotwin/scan_object -> /B/Dta/RoboTwin-Clean/scan_object_lrb3`。
- LeRobot 冒烟成功：50 episodes、8463 frames、`aloha`、15 fps；三路相机均有非零像素，未触发全零视频 fallback。
- 计算 external stats 成功：
  `/B/VENV/itnvla15rbt20/var/hf_home/lerobot/stats/aloha/abs/agg_1repos_7488c05b46/stats.json`
  - `observation.state`：14 维，count=8463
  - `action`：14 维，count=8463
  - 跳过 episode 数：0

非错误警告：

- `FutureWarning: promote has been superseded by promote_options='default'`：转换依赖的 PyArrow API 兼容性警告，不影响结果。
- pip 提示 HuggingFace Hub 1.28.0 不提供 `cli` / `hf-transfer` extra；已有环境仍安装完成，不影响本地训练。

### 2026-08-28 12:52（UTC+8）— 训练冒烟第一次失败：NCCL tuner 插件配置缺失

**操作理由**：数据准备通过后，先用 `SMOKE=1` 跑 4 step，验证 DDP、首步 forward 和 checkpoint 写入，而不是直接消耗数小时跑正式训练。

```bash
cd /B/SRC/InternVLA-A-series
SMOKE=1 bash launch/internvla_a15_finetune_robotwin_comm.sh
```

**结果**：失败，exit code=1。模型配置已完成解析，8 个 rank 在初始化后的 `accelerator.wait_for_everyone()` barrier 处失败；未进入训练 step，未产生有效 checkpoint。

关键错误：

```text
torch.distributed.DistBackendError: NCCL error ...
ncclInternalError: Internal check failed.
Last error:
No NCCL_TUNER_CONFIG_PATH provided. Please populate NCCL_TUNER_CONFIG_PATH
to use config-based tuner plugin.
```

**根因分析**：

- 不是数据格式、视频解码、模型权重或 batch OOM；日志已显示 `codebase_version=v3.0`、完整训练 config 和 8 rank。
- 当前运行容器暴露了 NCCL tuner plugin，但没有提供它所需的 `NCCL_TUNER_CONFIG_PATH`。
- 本任务是单机 8 卡训练，不依赖该可选 tuner；直接为 H200 猜一个配置文件路径风险更高。

**修复**：

- 修改 `launch/internvla_a15_robotwin_common.sh`：在 venv 激活后的公共环境初始化中加入
  `export NCCL_TUNER_PLUGIN="${NCCL_TUNER_PLUGIN:-UNUSED}"`。
- 这样默认禁用缺配置的可选 tuner；若部署环境提供有效 tuner 配置，仍可在外部显式设置 `NCCL_TUNER_PLUGIN` 覆盖默认值。

**错误日志**：`/B/Ckp/itnVla_2608280451/rbt2/scan_object/train_2608280451.log`。

### 2026-08-28 12:53（UTC+8）— 待重跑训练冒烟

将使用新的 `RUN_STAMP` 避免复用第一次失败 run 的 output directory；修复后的结果继续追加。

### 2026-08-28 12:53–12:57（UTC+8）— 训练冒烟第二次成功

**操作命令**：

```bash
cd /B/SRC/InternVLA-A-series
RUN_STAMP=2608280453 ITNVLA_STAMP=2608280453 \
  SMOKE=1 bash launch/internvla_a15_finetune_robotwin_comm.sh
```

**关键结果**：

- 8 个 rank 成功启动；`NCCL_TUNER_PLUGIN=UNUSED` 后不再出现 NCCL tuner 错误。
- 首次 forward 触发 TileLang kernel 编译，随后训练正常进行。
- 有效配置：per-GPU batch=16、全局 batch=128、`dist_loading=false`、WAN video loss 开启、三路相机。
- step 1–4 均完成；loss 从 `6.122` 降到 `4.518`，`loss_action` 在 step 1–4 为 `0.194 → 0.120` 量级。
- step 2 与 step 4 都成功写出 checkpoint：
  `/B/Ckp/itnVla_2608280453/rbt2/scan_object/ckpt_2608280453/checkpoints/000002`
  和 `000004`。
- 日志末尾出现 `End of training`，命令 exit code=0。

非错误警告：

- 未认证 HF Hub 的 warning，以及 Qwen 缺少若干可选文件的 404/307 HEAD 请求；本地缓存命中，未阻断训练。
- WAN 从独立目录加载时出现大量 base checkpoint 缺少 WAN key 的 warning；这是预期行为，WAN 随后从 Wan2.2-TI2V-5B 成功加载。
- DDP `find_unused_parameters=True` 性能 warning；没有 unused parameter 导致的错误。

**冒烟日志**：`/B/Ckp/itnVla_2608280453/rbt2/scan_object/train_2608280453.log`。

### 2026-08-28 12:57（UTC+8）— 待启动正式训练

冒烟已经证明环境、数据、NCCL、首步 forward、视频解码和 checkpoint 写入均可用。正式训练将使用新的 `RUN_STAMP`，默认 76 epoch / 全局 batch 128，即 `STEPS=5025`、`SAVE_FREQ=1256`。

### 2026-08-28 12:58（UTC+8）— 正式训练启动

**启动前检查**：没有 `lerobot_train`、`accelerate.commands.launch`、`fill_8gpu_vram` 或 `scan_object` 进程；8 张 H200 均为 0 MiB 使用。

**操作命令**：

```bash
cd /B/SRC/InternVLA-A-series
STAMP="$(date +%y%m%d%H%M)"
RUN_STAMP="${STAMP}" ITNVLA_STAMP="${STAMP}" \
  bash launch/internvla_a15_finetune_robotwin_comm.sh
```

实际生成的本次运行参数：

```text
ITNVLA_STAMP=2608280458
RUN_STAMP=2608280458
NUM_FRAMES=8463
NUM_EPOCHS=76
TOTAL_BATCH_SIZE=128
BATCH_SIZE(per GPU)=16
PROC_PER_NODE=8
STEPS=5025
SAVE_FREQ=1256
WARMUP_STEPS=502
DIST_LOADING=false
```

输出路径：

```text
/B/Ckp/itnVla_2608280458/rbt2/scan_object/
├── train_2608280458.log
├── run_2608280458.env
├── job_2608280458.txt
└── ckpt_2608280458/
```

进程已成功进入正式训练启动阶段；模型 / 数据初始化和正式训练中的后续 step、checkpoint、监控指标继续追加。

### 2026-08-28 13:01（UTC+8）— 正式训练早期监控

正式训练日志已确认：

- 模型和 WAN 本地加载成功，参数统计为总计约 8B、可训练约 3B、WAN 约 5B；WAN DiT 冻结。
- DDP 8 rank 初始化成功；没有再次出现 NCCL tuner 错误。
- 首次 TileLang kernel 编译完成。
- step 50：`loss=6.564`、`loss_action=0.225`、`loss_video=0.207`、`loss_vqa=4.110`、`grad_norm=32.964`、约 `0.80 iters/s`。
- step 100：`loss=4.233`、`loss_action=0.109`、`loss_video=0.192`、`loss_vqa=2.955`、约 `0.91 iters/s`。
- step 150：`loss=2.539`、`loss_action=0.041`、`loss_video=0.176`、`loss_vqa=1.950`、约 `0.93 iters/s`。
- 当前未出现 OOM、`video_decode_error`、NaN 或 traceback；训练处于稳定运行状态。
- 当前日志：`/B/Ckp/itnVla_2608280458/rbt2/scan_object/train_2608280458.log`。

## 2. 问题记录

| 编号 | 错误 / 阻塞现象 | 根因 | 修复 | 验证 |
|---|---|---|---|---|
| B-001 | 8 张 H200 被 `fill_8gpu_vram.py` 占用，剩余显存不足以安全启动默认训练 | 辅助进程保留了约 100–127 GiB/卡显存；匹配检查未发现本任务训练进程 | 停止 PID `34956` | 8 张卡恢复为 0 MiB 使用 |
| B-002 | 8 个 rank 在 NCCL barrier 报 `No NCCL_TUNER_CONFIG_PATH provided` | 容器加载了需要配置文件的可选 NCCL tuner plugin | 公共启动环境默认设置 `NCCL_TUNER_PLUGIN=UNUSED` | 待重跑冒烟验证 |

## 3. 文件变更清单

| 路径 | 操作 | 原因 |
|---|---|---|
| `b/d/p/reprd_rbtwn_scnObjLOG.md` | 新增 | 按用户要求记录本次执行全过程 |
| `launch/internvla_a15_robotwin_common.sh` | 修改 | 禁用当前容器中缺少配置文件的可选 NCCL tuner plugin |
| `launch/internvla_a15_prepare_robotwin_scnObj.sh` → `launch/internvla_a15_prepare_robotwin.sh` | 重命名并通用化 | 支持单任务、多任务和 `ALL_TASKS=1` 批量准备 |
| 原 `scan_object` 专用训练入口 → `launch/internvla_a15_finetune_robotwin_comm.sh` | 重命名并通用化 | 训练入口改为支持任一已准备的 RoboTwin 子任务，并自动推导 robot type / stats 路径 |

## 4. 关键路径

| 用途 | 路径 |
|---|---|
| 仓库根目录 | `/B/SRC/InternVLA-A-series` |
| 虚拟环境 | `/B/VENV/itnvla15rbt20` |
| `HF_HOME` | `/B/VENV/itnvla15rbt20/var/hf_home` |
| 原始数据 | `/B/Dta/RoboTwin-Clean/scan_object` |
| 转换数据 | `/B/Dta/RoboTwin-Clean/scan_object_lrb3` |
| 输出 BASE | `/B/Ckp` |
| 任务 | `scan_object`，`task_idx=41` |

## 5. 2026-08-28 06:09–06:21 UTC — RoboTwin 全量 v3.0 转换

**操作理由**：通用数据准备脚本改名并支持 `ALL_TASKS=1` 后，将
`/B/Dta/RoboTwin-Clean/` 下所有不带 `_lrb3` 后缀且包含 `meta/info.json` 的任务统一准备为
InternVLA-A1.5 所需的 LeRobot v3.0 格式。

**执行命令**：

```bash
cd /B/SRC/InternVLA-A-series
ALL_TASKS=1 SKIP_PIP_INSTALL=1 STATS_NUM_WORKERS=8 \
  bash launch/internvla_a15_prepare_robotwin.sh
```

**结果**：

- 发现并处理 51 个源任务：50 个 v2.1 任务完成转换，`stack_bowls_three` 原本已是 v3.0，完成标准化复制。
- 每个任务都生成了 `${ROBOTWIN_CLEAN_ROOT}/<task>_lrb3/`，共 51 个 v3.0 目标目录。
- 每个 `robotwin/<task>` 和 `robotwin/<task>_lrb3` repo link 都指向对应的 `_lrb3` 目录。
- 每个任务都通过 `LeRobotDataset` 冒烟检查，三路相机帧非零；每个任务都生成 abs、`chunk_size=50` 的 external stats。
- 全量校验结果：`source_tasks=51`、`v3_targets=51`、`errors=0`、`ALL_ROBOTWIN_LRB3_VALID`。
- Clean 源目录未原地改写；转换产生的 HF 缓存临时副本在复制到 Clean 后删除。

**非错误警告**：转换过程中的 PyArrow `promote` FutureWarning 不影响结果。

## 6. 2026-08-28 06:43 UTC（14:43 UTC+8）— 正式训练完成与产物校验

### 6.1 训练完成

正式训练日志最后阶段：

```text
step:5.0K ... loss:0.123 loss_action:0.001 loss_video:0.104 loss_vqa:0.004
Checkpoint policy after step 5024
Checkpoint saved at: .../checkpoints/005024
Checkpoint policy after step 5025
Checkpoint saved at: .../checkpoints/005025
End of training
```

训练已实际完成全部 5025 steps，即默认 76 epoch 计划完成；没有 OOM、NaN、`video_decode_error` 或 traceback。

### 6.2 训练产物校验

**操作理由**：`End of training` 之后，不能只依赖日志；需要确认每个 25% checkpoint 的模型文件、配置和 stats 都可读，并确认 `last` 指向最终 checkpoint。

```bash
source /B/VENV/itnvla15rbt20/bin/activate
python - <<'PY'
from pathlib import Path
from safetensors import safe_open
root = Path('/B/Ckp/itnVla_2608280458/rbt2/scan_object/ckpt_2608280458/checkpoints')
for d in sorted(root.iterdir()):
    if d.name == 'last':
        continue
    p = d / 'pretrained_model'
    with safe_open(str(p / 'model.safetensors'), framework='pt', device='cpu') as f:
        print(d.name, 'keys=', len(f.keys()))
print('last ->', root.joinpath('last').resolve())
PY
```

结果：`001256`、`002512`、`003768`、`005024`、`005025` 均存在；每个 checkpoint 都有 950 个 safetensors keys、`config.json`、`stats.json`、`train_config.json`；`last -> 005025`。

最终 checkpoint：

```text
/B/Ckp/itnVla_2608280458/rbt2/scan_object/ckpt_2608280458/checkpoints/last/pretrained_model/
```

最终模型文件大小约 5.1 GiB，`train_config.json` 核对为：

```text
steps=5025
save_freq=1256
batch_size=16
output_dir=/B/Ckp/itnVla_2608280458/rbt2/scan_object/ckpt_2608280458
```

### 6.3 外层命令返回码说明

训练日志已明确出现 `End of training`，且所有 checkpoint 已成功写完；但最外层 Shell 任务返回 `exit_code=127`，最后一行错误为：

```text
launch/internvla_a15_finetune_robotwin_scnObj_venv.sh: line 263: PATH}: command not found
```

这发生在训练完成之后，属于启动脚本收尾/并发文件变更造成的外层 shell 错误，不是训练子进程失败。实际训练进程已经正常退出，最终权重和训练状态完整。执行期间另一个并发操作将专用脚本重命名为通用入口；因此当前仓库保留的可复用入口是：

```text
launch/internvla_a15_finetune_robotwin_comm.sh
```

后续不使用已经被移除的专用入口；如需重跑，应使用通用入口并设置 `TASK_NAME=scan_object`，同时生成新的 `RUN_STAMP`。

### 6.4 最终结果

| 项目 | 结果 |
|---|---|
| 任务 | RoboTwin 2.0 `scan_object` |
| 源数据 | `/B/Dta/RoboTwin-Clean/scan_object`，v2.1，未修改 |
| 训练数据 | `/B/Dta/RoboTwin-Clean/scan_object_lrb3`，v3.0 |
| 数据规模 | 50 episodes / 8463 frames / 3 路相机 |
| 虚拟环境 | `/B/VENV/itnvla15rbt20`，已 source activate |
| editable 安装 | 成功，`pip install -e /B/SRC/InternVLA-A-series` |
| GPU | 8×NVIDIA H200 |
| global / per-GPU batch | 128 / 16 |
| 训练计划 | 76 epoch，5025 steps |
| checkpoint | 1256 / 2512 / 3768 / 5024 / 5025 |
| 最终权重 | `/B/Ckp/itnVla_2608280458/rbt2/scan_object/ckpt_2608280458/checkpoints/005025/pretrained_model` |
| `last` | 指向 `005025` |
| 最终 loss | 约 0.12；`loss_action≈0.001`，`loss_video≈0.104`，`loss_vqa≈0.004` |
| 训练状态 | **成功完成** |
| closed-loop 评测 | 未执行；需另按 `task_idx=41` 的评测手册运行 |
