# `pick_dual_bottles` 微调训练执行日志

> 对应操作手册：[`reprd_rbtwn_pikDulBtl.md`](reprd_rbtwn_pikDulBtl.md)
>
> 本文件按实际发生顺序记录本次使用 InternVLA-A1.5-base 在 RoboTwin 2.0
> `pick_dual_bottles` 数据上微调的命令、操作理由、输出、错误、根因、修复、
> 文件变更和最终产物。正式训练未完成前，后续记录持续追加到本文档末尾。

## 0. 执行目标与固定约束

- 任务：RoboTwin 2.0 `pick_dual_bottles`。
- 基座：InternVLA-A1.5-base。
- 虚拟环境：`/B/VENV/itnvla15rbt20`，所有 Python 操作先 `source` 激活。
- `HF_HOME`：`/B/VENV/itnvla15rbt20/var/hf_home`。
- 原始数据：`/B/Dta/RoboTwin-Clean/pick_dual_bottles/`，保持只读。
- 训练数据：`/B/Dta/RoboTwin-Clean/pick_dual_bottles_lrb3/`，LeRobot v3.0。
- 默认计划：76 epoch、global batch 128、动作模式 `abs`。
- 当前已核对数据量：50 episodes、6129 frames、ALOHA、15 FPS。
- 预期默认 8 卡训练：per-GPU batch 16、3640 steps、save frequency 910。
- 输出根：`/B/Ckp/itnVla_<ITNVLA_STAMP>/rbt2/pick_dual_bottles/`。
- 不使用 `nohup ... & disown` 启动 DDP；使用前台命令或 tmux。

## 1. 时间线

| 时间（UTC+8） | 操作 | 结果 |
|---|---|---|
| 2026-08-28 15:44–15:45 | 读取操作手册、通用训练/准备脚本；检查已有终端和历史训练进程 | 发现的旧训练记录已结束 |
| 2026-08-28 15:45 | 查询 `nvidia-smi` 和训练进程，避免重复启动 | 8×H200 均 0 MiB，未发现活动训练进程 |
| | 在指定 venv 中 editable 重装、复核数据和 stats | 待执行 |
| | 训练冒烟（4 steps） | 待执行 |
| | 正式训练（默认 3640 steps） | 待执行 |
| | 最终 checkpoint 校验 | 待执行 |

## 2. 操作记录

### 2.1 训练启动前资源检查

**理由**：DDP 会使用全部可见 GPU；必须确认没有其它训练、显存填充或旧的
`accelerate` 进程，避免互相抢占显存。

执行命令：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
ps -eo pid,etime,state,cmd | rg 'fill_8gpu_vram|lerobot_train|accelerate.commands.launch|pick_dual_bottles|scan_object' || true
```

结果：8 张 NVIDIA H200 均为 `0 MiB` used、约 `143156 MiB` free、GPU utilization
为 0%；未发现活动的 `fill_8gpu_vram`、`lerobot_train` 或 `accelerate` 训练进程。

### 2.2 venv editable 重装、数据准备和 stats

**理由**：按照手册先在指定 venv 中重新安装当前 checkout，确认当前仓库代码生效，
然后让通用准备脚本建立训练 repo link、检查数据、解码一帧视频并计算 external stats。

执行命令：

```bash
cd /B/SRC/InternVLA-A-series
source /B/VENV/itnvla15rbt20/bin/activate
export HF_HOME=/B/VENV/itnvla15rbt20/var/hf_home
export HF_LEROBOT_HOME="${HF_HOME}/lerobot"
export TASK_NAME=pick_dual_bottles
bash launch/internvla_a15_prepare_robotwin.sh
```

说明：本次命令没有显式传 `SKIP_CONVERT=1`，因此脚本按默认逻辑重新从源目录
`v2.1` 转换了 `pick_dual_bottles_lrb3`。这不是错误；转换器成功完成，且源目录
没有被原地改写。之后不需要重复转换时应使用：
`TASK_NAME=pick_dual_bottles SKIP_CONVERT=1 ...`。

结果：

- editable 安装成功，当前导入代码来自 `/B/SRC/InternVLA-A-series/src/lerobot`；
- Qwen3.5 patch 已存在；
- 源数据确认 `v2.1`、`aloha`；
- v3.0 转换成功，50 episodes、6129 frames；
- 训练 link：
  `/B/VENV/itnvla15rbt20/var/hf_home/lerobot/robotwin/pick_dual_bottles`
  → `/B/Dta/RoboTwin-Clean/pick_dual_bottles_lrb3`；
- 三路视频均解码成功且像素非零，输出 `SMOKE_DATASET_OK`；
- external stats 成功：
  `/B/VENV/itnvla15rbt20/var/hf_home/lerobot/stats/aloha/abs/agg_1repos_59c5e8f4cd/stats.json`；
- `observation.state` 和 `action` 均为 14 维、count 为 6129；
- 50 个 episode 均参与 stats，跳过数为 0。

非错误警告：`FutureWarning: promote has been superseded ...` 为 PyArrow API
兼容性警告，不影响转换结果；pip 关于 HuggingFace Hub extras 的 warning 也未阻断安装。

### 2.3 训练冒烟

**理由**：正式训练前先用完全相同的 DDP、模型、WAN、三路视频和 batch 配置运行
4 steps，验证首个 forward、显存、NCCL 和 checkpoint 写入。

执行命令：

```bash
cd /B/SRC/InternVLA-A-series
source /B/VENV/itnvla15rbt20/bin/activate
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
STAMP="$(date +%y%m%d%H%M)"
TASK_NAME=pick_dual_bottles RUN_STAMP="${STAMP}" ITNVLA_STAMP="${STAMP}" \
SMOKE=1 bash launch/internvla_a15_finetune_robotwin_comm.sh
```

本次冒烟时间戳为 `2608280746`，有效配置：

```text
GPU=8×NVIDIA H200
PROC_PER_NODE=8
TOTAL_BATCH_SIZE=128
BATCH_SIZE(per GPU)=16
DIST_LOADING=false
STEPS=4
SAVE_FREQ=2
```

结果：

- 8 个 rank 启动成功；
- `python -m accelerate.commands.launch` 工作正常；
- 模型总参数约 8B、可训练参数约 3B、WAN 参数约 5B，WAN DiT 冻结；
- step 1 到 4 均完成，loss 为 `7.084 → 5.135 → 6.483 → 4.861`，
  `loss_action` 约为 `0.256 → 0.145 → 0.234 → 0.136`；
- step 2 checkpoint：
  `/B/Ckp/itnVla_2608280746/rbt2/pick_dual_bottles/ckpt_2608280746/checkpoints/000002`；
- step 4 checkpoint：
  `/B/Ckp/itnVla_2608280746/rbt2/pick_dual_bottles/ckpt_2608280746/checkpoints/000004`；
- 日志出现 `End of training`，外层 exit code 为 0；
- 未出现 OOM、NaN、视频解码错误或 NCCL tuner 错误。

已知非错误 warning：

- HF Hub 未认证 warning；
- DDP `find_unused_parameters=True` 的性能 warning；
- WAN 从独立权重加载时 base checkpoint 缺少 WAN key 的 warning。

冒烟输出仅用于验证，正式训练将使用新的 `RUN_STAMP`。

### 2.4 正式训练启动

**理由**：冒烟已验证环境、数据、首个 forward、DDP 和 checkpoint 写入；使用新的
时间戳开始正式训练，避免覆盖冒烟结果。

执行命令：

```bash
cd /B/SRC/InternVLA-A-series
source /B/VENV/itnvla15rbt20/bin/activate
export HF_HOME=/B/VENV/itnvla15rbt20/var/hf_home
export HF_LEROBOT_HOME="${HF_HOME}/lerobot"
unset CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
STAMP="$(date +%y%m%d%H%M)"
TASK_NAME=pick_dual_bottles RUN_STAMP="${STAMP}" ITNVLA_STAMP="${STAMP}" \
NUM_EPOCHS=76 TOTAL_BATCH_SIZE=128 \
bash launch/internvla_a15_finetune_robotwin_comm.sh
```

启动时实际参数：

```text
ITNVLA_STAMP=2608280751
RUN_STAMP=2608280751
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PROC_PER_NODE=8
NUM_FRAMES=6129
NUM_EPOCHS=76
TOTAL_BATCH_SIZE=128
BATCH_SIZE(per GPU)=16
STEPS=3640
SAVE_FREQ=910
WARMUP_STEPS=364
DIST_LOADING=false
```

输出路径：

```text
/B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/
├── train_2608280751.log
├── job_2608280751.txt
├── run_2608280751.env
└── ckpt_2608280751/
```

正式训练已通过配置解析、数据版本预检和模型启动，当前正在运行。

### 2.5 正式训练早期监控

截至 08:01 UTC，训练已推进到 step 350，日志显示：

```text
step 50:  loss=6.137 loss_action=0.199 loss_video=0.242 loss_vqa=3.908 grad_norm=25.588
step 100: loss=3.381 loss_action=0.069 loss_video=0.204 loss_vqa=2.483 grad_norm=10.461
step 150: loss=2.076 loss_action=0.028 loss_video=0.175 loss_vqa=1.623 grad_norm=11.482
step 200: loss=1.580 loss_action=0.017 loss_video=0.163 loss_vqa=1.246 grad_norm=14.885
step 250: loss=1.312 loss_action=0.013 loss_video=0.151 loss_vqa=1.031 grad_norm=16.444
step 300: loss=1.164 loss_action=0.011 loss_video=0.148 loss_vqa=0.908 grad_norm=16.824
step 350: loss=1.126 loss_action=0.011 loss_video=0.146 loss_vqa=0.867 grad_norm=15.860
```

当前吞吐约 `0.90–0.93 iters/s`，日志 ETA 约 1 小时；尚未出现 OOM、NaN、
`video_decode_error` 或 traceback。训练继续运行。

2026-08-28T07:45:44Z
PROJ_ROOT            = /B/SRC/InternVLA-A-series
VENV_ROOT            = /B/VENV/itnvla15rbt20
HF_HOME              = /B/VENV/itnvla15rbt20/var/hf_home
HF_LEROBOT_HOME      = /B/VENV/itnvla15rbt20/var/hf_home/lerobot
ROBOTWIN_CLEAN_ROOT  = /B/Dta/RoboTwin-Clean
CKPT_BASE            = /B/Ckp
TASK_NAME            = pick_dual_bottles
which python         = /B/VENV/itnvla15rbt20/bin/python
Selected tasks (1): pick_dual_bottles
===== pip install -e . into /B/VENV/itnvla15rbt20 =====
Obtaining file:///B/SRC/InternVLA-A-series
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Checking if build backend supports build_editable: started
  Checking if build backend supports build_editable: finished with status 'done'
  Getting requirements to build editable: started
  Getting requirements to build editable: finished with status 'done'
  Preparing editable metadata (pyproject.toml): started
  Preparing editable metadata (pyproject.toml): finished with status 'done'
Requirement already satisfied: datasets<4.2.0,>=4.0.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (4.1.1)
Requirement already satisfied: diffusers<0.36.0,>=0.27.2 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (0.35.2)
Requirement already satisfied: huggingface-hub>=0.34.2 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from huggingface-hub[cli,hf-transfer]>=0.34.2->internvla-a1-5==1.0.0) (1.28.0)
Requirement already satisfied: accelerate<2.0.0,>=1.10.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (1.14.0)
Requirement already satisfied: setuptools<81.0.0,>=71.0.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (80.10.2)
Requirement already satisfied: einops<0.9.0,>=0.8.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (0.8.2)
Requirement already satisfied: opencv-python-headless<4.13.0,>=4.9.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (4.12.0.88)
Requirement already satisfied: av<16.0.0,>=15.0.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (15.1.0)
Requirement already satisfied: jsonlines<5.0.0,>=4.0.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (4.0.0)
Requirement already satisfied: msgpack<2.0.0,>=1.0.7 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (1.2.1)
Requirement already satisfied: packaging<26.0,>=24.2 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (25.0)
Requirement already satisfied: torch>=2.2.1 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (2.10.0+cu128)
Requirement already satisfied: torchvision>=0.21.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (0.25.0+cu128)
Requirement already satisfied: draccus<0.11.0,>=0.10.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (0.10.0)
Requirement already satisfied: omegaconf<3.0.0,>=2.3.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (2.3.1)
Requirement already satisfied: loguru<0.8.0,>=0.7.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (0.7.3)
Requirement already satisfied: numpy<2.3.0,>=1.26.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (2.2.6)
Requirement already satisfied: scipy<1.16.0,>=1.12.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (1.15.3)
Requirement already satisfied: wandb<0.22.0,>=0.20.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (0.21.4)
Requirement already satisfied: torchcodec>=0.2.1 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (0.10.0+cu128)
Requirement already satisfied: imageio<3.0.0,>=2.34.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from imageio[ffmpeg]<3.0.0,>=2.34.0->internvla-a1-5==1.0.0) (2.37.4)
Requirement already satisfied: mediapy<2.0.0,>=1.2.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (1.2.7)
Requirement already satisfied: deepdiff<9.0.0,>=7.0.1 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (8.6.2)
Requirement already satisfied: termcolor<4.0.0,>=2.4.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (3.3.0)
Requirement already satisfied: tqdm<5.0.0,>=4.66.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from internvla-a1-5==1.0.0) (4.70.0)
Requirement already satisfied: psutil in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from accelerate<2.0.0,>=1.10.0->internvla-a1-5==1.0.0) (7.2.2)
Requirement already satisfied: pyyaml in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from accelerate<2.0.0,>=1.10.0->internvla-a1-5==1.0.0) (6.0.3)
Requirement already satisfied: safetensors>=0.4.3 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from accelerate<2.0.0,>=1.10.0->internvla-a1-5==1.0.0) (0.8.0)
Requirement already satisfied: filelock in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from datasets<4.2.0,>=4.0.0->internvla-a1-5==1.0.0) (3.29.0)
Requirement already satisfied: pyarrow>=21.0.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from datasets<4.2.0,>=4.0.0->internvla-a1-5==1.0.0) (25.0.0)
Requirement already satisfied: dill<0.4.1,>=0.3.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from datasets<4.2.0,>=4.0.0->internvla-a1-5==1.0.0) (0.4.0)
Requirement already satisfied: pandas in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from datasets<4.2.0,>=4.0.0->internvla-a1-5==1.0.0) (3.0.5)
Requirement already satisfied: requests>=2.32.2 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from datasets<4.2.0,>=4.0.0->internvla-a1-5==1.0.0) (2.34.2)
Requirement already satisfied: xxhash in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from datasets<4.2.0,>=4.0.0->internvla-a1-5==1.0.0) (3.8.1)
Requirement already satisfied: multiprocess<0.70.17 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from datasets<4.2.0,>=4.0.0->internvla-a1-5==1.0.0) (0.70.16)
Requirement already satisfied: fsspec<=2025.9.0,>=2023.1.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from fsspec[http]<=2025.9.0,>=2023.1.0->datasets<4.2.0,>=4.0.0->internvla-a1-5==1.0.0) (2025.9.0)
Requirement already satisfied: orderly-set<6,>=5.4.1 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from deepdiff<9.0.0,>=7.0.1->internvla-a1-5==1.0.0) (5.5.0)
Requirement already satisfied: importlib_metadata in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from diffusers<0.36.0,>=0.27.2->internvla-a1-5==1.0.0) (9.0.0)
Requirement already satisfied: regex!=2019.12.17 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from diffusers<0.36.0,>=0.27.2->internvla-a1-5==1.0.0) (2026.7.19)
Requirement already satisfied: Pillow in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from diffusers<0.36.0,>=0.27.2->internvla-a1-5==1.0.0) (12.2.0)
Requirement already satisfied: mergedeep~=1.3 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from draccus<0.11.0,>=0.10.0->internvla-a1-5==1.0.0) (1.3.4)
Requirement already satisfied: pyyaml-include~=1.4 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from draccus<0.11.0,>=0.10.0->internvla-a1-5==1.0.0) (1.4.1)
Requirement already satisfied: toml~=0.10 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from draccus<0.11.0,>=0.10.0->internvla-a1-5==1.0.0) (0.10.2)
Requirement already satisfied: typing-inspect~=0.9.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from draccus<0.11.0,>=0.10.0->internvla-a1-5==1.0.0) (0.9.0)
Requirement already satisfied: click<9.0.0,>=8.4.2 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from huggingface-hub>=0.34.2->huggingface-hub[cli,hf-transfer]>=0.34.2->internvla-a1-5==1.0.0) (8.4.2)
Requirement already satisfied: hf-xet<2.0.0,>=1.5.2 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from huggingface-hub>=0.34.2->huggingface-hub[cli,hf-transfer]>=0.34.2->internvla-a1-5==1.0.0) (1.5.2)
Requirement already satisfied: httpx<1,>=0.23.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from huggingface-hub>=0.34.2->huggingface-hub[cli,hf-transfer]>=0.34.2->internvla-a1-5==1.0.0) (0.28.1)
Requirement already satisfied: typing-extensions>=4.1.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from huggingface-hub>=0.34.2->huggingface-hub[cli,hf-transfer]>=0.34.2->internvla-a1-5==1.0.0) (4.15.0)
WARNING: huggingface-hub 1.28.0 does not provide the extra 'cli'
WARNING: huggingface-hub 1.28.0 does not provide the extra 'hf-transfer'
Requirement already satisfied: imageio-ffmpeg in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from imageio[ffmpeg]<3.0.0,>=2.34.0->internvla-a1-5==1.0.0) (0.6.0)
Requirement already satisfied: attrs>=19.2.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from jsonlines<5.0.0,>=4.0.0->internvla-a1-5==1.0.0) (26.1.0)
Requirement already satisfied: ipython in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (9.16.1)
Requirement already satisfied: matplotlib in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (3.11.1)
Requirement already satisfied: antlr4-python3-runtime==4.9.* in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from omegaconf<3.0.0,>=2.3.0->internvla-a1-5==1.0.0) (4.9.3)
Requirement already satisfied: sympy>=1.13.3 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (3.6.1)
Requirement already satisfied: jinja2 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (3.1.6)
Requirement already satisfied: cuda-bindings==12.9.4 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (12.9.4)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.8.93 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (12.8.93)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.8.90 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (12.8.90)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.8.90 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (12.8.90)
Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (9.10.2.21)
Requirement already satisfied: nvidia-cublas-cu12==12.8.4.1 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (12.8.4.1)
Requirement already satisfied: nvidia-cufft-cu12==11.3.3.83 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (11.3.3.83)
Requirement already satisfied: nvidia-curand-cu12==10.3.9.90 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (10.3.9.90)
Requirement already satisfied: nvidia-cusolver-cu12==11.7.3.90 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (11.7.3.90)
Requirement already satisfied: nvidia-cusparse-cu12==12.5.8.93 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (12.5.8.93)
Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (0.7.1)
Requirement already satisfied: nvidia-nccl-cu12==2.27.5 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (2.27.5)
Requirement already satisfied: nvidia-nvshmem-cu12==3.4.5 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (3.4.5)
Requirement already satisfied: nvidia-nvtx-cu12==12.8.90 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (12.8.90)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.8.93 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (12.8.93)
Requirement already satisfied: nvidia-cufile-cu12==1.13.1.3 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (1.13.1.3)
Requirement already satisfied: triton==3.6.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from torch>=2.2.1->internvla-a1-5==1.0.0) (3.6.0)
Requirement already satisfied: cuda-pathfinder~=1.1 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from cuda-bindings==12.9.4->torch>=2.2.1->internvla-a1-5==1.0.0) (1.2.2)
Requirement already satisfied: gitpython!=3.1.29,>=1.0.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from wandb<0.22.0,>=0.20.0->internvla-a1-5==1.0.0) (3.1.57)
Requirement already satisfied: platformdirs in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from wandb<0.22.0,>=0.20.0->internvla-a1-5==1.0.0) (4.11.0)
Requirement already satisfied: protobuf!=4.21.0,!=5.28.0,<7,>=3.19.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from wandb<0.22.0,>=0.20.0->internvla-a1-5==1.0.0) (6.33.6)
Requirement already satisfied: pydantic<3 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from wandb<0.22.0,>=0.20.0->internvla-a1-5==1.0.0) (2.13.4)
Requirement already satisfied: sentry-sdk>=2.0.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from wandb<0.22.0,>=0.20.0->internvla-a1-5==1.0.0) (2.66.1)
Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from fsspec[http]<=2025.9.0,>=2023.1.0->datasets<4.2.0,>=4.0.0->internvla-a1-5==1.0.0) (3.14.3)
Requirement already satisfied: gitdb<5,>=4.0.1 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from gitpython!=3.1.29,>=1.0.0->wandb<0.22.0,>=0.20.0->internvla-a1-5==1.0.0) (4.0.12)
Requirement already satisfied: anyio in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from httpx<1,>=0.23.0->huggingface-hub>=0.34.2->huggingface-hub[cli,hf-transfer]>=0.34.2->internvla-a1-5==1.0.0) (4.14.2)
Requirement already satisfied: certifi in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from httpx<1,>=0.23.0->huggingface-hub>=0.34.2->huggingface-hub[cli,hf-transfer]>=0.34.2->internvla-a1-5==1.0.0) (2026.7.22)
Requirement already satisfied: httpcore==1.* in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from httpx<1,>=0.23.0->huggingface-hub>=0.34.2->huggingface-hub[cli,hf-transfer]>=0.34.2->internvla-a1-5==1.0.0) (1.0.9)
Requirement already satisfied: idna in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from httpx<1,>=0.23.0->huggingface-hub>=0.34.2->huggingface-hub[cli,hf-transfer]>=0.34.2->internvla-a1-5==1.0.0) (3.18)
Requirement already satisfied: h11>=0.16 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface-hub>=0.34.2->huggingface-hub[cli,hf-transfer]>=0.34.2->internvla-a1-5==1.0.0) (0.16.0)
Requirement already satisfied: annotated-types>=0.6.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from pydantic<3->wandb<0.22.0,>=0.20.0->internvla-a1-5==1.0.0) (0.8.0)
Requirement already satisfied: pydantic-core==2.46.4 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from pydantic<3->wandb<0.22.0,>=0.20.0->internvla-a1-5==1.0.0) (2.46.4)
Requirement already satisfied: typing-inspection>=0.4.2 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from pydantic<3->wandb<0.22.0,>=0.20.0->internvla-a1-5==1.0.0) (0.4.2)
Requirement already satisfied: charset_normalizer<4,>=2 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from requests>=2.32.2->datasets<4.2.0,>=4.0.0->internvla-a1-5==1.0.0) (3.4.9)
Requirement already satisfied: urllib3<3,>=1.26 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from requests>=2.32.2->datasets<4.2.0,>=4.0.0->internvla-a1-5==1.0.0) (2.7.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from sympy>=1.13.3->torch>=2.2.1->internvla-a1-5==1.0.0) (1.3.0)
Requirement already satisfied: mypy-extensions>=0.3.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from typing-inspect~=0.9.0->draccus<0.11.0,>=0.10.0->internvla-a1-5==1.0.0) (1.1.0)
Requirement already satisfied: zipp>=3.20 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from importlib_metadata->diffusers<0.36.0,>=0.27.2->internvla-a1-5==1.0.0) (4.1.0)
Requirement already satisfied: ipython-pygments-lexers>=1.0.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from ipython->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (1.1.1)
Requirement already satisfied: jedi>=0.18.2 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from ipython->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (0.20.0)
Requirement already satisfied: matplotlib-inline>=0.1.6 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from ipython->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (0.2.2)
Requirement already satisfied: pexpect>4.6 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from ipython->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (4.9.0)
Requirement already satisfied: prompt_toolkit<3.1.0,>=3.0.41 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from ipython->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (3.0.53)
Requirement already satisfied: pygments>=2.14.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from ipython->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (2.20.0)
Requirement already satisfied: stack_data>=0.6.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from ipython->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (0.6.3)
Requirement already satisfied: traitlets>=5.13.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from ipython->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (5.16.1)
Requirement already satisfied: MarkupSafe>=2.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from jinja2->torch>=2.2.1->internvla-a1-5==1.0.0) (3.0.3)
Requirement already satisfied: contourpy>=1.0.1 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from matplotlib->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (1.3.3)
Requirement already satisfied: cycler>=0.10 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from matplotlib->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (0.12.1)
Requirement already satisfied: fonttools>=4.28.2 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from matplotlib->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (4.63.0)
Requirement already satisfied: kiwisolver>=1.3.1 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from matplotlib->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (1.5.0)
Requirement already satisfied: pyparsing>=3 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from matplotlib->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (3.3.2)
Requirement already satisfied: python-dateutil>=2.7 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from matplotlib->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (2.9.0.post0)
Requirement already satisfied: aiohappyeyeballs>=2.5.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.9.0,>=2023.1.0->datasets<4.2.0,>=4.0.0->internvla-a1-5==1.0.0) (2.7.1)
Requirement already satisfied: aiosignal>=1.4.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.9.0,>=2023.1.0->datasets<4.2.0,>=4.0.0->internvla-a1-5==1.0.0) (1.4.0)
Requirement already satisfied: frozenlist>=1.1.1 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.9.0,>=2023.1.0->datasets<4.2.0,>=4.0.0->internvla-a1-5==1.0.0) (1.8.0)
Requirement already satisfied: multidict<7.0,>=4.5 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.9.0,>=2023.1.0->datasets<4.2.0,>=4.0.0->internvla-a1-5==1.0.0) (6.7.1)
Requirement already satisfied: propcache>=0.2.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.9.0,>=2023.1.0->datasets<4.2.0,>=4.0.0->internvla-a1-5==1.0.0) (0.5.2)
Requirement already satisfied: yarl<2.0,>=1.17.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2025.9.0,>=2023.1.0->datasets<4.2.0,>=4.0.0->internvla-a1-5==1.0.0) (1.24.5)
Requirement already satisfied: smmap<6,>=3.0.1 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.29,>=1.0.0->wandb<0.22.0,>=0.20.0->internvla-a1-5==1.0.0) (5.0.3)
Requirement already satisfied: parso<0.9.0,>=0.8.6 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from jedi>=0.18.2->ipython->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (0.8.7)
Requirement already satisfied: ptyprocess>=0.5 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from pexpect>4.6->ipython->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (0.7.0)
Requirement already satisfied: wcwidth>=0.1.4 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from prompt_toolkit<3.1.0,>=3.0.41->ipython->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (0.8.2)
Requirement already satisfied: six>=1.5 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from python-dateutil>=2.7->matplotlib->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (1.17.0)
Requirement already satisfied: executing>=1.2.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from stack_data>=0.6.0->ipython->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (2.2.1)
Requirement already satisfied: asttokens>=2.1.0 in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from stack_data>=0.6.0->ipython->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (3.0.2)
Requirement already satisfied: pure-eval in /B/VENV/itnvla15rbt20/lib/python3.11/site-packages (from stack_data>=0.6.0->ipython->mediapy<2.0.0,>=1.2.0->internvla-a1-5==1.0.0) (0.2.3)
Building wheels for collected packages: internvla-a1-5
  Building editable for internvla-a1-5 (pyproject.toml): started
  Building editable for internvla-a1-5 (pyproject.toml): finished with status 'done'
  Created wheel for internvla-a1-5: filename=internvla_a1_5-1.0.0-0.editable-py3-none-any.whl size=5960 sha256=b6a632a76ed5b64fed526ed6570b8168568b74834c4d8f41789eadeaf4619763
  Stored in directory: /tmp/pip-ephem-wheel-cache-6onc2m4y/wheels/00/13/a4/1f5e95c0f8fa5e74f56913ad1136383fe7d0cd2cf813c14d25
Successfully built internvla-a1-5
Installing collected packages: internvla-a1-5
  Attempting uninstall: internvla-a1-5
    Found existing installation: internvla-a1-5 1.0.0
    Not uninstalling internvla-a1-5 at /B/SRC/InternVLA-A-series/src, outside environment /B/VENV/itnvla15rbt20
    Can't uninstall 'internvla-a1-5'. No files were found to uninstall.
Successfully installed internvla-a1-5-1.0.0

[notice] A new release of pip is available: 24.0 -> 26.2.1
[notice] To update, run: pip install --upgrade pip
Transformers Qwen3.5 patch already present: /B/VENV/itnvla15rbt20/lib/python3.11/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py
data -> /B/VENV/itnvla15rbt20/var/hf_home/lerobot

###############################################################################
===== prepare RoboTwin task: pick_dual_bottles =====
source /B/Dta/RoboTwin-Clean/pick_dual_bottles codebase_version=v2.1 robot_type=aloha
===== convert robotwin/pick_dual_bottles (v2.1) -> robotwin/pick_dual_bottles_lrb3 (v3.0) =====
Using local dataset at /B/VENV/itnvla15rbt20/var/hf_home/lerobot/robotwin/pick_dual_bottles
INFO 2026-08-28 07:45:53 1_to_v30.py:439 Converting info from /B/VENV/itnvla15rbt20/var/hf_home/lerobot/robotwin/pick_dual_bottles to /B/VENV/itnvla15rbt20/var/hf_home/lerobot/robotwin/pick_dual_bottles_lrb3
INFO 2026-08-28 07:45:53 1_to_v30.py:162 Converting tasks from /B/VENV/itnvla15rbt20/var/hf_home/lerobot/robotwin/pick_dual_bottles to /B/VENV/itnvla15rbt20/var/hf_home/lerobot/robotwin/pick_dual_bottles_lrb3
INFO 2026-08-28 07:45:53 1_to_v30.py:213 Converting data files from 50 episodes

convert data files:   0%|          | 0/50 [00:00<?, ?it/s]
convert data files: 100%|██████████| 50/50 [00:00<00:00, 8993.32it/s]
/B/SRC/InternVLA-A-series/src/lerobot/datasets/v30/convert_my_dataset_v21_to_v30.py:245: FutureWarning: promote has been superseded by promote_options='default'.
  concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys)
INFO 2026-08-28 07:45:53 1_to_v30.py:265 Converting videos from /B/VENV/itnvla15rbt20/var/hf_home/lerobot/robotwin/pick_dual_bottles to /B/VENV/itnvla15rbt20/var/hf_home/lerobot/robotwin/pick_dual_bottles_lrb3

convert videos of observation.images.cam_high:   0%|          | 0/50 [00:00<?, ?it/s]
convert videos of observation.images.cam_high: 100%|██████████| 50/50 [00:00<00:00, 3333.57it/s]

convert videos of observation.images.cam_left_wrist:   0%|          | 0/50 [00:00<?, ?it/s]
convert videos of observation.images.cam_left_wrist: 100%|██████████| 50/50 [00:00<00:00, 2903.24it/s]

convert videos of observation.images.cam_right_wrist:   0%|          | 0/50 [00:00<?, ?it/s]
convert videos of observation.images.cam_right_wrist: 100%|██████████| 50/50 [00:00<00:00, 2866.10it/s]

convert videos:   0%|          | 0/50 [00:00<?, ?it/s]
convert videos: 100%|██████████| 50/50 [00:00<00:00, 375833.69it/s]
INFO 2026-08-28 07:45:53 1_to_v30.py:406 Converting episodes metadata from /B/VENV/itnvla15rbt20/var/hf_home/lerobot/robotwin/pick_dual_bottles to /B/VENV/itnvla15rbt20/var/hf_home/lerobot/robotwin/pick_dual_bottles_lrb3

Creating parquet from Arrow format:   0%|          | 0/1 [00:00<?, ?ba/s]
Creating parquet from Arrow format: 100%|██████████| 1/1 [00:00<00:00, 481.22ba/s]
converted dataset at /B/Dta/RoboTwin-Clean/pick_dual_bottles_lrb3 codebase_version=v3.0
training symlink: /B/VENV/itnvla15rbt20/var/hf_home/lerobot/robotwin/pick_dual_bottles -> /B/Dta/RoboTwin-Clean/pick_dual_bottles_lrb3
===== smoke: LeRobotDataset('robotwin/pick_dual_bottles') =====
version 3.0
episodes 50 frames 6129
robot aloha fps 15
cameras ['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist']
len 6129
observation.images.cam_high shape (3, 480, 640) min 0.07450980693101883 max 1.0
observation.images.cam_left_wrist shape (3, 480, 640) min 0.12941177189350128 max 1.0
observation.images.cam_right_wrist shape (3, 480, 640) min 0.14509804546833038 max 1.0
task Grab the dark drink bottle with logo with an arm, grab the medium green bottle with capped neck next.
SMOKE_DATASET_OK
===== compute_norm_stats_multi abs chunk_size=50 =====
---------- aggregate stats for 1 datasets ----------
  - robotwin/pick_dual_bottles

Computing per-repo stats:   0%|          | 0/1 [00:00<?, ?it/s]
Computing per-repo stats: 100%|██████████| 1/1 [00:00<00:00, 38836.15it/s]
---------- done ----------
robot_type: aloha
action_mode: abs
chunk_size: 50
group_name: agg_1repos_59c5e8f4cd
output: /B/VENV/itnvla15rbt20/var/hf_home/lerobot/stats/aloha/abs/agg_1repos_59c5e8f4cd/stats.json
total_frames (sum of episode lengths): 6129
total_episodes: 50 (skipped: 0 episodes with len < chunk_size)
EXTERNAL_STATS_PATH=/B/VENV/itnvla15rbt20/var/hf_home/lerobot/stats/aloha/abs/agg_1repos_59c5e8f4cd/stats.json
observation.state: dim=14 count=[6129]
action: dim=14 count=[6129]
timestamp: dim=1 count=[6129]
frame_index: dim=1 count=[6129]
episode_index: dim=1 count=[6129]
index: dim=1 count=[6129]
task_index: dim=1 count=[6129]
SMOKE_STATS_OK

===== prepare done: 1 task(s) =====
  pick_dual_bottles: /B/Dta/RoboTwin-Clean/pick_dual_bottles_lrb3
2026-08-28T07:46:25Z
0, 0 MiB, 143156 MiB
1, 0 MiB, 143156 MiB
2, 0 MiB, 143156 MiB
3, 0 MiB, 143156 MiB
4, 0 MiB, 143156 MiB
5, 0 MiB, 143156 MiB
6, 0 MiB, 143156 MiB
7, 0 MiB, 143156 MiB
 176711       00:00 S /bin/bash -O extglob -c snap=$(command cat <&3) && builtin shopt -s extglob && builtin eval -- "$snap" && { builtin set +u 2>/dev/null || true; builtin eval "${__CURSOR_SANDBOX_ENV_RESTORE:-}" 2>/dev/null; builtin export PWD="$(builtin pwd)"; builtin shopt -s expand_aliases 2>/dev/null; builtin eval "$1" < /dev/null; }; COMMAND_EXIT_CODE=$?; dump_bash_state >&4; builtin exit $COMMAND_EXIT_CODE -- set -o pipefail; { date -u '+%Y-%m-%dT%H:%M:%SZ'; nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader; ps -eo pid,etime,state,cmd | rg 'lerobot_train|accelerate.commands.launch|pick_dual_bottles' || true; STAMP="$(date +%y%m%d%H%M)"; TASK_NAME=pick_dual_bottles RUN_STAMP="${STAMP}" ITNVLA_STAMP="${STAMP}" SMOKE=1 bash launch/internvla_a15_finetune_robotwin_comm.sh; } 2>&1 | tee -a b/d/p/reprd_rbtwn_pikDulBtlLOG.md
 176725       00:00 S /bin/bash -O extglob -c snap=$(command cat <&3) && builtin shopt -s extglob && builtin eval -- "$snap" && { builtin set +u 2>/dev/null || true; builtin eval "${__CURSOR_SANDBOX_ENV_RESTORE:-}" 2>/dev/null; builtin export PWD="$(builtin pwd)"; builtin shopt -s expand_aliases 2>/dev/null; builtin eval "$1" < /dev/null; }; COMMAND_EXIT_CODE=$?; dump_bash_state >&4; builtin exit $COMMAND_EXIT_CODE -- set -o pipefail; { date -u '+%Y-%m-%dT%H:%M:%SZ'; nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader; ps -eo pid,etime,state,cmd | rg 'lerobot_train|accelerate.commands.launch|pick_dual_bottles' || true; STAMP="$(date +%y%m%d%H%M)"; TASK_NAME=pick_dual_bottles RUN_STAMP="${STAMP}" ITNVLA_STAMP="${STAMP}" SMOKE=1 bash launch/internvla_a15_finetune_robotwin_comm.sh; } 2>&1 | tee -a b/d/p/reprd_rbtwn_pikDulBtlLOG.md
 176734       00:00 R rg lerobot_train|accelerate.commands.launch|pick_dual_bottles
MASTER_ADDR=127.0.0.1, MASTER_PORT=36222
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PROC_PER_NODE=8
SMOKE=1: overriding STEPS=4 SAVE_FREQ=2
NUM_FRAMES=6129 NUM_EPOCHS=76 TOTAL_BATCH_SIZE=128
BATCH_SIZE(per GPU)=16 PROC_PER_NODE=8 DIST_LOADING=false
STEPS=4 SAVE_FREQ=2 WARMUP_STEPS=1 LOG_FREQ=1
ckpt steps ~= 2 / 4 / 6 / 4 (last always saved)
OUTPUT_ROOT=/B/Ckp/itnVla_2608280746/rbt2/pick_dual_bottles
OUTPUT_DIR =/B/Ckp/itnVla_2608280746/rbt2/pick_dual_bottles/ckpt_2608280746
LOG_FILE   =/B/Ckp/itnVla_2608280746/rbt2/pick_dual_bottles/train_2608280746.log
DATASET_REPO_ID=robotwin/pick_dual_bottles
ROBOT_TYPE=aloha
EXTERNAL_STATS_PATH=/B/VENV/itnvla15rbt20/var/hf_home/lerobot/stats/aloha/abs/agg_1repos_59c5e8f4cd/stats.json
dataset codebase_version=v3.0 at /B/VENV/itnvla15rbt20/var/hf_home/lerobot/robotwin/pick_dual_bottles/meta/info.json
===== launching training; log -> /B/Ckp/itnVla_2608280746/rbt2/pick_dual_bottles/train_2608280746.log =====
The following values were not passed to `accelerate launch` and had defaults used instead:
	`--mixed_precision` was set to a value of `'no'`
	`--dynamo_backend` was set to a value of `'no'`
To avoid this warning pass in values for each of the problematic parameters or run `accelerate config`.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
INFO 2026-08-28 07:46:53 ot_train.py:173 {'batch_size': 16,
 'checkpoint_path': None,
 'dataset': {'action_mode': 'abs',
             'buffer_size': 1024,
             'chunk_size': 50,
             'data_transforms': {'inputs': [{'height': 224,
                                             'mapping': {},
                                             'mode': 'bilinear',
                                             'type': 'resize_with_pad',
                                             'width': 224},
                                            {'mapping': {},
                                             'type': 'remap_image_key'},
                                            {'normalize_to_minus1_1': True,
                                             'source_view': 'observation.images.image0',
                                             'type': 'extract_video_frames',
                                             'video_key': 'observation.video_frames'},
                                            {'mode': 'mean_std',
                                             'norm_stats': {},
                                             'selected_keys': None,
                                             'type': 'normalize'},
                                            {'mapping': {},
                                             'type': 'compose_fields'},
                                            {'action_token_max': 250124,
                                             'action_token_min': 248077,
                                             'action_tokenizer_name': 'physical-intelligence/fast',
                                             'assistant_end_tokens': [248045,
                                                                      74455,
                                                                      198,
                                                                      248068,
                                                                      271,
                                                                      248069,
                                                                      271],
                                             'chunk_size': 50,
                                             'max_action_dim': 32,
                                             'max_action_tokens': 256,
                                             'qwen35_model_name': 'Qwen/Qwen3.5-2B',
                                             'stop_token_1': 248046,
                                             'stop_token_2': 198,
                                             'type': 'fast_internvla_a1_5_action_tokenizer'},
                                            {'_cache': {},
                                             '_loaded': False,
                                             'annotations_file': '',
                                             'memory_output_key': 'language_memory',
                                             'output_key': 'sub_task',
                                             'type': 'load_action_text_from_jsonl'},
                                            {'action_mode': 'joint',
                                             'action_text_key': 'action.action_text',
                                             'action_token_max': 250124,
                                             'action_token_min': 248077,
                                             'language_memory_key': 'language_memory',
                                             'max_length': 650,
                                             'max_state_dim': 32,
                                             'mode': 'train',
                                             'num_views': 3,
                                             'padding': 'max_length',
                                             'pretrained_model_name_or_path': 'Qwen/Qwen3.5-2B',
                                             'task_key': 'task',
                                             'tokenize_state': True,
                                             'truncation': True,
                                             'type': 'internvla_a1_5_chat_processor',
                                             'use_fast_action_tokens': True},
                                            {'max_action_dim': 32,
                                             'max_state_dim': 32,
                                             'type': 'pad_state_and_action'},
                                            {'action_reorder': None,
                                             'state_reorder': None,
                                             'type': 'reorder_state_action'},
                                            {'num_video_frames': 4,
                                             'type': 'unify_internvla_a1_5_inputs',
                                             'video_height': 224,
                                             'video_width': 224}],
                                 'outputs': []},
             'dist_loading': False,
             'episodes': None,
             'external_stats_path': '/B/VENV/itnvla15rbt20/var/hf_home/lerobot/stats/aloha/abs/agg_1repos_59c5e8f4cd/stats.json',
             'height': 224,
             'image_transforms': {'enable': False,
                                  'max_num_transforms': 3,
                                  'random_order': False,
                                  'tfs': {'affine': {'kwargs': {'degrees': [-5.0,
                                                                            5.0],
                                                                'translate': [0.05,
                                                                              0.05]},
                                                     'type': 'RandomAffine',
                                                     'weight': 1.0},
                                          'brightness': {'kwargs': {'brightness': [0.8,
                                                                                   1.2]},
                                                         'type': 'ColorJitter',
                                                         'weight': 1.0},
                                          'contrast': {'kwargs': {'contrast': [0.8,
                                                                               1.2]},
                                                       'type': 'ColorJitter',
                                                       'weight': 1.0},
                                          'hue': {'kwargs': {'hue': [-0.05,
                                                                     0.05]},
                                                  'type': 'ColorJitter',
                                                  'weight': 1.0},
                                          'saturation': {'kwargs': {'saturation': [0.5,
                                                                                   1.5]},
                                                         'type': 'ColorJitter',
                                                         'weight': 1.0},
                                          'sharpness': {'kwargs': {'sharpness': [0.5,
                                                                                 1.5]},
                                                        'type': 'SharpnessJitter',
                                                        'weight': 1.0}}},
             'max_action_dim': 32,
             'max_prompt_length': 650,
             'max_state_dim': 32,
             'mode': 'train',
             'model_transforms': {'inputs': [], 'outputs': []},
             'num_video_frames': 4,
             'repack_transforms': {'inputs': [], 'outputs': []},
             'repo_id': 'robotwin/pick_dual_bottles',
             'revision': None,
             'root': None,
             'streaming': False,
             'tokenize_state': True,
             'type': 'internvla_a1_5',
             'use_external_stats': True,
             'use_fast_action_tokens': True,
             'use_imagenet_stats': True,
             'video_backend': 'torchcodec',
             'video_height': 224,
             'video_width': 224,
             'weight_rules_path': None,
             'width': 224},
 'eval': {'batch_size': 50, 'n_episodes': 50, 'use_async_envs': False},
 'eval_freq': 20000,
 'job_name': 'smoke-2608280746-internvla_a1_5-robotwin-pick_dual_bottles-abs-finetune-2608280746',
 'log_freq': 1,
 'num_workers': 8,
 'optimizer': None,
 'output_dir': '/B/Ckp/itnVla_2608280746/rbt2/pick_dual_bottles/ckpt_2608280746',
 'policy': {'action_expert_hidden_size': 1024,
            'action_expert_intermediate_size': 3072,
            'action_loss_only': False,
            'action_token_max': 250124,
            'action_token_min': 248077,
            'block_action_attend_fast_tokens': True,
            'chunk_size': 50,
            'compile_mode': 'max-autotune',
            'compile_model': False,
            'device': 'cuda',
            'dtype': 'bfloat16',
            'empty_cameras': 0,
            'enable_vqa_loss': True,
            'freeze_learnable_tokens': True,
            'freeze_vision_encoder': False,
            'freeze_wan_dit': True,
            'gradient_checkpointing': False,
            'image_resolution': [224, 224],
            'inference_action_type': 'fm',
            'inference_backend': 'standard',
            'input_features': {},
            'knowledge_insulation': False,
            'lambda_vqa': 1.0,
            'license': None,
            'max_action_dim': 32,
            'max_period': 4.0,
            'max_state_dim': 32,
            'min_period': 0.004,
            'n_action_steps': 50,
            'n_obs_steps': 1,
            'normalization_mapping': {'ACTION': <NormalizationMode.IDENTITY: 'IDENTITY'>,
                                      'STATE': <NormalizationMode.IDENTITY: 'IDENTITY'>,
                                      'VISUAL': <NormalizationMode.IDENTITY: 'IDENTITY'>},
            'num_inference_steps': 10,
            'num_learnable_tokens': 50,
            'num_video_frames': 4,
            'optimizer_betas': [0.9, 0.95],
            'optimizer_eps': 1e-08,
            'optimizer_grad_clip_norm': 1.0,
            'optimizer_lr': 5e-05,
            'optimizer_weight_decay': 0.01,
            'output_features': {},
            'pretrained_path': '/B/VENV/itnvla15rbt20/var/hf_home/ckpts/InternVLA-A1.5-base',
            'private': None,
            'push_to_hub': False,
            'repo_id': 'lerobot_lab/internvla_a1_5',
            'scheduler_decay_lr': 5e-06,
            'scheduler_decay_steps': 4,
            'scheduler_warmup_steps': 1,
            'tags': None,
            'time_sampling_beta_alpha': 1.5,
            'time_sampling_beta_beta': 1.0,
            'time_sampling_offset': 0.001,
            'time_sampling_scale': 0.999,
            'tokenize_state': True,
            'tokenizer_max_length': 48,
            'train_expert_only': False,
            'type': 'internvla_a1_5',
            'use_amp': False,
            'use_sdpa': False,
            'vae_path': '/B/VENV/itnvla15rbt20/var/hf_home/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth',
            'video_height': 224,
            'video_loss_only': False,
            'video_loss_weight': 1.0,
            'video_precision': 'bfloat16',
            'video_width': 224,
            'vlm_model_name_or_path': 'Qwen/Qwen3.5-2B',
            'wan_checkpoint_path': '/B/VENV/itnvla15rbt20/var/hf_home/hub/Wan2.2-TI2V-5B',
            'wan_config_path': '/B/VENV/itnvla15rbt20/var/hf_home/hub/Wan2.2-TI2V-5B'},
 'rename_map': {},
 'resume': False,
 'save_checkpoint': True,
 'save_freq': 2,
 'scheduler': None,
 'seed': 42,
 'steps': 4,
 'use_policy_training_preset': True,
 'vqa_dataset': None,
 'wandb': {'disable_artifact': False,
           'enable': True,
           'entity': None,
           'mode': 'offline',
           'notes': None,
           'project': 'internvla_a1_5',
           'run_id': None}}
/B/VENV/itnvla15rbt20/lib/python3.11/site-packages/pydantic/_internal/_generate_schema.py:2274: UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used. 'repr' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/B/VENV/itnvla15rbt20/lib/python3.11/site-packages/pydantic/_internal/_generate_schema.py:2274: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
INFO 2026-08-28 07:46:57 db_utils.py:114 Logs will be synced with wandb.
INFO 2026-08-28 07:46:57 db_utils.py:119 Track this run --> /B/Ckp/itnVla_2608280746/rbt2/pick_dual_bottles/ckpt_2608280746/wandb/offline-run-20260828_074656-gnscwdaq
INFO 2026-08-28 07:46:57 ot_train.py:207 Creating dataset
INFO 2026-08-28 07:46:57 /factory.py:490 [make_dataset] all_repo_ids=['robotwin/pick_dual_bottles']
INFO 2026-08-28 07:46:57 /factory.py:526 [rank=00/08] repo_ids_for_this_rank:
[rank 0] repo_id=robotwin/pick_dual_bottles
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-ARX AC One.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-ARX Lift-2.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-AgileX Split Aloha.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-Franka.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-Genie-1.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-franka.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-frankarobotiq.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-genie1.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-piper.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-split_aloha.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a2d.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/agilex_3rgb.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/aloha.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/arx_lift2.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/arx_x5.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/calvin_franka.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/dex.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/fastumi-dual.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/fastumi-single.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/franka_droid.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/google_robot.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/libero.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/lift2.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/panda.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/piper_robotwin.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/r1lite.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/robocasa_piper.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/robotwin-aloha.yaml
INFO 2026-08-28 07:46:57 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/robotwin-ur5.yaml
INFO 2026-08-28 07:46:57 _client.py:1025 HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen3.5-2B/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:46:57 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:46:57 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/chat_template.json "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:46:57 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/chat_template.jinja "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:46:57 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/chat_template.jinja "HTTP/1.1 200 OK"
INFO 2026-08-28 07:46:58 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/audio_tokenizer_config.json "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:46:58 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:46:58 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:46:58 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/preprocessor_config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:46:58 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:46:58 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:46:58 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/preprocessor_config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:46:58 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:46:58 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:46:58 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:46:58 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/tokenizer_config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:46:58 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:46:58 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/tokenizer_config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:46:59 _client.py:1025 HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen3.5-2B/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:46:59 _client.py:1025 HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen3.5-2B/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
INFO 2026-08-28 07:47:00 _client.py:1025 HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen3.5-2B "HTTP/1.1 200 OK"
INFO 2026-08-28 07:47:00 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:47:00 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/video_preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:47:00 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/video_preprocessor_config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:47:00 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:47:00 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/preprocessor_config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:47:00 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:47:00 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/video_preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:47:00 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/video_preprocessor_config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:47:00 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:47:00 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/preprocessor_config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:47:01 /factory.py:413 Using external stats from /B/VENV/itnvla15rbt20/var/hf_home/lerobot/stats/aloha/abs/agg_1repos_59c5e8f4cd/stats.json
INFO 2026-08-28 07:47:01 /factory.py:559 None
INFO 2026-08-28 07:47:06 ot_train.py:234 Creating policy
INFO 2026-08-28 07:47:06 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:47:06 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:47:06 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/model.safetensors "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:47:07 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/model.safetensors.index.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:47:07 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/model.safetensors.index.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:47:07 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/model.safetensors-00001-of-00001.safetensors "HTTP/1.1 302 Found"
INFO 2026-08-28 07:47:07 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/generation_config.json "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:47:07 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:47:07 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:47:07 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:47:07 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/tokenizer_config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:47:07 _client.py:1025 HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen3.5-2B/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:47:07 _client.py:1025 HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen3.5-2B/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
INFO 2026-08-28 07:47:09 _client.py:1025 HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen3.5-2B "HTTP/1.1 200 OK"
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
INFO 2026-08-28 07:48:20 s/vae2_2.py:881 loading /B/VENV/itnvla15rbt20/var/hf_home/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
INFO 2026-08-28 07:48:21 wan_model.py:53 WAN Video Model initialized with 4,999,787,712 parameters
INFO 2026-08-28 07:48:21 an_model.py:118 Loading WAN weights from /B/VENV/itnvla15rbt20/var/hf_home/hub/Wan2.2-TI2V-5B

Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 63.72it/s]

Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 76.98it/s]

Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 71.56it/s]

Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 72.38it/s]

Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 74.85it/s]

Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 73.63it/s]

Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 74.78it/s]

Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 74.47it/s]
INFO 2026-08-28 07:48:30 an_model.py:162 Successfully loaded WAN weights from directory
Loading weights from local directory
Loading weights from local directory
Loading weights from local directory
Loading weights from local directory
Loading weights from local directory
Loading weights from local directory
Loading weights from local directory
Loading weights from local directory
WARNING 2026-08-28 07:48:33 ies/utils.py:90 Missing key(s) when loading model: {'model.wan_video_model.wan_model.blocks.8.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.6.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.8.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.12.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.7.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.17.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.23.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.18.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.11.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.26.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.26.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.27.modulation', 'model.wan_video_model.wan_model.blocks.29.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.12.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.14.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.29.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.1.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.11.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.27.ffn.0.weight', 'model.wan_video_model.wan_model.patch_embedding.bias', 'model.wan_video_model.wan_model.blocks.5.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.9.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.18.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.4.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.10.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.27.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.29.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.2.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.17.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.9.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.15.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.20.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.15.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.21.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.27.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.17.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.14.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.26.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.12.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.7.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.25.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.4.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.1.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.6.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.14.modulation', 'model.wan_video_model.wan_model.blocks.6.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.18.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.0.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.25.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.9.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.11.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.4.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.26.norm3.bias', 'model.wan_video_model.wan_model.blocks.11.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.25.modulation', 'model.wan_video_model.wan_model.blocks.14.norm3.bias', 'model.wan_video_model.wan_model.blocks.6.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.27.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.24.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.6.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.13.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.24.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.28.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.7.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.16.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.16.norm3.weight', 'model.wan_video_model.wan_model.blocks.18.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.15.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.27.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.2.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.3.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.18.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.18.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.7.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.7.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.29.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.14.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.27.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.18.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.28.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.4.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.7.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.4.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.11.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.15.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.14.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.26.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.20.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.24.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.24.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.11.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.9.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.21.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.29.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.21.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.4.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.19.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.27.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.17.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.11.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.1.norm3.weight', 'model.wan_video_model.wan_model.blocks.12.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.0.norm3.weight', 'model.wan_video_model.wan_model.blocks.24.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.16.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.11.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.13.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.18.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.21.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.16.ffn.0.bias', 'model.wan_video_model.wan_model.time_projection.1.weight', 'model.wan_video_model.wan_model.blocks.15.norm3.bias', 'model.wan_video_model.wan_model.blocks.2.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.7.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.16.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.9.norm3.weight', 'model.wan_video_model.wan_model.blocks.10.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.3.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.8.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.26.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.17.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.1.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.27.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.4.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.23.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.26.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.19.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.0.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.3.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.16.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.7.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.21.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.6.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.8.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.20.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.24.norm3.bias', 'model.wan_video_model.wan_model.blocks.25.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.29.norm3.bias', 'model.wan_video_model.wan_model.blocks.5.norm3.bias', 'model.wan_video_model.wan_model.blocks.16.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.15.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.25.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.8.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.1.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.29.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.9.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.27.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.29.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.13.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.4.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.19.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.10.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.13.norm3.bias', 'model.wan_video_model.wan_model.blocks.22.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.9.modulation', 'model.wan_video_model.wan_model.blocks.18.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.12.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.1.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.15.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.0.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.0.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.24.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.29.cross_attn.k.weight', 'model.wan_video_model.wan_model.time_embedding.0.weight', 'model.wan_video_model.wan_model.blocks.12.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.14.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.5.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.9.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.11.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.14.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.19.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.3.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.25.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.11.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.6.modulation', 'model.wan_video_model.wan_model.blocks.16.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.13.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.10.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.3.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.17.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.1.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.5.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.8.modulation', 'model.wan_video_model.wan_model.blocks.24.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.19.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.0.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.2.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.11.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.28.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.2.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.16.norm3.bias', 'model.wan_video_model.wan_model.blocks.3.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.19.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.6.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.27.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.21.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.11.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.20.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.3.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.15.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.25.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.29.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.4.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.16.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.25.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.20.norm3.bias', 'model.wan_video_model.wan_model.blocks.20.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.1.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.20.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.22.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.29.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.2.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.7.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.29.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.12.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.27.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.26.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.10.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.28.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.20.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.20.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.0.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.9.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.5.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.10.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.22.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.9.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.17.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.28.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.13.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.1.modulation', 'model.wan_video_model.wan_model.blocks.20.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.28.norm3.bias', 'model.wan_video_model.wan_model.blocks.7.norm3.bias', 'model.wan_video_model.wan_model.blocks.27.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.28.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.21.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.5.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.6.norm3.weight', 'model.wan_video_model.wan_model.blocks.18.modulation', 'model.wan_video_model.wan_model.blocks.0.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.8.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.0.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.6.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.8.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.18.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.18.cross_attn.k.bias', 'model.wan_video_model.wan_model.time_embedding.2.bias', 'model.wan_video_model.wan_model.blocks.20.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.28.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.27.self_attn.o.weight', 'model.wan_video_model.wan_model.patch_embedding.weight', 'model.wan_video_model.wan_model.blocks.23.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.7.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.5.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.10.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.22.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.3.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.2.norm3.weight', 'model.wan_video_model.wan_model.blocks.19.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.5.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.4.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.18.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.23.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.28.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.17.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.19.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.0.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.16.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.24.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.20.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.5.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.3.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.1.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.7.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.8.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.10.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.25.norm3.weight', 'model.wan_video_model.wan_model.blocks.28.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.23.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.23.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.26.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.16.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.12.modulation', 'model.wan_video_model.wan_model.blocks.15.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.8.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.0.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.12.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.11.norm3.bias', 'model.wan_video_model.wan_model.blocks.7.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.8.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.27.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.0.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.22.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.7.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.16.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.29.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.20.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.24.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.17.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.8.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.15.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.29.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.9.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.23.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.24.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.26.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.7.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.8.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.23.norm3.bias', 'model.wan_video_model.wan_model.blocks.21.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.19.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.time_projection.1.bias', 'model.wan_video_model.wan_model.blocks.19.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.10.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.29.modulation', 'model.wan_video_model.wan_model.blocks.15.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.4.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.5.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.10.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.16.modulation', 'model.wan_video_model.wan_model.blocks.29.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.20.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.11.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.23.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.22.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.6.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.1.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.3.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.3.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.16.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.16.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.9.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.19.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.24.modulation', 'model.wan_video_model.wan_model.blocks.23.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.25.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.12.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.17.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.18.norm3.bias', 'model.wan_video_model.wan_model.blocks.21.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.23.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.16.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.14.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.3.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.11.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.15.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.1.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.9.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.13.cross_attn.o.bias', 'model.wan_video_model.wan_model.text_embedding.0.bias', 'model.wan_video_model.wan_model.blocks.10.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.17.norm3.bias', 'model.wan_video_model.wan_model.blocks.6.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.5.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.24.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.25.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.29.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.28.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.0.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.23.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.6.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.14.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.7.norm3.weight', 'model.wan_video_model.wan_model.blocks.26.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.12.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.27.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.4.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.27.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.2.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.2.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.26.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.2.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.8.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.15.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.20.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.10.modulation', 'model.wan_video_model.wan_model.blocks.28.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.21.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.6.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.24.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.0.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.13.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.13.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.25.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.22.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.12.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.29.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.4.modulation', 'model.wan_video_model.wan_model.blocks.17.modulation', 'model.wan_video_model.wan_model.blocks.2.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.10.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.7.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.12.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.2.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.3.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.20.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.18.norm3.weight', 'model.wan_video_model.wan_model.blocks.13.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.16.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.2.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.18.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.0.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.9.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.12.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.20.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.3.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.3.norm3.weight', 'model.wan_video_model.wan_model.blocks.13.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.23.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.26.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.20.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.7.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.19.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.0.modulation', 'model.wan_video_model.wan_model.blocks.19.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.0.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.23.modulation', 'model.wan_video_model.wan_model.blocks.12.norm3.weight', 'model.wan_video_model.wan_model.blocks.13.modulation', 'model.wan_video_model.wan_model.blocks.8.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.16.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.11.modulation', 'model.wan_video_model.wan_model.blocks.28.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.10.norm3.weight', 'model.wan_video_model.wan_model.blocks.19.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.26.modulation', 'model.wan_video_model.wan_model.blocks.4.cross_attn.v.weight', 'model.wan_video_model.wan_model.text_embedding.2.bias', 'model.wan_video_model.wan_model.blocks.25.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.7.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.6.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.11.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.10.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.13.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.27.norm3.bias', 'model.wan_video_model.wan_model.blocks.14.norm3.weight', 'model.wan_video_model.wan_model.blocks.6.norm3.bias', 'model.wan_video_model.wan_model.blocks.5.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.19.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.28.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.12.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.19.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.16.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.26.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.14.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.15.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.21.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.22.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.13.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.14.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.15.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.17.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.24.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.5.norm3.weight', 'model.wan_video_model.wan_model.blocks.6.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.10.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.18.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.20.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.12.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.9.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.text_embedding.0.weight', 'model.wan_video_model.wan_model.blocks.2.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.18.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.19.modulation', 'model.wan_video_model.wan_model.blocks.9.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.8.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.12.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.12.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.23.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.6.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.15.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.2.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.1.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.21.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.14.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.25.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.3.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.4.norm3.weight', 'model.wan_video_model.wan_model.blocks.28.cross_attn.v.weight', 'model.wan_video_model.wan_model.head.head.bias', 'model.wan_video_model.wan_model.blocks.16.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.4.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.27.norm3.weight', 'model.wan_video_model.wan_model.blocks.1.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.15.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.19.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.6.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.25.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.28.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.7.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.0.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.24.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.20.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.3.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.4.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.6.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.5.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.9.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.17.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.8.norm3.weight', 'model.wan_video_model.wan_model.blocks.28.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.27.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.23.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.26.norm3.weight', 'model.wan_video_model.wan_model.blocks.8.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.6.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.0.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.13.norm3.weight', 'model.wan_video_model.wan_model.blocks.12.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.28.modulation', 'model.wan_video_model.wan_model.blocks.19.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.21.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.29.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.19.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.8.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.18.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.22.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.5.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.12.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.22.self_attn.k.weight', 'model.wan_video_model.wan_model.time_embedding.2.weight', 'model.wan_video_model.wan_model.blocks.14.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.22.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.4.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.22.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.20.norm3.weight', 'model.wan_video_model.wan_model.blocks.23.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.21.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.26.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.3.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.26.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.27.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.17.norm3.weight', 'model.wan_video_model.wan_model.blocks.24.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.7.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.23.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.2.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.1.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.9.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.11.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.22.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.3.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.26.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.2.norm3.bias', 'model.wan_video_model.wan_model.blocks.14.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.16.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.19.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.22.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.0.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.12.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.22.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.13.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.26.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.20.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.14.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.29.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.21.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.25.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.26.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.3.modulation', 'model.wan_video_model.wan_model.blocks.13.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.21.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.23.norm3.weight', 'model.wan_video_model.wan_model.blocks.18.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.18.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.25.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.17.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.3.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.4.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.15.modulation', 'model.wan_video_model.wan_model.blocks.17.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.15.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.10.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.6.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.8.norm3.bias', 'model.wan_video_model.wan_model.blocks.23.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.9.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.0.cross_attn.o.weight', 'model.wan_video_model.wan_model.text_embedding.2.weight', 'model.wan_video_model.wan_model.blocks.4.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.17.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.23.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.26.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.5.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.0.norm3.bias', 'model.wan_video_model.wan_model.blocks.6.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.2.modulation', 'model.wan_video_model.wan_model.blocks.27.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.28.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.23.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.25.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.13.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.1.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.2.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.5.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.2.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.25.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.9.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.10.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.12.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.13.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.3.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.2.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.1.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.0.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.22.norm3.bias', 'model.wan_video_model.wan_model.blocks.19.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.26.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.8.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.23.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.25.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.18.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.29.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.22.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.22.norm3.weight', 'model.wan_video_model.wan_model.blocks.11.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.13.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.25.norm3.bias', 'model.wan_video_model.wan_model.blocks.12.norm3.bias', 'model.wan_video_model.wan_model.blocks.2.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.13.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.27.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.21.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.21.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.9.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.20.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.29.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.17.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.19.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.5.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.22.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.1.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.10.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.2.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.16.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.4.norm3.bias', 'model.wan_video_model.wan_model.blocks.1.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.5.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.26.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.4.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.6.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.22.modulation', 'model.wan_video_model.wan_model.blocks.24.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.1.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.13.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.28.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.27.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.10.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.28.norm3.weight', 'model.wan_video_model.wan_model.blocks.16.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.10.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.13.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.17.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.9.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.18.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.24.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.21.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.15.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.5.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.14.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.17.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.7.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.22.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.16.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.11.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.15.norm3.weight', 'model.wan_video_model.wan_model.blocks.23.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.28.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.17.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.22.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.15.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.23.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.6.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.25.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.11.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.5.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.22.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.5.modulation', 'model.wan_video_model.wan_model.blocks.25.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.15.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.19.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.29.norm3.weight', 'model.wan_video_model.wan_model.blocks.14.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.8.ffn.0.bias', 'model.wan_video_model.wan_model.time_embedding.0.bias', 'model.wan_video_model.wan_model.blocks.20.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.22.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.14.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.20.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.14.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.24.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.0.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.1.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.9.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.11.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.5.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.0.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.13.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.13.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.18.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.10.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.27.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.10.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.29.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.14.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.28.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.12.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.19.norm3.bias', 'model.wan_video_model.wan_model.blocks.12.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.7.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.18.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.2.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.27.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.25.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.9.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.0.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.15.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.10.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.21.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.26.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.14.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.6.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.28.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.16.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.1.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.21.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.25.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.9.norm3.bias', 'model.wan_video_model.wan_model.blocks.5.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.15.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.8.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.7.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.26.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.1.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.3.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.11.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.3.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.3.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.2.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.17.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.1.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.7.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.17.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.2.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.20.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.17.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.4.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.17.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.18.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.15.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.13.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.28.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.3.norm3.bias', 'model.wan_video_model.wan_model.blocks.8.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.1.norm3.bias', 'model.wan_video_model.wan_model.blocks.28.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.21.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.2.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.9.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.25.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.5.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.10.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.27.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.24.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.5.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.14.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.3.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.15.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.4.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.10.norm3.bias', 'model.wan_video_model.wan_model.blocks.21.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.24.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.20.modulation', 'model.wan_video_model.wan_model.blocks.7.modulation', 'model.wan_video_model.wan_model.blocks.11.norm3.weight', 'model.wan_video_model.wan_model.blocks.6.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.21.norm3.weight', 'model.wan_video_model.wan_model.blocks.4.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.8.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.8.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.16.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.25.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.26.cross_attn.k.weight', 'model.wan_video_model.wan_model.head.modulation', 'model.wan_video_model.wan_model.blocks.24.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.14.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.29.ffn.2.bias', 'model.wan_video_model.wan_model.head.head.weight', 'model.wan_video_model.wan_model.blocks.1.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.23.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.22.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.24.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.9.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.12.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.7.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.4.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.11.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.4.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.10.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.24.norm3.weight', 'model.wan_video_model.wan_model.blocks.19.norm3.weight', 'model.wan_video_model.wan_model.blocks.1.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.17.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.22.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.11.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.21.norm3.bias', 'model.wan_video_model.wan_model.blocks.19.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.29.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.14.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.21.modulation', 'model.wan_video_model.wan_model.blocks.11.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.22.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.24.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.14.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.13.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.0.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.8.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.28.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.24.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.23.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.29.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.21.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.18.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.5.self_attn.v.bias'}
INFO 2026-08-28 07:48:35 ot_train.py:243 Creating optimizer and scheduler
INFO 2026-08-28 07:48:35 ot_train.py:261 Output dir: /B/Ckp/itnVla_2608280746/rbt2/pick_dual_bottles/ckpt_2608280746
INFO 2026-08-28 07:48:35 ot_train.py:262 cfg.steps=4 (4)
INFO 2026-08-28 07:48:35 ot_train.py:263 [91m[1mnum_frames=6129 (6K)[0m
INFO 2026-08-28 07:48:35 ot_train.py:264 [91m[1mnum_episodes=50 (50)[0m
INFO 2026-08-28 07:48:35 ot_train.py:265 Effective batch size: 16 x 8 = 128
INFO 2026-08-28 07:48:35 ot_train.py:266 policy info:
============================================================
Policy: InternVLAA15Policy

Parameter statistics:
  - Total params        : 8B
  - Trainable params    : 3B
  - Qwen3_5 params      : 2B
  - Action expert params: 460M
  - WAN params          : 5B
  - Learnable tokens    : 50
  - Knowledge insulation: False
  - Inference backend   : standard
  - Freeze WAN DiT      : True
============================================================
INFO 2026-08-28 07:48:35 ot_train.py:329 Start offline training on a fixed dataset
[rank2]:[W828 07:48:45.740678829 reducer.cpp:1500] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[rank3]:[W828 07:48:46.013854016 reducer.cpp:1500] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[rank1]:[W828 07:48:46.059748283 reducer.cpp:1500] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[rank6]:[W828 07:48:46.112158620 reducer.cpp:1500] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[rank5]:[W828 07:48:46.142237179 reducer.cpp:1500] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[rank4]:[W828 07:48:46.217351567 reducer.cpp:1500] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[rank0]:[W828 07:48:46.229422514 reducer.cpp:1500] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[rank7]:[W828 07:48:46.313412950 reducer.cpp:1500] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
INFO 2026-08-28 07:48:46 __init__.py:187 [FLA Backend] common.chunk_bwd_dqkwg -> tilelang
INFO 2026-08-28 07:48:49 ot_train.py:367  [92m[1m00:00:14 << 00:00:27[0m | [96m[1m0.11 iters/s[0m | step:1.0 | sample:128 | episode:1 | epoch:0.02 | loss:7.084 | loss_action:0.256 | grdn:33.071 | lr:4.3e-05 | updt_s:9.006 | data_s:5.103 | loss_vqa:4.293 | loss_video:0.228 | loss_fast:4.337 | loss_subtask:0.000
INFO 2026-08-28 07:48:51 ot_train.py:367  [92m[1m00:00:15 << 00:00:03[0m | [96m[1m0.54 iters/s[0m | step:2.0 | sample:256 | episode:2 | epoch:0.04 | loss:5.135 | loss_action:0.145 | grdn:42.206 | lr:2.8e-05 | updt_s:1.843 | data_s:0.003 | loss_vqa:3.441 | loss_video:0.245 | loss_fast:3.488 | loss_subtask:0.000
INFO 2026-08-28 07:48:51 ot_train.py:377 Checkpoint policy after step 2
INFO 2026-08-28 07:48:51 ot_train.py:379 Checkpoint saved at: /B/Ckp/itnVla_2608280746/rbt2/pick_dual_bottles/ckpt_2608280746/checkpoints/000002
INFO 2026-08-28 07:49:18 ot_train.py:367  [92m[1m00:00:43 << 00:00:00[0m | [96m[1m1.02 iters/s[0m | step:3.0 | sample:384 | episode:3 | epoch:0.06 | loss:6.483 | loss_action:0.234 | grdn:52.535 | lr:1.2e-05 | updt_s:0.978 | data_s:0.005 | loss_vqa:3.876 | loss_video:0.267 | loss_fast:3.965 | loss_subtask:0.001
INFO 2026-08-28 07:49:19 ot_train.py:367  [92m[1m00:00:44 << 00:00:00[0m | [96m[1m0.96 iters/s[0m | step:4.0 | sample:512 | episode:4 | epoch:0.08 | loss:4.861 | loss_action:0.136 | grdn:25.691 | lr:5.0e-06 | updt_s:1.041 | data_s:0.003 | loss_vqa:3.284 | loss_video:0.216 | loss_fast:3.404 | loss_subtask:0.000
INFO 2026-08-28 07:49:19 ot_train.py:377 Checkpoint policy after step 4
INFO 2026-08-28 07:49:19 ot_train.py:379 Checkpoint saved at: /B/Ckp/itnVla_2608280746/rbt2/pick_dual_bottles/ckpt_2608280746/checkpoints/000004
INFO 2026-08-28 07:49:42 ot_train.py:396 End of training
2026-08-28T07:51:56Z
0, 0 MiB, 143156 MiB, 0 %
1, 0 MiB, 143156 MiB, 0 %
2, 0 MiB, 143156 MiB, 0 %
3, 0 MiB, 143156 MiB, 0 %
4, 0 MiB, 143156 MiB, 0 %
5, 0 MiB, 143156 MiB, 0 %
6, 0 MiB, 143156 MiB, 0 %
7, 0 MiB, 143156 MiB, 0 %
 179787       00:00 S /bin/bash -O extglob -c snap=$(command cat <&3) && builtin shopt -s extglob && builtin eval -- "$snap" && { builtin set +u 2>/dev/null || true; builtin eval "${__CURSOR_SANDBOX_ENV_RESTORE:-}" 2>/dev/null; builtin export PWD="$(builtin pwd)"; builtin shopt -s expand_aliases 2>/dev/null; builtin eval "$1" < /dev/null; }; COMMAND_EXIT_CODE=$?; dump_bash_state >&4; builtin exit $COMMAND_EXIT_CODE -- set -o pipefail; { date -u '+%Y-%m-%dT%H:%M:%SZ'; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader; ps -eo pid,etime,state,cmd | rg 'lerobot_train|accelerate.commands.launch|pick_dual_bottles' || true; unset CUDA_VISIBLE_DEVICES; STAMP="$(date +%y%m%d%H%M)"; TASK_NAME=pick_dual_bottles RUN_STAMP="${STAMP}" ITNVLA_STAMP="${STAMP}" NUM_EPOCHS=76 TOTAL_BATCH_SIZE=128 bash launch/internvla_a15_finetune_robotwin_comm.sh; } 2>&1 | tee -a b/d/p/reprd_rbtwn_pikDulBtlLOG.md
 179801       00:00 S /bin/bash -O extglob -c snap=$(command cat <&3) && builtin shopt -s extglob && builtin eval -- "$snap" && { builtin set +u 2>/dev/null || true; builtin eval "${__CURSOR_SANDBOX_ENV_RESTORE:-}" 2>/dev/null; builtin export PWD="$(builtin pwd)"; builtin shopt -s expand_aliases 2>/dev/null; builtin eval "$1" < /dev/null; }; COMMAND_EXIT_CODE=$?; dump_bash_state >&4; builtin exit $COMMAND_EXIT_CODE -- set -o pipefail; { date -u '+%Y-%m-%dT%H:%M:%SZ'; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader; ps -eo pid,etime,state,cmd | rg 'lerobot_train|accelerate.commands.launch|pick_dual_bottles' || true; unset CUDA_VISIBLE_DEVICES; STAMP="$(date +%y%m%d%H%M)"; TASK_NAME=pick_dual_bottles RUN_STAMP="${STAMP}" ITNVLA_STAMP="${STAMP}" NUM_EPOCHS=76 TOTAL_BATCH_SIZE=128 bash launch/internvla_a15_finetune_robotwin_comm.sh; } 2>&1 | tee -a b/d/p/reprd_rbtwn_pikDulBtlLOG.md
 179807       00:00 R rg lerobot_train|accelerate.commands.launch|pick_dual_bottles
MASTER_ADDR=127.0.0.1, MASTER_PORT=36222
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PROC_PER_NODE=8
NUM_FRAMES=6129 NUM_EPOCHS=76 TOTAL_BATCH_SIZE=128
BATCH_SIZE(per GPU)=16 PROC_PER_NODE=8 DIST_LOADING=false
STEPS=3640 SAVE_FREQ=910 WARMUP_STEPS=364 LOG_FREQ=50
ckpt steps ~= 910 / 1820 / 2730 / 3640 (last always saved)
OUTPUT_ROOT=/B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles
OUTPUT_DIR =/B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/ckpt_2608280751
LOG_FILE   =/B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/train_2608280751.log
DATASET_REPO_ID=robotwin/pick_dual_bottles
ROBOT_TYPE=aloha
EXTERNAL_STATS_PATH=/B/VENV/itnvla15rbt20/var/hf_home/lerobot/stats/aloha/abs/agg_1repos_59c5e8f4cd/stats.json
dataset codebase_version=v3.0 at /B/VENV/itnvla15rbt20/var/hf_home/lerobot/robotwin/pick_dual_bottles/meta/info.json
===== launching training; log -> /B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/train_2608280751.log =====
The following values were not passed to `accelerate launch` and had defaults used instead:
	`--mixed_precision` was set to a value of `'no'`
	`--dynamo_backend` was set to a value of `'no'`
To avoid this warning pass in values for each of the problematic parameters or run `accelerate config`.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
INFO 2026-08-28 07:52:24 ot_train.py:173 {'batch_size': 16,
 'checkpoint_path': None,
 'dataset': {'action_mode': 'abs',
             'buffer_size': 1024,
             'chunk_size': 50,
             'data_transforms': {'inputs': [{'height': 224,
                                             'mapping': {},
                                             'mode': 'bilinear',
                                             'type': 'resize_with_pad',
                                             'width': 224},
                                            {'mapping': {},
                                             'type': 'remap_image_key'},
                                            {'normalize_to_minus1_1': True,
                                             'source_view': 'observation.images.image0',
                                             'type': 'extract_video_frames',
                                             'video_key': 'observation.video_frames'},
                                            {'mode': 'mean_std',
                                             'norm_stats': {},
                                             'selected_keys': None,
                                             'type': 'normalize'},
                                            {'mapping': {},
                                             'type': 'compose_fields'},
                                            {'action_token_max': 250124,
                                             'action_token_min': 248077,
                                             'action_tokenizer_name': 'physical-intelligence/fast',
                                             'assistant_end_tokens': [248045,
                                                                      74455,
                                                                      198,
                                                                      248068,
                                                                      271,
                                                                      248069,
                                                                      271],
                                             'chunk_size': 50,
                                             'max_action_dim': 32,
                                             'max_action_tokens': 256,
                                             'qwen35_model_name': 'Qwen/Qwen3.5-2B',
                                             'stop_token_1': 248046,
                                             'stop_token_2': 198,
                                             'type': 'fast_internvla_a1_5_action_tokenizer'},
                                            {'_cache': {},
                                             '_loaded': False,
                                             'annotations_file': '',
                                             'memory_output_key': 'language_memory',
                                             'output_key': 'sub_task',
                                             'type': 'load_action_text_from_jsonl'},
                                            {'action_mode': 'joint',
                                             'action_text_key': 'action.action_text',
                                             'action_token_max': 250124,
                                             'action_token_min': 248077,
                                             'language_memory_key': 'language_memory',
                                             'max_length': 650,
                                             'max_state_dim': 32,
                                             'mode': 'train',
                                             'num_views': 3,
                                             'padding': 'max_length',
                                             'pretrained_model_name_or_path': 'Qwen/Qwen3.5-2B',
                                             'task_key': 'task',
                                             'tokenize_state': True,
                                             'truncation': True,
                                             'type': 'internvla_a1_5_chat_processor',
                                             'use_fast_action_tokens': True},
                                            {'max_action_dim': 32,
                                             'max_state_dim': 32,
                                             'type': 'pad_state_and_action'},
                                            {'action_reorder': None,
                                             'state_reorder': None,
                                             'type': 'reorder_state_action'},
                                            {'num_video_frames': 4,
                                             'type': 'unify_internvla_a1_5_inputs',
                                             'video_height': 224,
                                             'video_width': 224}],
                                 'outputs': []},
             'dist_loading': False,
             'episodes': None,
             'external_stats_path': '/B/VENV/itnvla15rbt20/var/hf_home/lerobot/stats/aloha/abs/agg_1repos_59c5e8f4cd/stats.json',
             'height': 224,
             'image_transforms': {'enable': False,
                                  'max_num_transforms': 3,
                                  'random_order': False,
                                  'tfs': {'affine': {'kwargs': {'degrees': [-5.0,
                                                                            5.0],
                                                                'translate': [0.05,
                                                                              0.05]},
                                                     'type': 'RandomAffine',
                                                     'weight': 1.0},
                                          'brightness': {'kwargs': {'brightness': [0.8,
                                                                                   1.2]},
                                                         'type': 'ColorJitter',
                                                         'weight': 1.0},
                                          'contrast': {'kwargs': {'contrast': [0.8,
                                                                               1.2]},
                                                       'type': 'ColorJitter',
                                                       'weight': 1.0},
                                          'hue': {'kwargs': {'hue': [-0.05,
                                                                     0.05]},
                                                  'type': 'ColorJitter',
                                                  'weight': 1.0},
                                          'saturation': {'kwargs': {'saturation': [0.5,
                                                                                   1.5]},
                                                         'type': 'ColorJitter',
                                                         'weight': 1.0},
                                          'sharpness': {'kwargs': {'sharpness': [0.5,
                                                                                 1.5]},
                                                        'type': 'SharpnessJitter',
                                                        'weight': 1.0}}},
             'max_action_dim': 32,
             'max_prompt_length': 650,
             'max_state_dim': 32,
             'mode': 'train',
             'model_transforms': {'inputs': [], 'outputs': []},
             'num_video_frames': 4,
             'repack_transforms': {'inputs': [], 'outputs': []},
             'repo_id': 'robotwin/pick_dual_bottles',
             'revision': None,
             'root': None,
             'streaming': False,
             'tokenize_state': True,
             'type': 'internvla_a1_5',
             'use_external_stats': True,
             'use_fast_action_tokens': True,
             'use_imagenet_stats': True,
             'video_backend': 'torchcodec',
             'video_height': 224,
             'video_width': 224,
             'weight_rules_path': None,
             'width': 224},
 'eval': {'batch_size': 50, 'n_episodes': 50, 'use_async_envs': False},
 'eval_freq': 20000,
 'job_name': '2608280751-internvla_a1_5-robotwin-pick_dual_bottles-abs-finetune-2608280751',
 'log_freq': 50,
 'num_workers': 8,
 'optimizer': None,
 'output_dir': '/B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/ckpt_2608280751',
 'policy': {'action_expert_hidden_size': 1024,
            'action_expert_intermediate_size': 3072,
            'action_loss_only': False,
            'action_token_max': 250124,
            'action_token_min': 248077,
            'block_action_attend_fast_tokens': True,
            'chunk_size': 50,
            'compile_mode': 'max-autotune',
            'compile_model': False,
            'device': 'cuda',
            'dtype': 'bfloat16',
            'empty_cameras': 0,
            'enable_vqa_loss': True,
            'freeze_learnable_tokens': True,
            'freeze_vision_encoder': False,
            'freeze_wan_dit': True,
            'gradient_checkpointing': False,
            'image_resolution': [224, 224],
            'inference_action_type': 'fm',
            'inference_backend': 'standard',
            'input_features': {},
            'knowledge_insulation': False,
            'lambda_vqa': 1.0,
            'license': None,
            'max_action_dim': 32,
            'max_period': 4.0,
            'max_state_dim': 32,
            'min_period': 0.004,
            'n_action_steps': 50,
            'n_obs_steps': 1,
            'normalization_mapping': {'ACTION': <NormalizationMode.IDENTITY: 'IDENTITY'>,
                                      'STATE': <NormalizationMode.IDENTITY: 'IDENTITY'>,
                                      'VISUAL': <NormalizationMode.IDENTITY: 'IDENTITY'>},
            'num_inference_steps': 10,
            'num_learnable_tokens': 50,
            'num_video_frames': 4,
            'optimizer_betas': [0.9, 0.95],
            'optimizer_eps': 1e-08,
            'optimizer_grad_clip_norm': 1.0,
            'optimizer_lr': 5e-05,
            'optimizer_weight_decay': 0.01,
            'output_features': {},
            'pretrained_path': '/B/VENV/itnvla15rbt20/var/hf_home/ckpts/InternVLA-A1.5-base',
            'private': None,
            'push_to_hub': False,
            'repo_id': 'lerobot_lab/internvla_a1_5',
            'scheduler_decay_lr': 5e-06,
            'scheduler_decay_steps': 3640,
            'scheduler_warmup_steps': 364,
            'tags': None,
            'time_sampling_beta_alpha': 1.5,
            'time_sampling_beta_beta': 1.0,
            'time_sampling_offset': 0.001,
            'time_sampling_scale': 0.999,
            'tokenize_state': True,
            'tokenizer_max_length': 48,
            'train_expert_only': False,
            'type': 'internvla_a1_5',
            'use_amp': False,
            'use_sdpa': False,
            'vae_path': '/B/VENV/itnvla15rbt20/var/hf_home/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth',
            'video_height': 224,
            'video_loss_only': False,
            'video_loss_weight': 1.0,
            'video_precision': 'bfloat16',
            'video_width': 224,
            'vlm_model_name_or_path': 'Qwen/Qwen3.5-2B',
            'wan_checkpoint_path': '/B/VENV/itnvla15rbt20/var/hf_home/hub/Wan2.2-TI2V-5B',
            'wan_config_path': '/B/VENV/itnvla15rbt20/var/hf_home/hub/Wan2.2-TI2V-5B'},
 'rename_map': {},
 'resume': False,
 'save_checkpoint': True,
 'save_freq': 910,
 'scheduler': None,
 'seed': 42,
 'steps': 3640,
 'use_policy_training_preset': True,
 'vqa_dataset': None,
 'wandb': {'disable_artifact': False,
           'enable': True,
           'entity': None,
           'mode': 'offline',
           'notes': None,
           'project': 'internvla_a1_5',
           'run_id': None}}
/B/VENV/itnvla15rbt20/lib/python3.11/site-packages/pydantic/_internal/_generate_schema.py:2274: UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used. 'repr' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/B/VENV/itnvla15rbt20/lib/python3.11/site-packages/pydantic/_internal/_generate_schema.py:2274: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
INFO 2026-08-28 07:52:27 db_utils.py:114 Logs will be synced with wandb.
INFO 2026-08-28 07:52:27 db_utils.py:119 Track this run --> /B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/ckpt_2608280751/wandb/offline-run-20260828_075227-iwfgf9hd
INFO 2026-08-28 07:52:27 ot_train.py:207 Creating dataset
INFO 2026-08-28 07:52:27 /factory.py:490 [make_dataset] all_repo_ids=['robotwin/pick_dual_bottles']
INFO 2026-08-28 07:52:27 /factory.py:526 [rank=00/08] repo_ids_for_this_rank:
[rank 0] repo_id=robotwin/pick_dual_bottles
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-ARX AC One.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-ARX Lift-2.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-AgileX Split Aloha.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-Franka.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-Genie-1.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-franka.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-frankarobotiq.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-genie1.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-piper.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a1-split_aloha.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/a2d.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/agilex_3rgb.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/aloha.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/arx_lift2.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/arx_x5.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/calvin_franka.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/dex.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/fastumi-dual.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/fastumi-single.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/franka_droid.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/google_robot.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/libero.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/lift2.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/panda.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/piper_robotwin.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/r1lite.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/robocasa_piper.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/robotwin-aloha.yaml
INFO 2026-08-28 07:52:27 registry.py:112 Loaded schema(s) from /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/robotwin-ur5.yaml
INFO 2026-08-28 07:52:27 _client.py:1025 HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen3.5-2B/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:52:27 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:52:28 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/chat_template.json "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:52:28 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/chat_template.jinja "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:52:28 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/chat_template.jinja "HTTP/1.1 200 OK"
INFO 2026-08-28 07:52:28 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/audio_tokenizer_config.json "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:52:28 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:52:28 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:52:28 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/preprocessor_config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:52:28 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:52:28 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:52:28 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/preprocessor_config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:52:28 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:52:28 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:52:28 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:52:28 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/tokenizer_config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:52:28 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:52:28 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/tokenizer_config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:52:29 _client.py:1025 HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen3.5-2B/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:52:29 _client.py:1025 HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen3.5-2B/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
INFO 2026-08-28 07:52:30 _client.py:1025 HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen3.5-2B "HTTP/1.1 200 OK"
INFO 2026-08-28 07:52:30 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:52:30 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/video_preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:52:30 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/video_preprocessor_config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:52:30 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:52:30 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/preprocessor_config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:52:30 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:52:30 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/video_preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:52:30 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/video_preprocessor_config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:52:30 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/preprocessor_config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:52:30 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/preprocessor_config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:52:31 /factory.py:413 Using external stats from /B/VENV/itnvla15rbt20/var/hf_home/lerobot/stats/aloha/abs/agg_1repos_59c5e8f4cd/stats.json
INFO 2026-08-28 07:52:31 /factory.py:559 None
INFO 2026-08-28 07:52:36 ot_train.py:234 Creating policy
INFO 2026-08-28 07:52:37 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:52:37 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:52:37 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/model.safetensors "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:52:37 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/model.safetensors.index.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:52:37 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/model.safetensors.index.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:52:37 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/model.safetensors-00001-of-00001.safetensors "HTTP/1.1 302 Found"
INFO 2026-08-28 07:52:37 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/generation_config.json "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:52:37 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:52:38 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:52:38 _client.py:1025 HTTP Request: HEAD https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
INFO 2026-08-28 07:52:38 _client.py:1025 HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc/tokenizer_config.json "HTTP/1.1 200 OK"
INFO 2026-08-28 07:52:38 _client.py:1025 HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen3.5-2B/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
INFO 2026-08-28 07:52:38 _client.py:1025 HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen3.5-2B/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
INFO 2026-08-28 07:52:39 _client.py:1025 HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen3.5-2B "HTTP/1.1 200 OK"
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
INFO 2026-08-28 07:53:50 s/vae2_2.py:881 loading /B/VENV/itnvla15rbt20/var/hf_home/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth

Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 75.24it/s]

Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 71.97it/s]

Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 71.06it/s]
INFO 2026-08-28 07:53:52 wan_model.py:53 WAN Video Model initialized with 4,999,787,712 parameters
INFO 2026-08-28 07:53:52 an_model.py:118 Loading WAN weights from /B/VENV/itnvla15rbt20/var/hf_home/hub/Wan2.2-TI2V-5B

Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 72.90it/s]

Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 77.43it/s]

Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 76.44it/s]

Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 71.47it/s]

Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 73.59it/s]
Loading weights from local directory
Loading weights from local directory
Loading weights from local directory
INFO 2026-08-28 07:54:01 an_model.py:162 Successfully loaded WAN weights from directory
Loading weights from local directory
Loading weights from local directory
Loading weights from local directory
Loading weights from local directory
Loading weights from local directory
WARNING 2026-08-28 07:54:05 ies/utils.py:90 Missing key(s) when loading model: {'model.wan_video_model.wan_model.blocks.2.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.5.cross_attn.v.weight', 'model.wan_video_model.wan_model.head.head.bias', 'model.wan_video_model.wan_model.blocks.11.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.11.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.28.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.3.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.13.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.6.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.26.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.18.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.2.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.0.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.4.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.7.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.10.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.20.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.2.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.1.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.3.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.6.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.13.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.14.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.3.self_attn.o.bias', 'model.wan_video_model.wan_model.head.head.weight', 'model.wan_video_model.wan_model.blocks.27.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.26.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.9.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.16.norm3.weight', 'model.wan_video_model.wan_model.blocks.21.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.6.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.11.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.14.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.11.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.11.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.8.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.17.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.18.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.2.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.3.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.2.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.23.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.27.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.1.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.14.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.27.ffn.2.bias', 'model.wan_video_model.wan_model.time_embedding.0.weight', 'model.wan_video_model.wan_model.blocks.3.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.24.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.23.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.7.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.9.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.10.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.20.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.8.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.26.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.14.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.25.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.29.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.26.cross_attn.v.weight', 'model.wan_video_model.wan_model.patch_embedding.weight', 'model.wan_video_model.wan_model.blocks.5.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.8.norm3.weight', 'model.wan_video_model.wan_model.blocks.4.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.16.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.8.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.24.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.0.norm3.bias', 'model.wan_video_model.wan_model.blocks.3.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.5.modulation', 'model.wan_video_model.wan_model.blocks.6.norm3.bias', 'model.wan_video_model.wan_model.blocks.17.modulation', 'model.wan_video_model.wan_model.blocks.25.modulation', 'model.wan_video_model.wan_model.blocks.0.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.4.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.8.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.14.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.7.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.9.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.17.norm3.bias', 'model.wan_video_model.wan_model.blocks.14.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.23.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.25.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.15.modulation', 'model.wan_video_model.wan_model.blocks.11.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.7.modulation', 'model.wan_video_model.wan_model.blocks.11.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.1.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.1.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.21.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.10.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.3.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.13.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.23.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.28.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.14.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.25.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.25.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.20.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.7.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.21.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.19.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.28.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.29.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.22.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.22.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.25.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.10.norm3.weight', 'model.wan_video_model.wan_model.blocks.17.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.19.modulation', 'model.wan_video_model.wan_model.blocks.11.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.7.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.17.norm3.weight', 'model.wan_video_model.wan_model.blocks.0.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.14.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.13.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.2.norm3.weight', 'model.wan_video_model.wan_model.blocks.12.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.20.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.25.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.14.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.18.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.22.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.19.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.7.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.6.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.22.norm3.bias', 'model.wan_video_model.wan_model.blocks.23.self_attn.q.bias', 'model.wan_video_model.wan_model.time_embedding.0.bias', 'model.wan_video_model.wan_model.blocks.9.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.18.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.27.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.11.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.5.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.6.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.10.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.14.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.14.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.5.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.12.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.21.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.2.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.24.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.5.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.8.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.20.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.10.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.10.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.12.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.8.norm3.bias', 'model.wan_video_model.wan_model.blocks.8.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.27.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.3.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.11.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.8.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.20.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.0.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.3.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.11.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.21.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.15.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.7.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.9.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.14.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.19.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.16.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.3.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.10.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.19.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.22.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.3.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.5.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.12.modulation', 'model.wan_video_model.wan_model.blocks.20.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.27.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.18.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.7.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.22.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.7.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.8.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.20.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.11.norm3.bias', 'model.wan_video_model.wan_model.blocks.2.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.3.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.2.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.22.modulation', 'model.wan_video_model.wan_model.blocks.24.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.29.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.2.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.15.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.17.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.28.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.29.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.13.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.12.norm3.bias', 'model.wan_video_model.wan_model.blocks.24.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.24.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.3.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.5.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.14.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.5.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.18.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.22.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.20.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.4.norm3.bias', 'model.wan_video_model.wan_model.blocks.5.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.7.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.23.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.7.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.22.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.16.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.4.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.16.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.26.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.19.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.12.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.2.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.5.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.9.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.29.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.24.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.2.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.26.norm3.bias', 'model.wan_video_model.wan_model.blocks.0.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.17.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.8.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.22.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.3.modulation', 'model.wan_video_model.wan_model.blocks.4.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.20.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.14.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.15.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.29.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.14.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.4.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.3.norm3.weight', 'model.wan_video_model.wan_model.blocks.21.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.12.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.0.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.0.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.13.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.16.norm3.bias', 'model.wan_video_model.wan_model.blocks.10.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.6.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.7.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.21.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.11.norm3.weight', 'model.wan_video_model.wan_model.blocks.22.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.11.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.12.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.26.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.10.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.4.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.25.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.18.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.13.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.2.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.24.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.21.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.28.norm3.bias', 'model.wan_video_model.wan_model.blocks.22.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.1.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.19.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.18.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.19.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.11.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.22.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.25.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.15.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.17.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.23.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.7.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.24.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.27.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.25.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.28.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.27.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.head.modulation', 'model.wan_video_model.wan_model.blocks.8.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.6.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.9.norm3.bias', 'model.wan_video_model.wan_model.blocks.13.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.3.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.22.norm3.weight', 'model.wan_video_model.wan_model.blocks.24.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.23.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.0.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.17.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.21.modulation', 'model.wan_video_model.wan_model.blocks.18.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.28.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.9.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.13.norm3.bias', 'model.wan_video_model.wan_model.blocks.26.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.14.norm3.weight', 'model.wan_video_model.wan_model.blocks.22.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.6.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.25.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.1.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.13.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.16.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.23.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.11.modulation', 'model.wan_video_model.wan_model.blocks.4.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.7.norm3.weight', 'model.wan_video_model.wan_model.blocks.2.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.15.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.1.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.14.modulation', 'model.wan_video_model.wan_model.blocks.22.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.5.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.4.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.6.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.8.modulation', 'model.wan_video_model.wan_model.blocks.12.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.26.norm3.weight', 'model.wan_video_model.wan_model.blocks.1.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.19.norm3.bias', 'model.wan_video_model.wan_model.blocks.23.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.12.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.29.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.25.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.1.norm3.bias', 'model.wan_video_model.wan_model.blocks.6.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.14.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.20.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.12.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.9.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.24.modulation', 'model.wan_video_model.wan_model.blocks.13.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.24.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.27.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.28.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.21.norm3.bias', 'model.wan_video_model.wan_model.blocks.27.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.29.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.25.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.4.cross_attn.k.bias', 'model.wan_video_model.wan_model.text_embedding.2.bias', 'model.wan_video_model.wan_model.blocks.0.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.19.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.26.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.28.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.23.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.17.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.4.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.12.norm3.weight', 'model.wan_video_model.wan_model.blocks.2.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.10.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.10.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.4.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.2.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.25.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.18.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.19.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.26.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.17.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.11.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.22.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.8.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.12.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.24.norm3.bias', 'model.wan_video_model.wan_model.blocks.18.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.19.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.8.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.8.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.4.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.9.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.21.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.6.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.10.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.3.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.27.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.27.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.2.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.20.modulation', 'model.wan_video_model.wan_model.blocks.7.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.25.norm3.weight', 'model.wan_video_model.wan_model.blocks.1.modulation', 'model.wan_video_model.wan_model.blocks.23.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.3.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.1.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.11.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.6.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.0.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.2.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.10.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.0.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.25.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.25.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.16.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.26.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.15.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.16.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.10.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.10.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.9.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.28.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.23.modulation', 'model.wan_video_model.wan_model.blocks.22.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.1.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.29.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.25.norm3.bias', 'model.wan_video_model.wan_model.blocks.4.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.28.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.26.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.9.modulation', 'model.wan_video_model.wan_model.blocks.29.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.4.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.0.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.13.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.24.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.24.norm3.weight', 'model.wan_video_model.wan_model.blocks.6.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.24.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.9.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.12.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.10.norm3.bias', 'model.wan_video_model.wan_model.blocks.11.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.17.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.13.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.12.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.29.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.13.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.11.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.28.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.19.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.2.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.24.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.25.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.16.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.29.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.26.modulation', 'model.wan_video_model.wan_model.blocks.29.modulation', 'model.wan_video_model.wan_model.blocks.14.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.17.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.5.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.2.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.11.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.18.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.12.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.9.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.6.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.11.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.17.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.24.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.12.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.17.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.28.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.19.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.20.norm3.weight', 'model.wan_video_model.wan_model.blocks.5.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.28.norm3.weight', 'model.wan_video_model.wan_model.blocks.7.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.2.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.4.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.18.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.3.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.12.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.16.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.23.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.1.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.6.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.6.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.10.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.1.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.9.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.16.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.26.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.7.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.20.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.3.norm3.bias', 'model.wan_video_model.wan_model.blocks.13.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.13.norm3.weight', 'model.wan_video_model.wan_model.blocks.16.modulation', 'model.wan_video_model.wan_model.blocks.7.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.21.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.27.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.18.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.27.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.8.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.15.norm3.bias', 'model.wan_video_model.wan_model.blocks.29.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.1.norm3.weight', 'model.wan_video_model.wan_model.blocks.28.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.29.norm3.bias', 'model.wan_video_model.wan_model.blocks.16.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.14.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.1.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.15.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.9.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.20.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.27.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.8.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.16.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.15.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.1.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.19.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.17.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.21.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.17.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.1.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.20.norm3.bias', 'model.wan_video_model.wan_model.blocks.24.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.9.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.6.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.9.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.19.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.21.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.20.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.13.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.20.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.28.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.11.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.28.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.13.modulation', 'model.wan_video_model.wan_model.blocks.16.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.12.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.7.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.25.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.12.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.2.modulation', 'model.wan_video_model.wan_model.blocks.21.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.18.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.15.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.18.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.28.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.14.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.23.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.29.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.2.cross_attn.q.bias', 'model.wan_video_model.wan_model.time_embedding.2.weight', 'model.wan_video_model.wan_model.blocks.8.ffn.2.weight', 'model.wan_video_model.wan_model.patch_embedding.bias', 'model.wan_video_model.wan_model.blocks.12.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.9.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.17.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.24.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.18.modulation', 'model.wan_video_model.wan_model.blocks.1.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.4.norm3.weight', 'model.wan_video_model.wan_model.blocks.5.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.26.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.26.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.19.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.24.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.16.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.27.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.1.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.8.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.9.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.8.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.21.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.0.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.8.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.1.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.8.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.9.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.11.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.3.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.23.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.27.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.0.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.15.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.4.modulation', 'model.wan_video_model.wan_model.blocks.18.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.4.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.11.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.15.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.24.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.24.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.17.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.23.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.20.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.24.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.0.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.5.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.23.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.1.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.5.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.21.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.2.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.29.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.17.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.5.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.15.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.21.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.8.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.1.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.25.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.12.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.13.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.15.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.23.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.25.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.4.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.6.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.16.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.22.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.28.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.5.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.26.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.0.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.12.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.18.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.19.self_attn.k.bias', 'model.wan_video_model.wan_model.text_embedding.0.bias', 'model.wan_video_model.wan_model.blocks.26.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.20.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.3.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.0.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.7.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.23.norm3.weight', 'model.wan_video_model.wan_model.blocks.22.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.26.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.0.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.21.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.25.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.1.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.23.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.15.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.5.norm3.weight', 'model.wan_video_model.wan_model.blocks.20.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.20.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.10.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.14.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.5.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.17.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.9.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.9.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.18.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.0.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.9.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.10.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.14.norm3.bias', 'model.wan_video_model.wan_model.blocks.26.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.1.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.23.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.16.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.8.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.27.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.11.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.13.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.18.norm3.weight', 'model.wan_video_model.wan_model.blocks.5.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.27.norm3.weight', 'model.wan_video_model.wan_model.blocks.13.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.18.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.16.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.15.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.26.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.27.modulation', 'model.wan_video_model.wan_model.blocks.19.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.29.norm3.weight', 'model.wan_video_model.wan_model.blocks.3.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.18.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.28.modulation', 'model.wan_video_model.wan_model.blocks.6.modulation', 'model.wan_video_model.wan_model.blocks.0.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.7.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.8.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.17.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.7.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.29.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.29.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.10.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.13.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.6.norm3.weight', 'model.wan_video_model.wan_model.blocks.10.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.0.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.10.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.18.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.10.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.21.norm3.weight', 'model.wan_video_model.wan_model.blocks.15.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.28.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.26.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.5.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.17.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.17.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.4.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.13.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.15.norm3.weight', 'model.wan_video_model.wan_model.blocks.16.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.17.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.10.modulation', 'model.wan_video_model.wan_model.blocks.18.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.6.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.19.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.4.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.25.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.13.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.15.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.21.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.12.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.4.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.21.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.15.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.21.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.15.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.27.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.15.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.16.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.19.ffn.0.bias', 'model.wan_video_model.wan_model.time_projection.1.weight', 'model.wan_video_model.wan_model.blocks.16.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.22.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.21.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.27.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.4.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.16.cross_attn.v.bias', 'model.wan_video_model.wan_model.text_embedding.2.weight', 'model.wan_video_model.wan_model.blocks.20.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.22.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.19.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.26.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.0.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.13.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.14.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.11.cross_attn.v.weight', 'model.wan_video_model.wan_model.text_embedding.0.weight', 'model.wan_video_model.wan_model.blocks.27.norm3.bias', 'model.wan_video_model.wan_model.blocks.29.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.6.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.12.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.13.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.19.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.20.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.22.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.19.self_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.22.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.3.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.20.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.3.self_attn.v.weight', 'model.wan_video_model.wan_model.blocks.14.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.29.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.28.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.14.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.1.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.24.cross_attn.q.weight', 'model.wan_video_model.wan_model.blocks.2.norm3.bias', 'model.wan_video_model.wan_model.blocks.6.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.7.norm3.bias', 'model.wan_video_model.wan_model.blocks.8.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.26.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.0.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.2.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.23.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.28.self_attn.q.weight', 'model.wan_video_model.wan_model.blocks.1.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.15.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.5.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.7.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.25.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.26.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.27.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.23.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.22.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.28.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.5.norm3.bias', 'model.wan_video_model.wan_model.blocks.21.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.10.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.17.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.17.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.24.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.19.norm3.weight', 'model.wan_video_model.wan_model.blocks.2.self_attn.v.bias', 'model.wan_video_model.wan_model.blocks.10.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.25.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.22.cross_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.21.cross_attn.o.bias', 'model.wan_video_model.wan_model.blocks.28.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.15.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.28.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.20.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.15.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.0.norm3.weight', 'model.wan_video_model.wan_model.blocks.0.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.29.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.6.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.21.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.9.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.13.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.29.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.20.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.0.modulation', 'model.wan_video_model.wan_model.blocks.18.norm3.bias', 'model.wan_video_model.wan_model.blocks.7.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.4.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.15.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.22.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.3.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.7.cross_attn.o.weight', 'model.wan_video_model.wan_model.blocks.27.self_attn.k.weight', 'model.wan_video_model.wan_model.blocks.26.self_attn.norm_q.weight', 'model.wan_video_model.wan_model.blocks.24.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.27.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.16.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.6.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.19.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.23.norm3.bias', 'model.wan_video_model.wan_model.blocks.16.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.23.cross_attn.norm_k.weight', 'model.wan_video_model.wan_model.blocks.5.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.29.cross_attn.v.bias', 'model.wan_video_model.wan_model.blocks.28.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.14.ffn.0.weight', 'model.wan_video_model.wan_model.time_embedding.2.bias', 'model.wan_video_model.wan_model.blocks.6.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.12.self_attn.o.weight', 'model.wan_video_model.wan_model.blocks.9.norm3.weight', 'model.wan_video_model.wan_model.blocks.29.ffn.2.weight', 'model.wan_video_model.wan_model.blocks.4.ffn.2.bias', 'model.wan_video_model.wan_model.blocks.5.cross_attn.k.bias', 'model.wan_video_model.wan_model.blocks.12.cross_attn.v.weight', 'model.wan_video_model.wan_model.blocks.19.cross_attn.q.bias', 'model.wan_video_model.wan_model.blocks.27.self_attn.k.bias', 'model.wan_video_model.wan_model.blocks.25.self_attn.q.weight', 'model.wan_video_model.wan_model.time_projection.1.bias', 'model.wan_video_model.wan_model.blocks.9.cross_attn.k.weight', 'model.wan_video_model.wan_model.blocks.18.self_attn.o.bias', 'model.wan_video_model.wan_model.blocks.23.ffn.0.bias', 'model.wan_video_model.wan_model.blocks.16.ffn.0.weight', 'model.wan_video_model.wan_model.blocks.3.self_attn.q.bias', 'model.wan_video_model.wan_model.blocks.29.cross_attn.q.bias'}
INFO 2026-08-28 07:54:05 ot_train.py:243 Creating optimizer and scheduler
INFO 2026-08-28 07:54:05 ot_train.py:261 Output dir: /B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/ckpt_2608280751
INFO 2026-08-28 07:54:05 ot_train.py:262 cfg.steps=3640 (4K)
INFO 2026-08-28 07:54:05 ot_train.py:263 [91m[1mnum_frames=6129 (6K)[0m
INFO 2026-08-28 07:54:05 ot_train.py:264 [91m[1mnum_episodes=50 (50)[0m
INFO 2026-08-28 07:54:05 ot_train.py:265 Effective batch size: 16 x 8 = 128
INFO 2026-08-28 07:54:05 ot_train.py:266 policy info:
============================================================
Policy: InternVLAA15Policy

Parameter statistics:
  - Total params        : 8B
  - Trainable params    : 3B
  - Qwen3_5 params      : 2B
  - Action expert params: 460M
  - WAN params          : 5B
  - Learnable tokens    : 50
  - Knowledge insulation: False
  - Inference backend   : standard
  - Freeze WAN DiT      : True
============================================================
INFO 2026-08-28 07:54:06 ot_train.py:329 Start offline training on a fixed dataset
[rank2]:[W828 07:54:16.395338286 reducer.cpp:1500] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[rank6]:[W828 07:54:16.508571457 reducer.cpp:1500] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[rank7]:[W828 07:54:16.521888956 reducer.cpp:1500] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[rank4]:[W828 07:54:16.606856666 reducer.cpp:1500] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[rank0]:[W828 07:54:16.655314840 reducer.cpp:1500] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[rank3]:[W828 07:54:16.706248916 reducer.cpp:1500] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[rank1]:[W828 07:54:16.743785130 reducer.cpp:1500] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
[rank5]:[W828 07:54:17.970077965 reducer.cpp:1500] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
INFO 2026-08-28 07:54:17 __init__.py:187 [FLA Backend] common.chunk_bwd_dqkwg -> tilelang
INFO 2026-08-28 07:55:20 ot_train.py:367  [92m[1m00:01:14 << 01:16:53[0m | [96m[1m0.78 iters/s[0m | step:50.0 | sample:6K | episode:52 | epoch:1.04 | loss:6.137 | loss_action:0.199 | grdn:25.588 | lr:3.6e-06 | updt_s:1.285 | data_s:0.197 | loss_vqa:3.908 | loss_video:0.242 | loss_fast:3.964 | loss_subtask:0.000
INFO 2026-08-28 07:56:20 ot_train.py:367  [92m[1m00:02:14 << 01:03:45[0m | [96m[1m0.93 iters/s[0m | step:100.0 | sample:13K | episode:104 | epoch:2.09 | loss:3.381 | loss_action:0.069 | grdn:10.461 | lr:1.0e-05 | updt_s:1.081 | data_s:0.104 | loss_vqa:2.483 | loss_video:0.204 | loss_fast:2.609 | loss_subtask:0.000
INFO 2026-08-28 07:57:19 ot_train.py:367  [92m[1m00:03:13 << 01:03:20[0m | [96m[1m0.92 iters/s[0m | step:150.0 | sample:19K | episode:157 | epoch:3.13 | loss:2.076 | loss_action:0.028 | grdn:11.482 | lr:1.7e-05 | updt_s:1.089 | data_s:0.097 | loss_vqa:1.623 | loss_video:0.175 | loss_fast:1.734 | loss_subtask:0.000
INFO 2026-08-28 07:58:20 ot_train.py:367  [92m[1m00:04:13 << 01:02:39[0m | [96m[1m0.92 iters/s[0m | step:200.0 | sample:26K | episode:209 | epoch:4.18 | loss:1.580 | loss_action:0.017 | grdn:14.885 | lr:2.4e-05 | updt_s:1.093 | data_s:0.099 | loss_vqa:1.246 | loss_video:0.163 | loss_fast:1.338 | loss_subtask:0.000
INFO 2026-08-28 07:59:20 ot_train.py:367  [92m[1m00:05:14 << 01:01:42[0m | [96m[1m0.92 iters/s[0m | step:250.0 | sample:32K | episode:261 | epoch:5.22 | loss:1.312 | loss_action:0.013 | grdn:16.444 | lr:3.1e-05 | updt_s:1.092 | data_s:0.100 | loss_vqa:1.031 | loss_video:0.151 | loss_fast:1.128 | loss_subtask:0.000
INFO 2026-08-28 08:00:21 ot_train.py:367  [92m[1m00:06:14 << 01:01:40[0m | [96m[1m0.90 iters/s[0m | step:300.0 | sample:38K | episode:313 | epoch:6.27 | loss:1.164 | loss_action:0.011 | grdn:16.824 | lr:3.8e-05 | updt_s:1.108 | data_s:0.098 | loss_vqa:0.908 | loss_video:0.148 | loss_fast:0.994 | loss_subtask:0.000
INFO 2026-08-28 08:01:21 ot_train.py:367  [92m[1m00:07:15 << 01:00:13[0m | [96m[1m0.91 iters/s[0m | step:350.0 | sample:45K | episode:365 | epoch:7.31 | loss:1.126 | loss_action:0.011 | grdn:15.860 | lr:4.5e-05 | updt_s:1.098 | data_s:0.104 | loss_vqa:0.867 | loss_video:0.146 | loss_fast:0.941 | loss_subtask:0.000
INFO 2026-08-28 08:03:21 ot_train.py:367  [92m[1m00:09:15 << 00:58:11[0m | [96m[1m0.91 iters/s[0m | step:450.0 | sample:58K | episode:470 | epoch:9.40 | loss:0.943 | loss_action:0.009 | grdn:13.914 | lr:4.8e-05 | updt_s:1.094 | data_s:0.101 | loss_vqa:0.720 | loss_video:0.134 | loss_fast:0.782 | loss_subtask:0.000
INFO 2026-08-28 08:04:22 ot_train.py:367  [92m[1m00:10:16 << 00:58:05[0m | [96m[1m0.90 iters/s[0m | step:500.0 | sample:64K | episode:522 | epoch:10.44 | loss:0.856 | loss_action:0.008 | grdn:13.470 | lr:4.8e-05 | updt_s:1.110 | data_s:0.093 | loss_vqa:0.644 | loss_video:0.129 | loss_fast:0.715 | loss_subtask:0.000
INFO 2026-08-28 08:05:23 ot_train.py:367  [92m[1m00:11:17 << 00:56:46[0m | [96m[1m0.91 iters/s[0m | step:550.0 | sample:70K | episode:574 | epoch:11.49 | loss:0.790 | loss_action:0.007 | grdn:12.897 | lr:4.8e-05 | updt_s:1.102 | data_s:0.101 | loss_vqa:0.592 | loss_video:0.125 | loss_fast:0.649 | loss_subtask:0.000
INFO 2026-08-28 08:06:23 ot_train.py:367  [92m[1m00:12:17 << 00:56:04[0m | [96m[1m0.90 iters/s[0m | step:600.0 | sample:77K | episode:627 | epoch:12.53 | loss:0.742 | loss_action:0.007 | grdn:12.270 | lr:4.7e-05 | updt_s:1.107 | data_s:0.093 | loss_vqa:0.544 | loss_video:0.127 | loss_fast:0.596 | loss_subtask:0.000
INFO 2026-08-28 08:07:23 ot_train.py:367  [92m[1m00:13:17 << 00:53:57[0m | [96m[1m0.92 iters/s[0m | step:650.0 | sample:83K | episode:679 | epoch:13.57 | loss:0.698 | loss_action:0.007 | grdn:11.796 | lr:4.7e-05 | updt_s:1.083 | data_s:0.104 | loss_vqa:0.506 | loss_video:0.126 | loss_fast:0.549 | loss_subtask:0.000
INFO 2026-08-28 08:08:24 ot_train.py:367  [92m[1m00:14:18 << 00:53:48[0m | [96m[1m0.91 iters/s[0m | step:700.0 | sample:90K | episode:731 | epoch:14.62 | loss:0.644 | loss_action:0.006 | grdn:11.958 | lr:4.6e-05 | updt_s:1.098 | data_s:0.102 | loss_vqa:0.459 | loss_video:0.125 | loss_fast:0.506 | loss_subtask:0.000
INFO 2026-08-28 08:09:26 ot_train.py:367  [92m[1m00:15:20 << 00:54:24[0m | [96m[1m0.89 iters/s[0m | step:750.0 | sample:96K | episode:783 | epoch:15.66 | loss:0.600 | loss_action:0.005 | grdn:11.904 | lr:4.6e-05 | updt_s:1.130 | data_s:0.100 | loss_vqa:0.427 | loss_video:0.121 | loss_fast:0.469 | loss_subtask:0.000
INFO 2026-08-28 08:10:27 ot_train.py:367  [92m[1m00:16:21 << 00:53:00[0m | [96m[1m0.89 iters/s[0m | step:800.0 | sample:102K | episode:835 | epoch:16.71 | loss:0.591 | loss_action:0.006 | grdn:11.564 | lr:4.5e-05 | updt_s:1.120 | data_s:0.098 | loss_vqa:0.411 | loss_video:0.122 | loss_fast:0.451 | loss_subtask:0.000
INFO 2026-08-28 08:11:28 ot_train.py:367  [92m[1m00:17:22 << 00:51:32[0m | [96m[1m0.90 iters/s[0m | step:850.0 | sample:109K | episode:888 | epoch:17.75 | loss:0.542 | loss_action:0.005 | grdn:11.012 | lr:4.5e-05 | updt_s:1.108 | data_s:0.099 | loss_vqa:0.374 | loss_video:0.117 | loss_fast:0.413 | loss_subtask:0.000
INFO 2026-08-28 08:12:29 ot_train.py:367  [92m[1m00:18:23 << 00:50:26[0m | [96m[1m0.91 iters/s[0m | step:900.0 | sample:115K | episode:940 | epoch:18.80 | loss:0.532 | loss_action:0.006 | grdn:10.351 | lr:4.4e-05 | updt_s:1.105 | data_s:0.095 | loss_vqa:0.352 | loss_video:0.122 | loss_fast:0.389 | loss_subtask:0.000
INFO 2026-08-28 08:12:39 ot_train.py:377 Checkpoint policy after step 910
INFO 2026-08-28 08:12:39 ot_train.py:379 Checkpoint saved at: /B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/ckpt_2608280751/checkpoints/000910
INFO 2026-08-28 08:13:57 ot_train.py:367  [92m[1m00:19:51 << 00:50:24[0m | [96m[1m0.89 iters/s[0m | step:950.0 | sample:122K | episode:992 | epoch:19.84 | loss:0.488 | loss_action:0.005 | grdn:10.027 | lr:4.3e-05 | updt_s:1.124 | data_s:0.096 | loss_vqa:0.321 | loss_video:0.115 | loss_fast:0.358 | loss_subtask:0.000
INFO 2026-08-28 08:14:58 ot_train.py:367  [92m[1m00:20:52 << 00:48:59[0m | [96m[1m0.90 iters/s[0m | step:1.0K | sample:128K | episode:1K | epoch:20.88 | loss:0.447 | loss_action:0.005 | grdn:9.240 | lr:4.2e-05 | updt_s:1.113 | data_s:0.097 | loss_vqa:0.285 | loss_video:0.114 | loss_fast:0.313 | loss_subtask:0.000
INFO 2026-08-28 08:15:59 ot_train.py:367  [92m[1m00:21:53 << 00:47:37[0m | [96m[1m0.91 iters/s[0m | step:1.1K | sample:134K | episode:1K | epoch:21.93 | loss:0.440 | loss_action:0.004 | grdn:9.014 | lr:4.2e-05 | updt_s:1.103 | data_s:0.099 | loss_vqa:0.282 | loss_video:0.116 | loss_fast:0.312 | loss_subtask:0.000
INFO 2026-08-28 08:17:00 ot_train.py:367  [92m[1m00:22:54 << 00:46:45[0m | [96m[1m0.91 iters/s[0m | step:1.1K | sample:141K | episode:1K | epoch:22.97 | loss:0.415 | loss_action:0.004 | grdn:8.181 | lr:4.1e-05 | updt_s:1.105 | data_s:0.096 | loss_vqa:0.256 | loss_video:0.114 | loss_fast:0.282 | loss_subtask:0.000
INFO 2026-08-28 08:18:00 ot_train.py:367  [92m[1m00:23:54 << 00:45:44[0m | [96m[1m0.91 iters/s[0m | step:1.1K | sample:147K | episode:1K | epoch:24.02 | loss:0.386 | loss_action:0.004 | grdn:7.903 | lr:4.0e-05 | updt_s:1.102 | data_s:0.103 | loss_vqa:0.237 | loss_video:0.113 | loss_fast:0.266 | loss_subtask:0.000
INFO 2026-08-28 08:19:03 ot_train.py:367  [92m[1m00:24:57 << 00:45:52[0m | [96m[1m0.89 iters/s[0m | step:1.2K | sample:154K | episode:1K | epoch:25.06 | loss:0.386 | loss_action:0.005 | grdn:8.361 | lr:3.9e-05 | updt_s:1.128 | data_s:0.107 | loss_vqa:0.223 | loss_video:0.118 | loss_fast:0.242 | loss_subtask:0.000
INFO 2026-08-28 08:20:09 ot_train.py:367  [92m[1m00:26:03 << 00:45:59[0m | [96m[1m0.87 iters/s[0m | step:1.2K | sample:160K | episode:1K | epoch:26.11 | loss:0.357 | loss_action:0.004 | grdn:7.669 | lr:3.9e-05 | updt_s:1.155 | data_s:0.172 | loss_vqa:0.208 | loss_video:0.108 | loss_fast:0.229 | loss_subtask:0.000
INFO 2026-08-28 08:21:11 ot_train.py:367  [92m[1m00:27:05 << 00:43:39[0m | [96m[1m0.89 iters/s[0m | step:1.3K | sample:166K | episode:1K | epoch:27.15 | loss:0.340 | loss_action:0.004 | grdn:7.773 | lr:3.8e-05 | updt_s:1.120 | data_s:0.103 | loss_vqa:0.189 | loss_video:0.114 | loss_fast:0.210 | loss_subtask:0.000
INFO 2026-08-28 08:22:13 ot_train.py:367  [92m[1m00:28:07 << 00:42:45[0m | [96m[1m0.89 iters/s[0m | step:1.4K | sample:173K | episode:1K | epoch:28.19 | loss:0.321 | loss_action:0.003 | grdn:7.629 | lr:3.7e-05 | updt_s:1.120 | data_s:0.103 | loss_vqa:0.179 | loss_video:0.110 | loss_fast:0.199 | loss_subtask:0.000
INFO 2026-08-28 08:23:15 ot_train.py:367  [92m[1m00:29:09 << 00:42:13[0m | [96m[1m0.88 iters/s[0m | step:1.4K | sample:179K | episode:1K | epoch:29.24 | loss:0.311 | loss_action:0.004 | grdn:7.169 | lr:3.6e-05 | updt_s:1.131 | data_s:0.097 | loss_vqa:0.160 | loss_video:0.112 | loss_fast:0.180 | loss_subtask:0.000
INFO 2026-08-28 08:24:17 ot_train.py:367  [92m[1m00:30:11 << 00:41:05[0m | [96m[1m0.89 iters/s[0m | step:1.4K | sample:186K | episode:2K | epoch:30.28 | loss:0.293 | loss_action:0.004 | grdn:6.892 | lr:3.5e-05 | updt_s:1.126 | data_s:0.099 | loss_vqa:0.146 | loss_video:0.111 | loss_fast:0.163 | loss_subtask:0.000
INFO 2026-08-28 08:25:18 ot_train.py:367  [92m[1m00:31:12 << 00:40:14[0m | [96m[1m0.89 iters/s[0m | step:1.5K | sample:192K | episode:2K | epoch:31.33 | loss:0.286 | loss_action:0.003 | grdn:6.765 | lr:3.4e-05 | updt_s:1.128 | data_s:0.100 | loss_vqa:0.142 | loss_video:0.112 | loss_fast:0.158 | loss_subtask:0.000
INFO 2026-08-28 08:26:20 ot_train.py:367  [92m[1m00:32:14 << 00:38:59[0m | [96m[1m0.89 iters/s[0m | step:1.6K | sample:198K | episode:2K | epoch:32.37 | loss:0.257 | loss_action:0.003 | grdn:6.386 | lr:3.3e-05 | updt_s:1.119 | data_s:0.097 | loss_vqa:0.125 | loss_video:0.101 | loss_fast:0.139 | loss_subtask:0.000
INFO 2026-08-28 08:27:20 ot_train.py:367  [92m[1m00:33:14 << 00:37:10[0m | [96m[1m0.91 iters/s[0m | step:1.6K | sample:205K | episode:2K | epoch:33.41 | loss:0.266 | loss_action:0.004 | grdn:6.298 | lr:3.2e-05 | updt_s:1.093 | data_s:0.101 | loss_vqa:0.117 | loss_video:0.111 | loss_fast:0.130 | loss_subtask:0.000
INFO 2026-08-28 08:28:22 ot_train.py:367  [92m[1m00:34:16 << 00:37:27[0m | [96m[1m0.89 iters/s[0m | step:1.6K | sample:211K | episode:2K | epoch:34.46 | loss:0.247 | loss_action:0.003 | grdn:5.983 | lr:3.1e-05 | updt_s:1.129 | data_s:0.089 | loss_vqa:0.107 | loss_video:0.106 | loss_fast:0.118 | loss_subtask:0.000
INFO 2026-08-28 08:29:23 ot_train.py:367  [92m[1m00:35:17 << 00:36:06[0m | [96m[1m0.90 iters/s[0m | step:1.7K | sample:218K | episode:2K | epoch:35.50 | loss:0.244 | loss_action:0.004 | grdn:5.611 | lr:3.0e-05 | updt_s:1.117 | data_s:0.094 | loss_vqa:0.102 | loss_video:0.107 | loss_fast:0.113 | loss_subtask:0.000
INFO 2026-08-28 08:30:25 ot_train.py:367  [92m[1m00:36:19 << 00:35:49[0m | [96m[1m0.88 iters/s[0m | step:1.8K | sample:224K | episode:2K | epoch:36.55 | loss:0.236 | loss_action:0.003 | grdn:6.026 | lr:2.9e-05 | updt_s:1.137 | data_s:0.093 | loss_vqa:0.095 | loss_video:0.110 | loss_fast:0.104 | loss_subtask:0.000
INFO 2026-08-28 08:31:26 ot_train.py:367  [92m[1m00:37:20 << 00:34:11[0m | [96m[1m0.90 iters/s[0m | step:1.8K | sample:230K | episode:2K | epoch:37.59 | loss:0.216 | loss_action:0.003 | grdn:5.366 | lr:2.8e-05 | updt_s:1.115 | data_s:0.102 | loss_vqa:0.079 | loss_video:0.107 | loss_fast:0.088 | loss_subtask:0.000
INFO 2026-08-28 08:31:48 ot_train.py:377 Checkpoint policy after step 1820
INFO 2026-08-28 08:31:48 ot_train.py:379 Checkpoint saved at: /B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/ckpt_2608280751/checkpoints/001820
INFO 2026-08-28 08:32:54 ot_train.py:367  [92m[1m00:38:48 << 00:33:42[0m | [96m[1m0.88 iters/s[0m | step:1.9K | sample:237K | episode:2K | epoch:38.64 | loss:0.203 | loss_action:0.002 | grdn:5.221 | lr:2.7e-05 | updt_s:1.130 | data_s:0.098 | loss_vqa:0.074 | loss_video:0.105 | loss_fast:0.083 | loss_subtask:0.000
INFO 2026-08-28 08:33:57 ot_train.py:367  [92m[1m00:39:50 << 00:32:58[0m | [96m[1m0.88 iters/s[0m | step:1.9K | sample:243K | episode:2K | epoch:39.68 | loss:0.212 | loss_action:0.004 | grdn:4.790 | lr:2.6e-05 | updt_s:1.137 | data_s:0.100 | loss_vqa:0.064 | loss_video:0.108 | loss_fast:0.072 | loss_subtask:0.000
INFO 2026-08-28 08:34:58 ot_train.py:367  [92m[1m00:40:52 << 00:31:35[0m | [96m[1m0.89 iters/s[0m | step:1.9K | sample:250K | episode:2K | epoch:40.72 | loss:0.192 | loss_action:0.002 | grdn:4.846 | lr:2.5e-05 | updt_s:1.122 | data_s:0.105 | loss_vqa:0.061 | loss_video:0.106 | loss_fast:0.068 | loss_subtask:0.000
INFO 2026-08-28 08:36:00 ot_train.py:367  [92m[1m00:41:54 << 00:30:34[0m | [96m[1m0.89 iters/s[0m | step:2.0K | sample:256K | episode:2K | epoch:41.77 | loss:0.182 | loss_action:0.003 | grdn:4.646 | lr:2.4e-05 | updt_s:1.119 | data_s:0.103 | loss_vqa:0.054 | loss_video:0.102 | loss_fast:0.061 | loss_subtask:0.000
INFO 2026-08-28 08:37:01 ot_train.py:367  [92m[1m00:42:55 << 00:29:35[0m | [96m[1m0.90 iters/s[0m | step:2.0K | sample:262K | episode:2K | epoch:42.81 | loss:0.183 | loss_action:0.003 | grdn:4.353 | lr:2.4e-05 | updt_s:1.117 | data_s:0.098 | loss_vqa:0.051 | loss_video:0.106 | loss_fast:0.057 | loss_subtask:0.000
INFO 2026-08-28 08:38:02 ot_train.py:367  [92m[1m00:43:56 << 00:28:34[0m | [96m[1m0.90 iters/s[0m | step:2.1K | sample:269K | episode:2K | epoch:43.86 | loss:0.178 | loss_action:0.003 | grdn:4.080 | lr:2.3e-05 | updt_s:1.114 | data_s:0.102 | loss_vqa:0.046 | loss_video:0.107 | loss_fast:0.051 | loss_subtask:0.000
INFO 2026-08-28 08:39:04 ot_train.py:367  [92m[1m00:44:58 << 00:28:07[0m | [96m[1m0.88 iters/s[0m | step:2.1K | sample:275K | episode:2K | epoch:44.90 | loss:0.178 | loss_action:0.003 | grdn:4.034 | lr:2.2e-05 | updt_s:1.133 | data_s:0.098 | loss_vqa:0.042 | loss_video:0.111 | loss_fast:0.046 | loss_subtask:0.000
INFO 2026-08-28 08:40:06 ot_train.py:367  [92m[1m00:46:00 << 00:26:52[0m | [96m[1m0.89 iters/s[0m | step:2.2K | sample:282K | episode:2K | epoch:45.95 | loss:0.167 | loss_action:0.002 | grdn:3.679 | lr:2.1e-05 | updt_s:1.120 | data_s:0.099 | loss_vqa:0.036 | loss_video:0.106 | loss_fast:0.041 | loss_subtask:0.000
INFO 2026-08-28 08:41:07 ot_train.py:367  [92m[1m00:47:01 << 00:25:29[0m | [96m[1m0.91 iters/s[0m | step:2.2K | sample:288K | episode:2K | epoch:46.99 | loss:0.160 | loss_action:0.003 | grdn:3.596 | lr:2.0e-05 | updt_s:1.100 | data_s:0.110 | loss_vqa:0.034 | loss_video:0.100 | loss_fast:0.038 | loss_subtask:0.000
INFO 2026-08-28 08:42:07 ot_train.py:367  [92m[1m00:48:01 << 00:24:42[0m | [96m[1m0.90 iters/s[0m | step:2.3K | sample:294K | episode:2K | epoch:48.03 | loss:0.161 | loss_action:0.002 | grdn:3.426 | lr:1.9e-05 | updt_s:1.106 | data_s:0.090 | loss_vqa:0.030 | loss_video:0.107 | loss_fast:0.034 | loss_subtask:0.000
INFO 2026-08-28 08:43:09 ot_train.py:367  [92m[1m00:49:03 << 00:24:29[0m | [96m[1m0.88 iters/s[0m | step:2.4K | sample:301K | episode:2K | epoch:49.08 | loss:0.150 | loss_action:0.002 | grdn:3.004 | lr:1.8e-05 | updt_s:1.139 | data_s:0.082 | loss_vqa:0.027 | loss_video:0.103 | loss_fast:0.031 | loss_subtask:0.000
INFO 2026-08-28 08:44:10 ot_train.py:367  [92m[1m00:50:04 << 00:23:31[0m | [96m[1m0.88 iters/s[0m | step:2.4K | sample:307K | episode:3K | epoch:50.12 | loss:0.153 | loss_action:0.003 | grdn:2.905 | lr:1.7e-05 | updt_s:1.138 | data_s:0.088 | loss_vqa:0.025 | loss_video:0.102 | loss_fast:0.028 | loss_subtask:0.000
INFO 2026-08-28 08:45:17 ot_train.py:367  [92m[1m00:51:11 << 00:22:50[0m | [96m[1m0.87 iters/s[0m | step:2.5K | sample:314K | episode:3K | epoch:51.17 | loss:0.146 | loss_action:0.002 | grdn:2.951 | lr:1.6e-05 | updt_s:1.152 | data_s:0.162 | loss_vqa:0.022 | loss_video:0.106 | loss_fast:0.026 | loss_subtask:0.000
INFO 2026-08-28 08:46:18 ot_train.py:367  [92m[1m00:52:12 << 00:21:23[0m | [96m[1m0.89 iters/s[0m | step:2.5K | sample:320K | episode:3K | epoch:52.21 | loss:0.154 | loss_action:0.003 | grdn:2.710 | lr:1.5e-05 | updt_s:1.126 | data_s:0.087 | loss_vqa:0.021 | loss_video:0.108 | loss_fast:0.023 | loss_subtask:0.000
INFO 2026-08-28 08:47:19 ot_train.py:367  [92m[1m00:53:13 << 00:20:19[0m | [96m[1m0.89 iters/s[0m | step:2.5K | sample:326K | episode:3K | epoch:53.26 | loss:0.143 | loss_action:0.002 | grdn:2.743 | lr:1.5e-05 | updt_s:1.119 | data_s:0.085 | loss_vqa:0.018 | loss_video:0.103 | loss_fast:0.020 | loss_subtask:0.000
INFO 2026-08-28 08:48:20 ot_train.py:367  [92m[1m00:54:14 << 00:19:53[0m | [96m[1m0.87 iters/s[0m | step:2.6K | sample:333K | episode:3K | epoch:54.30 | loss:0.139 | loss_action:0.002 | grdn:2.461 | lr:1.4e-05 | updt_s:1.147 | data_s:0.079 | loss_vqa:0.017 | loss_video:0.105 | loss_fast:0.020 | loss_subtask:0.000
INFO 2026-08-28 08:49:21 ot_train.py:367  [92m[1m00:55:15 << 00:18:30[0m | [96m[1m0.89 iters/s[0m | step:2.6K | sample:339K | episode:3K | epoch:55.34 | loss:0.142 | loss_action:0.002 | grdn:2.483 | lr:1.3e-05 | updt_s:1.122 | data_s:0.081 | loss_vqa:0.016 | loss_video:0.105 | loss_fast:0.018 | loss_subtask:0.000
INFO 2026-08-28 08:50:23 ot_train.py:367  [92m[1m00:56:17 << 00:17:38[0m | [96m[1m0.89 iters/s[0m | step:2.7K | sample:346K | episode:3K | epoch:56.39 | loss:0.135 | loss_action:0.002 | grdn:2.167 | lr:1.2e-05 | updt_s:1.126 | data_s:0.093 | loss_vqa:0.015 | loss_video:0.103 | loss_fast:0.016 | loss_subtask:0.000
INFO 2026-08-28 08:50:55 ot_train.py:377 Checkpoint policy after step 2730
INFO 2026-08-28 08:50:55 ot_train.py:379 Checkpoint saved at: /B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/ckpt_2608280751/checkpoints/002730
INFO 2026-08-28 08:51:47 ot_train.py:367  [92m[1m00:57:41 << 00:16:31[0m | [96m[1m0.90 iters/s[0m | step:2.8K | sample:352K | episode:3K | epoch:57.43 | loss:0.132 | loss_action:0.002 | grdn:2.473 | lr:1.2e-05 | updt_s:1.114 | data_s:0.096 | loss_vqa:0.015 | loss_video:0.101 | loss_fast:0.017 | loss_subtask:0.000
INFO 2026-08-28 08:52:48 ot_train.py:367  [92m[1m00:58:42 << 00:15:18[0m | [96m[1m0.91 iters/s[0m | step:2.8K | sample:358K | episode:3K | epoch:58.48 | loss:0.135 | loss_action:0.002 | grdn:1.969 | lr:1.1e-05 | updt_s:1.093 | data_s:0.098 | loss_vqa:0.013 | loss_video:0.105 | loss_fast:0.015 | loss_subtask:0.000
INFO 2026-08-28 08:53:49 ot_train.py:367  [92m[1m00:59:43 << 00:14:45[0m | [96m[1m0.89 iters/s[0m | step:2.9K | sample:365K | episode:3K | epoch:59.52 | loss:0.136 | loss_action:0.002 | grdn:1.861 | lr:1.0e-05 | updt_s:1.121 | data_s:0.099 | loss_vqa:0.012 | loss_video:0.109 | loss_fast:0.014 | loss_subtask:0.000
INFO 2026-08-28 08:54:51 ot_train.py:367  [92m[1m01:00:45 << 00:13:55[0m | [96m[1m0.89 iters/s[0m | step:2.9K | sample:371K | episode:3K | epoch:60.56 | loss:0.135 | loss_action:0.002 | grdn:1.889 | lr:9.7e-06 | updt_s:1.130 | data_s:0.103 | loss_vqa:0.011 | loss_video:0.104 | loss_fast:0.013 | loss_subtask:0.000
INFO 2026-08-28 08:55:53 ot_train.py:367  [92m[1m01:01:47 << 00:12:58[0m | [96m[1m0.89 iters/s[0m | step:3.0K | sample:378K | episode:3K | epoch:61.61 | loss:0.138 | loss_action:0.002 | grdn:1.887 | lr:9.1e-06 | updt_s:1.128 | data_s:0.103 | loss_vqa:0.011 | loss_video:0.109 | loss_fast:0.012 | loss_subtask:0.000
INFO 2026-08-28 08:56:54 ot_train.py:367  [92m[1m01:02:48 << 00:11:47[0m | [96m[1m0.90 iters/s[0m | step:3.0K | sample:384K | episode:3K | epoch:62.65 | loss:0.135 | loss_action:0.002 | grdn:1.636 | lr:8.6e-06 | updt_s:1.105 | data_s:0.101 | loss_vqa:0.011 | loss_video:0.106 | loss_fast:0.012 | loss_subtask:0.000
INFO 2026-08-28 08:57:55 ot_train.py:367  [92m[1m01:03:49 << 00:10:55[0m | [96m[1m0.90 iters/s[0m | step:3.0K | sample:390K | episode:3K | epoch:63.70 | loss:0.132 | loss_action:0.002 | grdn:1.530 | lr:8.1e-06 | updt_s:1.111 | data_s:0.095 | loss_vqa:0.010 | loss_video:0.105 | loss_fast:0.011 | loss_subtask:0.000
INFO 2026-08-28 08:58:57 ot_train.py:367  [92m[1m01:04:51 << 00:10:06[0m | [96m[1m0.89 iters/s[0m | step:3.1K | sample:397K | episode:3K | epoch:64.74 | loss:0.135 | loss_action:0.002 | grdn:1.656 | lr:7.6e-06 | updt_s:1.122 | data_s:0.100 | loss_vqa:0.010 | loss_video:0.105 | loss_fast:0.011 | loss_subtask:0.000
INFO 2026-08-28 08:59:59 ot_train.py:367  [92m[1m01:05:53 << 00:09:15[0m | [96m[1m0.88 iters/s[0m | step:3.1K | sample:403K | episode:3K | epoch:65.79 | loss:0.123 | loss_action:0.001 | grdn:1.469 | lr:7.2e-06 | updt_s:1.134 | data_s:0.097 | loss_vqa:0.009 | loss_video:0.100 | loss_fast:0.010 | loss_subtask:0.000
INFO 2026-08-28 09:01:01 ot_train.py:367  [92m[1m01:06:55 << 00:08:17[0m | [96m[1m0.88 iters/s[0m | step:3.2K | sample:410K | episode:3K | epoch:66.83 | loss:0.130 | loss_action:0.002 | grdn:1.538 | lr:6.8e-06 | updt_s:1.131 | data_s:0.101 | loss_vqa:0.009 | loss_video:0.104 | loss_fast:0.010 | loss_subtask:0.000
INFO 2026-08-28 09:02:01 ot_train.py:367  [92m[1m01:07:55 << 00:07:11[0m | [96m[1m0.90 iters/s[0m | step:3.2K | sample:416K | episode:3K | epoch:67.87 | loss:0.128 | loss_action:0.002 | grdn:1.528 | lr:6.4e-06 | updt_s:1.106 | data_s:0.104 | loss_vqa:0.009 | loss_video:0.102 | loss_fast:0.010 | loss_subtask:0.000
INFO 2026-08-28 09:03:03 ot_train.py:367  [92m[1m01:08:57 << 00:06:21[0m | [96m[1m0.89 iters/s[0m | step:3.3K | sample:422K | episode:3K | epoch:68.92 | loss:0.131 | loss_action:0.002 | grdn:1.388 | lr:6.1e-06 | updt_s:1.122 | data_s:0.100 | loss_vqa:0.008 | loss_video:0.105 | loss_fast:0.009 | loss_subtask:0.000
INFO 2026-08-28 09:04:05 ot_train.py:367  [92m[1m01:09:59 << 00:05:27[0m | [96m[1m0.88 iters/s[0m | step:3.4K | sample:429K | episode:3K | epoch:69.96 | loss:0.124 | loss_action:0.002 | grdn:1.492 | lr:5.8e-06 | updt_s:1.130 | data_s:0.095 | loss_vqa:0.008 | loss_video:0.100 | loss_fast:0.009 | loss_subtask:0.000
INFO 2026-08-28 09:05:06 ot_train.py:367  [92m[1m01:11:00 << 00:04:26[0m | [96m[1m0.90 iters/s[0m | step:3.4K | sample:435K | episode:4K | epoch:71.01 | loss:0.126 | loss_action:0.001 | grdn:1.347 | lr:5.6e-06 | updt_s:1.112 | data_s:0.100 | loss_vqa:0.008 | loss_video:0.104 | loss_fast:0.009 | loss_subtask:0.000
INFO 2026-08-28 09:06:08 ot_train.py:367  [92m[1m01:12:02 << 00:03:33[0m | [96m[1m0.89 iters/s[0m | step:3.5K | sample:442K | episode:4K | epoch:72.05 | loss:0.127 | loss_action:0.001 | grdn:1.338 | lr:5.4e-06 | updt_s:1.121 | data_s:0.103 | loss_vqa:0.008 | loss_video:0.107 | loss_fast:0.009 | loss_subtask:0.000
INFO 2026-08-28 09:07:09 ot_train.py:367  [92m[1m01:13:03 << 00:02:33[0m | [96m[1m0.91 iters/s[0m | step:3.5K | sample:448K | episode:4K | epoch:73.10 | loss:0.131 | loss_action:0.002 | grdn:1.428 | lr:5.2e-06 | updt_s:1.097 | data_s:0.117 | loss_vqa:0.008 | loss_video:0.105 | loss_fast:0.009 | loss_subtask:0.000
INFO 2026-08-28 09:08:10 ot_train.py:367  [92m[1m01:14:04 << 00:01:38[0m | [96m[1m0.91 iters/s[0m | step:3.5K | sample:454K | episode:4K | epoch:74.14 | loss:0.128 | loss_action:0.001 | grdn:1.274 | lr:5.1e-06 | updt_s:1.098 | data_s:0.115 | loss_vqa:0.008 | loss_video:0.107 | loss_fast:0.009 | loss_subtask:0.000
INFO 2026-08-28 09:09:11 ot_train.py:367  [92m[1m01:15:05 << 00:00:43[0m | [96m[1m0.91 iters/s[0m | step:3.6K | sample:461K | episode:4K | epoch:75.18 | loss:0.134 | loss_action:0.002 | grdn:1.385 | lr:5.0e-06 | updt_s:1.099 | data_s:0.114 | loss_vqa:0.008 | loss_video:0.108 | loss_fast:0.009 | loss_subtask:0.000
INFO 2026-08-28 09:10:01 ot_train.py:377 Checkpoint policy after step 3640
INFO 2026-08-28 09:10:01 ot_train.py:379 Checkpoint saved at: /B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/ckpt_2608280751/checkpoints/003640
INFO 2026-08-28 09:10:31 ot_train.py:396 End of training
2026-08-28T09:15:45Z
--- completion markers ---
413:INFO 2026-08-28 08:12:39 ot_train.py:379 Checkpoint saved at: /B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/ckpt_2608280751/checkpoints/000910
433:INFO 2026-08-28 08:31:48 ot_train.py:379 Checkpoint saved at: /B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/ckpt_2608280751/checkpoints/001820
453:INFO 2026-08-28 08:50:55 ot_train.py:379 Checkpoint saved at: /B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/ckpt_2608280751/checkpoints/002730
473:INFO 2026-08-28 09:10:01 ot_train.py:379 Checkpoint saved at: /B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/ckpt_2608280751/checkpoints/003640
474:INFO 2026-08-28 09:10:31 ot_train.py:396 End of training
--- checkpoint validation ---
000910 keys= 950 policy= internvla_a1_5 stats= ['aloha']
001820 keys= 950 policy= internvla_a1_5 stats= ['aloha']
002730 keys= 950 policy= internvla_a1_5 stats= ['aloha']
003640 keys= 950 policy= internvla_a1_5 stats= ['aloha']
last -> /B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/ckpt_2608280751/checkpoints/003640
CHECKPOINT_VALIDATION_OK
--- disk usage ---
61G	/B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles

## 3. 正式训练完成总结

### 3.1 最终结果

正式训练于 2026-08-28 09:10 UTC（17:10 UTC+8）完成，训练进程 exit code 为 0，
日志出现 `End of training`。

| 项目 | 结果 |
|---|---|
| 任务 | `pick_dual_bottles` |
| 数据 | 50 episodes / 6129 frames / 3 路相机 |
| 数据版本 | LeRobot `v3.0` |
| robot type | `aloha` |
| GPU | 8× NVIDIA H200 |
| global batch | 128 |
| per-GPU batch | 16 |
| 训练计划 | 76 epochs |
| 实际总 step | 3640 |
| save frequency | 910 |
| warmup steps | 364 |
| 训练耗时 | 约 1 小时 18 分钟 |
| 最后记录的 step | 3600，loss=0.134，loss_action=0.002 |
| 训练输出根 | `/B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles` |
| 最终 checkpoint | `ckpt_2608280751/checkpoints/003640/pretrained_model` |
| `last` | 指向 `003640` |

### 3.2 Checkpoint 验证结果

以下 checkpoint 均存在 `config.json`、`model.safetensors` 和 `stats.json`，每个
模型包含 950 个 safetensors keys，policy type 为 `internvla_a1_5`，stats 顶层
包含 `aloha`：

```text
000910  OK
001820  OK
002730  OK
003640  OK
last -> 003640  OK
```

### 3.3 错误、警告和修复记录

本次数据准备、冒烟和正式训练没有发生需要修复的 fatal error。正式训练过程中没有：

- CUDA OOM；
- NCCL tuner error；
- `Traceback`；
- NaN；
- `video_decode_error`。

出现但不影响训练的 warning：

1. HuggingFace Hub 未认证 warning；本地缓存可用，未阻断加载；
2. accelerate 对 `mixed_precision`、`dynamo_backend` 使用默认值的提示；
3. DDP `find_unused_parameters=True` 性能提示；
4. base checkpoint 缺少 WAN key 的提示；WAN 从独立 `Wan2.2-TI2V-5B` 路径加载，
   且 WAN DiT 按配置冻结；
5. PyArrow `promote` FutureWarning；
6. HuggingFace Hub extras warning。

### 3.4 文件、目录和配置变更清单

| 路径 | 操作 | 原因 |
|---|---|---|
| `b/d/p/reprd_rbtwn_pikDulBtlLOG.md` | 新增并持续追加 | 按发生顺序保存本次完整执行记录 |
| `/B/VENV/itnvla15rbt20` | 执行 `pip install -e /B/SRC/InternVLA-A-series` | 确保指定 venv 使用当前 checkout |
| `/B/Dta/RoboTwin-Clean/pick_dual_bottles_lrb3` | 重新生成 v3.0 转换结果 | 本次准备命令未设置 `SKIP_CONVERT=1`；源目录仍未修改 |
| `/B/VENV/itnvla15rbt20/var/hf_home/lerobot/robotwin/pick_dual_bottles` | 更新 symlink | 使 `robotwin/pick_dual_bottles` 解析到 v3.0 数据 |
| `/B/VENV/itnvla15rbt20/var/hf_home/lerobot/stats/aloha/abs/agg_1repos_59c5e8f4cd/stats.json` | 新增/重算 | 为 abs 训练提供 external stats |
| `/B/Ckp/itnVla_2608280746/rbt2/pick_dual_bottles/` | 新增冒烟产物 | 验证 4-step 训练和 checkpoint 写入 |
| `/B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/` | 新增正式训练产物 | 保存正式日志、wandb、训练状态和四个 checkpoint |

本次没有修改仓库中的 Python、Shell 训练实现；使用了已有的通用入口：

```text
launch/internvla_a15_prepare_robotwin.sh
launch/internvla_a15_finetune_robotwin_comm.sh
launch/internvla_a15_robotwin_common.sh
```

### 3.5 关键产物路径

```text
训练日志：
/B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/train_2608280751.log

运行参数：
/B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/run_2608280751.env

最终权重：
/B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/ckpt_2608280751/checkpoints/003640/pretrained_model/

稳定链接：
/B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles/ckpt_2608280751/checkpoints/last/pretrained_model/
```

### 3.6 结论

`pick_dual_bottles` 已按手册完成 76 epoch、3640 step 的正式微调训练。
训练数据、external stats、四个阶段 checkpoint、最终权重和运行日志均已生成并通过校验。
RoboTwin closed-loop 评测尚未执行，若需要评测，应按操作手册第 10 节使用 task index 19。

## 4. 训练结束后的归档与 GPU 释放

### 4.1 成功状态复核

2026-08-28 17:16（UTC+8）再次复核：

```text
训练日志：包含 End of training
训练进程：不存在
GPU：8 张 H200 均为 0 MiB used、0% utilization
输出目录大小：61G
```

因此按成功分支归档，不执行错误目录归档。

### 4.2 成功产物上传

目标 bucket：

```text
gs://physical-ai-data-eu/VENV/tmp/Rbt2PikDulBtl0828/
```

上传对象为整个训练输出目录（包含训练日志、运行参数、WandB offline 数据、
全部 checkpoint 和 training state）：

```bash
gsutil -m cp -r \
  /B/Ckp/itnVla_2608280751/rbt2/pick_dual_bottles \
  gs://physical-ai-data-eu/VENV/tmp/Rbt2PikDulBtl0828/
```

上传状态：成功。云端成功列出 47 个文件，包含 60.1 GiB 训练产物。

### 4.3 GPU 清理与显存填充

归档完成后复查无训练进程，8 张 GPU 均为 0 MiB used。随后清理可见 CUDA
compute 进程并启动：

```bash
source /B/VENV/itnvla15rbt20/bin/activate
python /B/SRC/InternVLA-A-series/b/d/rbt/fill_8gpu_vram.py
```

首次启动和一次直接重启均因 PyTorch 多线程 CUDA Graph 初始化报错退出：

```text
CUDA error: operation not permitted when stream is capturing
CUDA error: operation failed due to a previous error during capture
```

根因为该辅助脚本在多卡并发初始化时，部分 CUDA stream capture 不被当前
PyTorch/CUDA 组合接受，并非训练产物或训练进程错误。已修改
`b/d/rbt/fill_8gpu_vram.py`：捕获该特定 RuntimeError 后回退到普通
bf16 matmul，同时保留显存 holder。

修复后再次启动成功，单一 Python 进程已在 8 张 H200 上分别占用约
102586 MiB（约 71.7% 显存）；日志显示 `GPU VRAM + compute held (low CPU)`。
采样时个别 GPU 利用率为 0% 是脚本 70%--90% 随机 duty cycle 的正常空闲窗口。
