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
Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 75.24it/s]
Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 71.97it/s]
Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 71.06it/s]
INFO 2026-08-28 07:53:52 wan_model.py:53 WAN Video Model initialized with 4,999,787,712 parameters
INFO 2026-08-28 07:53:52 an_model.py:118 Loading WAN weights from /B/VENV/itnvla15rbt20/var/hf_home/hub/Wan2.2-TI2V-5B
Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 72.90it/s]
Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 77.43it/s]
Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 76.44it/s]
Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 71.47it/s]
Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]Loading checkpoint shards: 100%|██████████| 3/3 [00:00<00:00, 73.59it/s]
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
