# InternVLA-A1.5 hanging_mug 微调执行日志

> 对应实施手册：[reprd_rbtwn_hngMg.md](reprd_rbtwn_hngMg.md)
> 启动脚本：[`launch/internvla_a15_finetune_robotwin_hngMg_venv.sh`](../../launch/internvla_a15_finetune_robotwin_hngMg_venv.sh)
> 基座权重：`/tmp/itnvla15rbt20/var/hf_home/ckpts/InternVLA-A1.5-base/`
> 源数据：`/tmp/RunPkg/Dta/RoboTwin-Clean/hanging_mug/`（LeRobot v2.1）
> 虚拟环境：`/tmp/itnvla15rbt20/`
>
> 本文件记录**实际执行**过程：每条命令及其理由、全部报错与根因、修复措施、文件增删改。时间以本机 `date -u` 为准（UTC）。

---

## 目录

- [0. 执行目标与约束](#0-执行目标与约束)
- [1. 时间线](#1-时间线)
- [2. 操作记录（命令 / 理由 / 输出）](#2-操作记录命令--理由--输出)
- [3. 问题记录（报错 → 根因 → 修复 → 验证）](#3-问题记录报错--根因--修复--验证)
- [4. 文件变更清单](#4-文件变更清单)
- [5. 关键路径](#5-关键路径)
- [6. 训练监控摘录](#6-训练监控摘录)
- [7. 最终结果](#7-最终结果)

---

## 0. 执行目标与约束

- 在 venv `/tmp/itnvla15rbt20/` 中，基于 InternVLA-A1.5-base，对 RoboTwin `hanging_mug` 单任务做 abs 微调。
- 不原地修改 `/tmp/RunPkg/Dta/RoboTwin-Clean/hanging_mug/`。
- 训练代码必须来自本仓库 `/tmp/SRC/InternVLA-A-series/src`（venv editable 默认指向 `/tmp/SRC/itvlaGp/src`）。
- 超参：8 GPU、`batch_size=16`、`steps=10000`、`action_loss_only=false`、`dist_loading=false`。
- 不用 `nohup & disown` 启 DDP。

---

## 1. 时间线

| 时间 (UTC) | 操作 | 结果 |
|---|---|---|
| 14:17 | 创建本日志；核对 GPU / 数据目录 | 8×H200 全空闲；无 `data` symlink；`RoboTwin-Clean-v30` 仅有 stack_bowls_three |
| 14:18 | 激活 venv，验证包 / patch / 权重 / `import lerobot` 路径 | torch 2.10.0+cu128、torchcodec 0.10.0、flash-attn 2.8.3、Qwen3.5 patch 已在；lerobot 来自本仓库 `src/` |
| 14:18 | 建 `data` symlink；rsync hanging_mug v2.1 副本；建转换用 symlink | 副本 119MB，`codebase_version=v2.1` |
| 14:18 | `convert_my_dataset_v21_to_v30.py`（`--push-to-hub false`，`HF_HUB_OFFLINE=1`） | 约 3s 完成；产物 v3.0、50 ep、16889 frames |
| 14:19 | rsync v3.0 到数据盘；训练 symlink 改指 `hanging_mug_v30`；`LeRobotDataset` 加载 | 三路相机帧非零；task 文本可读 |
| 14:19 | `compute_norm_stats_multi.py --action_mode abs --chunk_size 50` | `agg_1repos_4eb657cb6a/stats.json`，count=16889，skipped=0 |
| 14:19 | 首次启动训练 | **失败** 问题 #1：`accelerate: command not found` |
| 14:20 | 改脚本为 `python -m accelerate.commands.launch` 后重跑 | **失败** 问题 #2：`HF_HUB_OFFLINE=1` 导致 FAST tokenizer 无法加载 |
| 14:25 | 脚本 `unset HF_HUB_OFFLINE`；第三次启动 | 模型/WAN 加载成功 |
| 14:28 | `Start offline training` | JOB=`2026_08_26_14_25_52-internvla_a1_5-robotwin-hanging_mug-abs-a15_base-finetune` |
| 14:29 | step=50 | loss=7.328 / action=0.262；显存 135.7 GiB/卡 |
| 14:31 | step=100 | loss=5.525 / action=0.165；0.80 iters/s |
| 14:33 | step=200 | loss=2.489 / action=0.043；grdn=9.2；训练进入稳态 |
| （后续） | 跑完 10k steps | 见 §7 |

---

## 2. 操作记录（命令 / 理由 / 输出）

### 2.1 环境快照

**理由**：确认 GPU 空闲、确认尚未做过 hanging_mug 的 v3.0 转换。

```bash
date -u
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
ls /tmp/RunPkg/Dta/RoboTwin-Clean-v30/
```

输出：8×H200 used=0；`RoboTwin-Clean-v30` 仅有 stack_bowls_three 相关目录。

### 2.2 激活 venv 并验证

**理由**：手册要求全部操作在 `/tmp/itnvla15rbt20/`；`LD_LIBRARY_PATH` 必须以 `venv/lib` + `nvidia/*/lib` 开头，否则 torchcodec 解不出视频；`PYTHONPATH` 必须指向本仓库 `src/`。

```bash
source /tmp/itnvla15rbt20/bin/activate
export HF_HOME=/tmp/itnvla15rbt20/var/hf_home
export HF_LEROBOT_HOME=${HF_HOME}/lerobot
export PYTHONPATH=/tmp/SRC/InternVLA-A-series/src
VENV_ROOT=/tmp/itnvla15rbt20
NV_LIBS="$(find "${VENV_ROOT}/lib/python3.11/site-packages/nvidia" -type d -name lib | paste -sd:)"
export LD_LIBRARY_PATH="${VENV_ROOT}/lib:${NV_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
python -c "import torch,transformers,torchcodec,flash_attn,lerobot; ..."
```

输出摘要：

- `which python` = `/tmp/itnvla15rbt20/bin/python`
- torch 2.10.0+cu128 / transformers 5.2.0 / torchcodec 0.10.0+cu128 / flash_attn 2.8.3
- 8×H200
- `lerobot.__file__` = `/tmp/SRC/InternVLA-A-series/src/lerobot/__init__.py`
- Qwen3.5 patch 文件存在
- base `model.safetensors` 5.1G；WAN VAE 2.7G

### 2.3 数据根 symlink + v2.1 副本

**理由**：训练通过 `${HF_LEROBOT_HOME}/<repo_id>` 找数据；禁止原地改 Clean。

```bash
mkdir -p ${HF_LEROBOT_HOME}
ln -sfn ${HF_LEROBOT_HOME} /tmp/SRC/InternVLA-A-series/data
rsync -a /tmp/RunPkg/Dta/RoboTwin-Clean/hanging_mug/ /tmp/RunPkg/Dta/RoboTwin-Clean-v30/hanging_mug/
mkdir -p ${HF_LEROBOT_HOME}/robotwin
ln -sfn /tmp/RunPkg/Dta/RoboTwin-Clean-v30/hanging_mug ${HF_LEROBOT_HOME}/robotwin/hanging_mug
```

输出：副本 119MB，`codebase_version=v2.1`。

### 2.4 v2.1 → v3.0 转换

**理由**：本仓库 `CODEBASE_VERSION=v3.0`，v2.1 会 `BackwardCompatibilityError`。用 `convert_my_dataset` + `--push-to-hub false`，避免默认推 Hub。当时加了 `HF_HUB_OFFLINE=1`，防止脚本去 Hub 拉不存在的 `robotwin/hanging_mug`。

```bash
python src/lerobot/datasets/v30/convert_my_dataset_v21_to_v30.py \
  --old-repo-id robotwin/hanging_mug \
  --new-repo-id robotwin/hanging_mug_v30 \
  --push-to-hub false
```

输出：exit 0；`hanging_mug_v30` codebase_version=v3.0，50/16889。脚本曾打印 “Trying to download v3.0 from the hub”，因 offline 回退到本地目录，**未污染源数据**。

随后：

```bash
rsync -a ${HF_LEROBOT_HOME}/robotwin/hanging_mug_v30/ \
  /tmp/RunPkg/Dta/RoboTwin-Clean-v30/hanging_mug_v30/
ln -sfn /tmp/RunPkg/Dta/RoboTwin-Clean-v30/hanging_mug_v30 \
  ${HF_LEROBOT_HOME}/robotwin/hanging_mug
```

`LeRobotDataset('robotwin/hanging_mug')`：version=3.0，三相机 `min>0`（非全零 fallback），task=`Pick the mug with rounded handle up, twist it, place it back, then hang it onto the metal rack.`

### 2.5 外部 stats

**理由**：`use_external_stats=true`；Gr00t 的 `stats_gr00t.json` 不能直接用。group 名由 `sha1("robotwin/hanging_mug")[:10]` = `4eb657cb6a`。

```bash
python util_scripts/compute_norm_stats_multi.py \
  --action_mode abs --chunk_size 50 --repo_ids robotwin/hanging_mug
```

输出：`.../stats/aloha/abs/agg_1repos_4eb657cb6a/stats.json`；action/state dim=14，count=16889，skipped=0。

### 2.6 训练启动（第三次成功）

第一次、第二次失败见 §3。第三次（14:25 UTC）：

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_HOME=/tmp/itnvla15rbt20/var/hf_home
cd /tmp/SRC/InternVLA-A-series
bash launch/internvla_a15_finetune_robotwin_hngMg_venv.sh
```

stdout 归档：`/tmp/hngMg_logs/train.log`  
JOB_NAME：`2026_08_26_14_25_52-internvla_a1_5-robotwin-hanging_mug-abs-a15_base-finetune`  
OUTPUT_DIR：`outputs/internvla_a1_5/2026_08_26_14_25_52-internvla_a1_5-robotwin-hanging_mug-abs-a15_base-finetune`

加载摘要：external stats 命中；WAN 从本地目录加载成功；A1.5-base 本地权重加载。出现 WAN `Missing key(s)` 警告——base checkpoint 不含 WAN 模块，WAN 已由 `Wan2.2-TI2V-5B` 单独加载，与 stack_bowls_three 行为一致，**不是失败**。

`find_unused_parameters=True` 的 DDP 警告：WAN 冻结后 forward 无 unused param，可忽略。

---

## 3. 问题记录（报错 → 根因 → 修复 → 验证）

### 问题 #1：`accelerate: command not found`

- **时间**：14:19:53 UTC
- **报错**：`launch/internvla_a15_finetune_robotwin_hngMg_venv.sh: line 173: accelerate: command not found`，`TRAIN_EXIT:127`
- **根因**：venv 已安装 `accelerate==1.14.0`（`site-packages/accelerate`），但 **没有** 把 console script 装进 `${VENV_ROOT}/bin/`（`ls bin/` 无 `accelerate`）。`python -m accelerate` 也没有 `__main__`。真正入口是 `accelerate.commands.launch`。
- **修复**：把脚本最后一行改为  
  `"${VENV_ROOT}/bin/python" -m accelerate.commands.launch "${ARGS[@]}"`
- **验证**：第三次启动时 accelerate 成功 spawn 8 个 rank。失败日志：`/tmp/hngMg_logs/train_fail1_accelerate_not_found.log`

### 问题 #2：`HF_HUB_OFFLINE=1` 导致 FAST tokenizer 加载失败

- **时间**：14:20:53–14:21:08 UTC
- **报错**（发生在解析 `TrainPipelineConfig.dataset` 时，因为 `use_fast_action_tokens=true` 会构造 `FASTInternVLAA15ActionTokenizerTransformFn`）：

```
LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled.
OSError: We couldn't connect to 'https://huggingface.co' to load the files
draccus.utils.DecodingError: `dataset`: ... Underlying error is "OSError: ..."
```

栈：`transform_internvla_a1_5.py:387 AutoProcessor.from_pretrained("physical-intelligence/fast")`

- **根因**：
  1. 转换数据时设了 `HF_HUB_OFFLINE=1`，**该变量留在后续 shell 会话里**。
  2. 训练加载 FAST tokenizer 需要读 HF 缓存；offline 下 `cached_file("physical-intelligence/fast", "tokenizer.json")` 失败。
  3. 本地 FAST 缓存本身也不完整（snapshots 只有 `tokenizer.json` / `processor_config.json` / `processing_action_tokenizer.py`）。offline 打不开；online + `tokenizer_file=` 可以加载。
- **修复**：
  1. 启动脚本增加 `unset HF_HUB_OFFLINE` / `unset TRANSFORMERS_OFFLINE`。
  2. 第三次启动前在父 shell 同样 unset。
- **验证**：unset 后 `AutoProcessor.from_pretrained(..., tokenizer_file=...)` 成功；第三次训练越过 config 解析并加载完模型。失败日志：`/tmp/hngMg_logs/train_fail2_hf_hub_offline.log`

### 问题 #3：（预留）

（若 10k 训练中再出错，在此追加）

---

## 4. 文件变更清单

| 时间 (UTC) | 路径 | 操作 | 缘由 |
|---|---|---|---|
| 14:17 | `b/d/p/reprd_rbtwn_hngMgLOG.md` | 新增 | 用户要求独立执行日志 |
| 14:18 | `/tmp/SRC/InternVLA-A-series/data` | 新增 symlink → `${HF_HOME}/lerobot` | 训练/人工核对的数据根 |
| 14:18 | `/tmp/RunPkg/Dta/RoboTwin-Clean-v30/hanging_mug/` | 新增（v2.1 rsync） | 不污染 Clean 源数据 |
| 14:18 | `${HF_LEROBOT_HOME}/robotwin/hanging_mug` | symlink（先指 v2.1 副本，后改指 v3.0） | `repo_id=robotwin/hanging_mug` |
| 14:18 | `${HF_LEROBOT_HOME}/robotwin/hanging_mug_v30` 再 rsync 到 `/tmp/RunPkg/Dta/RoboTwin-Clean-v30/hanging_mug_v30/` | 新增（v3.0） | 训练实际读取 |
| 14:19 | `${HF_HOME}/lerobot/stats/aloha/abs/agg_1repos_4eb657cb6a/stats.json` | 新增 | external abs stats |
| 14:20 | `launch/internvla_a15_finetune_robotwin_hngMg_venv.sh` | 修改 | 问题 #1：`python -m accelerate.commands.launch` |
| 14:25 | 同上脚本 | 再修改 | 问题 #2：`unset HF_HUB_OFFLINE` |
| 14:19–14:21 | 两次失败 job 的空 `outputs/...` 目录 | 删除 | 避免 `resume=false` 目录冲突 |
| 14:25 | `outputs/internvla_a1_5/2026_08_26_14_25_52-internvla_a1_5-robotwin-hanging_mug-abs-a15_base-finetune/` | 新增（训练中） | 正式 10k run |
| — | `/tmp/hngMg_logs/train.log` | 正式训练 stdout | 监控 |
| — | `/tmp/hngMg_logs/train_fail1_*.log` / `train_fail2_*.log` | 失败归档 | 问题 #1/#2 |

**未修改**：`/tmp/RunPkg/Dta/RoboTwin-Clean/hanging_mug/`（只读）。

---

## 5. 关键路径

| 用途 | 路径 |
|---|---|
| 虚拟环境 | `/tmp/itnvla15rbt20/` |
| 仓库 | `/tmp/SRC/InternVLA-A-series` |
| HF_HOME | `/tmp/itnvla15rbt20/var/hf_home` |
| Base | `${HF_HOME}/ckpts/InternVLA-A1.5-base/` |
| WAN | `${HF_HOME}/hub/Wan2.2-TI2V-5B/` |
| 源数据 v2.1（只读） | `/tmp/RunPkg/Dta/RoboTwin-Clean/hanging_mug/` |
| 训练数据 v3.0 | `/tmp/RunPkg/Dta/RoboTwin-Clean-v30/hanging_mug_v30/` |
| 数据 symlink | `data/robotwin/hanging_mug` → 上述 v3.0 |
| External stats | `${HF_HOME}/lerobot/stats/aloha/abs/agg_1repos_4eb657cb6a/stats.json` |
| 启动脚本 | `launch/internvla_a15_finetune_robotwin_hngMg_venv.sh` |
| JOB | `2026_08_26_14_25_52-internvla_a1_5-robotwin-hanging_mug-abs-a15_base-finetune` |
| Checkpoint 根 | `outputs/internvla_a1_5/<JOB>/checkpoints/` |
| 训练日志 | `/tmp/hngMg_logs/train.log` |

---

## 6. 训练监控摘录

JOB：`2026_08_26_14_25_52-internvla_a1_5-robotwin-hanging_mug-abs-a15_base-finetune`  
有效 batch：16×8=128；`num_frames=16889`；`num_episodes=50`  
参数：Total 8B / Trainable 3B / WAN 5B（冻结 DiT）

| 时间 (UTC) | step | loss | loss_action | loss_video | loss_vqa | grdn | lr | iters/s | 显存 |
|---|---|---|---|---|---|---|---|---|---|
| 14:29 | 50 | 7.328 | 0.262 | 0.216 | 4.497 | 41.8 | 1.3e-06 | 0.63 | ~135.7 GiB |
| 14:31 | 100 | 5.525 | 0.165 | 0.208 | 3.665 | 24.8 | 3.8e-06 | 0.80 | 135681 MiB |
| 14:32 | 150 | 3.563 | 0.084 | 0.197 | 2.526 | 12.3 | 6.3e-06 | 0.77 | 同上 |
| 14:33 | 200 | 2.489 | 0.043 | 0.185 | 1.874 | 9.2 | 8.8e-06 | 0.82 | 同上 |

与 stack_bowls_three 早期曲线接近（彼处 step50 loss≈7.69 / action≈0.28）。`video_decode_error` 未出现在日志中（仅在计数 >0 时打印）。ETA 约 3.3h 跑完 10k。

预期 checkpoint：`002500` / `005000` / `007500` / `010000`。

---

## 7. 最终结果

| 指标 | 值 |
|---|---|
| 训练状态 | **进行中**（已过 step 200，稳态） |
| JOB | `2026_08_26_14_25_52-internvla_a1_5-robotwin-hanging_mug-abs-a15_base-finetune` |
| per-GPU batch / effective BS | 16 / 128 |
| GPU | 8×H200，~135.7 GiB/卡 |
| 数据 | v3.0 hanging_mug，50 ep / 16889 frames |
| 完成步数 | 待 10000 |
| `exit_code` | 待定 |

---

# Session 2（本机 6×H200 / 12500 步）

> 时间：2026-08-27 起（UTC）。本机与 Session 1 **不是同一运行现场**：
> - GPU 是 **6×H200**（不是 8），全部空闲。
> - Session 1 的 v3.0 数据、stats、`outputs/`、`/tmp/hngMg_logs/` **均不存在**，不能 resume。
> - 按改良后的脚本默认 `STEPS=12500`、`batch_size=16`、effective BS=96 新开一轮。
> - 本轮只训练，不做 RoboTwin closed-loop 评测。

## S2-0. 与 Session 1 的差异

| 项 | Session 1 | Session 2（本机） |
|---|---|---|
| GPU | 8×H200 | **6×H200** |
| steps | 10000 | **12500**（`STEPS` 可覆盖） |
| effective BS | 16×8=128 | 16×6=**96** |
| v3.0 / stats / outputs | 当时已做或训练中 | **启动时全部缺失** |
| 评测 | 手册写了 | 本轮不做 |

## S2-1. 时间线

| 时间 (UTC) | 操作 | 结果 |
|---|---|---|
| 00:46 | 核对本机 GPU / 路径；改良脚本与手册 | 6×H200 全空闲；v30/stats/`data` symlink/outputs 均缺失 |
| 00:46 | 激活 venv，验证包 / patch / 权重 / `import lerobot` | torch 2.10.0+cu128、torchcodec 0.10.0、flash-attn 2.8.3、6×H200、lerobot 来自本仓库 `src/` |
| 00:46 | 建 `data` symlink；rsync hanging_mug v2.1 副本；建转换用 symlink | 副本 119MB，`codebase_version=v2.1`；Clean 源仍为 v2.1 |
| 00:47 | `convert_my_dataset_v21_to_v30.py`（`--push-to-hub false`，`HF_HUB_OFFLINE=1`） | ~4s；产物 v3.0、50 ep、16889 frames |
| 00:47 | rsync v3.0 到数据盘；训练 symlink 改指 `hanging_mug_v30`；`LeRobotDataset` 加载 | 三路相机 `min>0`；task 文本可读 |
| 00:47 | `compute_norm_stats_multi.py --action_mode abs --chunk_size 50` | `agg_1repos_4eb657cb6a/stats.json`，count=16889，skipped=0 |
| 00:48 | 启动 `internvla_a15_finetune_robotwin_hngMg_venv.sh`（未设 `CUDA_VISIBLE_DEVICES`，由 nvidia-smi 探测） | 预检通过：6 卡、`STEPS=12500`、effective BS=96 |
| 00:50:13 | `Start offline training` | JOB=`2026_08_27_00_48_08-internvla_a1_5-robotwin-hanging_mug-abs-a15_base-finetune` |
| 00:51:32 | step=50 | loss=7.393 / action=0.260 / video=0.214；显存 135615 MiB/卡；无 OOM、无 `video_decode_error` |
| （后续） | 跑完 12500 steps | 见 §S2-7 |

## S2-2. 操作记录（命令 / 理由 / 输出）

### S2-2.1 环境快照

**理由**：确认本机是 6 卡而不是手册 Session 1 的 8 卡；确认 v3.0 / stats 必须现做。

```bash
date -u
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
```

输出：6×H200 used=0 MiB。`RoboTwin-Clean-v30`、`data` symlink、stats、`outputs/`、`/tmp/hngMg_logs` 均不存在。

### S2-2.2 改良脚本与手册

**理由**：默认 8 卡 / 10000 步与本机不符；训练步数需可配置；缺路径应在 accelerate 拉起前失败。

- 改 [`launch/internvla_a15_finetune_robotwin_hngMg_venv.sh`](../../launch/internvla_a15_finetune_robotwin_hngMg_venv.sh)：
  - 未设 `CUDA_VISIBLE_DEVICES` 时用 `nvidia-smi -L` 探测（本机得到 `0,1,2,3,4,5`，`PROC_PER_NODE=6`）
  - `STEPS="${STEPS:-12500}"`，`BATCH_SIZE` / `SAVE_FREQ` / `MASTER_PORT` 仍可覆盖
  - 预检：base `model.safetensors`、WAN VAE、external stats、dataset `info.json` 且 `codebase_version=v3.0`
- 改 [`b/d/p/reprd_rbtwn_hngMg.md`](reprd_rbtwn_hngMg.md) Part A：6×H200、12500 步、effective BS=96、本机缺失路径必须按 §2 现做。

### S2-2.3 激活 venv 并验证

**理由**：全部操作在 `/tmp/itnvla15rbt20/`；`LD_LIBRARY_PATH` 必须以 `venv/lib` + `nvidia/*/lib` 开头；`PYTHONPATH` 必须指向本仓库 `src/`。

```bash
source /tmp/itnvla15rbt20/bin/activate
export HF_HOME=/tmp/itnvla15rbt20/var/hf_home
export HF_LEROBOT_HOME=${HF_HOME}/lerobot
export PYTHONPATH=/tmp/SRC/InternVLA-A-series/src
VENV_ROOT=/tmp/itnvla15rbt20
NV_LIBS="$(find "${VENV_ROOT}/lib/python3.11/site-packages/nvidia" -type d -name lib | paste -sd:)"
export LD_LIBRARY_PATH="${VENV_ROOT}/lib:${NV_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
```

输出摘要：

- `which python` = `/tmp/itnvla15rbt20/bin/python`
- torch 2.10.0+cu128 / transformers 5.2.0 / torchcodec 0.10.0+cu128 / flash_attn 2.8.3
- **6×H200**（各约 140GB）
- `lerobot.__file__` = `/tmp/SRC/InternVLA-A-series/src/lerobot/__init__.py`
- Qwen3.5 patch 存在；base `model.safetensors` 5.1G；WAN VAE 2.7G

### S2-2.4 数据根 symlink + v2.1 副本

**理由**：训练通过 `${HF_LEROBOT_HOME}/<repo_id>` 找数据；禁止原地改 Clean。

```bash
mkdir -p ${HF_LEROBOT_HOME}
ln -sfn ${HF_LEROBOT_HOME} /tmp/SRC/InternVLA-A-series/data
rsync -a /tmp/RunPkg/Dta/RoboTwin-Clean/hanging_mug/ /tmp/RunPkg/Dta/RoboTwin-Clean-v30/hanging_mug/
mkdir -p ${HF_LEROBOT_HOME}/robotwin
ln -sfn /tmp/RunPkg/Dta/RoboTwin-Clean-v30/hanging_mug ${HF_LEROBOT_HOME}/robotwin/hanging_mug
```

输出：副本 119MB，`codebase_version=v2.1`。之后核对 Clean 源仍为 v2.1。

### S2-2.5 v2.1 → v3.0 转换

**理由**：本仓库 `CODEBASE_VERSION=v3.0`。用 `convert_my_dataset` + `--push-to-hub false`。转换时设 `HF_HUB_OFFLINE=1`，防止去 Hub 拉不存在的 `robotwin/hanging_mug`。

```bash
export HF_HUB_OFFLINE=1
python src/lerobot/datasets/v30/convert_my_dataset_v21_to_v30.py \
  --old-repo-id robotwin/hanging_mug \
  --new-repo-id robotwin/hanging_mug_v30 \
  --push-to-hub false
```

输出：exit 0。脚本打印 offline 下 `snapshot_download` 回退到本地目录，**未污染源数据**。产物 `hanging_mug_v30` codebase_version=v3.0，50/16889。

随后 unset `HF_HUB_OFFLINE`（训练加载 FAST tokenizer 需要读 HF 缓存，Session 1 问题 #2）：

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
rsync -a ${HF_LEROBOT_HOME}/robotwin/hanging_mug_v30/ \
  /tmp/RunPkg/Dta/RoboTwin-Clean-v30/hanging_mug_v30/
ln -sfn /tmp/RunPkg/Dta/RoboTwin-Clean-v30/hanging_mug_v30 \
  ${HF_LEROBOT_HOME}/robotwin/hanging_mug
```

`LeRobotDataset('robotwin/hanging_mug')`：version=3.0，三相机 `min>0`（非全零 fallback），task=`Pick the mug with rounded handle up, twist it, place it back, then hang it onto the metal rack.`

### S2-2.6 外部 stats

**理由**：`use_external_stats=true`；group 名 `sha1("robotwin/hanging_mug")[:10]` = `4eb657cb6a`。

```bash
python util_scripts/compute_norm_stats_multi.py \
  --action_mode abs --chunk_size 50 --repo_ids robotwin/hanging_mug
```

输出：`.../stats/aloha/abs/agg_1repos_4eb657cb6a/stats.json`；action/state dim=14，count=16889，skipped=0。

### S2-2.7 训练启动

**理由**：不用 `nohup & disown`。脚本会 unset `HF_HUB_OFFLINE`、按 nvidia-smi 探测 6 卡、预检路径后再 `python -m accelerate.commands.launch`。

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE CUDA_VISIBLE_DEVICES
export HF_HOME=/tmp/itnvla15rbt20/var/hf_home
export HF_LEROBOT_HOME=${HF_HOME}/lerobot
cd /tmp/SRC/InternVLA-A-series
bash launch/internvla_a15_finetune_robotwin_hngMg_venv.sh
```

预检输出：

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 PROC_PER_NODE=6
STEPS=12500 BATCH_SIZE=16 PROC_PER_NODE=6 DIST_LOADING=false
OUTPUT_DIR=outputs/internvla_a1_5/2026_08_27_00_48_08-internvla_a1_5-robotwin-hanging_mug-abs-a15_base-finetune
dataset codebase_version=v3.0
```

stdout 归档：`/tmp/hngMg_logs/train_s2.log`  
JOB_NAME：`2026_08_27_00_48_08-internvla_a1_5-robotwin-hanging_mug-abs-a15_base-finetune`

加载摘要：external stats 命中；WAN 从本地目录加载成功；A1.5-base 本地权重加载。`cfg.steps=12500`，`Effective batch size: 16 x 6 = 96`。WAN `Missing key(s)` 警告——base checkpoint 不含 WAN 模块，WAN 已由 `Wan2.2-TI2V-5B` 单独加载，与 Session 1 / stack_bowls_three 一致，**不是失败**。`find_unused_parameters=True` 的 DDP 警告可忽略。HF Hub 未认证 HEAD 请求有 404/307，本地缓存命中后继续，**不是失败**。

---

## S2-3. 问题记录（报错 → 根因 → 修复 → 验证）

本轮数据准备与启动**没有复现** Session 1 的问题 #1（`accelerate: command not found`，脚本已用 `python -m accelerate.commands.launch`）和问题 #2（启动脚本已 `unset HF_HUB_OFFLINE`）。

### 问题 S2-#1：（预留）

（若 12500 训练中再出错，在此追加）

**非错误、已记录的警告：**

1. WAN `Missing key(s) when loading model: model.wan_video_model...` — base 不含 WAN；WAN 已从 `Wan2.2-TI2V-5B` 单独 load。
2. DDP `find_unused_parameters=True` 但 forward 无 unused param — WAN 冻结后的已知行为。
3. `Warning: You are sending unauthenticated requests to the HF Hub` — 本机无 `HF_TOKEN`；Qwen/FAST 走本地缓存 + 偶发 HEAD，未阻断训练。

---

## S2-4. 文件变更清单

| 时间 (UTC) | 路径 | 操作 | 缘由 |
|---|---|---|---|
| 00:46 | `launch/internvla_a15_finetune_robotwin_hngMg_venv.sh` | 修改 | 6 卡自动探测、默认 STEPS=12500、路径预检 |
| 00:46 | `b/d/p/reprd_rbtwn_hngMg.md` | 修改 | Part A 对齐本机 6×H200 / 12500 步 / 缺失路径 |
| 00:46 | `b/d/p/reprd_rbtwn_hngMgLOG.md` | 追加 Session 2 | 用户要求独立执行日志追加在后面 |
| 00:46 | `/tmp/SRC/InternVLA-A-series/data` | 新增 symlink → `${HF_HOME}/lerobot` | 训练/人工核对的数据根 |
| 00:46 | `/tmp/RunPkg/Dta/RoboTwin-Clean-v30/hanging_mug/` | 新增（v2.1 rsync） | 不污染 Clean 源数据 |
| 00:47 | `${HF_LEROBOT_HOME}/robotwin/hanging_mug` | symlink（先指 v2.1 副本，后改指 v3.0） | `repo_id=robotwin/hanging_mug` |
| 00:47 | `/tmp/RunPkg/Dta/RoboTwin-Clean-v30/hanging_mug_v30/` | 新增（v3.0） | 训练实际读取 |
| 00:47 | `${HF_HOME}/lerobot/stats/aloha/abs/agg_1repos_4eb657cb6a/stats.json` | 新增 | external abs stats |
| 00:48 | `outputs/internvla_a1_5/2026_08_27_00_48_08-internvla_a1_5-robotwin-hanging_mug-abs-a15_base-finetune/` | 新增（训练中） | 正式 12500 步 run |
| 00:51 | `/tmp/hngMg_logs/train_s2.log` | 新增 | Session 2 训练 stdout 归档 |

**未修改**：`/tmp/RunPkg/Dta/RoboTwin-Clean/hanging_mug/`（只读，仍为 v2.1）。

---

## S2-5. 关键路径

| 用途 | 路径 |
|---|---|
| 虚拟环境 | `/tmp/itnvla15rbt20/` |
| 仓库 | `/tmp/SRC/InternVLA-A-series` |
| HF_HOME | `/tmp/itnvla15rbt20/var/hf_home` |
| Base | `${HF_HOME}/ckpts/InternVLA-A1.5-base/` |
| WAN | `${HF_HOME}/hub/Wan2.2-TI2V-5B/` |
| 源数据 v2.1（只读） | `/tmp/RunPkg/Dta/RoboTwin-Clean/hanging_mug/` |
| 训练数据 v3.0 | `/tmp/RunPkg/Dta/RoboTwin-Clean-v30/hanging_mug_v30/` |
| 数据 symlink | `data/robotwin/hanging_mug` → 上述 v3.0 |
| External stats | `${HF_HOME}/lerobot/stats/aloha/abs/agg_1repos_4eb657cb6a/stats.json` |
| 启动脚本 | `launch/internvla_a15_finetune_robotwin_hngMg_venv.sh` |
| JOB | `2026_08_27_00_48_08-internvla_a1_5-robotwin-hanging_mug-abs-a15_base-finetune` |
| Checkpoint 根 | `outputs/internvla_a1_5/<JOB>/checkpoints/` |
| 训练日志 | `/tmp/hngMg_logs/train_s2.log` |

---

## S2-6. 训练监控摘录

JOB：`2026_08_27_00_48_08-internvla_a1_5-robotwin-hanging_mug-abs-a15_base-finetune`  
有效 batch：16×6=**96**；`num_frames=16889`；`num_episodes=50`；`steps=12500`  
参数：Total 8B / Trainable 3B / WAN 5B（冻结 DiT）

| 时间 (UTC) | step | loss | loss_action | loss_video | loss_vqa | grdn | lr | iters/s | 显存 |
|---|---|---|---|---|---|---|---|---|---|
| 00:51:32 | 50 | 7.393 | 0.260 | 0.214 | 4.577 | 44.0 | 1.3e-06 | 0.70 | 135615 MiB |

与 Session 1 step50（loss≈7.328 / action≈0.262）接近。`video_decode_error` 未出现。step50 时 ETA 约 4h55m 跑完剩余步数。

预期 checkpoint：`002500` / `005000` / `007500` / `010000` / `012500`。

---

## S2-7. 最终结果

| 指标 | 值 |
|---|---|
| 训练状态 | **进行中**（已过 step 50，稳态） |
| JOB | `2026_08_27_00_48_08-internvla_a1_5-robotwin-hanging_mug-abs-a15_base-finetune` |
| per-GPU batch / effective BS | 16 / **96** |
| GPU | **6×H200**，~135.6 GiB/卡 |
| 数据 | v3.0 hanging_mug，50 ep / 16889 frames |
| 计划步数 | **12500** |
| 完成步数 | 待 12500 |
| `exit_code` | 待定 |


