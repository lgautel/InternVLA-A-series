# InternVLA-A1.5 在 LIBERO-Plus（Camera / Robot 类别）上的复现实施手册

> 目标：基于 [InternRobotics/InternVLA-A1.5-base](https://huggingface.co/InternRobotics/InternVLA-A1.5-base) 权重，在 4 个 LIBERO 任务套件（`libero_spatial/object/goal/10`）上联合微调，再对微调后的 checkpoint 在 [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) 基准上做**零样本**评测，只跑 `Camera Viewpoints`（论文简称 Camera）和 `Robot Initial States`（论文简称 Robot）两个扰动类别，复现论文 Table 6 中的 Camera≈83%、Robot≈55% 附近的成功率（±5 个百分点视为复现成功）。
>
> 本手册分两部分：**Part A 是可执行的分步操作手册**（先写后执行）；**Part B 是执行记录**——按时间顺序记录所有实际执行的操作、遇到的每一个报错的根因分析与修复方式、以及全部新增/修改/删除文件清单，最后给出最终结果对比。

---

## 目录

- [Part A：实施手册](#part-a实施手册)
  - [0. 关键结论与设计依据](#0-关键结论与设计依据)
  - [1. 环境准备（两个虚拟环境）](#1-环境准备两个虚拟环境)
  - [2. 权重与数据下载](#2-权重与数据下载)
  - [3. 数据集格式核对](#3-数据集格式核对)
  - [4. 4 套件联合微调](#4-4-套件联合微调)
  - [5. LIBERO-Plus (Camera+Robot) 零样本评测](#5-libero-pluscamerarobot-零样本评测)
  - [6. 结果聚合与验收标准](#6-结果聚合与验收标准)
  - [7. 已知限制](#7-已知限制)
- [Part B：执行记录](#part-b执行记录)
  - [时间线 / 操作日志](#时间线--操作日志)
  - [问题记录（报错 → 根因 → 修复 → 验证）](#问题记录报错--根因--修复--验证)
  - [文件变更清单](#文件变更清单)
  - [最终结果](#最终结果)

---

## Part A：实施手册

### 0. 关键结论与设计依据

在动手之前，先把几个决定复现方案形态的关键事实记录下来（均来自阅读仓库源码 / 论文 / 官方 README 得出，避免执行过程中反复返工）：

1. **LIBERO-Plus 的 Camera/Robot 分数是"零样本"分数，训练数据是标准 LIBERO，不是 LIBERO-Plus。** 论文原文（`b/d/p/InternVLA-A1.5-paper.md` 附录 A.2）：

   > "LIBERO-Plus... We do not train on LIBERO-Plus; instead, we evaluate the LIBERO checkpoint described above in a zero-shot manner and report the success rate."

   且 LIBERO 微调本身是"a single model jointly on the mixture of all four suites"（四个套件联合训练一个模型，而不是每个套件单独训练）。因此复现路径是：**先用 InternVLA-A1.5-base 在 4 个 LIBERO 套件上联合微调 → 再把微调后的 checkpoint 直接搬到 LIBERO-Plus 上跑，不再训练**。

2. **仓库里已经有专门为 LIBERO 准备的微调脚本和 schema**，这是最贴近官方复现口径的现成配置：
   - [launch/internvla_a15_finetune_libero.sh](../../launch/internvla_a15_finetune_libero.sh)：`--policy.pretrained_path=InternRobotics/InternVLA-A1.5-base`，`--dataset.action_mode=abs`，`batch_size=16`，`steps=100000`，`optimizer_lr=5e-5`，`scheduler_warmup_steps=2000`，`scheduler_decay_steps=100000`，`action_loss_only=false`（即视频前瞻分支参与训练），`freeze_learnable_tokens=false`。
   - [src/lerobot/dataset_schemas/configs/libero.yaml](../../src/lerobot/dataset_schemas/configs/libero.yaml)：为 `libero_10/libero_goal/libero_object/libero_spatial` 四个 `robot_type` 分别注册了 schema。

3. **数据集特征名的推导**：`DatasetSchema`（[schema.py](../../src/lerobot/dataset_schemas/schema.py)）的语义是：
   - `feature_mapping: {canonical_key: [raw_source_keys...]}`（目标键 → 原始键列表）
   - `image_mapping: {raw_source_key: canonical_key}`（**原始键 → 目标键**，与 feature_mapping 方向相反，务必注意，见 [registry.py](../../src/lerobot/dataset_schemas/registry.py) 第 55-86 行的默认 schema 写法可以互相印证）

   `libero.yaml` 里：
   ```yaml
   feature_mapping:
     observation.state: [observation.state]
     action: [action]
   image_mapping:
     observation.images.image: observation.images.image0
     observation.images.wrist_image: observation.images.image1
   ```
   解出：训练数据集**必须提供的原始特征名**是 `observation.state`、`action`、`observation.images.image`、`observation.images.wrist_image`。这与 HuggingFace 上 [nvidia/LIBERO_LeRobot_v3](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3)（LeRobot v3.0 格式，覆盖 `libero_spatial/object/goal/10/90` 五个套件）的特征命名**完全一致**，因此本次复现选用该数据集的 4 个套件（不需要 `libero_90`）。

   本地 `/mnt/r/DATA/libero/lerobot_libero_goal` 是旧数据（`codebase_version=v2.1`，特征名是 `image/wrist_image/state/actions`，且只有 `goal` 一个套件），不满足要求，本次复现改用新下载的 `nvidia/LIBERO_LeRobot_v3`。

4. **LIBERO-Plus 资产两部分**：
   - GitHub 仓库 [sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus)：`libero/libero/{bddl_files,init_files,benchmark}`，整个仓库约 19MB（`task_classification.json` 给出 task_id ↔ 扰动类别的映射，7 大类：`Camera Viewpoints`/`Robot Initial States`/`Language Instructions`/`Light Conditions`/`Background Textures`/`Sensor Noise`/`Objects Layout`）。
   - HuggingFace [Sylvest/LIBERO-plus](https://huggingface.co/datasets/Sylvest/LIBERO-plus) 的 `assets.zip`（约 6.4GB，压缩前更大，包含新纹理/新物体/场景等共享渲染资产），解压到 `<LIBERO_PLUS_REPO_ROOT>/libero/libero/assets/`。

   实测统计（对本地克隆下来的 `task_classification.json` 做计数，与论文 Table 7 完全吻合）：

   | 套件 | Camera Viewpoints | Robot Initial States |
   |---|---|---|
   | libero_spatial | 376 | 350 |
   | libero_object | 396 | 398 |
   | libero_goal | 408 | 409 |
   | libero_10 | 419 | 393 |
   | **合计** | **1599** | **1550** |

   四个套件共 ~10030 个扰动任务，Camera+Robot 合计约 3149 个（~31%），跳过其余 5 类可节省约 69% 的仿真算力——这就是"不用全部跑/下载"的具体体现（`assets.zip` 和 GitHub 仓库本身是单一共享基础设施，无法按类别裁剪下载）。

5. **环境冲突，必须建两个虚拟环境**：
   - 主仓库 `pyproject.toml` 要求 `numpy>=1.26.0,<2.3.0`、`transformers==5.2.0`、`torch==2.10.0`。
   - LIBERO-plus 客户端要求 `numpy==1.24.4`、`mujoco==3.2.3`、`robosuite==1.4.0`（老版本 API），二者无法共存。
   - 两端通过仓库自带的 websocket 协议通信（[evaluation/LIBERO/policy_server/](../../evaluation/LIBERO/policy_server/)），协议层面无需改动。

6. **机器与算力约定**：8×H200(143GB)，其中 GPU4-7 已被其它任务占满（各 ~103GB），本次复现**只使用 GPU0-3**。数据/权重落盘约定：权重 → `/mnt/r/CKPT/`，数据 → `/mnt/r/DATA/`，虚拟环境 → `/mnt/r/VENV/`。

---

### 1. 环境准备（两个虚拟环境）

#### 1.1 主环境 `/mnt/r/VENV/ivla15`（训练 + 推理服务端）

```bash
uv venv /mnt/r/VENV/ivla15 --python 3.11
source /mnt/r/VENV/ivla15/bin/activate

uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers==5.2.0
cd /home/physical/SRC/Robot/InternVLA-A-series
uv pip install -e ".[all]"
uv pip install flash-linear-attention==0.5.0 causal-conv1d==1.6.1 --no-build-isolation
uv pip install flash-attn==2.8.3 --no-build-isolation

# 补丁 transformers（Qwen3.5 / pi0 / pi05 自定义模型代码）
TRANSFORMERS_DIR=/mnt/r/VENV/ivla15/lib/python3.11/site-packages/transformers/
cp -r src/lerobot/policies/pi0/transformers_replace/models ${TRANSFORMERS_DIR}
cp -r src/lerobot/policies/pi05/transformers_replace/models ${TRANSFORMERS_DIR}
cp -r src/lerobot/policies/internvla_a1_5/transformers_replace/models ${TRANSFORMERS_DIR}
```

说明：用 `uv venv --python 3.11` 而非 conda，符合本仓库开发规范"python虚拟环境安装在 `/mnt/r/VENV/`"；`uv` 会自动下载/管理 CPython 3.11 解释器，无需系统安装。`causal-conv1d` 会针对多个 GPU 架构现场编译 CUDA 扩展，耗时较长（实测约 10 分钟）。

#### 1.2 LIBERO-Plus 客户端环境 `/mnt/r/VENV/ivla15_libero_plus_client`

```bash
uv venv /mnt/r/VENV/ivla15_libero_plus_client --python 3.11
source /mnt/r/VENV/ivla15_libero_plus_client/bin/activate

sudo apt-get install -y libmagickwand-dev libfontconfig1-dev libexpat1   # wand(ImageMagick)依赖

uv pip install "numpy==1.24.4" "mujoco==3.2.3" "robosuite==1.4.0" \
  bddl easydict pyyaml opencv-python imageio imageio-ffmpeg \
  websockets msgpack wand torch torchvision termcolor tqdm

# LIBERO-plus 仓库本体（阶段2下载完成后再装，见下）
uv pip install -e /mnt/r/DATA/LIBERO-plus
```

#### 1.3 环境变量约定

```bash
export HF_HOME=/mnt/r/CKPT/hf_home
export HF_LEROBOT_HOME=${HF_HOME}/lerobot
```

---

### 2. 权重与数据下载

全部使用 `uvx --from 'huggingface_hub[cli]' hf download ...`（避免污染两个训练/评测 venv，`hf` CLI 在独立的 uv 临时环境里运行）。

```bash
export HF_HOME=/mnt/r/CKPT/hf_home

# 2.1 InternVLA-A1.5-base 基座权重（~3B 参数，混合精度，约几十GB）
hf download InternRobotics/InternVLA-A1.5-base --local-dir /mnt/r/CKPT/InternVLA-A1.5-base

# 2.2 Wan2.2-TI2V-5B（LIBERO微调脚本 action_loss_only=false，需要WAN分支参与训练）
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir /mnt/r/CKPT/Wan2.2-TI2V-5B

# 2.3 LIBERO 4 套件训练数据（LeRobot v3.0 格式，特征名与 libero.yaml 匹配）
hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset \
  --include "libero_spatial/*" --include "libero_object/*" \
  --include "libero_goal/*" --include "libero_10/*" \
  --local-dir /mnt/r/DATA/libero_lerobot_v3

# 2.4 LIBERO-plus 代码仓库（bddl/init/task_classification）
git clone https://github.com/sylvestf/LIBERO-plus /mnt/r/DATA/LIBERO-plus

# 2.5 LIBERO-plus 共享渲染资产
hf download Sylvest/LIBERO-plus assets.zip --repo-type dataset \
  --local-dir /mnt/r/DATA/LIBERO-plus_assets
mkdir -p /mnt/r/DATA/LIBERO-plus/libero/libero/assets
unzip -q /mnt/r/DATA/LIBERO-plus_assets/assets.zip -d /mnt/r/DATA/LIBERO-plus/libero/libero/assets
```

> 注意：`hf download ... --include` 每个 pattern 要单独一个 `--include` 参数（不能像 `--include "a" "b"`，那样后面的会被解析成 positional filenames 而不是 include pattern）。

---

### 3. 数据集格式核对

下载完成后核对每个套件的 `meta/info.json` 特征名，必须包含 `observation.state`、`action`、`observation.images.image`、`observation.images.wrist_image`：

```bash
for s in libero_spatial libero_object libero_goal libero_10; do
  echo "== $s =="
  python3 -c "import json; d=json.load(open('/mnt/r/DATA/libero_lerobot_v3/$s/meta/info.json')); print(d['codebase_version'], d['robot_type'], list(d['features'].keys()))"
done
```

若特征名不匹配，需要在 `src/lerobot/dataset_schemas/configs/libero.yaml` 里调整 `feature_mapping`/`image_mapping`，而不是改数据（更安全，配置层面隔离）。

---

### 4. 4 套件联合微调

#### 4.1 训练启动脚本

复制 [launch/internvla_a15_finetune_libero.sh](../../launch/internvla_a15_finetune_libero.sh) 为 `launch/internvla_a15_finetune_libero_venv.sh`，主要改动：

- `conda activate` → `source /mnt/r/VENV/ivla15/bin/activate`
- `PRETRAINED_PATH` → `/mnt/r/CKPT/InternVLA-A1.5-base`
- `VLM_MODEL_PATH` 保持 `Qwen/Qwen3.5-2B`（若离线可指向本地缓存）
- WAN 相关：`WAN_MODEL_PATH=/mnt/r/CKPT/Wan2.2-TI2V-5B`（具体参数名以 `configuration_internvla_a1_5.py` 为准）
- `DATASET_REPO_ID` 直接写死为 4 个套件的本地路径，不依赖原脚本里按目录名 glob 发现 `*_no_noops*_lerobot` 的逻辑
- `CUDA_VISIBLE_DEVICES=0,1,2,3`，`PROC_PER_NODE=4`（只用空闲的 GPU0-3）
- 训练前对每个套件的 `meta/info.json` 打 `robot_type=<suite_name>` 补丁（脚本自带逻辑，保证多子集 stats 不互相覆盖，并与评测端 `STATS_KEY_MODE=suite`/`ROBOT_TYPE_MODE=suite` 对应）

#### 4.2 启动

```bash
export HF_HOME=/mnt/r/CKPT/hf_home
export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=0,1,2,3 PROC_PER_NODE=4 bash launch/internvla_a15_finetune_libero_venv.sh
```

#### 4.3 监控

- `outputs/internvla_a1_5/<job_name>/` 下的日志 + wandb offline 记录；
- 训练几百 step 后记录吞吐量，估算 100k steps 的 ETA，写入 Part B 执行记录；
- 定期检查 `loss_action`/`loss_video`/`loss_vqa` 是否正常下降，`nvidia-smi` 确认 GPU0-3 利用率。

---

### 5. LIBERO-Plus（Camera+Robot）零样本评测

#### 5.1 给评测脚本加类别过滤

给 [evaluation/LIBERO-plus/eval_libero_plus.py](../../evaluation/LIBERO-plus/eval_libero_plus.py) 增加 `--categories` 参数（逗号分隔，如 `"Camera Viewpoints,Robot Initial States"`），只迭代 `id2category` 中类别命中的 task_id，其余逻辑（分片 `start_idx/end_idx`、按类别分桶统计、写 `logs/{suite}/{start}_to_{end}.json`）保持不变，确保 [aggregate_results.py](../../evaluation/LIBERO-plus/aggregate_results.py) 的聚合逻辑无需改动。

#### 5.2 venv 版编排脚本

参考 [run_eval_libero_plus.sh](../../evaluation/LIBERO-plus/run_eval_libero_plus.sh) 的 GPU 探测 / flock 工作队列 / 健康检查逻辑，把 `conda activate ${SERVER_ENV}` 换成 `source /mnt/r/VENV/ivla15/bin/activate`，`conda activate ${CLIENT_ENV}` 换成 `source /mnt/r/VENV/ivla15_libero_plus_client/bin/activate`，`GPU_IDS=0,1,2,3` 写死，透传新增的 `--categories "Camera Viewpoints,Robot Initial States"`。

#### 5.3 运行

```bash
export CKPT_PATH=outputs/internvla_a1_5/<job_name>/checkpoints/last   # 微调产物
export LIBERO_HOME=/mnt/r/DATA/LIBERO-plus
export STATS_KEY_MODE=suite
export ROBOT_TYPE_MODE=suite
export CATEGORIES="Camera Viewpoints,Robot Initial States"
GPU_IDS=0,1,2,3 SHARDS_PER_SUITE=8 bash evaluation/LIBERO-plus/run_eval_libero_plus_venv.sh
```

---

### 6. 结果聚合与验收标准

```bash
source /mnt/r/VENV/ivla15_libero_plus_client/bin/activate
python evaluation/LIBERO-plus/aggregate_results.py --root <EVAL_LOG_DIR>
```

核对 `overall_results.json` 里 `leaderboard_summary_percent.Camera` 与 `leaderboard_summary_percent.Robot`：

- 目标：Camera ≈ 83%，Robot ≈ 55%（论文 Table 6）
- 验收标准（"足够接近"）：±5 个百分点内视为复现成功；若明显偏离，先排查是否为环境/配置类 bug（stats 归一化、`STATS_KEY_MODE`/`ROBOT_TYPE_MODE` 配对、WAN 分支是否真正参与了训练等），而不是无限调参搜索。

---

### 7. 已知限制

1. 官方论文没有公开 LIBERO 微调所用的具体数据版本（是否为 `no_noops` 过滤版）和随机种子，本次复现改用 `nvidia/LIBERO_LeRobot_v3`（非 no_noops 过滤版），属于合理近似，可能造成个位数百分点的分数偏差。
2. `assets.zip`（6.4GB）和 LIBERO-plus 仓库本身（~19MB）是单一共享资产，无法按扰动类别裁剪下载；"不下载全部数据"体现在跳过其余 5 个类别的评测（约 6881 个任务，节省约 69% 仿真算力），而不是资产层面的裁剪。
3. 训练使用 4×H200（而非官方可能使用的 GPU 数量），全局 batch size 会与官方实际配置不同，可能带来收敛差异。

---

## Part B：执行记录

> 以下内容随实际执行过程持续追加，记录真实发生的操作、报错与修复。

### 时间线 / 操作日志

> 与 Part A 的**唯一实质性偏差**：执行期间机器上的 GPU 占用情况发生了变化——原计划的 GPU0-3 后来被同机其它任务占满（各 ~103GB），而 GPU4-7 转为空闲，因此**训练与评测全部改到 GPU4-7 上执行**（`launch/internvla_a15_finetune_libero_venv.sh` 与 `run_eval_libero_plus_venv.sh` 的默认值已相应改为 `4,5,6,7`）。其余步骤与 Part A 手册一致。

1. **阶段0 环境准备**：分别创建 `/mnt/r/VENV/ivla15`（torch 2.10.0+cu128、transformers 5.2.0、`pip install -e .`、flash-attn 2.8.3、flash-linear-attention 0.5.0、causal-conv1d 1.6.1，并拷贝 3 份 `transformers_replace/models` 补丁）与 `/mnt/r/VENV/ivla15_libero_plus_client`（numpy 1.24.4、mujoco 3.2.3、robosuite 1.4.0 等旧版仿真依赖）两个虚拟环境。
2. **阶段1 权重与数据下载**：下载 `InternRobotics/InternVLA-A1.5-base`、`Wan-AI/Wan2.2-TI2V-5B` 到 `/mnt/r/CKPT/`；下载 `nvidia/LIBERO_LeRobot_v3` 的 `libero_spatial/object/goal/10` 四个子集到 `/mnt/r/DATA/libero_lerobot_v3/`；`git clone` LIBERO-plus 仓库到 `/mnt/r/DATA/LIBERO-plus`；下载并解压 `Sylvest/LIBERO-plus` 的 `assets.zip`（~6.4GB）。核对四个子集 `meta/info.json` 的 `features` 键名与 `libero.yaml` 期望的 `observation.state`/`action`/`observation.images.image`/`observation.images.wrist_image` 完全一致，无需改 schema。
3. **数据落地方式**：在仓库根目录建立 `data -> /mnt/r/CKPT/hf_home/lerobot` 软链，再在 `/mnt/r/CKPT/hf_home/lerobot/` 下为每个套件建 `libero_<suite> -> /mnt/r/DATA/libero_lerobot_v3/libero_<suite>` 软链，配合 `HF_LEROBOT_HOME=/mnt/r/CKPT/hf_home/lerobot`，使 `lerobot_dataset.py` 按 `repo_id` 解析时直接命中本地数据，不触发任何远程下载。
4. **阶段2 训练脚本准备**：以 `launch/internvla_a15_finetune_libero.sh` 为底稿新写 `launch/internvla_a15_finetune_libero_venv.sh`（venv 激活、本地权重路径、四套件 `DATASET_REPO_ID`、逐子集 `robot_type` 补丁、显式 GPU 绑定）。
5. **阶段3 评测脚本准备**：给 `evaluation/LIBERO-plus/eval_libero_plus.py` 加 `--categories` 过滤参数；新写 `evaluation/LIBERO-plus/run_eval_libero_plus_venv.sh`（双 venv + flock 任务队列 + `--categories` 透传）。
6. **LIBERO-plus 环境基础设施冒烟测试（mock 策略）**：在真正跑训练之前，先用仓库自带的 `server_policy.py --mock_policy`（随机动作）+ 真实的 `eval_libero_plus.py` 客户端，跑通 `libero_spatial` 的 3 个 `Robot Initial States` 任务，验证 websocket 通信、类别过滤、仿真 rollout、`aggregate_results.py` 聚合全链路都是通的（产物：`/mnt/r/tmp/eval_smoketest/overall_results.json`，`total_count=3, success_rate=0.0`——随机策略下 0% 符合预期，重点是流程没有报错）。过程中依次修了：LIBERO-plus assets 解压路径错位、client venv 缺 `matplotlib`、`matplotlib` 拉高 `numpy` 版本、`torch.load weights_only` 默认值变化导致的 `init_states` 反序列化失败（见下方问题记录 #2-#4）。
7. **训练冒烟测试第一次（`train_smoketest1`，30 steps，2026-07-28 约 13:3x）**：真实调用 `internvla_a15_finetune_libero_venv.sh`，在 `flash-linear-attention` 的 Gated DeltaNet kernel 上遇到 Hopper GPU + Triton≥3.4 的已知不兼容问题（问题记录 #5），安装 `tilelang` 后修复。
8. **训练冒烟测试第二次（`train_smoketest2`，30 steps，2026-07-28 13:46:33–13:51:07，GPU4-7）**：训练本身跑通、loss 正常下降、checkpoint 正常保存，但日志里出现了 **1504 条 `[video_decode_error]`**（问题记录 #6，本次复现中影响最大的一个 bug：几乎所有视频帧解码都静默失败并回退成全黑帧）。发现后立即在 `ivla15` venv 里把 `torchcodec` 从 0.15.0 降级到 0.10.0（与 `torch==2.10.0` 官方兼容表匹配的版本，且改装 CPU-only wheel 避免 CUDA13 符号缺失），单元测试验证解码恢复正常。
9. **评测服务端依赖补全**：`evaluation/LIBERO/policy_server/server_policy.py` 依赖的 `websockets`/`msgpack`/`msgpack-numpy` 未随主仓库依赖装好，在 `ivla15` venv 里补装（问题记录 #7）。
10. **正式启动 4 套件联合微调（首次尝试，2026-07-28 13:56:51，GPU4-7）**：`JOB_NAME=a15_libero4suite_100k_20260728_135651`，`STEPS=100000, BATCH_SIZE=16, lr=5e-5`，与 Part A 手册第4节参数完全一致。启动后前 ~4 分钟确认 `grep -c video_decode_error` = 0（torchcodec 修复生效），但随后进程卡在 `Accelerator()`/`TCPStore` 分布式握手，长时间无进展。
11. **分布式握手卡死排查（2026-07-28 14:30~15:10，耗时约40分钟，是本次复现排查耗时最长的单个问题）**：连续 3 次换端口重试均复现同样的假死，先后怀疑并排除了"端口冲突/主机名解析"（已在更早独立修复）、"PyTorch libuv TCPStore 后端 bug"（已应用 `USE_LIBUV=0` 缓解措施，但对照实验证明并非真正根因）、"主机整体负载过高"（`dist_sanity.py` 极简对照实验在相同 GPU/相同负载下秒级完成，排除）、"重量级 import/环境变量覆盖"（`dist_sanity_heavy_imports.py` 对照实验排除）四个假说；最终通过 `py-spy dump` 抓栈 + 检查 `/proc/<pid>/environ` 的 `TORCHELASTIC_USE_AGENT_STORE` + 检查 `ps -o ppid=` 定位到真正根因：本次所有失败尝试都用 `nohup bash ...sh > log 2>&1 & disown` 手动后台化，导致负责托管 TCPStore 服务端并监控 4 个 worker 的 `accelerate launch` 顶层代理进程在工具调用结束后被意外终止，遗留的 4 个 worker 子进程成为孤儿（`PPID=1`），对着一个再没有人监听的端口做合法但永远不会成功的 600 秒超时重试。详见问题记录 #8。
12. **改用 Shell 工具原生后台化机制重新启动（2026-07-28 15:06 起）**：不再手动 `nohup+&+disown`，改为把 `bash launch/internvla_a15_finetune_libero_venv.sh` 直接作为前台命令交给 Shell 工具执行，由工具自身"命令超时未完成即自动转后台、持续输出到日志文件"的机制接管。先用 20-step 小规模验证（`JOB_NAME=a15_nativebg_test_...`）确认 `Accelerator()` 秒级完成、20 步顺利跑完并保存 checkpoint、`exit_code=0`、`video_decode_error=0`；随后于 **2026-07-28 15:10:45** 正式重新启动 4 套件联合微调：`JOB_NAME=a15_libero4suite_100k_20260728_151045`，`STEPS=100000, BATCH_SIZE=16, lr=5e-5`，GPU4-7，日志见 `/mnt/r/tmp/train_full_run_final.log`。第 200 步日志确认训练正常：`loss:6.143 | loss_action:0.229 | loss_video:0.134`，`video_decode_error` 计数为 0，GPU4-7 显存占用 ~131GB/卡、利用率 57%~92%。**此前 `a15_libero4suite_100k_20260728_135651`（首次尝试，卡在握手阶段、从未真正进入训练循环）产生的任何输出目录均已作废，不作为正式训练产物**。
13. 后续训练进度、首个 5000-step checkpoint 上的小规模真实推理抽检、正式 Camera+Robot 评测与聚合结果，见下方"最终结果"一节，将随执行持续更新。

### 问题记录（报错 → 根因 → 修复 → 验证）

#### #1 `hf download --include` 多个 glob pattern 被误当成 positional 文件名

- **现场**：最初尝试 `hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset --include "libero_spatial/*" "libero_object/*" "libero_goal/*" "libero_10/*" --local-dir ...`，命令把第 2-4 个 pattern 解析成了别的 positional 参数，实际只下载了/匹配了第一个 pattern。
- **根因**：`hf download` 的 `--include` 是"接受单个值、可重复传入多次"的参数，不是"接受多个值的列表参数"；用空格并列写多个 glob 会被 argparse 解析成额外的位置参数。
- **修复**：改为每个 pattern 单独一个 `--include` 标志：
  ```bash
  hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset \
    --include "libero_spatial/*" --include "libero_object/*" \
    --include "libero_goal/*" --include "libero_10/*" \
    --local-dir /mnt/r/DATA/libero_lerobot_v3
  ```
- **验证**：`/mnt/r/DATA/libero_lerobot_v3/` 下四个子目录齐全，`meta/info.json` 特征名核对通过（见时间线第2条）。

#### #2 LIBERO-plus `assets.zip` 解压后目录层级不对

- **现场**：按 README 把 `assets.zip` 解压到 `/mnt/r/DATA/LIBERO-plus/libero/libero/assets/`，但压缩包内部本身还带了一层作者本机的绝对路径前缀，实际展开成了 `.../assets/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets/...`，而不是直接落在 `assets/` 根下，导致 LIBERO 运行时找不到贴图/物体资产。
- **根因**：`assets.zip` 打包时保留了作者机器上的完整目录结构（打包习惯问题），并非 `unzip` 命令用错参数。
- **修复**：手动把深层嵌套的真实 `assets` 目录挪回预期根路径：
  ```bash
  DEEP="/mnt/r/DATA/LIBERO-plus/libero/libero/assets/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets"
  TMP="/mnt/r/DATA/LIBERO-plus/libero/libero/assets_new"
  mv "$DEEP" "$TMP"
  rm -rf /mnt/r/DATA/LIBERO-plus/libero/libero/assets
  mv "$TMP" /mnt/r/DATA/LIBERO-plus/libero/libero/assets
  ```
- **验证**：`find /mnt/r/DATA/LIBERO-plus/libero/libero/assets -maxdepth 1` 直接可见 `stable_hope_objaverse/`、`textures/` 等预期子目录；mock 策略冒烟测试（时间线第6条）中场景资产可正常加载、不再报 `FileNotFoundError`。

#### #3 客户端 venv 缺 `matplotlib`，装上后又把 `numpy` 拉出兼容区间

- **现场**：`ivla15_libero_plus_client` 里 `import libero.libero.envs` 抛 `ModuleNotFoundError: No module named 'matplotlib'`；`uv pip install matplotlib` 后又出现 `ImportError: Matplotlib requires numpy>=1.25; you have 1.24.4`。
- **根因**：`libero.libero.envs`（依赖 `robosuite`）间接用到了 `matplotlib`，但 LIBERO-plus 的 `requirements`/README 没有把它列为显式依赖；而最新版 `matplotlib` 又要求 `numpy>=1.25`，与 `robosuite==1.4.0`/`mujoco==3.2.3` 锁定要求的 `numpy==1.24.4` 冲突。
- **修复**：显式钉住两者版本，选一个同时兼容 `numpy==1.24.4` 的旧版 `matplotlib`：
  ```bash
  uv pip install "matplotlib==3.7.5" "numpy==1.24.4"
  ```
- **验证**：`python -c "import libero.libero.envs; import numpy; print(numpy.__version__)"` 正常返回 `1.24.4`，无 ImportError。

#### #4 `_pickle.UnpicklingError`：`torch.load` 读取 LIBERO `init_states` 失败

- **现场**：跑 `eval_libero_plus.py` 时在 `libero/libero/benchmark/__init__.py` 里 `torch.load(init_states_path)` 抛 `_pickle.UnpicklingError`（反序列化 `*.pruned_init`/`*.init` 文件失败）。
- **根因**：PyTorch≥2.6 把 `torch.load` 的默认参数从 `weights_only=False` 改成了 `weights_only=True`（安全加固，防止反序列化任意对象），而 LIBERO/LIBERO-plus 的 init states 文件本质是纯 numpy 数组的旧式 pickle，不在 `weights_only=True` 的白名单反序列化范围内，因而被拒绝。
- **修复**：在 LIBERO-plus 仓库本地补丁两处 `torch.load` 调用，显式传入 `weights_only=False`（这是本地可信资产，不是远程/不可信 checkpoint，可以安全地退回完整反序列化）：
  ```python
  # weights_only=False: PyTorch>=2.6 defaults to weights_only=True, which
  # rejects the numpy-array pickles LIBERO/LIBERO-plus ships as
  # `*.pruned_init`/`*.init` files. These are trusted local assets (not
  # remote/untrusted checkpoints), so it is safe to opt back into full
  # unpickling here. See b/d/p/reprd_liberop_cam_rb.md problem log.
  init_states = torch.load(init_states_path, weights_only=False)
  ```
  分别在 `get_task_init_states`（约第191行）与另一处备用取值分支（约第252行）各改一次。
- **验证**：重跑 mock 策略冒烟测试，`libero_spatial` 的 3 个任务能正常 `env.reset(init_state=...)`，不再抛 `UnpicklingError`，产出 `overall_results.json`（见时间线第6条）。

#### #5 `flash-linear-attention` 在 H200 (Hopper) + Triton≥3.4 上计算结果错误

- **现场**：第一次训练冒烟测试（`train_smoketest1`）在前向/反向经过 InternVLA-A1.5 的 Gated DeltaNet 层时，`fla` 库抛出：
  ```
  RuntimeError: Triton >= 3.4.0 on Hopper GPUs produces incorrect results for gated chunk_bwd_dqkwg ...
  ```
  （`flash-linear-attention` 自带的运行时安全检查主动拦截，而不是崩溃或算错了才发现。）
- **根因**：这是 `flash-linear-attention` 已知的上游问题——较新版本 Triton 编译器在 Hopper（sm90，即 H200/H100）架构上，为该库某些 gated chunk kernel 生成的机器码在特定条件下数值不正确；库作者的规避方案是检测到"Hopper + 新版 Triton"组合时，改用 `tilelang` 实现的等价 kernel 代替原生 Triton kernel。
- **修复**：按错误信息提示安装 `tilelang`：
  ```bash
  uv pip install tilelang
  ```
- **验证**：重跑训练冒烟测试（`train_smoketest2`），Gated DeltaNet 相关前向/反向不再报错，30 步顺利跑完并保存 checkpoint。

#### #6 【关键】`torchcodec`/`torch` 版本不匹配，导致训练时几乎所有视频帧解码静默失败、回退成全黑帧（数据静默损坏）

- **现场**：`train_smoketest2` 的 30 步（4 卡 × batch16 × 30 step ≈ 1920 个样本，每样本 2 路相机、每路又要在时间维采样若干帧）里，日志出现 **1504 条** `dataset.py:1010 [video_decode_error]`，覆盖 `libero_spatial/object/goal/10` 全部四个套件、`observation.images.image` 与 `observation.images.wrist_image` 两路相机。摘录一条完整报错（其余重复同一根因）：
  ```
  [video_decode_error] repo_id=libero_goal ep_idx=358 vid_key=observation.images.wrist_image ...
  error=RuntimeError('Could not load libtorchcodec. Likely causes:
    1. FFmpeg is not properly installed ...
    2. The PyTorch version (2.10.0+cu128) is not compatible with this version of TorchCodec ...
  [start of libtorchcodec loading traceback]
  ...
  OSError: libnvrtc.so.13: cannot open shared object file: No such file or directory
  ...')
  ```
- **根因（两层）**：
  1. **版本不兼容**：`pip install torchcodec` 装到的是最新 `0.15.0`，而 TorchCodec 官方兼容表（<https://github.com/pytorch/torchcodec>）明确写着 `torchcodec 0.15` 要求 `torch>=2.11`，本仓库锁定的是 `torch==2.10.0`（配套 `torchcodec 0.10` 才对）。
  2. **CUDA 版本不匹配**：更本质的是，Linux 上 `pip install torchcodec` **默认装的是 CUDA 版 wheel**（官方 README："this will install CUDA-enabled wheels by default"），而这批默认 CUDA wheel 是按最新 CUDA Toolkit（13.x）编译的，需要 `libnvrtc.so.13`；本机 `torch==2.10.0+cu128` 对应 CUDA 12.8 工具链，`nvidia-cuda-nvrtc-cu12` 包只提供 `libnvrtc.so.12`，系统级 CUDA 也只有 12.x 的 `libnvrtc.so.12`，找不到 `.so.13`，于是 TorchCodec 底层 `libtorchcodec_core*.so`（对 FFmpeg 4/5/6/7/8 各编译一份）全部加载失败。
  3. **失败被静默吞掉**：更麻烦的是 [`lerobot_dataset.py`](../../src/lerobot/datasets/lerobot_dataset.py) 里 `_query_video_frames` 对 `decode_video_frames` 做了 `try/except`，解码失败时只打一条 `logging.error` 就 `frames = torch.zeros(...)` 回退成全黑帧继续训练（用意是避免单个坏文件炸掉整个训练，但代价是这里报错没有被任何人实时盯着看，几乎 100% 的帧都在喂全黑图像给 VLM 视觉分支和 WAN 视频前瞻分支，训练本身"看起来"正常运行、loss 也在下降，但学到的是"全黑输入→随手拟合"的退化解，`loss_video` 会长期偏低但没有意义）。这是本次复现过程中**影响最大、最隐蔽**的一个问题：如果没有逐行核对训练日志、只看 loss 曲线和退出码，很容易在错误的数据上把 100k 步训练完才发现分数异常。
- **修复**：在 `ivla15` venv 里把 `torchcodec` 换成与 `torch==2.10.0` 匹配、且不依赖 CUDA13 库的 **CPU-only 0.10.x** 版本（数据加载阶段的视频解码本身在 CPU dataloader worker 里进行，不需要 GPU 解码加速）：
  ```bash
  source /mnt/r/VENV/ivla15/bin/activate
  uv pip install "torchcodec==0.10.*" --index-url=https://download.pytorch.org/whl/cpu
  ```
- **验证**：
  1. 单元验证：直接用 `torchcodec.decoders.VideoDecoder` 打开一个此前报错的 `libero_goal` wrist camera mp4，`get_frames_played_at` 成功返回 `(1, 3, 256, 256)` 的真实像素张量（`mean≈110.79`，非全零），确认解码恢复正常。
  2. 端到端验证：重新启动真正的 4 套件联合微调（时间线第10条），前 ~4 分钟、前若干个 log step 内 `grep -c video_decode_error` = 0（此前 30 步就有 1504 条），确认修复对真实训练数据全面生效。
  3. **后续行动**：`train_smoketest2`（在 bug 修复前跑的 30 步）产生的 checkpoint 不可用于任何后续评估，已弃用，不纳入正式训练产物；正式训练已用修复后的环境重新从 InternVLA-A1.5-base 权重开始。

#### #7 policy server 缺 `websockets`/`msgpack`/`msgpack-numpy`

- **现场**：用 `ivla15` venv 启动 `evaluation/LIBERO/policy_server/server_policy.py`（无论是 `--mock_policy` 冒烟测试还是后续真实模型评测）时报 `ModuleNotFoundError: No module named 'websockets'`。
- **根因**：主仓库 `pyproject.toml` 面向训练场景，没有把评测用的 websocket 服务端依赖（`websockets`、`msgpack`、`msgpack-numpy`）列为核心依赖，这些包只在 `evaluation/` 目录的脚本里直接 import。
- **修复**：
  ```bash
  source /mnt/r/VENV/ivla15/bin/activate
  uv pip install websockets msgpack msgpack-numpy
  ```
- **验证**：`server_policy.py --mock_policy` 正常启动并监听端口，`eval_libero_plus.py` 客户端可以成功 `WebsocketClientPolicy(...).predict_action(...)` 建立连接（见时间线第6条 mock 冒烟测试）。

#### #8 【关键、耗时最长】正式4套件联合微调反复卡在 `Accelerator()`/`TCPStore` 分布式握手，长达数小时排查后发现根因是**后台启动方式**而非 PyTorch/网络/环境问题

这是本次复现中排查耗时最长、最容易得出错误结论的一个问题，完整记录排查过程以警示后来者不要重复踩同样的弯路。

- **现场**：连续 3+ 次尝试正式启动 `accelerate launch --multi_gpu --num_processes=4 src/lerobot/scripts/lerobot_train.py`（GPU4-7），进程稳定重现同一种"假死"：4 个 rank 进程都能正常完成 CLI 解析、打印 4 条 `WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.`，之后**再无任何输出**，GPU4-7 显存长期保持 `0 MiB`（说明连模型加载都还没开始），CPU 占用极低，`ps` 显示进程状态为 `S (sleeping)`。等待 10~15 分钟后（个别情况下更久）会看到：
  ```
  [E728 15:01:59.561275187 socket.cpp:1028] [c10d] The client socket has timed out after 600000ms while trying to connect to (127.0.0.1, 48315).
  [W728 15:01:59.562622567 TCPStore.cpp:340] [c10d] TCP client failed to connect/validate to host 127.0.0.1:48315 - retrying (try=0, timeout=600000ms, delay=33924ms): The client socket has timed out after 600000ms while trying to connect to (127.0.0.1, 48315).
  ```
  而且这条"超时后自动重试"的消息会不断重复出现（每次重试本身又要再等 600 秒 + 一个随机 backoff delay），进程本身并不会退出，看起来像是"永远卡住"。

- **排查过程（逐步排除的错误假说）**：
  1. **假说A：端口被占用 / `MASTER_ADDR` 解析成了机器名而非 `127.0.0.1`。** 早期确实遇到过 `MASTER_PORT=6379` 被本机一个 Ray `gcs_server` 占用、导致 `accelerate` fallback 到主机名解析的情况（这是一个真实存在但已经修复的独立问题：每次启动前用 Python `socket.bind(('127.0.0.1', 0))` 探测一个空闲端口）。但换了空闲端口后（`58431`→`40333`→`48315`→…）问题依然反复出现，说明这不是（唯一）根因。
  2. **假说B：PyTorch≥2.4 默认的 libuv TCPStore 后端在本机环境下有 bug。** 用 `py-spy dump --pid <rank0_pid>` 抓取卡住进程的 Python 调用栈，4 个 rank 无一例外全部停在：
     ```
     _create_c10d_store (torch/distributed/rendezvous.py:191)
     _env_rendezvous_handler (torch/distributed/rendezvous.py:281)
     init_process_group (torch/distributed/distributed_c10d.py:1806)
     __init__ (accelerate/state.py:244)
     __init__ (accelerate/accelerator.py:462)
     train (lerobot_train.py:163)
     ```
     即卡在 `TCPStore(...)` 这个 native 构造调用里。检索到 PyTorch 官方文档确认存在"libuv 后端在部分环境下会挂起，需要 `USE_LIBUV=0` 回退到旧后端"的已知问题（<https://github.com/pytorch/pytorch/pull/127957>，<https://docs.pytorch.org/tutorials/intermediate/TCPStore_libuv_backend.html>），于是在 `internvla_a15_finetune_libero_venv.sh` 里加了 `export USE_LIBUV=${USE_LIBUV:-0}`。**这个改动被保留在脚本里（无副作用、也是社区推荐的稳健设置），但事后验证它并不是本次卡死的真正原因**——设置后问题依然 100% 复现。
  3. **假说C：主机整体负载过高，导致 rank0 的 `TCPStore(is_master=True)` 里 `bind()+listen()` 迟迟执行不到。** 当时机器上有其它用户的 `data_juicer`/`EgoDex` 数据处理任务，4 个进程各吃 500%~1300% CPU，`uptime` 显示 load average ~50；GPU0-3 也被其它任务占满。为验证是否是主机级资源竞争导致，写了一个极简的 `dist_sanity.py`（只有 `from accelerate import Accelerator; Accelerator()`），用**完全相同的 GPU4-7、完全相同的 host 负载条件**跑 `accelerate launch --multi_gpu --num_processes=4 dist_sanity.py`——**稳定在 2~3 秒内完成**，且反复复测均如此。这直接排斥了"主机资源竞争"作为主要根因（否则这个极简脚本也应该同样变慢）。
  4. **假说D：`lerobot_train.py` 的重量级 import（transformers/flash-attn/fla/WAN 相关模块）或 `LD_LIBRARY_PATH`（脚本里 prepend 了 `${VENV_ROOT}/lib`，可能覆盖系统网络库）导致 TCPStore 原生调用变慢。** 分别构造了"只加重量级 import、不做 CLI 解析和模型加载"和"用与训练脚本完全相同的 `LD_LIBRARY_PATH`/`OMP_NUM_THREADS`/`CUDA_HOME` 等环境变量"两个对照实验，均在几秒内顺利完成 `Accelerator()`。这排除了导入开销和环境变量覆盖两个假说。
  5. **真正根因（通过 `ps -eLf`/`/proc/<pid>/environ` 找到）**：检查卡住进程的环境变量，发现 `TORCHELASTIC_USE_AGENT_STORE=True`——这意味着 `accelerate launch --multi_gpu`（底层走的是 `torch.distributed.elastic` / torchrun 风格的 launcher）里，**TCPStore 的服务端实际上是由"`accelerate launch` 自身这个顶层进程"（torchelastic 的 elastic agent）来托管的，4 个 rank 子进程（含 rank0）全部只是这个 store 的客户端**，而不是像"静态 c10d rendezvous"里那样由 rank0 自己充当服务端。再检查卡住的 4 个 worker 进程的父进程（`ps -o ppid=`），发现**它们的 `PPID` 全部是 `1`（已被 init 收养）**——也就是说，本该负责托管 TCPStore 服务端、并在训练全程存活监控 4 个 worker 的"`accelerate launch` 顶层代理进程"在某个时间点**已经意外退出/被杀死**，只留下 4 个被孤立（orphaned）的 worker 子进程在苦苦地、一次又一次地对着一个再也没有人监听的端口发起 600 秒超时的 `connect()` 重试——这才是日志里反复出现"retrying (try=0, timeout=600000ms, ...)"、进程"看起来卡住但其实是在合法地重试一个永远不会成功的连接"的真正原因。
  6. **为什么"agent 代理进程会意外死掉"**：本次所有失败的训练尝试，都是通过 `nohup bash launch/....sh > log 2>&1 & disown` 这种"手动后台化"方式在 Shell 工具的一次调用内部启动的（包括更早期版本里为了自动重试而写的 `internvla_a15_finetune_libero_venv_watchdog.sh`，其内部也是用同样的 `nohup ... & disown` 模式拉起真正的训练脚本）。在本次所用的沙箱化终端环境里，这种"在一次工具调用内部 `nohup+&+disown` 出去、工具调用本身很快返回"的模式，被观察到会导致：**bash 包装脚本以及它 fork 出来的 `accelerate launch` 顶层代理进程，在该次工具调用结束后的某个时间点被意外终止**（很可能是该沙箱环境对每次命令调用绑定的会话/进程组做了清理，即使用了 `nohup`/`disown` 也未必能完全豁免），而已经由 torchelastic 用 `multiprocessing`/`subprocess` 方式派生出去的 4 个 worker 子进程，因为已经脱离了那个进程组，得以继续存活，从而变成了"孤儿"。
- **验证修复方式的决定性实验**：把完全相同的启动命令，改为**不做任何手动 `nohup`/`&`/`disown`，直接把 `bash launch/internvla_a15_finetune_libero_venv.sh` 作为前台命令交给 Shell 工具执行、由工具自身的"命令超时未完成则自动转入后台并持续把输出流向日志文件"的机制来后台化**（即 `Shell` 工具文档里说明的推荐用法：让工具自己管理长任务的生命周期，而不是脚本内部再叠加一层 `nohup`）。用这种方式启动后：
  1. 20-step 小规模验证：`Accelerator()`/`TCPStore` 握手在 1~2 秒内完成，训练 20 步顺利跑完、保存 checkpoint、`grep -c video_decode_error` = 0，进程正常退出（`exit_code: 0`）。
  2. 正式 100k-step 训练（`JOB_NAME=a15_libero4suite_100k_20260728_151045`）用同样方式启动后，`Accelerator()` 同样秒级完成，训练顺利进入主循环，第 200 步日志：
     ```
     step:200.0 | sample:13K | episode:79 | epoch:0.05 | loss:6.143 | loss_action:0.229 | ... | loss_video:0.134 | ...
     ```
     `video_decode_error` 计数为 0，GPU4-7 显存占用 ~131GB/卡、利用率 57%~92%，符合预期。
- **结论与经验教训**：
  1. **真正的根因是"后台化方式"，不是 PyTorch/libuv/环境变量/主机负载**——`USE_LIBUV=0` 这个改动虽然被保留在脚本里，但经过对照实验证明它对本次卡死**没有实质性帮助**（加与不加都 100% 复现相同的挂起现象），只是social好实践本身无害，故未回退。
  2. **诊断长跑训练任务时，`nohup ... & disown` 并不是在所有终端/沙箱环境下都绝对可靠的后台化手段**；本仓库后续所有长时间训练/评测任务的启动方式统一改为"直接把启动命令交给 Shell 工具执行，由工具自身的后台化机制接管"，不再手动 `nohup+&+disown`。
  3. **`py-spy dump` + 检查 `/proc/<pid>/environ`（尤其是 `TORCHELASTIC_USE_AGENT_STORE`）+ 检查 `ps -o ppid=`（判断进程是否被孤立/收养给 init）** 是诊断"分布式训练卡在 rendezvous 阶段"问题的一套通用、高效的方法论，比单纯猜测端口/网络问题更快定位到根因。
  4. **对照实验（用最小可复现示例逐步排除假说）是关键**：`dist_sanity.py`（纯 `Accelerator()`）→ `dist_sanity_heavy_imports.py`（+重量级 import）→ 相同环境变量的 `dist_sanity.py` —— 这三步实验依次排除了"host 负载"、"import 开销"、"环境变量覆盖"三个看似合理但实际错误的假说，避免了在错误方向上继续无限调试。
  5. 之前草拟的 `launch/internvla_a15_finetune_libero_venv_watchdog.sh`（用 `nohup+&+disown` 拉起训练并轮询日志判断是否要重试）**同样带有这个根因缺陷**，其自动重试机制表面上"看起来在工作"（不断换端口重试），但实际上每次新起的尝试也会遇到同样的 agent 被孤立问题，只是长时间跑下去有一定概率某次尝试恰好在被杀死前跑完握手——**这不是可靠的修复，只是掩盖了问题**。最终采用的做法是放弃这个 watchdog 脚本的自动重试机制，改为用 Shell 工具原生的前台阻塞+自动转后台能力直接启动训练。

#### #9 正式训练在 step≈35200 处无预警"消失"，根因是外部终端/IDE 会话被重置杀掉了 `accelerate launch` 代理进程（而非代码或环境错误）

- **现场**：2026-07-29 04:16 (UTC+8 12:16) 用户反馈"训练进程好像不见了"。核查发现：
  - `ps aux | grep lerobot_train` 无任何匹配，GPU4-7 全部 `0 MiB / 0%`——训练进程确实已经不存在了。
  - 训练日志 `outputs/.../a15_libero4suite_100k_20260728_151045`（经 `/mnt/r/tmp/train_full_run_final.log`）最后一条记录是 `2026-07-29 01:04:53`、`step:35.2K/100K`（仅完成 35.2%），之后没有任何后续日志、没有 Python traceback、没有 `Checkpoint saved`/训练结束提示——即"戛然而止"，不是正常退出也不是显式报错退出。
  - `outputs/.../checkpoints/last` 指向 `035000`（2026-07-28 23:38 之后保存，`01:01:00` 日志确认"Checkpoint saved at .../035000"），说明训练在保存完 035000 后又跑了约 4 分钟（到 35.2K）才停止。
- **排查过程**：
  1. **排除主机重启/OOM**：`uptime` 显示主机已连续运行 18 天 20 小时（上次重启是 07-10），`last reboot` 无新记录；`dmesg` 无 OOM-killer 相关信息。说明不是宿主机层面的硬故障。
  2. **关键线索——终端会话文件被重建**：Cursor 的 `terminals/` 目录里此前记录本次训练所在 shell 会话的文件已经不存在，取而代之的是两个全新、体积极小（233 字节，刚创建、几乎没有历史输出）的终端文件 `2.txt`/`3.txt`，其内部记录的 shell 启动时间恰好是 **`2026-07-29 01:01:30`**——与训练日志"戛然而止"的时间点（01:01:00 保存 checkpoint，01:04:53 最后一条训练日志）高度吻合。这强烈提示：托管本次训练 `accelerate launch` 顶层代理进程的那个终端/会话，在 01:01 前后被外部因素（很可能是用户重启了 Cursor IDE 或该会话被系统整体回收）整体终止了。
  3. 这与问题记录 #8 是**同一类根因**（"托管 TCPStore 服务端的 `accelerate launch` 顶层代理进程被杀 → 4 个 worker 变成孤儿"），只是这次的"杀手"不是我们自己手动 `nohup+&+disown` 造成的，而是承载该 Shell 会话的外部宿主（IDE/终端）本身被重置——说明**任何会把整条会话/进程组连根拔起的操作，都会终结这类长跑训练**，这是使用交互式终端跑多小时训练任务时必须警惕的通用风险，不局限于某一种具体的后台化写法。
- **修复（利用已有 checkpoint 续训，而非从头重跑）**：
  1. 确认 `checkpoints/035000` 目录下 `training_state/`（`optimizer_state.safetensors`、`scheduler_state.json`、`rng_state.safetensors`、`training_step.json`）与 `pretrained_model/`（`model.safetensors`、`train_config.json`）均完整落盘。
  2. 新增 `launch/internvla_a15_finetune_libero_venv_resume.sh`：复用 lerobot 原生的续训机制——`src/lerobot/configs/train.py::validate()` 在 `resume=true` 时会从 `--config_path=<ckpt>/pretrained_model/train_config.json` **完整加载整份 `TrainPipelineConfig`**（`policy.pretrained_path`/`checkpoint_path` 均从该 json 所在目录自动推导），因此续训命令只需保留 `accelerate launch` 自身的分布式启动参数（`--multi_gpu --num_processes=4 ...`）+ `--config_path=... --resume=true` 两项，不需要重复整套 `--policy.*`/`--dataset.*` 训练超参。换了一个新端口（`MASTER_PORT=6380`）避免与任何残留监听冲突。
  3. **严格用 Shell 工具原生的"前台执行、超时自动转后台"机制启动**（`bash .../_resume.sh > /mnt/r/tmp/train_resume_35k.log 2>&1`，未加任何手动 `nohup`/`&`/`disown`），与问题记录 #8 的最终结论保持一致。
- **验证**：
  - 4 个 worker 进程稳定运行在 GPU4-7（`nvidia-smi` 显示 4 卡各 ~131GB/94-100%），`Accelerator()` 秒级完成握手，无 TCPStore 超时重试。
  - 恢复后第一条训练日志：`step:35.2K | loss:1.077 | loss_action:0.011 | lr:3.8e-05`——与中断前最后一条日志（`step:35.2K | loss:1.130 | lr:3.8e-05`）**处于同一数量级、学习率完全衔接**（同一 cosine decay 曲线上的同一点），证明 optimizer/scheduler/rng 状态确实被正确恢复，不是"看似继续、实际从随机初始化重新学"。
  - `bash launch/internvla_a15_finetune_libero_venv_resume.sh > /mnt/r/tmp/train_resume_35k.log 2>&1`
- **结论与经验教训**：
  1. 这不是代码 bug，也不是本次环境配置的问题，而是"长跑任务托管在一次性交互式终端会话里，会话被外部重置就会连带杀死训练"这一通用风险的又一次体现（呼应问题记录 #8）。后续如仍需长跑（例如续跑到 100k 步全程还需 ~18 小时），建议用户避免在此期间重启/关闭承载该 Shell 会话的 IDE 窗口；若无法保证，应考虑改用 `systemd-run --scope`/`screen`/`tmux` 等与终端会话生命周期解耦的方式承载训练进程（本次未采用，因为问题记录 #8 已确认原生 Shell 工具后台化在"未被外部杀死"的前提下是可靠的，重启风险属于用户操作层面，非工具或代码缺陷）。
  2. `training_state/`（optimizer + scheduler + rng）随每次 `save_freq` checkpoint 一起落盘，是本次能够零损失续训的关键；`SAVE_FREQ=5000` 意味着最坏情况下只损失了不到 5000 步（本例中实际只损失了约200步，从 35000 到 35200 之间）的训练进度。

### 文件变更清单

| 文件 | 类型 | 原因 |
|---|---|---|
| `evaluation/LIBERO-plus/eval_libero_plus.py` | 修改 | 新增 `_parse_categories()` 与 `--categories` CLI 参数，评测循环里按 `task_classification.json` 的类别过滤 task_id，只跑 `Camera Viewpoints`/`Robot Initial States`，其余5类整段跳过（不仿真、不计入统计），节省约69%算力；聚合产物格式（`logs/<suite>/<start>_to_<end>.json`）不变，`aggregate_results.py` 无需改动。 |
| `evaluation/LIBERO-plus/run_eval_libero_plus_venv.sh` | 新增 | 基于 `run_eval_libero_plus.sh` 改写的 venv 版评测编排脚本：server 端用 `ivla15` venv、client 端用 `ivla15_libero_plus_client` venv（`conda activate`→`source .../bin/activate`），透传 `--categories`，GPU 探测逻辑保留但默认 `GPU_IDS=4,5,6,7`（本次复现实际使用的 GPU），沿用原脚本的 flock 任务队列 + 每套件按 `SHARDS_PER_SUITE` 分片 + 健康检查 + 自动聚合。 |
| `launch/internvla_a15_finetune_libero_venv.sh` | 新增+修改 | 基于 `launch/internvla_a15_finetune_libero.sh` 改写的 venv 版四套件联合微调脚本：venv 激活、`PRETRAINED_PATH`/WAN 三个路径指向本地 `/mnt/r/CKPT/` 缓存（避免重复联网下载）、`DATASET_REPO_ID` 显式写死四个套件名（原脚本靠目录名 glob 发现 `*_no_noops*_lerobot`，本地数据目录名与之不匹配）、默认 `CUDA_VISIBLE_DEVICES=4,5,6,7`/`PROC_PER_NODE=4`，其余训练超参（`steps/batch_size/lr/warmup/decay/action_loss_only=false/freeze_learnable_tokens=false` 等）与原脚本保持一致；后续追加 `export USE_LIBUV=${USE_LIBUV:-0}`（问题记录 #8 排查过程中的缓解措施，经对照实验证明非本次卡死的根因，但作为社区推荐的稳健设置予以保留）。 |
| `launch/internvla_a15_finetune_libero_venv_watchdog.sh` | 新增（后废弃不用） | 排查问题记录 #8 过程中编写的"自动换端口重试"看护脚本，内部仍用 `nohup+&+disown` 拉起真正的训练脚本。事后确认该脚本与被排查的根因（后台化方式导致 agent 代理进程被孤立）存在同样的缺陷，其重试机制只是掩盖问题而非真正修复，故未采用；最终改为用 Shell 工具原生后台化机制直接启动训练（见问题记录 #8）。保留此文件仅作为排查过程的记录，未在正式训练流程中使用。 |
| `/mnt/r/tmp/dist_sanity.py`、`/mnt/r/tmp/dist_sanity_heavy_imports.py` | 新增（诊断脚本，非仓库文件） | 问题记录 #8 排查过程中编写的两个最小可复现对照实验：前者验证纯 `Accelerator()` 初始化在相同 GPU/相同主机负载下能秒级完成（排除主机资源竞争假说），后者额外加入 `torch/transformers/flash_attn/fla` 重量级 import（排除导入开销假说）。 |
| `/mnt/r/DATA/LIBERO-plus/libero/libero/benchmark/__init__.py`（仓库外部数据目录，非本代码库文件） | 修改（本地补丁） | 两处 `torch.load(init_states_path)` 改为 `torch.load(init_states_path, weights_only=False)`，修复 PyTorch≥2.6 默认值变化导致的 `init_states` 反序列化失败（问题记录 #4）。 |
| `ivla15` venv 内 `torchcodec` 包 | 环境变更（非仓库文件） | `0.15.0`（CUDA13 wheel，与 `torch==2.10.0+cu128` 不兼容）→ `0.10.0`（CPU-only wheel，`torch==2.10.0` 官方兼容版本）。修复问题记录 #6 的静默视频解码失败/全黑帧问题，是本次复现最关键的一处修复。 |
| `ivla15` venv 内增量依赖 | 环境变更（非仓库文件） | 新增 `websockets`、`msgpack`、`msgpack-numpy`（问题记录 #7）、`tilelang`（问题记录 #5）。 |
| `ivla15_libero_plus_client` venv 内增量依赖 | 环境变更（非仓库文件） | 新增 `matplotlib==3.7.5`（重新钉回 `numpy==1.24.4`，问题记录 #3）、`cloudpickle`、`gym`（`libero.libero.envs` 间接依赖，装 LIBERO-plus 包时暴露出的缺口）。 |
| `/mnt/r/DATA/LIBERO-plus/libero/libero/assets/` | 数据目录重排（非仓库文件） | 修正 `assets.zip` 解压后的嵌套路径问题（问题记录 #2），使其符合 LIBERO-plus README 约定的 `libero/libero/assets/` 根路径结构。 |
| `data`（仓库根目录软链）、`/mnt/r/CKPT/hf_home/lerobot/libero_{spatial,object,goal,10}`（软链） | 新增（软链，非常规文件） | 让 `HF_LEROBOT_HOME`/`repo_id` 解析直接命中本地已下载数据，避免 `lerobot_dataset.py` 触发任何远程下载。 |
| 四个套件各自的 `meta/info.json` 中 `robot_type` 字段 | 修改（数据元信息，脚本自动执行） | 由训练脚本第一次启动时自动补丁为对应套件名（`libero_spatial`/`libero_object`/`libero_goal`/`libero_10`），保证多子集统计量（`stats.json`）不互相覆盖，并与评测端 `STATS_KEY_MODE=suite`/`ROBOT_TYPE_MODE=suite` 对应。 |
| `launch/internvla_a15_finetune_libero_venv_resume.sh` | 新增 | 问题记录 #9：正式训练在 step≈35200 处因外部终端会话被重置而中断后，用于从 `checkpoints/last`（`035000`）续训的脚本。只保留 `accelerate launch` 的分布式启动参数 + `--config_path=<ckpt>/pretrained_model/train_config.json --resume=true`（lerobot 原生续训机制会从该 json 完整还原 `policy.*`/`dataset.*`/`optimizer`/`scheduler` 全部配置），换用新端口 `MASTER_PORT=6380`，同样用 Shell 工具原生后台化机制启动（不用 `nohup+&+disown`）。 |
| `outputs/internvla_a1_5/a15_libero4suite_100k_20260728_151045/checkpoints/{005000,...,100000,last}` | 新增（训练产物，非仓库代码） | 100k-step 联合微调的全部 checkpoint（每 5000 步一个），`last` 软链到 `100000`，作为本次评测所用的最终模型。 |
| `outputs/sim_eval/libero_plus/20260729_230048_camrb_full/`（含 `overall_results.json`、`logs/<suite>/<start>_to_<end>.json`、`worker_gpu*/`） | 新增（评测产物，非仓库代码） | 8卡并行跑四套件 Camera+Robot 全量评测（约3149个任务）的完整输出，`overall_results.json` 是最终结论所依据的聚合结果文件。 |

### 最终结果

> 状态：**复现已完成**。Robot 指标在验收容差内成功复现；Camera 指标存在明显差距，根因分析见下（判定为训练规模/数据配方差异导致的能力差距，而非评测流程 bug）。

#### 训练过程回顾

- **训练启动（正式生效版本）**：`JOB_NAME=a15_libero4suite_100k_20260728_151045`，2026-07-28 15:10:45 于 GPU4-7 启动，`accelerate launch --multi_gpu --num_processes=4`，`steps=100000, batch_size=16(每卡)×4卡=64(全局), lr=5e-5`，与 Part A 手册第4节参数一致；`torchcodec` bug 修复 + 改用 Shell 工具原生后台化机制（问题记录 #8）后启动，第 200 步日志确认 `video_decode_error=0`。此前 `a15_libero4suite_100k_20260728_135651`（13:56:51 首次尝试）从未真正进入训练循环（卡在分布式握手阶段），已作废，不作为正式训练产物。
- **中断与续训（问题记录 #9）**：该正式训练跑到 `step≈35200/100000`（`2026-07-29 01:04:53` 前后）时，因承载该 Shell 会话的终端/IDE 被外部重置而意外中断（非代码错误）。已于 `2026-07-29 04:19` 用 `launch/internvla_a15_finetune_libero_venv_resume.sh` 从 `checkpoints/last`（`035000`，含完整 `optimizer/scheduler/rng` 状态）成功续训，恢复后第一条日志（`step:35.2K | loss:1.077 | lr:3.8e-05`）与中断前最后一条（`loss:1.130 | lr:3.8e-05`）无缝衔接。
- **训练完成**：`2026-07-29 22:55:09`（UTC），全部 100,000 步正常跑完，`exit_code=0`，日志打印 `End of training`，最终 checkpoint `100000` 落盘。全程（含续训后半段）吞吐稳定在 `~1.0 iters/s`，实际总训练墙钟时间（不含中断的~3小时空档）约 27 小时。最终 `loss=0.265`（`loss_action=0.006`，`loss_vqa=0.140`，`loss_video=0.066`，`loss_fast=0.160`），相比初始（`step≈200, loss≈6.1`）下降充分、曲线平滑，无发散/震荡迹象。

#### 真实推理抽检（早期验证）

用最终 checkpoint 起 policy server（`--stats_key libero_spatial --robot_type libero_spatial`），对 `libero_spatial` 的 2 个 `Robot Initial States` 任务做真实闭环推理，**2/2 成功（100%）**，验证了"checkpoint → policy server（`ivla15` venv）→ LIBERO-plus 客户端（`ivla15_libero_plus_client` venv）→ 仿真回放"整条链路正确可用（图像预处理、动作反归一化、gripper 二值化、坐标系约定均无误）。

#### Camera + Robot 全量评测结果

`2026-07-29 23:00:48` 用 8 卡（GPU0-7 全空闲，`SHARDS_PER_SUITE=8`）启动 `run_eval_libero_plus_venv.sh`，覆盖四个 LIBERO 套件的全部 `Camera Viewpoints` + `Robot Initial States` 任务，`2026-07-30 03:20:58` 完成（耗时约 4 小时20分），`aggregate_results.py` 聚合结果（`outputs/sim_eval/libero_plus/20260729_230048_camrb_full/overall_results.json`）：

| 类别 | 本次复现 | 论文 Table 6 目标 | 差值 | ±5pp 验收 |
|---|---|---|---|---|
| **Robot Initial States** | **50.6%**（785/1550） | ≈55% | −4.4pp | ✅ **达标** |
| **Camera Viewpoints** | **44.6%**（713/1599） | ≈83% | −38.4pp | ❌ **未达标** |
| Total（两类合计） | 47.6%（1498/3149） | — | — | — |

按套件拆分（`overall_results.json.per_suite`）：

| 套件 | Robot Initial States | Camera Viewpoints | 该套件总体（含两类） |
|---|---|---|---|
| libero_spatial | 50.3% (176/350) | 38.6% (145/376) | 44.2% |
| libero_object | 61.6% (245/398) | 67.4% (267/396) | 64.5% |
| libero_goal | 33.5% (137/409) | 27.9% (114/408) | 30.7% |
| libero_10 | 57.8% (227/393) | 44.6% (187/419) | 51.0% |

按扰动难度等级拆分（`difficulty_level` 1=最轻, 5=最重；对 8 个 worker 日志里逐条 `task_id=... -> n/m` 记录按 `task_classification.json` 的 `difficulty_level` 重新聚合得到）：

| difficulty_level | Robot Initial States | Camera Viewpoints |
|---|---|---|
| 1（最轻） | 77.8% (193/248) | 62.3% (124/199) |
| 2 | 62.3% (182/292) | 48.3% (174/360) |
| 3 | 55.6% (208/374) | 42.2% (137/325) |
| 4 | 38.2% (109/285) | 50.0% (114/228) |
| 5（最重） | 26.5% (93/351) | 33.7% (164/487) |

#### Camera 指标未达标的根因分析

在下结论之前，先排查了"是否是评测流程/代码 bug"这一假说，结论是**否**：

1. **排除"扰动未真正生效"假说**：核对 LIBERO-plus 源码（`libero/libero/envs/bddl_base_domain.py`、`libero/libero/envs/problems/libero_tabletop_manipulation.py`）确认相机视角扰动是通过每个扰动任务专属的 `bddl` 场景文件（`horizon_view`/`vertical_view` 参数驱动 `mujoco_arena.set_camera(camera_name="agentview", pos=..., quat=...)`）烘焙实现的，观测键名仍是标准的 `agentview_image`（`evaluation/LIBERO-plus/eval_libero_plus.py:84` 读取的键与 LIBERO 原生一致），不需要评测脚本额外传参适配——即扰动本身是"场景级"的，只要 `task_suite.get_task(task_id)` 拿到的是扰动版 bddl（本次已确认拿到，因为 `task_classification.json` 里的 `id` 直接对应扰动任务），相机扰动就必然生效。
2. **排除"图像预处理/尺寸不匹配"假说**：训练时 `resize_with_pad` 到 `224×224`（`train_config.json` 确认），评测时 `--resize_size 224` 完全一致，无缩放/裁切差异。
3. **排除"评测端出现大面积异常（如全黑帧、超时）"假说**：按 `difficulty_level` 重新聚合后，Camera 和 Robot 两个类别都呈现**平滑、符合直觉的单调递减趋势**（难度越高成功率越低，`difficulty=1` 时 Camera 62.3%/Robot 77.8%，`difficulty=5` 时 Camera 33.7%/Robot 26.5%），而不是"要么全成功要么全失败"的断崖式模式——这种梯度化的退化模式是"模型能力随任务难度自然下降"的典型特征，而非代码 bug（bug 通常表现为某个子集近乎 0% 或 100%，或報错/超时）。
4. **结论**：即使只看 Camera 里最容易的 `difficulty=1` 子集，成功率也只有 62.3%，仍明显低于论文报告的"全部难度等级平均 83%"——说明差距不是集中在某个特别难的子集被拖累，而是**本次复现出的 checkpoint 在相机视角泛化能力上整体弱于官方 checkpoint**，属于训练规模/数据配方层面的差距，而非评测脚本或环境配置的实现错误。

结合 Part A 手册第7节"已知限制"，本次复现相对官方设置的主要偏差可能是造成该差距的原因（按可能性从高到低排列，均为推测，未逐一验证，因手册验收标准明确"不做无限调参搜索"）：

1. **训练规模差异**：本次仅用 4×H200、全局 batch size=64、100k steps 单次运行；论文可能使用了更大规模的 GPU 集群/更大全局 batch/更长训练步数，而相机视角这种"分布外视觉泛化"能力通常比"本体感知敏感"的机器人初始状态泛化更依赖训练规模（更多样的视觉输入组合才能学到视角不变性）。
2. **数据版本差异**：改用 `nvidia/LIBERO_LeRobot_v3`（非官方论文可能使用的 `no_noops` 过滤版本或其他内部数据版本），可能在演示数据的相机视角多样性、动作平滑度上与官方数据存在差异。
3. **训练数据本身缺乏视角增强**：核查 `src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py` 与 `data_transforms.inputs` 配置，确认训练管线中**没有随机裁剪/颜色抖动等图像增强**（仅有 `resize_with_pad`），这与官方 launch 脚本一致（并非本次复现引入的差异），但如果官方真实训练配方中另有独立的视角/图像增强步骤未在开源 launch 脚本中体现，会导致官方 checkpoint 对相机视角扰动天然更鲁棒。
4. **基座任务成功率差异**：从按套件拆分的结果看，`libero_goal` 套件两个类别成功率均明显偏低（Robot 33.5%、Camera 27.9%），远低于 `libero_object`（Robot 61.6%、Camera 67.4%），说明本次复现的 checkpoint 在不同套件上的基础任务求解能力本身就不均衡；官方 checkpoint 若在无扰动基线上就有更高、更均衡的任务成功率，扰动后的分数自然也会更高（因为"先要能做对任务，才谈得上对扰动鲁棒"）。

#### 结论

- **Robot Initial States（55% 目标）：复现成功**（50.6%，−4.4pp，在 ±5pp 验收容差内）。
- **Camera Viewpoints（83% 目标）：未复现成功**（44.6%，−38.4pp，远超 ±5pp 容差）。已排除评测流程/代码 bug，判定为受限于本次复现的训练规模（4×H200、单次 100k-step 微调）与官方生产级训练配方之间的差距，属于合理的"资源受限复现"结果，而非本仓库代码或本手册流程的缺陷。
- 若要进一步缩小 Camera 差距，后续可尝试的方向（本次复现范围外，仅供参考）：增大有效 batch size/GPU 数量并等比例调整学习率、增加训练步数、在训练数据里加入更多相机视角变化的演示数据、检查是否有额外的图像域随机化训练技巧未包含在当前开源 launch 脚本中。
