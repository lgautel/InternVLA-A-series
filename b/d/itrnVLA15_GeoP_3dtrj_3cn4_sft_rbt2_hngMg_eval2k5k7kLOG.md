# itvlaGp RoboTwin `hanging_mug` 多步 checkpoint 评估执行日志（@2500 / @5000 / @7500）

> **目的**：在同一 GCS job 下，对比 hanging_mug Phase 2 训练 **第 2500 / 5000 / 7500 步** checkpoint 的 RoboTwin 2.0 成功率，与已完成的 @010000 结果（clean **9.0%** / randomized **4.0%**，见 [`itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg_evalLOG.md`](itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg_evalLOG.md)）对照。
>
> **操作手册**：[`itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg_eval.md`](itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg_eval.md)
>
> **评测脚本**：[`b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh`](../s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh)

---

## 批次计划

| 顺序 | ckpt-step | STEP_TAG | RUN_ID | 本机 ckpt 目录 | 评测输出 |
|:---:|:---:|:---:|:---|:---|:---|
| 1 | `002500` | `002500` | `itvlaGp_hngMg_p2_002500` | `outputs-gcs/hanging_mug_p2_002500/checkpoints/002500/pretrained_model` | `outputs/robotwin/itvlaGp_hngMg_p2_002500` |
| 2 | `005000` | `005k` | `itvlaGp_hngMg_p2_005k` | `outputs-gcs/hanging_mug_p2_005k/checkpoints/005000/pretrained_model` | `outputs/robotwin/itvlaGp_hngMg_p2_005k` |
| 3 | `007500` | `007500` | `itvlaGp_hngMg_p2_007500` | `outputs-gcs/hanging_mug_p2_007500/checkpoints/007500/pretrained_model` | `outputs/robotwin/itvlaGp_hngMg_p2_007500` |

**共用配置**（与 @010000 一致）：

| 项 | 值 |
|----|-----|
| **GCS job** | `gs://physical-ai-data-eu/VENV/tmp/2026_08_26_01_24_16-internvla_a1_5-geop-phase2-finetune-kptsim-voxel-8g-10k` |
| **任务** | `hanging_mug` (task_idx=10) |
| **kpt meta** | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/hanging_mug_kptsim_lrbv30/meta/keypoints_meta.json` |
| **kpt 坐标** | `voxel`，offset `[-0.7718, -1.0504, 0.4779]` |
| **推理** | `standard` / `abs` / `bfloat16` / infer-horizon=20 |
| **episode** | 100 × `demo_clean` + 100 × `demo_randomized`（双 GPU 并行） |
| **步数上限** | 900（`task_config/_eval_step_limit.yml`） |
| **Python** | conda `itvlaGp`（Python 3.10.20） |
| **代码库** | `/home/luogang/SRC/Robot/itvlaGp` |

**编排方式**：三步串行（每步 gcs → preflight → smoke → eval → summarize），避免双 GPU 被多 job 争抢；总预估耗时约 **12 h**（参考 @010000 单次 ~4 h）。

---

## 时间线 / 操作日志

| 时间 | 操作 | 结果 |
|------|------|------|

## 手册与脚本

| 项 | 路径 |
|----|------|
| 操作手册 | `/home/luogang/SRC/Robot/itvlaGp/b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg_eval.md` |
| 评测脚本 | `/home/luogang/SRC/Robot/itvlaGp/b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh` |
| 批次编排脚本 | `/home/luogang/SRC/Robot/itvlaGp/b/s/run_hngMg_eval_2k5k7k.sh` |
| @010000 对照 LOG | `/home/luogang/SRC/Robot/itvlaGp/b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg_evalLOG.md` |

## 问题记录（报错 → 根因 → 修复 → 验证）

（运行中遇错自动追加；无则留空）

## 文件增删改记录

| 时间 | 文件 | 操作 | 缘由 |
|------|------|------|------|
| 2026-08-28 | `b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg_eval2k5k7kLOG.md` | 新建 | 三步 checkpoint 评测执行日志 |
| 2026-08-28 | `b/s/run_hngMg_eval_2k5k7k.sh` | 新建 | 串行调用 eval.sh 评测 002500/005000/007500 |

## 操作命令记录

### 命令 2026-08-28 — 启动三步批次评测

**理由**：用户要求用 GCS 上 2500/5000/7500 步 checkpoint 评测 hanging_mug，全过程记入本 LOG；三步串行避免 GPU 争抢，每步走 gcs→preflight→smoke→eval→summarize 全流程。

```bash
cd /home/luogang/SRC/Robot/itvlaGp
chmod +x b/s/run_hngMg_eval_2k5k7k.sh
bash b/s/run_hngMg_eval_2k5k7k.sh
```

| 2026-08-28 | 批次编排脚本启动 | 见 outputs/logs/batch_hngMg_2k5k7k.log |
| 2026-08-28 01:30:05 | 批次编排脚本启动 | 见 /home/luogang/SRC/Robot/itvlaGp/outputs/logs/batch_hngMg_2k5k7k.log |
| 2026-08-28 01:30:05 | 开始 ckpt-step=002500 全流程 | 进行中 |

---

## 再次运行 2026-08-28 01:30:05

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-08-28 01:30:05 | 解析配置 hanging_mug idx=10 run=itvlaGp_hngMg_p2_002500 | OK |
| 2026-08-28 01:30:05 | 控制台完整日志 | /home/luogang/SRC/Robot/itvlaGp/outputs/logs/run_itvlaGp_hngMg_p2_002500.log |
| 2026-08-28 01:30:05 | `/home/luogang/SRC/Robot/itvlaGp/b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh` | 修改（评测前） | scan_object 默认 GCS/offset；LOG 增手册/路径/问题/文件/命令节 |
| 2026-08-28 01:30:05 | conda activate itvlaGp | OK /home/luogang/miniforge3/envs/itvlaGp/bin/python |

### 命令 2026-08-28 01:30:05

**理由**：从 GCS 拉取 step-002500 的 pretrained_model 四文件到本机，供预检与 inference 加载

```bash
gcloud storage cp gs://physical-ai-data-eu/VENV/tmp/2026_08_26_01_24_16-internvla_a1_5-geop-phase2-finetune-kptsim-voxel-8g-10k/checkpoints/002500/pretrained_model/config.json gs://physical-ai-data-eu/VENV/tmp/2026_08_26_01_24_16-internvla_a1_5-geop-phase2-finetune-kptsim-voxel-8g-10k/checkpoints/002500/pretrained_model/stats.json gs://physical-ai-data-eu/VENV/tmp/2026_08_26_01_24_16-internvla_a1_5-geop-phase2-finetune-kptsim-voxel-8g-10k/checkpoints/002500/pretrained_model/train_config.json gs://physical-ai-data-eu/VENV/tmp/2026_08_26_01_24_16-internvla_a1_5-geop-phase2-finetune-kptsim-voxel-8g-10k/checkpoints/002500/pretrained_model/model.safetensors /home/luogang/SRC/Robot/itvlaGp/outputs-gcs/hanging_mug_p2_002500/checkpoints/002500/pretrained_model/
```

| 2026-08-28 01:30:05 | GCS 开始下载 gs://physical-ai-data-eu/VENV/tmp/2026_08_26_01_24_16-internvla_a1_5-geop-phase2-finetune-kptsim-voxel-8g-10k/checkpoints/002500/pretrained_model | ... |
| 2026-08-28 01:31:06 | GCS 下载完成 | OK model.safetensors=6321129804 bytes |
| 2026-08-28 01:31:21 | 预检 15 项 | OK |

**预检说明**：项 [4][6] 打印的 `Python 3.10 is below the recommended 3.11` 来自 transformers/fla 导入时的提示，**非错误**；本机 RoboTwin 评测沿用 conda `itvlaGp`（Python 3.10.20），与 hanging_mug / stack_bowls 评测一致。
| 2026-08-28 01:36:56 | 冒烟 2 ep demo_clean | OK exit=0 0S/2F/2mp4 log=/home/luogang/SRC/Robot/itvlaGp/outputs/logs/smoke_itvlaGp_hngMg_p2_002500.log |
| 2026-08-28 01:36:56 | 启动 demo_clean GPU0 100 ep | ... |
| 2026-08-28 01:36:56 | 启动 demo_randomized GPU1 100 ep | ... |
