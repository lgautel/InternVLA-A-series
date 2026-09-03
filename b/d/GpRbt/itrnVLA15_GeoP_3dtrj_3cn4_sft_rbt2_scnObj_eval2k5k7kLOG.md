# itvlaGp RoboTwin `scan_object` 多步 checkpoint 评估执行日志（@2500 / @5000 / @7500）

> **目的**：在同一 GCS job 下，对比 scan_object Phase 2 训练 **第 2500 / 5000 / 7500 步** checkpoint 的 RoboTwin 2.0 成功率，与已完成的 @010000 结果（clean **36.0%@149.25** / randomized **37.0%@149.25**，见 [`itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_evalLOG.md`](itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_evalLOG.md)）对照。
>
> **操作手册**：[`itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_eval.md`](itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_eval.md)（§6.4.4 其它 checkpoint 步数）
>
> **评测脚本**：[`b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh`](../s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh)

---

## 批次计划

| 顺序 | ckpt-step | STEP_TAG | RUN_ID | 本机 ckpt 目录 | 评测输出 |
|:---:|:---:|:---:|:---|:---|:---|
| 1 | `002500` | `002500` | `itvlaGp_scnObj_p2_002500` | `outputs-gcs/scan_object_p2_002500/checkpoints/002500/pretrained_model` | `outputs/robotwin/itvlaGp_scnObj_p2_002500` |
| 2 | `005000` | `005k` | `itvlaGp_scnObj_p2_005k` | `outputs-gcs/scan_object_p2_005k/checkpoints/005000/pretrained_model` | `outputs/robotwin/itvlaGp_scnObj_p2_005k` |
| 3 | `007500` | `007500` | `itvlaGp_scnObj_p2_007500` | `outputs-gcs/scan_object_p2_007500/checkpoints/007500/pretrained_model` | `outputs/robotwin/itvlaGp_scnObj_p2_007500` |

**共用配置**（与 @010000 一致）：

| 项 | 值 |
|----|-----|
| **GCS job** | `gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30` |
| **任务** | `scan_object` (task_idx=41) |
| **kpt meta** | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/scan_object_kptsim_lrbv30/meta/keypoints_meta.json` |
| **kpt 坐标** | `voxel`，offset `[-0.6748, -1.0345, 0.6219]` |
| **推理** | `standard` / `abs` / `bfloat16` / infer-horizon=20 |
| **episode** | 100 × `demo_clean` + 100 × `demo_randomized`（双 GPU 并行） |
| **Python** | conda `itvlaGp`（Python 3.10.20） |
| **代码库** | `/home/luogang/SRC/Robot/itvlaGp` |

**编排方式**：三步串行（每步 gcs → preflight → smoke → eval → summarize），避免双 GPU 被多 job 争抢；总预估耗时约 **7–8 h**（参考 @010000 单次 ~2.6 h）。

---

## 时间线 / 操作日志

| 时间 | 操作 | 结果 |
|------|------|------|

## 手册与脚本

| 项 | 路径 |
|----|------|
| 操作手册 | `/home/luogang/SRC/Robot/itvlaGp/b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_eval.md` |
| 评测脚本 | `/home/luogang/SRC/Robot/itvlaGp/b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh` |
| 批次编排脚本 | `/home/luogang/SRC/Robot/itvlaGp/b/s/run_scnObj_eval_2k5k7k.sh` |
| @010000 对照 LOG | `/home/luogang/SRC/Robot/itvlaGp/b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_evalLOG.md` |

## 问题记录（报错 → 根因 → 修复 → 验证）

（运行中遇错自动追加；无则留空）

## 文件增删改记录

| 时间 | 文件 | 操作 | 缘由 |
|------|------|------|------|
| 2026-08-27 | `b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_eval2k5k7kLOG.md` | 新建 | 三步 checkpoint 评测执行日志 |
| 2026-08-27 | `b/s/run_scnObj_eval_2k5k7k.sh` | 新建 | 串行调用 eval.sh 评测 002500/005000/007500 |

## 操作命令记录

### 命令 2026-08-27 — 启动三步批次评测

**理由**：用户要求用 GCS 上 2500/5000/7500 步 checkpoint 评测 scan_object，全过程记入本 LOG；三步串行避免 GPU 争抢，每步走 gcs→preflight→smoke→eval→summarize 全流程。

```bash
cd /home/luogang/SRC/Robot/itvlaGp
chmod +x b/s/run_scnObj_eval_2k5k7k.sh
bash b/s/run_scnObj_eval_2k5k7k.sh
```

| 2026-08-27 | 批次编排脚本启动 | 见 outputs/logs/batch_scnObj_2k5k7k.log |
| 2026-08-27 08:35:30 | 批次编排脚本启动 | 见 /home/luogang/SRC/Robot/itvlaGp/outputs/logs/batch_scnObj_2k5k7k.log |
| 2026-08-27 08:35:30 | 开始 ckpt-step=002500 全流程 | 进行中 |

---

## 再次运行 2026-08-27 08:35:30

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-08-27 08:35:30 | 解析配置 scan_object idx=41 run=itvlaGp_scnObj_p2_002500 | OK |
| 2026-08-27 08:35:30 | 控制台完整日志 | /home/luogang/SRC/Robot/itvlaGp/outputs/logs/run_itvlaGp_scnObj_p2_002500.log |
| 2026-08-27 08:35:30 | `/home/luogang/SRC/Robot/itvlaGp/b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh` | 修改（评测前） | scan_object 默认 GCS/offset；LOG 增手册/路径/问题/文件/命令节 |
| 2026-08-27 08:35:30 | conda activate itvlaGp | OK /home/luogang/miniforge3/envs/itvlaGp/bin/python |

### 命令 2026-08-27 08:35:30

**理由**：从 GCS 拉取 step-002500 的 pretrained_model 四文件到本机，供预检与 inference 加载

```bash
gcloud storage cp gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/002500/pretrained_model/config.json gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/002500/pretrained_model/stats.json gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/002500/pretrained_model/train_config.json gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/002500/pretrained_model/model.safetensors /home/luogang/SRC/Robot/itvlaGp/outputs-gcs/scan_object_p2_002500/checkpoints/002500/pretrained_model/
```

| 2026-08-27 08:35:30 | GCS 开始下载 gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/002500/pretrained_model | ... |
| 2026-08-27 08:36:29 | GCS 下载完成 | OK model.safetensors=6321129804 bytes |
| 2026-08-27 08:36:43 | 预检 15 项 | OK |

**预检说明**：项 [4][6] 打印的 `Python 3.10 is below the recommended 3.11` 来自 transformers/fla 导入时的提示，**非错误**；本机 RoboTwin 评测沿用 conda `itvlaGp`（Python 3.10.20），与 hanging_mug / stack_bowls 评测一致。
| 2026-08-27 08:42:07 | 冒烟 2 ep demo_clean | OK exit=0 0S/2F/2mp4 log=/home/luogang/SRC/Robot/itvlaGp/outputs/logs/smoke_itvlaGp_scnObj_p2_002500.log |
| 2026-08-27 08:42:07 | 启动 demo_clean GPU0 100 ep | ... |
| 2026-08-27 08:42:07 | 启动 demo_randomized GPU1 100 ep | ... |
| 2026-08-27 11:14:13 | demo_clean 100 ep | OK 44/100 = 44.0%@37.31 epoch |
| 2026-08-27 11:22:36 | demo_randomized 100 ep | OK 33/100 = 33.0%@37.31 epoch |

## 最终结果 (2026-08-27 11:22:36)

| 配置 | 成功 | 失败 | 总计 | Success Rate |
|------|------|------|------|--------------|
| **demo_clean** | 44 | 56 | 100 | **44.0%@37.31** |
| **demo_randomized** | 33 | 67 | 100 | **33.0%@37.31** |

**输出路径**:

- `/home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_scnObj_p2_002500/robotwin/demo_clean/scan_object/`
- `/home/luogang/SRC/Robot/itvlaGp/outputs/logs/eval_itvlaGp_scnObj_p2_002500_demo_clean.log`
- `/home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_scnObj_p2_002500/robotwin/demo_randomized/scan_object/`
- `/home/luogang/SRC/Robot/itvlaGp/outputs/logs/eval_itvlaGp_scnObj_p2_002500_demo_randomized.log`

| 2026-08-27 11:22:36 | 汇总写入 /home/luogang/SRC/Robot/itvlaGp/b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_eval2k5k7kLOG.md | OK |
| 2026-08-27 11:22:36 | ckpt-step=002500 全流程 | OK |
| 2026-08-27 11:22:36 | 开始 ckpt-step=005000 全流程 | 进行中 |

---

## 再次运行 2026-08-27 11:22:36

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-08-27 11:22:36 | 解析配置 scan_object idx=41 run=itvlaGp_scnObj_p2_005k | OK |
| 2026-08-27 11:22:36 | 控制台完整日志 | /home/luogang/SRC/Robot/itvlaGp/outputs/logs/run_itvlaGp_scnObj_p2_005k.log |
| 2026-08-27 11:22:36 | `/home/luogang/SRC/Robot/itvlaGp/b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh` | 修改（评测前） | scan_object 默认 GCS/offset；LOG 增手册/路径/问题/文件/命令节 |
| 2026-08-27 11:22:36 | conda activate itvlaGp | OK /home/luogang/miniforge3/envs/itvlaGp/bin/python |

### 命令 2026-08-27 11:22:36

**理由**：从 GCS 拉取 step-005000 的 pretrained_model 四文件到本机，供预检与 inference 加载

```bash
gcloud storage cp gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/005000/pretrained_model/config.json gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/005000/pretrained_model/stats.json gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/005000/pretrained_model/train_config.json gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/005000/pretrained_model/model.safetensors /home/luogang/SRC/Robot/itvlaGp/outputs-gcs/scan_object_p2_005k/checkpoints/005000/pretrained_model/
```

| 2026-08-27 11:22:36 | GCS 开始下载 gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/005000/pretrained_model | ... |
| 2026-08-27 11:23:36 | GCS 下载完成 | OK model.safetensors=6321129804 bytes |
| 2026-08-27 11:23:51 | 预检 15 项 | OK |

**预检说明**：项 [4][6] 打印的 `Python 3.10 is below the recommended 3.11` 来自 transformers/fla 导入时的提示，**非错误**；本机 RoboTwin 评测沿用 conda `itvlaGp`（Python 3.10.20），与 hanging_mug / stack_bowls 评测一致。
| 2026-08-27 11:27:53 | 冒烟 2 ep demo_clean | OK exit=0 1S/1F/2mp4 log=/home/luogang/SRC/Robot/itvlaGp/outputs/logs/smoke_itvlaGp_scnObj_p2_005k.log |
| 2026-08-27 11:27:53 | 启动 demo_clean GPU0 100 ep | ... |
| 2026-08-27 11:27:53 | 启动 demo_randomized GPU1 100 ep | ... |
| 2026-08-27 13:57:13 | demo_clean 100 ep | OK 45/100 = 45.0%@74.63 epoch |
| 2026-08-27 14:07:11 | demo_randomized 100 ep | OK 33/100 = 33.0%@74.63 epoch |

## 最终结果 (2026-08-27 14:07:11)

| 配置 | 成功 | 失败 | 总计 | Success Rate |
|------|------|------|------|--------------|
| **demo_clean** | 45 | 55 | 100 | **45.0%@74.63** |
| **demo_randomized** | 33 | 67 | 100 | **33.0%@74.63** |

**输出路径**:

- `/home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_scnObj_p2_005k/robotwin/demo_clean/scan_object/`
- `/home/luogang/SRC/Robot/itvlaGp/outputs/logs/eval_itvlaGp_scnObj_p2_005k_demo_clean.log`
- `/home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_scnObj_p2_005k/robotwin/demo_randomized/scan_object/`
- `/home/luogang/SRC/Robot/itvlaGp/outputs/logs/eval_itvlaGp_scnObj_p2_005k_demo_randomized.log`

| 2026-08-27 14:07:11 | 汇总写入 /home/luogang/SRC/Robot/itvlaGp/b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_eval2k5k7kLOG.md | OK |
| 2026-08-27 14:07:11 | ckpt-step=005000 全流程 | OK |
| 2026-08-27 14:07:11 | 开始 ckpt-step=007500 全流程 | 进行中 |

---

## 再次运行 2026-08-27 14:07:11

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-08-27 14:07:11 | 解析配置 scan_object idx=41 run=itvlaGp_scnObj_p2_007500 | OK |
| 2026-08-27 14:07:11 | 控制台完整日志 | /home/luogang/SRC/Robot/itvlaGp/outputs/logs/run_itvlaGp_scnObj_p2_007500.log |
| 2026-08-27 14:07:11 | `/home/luogang/SRC/Robot/itvlaGp/b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh` | 修改（评测前） | scan_object 默认 GCS/offset；LOG 增手册/路径/问题/文件/命令节 |
| 2026-08-27 14:07:11 | conda activate itvlaGp | OK /home/luogang/miniforge3/envs/itvlaGp/bin/python |

### 命令 2026-08-27 14:07:11

**理由**：从 GCS 拉取 step-007500 的 pretrained_model 四文件到本机，供预检与 inference 加载

```bash
gcloud storage cp gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/007500/pretrained_model/config.json gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/007500/pretrained_model/stats.json gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/007500/pretrained_model/train_config.json gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/007500/pretrained_model/model.safetensors /home/luogang/SRC/Robot/itvlaGp/outputs-gcs/scan_object_p2_007500/checkpoints/007500/pretrained_model/
```

| 2026-08-27 14:07:11 | GCS 开始下载 gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/007500/pretrained_model | ... |
| 2026-08-27 14:08:21 | GCS 下载完成 | OK model.safetensors=6321129804 bytes |
| 2026-08-27 14:08:35 | 预检 15 项 | OK |

**预检说明**：项 [4][6] 打印的 `Python 3.10 is below the recommended 3.11` 来自 transformers/fla 导入时的提示，**非错误**；本机 RoboTwin 评测沿用 conda `itvlaGp`（Python 3.10.20），与 hanging_mug / stack_bowls 评测一致。
| 2026-08-27 14:13:11 | 冒烟 2 ep demo_clean | OK exit=0 0S/2F/2mp4 log=/home/luogang/SRC/Robot/itvlaGp/outputs/logs/smoke_itvlaGp_scnObj_p2_007500.log |
| 2026-08-27 14:13:11 | 启动 demo_clean GPU0 100 ep | ... |
| 2026-08-27 14:13:11 | 启动 demo_randomized GPU1 100 ep | ... |
| 2026-08-27 16:42:47 | demo_clean 100 ep | OK 34/100 = 34.0%@111.94 epoch |
| 2026-08-27 16:51:32 | demo_randomized 100 ep | OK 31/100 = 31.0%@111.94 epoch |

## 最终结果 (2026-08-27 16:51:32)

| 配置 | 成功 | 失败 | 总计 | Success Rate |
|------|------|------|------|--------------|
| **demo_clean** | 34 | 66 | 100 | **34.0%@111.94** |
| **demo_randomized** | 31 | 69 | 100 | **31.0%@111.94** |

**输出路径**:

- `/home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_scnObj_p2_007500/robotwin/demo_clean/scan_object/`
- `/home/luogang/SRC/Robot/itvlaGp/outputs/logs/eval_itvlaGp_scnObj_p2_007500_demo_clean.log`
- `/home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_scnObj_p2_007500/robotwin/demo_randomized/scan_object/`
- `/home/luogang/SRC/Robot/itvlaGp/outputs/logs/eval_itvlaGp_scnObj_p2_007500_demo_randomized.log`

| 2026-08-27 16:51:32 | 汇总写入 /home/luogang/SRC/Robot/itvlaGp/b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_eval2k5k7kLOG.md | OK |
| 2026-08-27 16:51:32 | ckpt-step=007500 全流程 | OK |
| 2026-08-27 16:51:32 | 批次全部完成 | OK |

---

## 批次汇总对照（含 @010000 参考）

总耗时约 **8.3 h**（08:35–16:51 UTC，exit=0）；三步均无报错，预检/冒烟/正式评测全部通过。

| ckpt-step | demo_clean | demo_randomized | 备注 |
|:---:|:---:|:---:|:---|
| **002500** | **44.0%@37.31** (44/100) | **33.0%@37.31** (33/100) | clean 最高 |
| **005000** | **45.0%@74.63** (45/100) | **33.0%@74.63** (33/100) | clean 峰值 |
| **007500** | **34.0%@111.94** (34/100) | **31.0%@111.94** (31/100) | 开始回落 |
| **010000**（对照，见 evalLOG） | **36.0%@149.25** (36/100) | **37.0%@149.25** (37/100) | randomized 最高 |

**简要结论**：
- `demo_clean`：@5000 最佳（45%），@7500/@10000 降至 34–36%，与训练手册「@2500 后 open-loop MSE 可能变差」部分吻合，但仿真成功率在 @5000 仍优于 @10000。
- `demo_randomized`：四步均在 **31–37%** 窄幅波动，@10000 的 37% 为最高，整体对步数不敏感。
- 三步评测均无 error；问题记录节留空。

**epoch 计算**：`total_frames=8463`，有效 `batch_size=128`，`steps_per_epoch=ceil(8463/128)=67`；各 checkpoint 的 epoch 为 @2500=`37.31`、@5000=`74.63`、@7500=`111.94`、@10000=`149.25`。
