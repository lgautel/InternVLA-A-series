# itvlaGp RoboTwin `scan_object` 评估执行日志

> 由 `b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh` 自动记录。
> 手册：[`itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_eval.md`](itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_eval.md)

---

## 评估配置

| 项 | 值 |
|----|-----|
| **开始时间** | 2026-08-27 02:53:54 |
| **代码库** | `/home/luogang/SRC/Robot/itvlaGp` |
| **Conda 环境** | `itvlaGp`（`/home/luogang/miniforge3/envs/itvlaGp`） |
| **任务** | `scan_object` (task_idx=41) |
| **Checkpoint** | `/home/luogang/SRC/Robot/itvlaGp/outputs-gcs/scan_object_p2_010k/checkpoints/010000/pretrained_model` |
| **GCS** | `gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/010000/pretrained_model` |
| **kpt meta** | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/scan_object_kptsim_lrbv30/meta/keypoints_meta.json` |
| **kpt 坐标模式** | `voxel` |
| **推理后端** | `standard` |
| **动作模式** | `abs` |
| **dtype** | `bfloat16` |
| **infer-horizon** | 20 |
| **每配置 episode** | 100 |
| **配置** | `demo_clean,demo_randomized` |
| **GPU** | `0,1` |
| **输出目录** | `/home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_scnObj_p2_010k` |
| **RUN_ID** | `itvlaGp_scnObj_p2_010k` |
| **控制台日志** | `/home/luogang/SRC/Robot/itvlaGp/outputs/logs/run_itvlaGp_scnObj_p2_010k.log` |

---

## 时间线 / 操作日志

| 时间 | 操作 | 结果 |
|------|------|------|

## 手册与脚本

| 项 | 路径 |
|----|------|
| 操作手册 | `/home/luogang/SRC/Robot/itvlaGp/b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_eval.md` |
| 评测脚本 | `/home/luogang/SRC/Robot/itvlaGp/b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh` |
| inference 入口 | `/home/luogang/SRC/Robot/itvlaGp/evaluation/RoboTwin/inference.py` |
| RoboTwin 任务源码 | `/home/luogang/SRC/Robot/itvlaGp/third_party/RoboTwin/envs/scan_object.py` |

## 关键路径速查

| 用途 | 路径 |
|------|------|
| GCS job | `gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30` |
| GCS ckpt | `gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/010000/pretrained_model` |
| 本机 ckpt | `/home/luogang/SRC/Robot/itvlaGp/outputs-gcs/scan_object_p2_010k/checkpoints/010000/pretrained_model` |
| kpt meta | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/scan_object_kptsim_lrbv30/meta/keypoints_meta.json` |
| expect offset | `-0.6748,-1.0345,0.6219` |
| 评测输出 | `/home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_scnObj_p2_010k` |
| 冒烟视频 | `/home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_scnObj_p2_010k/smoke/demo_clean/scan_object` |
| clean 视频 | `/home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_scnObj_p2_010k/robotwin/demo_clean/scan_object` |
| randomized 视频 | `/home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_scnObj_p2_010k/robotwin/demo_randomized/scan_object` |
| 冒烟 inference 日志 | `/home/luogang/SRC/Robot/itvlaGp/outputs/logs/smoke_itvlaGp_scnObj_p2_010k.log` |
| clean inference 日志 | `/home/luogang/SRC/Robot/itvlaGp/outputs/logs/eval_itvlaGp_scnObj_p2_010k_demo_clean.log` |
| randomized inference 日志 | `/home/luogang/SRC/Robot/itvlaGp/outputs/logs/eval_itvlaGp_scnObj_p2_010k_demo_randomized.log` |

## 问题记录（报错 → 根因 → 修复 → 验证）

（运行中遇错自动追加；无则留空）

## 文件增删改记录

| 时间 | 文件 | 操作 | 缘由 |
|------|------|------|------|

## 操作命令记录

（各阶段关键命令见下文「命令」小节与时间线）

| 2026-08-27 02:53:54 | 解析配置 scan_object idx=41 run=itvlaGp_scnObj_p2_010k | OK |
| 2026-08-27 02:53:54 | 控制台完整日志 | /home/luogang/SRC/Robot/itvlaGp/outputs/logs/run_itvlaGp_scnObj_p2_010k.log |

### 命令 2026-08-27 02:53:54

**理由**：用户/Agent 启动本次评测的完整 shell 命令

```bash
bash b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh --reset-log --task-name scan_object --task-idx 41 --gcs-job gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30 --ckpt-step 010000 --expect-repo-id scan_object_kptsim_lrbv30 --expect-offset -0.6748,-1.0345,0.6219 --kpt-meta /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/scan_object_kptsim_lrbv30/meta/keypoints_meta.json --out /home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_scnObj_p2_010k --eval-log /home/luogang/SRC/Robot/itvlaGp/b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_evalLOG.md
```

| 2026-08-27 02:53:54 | `/home/luogang/SRC/Robot/itvlaGp/b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh` | 修改（评测前） | scan_object 默认 GCS/offset；LOG 增手册/路径/问题/文件/命令节；修正时间线表格顺序 |
| 2026-08-27 02:53:54 | conda activate itvlaGp | OK /home/luogang/miniforge3/envs/itvlaGp/bin/python |

### 命令 2026-08-27 02:53:54

**理由**：从 GCS 拉取 step-010000 的 pretrained_model 四文件到本机，供预检与 inference 加载

```bash
gcloud storage cp gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/010000/pretrained_model/config.json gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/010000/pretrained_model/stats.json gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/010000/pretrained_model/train_config.json gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/010000/pretrained_model/model.safetensors /home/luogang/SRC/Robot/itvlaGp/outputs-gcs/scan_object_p2_010k/checkpoints/010000/pretrained_model/
```

| 2026-08-27 02:53:54 | GCS 开始下载 gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30/checkpoints/010000/pretrained_model | ... |
| 2026-08-27 02:54:55 | GCS 下载完成 | OK model.safetensors=6321129804 bytes |
| 2026-08-27 02:55:09 | 预检 15 项 | OK |
| 2026-08-27 02:55:09 | 启动冒烟 2 ep demo_clean GPU0 | 进行中 |

**预检说明**：项 [4][6] 的 `Python 3.10 is below the recommended 3.11` 为 transformers/fla 提示，**非错误**；本机沿用 conda `itvlaGp`（Python 3.10.20）。
| 2026-08-27 03:00:32 | 冒烟 2 ep demo_clean | OK exit=0 0S/2F/2mp4 log=/home/luogang/SRC/Robot/itvlaGp/outputs/logs/smoke_itvlaGp_scnObj_p2_010k.log |

### Problem: eval.sh 语法错误导致正式评测未启动

| 项 | 内容 |
|----|------|
| **发现时机** | 2026-08-27 03:00:32，冒烟通过后进入 `stage_eval` |
| **症状** | `b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh: line 1180: unexpected EOF while looking for matching '"'`；进程 exit_code=2 |
| **根因** | 评测前修改 `init_eval_log` 时 heredoc 结构被破坏（`EOF` 提前闭合），bash 解析失败 |
| **修复** | 修正 `init_eval_log`：先写配置表 → `append_log_init_sections` → 再追加时间线表头；`bash -n eval.sh` 通过 |
| **验证** | 2026-08-27 续跑 `--from eval --skip-gcs --skip-smoke` |

| 2026-08-27 | `/home/luogang/SRC/Robot/itvlaGp/b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh` | 修改 | 修复 heredoc 语法；预检说明写入 LOG |

---

## 再次运行 2026-08-27 03:01:15

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-08-27 03:01:15 | 解析配置 scan_object idx=41 run=itvlaGp_scnObj_p2_010k | OK |
| 2026-08-27 03:01:15 | 控制台完整日志 | /home/luogang/SRC/Robot/itvlaGp/outputs/logs/run_itvlaGp_scnObj_p2_010k.log |

### 命令 2026-08-27 03:01:15

**理由**：用户/Agent 启动本次评测的完整 shell 命令

```bash
bash b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh --task-name scan_object --task-idx 41 --skip-gcs --skip-smoke --from eval --eval-log b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_evalLOG.md
```

| 2026-08-27 03:01:15 | `/home/luogang/SRC/Robot/itvlaGp/b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh` | 修改（评测前） | scan_object 默认 GCS/offset；LOG 增手册/路径/问题/文件/命令节 |
| 2026-08-27 03:01:15 | conda activate itvlaGp | OK /home/luogang/miniforge3/envs/itvlaGp/bin/python |
| 2026-08-27 03:01:15 | 启动 demo_clean GPU0 100 ep | ... |
| 2026-08-27 03:01:15 | 启动 demo_randomized GPU1 100 ep | ... |
| 2026-08-27 05:29:17 | demo_clean 100 ep | OK 36/100 = 36.0%@149.25 epoch |
| 2026-08-27 05:36:58 | demo_randomized 100 ep | OK 37/100 = 37.0%@149.25 epoch |

## 最终结果 (2026-08-27 05:36:58)

| 配置 | 成功 | 失败 | 总计 | Success Rate |
|------|------|------|------|--------------|
| **demo_clean** | 36 | 64 | 100 | **36.0%@149.25** |
| **demo_randomized** | 37 | 63 | 100 | **37.0%@149.25** |

**epoch 计算**：训练数据 `total_frames=8463`，有效 `batch_size=128`，按训练脚本规则 `steps_per_epoch=ceil(8463/128)=67`；因此 checkpoint @10000 的实际训练 epoch 为 `10000/67=149.25`。

**输出路径**:

- `/home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_scnObj_p2_010k/robotwin/demo_clean/scan_object/`
- `/home/luogang/SRC/Robot/itvlaGp/outputs/logs/eval_itvlaGp_scnObj_p2_010k_demo_clean.log`
- `/home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_scnObj_p2_010k/robotwin/demo_randomized/scan_object/`
- `/home/luogang/SRC/Robot/itvlaGp/outputs/logs/eval_itvlaGp_scnObj_p2_010k_demo_randomized.log`

| 2026-08-27 05:36:59 | 汇总写入 /home/luogang/SRC/Robot/itvlaGp/b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_evalLOG.md | OK |
