# itvlaGp RoboTwin `hanging_mug` 评估执行日志

> 由 `b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh` 自动记录。
> 手册：[`itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg_eval.md`](itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg_eval.md)

---

## 评估配置

| 项 | 值 |
|----|-----|
| **开始时间** | 2026-08-26 11:09:54 |
| **代码库** | `/home/luogang/SRC/Robot/itvlaGp` |
| **Conda 环境** | `itvlaGp`（`/home/luogang/miniforge3/envs/itvlaGp`） |
| **任务** | `hanging_mug` (task_idx=10) |
| **Checkpoint** | `/home/luogang/SRC/Robot/itvlaGp/outputs-gcs/hanging_mug_p2_010k/checkpoints/010000/pretrained_model` |
| **GCS** | `gs://physical-ai-data-eu/VENV/tmp/2026_08_26_01_24_16-internvla_a1_5-geop-phase2-finetune-kptsim-voxel-8g-10k/checkpoints/010000/pretrained_model` |
| **kpt meta** | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/hanging_mug_kptsim_lrbv30/meta/keypoints_meta.json` |
| **kpt 坐标模式** | `voxel` |
| **推理后端** | `standard` |
| **动作模式** | `abs` |
| **dtype** | `bfloat16` |
| **infer-horizon** | 20 |
| **每配置 episode** | 100 |
| **配置** | `demo_clean,demo_randomized` |
| **GPU** | `0,1` |
| **输出目录** | `/home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_hngMg_p2_010k` |
| **RUN_ID** | `itvlaGp_hngMg_p2_010k` |
| **控制台日志** | `/home/luogang/SRC/Robot/itvlaGp/outputs/logs/run_itvlaGp_hngMg_p2_010k.log` |

---

## 时间线 / 操作日志

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-08-26 11:09:54 | 解析配置 hanging_mug idx=10 run=itvlaGp_hngMg_p2_010k | OK |
| 2026-08-26 11:09:54 | 控制台完整日志 | /home/luogang/SRC/Robot/itvlaGp/outputs/logs/run_itvlaGp_hngMg_p2_010k.log |
| 2026-08-26 11:09:54 | conda activate itvlaGp | OK /home/luogang/miniforge3/envs/itvlaGp/bin/python |
| 2026-08-26 11:09:54 | GCS 开始下载 gs://physical-ai-data-eu/VENV/tmp/2026_08_26_01_24_16-internvla_a1_5-geop-phase2-finetune-kptsim-voxel-8g-10k/checkpoints/010000/pretrained_model | ... |
| 2026-08-26 11:10:54 | GCS 下载完成 | OK model.safetensors=6321129804 bytes |
| 2026-08-26 11:11:08 | 预检 15 项 | OK |
| 2026-08-26 11:16:33 | 冒烟 2 ep demo_clean | OK exit=0 0S/2F/2mp4 log=/home/luogang/SRC/Robot/itvlaGp/outputs/logs/smoke_itvlaGp_hngMg_p2_010k.log |
| 2026-08-26 11:16:33 | 启动 demo_clean GPU0 100 ep | ... |
| 2026-08-26 11:16:33 | 启动 demo_randomized GPU1 100 ep | ... |
| 2026-08-26 14:21:38 | demo_clean 100 ep | OK 9/100 = 9.0% |
| 2026-08-26 15:10:52 | demo_randomized 100 ep | OK 4/100 = 4.0% |

## 最终结果 (2026-08-26 15:10:52)

| 配置 | 成功 | 失败 | 总计 | Success Rate |
|------|------|------|------|--------------|
| **demo_clean** | 9 | 91 | 100 | **9.0%** |
| **demo_randomized** | 4 | 96 | 100 | **4.0%** |

**输出路径**:

- `/home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_hngMg_p2_010k/robotwin/demo_clean/hanging_mug/`
- `/home/luogang/SRC/Robot/itvlaGp/outputs/logs/eval_itvlaGp_hngMg_p2_010k_demo_clean.log`
- `/home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_hngMg_p2_010k/robotwin/demo_randomized/hanging_mug/`
- `/home/luogang/SRC/Robot/itvlaGp/outputs/logs/eval_itvlaGp_hngMg_p2_010k_demo_randomized.log`

| 2026-08-26 15:10:52 | 汇总写入 /home/luogang/SRC/Robot/itvlaGp/b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg_evalLOG.md | OK |
