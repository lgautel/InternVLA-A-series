# dtaug2ps 训练日志 (2026-09-23)

> **实验**: itvlagpFrkPlug0907_dtaug2ps — p-schedule epoch 递减数据增强微调
> **方案文档**: [dtaug_p_sft2.md](dtaug_p_sft2.md)
> **启动脚本**: [frk_plug_sft_dtaug2ps_launch.sh](frk_plug_sft_dtaug2ps_launch.sh)
> **日期**: 2026-09-23

---

## A. 训练配置

### A.1 起始 Checkpoint

| 项目 | 值 |
|:---|:---|
| 路径 | `/home/a26113/b/Ckp/itvlagpFrkPlug0907_dtaug2ps/mrg_ckp/datw5k015da1w015pure1w07/` |
| 文件 | `model.safetensors` (5.9G), `config.json`, `stats.json` |
| 类型 | 合并 checkpoint（无 training_state/，从 step 0 开始） |
| 来源 | datw5k015da1w015pure1w07 合并权重 |

### A.2 数据集

| 项目 | 值 |
|:---|:---|
| 路径 | `/B/Dta/plug_into_socket_lrb_4D` |
| repo_id | `plug_into_socket_lrb_4D` |
| total_frames | 66577 |
| total_episodes | 100 |
| fps | 30 |
| stats | `/B/Dta/plug_into_socket_lrb_4D/meta/stats/abs/stats.json` |

### A.3 训练参数

| 参数 | 值 |
|:---|:---|
| EXPR_NAME | `itvlagpFrkPlug0907_dtaug2ps` |
| GPU | 8 × H100 (143 GiB each) |
| batch_size | 16 (per GPU) |
| effective_batch_size | 128 (16 × 8) |
| steps | 20000 |
| save_freq | 5000 |
| log_freq | 50 |
| optimizer_lr | 5e-5 |
| scheduler_decay_lr | 5e-6 |
| scheduler_warmup_steps | 500 |
| 1 epoch | ≈ 520 steps (66577 / 128) |

### A.4 数据增强 p-schedule

| 参数 | 值 |
|:---|:---|
| p_schedule | `[0.3, 0.4, 0.5, 0.6, 0.5, 0.4]` |
| p_epoch_interval | 5 |
| 6 种增强 | brightness/contrast/saturation/hue/sharpness/affine |
| affine weight | 0.2 |

p 切换时间表 (1 epoch ≈ 520 steps):

| Epoch 区间 | Step 区间 | schedule_idx | p 值 |
|:---|:---|:---|:---|
| 0 – 4 | 0 – 2.6K | 0 | **0.3** |
| 5 – 9 | 2.6K – 5.2K | 1 | **0.4** |
| 10 – 14 | 5.2K – 7.8K | 2 | **0.5** |
| 15 – 19 | 7.8K – 10.4K | 3 | **0.6** |
| 20 – 24 | 10.4K – 13K | 4 | **0.5** |
| 25+ | 13K+ | 5 (clamped) | **0.4** |

### A.5 路径

| 类型 | 路径 |
|:---|:---|
| Checkpoint | `~/b/Ckp/itvlagpFrkPlug0907_dtaug2ps/<JOB>/checkpoints/` |
| 训练日志 | `/B/Log/itvlagpFrkPlug0907_dtaug2ps/<JOB_STAMP>/train.log` |
| Wrapper 日志 | `/tmp/frk_plug_sft_dtaug2ps_wrapper.log` |

---

## B. 实施过程

### B.1 预飞检查 (Pre-flight)

**时间**: 2026-09-23 09:18 UTC

- [x] GPU: 8 × H100, all free (0 MiB)
- [x] Code changes: all 4 files verified (`p_schedule`, `begin_sample`, `_update_augment_p_on_epoch`, `_prev_p_epoch`)
- [x] Launch script: `frk_plug_sft_dtaug2ps_launch.sh` verified, p_schedule=[0.3,0.4,0.5,0.6,0.5,0.4], p_epoch_interval=5
- [x] Checkpoint: 5.9G model.safetensors + config.json + stats.json present
- [x] Dataset: `/B/Dta/plug_into_socket_lrb_4D` — 66577 frames, 100 episodes

### B.2 启动训练

**时间**: 2026-09-23 09:18:27 UTC

```bash
PRETRAINED_CKPT="/home/a26113/b/Ckp/itvlagpFrkPlug0907_dtaug2ps/mrg_ckp/datw5k015da1w015pure1w07" \
nohup bash b/d/Frk/expr/sftaug2/frk_plug_sft_dtaug2ps_launch.sh \
  > /tmp/frk_plug_sft_dtaug2ps_wrapper.log 2>&1 &
```

| 项目 | 值 |
|:---|:---|
| Wrapper PID | 2500784 |
| Train PID | 2500796 |
| JOB_STAMP | `2026_09_23_09_18_27` |
| OUTPUT_DIR | `~/b/Ckp/itvlagpFrkPlug0907_dtaug2ps/2026_09_23_09_18_27-internvla_a1_5-frk-plug-sft-dtaug2ps` |
| LOG_FILE | `/B/Log/itvlagpFrkPlug0907_dtaug2ps/2026_09_23_09_18_27/train.log` |

### B.3 训练进度

#### Step 50 (09:24:31)

首次日志输出：
```
step:50.0 | epoch:0.10 | loss:0.168 | loss_action:0.002 | loss_vqa:0.057 | loss_video:0.056 | loss_fast:0.061 | loss_kpt_cur:0.0019 | loss_kpt_fut:0.0192
0.22 iters/s | ETA: 25:05:45
```

模型加载和首 step 正常，无 error。

### B.4 监控检查点

| 时间 | Step | Epoch | Loss | 速率 | 状态 |
|:---|:---|:---|:---|:---|:---|
| 09:24 | 50 | 0.10 | 0.168 | 0.22 it/s | ✅ 正常启动 |

