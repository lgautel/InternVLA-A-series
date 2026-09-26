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
| 09:57 | 500 | 0.96 | 0.212 | 0.23 it/s | ✅ 正常，LR warmup 中 (4.8e-5) |
| 10:26 | 900 | 1.73 | 0.281 | 0.24 it/s | ✅ 正常，LR=5e-5 (warmup done) |
| 10:58 | 1.4K | 2.60 | 0.258 | 0.23 it/s | ✅ 正常 |
| 11:31 | 1.8K | 3.46 | 0.250 | 0.23 it/s | ✅ 正常，p=0.3 (待epoch 5切换) |
| 12:04 | 2.2K | 4.33 | 0.225 | 0.23 it/s | ✅ 正常，p=0.3，epoch 5 (~step 2600) 即将触发首次p切换 |
| 12:34 | 2.6K | 5.09 | 0.212 | 0.23 it/s | ✅ **p 0.30→0.40** (epoch 5, 12:29:43)，schedule 验证通过 |
| 13:04 | 3.0K | 5.86 | 0.212 | 0.22 it/s | ✅ 正常，p=0.4 |
| 13:37 | 3.5K | 6.73 | 0.215 | 0.23 it/s | ✅ 正常，p=0.4，checkpoint@5K 约30min后 |
| 14:06 | 3.9K | 7.50 | 0.192 | 0.23 it/s | ✅ 正常，p=0.4，loss持续下降 |
| 14:39 | 4.3K | 8.36 | 0.184 | 0.23 it/s | ✅ 正常，p=0.4，checkpoint@5K 约15min后 |
| 15:09 | 4.8K | 9.13 | 0.164 | 0.23 it/s | ✅ 正常，p=0.4，ckpt@5K+p切换 即将到来 |
| 15:29 | 5.0K | 9.61 | 0.186 | 0.23 it/s | ✅ **checkpoint@5000 已保存** (18G, 含 training_state) |
| 15:57 | 5.3K | 10.29 | 0.164 | 0.24 it/s | ✅ **p 0.40→0.50** (epoch 10, 15:46:26)，loss 持续下降 |
| 16:31 | 5.8K | 11.15 | 0.144 | 0.23 it/s | ✅ 正常，p=0.5，loss 新低 |
| 17:01 | 6.2K | 11.92 | 0.165 | 0.23 it/s | ✅ 正常，p=0.5 |
| 17:31 | 6.6K | 12.69 | 0.157 | 0.23 it/s | ✅ 正常，p=0.5 |
| 18:01 | 7.0K | 13.46 | 0.149 | 0.22 it/s | ✅ 正常，p=0.5，loss_vqa/fast 显著下降 |
| 18:35 | 7.5K | 14.32 | 0.135 | 0.23 it/s | ✅ 正常，p=0.5，p→0.6 at epoch 15 约20min后 |
| 19:04 | 7.8K | 15.09 | 0.140 | 0.22 it/s | ✅ **p 0.50→0.60** (epoch 15, 19:00:29)，峰值增强 |
| 19:35 | 8.2K | 15.86 | 0.129 | 0.22 it/s | ✅ 正常，p=0.6，loss 新低 0.129 |
| 20:05 | 8.7K | 16.63 | 0.130 | 0.22 it/s | ✅ 正常，p=0.6，loss 稳定在 ~0.13 |
| 20:36 | 9.1K | 17.40 | 0.122 | 0.22 it/s | ✅ 正常，p=0.6，loss 新低 0.122 |
| 21:10 | 9.5K | 18.26 | 0.116 | 0.23 it/s | ✅ 正常，p=0.6，loss↓0.109(9.4K)，ckpt@10K ~30min |
| 21:49 | 10.0K | 19.23 | 0.101 | 0.22 it/s | ✅ **checkpoint@10000 已保存** (18G, 含 optimizer 12G)，loss 新低 0.101 |
| 22:21 | 10.4K | 19.99 | 0.110 | 0.23 it/s | ✅ **p 0.60→0.50** (epoch 20, 22:21:21)，进入ramp-down阶段 |
| 22:51 | 10.8K | 20.76 | 0.108 | 0.22 it/s | ✅ 正常，p=0.5，loss 稳定 ~0.108 |
| 23:22 | 11.2K | 21.53 | 0.106 | 0.22 it/s | ✅ 正常，p=0.5，grdn↓1.37 |
| 23:52 | 11.6K | 22.30 | 0.100 | 0.23 it/s | ✅ 正常，p=0.5，loss 触及 0.100 |
| 00:25 | 12.1K | 23.17 | 0.096 | 0.23 it/s | ✅ 正常，p=0.5，loss 新低 0.096，p→0.4 at epoch 25 ~1h |
| 00:55 | 12.4K | 23.94 | 0.103 | 0.23 it/s | ✅ 正常，p=0.5，p→0.4 at epoch 25 ~30min |
| 01:25 | 12.8K | 24.71 | 0.101 | 0.22 it/s | ✅ 正常，p=0.5，最终p切换 ~5min |
| 01:59 | 13.3K | 25.57 | 0.098 | 0.22 it/s | ✅ **p 0.50→0.40** (epoch 25, 01:36:38)，全部5次切换完成 |
| 02:29 | 13.7K | 26.34 | 0.086 | 0.23 it/s | ✅ 正常，p=0.4(final)，loss 新低 0.086 |
| 02:59 | 14.1K | 27.11 | 0.092 | 0.23 it/s | ✅ 正常，p=0.4，ckpt@15K ~1h |
| 03:28 | 14.5K | 27.88 | 0.092 | 0.23 it/s | ✅ 正常，p=0.4，ckpt@15K ~30min |
| 04:02 | 14.9K | 28.74 | 0.085 | 0.22 it/s | ✅ 正常，p=0.4，loss 新低 0.085，ckpt@15K 即将 |
| 04:07 | 15.0K | 28.84 | 0.088 | 0.23 it/s | ✅ **checkpoint@15000 已保存** (18G)，75% 完成 |
| 04:35 | 15.3K | 29.51 | 0.090 | 0.24 it/s | ✅ 正常，p=0.4，grdn 首次 <1.0 |
| 05:08 | 15.8K | 30.38 | 0.086 | 0.23 it/s | ✅ 正常，p=0.4，loss_fast=0.005 |
| 05:38 | 16.2K | 31.15 | 0.092 | 0.23 it/s | ✅ 正常，p=0.4，LR=8.9e-6 |
| 06:08 | 16.6K | 31.91 | 0.088 | 0.23 it/s | ✅ 正常，p=0.4 |
| 06:42 | 17.1K | 32.78 | 0.096 | 0.23 it/s | ✅ 正常，p=0.4，85% 完成 |
| 07:15 | 17.5K | 33.65 | 0.092 | 0.23 it/s | ✅ 正常，p=0.4，ETA ~3h |
| 07:44 | 17.9K | 34.41 | 0.096 | 0.23 it/s | ✅ 正常，p=0.4，ETA ~2.5h |
| 08:14 | 18.3K | 35.18 | 0.090 | 0.23 it/s | ✅ 正常，p=0.4，ETA ~2h |
| 08:48 | 18.8K | 36.05 | 0.089 | 0.23 it/s | ✅ 正常，p=0.4，ETA ~1.5h |
| 09:18 | 19.1K | 36.82 | 0.092 | 0.23 it/s | ✅ 正常，p=0.4，ETA ~1h，预计10:20完成 |
| 09:48 | 19.6K | 37.59 | 0.086 | 0.23 it/s | ✅ 正常，p=0.4，ETA 00:32，即将完成！ |
| 10:21 | **20.0K** | 38.45 | 0.091 | 0.23 it/s | ✅✅ **训练完成！** checkpoint@20000 已保存 (18G) |

---

## C. 训练完成总结

### C.1 训练结果

| 项目 | 值 |
|:---|:---|
| 开始时间 | 2026-09-23 09:18:27 |
| 结束时间 | 2026-09-24 10:21:07 |
| 总训练时长 | 25h 00m 34s |
| 最终 step | 20,000 |
| 最终 epoch | 38.45 |
| 最终 loss | 0.091 |
| 最低 loss | 0.082 (step 17.0K) |
| 平均速率 | 0.23 it/s |
| 最终 LR | 5.0e-6 |
| 总 samples | ~3M |
| 总 episodes | ~4K |

### C.2 Loss 演变

| 阶段 | Step | Loss | 说明 |
|:---|:---|:---|:---|
| 起始 | 50 | 0.168 | 初始 loss |
| p=0.3 稳定 | 2.2K | 0.225 | warmup 结束 |
| p=0.4 | 5.3K | 0.164 | 第二阶段 |
| p=0.5 | 7.5K | 0.135 | 第三阶段 |
| p=0.6 (峰) | 9.1K | 0.122 | 峰值增强阶段 |
| p=0.5 (下行) | 11.2K | 0.106 | ramp-down |
| p=0.4 (最终) | 13.7K | 0.086 | 最终阶段 |
| 最终 | 20.0K | 0.091 | 完成 |

### C.3 p-schedule 切换记录

| 时间 | Epoch | p 切换 | schedule_idx |
|:---|:---|:---|:---|
| 09-23 12:29:43 | 5 | 0.30 → 0.40 | 1/5 |
| 09-23 15:46:26 | 10 | 0.40 → 0.50 | 2/5 |
| 09-23 19:00:29 | 15 | 0.50 → 0.60 | 3/5 |
| 09-23 22:21:21 | 20 | 0.60 → 0.50 | 4/5 |
| 09-24 01:36:38 | 25 | 0.50 → 0.40 | 5/5 |

全部 5 次 p-schedule 切换均按设计准确触发。

### C.4 Checkpoints

| Step | 时间 | 大小 | 路径 |
|:---|:---|:---|:---|
| 5000 | 09-23 15:27 | 18G | `checkpoints/005000/` |
| 10000 | 09-23 21:47 | 18G | `checkpoints/010000/` |
| 15000 | 09-24 04:05 | 18G | `checkpoints/015000/` |
| 20000 | 09-24 10:21 | 18G | `checkpoints/020000/` |

所有 checkpoint 均含 `pretrained_model/` (model.safetensors 5.9G + config.json + stats.json + train_config.json) 和 `training_state/`。

Checkpoint 根目录: `~/b/Ckp/itvlagpFrkPlug0907_dtaug2ps/2026_09_23_09_18_27-internvla_a1_5-frk-plug-sft-dtaug2ps/`

### C.5 后处理

- [x] 训练日志已归档至 checkpoint 根目录 (`train.log`, `wrapper.log`)
- [x] GPU 占位任务 `bigmatrix_multiply_optimization.py` 已启动 (PID 3320199)
- [x] 训练过程中无任何 error 或 exception
- [x] 全程无中断，无需重启

