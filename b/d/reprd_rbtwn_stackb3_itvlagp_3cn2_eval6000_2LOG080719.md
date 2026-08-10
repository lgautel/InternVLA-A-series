# itvlaGp RoboTwin stack_bowls_three 评估 — kpt-only 080719 执行日志

> 参考：`reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG.md`、`2LOG2.md`、`reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2.md`
>
> **代码库**：`/home/luogang/SRC/Robot/itvlaGp/`（评估脚本）  
> **Checkpoint 项目**：`/home/luogang/SRC/Robot/itvlaGp080719/`（仅 outputs）  
> **Checkpoint**：`.../2026_08_07_12_25_57-internvla_a1_5-geop-phase2-action-kpt-only-stackb3-abs-10k/checkpoints/010000/pretrained_model`  
> **输出路径**：`itvlaGp080719/outputs/robotwin/kpt_only_stackb3_010k/`

---

## 时间线 / 操作日志

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-08-07 23:41 | 开始评估 (kpt-only 080719 checkpoint) | 进行中 |
| 2026-08-07 23:42 | Checkpoint 修复：`model.safetensors_.gstmp` → `model.safetensors` | OK (1303 keys, 5.9G) |
| 2026-08-07 23:42 | 15 项预检检查表 | **15/15 PASS** |
| 2026-08-07 23:42 | 2-episode 冒烟测试 (demo_clean, GPU0) | **BLOCKED** — SAPIEN Vulkan 失败 (Problem #2) |
| 2026-08-08 07:24 | 驱动修复后恢复评估 | nvidia-smi OK (580.173.02), sapien.Scene() OK |
| 2026-08-08 07:24 | 2-episode 冒烟测试 (demo_clean, GPU0) | **PASS** — exit 0, 2/2 episode 完成 (0/2 成功, 无崩溃) |
| 2026-08-08 07:30 | 双 GPU 并行正式评估 (各 100 ep) | 进行中 |

### 正式评估启动命令

```bash
# demo_clean → GPU0 (PID 15256)
CUDA_VISIBLE_DEVICES=0 python -u evaluation/RoboTwin/inference.py \
  --ckpt-path ".../010000/pretrained_model" \
  --video-dir ".../kpt_only_stackb3_010k/robotwin/demo_clean/stack_bowls_three" \
  --task-config demo_clean --task-idx 46 --action-mode abs \
  --infer-horizon 20 --inference-backend standard --num-episodes 100 --dtype bfloat16

# demo_randomized → GPU1 (PID 15334)
CUDA_VISIBLE_DEVICES=1 python -u evaluation/RoboTwin/inference.py \
  --video-dir ".../kpt_only_stackb3_010k/robotwin/demo_randomized/stack_bowls_three" \
  --task-config demo_randomized ...
```

| 2026-08-08 08:00 | 进度检查 #2 | demo_clean 0/28, demo_randomized 0/26, 双进程均 exit 1 |
| 2026-08-08 08:00 | **Problem #3**: 磁盘满导致评估中断 | 见下方 |

### Problem #3: 磁盘空间耗尽 (No space left on device)

| 项 | 内容 |
|----|------|
| **发现时机** | demo_clean ep29 / demo_randomized ep27 进行中 |
| **症状** | `tee: ...demo_clean.log: No space left on device`; 双进程 exit code 1 |
| **根因** | 评估日志 + mp4 视频持续写入，磁盘从 73% 涨至 100% |
| **已完成进度** | demo_clean 28/100 ep (0 成功), demo_randomized 26/100 ep (0 成功) |
| **修复方案** | 清理 pip cache (~20G); 备份 run1 部分结果; 重启 run2 双 GPU 评估 |
| 2026-08-08 12:58 | **run2 demo_clean 完成** | **0/100 = 0.0%**, exit code 0 |
| 2026-08-08 13:06 | **run2 demo_randomized 完成** | **0/100 = 0.0%**, exit code 0 |
| 2026-08-08 13:06 | **评估全部完成** | run2 有效结果见下表 |

---

## 最终结果 (run2, 有效)

| 配置 | 成功/总数 | 成功率 | exit | 日志 |
|------|-----------|--------|------|------|
| demo_clean | 0/100 | **0.0%** | 0 | `eval_kpt_only_stackb3_010k_demo_clean_run2.log` |
| demo_randomized | 0/100 | **0.0%** | 0 | `eval_kpt_only_stackb3_010k_demo_randomized_run2.log` |

### 与历史 baseline 对比 (itvlaGp p2 checkpoint, LOG/LOG2)

| 配置 | p2 (LOG1) | p2 rerun2 (LOG2) | **kpt-only 080719 (本次)** |
|------|-----------|------------------|---------------------------|
| demo_clean | 64.0% | 63.0% | **0.0%** |
| demo_randomized | 16.0% | 22.0% | **0.0%** |

**初步分析**：kpt-only phase2 checkpoint 在 RoboTwin 上零成功率，与 p2 差距极大，需排查：checkpoint config（`enable_keypoint_predictor`、normalization stats）、权重是否完整加载、action 输出是否合理（可抽查 mp4 / open-loop）。

---

### Problem #1: Checkpoint 权重文件未完成重命名

| 项 | 内容 |
|----|------|
| **发现时机** | 评估启动前检查 checkpoint 目录 |
| **症状** | 仅有 `model.safetensors_.gstmp` (5.9G)，无 `model.safetensors` |
| **根因** | 从训练机同步至 `itvlaGp080719` 时 safetensors 原子写入未完成 rename |
| **修复** | 验证 1303 keys 完整后 `mv model.safetensors_.gstmp model.safetensors` |
| **验证** | 预检 [11] PASS |

### Problem #2: NVIDIA 驱动内核/用户态版本不匹配 → SAPIEN 无法渲染（阻塞）

| 项 | 内容 |
|----|------|
| **发现时机** | 冒烟测试 seed 验证阶段 |
| **症状** | `RuntimeError: vk::PhysicalDevice::createDeviceUnique: ErrorExtensionNotPresent`；`Failed to initialize NVML: Driver/library version mismatch` |
| **根因** | 内核模块 NVRM **580.159.03** vs 用户态库 **580.173.02**（驱动升级后未 reload 内核模块/未 reboot） |
| **影响** | 所有 RoboTwin episode 在 `SapienRenderer()` 初始化时失败，零有效评估 |
| **修复方案** | **需 root**：`sudo reboot` 或 `sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia` |
| **验证** | `nvidia-smi` 无 mismatch 警告；`python -c "import sapien; sapien.Scene()"` 成功 |
| **状态** | **已修复** — reboot 后 nvidia-smi 580.173.02，sapien.Scene() OK (2026-08-08) |

---
