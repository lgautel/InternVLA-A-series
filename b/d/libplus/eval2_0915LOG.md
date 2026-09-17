# LIBERO-plus 正式评估执行日志 (step_032070)

> Checkpoint: `/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_12_08_52_49-internvla_a1_5-libplus-sft/checkpoints/032070/pretrained_model/`
> 操作手册: `eval2.md`
> 开始时间: 2026-09-15 02:14
> 上次评估: step_026725 (2026-09-14, 63.5% 完成率, 7个 shard 因 EGL 崩溃)

---

## §1. 初始状态

### §1.1 GPU 状态

- 8 × NVIDIA H200 (143 GB each)
- 训练进程占用全部 8 GPU (~138 GB/卡), 已 kill -9 清空
- 清空后确认: 8 GPU 均 0 MiB 占用

### §1.2 代码修复确认 (从上次评估继承, 全部已验证在位)

| # | 文件 | 修复内容 | 确认状态 |
|---|------|---------|---------|
| 1 | `transform_internvla_a1_5.py:142` | Prompt suffix 根据 `use_fast_action_tokens` 动态选择 | ✅ grep 确认 |
| 2 | `policy_backend_internvla_a1_5.py:148` | `getattr(config, "use_fast_action_tokens", True)` | ✅ grep 确认 |
| 3 | `evaluation/LIBERO/keypoint_utils.py:35,41,56` | `KeypointExtractor` 存储 `_env` 引用而非 `env.sim` | ✅ grep 确认 |
| 4 | `LIBERO-plus/libero/libero/envs/env_wrapper.py:52` | `np.fromstring` → `np.frombuffer` | ✅ grep 确认 |
| 5 | `LIBERO-plus/libero/libero/envs/env_wrapper.py:105` | `np.float_` → `np.float64` | ✅ grep 确认 |
| 6 | `eval_libero_plus.py:221-228` | `evaluate_task()` 用 try-except 包裹防 EGL 崩溃 | ✅ 代码确认 |

### §1.3 新增功能: 动作序列保存 (eval2.md §18 落地)

本次评估前实施了 eval2.md §18 中设计的动作序列保存功能:

| 改动 | 文件 | 内容 |
|------|------|------|
| 新增 | `eval_libero_plus.py` | `--save_actions` CLI 参数 |
| 修改 | `eval_libero_plus.py:evaluate_task()` | 返回 `(successes, task_desc, action_logs)`, 收集每步 action |
| 修改 | `eval_libero_plus.py:evaluate_policy()` | 保存 `.npz` 文件到 `actions/{suite}/task_{id}_ep{ep}.npz` |
| 新建 | `evaluation/LIBERO-plus/replay_episode.py` | 纯 CPU 回放脚本 (从 .npz 重建视频) |
| 修改 | `run_eval_libero_plus_venv.sh` | 添加 `SAVE_ACTIONS_FLAG` 环境变量支持 |

### §1.4 Checkpoint 信息 (032070)

| 项目 | 值 |
|------|-----|
| **Checkpoint 路径** | `.../checkpoints/032070/pretrained_model/` |
| **训练步数** | 32070 / 53450 (60%, 后期 checkpoint) |
| **模型大小** | 5.9 GB (`model.safetensors`, bfloat16) |
| **enable_keypoint_predictor** | `true` |
| **kpt_4d_mode** | `pos_rot` (7D) |
| **use_fast_action_tokens** | `true` (train_config.json dataset 节, config.json 无此字段 → getattr 默认 True) |
| **chunk_size** | 50 |
| **num_inference_steps** | 10 |
| **action_loss_only** | `false` (config), eval 时用 `--action_loss_only` flag 跳过 WAN |

### §1.5 评估配置

| 参数 | 值 |
|------|-----|
| GPU 数量 | 8 (GPU 0-7) |
| Shards per suite | 8 |
| 总工作单元 | 32 (4 suites × 8 shards) |
| 总任务数 | 10030 |
| Trials per task | 1 |
| `stats_key` / `robot_type` | `panda` / `panda` |
| `gripper_convention` | `libero_native` |
| `use_fast_action_tokens` | `True` (从 config getattr) |
| `replan_steps` | 8 |
| `inference_backend` | standard |
| `action_loss_only` | True |
| 保存视频 | 否 (`--no-save_videos`) |
| 保存动作序列 | 是 (`--save_actions`) |
| 评估结果目录 | `full_20260915_021415/` |

---

## §2. 评估启动

(待填充)

---

## §3. 运行时错误与修复

(待填充)

---

## §4. 评估结果

(待填充)
