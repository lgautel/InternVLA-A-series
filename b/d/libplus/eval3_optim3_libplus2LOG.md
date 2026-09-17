# eval3_optim3_libplus2LOG — LIBERO-plus 六 Server 正式评估日志

> 按 `eval3_optim3.md` §二十（标准部署方案）和 §十七（Phase 4 LIBERO-plus 重跑）执行。
> 记录所有操作、命令、错误、修复和关键路径。

---

## 0. 前提条件检查

**硬件**: 8× GPU (143,771 MiB each), 全部空闲 (0 MiB used)

**端口**: 5784-5789 全部空闲

**Preflight**: 137/137 PASS (test_preflight_f1f2.py)

**Gate 3 已通过**: 标准 LIBERO SR=95.25% (1905/2000)
- 结果路径: `.../032070/libero_std_2server_6client_20260916155404/overall_std_results.json`

**Checkpoint**: `/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_12_08_52_49-internvla_a1_5-libplus-sft/checkpoints/032070/pretrained_model`
- `model.safetensors`: 存在
- `stats.json` panda key: OK
- `state_dim=8`, `action_dim=7`, `gripper_min=-1.00` (libero_native)
- `enable_kpt_predictor=True`, `kpt_4d_mode=pos_rot`

**EGL vendor**: `/B/VENV/libero_plus_client/egl_vendor.d/10_nvidia.json` — OK

**task_classification.json**: 存在
- libero_spatial: 2402 tasks
- libero_object: 2518 tasks
- libero_goal: 2591 tasks
- libero_10: 2519 tasks
- **总计: 10,030 tasks** (每 task 1 trial = 10,030 episodes)

**启动器语法**: `bash -n run_eval_libero_plus_2server_6client_venv.sh` — OK

**S1 前提全部通过。**

---

## 1. Gate 4a：冒烟测试

**目的**: 验证六 server 启动、EGL 渲染、WebSocket 通信、结果写出全链路稳定。
smoke 模式使用 `MAX_STEPS_OVERRIDE=20`（步数太短，不测 SR），仅测稳定性。

**执行时间**: 2026-09-16

### 1.1 执行命令

```bash
cd /B/SRC/itvlaGpLibPlus

export CKPT_PATH="/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_12_08_52_49-internvla_a1_5-libplus-sft/checkpoints/032070/pretrained_model"
export LIBERO_HOME="/home/a26113/DATA/LIBERO-plus"
export SERVER_VENV="/B/VENV/itnvla15rbt20"
export CLIENT_VENV="/B/VENV/libero_plus_client"
export VLM_MODEL_PATH="/B/VENV/hf_home/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"
export PROJ="/B/SRC/itvlaGpLibPlus"
export ROTATE_IMAGES=false
unset NUM_TRIALS_PER_TASK

EVAL_MODE=smoke \
SMOKE_SUITE=libero_spatial \
SMOKE_TASKS=6 \
MAX_STEPS_OVERRIDE=20 \
SAVE_ACTIONS_FLAG=--no-save_actions \
bash evaluation/LIBERO-plus2/run_eval_libero_plus_2server_6client_venv.sh
```

### 1.2 执行结果

