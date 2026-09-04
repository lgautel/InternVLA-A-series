# Phase 2 SFT 训练执行日志（8×H200）

> 对应手册: [`p2sft_planH200.md`](p2sft_planH200.md)
> 执行日期: 2026-09-04
> 操作者: Claude Code (自动化执行)

---

## 用户提供的关键信息

| 项 | 值 |
|:---|:---|
| `WARMUP_CKPT` | `/home/a26113/b/Ckp/itvlaGpR1pro/elvat0714_4D_p1wrmup2609031326_2609031345/checkpoints/000426/pretrained_model` |
| `HF_HOME` | `/B/VENV/hf_home` |
| `HF_LEROBOT_HOME` | `/B/VENV/hf_home/lerobot` |
| `VENV` | `/B/VENV/itnvla15rbt20`（默认） |
| `PROJ_ROOT` | `/B/SRC/itvlaGp`（默认） |
| 数据集 | `/B/Dta/elevator0714_lerobot_4D`（默认） |

> 注: 用户的 Phase 1 ckpt 是 step 426（而非手册中的 400），说明 Phase 1 使用了不同的步数配置，但只要权重结构正确即可。

---

## 执行记录

### 1. 代码改动（§3 完成 05:10）

**§3.1 `configuration_internvla_a1_5.py`**（7 处改动）：
- §3.1.0: `InternVLAA15Config` 新增 `kpt_4d_mode`, `kpt_rot_loss_weight`, `_KPT_4D_DIM` ClassVar, `__post_init__` 派生 `keypoint_track_input_dim`
- §3.1.1: `InternVLAA15DatasetConfig` 新增 `kpt_4d_mode`, `keypoint_dim`, `__post_init__` 派生
- §3.1.2: `Extract3DKeypointTransformFn` 构建传入 `keypoint_dim`; 两个 `UnifyInputs` passthrough 加 `t.keypoint_dim`
- §3.1.3: `UnifyInternVLAA15InputsTransformFn` 新增 `keypoint_dim: int = 3`
- §3.1.4: `_kpt_fields_passthrough_or_zero` 新增参数 `keypoint_dim`, 所有 `3` → `d`
- §3.1.5: VQA 侧同步

**§3.2 `modeling_internvla_a1_5.py`**（4 处改动）：
- `keypoint_out_proj` 输出维度 → `config.keypoint_track_input_dim`
- `embed_kpt_suffix` fallback zeros → `self.config.keypoint_track_input_dim`
- 新增 `_kpt_split_loss` helper（pos MSE + rot normalize+MSE）
- kpt loss 计算使用 `kpt_dim` + `self._kpt_split_loss()`

**§3.3 `transform_internvla_a1_5.py`**（1 处改动）：
- `Extract3DKeypointTransformFn` 新增 `keypoint_dim`, 所有 `3` → `d`

### 2. 数据准备（§5 完成 05:12）

- symlink: `ln -sfn /B/Dta/elevator0714_lerobot_4D /B/VENV/hf_home/lerobot/elevator0714_lerobot_4D` ✓
- norm stats: 直接从 parquet 计算 → `observation.state` 25D, `action` 19D ✓

### 3. Smoke 测试中遇到的 5 个错误与修复

| # | Error | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | `unrecognized: --policy.use_fast_action_tokens` | 该字段仅在 DatasetConfig, 不在 PolicyConfig | 删除 launch L174 |
| 2 | `NCCL ncclInternalError` | 集群无 NCCL tuner | `NCCL_TUNER_PLUGIN=""` |
| 3 | WAN_SMOKE 仍启 8GPU | wrapper 无条件 export 覆盖 | 条件判断 smoke 模式 |
| 4 | `list<double>` vs `float32` | info.json dtype 与 parquet 不匹配 | info.json→float64 + load fallback |
| 5 | `pos_embedding [50,256] vs [75,256]` | Phase1 用 history=200, launch 传 300 | history→200 |

### 4. WAN Smoke 通过（05:25）

```
step:1 | loss:8.601 | loss_action:0.073 | loss_video:0.423 | loss_kpt_cur:0.0014 | loss_kpt_fut:0.0018
step:2 | loss:5.817 | loss_action:0.089 | loss_video:0.381 | loss_kpt_cur:0.0735 | loss_kpt_fut:0.0791
```

### 5. 正式训练（§11）

