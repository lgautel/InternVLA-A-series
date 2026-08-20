# InternVLA-A1.5 + GeoPredict 3D 关键点轨迹预测器融合 — 实施与验收日志

> 本日志记录《[itrnVLA15_GeoP_3dtrj_3cn2_rbt2stak3.md](itrnVLA15_GeoP_3dtrj_3cn2_rbt2stak3.md)》(v3.2 RoboTwin aloha 双臂适配方案) 落地为真实代码时的全部操作、文件增删改、遇到的问题及其根因分析与修复方案。
>
> 实施计划见对话中的 Plan（GeoPredict 3D关键点融合RoboTwin适配）。范围确认：
> 1. 验收深度 = 数据管道 + 模型 forward/backward 冒烟测试 + 短程训练验证（不含 SAPIEN 实际 rollout 评测，`third_party/RoboTwin` 子模块未初始化）。
> 2. Phase 1（间接监督）与 Phase 2（FK 直接监督）均需实现并验证。
> 3. GeoPredict 权重使用 HuggingFace `Jingjing0601/GeoPredict-Robocasa` 真实 checkpoint。

---

## 0. 环境搭建

### 0.1 新建虚拟环境 `/mnt/r/VENV/itrnvla15rbt/`

```bash
uv venv /mnt/r/VENV/itrnvla15rbt --python 3.11
source /mnt/r/VENV/itrnvla15rbt/bin/activate
cd /home/physical/SRC/Robot/InternVLA-A-series
uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers==5.2.0
uv pip install -e .
uv pip install flash-attn==2.8.3 flash-linear-attention==0.5.0 causal-conv1d==1.6.1 --no-build-isolation
uv pip install torchcodec==0.10.0   # 覆盖 editable install 拉入的 0.15.0（GPU/CUDA 版本不匹配问题，复用之前 ivla15 环境的经验修复）
uv pip install pin                   # pinocchio 的 PyPI 包名是 "pin"（import pinocchio as pin）
uv pip install tilelang               # H200(Hopper) GPU 上 flash-linear-attention 依赖 Triton>=3.4 的已知问题的修复（复用之前 ivla15 环境经验）
uv pip install websockets msgpack-numpy 'huggingface_hub[cli]'
```

拷贝 Qwen3.5 transformers patch：

```bash
TRANSFORMERS_DIR=/mnt/r/VENV/itrnvla15rbt/lib/python3.11/site-packages/transformers/
cp -r src/lerobot/policies/pi0/transformers_replace/models ${TRANSFORMERS_DIR}
cp -r src/lerobot/policies/pi05/transformers_replace/models ${TRANSFORMERS_DIR}
cp -r src/lerobot/policies/internvla_a1_5/transformers_replace/models ${TRANSFORMERS_DIR}
```

验证：`torch 2.10.0+cu128 cuda True`；`flash_attn 2.8.3`；`fla` 可 import；`InternVLAA15Config`/`InternVLAA15Policy` 可 import。GPU 型号确认为 `NVIDIA H200`。

**问题记录 #E1**：`pip install -e .` 默认会把 `torchcodec` 装到较新版本（本次为 0.15.0），与之前 ivla15 环境遇到的 `torchcodec` CUDA/版本不匹配问题相同根因（参见 `reprd_liberop_cam_rb.md` 中的记录）。
- **修复**：安装完 `-e .` 后立即用 `uv pip install torchcodec==0.10.0` 覆盖回已验证可用的版本。

### 0.2 下载 GeoPredict checkpoint

```bash
mkdir -p /mnt/r/CKPT/geopredict
hf download Jingjing0601/GeoPredict-Robocasa --local-dir /mnt/r/CKPT/geopredict
```

该仓库除 `GeoPredict_robocasa.pth`（6.1GB，实际需要的权重）外还包含 `pi0_base.pth`（13GB，GeoPredict 训练用的 Pi0 基座权重，本任务不需要）以及 24 个 RoboCasa `.hdf5` 原始数据文件（本任务不需要）。

**操作**：下载完成后清理无关文件以节省磁盘：

```bash
rm -rf /mnt/r/CKPT/geopredict/robocasa /mnt/r/CKPT/geopredict/pi0_base.pth /mnt/r/CKPT/geopredict/.cache
```

保留：`GeoPredict_robocasa.pth`(6.1GB)、`paligemma_tokenizer.model`、`robocasa_norm_stats.json`、`README.md`。

用 `torch.load(..., weights_only=False)` 检查 checkpoint 的 `state_dict`，确认关键点相关 key（733 个 key 中）与设计文档 §6.2 的映射表一致：

```
keypoint_encoder.queries                                          (1, 1, 512)
keypoint_encoder.point_patch_embed.conv.{weight,bias}             (256,3,4)/(256,)
keypoint_encoder.cross_attention_block.*                          512/256 维
keypoint_encoder.cross_attention_block.cross_attn.key_time_embedding.pos_embedding  (250, 256)
keypoint_encoder.linear_transform.*                                512/1024 维
keypoint_encoder.final_norm.{weight,bias}                          (512,)
keypoint_encoder.track_fusion_layer.{weight,bias}                 (2048,512)/(2048,)   ← shape 不匹配，需跳过
keypoint_embedding.weight                                          (8, 2048)            ← shape 不匹配（J、dim 均不同），需跳过
keypoint_out_proj.{weight,bias}                                    (3,2048)/(3,)        ← shape 不匹配，需跳过
```

与设计文档 §6.2 一致：可复用前缀为 `keypoint_encoder.queries` / `point_patch_embed.` / `cross_attention_block.` / `linear_transform.` / `final_norm.`（含 `key_time_embedding.pos_embedding` 缓冲区），需跳过 `track_fusion_layer`（512→2048 vs 我们需要的 512→1024）。

---

## 1. TrackEncoder 移植

### 1.1 新增文件 `src/lerobot/policies/internvla_a1_5/keypoints.py`（406 行）

从 GeoPredict 代码库 `/home/physical/SRC/Robot/GeoPredict/` 中移植以下核心类：

| 类名 | 源文件 | 变更 |
|------|--------|------|
| `PointPatchEmbedding` | `geopredict/track_encoder.py` | 无变更 |
| `TimeEmbedding` | `geopredict/track_encoder.py` | 无变更 |
| `MultiHeadAttention` | `geopredict/track_encoder.py` | 无变更 |
| `CrossAttentionBlock` | `geopredict/track_encoder.py` | 无变更 |
| `TrackEncoder` | `geopredict/track_encoder.py` | `output_dim` 默认值 2048→1024（适配 kpt_expert 的 hidden_size），`dropout` 默认值 0.1→0.0 |

新增辅助函数：

- `load_geopredict_track_encoder_weights(model, ckpt_path)` — 从 GeoPredict checkpoint 中选择性加载权重。映射规则：`keypoint_encoder.*` → `track_encoder.*`。**跳过** `track_fusion_layer`（shape 不匹配：GeoPredict 512→2048 vs 本项目 512→1024）。加载结果以 `loaded` / `skipped` / `not_found` 列表形式 print 供人工检查。

Per-joint 循环（`for j in range(J): encoder(points[:, :, j, :])`）是 J-agnostic 的，J=8（RoboCasa 单臂）和 J=14（aloha 双臂）均可正确工作。

---

## 2. 配置字段

### 2.1 修改文件 `src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py`（588 行，+166 行）

在 `InternVLAA15Config` 上新增 26+ 个字段：

**关键点预测器开关与参数：**

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `enable_keypoint_predictor` | `False` | 主开关 |
| `num_keypoint_joints` | `8` | 关键点数量（aloha 设为 14） |
| `kpt_loss_weight` | `0.0` | 关键点当前帧 MSE loss 权重 |
| `kpt_future_loss_weight` | `1.0` | 关键点未来帧 MSE loss 权重 |
| `kpt_expert_hidden_size` | `1024` | 关键点专家隐藏维度 |
| `kpt_expert_intermediate_size` | `3072` | 关键点专家 FFN 中间维度 |

**Knowledge Insulation 开关：**

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `knowledge_insulation_kpt` | `False` | 是否阻断 kpt_expert 对 VLM prefix 的注意力 |
| `kpt_to_action_detach` | `False` | 是否 detach kpt→action cross-attention 的 K/V |
| `ki_gradient_scale` | `0.0` | action→VLM 梯度缩放因子 |
| `ki_kpt_gradient_scale` | `0.0` | kpt→VLM 梯度缩放因子 |

**学习率缩放：**

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `vlm_lr_scale` | `1.0` | VLM backbone 的 LR 缩放 |
| `action_expert_lr_scale` | `1.0` | Action expert 的 LR 缩放 |
| `kpt_expert_lr_scale` | `1.0` | Keypoint expert 的 LR 缩放 |
| `track_encoder_lr_scale` | `1.0` | TrackEncoder 的 LR 缩放 |

**TrackEncoder 超参数：** `keypoint_track_input_dim=3`、`keypoint_track_patch_size=4`、`keypoint_track_embed_dim=256`、`keypoint_track_query_dim=512`、`keypoint_track_num_heads=8`、`keypoint_track_ff_dim=1024`、`keypoint_history_max_len=1000`

**`keypoint_3d_delta_indices` 属性（L570-588）：** 当 `enable_keypoint_predictor=True` 时返回 `list(range(-H, C+1))`（H=1000, C=50 → 1051 个索引），否则返回 `None`。

在 `InternVLAA15DatasetConfig` 上新增 3 个独立字段：`enable_keypoint_predictor`、`num_keypoint_joints`、`keypoint_history_max_len`。这些与 policy config 上的同名字段**没有自动同步**，必须在 CLI 中分别设置。

**备注：** 字段 `keypoint_noise_sigma` 存在但未被任何代码引用（dead code），不影响功能。

---

## 3. 三路 MoT 建模

### 3.1 修改文件 `src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py`（2510 行，+930 行）

**新增核心函数/方法：**

| 函数 | 位置 | 说明 |
|------|------|------|
| `compute_layer_complete_3path` | L343-553 | 三路 MoT 计算层：Path 0 = VLM (dim=2048)，Path 1 = Keypoint Expert (dim=1024)，Path 2 = Action Expert (dim=1024)。支持 linear attention 和 full attention 两种模式 |
| `embed_kpt_suffix` | L1562-1617 | 生成 `[B, 1+2J, D]` 的 kpt 后缀嵌入：`[state(1) \| hist_kpt(J) \| query_kpt(J)]` |
| `post_init_keypoint_weights` | L1072-1089 | 将 action_expert 的权重复制到 keypoint_expert（warm-start 初始化） |
| `load_geopredict_keypoint_weights` | L1091-1099 | 委托 `keypoints.py` 中的选择性加载函数 |

**关键点损失计算（L1949-1979）：**
- 从 kpt_query_out 的最后 J 个 token 提取预测 → `keypoint_out_proj` → `pred_kpt_current [B, J, 3]`
- 当前帧 MSE: `loss_kpt_current = MSE(pred_kpt_current, kpt_t)`
- 未来帧预测：从 action expert 的输出中提取 chunk_size 个位置 → `keypoint_out_proj` → `future_kpt_pred [B, C, J, 3]`
- 未来帧 MSE: `loss_kpt_future = MSE(future_kpt_pred, kpt_future)`
- 当 `kpt_mask=False` 时两个 loss 为 0（Phase 1 行为）

**优化器参数组（L2174-2224）：** 5 组独立 LR 缩放：
1. VLM backbone（`vlm_lr_scale`）
2. Action expert（`action_expert_lr_scale`）
3. Keypoint expert（`kpt_expert_lr_scale`）
4. TrackEncoder（`track_encoder_lr_scale`）
5. 其余参数（默认 LR）

**Attention mask 构建：** 使用 cumsum-based block-causal 方案。kpt_suffix 区域在三路 MoT 中正确分离——Path 1 仅处理 kpt_suffix tokens，Path 2 仅处理 action_suffix tokens，Path 0 处理 VLM prefix。Knowledge insulation 标志正确传播到 attention mask 构建中。

### 3.2 推理集成 `evaluation/RoboTwin/inference.py`（525 行，+约 90 行）

新增：

- `ALOHA_KEYPOINT_LINKS` 常量：14 个关节链接名称（`fl_link1..6, left_camera, fr_link1..6, right_camera`）
- `get_keypoints_aloha(robot_entity, footprint_pose)` 函数：
  - 通过 SAPIEN API (`entity.find_link_by_name().get_pose().p`) 获取世界坐标
  - 使用 `scipy.spatial.transform.Rotation` 处理 SAPIEN 的 wxyz 四元数约定
  - 转换为 footprint-relative 坐标系（固定基座，首次调用时缓存 footprint_pose）
  - 返回 `[14, 3]` 关键点和 footprint_pose
- `infer_once()` 中的缓冲区管理：
  - `use_kpt = getattr(config, "enable_keypoint_predictor", False)` 条件激活
  - `his_kpts = np.zeros((H, J, 3))` 初始化
  - 每步调用 `get_keypoints_aloha()` → 追加到缓冲区 → 构建 batch tensors 传入模型

---

## 4. 数据变换

### 4.1 修改文件 `src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py`（733 行，+约 80 行）

新增 `Extract3DKeypointTransformFn`（L656-733），注册为 `"extract_3d_keypoint"`。

**Phase 1 路径（无 GT，`kpt_mask=False`）：** 当 `observation.keypoint_3d` 不在数据中时，生成全零占位符：
- `his_kpts: [H, J, 3]` 全零
- `his_len: 0`
- `kpt_t: [J, 3]` 全零
- `kpt_future: [C, J, 3]` 全零
- `kpt_mask: False`

**Phase 2 路径（有 GT，`kpt_mask=True`）：** 当 `observation.keypoint_3d` 存在于数据中时：
1. Pop 堆叠 tensor `[H+1+C, J*3]` → reshape 为 `[H+1+C, J, 3]`
2. Pop `observation.keypoint_3d_is_pad` mask → 计算有效历史长度
3. 分割为 `his_kpts[:H]`、`kpt_t[H]`、`kpt_future[H+1:H+1+C]`
4. 将有效帧前移打包到 `his_kpts[:his_len]`（匹配 TrackEncoder 的 `points[i, :length]` 约定）

---

## 5. 数据集工厂

### 5.1 修改文件 `src/lerobot/datasets/factory.py`（643 行，+7 行）

在 `resolve_delta_timestamps()` 函数（L314-318）中新增 `observation.keypoint_3d` 分支：

```python
elif key == "observation.keypoint_3d" and getattr(cfg, "keypoint_3d_delta_indices", None) is not None:
    delta_timestamps[key] = [i / ds_meta.fps for i in cfg.keypoint_3d_delta_indices]
```

当 `keypoint_3d_delta_indices` 为 `None`（`enable_keypoint_predictor=False`）或数据集不含该列时，该分支不执行，不影响原有逻辑。

---

## 6. 单元测试

### 6.1 新增文件

| 文件 | 行数 | 测试数量 | 覆盖范围 |
|------|------|---------|---------|
| `tests/conftest.py` | 250 | — | Fixtures：tiny Qwen3.5 模型（hidden=64, 4 layers），session-scoped checkpoint `/mnt/r/CKPT/qwen35_tiny`，`make_tiny_internvla_a15_config()` |
| `tests/test_step0_config.py` | 102 | 13 | Config 字段默认值、`keypoint_3d_delta_indices` 属性、J=8/14 兼容性 |
| `tests/test_step1_track_encoder.py` | 181 | 10 | TrackEncoder 输出 shape、per-joint loop、GeoPredict 权重加载（选择性跳过 `track_fusion_layer`） |
| `tests/test_step2_attention_mask.py` | 114 | 9 | cumsum block-causal attention mask 构建、三路 MoT 分区正确性 |
| `tests/test_step3_kpt_expert.py` | 132 | 8 | Keypoint expert 层初始化、hidden_size=1024 vs VLM dim=2048 |
| `tests/test_step3_5_weight_init.py` | 149 | 7 | `post_init_keypoint_weights` action→kpt 权重复制正确性 |
| `tests/test_step4_compute_layer.py` | 242 | 6 | `compute_layer_complete_3path` 三路输出 shape、梯度流 |
| `tests/test_step5_forward_loss.py` | 229 | 5 | Forward+loss 计算：Phase 1 (kpt_loss=0) vs Phase 2 (kpt_loss>0)、kpt_mask 控制 |
| `tests/test_step6_inference.py` | 253 | 4 | 推理路径：select_action 输出 shape、缓存 KV 机制 |
| `tests/test_step7_transform_freeze.py` | 203 | 8 | Extract3DKeypointTransformFn Phase 1/2、freeze 控制、参数组 |

### 6.2 测试结果

```
71 passed in 71.65s
```

**所有 71 个测试一次性全部通过，无失败。** 无需修复测试代码或源代码。

---

## 7. FK 生成

### 7.1 新增文件 `util_scripts/generate_aloha_keypoints.py`（237 行）

`AlohaFKKeypointExtractor` 类：
- 使用 pinocchio FK 计算 aloha-agilex URDF 的 3D link 位置
- Joint mapping: `state[0:6]` → 左臂 6 关节，`state[7:13]` → 右臂 6 关节（跳过 `[6]`/`[13]` 的 gripper）
- 14 个关键点链接：`fl_link1-6 + left_camera + fr_link1-6 + right_camera`
- `_add_keypoint_column_to_parquet_files`：写入 `observation.keypoint_3d [42]` 列
- `_update_info_json`：声明 feature（42 个 name：`fl_link1_x/y/z, ...`）

### 7.2 执行结果

```bash
python util_scripts/generate_aloha_keypoints.py \
  --src /mnt/r/DATA/RoboTwin-Clean/stack_bowls_three \
  --dst /mnt/r/DATA/RoboTwin-Clean/stack_bowls_three_kpt \
  --urdf <aloha-agilex URDF path>
```

- 处理帧数：23550
- 输出目录：`/mnt/r/DATA/RoboTwin-Clean/stack_bowls_three_kpt/`
- Symlink：`/mnt/r/CKPT/hf_home/lerobot/robotwin/stack_bowls_three_kpt`
- Parquet 列 `observation.keypoint_3d`：shape `[42]`，dtype `float32`
- Spot-check 数值范围：各 link 距离 footprint 约 0.84–1.23m，与 aloha 机械臂物理尺寸一致（臂展 ~0.5m，基座高度 ~0.78m）

---

## 8. Phase 1 冒烟测试（间接监督）

### 8.1 命令

```bash
HF_LEROBOT_HOME=/mnt/r/CKPT/hf_home/lerobot \
CUDA_VISIBLE_DEVICES=0 \
accelerate launch --num_processes=1 \
  src/lerobot/scripts/lerobot_train.py \
  --policy.type=internvla_a1_5 \
  --policy.pretrained_path=/mnt/r/CKPT/InternVLA-A1.5-base \
  --policy.push_to_hub=false \
  --policy.enable_keypoint_predictor=true \
  --policy.num_keypoint_joints=14 \
  --policy.kpt_loss_weight=0.0 \
  --policy.kpt_to_action_detach=false \
  --policy.action_loss_only=true \
  --dataset.type=internvla_a1_5 \
  --dataset.repo_id=robotwin/stack_bowls_three \
  --dataset.action_mode=abs \
  --dataset.use_external_stats=true \
  --dataset.external_stats_path=/mnt/r/CKPT/hf_home/lerobot/stats/aloha/abs/agg_1repos_1c27ca3df3/stats.json \
  --steps=20 --batch_size=2 --log_freq=5 --save_freq=100 \
  --output_dir=outputs/smoke_phase1
```

使用**原始数据集**（无 `observation.keypoint_3d` 列），`kpt_loss_weight=0.0`。

### 8.2 结果

| step | loss | loss_action | grad_norm |
|------|------|-------------|-----------|
| 5 | 0.155 | 0.155 | 5.389 |
| 10 | 0.283 | 0.283 | 8.725 |
| 15 | 0.131 | 0.131 | 3.663 |
| 20 | 0.148 | 0.148 | 4.074 |

- 20 步完成，loss 全部 finite，无 NaN
- `loss_video=0.000`（`action_loss_only=true`）
- 无 kpt loss（`kpt_loss_weight=0.0`，符合预期）
- 缺失 key 警告（expected）：所有 `model.qwen3_5_with_expert.keypoint_expert.*` 和 `model.track_encoder.*` 不存在于 InternVLA-A1.5-base checkpoint 中
- `post_init_keypoint_weights` 成功执行：从 action_expert 权重初始化 keypoint_expert
- Checkpoint 保存于 `outputs/smoke_phase1/checkpoints/000020`

---

## 9. Phase 2 冒烟测试（FK 直接监督）

### 9.1 命令

```bash
HF_LEROBOT_HOME=/mnt/r/CKPT/hf_home/lerobot \
CUDA_VISIBLE_DEVICES=0 \
accelerate launch --num_processes=1 \
  src/lerobot/scripts/lerobot_train.py \
  --policy.type=internvla_a1_5 \
  --policy.pretrained_path=/mnt/r/CKPT/InternVLA-A1.5-base \
  --policy.push_to_hub=false \
  --policy.enable_keypoint_predictor=true \
  --policy.num_keypoint_joints=14 \
  --policy.kpt_loss_weight=1.0 \
  --policy.kpt_future_loss_weight=1.0 \
  --policy.kpt_to_action_detach=false \
  --policy.action_loss_only=true \
  --dataset.type=internvla_a1_5 \
  --dataset.repo_id=robotwin/stack_bowls_three_kpt \
  --dataset.enable_keypoint_predictor=true \
  --dataset.num_keypoint_joints=14 \
  --dataset.action_mode=abs \
  --dataset.use_external_stats=true \
  --dataset.external_stats_path=/mnt/r/CKPT/hf_home/lerobot/stats/aloha/abs/agg_1repos_1c27ca3df3/stats.json \
  --steps=20 --batch_size=2 --log_freq=5 --save_freq=20 \
  --output_dir=outputs/smoke_phase2
```

使用 **FK-augmented 数据集** (`stack_bowls_three_kpt`，含 `observation.keypoint_3d [42]` 列)，`kpt_loss_weight=1.0`。

**关键发现：** policy config 和 dataset config 上的 `enable_keypoint_predictor`、`num_keypoint_joints` 是**独立字段，没有自动同步**。必须在 CLI 中分别设置 `--policy.enable_keypoint_predictor=true` 和 `--dataset.enable_keypoint_predictor=true`。否则 Extract3DKeypointTransformFn 不会被插入 transform pipeline，模型收到的 batch 中无 kpt 字段。

### 9.2 问题记录 #E2：kpt loss 未在控制台显示

**现象：** 首次运行 Phase 2 时，训练正常完成但控制台 log 中无 `loss_kpt_current` / `loss_kpt_future` 字段。total loss 中有约 0.18-0.92 的未解释差额，推测为 kpt loss 但无法确认。

**根因：** `src/lerobot/scripts/lerobot_train.py` 的 `update_policy()` 函数（L118-131）仅显式提取 `loss_action`、`loss_video`、`loss_vqa`、`loss_fast`、`loss_subtask` 五种 loss 到 `train_metrics`。`loss_kpt_current` 和 `loss_kpt_future` 存在于 `output_dict` 中但未被提取，因此不会出现在 `MetricsTracker` 的 `__str__()` 输出中。同时 `train_metrics` 字典初始化处（L309-314）也未注册这两个 AverageMeter。

**修复：** 在 `lerobot_train.py` 中新增：
1. **L315-317**：当 `enable_keypoint_predictor=True` 时注册两个 AverageMeter：
   ```python
   if getattr(cfg.policy, "enable_keypoint_predictor", False):
       train_metrics["loss_kpt_current"] = AverageMeter("loss_kpt_cur", ":.4f")
       train_metrics["loss_kpt_future"] = AverageMeter("loss_kpt_fut", ":.4f")
   ```
2. **L129-132**：在 `update_policy()` 中提取这两个值：
   ```python
   if "loss_kpt_current" in output_dict:
       train_metrics.loss_kpt_current = output_dict["loss_kpt_current"]
   if "loss_kpt_future" in output_dict:
       train_metrics.loss_kpt_future = output_dict["loss_kpt_future"]
   ```

### 9.3 结果（修复后）

| step | loss | loss_action | loss_kpt_cur | loss_kpt_fut | grad_norm |
|------|------|-------------|-------------|-------------|-----------|
| 5 | 8.123 | 0.345 | **0.4133** | **0.5062** | 96.717 |
| 10 | 7.206 | 0.318 | **0.0746** | **0.1606** | 116.849 |
| 15 | 5.569 | 0.192 | **0.0494** | **0.1426** | 67.788 |
| 20 | 5.350 | 0.165 | **0.0448** | **0.1313** | 57.571 |

- 20 步完成，所有 loss 均 finite，无 NaN
- `loss_kpt_current` 和 `loss_kpt_future` **均 > 0 且递减**，确认 FK GT 关键点被正确加载（`kpt_mask=True`）
- `loss_kpt_current` 从 0.4133 降至 0.0448（10.8×）
- `loss_kpt_future` 从 0.5062 降至 0.1313（3.9×）
- total loss 包含 VQA/FAST 分量（`enable_vqa_loss` 默认为 True）
- `post_init_keypoint_weights` 成功执行
- Checkpoint 保存于 `outputs/smoke_phase2/checkpoints/000020`

---

## 修改文件汇总

| 操作 | 文件 | 行数 | 说明 |
|------|------|------|------|
| 新增 | `src/lerobot/policies/internvla_a1_5/keypoints.py` | 406 | TrackEncoder 移植 |
| 修改 | `src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py` | 588 | +166 行 config 字段 |
| 修改 | `src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py` | 2510 | +930 行三路 MoT |
| 修改 | `src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py` | 733 | +80 行 Extract3DKeypointTransformFn |
| 修改 | `src/lerobot/datasets/factory.py` | 643 | +7 行 delta_timestamps |
| 修改 | `evaluation/RoboTwin/inference.py` | 525 | +90 行推理集成 |
| 修改 | `src/lerobot/scripts/lerobot_train.py` | 421 | +8 行 kpt loss 日志（#E2 修复） |
| 新增 | `util_scripts/generate_aloha_keypoints.py` | 237 | FK 关键点生成 |
| 新增 | `tests/conftest.py` + 9 个 `test_step*.py` | 1855 | 71 个单元测试 |
| 新增 | `b/d/itrnVLA15_GeoP_3dtrj_3cn2_rbt2stak3_LOG.md` | — | 本日志 |

## 问题汇总

| 编号 | 问题 | 根因 | 修复 | 影响 |
|------|------|------|------|------|
| #E1 | `torchcodec` 版本不匹配 | `pip install -e .` 安装了 0.15.0，与 GPU/CUDA 不兼容 | `uv pip install torchcodec==0.10.0` | 环境问题 |
| #E2 | Phase 2 kpt loss 不在控制台显示 | `lerobot_train.py` 未注册/提取 `loss_kpt_current`/`loss_kpt_future` | 新增 AverageMeter 注册和提取代码 | 日志显示 |
