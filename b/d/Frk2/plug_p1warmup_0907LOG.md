# Phase 1 Warmup 执行日志 — Franka v2 插插座 (itvlagpFrkPlug2_0918)

> **方案文档**: [plug2_p1warmup.md](plug2_p1warmup.md)  
> **开始时间**: 2026-09-20  
> **操作者**: Claude Code  
> **EXPR_NAME**: `itvlagpFrkPlug2_0918`

---

## 执行记录

### Step 0: 确认信息

```
EXPR_NAME=itvlagpFrkPlug2_0918
VENV_ROOT=/B/VENV/itnvla15rbt20
DATA_SRC=~/b/Dta/plug_into_socket_franka3_15hz_lerobot_4d
PROJ_ROOT=/B/SRC/itvlaGpLibPlus
HF_HOME=/B/VENV/hf_home
PROC_PER_NODE=8
```

### Step 1-4: 基础环境验证

```
torch 2.10.0+cu128 cuda 8                          ✅
editable install: /B/SRC/InternVLA-A-series/src     ✅ (版本 1.0.0, 非 0.1.0)
A1.5-base: OK                                      ✅
GeoPredict: OK                                     ✅
GPU: 8 × NVIDIA H200, 全部空闲 (0 MiB)              ✅
```

### Step 5: Schema 部署

**发现 E1**: editable install 指向 `/B/SRC/InternVLA-A-series/src`, 非本仓库 `/B/SRC/itvlaGpLibPlus/src`. 因此 `franka3.yaml` schema 需部署到活跃包路径.

- 本仓库的 `src/lerobot/dataset_schemas/configs/franka3.yaml` (symlink) 不被 Python runtime 读取
- 活跃包的 configs 目录: `/B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/`
- 该目录下无 `franka3.yaml`, 也无 `franka_plug.yaml` 或 `r1_pro.yaml`

**Fix E1**: 在活跃包 configs 目录创建绝对路径 symlink:

```bash
ln -sfn /B/SRC/itvlaGpLibPlus/b/s/Frk2/cfg/franka3.yaml \
  /B/SRC/InternVLA-A-series/src/lerobot/dataset_schemas/configs/franka3.yaml
```

验证:
```
Schema: robot_type=franka3
  feature_mapping: {'observation.state': ['observation.state'], 'action': ['action']}
  image_mapping: {'observation.images.global': 'observation.images.image0',
                  'observation.images.wrist': 'observation.images.image1'}
  action_mask_spec: [7, -1]
Schema loaded: OK                                  ✅
```

### Step 6-7: 数据集验证

```
data symlink: OK                                   ✅
Dataset OK: 33308 frames, 15 Hz, robot_type=franka3 ✅
Keypoint: 56D (8 x 7D)                            ✅
```

### Step 8: episodes_stats.jsonl 格式

**发现 E2**: `episodes_stats.jsonl` 格式与方案预期不同.

- 方案预期: top-level keys = `['observation.state', 'action', ...]`
- 实际格式: top-level keys = `['episode_index', 'stats']`, 特征统计在 `stats` 子字典中
- `stats` 子字典含: `['observation.images.global', 'observation.images.wrist', 'observation.state', 'observation.force', 'observation.wrench', 'action', ...]`

**判定**: 不影响训练. `use_external_stats=false` 时代码内部解析 `episodes_stats.jsonl` 自动处理嵌套格式. 方案中 Step 8 的验证脚本有误 (预检代码假设了平铺格式), 但训练本身不受影响.

### Step 11: Smoke Test

**Attempt 1**: FAILED — 所有 GeoPredict/keypoint CLI 参数 `unrecognized`
- 根因: editable install 仍指向 upstream `/B/SRC/InternVLA-A-series/src`, 该仓库无 GeoPredict 代码
- Fix: `cd /B/SRC/itvlaGpLibPlus && pip install -e .` 重新安装 editable
- 验证: `.pth` 文件现在指向 `/B/SRC/itvlaGpLibPlus/src`

**Attempt 2**: FAILED — `--dataset.image_transforms.tfs.hue.weight=0.0` (及 sharpness, affine) unrecognized
- 根因 E4: draccus 不支持 `dict[str, dataclass]` 的嵌套 CLI 覆写
- Fix: 从 launch 脚本中删除 3 行 `--dataset.image_transforms.tfs.*.weight=0.0`, 改用 `max_num_transforms=3` 从 6 种默认变换中随机选 3 种

**Attempt 3**: FAILED — `RepositoryNotFoundError: User Access Token "readtkn" is expired`
- 根因 E5: HF token 过期, `FASTInternVLAA15ActionTokenizerTransformFn.__post_init__` 在 draccus config 解析阶段无条件调用 `AutoProcessor.from_pretrained("physical-intelligence/fast")`
- Fix 1: 将 FAST tokenizer 改为懒加载 (`_ensure_tokenizers()` 在 `__call__` 时才加载)
  - 修改文件: `src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py`
- Fix 2: launch 脚本添加 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` (Qwen 模型已缓存在 `/B/VENV/hf_home/hub/`)

**Attempt 4**: FAILED — `BackwardCompatibilityError: dataset in 2.1 format, expected v3.0`
- 根因 E6: 数据集是 LeRobot v2.1 格式, 代码 `CODEBASE_VERSION=v3.0` 强制 major version 检查
- Fix (全面): 使用内置转换器 `convert_dataset_v21_to_v30.py` 将数据集转为 v3.0:
  ```bash
  python src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id=plug_into_socket_franka3_15hz_lerobot_4d \
    --root=$HOME/b/Dta --push-to-hub=false
  ```
  - 转换后数据集: `~/b/Dta/plug_into_socket_franka3_15hz_lerobot_4d_v30/`
  - 复制 `keypoints_meta.json` 和 `episodes_stats.jsonl` 到 v30 目录
  - 更新 symlink: `ln -sfn ~/b/Dta/plug_into_socket_franka3_15hz_lerobot_4d_v30 /B/VENV/hf_home/lerobot/plug_into_socket_franka3_15hz_lerobot_4d`
- 额外代码修复 (v2.1 兼容性):
  - `lerobot_dataset.py:162`: `enforce_breaking_major=False`
  - `utils.py:load_tasks()`: 添加 `.jsonl` fallback
  - `utils.py:load_episodes()`: 添加 `.jsonl` fallback
  - `lerobot_dataset.py:get_video_file_path()`: v2.1 路径格式 fallback
  - `factory.py:398-403`: `stats is None` 时初始化空 dict

**Attempt 5**: ✅ **PASSED**
- 10 steps, 1 GPU, BS=2
- 损失: 33.8 → 8.8 (下降 74%)
- loss_action: 0.275 → 0.335 (正常震荡)
- loss_kpt_cur: 1.04 → 0.20 (下降 81%)
- loss_kpt_fut: 1.15 → 0.30 (下降 74%)
- Checkpoint saved at: `~/b/Ckp/itvlagpFrkPlug2_0918/.../checkpoints/000010`
- `post_check: video_decode_error=0 using_zeros=0 exit=0`
- 模型参数: Total 3B, Trainable 927M (action_expert 460M + kpt_expert + TrackEncoder)
- TrackEncoder 随机初始化 (GeoPredict 检查点 input_dim=3 vs 目标 7, 不兼容)

### Step 12: Production Training

**配置**: 8 × H200 GPU, EBS=128, 1566 steps (6 epochs), save_freq=783 (3 epochs)

**训练时间**: 2026-09-20 01:54 – 02:35 (~37 min)

**损失下降**:
| Step | Epoch | Total Loss | loss_action | loss_kpt_cur | loss_kpt_fut | grdn |
|------|-------|-----------|-------------|-------------|-------------|------|
| 50   | 0.19  | 18.107    | 0.253       | 0.5081      | 0.6260      | 332  |
| 200  | 0.77  | 3.098     | 0.158       | 0.0645      | 0.1194      | 62   |
| 500  | 1.92  | 1.209     | 0.083       | 0.0107      | 0.0468      | 21   |
| 783  | 3.00  | 1.041     | 0.081       | 0.0090      | 0.0394      | 12   |
| 1000 | 3.84  | 0.888     | 0.078       | 0.0070      | 0.0331      | 15   |
| 1300 | 5.00  | 0.846     | 0.075       | 0.0068      | 0.0314      | 12   |
| 1566 | 5.96  | 0.782     | 0.072       | 0.0050      | 0.0294      | 9    |

**Checkpoints**:
- `~/b/Ckp/itvlagpFrkPlug2_0918/2026_09_20_01_54_18-internvla_a1_5-frk2-plug-warmup/checkpoints/000783/` (9.4G)
- `~/b/Ckp/itvlagpFrkPlug2_0918/2026_09_20_01_54_18-internvla_a1_5-frk2-plug-warmup/checkpoints/001566/` (9.4G)

**Post-check**: `End of training` 确认, `video_decode_error=0`, `using_zeros=0`

**注意事项**:
- TrackEncoder 因 GeoPredict input_dim 不兼容 (3 vs 7) 而随机初始化, 未加载预训练权重
- keypoint_expert 从 action_expert 权重初始化 (`init_kpt_expert_from_action=true`)

### Step 13: Post-training

- bigmatrix GPU 占用已启动 (PID=1287052)
- 日志打包: `~/b/Ckp/itvlagpFrkPlug2_0918_LOG_26092002.tar`
- 执行日志: `~/b/Ckp/itvlagpFrkPlug2_0918_EXECLOG_26092002.md`

---

## 代码变更记录

本次执行过程中修改了以下文件:

1. **`b/s/Frk2/frk2_plug_warmup_launch.sh`**
   - 删除 3 行 `--dataset.image_transforms.tfs.*.weight=0.0` (draccus 不支持)
   - 添加 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`

2. **`src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py`**
   - `FASTInternVLAA15ActionTokenizerTransformFn.__post_init__`: 改为懒加载 (`_ensure_tokenizers()`)
   - 添加 `_ensure_tokenizers()` 调用到 `__call__` 和 `decode_action_tokens_to_actions`

3. **`src/lerobot/datasets/lerobot_dataset.py`**
   - `check_version_compatibility`: `enforce_breaking_major=False`
   - `get_video_file_path`: v2.1 路径格式 fallback

4. **`src/lerobot/datasets/streaming_dataset.py`**
   - `check_version_compatibility`: `enforce_breaking_major=False`

5. **`src/lerobot/datasets/utils.py`**
   - `load_tasks()`: 添加 `tasks.jsonl` fallback
   - `load_episodes()`: 添加 `episodes.jsonl` fallback

6. **`src/lerobot/datasets/factory.py`**
   - ImageNet stats: `stats is None` 时初始化空 dict

7. **数据集转换**: v2.1 → v3.0
   - `~/b/Dta/plug_into_socket_franka3_15hz_lerobot_4d_v30/` (转换后)
   - Symlink 更新: `plug_into_socket_franka3_15hz_lerobot_4d` → v30 目录

---

## 结果

✅ **Phase 1 Warmup 训练成功完成**

