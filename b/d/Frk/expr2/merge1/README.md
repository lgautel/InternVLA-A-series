# Checkpoint Merge Tool

独立的checkpoint加权平均合并工具，用于InternVLA-A1.5模型。

## 功能

- **加权平均合并**: 支持2个或多个checkpoint的加权平均合并
- **格式兼容**: 支持 `.safetensors` 和 `.bin` 格式的权重文件
- **配置文件复制**: 自动复制config和tokenizer文件到输出目录
- **Smoke Test**: 验证合并后的checkpoint能被transformers和lerobot框架正确加载

## 使用方法

### 快速开始

**2-way合并 (两个checkpoint)**

合并脚本在系统Python环境下运行（不需要conda环境）：

```bash
cd /home/nvidia/bt/s/4WVLA/b/d/Frk/expr2/merge1
bash run_merge.sh
```

该脚本会合并以下两个checkpoints：
- `/home/nvidia/bt/ckp/4wvlaFrk/plug/itvlagpFrkPlug0907Dtaug010000/` (数据增强, 权重0.3)
- `/home/nvidia/bt/ckp/4wvlaFrk/plug/4wvlaFrkPlugCkp010420/` (无数据增强, 权重0.7)
- 保存到 `/home/nvidia/bt/ckp/4wvlaFrk/plug/aug1w03pure1w07/`

**3-way合并 (三个checkpoint)**

```bash
cd /home/nvidia/bt/s/4WVLA/b/d/Frk/expr2/merge1
bash run_merge_3way.sh
```

该脚本会合并以下三个checkpoints：
- `/home/nvidia/bt/ckp/4wvlaFrk/plug/itvlagpFrkPlug0907Dtaug010000/` (数据增强10k步, 权重0.1)
- `/home/nvidia/bt/ckp/4wvlaFrk/plug/itvlagpFrkPlug0907_dtaug2_5k/` (affine数据增强5k步, 权重0.2)
- `/home/nvidia/bt/ckp/4wvlaFrk/plug/4wvlaFrkPlugCkp010420/` (无数据增强, 权重0.7)
- 保存到 `/home/nvidia/bt/ckp/4wvlaFrk/plug/aug1w01augtw5k02pure1w07/`

**运行Smoke Test (需要conda环境)**

Smoke test需要在 `internvla_a1_5` conda环境中运行：

```bash
cd /home/nvidia/bt/s/4WVLA/b/d/Frk/expr2/merge1
# 2-way合并验证
bash run_smoke_test_in_env.sh /home/nvidia/bt/ckp/4wvlaFrk/plug/aug1w03pure1w07/

# 3-way合并验证
bash run_smoke_test_in_env.sh /home/nvidia/bt/ckp/4wvlaFrk/plug/aug1w01augtw5k02pure1w07/
```

或者手动激活环境后运行：

```bash
conda activate internvla_a1_5
python3 smoke_test.py --checkpoint /home/nvidia/bt/ckp/4wvlaFrk/plug/aug1w03pure1w07/
```

### 自定义合并

**2-way合并 (两个checkpoint)**

使用 `merge_checkpoints.py` 脚本：

```bash
python merge_checkpoints.py \
    --ckpt1 /path/to/checkpoint1 --weight1 0.3 \
    --ckpt2 /path/to/checkpoint2 --weight2 0.7 \
    --output /path/to/output
```

**参数说明**:
- `--ckpt1`: 第一个checkpoint路径
- `--weight1`: 第一个checkpoint的权重
- `--ckpt2`: 第二个checkpoint路径
- `--weight2`: 第二个checkpoint的权重 (两个权重必须相加等于1.0)
- `--output`: 输出checkpoint保存路径

**多路合并 (3个或更多checkpoint)**

使用 `merge_checkpoints_multi.py` 脚本：

```bash
python merge_checkpoints_multi.py \
    --ckpt /path/to/checkpoint1 --weight 0.2 \
    --ckpt /path/to/checkpoint2 --weight 0.3 \
    --ckpt /path/to/checkpoint3 --weight 0.5 \
    --output /path/to/output
```

**参数说明**:
- `--ckpt`: checkpoint路径 (可多次指定)
- `--weight`: 对应checkpoint的权重 (可多次指定，必须与`--ckpt`数量匹配)
- `--output`: 输出checkpoint保存路径
- 所有权重必须相加等于1.0

### 运行Smoke Test

单独验证合并后的checkpoint：

```bash
python smoke_test.py --checkpoint /path/to/merged/checkpoint
```

## Smoke Test验证内容

1. **Transformers Loading**: 验证checkpoint能被 `transformers.AutoModel` 和 `AutoConfig` 加载
2. **LeRobot Policy Loading**: 验证checkpoint能被LeRobot的 `make_policy_from_checkpoint` 加载
3. **Forward Pass**: 验证模型能正常进行前向传播

## 合并算法

对于多个checkpoint中的相同权重参数：

```python
merged_weight = sum(weight_i * checkpoint_i[key] for i in range(N))
```

- 只合并名称相同且shape一致的参数
- 仅存在于部分checkpoint的参数会按对应权重缩放后保留
- Shape不一致的参数会跳过并打印警告

**示例 (3-way merge)**:
```python
# 假设三个checkpoint权重分别为 0.1, 0.2, 0.7
merged["model.layer.weight"] = (
    0.1 * ckpt1["model.layer.weight"] +
    0.2 * ckpt2["model.layer.weight"] +
    0.7 * ckpt3["model.layer.weight"]
)
```

## 文件结构

```
merge1/
├── merge_checkpoints.py        # 2-way合并脚本
├── merge_checkpoints_multi.py  # 多路合并脚本(支持3个及以上)
├── smoke_test.py               # 验证测试脚本
├── run_merge.sh                # 2-way合并一键运行脚本
├── run_merge_3way.sh           # 3-way合并一键运行脚本
├── run_smoke_test_in_env.sh    # conda环境下运行smoke test
└── README.md                   # 本文档
```

## 依赖

**合并脚本** (`merge_checkpoints.py`, `merge_checkpoints_multi.py`):
- PyTorch
- safetensors

**Smoke Test** (`smoke_test.py`):
- PyTorch
- transformers
- safetensors
- LeRobot framework (在项目根目录 `src/lerobot/`)
- 需要 `internvla_a1_5` conda环境

## 环境说明

- **合并操作**: 可在系统Python环境下运行，只需要 `torch` 和 `safetensors`
- **Smoke Test**: 必须在 `internvla_a1_5` conda环境中运行，需要完整的InternVLA-A1.5依赖

## 注意事项

1. **内存需求**: 合并过程会同时加载所有checkpoint到CPU内存（每个约6GB），3-way合并需要约18GB+ RAM
2. **权重和必须为1.0**: 脚本会检查所有权重相加等于1.0，否则会报错
3. **配置兼容性**: 脚本假设所有checkpoint的config文件兼容，会从第一个checkpoint复制config到输出目录
4. **GPU可选**: 合并和smoke test都在CPU上运行，不需要GPU
5. **环境隔离**: 合并脚本设计为独立运行，不影响项目其他文件

## 示例输出

```
Loading from /home/nvidia/bt/ckp/.../model.safetensors
Loading from /home/nvidia/bt/ckp/.../model.safetensors

Checkpoint 1: 1234567 parameters
Checkpoint 2: 1234567 parameters
Common parameters: 1234567

Saving merged weights to /home/nvidia/bt/ckp/.../model.safetensors
Copying config.json

✓ Merged checkpoint saved to /home/nvidia/bt/ckp/...
  Total parameters: 1234567
```
