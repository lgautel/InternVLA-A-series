# InternVLA-A1.5 图像数据增强深度分析

> 本文档深入分析本代码库中图像数据增强的完整代码路径，重点回答：**如果启用 affine 数据增强，是否会对整个 video 的所有帧做同样的变换？开启 affine 是否会使操控机器人的 VLA 模型效果下降？图像增强怎么用比较好？**

---

## 一、数据增强的完整代码路径

### 1.1 调用链全景

数据增强发生在训练数据加载阶段，共涉及两层处理：

```
DataLoader → __getitem__() → 两层处理 → 模型
                │
                ├── 第一层: image_transforms (在 LeRobotDataset.__getitem__ 内)
                │     ↓  对每个 camera key 独立调用一次
                │     ↓  输入是 [T, C, H, W] 或 [C, H, W] 的张量
                │
                └── 第二层: transform pipeline (在 TransformedLeRobotDataset.__getitem__ 内)
                      ↓  ResizeImagesWithPadFn → RemapImageKeyTransformFn →
                      ↓  ExtractVideoFramesTransformFn → ... → ChatProcessor → ...
```

### 1.2 第一层: `image_transforms` 的调用位置

**`LeRobotDataset.__getitem__`** ([lerobot_dataset.py:1076-1079](src/lerobot/datasets/lerobot_dataset.py#L1076-L1079)):

```python
if self.image_transforms is not None:
    image_keys = self.meta.camera_keys
    for cam in image_keys:
        item[cam] = self.image_transforms(item[cam])
```

**`StreamingLeRobotDataset`** ([streaming_dataset.py:338-341](src/lerobot/datasets/streaming_dataset.py#L338-L341)):

```python
if self.image_transforms is not None:
    image_keys = self.meta.camera_keys
    for cam in image_keys:
        video_frames[cam] = self.image_transforms(video_frames[cam])
```

两处逻辑完全相同：遍历所有 camera key，对每个 key 的图像张量独立调用 `self.image_transforms`。

### 1.3 传入 `image_transforms` 的张量形状

由 `_query_videos` 返回 ([lerobot_dataset.py:1039](src/lerobot/datasets/lerobot_dataset.py#L1039)):

```python
item[vid_key] = frames.squeeze(0)
```

张量形状取决于 `image_delta_indices` 的长度：

- **InternVLA-A1.5 配置** ([configuration_internvla_a1_5.py:594-596](src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py#L594-L596)):
  ```python
  @property
  def image_delta_indices(self) -> list | None:
      n = self.num_video_frames + 1  # 4 + 1 = 5
      return [self.chunk_size * i // (n - 1) for i in range(n)]
      # = [0, 12, 25, 37, 49]
  ```
  返回 5 个索引，因此 `frames` 的形状是 **`[5, C, H, W]`**。`squeeze(0)` 不改变形状（第一维是 5，不是 1）。

- 因此传给 `image_transforms` 的是 **`[5, 3, 480, 640]`** 的 4D 张量（5 帧视频序列）。

### 1.4 `image_transforms` 的内部结构

**`ImageTransforms`** ([transforms.py:232-261](src/lerobot/datasets/transforms.py#L232-L261)):

```python
class ImageTransforms(Transform):
    def __init__(self, cfg: ImageTransformsConfig) -> None:
        ...
        disabled = set(cfg.disabled_tfs)
        for tf_name, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0 or tf_name in disabled:
                continue
            self.transforms[tf_name] = make_transform_from_config(tf_cfg)
            self.weights.append(tf_cfg.weight)

        n_subset = min(len(self.transforms), cfg.max_num_transforms)
        self.tf = RandomSubsetApply(
            transforms=list(self.transforms.values()),
            p=self.weights, n_subset=n_subset, random_order=cfg.random_order,
        )

    def forward(self, *inputs: Any) -> Any:
        return self.tf(*inputs)
```

**`RandomSubsetApply.forward`** ([transforms.py:74-87](src/lerobot/datasets/transforms.py#L74-L87)):

```python
def forward(self, *inputs: Any) -> Any:
    selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
    if not self.random_order:
        selected_indices = selected_indices.sort().values
    self.selected_transforms = [self.transforms[i] for i in selected_indices]
    for transform in self.selected_transforms:
        outputs = transform(*inputs)
        inputs = outputs if needs_unpacking else (outputs,)
    return outputs
```

每次 `forward` 调用：
1. 从 6 种可用变换（brightness, contrast, saturation, hue, sharpness, affine）中随机抽取 `max_num_transforms=3` 种。
2. 按定义顺序（`random_order=False`）依次施加到输入张量上。

### 1.5 默认配置的 6 种变换

| 名称 | 类型 | 参数 | 效果 |
|:-----|:-----|:-----|:-----|
| brightness | `v2.ColorJitter` | `brightness=(0.8, 1.2)` | 亮度缩放 ×0.8~1.2 |
| contrast | `v2.ColorJitter` | `contrast=(0.8, 1.2)` | 对比度缩放 ×0.8~1.2 |
| saturation | `v2.ColorJitter` | `saturation=(0.5, 1.5)` | 饱和度缩放 ×0.5~1.5 |
| hue | `v2.ColorJitter` | `hue=(-0.05, 0.05)` | 色调偏移 ±5% |
| sharpness | `SharpnessJitter` | `sharpness=(0.5, 1.5)` | 清晰度缩放 ×0.5~1.5 |
| **affine** | `v2.RandomAffine` | `degrees=(-5,5), translate=(0.05,0.05)` | **旋转 ±5°，平移 ±5%** |

前 5 种都是**像素值变换**（pixel-level），不改变任何像素的空间位置。affine 是唯一的**几何变换**（geometric），会旋转和平移像素。

代码位置: [transforms.py:183-216](src/lerobot/datasets/transforms.py#L183-L216)

---

## 二、核心问题一: 同一 video 的所有帧是否做相同的 affine 变换？

### 2.1 结论

**是的，同一次 `image_transforms()` 调用中的所有帧会接受完全相同的 affine 变换参数。**

### 2.2 证据: torchvision v2 Transform 对 4D 张量的处理机制

torchvision v2 的 `Transform` 基类对 `[T, C, H, W]` 形状的输入采用以下处理策略：

1. **`make_params()`**：调用**一次**，生成随机参数（如 `angle`、`translate`）。
2. **`transform()`**：对 4D 张量中的**每一帧**应用**相同的参数**。

这是 torchvision v2 专门为视频数据设计的行为——当输入是 `[T, C, H, W]` 时，它把第一个维度视为时间维度（video dimension），并确保时序一致性。

**实验验证**:

```python
# 5 个相同帧组成的 [5, C, H, W]，经过 RandomAffine 后
# → 5 个输出帧完全一致（bitwise equal）
video_same = frame.unsqueeze(0).expand(5, -1, -1, -1).clone()
torch.manual_seed(42)
out = tf(video_same)
all(torch.equal(out[0], out[i]) for i in range(5))  # → True

# 5 个不同帧，其中 frame 0 和 frame 3 内容相同
# → out[0] == out[3] 为 True，证明同一变换矩阵
video_diff = torch.stack([img1, img2, img3, img1, img2])
torch.manual_seed(42)
out = tf(video_diff)
torch.equal(out[0], out[3])  # → True
```

### 2.3 但: 不同 camera 之间的变换是**不同的**

关键代码是 `for cam in image_keys:` 循环 ([lerobot_dataset.py:1078-1079](src/lerobot/datasets/lerobot_dataset.py#L1078-L1079))。每次循环调用 `self.image_transforms(item[cam])` 都会重新执行 `RandomSubsetApply.forward()`，其中的 `torch.multinomial()` 和各变换内部的 `make_params()` 都会消耗新的随机数，因此：

- **`observation.images.global`（全局摄像头）** 和 **`observation.images.wrist`（腕部摄像头）** 会获得**不同的**随机变换参数。

**实验验证**:

```python
torch.manual_seed(42)
global_out = tf(global_frame)    # 消耗随机数 -> 得到 angle=A1, translate=T1
wrist_out  = tf(wrist_frame)     # 消耗新的随机数 -> 得到 angle=A2, translate=T2
# global_out 和 wrist_out 的 affine 参数不同
```

**影响**:

对于颜色变换（brightness/contrast/saturation/hue/sharpness），两个摄像头获得不同参数是**可接受的**——现实中两个摄像头的色温、曝光本来就可能不同，颜色不一致是合理的变异。

但对于 affine 变换，两个摄像头获得不同的旋转/平移意味着**多视角之间的几何一致性被破坏**。如果模型需要从全局和腕部两个视角做 3D 空间推理，这种不一致可能引入错误的空间关系。

### 2.4 总结: 一致性矩阵

| 维度 | 颜色变换 | affine 变换 |
|:-----|:---------|:-----------|
| 同一 camera 的不同帧 | 相同参数 | 相同参数 |
| 不同 camera 之间 | 不同参数 | 不同参数 |
| 不同 sample 之间 | 不同参数 | 不同参数 |

---

## 三、核心问题二: 开启 affine 数据增强是否会使 VLA 模型效果下降？

### 3.1 结论

**存在多个使效果下降的风险通道。以 `degrees=(-5, 5), translate=(0.05, 0.05)` 的默认参数来说，风险为"中等"：不一定会导致灾难性失败，但也不推荐开启。**

### 3.2 风险通道逐项分析

#### 风险 1: 视觉-动作对应关系错位（核心风险）

这是 affine 增强在机器人操控任务中最根本的问题。

**机器人视觉操控的核心假设**: 模型从图像中看到物体在位置 $p$，然后输出到达 $p$ 的关节动作 $a$。训练数据中的 $(I, a)$ 对是精确对齐的——图像 $I$ 中物体的像素位置与动作 $a$ 对应的真实空间位置一致。

**affine 破坏了这个对齐**: 旋转/平移图像后，物体在图像中的表观位置发生了变化，但动作 $a$ 保持不变（action target 不受图像增强影响）。这等价于告诉模型：

> "物体看起来在这个位置（旋转/偏移后的像素坐标），但你应该执行到达另一个位置（原始位置）的动作。"

这直接违反了视觉定位的可靠性前提。

**影响量化（当前参数）**:

- `degrees=(-5, 5)`: 对于 640×480 图像，图像角落的像素位移约为 $\sqrt{320^2+240^2} \times \sin(5°) \approx 35$ 像素。
- `translate=(0.05, 0.05)`: 水平最多 $640 \times 0.05 = 32$ 像素，垂直最多 $480 \times 0.05 = 24$ 像素。
- 综合：最极端情况下，物体在图像中可能偏移约 **50-60 像素**（图像宽度的 ~8%）。

对于需要毫米级精度的插头插入任务（plug_into_socket），这个量级的位置偏差是**显著的**。

#### 风险 2: 黑色填充引入 OOD 纹理

`v2.RandomAffine` 默认使用 `fill=0`（黑色）填充旋转/平移后露出的空白区域。

**实验测量**: 对 plug_into_socket_lrb_4D 的真实帧施加 affine 后，黑色填充区域占图像面积的 **3.5%~7%**。这些纯黑区域在真实环境中不会出现，构成 OOD（out-of-distribution）像素。

```
原始帧                    affine 后（旋转+平移）
┌────────────┐           ┌────────────┐
│            │           │ ██         │  ← 黑色填充（OOD）
│   scene    │    ──►    │   scene    │
│            │           │       ██   │  ← 黑色填充（OOD）
│            │           │         ██ │
└────────────┘           └────────────┘
```

**风险**: VLM 的视觉编码器可能学到"黑色角落 = 某种特定场景"的 spurious correlation，或者在推理时因为没有黑色角落而导致特征分布偏移。

#### 风险 3: WAN 视频预测分支受到影响

图像增强发生在 `__getitem__` 中（第一层），而 `ExtractVideoFramesTransformFn` 在 transform pipeline（第二层）中运行 ([transform_internvla_a1_5.py:667-679](src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py#L667-L679)):

```python
def __call__(self, data: DataDict) -> DataDict:
    src = data[self.source_view]          # 已经被 image_transforms 处理过
    if src.ndim == 4:  # [T, C, H, W]
        video = src
        if self.normalize_to_minus1_1:
            video = video * 2.0 - 1.0
        data[self.video_key] = video      # ← 增强后的帧送入 WAN 视频分支
        ...
        for i in range(3):
            k = f"{OBS_IMAGES}.image{i}"
            if k in data and data[k].ndim == 4:
                data[k] = data[k][0]      # ← frame[0] 送入 VLM
```

这意味着 **WAN 视频预测分支也会接收到 affine 增强后的视频帧**。WAN 的学习目标是"根据 VLM 的 learnable token 预测未来视频帧"——如果训练时的视频帧包含黑色填充和几何畸变，WAN 需要额外学习预测这些 artifact，可能降低其对真实场景的预测质量。

不过，由于同一 sample 的 5 帧使用相同的 affine 参数，WAN 至少不需要处理帧间几何不一致的问题。

#### 风险 4: 3D keypoint 预测路径不受影响（但形成不一致）

`Extract3DKeypointTransformFn` ([transform_internvla_a1_5.py:684-755](src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py#L684-L755)) 处理的是由正运动学（FK）离线计算的 3D 关节位置，**不涉及图像数据**。因此 affine 增强不影响 keypoint 的 ground truth。

但这恰恰形成了**另一种不一致**: 图像被旋转/平移了，但 keypoint target 保持不变。如果模型尝试从视觉特征推断 3D 空间位置来辅助 keypoint 预测，affine 增强会干扰这个推理过程。

### 3.3 affine 增强在机器人操控中的文献背景

机器人视觉操控社区对 affine 增强的共识：

- **颜色增强（color jitter）** 被广泛使用且效果正面。RT-2, Octo, OpenVLA, π₀ 等主流 VLA 均使用 brightness/contrast/saturation 增强来提升对光照变化的鲁棒性。
- **几何增强（crop/affine/rotation）** 需要极其谨慎：
  - **Random crop**（随机裁剪）比 affine 更常用，因为裁剪可以通过调整相机内参矩阵来精确补偿——理论上可以同步修改 action target。但本代码库未实现 action 的同步修正。
  - **Affine/rotation** 很少在精确操控任务中使用，因为很难同步修正 action target（需要知道相机外参才能计算旋转后的 action 补偿）。
  - InternVLA-A1.5 论文本身**未提及**使用图像增强（搜索 paper 全文未找到 "augment" 相关讨论）。

- 本代码库中的所有已有训练脚本都显式禁用了 affine:
  - `b/s/Frk/frk_plug_sft_dtaug_launch.sh`: `disabled_tfs=["affine"]`
  - `b/s/Frk2/frk2_plug_warmup_launch.sh`: `disabled_tfs=["affine"]`

---

## 四、核心问题三: 图像增强怎么用比较好？

### 4.1 推荐配置（当前代码已采用）

```bash
--dataset.image_transforms.enable=true
--dataset.image_transforms.max_num_transforms=3
--dataset.image_transforms.random_order=false
'--dataset.image_transforms.disabled_tfs=["affine"]'
```

**只启用 5 种颜色/清晰度变换，禁用 affine。**

理由：
- 颜色变换不改变像素空间位置 → 不破坏视觉-动作对应关系。
- 颜色变换模拟了现实中的光照变化（灯光亮度、白平衡、镜头模糊）→ 提升泛化能力。
- `max_num_transforms=3` 表示每次从 5 种中随机选 3 种施加 → 增强的多样性足够但不过度。

### 4.2 如果一定要用几何增强: 安全的替代方案

如果需要增强模型对视角变化的鲁棒性，以下方案比 affine 更安全：

#### 方案 A: Random Resized Crop（随机裁剪+缩放）

比 affine 安全的原因：裁剪等价于"相机向前/后移动 + 视场角变化"，不引入旋转。对于固定安装的全局摄像头，微小的视角变化是合理的变异。

```python
# 不在当前 transforms.py 中，需要新增
"crop": ImageTransformConfig(
    weight=1.0,
    type="RandomResizedCrop",
    kwargs={"size": (480, 640), "scale": (0.9, 1.0), "ratio": (4/3 - 0.05, 4/3 + 0.05)},
)
```

注意 `scale=(0.9, 1.0)` 表示最多裁掉 10% 的边缘，非常保守。

#### 方案 B: 如果要用 affine，大幅缩小参数

```python
"affine": ImageTransformConfig(
    weight=0.5,  # 降低被抽中的概率
    type="RandomAffine",
    kwargs={
        "degrees": (-1.0, 1.0),      # ±1° 而非 ±5°
        "translate": (0.01, 0.01),    # ±1% 而非 ±5%
    },
)
```

±1° / ±1% 的变化量约等于摄像头的安装抖动，对精确操控任务的干扰有限。

#### 方案 C: 对不同 camera 施加同步变换

当前代码的 `for cam in image_keys:` 循环让每个 camera 独立随机。如果需要在多个 camera 之间保持几何一致性，需要修改代码：

```python
# 修改 LeRobotDataset.__getitem__ (lerobot_dataset.py:1076-1079)
# 原始:
for cam in image_keys:
    item[cam] = self.image_transforms(item[cam])

# 改为: 所有 camera 共享同一组随机参数
# 思路: 在调用前设置相同的随机种子
if self.image_transforms is not None:
    image_keys = self.meta.camera_keys
    seed = torch.randint(0, 2**31, (1,)).item()
    for cam in image_keys:
        torch.manual_seed(seed)  # 每个 camera 使用相同的种子
        item[cam] = self.image_transforms(item[cam])
```

> **注意**: 这个修改仅对几何变换有意义。对于颜色变换，不同 camera 获得不同颜色参数反而更好（增加多样性）。因此更好的方案是将颜色变换和几何变换拆分为两个独立的 `ImageTransforms` 实例。

### 4.3 不同场景下的推荐策略

| 场景 | 颜色增强 | affine 增强 | 理由 |
|:-----|:---------|:-----------|:-----|
| **精确操控 fine-tuning**（如 plug_into_socket） | 启用 (5种) | **禁用** | 插入任务需要毫米级精度，affine 引入的偏移会破坏 action target 对应 |
| **粗略操控 fine-tuning**（如 pick & place） | 启用 (5种) | 可选（±1°/±1%） | 抓取任务对精度要求较低，微小几何变化可接受 |
| **大规模预训练** | 启用 (5种) | 可选（±2°/±2%） | 预训练需要更高多样性，且后续 fine-tuning 会修正偏差 |
| **sim-to-real** | 启用 + 更强参数 | 可选（±3°/±3%） | sim-to-real gap 本身就包含摄像头姿态不确定性 |

### 4.4 当前配置的实验验证

使用 [test_image_augmentation.py](b/d/Frk/asset/test_image_augmentation.py) 测试当前配置（禁用 affine, 仅 5 种颜色变换）对 `plug_into_socket_lrb_4D` 数据的效果：

| 指标 | 测试结果 | 阈值 | 判定 |
|:-----|:---------|:-----|:-----|
| PSNR（信噪比） | 19.6~48.0 dB | ≥ 18 dB | PASS |
| 最大像素差 | 62 (uint8 尺度) | ≤ 80 | PASS |
| 空间偏移 | ≤ 1.26 px | ≤ 2 px | PASS |

增强后的图像在颜色上有轻微变化（亮度/饱和度/清晰度），但像素的空间位置完全不变，视觉-动作对应关系保持完整。

测试输出（对比网格 PNG 和视频）在 `b/d/Frk/asset/logs/` 中。

---

## 五、补充分析: 颜色增强的帧间一致性

与 affine 类似，颜色增强也对 `[T, C, H, W]` 视频张量保持帧间一致性。

### 5.1 torchvision v2 ColorJitter 对 4D 张量的行为

```python
# ColorJitter 对 [5, 3, H, W] 的 5 帧视频
# → make_params() 调用一次，生成一组 brightness/contrast/saturation/hue 参数
# → transform() 对每帧用相同参数

video_same = frame.unsqueeze(0).expand(5, -1, -1, -1).clone()
torch.manual_seed(42)
out = v2.ColorJitter(brightness=(0.8, 1.2))(video_same)
all(torch.equal(out[0], out[i]) for i in range(5))  # → True
```

这保证了同一 sample 的 5 帧视频在颜色上的一致性——不会出现"第 1 帧偏暗、第 3 帧偏亮"的帧间闪烁。

### 5.2 SharpnessJitter 同理

自定义的 `SharpnessJitter` ([transforms.py:98-144](src/lerobot/datasets/transforms.py#L98-L144)) 继承了 torchvision v2 的 `Transform` 基类，遵循相同的 `make_params()` + `transform()` 协议：

```python
def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
    sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
    return {"sharpness_factor": sharpness_factor}

def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
    sharpness_factor = params["sharpness_factor"]
    return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)
```

`make_params()` 调用一次生成 `sharpness_factor`，`transform()` 对每帧用同一个 factor。

### 5.3 RandomSubsetApply 的变换选择一致性

`RandomSubsetApply.forward()` 在每次调用开始时执行一次 `torch.multinomial()` 来选择 3 种变换 ([transforms.py:77](src/lerobot/datasets/transforms.py#L77))。选出的变换**依次**作用于**整个** `[T, C, H, W]` 张量。因此：

- 变换种类的选择: **5 帧相同**（因为是一次调用内的一次随机抽样）。
- 每种变换的参数: **5 帧相同**（因为 torchvision v2 对 4D 张量只调用一次 `make_params()`）。
- 变换的施加顺序: **5 帧相同**（`random_order=False` 按定义顺序，否则按同一次排序结果）。

---

## 六、完整数据流图

```
┌─────────────────────────────────────────────────────────────────────┐
│ LeRobotDataset.__getitem__(idx)                                     │
│                                                                     │
│  _query_videos(query_timestamps, ep_idx)                            │
│    → global_frames: [5, 3, 480, 640]   (5帧 video, 原始分辨率)      │
│    → wrist_frames:  [5, 3, 480, 640]                                │
│                                                                     │
│  ┌─── for cam in [global, wrist]: ──────────────────────────────┐   │
│  │                                                               │   │
│  │  image_transforms(frames)                                     │   │
│  │    │                                                          │   │
│  │    ├── RandomSubsetApply.forward()                            │   │
│  │    │    torch.multinomial() → 选出 3 种变换 (独立随机)         │   │
│  │    │                                                          │   │
│  │    ├── 变换 1 (如 brightness):                                │   │
│  │    │    make_params() → brightness_factor = 1.15 (调用一次)    │   │
│  │    │    transform(frame_0, {bf=1.15})                         │   │
│  │    │    transform(frame_1, {bf=1.15}) ← 同一 params            │   │
│  │    │    transform(frame_2, {bf=1.15})                         │   │
│  │    │    transform(frame_3, {bf=1.15})                         │   │
│  │    │    transform(frame_4, {bf=1.15})                         │   │
│  │    │                                                          │   │
│  │    ├── 变换 2 (如 saturation): 同上,5帧同参数                  │   │
│  │    └── 变换 3 (如 sharpness): 同上,5帧同参数                   │   │
│  │                                                               │   │
│  │  结果: global_aug [5, 3, 480, 640] — 5帧同色调/亮度/清晰度     │   │
│  │        wrist_aug  [5, 3, 480, 640] — 独立的色调/亮度/清晰度     │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  返回 item = {                                                      │
│    "observation.images.global": global_aug,  # [5, 3, 480, 640]     │
│    "observation.images.wrist":  wrist_aug,   # [5, 3, 480, 640]     │
│    "action": ..., "observation.state": ...,                         │
│  }                                                                  │
└──────────────────────────────────────────────┬──────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TransformedLeRobotDataset._transform(sample)                        │
│                                                                     │
│  Pipeline 依次执行:                                                  │
│                                                                     │
│  1. DeltaActionTransformFn      (action -= state, 不涉及图像)        │
│  2. ResizeImagesWithPadFn       (480×640 → 224×224, pad黑边)         │
│  3. RemapImageKeyTransformFn    (key 重命名, 缺失 camera 填 1)       │
│  4. ExtractVideoFramesTransformFn                                    │
│     │  global_aug [5,3,224,224]                                      │
│     │    ├── video_frames = global_aug * 2 - 1  → [5,3,224,224]      │
│     │    │   ↑ 送入 WAN 视频预测分支 (已含增强效果)                    │
│     │    └── data["image0"] = global_aug[0]     → [3,224,224]         │
│     │        ↑ 只取 frame 0 送入 Qwen VLM                            │
│     │                                                                │
│  5. NormalizeTransformFn        (state/action 归一化, 不涉及图像)      │
│  6. ComposeFieldsTransform      (合并 sub-features)                  │
│  7. FASTActionTokenizer         (action 离散化)                      │
│  8. InternVLAA15ChatProcessor   (构建 VLM 输入: text + images)       │
│     │  images = [image0, image1]  ← 只有 frame 0, 已含增强效果       │
│     │                                                                │
│  9. PadStateAndAction           (pad 到 max_dim)                     │
│ 10. ReorderStateAction          (重排序)                             │
│ 11. UnifyInputs                 (最终格式化)                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 七、总结

### 7.1 回答三个核心问题

| 问题 | 回答 |
|:-----|:-----|
| 同一 video 的所有帧是否做同样的 affine 变换？ | **是。** torchvision v2 对 `[T, C, H, W]` 输入只调用一次 `make_params()` 生成随机参数，所有帧用相同参数。帧间时序一致性得到保持。 |
| 不同 camera 之间呢？ | **不是。** `for cam in image_keys:` 循环导致每个 camera 获得独立的随机参数。颜色变换方面这是可接受的多样性；几何变换方面这破坏了多视角几何一致性。 |
| 开启 affine 是否会使 VLA 模型效果下降？ | **存在显著风险。** 4 个风险通道：视觉-动作对应错位（核心）、黑色填充 OOD 纹理、WAN 视频分支受干扰、多视角几何一致性破坏。对于精确操控任务（如 plug_into_socket），不推荐开启。 |

### 7.2 推荐实践

1. **精确操控 fine-tuning**: 禁用 affine (`disabled_tfs=["affine"]`)，仅用 5 种颜色/清晰度变换。**当前配置已正确采用此策略。**
2. 如需几何增强: 使用 ±1°/±1% 的极小参数，并降低采样权重。
3. 多 camera 同步: 如需多 camera 之间的几何一致性，需修改代码共享随机种子（见 §4.3 方案 C）。

### 7.3 关键代码参考

| 功能 | 文件 | 行号 |
|:-----|:-----|:-----|
| 图像增强总入口 | [lerobot_dataset.py](src/lerobot/datasets/lerobot_dataset.py) | L1076-1079 |
| 流式数据集增强入口 | [streaming_dataset.py](src/lerobot/datasets/streaming_dataset.py) | L338-341 |
| RandomSubsetApply（随机选取变换） | [transforms.py](src/lerobot/datasets/transforms.py) | L29-95 |
| SharpnessJitter（自定义清晰度） | [transforms.py](src/lerobot/datasets/transforms.py) | L98-144 |
| ImageTransformsConfig（默认 6 种变换） | [transforms.py](src/lerobot/datasets/transforms.py) | L165-216 |
| ImageTransforms（构建增强管线） | [transforms.py](src/lerobot/datasets/transforms.py) | L232-261 |
| ExtractVideoFramesTransformFn | [transform_internvla_a1_5.py](src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py) | L654-679 |
| image_delta_indices（5帧配置） | [configuration_internvla_a1_5.py](src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py) | L594-596 |
| 训练脚本 (禁用 affine) | [frk_plug_sft_dtaug_launch.sh](b/s/Frk/frk_plug_sft_dtaug_launch.sh) | L241-244 |
