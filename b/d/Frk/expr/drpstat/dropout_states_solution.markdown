# State Dropout 方案分析 — InternVLA-A1.5 随机丢弃/扰动状态输入

> **日期**: 2026-09-23
> **目的**: 分析如何在 InternVLA-A1.5 的三条 state 输入通路上实施随机 dropout, 以缓解模型对状态的过拟合 (尤其是 gripper 维度的 causal confusion)
> **相关文档**: `sumry0919_4trn.markdown` (D1, D5), `4dwvla_opt_idr.markdown`, `ds2_pblm_solv1.markdown` (C2)

---

## 1. 问题根因

### 1.1 Gripper 维度的同帧恒等式 (causal confusion)

`sumry0919_4trn.markdown` D1 通过数据分析发现了一个严重的 shortcut:

$$
a_t^{\text{gripper}} \approx 1 - \frac{w_t}{0.08}
$$

其中 $w_t$ 是当前帧的 gripper 宽度 (state 第 8 维, 即 index=7). 归一化后:

$$
z_{a}^{\text{gripper}} \equiv -z_{w}^{\text{gripper}} \quad (\text{max error} = 8.94 \times 10^{-8})
$$

这意味着模型可以通过**直接复制当前帧的 state gripper 维度, 取反**, 就得到 gripper 的 action — 无需理解图像或任务语义. 这是一种经典的 causal confusion.

### 1.2 关节状态的冗余性

`4dwvla_opt_idr.markdown` 指出 `state[0:7]` 是关节角, `state[8:15]`(如果是 15D 数据集) 是通过正向运动学 (FK) 从关节角计算出来的末端执行器位姿 — 是关节角的**确定性函数**, 没有独立信息, 反而增加了模型对 state 的依赖.

### 1.3 Keypoint 历史的自我强化

`sumry0919_4trn.markdown` D5 发现 keypoint 历史 (`his_kpts`) 实质上是关节角的确定性函数 (通过 FK), 且 `his_len` 标量被模型当作阶段指示器: 示教数据中 `his_len >= 120` 才出现闭合动作, 模型可能把 `his_len` 当成了"时钟".

### 1.4 核心矛盾

State 作为模型输入在正常情况下是有用的 (提供本体感受, 闭环控制), 但在当前数据集中:
- **Gripper state 与 gripper action 高度共线**, 形成了不需要视觉/语义的 shortcut
- **关节 state 与 keypoint 历史高度冗余**, 加剧了模型对 state 的依赖
- **推理时**环境噪声和执行延迟使 state 分布偏移, 导致上述 shortcut 失效, 模型行为崩溃

---

## 2. 模型中 State 的三条输入通路

InternVLA-A1.5 中 state (`observation.state` = `[arm(7D) | gripper(1D)]`, padded to 32D) 通过三条独立通路进入模型:

### 2.1 通路 A: 离散文本 tokenized state → VLM

**代码路径**: `transform_internvla_a1_5.py:110-135`

```
_encode_state(data):
    state = pad_vector(data["observation.state"], 32)
    state_np = state / 3                             # 缩放
    discretized = digitize(state_np, bins=linspace(-1,1,257)) - 1   # 量化到 256 档
    return "State: " + " ".join(map(str, discretized))
```

生成的文本 (如 `"State: 112 128 135 ... 200 128 128 ... 128"`) 被拼接到 user prompt:

```
"Task: pick up the plug; Control Mode: <abs>; State: 112 128 135 ..."
```

然后由 Qwen3.5 tokenizer 编码为 token ids, 进入 VLM prefix.

**特点**:
- 信息经过量化 (256档) 后精度较低, 但仍保留了 gripper 的开合状态
- 作为 VLM prefix 的一部分, 影响所有后续 cross-attention (action expert, kpt expert 都 attend prefix)
- 当 `tokenize_state=True` (当前默认) 时, 通路 B (连续 state → action expert) 被**关闭**

### 2.2 通路 B: 连续 state → Action Expert (条件性)

**代码路径**: `modeling_internvla_a1_5.py:997-998, 1512-1527`

```python
# 仅在 tokenize_state=False 时创建:
self.state_proj = nn.Linear(max_state_dim, action_expert_hidden_size)  # 32→1024

# embed_suffix() 中:
if not self.config.tokenize_state:
    state_emb = self.state_proj(state)                     # [B, 32] → [B, 1024]
    embs.append(state_emb[:, None, :])                     # 作为 suffix 的第一个 token
```

**特点**:
- 当 `tokenize_state=True` 时, 此通路**不存在** (`state_proj` 不被创建)
- 当前所有训练配置都用 `tokenize_state=True`, 所以此通路实际上是 inactive
- 如果需要同时启用连续 state 和文本 state, 需要修改代码

### 2.3 通路 C: 连续 state → 4D Keypoint Expert

**代码路径**: `modeling_internvla_a1_5.py:1018, 1590-1604`

```python
# 始终创建 (当 enable_keypoint_predictor=True):
self.kpt_state_proj = nn.Linear(max_state_dim, kpt_hidden_size)  # 32→1024

# embed_kpt_suffix() 中:
state_emb = self.kpt_state_proj(state)                    # [B, 32] → [B, 1024]
embs.append(state_emb[:, None, :])                        # kpt suffix 的第一个 token
```

**特点**:
- **始终活跃** (只要 `enable_keypoint_predictor=True`)
- 即使 `tokenize_state=True`, kpt expert 仍然接收**完整的连续 state 向量**
- 这意味着 gripper 维度的 shortcut 可以通过 kpt expert 的 state 输入绕过文本 state 的 dropout

### 2.4 三条通路的信息流总结

```
observation.state [B, 8] → pad_vector → [B, 32]
        │
        ├──→ [通路 A] _encode_state() → digitize → "State: 112 ..." → Qwen tokenizer
        │         → VLM prefix (所有 expert 的 cross-attention 前缀)
        │         ❗ 当前 tokenize_state=True, 此通路活跃
        │
        ├──→ [通路 B] state_proj → [B,1,1024] → action expert suffix 的 state token
        │         ❗ 当前 tokenize_state=True, 此通路关闭
        │
        └──→ [通路 C] kpt_state_proj → [B,1,1024] → kpt expert suffix 的 state token
                  ❗ 始终活跃 (enable_keypoint_predictor=True)
```

> **关键发现**: 即使在 `tokenize_state=True` 配置下, state 仍然通过**两条**路径进入模型 (通路 A + 通路 C). 任何 state dropout 方案如果只处理一条通路, shortcut 仍可通过另一条通路被利用.

---

## 3. 现有参考: 相关策略中的 State 处理

### 3.1 pi0 / pi05 中的处理

经审查, pi0 (`src/lerobot/policies/pi0/`) 和 pi05 (`src/lerobot/policies/pi05/`) **均未实现 state dropout**. pi0 的 `embed_suffix()` 中 state 以连续值直接投影, 无任何 masking:

```python
# pi0 modeling_pi0.py:685
state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
pad_masks.append(state_mask)
```

### 3.2 InternVLA-A1.5 论文

InternVLA-A1.5 论文中**未提及 state dropout**. 论文聚焦于 vision-action-video 的多模态融合, state 只是作为标准输入被简要提及.

### 3.3 学术界的 causal confusion 缓解方法

1. **Input Dropout / Masking**: 最直接的方法, 训练时随机将 state 向量置零. 出处: 参见 Octo (2024) 中对 proprioception 的可选 dropout, 以及 RT-2 系列中对 state conditioning 的讨论.
2. **Information Bottleneck**: 通过 VIB (Variational Information Bottleneck) 限制 state 编码的互信息.
3. **Domain Randomization**: 给 state 加噪声, 模拟推理时的偏移.
4. **Causal Masking**: 只允许 state 影响特定的 action 维度 (如 arm action 只看 arm state, gripper action 只看 gripper state).
5. **Gradient Penalty**: 对 $\frac{\partial a_{\text{gripper}}}{\partial s_{\text{gripper}}}$ 加正则化惩罚.

---

## 4. 方案设计

### 4.1 设计原则

1. **三条通路必须同步处理**: 只 drop 一条通路没有效果, 其它通路仍然泄露信息
2. **推理时不 drop**: dropout 只在训练时生效 (`self.training`)
3. **粒度可控**: 应支持 (a) 整个 state 全 drop, (b) 仅 drop gripper 维, (c) 按维度独立 drop
4. **向后兼容**: `state_dropout_prob=0.0` 时行为与当前完全一致
5. **扩展优先**: 通过配置控制, 不修改核心 forward 逻辑的结构

### 4.2 配置参数设计

```python
# configuration_internvla_a1_5.py 中新增:
state_dropout_prob: float = 0.0          # 整个 state 向量被置零的概率
state_dim_dropout_prob: float = 0.0      # 每个 state 维度独立被置零的概率
gripper_dropout_prob: float = 0.0        # gripper 维度 (index=7) 被单独置零的概率
state_noise_sigma: float = 0.0           # 给 state 添加高斯噪声的标准差
gripper_dim_index: int = 7               # gripper 维度的索引 (franka=7)
```

**为什么四个参数?**

| 参数 | 作用 | 风险 |
|:---|:---|:---|
| `state_dropout_prob` | 完全移除 state 信息, 迫使模型依赖视觉 | 太高会让模型完全忽略 state, 推理时无法闭环 |
| `state_dim_dropout_prob` | 每维独立 drop, 保留部分信息 | 比全 drop 温和, 但 gripper 维只占 1/32, 被 drop 的概率低 |
| `gripper_dropout_prob` | 精准打击 gripper shortcut | 最精准, 风险最低 |
| `state_noise_sigma` | 软扰动, 不完全抹除但降低精度 | 最温和, 但可能不足以打破高相关性 |

### 4.3 方案一: 最小侵入 — 仅 Gripper 维 Dropout (推荐)

**核心思路**: 仅对 gripper 维度 (index=7) 做高概率 dropout, 精准打击 `z_a ≡ -z_w` shortcut.

**实施点**: 在 `prepare_state()` 方法中, 对 state tensor 做 in-place masking. 这是**所有三条通路的公共入口**:

```python
# modeling_internvla_a1_5.py, InternVLAA15Policy.prepare_state()
# 注: 此方案已被 §5 的 transform-level 方案取代 (在 transform 中修改更优)
# 保留仅作为对比参考
def prepare_state(self, batch):
    state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)   # [B, 32]
    if self.training and self.config.gripper_dropout_prob > 0:
        mask = torch.rand(state.shape[0], device=state.device) < self.config.gripper_dropout_prob
        state[mask, self.config.gripper_dim_index] = self.state_mean[self.config.gripper_dim_index]
        # 注: 用数据集均值而非 0.0, 避免语义歧义和 OOD 风险 (见 §11)
    return state
```

**但这只覆盖通路 B 和 C**, 通路 A (tokenized state) 在 transform 阶段 (`_encode_state`) 已经完成, `prepare_state()` 的修改不影响它.

**因此需要同时在 transform 中做 dropout**:

```python
# transform_internvla_a1_5.py, _encode_state()
def _encode_state(self, data: DataDict) -> str:
    if not self.tokenize_state or OBS_STATE not in data:
        return ""
    state = deepcopy(data[OBS_STATE])
    # ── 训练时 gripper dropout ──
    if self.training and self.gripper_dropout_prob > 0:
        if random.random() < self.gripper_dropout_prob:
            state[self.gripper_dim_index] = 0.0
    state = pad_vector(state, self.max_state_dim)
    ...
```

**问题**: transform 是 data pipeline 的一部分, 不在 `model.train()` 控制下. 需要用一个标记或在 dataset 层传递训练模式.

**更好的方案**: 在 `_encode_state` 中**直接用概率**, 因为 transform 只在训练时被调用 (推理时用不同的 code path):

```python
def _encode_state(self, data: DataDict) -> str:
    if not self.tokenize_state or OBS_STATE not in data:
        return ""
    state = deepcopy(data[OBS_STATE])
    if self.gripper_dropout_prob > 0 and random.random() < self.gripper_dropout_prob:
        state[self.gripper_dim_index] = 0.0
    state = pad_vector(state, self.max_state_dim)
    ...
```

> 推理时 `_encode_state` 也会被调用, 但 `gripper_dropout_prob` 可以在推理 config 中设为 0.

### 4.4 方案二: 统一 State Masking — 在 prepare_state 中做, 同步到 transform

**核心思路**: 在 batch 层面做 state masking, 确保三条通路看到**同一个被 mask 后的 state**.

**实施方案**:

1. 在 `forward()` 入口处, 先对 batch 中的 `observation.state` 做 in-place masking
2. masking 后的 state 被 `prepare_state()` 用于通路 B/C
3. 通路 A (tokenized state) 已经在 transform 阶段完成, 无法在 forward 中修改

**因此, 只能在 forward 之前 (transform/dataset 阶段) 和 forward 入口各做一次**:

```
Transform 阶段 (data pipeline):
    _encode_state() → 决定文本中是否包含 state 信息 [通路 A]

Forward 阶段 (model):
    prepare_state() → 决定连续 state 是否被 mask [通路 B/C]
```

**同步问题**: 两阶段的 dropout 事件必须**协同** — 如果 tokenized state 被 drop, 连续 state 也应该被 drop, 否则模型会学到 "当文本中没有 state 时, 从 kpt expert 的连续 state 读取" 这种 fallback.

**同步方案**: 在 transform 阶段做一次 dropout 决策, 并将 mask 存入 batch:

```python
# transform_internvla_a1_5.py __call__:
state_dropped = False
if self.gripper_dropout_prob > 0 and random.random() < self.gripper_dropout_prob:
    state_dropped = True
    data[OBS_STATE][self.gripper_dim_index] = 0.0   # 修改源数据

# _encode_state 直接使用已修改的 data[OBS_STATE]
# prepare_state 也使用已修改的 data[OBS_STATE] (因为 batch 来自 dataset)
```

**关键洞察**: 如果在 transform 的 `__call__` 中**直接修改** `data[OBS_STATE]` 的 gripper 维, 那么后续所有使用 `data[OBS_STATE]` 的地方 (包括 `_encode_state`, 以及 forward 中的 `prepare_state`) 都会看到被 mask 后的值. 这实现了天然的三通路同步.

### 4.5 方案三: 全面 State Dropout (高强度)

**核心思路**: 以概率 $p$ 将**整个 state 向量**置零, 迫使模型学会仅依赖视觉和语言.

**风险**:
- State 包含关节角信息, 对精细操作 (如对准插座) 是有用的
- 完全 drop state 可能导致模型在推理时忽略 state, 无法实现闭环控制
- $p$ 需要仔细调节: 太高 → 模型忽略 state; 太低 → shortcut 仍然被利用

**建议**: 仅在极端过拟合场景下使用, $p \leq 0.2$.

### 4.6 方案四: 高斯噪声扰动 (最温和)

**核心思路**: 不完全 drop, 而是给 state 加高斯噪声:

$$
s_t' = s_t + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

**优点**: 保留了 state 的大致信息, 但降低了 shortcut 的精度.

**风险**: 如果 $\sigma$ 太小, 模型仍能从 $s_t'$ 精确恢复 gripper action (因为恒等式的误差本来就是 $10^{-8}$ 级别, 需要 $\sigma$ 足够大才能打破它).

**建议**: 对 gripper 维使用较大的 $\sigma_{\text{gripper}}$, 对 arm 维使用较小的 $\sigma_{\text{arm}}$.

---

## 5. 推荐方案: 方案二的简化实现

### 5.1 推荐理由

| 考量 | 方案一 | 方案二 | 方案三 | 方案四 |
|:---|:---|:---|:---|:---|
| 精准度 | 高 (仅 gripper) | 高 | 低 (全 drop) | 中 |
| 三通路同步 | 需双点修改 | **自然同步** | 同方案二 | 同方案二 |
| 风险 | 中 | **低** | 高 | 低 |
| 实现复杂度 | 中 | **低** | 低 | 低 |
| 破坏 arm 控制 | 否 | 否 | **可能** | 轻微 |

**推荐方案二 + 方案四 组合**: 在 transform 的 `__call__` 中修改 `data[OBS_STATE]`, 同时支持:
1. Gripper 维 dropout (高概率, 如 0.5)
2. 全维度高斯噪声 (小 sigma, 如 0.01)

### 5.2 实施点 (最少修改量)

只需修改 **2 个文件**:

#### 文件 1: `configuration_internvla_a1_5.py` — 新增配置

```python
# 新增字段:
state_dropout_prob: float = 0.0
gripper_dropout_prob: float = 0.0
state_noise_sigma: float = 0.0
gripper_dim_index: int = 7
```

#### 文件 2: `transform_internvla_a1_5.py` — 在 `__call__` 中修改 state

```python
def __call__(self, data: DataDict) -> DataDict:
    # ── State dropout (三通路同步, 训练推理皆通过此处) ──
    if OBS_STATE in data:
        state = data[OBS_STATE]
        need_clone = False
        if self.gripper_dropout_prob > 0 and random.random() < self.gripper_dropout_prob:
            state = state.clone()
            need_clone = True
            # 用数据集均值填充 (见 §11: 为何不用 0.0)
            state[self.gripper_dim_index] = self.state_mean[self.gripper_dim_index]
            data[OBS_STATE] = state
        if self.state_noise_sigma > 0:
            if not need_clone:
                state = state.clone()
            noise = torch.randn_like(state) * self.state_noise_sigma
            data[OBS_STATE] = state + noise

    state_str = self._encode_state(data)   # 使用已修改的 state
    ...
```

> **注意**: 推理时 `InternVLAA15ActionTransformFn.__call__` (另一个 class) 会被调用, 那里 `gripper_dropout_prob` 应为 0, 所以不影响推理.

但有一个问题: `InternVLAA15ActionTransformFn` 和 `InternVLAA15TrainTransformFn` 是**两个不同的类**吗?

检查:

- `InternVLAA15VLMTransformFn` — 训练用的 transform (在 `__call__` 中处理 state → tokens)
- `predict_action_chunk()` — 推理入口, 直接调用 `prepare_state(batch)`, 不走 transform

所以 **transform 中的 dropout 只影响训练**, 推理时 state 直接从 batch 进入 `prepare_state`. 但 kpt expert 的连续 state (通路 C) 在推理时也使用 `prepare_state` 的输出. 如果我们只在 transform 中做 dropout, 推理时通路 C 不受影响 — 这正是我们想要的.

但训练时的三通路同步需要确保:
- 通路 A: transform `_encode_state` 使用修改后的 state ✓ (在 `__call__` 开头修改 `data[OBS_STATE]`)
- 通路 C: `prepare_state(batch)` 使用 `batch["observation.state"]` — **这是来自 transform 修改后的 data 吗?**

**关键问题**: transform 修改 `data[OBS_STATE]` 后, batch collation 时 `batch["observation.state"]` 是否会使用这个修改后的值?

答案: **是的**. transform 的输出 data dict 包含 `observation.state`, collate 后成为 `batch["observation.state"]`, 进入 `prepare_state()`. 所以在 transform 中修改 `data[OBS_STATE]` 可以自然同步到通路 C.

### 5.3 完整实施步骤

```
Step 1: configuration_internvla_a1_5.py
    + gripper_dropout_prob: float = 0.0
    + state_noise_sigma: float = 0.0
    + gripper_dim_index: int = 7
    + state_dropout_fill_mode: str = "mean"   # "mean" | "noise" | "zero" (见 §11)

Step 2: transform_internvla_a1_5.py (InternVLAA15VLMTransformFn)
    __init__: 从 config 读取 dropout 参数
    hydrate(): 从 dataset.meta.stats 加载 state_mean, state_std
    __call__: 在调用 _encode_state 之前, 对 data[OBS_STATE] 做 dropout/noise
              fill 值使用 state_mean (见 §11 分析)

Step 3: dataset config
    在 launch 脚本中添加 CLI 参数:
    --dataset.gripper_dropout_prob=0.5
    --dataset.state_noise_sigma=0.01
    --dataset.state_dropout_fill_mode=mean

Step 4: 无需修改 modeling_internvla_a1_5.py (!!!)
    因为 prepare_state() 读取的 batch["observation.state"]
    已经是 transform 修改过的值.
```

> **最大优势**: 不修改模型代码, 只修改 data transform 和 config, 风险极低.

### 5.4 需要注意的边界情况

1. **Tokenized state 的 dropout 一致性**: `_encode_state` 中 `state / 3` 后 digitize. 当 gripper 维被置零时, digitize 结果为 128 (中间值). 这与 "gripper 半开" 的编码相同. 如果担心这个语义混淆, 可以考虑用一个特殊 token (如 `"[MASK]"`) 代替, 但这增加了 tokenizer 修改的复杂度.

2. **Delta mode**: 在 `action_mode=delta` 时, action 是差分值, gripper shortcut 的形式可能不同. 但 `action_mask_spec: [7, -1]` 表示 gripper 维始终是绝对值 (不做差分), 所以 shortcut 仍然存在.

3. **Multi-dataset 训练**: 不同数据集的 gripper dim index 可能不同. `gripper_dim_index` 应作为 dataset-level 配置而非 policy-level.

4. **FAST action tokens**: `use_fast_action_tokens=True` 时, FAST tokenizer 也生成 gripper action 的离散 token. State dropout 不影响 FAST loss 的计算, 但可以防止 FAST token 预测也走 state shortcut.

5. **Keypoint 历史截断**: `sumry0919_4trn.markdown` D5 建议随机截断 `his_len`. 这与 state dropout 是**互补的** — state dropout 打击同帧 shortcut, his_len 截断打击跨帧 shortcut. 可以在同一 transform 中一起实现.

---

## 6. 推荐超参数

| 场景 | `gripper_dropout_prob` | `state_noise_sigma` | 理由 |
|:---|:---|:---|:---|
| **保守方案** (先验证有效) | 0.3 | 0.0 | 30% 的概率 drop gripper 维, 足以打破恒等式, 但保留 70% 训练样本的闭环信号 |
| **标准方案** (推荐) | 0.5 | 0.01 | 50% drop gripper + 小噪声扰动其他维度 |
| **激进方案** (强过拟合) | 0.7 | 0.03 | 70% drop + 明显噪声, 可能牺牲一些精度 |

### 6.1 验收指标

根据 `sumry0919_4trn.markdown` D1 的建议:

1. **训练集上的 shortcut 检验**: 训练后, 在训练数据上回归 $a_t^{\text{gripper}} \sim s_t^{\text{gripper}}$, 要求 $R^2 < 0.9$ (当前 $R^2 \approx 1.0$)
2. **冻结宽度测试**: 将 gripper state 固定为 66.4mm (初始宽度), 连续喂 100 步, 策略应在若干步后仍输出 $a \geq 0.8$ (闭合命令). 当前 checkpoint 在此测试下必然失败.
3. **正常任务成功率**: 在 plug_into_socket 上的成功率不应因 state dropout 而显著下降 (容忍 ≤5% 下降).

---

## 7. 实施优先级

```
Priority 1 (高): gripper_dropout_prob=0.5, 在 transform 中实施
    → 精准打击 gripper shortcut, 不影响 arm 控制
    → 只修改 2 个文件 (config + transform), 不动 model code

Priority 2 (中): state_noise_sigma=0.01, 给全维度加小噪声
    → 温和扰动, 增强 state 的 domain robustness
    → 与 Priority 1 同时实施, 几乎零额外成本

Priority 3 (低): his_len 随机截断
    → 打击 keypoint history 的阶段指示器 shortcut
    → 需要修改 transform 中 keypoint 处理逻辑
    → 可在验证 P1+P2 有效后再加

Priority 4 (备选): 全维度 state_dropout_prob
    → 仅在 P1+P2 仍不足时使用
    → 风险: 可能影响 arm 精度
```

---

## 8. 与其他策略的协同

| 策略 | 与 state dropout 的关系 | 建议 |
|:---|:---|:---|
| 数据增强 (图像) | **互补**: 图像增强增加视觉多样性, state dropout 强制模型使用视觉 | 同时使用 |
| FAST action tokens | **兼容**: state dropout 不影响 FAST token 生成, 但防止 FAST 也走 shortcut | 同时使用 |
| 学习率调度 | **无冲突**: state dropout 不改变优化器行为 | 无需调整 |
| Knowledge insulation | **互补**: KI 限制 expert 对 prefix 的 attend, state dropout 限制 prefix 中的 state 信息 | 可叠加 |
| Checkpoint merge | **兼容**: 有/无 state dropout 的 checkpoint 权重 shape 完全相同, 可以 merge | 可作为后处理 |

---

## 9. 已有设计文档中的方案对比

### 9.1 ds2_pblm_solv1.markdown C2 节 — 最完整的已有设计

`b/d/Frk2/ds2_pblm_solv1.markdown` 第 953-1131 行已包含了一个**完整但未实现**的 state dropout 设计方案. 其要点:

**配置字段**:
```python
state_dropout_prob: float = 0.0
state_dropout_probs: list[float] | None = None   # 逐维概率
state_dropout_mode: str = "noise"                 # zero / noise / mean
state_dropout_noise_scale: float = 0.5
```

**实施点**:
- 路径 1 (`state_proj`): 在 `embed_suffix()` 中调用 `_apply_state_dropout()`
- 路径 2 (`tokenize_state`): 在 `_encode_state()` 中用 `data.get("_is_training", False)` 标记判断训练模式

**逐维概率示例 (franka3, 15D state)**:
- q1-q4: 0.0 (伺服滞后小)
- q5-q7: 0.2, 0.25, 0.3 (伺服滞后大)
- gripper: 0.0 (Frk2 场景)
- ee_pos/ee_quat: 0.4 (FK 冗余)

### 9.2 ds2 设计方案的局限性

| 维度 | ds2 方案 | 本分析的发现 |
|:---|:---|:---|
| 覆盖的通路数 | **2 条** (state_proj + tokenize_state) | 实际有 **3 条** (+ kpt_state_proj), ds2 未覆盖通路 C |
| 实施位置 | 两处: model code + transform code | 推荐 **1 处**: transform `__call__` (天然同步 3 条通路) |
| 训练模式检测 | 需要 `_is_training` 标记注入到 data dict | transform 只在训练 DataLoader 中运行, 推理用不同 code path, 天然安全 |
| gripper 维度处理 | Frk2 场景设 gripper=0.0 (不 drop) | 本场景 (Frk1) gripper 恒等式 R²≈1.0, **必须高概率 drop** |
| 逐维配置灵活性 | 高 (per-dim) | 高 (兼容, 但推荐先用简单的 gripper-only drop) |

### 9.3 pi05 的极端方案 — 完全移除 state

pi05 (`src/lerobot/policies/pi05/modeling_pi05.py`) **完全从 action expert 中移除了 state 输入**:
- `embed_suffix` 只接受 `(noisy_actions, timestep)`, 没有 `state` 参数
- 加载 pi0 checkpoint 时, `state_proj` 的 key 被显式跳过
- State 信息仅通过 VLM prefix 的文本 token 间接传递

这是 state dropout 的极端形式 (prob=1.0 for continuous path). 在 pi05 的设计中, 这被认为是足够的 — state 通过文本 token 仍然可以被 action expert 间接 attend.

**对比**: 本方案不建议采用 pi05 的极端做法, 因为:
1. InternVLA-A1.5 的 kpt expert 需要连续 state 输入 (通路 C 无法移除)
2. 精细操作 (如插座对准) 仍然需要精确的关节角信息做闭环

### 9.4 替代方案 — 离线数据修正

`ds2_pblm_solv1.markdown` 6.3 节提出了一种不修改模型代码的替代方案:

**离线 mask gripper state**: 用 `build_franka3_fixed_dataset.py --gripper-state-mode mask` 将数据集中 `state[7]` 替换为常数, 训练时模型看不到 gripper width 的变化.

**优点**: 零代码修改, 完全可逆 (不同的数据集版本)
**缺点**: 需要重建数据集; 推理时 gripper state 仍然会被模型接收到, 但因为训练时总是看到常数, 模型会学会忽略它 — 这可能导致推理时 state 和训练的分布更不一致.

**对比**: 在线 state dropout (本文推荐) 比离线 mask 更灵活 (概率可调, 支持噪声模式), 且推理时 state 正常传入, 不存在分布不一致问题.

---

## 10. 思考过程总结

1. **问题定位**: 从 `sumry0919_4trn.markdown` 的数据分析出发, 确认 gripper state → gripper action 的 causal confusion 是核心问题, 而非一般性的 state 过拟合.

2. **通路分析**: 追踪代码发现 state 有三条输入通路, 且**通路 C (kpt expert 的连续 state) 在 tokenize_state=True 时仍然活跃** — 这是之前分析文档中未充分强调的点. 任何方案必须同步处理所有活跃通路.

3. **注入点选择**: 在 transform 的 `__call__` 中修改 `data[OBS_STATE]` 是唯一能同时影响通路 A 和通路 C 的**公共注入点**. 这避免了在 model code 中做修改, 大幅降低风险.

4. **精准 vs 粗放**: 与其 drop 整个 state (影响 arm 控制精度), 不如精准 drop gripper 维 — 因为问题的根因就是 gripper 维的恒等式, 不是 arm state 的问题.

5. **噪声作为补充**: 高斯噪声单独使用可能不足以打破恒等式 (需要 $\sigma$ 远大于 $10^{-8}$ 的误差, 但又不能太大以至于破坏 arm 信息). 作为 gripper dropout 的补充是合理的.

---

## 11. Dropout 填充值分析: 被丢弃维度应填什么值?

### 11.1 问题提出

前面各节推荐在 transform `__call__` 中将被 dropout 的 state 维度"置零", 即 `state[dim] = 0.0`. 但这里有一个被忽视的重要问题: **0.0 在所有 state 维度中都有明确的物理含义**:

| 维度 | 0.0 的物理含义 | 与数据分布的关系 |
|:---|:---|:---|
| gripper (index=7) | 夹爪完全闭合 (width = 0mm) | = 分布最小值 `min=0.0` |
| arm joint 0 | 关节角 0 rad (特定位姿) | 在分布范围 [-0.48, +0.05] 内, 接近上界 |
| arm joint 3 | 关节角 0 rad | **远离**分布范围 [-2.20, -1.53], 是 OOD 值! |
| arm joint 5 | 关节角 0 rad | **远离**分布范围 [1.57, 2.45], 是 OOD 值! |
| arm joint 6 | 关节角 0 rad | **远离**分布范围 [0.48, 0.98], 是 OOD 值! |
| padded dims (8-31) | padding | 恒为 0.0 (合理) |

**核心风险**: 如果用 0.0 作为 dropout 填充值:
- **Gripper**: 0.0 = 完全闭合. 模型会学到"当 gripper bin=128 时, 夹爪是闭合的", 而不是"这个维度被 mask 了". 在推理时如果 gripper 真的处于闭合状态, 模型无法区分是真实状态还是 dropout, 产生歧义.
- **Arm joints**: 对 joint 3/5/6 来说, 0.0 处于训练数据分布之外 (OOD). 线性投影 (`state_proj`, `kpt_state_proj`) 会产生从未见过的激活值, 可能导致不稳定的 hidden state.

### 11.2 各填充策略的详细分析

#### 策略 F1: 零值填充 (`fill = 0.0`)

```
实现: state[dropped_dim] = 0.0
tokenize 映射: 0.0 / 3 = 0.0  → digitize → bin 128 (中间 bin)
连续路径映射: state_proj(0.0) → 固定的 hidden vector
```

| 优点 | 风险 |
|:---|:---|
| 实现最简单 | **语义歧义**: gripper=0 = 闭合, 不是 "无信息" |
| 对 pad 维度 (8-31) 无影响 | **OOD**: joint 3/5/6 的 0.0 远离训练分布 |
| 在 mean-std 归一化体系中 0=均值 | InternVLA 用 IDENTITY 归一化, 0.0 ≠ 均值 |

**结论**: ❌ 不推荐. 对 un-normalized 的 raw state (InternVLA-A1.5 的默认配置), 零值既有物理含义又可能 OOD.

#### 策略 F2: 数据集均值填充 (`fill = stats.mean[dim]`)

```
实现: state[dropped_dim] = dataset_mean[dim]
gripper: fill = 0.0337 (约 42% 开合) → /3 = 0.0112 → bin 129
joint 3: fill = -2.0600           → /3 = -0.6867 → bin 40
joint 5: fill = 2.2011            → /3 = 0.7337  → bin 221
```

对 franka_plug 数据集, 各维度的统计均值 (来自 `stats.json`):

$$
\mu = [-0.2406, 0.1457, 0.1872, -2.0600, -0.0553, 2.2011, 0.6998, 0.0337]
$$

| 优点 | 风险 |
|:---|:---|
| 在训练分布的中心, 不会 OOD | 模型可学会识别均值 → 检测 dropout |
| 信息量最小 (最"平淡"的值) | 需要从 stats 文件读取均值 |
| 对所有维度一致安全 | 均值 bin 是固定的, 成为新的 sentinel |

**数学直觉**: 在信息论意义上, 已知某随机变量服从分布 $P$, 要选一个值使其**携带最少的关于真实值的信息**, 最优选择是 $\mathbb{E}[X]$ — 均值是最大熵分布的中心, 相当于"我不知道真值是什么, 给你一个最无偏的猜测".

**关于"模型能检测 dropout"**: 这其实**不是缺点**. State dropout 的目的不是隐藏 dropout 事件, 而是**移除真实状态信息**. 即使模型知道"这个维度被 drop 了", 它也无法从均值中提取到任何关于当前真实状态的信息. 这恰恰是我们想要的效果 — 迫使模型在 dropout 时依赖视觉和其它输入.

**结论**: ✅ 推荐. 安全、简单、信息论上最优.

#### 策略 F3: 随机采样填充 (`fill ~ N(mean, std²)`)

```
实现: state[dropped_dim] = mean[dim] + std[dim] * torch.randn(1)
gripper: fill ~ N(0.0337, 0.0324²) → 范围约 [-0.063, 0.130]
joint 3: fill ~ N(-2.06, 0.085²)   → 范围约 [-2.32, -1.80]
```

| 优点 | 风险 |
|:---|:---|
| 模型无法检测到 dropout (值看起来正常) | 比均值更复杂, 引入额外随机性 |
| 填充值在训练分布内 | gripper 的随机值可能 < 0 (需 clamp 到 [0, 0.08]) |
| 打破任何固定值的 sentinel 效应 | 随机值仍与真实 action **不相关**, 效果与均值相当 |

**与均值对比**: 从信息论角度看, N(mean, std²) 的采样值也不包含关于真实值的信息 (因为 fill 值与当前帧无关). 但它引入了额外的训练噪声 (每次 dropout 看到不同的 fill 值), 这会**轻微增加 loss 的方差**, 可能需要适当降低学习率.

**对 kpt_state_proj 的影响**: 线性投影 `kpt_state_proj` 的输入是 32D 向量. 如果某个维度的 fill 值每次都不同 (N(μ, σ²)), 投影输出会有额外方差, 但因为投影权重会学会对该维度降权, 长期影响可控.

**结论**: ✅ 可选. 比均值更"隐蔽", 但额外复杂度换来的收益有限. 如果担心模型对均值常量的 sentinel 效应, 可以使用此策略.

#### 策略 F4: 特殊 sentinel 值 (OOD 常量)

```
方案 a: fill = -999.0 (极端 OOD)
方案 b: fill = NaN → 在 tokenize 中处理为特殊 token
方案 c: fill = -3.0  → /3 = -1.0 → bin 0 (最小 bin)
```

| 优点 | 风险 |
|:---|:---|
| 明确标记 "这是 dropout" | **严重 OOD**: 线性投影从未见过如此极端的输入值 |
| 无语义歧义 | NaN 导致梯度 NaN, 需要特殊 guard |
| — | -999 经过 state_proj 后产生极大激活, 可能导致训练 diverge |

**为什么不可行**: `state_proj` 和 `kpt_state_proj` 是**普通的 `nn.Linear`**, 没有 clamp/normalization. 输入 -999 会产生:

$$
\text{hidden} = W \cdot (-999) + b \approx -999 \cdot W_{\text{col}}
$$

这个值远超训练时见过的激活值范围, 会导致 attention logits 爆炸 (softmax 输入过大 → 数值不稳定).

对 tokenize 路径: `(-999) / 3 = -333` → digitize 到 bin=-1 → 需要 clamp 到 [0, 255], 变成 bin 0. bin 0 是有合法物理含义的 (state 值 = -3.0), 不是一个无歧义的 sentinel.

**结论**: ❌ 强烈不推荐. OOD 值在连续路径上有数值风险, 在离散路径上也不能无歧义地表示 dropout.

#### 策略 F5: Learnable Mask Embedding (学习一个 dropout 嵌入)

```
实现: 不修改 state 值, 而是在 state_proj 后替换 hidden state:
    if dropped: state_emb = self.mask_embedding  # nn.Parameter, [1, hidden_size]
```

| 优点 | 风险 |
|:---|:---|
| 模型自主学习"无信息"的最优表达 | 需要修改 model code (embed_suffix, embed_kpt_suffix) |
| 无 OOD 风险 (embedding 在训练中被优化) | 无法同步通路 A (tokenized state) |
| BERT/MAE 等预训练方法的标准做法 | 两个 expert 需要各自一个 mask_embedding |

**可行性分析**: 对连续路径 (通路 B/C), learnable mask embedding 是优雅的方案. 但对通路 A (tokenized state), 文本 "State: 128 128 ..." 无法被一个 embedding 替换 — 除非我们:
1. 在 dropout 时完全省略 state 文本 (变成 `"Task: ...; Control Mode: <abs>"` 不含 State 字段)
2. 或用一个特殊 token (如 `"State: [MASK]"`) 替换

方案 (1) 更简单, 但改变了 prompt 长度和结构. 方案 (2) 需要 tokenizer 注册新 token.

**结论**: ⚠️ 技术可行但实现复杂. 需修改 model code + tokenizer. 可作为高级方案, 但不建议作为首选.

### 11.3 通路差异化分析: 填充值在三条通路中的效果

填充值经过三条通路会被不同方式处理, 效果截然不同:

#### 通路 A (tokenized → VLM):

```python
state[dim] → / 3 → digitize(linspace(-1,1,257)) → "State: ... {bin} ..."
```

**数值分析** (franka_plug 数据集):

| 填充值 | 计算过程 | 结果 bin | 含义 |
|:---|:---|:---|:---|
| 0.0 (gripper) | 0/3=0 → bin 128 | 128 | "约 42% 开合" (实际: 完全闭合) |
| 0.0337 (gripper mean) | 0.0337/3=0.0112 → bin 129 | 129 | 差异很小, 仅 1 bin |
| 0.0 (joint 3) | 0/3=0 → bin 128 | 128 | "关节在中间位置" |
| -2.06 (joint 3 mean) | -2.06/3=-0.687 → bin 40 | 40 | 正常分布位置 |

**关键发现**: 对 gripper 维 (值域 [0, 0.08]), 零值和均值差异极小 (bin 128 vs 129, 仅差 1 bin). 对 arm joints (值域大), 零值和均值差异很大 (bin 128 vs bin 40 对 joint 3). 这意味着:
- **Gripper**: 零值填充和均值填充在 tokenize 路径上**几乎等效** (因为 gripper 的 raw 值本来就很接近 0)
- **Arm joints**: 零值填充可能产生**与真实分布截然不同的 bin**, 模型可能学到错误的 sentinel (bin 128 = dropout)

#### 通路 C (连续 → kpt expert):

```python
state[B, 32] → kpt_state_proj(Linear 32→1024) → [B, 1, 1024] hidden state
```

线性投影的输入值直接影响输出. 对于训练过的 `kpt_state_proj`:
- 输入均值处的值: 产生的 hidden state ≈ 投影权重列均值的加权和 → 偏向"无信息"
- 输入零值: 产生 $W \cdot 0 + b = b$ → 仅 bias, 如果 bias 不为 0 (通常初始化为 0 但训练后会变化), 这个 hidden state 也有特定含义
- 输入 OOD 值 (如 0.0 对 joint 3 来说): 产生模型训练时未充分覆盖的 hidden state → 可能不稳定

### 11.4 推荐方案: 分维度均值填充 + 可选随机扰动

基于以上分析, 推荐以下填充策略:

```python
# ── 配置 (configuration_internvla_a1_5.py) ──
state_dropout_fill_mode: str = "mean"   # "mean" | "noise" | "zero"
# "mean":  用 stats.json 中各维度的均值填充
# "noise": 用 N(mean, std²) 采样填充
# "zero":  用 0.0 填充 (仅适用于 normalized 数据)
```

```python
# ── 实现 (transform_internvla_a1_5.py __call__) ──
def __call__(self, data: DataDict) -> DataDict:
    if OBS_STATE in data:
        state = data[OBS_STATE]
        need_clone = False

        # ── Gripper dropout ──
        if self.gripper_dropout_prob > 0 and random.random() < self.gripper_dropout_prob:
            if not need_clone:
                state = state.clone()
                need_clone = True
            if self.state_dropout_fill_mode == "mean":
                state[self.gripper_dim_index] = self.state_mean[self.gripper_dim_index]
            elif self.state_dropout_fill_mode == "noise":
                state[self.gripper_dim_index] = (
                    self.state_mean[self.gripper_dim_index]
                    + self.state_std[self.gripper_dim_index] * torch.randn(1).item()
                )
            else:  # "zero"
                state[self.gripper_dim_index] = 0.0

        # ── 全维度噪声 (可与 dropout 叠加) ──
        if self.state_noise_sigma > 0:
            if not need_clone:
                state = state.clone()
                need_clone = True
            noise = torch.randn_like(state) * self.state_noise_sigma
            state = state + noise

        if need_clone:
            data[OBS_STATE] = state

    state_str = self._encode_state(data)
    ...
```

**关键实现细节**:
1. `self.state_mean` 和 `self.state_std` 来自 `stats.json`, 在 transform `hydrate()` 时加载
2. 对 gripper 维, 均值填充 ≈ 0.0337, 在 tokenize 中映射到 bin 129 (与 bin 128 相邻), 对模型影响温和
3. 对 arm 维 (若后续需要 arm dim dropout), 均值填充确保 bin 在正常范围内, 不会 OOD
4. `state_dropout_fill_mode` 作为配置项, 支持切换和消融实验

### 11.5 Gripper 维度的特殊考量

Gripper 维度有独特性质需要额外讨论:

**franka_plug 数据集 gripper 分布**:
- min = 0.0 (完全闭合), max = 0.0794 (完全打开)
- mean = 0.0337, std = 0.0324
- 分布近似双峰 (大部分时间处于全开 0.08 或全闭 0.0, 过渡很快)

**均值 0.0337 的含义**: "夹爪约 42% 开合", 这在真实轨迹中是**罕见的瞬态** (抓/放的过渡态). 模型在训练集中很少看到 gripper ≈ 0.034 的状态, 所以均值反而是一种自然的"异常"值, 模型不太可能通过它学到有用的 shortcut.

**对比 zero (0.0)**: 在 franka_plug 中, gripper=0 意味着"已经闭合" → action 应该是"继续保持闭合 (≈0)" 或 "开始打开 (≈1)". 如果大量样本被 fill 为 gripper=0, 模型可能学到新的偏见: "当 gripper=0 时总是保持闭合" → 推理时遇到真正的 gripper=0 (物理闭合) 时行为正确, 但这是一种**新的 shortcut**, 只是碰巧方向正确.

**均值 vs 零对 gripper shortcut 的破坏效果**:

原始 shortcut: $a_t^{\text{gripper}} \approx 1 - w_t/0.08$

- fill=0.0: $a_{\text{fill}} \approx 1 - 0/0.08 = 1.0$ → 模型看到 "gripper=0 → action=1" (开始打开)
  - 这与真实闭合状态的 action 恰好一致, **未打破 shortcut 模式**
- fill=mean(0.034): $a_{\text{fill}} \approx 1 - 0.034/0.08 = 0.575$ → 模型看到 "gripper=0.034 → action=0.575"
  - 这是一个**随机的 fill→action 对**, 与真实数据不完全一致, **部分打破 shortcut**

但更关键的是: dropout 的目的不是让 fill 值和 action 不匹配 (那只是增加 loss), 而是让模型在**看不到真实 gripper state 时仍能从图像中判断 gripper action**. 从这个角度看, fill 值是什么并不重要 — 重要的是**不是真实值**.

$$
\text{fill} \neq w_t \implies \text{shortcut } a_t = f(w_t) \text{ 无法在 drop 样本上被优化}
$$

只要 fill ≠ 真实值, shortcut 在该样本上的 loss gradient 就不会强化 shortcut.

### 11.6 实施建议 (更新 §5.2 的推荐)

综合上述分析, 更新推荐配置:

```python
# configuration_internvla_a1_5.py 新增:
gripper_dropout_prob: float = 0.0         # 推荐 0.5
state_noise_sigma: float = 0.0            # 推荐 0.01
gripper_dim_index: int = 7
state_dropout_fill_mode: str = "mean"     # 推荐 "mean"; 备选 "noise"
```

```python
# transform __init__ 中加载均值:
if config.state_dropout_fill_mode in ("mean", "noise"):
    stats = load_stats(config.stats_path)  # 从 stats.json
    self.state_mean = torch.tensor(stats["observation.state"]["mean"])  # [8]
    self.state_std = torch.tensor(stats["observation.state"]["std"])    # [8]
```

**对 §5 推荐方案的调整**:
- 原: `state[self.gripper_dim_index] = 0.0`
- **新**: `state[self.gripper_dim_index] = self.state_mean[self.gripper_dim_index]`
- 差异: gripper 的 fill 从 0.0 改为 0.0337 (raw), 在 tokenize 中 bin 从 128 改为 129
- 对 gripper 维: 差异极小, 但避免了 "fill=闭合" 的语义歧义
- 对 arm 维 (若后续启用): 差异很大 (bin 128 → bin 40 对 joint 3), 避免了 OOD

### 11.7 填充模式消融实验建议

| 实验 | `gripper_dropout_prob` | `fill_mode` | `state_noise_sigma` | 预期效果 |
|:---|:---|:---|:---|:---|
| Baseline | 0.0 | — | 0.0 | 对照组, R²≈1.0 |
| G1 | 0.5 | zero | 0.0 | 测试零值填充效果 |
| G2 | 0.5 | mean | 0.0 | 测试均值填充效果 |
| G3 | 0.5 | noise | 0.0 | 测试随机采样填充效果 |
| G2+N | 0.5 | mean | 0.01 | 均值填充 + 全维噪声 |

**验收指标** (同 §6.1):
1. 训练集 gripper shortcut $R^2 < 0.9$
2. 冻结宽度测试通过
3. 任务成功率下降 ≤ 5%

**预期**: G1/G2/G3 在 shortcut 打破效果上应**非常接近** (因为核心机制是"移除真实值", 不是"填什么替代值"). 差异主要体现在训练稳定性和边界情况:
- G1 可能在 arm dim dropout (若启用) 时不稳定 (OOD)
- G3 引入额外随机性, loss 方差略大
- G2 是最佳平衡点

### 11.8 小结

| 填充策略 | tokenize 安全 | 连续路径安全 | 实现复杂度 | 推荐度 |
|:---|:---|:---|:---|:---|
| F1: Zero | ⚠️ gripper 歧义 | ❌ arm OOD | ★ | ❌ |
| F2: Mean | ✅ | ✅ | ★★ | ✅ **推荐** |
| F3: N(μ,σ²) | ✅ | ✅ | ★★★ | ✅ 备选 |
| F4: Sentinel | ❌ OOD | ❌ 数值爆炸 | ★★ | ❌ |
| F5: Learnable | N/A (需改 prompt) | ✅ | ★★★★★ | ⚠️ 高级 |

**最终结论**: 使用**数据集均值** (`state_dropout_fill_mode="mean"`) 作为默认填充值. 它在信息论意义上最"无信息", 在所有维度上都 in-distribution, 实现简单 (只需读 stats.json), 且对三条通路都安全.

---

## 参考

**设计文档**:
- `b/d/Frk2/ds2_pblm_solv1.markdown` §C2 (L953-1131) — 最完整的 state dropout 设计方案 (未实现)
- `b/d/Frk2/realwrld_debug/sumry0919_4trn.markdown` D1 (gripper shortcut, L125-156), D5 (kpt 历史冗余, L265-291)
- `b/d/Frk2/realwrld_debug/4dwvla_opt_idr.markdown` — state dropout 初步提议, EE dropout 率建议
- `b/d/Frk2/realwrld_debug/grperr_1.1.markdown` §3.3 — state 进入模型的路径分析
- `b/d/Frk/dta_prblm_solv1.markdown` L315-482 — Frk1 数据集的 copy shortcut 量化分析

**代码**:
- `src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py` — 三条 state 通路: `state_proj` (L997-998), `embed_suffix` (L1512-1527), `kpt_state_proj` (L1018), `embed_kpt_suffix` (L1590-1604), `prepare_state` (L2271-2272)
- `src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py` — `_encode_state` (L110-117), `__call__` (L119-135)
- `src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py` — 配置 dataclass, 无现有 state dropout 参数
- `src/lerobot/policies/pi05/modeling_pi05.py` — state 完全移除的参考实现
- `b/s/Frk/cfg/franka_plug.yaml` — schema: `state = [arm(7D) | gripper(1D)]`, `action_mask_spec = [7, -1]`
