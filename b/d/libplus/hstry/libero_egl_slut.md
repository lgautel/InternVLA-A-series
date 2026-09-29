# LIBERO / LIBERO-plus：双/六模型 Server + 六 EGL Client 部署方案

> **验证日期**：2026-09-16  
> **目标**：比较两张 server GPU 上运行 2 个或 6 个模型 server、另外六张 GPU 只负责 LIBERO MuJoCo EGL 渲染时的吞吐与稳定性。  
> **结论**：在相同 24 个标准 LIBERO task、120 个 episode 的 A/B 冒烟中，六 server 的 client 阶段比双 server 快 28.1%，端到端快 20.3%；两者都无 SIGABRT、SIGSEGV、CRASHED 或缺失结果 JSON。六 server 已通过真实生产启动器短冒烟，但这仍是有限 workload 的证据，不能把 NVIDIA/MuJoCo EGL 的所有驱动级故障宣称为绝对根治。

## 1. 结论先行

**默认拓扑（`SERVER_INSTANCES_PER_GPU=3`，即六 server）**：每个 client 使用
自己独立的 server 端口；同一张模型 GPU 上的三个 server 共享模型 GPU，但不
共享 WebSocket 请求队列：

```text
GPU 0 ──▶ server-0 :5784 ◀── client GPU 2
       ├─▶ server-1 :5785 ◀── client GPU 3
       └─▶ server-2 :5786 ◀── client GPU 4

GPU 1 ──▶ server-3 :5787 ◀── client GPU 5
       ├─▶ server-4 :5788 ◀── client GPU 6
       └─▶ server-5 :5789 ◀── client GPU 7
```

**回退拓扑（显式设置 `SERVER_INSTANCES_PER_GPU=1`，即双 server）**：两个
client 共享一个 server 上的同步推理队列，显存/进程数更省：

```text
                    WebSocket
          ┌──────────────────────────┐
GPU 0 ───▶│ model server-0 :5784     │◀── client GPU 2
           │ InternVLA-A1.5           │◀── client GPU 3
           │ no MuJoCo / no EGL      │◀── client GPU 4
          └──────────────────────────┘

          ┌──────────────────────────┐
GPU 1 ───▶│ model server-1 :5785     │◀── client GPU 5
           │ InternVLA-A1.5           │◀── client GPU 6
           │ no MuJoCo / no EGL      │◀── client GPU 7
          └──────────────────────────┘
```

标准 LIBERO（无扰动）现在也有对等的六 server launcher：
[`evaluation/LIBERO2/run_eval_libero_std_2server_6client_venv.sh`](../../../evaluation/LIBERO2/run_eval_libero_std_2server_6client_venv.sh)，
默认值和端口/GPU 映射规则与下文完全一致，唯一区别是客户端调用
`evaluation/LIBERO2/eval_libero_std.py`（四个无扰动 suite）而不是
LIBERO-plus 的扰动客户端。完整、自包含的标准操作手册见
[`eval3_optim3.md`](eval3_optim3.md) §二十；本文件保留原始探索记录。

验证结果：

| 门禁 | 结果 |
|---|---|
| 两 server / 六 client 的 GPU 集合不重叠 | PASS |
| 六个真实 EGL client + 两个 mock server，5 episode × 20 步 | PASS，30/30 episode |
| 两个真实 InternVLA server + 六个 EGL client，LIBERO-plus 短冒烟 | PASS，无 SIGABRT/SIGSEGV |
| 两个真实 InternVLA server + 六个 EGL client，标准 LIBERO spatial | PASS，6 个 task × 5 episode = 30/30，6 个结果 JSON 全部写出 |
| 标准 LIBERO 多 episode 回归中的 SIGABRT/SIGSEGV/CRASHED | 0 |
| 双 server vs 六 server，24 task × 5 episode = 120 episode/拓扑 | 两者均 PASS；六 server client 阶段快 28.1% |
| 六 server 正式启动器 `EVAL_MODE=smoke` | PASS，6 client 全部完成，无崩溃或缺失结果 |

结果摘要持久化在：

- [`asset/libero_egl_split_smoke_results.json`](asset/libero_egl_split_smoke_results.json)
- [`asset/libero_egl_topology_ab_results.json`](asset/libero_egl_topology_ab_results.json)

## 2. 为什么这个方案可能有效

之前的典型部署把一张 GPU 同时用于：

1. server 侧的 CUDA/VLM 推理；
2. client 侧的 MuJoCo EGL framebuffer、OpenGL context 和纹理；
3. 多个 forked task child 的 EGL context 创建与销毁。

当 server 推理和 EGL 渲染共用 GPU 时，显存压力、驱动上下文调度和进程生命周期会耦合在一起。之前出现的“同一 child 连续完成约 4 个 episode 后 SIGABRT”不能证明一定由模型推理触发，但同 GPU 争用是合理的放大因素。

本方案将两类 GPU 工作物理隔离：

- GPU 0、1：只由模型 server 使用；
- GPU 2–7：只由 LIBERO client 的 EGL 使用；
- server 进程显式清除 `MUJOCO_GL`、`PYOPENGL_PLATFORM` 和 `MUJOCO_EGL_DEVICE_ID`；
- client 进程显式设置 `MUJOCO_GL=egl`，并将 `MUJOCO_EGL_DEVICE_ID` 指向自己的物理 GPU。

因此，即使 client child 的 EGL context 发生异常，也不会直接破坏模型 server 所在 GPU 的 CUDA 推理上下文。

但是，这仍然不是 EGL 的数学意义上的“绝对根治”：每个 client 内部仍然会创建和销毁 EGL context。如果根因是纯 MuJoCo/NVIDIA EGL 的进程内生命周期 bug，而不是同 GPU 争用，理论上 client GPU 上仍可能出现 SIGABRT。30 episode 的真实多 episode 回归通过，说明该部署显著降低了当前故障的实际风险，但不等价于无限长 soak 的证明。

## 3. GPU 与端口映射规则

令：

- \(N_c=6\)：EGL client 数量；
- \(N_s\)：模型 server instance 数量，双 server 时为 2，六 server 时为 6；
- \(r=\texttt{SERVER\_INSTANCES\_PER\_GPU}\)：每张 server GPU 上的 instance 数量，取 1 或 3；
- \(k=N_c/N_s\)：每个 server 绑定的 client 数量；
- \(i\)：client 在 `CLIENT_GPU_IDS` 中的下标，从 0 开始；
- \(g(i)=\lfloor i/k\rfloor\)：client 所属 server group；
- \(P\)：`BASE_PORT`。

则：

\[
k=\frac{N_c}{N_s},\qquad
g(i)=\left\lfloor\frac{i}{k}\right\rfloor,\qquad
\operatorname{port}(i)=P+g(i)
\]

其中 \(g(i)\) 是 server instance 的编号，不是物理 GPU 编号。server
instance 按 `SERVER_GPU_IDS` 顺序重复 \(r\) 次绑定到 GPU。例如
`BASE_PORT=5784` 时：

| Server group | Server GPU | Server port | Client GPU | `MUJOCO_EGL_DEVICE_ID` |
|---|---:|---:|---:|---:|
| 0 | 0 | 5784 | 2, 3, 4 | 2, 3, 4 |
| 1 | 1 | 5785 | 5, 6, 7 | 5, 6, 7 |

六 server 时，\(k=1\)，映射变为：

| Server instance | Server GPU | Server port | Client GPU |
|---|---:|---:|---:|
| 0, 1, 2 | 0 | 5784, 5785, 5786 | 2, 3, 4 |
| 3, 4, 5 | 1 | 5787, 5788, 5789 | 5, 6, 7 |

这里的 `MUJOCO_EGL_DEVICE_ID` 必须使用物理 EGL 枚举编号。当前安装的 MuJoCo EGL 实现通过 `eglQueryDevicesEXT()` 直接枚举设备，并不会因为 `CUDA_VISIBLE_DEVICES=2` 就把 EGL 的 GPU 2 重编号成 0。因此 client 同时设置：

```bash
CUDA_VISIBLE_DEVICES=2
MUJOCO_EGL_DEVICE_ID=2
```

前者限制该进程的 CUDA 可见范围，后者选择物理 EGL 设备；二者不是同一个编号体系。

## 4. 代码与启动器

### 4.1 生产启动器

新增：

- [`evaluation/LIBERO-plus2/run_eval_libero_plus_2server_6client_venv.sh`](../../../evaluation/LIBERO-plus2/run_eval_libero_plus_2server_6client_venv.sh)

职责：

1. 校验必须是 2 张 server GPU、6 张 client GPU，且集合无重复；
2. 根据 `SERVER_INSTANCES_PER_GPU=1|3`（默认 `3`）在两张 server GPU 上启动 2 或 6 个模型 server；
3. 所有 server 都通过 WebSocket metadata healthcheck 后，才启动 client；
4. `SERVER_INSTANCES_PER_GPU=1` 时将六个 client 按 3:3 分到两个 server；默认
   `3` 时按 1:1 分配到六个 server；
5. client 强制使用 EGL，并为每个 client 设置独立的物理 EGL device；
6. 每个 client 可以顺序处理多个 task shard；
7. server 提前退出时立即失败，不再无意义等待完整启动超时；
8. 启动 server 后记录 `gpu_topology_snapshot.txt`，便于核对 GPU 使用情况。

默认 `SERVER_INSTANCES_PER_GPU=3`（六 server），需要更省资源/更快启动时可
显式设置为 `1` 回退到双 server 行为。六 server 会使每张 H200 多出两个模型
进程，但本次实测每个 instance 约占 12.9 GiB，三份模型仍适合当前 144 GiB
H200。显存可行不代表吞吐必然线性提升，因为三个 instance 仍共享同一张 GPU
的计算资源；§8.2 的 A/B 实测证明六 server 在当前 workload 下仍净胜出。

启动器默认使用 `INFERENCE_BACKEND=standard`。原因是当前 LIBERO2 client 启用 keypoint 时会发送 `his_kpts`，而当前 `InternVLAA15Optimized.sample_actions()` 不接受该参数。若关闭 keypoint 或修复 optimized backend，可以显式切换 optimized；在此之前不应把 optimized 作为 LIBERO2 的默认值。

### 4.2 标准 LIBERO 多 episode 回归 harness

新增：

- [`evaluation/LIBERO2/test_egl_split_standard_smoke.py`](../../../evaluation/LIBERO2/test_egl_split_standard_smoke.py)

该脚本不是全量评估器，而是稳定性验收工具。它固定执行：

- 两个真实 InternVLA-A1.5 server；
- GPU 0/1 作为 server；
- GPU 2–7 作为六个 EGL client；
- `libero_spatial` task 0–5；
- 每个 task 在一个 forked task child 中连续跑 5 个 episode；
- 检查子进程返回码、结果 JSON、`SIGABRT`、`SIGSEGV` 和 `CRASHED`。

### 4.3 拓扑静态与 mock 活体测试

新增：

- [`evaluation/LIBERO-plus2/test_egl_split_topology.py`](../../../evaluation/LIBERO-plus2/test_egl_split_topology.py)

不加 `--live` 时只做离线静态检查；加 `--live` 时启动两个 mock server，但 client 仍然使用真实 LIBERO `OffScreenRenderEnv` 和真实 EGL。这样可以把“EGL/拓扑问题”和“模型推理问题”分离。

### 4.4 双/六 server A/B 冒烟 harness

新增：

- [`evaluation/LIBERO2/test_egl_topology_ab.py`](../../../evaluation/LIBERO2/test_egl_topology_ab.py)

该 harness 以相同的 checkpoint、seed、EGL client GPU 和 task 分配顺序，
顺序执行 `two` 与 `six` 两种拓扑，收集 server ready time、client wall time、
端到端耗时、每个 task 的结果 JSON、返回码和 crash 关键词。它不把 policy
success rate 当作拓扑稳定性指标；拓扑验收关注 episode 是否完整结束、结果
是否齐全以及是否发生 signal crash。

## 5. 动态调用关系

```mermaid
sequenceDiagram
    participant L as Launcher
    participant S0 as Server-0 GPU0 :5784
    participant S1 as Server-1 GPU1 :5785
    participant C2 as Client GPU2
    participant C3 as Client GPU3
    participant C4 as Client GPU4
    participant C5 as Client GPU5
    participant C6 as Client GPU6
    participant C7 as Client GPU7
    participant E as Forked task child

    L->>S0: CUDA_VISIBLE_DEVICES=0; load model
    L->>S1: CUDA_VISIBLE_DEVICES=1; load model
    L->>S0: metadata healthcheck
    L->>S1: metadata healthcheck
    L->>C2: start EGL, device=2
    L->>C3: start EGL, device=3
    L->>C4: start EGL, device=4
    L->>C5: start EGL, device=5
    L->>C6: start EGL, device=6
    L->>C7: start EGL, device=7
    C2->>E: fork task child and create MuJoCo EGL context
    C3->>E: fork task child and create MuJoCo EGL context
    C4->>E: fork task child and create MuJoCo EGL context
    C2->>S0: image/state/keypoint request
    C3->>S0: image/state/keypoint request
    C4->>S0: image/state/keypoint request
    C5->>S1: image/state/keypoint request
    C6->>S1: image/state/keypoint request
    C7->>S1: image/state/keypoint request
    S0-->>C2: action chunk
    S0-->>C3: action chunk
    S0-->>C4: action chunk
    S1-->>C5: action chunk
    S1-->>C6: action chunk
    S1-->>C7: action chunk
```

当前 `WebsocketPolicyServer` 的 `_route_message()` 是同步调用 policy inference 的。因此
双 server 拓扑中同一个 server 上的三个 client 请求会排队；六 server 拓扑将这个
队列拆成六个独立 endpoint，减少了 client 间的排队等待，但三个 server 仍会在
同一张模型 GPU 上争用 CUDA 计算资源。

## 6. 运行方式

### 6.1 环境变量

```bash
export SERVER_VENV=/B/VENV/itnvla15rbt20
export CLIENT_VENV=/B/VENV/libero_plus_client
export LIBERO_HOME=/home/a26113/DATA/LIBERO-plus
export CKPT_PATH=/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_12_08_52_49-internvla_a1_5-libplus-sft/checkpoints/032070/pretrained_model
export VLM_MODEL_PATH=/B/VENV/hf_home/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
export SERVER_GPU_IDS=0,1
export CLIENT_GPU_IDS=2,3,4,5,6,7
export SERVER_INSTANCES_PER_GPU=3  # 默认 3=六 server；显式设为 1 回退双 server
export RENDER_BACKEND=egl
export INFERENCE_BACKEND=standard
```

### 6.2 LIBERO-plus 短冒烟

该命令只验证启动、通信、EGL、推理和结果写出，不用于计算 benchmark SR。不显式
设置 `SERVER_INSTANCES_PER_GPU` 时默认即为六 server（六个 client 与六个
server 一一对应）：

```bash
EVAL_MODE=smoke \
SMOKE_SUITE=libero_spatial \
SMOKE_TASKS=6 \
MAX_STEPS_OVERRIDE=20 \
NUM_TRIALS_PER_TASK=1 \
SAVE_ACTIONS_FLAG=--no-save_actions \
bash evaluation/LIBERO-plus2/run_eval_libero_plus_2server_6client_venv.sh
```

需要回退到双 server（更省显存/启动更快）时显式设置为 `1`：

```bash
SERVER_INSTANCES_PER_GPU=1 \
EVAL_MODE=smoke \
SMOKE_SUITE=libero_spatial \
SMOKE_TASKS=6 \
MAX_STEPS_OVERRIDE=20 \
NUM_TRIALS_PER_TASK=1 \
SAVE_ACTIONS_FLAG=--no-save_actions \
bash evaluation/LIBERO-plus2/run_eval_libero_plus_2server_6client_venv.sh
```

该短冒烟只验收启动、metadata healthcheck、EGL、WebSocket、client 正常退出
和结果写出；`MAX_STEPS_OVERRIDE=20` 过短，不能用于解释 policy success rate。

### 6.3 LIBERO-plus 正式运行

正式运行不设置 `EVAL_MODE=smoke`：

```bash
SHARDS_PER_SUITE=6 \
NUM_TRIALS_PER_TASK=1 \
bash evaluation/LIBERO-plus2/run_eval_libero_plus_2server_6client_venv.sh
```

正式运行前必须先通过第 7 节的标准 LIBERO 5 episode 稳定性门禁。正式评估仍然需要检查：

1. `server*/server.log` 中两个 server 均保持存活；
2. `client*/` 中每个 shard 都有正常退出码；
3. `logs/<suite>/` 中结果 JSON 数量与 shard 数量一致；
4. `gpu_topology_snapshot.txt` 与预期 GPU 分工一致；
5. 没有 SIGABRT、SIGSEGV 或缺失结果 JSON。

### 6.4 标准 LIBERO 5 episode 回归

```bash
/B/VENV/libero_plus_client/bin/python \
  evaluation/LIBERO2/test_egl_split_standard_smoke.py \
  --ckpt_path "${CKPT_PATH}" \
  --vlm_model_path "${VLM_MODEL_PATH}" \
  --libero_home "${LIBERO_HOME}" \
  --server_venv "${SERVER_VENV}" \
  --client_venv "${CLIENT_VENV}" \
  --episodes 5 \
  --json_out /tmp/egl_split_standard_smoke.json
```

### 6.5 双/六 server A/B 冒烟

以下命令使用 24 个 task definition、每个 5 个 episode，即每种拓扑 120 个
episode；两种拓扑顺序运行，第二种拓扑不会继承第一种拓扑的 server 进程：

```bash
/B/VENV/libero_plus_client/bin/python \
  evaluation/LIBERO2/test_egl_topology_ab.py \
  --topology both \
  --ckpt_path "${CKPT_PATH}" \
  --vlm_model_path "${VLM_MODEL_PATH}" \
  --libero_home "${LIBERO_HOME}" \
  --server_venv "${SERVER_VENV}" \
  --client_venv "${CLIENT_VENV}" \
  --tasks_per_suite 6 \
  --episodes 5 \
  --json_out /tmp/egl_topology_ab.json
```

本次结果保存在 [`asset/libero_egl_topology_ab_results.json`](asset/libero_egl_topology_ab_results.json)。

## 7. 验收门禁

### S1：离线拓扑门禁

```bash
/B/VENV/libero_plus_client/bin/python \
  evaluation/LIBERO-plus2/test_egl_split_topology.py
```

必须满足：

- 2 个 server GPU、6 个 client GPU；
- server/client GPU 集合不相交；
- 每个 client 的 server group 和 port 映射正确；
- client 设置 EGL；
- server 清除 EGL 变量；
- 默认 inference backend 为兼容 keypoint 的 `standard`。

### S2：EGL 活体门禁

```bash
export LIBERO_HOME=/home/a26113/DATA/LIBERO-plus
export LIBERO_CONFIG_PATH=/tmp/test_libero_config
/B/VENV/libero_plus_client/bin/python \
  evaluation/LIBERO-plus2/test_egl_split_topology.py \
  --live \
  --server_venv /B/VENV/itnvla15rbt20 \
  --client_venv /B/VENV/libero_plus_client \
  --episodes 5 \
  --steps 20
```

必须满足：

- 六个 client 都返回 0；
- 每个 client 完成 5 个 episode；
- `mujoco_egl_device_id` 分别为 2、3、4、5、6、7；
- `GL_RENDERER` 成功显示 NVIDIA H200；
- 没有 signal-killed child。

### S3：真实模型多 episode 门禁

必须满足：

- 两个 server metadata 都是 `protocol_version >= 2.1`；
- `preprocessing_owner=server_canonical`；
- 六个 client 都正常退出；
- 6 个 task result JSON 全部写出；
- 总 episode 数为 30；
- 没有 SIGABRT、SIGSEGV、CRASHED；
- 不允许用“补写部分结果”作为 PASS 条件。

本次 S3 实测结果：

```text
server GPUs: 0, 1
client GPUs: 2, 3, 4, 5, 6, 7
tasks: 6
episodes/task: 5
total episodes: 30
successful episodes: 30
result JSONs: 6
signal-killed clients: 0
SIGABRT/SIGSEGV/CRASHED: 0
OVERALL: PASS
```

## 8. 性能判断与限制

### 8.1 已观察到的性能

标准 LIBERO 实测中，模型 server 加载和每个 client 的 LIBERO/keypoint 初始化是
额外启动成本。EGL client 的渲染仍然使用 NVIDIA H200，不会退化到 OSMesa；
相比 OSMesa，这保留了 GPU rasterization 的速度；相比“server 与 client 共用
GPU”，server GPU 不再承担 EGL framebuffer 和渲染负载。

### 8.2 24 task / 120 episode A/B 实测

两种拓扑严格使用相同的 24 个 task definition（四个 suite 各 6 个 task）、
每个 task 5 个 episode、相同 seed 和 checkpoint。结果如下：

| 指标 | 双 server（1 instance/GPU） | 六 server（3 instances/GPU） |
|---|---:|---:|
| task definition | 24 | 24 |
| episode | 120 | 120 |
| 完整结果 episode | 120 | 120 |
| policy success | 115/120（95.83%） | 114/120（95.00%） |
| server ready | 40.669 s | 87.936 s |
| client wall time | 712.065 s | 512.169 s |
| 端到端时间 | 753.010 s | 600.436 s |
| 缺失结果 JSON | 0 | 0 |
| SIGABRT/SIGSEGV/CRASHED | 0 | 0 |

因此，六 server 的 client 阶段耗时是双 server 的 71.93%，即快 **28.07%**；
计入额外的模型启动时间后，端到端仍快 **20.26%**。六 server 启动慢约 47.267
秒，但拆分了双 server 上三个 client 共享的同步 inference 队列。

两次 policy success rate 相差 1 个 episode，不能作为拓扑稳定性结论：它受
并发请求时序、GPU 浮点执行顺序和环境状态影响；本次真正的稳定性结论是两种
拓扑都完整处理了 120 个 episode，且 crash/missing-result 均为 0。

### 8.3 双 server 与六 server 的适用边界

双 server 使用两份模型，六 server 使用六份模型；每张模型 GPU 上的三个
instance 会共享 CUDA 计算资源。因此：

- 双 server 的优点：模型显存和模型进程数更少，启动更快，资源余量更大；
- 双 server 的缺点：每个 server 的三个 client 共享一个同步 inference 队列；
- 六 server 的优点：每个 client 有独立 endpoint，当前 workload 下吞吐更高；
- 六 server 的缺点：启动时间更长，模型 GPU 上同时运行三个模型进程，CUDA
  计算与显存管理更拥挤；
- 两者都将模型 GPU 与 EGL client GPU 分离，因此本次没有观察到稳定性差异。

如果模型 GPU 显存不足、模型 instance 初始化失败或更高并发导致 CUDA 争用，
应退回双 server；如果当前模型规模和 GPU 余量与本机一致，优先使用六 server
以获得更高吞吐。正式 benchmark 时仍建议记录：

- 从首个 client task 开始到最后一个 shard 完成的 wall time；
- server 每次 inference latency 的 p50/p95；
- 每个 client 的 env step latency；
- GPU 0/1 的模型利用率与显存；
- GPU 2–7 的 EGL 利用率与显存；
- 每 100 个 episode 的 SIGABRT 计数和缺失 JSON 计数。

## 9. 本次错误与修复

### 9.1 初次模型命令的 shell 拼接错误

第一次执行时把脚本路径错误地放到了 `export` 命令中，shell 报：

```text
export: evaluation/LIBERO-plus2/run_eval_libero_plus_2server_6client_venv.sh: not a valid identifier
```

模型尚未启动。修复为先 `export` 环境变量，再单独执行 `bash script`。

### 9.2 错用 base checkpoint

初次真实模型启动使用了 `InternVLA-A1.5-base`。其 `stats.json` 没有 `panda`，只有预训练数据集 key，server 在模型加载后报：

```text
KeyError: stats_key 'panda' not found in stats.json
```

修复为使用 LIBERO-compatible 的 step 032070 checkpoint：

```text
/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/
2026_09_12_08_52_49-internvla_a1_5-libplus-sft/
checkpoints/032070/pretrained_model/
```

### 9.3 optimized backend 与 keypoint payload 不兼容

第一次使用 `INFERENCE_BACKEND=optimized` 时，server 返回：

```text
InternVLAA15Optimized.sample_actions()
got an unexpected keyword argument 'his_kpts'
```

这不是 EGL 崩溃，而是当前 optimized backend API 与 LIBERO2 keypoint client 不一致。修复是：

1. 本方案 launcher 默认改为 `INFERENCE_BACKEND=standard`；
2. 在 optimized backend 接受 `his_kpts` 之前，不把 optimized 作为启用 keypoint 的 LIBERO2 默认路径；
3. 该兼容性问题不应被误判为双 server 拓扑失败。

### 9.4 mock smoke 的 websocket envelope 错误

最初 mock smoke 直接读取 `response["actions"]`，但协议返回的是：

```text
response["ok"]
response["data"]["actions"]
```

测试已按实际 WebSocket envelope 修复，并重新通过六 client 活体测试。

### 9.5 server 提前退出时等待过久

启动器原先只按固定超时等待。现在 healthcheck 循环中增加 server PID 存活检查；如果 server 因 checkpoint、stats 或模型加载错误提前退出，会立即提示对应 `server*/server.log`。

### 9.6 A/B harness 的端口参数错误

A/B harness 首次运行时，父进程误用了 worker 专用的默认 `port_base=-1`，
导致等待 `port -1` 并在超时后退出；六个 server 当时尚未进入有效评估。修复
为父进程使用公共 `base_port`，worker 仍接收每种拓扑实际的端口基址。修复后
重新完成双/六 server 的 120 episode 对比，结果 JSON 已写入
[`asset/libero_egl_topology_ab_results.json`](asset/libero_egl_topology_ab_results.json)。

## 10. 最终建议

在当前 8×H200 机器和本次 checkpoint 上，六 server（`SERVER_INSTANCES_PER_GPU=3`）
已是两个生产 launcher 不显式设置时的默认值，也是追求吞吐时的 EGL 正式评估
首选；双 server（显式设置为 `1`）作为更保守的资源/启动成本回退方案：

1. GPU 0/1 固定为模型 server，默认每张卡启动三个（`SERVER_INSTANCES_PER_GPU=3`）；
2. GPU 2–7 固定为六个 EGL client；
3. client/server GPU 集合必须不重叠；
4. 使用 `INFERENCE_BACKEND=standard` + keypoints；
5. 正式运行前通过 S1、S2、S3，并在需要比较吞吐时运行 6.5；
6. 任何 SIGABRT 或缺失 JSON 都判定本轮失败，不把部分结果当作成功；
7. 若六 server 初始化失败、显存不足或再次出现 CUDA 争用，改用
   `SERVER_INSTANCES_PER_GPU=1`；
8. 若分离 GPU 后仍出现 EGL SIGABRT，再切换 [`eval3_optim3.md`](eval3_optim3.md) 中的 OSMesa 方案。

本次证据支持的准确表述是：

> “在当前 8×H200 容器上，双 server 和六 server + 六 EGL client 都完成了相同
> 的 24 task / 120 episode A/B，无 SIGABRT、SIGSEGV、CRASHED 或缺失结果；
> 六 server 的 client 阶段快 28.1%、端到端快 20.3%。这证明六 server 在本次
> workload 下吞吐更高，但不证明所有 workload 下都更快，也不是对所有 EGL
> 驱动级 SIGABRT 的绝对保证。”

