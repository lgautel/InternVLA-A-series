# RoboTwin 2.0 双任务微调执行日志

> 对应操作手册：[`sft0827.md`](sft0827.md)  
> 设计依据：[`run_ech_rbt_p012.md`](run_ech_rbt_p012.md)  
> 目标任务：`place_bread_skillet` → `pick_dual_bottles`

## 2026-08-27

### 20:32 — 执行启动与只读前置核查

**目标**：严格按操作手册，串行完成两个 RoboTwin 2.0 任务的 Phase 0、Phase 1 和 Phase 2；所有阶段、错误、修复和产物按时间顺序记录。

**已核对的硬件和路径**：

```text
GPU: 8 × NVIDIA A800-SXM4-80GB
itvlaGp: /home/a26113/SRC/itvlaGp
GeoPredict: /home/a26113/SRC/GeoPredict
RoboTwin: /home/a26113/SRC/RoboTwin
源数据: /home/a26113/Dta/RoboTwin-Clean
SAPIEN Python: /tmp/B/VENV/RoboTwin/bin/python
训练 Python（配置修正前模板值）: /tmp/itnvla15rbt20/bin/python
训练 Python（实际已发现）: /home/a26113/VENV/itnvla15rbt20/bin/python
```

两个源任务均存在：

```text
/home/a26113/Dta/RoboTwin-Clean/place_bread_skillet/meta/info.json
/home/a26113/Dta/RoboTwin-Clean/pick_dual_bottles/meta/info.json
```

RoboTwin URDF 存在：

```text
/home/a26113/SRC/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf
```

只读核查发现训练 venv 的模板路径 `/tmp/itnvla15rbt20` 不存在，但实际 venv 已存在，且所需权重目录也已存在：

```text
/home/a26113/VENV/itnvla15rbt20/var/hf_home/ckpts/InternVLA-A1.5-base
/home/a26113/VENV/itnvla15rbt20/var/hf_home/ckpts/GeoPredict_robocasa.pth
/home/a26113/VENV/itnvla15rbt20/var/hf_home/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
```

### 20:33 — 修正本机配置

**原因**：`/home/a26113/Cfg/itvlaGp_rbt_batch1.env` 中的 `ROBOTWIN_ROOT` 已正确，但 `EXTRACT_PYTHON` 仍为占位路径，`VENV_ROOT` 指向不存在的 `/tmp/itnvla15rbt20`。如果直接运行编排脚本，训练解释器和权重路径会错误。

**执行的修改**：

```diff
- VENV_ROOT=/tmp/itnvla15rbt20
+ VENV_ROOT=/home/a26113/VENV/itnvla15rbt20
- EXTRACT_PYTHON=/path/to/robotwin/conda/envs/robotwin/bin/python
+ EXTRACT_PYTHON=/tmp/B/VENV/RoboTwin/bin/python
```

`ROBOTWIN_ROOT` 确认为：

```text
/home/a26113/SRC/RoboTwin
```

保留配置：`PROC_PER_NODE=8`、`BATCH_SIZE=16`、`WARMUP_STEPS=400`、`SFT_EPOCHS=76`、`CUDA_VISIBLE_DEVICES=0..7`。

**本次执行的文件变更**：

- 修改 `/home/a26113/Cfg/itvlaGp_rbt_batch1.env`：使配置匹配本机实际 venv 和 SAPIEN 解释器。
- 新建本日志文件：按用户要求保存全过程。
- 未修改计划文件。
- 保留既有无关修改 `/home/a26113/SRC/itvlaGp/b/d/rbt/fill_8gpu_vram.py`，不将其纳入本次训练改动。

### 20:36 — 环境验收第一次失败

执行了训练依赖逐项导入和权重验收命令：

```bash
source /home/a26113/Cfg/itvlaGp_rbt_batch1.env
"${EXTRACT_PYTHON}" -c 'import sapien; print("sapien", sapien.__file__)'
"${TRAIN_PYTHON}" -c 'import torch, transformers, accelerate, lerobot, torchcodec; ...'
```

已通过：

- `EXTRACT_PYTHON` 能导入 SAPIEN。
- 两个源任务均为 `v2.1`，各有 50 个 episode。
- `place_bread_skillet` 为 8277 帧，`pick_dual_bottles` 为 6129 帧。
- 8 张 GPU 均为 NVIDIA A800-SXM4-80GB。
- URDF、A1.5-base、GeoPredict checkpoint、WAN VAE 文件均存在。

发现的问题：

1. 单独导入 `lerobot` 报错：

   ```text
   ModuleNotFoundError: No module named 'lerobot'
   ```

2. 单独导入 `torchcodec` 报错：

   ```text
   OSError: libnppicc.so.12: cannot open shared object file: No such file or directory
   ```

3. 组合导入命令在导入 `transformers` 阶段长时间无输出，未作为训练启动；后续拆分验证。

**根因分析**：

- 现有训练 venv `/home/a26113/VENV/itnvla15rbt20` 的 editable 项目链接未指向当前 `/home/a26113/SRC/itvlaGp`，因此 Python 找不到 `lerobot`。
- venv 中 torchcodec 依赖的 NVIDIA NPP 动态库未安装或未加入库搜索路径。

**处理计划**：先检查 editable 安装元数据和现有 torchcodec 包，再执行当前源码的 editable 安装，并安装 `nvidia-npp-cu12`；随后重新进行导入验收。

### 20:42 — 修复训练环境并完成依赖验收

进一步检查发现这是一个从 `/tmp` 搬到 `/home` 的 venv：

- `internvla-a1-5` 的旧 editable location 为 `/tmp/SRC/itvlaGp`。
- `bin/pip` 的 shebang 仍指向已不存在的 `/tmp/itnvla15rbt20/bin/python3`，因此不能直接使用 `${VENV}/bin/pip`。
- NPP 包实际已安装，`libnppicc.so.12` 位于：

  ```text
  /home/a26113/VENV/itnvla15rbt20/lib/python3.11/site-packages/nvidia/npp/lib/libnppicc.so.12
  ```

**修复命令**：

```bash
V=/home/a26113/VENV/itnvla15rbt20
"${V}/bin/python" -m pip install \
  --ignore-installed --no-deps --no-build-isolation \
  -e /home/a26113/SRC/itvlaGp
```

**修复原因**：使用 venv 内实际可执行的 Python 调用 pip，避免损坏的旧 pip shebang；`--ignore-installed` 避免 pip 在 Ceph 文件系统上卸载旧 `/tmp` editable 安装时长时间阻塞；`--no-deps` 防止已验证的 CUDA/PyTorch 依赖被重新解析或替换。

修复结果：

```text
lerobot -> /home/a26113/SRC/itvlaGp/src/lerobot/__init__.py
transformers 5.2.0
accelerate 1.14.0
torch 2.10.0+cu128
CUDA available=True, device_count=8
torchcodec import OK（需使用 launch 同样的 LD_LIBRARY_PATH）
Qwen3.5 patch present
```

`torchcodec` 直接不设置 `LD_LIBRARY_PATH` 时仍会报 `libnppicc.so.12` 找不到；训练 launch 会导出包含 `${VENV}/lib/python3.11/site-packages/nvidia/npp/lib` 的库路径，因此使用 launch 环境验收已通过。SAPIEN 导入也已通过，但出现 Vulkan fallback warning；该 warning 不阻止当前离线 FK 关键点提取。

### 20:45 — 第一次 dry-run 失败及修复

执行：

```bash
cd /home/a26113/SRC/itvlaGp
bash b/s/rbt/run_each_rbt_p012.sh \
  --config /home/a26113/Cfg/itvlaGp_rbt_batch1.env \
  --dry-run --skip-smoke
```

`--list-tasks` 已成功发现 50 个源任务；目标两个任务均为 v2.1。随后 dry-run 在 `place_bread_skillet` 结束时失败：

```text
错误: Phase0 结束后 v3.0 仍不完整: /home/a26113/Dta/RoboTwin-Clean/place_bread_skillet_kptsim_lrbv30
```

**根因**：`phase0_prep_data.sh` 在 `DRY_RUN=1` 时正确跳过提取、注入和转换，但脚本末尾仍无条件执行 `v30_ready` 检查；dry-run 本来不应要求产物已生成。

**修复**：在 `b/s/rbt/phase0_prep_data.sh` 中加入 dry-run 提前结束分支：

```bash
if [[ "${DRY_RUN}" == "1" ]]; then
  write_state "phase0" "dry_run" ...
  rbt_log "Phase0 dry-run 完成: 未创建数据产物"
  exit 0
fi
```

这样不会把“预览未创建数据”误报为 Phase 0 失败；真实运行仍执行原有 v3.0 完整性检查。

### 20:46 — 第二次 dry-run 失败及修复

修复 Phase 0 后重新执行相同 dry-run。Phase 0 已正确完成预览，但随后 Phase 1 失败：

```text
[2026-08-27 12:46:21] Phase0 dry-run 完成: 未创建数据产物
错误: Phase1 需要完整 v3.0 数据: /home/a26113/Dta/RoboTwin-Clean/place_bread_skillet_kptsim_lrbv30
```

**根因**：`phase1_warmup.sh` 和 `phase2_sft.sh` 都在进入 dry-run 分支前无条件执行 `v30_ready` 和 `ensure_lerobot_home_link`。Phase 0 dry-run 没有创建 v3.0 数据，故后续预览被错误拦截。

**修复**：

- `phase1_warmup.sh`：仅真实运行时检查 v3.0 并建立 LeRobot symlink；dry-run 打印并跳过。
- `phase2_sft.sh`：采用同样处理。

真实执行路径仍保留 v3.0 完整性检查和 symlink 创建，不降低运行时数据校验强度。

### 20:47 — 第三次 dry-run 失败及修复

再次执行完整 dry-run 时，Phase 0 和 Phase 1 预览均已通过；Phase 2 在进入训练命令预览前失败：

```text
错误: 找不到 warmup ckpt@400, 请先跑 Phase1
```

**根因**：Phase 2 在 dry-run 模式仍无条件要求已存在 Warmup ckpt@400；同时它只能从 v3.0 `meta/info.json` 读取任务帧数，而完整 dry-run 尚未创建 v3.0。

**修复**：`b/s/rbt/phase2_sft.sh` 现在在 dry-run 下：

- 对待生成的 Warmup ckpt 使用 `${TASK_WARMUP_DIR}/<pending-ckpt-400>` 占位路径；
- 若 v3.0 `info.json` 尚不存在，则回退读取源 v2.1 `meta/info.json` 估算 SFT schedule；
- 真实运行仍严格要求真实 Warmup ckpt 和 v3.0 数据，不接受占位路径。

### 20:47 — 完整 dry-run 通过

重新执行：

```bash
cd /home/a26113/SRC/itvlaGp
bash b/s/rbt/run_each_rbt_p012.sh \
  --config /home/a26113/Cfg/itvlaGp_rbt_batch1.env \
  --dry-run --skip-smoke
```

结果：退出码 0；两个任务均按 `place_bread_skillet` → `pick_dual_bottles` 顺序完成 Phase 0、Phase 1、Phase 2 预览。

预览得到：

```text
有效 batch = 8 * 16 * 1 = 128
place_bread_skillet: 8277 frames, 4940 steps, save_freq=1235
pick_dual_bottles: 6129 frames, 3648 steps, save_freq=912
两任务保存 epoch: 19, 38, 57, 76
```

此后开始真实 Phase 0；dry-run 修复不会削弱真实运行时的 v3.0 和 Warmup checkpoint 强校验。

### 20:47 — 启动真实 Phase 0

执行命令：

```bash
cd /home/a26113/SRC/itvlaGp
bash b/s/rbt/run_each_rbt_p012.sh \
  --config /home/a26113/Cfg/itvlaGp_rbt_batch1.env \
  --until phase0 --skip-smoke
```

原因：先只完成两个任务的数据准备，确认 SAPIEN 提取、关键点注入、Layer-1 和 v3.0 转换均通过后，再启动 GPU 训练，避免训练阶段才发现数据错误。

实际执行顺序已确认：

```text
place_bread_skillet
pick_dual_bottles
```

### 20:49 — Phase 0 第一次真实运行失败：注入依赖缺失

`place_bread_skillet` 的 SAPIEN 提取已完成 50/50 episodes，生成 8277 帧关键点；该任务的 norm stats 也已成功生成。随后注入阶段失败：

```text
FileNotFoundError: [Errno 2] No such file or directory: 'rsync'
```

失败日志：

```text
/home/a26113/Ckp/itvlaGp/place_bread_skillet/logs/phase0_2026_08_27_12_47_29.log
```

**根因**：`util_scripts/inject_kptsim_keypoints.py` 的 `_copy_dataset()` 固定调用系统命令 `rsync`，但当前机器未安装 `rsync`。这不是数据格式或关键点计算错误。

**修复**：修改 `util_scripts/inject_kptsim_keypoints.py`：

- 若 `rsync` 存在，继续使用原来的 `rsync -a`；
- 若不存在，改用 `shutil.copytree(..., symlinks=True)`；
- 保留已有目标目录时的 `--force` 删除逻辑。

**选择该修复的原因**：当前任务只需要本地复制一个 v2.1 数据集，Python 标准库可以完成同等复制；这样不必安装系统级包，也不影响已有机器上的 rsync 快速路径。

### 20:51 — 修复后重跑 Phase 0

在完成 Python 编译和 shell 语法检查（均通过）后，重新执行同一条 Phase 0 命令：

```bash
cd /home/a26113/SRC/itvlaGp
bash b/s/rbt/run_each_rbt_p012.sh \
  --config /home/a26113/Cfg/itvlaGp_rbt_batch1.env \
  --until phase0 --skip-smoke
```

本次会复用已经成功生成的 `place_bread_skillet_kptsim` 与 norm stats，只重做失败的注入及后续验收/转换；随后继续处理 `pick_dual_bottles`。

### 21:02 — Phase 0 第二次失败：转换同步仍依赖 rsync

本次重跑结果：

- `place_bread_skillet` 的注入成功；
- Layer-1 验收成功；
- v2.1→v3.0 转换本身完成，但同步转换结果时失败：

```text
/home/a26113/SRC/itvlaGp/b/s/rbt/phase0_prep_data.sh: line 116: rsync: command not found
```

**根因**：之前只修复了 `inject_kptsim_keypoints.py` 的数据复制；`phase0_prep_data.sh` 在把隔离转换工作区同步到最终 `${TASK}_kptsim_lrbv30` 时仍直接调用 `rsync -a --delete`。

**修复**：修改 `phase0_prep_data.sh`：

- 有 `rsync` 时保持原同步逻辑；
- 无 `rsync` 时先删除目标 v3.0 目录，再用训练 Python 的 `shutil.copytree(..., symlinks=True)` 复制；
- 继续保留转换工作区按任务隔离的设计。

**重跑策略**：复用已生成的关键点、norm stats 和注入数据；重新执行转换结果同步及后续 Layer-2 检查。

### 21:04 — 修复后再次重跑 Phase 0

执行：

```bash
cd /home/a26113/SRC/itvlaGp
bash b/s/rbt/run_each_rbt_p012.sh \
  --config /home/a26113/Cfg/itvlaGp_rbt_batch1.env \
  --until phase0 --skip-smoke
```

执行前已通过 `bash -n`、Python 编译检查和 IDE lint；本次仍复用 `place_bread_skillet` 已完成的提取/注入结果，并继续按任务顺序处理两个任务。

Phase 0 使用的关键路径：

```text
源数据: /home/a26113/Dta/RoboTwin-Clean/<task>
kptsim: /home/a26113/Dta/RoboTwin-Clean/<task>_kptsim
lrb: /home/a26113/Dta/RoboTwin-Clean/<task>_kptsim_lrb
v30: /home/a26113/Dta/RoboTwin-Clean/<task>_kptsim_lrbv30
转换隔离区: /home/a26113/Ckp/itvlaGp/.convert_ws/<task>
```

### 21:05 — Phase 0 全部成功

第三次真实 Phase 0 执行退出码为 0：

```text
place_bread_skillet: Phase0 完成，13:03:24
pick_dual_bottles: Phase0 完成，13:05:26
全部 2 个任务完成
```

两套 v3.0 数据均已通过 Layer-2 检查：

```text
/home/a26113/Dta/RoboTwin-Clean/place_bread_skillet_kptsim_lrbv30
  codebase_version=v3.0, episodes=50, frames=8277

/home/a26113/Dta/RoboTwin-Clean/pick_dual_bottles_kptsim_lrbv30
  codebase_version=v3.0, episodes=50, frames=6129
```

每个任务均有独立的 `norm_stat.json`、`meta/keypoints_meta.json` 和 `coord_offset`；转换工作区已按任务隔离并在成功后清理。由于机器没有 `rsync`，两次 v3.0 同步均使用 `shutil.copytree` 回退。

### 21:06 — 启动 Phase 1 Warmup

执行命令：

```bash
cd /home/a26113/SRC/itvlaGp
bash b/s/rbt/run_each_rbt_p012.sh \
  --config /home/a26113/Cfg/itvlaGp_rbt_batch1.env \
  --from warmup --until warmup
```

原因：Phase 0 已完成，按手册开始对两个任务分别执行 1 GPU × 1 step smoke，再执行 8 GPU × 400 step 正式 Warmup；两个任务严格串行，防止 CUDA、端口和输出目录冲突。

Warmup 预期产物：

```text
${CKPT_ROOT}/place_bread_skillet/warmup/latest/checkpoints/000400/pretrained_model
${CKPT_ROOT}/pick_dual_bottles/warmup/latest/checkpoints/000400/pretrained_model
```

### 21:23 — Warmup smoke 第一次失败：系统缺少 C 编译器

`place_bread_skillet` 的 smoke 已完成模型和 GeoPredict TrackEncoder 初始化，并进入实际前向；运行约 15 分钟后在 Triton kernel 首次编译阶段失败：

```text
RuntimeError: Failed to find C compiler.
Please specify via CC environment variable or set triton.knobs.build.impl.
```

相关日志：

```text
/home/a26113/Ckp/itvlaGp/place_bread_skillet/logs/warmup_smoke_2026_08_27_13_08_01.log
```

**根因**：当前机器没有 `/usr/bin/gcc`、`/usr/bin/g++` 或 `cc`；Triton 的 `gated_delta_rule` 在第一次 GPU 前向时需要 C 编译器生成 launcher。

**已确认**：

- CUDA、PyTorch、模型加载、数据读取均已通过；
- `nvidia-smi` 可见 8 张 A800；
- 失败发生在 Triton 编译器探测，非 OOM、非数据错误；
- Qwen3.5 所需文件已在 `${HF_HOME}` 缓存，后续不应再次下载大模型。

**处理措施**：检查到当前用户具有免密码 sudo 权限；下一步安装系统 `gcc/g++`，并在配置中显式设置 `CC=/usr/bin/gcc`、`CXX=/usr/bin/g++`，然后重新执行 Warmup smoke。

### 21:24 — 安装系统编译器

执行：

```bash
sudo apt-get update
sudo apt-get install -y gcc g++
```

结果：安装成功，系统新增 `/usr/bin/gcc` 和 `/usr/bin/g++`；同时安装 Triton 编译所需的 binutils、libc-dev 等依赖。

配置文件新增：

```text
CC=/usr/bin/gcc
CXX=/usr/bin/g++
```

**原因**：显式固定 Triton 使用的 C/C++ 编译器，避免不同 shell 或 accelerate 子进程找不到编译器。

### 21:25 — 重新启动 Warmup（含 smoke）

重新执行：

```bash
cd /home/a26113/SRC/itvlaGp
bash b/s/rbt/run_each_rbt_p012.sh \
  --config /home/a26113/Cfg/itvlaGp_rbt_batch1.env \
  --from warmup --until warmup
```

本次会先重新执行 `place_bread_skillet` 的 1 GPU × 1 step smoke；通过后继续 8 GPU × 400 step，并再处理 `pick_dual_bottles`。

### 21:35 — Warmup smoke 第二次失败：TileLang 找不到 CUDA toolkit

安装 gcc/g++ 后，smoke 已成功完成前向 Triton kernel 编译并进入反向；随后 TileLang 报错：

```text
ValueError: No registered target detector found an available target.
```

单独诊断确认：

```text
torch.cuda.is_available() = True
torch.cuda.device_count() = 8
tilelang target detectors = ('cuda', 'hip', 'metal')
check_cuda_availability() = False
CUDA_HOME=''
nvcc: command not found
```

**根因**：TileLang 的 CUDA detector 只在 `tilelang.contrib.nvcc.find_cuda_path()` 成功时返回 CUDA target；该函数要求 `CUDA_HOME` 非空，而当前机器没有系统 CUDA toolkit 路径。PyTorch 自带的 CUDA runtime 足以运行现有 kernel，但未被 TileLang 当作 toolkit 发现。

**修复**：在 `/home/a26113/Cfg/itvlaGp_rbt_batch1.env` 增加：

```text
CUDA_HOME=${VENV_ROOT}
```

当前 TileLang 验证结果：

```text
cuda_available=True
target={"kind":"cuda", "arch":"sm_80"}
```

这一步只为 TileLang 的目标检测提供有效的 CUDA runtime/toolkit 根路径；没有改动模型代码，也没有伪造 GPU 架构。

### 21:49 — Warmup smoke 第三次失败：TileLang 仍要求 nvcc

重新运行后，TileLang 目标检测已通过，但生成 CUDA PTX 时失败：

```text
RuntimeError: [Errno 2] No such file or directory: '/home/a26113/VENV/itnvla15rbt20/bin/nvcc'
```

**根因**：本机没有完整 CUDA toolkit/nvcc；`CUDA_HOME=${VENV_ROOT}` 只能让 TileLang 通过路径检测，不能提供不存在的 `${VENV_ROOT}/bin/nvcc`。继续安装完整 toolkit 会增加不必要的环境和磁盘风险。

**修复**：在配置中增加：

```text
FLA_TILELANG=0
```

FLA 的 backend dispatch 因此跳过 TileLang，使用已有的 Triton CUDA backend。该选择与当前 A800（sm80）硬件匹配，且不改变模型或训练超参数。

### 22:08 — 正式 Warmup 首个 batch 长时间无进度：Ceph 数据加载阻塞

smoke 已经成功完成后，正式执行命令为：

```bash
cd /home/a26113/SRC/itvlaGp
bash b/s/rbt/run_each_rbt_p012.sh \
  --config /home/a26113/Cfg/itvlaGp_rbt_batch1.env \
  --from warmup --until warmup
```

正式运行已完成模型创建、GeoPredict TrackEncoder 权重加载和 optimizer 初始化：

```text
cfg.steps=400 (400)
num_frames=8277 (8K)
num_episodes=50 (50)
Effective batch size: 16 x 8 = 128
Trainable params: 927M
WAN params: 0
Start offline training on a fixed dataset
```

但在 `14:06:08 UTC` 输出 `Start offline training...` 后约 4 分钟仍无
`step`、loss 或 checkpoint 输出。只读诊断命令：

```bash
ps -eo pid,ppid,stat,etime,%cpu,rss,wchan:24,cmd \
  | rg '125443|12546[2-9]|python -u src/lerobot'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

观察到：

```text
8 个训练进程均处于 Dsl，wchan 为 ceph_mdsc_wait_request /
rwsem_down_write_slowpath；部分进程及其 DataLoader worker 处于
futex_wait_queue_me / do_poll.constprop.0。
GPU 显存约 13403 MiB/卡，利用率 0%。
```

**根因分析**：正式脚本默认 `NUM_WORKERS=12`，在 8 卡训练中会创建最多
96 个 DataLoader worker。数据和视频文件位于 `/home/a26113/Dta` 的 Ceph
文件系统；首个 batch 的视频读取被大量并发 worker 卡在 Ceph 元数据/读请求
上。此时不是 CUDA OOM、模型权重缺失或训练进程崩溃，且日志没有出现 traceback。

为避免继续占用 8 张 GPU 并无限等待，停止本次无进度运行：

```bash
descendants() {
  for child in $(pgrep -P "$1" 2>/dev/null || true); do
    descendants "$child"
    echo "$child"
  done
}
pids="$(descendants 122773) 122773"
kill -TERM $pids 2>/dev/null || true
```

停止后复核，原 warmup 进程已退出。该次运行未产生新的
`checkpoints/000100`，也未被标记为成功；保留 smoke 产物和该正式运行目录，
用于审计和故障记录。

**修复措施**：下一次重跑正式 Warmup 时将显式设置
`NUM_WORKERS=0`，让每个训练进程在主进程中读取 batch，降低 Ceph 并发；
不改动模型、数据集、batch size、step 数或 checkpoint 保存策略。命令如下：

```bash
cd /home/a26113/SRC/itvlaGp
NUM_WORKERS=0 bash b/s/rbt/run_each_rbt_p012.sh \
  --config /home/a26113/Cfg/itvlaGp_rbt_batch1.env \
  --from warmup --until warmup

### 22:48 — 8 卡首步 DDP 等待：将 Triton cache 从 Ceph 移到本机 XFS

第三次 Warmup 使用本机 v3 数据和 `NUM_WORKERS=0`。smoke 阶段通过；正式
阶段已完成模型加载并在 `14:28:01 UTC` 输出开始训练。之后约 20 分钟没有
任何 step/loss/checkpoint 输出。

诊断：

```bash
ps -eo pid,ppid,stat,etime,%cpu,rss,wchan:24,cmd \
  | rg 'accelerate.commands.launch|python -u src/lerobot/scripts/lerobot_train.py'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
du -sh /home/a26113/.triton
ls -lt /home/a26113/.triton
```

现象是 8 个 rank 均在 `futex_wait_queue_me`，GPU 显存约 36.2 GiB/卡但
利用率为 0%；`/home/a26113/.triton/cache` 在首步期间更新。这里的
`/home/a26113` 是 Ceph，故 fused linear-attention/Triton 首次编译的共享
cache 可能发生跨 rank 的 Ceph 文件锁/元数据竞争。smoke 只有单 rank，
因此不能覆盖这个 8-rank cache 竞争场景。

停止该次无进度运行：

```bash
descendants() {
  for child in $(pgrep -P "$1" 2>/dev/null || true); do
    descendants "$child"
    echo "$child"
  done
}
pids="$(descendants 133685) 133685"
kill -TERM $pids 2>/dev/null || true
```

在 `/home/a26113/Cfg/itvlaGp_rbt_batch1.env` 增加：

```text
TRITON_CACHE_DIR=/tmp/itvla-triton-cache
```

**修复理由**：让 Triton 编译缓存、锁和中间产物全部位于本机 XFS，避免
Ceph 元数据锁参与多 rank 首次编译；不删除原 Ceph cache，以保留审计信息，
也不改变模型、数据、batch、step 或学习率。

### 23:17 — 两任务 400-step Warmup 正式完成

使用本机 XFS v3 数据、`NUM_WORKERS=0` 和
`TRITON_CACHE_DIR=/tmp/itvla-triton-cache` 的重跑最终成功：

```text
place_bread_skillet:
  smoke: 1 GPU × 1 step, exit=0, video_decode_error=0, using_zeros=0
  formal: 8 GPU × 400 step, exit=0, video_decode_error=0, using_zeros=0
  loss(step 400)=0.303, loss_action=0.125
  ckpt: /home/a26113/Ckp/itvlaGp/place_bread_skillet/warmup/latest/checkpoints/000400/pretrained_model

pick_dual_bottles:
  smoke: 1 GPU × 1 step, exit=0, video_decode_error=0, using_zeros=0
  formal: 8 GPU × 400 step, exit=0, video_decode_error=0, using_zeros=0
  loss(step 400)=0.344, loss_action=0.117
  ckpt: /home/a26113/Ckp/itvlaGp/pick_dual_bottles/warmup/latest/checkpoints/000400/pretrained_model
```

正式 Warmup 保存点均为 `000100`、`000200`、`000300`、`000400`。训练日志显示
两任务均在 8 GPU 下使用有效 batch 128；未出现视频解码告警、OOM 或 traceback。
两任务的 `pipeline_state.json` 已写入 warmup `ok` 状态。

Warmup 结束后开始 Phase 2 前的 schedule 计算命令：

```bash
source /home/a26113/Cfg/itvlaGp_rbt_batch1.env
/home/a26113/VENV/itnvla15rbt20/bin/python \
  /home/a26113/SRC/itvlaGp/b/s/rbt/compute_sft_steps.py \
  --info /tmp/RoboTwin-Clean/place_bread_skillet_kptsim_lrbv30/meta/info.json \
  --epochs "$SFT_EPOCHS" --n-gpus "$PROC_PER_NODE" \
  --batch-size "$BATCH_SIZE" --n-nodes "$NODE_COUNT" --as-exports
/home/a26113/VENV/itnvla15rbt20/bin/python \
  /home/a26113/SRC/itvlaGp/b/s/rbt/compute_sft_steps.py \
  --info /tmp/RoboTwin-Clean/pick_dual_bottles_kptsim_lrbv30/meta/info.json \
  --epochs "$SFT_EPOCHS" --n-gpus "$PROC_PER_NODE" \
  --batch-size "$BATCH_SIZE" --n-nodes "$NODE_COUNT" --as-exports
```

计算结果：

```text
place_bread_skillet: total_frames=8277, epochs=76, effective_bs=128,
  steps=4940, steps_per_epoch=65, warmup=494,
  save_freq=1235, save_steps=1235,2470,3705,4940
pick_dual_bottles: total_frames=6129, epochs=76, effective_bs=128,
  steps=3648, steps_per_epoch=48, warmup=364,
  save_freq=912, save_steps=912,1824,2736,3648
```

Phase 2 启动命令：

```bash
cd /home/a26113/SRC/itvlaGp
NUM_WORKERS=0 bash b/s/rbt/run_each_rbt_p012.sh \
  --config /home/a26113/Cfg/itvlaGp_rbt_batch1.env \
  --from sft --until sft
```

### 2026-08-28 00:41 — Phase 2 首次正式运行 OOM

本次启动先完成 `place_bread_skillet` 的 SFT smoke：

```text
1 GPU × 1 step, checkpoint 生成成功，video_decode_error=0，
using_zeros=0，exit=0
```

随后正式 SFT 使用计划中的 8 GPU、每卡 batch 16、4940 steps 启动。约在
正式运行 30 分钟后，8 个 rank 同时因显存不足退出；代表性错误：

```text
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 4.85 GiB.
GPU 1 has a total capacity of 79.25 GiB of which 4.84 GiB is free.
Process ... has 74.40 GiB memory in use.
Of the allocated memory 73.00 GiB is allocated by PyTorch.
```

**根因**：`launch/internvla_a15_geop_phase2_finetune_kptsim_8g.sh` 原先传入
`--policy.gradient_checkpointing=false`。Phase 2 不是 Warmup 的 expert-only
配置，而是同时训练 VLM、action expert 和 keypoint 模块；在 A800 80 GiB 上，
首个完整 SFT batch 的 activation 峰值超过可用显存。该错误不是视频解码或
数据路径问题，smoke 的视频检查已通过。

本次运行由 accelerate 清理其余 rank，最终：

```text
place_bread_skillet SFT: failed, exit=1
未创建 sft/latest；部分正式 run 目录仅作为失败审计产物保留。
```

修改文件：

```text
/home/a26113/SRC/itvlaGp/launch/internvla_a15_geop_phase2_finetune_kptsim_8g.sh
```

修改内容：

```text
--policy.gradient_checkpointing=true
```

**修复理由**：启用 activation/gradient checkpointing 以降低反向传播激活
内存；不降低每卡 batch、不改变有效 batch 128、学习率、总 epoch、总 steps
或 checkpoint schedule。后续将以新的时间戳 run 重试两个任务。

### 22:18 — `NUM_WORKERS=0` 仍受 Ceph 首 batch 读取影响，切换 v3 数据到本机 XFS

第二次尝试确实将训练参数变为 `--num_workers=0`，但在
`Start offline training on a fixed dataset` 后约 6 分钟，8 个进程仍处于
`Dsl`，等待点为 `rwsem_down_write_slowpath`、`lock_rename` 和
`ceph_mdsc_wait_request`；GPU 约 9.8 GiB/卡且利用率为 0%。因此问题并非
DataLoader worker 数量本身，而是训练主进程从 Ceph 上的 v3 数据读取视频时
发生阻塞。

停止第二次尝试（同样没有生成正式 checkpoint）：

```bash
descendants() {
  for child in $(pgrep -P "$1" 2>/dev/null || true); do
    descendants "$child"
    echo "$child"
  done
}
pids="$(descendants 129068) 129068"
kill -TERM $pids 2>/dev/null || true
```

确认文件系统：

```bash
df -T /home/a26113/Dta/RoboTwin-Clean /tmp
du -sh /home/a26113/Dta/RoboTwin-Clean/\
place_bread_skillet_kptsim_lrbv30 \
/home/a26113/Dta/RoboTwin-Clean/pick_dual_bottles_kptsim_lrbv30
```

结果：`/home/a26113` 为 Ceph，`/tmp` 为本机 XFS；两个 v3 数据集大小分别
约 68 MiB 和 57 MiB。执行本地复制：

```bash
mkdir -p /tmp/RoboTwin-Clean-v30
cp -a /home/a26113/Dta/RoboTwin-Clean/place_bread_skillet_kptsim_lrbv30 \
  /tmp/RoboTwin-Clean-v30/
cp -a /home/a26113/Dta/RoboTwin-Clean/pick_dual_bottles_kptsim_lrbv30 \
  /tmp/RoboTwin-Clean-v30/
du -sh /tmp/RoboTwin-Clean-v30/*
```

修改 `/home/a26113/Cfg/itvlaGp_rbt_batch1.env`：

```text
V30_ROOT=/tmp/RoboTwin-Clean-v30
HF_LEROBOT_HOME="${V30_ROOT}"
```

**原因**：`CLEAN_ROOT` 仍保留 `/home/a26113/Dta/RoboTwin-Clean`，用于源任务、
URDF 前置检查和保留原始流水线产物；仅将已完成验收的 v3 训练数据及其
LeRobot 注册根切换到本机 XFS。这样不会混用两个任务的数据，也不重做
Phase 0。

下一次 Warmup 将从 `--from warmup` 启动，继续使用每卡 batch 16、8 卡、
400 step、每 100 step 保存，并保留 `NUM_WORKERS=0`：

```bash
cd /home/a26113/SRC/itvlaGp
NUM_WORKERS=0 bash b/s/rbt/run_each_rbt_p012.sh \
  --config /home/a26113/Cfg/itvlaGp_rbt_batch1.env \
  --from warmup --until warmup
```
```

### 2026-08-27 20:35 — Phase 2 SFT 第二次失败：开启 checkpointing 后仍在 WAN VAE OOM

第二次 `place_bread_skillet` SFT 尝试使用了前一节记录的
`--policy.gradient_checkpointing=true`，smoke 已成功并保存了 1-step
checkpoint；正式训练仍在第一个 batch（总 schedule 为 4940 step）失败：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 124.00 MiB.
GPU 5 total capacity 79.25 GiB, only 10.81 MiB free.
PyTorch allocated 77.20 GiB, reserved but unallocated 1.10 GiB.
```

这次错误发生在 `InternVLAA15._compute_video_loss()` 内的 WAN VAE
`encode_video()`，不是上一轮的 Qwen 主干反向激活峰值。原因是每卡 batch=16
时，VAE 即使在 `torch.no_grad()` 中仍需为 16 个视频同时保留编码中间工作区；
gradient checkpointing 不会降低这个无梯度 VAE 前向的峰值。若直接设置
`action_loss_only=true` 虽能绕过 WAN，但会改变本次完整 SFT 的 video
supervision 目标，因此没有采用。

为保留每卡 batch=16、有效 batch=128、76 epoch、4940 step 及原 checkpoint
schedule，新增 WAN video micro-batch：

```text
/home/a26113/SRC/itvlaGp/src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py
  video_micro_batch_size: int = 1

/home/a26113/SRC/itvlaGp/src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py
  _compute_video_loss(): VAE encode_video 与 WAN DiT 按 micro-batch 分块，
  最后拼接 latent/prediction 后计算同一个 mean MSE。

/home/a26113/SRC/itvlaGp/launch/internvla_a15_geop_phase2_finetune_kptsim_8g.sh
  --policy.video_micro_batch_size="${VIDEO_MICRO_BATCH_SIZE:-1}"
```

该改动只降低 WAN 辅助分支的瞬时显存峰值，不改变样本数量、loss 定义或
优化器有效 batch；代价是正式训练会变慢。代码检查命令：

```bash
cd /home/a26113/SRC/itvlaGp
/home/a26113/VENV/itnvla15rbt20/bin/python -m py_compile \
  src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py \
  src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py
bash -n launch/internvla_a15_geop_phase2_finetune_kptsim_8g.sh
```

以上检查通过。下一次重试命令如下，继续沿用本机 XFS 数据与 Triton cache：

```bash
cd /home/a26113/SRC/itvlaGp
NUM_WORKERS=0 VIDEO_MICRO_BATCH_SIZE=1 \
  bash b/s/rbt/run_each_rbt_p012.sh \
  --config /home/a26113/Cfg/itvlaGp_rbt_batch1.env \
  --from sft --until sft
```

### 2026-08-27 22:10 — 降 batch 后 `place_bread_skillet` 正式 SFT 已越过首步

本次运行使用：

```text
CKPT_ROOT=/tmp/itvla-ckpt
WARMUP_CKPT=/tmp/itvla-warmup-ckpts/place_bread_skillet-000400
BATCH_SIZE=4, PROC_PER_NODE=8, effective batch=32
VIDEO_MICRO_BATCH_SIZE=1
SFT_EPOCHS=76, total steps=19684
save steps=4921, 9842, 14763, 19684
```

smoke 已成功保存 000001；正式训练已成功完成至少 300 step，日志中
`loss_video`、`loss_kpt_cur`、`loss_kpt_fut` 均正常出现，且没有出现新的
OOM、视频解码错误或 Ceph I/O 阻塞。例如 step 300：

```text
0.49 iters/s | step:300 | epoch:1.16
loss:2.316 | loss_action:0.050 | loss_vqa:1.507
loss_video:0.307 | loss_kpt_cur:0.0017 | loss_kpt_fut:0.0022
```

这证明 WAN VAE/DiT micro-batch、gradient checkpointing、本地 checkpoint
读取和本地 output 写入共同生效。训练仍在后台继续，完成后再验收四个保存点、
`sft/latest` 和第二个任务。

### 2026-08-27 21:34 — smoke 保存目录仍在 Ceph，训练 checkpoint 根切换到本地

使用本地 warmup checkpoint 的重试已确认：

```text
WARMUP_CKPT=/tmp/itvla-warmup-ckpts/place_bread_skillet-000400
```

但 smoke checkpoint 写入原 `/home/a26113/Ckp/itvlaGp/...` 时进程进入
`netfs_write_begin`，说明输出文件系统也会造成阻塞。停止该 smoke 后修改：

```text
/home/a26113/Cfg/itvlaGp_rbt_batch1.env
  CKPT_ROOT=/tmp/itvla-ckpt
```

原因：SFT 的运行目录、checkpoint、日志和 pipeline state 全部先写本机
XFS；Phase 0 已完成且 v3 数据使用 `/tmp/RoboTwin-Clean-v30`，不会影响
数据内容或任务隔离。两任务完成后，按任务分别将成功的 `warmup/`（如需）、
`sft/`、日志和 `pipeline_state.json` 归档到持久 Ceph 路径
`/home/a26113/Ckp/itvlaGp/<task>/`，归档动作另行记录。

停止与重试的依据：

```bash
ps -eo pid,ppid,stat,etime,wchan:24,cmd | rg 'lerobot_train|accelerate'
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
```

### 2026-08-27 21:43 — 本地 XFS 后正式首步仍 OOM，降低每卡 batch 并重算 schedule

第三次正式尝试已从本地 XFS checkpoint 和本地 XFS 输出目录启动，绕过了
Ceph 读取/写入阻塞；smoke 成功保存。正式首个 batch 的错误为：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 216.00 MiB.
GPU 4 total capacity 79.25 GiB, only 82.81 MiB free.
PyTorch allocated 77.79 GiB, reserved but unallocated 450.17 MiB.
```

调用栈位于 `_compute_video_loss()` 的 WAN `time_projection`。这说明 WAN
micro-batch 已成功避免 VAE 的 124 MiB OOM，但每卡 batch=16 的 Qwen/主模型
激活仍将显存占到约 77.8 GiB，WAN 前向的最小额外工作区也无法分配。

由于该训练器没有 gradient accumulation 参数，不能在不改训练器语义的情况下
维持 effective batch=128；直接关闭 video loss 会改变本次 SFT 目标。故采用
保留全部 loss、按 epoch 语义精确重算 steps 的安全方案：

```text
/home/a26113/Cfg/itvlaGp_rbt_batch1.env
  BATCH_SIZE=4
  SFT_EFFECTIVE_BATCH_TARGET=32
```

有效 batch 从 `8*16=128` 变为 `8*4=32`，`compute_sft_steps.py` 会据每个
任务的 total_frames 与 76 epoch 自动重新计算总 steps、save frequency 和
scheduler warmup；总训练样本覆盖量（epoch）不变。下次命令：

```bash
cd /home/a26113/SRC/itvlaGp
NUM_WORKERS=0 VIDEO_MICRO_BATCH_SIZE=1 \
  bash b/s/rbt/run_each_rbt_p012.sh \
  --config /home/a26113/Cfg/itvlaGp_rbt_batch1.env \
  --from sft --until sft
```

### 2026-08-27 21:28 — 正式重试初始化阶段访问 Ceph checkpoint，切换为本地副本

修复 dtype 后的重试中，1-GPU smoke 在约 1 step 内成功并保存：

```text
/home/a26113/Ckp/itvlaGp/place_bread_skillet/sft/2026_08_27_17_18_40-internvla_a1_5-geop-kpt-sft-place_bread_skillet_smoke/checkpoints/000001
```

随后正式 8 卡启动，但各 rank 长时间停留在加载自己的 warmup checkpoint，
GPU 尚未进入训练；源 checkpoint 位于 CephFS，每个约 5.9 GiB。为避免再次
触发并发 Ceph 元数据/读取瓶颈，停止这次尚未执行训练 step 的正式进程，并
执行：

```bash
mkdir -p /tmp/itvla-warmup-ckpts
cp -a /home/a26113/Ckp/itvlaGp/place_bread_skillet/warmup/latest/checkpoints/000400/pretrained_model \
  /tmp/itvla-warmup-ckpts/place_bread_skillet-000400
cp -a /home/a26113/Ckp/itvlaGp/pick_dual_bottles/warmup/latest/checkpoints/000400/pretrained_model \
  /tmp/itvla-warmup-ckpts/pick_dual_bottles-000400
du -sh /tmp/itvla-warmup-ckpts/*
```

结果：两个本地 checkpoint 均约 5.9 GiB。新增配置：

```text
/home/a26113/Cfg/itvlaGp_rbt_batch1.env
  LOCAL_WARMUP_CKPT_ROOT=/tmp/itvla-warmup-ckpts
  VIDEO_MICRO_BATCH_SIZE=1

/home/a26113/SRC/itvlaGp/b/s/rbt/lib.sh
  warmup_ckpt_path() 优先查找
  ${LOCAL_WARMUP_CKPT_ROOT}/${TASK_NAME}-000400
```

这样仅改变 checkpoint 读取位置，不改变权重内容、任务隔离、训练 batch 或
SFT schedule。后续命令仍为：

```bash
cd /home/a26113/SRC/itvlaGp
NUM_WORKERS=0 VIDEO_MICRO_BATCH_SIZE=1 \
  bash b/s/rbt/run_each_rbt_p012.sh \
  --config /home/a26113/Cfg/itvlaGp_rbt_batch1.env \
  --from sft --until sft
```

### 2026-08-27 21:18 — micro-batch 重试先发现 dtype 回归并修复

上述重试在 `place_bread_skillet` 的 1-step smoke 阶段没有进入正式训练，
报错：

```text
RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same
```

根因是首次把 WAN DiT 前向改成列表推导的 micro-batch 时，遗漏了原有包围
`wan_dit_forward()` 的 `torch.amp.autocast("cuda", dtype=wan_dtype)` 上下文；
因此 `noisy_latent` 保持 float，而 WAN `patch_embedding` 的 bias 为
bfloat16。该错误与数据和显存无关。

修复文件：

```text
/home/a26113/SRC/itvlaGp/src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py
```

修复措施：将整个 WAN micro-batch 列表计算重新置于原有 CUDA bfloat16
autocast 上下文中。随后重新执行同一条命令：

```bash
cd /home/a26113/SRC/itvlaGp
NUM_WORKERS=0 VIDEO_MICRO_BATCH_SIZE=1 \
  bash b/s/rbt/run_each_rbt_p012.sh \
  --config /home/a26113/Cfg/itvlaGp_rbt_batch1.env \
  --from sft --until sft
```

### 2026-08-27 22:35 — batch=4 训练在 step 300 后无输出，停止并准备解码复核

本次运行的关键结果：

```text
正式启动：2026_08_27_17_46_33
effective batch=32
total steps=19684
save steps=4921,9842,14763,19684
```

在本机 XFS 上，显存稳定约 58.2 GiB/卡，且 step 50、100、150、200、
250、300 均正常完成；step 300 的 loss 包含：

```text
loss=2.316 loss_action=0.050 loss_vqa=1.507
loss_video=0.307 loss_kpt_cur=0.0017 loss_kpt_fut=0.0022
```

之后约 20 分钟没有新的 step 日志，所有 8 个 rank 均处于
`futex_wait_queue_me`，GPU 利用率降至 0%，输出文件 mtime 停在 step 300。
因此这不是 OOM，而是某个 rank 在数据/视频读取或 DDP 同步点无进展。已用
进程树 TERM 停止该 run；失败 run 目录保留在：

```text
/tmp/itvla-ckpt/place_bread_skillet/sft/2026_08_27_17_46_33-internvla_a1_5-geop-kpt-sft-place_bread_skillet
```

当前结论：WAN micro-batch 和 batch=4 已解决显存问题，但两个任务 SFT 尚未
完成，不能宣称成功。下一步应先用同一 local-XFS 数据做短步复现，重点检查
step 300 后的数据集边界/视频文件和 `NUM_WORKERS=0` 的迭代行为；确认后再
启动正式长跑。此前已记录的 21:18、21:28、21:34、21:43 条目按实际发生顺序
解释本次重试链，后续新增记录继续追加于本文末尾。

### 2026-08-27 22:48 — 两 epoch 短步复现通过，排除 step 300 固定边界故障

为区分瞬时分布式阻塞与数据集边界问题，使用同一套本地 XFS 数据、
本地 warmup checkpoint、`BATCH_SIZE=4` 和 `VIDEO_MICRO_BATCH_SIZE=1`，
仅将 epoch 临时覆盖为 2，并跳过 smoke：

```bash
cd /home/a26113/SRC/itvlaGp
NUM_WORKERS=0 VIDEO_MICRO_BATCH_SIZE=1 \
  bash b/s/rbt/run_each_rbt_p012.sh \
  --config /home/a26113/Cfg/itvlaGp_rbt_batch1.env \
  --tasks place_bread_skillet --from sft --until sft \
  --sft-epochs 2 --skip-smoke
```

该复现成功跑过原先停滞的 step 300、400，并完成整个 518 step；在 epoch
边界保存了 259 和 518 两个 checkpoint：

```text
post_check: video_decode_error=0 using_zeros=0 exit=0
Phase2 完成 OUTPUT_DIR=/tmp/itvla-ckpt/place_bread_skillet/sft/2026_08_27_18_21_26-internvla_a1_5-geop-kpt-sft-place_bread_skillet
```

因此没有发现固定的视频文件或 epoch 边界错误；此前 step 300 停滞属于
一次性分布式运行异常，短步复现已验证完整 forward/backward、video loss、
keypoint loss 和 checkpoint 写入均正常。该临时 2-epoch 目录仅作为诊断
产物，不作为 76-epoch 最终模型。正式任务将重新从各自的 400-step warmup
启动，继续使用 76 epoch、effective batch=32、各任务独立 schedule。

### 2026-08-27 22:51 — 正式双任务命令误触发诊断 latest 跳过，改用 no-skip-existing

短步诊断成功后保留了该任务的 `sft/latest` 符号链接。直接启动正式双任务
命令时，编排器按默认 `SKIP_EXISTING=1` 将 `place_bread_skillet` 误判为
已完成并跳过；它随后刚开始 `pick_dual_bottles` 的正式初始化，尚未产生
训练 step。为避免把 2-epoch 诊断产物当成 76-epoch 结果，已停止该命令。

后续正式命令必须显式使用 `--no-skip-existing`，它会保留诊断目录并为两
任务创建新的时间戳输出目录：

```bash
cd /home/a26113/SRC/itvlaGp
NUM_WORKERS=0 VIDEO_MICRO_BATCH_SIZE=1 \
  bash b/s/rbt/run_each_rbt_p012.sh \
  --config /home/a26113/Cfg/itvlaGp_rbt_batch1.env \
  --from sft --until sft --sft-epochs 76 \
  --skip-smoke --no-skip-existing

### 2026-08-27 22:52 — 正式双任务 76-epoch SFT 已按正确参数启动

最终启动命令：

```bash
cd /home/a26113/SRC/itvlaGp
NUM_WORKERS=0 VIDEO_MICRO_BATCH_SIZE=1 \
  bash b/s/rbt/run_each_rbt_p012.sh \
  --config /home/a26113/Cfg/itvlaGp_rbt_batch1.env \
  --from sft --until sft --sft-epochs 76 \
  --skip-smoke --no-skip-existing
```

启动日志确认没有跳过第一个任务，当前顺序为
`place_bread_skillet` → `pick_dual_bottles`。当前 schedule：

```text
place_bread_skillet: 8277 frames, effective_bs=32, 19684 steps,
  save=4921,9842,14763,19684
pick_dual_bottles: 6129 frames, effective_bs=32, 14592 steps,
  save=3648,7296,10944,14592
```

两任务均从各自 `/tmp/itvla-warmup-ckpts/<task>-000400` 启动，输出写入
`/tmp/itvla-ckpt/<task>/sft/`；正式训练不再使用诊断 latest。
```
