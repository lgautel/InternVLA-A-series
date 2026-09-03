# itvlaGp RoboTwin stack_bowls_three 评估 — V2 执行日志

> 本文档记录按照 V2 操作手册（`reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2.md`）执行代码修改、测试验证和环境搭建的全过程。

---

## 时间线 / 操作日志

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-08-05 14:30 | `conda create -n itvlaGp python=3.10 -y` | OK, Python 3.10.20 |
| 2026-08-05 14:31 | `pip install torch torchvision torchaudio --index-url .../cu128` | OK, torch=2.11.0+cu128 |
| 2026-08-05 14:32 | `pip install -e ".[all]"` | OK, internvla-a1-5 1.0.0 installed, BUT transformers=5.14.1 (too new) |
| 2026-08-05 14:33 | `pip install "transformers==5.2.0"` | OK, downgraded to 5.2.0 |
| 2026-08-05 14:33 | Copy Qwen3.5/pi0/pi05 patches to transformers | OK, `Qwen3_5ForConditionalGeneration` import verified |
| 2026-08-05 14:34 | `MAX_JOBS=16 pip install flash-attn --no-build-isolation --no-cache-dir` (background) | 编译中... |
| 2026-08-05 14:34 | `pip install flash-linear-attention==0.5.2 causal-conv1d==1.6.1 --no-build-isolation` | OK |
| 2026-08-05 14:35 | `pip install -r evaluation/RoboTwin/requirements.txt && pip install gymnasium` | OK, 但 numpy 降到 1.26.4 (sapien 依赖) |
| 2026-08-05 14:38 | CuRobo sm_120 检查 | .so 已包含 sm_120 |
| 2026-08-05 14:38 | `pip install -e . --no-build-isolation --no-deps` (CuRobo) | OK |
| 2026-08-05 14:39 | `conda install -c conda-forge "ffmpeg>=7" -y` | OK |
| 2026-08-05 14:40 | SAPIEN urdf_loader.py 补丁 + mplib planner.py 补丁 | OK |
| 2026-08-05 14:40 | `ln -sfn .../RoboTwin third_party/RoboTwin` | OK |
| 2026-08-05 14:41 | **修改一**: `inference.py` check_success 排序 bug 修复 | OK, L392: expert_success=..., L393: maybe_close_env() |
| 2026-08-05 14:41 | **修改二**: `eval.sh` CONDA_ENV 改为 `${CONDA_ENV:-itvlaGp}` | OK |
| 2026-08-05 14:41 | **修改三**: `eval.sh` 添加 `--resize-size "${RESIZE_SIZE}"` | OK |
| 2026-08-05 14:41 | **修改四**: `requirements.txt` 添加 `scipy` | OK |
| 2026-08-05 14:42 | 静态测试首次运行 (v1 regex) | **FAIL** T1.2 — regex 匹配到了第一个 try: 块 (L340, maybe_close_env 函数定义) 而非目标 try: 块 (L387) |
| 2026-08-05 14:43 | 静态测试第二次运行 (v2 regex: 匹配 play_once) | **FAIL** T1.4 — regex 仍然太贪婪, close_env 出现在 L1 |
| 2026-08-05 14:44 | 静态测试第三次运行 (v3: 行号法) | **FAIL** T1.4 — check_success 和 close_env 都在同一行 L390, 因为注释行包含两个关键词 |
| 2026-08-05 14:45 | 静态测试第四次运行 (v4: 跳过注释行) | **PASS** 全部 16/16 通过 |
| 2026-08-05 14:46 | 15 项验证检查表 | 13/15 通过; flash-attn: PENDING (编译中); CuRobo: FAIL |
| 2026-08-05 14:46 | CuRobo 诊断: `ModuleNotFoundError: No module named 'setuptools_scm'` | 根因: CuRobo `__init__.py` 第 55 行调用 `import setuptools_scm` 获取版本号 |
| 2026-08-05 14:46 | `pip install setuptools_scm` | OK, CuRobo import 成功 |
| 2026-08-05 ~15:30 | flash-attn 编译完成 | OK, flash-attn 2.8.3.post1 |
| 2026-08-05 ~15:30 | 15 项验证检查表复查 | 15/15 全 PASS |
| 2026-08-05 21:36 | 集成冒烟测试 #1 (2 episode) | **FAIL** — CuRobo import 失败: `No module named 'warp'` |
| 2026-08-05 21:47 | `pip install warp-lang` | OK |
| 2026-08-05 21:48 | CuRobo import 再次测试 | **FAIL** — `No module named 'yourdfpy'` |
| 2026-08-05 21:48 | `pip install yourdfpy` | OK |
| 2026-08-05 22:30 | 集成冒烟测试 #2 (2 episode) | **FAIL** — CuRobo CUDA kernel: `no kernel image is available for execution on the device` |
| 2026-08-05 22:40 | CuRobo sm_120 诊断: 所有 .so 文件均无 sm_120 段 | 确认根因: .so 未编译 sm_120 |
| 2026-08-05 22:49 | CuRobo 清理: `rm -f *.so && rm -rf build` | OK |
| 2026-08-05 22:49 | CuRobo 重编译: `TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=32 pip install -e . --no-build-isolation --no-cache-dir --force-reinstall --no-deps` | OK, 5个 .so 文件全部含 sm_120 |
| 2026-08-05 22:54 | `pip install setuptools_scm warp-lang yourdfpy` (重编译后重装运行时依赖) | OK |
| 2026-08-05 22:54 | CuRobo import + kinematics 验证 | OK |
| 2026-08-05 22:54 | 集成冒烟测试 #3 (2 episode, stack_bowls_three, demo_clean) | **PASS** — 2/2 成功, 100% success rate, 无 CUDA 或 AttributeError |

---

## 问题记录

### Problem #1: transformers 版本过高

| 项 | 内容 |
|----|------|
| **发现时机** | `pip install -e ".[all]"` 安装完成后 |
| **症状** | transformers 安装为 5.14.1（pyproject.toml 无上限约束） |
| **根因** | `pyproject.toml` 未指定 transformers 版本约束, pip 自动安装最新版 |
| **影响** | 5.2.0 以上版本与 Qwen3.5 补丁不兼容, 会报 `create_causal_mask()` API 错误 |
| **修复** | `pip install "transformers==5.2.0"` 降级后重新复制补丁文件 |
| **验证** | `python -c "import transformers; print(transformers.__version__)"` → 5.2.0 |

### Problem #2: 静态测试 regex 匹配错误 (3 次迭代)

| 项 | 内容 |
|----|------|
| **发现时机** | 首次运行静态测试 |
| **症状** | T1.2 或 T1.4 报 FAIL, 但代码修改已正确应用 |
| **根因** | `inference.py` 有两个 `try:` 块 (L340 和 L387); V2 手册中的测试脚本使用的 regex `re.search(r'try:\s*\n(.*?)(?=\n\s*except )')` 匹配到了第一个 `try:` 块 (L340, `maybe_close_env` 函数定义), 该块不包含 `check_success`。第二次迭代改用 `play_once` 定位, 但 regex 仍然太贪婪。第三次迭代改用行号法, 但注释行 `# check_success() requires self.robot, which close_env() sets to None,` 同时包含 `check_success` 和 `close_env`, 导致两者被判为同一行 |
| **修复** | 最终版 (v4) 使用行号法 + 跳过以 `#` 开头的注释行: `is_comment = stripped.startswith("#")` |
| **验证** | v4 测试 16/16 全部 PASS |
| **手册修正建议** | V2 手册 Section 2.2 的测试脚本需更新: (1) 匹配包含 `play_once` 的 try 块而非首个 try 块, (2) 跳过注释行 |

### Problem #3: CuRobo import 失败 — setuptools_scm 缺失

| 项 | 内容 |
|----|------|
| **发现时机** | 15 项验证检查表第 8 项 |
| **症状** | `ModuleNotFoundError: No module named 'setuptools_scm'` |
| **根因** | CuRobo 的 `__init__.py` 第 55 行在获取版本号时调用 `import setuptools_scm`。以 `--no-deps` 方式安装跳过了此依赖 |
| **影响** | `import curobo` 失败, seed 验证阶段无法运行 |
| **修复** | `pip install setuptools_scm` |
| **验证** | `python -c "import curobo; print('OK')"` 成功 |
| **手册修正建议** | V2 手册 Section 3.8 的 CuRobo 安装步骤应在 `pip install -e . --no-build-isolation --no-deps` 后追加 `pip install setuptools_scm` |

### Problem #4: numpy 降级

| 项 | 内容 |
|----|------|
| **发现时机** | 安装 RoboTwin 依赖时 |
| **症状** | pip 警告 `opencv-python-headless 4.12.0.88 requires numpy<2.3.0,>=2` 但 numpy 降为 1.26.4 |
| **根因** | sapien==3.0.0b1 依赖 numpy<2.0, 与 opencv-python-headless 的 numpy>=2 冲突, pip 优先满足 sapien 的约束 |
| **影响** | 功能上无影响 (opencv-python-headless 实际可在 numpy 1.26 上工作), 但有兼容性警告 |
| **修复** | 无需修复, 仅记录 |

### Problem #5: CuRobo planner 导入失败 — warp 模块缺失

| 项 | 内容 |
|----|------|
| **发现时机** | 集成冒烟测试 #1 运行时 |
| **症状** | `[planner.py]: Something wrong happened when importing CuroboPlanner!` 随后所有 seed 均报 `AssertionError: CuroboPlanner is not imported correctly` |
| **根因** | CuRobo 的 `curobo.geom.transform` 第 20 行执行 `import warp as wp`。CuRobo 以 `--no-deps` 安装, 跳过了 warp 依赖。warp 的 pip 包名为 `warp-lang` 而非 `warp` |
| **影响** | CuroboPlanner 无法初始化, expert rollout 全部失败, 评估流程无法使用专家引导路径 |
| **修复** | `pip install warp-lang` |
| **验证** | `python -c "from curobo.types.math import Pose as CuroboPose; print('OK')"` 成功 |
| **手册修正建议** | V2 手册 Section 3.8 应在 CuRobo 安装步骤后追加 `pip install warp-lang yourdfpy` |
| **完整 traceback** | `File ".../planner.py", line 297: from curobo.types.math import Pose as CuroboPose` → `File ".../curobo/geom/transform.py", line 20: import warp as wp` → `ModuleNotFoundError: No module named 'warp'` |

### Problem #6: CuRobo planner 导入失败 — yourdfpy 模块缺失

| 项 | 内容 |
|----|------|
| **发现时机** | 修复 Problem #5 后再次测试 CuRobo import |
| **症状** | `ModuleNotFoundError: No module named 'yourdfpy'` |
| **根因** | CuRobo 的 `cuda_robot_model/urdf_kinematics_parser.py` 第 21 行 `import yourdfpy`。同样被 `--no-deps` 安装跳过 |
| **影响** | CuRobo 基本 import 成功, 但实际使用 URDF 解析时失败 |
| **修复** | `pip install yourdfpy` |
| **验证** | `python -c "import curobo; from curobo.types.math import Pose; print('OK')"` 成功 |

### Problem #7: CuRobo CUDA kernel 运行时失败 — 无 sm_120 内核镜像

| 项 | 内容 |
|----|------|
| **发现时机** | 集成冒烟测试 #2 运行时 |
| **症状** | `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device` |
| **根因** | CuRobo 的 5 个 .so 文件 (`geom_cu`, `kinematics_fused_cu`, `lbfgs_step_cu`, `line_search_cu`, `tensor_step_cu`) 均无 sm_120 (Blackwell) 的 CUDA 内核。这些 .so 是在其他环境中编译的, 未包含 sm_120 架构支持。`cuobjdump --list-elf` 对所有 5 个 .so 文件均显示 `sm_120 sections=0` |
| **影响** | CuRobo import 成功, 但运行时 (调用 `KinematicsFusedFunction.apply()`) 触发 CUDA 错误。每个 seed 的 expert rollout 全部失败 |
| **修复步骤** | 1) `rm -f src/curobo/curobolib/*.so && rm -rf build` (清理旧编译产物) 2) `TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=32 pip install -e . --no-build-isolation --no-cache-dir --force-reinstall --no-deps` (重编译) 3) `pip install setuptools_scm warp-lang yourdfpy` (重装运行时依赖) |
| **验证** | `cuobjdump --list-elf` 确认所有 5 个 .so 文件包含 sm_120。`python -c "from curobo.curobolib import kinematics; print('OK')"` 成功 |
| **完整 traceback** | `File ".../cuda_robot_generator.py", line 1072: link_pos_seq, link_quat_seq, _ = get_cuda_kinematics(` → `File ".../curobolib/kinematics.py", line 250: link_pos, link_quat, robot_spheres = KinematicsFusedFunction.apply(` → `File ".../kinematics.py", line 73: r = kinematics_fused_cu.forward(` → `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device` |
| **手册修正建议** | V2 手册 Section 3.8 中的 sm_120 检查步骤实际并不可靠 (之前的检查声称有 sm_120 但实际运行时失败), 应改为直接重编译或增加运行时验证步骤 |

---

## 文件增删改记录

### 修改的文件

| 文件 | 修改内容 | 原因 |
|------|---------|------|
| `evaluation/RoboTwin/inference.py` | L387-410: 将 `maybe_close_env()` 移至 `check_success()` 之后; 引入 `expert_success` 变量; 将 `if task_env.plan_success and task_env.check_success()` 改为 `if not expert_success` | **CRITICAL bug**: `close_env()` 将 `self.robot` 设为 None, 随后 `check_success()` 访问 `self.robot` 触发 AttributeError, 导致评估完全无法运行 |
| `evaluation/RoboTwin/eval.sh` L7 | `CONDA_ENV=internvla_a1_5` → `CONDA_ENV=${CONDA_ENV:-itvlaGp}` | 硬编码的 conda 环境名不存在, 改为可覆盖的默认值 |
| `evaluation/RoboTwin/eval.sh` L45 | 在 `--task-idx` 后插入 `--resize-size "${RESIZE_SIZE}"` | RESIZE_SIZE 变量已定义但未传递给 inference.py |
| `evaluation/RoboTwin/requirements.txt` | 末尾添加 `scipy` | `get_keypoints_aloha()` 需要 `scipy.spatial.transform.Rotation` 但 requirements.txt 未列出 |

### 新建的文件

| 文件 | 用途 |
|------|------|
| `b/d/reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG.md` | 本执行日志 |

### 未修改的文件 (已验证正确)

| 文件 | 验证结论 |
|------|---------|
| `src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py` | 3-path MoT 推理完整支持关键点 |
| `src/lerobot/policies/internvla_a1_5/keypoints.py` | TrackEncoder 正确处理可变长度历史 |
| `src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py` | 28 个关键点配置字段完整 |

---

## 测试结果汇总

### 静态测试 (v4, 最终版)

| 测试 ID | 测试内容 | 结果 |
|---------|---------|------|
| T1.1 | play_once() try block found | PASS (line 389) |
| T1.2 | check_success inside try block | PASS (line 392) |
| T1.3 | close_env inside try block | PASS (line 393) |
| T1.4 | check_success BEFORE close_env | PASS (L392 < L393) |
| T1.5 | expert_success variable exists | PASS |
| T1.6 | no bare check_success after try/except | PASS |
| T2.1 | CONDA_ENV uses overridable default | PASS |
| T2.2 | no hardcoded internvla_a1_5 | PASS |
| T3.1 | scipy in requirements.txt | PASS |
| T4.1 | --resize-size in eval.sh | PASS |
| T4.2 | All 6 variables pass-through | PASS (6/6) |
| **总计** | | **16/16 PASS** |

### 15 项验证检查表

| 项 | 检查内容 | 结果 | 备注 |
|----|---------|------|------|
| 1 | Conda env | OK | |
| 2 | PyTorch + CUDA | OK | torch=2.11.0+cu128 |
| 3 | transformers | OK | 5.2.0 |
| 4 | Qwen3.5 patch | OK | |
| 5 | flash-attn | OK | 2.8.3.post1 |
| 6 | flash-linear-attention | OK | |
| 7 | SAPIEN | OK | 3.0.0b1 |
| 8 | CuRobo | OK | 重编译 sm_120 后, 含运行时依赖 |
| 9 | scipy | OK | |
| 10 | RoboTwin link | OK | |
| 11 | stack_bowls_three | OK | |
| 12 | Checkpoint | OK | |
| 13 | Config check | OK | kpt=True, J=14 |
| 14 | check_success fix | OK | L392 < L393 |
| 15 | Disk space | OK | 251G free |

### 集成冒烟测试 (第 3 次, 最终通过)

| 检查项 | 预期 | 实际结果 |
|--------|------|---------|
| 退出码 | 0 | **0** |
| AttributeError | 无 | **无 (0 occurrences)** |
| CUDA errors | 无 | **无 (0 occurrences)** |
| 视频文件 | ≥ 2 个 .mp4 | **2 个 (success_1.mp4, success_2.mp4)** |
| Success rate | ≥ 0% (关键是不崩溃) | **2/2 = 100%** |
| keypoints | 应有 DeprecationWarning | **是 (inference.py L63, L74)** |

**测试输出摘要**：

```
stack_bowls_three | demo_clean
Success rate: 1/1 => 100.0%, current seed: 4300001

stack_bowls_three | demo_clean
Success rate: 2/2 => 100.0%, current seed: 4300002
```

### 集成冒烟测试历史

| 次数 | 时间 | 结果 | 失败原因 | 修复 |
|------|------|------|---------|------|
| #1 | 21:36 | FAIL | CuRobo: `No module named 'warp'` (Problem #5) | `pip install warp-lang` |
| #2 | 22:30 | FAIL | CuRobo CUDA: `no kernel image for sm_120` (Problem #7) | 重编译 CuRobo with `TORCH_CUDA_ARCH_LIST="12.0"` |
| #3 | 22:54 | **PASS** | — | — |

> 注: Problem #6 (yourdfpy 缺失) 在 #1 和 #2 之间发现并修复, 未单独触发一次完整的集成测试。

---

## 总结

所有代码修改已应用并通过验证:

1. **4 项代码修改**: 全部已应用, 静态测试 16/16 PASS
2. **7 个问题**: 全部已解决 (Problems #1-#7)
3. **集成冒烟测试**: 2/2 episode PASS, 100% success rate
4. **环境就绪**: conda `itvlaGp` 可用于完整的 100-episode 评估

### 关键发现

- CuRobo 以 `--no-deps` 安装时, 缺少 3 个运行时依赖 (`setuptools_scm`, `warp-lang`, `yourdfpy`), 需额外安装
- CuRobo 的 .so 文件在 Blackwell GPU (sm_120) 上必须重编译, 即使之前的检查工具 (cuobjdump) 可能给出误导性结果
- `inference.py` 的 `check_success` 排序 bug 是导致评估完全无法运行的关键问题, 修复后评估正常运行
