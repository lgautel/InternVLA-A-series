# Libero-plus evaluation analysis
_Exported on 9/18/2026 at 10:07:04 GMT+8 from Cursor (3.15.19)_

---

**User**

基于 @itvlaGpLibPlus/b/d/libplus/eval3.md 实现的一个 libero-plus 评估程序正在运行, 它的运行日志记录在 @itvlaGpLibPlus/b/d/libplus/eval3_0915LOG.md 中. 但通过 @itvlaGpLibPlus/evaluation/LIBERO-plus2/check_progress.py  查出来的截止到目前的评估成功率特别低, 请深入分析 @\B\SRC\itvlaGpLibPlus\ 的代码, 特别是 @itvlaGpLibPlus/b/d/libplus/ 中与 libero-plus 训练数据处理, 3D keypoint 轨迹抽取, 基于 libero-plus 做训练与评估等等相关的方案文档, 以及这些方案文档相关的代码脚本. 但最终还是以 @itvlaGpLibPlus/ 这个代码库中的真实代码为准. 对目前 libero-plus 评估出来的 SR (成功率) 特别低的问题进行深入分析, 找找到底是什么原因造成的, 有些什么好的解决方案. 把分析过程和方案写到 @itvlaGpLibPlus/b/d/libplus/eval3_optim2.md 中. 

---

**Cursor**

先从评估方案、运行日志和进度脚本入手，再对照训练/评估相关代码，定位成功率偏低的原因。

评估手册很长，我接着并行读日志、进度脚本和训练/评估相关方案，缩小成功率偏低的可能原因。

日志里 smoke test 已是 0/1，接下来对照全量评估进度、实际评估代码和关键点 FK 实现，核对训推是否一致。

接下来核对实际评估代码、训练关键点抽取，以及当前评估目录里的成功率数字。

已有方向：图像可能被 180° 翻转。接下来核训练数据朝向、官方评估约定、当前 SR，以及关键点/动作管线是否还有其它训推不一致。

已有一份朝向分析；接下来用代码把动作表示、图像映射、关键点历史和腕部相机再核对一遍，避免把次因漏掉。

训练元数据里 EEF 体名是 `gripper0_right_eef`，评估代码写的是 `gripper0_eef`。我去对一下 MJCF 和验证脚本，确认这是不是第二处训推不一致。

---

**User**

LIBERO-plus 评估 SR 偏低：分析文档撰写计划

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

先刷新当前评估成功率，再按计划撰写完整分析文档。

成功率已更新。接着按扰动类别和难度聚合，并核对夹爪符号等关键细节。

数字已齐，开始撰写完整分析文档。

同目录里已有 `eval3_optim.md` 和 `eval3_kpt.md`，我把文档里指到 `../asset/` 的链接改成正确相对路径。

分析已经写进 [`b/d/libplus/eval3_optim2.md`](b/d/libplus/eval3_optim2.md)。未改评估代码，也没有动正在跑的全量评估。

**刷新后的 SR**（`full_20260915_095437`，Crash = 0）：

- libero_spatial：172/2402 = **7.16%**（已跑完）
- libero_object：606/2334 = **25.96%**
- libero_goal：99/546 = **18.13%**
- 合计：877/5282 = **16.60%**（对比论文 Table 6 的 84.8%）

**主因**：评估客户端默认对 live 图像做了 `arr[::-1, ::-1]`，而 `/B/Dta/opvla_libero_merged_kpt/` 的训练帧是 robosuite 原始朝向。像素对照里，评估输入的 MSE（6026）和「再转 180°」的控制组（6017）重合；撤销旋转后降到 508。这解释了七类扰动一起崩、换 checkpoint / 换评估栈 SR 几乎不动、改关键点管线只动约 1 个百分点。

**立刻能做、但不要热改当前 run**：给评估加 `--no_rotate_images`，并在启动脚本里做成 `ROTATE_IMAGES` 配置项。eval3.md 里「训练数据已旋转、不要传这个 flag」与真实代码和磁盘数据不符。

文档里还写了次要问题（resize 空操作、ckpt 只训到 60%、EEF 名/R_pad 文档错误）、夹爪符号待核对、以及先跑标准 LIBERO 再建 LIBERO-plus 的验收门禁。

---

**User**

在  @itvlaGpLibPlus/evaluation/LIBERO2/  和 @itvlaGpLibPlus/evaluation/LIBERO-plus2/ 中的代码已经被别人改动过, 你深入检查一下这些代码, 也包括整个  @itvlaGpLibPlus/ 代码库的代码. 看看你写的  @itvlaGpLibPlus/b/d/libplus/eval3_optim2.md 中提到的问题哪些已经被修复, 哪些尚未被修复. 

---

**Cursor**

先对照 `eval3_optim2.md` 里列出的问题，再逐项核对 LIBERO2 / LIBERO-plus2 和相关评估代码的当前实现。

目录里多了朝向测试和标准 LIBERO 评估脚本，我接着把这些新文件和文档里列出的其余问题逐项对上。

对照 [`eval3_optim2.md`](b/d/libplus/eval3_optim2.md) 的问题清单，以及当前 `evaluation/LIBERO2/`、`evaluation/LIBERO-plus2/` 和相关代码，结论如下。

---

## 已修复（代码里已经落地）

### P0-A：朝向配置项（主因）

[`run_eval_libero_plus_venv.sh`](evaluation/LIBERO-plus2/run_eval_libero_plus_venv.sh) 已增加 `ROTATE_IMAGES="${ROTATE_IMAGES:-false}"`，默认传 `--no_rotate_images`，启动摘要会打印 `Rotation: DISABLED (raw)`。这正是文档方案 A2。

标准 LIBERO 启动器 [`run_eval_libero_std_venv.sh`](evaluation/LIBERO2/run_eval_libero_std_venv.sh) 也硬编码了 `--no_rotate_images`。

### P0-C：T1 腕部相机（原先的开放风险）

新增 [`orientation_contract.py`](evaluation/LIBERO2/orientation_contract.py) 和 [`test_live_orientation.py`](evaluation/LIBERO2/test_live_orientation.py)。实测结果在 [`asset/eval3_optim_wrist_t1.json`](b/d/libplus/asset/eval3_optim_wrist_t1.json)：

- agentview：raw vs train MSE 545.6，rot180 为 6029，比值 11.05（≥5×）
- wrist：raw 1228.1 vs rot180 7992.9，比值 6.51（≥5×）
- 两路 80 帧投票全部 prefer raw

两路相机同一约定，**不需要 per-camera 开关**。

### P0-B：标准 LIBERO 基线（代码已有，验收是否跑过另说）

新增 [`eval_libero_std.py`](evaluation/LIBERO2/eval_libero_std.py) 和 `run_eval_libero_std_venv.sh`，smoke 门槛 50%、full 门槛 85%，对应文档 T2/T3。这是**脚本已具备**；是否已经跑出分布内 SR，代码里看不出来。

### P1-D：resize 空操作

[`policy_backend_internvla_a1_5.py`](evaluation/LIBERO2/policy_server/backends/policy_backend_internvla_a1_5.py) 现在给 `ResizeImagesWithPadFn` 传了 canonical 键：

```107:111:evaluation/LIBERO2/policy_server/backends/policy_backend_internvla_a1_5.py
        self.resize = ResizeImagesWithPadFn(
            height=self.resize_size,
            width=self.resize_size,
            mapping={f"{OBS_IMAGES}.image{i}": f"{OBS_IMAGES}.image{i}" for i in range(3)},
        )
```

评估路径会按 `image0/1/2` 做 256→224，不再是空 mapping。

### P1-E 的一半：replay 跟客户端同一开关

[`eval_libero_plus.py`](evaluation/LIBERO-plus2/eval_libero_plus.py) L96 已从硬编码 `[::-1, ::-1]` 改为 `client._maybe_rotate(...)`。标准 LIBERO 脚本同样如此。

### 护栏测试

新增 [`test_orientation.py`](evaluation/LIBERO2/test_orientation.py)、[`test_preflight_f1f2.py`](evaluation/LIBERO2/test_preflight_f1f2.py)，覆盖 ROTATE_IMAGES 接线、replay、resize mapping、B1/B2/B7/B9/B10 回归。

---

## 尚未修复

### 1. `LiberoModelClient` 默认仍是 `rotate_images=True`（脚枪）

```45:45:evaluation/LIBERO2/model2libero_interface.py
        rotate_images: bool = True,
```

启动脚本现在会传 `--no_rotate_images`，所以**走官方 shell 是安全的**。任何人直接 `LiberoModelClient(...)` 或漏传 flag，就会回到颠倒世界。`test_preflight_f1f2.py` A17 也把这记成 residual。

### 2. Healthcheck / metadata 仍不断言朝向（P1-E 另一半）

两套 shell 的 healthcheck 仍只查 `action_mode == joint` 和 `preprocessing_owner`。  
`InternVLAA15Backend.metadata()` 没有 `rotate_images`、`use_fast_action_tokens`、`image_grid_thw`。  
A18 明确写了：healthcheck 尚未断言旋转。

`use_fast_action_tokens` 仍是 `getattr(config, ..., True)` 的巧合正确，没有固化进 checkpoint `config.json` 或启动断言。

### 3. 文档契约没改（eval3.md 仍在教错误朝向）

[`eval3.md`](b/d/libplus/eval3.md) §3.2 / §10.1 仍写「训练数据已旋转 180°，不要传 `--no_rotate_images`」。代码默认已经反过来了，手册还是旧的。

### 4. EEF body 名仍与训练 metadata 不一致

`keypoint_utils.py` 仍硬编码 `gripper0_eef`。磁盘 `keypoints_meta.json` 是 `gripper0_right_eef`。  
更麻烦的是：`test_preflight_f1f2.py` A10 **把 `gripper0_eef` 当成正确并强制检查**，等于把文档错误写进了回归测试。

### 5. `tests/test_keypoint_utils.py` 仍指向已删除的旧路径

它 `from evaluation.LIBERO.keypoint_utils import ...`，而现在只有 `evaluation/LIBERO2/keypoint_utils.py`。这份单测是坏的，也没有覆盖 StandaloneFK。

### 6. 关键点 history 仍偏 1 帧

评估仍是 `env.step` 之后 `push_keypoint()`，第一次推理时 `his_kpts` 最后一帧是当前姿态；训练 `Extract3DKeypointTransformFn` 的 history 不含当前帧（当前在 `kpt_t`）。文档 §7.4 这条没动。

### 7. T8 夹爪符号、fps=10 vs 20 Hz（P2）

没有 parquet 上 `action[6]` 与 \(g_L\) 的相关脚本。  
没有对照原始 HDF5 步数来核实 fps 元数据。

### 8. P2 换训完的 checkpoint、P3 基座相对 kpt

这不是评估代码能单独修的：要换 `53450` ckpt，以及重生成关键点再 SFT。代码仍维持 StandaloneFK + \(R_{\mathrm{pad}}=1.8213\)，这与「对本 ckpt 不要改 FK」的建议一致。

### 9. 新引入的风险：标准 LIBERO 父进程 import 了 libero

[`eval_libero_std.py`](evaluation/LIBERO2/eval_libero_std.py) 的 `evaluate_policy()` 在 **fork 之前** 执行 `from libero.libero import benchmark`。  
LIBERO-plus2 特意避免这一点（B10：父进程零 EGL）。标准 LIBERO 路径可能在第 2 个 task 再 SIGABRT。plus 评估脚本本身没回退。

---

## 对照总表

| eval3_optim2 条目 | 状态 | 依据 |
|-------------------|------|------|
| A 朝向做成 `ROTATE_IMAGES`，默认不转 | **已修** | venv.sh L31、L168–171 |
| A3 replay 走 `_maybe_rotate` | **已修** | eval_libero_plus.py L96 |
| C T1 agentview + wrist | **已修且实测通过** | wrist_t1.json，两路 ≥5× |
| B 标准 LIBERO 评估脚本 | **代码已有，结果未知** | eval_libero_std.py |
| D server resize 对齐 224 | **已修** | backend mapping image0/1/2 |
| E healthcheck 断言朝向 / grid / use_fast | **未修** | metadata 与 healthcheck 均无 |
| Client 默认 `rotate_images=True` | **未修** | 依赖 shell 传 flag |
| eval3.md 错误朝向说明 | **未修** | §3.2 / §10.1 仍是旧文 |
| EEF `gripper0_eef` vs metadata | **未修**（测试还在加固错误名） | keypoint_utils.py；A10 |
| 旧 `tests/test_keypoint_utils.py` | **未修（已失效）** | import 已不存在的 LIBERO 路径 |
| history 含当前帧 | **未修** | push 仍在 step 之后 |
| T8 夹爪语义 | **未修** | 无 parquet 相关 |
| fps 10 vs env 20 Hz | **未修** | 无核对脚本 |
| 换 53450 ckpt | **未做**（运维） | — |
| 基座相对 kpt + 重训 | **未做**（P3，合理） | — |
| 标准 LIBERO 父进程 EGL | **新风险** | eval_libero_std.py L171 |

**主因（180°）在「走 LIBERO-plus2 / LIBERO2 官方 shell」这条路径上已经接上。** 还没关上的是：Python 默认值、healthcheck、eval3.md、以及关键点 body 名 / history / 夹爪与 fps 这些次要项。若直接实例化 `LiberoModelClient` 而不传 `rotate_images=False`，主因会复发。

---

**User**

在  @itvlaGpLibPlus/evaluation/LIBERO2/  和 @itvlaGpLibPlus/evaluation/LIBERO-plus2/ 中的代码已经被别人改动过, 你深入检查一下这些代码, 也包括整个  @itvlaGpLibPlus/ 代码库的代码. 参考  @itvlaGpLibPlus/b/d/libplus/eval3_optim.md 和  @itvlaGpLibPlus/b/d/libplus/eval3_optim3.md , 再仔细对照检查一下你写的  @itvlaGpLibPlus/b/d/libplus/eval3_optim2.md 中检查出的问题哪些已经被修复, 哪些尚未被修复. 特别是你提到的 `agentview` 和 `腕部相机` 的图片朝向相关的问题, "EEF body 名"的问题, "关键点 history 仍偏 1 帧", "test_keypoint_utils.py 仍指向已删除的旧路径", "T8 夹爪符号、fps=10 vs 20 Hz", "标准 LIBERO 父进程 import 了 libero" 等等问题有没有被修复. 现在 @itvlaGpLibPlus/ 中关于libero-plus的代码是不是已经训推一致了. 然后在  eval3_optim2.md 中补充全面的测试验收代码, 以检查你提到的问题是否还存在, 当然, 这些代码也最好尽量能复用. 然后, 在 eval3_optim.md 列出没解决的问题, 并对这些问题撰写详细的解决方案和实施落地方案, 真对些未解决的问题的方案要细化到代码层面, 并且有测试和验收脚本覆盖. 请检查未解决的问题, 并按照我的意见修改 eval3_optim2.md.

---

**Cursor**

先对照三份分析和当前评估代码，确认每个未解决问题的真实状态，再拟定改文档的方案。

`eval3_optim.md` 已有 U1–U7 落地记录。接着核对文档声称的修复与仓库里的真实代码是否一致。

`eval3_optim.md` 已声称 U1–U7 落地。接下来逐项核对仓库实树，确认朝向、EEF、history、父进程 import 等是否真的修好。

**文档与代码落地范围**

> **本次除了更新 eval3_optim2.md / eval3_optim.md 并补测试脚本之外，是否同步修复仓库里仍未解决的生产代码？（例如：标准 LIBERO 父进程 import benchmark、his_kpts 含当前帧、test_keypoint_utils 导入路径、EEF body 解析、T8/​fps 核对脚本）**
> - 只改文档 + 写/改可复用的检测测试（先不改评估运行路径，测试用于发现问题是否还在）
> - 文档 + 测试 + 同步修余留生产代码（例如 std 父进程不 import libero、history 不含当前帧、修 test_keypoint_utils 导入） ✔
> - 只更新 eval3_optim2.md 与 eval3_optim.md，不新建 Python 测试文件

---

**User**

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

开始按计划落地：先改评估循环里的关键点 history（`push` 必须在 `env.step` 之前）。

先改评估循环、标准 LIBERO 父进程，以及 EEF 解析。

T8 实测 `corr(action[6], next g_L) ≈ −0.84`，与 `libero_native` 一致。接下来补可复用契约模块、单测和预检。

测试已通过。开始更新三份文档：先改 `eval3_optim2.md` 的复审与验收章节。

对照当前仓库后：**走 LIBERO2 / LIBERO-plus2 官方入口时，视觉主契约已经训推一致**（双相机 raw、默认不转、224 resize、`libero_native`、StandaloneFK）。原先点名的余留项已按计划落地，并用预检钉死。

**代码**
- **U8 history**：plus2 / std 改为在 `env.step` **之前** `push_keypoint()`，`his_kpts` 不再含当前帧。
- **U9**：标准 LIBERO 父进程不再 `import libero`，改用 `TASK_SUITE_N_TASKS`。
- **U10**：`resolve_eef_body_name` 与生成脚本同一候选；Lift MJCF 解析为 `gripper0_eef`，不要求与 metadata 的 `gripper0_right_eef` 逐字相同。
- **U11**：`tests/test_keypoint_utils.py` 改导 `evaluation.LIBERO2`。
- 新增可复用 [`train_eval_extra_contract.py`](evaluation/LIBERO2/train_eval_extra_contract.py)，预检增加 A23–A26 / B8 / T8 / T9。

**验收（本次已跑）**
- Part A 69/69；CLIENT Part AB 115/115；SERVER Part C（含 T8/T9）22/22；pytest 14 passed。
- **T8**：\(\mathrm{corr}(a_6, g_L^{\mathrm{next}})=-0.846<0\)，与 `libero_native` 一致，**不翻转夹爪**。
- **T9**：`fps=10`、\(\Delta t=0.1\)、mean episode 123.7，**不是 2× 抽帧**（本机无 HDF5，未做逐步对照）。
- **未跑** T2–T5 正式 SR。

**文档**：[`eval3_optim2.md`](b/d/libplus/eval3_optim2.md) §12–§15 复审与方案；[`eval3_optim.md`](b/d/libplus/eval3_optim.md) §11.11；[`eval3.md`](b/d/libplus/eval3.md) 朝向契约与 §12.11 已改成 raw / history 不含当前帧。

---

**User**

把 @itvlaGpLibPlus/b/d/libplus/eval3_optim2.md 中提到的需要解决但未解决问题按里面的解决方案修复一下, 然后跑里面提到的测试和验收代码(除了要正式跑LIBERO和LIBERO-plus的那些测试)都跑一下, 确保都通过. 若遇到err就fix, 直到测试全部通过为止. 遇到的err机器解决方案要回写更新到 eval3_optim.md 中.

---

**Cursor**

先对照 `eval3_optim2.md` 里仍标为未解决的项和现有测试入口，再逐项修复并跑验收（不含正式 LIBERO / LIBERO-plus）。

代码侧 U8–U11 已在树上。接下来按文档把非正式评估的测试全部跑一遍，失败就修，并把错误写回 `eval3_optim.md`。

T6 文档片段把 `numpy` 图直接交给 `resize_with_pad`（该函数只要 torch），先改成可跑的验收并补进 `test_orientation.py`。smoke 要在有 mujoco 的 client 环境里重跑。

eval3_optim2 里仍需落地的 U8–U11 生产代码已经在树上（history 在 `env.step` 前 push、std 父进程不 `import libero`、EEF resolve、pytest 走 LIBERO2）。本轮按该方案把除 T2–T5 外的门禁全部跑通；唯一需要改代码的失败是 T6 文档片段不可执行。

**本轮补丁：** 在 `evaluation/LIBERO2/test_orientation.py` 增加 Test 6（`test_t6_image_grid_thw`）。`resize_with_pad` 只要 CHW `float` 张量，不能直接吃 HWC `uint8` numpy。error 与改法已写入 `b/d/libplus/eval3_optim.md` §8.4 / §11.12。

**验收（均退出码 0；未跑正式 LIBERO / LIBERO-plus）：**

| 门禁 | 结果 |
|------|------|
| Part A | 69/69 |
| Part AB（CLIENT） | 115/115 |
| Part C（SERVER，含 T8/T9） | 22/22；夹爪 \(\mathrm{corr}=-0.846\)；`fps=10`、`median_dt=0.1` |
| pytest `test_keypoint_utils.py` | 14 passed |
| `test_orientation.py` | t1–t6 PASS；T6 网格均为 `[[1,16,16]]` |
| T1 live EGL | PASS：agentview 11.056×、wrist 6.506×，vote 80:0 raw |

**遇到的 error：**

1. **T6** `AttributeError: ndarray has no unsqueeze` — 已按 torch CHW 路径修好并做成 Test 6。
2. **`smoke_test_libero_fk.py`** 不是 eval3_optim2 门禁：SERVER 无 mujoco、CLIENT 无 pandas。U10 仍由 B5 覆盖，未混装两个 venv。
3. T1 的 Gym *unmaintained* 警告不阻塞（U7 `os._exit`，退出码仍为 0）。

---

**User**

你是所有测试和验收脚本都跑完了么. 如果是, 请把过程中遇到的 error 及其解决方案回写更新到 @itvlaGpLibPlus/b/d/libplus/eval3_optim2.md 中. 然后, 再整体检查一下, 基于现在的代码情况下, 和测试结果下,  eval3_optim2.md 需要做什么更新并更新它. 特别是一些未解决的问题, 要特别地列出来.

---

**Cursor**

先核对这些验收是否都跑过、文档和代码是否一致，再把 error/方案和仍未解决的问题写回 `eval3_optim2.md`。

确认：除 T2–T5 正式评估外，eval3_optim2 列出的门禁都已跑完。接下来会把 error/方案、过时表述，以及仍未解决的问题写回该文档。

除 **T2–T5 正式 LIBERO / LIBERO-plus SR** 外，`eval3_optim2.md` §10 / §14 列出的门禁都已跑完且退出码为 0。error、过时表述，以及**现在仍未解决的问题**已写回该文档。

**已跑通：** Part A 69/69、AB 115/115、C 22/22（含 T8/T9）、pytest 14、`test_orientation.py` t1–t6、T1 live EGL。  
**按你的要求没跑：** 标准 LIBERO 与 LIBERO-plus 正式评估。  
**不是本文门禁：** `smoke_test_libero_fk.py`（venv 拆开：SERVER 无 mujoco，CLIENT 无 pandas）。

**回写位置**

- §14 / §14.1：下午全量复跑结果 + **Error 1**（T6 把 numpy 传给 `resize_with_pad`）与 **Error 2**（smoke 跨 venv）
- §6.2 / §7.1 / §9 / §10：按当前代码改掉「正在跑评估 / 腕部未测 / resize 仍空 mapping / shell 没有 rotate 变量」等过时句
- **新增 §16**：专门列未解决问题

**现在仍未解决（§16，按阻塞程度）**

| 优先级 | 问题 | 含义 |
|--------|------|------|
| **P0** | T2–T5 | 正确朝向下还没有任何 SR；§1 的 16.60% 是已停的错误朝向对照，且不完整 |
| **P1** | ckpt 只到 60%；旧 `evaluation/LIBERO/` 路径；T7 只有静态 fork、没有新评估 Crash 数；旋转 vs 不转 / kpt 开 vs 关的对照没跑 | 会让 T2–T5 的数字难解释 |
| **P3** | kpt 世界系 ≠ live 桌面；T9 找不到源 HDF5 | 不能只改 eval |

朝向、224 resize、夹爪符号、history、std 父进程 EGL、EEF resolve、pytest **不再算未解决**。在 T2–T5 跑完之前，任何 LIBERO-plus 数字都不能当成鲁棒性结论。

