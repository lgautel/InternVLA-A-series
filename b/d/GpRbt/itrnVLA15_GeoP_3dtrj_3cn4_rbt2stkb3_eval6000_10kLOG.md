# itvlaGp0801116 RoboTwin stack_bowls_three 010k 评估执行日志

> 评估 **InternVLA-A1.5 + GeoPredict 3-path MoT**（kptsim 体素 GT 训练）在 RoboTwin 2.0 `stack_bowls_three` 上的 step-010000 checkpoint。
>
> 参考计划：体素坐标对齐改造 + demo_clean/demo_randomized 各 100 episode。

---

## 评估配置

| 项 | 值 |
|----|-----|
| **代码库** | `/home/luogang/SRC/Robot/itvlaGp` |
| **Conda 环境** | `itvlaGp` |
| **Checkpoint** | `/home/luogang/SRC/Robot/itvlaGp0801116/p2/checkpoints/010000/pretrained_model` |
| **训练数据** | `stack_bowls_three_kptsim_lrbv30`（体素坐标 GT） |
| **kpt meta** | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three_kptsim_lrbv30/meta/keypoints_meta.json` |
| **任务** | `stack_bowls_three` (task_idx=46) |
| **推理后端** | `standard` (3-path MoT) |
| **kpt 坐标模式** | `voxel`（world − coord_offset） |
| **动作模式** | `abs` |
| **dtype** | `bfloat16` |
| **infer-horizon** | 20 |
| **每配置 episode** | 100 |
| **输出目录** | `outputs/robotwin/itvlaGp0801116_p2_010k/` |

---

## 时间线 / 操作日志

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-08-11 | 用户确认计划，开始执行 | OK |
| 2026-08-11 | 审计：checkpoint 为 kptsim 体素训练；inference.py 仍用 footprint-relative | **需改造** |
| 2026-08-11 | 创建本 LOG 文件 | OK |

---

## 问题记录（报错 → 根因 → 修复 → 验证）


| 2026-08-11 09:20 | 修改 inference.py：体素 kpt 提取 + CLI + load_stats 兼容 mean/std | OK |
| 2026-08-11 09:22 | 静态测试 14/14 PASS（含体素项） | OK |
| 2026-08-11 09:22 | Problem #1: stats.json 无 min/max → load_stats KeyError | 已修复 |
| 2026-08-11 09:27 | 冒烟测试 2/2 ep，voxel kpt min=[0.51,0.37,0.33] max=[1.12,0.83,0.52] | PASS |
| 2026-08-11 09:27 | 启动 demo_clean(GPU0) + demo_randomized(GPU1) 各 100 ep | OK |
| 2026-08-11 09:54 | demo_clean 完成 81/100 = 81.0% | OK |
| 2026-08-11 10:56 | demo_randomized 完成 57/100 = 57.0% | OK |

## 问题记录（报错 → 根因 → 修复 → 验证）

### Problem #1: load_stats KeyError 'min'

| 项 | 内容 |
|----|------|
| **发现时机** | 首次冒烟测试 |
| **症状** | `KeyError: 'min'` in load_stats |
| **根因** | itvlaGp0801116 checkpoint 的 stats.json 仅含 mean/std/q01/q99，无 min/max |
| **修复** | load_stats 改为按需读取 mean/std，q01/q99 作 min/max fallback |
| **验证** | 冒烟测试通过 |

### Problem #2: 训练/推理坐标系不一致（设计问题）

| 项 | 内容 |
|----|------|
| **根因** | 训练用 kptsim 体素 GT，原 inference 用 footprint-relative + camera EEF |
| **修复** | 新增 get_keypoints_kptsim_voxel()：world−offset + TCP EEF |
| **验证** | 首帧 kpt 值域 ∈ [0.33, 1.12]，符合体素空间 |

## 文件增删改记录

| 文件 | 操作 | 原因 |
|------|------|------|
| evaluation/RoboTwin/inference.py | 修改 | 体素 kpt 提取、CLI、load_stats 兼容 |


---

## 文件增删改记录

| 文件 | 操作 | 原因 |
|------|------|------|
| `b/d/itrnVLA15_GeoP_3dtrj_3cn4_rbt2stkb3_eval6000_10kLOG.md` | 新建 | 本次评估执行日志 |
| `evaluation/RoboTwin/inference.py` | 待修改 | kptsim 体素关键点 runtime 提取 |

---

## 最终结果

| 配置 | 成功 | 失败 | 总计 | Success Rate |
|------|------|------|------|--------------|
| **demo_clean** | 81 | 19 | 100 | **81.0%** |
| **demo_randomized** | 57 | 43 | 100 | **57.0%** |

**对比历史 run（仅供参考，checkpoint 不同）**：

| Run | 训练数据 | demo_clean | demo_randomized |
|-----|---------|------------|-----------------|
| LOG1/LOG2 (FK) | stack_bowls_three_kpt | 63-64% | 16-22% |
| LOG0807192 (080719_2) | FK | 58% | 8% |
| **本次 itvlaGp0801116 (kptsim 体素)** | kptsim lrbv30 | **81%** | **57%** |

**结论**：体素坐标对齐改造后评估成功完成。demo_clean 81%、demo_randomized 57%，randomized 相对 clean 降幅 24pp。seed 验证阶段偶发 `AssertionError: target_pose cannot be None`（expert rollout），已被 try/except 跳过，不影响 100 ep 完成。

**输出路径**：
- `outputs/robotwin/itvlaGp0801116_p2_010k/robotwin/demo_clean/stack_bowls_three/` — 100 mp4
- `outputs/robotwin/itvlaGp0801116_p2_010k/robotwin/demo_randomized/stack_bowls_three/` — 100 mp4
- `outputs/logs/eval_itvlaGp0801116_p2_010k_demo_clean.log`
- `outputs/logs/eval_itvlaGp0801116_p2_010k_demo_randomized.log`
