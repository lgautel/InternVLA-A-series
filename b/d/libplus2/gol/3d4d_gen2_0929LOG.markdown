# 3D/4D Keypoint Generation Execution Log — 2026-09-29

> **方案文档**: `b/d/libplus2/gol/3d4d_gen2.markdown`
> **代码目录**: `b/s/libplus2/gol/`
> **数据源**: `/B/Dta/LIBERO/rlds/libero_goal/1.0.0` (RLDS TFRecord, 256 shards, ~18 GB)
> **输出目标**: `/home/a26113/b/Dta/libero_plus_goal_lrb3_4D/`
> **Python 环境**: `/B/VENV/itnvla15rbt20/` (Python 3.11.9, mujoco 3.13.0, lerobot 1.0.0)

---

## 环境确认

| 项目 | 值 |
|------|-----|
| Python | 3.11.9 |
| mujoco | 3.13.0 |
| lerobot | 1.0.0 |
| RLDS reader | `b/d/libplus/ds/asset/rlds_reader.py` (存在) |
| RLDS 数据 | `/B/Dta/LIBERO/rlds/libero_goal/1.0.0/` (256 shards) |
| 输出目录 | `/home/a26113/b/Dta/libero_plus_goal_lrb3_4D/` |
| 工作目录 | `/B/SRC/itvlaGpLibPlus` |

---

## R0: 离线验收 (accept_goal_4d.sh, S1-S3)

**命令**:
```bash
PY=/B/VENV/itnvla15rbt20/bin/python bash b/s/libplus2/gol/accept_goal_4d.sh
```

**结果**: ✅ 全部通过

| 阶段 | 内容 | 结果 |
|------|------|------|
| S1 | MJCF 派生文件一致性 (`panda_goal_table.xml`) | OK |
| S1b | eval wrapper 恰好两处替换 + 拒绝 `ROTATE_IMAGES=true` | OK |
| S2 | gen1 tests (17 项: FK, history packing, generator, eval clock) | 17/17 OK, 10.9s |
| S3 | gen2 tests (37 项: A-G 组) | 37/37 OK, 344.5s |

**总计**: 54/54 tests passed, 约 6 分钟

**详细结果**:
- A 组 (FK 等价, 4 项): GoalTableFK vs StandaloneFK 数值一致; Goal XML vs Lift XML 仅差底座平移 (0.10m in x)
- B 组 (常数一致, 5 项): H=200 与 launch/client/KeypointHistory 一致; panda.yaml 双相机; fps=20; 朝向匹配全局契约
- C 组 (RLDS 朝向, 3 项): agentview/wrist 均为 flipH (即 rot180 of raw); 负对照通过
- D 组 (生成器护栏, 6 项): raw = rot180(RLDS); validate_episode 正确接受/拒绝; --confirm-full-run 拒绝测试开关
- E 组 (评估客户端, 12 项): 时序、历史、活体 FK 护栏、MD5/朝向/覆盖冲突检测全通过
- F 组 (导入钩子, 3 项): 父进程无 mujoco; 子进程拿到子类; 懒加载验证
- G 组 (端到端, 4 项): raw 数据集全检查通过; 契约验证; 负对照拦截

---

## R1: 全量数据生成

**命令**:
```bash
/B/VENV/itnvla15rbt20/bin/python b/s/libplus2/gol/generate_goal_4d.py \
  --dest /home/a26113/b/Dta/libero_plus_goal_lrb3_4D \
  --confirm-full-run --force
```

### Pass 1 结果

✅ **全部 4,243 episodes 通过验证**

| 指标 | 值 | 阈值/预期 |
|------|-----|-----------|
| R_pad | **1.8212723273** | 计算得出（vs Lift default 1.8212722540, 差 7.3e-8） |
| episodes | 4,243 | 4,243 ✅ |
| frames | 512,604 | 512,604 ✅ |
| worst FK-vs-state EEF err | **7.22e-4 m** (0.722 mm) | < 2.0e-3 m (2 mm) ✅ |
| EpisodeValidationError | 0 | 0 ✅ |

**分析**: R_pad 值与 Lift 场景的默认值（1.8212722540）极为接近，差异仅 7.3e-8。这是因为 Goal 场景底座 (-0.66, 0, 0.912) 与 Lift 场景底座 (-0.56, 0, 0.912) 仅在 x 方向差 0.10m，但 R_pad 由全局包围盒的最大绝对值决定，0.10m 的平移对极端位置的影响很小。FK vs EEF 的最大误差仅 0.722mm，远低于 2mm 阈值，说明底座设置正确且 FK 计算准确。

### Pass 2 状态

**运行中**。使用 LeRobot v3.0 的 `LeRobotDataset.create()` + `add_frame()` + `save_episode()` + `finalize()` API 写入数据。视频编码为 SVT-AV1（256×256, 20fps, CRF 30）。

预计总耗时约 4.5-5 小时（基于方案中 30 episode / 2 min 的基准外推）。

