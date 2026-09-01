# RoboTwin 2.0 `place_bread_skillet` 评估执行记录

> 本文按时间顺序记录本次 RoboTwin 2.0 `place_bread_skillet` 评估的实际操作、错误、根因、修复、文件变更、关键路径和最终结果。
>
> 评估对象：`gs://physical-ai-data-eu/VENV/tmp/Rbt2PlcBrdSle0828/place_bread_skillet/ckpt_2608280803/checkpoints/004912/pretrained_model/`
>
> 评估约束：`ivla15` conda 环境、单张 GPU（`CUDA_VISIBLE_DEVICES=1`，因 GPU 0 被 `scan_object` 评估占用）、遵循 `reprd_rbtwn_stackb3_eval6000.md` 中已验证的 InternVLA-A1.5 评估流程。

## 一、时间线 / 操作日志

| 时间 | 操作 | 结果 |
|---|---|---|
| 2026-08-30 17:25（UTC+8） | 阅读 `reprd_rbtwn_stackb3_eval6000.md`、`reprd_rbtwn_scnObj.md`、`reprd_rbtwn_scnObj_evalLOG.md`；核对 GCS checkpoint 与 GPU 状态 | GCS 四件套可访问；GPU 0 被 `scan_object` demo_clean 评估占用（~11 GB）；GPU 1 空闲 |
| 2026-08-30 17:25（UTC+8） | 确认 `place_bread_skillet` 的 task index | `task_idx=23`（`inference.py` 的 `TASK_NAMES`） |
| 2026-08-30 17:26（UTC+8） | 创建本地目标目录并用 `gcloud storage cp` 下载 GCS 四件套 | 成功，约 54 秒，平均吞吐 113.2 MiB/s；本地约 5.1 GiB |
| 2026-08-30 17:26（UTC+8） | 核对下载结果 | `config.type=internvla_a1_5`；`stats.keys=['aloha']`；state/action 均 14 维；`train_config.action_mode=abs` |
| 2026-08-30 17:27（UTC+8） | 在 GPU 1 上启动 2-episode 冒烟评估（`demo_clean`，`task_idx=23`） | 日志：`outputs/logs/smoke_rbtwn2_place_bread_skillet_4912.log`；视频：`outputs/robotwin/rbtwn2_place_bread_skillet_4912_smoke/robotwin/demo_clean/place_bread_skillet/` |
| 2026-08-30 17:29（UTC+8） | 冒烟评估结束 | `EXIT_CODE=0`；2 episodes：1 success + 1 failure（50%）；episode1 117 steps 成功，episode2 500 steps 超时失败 |
| 2026-08-30 17:29（UTC+8） | 启动正式 `demo_clean` 100-episode 评估（GPU 1） | 使用 `ivla15`、`CUDA_VISIBLE_DEVICES=1`、`bfloat16`、`standard` backend、`infer_horizon=20`；日志：`outputs/logs/eval_rbtwn2_place_bread_skillet_4912_demo_clean.log`；视频：`outputs/robotwin/rbtwn2_place_bread_skillet_4912/robotwin/demo_clean/place_bread_skillet/` |
| 2026-08-30 19:34（UTC+8） | 正式 `demo_clean` 100-episode 评估结束 | `EXIT_CODE=0`；100 个 episode 全部产出视频，其中 success 31、failure 69；成功率 **31.0%**；最终 seed 4300354；日志无 `ERROR`/`Traceback` |
| 2026-08-30 19:35（UTC+8） | 启动正式 `demo_randomized` 100-episode 评估（GPU 1） | 日志：`outputs/logs/eval_rbtwn2_place_bread_skillet_4912_demo_randomized.log`；视频：`outputs/robotwin/rbtwn2_place_bread_skillet_4912/robotwin/demo_randomized/place_bread_skillet/` |
| 2026-08-30 23:30（UTC+8） | 正式 `demo_randomized` 100-episode 评估结束 | `EXIT_CODE=0`；100 个 episode 全部产出视频，其中 success 20、failure 80；成功率 **20.0%**；最终 seed 4300334；日志无 `ERROR`/`Traceback` |
| 2026-08-31 11:37（UTC+8） | 运行 `util_scripts/robotwin_result_stats.py` 汇总结果 | 生成 `outputs/robotwin/rbtwn2_place_bread_skillet_4912/results_robotwin.csv`；与视频计数一致 |

## 二、问题记录（报错 → 根因 → 修复 → 验证）

| 编号 | 报错现象 | 根因分析 | 修复方案 | 验证结果 |
|---|---|---|---|---|
| — | 本次评估无阻塞性错误 | 沿用 `stack_bowls_three` 评估时已修复的环境与 `inference.py` seed 验证逻辑；`place_bread_skillet` 全流程未出现 Python traceback 或进程异常退出 | 无需额外修复 | 冒烟、demo_clean、demo_randomized 三次运行均 `EXIT_CODE=0` |

### 非致命提示（未阻塞评估）

| 提示 | 说明 | 处理 |
|---|---|---|
| `missing pytorch3d` | RoboTwin 部分可视化/几何功能可选依赖缺失 | 未安装；不影响 closed-loop 评估 |
| HF Hub 未认证警告 | 未设置 `HF_TOKEN`，访问 Qwen3.5-2B 配置时有速率限制提示 | 模型权重从本地 checkpoint 加载，评估正常完成 |
| flash-linear-attention fast path 不可用 | 回退到 torch 实现 | 推理可完成，仅略慢 |
| `Unexpected key(s) when loading model: [...wan...]` | action-only 推理跳过了 WAN 视频分支权重 | 预期行为（`action_loss_only` 路径） |
| cuRobo `Batch mode enable graph is only supported with num_graph_seeds==1` | cuRobo 图优化警告 | 不影响 episode 执行与成功率统计 |

## 三、文件与其它变更清单

| 路径 / 对象 | 操作 | 原因 |
|---|---|---|
| `b/d/p/reprd_rbtwn_PlcBrdSle_evalLOG.md` | 新增并持续更新本记录文件 | 按用户要求记录本次评估全过程 |
| `/home/luogang/CKPT/VLA/itrnVLA15/rbtwn2/place_bread_skillet/ckpt_2608280803/checkpoints/004912/pretrained_model/` | 新增 4 个 checkpoint 文件 | 保存从 GCS 下载的待测权重 |
| `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/robotwin/rbtwn2_place_bread_skillet_4912_smoke/` | 新增冒烟评估视频目录 | 保留 2-episode 冒烟产物 |
| `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/logs/smoke_rbtwn2_place_bread_skillet_4912.log` | 新增冒烟日志 | 记录冒烟评估过程，`EXIT_CODE=0` |
| `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/robotwin/rbtwn2_place_bread_skillet_4912/` | 新增正式评估结果目录 | 存放 demo_clean / demo_randomized 各 100 个 episode 视频 |
| `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/logs/eval_rbtwn2_place_bread_skillet_4912_demo_clean.log` | 新增正式评估日志 | 记录 demo_clean 100-episode 全过程 |
| `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/logs/eval_rbtwn2_place_bread_skillet_4912_demo_randomized.log` | 新增正式评估日志 | 记录 demo_randomized 100-episode 全过程 |
| `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/robotwin/rbtwn2_place_bread_skillet_4912/results_robotwin.csv` | 新增汇总 CSV | `robotwin_result_stats.py` 自动生成 |

**代码修改**：无（复用 `stack_bowls_three` 评估阶段已修复的 `evaluation/RoboTwin/inference.py`）。

## 四、关键路径

| 用途 | 路径 | 状态 |
|---|---|---|
| GCS checkpoint | `gs://physical-ai-data-eu/VENV/tmp/Rbt2PlcBrdSle0828/place_bread_skillet/ckpt_2608280803/checkpoints/004912/pretrained_model/` | 已确认可访问 |
| 本地 checkpoint | `/home/luogang/CKPT/VLA/itrnVLA15/rbtwn2/place_bread_skillet/ckpt_2608280803/checkpoints/004912/pretrained_model/` | 已下载并核验 |
| 项目根目录 | `/home/luogang/SRC/Robot/InternVLA-A-series/` | 已确认 |
| 环境激活脚本 | `/home/luogang/SRC/Robot/InternVLA-A-series/activate_ivla15.sh` | 已使用 |
| RoboTwin 评估脚本 | `/home/luogang/SRC/Robot/InternVLA-A-series/evaluation/RoboTwin/inference.py` | 已确认；`task_idx=23` |
| 评估结果根目录 | `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/robotwin/rbtwn2_place_bread_skillet_4912/` | 已完成；共 200 个 mp4 |
| demo_clean 视频 | `.../rbtwn2_place_bread_skillet_4912/robotwin/demo_clean/place_bread_skillet/` | 31 success + 69 failure |
| demo_randomized 视频 | `.../rbtwn2_place_bread_skillet_4912/robotwin/demo_randomized/place_bread_skillet/` | 20 success + 80 failure |
| 冒烟日志 | `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/logs/smoke_rbtwn2_place_bread_skillet_4912.log` | 已完成，`EXIT_CODE=0` |
| demo_clean 日志 | `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/logs/eval_rbtwn2_place_bread_skillet_4912_demo_clean.log` | 已完成，`EXIT_CODE=0` |
| demo_randomized 日志 | `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/logs/eval_rbtwn2_place_bread_skillet_4912_demo_randomized.log` | 已完成，`EXIT_CODE=0` |
| 汇总 CSV | `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/robotwin/rbtwn2_place_bread_skillet_4912/results_robotwin.csv` | 已生成 |

## 五、评估参数（与 stack_bowls_three 一致）

| 参数 | 值 |
|---|---|
| checkpoint step | 004912 |
| `task_idx` | 23 (`place_bread_skillet`) |
| `action_mode` | `abs`（与 `train_config` 一致） |
| `infer_horizon` | 20 |
| `inference_backend` | `standard` |
| `dtype` | `bfloat16` |
| `num_episodes` | 100（每配置） |
| GPU | `CUDA_VISIBLE_DEVICES=1` |

## 六、最终结果

| 配置 | Episodes | 成功数 | 成功率 |
|---|---:|---:|---:|
| `demo_clean` | 100 | 31 | **31.0%** |
| `demo_randomized` | 100 | 20 | **20.0%** |

`results_robotwin.csv` 内容：

```csv
names,rbtwn2_place_bread_skillet_4912,
,demo_clean,demo_randomized
Average,31.00% (31/100),20.00% (20/100)
place_bread_skillet,31.00% (31/100),20.00% (20/100)
```

## 七、复现命令

```bash
source /home/luogang/SRC/Robot/InternVLA-A-series/activate_ivla15.sh
export CUDA_VISIBLE_DEVICES=1
CKPT=/home/luogang/CKPT/VLA/itrnVLA15/rbtwn2/place_bread_skillet/ckpt_2608280803/checkpoints/004912/pretrained_model
OUT=${REPO_ROOT}/outputs/robotwin/rbtwn2_place_bread_skillet_4912
cd ${REPO_ROOT}/third_party/RoboTwin

# demo_clean
python -u ${REPO_ROOT}/evaluation/RoboTwin/inference.py \
  --ckpt-path "${CKPT}" \
  --video-dir "${OUT}/robotwin/demo_clean/place_bread_skillet" \
  --task-config demo_clean --task-idx 23 \
  --action-mode abs --infer-horizon 20 \
  --inference-backend standard --num-episodes 100 --dtype bfloat16 \
  > ${REPO_ROOT}/outputs/logs/eval_rbtwn2_place_bread_skillet_4912_demo_clean.log 2>&1

# demo_randomized
python -u ${REPO_ROOT}/evaluation/RoboTwin/inference.py \
  --ckpt-path "${CKPT}" \
  --video-dir "${OUT}/robotwin/demo_randomized/place_bread_skillet" \
  --task-config demo_randomized --task-idx 23 \
  --action-mode abs --infer-horizon 20 \
  --inference-backend standard --num-episodes 100 --dtype bfloat16 \
  > ${REPO_ROOT}/outputs/logs/eval_rbtwn2_place_bread_skillet_4912_demo_randomized.log 2>&1

# 汇总
python ${REPO_ROOT}/util_scripts/robotwin_result_stats.py ${OUT}
```
