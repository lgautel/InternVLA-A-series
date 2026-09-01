# RoboTwin 2.0 `scan_object` 评估执行记录

> 本文记录按时间顺序保存本次 RoboTwin 2.0 `scan_object` 评估的实际操作、错误、根因、修复、文件变更、关键路径和最终结果。
>
> 评估对象：`gs://physical-ai-data-eu/VENV/tmp/Rbt2ScnObj0828/scan_object/ckpt_2608280458/checkpoints/005024/pretrained_model/`
>
> 评估约束：`ivla15` conda 环境、单张 GPU、RoboTwin 2.0 `scan_object`（task index 41）、遵循 `reprd_rbtwn_stackb3_eval6000.md` 中已验证的 InternVLA-A1.5 评估流程。

## 一、时间线 / 操作日志

| 时间 | 操作 | 结果 |
|---|---|---|
| 2026-08-30 07:48（UTC+8） | 阅读 `b/d/p/reprd_rbtwn_stackb3_eval6000.md` 与 `b/d/p/reprd_rbtwn_scnObj.md`；核对仓库、评估脚本和目标 checkpoint 的 GCS 路径 | 确认使用 `evaluation/RoboTwin/inference.py`，`scan_object` 的 `task_idx=41`；GCS 中存在 `config.json`、`model.safetensors`、`stats.json`、`train_config.json` |
| 2026-08-30 07:48（UTC+8） | 检查正在运行的评估进程、目标本地目录和磁盘空间 | 无正在运行的 RoboTwin 评估；目标 checkpoint 本地目录尚不存在；根分区可用约 238 GB |
| 2026-08-30 07:48（UTC+8） | 检查 `gcloud` 版本及 GCS 对象可访问性 | Google Cloud SDK 582.0.0；目标 GCS 文件列表访问成功 |
| 2026-08-30 07:50（UTC+8） | 执行 `gcloud storage cp --recursive gs://.../pretrained_model /home/luogang/CKPT/VLA/itrnVLA15/rbtwn2/scan_object/ckpt_2608280458/checkpoints/005024/pretrained_model` | **失败**：`Destination URL must name an existing directory`；GCS 源对象可访问，但本地目标目录尚未创建 |
| 2026-08-30 07:50（UTC+8） | 分析下载错误 | `gcloud storage cp` 的本地目标参数必须指向已存在目录；这是本地目录准备问题，与 checkpoint 内容、权限和网络无关 | 
| 2026-08-30 07:51（UTC+8） | 创建本地目标目录，并用 `gcloud storage cp` 复制 GCS 下 4 个 checkpoint 文件 | 成功，耗时约 54 秒，平均吞吐 110.5 MiB/s；本地得到约 5.1 GiB checkpoint |
| 2026-08-30 09:38（UTC+8） | 核对下载结果 | `config.type=internvla_a1_5`；`stats.keys=['aloha']`；state/action 均 14 维；`train_config.action_mode=abs`；四个必需文件均存在 |
| 2026-08-30 09:38（UTC+8） | 在 `ivla15` 中核验评估依赖、GPU 和任务索引 | torch 2.11.0+cu128、transformers 5.2.0、flash-attn 2.8.3.post1、SAPIEN 3.0.0b1 及 RoboTwin 依赖导入成功；GPU 0 为 RTX PRO 6000 Blackwell；确认 `scan_object` 为 task index 41 |
| 2026-08-30 09:39–09:43（UTC+8） | 使用 GPU 0、`demo_clean`、2 episodes、bfloat16、standard backend 运行冒烟评估 | RoboTwin 推理实际完成，产出 1 个 success 和 1 个 failure 视频；但外层 zsh 收尾命令使用变量名 `status` 时触发只读变量错误，进程最终返回 1 |
| 2026-08-30 09:44（UTC+8） | 修正收尾变量后准备重跑冒烟评估 | 命令中的日志重定向引号误配，zsh 报 `unmatched '`；命令在解析阶段退出，未启动 Python 评估、未产生新输出 |
| 2026-08-30 09:47–09:50（UTC+8） | 修正引号和收尾变量后，在 GPU 0、`demo_clean` 上重跑 2-episode 冒烟评估 | 成功退出（`EXIT_CODE=0`）；产出 1 个 success 和 1 个 failure 视频，评估成功率 1/2（50.0%）；无 Python traceback |
| 2026-08-30 09:50（UTC+8） | 检查冒烟日志中的非致命提示 | `missing pytorch3d`、HF 未认证提示、flash-linear-attention fast path 不可用、cuRobo batch graph warning 均未阻止评估；本次评估使用可用的 torch fallback，并正常完成 |
| 2026-08-30 16:38（UTC+8） | 正式启动 `demo_clean` 100-episode 评估 | 使用 `ivla15`、`CUDA_VISIBLE_DEVICES=0`、`bfloat16`、`standard` backend、`infer_horizon=20`；日志和视频分别写入下方关键路径，进程运行中 |
| 2026-08-30 19:05（UTC+8） | `demo_clean` 正式评估完成 | `EXIT_CODE=0`；100 个 episode 全部产出视频，其中 success 45、failure 55；成功率 **45.0%**；最终日志 seed 为 4300487；检查日志无 `ERROR`/`Traceback` |
| 2026-08-30 19:06（UTC+8） | 检查 `demo_clean` 结束后的 GPU、进程和磁盘状态 | Python 评估进程已退出；GPU 无计算进程；根分区仍有约 228 GB 可用，满足继续评估条件 |
| 2026-08-30 19:33（UTC+8） | 检查 randomized 启动前资源 | GPU 1 的独立 `place_bread_skillet` 评估仍在运行；GPU 0 空闲；选择 GPU 0 作为本次唯一评估 GPU；磁盘可用约 228 GB |
| 2026-08-30 19:33（UTC+8） | 正式启动 `demo_randomized` 100-episode 评估 | 使用 `ivla15`、`CUDA_VISIBLE_DEVICES=0`、`bfloat16`、`standard` backend、`infer_horizon=20`；进程运行中 |
| 2026-08-30 20:26（UTC+8） | randomized 的 seed=4300171 出现 expert rollout 错误 | `RuntimeWarning: invalid value encountered in arccos` 后，`transforms3d.mat2quat` 报 `numpy.linalg.LinAlgError: Eigenvalues did not converge`；`inference.py` 捕获后跳过 seed，评估继续 |
| 2026-08-30 20:33（UTC+8） | 监控 randomized 评估进度 | 已完成 36/100，success 10、failure 26；当前 GPU 0 评估进程正常运行，磁盘可用约 228 GB |
| 2026-08-30 21:24（UTC+8） | randomized 的 seed=4300354 出现 expert rollout 错误 | `scan_object.grasp_actor()` 返回空的 pre-grasp target pose，触发 `AssertionError: target_pose cannot be None for move action.`；评估主循环捕获后跳过该无效 seed |
| 2026-08-30 21:34（UTC+8） | 监控 randomized 评估进度 | 已完成 76/100，success 22、failure 54；GPU 0 进程正常运行，磁盘可用约 228 GB |
| 2026-08-30 22:12（UTC+8） | `demo_randomized` 正式评估完成 | `EXIT_CODE=0`；100 个 episode 全部产出视频，其中 success 28、failure 72；成功率 **28.0%**；最终日志 seed 为 4300499；日志中的异常仅为 expert seed 过滤错误 |
| 2026-08-31 11:37（UTC+8） | 在 `ivla15` 中运行 `util_scripts/robotwin_result_stats.py /home/luogang/SRC/Robot/InternVLA-A-series/outputs/robotwin/rbtwn2_scan_object_5024` | 生成最终 `results_robotwin.csv`，统计结果为 `demo_clean=45.0%`、`demo_randomized=28.0%` |
| 2026-08-31 11:37（UTC+8） | 最终产物核验 | clean 视频 45 success + 55 failure；randomized 视频 28 success + 72 failure；两个正式评估日志均有 `EXIT_CODE=0`；磁盘剩余约 231 GB；无残留 scan_object 评估进程 |

### 评估参数

| 参数 | 值 |
|---|---|
| checkpoint | `/home/luogang/CKPT/VLA/itrnVLA15/rbtwn2/scan_object/ckpt_2608280458/checkpoints/005024/pretrained_model/` |
| task | `scan_object`，task index 41 |
| action mode | `abs`（由 `train_config.json` 核验） |
| inference backend | `standard` |
| infer horizon | 20 |
| dtype | `bfloat16` |
| episodes | 每种配置 100 |
| GPU | `CUDA_VISIBLE_DEVICES=0`，单卡 |
| conda 环境 | `ivla15` |
| 评估入口 | `evaluation/RoboTwin/inference.py` |
| seed 起点 | 默认 `--seed 42`，候选 seed 从 4300000 开始 |

## 二、问题记录（报错 → 根因 → 修复 → 验证）

| 编号 | 报错现象 | 根因分析 | 修复方案 | 验证结果 |
|---|---|---|---|---|
| 1 | `ERROR: (gcloud.storage.cp) Destination URL must name an existing directory.` | 本地 checkpoint 目标目录不存在，`gcloud storage cp` 不会自动创建该目录 | 创建目标目录后重新执行 `gcloud storage cp` 下载命令 | 下载成功；四个文件完整，metadata 核验通过 |
| 2 | 外层命令报 `read-only variable: status`；评估日志本身无 Python traceback | zsh 将 `status` 作为只读特殊变量，不能执行 `status=$?`；错误发生在 Python 评估结束后的包装命令，而非评估逻辑 | 将包装命令改为使用普通变量 `rc=$?`，并在新的日志和输出目录中重跑 2-episode 冒烟评估 | 修正后的冒烟评估 `EXIT_CODE=0` |
| 3 | zsh 报 `unmatched '` | 重试命令中日志路径的单引号和双引号不匹配，属于命令拼写错误 | 修正重定向路径的引号后重新执行 | 修正后的冒烟评估 `EXIT_CODE=0` |
| 4 | randomized seed=4300171：`RuntimeWarning: invalid value encountered in arccos`，随后 `numpy.linalg.LinAlgError: Eigenvalues did not converge` | RoboTwin 域随机化产生了数值不稳定的场景姿态，旋转矩阵/四元数计算出现 NaN，expert 的 `mat2quat` 无法完成特征值分解；属于被测环境 seed 的可恢复失败，不是模型推理错误 | 不修改模型或评估代码；沿用 `inference.py` 的 expert seed 验证协议，捕获异常并递增 seed，跳过该无效 seed | 评估继续运行，已完成 36/100；该 seed 不计入有效 episode |
| 5 | randomized seed=4300354：`AssertionError: target_pose cannot be None for move action.` | 域随机化下该场景的 scanner 或其 pre-grasp 位姿无法被 RoboTwin expert 规划器生成，`grasp_actor()` 将空 target pose 传入 `Action`；属于 expert seed 验证失败，不是 policy 推理失败 | 不修改任务逻辑；由 `inference.py` 捕获异常、关闭环境并跳过该 seed | 评估继续运行，已完成 76/100；该 seed 不计入有效 episode |

## 三、文件与其它变更清单

| 路径 / 对象 | 操作 | 原因 |
|---|---|---|
| `b/d/p/reprd_rbtwn_scnObj_evalLOG.md` | 新增本记录文件 | 按用户要求记录本次评估全过程 |
| `/home/luogang/CKPT/VLA/itrnVLA15/rbtwn2/scan_object/ckpt_2608280458/checkpoints/005024/` | 新增目录层级 | 为 `gcloud storage cp` 准备本地目标目录 |
| `/home/luogang/CKPT/VLA/itrnVLA15/rbtwn2/scan_object/ckpt_2608280458/checkpoints/005024/pretrained_model/` | 新增 `config.json`、`model.safetensors`、`stats.json`、`train_config.json` | 保存待测 scan_object checkpoint |
| `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/robotwin/rbtwn2_scan_object_5024_smoke/` | 新增冒烟评估视频目录 | 保留第一次冒烟评估的 success/failure 产物 |
| `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/logs/smoke_rbtwn2_scan_object_5024.log` | 新增冒烟日志 | 记录第一次冒烟评估及外层收尾错误 |
| `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/robotwin/rbtwn2_scan_object_5024_smoke_retry/` | 新增冒烟评估视频目录 | 保存修正包装命令后成功完成的 2-episode 冒烟产物 |
| `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/logs/smoke_rbtwn2_scan_object_5024_retry.log` | 新增冒烟日志 | 记录修正后 `EXIT_CODE=0` 的验证运行 |

## 四、关键路径

| 用途 | 路径 | 状态 |
|---|---|---|
| GCS checkpoint | `gs://physical-ai-data-eu/VENV/tmp/Rbt2ScnObj0828/scan_object/ckpt_2608280458/checkpoints/005024/pretrained_model/` | 已确认可访问 |
| 本地 checkpoint | `/home/luogang/CKPT/VLA/itrnVLA15/rbtwn2/scan_object/ckpt_2608280458/checkpoints/005024/pretrained_model/` | 已下载并核验 |
| 项目根目录 | `/home/luogang/SRC/Robot/InternVLA-A-series/` | 已确认 |
| RoboTwin 评估脚本 | `/home/luogang/SRC/Robot/InternVLA-A-series/evaluation/RoboTwin/inference.py` | 已确认 |
| 评估结果根目录 | `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/robotwin/rbtwn2_scan_object_5024/` | 已生成 |
| clean 评估日志 | `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/logs/eval_rbtwn2_scan_object_5024_demo_clean.log` | 已生成，`EXIT_CODE=0`，约 1.7 MiB |
| randomized 评估日志 | `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/logs/eval_rbtwn2_scan_object_5024_demo_randomized.log` | 已生成，`EXIT_CODE=0`，约 1.8 MiB |
| clean 评估视频 | `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/robotwin/rbtwn2_scan_object_5024/robotwin/demo_clean/scan_object/` | 100 个 mp4（45 success / 55 failure） |
| randomized 评估视频 | `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/robotwin/rbtwn2_scan_object_5024/robotwin/demo_randomized/scan_object/` | 100 个 mp4（28 success / 72 failure） |
| 冒烟评估日志 | `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/logs/smoke_rbtwn2_scan_object_5024.log` | 已生成；评估完成但包装命令收尾失败 |
| 成功冒烟评估日志 | `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/logs/smoke_rbtwn2_scan_object_5024_retry.log` | 已生成；`EXIT_CODE=0` |
| 汇总 CSV | `/home/luogang/SRC/Robot/InternVLA-A-series/outputs/robotwin/rbtwn2_scan_object_5024/results_robotwin.csv` | 已生成；clean 45.0%，randomized 28.0% |
| TensorBoard | 无 | 本仓库该评估入口不产生 TensorBoard 文件；以日志、mp4 和 CSV 为准 |

## 五、关键复现命令

以下是本次实际采用的核心命令；两次正式评估仅将 `--task-config` 和输出子目录分别设为
`demo_clean`、`demo_randomized`，均使用同一个 checkpoint 和 GPU 0。

```bash
# 下载 checkpoint（第一次因目标目录不存在失败；创建目录后使用此命令成功）
mkdir -p /home/luogang/CKPT/VLA/itrnVLA15/rbtwn2/scan_object/ckpt_2608280458/checkpoints/005024/pretrained_model
gcloud storage cp --recursive \
  gs://physical-ai-data-eu/VENV/tmp/Rbt2ScnObj0828/scan_object/ckpt_2608280458/checkpoints/005024/pretrained_model \
  /home/luogang/CKPT/VLA/itrnVLA15/rbtwn2/scan_object/ckpt_2608280458/checkpoints/005024/pretrained_model

source /home/luogang/SRC/Robot/InternVLA-A-series/activate_ivla15.sh
export CUDA_VISIBLE_DEVICES=0
cd /home/luogang/SRC/Robot/InternVLA-A-series/third_party/RoboTwin

python -u /home/luogang/SRC/Robot/InternVLA-A-series/evaluation/RoboTwin/inference.py \
  --ckpt-path /home/luogang/CKPT/VLA/itrnVLA15/rbtwn2/scan_object/ckpt_2608280458/checkpoints/005024/pretrained_model \
  --video-dir /home/luogang/SRC/Robot/InternVLA-A-series/outputs/robotwin/rbtwn2_scan_object_5024/robotwin/demo_clean/scan_object \
  --task-config demo_clean --task-idx 41 --action-mode abs \
  --infer-horizon 20 --inference-backend standard --num-episodes 100 --dtype bfloat16

python -u /home/luogang/SRC/Robot/InternVLA-A-series/evaluation/RoboTwin/inference.py \
  --ckpt-path /home/luogang/CKPT/VLA/itrnVLA15/rbtwn2/scan_object/ckpt_2608280458/checkpoints/005024/pretrained_model \
  --video-dir /home/luogang/SRC/Robot/InternVLA-A-series/outputs/robotwin/rbtwn2_scan_object_5024/robotwin/demo_randomized/scan_object \
  --task-config demo_randomized --task-idx 41 --action-mode abs \
  --infer-horizon 20 --inference-backend standard --num-episodes 100 --dtype bfloat16

python /home/luogang/SRC/Robot/InternVLA-A-series/util_scripts/robotwin_result_stats.py \
  /home/luogang/SRC/Robot/InternVLA-A-series/outputs/robotwin/rbtwn2_scan_object_5024
```

## 六、最终结果

| 配置 | Episodes | 成功数 | 成功率 |
|---|---:|---:|---:|
| `demo_clean` | 100 | 45 | **45.0%** |
| `demo_randomized` | 100 | 28 | **28.0%** |

最终 CSV：

```csv
names,rbtwn2_scan_object_5024,
,demo_clean,demo_randomized
Average,45.00% (45/100),28.00% (28/100)
scan_object,45.00% (45/100),28.00% (28/100)
```

本次任务没有修改 `evaluation/RoboTwin/inference.py` 或其它源码；直接复用了前序
`stack_bowls_three` 评估中已经验证的评估入口和异常 seed 跳过机制。没有新增或修改源码、配置文件；
本文档为记录文件，checkpoint、日志、视频和 CSV 均为评估产生或下载到仓库外/输出目录的运行产物。
