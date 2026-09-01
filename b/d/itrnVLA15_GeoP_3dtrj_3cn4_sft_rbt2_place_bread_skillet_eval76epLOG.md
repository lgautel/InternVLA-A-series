# itvlaGp RoboTwin `place_bread_skillet` 评估执行日志（SFT 76-epoch @019684）

> **目的**：评估 GCS 上 `place_bread_skillet` SFT（76 epoch）训练 **第 19684 步**（末步）checkpoint 的 RoboTwin 2.0 成功率。
>
> **参考手册**：[`itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_eval.md`](itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_eval.md)（同结构：GCS 下载 → 预检 → 冒烟 → 双卡正式评测 → 汇总），本次任务改为 `place_bread_skillet`。
>
> **评测脚本**：[`b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh`](../s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh)
>
> **训练侧参考**：[`b/d/rbt/sft0827.md`](rbt/sft0827.md)（本 checkpoint 的训练手册，训练机为 `a26113`，非本机）

---

## 0. 评测对象

| 项 | 值 |
|:---|:---|
| **GCS checkpoint** | `gs://physical-ai-data-eu/VENV/tmp/itvlaGpS_plcBrdSkl0828/sft-output/2026_08_27_18_51_08-internvla_a1_5-geop-kpt-sft-place_bread_skillet/checkpoints/019684/pretrained_model/` |
| **任务** | `place_bread_skillet`（`TASK_NAMES` 索引 **23**） |
| **训练步数** | `019684`（SFT 末步，对应 76 epoch，`scheduler_decay_steps=19684`） |
| **训练用 repo_id** | `place_bread_skillet_kptsim_lrbv30`（见 `train_config.json` `dataset.repo_id`） |
| **动作模式** | `abs` |
| **kpt 坐标** | `voxel`（`enable_keypoint_predictor=true`, `num_keypoint_joints=14`） |
| **推理后端** | `standard`（`train_config.json` 内 `inference_backend="standard"`） |
| **dtype** | `bfloat16` |
| **步数上限** | 500（`task_config/_eval_step_limit.yml` 已有 `place_bread_skillet: 500`） |
| **训练机（非本机）** | `a26113`（见 [`sft0827.md`](rbt/sft0827.md) §1.3，`CLEAN_ROOT=/home/a26113/Dta/RoboTwin-Clean`） |
| **本机代码库** | `/home/luogang/SRC/Robot/itvlaGp` |
| **Python 环境** | conda `itvlaGp`（`/home/luogang/miniforge3/envs/itvlaGp`，Python 3.10.20） |

---

## 手册与脚本

| 项 | 路径 |
|----|------|
| 参考手册 | `/home/luogang/SRC/Robot/itvlaGp/b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_eval.md` |
| 评测脚本 | `/home/luogang/SRC/Robot/itvlaGp/b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh` |
| 训练手册（本 ckpt 来源） | `/home/luogang/SRC/Robot/itvlaGp/b/d/rbt/sft0827.md` |
| RoboTwin 任务源码 | `/home/luogang/SRC/Robot/itvlaGp/third_party/RoboTwin/envs/place_bread_skillet.py` |
| SAPIEN 关键点提取器（GeoPredict） | `/home/luogang/SRC/Robot/GeoPredict/b/script/kpt/run_extract.py` |
| inference 入口 | `/home/luogang/SRC/Robot/itvlaGp/evaluation/RoboTwin/inference.py` |

## 问题记录（报错 → 根因 → 修复 → 验证）

### Problem #1：本机缺少 `place_bread_skillet_kptsim_lrbv30` 的 `keypoints_meta.json`

| 项 | 内容 |
|----|------|
| **发现时机** | 启动评测前的例行核对（参照 hanging_mug / scan_object 均已在本机有对应 `*_kptsim_lrbv30/meta/keypoints_meta.json`） |
| **症状** | `ls /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/` 中只有 `place_bread_skillet`（原始 v2.1，无 kpt 列），**没有** `place_bread_skillet_kptsim` / `_kptsim_lrb` / `_kptsim_lrbv30`；而 `hanging_mug_kptsim_lrbv30`、`scan_object_kptsim_lrbv30`、`stack_bowls_three_kptsim_lrbv30` 均已存在 |
| **根因分析** | 1) 本 checkpoint 的 SFT 训练是在**另一台机器**（`a26113`，见 [`sft0827.md`](rbt/sft0827.md) `CLEAN_ROOT=/home/a26113/Dta/RoboTwin-Clean`）上完成的，训练时生成的 `place_bread_skillet_kptsim_lrbv30`（含 SAPIEN FK 提取的 `keypoints_meta.json`）留在了那台机器，只有 checkpoint 本身被同步到了 GCS。2) `inference.py` 做 voxel 坐标推理时只需要 `keypoints_meta.json` 里的 `coord_offset`（`load_kptsim_coord_offset()`），此文件由 GeoPredict 的 `b/script/kpt/run_extract.py`（SAPIEN FK）对**原始** `place_bread_skillet`（v2.1，`observation.state`）计算得到，属于**确定性算法**（`compute_auto_offset` = 全局 workspace 中心 − 体素盒中心），只要本机的原始 `place_bread_skillet` 数据集与训练机上一致（均来自同一份 RoboTwin-Clean 参考数据），重新在本机跑一遍提取即可得到**完全相同**的 `coord_offset` |
| **修复措施** | 用本机 conda `itvlaGp`（已含 SAPIEN 3.0.0b1，此前 hanging_mug/scan_object 预检 `[7] sapien` 均通过）跑 `GeoPredict/b/script/kpt/run_extract.py`，输出到 `place_bread_skillet_kptsim/`，再把生成的 `keypoints_meta.json` 放到 `place_bread_skillet_kptsim_lrbv30/meta/`（与 `--kpt-data-root` + `--kpt-variant=kptsim_lrbv30` 默认拼路径一致，`eval.sh` 无需额外传 `--kpt-meta`）。**不需要**跑完整的 Phase0（inject 3D kpt 列 + v2.1→v3.0 转换），因为评测只读取 `coord_offset`，不加载训练数据集本身 |
| **验证** | 见下方「SAPIEN 提取」时间线条目 + `preflight [13] kpt meta` |

## 文件增删改记录

| 时间 | 文件/目录 | 操作 | 缘由 |
|------|------|------|------|
| 2026-08-29 | `b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_place_bread_skillet_eval76epLOG.md` | 新建 | 本次评测执行日志 |
| 2026-08-29 | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/place_bread_skillet_kptsim/` | 新建（SAPIEN FK 提取输出） | Problem #1 修复：生成 `keypoints_meta.json`（`coord_offset`）+ 50 episode `keypoints.npy`（provenance，非训练必需） |
| 2026-08-29 | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/place_bread_skillet_kptsim_lrbv30/meta/keypoints_meta.json` | 新建（复制自上一行输出） | 匹配 `eval.sh` 默认 `--kpt-data-root` + `--kpt-variant=kptsim_lrbv30` 拼路径规则，使 `--kpt-meta` 可省略 |

## 操作命令记录

### 命令 2026-08-29 — SAPIEN FK 关键点提取（修复 Problem #1）

**理由**：本机原始 `place_bread_skillet` 数据集只有 `observation.state`（14 维关节角），需用 SAPIEN 正向运动学算出 14 个关键点的 world 坐标，再自动计算体素空间 `coord_offset`（`compute_auto_offset` = workspace 中心 − 体素盒中心，纯确定性算法），供 `inference.py` voxel 模式推理时对齐训练坐标系。

```bash
source /home/luogang/miniforge3/etc/profile.d/conda.sh
conda activate itvlaGp
cd /home/luogang/SRC/Robot/GeoPredict
python b/script/kpt/run_extract.py \
  --dataset_dir /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/place_bread_skillet \
  --urdf_path /home/luogang/share/zwy/Projects/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf \
  --output_dir /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/place_bread_skillet_kptsim
```

**结果**：50/50 episode 提取成功（8277 帧，与 [`sft0827.md`](rbt/sft0827.md) 记录的本机源数据帧数一致）；`Range validation: PASS`，`out_of_range_count=0`；

`coord_offset = [-0.7938639521598816, -1.04155695438385, 0.4793882369995117]`

```bash
mkdir -p /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/place_bread_skillet_kptsim_lrbv30/meta
cp /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/place_bread_skillet_kptsim/keypoints_meta.json \
   /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/place_bread_skillet_kptsim_lrbv30/meta/keypoints_meta.json
```

| 2026-08-29 | SAPIEN FK 提取 place_bread_skillet → place_bread_skillet_kptsim | OK 50/50 ep, offset=[-0.7939, -1.0416, 0.4794] |
| 2026-08-29 | 复制 keypoints_meta.json → place_bread_skillet_kptsim_lrbv30/meta/ | OK |

## 时间线 / 操作日志

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-08-29 03:26:09 | 解析配置 place_bread_skillet idx=23 run=itvlaGp_plcBrdSkl_p2_019684 | OK |
| 2026-08-29 03:26:09 | 控制台完整日志 | `/home/luogang/SRC/Robot/itvlaGp/outputs/logs/run_itvlaGp_plcBrdSkl_p2_019684.log` |
| 2026-08-29 03:26:09 | conda activate itvlaGp | OK `/home/luogang/miniforge3/envs/itvlaGp/bin/python` |
| 2026-08-29 03:26:09 | GCS 开始下载 checkpoints/019684/pretrained_model | ... |
| 2026-08-29 03:27:13 | GCS 下载完成 | OK model.safetensors=6321129804 bytes |
| 2026-08-29 03:27:27 | 预检 15 项 | **OK**（含 `[12] ckpt kpt + repo_id` 与 `[13] kpt meta` = `[-0.7939, -1.0416, 0.4794]`，与本机新生成的 `coord_offset` 完全一致，验证 Problem #1 修复成功） |
| 2026-08-29 03:30:42 | 冒烟 2 ep demo_clean | OK exit=0 1S/1F/2mp4 log=`outputs/logs/smoke_itvlaGp_plcBrdSkl_p2_019684.log` |
| 2026-08-29 03:30:42 | 启动 demo_clean GPU0 100 ep | 进行中 |
| 2026-08-29 03:30:42 | 启动 demo_randomized GPU1 100 ep | 进行中 |

**预检说明**：`[4][6]` 的 `Python 3.10 is below the recommended 3.11` 为 transformers/fla 提示，非错误；本机沿用 conda `itvlaGp`（Python 3.10.20），与 hanging_mug/scan_object 评测一致。

---

## 再次运行 2026-08-29 03:26:09

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-08-29 03:26:09 | 解析配置 place_bread_skillet idx=23 run=itvlaGp_plcBrdSkl_p2_019684 | OK |
| 2026-08-29 03:26:09 | 控制台完整日志 | /home/luogang/SRC/Robot/itvlaGp/outputs/logs/run_itvlaGp_plcBrdSkl_p2_019684.log |
| 2026-08-29 03:26:09 | `/home/luogang/SRC/Robot/itvlaGp/b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh` | 修改（评测前） | scan_object 默认 GCS/offset；LOG 增手册/路径/问题/文件/命令节 |
| 2026-08-29 03:26:09 | conda activate itvlaGp | OK /home/luogang/miniforge3/envs/itvlaGp/bin/python |

### 命令 2026-08-29 03:26:09

**理由**：从 GCS 拉取 step-019684 的 pretrained_model 四文件到本机，供预检与 inference 加载

```bash
gcloud storage cp gs://physical-ai-data-eu/VENV/tmp/itvlaGpS_plcBrdSkl0828/sft-output/2026_08_27_18_51_08-internvla_a1_5-geop-kpt-sft-place_bread_skillet/checkpoints/019684/pretrained_model/config.json gs://physical-ai-data-eu/VENV/tmp/itvlaGpS_plcBrdSkl0828/sft-output/2026_08_27_18_51_08-internvla_a1_5-geop-kpt-sft-place_bread_skillet/checkpoints/019684/pretrained_model/stats.json gs://physical-ai-data-eu/VENV/tmp/itvlaGpS_plcBrdSkl0828/sft-output/2026_08_27_18_51_08-internvla_a1_5-geop-kpt-sft-place_bread_skillet/checkpoints/019684/pretrained_model/train_config.json gs://physical-ai-data-eu/VENV/tmp/itvlaGpS_plcBrdSkl0828/sft-output/2026_08_27_18_51_08-internvla_a1_5-geop-kpt-sft-place_bread_skillet/checkpoints/019684/pretrained_model/model.safetensors /home/luogang/SRC/Robot/itvlaGp/outputs-gcs/place_bread_skillet_p2_019684/checkpoints/019684/pretrained_model/
```

| 2026-08-29 03:26:09 | GCS 开始下载 gs://physical-ai-data-eu/VENV/tmp/itvlaGpS_plcBrdSkl0828/sft-output/2026_08_27_18_51_08-internvla_a1_5-geop-kpt-sft-place_bread_skillet/checkpoints/019684/pretrained_model | ... |
| 2026-08-29 03:27:13 | GCS 下载完成 | OK model.safetensors=6321129804 bytes |
| 2026-08-29 03:27:27 | 预检 15 项 | OK |

**预检说明**：项 [4][6] 打印的 `Python 3.10 is below the recommended 3.11` 来自 transformers/fla 导入时的提示，**非错误**；本机 RoboTwin 评测沿用 conda `itvlaGp`（Python 3.10.20），与 hanging_mug / stack_bowls 评测一致。
| 2026-08-29 03:30:42 | 冒烟 2 ep demo_clean | OK exit=0 1S/1F/2mp4 log=/home/luogang/SRC/Robot/itvlaGp/outputs/logs/smoke_itvlaGp_plcBrdSkl_p2_019684.log |
| 2026-08-29 03:30:42 | 启动 demo_clean GPU0 100 ep | ... |
| 2026-08-29 03:30:42 | 启动 demo_randomized GPU1 100 ep | ... |
| 2026-08-29 05:27:37 | demo_clean 100 ep | OK 39/100 = 39.0% |
| 2026-08-29 05:42:00 | demo_randomized 100 ep | OK 29/100 = 29.0% |

## 最终结果 (2026-08-29 05:42:00)

| 配置 | 成功 | 失败 | 总计 | Success Rate |
|------|------|------|------|--------------|
| **demo_clean** | 39 | 61 | 100 | **39.0%** |
| **demo_randomized** | 29 | 71 | 100 | **29.0%** |

**输出路径**:

- `/home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_plcBrdSkl_p2_019684/robotwin/demo_clean/place_bread_skillet/`
- `/home/luogang/SRC/Robot/itvlaGp/outputs/logs/eval_itvlaGp_plcBrdSkl_p2_019684_demo_clean.log`
- `/home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_plcBrdSkl_p2_019684/robotwin/demo_randomized/place_bread_skillet/`
- `/home/luogang/SRC/Robot/itvlaGp/outputs/logs/eval_itvlaGp_plcBrdSkl_p2_019684_demo_randomized.log`

| 2026-08-29 05:42:00 | 汇总写入 /home/luogang/SRC/Robot/itvlaGp/b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_place_bread_skillet_eval76epLOG.md | OK |

---

## 最终批次结论

评测全流程 exit=0，耗时约 2 小时 16 分钟；GCS 下载、15 项预检、2 episode 冒烟、两种配置各 100 episode 正式评测及结果汇总均完成。除 Python 3.10 的推荐版本提示外，未发生 error。

| Checkpoint | demo_clean | demo_randomized |
|:---:|:---:|:---:|
| `019684` | **39.0%**（39/100） | **29.0%**（29/100） |

本次评测使用的本机 checkpoint：

`/home/luogang/SRC/Robot/itvlaGp/outputs-gcs/place_bread_skillet_p2_019684/checkpoints/019684/pretrained_model/`
