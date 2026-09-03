# RoboTwin 2.0 全量 Phase 0 转换执行日志

> 执行时间: 2026-09-02
> 操作者: Claude Code
> 计划文档: `b/d/rbt/prepare_ech_rbt_p0.md`
> 目标: 将 `/B/Dta/RoboTwin-Clean/` 下所有 v2.1 源任务生成含 kptsim 3D 关键点的 LeRobot v3.0 数据，存入 `{task}_lrb3_kptsim/`

---

## Step 0: 环境准备

### 0.1 初始状态检查

- 磁盘: 12TB 可用（`/B/Dta/RoboTwin-Clean/` 所在分区），空间充裕
- 已有 `_lrb3_kptsim` 目录: 0 个（全新开始）
- Python: `/opt/conda/bin/python3` (3.11)
- 缺少依赖: `sapien`, `transforms3d`, `lerobot`

### 0.2 安装依赖

```bash
pip install sapien transforms3d
# 结果: sapien-3.0.3, transforms3d-0.4.2, opencv-python-5.0.0.93, lxml-6.1.2, pyperclip-1.11.0

pip install -e /B/SRC/itvlaGp
# 结果: internvla-a1-5-1.0.0 及其依赖 (accelerate, datasets, huggingface-hub, wandb 等)
```

验证:
- `import sapien` → OK (Vulkan 警告是正常的，SAPIEN FK 不需要 GPU 渲染)
- `import transforms3d` → OK
- `from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset` → OK

### 0.3 修复 `discover_source_tasks.py`

**问题**: `DERIVED_SUFFIXES` 缺少 `_lrb3` 和 `_lrb3_kptsim`，导致 `_lrb3` 目录被误识别为源任务。

**修改文件**: `b/s/rbt/discover_source_tasks.py` 第 13 行

```python
# 修改前:
DERIVED_SUFFIXES = ("_kptsim_lrbv30", "_kptsim_lrb", "_kptsim")

# 修改后:
DERIVED_SUFFIXES = ("_lrb3_kptsim", "_kptsim_lrbv30", "_kptsim_lrb", "_kptsim", "_lrb3")
```

**缘由**: 新增 `_lrb3` 过滤已有的不含关键点的 v3.0 目录；新增 `_lrb3_kptsim` 过滤本次即将生成的最终产物目录。长后缀在前，避免短后缀误匹配。

**验证**: 修改后 `discover_source_tasks.py --clean-root /B/Dta/RoboTwin-Clean` 输出 50 个任务（49 个 v2.1 + 1 个 v3.0 `stack_bowls_three`），不再包含 `_lrb3` 目录。

---

## Step 1: 试跑单任务 `click_bell`

选择 `click_bell`（3855 帧，全部任务中最小）作为试跑对象。

### 阶段 1: SAPIEN FK 提取

```bash
cd /B/SRC/GeoPredict
python3 b/script/kpt/run_extract.py \
  --dataset_dir /B/Dta/RoboTwin-Clean/click_bell \
  --urdf_path /B/SRC/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf \
  --output_dir /B/Dta/RoboTwin-Clean/click_bell_kptsim
```

- 结果: 50 个 episode 全部提取成功
- offset: `[-0.8042, -1.0732, 0.5082]`
- 变换后值域: min=`[0.447, 0.415, 0.290]`, max=`[1.153, 1.185, 0.710]`
- Range validation: PASS (0 个越界点)

### 阶段 2: 归一化统计

```bash
python3 /B/SRC/GeoPredict/tools/compute_robotwin_norm_stats.py \
  --dataset_dir /B/Dta/RoboTwin-Clean/click_bell \
  --output /B/Dta/RoboTwin-Clean/.norm_stats/robotwin_norm_stats_click_bell.json
```

- 结果: 成功生成 JSON (键: `state`/`actions`，各含 mean/std/q01/q99)

### 阶段 3: 关键点注入

```bash
python3 /B/SRC/itvlaGp/util_scripts/inject_kptsim_keypoints.py \
  --source /B/Dta/RoboTwin-Clean/click_bell \
  --kptsim_dir /B/Dta/RoboTwin-Clean/click_bell_kptsim \
  --dest /B/Dta/RoboTwin-Clean/click_bell_kptsim_lrb \
  --norm_stats_path /B/Dta/RoboTwin-Clean/.norm_stats/robotwin_norm_stats_click_bell.json \
  --coord_mode voxel --force
```

- 结果: 50 个 episode, 3855 帧注入完成
- 关键输出: `norm_stat.json`（键重映射为 `observation.state`/`action`）, `meta/keypoints_meta.json`, `meta/info.json` 已更新

### 阶段 4: Layer-1 验收

```bash
python3 /B/SRC/itvlaGp/b/s/rbt/layer1_check.py \
  --dest-root /B/Dta/RoboTwin-Clean/click_bell_kptsim_lrb \
  --kptsim-root /B/Dta/RoboTwin-Clean/click_bell_kptsim \
  --task click_bell
```

- 结果: **ALL PASS** (6 项检查全部通过)
- TCP 最大跳变: 0.0230 (远低于 0.15 阈值)

### 阶段 5: v2.1→v3.0 转换

```bash
# 隔离工作区
CONVERT_WS="/B/Dta/RoboTwin-Clean/.convert_ws/click_bell"
mkdir -p "${CONVERT_WS}/robotwin"
ln -sfn "/B/Dta/RoboTwin-Clean/click_bell_kptsim_lrb" "${CONVERT_WS}/robotwin/click_bell_kptsim"

# 转换
python3 /B/SRC/itvlaGp/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
  --repo-id="robotwin/click_bell_kptsim" --root="${CONVERT_WS}" \
  --push-to-hub=false --force-conversion

# 搬运到最终位置
rsync -a --delete "${CONVERT_WS}/robotwin/click_bell_kptsim_v30/" \
  "/B/Dta/RoboTwin-Clean/click_bell_lrb3_kptsim/"
cp -f ".../keypoints_meta.json" ".../norm_stat.json"  # 补拷 meta

# Layer-2 验证
# Layer-2 OK: v3.0, episodes=50, frames=3855
```

- 结果: 转换成功，Layer-2 OK
- 清理: 删除了 `click_bell_kptsim/`, `click_bell_kptsim_lrb/`, `.norm_stats/robotwin_norm_stats_click_bell.json`, `.convert_ws/click_bell/`

### 试跑总结

`click_bell` 全部 5 个阶段通过，最终产物 `/B/Dta/RoboTwin-Clean/click_bell_lrb3_kptsim/` 生成成功。

---

## Step 2: 新增脚本

### 新增文件: `b/s/rbt/prepare_all_kptsim.sh`

**路径**: `/B/SRC/itvlaGp/b/s/rbt/prepare_all_kptsim.sh`

**缘由**: 封装 5 阶段流水线为可重复执行的批量脚本，支持 `--task`/`--tasks`/`--force`/`--keep-going`，内置任务发现逻辑（过滤 `_lrb3`, `_lrb3_kptsim`, `_kptsim*`, `_old`, 隐藏目录，且只保留 `codebase_version=v2.1`）。

**关键特性**:
- 约 160 行 Bash，独立可执行（不依赖 `lib.sh`）
- 每个任务成功后清理中间产物
- 已完成任务自动跳过（检查 `meta/info.json` + `norm_stat.json` + `meta/keypoints_meta.json`）
- `--keep-going` 模式下单任务失败继续处理下一个

```bash
chmod +x /B/SRC/itvlaGp/b/s/rbt/prepare_all_kptsim.sh
```

---

## Step 3: 全量批量执行

### 执行命令

```bash
cd /B/SRC/itvlaGp
bash b/s/rbt/prepare_all_kptsim.sh --keep-going 2>&1 | tee /tmp/prepare_all_kptsim.log
```

### 执行过程记录

**时间线**:

| 时间 | 任务 | 结果 |
|:---|:---|:---|
| 09:00:20 | 批量启动，50 个任务入列 | - |
| 09:00:21–09:00:34 | adjust_bottle | DONE |
| 09:00:34–09:00:46 | beat_block_hammer | DONE |
| 09:00:46–09:01:03 | blocks_ranking_rgb (23041 帧) | DONE |
| 09:01:03–09:01:20 | blocks_ranking_size (23170 帧) | DONE |
| 09:01:20–09:01:32 | click_alarmclock | DONE |
| 09:01:32 | click_bell | Skip (已完成) |
| 09:01:32–09:01:46 | dump_bin_bigbin | DONE |
| 09:01:46–09:01:57 | grab_roller | DONE |
| 09:01:57–09:02:12 | handover_block (14084 帧) | DONE |
| 09:02:12–09:02:26 | handover_mic | DONE |
| 09:02:26–09:02:41 | hanging_mug (16889 帧) | DONE |
| 09:02:41–09:02:53 | lift_pot | DONE |
| 09:02:53–09:03:06 | move_can_pot | DONE |
| 09:03:06–09:03:18 | move_pillbottle_pad | DONE |
| 09:03:18–09:03:30 | move_playingcard_away | DONE |
| 09:03:30–09:03:43 | move_stapler_pad | DONE |
| 09:03:43–09:03:56 | open_laptop | DONE |
| 09:03:56–09:04:14 | open_microwave (24333 帧) | DONE |
| 09:04:14–09:04:26 | pick_diverse_bottles | DONE |
| 09:04:26–09:04:38 | pick_dual_bottles | DONE |
| 09:04:38–09:04:51 | place_a2b_left | DONE |
| 09:04:51–09:05:03 | place_a2b_right | DONE |
| 09:05:03–09:05:17 | place_bread_basket | DONE |
| 09:05:17–09:05:30 | place_bread_skillet | DONE |
| 09:05:30–09:05:44 | place_burger_fries | DONE |
| 09:05:44–09:05:58 | place_can_basket | DONE |
| 09:05:58–09:06:13 | place_cans_plasticbox (14375 帧) | DONE |
| 09:06:13–09:06:26 | place_container_plate | DONE |
| 09:06:26–09:06:40 | place_dual_shoes | DONE |
| 09:06:40–09:06:52 | place_empty_cup | DONE |
| 09:06:52–09:07:05 | place_fan | DONE |
| 09:07:05–09:07:17 | place_mouse_pad | DONE |
| 09:07:17–09:07:31 | place_object_basket | DONE |
| 09:07:31–09:07:44 | place_object_scale | DONE |
| 09:07:44–09:07:56 | place_object_stand | DONE |
| 09:07:56–09:08:09 | place_phone_stand | DONE |
| 09:08:09–09:08:22 | place_shoe | DONE |
| 09:08:22–09:08:34 | press_stapler | DONE |
| 09:08:34–09:08:54 | put_bottles_dustbin (31231 帧，最大任务) | DONE |
| 09:08:54–09:09:08 | put_object_cabinet | DONE |
| 09:09:08–09:09:21 | rotate_qrcode | DONE |
| 09:09:21–09:09:34 | scan_object | DONE |
| 09:09:34–09:09:48 | shake_bottle | DONE |
| 09:09:48–09:10:02 | shake_bottle_horizontally | DONE |
| 09:10:02–09:10:21 | stack_blocks_three (23619 帧) | DONE |
| 09:10:21–09:10:34 | stack_blocks_two (15647 帧) | DONE |
| 09:10:34–09:10:43 | (脚本 bug 产生的伪任务) | FAILED |
| 09:10:43–09:10:58 | stack_bowls_two | DONE |
| 09:10:58–09:11:10 | stamp_seal | DONE |
| 09:11:10–09:11:22 | turn_switch | DONE |
| 09:11:22 | SUMMARY: 48 成功, 1 跳过, 1 失败 | - |

**总耗时**: 约 11 分钟（09:00:20 → 09:11:22）

### 遇到的 Error 及修复

#### Error 1: `stack_bowls_three` skip 消息被误当作任务名

**现象**: 脚本报告 `[2026-09-02 09:00:20] Skip stack_bowls_three: version v3.0 (not v2.1)` 被当作任务名执行，导致 `FileNotFoundError` 和 `HFValidationError`。

**根因**: `prepare_all_kptsim.sh` 第 49 行的任务发现子 shell 中，`log "Skip ..."` 输出到 stdout，而 stdout 被 `while IFS= read -r name` 管道捕获作为任务名。`log()` 函数的输出混入了 `echo "${name}"` 的输出流。

**修复**: 将 skip 日志重定向到 stderr:

```bash
# 修改前 (prepare_all_kptsim.sh 第 49 行):
[[ "${ver}" == "v2.1" ]] || { log "Skip ${name}: version ${ver} (not v2.1)"; continue; }

# 修改后:
[[ "${ver}" == "v2.1" ]] || { log "Skip ${name}: version ${ver} (not v2.1)" >&2; continue; }
```

**影响**: 该 bug 未影响任何实际数据处理。`stack_bowls_three` 本身就应该被跳过（源已是 v3.0，提取器不兼容）。Bug 只导致了一条 FAILED 记录，无实际产物损坏。

**修复文件**: `b/s/rbt/prepare_all_kptsim.sh` 第 49 行

### 残留清理

- `.convert_ws/` 目录下有一个 bug 产生的伪目录（名字含方括号和空格），已手动 `rm -rf /B/Dta/RoboTwin-Clean/.convert_ws` 清除
- `.norm_stats/` 目录已在各任务成功后逐个清理，已为空
- 所有 `{task}_kptsim/` 和 `{task}_kptsim_lrb/` 中间目录均已自动清理

---

## Step 4: 全量验收

### 4.1 数量检查

```
_lrb3_kptsim 目录数: 50 (全部 50 个任务, 含补跑的 stack_bowls_three)
```

### 4.2 批量 Layer-2 验证

```
Total: 50 PASS, 0 FAIL (out of 50)
Total frames across all datasets: 549,787 (526,237 + stack_bowls_three 23,550)
```

每个 `_lrb3_kptsim` 目录均满足:
- `meta/info.json` 存在且 `codebase_version` 含 `3.0`
- `features` 含 `observation.keypoint_3d`
- `norm_stat.json` 存在
- `meta/keypoints_meta.json` 存在

### 4.3 源数据与 `_lrb3` 完好性

```
OK: All source and _lrb3 directories are clean (no contamination)
Checked 151 directories
```

无任何源数据或 `_lrb3` 目录被注入 `observation.keypoint_3d`。

### 4.4 中间目录清理确认

```
_kptsim 残留: 无
_kptsim_lrb 残留: 无
.norm_stats 残留: 已清理
.convert_ws 残留: 已手动清理 (bug 遗留)
```

---

## 变更文件汇总

| 操作 | 文件路径 | 缘由 |
|:---|:---|:---|
| **新增** | `b/s/rbt/prepare_all_kptsim.sh` (itvlaGp) | 批量处理主脚本，约 160 行 |
| **修改** | `b/s/rbt/discover_source_tasks.py` L13 (itvlaGp) | 添加 `_lrb3` 和 `_lrb3_kptsim` 到 `DERIVED_SUFFIXES` |
| **修复** | `b/s/rbt/prepare_all_kptsim.sh` L49 (itvlaGp) | `log` 输出重定向到 stderr，修复 skip 消息被误捕获为任务名的 bug |
| **生成** | 49 个 `{task}_lrb3_kptsim/` 目录 (/B/Dta/RoboTwin-Clean/) | 最终产物，含 kptsim 3D 关键点的 LeRobot v3.0 数据 |

---

## 关键路径索引

| 路径 | 说明 |
|:---|:---|
| `/B/Dta/RoboTwin-Clean/` | RoboTwin 2.0 数据根目录 |
| `/B/Dta/RoboTwin-Clean/{task}/` | 源 v2.1 数据（50 个，49 v2.1 + 1 v3.0） |
| `/B/Dta/RoboTwin-Clean/{task}_lrb3/` | 已有 v3.0 数据（不含关键点，未触碰） |
| `/B/Dta/RoboTwin-Clean/{task}_lrb3_kptsim/` | 本次生成的最终产物（50 个） |
| `/B/SRC/itvlaGp/b/s/rbt/prepare_all_kptsim.sh` | 批量处理脚本 |
| `/B/SRC/itvlaGp/b/s/rbt/discover_source_tasks.py` | 修改过的任务发现脚本 |
| `/B/SRC/GeoPredict/b/script/kpt/run_extract.py` | SAPIEN FK 提取入口 |
| `/B/SRC/itvlaGp/util_scripts/inject_kptsim_keypoints.py` | 关键点注入脚本 |
| `/B/SRC/itvlaGp/b/s/rbt/layer1_check.py` | Layer-1 验收脚本 |
| `/B/SRC/itvlaGp/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py` | v2.1→v3.0 转换脚本 |
| `/B/SRC/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf` | 双臂机器人 URDF |
| `/tmp/prepare_all_kptsim.log` | 批量执行完整日志 |

---

## ~~未处理的任务~~ (已全部完成)

~~`stack_bowls_three` 因源目录被误判为 v3.0 而跳过~~ — 已在 Step 5 中补跑完成。

---

## Step 5: 补跑 `stack_bowls_three`

### 背景

批量执行时，`prepare_all_kptsim.sh` 的任务发现逻辑在子 shell 中执行 `python3 -c "... json.load ... get('codebase_version')"` 来读取 `info.json` 的版本号。由于 `stack_bowls_three` 的 skip 消息被误捕获为任务名（Error 1），它实际上从未被正确检测。

重新手动检查发现：**`/B/Dta/RoboTwin-Clean/stack_bowls_three/` 的源数据实际上是 v2.1 格式**（`codebase_version=v2.1`，含 50 个 `episode_*.parquet`），之前对话中误判它为 v3.0 是因为 `/B/Dta2/RoboTwin-Clean/stack_bowls_three/` 是 v3.0（Dta2 上的那份确实是 v3.0，但 Dta 上的源是 v2.1）。

因此可以直接使用标准 5 阶段流水线。

### 执行

手动逐阶段执行（不使用批量脚本，避免再次触发 bug）：

```bash
# Stage 1: SAPIEN FK 提取
cd /B/SRC/GeoPredict
python3 b/script/kpt/run_extract.py \
  --dataset_dir /B/Dta/RoboTwin-Clean/stack_bowls_three \
  --urdf_path /B/SRC/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf \
  --output_dir /B/Dta/RoboTwin-Clean/stack_bowls_three_kptsim
# 结果: 50 ep, offset=[-0.8117, -1.0236, 0.5046], Range PASS

# Stage 2: 归一化统计
python3 /B/SRC/GeoPredict/tools/compute_robotwin_norm_stats.py \
  --dataset_dir /B/Dta/RoboTwin-Clean/stack_bowls_three \
  --output /B/Dta/RoboTwin-Clean/.norm_stats/robotwin_norm_stats_stack_bowls_three.json

# Stage 3: 关键点注入
python3 /B/SRC/itvlaGp/util_scripts/inject_kptsim_keypoints.py \
  --source /B/Dta/RoboTwin-Clean/stack_bowls_three \
  --kptsim_dir /B/Dta/RoboTwin-Clean/stack_bowls_three_kptsim \
  --dest /B/Dta/RoboTwin-Clean/stack_bowls_three_kptsim_lrb \
  --norm_stats_path /B/Dta/RoboTwin-Clean/.norm_stats/robotwin_norm_stats_stack_bowls_three.json \
  --coord_mode voxel --force
# 结果: 23,550 帧注入完成

# Stage 4: Layer-1 验收
python3 /B/SRC/itvlaGp/b/s/rbt/layer1_check.py \
  --dest-root /B/Dta/RoboTwin-Clean/stack_bowls_three_kptsim_lrb \
  --kptsim-root /B/Dta/RoboTwin-Clean/stack_bowls_three_kptsim \
  --task stack_bowls_three
# 结果: ALL PASS

# Stage 5: v2.1→v3.0 转换
CONVERT_WS="/B/Dta/RoboTwin-Clean/.convert_ws/stack_bowls_three"
mkdir -p "${CONVERT_WS}/robotwin"
ln -sfn "/B/Dta/RoboTwin-Clean/stack_bowls_three_kptsim_lrb" "${CONVERT_WS}/robotwin/stack_bowls_three_kptsim"
python3 /B/SRC/itvlaGp/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
  --repo-id="robotwin/stack_bowls_three_kptsim" --root="${CONVERT_WS}" \
  --push-to-hub=false --force-conversion
rsync -a --delete "${CONVERT_WS}/robotwin/stack_bowls_three_kptsim_v30/" \
  "/B/Dta/RoboTwin-Clean/stack_bowls_three_lrb3_kptsim/"
cp -f ".../keypoints_meta.json" ".../norm_stat.json"
# Layer-2 OK: v3.0, episodes=50, frames=23550

# 清理中间产物
rm -rf /B/Dta/RoboTwin-Clean/stack_bowls_three_kptsim
rm -rf /B/Dta/RoboTwin-Clean/stack_bowls_three_kptsim_lrb
rm -f  /B/Dta/RoboTwin-Clean/.norm_stats/robotwin_norm_stats_stack_bowls_three.json
rm -rf /B/Dta/RoboTwin-Clean/.convert_ws
```

### 结果

- 最终产物: `/B/Dta/RoboTwin-Clean/stack_bowls_three_lrb3_kptsim/` (v3.0, 50 ep, 23,550 帧, 含 `observation.keypoint_3d`)
- 中间产物: 全部清理
- 源数据: 未修改（`codebase_version=v2.1`, 无 `observation.keypoint_3d`）
- **全部 50 个 `_lrb3_kptsim` 目录现已齐全**

### 误判根因回溯

之前 `stack_bowls_three` 被认为"源已是 v3.0"，原因有二：
1. `/B/Dta2/RoboTwin-Clean/stack_bowls_three/` 确实是 v3.0（Dta**2** 上的），之前探查时混淆了 Dta 和 Dta2
2. 批量脚本中 skip 消息泄漏到任务列表（Error 1），使得该任务从未被正确执行或正确跳过，掩盖了它实际可以处理的事实

实际上 `/B/Dta/RoboTwin-Clean/stack_bowls_three/`（Dta，不是 Dta2）始终是 v2.1 格式，含 50 个 `episode_*.parquet`，完全兼容提取器。

