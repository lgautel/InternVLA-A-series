# Libero training data analysis
_Exported on 9/18/2026 at 10:06:05 GMT+8 from Cursor (3.15.19)_

---

**User**

对 /B/Dta/opvla_libero/ 这份机器人的训练数据进行深入分析, 写一份数据分析报告到 @itvlaGpLibPlus/b/d/libplus/ds/libero_raw_analyz.md.

---

**Cursor**

先摸清数据目录结构和现有相关文档，再据此做深入分析并写报告。

用户指定了 Markdown 报告路径，接下来会先摸清数据目录、现有分析和代码里怎么读这份数据。

目录是 LeRobot 格式的四个 LIBERO 子集。接着读 README、元数据和代码里的消费路径。

这是 OpenVLA 的 RLDS/TFRecord 原始数据。接下来读取 schema，并对照代码与论文里的用法。

接下来对照现有 RLDS 解析脚本、论文附录和下游 LeRobot 数据，再写全量统计。

protobuf 可用。接下来先拆一条 TFRecord，确认 schema，再全量统计。

解析逻辑已经对齐。接下来写无 TensorFlow 的 TFRecord 解析器，先在单条记录上验证，再跑全量统计。

解析器可用。正在写全量分析脚本：统计规模、动作/状态、图像朝向，并导出图表。

全量解析完成：1693 条 episode、273465 帧，完整性检查全部通过。接下来读统计结果和样例图，准备写报告。

统计和图像都齐了。开始把报告写到指定路径，并把关键图表链进去。

报告已写到 [`b/d/libplus/ds/libero_raw_analyz.md`](/B/SRC/itvlaGpLibPlus/b/d/libplus/ds/libero_raw_analyz.md)。统计来自对 `/B/Dta/opvla_libero/` 全部 TFRecord 的离线解析（1,693 条 episode、273,465 帧），图表在 `b/d/libplus/ds/asset/`。

**这是什么。** 本地副本对应 Hugging Face 的 `openvla/modified_libero_rlds`：OpenVLA 把官方 2,000 条 LIBERO 演示回放成 256×256，丢掉失败轨迹和 no-op 空转后再封成 RLDS。`file_path` 指向 `regenerated--no_noops/`，和下游 `opvla_libero_merged_kpt` 的规模（1,693 / 273,465）完全一致。

**对训练最重要的几件事：**

1. **不是均衡的 50×40。** 只留下 84.65% 的演示；Long 只剩 75.8%。按帧采样时 Long 占 **37.1%** 梯度。`put both moka pots on the stove` 只剩 29 条，却有 11,808 帧。
2. **动作已经是末端增量 + 二值夹爪**（\(\pm 0.9375\) 裁剪，夹爪只有 ±1）。OpenVLA 的 no-op 判据复查剩余 **0 帧**。应使用 `action_mode=abs`、`[6,-1]`，不要再差分。
3. **图像朝向和 merged_kpt 相反。** 这份 RLDS 的 JPEG 已经转过 180°（人眼正向）；`opvla_libero_merged_kpt` 的视频是 robosuite 原始朝向。混用等于系统性上下颠倒。
4. **数据很干净，但没有 val。** 边界标记、稀疏 reward、256×256、无 NaN 全部通过；含 7 维关节角，可直接做 FK。验证必须按 episode 切。
5. **物理 20 Hz，LeRobot 常标 10 Hz**，并没有 2× 抽帧。动作 chunk 的“秒”会差一倍。

---

**User**

<timestamp>Friday, Sep 18, 2026, 12:11 AM (UTC+8)</timestamp>
<system_notification>
The following task has finished. If you were already aware, ignore this notification and do not restate prior responses.

<task>
kind: shell
status: success
task_id: 413881
title: Find protobuf/TF and conversion scripts
tool_call_id: call-0c784f8c-79b3-4377-8a8f-36159cca0498-30
fc_7f4e4f12-0997-93f9-9d24-edd57b8b11c1_3
output_path: /home/a26113/.cursor/projects/B-SRC-itvlaGpLibPlus/terminals/413881.txt
</task>
</system_notification>
<user_query>Briefly inform the user about the task result and perform any follow-up actions (if needed).</user_query>

---

**Cursor**

那条后台命令是我在分析开始时用来探测本机环境的，现在已经跑完了（耗时约 7 分钟，主要卡在全盘 `find` 上）。

结果：本机有 `protobuf 7.35.1`，但**没有安装 TensorFlow、tensorflow-datasets、tfrecord 或 crc32c**，也没有除 `base` 以外的 conda 环境。

这个结论在当时已经用上了，所以不需要任何后续动作：我没有依赖 TensorFlow，而是自己写了纯 Python 的 TFRecord + `tf.train.Example` 解码器（`b/d/libplus/ds/asset/rlds_tfrecord.py`），直接按二进制帧格式读取 96 个 shard。分析已经完成，1,693 条 episode、273,465 帧全部解析成功，报告在 [`b/d/libplus/ds/libero_raw_analyz.md`](/B/SRC/itvlaGpLibPlus/b/d/libplus/ds/libero_raw_analyz.md)。

唯一的附带说明：因为没装 `crc32c`，解析器跳过了 TFRecord 每条记录的 CRC 校验。这不影响结论——所有 shard 的记录长度、episode 数与 `dataset_info.json` 声明的 `shardLengths` 逐一吻合，且全部字段通过了完整性检查。

---

**User**

结合 @itvlaGpLibPlus/ 代码的深入分析, 对刚写的 @itvlaGpLibPlus/b/d/libplus/ds/libero_raw_analyz.md 做一些补充和优化, 要考虑到如何达到训推一致性, 要考虑到 3D keypoint 轨迹等4D信息应该如何抽取, 4D信息的计算要考虑到训推一致性以及仿真引擎与评估任务的特点等等. 

---

**Cursor**

我先深入读代码：关键点生成脚本、GeoP 训练侧的消费路径、评估端的关键点来源，以及现有 kpt 文档里的坐标系结论。

关键点元数据里的 `bbox_radius` 是 1.8213，而之前 canvas 诊断引用的是 2.0945——这个矛盾必须实测澄清。同时去看评估端的关键点来源。

这是个关键发现——需要立刻用全量数据核实关键点的尺度与坐标系是否真的对齐。

关键点其实是**归一化过的**，但 EEF 与 `observation.state` 的偏移呈现明显的分套件结构——这正是训推一致性的核心。继续挖。

4 个 arena 各有一个固定偏移——这解释了此前"Z 相关系数只有 0.037"的假警报。现在看训练与评估两侧如何消费这些关键点。

评估端确实用同一份 Lift MJCF，`eval3_kpt.md` 已有坐标系分析。我再确认历史窗口与控制频率在训推两侧是否逐步对齐。

掌握了完整链路：推理只消费 `his_kpts`+`his_len`，`kpt_t`/`kpt_future` 纯属监督目标。现在量化窗口填充、arena 偏移和关键点冗余。

已有一份姊妹报告 `libero_mrg4d_analyz.md` 覆盖了合并集的 4D 一致性。我核对它的结论，确保补充而非重复。

图表就绪。现在把新分析补进报告——先确认没有内部章节交叉引用会被编号变更打断。

Now我把新的 4D 与训推一致性分析插入报告，并同步更新头部、结论与风险清单。

Now插入核心的三个新章节（训推一致性契约、4D 抽取、坐标系与时间轴）。

现在更新训练建议、风险清单和复现小节，把 4D 相关项补齐。

最后校验文档结构、链接和图片引用是否完整。

链接深度算错了一层（`ds/` 目录需要 4 级才到仓库根）。修正。

报告已补充完成。链接、图片和章节编号全部校验通过。

**做了什么：** 深读了 `util_scripts/`、`evaluation/LIBERO2/`、`policies/internvla_a1_5/` 的关键点生成与消费代码，然后写了 [`analyze_4d_consistency.py`](/B/SRC/itvlaGpLibPlus/b/d/libplus/ds/asset/analyze_4d_consistency.py) 对 RLDS 和 `merged_kpt` 做全量交叉核对，给 [`libero_raw_analyz.md`](/B/SRC/itvlaGpLibPlus/b/d/libplus/ds/libero_raw_analyz.md) 新增了五章（§12–§16）。

**最重要的三个发现：**

1. **训练集内部就有 4 个不同的机器人基座位置。** 把关键点反归一化后减去 `observation.state`，273,465 帧的差值不是噪声云而是恰好 4 个点：Spatial/Goal/Long-厨房 在 \((-0.66,0,0.912)\)、Object 在 \((-0.60,0,0)\)、Long-客厅 在 \((-0.51,0,0.42)\)、Long-书房 在 \((-0.75,0,0.912)\)。这与 `eval3_kpt.md` 从 XML 整理的基座表逐行吻合，但我这张表是从训练张量直接测出来的，还附带了任务→arena 的映射和帧占比。

![四个 arena 基座](/B/SRC/itvlaGpLibPlus/b/d/libplus/ds/asset/fig_kpt_arena_offsets.png)

2. **"z 相关系数只有 0.037"是 Simpson 悖论，不是 bug。** 组内相关系数都 ≥ 0.999996，是组间常量偏移把合并相关打到了 0。同时确认关键点**确实**已除以 \(R_{\mathrm{pad}}=1.8212722539901733\)；此前那条"未归一化"的阻断判断源于用错了常数。这两点都不构成阻断项。

3. **\(R_{\mathrm{pad}}\) 是最脆弱的一环。** 它等于 \(1.15\times1.5837\)，而那个 1.5837 主要是 Lift 基座高度 0.912 撑出来的——尺度由一个与任务无关的量决定，导致三轴只用掉 \([-1,1]\) 的 28.6%/24.0%/18.5%。更危险的是评估端把它**硬编码**成字面量：换数据集重新生成关键点后会静默缩放错误且不报错。

**关于训推一致性**，我列了 12 条逐通道契约（C1–C12）。九条已对齐，三条有残余缺口且都量化了：`H=200` 的历史窗 55% 是 padding 且 Spatial 没有任何一帧能填满（最长 episode 只有 193 帧）；no-op 过滤让训练时间轴带洞，但实测影响有限（训练里仍保留 6.4% 的"位移<1mm"帧）；评估在 10 个 settle step 也 push 关键点，导致首请求 `his_len=10` 而训练是 0。

另外顺带查到 `keypoint_noise_sigma` 只在 config 里声明、modeling 里没有任何使用点——设了不会生效也不会报错。

有一点需要你判断：`pos_only` 模式下 `link1`/`link2` 位置恒定且重复、`link5`/`link6` 重复，8 个关键点里有 2 个是纯死通道，24 维位置只有 15 维在动。这是支持继续用 `pos_rot` 的实证理由，但也说明 body 选择本身还有优化空间。

