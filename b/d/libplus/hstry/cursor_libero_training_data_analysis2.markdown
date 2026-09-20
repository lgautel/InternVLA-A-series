# Libero training data analysis
_Exported on 9/19/2026 at 10:46:24 GMT+8 from Cursor (3.15.19)_

---

**User**

你同事写了一份数据分析文档,但文档丢失了,只留下与AI的对话历史 @b/d/libplus/hstry/cursor_libero_training_data_analysis.md . 请参考该对话历史, 重写一份更好的对对 `/B/Dta/opvla_libero/` 这份机器人的训练数据进行深入分析的文档.

该数据分析文档有如下要求:
- 要考虑到如何达到训推一致性. 可参考训练完后要进行评估的LIBERO 仿真 benchmark的代码`/B/SRC/LIBERO/` , 和 LIBERO-plus 仿真benchmark的代码`/B/SRC/LIBERO-plus/` , 以及运行这些仿真benchmark的python虚拟环境`/B/VENV/libero_plus_client/`
- 要考虑到 3D keypoint 轨迹, 旋转四元数, 和EEF位姿 等4D信息应该如何抽取, 这些4D信息的计算要考虑到训推一致性以及仿真引擎与评估任务的特点等等. 

这份机器人训练数据的分析文档请写在 @itvlaGpLibPlus/b/d/libplus/ds/libero_raw_analyz2.md 中.

---

**Cursor**

我先读一下这份对话历史,以及确认几处关键路径的实际内容,再决定文档结构。

让我并行探查数据目录、遗留的分析脚本资产,以及 `b/d/libplus/` 下已有文档。

`ds/` 目录和之前的分析脚本资产也一起丢了。继续摸数据与代码。

数据是 RLDS/TFRecord,`merged_kpt` 只剩 tar 包。我现在并行派多个探查代理去摸代码与已有文档。

数据是 RLDS/TFRecord,而衍生的 `merged_kpt` 只剩 tar 包。我来确认 schema 和可用工具链。

重要发现:`/home/a26113/DATA/libero_lerobot_v3/` 存在。我来确认它是否是 LeRobot 中间态数据集。

---

**User**

<timestamp>Saturday, Sep 19, 2026, 8:33 AM (UTC+8)</timestamp>
<system_notification>
The following task has finished. If you were already aware, ignore this notification and do not restate prior responses.

<task>
kind: shell
status: success
task_id: 283936
title: 查找 keypoint 数据集与 LeRobot 主目录
tool_call_id: toolu_011wLkFarjNkbRgviHPaoZXY
output_path: /home/a26113/.cursor/projects/B-SRC-itvlaGpLibPlus/terminals/283936.txt
</task>
</system_notification>
<user_query>Briefly inform the user about the task result and perform any follow-up actions (if needed).</user_query>

你同事写了一份数据分析文档,但文档丢失了,只留下与AI的对话历史 @b/d/libplus/hstry/cursor_libero_training_data_analysis.md . 请参考该对话历史, 重写一份更好的对对 `/B/Dta/opvla_libero/` 这份机器人的训练数据进行深入分析的文档.

该数据分析文档有如下要求:
- 要考虑到如何达到训推一致性. 可参考训练完后要进行评估的LIBERO 仿真 benchmark的代码`/B/SRC/LIBERO/` , 和 LIBERO-plus 仿真benchmark的代码`/B/SRC/LIBERO-plus/` , 以及运行这些仿真benchmark的python虚拟环境`/B/VENV/libero_plus_client/`
- 要考虑到 3D keypoint 轨迹, 旋转四元数, 和EEF位姿 等4D信息应该如何抽取, 这些4D信息的计算要考虑到训推一致性以及仿真引擎与评估任务的特点等等. 
- 可参考 @b/d/libplus/ 里面的各个文档, 避免踩之前踩过的坑.
- 注意, **不要搜索 /home/a26113/**

这份机器人训练数据的分析文档请写在 @itvlaGpLibPlus/b/d/libplus/ds/libero_raw_analyz2.md 中.

---

**Cursor**

那条后台搜索已经跑完了:在扫描到的位置里**没有**任何带 `observation.keypoint_3d` 或 `joint_position` 的 LeRobot 数据集——也就是说 `merged_kpt`(训练实际用的那份带关键点的数据)在本机只剩 `/B/Dta/opvla_libero_merged_kpt.tar.gz` 压缩包,没有解包副本。后续我不再检索 `/home/a26113/`。

这对文档是个**关键约束**:4D 关键点必须从 `/B/Dta/opvla_libero/` 的 `joint_state` 重新做 FK 才能复现,而 FK 需要的 Lift MJCF 也不在仓库里。我先验证这条路能不能走通。

解析器与 FK 两条路都通了。最后验证一件事:能否绕开 robosuite(本机无 GL 无法导入)复现 EEF 位姿。

FK 路径完全验证通过——纯 MuJoCo 就能从 RLDS 反推出每条 episode 的 arena 基座偏移,且与 LIBERO 源码的表格逐行吻合。最后确认 LIBERO-plus 的机器人扰动是否动到基座。

发现一个之前文档没记录的重要问题:LIBERO-plus 的**基线** agentview 相机位姿与 LIBERO 不同。我核实一下。

研究完成。我已经验证了所有关键的技术路径,现在给出方案。

---

**User**

<timestamp>Saturday, Sep 19, 2026, 8:42 AM (UTC+8)</timestamp>
<system_notification>
The following task has finished. If you were already aware, ignore this notification and do not restate prior responses.

<task>
kind: shell
status: success
task_id: 283936
title: 查找 keypoint 数据集与 LeRobot 主目录
tool_call_id: toolu_011wLkFarjNkbRgviHPaoZXY
output_path: /home/a26113/.cursor/projects/B-SRC-itvlaGpLibPlus/terminals/283936.txt
</task>
</system_notification>
<user_query>Briefly inform the user about the task result and perform any follow-up actions (if needed).</user_query>

---

**Cursor**

这条后台搜索就是我先前已经汇报过的那次(在扫描到的位置里没有任何带 `observation.keypoint_3d` 或 `joint_position` 的 LeRobot 数据集),没有新信息,也不需要后续动作。

方案已经写好了,等你确认。核心内容:

**研究阶段已经验证通的四件事**(这些决定了方案可行):
- RLDS/TFRecord 能用纯 Python 解析(本机无 TensorFlow),已成功读出 action、state、joint_state、JPEG。
- 规模自洽:96 个 shard、1693 条 episode,与对话历史一致。
- 关键点 FK 能在**不导入 robosuite** 的前提下精确复现——本机无 EGL 设备也无 OSMesa,`robosuite` 根本导不进来,仓库里也没有 `panda_robosuite_lift.xml`,所以旧的 `export_panda_mjcf.py` 路线是死路;改用 `mujoco` 直接加载 `robots/panda/robot.xml` 再串手爪静态变换链即可。
- 实测反推出 4 个离散 arena 基座偏移,与 LIBERO 源码的 `base_xpos_offset` 表逐行吻合。顺带证明了 `grip_site` 就是 `eef` body 原点,因此**第 8 个关键点其实是 `observation.state[0:3]` 的冗余副本**(模掉一个常数基座偏移)。

**一条需要纠正的猜测**:我一度怀疑 LIBERO-plus 的基线 agentview 相机与 LIBERO 不同,核实后两边**完全一致**——scene XML 里的 `pos="0.5 0 1.35"` 被 problem 类覆盖成了 per-scene 位姿,LIBERO-plus 只在显式 `_view_` 扰动时才改相机。这反而是个有用结论:相机/光照/纹理扰动完全不触及关键点,所以关键点分支在 LIBERO-plus 上是鲁棒性资产。

文档会有 14 章,重心在**训推一致性契约表**(逐通道标注训练侧/评估侧 `file:line` 与对应断言)和**4D 抽取章**(两个世界系、四元数双重覆盖、`R_pad` 两种归一化方案对比)。所有数字来自 `asset/` 下可复现的脚本全量实测,不抄旧文档的转述。

---

**User**

# 背景:已完成的可行性验证

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

I'll start implementing. First the scaffolding: the two foundation scripts.

`libero_10` 的分片文件名是 `liber_o10-train.tfrecord-*`——OpenVLA builder 里的一个拼写错误。修正映射。

All 96 shards verified: 1,693 episodes. Now the GL-free FK module.

位置误差 0.55 mm 正常,但旋转恰好差 90°——说明 robosuite 的 `eef_quat` 取的不是 `eef` body。查证。

确认了一个值得写进文档的细节:`observation.state` 的位置和姿态来自**两个不同的坐标系**——位置取 `gripper0_grip_site`(= `eef` body 原点),姿态取 `robot0_right_hand` body,两者相差固定的 −90° 绕 z 旋转。

FK 链验证通过(0.55 mm / 0.10°)。下一步:从 tar 包里取出真实训练关键点做对账。

R_pad 与 fps 都拿到权威值了。现在用真实训练关键点验证我的 FK 复现。

FK **位级复现**了真实训练关键点(误差 2.2e-7 m,即 float32 舍入极限)。现在做全量统计。

全量统计完成。有三处要深挖:动作量化结构、关节限位轻微越界、以及 axis-angle 的取值范围超过 π。

量化结构清楚了:来自 robosuite SpaceMouse 的 `axis_scale=350.0`。修正脚本里的量化分析,并补上 axis-angle 不连续性的度量。

Now the axis-angle discontinuity and joint-limit reporting:

`pos_sensitivity` 默认 1.5、`rot_sensitivity` 1.0。两个通道都恰好用满 1050 级(=3×350),我把措辞收紧到实测可证的部分。

Pass 1 完成。现在是核心的 4D 分析。

Pass 2 完成。现在做朝向判定——用相机投影几何,而不是靠引用旧结论。

55 px 的绝对误差偏大——运动质心是个粗估计量。改用轨迹相关性做主判据,它对常数偏置免疫。

两个判据一致:`rot180`(行相关 −0.87、列相关 −0.97,双轴均翻转)。这在无仿真器的情况下独立验证了旧文档的像素结论。

契约里写着训练帧是 robosuite raw,但我刚证明 RLDS 是 rot180——这意味着转换链上有第二次旋转。我要直接验证这一点。

有 ffmpeg。直接解一帧训练视频与 RLDS 对比,把整条旋转链钉死。

彻底钉死了:MSE 8.61 vs 3929+,456 倍差距。现在整条旋转链闭合。我来核实评估端实际送的是哪一版。

整条链闭合了:`IMAGE_CONVENTION="opengl"` 意味着评估端送的就是 OpenGL 原始缓冲。现在收集契约两侧的精确引用。

现在写验收断言脚本,把所有契约通道机器化核对。

21 项全通过。不过 C8 的断言语义反了——它"通过"却在描述一个问题。我先核实 delta 索引是按帧还是按秒。

按帧索引,所以 fps 标签只影响物理秒数的解读,不影响窗口构造。我把 C8 的断言语义改正。

Now the figures.

npz 里的 object 数组在两个 numpy 版本间无法共享。改成纯数值数组。

Let me inspect the generated figures for correctness.

图像方向的结论最好直接用画面呈现。我把 fig09 改成两行:上面放两张真实帧对比。

All ten figures are in place. Now writing the document.

Now I have everything. Writing the document.

Now verifying every claim, link, and reference in the document.

三个真问题:一个锚点、一批四舍五入差异,以及 §11.1 表里我写错了 LIBERO-plus 的类别计数。先取真实数字。

Now making the number checker tolerant of prose rounding while still catching real drift:

218/218 通过。自动校验只确认了行号存在,我再逐条核对它们指向的内容是否属实。

三处都是把 `def` 行写成了 `return` 行的偏移。取准确行号。

Now let me strengthen the verifier to check citation *content*, not just that the line exists:

校验器的正则只匹配 URL 里的行号,但文档里行号写在链接标签中。修正匹配方式。

同名文件导致解析歧义。改成以 `CITATION_CONTENT` 的路径后缀为准来消歧。

存在一个上游克隆 `/B/SRC/InternVLA-A-series/`,解析时抢先命中了。把本仓库设为优先根目录。

Now a full clean re-run of the whole pipeline from scratch to confirm reproducibility:

完整流程从零复跑 64 秒、全部通过。我核实一个我在 §9.7 里明确断言但还没实测的点:`link1`/`link2` 的姿态是否真的在变。

确认了:`link1` 的四元数在 qz 分量上变化 0.59——它随关节 1 转动。我把这个度量固化进脚本,让文档里的说法可复现。

Now updating §9.7 with the measured quaternion ranges:

All todos are complete. Let me do a final end-to-end confirmation:

计划已全部执行完毕。文档写在 [`b/d/libplus/ds/libero_raw_analyz2.md`](/B/SRC/itvlaGpLibPlus/b/d/libplus/ds/libero_raw_analyz2.md)(1061 行),配套 11 个脚本和 10 张图在 [`asset/`](/B/SRC/itvlaGpLibPlus/b/d/libplus/ds/asset)。整链从零复跑 64 秒,契约断言 21/21 通过,文档自检 259/259 通过。

## 本次实测得出的主要结论

**最重要的一条是新发现的阻断项**:`/B/Dta/opvla_libero/` 里的 JPEG 与评估端送进模型的画面**差 180° 旋转**。我用两个互相独立的方法钉死了这条旋转链——先用几何投影(把 `state[0:3]` 通过场景固定的 agentview 位姿投到像素,与运动能量质心比对,行/列相关系数分别是 −0.87 和 −0.97)证明 RLDS 是 MuJoCo 缓冲的 rot180;再从 tar 里解出一帧训练视频直接比像素,证明 `merged_kpt` 又是 RLDS 的 rot180(MSE 8.61 对次优 3929,456 倍差距)。结论是评估端现在的 `rotate_images=false` 是对的,但任何直接吃 RLDS 原图的新管线必须自己补一次旋转。

![朝向链](/B/SRC/itvlaGpLibPlus/b/d/libplus/ds/asset/fig09_orientation.png)

**`R_pad` 之争彻底结束**。从 tar 包里取出权威的 `keypoints_meta.json`,`bbox_radius = 1.8212722539901733`,与评估端硬编码值完全相同;我从 RLDS 全量重算得 `1.8212723272872922`,差 `7.3e-08`。`2.0945` 是误写,0.15 的 margin 已含在内。

**FK 复现是位级的**。绕开无法 import 的 `robosuite`,直接用 `mujoco` 加载裸 MJCF 重建运动链,与 tar 里真实训练关键点的最大误差是 **2.2e-07 米**(float32 舍入极限)。这条路径同时反推出了数据里全部 5 个 arena key、4 个基座位置,与 LIBERO 源码吻合到 0.07 mm。

几个值得注意的量化发现:归一化方案只用掉了 `[-1,1]` 的 **28.6%/24.0%/18.5%**,z 轴永远为正;四元数的 `qw≥0` 半球约定让 link7 和 eef 有六成以上的帧骑在分支切割线上,产生 7658 次对跖跳变,MSE 损失会把每一次都当成最大误差;LIBERO-plus 的 **84.5%** 扰动关键点通道完全看不见(这是该分支的鲁棒性资产),但剩下的 `Robot Initial States` 即使最温和档位(0.1 rad → 3.4 cm)也已超过训练首帧散布的 p95(2.14 cm)。

## 过程中修正的几处判断

FK 自检最初旋转误差恰好是 90°,查下来是 robosuite 的 `eef_pos` 取 `grip_site`、`eef_quat` 却取 `right_hand`,两者不同源——这本身成了文档里的 N7 条。另外我起初把 LIBERO-plus 的类别计数按图表目测写进了表格,自检脚本抓出来后换成了实际值。

文档里每个数字都有对应的 JSON 产物,`verify_doc.py` 会逐条核对数值、相对链接、章节锚点,以及每处 `file:line` 引用指向的那一行是否还包含所声称的内容。

---

**User**

你上面提到"从 tar 包里取出权威的 keypoints_meta.json,bbox_radius = 1.8212722539901733", 你是从哪个tar包得到这个信息的?

---

**Cursor**

来自这个 tar 包：

`/B/Dta/opvla_libero_merged_kpt.tar.gz`

读取的成员文件是：

`meta/keypoints_meta.json`

其中记录：

```json
"bbox_radius": 1.8212722539901733
```

对应脚本是：

`b/d/libplus/ds/asset/peek_merged_kpt_tar.py`

不是 `/B/Dta/opvla_libero/` 下的 RLDS TFRecord 文件。

