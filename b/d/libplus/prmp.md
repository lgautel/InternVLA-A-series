

## 基于方案执行libero-plus 评估

请根据 "用libero-plus对checkpoint进行评估的实施方案与操作手册" @itvlaGpLibPlus/b/d/libplus/eval3.md 的指示, 去生成和修改代码或脚本等文件, 并执行文档中的所有测试与验收, 测试与验收通过后就用 libero-plus 对在`/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_12_08_52_49-internvla_a1_5-libplus-sft/checkpoints/032070/pretrained_model/` 中的checkpoint进行评估. 过程中若遇到error就fix, 直到所有测试和验收都通过并且在 libero-plus 上的评估都成功完成. 记录过程中的一切细节, 包括但不限于:所有的error及其根因分析, fix方案, 记录所有的操作, 命令, 关键路径和任何文件的增删改以及做这些操作的原因, 过程中的一切细节都记录在 @itvlaGpLibPlus/b/d/libplus/eval3_0915LOG.md 后面. 日志文件和训练结果到 ~/b/Ckp/ 中. 但遇到某个任务没做成功时要保存该任务的id, 动作序列,随机种子等供后续复现用. 但要注意不要覆盖任何已有的文件.  开始评估吧.


A critical finding is relevant to the current eval's keypoint accuracy:

**Robot base position is NOT constant across LIBERO-plus tasks**. It varies by arena type:

|Arena	|Robot	|Base X	|Base Z|
|---|---|---|---|
|table / kitchen_table	|MountedPanda	|-0.66	|0|
|study_table	|MountedPanda	|-0.75	|0|
|empty/floor	|both	|-0.60	|0|
|coffee_table	|OnTheGroundPanda	|-0.51	|0.41|
|living_room_table	|OnTheGroundPanda	|-0.51	|0.42|

This means the StandaloneFK (which uses a fixed base from the Lift MJCF) will produce systematically incorrect keypoint positions for non-table arenas — 5-27% error in X, and up to 0.42m offset in Z. This was already flagged in the project memory as a known issue but hasn't been fixed yet.

This doesn't affect the current eval's stability (it's running fine), but the keypoint inputs to the model will have arena-dependent errors. Worth noting for result interpretation — tasks on coffee_table and living_room_table arenas may show lower success rates partly due to this keypoint mismatch.

---
请深入分析  @itvlaGpLibPlus/ 中的代码, 和  @itvlaGpLibPlus/b/d/libplus/ 中的文档及其相关代码. 广度探索, 深入思考一下, 有什么比较好的解决方案.


你的 libero-plus 评估程序的 GPU-Util 有点低啊, 分析一下有什么问题, 有什么优化方案么. 先不作任何修改和操作, 单纯把问题分析和优化方案写到 @itvlaGpLibPlus/b/d/libplus/eval3_gpuoptim.md 中. 


请按照 @itvlaGpLibPlus/b/d/libplus/b/d/libplus/eval3_optim3.md 的章节 `# 实施落地方案与操作手册: 修 F1 → 修 F2 → 标准 LIBERO 基线验收` 的指示, 先跑修复F1, 修复F2, 以及所有的测试和验收代码, **注意:暂时不跑 LIBERO**. 过程中若遇到error就fix, 直到所有测试和验收都通过, LIBERO评估顺利都跑完. 记录过程中的一切细节, 包括但不限于:所有的error及其根因分析, fix方案, 记录所有的操作, 命令, 关键路径和任何文件的增删改以及做这些操作的原因, 过程中的一切细节都记录在 @itvlaGpLibPlus/b/d/libplus/b/d/libplus/eval3_optim3_libLOG.md 后面. 


由于 EGL 经常由于SIGABRT而崩溃, 建议用如下方案:
---
方案 B：彻底不用 EGL，改用 OSMesa
如果最重要的是“绝不再出现 EGL SIGABRT”，可以使用 CPU 渲染：

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
优点：

完全绕过 NVIDIA EGL；
不存在 EGL context 跨进程问题；
适合作为稳定性基线或故障回退路径。
缺点：

渲染速度明显下降；
CPU 占用上升；
必须重新验证 OSMesa 与训练数据的图像方向、颜色、相机内容是否一致；
不适合长时间大规模高吞吐评估，除非性能可以接受。
---
请按该方案对 @itvlaGpLibPlus/b/d/libplus/b/d/libplus/eval3_optim3.md 相关内容进行修改, 并补充测试和验收.