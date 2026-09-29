# 4D 生成

深入分析 @itvlaGpLibPlus/  的代码与文档，深入分析 @LIBERO-plus/  的代码与文档. 并参考这些文档：
+ 评估`libero_goal`子集的`LIBERO-plus`评估实施方案 @LIBERO-plus/b/d/ds/gol/eval_solu1.markdown 。
+ 数据分析文档 @LIBERO-plus/b/d/ds/gol/rlds/dsanalyz1.markdown 是对`LIBERO-plus`的训练数据`libero_goal`子集 `/B/Dta/LIBERO/rlds/libero_goal/`的分析。
+ Franka训练数据的 3D、4D 信息生成方案文档 @itvlaGpLibPlus/b/d/Frk3/ds/cubinbx/3d4d_gen_3.markdown 是对"Franka夹取方块放入盒子"的训练数据做 3D、4D 关键点生成并转换成 lerobot v3.0 格式的文件。

请基于`itvlaGpLibPlus`和`LIBERO-plus`的真实代码，基于 `/B/Dta/LIBERO/rlds/libero_goal/`中的真实数据，基于对 `/B/Dta/LIBERO/rlds/libero_goal/`数据的自己的亲自测量与分析，基于 MuJoCo/robosuite 仿真环境的真实情况与`LIBERO-plus`具体实现方式，写一份在  `/B/Dta/LIBERO/rlds/libero_goal/` 数据中生成并加入3D、4D 信息(包括3D关键点及其四元素、3D关键点历史等等)的实现方案，以及在`libero_goal`子集的评估时做3D、4D 信息生成的实现方案, 要注意训练和推理的一致性，训练数据、坐标系等和仿真环境的一致性，两个方案都写在 @itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen1.markdown 中。要细化到代码逻辑实现，代码生成在 @itvlaGpLibPlus/b/s/libplus2/gol/ 中，尽量不要修改已有的代码。测试与验收脚本是否覆盖完全。文档要完整和自包含。

---

@itvlaGpLibPlus/b/d/libplus/hstry/ 里的各个文档记录了上次在对`LIBERO-plus`的训练数据添加3D、4D信息并进行训练和评估时所遇到的各种问题，请深入检查一下  @itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen1.markdown 中的方案会不会导致类似的问题，实现步骤与代码逻辑是否写得够细，或者有什么需要补充和修正的， 对`3d4d_gen1.markdown`进行改良后写到 @itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen2.markdown 中。

---

请基于 @itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen2.markdown 中的方案，在python虚拟环境`/B/VENV/itnvla15rbt20/`中运行，对 `/B/Dta/LIBERO/rlds/libero_goal/`数据进行处理，生成包含3D、4D信息的 lerobot v3 格式的`libero_plus_goal`训练数据, 新数据保存到 `/home/a26113/b/Dta/libero_plus_goal_lrb3_4D/` 中。 过程中若遇到error就fix，直到所有测试和验收全都通过而且所有数据转换成功完成并验证没问题。记录过程中的一切细节，包括但不限于：所有的error及其根因分析，fix方案， 记录所有的操作、命令、关键路径和任何文件的增删改以及做这些操作的原因，过程中的一切细节都记录在 @itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen2_0929LOG.markdown 后面。 在数据转换成功后或失败而挂起30分钟时，启动 bigmatrix_multiply_optimization.py 后台任务占GPU。开始数据转换吧。

# Warmup 训练

参考之基于"用Franka机器臂夹取方块到盒子中"的训练数据进行warmup训练的实施落地方案（含详细操作手册）@itvlaGpLibPlus/b/d/Frk3/ds/cubinbx/cubbx_warmup1.md 和实际对这些数据做 warmup 训练的执行日志 @itvlaGpLibPlus/b/d/Frk3/ds/cubinbx/cubbx_warmup1_0924LOG.md 。 写一篇基于含3D、4D信息的`LIBERO-plus`的`libero_goal`子集训练数据`/home/a26113/b/Dta/libero_plus_goal_lrb3_4D/` 进行warmup训练的实施落地方案(含详细操作手册), 写到 @itvlaGpLibPlus/b/d/libplus2/gol/p1_warmup1.md 中。除了之前做“`Franka夹取方块放入盒子`的数据做的warmup训练实施方案”时要注意的事项外，在写该 “对`LIBERO-plus`的`libero_goal`子集训练数据进行warmup训练的实施落地方案(含详细操作手册)”时还需注意以下事项:
- EXPR_NAME 变量定义为 `4dwvlaLbPlusGol0929`.
- 要随机输入黑图, 无文本, 去掉tokenize_state, 以降低图片和文本模态对4D模态(keypoint_expert)的影响
- Warmup 训练跑 2 个 epoch，基于要跑的epoch数重新计算与epoch, step, learning rate warmup / decay 相关的超参。
- keypoint_history_max_len 设为 92


请按照"基于`用Franka机器臂夹取方块到盒子中`的训练数据(含3D, 4D信息的)`/B/Dta/put_cube_into_box_lrb3_4D/` 进行warmup训练的实施落地方案(含详细操作手册)" 也就是基于  @b/d/Frk3/ds/cubinbx/cubbx_warmup1.md 里的指示 ,  用InternVLA-A1.5-base 对 `/B/Dta/put_cube_into_box_lrb3_4D/ 中的数据进行warmup训练. 训练过程中若遇到error就fix, 直到所有测试和验收全都通过并且warmup训练成功以及它的checkpoinnt完整保存. 记录所有训练过程中的一切细节, 包括但不限于:所有的error及其根因分析, fix方案, 记录所有的操作, 命令, 关键路径和任何文件的增删改以及做这些操作的原因, 过程中的一切细节都记录在   后面. 在warmup成功后或失败而挂起30分钟时, 启动 bigmatrix_multiply_optimization.py 后台任务占GPU, 然后把日志文件打包到 ~/b/Ckp/ 中.  开始warmup训练吧.