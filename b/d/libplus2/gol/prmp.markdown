# 4D 生成

深入分析 @itvlaGpLibPlus/  的代码与文档，深入分析 @LIBERO-plus/  的代码与文档. 并参考这些文档：
+ 评估`libero_goal`子集的`LIBERO-plus`评估实施方案 @LIBERO-plus/b/d/ds/gol/eval_solu1.markdown 。
+ 数据分析文档 @LIBERO-plus/b/d/ds/gol/rlds/dsanalyz1.markdown 是对`LIBERO-plus`的训练数据`libero_goal`子集 `/B/Dta/LIBERO/rlds/libero_goal/`的分析。
+ Franka训练数据的 3D、4D 信息生成方案文档 @itvlaGpLibPlus/b/d/Frk3/ds/cubinbx/3d4d_gen_3.markdown 是对"Franka夹取方块放入盒子"的训练数据做 3D、4D 关键点生成并转换成 lerobot v3.0 格式的文件。

请基于`itvlaGpLibPlus`和`LIBERO-plus`的真实代码，基于 `/B/Dta/LIBERO/rlds/libero_goal/`中的真实数据，基于对 `/B/Dta/LIBERO/rlds/libero_goal/`数据的自己的亲自测量与分析，基于 MuJoCo/robosuite 仿真环境的真实情况与`LIBERO-plus`具体实现方式，写一份在  `/B/Dta/LIBERO/rlds/libero_goal/` 数据中生成并加入3D、4D 信息(包括3D关键点及其四元素、3D关键点历史等等)的实现方案，以及在`libero_goal`子集的评估时做3D、4D 信息生成的实现方案, 要注意训练和推理的一致性，训练数据、坐标系等和仿真环境的一致性，两个方案都写在 @itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen1.markdown 中。要细化到代码逻辑实现，代码生成在 @itvlaGpLibPlus/b/s/libplus2/gol/ 中，尽量不要修改已有的代码。测试与验收脚本是否覆盖完全。文档要完整和自包含。

---

@itvlaGpLibPlus/b/d/libplus/hstry/ 里的各个文档记录了上次在对`LIBERO-plus`的训练数据添加3D、4D信息并进行训练和评估时所遇到的各种问题，请深入检查一下  @itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen1.markdown 中的方案会不会导致类似的问题，实现步骤与代码逻辑是否写得够细，或者有什么需要补充和修正的， 对`3d4d_gen1.markdown`进行改良后写到 @itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen2.markdown 中。

---

请基于 @itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen2.markdown 中的方案，在python虚拟环境`/B/VENV/itnvla15rbt20/`中运行，对 `/B/Dta/LIBERO/rlds/libero_goal/`数据进行处理，生成包含3D、4D信息的 lerobot v3 格式的`libero_plus_goal`训练数据, 新数据保存到 `/B/Dta/LIBERO/libero_plus_goal_lrb3_4D/` 中。 过程中若遇到error就fix，直到所有测试和验收全都通过而且所有数据转换成功完成并验证没问题。记录过程中的一切细节，包括但不限于：所有的error及其根因分析，fix方案， 记录所有的操作、命令、关键路径和任何文件的增删改以及做这些操作的原因，过程中的一切细节都记录在 @itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen2_0929LOG.markdown 后面。 在数据转换成功后或失败而挂起30分钟时，启动 bigmatrix_multiply_optimization.py 后台任务占GPU。开始数据转换吧。

# Warmup 训练

参考之基于"用Franka机器臂夹取方块到盒子中"的训练数据进行warmup训练的实施落地方案（含详细操作手册）@itvlaGpLibPlus/b/d/Frk3/ds/cubinbx/cubbx_warmup1.md 和实际对这些数据做 warmup 训练的执行日志 @itvlaGpLibPlus/b/d/Frk3/ds/cubinbx/cubbx_warmup1_0924LOG.md 。 写一篇基于含3D、4D信息的`LIBERO-plus`的`libero_goal`子集训练数据`/B/Dta/LIBERO/libero_plus_goal_lrb3_4D/` 进行warmup训练的实施落地方案(含详细操作手册), 写到 @itvlaGpLibPlus/b/d/libplus2/gol/p1_warmup1.md 中。除了之前做“`Franka夹取方块放入盒子`的数据做的warmup训练实施方案”时要注意的事项外，在写该 “对`LIBERO-plus`的`libero_goal`子集训练数据进行warmup训练的实施落地方案(含详细操作手册)”时还需注意以下事项:
- EXPR_NAME 变量定义为 `4dwvlaLbPlusGol0929`.
- 要随机输入黑图, 无文本, 以降低图片和文本模态对4D模态(keypoint_expert)的影响
- Warmup 训练跑 2 个 epoch，基于要跑的epoch数重新计算与epoch, step, learning rate warmup / decay 相关的超参。
- keypoint_history_max_len 设为 92


请按照“基于含3D、4D信息的`LIBERO-plus`的`libero_goal`子集训练数据`/B/Dta/LIBERO/libero_plus_goal_lrb3_4D/` 进行warmup训练的实施落地方案(含详细操作手册)” 也就是基于  @itvlaGpLibPlus/b/d/libplus2/gol/p1_warmup1.md 里的指示 ,  用InternVLA-A1.5-base 对 `/B/Dta/LIBERO/libero_plus_goal_lrb3_4D/` 中的数据进行warmup训练。训练过程中若遇到error就fix，直到所有测试和验收全都通过并且warmup训练成功以及它的checkpoinnt完整保存。 记录所有训练过程中的一切细节，包括但不限于：所有的error及其根因分析， fix方案， 记录所有的操作、命令、关键路径和任何文件的增删改以及做这些操作的原因，过程中的一切细节都记录在 @itvlaGpLibPlus/b/d/libplus2/gol/p1_warmup1_0929.md 后面。 在warmup成功后或失败而挂起30分钟时， 启动 bigmatrix_multiply_optimization.py 后台任务占GPU， 然后把日志文件打包到 ~/b/Ckp/ 中。开始warmup训练吧。

# SFT 训练

参考之前的“基于Franka机器臂夹取方块到盒子中的训练数据做微调训练的实施落地方案(含详细操作手册)” @itvlaGpLibPlus/b/d/Frk3/ds/cubinbx/cubbx_sft1.md 和对应的微调训练执行日志 @itvlaGpLibPlus/b/d/Frk3/ds/cubinbx/cubbx_sft1_0924LOG.md 。 写一篇基于含3D、4D信息的`LIBERO-plus`的`libero_goal`子集训练数据`/B/Dta/LIBERO/libero_plus_goal_lrb3_4D/` 进行微调训练的实施落地方案(含详细操作手册)，写到 @itvlaGpLibPlus/b/d/libplus2/gol/p2_sft1.md  中。 除了之前做"`Franka夹取方块到盒子`的训练数据做的微调训练实施方案"时要注意的事项外， 写该“对`LIBERO-plus`的`libero_goal`子集训练数据进行微调训练的实施落地方案(含详细操作手册)”时还需注意以下事项：
- EXPR_NAME 变量定义为 `4dwvlaLbPlusGol0929`。
- 微调训练4个epoch，每2个epoch保存一次checkpoint。基于要跑的epoch数重新计算与epoch, step, learning rate warmup / decay 相关的超参。
- 使用数据增强：`image_transforms.p_schedule`设为`[0.5, 0.3, 0.1]`，`image_transforms.p_epoch_interval`设为 1。
- `use_fast_action_tokens`=true , `enable_vqa_loss`=true , `tokenize_state`=true 。 
- `keypoint_history_max_len` 改为 92 ,`log_freq`设为 100 。
- 从被warmup训练过的checkpoint `/home/a26113/b/Ckp/4dwvlaLbPlusGol0929/2026_09_29_17_42_23-internvla_a1_5-lbplus-gol-warmup/checkpoints/008010/pretrained_model/` 开始微调.
- 文档要完整,要自包含. 要细到代码或命令行级别, 细到代码逻辑和调用关系.
- 文档中要汇总一下该微调训练方案涉及的各个超参,参数,变量等配置的有效值是什么, 为什么要设这个值, 这个值是在哪里设的. 也汇总一下该方案, 涉及到了哪些文件增删改, 要改文件的哪些内容, 为什么要这样做. 还要汇总一下, 这个该方案在实际做微调训练时所涉及的关键路径, 包括但不限于checkpoint和日志等保存路径。

请按照“对`LIBERO-plus`的`libero_goal`子集训练数据进行微调训练的实施落地方案(含详细操作手册)”  @itvlaGpLibPlus/b/d/libplus2/gol/p2_sft1.md 里的指示，用基于含3D、4D信息的`LIBERO-plus`的`libero_goal`子集训练数据`/B/Dta/LIBERO/libero_plus_goal_lrb3_4D/` 进行warmup训练得到的checkpoint（`/home/a26113/b/Ckp/4dwvlaLbPlusGol0929/2026_09_29_17_42_23-internvla_a1_5-lbplus-gol-warmup/checkpoints/008010/pretrained_model/`）， 在 `/B/Dta/LIBERO/libero_plus_goal_lrb3_4D/`里的数据上再进行SFT微调训练。 训练过程中若遇到error就fix，直到所有测试和验收全都通过，直到SFT训练成功完成而且checkpoint和日志等产物成功完整保存。记录所有训练过程中的一切细节，包括但不限于：所有的error及其根因分析， fix方案， 记录所有的操作、命令、关键路径和任何文件的增删改以及做这些操作的原因， 过程中的一切细节都记录在 @itvlaGpLibPlus/b/d/libplus2/gol/p2_sft1_0929LOG.md 后面。 在SFT成功后或失败而挂起30分钟时，启动 bigmatrix_multiply_optimization.py 后台任务占GPU, 然后把日志文件打包到 ~/b/Ckp/ 中。  开始SFT微调训练吧。


# 评估

深入分析 @LIBERO-plus/ 和 @itvlaGpLibPlus/ 中的代码与文档。并参考如下文档和代码库：
- 以前在`LIBERO-plus`仿真环境中做测评时遇到的问题记录都在 @itvlaGpLibPlus/b/d/libplus/hstry/ 里面了， 请想尽办法，尽量避免再次遇到老问题。
- 以前在`LIBERO-plus`仿真环境中做测评时搭建环境的方案文档是 @itvlaGpLibPlus/b/d/libplus/ 里的所有以`eval`为前缀的`eval*.md`文档，其中以`eval3`为前缀的`eval3_*.md`文档会新一些，但**注意**这些文档有可能是错误的和过时的，仅供参考，一切都要基于`LIBERO-plus`代码库和`itvlaGpLibPlus`代码库中的最新代码为准。
- 认真研读为数据`/B/Dta/LIBERO/libero_plus_goal_lrb3_4D/`和仿真评估环境生成3D、4D信息并保持训推一致性的实现方案 @itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen2.markdown ， 特别注意章节`## 7. 评估侧`。
- 一切基于`LIBERO-plus`代码库和`itvlaGpLibPlus`代码库中的最新代码和`/B/Dta/LIBERO/libero_plus_goal_lrb3_4D/`中的实际数据。

写一份对checkpoint `/home/a26113/b/Ckp/4dwvlaLbPlusGol0929/2026_09_30_03_18_02-internvla_a1_5-lbplus-gol-sft/checkpoints/016020/pretrained_model/`在`LIBERO-plus`仿真benchmark上进行评估的实现方案， 写在 @itvlaGpLibPlus/b/d/libplus2/gol/eval1.md 中。要注意以下几点：
- 要细化到代码逻辑实现，代码生成在 @itvlaGpLibPlus/b/s/libplus2/gol/ 中，尽量不要修改已有的代码，设计要做到扩展大于修改。测试与验收脚本要覆盖完全。文档要完整和自包含。
- `LIBERO-plus`仿真安装在`/B/VENV/libero_plus_client/`中。
- 采用Server-Client模式， `itvlaGpLibPlus` 的模型推理服务在python虚拟环境 `/B/VENV/itnvla15rbt20/` 上作为 server 端运行。`LIBERO-plus`的仿真测评环境在 python虚拟环境 `/B/VENV/libero_plus_client/` 中作为 client 端运行。
- EXPR_NAME 变量定义为 `4dwvlaLbPlusGol0929`。
- 要有一个仅对`libero_goal`子集进行评估的例子和操作手册，并列出哪些参数或变量可以调试。
- 注意计算速度的问题，`LIBERO-plus`的仿真评估能跑得越快越好。
- 日志，中间产物和测评结果保存在/B/Log/<EXPR_NAME>/<timestamp>_eval 中。
- 文档中要汇总一下该`LIBERO-plus`评估方案涉及的各个超参,参数,变量等配置的有效值是什么, 为什么要设这个值, 这个值是在哪里设的. 也汇总一下该方案, 涉及到了哪些文件增删改, 要改文件的哪些内容, 为什么要这样做. 还要汇总一下, 这个该方案在实际做微调训练时所涉及的关键路径, 包括但不限于测评结果和日志等保存路径。
- 当遇到某次评估或trial没做成功时要保存该任务的id、动作序列、随机种子等供后续复现用，同时要把如何基于保存的信息进行复现写成详细的操作指南。

---

先清空占用GPU的所有程序和进程。然后，请根据 "在`LIBERO-plus`仿真benchmark上进行评估的实现方案"  @itvlaGpLibPlus/b/d/libplus2/gol/eval1.md 的指示, 去创建或更新python虚拟环境、生成和修改代码或脚本等文件, 并执行文档中的所有测试与验收, 测试与验收通过后就用对在`/home/a26113/b/Ckp/4dwvlaLbPlusGol0929/2026_09_30_03_18_02-internvla_a1_5-lbplus-gol-sft/checkpoints/016020/pretrained_model/` 中的checkpoint用`LIBERO-plus`仿真 benchmark 上进行评估. 过程中若遇到error就fix, 直到所有测试和验收都通过并且在 `LIBERO-plus`上的评估都成功完成。 记录过程中的一切细节，包括但不限于：所有的error及其根因分析，fix方案，记录所有的操作、命令、关键路径和任何文件的增删改以及做这些操作的原因，过程中的一切细节都记录在 @itvlaGpLibPlus/b/d/libplus2/gol/eval1_0929LOG.md 后面。 当遇到某次评估或 trial 没做成功时要保存该任务的 id、动作序列、随机种子等供后续复现用。另外做评估记录各种日志、评测结果和中间结果时要注意不能覆盖任何已有的文件。在评估成功后或失败而挂起30分钟时，启动 bigmatrix_multiply_optimization.py 后台任务占GPU, 然后把评估结果等日志文件打包到 ~/b/Ckp/ 中。开始评估吧。



# 数据v2-合并states与finger

现在发现了你写的 @itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen2.markdown 中，有上面提到的“需要处理 8D state / 7D action 的维度映射”的问题，请想办法把”observation.state“的”finger_l“，”finger_r“合成一个”gripper”，（原”finger_l“，”finger_r“）可以保留。另外想办法把`observation.state.joint_position`的数据整合到`observation.state`中，但`observation.state.joint_position`的数据保留。把这些基于`b/d/libplus2/gol/3d4d_gen2.markdown`的改良方案写到 @itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen2_2.markdown 中，需要生成的代码放在 @itvlaGpLibPlus/b/s/libplus2/gol/ 里面，尽量不动其它代码，改良后的新数据保存到`/B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2/`中。目前只是写方案，不做任何实际的代码生成与执行，但方案要细化到代码级别，要仔细思考各种风险，相应的测试和验收脚本都要齐全。该文档也要完整和自包含。

---

你上面提到`3d4d_gen2_2.markdown`这个方案中”视频用硬链接“，不要用任何链接，因为后续老的数据集可能会被删掉，直接复制就可以。请修改一下 @itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen2_2.markdown 

---
如果要基于你生成的`/B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2/`数据做训练的话，panda.yaml 应该怎么写？之前提的那些数据问题，对齐问题是否都能解决？如果不能应该如何解决？在 @itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen2_2.markdown 后面补上你的分析和对问题的解决方案。

---
@itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen2_2.markdown 有没有包含对新生成的数据的所有feature或字段从新做 norm stats， 对新数据从新生成所有 meta 文件的详细方案，如果没有，请补上。

---

请根据 @itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen2_2.markdown 对训练数据`/B/Dta/LIBERO/libero_plus_goal_lrb3_4D/`进行处理与改良，新数据保存在`/B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2/`中。过程中若遇到error就fix，直到所有数据成功处理完而且所有测试和验收都通过。记录所有训练过程中的一切细节，包括但不限于：所有的error及其根因分析， fix方案， 记录所有的操作、命令、关键路径和任何文件的增删改以及做这些操作的原因， 过程中的一切细节都记录在 @itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen2_2_1001LOG.markdown 后面。

# 数据v2 Pipeline

你要写一个包含了warmup训练，微调训练和`LIBERO-plus`仿真评估的训练+评估的Pipeline实现方案。需参考以下文档：
- 训练数据与仿真评估时的3D、4D信息生成的实现方案：@itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen2.markdown 。
- 基于老训练数据v1进行改良与处理得到新训练数据v2 `/B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2/` 的实现方案： @itvlaGpLibPlus/b/d/libplus2/gol/3d4d_gen2_2.markdown 。
- 基于老训练数据v1进行warmup训练的实现方案：@itvlaGpLibPlus/b/d/libplus2/gol/p1_warmup1.md 。
- 基于`p1_warmup1.md`方案warmup训练出来的checkpoint进行微调的方案：@itvlaGpLibPlus/b/d/libplus2/gol/p2_sft1.md 。
- 基于`p2_sft1.md`方案微调训练出来的checkpoint进行仿真评估的方案：@itvlaGpLibPlus/b/d/libplus2/gol/eval1.md 。

你的Pipeline包括了warmup训练、微调训练与`LIBERO-plus`仿真评估，所以除了之前在warmup训练方案`p1_warmup1.md`和微调训练方案`p2_sft1.md`以及仿真评估方案`eval1.md`这三个方案中要注意的事项外，还要注意以下事项：
- EXPR_NAME 变量定义为 `4dwvlaLbPlusGolV2_1001`。
- 除了自身Pipeline的实现方案外，还要另外包含3部分，分别对应与参考warmup训练方案`p1_warmup1.md`、微调训练方案`p2_sft1.md`以及仿真评估方案`eval1.md`。都要自包含与完整。内容结构也参考之前的方案。
- 训练数据用的是新训练数据v2 `/B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2/`,所以训练要基于对该数据的深入分析，而且数据schema，yaml配置文件等都要用该数据配套的。
- 要细化到代码逻辑实现，设计要做到修改大于实现，尽量不要修改已有代码。代码生成在 @itvlaGpLibPlus/b/s/libplus2/gol2/ 中。测试与验收脚本要覆盖完全。文档要完整和自包含。
- Pipeline 中的warmup训练、SFT训练、仿真评估三个阶段或子方案都要各自汇总一下各自子方案所涉及的各个超参、参数、变量等配置的有效值是什么，为什么要设这个值，这个值是在哪里设的。也各自汇总一下各自子方案，涉及到了哪些文件增删改，要改文件的哪些内容，为什么要这样做。还要各自在各自的子方案中汇总一下，这个该方案在实际做微调训练时所涉及的关键路径，包括但不限于checkpoint和日志等保存路径。
- warmup训练的`freeze_learnable_tokens`设为 true，SFT训练的`freeze_learnable_tokens`设为 false 。
- 仿真评估时的`inference_backend`确保为 standard 。仿真评估时若遇到某次评估或trial没做成功时要保存该任务的id、动作序列、随机种子等供后续复现用，同时要把如何基于保存的信息进行复现写成详细的操作指南。

请把该"训练+评估的Pipeline实现方案"写在 @itvlaGpLibPlus/b/d/libplus2/gol2/pipeline1.markdown 中。

---

先清空占用GPU的所有程序和进程。然后，请根据 "包含了warmup训练，微调训练和`LIBERO-plus`仿真评估的训练+评估的Pipeline实现方案"  @itvlaGpLibPlus/b/d/libplus2/gol2/pipeline3.markdown 的指示, 去实施在数据`/B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2/`上做的warmup训练，微调训练和仿真评估，创建或更新python虚拟环境、生成和修改代码或脚本等文件, 并执行文档中的warmup训练、SFT训练、仿真评估以及所有测试与验收。过程中若遇到error就fix, 直到所有测试和验收都通过、warmup训练和SFT训练都成功完成以及它们的日志和checkpoint都成功保存、 `LIBERO-plus`上的评估也都成功完成。 记录过程中的一切细节，包括但不限于：所有的error及其根因分析，fix方案，记录所有的操作、命令、关键路径和任何文件的增删改以及做这些操作的原因，过程中的一切细节都记录在 @itvlaGpLibPlus/b/d/libplus2/gol2/pipeline3_1001LOG.markdown 的后面。

---

由于之前发生过评估脚本不释放GPU的问题，请分析该问题的根因，在 eval_v2_wrapper.sh 里加 server 收尾逻辑或基于根因用其它方法 fix 掉这个问题。

fix掉上述问题后，请先对 "包含了warmup训练，微调训练和`LIBERO-plus`仿真评估的训练+评估的Pipeline实现方案"  @itvlaGpLibPlus/b/d/libplus2/gol2/pipeline3.markdown 进行一些修改，将 Pipeline 方案中的 warmup 训练的 epoch 数改为 **1**，SFT 训练的 epoch 数改为 **3** 且 SFT 阶段的`p_schedule`改为 **[0.3, 0.1]**, 与这些超参相关的配置也要作出相应改变，比如 step 数和 learning rate 相关的一些配置等等，除了这些，别忘了把 EXPR_NAME 变量定义为 `4dwvlaLbPlusGolV2_1002`以及修改相应的配置和路径。修改后的版本保存到 @itvlaGpLibPlus/b/d/libplus2/gol2/pipeline3_2.markdown 。然后，修改后并检查无误后，请基于这个新 Pipeline 方案`pipeline3_2.markdown`的指示，去实施在数据`/B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2/`上做的warmup训练，微调训练和仿真评估，创建或更新python虚拟环境、生成和修改代码或脚本等文件, 并执行文档中的warmup训练、SFT训练、仿真评估以及所有测试与验收。过程中若遇到error就fix, 直到所有测试和验收都通过、warmup训练和SFT训练都成功完成以及它们的日志和checkpoint都成功保存、 `LIBERO-plus`上的评估也都成功完成。 记录过程中的一切细节，包括但不限于：所有的error及其根因分析，fix方案，记录所有的操作、命令、关键路径和任何文件的增删改以及做这些操作的原因，过程中的一切细节都记录在 @itvlaGpLibPlus/b/d/libplus2/gol2/pipeline3__2_1002LOG.markdown 的后面。

把实验`4dwvlaLbPlusGolV2_1002`相关的所有日志、测评结果、中间结果等都打包到 /home/a26113/b/Ckp/ 中。并清除所有占用GPU的程序和相关进程。并在 @itvlaGpLibPlus/b/d/libplus2/gol2/pipeline3_2.markdown 里的章节`## 4. Phase 3：LIBERO-plus 评估` 中加入“在评估成功后或失败而挂起30分钟时，启动 bigmatrix_multiply_optimization.py 后台任务占GPU, 然后把评估结果、中间产物、日志日志文件打包到 /home/a26113/b/Ckp/ 中" 的功能。

# eval small

请根据"包含了warmup训练，微调训练和`LIBERO-plus`仿真评估的训练+评估的Pipeline实现方案"  @itvlaGpLibPlus/b/d/libplus2/gol2/pipeline3_2.markdown 中的章节`## 4. Phase 3：LIBERO-plus 评估`进行`EVAL_MODE`为 full 的 Evaluation, 但要显式设置`MAX_STEPS_OVERRIDE`为**100**，以便更快地做完 Evaluation 。本次 LIBERO-plus 评估的 EXPR_NAME 为 `4dwvlaLbPlusGolV2_1002_evlsml`,使用的checkpoint是`/home/a26113/b/Ckp/4dwvlaLbPlusGolV2_1002/2026_10_04_02_17_33-internvla_a1_5-lbplus-golv2-sft/checkpoints/012015/pretrained_model/`。`LIBERO-plus`评估的过程中若遇到error就fix, 直到所有测试和验收都通过且`LIBERO-plus`的`libero_goal`子集上的所有评估都完成。 记录过程中的一切细节，包括但不限于：所有的error及其根因分析，fix方案，记录所有的操作、命令、关键路径和任何文件的增删改以及做这些操作的原因，过程中的一切细节都记录在 @itvlaGpLibPlus/b/d/libplus2/gol2/pipeline3_2_1002evlsmlLOG.markdown 的后面。
