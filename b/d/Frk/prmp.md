

# Phase 1 warmup

## 参考资料
参考 基于 RoboTwin 2.0 仿真环境的训练数据进行warmup训练的文档: @b/d/GpRbt/run_ech_rbt_p012.md , @b/d/GpRbt/run_ech_rbt_p1_0906LOG.md , @b/d/GpRbt/run_ech_rbt_p1.md .
参考 基于 R1Pro 机器人的数据进行微调训练的文档: @b/d/R1Pro/p2sft_planH200_0904LOG.md , @b/d/R1Pro/p2sft_plan.md , @b/d/R1Pro/p2sft_planH200.md . 
参考 Franka 机器人数据的处理方案 @b/d/Frk/dta_4dtrj_plan_0904LOG.md 和 @b/d/Frk/dta_4dtrj_plan.md, 以及Franka机器人本身的URDF文件 @b/d/Frk/fr3v2_1_franka_hand.urdf .

## 用户提出的要求
写一篇基于 Franka 机器人的 `单机器臂插插座的数据`(在 /B/Dta/plug_into_socket_lrb_4D 中) 进行warmup训练的实施落地方案(含详细操作手册), 写到 `b/d/Frk/plug_p1warmup.md` 中. 写该 "warmup训练的实施落地方案(含详细操作手册)" 时要注意以下事项:
- EXPR_NAME 变量定义为 `itvlagpFrkPlug0907`.
- 方案要基于对目前服务器的软硬件环境的深入分析.
- 方案要基于对数据 `/B/Dta/plug_into_socket_lrb_4D`的深入分析. action mode 使用 abs 模式, 因此归一化用的统计量在 /B/Dta/lug_into_socket_lrb_4D/meta/stats/abs/stats.json 中.
- `关键变量定义` 参考 `b/d/rbt/run_ech_rbt_p1.md` 的章节`## 2. 关键变量定义`, `b/d/R1Pro/p2sft_plan.md` 的章节`## 1. 可配置变量总（换机器必读）`, `b/d/R1Pro/p2sft_planH200.md` 的章节`## ⚡ 执行前：需用户提供的信息`.
- Franka 机器人数据做训练时要跑 6 epoch, 每3个epoch保存一次checkpoint, 最后一个epoch或step训完后也要保存一次checkpoint. 且要基于总poch数和总batch size 128 去计算要训练的总step数, 以及保存checkpoint时的step数.
- 要在python虚拟环境 VENV_ROOT 中进行操作, 且要先用 source 命令去 activate 该环境, 而不仅是简单用里面的python. 且虚拟环境需要可以灵活置.
- 代码或项目路径用本代码库的路径, 不要写死, 要能推理出来那种.
- 所有输出的保存路径参考  `b/d/rbt/run_ech_rbt_p1.md` 的做法. 即保存训练各种日志和wandb/tensorboard文件的路径, 要和保存模型权重heckpoint的路径分开, 各种日志和wandb/tensorboard文件可保存在/B/Log/${EXPR_NAME} 中, 训练时保存的模型权重checkpoint要放到 ~/b/Ckp/{EXPR_NAME} 中. 所有的checkpoint, 日志, 配置文件的路径都要以 EXPR_NAME 做隔离, 避免不同实验间互相覆盖. 同一实验的不同运行批次的heckpoint, 日志, 配置文件等最好在 EXPR_NAME 对应的文件夹中以时间戳文件夹做隔离. 特别注意的是, 保存时要避免覆盖已有的文件.
- 文档要自包含, 不要引用其它文档的内容, 但要标明内容的出处.  文档要包含详细的测试和验收部分.
- 方案要基于对现有代码的深入分析, 基于实际已有的代码. 同时要尽量复用已有的脚本或代码. 同时要写清楚复用了哪些脚本或代码, 新增和修改了哪些脚或代码, 新增或修改的内容是什么, 为什么这样做.
- 要有专门章节说明训练时的有效超参或者说在训练时会生效的超参以及它生效的值, 同时要说清楚这些生效值是在哪个文件被定义的, 这个超参的含义和作是什么, 超参值的不同会带来什么影响, 以及这个超参为什么要设置为这个值. 也要有专门章节说明训练时哪些模块被冻结, 哪些模块没有, 哪些模块的权会被更新, 各个模块更新的learning rate是多大等等.
- 文档要把要增删改哪些文件或代码, 以及增删改哪些内容都要列出来, 要细到代码级别, 也要详细解释为什么要做这样的增删改, 也要列出复用了什么, 以为什么要这样复用. 也要有专门章节描述, 当训练开始后, 各个脚本和代码间的调用顺序, 以及它们各自负责什么工作, 数据是如何流动的, 输入和输出分涉及什么关键目录等等.
- 文档要包括操作手册部分, 操作手册要详细到即使对该项目一无所知的第三方工程师按照该操作手册部分一步一步地执行也能成功解决问题完成训练. 除了步一步的步骤也写清楚外, 也要写清楚做训练前, 用户要做什么? 要收集并提供什么信息? 要配置什么? 如何配置? 等等.
- 文档也要设计这样的训练脚本, 该训练脚本要有等待训练进入稳定期后会进行定时监控的功能, 默认每15分钟检查一次(间隔时间可配置), 以监控该训练有没有成功训练完而得到最后完整的结束还是在训练过程中出错卡住了: 
	+ 如果是出错卡住, 有可能GPU在跑, 但日志15分钟没变化或者GPU连续15分钟没有任何程序或进程在跑, 而且训练日志和checkpoint也不完整, 这情况下,  要先清空占用GPU的程序或进程, 执行后台任务占GPU, 也就是调用`nohup python -u <本代码库根目录>/b/d/GpRbtbigmatrix_multiply_optimization.py > /tmp/bigmatrix_multiply_optimization.log 2>&1 &` 和`disown`, 如果遇到error就fix, 到 bigmatrix_multiply_optimization.py 成功在后台启动并开始占用GPU为止. 然后把整个 /B/Log/${EXPR_NAME} 文件夹打包成名为 "{EXPR_NAME}_LOG_<精确到小时的时间戳>_err" 的 tar 包, 拷贝到 ~/b/Ckp/ 中.
	+ 如果训练完全成功, GPU会连续15分钟没有任何程序或进程在跑, 但训练日志和checkpoint等产出物是完整的全部输出的, 这种情况下,  要先清空用GPU的程序或进程, 再执行后台任务占GPU, 也就是调用`nohup python -u <本代码库根目录>/b/d/GpRbtbigmatrix_multiply_optimization.py > /tmp/bigmatrix_multiply_optimization.log 2>&1 &` 和`disown`, 如果遇到error就fix, 到 bigmatrix_multiply_optimization.py 成功在后台启动并开始占用GPU为止. 然后把整个 /B/Log/${EXPR_NAME} 文件夹打包成名为 "{EXPR_NAME}_LOG_<精确到小时的时间戳>" 的 tar 包, 拷贝到 ~/b/Ckp/ 中(若目录不存在则新建一个).  
	+ 上面提到的<精确到小时的时间戳>, 大概长这样`26090413`.
- 最重要的是, 对代码, 配置或其它文件所做的增删改等操作要兼容之前的"基于 RoboTwin 2.0 仿真环境的训练数据进行SFT微调训练"和"基于 R1Pro 机器人的数据进行SFT微调训练" 的功能, 不能因为加入一个功能而破坏了以前的功能.

---
请参考上述列出的`参考资料`, 以及按照`用户提出的要求`, 将基于 Franka 机器人的 `单机器臂插插座的数据`(在 /B/Dta/plug_into_socket_lrb_4D 中) 进行warmup训练的实施落地方案(含详细操作手册) 写到 @b/d/Frk/plug_p1warmup.md 中.

## 基于方案执行warmup

请按照"基于 Franka 机器人的 `单机器臂插插座的数据`(在 /B/Dta/plug_into_socket_lrb_4D 中) 进行warmup训练的实施落地方案(含详细操作手册)"  @b/d/Frk/plug_p1warmup.md 里的指示 ,  用InternVLA-A1.5-base 对 /B/Dta/plug_into_socket_lrb_4D 中的数据进行warmup训练. 训练过程中若遇到error就fix, 直到warmup训练成功完成而且相关测试和验收也全都通过. 记录所有训练过程中的一切细节, 包括但不限于:所有的error及其根因分析, fix方案, 记录所有的操作, 命令, 关键路径和任何文件的增删改以及做这些操作的原因, 过程中的一切细节都记录在 @b/d/Frk/plug_p1warmup_0907LOG.md 后面. 在warmup成功后或失败而挂起时, 启动 bigmatrix_multiply_optimization.py 后台任务占GPU, 然后把日志文件打包到 ~/b/Ckp/ 中.  开始warmup训练吧.


# Phase 2 SFT

## 参考资料
参考 基于 RoboTwin 2.0 仿真环境的训练数据进行SFT微调训练的文档: @b/d/GpRbt/run_ech_rbt_p012.md , @b/d/GpRbt/sft0827LOG.md , @b/d/GpRbt/sft0827.md @b/d/GpRbt/run_ech_rbt_p2_0906LOG.md , @b/d/GpRbt/run_ech_rbt_p2.md .
参考 基于 R1Pro 机器人的数据进行SFT微调训练的文档: @b/d/R1Pro/p2sft_planH200_0904LOG.md , @b/d/R1Pro/p2sft_plan.md , @b/d/R1Pro/p2sft_planH200.md . 

参考 Franka 机器人数据的处理方案 @b/d/Frk/dta_4dtrj_plan_0904LOG.md 和 @b/d/Frk/dta_4dtrj_plan.md, Franka 机器人的warmup训练方案 @b/d/Frk/plug_p1warmup_0907LOG.md 和 @b/d/Frk/plug_p1warmup.md , 以及Franka机器人本身的URDF文件 @b/d/Frk/fr3v2_1_franka_hand.urdf .

## 用户提出的要求
写一篇基于 Franka 机器人的 `单机器臂插插座的数据`(在 /B/Dta/plug_into_socket_lrb_4D 中)用warmup训练出来的checkpoint(`/home/a26113/b/Ckp/itvlagpFrkPlug0907/2026_09_07_04_53_50-internvla_a1_5-frk-plug-warmup/checkpoints/003126/pretrained_model`) 进行SFT训练的实施落地方案(含详细操作手册), 写到 `b/d/Frk/plug_p2sft.md` 中. 写该 "SFT微调训练的实施落地方案(含详细操作手册)" 时要注意以下事项:
 - `EXPR_NAME` 变量定义为 `itvlagpFrkPlug0907`.
 - 方案要基于对目前服务器的软硬件环境的深入分析.
 - 方案要基于对数据 `/B/Dta/plug_into_socket_lrb_4D`的深入分析. action mode 使用 abs 模式, 因此归一化用的统计在 /B/Dta/plug_into_socket_lrb_4D/meta/stats/abs/stats.json 中.
- `关键变量定义` 参考 `b/d/rbt/run_ech_rbt_p2.md` 的章节`## 2. 关键变量定义`, `b/d/R1Pro/p2sft_plan.md` 的章`## 1. 可配置变量总表（换机器必读）`, `b/d/R1Pro/p2sft_planH200.md` 的章节`## ⚡ 执行前：需用户提供的信息`, 特别参考最近的`b/d/Frk/plug_p1warmup.md`的章节`## 2. 可配置变量总表（换机器必读）`.
 - Franka 机器人数据做SFT微调训练时要跑 100 epoch, 每20个epoch保存一次checkpoint, 最后一个epoch或step训完后也要保存一checkpoint. 且要基于总epoch数和总batch size 128 去计算要训练的总step数, 以及保存checkpoint时的step数.
- 要在python虚拟环境 VENV_ROOT 中进行操作, 且要先用 source 命令去 activate 该环境, 而不仅是简单用里面的python. 虚拟环境需要可以灵活配置.
- 代码或项目路径用本代码库的路径, 不要写死, 要能推理出来那种.
- 所有输出的保存路径参考 `b/d/Frk/plug_p1warmup.md` 的做法. 即保存训练各种日志和wandb/tensorboard文件的路径, 和保存模型权重checkpoint的路径分开, 各种日志和wandb/tensorboard文件可保存在`/B/Log/${EXPR_NAME}` 中, 训练时保存的模型权重checkpoint要放到 `~/b/Ckp/${EXPR_NAME}` 中. 所有的checkpoint, 日志, 配置文件的路径都要以 EXPR_NAME 做隔离, 避免不同实验间互相覆盖. 同一实验的不同运行批次的checkpoint, 日志, 配置文件等最好在 `EXPR_NAME` 对应的文件夹中以时间戳文件夹做隔离. 特别注意的是, 保存时要避免覆盖已有的文件.
- 文档要自包含, 不要引用其它文档的内容, 但要标明内容的出处.  文档要包含详细的测试和验收部分.
- 方案要基于对现有代码的深入分析, 基于实际已有的代码. 同时要尽量复用已有的脚本或代码. 同时要写清楚复用了哪些脚本或代码, 新增和修改了哪些脚本或代码, 新增或修改的内容是什么, 为什么这样做.
- 要有专门章节说明训练时的有效超参或者说在训练时会生效的超参以及它生效的值, 同时要说清楚这些生效值是在哪个文件被定义的, 这个超参的含义和作用是什么, 超参值的不同会带来什么影响, 以及这个超参为什么要设置为这个值. 也要有专门章节说明训练时哪些模块被冻结, 哪些模块没有, 哪些模块的权重会被更新, 各个模块更新的learning rate是多大等等.
- 文档要把要增删改哪些文件或代码, 以及增删改哪些内容都要列出来, 要细到代码级别, 也要详细解释为什么要做这样的增删改, 也要列出复用了什么, 以及为什么要这样复用. 也要有专门章节描述, 当训练开始后, 各个脚本和代码间的调用顺序, 以及它们各自负责什么工作, 数据是如何流动的, 输入和输出分别涉及什么关键目录等等.
- 文档要包括操作手册部分, 操作手册要详细到即使对该项目一无所知的第三方工程师按照该操作手册部分一步一步地执行也能成功解决问题完成训练. 除了一步一步的步骤也写清楚外, 也要写清楚做训练前, 用户要做什么? 要收集并提供什么信息? 要配置什么? 如何配置? 等等.
- 文档也要设计这样的训练脚本, 该训练脚本要有等待训练进入稳定期后会进行定时监控的功能, 默认每15分钟检查一次(间隔时间可配置), 以监控该训练有没有成功训练完而得到最后完整的结束还是在训练过程中出错卡住了: 
	+ 如果是出错卡住, 有可能GPU在跑, 但日志15分钟没变化或者GPU连续15分钟没有任何程序或进程在跑, 而且训练日志checkpoint也不完整, 这种情况下,  要先清空占用GPU的程序或进程, 执行后台任务占GPU, 也就是调用`nohup python -u 本代码库根目录>/b/d/GpRbt/bigmatrix_multiply_optimization.py > /tmp/bigmatrix_multiply_optimizationlog 2>&1 &` 和`disown`, 如果遇到error就fix, 直到 bigmatrix_multiply_optimization.py 成功在后台启动并开占用GPU为止. 然后把整个 `/B/Log/${EXPR_NAME}` 文件夹打包成名为 `${EXPR_NAME}_LOG_<精确到小时的时间戳>_err`的 tar 包, 拷贝到 ~/b/Ckp/ 中.
	+ 如果训练完全成功, GPU会连续15分钟没有任何程序或进程在跑, 但训练日志和checkpoint等产出物是完整的全部输出的, 种情况下,  要先清空占用GPU的程序或进程, 再执行后台任务占GPU, 也就是调用`nohup python -u <本代码库根目录>/b/dGpRbt/bigmatrix_multiply_optimization.py > /tmp/bigmatrix_multiply_optimization.log 2>&1 &` `disown`, 如果遇到error就fix, 直到 bigmatrix_multiply_optimization.py 成功在后台启动并开始占用GPU为止. 后把整个 `/B/Log/${EXPR_NAME}` 文件夹打包成名为 `${EXPR_NAME}_LOG_<精确到小时的时间戳>` 的 tar 包, 拷贝到 ~b/Ckp/ 中(若目录不存在则新建一个).  
	+ 上面提到的<精确到小时的时间戳>, 大概长这样`26090413`.
- 最重要的是, 对代码, 配置或其它文件所做的增删改等操作要兼容之前的"基于 RoboTwin 2.0 仿真环境的训练数据进行SFT微调训练"和"基于 R1Pro 机器人的数据进行SFT微调训练" 的功能, 不能因为加入一个功能而破坏了以前的功能.

---
请参考上述列出的`参考资料`, 以及按照`用户提出的要求`, 将基于 Franka 机器人的 `单机器臂插插座的数据`warmup训练出来的checkpoint进行SFT微调训练的实施落地方案(含详细操作手册) 写到 @b/d/Frk/plug_p2sft.md 中.

## 基于方案执行SFT

请按照"基于 Franka 机器人的 `单机器臂插插座的数据`warmup训练出来的checkpoint进行SFT微调训练的实施落地方案(含详细操作手册)"  @b/d/Frk/plug_p2sft.md 里的指示 ,  用在Franka 机器人的 `单机器臂插插座的数据`上进行warmup训练得到的checkpoint(`/home/a26113/b/Ckp/itvlagpFrkPlug0907/2026_09_07_04_53_50-internvla_a1_5-frk-plug-warmup/checkpoints/003126/pretrained_model`), 在 /B/Dta/plug_into_socket_lrb_4D 里的数据上进行SFT微调训练. 训练过程中若遇到error就fix, 直到SFT训练成功完成而且相关测试和验收也全都通过. 记录所有训练过程中的一切细节, 包括但不限于:所有的error及其根因分析, fix方案, 记录所有的操作, 命令, 关键路径和任何文件的增删改以及做这些操作的原因, 过程中的一切细节都记录在 @b/d/Frk/plug_p2sft_0907LOG.md 后面. 在SFT成功后或失败而挂起时, 启动 bigmatrix_multiply_optimization.py 后台任务占GPU, 然后把日志文件打包到 ~/b/Ckp/ 中.  开始SFT微调训练吧.

