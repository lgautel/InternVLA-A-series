# Debug


# 数据处理

请基于新的Franka插插座数据集( `/B/Dta/plug_into_socket_franka3_15hz_lerobot/` )和它的数据分析报告 @b/d/Frk2/ds2_analyz.md 写一份从这个数据集生成3D关键点及其旋转分量四元数和EEF位姿等3D/4D信息的实施落地方案. 可参考以下材料:
+ 旧的生成3D/4D信息和warmup与sft的文档都在 @b/d/Frk/ 里. 但这些设计和实施方案文档**有可能包含了错误**, 因为它在真机实验中产生了Franka机械臂抖动和夹爪无法夹上的问题, 相关问题都记录在 @b/d/Frk2/realwrld_debug/ 中的各个 md 文档中. 为什么只是"有可能包含了错误", 因为真机实验的环境与数据采集时不一样了,所以也有可能只是环境变化的问题.
+ Franka的urdf @b/d/Frk2/fr3v2_1_franka_hand.urdf

要写的这份新的"从Franka插插座数据集生成3D关键点及其旋转分量四元数和EEF位姿等3D/4D信息的实施落地方案"有如下要求:

+ 要考虑到如何达到训推一致性.
+ 要考虑到Franka的机器特性(现在用的是`Franka Research 3`, 控制器/系统镜像是`System Image 5.10`,软件侧是通过franky+libfranka来对机器人实施控制的)
+ 要考虑到 3D keypoint 轨迹, 旋转四元数, 和EEF位姿 等4D信息应该如何抽取, 这些4D信息的计算要考虑到训推一致性以及与真实的机器和实验环境结合等等. 
+ 要考虑周全一点, 要写的和解释得细致一点, 要细化到代码和脚本级别, 要有充足的测试和验收脚本以保证逻辑正确. 
+ 注意, **不要搜索 /home/a26113/**, **不要相信 opvla_libero_merged_kpt.tar.gz 和 opvla_libero_merged_kpt 数据集的信息和相关代码,那可能是错的,只能参考**.
+ 代码和脚本可生成在 @b/s/Frk2/ 中. 使用的python虚拟环境是 `/B/VENV/itnvla15rbt20/`. HF_HOME 在 `/B/VENV/hf_home/`.
+ 要基于对当前代码的深入分析, 尽量扩展和复用当前的代码.
+ 总之, 目的是使得在处理后的数据上训练出来的模型能在真实Franka的真实插插座实验上得到更好的评估分数. 

方案请写在 @b/d/Frk2/3d4d_gen_1.md.



请按 @b/d/Frk2/3d4d_gen_1.md 对数据集`/B/Dta/plug_into_socket_franka3_15hz_lerobot/`执行生成"3D关键点及其旋转分量四元数和EEF位姿等3D/4D信息"的操作, 新的包含了3D/4D信息的数据集保存在 `~/b/Dta/plug_into_socket_franka3_15hz_lerobot_4d/` 中. 过程中若遇到error就fix, 直到所有测试和验收都通过并且新的包含了3d/4d信息的数据集成功生成. 记录过程中的一切细节, 包括但不限于:所有的error及其根因分析, fix方案, 记录所有的操作, 命令, 关键路径和任何文件的增删改以及做这些操作的原因, 过程中的一切细节都记录在 @b/d/Frk2/3d4d_gen_1_0918LOG.md 后面. 

上面你提到了"（1）图像 stats 是 10000 帧抽样，不是 33308" 这个问题, 也提到"训练时相机键会被 ImageNet mean/std 覆盖，这份图像 stats 基本不生效。", 请问用standard模式推理时, 是怎么做图像的处理的, 也和训练一样吗. 了如果我们对"训练时相机键会被 ImageNet mean/std 覆盖，这份图像 stats 基本不生效。"进行修改, 即训练时不用"ImageNet mean/std", 而是用"图像 stats", 会有什么优点缺点和风险, 请深入分析一下.

# Warmup

## 参考资料以及相关要求
- 参考数据分析和处理文档: @b/d/Frk2/ds2_analyz.md , @b/d/Frk2/3d4d_gen_1.md , @b/d/Frk2/3d4d_gen_1_0918LOG.md .
- 参考训练优化建议: b/d/p/itvla_trn_strategy.markdown , 按照它的章节 `### 6.6 P1：图像增强默认关闭，而且当前增强不保证时序一致`的建议, "开启同步 brightness/contrast/saturation"开启这三种数据增强
- 参考老版的 warmup 实施方案: @b/d/Frk/plug_p1warmup.md 和 b/d/Frk/plug_p1warmup_0907LOG.md . 要参考老版实施方案的代码规范去命名与编写代码, 参考老版方案的日志 & checkpoint & 其它中间结果的保存模式去保存各种结果.

## 用户提出的要求
写一篇基于第二版 Franka 机器人的 `单机器臂插插座的数据`(在 `~/b/Dta/plug_into_socket_franka3_15hz_lerobot_4d/` 中) 进行warmup训练的实施落地方案(含详细操作手册), 写到 `b/d/Frk2/plug2_p1warmup.md` 中. 写该 "warmup训练的实施落地方案(含详细操作手册)" 时要注意以下事项:
- EXPR_NAME 变量定义为 `itvlagpFrkPlug2_0918`.
- 方案要基于对目前服务器的软硬件环境的深入分析.
- 方案要基于对数据 `~/b/Dta/plug_into_socket_franka3_15hz_lerobot_4d/`的深入分析. action mode 使用 abs 模式.
- `关键变量定义` 参考 `b/d/Frk/plug_p1warmup.md` 中的相关内容.
- Franka 机器人数据做训练时要跑 6 epoch, 每3个epoch保存一次checkpoint, 最后一个epoch或step训完后也要保存一次checkpoint. 且要基于总poch数和总batch size 128 去计算要训练的总step数, 以及保存checkpoint时的step数.
- 要在python虚拟环境 VENV_ROOT 中进行操作, 且要先用 source 命令去 activate 该环境, 而不仅是简单用里面的python. 且虚拟环境需要可以灵活置.
- 代码或项目路径用本代码库的路径, 不要写死, 要能推理出来那种.
- 所有输出的保存路径参考  `b/d/Frk/plug_p1warmup.md` 的做法. 即保存训练各种日志和wandb/tensorboard文件的路径, 要和保存模型权重heckpoint的路径分开, 各种日志和wandb/tensorboard文件可保存在/B/Log/${EXPR_NAME} 中, 训练时保存的模型权重checkpoint要放到 ~/b/Ckp/{EXPR_NAME} 中. 所有的checkpoint, 日志, 配置文件的路径都要以 EXPR_NAME 做隔离, 避免不同实验间互相覆盖. 同一实验的不同运行批次的heckpoint, 日志, 配置文件等最好在 EXPR_NAME 对应的文件夹中以时间戳文件夹做隔离. 特别注意的是, 保存时要避免覆盖已有的文件.
- 文档要自包含, 不要引用其它文档的内容, 但要标明内容的出处.  文档要包含详细的测试和验收部分.
- 方案要基于对现有代码的深入分析, 基于实际已有的代码. 同时要尽量复用已有的脚本或代码. 同时要写清楚复用了哪些脚本或代码, 新增和修改了哪些脚或代码, 新增或修改的内容是什么, 为什么这样做.
- 要有专门章节说明训练时的有效超参或者说在训练时会生效的超参以及它生效的值, 同时要说清楚这些生效值是在哪个文件被定义的, 这个超参的含义和作是什么, 超参值的不同会带来什么影响, 以及这个超参为什么要设置为这个值. 也要有专门章节说明训练时哪些模块被冻结, 哪些模块没有, 哪些模块的权会被更新, 各个模块更新的learning rate是多大等等.
- 文档要把要增删改哪些文件或代码, 以及增删改哪些内容都要列出来, 要细到代码级别, 也要详细解释为什么要做这样的增删改, 也要列出复用了什么, 以为什么要这样复用. 也要有专门章节描述, 当训练开始后, 各个脚本和代码间的调用顺序, 以及它们各自负责什么工作, 数据是如何流动的, 输入和输出分涉及什么关键目录等等.
- 文档要包括操作手册部分, 操作手册要详细到即使对该项目一无所知的第三方工程师按照该操作手册部分一步一步地执行也能成功解决问题完成训练. 除了一步一步的步骤也写清楚外, 也要写清楚做训练前, 用户要做什么? 要收集并提供什么信息? 要配置什么? 如何配置? 等等.
- 文档也要设计这样的训练脚本, 该训练脚本要有等待训练进入稳定期后会进行定时监控的功能, 默认每15分钟检查一次(间隔时间可配置), 以监控该训练有没有成功训练完而得到最后完整的结束还是在训练过程中出错卡住了: 
	+ 如果是出错卡住, 有可能GPU在跑, 但日志15分钟没变化或者GPU连续15分钟没有任何程序或进程在跑, 而且训练日志和checkpoint也不完整, 这情况下,  要先清空占用GPU的程序或进程, 执行后台任务占GPU, 也就是调用`nohup python -u <本代码库根目录>/b/d/GpRbtbigmatrix_multiply_optimization.py > /tmp/bigmatrix_multiply_optimization.log 2>&1 &` 和`disown`, 如果遇到error就fix, 到 bigmatrix_multiply_optimization.py 成功在后台启动并开始占用GPU为止. 然后把整个 /B/Log/${EXPR_NAME} 文件夹打包成名为 "{EXPR_NAME}_LOG_<精确到小时的时间戳>_err" 的 tar 包, 拷贝到 ~/b/Ckp/ 中.
	+ 如果训练完全成功, GPU会连续15分钟没有任何程序或进程在跑, 但训练日志和checkpoint等产出物是完整的全部输出的, 这种情况下,  要先清空用GPU的程序或进程, 再执行后台任务占GPU, 也就是调用`nohup python -u <本代码库根目录>/b/d/GpRbtbigmatrix_multiply_optimization.py > /tmp/bigmatrix_multiply_optimization.log 2>&1 &` 和`disown`, 如果遇到error就fix, 到 bigmatrix_multiply_optimization.py 成功在后台启动并开始占用GPU为止. 然后把整个 /B/Log/${EXPR_NAME} 文件夹打包成名为 "{EXPR_NAME}_LOG_<精确到小时的时间戳>" 的 tar 包, 拷贝到 ~/b/Ckp/ 中(若目录不存在则新建一个).  
	+ 上面提到的<精确到小时的时间戳>, 大概长这样`26091813`.
- 最重要的是, 对代码, 配置或其它文件所做的增删改等操作要兼容之前的"基于 RoboTwin 2.0 仿真环境的训练数据进行SFT微调训练"和"基于 R1Pro 机器人的数据进行SFT微调训练" 的功能, 不能因为加入一个功能而破坏了以前的功能.

---
请参考上述列出的`参考资料`, 以及按照`用户提出的要求`, 将基于第二版 Franka 机器人的 `单机器臂插插座的数据`(在 `~/b/Dta/plug_into_socket_franka3_15hz_lerobot_4d/` 中) 进行warmup训练的实施落地方案(含详细操作手册) 写到 @b/d/Frk2/plug2_p1warmup.md 中.

改良一下 @b/d/Frk2/plug2_p1warmup.md , 代码和脚本要生成在 @b/s/Frk2/ 中, 但要尽量复用或扩展现有的代码. 使用的python虚拟环境是 `/B/VENV/itnvla15rbt20/`. HF_HOME 在 `/B/VENV/hf_home/`.

## 基于方案执行warmup

请按照"基于第二版 Franka 机器人的 `单机器臂插插座的数据`(在 `~/b/Dta/plug_into_socket_franka3_15hz_lerobot_4d/`中) 进行warmup训练的实施落地方案(含详细操作手册)"  @b/d/Frk2/plug2_p1warmup.md 里的指示 ,  用InternVLA-A1.5-base 对`~/b/Dta/plug_into_socket_franka3_15hz_lerobot_4d/`中的数据进行warmup训练. 训练过程中若遇到error就fix, 直到warmup训练成功完成而且相关测试和验收也全都通过. 记录所有训练过程中的一切细节, 包括但不限于:所有的error及其根因分析, fix方案, 记录所有的操作, 命令, 关键路径和任何文件的增删改以及做这些操作的原因, 过程中的一切细节都记录在 @b/d/Frk2/plug_p1warmup_0907LOG.md 后面. 在warmup成功后或失败而挂起时, 启动 bigmatrix_multiply_optimization.py 后台任务占GPU, 然后把日志文件打包到 ~/b/Ckp/ 中.  开始warmup训练吧.







# SFT






