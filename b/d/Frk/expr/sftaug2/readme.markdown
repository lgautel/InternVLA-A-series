# Prmp

## 加入 affine

#先清空GPU上所有程序与进程及其相关进程.

---

#然后开始基于旧的 Franka 插插座数据的微调训练:
深入分析 @b/d/Frk/ 里的文档和相关代码和脚本和配置文件, 特别是与微调相关的内容, 深入分析旧的Franka插插座数据 `/B/Dta/plug_into_socket_lrb_4D`, 参考 @b/s/Frk2/frk2_plug_warmup_launch.sh 的做法, 但启用所有数据增强手段(含"affine"), 但对 affine 数据增强要降低发生的概率(比如设置`--dataset.image_transforms.tfs.affine.weight=0.2`而其它数据增强的权重是1.0),  对保存在 `/home/a26113/b/Ckp/itvlagpFrkPlug0907/2026_09_07_07_17_08-internvla_a1_5-frk-plug-sft/checkpoints/010420/pretrained_model/`的checkpoint进行继续微调训练, 一直训练到20000 steps为止. `EXPR_NAME` 设为 `itvlagpFrkPlug0907_dtaug2`, 这也意味着checkpoint和日志等的保存路径的改变.  请先把启动该微调训练的脚本写在 @b/d/Frk/expr/sftaug2/frk_plug_sft_dtaug2_launch.sh , 然后在把如何启动微调训练,如何监控和调试该微调训练的一步步的操作细节作为该微调训练的操作手册章节写到  @b/d/Frk/expr/sftaug2/plug_p2sft_0907_dtaug2LOG.md 中.

## 加入动态aug

停止 dtaug2 训练, 也停止相关的监控进程, 清空GPU上所有程序与进程及其相关进程.

---

请基于 @b/d/Frk/expr/sftaug2/dtaug_p_sft2.md 对代码实施修改, 然后进行测试与验收. 过程中若遇到error就fix, 直到所有相关测试和验收都通过. 过程中所做的涉及到方案改动的事件都要更新到 `dtaug_p_sft2.md` 中去. 过程中的一切细节, 包括但不限于:所有的error及其根因分析, fix方案, 记录所有的操作, 命令, 关键路径和任何文件的增删改以及做这些操作的原因等等, 总之一切过程都以实施日志的名义记录到`dtaug_p_sft2.md`后面. 然后, 根据最终的方案, 再更新一下`dtaug_p_sft2.md`中与操作手册相关的章节, 写清楚该如何一步一步地进行微调训练, 要做什么配置, 跑什么命令, 怎么调试, 日志和checkpoint保存在哪里等等, 要细到代码或命令行级别.

---

深入分析 @b/d/Frk/ 里的文档和相关代码和脚本和配置文件, 特别是与微调相关的内容, 深入分析旧的Franka插插座数据 `/B/Dta/plug_into_socket_lrb_4D`, 按照 @b/d/Frk/expr/sftaug2/dtaug_p_sft2.md 中的操作指引, 用已配置好的 @b/d/Frk/expr/sftaug2/frk_plug_sft_dtaug2ps_launch.sh,  对保存在 `/home/a26113/b/Ckp/itvlagpFrkPlug0907_dtaug2ps/mrg_ckp/datw5k015da1w015pure1w07/`的checkpoint进行继续微调训练, 一直训练到20000 steps为止. `EXPR_NAME` 设为 `itvlagpFrkPlug0907_dtaug2ps`, 这也意味着checkpoint和日志等的保存路径的改变. 过程中若遇到error就fix, 直到所有相关测试和验收都通过且训练完全成功, 同时checkpoint和日志等中间结果也都成功保存. 记录所有训练过程中的一切细节, 包括但不限于:所有的error及其根因分析, fix方案, 记录所有的操作, 命令, 关键路径和任何文件的增删改以及做这些操作的原因, 过程中的一切细节都记录在 @b/d/Frk/expr/sftaug2/dtaug_p_sft2ps0923LOG.md 后面. 在训练成功结束后或失败而挂起超过30分钟时, 把日志文件打包到 ~/b/Ckp/<EXPR_NAME> 里面后启动 bigmatrix_multiply_optimization.py 后台任务占GPU.  


## 监控并启动

请监控这个 PID 为 1911832 的微调训练任务, 当它的第一个 10000 step 的checkpoint在`/home/a26113/b/Ckp/itvlagpFrkPlug0907_dtaug/2026_09_22_07_05_00-internvla_a1_5-frk-plug-sft-dtaug/checkpoints/`确保成功完全保存下来后, 等15分钟后就杀掉该进程, 并清空GPU上所有程序与进程及其相关进程.

然后按 @b/d/Frk/expr/sftaug2/plug_p2sft_0907_dtaug2LOG.md 中的微调操作手册, 启动你的`EXPR_NAME` 设为 `itvlagpFrkPlug0907_dtaug2`的微调训练. 过程中若遇到error就fix, 直到所有相关测试和验收都通过且训练完全成功, 同时checkpoint和日志等中间结果也都成功保存. 记录所有训练过程中的一切细节, 包括但不限于:所有的error及其根因分析, fix方案, 记录所有的操作, 命令, 关键路径和任何文件的增删改以及做这些操作的原因, 过程中的一切细节都记录在 `b/d/Frk/expr/sftaug2/plug_p2sft_0907_dtaug2LOG.md` 后面. 在训练成功结束后或失败而挂起超过30分钟时, 把日志文件打包到 ~/b/Ckp/<EXPR_NAME> 里面后启动 bigmatrix_multiply_optimization.py 后台任务占GPU.  
