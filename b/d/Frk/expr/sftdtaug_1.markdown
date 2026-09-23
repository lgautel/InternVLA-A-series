# Prmp

先清空GPU上所有程序与进程及其相关进程.

---

然后开始基于旧的 Franka 插插座数据的微调训练:
深入分析 @b/d/Frk/ 里的文档和相关代码和脚本和配置文件, 特别是与微调相关的内容, 深入分析旧的Franka插插座数据 `/B/Dta/plug_into_socket_lrb_4D`, 参考 @b/s/Frk2/frk2_plug_warmup_launch.sh 的做法, 启用除"affine"以外的其它数据增强手段,  对保存在 `/home/a26113/b/Ckp/itvlagpFrkPlug0907/2026_09_07_07_17_08-internvla_a1_5-frk-plug-sft/checkpoints/010420/pretrained_model/`的checkpoint进行继续微调训练, 一直训练到50000 steps为止. `EXPR_NAME` 设为 `itvlagpFrkPlug0907_dtaug`, 这也意味着checkpoint和日志等的保存路径的改变.  过程中若遇到error就fix, 直到所有相关测试和验收都通过且训练完全成功并且把checkpoint和日志等中间结果也都成功保存. 记录所有训练过程中的一切细节, 包括但不限于:所有的error及其根因分析, fix方案, 记录所有的操作, 命令, 关键路径和任何文件的增删改以及做这些操作的原因, 过程中的一切细节都记录在 b/d/Frk/plug_p2sft_0907_dtaugLOG.md 后面. 在训练成功后或失败而挂起时, 启动 bigmatrix_multiply_optimization.py 后台任务占GPU, 然后把日志文件打包到 ~/b/Ckp/ 中.  开始微调训练吧.
