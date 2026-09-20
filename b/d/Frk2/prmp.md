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