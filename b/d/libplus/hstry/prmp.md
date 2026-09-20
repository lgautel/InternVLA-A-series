你同事写了一份数据分析文档,但文档写错了, 错误相信了 opvla_libero_merged_kpt.tar.gz 里的信息,这两份是你同事的分析和讨论历史, 仅供参考 @b/d/libplus/hstry/cursor_libero_training_data_analysis.md, @b/d/libplus/hstry/cursor_libero_training_data_analysis2.markdown . 请重写一篇更好的基于现实的对 `/B/Dta/opvla_libero/` 这份机器人训练数据进行深入分析的文档.

该数据分析文档有如下要求:
- 要考虑到如何达到训推一致性. 可参考训练完后要进行评估的LIBERO 仿真 benchmark的代码`/B/SRC/LIBERO/` , 和 LIBERO-plus 仿真benchmark的代码`/B/SRC/LIBERO-plus/` , 以及运行这些仿真benchmark的python虚拟环境`/B/VENV/libero_plus_client/`
- 要考虑到 3D keypoint 轨迹, 旋转四元数, 和EEF位姿 等4D信息应该如何抽取, 这些4D信息的计算要考虑到训推一致性以及仿真引擎与评估任务的特点等等. 
- 可参考 @b/d/libplus/ 里面的各个文档, 避免踩之前踩过的坑.
- 分析文档要完整,要自包含.
- 注意, **不要搜索 /home/a26113/**, **不要相信 opvla_libero_merged_kpt.tar.gz 和 opvla_libero_merged_kpt 数据集的信息,那可能是错的,只能参考**.

这份机器人训练数据的分析文档请写在 @b/d/libplus/ds/libero_raw_analyz3.md 中.

---


我打算**重新生成3D关键点及其旋转分量四元数并重训**, 因此请你基于对当前代码库的真实代码进行深入分析的基础上, 参考以下材料:

+ 之前写类似的3D关键点及其旋转分量四元数的方案的相关分析过程 @b/d/libplus/hstry/cursor_lbplus_4d_gen.md
+ 旧的从原始数据( /B/Dta/opvla_libero/ )抽取3D 关键点轨迹和每个点的旋转分量四元数并形成`/B/Dta/opvla_libero_merged_kpt/`数据集的代码, 但代码中的方法可能有误或者不是最优: @util_scripts/export_panda_mjcf.py, @src/lerobot/dataset_schemas/configs/libero.yaml , @util_scripts/split_libero_merged.py , @util_scripts/smoke_test_libero_fk.py , @util_scripts/generate_libero_keypoints.py , @util_scripts/setup_libero_geop.sh . 这套代码可能有问题, 所以转出来的数据( /B/Dta/opvla_libero_merged_kpt/ )可能也有问题.
+ 基于 /B/Dta/opvla_libero_merged_kpt/ 数据做微调的实施方案在 @b/d/libplus/sft.md , 做LIBERO 和 LIBERO-plus评估的实施方案在 @b/d/libplus/eval3.md , @b/d/libplus/eval3_optim2.md 和 @b/d/libplus/eval3_optim3.md . 但当时抽取3D关键点及其四元数并形成`opvla_libero_merged_kpt` 数据集的方法可能是错误或者不是最优的.
+ 原始数据在 /B/Dta/opvla_libero/ , 对它的数据分析在  @b/d/libplus/ds/libero_raw_analyz4.md 和 @b/d/libplus/ds/ds_libplus_analyz.md 中. 
+ 仿真环境里的机器人信息在 @b/d/libplus/panda_arm_hand.urdf. 能运行LIBERO仿真环境的python虚拟环境在 `/B/VENV/libero_plus_client/`. 但注意, 那个 /B/SRC/itvlaGpLibPlus/b/d/libplus/panda_arm_hand.urdf **不一定是最适合的**, 你最好自己找一个更适合的来用.

然后写一份新的, 能考虑训推一致性的, 能提升训练出来的模型在LIBERO和LIBERO-plus上的评估效果的"3D关键点及其旋转分量四元数"的实施落地方案. 该方案有如下要求:

+ 要考虑到如何达到训推一致性. 可参考训练完后要进行评估的LIBERO 仿真 benchmark的代码`/B/SRC/LIBERO/` , 和 LIBERO-plus 仿真benchmark的代码`/B/SRC/LIBERO-plus/` , 以及运行这些仿真benchmark的python虚拟环境`/B/VENV/libero_plus_client/`.
+ 要考虑到 3D keypoint 轨迹, 旋转四元数, 和EEF位姿 等3D/4D信息应该如何抽取, 这些3D/4D信息的计算要考虑到训推一致性以及仿真引擎与评估任务的特点等等. 
+ 可参考 @b/d/libplus/ 里面的各个文档, 避免踩之前踩过的坑.
+ 代码和脚本可生成在 @b/s/libplus/ 中. VLA代码使用的python虚拟环境是 `/B/VENV/itnvla15rbt20/`. HF_HOME 在 `/B/VENV/hf_home/`.
+ 要基于对当前代码的深入分析, 尽量扩展和复用当前的代码.
+ 要考虑周全一点, 要写的和解释得细致一点, 要细化到代码和脚本级别, 要有充足的测试和验收脚本以保证逻辑正确. 
+ 注意, **不要搜索 /home/a26113/**, **不要相信 opvla_libero_merged_kpt.tar.gz 和 opvla_libero_merged_kpt 数据集的信息和相关代码,那可能是错的,只能参考**.
+ 总之, 目的是使得在处理后的数据上训练出来的模型能在LIBERO和LIBERO-plus上得到更好的评估分数. 

方案请写在 @b/d/libplus/ds/3d4d_gen_0918_2.md .


请按 @b/d/libplus/ds/3d4d_gen_0918_2.md 对数据集`/B/Dta/opvla_libero/`执行生成"3D关键点及其旋转分量四元数和EEF位姿等3D/4D信息"的操作, 新的包含了3D/4D信息的数据集保存在 `~/b/Dta/opvla_libero_4d/` 中. 过程中若遇到error就fix, 直到所有测试和验收都通过并且新的包含了3d/4d信息的数据集成功生成. 记录过程中的一切细节, 包括但不限于:所有的error及其根因分析, fix方案, 记录所有的操作, 命令, 关键路径和任何文件的增删改以及做这些操作的原因, 过程中的一切细节都记录在 @b/d/libplus/ds/3d4d_gen_0918_2_0918LOG.md 后面. 
