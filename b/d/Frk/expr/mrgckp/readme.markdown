# Prmp

写一个独立的程序, 可以将两个checkpoint中的相同权重按不同比例做加权平均. 我现在需要对 `/home/a26113/b/Ckp/itvlagpFrkPlug0907_dtaug2/2026_09_22_20_04_17-internvla_a1_5-frk-plug-sft-dtaug2/checkpoints/005000/pretrained_model/` 和 `/home/a26113/b/Ckp/itvlagpFrkPlug0907/2026_09_07_07_17_08-internvla_a1_5-frk-plug-sft/checkpoints/010420/pretrained_model/` 这两个checkpoint中名字相同的权重进行加权平均的merge, 前者是数据增强过的5000步的checkpoint, 权重0.3, 后者没做数据增强的checkpoint权重0.7 , 两个merge后得到新的checkpoint保存在 `/home/a26113/b/Ckp/itvlagpFrkPlug0907_dtaug2/2026_09_22_20_04_17-internvla_a1_5-frk-plug-sft-dtaug2/mergepoints/005000/pure1w_0_7/`中. 要写smoke test保证新checkpoint能被transformers和lerobot框架读取并训练. 代码写在 @b/d/Frk/expr/mrgckp/ 中, 代码文件名字自己起, 但不要影响同一项目的其它文件.

请执行下面脚本
```bash
python3 b/d/Frk/expr/mrgckp/merge_checkpoints_multi.py.py \
    --ckpt /home/a26113/b/Ckp/itvlagpFrkPlug0907_dtaug2/2026_09_22_20_04_17-internvla_a1_5-frk-plug-sft-dtaug2/checkpoints/005000/pretrained_model/ --weight 0.15 \
    --ckpt /home/a26113/b/Ckp/itvlagpFrkPlug0907_dtaug/2026_09_22_07_05_00-internvla_a1_5-frk-plug-sft-dtaug/checkpoints/010000/pretrained_model/ --weight 0.15 \
    --ckpt /home/a26113/b/Ckp/itvlagpFrkPlug0907/2026_09_07_07_17_08-internvla_a1_5-frk-plug-sft/checkpoints/010420/pretrained_model/ --weight 0.7 \
    --output /home/a26113/b/Ckp/itvlagpFrkPlug0907_dtaug2ps/mrg_ckp/datw5k015da1w015pure1w07/
```
就可以把`/home/a26113/b/Ckp/itvlagpFrkPlug0907_dtaug2/2026_09_22_20_04_17-internvla_a1_5-frk-plug-sft-dtaug2/checkpoints/005000/pretrained_model/` 按权重**0.15** , `/home/a26113/b/Ckp/itvlagpFrkPlug0907_dtaug/2026_09_22_07_05_00-internvla_a1_5-frk-plug-sft-dtaug/checkpoints/010000/pretrained_model/` 按权重**0.15**, `/home/a26113/b/Ckp/itvlagpFrkPlug0907/2026_09_07_07_17_08-internvla_a1_5-frk-plug-sft/checkpoints/010420/pretrained_model/`  按权重**0.7** 的方式做加权平均的三个checkpoint融合. 然后把融合后的新checkpoint的保存到`/home/a26113/b/Ckp/itvlagpFrkPlug0907_dtaug2ps/mrg_ckp/datw5k015da1w015pure1w07/`中. 新checkpoint保存完后, 复制`/home/a26113/b/Ckp/itvlagpFrkPlug0907/2026_09_07_07_17_08-internvla_a1_5-frk-plug-sft/checkpoints/010420/pretrained_model/stats.json`到`datw5k015da1w015pure1w07`中.

