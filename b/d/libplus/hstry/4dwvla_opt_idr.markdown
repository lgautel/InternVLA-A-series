use_fast_action_tokens=true
旧版 `evaluation/LIBERO` backend 曾硬编码 use_fast_action_tokens=false, 所以训练时4DWVLA是不是也不要这个,因为`itvla_trn_strategy.md`认为 fast token 的归一化需要的不是现在mean_std. 但要注意设为 false 后的训推一致性.
但itvla建议后训练use_fast_action_tokens=true, 而且当data schema为panda时, ReorderStateActionTransform是空操作, 所以 Fast Token 与 连续型 state emb 一个在 reorder 前, 一个在 reorder 后的影响不大, 所以, use_fast_action_tokens=true, 且保证训推一致
而关于 FAST token 也用了 mean_std 归一化的问题, 估计预训练也是用这个归一化方法, 所以不管也行.

`tokenize_state=true` 时 foresight token 漂移的问题已通过"强行让 tokenize_state=true 也加入连续 state token"解决

home的位置比采数时低啊


TODO
1. **FAST 的输入归一化需要单独核对。** 当前 `InternVLA-A1.5` transform 链默认用 `mean_std` 归一化，而官方 FAST 以每个动作维度的 `q01/q99` 映射到 `[-1,1]`。如果同一个归一化后的 action 同时喂给 flow matching 和 FAST，两个分支可能处于不一致的数值空间。

腕部相机:RealSense D435I SN: 420122070525
三方相机:RealSense D435I SN: 250222073513

6R6P1fay