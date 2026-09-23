use_fast_action_tokens=true
旧版 `evaluation/LIBERO` backend 曾硬编码 use_fast_action_tokens=false, 所以训练时4DWVLA是不是也不要这个,因为`itvla_trn_strategy.md`认为 fast token 的归一化需要的不是现在mean_std. 但要注意设为 false 后的训推一致性.
但itvla建议后训练use_fast_action_tokens=true, 而且当data schema为panda时, ReorderStateActionTransform是空操作, 所以 Fast Token 与 连续型 state emb 一个在 reorder 前, 一个在 reorder 后的影响不大, 所以, use_fast_action_tokens=true, 且保证训推一致
而关于 FAST token 也用了 mean_std 归一化的问题, 估计预训练也是用这个归一化方法, 所以不管也行.

`tokenize_state=true` 时 foresight token 漂移的问题已通过"强行让 tokenize_state=true 也加入连续 state token"解决
    TODO 但这会不会让state的影响变得更大, 因为采数的问题, state和action的相关性本来就很大了
home的位置比采数时低啊


# b/d/Frk2/ds2_pblm_solv1.md

- "**P3** | 数据集里根本没有夹爪实测量 | 高 | `state[7]` 就是命令本身；69.40% 的相邻帧逐位不变" 
    + 对数据的实际观察得到在发出闭合指令时action=1, state=1 会持续很多帧, 估计需要时间闭合. 类似 sumry0919_4trn.markdown 提到的 `D1 夹爪动作通道没有信息量 **最高优先级**`
    + drop state [训]
    + 不同模态的门控和权重 [训]
- "**P4** | 手臂动作是带伺服滞后的指令，分布探出状态分布. **但也可能是实验环境被移动造成**" 但也有可能是采数时用了阻抗,eval时没有用
    + 检查一下eval时有没有用阻抗, 没有的话记得加上. eval 时的阻抗要和数采时对齐. 好像yuhan那个版本不对齐也没事. [推]
- "**P5** | `state[8:15]` 是关节的 FK，7/15 维零独立信息，且放大手臂捷径" 
    + 考虑关于EE 的state[8:15] 的 drop out 率比其它state更高 [训]
    + "把tokenize_state时依然加入连续值"state的版本改成"tokenize_state但不加连续值"的intervla原来想做的那个版本. [训]
|- "**`state[11:15]`** | **4** | **末端执行器的姿态四元数**，实际存储顺序为 `[qx, qy, qz, qw]`（`info.json` 误标为 `wxyz`，见下） "
    + 在章节"### 2.3 根因：又一次相信了 `info.json` 里的名字"及其子章节有详细说明. 要修正 info.json [数]
- "**P6** | 时间基准是零抖动的完美 15 Hz 网格"
    + 涉及 his_len 和   TODO
- 执行"#### 步骤 1（P0，零成本）：固化夹爪极性契约" [数][推]


# sumry0919_4trn.markdown

<4DWVLA>/b/d/Frk2/realwrld_debug/sumry0919_4trn.markdown



# 洞察与措施

# action和state关联性高,特别griper,开/闭命令发出时好几帧都是图像在变而act和sta不变


# 数据增强
- 已开intrnvla的, 但要验证有没有开错  TODO
- 比较openpi和lingbot-vla的数据增强,可用它们的那版  TODO

# 推理部署
- 改 hand_off 的问题
- 真机频率低的问题可通过用带阻抗`JointImpedanceTracker`的`rlinf/envs/realworld/franka/franky_controller.py`来解决. 然后把频率调高.  TODO 改成阻抗就可以调高频率了吗?就可以打到训练集那种"即使发了action命令, 但state未必能打到action所指定的那个地方" 的效果么
- TODO 各关节限制放宽
- 可能过拟合, 换个checkpoint

# TODO
- his_len(每5step )
- 六、his_len 历史里没有夹爪历史
- 为什么有ds2_pblm_solv1.md所说的"真机实测频率 2.5–3.6 Hz", Franka这么慢么? 怎么和数采频率相差那么大
- **FAST 的输入归一化需要单独核对。** 当前 `InternVLA-A1.5` transform 链默认用 `mean_std` 归一化，而官方 FAST 以每个动作维度的 `q01/q99` 映射到 `[-1,1]`。如果同一个归一化后的 action 同时喂给 flow matching 和 FAST，两个分支可能处于不一致的数值空间。
- 参考openpi jax 的"训练时维护一份参数的指数滑动平均（EMA / Polyak averaging）"
- 天机训的chunksz=10但在实际以10tep推理时依然弯不下腰, chunksz=50却可用, 是电机执行频率高的缘故吗
- yuhan认为第三视角没什么用,更增强了我对drop out的想法
- 部署时并没与采数时的阻抗及其超参进行对齐

腕部相机:RealSense D435I SN: 420122070525
三方相机:RealSense D435I SN: 250222073513

6R6P1fay

## Debug

真机上遇到的问题
    数据的0是开, 1是闭合, eval是把它弄反了, 所以eval的设置会有问题
trn_strategy 发现的代码问题, markdown上已经


修改一份真机版本重跑

然后是libero-plus的,结合zwy的实验(注意该实验尚未没验证过数据增强是否正确,没强调训推一致性,推理时的代码也没思考过)





