# Libero-Plus OpVLA SFT 实施操作日志

> **方案**: [`sft.md`](sft.md)  
> **EXPR_NAME**: `4dwvlaOpvlaLibplusKpt0911`  
> **日期**: 2026-09-11  
> **环境**: `/B/VENV/itnvla15rbt20/`，`HF_HOME=/B/VENV/hf_home`

---

## 0. 任务目标

按 `sft.md` 在 `itvlaGpLibPlus` 内实现 `launch/libplus_sft_launch.sh`，完成前置检查、单元测试、WAN/SMOKE 测试，启动 50 epoch（53450 steps）正式训练，fix 所有 error 直至训练成功。

---

## 1. 清空 GPU（2026-09-11 启动前）

**操作原因**: 释放 GPU 供本实验使用。

**命令**:
```bash
source /B/VENV/itnvla15rbt20/bin/activate
pkill -9 -f "lerobot_train"
pkill -9 -f "accelerate.commands.launch"
pkill -9 -f "bigmatrix_multiply"
```

**清空前**: 8 个 python 进程各占 ~98 GiB（疑似上一轮 lbrp 训练残留）。

**结果**: （见下方命令输出）

---
