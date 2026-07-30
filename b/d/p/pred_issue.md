# [Reproduction] LIBERO-Plus Camera Viewpoints: 44.6% vs ~83% — requesting the exact fine-tuning recipe

Hi, thank you for the great work and for open-sourcing the code and weights. I have been trying to reproduce the LIBERO-Plus results (Table 6) by fine-tuning from the released base checkpoint, and would appreciate your help with a discrepancy I encountered.

## Setup

I followed `launch/internvla_a15_finetune_libero.sh` exactly as provided:

- **Checkpoint**: `InternRobotics/InternVLA-A1.5-base`
- **Dataset**: `nvidia/LIBERO_LeRobot_v3` (4 suites, feature names match `libero.yaml`)
- **Hyperparameters**: 100k steps, bs=16/GPU x 4 GPUs, lr=5e-5, warmup=2000, `action_loss_only=false`, `freeze_learnable_tokens=false` — all matching the launch script
- **Evaluation**: full Camera (1599 tasks) + Robot (1550 tasks) via `evaluation/LIBERO-plus/`
- **Hardware**: 4x H200

Training completed normally (final loss=0.265, loss_action=0.006, no anomalies).

## Results

| Category | My Result | Paper (Table 6) | Gap |
|---|---|---|---|
| Robot Initial States | 50.6% (785/1550) | ~55% | -4.4 pp |
| Camera Viewpoints | 44.6% (713/1599) | ~83% | **-38.4 pp** |

Robot is within a reasonable margin. Camera shows a significant gap.

## What I have ruled out

- **Perturbations are active**: baked into per-task BDDL files, confirmed by inspecting the loaded scenes.
- **Preprocessing matches**: both training and evaluation use `resize_with_pad` to 224x224.
- **No cliff-like failures**: success rates decrease smoothly with difficulty level (Camera: 62.3% at level 1 down to 33.7% at level 5), consistent with a capability gap rather than a bug.
- **Same pipeline works for Robot**: the Robot score is on target, so the evaluation infrastructure itself is sound.

## What may be missing from the released code

Since I have used every script, config, and publicly available dataset in this repository as-is, I suspect the gap comes from details not yet reflected in the released code:

- The launch script's glob pattern `*_no_noops*_lerobot` suggests a filtered dataset may have been used, but no such dataset or filtering script is publicly available.
- The actual training may have used a different GPU count / batch size / number of steps.
- There may be additional data augmentation not present in the current pipeline.

## Request

I understand the fine-tuned checkpoint is already released (thank you!). However, being able to reproduce training from the base checkpoint is important for anyone wanting to adapt the pipeline to new datasets or tasks. Would it be possible to share:

1. The exact fine-tuning script and configuration used for Table 6, if it differs from the released version.
2. The dataset preparation / filtering script, if a preprocessed version of LIBERO was used.
3. Or even a brief note on what differs from the released `internvla_a15_finetune_libero.sh`.

Any of these would be very helpful. I am happy to contribute a verified reproduction guide in return once the gap is resolved. Thank you for your time!

<details>
<summary>Notes for other reproducers (environment pitfalls)</summary>

- **`torchcodec` (CRITICAL)**: `pip install torchcodec` installs 0.15.0 (needs torch>=2.11). With torch 2.10.0, video decoding **silently fails to all-black frames**. Fix: `pip install "torchcodec==0.10.*" --index-url=https://download.pytorch.org/whl/cpu`
- **`flash-linear-attention` on Hopper**: Triton>=3.4 produces wrong results for Gated DeltaNet. Fix: `pip install tilelang`
- **LIBERO-Plus `assets.zip`**: contains nested paths from the author's machine; move extracted assets up to `libero/libero/assets/`
- **LIBERO-Plus `torch.load`**: PyTorch>=2.6 defaults to `weights_only=True`, breaking `*.pruned_init` files. Fix: pass `weights_only=False`

</details>
