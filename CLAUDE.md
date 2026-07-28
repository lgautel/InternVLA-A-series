# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InternVLA-A1.5 is a vision-language-action (VLA) robot policy that unifies visual understanding, latent video foresight, and continuous action generation. It attaches a lightweight action expert to a Qwen3.5-2B VLM backbone, uses learnable foresight tokens supervised by a frozen WAN2.2-5B video generation model during training, and predicts actions via flow matching at inference.

Built on a modified LeRobot framework. The package installs as `internvla-a1-5` from `src/lerobot/`.

## Common Commands

### Installation
```bash
conda create -y -n internvla_a1_5 python=3.11 && conda activate internvla_a1_5
conda install -c conda-forge ffmpeg svt-av1 -y
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.2.0
pip install -e .
pip install flash-attn==2.8.3 flash-linear-attention==0.5.0 causal-conv1d==1.6.1 --no-build-isolation
```

After installing, you must patch HuggingFace Transformers with custom Qwen3.5 model code:
```bash
TRANSFORMERS_DIR=${CONDA_PREFIX}/lib/python3.11/site-packages/transformers/
cp -r src/lerobot/policies/pi0/transformers_replace/models ${TRANSFORMERS_DIR}
cp -r src/lerobot/policies/pi05/transformers_replace/models ${TRANSFORMERS_DIR}
cp -r src/lerobot/policies/internvla_a1_5/transformers_replace/models ${TRANSFORMERS_DIR}
```

### Training
Fine-tuning (single dataset, 2 GPUs default):
```bash
bash launch/internvla_a15_finetune.sh <dataset_repo_id> [abs|delta] [true|false]
# Example: bash launch/internvla_a15_finetune.sh lerobot/pusht abs false
```

Pre-training (auto-discovers datasets under `data/a1/`):
```bash
bash launch/internvla_a15_pretrain.sh
```

Training uses `accelerate launch` with the entry point `src/lerobot/scripts/lerobot_train.py`. Key env vars: `PROC_PER_NODE`, `NODE_COUNT`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `HF_HOME`, `WANDB_TOKEN`.

### Evaluation
Each benchmark has its own eval script under `evaluation/`:
```bash
bash evaluation/RoboTwin/eval.sh <checkpoint> [output_path] [task_config] [task_idx]
bash evaluation/LIBERO/run_eval_libero_server_client.sh
bash evaluation/LIBERO-plus/run_eval_libero_plus.sh
bash evaluation/DOMINO/eval.sh
```

### Open-loop Testing
```bash
python tests/openloop_internvla_a1_5.py --ckpt-path <path> --dataset-root <path> [--visualize-future]
```

### Linting
```bash
ruff check src/
ruff format --check src/
```
Ruff config: target Python 3.10, line-length 110, ignores E501/T201/T203/B008.

## Architecture

### Source Layout (`src/lerobot/`)

**Policies** (`policies/`): Each policy has `configuration_*.py` (dataclass config), `modeling_*.py` (model), and `transform_*.py` (data transforms). Available: `internvla_a1_5`, `pi0`, `pi0_fast`, `pi05`. Policy selection is via `policies/factory.py` using `cfg.policy.type`.

**InternVLA-A1.5 model** (`policies/internvla_a1_5/`):
- `modeling_internvla_a1_5.py` — Main policy: `InternVLAA15Policy` wraps Qwen3.5 VLM + action expert + WAN video branch. Forward produces `loss_action`, `loss_video`, `loss_vqa`, `loss_fast` components.
- `modeling_internvla_a1_5_optimized.py` — Optimized inference backend (action-only, no WAN loading). Enable with `config.inference_backend="optimized"` and `config.action_loss_only=True`.
- `wan_model.py` + `wan/` — WAN2.2 video generation model used for foresight supervision. Frozen DiT + VAE.
- `action_tokens.py` — FAST action tokenization for Qwen3.5.
- `transform_internvla_a1_5.py` — Chat processor, video frame extraction, FAST token transforms.
- `transformers_replace/` — Patched HuggingFace model files that must be copied into the installed transformers package.

**Configs** (`configs/`):
- `train.py` — `TrainPipelineConfig` dataclass; orchestrates dataset, policy, optimizer, scheduler, wandb configs. Uses `draccus` for CLI parsing and JSON serialization.
- `default.py` — `DatasetConfig`, `VQADatasetConfig`, `WandBConfig`, `EvalConfig` base classes. `DatasetConfig` uses `draccus.ChoiceRegistry` for subclass dispatch by `type` field.
- `policies.py` — `PreTrainedConfig` base with `register_subclass` pattern.

**Datasets** (`datasets/`):
- `factory.py` — `make_dataset()` builds single or multi-repo datasets with optional distributed loading (`dist_loading=True` shards repos across ranks). Supports weighted sampling via YAML config (`weight_rules_path`). Can mix robot data with VQA data via `MixedMultimodalDataset`.
- `lerobot_dataset.py` — Core LeRobot dataset (reads from HF or local `data/` symlink).
- `transformed_dataset.py` — Wraps base datasets with transform pipelines.
- `streaming_dataset.py` — Streaming variant for large datasets.

**Transforms** (`transforms/`):
- `core.py` — Transform registry with `DataTransformFn.register_subclass()`. Pipeline defined in config's `data_transforms.inputs` list. Key transforms: `DeltaActionTransformFn`, `NormalizeTransformFn`, `ResizeImagesWithPadFn`.

### Config / CLI System

All training config is driven by `draccus` dataclass parsing. Policy/dataset configs use `ChoiceRegistry` with `register_subclass("internvla_a1_5")` for type-dispatched instantiation. CLI args follow dotted paths: `--policy.type=internvla_a1_5 --dataset.action_mode=delta`.

### Data Flow

Datasets are expected in LeRobot format under `data/` (symlinked to `$HF_LEROBOT_HOME`). The pipeline: raw dataset → `TransformedLeRobotDataset` applies `data_transforms.inputs` chain → collated batches → `policy.forward(batch)` returns `(loss, output_dict)`.

### Key Training Flags (InternVLA-A1.5)

- `action_loss_only=True` — Skip WAN video branch (faster training/inference for action-only use).
- `video_loss_only=True` — Train only the video foresight branch.
- `enable_vqa_loss=True` — Include language token loss (VQA/FAST tokens).
- `knowledge_insulation=True` — Block action expert from attending to prefix context.
- `freeze_learnable_tokens=True` — Freeze foresight tokens (typical for fine-tuning).
- `tokenize_state=True` — Encode robot state into prompt tokens.
- `use_fast_action_tokens=True` — Use FAST discretized action token supervision alongside flow matching.
- `action_mode=abs|delta` — Absolute vs delta action representation.

### Real-Robot Inference

For deployment, use the optimized backend:
```python
config.inference_backend, config.action_loss_only = "optimized", True
```
This skips WAN video loading and uses the low-latency action-only path.

### Git Submodules

- `third_party/RoboTwin` — RoboTwin simulation platform (used for evaluation).
