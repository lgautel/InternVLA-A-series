import logging
from dataclasses import dataclass, field, replace
from typing import ClassVar, Sequence

from lerobot.configs.default import DatasetConfig, VQADatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
    Extract3DKeypointTransformFn,
    ExtractVideoFramesTransformFn,
    FASTInternVLAA15ActionTokenizerTransformFn,
    InternVLAA15ChatProcessorTransformFn,
    InternVLAA15VQAProcessorTransformFn,
)
from lerobot.transforms.core import *
from lerobot.utils.constants import HF_HOME, OBS_IMAGES


@DatasetConfig.register_subclass("internvla_a1_5")
@dataclass
class InternVLAA15DatasetConfig(DatasetConfig):
    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32
    tokenize_state: bool = False
    max_prompt_length: int = 650
    mode: str = "train"
    chunk_size: int = 50
    use_fast_action_tokens: bool = True
    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224

    # GeoPredict 3D keypoint fusion (see v3.2 design doc §4). Disabled by default.
    enable_keypoint_predictor: bool = False
    num_keypoint_joints: int = 8
    keypoint_history_max_len: int = 1000
    kpt_4d_mode: str = "pos_only"
    keypoint_dim: int = 3  # auto-derived from kpt_4d_mode

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                DeltaActionTransformFn(),
                ResizeImagesWithPadFn(
                    height=InternVLAA15DatasetConfig.height,
                    width=InternVLAA15DatasetConfig.width,
                ),
                RemapImageKeyTransformFn(),
                ExtractVideoFramesTransformFn(),
                NormalizeTransformFn(),
                ComposeFieldsTransform(),
                FASTInternVLAA15ActionTokenizerTransformFn(),
                LoadActionTextFromJsonlTransformFn(),
                InternVLAA15ChatProcessorTransformFn(),
                PadStateAndActionTransformFn(
                    max_state_dim=InternVLAA15DatasetConfig.max_state_dim,
                    max_action_dim=InternVLAA15DatasetConfig.max_action_dim,
                ),
                ReorderStateActionTransform(),
                UnifyInternVLAA15InputsTransformFn(
                    num_video_frames=InternVLAA15DatasetConfig.num_video_frames,
                    video_height=InternVLAA15DatasetConfig.video_height,
                    video_width=InternVLAA15DatasetConfig.video_width,
                ),
            ],
            outputs=[],
        )
    )

    def __post_init__(self):
        super().__post_init__()
        _KPT_4D_DIM = {"pos_only": 3, "pos_rot": 7}
        if self.kpt_4d_mode not in _KPT_4D_DIM:
            raise ValueError(f"Unsupported kpt_4d_mode={self.kpt_4d_mode!r}")
        self.keypoint_dim = _KPT_4D_DIM[self.kpt_4d_mode]

        inputs = list(self.data_transforms.inputs)
        has_delta = any(isinstance(t, DeltaActionTransformFn) for t in inputs)
        if self.action_mode == "delta" and not has_delta:
            logging.info("action_mode='delta' -> Adding DeltaActionTransformFn")
            inputs = [DeltaActionTransformFn(), *inputs]
        elif self.action_mode == "abs" and has_delta:
            logging.info("action_mode='abs' -> Removing DeltaActionTransformFn")
            inputs = [t for t in inputs if not isinstance(t, DeltaActionTransformFn)]

        processor = InternVLAA15ChatProcessorTransformFn(
            tokenize_state=self.tokenize_state,
            max_state_dim=self.max_state_dim,
            max_length=self.max_prompt_length,
            use_fast_action_tokens=self.use_fast_action_tokens,
            mode=self.mode,
        )
        inputs = [t for t in inputs if not isinstance(t, InternVLAA15ChatProcessorTransformFn)]
        insert_idx = next(
            (i for i, t in enumerate(inputs) if isinstance(t, PadStateAndActionTransformFn)),
            len(inputs),
        )
        inputs.insert(insert_idx, processor)

        has_fast = any(isinstance(t, FASTInternVLAA15ActionTokenizerTransformFn) for t in inputs)
        if self.use_fast_action_tokens and not has_fast:
            inputs.insert(insert_idx, FASTInternVLAA15ActionTokenizerTransformFn())
        elif not self.use_fast_action_tokens and has_fast:
            inputs = [t for t in inputs if not isinstance(t, FASTInternVLAA15ActionTokenizerTransformFn)]

        for t in inputs:
            if isinstance(t, FASTInternVLAA15ActionTokenizerTransformFn):
                t.chunk_size = self.chunk_size
                break

        inputs = [t for t in inputs if not isinstance(t, Extract3DKeypointTransformFn)]
        if self.enable_keypoint_predictor:
            kpt_extract = Extract3DKeypointTransformFn(
                num_joints=self.num_keypoint_joints,
                history_max_len=self.keypoint_history_max_len,
                chunk_size=self.chunk_size,
                keypoint_dim=self.keypoint_dim,
            )
            # Insert right after NormalizeTransformFn (and before ComposeFieldsTransform), per
            # v3.2 §4.4: the raw stacked `observation.keypoint_3d` delta window must still be
            # normalization-free (kpt values are not part of NormalizeTransformFn's state/action
            # normalization), so we simply place it right before ComposeFieldsTransform.
            insert_idx = next(
                (i for i, t in enumerate(inputs) if isinstance(t, ComposeFieldsTransform)),
                len(inputs),
            )
            inputs.insert(insert_idx, kpt_extract)

        for t in inputs:
            if isinstance(t, UnifyInternVLAA15InputsTransformFn):
                t.enable_keypoint_predictor = self.enable_keypoint_predictor
                t.num_keypoint_joints = self.num_keypoint_joints
                t.keypoint_history_max_len = self.keypoint_history_max_len
                t.chunk_size = self.chunk_size
                t.keypoint_dim = self.keypoint_dim
                break

        self.data_transforms = replace(self.data_transforms, inputs=inputs)


@DataTransformFn.register_subclass("unify_internvla_a1_5_inputs")
@dataclass
class UnifyInternVLAA15InputsTransformFn(DataTransformFn):
    """Unify robot samples and always include video_frames for WAN.

    Always outputs observation.video_frames so that robot and VQA samples
    have identical keys and can be collated in the same batch.
    """

    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224

    # GeoPredict 3D keypoint fusion (see v3.2 design doc §4.5). When enabled, `data` is expected
    # to already carry the 5 kpt fields produced upstream by Extract3DKeypointTransformFn
    # (zero-filled there when the dataset has no `observation.keypoint_3d` column, i.e. Phase 1).
    enable_keypoint_predictor: bool = False
    num_keypoint_joints: int = 8
    keypoint_history_max_len: int = 1000
    chunk_size: int = 50
    keypoint_dim: int = 3

    def __call__(self, data: DataDict) -> DataDict:
        from lerobot.utils.constants import OBS_STATE, ACTION, OBS_STR
        from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import LABEL_MODE_NONE
        import torch

        input_ids = data[f"{OBS_STR}.input_ids"]
        fast_token_mask = data.get(
            f"{OBS_STR}.fast_token_mask",
            torch.zeros_like(input_ids, dtype=torch.bool),
        )
        label_mode = data.get("label_mode", torch.tensor(LABEL_MODE_NONE, dtype=torch.long))

        video_key = "observation.video_frames"
        if video_key in data:
            video_frames = data[video_key]
        else:
            video_frames = torch.zeros(
                self.num_video_frames + 1, 3, self.video_height, self.video_width
            )

        result = {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": input_ids,
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
            f"{OBS_STR}.fast_token_mask": fast_token_mask,
            "vqa_type": data["vqa_type"],
            "VQA.labels": data["VQA.labels"],
            "label_mode": label_mode,
            video_key: video_frames,
        }
        if self.enable_keypoint_predictor:
            result.update(_kpt_fields_passthrough_or_zero(data, self.num_keypoint_joints, self.keypoint_history_max_len, self.chunk_size, self.keypoint_dim))
        return result


def _kpt_fields_passthrough_or_zero(
    data: DataDict, num_joints: int, history_max_len: int, chunk_size: int, keypoint_dim: int = 3
) -> DataDict:
    """Return the 5 GeoPredict kpt fields, passed through from `data` if present, otherwise
    zero-filled with `kpt_mask=False` (used for VQA samples, which never have 3D keypoints)."""
    import torch

    h, j, c, d = history_max_len, num_joints, chunk_size, keypoint_dim
    return {
        "observation.his_kpts": data.get("observation.his_kpts", torch.zeros(h, j, d)),
        "observation.his_len": data.get("observation.his_len", torch.tensor(0, dtype=torch.long)),
        "observation.kpt_t": data.get("observation.kpt_t", torch.zeros(j, d)),
        "observation.kpt_future": data.get("observation.kpt_future", torch.zeros(c, j, d)),
        "observation.kpt_mask": data.get("observation.kpt_mask", torch.tensor(False)),
    }


@DataTransformFn.register_subclass("unify_internvla_a1_5_vqa_inputs")
@dataclass
class UnifyInternVLAA15VQAInputsTransformFn(DataTransformFn):
    """VQA unify transform that includes a dummy video_frames tensor.

    Ensures VQA samples have the same keys as robot samples so they can
    be collated in the same batch.
    """

    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224

    # See UnifyInternVLAA15InputsTransformFn above. VQA samples never carry real 3D keypoints,
    # so when the keypoint predictor is enabled these are always zero-filled with kpt_mask=False.
    enable_keypoint_predictor: bool = False
    num_keypoint_joints: int = 8
    keypoint_history_max_len: int = 1000
    chunk_size: int = 50
    keypoint_dim: int = 3

    def __call__(self, data: DataDict) -> DataDict:
        from lerobot.utils.constants import OBS_STATE, ACTION, OBS_STR
        from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import LABEL_MODE_TEXT
        import torch

        input_ids = data[f"{OBS_STR}.input_ids"]
        fast_token_mask = data.get(
            f"{OBS_STR}.fast_token_mask",
            torch.zeros_like(input_ids, dtype=torch.bool),
        )
        label_mode = data.get("label_mode", torch.tensor(LABEL_MODE_TEXT, dtype=torch.long))

        video_frames = torch.zeros(
            self.num_video_frames + 1, 3, self.video_height, self.video_width
        )

        result = {
            OBS_STATE: data[OBS_STATE],
            ACTION: data[ACTION],
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"],
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"],
            f"{OBS_STR}.input_ids": input_ids,
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"],
            f"{OBS_STR}.fast_token_mask": fast_token_mask,
            "vqa_type": torch.tensor(1, dtype=torch.long),
            "VQA.labels": data["VQA.labels"],
            "label_mode": label_mode,
            "observation.video_frames": video_frames,
        }
        if self.enable_keypoint_predictor:
            result.update(
                _kpt_fields_passthrough_or_zero(
                    {}, self.num_keypoint_joints, self.keypoint_history_max_len, self.chunk_size, self.keypoint_dim
                )
            )
        return result


@VQADatasetConfig.register_subclass("internvla_a1_5")
@dataclass
class InternVLAA15VQADatasetConfig(VQADatasetConfig):
    """VQA dataset config with dummy video_frames for shared batching."""

    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224

    # GeoPredict 3D keypoint fusion: VQA samples never have real 3D keypoints, so these only
    # control the shape of the zero-filled placeholders (see UnifyInternVLAA15VQAInputsTransformFn).
    enable_keypoint_predictor: bool = False
    num_keypoint_joints: int = 8
    keypoint_history_max_len: int = 1000
    chunk_size: int = 50
    kpt_4d_mode: str = "pos_only"
    keypoint_dim: int = 3

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                # Note: If you resize the VQA images, the coordinates in the VQA Data should be scaled accordingly.
                # Please pre-process the VQA data offline based on its format.
                ResizeVQAImagesWithPadFn(
                    height=InternVLAA15VQADatasetConfig.height,
                    width=InternVLAA15VQADatasetConfig.width,
                ),
                PadStateAndActionTransformFn(
                    max_state_dim=InternVLAA15VQADatasetConfig.max_state_dim,
                    max_action_dim=InternVLAA15VQADatasetConfig.max_action_dim,
                ),
                InternVLAA15VQAProcessorTransformFn(),
                UnifyInternVLAA15VQAInputsTransformFn(
                    num_video_frames=InternVLAA15VQADatasetConfig.num_video_frames,
                    video_height=InternVLAA15VQADatasetConfig.video_height,
                    video_width=InternVLAA15VQADatasetConfig.video_width,
                ),
            ],
            outputs=[],
        )
    )

    def __post_init__(self):
        _KPT_4D_DIM = {"pos_only": 3, "pos_rot": 7}
        if self.kpt_4d_mode not in _KPT_4D_DIM:
            raise ValueError(f"Unsupported kpt_4d_mode={self.kpt_4d_mode!r}")
        self.keypoint_dim = _KPT_4D_DIM[self.kpt_4d_mode]

        inputs = list(self.data_transforms.inputs)
        processor_type = InternVLAA15VQAProcessorTransformFn
        inputs = [t for t in inputs if not isinstance(t, processor_type)]
        insert_idx = next(
            (
                i
                for i, t in enumerate(inputs)
                if isinstance(t, UnifyInternVLAA15VQAInputsTransformFn)
            ),
            len(inputs),
        )
        inputs.insert(insert_idx, InternVLAA15VQAProcessorTransformFn())

        for t in inputs:
            if isinstance(t, UnifyInternVLAA15VQAInputsTransformFn):
                t.enable_keypoint_predictor = self.enable_keypoint_predictor
                t.num_keypoint_joints = self.num_keypoint_joints
                t.keypoint_history_max_len = self.keypoint_history_max_len
                t.chunk_size = self.chunk_size
                t.keypoint_dim = self.keypoint_dim
                break

        self.data_transforms = replace(self.data_transforms, inputs=inputs)


@PreTrainedConfig.register_subclass("internvla_a1_5")
@dataclass
class InternVLAA15Config(PreTrainedConfig):
    # VLM model selection - supports Qwen3.5-2B/4B/8B
    vlm_model_name_or_path: str = "Qwen/Qwen3.5-2B"

    # Action expert customization
    action_expert_hidden_size: int | None = 1024
    action_expert_intermediate_size: int | None = 3072

    dtype: str = "bfloat16"

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    image_resolution: tuple[int, int] = (224, 224)
    empty_cameras: int = 0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Training settings
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None

    # Optimizer settings
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 48

    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    # VQA configurations
    enable_vqa_loss: bool = True
    lambda_vqa: float = 1.0
    tokenize_state: bool = True

    # FAST action tokens
    action_token_min: int = 248077
    action_token_max: int = 250124

    # Knowledge insulation
    knowledge_insulation: bool = False
    block_action_attend_fast_tokens: bool = True

    inference_action_type: str = "fm"  # "fm" for flow matching, "fast" for fast-token supervision
    inference_backend: str = "standard"  # "standard" or "optimized"

    # Use SDPA (scaled_dot_product_attention) instead of eager attention in training
    use_sdpa: bool = False

    num_learnable_tokens: int = 50

    wan_checkpoint_path: str = f"{HF_HOME}/hub/Wan2.2-TI2V-5B"
    wan_config_path: str = f"{HF_HOME}/hub/Wan2.2-TI2V-5B"
    vae_path: str = f"{HF_HOME}/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    video_precision: str = "bfloat16"

    freeze_wan_dit: bool = True
    num_video_frames: int = 4
    video_height: int = 224
    video_width: int = 224
    # Chunk WAN VAE/DiT video supervision to fit 80 GiB GPUs without changing
    # the optimizer batch size or the number of training samples per step.
    video_micro_batch_size: int = 1
    video_loss_weight: float = 1.0
    video_loss_only: bool = False
    action_loss_only: bool = False
    freeze_learnable_tokens: bool = False

    # ------------------------------------------------------------------
    # GeoPredict 3D keypoint trajectory predictor fusion (three-path MoT).
    # See b/d/itrnVLA15_GeoP_3dtrj_3cn2.md (v3.1, general design) and
    # b/d/itrnVLA15_GeoP_3dtrj_3cn2_rbt2stak3.md (v3.2, RoboTwin aloha dual-arm adaptation).
    # ------------------------------------------------------------------
    enable_keypoint_predictor: bool = False
    num_keypoint_joints: int = 8  # 8 for RoboCasa single-arm (GeoPredict default); 14 for aloha dual-arm.

    # Loss weights. `action_loss_weight` replaces the previously hardcoded `10 *` multiplier.
    action_loss_weight: float = 10.0
    kpt_loss_weight: float = 1.0  # beta: overall keypoint loss weight (0.0 => Phase 1, indirect supervision only)
    kpt_future_loss_weight: float = 1.0  # gamma: weight of the future-trajectory term relative to the current-frame term

    # Keypoint expert architecture (mirrors ActionExpertConfig; head_dim/num_(kv_)heads inherited from the VLM).
    kpt_expert_hidden_size: int = 1024
    kpt_expert_intermediate_size: int = 3072

    # Knowledge insulation / gradient routing for the keypoint path.
    knowledge_insulation_kpt: bool = False  # detach VLM K/V when kpt expert queries attend to the prefix
    kpt_to_action_detach: bool = False  # detach kpt expert K/V when the action expert queries attend to it
    ki_gradient_scale: float = 0.0  # soft KI: action loss -> VLM gradient scale (0=hard KI, 1=no insulation)
    ki_kpt_gradient_scale: float = 0.0  # soft KI: kpt loss -> VLM gradient scale (0=hard KI, 1=no insulation)

    freeze_keypoint_modules: bool = False  # freeze TrackEncoder + kpt_expert + kpt_state_proj/keypoint_embedding/keypoint_out_proj

    # Per-module learning-rate scales (multiplied against the base optimizer LR).
    vlm_lr_scale: float = 1.0
    action_expert_lr_scale: float = 1.0
    kpt_expert_lr_scale: float = 1.0
    track_encoder_lr_scale: float = 1.0

    # Weight initialization.
    init_kpt_expert_from_action: bool = True  # Stage 3: warm-start kpt_expert from action_expert weights
    geopredict_checkpoint_path: str | None = None  # Stage 4: optional GeoPredict checkpoint for TrackEncoder init

    # TrackEncoder hyperparameters (see keypoints.py::TrackEncoder).
    keypoint_track_input_dim: int = 3
    keypoint_track_patch_size: int = 4
    keypoint_track_embed_dim: int = 256
    keypoint_track_query_dim: int = 512
    keypoint_track_num_heads: int = 8
    keypoint_track_ff_dim: int = 1024
    keypoint_history_max_len: int = 1000  # H: max number of history frames fed to TrackEncoder

    kpt_4d_mode: str = "pos_only"  # "pos_only" (3D) or "pos_rot" (7D)
    kpt_rot_loss_weight: float = 1.0  # rotation MSE weight (pos_rot only)
    keypoint_noise_sigma: float = 0.0  # optional additive Gaussian noise on kpt_t during training (0=disabled)

    _KPT_4D_DIM: ClassVar[dict[str, int]] = {"pos_only": 3, "pos_rot": 7}

    def __post_init__(self):
        super().__post_init__()
        if self.kpt_4d_mode not in self._KPT_4D_DIM:
            raise ValueError(f"Unsupported kpt_4d_mode={self.kpt_4d_mode!r}, expected {list(self._KPT_4D_DIM)}")
        self.keypoint_track_input_dim = self._KPT_4D_DIM[self.kpt_4d_mode]

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.lambda_vqa < 0:
            raise ValueError(f"lambda_vqa must be >= 0, got {self.lambda_vqa}")
        if self.action_token_min > self.action_token_max:
            raise ValueError(
                f"action_token_min ({self.action_token_min}) must be <= action_token_max ({self.action_token_max})"
            )
        if self.inference_backend not in {"standard", "optimized"}:
            raise ValueError(
                "inference_backend must be either 'standard' or 'optimized', "
                f"got {self.inference_backend!r}"
            )
        if self.inference_backend == "optimized" and not self.action_loss_only:
            raise ValueError("inference_backend='optimized' requires action_loss_only=True")
        if self.enable_keypoint_predictor and self.num_keypoint_joints <= 0:
            raise ValueError(f"num_keypoint_joints must be > 0, got {self.num_keypoint_joints}")
        if self.kpt_loss_weight < 0:
            raise ValueError(f"kpt_loss_weight must be >= 0, got {self.kpt_loss_weight}")
        if self.kpt_future_loss_weight < 0:
            raise ValueError(f"kpt_future_loss_weight must be >= 0, got {self.kpt_future_loss_weight}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features["action"] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def image_delta_indices(self) -> list | None:
        n = self.num_video_frames + 1
        return [self.chunk_size * i // (n - 1) for i in range(n)]

    @property
    def keypoint_3d_delta_indices(self) -> list[int] | None:
        """Delta-timestamp indices for `observation.keypoint_3d`, covering the full
        [-H, ..., -1, 0, 1, ..., C] window (H history frames strictly before the current frame,
        the current frame itself, and C future frames), or None when the keypoint predictor is
        disabled (no extra column is requested from the dataset).

        Note: v3.1 design doc §14.1 writes this as ``range(-H + 1, C + 1)`` with a comment
        claiming "H+1+C" indices, but that formula actually yields only H+C indices (off-by-one).
        We use ``range(-H, C + 1)`` here to get the intended H+1+C indices, consistent with
        GeoPredict's own history-buffer convention (history is `keypoints[:step]`, i.e. exactly
        H frames strictly before the current step, zero-padded — see
        GeoPredict/data_processing/robocasa_dataset.py lines 79-90) and with the §4.3/§15.3
        slicing convention (`[:H] -> his_kpts`, `[H] -> kpt_t`, `[H+1:H+1+C] -> kpt_future`).
        """
        if not self.enable_keypoint_predictor:
            return None
        h = self.keypoint_history_max_len
        c = self.chunk_size
        return list(range(-h, c + 1))  # H + 1 + C indices: [-H, ..., -1, 0, 1, ..., C]
