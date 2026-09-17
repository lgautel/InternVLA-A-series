"""LIBERO2 InternVLA-A1.5 backend — fixes B2 (use_fast_action_tokens) + keypoint injection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.dataset_schemas import get_schema
from lerobot.policies.internvla_a1_5 import InternVLAA15Config, InternVLAA15Policy
from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
    InternVLAA15ChatProcessorTransformFn,
)
from lerobot.transforms.core import NormalizeTransformFn, ResizeImagesWithPadFn
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from evaluation.LIBERO.policy_server.backends.base_backend import BasePolicyBackend, PROTOCOL_VERSION
from evaluation.LIBERO.policy_server.backends.canonical_preprocess import build_base_sample
from evaluation.LIBERO.policy_server.backends.input_semantics import expected_num_input_images, required_image_keys_for_robot


class InternVLAA15Backend(BasePolicyBackend):
    """InternVLA-A1.5 backend with B2 fix and keypoint support."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        stats_key: str | None = None,
        robot_type: str | None = None,
        resize_size: int = 224,
        vlm_model_path: str | None = None,
        no_state_prompt: bool = False,
        max_prompt_length: int = 650,
        wan_model_path: str | None = None,
        wan_vae_path: str | None = None,
        action_loss_only: bool = True,
        inference_backend: str = "standard",
    ) -> None:
        super().__init__(ckpt_path=ckpt_path, device=device, stats_key=stats_key, robot_type=robot_type)

        config = PreTrainedConfig.from_pretrained(self.ckpt_path)
        if not isinstance(config, InternVLAA15Config):
            raise TypeError(f"Expected InternVLAA15Config, got {type(config)}")

        if vlm_model_path:
            config.vlm_model_name_or_path = vlm_model_path
        if not config.vlm_model_name_or_path:
            raise ValueError(
                "InternVLA-A1.5 checkpoint has an empty vlm_model_name_or_path. "
                "Pass --vlm_model_path <hf-id-or-local-dir> to override."
            )

        if wan_model_path:
            config.wan_checkpoint_path = wan_model_path
            config.wan_config_path = wan_model_path
            if not wan_vae_path:
                default_vae = Path(wan_model_path) / "Wan2.2_VAE.pth"
                if default_vae.exists():
                    config.vae_path = str(default_vae)
        if wan_vae_path:
            config.vae_path = wan_vae_path

        config.action_loss_only = bool(action_loss_only)
        config.inference_backend = inference_backend
        if config.inference_backend == "optimized" and not config.action_loss_only:
            raise ValueError("inference_backend='optimized' requires action_loss_only=True")

        if not config.action_loss_only:
            wan_config_json = Path(config.wan_config_path) / "config.json"
            if not wan_config_json.exists():
                raise FileNotFoundError(
                    f"WAN config.json not found at {wan_config_json}. "
                    "Pass --wan_model_path <local-wan-dir> or run with action_loss_only=True."
                )
            if not Path(config.vae_path).exists():
                raise FileNotFoundError(
                    f"WAN VAE weights not found at {config.vae_path}. "
                    "Pass --wan_vae_path <vae.pth> to override."
                )

        self.policy = InternVLAA15Policy.from_pretrained(
            config=config, pretrained_name_or_path=self.ckpt_path
        )
        self.policy.to(self.device)
        self.policy.eval()

        if config.dtype == "bfloat16":
            self.compute_dtype = torch.bfloat16
        elif config.dtype == "float32":
            self.compute_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported config.dtype={config.dtype!r}")

        self.state_dim = config.input_features[OBS_STATE].shape[0]
        self.chunk_size = int(config.chunk_size)
        self.resize_size = int(resize_size)

        self.stats_key, self.robot_type, self.state_stat, self.action_stat = self._load_stats()
        self.state_input_dim = int(
            np.asarray(self.state_stat[OBS_STATE]["mean"], dtype=np.float32).reshape(-1).shape[0]
        )
        self.expected_num_input_images = expected_num_input_images(self.robot_type)
        self.state_normalizer = NormalizeTransformFn(selected_keys=[OBS_STATE], norm_stats=self.state_stat)
        self.resize = ResizeImagesWithPadFn(
            height=self.resize_size,
            width=self.resize_size,
            mapping={f"{OBS_IMAGES}.image{i}": f"{OBS_IMAGES}.image{i}" for i in range(3)},
        )

        action_mode = "joint"
        if self.robot_type:
            try:
                schema = get_schema(self.robot_type)
                action_mode = getattr(schema, "action_mode", "joint")
            except ValueError:
                pass

        processor_path = vlm_model_path or config.vlm_model_name_or_path
        processor_tokenize_state = False if no_state_prompt else config.tokenize_state

        # B2 FIX: use_fast_action_tokens from config (not hardcoded False)
        use_fast = bool(getattr(config, "use_fast_action_tokens", True))
        self.use_fast_action_tokens = use_fast

        self.processor = InternVLAA15ChatProcessorTransformFn(
            pretrained_model_name_or_path=processor_path,
            max_length=int(max_prompt_length),
            tokenize_state=processor_tokenize_state,
            max_state_dim=config.max_state_dim,
            use_fast_action_tokens=use_fast,
            mode="eval",
            action_mode=action_mode,
        )
        self.no_state_prompt = bool(no_state_prompt)
        self.action_mode = action_mode

    def _prepare_single(self, example: dict[str, Any]) -> dict[str, Any]:
        sample = build_base_sample(
            example,
            robot_type=self.robot_type,
            expected_state_dim=self.state_input_dim,
            resize_transform=self.resize,
            mask_as_tensor=False,
        )
        sample = self.state_normalizer(sample)
        sample = self.processor(sample)

        kpt_history = example.get("kpt_history")
        his_len = example.get("his_len", 0)
        if kpt_history is not None and his_len > 0:
            kh = np.asarray(kpt_history, dtype=np.float32)
            sample["observation.his_kpts"] = torch.from_numpy(kh)
            sample["observation.his_len"] = torch.tensor(int(his_len), dtype=torch.long)

        return sample

    def _sample_to_inputs(self, sample: dict[str, Any]) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        for key, value in sample.items():
            if key == "task":
                inputs[key] = [value]
                continue
            if isinstance(value, bool):
                continue
            if not isinstance(value, torch.Tensor):
                continue
            inputs[key] = value.unsqueeze(0).to(self.device)
        return inputs

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        examples = payload.get("examples")
        if not isinstance(examples, list) or len(examples) == 0:
            raise ValueError("payload.examples must be a non-empty list")

        autocast_device = self.device.type if self.device.type != "cpu" else "cpu"
        autocast_enabled = self.compute_dtype != torch.float32

        outputs = []
        with torch.no_grad(), torch.amp.autocast(
            device_type=autocast_device,
            dtype=self.compute_dtype,
            enabled=autocast_enabled,
        ):
            for example in examples:
                sample = self._prepare_single(example)
                inputs = self._sample_to_inputs(sample)
                chunk = self.policy.predict_action_chunk(inputs)
                outputs.append(chunk.detach().float().cpu().numpy())

        normalized_actions = np.concatenate(outputs, axis=0)
        actions = self.denormalize_actions(normalized_actions)
        actions = self.postprocess_actions(actions)
        return {
            "actions": actions,
            "action_space": self.action_space(),
            "action_dim": int(actions.shape[-1]),
            "chunk_size": self.chunk_size,
            "postprocess": {
                "clip": True,
                "gripper_binarize": False,
            },
        }

    def metadata(self) -> dict[str, Any]:
        action_space = self.action_space()
        return {
            "policy_type": "internvla_a1_5",
            "ckpt_path": str(self.ckpt_path),
            "stats_key": self.stats_key,
            "robot_type": self.robot_type,
            "action_mode": self.action_mode,
            "chunk_size": self.chunk_size,
            "action_dim": int(action_space["dim"]),
            "action_denorm_mode": self.action_denorm_mode,
            "protocol_version": PROTOCOL_VERSION,
            "returns": "actions",
            "expected_num_input_images": self.expected_num_input_images,
            "image_mask_policy": "training_semantics_strict",
            "strict_input_validation": True,
            "preprocessing_owner": "server_canonical",
            "deterministic_inference_preprocess": True,
            "required_image_keys": required_image_keys_for_robot(self.robot_type),
            "expected_state_dim": self.state_input_dim,
            "resize_size": int(self.resize_size),
            "use_fast_action_tokens": bool(self.use_fast_action_tokens),
        }
