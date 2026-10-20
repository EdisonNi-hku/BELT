from omegaconf import OmegaConf
from pathlib import Path
# Repository was removed in huggingface_hub 1.x. Importing it here broke the
# whole package on any environment with a modern hub -- including the one needed
# for transformers>=5.14, which is required for Qwen3.5. push_to_hub below uses
# HfApi.upload_folder instead, which is the supported replacement.
from huggingface_hub import HfApi, hf_hub_download
import tempfile
from abc import abstractmethod

import torch
import torch.nn as nn

from ..utils import load_feature_extractor

import logging

log = logging.getLogger()


class UncertaintyHeadBase(nn.Module):
    def __init__(
        self,
        feature_extractor,
        cfg=None,  # Temporary we save initializing cfg in the head itself
        model_type="token"
    ):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.cfg = cfg
        self.model_type = model_type

    @abstractmethod
    def _compute_tensors(self, llm_inputs, X, X_attn_mask):
        pass

    def _get_attn_mask(self, llm_inputs, llm_outputs):
        is_training = not hasattr(llm_outputs, "sequences")
        if is_training:
            return llm_inputs["attention_mask"][:, :-1]  # no new tokens introduced
        else:
            return llm_outputs["full_attention_mask"][:, 1:]  # all feature_calculators ignore first token

    def forward(self, llm_inputs, llm_outputs):
        features = self.feature_extractor(llm_inputs, llm_outputs)
        features_attn_mask = self._get_attn_mask(llm_inputs, llm_outputs)
        return self._compute_tensors(llm_inputs, features, features_attn_mask)

    @property
    def output_attentions(self):
        return self.feature_extractor.output_attention()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        base_model,
        revision: str = "main",
        use_auth_token: str = None,
    ):
        if Path(pretrained_path).exists():  # Local path
            config_path = Path(pretrained_path) / "config.yaml"
            weights_path = Path(pretrained_path) / "weights.pth"
        else:  # Remote Hugging Face repository
            config_path = hf_hub_download(
                repo_id=pretrained_path,
                filename="config.yaml",
                revision=revision,
                use_auth_token=use_auth_token,
            )
            weights_path = hf_hub_download(
                repo_id=pretrained_path,
                filename="weights.pth",
                revision=revision,
                use_auth_token=use_auth_token,
            )

        cfg = OmegaConf.load(config_path)
        weights = torch.load(weights_path, weights_only=True, map_location=torch.device('cpu'))
        feature_extractor = load_feature_extractor(cfg.feature_extractor, base_model)
        ue_head_cfg = cfg.uncertainty_head if cfg.uncertainty_head is not None else dict()
        uq_head = cls(feature_extractor, cfg=cfg, **ue_head_cfg)
        incompatible_keys = uq_head.load_state_dict(weights)
        assert (
            incompatible_keys.missing_keys == []
            and incompatible_keys.unexpected_keys == []
        ), f"LuqSequenceEstimator cannot be loaded. Missing keys: {incompatible_keys.missing_keys}; Unexpected keys: {incompatible_keys}."

        return uq_head

    @property
    def output_attentions(self):
        return self.feature_extractor.output_attention()

    def save(self, output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.state_dict(),
            Path(output_dir) / "weights.pth",
        )
        OmegaConf.save(self.cfg, Path(output_dir) / "config.yaml")

    def push_to_hub(self, repo_id: str, token: str = None):
        api = HfApi()
        api.create_repo(repo_id=repo_id, exist_ok=True, token=token)

        # Write the artefacts locally, then upload the folder in one commit.
        # This replaces the old Repository clone/add/commit/push flow, which
        # relied on an API huggingface_hub removed in 1.x.
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            self.save(output_dir)

            api.upload_folder(
                repo_id=repo_id,
                folder_path=str(output_dir),
                commit_message="Upload UncertaintyHead model and config",
                token=token,
            )

        log.info(f"Model pushed to Hugging Face Hub: https://huggingface.co/{repo_id}")
