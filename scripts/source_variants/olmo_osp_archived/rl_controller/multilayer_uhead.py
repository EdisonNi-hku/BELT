from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
import torch.utils.checkpoint
from torch import nn


_ARCHITECTURES = ("scalar_mix", "compressed_concat", "dynamic_attention", "naive_concat")
_VERSION = 1
_INPUT_NORMALIZATION = "per_layer_rms_fp32_nonaffine"


@dataclass(frozen=True)
class MultilayerUHeadConfig:
    architecture: str
    layers: tuple[int, ...] = (16, 20, 24, 28)
    hidden_size: int = 2048
    head_dim: int = 768
    n_layers: int = 2
    n_heads: int = 8
    dropout: float = 0.1
    norm_eps: float = 1e-6

    def __post_init__(self):
        if self.architecture not in _ARCHITECTURES:
            raise ValueError(f"architecture must be one of {_ARCHITECTURES}")
        if not isinstance(self.layers, (tuple, list)) or not self.layers:
            raise ValueError("layers must be a non-empty ordered sequence")
        if any(type(layer) is not int or layer <= 0 for layer in self.layers):
            raise ValueError("layers must contain positive integer indices")
        if any(left >= right for left, right in zip(self.layers, self.layers[1:])):
            raise ValueError("layers must be distinct and strictly increasing")
        object.__setattr__(self, "layers", tuple(self.layers))
        for field in ("hidden_size", "head_dim", "n_layers", "n_heads"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.head_dim % self.n_heads:
            raise ValueError("head_dim must be divisible by n_heads")
        if self.architecture == "compressed_concat" and 2 * self.head_dim % len(self.layers):
            raise ValueError("compressed_concat requires 2 * head_dim divisible by the layer count")
        for field in ("dropout", "norm_eps"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["layers"] = list(self.layers)
        return {"version": _VERSION, "input_normalization": _INPUT_NORMALIZATION, **result}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MultilayerUHeadConfig":
        fields = dict(value)
        if fields.pop("version", None) != _VERSION:
            raise ValueError("Unsupported multilayer UHead configuration version")
        if fields.pop("input_normalization", None) != _INPUT_NORMALIZATION:
            raise ValueError("Incompatible multilayer UHead input normalization")
        return cls(**fields)


class _MultilayerProjection(nn.Module):

    def __init__(self, config: MultilayerUHeadConfig):
        super().__init__()
        self.config = config
        k, d, m = len(config.layers), config.hidden_size, config.head_dim
        tail = [
            nn.LayerNorm(2 * m), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(2 * m, m), nn.LayerNorm(m), nn.GELU(),
        ]
        if config.architecture == "compressed_concat":
            self.layer_projections = nn.ModuleList(nn.Linear(d, 2 * m // k) for _ in range(k))
            self.projection = nn.Sequential(*tail)
        else:
            input_dim = k * d if config.architecture == "naive_concat" else d
            self.projection = nn.Sequential(nn.Linear(input_dim, 2 * m), *tail)
            if config.architecture == "scalar_mix":
                self.layer_logits = nn.Parameter(torch.zeros(k))
            elif config.architecture == "dynamic_attention":
                self.query = nn.Parameter(torch.zeros(d))
                self.layer_biases = nn.Parameter(torch.zeros(k))

    def _normalized_layers(self, features: torch.Tensor) -> torch.Tensor:
        k, d = len(self.config.layers), self.config.hidden_size
        if not isinstance(features, torch.Tensor) or features.ndim != 3 or features.shape[-1] != k * d:
            raise ValueError(f"features must have shape [B, T, {k * d}]")
        if not features.is_floating_point():
            raise ValueError("features must have a floating-point dtype")
        depth = features.reshape(*features.shape[:2], k, d).float()
        return depth * torch.rsqrt(depth.square().mean(dim=-1, keepdim=True) + self.config.norm_eps)

    def _layer_weights(self, normalized: torch.Tensor) -> torch.Tensor:
        if self.config.architecture == "scalar_mix":
            return self.layer_logits.float().softmax(-1).expand(*normalized.shape[:-1])
        if self.config.architecture == "dynamic_attention":
            scores = (normalized * self.query.float()).sum(-1) / math.sqrt(self.config.hidden_size)
            return (scores + self.layer_biases.float()).softmax(-1)
        raise ValueError(f"{self.config.architecture} does not define mixture weights")

    def layer_weights(self, features: torch.Tensor) -> torch.Tensor:
        return self._layer_weights(self._normalized_layers(features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self._normalized_layers(features)
        if self.config.architecture == "naive_concat":
            return self.projection(normalized.flatten(-2).to(dtype=self.projection[0].weight.dtype))
        if self.config.architecture == "compressed_concat":
            compressed = [
                layer(normalized[..., i, :].to(dtype=layer.weight.dtype))
                for i, layer in enumerate(self.layer_projections)
            ]
            return self.projection(torch.cat(compressed, dim=-1))
        mixed = (normalized * self._layer_weights(normalized).unsqueeze(-1)).sum(-2)
        return self.projection(mixed.to(dtype=self.projection[0].weight.dtype))


class MultilayerUHead(nn.Module):

    def __init__(self, config: MultilayerUHeadConfig):
        super().__init__()
        if not isinstance(config, MultilayerUHeadConfig):
            raise TypeError("config must be a MultilayerUHeadConfig")
        self.config = config
        self.head_dim = config.head_dim
        self.mask_future_tokens = True
        self.checkpoint_layers = True
        self.proj = _MultilayerProjection(config)
        self.step_end_embedding = nn.Embedding(2, config.head_dim)

        def encoder_layer():
            return nn.TransformerEncoderLayer(
                d_model=config.head_dim, nhead=config.n_heads, dim_feedforward=2048,
                dropout=config.dropout, activation="gelu", batch_first=True, norm_first=False,
            )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer(), num_layers=config.n_layers, enable_nested_tensor=False,
        )
        for index in range(1, config.n_layers):
            self.transformer_encoder.layers[index] = encoder_layer()
        m = config.head_dim
        self.classifier = nn.Sequential(
            nn.Linear(m, m), nn.LayerNorm(m), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(m, 1),
        )

    def _can_stream_causal(self, seq_padding) -> bool:
        return False

    def _encode_causal_streaming(self, features):
        raise NotImplementedError("MultilayerUHead requires the encoder with an explicit causal mask")

    def _build_causal_step_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.ones(T, T, dtype=torch.bool, device=device).triu(diagonal=1)

    def _encode(self, seq, padding, seq_len):
        mask = self._build_causal_step_mask(seq_len, seq.device)
        if not self.checkpoint_layers or not torch.is_grad_enabled() or not seq.requires_grad:
            return self.transformer_encoder(
                seq, mask=mask, src_key_padding_mask=padding, is_causal=True)
        encoded = seq
        for layer in self.transformer_encoder.layers:
            encoded = torch.utils.checkpoint.checkpoint(
                layer, encoded, src_mask=mask, src_key_padding_mask=padding, is_causal=True,
                use_reentrant=False)
        return encoded

    @staticmethod
    def _extract_span_boundaries(entity_mask: torch.Tensor):
        if entity_mask.ndim != 2:
            raise ValueError("claims must have shape [C, T]")
        if not bool(((entity_mask == 0) | (entity_mask == 1)).all()):
            raise ValueError("claims must be binary 0/1 masks")
        mask = entity_mask.bool()
        counts = mask.sum(-1)
        valid = counts > 0
        if not bool(valid.any()):
            empty = torch.empty(0, dtype=torch.long, device=mask.device)
            return empty, empty, valid
        spans = mask[valid]
        starts = spans.int().argmax(-1)
        ends = mask.shape[1] - 1 - spans.flip(-1).int().argmax(-1)
        if not bool((counts[valid] == ends - starts + 1).all()):
            raise ValueError("Each non-empty claim must be one contiguous span")
        if not bool((ends[1:] > ends[:-1]).all()):
            raise ValueError("claim span end positions must be distinct and strictly increasing")
        return starts, ends, valid

    def forward_from_features(self, features, attention_mask, claims):
        projected = self.proj(features)
        batch_size, seq_len = projected.shape[:2]
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != features.shape[:2]:
            raise ValueError("attention_mask must have shape [B, T] matching features")
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("attention_mask must be binary")
        if bool(((attention_mask[:, :-1] == 0) & (attention_mask[:, 1:] == 1)).any()):
            raise ValueError("attention_mask must use right padding: no 0-to-1 transition is allowed")
        if not isinstance(claims, (list, tuple)) or len(claims) != batch_size:
            raise ValueError("claims must contain one [C, T] tensor per batch item")
        padding = attention_mask.to(device=features.device) == 0
        markers = torch.zeros(batch_size, seq_len, dtype=torch.long, device=features.device)
        boundaries = []
        for index, claim in enumerate(claims):
            if not isinstance(claim, torch.Tensor) or claim.ndim != 2 or claim.shape[1] != seq_len:
                raise ValueError(f"claims[{index}] must have shape [C, {seq_len}]")
            claim = claim.to(device=features.device)
            _, ends, valid = self._extract_span_boundaries(claim)
            if bool((claim.bool() & padding[index]).any()):
                raise ValueError("claim spans must not include padding tokens")
            boundaries.append((ends, valid))
            markers[index, ends] = 1

        active = [index for index, (ends, _) in enumerate(boundaries) if ends.numel()]
        if active:
            seq = projected[active] + self.step_end_embedding(markers[active])
            encoded = self._encode(seq, padding[active], seq_len)
            end_representations = torch.cat([
                encoded[row, boundaries[index][0]] for row, index in enumerate(active)
            ], dim=0)
        else:
            end_representations = projected.new_empty((0, self.head_dim))
        logits = self.classifier(end_representations)
        max_claims = max((claim.shape[0] for claim in claims), default=0)
        result = logits.new_full((batch_size, max_claims, 1), -100.0)
        offset = 0
        for index, (ends, valid) in enumerate(boundaries):
            count = ends.numel()
            result[index, :valid.numel()][valid] = logits[offset:offset + count]
            offset += count
        return result

    def forward(self, features, attention_mask, claims):
        return self.forward_from_features(features, attention_mask, claims)
