from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
import transformers


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def decoder_body(model):
    body = getattr(model, "model", model)
    if not hasattr(body, "layers"):
        raise ValueError("multilayer extraction requires a decoder with .layers")
    return body


class _Captured(Exception):
    pass


class MultilayerHiddenStateProvider:
    def __init__(self, model, *, layers, model_identity, chunk_size=1024, cache_dir=None):
        self.model = model.eval().requires_grad_(False)
        self.body = decoder_body(model)
        self.selected_layers = tuple(layers)
        if (not self.selected_layers or any(type(k) is not int for k in self.selected_layers)
                or tuple(sorted(set(self.selected_layers))) != self.selected_layers
                or not 1 <= self.selected_layers[0] <= self.selected_layers[-1] <= len(self.body.layers)):
            raise ValueError("layers must be distinct increasing one-based decoder block indices")
        if type(chunk_size) is not int or chunk_size < 0:
            raise ValueError("chunk_size must be a nonnegative integer")
        if not model_identity:
            raise ValueError("model_identity is required to isolate cached features")
        self.chunk_size = chunk_size
        self.hidden_size = int(model.config.hidden_size)
        self.device = next(model.parameters()).device
        self.contract = {"schema": "multilayer-raw-features/v1", "model": dict(model_identity),
                         "layers": list(self.selected_layers), "hidden_size": self.hidden_size,
                         "dtype": str(next(model.parameters()).dtype),
                         "layer_index_convention": "one_based_block_output",
                         "final_norm_applied": False, "drop_last_position": True,
                         "chunk_size": chunk_size, "torch_version": str(torch.__version__),
                         "transformers_version": str(transformers.__version__),
                         "attention_implementation": getattr(model.config, "_attn_implementation", None),
                         "device_type": self.device.type,
                         "device_name": (torch.cuda.get_device_name(self.device)
                                         if self.device.type == "cuda" else "cpu")}
        self.cache_root = Path(cache_dir) / _digest(self.contract) if cache_dir else None
        if self.cache_root is not None:
            self.cache_root.mkdir(parents=True, exist_ok=True)
        self.stats = {"cache_hits": 0, "cache_misses": 0, "backbone_calls": 0,
                      "backbone_tokens": 0, "cache_bytes_written": 0}

    def _capture(self, ids, attention_mask, cache=None, start=0):
        captured, handles = {}, []

        def hook(layer):
            def capture(_module, _inputs, output):
                value = output[0] if isinstance(output, (tuple, list)) else output
                captured[layer] = value.detach()
                if layer == self.selected_layers[-1]:
                    raise _Captured()
            return capture

        for layer in self.selected_layers:
            handles.append(self.body.layers[layer - 1].register_forward_hook(hook(layer)))
        extra = {}
        if cache is not None:
            extra = {"past_key_values": cache,
                     "cache_position": torch.arange(start, start + ids.shape[1], device=ids.device),
                     "position_ids": torch.arange(start, start + ids.shape[1], device=ids.device)[None]}
        try:
            with torch.no_grad():
                try:
                    self.body(input_ids=ids, attention_mask=attention_mask,
                              use_cache=cache is not None, **extra)
                except _Captured:
                    pass
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(self.selected_layers):
            raise RuntimeError("backbone did not produce all requested block outputs")
        self.stats["backbone_calls"] += 1
        self.stats["backbone_tokens"] += ids.numel()
        return torch.cat([captured[k] for k in self.selected_layers], -1)

    def _extract(self, token_ids):
        ids = torch.tensor([token_ids], device=self.device, dtype=torch.long)
        width = self.chunk_size or len(token_ids)
        cache = None
        if len(token_ids) > width:
            from transformers import DynamicCache
            try:
                cache = DynamicCache(config=self.model.config)
            except TypeError:
                cache = DynamicCache()
        chunks = []
        for start in range(0, len(token_ids), width):
            end = min(start + width, len(token_ids))
            mask = torch.ones(1, end, dtype=torch.long, device=self.device)
            values = self._capture(ids[:, start:end], mask, cache=cache, start=start)
            chunks.append(values[0].cpu())
        return torch.cat(chunks, 0)[:-1].contiguous()

    def _row(self, token_ids):
        if len(token_ids) < 2 or any(type(t) is not int or t < 0 for t in token_ids):
            raise ValueError("features require at least two nonnegative integer token ids")
        key = _digest(token_ids)
        path = self.cache_root / f"{key}.pt" if self.cache_root else None
        if path is not None and path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("contract") != self.contract or payload.get("token_digest") != key:
                raise ValueError(f"feature cache identity mismatch: {path}")
            row = payload.get("features")
            self.stats["cache_hits"] += 1
        else:
            self.stats["cache_misses"] += 1
            row = self._extract(token_ids)
            if path is not None:
                fd, temp = tempfile.mkstemp(dir=path.parent, prefix=key, suffix=".partial")
                os.close(fd)
                try:
                    torch.save({"contract": self.contract, "token_digest": key, "features": row}, temp)
                    self.stats["cache_bytes_written"] += os.path.getsize(temp)
                    os.replace(temp, path)
                finally:
                    if os.path.exists(temp):
                        os.unlink(temp)
        expected = (len(token_ids) - 1, self.hidden_size * len(self.selected_layers))
        if not isinstance(row, torch.Tensor) or tuple(row.shape) != expected or not row.is_floating_point():
            raise ValueError(f"invalid cached feature shape/type: expected {expected}")
        if not torch.isfinite(row).all():
            raise ValueError("non-finite backbone features")
        return row.detach()

    def get_features(self, sequences, device):
        rows = [self._row(list(seq)) for seq in sequences]
        length = max((len(row) for row in rows), default=0)
        features = torch.zeros(len(rows), length, self.hidden_size * len(self.selected_layers),
                               device=device, dtype=torch.float32)
        mask = torch.zeros(len(rows), length, device=device, dtype=torch.long)
        for i, row in enumerate(rows):
            features[i, :len(row)] = row.to(device=device, dtype=torch.float32)
            mask[i, :len(row)] = 1
        return features, mask, list(range(len(rows))), []
