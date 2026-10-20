"""Extract frozen-model hidden states with transformers instead of vLLM.

Why this exists
---------------
``speculators``' ``VllmHiddenStatesGenerator`` calls ``unify_hybrid_kv_cache_specs``
and then ``_get_kv_cache_groups_uniform_spec``, which require every layer's KV
cache spec to collapse to one type. That holds for attention-only hybrids (its
own comment cites GPT-OSS sliding/full attention), but Qwen3.5 interleaves Gated
DeltaNet state with attention KV -- genuinely different spec types -- so it dies
with "Hybrid KV cache manager is disabled but failed to convert the KV cache
specs to one unified type" (observed in job 2911366). There is no configuration
knob; the call is unconditional.

The trainer only ever needs hidden states from a frozen model, which transformers
can produce directly. This provider is a drop-in for ``VLLMHiddenStateProvider``:
same ``get_features`` contract, no vLLM and no speculators involved.

Layer indexing
--------------
Layer numbers are used exactly as ``FeatureExtractorBasicHiddenStates`` uses
them: direct indices into the transformers ``hidden_states`` tuple, negatives
included (``hidden_states[-1]`` is the final layer). ``get_layer_nums`` does not
resolve negatives -- it only expands the literal ``'all'`` -- so no conversion
happens anywhere in that path.

This is deliberately *not* what the vLLM provider does. There, layer ids are
converted with ``num_hidden_layers + x`` because vLLM indexes decoder layers,
a different space that excludes the embedding output. Applying that conversion
here would agree by coincidence for ``-1`` and silently read the wrong layer for
any positive layer number, which is why the mapping is asserted in tests.

Long sequences
--------------
A single forward over a long trace materialises every layer's hidden states at
once and runs out of memory well before the generation cap. Scoring a best-of-N
pool that way silently kept candidates under ~23k tokens and dropped everything
from ~31k up, which is not random attrition -- long candidates have the most
steps, so they are exactly where step-level scoring matters.

Sequences longer than ``hs_chunk_size`` are therefore run in chunks that carry
the KV cache forward. For a causal model this is exact rather than an
approximation: every position still attends over the full prefix, it just
arrives through the cache instead of the current chunk. Only the requested
layers are kept, and each chunk moves to CPU before the next one runs.
"""

import logging

import torch

log = logging.getLogger(__name__)


class HFHiddenStateProvider:
    """Produce next-token-aligned features from a frozen causal LM.

    Mirrors VLLMHiddenStateProvider's contract exactly: ``get_features`` returns
    ``(feats [B, T-1, H_total], attn_mask [B, T-1], keep_indices, skipped_indices)``
    where ``H_total`` is the concatenation of the requested layers' hidden sizes.
    """

    def __init__(self, config, uhead, model=None, tokenizer=None, layer_nums=None):
        self.config = config
        self.uhead = uhead
        self.model_path = getattr(
            getattr(config, "vllm", None), "model_path", None
        ) or config.model.pretrained_model_name_or_path
        self.layer_nums = (
            self._validate_explicit_layer_nums(layer_nums)
            if layer_nums is not None
            else self._resolve_layer_nums(uhead)
        )
        self.device = torch.device(getattr(config, "hs_device", "cuda:0"))
        # 0 disables chunking and restores the plain single forward.
        self.chunk_size = int(getattr(config, "hs_chunk_size", 8192) or 0)
        self.model = model if model is not None else self._build_model()
        log.info("HF hidden-state layer indices (into hidden_states): %s", self.layer_nums)

    @staticmethod
    def _validate_explicit_layer_nums(layer_nums) -> list[int]:
        if not isinstance(layer_nums, (list, tuple)) or not layer_nums:
            raise ValueError("layer_nums must be a nonempty list or tuple")
        if any(type(layer_num) is not int for layer_num in layer_nums):
            raise ValueError("layer_nums must contain only integers")
        if len(set(layer_nums)) != len(layer_nums):
            raise ValueError("layer_nums must not contain duplicates")
        return list(layer_nums)

    def _resolve_layer_nums(self, uhead) -> list[int]:
        """Read the extractor's own layer numbers, verbatim.

        Attribute names are private on purpose in the extractors
        (``_feature_extractors`` / ``_layer_nums``); reading a non-underscored
        name silently yields nothing and the head would train on no features.
        """
        from luh.feature_extractors.basic_hidden_states import (
            FeatureExtractorBasicHiddenStates,
        )
        from luh.feature_extractors.combined import FeatureExtractorCombined

        extractor = getattr(uhead, "feature_extractor", None)
        if isinstance(extractor, FeatureExtractorCombined):
            parts = list(extractor._feature_extractors)
        else:
            parts = [extractor]

        layer_nums: list[int] = []
        unsupported = []
        for part in parts:
            if isinstance(part, FeatureExtractorBasicHiddenStates):
                layer_nums.extend(part._layer_nums)
            else:
                unsupported.append(type(part).__name__)

        if unsupported:
            raise ValueError(
                f"transformers hidden-state backend supports only hidden-state "
                f"feature extractors; got {unsupported}"
            )
        if not layer_nums:
            raise ValueError("Could not infer hidden-state layer numbers from uhead.feature_extractor")
        return layer_nums

    def _build_model(self):
        from transformers import AutoModelForCausalLM

        torch_dtype = getattr(self.config.model, "torch_dtype", "bfloat16")
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype.replace("torch.", ""))
        log.info("Loading frozen base model %s (%s)", self.model_path, torch_dtype)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch_dtype,
            trust_remote_code=True,
            revision=getattr(self.config.model, "revision", None),
            cache_dir=getattr(self.config, "hf_cache", None),
            token=getattr(self.config, "hf_token", None),
        )
        model.eval()
        # Frozen: only the uncertainty head trains. Also keeps autograd from
        # retaining activations for a 2B backbone we never update.
        for param in model.parameters():
            param.requires_grad = False
        return model.to(self.device)

    def _check_layer_range(self, n_hidden: int) -> None:
        for num in self.layer_nums:
            if not -n_hidden <= num < n_hidden:
                raise RuntimeError(
                    f"layer number {num} out of range for hidden_states of "
                    f"length {n_hidden}"
                )

    def _picked_layers(self, ids) -> list:
        """The requested layers' hidden states for one sequence, on CPU.

        Chunked when the sequence is long enough to be worth it. The chunks share
        one KV cache, so a position in chunk k still attends over chunks 0..k-1
        and the result is identical to a single forward, not an approximation of
        one.
        """
        total = ids.shape[1]
        if not self.chunk_size or total <= self.chunk_size:
            out = self.model(ids, output_hidden_states=True, use_cache=False)
            self._check_layer_range(len(out.hidden_states))
            return [out.hidden_states[num][0].float().cpu() for num in self.layer_nums]

        parts = [[] for _ in self.layer_nums]
        cache = None
        for start in range(0, total, self.chunk_size):
            chunk = ids[:, start:start + self.chunk_size]
            out = self.model(
                chunk,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
                cache_position=torch.arange(
                    start, start + chunk.shape[1], device=ids.device
                ),
            )
            if start == 0:
                self._check_layer_range(len(out.hidden_states))
            cache = out.past_key_values
            for slot, num in enumerate(self.layer_nums):
                parts[slot].append(out.hidden_states[num][0].float().cpu())
            del out
        del cache
        return [torch.cat(chunks, dim=0) for chunks in parts]

    @torch.no_grad()
    def get_features(self, input_ids_list: list[list[int]], device: torch.device):
        per_sample_feats = []
        lengths = []
        keep_indices = []
        skipped_indices = []
        feat_dim = None

        for i, seq_ids in enumerate(input_ids_list):
            expected = len(seq_ids)
            if expected < 2:
                # A single token yields no next-token position at all.
                skipped_indices.append(i)
                continue
            try:
                ids = torch.tensor([seq_ids], dtype=torch.long, device=self.device)
                # Indexed exactly as FeatureExtractorBasicHiddenStates does:
                # straight into hidden_states, negatives included.
                sample = torch.cat(self._picked_layers(ids), dim=-1)

                if sample.shape[0] > expected:
                    sample = sample[:expected]
                if sample.shape[0] < expected:
                    log.warning(
                        "Skipping sample %d: hidden states shorter than input (got=%d expected=%d)",
                        i, sample.shape[0], expected,
                    )
                    skipped_indices.append(i)
                    continue
                sample = sample[:-1]  # next-token alignment, matching the vLLM path
            except torch.cuda.OutOfMemoryError:
                log.warning("Skipping sample %d: OOM at length %d", i, expected)
                torch.cuda.empty_cache()
                skipped_indices.append(i)
                continue
            except Exception as exc:  # noqa: BLE001 - skip the sample, keep training
                log.warning("Skipping sample %d due to feature extraction error: %s", i, exc)
                skipped_indices.append(i)
                continue

            per_sample_feats.append(sample)
            lengths.append(sample.shape[0])
            feat_dim = sample.shape[-1]
            keep_indices.append(i)

        if skipped_indices:
            log.warning(
                "Skipped %d/%d samples due to hidden-state extraction failures. indices=%s",
                len(skipped_indices), len(input_ids_list), skipped_indices[:20],
            )

        if not per_sample_feats:
            return (
                torch.empty((0, 0, 0), dtype=torch.float32, device=device),
                torch.empty((0, 0), dtype=torch.long, device=device),
                keep_indices,
                skipped_indices,
            )

        max_len = max(lengths)
        feats = torch.zeros((len(per_sample_feats), max_len, feat_dim), dtype=torch.float32)
        attn_mask = torch.zeros((len(per_sample_feats), max_len), dtype=torch.long)
        for i, sample in enumerate(per_sample_feats):
            length = sample.shape[0]
            feats[i, :length] = sample
            attn_mask[i, :length] = 1

        return (
            feats.to(device, non_blocking=True),
            attn_mask.to(device, non_blocking=True),
            keep_indices,
            skipped_indices,
        )
