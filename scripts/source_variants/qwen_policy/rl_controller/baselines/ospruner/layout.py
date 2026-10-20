from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rl_controller.single_cot import probe_positions_in_ends

DEFAULT_N_UNFROZEN = 2

ALWAYS_FROZEN_PREFIXES = ("embed_tokens", "norm", "rotary_emb", "lm_head")

HEAD_PREFIX = "heads"


@dataclass(frozen=True)
class BackboneSpec:

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    tie_word_embeddings: bool = True

    @classmethod
    def from_config(cls, config: dict) -> "BackboneSpec":
        required = ("hidden_size", "num_hidden_layers", "num_attention_heads",
                    "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size")
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"backbone config lacks {missing}")
        return cls(**{k: int(config[k]) for k in required},
                   tie_word_embeddings=bool(config.get("tie_word_embeddings", True)))

    @classmethod
    def from_config_json(cls, path) -> "BackboneSpec":
        return cls.from_config(json.loads(Path(path).read_text(encoding="utf-8")))


    @property
    def q_out(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_out(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def attention_params(self) -> int:
        proj = (self.hidden_size * self.q_out
                + self.hidden_size * self.kv_out
                + self.hidden_size * self.kv_out
                + self.q_out * self.hidden_size)
        return proj + 2 * self.head_dim

    @property
    def mlp_params(self) -> int:
        return 3 * self.hidden_size * self.intermediate_size

    @property
    def block_params(self) -> int:
        return self.attention_params + self.mlp_params + 2 * self.hidden_size

    @property
    def total_params(self) -> int:
        embed = self.hidden_size * self.vocab_size
        untied = 0 if self.tie_word_embeddings else embed
        return embed + untied + self.num_hidden_layers * self.block_params + self.hidden_size


QWEN3_1P7B = BackboneSpec(
    hidden_size=2048,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=6144,
    vocab_size=151936,
    tie_word_embeddings=True,
)


def layer_split(n_layers: int, n_unfrozen: int = DEFAULT_N_UNFROZEN
                ) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    if not 0 <= n_unfrozen <= n_layers:
        raise ValueError(f"n_unfrozen must be in [0, {n_layers}], got {n_unfrozen}")
    cut = n_layers - n_unfrozen
    return tuple(range(cut)), tuple(range(cut, n_layers))


def describe_split(n_layers: int, n_unfrozen: int = DEFAULT_N_UNFROZEN) -> dict:
    frozen, trainable = layer_split(n_layers, n_unfrozen)
    return {
        "n_layers": n_layers,
        "n_unfrozen": n_unfrozen,
        "frozen_zero_based": list(frozen),
        "trainable_zero_based": list(trainable),
        "frozen_one_based": [i + 1 for i in frozen],
        "trainable_one_based": [i + 1 for i in trainable],
    }


def _layer_index(name: str) -> int | None:
    parts = name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def frozen_by_construction(name: str) -> bool:
    return (_layer_index(name) is None
            and bool(set(name.split(".")) & set(ALWAYS_FROZEN_PREFIXES)))


def is_trainable_param(name: str, *, n_layers: int,
                       n_unfrozen: int = DEFAULT_N_UNFROZEN) -> bool:
    _frozen, trainable = layer_split(n_layers, n_unfrozen)
    parts = name.split(".")
    if parts[0] == HEAD_PREFIX:
        return True
    if frozen_by_construction(name):
        return False
    index = _layer_index(name)
    if index is None:
        return False
    if not 0 <= index < n_layers:
        raise ValueError(f"parameter {name!r} names layer {index}, outside [0, {n_layers})")
    return index in trainable


def partition_parameters(names, *, n_layers: int,
                         n_unfrozen: int = DEFAULT_N_UNFROZEN
                         ) -> tuple[list[str], list[str]]:
    trainable, frozen = [], []
    for name in names:
        (trainable if is_trainable_param(name, n_layers=n_layers, n_unfrozen=n_unfrozen)
         else frozen).append(name)
    return trainable, frozen


def assert_frozen_backbone(named_requires_grad, *, n_layers: int,
                           n_unfrozen: int = DEFAULT_N_UNFROZEN) -> dict:
    leaked, missing = [], []
    n_trainable = 0
    for name, requires_grad in named_requires_grad:
        expected = is_trainable_param(name, n_layers=n_layers, n_unfrozen=n_unfrozen)
        if expected:
            n_trainable += 1
            if not requires_grad:
                missing.append(name)
        elif requires_grad:
            leaked.append(name)
    if leaked or missing:
        raise ValueError(
            f"parameter freeze plan violated: {len(leaked)} frozen-but-trainable "
            f"(e.g. {leaked[:3]}), {len(missing)} trainable-but-frozen (e.g. {missing[:3]})")
    if not n_trainable:
        raise ValueError("no trainable parameters: this reader would learn nothing")
    return {"n_trainable_tensors": n_trainable}


def trainable_param_count(spec: BackboneSpec, *, n_unfrozen: int = DEFAULT_N_UNFROZEN,
                          head_outputs: int = 1) -> dict:
    if head_outputs < 1:
        raise ValueError(f"head_outputs must be >= 1, got {head_outputs}")
    blocks = n_unfrozen * spec.block_params
    heads = head_outputs * (spec.hidden_size + 1)
    return {
        "blocks": blocks,
        "heads": heads,
        "trainable": blocks + heads,
        "backbone_total": spec.total_params,
        "trainable_fraction": (blocks + heads) / spec.total_params,
    }


def boundary_forward_flops(spec: BackboneSpec, *, prefix_tokens: int,
                           n_unfrozen: int = DEFAULT_N_UNFROZEN) -> dict:
    if prefix_tokens < 1:
        raise ValueError(f"prefix_tokens must be >= 1, got {prefix_tokens}")
    t = prefix_tokens
    per_block_matmul = 2 * spec.block_params * t
    per_block_attn = 2 * spec.hidden_size * t * t
    per_block = per_block_matmul + per_block_attn
    return {
        "prefix_tokens": t,
        "per_block_flops": per_block,
        "top_k_flops": n_unfrozen * per_block,
        "full_prefill_flops": spec.num_hidden_layers * per_block,
        "fraction_of_prefill": n_unfrozen / spec.num_hidden_layers,
        "aux_generated_tokens": 0,
    }


def boundary_token_index(prompt_len: int, end: int) -> int:
    if prompt_len < 0:
        raise ValueError(f"prompt_len must be >= 0, got {prompt_len}")
    if end < 1:
        raise ValueError(f"block end must be >= 1 (it is a prefix length), got {end}")
    return prompt_len + end - 1


def boundary_indices(prompt_len: int, block_ends) -> list[int]:
    ends = [int(e) for e in block_ends]
    if not ends:
        raise ValueError("chain has no block ends")
    for a, b in zip(ends, ends[1:]):
        if b <= a:
            raise ValueError(f"block_ends must be strictly increasing, got {a} then {b}")
    return [boundary_token_index(prompt_len, e) for e in ends]


def probe_end_positions(block_ends, query_points) -> list[int]:
    return probe_positions_in_ends(block_ends, query_points)


def probe_token_indices(prompt_len: int, block_ends, query_points) -> list[int]:
    all_indices = boundary_indices(prompt_len, block_ends)
    return [all_indices[p] for p in probe_end_positions(block_ends, query_points)]


@dataclass(frozen=True)
class BatchPlan:

    lengths: tuple[int, ...]
    gather: tuple[tuple[int, int], ...]
    offsets: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return len(self.lengths)

    @property
    def max_len(self) -> int:
        return max(self.lengths) if self.lengths else 0

    @property
    def n_boundaries(self) -> int:
        return len(self.gather)

    def rows(self) -> list[list[tuple[int, int]]]:
        return [list(self.gather[self.offsets[r]:self.offsets[r + 1]])
                for r in range(self.batch_size)]

    def attention_mask(self) -> list[list[int]]:
        width = self.max_len
        return [[1] * n + [0] * (width - n) for n in self.lengths]


def plan_batch(items) -> BatchPlan:
    lengths, gather, offsets = [], [], [0]
    for row, (prompt_len, seq_len, indices) in enumerate(items):
        if seq_len < 1:
            raise ValueError(f"row {row}: seq_len must be >= 1, got {seq_len}")
        if prompt_len < 0 or prompt_len > seq_len:
            raise ValueError(f"row {row}: prompt_len {prompt_len} outside [0, {seq_len}]")
        for index in indices:
            index = int(index)
            if not prompt_len <= index < seq_len:
                raise ValueError(
                    f"row {row}: boundary index {index} outside the completion span "
                    f"[{prompt_len}, {seq_len}) — it would read a prompt or PAD state")
            gather.append((row, index))
        lengths.append(int(seq_len))
        offsets.append(len(gather))
    if not lengths:
        raise ValueError("empty batch")
    return BatchPlan(tuple(lengths), tuple(gather), tuple(offsets))


def plan_chain_batch(chains, *, prompt_lens, probes_only: bool = True) -> BatchPlan:
    items = []
    for chain in chains:
        qkey = chain["qkey"]
        if qkey not in prompt_lens:
            raise ValueError(f"chain {(qkey, chain.get('rep'))}: no prompt length for {qkey!r}")
        prompt_len = int(prompt_lens[qkey])
        ends = [int(e) for e in chain["block_ends"]]
        if not ends:
            raise ValueError(f"chain {(qkey, chain.get('rep'))}: no block ends")
        seq_len = prompt_len + ends[-1]
        if probes_only:
            indices = probe_token_indices(prompt_len, ends, chain["query_points"])
            if len(indices) != len(chain["probes"]):
                raise ValueError(
                    f"chain {(qkey, chain['rep'])}: {len(indices)} probed boundaries for "
                    f"{len(chain['probes'])} probes — the objective zips pi against per-probe "
                    "rewards, so a mismatch would train on misaligned labels")
        else:
            indices = boundary_indices(prompt_len, ends)
        items.append((prompt_len, seq_len, indices))
    return plan_batch(items)
