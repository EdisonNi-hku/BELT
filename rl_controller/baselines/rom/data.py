from __future__ import annotations

from dataclasses import dataclass, field

from rl_controller.baselines.rom import class_balance as cb
from rl_controller.baselines.rom.features import FeatureSpec
from rl_controller.iclr import learning_curve as lc
from rl_controller.iclr import targets as tg
from rl_controller.iclr.chains import is_trainable
from rl_controller.single_cot import probe_positions_in_ends

DT_MODES = ("token_gap", "uniform", "zero")


def block_dt(prefix_lens, thinking_tokens: int, mode: str = "token_gap"):
    if mode not in DT_MODES:
        raise ValueError(f"dt mode must be one of {DT_MODES}, got {mode!r}")
    ends = [int(e) for e in prefix_lens]
    n = len(ends)
    if n == 0:
        return []
    if any(b <= a for a, b in zip(ends, ends[1:])):
        raise ValueError(f"block ends must be strictly increasing, got {ends[:8]}"
                         f"{'...' if n > 8 else ''}")
    if ends[0] < 0:
        raise ValueError(f"block ends must be non-negative, got {ends[0]}")
    total = float(thinking_tokens)
    if total <= 0:
        raise ValueError(f"thinking_tokens must be positive when a chain has {n} block ends, "
                         f"got {thinking_tokens!r}")
    if ends[-1] > total:
        raise ValueError(f"last block end {ends[-1]} exceeds thinking_tokens {total:.0f}; "
                         "dt would exceed 1 and amplify the recurrent state")
    if mode == "zero":
        return [0.0] * n
    if mode == "uniform":
        step = 1.0 / (n - 1) if n > 1 else 0.0
        return [0.0] + [step] * (n - 1)
    out = [0.0]
    for prev, cur in zip(ends, ends[1:]):
        out.append((cur - prev) / total)
    if any(not (0.0 <= v <= 1.0) for v in out):
        raise ValueError(f"dt out of [0, 1]: {out[:8]}")
    return out


VAL_COUNT = 320
VAL_SALT = lc.VAL_SALT


@dataclass
class Example:
    qkey: str
    rep: int
    x: list
    dt: list
    probe_pos: list
    labels: list
    fcs_index: int | None
    prompt: object = None
    weight: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def n_blocks(self) -> int:
        return len(self.x)

    @property
    def n_labelled(self) -> int:
        return sum(1 for value in self.labels if value is not None)


def _check_alignment(chain: dict) -> None:
    probes = chain["probes"]
    if not probes:
        return
    qp_prefixes = [qp[1] for qp in chain["query_points"]]
    probe_prefixes = [p["prefix_len"] for p in probes]
    if qp_prefixes != probe_prefixes:
        raise ValueError(
            f"chain {(chain['qkey'], chain['rep'])}: query_points prefixes {qp_prefixes} "
            f"do not align with probes[].prefix_len {probe_prefixes}")


def _validate(chain: dict, probe_pos: list, labels: list) -> None:
    probes = chain["probes"]
    if len(probe_pos) != len(probes):
        raise ValueError(f"chain {(chain['qkey'], chain['rep'])}: {len(probe_pos)} probe "
                         f"positions for {len(probes)} probes")
    if len(labels) != len(probes):
        raise ValueError(f"chain {(chain['qkey'], chain['rep'])}: {len(labels)} `o` labels for "
                         f"{len(probes)} probes -- targets.labels must be probe-aligned")
    if len(set(probe_pos)) != len(probe_pos):
        raise ValueError(f"chain {(chain['qkey'], chain['rep'])}: two probes share a block end "
                         f"(positions {probe_pos})")


def build_example(chain: dict, cache, spec: FeatureSpec, *, dt_mode: str = "token_gap",
                  drop_never_correct: bool = False, with_prompt: bool = True) -> Example:
    ends = [int(e) for e in chain["block_ends"]]
    _check_alignment(chain)
    probe_pos = probe_positions_in_ends(ends, chain["query_points"])
    marked = dict(chain)
    if drop_never_correct:
        marked[tg.ROM_DROP_FLAG] = True
    labels = tg.labels("o", marked)
    _validate(chain, probe_pos, labels)
    x = cache.chain_matrix(chain, spec)
    prompt = cache.prompt_matrix(chain, spec) if with_prompt else None
    return Example(
        qkey=chain["qkey"], rep=int(chain["rep"]), x=x,
        dt=block_dt(ends, int(chain["thinking_tokens"]), dt_mode),
        probe_pos=probe_pos, labels=list(labels),
        fcs_index=tg.first_correct_probe(chain), prompt=prompt,
        meta={"thinking_tokens": int(chain["thinking_tokens"]), "n_ends": len(ends)})


def build_examples(chains, cache, spec: FeatureSpec, *, dt_mode: str = "token_gap",
                   drop_never_correct: bool = False, with_prompt: bool = True,
                   require_trainable: bool = True):
    from rl_controller.baselines.rom.features import FeatureCacheError

    out, skipped = [], {"not_trainable": 0, "no_probes": 0, "uncached": [], "no_blocks": 0}
    for chain in chains:
        if require_trainable and not is_trainable(chain):
            skipped["not_trainable"] += 1
            continue
        if not chain["probes"]:
            skipped["no_probes"] += 1
            continue
        if not chain["block_ends"]:
            skipped["no_blocks"] += 1
            continue
        try:
            out.append(build_example(chain, cache, spec, dt_mode=dt_mode,
                                     drop_never_correct=drop_never_correct,
                                     with_prompt=with_prompt))
        except FeatureCacheError as exc:
            skipped["uncached"].append({"qkey": chain["qkey"], "rep": chain["rep"],
                                        "reason": str(exc)})
    return out, skipped


def class_counts(examples) -> dict:
    pos = neg = 0.0
    for example in examples:
        for value in example.labels:
            if value is None:
                continue
            if float(value) == 1.0:
                pos += example.weight
            else:
                neg += example.weight
    return {"n_pos": pos, "n_neg": neg, "n_labelled": pos + neg}


def pos_weight(examples) -> float | None:
    counts = class_counts(examples)
    if counts["n_pos"] == 0 or counts["n_neg"] == 0:
        return None
    return counts["n_neg"] / counts["n_pos"]


def oversample(examples, *, seed: int):
    major = [(e.qkey, e.rep) for e in examples if e.fcs_index == 0]
    minor = [(e.qkey, e.rep) for e in examples if e.fcs_index is not None and e.fcs_index >= 1]
    plan = {"majority": len(major), "minority": len(minor),
            "never_correct": sum(1 for e in examples if e.fcs_index is None)}
    if not minor or len(minor) >= len(major):
        plan.update(applicable=False, ratio=(len(major) / len(minor)) if minor else None,
                    reason=("no chains with fcs_index >= 1 to oversample" if not minor
                            else "minority already >= majority; ratio would be <= 1"))
        return list(examples), plan
    mult = cb.oversample_multiplicities({"major": major, "minor": minor}, seed=seed)
    out = []
    for example in examples:
        out.extend([example] * mult.get((example.qkey, example.rep), 1))
    plan.update(applicable=True, ratio=len(major) / len(minor), reason=None,
                n_examples_before=len(examples), n_examples_after=len(out))
    return out, plan


def val_train_split(chains, *, val_count: int = VAL_COUNT, salt: str = VAL_SALT):
    pool_keys = {c["qkey"] for c in chains if is_trainable(c)}
    val_keys, train_order = lc.partition(pool_keys, n_val=val_count, salt=salt)
    return set(val_keys), set(train_order)


def collate(examples, *, device=None, dtype=None, pad_prompt: bool = True):
    import torch

    if not examples:
        raise ValueError("cannot collate an empty batch")
    dtype = dtype or torch.float32
    batch = len(examples)
    steps = max(e.n_blocks for e in examples)
    probes = max(len(e.probe_pos) for e in examples)
    width = len(examples[0].x[0])

    x = torch.zeros(batch, steps, width, dtype=dtype)
    dt = torch.zeros(batch, steps, dtype=dtype)
    lengths = torch.zeros(batch, dtype=torch.long)
    gather = torch.zeros(batch, probes, dtype=torch.long)
    label = torch.zeros(batch, probes, dtype=dtype)
    mask = torch.zeros(batch, probes, dtype=dtype)

    for row, example in enumerate(examples):
        n = example.n_blocks
        x[row, :n] = torch.as_tensor(example.x, dtype=dtype).reshape(n, width)
        dt[row, :n] = torch.as_tensor(example.dt, dtype=dtype)
        lengths[row] = n
        for k, (position, value) in enumerate(zip(example.probe_pos, example.labels)):
            gather[row, k] = int(position)
            if value is not None:
                label[row, k] = float(value)
                mask[row, k] = example.weight

    prompt = prompt_mask = None
    prompts = [e.prompt for e in examples]
    if pad_prompt and any(p is not None for p in prompts):
        if any(p is None for p in prompts):
            raise ValueError("prompt features are present for some chains and missing for "
                             "others; m_0 must be initialised the same way for every chain "
                             "in a run (deviation D7 is per-run, not per-chain)")
        max_prompt = max(len(p) for p in prompts)
        prompt = torch.zeros(batch, max_prompt, width, dtype=dtype)
        prompt_mask = torch.zeros(batch, max_prompt, dtype=torch.bool)
        for row, rows in enumerate(prompts):
            n = len(rows)
            prompt[row, :n] = torch.as_tensor(rows, dtype=dtype).reshape(n, width)
            prompt_mask[row, :n] = True

    out = {"x": x, "dt": dt, "lengths": lengths, "gather": gather, "label": label,
           "mask": mask, "prompt": prompt, "prompt_mask": prompt_mask}
    if device is not None:
        out = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in out.items()}
    return out


def gather_probe_logits(logits, gather):
    if logits.dim() != 3:
        raise ValueError(f"expected (B, M, C) logits, got {tuple(logits.shape)}")
    index = gather.unsqueeze(-1).expand(-1, -1, logits.size(-1))
    return logits.gather(1, index)
