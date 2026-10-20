from __future__ import annotations

from rl_controller.iclr import targets as tg
from rl_controller.iclr.chains import is_correct, is_trainable

TARGET = "stability"

DEVIATION = (
    "consistency is measured against the chain's NATURAL </think> end answer "
    "(arXiv:2505.18404 uses a forced max-budget completion); equality is the "
    "ladder's loose `==` with a not-None guard, not the type-exact grader"
)


def consistency_labels(chain: dict) -> list:
    return tg.labels(TARGET, chain)


def strict_labels(chain: dict) -> list:
    answer = chain["answer"]
    if answer is None:
        return [None] * len(chain["probes"])
    return [float(is_correct(p["forced"], answer)) for p in chain["probes"]]


def convention_disagreement(chains) -> dict:
    n, n_diff = 0, 0
    examples = []
    for chain in chains:
        if not is_trainable(chain):
            continue
        for i, (loose, strict) in enumerate(zip(consistency_labels(chain), strict_labels(chain))):
            if loose is None or strict is None:
                continue
            n += 1
            if loose != strict:
                n_diff += 1
                if len(examples) < 10:
                    examples.append({"qkey": chain["qkey"], "rep": chain["rep"], "probe": i,
                                     "forced": chain["probes"][i]["forced"],
                                     "answer": chain["answer"],
                                     "loose": loose, "strict": strict})
    return {"n_boundaries": n, "n_disagree": n_diff,
            "rate": (n_diff / n) if n else None,
            "threshold": 0.005,
            "strict_arm_required": bool(n and (n_diff / n) > 0.005),
            "examples": examples}


def label_stats(chains) -> dict:
    n_chains = n_labelled = n_positive = 0
    n_masked_chains = n_all_one = n_all_zero = 0
    per_chain_rates = []
    for chain in chains:
        if not is_trainable(chain):
            continue
        n_chains += 1
        labels = [x for x in consistency_labels(chain) if x is not None]
        if not labels:
            n_masked_chains += 1
            continue
        n_labelled += len(labels)
        positives = sum(1 for x in labels if x > 0.5)
        n_positive += positives
        per_chain_rates.append(positives / len(labels))
        if positives == len(labels):
            n_all_one += 1
        elif positives == 0:
            n_all_zero += 1
    usable = n_chains - n_masked_chains
    return {
        "target": TARGET,
        "deviation": DEVIATION,
        "n_chains": n_chains,
        "n_masked_chains": n_masked_chains,
        "masked_chain_fraction": (n_masked_chains / n_chains) if n_chains else None,
        "n_labelled_boundaries": n_labelled,
        "base_rate": (n_positive / n_labelled) if n_labelled else None,
        "all_one_fraction": (n_all_one / usable) if usable else None,
        "all_zero_fraction": (n_all_zero / usable) if usable else None,
        "degenerate_all_one_gate": 0.8,
        "weighted_arm_required": bool(usable and (n_all_one / usable) > 0.8),
        "mean_per_chain_rate": (sum(per_chain_rates) / len(per_chain_rates))
                               if per_chain_rates else None,
    }


def trailing_mean(values, window: int) -> list:
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    out = []
    for i in range(len(values)):
        seen = [v for v in values[max(0, i - window + 1): i + 1] if v is not None]
        out.append((sum(seen) / len(seen)) if seen else None)
    return out
