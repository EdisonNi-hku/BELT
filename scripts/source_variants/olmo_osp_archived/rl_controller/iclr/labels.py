from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from rl_controller.iclr.chains import is_correct, probe_draws

DEPTH_BLOCKS = 64.0
DEPTH_BINS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))
CELL_NAMES = {0: "wrong_wrong", 1: "recovery", 2: "overthinking", 3: "right_right"}


def depth_of(block: int) -> float:
    return min(block / DEPTH_BLOCKS, 1.0)


def rel_depth_of(prefix_len: int, thinking_tokens: int) -> float:
    if thinking_tokens == 0:
        return 0.0
    return prefix_len / thinking_tokens


def commit_index(probes: list[dict], final_answer: Any) -> int | None:
    if final_answer is None:
        return None
    start = None
    for index in range(len(probes) - 1, -1, -1):
        forced = probes[index]["forced"]
        if forced is None:
            continue
        if forced == final_answer:
            start = index
        else:
            break
    return start


def draw_labels(probe: dict, gold: Any) -> dict:
    draws = probe_draws(probe)
    y_now_mean = sum(is_correct(d["forced"], gold) for d in draws) / len(draws)
    consensus = None
    agree = None
    if len(draws) > 1:
        counts = Counter(d["forced"] for d in draws if d["forced"] is not None)
        if counts:
            consensus = counts.most_common(1)[0][1] / sum(counts.values())
        rest = Counter(d["forced"] for d in draws[1:] if d["forced"] is not None)
        if rest:
            ranked = rest.most_common()
            if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                mode = ranked[0][0]
                agree = int(probe["forced"] is not None and probe["forced"] == mode)
    return {
        "n_draws": len(draws),
        "y_now_mean": y_now_mean,
        "draw_consensus": consensus,
        "draw_agree_mode": agree,
    }


def chain_labels(chain: dict) -> list[dict]:
    probes = chain["probes"]
    gold = chain["gold"]
    final = chain["answer"]
    thinking_tokens = chain["thinking_tokens"]
    y_final = int(bool(chain["correct"]))
    y_now = [is_correct(p["forced"], gold) for p in probes]
    commit = commit_index(probes, final)
    runaway = int(chain["outcome"] == "length")
    fcs = next((i for i, y in enumerate(y_now) if y), None)
    rows = []
    for index, probe in enumerate(probes):
        reach = int(any(y_now[index + 1:]) or bool(y_final))
        rows.append({
            "qkey": chain["qkey"],
            "rep": chain["rep"],
            "probe_index": index,
            "block": probe["block"],
            "prefix_len": probe["prefix_len"],
            "depth": depth_of(probe["block"]),
            "rel_depth": rel_depth_of(probe["prefix_len"], thinking_tokens),
            "y_now": y_now[index],
            **draw_labels(probe, gold),
            "y_final": y_final,
            "cell": 2 * y_now[index] + y_final,
            "reach": reach,
            "futility": 1 - reach,
            "fcs_index": fcs,
            "post_fcs": int(fcs is not None and fcs < index),
            "stability": (None if final is None
                          else int(probe["forced"] is not None and probe["forced"] == final)),
            "time_to_commit": (None if commit is None
                               else max(0, probes[commit]["block"] - probe["block"])),
            "time_to_commit_tokens": (None if commit is None
                                      else max(0, probes[commit]["prefix_len"] - probe["prefix_len"])),
            "remaining_tokens": thinking_tokens - probe["prefix_len"],
            "runaway": runaway,
        })
    return rows


def sibling_context(chains: Sequence[dict]) -> dict[tuple[str, int], dict]:
    by_question: dict[str, list[dict]] = {}
    for chain in chains:
        by_question.setdefault(chain["qkey"], []).append(chain)
    context = {}
    for qkey, group in by_question.items():
        for chain in group:
            others = [o for o in group if o["rep"] != chain["rep"]]
            if not others:
                context[(qkey, chain["rep"])] = {
                    "sibling_pass_rate": None,
                    "sibling_majority": None,
                    "sibling_consensus": None,
                }
                continue
            pass_rate = sum(int(bool(o["correct"])) for o in others) / len(others)
            counts = Counter(o["answer"] for o in others if o["answer"] is not None)
            majority = None
            consensus = None
            if counts:
                ranked = counts.most_common()
                n_answered = sum(counts.values())
                consensus = ranked[0][1] / n_answered
                if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                    majority = ranked[0][0]
            context[(qkey, chain["rep"])] = {
                "sibling_pass_rate": pass_rate,
                "sibling_majority": majority,
                "sibling_consensus": consensus,
            }
    return context


def derive_labels(chains: list[dict]) -> list[dict]:
    context = sibling_context(chains)
    rows = []
    for chain in chains:
        siblings = context[(chain["qkey"], chain["rep"])]
        majority = siblings["sibling_majority"]
        for row in chain_labels(chain):
            probe = chain["probes"][row["probe_index"]]
            row["sibling_pass_rate"] = siblings["sibling_pass_rate"]
            row["sibling_consensus"] = siblings["sibling_consensus"]
            row["sibling_agree"] = (None if majority is None
                                    else int(probe["forced"] is not None
                                             and probe["forced"] == majority))
            rows.append(row)
    return rows


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def label_summary(rows: list[dict], *, n_chains: int | None = None) -> dict:
    n = len(rows)
    distinct_chains: dict[tuple, dict] = {}
    for r in rows:
        key = (r["qkey"], r["rep"])
        if key not in distinct_chains:
            distinct_chains[key] = r
    n_probed_chains = len(distinct_chains)
    cells = Counter(r["cell"] for r in rows)

    def _defined_fraction(key: str) -> float:
        return sum(1 for r in rows if r[key] is not None) / max(1, n)

    summary = {
        "n_probes": n,
        "n_probed_chains": n_probed_chains,
        "n_chains": n_chains if n_chains is not None else n_probed_chains,
        "n_questions": len({r["qkey"] for r in rows}),
        "y_now": _mean([r["y_now"] for r in rows]),
        "y_now_mean": _mean([r["y_now_mean"] for r in rows]),
        "draw_consensus": _mean([r["draw_consensus"] for r in rows]),
        "draw_consensus_defined_fraction": _defined_fraction("draw_consensus"),
        "n_draws_max": max((r["n_draws"] for r in rows), default=0),
        "y_final": _mean([r["y_final"] for r in rows]),
        "y_final_chain": _mean([r["y_final"] for r in distinct_chains.values()]),
        "reach": _mean([r["reach"] for r in rows]),
        "post_fcs": _mean([r["post_fcs"] for r in rows]),
        "fcs_first_probe_chain_fraction": _mean(
            [int(r["fcs_index"] == 0) for r in distinct_chains.values()]),
        "fcs_undefined_chain_fraction": _mean(
            [int(r["fcs_index"] is None) for r in distinct_chains.values()]),
        "stability": _mean([r["stability"] for r in rows]),
        "stability_defined_fraction": _defined_fraction("stability"),
        "time_to_commit_defined_fraction": _defined_fraction("time_to_commit"),
        "sibling_agree": _mean([r["sibling_agree"] for r in rows]),
        "sibling_agree_defined_fraction": _defined_fraction("sibling_agree"),
        "sibling_pass_rate": _mean([r["sibling_pass_rate"] for r in rows]),
        "sibling_pass_rate_defined_fraction": _defined_fraction("sibling_pass_rate"),
        "sibling_consensus": _mean([r["sibling_consensus"] for r in rows]),
        "sibling_consensus_defined_fraction": _defined_fraction("sibling_consensus"),
        "runaway": _mean([r["runaway"] for r in rows]),
        "runaway_chain": _mean([r["runaway"] for r in distinct_chains.values()]),
        "cell_fraction": {CELL_NAMES[c]: cells.get(c, 0) / max(1, n) for c in range(4)},
        "by_depth_axis": "rel_depth",
        "by_depth": [],
    }
    for lo, hi in DEPTH_BINS:
        sel = [r for r in rows if lo <= r["rel_depth"] < hi]
        if not sel:
            continue
        summary["by_depth"].append({
            "depth_lo": lo, "depth_hi": hi, "n": len(sel),
            "y_now": _mean([r["y_now"] for r in sel]),
            "reach": _mean([r["reach"] for r in sel]),
            "futility": _mean([r["futility"] for r in sel]),
            "post_fcs": _mean([r["post_fcs"] for r in sel]),
            "stability": _mean([r["stability"] for r in sel]),
            "overthinking": sum(r["cell"] == 2 for r in sel) / len(sel),
            "recovery": sum(r["cell"] == 1 for r in sel) / len(sel),
            "remaining_tokens": _mean([r["remaining_tokens"] for r in sel]),
        })
    return summary
