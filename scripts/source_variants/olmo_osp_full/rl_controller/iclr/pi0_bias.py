from __future__ import annotations

from typing import Callable, Mapping

from rl_controller.iclr.chains import is_correct
from rl_controller.iclr.labels import DEPTH_BINS, rel_depth_of
from rl_controller.single_cot import stop_earliest_agreement

Plan = Callable[[dict], dict | None]


def natural_plan(chain: dict) -> dict | None:
    return None


def cap_plan(cap: int) -> Plan:
    def plan(view: dict) -> dict | None:
        if view["blocks"] <= cap:
            return None
        under = None
        for probe in view["probes"]:
            if probe["block"] > cap:
                return under if under is not None else probe
            under = probe
        return under
    return plan


def consecutive_agreement(view: dict) -> dict | None:
    probes = view["probes"]
    for j in range(1, len(probes)):
        forced = probes[j]["forced"]
        if forced is not None and forced == probes[j - 1]["forced"]:
            return probes[j]
    return None


DOWNSTREAM: dict[str, Plan] = {
    "natural": natural_plan,
    "earliest_agreement": stop_earliest_agreement,
    "consecutive_agreement": consecutive_agreement,
    "cap32": cap_plan(32),
    "cap64": cap_plan(64),
}


def plan_from_predicate(predicate) -> Plan:
    from rl_controller import s1_depth_margin as dm

    def plan(chain: dict) -> dict | None:
        scores = chain.get("scores")
        if scores is None:
            raise ValueError("predicate plans need scored chains (chain['scores'])")
        for probe, score in zip(chain["probes"], scores):
            if predicate(dm.probe_features(probe, score)):
                return probe
        return None

    return plan


def endpoint_after(chain: dict, probe_index: int, plan: Plan) -> dict:
    view = dict(chain)
    view["probes"] = chain["probes"][probe_index + 1:]
    if "scores" in chain:
        view["scores"] = chain["scores"][probe_index + 1:]
    probe = plan(view)
    if probe is None:
        return {"endpoint": "natural", "y": int(bool(chain["correct"])),
                "thinking_tokens": chain["thinking_tokens"]}
    return {"endpoint": "forced", "y": is_correct(probe["forced"], chain["gold"]),
            "thinking_tokens": probe["prefix_len"]}


def continue_targets(chains: list[dict], plan: Plan) -> list[dict]:
    rows = []
    for chain in chains:
        for index, probe in enumerate(chain["probes"]):
            endpoint = endpoint_after(chain, index, plan)
            rows.append({
                "qkey": chain["qkey"], "rep": chain["rep"], "probe_index": index,
                "rel_depth": rel_depth_of(probe["prefix_len"], chain["thinking_tokens"]),
                "y_now": is_correct(probe["forced"], chain["gold"]),
                "y_cont": endpoint["y"],
                "cont_tokens": endpoint["thinking_tokens"],
                "endpoint": endpoint["endpoint"],
            })
    return rows


def _compare(base: list[dict], rows: list[dict]) -> dict:
    assert len(base) == len(rows), "base/rows must be positionally aligned continue_targets"
    n = len(rows)
    if n == 0:
        return {"n": 0, "y_cont_natural": None, "y_cont_pi0": None,
                "changed_fraction": None, "greedy_flip_fraction": None,
                "mean_token_delta": None, "forced_fraction": None}
    greedy = lambda r: r["y_now"] >= r["y_cont"]
    return {
        "n": n,
        "y_cont_natural": sum(b["y_cont"] for b in base) / n,
        "y_cont_pi0": sum(r["y_cont"] for r in rows) / n,
        "changed_fraction": sum(b["y_cont"] != r["y_cont"] for b, r in zip(base, rows)) / n,
        "greedy_flip_fraction": sum(greedy(b) != greedy(r) for b, r in zip(base, rows)) / n,
        "mean_token_delta": sum(r["cont_tokens"] - b["cont_tokens"] for b, r in zip(base, rows)) / n,
        "forced_fraction": sum(r["endpoint"] == "forced" for r in rows) / n,
    }


def bias_report(chains: list[dict], policies: Mapping[str, Plan] = DOWNSTREAM) -> dict:
    base = continue_targets(chains, natural_plan)
    n_chains = len({(chain["qkey"], chain["rep"]) for chain in chains})
    report = {}
    for name, plan in policies.items():
        rows = continue_targets(chains, plan)
        overall = _compare(base, rows)
        by_depth = []
        for lo, hi in DEPTH_BINS:
            pairs = [(b, r) for b, r in zip(base, rows) if lo <= r["rel_depth"] < hi]
            if pairs:
                entry = _compare([b for b, _ in pairs], [r for _, r in pairs])
                entry.update({"depth_lo": lo, "depth_hi": hi})
                by_depth.append(entry)
        report[name] = {
            "n_probes": overall["n"],
            "n_chains": n_chains,
            **{k: v for k, v in overall.items() if k != "n"},
            "by_depth_axis": "rel_depth",
            "by_depth": by_depth,
        }
    return report
