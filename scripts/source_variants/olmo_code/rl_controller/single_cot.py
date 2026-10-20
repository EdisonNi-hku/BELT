from __future__ import annotations

from rl_controller.budget_episode import CONTINUE
from rl_controller.rollout_runner import GenRequest, draw_steps

QUERY_STRIDE = 4


def _always_continue(observation) -> str:
    return CONTINUE


def run_chain_batch(engine, specs, *, encode, budget: int, query_stride: int = QUERY_STRIDE,
                    block_cap: int = 0):
    active = []
    for spec in specs:
        records: list = []

        def observe(block, completion_ids, block_ends, _r=records):
            _r.append((int(block), len(completion_ids)))
            return {}

        gen = draw_steps(prompt_ids=spec["prompt_ids"], policy=_always_continue,
                         observe=observe, encode=encode,
                         max_blocks=10 ** 9, max_total_tokens=budget,
                         budget_left=budget, query_stride=query_stride,
                         block_cap=block_cap, request_seed_base=spec["seed_key"])
        item = {"spec": spec, "gen": gen, "query_points": records,
                "pending": gen.send(None), "result": None}
        active.append(item)

    while any(a["result"] is None for a in active):
        waiting = [a for a in active if a["result"] is None]
        outs = engine.generate_batch([a["pending"] for a in waiting])
        if len(outs) != len(waiting):
            raise RuntimeError(f"engine returned {len(outs)} outputs for {len(waiting)} requests")
        for a, out in zip(waiting, outs):
            try:
                a["pending"] = a["gen"].send(out)
            except StopIteration as stop:
                a["result"] = stop.value
                a["pending"] = None
    return active


def stop_earliest_agreement(chain) -> dict | None:
    final = chain.get("answer")
    if final is None:
        return None
    for probe in chain["probes"]:
        if probe["forced"] is not None and probe["forced"] == final:
            return probe
    return None


def stop_earliest_correct(chain) -> dict | None:
    for probe in chain["probes"]:
        if probe["forced"] is not None and probe["forced"] == chain["gold"]:
            return probe
    return None


def stop_at_cap(chain, cap_block: int) -> dict | None:
    if chain["blocks"] <= cap_block:
        return None
    candidate = None
    for probe in chain["probes"]:
        if probe["block"] <= cap_block:
            candidate = probe
        else:
            break
    return candidate


def frontier_row(chains, plan, *, name: str) -> dict:
    if not chains:
        raise ValueError("no chains")
    n = len(chains)
    natural_thinking = sum(c["thinking_tokens"] for c in chains)
    correct = 0
    thinking = 0
    total = 0
    fired = 0
    for c in chains:
        probe = plan(c)
        if probe is None:
            correct += bool(c["correct"])
            thinking += c["thinking_tokens"]
            total += c["thinking_tokens"]
        else:
            fired += 1
            correct += (probe["forced"] is not None and probe["forced"] == c["gold"])
            thinking += probe["prefix_len"]
            total += probe["prefix_len"] + probe["forced_tokens"]
    return {
        "name": name,
        "n": n,
        "accuracy": correct / n,
        "fired_fraction": fired / n,
        "mean_thinking_tokens": thinking / n,
        "mean_total_tokens": total / n,
        "thinking_saving": (1.0 - thinking / natural_thinking) if natural_thinking else 0.0,
    }


def oracle_report(chains, *, caps=(16, 32, 64, 96, 128)) -> dict:
    rows = [frontier_row(chains, lambda c: None, name="natural")]
    rows.append(frontier_row(chains, stop_earliest_agreement, name="oracle_agreement"))
    rows.append(frontier_row(chains, stop_earliest_correct, name="oracle_correct"))
    for cap in caps:
        rows.append(frontier_row(chains, lambda c, _b=cap: stop_at_cap(c, _b), name=f"cap{cap}"))
    natural = rows[0]
    agreement = rows[1]
    loss_pp = (natural["accuracy"] - agreement["accuracy"]) * 100.0
    dominated = any(
        r["name"].startswith("cap")
        and r["thinking_saving"] >= agreement["thinking_saving"]
        and r["accuracy"] >= agreement["accuracy"]
        for r in rows
    )
    return {
        "rows": rows,
        "gate_s0": {
            "agreement_saving": agreement["thinking_saving"],
            "agreement_loss_pp": loss_pp,
            "saving_ok": agreement["thinking_saving"] >= 0.15,
            "loss_ok": loss_pp <= 1.0,
            "dominated_by_cap": dominated,
            "pass": (agreement["thinking_saving"] >= 0.15 and loss_pp <= 1.0
                     and not dominated),
        },
    }


def stop_by_scores(chain, *, margin: float) -> dict | None:
    for probe, s in zip(chain["probes"], chain["scores"]):
        if s["q"] >= s["v"] - margin:
            return probe
    return None


def stop_by_threshold(chain, *, theta: float) -> dict | None:
    for probe, s in zip(chain["probes"], chain["scores"]):
        if s["q"] >= theta:
            return probe
    return None


def score_frontier(chains, *, margins, thetas) -> list:
    rows = []
    for m in margins:
        rows.append(frontier_row(chains, lambda c, _m=m: stop_by_scores(c, margin=_m),
                                 name=f"rule_margin{m:+.2f}"))
    for t in thetas:
        rows.append(frontier_row(chains, lambda c, _t=t: stop_by_threshold(c, theta=_t),
                                 name=f"rule_theta{t:.2f}"))
    return rows


def average_precision(labels, scores) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(1 for _, y in pairs if y)
    if n_pos == 0:
        return 0.0
    tp = 0
    ap = 0.0
    for i, (_, y) in enumerate(pairs, start=1):
        if y:
            tp += 1
            ap += tp / i
    return ap / n_pos


def probe_positions_in_ends(block_ends, query_points) -> list:
    where = {int(e): i for i, e in enumerate(block_ends)}
    out = []
    for _block, plen in query_points:
        if int(plen) not in where:
            raise ValueError(f"probe prefix {plen} is not a recorded block end")
        out.append(where[int(plen)])
    return out


def report_frontier(chains, *, margin: float, caps=(16, 32, 64, 96, 128)) -> dict:
    if not isinstance(margin, float):
        raise TypeError(f"margin must be ONE float chosen on select, got {type(margin).__name__}")
    rows = [frontier_row(chains, lambda c: None, name="natural")]
    rows.append(frontier_row(chains, lambda c, _m=margin: stop_by_scores(c, margin=_m),
                             name=f"rule_margin{margin:+.2f}"))
    for cap in caps:
        rows.append(frontier_row(chains, lambda c, _b=cap: stop_at_cap(c, _b), name=f"cap{cap}"))
    rows.append(frontier_row(chains, stop_earliest_agreement, name="oracle_agreement_REFERENCE"))
    rows.append(frontier_row(chains, stop_earliest_correct, name="oracle_correct_REFERENCE"))
    natural, rule = rows[0], rows[1]
    return {
        "rows": rows,
        "report": {
            "margin": margin,
            "rule_saving": rule["thinking_saving"],
            "rule_loss_pp": (natural["accuracy"] - rule["accuracy"]) * 100.0,
            "rule_fired": rule["fired_fraction"],
            "dominated_by_cap": any(
                r["name"].startswith("cap")
                and r["thinking_saving"] >= rule["thinking_saving"]
                and r["accuracy"] >= rule["accuracy"]
                for r in rows
            ),
        },
    }
