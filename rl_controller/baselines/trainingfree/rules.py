from __future__ import annotations

import math

from rl_controller.iclr import ladder as ld


BELIEF = "q"
OVERHEAD_KEY = "ovh"
STAT = "s"
NOFIRE_KEY = "nofire"
EWT_KEY = "ewt"


def _fireable(scores) -> list[bool]:
    return [not float(s.get(NOFIRE_KEY, 0.0)) for s in scores]


def _measured(values, positions, fireable, upto: int):
    keep = [j for j in range(upto + 1) if fireable[j]]
    return [values[j] for j in keep], [positions[j] for j in keep]


def _probe_stats(chain, belief: str = BELIEF):
    scores, probes = chain["scores"], chain["probes"]
    if len(scores) != len(probes):
        raise ValueError(
            f"chain {(chain.get('qkey'), chain.get('rep'))}: {len(scores)} scores for "
            f"{len(probes)} probes")
    return ([float(s[belief]) for s in scores],
            [max(1, int(p["prefix_len"])) for p in probes])


CODE_DELTA = 0.55
CODE_RAMP_MIN = 0.9
CODE_RAMP_GAP = 0.05
CODE_RAMP_STEPS = 2
CODE_MIN_CHECKS = 3


def degeneration_score(confidences, positions, *, use_log: bool = True,
                       delta: float = CODE_DELTA) -> float:
    if len(confidences) != len(positions):
        raise ValueError("confidences and positions must align")
    if len(confidences) < CODE_MIN_CHECKS:
        return 0.0
    values = ([math.log(max(c, 1e-12)) for c in confidences] if use_log
              else list(confidences))
    horizon = positions[-1]
    total = 0.0
    for j in range(len(values) - 1):
        p1, p2 = values[j], values[j + 1]
        if not p1 > p2:
            continue
        if 2.0 * p2 - p1 < delta:
            total += math.log1p(horizon / positions[j + 1] - 1.0) + 1.0
    return total


def ramp_floor(conf_threshold: float) -> float:
    return max(0.0, round(conf_threshold - CODE_RAMP_GAP, 10))


def ramped_threshold(index: int, *, conf_threshold: float, ramp_steps: int = CODE_RAMP_STEPS,
                     ramp_min: float | None = None) -> float:
    if ramp_min is None:
        ramp_min = ramp_floor(conf_threshold)
    alpha = min(1.0, index / ramp_steps) if ramp_steps > 0 else 1.0
    return ramp_min + (conf_threshold - ramp_min) * alpha


def code_stop_plan(tau: float, conf_threshold: float, *, use_log: bool = True,
                   delta: float = CODE_DELTA, ramp_steps: int = CODE_RAMP_STEPS,
                   ramp_min: float | None = None, spec=None, belief: str | None = None):
    key = belief or (spec.belief_key if spec is not None else BELIEF)

    def plan(chain):
        confidences, positions = _probe_stats(chain, key)
        fireable = _fireable(chain["scores"])
        scores = chain["scores"]
        n_q, sum_log_t, seen, prev = 0, 0.0, 0, None
        for i in range(len(confidences)):
            if not fireable[i]:
                continue
            value = (math.log(max(confidences[i], 1e-12)) if use_log else confidences[i])
            if prev is not None and prev > value and 2.0 * value - prev < delta:
                n_q += 1
                sum_log_t += math.log(positions[i])
            prev = value
            seen += 1
            degen = (n_q + (n_q * math.log(positions[i]) - sum_log_t)
                     if seen >= CODE_MIN_CHECKS else 0.0)
            r = ramped_threshold(i, conf_threshold=conf_threshold,
                                 ramp_steps=ramp_steps, ramp_min=ramp_min)
            ewt = float(scores[i].get(EWT_KEY, 1.0)) > 0.0
            if (ewt and confidences[i] > r) or degen > tau:
                return chain["probes"][i]
        return None

    return plan


CONFIDENT, DEGENERATE, BOTH, NONE = "confident", "degenerate", "both", "none"


def code_stop_exit(chain, tau: float, conf_threshold: float, **kw) -> tuple:
    confidences, positions = _probe_stats(chain)
    use_log = kw.get("use_log", True)
    delta = kw.get("delta", CODE_DELTA)
    ramp_steps = kw.get("ramp_steps", CODE_RAMP_STEPS)
    ramp_min = kw.get("ramp_min", ramp_floor(conf_threshold))
    fireable = _fireable(chain["scores"])
    for i in range(len(confidences)):
        if not fireable[i]:
            continue
        conf_seen, pos_seen = _measured(confidences, positions, fireable, i)
        degen = degeneration_score(conf_seen, pos_seen, use_log=use_log, delta=delta)
        r = ramped_threshold(i, conf_threshold=conf_threshold, ramp_steps=ramp_steps,
                             ramp_min=ramp_min)
        ewt = float(chain["scores"][i].get(EWT_KEY, 1.0)) > 0.0
        hot, dead = (ewt and confidences[i] > r), degen > tau
        if hot or dead:
            return (BOTH if hot and dead else (CONFIDENT if hot else DEGENERATE),
                    chain["probes"][i])
    return (NONE, None)


def exit_attribution(chains, tau: float, conf_threshold: float, **kw) -> dict:
    counts = {CONFIDENT: 0, DEGENERATE: 0, BOTH: 0, NONE: 0}
    for chain in chains:
        counts[code_stop_exit(chain, tau, conf_threshold, **kw)[0]] += 1
    stopped = counts[CONFIDENT] + counts[DEGENERATE] + counts[BOTH]
    total = max(1, sum(counts.values()))
    return {**counts, "n_chains": sum(counts.values()), "stopped": stopped,
            "stop_fraction": stopped / total,
            "confident_fraction_of_stops": (counts[CONFIDENT] + counts[BOTH]) / max(1, stopped),
            "degenerate_fraction_of_stops": (counts[DEGENERATE] + counts[BOTH]) / max(1, stopped)}


def mean_steps(chains) -> float:
    if not chains:
        raise ValueError("mean_steps of an empty pool")
    return sum(len(c["probes"]) for c in chains) / len(chains)


def kbar_transfer(tau: float, kbar_src: float, kbar_dst: float) -> float:
    if kbar_src <= 0:
        raise ValueError("kbar_src must be > 0")
    return tau * kbar_dst / kbar_src


def exponential_moving_variance(values, timescale: float) -> list[float]:
    if timescale <= 0:
        raise ValueError("timescale must be > 0")
    if not values:
        return []
    alpha = 1.0 / timescale
    ema = [0.0] * len(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1.0 - alpha) * ema[i - 1]
    var = [0.0] * len(values)
    for i in range(1, len(values)):
        diff = values[i] - ema[i - 1]
        var[i] = (1.0 - alpha) * var[i - 1] + alpha * diff * diff
    return var


def eat_plan(threshold: float, timescale: float, warmup: int = 1, *, spec=None,
             belief: str | None = None):
    key = belief or (spec.belief_key if spec is not None else BELIEF)

    def plan(chain):
        values, _ = _probe_stats(chain, key)
        fireable = _fireable(chain["scores"])
        seen = [(i, v) for i, v in enumerate(values) if fireable[i]]
        if not seen:
            return None
        var = exponential_moving_variance([v for _, v in seen], timescale)
        for pos in range(max(1, warmup), len(var)):
            if var[pos] < threshold:
                return chain["probes"][seen[pos][0]]
        return None

    return plan


def answer_convergence_plan(k: int, *, spec=None):
    if k < 2:
        raise ValueError(f"consistency threshold k must be >= 2, got {k}")
    del spec

    def plan(chain):
        count, last = 0, None
        for i, probe in enumerate(chain["probes"]):
            answer = probe["forced"]
            if answer is not None and last is not None and answer == last:
                count += 1
            else:
                count = 1
                last = answer
            if count >= k:
                return probe
        return None

    return plan


WHEELS_CONDITIONS = (
    (1, (5,)),
    (4, (10, 50, 100, 1000)),
    (3, (20, 20, 20)),
    (6, (50, 50, 50, 50, 50, 50)),
)


def wheels_fires(ranks, scale: float = 1.0) -> bool:
    if not ranks:
        return False
    for lookback, cutoffs in WHEELS_CONDITIONS:
        if len(ranks) < lookback:
            continue
        window = ranks[-lookback:][::-1]
        if all(r <= cutoff * scale for r, cutoff in zip(window, cutoffs)):
            return True
    return False


def wheels_plan(scale: float, *, spec=None, belief: str | None = None):
    key = belief or (spec.belief_key if spec is not None else BELIEF)

    def plan(chain):
        series, _ = _probe_stats(chain, key)
        ranks = []
        for i, probe in enumerate(chain["probes"]):
            ranks.append(series[i])
            if wheels_fires(ranks, scale):
                return probe
        return None

    return plan


CODE_TAUS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0,
             8.0, 10.0, 13.0, 17.0, 22.0, 30.0, 40.0)
CODE_CONFS = (0.70, 0.80, 0.90, 0.95, 0.99)
DEER_THETAS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85,
               0.90, 0.925, 0.95, 0.965, 0.98, 0.99, 0.995, 0.999)
EAT_THRESHOLDS = (1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
EAT_TIMESCALES = (2.0, 5.0, 10.0)
ANSCONV_K = (2, 3, 4, 5, 6, 7, 8, 10)
WHEELS_SCALES = (0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0,
                 1000.0, 3000.0, 10000.0, 30000.0)

FACTORIES = {
    "tf_code_stop": lambda tau, conf, *, spec=None: code_stop_plan(tau, conf, spec=spec),
    "tf_code_stop_nolog": lambda tau, conf, *, spec=None: code_stop_plan(
        tau, conf, use_log=False, spec=spec),
    "tf_eat": lambda thr, ts, warmup, *, spec=None: eat_plan(thr, ts, warmup, spec=spec),
    "tf_ansconv": lambda k, *, spec=None: answer_convergence_plan(int(k), spec=spec),
    "tf_wheels": lambda scale, *, spec=None: wheels_plan(scale, spec=spec),
}

SPECS = (
    ld.FamilySpec(
        name="CoDE", kind="custom", factory="tf_code_stop", belief_key=BELIEF,
        direction="ge", params=("tau", "c"), formats=("{:g}", "{:.2f}"),
        requires=(NOFIRE_KEY,),
        grid=tuple((tau, conf) for conf in CODE_CONFS for tau in CODE_TAUS),
        belief_source="CoDE-Stop (arXiv 2604.04930) dual exit on the teacher-forced "
                      "probe-answer confidence; delta=0.55 (inert under their log default), "
                      "ramp_min=0.9, ramp_steps=2; ewt gate not reproducible"),
    ld.FamilySpec(
        name="CoDEnolog", kind="custom", factory="tf_code_stop_nolog", belief_key=BELIEF,
        direction="ge", params=("tau", "c"), formats=("{:g}", "{:.2f}"),
        requires=(NOFIRE_KEY,),
        grid=tuple((tau, conf) for conf in CODE_CONFS for tau in CODE_TAUS),
        belief_source="CoDE-Stop with use_log=False -- the variant in which their "
                      "delta=0.55 instability threshold is actually live, i.e. the rule "
                      "their PAPER describes rather than the one their code runs"),
    ld.FamilySpec(
        name="DEER", kind="threshold", belief_key=BELIEF, direction="ge",
        grid=DEER_THETAS, params=("t",), formats=("{:.3f}",),
        belief_source="DEER (Yang et al. 2025) as shipped in the CoDE-Stop repo: one "
                      "fixed confidence threshold on the same probe-answer confidence, "
                      "no ramp and no degeneration exit (method_deer.py:148)"),
    ld.FamilySpec(
        name="EAT", kind="custom", factory="tf_eat", belief_key=STAT, direction="le",
        params=("e", "ts", "w"), formats=("{:g}", "{:g}", "{:d}"),
        requires=(NOFIRE_KEY,),
        grid=tuple((thr, ts, w) for ts in EAT_TIMESCALES for thr in EAT_THRESHOLDS
                   for w in (1, 3)),
        belief_source="EAT (Wang et al. 2025) EMA-variance stop rule on a DETERMINISTIC "
                      "entropy proxy (their 10x20 sampled rollouts are not run); warmup "
                      "swept from 1 because their default 50 exceeds our boundary count"),
    ld.FamilySpec(
        name="CoDEcal", kind="custom", factory="tf_code_stop", belief_key="q_cal",
        direction="ge", params=("tau", "c"), formats=("{:g}", "{:.2f}"),
        requires=(NOFIRE_KEY,),
        grid=tuple((tau, conf) for conf in CODE_CONFS for tau in CODE_TAUS),
        belief_source="CoDE-Stop on the ladder's depth-aware logistic recalibration of "
                      "the probe-answer confidence (fitted on val, cross-fit within val). "
                      "C2 cell: zero trained parameters plus a calibrator"),
    ld.FamilySpec(
        name="CoDEiso", kind="custom", factory="tf_code_stop", belief_key="q_iso",
        direction="ge", params=("tau", "c"), formats=("{:g}", "{:.2f}"),
        requires=(NOFIRE_KEY,),
        grid=tuple((tau, conf) for conf in CODE_CONFS for tau in CODE_TAUS),
        belief_source="CoDE-Stop on a depth-AGNOSTIC isotonic recalibration fitted on val "
                      "(out-of-fold within val). C2 cell"),
    ld.FamilySpec(
        name="DEERcal", kind="threshold", belief_key="q_cal", direction="ge",
        grid=DEER_THETAS, params=("t",), formats=("{:.3f}",),
        belief_source="DEER on the ladder's depth-aware logistic recalibration. C2 cell"),
    ld.FamilySpec(
        name="DEERiso", kind="threshold", belief_key="q_iso", direction="ge",
        grid=DEER_THETAS, params=("t",), formats=("{:.3f}",),
        belief_source="DEER on a depth-agnostic isotonic recalibration fitted on val. "
                      "C2 cell"),
    ld.FamilySpec(
        name="AnsConv", kind="custom", factory="tf_ansconv", belief_key=BELIEF,
        direction="ge", params=("k",), formats=("{:d}",), grid=ANSCONV_K,
        belief_source="Answer Convergence (Liu & Wang 2025) consecutive-agreement counter, "
                      "their exact reset semantics, on our pools' parsed forced answers; "
                      "k=2 should reproduce ladder.free_family's consecutive_agreement"),
    ld.FamilySpec(
        name="Wheels", kind="custom", factory="tf_wheels", belief_key=STAT,
        direction="le", params=("s",), formats=("{:g}",), grid=WHEELS_SCALES,
        belief_source="Stop Spinning Wheels: their fixed 4-condition cascade on the rank "
                      "of </think> in the next-token distribution, cutoffs scaled by s "
                      "(s=1 is their published rule; the scale knob is ours, to get a curve). "
                      "Generates nothing at decision time"),
)

NOT_IMPLEMENTABLE = {
    "think_or_not": (
        "Think-or-Not (Yong et al. 2025) scores each boundary by the entropy of the top-5 "
        "COMPLETED BEAM probabilities from a max_steps=10, top_k=5 beam search over the "
        "trial answer (method_think_or_not.py:31-60,104-106). That is ~50 extra forward "
        "passes per boundary over branched KV caches, and the statistic is a property of "
        "the beam frontier, not of any column a teacher-forced pass produces. It is "
        "deterministic, so it IS reproducible -- but only by a second GPU pass we did not "
        "cache. Marked N/A rather than approximated: a 1-step top-5 entropy is a different "
        "statistic and would misrepresent their method."),
}


def register(ladder_module=None) -> list:
    mod = ladder_module or ld
    for name, factory in FACTORIES.items():
        mod.register_rule_factory(name, factory)
    return [mod.register_family(spec) for spec in SPECS]


register()
