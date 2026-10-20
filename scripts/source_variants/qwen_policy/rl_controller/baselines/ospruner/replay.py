from __future__ import annotations

import hashlib

from rl_controller.iclr import osp_policy as osp
from rl_controller.iclr import pareto as pt

CONTRACT_IDENTITY_FIELDS = ("chain_bank_sha256", "manifest_sha256", "pool")

FAITHFUL_SALT = "ws-f-faithful/v1"

STRENGTHENED_V = 0.0

BERNOULLI_THRESHOLD = 0.5


def validated_for_consumer(rows) -> list[dict]:
    return [pt.validate_policy_replay_row(row) for row in rows]


def assert_pareto_basis_fix() -> dict:
    try:
        pt.Bundle()
    except Exception:
        return {"pareto_basis_fix": True}
    raise SystemExit(
        "REFUSING to read OSP results through this `rl_controller.iclr.pareto`: "
        "`Bundle()` constructs without a declared `normal_form`, which means WS-A's basis fix "
        "(v1.7 §5.10) is not present in this checkout. Policy-replay rows are CHAIN-SUMMED by "
        "construction, so an undeclared bundle labels them question-normal and every OSP "
        "statistic comes out mislabelled -- silently. Merge integration once sync-11 has landed "
        "the fix, then re-read.")


def question_summary(rows, *, allow_partial: bool = False) -> dict:
    missing = [r for r in rows if r["missing_reason"] is not None]
    if missing and not allow_partial:
        raise ValueError(
            f"{len(missing)} of {len(rows)} questions are unavailable (e.g. "
            f"{missing[0]['qkey']!r}): this config is UNAVAILABLE on this pool, not a partial "
            "frontier point (contract §2.2 rule 7).")
    usable = [r for r in rows if r["missing_reason"] is None]
    n = len(usable)
    if not n:
        raise ValueError("no usable questions in the replay rows")
    return {
        "basis": "question",
        "n_questions": n,
        "n_missing_questions": len(rows) - n,
        "accuracy": sum(r["e_correct"] for r in usable) / n,
        "mean_thinking_tokens": sum(r["e_thinking_tokens"] for r in usable) / n,
        "mean_generated_tokens": sum(r["e_generated_tokens"] for r in usable) / n,
        "mean_chains_per_question": sum(r["n_chains"] for r in usable) / n,
    }


def mandatory_captions(*, arm: str, faithful: bool, n_lambda: int, n_seeds: int,
                       reader: dict | None = None) -> dict:
    if reader is None:
        capacity = ("UNVERIFIED diagnostic: no trained-lane summary was supplied; capacity "
                    "and per-boundary compute may not be reported from this artifact.")
        overhead = ("aux_generated_tokens = 0; reader compute is UNVERIFIED because no "
                    "trained-lane summary was supplied.")
    else:
        for key in ("n_trainable_params", "n_unfrozen", "n_layers"):
            if not isinstance(reader.get(key), int) or reader[key] <= 0:
                raise ValueError(f"reader.{key} must be a positive integer")
        trainable = f"{reader['n_trainable_params']:,}"
        capacity = (
            f"Trainable parameters {trainable} ({reader['n_unfrozen']} unfrozen backbone "
            "blocks + head) against UHead's ~1.5e7. The generator stays frozen and scoring "
            "is offline, so this is a reader-capacity ablation, not an unfrozen-LLM "
            "controller.")
        overhead = (
            "aux_generated_tokens = 0: a hidden-state head generates nothing at decision "
            f"time. The extra top-{reader['n_unfrozen']} forward per boundary "
            f"({reader['n_unfrozen']}/{reader['n_layers']} of a full-block pass) is disclosed "
            "as FLOPs, not folded into the token axis.")
    return {
        "reporting_basis": (
            "Primary numbers are QUESTION-basis (unweighted mean over questions); the "
            "chain-weighted figure is reported beside them as the appendix view. The two "
            "coincide only where every question contributes equally many chains."),
        "faithful_invariance": (
            "The faithful (1-chain-per-question) arm's number is invariant to the choice of "
            "reporting basis, so any question-vs-chain gap in the head-to-head is "
            "attributable to the all-chains arm alone."),
        "determinization": (
            "Replay uses the EXACT survival-product expectation, a favorable determinization "
            "of OS-Pruner's Bernoulli stopping rule; the seeded Bernoulli variant is reported "
            "as the sanity arm."),
        "knob_difference": (
            f"One operating point per (lambda, seed) -- {n_lambda} lambda x {n_seeds} seeds -- "
            "versus 13-52 swept members for our own heads. The curves are not equal-cost and "
            "sweep-hull statistics are captioned with member counts."),
        "capacity": capacity,
        "overhead": overhead,
        "lambda_grid": (
            "The lambda grid is the amended one, derived from the pre-registered "
            "lambda*mean_l_nat*start_mult in [0.01, 3.0] bound at the measured mean_l_nat; "
            "values above the bound are reported as the degeneracy ablation, not dropped."),
        "arm": arm,
        "faithful_subsample": bool(faithful),
    }


def replay_rows(chains, *, header: dict, pi_key: str = "pi", grade: str = "loose",
                natural_answer_tokens=osp.natural_answer_tokens_archived_math,
                aux_generated_tokens: float = 0.0) -> list[dict]:
    rows = osp.replay_question_rows(
        chains, header=header, pi_key=pi_key, grade=grade,
        expectation="survival_product", natural_answer_tokens=natural_answer_tokens,
        aux_generated_tokens=aux_generated_tokens)
    return validated_for_consumer(rows)


def unavailable_rows(qkeys, *, header: dict, reason: str) -> list[dict]:
    return validated_for_consumer([osp.unavailable_row(qkey, header, reason)
                                   for qkey in qkeys])


def faithful_rep(qkey: str, reps, *, salt: str = FAITHFUL_SALT):
    reps = list(reps)
    if not reps:
        raise ValueError(f"question {qkey!r} has no chains")

    def key(rep):
        digest = hashlib.blake2b(f"{salt}:{qkey}:{rep}".encode("utf-8"), digest_size=8).digest()
        return (int.from_bytes(digest, "big"), rep)

    return min(reps, key=key)


def faithful_subsample(chains, *, salt: str = FAITHFUL_SALT) -> list[dict]:
    by_q: dict[str, list] = {}
    for chain in chains:
        by_q.setdefault(chain["qkey"], []).append(chain)
    keep = set()
    for qkey, group in by_q.items():
        rep = faithful_rep(qkey, [c["rep"] for c in group], salt=salt)
        keep.add((qkey, rep))
    return [c for c in chains if (c["qkey"], c["rep"]) in keep]


def bernoulli_fires(qkey: str, rep, pis, *, seed: int) -> list[int]:
    out = []
    for index, pi in enumerate(pis):
        if not 0.0 <= pi <= 1.0:
            raise ValueError(f"chain {(qkey, rep)} boundary {index}: pi {pi!r} is not a "
                             "probability")
        payload = f"{osp.REPLAY_SCHEMA}:{seed}:{qkey}:{rep}:{index}".encode("utf-8")
        draw = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") / 2 ** 64
        out.append(int(draw < pi))
    return out


def bernoulli_scores(chain: dict, *, seed: int, pi_key: str = "pi") -> list[dict]:
    pis = osp.chain_pis(chain, pi_key)
    fires = bernoulli_fires(chain["qkey"], chain["rep"], pis, seed=seed)
    return [{"q": float(f), "v": STRENGTHENED_V} for f in fires]


def strengthened_scores(chain: dict, *, pi_key: str = "pi") -> list[dict]:
    pis = osp.chain_pis(chain, pi_key)
    return [{"q": float(pi), "v": STRENGTHENED_V} for pi in pis]


def apply_scores(chains, score_fn) -> list[dict]:
    out = []
    for chain in chains:
        scores = score_fn(chain)
        if len(scores) != len(chain["probes"]):
            raise ValueError(f"chain {(chain['qkey'], chain['rep'])}: {len(scores)} scores for "
                             f"{len(chain['probes'])} probes")
        out.append({**chain, "scores": scores})
    return out


def arm_config(arm: str, *, lam: float, seed: int, pi_key: str = "pi", **extra) -> dict:
    if arm not in ("strengthened", "bernoulli"):
        raise ValueError(f"unknown arm {arm!r}")
    base = {
        "method": "OSP-pi-threshold" if arm == "strengthened" else "OSP-bernoulli",
        "knob": {"name": "lambda", "value": lam},
        "osp_arm": arm,
        "osp_lambda": lam,
        "q_is": "stop probability pi" if arm == "strengthened" else "seeded Bernoulli fire (0/1)",
        "v_is": f"constant {STRENGTHENED_V} placeholder — margin rules (L4) are NOT meaningful",
        "readable_rules": ["L1"],
        "pi_key": pi_key,
        "expectation": "survival_product" if arm == "strengthened" else "bernoulli_seeded",
        "overhead_accounting": "none",
        "aux_generated_tokens": 0.0,
    }
    if arm == "bernoulli":
        base["bernoulli_seed"] = seed
        base["bernoulli_threshold"] = BERNOULLI_THRESHOLD
        base["bernoulli_byte_order"] = "big"
    return {**base, **extra}
