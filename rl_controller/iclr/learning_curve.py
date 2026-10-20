from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence

VAL_SALT = "iclr-lc-val-2026-08-26"
SIZES = (32, 64, 128, 255)
N_VAL = 64
CHAIN_CAPS = (4, 8, 16)
GATE_MAX_DACC_PP = 1.0
GATE_MAX_DTOK_FRAC = 0.03
MARGIN_GRID = tuple(round(-0.30 + 0.05 * i, 2) for i in range(13))


def question_rank(qkey: str, salt: str) -> float:
    h = hashlib.blake2b(f"{salt}:{qkey}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64


def partition(qkeys: Iterable[str], *, n_val: int, salt: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    keys = list(qkeys)
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate question keys")
    if not isinstance(n_val, int) or n_val < 1:
        raise ValueError(f"n_val must be a positive int, got {n_val!r}")
    if n_val >= len(keys):
        raise ValueError(f"n_val={n_val} leaves no train questions out of {len(keys)}")
    ordered = sorted(keys, key=lambda k: (question_rank(k, salt), k))
    return tuple(ordered[:n_val]), tuple(ordered[n_val:])


def nested_subset(train_order: Sequence[str], n: int) -> tuple[str, ...]:
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be a non-negative int, got {n!r}")
    if n > len(train_order):
        raise ValueError(f"asked for {n} train questions, pool has {len(train_order)}")
    return tuple(train_order) if n == 0 else tuple(train_order[:n])


def cap_chains(chains: Sequence[dict], max_per_question: int) -> list[dict]:
    if not isinstance(max_per_question, int) or max_per_question < 0:
        raise ValueError(f"max_per_question must be a non-negative int, got {max_per_question!r}")
    for c in chains:
        if type(c["rep"]) is not int:
            raise ValueError(f"rep must be an int, got {c['rep']!r} for {c['qkey']}")
    ordered = sorted(chains, key=lambda c: (c["qkey"], c["rep"]))
    if max_per_question == 0:
        return ordered
    kept: list[dict] = []
    seen: dict[str, int] = {}
    for c in ordered:
        k = seen.get(c["qkey"], 0)
        if k < max_per_question:
            kept.append(c)
            seen[c["qkey"]] = k + 1
    return kept


def bce(p: float, y: float, eps: float = 1e-7) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def choose_epoch(history: Sequence[dict], *, key: str, minimise: bool, patience: int) -> dict:
    if not isinstance(patience, int) or patience < 1:
        raise ValueError(f"patience must be a positive int, got {patience!r}")
    if not history:
        raise ValueError("empty history")
    best_epoch, best_value, stopped_at = None, None, None
    prev_epoch = None
    for row in history:
        epoch = row["epoch"]
        if prev_epoch is not None and epoch <= prev_epoch:
            raise ValueError(
                f"history epochs must be strictly ascending, got {prev_epoch} then {epoch}"
            )
        prev_epoch = epoch
        v = row[key]
        if not math.isfinite(v):
            raise ValueError(
                f"non-finite {key}={v!r} at epoch {epoch} -- a NaN/inf statistic is a "
                "training failure and must surface, not be silently skipped"
            )
        better = best_value is None or (v < best_value if minimise else v > best_value)
        if better:
            best_epoch, best_value = epoch, v
        elif epoch - best_epoch >= patience:
            stopped_at = epoch
            break
    return {"epoch": best_epoch, "value": best_value, "stopped_at": stopped_at}


def upper_hull(rows: Iterable[dict]) -> list[tuple[float, float]]:
    pts = sorted({(float(r["mean_thinking_tokens"]), float(r["accuracy"])) for r in rows})
    pareto: list[tuple[float, float]] = []
    for t, a in pts:
        if pareto and t == pareto[-1][0]:
            pareto[-1] = (t, a)
            continue
        if pareto and a <= pareto[-1][1]:
            continue
        pareto.append((t, a))
    hull: list[tuple[float, float]] = []
    for p in pareto:
        while len(hull) >= 2:
            (t0, a0), (t1, a1) = hull[-2], hull[-1]
            cross = (t1 - t0) * (p[1] - a0) - (a1 - a0) * (p[0] - t0)
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def anchor_hull(rule_rows, natural: dict, always: dict) -> list[tuple[float, float]]:
    hull = upper_hull([*rule_rows, natural, always])
    if hull[-1][0] < natural["mean_thinking_tokens"]:
        hull.append((float(natural["mean_thinking_tokens"]), hull[-1][1]))
    return hull


def question_folds(qkeys, *, n_folds: int, salt: str) -> dict[str, int]:
    if not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds!r}")
    keys = list(qkeys)
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate question keys")
    ordered = sorted(keys, key=lambda k: (question_rank(k, salt), k))
    return {k: i % n_folds for i, k in enumerate(ordered)}


def acc_at_tokens(hull: Sequence[tuple[float, float]], tokens: float) -> float | None:
    if not hull or tokens < hull[0][0] or tokens > hull[-1][0]:
        return None
    for (t0, a0), (t1, a1) in zip(hull, hull[1:]):
        if t0 <= tokens <= t1:
            return a0 if t1 == t0 else a0 + (a1 - a0) * (tokens - t0) / (t1 - t0)
    return hull[-1][1]


def tokens_at_acc(hull: Sequence[tuple[float, float]], acc: float) -> float | None:
    if not hull or acc > hull[-1][1]:
        return None
    if acc <= hull[0][1]:
        return hull[0][0]
    for (t0, a0), (t1, a1) in zip(hull, hull[1:]):
        if a0 <= acc <= a1:
            if a1 == a0 or acc == a0:
                return t0
            if acc == a1:
                return t1
            return t0 + (t1 - t0) * (acc - a0) / (a1 - a0)
    return hull[-1][0]


def matched_delta(small: Sequence[tuple[float, float]], big: Sequence[tuple[float, float]]) -> dict:
    out = {"dacc_pp": None, "dtok_frac": None, "token_range": None, "acc_range": None}
    if not small or not big:
        return out
    t_lo, t_hi = max(small[0][0], big[0][0]), min(small[-1][0], big[-1][0])
    if t_hi - t_lo > 0:
        token_points = sorted({t for t, _ in (*small, *big) if t_lo <= t <= t_hi} | {t_lo, t_hi})
        diffs = []
        for t in token_points:
            a_small, a_big = acc_at_tokens(small, t), acc_at_tokens(big, t)
            assert a_small is not None and a_big is not None, (
                f"acc_at_tokens returned None for t={t} inside overlap [{t_lo}, {t_hi}]")
            diffs.append(a_big - a_small)
        out["dacc_pp"] = 100.0 * max(diffs)
        out["token_range"] = [t_lo, t_hi]
    a_lo, a_hi = max(small[0][1], big[0][1]), min(small[-1][1], big[-1][1])
    if a_hi - a_lo > 0:
        acc_points = sorted({a for _, a in (*small, *big) if a_lo <= a <= a_hi} | {a_lo, a_hi})
        fracs = []
        for a in acc_points:
            ts, tb = tokens_at_acc(small, a), tokens_at_acc(big, a)
            assert ts is not None and tb is not None, (
                f"tokens_at_acc returned None for a={a} inside overlap [{a_lo}, {a_hi}]")
            assert ts > 0, f"tokens_at_acc(small, {a}) returned non-positive {ts}"
            fracs.append((ts - tb) / ts)
        out["dtok_frac"] = max(fracs)
        out["acc_range"] = [a_lo, a_hi]
    return out


def operating_point_delta(small: Sequence[tuple[float, float]], big: Sequence[tuple[float, float]],
                          natural: dict) -> dict:
    nat_tok, nat_acc = float(natural["mean_thinking_tokens"]), float(natural["accuracy"])
    t_star = tokens_at_acc(small, nat_acc)
    if t_star is not None:
        t_star = min(max(t_star, small[0][0]), small[-1][0])
    out = {"dacc_pp": None, "dsaving_frac": None, "tokens": t_star}
    if t_star is None:
        return out
    a_small, a_big = acc_at_tokens(small, t_star), acc_at_tokens(big, t_star)
    if a_small is not None and a_big is not None:
        out["dacc_pp"] = 100.0 * (a_big - a_small)
    t_big_at_nat = tokens_at_acc(big, nat_acc)
    if t_big_at_nat is not None and nat_tok > 0:
        out["dsaving_frac"] = (t_star - t_big_at_nat) / nat_tok
    return out


def matched_saving(hull: Sequence[tuple[float, float]], natural: dict) -> dict:
    nat_tok, nat_acc = float(natural["mean_thinking_tokens"]), float(natural["accuracy"])
    if nat_tok <= 0:
        raise ValueError("natural row has zero thinking tokens")
    out = {}
    for tag, target in (("saving_at_0pp", nat_acc), ("saving_at_1pp", nat_acc - 0.01)):
        t = tokens_at_acc(hull, target)
        out[tag] = None if t is None else 1.0 - t / nat_tok
    return out


_STEP_KEY_RE = re.compile(r"(\d+|all)->(\d+|all)")


def _step_size(token: str) -> float:
    return float("inf") if token == "all" else int(token)


def plateau_gate(deltas_by_step: dict, *, acc_key: str = "dacc_pp", tok_key: str = "dtok_frac") -> dict:
    if not deltas_by_step:
        raise ValueError("no doublings to gate on")
    for step in deltas_by_step:
        if not _STEP_KEY_RE.fullmatch(step):
            raise ValueError(f"bad step key {step!r}, expected 'N1->N2' (N may be 'all')")
    last = max(deltas_by_step, key=lambda s: _step_size(s.split("->")[1]))
    d = deltas_by_step[last]
    dacc, dtok = d[acc_key], d[tok_key]
    acc_fails = dacc is not None and dacc >= GATE_MAX_DACC_PP
    tok_fails = dtok is not None and dtok >= GATE_MAX_DTOK_FRAC
    if acc_fails or tok_fails:
        verdict = False
    elif dacc is None or dtok is None:
        verdict = None
    else:
        verdict = True
    return {"last_step": last, "dacc_pp": dacc, "dtok_frac": dtok, "pass": verdict,
            "max_dacc_pp": GATE_MAX_DACC_PP, "max_dtok_frac": GATE_MAX_DTOK_FRAC,
            "acc_key": acc_key, "tok_key": tok_key}
