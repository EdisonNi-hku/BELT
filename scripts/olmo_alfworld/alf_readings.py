from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

import numpy as np

TOLERANCES = (0.0, 2.0, 5.0)
BUDGETS = (0.20, 0.30, 0.40, 0.50, 0.60)


HORIZON_FILES = ("episodes.h30.jsonl",)

OUTCOME = "won"


def outcome(e: dict) -> float:
    return float(e["won"]) if OUTCOME == "won" else e["score"] / 100.0


def load(pattern: str, seen: dict = None) -> dict:
    out = {}
    seen = {} if seen is None else seen
    for f in sorted(glob.glob(pattern)):
        if Path(f).name in HORIZON_FILES:
            continue
        for line in open(f):
            e = json.loads(line)
            k = (e["member"], e["game"], e["seed"])
            sig = (e["won"], e["steps"], e["cost"]) + ((e["score"],) if OUTCOME == "score" else ())
            if k in seen and seen[k][0] != sig:
                raise SystemExit(f"episode {k} differs between {seen[k][1]} {seen[k][0]} and {f} {sig}")
            seen.setdefault(k, (sig, f))
            out.setdefault(e["member"], {}).setdefault((e["game"], e["seed"]), e)
    return out


def point(eps: dict, keys) -> tuple:
    return (float(np.mean([eps[k]["cost"] for k in keys])), float(np.mean([outcome(eps[k]) for k in keys])))


def frontier(points: list) -> list:
    pts = sorted(set(points), key=lambda p: (p[0], -p[1]))
    out = []
    for c, s in pts:
        if not out or s > out[-1][1]:
            out.append((c, s))
    return out


def cheapest_within(front: list, target: float):
    if front[0][1] >= target:
        return front[0][0]
    for (c0, s0), (c1, s1) in zip(front, front[1:]):
        if s1 >= target:
            return c0 + (target - s0) / (s1 - s0) * (c1 - c0)
    return None


def success_at(front: list, cost: float):
    if cost < front[0][0]:
        return None
    for (c0, s0), (c1, s1) in zip(front, front[1:]):
        if c0 <= cost <= c1:
            return s0 + (cost - c0) / (c1 - c0) * (s1 - s0) if c1 > c0 else max(s0, s1)
    return front[-1][1]


def readings(eps: dict, members: list, keys) -> dict:
    nat = point(eps["natural"], keys)
    front = frontier([point(eps[m], keys) for m in members] + [nat])
    out = {"natural_cost": nat[0], "natural_success": nat[1], "frontier": front}
    for d in TOLERANCES:
        c = cheapest_within(front, nat[1] - d / 100.0)
        out[f"saving_at_tol_{d:g}pp"] = None if c is None else 1.0 - c / nat[0]
    for b in BUDGETS:
        s = success_at(front, (1.0 - b) * nat[0])
        out[f"success_at_saving_{round(b * 100)}"] = s
    return out


READING_KEYS = [f"saving_at_tol_{d:g}pp" for d in TOLERANCES] + [f"success_at_saving_{round(b * 100)}" for b in BUDGETS]


def set_budgets(budgets) -> None:
    global BUDGETS, READING_KEYS
    BUDGETS = tuple(budgets)
    READING_KEYS = [f"saving_at_tol_{d:g}pp" for d in TOLERANCES] + [f"success_at_saving_{round(b * 100)}" for b in BUDGETS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, action="append", help="glob(s) of episodes*.jsonl")
    ap.add_argument("--methods", required=True)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--pairs", default="H-T:natural,H-T:OSP-1,H-T:OSP-full")
    ap.add_argument("--outcome", choices=("won", "score"), default="won")
    ap.add_argument("--budgets", default=None, help="comma list of savings, e.g. 0.2,0.3,...,0.8 (default BUDGETS)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global OUTCOME
    OUTCOME = a.outcome
    if a.budgets:
        set_budgets(float(b) for b in a.budgets.split(","))
    eps, seen = {}, {}
    for pat in a.episodes:
        for m, d in load(pat, seen).items():
            for k, e in d.items():
                eps.setdefault(m, {}).setdefault(k, e)
    methods = json.loads(Path(a.methods).read_text())
    keys = sorted(eps["natural"])
    missing = {m: len(set(keys) - set(eps.get(m, {}))) for fam in methods.values() for ms in fam.values() for m in ms}
    if any(missing.values()):
        raise SystemExit(f"members missing episodes: { {m: n for m, n in missing.items() if n} }")
    rep = {"outcome": OUTCOME, "episodes_per_member": len(keys), "episode_files": sorted({f for _, f in seen.values()}),
           "members": {}, "methods": {}}
    for m, d in eps.items():
        if all(k in d for k in keys):
            c, s = point(d, keys)
            rep["members"][m] = {"success": s, "cost": c, "saving": 1.0 - c / point(eps["natural"], keys)[0],
                                 "thinking": float(np.mean([d[k]["thinking"] for k in keys if "thinking" in d[k]] or [0]))}
    for fam, runs in methods.items():
        per_run = {run: readings(eps, ms, keys) for run, ms in runs.items()}
        agg = {}
        for rk in READING_KEYS:
            vals = [r[rk] for r in per_run.values()]
            ok = [v for v in vals if v is not None]
            agg[rk] = {"mean": statistics.mean(ok) if len(ok) == len(vals) else None,
                       "sd": statistics.stdev(ok) if len(ok) == len(vals) and len(ok) > 1 else None,
                       "runs": vals}
        rep["methods"][fam] = {"runs": per_run, "aggregate": agg}
    rng = np.random.default_rng(0)
    games = sorted({g for g, _ in keys})
    by_game = {g: [k for k in keys if k[0] == g] for g in games}
    draws = [[k for g in rng.choice(games, size=len(games), replace=True) for k in by_game[g]] for _ in range(a.boot)]

    def mean_reading(name, ks):
        runs = {"natural": []} if name == "natural" else methods[name]
        vals = {rk: [] for rk in READING_KEYS}
        for ms in runs.values():
            r = readings(eps, ms, ks)
            for rk in READING_KEYS:
                vals[rk].append(r[rk])
        return {rk: (None if any(v is None for v in vs) else float(np.mean(vs))) for rk, vs in vals.items()}

    boot = {}
    for pair in filter(None, a.pairs.split(",")):
        x, y = pair.split(":")
        if (x != "natural" and x not in methods) or (y != "natural" and y not in methods):
            continue
        point_x, point_y = mean_reading(x, keys), mean_reading(y, keys)
        diffs = {rk: [] for rk in READING_KEYS}
        for ks in draws:
            rx, ry = mean_reading(x, ks), mean_reading(y, ks)
            for rk in READING_KEYS:
                if rx[rk] is not None and ry[rk] is not None:
                    diffs[rk].append(rx[rk] - ry[rk])
        boot[pair] = {rk: {"diff": (None if point_x[rk] is None or point_y[rk] is None else point_x[rk] - point_y[rk]),
                           "n": len(v), "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
                           if v else None} for rk, v in diffs.items()}
    rep["paired_bootstrap"] = boot
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({f: {rk: v["mean"] for rk, v in r["aggregate"].items()} for f, r in rep["methods"].items()},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
