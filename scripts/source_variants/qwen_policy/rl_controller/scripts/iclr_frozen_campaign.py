"""Fit a frozen-input stopping policy on one pool and seed and evaluate it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rl_controller.iclr import learned_linear_policy as ll
from rl_controller.iclr import learned_linear_mlp as mlp
from rl_controller.iclr import learned_linear_frozen as fz
from rl_controller.scripts.iclr_score_pool import ReprDump

ARMS = ("LB_QVRS", "LB_QVRS_MLP", "LB_H", "LB_L16")
BELIEF_COLUMNS = ("q", "v", "r", "s", "d")
REPORT_PRICES = (2e-6, 5e-6, 1e-5, 2e-5, 5e-5)


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_chains(chains, n_fit_questions=None):
    fit_q, dev_q = ll.split_questions([c["qkey"] for c in chains])
    if n_fit_questions is not None:
        fit_q = fit_q[:n_fit_questions]
    fit_set, dev_set = set(fit_q), set(dev_q)
    return ([c for c in chains if c["qkey"] in fit_set],
            [c for c in chains if c["qkey"] in dev_set])


def build_buckets(arm, chains, dump, mean, std):
    if arm in ("LB_QVRS", "LB_QVRS_MLP"):
        return ll.to_torch(ll.pack(chains, mean, std), BELIEF_COLUMNS)
    return fz.to_torch_frozen(fz.pack_frozen(chains, dump, mean, std))


def standardize(arm, fit_chains, dump):
    if arm in ("LB_QVRS", "LB_QVRS_MLP"):
        return ll.standardizer(fit_chains)
    return fz.standardizer_frozen(fit_chains, dump)


def natural_point(buckets):
    total_w = sum(float(t["weight"].sum()) for t in buckets)
    acc = sum(float((t["weight"] * t["a_nat"]).sum()) for t in buckets)
    think = sum(float((t["weight"] * t["think_nat"]).sum()) for t in buckets)
    return {"accuracy": acc, "thinking_tokens": think, "weight": total_w}


def point_of(arm, params, buckets):
    if arm == "LB_QVRS_MLP":
        return {"accuracy": _expect(buckets, params, "a", "a_nat"),
                "thinking_tokens": _expect(buckets, params, "think", "think_nat")}
    import torch
    w = torch.tensor(params["w"], dtype=torch.float64)
    b = torch.tensor(params["b"], dtype=torch.float64)
    return {"accuracy": ll.expect(buckets, w, b, "a", "a_nat"),
            "thinking_tokens": ll.expect(buckets, w, b, "think", "think_nat")}


def utility_at(arm, params, buckets, lam):
    import torch
    from rl_controller.iclr import osp_policy as osp
    if arm == "LB_QVRS_MLP":
        return float(sum((t["weight"] * osp.expected_utility(
            mlp.mlp_logits(t, params), t["a"], t["c"], t["a_nat"], t["c_nat"], lam)).sum()
            for t in buckets))
    w = torch.tensor(params["w"], dtype=torch.float64)
    b = torch.tensor(params["b"], dtype=torch.float64)
    return float(ll.utility(buckets, w, b, lam))


def read_pool(arm, params, buckets, lam):
    import torch
    if arm == "LB_QVRS_MLP":
        acc = _expect(buckets, params, "a", "a_nat")
        total = _expect(buckets, params, "c", "c_nat")
        think = _expect(buckets, params, "think", "think_nat")
        p_nat = _expect(buckets, params, 0.0, 1.0)
        return dict(accuracy=acc, total_tokens=total, utility=acc - lam * total,
                    thinking_tokens=think, p_natural=p_nat)
    w = torch.tensor(params["w"], dtype=torch.float64)
    b = torch.tensor(params["b"], dtype=torch.float64)
    return ll.stochastic_summary(buckets, w, b, lam)


def _expect(buckets, params, value, natural):
    from rl_controller.iclr import osp_policy as osp
    total = 0.0
    for t in buckets:
        z = mlp.mlp_logits(t, params)
        total += float((t["weight"] * osp.expected_utility(
            z, t[value] if isinstance(value, str) else value, 0.0,
            t[natural] if isinstance(natural, str) else natural, 0.0, 0.0)).sum())
    return total


SHARD_INVARIANTS = ("schema", "arm", "pool", "seed", "opt_seed", "n_fit_questions",
                    "n_features", "n_fit_chains", "n_dev_chains", "n_pool_chains",
                    "scored_run", "report_prices", "natural")


def merge_shards(paths, out):
    shards = []
    for path in paths:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        shards.append((str(path), doc))
    if not shards:
        raise SystemExit("--merge needs at least one shard")

    ref_path, ref = shards[0]
    for path, doc in shards[1:]:
        bad = [k for k in SHARD_INVARIANTS if doc.get(k) != ref.get(k)]
        if bad:
            raise SystemExit(
                f"{path} disagrees with {ref_path} on {bad}. These are not shards of one cell; "
                "merging them would report a cell that was never fitted as a whole.")

    rows, seen = [], {}
    for path, doc in shards:
        for r in doc["rows"]:
            lam = r["lam"]
            if lam in seen:
                raise SystemExit(
                    f"price {lam:g} appears in both {seen[lam]} and {path}: two fits claim the "
                    "same row and there is no rule for choosing between them.")
            seen[lam] = path
            rows.append(r)
    rows.sort(key=lambda r: r["lam"])

    doc = {k: ref[k] for k in ref if k not in ("rows", "lambda_indices")}
    doc["rows"] = rows
    doc["merged_from"] = [p for p, _ in shards]
    missing = [f"{lam:g}" for lam in ll.LAMBDAS if lam not in seen]
    if missing:
        doc["incomplete_prices"] = missing
    Path(out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"merged": len(shards), "prices": len(rows),
                      "incomplete_prices": missing or None, "out": str(out)}))
    if missing:
        print(f"WARNING: {len(missing)} of {len(ll.LAMBDAS)} prices are missing: {missing}")
    return 0


def align_threads():
    import os
    n = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("OMP_NUM_THREADS")
    if not n:
        return None
    n = max(1, int(n))
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    import torch
    torch.set_num_threads(n)
    return n


KAPPAS_FROZEN = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
KAPPA_EXTEND_MAX = 1e4
DEV_EVAL_EVERY = 25
MAX_LBFGS_ITER = 500


def fit_dev_stopped(fit_b, dev_b, n_features, lam, kappa, init_stop):
    import torch

    w = torch.zeros(n_features, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(ll.logit(init_stop), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], **{**ll.LBFGS, "max_iter": DEV_EVAL_EVERY})

    def closure():
        opt.zero_grad()
        loss = -ll.utility(fit_b, w, b, lam) + kappa * (w * w).sum()
        loss.backward()
        return loss

    best, steps, last_n_iter = None, 0, 0
    while steps < MAX_LBFGS_ITER:
        opt.step(closure)
        steps += DEV_EVAL_EVERY
        with torch.no_grad():
            wd, bd = w.detach().clone(), b.detach().clone()
            dev = float(ll.utility(dev_b, wd, bd, lam))
            fit_u = float(ll.utility(fit_b, wd, bd, lam))
        if best is None or dev > best["dev_utility"]:
            best = dict(init_stop=init_stop, kappa=kappa, dev_utility=dev,
                        w=[float(x) for x in wd], b=float(bd), objective=fit_u,
                        selected_step=steps)
        n_iter = int(opt.state[opt._params[0]].get("n_iter", 0))
        if n_iter - last_n_iter < DEV_EVAL_EVERY:
            break
        last_n_iter = n_iter
    best["steps_run"] = steps
    return best


def fit_linear_cell(fit_b, dev_b, n_features, lam):
    grid = list(KAPPAS_FROZEN)
    while True:
        cand = []
        for kappa in grid:
            for init in ll.INIT_STOP:
                cand.append(fit_dev_stopped(fit_b, dev_b, n_features, lam, kappa, init))
        best = max(cand, key=lambda f: (f["dev_utility"], f["kappa"]))
        if best["kappa"] < grid[-1] or grid[-1] >= KAPPA_EXTEND_MAX:
            best["kappa_grid"] = list(grid)
            return best
        grid.append(grid[-1] * 10.0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--scored-run", required=True,
                   help="reader scored dir: val_scored.jsonl fits, select_scored.jsonl is read")
    p.add_argument("--val-dump", default=None,
                   help="LB_H/LB_L16: dump dir for the VAL chains (the fit side)")
    p.add_argument("--pool-dump", default=None,
                   help="LB_H/LB_L16: dump dir for the select chains (the read side)")
    p.add_argument("--pool", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-fit-questions", type=int, default=None,
                   help="learning curve: 40 or 80, nested inside the registered 160")
    p.add_argument("--init-stop", type=float, default=0.1)
    p.add_argument("--opt-seed", type=int, default=0,
                   help="MLP only: the optimiser seed, for the multi-init check")
    p.add_argument("--out", required=True)
    p.add_argument("--lambda-indices", default=None,
                   help="steer 33.2: comma-separated indices into learned_linear_policy.LAMBDAS "
                        "(0..16); default all 17. One index per task is the array shard, and the "
                        "shards merge with --merge. The fit at one price does not depend on any "
                        "other price -- each is its own L-BFGS from its own init -- so sharding "
                        "by price changes nothing about the numbers, only the wall clock.")
    p.add_argument("--merge", nargs="+", default=None,
                   help="merge shard cell JSONs (written by --lambda-indices) into one cell JSON "
                        "at --out. Every field except `rows` must agree across shards.")
    args = p.parse_args(argv)

    if args.merge:
        return merge_shards(args.merge, args.out)

    threads = align_threads()

    if args.lambda_indices is None:
        wanted = set(range(len(ll.LAMBDAS)))
    else:
        wanted = {int(x) for x in args.lambda_indices.split(",") if x.strip() != ""}
        bad = sorted(i for i in wanted if not 0 <= i < len(ll.LAMBDAS))
        if bad:
            raise SystemExit(f"--lambda-indices {bad} outside 0..{len(ll.LAMBDAS) - 1}")
        if not wanted:
            raise SystemExit("--lambda-indices selected no price")

    frozen_arm = args.arm in ("LB_H", "LB_L16")
    if frozen_arm and not (args.val_dump and args.pool_dump):
        raise SystemExit(f"{args.arm} needs --val-dump and --pool-dump: the fit lives on the val "
                         "split, and a pool-side dump alone cannot fit anything")

    run = Path(args.scored_run)
    val_chains = load_jsonl(run / "val_scored.jsonl")
    pool_chains = load_jsonl(run / "select_scored.jsonl")
    val_dump = pool_dump = None
    if frozen_arm:
        val_dump, _ = ReprDump.load(args.val_dump, args.arm)
        pool_dump, _ = ReprDump.load(args.pool_dump, args.arm)

    fit_chains, dev_chains = split_chains(val_chains, args.n_fit_questions)
    mean, std = standardize(args.arm, fit_chains, val_dump)
    fit_b = build_buckets(args.arm, fit_chains, val_dump, mean, std)
    dev_b = build_buckets(args.arm, dev_chains, val_dump, mean, std)
    pool_b = build_buckets(args.arm, pool_chains, pool_dump, mean, std)
    n_features = int(fit_b[0]["x"].shape[-1])

    rows = []
    for lam_index, lam in enumerate(ll.LAMBDAS):
        if lam_index not in wanted:
            continue
        lam_started = time.monotonic()
        if args.arm == "LB_QVRS_MLP":
            best = None
            for lr in mlp.LEARNING_RATES:
                out = mlp.fit_mlp(fit_b, dev_b, n_features, lam, lr, args.init_stop,
                                  seed=args.opt_seed)
                if best is None or out["dev_utility"] > best["dev_utility"]:
                    best = out
            import torch
            params = [torch.tensor(best[k], dtype=torch.float64)
                      for k in ("w1", "b1", "w2", "b2")]
            point = read_pool(args.arm, params, pool_b, lam)
            at = {f"{lr_:g}": {"dev": utility_at(args.arm, params, dev_b, lr_),
                               "pool": utility_at(args.arm, params, pool_b, lr_)}
                  for lr_ in REPORT_PRICES}
            dev_point = point_of(args.arm, params, dev_b)
            rows.append({**{k: best[k] for k in ("lr", "dev_utility", "fit_utility",
                                                 "selected_step", "steps_run")},
                         "lam": lam, "lambda_index": lam_index, "pool": point,
                         "at_report_prices": at, "dev_point": dev_point,
                         "seconds": round(time.monotonic() - lam_started, 1),
                         "fit_minus_dev": best["fit_utility"] - best["dev_utility"]})
        else:
            best = fit_linear_cell(fit_b, dev_b, n_features, lam)
            point = read_pool(args.arm, best, pool_b, lam)
            at = {f"{pr:g}": {"dev": utility_at(args.arm, best, dev_b, pr),
                              "pool": utility_at(args.arm, best, pool_b, pr)}
                  for pr in REPORT_PRICES}
            dev_point = point_of(args.arm, best, dev_b)
            rows.append({"lam": lam, "lambda_index": lam_index,
                         "kappa": best["kappa"], "init_stop": best["init_stop"],
                         "kappa_grid": best["kappa_grid"],
                         "selected_step": best["selected_step"],
                         "steps_run": best["steps_run"],
                         "seconds": round(time.monotonic() - lam_started, 1),
                         "at_report_prices": at, "dev_point": dev_point,
                         "dev_utility": best["dev_utility"],
                         "fit_utility": best["objective"],
                         "fit_minus_dev": best["objective"] - best["dev_utility"],
                         "w": best["w"], "b": best["b"], "pool": point})

    doc = {"schema": "iclr-frozen-cell/v1", "arm": args.arm, "pool": args.pool,
           "seed": args.seed, "opt_seed": args.opt_seed,
           "n_fit_questions": args.n_fit_questions or len(
               {c["qkey"] for c in fit_chains}),
           "n_features": n_features,
           "n_fit_chains": len(fit_chains), "n_dev_chains": len(dev_chains),
           "n_pool_chains": len(pool_chains),
           "scored_run": str(run), "report_prices": list(REPORT_PRICES),
           "natural": {"dev": natural_point(dev_b), "pool": natural_point(pool_b)},
           "rows": rows}
    if args.lambda_indices is not None:
        doc["lambda_indices"] = sorted(wanted)
    if threads:
        doc["threads"] = threads
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    for r in rows:
        if r["lam"] in REPORT_PRICES:
            print(json.dumps({"arm": args.arm, "pool": args.pool, "seed": args.seed,
                              "lam": r["lam"], "dev_U": round(r["dev_utility"], 6),
                              "fit_minus_dev": round(r["fit_minus_dev"], 6),
                              "pool_U": round(r["pool"]["utility"], 6)}), flush=True)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
