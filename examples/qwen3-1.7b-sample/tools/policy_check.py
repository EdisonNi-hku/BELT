#!/usr/bin/env python3
"""Recover the fitted B-R / H-R policies' normalisation, verify it against the archived fit
cells, and export the policies plus per-probe reference values for the sample chains.

Run under scripts/belt_ext/d53/run_patched.py --producer <qwen_policy snapshot> --script ...
so that the natural-completion cost charges elicitation tokens exactly as the fit did.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from rl_controller.scripts import iclr_frozen_campaign as fc
from rl_controller.scripts.iclr_score_pool import ReprDump
from rl_controller.iclr import learned_linear_policy as ll
from rl_controller.iclr import osp_policy as osp

ARM_NAME = {"LB_QVRS": "BELT-B-R", "LB_H": "BELT-H-R"}
POINT_FIELDS = ("accuracy", "total_tokens", "utility", "thinking_tokens", "p_natural")
WIDTH = 768


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sample_chain(row):
    nat = row["natural"]
    c = {"qkey": row["qkey"], "rep": row["rep"], "gold": row["gold"], "outcome": row["outcome"],
         "answer": nat["answer"], "correct": nat["correct"], "blocks": row["blocks"],
         "thinking_tokens": row["thinking_tokens"], "block_ends": row["block_ends"],
         "query_points": row["query_points"], "probes": row["probes"], "scores": row["scores"]}
    if nat["elicited"]:
        c["natural_fallback_tokens"] = nat["elicitation_tokens"]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr-run", required=True)
    ap.add_argument("--val-dump", required=True)
    ap.add_argument("--pool-dump", required=True)
    ap.add_argument("--cell-b", required=True)
    ap.add_argument("--cell-h", required=True)
    ap.add_argument("--sample-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import d53lib
    setup = d53lib._STATE.get("setup")
    run = Path(a.corr_run)
    rec = {"setup": setup, "inputs": {}}
    for p in (run / "val_scored.jsonl", run / "select_scored.jsonl", a.cell_b, a.cell_h,
              Path(a.val_dump) / "LB_H.f16", Path(a.val_dump) / "LB_H.index.jsonl",
              Path(a.pool_dump) / "LB_H.f16", Path(a.pool_dump) / "LB_H.index.jsonl"):
        rec["inputs"][str(p)] = sha_file(p)

    val = fc.load_jsonl(run / "val_scored.jsonl")
    pool = fc.load_jsonl(run / "select_scored.jsonl")
    fit, dev = fc.split_chains(val, None)
    fit_q, dev_q = ll.split_questions([c["qkey"] for c in val])
    rec["split"] = {"n_val_chains": len(val), "n_fit_chains": len(fit), "n_dev_chains": len(dev),
                    "n_fit_questions": len(fit_q), "n_dev_questions": len(dev_q)}
    with open(out / "policy_fit_split.json", "w") as f:
        json.dump({"schema": "belt-sample-policy-split/v1", "salt": ll.SPLIT_SALT,
                   "rule": "sort the validation-bank question keys by sha256(salt + ':' + qkey) "
                           "hex digest; first half = policy fit, second half = development",
                   "fit_qkeys": list(fit_q), "dev_qkeys": list(dev_q)}, f, indent=1)

    val_dump, _ = ReprDump.load(a.val_dump, "LB_H")
    pool_dump, _ = ReprDump.load(a.pool_dump, "LB_H")

    sample_rows = [json.loads(l) for l in open(Path(a.sample_dir) / "chains.jsonl") if l.strip()]
    blob = (Path(a.sample_dir) / "belt_h.f16").read_bytes()

    policies, verify = {}, {}
    for arm, cell_path in (("LB_QVRS", a.cell_b), ("LB_H", a.cell_h)):
        cell = json.load(open(cell_path))
        assert cell["arm"] == arm and cell["seed"] == 0
        assert cell["n_fit_chains"] == len(fit) and cell["n_dev_chains"] == len(dev)
        assert cell["n_pool_chains"] == len(pool)
        dump_v = val_dump if arm == "LB_H" else None
        dump_p = pool_dump if arm == "LB_H" else None
        mean, std = fc.standardize(arm, fit, dump_v)
        pool_b = fc.build_buckets(arm, pool, dump_p, mean, std)
        dev_b = fc.build_buckets(arm, dev, dump_v, mean, std)
        worst_pool, worst_dev = 0.0, 0.0
        for row in cell["rows"]:
            params = {"w": row["w"], "b": row["b"]}
            got = fc.read_pool(arm, params, pool_b, row["lam"])
            worst_pool = max([worst_pool] + [abs(got[k] - row["pool"][k]) for k in POINT_FIELDS])
            dp = fc.point_of(arm, params, dev_b)
            worst_dev = max([worst_dev] + [abs(dp[k] - row["dev_point"][k]) for k in dp])
        verify[arm] = {"rows": len(cell["rows"]),
                       "max_abs_diff_pool_readout": worst_pool,
                       "max_abs_diff_dev_readout": worst_dev}
        print(json.dumps({arm: verify[arm]}), flush=True)

        doc = {
            "schema": "belt-sample-policy/v1",
            "method": ARM_NAME[arm], "producer_arm": arm,
            "checkpoint": {"repository": "JingweiNi/BELT-Qwen3-1.7B-ReProbeInit",
                           "revision": "664179ee7431db265c6fed041790d66381f1b1d6",
                           "run": "seed-0/s1_best.pt",
                           "sha256": "a14b5b5e527ce71ff9a90e167ae7251238da983dc5ed9e37f5e27d891d0829cd"},
            "features": (["logit(q)", "logit(v)", "logit(r)", "logit(stability)",
                          "prefix_len / 12000"] if arm == "LB_QVRS"
                         else [f"h[{i}]" for i in range(WIDTH)]),
            "logit_clip": ll.CLIP if arm == "LB_QVRS" else None,
            "length_scale": ll.BUDGET if arm == "LB_QVRS" else None,
            "hidden_dtype": "float16" if arm == "LB_H" else None,
            "mean": [float(x) for x in mean], "std": [float(x) for x in std],
            "decision": "z_i = w . ((x_i - mean) / std) + b at probe i; stop probability "
                        "pi_i = sigmoid(z_i); the run stops at probe i with probability "
                        "pi_i * prod_{j<i} (1 - pi_j) and otherwise reaches natural completion",
            "fit": {"split_salt": ll.SPLIT_SALT, "fit_questions": len(fit_q),
                    "dev_questions": len(dev_q), "objective": "survival_product_expected_utility",
                    "grade": "loose", "cost": "total", "optimizer": "float64 L-BFGS",
                    "lbfgs": ll.LBFGS, "init_stop": list(ll.INIT_STOP),
                    "selection": "development utility every 25 iterations"},
            "rows": [{"lambda_index": r["lambda_index"], "lambda": r["lam"], "kappa": r["kappa"],
                      "init_stop": r["init_stop"], "selected_step": r["selected_step"],
                      "w": r["w"], "b": r["b"]} for r in cell["rows"]],
        }
        policies[arm] = doc
        name = "belt_b_r_seed0.json" if arm == "LB_QVRS" else "belt_h_r_seed0.json"
        with open(out / name, "w") as f:
            json.dump(doc, f, indent=1)

    replay = []
    for row in sample_rows:
        chain = sample_chain(row)
        n = row["h"]["n_probes"]
        h = np.frombuffer(blob, dtype=np.float16, count=n * WIDTH,
                          offset=row["h"]["offset"]).reshape(n, WIDTH).astype(np.float64)
        xb = ll.probe_features(chain)
        strict = osp.chain_terms(chain, grade="strict", cost="total")
        loose = osp.chain_terms(chain, grade="loose", cost="total")
        entry = {"qkey": row["qkey"], "rep": row["rep"],
                 "probe_correct_strict": [int(x) for x in strict.a],
                 "probe_correct_loose": [int(x) for x in loose.a],
                 "probe_stop_cost_tokens": [int(x) for x in strict.l_total],
                 "natural_correct": int(strict.a_nat),
                 "natural_cost_tokens": int(strict.l_total_nat),
                 "b_features_raw": xb.tolist(), "policies": {}}
        for arm, doc in policies.items():
            x = xb if arm == "LB_QVRS" else h
            xs = (x - np.array(doc["mean"])) / np.array(doc["std"])
            per = []
            for r in doc["rows"]:
                z = xs @ np.array(r["w"], dtype=np.float64) + float(r["b"])
                pis = sigmoid(z)
                corr, tokens = osp.expected_outcome(list(pis), loose, total_cost=True)
                per.append({"lambda_index": r["lambda_index"], "logit": z.tolist(),
                            "expected_correct_loose": corr, "expected_cost_tokens": tokens})
            entry["policies"][ARM_NAME[arm]] = per
        replay.append(entry)
    with open(out / "policy_reference.jsonl", "w") as f:
        for e in replay:
            f.write(json.dumps(e) + "\n")
    rec["verify"] = verify
    with open(out / "policy_log.json", "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(json.dumps(verify, indent=1))


if __name__ == "__main__":
    main()
