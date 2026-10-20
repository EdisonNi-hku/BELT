import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/cluster/project/sachan/njingwei/iclr/olmo3_rep/code/wt-0647deab")

from rl_controller.iclr import learned_linear_policy as ll
from scripts import olmo3_free_rows as fr


def load(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def chain_logits(chain, mean, std, idx, w, b):
    x = (ll.probe_features(chain) - mean) / std
    z = np.full(len(x), float(b), dtype=np.float64)
    for column, weight in zip(idx, w):
        z += float(weight) * x[:, column]
    return z


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True, help="the x13-one-direction/v1 doc")
    ap.add_argument("--reference-export", required=True)
    ap.add_argument("--pool-dump", required=True, help="the 1,025 lane select_scored.jsonl")
    ap.add_argument("--val-dump", required=True, help="the dev160 val_scored.jsonl")
    ap.add_argument("--tag", required=True, help="family tag, e.g. olmo-thr-s1")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    fit = json.load(open(args.fit))
    ref = json.loads(Path(args.reference_export).read_text(encoding="utf-8"))
    if ref.get("schema_version") != "iclr-policy-replay/v1":
        raise SystemExit(f"--reference-export declares {ref.get('schema_version')!r}")

    mean = np.asarray(fit["standardization"]["mean"], dtype=np.float64)
    std = np.asarray(fit["standardization"]["std"], dtype=np.float64)
    columns = ll.FEATURE_SETS[fit["feature_set"]]
    idx = [ll.RAW_FEATURES.index(c) for c in columns]
    w, b = fit["w"], fit["b"]
    taus = fit["thresholds"]["tau_logit"]
    if len(taus) != len(ll.LEVELS):
        raise SystemExit(f"{args.fit} carries {len(taus)} thresholds, expected {len(ll.LEVELS)}")

    cache = {}

    def logits_of(chain):
        key = id(chain)
        if key not in cache:
            cache[key] = chain_logits(chain, mean, std, idx, w, b)
        return cache[key]

    members = []
    for k, tau in enumerate(taus):
        def rule(chain, _tau=tau):
            z = logits_of(chain)
            hits = np.flatnonzero(z >= _tau)
            return int(hits[0]) if len(hits) else None
        members.append((f"k{k:02d}",
                        {"name": "threshold_level", "value": k}, rule))

    pool_chains, val_chains = load(args.pool_dump), load(args.val_dump)
    dumps = {fr.READ_SPLIT: (pool_chains, args.pool_dump, fr.sha256(args.pool_dump)),
             fr.VAL_SPLIT: (val_chains, args.val_dump, fr.sha256(args.val_dump))}

    family = f"{fit['family']}@{args.tag}"
    frozen_extra = {
        "reader_seed": fit["seed"],
        "fit_price": fit["price"],
        "fit_price_on_registered_grid": fit["price_on_registered_grid"],
        "threshold_set": {
            "split": fit["threshold_split"],
            "statistic": "decision logit z = w.x + b",
            "levels": list(ll.LEVELS),
            "thresholds": taus,
            "rule": "stop at the first probe with z >= tau_k, else natural completion",
            "n_threshold_probes": fit["n_threshold_probes"],
            "p_saturated_fraction": fit["thresholds"]["p_saturated_fraction"],
            "n_distinct_thresholds_logit": fit["thresholds"]["n_distinct_logit"],
            "n_distinct_thresholds_p": fit["thresholds"]["n_distinct_p"],
        },
        "kappa": fit["kappa"],
        "feature_set": fit["feature_set"],
    }
    out = fr.emit(ref, members, dumps, family=family,
                  out_dir=Path(args.out_dir), frozen_extra=frozen_extra)
    print(json.dumps({**out, "reader_seed": fit["seed"], "price": fit["price"],
                      "n_pool_chains": len(pool_chains), "n_val_chains": len(val_chains)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
