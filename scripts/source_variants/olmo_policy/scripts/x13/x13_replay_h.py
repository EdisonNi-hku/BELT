import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/cluster/project/sachan/njingwei/iclr/olmo3_rep/code/wt-0647deab")

from rl_controller.iclr import learned_linear_h as lh
from rl_controller.iclr import learned_linear_policy as lp
from scripts import olmo3_free_rows as fr


def load(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index_path(npy):
    return Path(npy).with_suffix(".index.jsonl")


def merged_reprs(paths):
    out = lh.Reprs()
    dims = set()
    for path in paths:
        part = lh.load_reprs(path, index_path(path))
        dims.add(part.dim)
        for key, rows in part.items():
            if key in out:
                raise SystemExit(f"chain {key} appears in more than one repr dump; "
                                 "the pool's dumps are supposed to partition it")
            out[key] = rows
    if len(dims) != 1:
        raise SystemExit(f"the repr dumps disagree about row width: {sorted(dims)}")
    out.dim = dims.pop()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--reference-export", required=True)
    ap.add_argument("--pool-dump", required=True)
    ap.add_argument("--val-dump", required=True)
    ap.add_argument("--reprs-pool", required=True, nargs="+",
                    help="every repr .npy covering the pool (951 select plus ext74)")
    ap.add_argument("--reprs-val", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    fit = json.load(open(args.fit))
    ref = json.loads(Path(args.reference_export).read_text(encoding="utf-8"))
    if ref.get("schema_version") != "iclr-policy-replay/v1":
        raise SystemExit(f"--reference-export declares {ref.get('schema_version')!r}")

    mean = np.asarray(fit["standardization"]["mean"], dtype=np.float64)
    std = np.asarray(fit["standardization"]["std"], dtype=np.float64)
    w, b = fit["w"], fit["b"]
    taus = fit["thresholds"]["tau_logit"]

    pool_chains, val_chains = load(args.pool_dump), load(args.val_dump)
    pool_features = lh.features_from(merged_reprs(args.reprs_pool))
    val_features = lh.features_from(merged_reprs([args.reprs_val]))

    cache = {}

    def logits_of(chain, features):
        key = (id(chain),)
        if key not in cache:
            x = (features(chain) - mean) / std
            cache[key] = lh.decision_logits_h(x, w, b)
        return cache[key]

    pool_ids = {id(c) for c in pool_chains}

    members = []
    for k, tau in enumerate(taus):
        def rule(chain, _tau=tau):
            features = pool_features if id(chain) in pool_ids else val_features
            z = logits_of(chain, features)
            hits = np.flatnonzero(z >= _tau)
            return int(hits[0]) if len(hits) else None
        members.append((f"k{k:02d}", {"name": "threshold_level", "value": k}, rule))

    dumps = {fr.READ_SPLIT: (pool_chains, args.pool_dump, fr.sha256(args.pool_dump)),
             fr.VAL_SPLIT: (val_chains, args.val_dump, fr.sha256(args.val_dump))}
    family = f"{fit['family']}@{args.tag}"
    frozen_extra = {
        "reader_seed": fit["seed"], "fit_price": fit["price"],
        "fit_price_on_registered_grid": fit["price_on_registered_grid"],
        "threshold_set": {
            "split": fit["threshold_split"], "statistic": "decision logit z = w.x + b",
            "levels": list(lp.LEVELS), "thresholds": taus,
            "rule": "stop at the first probe with z >= tau_k, else natural completion",
            "n_threshold_probes": fit["n_threshold_probes"],
            "p_saturated_fraction": fit["thresholds"]["p_saturated_fraction"],
            "n_distinct_thresholds_logit": fit["thresholds"]["n_distinct_logit"],
            "n_distinct_thresholds_p": fit["thresholds"]["n_distinct_p"]},
        "kappa": fit["kappa"], "feature_set": "H", "n_features": fit["n_features"]}
    out = fr.emit(ref, members, dumps, family=family,
                  out_dir=Path(args.out_dir), frozen_extra=frozen_extra)
    print(json.dumps({**out, "reader_seed": fit["seed"],
                      "n_pool_chains": len(pool_chains), "n_val_chains": len(val_chains)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
