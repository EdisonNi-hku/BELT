import argparse
import json
import sys
from pathlib import Path

import numpy as np

from rl_controller.iclr import learned_linear_h as lh
from rl_controller.iclr import learned_linear_policy as lp
from rl_controller.iclr import read_splits as rs
from rl_controller.scripts.iclr_single_checkpoint_panel import quantile


def read(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index_path(npy):
    return Path(npy).with_suffix(".index.jsonl")


def fit_one_price_h(fit_t, dev_t, n_features, price, kappas=lh.KAPPAS_H):
    import torch
    fits = []
    for kappa in kappas:
        best = None
        for init in lp.INIT_STOP:
            record = lp.fit(fit_t, n_features, price, kappa, init)
            w = torch.as_tensor(record["w"], dtype=torch.float64)
            b = torch.as_tensor(record["b"], dtype=torch.float64)
            with torch.no_grad():
                record.update(kappa=kappa, init_stop=init, n_features=n_features,
                              fit_utility=float(lp.utility(fit_t, w, b, price)),
                              dev_utility=float(lp.utility(dev_t, w, b, price)),
                              fit_p_natural=lp.expect(fit_t, w, b, 0.0, 1.0),
                              w_max_abs=max(abs(x) for x in record["w"]))
            record["plateau"] = lp.is_plateau(record)
            if best is None or record["dev_utility"] > best["dev_utility"]:
                best = record
        fits.append(best)
    return fits, lp.select_kappa(fits)


def thresholds_both_ways(z):
    p = 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))
    tau_z = [quantile(list(map(float, z)), level) for level in lp.LEVELS]
    tau_p = [quantile(list(map(float, p)), level) for level in lp.LEVELS]
    fire_z = [set(np.flatnonzero(np.asarray(z) >= t).tolist()) for t in tau_z]
    fire_p = [set(np.flatnonzero(p >= t).tolist()) for t in tau_p]
    return {"tau_logit": tau_z, "tau_p": tau_p,
            "n_distinct_logit": len(set(tau_z)), "n_distinct_p": len(set(tau_p)),
            "firing_sets_identical": fire_z == fire_p,
            "n_levels_where_firing_differs": sum(a != b for a, b in zip(fire_z, fire_p)),
            "p_saturated_fraction": float(np.mean((p <= 0.0) | (p >= 1.0)))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-val", required=True)
    ap.add_argument("--weights", required=True, help="weights.json, for the split digests")
    ap.add_argument("--reprs-val", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--price", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    chains = read(args.source_val)
    qkeys = sorted({c["qkey"] for c in chains})
    fit_q, dev_q = lp.split_questions(qkeys)
    declared = json.load(open(args.weights))["split"]
    got_fit, got_dev = rs.qkeys_sha256(fit_q), rs.qkeys_sha256(dev_q)
    if got_fit != declared["fit_qkeys_sha256"] or got_dev != declared["dev_qkeys_sha256"]:
        raise SystemExit("reconstructed split does not match the producer's")

    fit_set, dev_set = set(fit_q), set(dev_q)
    fit_chains = [c for c in chains if c["qkey"] in fit_set]
    dev_chains = [c for c in chains if c["qkey"] in dev_set]

    features = lh.features_from(lh.load_reprs(args.reprs_val, index_path(args.reprs_val)))
    mean, std = lp.standardizer(fit_chains, features=features)
    n_features = len(mean)
    fit_b = lp.pack(fit_chains, mean, std, features=features, n_features=n_features)
    dev_b = lp.pack(dev_chains, mean, std, features=features, n_features=n_features)
    fits, chosen = fit_one_price_h(lp.to_torch(fit_b), lp.to_torch(dev_b), n_features, args.price)
    model = fits[chosen]

    dev_x = (np.concatenate([features(c) for c in dev_chains]) - mean) / std
    z = lh.decision_logits_h(dev_x, model["w"], model["b"])
    taus = thresholds_both_ways(z)

    doc = {"schema": "x13-one-direction/v1", "seed": args.seed, "feature_set": "H",
           "family": "LB_H_1", "price": args.price,
           "price_on_registered_grid": args.price in lp.LAMBDAS,
           "lambda_index": (lp.LAMBDAS.index(args.price) if args.price in lp.LAMBDAS else None),
           "threshold_split": "dev", "threshold_levels": list(lp.LEVELS),
           "n_threshold_probes": int(len(z)), "n_features": n_features,
           "w": list(model["w"]), "b": float(model["b"]), "kappa": model["kappa"],
           "init_stop": model["init_stop"], "plateau": model["plateau"],
           "all_kappas_plateau": all(f["plateau"] for f in fits),
           "fit_utility": model["fit_utility"], "dev_utility": model["dev_utility"],
           "fit_p_natural": model["fit_p_natural"], "w_max_abs": model["w_max_abs"],
           "max_abs_grad": model["max_abs_grad"],
           "standardization": {"mean": mean.tolist(), "std": std.tolist(),
                               "population": "policy-fit probes, unweighted",
                               "features": "768-d pre-classifier vector"},
           "split": {"fit_qkeys_sha256": got_fit, "dev_qkeys_sha256": got_dev,
                     "n_fit_questions": len(fit_q), "n_dev_questions": len(dev_q),
                     "n_fit_chains": len(fit_chains), "n_dev_chains": len(dev_chains)},
           "thresholds": taus,
           "kappa_grid": [{k: f[k] for k in ("kappa", "dev_utility", "fit_utility", "plateau",
                                             "w_max_abs", "fit_p_natural", "init_stop")}
                          for f in fits]}
    with open(args.out, "w") as handle:
        json.dump(doc, handle, indent=1, sort_keys=True)
    print(json.dumps({"seed": args.seed, "family": "LB_H_1", "price": args.price,
                      "kappa": model["kappa"], "plateau": model["plateau"],
                      "dev_utility": round(model["dev_utility"], 6),
                      "n_features": n_features,
                      "n_distinct_logit": taus["n_distinct_logit"],
                      "n_distinct_p": taus["n_distinct_p"],
                      "firing_identical": taus["firing_sets_identical"],
                      "p_saturated_fraction": round(taus["p_saturated_fraction"], 6),
                      "out": args.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
