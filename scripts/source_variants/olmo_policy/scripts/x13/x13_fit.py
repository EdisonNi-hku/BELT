import argparse
import json
import sys

import numpy as np

from rl_controller.iclr import learned_linear_policy as ll
from rl_controller.iclr import read_splits as rs
from rl_controller.scripts.iclr_single_checkpoint_panel import quantile


def read(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fit_one_price(fit_t, dev_t, n_features, price):
    import torch
    fits = []
    for kappa in ll.KAPPAS:
        best = None
        for init in ll.INIT_STOP:
            trial = ll.fit(fit_t, n_features, price, kappa, init)
            if best is None or trial["objective"] > best["objective"]:
                best = trial
        w = torch.tensor(best["w"], dtype=torch.float64)
        b = torch.tensor(best["b"], dtype=torch.float64)
        with torch.no_grad():
            best.update(kappa=kappa,
                        fit_utility=float(ll.utility(fit_t, w, b, price)),
                        dev_utility=float(ll.utility(dev_t, w, b, price)),
                        fit_p_natural=ll.expect(fit_t, w, b, 0.0, 1.0),
                        w_max_abs=max(abs(x) for x in best["w"]))
        best["plateau"] = ll.is_plateau(best)
        fits.append(best)
    chosen = ll.select_kappa(fits)
    return fits, chosen


def thresholds_both_ways(z):
    p = 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))
    tau_z = [quantile(list(map(float, z)), level) for level in ll.LEVELS]
    tau_p = [quantile(list(map(float, p)), level) for level in ll.LEVELS]
    fire_z = [set(np.flatnonzero(np.asarray(z) >= t).tolist()) for t in tau_z]
    fire_p = [set(np.flatnonzero(p >= t).tolist()) for t in tau_p]
    return {
        "tau_logit": tau_z,
        "tau_p": tau_p,
        "n_distinct_logit": len(set(tau_z)),
        "n_distinct_p": len(set(tau_p)),
        "firing_sets_identical": fire_z == fire_p,
        "n_levels_where_firing_differs": sum(a != b for a, b in zip(fire_z, fire_p)),
        "p_saturated_fraction": float(np.mean((p <= 0.0) | (p >= 1.0))),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-val", required=True, help="the 320-question reader val")
    ap.add_argument("--weights", required=True, help="the producer's weights.json, for the split digests")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--feature-set", default="QVRS", choices=sorted(ll.FEATURE_SETS))
    ap.add_argument("--price", type=float, required=True)
    ap.add_argument("--threshold-split", choices=("dev", "fit"), default="dev")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    chains = read(args.source_val)
    qkeys = sorted({c["qkey"] for c in chains})
    fit_q, dev_q = ll.split_questions(qkeys)

    declared = json.load(open(args.weights))["split"]
    got_fit, got_dev = rs.qkeys_sha256(fit_q), rs.qkeys_sha256(dev_q)
    if got_fit != declared["fit_qkeys_sha256"] or got_dev != declared["dev_qkeys_sha256"]:
        raise SystemExit(
            f"reconstructed split does not match the producer's: fit {got_fit} vs "
            f"{declared['fit_qkeys_sha256']}, dev {got_dev} vs {declared['dev_qkeys_sha256']}. "
            "Fitting would use a different half than the arm X13 is compared with.")

    fit_set, dev_set = set(fit_q), set(dev_q)
    fit_chains = [c for c in chains if c["qkey"] in fit_set]
    dev_chains = [c for c in chains if c["qkey"] in dev_set]

    mean, std = ll.standardizer(fit_chains)
    columns = ll.FEATURE_SETS[args.feature_set]
    fit_t = ll.to_torch(ll.pack(fit_chains, mean, std), columns)
    dev_t = ll.to_torch(ll.pack(dev_chains, mean, std), columns)

    fits, chosen = fit_one_price(fit_t, dev_t, len(columns), args.price)
    model = fits[chosen]

    source = fit_chains if args.threshold_split == "fit" else dev_chains
    x = (ll.flat_features(source) - mean) / std
    z = ll.decision_logits(x, columns, model["w"], model["b"])
    taus = thresholds_both_ways(z)

    doc = {
        "schema": "x13-one-direction/v1",
        "seed": args.seed,
        "feature_set": args.feature_set,
        "family": f"LB_{args.feature_set}_1",
        "price": args.price,
        "price_on_registered_grid": args.price in ll.LAMBDAS,
        "lambda_index": (ll.LAMBDAS.index(args.price) if args.price in ll.LAMBDAS else None),
        "threshold_split": args.threshold_split,
        "threshold_levels": list(ll.LEVELS),
        "n_threshold_probes": int(len(z)),
        "w": model["w"], "b": model["b"], "kappa": model["kappa"],
        "init_stop": model["init_stop"], "plateau": model["plateau"],
        "all_kappas_plateau": all(f["plateau"] for f in fits),
        "fit_utility": model["fit_utility"], "dev_utility": model["dev_utility"],
        "fit_p_natural": model["fit_p_natural"], "w_max_abs": model["w_max_abs"],
        "max_abs_grad": model["max_abs_grad"], "n_iter": model["n_iter"],
        "standardization": {"features": list(ll.RAW_FEATURES), "mean": mean.tolist(),
                            "std": std.tolist(), "population": "policy-fit probes, unweighted"},
        "split": {"fit_qkeys_sha256": got_fit, "dev_qkeys_sha256": got_dev,
                  "n_fit_questions": len(fit_q), "n_dev_questions": len(dev_q),
                  "n_fit_chains": len(fit_chains), "n_dev_chains": len(dev_chains)},
        "thresholds": taus,
        "kappa_grid": [{k: f[k] for k in ("kappa", "dev_utility", "fit_utility", "plateau",
                                          "w_max_abs", "fit_p_natural")} for f in fits],
    }
    with open(args.out, "w") as handle:
        json.dump(doc, handle, indent=1, sort_keys=True)
    print(json.dumps({"seed": args.seed, "family": doc["family"], "price": args.price,
                      "on_grid": doc["price_on_registered_grid"],
                      "threshold_split": args.threshold_split,
                      "kappa": model["kappa"], "plateau": model["plateau"],
                      "dev_utility": round(model["dev_utility"], 6),
                      "n_distinct_logit": taus["n_distinct_logit"],
                      "n_distinct_p": taus["n_distinct_p"],
                      "firing_identical": taus["firing_sets_identical"],
                      "p_saturated_fraction": round(taus["p_saturated_fraction"], 6),
                      "out": args.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
