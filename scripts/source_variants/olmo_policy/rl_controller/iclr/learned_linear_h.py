from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl_controller.iclr import learned_linear_policy as lp

KAPPAS_H = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
FEATURE_SET = "H"


class Reprs(dict):
    dim = 0


def load_reprs(npy_path, index_path) -> "Reprs":
    rows = np.load(str(npy_path))
    by_chain: dict = {}
    for line in Path(index_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        by_chain.setdefault((entry["qkey"], int(entry["rep"])), []).append(
            (int(entry["probe_index"]), int(entry["row"])))
    out = Reprs()
    for key, pairs in by_chain.items():
        pairs.sort()
        if [i for i, _ in pairs] != list(range(len(pairs))):
            raise ValueError(f"{key}: probe indices are not 0..n-1; the dump index is not a "
                             "complete, ordered record of this chain's probes")
        out[key] = rows[[r for _, r in pairs]].astype(np.float64)
    out.dim = int(rows.shape[1]) if rows.ndim == 2 and rows.shape[0] else 0
    return out


def features_from(reprs: dict):
    dim = getattr(reprs, "dim", 0)

    def features(chain):
        key = (chain["qkey"], int(chain["rep"]))
        block = reprs.get(key)
        if block is None and not chain["probes"]:
            return np.zeros((0, dim), dtype=np.float64)
        if block is None:
            raise ValueError(f"chain {key} has no representation rows in the dump")
        if len(block) != len(chain["probes"]):
            raise ValueError(f"chain {key}: {len(block)} representation rows against "
                             f"{len(chain['probes'])} probes -- the dump and the chain disagree "
                             "about how many decision points there are")
        return block
    return features


def fit_h(fit_buckets, dev_buckets, *, kappas=KAPPAS_H):
    import torch

    fit_t, dev_t = lp.to_torch(fit_buckets), lp.to_torch(dev_buckets)
    n_features = fit_buckets[0]["x"].shape[-1]
    models = []
    for index, lam in enumerate(lp.LAMBDAS):
        fits = []
        for kappa in kappas:
            best = None
            for init in lp.INIT_STOP:
                record = lp.fit(fit_t, n_features, lam, kappa, init)
                w = torch.as_tensor(record["w"], dtype=torch.float64)
                b = torch.as_tensor(record["b"], dtype=torch.float64)
                with torch.no_grad():
                    record.update(kappa=kappa, init_stop=init, n_features=n_features,
                                  fit_utility=float(lp.utility(fit_t, w, b, lam)),
                                  dev_utility=float(lp.utility(dev_t, w, b, lam)),
                                  fit_p_natural=lp.expect(fit_t, w, b, 0.0, 1.0),
                                  w_max_abs=max(abs(x) for x in record["w"]))
                record["plateau"] = lp.is_plateau(record)
                if best is None or record["dev_utility"] > best["dev_utility"]:
                    best = record
            fits.append(best)
        chosen = fits[lp.select_kappa(fits)]
        models.append({"lambda_index": index, "lambda": lam, "chosen": chosen, "fits": fits,
                       "all_kappas_plateau": all(f["plateau"] for f in fits)})
    return models


def grids():
    return {"feature_set": FEATURE_SET, "kappas": list(KAPPAS_H),
            "lambdas": list(lp.LAMBDAS), "levels": list(lp.LEVELS),
            "init_stop": list(lp.INIT_STOP), "split_salt": lp.SPLIT_SALT,
            "kappa_selection": "max policy-dev utility among non-plateau fits (tie: larger kappa)",
            "init_selection": "max policy-dev utility across the two registered inits",
            "features": "768-d pre-classifier vector from iclr_score_pool --dump-reprs"}


def decision_logits_h(x_std, w, b):
    z = np.full(len(x_std), float(b), dtype=np.float64)
    for j, weight in enumerate(w):
        z += float(weight) * x_std[:, j]
    return z


def fit_models_h(fit_chains, dev_chains, features, *, kappas=KAPPAS_H):
    mean, std = lp.standardizer(fit_chains, features=features)
    n_features = len(mean)
    fit_buckets = lp.pack(fit_chains, mean, std, features=features, n_features=n_features)
    dev_buckets = lp.pack(dev_chains, mean, std, features=features, n_features=n_features)
    models = fit_h(fit_buckets, dev_buckets, kappas=kappas)
    fit_x = (np.concatenate([features(c) for c in fit_chains]) - mean) / std
    out = []
    for row in models:
        chosen = row["chosen"]
        z = decision_logits_h(fit_x, chosen["w"], chosen["b"])
        out.append({"lambda_index": row["lambda_index"], "lam": row["lambda"],
                    "kappa": chosen["kappa"], "w": list(chosen["w"]), "b": float(chosen["b"]),
                    "dev_utility": chosen["dev_utility"], "init_stop": chosen["init_stop"],
                    "plateau": chosen["plateau"], "all_kappas_plateau": row["all_kappas_plateau"],
                    "taus": lp.thresholds(z)})
    return {"standardization": {"mean": mean.tolist(), "std": std.tolist()},
            "models": {FEATURE_SET: out}, "grids": grids()}


def deploy_h(weights, chains, features):
    mean, std = lp._standardization(weights)
    x = (np.concatenate([features(c) for c in chains]) - mean) / std
    rows = weights["models"][FEATURE_SET]
    return {FEATURE_SET: np.stack([lp.ranks(decision_logits_h(x, m["w"], m["b"]), m["taus"])
                                   for m in rows])}
