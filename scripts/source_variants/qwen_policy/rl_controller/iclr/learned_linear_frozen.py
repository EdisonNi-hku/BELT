from __future__ import annotations

from collections import Counter

import numpy as np

from rl_controller.iclr import osp_policy as osp
from rl_controller.iclr.learned_linear_policy import BUCKETS, PAD


def frozen_matrix(chain, dump, width=None):
    key = (chain["qkey"], chain.get("rep"))
    n = len(chain.get("probes") or [])
    rows = dump.get(key)
    if rows is None:
        if n == 0:
            return np.zeros((0, int(width or 0)), dtype=np.float32)
        raise SystemExit(f"the dump does not cover chain {key}, which has {n} probes; a fit may "
                         "not silently drop chains the policy is supposed to see")
    if rows.shape[0] != n:
        raise SystemExit(f"chain {key}: dump has {rows.shape[0]} rows for {n} probes. The dump "
                         "writes one row per scored probe in probe order; a mismatch means the "
                         "features are not aligned with the probes and the fit would be silently "
                         "wrong.")
    return rows


def flat_frozen(chains, dump):
    width = int(next(iter(dump.values())).shape[1]) if dump else 0
    mats = [frozen_matrix(c, dump, width) for c in chains]
    if not mats:
        return np.zeros((0, 0), dtype=np.float64)
    return np.concatenate(mats).astype(np.float64)


def standardizer_frozen(fit_chains, dump):
    x = flat_frozen(fit_chains, dump)
    if not len(x):
        raise ValueError("policy-fit split has no probes")
    std = x.std(0)
    return x.mean(0), np.where(std > 0, std, 1.0)


def pack_frozen(chains, dump, mean, std):
    width = int(next(iter(dump.values())).shape[1]) if dump else 0
    counts = Counter(c["qkey"] for c in chains)
    order = sorted(range(len(chains)), key=lambda i: (len(chains[i].get("probes") or []), i))
    buckets = []
    for part in np.array_split(np.array(order, dtype=np.int64), min(BUCKETS, max(1, len(order)))):
        rows = [chains[i] for i in part]
        if not len(rows):
            continue
        n = len(rows)
        t = max(1, max(len(c.get("probes") or []) for c in rows))
        b = dict(index=part, x=np.zeros((n, t, width)), mask=np.zeros((n, t), bool),
                 a=np.zeros((n, t)), c=np.zeros((n, t)), think=np.zeros((n, t)),
                 a_nat=np.zeros(n), c_nat=np.zeros(n), think_nat=np.zeros(n), weight=np.zeros(n))
        for j, chain in enumerate(rows):
            terms = osp.chain_terms(chain, grade="loose", cost="total")
            k = terms.n_probes
            if k:
                b["x"][j, :k] = (frozen_matrix(chain, dump, width)[:k] - mean) / std
            b["mask"][j, :k] = True
            b["a"][j, :k], b["c"][j, :k] = terms.a, terms.l
            b["think"][j, :k] = [p["prefix_len"] for p in chain["probes"][:k]]
            b["a_nat"][j], b["c_nat"][j] = terms.a_nat, terms.l_nat
            b["think_nat"][j] = chain["thinking_tokens"]
            b["weight"][j] = 1.0 / (len(counts) * counts[chain["qkey"]])
        buckets.append(b)
    return buckets


def to_torch_frozen(buckets):
    import torch
    return [{k: torch.from_numpy(np.ascontiguousarray(v))
             for k, v in b.items() if k != "index"} for b in buckets]
