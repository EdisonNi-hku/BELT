import hashlib
import math
from collections import Counter

import numpy as np

from rl_controller.iclr import osp_policy as osp

SPLIT_SALT = 'iclr-policy-split-2026-09-15'
BUDGET = 12000
CLIP = 1e-4
RAW_FEATURES = ('q', 'v', 'r', 's', 'd')
FEATURE_SETS = {'DEPTH': ('d',), 'QV': ('q', 'v'), 'QV_D': ('q', 'v', 'd'), 'QS': ('q', 's'),
                'QS_D': ('q', 's', 'd'), 'QVRS': ('q', 'v', 'r', 's'),
                'QVRS_D': ('q', 'v', 'r', 's', 'd')}
LAMBDAS = tuple(2e-6 * 20.0 ** (k / 12.0) for k in range(17))
KAPPAS = (0.0, 1e-5, 1e-4)
INIT_STOP = (0.1, 0.01)
LEVELS = tuple(k / 12 for k in range(13))
BUCKETS = 8
PAD = -1e4
PLATEAU_P_NATURAL = 0.99
PLATEAU_W_MAX_ABS = 1e-3
LBFGS = dict(lr=1.0, max_iter=500, history_size=20, tolerance_grad=1e-9,
             tolerance_change=1e-12, line_search_fn='strong_wolfe')


def grids():
    return dict(split_salt=SPLIT_SALT, budget=BUDGET, clip=CLIP, raw_features=list(RAW_FEATURES),
                feature_sets={k: list(v) for k, v in FEATURE_SETS.items()}, lambdas=list(LAMBDAS),
                kappas=list(KAPPAS), init_stop=list(INIT_STOP), levels=list(LEVELS),
                buckets=BUCKETS, pad=PAD, lbfgs=LBFGS, grade='loose', cost='total',
                plateau=dict(fit_p_natural_ge=PLATEAU_P_NATURAL, w_max_abs_lt=PLATEAU_W_MAX_ABS),
                kappa_selection='max policy-dev utility among non-plateau fits (tie: larger kappa); '
                                'kappa = 0 if every kappa is in the plateau',
                weighting='question', objective='survival_product_expected_utility')


def split_questions(qkeys):
    ranked = sorted(set(qkeys), key=lambda q: hashlib.sha256(f'{SPLIT_SALT}:{q}'.encode()).hexdigest())
    half = len(ranked) // 2
    return ranked[:half], ranked[half:]


def logit(p):
    p = min(max(float(p), CLIP), 1.0 - CLIP)
    return math.log(p / (1.0 - p))


def probe_features(chain):
    rows = [[logit(s['q']), logit(s['v']), logit(s['r']), logit(s['stability']),
             p['prefix_len'] / BUDGET] for p, s in zip(chain['probes'], chain['scores'])]
    return np.array(rows, dtype=np.float64).reshape(-1, len(RAW_FEATURES))


def flat_features(chains):
    return np.concatenate([probe_features(c) for c in chains] or [np.zeros((0, len(RAW_FEATURES)))])


def standardizer(fit_chains):
    x = flat_features(fit_chains)
    if not len(x):
        raise ValueError('policy-fit split has no probes')
    std = x.std(0)
    return x.mean(0), np.where(std > 0, std, 1.0)


def pack(chains, mean, std):
    counts = Counter(c['qkey'] for c in chains)
    order = sorted(range(len(chains)), key=lambda i: (len(chains[i]['probes']), i))
    buckets = []
    for part in np.array_split(np.array(order, dtype=np.int64), min(BUCKETS, len(order))):
        rows = [chains[i] for i in part]
        n, t = len(rows), max(1, max(len(c['probes']) for c in rows))
        b = dict(index=part, x=np.zeros((n, t, len(RAW_FEATURES))), mask=np.zeros((n, t), bool),
                 a=np.zeros((n, t)), c=np.zeros((n, t)), think=np.zeros((n, t)), a_nat=np.zeros(n),
                 c_nat=np.zeros(n), think_nat=np.zeros(n), weight=np.zeros(n))
        for j, chain in enumerate(rows):
            terms = osp.chain_terms(chain, grade='loose', cost='total')
            k = terms.n_probes
            b['x'][j, :k] = (probe_features(chain) - mean) / std
            b['mask'][j, :k] = True
            b['a'][j, :k], b['c'][j, :k] = terms.a, terms.l
            b['think'][j, :k] = [p['prefix_len'] for p in chain['probes']]
            b['a_nat'][j], b['c_nat'][j] = terms.a_nat, terms.l_nat
            b['think_nat'][j] = chain['thinking_tokens']
            b['weight'][j] = 1.0 / (len(counts) * counts[chain['qkey']])
        buckets.append(b)
    return buckets


def to_torch(buckets, columns):
    import torch
    idx = [RAW_FEATURES.index(f) for f in columns]
    return [{k: (torch.from_numpy(np.ascontiguousarray(v[..., idx])) if k == 'x' else torch.from_numpy(v))
             for k, v in b.items() if k != 'index'} for b in buckets]


def logits(bucket, w, b):
    import torch
    z = bucket['x'] @ w + b
    return torch.where(bucket['mask'], z, torch.full_like(z, PAD))


def utility(buckets, w, b, lam):
    return sum((t['weight'] * osp.expected_utility(logits(t, w, b), t['a'], t['c'], t['a_nat'],
                                                   t['c_nat'], lam)).sum() for t in buckets)


def expect(buckets, w, b, value, natural):
    return float(sum((t['weight'] * osp.expected_utility(
        logits(t, w, b), t[value] if isinstance(value, str) else value, 0.0,
        t[natural] if isinstance(natural, str) else natural, 0.0, 0.0)).sum() for t in buckets))


def stochastic_summary(buckets, w, b, lam):
    import torch
    w, b = torch.as_tensor(w, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    acc, total = expect(buckets, w, b, 'a', 'a_nat'), expect(buckets, w, b, 'c', 'c_nat')
    return dict(accuracy=acc, total_tokens=total, utility=acc - lam * total,
                thinking_tokens=expect(buckets, w, b, 'think', 'think_nat'),
                p_natural=expect(buckets, w, b, 0.0, 1.0))


def fit(buckets, n_features, lam, kappa, init_stop):
    import torch
    w = torch.zeros(n_features, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(logit(init_stop), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], **LBFGS)

    def loss_fn():
        return -utility(buckets, w, b, lam) + kappa * (w * w).sum()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss
    optimizer.step(closure)
    optimizer.zero_grad()
    loss = loss_fn()
    loss.backward()
    grad = float(torch.cat([w.grad.reshape(-1), b.grad.reshape(-1)]).abs().max())
    state = optimizer.state[optimizer._params[0]]
    return dict(init_stop=init_stop, w=[float(x) for x in w.detach()], b=float(b.detach()),
                objective=-loss.item(), n_iter=int(state.get('n_iter', 0)),
                func_evals=int(state.get('func_evals', 0)), max_abs_grad=grad)


def is_plateau(fit_record):
    return bool(fit_record['fit_p_natural'] >= PLATEAU_P_NATURAL and fit_record['w_max_abs'] < PLATEAU_W_MAX_ABS)


def select_kappa(fits):
    eligible = [i for i, f in enumerate(fits) if not f['plateau']]
    if not eligible:
        return next(i for i, f in enumerate(fits) if f['kappa'] == 0.0)
    return max(eligible, key=lambda i: (fits[i]['dev_utility'], fits[i]['kappa']))


def fit_feature_set(name, fit_buckets, dev_buckets):
    import torch
    columns = FEATURE_SETS[name]
    fit_t, dev_t = to_torch(fit_buckets, columns), to_torch(dev_buckets, columns)
    models = []
    for index, lam in enumerate(LAMBDAS):
        fits = []
        for kappa in KAPPAS:
            best = None
            for init in INIT_STOP:
                trial = fit(fit_t, len(columns), lam, kappa, init)
                if best is None or trial['objective'] > best['objective']:
                    best = trial
            w, b = torch.tensor(best['w'], dtype=torch.float64), torch.tensor(best['b'], dtype=torch.float64)
            with torch.no_grad():
                best.update(kappa=kappa, fit_utility=float(utility(fit_t, w, b, lam)),
                            dev_utility=float(utility(dev_t, w, b, lam)),
                            fit_p_natural=expect(fit_t, w, b, 0.0, 1.0),
                            w_max_abs=max(abs(x) for x in best['w']))
            best['plateau'] = is_plateau(best)
            fits.append(best)
        chosen = select_kappa(fits)
        models.append(dict(lambda_index=index, lam=lam, kappa=fits[chosen]['kappa'], w=fits[chosen]['w'],
                           b=fits[chosen]['b'], plateau=fits[chosen]['plateau'],
                           all_kappas_plateau=all(f['plateau'] for f in fits), fits=fits))
    return models


def decision_logits(x_std, columns, w, b):
    z = np.full(len(x_std), float(b))
    for column, weight in zip(columns, w):
        z += float(weight) * x_std[:, RAW_FEATURES.index(column)]
    return z


def thresholds(fit_z):
    from rl_controller.scripts.iclr_single_checkpoint_panel import quantile
    ordered = sorted(float(z) for z in fit_z)
    return [quantile(ordered, level) for level in LEVELS]


def ranks(z, taus):
    return np.searchsorted(np.asarray(taus, dtype=np.float64), z, side='right').astype(np.uint8)


def fit_models(fit_chains, dev_chains):
    mean, std = standardizer(fit_chains)
    fit_buckets, dev_buckets = pack(fit_chains, mean, std), pack(dev_chains, mean, std)
    fit_x = (flat_features(fit_chains) - mean) / std
    models = {}
    for name, columns in FEATURE_SETS.items():
        rows = fit_feature_set(name, fit_buckets, dev_buckets)
        for row in rows:
            row['taus'] = thresholds(decision_logits(fit_x, columns, row['w'], row['b']))
            row['constant_score'] = row['taus'][0] == row['taus'][-1]
        models[name] = rows
    return dict(schema='iclr-learned-linear-weights/v1', grids=grids(),
                standardization=dict(features=list(RAW_FEATURES), mean=mean.tolist(), std=std.tolist(),
                                     population='policy-fit probes, unweighted'),
                models=models)


def _standardization(weights):
    s = weights['standardization']
    return np.asarray(s['mean']), np.asarray(s['std'])


def deploy(weights, chains):
    mean, std = _standardization(weights)
    x = (flat_features(chains) - mean) / std
    return {name: np.stack([ranks(decision_logits(x, FEATURE_SETS[name], m['w'], m['b']), m['taus'])
                            for m in rows]) for name, rows in weights['models'].items()}


def diagnostics(weights, chains):
    mean, std = _standardization(weights)
    buckets, codes = pack(chains, mean, std), deploy(weights, chains)
    out = {}
    for name, rows in weights['models'].items():
        tensors = to_torch(buckets, FEATURE_SETS[name])
        out[name] = [dict(stochastic_summary(tensors, m['w'], m['b'], m['lam']), lambda_index=m['lambda_index'],
                          rank_counts=np.bincount(codes[name][i], minlength=len(LEVELS) + 1).tolist())
                     for i, m in enumerate(rows)]
    return out
