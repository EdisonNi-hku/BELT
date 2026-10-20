from __future__ import annotations

HIDDEN = 16
LEARNING_RATES = (1e-2, 3e-3)
MAX_STEPS = 2000
PATIENCE = 50


def init_params(n_features, hidden, init_stop, generator=None):
    import torch
    from rl_controller.iclr.learned_linear_policy import logit

    w1 = torch.empty(n_features, hidden, dtype=torch.float64)
    torch.nn.init.normal_(w1, std=0.1, generator=generator)
    b1 = torch.zeros(hidden, dtype=torch.float64)
    w2 = torch.zeros(hidden, dtype=torch.float64)
    b2 = torch.tensor(logit(init_stop), dtype=torch.float64)
    for t in (w1, b1, w2, b2):
        t.requires_grad_(True)
    return w1, b1, w2, b2


def mlp_logits(bucket, params):
    import torch
    from rl_controller.iclr.learned_linear_policy import PAD

    w1, b1, w2, b2 = params
    z = torch.tanh(bucket["x"] @ w1 + b1) @ w2 + b2
    return torch.where(bucket["mask"], z, torch.full_like(z, PAD))


def utility(buckets, params, lam):
    from rl_controller.iclr import osp_policy as osp
    return sum((t["weight"] * osp.expected_utility(
        mlp_logits(t, params), t["a"], t["c"], t["a_nat"], t["c_nat"], lam)).sum()
        for t in buckets)


def fit_mlp(fit_buckets, dev_buckets, n_features, lam, lr, init_stop, *,
            hidden=HIDDEN, max_steps=MAX_STEPS, patience=PATIENCE, seed=0):
    import torch

    generator = torch.Generator().manual_seed(seed)
    params = init_params(n_features, hidden, init_stop, generator=generator)
    optimizer = torch.optim.Adam(params, lr=lr)
    best = {"dev_utility": float("-inf"), "step": -1,
            "params": [p.detach().clone() for p in params]}
    stale = 0
    for step in range(max_steps):
        optimizer.zero_grad()
        loss = -utility(fit_buckets, params, lam)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            dev = float(utility(dev_buckets, params, lam))
        if dev > best["dev_utility"]:
            best = {"dev_utility": dev, "step": step,
                    "params": [p.detach().clone() for p in params]}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    with torch.no_grad():
        fit_utility = float(utility(fit_buckets, best["params"], lam))
    return {"policy": "mlp", "hidden": hidden, "lr": lr, "lam": lam,
            "init_stop": init_stop, "seed": seed,
            "w1": best["params"][0].tolist(), "b1": best["params"][1].tolist(),
            "w2": best["params"][2].tolist(), "b2": float(best["params"][3]),
            "dev_utility": best["dev_utility"], "fit_utility": fit_utility,
            "selected_step": best["step"], "steps_run": step + 1,
            "stopped_early": stale >= patience}
