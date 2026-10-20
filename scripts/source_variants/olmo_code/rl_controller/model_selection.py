from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from rl_controller.logistic import fit_logistic


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be nonempty and equal length")
    positives = [s for s, y in zip(scores, labels) if y]
    negatives = [s for s, y in zip(scores, labels) if not y]
    if not positives or not negatives:
        raise ValueError("AUC needs both classes")
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))


def grouped_kfold(groups: Sequence, k: int, seed: int) -> list[tuple[list, list]]:
    unique = sorted(set(groups), key=str)
    if k < 2 or k > len(unique):
        raise ValueError("k must be within [2, number of groups]")
    rng = random.Random(seed)
    rng.shuffle(unique)
    assignment = {group: index % k for index, group in enumerate(unique)}
    folds = []
    for fold in range(k):
        test_idx = [i for i, g in enumerate(groups) if assignment[g] == fold]
        train_idx = [i for i, g in enumerate(groups) if assignment[g] != fold]
        folds.append((train_idx, test_idx))
    return folds


def bootstrap_auc_ci(
    scores, labels, groups, *, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    by_group: dict = {}
    for s, y, g in zip(scores, labels, groups, strict=True):
        by_group.setdefault(g, []).append((s, y))
    keys = sorted(by_group, key=str)
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        sample_scores, sample_labels = [], []
        for _ in keys:
            rows = by_group[keys[rng.randrange(len(keys))]]
            sample_scores += [s for s, _ in rows]
            sample_labels += [y for _, y in rows]
        try:
            values.append(auc(sample_scores, sample_labels))
        except ValueError:
            continue
    if not values:
        raise ValueError("bootstrap produced no valid resamples")
    values.sort()
    lo = values[round((1 - level) / 2 * (len(values) - 1))]
    hi = values[round((1 + level) / 2 * (len(values) - 1))]
    return lo, hi


def _cv_auc(X, y, groups, l2, k, seed) -> float:
    scores = [0.0] * len(y)
    for train_idx, test_idx in grouped_kfold(groups, k, seed):
        model = fit_logistic([X[i] for i in train_idx], [y[i] for i in train_idx], l2=l2)
        for i in test_idx:
            scores[i] = model.predict_proba(X[i])
    return auc(scores, y)


def forward_select(
    feature_groups: Mapping[str, Sequence[Sequence[float]]],
    y: Sequence[bool],
    groups: Sequence,
    *,
    l2_grid: Sequence[float] = (0.3, 1.0, 3.0, 10.0),
    k: int = 5,
    seed: int = 0,
    epsilon: float = 1e-3,
) -> dict:
    names = sorted(feature_groups)
    chosen: list[str] = []
    best_auc, best_l2 = 0.5, l2_grid[0]
    path = []
    while True:
        best_candidate = None
        for name in names:
            if name in chosen:
                continue
            columns = chosen + [name]
            X = [
                [value for group_name in columns for value in feature_groups[group_name][i]]
                for i in range(len(y))
            ]
            for l2 in l2_grid:
                score = _cv_auc(X, y, groups, l2, k, seed)
                if best_candidate is None or score > best_candidate[0]:
                    best_candidate = (score, name, l2)
        if best_candidate is None or best_candidate[0] <= best_auc + epsilon:
            break
        best_auc, added, best_l2 = best_candidate
        chosen.append(added)
        path.append({"added": added, "cv_auc": best_auc, "l2": best_l2})
    X = [
        [value for group_name in chosen for value in feature_groups[group_name][i]]
        for i in range(len(y))
    ]
    train_auc = None
    if chosen:
        model = fit_logistic(X, y, l2=best_l2)
        train_auc = auc([model.predict_proba(row) for row in X], y)
    return {
        "chosen": chosen,
        "l2": best_l2,
        "cv_auc": best_auc,
        "train_auc": train_auc,
        "auc_gap": None if train_auc is None else train_auc - best_auc,
        "path": path,
    }


def bootstrap_auc_summary(
    scores, labels, groups, *, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> dict:
    by_group: dict = {}
    for score, label, group in zip(scores, labels, groups, strict=True):
        by_group.setdefault(group, []).append((score, label))
    keys = sorted(by_group, key=str)
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        sample_scores, sample_labels = [], []
        for _ in keys:
            rows = by_group[keys[rng.randrange(len(keys))]]
            sample_scores += [s for s, _ in rows]
            sample_labels += [y for _, y in rows]
        try:
            values.append(auc(sample_scores, sample_labels))
        except ValueError:
            continue
    if not values:
        raise ValueError("bootstrap produced no valid resamples")
    at_or_below = sum(1 for value in values if value <= 0.5)
    values.sort()
    return {
        "ci": [
            values[round((1 - level) / 2 * (len(values) - 1))],
            values[round((1 + level) / 2 * (len(values) - 1))],
        ],
        "pvalue": max(at_or_below / len(values), 1.0 / (len(values) + 1)),
        "n_resamples": len(values),
    }


def average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    n_pos = sum(1 for y in labels if y)
    if n_pos == 0:
        raise ValueError("average_precision needs at least one positive")
    hits, total = 0, 0.0
    for rank, i in enumerate(order, start=1):
        if labels[i]:
            hits += 1
            total += hits / rank
    return total / n_pos


def lift(scores: Sequence[float], labels: Sequence[bool]) -> float:
    base = sum(1 for y in labels if y) / len(labels)
    if base in (0.0, 1.0):
        raise ValueError(f"degenerate base rate {base}: lift is undefined")
    return average_precision(scores, labels) / base
