from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class LogisticModel:
    weights: tuple[float, ...]
    intercept: float
    means: tuple[float, ...]
    stds: tuple[float, ...]

    def predict_proba(self, x: Sequence[float]) -> float:
        if len(x) != len(self.weights):
            raise ValueError("feature length mismatch")
        z = self.intercept
        for value, weight, mean, std in zip(x, self.weights, self.means, self.stds):
            z += weight * ((value - mean) / std)
        return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, z))))


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("singular IRLS system (l2 too small?)")
        a[col], a[pivot] = a[pivot], a[col]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col] / a[col][col]
            for k in range(col, n + 1):
                a[row][k] -= factor * a[col][k]
    return [a[i][n] / a[i][i] for i in range(n)]


def fit_logistic(
    X: Sequence[Sequence[float]],
    y: Sequence[bool],
    *,
    l2: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> LogisticModel:
    if l2 <= 0.0:
        raise ValueError("l2 must be positive")
    if len(X) != len(y) or not X:
        raise ValueError("X and y must be nonempty and of equal length")
    n, d = len(X), len(X[0])
    if any(len(row) != d for row in X):
        raise ValueError("ragged feature rows")
    means = [sum(row[j] for row in X) / n for j in range(d)]
    stds = []
    for j in range(d):
        var = sum((row[j] - means[j]) ** 2 for row in X) / n
        stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    Z = [[(row[j] - means[j]) / stds[j] for j in range(d)] for row in X]
    labels = [1.0 if value else 0.0 for value in y]

    beta = [0.0] * (d + 1)
    for _ in range(max_iter):
        gradient = [0.0] * (d + 1)
        hessian = [[0.0] * (d + 1) for _ in range(d + 1)]
        for row, label in zip(Z, labels):
            z = beta[0] + sum(beta[j + 1] * row[j] for j in range(d))
            p = 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, z))))
            w = max(p * (1.0 - p), 1e-9)
            residual = label - p
            extended = [1.0] + list(row)
            for a_idx in range(d + 1):
                gradient[a_idx] += residual * extended[a_idx]
                for b_idx in range(d + 1):
                    hessian[a_idx][b_idx] += w * extended[a_idx] * extended[b_idx]
        for j in range(1, d + 1):
            gradient[j] -= l2 * beta[j]
            hessian[j][j] += l2
        step = _solve(hessian, gradient)
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < tol:
            break
    return LogisticModel(
        weights=tuple(beta[1:]),
        intercept=beta[0],
        means=tuple(means),
        stds=tuple(stds),
    )
