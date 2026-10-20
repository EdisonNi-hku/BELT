from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

from rl_controller.model_selection import auc

__all__ = [
    "DEFAULT_THRESHOLDS",
    "DriftError",
    "DriftThresholds",
    "assert_head_healthy",
    "expected_calibration_error",
    "head_drift",
    "spearman",
]


class DriftError(RuntimeError):
    pass


@dataclass(frozen=True)
class DriftThresholds:
    min_rank_correlation: float = 0.50
    max_median_uncertainty: float = 0.90
    max_ece_ratio: float = 2.0
    min_ece_absolute_margin: float = 0.05


DEFAULT_THRESHOLDS = DriftThresholds()


def _validated(name, values):
    out = [float(v) for v in values]
    if not out:
        raise DriftError(f"{name} is empty; the guard cannot pass on no evidence")
    for v in out:
        if not math.isfinite(v):
            raise DriftError(f"{name} contains a non-finite value {v!r}")
    return out


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in order[i:j + 1]:
            ranks[k] = shared
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    a = _validated("reference", a)
    b = _validated("current", b)
    if len(a) != len(b):
        raise DriftError(
            f"paired comparison needs equal lengths, got {len(a)} and {len(b)}"
        )
    ra, rb = _ranks(a), _ranks(b)
    n = len(ra)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    if var_a <= 0 or var_b <= 0:
        return 0.0
    return cov / math.sqrt(var_a * var_b)


def expected_calibration_error(probs, labels, *, n_bins: int = 10) -> float:
    probs = _validated("probs", probs)
    labels = [bool(y) for y in labels]
    if len(probs) != len(labels):
        raise DriftError(
            f"ECE needs one label per probability, got {len(probs)} and {len(labels)}"
        )
    if any(not 0.0 <= p <= 1.0 for p in probs):
        raise DriftError("ECE needs probabilities in [0, 1]")
    if n_bins < 1:
        raise DriftError(f"n_bins {n_bins} must be at least 1")
    bins: list = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        index = min(int(p * n_bins), n_bins - 1)
        bins[index].append((p, y))
    total = len(probs)
    error = 0.0
    for rows in bins:
        if not rows:
            continue
        confidence = sum(p for p, _ in rows) / len(rows)
        accuracy = sum(1 for _, y in rows if y) / len(rows)
        error += len(rows) / total * abs(confidence - accuracy)
    return error


def head_drift(pretrained_probs, current_probs, labels, *, n_bins: int = 10) -> dict:
    pretrained = _validated("pretrained_probs", pretrained_probs)
    current = _validated("current_probs", current_probs)
    labels = [bool(y) for y in labels]
    if not (len(pretrained) == len(current) == len(labels)):
        raise DriftError(
            "the three inputs must be one row per held-out block, got "
            f"{len(pretrained)}, {len(current)}, {len(labels)}"
        )
    if not any(labels) or all(labels):
        raise DriftError(
            "the held-out set has a single class, so AUROC is undefined and the "
            "guard cannot certify anything"
        )
    return {
        "n": len(labels),
        "rank_correlation": spearman(pretrained, current),
        "median_pretrained": float(median(pretrained)),
        "median_current": float(median(current)),
        "ece_pretrained": expected_calibration_error(pretrained, labels, n_bins=n_bins),
        "ece_current": expected_calibration_error(current, labels, n_bins=n_bins),
        "auroc_pretrained": auc(pretrained, labels),
        "auroc_current": auc(current, labels),
    }


def violations(report, thresholds: DriftThresholds = DEFAULT_THRESHOLDS) -> list[str]:
    out = []
    if report["rank_correlation"] < thresholds.min_rank_correlation:
        out.append(
            f"rank correlation {report['rank_correlation']:.3f} below "
            f"{thresholds.min_rank_correlation}: the policy is consuming a "
            "different ordering than the one that was validated"
        )
    if report["median_current"] > thresholds.max_median_uncertainty:
        out.append(
            f"median uncertainty {report['median_current']:.3f} above "
            f"{thresholds.max_median_uncertainty} (pretrained "
            f"{report['median_pretrained']:.3f}): the head is saturated and "
            "carries no information whatever its AUROC says"
        )
    ratio_broken = (
        report["ece_current"] > thresholds.max_ece_ratio * report["ece_pretrained"]
    )
    margin_broken = (
        report["ece_current"] - report["ece_pretrained"]
        > thresholds.min_ece_absolute_margin
    )
    if ratio_broken and margin_broken:
        out.append(
            f"ECE {report['ece_current']:.3f} against a pretrained "
            f"{report['ece_pretrained']:.3f}: past both the "
            f"{thresholds.max_ece_ratio}x ratio and the "
            f"+{thresholds.min_ece_absolute_margin} margin, so the number has "
            "stopped being a probability the break-evens can be read against"
        )
    return out


def assert_head_healthy(report, thresholds: DriftThresholds = DEFAULT_THRESHOLDS):
    broken = violations(report, thresholds)
    if broken:
        raise DriftError(
            "the unfrozen head has drifted past its registered limits, so any "
            "result from here on cannot be attributed:\n  - "
            + "\n  - ".join(broken)
        )
    return report
