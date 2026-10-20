from __future__ import annotations

import math

from rl_controller.dense_reprs import position_features

TARGETS = ("determined", "harmless", "futile")

FEATURE_SETS = ("R", "RP", "P")

_LOGIT_CLAMP = 1e-6


class LeakageError(RuntimeError):
    pass


def _require_bool(name, value):
    if not isinstance(value, bool):
        raise ValueError(
            f"{name} must be a bool, got {value!r}. There is no safe default: "
            "treating an unknown natural outcome as False would relabel every "
            "doomed trace as `do not stop`."
        )
    return value


def harmless(*, determined, natural_correct) -> bool:
    _require_bool("determined", determined)
    _require_bool("natural_correct", natural_correct)
    return bool(determined or not natural_correct)


def harmful(*, determined, natural_correct) -> bool:
    return not harmless(determined=determined, natural_correct=natural_correct)


def assert_disjoint_questions(train_questions, dev_questions) -> None:
    shared = set(train_questions) & set(dev_questions)
    if shared:
        raise LeakageError(
            f"{len(shared)} question(s) appear in both splits, e.g. "
            f"{sorted(shared, key=repr)[:5]}. Under this target that leaks the "
            "label: `natural_correct` is a question-correlated property."
        )


def features_for(feature_set, *, repr_vec, block, block_end_offsets):
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}")
    if feature_set == "R":
        return list(repr_vec)
    position = position_features(block, block_end_offsets)
    if feature_set == "P":
        return position
    return list(repr_vec) + position


def _logit(p):
    p = min(max(float(p), _LOGIT_CLAMP), 1.0 - _LOGIT_CLAMP)
    return math.log(p / (1.0 - p))


def stack(*, p_ready, p_futile, coefs):
    a, b, c = coefs
    z = a * _logit(p_ready) + b * _logit(p_futile) + c
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)
