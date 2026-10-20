from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping

from rl_controller.futility import LeakageError, assert_disjoint_questions

__all__ = [
    "LeakageError",
    "SPLITS",
    "assert_disjoint_questions",
    "assign_split",
    "question_key",
    "split_counts",
    "three_way_split",
]

SPLITS = ("fit", "select", "report")
DEFAULT_SHARES = (0.5, 0.25, 0.25)

_KEY_RE = re.compile(r"\A[0-9a-f]{16}\Z")


def question_key(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(
            f"question_key needs the question TEXT, got {type(text).__name__}. "
            "An index is not a question identity here: the pools' index ranges "
            "collide completely."
        )
    return hashlib.sha256(
        re.sub(r"\s+", " ", text).strip().lower().encode("utf-8")
    ).hexdigest()[:16]


def _require_question_keys(keys: Iterable) -> list[str]:
    out = []
    for key in keys:
        if not isinstance(key, str) or not _KEY_RE.match(key):
            raise TypeError(
                f"{key!r} is not a question_key digest. Split assignment must "
                "be keyed on normalised question text, never on "
                "`original_index` -- both pools number from 0 and their index "
                "ranges collide, so an index cannot distinguish questions "
                "across pools."
            )
        out.append(key)
    return out


def _unit_interval(key: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def assign_split(key: str, *, salt: str, shares: tuple = DEFAULT_SHARES) -> str:
    _require_question_keys([key])
    if len(shares) != len(SPLITS):
        raise ValueError(f"need {len(SPLITS)} shares, got {len(shares)}")
    if any(s < 0 for s in shares):
        raise ValueError(f"negative share in {shares!r}")
    total = sum(shares)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"shares must sum to 1, got {total!r}")
    position = _unit_interval(key, salt)
    cumulative = 0.0
    for name, share in zip(SPLITS, shares):
        cumulative += share
        if position < cumulative:
            return name
    return SPLITS[-1]


def three_way_split(
    keys: Iterable[str], *, salt: str, shares: tuple = DEFAULT_SHARES
) -> dict[str, str]:
    unique = sorted(set(_require_question_keys(keys)))
    if not unique:
        raise ValueError("no questions to split")
    assignment = {key: assign_split(key, salt=salt, shares=shares) for key in unique}
    covered = set()
    for name in SPLITS:
        part = {k for k, v in assignment.items() if v == name}
        if part & covered:
            raise LeakageError(f"split {name} overlaps an earlier split")
        covered |= part
    if covered != set(unique):
        raise LeakageError("split does not cover every question")
    return assignment


def split_counts(assignment: Mapping[str, str]) -> dict[str, int]:
    counts = {name: 0 for name in SPLITS}
    for value in assignment.values():
        if value not in counts:
            raise ValueError(f"unknown split {value!r}")
        counts[value] += 1
    return counts
