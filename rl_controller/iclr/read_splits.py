from __future__ import annotations

from typing import Iterable

REGISTRY_ID = "rl_controller.iclr.read_splits.READ_SPLITS"

QKEY_SHA256_CONVENTION = (
    "sha256 of the pool's sorted unique qkeys, joined by newline, with a "
    "trailing newline")

READ_SPLITS: dict[str, dict] = {
    "select": {
        "n_questions": 1065,
        "qkeys_sha256": "c1cdb013c79421a199a785674471e2ed73df6cd2faaf2b0018bfda52f237f3cb",
    },
    "math_ood_report": {
        "n_questions": 951,
        "qkeys_sha256": "e1ff3b12a1f9d7cdf9ef3d29f2bdd07bb4ff2d29a01d60b60cc210955fc2511e",
    },
    "math_t1_report": {
        "n_questions": 841,
        "qkeys_sha256": "12b343cd482158fdcc4a4c51bbcde9479f68d5c850a24bedf6577260d51c1e62",
    },
    "math_t2_report": {
        "n_questions": 853,
        "qkeys_sha256": "5fde9b465557c8b3c71384c3d434af9a5660335bbf40e5b4da7a30a12335c047",
    },
    "ood2_math_report": {
        "n_questions": 2430,
        "qkeys_sha256": "66c2fa8583769dbae5655fb0f15cfea951e7294699e168a8450b6e90ab2fbb80",
    },
}


def qkeys_sha256(qkeys: Iterable[str]) -> str:
    from rl_controller.iclr.banked_baselines import qkey_sha256

    return qkey_sha256(qkeys)


def is_registered(pool) -> bool:
    return isinstance(pool, str) and pool in READ_SPLITS


def registered_read_split(pool) -> dict:
    if not isinstance(pool, str) or not pool.strip():
        raise ValueError("pool must be a non-empty str")
    entry = READ_SPLITS.get(pool.strip())
    if entry is None:
        raise ValueError(
            f"pool {pool!r} has no REGISTERED READ SPLIT. Registered pools are "
            f"{sorted(READ_SPLITS)}. A pool's population is two frozen numbers -- "
            f"n_questions and qkeys_sha256 ({QKEY_SHA256_CONVENTION}) -- discovered "
            "by letting the report compute the digest over the real tables and refuse, "
            "then registering what it observed in "
            f"{REGISTRY_ID}. Until that has happened there is nothing to verify an "
            "exported population against, and a manifest file is not a substitute: a "
            "pool's parent manifest is a superset of its read split.")
    return dict(entry)
