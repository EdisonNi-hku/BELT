from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, TextIO

from rl_controller.scripts.score_scaleup_s1 import CHAIN_FIELDS, PROBE_FIELDS

REQUIRED_FIELDS = frozenset(CHAIN_FIELDS) - {"completion_ids"}
SCORE_FIELDS = frozenset({"q", "v"})
DRAW_FIELDS = frozenset({"draw", "forced", "forced_tokens"})
OPTIONAL_PROBE_FIELDS = frozenset({"draws"})
OUTCOMES = ("natural", "length")


def is_correct(answer: Any, gold: Any) -> int:
    return int(answer is not None and type(answer) is type(gold) and answer == gold)


def is_trainable(chain: dict) -> bool:
    return bool(chain["probes"]) and bool(chain.get("block_ends")) \
        and len(chain["probes"]) == len(chain["query_points"])


def _is_forced_answer(value: Any) -> bool:
    return value is None or isinstance(value, str) \
        or (isinstance(value, int) and not isinstance(value, bool))


def _is_token_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_draws(identity: tuple, probe: dict, expected_k: int | None) -> int:
    draws = probe["draws"]
    if not isinstance(draws, list):
        raise ValueError(f"chain {identity} probe {probe['block']} draws must be a list")
    if len(draws) < 2:
        raise ValueError(f"chain {identity} probe {probe['block']} draws needs K >= 2, "
                         f"got {len(draws)} (omit `draws` for a single-draw probe)")
    for draw in draws:
        if not isinstance(draw, dict) or set(draw) != DRAW_FIELDS:
            raise ValueError(f"chain {identity} probe {probe['block']} draw fields must be "
                             f"exactly {sorted(DRAW_FIELDS)}")
        if not _is_forced_answer(draw["forced"]):
            raise ValueError(f"chain {identity} probe {probe['block']} draw forced must be "
                             f"None, int, or str, got {type(draw['forced']).__name__}")
        if not _is_token_count(draw["forced_tokens"]):
            raise ValueError(f"chain {identity} probe {probe['block']} draw forced_tokens must "
                             f"be an int >= 0, got {draw['forced_tokens']!r}")
    if [d["draw"] for d in draws] != list(range(len(draws))):
        raise ValueError(f"chain {identity} probe {probe['block']} draw indices must be "
                         f"0..{len(draws) - 1} in order")
    head = draws[0]
    if not (type(head["forced"]) is type(probe["forced"]) and head["forced"] == probe["forced"]
            and head["forced_tokens"] == probe["forced_tokens"]):
        raise ValueError(f"chain {identity} probe {probe['block']} draw 0 must copy the "
                         f"probe's forced/forced_tokens")
    if expected_k is not None and len(draws) != expected_k:
        raise ValueError(f"chain {identity} draws K must be uniform across probes: "
                         f"{expected_k} vs {len(draws)} at block {probe['block']}")
    return len(draws)


def validate_chain(row: dict) -> dict:
    identity = (row.get("qkey"), row.get("rep"))
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        raise ValueError(f"chain {identity} missing fields {sorted(missing)}")
    if row["outcome"] not in OUTCOMES:
        raise ValueError(f"chain {identity} outcome {row['outcome']!r} not in {OUTCOMES}")
    probes = row["probes"]
    if not isinstance(probes, list):
        raise ValueError(f"chain {identity} probes must be a list")
    last_block = 0
    draws_k: int | None = None
    n_with_draws = 0
    for probe in probes:
        if not isinstance(probe, dict) or set(probe) - OPTIONAL_PROBE_FIELDS != PROBE_FIELDS:
            raise ValueError(f"chain {identity} probe fields must be exactly "
                             f"{sorted(PROBE_FIELDS)} (optionally draws)")
        if probe["block"] <= last_block:
            raise ValueError(f"chain {identity} probes must be strictly increasing in block")
        last_block = probe["block"]
        if probe["prefix_len"] > row["thinking_tokens"]:
            raise ValueError(f"chain {identity} probe prefix {probe['prefix_len']} exceeds "
                             f"thinking_tokens {row['thinking_tokens']}")
        if "draws" in probe:
            draws_k = _validate_draws(identity, probe, draws_k)
            n_with_draws += 1
    if n_with_draws not in (0, len(probes)):
        raise ValueError(f"chain {identity} draws must be present on every probe or on none "
                         f"({n_with_draws} of {len(probes)})")
    scores = row.get("scores")
    if scores is not None:
        if not isinstance(scores, list) or len(scores) != len(probes):
            raise ValueError(f"chain {identity} scores/probes cardinality mismatch")
        for score in scores:
            if not isinstance(score, dict) or not SCORE_FIELDS <= set(score):
                raise ValueError(f"chain {identity} score fields must include {sorted(SCORE_FIELDS)}")
            for key, value in score.items():
                values = value if isinstance(value, list) else [value]
                if not values or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                     or not math.isfinite(x) for x in values):
                    raise ValueError(f"chain {identity} score {key!r} must be finite numbers")
    if bool(row["correct"]) is not bool(is_correct(row["answer"], row["gold"])):
        raise ValueError(f"chain {identity} correct contradicts exact answer/gold equality")
    return row


def probe_draws(probe: dict) -> list[dict]:
    draws = probe.get("draws")
    if draws is not None:
        return draws
    return [{"draw": 0, "forced": probe["forced"], "forced_tokens": probe["forced_tokens"]}]


def draw_count(chain: dict) -> int:
    probes = chain["probes"]
    if not probes:
        return 0
    return len(probe_draws(probes[0]))


def _load_chains_from_handle(handle: TextIO, keep_first_duplicate=False, dropped=None) -> list[dict]:
    rows = []
    seen: set[tuple] = set()
    for line in handle:
        if not line.strip():
            continue
        chain = validate_chain(json.loads(line))
        identity = (chain["qkey"], chain["rep"])
        if identity in seen:
            if not keep_first_duplicate:
                raise ValueError(f"duplicate chain identity {identity}")
            if dropped is not None:
                dropped.append(identity)
            continue
        seen.add(identity)
        rows.append(chain)
    return rows


def load_chains(path: str | os.PathLike[str] | TextIO, *,
                keep_first_duplicate: bool = False, dropped: list | None = None) -> list[dict]:
    if not isinstance(path, (str, os.PathLike)):
        return _load_chains_from_handle(path, keep_first_duplicate, dropped)
    with open(path, encoding="utf-8") as handle:
        return _load_chains_from_handle(handle, keep_first_duplicate, dropped)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
