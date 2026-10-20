from __future__ import annotations

import json
from pathlib import Path

from rl_controller.iclr.chains import (REQUIRED_FIELDS, SCORE_FIELDS, draw_count, is_correct,
                                       is_trainable, probe_draws, validate_chain, write_jsonl)
from rl_controller.iclr.chains import load_chains as load_scored_chains

SCORED_FIELDS = frozenset(REQUIRED_FIELDS) | {"scores"}
SCORED_ORDER = ("qkey", "rep", "gold", "outcome", "answer", "correct", "blocks",
                "thinking_tokens", "block_ends", "query_points", "probes", "scores")

_TRAINER_EXPORTS = (
    "build_prompt_table",
    "dump_scored",
    "load_chains",
    "prepare_chains",
    "q_labels",
    "readout_values",
    "build_eval_record",
    "collect_scores",
    "SCORED_KEEP",
)

__all__ = (
    "SCORED_FIELDS", "SCORED_ORDER", "REQUIRED_FIELDS", "SCORE_FIELDS",
    "load_scored_chains", "validate_chain", "write_jsonl", "is_correct", "is_trainable",
    "probe_draws", "draw_count", "write_scored", "validated_rows", "summary_stub",
    "write_run_dir",
    *_TRAINER_EXPORTS,
)


def __getattr__(name: str):
    if name in _TRAINER_EXPORTS:
        from rl_controller.scripts import single_cot_train

        value = getattr(single_cot_train, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


def write_scored(path, chains) -> int:
    rows = validated_rows(chains)
    write_jsonl(path, rows)
    return len(rows)


def validated_rows(chains) -> list[dict]:
    rows = []
    for chain in chains:
        identity = (chain.get("qkey"), chain.get("rep"))
        missing = [k for k in SCORED_ORDER if k not in chain]
        if missing:
            raise ValueError(f"chain {identity} lacks {missing}")
        if len(chain["scores"]) != len(chain["probes"]):
            raise ValueError(
                f"chain {identity}: {len(chain['scores'])} scores for "
                f"{len(chain['probes'])} probes -- the ladder zips them, so a short scores "
                "list silently truncates the rule instead of failing")
        row = {k: chain[k] for k in SCORED_ORDER}
        try:
            validate_chain(row)
        except ValueError as exc:
            raise ValueError(f"chain {identity} would not load back into the ladder: {exc}") \
                from None
        rows.append(row)
    return rows


def summary_stub(*, seed: int, targets, method: str, provenance: dict | None = None,
                 **extra) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")
    targets = [targets] if isinstance(targets, str) else list(targets)
    if targets[:2] != ["q", "v"]:
        raise ValueError(f"targets must start with q, v (the ladder's mandatory beliefs), "
                         f"got {targets!r}")
    return {"config": {"seed": seed, "targets": targets, "method": method, **extra},
            "provenance": provenance or {}}


def write_run_dir(out_dir, *, val, select, summary: dict) -> dict:
    if "config" not in summary or "seed" not in summary["config"]:
        raise ValueError("summary needs config.seed -- iclr_ladder.load_run exits without it")
    val, select = list(val), list(select)
    if not val or not select:
        raise ValueError(f"both splits must be non-empty (val {len(val)}, select {len(select)})")
    val_rows = validated_rows(val)
    select_rows = validated_rows(select)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "val_scored.jsonl", val_rows)
    write_jsonl(out / "select_scored.jsonl", select_rows)
    n_val, n_select = len(val_rows), len(select_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    return {"path": str(out), "n_val": n_val, "n_select": n_select}
