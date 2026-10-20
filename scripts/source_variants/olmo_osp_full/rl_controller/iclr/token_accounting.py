from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable, Mapping, Sequence


class AccountingError(ValueError):
    pass


def program_key(record: Mapping) -> tuple:
    return (record["qkey"], int(record["rep"]), record["kind"], record["block"])


def index_programs(records: Iterable[Mapping]) -> dict[tuple, dict]:
    out = {}
    for raw in records:
        row = dict(raw)
        key = program_key(row)
        if key in out:
            raise AccountingError(f"duplicate program sidecar key {key}")
        tokens = row.get("tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise AccountingError(f"invalid program token count for {key}")
        out[key] = row
    return out


def _natural_answer_tokens(
    chain: Mapping,
    *,
    answer_format: str,
    force_natural_answer: bool,
    encode: Callable[[str], Sequence[int]] | None,
    programs: Mapping[tuple, Mapping] | None,
) -> int:
    if chain["outcome"] != "natural":
        return 0
    if answer_format == "code":
        key = (chain["qkey"], int(chain["rep"]), "natural", None)
        try:
            record = programs[key] if programs is not None else None
        except KeyError:
            record = None
        if record is None:
            raise AccountingError(f"natural code chain {key[:2]} has no program sidecar")
        if int(record["prefix_len"]) != int(chain["thinking_tokens"]):
            raise AccountingError(f"program sidecar prefix mismatch for {key[:2]}")
        return int(record["tokens"])
    if answer_format != "boxed":
        raise AccountingError(f"unsupported answer_format {answer_format!r}")
    if not force_natural_answer:
        return int(chain.get("natural_fallback_tokens", 0))
    if chain.get("answer") is None:
        raise AccountingError("forced natural answer is missing from an answered chain")
    if encode is None:
        raise AccountingError("forced boxed natural accounting requires the pinned tokenizer")
    return len(encode(f"{chain['answer']}}}"))


def deployment_cost(
    chain: Mapping,
    *,
    probe: Mapping | None = None,
    answer_format: str,
    force_natural_answer: bool = False,
    encode: Callable[[str], Sequence[int]] | None = None,
    programs: Mapping[tuple, Mapping] | None = None,
) -> dict[str, int]:
    if probe is None:
        thinking = int(chain["thinking_tokens"])
        answer = _natural_answer_tokens(
            chain,
            answer_format=answer_format,
            force_natural_answer=force_natural_answer,
            encode=encode,
            programs=programs,
        )
    else:
        thinking = int(probe["prefix_len"])
        answer = int(probe["forced_tokens"])
    if thinking < 0 or answer < 0:
        raise AccountingError("token counts must be nonnegative")
    generated = thinking + answer
    return {
        "thinking_tokens": thinking,
        "answer_tokens": answer,
        "generated_tokens": generated,
        "candidate_tokens_generated": generated,
    }


def audit_natural_answer_total(
    chains: Iterable[Mapping],
    *,
    summaries: Iterable[Mapping],
    answer_format: str,
    force_natural_answer: bool = False,
    encode: Callable[[str], Sequence[int]] | None = None,
    programs: Mapping[tuple, Mapping] | None = None,
) -> dict[str, int]:
    natural = [row for row in chains if row["outcome"] == "natural"]
    reconstructed = sum(
        _natural_answer_tokens(
            row,
            answer_format=answer_format,
            force_natural_answer=force_natural_answer,
            encode=encode,
            programs=programs,
        )
        for row in natural
    )
    recorded_requests = 0
    recorded_tokens = 0
    for summary in summaries:
        meta = summary["meta"]
        if answer_format == "code":
            recorded_requests += int(meta.get("natural_requests", 0))
        elif force_natural_answer:
            recorded_requests += int(meta.get("forced_natural_requests", 0))
        recorded_tokens += int(meta.get("natural_answer_tokens_total", 0))
    if recorded_requests != len(natural):
        raise AccountingError(
            f"summary request count {recorded_requests} != natural rows {len(natural)}"
        )
    if recorded_tokens != reconstructed:
        raise AccountingError(
            f"summary total {recorded_tokens} != reconstructed answer tokens {reconstructed}"
        )
    return {
        "natural_rows": len(natural),
        "answer_tokens": reconstructed,
        "summary_requests": recorded_requests,
        "summary_answer_tokens": recorded_tokens,
    }


def write_natural_costs(
    path: str | Path,
    chains: Iterable[Mapping],
    *,
    answer_format: str,
    force_natural_answer: bool = False,
    encode=None,
    programs=None,
) -> None:
    source = sorted((dict(row) for row in chains), key=lambda row: (row["qkey"], row["rep"]))
    seen = set()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                         dir=target.parent, prefix=f".{target.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for chain in source:
                key = (chain["qkey"], int(chain["rep"]))
                if key in seen:
                    raise AccountingError(f"duplicate natural-cost identity {key}")
                seen.add(key)
                cost = deployment_cost(
                    chain,
                    answer_format=answer_format,
                    force_natural_answer=force_natural_answer,
                    encode=encode,
                    programs=programs,
                )
                row = {"qkey": key[0], "rep": key[1],
                       "thinking_tokens": cost["thinking_tokens"],
                       "answer_tokens": cost["answer_tokens"]}
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
