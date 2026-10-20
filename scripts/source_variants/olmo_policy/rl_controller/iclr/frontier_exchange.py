from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = "iclr-acc-token-frontier/v1"
CONSUMED_SCHEMA_VERSION = SCHEMA_VERSION

IDENTITY_FIELDS = (
    "schema_version", "run_id", "method_family", "method_name", "config_id",
    "source_branch", "source_commit", "source_artifact", "source_sha256",
    "model_id", "model_revision", "tokenizer_revision", "dataset", "split",
    "qkey", "rep", "seed", "evaluation_unit", "sample_count",
    "prompt_contract_id", "grader_id", "selection_split", "selection_rule",
    "frozen_config",
)
OUTCOME_FIELDS = (
    "correct", "selected_answer", "thinking_tokens", "answer_tokens",
    "generated_tokens", "prompt_tokens", "aux_generated_tokens",
    "model_forward_passes", "scorer_forward_passes", "tflops", "wall_time_s",
    "candidate_tokens_generated", "missing_reason",
)
ROW_FIELDS = IDENTITY_FIELDS + OUTCOME_FIELDS
ANNEX_FIELDS = ("seed_semantics",)
SEED_SEMANTICS = ("training", "replay")
CONSUMED_FIELDS = (
    "schema_version", "run_id", "config_id", "qkey", "sample_index", "rep",
    "chain_sha256", "answer", "correct", "thinking_tokens", "answer_tokens",
    "generated_tokens",
)

COMPARABILITY_FIELDS = (
    "model_id", "model_revision", "tokenizer_revision", "dataset", "split",
    "prompt_contract_id", "grader_id", "evaluation_unit",
)


class SchemaError(ValueError):
    pass


def _exact_fields(row: Mapping, expected: Sequence[str], kind: str) -> None:
    keys = set(row)
    missing = set(expected) - keys
    unknown = keys - set(expected)
    if missing:
        raise SchemaError(f"{kind} missing fields: {sorted(missing)}")
    if unknown:
        raise SchemaError(f"{kind} unknown fields: {sorted(unknown)}")


def _nonnegative_int(row: Mapping, name: str, *, nullable: bool = False) -> None:
    value = row[name]
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{name} must be a nonnegative integer")


def _finite_nonnegative(row: Mapping, name: str) -> None:
    value = row[name]
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{name} must be numeric or null")
    if not math.isfinite(value) or value < 0:
        raise SchemaError(f"{name} must be finite and nonnegative")


def validate_row(row: Mapping) -> dict:
    _exact_fields({k: v for k, v in row.items() if k not in ANNEX_FIELDS},
                  ROW_FIELDS, "frontier row")
    out = dict(row)
    if "seed_semantics" in out and out["seed_semantics"] not in SEED_SEMANTICS:
        raise SchemaError(f"seed_semantics must be one of {SEED_SEMANTICS}, "
                          f"got {out['seed_semantics']!r}")
    if out["schema_version"] != SCHEMA_VERSION:
        raise SchemaError(f"unsupported schema_version: {out['schema_version']!r}")
    if out["evaluation_unit"] not in {"chain", "question"}:
        raise SchemaError("evaluation_unit must be 'chain' or 'question'")
    _nonnegative_int(out, "sample_count")
    if out["sample_count"] < 1:
        raise SchemaError("sample_count must be positive")
    if out["evaluation_unit"] == "chain":
        if out["sample_count"] != 1:
            raise SchemaError("chain sample_count must equal 1")
        if out["rep"] is None:
            raise SchemaError("chain rep must not be null")
    elif out["rep"] is not None:
        raise SchemaError("question rep must be null; reps live in consumed samples")

    if not isinstance(out["frozen_config"], dict):
        raise SchemaError("frozen_config must be a JSON object")
    if out["missing_reason"] is not None:
        if not isinstance(out["missing_reason"], str) or not out["missing_reason"]:
            raise SchemaError("missing_reason must be a nonempty string or null")
        if any(out[name] is not None for name in OUTCOME_FIELDS if name != "missing_reason"):
            raise SchemaError("missing rows must set all outcome and cost fields to null")
        return out
    if not isinstance(out["correct"], bool):
        raise SchemaError("correct must be bool for a valid row")

    for name in ("thinking_tokens", "answer_tokens", "generated_tokens",
                 "aux_generated_tokens", "candidate_tokens_generated"):
        _nonnegative_int(out, name)
    for name in ("prompt_tokens", "model_forward_passes", "scorer_forward_passes"):
        _nonnegative_int(out, name, nullable=True)
    for name in ("tflops", "wall_time_s"):
        _finite_nonnegative(out, name)

    if out["evaluation_unit"] == "chain":
        expected = out["thinking_tokens"] + out["answer_tokens"]
        if out["generated_tokens"] != expected:
            raise SchemaError("generated_tokens must equal thinking_tokens + answer_tokens")
        if out["candidate_tokens_generated"] != expected:
            raise SchemaError("candidate_tokens_generated must equal chain generated tokens")
        if out["aux_generated_tokens"] != 0:
            raise SchemaError("chain rows with auxiliary generation require a question audit")
    return out


def validate_consumed_sample(row: Mapping) -> dict:
    _exact_fields(row, CONSUMED_FIELDS, "consumed-sample row")
    out = dict(row)
    if out["schema_version"] != CONSUMED_SCHEMA_VERSION:
        raise SchemaError(f"unsupported consumed schema_version: {out['schema_version']!r}")
    for name in ("sample_index", "rep", "thinking_tokens", "answer_tokens",
                 "generated_tokens"):
        _nonnegative_int(out, name)
    if (not isinstance(out["chain_sha256"], str) or len(out["chain_sha256"]) != 64
            or any(ch not in "0123456789abcdef" for ch in out["chain_sha256"])):
        raise SchemaError("chain_sha256 must be 64 lowercase hex characters")
    if out["correct"] is not None and not isinstance(out["correct"], bool):
        raise SchemaError("consumed correct must be bool or null")
    if out["generated_tokens"] != out["thinking_tokens"] + out["answer_tokens"]:
        raise SchemaError("consumed generated_tokens must equal thinking_tokens + answer_tokens")
    return out


def validate_question_audit(row: Mapping, samples: Iterable[Mapping]) -> dict:
    aggregate = validate_row(row)
    if aggregate["evaluation_unit"] != "question":
        raise SchemaError("consumed-sample audit requires evaluation_unit=question")
    checked = [validate_consumed_sample(sample) for sample in samples]
    if len(checked) != aggregate["sample_count"]:
        raise SchemaError("sample_count does not match consumed samples")
    indexes = [sample["sample_index"] for sample in checked]
    if sorted(indexes) != list(range(len(checked))) or len(set(indexes)) != len(indexes):
        raise SchemaError("sample_index must be unique and contiguous from zero")
    join = (aggregate["run_id"], aggregate["config_id"], aggregate["qkey"])
    for sample in checked:
        sample_join = (sample["run_id"], sample["config_id"], sample["qkey"])
        if sample_join != join:
            raise SchemaError("consumed sample does not match aggregate join key")
    thinking = sum(sample["thinking_tokens"] for sample in checked)
    answers = sum(sample["answer_tokens"] for sample in checked)
    candidates = sum(sample["generated_tokens"] for sample in checked)
    if aggregate["thinking_tokens"] != thinking or aggregate["answer_tokens"] != answers:
        raise SchemaError("aggregate thinking/answer tokens do not match consumed samples")
    if aggregate["candidate_tokens_generated"] != candidates:
        raise SchemaError("candidate_tokens_generated does not match consumed samples")
    expected = candidates + aggregate["aux_generated_tokens"]
    if aggregate["generated_tokens"] != expected:
        raise SchemaError("generated_tokens must equal candidates plus auxiliary generation")
    return aggregate


def dump_rows(path: str | Path, rows: Iterable[Mapping]) -> dict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    digest = hashlib.sha256()
    count = 0
    try:
        with tempfile.NamedTemporaryFile("wb", dir=target.parent,
                                         prefix=f".{target.name}.", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            for row in rows:
                checked = validate_row(row)
                encoded = (json.dumps(
                    checked, ensure_ascii=False, separators=(",", ":"),
                    sort_keys=True, allow_nan=False,
                ) + "\n").encode("utf-8")
                handle.write(encoded)
                digest.update(encoded)
                count += 1
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"rows": count, "sha256": digest.hexdigest()}


def load_rows(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"invalid JSON on line {lineno}: {exc.msg}") from exc
            if not isinstance(raw, dict):
                raise SchemaError(f"line {lineno} is not a JSON object")
            try:
                rows.append(validate_row(raw))
            except SchemaError as exc:
                raise SchemaError(f"line {lineno}: {exc}") from exc
    return rows


def dump_consumed_samples(path: str | Path, rows: Iterable[Mapping]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                         dir=target.parent, prefix=f".{target.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for row in rows:
                checked = validate_consumed_sample(row)
                handle.write(json.dumps(checked, ensure_ascii=False, separators=(",", ":"),
                                        sort_keys=True, allow_nan=False))
                handle.write("\n")
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _single_value(rows: Sequence[Mapping], field: str):
    values = {row[field] for row in rows}
    return next(iter(values)) if len(values) == 1 else values


def comparability_report(left: Sequence[Mapping], right: Sequence[Mapping]) -> dict:
    if not left or not right:
        raise SchemaError("comparability requires nonempty row sets")
    left_checked = [validate_row(row) for row in left]
    right_checked = [validate_row(row) for row in right]
    mismatches = {}
    for field in COMPARABILITY_FIELDS:
        lv = _single_value(left_checked, field)
        rv = _single_value(right_checked, field)
        if lv != rv:
            mismatches[field] = {"left": sorted(lv) if isinstance(lv, set) else lv,
                                 "right": sorted(rv) if isinstance(rv, set) else rv}
    left_qkeys = {row["qkey"] for row in left_checked if row["missing_reason"] is None}
    right_qkeys = {row["qkey"] for row in right_checked if row["missing_reason"] is None}
    intersection = left_qkeys & right_qkeys
    if left_qkeys != right_qkeys:
        mismatches["qkey_set"] = {
            "left_only": len(left_qkeys - right_qkeys),
            "right_only": len(right_qkeys - left_qkeys),
        }
    return {
        "comparable": not mismatches,
        "mismatches": mismatches,
        "left_qkeys": len(left_qkeys),
        "right_qkeys": len(right_qkeys),
        "intersection_qkeys": len(intersection),
        "left_missing_qkeys": len(right_qkeys - left_qkeys),
        "right_missing_qkeys": len(left_qkeys - right_qkeys),
    }
