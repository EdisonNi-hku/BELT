from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


REPLAY_SEED = -1

SCHEMA_VERSION = "iclr-acc-token-frontier/v1"
TTS_FIELDS = frozenset({
    "schema_version", "run_id", "method_family", "method_name", "config_id",
    "source_branch", "source_commit", "source_artifact", "source_sha256",
    "model_id", "model_revision", "tokenizer_revision", "dataset", "split",
    "qkey", "rep", "seed", "evaluation_unit", "sample_count",
    "prompt_contract_id", "grader_id", "selection_split", "selection_rule",
    "frozen_config", "correct", "selected_answer", "thinking_tokens",
    "answer_tokens", "generated_tokens", "prompt_tokens", "aux_generated_tokens",
    "candidate_tokens_generated", "model_forward_passes", "scorer_forward_passes",
    "tflops", "wall_time_s", "missing_reason",
})
CHILD_FIELDS = frozenset({
    "schema_version", "run_id", "config_id", "qkey", "sample_index", "rep",
    "chain_sha256", "answer", "correct", "thinking_tokens", "answer_tokens",
    "generated_tokens",
})
COMPARABILITY_FIELDS = frozenset({
    "method_family", "method_name", "config_id", "feasibility_tier", "model_id",
    "model_revision", "tokenizer_revision", "thinking_mode", "prompt_contract_id",
    "qkey_set", "grader_id", "evaluation_unit", "samples_per_question", "max_budget",
    "generation_parameters", "token_definition", "scorer_aux_cost", "validation_source",
    "frozen_config", "status", "reason",
})

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_PARENT_STRINGS = frozenset({
    "run_id", "method_family", "method_name", "config_id", "source_branch",
    "source_artifact", "model_id", "dataset", "split", "qkey", "prompt_contract_id",
    "grader_id", "selection_split", "selection_rule",
})
_OPTIONAL_INT_FIELDS = frozenset({
    "prompt_tokens", "aux_generated_tokens", "candidate_tokens_generated",
    "model_forward_passes", "scorer_forward_passes",
})
_CORE_TOKEN_FIELDS = frozenset({"thinking_tokens", "answer_tokens", "generated_tokens"})
_FROZEN_MODEL = "Qwen/Qwen3-1.7B"
_FROZEN_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
_FROZEN_PROMPT = "qwen3-1.7b-native-thinking-v1"
_CHAIN_TOKEN_DEFINITION = "generated_tokens=thinking_tokens+answer_tokens"
_QUESTION_TOKEN_DEFINITION = (
    "candidate_tokens_generated=sum_children(thinking_tokens+answer_tokens);"
    "generated_tokens=candidate_tokens_generated+aux_generated_tokens"
)
_FROZEN_QKEY_SET = "iclr-p1-math-union12/select"
_FROZEN_GENERATION_PARAMETERS = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
_OUTCOME_COST_FIELDS = frozenset({
    "correct", "selected_answer", "thinking_tokens", "answer_tokens", "generated_tokens",
    "prompt_tokens", "aux_generated_tokens", "candidate_tokens_generated",
    "model_forward_passes", "scorer_forward_passes", "tflops", "wall_time_s",
})
_REGISTERED_COMMON = {
    "model_id": _FROZEN_MODEL,
    "model_revision": _FROZEN_REVISION,
    "tokenizer_revision": _FROZEN_REVISION,
    "thinking_mode": "native_on",
    "prompt_contract_id": _FROZEN_PROMPT,
    "qkey_set": _FROZEN_QKEY_SET,
    "max_budget": 12000,
    "generation_parameters": _FROZEN_GENERATION_PARAMETERS,
    "validation_source": "archived_head_validation",
}
_REGISTERED_METHOD_SPECS = {
    ("natural", "natural-v1"): {
        "method_family": "natural", "method_name": "natural", "config_id": "natural-v1",
        "feasibility_tier": "R0", "grader_id": "archived_math_equal", "evaluation_unit": "chain",
        "samples_per_question": 1, "scorer_aux_cost": "none", "frozen_config": {"strategy": "natural"},
        "status": "comparable", "reason": None,
    },
    ("always_fire", "always-fire-v1"): {
        "method_family": "fixed_cap", "method_name": "always_fire", "config_id": "always-fire-v1",
        "feasibility_tier": "R0", "grader_id": "archived_math_equal", "evaluation_unit": "chain",
        "samples_per_question": 1, "scorer_aux_cost": "none", "frozen_config": {"strategy": "always_fire"},
        "status": "comparable", "reason": None,
    },
    ("consecutive_agreement", "consecutive-agreement-v1"): {
        "method_family": "agreement", "method_name": "consecutive_agreement", "config_id": "consecutive-agreement-v1",
        "feasibility_tier": "R0", "grader_id": "archived_math_equal", "evaluation_unit": "chain",
        "samples_per_question": 1, "scorer_aux_cost": "none", "frozen_config": {"rule": "consecutive_agreement"},
        "status": "comparable", "reason": None,
    },
    ("earliest_agreement", "earliest-agreement-v1"): {
        "method_family": "agreement", "method_name": "earliest_agreement", "config_id": "earliest-agreement-v1",
        "feasibility_tier": "R0", "grader_id": "archived_math_equal", "evaluation_unit": "chain",
        "samples_per_question": 1, "scorer_aux_cost": "none",
        "frozen_config": {"rule": "earliest_agreement", "offline_future_answer_anchor": True},
        "status": "faceted", "reason": "Uses an offline future-answer anchor rather than an online deployed signal.",
    },
    ("gold_free_bon", "bon-n1-grid-unselected-v1"): {
        "method_family": "best_of_n", "method_name": "gold_free_bon", "config_id": "bon-n1-grid-unselected-v1",
        "feasibility_tier": "R1", "grader_id": "archived_math_equal", "evaluation_unit": "question",
        "samples_per_question": 1, "scorer_aux_cost": "unresolved",
        "frozen_config": {"registered_base_n": 1, "n_grid": [1, 2, 4, 8], "n_selected": False,
                          "selector": "unresolved_gold_free"},
        "status": "incompatible", "reason": "No frozen gold-free archived selector is registered for R0 replay.",
    },
    ("deepconf_offline", "deepconf-offline-n1-grid-v1"): {
        "method_family": "deepconf", "method_name": "deepconf_offline", "config_id": "deepconf-offline-n1-grid-v1",
        "feasibility_tier": "R1", "grader_id": "archived_math_equal", "evaluation_unit": "question",
        "samples_per_question": 1, "scorer_aux_cost": "pending_exact_logprob_reconstruction",
        "frozen_config": {"registered_base_n": 1, "n_grid": [1, 2, 4, 8], "n_selected": False},
        "status": "incompatible", "reason": "Exact source and offline algorithm are not frozen for replay.",
    },
    ("deepconf_online", "deepconf-online-n1-grid-v1"): {
        "method_family": "deepconf", "method_name": "deepconf_online", "config_id": "deepconf-online-n1-grid-v1",
        "feasibility_tier": "R1", "grader_id": "archived_math_equal", "evaluation_unit": "question",
        "samples_per_question": 1, "scorer_aux_cost": "pending_exact_logprob_reconstruction",
        "frozen_config": {"registered_base_n": 1, "n_grid": [1, 2, 4, 8], "n_selected": False},
        "status": "incompatible", "reason": "Exact source and online algorithm are not frozen for replay.",
    },
    ("uncertainty_cot", "uncertainty-cot-v1"): {
        "method_family": "uncertainty", "method_name": "uncertainty_cot", "config_id": "uncertainty-cot-v1",
        "feasibility_tier": "R1", "grader_id": "archived_math_equal", "evaluation_unit": "chain",
        "samples_per_question": 1, "scorer_aux_cost": "pending_exact_algorithm", "frozen_config": {"algorithm": "unresolved"},
        "status": "incompatible", "reason": "Exact source and uncertainty algorithm are not frozen for replay.",
    },
    ("mur", "mur-v1"): {
        "method_family": "mur", "method_name": "mur", "config_id": "mur-v1", "feasibility_tier": "R2",
        "grader_id": "archived_math_equal", "evaluation_unit": "question", "samples_per_question": 1,
        "scorer_aux_cost": "branching_compute_pending", "frozen_config": {"branching": "may_require_unseen_continuations"},
        "status": "incompatible", "reason": "Exact source and MUR algorithm are not frozen for replay.",
    },
    ("extended_thinking", "extended-thinking-v1"): {
        "method_family": "extended_thinking", "method_name": "extended_thinking", "config_id": "extended-thinking-v1",
        "feasibility_tier": "R2", "grader_id": "archived_math_equal", "evaluation_unit": "chain",
        "samples_per_question": 1, "scorer_aux_cost": "none", "frozen_config": {"generation_change": "beyond_archived_ceiling"},
        "status": "incompatible", "reason": "Exact source and extended-thinking algorithm are not frozen for replay.",
    },
}
for _cap in (16, 32, 64, 96, 128):
    _REGISTERED_METHOD_SPECS[(f"cap{_cap}", f"cap{_cap}-v1")] = {
        "method_family": "fixed_cap", "method_name": f"cap{_cap}", "config_id": f"cap{_cap}-v1",
        "feasibility_tier": "R0", "grader_id": "archived_math_equal", "evaluation_unit": "chain",
        "samples_per_question": 1, "scorer_aux_cost": "none",
        "frozen_config": {"strategy": "stop_at_cap", "block_cap": _cap}, "status": "comparable", "reason": None,
    }
for _n in (1, 2, 4, 8):
    _REGISTERED_METHOD_SPECS[("self_consistency", f"sc-n{_n}-grid-unselected-v1")] = {
        "method_family": "self_consistency", "method_name": "self_consistency", "config_id": f"sc-n{_n}-grid-unselected-v1",
        "feasibility_tier": "R0", "grader_id": "archived_math_equal", "evaluation_unit": "question",
        "samples_per_question": _n, "scorer_aux_cost": "none",
        "frozen_config": {"strategy": "natural", "samples_per_question": _n,
                          "vote_rule": "integer_only_deterministic_plurality"},
        "status": "comparable", "reason": None,
    }


def _exact_fields(row: Any, fields: frozenset[str], label: str) -> dict:
    if not isinstance(row, dict):
        raise ValueError(f"{label} must be a JSON object")
    actual = set(row)
    if actual != fields:
        missing = sorted(fields - actual)
        extra = sorted((repr(key) for key in actual - fields))
        raise ValueError(f"{label} fields must be exact; missing={missing}, extra={extra}")
    return row


def _nonempty_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")


def _nonnegative_int(value: Any, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")


def _optional_nonnegative_int(value: Any, label: str) -> None:
    if value is not None:
        _nonnegative_int(value, label)


def _optional_nonnegative_number(value: Any, label: str) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite nonnegative number or null")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{label} must be a finite nonnegative number or null")
        return
    if not isinstance(value, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number or null")


def _json_value(value: Any, label: str) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if math.isfinite(value):
            return
        raise ValueError(f"{label} must contain only finite JSON numbers")
    if type(value) is list:
        for index, item in enumerate(value):
            _json_value(item, f"{label}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{label} object keys must be strings")
            _json_value(item, f"{label}.{key}")
        return
    raise ValueError(f"{label} must be a lossless native JSON value")


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(
            _strict_json_equal(actual[key], expected[key]) for key in expected)
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _strict_json_equal(value, expected[index]) for index, value in enumerate(actual))
    return actual == expected


def _matches_hex(value: Any, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _registered_expected_row(method_name: str, config_id: str) -> dict | None:
    spec = _REGISTERED_METHOD_SPECS.get((method_name, config_id))
    if spec is None:
        return None
    token_definition = (_CHAIN_TOKEN_DEFINITION if spec["evaluation_unit"] == "chain"
                        else _QUESTION_TOKEN_DEFINITION)
    return {**_REGISTERED_COMMON, "token_definition": token_definition, **spec}


def _validate_generated_identity(row: dict) -> None:
    thinking, answer, generated = (row[name] for name in ("thinking_tokens", "answer_tokens", "generated_tokens"))
    if thinking is not None and answer is not None and generated is not None \
            and generated != thinking + answer:
        raise ValueError("generated_tokens must equal thinking_tokens + answer_tokens")


def validate_tts_row(row: dict) -> dict:
    row = _exact_fields(row, TTS_FIELDS, "tts row")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    for field in _PARENT_STRINGS:
        _nonempty_string(row[field], field)
    if not _matches_hex(row["source_commit"], _HEX40):
        raise ValueError("source_commit must be a 40-hex git oid")
    if not _matches_hex(row["source_sha256"], _HEX64):
        raise ValueError("source_sha256 must be a 64-hex sha256")
    for field in ("model_revision", "tokenizer_revision"):
        if not _matches_hex(row[field], _HEX40):
            raise ValueError(f"{field} must be a 40-hex revision")
    if row["seed"] != REPLAY_SEED:
        _nonnegative_int(row["seed"], "seed")
    if isinstance(row["seed"], bool) or not isinstance(row["seed"], int):
        raise ValueError("seed must be an int")
    if not isinstance(row["evaluation_unit"], str) or row["evaluation_unit"] not in {"chain", "question"}:
        raise ValueError("evaluation_unit must be 'chain' or 'question'")
    _nonnegative_int(row["sample_count"], "sample_count")
    if row["evaluation_unit"] == "chain":
        _nonnegative_int(row["rep"], "rep")
        if row["sample_count"] != 1:
            raise ValueError("chain rows require sample_count == 1")
    elif row["rep"] is not None:
        raise ValueError("question rows require rep == null")
    elif row["sample_count"] < 1:
        raise ValueError("question rows require sample_count >= 1")
    if not isinstance(row["frozen_config"], dict):
        raise ValueError("frozen_config must be a JSON object")
    _json_value(row["frozen_config"], "frozen_config")
    missing = row["missing_reason"]
    if missing is not None:
        _nonempty_string(missing, "missing_reason")
        if any(row[field] is not None for field in _OUTCOME_COST_FIELDS):
            raise ValueError("missing rows must set all outcome and cost fields to null")
        return row
    if not isinstance(row["correct"], bool):
        raise ValueError("correct must be bool on a present row")
    for field in _CORE_TOKEN_FIELDS:
        _nonnegative_int(row[field], field)
    for field in _OPTIONAL_INT_FIELDS - {"candidate_tokens_generated"}:
        _optional_nonnegative_int(row[field], field)
    _nonnegative_int(row["candidate_tokens_generated"], "candidate_tokens_generated")
    _nonnegative_int(row["aux_generated_tokens"], "aux_generated_tokens")
    _optional_nonnegative_number(row["tflops"], "tflops")
    _optional_nonnegative_number(row["wall_time_s"], "wall_time_s")
    _json_value(row["selected_answer"], "selected_answer")
    if row["evaluation_unit"] == "chain":
        _validate_generated_identity(row)
        if row["candidate_tokens_generated"] != row["generated_tokens"]:
            raise ValueError("chain rows require candidate_tokens_generated == generated_tokens")
        if row["aux_generated_tokens"] != 0:
            raise ValueError("chain rows require aux_generated_tokens == 0")
    else:
        if row["candidate_tokens_generated"] != row["thinking_tokens"] + row["answer_tokens"]:
            raise ValueError("question rows require candidate_tokens_generated == thinking_tokens + answer_tokens")
        if row["generated_tokens"] != row["candidate_tokens_generated"] + row["aux_generated_tokens"]:
            raise ValueError(
                "question rows require generated_tokens == candidate_tokens_generated + aux_generated_tokens")
    return row


def write_tts_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for row in rows:
                validate_tts_row(row)
                try:
                    encoded = json.dumps(row, separators=(",", ":"), sort_keys=True, allow_nan=False)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("cannot serialize validated tts row") from exc
                handle.write(encoded + "\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_tts_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}") from exc
            try:
                rows.append(validate_tts_row(decoded))
            except ValueError as exc:
                raise ValueError(f"invalid tts row at line {line_number}: {exc}") from exc
    return rows


def validate_consumed_child(row: dict) -> dict:
    row = _exact_fields(row, CHILD_FIELDS, "consumed child")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    for field in ("run_id", "config_id", "qkey"):
        _nonempty_string(row[field], field)
    for field in ("sample_index", "rep", "thinking_tokens", "answer_tokens", "generated_tokens"):
        _nonnegative_int(row[field], field)
    if not _matches_hex(row["chain_sha256"], _HEX64):
        raise ValueError("chain_sha256 must be a 64-hex sha256")
    if row["correct"] is not None and not isinstance(row["correct"], bool):
        raise ValueError("child correct must be bool or null")
    _json_value(row["answer"], "answer")
    _validate_generated_identity(row)
    return row


def validate_parent_children(parent: dict, children: Iterable[dict]) -> list[dict]:
    validate_tts_row(parent)
    if parent["evaluation_unit"] != "question":
        raise ValueError("a chain parent must not have consumed children")
    checked = [validate_consumed_child(child) for child in children]
    for child in checked:
        for field in ("schema_version", "run_id", "config_id", "qkey"):
            if child[field] != parent[field]:
                raise ValueError(f"child {field} does not match parent")
    expected_indices = list(range(parent["sample_count"]))
    if [child["sample_index"] for child in checked] != expected_indices:
        raise ValueError("child sample_index values must be exactly 0..N-1 in order")
    reps = [child["rep"] for child in checked]
    if any(right <= left for left, right in zip(reps, reps[1:])):
        raise ValueError("child reps must be strictly increasing in deterministic sibling order")
    candidate_tokens = parent["candidate_tokens_generated"]
    if candidate_tokens is None:
        raise ValueError("question parent candidate_tokens_generated is required with children")
    if sum(child["thinking_tokens"] for child in checked) != parent["thinking_tokens"]:
        raise ValueError("child thinking token sum must equal parent thinking_tokens")
    if sum(child["answer_tokens"] for child in checked) != parent["answer_tokens"]:
        raise ValueError("child answer token sum must equal parent answer_tokens")
    if sum(child["generated_tokens"] for child in checked) != candidate_tokens:
        raise ValueError("child generated token sum must equal parent candidate_tokens_generated")
    if parent["generated_tokens"] != candidate_tokens + parent["aux_generated_tokens"]:
        raise ValueError("question parent generated_tokens must equal candidate_tokens_generated + aux_generated_tokens")
    return checked


def validate_comparability_row(row: dict) -> dict:
    row = _exact_fields(row, COMPARABILITY_FIELDS, "comparability row")
    for field in ("method_family", "method_name", "config_id", "model_id", "model_revision",
                  "tokenizer_revision", "thinking_mode", "prompt_contract_id", "qkey_set",
                  "grader_id", "token_definition", "scorer_aux_cost", "validation_source", "status"):
        _nonempty_string(row[field], field)
    if not isinstance(row["feasibility_tier"], str) or row["feasibility_tier"] not in {"R0", "R1", "R2"}:
        raise ValueError("feasibility_tier must be R0, R1, or R2")
    if not isinstance(row["evaluation_unit"], str) or row["evaluation_unit"] not in {"chain", "question"}:
        raise ValueError("evaluation_unit must be chain or question")
    _nonnegative_int(row["samples_per_question"], "samples_per_question")
    if row["samples_per_question"] < 1:
        raise ValueError("samples_per_question must be >= 1")
    _nonnegative_int(row["max_budget"], "max_budget")
    for field in ("generation_parameters", "frozen_config"):
        if not isinstance(row[field], dict):
            raise ValueError(f"{field} must be a JSON object")
        _json_value(row[field], field)
    if row["status"] not in {"comparable", "faceted", "incompatible"}:
        raise ValueError("status must be comparable, faceted, or incompatible")
    if row["reason"] is not None and (not isinstance(row["reason"], str) or not row["reason"]):
        raise ValueError("reason must be null or a nonempty string")
    if row["status"] in {"faceted", "incompatible"} and not isinstance(row["reason"], str):
        raise ValueError("faceted and incompatible rows require a nonempty reason")
    if row["status"] == "comparable":
        expected = _registered_expected_row(row["method_name"], row["config_id"])
        if expected is None or expected["status"] != "comparable" \
                or not _strict_json_equal(row, expected):
            raise ValueError("comparable rows must match a complete registered B0 basis")
    return row


def _registry_exact_fields(registry: Any) -> dict:
    fields = frozenset({
        "schema_version", "registry_version", "frozen_basis", "primary_math_question_set",
        "select_chain_asset", "natural_cost_asset", "selection_protocol", "comparability_rows",
        "parity_expectations", "execution_gate",
    })
    return _exact_fields(registry, fields, "B0 registry")


_NATURAL_COST_ASSET_FIELDS = frozenset({
    "status", "expected_rows", "sha256", "source_artifact", "source_revision", "reason",
    "answer_format", "force_natural_answer",
})
_ANSWER_FORMATS = ("boxed", "code")


def validate_natural_cost_asset(asset: Any, *, expected_rows: int | None = None) -> dict:
    if type(asset) is not dict:
        raise ValueError("natural-cost asset must be a JSON object")
    asset = dict(_exact_fields(asset, _NATURAL_COST_ASSET_FIELDS, "natural-cost asset"))
    _nonnegative_int(asset["expected_rows"], "natural-cost expected_rows")
    if type(asset["answer_format"]) is not str:
        raise ValueError("natural-cost answer_format must be a string")
    if asset["answer_format"] not in _ANSWER_FORMATS:
        raise ValueError(f"natural-cost answer_format must be one of {list(_ANSWER_FORMATS)}")
    if type(asset["force_natural_answer"]) is not bool:
        raise ValueError("natural-cost force_natural_answer must be a bool")
    if expected_rows is not None:
        _nonnegative_int(expected_rows, "expected natural-cost rows")
        if asset["expected_rows"] != expected_rows:
            raise ValueError("natural-cost expected_rows differs from audited chain rows")
    status = asset["status"]
    if status == "unregistered":
        if any(asset[field] is not None for field in ("sha256", "source_artifact", "source_revision")):
            raise ValueError("unregistered natural-cost asset must not contain hash or source provenance")
        _nonempty_string(asset["reason"], "unregistered natural-cost reason")
    elif status == "registered":
        if asset["expected_rows"] < 1:
            raise ValueError("registered natural-cost expected_rows must be positive")
        if not _matches_hex(asset["sha256"], _HEX64):
            raise ValueError("registered natural-cost asset sha256 must be a 64-hex digest")
        _nonempty_string(asset["source_artifact"], "registered natural-cost source_artifact")
        if not _matches_hex(asset["source_revision"], _HEX40):
            raise ValueError("registered natural-cost source_revision must be a 40-hex git oid")
        if asset["reason"] is not None:
            _nonempty_string(asset["reason"], "registered natural-cost reason")
    else:
        raise ValueError("natural-cost asset status must be registered or unregistered")
    return asset


def natural_cost_zero_answer_allowed(asset: Any) -> bool:
    asset = validate_natural_cost_asset(asset)
    return (asset["status"] == "registered"
            and asset["answer_format"] == "boxed"
            and asset["force_natural_answer"] is False)


def validate_natural_cost_execution_gate(asset: Mapping[str, Any], execution_gate: Any) -> str:
    if not isinstance(execution_gate, dict):
        raise ValueError("natural-cost execution gate must be a JSON object")
    status = asset.get("status")
    expected = "registered" if status == "registered" else "blocked_unregistered"
    observed = execution_gate.get("natural_cost_sidecar_registration")
    if observed != expected:
        raise ValueError("natural-cost asset/execution gate contradiction")
    return expected


def validate_b0_registry(registry: dict) -> dict:
    registry = _registry_exact_fields(registry)
    if registry["schema_version"] != SCHEMA_VERSION or registry["registry_version"] != "b0/v1":
        raise ValueError("unexpected B0 registry version")
    basis = registry["frozen_basis"]
    expected_basis = {
        "model_id": _FROZEN_MODEL,
        "model_revision": _FROZEN_REVISION,
        "tokenizer_revision": _FROZEN_REVISION,
        "thinking_mode": "native_on",
        "block_separator": "\\n[ \\t]*\\n",
        "think_close_tag": "</think>",
        "max_budget": 12000,
        "block_cap": 128,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
        "prompt_contract_id": _FROZEN_PROMPT,
        "grader_ids": ["archived_math_equal", "archived_code_sandbox"],
    }
    if not _strict_json_equal(basis, expected_basis):
        raise ValueError("B0 frozen_basis differs from the registered basis")
    question_set = registry["primary_math_question_set"]
    expected_question_set = {
        "manifest_path": "docs/superpowers/contracts/2026-08-27-iclr-p1-math-manifest-union12.jsonl",
        "file_sha256": "86b90b574944d135fb93027233ec8b797381be68f89042a22c299b8f5a25cf80",
        "qkey_sha256_convention": "sha256(sorted qkeys joined by newline with final newline)",
        "splits": {
            "fit": {"qkey_count": 4241, "qkey_sha256": "210cc15462082464d3d17555c7f7b475e515a13f1a88cbfdb0d220b88824699e"},
            "select": {"qkey_count": 1065, "qkey_sha256": "c1cdb013c79421a199a785674471e2ed73df6cd2faaf2b0018bfda52f237f3cb"},
            "report": {"qkey_count": 1694, "qkey_sha256": "bd438c13da7cb5b4e98dbafb94012878376e5f6880724e7b2d73e4c54200a26c"},
        },
    }
    if not _strict_json_equal(question_set, expected_question_set):
        raise ValueError("B0 primary_math_question_set differs from the frozen manifest")
    if not _strict_json_equal(registry["select_chain_asset"], {
        "expected_rows": 10648,
        "generation_seed": 0,
        "sha256": "f4441bccee55615d0b45b55ba48f3a5e33c4ac17fccb15a203bb416c066ebcc8",
    }):
        raise ValueError("B0 select chain asset differs from the frozen asset")
    natural_cost_asset = validate_natural_cost_asset(registry["natural_cost_asset"], expected_rows=10648)
    if natural_cost_asset["status"] == "unregistered" and not _strict_json_equal(natural_cost_asset, {
            "status": "unregistered", "expected_rows": 10648, "sha256": None,
            "source_artifact": None, "source_revision": None,
            "answer_format": "boxed", "force_natural_answer": False,
            "reason": "Private natural-cost token splits are absent locally and are not registered.",
    }):
        raise ValueError("unregistered B0 natural-cost asset differs from the frozen block")
    if (natural_cost_asset["answer_format"] != "boxed"
            or natural_cost_asset["force_natural_answer"] is not False):
        raise ValueError("B0 natural-cost asset must declare the frozen boxed / no-force pool")
    if not _strict_json_equal(registry["selection_protocol"], {
        "head_seeds": [0, 1], "sibling_order": ["qkey", "rep"], "n_grid": [1, 2, 4, 8],
        "roles": {"head_validation": "archived_only_for_tuning", "select": "claim_bearing_replay", "report": "sealed"},
    }):
        raise ValueError("B0 selection protocol differs from the frozen protocol")
    natural_cost_gate = validate_natural_cost_execution_gate(natural_cost_asset, registry["execution_gate"])
    if not _strict_json_equal(registry["execution_gate"], {
        "full_archive_parity": "blocked_locally_private_chain_and_scored_dumps_absent",
        "fixture_and_code_work_allowed": True,
        "b1_job_launch": "prohibited",
        "natural_cost_sidecar_registration": natural_cost_gate,
        "shared_fixture_byte_parity": "blocked_pending_frontier_import",
    }):
        raise ValueError("B0 execution gate must keep local parity blocked and B1 prohibited")
    rows = registry["comparability_rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("comparability_rows must be a nonempty list")
    for row in rows:
        validate_comparability_row(row)
    identities = [(row["method_name"], row["config_id"]) for row in rows]
    if set(identities) != set(_REGISTERED_METHOD_SPECS) or len(identities) != len(set(identities)):
        raise ValueError("comparability_rows must cover each registered B0 method/config exactly once")
    for row in rows:
        expected = _registered_expected_row(row["method_name"], row["config_id"])
        if expected is None or not _strict_json_equal(row, expected):
            raise ValueError("comparability rows differ from the complete frozen B0 registry")
    parity = registry["parity_expectations"]
    if not isinstance(parity, dict) or not _strict_json_equal(
            parity.get("source"), "results/iclr/ladders/p2_union/row3_full11.md"):
        raise ValueError("parity_expectations must cite the committed full11 ladder")
    if not _strict_json_equal(parity.get("expected_chains"), 10648) or not isinstance(parity.get("rows"), dict):
        raise ValueError("parity_expectations must retain the 10648-chain fixture")
    expected_parity_rows = {
        "natural": {"accuracy": 0.5092975206611571, "accuracy_tolerance": 1e-12,
                    "mean_thinking_tokens": 6692.4346356123215, "mean_thinking_tokens_tolerance": 1e-9},
        "always_fire": {"accuracy": 0.099, "accuracy_tolerance": 0.0005,
                        "mean_thinking_tokens": 331, "mean_thinking_tokens_tolerance": 0.5},
        "cap16": {"accuracy": 0.202, "accuracy_tolerance": 0.0005,
                  "mean_thinking_tokens": 1090, "mean_thinking_tokens_tolerance": 0.5},
        "cap32": {"accuracy": 0.287, "accuracy_tolerance": 0.0005,
                  "mean_thinking_tokens": 1883, "mean_thinking_tokens_tolerance": 0.5},
        "cap64": {"accuracy": 0.406, "accuracy_tolerance": 0.0005,
                  "mean_thinking_tokens": 3152, "mean_thinking_tokens_tolerance": 0.5},
        "cap96": {"accuracy": 0.460, "accuracy_tolerance": 0.0005,
                  "mean_thinking_tokens": 4104, "mean_thinking_tokens_tolerance": 0.5},
        "cap128": {"accuracy": 0.489, "accuracy_tolerance": 0.0005,
                   "mean_thinking_tokens": 4818, "mean_thinking_tokens_tolerance": 0.5},
        "consecutive_agreement": {"accuracy": 0.241, "accuracy_tolerance": 0.0005,
                                  "mean_thinking_tokens": 1494, "mean_thinking_tokens_tolerance": 0.5},
        "earliest_agreement": {"accuracy": 0.509, "accuracy_tolerance": 0.0005,
                                "mean_thinking_tokens": 4951, "mean_thinking_tokens_tolerance": 0.5},
    }
    if not _strict_json_equal(parity["rows"], expected_parity_rows):
        raise ValueError("parity expectations differ from the frozen full11 fixture")
    return registry
