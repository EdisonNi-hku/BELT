from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from rl_controller.iclr.chains import is_correct, load_chains, validate_chain
from rl_controller.iclr.ladder import always_fire, natural
from rl_controller.iclr.ladder import free_family
from rl_controller.iclr.pi0_bias import consecutive_agreement
from rl_controller.iclr.tts_contract import (
    REPLAY_SEED,
    SCHEMA_VERSION,
    natural_cost_zero_answer_allowed,
    validate_b0_registry,
    validate_consumed_child,
    validate_parent_children,
    validate_tts_row,
    write_tts_jsonl,
)
from rl_controller.single_cot import stop_at_cap, stop_earliest_agreement


Plan = Callable[[dict], dict | None]
_COST_FIELDS = frozenset({"qkey", "rep", "thinking_tokens", "answer_tokens"})
_N_GRID = frozenset({1, 2, 4, 8})
_MODEL_ID = "Qwen/Qwen3-1.7B"
_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
_PROMPT_CONTRACT = "qwen3-1.7b-native-thinking-v1"
_DATASET = "iclr-p1-math-union12"

DEFAULT_SPLIT = "select"

CANONICAL_MEMBER_ID = {
    "natural-v1": "natural",
    "always-fire-v1": "always_fire",
    "consecutive-agreement-v1": "consecutive_agreement",
    "earliest-agreement-v1": "earliest_agreement",
    **{f"cap{c}-v1": f"cap{c}" for c in (16, 32, 64, 96, 128)},
}


def canonical_member_id(config_id: str) -> str:
    return CANONICAL_MEMBER_ID.get(config_id, config_id)
_GRADER = "archived_math_equal"


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class _HashingJsonlLines:

    def __init__(self, handle) -> None:
        self._handle = handle
        self._digest = hashlib.sha256()

    def __iter__(self):
        return self

    def __next__(self) -> str:
        line = self._handle.readline()
        if not line:
            raise StopIteration
        self._digest.update(line)
        return line.decode("utf-8")

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _parse_hashed_jsonl(path: str | Path, parser: Callable[[Iterable[str]], Any]) -> tuple[Any, str]:
    with Path(path).open("rb") as handle:
        lines = _HashingJsonlLines(handle)
        parsed = parser(lines)
        return parsed, lines.hexdigest()


def canonical_chain_sha256(chain: dict) -> str:
    encoded = json.dumps(chain, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def qkey_sha256(qkeys: Iterable[str]) -> str:
    ordered = sorted(qkeys)
    if len(ordered) != len(set(ordered)):
        raise ValueError("qkeys must be unique before computing qkey_sha256")
    if any(not isinstance(qkey, str) or not qkey for qkey in ordered):
        raise ValueError("qkeys must be nonempty strings")
    return hashlib.sha256(("\n".join(ordered) + "\n").encode()).hexdigest()


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _strict_equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _primary_math_answer(value: Any, label: str, *, allow_none: bool = True) -> None:
    if value is None and allow_none:
        return
    if type(value) is not int:
        suffix = "integer or None" if allow_none else "integer"
        raise ValueError(f"{label} must be an archived math {suffix}")


def validate_primary_math_chain(chain: dict) -> dict:
    validate_chain(chain)
    if not isinstance(chain["qkey"], str) or not chain["qkey"]:
        raise ValueError("primary math chain qkey must be a nonempty string")
    _nonnegative_int(chain["rep"], "primary math chain rep")
    _primary_math_answer(chain["gold"], "primary math chain gold", allow_none=False)
    _primary_math_answer(chain["answer"], "primary math chain answer")
    if type(chain["correct"]) is not bool:
        raise ValueError("primary math chain correct must be bool")
    _nonnegative_int(chain["blocks"], "primary math chain blocks")
    _nonnegative_int(chain["thinking_tokens"], "primary math chain thinking_tokens")
    if not isinstance(chain["block_ends"], list) or not isinstance(chain["query_points"], list):
        raise ValueError("primary math block_ends and query_points must be lists")
    for index, end in enumerate(chain["block_ends"]):
        _nonnegative_int(end, f"primary math block_ends[{index}]")
    for index, point in enumerate(chain["query_points"]):
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"primary math query_points[{index}] must be [block, prefix_len]")
        _nonnegative_int(point[0], f"primary math query_points[{index}][0]")
        _nonnegative_int(point[1], f"primary math query_points[{index}][1]")
    for probe_index, probe in enumerate(chain["probes"]):
        _nonnegative_int(probe["block"], f"primary math probe {probe_index} block")
        if probe["block"] < 1:
            raise ValueError(f"primary math probe {probe_index} block must be >= 1")
        _nonnegative_int(probe["prefix_len"], f"primary math probe {probe_index} prefix_len")
        _nonnegative_int(probe["forced_tokens"], f"primary math probe {probe_index} forced_tokens")
        _primary_math_answer(probe["forced"], f"primary math probe {probe_index} forced")
        for draw_index, draw in enumerate(probe.get("draws", ())):
            _nonnegative_int(draw["draw"], f"primary math probe {probe_index} draw {draw_index} index")
            _primary_math_answer(draw["forced"], f"primary math probe {probe_index} draw {draw_index} forced")
            _nonnegative_int(draw["forced_tokens"],
                             f"primary math probe {probe_index} draw {draw_index} forced_tokens")
    return chain


def _load_natural_cost_lines(lines: Iterable[str]) -> dict[tuple[str, int], dict[str, int]]:
    loaded: dict[tuple[str, int], dict[str, int]] = {}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"natural costs line {line_number}: blank lines are not allowed")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"natural costs line {line_number}: invalid JSON") from exc
        if not isinstance(row, dict) or set(row) != _COST_FIELDS:
            actual = set(row) if isinstance(row, dict) else set()
            raise ValueError(f"natural costs line {line_number}: fields must be exactly "
                             f"{sorted(_COST_FIELDS)}, got {sorted(actual)}")
        if not isinstance(row["qkey"], str) or not row["qkey"]:
            raise ValueError(f"natural costs line {line_number}: qkey must be a nonempty string")
        rep = _nonnegative_int(row["rep"], f"natural costs line {line_number}: rep")
        thinking = _nonnegative_int(row["thinking_tokens"],
                                    f"natural costs line {line_number}: thinking_tokens")
        answer = _nonnegative_int(row["answer_tokens"],
                                  f"natural costs line {line_number}: answer_tokens")
        identity = (row["qkey"], rep)
        if identity in loaded:
            raise ValueError(f"natural costs line {line_number}: duplicate identity {identity}")
        loaded[identity] = {"thinking_tokens": thinking, "answer_tokens": answer}
    return loaded


def load_natural_costs(path: str | Path) -> dict[tuple[str, int], dict[str, int]]:
    with Path(path).open(encoding="utf-8") as handle:
        return _load_natural_cost_lines(handle)


def load_natural_costs_snapshot(path: str | Path) -> tuple[dict[tuple[str, int], dict[str, int]], str]:
    return _parse_hashed_jsonl(path, _load_natural_cost_lines)


def validate_natural_cost_identities(chains: Iterable[dict],
                                     natural_costs: Mapping[tuple[str, int], Mapping[str, int]]) -> None:
    chain_ids: set[tuple[str, int]] = set()
    for chain in chains:
        validate_primary_math_chain(chain)
        chain_ids.add((chain["qkey"], chain["rep"]))
    cost_ids = set(natural_costs)
    missing, extra = sorted(chain_ids - cost_ids), sorted(cost_ids - chain_ids)
    if missing or extra:
        pieces = []
        if missing:
            pieces.append(f"missing natural cost identities {missing}")
        if extra:
            pieces.append(f"extra natural cost identities {extra}")
        raise ValueError("; ".join(pieces))


def validate_registered_natural_costs(chains: Iterable[dict],
                                      natural_costs: Mapping[tuple[str, int], Mapping[str, int]],
                                      *, asset: Mapping[str, Any]) -> None:
    chain_rows = list(chains)
    validate_natural_cost_identities(chain_rows, natural_costs)
    zero_allowed = natural_cost_zero_answer_allowed(asset)
    for chain in chain_rows:
        identity = (chain["qkey"], chain["rep"])
        answer = _nonnegative_int(natural_costs[identity]["answer_tokens"],
                                  f"natural token split {identity} answer_tokens")
        if answer <= 0 and not zero_allowed:
            raise ValueError(f"natural token split {identity} answer_tokens must be positive for registered export")


def _load_select_manifest_lines(lines: Iterable[str]) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"manifest line {line_number}: blank lines are not allowed")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest line {line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"manifest line {line_number}: row must be an object")
        required = {"qkey", "gold", "split", "target_chains"}
        if not required <= set(row):
            raise ValueError(f"manifest line {line_number}: missing {sorted(required - set(row))}")
        if row["split"] != "select":
            continue
        qkey = row["qkey"]
        if not isinstance(qkey, str) or not qkey:
            raise ValueError(f"manifest line {line_number}: qkey must be a nonempty string")
        _nonnegative_int(row["target_chains"], f"manifest line {line_number}: target_chains")
        if row["target_chains"] < 1:
            raise ValueError(f"manifest line {line_number}: target_chains must be >= 1")
        if row["gold"] is None or isinstance(row["gold"], bool) or not isinstance(row["gold"], int):
            raise ValueError(f"manifest line {line_number}: archived math gold must be an integer")
        if qkey in selected:
            raise ValueError(f"manifest line {line_number}: duplicate select qkey {qkey!r}")
        selected[qkey] = {"gold": row["gold"], "target_chains": row["target_chains"]}
    if not selected:
        raise ValueError("manifest has no select rows")
    return selected


def _load_select_manifest_snapshot(path: str | Path) -> tuple[dict[str, dict], str]:
    return _parse_hashed_jsonl(path, _load_select_manifest_lines)


def load_chains_snapshot(path: str | Path) -> tuple[list[dict], str]:
    return _parse_hashed_jsonl(path, load_chains)


def audit_chain_bank(chains_path: str | Path, manifest_path: str | Path, expected: Mapping[str, Any],
                     *, return_chains: bool = False,
                     manifest_snapshot: tuple[dict[str, dict], str] | None = None) -> dict | tuple[dict, list[dict]]:
    required = {"manifest_sha256", "chain_sha256", "qkeys", "qkey_sha256", "rows"}
    if set(expected) != required:
        raise ValueError(f"audit expectations must be exactly {sorted(required)}")
    chains_path, manifest_path = Path(chains_path), Path(manifest_path)
    if manifest_snapshot is None:
        manifest, manifest_sha = _load_select_manifest_snapshot(manifest_path)
    else:
        manifest, manifest_sha = manifest_snapshot
    chains, chain_sha = load_chains_snapshot(chains_path)
    if manifest_sha != expected["manifest_sha256"]:
        raise ValueError(f"manifest sha256 mismatch: expected {expected['manifest_sha256']}, got {manifest_sha}")
    if chain_sha != expected["chain_sha256"]:
        raise ValueError(f"chain sha256 mismatch: expected {expected['chain_sha256']}, got {chain_sha}")
    expected_qkeys = set(expected["qkeys"])
    if set(manifest) != expected_qkeys:
        raise ValueError("manifest select qkey set differs from the frozen expectation")
    observed_qhash = qkey_sha256(manifest)
    if observed_qhash != expected["qkey_sha256"]:
        raise ValueError(f"select qkey sha256 mismatch: expected {expected['qkey_sha256']}, got {observed_qhash}")
    chains = [validate_primary_math_chain(chain) for chain in chains]
    if len(chains) != expected["rows"]:
        raise ValueError(f"chain row count mismatch: expected {expected['rows']}, got {len(chains)}")
    groups: dict[str, list[dict]] = defaultdict(list)
    for chain in chains:
        if chain["qkey"] not in manifest:
            raise ValueError(f"chain identity {(chain['qkey'], chain['rep'])} is outside manifest select qkeys")
        expected_gold = manifest[chain["qkey"]]["gold"]
        if not _strict_equal(chain["gold"], expected_gold):
            raise ValueError(f"chain {(chain['qkey'], chain['rep'])} gold differs from manifest")
        groups[chain["qkey"]].append(chain)
    if set(groups) != set(manifest):
        raise ValueError("chain qkey set differs from manifest select qkeys")
    for qkey, metadata in manifest.items():
        siblings = sorted(groups[qkey], key=lambda chain: chain["rep"])
        reps = [chain["rep"] for chain in siblings]
        target = metadata["target_chains"]
        if reps != list(range(target)):
            raise ValueError(f"qkey {qkey!r} reps must be exactly 0..{target - 1}, got {reps}")
        if any(not _strict_equal(chain["gold"], siblings[0]["gold"]) for chain in siblings):
            raise ValueError(f"qkey {qkey!r} chains disagree on gold")
    audit = {"manifest_sha256": manifest_sha, "chain_sha256": chain_sha, "rows": len(chains),
             "qkey_count": len(manifest), "qkey_sha256": observed_qhash}
    return (audit, chains) if return_chains else audit


def load_b0_registry(path: str | Path) -> dict:
    try:
        registry = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid B0 registry JSON: {path}") from exc
    return validate_b0_registry(registry)


def audit_primary_inputs(chains_path: str | Path, manifest_path: str | Path,
                         registry_path: str | Path, *, return_chains: bool = False) -> dict | tuple[dict, list[dict]]:
    registry = load_b0_registry(registry_path)
    question_set = registry["primary_math_question_set"]
    select = question_set["splits"]["select"]
    manifest, manifest_sha = _load_select_manifest_snapshot(manifest_path)
    if manifest_sha != question_set["file_sha256"]:
        raise ValueError("manifest sha256 differs from the frozen registry")
    expected = {
        "manifest_sha256": manifest_sha,
        "chain_sha256": registry["select_chain_asset"]["sha256"],
        "qkeys": manifest.keys(),
        "qkey_sha256": select["qkey_sha256"],
        "rows": registry["select_chain_asset"]["expected_rows"],
    }
    if len(expected["qkeys"]) != select["qkey_count"]:
        raise ValueError("manifest select qkey count differs from the frozen registry")
    return audit_chain_bank(chains_path, manifest_path, expected, return_chains=return_chains,
                            manifest_snapshot=(manifest, manifest_sha))


def outcome_for_plan(chain: dict, plan: Plan, natural_costs: Mapping[tuple[str, int], Mapping[str, int]]) -> dict:
    validate_primary_math_chain(chain)
    probe = plan(chain)
    if probe is None:
        identity = (chain["qkey"], chain["rep"])
        if identity not in natural_costs:
            raise ValueError(f"missing natural token split for chain {identity}")
        split = natural_costs[identity]
        if not isinstance(split, Mapping) or set(split) != {"thinking_tokens", "answer_tokens"}:
            raise ValueError(f"natural token split for {identity} must contain exactly thinking_tokens and answer_tokens")
        thinking = _nonnegative_int(split["thinking_tokens"], f"natural token split {identity} thinking_tokens")
        answer = _nonnegative_int(split["answer_tokens"], f"natural token split {identity} answer_tokens")
        if thinking + answer != chain["thinking_tokens"]:
            raise ValueError(f"natural token split for {identity} sums to {thinking + answer}, "
                             f"not legacy completion total {chain['thinking_tokens']}")
        selected = chain["answer"]
    else:
        if not isinstance(probe, dict) or probe not in chain["probes"]:
            raise ValueError("plan returned a probe not present in the validated chain")
        thinking = _nonnegative_int(probe["prefix_len"], "probe prefix_len")
        answer = _nonnegative_int(probe["forced_tokens"], "probe forced_tokens")
        selected = probe["forced"]
    return {"selected_answer": selected, "correct": bool(is_correct(selected, chain["gold"])),
            "thinking_tokens": thinking, "answer_tokens": answer,
            "generated_tokens": thinking + answer}


def _legacy_metric(chains: list[dict], plan: Plan, *, natural_plan: bool = False) -> dict:
    n = len(chains)
    if not n:
        raise ValueError("legacy anchors require at least one chain")
    correct = thinking = 0
    fired = 0
    total_tokens = 0
    for chain in chains:
        probe = plan(chain)
        if probe is None:
            correct += int(bool(chain["correct"]))
            thinking += chain["thinking_tokens"]
            total_tokens += chain["thinking_tokens"]
        else:
            fired += 1
            correct += is_correct(probe["forced"], chain["gold"])
            thinking += probe["prefix_len"]
            total_tokens += probe["prefix_len"] + probe["forced_tokens"]
    return {"n": n, "accuracy": correct / n, "mean_thinking_tokens": thinking / n,
            "mean_total_tokens": total_tokens / n,
            "fired_fraction": None if natural_plan else fired / n}


def legacy_anchor_metrics(chains: Iterable[dict]) -> dict[str, dict]:
    checked = [validate_primary_math_chain(chain) for chain in chains]
    plans: dict[str, Plan] = dict(free_family())
    plans["consecutive_agreement"] = consecutive_agreement
    plans["earliest_agreement"] = stop_earliest_agreement
    return {name: _legacy_metric(checked, plan, natural_plan=(name == "natural"))
            for name, plan in plans.items()}


def check_parity(metrics: Mapping[str, Mapping[str, Any]], expectations: Mapping[str, Mapping[str, Any]],
                 *, expected_chains: int) -> None:
    _nonnegative_int(expected_chains, "expected_chains")
    if expected_chains < 1:
        raise ValueError("expected_chains must be >= 1")
    if not isinstance(metrics, Mapping) or not isinstance(expectations, Mapping):
        raise ValueError("parity metrics and expectations must be mappings")
    def finite(value: Any, label: str, *, nonnegative: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{label} must be a finite number")
        if nonnegative and value < 0:
            raise ValueError(f"{label} must be a finite nonnegative number")
        return float(value)
    mismatches: list[str] = []
    observed_ns = set()
    for name, row in metrics.items():
        if not isinstance(row, Mapping):
            raise ValueError(f"{name} parity metric must be a mapping")
        observed_ns.add(_nonnegative_int(row.get("n"), f"{name} n"))
        finite(row.get("accuracy"), f"{name} accuracy")
        finite(row.get("mean_thinking_tokens"), f"{name} mean_thinking_tokens", nonnegative=True)
    if observed_ns != {expected_chains}:
        mismatches.append(f"expected_chains expected {expected_chains}, got {sorted(observed_ns)!r}")
    for name, expected in expectations.items():
        if name not in metrics:
            mismatches.append(f"{name} missing metric row")
            continue
        row = metrics[name]
        for field, tolerance_field in (("accuracy", "accuracy_tolerance"),
                                       ("mean_thinking_tokens", "mean_thinking_tokens_tolerance")):
            actual = finite(row.get(field), f"{name} {field}")
            target = finite(expected.get(field), f"{name} expected {field}")
            tolerance = finite(expected.get(tolerance_field), f"{name} {field} tolerance", nonnegative=True)
            if abs(actual - target) > tolerance:
                mismatches.append(f"{name} {field} expected {target!r} +/- {tolerance!r}, got {actual!r}")
    if mismatches:
        raise ValueError("legacy B0 parity mismatch: " + "; ".join(mismatches))


def group_siblings(chains: Iterable[dict], n: int) -> dict[str, list[dict]]:
    _nonnegative_int(n, "N")
    if n not in _N_GRID:
        raise ValueError(f"N must be one of {sorted(_N_GRID)}, got {n}")
    groups: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for chain in chains:
        validate_primary_math_chain(chain)
        identity = (chain["qkey"], chain["rep"])
        if identity in seen:
            raise ValueError(f"duplicate rep identity {identity}")
        seen.add(identity)
        groups[chain["qkey"]].append(chain)
    if not groups:
        raise ValueError("question export requires at least one chain")
    selected: dict[str, list[dict]] = {}
    for qkey, siblings in sorted(groups.items()):
        siblings.sort(key=lambda chain: chain["rep"])
        reps = [chain["rep"] for chain in siblings]
        if len(siblings) < n:
            raise ValueError(f"qkey {qkey!r} has insufficient siblings for N={n}")
        if len(reps) != len(set(reps)):
            raise ValueError(f"qkey {qkey!r} has duplicate reps")
        if any(not _strict_equal(sibling["gold"], siblings[0]["gold"]) for sibling in siblings):
            raise ValueError(f"qkey {qkey!r} siblings disagree on gold")
        selected[qkey] = siblings[:n]
    return selected


def deterministic_plurality(answers: Iterable[Any]) -> Any:
    clusters: list[dict[str, Any]] = []
    for answer in answers:
        if answer is None:
            continue
        if type(answer) is not int:
            raise ValueError("archived math plurality answers must be integer or None")
        for cluster in clusters:
            if _strict_equal(answer, cluster["answer"]):
                cluster["count"] += 1
                break
        else:
            clusters.append({"answer": answer, "count": 1})
    if not clusters:
        return None
    return max(clusters, key=lambda cluster: cluster["count"])["answer"]


def _base_parent(*, run_id: str, method_family: str, method_name: str, config_id: str,
                 source_artifact: str, source_sha256: str, source_branch: str, source_commit: str,
                 qkey: str, rep: int | None, evaluation_unit: str, sample_count: int,
                 selection_rule: str, frozen_config: dict, generation_seed: int,
                 split: str = DEFAULT_SPLIT) -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "method_family": method_family,
        "method_name": method_name, "config_id": config_id, "source_branch": source_branch,
        "source_commit": source_commit, "source_artifact": source_artifact,
        "source_sha256": source_sha256, "model_id": _MODEL_ID,
        "model_revision": _MODEL_REVISION, "tokenizer_revision": _MODEL_REVISION,
        "dataset": _DATASET, "split": split, "qkey": qkey, "rep": rep, "seed": REPLAY_SEED,
        "evaluation_unit": evaluation_unit, "sample_count": sample_count,
        "prompt_contract_id": _PROMPT_CONTRACT, "grader_id": _GRADER,
        "selection_split": "archived_head_validation", "selection_rule": selection_rule,
        "frozen_config": frozen_config, "correct": None, "selected_answer": None,
        "thinking_tokens": None, "answer_tokens": None, "generated_tokens": None,
        "prompt_tokens": None, "aux_generated_tokens": None, "candidate_tokens_generated": None,
        "model_forward_passes": None, "scorer_forward_passes": None, "tflops": None,
        "wall_time_s": None, "missing_reason": None,
    }


def _bound_frozen_config(frozen_config: dict, natural_costs_sha256: str) -> dict:
    if not isinstance(frozen_config, dict):
        raise ValueError("frozen_config must be a JSON object")
    if (not isinstance(natural_costs_sha256, str) or len(natural_costs_sha256) != 64
            or any(char not in "0123456789abcdef" for char in natural_costs_sha256)):
        raise ValueError("natural_costs_sha256 must be a sha256")
    if "natural_costs_sha256" in frozen_config and frozen_config["natural_costs_sha256"] != natural_costs_sha256:
        raise ValueError("frozen_config natural_costs_sha256 contradicts sidecar provenance")
    return {**frozen_config, "natural_costs_sha256": natural_costs_sha256}


def iter_chain_rows(chains: Iterable[dict], plan: Plan, *, method_name: str,
                    source_artifact: str, source_sha256: str, source_branch: str,
                    source_commit: str, natural_costs: Mapping[tuple[str, int], Mapping[str, int]],
                    natural_costs_sha256: str, generation_seed: int,
                    method_family: str | None = None, config_id: str | None = None,
                    frozen_config: dict | None = None, split: str = DEFAULT_SPLIT
                    ) -> Iterable[dict]:
    _nonnegative_int(generation_seed, "generation_seed")
    method_family = method_family or method_name
    config_id = canonical_member_id(config_id or f"{method_name}-v1")
    frozen_config = _bound_frozen_config(
        frozen_config or {"strategy": method_name, "replay": "archived_chain_bank"}, natural_costs_sha256)
    run_id = f"b0-chain-{method_name}-{config_id}"
    for chain in sorted(chains, key=lambda item: (item["qkey"], item["rep"])):
        outcome = outcome_for_plan(chain, plan, natural_costs)
        row = _base_parent(run_id=run_id, method_family=method_family, method_name=method_name,
                           config_id=config_id, source_artifact=source_artifact,
                           source_sha256=source_sha256, source_branch=source_branch,
                           source_commit=source_commit, qkey=chain["qkey"], rep=chain["rep"],
                           evaluation_unit="chain", sample_count=1, selection_rule=method_name,
                           frozen_config=frozen_config, generation_seed=generation_seed,
                           split=split)
        row.update(outcome, prompt_tokens=None, aux_generated_tokens=0,
                   candidate_tokens_generated=outcome["generated_tokens"])
        yield validate_tts_row(row)


def build_chain_rows(chains: Iterable[dict], plan: Plan, **kwargs: Any) -> list[dict]:
    return list(iter_chain_rows(chains, plan, **kwargs))


def build_question_export(chains: Iterable[dict], plan: Plan, n: int, *, method_family: str,
                          method_name: str, config_id: str, source_artifact: str,
                          source_sha256: str, source_branch: str, source_commit: str,
                          natural_costs: Mapping[tuple[str, int], Mapping[str, int]],
                          natural_costs_sha256: str, generation_seed: int,
                          frozen_config: dict | None = None, split: str = DEFAULT_SPLIT
                          ) -> tuple[list[dict], list[dict]]:
    groups = group_siblings(chains, n)
    _nonnegative_int(generation_seed, "generation_seed")
    frozen_config = _bound_frozen_config(
        frozen_config or {"strategy": method_name, "samples_per_question": n,
                          "vote_rule": "type_strict_deterministic_plurality"}, natural_costs_sha256)
    config_id = canonical_member_id(config_id)
    run_id = f"b0-question-{method_name}-{config_id}-n{n}"
    parents: list[dict] = []
    children: list[dict] = []
    for qkey, siblings in groups.items():
        outcomes = [outcome_for_plan(chain, plan, natural_costs) for chain in siblings]
        voted = deterministic_plurality(outcome["selected_answer"] for outcome in outcomes)
        thinking = sum(outcome["thinking_tokens"] for outcome in outcomes)
        answer = sum(outcome["answer_tokens"] for outcome in outcomes)
        generated = sum(outcome["generated_tokens"] for outcome in outcomes)
        parent = _base_parent(run_id=run_id, method_family=method_family, method_name=method_name,
                              config_id=config_id, source_artifact=source_artifact,
                              source_sha256=source_sha256, source_branch=source_branch,
                              source_commit=source_commit, qkey=qkey, rep=None, evaluation_unit="question",
                              sample_count=n, selection_rule="type_strict_deterministic_plurality",
                              frozen_config=frozen_config, generation_seed=generation_seed,
                              split=split)
        parent.update(correct=bool(is_correct(voted, siblings[0]["gold"])), selected_answer=voted,
                      thinking_tokens=thinking, answer_tokens=answer, generated_tokens=generated,
                      prompt_tokens=None, aux_generated_tokens=0, candidate_tokens_generated=generated)
        parent = validate_tts_row(parent)
        local_children = []
        for index, (chain, outcome) in enumerate(zip(siblings, outcomes)):
            child = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "config_id": config_id,
                     "qkey": qkey, "sample_index": index, "rep": chain["rep"],
                     "chain_sha256": canonical_chain_sha256(chain), "answer": outcome["selected_answer"],
                     "correct": outcome["correct"], "thinking_tokens": outcome["thinking_tokens"],
                     "answer_tokens": outcome["answer_tokens"], "generated_tokens": outcome["generated_tokens"]}
            local_children.append(validate_consumed_child(child))
        validate_parent_children(parent, local_children)
        parents.append(parent)
        children.extend(local_children)
    return parents, children


def _n1_equals_chain_plan_first_sibling(parents: Iterable[dict], children: Iterable[dict], chains: Iterable[dict],
                                        plan: Plan,
                                        natural_costs: Mapping[tuple[str, int], Mapping[str, int]]) -> bool:
    first = group_siblings(chains, 1)
    parent_rows, child_rows = list(parents), list(children)
    parent_ids: set[tuple[Any, ...]] = set()
    parents_by_qkey: dict[str, list[dict]] = defaultdict(list)
    for parent in parent_rows:
        validate_tts_row(parent)
        identity = (parent["run_id"], parent["config_id"], parent["qkey"],
                    parent["evaluation_unit"], parent["sample_count"])
        if identity in parent_ids:
            raise ValueError(f"duplicate parent identity {identity}")
        parent_ids.add(identity)
        parents_by_qkey[parent["qkey"]].append(parent)
    child_ids: set[tuple[Any, ...]] = set()
    children_by_qkey: dict[str, list[dict]] = defaultdict(list)
    for child in child_rows:
        validate_consumed_child(child)
        identity = (child["run_id"], child["config_id"], child["qkey"], child["sample_index"])
        if identity in child_ids:
            raise ValueError(f"duplicate child identity {identity}")
        child_ids.add(identity)
        children_by_qkey[child["qkey"]].append(child)
    if set(first) != set(children_by_qkey) or set(first) != set(parents_by_qkey):
        return False
    for qkey, siblings in first.items():
        if len(parents_by_qkey[qkey]) != 1:
            raise ValueError(f"qkey {qkey!r} must have exactly one reconciliation parent")
        parent = parents_by_qkey[qkey][0]
        local_children = children_by_qkey[qkey]
        validate_parent_children(parent, local_children)
        outcome = outcome_for_plan(siblings[0], plan, natural_costs)
        if len(local_children) != 1:
            raise ValueError(f"qkey {qkey!r} must have exactly one reconciliation child")
        child = local_children[0]
        child_outcome = {"selected_answer": child["answer"], "correct": child["correct"],
                         "thinking_tokens": child["thinking_tokens"], "answer_tokens": child["answer_tokens"],
                         "generated_tokens": child["generated_tokens"]}
        if (not _strict_equal(child["rep"], siblings[0]["rep"])
                or child["chain_sha256"] != canonical_chain_sha256(siblings[0])
                or child_outcome != outcome or any(parent[key] != outcome[key] for key in outcome)):
            return False
    return True


def sc1_equals_natural_first_sibling(parents: Iterable[dict], children: Iterable[dict], chains: Iterable[dict],
                                     natural_costs: Mapping[tuple[str, int], Mapping[str, int]]) -> bool:
    return _n1_equals_chain_plan_first_sibling(parents, children, chains, natural, natural_costs)


def controller1_equals_chain_first_sibling(parents: Iterable[dict], children: Iterable[dict], chains: Iterable[dict],
                                            plan: Plan,
                                            natural_costs: Mapping[tuple[str, int], Mapping[str, int]]) -> bool:
    return _n1_equals_chain_plan_first_sibling(parents, children, chains, plan, natural_costs)


def write_children_jsonl(path: str | Path, children: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for child in children:
                validate_consumed_child(child)
                handle.write(json.dumps(child, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_children_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"children JSONL line {line_number}: blank lines are not allowed")
            try:
                rows.append(validate_consumed_child(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"children JSONL line {line_number}: invalid child") from exc
    return rows


def write_json_atomic(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


_FIXTURE_ARTIFACT = "tests/fixtures/iclr_tts_b0/chains.jsonl"
_FIXTURE_SOURCE_COMMIT = "ae354a0c746c5cba7bc3956aff3f91dcf80f88b2"
_FIXTURE_GENERATOR_PATH = "rl_controller/iclr/banked_baselines.py"
_FIXTURE_GENERATOR_REVISION = "5c6e3641082142997b9c92f4be1c3735e90e3433"


def _revision(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{label} must be a full lowercase git revision")
    return value


def _git_blob(repository_root: Path, revision: str, path: str, label: str) -> bytes:
    if not isinstance(path, str) or not path or path.startswith("/") or ".." in Path(path).parts:
        raise ValueError(f"{label} path must be a relative repository path")
    result = subprocess.run(["git", "show", f"{revision}:{path}"], cwd=repository_root,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise ValueError(f"{label} is absent from git revision {revision}: {path}")
    return result.stdout


def fixture_generation_provenance(manifest: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(manifest, Mapping):
        raise ValueError("shared fixture manifest must be an object")
    source = manifest.get("source_fixture")
    generator = manifest.get("generator")
    if not isinstance(source, Mapping) or not isinstance(generator, Mapping):
        raise ValueError("shared fixture manifest lacks explicit source/generator provenance")
    chains = source.get("chains")
    if not isinstance(chains, Mapping):
        raise ValueError("shared fixture manifest lacks source chains provenance")
    source_artifact = chains.get("path")
    source_branch = source.get("branch")
    if not isinstance(source_artifact, str) or not isinstance(source_branch, str) or not source_branch:
        raise ValueError("shared fixture source artifact/branch must be nonempty strings")
    generator_path = generator.get("path")
    if not isinstance(generator_path, str) or not generator_path:
        raise ValueError("shared fixture generator path must be a nonempty string")
    return {"source_artifact": source_artifact, "source_commit": _revision(source.get("revision"), "source revision"),
            "source_branch": source_branch, "generator_path": generator_path,
            "generator_revision": _revision(generator.get("revision"), "generator revision")}


def verify_shared_fixture_provenance(manifest: Mapping[str, Any], rows: Iterable[dict], *,
                                     repository_root: str | Path) -> dict[str, str]:
    provenance = fixture_generation_provenance(manifest)
    root = Path(repository_root)
    source = manifest["source_fixture"]
    for name in ("chains", "natural_costs", "manifest"):
        entry = source.get(name)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ValueError(f"shared fixture {name} provenance is malformed")
        blob = _git_blob(root, provenance["source_commit"], entry["path"], f"source fixture {name}")
        if hashlib.sha256(blob).hexdigest() != entry.get("sha256"):
            raise ValueError(f"shared fixture {name} sha256 differs from source revision")
    _git_blob(root, provenance["generator_revision"], provenance["generator_path"], "fixture generator")
    source_sha = source["chains"]["sha256"]
    for row in rows:
        if (row.get("source_artifact") != provenance["source_artifact"]
                or row.get("source_commit") != provenance["source_commit"]
                or row.get("source_branch") != provenance["source_branch"]
                or row.get("source_sha256") != source_sha):
            raise ValueError("fixture row source provenance contradicts exchange manifest")
    return provenance


def generate_shared_fixture(out_dir: str | Path, *, source_artifact: str = _FIXTURE_ARTIFACT,
                            source_commit: str = _FIXTURE_SOURCE_COMMIT,
                            source_branch: str = "paper/iclr-tts-baselines",
                            generator_path: str | None = _FIXTURE_GENERATOR_PATH,
                            generator_revision: str | None = _FIXTURE_GENERATOR_REVISION,
                            repository_root: str | Path | None = None) -> dict[str, Path]:
    root = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[2]
    if source_artifact != _FIXTURE_ARTIFACT or not isinstance(source_branch, str) or not source_branch:
        raise ValueError("shared fixture source provenance is not the fixed fixture artifact")
    if (generator_path is None) != (generator_revision is None):
        raise ValueError("shared fixture generator path and revision must be supplied together")
    explicit_provenance = generator_path is not None
    if explicit_provenance:
        source_commit = _revision(source_commit, "source revision")
        generator_revision = _revision(generator_revision, "generator revision")
        if not isinstance(generator_path, str) or not generator_path:
            raise ValueError("generator path must be a nonempty string")
    fixture = root / "tests" / "fixtures" / "iclr_tts_b0"
    chain_path, costs_path, manifest_path = (fixture / "chains.jsonl", fixture / "natural_costs.jsonl",
                                              fixture / "manifest.jsonl")
    chains, source_sha = load_chains_snapshot(chain_path)
    costs, costs_sha = load_natural_costs_snapshot(costs_path)
    _, manifest_sha = _load_select_manifest_snapshot(manifest_path)
    registry = load_b0_registry(root / "results" / "iclr" / "tts_baselines" / "b0_registry.json")
    generation_seed = registry["select_chain_asset"]["generation_seed"]
    common = {"source_artifact": source_artifact, "source_sha256": source_sha,
              "source_branch": source_branch, "source_commit": source_commit,
              "natural_costs": costs, "natural_costs_sha256": costs_sha, "generation_seed": generation_seed}
    natural_rows = build_chain_rows(chains, natural, method_name="natural", **common)
    sc1_rows, sc1_children = build_question_export(
        chains, natural, 1, method_family="self_consistency", method_name="self_consistency",
        config_id="sc-n1-grid-unselected-v1", **common)
    sc2_rows, sc2_children = build_question_export(
        chains, natural, 2, method_family="self_consistency", method_name="self_consistency",
        config_id="sc-n2-grid-unselected-v1", **common)
    controller_rows, controller_children = build_question_export(
        chains, always_fire, 1, method_family="controller", method_name="controller_always_fire",
        config_id="controller-always-fire-fixture-v1", **common)
    rows, children = natural_rows + sc1_rows + sc2_rows + controller_rows, \
        sc1_children + sc2_children + controller_children
    if not sc1_equals_natural_first_sibling(sc1_rows, sc1_children, chains, costs):
        raise AssertionError("SC@1 fixture must equal natural first siblings")
    if not controller1_equals_chain_first_sibling(controller_rows, controller_children, chains, always_fire, costs):
        raise AssertionError("controller@1 fixture must equal its chain first siblings")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row_path, child_path = out_dir / "shared_fixture.rows.jsonl", out_dir / "shared_fixture.children.jsonl"
    manifest_out = out_dir / "shared_fixture.manifest.json"
    write_tts_jsonl(row_path, rows)
    write_children_jsonl(child_path, children)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_fixture": {
            "chains": {"path": "tests/fixtures/iclr_tts_b0/chains.jsonl", "sha256": source_sha},
            "natural_costs": {"path": "tests/fixtures/iclr_tts_b0/natural_costs.jsonl", "sha256": costs_sha},
            "manifest": {"path": "tests/fixtures/iclr_tts_b0/manifest.jsonl", "sha256": manifest_sha},
        },
        "exchange": {
            "rows": {"path": "results/iclr/tts_baselines/shared_fixture.rows.jsonl",
                     "sha256": sha256_file(row_path), "row_count": len(rows)},
            "children": {"path": "results/iclr/tts_baselines/shared_fixture.children.jsonl",
                         "sha256": sha256_file(child_path), "row_count": len(children)},
        },
        "tie_equivalence_rule": "Only non-bool integers vote; None never votes; equal integers cluster; tied clusters choose earliest rep.",
        "invariants": {"sc1_equals_natural_first_sibling": True,
                       "controller1_equals_chain_first_sibling": True},
    }
    if explicit_provenance:
        manifest["source_fixture"].update(revision=source_commit, branch=source_branch)
        manifest["generator"] = {"path": generator_path, "revision": generator_revision}
        verify_shared_fixture_provenance(manifest, rows, repository_root=root)
    write_json_atomic(manifest_out, manifest)
    return {"shared_fixture.rows.jsonl": row_path, "shared_fixture.children.jsonl": child_path,
            "shared_fixture.manifest.json": manifest_out}
