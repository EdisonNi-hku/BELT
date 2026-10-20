from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import bisect
from typing import Any, Callable, Iterable, Mapping, Sequence

from rl_controller.scripts.score_scaleup_s1 import (
    CHAIN_FIELDS,
    _validate_chain,
    validate_scored_chain,
)


MANIFEST_SHA256 = "efcacbe076e51c1739be9a1266e6d1176786eaa93f2cc1da08a46dddc9f9e1ae"
SCORED_CHAINS_SHA256 = "fce63336d1a0ab32c58688739ebc32c1cb855393a6b8b514f02c10f9d3dda825"
S1_CHECKPOINT_SHA256 = "863d4257906d1c78e7c1d9d5e1720f322a4e89fb9d64ef530059abc0927cf418"
CHAINS_SHA256 = "f2ee19e3c1f25c90dfca85d48abbabdcb2618fa864ea53e6aaae0d8ed2b15c12"
ROWS_SHA256 = "8c4915663d7d04abc968623751c0ad290e5da5cacc5df106631f5ea88d648c14"
BASE_INTEGRITY_SHA256 = "ef214c0780f3126f137955542732ed23d37955025720303a167f429bc156cba9"
FIT_QUESTION_COUNT = 319
FIT_CHAIN_COUNT = 6480
MANIFEST_SPLIT_COUNTS = {"fit": 319, "select": 145, "report": 136}
GENERATION_CAP = 12_000
BASELINE_POLICY_NAMES = ("row2", "natural_end", "fixed_cap_8", "fixed_cap_16",
                         "fixed_cap_32", "fixed_cap_64")
GUARD_POLICY_NAMES = ("natural_end", "first_usable", "fixed_cap_8", "fixed_cap_16",
                      "fixed_cap_32", "fixed_cap_64")
SCORED_INTEGRITY_FIELDS = {
    "split", "expected_chains", "observed_chains", "s1_checkpoint_sha256",
    "source_fit_files", "score_shards", "source_hashes", "scored_chains_sha256",
}


def _sha(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _validate_scored_provenance(integrity: dict[str, Any]) -> None:
    sources = integrity.get("source_fit_files")
    if (not isinstance(sources, list) or len(sources) != 16
            or any(not isinstance(item, dict) or set(item) != {"path", "sha256"}
                   or type(item["path"]) is not str or not item["path"]
                   or not _sha(item["sha256"]) for item in sources)
            or len({item["path"] for item in sources}) != 16):
        raise ValueError("scored integrity source_fit_files schema mismatch")
    shards = integrity.get("score_shards")
    fields = {"path", "integrity_sha256", "summary_sha256", "scored_chains_sha256"}
    if (not isinstance(shards, list) or len(shards) != 4
            or any(not isinstance(item, dict) or set(item) != fields
                   or type(item["path"]) is not str or not item["path"]
                   or any(not _sha(item[field]) for field in fields - {"path"})
                   for item in shards)
            or len({item["path"] for item in shards}) != 4):
        raise ValueError("scored integrity score_shards schema mismatch")


@dataclass(frozen=True)
class ProbeFeatures:
    q: float
    v: float
    depth: float
    remaining: float


@dataclass(frozen=True)
class Rule:
    name: str
    predicate: Callable[[ProbeFeatures], bool] | None = None
    cap: int | None = None

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("rule name must be a nonempty exact string")
        if self.cap is not None and (type(self.cap) is not int or self.cap < 1):
            raise ValueError("rule cap must be an exact positive integer")
        if self.predicate is not None and not callable(self.predicate):
            raise TypeError("rule predicate must be callable")
        actions = int(self.predicate is not None) + int(self.cap is not None)
        if self.name == "natural_end":
            if actions:
                raise ValueError("natural_end rule cannot have a predicate or cap")
        elif actions != 1:
            raise ValueError("only natural_end may omit an action; non-natural Rule must define "
                             "exactly one of predicate or cap")


@dataclass(frozen=True)
class Replay:
    qkey: str
    rep: int
    endpoint: str
    probe_index: int | None
    answer: int | str | None
    correct: bool
    tokens: int

    def __post_init__(self) -> None:
        if type(self.qkey) is not str or not self.qkey:
            raise ValueError("replay qkey must be a nonempty exact string")
        if type(self.rep) is not int or self.rep < 0:
            raise ValueError("replay rep must be an exact nonnegative integer")
        if self.endpoint not in ("forced", "natural"):
            raise ValueError("replay endpoint must be forced or natural")
        if self.endpoint == "forced":
            if type(self.probe_index) is not int or self.probe_index < 0:
                raise ValueError("forced replay requires an exact nonnegative probe_index")
        elif self.probe_index is not None:
            raise ValueError("natural replay requires probe_index None")
        if (self.answer is not None
                and (isinstance(self.answer, bool) or type(self.answer) not in (int, str))):
            raise ValueError("replay answer must be an exact int/string or None")
        if type(self.correct) is not bool:
            raise ValueError("replay correct must be an exact bool")
        if type(self.tokens) is not int or self.tokens < 0:
            raise ValueError("replay tokens must be an exact nonnegative integer")


SHIPPED_RULE = Rule("shipped_global", lambda x: x.q >= x.v + 0.10)
FIRST_USABLE_RULE = Rule("first_usable", lambda _x: True)
NATURAL_END_RULE = Rule("natural_end")


def _snapshot(path: Path) -> tuple[bytes, str]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read input snapshot: {path}") from error
    return payload, hashlib.sha256(payload).hexdigest()


def _jsonl(payload: bytes, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("row is not an object")
            rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSONL snapshot at {path}:{locals().get('number', 0)}") from error
    return rows


def _manifest_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from rl_controller.splits import question_key

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if set(row) != {"qkey", "question_sha256", "question", "gold", "split",
                        "allocation_subset", "target_chains"}:
            raise ValueError("manifest row schema mismatch")
        qkey = row["qkey"]
        if type(qkey) is not str or not qkey or qkey in result:
            raise ValueError("manifest qkey must be unique nonempty exact strings")
        if type(row["split"]) is not str or row["split"] not in MANIFEST_SPLIT_COUNTS:
            raise ValueError("manifest split mismatch")
        if type(row["target_chains"]) is not int or row["target_chains"] < 1:
            raise ValueError("manifest target_chains must be exact positive integers")
        if isinstance(row["gold"], bool) or type(row["gold"]) not in (int, str):
            raise ValueError("manifest gold must be exact int or string")
        if type(row["question"]) is not str or not row["question"]:
            raise ValueError("manifest question must be a nonempty exact string")
        if question_key(row["question"]) != qkey:
            raise ValueError("manifest question/qkey mismatch")
        if not _sha(row["question_sha256"]):
            raise ValueError("manifest question_sha256 mismatch")
        if row["question_sha256"] != hashlib.sha256(row["question"].encode("utf-8")).hexdigest():
            raise ValueError("manifest question_sha256 content mismatch")
        if type(row["allocation_subset"]) is not bool:
            raise ValueError("manifest allocation_subset must be an exact bool")
        result[qkey] = row
    if dict(Counter(row["split"] for row in rows)) != MANIFEST_SPLIT_COUNTS:
        raise ValueError("manifest split counts mismatch")
    return result


def validate_chain(row: dict[str, Any], manifest_row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict) or set(row) != CHAIN_FIELDS | {"scores"}:
        raise ValueError("scored chain schema mismatch")
    source = dict(row)
    source.pop("scores")
    _validate_chain(source, manifest_row)
    validate_scored_chain(row)
    return row


def _validate_replay_chain(row: object) -> dict[str, Any]:
    if not isinstance(row, dict) or set(row) != CHAIN_FIELDS | {"scores"}:
        raise ValueError("scored chain schema mismatch")
    qkey, rep, gold = row.get("qkey"), row.get("rep"), row.get("gold")
    if type(qkey) is not str or not qkey:
        raise ValueError("chain qkey must be a nonempty exact string")
    if type(rep) is not int or rep < 0:
        raise ValueError("chain rep must be an exact nonnegative integer")
    if isinstance(gold, bool) or type(gold) not in (int, str):
        raise ValueError("chain gold must be exact int or string")
    source = dict(row)
    source.pop("scores")
    _validate_chain(source, {"gold": gold, "target_chains": rep + 1})
    validate_scored_chain(row)
    return row


def load_frozen_fit_payloads(
    manifest_bytes: bytes, scored_bytes: bytes, integrity_bytes: bytes,
    *, expected_integrity_sha256: str,
) -> list[dict[str, Any]]:
    if any(type(payload) is not bytes for payload in (
            manifest_bytes, scored_bytes, integrity_bytes)):
        raise TypeError("held fit payloads must be exact bytes")
    manifest_path = Path("<held-manifest>")
    scored_path = Path("<held-scored>")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    scored_hash = hashlib.sha256(scored_bytes).hexdigest()
    integrity_hash = hashlib.sha256(integrity_bytes).hexdigest()
    if not _sha(expected_integrity_sha256):
        raise ValueError("expected_integrity_sha256 must be 64 lowercase hexadecimal characters")
    if integrity_hash != expected_integrity_sha256:
        raise ValueError("scored integrity hash mismatch")
    if manifest_hash != MANIFEST_SHA256:
        raise ValueError("manifest hash mismatch")
    if scored_hash != SCORED_CHAINS_SHA256:
        raise ValueError("registered scored chains hash mismatch")
    try:
        integrity = json.loads(integrity_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid scored integrity JSON") from error
    if not isinstance(integrity, dict) or set(integrity) != SCORED_INTEGRITY_FIELDS:
        raise ValueError("scored integrity schema mismatch")
    if integrity.get("split") != "fit":
        raise ValueError("scored integrity split mismatch")
    if integrity.get("s1_checkpoint_sha256") != S1_CHECKPOINT_SHA256:
        raise ValueError("scored integrity checkpoint mismatch")
    if (type(integrity.get("expected_chains")) is not int
            or type(integrity.get("observed_chains")) is not int
            or integrity["expected_chains"] != FIT_CHAIN_COUNT
            or integrity["observed_chains"] != FIT_CHAIN_COUNT):
        raise ValueError("scored integrity chain counts mismatch")
    if integrity.get("source_hashes") != {
        "manifest_sha256": "efcacbe076e51c1739be9a1266e6d1176786eaa93f2cc1da08a46dddc9f9e1ae",
        "chains_sha256": CHAINS_SHA256, "rows_sha256": ROWS_SHA256,
        "integrity_sha256": BASE_INTEGRITY_SHA256,
    }:
        raise ValueError("scored integrity source hashes mismatch")
    if integrity.get("scored_chains_sha256") != scored_hash:
        raise ValueError("scored chains hash mismatch")
    _validate_scored_provenance(integrity)

    manifest = _manifest_rows(_jsonl(manifest_bytes, manifest_path))
    if sum(row["split"] == "fit" for row in manifest.values()) != FIT_QUESTION_COUNT:
        raise ValueError("fit question coverage mismatch")
    expected = {(qkey, rep) for qkey, item in manifest.items() if item["split"] == "fit"
                for rep in range(item["target_chains"])}
    if len(expected) != FIT_CHAIN_COUNT:
        raise ValueError("manifest fit identity count mismatch")
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _jsonl(scored_bytes, scored_path):
        qkey, rep = row.get("qkey"), row.get("rep")
        if type(qkey) is not str or not qkey or type(rep) is not int or rep < 0:
            raise ValueError("scored qkey/rep identity mismatch")
        identity = (qkey, rep)
        if identity in observed:
            raise ValueError(f"duplicate scored chain identity: {identity}")
        manifest_row = manifest.get(qkey)
        if manifest_row is None or manifest_row["split"] != "fit":
            raise ValueError(f"non-fit scored identity: {identity}")
        observed[identity] = validate_chain(row, manifest_row)
    if set(observed) != expected:
        raise ValueError("scored fit identity coverage mismatch")
    return [observed[key] for key in sorted(observed)]


def load_frozen_fit(
    manifest_path: str | Path, scored_path: str | Path, integrity_path: str | Path,
    *, expected_integrity_sha256: str,
) -> list[dict[str, Any]]:
    manifest_path, scored_path, integrity_path = map(
        Path, (manifest_path, scored_path, integrity_path),
    )
    manifest_bytes, manifest_hash = _snapshot(manifest_path)
    scored_bytes, scored_hash = _snapshot(scored_path)
    integrity_bytes, _integrity_hash = _snapshot(integrity_path)
    result = load_frozen_fit_payloads(
        manifest_bytes, scored_bytes, integrity_bytes,
        expected_integrity_sha256=expected_integrity_sha256,
    )
    if (_snapshot(manifest_path)[1] != manifest_hash
            or _snapshot(scored_path)[1] != scored_hash
            or _snapshot(integrity_path)[0] != integrity_bytes):
        raise ValueError("input changed during load")
    return result


def probe_features(probe: Mapping[str, Any], score: Mapping[str, Any]) -> ProbeFeatures:
    block, prefix_len = probe.get("block"), probe.get("prefix_len")
    if type(block) is not int or block < 1:
        raise ValueError("probe block must be an exact positive integer")
    if type(prefix_len) is not int or prefix_len < 1:
        raise ValueError("probe prefix_len must be an exact positive integer")
    if not isinstance(score, Mapping) or set(score) != {"q", "v"}:
        raise ValueError("score fields must be exactly q/v")
    values: list[float] = []
    for field in ("q", "v"):
        value = score[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("q/v must be finite numeric probabilities")
        if not 0 <= value <= 1:
            raise ValueError("q/v must be in [0,1]")
        values.append(float(value))
    return ProbeFeatures(values[0], values[1], min(block / 64.0, 1.0),
                         max(GENERATION_CAP - prefix_len, 0) / GENERATION_CAP)


def replay_rule(chain: dict[str, Any], rule: Rule) -> Replay:
    if not isinstance(rule, Rule):
        raise TypeError("rule must be an exact Rule")
    chain = _validate_replay_chain(chain)
    return _replay_trusted(chain, rule)


def _replay_trusted(chain: dict[str, Any], rule: Rule) -> Replay:
    qkey, rep = chain.get("qkey"), chain.get("rep")
    probes, scores = chain.get("probes"), chain.get("scores")
    chosen: int | None = None
    for index, (probe, score) in enumerate(zip(probes, scores)):
        features = probe_features(probe, score)
        if rule.cap is not None and probe["block"] >= rule.cap:
            chosen = index
            break
        if rule.predicate is not None and rule.predicate(features):
            chosen = index
            break
    if chosen is not None:
        probe = probes[chosen]
        forced = probe.get("forced")
        forced_tokens = probe.get("forced_tokens")
        if type(forced_tokens) is not int or forced_tokens < 0:
            raise ValueError("forced_tokens must be an exact nonnegative integer")
        answer = forced
        correct = answer is not None and type(answer) is type(chain.get("gold")) and answer == chain["gold"]
        return Replay(qkey, rep, "forced", chosen, answer, correct,
                      probe["prefix_len"] + forced_tokens)
    thinking_tokens = chain.get("thinking_tokens")
    if type(thinking_tokens) is not int or thinking_tokens < 0:
        raise ValueError("thinking_tokens must be an exact nonnegative integer")
    if type(chain.get("correct")) is not bool:
        raise ValueError("natural correct must be an exact bool")
    return Replay(qkey, rep, "natural", None, chain.get("answer"), chain["correct"], thinking_tokens)


def guard_rules() -> dict[str, Rule]:
    return {
        "natural_end": NATURAL_END_RULE,
        "first_usable": FIRST_USABLE_RULE,
        **{f"fixed_cap_{cap}": Rule(f"fixed_cap_{cap}", cap=cap)
           for cap in (8, 16, 32, 64)},
    }


FITTING_SEEDS = (1729, 2718)
LOGISTIC_CS = (0.1, 1.0, 10.0)
LINEAR_A = (-0.30, -0.20, -0.10, 0.00, 0.10)
LINEAR_B = (-0.30, -0.15, 0.00, 0.15, 0.30)
DEPTH_BINS = (0.0, 0.125, 0.25, 0.5, 1.0)
MONOTONE_VALUES = LINEAR_A
LOW_CAPACITY_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


class InsufficientSupport(ValueError):
    pass


def _checked_qkeys(qkeys: Iterable[str]) -> tuple[str, ...]:
    values = tuple(qkeys)
    if any(type(qkey) is not str or not qkey for qkey in values):
        raise ValueError("qkeys must be nonempty exact strings")
    if len(set(values)) != len(values):
        raise ValueError("qkeys must be unique")
    return values


def _group_folds(qkeys: Iterable[str], seed: int, folds: int, prefix: str) -> dict[str, int]:
    values = _checked_qkeys(qkeys)
    if type(seed) is not int or seed not in FITTING_SEEDS:
        raise ValueError("fitting seed must be exactly 1729 or 2718")
    ordered = sorted(values, key=lambda qkey: hashlib.sha256(
        f"{prefix}{seed}{qkey}".encode("utf-8")).digest())
    return {qkey: index % folds for index, qkey in enumerate(ordered)}


def outer_folds(qkeys: Iterable[str], seed: int) -> dict[str, int]:
    return _group_folds(qkeys, seed, 5, "spec-a-phase0-outer-v1")


def inner_folds(qkeys: Iterable[str], seed: int, outer_fold: int) -> dict[str, int]:
    if type(outer_fold) is not int or not 0 <= outer_fold < 5:
        raise ValueError("outer_fold must be an exact integer in [0,4]")
    values = _checked_qkeys(qkeys)
    if type(seed) is not int or seed not in FITTING_SEEDS:
        raise ValueError("fitting seed must be exactly 1729 or 2718")
    ordered = sorted(values, key=lambda qkey: hashlib.sha256(
        f"spec-a-phase0-inner-v1{seed}{outer_fold}{qkey}".encode("utf-8")).digest())
    return {qkey: index % 4 for index, qkey in enumerate(ordered)}


def assert_disjoint_qkeys(training_qkeys: Iterable[str], evaluation_qkeys: Iterable[str]) -> None:
    if set(training_qkeys) & set(evaluation_qkeys):
        raise AssertionError("training and evaluation qkeys must be disjoint")


@dataclass(frozen=True)
class WeightedExample:
    qkey: str
    p: float
    depth: float
    label: int
    weight: float

    def __post_init__(self) -> None:
        if (type(self.qkey) is not str or not self.qkey
                or isinstance(self.p, bool) or not math.isfinite(self.p) or not 0 <= self.p <= 1
                or isinstance(self.depth, bool) or not math.isfinite(self.depth) or not 0 <= self.depth <= 1
                or self.label not in (0, 1) or type(self.label) is not int
                or isinstance(self.weight, bool) or not math.isfinite(self.weight) or self.weight <= 0):
            raise ValueError("invalid weighted calibration example")


def calibration_examples(
    chains: Sequence[dict[str, Any]], training_qkeys: Iterable[str],
    evaluation_qkeys: Iterable[str] = (),
) -> tuple[list[WeightedExample], list[WeightedExample]]:
    return _calibration_examples_impl(chains, training_qkeys, evaluation_qkeys, trusted=False)


def _calibration_examples_trusted(
    chains: Sequence[dict[str, Any]], training_qkeys: Iterable[str],
    evaluation_qkeys: Iterable[str] = (),
) -> tuple[list[WeightedExample], list[WeightedExample]]:
    return _calibration_examples_impl(chains, training_qkeys, evaluation_qkeys, trusted=True)


def _calibration_examples_impl(
    chains: Sequence[dict[str, Any]], training_qkeys: Iterable[str],
    evaluation_qkeys: Iterable[str], *, trusted: bool,
) -> tuple[list[WeightedExample], list[WeightedExample]]:
    training = set(_checked_qkeys(training_qkeys))
    assert_disjoint_qkeys(training, evaluation_qkeys)
    by_question: dict[str, list[dict[str, Any]]] = {qkey: [] for qkey in training}
    for chain in chains:
        if chain.get("qkey") in training:
            by_question[chain["qkey"]].append(chain if trusted else _validate_replay_chain(chain))
    if any(not values for values in by_question.values()):
        raise ValueError("training qkey has no chains")
    q_rows: list[WeightedExample] = []
    v_rows: list[WeightedExample] = []
    for qkey in sorted(by_question):
        question_chains = by_question[qkey]
        chain_weight = 1.0 / len(question_chains)
        for chain in sorted(question_chains, key=lambda item: item["rep"]):
            probes, scores = chain["probes"], chain["scores"]
            if not probes:
                continue
            probe_weight = chain_weight / len(probes)
            for probe, score in zip(probes, scores):
                features = probe_features(probe, score)
                forced = probe["forced"]
                label = int(forced is not None and type(forced) is type(chain["gold"])
                            and forced == chain["gold"])
                q_rows.append(WeightedExample(qkey, features.q, features.depth, label, probe_weight))
            last = probe_features(probes[-1], scores[-1])
            v_rows.append(WeightedExample(qkey, last.v, last.depth, int(chain["correct"]), chain_weight))
    return q_rows, v_rows


def _logit(probability: float) -> float:
    clipped = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


@dataclass(frozen=True)
class LogisticModel:
    coefficients: tuple[float, ...]

    def predict_features(self, features: Sequence[float]) -> float:
        if len(features) != len(self.coefficients):
            raise ValueError("logistic feature width mismatch")
        return _sigmoid(sum(coefficient * value
                            for coefficient, value in zip(self.coefficients, features)))

    def calibrate(self, probability: float, depth: float) -> float:
        return self.predict_features((1.0, _logit(probability), depth))


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise InsufficientSupport("singular logistic fit")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            scale = augmented[row][column]
            augmented[row] = [left - scale * right
                              for left, right in zip(augmented[row], augmented[column])]
    return [augmented[row][-1] for row in range(size)]


def fit_weighted_logistic(
    features: Sequence[Sequence[float]], labels: Sequence[int], weights: Sequence[float], c: float,
) -> LogisticModel:
    if (isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c)
            or float(c) not in LOGISTIC_CS or not features
            or not (len(features) == len(labels) == len(weights))):
        raise ValueError("invalid frozen logistic fit")
    width = len(features[0])
    if (width < 2 or any(len(row) != width for row in features)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for row in features for value in row)
            or any(type(label) is not int or label not in (0, 1) for label in labels)
            or any(isinstance(weight, bool) or not isinstance(weight, (int, float))
                   or not math.isfinite(weight) or weight <= 0 for weight in weights)):
        raise ValueError("logistic feature width mismatch")
    if set(labels) != {0, 1}:
        raise InsufficientSupport(
            "positive-weight logistic samples must contain both exact labels 0 and 1")
    coefficients = [0.0] * width
    penalty = 1.0 / float(c)
    def objective(values: Sequence[float]) -> float:
        loss = 0.0
        for row, label, weight in zip(features, labels, weights):
            linear = sum(a * b for a, b in zip(values, row))
            loss += weight * (max(linear, 0.0) - label * linear
                              + math.log1p(math.exp(-abs(linear))))
        loss += .5 * penalty * sum(value * value for value in values[1:])
        return loss
    current_objective = objective(coefficients)
    for _iteration in range(100):
        gradient = [0.0] * width
        hessian = [[0.0] * width for _ in range(width)]
        for row, label, weight in zip(features, labels, weights):
            probability = _sigmoid(sum(a * b for a, b in zip(coefficients, row)))
            curvature = weight * probability * (1.0 - probability)
            for j in range(width):
                gradient[j] += weight * (probability - label) * row[j]
                for k in range(width):
                    hessian[j][k] += curvature * row[j] * row[k]
        for j in range(1, width):
            gradient[j] += penalty * coefficients[j]
            hessian[j][j] += penalty
        hessian[0][0] += 1e-10
        if (not all(math.isfinite(value) for value in gradient)
                or not all(math.isfinite(value) for row in hessian for value in row)):
            raise InsufficientSupport("non-finite logistic system")
        step = _solve(hessian, gradient)
        if not all(math.isfinite(value) for value in step):
            raise InsufficientSupport("non-finite logistic step")
        scale = 1.0
        accepted: list[float] | None = None
        accepted_objective = math.inf
        while scale >= 2.0 ** -30:
            proposal = [value - scale * delta for value, delta in zip(coefficients, step)]
            if all(math.isfinite(value) for value in proposal):
                proposal_objective = objective(proposal)
                if math.isfinite(proposal_objective) and proposal_objective <= current_objective + 1e-12:
                    accepted, accepted_objective = proposal, proposal_objective
                    break
            scale *= .5
        if accepted is None:
            raise InsufficientSupport("logistic line search failed")
        step_norm = max(abs(scale * value) for value in step)
        improvement = current_objective - accepted_objective
        coefficients, current_objective = accepted, accepted_objective
        if step_norm <= 1e-9 or improvement <= 1e-12 * (1.0 + abs(current_objective)):
            return LogisticModel(tuple(coefficients))
    raise InsufficientSupport("logistic fit did not converge in 100 iterations")


def _fit_calibrator_with_support(
    examples: Sequence[WeightedExample], c: float, positives: int, negatives: int,
) -> LogisticModel:
    if positives < 20 or negatives < 20:
        raise InsufficientSupport("calibration requires at least 20 positive and 20 negative labels")
    example_positives = sum(row.label == 1 for row in examples)
    example_negatives = sum(row.label == 0 for row in examples)
    if example_positives < 20 or example_negatives < 20:
        raise InsufficientSupport(
            "calibration fitting examples require at least 20 positive and 20 negative labels")
    return fit_weighted_logistic(
        [(1.0, _logit(example.p), example.depth) for example in examples],
        [example.label for example in examples], [example.weight for example in examples], c,
    )


def fit_calibrator(examples: Sequence[WeightedExample], c: float) -> LogisticModel:
    return _fit_calibrator_with_support(examples, c,
                                        sum(row.label == 1 for row in examples),
                                        sum(row.label == 0 for row in examples))


def select_calibration_c(
    examples: Sequence[WeightedExample], fold_by_qkey: Mapping[str, int],
) -> float:
    if set(fold_by_qkey.values()) != {0, 1, 2, 3}:
        raise ValueError("calibration selection requires exactly four inner folds")
    if any(example.qkey not in fold_by_qkey for example in examples):
        raise ValueError("calibration example has no inner fold")
    scores: list[tuple[float, float]] = []
    for c in LOGISTIC_CS:
        losses: list[float] = []
        for fold in range(4):
            train_qkeys = {qkey for qkey, value in fold_by_qkey.items() if value != fold}
            validation_qkeys = {qkey for qkey, value in fold_by_qkey.items() if value == fold}
            assert_disjoint_qkeys(train_qkeys, validation_qkeys)
            training = [row for row in examples if row.qkey in train_qkeys]
            validation = [row for row in examples if row.qkey in validation_qkeys]
            model = fit_calibrator(training, c)
            for qkey in sorted(validation_qkeys):
                q_rows = [row for row in validation if row.qkey == qkey]
                if q_rows:
                    losses.append(question_log_loss(q_rows, model))
        if len(losses) != len(fold_by_qkey):
            raise InsufficientSupport("calibration held-out question coverage mismatch")
        scores.append((sum(losses) / len(losses), c))
    return min(scores)[1]


def question_log_loss(examples: Sequence[WeightedExample], model: LogisticModel) -> float:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for example in examples:
        probability = min(max(model.calibrate(example.p, example.depth), 1e-15), 1 - 1e-15)
        loss = -(example.label * math.log(probability) + (1 - example.label) * math.log(1 - probability))
        grouped.setdefault(example.qkey, []).append((loss, example.weight))
    if not grouped:
        raise InsufficientSupport("question log loss is undefined")
    return sum(sum(loss * weight for loss, weight in values) / sum(weight for _, weight in values)
               for values in grouped.values()) / len(grouped)


def calibrated_global_rule(q_model: LogisticModel, v_model: LogisticModel) -> Rule:
    return Rule("depth_calibrated_global", lambda x: q_model.calibrate(x.q, x.depth)
                >= v_model.calibrate(x.v, x.depth) + 0.10)


def linear_candidates() -> tuple[tuple[float, float], ...]:
    return tuple((a, b) for a in LINEAR_A for b in LINEAR_B)


def monotone_candidates() -> tuple[tuple[float, float, float, float], ...]:
    return tuple((a, b, c, d) for a in MONOTONE_VALUES for b in MONOTONE_VALUES
                 for c in MONOTONE_VALUES for d in MONOTONE_VALUES if a <= b <= c <= d)


def low_capacity_candidates() -> tuple[tuple[float, float], ...]:
    return tuple((c, threshold) for c in LOGISTIC_CS for threshold in LOW_CAPACITY_THRESHOLDS)


def depth_bin(depth: float) -> int:
    if isinstance(depth, bool) or not math.isfinite(depth) or not 0 <= depth <= 1:
        raise ValueError("depth must be finite in [0,1]")
    if depth < .125:
        return 0
    if depth < .25:
        return 1
    if depth < .5:
        return 2
    return 3


def linear_depth_rule(q_model: LogisticModel, v_model: LogisticModel, a: float, b: float) -> Rule:
    if (a, b) not in linear_candidates():
        raise ValueError("linear depth parameters are outside the frozen grammar")
    return Rule(f"row3:{a:.2f},{b:.2f}", lambda x: q_model.calibrate(x.q, x.depth)
                >= v_model.calibrate(x.v, x.depth) - (a + b * x.depth))


def monotone_depth_rule(q_model: LogisticModel, v_model: LogisticModel,
                        margins: tuple[float, float, float, float]) -> Rule:
    if margins not in monotone_candidates():
        raise ValueError("monotone margins are outside the frozen grammar")
    return Rule("row4:" + ",".join(f"{value:.2f}" for value in margins),
                lambda x: q_model.calibrate(x.q, x.depth)
                >= v_model.calibrate(x.v, x.depth) - margins[depth_bin(x.depth)])


def low_capacity_rule(model: LogisticModel, threshold: float) -> Rule:
    if threshold not in LOW_CAPACITY_THRESHOLDS or len(model.coefficients) != 5:
        raise ValueError("low-capacity policy is outside the frozen grammar")
    return Rule(f"row5:{threshold:.2f}", lambda x: model.predict_features(
        (1.0, _logit(x.q), _logit(x.v), x.depth, x.remaining)) >= threshold)


@dataclass(frozen=True)
class ObservableExample:
    qkey: str
    features: tuple[float, float, float, float, float]
    label: int
    weight: float


def _chains_for_qkeys(chains: Sequence[dict[str, Any]], qkeys: set[str]) -> list[dict[str, Any]]:
    selected = [_validate_replay_chain(chain) for chain in chains if chain.get("qkey") in qkeys]
    return _check_chain_coverage(selected, qkeys)


def _chains_for_qkeys_trusted(
    chains: Sequence[dict[str, Any]], qkeys: set[str],
) -> list[dict[str, Any]]:
    return _check_chain_coverage([chain for chain in chains if chain["qkey"] in qkeys], qkeys)


def _check_chain_coverage(
    selected: list[dict[str, Any]], qkeys: set[str],
) -> list[dict[str, Any]]:
    if {chain["qkey"] for chain in selected} != qkeys:
        raise ValueError("qkey chain coverage mismatch")
    return selected


def _forced_support(chains: Sequence[dict[str, Any]]) -> tuple[int, int]:
    labels = [int(probe["forced"] is not None
                  and type(probe["forced"]) is type(chain["gold"])
                  and probe["forced"] == chain["gold"])
              for chain in chains for probe in chain["probes"]]
    return sum(labels), len(labels) - sum(labels)


def fit_low_capacity(
    chains: Sequence[dict[str, Any]], training_qkeys: Iterable[str], c: float,
    evaluation_qkeys: Iterable[str] = (),
) -> LogisticModel:
    training = set(_checked_qkeys(training_qkeys))
    assert_disjoint_qkeys(training, evaluation_qkeys)
    selected = _chains_for_qkeys(chains, training)
    return _fit_low_capacity_trusted(selected, training, c)


def _fit_low_capacity_trusted(
    selected: Sequence[dict[str, Any]], training: set[str], c: float,
) -> LogisticModel:
    by_q = Counter(chain["qkey"] for chain in selected)
    examples: list[ObservableExample] = []
    for chain in selected:
        probes = chain["probes"]
        if not probes:
            continue
        weight = 1.0 / (by_q[chain["qkey"]] * len(probes))
        for probe, score in zip(probes, chain["scores"]):
            x = probe_features(probe, score)
            forced = probe["forced"]
            label = int(forced is not None and type(forced) is type(chain["gold"])
                        and forced == chain["gold"])
            examples.append(ObservableExample(chain["qkey"],
                (1.0, _logit(x.q), _logit(x.v), x.depth, x.remaining), label, weight))
    positives, negatives = _forced_support(selected)
    if positives < 20 or negatives < 20:
        raise InsufficientSupport("row5 requires at least 20 positive and 20 negative labels")
    return fit_weighted_logistic([row.features for row in examples],
                                 [row.label for row in examples],
                                 [row.weight for row in examples], c)


@dataclass(frozen=True)
class QuestionSummary:
    qkey: str
    policy: str
    correctness: float
    tokens: float
    firing_rate: float

    def __post_init__(self) -> None:
        if type(self.qkey) is not str or not self.qkey:
            raise ValueError("question summary qkey must be a nonempty exact string")
        if type(self.policy) is not str or not self.policy:
            raise ValueError("question summary policy must be a nonempty exact string")
        if (isinstance(self.correctness, bool)
                or not isinstance(self.correctness, (int, float))
                or not math.isfinite(self.correctness) or not 0 <= self.correctness <= 1):
            raise ValueError("question summary correctness must be finite numeric in [0,1]")
        if (isinstance(self.tokens, bool) or not isinstance(self.tokens, (int, float))
                or not math.isfinite(self.tokens) or self.tokens < 0):
            raise ValueError("question summary tokens must be finite nonnegative numeric")
        if (isinstance(self.firing_rate, bool)
                or not isinstance(self.firing_rate, (int, float))
                or not math.isfinite(self.firing_rate) or not 0 <= self.firing_rate <= 1):
            raise ValueError("question summary firing_rate must be finite numeric in [0,1]")


@dataclass(frozen=True)
class MatchedScore:
    saving: float
    accuracy: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.saving) or not math.isfinite(self.accuracy):
            raise ValueError("matched score must be finite")


@dataclass(frozen=True)
class _EvaluatedCandidate:
    policy_class: str
    hyperparameters: tuple[Any, ...]
    fold_savings: tuple[float, float, float, float]
    pooled_saving: float
    pooled_accuracy: float
    fitted_coefficients: int


@dataclass(frozen=True)
class CandidateTrace:
    policy_class: str
    hyperparameters: tuple[Any, ...]
    fold_savings: tuple[float | None, float | None, float | None, float | None]
    pooled_saving: float | None
    pooled_accuracy: float | None
    undefined_reasons: tuple[str, ...]
    eligible: bool
    match_audits: tuple["MatchAudit", ...] = ()


@dataclass(frozen=True)
class MatchAudit:
    fold: int | None
    candidate: PopulationSummary
    endpoints: tuple[PopulationSummary, ...]
    hull: tuple[HullVertex, ...]
    match: MatchResult | None
    reason: str | None


def _select_evaluated(candidates: Sequence[_EvaluatedCandidate]) -> _EvaluatedCandidate:
    eligible = [candidate for candidate in candidates
                if len(candidate.fold_savings) == 4
                and all(math.isfinite(value) and value >= 0 for value in candidate.fold_savings)
                and math.isfinite(candidate.pooled_saving)
                and math.isfinite(candidate.pooled_accuracy)]
    if not eligible:
        raise InsufficientSupport("no eligible inner candidate")
    serialized = lambda item: json.dumps((item.policy_class, item.hyperparameters),
                                          separators=(",", ":"))
    return sorted(eligible, key=lambda item: (-item.pooled_saving, -min(item.fold_savings),
                                              -item.pooled_accuracy, item.fitted_coefficients,
                                              serialized(item)))[0]


def _question_summaries(chains: Sequence[dict[str, Any]], rule: Rule) -> list[QuestionSummary]:
    grouped: dict[str, list[Replay]] = {}
    for chain in chains:
        replay = _replay_trusted(chain, rule)
        grouped.setdefault(replay.qkey, []).append(replay)
    return [QuestionSummary(qkey, rule.name,
            sum(row.correct for row in rows) / len(rows),
            sum(row.tokens for row in rows) / len(rows),
            sum(row.endpoint == "forced" for row in rows) / len(rows))
            for qkey, rows in sorted(grouped.items())]


def _calibration_support(chains: Sequence[dict[str, Any]], kind: str) -> tuple[int, int]:
    if kind == "q":
        return _forced_support(chains)
    labels = [int(chain["correct"]) for chain in chains]
    return sum(labels), len(labels) - sum(labels)


def fit_chain_calibrator(
    chains: Sequence[dict[str, Any]], training_qkeys: Iterable[str], kind: str, c: float,
    evaluation_qkeys: Iterable[str] = (),
) -> LogisticModel:
    if kind not in ("q", "v"):
        raise ValueError("calibration kind must be q or v")
    training = set(_checked_qkeys(training_qkeys))
    assert_disjoint_qkeys(training, evaluation_qkeys)
    selected = _chains_for_qkeys(chains, training)
    q_rows, v_rows = calibration_examples(selected, training, evaluation_qkeys)
    positives, negatives = _calibration_support(selected, kind)
    return _fit_calibrator_with_support(q_rows if kind == "q" else v_rows,
                                        c, positives, negatives)


def _select_crossfit_calibration(
    chains: Sequence[dict[str, Any]], fold_map: Mapping[str, int], kind: str,
) -> tuple[float, dict[int, LogisticModel]]:
    scored: list[tuple[float, float, dict[int, LogisticModel]]] = []
    all_qkeys = set(fold_map)
    for c in LOGISTIC_CS:
        losses: list[float] = []
        models: dict[int, LogisticModel] = {}
        for fold in range(4):
            train = {q for q, value in fold_map.items() if value != fold}
            validation = all_qkeys - train
            assert_disjoint_qkeys(train, validation)
            train_chains = _chains_for_qkeys_trusted(chains, train)
            q_rows, v_rows = _calibration_examples_trusted(train_chains, train, validation)
            rows = q_rows if kind == "q" else v_rows
            positives, negatives = _calibration_support(train_chains, kind)
            model = _fit_calibrator_with_support(rows, c, positives, negatives)
            models[fold] = model
            val_chains = _chains_for_qkeys_trusted(chains, validation)
            val_q, val_v = _calibration_examples_trusted(val_chains, validation, train)
            val_rows = val_q if kind == "q" else val_v
            for qkey in sorted(validation):
                q_examples = [row for row in val_rows if row.qkey == qkey]
                if q_examples:
                    losses.append(question_log_loss(q_examples, model))
        if len(losses) != len(all_qkeys):
            raise InsufficientSupport("calibration held-out question coverage mismatch")
        scored.append((sum(losses) / len(losses), c, models))
    _, chosen_c, chosen_models = min(scored, key=lambda item: (item[0], item[1]))
    return chosen_c, chosen_models


def _candidate_rule(
    policy_class: str, hyperparameters: tuple[Any, ...], q_model: LogisticModel,
    v_model: LogisticModel, row5_model: LogisticModel | None,
) -> Rule:
    if policy_class == "row3":
        return linear_depth_rule(q_model, v_model, *hyperparameters)
    if policy_class == "row4":
        return monotone_depth_rule(q_model, v_model, hyperparameters)
    if row5_model is None:
        raise ValueError("row5 model missing")
    return low_capacity_rule(row5_model, hyperparameters[1])


@dataclass(frozen=True)
class FrozenFittedCandidate:
    policy_class: str
    hyperparameters: tuple[Any, ...]
    q_c: float
    v_c: float
    fold_savings: tuple[float, float, float, float]
    pooled_saving: float
    pooled_accuracy: float
    refit_q_model: LogisticModel
    refit_v_model: LogisticModel
    refit_row5_model: LogisticModel | None
    rule: Rule
    selection_trace: tuple[CandidateTrace, ...]


def fit_nested_candidate(
    chains: Sequence[dict[str, Any]], outer_training_qkeys: Iterable[str], seed: int,
    outer_fold: int, matched_scorer: Callable[[tuple[QuestionSummary, ...],
                                              tuple[QuestionSummary, ...]], MatchedScore],
) -> FrozenFittedCandidate:
    if not callable(matched_scorer):
        raise TypeError("matched_scorer must be callable")
    validated_chains = [_validate_replay_chain(chain) for chain in chains]
    raw_qkeys = [chain["qkey"] for chain in validated_chains]
    if any(type(qkey) is not str or not qkey for qkey in raw_qkeys):
        raise ValueError("chains must carry nonempty exact qkeys")
    unique_qkeys = tuple(sorted(set(raw_qkeys)))
    expected_outer = outer_folds(unique_qkeys, seed)
    expected_training = {q for q, fold in expected_outer.items() if fold != outer_fold}
    training = set(_checked_qkeys(outer_training_qkeys))
    if training != expected_training:
        raise AssertionError("outer_training_qkeys must exactly exclude the outer held-out fold")
    training_chains = _chains_for_qkeys_trusted(validated_chains, training)
    folds = inner_folds(training, seed, outer_fold)
    q_c, q_models = _select_crossfit_calibration(training_chains, folds, "q")
    v_c, v_models = _select_crossfit_calibration(training_chains, folds, "v")
    fold_chains = {fold: _chains_for_qkeys_trusted(training_chains,
                   {q for q, value in folds.items() if value == fold}) for fold in range(4)}
    baseline_rules = {fold: (
        Rule("row2", predicate=calibrated_global_rule(q_models[fold], v_models[fold]).predicate),
        NATURAL_END_RULE,
        *(guard_rules()[f"fixed_cap_{cap}"] for cap in (8, 16, 32, 64)),
    ) for fold in range(4)}
    baselines = {fold: [summary for rule in baseline_rules[fold]
                        for summary in _question_summaries(fold_chains[fold], rule)]
                 for fold in range(4)}
    row5_models: dict[tuple[float, int], LogisticModel] = {}
    for c in LOGISTIC_CS:
        for fold in range(4):
            train = {q for q, value in folds.items() if value != fold}
            validation = set(folds) - train
            assert_disjoint_qkeys(train, validation)
            row5_models[(c, fold)] = _fit_low_capacity_trusted(
                _chains_for_qkeys_trusted(training_chains, train), train, c)

    grammar = ([('row3', tuple(item), 8) for item in linear_candidates()]
               + [('row4', tuple(item), 10) for item in monotone_candidates()]
               + [('row5', tuple(item), 5) for item in low_capacity_candidates()])
    evaluated: list[_EvaluatedCandidate] = []
    traces: list[CandidateTrace] = []
    for policy_class, hyperparameters, coefficient_count in grammar:
        candidate_by_fold: dict[int, list[QuestionSummary]] = {}
        fold_scores: list[MatchedScore] = []
        fold_savings: list[float | None] = []
        undefined_reasons: list[str] = []
        audits: list[MatchAudit] = []
        for fold in range(4):
            row5_model = row5_models.get((hyperparameters[0], fold)) if policy_class == "row5" else None
            rule = _candidate_rule(policy_class, hyperparameters, q_models[fold], v_models[fold], row5_model)
            summaries = _question_summaries(fold_chains[fold], rule)
            candidate_by_fold[fold] = summaries
            try:
                score = matched_scorer(tuple(summaries), tuple(baselines[fold]))
            except InsufficientSupport as error:
                fold_savings.append(None)
                undefined_reasons.append(f"fold_{fold}: {error}")
                endpoint_points = tuple(summarize_population(
                    tuple(x for x in baselines[fold] if x.policy == name), name)
                    for name in BASELINE_POLICY_NAMES)
                audits.append(MatchAudit(fold, summarize_population(summaries),
                    endpoint_points, baseline_hull(endpoint_points), None, str(error)))
                continue
            if not isinstance(score, MatchedScore):
                raise TypeError("matched_scorer must return MatchedScore")
            fold_scores.append(score)
            fold_savings.append(score.saving)
            endpoint_points = tuple(summarize_population(
                tuple(x for x in baselines[fold] if x.policy == name), name)
                for name in BASELINE_POLICY_NAMES)
            hull = baseline_hull(endpoint_points)
            audits.append(MatchAudit(fold, summarize_population(summaries),
                endpoint_points, hull, match_accuracy(summarize_population(summaries), hull), None))
        if undefined_reasons:
            traces.append(CandidateTrace(policy_class, hyperparameters,
                tuple(fold_savings), None, None, tuple(undefined_reasons), False,
                tuple(audits)))
            continue
        pooled_candidate = [row for fold in range(4) for row in candidate_by_fold[fold]]
        pooled_baseline = [row for fold in range(4) for row in baselines[fold]]
        if {row.qkey for row in pooled_candidate} != training:
            raise AssertionError("inner held-out summaries do not concatenate to outer training")
        try:
            pooled = matched_scorer(tuple(pooled_candidate), tuple(pooled_baseline))
        except InsufficientSupport as error:
            endpoint_points = tuple(summarize_population(
                tuple(x for x in pooled_baseline if x.policy == name), name)
                for name in BASELINE_POLICY_NAMES)
            audits.append(MatchAudit(None, summarize_population(pooled_candidate),
                endpoint_points, baseline_hull(endpoint_points), None, str(error)))
            traces.append(CandidateTrace(policy_class, hyperparameters,
                tuple(fold_savings), None, None, (f"pooled: {error}",), False,
                tuple(audits)))
            continue
        if not isinstance(pooled, MatchedScore):
            raise TypeError("matched_scorer must return MatchedScore")
        evaluated_candidate = _EvaluatedCandidate(policy_class, hyperparameters,
                         tuple(score.saving for score in fold_scores), pooled.saving,
                         pooled.accuracy, coefficient_count)
        evaluated.append(evaluated_candidate)
        eligible = all(value >= 0 for value in evaluated_candidate.fold_savings)
        endpoint_points = tuple(summarize_population(
            tuple(x for x in pooled_baseline if x.policy == name), name)
            for name in BASELINE_POLICY_NAMES)
        pooled_hull = baseline_hull(endpoint_points)
        audits.append(MatchAudit(None, summarize_population(pooled_candidate),
            endpoint_points, pooled_hull,
            match_accuracy(summarize_population(pooled_candidate), pooled_hull), None))
        traces.append(CandidateTrace(policy_class, hyperparameters,
            tuple(fold_savings), pooled.saving, pooled.accuracy, (), eligible,
            tuple(audits)))
    selected = _select_evaluated(evaluated)
    q_rows, v_rows = _calibration_examples_trusted(training_chains, training)
    q_pos, q_neg = _calibration_support(training_chains, "q")
    v_pos, v_neg = _calibration_support(training_chains, "v")
    refit_q = _fit_calibrator_with_support(q_rows, q_c, q_pos, q_neg)
    refit_v = _fit_calibrator_with_support(v_rows, v_c, v_pos, v_neg)
    refit_row5 = (_fit_low_capacity_trusted(training_chains, training, selected.hyperparameters[0])
                  if selected.policy_class == "row5" else None)
    rule = _candidate_rule(selected.policy_class, selected.hyperparameters,
                           refit_q, refit_v, refit_row5)
    return FrozenFittedCandidate(selected.policy_class, selected.hyperparameters, q_c, v_c,
        selected.fold_savings, selected.pooled_saving, selected.pooled_accuracy,
        refit_q, refit_v, refit_row5, rule, tuple(traces))


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class PopulationSummary:
    policy: str
    accuracy: float
    tokens: float
    qkeys: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.policy) is not str or not self.policy:
            raise ValueError("policy must be a nonempty exact string")
        accuracy = _finite_number(self.accuracy, "accuracy")
        tokens = _finite_number(self.tokens, "tokens")
        if not 0 <= accuracy <= 1 or tokens < 0:
            raise ValueError("population accuracy/tokens are outside their domains")
        if (not isinstance(self.qkeys, tuple) or not self.qkeys
                or any(type(qkey) is not str or not qkey for qkey in self.qkeys)
                or tuple(sorted(set(self.qkeys))) != self.qkeys):
            raise ValueError("qkeys must be a nonempty sorted unique exact-string tuple")

    @property
    def question_count(self) -> int:
        return len(self.qkeys)


@dataclass(frozen=True)
class HullVertex:
    policy: str
    accuracy: float
    tokens: float

    def __post_init__(self) -> None:
        if type(self.policy) is not str or not self.policy:
            raise ValueError("policy must be a nonempty exact string")
        accuracy = _finite_number(self.accuracy, "accuracy")
        tokens = _finite_number(self.tokens, "tokens")
        if not 0 <= accuracy <= 1 or tokens < 0:
            raise ValueError("hull accuracy/tokens are outside their domains")


@dataclass(frozen=True)
class Mixture:
    policies: tuple[str, ...]
    weights: tuple[float, ...]
    accuracy: float
    tokens: float

    def __post_init__(self) -> None:
        if (not self.policies or len(self.policies) != len(self.weights)
                or any(type(policy) is not str or not policy for policy in self.policies)):
            raise ValueError("mixture policies/weights mismatch")
        weights = tuple(_finite_number(value, "mixture weight") for value in self.weights)
        if any(value < 0 for value in weights) or not math.isclose(sum(weights), 1.0,
                                                                    rel_tol=0, abs_tol=1e-12):
            raise ValueError("mixture weights must be nonnegative and sum to one")
        accuracy = _finite_number(self.accuracy, "mixture accuracy")
        tokens = _finite_number(self.tokens, "mixture tokens")
        if not 0 <= accuracy <= 1 or tokens < 0:
            raise ValueError("mixture outcome is outside its domain")


@dataclass(frozen=True)
class MatchResult:
    saving: float
    accuracy: float
    candidate_accuracy: float
    candidate_tokens: float
    mixture: Mixture

    def __post_init__(self) -> None:
        for name in ("saving", "accuracy", "candidate_accuracy", "candidate_tokens"):
            _finite_number(getattr(self, name), name)


@dataclass(frozen=True)
class CapFailure:
    strict_guard_dominance: bool
    stopped_over_cap: int

    def __post_init__(self) -> None:
        if type(self.strict_guard_dominance) is not bool:
            raise ValueError("strict_guard_dominance must be an exact bool")
        if type(self.stopped_over_cap) is not int or self.stopped_over_cap < 0:
            raise ValueError("stopped_over_cap must be an exact nonnegative integer")


@dataclass(frozen=True)
class BootstrapReplicate:
    index: int
    status: str
    value: float | None
    matched_accuracy: float | None = None
    mixture: Mixture | None = None
    hull: tuple[HullVertex, ...] = ()
    candidate_tokens: float | None = None
    candidate_accuracy: float | None = None
    endpoints: tuple[PopulationSummary, ...] = ()

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("bootstrap replicate index must be an exact nonnegative integer")
        if self.status not in ("DEFINED", "UNDEFINED"):
            raise ValueError("bootstrap replicate status is invalid")
        if self.status == "DEFINED":
            if self.value is None:
                raise ValueError("defined bootstrap replicate requires a value")
            _finite_number(self.value, "bootstrap replicate value")
        elif self.value is not None:
            raise ValueError("undefined bootstrap replicate value must be None")
        if self.status == "DEFINED" and (self.matched_accuracy is None
                or self.mixture is None or self.candidate_tokens is None
                or not self.hull or not self.endpoints):
            raise ValueError("defined bootstrap audit fields are required")
        if self.candidate_accuracy is None or self.candidate_tokens is None:
            raise ValueError("bootstrap candidate outcomes are always required")
        if self.status == "UNDEFINED" and any(value is not None for value in
                (self.matched_accuracy, self.mixture)):
            raise ValueError("undefined bootstrap audit fields must be null")


@dataclass(frozen=True)
class BootstrapResult:
    replicates: int
    bootstrap_seed: int
    point_saving: float | None
    lower_95: float | None
    upper_95: float | None
    undefined_count: int
    records: tuple[BootstrapReplicate, ...]
    conditional_oof: bool = True

    def __post_init__(self) -> None:
        if (type(self.replicates) is not int or self.replicates < 1
                or type(self.bootstrap_seed) is not int
                or type(self.undefined_count) is not int or self.undefined_count < 0
                or type(self.conditional_oof) is not bool
                or len(self.records) != self.replicates
                or any(not isinstance(record, BootstrapReplicate) for record in self.records)
                or tuple(record.index for record in self.records) != tuple(range(self.replicates))
                or sum(record.status == "UNDEFINED" for record in self.records) != self.undefined_count):
            raise ValueError("invalid bootstrap accounting")
        for value in (self.point_saving, self.lower_95, self.upper_95):
            if value is not None:
                _finite_number(value, "bootstrap estimate")

    @property
    def defined_savings(self) -> tuple[float, ...]:
        return tuple(record.value for record in self.records if record.status == "DEFINED")


def aggregate_questions(
    replays: Sequence[Replay], policy: str, manifest_reps: Mapping[str, int],
) -> tuple[QuestionSummary, ...]:
    if type(policy) is not str or not policy:
        raise ValueError("policy must be a nonempty exact string")
    if not isinstance(manifest_reps, Mapping) or not manifest_reps:
        raise ValueError("manifest_reps must be a nonempty mapping")
    expected: set[tuple[str, int]] = set()
    for qkey, count in manifest_reps.items():
        if type(qkey) is not str or not qkey or type(count) is not int or count < 1:
            raise ValueError("manifest rep counts must be exact positive integers")
        expected.update((qkey, rep) for rep in range(count))
    observed: dict[tuple[str, int], Replay] = {}
    for replay in replays:
        if not isinstance(replay, Replay):
            raise TypeError("replays must contain Replay records")
        if (type(replay.qkey) is not str or not replay.qkey or type(replay.rep) is not int
                or replay.rep < 0 or type(replay.correct) is not bool
                or type(replay.tokens) is not int or replay.tokens < 0
                or replay.endpoint not in ("forced", "natural")):
            raise ValueError("invalid replay identity/correctness/tokens")
        identity = (replay.qkey, replay.rep)
        if identity in observed:
            raise ValueError("duplicate replay identity")
        observed[identity] = replay
    if set(observed) != expected:
        raise ValueError("manifest rep coverage mismatch")
    grouped: dict[str, list[Replay]] = {qkey: [] for qkey in manifest_reps}
    for identity in sorted(observed):
        grouped[identity[0]].append(observed[identity])
    return tuple(QuestionSummary(qkey, policy,
        sum(int(row.correct) for row in rows) / len(rows),
        sum(row.tokens for row in rows) / len(rows),
        sum(row.endpoint == "forced" for row in rows) / len(rows))
        for qkey, rows in sorted(grouped.items()))


def summarize_population(
    summaries: Sequence[QuestionSummary], policy: str | None = None,
) -> PopulationSummary:
    if not summaries:
        raise InsufficientSupport("population has no questions")
    if any(not isinstance(row, QuestionSummary) for row in summaries):
        raise TypeError("summaries must contain QuestionSummary records")
    names = {row.policy for row in summaries}
    chosen = next(iter(names)) if policy is None and len(names) == 1 else policy
    if type(chosen) is not str or not chosen or names != {chosen}:
        raise ValueError("population policy mismatch")
    qkeys = [row.qkey for row in summaries]
    if len(set(qkeys)) != len(qkeys):
        raise ValueError("population qkeys must be unique")
    for row in summaries:
        if (type(row.qkey) is not str or not row.qkey
                or isinstance(row.correctness, bool) or not math.isfinite(row.correctness)
                or not 0 <= row.correctness <= 1
                or isinstance(row.tokens, bool) or not math.isfinite(row.tokens) or row.tokens < 0
                or isinstance(row.firing_rate, bool) or not math.isfinite(row.firing_rate)
                or not 0 <= row.firing_rate <= 1):
            raise ValueError("invalid question summary")
    count = len(summaries)
    return PopulationSummary(chosen, sum(row.correctness for row in summaries) / count,
                             sum(row.tokens for row in summaries) / count,
                             tuple(sorted(qkeys)))


def dominance_filtered(points: Sequence[PopulationSummary]) -> tuple[HullVertex, ...]:
    if not points:
        raise InsufficientSupport("baseline has no endpoints")
    if any(not isinstance(point, PopulationSummary) for point in points):
        raise TypeError("baseline endpoints must be PopulationSummary records")
    if any(point.qkeys != points[0].qkeys for point in points):
        raise ValueError("baseline endpoints require exact common question identity")
    unique: dict[float, PopulationSummary] = {}
    for point in points:
        previous = unique.get(point.accuracy)
        if previous is None or (point.tokens, point.policy) < (previous.tokens, previous.policy):
            unique[point.accuracy] = point
    survivors = [point for point in unique.values() if not any(
        other.accuracy >= point.accuracy and other.tokens <= point.tokens
        and (other.accuracy > point.accuracy or other.tokens < point.tokens)
        for other in unique.values())]
    survivors.sort(key=lambda point: (point.accuracy, point.tokens, point.policy))
    return tuple(HullVertex(point.policy, point.accuracy, point.tokens) for point in survivors)


def baseline_hull(points: Sequence[PopulationSummary]) -> tuple[HullVertex, ...]:
    filtered = dominance_filtered(points)
    survivors = [PopulationSummary(point.policy, point.accuracy, point.tokens,
                                   points[0].qkeys) for point in filtered]
    hull: list[PopulationSummary] = []
    for point in survivors:
        while len(hull) >= 2:
            left, middle = hull[-2], hull[-1]
            old_slope = (middle.tokens - left.tokens) / (middle.accuracy - left.accuracy)
            new_slope = (point.tokens - middle.tokens) / (point.accuracy - middle.accuracy)
            if new_slope > old_slope:
                break
            hull.pop()
        hull.append(point)
    return tuple(HullVertex(point.policy, point.accuracy, point.tokens) for point in hull)


def _validated_hull(hull: Sequence[HullVertex]) -> tuple[HullVertex, ...]:
    vertices = tuple(hull)
    if (not vertices or any(not isinstance(item, HullVertex) for item in vertices)
            or any(right.accuracy <= left.accuracy or right.tokens <= left.tokens
                   for left, right in zip(vertices, vertices[1:]))):
        raise ValueError("hull vertices must increase in accuracy and tokens")
    return vertices


def _mix(left: HullVertex, right: HullVertex, weight_right: float) -> Mixture:
    if weight_right == 0:
        return Mixture((left.policy,), (1.0,), left.accuracy, left.tokens)
    if weight_right == 1:
        return Mixture((right.policy,), (1.0,), right.accuracy, right.tokens)
    weight_left = 1.0 - weight_right
    return Mixture((left.policy, right.policy), (weight_left, weight_right),
                   weight_left * left.accuracy + weight_right * right.accuracy,
                   weight_left * left.tokens + weight_right * right.tokens)


def match_accuracy(candidate: PopulationSummary, hull: Sequence[HullVertex]) -> MatchResult:
    if not isinstance(candidate, PopulationSummary):
        raise TypeError("candidate must be a PopulationSummary")
    vertices = _validated_hull(hull)
    if not vertices[0].accuracy <= candidate.accuracy <= vertices[-1].accuracy:
        raise InsufficientSupport("candidate lies outside closed baseline accuracy support")
    for vertex in vertices:
        if candidate.accuracy == vertex.accuracy:
            mixture = _mix(vertex, vertex, 0)
            break
    else:
        left, right = next((a, b) for a, b in zip(vertices, vertices[1:])
                           if a.accuracy < candidate.accuracy < b.accuracy)
        mixture = _mix(left, right, (candidate.accuracy - left.accuracy)
                       / (right.accuracy - left.accuracy))
    mixture = Mixture(mixture.policies, mixture.weights, candidate.accuracy, mixture.tokens)
    return MatchResult(mixture.tokens - candidate.tokens, mixture.accuracy,
                       candidate.accuracy, candidate.tokens, mixture)


def match_tokens(candidate: PopulationSummary, hull: Sequence[HullVertex]) -> MatchResult:
    if not isinstance(candidate, PopulationSummary):
        raise TypeError("candidate must be a PopulationSummary")
    vertices = _validated_hull(hull)
    if not vertices[0].tokens <= candidate.tokens <= vertices[-1].tokens:
        raise InsufficientSupport("candidate lies outside closed baseline token support")
    for vertex in vertices:
        if candidate.tokens == vertex.tokens:
            mixture = _mix(vertex, vertex, 0)
            break
    else:
        left, right = next((a, b) for a, b in zip(vertices, vertices[1:])
                           if a.tokens < candidate.tokens < b.tokens)
        mixture = _mix(left, right, (candidate.tokens - left.tokens)
                       / (right.tokens - left.tokens))
    return MatchResult(candidate.accuracy - mixture.accuracy, mixture.accuracy,
                       candidate.accuracy, candidate.tokens, mixture)


def matched_scorer(
    candidate: tuple[QuestionSummary, ...], baseline: tuple[QuestionSummary, ...],
) -> MatchedScore:
    candidate_population = summarize_population(candidate)
    baseline_names = sorted({row.policy for row in baseline})
    required_guards = {"natural_end", "fixed_cap_8", "fixed_cap_16",
                       "fixed_cap_32", "fixed_cap_64"}
    row2_names = {name for name in baseline_names
                  if name in ("row2", "depth_calibrated_global")}
    if len(row2_names) != 1 or set(baseline_names) - row2_names != required_guards:
        raise ValueError("matched_scorer requires all six frozen endpoints")
    candidate_qkeys = {row.qkey for row in candidate}
    populations: list[PopulationSummary] = []
    for name in baseline_names:
        rows = tuple(row for row in baseline if row.policy == name)
        if {row.qkey for row in rows} != candidate_qkeys:
            raise ValueError("candidate/baseline question support mismatch")
        populations.append(summarize_population(rows, name))
    result = match_accuracy(candidate_population, baseline_hull(populations))
    return MatchedScore(result.saving, candidate_population.accuracy)


def cap_failures(
    candidate: PopulationSummary, guards: Sequence[PopulationSummary],
    replays_by_policy: Mapping[str, Sequence[Replay]],
    *, expected_identities: set[tuple[str, int]],
) -> CapFailure:
    if not isinstance(candidate, PopulationSummary):
        raise TypeError("candidate must be a PopulationSummary")
    if candidate.policy in GUARD_POLICY_NAMES:
        raise ValueError("candidate policy must not collide with registered guard names")
    if (len(guards) != len(GUARD_POLICY_NAMES)
            or {guard.policy for guard in guards if isinstance(guard, PopulationSummary)}
            != set(GUARD_POLICY_NAMES)):
        raise ValueError("cap failure requires all exact registered guards")
    if any(not isinstance(guard, PopulationSummary) for guard in guards):
        raise TypeError("guards must be PopulationSummary records")
    if (not isinstance(expected_identities, set) or not expected_identities
            or any(not isinstance(identity, tuple) or len(identity) != 2
                   or type(identity[0]) is not str or not identity[0]
                   or type(identity[1]) is not int or identity[1] < 0
                   for identity in expected_identities)):
        raise TypeError("expected_identities must be explicit exact (qkey,rep) pairs")
    if len(expected_identities) != FIT_CHAIN_COUNT:
        raise ValueError(f"cap gate requires exactly {FIT_CHAIN_COUNT} identities")
    expected_qkeys = {qkey for qkey, _ in expected_identities}
    expected = tuple(sorted(expected_qkeys))
    if len(expected) != FIT_QUESTION_COUNT:
        raise ValueError(f"cap gate requires exactly {FIT_QUESTION_COUNT} questions")
    if candidate.qkeys != expected or any(guard.qkeys != expected for guard in guards):
        raise ValueError("candidate/guard question identity mismatch")
    required_policies = {candidate.policy, *(guard.policy for guard in guards)}
    if not isinstance(replays_by_policy, Mapping) or set(replays_by_policy) != required_policies:
        raise ValueError("candidate/guard replay policy coverage mismatch")
    dominated = any(guard.accuracy > candidate.accuracy and guard.tokens < candidate.tokens
                    for guard in guards)
    over = 0
    for policy in sorted(required_policies):
        rows = replays_by_policy[policy]
        if any(not isinstance(replay, Replay) for replay in rows):
            raise TypeError("replays must contain Replay records")
        identities = [(replay.qkey, replay.rep) for replay in rows]
        if len(set(identities)) != len(identities):
            raise ValueError(f"duplicate replay identity for policy {policy}")
        if set(identities) != expected_identities:
            raise ValueError(f"replay identity coverage mismatch for policy {policy}")
        over += sum(replay.endpoint == "forced" and replay.tokens > GENERATION_CAP
                    for replay in rows)
    return CapFailure(dominated, over)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap(
    summaries: Sequence[QuestionSummary], candidate_policy: str,
    baseline_policies: Sequence[str], *, expected_qkeys: set[str],
    replicates: int = 10_000, seed: int = 314159,
) -> BootstrapResult:
    if (len(baseline_policies) != len(BASELINE_POLICY_NAMES)
            or set(baseline_policies) != set(BASELINE_POLICY_NAMES)):
        raise ValueError("paired bootstrap requires all exact six baseline endpoints")
    if not isinstance(expected_qkeys, set):
        raise TypeError("expected_qkeys must be an explicit exact-string set")
    if (not expected_qkeys or any(type(qkey) is not str or not qkey for qkey in expected_qkeys)
            or len(expected_qkeys) != FIT_QUESTION_COUNT):
        raise ValueError(f"bootstrap requires exactly {FIT_QUESTION_COUNT} expected questions")
    if (type(candidate_policy) is not str or not candidate_policy
            or type(replicates) is not int or replicates != 10_000
            or type(seed) is not int or seed != 314159):
        raise ValueError("invalid frozen bootstrap arguments")
    policies = (candidate_policy,) + tuple(baseline_policies)
    if (not baseline_policies or len(set(policies)) != len(policies)
            or any(type(policy) is not str or not policy for policy in policies)):
        raise ValueError("bootstrap policies must be unique nonempty exact strings")
    lookup: dict[tuple[str, str], QuestionSummary] = {}
    for row in summaries:
        if not isinstance(row, QuestionSummary) or row.policy not in policies:
            raise ValueError("bootstrap contains an unexpected summary/policy")
        identity = (row.qkey, row.policy)
        if identity in lookup:
            raise ValueError("duplicate bootstrap qkey/policy")
        lookup[identity] = row
    qkeys = sorted(expected_qkeys)
    if set(lookup) != {(qkey, policy) for qkey in qkeys for policy in policies}:
        raise ValueError("bootstrap requires exact paired question coverage")

    def evaluate(sample: Sequence[str]) -> tuple[MatchResult | None, tuple[HullVertex, ...],
                                                 tuple[PopulationSummary, ...]]:
        populations = []
        for policy in policies:
            accuracy = sum(lookup[(qkey, policy)].correctness for qkey in sample) / len(sample)
            tokens = sum(lookup[(qkey, policy)].tokens for qkey in sample) / len(sample)
            populations.append(PopulationSummary(policy, accuracy, tokens,
                                                  tuple(sorted(set(sample)))))
        hull = baseline_hull(populations[1:])
        try:
            matched = match_accuracy(populations[0], hull)
        except InsufficientSupport:
            matched = None
        return matched, hull, tuple(populations[1:])

    try:
        point_match = evaluate(qkeys)[0]
        point: float | None = point_match.saving if point_match else None
    except InsufficientSupport:
        point = None
    generator = random.Random(seed)
    records: list[BootstrapReplicate] = []
    undefined = 0
    for index in range(replicates):
        sample = [qkeys[generator.randrange(len(qkeys))] for _ in qkeys]
        try:
            matched, hull, endpoints = evaluate(sample)
            if matched is None:
                raise InsufficientSupport("candidate accuracy outside baseline support")
            records.append(BootstrapReplicate(index, "DEFINED", matched.saving,
                                              matched.accuracy, matched.mixture, hull,
                                              matched.candidate_tokens,
                                              matched.candidate_accuracy, endpoints))
        except InsufficientSupport:
            undefined += 1
            candidate_population = PopulationSummary(candidate_policy,
                sum(lookup[(qkey, candidate_policy)].correctness for qkey in sample)/len(sample),
                sum(lookup[(qkey, candidate_policy)].tokens for qkey in sample)/len(sample),
                tuple(sorted(set(sample))))
            records.append(BootstrapReplicate(index, "UNDEFINED", None, None, None,
                hull, candidate_population.tokens, candidate_population.accuracy, endpoints))
    defined = [record.value for record in records if record.value is not None]
    lower = _percentile(defined, .025) if defined else None
    upper = _percentile(defined, .975) if defined else None
    return BootstrapResult(replicates, seed, point, lower, upper, undefined,
                           tuple(records), True)


CELL_LAMBDAS = (0.0, 1e-5, 2e-5, 5e-5, 1e-4)
CLAIRVOYANT_REFERENCE_LABEL = "clairvoyant_trajectory_REFERENCE"
OBSERVABLE_CELL_REFERENCE_LABEL = "observable_cell_hindsight_REFERENCE"
DIAGNOSTIC_PREDICATES = (
    "global_threshold_rigidity", "depth_calibration_base_rate_shift",
    "consistent_with_data_scarcity", "registered_observable_cell_ceiling",
    "overfitting",
)
Cell = tuple[int, int, int, int]
REGISTERED_CELLS: tuple[Cell, ...] = tuple(
    (q, v, depth, remaining)
    for q in range(5) for v in range(5)
    for depth in range(4) for remaining in range(4)
)
UNFIRED_CATEGORIES = (
    "no_usable_probe", "selected_rule_does_not_fire",
    "global_blocked_selected_rule_fires",
)


def _valid_cell(cell: object) -> bool:
    return (type(cell) is tuple and len(cell) == 4
            and all(type(value) is int for value in cell)
            and all(0 <= value <= maximum
                    for value, maximum in zip(cell, (4, 4, 3, 3))))


def _unit_bin(value: float, edges: tuple[float, ...]) -> int:
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("cell coordinate must be a finite exact number")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError("cell coordinate must lie in [0,1]")
    return min(bisect.bisect_right(edges, number) - 1, len(edges) - 2)


def observable_cell(features: ProbeFeatures) -> Cell:
    if not isinstance(features, ProbeFeatures):
        raise TypeError("features must be ProbeFeatures")
    return (_unit_bin(features.q, (0., .2, .4, .6, .8, 1.)),
            _unit_bin(features.v, (0., .2, .4, .6, .8, 1.)),
            _unit_bin(features.depth, (0., .125, .25, .5, 1.)),
            _unit_bin(features.remaining, (0., .25, .5, .75, 1.)))


@dataclass(frozen=True)
class CellAction:
    cell: Cell
    action: str
    stop_utility: float
    continuation_utility: float
    occupancy: int
    reachable_occurrences: int
    repeat_visits: int

    def __post_init__(self) -> None:
        if not _valid_cell(self.cell):
            raise ValueError("invalid exact cell")
        if self.action not in ("stop", "continue") or type(self.action) is not str:
            raise ValueError("cell action must be stop or continue")
        if any(isinstance(value, bool) or type(value) not in (int, float)
               or not math.isfinite(value)
               for value in (self.stop_utility, self.continuation_utility)):
            raise ValueError("cell utilities must be finite exact numbers")
        if any(type(x) is not int or x < 0
               for x in (self.occupancy, self.reachable_occurrences, self.repeat_visits)):
            raise ValueError("cell counts must be exact nonnegative integers")
        if (self.reachable_occurrences > self.occupancy
                or self.repeat_visits > self.occupancy
                or (self.occupancy > 0 and self.repeat_visits >= self.occupancy)):
            raise ValueError("cell counts are inconsistent")

    @property
    def stop(self) -> bool:
        return self.action == "stop"


@dataclass(frozen=True)
class CellPass:
    iteration: int
    changed_cells: tuple[Cell, ...]
    actions: tuple[tuple[Cell, bool], ...]
    audit: tuple[CellAction, ...] = ()

    def __post_init__(self) -> None:
        if type(self.iteration) is not int or self.iteration < 1:
            raise ValueError("cell pass iteration must be an exact positive integer")
        if type(self.changed_cells) is not tuple or type(self.actions) is not tuple:
            raise ValueError("cell pass collections must be immutable tuples")
        if (any(not _valid_cell(cell) for cell in self.changed_cells)
                or len(set(self.changed_cells)) != len(self.changed_cells)):
            raise ValueError("changed cells must be valid and unique")
        if (any(type(item) is not tuple or len(item) != 2 or not _valid_cell(item[0])
                or type(item[1]) is not bool for item in self.actions)):
            raise ValueError("cell pass actions must be exact cell/bool tuples")
        action_cells = tuple(item[0] for item in self.actions)
        if len(set(action_cells)) != len(action_cells):
            raise ValueError("cell pass actions must be unique")
        if not set(self.changed_cells) <= set(action_cells):
            raise ValueError("changed cells must occur in pass actions")
        if action_cells != REGISTERED_CELLS:
            raise ValueError("cell pass actions require exact registered coverage")
        if (type(self.audit) is not tuple
                or len(self.audit) != len(REGISTERED_CELLS)
                or any(not isinstance(item, CellAction) for item in self.audit)
                or tuple(item.cell for item in self.audit) != REGISTERED_CELLS):
            raise ValueError("cell pass audit requires exact registered coverage")
        if any(item.stop != action for item, (_, action) in zip(self.audit, self.actions)):
            raise ValueError("cell pass audit actions disagree with the action map")
        for item in self.audit:
            if (item.reachable_occurrences > 0
                    and item.stop != (item.stop_utility > item.continuation_utility)):
                raise ValueError("cell pass action disagrees with strict utility comparison")
            if (item.reachable_occurrences == 0
                    and (item.stop_utility != 0. or item.continuation_utility != 0.)):
                raise ValueError("unreachable cell utilities must both be zero")
            if item.occupancy == 0 and item.stop:
                raise ValueError("empty cells must continue")


@dataclass(frozen=True)
class CellReference:
    label: str
    lambda_: float
    status: str
    reason: str
    iterations: int
    first_occurrence_pass: int | None
    repeat_pass: int | None
    cycle_length: int | None
    actions: tuple[CellAction, ...]
    trace: tuple[CellPass, ...]
    replays: tuple[Replay, ...]
    empty_cell_count: int
    unreachable_cell_count: int
    repeated_visit_count: int

    def __post_init__(self) -> None:
        if self.label != OBSERVABLE_CELL_REFERENCE_LABEL or not self.label.endswith("_REFERENCE"):
            raise ValueError("row 6 must carry its frozen reference label")
        if self.lambda_ not in CELL_LAMBDAS or type(self.lambda_) is not float:
            raise ValueError("invalid row-6 lambda")
        if self.status not in ("CONVERGED", "NONCONVERGENT_REFERENCE") \
                or type(self.status) is not str:
            raise ValueError("invalid cell reference status")
        allowed_reasons = {
            "CONVERGED": ("FIXED_POINT",),
            "NONCONVERGENT_REFERENCE": ("ACTION_MAP_CYCLE", "NO_REPEAT_WITHIN_100"),
        }
        if self.reason not in allowed_reasons[self.status] or type(self.reason) is not str:
            raise ValueError("cell reference status/reason mismatch")
        if type(self.iterations) is not int or not 1 <= self.iterations <= 100:
            raise ValueError("invalid cell iteration count")
        if any(type(value) is not tuple for value in (self.actions, self.trace, self.replays)):
            raise ValueError("cell reference collections must be immutable tuples")
        if (any(not isinstance(value, CellPass) for value in self.trace)
                or tuple(value.iteration for value in self.trace) != tuple(range(1, self.iterations + 1))):
            raise ValueError("cell reference trace must have a consecutive pass sequence")
        if any(not value.audit for value in self.trace):
            raise ValueError("cell reference trace requires a complete audit for every pass")
        initial = tuple((cell, False) for cell in REGISTERED_CELLS)
        action_maps = (initial,) + tuple(value.actions for value in self.trace)
        for previous, current, cell_pass in zip(action_maps, action_maps[1:], self.trace):
            changed = tuple(cell for (cell, old), (_, new) in zip(previous, current)
                            if old != new)
            if cell_pass.changed_cells != changed:
                raise ValueError("cell reference changed_cells disagree with adjacent action maps")
            changed_set = set(changed)
            if any(item.cell in changed_set and item.reachable_occurrences == 0
                   for item in cell_pass.audit):
                raise ValueError("unreachable cell action cannot change")
        static_counts = tuple((item.occupancy, item.repeat_visits)
                              for item in self.trace[0].audit)
        if any(tuple((item.occupancy, item.repeat_visits) for item in cell_pass.audit)
               != static_counts for cell_pass in self.trace[1:]):
            raise ValueError("cell occupancy and repeat counts must be static across passes")
        repeat_fields = (self.first_occurrence_pass, self.repeat_pass, self.cycle_length)
        if self.status == "CONVERGED":
            if any(value is not None for value in repeat_fields):
                raise ValueError("converged cell reference cannot carry repeat fields")
            if self.trace[-1].changed_cells or action_maps[-1] != action_maps[-2]:
                raise ValueError("converged cell reference must end with an unchanged pass")
            if len(set(action_maps[:-1])) != len(action_maps) - 1:
                raise ValueError("converged cell reference cannot contain an earlier repeat")
            if (any(not isinstance(value, CellAction) for value in self.actions)
                    or tuple(value.cell for value in self.actions) != REGISTERED_CELLS):
                raise ValueError("cell reference actions require exact registered coverage")
            if self.actions != self.trace[-1].audit:
                raise ValueError("cell reference actions disagree with the final pass")
        else:
            if self.actions or self.replays:
                raise ValueError("nonconvergent cell reference cannot select actions or replays")
            if any(not value.changed_cells for value in self.trace):
                raise ValueError("nonconvergent cell reference cannot contain a fixed pass")
            if self.reason == "ACTION_MAP_CYCLE":
                first, repeat, length = repeat_fields
                if (type(first) is not int or first < 0
                        or type(repeat) is not int or repeat < 1
                        or type(length) is not int or length < 2
                        or repeat != self.iterations or length != repeat - first
                        or first >= repeat or action_maps[-1] == action_maps[-2]):
                    raise ValueError("invalid action-map cycle fields")
                if (len(set(action_maps[:-1])) != len(action_maps) - 1
                        or action_maps[-1] != action_maps[first]):
                    raise ValueError("action-map cycle fields disagree with the trace")
            elif (any(value is not None for value in repeat_fields)
                  or self.iterations != 100 or len(set(action_maps)) != len(action_maps)):
                raise ValueError("invalid no-repeat terminal fields")
        if any(not isinstance(value, Replay) for value in self.replays):
            raise TypeError("cell reference replays must be exact Replay records")
        replay_ids = tuple((value.qkey, value.rep) for value in self.replays)
        if len(set(replay_ids)) != len(replay_ids):
            raise ValueError("cell reference requires unique replay coverage")
        counts = (self.empty_cell_count, self.unreachable_cell_count,
                  self.repeated_visit_count)
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("invalid cell reference counts")
        terminal_audit = self.trace[-1].audit
        if self.empty_cell_count != sum(value.occupancy == 0 for value in terminal_audit):
            raise ValueError("empty cell count is inconsistent")
        if self.unreachable_cell_count != sum(value.occupancy > 0
                                              and value.reachable_occurrences == 0
                                              for value in terminal_audit):
            raise ValueError("unreachable cell count is inconsistent")
        if self.repeated_visit_count != sum(value.repeat_visits for value in terminal_audit):
            raise ValueError("repeated visit count is inconsistent")

    @property
    def converged(self) -> bool:
        return self.status == "CONVERGED"


def _cell_replay(chain: dict[str, Any], actions: Mapping[Cell, bool], start: int = 0) -> Replay:
    for index in range(start, len(chain["probes"])):
        cell = observable_cell(probe_features(chain["probes"][index], chain["scores"][index]))
        if actions.get(cell, False):
            probe = chain["probes"][index]
            answer = probe["forced"]
            return Replay(chain["qkey"], chain["rep"], "forced", index, answer,
                          answer == chain["gold"], probe["prefix_len"] + probe["forced_tokens"])
    return Replay(chain["qkey"], chain["rep"], "natural", None, chain["answer"],
                  chain["correct"], chain["thinking_tokens"])


def _cell_suffix_replays(chain: dict[str, Any], actions: Mapping[Cell, bool]) -> tuple[Replay, ...]:
    natural = Replay(chain["qkey"], chain["rep"], "natural", None, chain["answer"],
                     chain["correct"], chain["thinking_tokens"])
    suffix = [natural] * (len(chain["probes"]) + 1)
    for index in range(len(chain["probes"]) - 1, -1, -1):
        probe = chain["probes"][index]
        cell = observable_cell(probe_features(probe, chain["scores"][index]))
        if actions.get(cell, False):
            answer = probe["forced"]
            suffix[index] = Replay(chain["qkey"], chain["rep"], "forced", index,
                                   answer, answer == chain["gold"],
                                   probe["prefix_len"] + probe["forced_tokens"])
        else:
            suffix[index] = suffix[index + 1]
    return tuple(suffix)


def observable_cell_reference(
    chains: Sequence[dict[str, Any]], lambda_: float, *, max_passes: int = 100,
) -> CellReference:
    if lambda_ not in CELL_LAMBDAS or type(lambda_) is not float:
        raise ValueError("lambda must be one of the exact row-6 values")
    if type(max_passes) is not int or isinstance(max_passes, bool) or not 1 <= max_passes <= 100:
        raise ValueError("max_passes must be an exact integer in [1,100]")
    trusted = tuple(_validate_replay_chain(chain) for chain in chains)
    identities = [(c["qkey"], c["rep"]) for c in trusted]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate chain identity")
    occurrences: dict[Cell, list[tuple[dict[str, Any], int]]] = {}
    for chain in trusted:
        for index, (probe, score) in enumerate(zip(chain["probes"], chain["scores"])):
            occurrences.setdefault(observable_cell(probe_features(probe, score)), []).append((chain, index))
    all_cells = REGISTERED_CELLS
    actions: dict[Cell, bool] = {cell: False for cell in all_cells}
    initial_action_map = tuple((cell, False) for cell in all_cells)
    first_seen = {initial_action_map: 0}
    cycle: tuple[int, int, int] | None = None
    trace: list[CellPass] = []
    reachable_counts: dict[Cell, int] = {cell: 0 for cell in all_cells}
    cell_means: dict[Cell, tuple[float, float]] = {}
    for iteration in range(1, max_passes + 1):
        utilities: dict[Cell, dict[str, dict[int, list[tuple[float, float]]]]] = {}
        for chain in trusted:
            suffix = _cell_suffix_replays(chain, actions)
            reached = True
            for index, (probe, score) in enumerate(zip(chain["probes"], chain["scores"])):
                if not reached:
                    break
                cell = observable_cell(probe_features(probe, score))
                stop = float(probe["forced"] == chain["gold"]) - lambda_ * (probe["prefix_len"] + probe["forced_tokens"])
                downstream = suffix[index + 1]
                cont = float(downstream.correct) - lambda_ * downstream.tokens
                utilities.setdefault(cell, {}).setdefault(chain["qkey"], {}).setdefault(chain["rep"], []).append((stop, cont))
                if actions[cell]:
                    reached = False
        new_actions = dict(actions)
        reachable_counts = {cell: 0 for cell in all_cells}
        cell_means = {}
        for cell, by_q in utilities.items():
            q_values = []
            for by_rep in by_q.values():
                rep_values = []
                for visits in by_rep.values():
                    rep_values.append((sum(x[0] for x in visits) / len(visits),
                                       sum(x[1] for x in visits) / len(visits)))
                    reachable_counts[cell] += len(visits)
                q_values.append((sum(x[0] for x in rep_values) / len(rep_values),
                                 sum(x[1] for x in rep_values) / len(rep_values)))
            stop_mean = sum(x[0] for x in q_values) / len(q_values)
            continue_mean = sum(x[1] for x in q_values) / len(q_values)
            cell_means[cell] = (stop_mean, continue_mean)
            new_actions[cell] = stop_mean > continue_mean
        changed = tuple(cell for cell in all_cells if new_actions[cell] != actions[cell])
        actions = new_actions
        pass_audit = []
        for cell in all_cells:
            occupied = len(occurrences.get(cell, ()))
            distinct = len({(c["qkey"], c["rep"])
                            for c, _ in occurrences.get(cell, ())})
            stop_mean, continue_mean = cell_means.get(cell, (0., 0.))
            pass_audit.append(CellAction(cell, "stop" if actions[cell] else "continue",
                stop_mean, continue_mean, occupied, reachable_counts[cell],
                occupied - distinct))
        trace.append(CellPass(iteration, changed,
            tuple((cell, actions[cell]) for cell in all_cells), tuple(pass_audit)))
        if not changed:
            break
        action_map = trace[-1].actions
        if action_map in first_seen:
            first_occurrence_pass = first_seen[action_map]
            repeat_pass = iteration
            cycle = (first_occurrence_pass, repeat_pass,
                     repeat_pass - first_occurrence_pass)
            break
        first_seen[action_map] = iteration
    else:
        if max_passes < 100:
            raise RuntimeError("observable-cell reference did not converge after 100 passes")
    terminal_audit = trace[-1].audit
    empty = sum(record.occupancy == 0 for record in terminal_audit)
    unreachable = sum(record.occupancy > 0 and record.reachable_occurrences == 0
                      for record in terminal_audit)
    repeated = sum(record.repeat_visits for record in terminal_audit)
    if not trace[-1].changed_cells:
        replays = tuple(_cell_replay(chain, actions) for chain in trusted)
        return CellReference(OBSERVABLE_CELL_REFERENCE_LABEL, lambda_, "CONVERGED",
                             "FIXED_POINT", len(trace), None, None, None,
                             terminal_audit, tuple(trace), replays, empty,
                             unreachable, repeated)
    if cycle is not None:
        first_occurrence_pass, repeat_pass, cycle_length = cycle
        return CellReference(OBSERVABLE_CELL_REFERENCE_LABEL, lambda_,
                             "NONCONVERGENT_REFERENCE", "ACTION_MAP_CYCLE",
                             len(trace), first_occurrence_pass, repeat_pass,
                             cycle_length, (), tuple(trace), (), empty,
                             unreachable, repeated)
    return CellReference(OBSERVABLE_CELL_REFERENCE_LABEL, lambda_,
                         "NONCONVERGENT_REFERENCE", "NO_REPEAT_WITHIN_100",
                         len(trace), None, None, None, (), tuple(trace), (), empty,
                         unreachable, repeated)


def observable_cell_frontier(chains: Sequence[dict[str, Any]]) -> tuple[CellReference, ...]:
    results = tuple(observable_cell_reference(chains, value) for value in CELL_LAMBDAS)
    if not all(result.converged for result in results):
        return ()
    summaries = [(result, summarize_population(aggregate_questions(
        result.replays, result.label, Counter(r.qkey for r in result.replays))))
                 for result in results]
    return tuple(result for result, point in summaries if not any(
        other.accuracy >= point.accuracy and other.tokens <= point.tokens
        and (other.accuracy > point.accuracy or other.tokens < point.tokens)
        for _, other in summaries))


def clairvoyant_reference(chain: dict[str, Any]) -> Replay:
    chain = _validate_replay_chain(chain)
    endpoints = [_replay_trusted(chain, NATURAL_END_RULE)]
    for index, probe in enumerate(chain["probes"]):
        answer = probe["forced"]
        endpoints.append(Replay(chain["qkey"], chain["rep"], "forced", index, answer,
                                answer == chain["gold"],
                                probe["prefix_len"] + probe["forced_tokens"]))
    correct = [item for item in endpoints if item.correct]
    return min(correct, key=lambda item: (item.tokens, item.endpoint, item.probe_index or -1)) \
        if correct else endpoints[0]


@dataclass(frozen=True)
class UnfiredCategory:
    seed: int
    candidate: str
    category: str
    count: int
    correct_count: int
    token_total: int

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in FITTING_SEEDS:
            raise ValueError("unfired seed must be a frozen fitting seed")
        if type(self.candidate) is not str or not self.candidate:
            raise ValueError("unfired candidate must be a nonempty exact string")
        if type(self.category) is not str or self.category not in UNFIRED_CATEGORIES:
            raise ValueError("unfired category is not registered")
        if any(type(value) is not int or value < 0
               for value in (self.count, self.correct_count, self.token_total)):
            raise ValueError("unfired counts and token total must be exact nonnegative integers")
        if self.correct_count > self.count:
            raise ValueError("unfired correct count cannot exceed count")


def unfired_decomposition(
    chains: Sequence[dict[str, Any]],
    row1_by_candidate: Mapping[tuple[int, str], Sequence[Replay]],
    candidate_replays: Mapping[tuple[int, str], Sequence[Replay]],
) -> tuple[UnfiredCategory, ...]:
    if set(row1_by_candidate) != set(candidate_replays):
        raise ValueError("decomposition key coverage mismatch")
    trusted = tuple(_validate_replay_chain(chain) for chain in chains)
    identities = [(c["qkey"], c["rep"]) for c in trusted]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate chain identity")
    chain_map = {(c["qkey"], c["rep"]): c for c in trusted}
    output = []
    categories = UNFIRED_CATEGORIES
    for key in sorted(row1_by_candidate):
        seed, candidate = key
        if type(seed) is not int or type(candidate) is not str or not candidate:
            raise ValueError("invalid seed/candidate key")
        if (any(not isinstance(r, Replay) for r in row1_by_candidate[key])
                or any(not isinstance(r, Replay) for r in candidate_replays[key])):
            raise TypeError("decomposition values must be Replay records")
        base = {(r.qkey, r.rep): r for r in row1_by_candidate[key]}
        selected = {(r.qkey, r.rep): r for r in candidate_replays[key]}
        if (len(base) != len(row1_by_candidate[key])
                or len(selected) != len(candidate_replays[key])):
            raise ValueError("duplicate decomposition replay identity")
        if set(base) != set(selected) or set(base) != set(chain_map):
            raise ValueError("decomposition identity coverage mismatch")
        buckets = {name: [] for name in categories}
        for identity, row1 in base.items():
            if row1.endpoint != "natural":
                continue
            replay, chain = selected[identity], chain_map[identity]
            category = (categories[0] if not chain["probes"] else
                        categories[1] if replay.endpoint == "natural" else categories[2])
            buckets[category].append(replay)
        output.extend(UnfiredCategory(seed, candidate, name, len(rows),
                                      sum(r.correct for r in rows), sum(r.tokens for r in rows))
                      for name, rows in buckets.items())
    return tuple(output)


@dataclass(frozen=True)
class DiagnosticInputs:
    gate_a0_passes: bool
    modal_class: str
    support_ok: bool
    row2_recovery_fraction: float
    largest_positive_saving: float
    row6_improvement: float
    row6_accuracy_improvement: float
    row7_accuracy_improvement: float
    mean_inner_saving: float
    pooled_outer_saving: float
    row2_mean_tokens: float
    nested_interval_lower: float
    nested_interval_upper: float
    outer_seed_savings: tuple[float, float]

    def __post_init__(self) -> None:
        if type(self.gate_a0_passes) is not bool or type(self.support_ok) is not bool:
            raise ValueError("diagnostic booleans must be exact")
        if type(self.modal_class) is not str or not self.modal_class:
            raise ValueError("modal class must be nonempty")
        if type(self.outer_seed_savings) is not tuple or len(self.outer_seed_savings) != 2:
            raise ValueError("outer_seed_savings must be an exact tuple of length 2")
        for value in (*self.__dict__.values(), *self.outer_seed_savings):
            if isinstance(value, bool) or isinstance(value, str):
                continue
            if isinstance(value, tuple):
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("diagnostic values must be finite exact numbers")
        if self.nested_interval_lower > self.nested_interval_upper:
            raise ValueError("diagnostic interval bounds must be ordered")


@dataclass(frozen=True)
class DiagnosticPredicate:
    name: str
    value: bool

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in DIAGNOSTIC_PREDICATES:
            raise ValueError("diagnostic predicate name is not frozen")
        if type(self.value) is not bool:
            raise ValueError("diagnostic predicate value must be an exact bool")


@dataclass(frozen=True)
class DiagnosticTrace:
    predicates: tuple[DiagnosticPredicate, ...]
    state: str

    def __post_init__(self) -> None:
        if (type(self.predicates) is not tuple
                or any(not isinstance(item, DiagnosticPredicate) for item in self.predicates)
                or tuple(item.name for item in self.predicates) != DIAGNOSTIC_PREDICATES):
            raise ValueError("diagnostic predicates must be unique and frozen ordered")
        allowed = {*DIAGNOSTIC_PREDICATES, "AMBIGUOUS", "INSUFFICIENT_SUPPORT"}
        if type(self.state) is not str or self.state not in allowed:
            raise ValueError("diagnostic final state is not frozen")
        if self.state != "INSUFFICIENT_SUPPORT":
            expected = next((item.name for item in self.predicates if item.value), "AMBIGUOUS")
            if self.state != expected:
                raise ValueError("diagnostic state must match the first true predicate")


def diagnose(x: DiagnosticInputs) -> DiagnosticTrace:
    if not isinstance(x, DiagnosticInputs):
        raise TypeError("diagnostic inputs required")
    p1 = x.gate_a0_passes and x.modal_class in ("row3", "row4")
    p2 = (not x.gate_a0_passes and x.row2_recovery_fraction >= .75
          and x.largest_positive_saving > 0
          and x.nested_interval_lower <= 0 <= x.nested_interval_upper)
    p3 = (not x.gate_a0_passes and x.row6_improvement >= .10 * x.row2_mean_tokens
          and x.row6_accuracy_improvement >= 0 and x.mean_inner_saving > 0
          and x.pooled_outer_saving <= 0)
    p4 = (not x.gate_a0_passes and x.row6_improvement < .05 * x.row2_mean_tokens
          and x.row6_accuracy_improvement >= 0
          and x.row7_accuracy_improvement >= .05)
    p5 = (x.mean_inner_saving > 0 and all(value <= 0 for value in x.outer_seed_savings)
          and not p2 and not p3 and not p4)
    predicates = tuple(DiagnosticPredicate(name, value) for name, value in
                       zip(DIAGNOSTIC_PREDICATES, (p1, p2, p3, p4, p5)))
    if not x.support_ok:
        state = "INSUFFICIENT_SUPPORT"
    else:
        state = next((item.name for item in predicates if item.value), "AMBIGUOUS")
    return DiagnosticTrace(predicates, state)
