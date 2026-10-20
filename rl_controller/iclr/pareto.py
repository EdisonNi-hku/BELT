from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import random
import re
import statistics
import traceback
import uuid
import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rl_controller.iclr import ladder as ld
from rl_controller.iclr import learning_curve as lc
from rl_controller.iclr import token_accounting as ta

QUESTION_TABLE_SCHEMA = "iclr-ladder-question-tables/v1"
POLICY_REPLAY_SCHEMA = "iclr-policy-replay/v1"
EXCHANGE_SCHEMA = "iclr-acc-token-frontier/v1"
SUMMARY_SCHEMA = "iclr-cross-method-pareto/v1"

COST_AXES = ("thinking", "total")
EVALUATION_UNITS = ("chain", "question")
NORMAL_FORMS = ("chain", "question")
NATURAL = ld.NATURAL
ALWAYS_FIRE = ld.ALWAYS_FIRE
AGREEMENT = ld.AGREEMENT
ANCHORS = (NATURAL, ALWAYS_FIRE)
ANCHOR_COMPONENTS = ("n_chains", "correct", "thinking_tokens", "generated_tokens",
                     "overhead_tokens")
SELECT_LEVELS = ld.SELECT_LEVELS
DEFAULT_BOOTSTRAP = 500

POLICY_REPLAY_FIELDS = frozenset({
    "schema_version", "run_id", "method_family", "method_name", "config_id", "knob", "seed",
    "dataset", "split", "model_id", "model_revision", "tokenizer_revision",
    "prompt_contract_id", "grader_id", "selection_split", "selection_rule", "frozen_config",
    "source_branch", "source_commit", "source_artifact", "source_sha256",
    "chain_bank_sha256", "manifest_sha256", "pool",
    "qkey", "n_chains", "e_correct", "e_thinking_tokens", "e_answer_tokens",
    "e_generated_tokens", "aux_generated_tokens", "expectation", "missing_reason",
})
EXPECTATIONS = ("survival_product", "bernoulli_seeded")
COMPARABILITY_FIELDS = ("model_id", "model_revision", "tokenizer_revision", "dataset",
                        "prompt_contract_id", "grader_id",
                        "chain_bank_sha256", "manifest_sha256", "pool")
REQUIRED_COMPARABILITY = ("model_id", "dataset", "grader_id")
CHAIN_IDENTITY_FIELDS = ("chain_bank_sha256", "manifest_sha256", "pool")
IDENTITY_REQUIRED_KINDS = ("policy_replay",)

POOL_ALIASES = {
    "iclr-p1-math-union12": "math_union",
    "math_union_select": "math_union",
    "math_union_fit": "math_union",
    "iclr-p1-math-ood": "math_ood",
    "math_ood_report": "math_ood",
    "iclr-p1-code": "code_pool",
    "code_pool_select": "code_pool",
    "code_pool_fit": "code_pool",
    "code_ood_pooled": "code_ood",
    "math_t1_report": "math_t1",
    "math_t2_report": "math_t2",
    "ood2_math_report": "ood2_math",
    "ood2_code_report": "ood2_code",
}
ALIASED_FIELDS = ("dataset", "pool")


def canonical_pool(value):
    return POOL_ALIASES.get(value, value) if isinstance(value, str) else value


def canonicalize_identity(meta: dict) -> dict:
    for field_ in ALIASED_FIELDS:
        if field_ in meta:
            meta[field_] = canonical_pool(meta[field_])
    return meta


INVENTORY_POOL_BANKS = {
    "select": "math_union",
    "math_ood_report": "math_ood",
    "code_ood_pooled": "code_ood",
    "gpqa_report": "gpqa",
    "math_t1_report": "math_t1",
    "math_t2_report": "math_t2",
    "code_pool_select": "code_pool",
    "ood2_math_report": "ood2_math",
    "ood2_code_report": "ood2_code",
}
CANONICAL_POOLS = frozenset(POOL_ALIASES.values()) | frozenset(INVENTORY_POOL_BANKS.values())


def is_canonical_pool(value) -> bool:
    return isinstance(value, str) and value in CANONICAL_POOLS


def resolve_pool(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MergeError("pool must be a non-empty str")
    value = value.strip()
    resolved = INVENTORY_POOL_BANKS.get(value, canonical_pool(value))
    if not is_canonical_pool(resolved):
        raise MergeError(
            f"unknown pool id {value!r}: expected an assembly-inventory pool "
            f"({sorted(INVENTORY_POOL_BANKS)}), a declared alias "
            f"({sorted(POOL_ALIASES)}) or a canonical bank id "
            f"({sorted(CANONICAL_POOLS)}). A new bank is added to "
            "pareto.INVENTORY_POOL_BANKS / POOL_ALIASES, never guessed at export time.")
    return resolved
SEED_ANY = -1

_TOKEN_EPS = 1e-6


_REQUIRED = "<required>"


class MergeError(ValueError):
    pass


class LegacyStampMissing(MergeError):
    pass


class NoHonestMember(MergeError):
    pass


class NoAnchorsForUnit(MergeError):
    pass


SEED_SEMANTICS = ("training", "replay")

_CAP_SPELLING = re.compile(r"^cap[ _-]?(?P<budget>[0-9]+)$")
_VERSION_SUFFIX = re.compile(r"^(?P<base>.+?)-v[0-9]+$")

FREE_RULES = frozenset({NATURAL, ALWAYS_FIRE, AGREEMENT, "consecutive_agreement"})

_COALESCE_TOL = 1e-6


def is_free_rule(method) -> bool:
    if not isinstance(method, str):
        return False
    return method in FREE_RULES or _CAP_SPELLING.match(method) is not None


def canonical_member(name, *, method=None):
    if not isinstance(name, str) or not name:
        return name
    if method is not None and not is_free_rule(method):
        return name
    out = name
    version = _VERSION_SUFFIX.match(out)
    if version:
        out = version.group("base")
    cap = _CAP_SPELLING.match(out)
    if cap:
        out = f"cap{int(cap.group('budget'))}"
    return out


def seed_semantics_for(method_name, declared=None) -> str:
    if declared is not None:
        if declared not in SEED_SEMANTICS:
            raise MergeError(f"seed_semantics must be one of {SEED_SEMANTICS}, "
                             f"got {declared!r}")
        return declared
    return "replay" if is_free_rule(method_name) else "training"


@dataclass(frozen=True)
class SeriesKey:
    method: str
    member: str
    seed: int
    split: str

    def __str__(self) -> str:
        return f"{self.method}/{self.member}/s{self.seed}/{self.split}"


_EVIDENCE_TOKEN = object()


class _TableEvidence:

    __slots__ = ("origin", "path", "sha256", "record")

    def __init__(self, token, *, origin: str, path: str, sha256: str, record: Mapping):
        if token is not _EVIDENCE_TOKEN:
            raise MergeError(
                "verification evidence is minted by `load_verified_question_tables` alone: "
                "it is the record of dumps that were opened and re-hashed, not a flag "
                "(contract v1.12 §2.10; finding H23).")
        self.origin, self.path, self.sha256 = origin, path, sha256
        self.record = dict(record)

    def to_dict(self) -> dict:
        return {"table": self.path, "table_sha256": self.sha256, **self.record}


@dataclass
class Bundle:

    pool: str = ""
    stats: dict[SeriesKey, dict[str, tuple[float, float, float, float, float]]] = \
        field(default_factory=dict)
    meta: dict[SeriesKey, dict] = field(default_factory=dict)
    unavailable: dict[str, dict] = field(default_factory=dict)
    notes: list[dict] = field(default_factory=list)
    normal_form: str = _REQUIRED


    def __post_init__(self):
        if self.normal_form == _REQUIRED:
            raise MergeError(
                "Bundle(normal_form=...) is REQUIRED: say whether these clusters are "
                f"{NORMAL_FORMS[0]!r} (chain-summed, the archived basis) or "
                f"{NORMAL_FORMS[1]!r} (one row per question, the paper's basis). There is no "
                "default because a default cannot tell 'the caller means question' from 'the "
                "caller did not think about it', and the wrong label silently rewrites every "
                "number downstream (contract v1.7 section 5.10).")
        if self.normal_form not in NORMAL_FORMS:
            raise MergeError(f"normal_form must be one of {NORMAL_FORMS}, "
                             f"got {self.normal_form!r}")
        self._origins: dict[SeriesKey, set] = {}
        self._origin_paths: dict[str, str] = {}
        self._evidence: dict[str, _TableEvidence] = {}
        self._selection_cache: dict | None = {}

    def add(self, key: SeriesKey, per_q: Mapping[str, Sequence[float]], *,
            kind: str, meta: dict | None = None) -> None:
        clean = self._clean(key, per_q)
        self.invalidate_selection_cache()
        if key in self.stats:
            self._coalesce(key, clean, kind=kind, meta=meta)
            return
        self.stats[key] = clean
        self.meta[key] = {"kind": kind, **(meta or {})}

    def invalidate_selection_cache(self) -> None:
        if self._selection_cache is not None:
            self._selection_cache.clear()

    def _coalesce(self, key: SeriesKey, clean: dict, *, kind: str,
                  meta: dict | None) -> None:
        existing = self.stats[key]
        if set(existing) != set(clean):
            raise MergeError(f"conflicting duplicate series {key}: question sets differ "
                             f"({len(existing)} vs {len(clean)} questions)")
        drifted = []
        for qkey in sorted(existing):
            if existing[qkey] == clean[qkey]:
                continue
            gap = max(abs(a - b) for a, b in zip(existing[qkey], clean[qkey]))
            if not (gap <= _COALESCE_TOL):
                raise MergeError(
                    f"conflicting duplicate series {key}: question {qkey!r} is "
                    f"{existing[qkey]} in one producer and {clean[qkey]} in another "
                    f"(worst component differs by {gap:.6g} > {_COALESCE_TOL:g}). Two "
                    "producers disagree about a shared series -- they did not read the same "
                    "chains, and comparing them would be meaningless.")
            drifted.append((qkey, gap))
        if drifted:
            worst_q, worst_gap = max(drifted, key=lambda item: item[1])
            self.notes.append({
                "note": "coalesced_within_tolerance", "series": str(key),
                "n_questions_differing": len(drifted), "worst_qkey": worst_q,
                "worst_abs_gap": worst_gap, "tolerance": _COALESCE_TOL})
        current = self.meta[key]
        if current.get("kind", kind) != kind:
            raise MergeError(f"duplicate series {key} declared as both "
                             f"{current.get('kind')!r} and {kind!r}")
        for field_, value in (meta or {}).items():
            if field_ in current and current[field_] != value:
                raise MergeError(f"duplicate series {key}: metadata {field_!r} is "
                                 f"{current[field_]!r} in one producer and {value!r} in another")
            current[field_] = value

    def _clean(self, key: SeriesKey, per_q: Mapping[str, Sequence[float]]) -> dict:
        clean: dict[str, tuple[float, float, float, float, float]] = {}
        for qkey, value in per_q.items():
            values = tuple(float(v) for v in value)
            if len(values) != 5:
                raise MergeError(f"{key} / {qkey}: need (n, correct, thinking, generated, aux), "
                                 f"got {len(values)} value(s)")
            n, correct, thinking, generated, aux = values
            if not math.isfinite(n) or n <= 0:
                raise MergeError(f"{key} / {qkey}: n must be finite and > 0, got {n}")
            if not math.isfinite(correct):
                raise MergeError(f"{key} / {qkey}: correct must be finite")
            if not 0 <= correct <= n + _TOKEN_EPS:
                raise MergeError(f"{key} / {qkey}: correct {correct} outside [0, {n}]")
            for name, value_ in (("thinking", thinking), ("generated", generated), ("aux", aux)):
                if not math.isfinite(value_) or value_ < 0:
                    raise MergeError(f"{key} / {qkey}: {name} must be finite and >= 0")
            if generated + _TOKEN_EPS < thinking:
                raise MergeError(f"{key} / {qkey}: generated {generated} < thinking {thinking} -- "
                                 "generated = thinking + answer, so it can never be smaller")
            clean[qkey] = values
        if not clean:
            raise MergeError(f"{key}: no questions")
        return clean


    def keys_for(self, *, split: str | None = None, method: str | None = None,
                 seed: int | None = None) -> list[SeriesKey]:
        return sorted(
            (k for k in self.stats
             if (split is None or k.split == split)
             and (method is None or k.method == method)
             and (seed is None or k.seed == seed)),
            key=lambda k: (k.method, k.member, k.seed, k.split))

    def methods(self, split: str) -> list[str]:
        return sorted({k.method for k in self.stats if k.split == split})

    def seeds(self, method: str, split: str) -> list[int]:
        return sorted({k.seed for k in self.stats if k.method == method and k.split == split})

    def resolve_seed(self, method: str, split: str, seed: int) -> int:
        available = self.seeds(method, split)
        if not available:
            raise MergeError(f"no series for method {method!r} on split {split!r}")
        if seed in available:
            return seed
        if SEED_ANY in available:
            return SEED_ANY
        raise MergeError(f"{method!r} has no series for seed {seed} on {split!r} "
                         f"(available: {available}); a seed-independent method must use "
                         f"seed {SEED_ANY}")

    def qkeys(self, split: str) -> tuple[str, ...]:
        sets = {}
        for key in self.keys_for(split=split):
            sets[key] = tuple(sorted(self.stats[key]))
        if not sets:
            raise MergeError(f"no series on split {split!r}")
        distinct = set(sets.values())
        if len(distinct) != 1:
            sizes = {str(k): len(v) for k, v in sets.items()}
            raise MergeError(f"split {split!r}: series disagree on the question set "
                             f"({sizes}) -- a cross-method comparison needs one question set")
        return next(iter(distinct))

    def comparability_report(self, *, require: Sequence[str] = REQUIRED_COMPARABILITY,
                             identity_required_kinds: Sequence[str] = IDENTITY_REQUIRED_KINDS
                             ) -> dict:
        def canonical(value):
            return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value

        seen: dict[tuple[str, str], set] = defaultdict(set)
        undeclared: dict[str, list[str]] = defaultdict(list)
        identity_undeclared: dict[str, list[str]] = defaultdict(list)
        for key, meta in self.meta.items():
            for field_ in COMPARABILITY_FIELDS:
                if field_ in meta and meta[field_] is not None:
                    seen[(key.split, field_)].add(canonical(meta[field_]))
            for field_ in require:
                if meta.get(field_) is None:
                    undeclared[field_].append(str(key))
            mandatory = meta.get("kind") in identity_required_kinds
            for field_ in CHAIN_IDENTITY_FIELDS:
                if meta.get(field_) is None:
                    (undeclared if mandatory else identity_undeclared)[field_].append(str(key))
        per_split: dict[str, dict[str, object]] = defaultdict(dict)
        for (split, f), values in seen.items():
            if len(values) == 1:
                per_split[f][split] = next(iter(values))
        split_mismatched = {f for (_s, f), values in seen.items() if len(values) > 1}
        agreed: dict[str, object] = {}
        agreed_by_split: dict[str, dict[str, object]] = {}
        for f, values_by_split in sorted(per_split.items()):
            distinct = set(values_by_split.values())
            if len(distinct) == 1 and f not in split_mismatched:
                agreed[f] = next(iter(distinct))
            else:
                agreed_by_split[f] = dict(sorted(values_by_split.items()))
        return {"agreed": agreed, "agreed_by_split": agreed_by_split,
                "declared": {f"{split}/{f}": sorted(map(str, v))
                             for (split, f), v in sorted(seen.items())},
                "mismatch": {f"{split}/{f}": sorted(map(str, v))
                             for (split, f), v in sorted(seen.items()) if len(v) > 1},
                "undeclared": {f: sorted(v) for f, v in sorted(undeclared.items())},
                "identity_undeclared": {f: sorted(set(v))
                                        for f, v in sorted(identity_undeclared.items())}}

    def check_comparability(self, *, require: Sequence[str] = REQUIRED_COMPARABILITY,
                            identity_required_kinds: Sequence[str] = IDENTITY_REQUIRED_KINDS
                            ) -> dict:
        report = self.comparability_report(require=require,
                                           identity_required_kinds=identity_required_kinds)
        if report["mismatch"]:
            raise MergeError("merged series disagree on comparability fields: "
                             f"{report['mismatch']}")
        if report["undeclared"]:
            summary = {f: (len(v), v[:3]) for f, v in sorted(report["undeclared"].items())}
            raise MergeError(
                f"series do not DECLARE required comparability field(s) {summary}. Undeclared "
                "is not agreed: without them a bundle can silently compare different models, "
                "graders or chain banks. `comparability_report()` describes a bundle without "
                "asserting it is comparable; nothing may compute a statistic from one that "
                "fails this gate.")
        return {"agreed": report["agreed"], "agreed_by_split": report["agreed_by_split"],
                "undeclared": {f: len(v) for f, v in sorted(report["undeclared"].items())},
                "identity_undeclared": report["identity_undeclared"]}

    def check_anchor_agreement(self, *, tolerance: float = _COALESCE_TOL) -> dict:
        report: dict[str, dict] = {}
        for split in sorted({k.split for k in self.stats}):
            try:
                sample = self.qkeys(split)
            except MergeError:
                continue
            for anchor in ANCHORS:
                keys = self.keys_for(split=split, method=anchor)
                if not keys:
                    continue
                if len(keys) > 1:
                    self._compare_anchor_rows(split, anchor, keys, sample, tolerance)
                report.setdefault(split, {})[anchor] = len(keys)
        return report

    def _compare_anchor_rows(self, split: str, anchor: str, keys: Sequence[SeriesKey],
                             sample: Sequence[str], tolerance: float) -> None:
        first, rows = keys[0], self.stats[keys[0]]
        for key in keys[1:]:
            other = self.stats[key]
            for qkey in sample:
                a, b = rows.get(qkey), other.get(qkey)
                if a is None or b is None:
                    continue
                for name, x, y in zip(ANCHOR_COMPONENTS, a, b):
                    if abs(x - y) <= tolerance * max(1.0, abs(y)):
                        continue
                    raise MergeError(
                        f"split {split!r}: the shared {anchor!r} anchor DISAGREES across "
                        f"series on question {qkey!r}: {name} is {x!r} in {first} and {y!r} "
                        f"in {key}. `natural` / `always_fire` are score-free replays of a "
                        "fixed chain bank, so every seed and every producer must read the "
                        "SAME ROW for the same question; a difference means these series are "
                        "not on the same chains (or one producer charged the shared anchor "
                        "its own decision overhead, contract §5.2). Compared per question "
                        "and BEFORE any mean, because swapping two questions' anchor rows "
                        "between seeds leaves every mean identical (finding C21). Refusing "
                        "the pool rather than hanging two hulls on two different anchors "
                        "(findings C16 / C21).")


    def _record_origin(self, key: SeriesKey, origin: str) -> None:
        self._origins.setdefault(key, set()).add(origin)

    def _attach_evidence(self, evidence) -> None:
        if type(evidence) is not _TableEvidence:
            raise MergeError("only `_TableEvidence` minted by "
                             "`load_verified_question_tables` may be attached")
        self._evidence[evidence.origin] = evidence

    def carry_evidence(self, other: "Bundle", mapping: Mapping | None = None) -> None:
        self._evidence.update(other._evidence)
        self._origin_paths.update(other._origin_paths)
        for key, origins in other._origins.items():
            target = key if mapping is None else mapping.get(key, key)
            if target in self.stats:
                self._origins.setdefault(target, set()).update(origins)

    def is_verified(self) -> bool:
        return not self.unverified_tables()

    def unverified_tables(self) -> list[str]:
        out = set()
        for origins in self._origins.values():
            for origin in origins:
                if type(self._evidence.get(origin)) is not _TableEvidence:
                    out.add(self._origin_paths.get(origin, origin))
        return sorted(out)

    def verified_tables(self) -> list[str]:
        return sorted({self._evidence[o].path
                       for origins in self._origins.values() for o in origins
                       if type(self._evidence.get(o)) is _TableEvidence})

    def verification_records(self) -> list[dict]:
        return [self._evidence[o].to_dict()
                for o in sorted({o for origins in self._origins.values() for o in origins
                                 if type(self._evidence.get(o)) is _TableEvidence},
                                key=lambda o: (self._evidence[o].path, o))]

    def assert_verified(self, what: str = "this statistic") -> None:
        unverified = self.unverified_tables()
        if unverified:
            raise MergeError(
                f"unverified question tables: {unverified} entered this bundle without a "
                f"verification record. `load_question_tables` reads ONE file and never opens "
                f"the dumps it names -- so an export whose `identities` and `sources.sha256` "
                f"were rewritten together reads strictly -- and a bundle built or relabelled "
                f"by hand carries no evidence at all. Nothing may compute {what} from them. "
                f"Load through `pareto.load_verified_question_tables(paths, root=...)`, which "
                f"verifies every table against its dumps and leaves the record behind "
                f"(contract v1.12 §2.10; findings C14 / H23).")

    def clone(self) -> "Bundle":
        out = Bundle(pool=self.pool, normal_form=self.normal_form)
        out.stats = {key: dict(per_q) for key, per_q in self.stats.items()}
        out.meta = {key: dict(meta) for key, meta in self.meta.items()}
        out.unavailable = {name: dict(v) if isinstance(v, dict) else v
                           for name, v in self.unavailable.items()}
        out.notes = [dict(note) for note in self.notes]
        out.carry_evidence(self)
        return out

    def as_question_normal(self) -> "Bundle":
        if self.normal_form == "question":
            return self
        out = Bundle(pool=self.pool, normal_form="question")
        out.unavailable = dict(self.unavailable)
        for key, per_q in self.stats.items():
            out.add(key, {q: (1.0, c / n, t / n, g / n, a / n)
                          for q, (n, c, t, g, a) in per_q.items()},
                    kind=self.meta[key].get("kind", "plan_rule"),
                    meta={k: v for k, v in self.meta[key].items() if k != "kind"})
        out.carry_evidence(self)
        return out

    def assert_weighting(self, *splits: str) -> None:
        for split in dict.fromkeys(s for s in splits if s):
            if any(k.split == split for k in self.stats):
                self.check_weighting(split)

    def check_weighting(self, split: str) -> dict:
        report: dict[str, dict] = {}
        units: set[str] = set()
        offenders: list[str] = []
        for key in self.keys_for(split=split):
            values = {stat[0] for stat in self.stats[key].values()}
            unit = self.meta[key].get("evaluation_unit", "chain")
            units.add(unit)
            constant = len(values) == 1
            report[str(key)] = {"unit": unit, "n_values": sorted(values)[:5],
                                "constant": constant}
            if not constant:
                offenders.append(str(key))
        if self.normal_form == "question":
            return {"units": sorted(units), "series": report, "all_constant": True,
                    "normal_form": "question"}
        if len(units) > 1:
            raise MergeError(
                f"split {split!r} mixes evaluation units {sorted(units)} under the CHAIN normal "
                "form. A question-unit family has no chain weight to carry, so the two are not "
                "on one estimand. Load with normalize='question' (or call "
                "Bundle.as_question_normal()); note that changes the estimand relative to the "
                "archived tables.")
        profiles = {}
        for key in self.keys_for(split=split):
            profiles[str(key)] = tuple(self.stats[key][q][0] for q in sorted(self.stats[key]))
        distinct = set(profiles.values())
        if len(distinct) > 1:
            sizes = {name: sum(prof) for name, prof in list(profiles.items())[:4]}
            raise MergeError(
                f"split {split!r}: series weight the same questions differently "
                f"({len(distinct)} distinct per-question n profiles, e.g. {sizes}). Two series "
                "with identical per-question accuracies but opposite n profiles score "
                "differently, so this comparison is not defined. Use normalize='question'.")
        return {"units": sorted(units), "series": report,
                "all_constant": not offenders, "normal_form": self.normal_form,
                "ragged": bool(offenders)}


UNION_SHARED_IDENTITY = ("model_id", "model_revision", "tokenizer_revision",
                         "prompt_contract_id", "grader_id", "expectation", "knob")
UNION_BANK_SCOPED = ("chain_bank_sha256", "manifest_sha256", "pool", "dataset")


def restrict_split(bundle: Bundle, *, split: str, qkeys: Sequence[str],
                   pool: str | None = None, note: str = "") -> Bundle:
    keep = sorted(set(qkeys))
    if not keep:
        raise MergeError(
            "restrict_split was given an empty question set. An empty dataset row is not a "
            "small sample -- there is no hull, no anchor and no interval -- so this is refused "
            "here rather than reported as an unavailable contrast further downstream.")
    present = set(bundle.qkeys(split))
    missing = [q for q in keep if q not in present]
    if missing:
        raise MergeError(
            f"restrict_split({split!r}): {len(missing)} requested question(s) are not in this "
            f"split, e.g. {missing[:3]} (the split holds {len(present)}). The subset is declared "
            "by a frozen roster, so a missing question means this bundle is not the population "
            "the roster was built from. Refusing rather than intersecting: an intersection "
            "would publish the roster's N beside a number computed on fewer questions.")
    out = bundle.clone()
    for key in list(out.stats):
        if key.split != split:
            continue
        source = out.stats[key]
        out.stats[key] = {q: source[q] for q in keep}
    if pool is not None:
        out.pool = pool
    out.notes.append({
        "note": "restricted_to_declared_subset", "split": split,
        "n_questions_before": len(present), "n_questions_after": len(keep),
        "qkeys_sha256": _qkeys_digest(keep), "derived_pool": pool or bundle.pool,
        "detail": note or ("a DERIVED evaluation over a declared subset of this pool's read "
                           "split; it is not a registered pool population"),
    })
    return out


def _qkeys_digest(qkeys: Sequence[str]) -> str:
    from rl_controller.iclr.banked_baselines import qkey_sha256

    return qkey_sha256(qkeys)


def _series_meta_conflicts(a: Mapping, b: Mapping, fields: Sequence[str]) -> list[tuple]:
    return [(f, a.get(f), b.get(f)) for f in fields
            if (f in a or f in b) and a.get(f) != b.get(f)]


def union_on_split(bundles: Sequence[Bundle], *, split: str, pool: str | None = None,
                   tolerance: float = _COALESCE_TOL, note: str = "") -> Bundle:
    bundles = list(bundles)
    if len(bundles) < 2:
        raise MergeError(
            f"union_on_split needs at least two bundles, got {len(bundles)}. A union of one is "
            "a subset, and `restrict_split` is the call that says so on the file.")
    first = bundles[0]
    forms = {b.normal_form for b in bundles}
    if len(forms) != 1:
        raise MergeError(
            f"union_on_split: the tranches are on different normal forms {sorted(forms)}. A "
            "chain-summed cluster and a per-question mean are different quantities, so merging "
            "them would produce a row weighted one way for one tranche and another way for the "
            "other, with nothing in the output saying which question got which weight.")

    base_keys = set(first.keys_for(split=split))
    tranche_identity = []
    for index, other in enumerate(bundles[1:], start=1):
        other_keys = set(other.keys_for(split=split))
        only_first = sorted(str(k) for k in base_keys - other_keys)
        only_other = sorted(str(k) for k in other_keys - base_keys)
        if only_first or only_other:
            raise MergeError(
                f"union_on_split({split!r}): tranche 0 and tranche {index} were read on "
                f"different arm sets -- {len(only_first)} only in the first "
                f"(e.g. {only_first[:3]}), {len(only_other)} only in the {index}th "
                f"(e.g. {only_other[:3]}). A merged row would then be over both tranches for "
                "some arms and over one for others, while every column carried the same N.")
        for key in sorted(base_keys, key=str):
            shared = sorted(set(first.stats[key]) & set(other.stats[key]))
            if shared:
                raise MergeError(
                    f"union_on_split({split!r}): tranche 0 and tranche {index} share "
                    f"{len(shared)} question(s) on {key}, e.g. {shared[:3]}. Merging them keeps "
                    "one tranche's cluster and silently drops the other's, so the row would be "
                    "computed on fewer distinct questions than it reports.")
            conflicts = _series_meta_conflicts(first.meta[key], other.meta[key],
                                               UNION_SHARED_IDENTITY)
            if conflicts:
                detail = "; ".join(f"{f}: {a!r} in tranche 0 vs {b!r} in tranche {index}"
                                   for f, a, b in conflicts)
                raise MergeError(
                    f"union_on_split({split!r}): {key} is the SAME NAME in both tranches but "
                    f"not the same thing -- {detail}. These tranches did not run one "
                    "deployment: presenting them as one row would publish a policy that was "
                    "never deployed, under a member id that says it was.")
        tranche_identity.append({
            "tranche": index,
            "bank": {f: sorted({str(other.meta[k].get(f)) for k in other_keys
                                if other.meta[k].get(f) is not None})
                     for f in UNION_BANK_SCOPED},
        })

    other_splits = sorted({k.split for b in bundles for k in b.stats} - {split})
    out = first.clone()
    for index, other in enumerate(bundles[1:], start=1):
        for key in sorted(base_keys, key=str):
            out.stats[key] = {**out.stats[key], **other.stats[key]}
        _check_carried_splits(first, other, other_splits, index, tolerance)
        out.carry_evidence(other)
    if pool is not None:
        out.pool = pool
    merged = out.qkeys(split)
    out.check_anchor_agreement()
    out.notes.append({
        "note": "unioned_disjoint_tranches", "split": split, "n_tranches": len(bundles),
        "n_questions": len(merged), "qkeys_sha256": _qkeys_digest(merged),
        "per_tranche_n": [len(b.qkeys(split)) for b in bundles],
        "bank_scoped_identity": [
            {"tranche": 0,
             "bank": {f: sorted({str(first.meta[k].get(f)) for k in base_keys
                                 if first.meta[k].get(f) is not None})
                      for f in UNION_BANK_SCOPED}},
            *tranche_identity],
        "carried_splits": other_splits,
        "derived_pool": pool or first.pool,
        "detail": note or ("raw per-question sufficient statistics merged over the UNIQUE "
                           "question set; no ABH, hull, window or interval was averaged"),
    })
    return out


def _check_carried_splits(first: Bundle, other: Bundle, splits: Sequence[str], index: int,
                          tolerance: float) -> None:
    for split in splits:
        left_keys = {k for k in first.stats if k.split == split}
        right_keys = {k for k in other.stats if k.split == split}
        only_left = sorted(str(k) for k in left_keys - right_keys)
        only_right = sorted(str(k) for k in right_keys - left_keys)
        if only_left or only_right:
            raise MergeError(
                f"union_on_split: the carried split {split!r} holds different series in tranche "
                f"0 and tranche {index} ({len(only_left)} only in the first, e.g. "
                f"{only_left[:3]}; {len(only_right)} only in the {index}th, e.g. "
                f"{only_right[:3]}). {split!r} is carried from the first tranche, so this "
                "difference would be resolved in its favour without a word -- and on `val` that "
                "silently decides which members the merged row evaluates.")
        for key in sorted(left_keys, key=str):
            left, right = first.stats[key], other.stats[key]
            if set(left) != set(right):
                raise MergeError(
                    f"union_on_split: {key} spans {len(left)} questions in tranche 0 and "
                    f"{len(right)} in tranche {index}. The carried split must be the SAME "
                    "population in both: honest selection happens there, and two populations "
                    "do not select one member list.")
            for qkey in sorted(left):
                for name, x, y in zip(ANCHOR_COMPONENTS, left[qkey], right[qkey]):
                    if abs(x - y) <= tolerance * max(1.0, abs(y)):
                        continue
                    raise MergeError(
                        f"union_on_split: {key} disagrees between tranche 0 and tranche {index} "
                        f"on question {qkey!r}: {name} is {x!r} vs {y!r}. On a carried split "
                        "this is decided in the first tranche's favour silently; on `val` that "
                        "means the merged row evaluates members the other tranche never "
                        "selected.")


def mean_row(name: str, per_q: Mapping[str, Sequence[float]], sample: Sequence[str],
             *, cost: str = "total", basis: str | None = None) -> dict:
    if cost not in COST_AXES:
        raise ValueError(f"cost must be one of {COST_AXES}, got {cost!r}")
    if basis is not None and basis not in NORMAL_FORMS:
        raise ValueError(f"basis must be one of {NORMAL_FORMS} or None, got {basis!r}")
    n = correct = thinking = generated = aux = 0.0
    for qkey in sample:
        try:
            a, b, c, d, e = per_q[qkey]
        except KeyError:
            raise MergeError(f"series {name!r} has no question {qkey!r}") from None
        if basis == "question":
            if a <= 0:
                raise MergeError(f"series {name!r} question {qkey!r}: n must be > 0")
            a, b, c, d, e = 1.0, b / a, c / a, d / a, e / a
        n += a
        correct += b
        thinking += c
        generated += d
        aux += e
    if n <= 0:
        raise MergeError(f"series {name!r}: no chains over the given questions")
    mean_thinking = thinking / n
    mean_total = (generated + aux) / n
    return {"name": name, "n": n, "accuracy": correct / n,
            "mean_thinking_tokens": mean_thinking if cost == "thinking" else mean_total,
            "thinking_only_tokens": mean_thinking, "total_cost_tokens": mean_total,
            "overhead_tokens": aux / n}


def resolve_basis(bundle: "Bundle", basis: str | None) -> str | None:
    if basis is None or basis == bundle.normal_form:
        return None
    if basis == "chain" and bundle.normal_form == "question":
        raise MergeError(
            "cannot report a question-normal bundle on the CHAIN basis: normalising set every "
            "per-question n to 1, so the chain weights are unrecoverable. Load with "
            "normalize='chain' from the source rows instead of converting back.")
    return basis


def anchored_hull(rows: Sequence[dict], natural: dict, always: dict) -> list[tuple[float, float]]:
    return lc.anchor_hull(list(rows), natural, always)


def _acc_at(hull: Sequence[tuple[float, float]], tokens: float) -> float:
    if not hull:
        raise ValueError("empty hull")
    if tokens <= hull[0][0]:
        return hull[0][1]
    if tokens >= hull[-1][0]:
        return hull[-1][1]
    value = lc.acc_at_tokens(hull, tokens)
    if value is None:
        raise ValueError(f"hull does not cover {tokens}")
    return value


def _selection_context(*, method: str, seed: int, val_split: str, cost: str,
                       levels: Sequence[float], basis: str | None) -> tuple:
    return (method, seed, val_split, cost, tuple(levels), basis)


def honest_members(bundle: Bundle, *, method: str, seed: int, val_split: str = "val",
                   cost: str = "total", levels: Sequence[float] = SELECT_LEVELS,
                   basis: str | None = None) -> dict:
    bundle.assert_verified("honest member selection")
    cache = bundle._selection_cache
    context = _selection_context(method=method, seed=seed, val_split=val_split, cost=cost,
                                 levels=levels, basis=basis)
    if cache is not None:
        entry = cache.get(context)
        if entry is not None and entry["n_series"] == len(bundle.stats):
            return _copy_selection(entry["selection"])
    basis = resolve_basis(bundle, basis)
    bundle.assert_weighting(val_split)
    val_qkeys = bundle.qkeys(val_split)
    seed = bundle.resolve_seed(method, val_split, seed)
    rows, natural = [], None
    for key in bundle.keys_for(split=val_split, method=method, seed=seed):
        row = mean_row(key.member, bundle.stats[key], val_qkeys, cost=cost, basis=basis)
        rows.append(row)
    for key in bundle.keys_for(split=val_split, method=NATURAL, seed=seed):
        natural = mean_row(NATURAL, bundle.stats[key], val_qkeys, cost=cost, basis=basis)
    if natural is None:
        for key in bundle.keys_for(split=val_split, method=NATURAL):
            natural = mean_row(NATURAL, bundle.stats[key], val_qkeys, cost=cost, basis=basis)
    if natural is None:
        raise NoAnchorsForUnit(
            f"no {NATURAL!r} anchor on split {val_split!r}: honest selection needs "
            "the natural accuracy to define its levels")
    if not rows:
        raise MergeError(f"no val series for method {method!r} seed {seed}")
    members = ld.select_members(rows, natural, levels=levels)
    by_name = {r["name"]: r for r in rows}
    by_level = {}
    for level in levels:
        picked = ld.select_members(rows, natural, levels=(level,))
        by_level[f"{int(round(100 * level))}pp"] = picked[0] if picked else None
    selection = {"members": members, "by_level": by_level, "n_levels": len(levels),
                 "n_distinct": len(members),
                 "val_points": {m: by_name[m] for m in members}}
    if cache is not None:
        cache[context] = {
            "n_series": len(bundle.stats), "selection": _copy_selection(selection),
            "val_identity": {
                "val_split": val_split, "n_val_questions": len(val_qkeys),
                "val_qkeys_sha256": _qkeys_digest(val_qkeys),
                "resolved_seed": seed, "resolved_basis": basis,
                "val_series": [str(k) for k in
                               bundle.keys_for(split=val_split, method=method, seed=seed)],
            },
        }
    return selection


def _copy_selection(selection: dict) -> dict:
    return {"members": list(selection["members"]),
            "by_level": dict(selection["by_level"]),
            "n_levels": selection["n_levels"], "n_distinct": selection["n_distinct"],
            "val_points": {m: dict(row) for m, row in selection["val_points"].items()}}


def natural_val_row(bundle: Bundle, *, split: str = "val", seed: int, cost: str = "total",
                    basis: str | None = None) -> dict:
    bundle.assert_verified("the natural val row")
    basis = resolve_basis(bundle, basis)
    qkeys = bundle.qkeys(split)
    for keys in (bundle.keys_for(split=split, method=NATURAL, seed=seed),
                 bundle.keys_for(split=split, method=NATURAL)):
        for key in keys:
            return mean_row(NATURAL, bundle.stats[key], qkeys, cost=cost, basis=basis)
    raise NoAnchorsForUnit(f"no {NATURAL!r} anchor on split {split!r} for seed {seed}")


def _anchor_rows(bundle: Bundle, *, split: str, seed: int, sample: Sequence[str],
                 cost: str, basis: str | None = None) -> tuple[dict, dict]:
    out = {}
    for anchor in ANCHORS:
        try:
            resolved = bundle.resolve_seed(anchor, split, seed)
        except MergeError as exc:
            raise NoAnchorsForUnit(
                f"split {split!r}: no {anchor!r} anchor series to hang a hull on "
                f"({exc})") from None
        keys = bundle.keys_for(split=split, method=anchor, seed=resolved)
        if not keys:
            raise NoAnchorsForUnit(
                f"split {split!r} seed {seed}: missing the {anchor!r} anchor -- "
                "hulls are only comparable when they share anchors")
        rows = {k: mean_row(anchor, bundle.stats[k], sample, cost=cost, basis=basis)
                for k in keys}
        distinct = {(round(r["accuracy"], 12), round(r["mean_thinking_tokens"], 9))
                    for r in rows.values()}
        if len(distinct) > 1:
            raise MergeError(
                f"split {split!r} seed {seed}: {len(keys)} DIFFERENT {anchor!r} anchor series "
                f"({[str(k) for k in keys]}) -> {sorted(distinct)}. Every method must be "
                "anchored on the same chains; refusing to pick one.")
        out[anchor] = next(iter(rows.values()))
    return out[NATURAL], out[ALWAYS_FIRE]


def method_hull(bundle: Bundle, *, method: str, seed: int, split: str,
                sample: Sequence[str], members: Sequence[str] | None = None,
                cost: str = "total", basis: str | None = None) -> list[tuple[float, float]]:
    bundle.assert_verified("a hull")
    basis = resolve_basis(bundle, basis)
    bundle.assert_weighting(split)
    seed = bundle.resolve_seed(method, split, seed)
    keys = bundle.keys_for(split=split, method=method, seed=seed)
    if members is not None:
        wanted = set(members)
        keys = [k for k in keys if k.member in wanted]
        missing = wanted - {k.member for k in keys}
        if missing:
            raise MergeError(f"{method}/s{seed}/{split}: val-selected member(s) {sorted(missing)} "
                             "were not read on this split")
    if not keys:
        raise MergeError(f"no series for {method}/s{seed} on {split!r}")
    rows = [mean_row(k.member, bundle.stats[k], sample, cost=cost, basis=basis) for k in keys]
    natural, always = _anchor_rows(bundle, split=split, seed=seed, sample=sample, cost=cost,
                                   basis=basis)
    return anchored_hull(rows, natural, always)


def honest_hull(bundle: Bundle, *, method: str, seed: int, split: str,
                sample: Sequence[str], val_split: str = "val", cost: str = "total",
                basis: str | None = None) -> tuple[list[tuple[float, float]], dict]:
    selection = honest_members(bundle, method=method, seed=seed, val_split=val_split,
                               cost=cost, basis=basis)
    if not selection["members"]:
        raise NoHonestMember(
            f"{method}/s{seed}: no member reaches natural accuracy - 3 pp on {val_split!r}, so "
            "the method has no honest operating point. Report it as unavailable -- do NOT fall "
            "back to its sweep hull, which would compare an honestly-selected method against an "
            "oracle-selected one.")
    hull = method_hull(bundle, method=method, seed=seed, split=split, sample=sample,
                       members=selection["members"], cost=cost, basis=basis)
    return hull, selection


def free_frontier(bundle: Bundle, *, split: str, sample: Sequence[str], cost: str = "total",
                  basis: str | None = None) -> dict:
    bundle.assert_verified("the score-free frontier")
    basis = resolve_basis(bundle, basis)
    rules = sorted({k.method for k in bundle.keys_for(split=split) if is_free_rule(k.method)})
    if not rules:
        return {"rules": [], "points": [], "hull": [], "n_points": 0}
    rows, points = [], []
    for rule in rules:
        for key in bundle.keys_for(split=split, method=rule):
            row = mean_row(key.member, bundle.stats[key], sample, cost=cost, basis=basis)
            rows.append(row)
            points.append({"rule": rule, "member": key.member, "seed": key.seed,
                           "tokens": row["mean_thinking_tokens"], "accuracy": row["accuracy"],
                           "thinking_only_tokens": row["thinking_only_tokens"],
                           "total_cost_tokens": row["total_cost_tokens"]})
    anchor_seed = bundle.seeds(NATURAL, split)[0]
    natural, always = _anchor_rows(bundle, split=split, seed=anchor_seed, sample=sample,
                                   cost=cost, basis=basis)
    return {"rules": rules, "n_points": len(points),
            "points": sorted(points, key=lambda pt_: (pt_["tokens"], pt_["rule"])),
            "hull": anchored_hull(rows, natural, always)}


def abh(hull: Sequence[tuple[float, float]],
        reference: Sequence[tuple[float, float]]) -> dict:
    if not hull or not reference:
        raise ValueError("ABH needs two non-empty hulls")
    t_lo = max(hull[0][0], reference[0][0])
    t_hi = min(hull[-1][0], reference[-1][0])
    if t_hi - t_lo <= _TOKEN_EPS:
        return {"abh_pp": None, "abh_area": None, "t_lo": t_lo, "t_hi": t_hi, "width": 0.0}
    breaks = sorted({t for t, _ in hull if t_lo < t < t_hi}
                    | {t for t, _ in reference if t_lo < t < t_hi} | {t_lo, t_hi})
    area = 0.0
    for x0, x1 in zip(breaks, breaks[1:]):
        d0 = _acc_at(hull, x0) - _acc_at(reference, x0)
        d1 = _acc_at(hull, x1) - _acc_at(reference, x1)
        area += 0.5 * (d0 + d1) * (x1 - x0)
    width = t_hi - t_lo
    return {"abh_pp": 100.0 * area / width, "abh_area": area,
            "t_lo": t_lo, "t_hi": t_hi, "width": width}


def abh_seed_mean(per_seed: Mapping[int, dict]) -> dict:
    values = [v["abh_pp"] for v in per_seed.values()]
    defined = [v for v in values if v is not None]
    complete = len(defined) == len(values) and bool(values)
    return {"abh_pp": (sum(defined) / len(defined)) if complete else None,
            "per_seed": {str(s): v for s, v in per_seed.items()},
            "n_seeds": len(per_seed), "n_defined": len(defined), "complete": complete,
            "same_sign": complete
                         and (all(v > 0 for v in defined) or all(v < 0 for v in defined))}


def bootstrap_plan(qkeys: Sequence[str], *, n: int = DEFAULT_BOOTSTRAP,
                   seed: int = 0) -> list[tuple[str, ...]]:
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError(f"n must be a non-negative int, got {n!r}")
    ordered = sorted(set(qkeys))
    if not ordered:
        raise ValueError("bootstrap plan needs at least one question")
    rng = random.Random(seed)
    return [tuple(ordered[rng.randrange(len(ordered))] for _ in ordered) for _ in range(n)]


def abh_with_ci(bundle: Bundle, *, method: str, reference: str, split: str,
                plan: Sequence[Sequence[str]], val_split: str = "val", cost: str = "total",
                honest: bool = True, basis: str | None = None) -> dict:
    bundle.assert_verified("an ABH")
    basis = resolve_basis(bundle, basis)
    qkeys = bundle.qkeys(split)
    seeds = _paired_seeds(bundle, method, reference, split)

    def one(sample):
        per_seed = {}
        for seed in seeds:
            a = (honest_hull(bundle, method=method, seed=seed, split=split, sample=sample,
                             val_split=val_split, cost=cost, basis=basis)[0] if honest else
                 method_hull(bundle, method=method, seed=seed, split=split, sample=sample,
                             cost=cost, basis=basis))
            b = (honest_hull(bundle, method=reference, seed=seed, split=split, sample=sample,
                             val_split=val_split, cost=cost, basis=basis)[0] if honest else
                 method_hull(bundle, method=reference, seed=seed, split=split, sample=sample,
                             cost=cost, basis=basis))
            per_seed[seed] = abh(a, b)
        return abh_seed_mean(per_seed)

    point = one(qkeys)
    values, undefined = [], 0
    for sample in plan:
        value = one(sample)["abh_pp"]
        if value is None:
            undefined += 1
        else:
            values.append(value)
    ci_ok = undefined == 0 and len(values) > 1
    out = {"method": method, "reference": reference, "split": split, "cost_axis": cost,
           "basis": basis or bundle.normal_form,
           "hull": "honest" if honest else "sweep",
           "abh_pp": point["abh_pp"], "per_seed": point["per_seed"],
           "same_sign": point["same_sign"], "n_seeds": point["n_seeds"],
           "complete_seeds": point["complete"],
           "n_questions": len(qkeys), "replicates": len(plan), "n_defined": len(values),
           "n_undefined_replicates": undefined,
           "se": statistics.stdev(values) if ci_ok else None,
           "ci95": _percentile_ci(values) if ci_ok else [None, None],
           "ci_status": ("ok" if ci_ok else
                         "unavailable: some bootstrap replicate had no shared token range"
                         if undefined else "unavailable: fewer than 2 defined replicates"),
           "semantics": "question-paired across methods, configs and observed seeds under one "
                        "shared resample plan; SE conditional on the observed seeds"}
    if honest:
        out["members_by_seed"] = {
            str(seed): honest_members(bundle, method=method, seed=seed, val_split=val_split,
                                      cost=cost, basis=basis)["members"] for seed in seeds}
        out["n_distinct_members"] = {
            str(seed): len(v) for seed, v in
            ((s, out["members_by_seed"][str(s)]) for s in seeds)}
    return out


def _paired_seeds(bundle: Bundle, method: str, reference: str, split: str) -> list[int]:
    left = bundle.seeds(method, split)
    right = bundle.seeds(reference, split)
    if not left or not right:
        raise MergeError(f"{method!r} or {reference!r} has no series on {split!r}")
    if left == right:
        return left
    if right == [-1]:
        return left
    if left == [-1]:
        return right
    shared = sorted(set(left) & set(right))
    if not shared:
        raise MergeError(f"{method!r} (seeds {left}) and {reference!r} (seeds {right}) share no "
                         "seed -- refusing an unpaired comparison")
    raise MergeError(f"{method!r} (seeds {left}) and {reference!r} (seeds {right}) were read on "
                     f"different seed sets; averaging over {shared} would not be paired")


def _percentile_ci(values: Sequence[float], lo: float = 0.025, hi: float = 0.975):
    if not values:
        return [None, None]
    return [_percentile(values, lo), _percentile(values, hi)]


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def question_table_costed(chains: Sequence[dict], family: Iterable[tuple[str, ld.Plan]], *,
                          answer_format: str = "boxed", force_natural_answer: bool = False,
                          encode=None, programs=None, overhead_per_boundary: float = 0.0,
                          overhead_key: str | None = None
                          ) -> dict[str, dict[str, tuple[float, float, float, float, float]]]:
    members = list(family)
    names = [name for name, _ in members]
    if len(set(names)) != len(names):
        raise MergeError("duplicate member names in family")
    if overhead_per_boundary < 0:
        raise MergeError("overhead_per_boundary must be >= 0")
    if overhead_key and overhead_per_boundary:
        raise MergeError("declare overhead as EITHER overhead_per_boundary (constant) OR "
                         "overhead_key (per probe), never both -- two modes would double-charge")
    table: dict[str, dict[str, tuple[float, float, float, float, float]]] = {}
    for name, plan in members:
        sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
        for chain in chains:
            probe = plan(chain)
            correct, _ = ld.stop_outcome(chain, plan)
            cost = ta.deployment_cost(chain, probe=probe, answer_format=answer_format,
                                      force_natural_answer=force_natural_answer,
                                      encode=encode, programs=programs)
            n_reached = (len(chain["probes"]) if probe is None
                         else sum(1 for p in chain["probes"] if p["block"] <= probe["block"]))
            if overhead_key:
                try:
                    aux = sum(float(s[overhead_key])
                              for s in chain["scores"][:n_reached])
                except KeyError:
                    raise MergeError(
                        f"chain {(chain['qkey'], chain['rep'])}: overhead_key "
                        f"{overhead_key!r} is not on every score dict") from None
            else:
                aux = overhead_per_boundary * n_reached
            entry = sums[chain["qkey"]]
            entry[0] += 1
            entry[1] += correct
            entry[2] += cost["thinking_tokens"]
            entry[3] += cost["generated_tokens"]
            entry[4] += aux
        table[name] = {q: tuple(v) for q, v in sums.items()}
    return table


def add_run_dir(bundle: Bundle, run_dir, *, method_of=None, split_map=None, seed=None,
                specs=None, include_builtin: bool = True, calibrate: bool = True,
                meta=None, **costing) -> list[SeriesKey]:
    from rl_controller.scripts import iclr_ladder as script

    run = script.load_run(run_dir)
    seed = run["seed"] if seed is None else seed
    split_map = split_map or {"val": "val", "select": "select"}
    models = ld.fit_calibrators(run["val"]) if calibrate else None
    added = []
    for dump_split, out_split in split_map.items():
        if not calibrate:
            chains = run[dump_split]
        elif dump_split == "val":
            chains = ld.crossfit_calibrated(run["val"])[0]
        else:
            chains = ld.attach_calibrated(run[dump_split], models)
        keys = ld.score_keys(chains)
        fams = ld.families(keys, specs=specs or (), include_registered=False)
        if not include_builtin:
            declared = {s.name for s in (specs or ())}
            fams = {name: family for name, family in fams.items() if name in declared}
        for rung, family in fams.items():
            name = method_of(rung) if method_of else rung
            for member, per_q in question_table_costed(chains, family, **costing).items():
                key = SeriesKey(name, member, seed, out_split)
                bundle.add(key, per_q, kind="plan_rule", meta=dict(meta or {}))
                added.append(key)
        anchor_costing = {**costing, "overhead_per_boundary": 0.0, "overhead_key": None}
        for anchor, plan in ((NATURAL, ld.natural), (ALWAYS_FIRE, ld.always_fire)):
            for member, per_q in question_table_costed(chains, [(anchor, plan)],
                                                       **anchor_costing).items():
                key = SeriesKey(anchor, member, seed, out_split)
                if key not in bundle.stats:
                    bundle.add(key, per_q, kind="plan_rule", meta=dict(meta or {}))
                    added.append(key)
    return added


BINDING_LEVELS = ("chains", "questions")

_REEXPORT_HINT = (
    "RE-EXPORT it (the numbers are cheap to re-ladder from the scored dumps; the identity is "
    "not stampable after the fact) with `python -m rl_controller.scripts.iclr_ladder --runs "
    "<the run dirs this ladder read> --pool <inventory pool id> --export-question-tables "
    "<new file> --out <ladder json> --overhead-accounting <this method's declaration>`, "
    "keeping the producer's own cost flags.")


def _is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def validate_self_authentication(path, sources: Mapping, identities: Mapping,
                                 qkeys: Mapping, *, required, allow_absent: bool = False) -> dict:
    if not sources and not identities:
        if allow_absent:
            return {}
        raise MergeError(
            f"{path}: declares NO chain binding at all -- no `sources`, no `identities`. A "
            f"question table's numbers are readable only when it names the dump behind every "
            f"(seed, split) and the exact (qkey, rep) chains it read from it (contract v1.10 "
            f"\u00a72.10). {_REEXPORT_HINT}")
    if not sources or not identities:
        raise MergeError(f"{path}: `sources` and `identities` are all-or-none -- a table that "
                         f"names its dumps must also name the chains it read from them, and "
                         f"vice versa. {_REEXPORT_HINT}")
    out_sources: dict[str, dict] = {}
    out_identities: dict[str, dict] = {}
    for seed, by_split in sorted(dict(sources).items()):
        for split, entry in sorted(dict(by_split).items()):
            entry = dict(entry)
            if not isinstance(entry.get("path"), str) or not entry["path"]:
                raise MergeError(f"{path}: sources[{seed}][{split}] names no dump path")
            if not _is_sha256(entry.get("sha256")):
                raise MergeError(f"{path}: sources[{seed}][{split}].sha256 must be 64 lowercase "
                                 f"hex, got {entry.get('sha256')!r}")
            rows = ((identities.get(seed) or {}).get(split))
            if rows is None:
                raise MergeError(f"{path}: sources[{seed}][{split}] declares a dump but no "
                                 "chain identities were exported for it")
            pairs = [(str(q), int(r)) for q, r in rows]
            if len(set(pairs)) != len(pairs):
                raise MergeError(f"{path}: identities[{seed}][{split}] repeats a (qkey, rep); "
                                 "a scored dump holds each chain once")
            if pairs != sorted(pairs):
                raise MergeError(f"{path}: identities[{seed}][{split}] is not sorted; the list "
                                 "is an identity SET and must be written in canonical order")
            if entry.get("n_chains") != len(pairs):
                raise MergeError(f"{path}: sources[{seed}][{split}].n_chains "
                                 f"{entry.get('n_chains')!r} != {len(pairs)} declared chain "
                                 "identities")
            declared = {q for q, _ in pairs}
            table_qkeys = set(qkeys.get(split) or ())
            if declared != table_qkeys:
                only_ids = sorted(declared - table_qkeys)[:3]
                only_table = sorted(table_qkeys - declared)[:3]
                raise MergeError(
                    f"{path}: identities[{seed}][{split}] covers {len(declared)} questions but "
                    f"the table reports {len(table_qkeys)}; only in identities {only_ids}, only "
                    f"in table {only_table}. A table may not claim chains it does not report, "
                    "nor report questions no chain covers.")
            out_sources.setdefault(str(seed), {})[str(split)] = {
                "path": entry["path"], "sha256": entry["sha256"], "n_chains": len(pairs)}
            out_identities.setdefault(str(seed), {})[str(split)] = [[q, r] for q, r in pairs]
    missing = sorted((s, sp) for s, sp in required
                     if sp not in (out_sources.get(str(s)) or {}))
    if missing:
        raise MergeError(f"{path}: no source dump / chain identities for {missing}; every "
                         f"(seed, split) a series covers must be bound to the chains it read. "
                         f"{_REEXPORT_HINT}")
    return {"sources": out_sources, "identities": out_identities}


def _dump_identities(lines: Iterable[str]) -> list[tuple[str, int]]:
    out = []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise MergeError(f"line {lineno} of the dump is not JSON: {exc}") from None
        if "qkey" not in row or "rep" not in row:
            raise MergeError(f"line {lineno} of the dump carries no (qkey, rep); it is not a "
                             "scored chain dump")
        out.append((str(row["qkey"]), int(row["rep"])))
    return out


RESTRICTION_KINDS = ("none", "select_manifest", "subset_domain")
RESTRICTION_DOMAINS = ("code", "math")
RESTRICTION_SPLITS = ("val", "select")
RESTRICTION_USAGES = ("calibration", "exploration", "final")
_RESTRICTION_HINT = (
    "`restriction` is written by `iclr_ladder --export-question-tables` from its OWN "
    "--select-manifest / --subset-manifest + --subset-domain flags; re-export to obtain one. "
    "The registry row must authorize its restriction kind, canonical pool, affected split, "
    "and usage with machine-readable `allowed_*` lists.")


MANIFEST_REGISTRY = (Path(__file__).resolve().parents[2] / "docs" / "superpowers"
                     / "contracts" / "2026-09-03-registered-manifests.jsonl")
_REGISTRY_CACHE: dict[str, dict] = {}
_REGISTRY_HINT = (
    "A restriction manifest must be REGISTERED: add `{path, sha256, purpose}` to "
    "docs/superpowers/contracts/2026-09-03-registered-manifests.jsonl in a reviewed commit "
    "and re-export. A manifest the producer wrote and declared for itself authorises "
    "nothing -- it is the producer's claim about its own subsetting, not evidence for it "
    "(contract v1.12 §2.10; finding C20).")


def load_manifest_registry(path=None) -> dict:
    path = Path(MANIFEST_REGISTRY if path is None else path)
    try:
        stat = path.stat()
    except OSError as exc:
        raise MergeError(f"{path}: the restriction-manifest registry is unreadable ({exc}). "
                         f"{_REGISTRY_HINT}") from None
    token = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _REGISTRY_CACHE.get(str(path))
    if cached is not None and cached["token"] == token:
        return cached
    by_path: dict[str, str] = {}
    by_name: dict[str, set] = {}
    entries: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise MergeError(f"{path}:{lineno}: registry row is not JSON: {exc}") from None
        declared, sha = row.get("path"), row.get("sha256")
        if not isinstance(declared, str) or not declared.strip():
            raise MergeError(f"{path}:{lineno}: registry row names no `path`")
        if not _is_sha256(sha):
            raise MergeError(f"{path}:{lineno}: registry row `sha256` must be 64 lowercase "
                             f"hex, got {sha!r}")
        if not str(row.get("purpose") or "").strip():
            raise MergeError(f"{path}:{lineno}: registry row for {declared} carries no "
                             "`purpose`; an unexplained registration is indistinguishable "
                             "from a mistake")
        permissions = {}
        allowed_values = {
            "allowed_restriction_kinds": set(RESTRICTION_KINDS) - {"none"},
            "allowed_pools": set(CANONICAL_POOLS),
            "allowed_splits": set(RESTRICTION_SPLITS),
            "allowed_usages": set(RESTRICTION_USAGES),
        }
        for field_, allowed in allowed_values.items():
            values = row.get(field_)
            if (not isinstance(values, list) or not values
                    or any(not isinstance(value, str) for value in values)):
                raise MergeError(
                    f"{path}:{lineno}: registry row for {declared} must carry a non-empty "
                    f"string list `{field_}`; prose `purpose` is not authorization")
            if len(values) != len(set(values)):
                raise MergeError(f"{path}:{lineno}: `{field_}` contains duplicates")
            unknown = sorted(set(values) - allowed)
            if unknown:
                raise MergeError(f"{path}:{lineno}: `{field_}` contains unknown values "
                                 f"{unknown}; expected a subset of {sorted(allowed)}")
            permissions[field_] = tuple(values)
        key = _registry_key(declared)
        if key in by_path:
            raise MergeError(f"{path}:{lineno}: {declared} is registered twice; one path, "
                             "one reviewed authorization row")
        by_path[key] = sha
        by_name.setdefault(key.rsplit("/", 1)[-1], set()).add(sha)
        entries.append({"path": declared, "sha256": sha, "purpose": row["purpose"],
                        **permissions})
    if not entries:
        raise MergeError(f"{path}: the restriction-manifest registry is empty, so no export "
                         f"may declare a manifest at all. {_REGISTRY_HINT}")
    out = {"token": token, "source": str(path), "by_path": by_path, "by_name": by_name,
           "entries": entries}
    _REGISTRY_CACHE[str(path)] = out
    return out


def _registry_key(value) -> str:
    key = str(value).replace("\\", "/")
    while key.startswith("./"):
        key = key[2:]
    return key


def check_manifest_registered(declared_path, sha256, *, registry=None,
                              restriction_kind=None, pool=None, splits=(), usage=None) -> dict:
    reg = registry if isinstance(registry, Mapping) else load_manifest_registry(registry)
    key = _registry_key(declared_path)
    name = key.rsplit("/", 1)[-1]
    matched = None
    matched_by = None
    if reg["by_path"].get(key) == sha256:
        matched = next(entry for entry in reg["entries"]
                       if _registry_key(entry["path"]) == key and entry["sha256"] == sha256)
        matched_by = "path"
    known = reg["by_name"].get(name) or set()
    if matched is None and sha256 in known:
        candidates = [entry for entry in reg["entries"]
                      if _registry_key(entry["path"]).rsplit("/", 1)[-1] == name
                      and entry["sha256"] == sha256]
        if len(candidates) != 1:
            raise MergeError(
                f"restriction manifest {name} matches {len(candidates)} registry rows with the "
                "same digest; basename authorization is ambiguous. Keep one reviewed row or "
                "use distinct basenames.")
        matched, matched_by = candidates[0], "basename"
    if matched is not None:
        checks = (
            ("restriction kind", restriction_kind, "allowed_restriction_kinds"),
            ("pool", resolve_pool(pool) if pool is not None else None, "allowed_pools"),
            ("usage", usage, "allowed_usages"),
        )
        for label, value, field_ in checks:
            if value is not None and value not in matched[field_]:
                raise MergeError(
                    f"registered manifest {matched['path']} does not permit {label} "
                    f"{value!r}; allowed by `{field_}`: {list(matched[field_])}")
        disallowed = sorted(set(splits or ()) - set(matched["allowed_splits"]))
        if disallowed:
            raise MergeError(
                f"registered manifest {matched['path']} does not permit affected split(s) "
                f"{disallowed}; allowed by `allowed_splits`: "
                f"{list(matched['allowed_splits'])}")
        return {"matched": matched_by, "path": key, "sha256": sha256,
                "registry": reg["source"],
                "authorization": {field_: list(matched[field_]) for field_ in (
                    "allowed_restriction_kinds", "allowed_pools", "allowed_splits",
                    "allowed_usages")}}
    if known:
        raise MergeError(
            f"restriction declares manifest {declared_path} with sha256 {sha256}, but "
            f"{name} is registered in {reg['source']} under {sorted(known)}. This is a "
            f"DIFFERENT file wearing a registered name. {_REGISTRY_HINT}")
    raise MergeError(
        f"restriction declares manifest {declared_path} (sha256 {sha256}), which is NOT "
        f"registered in {reg['source']}. A restriction is the producer's account of which "
        f"questions it was entitled to drop; verified against a manifest the same producer "
        f"supplied, it authorises itself. Registered manifests: "
        f"{sorted(reg['by_name'])}. {_REGISTRY_HINT}")


def manifest_domains(lines: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise MergeError(f"line {lineno}: manifest row is not JSON: {exc}") from None
        if "qkey" not in row:
            raise MergeError(f"line {lineno}: manifest row has no 'qkey'")
        qkey, domain = str(row["qkey"]), str(row.get("domain") or "math")
        if out.get(qkey, domain) != domain:
            raise MergeError(f"line {lineno}: qkey {qkey} has domain {domain!r} here but "
                             f"{out[qkey]!r} earlier in the same manifest")
        out[qkey] = domain
    return out


def validate_restriction(path, restriction, *, allow_absent: bool = False,
                         registry=None, pool=None, splits=(), usage=None) -> dict | None:
    if restriction is None:
        if allow_absent:
            return None
        raise MergeError(
            f"{path}: declares no `restriction` block. Without it, an export that DROPPED "
            f"whole questions is indistinguishable from one that legitimately subset its bank "
            f"by manifest, so its question set cannot be checked at all (contract v1.11 §2.10; "
            f"finding C15). {_RESTRICTION_HINT}")
    if not isinstance(restriction, Mapping):
        raise MergeError(f"{path}: `restriction` must be an object, got "
                         f"{type(restriction).__name__}")
    kind = restriction.get("kind")
    if kind not in RESTRICTION_KINDS:
        raise MergeError(f"{path}: restriction.kind {kind!r} is not one of {RESTRICTION_KINDS}")
    if usage is not None and usage not in RESTRICTION_USAGES:
        raise MergeError(f"{path}: restriction usage {usage!r} is not one of "
                         f"{RESTRICTION_USAGES}")
    affected_splits = ({"select"} if kind == "select_manifest"
                       else set(splits or ()) if kind == "subset_domain" else set())
    manifests = []
    for i, entry in enumerate(restriction.get("manifests") or ()):
        if not isinstance(entry, Mapping):
            raise MergeError(f"{path}: restriction.manifests[{i}] must be an object")
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            raise MergeError(f"{path}: restriction.manifests[{i}] names no path")
        if not _is_sha256(entry.get("sha256")):
            raise MergeError(f"{path}: restriction.manifests[{i}].sha256 must be 64 lowercase "
                             f"hex, got {entry.get('sha256')!r}")
        try:
            check_manifest_registered(
                entry["path"], entry["sha256"], registry=registry,
                restriction_kind=kind, pool=pool, splits=affected_splits, usage=usage)
        except MergeError as exc:
            raise MergeError(f"{path}: restriction.manifests[{i}]: {exc}") from None
        manifests.append({"path": entry["path"], "sha256": entry["sha256"]})
    domain = restriction.get("domain")
    if kind == "none" and (manifests or domain is not None):
        raise MergeError(f"{path}: restriction.kind 'none' names a manifest or a domain; an "
                         "unrestricted export restricts nothing")
    if kind != "none" and not manifests:
        raise MergeError(f"{path}: restriction.kind {kind!r} names no manifest, so the question "
                         f"set it claims cannot be derived. {_RESTRICTION_HINT}")
    if kind == "select_manifest" and domain is not None:
        raise MergeError(f"{path}: restriction.kind 'select_manifest' carries a domain; "
                         "--select-manifest selects by qkey, not by domain")
    if kind == "subset_domain" and domain not in RESTRICTION_DOMAINS:
        raise MergeError(f"{path}: restriction.domain {domain!r} is not one of "
                         f"{RESTRICTION_DOMAINS}")
    mirror = (manifests[0] if len(manifests) == 1 else {"path": None, "sha256": None})
    for field_, want in (("manifest_path", mirror["path"]),
                         ("manifest_sha256", mirror["sha256"])):
        got = restriction.get(field_)
        if got is not None and got != want:
            raise MergeError(f"{path}: restriction.{field_} {got!r} != {want!r} from "
                             "restriction.manifests; one file, one declaration")
    return {"kind": kind, "manifests": manifests, "domain": domain,
            "manifest_path": mirror["path"], "manifest_sha256": mirror["sha256"]}


def read_declared_manifests(restriction: Mapping, *, root=None, registry=None) -> dict[str, str]:
    from rl_controller.iclr.provenance import parse_and_record

    root = None if root is None else Path(root)
    out: dict[str, str] = {}
    for entry in restriction["manifests"]:
        check_manifest_registered(entry["path"], entry["sha256"], registry=registry)
        path = Path(entry["path"])
        if not path.is_absolute() and root is not None:
            path = root / path
        if not path.is_file():
            raise MergeError(
                f"restriction names manifest {entry['path']}, which is not a readable file. "
                "The question set an export claims is derived from the manifest it applied, so "
                "verification needs that file; run the assembly where the artifacts live.")
        try:
            rows, record = parse_and_record(path, manifest_domains)
        except MergeError as exc:
            raise MergeError(f"{path}: {exc}") from None
        if record["sha256"] != entry["sha256"]:
            raise MergeError(
                f"restriction declares sha256 {entry['sha256']} for manifest {path}, but that "
                f"file hashes to {record['sha256']}. The manifest has changed since the export, "
                "so the question set it now defines is not the one that was read.")
        for qkey, domain in rows.items():
            if out.get(qkey, domain) != domain:
                raise MergeError(f"{path}: qkey {qkey} has domain {domain!r} here but "
                                 f"{out[qkey]!r} in another declared manifest")
            out[qkey] = domain
    return out


def _expected_questions(restriction: Mapping, domains: Mapping[str, str] | None,
                        dump_questions: set, *, split: str) -> set:
    kind = restriction["kind"]
    if kind == "none":
        return set(dump_questions)
    if kind == "select_manifest":
        if split != "select":
            return set(dump_questions)
        return {q for q in dump_questions if q in domains}
    unknown = sorted(set(dump_questions) - set(domains))
    if unknown:
        raise MergeError(
            f"restriction 'subset_domain' declares manifests that do not cover "
            f"{len(unknown)} question(s) of the dump (e.g. {unknown[:3]}); the ladder refuses "
            "that at export time, so this table was not produced by the run it declares")
    return {q for q in dump_questions if domains[q] == restriction["domain"]}


def declares_nothing(payload: Mapping) -> bool:
    if payload.get("comparability"):
        return False
    if any((row.get("meta") or {}).get(field_) is not None
           for row in payload.get("series") or ()
           for field_ in SERIES_IDENTITY_FIELDS):
        return False
    return not (payload.get("sources") or payload.get("identities"))


def legacy_refusal(path, n_series: int) -> LegacyStampMissing:
    return LegacyStampMissing(
        f"{path}: carries no comparability block, no series identity and no chain "
        f"binding at all ({n_series} series): it predates the `--pool` requirement, "
        f"which is what introduced the block (contract v1.9 §2.9). {_REEXPORT_HINT}")


def verify_self_authentication(table, *, root=None, registry=None, usage="final") -> dict:
    from rl_controller.iclr.provenance import parse_and_record

    table = Path(table)
    payload = json.loads(table.read_text(encoding="utf-8"))
    if declares_nothing(payload):
        raise legacy_refusal(str(table), len(payload.get("series") or ()))
    covered = {(str(row["seed"]), str(row["split"])) for row in payload.get("series") or ()}
    bound = validate_self_authentication(str(table), payload.get("sources") or {},
                                         payload.get("identities") or {},
                                         payload.get("qkeys") or {}, required=covered)
    restriction = validate_restriction(
        str(table), payload.get("restriction"), registry=registry,
        pool=payload.get("pool"), splits=(payload.get("qkeys") or {}).keys(), usage=usage)
    root = None if root is None else Path(root)
    domains = (read_declared_manifests(restriction, root=root, registry=registry)
               if restriction["kind"] != "none" else None)
    verified = []
    for seed, by_split in sorted(bound["sources"].items()):
        for split, entry in sorted(by_split.items()):
            declared = Path(entry["path"])
            if not declared.is_absolute() and root is not None:
                declared = root / declared
            if not declared.is_file():
                raise MergeError(
                    f"{table}: sources[{seed}][{split}] names {entry['path']}, which is not a "
                    "readable file. A table's numbers are verifiable only where its dumps are; "
                    "run the assembly where the artifacts live rather than trusting the "
                    "declaration.")
            try:
                pairs, record = parse_and_record(declared, _dump_identities)
            except MergeError as exc:
                raise MergeError(f"{table}: sources[{seed}][{split}] ({declared}): {exc}") from None
            if record["sha256"] != entry["sha256"]:
                raise MergeError(
                    f"{table}: sources[{seed}][{split}] declares sha256 {entry['sha256']} for "
                    f"{declared}, but that file hashes to {record['sha256']}. The dump has "
                    "changed since the export, or the table names a different dump; either way "
                    "these numbers are not bound to these bytes.")
            in_dump = set(pairs)
            claimed = {(str(q), int(r)) for q, r in bound["identities"][seed][split]}
            phantom = sorted(claimed - in_dump)
            if phantom:
                raise MergeError(
                    f"{table}: identities[{seed}][{split}] claims {len(phantom)} chain(s) the "
                    f"dump does not contain (e.g. {phantom[:3]}). The identity list was written "
                    "by something other than the parse that produced these numbers.")
            questions = {q for q, _ in claimed}
            dump_questions = {q for q, _ in in_dump}
            try:
                expected = _expected_questions(restriction, domains, dump_questions,
                                               split=split)
            except MergeError as exc:
                raise MergeError(f"{table}: sources[{seed}][{split}]: {exc}") from None
            if questions != expected:
                dropped = sorted(expected - questions)
                extra = sorted(questions - expected)
                raise MergeError(
                    f"{table}: identities[{seed}][{split}] reports {len(questions)} questions "
                    f"but the declared restriction ({restriction['kind']!r}"
                    + (f", domain {restriction['domain']!r}" if restriction["domain"] else "")
                    + f") over {len(dump_questions)} dump questions implies {len(expected)}: "
                    f"{len(dropped)} missing (e.g. {dropped[:3]}), {len(extra)} not in the "
                    f"derived set (e.g. {extra[:3]}). An export reports EVERY question its "
                    "declared restriction leaves; dropping whole questions from the table, the "
                    "series and the identities together is a silent subsample that reads "
                    "strictly (finding C15).")
            withheld = sorted({p for p in in_dump if p[0] in questions} - claimed)
            if withheld:
                raise MergeError(
                    f"{table}: the dump holds {len(withheld)} chain(s) for questions the table "
                    f"REPORTS that identities[{seed}][{split}] does not declare (e.g. "
                    f"{withheld[:3]}). A run may restrict itself by QUESTION (--select-manifest "
                    "/ --subset-domain); dropping individual reps of a reported question is a "
                    "silent subsample.")
            verified.append({"seed": seed, "split": split, "path": str(declared),
                             "sha256": record["sha256"], "n_chains": len(claimed),
                             "n_dump_chains": len(in_dump),
                             "n_questions": len(questions),
                             "n_expected_questions": len(expected),
                             "n_dump_questions": len(dump_questions),
                             "restricted": len(claimed) != len(in_dump)})
    return {"table": str(table), "restriction": restriction, "verified": verified,
            "restricted": any(v["restricted"] for v in verified)}


def dump_question_tables(path, bundle: Bundle, *, scope: str = "honest",
                         provenance: Sequence[dict] = (),
                         comparability: Mapping | None = None,
                         sources: Mapping | None = None,
                         identities: Mapping | None = None,
                         restriction: Mapping | None = None,
                         head_tag: str | None = None) -> dict:
    bundle.assert_verified("a question-table export")
    splits = sorted({k.split for k in bundle.stats})
    qkeys = {split: list(bundle.qkeys(split)) for split in splits}
    series = []
    for key in sorted(bundle.stats, key=lambda k: (k.split, k.method, k.member, k.seed)):
        per_q = bundle.stats[key]
        columns = list(zip(*(per_q[q] for q in qkeys[key.split])))
        series.append({"method": key.method, "member": key.member, "seed": key.seed,
                       "split": key.split, "kind": bundle.meta[key].get("kind", "plan_rule"),
                       "meta": {k: v for k, v in bundle.meta[key].items() if k != "kind"},
                       "n": list(columns[0]), "correct": list(columns[1]),
                       "thinking": list(columns[2]), "generated": list(columns[3]),
                       "aux": list(columns[4])})
    block = dict(comparability or {})
    if block and block.get("pool") not in (None, bundle.pool):
        raise MergeError(f"comparability block declares pool {block['pool']!r} but the bundle "
                         f"is {bundle.pool!r}; one file, one pool")
    bound = validate_self_authentication(
        str(path), sources or {}, identities or {}, qkeys,
        required={(str(key.seed), key.split) for key in bundle.stats},
        allow_absent=True)
    payload = {"schema_version": QUESTION_TABLE_SCHEMA, "pool": bundle.pool, "scope": scope,
               "comparability": block,
               "binding_level": "chains" if bound else None,
               "sources": bound.get("sources") or {},
               "identities": bound.get("identities") or {},
               "restriction": validate_restriction(
                   str(path), restriction, allow_absent=True, pool=bundle.pool, splits=splits),
               "normal_form": bundle.normal_form,
               "cost_convention": "thinking = ladder prefix; generated = thinking + answer "
                                  "(frontier); aux = declared overhead_tokens",
               "qkeys": qkeys, "series": series, "provenance": list(provenance)}
    if head_tag:
        payload["head_tag"] = head_tag
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {"path": str(path), "series": len(series),
            "questions": {s: len(v) for s, v in qkeys.items()},
            "binding_level": payload["binding_level"],
            "restriction": (payload["restriction"] or {}).get("kind"),
            "sources": {seed: {split: entry["sha256"] for split, entry in sorted(by.items())}
                        for seed, by in sorted((bound.get("sources") or {}).items())},
            "chains": {seed: {split: entry["n_chains"] for split, entry in sorted(by.items())}
                       for seed, by in sorted((bound.get("sources") or {}).items())}}


STAMPED_COMPARABILITY = ("pool", "dataset", "grader_id", "model_id", "model_revision")
STAMPED_IDENTITY = STAMPED_COMPARABILITY + ("tokenizer_revision", "prompt_contract_id")
SERIES_IDENTITY_FIELDS = tuple(sorted(
    set(STAMPED_IDENTITY) | set(COMPARABILITY_FIELDS) | set(CHAIN_IDENTITY_FIELDS)))


def load_verified_question_tables(paths, *, root=None, bundle: Bundle | None = None,
                                  registry=None, usage="final") -> Bundle:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if bundle is not None:
        bundle.assert_verified("a merge of verified question tables")
    for path in paths:
        record = verify_self_authentication(path, root=root, registry=registry, usage=usage)
        bundle, origin = _load_one_question_table(
            path, bundle, registry=registry, usage=usage)
        bundle._attach_evidence(_TableEvidence(
            _EVIDENCE_TOKEN, origin=origin, path=str(path),
            sha256=_file_sha256(path), record=record))
    if bundle is None:
        raise MergeError("load_verified_question_tables: no table given")
    return bundle


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_question_tables(path, bundle: Bundle | None = None) -> Bundle:
    bundle, _origin = _load_one_question_table(path, bundle)
    return bundle


def _load_one_question_table(path, bundle: Bundle | None, *, registry=None,
                             usage=None) -> tuple[Bundle, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != QUESTION_TABLE_SCHEMA:
        raise MergeError(f"{path}: schema_version {payload.get('schema_version')!r} != "
                         f"{QUESTION_TABLE_SCHEMA!r}")
    stored = payload.get("normal_form", "chain")
    if bundle is not None and bundle.normal_form != stored:
        raise MergeError(
            f"{path}: export is on the {stored!r} basis but the bundle is "
            f"{bundle.normal_form!r}. Convert explicitly (Bundle.as_question_normal) rather "
            "than relabelling numbers.")
    declared_pool = payload.get("pool", "")
    table_pool = canonical_pool(declared_pool)
    block = canonicalize_identity(dict(payload.get("comparability") or {}))
    series_declare = any(
        (row.get("meta") or {}).get(f) is not None
        for row in payload["series"] for f in SERIES_IDENTITY_FIELDS)
    binds = bool(payload.get("sources")) or bool(payload.get("identities"))
    if not block:
        if not series_declare and not binds:
            raise legacy_refusal(path, len(payload["series"]))
        raise MergeError(
            f"{path}: has NO table-level comparability block, but "
            f"{'its series declare identity fields' if series_declare else ''}"
            f"{' and ' if series_declare and binds else ''}"
            f"{'it declares a chain binding' if binds else ''}. That is a stripped or partial "
            f"header, not a pre-F17 legacy export, and it is refused everywhere: a reader "
            f"cannot see what the file claims without walking every series. {_REEXPORT_HINT}")
    if not is_canonical_pool(table_pool):
        raise MergeError(
            f"{path}: pool {declared_pool!r} is not a canonical chain-bank id "
            f"({sorted(CANONICAL_POOLS)}). A table's pool is the BANK it read, never "
            f"the file it was written to. {_REEXPORT_HINT}")
    missing = [f for f in STAMPED_IDENTITY if block.get(f) is None]
    if missing:
        raise MergeError(
            f"{path}: table-level comparability block is missing {missing}. "
            f"{_REEXPORT_HINT}")
    if block["pool"] != table_pool:
        raise MergeError(f"{path}: comparability pool {block['pool']!r} != table pool "
                         f"{table_pool!r}")
    family = resolve_pool(block["pool"])
    if canonical_pool(block["dataset"]) != family:
        raise MergeError(
            f"{path}: dataset {block['dataset']!r} is not the canonical family of pool "
            f"{block['pool']!r} ({family!r}). The campaign's dataset id IS the chain-bank "
            f"family (contract v1.9 §2.9); one file declares one bank.")
    level = payload.get("binding_level")
    if level != "chains":
        raise MergeError(
            f"{path}: binding_level {level!r} -- a statistic may only be computed from a table "
            f"bound at the CHAIN level, i.e. one that names every (qkey, rep) it read and the "
            f"sha256 of the dump it read them from. A question-level binding pins only the "
            f"question SET, so any run dir over the same questions binds equally well; that is "
            f"not provenance (finding C3). {_REEXPORT_HINT}")
    qkeys = payload["qkeys"]
    covered = {(str(row["seed"]), str(row["split"])) for row in payload["series"]}
    validate_self_authentication(str(path), payload.get("sources") or {},
                                 payload.get("identities") or {}, qkeys, required=covered)
    validate_restriction(str(path), payload.get("restriction"), registry=registry,
                         pool=table_pool, splits=qkeys.keys(), usage=usage)
    if bundle is None:
        bundle = Bundle(pool=table_pool, normal_form=stored)
    origin = f"{uuid.uuid4().hex}:{path}"
    bundle._origin_paths[origin] = str(path)
    for row in payload["series"]:
        keys = qkeys[row["split"]]
        columns = (row["n"], row["correct"], row["thinking"], row["generated"], row["aux"])
        if any(len(c) != len(keys) for c in columns):
            raise MergeError(f"{path}: {row['method']}/{row['member']} column length != "
                             f"{len(keys)} questions")
        per_q = {q: tuple(c[i] for c in columns) for i, q in enumerate(keys)}
        key = SeriesKey(row["method"], row["member"], int(row["seed"]), row["split"])
        meta = canonicalize_identity(dict(row.get("meta") or {}))
        missing = [f for f in STAMPED_IDENTITY if meta.get(f) is None]
        if missing:
            raise MergeError(f"{path}: series {key} does not declare {missing}. "
                             f"{_REEXPORT_HINT}")
        disagree = {f: (meta[f], block[f]) for f in STAMPED_IDENTITY
                    if meta[f] != block[f]}
        if disagree:
            raise MergeError(f"{path}: series {key} disagrees with the table-level "
                             f"comparability block on {disagree}")
        bundle.add(key, per_q, kind=row.get("kind", "plan_rule"), meta=meta)
        bundle._record_origin(key, origin)
    return bundle, origin


def question_table_metadata(path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"path": str(path), "unreadable": str(exc)}
    block = payload.get("comparability") or {}
    series = payload.get("series") or []
    qkeys = payload.get("qkeys") or {}
    missing_series = {f: sum(1 for row in series if not (row.get("meta") or {}).get(f))
                      for f in STAMPED_IDENTITY}
    return {
        "path": str(path),
        "declared_pool": payload.get("pool"),
        "pool_is_canonical": is_canonical_pool(canonical_pool(payload.get("pool"))),
        "schema_version": payload.get("schema_version"),
        "normal_form": payload.get("normal_form", "chain"),
        "scope": payload.get("scope"),
        "binding_level": payload.get("binding_level"),
        "restriction": (payload.get("restriction") or {}).get("kind"),
        "binds_to_chains": bool(payload.get("sources")) and bool(payload.get("identities")),
        "n_series": len(series),
        "n_questions": {split: len(keys) for split, keys in sorted(qkeys.items())},
        "splits": sorted(qkeys),
        "seeds": sorted({int(row["seed"]) for row in series}) if series else [],
        "n_series_declaring_nothing": sum(
            1 for row in series
            if not any((row.get("meta") or {}).get(f) for f in SERIES_IDENTITY_FIELDS)),
        "missing_table_level": [f for f in STAMPED_IDENTITY if not block.get(f)],
        "missing_on_series": {f: n for f, n in missing_series.items() if n},
    }


def diagnose_legacy_question_table(path) -> dict:
    try:
        load_question_tables(path)
    except LegacyStampMissing as exc:
        return {**question_table_metadata(path), "legacy": True, "refusal": str(exc),
                "suggested_command": _REEXPORT_HINT}
    raise MergeError(f"{path}: reads strictly; there is nothing to diagnose")


def load_exchange_rows(paths, bundle: Bundle | None = None, *, split: str | None = None,
                       overhead_per_chain: float = 0.0, units: Sequence[str] = EVALUATION_UNITS,
                       chain_bank_sha256: str | None = None, manifest_sha256: str | None = None,
                       pool: str | None = None, normalize: str | None = None,
                       seed_semantics: str | None = None) -> Bundle:
    from rl_controller.iclr import frontier_exchange as fx

    units = tuple(units)
    unknown_units = [u for u in units if u not in EVALUATION_UNITS]
    if unknown_units:
        raise MergeError(f"unknown evaluation unit(s) {unknown_units}; "
                         f"expected a subset of {EVALUATION_UNITS}")
    declared = {"chain_bank_sha256": chain_bank_sha256, "manifest_sha256": manifest_sha256,
                "pool": pool}
    supplied = {k: v for k, v in declared.items() if v is not None}
    if supplied and len(supplied) != len(declared):
        raise MergeError(
            f"chain identity must be declared all-or-none; got {sorted(supplied)} and not "
            f"{sorted(set(declared) - set(supplied))}. A partial declaration reads as verified "
            "while leaving the bank unpinned.")
    for name in ("chain_bank_sha256", "manifest_sha256"):
        value = supplied.get(name)
        if value is not None and (not isinstance(value, str) or len(value) != 64
                                  or any(c not in "0123456789abcdef" for c in value)):
            raise MergeError(f"{name} must be 64 lowercase hex chars, got {value!r}")
    if "pool" in supplied and (not isinstance(supplied["pool"], str) or not supplied["pool"]):
        raise MergeError("pool must be a non-empty str")
    declared = supplied
    if seed_semantics is not None and seed_semantics not in SEED_SEMANTICS:
        raise MergeError(f"seed_semantics must be one of {SEED_SEMANTICS}, "
                         f"got {seed_semantics!r}")
    if normalize is not None and normalize not in NORMAL_FORMS:
        raise MergeError(f"normalize must be one of {NORMAL_FORMS}, got {normalize!r}")
    target_form = normalize or (bundle.normal_form if bundle is not None else "question")
    if bundle is not None and normalize is not None and bundle.normal_form != normalize:
        raise MergeError(f"bundle is in {bundle.normal_form!r} normal form; cannot load as "
                         f"{normalize!r}")
    bundle = bundle if bundle is not None else Bundle(normal_form=target_form)
    grouped: dict[SeriesKey, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0]))
    meta: dict[SeriesKey, dict] = {}
    seen: set[tuple] = set()
    folded: dict[tuple, bool] = {}
    for path in ([paths] if isinstance(paths, (str, Path)) else list(paths)):
        for row in fx.load_rows(path):
            unit = row["evaluation_unit"]
            if unit not in units:
                raise MergeError(
                    f"{path}: row has evaluation_unit {unit!r}, which this call excludes "
                    f"(units={list(units)}). Pass units=('chain','question') to merge both, "
                    "or split the families into separate calls -- do not mix a chain-unit and a "
                    "question-unit reading of the SAME method, which would double-count it.")
            if row.get("missing_reason"):
                continue
            qkey = row["qkey"]
            if not isinstance(qkey, str) or not qkey:
                raise MergeError(f"{path}: qkey must be a non-empty str, got {qkey!r}")
            method_name = canonical_member(row["method_name"], method=row["method_name"])
            config_id = canonical_member(row["config_id"], method=row["method_name"])
            how = seed_semantics_for(method_name,
                                     row.get("seed_semantics", seed_semantics))
            seed_value = SEED_ANY if how == "replay" else int(row["seed"])
            fold = (row["method_name"], row["config_id"], int(row["seed"]))
            if fold not in folded:
                folded[fold] = True
                if (method_name, config_id, seed_value) != fold:
                    bundle.notes.append({
                        "note": "canonicalised_at_load", "source": str(path),
                        "from": {"method_name": fold[0], "config_id": fold[1], "seed": fold[2]},
                        "to": {"method_name": method_name, "config_id": config_id,
                               "seed": seed_value},
                        "seed_semantics": how})
            key = SeriesKey(method_name, config_id, seed_value, split or row["split"])
            identity = (key, qkey, row["rep"])
            if identity in seen:
                raise MergeError(f"{path}: duplicate row for {key} qkey={row['qkey']} "
                                 f"rep={row['rep']}")
            seen.add(identity)
            entry = grouped[key][qkey]
            entry[0] += 1
            entry[1] += int(bool(row["correct"]))
            entry[2] += int(row["thinking_tokens"])
            entry[3] += int(row["candidate_tokens_generated"])
            entry[4] += int(row["aux_generated_tokens"]) + overhead_per_chain
            row_meta = canonicalize_identity({f: row[f] for f in COMPARABILITY_FIELDS
                                              if f in row})
            row_meta["evaluation_unit"] = unit
            row_meta["sample_count"] = int(row["sample_count"])
            row_meta["seed_semantics"] = how
            if key not in meta:
                meta[key] = dict(row_meta)
                if declared:
                    meta[key].update(canonicalize_identity(dict(declared)))
                    meta[key]["identity_declared_by"] = "loader"
            else:
                drift = {f: (meta[key][f], v) for f, v in row_meta.items()
                         if f in meta[key] and meta[key][f] != v}
                if drift:
                    raise MergeError(
                        f"{path}: {key} qkey={qkey} disagrees with earlier rows of the same "
                        f"series on {sorted(drift)}: {drift}. A series is one method read one "
                        "way; differing rows are two methods.")
    loaded_units = {m["evaluation_unit"] for m in meta.values()}
    if len(loaded_units) > 1 and target_form != "question":
        raise MergeError(
            f"these rows mix evaluation units {sorted(loaded_units)}, which is only defined "
            "under the QUESTION normal form (a voted SC@N answer has no chain weight to "
            "match). Pass normalize='question'. Note that changes the estimand relative to "
            "every archived table -- on the math union bank `natural` reads 0.5092975207 "
            "chain-weighted and 0.5146713615 per-question.")
    for key, per_q in grouped.items():
        stats = ({q: (1.0, c / n, t / n, g / n, a / n)
                  for q, (n, c, t, g, a) in ((q, tuple(v)) for q, v in per_q.items())}
                 if target_form == "question" else per_q)
        bundle.add(key, stats, kind="plan_rule", meta=meta[key])
    return bundle


def validate_policy_replay_row(row: Mapping) -> dict:
    if not isinstance(row, Mapping):
        raise MergeError("policy-replay row must be an object")
    extra = sorted(set(row) - POLICY_REPLAY_FIELDS)
    missing = sorted(POLICY_REPLAY_FIELDS - set(row))
    if extra or missing:
        raise MergeError(f"policy-replay row field mismatch: missing {missing}, unknown {extra}")
    if row["schema_version"] != POLICY_REPLAY_SCHEMA:
        raise MergeError(f"schema_version {row['schema_version']!r} != {POLICY_REPLAY_SCHEMA!r}")
    if row["expectation"] not in EXPECTATIONS:
        raise MergeError(f"expectation must be one of {EXPECTATIONS}, got {row['expectation']!r}")
    if not isinstance(row["qkey"], str) or not row["qkey"]:
        raise MergeError("qkey must be a non-empty str")
    for name in ("chain_bank_sha256", "manifest_sha256"):
        value = row[name]
        if not isinstance(value, str) or len(value) != 64 or \
                any(c not in "0123456789abcdef" for c in value):
            raise MergeError(f"{name} must be 64 lowercase hex chars (it is what proves this "
                             f"method read the same chains as the shared anchors), got {value!r}")
    if not isinstance(row["pool"], str) or not row["pool"]:
        raise MergeError("pool must be a non-empty str")
    if isinstance(row["seed"], bool) or not isinstance(row["seed"], int):
        raise MergeError("seed must be an int")
    if row.get("missing_reason"):
        blanks = ["n_chains", "e_correct", "e_thinking_tokens", "e_answer_tokens",
                  "e_generated_tokens", "aux_generated_tokens"]
        nonnull = [f for f in blanks if row[f] is not None]
        if nonnull:
            raise MergeError(f"missing_reason set but {nonnull} are not null")
        return dict(row)
    n = row["n_chains"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise MergeError(f"n_chains must be an int >= 1, got {n!r}")
    if not 0.0 <= float(row["e_correct"]) <= 1.0:
        raise MergeError(f"e_correct {row['e_correct']} outside [0, 1]")
    for name in ("e_thinking_tokens", "e_answer_tokens", "e_generated_tokens",
                 "aux_generated_tokens"):
        value = float(row[name])
        if not math.isfinite(value) or value < 0:
            raise MergeError(f"{name} must be finite and >= 0, got {row[name]!r}")
    total = float(row["e_thinking_tokens"]) + float(row["e_answer_tokens"])
    if abs(total - float(row["e_generated_tokens"])) > 1e-6:
        raise MergeError(f"e_generated_tokens {row['e_generated_tokens']} != e_thinking_tokens + "
                         f"e_answer_tokens = {total}")
    return dict(row)


def load_policy_replay(paths, bundle: Bundle | None = None, *,
                       split: str | None = None, strict: bool = True,
                       normalize: str | None = None,
                       seed_semantics: str | None = None) -> Bundle:
    if seed_semantics is not None and seed_semantics not in SEED_SEMANTICS:
        raise MergeError(f"seed_semantics must be one of {SEED_SEMANTICS}, "
                         f"got {seed_semantics!r}")
    if normalize is not None and normalize not in NORMAL_FORMS:
        raise MergeError(f"normalize must be one of {NORMAL_FORMS}, got {normalize!r}")
    target_form = normalize or (bundle.normal_form if bundle is not None else "chain")
    if bundle is not None and normalize is not None and bundle.normal_form != normalize:
        raise MergeError(f"bundle is in {bundle.normal_form!r} normal form; cannot load as "
                         f"{normalize!r}")
    bundle = bundle if bundle is not None else Bundle(normal_form=target_form)
    grouped: dict[SeriesKey, dict[str, tuple]] = defaultdict(dict)
    meta: dict[SeriesKey, dict] = {}
    unavailable: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in ([paths] if isinstance(paths, (str, Path)) else list(paths)):
        for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = validate_policy_replay_row(json.loads(line))
            except MergeError as exc:
                raise MergeError(f"{path}:{lineno}: {exc}") from None
            if row.get("missing_reason"):
                unavailable[(row["method_name"], row["config_id"])].append(
                    str(row["missing_reason"]))
                continue
            method_name = canonical_member(row["method_name"], method=row["method_name"])
            config_id = canonical_member(row["config_id"], method=row["method_name"])
            how = seed_semantics_for(method_name, seed_semantics)
            key = SeriesKey(method_name, config_id,
                            SEED_ANY if how == "replay" else int(row["seed"]),
                            split or row["split"])
            if row["qkey"] in grouped[key]:
                raise MergeError(f"{path}:{lineno}: duplicate row for {key} qkey={row['qkey']}")
            n = float(row["n_chains"])
            grouped[key][row["qkey"]] = (
                n, float(row["e_correct"]) * n, float(row["e_thinking_tokens"]) * n,
                float(row["e_generated_tokens"]) * n, float(row["aux_generated_tokens"]) * n)
            meta.setdefault(key, canonicalize_identity(
                {**{f: row[f] for f in COMPARABILITY_FIELDS if f in row},
                 "expectation": row["expectation"], "knob": row["knob"]}))
    for (method, config), reasons in unavailable.items():
        bundle.unavailable[f"{method}/{config}"] = {
            "reason": sorted(set(reasons))[0], "rows": len(reasons)}
        for key in [k for k in grouped if k.method == method and k.member == config]:
            del grouped[key]
    if strict:
        _check_policy_rectangle(grouped)
    for key, per_q in grouped.items():
        stats = ({q: (1.0, c / n, t / n, g / n, a / n) for q, (n, c, t, g, a) in per_q.items()}
                 if target_form == "question" else per_q)
        bundle.add(key, stats, kind="policy_replay", meta=meta[key])
    return bundle


def _check_policy_rectangle(grouped: Mapping[SeriesKey, Mapping[str, tuple]]) -> None:
    if not grouped:
        return
    splits = sorted({k.split for k in grouped})
    by_split_qkeys = {s: {q for k, v in grouped.items() if k.split == s for q in v}
                      for s in splits}
    configs = sorted({(k.method, k.member) for k in grouped})
    seeds = sorted({k.seed for k in grouped})
    holes = []
    for method, member in configs:
        for seed in seeds:
            for split in splits:
                per_q = grouped.get(SeriesKey(method, member, seed, split))
                if per_q is None:
                    holes.append(f"{method}/{member}/s{seed}/{split}: absent")
                elif set(per_q) != by_split_qkeys[split]:
                    missing = len(by_split_qkeys[split] - set(per_q))
                    extra = len(set(per_q) - by_split_qkeys[split])
                    holes.append(f"{method}/{member}/s{seed}/{split}: "
                                 f"{missing} missing / {extra} extra question(s)")
    if holes:
        raise MergeError(
            "policy-replay coverage is not rectangular over (split, config, seed, qkey) -- "
            f"{len(holes)} hole(s), e.g. {holes[:4]}. Honest selection happens on val, so a "
            "config missing there has no operating point; report it unavailable via "
            "missing_reason instead of omitting the rows.")


_CELL_CONTEXT: dict | None = None


def _summary_cell(bundle: Bundle, *, method: str, cost: str, **kwargs) -> dict:
    try:
        return abh_with_ci(bundle, method=method, cost=cost, **kwargs)
    except MergeError as exc:
        return {"status": "unavailable", "reason": str(exc)}


def _task_worker(task):
    index, payload = task
    ctx = _CELL_CONTEXT
    try:
        return index, True, ctx["run"](ctx["context"], payload)
    except BaseException:
        return index, False, traceback.format_exc()


def run_tasks_parallel(context, payloads, *, jobs: int, run, failure,
                       initializer=None) -> list | None:
    global _CELL_CONTEXT
    payloads = list(payloads)
    try:
        ctx = multiprocessing.get_context("fork")
    except ValueError:
        warnings.warn(
            "no `fork` start method on this platform, so a worker would have to pickle the "
            "whole bundle per task; running serially instead",
            RuntimeWarning, stacklevel=3)
        return None
    _CELL_CONTEXT = {"context": context, "run": run}
    try:
        with ctx.Pool(processes=min(jobs, len(payloads)), initializer=initializer) as pool:
            results = list(pool.imap_unordered(
                _task_worker, list(enumerate(payloads)), chunksize=1))
    finally:
        _CELL_CONTEXT = None
    entries: list = [None] * len(payloads)
    for index, ok, payload in sorted(results, key=lambda r: r[0]):
        if not ok:
            raise RuntimeError(failure(payloads[index]) + "\n" + payload)
        entries[index] = payload
    return entries


def _run_cells_parallel(bundle: Bundle, cells, *, jobs: int, kwargs: dict) -> list | None:
    cells = list(cells)
    return run_tasks_parallel(
        {"bundle": bundle, "kwargs": kwargs}, cells, jobs=jobs,
        run=lambda ctx, cell: _summary_cell(ctx["bundle"], method=cell[0], cost=cell[1],
                                            **ctx["kwargs"]),
        failure=lambda cell: (
            f"the summary cell (method={cell[0]!r}, cost={cell[1]!r}) failed in a worker. A "
            "pool that swallowed this would return a summary silently missing that row:"))


def summarize(bundle: Bundle, *, split: str, reference: str, methods: Sequence[str] | None = None,
              n_boot: int = DEFAULT_BOOTSTRAP, seed: int = 0, val_split: str = "val",
              cost_axes: Sequence[str] = COST_AXES, honest: bool = True,
              basis: str | None = None, jobs: int = 1) -> dict:
    if not isinstance(jobs, int) or isinstance(jobs, bool) or jobs < 1:
        raise ValueError(f"jobs must be an int >= 1, got {jobs!r}")
    bundle.assert_verified("a cross-method summary")
    comparability = bundle.check_comparability()
    bundle.assert_weighting(split, val_split)
    weighting = bundle.check_weighting(split)
    qkeys = bundle.qkeys(split)
    plan = bootstrap_plan(qkeys, n=n_boot, seed=seed) if n_boot else []
    candidates = [m for m in (methods or bundle.methods(split))
                  if not is_free_rule(m) and m != reference]
    out: dict = {"schema_version": SUMMARY_SCHEMA, "pool": bundle.pool, "split": split,
                 "reference": reference, "hull": "honest" if honest else "sweep",
                 "n_questions": len(qkeys), "bootstrap": {
                     "replicates": n_boot, "seed": seed, "resampling_unit": "qkey",
                     "plan": "one shared plan across every method, config, seed and anchor",
                     "confidence_interval": "95% percentile"},
                 "comparability": comparability,
                 "comparability_strict": True,
                 "weighting": weighting,
                 "normal_form": basis or bundle.normal_form,
                 "basis_banner": (
                     "BASIS: question (per-question mean)."
                     if bundle.normal_form == "question" else
                     "BASIS: chain (sum(correct)/sum(chains)); archive-parity view."),
                 "estimand": ("mean per-question accuracy" if bundle.normal_form == "question"
                              else "chain-weighted accuracy, sum(correct)/sum(chains) -- the "
                                   "ARCHIVED basis; equals the per-question mean only on a "
                                   "rectangular bank"),
                 "identity_declared": not comparability["identity_undeclared"],
                 "cost_definition": {
                     "thinking": "ladder convention: prefix thinking tokens at the stop",
                     "total": "frontier convention: generated = thinking + answer, plus the "
                              "method's declared overhead_tokens"},
                 "seed_convention": "trained methods: 2 seeds, verdict needs the same sign on "
                                    "both; training-free: deterministic replay, verdict on the "
                                    "paired CI alone; SE is conditional on the observed seeds",
                 "methods": {}}
    cells = [(method, cost) for method in candidates for cost in cost_axes]
    cell_kwargs = {"reference": reference, "split": split, "plan": plan,
                   "val_split": val_split, "honest": honest, "basis": basis}
    entries = None
    if jobs > 1 and cells:
        entries = _run_cells_parallel(bundle, cells, jobs=jobs, kwargs=cell_kwargs)
    if entries is None:
        entries = [_summary_cell(bundle, method=method, cost=cost, **cell_kwargs)
                   for method, cost in cells]
    for method in candidates:
        out["methods"][method] = {}
    for (method, cost), value in zip(cells, entries):
        out["methods"][method][cost] = value
    return out


