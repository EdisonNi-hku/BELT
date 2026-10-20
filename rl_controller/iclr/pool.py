from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping

from rl_controller.splits import DEFAULT_SHARES, SPLITS, assign_split, question_key, three_way_split

__all__ = [
    "NGramIndex",
    "build_pool",
    "contract",
    "excluded_by_hash_rule",
    "has_figure",
    "normalise_text",
    "parse_integer_answer",
    "rank",
    "word_tokens",
]

CONTRACT_VERSION = "iclr-p1-math-pool-2026-08-26-v1"
NGRAM_N = 13

_NON_TOKEN_RE = re.compile(r"[^\w+\-*/=^{}()]+")
_WS_RE = re.compile(r"\s+")
_INT_RE = re.compile(r"\A-?\d+\Z")
_BOXED_RE = re.compile(r"\A\\boxed\{(.*)\}\Z", re.DOTALL)
_FIGURE_MARKERS = ("[asy]", "\\includegraphics", "<img")


_DASHES = {"\u2212": "-", "\u2013": "-"}


def normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for dash, plain in _DASHES.items():
        text = text.replace(dash, plain)
    return _WS_RE.sub(" ", text).strip().lower()


def word_tokens(text: str) -> list[str]:
    return _NON_TOKEN_RE.sub(" ", text).split()


def parse_integer_answer(raw) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw
    for _ in range(8):
        before = text
        text = text.strip().strip("$").strip()
        text = text.rstrip(".。").strip()
        m = _BOXED_RE.match(text)
        if m:
            text = m.group(1).strip()
        if text == before:
            break
    if _INT_RE.match(text):
        return int(text)
    return None


def has_figure(question: str) -> bool:
    lowered = question.lower()
    return any(marker in lowered for marker in _FIGURE_MARKERS)


class NGramIndex:

    def __init__(self, n: int = NGRAM_N):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self._grams: dict[tuple[str, ...], str] = {}
        self._short_lengths: set[int] = set()
        self._n_texts = 0

    def __len__(self) -> int:
        return self._n_texts

    @property
    def n_grams(self) -> int:
        return len(self._grams)

    def _grams_of(self, text: str) -> list[tuple[str, ...]]:
        tokens = word_tokens(normalise_text(text))
        if not tokens:
            return []
        if len(tokens) < self.n:
            return [("<whole>", *tokens)]
        return [tuple(tokens[i:i + self.n]) for i in range(len(tokens) - self.n + 1)]

    def add(self, text: str, tag: str) -> None:
        self._n_texts += 1
        tokens = word_tokens(normalise_text(text))
        if tokens and len(tokens) < self.n:
            self._short_lengths.add(len(tokens))
        for gram in self._grams_of(text):
            self._grams.setdefault(gram, tag)

    def overlaps(self, text: str) -> str | None:
        tokens = word_tokens(normalise_text(text))
        if len(tokens) >= self.n:
            for i in range(len(tokens) - self.n + 1):
                tag = self._grams.get(tuple(tokens[i:i + self.n]))
                if tag is not None:
                    return tag
        for length in sorted(self._short_lengths):
            if length > len(tokens):
                continue
            for i in range(len(tokens) - length + 1):
                tag = self._grams.get(("<whole>", *tokens[i:i + length]))
                if tag is not None:
                    return tag
        return None


def excluded_by_hash_rule(qkey: str, rule: Mapping) -> bool:
    shares = tuple(rule.get("shares", DEFAULT_SHARES))
    return assign_split(qkey, salt=str(rule["salt"]), shares=shares) in set(rule["exclude_splits"])


def rank(salt: str, purpose: str, qkey: str) -> str:
    return hashlib.sha256(f"{salt}:{purpose}:{qkey}".encode()).hexdigest()


def _new_source_report(quota: int) -> dict:
    return {
        "candidates": 0,
        "dropped_duplicate": 0,
        "dropped_empty": 0,
        "dropped_too_long": 0,
        "dropped_figure": 0,
        "dropped_excluded": {"qkey": 0, "sha256": 0, "hash_rule": 0},
        "dropped_decontam": {},
        "eligible": 0,
        "quota_requested": int(quota),
        "quota_filled": 0,
    }


def build_pool(
    candidates: Iterable[Mapping],
    *,
    quotas: Mapping[str, int],
    exclusions: Mapping,
    eval_index: NGramIndex,
    eval_exact: Mapping[str, str],
    salt: str,
    shares: tuple = (0.6, 0.15, 0.25),
    allocation_fraction: float = 0.25,
    base_chains: int = 8,
    allocation_chains: int = 16,
    tranche: int = 1,
    max_question_chars: int = 2000,
    rank_fn=None,
) -> tuple[list[dict], dict]:
    if rank_fn is None:
        rank_fn = lambda qkey: rank(salt, "pool", qkey)
    source_order = {source: i for i, source in enumerate(quotas)}
    ordered = sorted(candidates, key=lambda c: source_order.get(str(c["source"]), len(source_order)))
    excluded_qkeys = set(exclusions.get("qkeys", []))
    excluded_shas = set(exclusions.get("question_sha256s", []))
    hash_rules = list(exclusions.get("hash_rules", []))

    per_source = {source: _new_source_report(quotas[source]) for source in quotas}
    decontam_hits: list[dict] = []
    seen: set[str] = set()
    eligible: dict[str, list[dict]] = {source: [] for source in quotas}

    for cand in ordered:
        source = str(cand["source"])
        if source not in per_source:
            raise ValueError(f"candidate source {source!r} has no quota (quotas={list(quotas)})")
        counts = per_source[source]
        counts["candidates"] += 1
        question = str(cand["question"])
        gold = cand["gold"]
        if isinstance(gold, bool) or not isinstance(gold, int):
            raise ValueError(f"gold must be an int after integer filtering, got {gold!r} ({source}/{cand.get('source_id')})")
        qkey = question_key(question)
        if qkey in seen:
            counts["dropped_duplicate"] += 1
            continue
        seen.add(qkey)
        if not question.strip():
            counts["dropped_empty"] += 1
            continue
        if len(question) > max_question_chars:
            counts["dropped_too_long"] += 1
            continue
        if has_figure(question):
            counts["dropped_figure"] += 1
            continue
        raw_sha = hashlib.sha256(question.encode()).hexdigest()
        if qkey in excluded_qkeys:
            counts["dropped_excluded"]["qkey"] += 1
            continue
        if raw_sha in excluded_shas:
            counts["dropped_excluded"]["sha256"] += 1
            continue
        if any(excluded_by_hash_rule(qkey, rule) for rule in hash_rules):
            counts["dropped_excluded"]["hash_rule"] += 1
            continue
        matched = eval_exact.get(qkey)
        if matched is None:
            matched = eval_index.overlaps(question)
        if matched is not None:
            dd = counts["dropped_decontam"]
            dd[matched] = dd.get(matched, 0) + 1
            decontam_hits.append({"qkey": qkey, "source": source, "matched_eval_set": matched})
            continue
        counts["eligible"] += 1
        eligible[source].append({
            "qkey": qkey,
            "question": question,
            "question_sha256": raw_sha,
            "gold": int(gold),
            "source": source,
            "source_id": cand.get("source_id"),
            "level": cand.get("level"),
        })

    selected: list[dict] = []
    for source, quota in quotas.items():
        picked = sorted(eligible[source], key=lambda r: rank_fn(r["qkey"]))[: int(quota)]
        per_source[source]["quota_filled"] = len(picked)
        selected.extend(picked)

    shares = tuple(shares)
    splits = three_way_split((r["qkey"] for r in selected), salt=salt, shares=shares) if selected else {}
    allocation: set[str] = set()
    split_counts = {name: 0 for name in SPLITS}
    allocation_counts = {name: 0 for name in SPLITS}
    for name in SPLITS:
        keys = [r["qkey"] for r in selected if splits[r["qkey"]] == name]
        split_counts[name] = len(keys)
        n_alloc = round(allocation_fraction * len(keys))
        chosen = sorted(keys, key=lambda k: rank(salt, "allocation", k))[:n_alloc]
        allocation.update(chosen)
        allocation_counts[name] = len(chosen)

    rows = []
    for r in sorted(selected, key=lambda r: r["qkey"]):
        is_alloc = r["qkey"] in allocation
        rows.append({
            "qkey": r["qkey"],
            "question": r["question"],
            "question_sha256": r["question_sha256"],
            "gold": r["gold"],
            "split": splits[r["qkey"]],
            "target_chains": int(allocation_chains if is_alloc else base_chains),
            "allocation_subset": is_alloc,
            "source": r["source"],
            "source_id": r["source_id"],
            "level": r["level"],
            "tranche": int(tranche),
        })

    report = {
        "salt": salt,
        "shares": list(shares),
        "allocation_fraction": allocation_fraction,
        "base_chains": int(base_chains),
        "allocation_chains": int(allocation_chains),
        "tranche": int(tranche),
        "max_question_chars": int(max_question_chars),
        "hash_rule_scope": "all sources",
        "sources": per_source,
        "pool_size": len(rows),
        "split_counts": split_counts,
        "allocation_counts": allocation_counts,
        "total_chains": sum(r["target_chains"] for r in rows),
        "decontam_hits": decontam_hits,
    }
    return rows, report


def contract(
    report: Mapping,
    *,
    manifest_path: str,
    manifest_sha256: str,
    source_records: list,
    eval_records: list,
    exclusions_sha256: str,
    salt: str,
    shares: tuple,
    quotas: Mapping[str, int],
    gpqa_status: str,
) -> dict:
    per_eval: dict[str, int] = {}
    for src in report["sources"].values():
        for name, n in src["dropped_decontam"].items():
            per_eval[name] = per_eval.get(name, 0) + n
    body = {key: value for key, value in report.items() if key != "decontam_hits"}
    return {
        "contract_version": CONTRACT_VERSION,
        "status": "FROZEN_BEFORE_GENERATION",
        "report_generated": False,
        "salt": salt,
        "shares": list(shares),
        "quotas": dict(quotas),
        "tranche": int(report["tranche"]),
        "pool_size": int(report["pool_size"]),
        "split_counts": dict(report["split_counts"]),
        "allocation_split_counts": dict(report["allocation_counts"]),
        "base_chains_per_question": int(report["base_chains"]),
        "allocation_chains_per_question": int(report["allocation_chains"]),
        "allocation_subset_size": sum(report["allocation_counts"].values()),
        "total_chains_all_splits": int(report["total_chains"]),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "exclusions_sha256": exclusions_sha256,
        "exclusion_hash_rule_scope": "all sources",
        "sources": list(source_records),
        "eval_sets": list(eval_records),
        "decontamination": {
            "ngram_n": NGRAM_N,
            "n_hits": len(report["decontam_hits"]),
            "hits_per_eval_set": dict(sorted(per_eval.items())),
            "gpqa_diamond": gpqa_status,
        },
        "report": body,
    }
