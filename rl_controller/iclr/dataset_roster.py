from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCHEMA = "iclr-dataset-roster/v1"

ANSWER_FILTER = ("strict integer gold (`parse_integer_answer`): a benchmark question whose "
                 "reference answer does not parse to an integer is not in the pool at all, so "
                 "every row below is a FILTERED SUBSET of the benchmark it names and its N is "
                 "not that benchmark's size")


class RosterError(ValueError):
    pass


ORIGIN_MANIFESTS: dict[str, dict] = {
    "math_t1_report": {
        "tranche": 1,
        "parent": {
            "path": "docs/superpowers/contracts/2026-08-26-iclr-p1-math-manifest.jsonl",
            "sha256": "dda318505e56cde15d36c3f034b5445e7d6afdd4baab9401a5713603ad40c1f1",
            "n_rows": 3500},
        "read_split": {
            "path": "/cluster/project/sachan/njingwei/iclr/read_splits/"
                    "math_t1_report.manifest.jsonl",
            "sha256": "e3f79a10e169cc1afbb01678410c80eb5ea5097535d74c108fe513788ca98e2c",
            "n_rows": 841},
    },
    "math_t2_report": {
        "tranche": 2,
        "parent": {
            "path": "docs/superpowers/contracts/2026-08-26-iclr-p1-math-t2-manifest.jsonl",
            "sha256": "ec0e6cb4ef66061ba851cc9ab3e1c75858e7b07f56d88ec28694f1915d539e97",
            "n_rows": 3500},
        "read_split": {
            "path": "/cluster/project/sachan/njingwei/iclr/read_splits/"
                    "math_t2_report.manifest.jsonl",
            "sha256": "f5633be498641349c6c989855f12f49babb15c6c3585b6c0be4280047479007d",
            "n_rows": 853},
    },
    "math_ood_report": {
        "tranche": 0,
        "parent": {
            "path": "docs/superpowers/contracts/2026-08-27-iclr-ood-manifest.jsonl",
            "sha256": "27f2165a204ddaa41459cd39c42bdbd296bc5a551803bd88b66857ca73e63749",
            "n_rows": 951},
        "read_split": {
            "path": "docs/superpowers/contracts/2026-08-27-iclr-ood-manifest.jsonl",
            "sha256": "27f2165a204ddaa41459cd39c42bdbd296bc5a551803bd88b66857ca73e63749",
            "n_rows": 951},
    },
    "ood2_math_report": {
        "tranche": 2,
        "parent": {
            "path": "docs/superpowers/contracts/2026-09-01-iclr-bl-ood2-math-manifest.jsonl",
            "sha256": "e611c5a4ebd85b6ebba16e3f06358e450e2e8e377b4efbdb8f5c48bf41573a75",
            "n_rows": 2430},
        "read_split": {
            "path": "docs/superpowers/contracts/2026-09-01-iclr-bl-ood2-math-manifest.jsonl",
            "sha256": "e611c5a4ebd85b6ebba16e3f06358e450e2e8e377b4efbdb8f5c48bf41573a75",
            "n_rows": 2430},
    },
}

REPORT_SPLIT = "report"
NON_REPORT_SPLITS = ("fit", "select")


@dataclass(frozen=True)
class DatasetRule:

    dataset_id: str
    display_name: str
    benchmark: str
    members: tuple[tuple[str, str], ...]
    expected_n: int
    filter_note: str
    keep_fields: tuple[str, ...] = ()
    year_resolved: bool = False
    no_pooling_with: tuple[tuple[str, str], ...] = ()
    note: str = ""

    @property
    def pools(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(pool for pool, _ in self.members))

    @property
    def subset_label(self) -> str:
        return f"{self.display_name} (filtered subset)"


DATASET_RULES: tuple[DatasetRule, ...] = (
    DatasetRule(
        dataset_id="dapo", display_name="DAPO", benchmark="DAPO-Math-17k",
        members=(("math_t1_report", "dapo"), ("math_t2_report", "dapo")),
        expected_n=740,
        filter_note=ANSWER_FILTER,
        note="one source, two report tranches of it; t1 and t2 are allocation tranches, "
             "not two datasets"),
    DatasetRule(
        dataset_id="numina_hard", display_name="Numina-hard", benchmark="NuminaMath (hard split)",
        members=(("math_t1_report", "numina_hard"), ("math_t2_report", "numina_hard")),
        expected_n=473,
        filter_note=ANSWER_FILTER,
        keep_fields=("level",),
        note="`level` carries Numina's own subject label and is kept per question"),
    DatasetRule(
        dataset_id="deepscaler", display_name="DeepScaleR", benchmark="DeepScaleR-Preview",
        members=(("math_t1_report", "deepscaler"), ("math_t2_report", "deepscaler")),
        expected_n=481,
        filter_note=ANSWER_FILTER),
    DatasetRule(
        dataset_id="math500", display_name="MATH-500", benchmark="MATH-500",
        members=(("math_ood_report", "math500"),),
        expected_n=295,
        filter_note=ANSWER_FILTER,
        keep_fields=("level",),
        note="the MATH-500 test subset; `level` is its subject/level label"),
    DatasetRule(
        dataset_id="olympiadbench", display_name="OlympiadBench", benchmark="OlympiadBench",
        members=(("math_ood_report", "olympiadbench"),),
        expected_n=366,
        filter_note=ANSWER_FILTER,
        keep_fields=("level",)),
    DatasetRule(
        dataset_id="gaokao23en", display_name="Gaokao2023-Math-En",
        benchmark="Gaokao2023 Math (English)",
        members=(("math_ood_report", "gaokao23en"),),
        expected_n=191,
        filter_note=ANSWER_FILTER),
    DatasetRule(
        dataset_id="aime24", display_name="AIME 2024", benchmark="AIME 2024",
        members=(("math_ood_report", "aime24"),),
        expected_n=29,
        filter_note=ANSWER_FILTER,
        year_resolved=True,
        no_pooling_with=(("aime_extra", "nine of these problems appear again in the ood2 AIME "
                                        "extra batch under different qkeys and a different LaTeX "
                                        "rendering; the two rows are not independent samples"),),
        note="`source_id` is the contest id (`2024-I-3`), so the year is recorded, not guessed"),
    DatasetRule(
        dataset_id="aime25", display_name="AIME 2025", benchmark="AIME 2025",
        members=(("math_ood_report", "aime25"),),
        expected_n=30,
        filter_note=ANSWER_FILTER,
        year_resolved=True,
        keep_fields=("level",)),
    DatasetRule(
        dataset_id="amc23", display_name="AMC 2023", benchmark="AMC 2023",
        members=(("math_ood_report", "amc23"),),
        expected_n=40,
        filter_note=ANSWER_FILTER,
        year_resolved=True),
    DatasetRule(
        dataset_id="math_test_rest", display_name="MATH-test-rest",
        benchmark="MATH test split, excluding the MATH-500 subset",
        members=(("ood2_math_report", "math_test_rest"),),
        expected_n=2347,
        filter_note=ANSWER_FILTER,
        keep_fields=("level",),
        note="the MATH test questions that are NOT in MATH-500. It is not MATH-500 and may "
             "never be labelled as it"),
    DatasetRule(
        dataset_id="aime_extra", display_name="AIME extra batch",
        benchmark="AIME (year not resolved by a recorded rule)",
        members=(("ood2_math_report", "aime_val"),),
        expected_n=48,
        filter_note=ANSWER_FILTER,
        keep_fields=("level",),
        year_resolved=False,
        no_pooling_with=(("aime24", "nine of these problems are the same AIME 2024 problems as "
                                    "the math_ood AIME 2024 row, under different qkeys"),),
        note="the manifest's `level` is an AoPS url that NAMES a year, but no pre-recorded rule "
             "turns that into a dataset split, so this stays one batch; the evidence is "
             "recorded in `contest_identity` for a later, deliberate decision"),
    DatasetRule(
        dataset_id="amc_extra", display_name="AMC extra batch",
        benchmark="AMC (year not resolved by a recorded rule)",
        members=(("ood2_math_report", "amc_val"),),
        expected_n=35,
        filter_note=ANSWER_FILTER,
        keep_fields=("level",),
        year_resolved=False,
        note="same as the AIME extra batch: the AoPS url names a year, no recorded rule "
             "consumes it"),
)

FORBIDDEN_NAMES: dict[str, tuple[str, ...]] = {
    "math_test_rest": ("MATH-500", "MATH500", "MATH-500 subset"),
}

COMMON_FIELDS = ("qkey", "question_sha256", "source", "source_id", "split", "target_chains",
                 "allocation_subset")


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qkeys_sha256(qkeys: Iterable[str]) -> str:
    from rl_controller.iclr.read_splits import qkeys_sha256 as convention

    return convention(qkeys)


def canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def payload_sha256(payload) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def read_manifest_rows(path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RosterError(f"{path}:{lineno}: not JSON ({exc}). A manifest this roster "
                                  "cannot parse is a population it cannot state.") from None
    return rows


def bind_manifest(pool: str, path) -> dict:
    if pool not in ORIGIN_MANIFESTS:
        raise RosterError(
            f"{pool!r} is not one of the campaign's report pools ({sorted(ORIGIN_MANIFESTS)}). "
            "The roster is defined over those four populations; a fifth would need its own "
            "registered manifest and its own dataset rules before it could be grouped.")
    declared = ORIGIN_MANIFESTS[pool]
    found = file_sha256(path)
    for role in ("read_split", "parent"):
        if declared[role]["sha256"] == found:
            return {"pool": pool, "role": role, "sha256": found,
                    "declared_path": declared[role]["path"],
                    "declared_n_rows": declared[role]["n_rows"],
                    "identified_by": "sha256, not filename"}
    raise RosterError(
        f"{path} has sha256 {found[:16]}..., which is neither {pool}'s registered parent "
        f"manifest ({declared['parent']['sha256'][:16]}..., {declared['parent']['n_rows']} rows) "
        f"nor its registered read split ({declared['read_split']['sha256'][:16]}..., "
        f"{declared['read_split']['n_rows']} rows). Refusing to group questions from a file "
        "nobody registered: the dataset rows would still come out at plausible sizes while "
        "naming a different population than the one every published number was computed on.")


def report_rows(pool: str, path) -> tuple[list[dict], dict]:
    binding = bind_manifest(pool, path)
    rows = read_manifest_rows(path)
    if len(rows) != binding["declared_n_rows"]:
        raise RosterError(
            f"{path}: {len(rows)} rows, but the registered {binding['role']} for {pool!r} has "
            f"{binding['declared_n_rows']}. The digest matched, so this is a reading fault, not "
            "a substitution -- refusing rather than grouping a partly-read manifest.")
    if binding["role"] == "parent":
        kept = [r for r in rows if r.get("split") == REPORT_SPLIT]
        binding = {**binding, "narrowed_to_split": REPORT_SPLIT,
                   "n_rows_before_narrowing": len(rows), "n_rows": len(kept)}
        return kept, binding
    return rows, {**binding, "narrowed_to_split": None, "n_rows": len(rows)}


def check_question_hashes(pool: str, rows: Sequence[Mapping]) -> int:
    bad = []
    for row in rows:
        text, declared = row.get("question"), row.get("question_sha256")
        if not isinstance(text, str) or not isinstance(declared, str):
            bad.append((row.get("qkey"), "missing question / question_sha256"))
            continue
        found = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if found != declared:
            bad.append((row.get("qkey"), f"{found[:16]}... != {declared[:16]}..."))
    if bad:
        raise RosterError(
            f"{pool}: {len(bad)} row(s) whose declared question_sha256 is not the sha256 of "
            f"their own question text, e.g. {bad[:3]}. The roster's claim is about QUESTIONS; "
            "a row whose hash does not match its text cannot be traced back to the registry it "
            "came from, and grouping it would attach a real dataset name to an unverified item.")
    return len(rows)


def check_registered_population(pool: str, rows: Sequence[Mapping]) -> dict:
    from rl_controller.iclr import read_splits as rs

    qkeys = [row["qkey"] for row in rows]
    digest = qkeys_sha256(qkeys)
    registered = rs.registered_read_split(pool)
    if (len(qkeys), digest) != (registered["n_questions"], registered["qkeys_sha256"]):
        raise RosterError(
            f"{pool}: the rows this roster would group are {len(qkeys)} questions "
            f"({digest[:16]}...), but {rs.REGISTRY_ID} registers {registered['n_questions']} "
            f"({registered['qkeys_sha256'][:16]}...). The dataset rows are slices OF the "
            "campaign's population; a slice of some other population cannot be read beside the "
            "pool-level tables even when the counts agree.")
    return {"pool": pool, "n_questions": len(qkeys), "qkeys_sha256": digest,
            "registry": rs.REGISTRY_ID, "matches_registry": True}


def check_split_isolation(pool: str, rows: Sequence[Mapping], parent_rows=None) -> dict:
    wrong = sorted({str(r.get("split")) for r in rows} - {REPORT_SPLIT})
    if wrong:
        raise RosterError(
            f"{pool}: read-split rows carry split label(s) {wrong}, not {REPORT_SPLIT!r}. A "
            "fit or select row in a report denominator is a training question scored as "
            "evaluation, and nothing in the resulting number says so.")
    out = {"pool": pool, "report_rows": len(rows), "checked_against_parent": False}
    if parent_rows is None:
        return out
    held_out = {r["qkey"] for r in parent_rows if r.get("split") in NON_REPORT_SPLITS}
    leaked = sorted({r["qkey"] for r in rows} & held_out)
    if leaked:
        raise RosterError(
            f"{pool}: {len(leaked)} question(s) are in the read split AND in the parent "
            f"manifest's fit/select splits, e.g. {leaked[:3]}. Train/val separation is the "
            "premise of every honest-selection number here; a question on both sides makes the "
            "dataset row an in-sample read of the split its members were chosen on.")
    return {**out, "checked_against_parent": True,
            "parent_held_out": len(held_out), "leaked": 0}


def check_sources_covered(rows_by_pool: Mapping[str, Sequence[Mapping]],
                          rules: Sequence[DatasetRule]) -> dict:
    claimed: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rule in rules:
        for member in rule.members:
            claimed[member].append(rule.dataset_id)
    present = {(pool, str(row.get("source")))
               for pool, rows in rows_by_pool.items() for row in rows}
    twice = {m: ds for m, ds in claimed.items() if len(ds) > 1}
    if twice:
        raise RosterError(
            f"source(s) claimed by more than one dataset rule: "
            f"{ {f'{p}/{s}': d for (p, s), d in twice.items()} }. Every question must land in "
            "exactly one row; a source in two rows is counted twice in any total and its "
            "questions appear in two denominators.")
    unclaimed = sorted(present - set(claimed))
    if unclaimed:
        raise RosterError(
            f"manifest source(s) {unclaimed} are in the pools but named by no dataset rule. "
            "Their questions would be dropped from every row while each printed row stayed "
            "self-consistent -- the only symptom is a total smaller than the pools, which is "
            "precisely what a reader cannot check. Add a rule (or state the exclusion) rather "
            "than letting the grouping decide silently.")
    absent = sorted(m for m in claimed if m not in present)
    if absent:
        raise RosterError(
            f"dataset rule(s) name source(s) {absent} that no manifest holds "
            f"({ {f'{p}/{s}': claimed[(p, s)] for p, s in absent} }). A row whose source is not "
            "in the data prints as an empty dataset, which reads as 'this dataset was evaluated "
            "and scored nothing' rather than 'this rule is misspelled'.")
    return {"claimed": {f"{p}/{s}": d[0] for (p, s), d in sorted(claimed.items())},
            "n_sources": len(claimed)}


def check_unique_assignment(assignment: Mapping[str, list[str]]) -> None:
    multi = {q: ds for q, ds in assignment.items() if len(ds) > 1}
    if multi:
        sample = {q: ds for q, ds in list(sorted(multi.items()))[:3]}
        raise RosterError(
            f"{len(multi)} question(s) were assigned to more than one dataset, e.g. {sample}. A "
            "question in two rows is counted twice in any total and appears in two denominators, "
            "so the two rows are not disjoint evidence about two datasets.")


def check_cross_pool_duplicate_qkeys(rows_by_pool: Mapping[str, Sequence[Mapping]]) -> dict:
    seen: dict[str, list[str]] = defaultdict(list)
    for pool, rows in sorted(rows_by_pool.items()):
        for qkey in {row["qkey"] for row in rows}:
            seen[qkey].append(pool)
    shared = {q: pools for q, pools in seen.items() if len(pools) > 1}
    if shared:
        sample = dict(list(sorted(shared.items()))[:3])
        raise RosterError(
            f"{len(shared)} qkey(s) occur in more than one pool, e.g. {sample}. Merging two "
            "tranches on qkey would then keep one pool's cluster for that question and drop the "
            "other's without raising, so the dataset row would be computed on a question set it "
            "does not report.")
    return {"n_qkeys": len(seen), "n_shared": 0}


def check_intra_pool_duplicate_qkeys(pool: str, rows: Sequence[Mapping]) -> None:
    counts = Counter(row["qkey"] for row in rows)
    dup = sorted(q for q, n in counts.items() if n > 1)
    if dup:
        raise RosterError(
            f"{pool}: {len(dup)} duplicate qkey(s) in the read split, e.g. {dup[:3]}. The "
            "population's own digest convention refuses duplicates, so a pool that needs "
            "deduplication to reach its registered size was never that size.")


def check_counts(measured: Mapping[str, int], rules: Sequence[DatasetRule]) -> dict:
    problems = []
    for rule in rules:
        got = measured.get(rule.dataset_id, 0)
        if got != rule.expected_n:
            problems.append(f"{rule.display_name}: {got} questions, rule expects "
                            f"{rule.expected_n}")
    if problems:
        raise RosterError(
            "measured dataset sizes disagree with the frozen rule table: " + "; ".join(problems)
            + ". STOP and report this: the expectations were verified against these manifests "
            "independently, so a disagreement means the inputs or the members changed, not that "
            "the grouping should be adjusted until it fits.")
    return {rule.dataset_id: rule.expected_n for rule in rules}


def check_names(rules: Sequence[DatasetRule]) -> dict:
    names = Counter(rule.display_name for rule in rules)
    clash = sorted(n for n, c in names.items() if c > 1)
    if clash:
        raise RosterError(f"display name(s) {clash} are used by more than one dataset rule; two "
                          "rows under one name cannot be told apart in any table.")
    for rule in rules:
        if not rule.filter_note.strip():
            raise RosterError(
                f"{rule.dataset_id!r} declares no `filter_note`. Every row here is a filtered "
                "subset of the benchmark it names, and a row that does not say so is a subset "
                "impersonating a full benchmark.")
        forbidden = FORBIDDEN_NAMES.get(rule.dataset_id, ())
        if rule.display_name in forbidden or rule.benchmark in forbidden:
            raise RosterError(
                f"{rule.dataset_id!r} may not be called {rule.display_name!r}: "
                f"{sorted(forbidden)} name a DIFFERENT question set. These two are disjoint "
                "halves of one test split, so the name would make a reader believe a benchmark "
                "was evaluated at several times its own size.")
        if rule.year_resolved and "extra batch" in rule.display_name:
            raise RosterError(
                f"{rule.dataset_id!r} is marked `year_resolved` but still named "
                f"{rule.display_name!r}. Say which year, or keep it a batch -- a row that claims "
                "both is a guessed year wearing a hedge.")
        if not rule.year_resolved and re.search(r"\b(19|20)\d{2}\b", rule.display_name):
            raise RosterError(
                f"{rule.dataset_id!r} is named {rule.display_name!r} -- a year -- but no "
                "pre-recorded rule resolved one. A year in a row name is a claim about which "
                "contest the questions came from, and this roster never guesses it.")
    return {"names": [rule.display_name for rule in rules],
            "year_resolved": {rule.dataset_id: rule.year_resolved for rule in rules}}


def check_no_pooling(dataset_ids: Sequence[str],
                    rules: Sequence[DatasetRule] | None = None) -> None:
    rules = DATASET_RULES if rules is None else rules
    wanted = set(dataset_ids)
    by_id = {rule.dataset_id: rule for rule in rules}
    for dataset_id in sorted(wanted):
        rule = by_id.get(dataset_id)
        if rule is None:
            continue
        for other, why in rule.no_pooling_with:
            if other in wanted:
                raise RosterError(
                    f"refusing to pool {rule.display_name!r} with "
                    f"{by_id[other].display_name if other in by_id else other!r}: {why}. Every "
                    "mechanical check passes on this pair -- they share no qkey and no exact "
                    "question text -- so pooling them would silently double-count problems and "
                    "print one N that is larger than the number of distinct problems behind it.")


AOPS_URL = re.compile(
    r"/((?:19|20)\d{2})_(AIME|AMC)_([IVX]+|\d+[AB])_Problems/Problem_(\d+)")
CONTEST_ID = re.compile(r"^((?:19|20)\d{2})-([IVX]+)-(\d+)$")


def contest_identity(row: Mapping) -> tuple | None:
    level = row.get("level")
    if isinstance(level, str):
        found = AOPS_URL.search(level)
        if found:
            year, contest, half, number = found.groups()
            return (year, contest, half, str(int(number)))
    source_id = row.get("source_id")
    if isinstance(source_id, str):
        found = CONTEST_ID.match(source_id)
        if found:
            year, half, number = found.groups()
            return (year, "AIME", half, str(int(number)))
    return None


def cross_reference_contests(membership: Sequence[Mapping]) -> dict:
    by_identity: dict[tuple, list[dict]] = defaultdict(list)
    for row in membership:
        identity = row.get("contest_identity")
        if identity:
            by_identity[tuple(identity)].append(row)
    overlaps = []
    for identity, rows in sorted(by_identity.items()):
        datasets = sorted({r["dataset_id"] for r in rows})
        if len(datasets) > 1:
            overlaps.append({
                "contest": "/".join(identity),
                "datasets": datasets,
                "qkeys": {r["dataset_id"]: r["qkey"] for r in rows},
                "gold": sorted({r.get("gold") for r in rows}),
            })
    pairs = Counter(tuple(o["datasets"]) for o in overlaps)
    return {
        "n_rows_with_contest_identity": sum(len(v) for v in by_identity.values()),
        "n_shared_contest_problems": len(overlaps),
        "pairs": {" + ".join(k): v for k, v in sorted(pairs.items())},
        "overlaps": overlaps,
        "note": ("same contest problem, different qkey and different LaTeX rendering. NOT "
                 "deduplicated and NOT pooled: both rows keep every question, and "
                 "`check_no_pooling` refuses a union of the two."),
    }


def year_evidence(membership: Sequence[Mapping], rules: Sequence[DatasetRule]) -> dict:
    unresolved = {rule.dataset_id for rule in rules if not rule.year_resolved}
    out: dict[str, dict] = {}
    for dataset_id in sorted(unresolved):
        rows = [r for r in membership if r["dataset_id"] == dataset_id]
        years = Counter(r["contest_identity"][0] for r in rows if r.get("contest_identity"))
        if not years:
            continue
        out[dataset_id] = {
            "n_questions": len(rows),
            "n_with_metadata": sum(years.values()),
            "years": dict(sorted(years.items())),
            "contests": dict(sorted(Counter(
                "/".join(r["contest_identity"][:3]) for r in rows
                if r.get("contest_identity")).items())),
            "applied": False,
            "why_not": ("no pre-recorded rule turns this metadata into a dataset split; the "
                        "row keeps its 'extra batch' name until one is recorded"),
        }
    return out


@dataclass
class Roster:
    roster: dict
    membership: list[dict]
    digests: dict = field(default_factory=dict)


def membership_lines(membership: Sequence[Mapping]) -> str:
    ordered = sorted(membership, key=lambda r: (r["dataset_id"], r["qkey"]))
    return "".join(canonical_json(row) + "\n" for row in ordered)


def build_roster(manifest_paths: Mapping[str, str], *,
                 rules: Sequence[DatasetRule] = DATASET_RULES,
                 parent_paths: Mapping[str, str] | None = None) -> Roster:
    check_names(rules)
    missing = sorted(set(ORIGIN_MANIFESTS) - set(manifest_paths))
    if missing:
        raise RosterError(
            f"no manifest given for pool(s) {missing}. All four report pools are built at once "
            "on purpose: the cross-pool duplicate-qkey check and the 5,075-question total are "
            "statements about the four together, and a roster built from three of them cannot "
            "make either.")

    rows_by_pool: dict[str, list[dict]] = {}
    bindings: dict[str, dict] = {}
    populations: dict[str, dict] = {}
    splits: dict[str, dict] = {}
    for pool in sorted(ORIGIN_MANIFESTS):
        rows, binding = report_rows(pool, manifest_paths[pool])
        check_question_hashes(pool, rows)
        check_intra_pool_duplicate_qkeys(pool, rows)
        parent_rows = None
        if binding["role"] == "parent":
            parent_rows = read_manifest_rows(manifest_paths[pool])
        elif parent_paths and pool in parent_paths:
            parent_binding = bind_manifest(pool, parent_paths[pool])
            if parent_binding["role"] != "parent":
                raise RosterError(
                    f"--parent for {pool!r} is the READ SPLIT again ({parent_binding['declared_path']}), "
                    "not the parent manifest. The train/val separation check compares the read "
                    "split against the fit and select rows; against itself it proves nothing "
                    "and would report a clean separation for any input.")
            parent_rows = read_manifest_rows(parent_paths[pool])
            bindings[f"{pool}::parent"] = parent_binding
        splits[pool] = check_split_isolation(pool, rows, parent_rows)
        populations[pool] = check_registered_population(pool, rows)
        rows_by_pool[pool] = rows
        bindings[pool] = binding

    duplicates = check_cross_pool_duplicate_qkeys(rows_by_pool)
    coverage = check_sources_covered(rows_by_pool, rules)

    by_member = {member: rule for rule in rules for member in rule.members}
    assignment: dict[str, list[str]] = defaultdict(list)
    membership: list[dict] = []
    for pool in sorted(rows_by_pool):
        tranche = ORIGIN_MANIFESTS[pool]["tranche"]
        for row in rows_by_pool[pool]:
            rule = by_member[(pool, str(row["source"]))]
            assignment[row["qkey"]].append(rule.dataset_id)
            entry = {
                "dataset_id": rule.dataset_id,
                "display_name": rule.display_name,
                "origin_pool": pool,
                "tranche": tranche,
                "gold": row.get("gold"),
                **{f: row.get(f) for f in COMMON_FIELDS},
            }
            for extra in rule.keep_fields:
                entry[f"source_{extra}"] = row.get(extra)
            identity = contest_identity(row)
            if identity is not None:
                entry["contest_identity"] = list(identity)
            membership.append(entry)
    check_unique_assignment(assignment)

    measured = Counter(row["dataset_id"] for row in membership)
    check_counts(measured, rules)

    datasets = []
    for rule in rules:
        rows = [r for r in membership if r["dataset_id"] == rule.dataset_id]
        qkeys = [r["qkey"] for r in rows]
        by_pool = Counter(r["origin_pool"] for r in rows)
        datasets.append({
            "dataset_id": rule.dataset_id,
            "display_name": rule.display_name,
            "subset_label": rule.subset_label,
            "benchmark": rule.benchmark,
            "n_questions": len(qkeys),
            "expected_n": rule.expected_n,
            "qkeys_sha256": qkeys_sha256(qkeys),
            "question_sha256_digest": payload_sha256(sorted(r["question_sha256"] for r in rows)),
            "members": [{"origin_pool": pool, "source": source,
                         "tranche": ORIGIN_MANIFESTS[pool]["tranche"],
                         "n_questions": by_pool[pool]}
                        for pool, source in rule.members],
            "origin_pools": list(rule.pools),
            "is_union": len(rule.pools) > 1,
            "target_chains": {str(k): v for k, v in
                              sorted(Counter(r["target_chains"] for r in rows).items())},
            "filter_note": rule.filter_note,
            "year_resolved": rule.year_resolved,
            "no_pooling_with": [{"dataset_id": o, "why": w} for o, w in rule.no_pooling_with],
            "note": rule.note,
            "source_id_kind": sorted({type(r["source_id"]).__name__ for r in rows}),
        })

    total = sum(d["n_questions"] for d in datasets)
    pool_total = sum(p["n_questions"] for p in populations.values())
    if total != pool_total:
        raise RosterError(
            f"the dataset rows hold {total} questions and the four pools hold {pool_total}. "
            "Every pool question must land in exactly one row: a difference means the grouping "
            "dropped or duplicated questions, and each printed row would still be internally "
            "consistent.")

    contests = cross_reference_contests(membership)
    roster = {
        "schema": SCHEMA,
        "exploratory": True,
        "requested_by": "user (steer 2026-09-13 section 34); exploratory, pointwise CIs",
        "answer_filter": ANSWER_FILTER,
        "n_questions_total": total,
        "n_datasets": len(datasets),
        "datasets": datasets,
        "pools": {pool: {**populations[pool],
                         "tranche": ORIGIN_MANIFESTS[pool]["tranche"],
                         "manifest": bindings[pool],
                         "split_isolation": splits[pool]}
                  for pool in sorted(populations)},
        "pool_totals": {pool: populations[pool]["n_questions"] for pool in sorted(populations)},
        "source_coverage": coverage,
        "duplicate_qkeys": duplicates,
        "contest_cross_reference": contests,
        "year_evidence": year_evidence(membership, rules),
        "conventions": {
            "qkeys_sha256": ("sha256 of the sorted unique qkeys joined by newline with a "
                             "trailing newline (rl_controller.iclr.read_splits convention); "
                             "duplicates are refused, never folded"),
            "membership_sha256": ("sha256 of the membership file's bytes: one canonical JSON "
                                  "object per line, sorted by (dataset_id, qkey)"),
            "roster_sha256": ("sha256 of this object's canonical JSON with the `digests` block "
                              "removed"),
        },
        "pool_and_tranche_are_metadata": (
            "t1 and t2 are two report tranches of the SAME three sources and are never two "
            "datasets; math_ood and ood2_math are different external compositions. Pool and "
            "tranche are kept per question for traceability and are not report rows."),
    }
    return Roster(roster=roster, membership=membership)


def finalize(built: Roster) -> Roster:
    lines = membership_lines(built.membership)
    membership_digest = hashlib.sha256(lines.encode("utf-8")).hexdigest()
    body = {k: v for k, v in built.roster.items() if k != "digests"}
    roster_digest = payload_sha256(body)
    built.roster["digests"] = {
        "membership_sha256": membership_digest,
        "n_membership_rows": len(built.membership),
        "roster_sha256": roster_digest,
    }
    built.digests = dict(built.roster["digests"])
    return built


def verify_frozen(roster: Mapping, membership_bytes: bytes) -> dict:
    declared = dict(roster.get("digests") or {})
    body = {k: v for k, v in roster.items() if k != "digests"}
    found_roster = payload_sha256(body)
    found_membership = hashlib.sha256(membership_bytes).hexdigest()
    problems = []
    if declared.get("roster_sha256") != found_roster:
        problems.append(f"roster body hashes to {found_roster[:16]}..., the file declares "
                        f"{str(declared.get('roster_sha256'))[:16]}...")
    if declared.get("membership_sha256") != found_membership:
        problems.append(f"membership bytes hash to {found_membership[:16]}..., the roster "
                        f"declares {str(declared.get('membership_sha256'))[:16]}...")
    if problems:
        raise RosterError(
            "the frozen roster does not match its own contents: " + "; ".join(problems)
            + ". The grouping is frozen BEFORE any result is computed precisely so that no "
            "number can have influenced it; a roster edited or regenerated afterwards cannot "
            "make that claim, whatever the two files say about each other.")
    return {"roster_sha256": found_roster, "membership_sha256": found_membership,
            "verified": True}


def load_membership(path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RosterError(f"{path}:{lineno}: not JSON ({exc})") from None
    return rows


def dataset_qkeys(membership: Sequence[Mapping], dataset_id: str) -> dict[str, list[str]]:
    out: dict[str, set] = defaultdict(set)
    for row in membership:
        if row["dataset_id"] == dataset_id:
            out[row["origin_pool"]].add(row["qkey"])
    if not out:
        raise RosterError(
            f"the frozen membership names no questions for dataset {dataset_id!r} (it holds "
            f"{sorted({r['dataset_id'] for r in membership})}). A dataset row computed on an "
            "empty question set is not a small sample, it is a different file than the one this "
            "run was pointed at.")
    return {pool: sorted(qkeys) for pool, qkeys in sorted(out.items())}


def rule_for(dataset_id: str, rules: Sequence[DatasetRule] | None = None) -> DatasetRule:
    rules = DATASET_RULES if rules is None else rules
    for rule in rules:
        if rule.dataset_id == dataset_id:
            return rule
    raise RosterError(f"no dataset rule named {dataset_id!r}; the frozen table holds "
                      f"{[r.dataset_id for r in rules]}")
