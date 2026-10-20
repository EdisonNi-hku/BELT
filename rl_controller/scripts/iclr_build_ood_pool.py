"""Build the out-of-distribution math benchmark manifest."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import os
from collections.abc import Container, Iterable, Iterator, Mapping
from pathlib import Path

from rl_controller.iclr import pool
from rl_controller.scripts import iclr_build_math_pool as math_builder
from rl_controller.scripts.iclr_build_math_pool import (
    EVAL_SETS,
    _cached,
    _eval_text,
    _read_jsonl,
    _read_meta,
    _Recorder,
    _sha256_file,
    _write_rows_atomically,
    write_meta,
)
from rl_controller.splits import question_key

CONTRACT_VERSION = "iclr-ood-pool-2026-08-26-v1"
SPLIT = "report"
TRANCHE = 0

OOD_SETS = {
    "math500": {"dataset": "HuggingFaceH4/MATH-500", "config": None, "split": "test",
                "field_map": {"question": "problem", "gold_raw": "answer",
                              "level": "subject/L{level}", "source_id": "unique_id"}},
    "aime24": {"dataset": "Maxwell-Jia/AIME_2024", "config": None, "split": "train",
               "field_map": {"question": "Problem", "gold_raw": "Answer", "level": None,
                             "source_id": "ID"}},
    "aime25": {"dataset": "yentinglin/aime_2025", "config": None, "split": "train",
               "field_map": {"question": "problem", "gold_raw": "answer", "level": "year",
                             "source_id": "id"}},
    "amc23": {"dataset": "AI-MO/aimo-validation-amc", "config": None, "split": "train",
              "field_map": {"question": "problem",
                            "gold_raw": "answer (integer-valued float coerced to int)",
                            "level": None, "source_id": "row index",
                            "filter": "'2023_AMC' in url"}},
    "olympiadbench": {"dataset": "Hothan/OlympiadBench", "config": "OE_TO_maths_en_COMP",
                      "split": "train",
                      "field_map": {"question": "context + ' ' + question (context when non-empty)",
                                    "gold_raw": "final_answer[0] (single-element list; "
                                                "is_multiple_answer != True)",
                                    "level": "subfield", "source_id": "id"}},
    "gaokao23en": {"dataset": "MARIO-Math-Reasoning/Gaokao2023-Math-En", "config": None,
                   "split": "train", "split_fallback": True,
                   "field_map": {"question": "question", "gold_raw": "answer", "level": None,
                                 "source_id": "source (row index if absent)"}},
    "gpqa_diamond": {"dataset": "Idavidrein/gpqa", "config": "gpqa_diamond", "split": "train",
                     "gated": True,
                     "field_map": {"question": "Question + numbered options (salted permutation) + instruction",
                                   "gold_raw": "1-based position of 'Correct Answer' after the permutation",
                                   "level": "High-level domain", "source_id": "Record ID"}},
    "math_test_rest": {"dataset": "EleutherAI/hendrycks_math", "config": None, "split": "test",
                       "all_configs": True, "tranche2": True,
                       "field_map": {"question": "problem",
                                     "gold_raw": "last \\boxed{...} of solution",
                                     "level": "<config>/L{level}", "source_id": "<config>/row index"}},
    "aime_val": {"dataset": "AI-MO/aimo-validation-aime", "config": None, "split": "train",
                 "tranche2": True,
                 "field_map": {"question": "problem", "gold_raw": "answer", "level": "url",
                               "source_id": "id (row index if absent)"}},
    "amc_val": {"dataset": "AI-MO/aimo-validation-amc", "config": None, "split": "train",
                "tranche2": True,
                "field_map": {"question": "problem",
                              "gold_raw": "answer (integer-valued float coerced to int)",
                              "level": "url", "source_id": "row index"}},
}

DEFAULT_SETS = [name for name, spec in OOD_SETS.items()
                if not spec.get("gated") and not spec.get("tranche2")]
TRANCHE2_SETS = [name for name, spec in OOD_SETS.items() if spec.get("tranche2")]

_BOXED_OPEN = re.compile(r"\\boxed\{")


def last_boxed(text) -> str | None:
    if not isinstance(text, str):
        return None
    opens = [m.end() for m in _BOXED_OPEN.finditer(text)]
    if not opens:
        return None
    i, depth, out = opens[-1], 1, []
    while i < len(text):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(char)
        i += 1
    return None

GPQA_HUB_SHA = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
GPQA_SALT = "iclr-gpqa-options-2026-08-29-v1"
GPQA_OPTIONS_HEADER = "\n\nOptions:\n"
GPQA_INSTRUCTION = ("\n\nAnswer with the number (1-4) of the correct option, not the option's content."
                    " Put the option number in \\boxed{} at the end of your reasoning.")
GPQA_INTEGER_OPTION = re.compile(r"^[+-]?\d+$")
GPQA_COLUMNS = ("Record ID", "Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2",
                "Incorrect Answer 3", "High-level domain")

DEFAULT_CHAINS = {"aime24": 16, "aime25": 16, "amc23": 16}
AMC23_URL_MARKER = "2023_AMC"


def _parse_sets(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_SETS)
    names = [n.strip() for n in value.split(",") if n.strip()]
    unknown = [n for n in names if n not in OOD_SETS]
    if unknown:
        raise SystemExit(f"unknown OOD set(s) {unknown}; choose from {list(OOD_SETS)}")
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise SystemExit(f"duplicate OOD set(s) in --sets: {duplicates}")
    return names


def _col(name: str, row: dict, key: str):
    try:
        return row[key]
    except KeyError:
        raise SystemExit(f"{name}: hub row is missing column {key!r} (columns: {sorted(row)}); "
                         "the dataset schema changed -- refusing to cache") from None


def _coerce_integer_float(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def gpqa_options(row: dict) -> tuple[list[str], int]:
    raw = [" ".join(str(row[k]).split()) for k in ("Correct Answer", "Incorrect Answer 1",
                                                     "Incorrect Answer 2", "Incorrect Answer 3")]
    rid = str(row["Record ID"])
    order = sorted(range(4), key=lambda i: hashlib.sha256(f"{GPQA_SALT}:{rid}:{i}".encode()).hexdigest())
    options = [raw[i] for i in order]
    return options, order.index(0) + 1


def gpqa_question(row: dict) -> tuple[str, int]:
    options, gold = gpqa_options(row)
    body = "".join(f"({i}) {opt}\n" for i, opt in enumerate(options, 1)).rstrip("\n")
    return str(row["Question"]).strip() + GPQA_OPTIONS_HEADER + body + GPQA_INSTRUCTION, gold


def _gpqa_rows(csv_path: Path) -> Iterator[dict]:
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in GPQA_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{csv_path}: missing GPQA columns {missing} (have {reader.fieldnames})")
        yield from reader


def ood_question_text(name: str, row: dict) -> str:
    if name == "gpqa_diamond":
        return gpqa_question(row)[0]
    if name in EVAL_SETS:
        return _eval_text(EVAL_SETS[name], row)
    return str(_col(name, row, OOD_SETS[name]["field_map"]["question"]))


def _olympiad_gold(row: dict) -> tuple[str | None, str | None]:
    multiple = row.get("is_multiple_answer")
    if multiple is True or str(multiple) == "True":
        return None, "multi_answer"
    answer = row.get("final_answer")
    if isinstance(answer, str):
        try:
            answer = ast.literal_eval(answer)
        except (ValueError, SyntaxError):
            return None, "unparseable_answer"
    if not isinstance(answer, (list, tuple)):
        return None, "unparseable_answer"
    if len(answer) != 1:
        return None, "multi_answer"
    return str(answer[0]), None


def _convert_set(name: str, rows: Iterable[dict], stats: dict) -> Iterator[dict]:
    for idx, row in enumerate(rows):
        stats["n_rows_raw"] += 1
        if name == "gpqa_diamond":
            for column in GPQA_COLUMNS:
                _col(name, row, column)
            question, gold = gpqa_question(row)
            gold_raw, level, source_id = str(gold), row.get("High-level domain"), row.get("Record ID", idx)
            stats["n_kept"] += 1
            yield {"source": name, "source_id": source_id, "question": question,
                   "gold_raw": gold_raw, "level": level}
            continue
        if name in EVAL_SETS:
            _col(name, row, EVAL_SETS[name]["field"])
        question = ood_question_text(name, row)
        if name == "math500":
            gold_raw = _col(name, row, "answer")
            level, source_id = f"{row.get('subject')}/L{row.get('level')}", row.get("unique_id", idx)
        elif name == "aime24":
            gold_raw, level, source_id = _col(name, row, "Answer"), None, row.get("ID", idx)
        elif name == "aime25":
            year = row.get("year")
            gold_raw, level = _col(name, row, "answer"), (str(year) if year is not None else None)
            source_id = row.get("id", idx)
        elif name == "amc23":
            if AMC23_URL_MARKER not in str(_col(name, row, "url")):
                stats["n_dropped_other_year"] += 1
                continue
            gold_raw, level, source_id = _coerce_integer_float(_col(name, row, "answer")), None, idx
        elif name == "olympiadbench":
            _col(name, row, "final_answer")
            gold_raw, reason = _olympiad_gold(row)
            if reason is not None:
                stats[f"n_dropped_{reason}"] += 1
                continue
            level, source_id = row.get("subfield"), row.get("id", idx)
        elif name == "gaokao23en":
            gold_raw, level, source_id = _col(name, row, "answer"), None, (row.get("source") or idx)
        elif name == "math_test_rest":
            config = str(row.get("_config") or "")
            gold_raw = last_boxed(_col(name, row, "solution"))
            level, source_id = f"{config}/L{row.get('level')}", f"{config}/{row.get('_index', idx)}"
        elif name == "aime_val":
            gold_raw, level = _col(name, row, "answer"), str(row.get("url") or "")[:200] or None
            source_id = row.get("id", idx)
        elif name == "amc_val":
            gold_raw = _coerce_integer_float(_col(name, row, "answer"))
            level, source_id = str(row.get("url") or "")[:200] or None, idx
        else:
            raise ValueError(f"unknown OOD set {name!r}")
        stats["n_kept"] += 1
        yield {"source": name, "source_id": source_id, "question": question,
               "gold_raw": gold_raw, "level": level}


def fetch_ood_set(name: str, cache_dir: Path, *, rows: Iterable[dict] | None = None,
                  split_used: str | None = None, csv_path: str | Path | None = None) -> None:
    spec = OOD_SETS[name]
    cache_dir = Path(cache_dir)
    cache_name = f"ood_{name}"
    if _cached(cache_dir, cache_name):
        if csv_path is not None:
            meta = _read_meta(cache_dir / f"{cache_name}.meta.json") or {}
            if meta.get("csv_sha256") != _sha256_file(Path(csv_path)):
                raise SystemExit(f"{name} is cached from a different CSV (meta csv_sha256 {meta.get('csv_sha256')}); "
                                 "move the cache aside to re-fetch")
        print(json.dumps({"fetch": name, "status": "skipped: cached"}), flush=True)
        return
    extra_meta: dict = {}
    if name == "gpqa_diamond":
        revision = GPQA_HUB_SHA
        if rows is None:
            if csv_path is None:
                raise SystemExit("gpqa_diamond is gated: pass --gpqa-csv <file downloaded with an "
                                 "authorised HF account>")
            csv_path = Path(csv_path)
            if not csv_path.is_file():
                raise SystemExit(f"gpqa_diamond: --gpqa-csv {csv_path} is not a file")
            extra_meta = {"csv_path": str(csv_path.resolve()), "csv_sha256": _sha256_file(csv_path)}
            rows = _gpqa_rows(csv_path)
    else:
        revision = math_builder.dataset_revision(spec["dataset"])
    if rows is None and spec.get("all_configs"):
        from datasets import get_dataset_config_names

        configs = sorted(get_dataset_config_names(spec["dataset"], revision=revision))
        if not configs:
            raise SystemExit(f"{name}: no configs at revision {revision}; refusing to cache")
        split_used = spec["split"]
        extra_meta = {**extra_meta, "configs": configs}

        def _all_config_rows():
            for config in configs:
                for index, row in enumerate(math_builder.iter_hub_rows(
                        spec["dataset"], config, split_used, revision=revision)):
                    yield {**row, "_config": config, "_index": index}

        rows = _all_config_rows()
    elif rows is None:
        split_used = math_builder.resolve_split(spec["dataset"], spec["config"], spec["split"],
                                                bool(spec.get("split_fallback")), revision=revision)
        rows = math_builder.iter_hub_rows(spec["dataset"], spec["config"], split_used,
                                          revision=revision)
    stats = {"n_rows_raw": 0, "n_kept": 0, "n_dropped_multi_answer": 0,
             "n_dropped_unparseable_answer": 0, "n_dropped_other_year": 0}
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{cache_name}.jsonl"
    n_rows = _write_rows_atomically(out, _convert_set(name, rows, stats))
    if n_rows == 0:
        out.unlink()
        raise SystemExit(f"{name}: no rows survived conversion (n_rows_raw={stats['n_rows_raw']}); "
                         "a column may be missing or renamed; refusing to cache")
    extra = {k: v for k, v in stats.items()
             if k in ("n_rows_raw", "n_kept")
             or (name == "olympiadbench" and k in ("n_dropped_multi_answer", "n_dropped_unparseable_answer"))
             or (name == "amc23" and k == "n_dropped_other_year")}
    meta = write_meta(cache_dir / f"{cache_name}.meta.json", name=name, kind="ood_set",
                      dataset=spec["dataset"], config=spec["config"],
                      split=split_used or spec["split"], revision=revision, n_rows=n_rows,
                      field_map=spec["field_map"], status="ok", **extra, **extra_meta)
    print(json.dumps({"fetch": name, **{k: meta[k] for k in ("n_rows", "split", "revision", "status")}}),
          flush=True)


def cmd_fetch(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)
    for name in _parse_sets(args.sets):
        fetch_ood_set(name, cache_dir, csv_path=args.gpqa_csv if name == "gpqa_diamond" else None)
    return 0


def _write_text_atomically(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def _manifest_text(rows: Iterable[dict]) -> str:
    return "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)


def _parse_chains(items: list[str]) -> dict[str, int]:
    chains: dict[str, int] = {}
    for item in items:
        name, _, value = item.partition("=")
        if name not in OOD_SETS or not value.isdigit() or int(value) < 1:
            raise SystemExit(f"--chains expects <set>=<positive int> with set in {list(OOD_SETS)}, "
                             f"got {item!r}")
        chains[name] = int(value)
    return chains


def _new_counts(cached_rows: int, meta: dict | None, target_chains: int) -> dict:
    meta = meta or {}
    return {
        "cached_rows": cached_rows,
        "kept": 0,
        "dropped_non_integer": 0,
        "dropped_figure": 0,
        "dropped_too_long": 0,
        "dropped_empty": 0,
        "dropped_duplicate": 0,
        "dropped_multi_answer": int(meta.get("n_dropped_multi_answer", 0) or 0),
        "dropped_unparseable_answer": int(meta.get("n_dropped_unparseable_answer", 0) or 0),
        "dropped_excluded": {"qkey": 0, "question_sha256": 0},
        "dropped_decontam": {},
        "target_chains": int(target_chains),
    }


def build_rows(cached: dict[str, list[dict]], *, chains_per_set: dict[str, int],
               metas: dict[str, dict | None], max_question_chars: int,
               exclude_qkeys: Container[str] = frozenset(),
               exclude_sha256s: Container[str] = frozenset(),
               decontam_index: "pool.NGramIndex | None" = None,
               decontam_exact: Mapping[str, str] | None = None,
               tranche: int = TRANCHE) -> tuple[list[dict], dict, list[dict]]:
    counts: dict[str, dict] = {}
    seen: set[str] = set()
    rows: list[dict] = []
    hits: list[dict] = []
    for name, cached_rows in cached.items():
        c = counts[name] = _new_counts(len(cached_rows), metas.get(name), chains_per_set[name])
        for raw in cached_rows:
            gold = pool.parse_integer_answer(raw.get("gold_raw"))
            if gold is None:
                c["dropped_non_integer"] += 1
                continue
            question = str(raw["question"])
            if not question.strip():
                c["dropped_empty"] += 1
                continue
            if len(question) > max_question_chars:
                c["dropped_too_long"] += 1
                continue
            if pool.has_figure(question):
                c["dropped_figure"] += 1
                continue
            qkey = question_key(question)
            if qkey in seen:
                c["dropped_duplicate"] += 1
                continue
            question_sha256 = hashlib.sha256(question.encode()).hexdigest()
            if qkey in exclude_qkeys:
                c["dropped_excluded"]["qkey"] += 1
                continue
            if question_sha256 in exclude_sha256s:
                c["dropped_excluded"]["question_sha256"] += 1
                continue
            matched = (decontam_exact or {}).get(qkey)
            if matched is None and decontam_index is not None:
                matched = decontam_index.overlaps(question)
            if matched is not None:
                c["dropped_decontam"][matched] = c["dropped_decontam"].get(matched, 0) + 1
                hits.append({"qkey": qkey, "source": name, "source_id": raw.get("source_id"),
                             "matched": matched})
                continue
            seen.add(qkey)
            c["kept"] += 1
            rows.append({
                "qkey": qkey,
                "question": question,
                "question_sha256": question_sha256,
                "gold": int(gold),
                "split": SPLIT,
                "target_chains": int(chains_per_set[name]),
                "allocation_subset": False,
                "source": name,
                "source_id": raw.get("source_id"),
                "level": raw.get("level"),
                "tranche": int(tranche),
            })
    rows.sort(key=lambda r: (r["source"], r["qkey"]))
    return rows, counts, hits


def _gpqa_ngram_check(rows: list[dict], pool_rows: list[tuple[Path, list[dict]]]) -> dict:
    index = pool.NGramIndex()
    exact: set[str] = set()
    n_gpqa = 0
    for row in rows:
        if row["source"] != "gpqa_diamond":
            continue
        n_gpqa += 1
        question = str(row["question"])
        for text in (question, question.split(GPQA_OPTIONS_HEADER, 1)[0]):
            index.add(text, "gpqa_diamond")
            exact.add(question_key(text))
    hits: list[dict] = []
    n_training = 0
    for path, prev in pool_rows:
        for prev_row in prev:
            n_training += 1
            question = str(prev_row.get("question") or "")
            matched = index.overlaps(question)
            if matched is None and question.strip() and question_key(question) in exact:
                matched = "gpqa_diamond"
            if matched is not None:
                hits.append({"qkey": prev_row.get("qkey"), "manifest": str(path.resolve()), "matched": matched})
    return {"ngram_n": index.n, "n_gpqa_questions": n_gpqa, "n_training_questions": n_training,
            "n_hits": len(hits), "hits": hits}


def cmd_build(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)
    manifest_out, contract_out = Path(args.manifest_out), Path(args.contract_out)
    for path in (manifest_out, contract_out):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing {path}")
    sets = _parse_sets(args.sets)
    overrides = _parse_chains(args.chains)
    chains_per_set = {name: overrides.get(name, DEFAULT_CHAINS.get(name, args.default_chains))
                      for name in sets}

    cached: dict[str, list[dict]] = {}
    metas: dict[str, dict | None] = {}
    for name in sets:
        path = cache_dir / f"ood_{name}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing cached OOD set {name} ({path}); run `fetch` first")
        cached[name] = _read_jsonl(path)
        metas[name] = _read_meta(cache_dir / f"ood_{name}.meta.json")

    record = _Recorder(cache_dir / f"ood_{sets[0]}.jsonl")
    exclude_qkeys: set[str] = set()
    exclude_sha256s: set[str] = set()
    exclude_records: list[dict] = []
    for path in args.exclude_manifest:
        prev = _read_jsonl(Path(path))
        for prev_row in prev:
            if prev_row.get("qkey"):
                exclude_qkeys.add(str(prev_row["qkey"]))
            if prev_row.get("question_sha256"):
                exclude_sha256s.add(str(prev_row["question_sha256"]))
        exclude_records.append({**record(Path(path)), "n_questions": len(prev)})
    decontam_index = pool.NGramIndex()
    decontam_exact: dict[str, str] = {}
    decontam_records: list[dict] = []
    for path in args.decontam_manifest:
        prev = _read_jsonl(Path(path))
        tag = Path(path).name
        for prev_row in prev:
            question = str(prev_row.get("question") or "")
            if not question.strip():
                continue
            decontam_exact.setdefault(question_key(question), tag)
            decontam_index.add(question, tag)
        decontam_records.append({**record(Path(path)), "n_questions": len(prev), "tag": tag})

    rows, counts, decontam_hits = build_rows(
        cached, chains_per_set=chains_per_set, metas=metas,
        max_question_chars=args.max_question_chars,
        exclude_qkeys=exclude_qkeys, exclude_sha256s=exclude_sha256s,
        decontam_index=decontam_index if args.decontam_manifest else None,
        decontam_exact=decontam_exact, tranche=args.tranche)
    empty = [name for name in sets if counts[name]["kept"] == 0]
    if empty:
        raise SystemExit(f"OOD set(s) {empty} keep 0 rows after filtering "
                         f"({ {n: counts[n] for n in empty} }); refusing to write a manifest")

    ood_qkeys = {r["qkey"]: r["source"] for r in rows}
    pool_manifests: list[dict] = []
    pool_records: list[dict] = []
    pool_rows: list[tuple[Path, list[dict]]] = []
    for path in args.pool_manifest:
        path = Path(path)
        prev = _read_jsonl(path)
        pool_rows.append((path, prev))
        overlap: dict[str, int] = {}
        for prev_row in prev:
            source = ood_qkeys.get(str(prev_row.get("qkey")))
            if source is not None:
                overlap[source] = overlap.get(source, 0) + 1
        if overlap:
            detail = ", ".join(f"{s}: {n}" for s, n in sorted(overlap.items()))
            raise SystemExit(
                f"OOD manifest overlaps the in-distribution pool manifest {path} on "
                f"{sum(overlap.values())} question(s) ({detail}); the decontamination "
                "invariant is broken -- stopping")
        rec = record(path)
        pool_manifests.append({"path": rec["path"], "sha256": rec["sha256"],
                               "n_questions": len(prev), "overlap": 0})
        pool_records.append(rec)
    gpqa_ngram_check = (_gpqa_ngram_check(rows, pool_rows) if "gpqa_diamond" in sets else None)
    if gpqa_ngram_check is not None:
        opts = [line[4:] for r in rows if r["source"] == "gpqa_diamond" for line in r["question"].split("\n")
                if len(line) > 3 and line[0] == "(" and line[2:4] == ") "]
        gpqa_ngram_check["integer_option_rows"] = sum(
            any(GPQA_INTEGER_OPTION.match(line[4:]) for line in r["question"].split("\n")
                if len(line) > 3 and line[0] == "(" and line[2:4] == ") ")
            for r in rows if r["source"] == "gpqa_diamond")
        gpqa_ngram_check["integer_option_in_1_to_4_rows"] = sum(
            any(GPQA_INTEGER_OPTION.match(line[4:]) and line[4:].lstrip("+") in ("1", "2", "3", "4")
                for line in r["question"].split("\n") if len(line) > 3 and line[0] == "(" and line[2:4] == ") ")
            for r in rows if r["source"] == "gpqa_diamond")
        del opts
    if gpqa_ngram_check and gpqa_ngram_check["hits"]:
        print(json.dumps({"gpqa_ngram_check": "WARNING: training question(s) overlap GPQA-Diamond",
                          "n_hits": len(gpqa_ngram_check["hits"])}), flush=True)

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomically(manifest_out, _manifest_text(rows))
    manifest_sha256 = _sha256_file(manifest_out)
    set_records = [{"name": name, "kind": "ood_set", "dataset": OOD_SETS[name]["dataset"],
                    "config": OOD_SETS[name]["config"], **record(cache_dir / f"ood_{name}.jsonl"),
                    "meta": metas[name]} for name in sets]
    cache_files = sorted(p for name in sets
                         for p in (cache_dir / f"ood_{name}.jsonl", cache_dir / f"ood_{name}.meta.json")
                         if p.exists())
    contract = {
        "contract_version": CONTRACT_VERSION,
        "status": "FROZEN_BEFORE_GENERATION",
        "report_generated": False,
        "split": SPLIT,
        "tranche": int(args.tranche),
        "salt": args.salt,
        "allocation_subset": False,
        "max_question_chars": int(args.max_question_chars),
        "default_chains": int(args.default_chains),
        "chains_per_set": chains_per_set,
        "set_order": sets,
        "sets": counts,
        "total_questions": len(rows),
        "total_chains": sum(r["target_chains"] for r in rows),
        "pool_manifests": pool_manifests,
        "exclude_manifests": exclude_records,
        "n_excluded_qkeys": len(exclude_qkeys),
        "decontamination": {
            "ngram_n": decontam_index.n,
            "manifests": decontam_records,
            "n_indexed_questions": len(decontam_index),
            "n_hits": len(decontam_hits),
            "hits_per_set": {name: dict(sorted(counts[name]["dropped_decontam"].items()))
                             for name in sets},
        },
        **({"gpqa_ngram_check": gpqa_ngram_check} if gpqa_ngram_check is not None else {}),
        "manifest_path": str(manifest_out.resolve()),
        "manifest_sha256": manifest_sha256,
        "set_records": set_records,
        "provenance": {
            "manifest": record(manifest_out),
            "cache_files": [record(p) for p in cache_files],
            "pool_manifests": pool_records,
            "exclude_manifests": exclude_records,
            "decontam_manifests": decontam_records,
        },
    }
    if args.decontam_report_out:
        _write_text_atomically(Path(args.decontam_report_out), json.dumps(
            {"salt": args.salt, "ngram_n": decontam_index.n, "manifest_sha256": manifest_sha256,
             "n_hits": len(decontam_hits), "manifests": decontam_records, "hits": decontam_hits},
            indent=2, sort_keys=True) + "\n")
    try:
        _write_text_atomically(contract_out, json.dumps(contract, indent=2, sort_keys=True) + "\n")
    except BaseException:
        manifest_out.unlink(missing_ok=True)
        raise

    summary = {
        "sets": sets, "kept": {name: counts[name]["kept"] for name in sets},
        "total_questions": len(rows), "total_chains": contract["total_chains"],
        "excluded": {name: dict(counts[name]["dropped_excluded"]) for name in sets},
        "decontam_hits": len(decontam_hits),
        "pool_manifests": len(pool_manifests),
        "manifest": str(manifest_out), "manifest_sha256": manifest_sha256,
        "contract": str(contract_out),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="materialise the OOD sets from the HF hub")
    fetch.add_argument("--cache-dir", required=True)
    fetch.add_argument("--sets", default=None,
                       help=f"comma-separated subset of {list(OOD_SETS)} (default: {DEFAULT_SETS})")
    fetch.add_argument("--gpqa-csv", default=None, metavar="FILE",
                       help="gpqa_diamond only: the gated CSV downloaded with an authorised HF account "
                            f"(revision {GPQA_HUB_SHA})")

    build = sub.add_parser("build", help="pure-python OOD manifest build over the cache")
    build.add_argument("--cache-dir", required=True)
    build.add_argument("--manifest-out", required=True)
    build.add_argument("--contract-out", required=True)
    build.add_argument("--sets", default=None,
                       help=f"comma-separated subset of {list(OOD_SETS)} (listed order = dedupe order; "
                            f"default: {DEFAULT_SETS})")
    build.add_argument("--chains", action="append", default=[], metavar="SET=N",
                       help="chains per question for one set (repeatable; overrides DEFAULT_CHAINS)")
    build.add_argument("--default-chains", type=int, default=8,
                       help="chains per question for sets not in DEFAULT_CHAINS / --chains")
    build.add_argument("--pool-manifest", action="append", default=[], metavar="MANIFEST",
                       help="an in-distribution manifest that must NOT share any qkey with this "
                            "manifest (repeatable; recorded in the contract)")
    build.add_argument("--max-question-chars", type=int, default=2000,
                       help="drop questions longer than this many characters")
    build.add_argument("--exclude-manifest", action="append", default=[], metavar="MANIFEST",
                       help="an earlier OOD manifest whose questions are already spent: every "
                            "matching qkey / question_sha256 is dropped and counted (repeatable)")
    build.add_argument("--decontam-manifest", action="append", default=[], metavar="MANIFEST",
                       help="a training-pool manifest to decontaminate against: exact question_key "
                            "or a shared 13-gram drops the question (repeatable)")
    build.add_argument("--decontam-report-out", default=None, metavar="FILE",
                       help="write the full per-question decontamination hit list here")
    build.add_argument("--salt", default=None,
                       help="provenance salt recorded in the contract (no split hashing happens "
                            "here: every OOD row is split=report)")
    build.add_argument("--tranche", type=int, default=TRANCHE,
                       help=f"tranche stamped on every row (default {TRANCHE})")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "fetch":
        return cmd_fetch(args)
    return cmd_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
