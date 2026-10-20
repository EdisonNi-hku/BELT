"""Build the code-domain question pools."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import pickle
import re
import zlib
from collections import Counter
from pathlib import Path

from rl_controller.iclr import code_pool, pool
from rl_controller.scripts.iclr_build_math_pool import (
    _cached,
    _read_jsonl,
    _read_meta,
    _Recorder,
    _sha256_file,
    _write_rows_atomically,
    dataset_revision,
    iter_hub_rows,
    write_meta,
)
from rl_controller.scripts.iclr_build_ood_pool import _manifest_text, _write_text_atomically
from rl_controller.splits import question_key

CONTRACT_VERSION = "iclr-code-pool-2026-08-28-v1"
SPLIT = "report"
TRANCHE = 0
DEFAULT_CHAINS = 8
DEFAULT_MAX_QUESTION_CHARS = 6000
PROBE = {"query_stride": 4, "max_forced_tokens": 768}
LCB_POST_CUTOFF = "2024-08-01"
LCB_STRICT_CUTOFF = "2025-02-01"
LCB_FILES = ("test4.jsonl", "test5.jsonl", "test6.jsonl")
LCB_ALL_FILES = ("test.jsonl", "test2.jsonl", "test3.jsonl") + LCB_FILES
LCB_STATS = ("n_rows_raw", "n_kept", "n_dropped_pre_cutoff", "n_dropped_tests", "n_dropped_malformed",
             "n_dropped_duplicate", "n_strict_post_qwen3")
LCB_MALFORMED = (ValueError, TypeError, KeyError, AttributeError, zlib.error, pickle.UnpicklingError)
LCB_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
SYNTHETIC2_CODE_PREFIX = "prime_rl_code_"
SYNTHETIC2_MAX_INFO_CHARS = 2 << 20
REWARD_COLUMNS = {"qwen3_4b_reward": "Qwen/Qwen3-4B_avg_reward",
                  "qwen3_32b_reward": "Qwen/Qwen3-32B_avg_reward",
                  "r1_qwen3_8b_reward": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B_avg_reward",
                  "r1_reward": "deepseek-ai/DeepSeek-R1-0528_avg_reward"}

CODE_SETS = {
    "humanevalplus": {"dataset": "evalplus/humanevalplus", "config": None, "split": "test",
                      "kind": "benchmark", "field_map": {"question": "SCRIPT_SUFFIX(entry_point, prompt)",
                                                         "test": "test (+ check(entry_point))"}},
    "mbppplus": {"dataset": "evalplus/mbppplus", "config": None, "split": "test",
                 "kind": "benchmark", "field_map": {"question": "prompt + SCRIPT_SUFFIX(def line of code)",
                                                    "test": "test_imports + test (module-level asserts)"}},
    "lcb_post": {"dataset": "livecodebench/code_generation_lite", "files": LCB_FILES,
                 "kind": "benchmark", "field_map": {"question": "question_content + stdin/call suffix",
                                                    "tests": "public + private (base64/zlib/pickle), 15 longest",
                                                    "filter": f"contest_date >= {LCB_POST_CUTOFF}"}},
    "lcb_rest": {"dataset": "livecodebench/code_generation_lite", "files": LCB_ALL_FILES,
                 "kind": "benchmark", "tranche2": True,
                 "field_map": {"question": "question_content + stdin/call suffix",
                               "tests": "public + private (base64/zlib/pickle), 1-MiB-input-first",
                               "filter": "no contest_date cutoff (tranche-0 qkeys excluded instead)"}},
    "synthetic2": {"dataset": "PrimeIntellect/SYNTHETIC-2-RL", "config": None, "split": "train",
                   "kind": "training_source", "field_map": {"prompt": "prompt", "verification_info": "raw",
                                                            **REWARD_COLUMNS}},
    "synthetic2_sft": {"dataset": "PrimeIntellect/SYNTHETIC-2-SFT-verified", "config": None, "split": "train",
                       "kind": "solutions",
                       "field_map": {"program": "last ```python block of the assistant message", "reward": "reward",
                                     "task_type": "task_type"}},
}
MANIFEST_FIELDS = ("qkey", "question", "question_sha256", "gold", "split", "target_chains",
                   "allocation_subset", "source", "source_id", "level", "tranche", "domain",
                   "answer_format", "test_kind", "tests_sha256", "n_tests", "benchmark",
                   "strict_post_qwen3", "contest_date")


def _col(name: str, row: dict, key: str):
    try:
        return row[key]
    except KeyError:
        raise SystemExit(f"{name}: hub row is missing column {key!r} (columns: {sorted(row)}); "
                         "the dataset schema changed -- refusing to cache") from None


def convert_humanevalplus(row: dict, idx: int):
    prompt, entry = str(_col("humanevalplus", row, "prompt")), str(_col("humanevalplus", row, "entry_point"))
    test = str(_col("humanevalplus", row, "test"))
    question = code_pool.compose_question("", test_kind="script", entry_point=entry, signature=prompt)
    tests = [{"kind": "script", "entry_point": entry, "test": test}]
    return question, tests, {"source_id": row.get("task_id", idx), "test_kind": "script",
                             "entry_point": entry, "fn_name": None, "level": None,
                             "contest_date": None, "strict_post_qwen3": None}


def _assert_root_call(test: str, defs: set[str]) -> str | None:
    try:
        tree = ast.parse(test)
    except SyntaxError:
        return None
    asserts = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Assert)),
                     key=lambda n: (n.lineno, n.col_offset))
    for node in asserts:
        for sub in ast.walk(node.test):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in defs:
                return sub.func.id
    return None


def _mbpp_entry_point(code: str, test: str, task_id) -> tuple[str, str]:
    code_defs = re.findall(r"^(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", code, re.M)
    defs = set(code_defs)
    entry = _assert_root_call(test, defs)
    if entry is None:
        called = re.findall(r"\b([A-Za-z_]\w*)\s*\(", test)
        entry = next((name for name in called if name in defs), None)
    if entry is None:
        raise SystemExit(f"mbppplus {task_id}: no top-level def of {code_defs} is called by the test")
    start = re.search(rf"^(?:async\s+)?def\s+{re.escape(entry)}\s*\(", code, re.M).start()
    return entry, code_pool.signature_from_code(code[start:])


def convert_mbppplus(row: dict, idx: int):
    code = str(_col("mbppplus", row, "code"))
    imports = "\n".join(str(line) for line in (row.get("test_imports") or []))
    test = (imports + "\n" if imports else "") + str(_col("mbppplus", row, "test"))
    entry, signature = _mbpp_entry_point(code, test, row.get("task_id", idx))
    question = code_pool.compose_question(str(_col("mbppplus", row, "prompt")), test_kind="script",
                                          entry_point=entry, signature=signature)
    tests = [{"kind": "script", "entry_point": entry, "test": test}]
    return question, tests, {"source_id": row.get("task_id", idx), "test_kind": "script",
                             "entry_point": entry, "fn_name": None, "level": None,
                             "contest_date": None, "strict_post_qwen3": None}


def _lcb_private_tests(encoded) -> list:
    if isinstance(encoded, list):
        return encoded
    text = str(encoded)
    try:
        return json.loads(text)
    except ValueError:
        obj = pickle.loads(zlib.decompress(base64.b64decode(text.encode("utf-8"))))
        return obj if isinstance(obj, list) else json.loads(obj)


def convert_lcb(row: dict, stats: dict, *, name: str = "lcb_post", cutoff: str | None = LCB_POST_CUTOFF):
    raw_date = _col(name, row, "contest_date")
    date = raw_date[:10] if isinstance(raw_date, str) else None
    if date is None or not LCB_DATE.fullmatch(date):
        stats["n_dropped_malformed"] += 1
        return None
    span = stats.setdefault("files", {}).setdefault(row.get("_file", "rows"),
                                                    {"n": 0, "min_date": date, "max_date": date})
    span["n"] += 1
    span["min_date"], span["max_date"] = min(span["min_date"], date), max(span["max_date"], date)
    if cutoff is not None and date < cutoff:
        stats["n_dropped_pre_cutoff"] += 1
        return None
    statement = _col(name, row, "question_content")
    if not isinstance(statement, str) or not statement.strip():
        stats["n_dropped_malformed"] += 1
        return None
    try:
        public = json.loads(_col(name, row, "public_test_cases"))
        private = _lcb_private_tests(_col(name, row, "private_test_cases"))
        metadata = json.loads(row.get("metadata") or "{}")
        tests = code_pool.normalise_tests("lcb", list(public) + list(private), metadata)
        tests = code_pool.select_tests(tests) if tests is not None else None
    except LCB_MALFORMED:
        stats["n_dropped_malformed"] += 1
        return None
    if tests is None:
        stats["n_dropped_tests"] += 1
        return None
    kind = tests[0]["kind"]
    fn_name = metadata.get("func_name") if kind == "call" else None
    question = code_pool.compose_question(statement, test_kind=kind,
                                          fn_name=fn_name, starter_code=str(row.get("starter_code") or ""))
    strict = date >= LCB_STRICT_CUTOFF
    stats["n_strict_post_qwen3"] += int(strict)
    return question, tests, {"source_id": _col(name, row, "question_id"), "test_kind": kind,
                             "entry_point": None, "fn_name": fn_name, "level": row.get("difficulty"),
                             "contest_date": date, "strict_post_qwen3": strict}


CONVERTERS = {"humanevalplus": lambda row, idx, stats: convert_humanevalplus(row, idx),
              "mbppplus": lambda row, idx, stats: convert_mbppplus(row, idx),
              "lcb_post": lambda row, idx, stats: convert_lcb(row, stats),
              "lcb_rest": lambda row, idx, stats: convert_lcb(row, stats, name="lcb_rest", cutoff=None)}


def _write_tests_file(tests_dir: Path, qkey: str, tests: list[dict]) -> tuple[str, int]:
    data, sha = code_pool.tests_payload(qkey, tests)
    path = tests_dir / f"{qkey}.json"
    if path.exists() and path.read_bytes() != data:
        raise SystemExit(f"tests file {path} exists with different content (qkey collision across sets?)")
    path.write_bytes(data)
    return sha, len(tests)


def _convert_benchmark(name: str, rows, stats: dict, tests_dir: Path):
    seen: set[str] = set()
    convert = CONVERTERS[name]
    for idx, row in enumerate(rows):
        stats["n_rows_raw"] += 1
        converted = convert(row, idx, stats)
        if converted is None:
            continue
        question, tests, extra = converted
        qkey = question_key(question)
        if qkey in seen:
            stats["n_dropped_duplicate"] += 1
            continue
        seen.add(qkey)
        sha, n_tests = _write_tests_file(tests_dir, qkey, tests)
        stats["n_kept"] += 1
        yield {"source": name, "source_id": extra["source_id"], "qkey": qkey, "question": question,
               "test_kind": extra["test_kind"], "entry_point": extra["entry_point"],
               "fn_name": extra["fn_name"], "tests_sha256": sha, "n_tests": n_tests,
               "level": extra["level"], "contest_date": extra["contest_date"],
               "strict_post_qwen3": extra["strict_post_qwen3"], "benchmark": name}


def _convert_synthetic2(rows, stats: dict, key_sets: Counter):
    for row in rows:
        stats["n_rows_raw"] += 1
        pid = str(_col("synthetic2", row, "problem_id"))
        if not pid.startswith(SYNTHETIC2_CODE_PREFIX):
            stats["n_non_code"] += 1
            continue
        info = _col("synthetic2", row, "verification_info")
        info = info if isinstance(info, str) else json.dumps(info)
        if len(info) > SYNTHETIC2_MAX_INFO_CHARS:
            stats["n_dropped_tests_too_large"] += 1
            continue
        try:
            parsed = json.loads(info)
        except ValueError:
            stats["n_dropped_unparseable"] += 1
            continue
        key_sets[",".join(sorted(parsed))] += 1
        cases = parsed.get("test_cases") if isinstance(parsed, dict) else None
        has_fn = isinstance(cases, dict) and bool(cases.get("fn_name"))
        stats["n_with_fn_name" if has_fn else "n_without_fn_name"] += 1
        stats["n_kept"] += 1
        yield {"source": "synthetic2", "source_id": pid, "prompt": str(_col("synthetic2", row, "prompt")),
               "verification_info": info,
               **{ours: row.get(theirs) for ours, theirs in REWARD_COLUMNS.items()}}


def _convert_synthetic2_sft(rows, stats: dict):
    seen: set[str] = set()
    for row in rows:
        stats["n_rows_raw"] += 1
        pid = str(_col("synthetic2_sft", row, "problem_id"))
        if not pid.startswith(SYNTHETIC2_CODE_PREFIX):
            stats["n_non_code"] += 1
            continue
        messages = _col("synthetic2_sft", row, "messages") or []
        replies = [m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
        reply = str(replies[-1]) if replies else ""
        program = code_pool.last_python_block(reply.rsplit("</think>", 1)[-1]) or code_pool.last_python_block(reply)
        if program is None:
            stats["n_no_program"] += 1
            continue
        if pid in seen:
            stats["n_dropped_duplicate"] += 1
            continue
        seen.add(pid)
        stats["n_kept"] += 1
        yield {"source": "synthetic2_sft", "source_id": pid, "reward": row.get("reward"),
               "task_type": row.get("task_type"), "program": program}


def _lcb_rows(revision: str | None, hf_cache: str | None, files: tuple = LCB_FILES):
    from huggingface_hub import hf_hub_download

    for filename in files:
        path = hf_hub_download("livecodebench/code_generation_lite", filename, repo_type="dataset",
                               revision=revision, cache_dir=hf_cache)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield {**json.loads(line), "_file": filename}


def fetch_set(name: str, cache_dir: Path, tests_dir: Path, *, rows=None, hf_cache: str | None = None) -> None:
    spec = CODE_SETS[name]
    cache_dir, tests_dir = Path(cache_dir), Path(tests_dir)
    cache_name = f"code_{name}"
    if _cached(cache_dir, cache_name):
        print(json.dumps({"fetch": name, "status": "skipped: cached"}), flush=True)
        return
    revision = dataset_revision(spec["dataset"])
    from_hub = rows is None
    if from_hub:
        if name in ("lcb_post", "lcb_rest"):
            rows = _lcb_rows(revision, hf_cache, files=CODE_SETS[name]["files"])
        else:
            rows = iter_hub_rows(spec["dataset"], spec["config"], spec["split"], revision=revision)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{cache_name}.jsonl"
    if name == "synthetic2":
        stats = {"n_rows_raw": 0, "n_kept": 0, "n_non_code": 0, "n_dropped_tests_too_large": 0,
                 "n_dropped_unparseable": 0, "n_with_fn_name": 0, "n_without_fn_name": 0}
        key_sets: Counter = Counter()
        n_rows = _write_rows_atomically(out, _convert_synthetic2(rows, stats, key_sets))
        extra = {**stats, "verification_key_sets": dict(key_sets)}
    elif name == "synthetic2_sft":
        stats = {"n_rows_raw": 0, "n_kept": 0, "n_non_code": 0, "n_no_program": 0, "n_dropped_duplicate": 0}
        n_rows = _write_rows_atomically(out, _convert_synthetic2_sft(rows, stats))
        extra = dict(stats)
    else:
        stats = {k: 0 for k in LCB_STATS}
        n_rows = _write_rows_atomically(out, _convert_benchmark(name, rows, stats, tests_dir))
        extra = dict(stats)
    if from_hub and stats["n_rows_raw"] == 0:
        out.unlink()
        raise SystemExit(f"{name}: the hub stream yielded no rows; refusing to cache an empty set")
    if n_rows == 0 and stats["n_rows_raw"] > 0:
        out.unlink()
        raise SystemExit(f"{name}: no rows survived conversion (n_rows_raw={stats['n_rows_raw']}); "
                         "a column may be missing or renamed; refusing to cache")
    meta = write_meta(cache_dir / f"{cache_name}.meta.json", name=name, kind=spec["kind"],
                      dataset=spec["dataset"], config=spec.get("config"),
                      split=spec.get("split") or ",".join(spec.get("files", ())), revision=revision,
                      n_rows=n_rows, field_map=spec["field_map"], status="ok",
                      tests_dir=str(tests_dir.resolve()), **extra)
    print(json.dumps({"fetch": name, **{k: meta[k] for k in ("n_rows", "revision", "status")}}), flush=True)


def _parse_sets(value: str | None) -> list[str]:
    if not value:
        return list(CODE_SETS)
    names = [n.strip() for n in value.split(",") if n.strip()]
    unknown = [n for n in names if n not in CODE_SETS]
    if unknown:
        raise SystemExit(f"unknown set(s) {unknown}; choose from {list(CODE_SETS)}")
    return names


def cmd_fetch(args: argparse.Namespace) -> int:
    for name in _parse_sets(args.sets):
        fetch_set(name, Path(args.cache_dir), Path(args.tests_dir), hf_cache=args.hf_cache)
    return 0


_CODE_SUFFIX_MARKERS = (code_pool.STDIN_SUFFIX,
                        "\n\nUse this starter code and implement ",
                        "\n\nImplement the function ",
                        "Complete the function ")


def decontam_text(question: str) -> str:
    for marker in _CODE_SUFFIX_MARKERS:
        cut = question.rfind(marker)
        if cut != -1:
            return question[:cut]
    return question


def build_rows(cached: list[dict], *, name: str, tests_dir: Path, chains: int,
               max_question_chars: int, exclude_qkeys=frozenset(), exclude_sha256s=frozenset(),
               decontam_index=None, decontam_exact=None,
               tranche: int = TRANCHE) -> tuple[list[dict], dict, list[dict]]:
    counts = {"cached_rows": len(cached), "kept": 0, "dropped_empty": 0, "dropped_too_long": 0,
              "dropped_duplicate": 0, "strict_post_qwen3": 0, "by_test_kind": {},
              "dropped_excluded": {"qkey": 0, "question_sha256": 0}, "dropped_decontam": {}}
    seen: set[str] = set()
    rows: list[dict] = []
    hits: list[dict] = []
    for raw in cached:
        question = str(raw["question"])
        if not question.strip():
            counts["dropped_empty"] += 1
            continue
        if len(question) > max_question_chars:
            counts["dropped_too_long"] += 1
            continue
        qkey = question_key(question)
        if qkey != raw["qkey"]:
            raise SystemExit(f"{name}: cached qkey {raw['qkey']} != question_key(question) {qkey}")
        if qkey in seen:
            counts["dropped_duplicate"] += 1
            continue
        question_sha256 = hashlib.sha256(question.encode()).hexdigest()
        if qkey in exclude_qkeys:
            counts["dropped_excluded"]["qkey"] += 1
            continue
        if question_sha256 in exclude_sha256s:
            counts["dropped_excluded"]["question_sha256"] += 1
            continue
        matched = (decontam_exact or {}).get(qkey)
        if matched is None and decontam_index is not None:
            statement = decontam_text(question)
            matched = decontam_index.overlaps(statement) if statement.strip() else None
        if matched is not None:
            counts["dropped_decontam"][matched] = counts["dropped_decontam"].get(matched, 0) + 1
            hits.append({"qkey": qkey, "source": name, "source_id": raw.get("source_id"),
                         "matched": matched})
            continue
        seen.add(qkey)
        tests_file = tests_dir / f"{qkey}.json"
        if not tests_file.exists():
            raise SystemExit(f"{name}: missing tests file {tests_file}")
        sha = hashlib.sha256(tests_file.read_bytes()).hexdigest()
        if sha != raw["tests_sha256"]:
            raise SystemExit(f"{name}: tests_sha256 mismatch for {qkey}: cache {raw['tests_sha256']} != file {sha}")
        counts["kept"] += 1
        counts["strict_post_qwen3"] += int(bool(raw.get("strict_post_qwen3")))
        counts["by_test_kind"][raw["test_kind"]] = counts["by_test_kind"].get(raw["test_kind"], 0) + 1
        rows.append({
            "qkey": qkey, "question": question,
            "question_sha256": question_sha256,
            "gold": 1, "split": SPLIT, "target_chains": int(chains), "allocation_subset": False,
            "source": name, "source_id": raw.get("source_id"), "level": raw.get("level"),
            "tranche": int(tranche), "domain": "code", "answer_format": "code",
            "test_kind": raw["test_kind"], "tests_sha256": raw["tests_sha256"],
            "n_tests": int(raw["n_tests"]), "benchmark": name,
            "strict_post_qwen3": raw.get("strict_post_qwen3"), "contest_date": raw.get("contest_date"),
        })
    rows.sort(key=lambda r: r["qkey"])
    for row in rows:
        assert tuple(row) == MANIFEST_FIELDS
    return rows, counts, hits


def cmd_build_ood(args: argparse.Namespace) -> int:
    name = args.set
    if name not in CODE_SETS or CODE_SETS[name]["kind"] != "benchmark":
        raise SystemExit(f"--set must be a benchmark in {[n for n, s in CODE_SETS.items() if s['kind'] == 'benchmark']}")
    cache_dir, tests_dir = Path(args.cache_dir), Path(args.tests_dir)
    manifest_out, contract_out = Path(args.manifest_out), Path(args.contract_out)
    for path in (manifest_out, contract_out):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing {path}")
    cache_path = cache_dir / f"code_{name}.jsonl"
    if not cache_path.exists():
        raise SystemExit(f"missing cached set {name} ({cache_path}); run `fetch` first")
    cached = _read_jsonl(cache_path)
    meta = _read_meta(cache_dir / f"code_{name}.meta.json")
    record = _Recorder(cache_path)
    exclude_qkeys: set[str] = set()
    exclude_sha256s: set[str] = set()
    exclude_records: list[dict] = []
    for path in getattr(args, "exclude_manifest", []) or []:
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
    for path in getattr(args, "decontam_manifest", []) or []:
        prev = _read_jsonl(Path(path))
        tag = Path(path).name
        for prev_row in prev:
            question = str(prev_row.get("question") or "")
            if not question.strip():
                continue
            decontam_exact.setdefault(question_key(question), tag)
            statement = decontam_text(question)
            if statement.strip():
                decontam_index.add(statement, tag)
        decontam_records.append({**record(Path(path)), "n_questions": len(prev), "tag": tag})

    rows, counts, decontam_hits = build_rows(
        cached, name=name, tests_dir=tests_dir, chains=args.chains,
        max_question_chars=args.max_question_chars,
        exclude_qkeys=exclude_qkeys, exclude_sha256s=exclude_sha256s,
        decontam_index=decontam_index if decontam_records else None,
        decontam_exact=decontam_exact, tranche=args.tranche)
    if not rows:
        raise SystemExit(f"{name}: no rows survive ({counts}); refusing to write a manifest")

    qkeys = {r["qkey"] for r in rows}
    pool_manifests: list[dict] = []
    for path in args.pool_manifest:
        prev = _read_jsonl(Path(path))
        overlap = sum(str(r.get("qkey")) in qkeys for r in prev)
        if overlap:
            raise SystemExit(f"code manifest overlaps the pool manifest {path} on {overlap} question(s); "
                             "the decontamination invariant is broken -- stopping")
        pool_manifests.append({**record(Path(path)), "n_questions": len(prev), "overlap": 0})

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomically(manifest_out, _manifest_text(rows))
    manifest_sha256 = _sha256_file(manifest_out)
    tests_dir_sha256 = hashlib.sha256(
        "".join(f"{r['qkey']} {r['tests_sha256']}\n" for r in rows).encode()).hexdigest()
    contract = {
        "contract_version": CONTRACT_VERSION,
        "status": "FROZEN_BEFORE_GENERATION",
        "report_generated": False,
        "set": name,
        "split": SPLIT,
        "tranche": int(args.tranche),
        "salt": args.salt,
        "allocation_subset": False,
        "domain": "code",
        "answer_format": "code",
        "probe": dict(PROBE),
        "chains_per_question": int(args.chains),
        "max_question_chars": int(args.max_question_chars),
        "lcb_cutoffs": ({"post": LCB_POST_CUTOFF, "strict": LCB_STRICT_CUTOFF} if name == "lcb_post"
                        else {"post": None, "strict": LCB_STRICT_CUTOFF, "files": list(LCB_ALL_FILES)}
                        if name == "lcb_rest" else None),
        "counts": counts,
        "total_questions": len(rows),
        "total_chains": sum(r["target_chains"] for r in rows),
        "manifest_path": str(manifest_out.resolve()),
        "manifest_sha256": manifest_sha256,
        "tests_dir": str(tests_dir.resolve()),
        "tests_dir_sha256": tests_dir_sha256,
        "pool_manifests": pool_manifests,
        "exclude_manifests": exclude_records,
        "n_excluded_qkeys": len(exclude_qkeys),
        "decontamination": {
            "ngram_n": decontam_index.n,
            "manifests": decontam_records,
            "n_indexed_questions": len(decontam_index),
            "n_hits": len(decontam_hits),
            "hits_per_manifest": dict(sorted(counts["dropped_decontam"].items())),
        },
        "set_record": {"name": name, "dataset": CODE_SETS[name]["dataset"], **record(cache_path), "meta": meta},
        "provenance": {"manifest": record(manifest_out), "exclude_manifests": exclude_records,
                       "decontam_manifests": decontam_records},
    }
    if getattr(args, "decontam_report_out", None):
        _write_text_atomically(Path(args.decontam_report_out), json.dumps(
            {"salt": args.salt, "ngram_n": decontam_index.n, "manifest_sha256": manifest_sha256,
             "n_hits": len(decontam_hits), "manifests": decontam_records, "hits": decontam_hits},
            indent=2, sort_keys=True) + "\n")
    try:
        _write_text_atomically(contract_out, json.dumps(contract, indent=2, sort_keys=True) + "\n")
    except BaseException:
        manifest_out.unlink(missing_ok=True)
        raise
    print(json.dumps({"set": name, "kept": counts["kept"], "total_chains": contract["total_chains"],
                      "excluded": dict(counts["dropped_excluded"]), "decontam_hits": len(decontam_hits),
                      "by_test_kind": counts["by_test_kind"], "manifest": str(manifest_out),
                      "manifest_sha256": manifest_sha256, "tests_dir_sha256": tests_dir_sha256,
                      "contract": str(contract_out)}, sort_keys=True), flush=True)
    return 0


CANDIDATE_FIELDS = MANIFEST_FIELDS + ("qwen3_4b_reward", "qwen3_32b_reward", "r1_qwen3_8b_reward",
                                      "solution_verified")
CANDIDATE_SPLIT = "fit"
PILOT_CHAINS = 2
DEFAULT_BAND = "0.1:0.9"
DEFAULT_CANDIDATES = 3000
SYNTHETIC2_MIN_STRIPPED = 0.99
SOLUTION_FAILURE_WARNING = 0.03
BENCHMARK_SETS = ("humanevalplus", "mbppplus", "lcb_post")


def _parse_band(text: str) -> tuple[float, float]:
    try:
        lo, hi = (float(x) for x in text.split(":"))
    except ValueError:
        raise SystemExit(f"--band must be lo:hi, got {text!r}") from None
    if not 0.0 <= lo < hi <= 1.0:
        raise SystemExit(f"--band must satisfy 0 <= lo < hi <= 1, got {text!r}")
    return lo, hi


def _in_band(value, lo: float, hi: float) -> bool:
    return value is not None and lo < float(value) < hi


def _mem_bytes(mem_gib: float) -> int:
    return int(mem_gib * (1 << 20)) << 10


def load_eval_index(cache_dir: Path) -> tuple[pool.NGramIndex, dict[str, str], list[dict]]:
    index, exact, status = pool.NGramIndex(), {}, []
    for name in BENCHMARK_SETS:
        path = cache_dir / f"code_{name}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing cached benchmark {name} ({path}); the decontamination gate needs all of "
                             f"{BENCHMARK_SETS} (run `fetch` first)")
        n = 0
        for row in _read_jsonl(path):
            question = str(row["question"])
            exact.setdefault(question_key(question), name)
            index.add(question, name)
            n += 1
        status.append({"name": name, "n_texts": n, "path": str(path)})
    return index, exact, status


def _figure(statement: str) -> bool:
    return pool.has_figure(statement) or "<image>" in statement.lower()


ANNOTATION_COLUMNS = ("qwen3_4b_reward", "qwen3_32b_reward", "r1_qwen3_8b_reward")


def _peek_fn_name(info) -> bool:
    try:
        parsed = json.loads(info) if isinstance(info, str) else info
        cases = parsed.get("test_cases") if isinstance(parsed, dict) else None
        cases = json.loads(cases) if isinstance(cases, str) else cases
        return isinstance(cases, dict) and bool(cases.get("fn_name"))
    except (ValueError, TypeError, AttributeError):
        return False


def _unannotated(raw: dict) -> bool:
    return _peek_fn_name(raw.get("verification_info")) and all(
        (raw.get(k) is None or float(raw.get(k)) == 0.0) for k in ANNOTATION_COLUMNS)


def candidate_rows(cached: list[dict], *, band: tuple[float, float], tests_dir: Path, eval_index, eval_exact,
                   max_question_chars: int, solutions: dict[str, str],
                   include_unannotated: bool = False) -> tuple[list[dict], dict, list[dict]]:
    counts = {"cached_rows": len(cached), "reward_missing": 0, "outside_band": 0, "in_band": 0,
              "unannotated": 0, "unannotated_included": 0,
              "not_stripped": 0, "dropped_empty": 0, "dropped_figure": 0, "dropped_tests": 0,
              "dropped_tests_selection": 0, "dropped_too_long": 0, "dropped_duplicate": 0,
              "dropped_decontam": {}, "eligible": 0, "by_test_kind": {}}
    hits: list[dict] = []
    seen: set[str] = set()
    out: list[dict] = []
    lo, hi = band
    for raw in cached:
        reward = raw.get("qwen3_4b_reward")
        if reward is None:
            counts["reward_missing"] += 1
        unannotated = _unannotated(raw)
        counts["unannotated"] += int(unannotated)
        if unannotated and include_unannotated:
            counts["unannotated_included"] += 1
        elif not _in_band(reward, lo, hi):
            counts["outside_band"] += 1
            continue
        else:
            counts["in_band"] += 1
        statement, stripped = code_pool.strip_synthetic2_instruction(str(raw["prompt"]))
        counts["not_stripped"] += int(not stripped)
        if not statement:
            counts["dropped_empty"] += 1
            continue
        if _figure(statement):
            counts["dropped_figure"] += 1
            continue
        tests = code_pool.normalise_tests("synthetic2", raw["verification_info"])
        if tests is None:
            counts["dropped_tests"] += 1
            continue
        selected = code_pool.select_tests(tests)
        if selected is None:
            counts["dropped_tests_selection"] += 1
            continue
        kind = selected[0]["kind"]
        fn_name = selected[0].get("fn_name") if kind == "call" else None
        question = code_pool.compose_question(statement, test_kind=kind, fn_name=fn_name)
        if len(question) > max_question_chars:
            counts["dropped_too_long"] += 1
            continue
        qkey = question_key(question)
        if qkey in seen:
            counts["dropped_duplicate"] += 1
            continue
        seen.add(qkey)
        matched = eval_exact.get(qkey) or eval_index.overlaps(statement)
        if matched is not None:
            counts["dropped_decontam"][matched] = counts["dropped_decontam"].get(matched, 0) + 1
            hits.append({"qkey": qkey, "source": "synthetic2", "source_id": raw["source_id"],
                         "matched_eval_set": matched})
            continue
        sha, n_tests = _write_tests_file(tests_dir, qkey, selected)
        counts["eligible"] += 1
        counts["by_test_kind"][kind] = counts["by_test_kind"].get(kind, 0) + 1
        out.append({"qkey": qkey, "question": question, "source_id": str(raw["source_id"]), "test_kind": kind,
                    "tests_sha256": sha, "n_tests": n_tests, "tests": selected,
                    "qwen3_4b_reward": reward, "qwen3_32b_reward": raw.get("qwen3_32b_reward"),
                    "r1_qwen3_8b_reward": raw.get("r1_qwen3_8b_reward"),
                    "program": solutions.get(str(raw["source_id"]))})
    return out, counts, hits


def verify_solutions(cands: list[dict], *, workers: int, limits, prefix: list[str]) -> dict:
    from rl_controller.iclr.code_exec import check_sandbox, grade_many
    items = [(c["qkey"], c["program"], c["tests"]) for c in cands if c["program"] is not None]
    try:
        if items:
            verdicts = grade_many(items, workers=workers, limits=limits, prefix=prefix)
        else:
            check_sandbox(limits, prefix=prefix)
            verdicts = {}
    except (RuntimeError, ValueError) as err:
        raise SystemExit(f"reference-solution verification aborted: {err}") from None
    failed: dict[str, int] = {}
    keep = []
    for c in cands:
        if c["program"] is None:
            c["solution_verified"] = None
            keep.append(c)
            continue
        status = verdicts[c["qkey"]].status
        if status == "grader_error":
            raise SystemExit(f"reference-solution verification: grader fault on {c['qkey']} "
                             f"({verdicts[c['qkey']].first_failure}); the sandbox, not the data, is broken -- stopping")
        if status == "pass":
            c["solution_verified"] = True
            keep.append(c)
        else:
            failed[status] = failed.get(status, 0) + 1
    cands[:] = keep
    n_with = len(items)
    n_failed = sum(failed.values())
    return {"n_with_program": n_with, "solution_verified": n_with - n_failed,
            "dropped_solution_failed": failed, "solution_exempt": sum(c["program"] is None for c in cands),
            "solution_failure_rate": (n_failed / n_with) if n_with else 0.0,
            "solution_failure_warning": bool(n_with) and (n_failed / n_with) > SOLUTION_FAILURE_WARNING}


def _candidate_manifest_row(c: dict) -> dict:
    question = c["question"]
    row = {"qkey": c["qkey"], "question": question,
           "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
           "gold": 1, "split": CANDIDATE_SPLIT, "target_chains": PILOT_CHAINS, "allocation_subset": False,
           "source": "synthetic2", "source_id": c["source_id"], "level": None, "tranche": TRANCHE,
           "domain": "code", "answer_format": "code", "test_kind": c["test_kind"],
           "tests_sha256": c["tests_sha256"], "n_tests": int(c["n_tests"]), "benchmark": None,
           "strict_post_qwen3": None, "contest_date": None,
           "qwen3_4b_reward": c["qwen3_4b_reward"], "qwen3_32b_reward": c["qwen3_32b_reward"],
           "r1_qwen3_8b_reward": c["r1_qwen3_8b_reward"], "solution_verified": c["solution_verified"]}
    assert tuple(row) == CANDIDATE_FIELDS
    return row


def cmd_build_candidates(args: argparse.Namespace) -> int:
    from rl_controller.iclr.code_exec import Limits, detect_net_isolation
    cache_dir, tests_dir = Path(args.cache_dir), Path(args.tests_dir)
    manifest_out, contract_out = Path(args.manifest_out), Path(args.contract_out)
    decontam_out = (Path(args.decontam_report_out) if args.decontam_report_out
                    else contract_out.parent / "decontam_report.json")
    for path in (manifest_out, contract_out, decontam_out):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing {path}")
    band = _parse_band(args.band)
    cache_path = cache_dir / "code_synthetic2.jsonl"
    if not cache_path.exists():
        raise SystemExit(f"missing cached set synthetic2 ({cache_path}); run `fetch` first")
    cached = _read_jsonl(cache_path)
    sft_path = cache_dir / "code_synthetic2_sft.jsonl"
    solutions: dict[str, str] = {}
    if args.verify_solutions:
        if not sft_path.exists():
            raise SystemExit(f"--verify-solutions needs the cached set synthetic2_sft ({sft_path}); run `fetch` first")
        solutions = {str(r["source_id"]): str(r["program"]) for r in _read_jsonl(sft_path)}
    eval_index, eval_exact, eval_status = load_eval_index(cache_dir)
    tests_dir.mkdir(parents=True, exist_ok=True)
    cands, counts, hits = candidate_rows(cached, band=band, tests_dir=tests_dir, eval_index=eval_index,
                                         eval_exact=eval_exact, max_question_chars=args.max_question_chars,
                                         solutions=solutions, include_unannotated=args.include_unannotated)
    n_considered = counts["in_band"] + counts["unannotated_included"]
    stripped_fraction = (1 - counts["not_stripped"] / n_considered) if n_considered else 0.0
    if stripped_fraction < args.min_stripped:
        raise SystemExit(f"stripped fraction {stripped_fraction:.3%} < --min-stripped {args.min_stripped:.0%}: only that "
                         "share of the in-band prompts ends with the SYNTHETIC-2 format instruction; the prompt "
                         "format changed -- stopping before any tests file is used (files written so far are "
                         "content-addressed and harmless)")
    limits = Limits(wall_s=float(args.timeout), cpu_s=int(args.timeout), mem_bytes=_mem_bytes(args.mem_gib))
    prefix = detect_net_isolation() if args.verify_solutions else None
    if args.verify_solutions:
        verification = verify_solutions(cands, workers=args.workers, limits=limits, prefix=prefix)
    else:
        verification = {"n_with_program": 0, "solution_verified": 0, "dropped_solution_failed": {},
                        "solution_exempt": len(cands), "solution_failure_rate": None,
                        "solution_failure_warning": False}
        for c in cands:
            c["solution_verified"] = None
    if not cands:
        raise SystemExit(f"no candidates survive ({counts}); refusing to write a manifest")
    picked = sorted(cands, key=lambda c: pool.rank(args.salt, "cand", c["qkey"]))[: int(args.n)]
    rows = sorted((_candidate_manifest_row(c) for c in picked), key=lambda r: r["qkey"])
    counts.update({k: v for k, v in verification.items() if k in ("solution_verified", "dropped_solution_failed",
                                                                  "solution_exempt")})
    counts["selected"] = len(rows)

    record = _Recorder(cache_path)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomically(manifest_out, _manifest_text(rows))
    manifest_sha256 = _sha256_file(manifest_out)
    contract = {
        "contract_version": CONTRACT_VERSION, "status": "FROZEN_BEFORE_GENERATION", "report_generated": False,
        "kind": "candidates", "set": "synthetic2", "split": CANDIDATE_SPLIT, "tranche": TRANCHE,
        "domain": "code", "answer_format": "code", "chains_per_question": PILOT_CHAINS,
        "probe": {"query_stride": 1000, "max_forced_tokens": PROBE["max_forced_tokens"]},
        "salt": args.salt, "band": list(band), "include_unannotated": bool(args.include_unannotated),
        "n_requested": int(args.n), "band_short": len(cands) < int(args.n),
        "n_eligible": len(cands), "total_questions": len(rows), "total_chains": len(rows) * PILOT_CHAINS,
        "stripped_fraction": stripped_fraction, "min_stripped": args.min_stripped,
        "max_question_chars": int(args.max_question_chars), "counts": counts,
        "n_with_program": verification["n_with_program"],
        "solution_failure_rate": verification["solution_failure_rate"],
        "solution_failure_warning": verification["solution_failure_warning"],
        "verification": {"enabled": bool(args.verify_solutions),
                         "limits": {"wall_s": limits.wall_s, "cpu_s": limits.cpu_s, "mem_bytes": limits.mem_bytes},
                         "net_isolation": prefix, "workers": int(args.workers)},
        "manifest_path": str(manifest_out.resolve()), "manifest_sha256": manifest_sha256,
        "tests_dir": str(tests_dir.resolve()),
        "tests_dir_sha256": hashlib.sha256("".join(f"{r['qkey']} {r['tests_sha256']}\n" for r in rows).encode()).hexdigest(),
        "eval_sets": {s["name"]: {"n_texts": s["n_texts"], **record(Path(s["path"]))} for s in eval_status},
        "decontamination": {"ngram_n": eval_index.n, "n_hits": len(hits),
                            "hits_per_eval_set": dict(sorted(counts["dropped_decontam"].items()))},
        "set_record": {"name": "synthetic2", "dataset": CODE_SETS["synthetic2"]["dataset"], **record(cache_path),
                       "meta": _read_meta(cache_dir / "code_synthetic2.meta.json")},
        "solutions_record": ({"name": "synthetic2_sft", "dataset": CODE_SETS["synthetic2_sft"]["dataset"],
                              **record(sft_path), "meta": _read_meta(cache_dir / "code_synthetic2_sft.meta.json")}
                             if args.verify_solutions else None),
        "provenance": {"manifest": record(manifest_out)},
    }
    try:
        _write_text_atomically(contract_out, json.dumps(contract, indent=2, sort_keys=True) + "\n")
        _write_text_atomically(decontam_out, json.dumps({"salt": args.salt, "ngram_n": eval_index.n, "n_hits": len(hits),
                                                         "hits": hits, "eval_sets": eval_status,
                                                         "manifest_sha256": manifest_sha256}, indent=2, sort_keys=True) + "\n")
    except BaseException:
        manifest_out.unlink(missing_ok=True)
        contract_out.unlink(missing_ok=True)
        raise
    if verification["solution_failure_warning"]:
        print(f"WARNING: {verification['solution_failure_rate']:.1%} of the reference programs failed the sandbox "
              f"(> {SOLUTION_FAILURE_WARNING:.0%}); spec decision 11 says the limits, not the data, are wrong -- "
              "inspect grade statuses before launching the pilot", flush=True)
        if not args.allow_solution_failures:
            for path in (manifest_out, contract_out, decontam_out):
                path.unlink(missing_ok=True)
            raise SystemExit("refusing to seal a candidate set with that failure rate (pass --allow-solution-failures "
                             "after inspecting the statuses)")
    print(json.dumps({"candidates": len(rows), "eligible": len(cands), "band": list(band), "band_short": contract["band_short"],
                      "by_test_kind": counts["by_test_kind"], "decontam_hits": len(hits),
                      "solution_verified": counts["solution_verified"], "solution_failed": counts["dropped_solution_failed"],
                      "solution_exempt": counts["solution_exempt"], "stripped_fraction": round(stripped_fraction, 4),
                      "manifest": str(manifest_out), "manifest_sha256": manifest_sha256, "contract": str(contract_out)},
                     sort_keys=True), flush=True)
    return 0


POOL_FIELDS = MANIFEST_FIELDS + ("pilot_pass", "qwen3_4b_reward")
POOL_TRANCHE = 1
REWARD_BINS = ((0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9))


def pilot_pass_counts(chains: list[dict], *, expected: int = PILOT_CHAINS) -> tuple[dict[str, int], dict[str, list]]:
    by_q: dict[str, list[dict]] = {}
    for c in chains:
        by_q.setdefault(str(c["qkey"]), []).append(c)
    counts, bad = {}, {}
    for q, cs in by_q.items():
        reps = sorted(int(c["rep"]) for c in cs)
        if reps != list(range(expected)):
            bad[q] = reps
            continue
        counts[q] = sum(int(bool(c.get("correct"))) for c in cs)
    return counts, bad


def _reward_bin(value) -> str:
    for lo, hi in REWARD_BINS:
        if value is not None and lo <= float(value) < hi:
            return f"[{lo},{hi})"
    return "other"


def cmd_build(args: argparse.Namespace) -> int:
    from rl_controller.iclr.chains import load_chains
    cand_path, cand_contract_path = Path(args.candidates), Path(args.candidates_contract)
    pilot_path, tests_dir = Path(args.pilot_chains), Path(args.tests_dir)
    manifest_out, contract_out = Path(args.manifest_out), Path(args.contract_out)
    for path in (manifest_out, contract_out):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing {path}")
    for path in (cand_path, cand_contract_path, pilot_path):
        if not path.exists():
            raise SystemExit(f"missing {path}")
    if int(args.pool_seed) == int(args.pilot_seed):
        raise SystemExit("--pool-seed must differ from --pilot-seed: chain reps are seeded by (qkey, rep, seed), so the "
                         "pool's reps 0-1 would be the very pilot draws that ranked the question")
    cand_contract = json.loads(cand_contract_path.read_text(encoding="utf-8"))
    cand_sha = _sha256_file(cand_path)
    if cand_contract.get("manifest_sha256") != cand_sha:
        raise SystemExit(f"candidates manifest sha {cand_sha} != contract {cand_contract.get('manifest_sha256')}")
    cands = _read_jsonl(cand_path)
    pilot = load_chains(pilot_path)
    passes, bad = pilot_pass_counts(pilot)
    missing = [c["qkey"] for c in cands if c["qkey"] not in passes]
    if missing and not args.allow_missing_pilot:
        raise SystemExit(f"{len(missing)} candidate(s) lack a complete {PILOT_CHAINS}-chain pilot (e.g. {missing[:3]}; "
                         f"{len(bad)} questions with partial reps); merge every pilot shard or pass --allow-missing-pilot")
    usable = [c for c in cands if c["qkey"] in passes]
    order = code_pool.pilot_order([c["qkey"] for c in usable], passes, args.salt)
    position = {k: i for i, k in enumerate(order)}
    pool_cands = [{"source": "synthetic2", "source_id": c["source_id"], "question": c["question"], "gold": 1,
                   "level": None, "test_kind": c["test_kind"], "tests_sha256": c["tests_sha256"], "n_tests": c["n_tests"]}
                  for c in usable]
    rows, report = code_pool.build_code_pool(pool_cands, quota=int(args.n), salt=args.salt,
                                             rank_fn=position.__getitem__, tranche=POOL_TRANCHE,
                                             max_question_chars=args.max_question_chars)
    src = report["sources"]["synthetic2"]
    def _nonzero(v):
        return any(_nonzero(x) for x in v.values()) if isinstance(v, dict) else bool(v)
    redropped = {k: v for k, v in src.items() if k.startswith("dropped_") and _nonzero(v)}
    if src["eligible"] != len(usable) or redropped:
        raise SystemExit(f"build_pool dropped candidates that build-candidates had kept ({redropped}); "
                         f"eligible {src['eligible']} != usable {len(usable)} -- the filters (max_question_chars) differ")
    by_key = {c["qkey"]: c for c in usable}
    for row in rows:
        c = by_key[row["qkey"]]
        tests_file = tests_dir / f"{row['qkey']}.json"
        if not tests_file.exists():
            raise SystemExit(f"missing tests file {tests_file}")
        sha = hashlib.sha256(tests_file.read_bytes()).hexdigest()
        if sha != row["tests_sha256"]:
            raise SystemExit(f"tests_sha256 mismatch for {row['qkey']}: candidates {row['tests_sha256']} != file {sha}")
        row["pilot_pass"] = int(passes[row["qkey"]])
        row["qwen3_4b_reward"] = c["qwen3_4b_reward"]
        assert tuple(row) == POOL_FIELDS
    if not rows:
        raise SystemExit("no rows survive; refusing to write a manifest")

    histogram = {str(k): sum(1 for c in usable if passes[c["qkey"]] == k) for k in (0, 1, 2)}
    cross = {str(k): {} for k in (0, 1, 2)}
    for c in usable:
        b_ = _reward_bin(c["qwen3_4b_reward"])
        cross[str(passes[c["qkey"]])][b_] = cross[str(passes[c["qkey"]])].get(b_, 0) + 1
    selected_by_pass = {str(k): sum(1 for r in rows if r["pilot_pass"] == k) for k in (0, 1, 2)}
    pilot_chains_used = [ch for ch in pilot if ch["qkey"] in by_key]
    natural = [ch for ch in pilot_chains_used if ch.get("outcome") == "natural"]
    pilot_stats = {"chains_path": str(pilot_path.resolve()), "chains_sha256": _sha256_file(pilot_path),
                   "n_chains": len(pilot), "n_questions": len(passes), "questions_with_partial_reps": len(bad),
                   "candidates_without_pilot": len(missing), "histogram": histogram, "cross_table": cross,
                   "selected_by_pass": selected_by_pass,
                   "pilot_pass_rate": (sum(bool(ch.get("correct")) for ch in pilot_chains_used) / len(pilot_chains_used)
                                       if pilot_chains_used else None),
                   "pilot_seed": int(args.pilot_seed), "pool_seed": int(args.pool_seed),
                   "natural_fraction": (len(natural) / len(pilot_chains_used)) if pilot_chains_used else None,
                   "mean_thinking_tokens": (sum(int(ch.get("thinking_tokens", 0)) for ch in pilot_chains_used)
                                            / len(pilot_chains_used)) if pilot_chains_used else None}

    record = _Recorder(cand_path)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomically(manifest_out, _manifest_text(rows))
    manifest_sha256 = _sha256_file(manifest_out)
    body = {k: v for k, v in report.items() if k != "decontam_hits"}
    contract = {
        "contract_version": CONTRACT_VERSION, "status": "FROZEN_BEFORE_GENERATION", "report_generated": False,
        "kind": "pool", "set": "synthetic2", "tranche": POOL_TRANCHE, "domain": "code", "answer_format": "code",
        "probe": dict(PROBE), "salt": args.salt, "shares": list(code_pool.CODE_SHARES), "quota": int(args.n),
        "pool_size": len(rows), "split_counts": report["split_counts"], "allocation_split_counts": report["allocation_counts"],
        "base_chains_per_question": 8, "allocation_chains_per_question": 16,
        "allocation_subset_size": sum(report["allocation_counts"].values()),
        "total_chains_all_splits": report["total_chains"], "by_test_kind": dict(sorted(
            (k, sum(1 for r in rows if r["test_kind"] == k)) for k in {r["test_kind"] for r in rows})),
        "max_question_chars": int(args.max_question_chars),
        "candidates": {"path": str(cand_path.resolve()), "manifest_sha256": cand_sha, "n": len(cands),
                       "contract_path": str(cand_contract_path.resolve()), "contract_sha256": _sha256_file(cand_contract_path),
                       "band": cand_contract.get("band"), "band_short": cand_contract.get("band_short")},
        "pilot": pilot_stats,
        "generation_seed": int(args.pool_seed),
        "manifest_path": str(manifest_out.resolve()), "manifest_sha256": manifest_sha256,
        "tests_dir": str(tests_dir.resolve()),
        "tests_dir_sha256": hashlib.sha256("".join(f"{r['qkey']} {r['tests_sha256']}\n" for r in rows).encode()).hexdigest(),
        "report": body,
        "provenance": {"manifest": record(manifest_out), "candidates": record(cand_path), "pilot_chains": record(pilot_path)},
    }
    try:
        _write_text_atomically(contract_out, json.dumps(contract, indent=2, sort_keys=True) + "\n")
    except BaseException:
        manifest_out.unlink(missing_ok=True)
        raise
    print(json.dumps({"pool_size": len(rows), "split_counts": report["split_counts"], "total_chains": report["total_chains"],
                      "pilot_histogram": histogram, "selected_by_pass": selected_by_pass,
                      "pilot_pass_rate": pilot_stats["pilot_pass_rate"], "by_test_kind": contract["by_test_kind"],
                      "manifest": str(manifest_out), "manifest_sha256": manifest_sha256, "contract": str(contract_out)},
                     sort_keys=True), flush=True)
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="materialise the code sets from the HF hub")
    fetch.add_argument("--cache-dir", required=True)
    fetch.add_argument("--tests-dir", required=True, help="where <qkey>.json tests files go")
    fetch.add_argument("--sets", default=None, help=f"comma-separated subset of {list(CODE_SETS)}")
    fetch.add_argument("--hf-cache", default=None, help="huggingface_hub cache_dir for the raw LCB files")

    build = sub.add_parser("build-ood", help="one sealed report-only manifest for one benchmark")
    build.add_argument("--cache-dir", required=True)
    build.add_argument("--tests-dir", required=True)
    build.add_argument("--set", required=True)
    build.add_argument("--manifest-out", required=True)
    build.add_argument("--contract-out", required=True)
    build.add_argument("--chains", type=int, default=DEFAULT_CHAINS)
    build.add_argument("--pool-manifest", action="append", default=[], metavar="MANIFEST",
                       help="a training manifest that must NOT share any qkey with this one (repeatable)")
    build.add_argument("--max-question-chars", type=int, default=DEFAULT_MAX_QUESTION_CHARS)
    build.add_argument("--exclude-manifest", action="append", default=[], metavar="MANIFEST",
                       help="an earlier manifest whose questions are already spent: every matching "
                            "qkey / question_sha256 is dropped and counted (repeatable)")
    build.add_argument("--decontam-manifest", action="append", default=[], metavar="MANIFEST",
                       help="a training-pool manifest to decontaminate against: exact question_key "
                            "or a shared 13-gram drops the question (repeatable)")
    build.add_argument("--decontam-report-out", default=None, metavar="FILE",
                       help="write the full per-question decontamination hit list here")
    build.add_argument("--salt", default=None, help="provenance salt recorded in the contract")
    build.add_argument("--tranche", type=int, default=TRANCHE,
                       help=f"tranche stamped on every row (default {TRANCHE})")

    cand = sub.add_parser("build-candidates", help="in-band SYNTHETIC-2 candidates for the 2-chain pilot")
    cand.add_argument("--cache-dir", required=True)
    cand.add_argument("--tests-dir", required=True)
    cand.add_argument("--band", default=DEFAULT_BAND, help="open Qwen3-4B avg_reward interval lo:hi")
    cand.add_argument("--allow-solution-failures", action="store_true",
                      help="seal the candidates even when > 3 %% of the reference programs fail the sandbox")
    cand.add_argument("--include-unannotated", action="store_true",
                      help="rows without a difficulty annotation (qwen3_32b_reward null; all the function-call "
                           "problems) bypass the band -- the 1.7B pilot is their difficulty filter"),
    cand.add_argument("--n", type=int, default=DEFAULT_CANDIDATES, help="candidate cap (every eligible row when fewer)")
    cand.add_argument("--salt", required=True)
    cand.add_argument("--manifest-out", required=True)
    cand.add_argument("--contract-out", required=True)
    cand.add_argument("--decontam-report-out", default=None)
    cand.add_argument("--verify-solutions", action="store_true", help="sandbox the SFT-verified reference programs")
    cand.add_argument("--workers", type=int, default=8)
    cand.add_argument("--timeout", type=float, default=10.0)
    cand.add_argument("--mem-gib", type=float, default=1.0)
    cand.add_argument("--max-question-chars", type=int, default=DEFAULT_MAX_QUESTION_CHARS)
    cand.add_argument("--min-stripped", type=float, default=SYNTHETIC2_MIN_STRIPPED)

    build_pool = sub.add_parser("build", help="pilot-ordered 1 500-question code pool with fit/select/report splits")
    build_pool.add_argument("--candidates", required=True)
    build_pool.add_argument("--candidates-contract", required=True)
    build_pool.add_argument("--pilot-chains", required=True, help="merged pilot chains.jsonl (2 chains per candidate)")
    build_pool.add_argument("--tests-dir", required=True)
    build_pool.add_argument("--n", type=int, default=1500)
    build_pool.add_argument("--salt", required=True)
    build_pool.add_argument("--manifest-out", required=True)
    build_pool.add_argument("--contract-out", required=True)
    build_pool.add_argument("--allow-missing-pilot", action="store_true")
    build_pool.add_argument("--pilot-seed", type=int, default=0, help="the --seed the pilot chains were generated with")
    build_pool.add_argument("--pool-seed", type=int, default=1,
                            help="the --seed the pool MUST be generated with (recorded; refused when equal to --pilot-seed)")
    build_pool.add_argument("--max-question-chars", type=int, default=DEFAULT_MAX_QUESTION_CHARS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "fetch":
        return cmd_fetch(args)
    if args.command == "build-candidates":
        return cmd_build_candidates(args)
    if args.command == "build":
        return cmd_build(args)
    return cmd_build_ood(args)


if __name__ == "__main__":
    raise SystemExit(main())
