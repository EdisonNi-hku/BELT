"""Build the multi-source integer-answer math pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path

from rl_controller.iclr import pool
from rl_controller.iclr.chains import write_jsonl
from rl_controller.iclr.provenance import REPO_ROOT, source_record
from rl_controller.splits import question_key

DEFAULT_SPEC_B_MANIFEST = (
    REPO_ROOT / "docs/superpowers/contracts/2026-08-21-native-1p7b-pool-scaleup-manifest.jsonl")
DEFAULT_EXCLUSIONS = (
    REPO_ROOT / "docs/superpowers/contracts/2026-08-26-iclr-p1-exclusions.json.txt")
DEFAULT_SALT = "iclr-p1-2026-08-26-v1"
SPEC_B_MIN_REFOUND = 0.99
DAPO_MIN_STRIPPED = 0.99

DAPO_LEAD = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $Answer (without quotes) where "
    "$Answer is the answer to the problem."
)
DAPO_TAIL = 'Remember to put your answer on its own line after "Answer:".'
NUMINA_HARD_SOURCES = {"olympiads", "amc_aime", "aops_forum"}

SOURCES = {
    "dapo": {"dataset": "BytedTsinghua-SIA/DAPO-Math-17k", "config": None, "split": "train",
             "field_map": {"question": "prompt[0].content (instruction stripped)",
                           "gold_raw": "reward_model.ground_truth", "level": None,
                           "source_id": "row index"}},
    "numina_hard": {"dataset": "AI-MO/NuminaMath-1.5", "config": None, "split": "train",
                    "field_map": {"question": "problem", "gold_raw": "answer",
                                  "level": "problem_type", "source_id": "row index",
                                  "filter": "source in {olympiads, amc_aime, aops_forum} and "
                                            "problem_is_valid == 'Yes'"}},
    "deepscaler": {"dataset": "agentica-org/DeepScaleR-Preview-Dataset", "config": None,
                   "split": "train",
                   "field_map": {"question": "problem", "gold_raw": "answer", "level": None,
                                 "source_id": "row index"}},
}

EVAL_SETS = {
    "math500": {"dataset": "HuggingFaceH4/MATH-500", "config": None, "split": "test",
                "field": "problem"},
    "aime24": {"dataset": "Maxwell-Jia/AIME_2024", "config": None, "split": "train",
               "field": "Problem"},
    "aime25": {"dataset": "yentinglin/aime_2025", "config": None, "split": "train",
               "field": "problem"},
    "olympiadbench": {"dataset": "Hothan/OlympiadBench",
                      "configs": ["OE_TO_maths_en_COMP", "TP_TO_maths_en_COMP"], "split": "train",
                      "field": "question", "context_field": "context"},
    "gaokao23en": {"dataset": "MARIO-Math-Reasoning/Gaokao2023-Math-En", "config": None,
                   "split": "test", "split_fallback": True, "field": "question"},
    "gpqa_diamond": {"dataset": "Idavidrein/gpqa", "config": "gpqa_diamond", "split": "train",
                     "field": "Question", "gated": True},
}


def strip_dapo_instruction(text: str) -> tuple[str, bool, bool]:
    text = text.strip()
    stripped_lead = text.startswith(DAPO_LEAD)
    if stripped_lead:
        text = text[len(DAPO_LEAD):].strip()
    stripped_tail = text.endswith(DAPO_TAIL)
    if stripped_tail:
        text = text[: -len(DAPO_TAIL)].strip()
    return text, stripped_lead, stripped_tail


def write_meta(path: Path, **fields) -> dict:
    meta = {"dataset": None, "config": None, "split": None, "revision": None, "n_rows": None,
            "field_map": {}, "status": None}
    meta.update(fields)
    meta["fetched_at"] = datetime.now(timezone.utc).isoformat()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def dataset_revision(dataset: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(dataset).sha
    except Exception:
        return None


def resolve_split(dataset: str, config: str | None, preferred: str, fallback: bool,
                  revision: str | None = None) -> str:
    if not fallback:
        return preferred
    from datasets import get_dataset_split_names

    names = list(get_dataset_split_names(dataset, config, revision=revision))
    if preferred in names or not names:
        return preferred
    return names[0]


def iter_hub_rows(dataset: str, config: str | None, split: str,
                  revision: str | None = None) -> Iterator[dict]:
    from datasets import load_dataset

    kwargs = {"revision": revision} if revision is not None else {}
    stream = load_dataset(dataset, config, split=split, streaming=True, **kwargs)
    yield from stream


def _write_rows_atomically(path: Path, rows: Iterable[dict]) -> int:
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    return n


def _cached(cache_dir: Path, name: str) -> bool:
    return (cache_dir / f"{name}.jsonl").exists() and (cache_dir / f"{name}.meta.json").exists()


def _convert_source(name: str, rows: Iterable[dict], stats: dict) -> Iterator[dict]:
    seen: set[str] = set()
    for idx, row in enumerate(rows):
        stats["n_rows_raw"] += 1
        if name == "dapo":
            question, lead, tail = strip_dapo_instruction(row["prompt"][0]["content"])
            stats["n_stripped_lead"] += int(lead)
            stats["n_stripped_tail"] += int(tail)
            stats["n_stripped"] += int(lead and tail)
            gold_raw, level = row["reward_model"]["ground_truth"], None
        elif name == "numina_hard":
            if row.get("source") not in NUMINA_HARD_SOURCES:
                continue
            stats["n_hard_source"] += 1
            if row.get("problem_is_valid") != "Yes":
                stats["n_dropped_invalid"] += 1
                continue
            question, gold_raw, level = row["problem"], row["answer"], row.get("problem_type")
        elif name == "deepscaler":
            question, gold_raw, level = row["problem"], row["answer"], None
        else:
            raise ValueError(f"unknown source {name!r}")
        qkey = question_key(question)
        if qkey in seen:
            stats["n_duplicate"] += 1
            continue
        seen.add(qkey)
        stats["n_unique"] += 1
        yield {"source": name, "source_id": idx, "question": question,
               "gold_raw": gold_raw, "level": level}


def fetch_source(name: str, cache_dir: Path, *, rows: Iterable[dict] | None = None) -> None:
    spec = SOURCES[name]
    cache_dir = Path(cache_dir)
    if _cached(cache_dir, name):
        print(json.dumps({"fetch": name, "status": "skipped: cached"}), flush=True)
        return
    revision = dataset_revision(spec["dataset"])
    if rows is None:
        rows = iter_hub_rows(spec["dataset"], spec["config"], spec["split"], revision=revision)
    stats = {"n_rows_raw": 0, "n_unique": 0, "n_duplicate": 0, "n_stripped": 0,
             "n_stripped_lead": 0, "n_stripped_tail": 0, "n_hard_source": 0, "n_dropped_invalid": 0}
    out = cache_dir / f"{name}.jsonl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_rows = _write_rows_atomically(out, _convert_source(name, rows, stats))
    if name == "dapo" and stats["n_rows_raw"] and stats["n_stripped"] < DAPO_MIN_STRIPPED * stats["n_rows_raw"]:
        out.unlink()
        raise SystemExit(
            f"DAPO wrapper (lead AND tail) stripped on only {stats['n_stripped']}/{stats['n_rows_raw']} "
            f"rows (lead {stats['n_stripped_lead']}, tail {stats['n_stripped_tail']}; "
            f"< {DAPO_MIN_STRIPPED:.0%}): the prompt format changed; refusing to cache")
    if stats["n_unique"] == 0:
        out.unlink()
        raise SystemExit(f"{name}: no rows survived conversion (n_rows_raw={stats['n_rows_raw']}); "
                         "a column such as `problem_is_valid` may be missing or renamed; refusing to cache")
    extra = {k: v for k, v in stats.items()
             if k in ("n_rows_raw", "n_unique", "n_duplicate")
             or (name == "dapo" and k in ("n_stripped", "n_stripped_lead", "n_stripped_tail"))
             or (name == "numina_hard" and k in ("n_hard_source", "n_dropped_invalid"))}
    meta = write_meta(cache_dir / f"{name}.meta.json", name=name, kind="source",
                      dataset=spec["dataset"], config=spec["config"], split=spec["split"],
                      revision=revision, n_rows=n_rows,
                      field_map=spec["field_map"], status="ok", **extra)
    print(json.dumps({"fetch": name, **{k: meta[k] for k in ("n_rows", "revision", "status")}}),
          flush=True)


def _eval_text(spec: dict, row: dict) -> str:
    text = str(row[spec["field"]])
    context_field = spec.get("context_field")
    if context_field:
        context = row.get(context_field)
        if context and str(context) != "None":
            text = f"{context} {text}"
    return text


def fetch_eval_set(name: str, cache_dir: Path, *, rows: Iterable[dict] | None = None,
                   split_used: str | None = None) -> None:
    spec = EVAL_SETS[name]
    cache_dir = Path(cache_dir)
    if _cached(cache_dir, name):
        print(json.dumps({"fetch": name, "status": "skipped: cached"}), flush=True)
        return
    config = spec.get("configs", spec.get("config"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    revision = None
    try:
        if rows is None:
            revision = dataset_revision(spec["dataset"])
            split_used = resolve_split(spec["dataset"], spec.get("config"), spec["split"],
                                       bool(spec.get("split_fallback")), revision=revision)
            configs = spec.get("configs") or [spec.get("config")]
            rows = (row for cfg in configs
                    for row in iter_hub_rows(spec["dataset"], cfg, split_used, revision=revision))
        else:
            revision = dataset_revision(spec["dataset"])
        out = cache_dir / f"{name}.jsonl"
        n_rows = _write_rows_atomically(
            out, ({"eval_set": name, "text": _eval_text(spec, row)} for row in rows))
    except Exception as exc:
        if not spec.get("gated"):
            raise
        write_meta(cache_dir / f"{name}.meta.json", name=name, kind="eval_set",
                   dataset=spec["dataset"], config=config, split=spec["split"], revision=None,
                   n_rows=0, field_map={"text": spec["field"]}, status="skipped: gated",
                   error=str(exc))
        print(json.dumps({"fetch": name, "status": "skipped: gated", "error": str(exc)}), flush=True)
        return
    field_map = {"text": spec["field"]}
    if spec.get("context_field"):
        field_map["text"] = f"{spec['context_field']} + ' ' + {spec['field']} (context when non-empty)"
    meta = write_meta(cache_dir / f"{name}.meta.json", name=name, kind="eval_set",
                      dataset=spec["dataset"], config=config, split=split_used or spec["split"],
                      revision=revision, n_rows=n_rows,
                      field_map=field_map, status="ok")
    print(json.dumps({"fetch": name, **{k: meta[k] for k in ("n_rows", "split", "revision", "status")}}),
          flush=True)


def cmd_fetch(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)
    for name in SOURCES:
        fetch_source(name, cache_dir)
    for name, spec in EVAL_SETS.items():
        if spec.get("gated") and not args.with_gpqa:
            if not (cache_dir / f"{name}.meta.json").exists():
                write_meta(cache_dir / f"{name}.meta.json", name=name, kind="eval_set",
                           dataset=spec["dataset"], config=spec["config"], split=spec["split"],
                           revision=None, n_rows=0, field_map={"text": spec["field"]},
                           status="skipped: gated", error="not requested (--with-gpqa)")
            print(json.dumps({"fetch": name, "status": "skipped: gated",
                              "error": "not requested (--with-gpqa)"}), flush=True)
            continue
        fetch_eval_set(name, cache_dir)
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_meta(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Recorder:

    def __init__(self, anchor: Path):
        first = source_record(anchor)
        self.commit, self.dirty = first["commit"], first["dirty"]

    def __call__(self, path: str | Path) -> dict:
        path = Path(path)
        return {"path": str(path.resolve()), "sha256": _sha256_file(path),
                "commit": self.commit, "dirty": self.dirty}


def _parse_quotas(items: list[str]) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for item in items:
        name, _, value = item.partition("=")
        if name not in SOURCES or not value.isdigit():
            raise SystemExit(f"--quota expects <source>=<int> with source in {list(SOURCES)}, got {item!r}")
        quotas[name] = int(value)
    if not quotas:
        raise SystemExit("at least one --quota is required")
    return {name: quotas[name] for name in SOURCES if name in quotas}


def cmd_build(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)
    manifest_out, contract_out = Path(args.manifest_out), Path(args.contract_out)
    decontam_out = (Path(args.decontam_report_out) if args.decontam_report_out
                    else contract_out.parent / "decontam_report.json")
    for path in (manifest_out, contract_out, decontam_out):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing {path}")
    quotas = _parse_quotas(args.quota)
    shares = tuple(float(s) for s in args.shares.split(","))

    candidates: list[dict] = []
    dropped_non_integer: dict[str, int] = {}
    raw_by_source: dict[str, list[dict]] = {}
    for source in quotas:
        path = cache_dir / f"{source}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing cached source {path} (run `fetch` first)")
        rows = _read_jsonl(path)
        raw_by_source[source] = rows
        dropped_non_integer[source] = 0
        for row in rows:
            gold = pool.parse_integer_answer(row.get("gold_raw"))
            if gold is None:
                dropped_non_integer[source] += 1
                continue
            candidates.append({"source": source, "source_id": row.get("source_id"),
                               "question": str(row["question"]), "gold": gold,
                               "level": row.get("level")})

    spec_b_path = Path(args.manifest_spec_b)
    spec_b_rows = _read_jsonl(spec_b_path)
    spec_b_consistency: dict = {"n_spec_b": len(spec_b_rows), "n_found": None, "fraction": None}
    if "dapo" in quotas and spec_b_rows:
        dapo_qkeys = {question_key(str(row["question"])) for row in raw_by_source["dapo"]}
        n_found = sum(str(row["qkey"]) in dapo_qkeys for row in spec_b_rows)
        fraction = n_found / len(spec_b_rows)
        spec_b_consistency.update({"n_found": n_found, "fraction": fraction})
        if fraction < SPEC_B_MIN_REFOUND:
            raise SystemExit(
                f"spec-B consistency failed: {n_found}/{len(spec_b_rows)} spec-B questions "
                f"re-found among DAPO rows (< {SPEC_B_MIN_REFOUND:.0%}); the DAPO extraction "
                "differs from the historical one -- stopping")

    eval_index = pool.NGramIndex()
    eval_exact: dict[str, str] = {}
    eval_status: dict[str, dict] = {}
    eval_paths: list[tuple[str, Path, dict | None]] = []
    for name, spec in EVAL_SETS.items():
        path = cache_dir / f"{name}.jsonl"
        meta = _read_meta(cache_dir / f"{name}.meta.json")
        if not path.exists():
            if spec.get("gated"):
                status = (meta or {}).get("status") or "skipped: gated"
                if not str(status).startswith("skipped"):
                    status = "skipped: gated"
                eval_status[name] = {"status": status, "n_texts": 0}
                continue
            raise SystemExit(f"missing cached eval set {name} ({path}); the decontamination "
                             "gate needs every ungated eval set (run `fetch` first)")
        n_texts = 0
        for row in _read_jsonl(path):
            text = str(row["text"])
            eval_exact.setdefault(question_key(text), name)
            eval_index.add(text, name)
            n_texts += 1
        eval_status[name] = {"status": "ok", "n_texts": n_texts}
        eval_paths.append((name, path, meta))
    gpqa_status = eval_status["gpqa_diamond"]["status"]
    record = _Recorder(Path(args.exclusions))
    eval_records = [{"name": name, "kind": "eval_set", **record(path), "meta": meta}
                    for name, path, meta in eval_paths]

    exclusions = json.loads(Path(args.exclusions).read_text())
    excluded_manifests = []
    for path in args.exclude_manifest:
        prev = _read_jsonl(Path(path))
        exclusions["qkeys"] = list(exclusions.get("qkeys", [])) + [str(r["qkey"]) for r in prev]
        exclusions["question_sha256s"] = (list(exclusions.get("question_sha256s", []))
                                          + [str(r["question_sha256"]) for r in prev
                                             if r.get("question_sha256")])
        excluded_manifests.append({**record(Path(path)), "n_questions": len(prev)})
    rows, report = pool.build_pool(
        candidates, quotas=quotas, exclusions=exclusions,
        eval_index=eval_index, eval_exact=eval_exact, salt=args.salt, shares=shares,
        allocation_fraction=args.allocation_fraction, base_chains=args.base_chains,
        allocation_chains=args.allocation_chains, tranche=args.tranche,
        max_question_chars=args.max_question_chars)
    for source, n in dropped_non_integer.items():
        report["sources"][source]["dropped_non_integer"] = n
        report["sources"][source]["cached_rows"] = len(raw_by_source[source])

    write_jsonl(manifest_out, rows)
    manifest_sha256 = _sha256_file(manifest_out)
    source_records = [{"name": source, "kind": "source",
                       **record(cache_dir / f"{source}.jsonl"),
                       "meta": _read_meta(cache_dir / f"{source}.meta.json")}
                      for source in quotas]
    contract = pool.contract(
        report, manifest_path=str(manifest_out.resolve()), manifest_sha256=manifest_sha256,
        source_records=source_records, eval_records=eval_records,
        exclusions_sha256=_sha256_file(Path(args.exclusions)), salt=args.salt, shares=shares,
        quotas=quotas, gpqa_status=gpqa_status)
    contract["spec_b_consistency"] = spec_b_consistency
    contract["eval_set_status"] = eval_status
    contract["excluded_manifests"] = excluded_manifests
    contract["provenance"] = {
        "exclusions": record(args.exclusions),
        "spec_b_manifest": record(spec_b_path),
        "manifest": record(manifest_out),
        "cache_files": [record(p) for p in sorted(cache_dir.iterdir()) if p.is_file()],
    }
    contract_out.parent.mkdir(parents=True, exist_ok=True)
    contract_out.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    per_eval = {name: [] for name in eval_status if eval_status[name]["status"] == "ok"}
    for hit in report["decontam_hits"]:
        per_eval.setdefault(hit["matched_eval_set"], []).append(hit)
    decontam = {
        "salt": args.salt, "ngram_n": eval_index.n, "n_hits": len(report["decontam_hits"]),
        "hits": report["decontam_hits"], "per_eval_set": per_eval, "eval_sets": eval_status,
        "gpqa_diamond": gpqa_status, "manifest_sha256": manifest_sha256,
    }
    decontam_out.parent.mkdir(parents=True, exist_ok=True)
    decontam_out.write_text(json.dumps(decontam, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "pool_size": report["pool_size"], "split_counts": report["split_counts"],
        "allocation_counts": report["allocation_counts"], "total_chains": report["total_chains"],
        "quota": {s: f"{r['quota_filled']}/{r['quota_requested']}" for s, r in report["sources"].items()},
        "dropped_non_integer": dropped_non_integer, "decontam_hits": len(report["decontam_hits"]),
        "gpqa_diamond": gpqa_status, "spec_b_consistency": spec_b_consistency,
        "manifest": str(manifest_out), "manifest_sha256": manifest_sha256,
        "contract": str(contract_out), "decontam_report": str(decontam_out),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="materialise sources + eval sets from the HF hub")
    fetch.add_argument("--cache-dir", required=True)
    fetch.add_argument("--with-gpqa", action="store_true",
                       help="also fetch the gated Idavidrein/gpqa gpqa_diamond (needs a HF token)")

    build = sub.add_parser("build", help="pure-python pool build over the cache")
    build.add_argument("--cache-dir", required=True)
    build.add_argument("--exclusions", default=str(DEFAULT_EXCLUSIONS))
    build.add_argument("--salt", default=DEFAULT_SALT)
    build.add_argument("--quota", action="append", default=[], metavar="SOURCE=N",
                       help="per-source quota, e.g. --quota dapo=1500 (repeatable)")
    build.add_argument("--manifest-out", required=True)
    build.add_argument("--contract-out", required=True)
    build.add_argument("--decontam-report-out", default=None,
                       help="default: decontam_report.json next to --contract-out")
    build.add_argument("--manifest-spec-b", default=str(DEFAULT_SPEC_B_MANIFEST))
    build.add_argument("--tranche", type=int, default=1)
    build.add_argument("--exclude-manifest", action="append", default=[], metavar="MANIFEST",
                       help="a previous tranche's manifest whose questions are excluded "
                            "(repeatable; recorded in the contract as excluded_manifests)")
    build.add_argument("--shares", default="0.6,0.15,0.25")
    build.add_argument("--allocation-fraction", type=float, default=0.25)
    build.add_argument("--base-chains", type=int, default=8)
    build.add_argument("--allocation-chains", type=int, default=16)
    build.add_argument("--max-question-chars", type=int, default=2000,
                       help="Decision 2: drop questions longer than this many characters")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "fetch":
        return cmd_fetch(args)
    return cmd_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
