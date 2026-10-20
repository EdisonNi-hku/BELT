"""Grade one code-mode probe shard with unit tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from rl_controller.iclr.chains import validate_chain, write_jsonl
from rl_controller.iclr.code_exec import (STATUSES, Limits, Verdict, detect_net_isolation, grade_many,
                                          run_program, verdict_to_forced)
from rl_controller.single_cot import oracle_report

DEFAULT_CAPS = (16, 32, 64, 96, 128)
PREFLIGHT_PROGRAM = "import resource\nprint(resource.getrlimit(resource.RLIMIT_AS)[0])\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_marker(path: Path) -> str:
    marker = path.with_name(path.name + ".sha256")
    if not marker.exists():
        raise SystemExit(f"{marker} missing: the generator did not finish this shard")
    tokens = marker.read_text(encoding="utf-8").split()
    declared = tokens[0].lower() if tokens else ""
    actual = sha256_file(path)
    if declared != actual:
        raise SystemExit(f"{path}: sha256 {actual} != marker {marker} {declared}")
    return actual


PARTIAL_NAME = "grades.partial.jsonl"
CACHE_VERSION = "iclr_grade_code/resume-v1"
VERDICT_FIELDS = ("status", "n_tests", "n_passed", "first_failure", "wall_s")


def grade_row(key: tuple, verdict: Verdict) -> dict:
    return {"qkey": key[0], "rep": key[1], "kind": key[2], "block": key[3], **verdict.as_dict()}


def row_key(row: dict) -> tuple:
    return (str(row["qkey"]), int(row["rep"]), str(row["kind"]),
            None if row.get("block") is None else int(row["block"]))


def row_verdict(row: dict) -> Verdict:
    return Verdict(**{field: row[field] for field in VERDICT_FIELDS})


def cache_fingerprint(*, programs_sha256: str, manifest_sha256: str, limits: Limits, python: str,
                      prefix: list) -> dict:
    return {"programs_sha256": programs_sha256, "manifest_sha256": manifest_sha256,
            "limits": dict(sorted(vars(limits).items())), "python": python,
            "net_isolation": list(prefix)}


class ResumeCache:

    def __init__(self, path: Path, fingerprint: dict):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self._handle = None

    def load(self, programs: dict, expected_n_tests: dict) -> tuple[dict, str]:
        if not self.path.exists():
            return {}, "absent"
        done: dict = {}
        stale = None
        raw = self.path.read_bytes()
        header, _, rest = raw.partition(b"\n")
        try:
            meta = json.loads(header.decode("utf-8")) if header.strip() else {}
        except (ValueError, UnicodeDecodeError):
            meta = {}
        if meta.get("_cache") != CACHE_VERSION:
            stale = "header missing or unrecognised"
        else:
            differing = [field for field, value in self.fingerprint.items()
                         if meta.get(field) != value]
            if differing:
                stale = "; ".join(f"{field}: cache {meta.get(field)!r} != shard "
                                  f"{self.fingerprint[field]!r}" for field in differing)
        torn = False
        good_end = min(len(header) + 1, len(raw))
        if stale is None:
            pos = good_end
            for chunk in rest.split(b"\n"):
                nxt = min(pos + len(chunk) + 1, len(raw))
                if not chunk.strip():
                    pos = nxt
                    continue
                try:
                    row = json.loads(chunk.decode("utf-8"))
                    key, verdict = row_key(row), row_verdict(row)
                except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                    torn = True
                    break
                if key not in programs:
                    stale = f"cached key {key} is not in programs.jsonl"
                    break
                if key in done:
                    stale = f"cached key {key} appears twice"
                    break
                if verdict.status not in STATUSES:
                    stale = f"cached key {key} has unknown status {verdict.status!r}"
                    break
                if verdict.n_tests != expected_n_tests.get(key[0]):
                    stale = (f"cached key {key} ran {verdict.n_tests} tests, the manifest has "
                             f"{expected_n_tests.get(key[0])}")
                    break
                done[key] = verdict
                pos = good_end = nxt
        if stale is None and torn:
            os.truncate(self.path, good_end)
            print(f"[iclr_grade_code] repaired a torn resume cache: truncated to {good_end} bytes "
                  f"({len(done)} intact verdicts)", file=sys.stderr)
        if stale is not None:
            aside = self.path.with_name(self.path.name + ".stale." + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
            os.replace(self.path, aside)
            print(f"[iclr_grade_code] ignoring a stale resume cache ({stale}); moved to {aside}",
                  file=sys.stderr)
            return {}, f"stale: {stale}"
        return done, ("resumed" if done else "empty")

    def open(self) -> None:
        fresh = not self.path.exists() or self.path.stat().st_size == 0
        self._handle = open(self.path, "a", encoding="utf-8")
        if fresh:
            self._write({"_cache": CACHE_VERSION, **self.fingerprint})

    def _write(self, obj: dict) -> None:
        self._handle.write(json.dumps(obj, sort_keys=True) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def append(self, key: tuple, verdict: Verdict) -> None:
        if verdict.status == "grader_error":
            return
        self._write(grade_row(key, verdict))

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def program_key(record: dict) -> tuple:
    return (str(record["qkey"]), int(record["rep"]), str(record["kind"]),
            (None if record.get("block") is None else int(record["block"])))


def index_programs(records: list[dict]) -> dict:
    out = {}
    for record in records:
        key = program_key(record)
        if key in out:
            raise SystemExit(f"duplicate program record {key}")
        out[key] = record
    return out


def expected_keys(chains: list[dict]) -> set:
    keys = set()
    for c in chains:
        for p in c["probes"]:
            keys.add((str(c["qkey"]), int(c["rep"]), "probe", int(p["block"])))
        if c["outcome"] == "natural":
            keys.add((str(c["qkey"]), int(c["rep"]), "natural", None))
    return keys


def check_coverage(chains: list[dict], programs: dict) -> None:
    expected, got = expected_keys(chains), set(programs)
    missing, extra = sorted(expected - got, key=str), sorted(got - expected, key=str)
    if missing or extra:
        raise SystemExit(f"programs.jsonl does not cover the chains: missing {missing[:5]} "
                         f"({len(missing)}), unexpected {extra[:5]} ({len(extra)})")


def load_tests(tests_dir: Path, manifest_rows: list[dict], qkeys: set) -> tuple[dict, dict]:
    by_key = {str(r["qkey"]): r for r in manifest_rows}
    tests, kinds = {}, {}
    for qkey in sorted(qkeys):
        row = by_key.get(qkey)
        if row is None:
            raise SystemExit(f"chain question {qkey} is not in the manifest")
        path = tests_dir / f"{qkey}.json"
        if not path.exists():
            raise SystemExit(f"missing tests file {path}")
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha != row["tests_sha256"]:
            raise SystemExit(f"{path}: tests_sha256 {sha} != manifest {row['tests_sha256']}")
        tests[qkey] = json.loads(data)["tests"]
        kinds[qkey] = str(row["test_kind"])
    return tests, kinds


def numpy_available(python: str) -> bool:
    try:
        return subprocess.run([python, "-I", "-c", "import numpy"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def kib_exact_bytes(mem_gib: float) -> int:
    return int(mem_gib * (1 << 20)) << 10


def limits_preflight(limits: Limits, *, python: str, prefix: list[str]) -> None:
    verdict = run_program(PREFLIGHT_PROGRAM, [{"kind": "stdin", "input": "", "output": f"{limits.mem_bytes}\n"}],
                          limits=limits, python=python, prefix=prefix)
    if verdict.status != "pass":
        detail = (verdict.first_failure or {}).get("detail")
        raise SystemExit("sandbox preflight failed: the rlimit wrapper does not work on this node "
                         f"(status {verdict.status}, python {python}, prefix {prefix}, detail {detail!r})")


def apply_verdicts(chains: list[dict], verdicts: dict) -> None:
    for c in chains:
        for p in c["probes"]:
            p["forced"] = verdict_to_forced(verdicts[(str(c["qkey"]), int(c["rep"]), "probe", int(p["block"]))])
        if c["outcome"] == "natural":
            c["answer"] = verdict_to_forced(verdicts[(str(c["qkey"]), int(c["rep"]), "natural", None)])
        else:
            c["answer"] = None
        c["correct"] = c["answer"] == 1
        try:
            validate_chain(c)
        except ValueError as err:
            raise SystemExit(f"graded chain failed the contract: {err}") from err


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard-dir", required=True, help="the shard's split dir (chains.ungraded.jsonl, programs.jsonl)")
    p.add_argument("--tests-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--timeout", type=float, default=10.0, help="wall seconds per stdin/call test")
    p.add_argument("--mem-gib", type=float, default=1.0)
    p.add_argument("--python", default=sys.executable, help="interpreter that runs the programs")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    shard = Path(args.shard_dir)
    if (shard / "chains.jsonl").exists():
        raise SystemExit(f"{shard / 'chains.jsonl'} exists: refusing to grade a shard twice "
                         "(move it aside if the grade must be redone)")
    programs_sha = check_marker(shard / "programs.jsonl")
    chains = load_jsonl(shard / "chains.ungraded.jsonl")
    if not chains:
        raise SystemExit("chains.ungraded.jsonl is empty")
    for c in chains:
        try:
            validate_chain(c)
        except ValueError as err:
            raise SystemExit(f"chains.ungraded.jsonl: {err}") from err
    programs = index_programs(load_jsonl(shard / "programs.jsonl"))
    check_coverage(chains, programs)
    manifest_rows = load_jsonl(Path(args.manifest))
    tests, kinds = load_tests(Path(args.tests_dir), manifest_rows, {str(c["qkey"]) for c in chains})
    if "script" in kinds.values() and not numpy_available(args.python):
        raise SystemExit(f"{args.python} cannot import numpy, which the EvalPlus script tests need")
    limits = Limits(wall_s=float(args.timeout), cpu_s=int(args.timeout), mem_bytes=kib_exact_bytes(args.mem_gib))
    prefix = detect_net_isolation()
    limits_preflight(limits, python=args.python, prefix=prefix)
    cache = ResumeCache(shard / PARTIAL_NAME, cache_fingerprint(
        programs_sha256=programs_sha, manifest_sha256=sha256_file(Path(args.manifest)),
        limits=limits, python=args.python, prefix=prefix))
    done, cache_note = cache.load(programs, {qkey: len(rows) for qkey, rows in tests.items()})
    pending = [(key, rec["program"], tests[key[0]]) for key, rec in programs.items() if key not in done]
    print(json.dumps({"resume_cache": cache_note, "cached": len(done), "to_grade": len(pending),
                      "programs": len(programs)}), flush=True)
    t0 = time.time()
    cache.open()
    try:
        fresh = grade_many(pending, workers=args.workers, limits=limits, python=args.python,
                           prefix=prefix, on_result=cache.append)
    except (RuntimeError, ValueError) as err:
        raise SystemExit(f"grading aborted: {err}") from None
    finally:
        cache.close()
    wall = time.time() - t0
    verdicts = {key: (done[key] if key in done else fresh[key]) for key in programs}
    grades = [grade_row(key, verdicts[key]) for key in programs]
    write_jsonl(shard / "grades.jsonl", grades)
    broken = [(key, v) for key, v in verdicts.items() if v.status == "grader_error"]
    if broken:
        first_key, first = broken[0]
        detail = (first.first_failure or {}).get("detail")
        raise SystemExit(f"{len(broken)} grader errors (see grades.jsonl), e.g. {first_key}: {detail}")
    apply_verdicts(chains, verdicts)
    write_jsonl(shard / "chains.jsonl", chains)
    (shard / PARTIAL_NAME).unlink(missing_ok=True)
    counts = dict(Counter(v.status for v in verdicts.values()))
    info = {"limits": limits.__dict__, "net_isolation": list(prefix),
            "limits_preflight": "pass",
            "python": args.python, "workers": args.workers, "programs": len(programs),
            "status_counts": counts, "grade_wall_s": round(wall, 1), "programs_sha256": programs_sha,
            "resume": {"cache": cache_note, "from_cache": len(done), "graded_now": len(pending)},
            "manifest": str(Path(args.manifest).resolve()), "tests_dir": str(Path(args.tests_dir).resolve())}
    (shard / "grade.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path = shard / "summary.json"
    if summary_path.exists():
        meta = json.load(open(summary_path)).get("meta", {})
    else:
        print(f"[iclr_grade_code] warning: {summary_path} is absent (the generator writes it); "
              f"summary.json will carry no generator meta and the default caps {DEFAULT_CAPS}", file=sys.stderr)
        meta = {}
    report = oracle_report(chains, caps=tuple(meta.get("caps") or DEFAULT_CAPS))
    report["meta"] = {**meta, "graded": True, "grade": info}
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"graded": len(chains), "programs": len(programs), "status_counts": counts,
                      "accuracy": report["rows"][0]["accuracy"], "grade_wall_s": round(wall, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
