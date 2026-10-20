from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

HARNESS = Path(__file__).with_name("_code_harness.py")
STATUSES = ("pass", "fail", "error", "timeout", "memory", "no_program", "grader_error")
KINDS = ("stdin", "call", "script")
FLOAT_REL_TOL = 1e-6
_INT_RE = re.compile(r"[+-]?[0-9]+\Z")
_FLOAT_RE = re.compile(r"[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?\Z")
_STDERR_TAIL = 2000
_VERDICT_TAIL = 64 << 10
_REQUIRED_KEYS = {"stdin": ("input", "output"), "call": ("fn_name", "input", "output"),
                  "script": ("entry_point", "test")}


@dataclass(frozen=True)
class Limits:
    wall_s: float = 10.0
    cpu_s: int = 10
    mem_bytes: int = 1 << 30
    fsize_bytes: int = 8 << 20
    stdout_bytes: int = 4 << 20
    script_wall_s: float = 60.0


@dataclass(frozen=True)
class Verdict:
    status: str
    n_tests: int
    n_passed: int
    first_failure: dict | None
    wall_s: float

    def as_dict(self) -> dict:
        return asdict(self)


def verdict_to_forced(verdict: Verdict) -> int | None:
    if verdict.status == "grader_error":
        raise ValueError(f"grader_error verdict cannot become a chain verdict: {verdict.first_failure}")
    if verdict.status not in STATUSES:
        raise ValueError(f"unknown verdict status {verdict.status!r}; expected one of {STATUSES}")
    if verdict.status == "pass":
        return 1
    if verdict.status == "no_program":
        return None
    return 0


def detect_net_isolation() -> list[str]:
    unshare = shutil.which("unshare")
    if not unshare:
        return []
    for flags in (["-rn", "-p", "-f", "--kill-child"], ["-rn"]):
        try:
            ok = subprocess.run([unshare, *flags, "true"], capture_output=True, timeout=5).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        if ok:
            return [unshare, *flags]
    return []


def _lines(text: str) -> list[str]:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _tokens_match(a: str, b: str) -> bool:
    if a == b:
        return True
    a_int, b_int = _INT_RE.match(a), _INT_RE.match(b)
    if a_int and b_int:
        try:
            return int(a) == int(b)
        except ValueError:
            return a == b
    if not (a_int or _FLOAT_RE.match(a)) or not (b_int or _FLOAT_RE.match(b)):
        return False
    fa, fb = float(a), float(b)
    return abs(fa - fb) <= FLOAT_REL_TOL * max(1.0, abs(fb))


def compare_stdout(got: str, expected: str) -> tuple[bool, str]:
    g, e = _lines(got), _lines(expected)
    if g == e:
        return True, ""
    if len(g) == len(e):
        close = True
        for gl, el in zip(g, e):
            gt, et = gl.split(), el.split()
            if len(gt) != len(et) or not all(_tokens_match(a, b) for a, b in zip(gt, et)):
                close = False
                break
        if close:
            return True, ""
    return False, f"expected {e[:3]!r} got {g[:3]!r}"


BASH = "/bin/bash"
_ULIMIT_SCRIPT = 'ulimit -v "$1" && ulimit -t "$2" && ulimit -f "$3" && ulimit -c 0 && shift 3 && exec "$@"'


def _limited_argv(cmd: list[str], *, limits: Limits, cpu_s: int, prefix: list[str]) -> list[str]:
    return [BASH, "-c", _ULIMIT_SCRIPT, "limits", str(limits.mem_bytes // 1024), str(cpu_s),
            str(limits.fsize_bytes // 1024), *prefix, *cmd]


def _kill_group(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _head(handle, n: int) -> bytes:
    handle.seek(0)
    return handle.read(n)


def _tail(handle, n: int) -> bytes:
    size = os.fstat(handle.fileno()).st_size
    handle.seek(max(0, size - n))
    return handle.read()


@dataclass(frozen=True)
class _Run:
    rc: int | str
    out: bytes
    out_tail: bytes
    err: bytes
    wall: float


def _run(argv, *, stdin_text: str, cwd: str, limits: Limits, wall_s: float, prefix: list[str]) -> _Run:
    t0 = time.monotonic()
    cpu_s = max(limits.cpu_s, int(wall_s) + 1)
    os.makedirs(cwd, exist_ok=True)
    d = Path(cwd)
    stdin_path, out_path, err_path = d / "stdin.txt", d / "out.txt", d / "err.txt"
    stdin_path.write_bytes(stdin_text.encode("utf-8", errors="replace"))
    handles = []
    try:
        fin = open(stdin_path, "rb")
        fout, ferr = open(out_path, "wb"), open(err_path, "wb")
        fout_r, ferr_r = open(out_path, "rb"), open(err_path, "rb")
        handles = [fin, fout, ferr, fout_r, ferr_r]
        proc = subprocess.Popen(
            _limited_argv(list(argv), limits=limits, cpu_s=cpu_s, prefix=list(prefix)),
            stdin=fin, stdout=fout, stderr=ferr, cwd=cwd,
            env={"PATH": "", "HOME": cwd, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
                 "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8"},
            start_new_session=True)
        try:
            try:
                rc = proc.wait(timeout=wall_s)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                proc.wait()
                return _Run("timeout", b"", b"", b"", time.monotonic() - t0)
        finally:
            _kill_group(proc)
        return _Run(rc, _head(fout_r, limits.stdout_bytes), _tail(fout_r, _VERDICT_TAIL),
                    _tail(ferr_r, _STDERR_TAIL), time.monotonic() - t0)
    finally:
        for h in handles:
            h.close()


def _failure_status(err: bytes) -> str:
    return "memory" if b"MemoryError" in err else "error"


def _stdin_case(sol: Path, test: dict, *, python: str, cwd: str, limits: Limits, prefix):
    r = _run([python, "-I", str(sol)], stdin_text=str(test["input"]), cwd=cwd,
             limits=limits, wall_s=limits.wall_s, prefix=prefix)
    if r.rc == "timeout":
        return False, "timeout", None
    if r.rc != 0:
        return False, _failure_status(r.err), r.err.decode("utf-8", "replace")
    ok, why = compare_stdout(r.out.decode("utf-8", "replace"), str(test["output"]))
    return ok, ("pass" if ok else "fail"), why


def _harness_case(program: str, test: dict, mode: str, *, python: str, cwd: str,
                  limits: Limits, prefix):
    nonce = secrets.token_hex(8)
    payload = json.dumps({"program": program, "test": test, "nonce": nonce})
    wall = limits.script_wall_s if mode == "script" else limits.wall_s
    r = _run([python, "-I", str(HARNESS), mode], stdin_text=payload, cwd=cwd,
             limits=limits, wall_s=wall, prefix=prefix)
    if r.rc == "timeout":
        return False, "timeout", None
    if r.rc != 0:
        return False, _failure_status(r.err), r.err.decode("utf-8", "replace")
    text = r.out_tail.decode("utf-8", "replace")
    line = next((l for l in reversed(text.splitlines()) if l.startswith(nonce + " ")), None)
    if line is None:
        return False, "error", "harness produced no verdict line"
    verdict = json.loads(line[len(nonce) + 1:])
    if verdict.get("error"):
        return False, "error", verdict.get("why")
    return bool(verdict["ok"]), ("pass" if verdict["ok"] else "fail"), verdict.get("why")


def _test_kind(tests) -> str:
    if not isinstance(tests, list) or not tests:
        raise ValueError("no tests")
    for i, t in enumerate(tests):
        if not isinstance(t, dict):
            raise ValueError(f"test {i} is not a dict")
        if "kind" not in t:
            raise ValueError(f"test {i} has no kind")
    kinds = {t["kind"] for t in tests}
    if len(kinds) != 1 or not kinds <= set(KINDS):
        raise ValueError(f"tests must be of one kind in {KINDS}, got {sorted(map(str, kinds))}")
    kind = kinds.pop()
    for i, t in enumerate(tests):
        missing = [k for k in _REQUIRED_KEYS[kind] if k not in t]
        if missing:
            raise ValueError(f"{kind} test {i} lacks {missing}")
    if kind == "script" and len(tests) != 1:
        raise ValueError("a script problem has exactly one test (the whole script)")
    return kind


def run_program(program: str | None, tests: list[dict], *, limits: Limits = Limits(),
                python: str = sys.executable, stop_at_first_failure: bool = True,
                prefix: list[str] | None = None) -> Verdict:
    t0 = time.monotonic()
    kind = _test_kind(tests)
    if program is None:
        return Verdict("no_program", len(tests), 0, None, 0.0)
    prefix = list(prefix) if prefix is not None else detect_net_isolation()
    status, n_passed, first = "pass", 0, None
    with tempfile.TemporaryDirectory(prefix="grade_") as tmp:
        sol = Path(tmp) / "sol.py"
        sol.write_text(program, encoding="utf-8", errors="replace")
        for index, test in enumerate(tests):
            try:
                if kind == "stdin":
                    ok, st, detail = _stdin_case(sol, test, python=python, cwd=tmp, limits=limits,
                                                 prefix=prefix)
                else:
                    ok, st, detail = _harness_case(program, test, kind, python=python, cwd=tmp,
                                                   limits=limits, prefix=prefix)
            except Exception as exc:
                first = {"index": index, "kind": kind,
                         "detail": f"grader exception: {type(exc).__name__}: {exc}"[:500]}
                status = "grader_error"
                break
            if ok:
                n_passed += 1
                continue
            status = st
            first = {"index": index, "kind": kind, "detail": (detail[-500:] if detail else None)}
            if stop_at_first_failure:
                break
    return Verdict(status, len(tests), n_passed, first, round(time.monotonic() - t0, 3))


def check_sandbox(limits: Limits = Limits(), *, python: str = sys.executable,
                  prefix: list[str] | None = None) -> None:
    prefix = list(prefix) if prefix is not None else detect_net_isolation()
    canaries = [("print(1)\n", "1\n", "pass")]
    if limits.mem_bytes <= (1 << 30):
        canaries.append(("x = bytearray(2 << 30)\nprint(len(x))\n", "", "memory"))
    for program, expected_out, expected_status in canaries:
        v = run_program(program, [{"kind": "stdin", "input": "", "output": expected_out}],
                        limits=limits, python=python, prefix=prefix)
        if v.status != expected_status:
            detail = (v.first_failure or {}).get("detail")
            raise RuntimeError(f"sandbox preflight failed: {program.splitlines()[0]!r} -> {v.status} "
                               f"(expected {expected_status}); python={python!r} prefix={prefix!r}; "
                               f"{detail!r}"[:1500])


def grade_many(items: Iterable[tuple], *, workers: int, limits: Limits = Limits(),
               python: str = sys.executable, prefix: list[str] | None = None,
               on_result=None) -> dict:
    items = list(items)
    seen = set()
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise ValueError(f"item {str(item)[:80]!r}: expected (key, program, tests)")
        key, _program, tests = item
        if key in seen:
            raise ValueError(f"item {key!r}: duplicate key")
        seen.add(key)
        try:
            _test_kind(tests)
        except ValueError as exc:
            raise ValueError(f"item {key!r}: {exc}") from None
    prefix = list(prefix) if prefix is not None else detect_net_isolation()
    check_sandbox(limits, python=python, prefix=prefix)

    def one(item):
        return run_program(item[1], item[2], limits=limits, python=python, prefix=prefix)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(one, item): item for item in items}
        if on_result is not None:
            for future in as_completed(futures):
                on_result(futures[future][0], future.result())
        return {item[0]: future.result() for future, item in futures.items()}
