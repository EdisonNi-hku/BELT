from __future__ import annotations

import hashlib
import json
import re
from typing import Iterable, Mapping

from rl_controller.iclr import pool
from rl_controller.splits import question_key

MAX_TESTS = 15
MIN_TESTS = 5
MAX_TEST_BYTES = 1 << 20
MAX_TESTS_BYTES = 4 << 20
MAX_SIGNATURE_LINES = 8
KINDS = ("stdin", "call", "script")

STDIN_SUFFIX = "\n\nRead the input from standard input and write the answer to standard output."
CALL_SUFFIX_STARTER = "\n\nUse this starter code and implement `{fn_name}`:\n```python\n{starter_code}\n```"
CALL_SUFFIX_PLAIN = "\n\nImplement the function `{fn_name}`."
SCRIPT_SUFFIX = "Complete the function `{entry_point}` below; keep its signature.\n```python\n{signature}\n```"

SYNTHETIC2_TAIL = ("Write Python code to solve the problem. Present the code in \n"
                   "```python\nYour code\n```\nat the end.")
CODE_SHARES = (0.6, 0.15, 0.25)
_PY_FENCE = re.compile(r"```[Pp]ython3?[ \t]*\r?\n(.*?)```", re.DOTALL)


def strip_synthetic2_instruction(prompt: str) -> tuple[str, bool]:
    text = str(prompt).strip()
    stripped = text.endswith(SYNTHETIC2_TAIL)
    if stripped:
        text = text[: -len(SYNTHETIC2_TAIL)].strip()
    return text, stripped


def last_python_block(text) -> str | None:
    if not isinstance(text, str):
        return None
    found = _PY_FENCE.findall(text)
    return found[-1] if found else None


def _json_arg(text):
    return json.loads(text) if isinstance(text, str) else text


def normalise_tests(source: str, tests_raw, metadata=None) -> list[dict] | None:
    metadata = metadata or {}
    if source == "synthetic2":
        return _normalise_synthetic2(tests_raw)
    if source != "lcb":
        raise ValueError(f"unknown test source {source!r}")
    if not isinstance(tests_raw, list) or not all(isinstance(t, dict) for t in tests_raw):
        return None
    out = []
    for t in tests_raw:
        if t.get("testtype") == "functional":
            fn = metadata.get("func_name")
            if not fn:
                return None
            try:
                args = [json.loads(line) for line in str(t["input"]).split("\n") if line.strip()]
                output = _json_arg(t["output"])
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
            out.append({"kind": "call", "fn_name": str(fn), "input": args, "output": output})
        else:
            out.append({"kind": "stdin", "input": str(t["input"]), "output": str(t["output"])})
    if len({t["kind"] for t in out}) != 1 or len(out) < MIN_TESTS:
        return None
    return out


def _normalise_synthetic2(info) -> list[dict] | None:
    try:
        info = _json_arg(info)
        cases = info.get("test_cases") if isinstance(info, dict) else None
        cases = _json_arg(cases) if isinstance(cases, str) else cases
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(cases, dict):
        return None
    ins, outs = cases.get("inputs"), cases.get("outputs")
    if not isinstance(ins, list) or not isinstance(outs, list) or len(ins) != len(outs) or not ins:
        return None
    fn = cases.get("fn_name")
    out = []
    for i, o in zip(ins, outs):
        if fn:
            if not isinstance(i, list):
                return None
            test = {"kind": "call", "fn_name": str(fn), "input": i, "output": o}
            if isinstance(o, list) and o:
                test["output_alt"] = o[0]
            out.append(test)
        else:
            if not isinstance(i, str) or not isinstance(o, str):
                return None
            out.append({"kind": "stdin", "input": i, "output": o})
    return out if len(out) >= MIN_TESTS else None


def pilot_order(qkeys: Iterable[str], pilot_pass: Mapping[str, int], salt: str) -> list[str]:
    groups: dict[int, list[str]] = {0: [], 1: [], 2: []}
    for key in sorted(set(qkeys)):
        count = pilot_pass[key]
        if count not in groups:
            raise ValueError(f"pilot pass count {count!r} for {key} is not in 0..2")
        groups[count].append(key)
    for keys in groups.values():
        keys.sort(key=lambda k: pool.rank(salt, "pool", k))
    order = list(groups[1])
    zeros, twos = groups[0], groups[2]
    for i in range(max(len(zeros), len(twos))):
        if i < len(zeros):
            order.append(zeros[i])
        if i < len(twos):
            order.append(twos[i])
    return order


def build_code_pool(candidates, *, quota: int, salt: str, rank_fn=None, tranche: int = 1,
                    eval_index=None, eval_exact=None, exclusions=None,
                    max_question_chars: int = 6000) -> tuple[list[dict], dict]:
    candidates = list(candidates)
    extras: dict[str, dict] = {}
    for c in candidates:
        extras.setdefault(question_key(str(c["question"])), c)
    rows, report = pool.build_pool(
        candidates, quotas={"synthetic2": int(quota)}, exclusions=exclusions or {},
        eval_index=eval_index if eval_index is not None else pool.NGramIndex(),
        eval_exact=eval_exact or {}, salt=salt, shares=CODE_SHARES, allocation_fraction=0.25,
        base_chains=8, allocation_chains=16, tranche=tranche,
        max_question_chars=max_question_chars, rank_fn=rank_fn)
    for row in rows:
        c = extras[row["qkey"]]
        row.update({"domain": "code", "answer_format": "code", "test_kind": str(c["test_kind"]),
                    "tests_sha256": str(c["tests_sha256"]), "n_tests": int(c["n_tests"]),
                    "benchmark": None, "strict_post_qwen3": None, "contest_date": None})
    return rows, report


def _input_text(test: dict) -> str:
    return json.dumps(test.get("input"), ensure_ascii=False)


def select_tests(tests: list[dict], k: int = MAX_TESTS, *, max_test_bytes: int = MAX_TEST_BYTES,
                 max_total_bytes: int = MAX_TESTS_BYTES) -> list[dict] | None:
    eligible = [i for i, t in enumerate(tests)
                if len(_input_text(t).encode("utf-8")) <= max_test_bytes]
    if len(eligible) < MIN_TESTS:
        return None
    ranked = sorted(eligible, key=lambda i: (-len(_input_text(tests[i])), i))
    keep, total = [], 0
    for i in ranked:
        if len(keep) >= k:
            break
        size = len(json.dumps(tests[i]).encode("utf-8"))
        if total + size > max_total_bytes:
            continue
        keep.append(i)
        total += size
    if len(keep) < min(MIN_TESTS, k):
        return None
    return [tests[i] for i in sorted(keep)]


def compose_question(statement: str, *, test_kind: str, fn_name: str | None = None,
                     starter_code: str = "", entry_point: str | None = None,
                     signature: str | None = None) -> str:
    if test_kind == "stdin":
        return statement + STDIN_SUFFIX
    if test_kind == "call":
        if not fn_name:
            raise ValueError("a call problem needs fn_name")
        if starter_code and starter_code.strip():
            starter = starter_code.replace("\r\n", "\n").strip("\n").rstrip()
            return statement + CALL_SUFFIX_STARTER.format(fn_name=fn_name, starter_code=starter)
        return statement + CALL_SUFFIX_PLAIN.format(fn_name=fn_name)
    if test_kind == "script":
        if not entry_point or signature is None:
            raise ValueError("a script problem needs entry_point and signature")
        head = statement + "\n\n" if statement.strip() else ""
        return head + SCRIPT_SUFFIX.format(entry_point=entry_point, signature=signature)
    raise ValueError(f"unknown test kind {test_kind!r}; choose from {KINDS}")


def _def_head(line: str) -> str | None:
    stripped = line.strip()
    for opener in ("async def ", "def "):
        if stripped.startswith(opener) and "(" in stripped:
            return stripped[len(opener):]
    return None


def _strip_comment(line: str) -> str:
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#":
            return line[:i].rstrip()
        i += 1
    return line.rstrip()


def _signature_end(line: str, depth: int) -> tuple[int | None, int]:
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ":" and depth == 0:
            return i, depth
        i += 1
    return None, depth


def entry_point_from_code(code: str) -> str:
    for line in code.splitlines():
        head = _def_head(line)
        if head is not None:
            return head[: head.index("(")].strip()
    raise ValueError("no `def` in the reference code")


def signature_from_code(code: str) -> str:
    lines = code.splitlines()
    start = next((i for i, line in enumerate(lines) if _def_head(line) is not None), None)
    if start is None:
        raise ValueError("no `def` in the reference code")
    out, depth = [], 0
    for line in lines[start:start + MAX_SIGNATURE_LINES]:
        line = _strip_comment(line)
        end, depth = _signature_end(line, depth)
        if end is not None:
            out.append(line[: end + 1].rstrip())
            return "\n".join(out)
        out.append(line)
    raise ValueError(f"the signature does not close within {MAX_SIGNATURE_LINES} lines")


def tests_payload(qkey: str, tests: list[dict]) -> tuple[bytes, str]:
    data = json.dumps({"qkey": qkey, "tests": tests}, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return data, hashlib.sha256(data).hexdigest()
