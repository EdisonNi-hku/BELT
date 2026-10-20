import io
import json
import numbers
import os
import sys

PREAMBLE = (
    "import sys, math, itertools, collections, functools, heapq, bisect, re, string\n"
    "from typing import *\n"
    "from collections import *\n"
    "from itertools import *\n"
    "from functools import *\n"
    "from heapq import *\n"
    "from bisect import *\n"
    "from math import *\n"
)


def _jsonable(value):
    if isinstance(value, bool):
        return value
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    return value


def _resolve(namespace, fn_name):
    cls = namespace.get("Solution")
    if isinstance(cls, type) and hasattr(cls, fn_name):
        return getattr(cls(), fn_name)
    fn = namespace.get(fn_name)
    if callable(fn):
        return fn
    raise NameError(f"function {fn_name!r} is not defined by the program")


def _emit(fd, line):
    data = line.encode("utf-8", "replace")
    while data:
        n = os.write(fd, data)
        data = data[n:]


def main():
    mode = sys.argv[1]
    payload = json.loads(sys.stdin.read())
    program, test, nonce = payload["program"], payload["test"], payload["nonce"]
    verdict_fd = os.dup(1)
    sys.stdout = open(os.devnull, "w")
    sys.stdin = io.TextIOWrapper(io.BytesIO())
    ok, why, error = False, "", False
    try:
        namespace = {"__name__": "__main__"}
        if mode not in ("call", "script"):
            why = f"unknown harness mode {mode!r}"
        else:
            if mode == "call":
                exec(PREAMBLE, namespace)
            try:
                exec(compile(program, "sol.py", "exec"), namespace)
            except BaseException:
                error = True
                raise
            if mode == "call":
                fn = _resolve(namespace, test["fn_name"])
                result = json.loads(json.dumps(_jsonable(fn(*test["input"]))))
                expected = json.loads(json.dumps(test["output"]))
                ok = result == expected
                if not ok and "output_alt" in test:
                    ok = result == json.loads(json.dumps(test["output_alt"]))
                if not ok:
                    why = f"got {json.dumps(result)[:200]} expected {json.dumps(expected)[:200]}"
            else:
                try:
                    exec(compile(test["test"], "test.py", "exec"), namespace)
                except ImportError:
                    error = True
                    raise
                check = namespace.get("check")
                if callable(check):
                    check(namespace[test["entry_point"]])
                ok = True
    except SystemExit as exc:
        why = f"SystemExit: {exc.code!r}"
    except BaseException as exc:
        why = f"{type(exc).__name__}: {str(exc)[:300]}"
    _emit(verdict_fd, nonce + " " + json.dumps({"ok": bool(ok), "why": why[:500], "error": error}) + "\n")


if __name__ == "__main__":
    main()
