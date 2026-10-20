"""Apply the natural-completion fallback to a scored dump."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rl_controller.scripts.belt_fallback_bank import derive_row, load_fallback, sha256


def correct_all(rows, fallback: dict, *, coverage: str = "full", accept_precorrected: bool = False):
    if coverage not in ("full", "questions", "subset"):
        raise ValueError(f"coverage must be 'full', 'questions' or 'subset', got {coverage!r}")
    out, used, seen, qkeys = [], set(), set(), set()
    n_applied = n_parseable = n_up = n_down = n_pre = 0
    for row in rows:
        key = (row["qkey"], int(row["rep"]))
        if key in seen:
            raise ValueError(f"duplicate dump row {key}")
        seen.add(key)
        qkeys.add(row["qkey"])
        fb = fallback.get(key)
        if fb is not None and int(fb["thinking_tokens"]) != int(row["thinking_tokens"]):
            raise ValueError(f"fallback row {key} thinking_tokens {fb['thinking_tokens']} != dump "
                             f"{row['thinking_tokens']}: the join hit a different chain")
        pre = False
        if fb is not None and row["outcome"] == "natural" and row["answer"] is not None:
            if not accept_precorrected:
                raise ValueError(f"row {key} already carries an answer where the bank's was unparseable "
                                 "(a pre-corrected dump); pass accept_precorrected")
            if row["answer"] != fb["forced"] or type(row["answer"]) is not type(fb["forced"]):
                raise ValueError(f"pre-corrected row {key} answer {row['answer']!r} != fallback {fb['forced']!r}")
            raw = dict(row, answer=None, correct=False)
            new = derive_row(raw, fb)
            if bool(new["correct"]) is not bool(row["correct"]):
                raise ValueError(f"pre-corrected row {key} correct {row['correct']!r} != fallback grade")
            pre = True
            n_pre += 1
        else:
            new = derive_row(row, fb)
        if fb is not None:
            used.add(key)
            n_applied += 1
            n_parseable += new["answer"] is not None
            n_up += bool(new["correct"]) and not bool(new["correct_raw"])
            n_down += bool(new["correct_raw"]) and not bool(new["correct"])
        out.append(new)
    unused = set(fallback) - used
    out_of_scope = {k for k in unused if k[0] not in qkeys}
    surplus = {"full": unused, "questions": unused - out_of_scope, "subset": set()}[coverage]
    if surplus:
        raise ValueError(f"{len(surplus)} fallback rows match no dump row (coverage={coverage}), "
                         f"e.g. {sorted(surplus)[:3]}")
    stats = {"n_rows": len(out), "n_questions": len(qkeys), "n_fallback_applied": n_applied,
             "n_fallback_parseable": n_parseable, "n_correct_0_to_1": n_up,
             "n_correct_1_to_0": n_down, "n_precorrected": n_pre, "n_fallback_rows": len(fallback),
             "n_fallback_unused": len(unused), "n_fallback_out_of_scope": len(out_of_scope),
             "coverage": coverage}
    return out, stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True)
    p.add_argument("--fallback", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--coverage", default="full", choices=("full", "questions", "subset"))
    p.add_argument("--accept-precorrected", action="store_true")
    a = p.parse_args(argv)
    if Path(a.out).resolve() == Path(a.dump).resolve():
        raise SystemExit("--out must differ from --dump (the input is only read)")
    fallback = load_fallback(a.fallback)
    with open(a.dump) as f:
        rows = (json.loads(line) for line in f if line.strip())
        try:
            new, stats = correct_all(rows, fallback, coverage=a.coverage,
                                     accept_precorrected=a.accept_precorrected)
        except ValueError as exc:
            raise SystemExit(f"{a.dump}: {exc}") from None
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w") as dst:
        for row in new:
            dst.write(json.dumps(row) + "\n")
    tmp.rename(out)
    summary = {"schema": "belt-fallback-dump/v1",
               "dump": {"path": str(a.dump), "sha256": sha256(a.dump)},
               "fallback": {"path": str(a.fallback), "sha256": sha256(a.fallback)},
               "out": {"path": str(out), "sha256": sha256(out)}, **stats}
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
