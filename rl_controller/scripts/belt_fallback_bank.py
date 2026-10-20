from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from rl_controller.iclr.chains import is_correct


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_fallback(path) -> dict:
    out = {}
    for line in open(path):
        r = json.loads(line)
        key = (r["qkey"], int(r["rep"]))
        if key in out:
            raise SystemExit(f"duplicate fallback row {key}")
        out[key] = r
    return out


def derive_row(row: dict, fb: dict | None) -> dict:
    new = dict(row)
    new["answer_raw"], new["correct_raw"] = row["answer"], row["correct"]
    needs = row["outcome"] == "natural" and row["answer"] is None
    if needs:
        if fb is None:
            raise ValueError(f"closed unparseable chain {(row['qkey'], row['rep'])} has no fallback row")
        if fb["gold"] != row["gold"]:
            raise ValueError(f"fallback gold mismatch for {(row['qkey'], row['rep'])}")
        new["answer"] = fb["forced"]
        new["correct"] = bool(is_correct(fb["forced"], row["gold"]))
        new["natural_fallback_tokens"] = int(fb["forced_tokens"])
    else:
        if fb is not None:
            raise ValueError(f"fallback row for a chain that does not need one {(row['qkey'], row['rep'])}")
        new["natural_fallback_tokens"] = 0
    return new


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--chains", required=True); p.add_argument("--fallback", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    fbs = load_fallback(a.fallback)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    n = n_applied = n_flip_up = n_parseable = 0
    used = set()
    with open(a.chains) as src, open(tmp, "w") as dst:
        for line in src:
            row = json.loads(line)
            key = (row["qkey"], int(row["rep"]))
            fb = fbs.get(key)
            new = derive_row(row, fb)
            if fb is not None:
                used.add(key); n_applied += 1
                n_parseable += new["answer"] is not None
                n_flip_up += bool(new["correct"]) and not bool(row["correct"])
            dst.write(json.dumps(new) + "\n"); n += 1
    unused = set(fbs) - used
    if unused:
        raise SystemExit(f"{len(unused)} fallback rows match no bank chain, e.g. {sorted(unused)[:3]}")
    tmp.rename(out)
    summary = {"chains": {"path": a.chains, "sha256": sha256(a.chains)},
               "fallback": {"path": a.fallback, "sha256": sha256(a.fallback)},
               "out": {"path": str(out), "sha256": sha256(out)},
               "n_rows": n, "n_fallback_applied": n_applied, "n_fallback_parseable": n_parseable,
               "n_correct_0_to_1": n_flip_up}
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
