from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--k1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-chains", type=int, default=0)
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    k1 = {}
    for line in open(a.k1):
        if line.strip():
            r = json.loads(line)
            k1[(r["qkey"], int(r["rep"]))] = int(r["natural_fallback_tokens"])
    out.mkdir(parents=True, exist_ok=True)
    n = charged = total = 0
    with open(run / "select_scored.jsonl") as fi, open(out / "select_scored.jsonl.tmp", "w") as fo:
        for line in fi:
            if not line.strip():
                continue
            c = json.loads(line)
            k = (c["qkey"], int(c["rep"]))
            if k not in k1:
                raise SystemExit(f"{k} is not in the K1 sidecar")
            if "natural_fallback_tokens" in c:
                raise SystemExit(f"{k} already carries natural_fallback_tokens")
            v = k1[k] if c["outcome"] == "natural" else 0
            if c["outcome"] != "natural" and k1[k]:
                raise SystemExit(f"{k}: capped chain with a natural charge in the sidecar")
            c["natural_fallback_tokens"] = v
            n += 1
            charged += v > 0
            total += v
            fo.write(json.dumps(c, separators=(",", ":")) + "\n")
    if n != len(k1):
        raise SystemExit(f"{n} select rows against {len(k1)} sidecar rows")
    if a.expect_chains and n != a.expect_chains:
        raise SystemExit(f"{n} chains, expected {a.expect_chains}")
    (out / "select_scored.jsonl.tmp").replace(out / "select_scored.jsonl")
    for f in ("summary.json", "val_scored.jsonl"):
        if (run / f).exists():
            shutil.copyfile(run / f, out / f)
    rec = {"schema": "olcb-k1-join/v1", "run": str(run.resolve()), "select_in_sha256": sha(run / "select_scored.jsonl"),
           "k1": str(Path(a.k1).resolve()), "k1_sha256": sha(a.k1), "rows": n, "charged_rows": charged,
           "natural_answer_tokens_total": total, "select_out_sha256": sha(out / "select_scored.jsonl")}
    (out / "k1_join.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
