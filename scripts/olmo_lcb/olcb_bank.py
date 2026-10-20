from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--split", default="report")
    ap.add_argument("--chains-per-question", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if sha(a.manifest) != a.manifest_sha256:
        raise SystemExit(f"{a.manifest}: sha256 differs from {a.manifest_sha256}")
    want_q = set()
    for line in open(a.manifest):
        if line.strip():
            r = json.loads(line)
            if r.get("split", a.split) == a.split:
                want_q.add(r["qkey"])
    shards = sorted(glob.glob(f"{a.shards}/shard_*/report"), key=lambda p: int(p.split("shard_")[1].split("/")[0]))
    chains, programs, inputs = [], [], []
    for d in shards:
        d = Path(d)
        marker = (d / "chains.jsonl.sha256").read_text().split()[0]
        got = sha(d / "chains.jsonl")
        if got != marker:
            raise SystemExit(f"{d}/chains.jsonl sha256 {got} != marker {marker}")
        inputs.append({"shard": str(d), "chains_sha256": got, "programs_sha256": sha(d / "programs.jsonl")})
        chains += [json.loads(l) for l in open(d / "chains.jsonl") if l.strip()]
        programs += [json.loads(l) for l in open(d / "programs.jsonl") if l.strip()]
    ids = Counter((c["qkey"], int(c["rep"])) for c in chains)
    dup = [k for k, n in ids.items() if n > 1]
    if dup:
        raise SystemExit(f"{len(dup)} duplicate chains, e.g. {dup[:3]}")
    got_q = {c["qkey"] for c in chains}
    if got_q != want_q:
        raise SystemExit(f"qkeys: {len(got_q)} in the bank, {len(want_q)} in the manifest; "
                         f"missing {sorted(want_q - got_q)[:3]}, extra {sorted(got_q - want_q)[:3]}")
    per_q = Counter(c["qkey"] for c in chains)
    bad_q = {q: n for q, n in per_q.items() if n != a.chains_per_question}
    if bad_q:
        raise SystemExit(f"{len(bad_q)} questions without {a.chains_per_question} chains, e.g. {list(bad_q.items())[:3]}")
    nat = {}
    for p in programs:
        if p.get("kind") == "natural":
            k = (p["qkey"], int(p["rep"]))
            if k in nat:
                raise SystemExit(f"two natural programs for {k}")
            nat[k] = p
    k1, n_closed = [], 0
    for c in sorted(chains, key=lambda c: (c["qkey"], int(c["rep"]))):
        k = (c["qkey"], int(c["rep"]))
        if c["outcome"] == "natural":
            n_closed += 1
            p = nat.get(k)
            if p is None:
                raise SystemExit(f"closed chain {k} has no natural program")
            if int(p["prefix_len"]) != int(c["thinking_tokens"]):
                raise SystemExit(f"natural program of {k}: prefix_len {p['prefix_len']} != thinking {c['thinking_tokens']}")
            k1.append({"qkey": k[0], "rep": k[1], "natural_fallback_tokens": int(p["tokens"])})
        else:
            k1.append({"qkey": k[0], "rep": k[1], "natural_fallback_tokens": 0})
    extra_nat = set(nat) - {(c["qkey"], int(c["rep"])) for c in chains if c["outcome"] == "natural"}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"chains": out / f"{a.pool}.chains.jsonl", "natural_programs": out / f"{a.pool}.natural_programs.jsonl",
             "k1": out / f"{a.pool}.k1.jsonl"}
    with open(paths["chains"], "w") as f:
        for c in sorted(chains, key=lambda c: (c["qkey"], int(c["rep"]))):
            f.write(json.dumps(c) + "\n")
    with open(paths["natural_programs"], "w") as f:
        for k in sorted(nat):
            f.write(json.dumps(nat[k]) + "\n")
    with open(paths["k1"], "w") as f:
        for r in k1:
            f.write(json.dumps(r) + "\n")
    rep = {"pool": a.pool, "split": a.split, "manifest": a.manifest, "manifest_sha256": a.manifest_sha256,
           "questions": len(got_q), "chains": len(chains), "closed_chains": n_closed,
           "length_chains": sum(c["outcome"] == "length" for c in chains),
           "natural_correct": sum(bool(c["correct"]) for c in chains),
           "natural_programs": len(nat), "natural_programs_on_unclosed_chains": len(extra_nat),
           "natural_answer_tokens_total": sum(r["natural_fallback_tokens"] for r in k1),
           "shards": len(shards), "inputs": inputs,
           "outputs": {k: {"path": str(v), "sha256": sha(v)} for k, v in paths.items()}}
    (out / f"{a.pool}.bank.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("inputs",)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
