#!/usr/bin/env python3
"""Assemble a small Qwen3-1.7B BELT trajectory sample from archived paper artifacts.

Runs from a BELT release checkout with the root Qwen runtime first on PYTHONPATH.
Reads only existing files; generates nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SELECT_SALT = "belt-sample-v1"
WIDTH = 768


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def rank(text: str) -> str:
    return hashlib.sha256(f"{SELECT_SALT}:{text}".encode()).hexdigest()


def stream_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def category(row) -> str:
    if row["outcome"] == "length":
        return "length_capped"
    if row["outcome"] == "natural":
        return "natural_elicited" if int(row.get("natural_fallback_tokens") or 0) > 0 else "natural_parseable"
    return f"other:{row['outcome']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--chains", required=True, help="original report chains.jsonl")
    ap.add_argument("--fallback-bank", required=True, help="chains.fallback.jsonl")
    ap.add_argument("--scored", required=True, help="select_scored.jsonl of the scored pool")
    ap.add_argument("--scored-summary", required=True)
    ap.add_argument("--repr-dir", required=True, help="vectors_v1 directory with LB_H.*")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--prefer-source", default="math500")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = {"inputs": {}}
    for name in ("manifest", "chains", "fallback_bank", "scored", "scored_summary"):
        p = getattr(a, name)
        log["inputs"][name] = {"path": p, "sha256": sha_file(p)}
    repr_dir = Path(a.repr_dir)
    for name in ("LB_H.f16", "LB_H.index.jsonl", "LB_H.source.json"):
        log["inputs"][name] = {"path": str(repr_dir / name), "sha256": sha_file(repr_dir / name)}

    summary = json.load(open(a.scored_summary))
    declared_chains = summary["provenance"]["chains"]["sha256"]
    declared_manifest = summary["provenance"]["questions"]["sha256"]
    source_json = json.load(open(repr_dir / "LB_H.source.json"))
    assert log["inputs"]["chains"]["sha256"] == declared_chains, "chains differ from scored summary"
    assert log["inputs"]["manifest"]["sha256"] == declared_manifest, "manifest differs from scored summary"
    assert source_json["chains_sha256"] == declared_chains
    assert source_json["checkpoint_sha256"] == summary["provenance"]["s1_checkpoint"]["sha256"]

    manifest = {r["qkey"]: r for r in stream_jsonl(a.manifest)}
    meta = {}
    for r in stream_jsonl(a.fallback_bank):
        r.pop("completion_ids", None)
        meta[(r["qkey"], r["rep"])] = r
    cats = Counter(category(r) for r in meta.values())
    log["category_counts_all_chains"] = dict(cats)

    by_q = defaultdict(list)
    for key, r in meta.items():
        by_q[key[0]].append(r)
    log["questions_per_source"] = dict(Counter(manifest[q]["source"] for q in by_q))
    needs = [{"natural_parseable", "natural_elicited"}, {"natural_parseable", "length_capped"}]

    def eligible(need, src, taken):
        return sorted((q for q, rows in by_q.items()
                       if q not in taken and (src is None or manifest[q]["source"] == src)
                       and need <= {category(x) for x in rows}), key=rank)

    log["selection"] = {"rule": "for each required category set, the question whose chains cover it, "
                                f"source={a.prefer_source!r} if any, with minimum "
                                f"sha256('{SELECT_SALT}:'+qkey); all chains of each chosen question",
                        "picks": []}
    chosen = []
    for need in needs:
        cands = eligible(need, a.prefer_source, set(chosen))
        src = a.prefer_source
        if not cands:
            cands, src = eligible(need, None, set(chosen)), None
        if not cands:
            raise SystemExit(f"no question covers {sorted(need)}")
        chosen.append(cands[0])
        log["selection"]["picks"].append({"need": sorted(need), "source_filter": src,
                                          "candidates": len(cands), "qkey": cands[0],
                                          "n_chains": len(by_q[cands[0]])})
    keys = [(q, rep) for q in chosen for rep in sorted(r["rep"] for r in by_q[q])]
    keyset = set(keys)

    scored = {}
    for r in stream_jsonl(a.scored):
        k = (r["qkey"], r["rep"])
        if k in keyset:
            scored[k] = r
    comp = {}
    for r in stream_jsonl(a.chains):
        k = (r["qkey"], r["rep"])
        if k in keyset:
            comp[k] = r
    fb_comp = {}
    for r in stream_jsonl(a.fallback_bank):
        k = (r["qkey"], r["rep"])
        if k in keyset:
            fb_comp[k] = r["completion_ids"]
    assert set(scored) == set(keys) == set(comp) == set(fb_comp)

    index = {}
    for r in stream_jsonl(repr_dir / "LB_H.index.jsonl"):
        k = (r["qkey"], r["rep"])
        if k in keyset:
            index[k] = r
    assert set(index) == set(keys)

    from transformers import AutoTokenizer
    import transformers
    from rl_controller import rollout_contract as rc

    tok = AutoTokenizer.from_pretrained(a.model_path)
    encode = lambda s: tok.encode(s, add_special_tokens=False)
    decode = lambda ids: tok.decode(ids)
    prompts = {qk: rc.prompt_ids_for(tok, str(manifest[qk]["question"]), answer_format="boxed",
                                     encode=encode, decode=decode) for qk in chosen}
    forced_suffix = list(encode(rc.NATURAL_CLOSE + rc.BOXED_SEED))
    log["tokenizer"] = {"model_path": a.model_path, "transformers": transformers.__version__,
                        "contract": rc.active_prompt_contract()}

    checks = defaultdict(list)
    chain_rows = []
    h_parts = []
    offset = 0
    with open(repr_dir / "LB_H.f16", "rb") as dump:
        for i, k in enumerate(keys):
            s, c, m = scored[k], comp[k], meta[k]
            q, qkey, prompt_ids = manifest[k[0]], k[0], prompts[k[0]]
            ids = list(c["completion_ids"])
            for field in ("gold", "outcome", "blocks", "thinking_tokens", "block_ends",
                          "query_points", "probes"):
                if s[field] != c[field] or m[field] != c[field]:
                    checks["field_mismatch"].append([k, field])
            if fb_comp[k] != ids:
                checks["fallback_bank_completion_ids_differ"].append(k)
            if s["answer"] != c["answer"] or s["correct"] != c["correct"]:
                checks["scored_answer_differs_from_chains"].append(k)
            ix = index[k]
            n = int(ix["n_probes"])
            if n != len(s["probes"]) or n != len(s["scores"]):
                checks["probe_count_mismatch"].append([k, n, len(s["probes"]), len(s["scores"])])
            dump.seek(int(ix["offset"]))
            raw = dump.read(n * WIDTH * 2)
            assert len(raw) == n * WIDTH * 2
            arr = np.frombuffer(raw, dtype=np.float16).reshape(n, WIDTH)
            if not np.isfinite(arr.astype(np.float32)).all():
                checks["nonfinite_h"].append(k)
            h_parts.append(raw)
            ends = [int(e) for e in c["block_ends"]]
            probe_pos = [ends.index(int(qp[1])) for qp in c["query_points"]]
            for p, qp in zip(c["probes"], c["query_points"]):
                if [p["block"], p["prefix_len"]] != [qp[0], qp[1]]:
                    checks["probe_vs_query_point"].append([k, p["block"], qp])
                if not decode(ids[: p["prefix_len"]]).endswith("\n\n"):
                    checks["prefix_not_on_blank_line"].append([k, p["prefix_len"]])
            if c["thinking_tokens"] != len(ids) and c["outcome"] == "natural":
                checks.setdefault("thinking_tokens_vs_len_completion", []).append(
                    [k, c["thinking_tokens"], len(ids)])
            text = decode(ids)
            natural = {
                "answer": m["answer"], "correct": m["correct"],
                "answer_before_elicitation": m.get("answer_raw", m["answer"]),
                "correct_before_elicitation": m.get("correct_raw", m["correct"]),
                "elicited": category(m) == "natural_elicited",
                "elicitation_tokens": int(m.get("natural_fallback_tokens") or 0),
            }
            row = {
                "sample_index": i,
                "pool": "math_ood_report", "split": q["split"],
                "source": q["source"], "source_id": q["source_id"], "level": q.get("level"),
                "qkey": qkey, "rep": k[1], "question": q["question"],
                "question_sha256": q["question_sha256"], "gold": q["gold"],
                "prompt_ids": prompt_ids, "completion_ids": ids,
                "outcome": c["outcome"], "category": category(m),
                "think_closed": rc.CLOSE_TAG in text,
                "thinking_tokens": c["thinking_tokens"], "blocks": c["blocks"],
                "block_ends": ends, "query_points": c["query_points"],
                "probe_block_end_index": probe_pos,
                "probes": c["probes"], "scores": s["scores"],
                "natural": natural,
                "forced_answer_suffix_ids": forced_suffix,
                "h": {"file": "belt_h.f16", "offset": offset, "n_probes": n},
            }
            if natural["elicited"]:
                full = rc.natural_boxed_prompt(ids, encode=encode, decode=decode)
                assert full[: len(ids)] == ids
                row["natural_elicitation_suffix_ids"] = full[len(ids):]
            chain_rows.append(row)
            offset += len(raw)

    with open(out / "chains.jsonl", "w") as f:
        for r in chain_rows:
            f.write(json.dumps(r) + "\n")
    with open(out / "belt_h.f16", "wb") as f:
        for part in h_parts:
            f.write(part)
    with open(out / "belt_h.index.jsonl", "w") as f:
        for r in chain_rows:
            f.write(json.dumps({"qkey": r["qkey"], "rep": r["rep"], "n_probes": r["h"]["n_probes"],
                                "offset": r["h"]["offset"]}) + "\n")
    src = dict(source_json)
    src["parent_dump_sha256"] = log["inputs"]["LB_H.f16"]["sha256"]
    src["parent_index_sha256"] = log["inputs"]["LB_H.index.jsonl"]["sha256"]
    src["subset"] = [[r["qkey"], r["rep"]] for r in chain_rows]
    with open(out / "belt_h.source.json", "w") as f:
        json.dump(src, f, indent=1)
    log["checks"] = {k: v for k, v in checks.items()}
    log["n_chains"] = len(chain_rows)
    log["categories"] = Counter(r["category"] for r in chain_rows)
    log["scored_summary"] = summary
    with open(out / "build_log.json", "w") as f:
        json.dump(log, f, indent=1, default=str)
    print(json.dumps({k: log[k] for k in ("selection", "n_chains", "categories", "checks",
                                           "category_counts_all_chains", "tokenizer")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
