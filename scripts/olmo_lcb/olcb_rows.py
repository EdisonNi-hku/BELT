from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts/belt_ext/d53"))
import d53lib as L

SPLIT = {"select": "pool", "val": "dev"}
LAMBDAS = ("2e-6", "5e-6", "1e-5", "2e-5", "4e-5")


def tags(seed):
    lb = "qvrs-lbdev" if seed == 0 else f"qvrs-lbdev-s{seed}"
    return {"BELT-B": f"LB_QVRS@{lb}", "BELT-H": f"LB_H@{lb}", "BELT-B-1": f"LB_QVRS_1@olmo-thr-s{seed}",
            "BELT-H-1": f"LB_H_1@olmo-thr-s{seed}"}


def osp1_method(seed):
    return "OSP-single-threshold@dev160" + ("" if seed == 0 else f"-s{seed}")


def paper_files(shadow: Path, seed: int) -> dict:
    ev = shadow / "evidence/olmo_20260921_1025"
    x13 = shadow / "evidence/x13_20260922/exports3" / f"exports3-s{seed}.json.gz"
    reader = ev / "baselines_math_ood_1025.json.gz" if seed == 0 else ev / "seeds" / f"export_reader_s{seed}.json.gz"
    osp1 = ev / "baselines_math_ood_1025.json.gz" if seed == 0 else ev / "osp_seeds" / f"export_osp-s{seed}.json.gz"
    return {"BELT-B": reader, "BELT-H": reader, "BELT-B-1": x13, "BELT-H-1": x13, "OSP-1": osp1}


def export_rows(path, method, split):
    d = json.load(gzip.open(path, "rt"))
    return {r["member"]: r["q"] for r in d["rows"] if r["method"] == method and r["split"] == split}


def load_records(path, method):
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["method_name"] != method or r["split"] not in SPLIT:
                continue
            if r.get("missing_reason") is not None:
                raise SystemExit(f"{path}: {r['config_id']} {r['qkey']} missing ({r['missing_reason']})")
            cell = out.setdefault((SPLIT[r["split"]], r["config_id"]), {})
            if r["qkey"] in cell:
                raise SystemExit(f"{path}: duplicate {r['config_id']} {r['qkey']}")
            cell[r["qkey"]] = [r["e_correct"], r["e_thinking_tokens"], r["e_generated_tokens"]]
    return out


def k1_join(chains, k1):
    out = []
    for c in chains:
        k = (c["qkey"], int(c["rep"]))
        v = k1[k] if c["outcome"] == "natural" else 0
        out.append(dict(c, natural_fallback_tokens=v))
    return out


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--producer", required=True)
    ap.add_argument("--members", required=True, help="members/<pool> (records/, lanes/s<k>/)")
    ap.add_argument("--osp-full-root", required=True)
    ap.add_argument("--osp1-run", required=True)
    ap.add_argument("--k1", required=True)
    ap.add_argument("--bank-json", required=True)
    ap.add_argument("--paper-shadow", required=True)
    ap.add_argument("--osp-full-delivery", required=True)
    ap.add_argument("--osp1-receipt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    setup = L.setup(a.producer, needs=("chain_terms_total", "replay_chain"))
    sys.path.insert(0, str(HERE.parents[1] / "scripts/belt_ext/d53"))
    import fam_osp as F
    from rl_controller.iclr import osp_policy as osp
    from rl_controller.scripts import iclr_single_checkpoint_panel as scp

    k1 = {(r["qkey"], int(r["rep"])): int(r["natural_fallback_tokens"]) for r in read_jsonl(a.k1)}
    bank_sha = json.loads(Path(a.bank_json).read_text())["outputs"]["chains"]["sha256"]

    def provenance(run, want_ckpt=None):
        prov = json.loads((Path(run) / "summary.json").read_text())["provenance"]
        if prov["chains"]["sha256"] != bank_sha:
            raise SystemExit(f"{run} scored chains {prov['chains']['sha256'][:12]}, not the pool bank {bank_sha[:12]}")
        ck = prov["osp_checkpoint"]["sha256"]
        if want_ckpt is not None and ck != want_ckpt:
            raise SystemExit(f"{run} scored with checkpoint {ck[:12]}, the threshold receipt names {want_ckpt[:12]}")
        return {"chains_sha256": prov["chains"]["sha256"], "osp_checkpoint_sha256": ck}
    pool_q = {q for q, _ in k1}
    members = Path(a.members)
    rows, gates = [], {}

    def add(method, member, split, q):
        if split == "pool" and set(q) != pool_q:
            raise SystemExit(f"{method} {member}: {len(q)} pool questions, the pool has {len(pool_q)}")
        rows.append({"method": method, "member": member, "seed": a.seed, "split": split, "q": q})

    lane = read_jsonl(members / "lanes" / f"s{a.seed}" / "select_scored.jsonl")
    if any("natural_fallback_tokens" not in c for c in lane):
        raise SystemExit("the BELT lane is not K1-charged")
    nat = F.replay_q([dict(c, scores=[dict(s, pi=0.0) for s in (c.get("scores") or [])]) for c in lane], osp)
    add("natural", "natural", "pool", nat)

    paper = paper_files(Path(a.paper_shadow), a.seed)
    for fam, method in tags(a.seed).items():
        rec = load_records(members / "records" / f"{method}_{a.pool}.policy_replay.jsonl", method)
        want_dev = export_rows(paper[fam], method, "dev")
        got_dev = {m: q for (s, m), q in rec.items() if s == "dev"}
        gates[f"{fam} dev == paper"] = {"members": len(want_dev), "exact": got_dev == want_dev,
                                        "n_values": sum(len(v) for v in want_dev.values())}
        for (split, member), q in sorted(rec.items()):
            add(method, member, split, q)

    deliv = Path(a.osp_full_delivery) / "osp_full" / f"export_osp_full-s{a.seed}.json.gz"
    dev_full = export_rows(deliv, "OSP@dev160", "dev")
    inputs = {}
    for lam in LAMBDAS:
        member = f"OSP@dev160:lam{float(lam):.0e}"
        run = Path(a.osp_full_root) / f"osp-lam{lam}-s{a.seed}-fb"
        chains = k1_join(read_jsonl(run / "select_scored.jsonl"), k1)
        inputs[member] = {"path": str(run / "select_scored.jsonl"), "sha256": L.sha_file(run / "select_scored.jsonl"),
                          **provenance(run)}
        add("OSP@dev160", member, "pool", F.replay_q(chains, osp))
        add("OSP@dev160", member, "dev", dev_full[member])

    rec = json.loads(Path(a.osp1_receipt).read_text())
    taus = rec["pools"][0]["seeds"][0]["pi_thresholds"]
    if len(taus) != 13 or any(b < a for a, b in zip(taus, taus[1:])):
        raise SystemExit("13 non-decreasing OSP-1 thresholds expected (vq00 = the lowest)")
    m1 = osp1_method(a.seed)
    dev1 = export_rows(paper["OSP-1"], m1, "dev")
    chains = k1_join(read_jsonl(Path(a.osp1_run) / "select_scored.jsonl"), k1)
    inputs[m1] = {"path": str(Path(a.osp1_run) / "select_scored.jsonl"),
                  "sha256": L.sha_file(Path(a.osp1_run) / "select_scored.jsonl"), "thresholds": taus,
                  **provenance(a.osp1_run, rec["pools"][0]["seeds"][0]["checkpoint_sha256"])}
    for i, tau in enumerate(taus):
        member = f"{m1}:vq{i:02d}"
        add(m1, member, "pool", F.replay_q(scp.fire_at(chains, tau), osp))
        add(m1, member, "dev", dev1[member])
    gates["OSP-1 never-fire == natural"] = {"exact": F.replay_q([dict(c, scores=[dict(s, pi=0.0) for s in (c.get("scores") or [])])
                                                                 for c in chains], osp) == nat}
    doc = {"schema": "olcb-perquestion/v1", "pool": a.pool, "seed": a.seed,
           "columns": ["strict_accuracy", "thinking_tokens", "total_tokens"], "rows": rows,
           "meta": {"lifting": "gen", "grade": "loose (= strict on code)", "natural_answer": "K1 natural program tokens",
                    "setup": setup, "inputs": inputs, "gates": gates,
                    "paper_exports": {k: {"path": str(v), "sha256": L.sha_file(v)} for k, v in paper.items()},
                    "osp_full_delivery_export": {"path": str(deliv), "sha256": L.sha_file(deliv)},
                    "osp1_receipt": {"path": a.osp1_receipt, "sha256": L.sha_file(a.osp1_receipt)}}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out + ".tmp", "wt") as f:
        json.dump(doc, f, separators=(",", ":"))
    Path(a.out + ".tmp").rename(a.out)
    ok = all(g.get("exact") for g in gates.values())
    print(json.dumps({"pool": a.pool, "seed": a.seed, "rows": len(rows), "gates": gates, "ok": ok}, indent=1))
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
