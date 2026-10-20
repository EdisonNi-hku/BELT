from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d53lib as L

LAMBDAS = ("2e-6", "5e-6", "1e-5", "2e-5", "4e-5")
SEEDS = (0, 1, 2)
DEV160 = L.E / "two_belief/seed_slices/policy_dev160.json"


def replay_q(chains, osp, *, grade="loose"):
    per_q: dict[str, list[float]] = {}
    for chain in chains:
        row = osp.replay_chain(chain, pi_key="pi", grade=grade)
        acc = per_q.setdefault(row["qkey"], [0.0, 0.0, 0.0, 0.0])
        acc[0] += 1
        acc[1] += row["e_correct"]
        acc[2] += row["e_thinking_tokens"]
        acc[3] += row["e_answer_tokens"]
    out = {}
    for qkey in sorted(per_q):
        n, correct, thinking, answer = per_q[qkey]
        e_t, e_a = thinking / n, answer / n
        e_c, e_g, aux = correct / n, e_t + e_a, 0.0
        nn = float(int(n))
        c, t, g, a = e_c * nn, e_t * nn, e_g * nn, aux * nn
        s = (1.0, c / nn, t / nn, g / nn, a / nn)
        out[qkey] = [s[1] / s[0], s[2] / s[0], (s[3] + s[4]) / s[0]]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--producer", required=True)
    ap.add_argument("--pool", required=True, choices=L.POOLS)
    ap.add_argument("--which", required=True, choices=("full", "single"))
    ap.add_argument("--seed", type=int, required=True, choices=SEEDS)
    ap.add_argument("--archived", default=None, help="the paper's baselines_<pool>.json.gz")
    ap.add_argument("--records", default=None, help="archived single-threshold records (threshold cross-check)")
    ap.add_argument("--scored-root", default=str(L.E / "two_belief/scored_osp"))
    ap.add_argument("--val-root", default=str(L.E / "two_belief/oalign"))
    ap.add_argument("--lane-suffix", default="", help="v1/v2: the retrained lanes' tag, e.g. '-fb' (lam1e-5_s0-fb)")
    ap.add_argument("--no-gate", action="store_true", help="retrained checkpoints: no archived rows to compare")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    L.setup(a.producer, needs=("chain_terms_total", "replay_chain"))
    from rl_controller.iclr import osp_policy as osp
    dev = set(json.loads(DEV160.read_text())["qkeys"])
    lams = LAMBDAS if a.which == "full" else ("1e-5",)
    rows = {"orig": [], "fb": []}
    inputs, thresholds_rec = [], {}
    for lam in lams:
        lane = f"lam{lam}_s{a.seed}{a.lane_suffix}"
        dv = L.load_dump(Path(a.val_root) / lane / "val_scored.jsonl", "val")
        dp = L.load_dump(Path(a.scored_root) / a.pool / lane / "select_scored.jsonl", a.pool)
        inputs += [{"lane": lane, "split": "val", **dv["record"]}, {"lane": lane, "split": "select", **dp["record"]}]
        if a.which == "full":
            method = "OSP@dev160"
            members = [(f"OSP@dev160:lam{float(lam):.0e}", None)]
        else:
            from rl_controller.scripts import iclr_single_checkpoint_panel as scp
            taus = scp.source_val_thresholds(dv["orig"])
            thresholds_rec[lane] = taus
            method = "OSP-single-threshold@dev160"
            members = [(f"OSP-single-threshold@dev160:vq{i:02d}", tau) for i, tau in enumerate(taus)]
        for variant in ("orig", "fb"):
            val_dev = [c for c in dv[variant] if c["qkey"] in dev]
            for member, tau in members:
                for split, cs in (("dev", val_dev), ("pool", dp[variant])):
                    if tau is not None:
                        cs = scp.fire_at(cs, tau)
                    rows[variant].append({"method": method, "member": member, "seed": a.seed,
                                          "split": split, "q": replay_q(cs, osp)})
        del dv, dp
    method = "OSP@dev160" if a.which == "full" else "OSP-single-threshold@dev160"
    if a.no_gate:
        L.write_shard(a.out, f"osp_{a.which}_{a.pool}_s{a.seed}", rows, {
            "family": f"osp_{a.which}", "pool": a.pool, "seed": a.seed, "grade": "loose", "lane_suffix": a.lane_suffix,
            "dev160": {"path": str(DEV160), "sha256": L.sha_file(DEV160), "n": len(dev)},
            "thresholds": thresholds_rec, "inputs": inputs, "gate": None})
        return 0
    archived = L.load_archived(a.archived)
    got_members = {r["member"] for r in rows["orig"]}
    arch_members = {k[1] for k in archived if k[0] == method and k[2] == a.seed}
    if got_members != arch_members:
        raise SystemExit(f"member names {sorted(got_members)} != archived {sorted(arch_members)}")
    want = lambda k: k[0] == method and k[2] == a.seed
    g = L.gate(rows["orig"], archived, keys_filter=want)
    L.log("gate", json.dumps({k: g[k] for k in ("n_series", "n_values", "n_exact", "max_abs_diff", "n_missing", "n_extra", "exact")}))
    L.write_shard(a.out, f"osp_{a.which}_{a.pool}_s{a.seed}", rows, {
        "family": f"osp_{a.which}", "pool": a.pool, "seed": a.seed, "grade": "loose",
        "dev160": {"path": str(DEV160), "sha256": L.sha_file(DEV160), "n": len(dev)},
        "thresholds": thresholds_rec, "inputs": inputs,
        "archived": {"path": a.archived, "sha256": L.sha_file(a.archived)}, "gate": g})
    return 0 if g["exact"] else 3


if __name__ == "__main__":
    sys.exit(main())
