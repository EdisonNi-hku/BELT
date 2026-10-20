from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import alf_members as M

TARGETS = (0.20, 0.35, 0.50, 0.65, 0.80)
GRID = 4001


class TreeView:

    def __init__(self, tree: str, episodes: str = "episodes.jsonl"):
        import alf_scorer as SC
        import alf_tree as T

        self.nodes, self.reprs_of = {}, {}
        for lane in sorted(glob.glob(f"{tree}/lane*/")):
            st = T.lane_store(Path(lane), "calibration-readonly", readonly=True)
            rs = SC.ReprStore(Path(lane), "calibration-readonly")
            for k, n in st.nodes.items():
                self.nodes.setdefault(k, n)
                self.reprs_of.setdefault(k, rs)
        self.games = {}
        for f in sorted(glob.glob(f"{tree}/lane*/{episodes}")):
            for line in open(f):
                e = json.loads(line)
                if e["member"] == "natural":
                    self.games[(e["game"], e["seed"])] = [t["key"] for t in e["trace"]]

    def get(self, key, scorer):
        import alf_lazy as AL

        return AL.MergedReprs(self.reprs_of[key], AL.merged_parts(self.nodes[key])).get(key, scorer)


SKIP_UNSCORED = False


def node_arrays(view: TreeView, feature: dict):
    out = {}
    for keys in view.games.values():
        for k in keys:
            if k in out:
                continue
            import alf_lazy as AL

            n = {**view.nodes[k], **AL.merged_parts(view.nodes[k])}
            full = n["full"]
            pos = full["positions"]
            probes = [p for p in pos[1:] if p["block"] != "cap"]
            vals = M.feature(feature, n, k, view) if probes else []
            if vals is None or (SKIP_UNSCORED and len(vals) != len(probes)):
                if SKIP_UNSCORED:
                    out[k] = (np.asarray([]), np.asarray([], dtype=int), full["thinking_tokens"])
                    continue
                raise SystemExit(f"node {k} lacks {feature['scorer']} scores")
            if len(vals) != len(probes):
                raise SystemExit(f"node {k}: {len(vals)} feature values for {len(probes)} probes")
            cap = pos[-1]["prefix_len"] if full["outcome"] == "length" else None
            natural = cap if cap is not None else full["thinking_tokens"]
            out[k] = (np.asarray(vals, dtype=np.float64), np.asarray([p["prefix_len"] for p in probes]), natural)
    return out


def saving_curve(view: TreeView, arrays: dict, taus: np.ndarray, per_game: bool = False) -> np.ndarray:
    kept = {}
    for k, (vals, pref, natural) in arrays.items():
        if len(vals) == 0:
            kept[k] = np.full(len(taus), float(natural))
            continue
        pm = np.maximum.accumulate(vals)
        j = np.searchsorted(pm, taus, side="left")
        stop = np.where(j < len(pref), pref[np.minimum(j, len(pref) - 1)], natural)
        kept[k] = np.minimum(stop, natural).astype(np.float64)
    if not per_game:
        keys = [k for ks in view.games.values() for k in ks]
        return 1.0 - sum(kept[k] for k in keys) / float(sum(arrays[k][2] for k in keys))
    games = []
    for keys in view.games.values():
        nat = sum(arrays[k][2] for k in keys)
        if nat > 0:
            games.append(1.0 - sum(kept[k] for k in keys) / nat)
    return np.mean(np.stack(games), axis=0)


def choose(savings: np.ndarray, labels: list) -> list:
    picks = []
    for t in TARGETS:
        i = int(np.argmin(np.abs(savings - t)))
        picks.append({"target": t, "index": i, "label": labels[i], "projected_saving": float(savings[i])})
    flags = []
    for p in picks:
        if not 0.20 <= p["projected_saving"] <= 0.80:
            flags.append(f"target {p['target']:.2f}: nearest candidate saves {p['projected_saving']:.3f}, outside 20-80 %")
    seen = {}
    for p in picks:
        seen.setdefault(p["index"], []).append(p["target"])
    flags += [f"targets {v} land on the same candidate" for v in seen.values() if len(v) > 1]
    return picks, flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--episodes", default="episodes.jsonl", help="episode file name in each lane dir (natural member)")
    ap.add_argument("--skip-unscored", action="store_true", help="tests only")
    ap.add_argument("--targets", default="0.20,0.35,0.50,0.65,0.80",
                    help="projected-saving targets (D22: 0.20,0.35,0.50,0.65 for every calibration after the seed-0 runs)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global SKIP_UNSCORED, TARGETS
    SKIP_UNSCORED = a.skip_unscored
    TARGETS = tuple(float(t) for t in a.targets.split(","))
    view = TreeView(a.tree, a.episodes)
    fams = json.loads(Path(a.families).read_text())
    report = {"tree": a.tree, "games": len(view.games), "natural_nodes": sum(len(v) for v in view.games.values()),
              "targets": TARGETS, "families": []}
    for fam in fams:
        rec = {k: fam[k] for k in ("family", "seed", "kind") if k in fam}
        if fam["kind"] == "threshold":
            arrays = node_arrays(view, fam["feature"])
            vals = [v for v, _, _ in arrays.values() if len(v)]
            if not vals:
                rec.update(picks=[], flags=["no scored probe on the tree"])
                report["families"].append(rec)
                print(json.dumps({"family": rec.get("family"), "seed": rec.get("seed"), "flags": rec["flags"]}))
                continue
            allv = np.concatenate(vals)
            taus = np.unique(np.quantile(allv, np.linspace(0, 1, GRID)))
            taus = np.concatenate([taus, [np.inf]])
            s = saving_curve(view, arrays, taus)
            picks, flags = choose(s, [float(t) for t in taus])
            sg = saving_curve(view, arrays, taus, per_game=True)
            for p in picks:
                p["tau"] = p.pop("label")
                p["projected_saving_per_game"] = float(sg[p["index"]])
            rec.update(picks=picks, flags=flags, feature=fam["feature"], range=[float(s.min()), float(s.max())])
            if fam.get("math_taus"):
                ids = list(fam["math_taus"])
                mt = np.asarray([fam["math_taus"][i] for i in ids], dtype=np.float64)
                ms, mg = saving_curve(view, arrays, mt), saving_curve(view, arrays, mt, per_game=True)
                rec["math_thresholds"] = {i: {"tau": fam["math_taus"][i], "projected_saving": float(v),
                                              "projected_saving_per_game": float(g)}
                                          for i, v, g in zip(ids, ms, mg)}
        elif fam["kind"] == "members":
            savings, labels = [], []
            cache = {}
            for m in fam["members"]:
                fk = json.dumps(m["feature"], sort_keys=True)
                if fk not in cache:
                    cache[fk] = node_arrays(view, m["feature"])
                savings.append(float(saving_curve(view, cache[fk], np.asarray([float(m["tau"])]))[0]))
                labels.append(m["name"])
            picks, flags = choose(np.asarray(savings), labels)
            for p in picks:
                p["member"] = fam["members"][p["index"]]
            rec.update(picks=picks, flags=flags, candidates=len(labels),
                       range=[float(min(savings)), float(max(savings))],
                       all_savings=dict(zip(labels, savings)))
        else:
            raise SystemExit(f"unknown family kind {fam['kind']}")
        report["families"].append(rec)
        print(json.dumps({"family": rec.get("family"), "seed": rec.get("seed"),
                          "picks": [(p["target"], round(p["projected_saving"], 4)) for p in rec["picks"]],
                          "flags": rec["flags"]}), flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
