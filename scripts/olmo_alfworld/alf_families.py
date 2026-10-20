from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TARGET_NAMES = {0.20: "t20", 0.35: "t35", 0.50: "t50", 0.65: "t65", 0.80: "t80"}


def families(params: Path, osp1: str, osp1_seed_dir: str = "s5-1025") -> list:
    out = []
    for k in (0, 1, 2):
        f = params / "x13fits" / f"LB_H_1_s{k}.json"
        fit = json.loads(f.read_text())
        out.append({"family": "H-T", "seed": k, "kind": "threshold",
                    "feature": {"type": "repr_linear", "scorer": f"belt-s{k}", "source": {"file": str(f), "model": None}},
                    "math_taus": {f"k{i:02d}": t for i, t in enumerate(fit["thresholds"]["tau_logit"])}})
    for k in (0, 1, 2):
        f = params / "x13fits" / f"LB_QVRS_1_s{k}.json"
        fit = json.loads(f.read_text())
        out.append({"family": "B-T", "seed": k, "kind": "threshold",
                    "feature": {"type": "belief_linear", "scorer": f"belt-s{k}", "source": {"file": str(f), "model": None}},
                    "math_taus": {f"k{i:02d}": t for i, t in enumerate(fit["thresholds"]["tau_logit"])}})
    for fam, fname, key, ftype in (("H-R", "weights_h.json", "H", "repr_linear"),
                                   ("B-R", "weights.json", "QVRS", "belief_linear")):
        f = params / "campaign_s0" / fname
        models = json.loads(f.read_text())["models"][key]
        if len(models) != 17 or any(len(m["taus"]) != 13 for m in models):
            raise SystemExit(f"{f}: expected 17 models x 13 taus")
        members = [{"name": f"l{l:02d},k{kk:02d}", "tau": float(m["taus"][kk]),
                    "feature": {"type": ftype, "scorer": "belt-s0", "source": {"file": str(f), "model": [key, l]}}}
                   for l, m in enumerate(models) for kk in range(13)]
        out.append({"family": fam, "seed": 0, "kind": "members", "members": members})
    rec = json.loads((params / "osp1_thresholds" / osp1_seed_dir / "threshold_receipt.json").read_text())
    taus = rec["pools"][0]["seeds"][0]["pi_thresholds"]
    out.append({"family": "OSP-1", "seed": 0, "kind": "threshold",
                "feature": {"type": "field", "scorer": f"osp:{osp1}", "field": "pi"},
                "math_taus": {f"vq{i:02d}": t for i, t in enumerate(taus)}})
    return out


def members(calibration: dict, block: str) -> list:
    out = []
    for fam in calibration["families"]:
        if fam["family"] != block:
            continue
        for p in fam["picks"]:
            name = f"{block}-s{fam['seed']}-{TARGET_NAMES[round(p['target'], 2)]}"
            if fam["kind"] == "threshold":
                out.append({"name": name, "kind": "threshold", "feature": fam["feature"], "tau": p["tau"],
                            "calibration": {k: p[k] for k in p if k not in ("tau",)}})
            else:
                m = p["member"]
                out.append({"name": name, "kind": "threshold", "feature": m["feature"], "tau": m["tau"],
                            "calibration": {"math_member": m["name"], "target": p["target"],
                                            "projected_saving": p["projected_saving"]}})
    names = [m["name"] for m in out]
    if len(set(names)) != len(names) or not out:
        raise SystemExit(f"block {block}: {len(out)} members, names {names}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("families", "members"))
    ap.add_argument("--params")
    ap.add_argument("--osp1", default="osp1-s0")
    ap.add_argument("--calibration")
    ap.add_argument("--block")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "families":
        data = families(Path(a.params), a.osp1)
    else:
        data = members(json.loads(Path(a.calibration).read_text()), a.block)
    Path(a.out).write_text(json.dumps(data, indent=1) + "\n")
    print(json.dumps({"written": a.out, "entries": len(data)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
