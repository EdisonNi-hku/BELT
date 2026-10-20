"""Fit representation-input linear stopping policies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from rl_controller.iclr import families_learned_linear as lb
from rl_controller.iclr import learned_linear_h as lh
from rl_controller.scripts import iclr_learned_linear_campaign as camp


def digest(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def index_path(npy: Path) -> Path:
    if npy.suffix != ".npy":
        raise ValueError(f"{npy}: expected the scorer's .npy matrix")
    return npy.with_suffix(".index.jsonl")


def read_chains(path: Path, budget):
    from rl_controller.iclr.chains import load_chains
    chains = load_chains(str(path))
    for chain in chains:
        camp.rules.validate_chain(chain, budget=budget)
    return chains


def merge_codes(source: Path, out: Path, codes):
    key = lb.score_key(lh.FEATURE_SET)
    table = codes[lh.FEATURE_SET].T.tolist()
    tmp, offset = out.with_name(out.name + ".tmp"), 0
    with open(source) as src, open(tmp, "w") as dst:
        for line in src:
            chain = json.loads(line)
            if offset + len(chain["scores"]) > len(table):
                raise ValueError(f"{source.name}: the dump has more probes than the "
                                 f"{len(table)} code rows cover")
            scores = []
            for i, score in enumerate(chain["scores"]):
                if key in score:
                    raise ValueError(f"{chain['qkey']}: {key} is already present in the source dump")
                scores.append({**score, key: table[offset + i]})
            offset += len(chain["probes"])
            dst.write(json.dumps(dict(chain, scores=scores), separators=(",", ":")) + "\n")
    if offset != len(table):
        raise ValueError(f"{source.name}: {offset} probes against {len(table)} code rows")
    tmp.replace(out)
    return offset


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaign", required=True, help="the lane B campaign out dir (holds LOCK.json)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reprs-val", required=True, help="768-d rows for the reader's validation split")
    p.add_argument("--reprs-select", required=True, help="768-d rows for the evaluation pool")
    p.add_argument("--out", required=True, help="the merged derived run dir to write")
    args = p.parse_args(argv)

    started = time.monotonic()
    campaign, out = Path(args.campaign), Path(args.out)
    lock = json.loads((campaign / "LOCK.json").read_text())
    budget = lock.get("budget")
    entry = next(e for e in camp.seed_entries(lock) if int(e["seed"]) == args.seed)
    source_run, derived = Path(entry["run_dir"]), campaign / "fits" / f"s{args.seed}" / "run"
    if not (campaign / "fits" / f"s{args.seed}" / "FIT_DONE.json").is_file():
        raise SystemExit(f"seed {args.seed} has no finished campaign fit to merge into")

    fit_set, dev_set = set(lock["split"]["fit"]), set(lock["split"]["dev"])
    val = read_chains(source_run / "val_scored.jsonl", budget)
    fit_chains = [c for c in val if c["qkey"] in fit_set]
    dev_chains = [c for c in val if c["qkey"] in dev_set]
    select = read_chains(source_run / "select_scored.jsonl", budget)

    val_npy, select_npy = Path(args.reprs_val), Path(args.reprs_select)
    features = lh.features_from(lh.load_reprs(val_npy, index_path(val_npy)))
    weights = lh.fit_models_h(fit_chains, dev_chains, features)
    weights.update(seed=args.seed, feature_set=lh.FEATURE_SET,
                   source_val_sha256=digest(source_run / "val_scored.jsonl"),
                   reprs_sha256={"val": digest(val_npy), "select": digest(select_npy)},
                   split={k: v for k, v in lock["split"].items() if k not in ("fit", "dev")})
    out.mkdir(parents=True, exist_ok=True)
    camp.atomic(out.parent / "weights_h.json", weights)

    select_features = lh.features_from(lh.load_reprs(select_npy, index_path(select_npy)))
    n_dev = merge_codes(derived / "val_scored.jsonl", out / "val_scored.jsonl",
                        lh.deploy_h(weights, dev_chains, features))
    n_select = merge_codes(derived / "select_scored.jsonl", out / "select_scored.jsonl",
                           lh.deploy_h(weights, select, select_features))

    summary = json.loads((derived / "summary.json").read_text())
    summary.setdefault("provenance", {})["learned_linear_h"] = dict(
        schema="olmo3-learned-linear-h/v1", amendment="A7", derived_from=str(derived.resolve()),
        source_sha256={"val_scored": digest(derived / "val_scored.jsonl"),
                       "select_scored": digest(derived / "select_scored.jsonl")},
        weights_sha256=digest(out.parent / "weights_h.json"),
        kappas=list(lh.KAPPAS_H), n_dev_probes=n_dev, n_select_probes=n_select,
        score_keys={lb.score_key(lh.FEATURE_SET):
                    "per probe: list over lambda index l of #{k: z_l >= tau_lk}"},
        note="A7 control; the registered verdict is on LB_QVRS alone")
    camp.atomic(out / "summary.json", summary)
    print(json.dumps({"fitted": "LB_H", "seed": args.seed, "n_dev_probes": n_dev,
                      "n_select_probes": n_select, "seconds": time.monotonic() - started}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
