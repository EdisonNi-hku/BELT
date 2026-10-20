"""Read fitted linear policy families on another pool."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_controller.iclr import families_learned_linear as lb
from rl_controller.iclr import learned_linear_h as lh
from rl_controller.iclr import learned_linear_policy as lp

FROM_WEIGHTS = ("QVRS", "DEPTH")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_chains(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def checkpoint_of(run_dir: Path):
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    block = (summary.get("provenance") or {}).get("s1_checkpoint") or {}
    return block.get("sha256")


def attach(chains, codes, key):
    table = codes.T.tolist()
    offset = 0
    for chain in chains:
        if offset + len(chain["scores"]) > len(table):
            raise SystemExit(f"the dump has more probes than the {len(table)} code rows cover")
        for i, score in enumerate(chain["scores"]):
            if key in score:
                raise SystemExit(f"{chain['qkey']}: {key} is already present in this dump")
            score[key] = table[offset + i]
        offset += len(chain["probes"])
    if offset != len(table):
        raise SystemExit(f"{offset} probes against {len(table)} code rows")
    return offset


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="lane B weights.json (QVRS, DEPTH)")
    p.add_argument("--weights-h", required=True, help="weights_h.json (H)")
    p.add_argument("--reader-run", required=True, help="this pool's reader run dir")
    p.add_argument("--reprs", required=True, help="this pool's reprs-select.npy (for H)")
    p.add_argument("--val-from", required=True,
                   help="the fitted lane dir whose val_scored.jsonl is copied verbatim")
    p.add_argument("--fitted-reader-run", required=True,
                   help="the run dir that scored the pool the weights were fitted over -- the "
                        "RESCORE run, not the lane dir, which carries no checkpoint digest")
    p.add_argument("--out", required=True)
    p.add_argument("--expect-chains", type=int, default=0,
                   help="refuse unless the pool has exactly this many chains")
    a = p.parse_args(argv)

    reader, val_from, out = Path(a.reader_run), Path(a.val_from), Path(a.out)
    fitted_run = Path(a.fitted_reader_run)
    fitted_ckpt, pool_ckpt = checkpoint_of(fitted_run), checkpoint_of(reader)
    if not fitted_ckpt or not pool_ckpt:
        raise SystemExit(
            "a run dir declares no provenance.s1_checkpoint.sha256 "
            f"({fitted_run}={fitted_ckpt!r}, {reader}={pool_ckpt!r}), so the two readers "
            "cannot be shown to be the same and the rank codes could come from another "
            "model's beliefs")
    if fitted_ckpt != pool_ckpt:
        raise SystemExit(
            f"the weights were fitted over reader {fitted_ckpt[:12]} but this pool was scored "
            f"by {pool_ckpt[:12]}. Deploying one head's weights on another head's beliefs is "
            "not a read-out of anything.")

    chains = read_chains(reader / "select_scored.jsonl")
    if a.expect_chains and len(chains) != a.expect_chains:
        raise SystemExit(f"{reader} has {len(chains)} chains, not the expected {a.expect_chains}")

    weights = json.loads(Path(a.weights).read_text(encoding="utf-8"))
    weights_h = json.loads(Path(a.weights_h).read_text(encoding="utf-8"))
    missing = [s for s in FROM_WEIGHTS if s not in (weights.get("models") or {})]
    if missing:
        raise SystemExit(f"{a.weights} carries models for "
                         f"{sorted(weights.get('models') or {})} and not {missing}")

    codes = lp.deploy(weights, chains)
    written = {}
    for name in FROM_WEIGHTS:
        written[lb.score_key(name)] = attach(chains, codes[name], lb.score_key(name))

    reprs = Path(a.reprs)
    index = reprs.with_name(reprs.name.replace(".npy", ".index.jsonl"))
    features = lh.features_from(lh.load_reprs(reprs, index))
    codes_h = lh.deploy_h(weights_h, chains, features)
    written[lb.score_key(lh.FEATURE_SET)] = attach(
        chains, codes_h[lh.FEATURE_SET], lb.score_key(lh.FEATURE_SET))

    out.mkdir(parents=True, exist_ok=True)
    (out / "val_scored.jsonl").write_bytes((val_from / "val_scored.jsonl").read_bytes())
    with open(out / "select_scored.jsonl", "w", encoding="utf-8") as fh:
        for chain in chains:
            fh.write(json.dumps(chain, separators=(",", ":")) + "\n")
    summary = {
        "schema": "olmo3-lb-readout/v1",
        "config": {"pool": (json.loads((reader / "summary.json").read_text())
                            .get("config") or {}).get("pool"),
                   "seed": 0, "method": "lb_readout"},
        "provenance": {
            "note": "the already-fitted lane B weights applied to a pool the fit never saw; "
                    "no label is read here and nothing is refitted",
            "reader_run": {"path": str(reader.resolve()), "s1_checkpoint_sha256": pool_ckpt},
            "fitted_reader_run": {"path": str(fitted_run.resolve()),
                                  "s1_checkpoint_sha256": fitted_ckpt},
            "weights": {"path": str(Path(a.weights).resolve()), "sha256": sha256(a.weights)},
            "weights_h": {"path": str(Path(a.weights_h).resolve()),
                          "sha256": sha256(a.weights_h)},
            "reprs": {"path": str(reprs.resolve()), "sha256": sha256(reprs)},
            "val_copied_from": {"path": str((val_from / "val_scored.jsonl").resolve()),
                                "sha256": sha256(val_from / "val_scored.jsonl")},
            "score_keys": sorted(written),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                      encoding="utf-8")
    print(json.dumps({"out": str(out), "pool_chains": len(chains),
                      "val_chains": sum(1 for _ in open(out / "val_scored.jsonl")),
                      "probes_coded_per_key": written,
                      "select_sha256": sha256(out / "select_scored.jsonl")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
