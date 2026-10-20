"""Emit fitted linear policies as policy-replay records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_controller.iclr import families_learned_linear as lb
from rl_controller.iclr import osp_policy as osp
from scripts import olmo3_free_rows as fr

FEATURE_SETS = ("QVRS", "H", "DEPTH")


def members_for(feature_set):
    key = lb.score_key(feature_set)
    out = []
    for l in range(lb.N_LAMBDAS):
        for k in range(lb.N_LEVELS):
            plan = lb.make_plan(key, l, k)

            def rule(chain, _plan=plan):
                probe = _plan(chain)
                if probe is None:
                    return None
                for i, candidate in enumerate(chain.get("probes") or []):
                    if candidate is probe:
                        return i
                raise SystemExit("a plan returned a probe that is not in the chain")

            out.append(("l%02d,k%02d" % (l, k),
                        {"name": "lambda_index_level", "value": l * lb.N_LEVELS + k}, rule))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference-export", required=True)
    p.add_argument("--lane-dir", required=True,
                   help="the DERIVED lane B run dir: its scores carry lb_<SET>")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", default="qvrs-lbdev", help="the head, as a family tag")
    p.add_argument("--sets", nargs="*", default=list(FEATURE_SETS))
    p.add_argument("--seed", type=int, default=None,
                   help="override the replay header's `seed`. NOT the way to carry a reader "
                        "seed into the appendix -- see --reader-seed. literature_metrics reads "
                        "one seed per call and refuses a family with no members at it, and the "
                        "OS-Pruner reference exists at seed 0 only, so a family declared at "
                        "seed 1 cannot be priced against it at all")
    p.add_argument("--reader-seed", type=int, default=None,
                   help="the reader seed these rows were fitted at, written into "
                        "frozen_config.reader_seed. Use this WITH a distinct --tag "
                        "(qvrs-lbdev-s1): the tag makes the three seeds three FAMILIES at the "
                        "reference's own seed, which is what lets each be priced against it, "
                        "and this field states the number so that reading does not rest on a "
                        "tag's spelling. Without it a seed-1 arm would be indistinguishable "
                        "from a seed-0 one in the artifact itself")
    a = p.parse_args(argv)

    ref = json.loads(Path(a.reference_export).read_text(encoding="utf-8"))
    if ref.get("schema_version") != osp.REPLAY_SCHEMA:
        raise SystemExit(f"--reference-export declares {ref.get('schema_version')!r}, "
                         f"not {osp.REPLAY_SCHEMA!r}")
    if a.seed is not None:
        ref = dict(ref, seed=int(a.seed))

    lane = Path(a.lane_dir)
    pool_path, val_path = lane / "select_scored.jsonl", lane / "val_scored.jsonl"
    dumps = {fr.READ_SPLIT: (fr.load(pool_path), pool_path.resolve(), fr.sha256(pool_path)),
             fr.VAL_SPLIT: (fr.load(val_path), val_path.resolve(), fr.sha256(val_path))}

    sample = dumps[fr.READ_SPLIT][0][0]
    have = set((sample.get("scores") or [{}])[0])
    missing = [s for s in a.sets if lb.score_key(s) not in have]
    if missing:
        raise SystemExit(f"{pool_path} carries {sorted(have)} and none of "
                         f"{[lb.score_key(s) for s in missing]}: this is not a lane B DERIVED "
                         "run dir, so no LB member could fire")

    written = []
    for feature_set in a.sets:
        family = f"LB_{feature_set}@{a.tag}"
        written.append(fr.emit(ref, members_for(feature_set), dumps,
                               family=family, out_dir=Path(a.out_dir),
                               frozen_extra=({"reader_seed": int(a.reader_seed)}
                                             if a.reader_seed is not None else None)))
    print(json.dumps({"lane_dir": str(lane), "seed": int(ref["seed"]),
                      "reader_seed": a.reader_seed, "tag": a.tag,
                      "val_chains": len(dumps[fr.VAL_SPLIT][0]),
                      "pool_chains": len(dumps[fr.READ_SPLIT][0]),
                      "members_per_family": lb.N_LAMBDAS * lb.N_LEVELS,
                      "written": written}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
