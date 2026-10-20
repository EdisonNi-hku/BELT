"""Compute training-free reference rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_controller.baselines.ospruner import replay as R
from rl_controller.iclr import osp_policy as osp
from rl_controller.iclr import pareto as pt
from rl_controller.scripts import iclr_direct_export as dx

CAPS = (512, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
READ_SPLIT, VAL_SPLIT = "select", "val"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path) -> list:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def fire(chain, index):
    scores = [dict(s, pi=float(i == index)) for i, s in enumerate(chain.get("scores") or [])]
    return dict(chain, scores=scores)


def cap_rule(chain, n: int):
    for i, probe in enumerate(chain.get("probes") or []):
        if int(probe["prefix_len"]) >= n:
            return i
    return None


def oracle_rule(chain):
    gold = chain.get("gold")
    probes = chain.get("probes") or []
    for i, probe in enumerate(probes):
        if osp.is_correct(probe.get("forced"), gold):
            return i
    return 0 if probes else None


def header_for(ref, *, family, config_id, knob, split, artifact, sha, frozen_extra=None):
    inherited = {k: ref[k] for k in ("dataset", "model_id", "model_revision",
                                     "tokenizer_revision", "prompt_contract_id", "grader_id",
                                     "selection_split", "selection_rule",
                                     "chain_bank_sha256", "manifest_sha256", "pool")}
    if split == VAL_SPLIT:
        block = ((ref.get("split_chain_identity") or {}).get(VAL_SPLIT) or {})
        inherited["chain_bank_sha256"] = block.get("chain_bank_sha256",
                                                   inherited["chain_bank_sha256"])
        inherited["manifest_sha256"] = block.get("manifest_sha256", inherited["manifest_sha256"])
    return osp.replay_header(
        run_id=f"{family}-{config_id}", method_family="free", method_name=family,
        config_id=config_id, knob=knob, seed=int(ref["seed"]), split=split,
        frozen_config={"rule": family, "member": config_id,
                       "expectation": "survival_product",
                       "deterministic": True, "q_is": "1 at the stopping probe, 0 elsewhere",
                       "overhead_accounting": "none", **(frozen_extra or {})},
        source_branch=ref["source_branch"], source_commit=ref["source_commit"],
        source_artifact=str(artifact), source_sha256=sha, **inherited)


def emit(ref, members, dumps, *, family, out_dir: Path, frozen_extra=None) -> dict:
    natural_fn = dx.natural_answer_fn(ref["answer_format"], bool(ref["force_natural_answer"]))
    rows = []
    for config_id, knob, rule in members:
        for split, (chains, artifact, sha) in dumps.items():
            fired = []
            for chain in chains:
                index = rule(chain)
                fired.append(fire(chain, -1 if index is None else index))
            head = header_for(ref, family=family, config_id=config_id, knob=knob,
                              split=split, artifact=artifact, sha=sha,
                              frozen_extra=frozen_extra)
            rows.extend(R.replay_rows(fired, header=head, pi_key="pi", grade=ref["grading"],
                                      natural_answer_tokens=natural_fn,
                                      aux_generated_tokens=ref["aux_generated_tokens"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{family}_{ref['pool']}.policy_replay.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in sorted(rows, key=lambda r: (r["config_id"], r["seed"], r["split"], r["qkey"])):
            fh.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    try:
        pt.load_policy_replay(str(out), normalize="chain", seed_semantics="training")
    except pt.MergeError as exc:
        raise SystemExit(f"{out}: the merger's loader refuses this artifact: {exc}") from None
    return {"family": family, "out": str(out), "n_rows": len(rows),
            "n_members": len(members), "sha256": sha256(out)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference-export", required=True,
                   help="the REGISTERED export's sidecar; every comparability field is taken "
                        "from it rather than declared here")
    p.add_argument("--pool-dump", required=True)
    p.add_argument("--val-dump", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", default="dev160", help="the selection population, as a family tag")
    args = p.parse_args(argv)

    ref = json.loads(Path(args.reference_export).read_text(encoding="utf-8"))
    if ref.get("schema_version") != osp.REPLAY_SCHEMA:
        raise SystemExit(f"--reference-export declares {ref.get('schema_version')!r}")
    dumps = {READ_SPLIT: (load(args.pool_dump), Path(args.pool_dump).resolve(),
                          sha256(args.pool_dump)),
             VAL_SPLIT: (load(args.val_dump), Path(args.val_dump).resolve(),
                         sha256(args.val_dump))}
    out_dir = Path(args.out_dir)
    written = [
        emit(ref, [(f"cap{n}", {"name": "cap_tokens", "value": n}, (lambda c, _n=n: cap_rule(c, _n)))
                   for n in CAPS], dumps, family=f"cap@{args.tag}", out_dir=out_dir),
        emit(ref, [("oracle", {"name": "oracle", "value": 0}, oracle_rule)], dumps,
             family=f"oracle@{args.tag}", out_dir=out_dir),
    ]
    print(json.dumps({"written": written, "caps": list(CAPS),
                      "n_pool_chains": len(dumps[READ_SPLIT][0]),
                      "n_val_chains": len(dumps[VAL_SPLIT][0])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
