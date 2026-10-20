"""Score a pool with a trained ROM head."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from rl_controller.baselines.rom import c2_cell
from rl_controller.baselines.rom import data as rom_data
from rl_controller.baselines.rom import export as rom_export
from rl_controller.baselines.rom import train as rom_train
from rl_controller.baselines.rom.class_balance import file_digest
from rl_controller.baselines.rom.features import FeatureSpec, open_cache
from rl_controller.baselines.rom.streaming_head import StreamingHead


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--head", required=True, help="head.pt written beside a run dir")
    p.add_argument("--chains", required=True, help="the pool to read")
    p.add_argument("--features", required=True, help="that pool's cache manifest")
    p.add_argument("--val-chains", required=True)
    p.add_argument("--val-features", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-count", type=int, default=rom_data.VAL_COUNT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--answer-format", default="boxed", choices=("boxed", "code"))
    return p.parse_args(argv)


def score_incrementally(scorer, head, chains, cache, spec, *, config, partial_path,
                        chunk_size: int = 2000):
    partial_path = Path(partial_path)
    done = {}
    if partial_path.exists():
        with open(partial_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[(row["qkey"], int(row["rep"]))] = row["scores"]
    if done:
        print(json.dumps({"resumed_scored_chains": len(done),
                          "partial": str(partial_path)}), flush=True)

    todo, report = [], {"unscored": [], "probeless_natural_rows": 0}
    for chain in chains:
        ident = (chain["qkey"], int(chain["rep"]))
        if ident in done:
            chain["scores"] = done[ident]
            continue
        if not chain["probes"]:
            chain["scores"] = []
            report["probeless_natural_rows"] += 1
            continue
        todo.append(chain)

    partial_path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(todo), chunk_size):
        batch = todo[start:start + chunk_size]
        sub = scorer(head, batch, cache, spec, config=config)
        report["unscored"].extend(sub.get("unscored") or [])
        with open(partial_path, "a", encoding="utf-8") as handle:
            for chain in batch:
                if chain.get("scores") or not chain["probes"]:
                    handle.write(json.dumps({"qkey": chain["qkey"], "rep": chain["rep"],
                                             "scores": chain["scores"]}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({"scored": min(start + chunk_size, len(todo)),
                          "of": len(todo), "partial": partial_path.name}), flush=True)
    return report


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


def main(argv=None) -> int:
    import torch

    from rl_controller.iclr import scoring_io as sio

    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    if (out_dir / "summary.json").exists():
        raise SystemExit(f"{out_dir}/summary.json exists; refusing to overwrite")

    checkpoint = torch.load(args.head, map_location="cpu", weights_only=False)
    kind = checkpoint.get("kind", "rom_full")
    spec = FeatureSpec(layer=int(checkpoint["spec"]["layer"]),
                       pooling=str(checkpoint["spec"]["pooling"]))
    saved = dict(checkpoint["config"])
    saved["device"] = args.device
    saved.pop("betas", None)
    config = rom_train.TrainConfig(**{k: v for k, v in saved.items()
                                      if k in rom_train.TrainConfig.__dataclass_fields__
                                      and k != "betas"})
    head_config = checkpoint["head_config"]
    head = StreamingHead(input_dim=head_config["input_dim"], proj_dim=head_config["proj_dim"],
                         mem_dim=head_config["mem_dim"], num_labels=head_config["num_labels"],
                         cfc=head_config.get("cfc", True),
                         attend_prompt=head_config.get("attend_prompt", True),
                         head_mode=head_config.get("head_mode"))
    head.load_state_dict(checkpoint["model_state_dict"])
    head.to(device=args.device, dtype=getattr(torch, config.dtype)).eval()

    pool_cache = open_cache(args.features)
    val_cache = open_cache(args.val_features)
    if pool_cache.manifest.hidden_size != head_config["input_dim"]:
        raise SystemExit(f"cache hidden_size {pool_cache.manifest.hidden_size} != head input "
                         f"{head_config['input_dim']}")

    fit = sio.load_chains(args.val_chains)
    val_keys, _ = rom_data.val_train_split(fit, val_count=args.val_count)
    val_chains = [c for c in fit if c["qkey"] in val_keys]
    pool_chains = sio.load_chains(args.chains)

    scorer = c2_cell.score_c2 if kind == "rom_cfc_c2" else rom_export.score_chains
    out_dir.mkdir(parents=True, exist_ok=True)
    val_skip = score_incrementally(scorer, head, val_chains, val_cache, spec, config=config,
                                   partial_path=out_dir / "partial_val.jsonl")
    pool_skip = score_incrementally(scorer, head, pool_chains, pool_cache, spec, config=config,
                                    partial_path=out_dir / "partial_select.jsonl")
    if val_skip["unscored"] or pool_skip["unscored"]:
        raise SystemExit(json.dumps({"error": "cache does not cover a split",
                                     "val": len(val_skip["unscored"]),
                                     "pool": len(pool_skip["unscored"])}))

    targets = list(c2_cell.TARGETS) if kind == "rom_cfc_c2" else ["q", "v"]
    provenance = dict(checkpoint.get("provenance") or {})
    provenance.update({
        "chains_sha256": file_digest(args.chains), "commit": _commit(),
        "answer_format": args.answer_format, "overhead_accounting": "none",
        "overhead_tokens_per_boundary": 0,
        "rom_scored_from_frozen_head": str(args.head),
        "rom_readout_only": True,
        **pool_cache.manifest.provenance()})
    summary = sio.summary_stub(seed=int(config.seed), targets=targets,
                               method=("rom_cfc_c2" if kind == "rom_cfc_c2" else "rom"),
                               provenance=provenance,
                               knob=("theta" if kind == "rom_cfc_c2" else "tau"),
                               layer=spec.layer)
    written = sio.write_run_dir(out_dir, val=val_chains, select=pool_chains, summary=summary)
    print(json.dumps({**written, "kind": kind, "layer": spec.layer,
                      "probeless": pool_skip.get("probeless_natural_rows", 0)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
