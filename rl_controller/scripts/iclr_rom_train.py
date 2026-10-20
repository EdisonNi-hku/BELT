"""Train a ROM detector on cached block features and score pools."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from rl_controller.baselines.rom import data as rom_data
from rl_controller.baselines.rom import export
from rl_controller.baselines.rom import train as rom_train
from rl_controller.baselines.rom.class_balance import file_digest
from rl_controller.baselines.rom.features import FeatureSpec, open_cache


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True, help="WS-E cache manifest for the FIT pool")
    parser.add_argument("--fit-chains", required=True)
    parser.add_argument("--score-chains", required=True, help="the pool to read (select/OOD)")
    parser.add_argument("--score-features", required=True, help="cache manifest for that pool")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layer", type=int, default=24,
                        help="24 = ROM's ~85%%-depth prescription for Qwen3-1.7B; 16 = our UHead's")
    parser.add_argument("--pooling", default="last", choices=("last", "mean"))
    parser.add_argument("--arm", default="R0", choices=rom_train.ARMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drop-never-correct", action="store_true",
                        help="the pre-registered FCS-undefined ABLATION (prereg §1)")
    parser.add_argument("--dt-mode", default="token_gap", choices=rom_data.DT_MODES,
                        help="fixed at token_gap by prereg amendment A1; the others are "
                             "val-only descriptive ablations")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--mem-dim", type=int, default=512)
    parser.add_argument("--num-labels", type=int, default=1, choices=(1, 2))
    parser.add_argument("--no-cfc", action="store_true", help="upstream's recurrence ablation")
    parser.add_argument("--m0-grad", action="store_true",
                        help="train the query-aware m_0 projection (upstream does NOT)")
    parser.add_argument("--val-count", type=int, default=rom_data.VAL_COUNT)
    parser.add_argument("--val-reference-table", default=None, metavar="JSON",
                        help="OBSERVE (never select on) a second validation population: the "
                             "exact (qkey, rep) chains of this authenticated question table. "
                             "Every epoch logs `val_loss_ref` beside `val_loss`, and the "
                             "summary reports both argmins. This measures the 2026-09-03 "
                             "exposure: this trainer picks its checkpoint on the WHOLE val "
                             "split while the campaign's reference method was early-stopped on "
                             "the subset the ladder projects onto, and no read-time projection "
                             "can undo which checkpoint was chosen. Selection is unchanged, so "
                             "a probe run is comparable to the lanes it is probing")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore an existing resume checkpoint and start over")
    parser.add_argument("--manifest-sha256", default="", help="question-manifest sha for provenance")
    return parser.parse_args(argv)


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


def main(argv=None) -> int:
    args = parse_args(argv)
    from rl_controller.iclr import scoring_io as sio

    out_dir = Path(args.out_dir)
    if (out_dir / "summary.json").exists():
        raise SystemExit(f"{out_dir}/summary.json exists; refusing to overwrite a scored run")

    spec = FeatureSpec(layer=args.layer, pooling=args.pooling)
    fit_cache = open_cache(args.features)
    score_cache = open_cache(args.score_features)
    if fit_cache.manifest.hidden_size != score_cache.manifest.hidden_size:
        raise SystemExit("fit and score caches disagree on hidden_size")

    config = rom_train.TrainConfig(
        arm=args.arm, seed=args.seed, dt_mode=args.dt_mode, m0_grad=args.m0_grad,
        proj_dim=args.proj_dim, mem_dim=args.mem_dim, num_labels=args.num_labels,
        cfc=not args.no_cfc, device=args.device,
        **{k: v for k, v in (("epochs", args.epochs), ("lr", args.lr),
                             ("batch_size", args.batch_size)) if v is not None})

    fit = sio.load_chains(args.fit_chains)
    val_keys, _train_keys = rom_data.val_train_split(fit, val_count=args.val_count)
    train_chains = [c for c in fit if c["qkey"] not in val_keys]
    val_chains = [c for c in fit if c["qkey"] in val_keys]

    build = dict(dt_mode=config.dt_mode, drop_never_correct=args.drop_never_correct)
    train_ex, train_skip = rom_data.build_examples(train_chains, fit_cache, spec, **build)
    val_ex, val_skip = rom_data.build_examples(val_chains, fit_cache, spec, **build)
    print(json.dumps({"train_examples": len(train_ex), "val_examples": len(val_ex),
                      "train_skipped": {k: (len(v) if isinstance(v, list) else v)
                                        for k, v in train_skip.items()},
                      "val_skipped": {k: (len(v) if isinstance(v, list) else v)
                                      for k, v in val_skip.items()}}))

    val_extra = None
    reference_probe = None
    if args.val_reference_table:
        from rl_controller.iclr import validation_basis as vb

        basis = vb.reference_validation_basis(args.val_reference_table)
        wanted = set(basis["chains"])
        ref_chains = [c for c in val_chains
                      if (str(c["qkey"]), int(c["rep"])) in wanted]
        if not ref_chains:
            raise SystemExit(
                f"--val-reference-table {args.val_reference_table}: none of its "
                f"{len(wanted)} chains is in this run's {len(val_chains)}-chain val split, "
                "so there is no second population to observe")
        ref_ex, ref_skip = rom_data.build_examples(ref_chains, fit_cache, spec, **build)
        reference_probe = {
            "table": basis["record"]["reference_table"],
            "table_sha256": basis["record"]["table_sha256"],
            "n_reference_chains": len(wanted), "n_val_chains": len(val_chains),
            "n_matched_chains": len(ref_chains), "n_examples": len(ref_ex),
            "skipped": {k: (len(v) if isinstance(v, list) else v)
                        for k, v in ref_skip.items()}}
        print(json.dumps({"val_reference_probe": reference_probe}))
        val_extra = {"ref": ref_ex}

    out_dir.mkdir(parents=True, exist_ok=True)
    result = rom_train.train_detector(
        train_ex, val_ex, config, input_dim=fit_cache.manifest.hidden_size,
        resume_path=out_dir / "resume.pt", resume=not args.no_resume, val_extra=val_extra)
    if reference_probe is not None:
        losses = [(row["val_loss_ref"], row["epoch"]) for row in result.history
                  if "val_loss_ref" in row]
        reference_probe["best_epoch_ref"] = min(losses)[1] if losses else None
        reference_probe["best_val_loss_ref"] = min(losses)[0] if losses else None
        reference_probe["best_epoch_full"] = result.best_epoch
        reference_probe["best_val_loss_full"] = result.best_val_loss
        reference_probe["agrees"] = (reference_probe["best_epoch_ref"] == result.best_epoch)
        print(json.dumps({"val_reference_probe_result": reference_probe}))

    select_chains = sio.load_chains(args.score_chains)
    val_unscored = export.score_chains(result.head, val_chains, fit_cache, spec, config=config,
                                       drop_never_correct=args.drop_never_correct)
    sel_unscored = export.score_chains(result.head, select_chains, score_cache, spec,
                                       config=config,
                                       drop_never_correct=args.drop_never_correct)
    if val_unscored["unscored"] or sel_unscored["unscored"]:
        raise SystemExit(json.dumps({
            "error": "feature cache does not cover every chain of a split",
            "val_unscored": len(val_unscored["unscored"]),
            "select_unscored": len(sel_unscored["unscored"]),
            "examples": (val_unscored["unscored"] + sel_unscored["unscored"])[:5]}))

    m0_init = "query_aware" if fit_cache.manifest.has_prompt_features else "zeros"
    provenance = export.provenance(
        cache_manifest=score_cache.manifest,
        chains_sha256=file_digest(args.score_chains),
        manifest_sha256=args.manifest_sha256, commit=_commit(), config=config, spec=spec,
        balance=result.balance, m0_init=m0_init,
        extra={"rom_params": result.head.parameter_count(),
               "rom_fit_chains_sha256": file_digest(args.fit_chains),
               "rom_drop_never_correct": bool(args.drop_never_correct),
               "rom_best_epoch": result.best_epoch,
               "rom_best_val_loss": result.best_val_loss,
               "rom_val_count": args.val_count,
               **({"rom_val_reference_probe": reference_probe}
                  if reference_probe is not None else {})})

    written = export.write_run_dir(out_dir, val_chains=val_chains, select_chains=select_chains,
                                   seed=args.seed, provenance_row=provenance,
                                   extra_config={"arm": args.arm, "layer": args.layer,
                                                 "pooling": args.pooling})
    import torch

    torch.save({"model_state_dict": result.head.state_dict(),
                "head_config": result.head.config(), "train": result.summary(),
                "spec": spec.to_dict(), "config": config.to_dict(),
                "kind": "rom_full", "provenance": provenance},
               out_dir / "head.pt")
    (out_dir / "train_summary.json").write_text(
        json.dumps(result.summary(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    export.write_family_spec(out_dir / "rom_family.json")
    print(json.dumps({**written, "params": result.head.parameter_count(),
                      "best_epoch": result.best_epoch, "balance": result.balance["applied"],
                      "m0_init": m0_init}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
