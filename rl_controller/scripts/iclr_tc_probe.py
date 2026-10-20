"""Fit Thought Calibration consistency probes and write run directories."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rl_controller.baselines.tc import dataset as ds
from rl_controller.baselines.tc import labels as tcl
from rl_controller.baselines.tc import lean
from rl_controller.baselines.tc import probe as pr
from rl_controller.iclr import scoring_io as sio

DEFAULT_V = 0.5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit-features", required=True, help="pool manifest_<positions>.json to FIT on")
    p.add_argument("--fit-chains", required=True, help="the fit pool's chains.jsonl")
    p.add_argument("--select-features", required=True, help="pool manifest to SCORE")
    p.add_argument("--select-chains", required=True, help="the scored pool's chains.jsonl")
    p.add_argument("--val-scored", required=True,
                   help="an archived head's val_scored.jsonl -- the SHARED val question set")
    p.add_argument("--out-root", required=True)
    p.add_argument("--target", default="stability",
                   help="'stability' = TC's consistency (C1 primary); 'q' = their Supervised "
                        "probe, which is a C2 cell ONLY (protocol 9.4)")
    p.add_argument("--layer", default="L16")
    p.add_argument("--variant", default="mean", choices=sorted(pr.VARIANT_INDEX))
    p.add_argument("--pca-k", type=int, default=256)
    p.add_argument("--pca-sample", type=int, default=400_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--windows", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--class-weight", action="store_true",
                   help="pre-registered inverse-frequency arm; A1.1 resolved NOT to run it")
    p.add_argument("--v-const", type=float, default=0.5)
    p.add_argument("--max-train-rows", type=int, default=0, help="smoke cap")
    p.add_argument("--no-resume", action="store_true",
                   help="refit and rescore even if a cached probe/score file exists "
                        "(operating rule 5 persistence is ON by default)")
    p.add_argument("--extra-pool", action="append", default=[],
                   help="NAME:FEATURES_MANIFEST:CHAINS -- score another pool with the SAME "
                        "fitted probe, into <out>/<tag>__<NAME>/. OOD is a readout, so it must "
                        "be the identical fit, never a refit")
    p.add_argument("--commit", default="")
    return p.parse_args(argv)


def chain_labels(chains, target):
    out = {}
    for chain in chains:
        if not chain["_trainable"]:
            continue
        out[(chain["qkey"], int(chain["rep"]))] = (
            tcl.consistency_labels(chain) if target == "stability" else sio.q_labels(chain))
    return out


def training_rows(index, labels, qkeys, *, cap=0):
    addresses, y = [], []
    for i in range(index.n_rows):
        pi = index.probe_index[i]
        if pi < 0:
            continue
        key = index.chains[index.chain_of[i]]
        if key[0] not in qkeys:
            continue
        table = labels.get(key)
        if table is None or pi >= len(table):
            continue
        value = table[pi]
        if value is None:
            continue
        addresses.append((index.shard[i], index.row[i]))
        y.append(float(value))
        if cap and len(y) >= cap:
            break
    return addresses, y


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    fit_manifest, fit_index = lean.load_lean_index(args.fit_features)
    sel_manifest, sel_index = lean.load_lean_index(args.select_features)
    from rl_controller.iclr.provenance import source_record
    for manifest, path, name in ((fit_manifest, args.fit_chains, "fit"),
                                 (sel_manifest, args.select_chains, "select")):
        got = source_record(path)["sha256"]
        if manifest["chains_sha256"] != got:
            raise SystemExit(f"{name}: cache was built from chains sha256 "
                             f"{manifest['chains_sha256']} but --{name}-chains is {got}")

    fit_chains = lean.stream_chains_lite(args.fit_chains)
    sel_chains = lean.stream_chains_lite(args.select_chains)
    labels = chain_labels(fit_chains, args.target)
    stats = tcl.label_stats(fit_chains)
    conventions = tcl.convention_disagreement(fit_chains) if args.target == "stability" else None
    print(json.dumps({"label_stats": stats}, indent=2), flush=True)

    val_qkeys = ds.val_qkeys_from_scored(args.val_scored)
    present = {k[0] for k in fit_index.chains}
    val_keys = {q for q in val_qkeys if q in present}
    train_keys = present - val_keys
    if not val_keys:
        raise SystemExit("no val question present in the fit cache")

    fit_reader = pr.FeatureReader(fit_manifest, args.layer, args.variant)
    sel_reader = pr.FeatureReader(sel_manifest, args.layer, args.variant)
    train_addr, train_y = training_rows(fit_index, labels, train_keys, cap=args.max_train_rows)
    if not train_addr:
        raise SystemExit("no labelled training rows")
    print(json.dumps({"train_rows": len(train_addr),
                      "base_rate": round(sum(train_y) / len(train_y), 4),
                      "n_train_questions": len(train_keys), "n_val_questions": len(val_keys),
                      "seconds": round(time.time() - t0, 1)}), flush=True)

    Xtr = pr.materialize(fit_reader, train_addr)
    print(json.dumps({"materialised": list(Xtr.shape), "gb": round(Xtr.nbytes / 2**30, 2),
                      "seconds": round(time.time() - t0, 1)}), flush=True)
    pca_path = out_root / f"pca_tc_{args.target}_{args.layer}_{args.variant}_k{args.pca_k}.npz"
    cached_pca = None if args.no_resume else pr.load_pca_safe(pca_path)
    if cached_pca is not None:
        pca = cached_pca
        print(json.dumps({"resumed": "pca", "path": str(pca_path)}), flush=True)
    else:
        sample = Xtr[::max(1, Xtr.shape[0] // max(args.pca_sample, 1))]
        pca = pr.fit_pca_matrix(sample, k=args.pca_k)
    pca_prov = pr.pca_record(k=args.pca_k, hidden=pca["hidden"], n_rows=pca["n_rows"],
                             layer_key=args.layer, variant=args.variant,
                             explained=pca["explained_variance_ratio"], path=pca_path,
                             sha=(pr.save_pca(pca, pca_path) if cached_pca is None
                                  else pr.sha256_bytes(pca_path.read_bytes())),
                             train_questions=len(train_keys))
    print(json.dumps({"pca": pca_prov}, indent=2), flush=True)

    Ztr = pr.project(pca, Xtr)
    del Xtr
    print(json.dumps({"projected": list(Ztr.shape), "gb": round(Ztr.nbytes / 2**30, 2),
                      "seconds": round(time.time() - t0, 1)}), flush=True)

    val_by_chain = {k: v for k, v in lean.probe_rows_by_chain(fit_index).items()
                    if k[0] in val_keys}
    sel_by_chain = lean.probe_rows_by_chain(sel_index)
    val_chains = [c for c in fit_chains if c["qkey"] in val_keys]

    def raw_scores(reader, by_chain, probe):
        flat = [(key, pi, sh, rw) for key, rows in by_chain.items() for pi, sh, rw in rows]
        addresses = [(sh, rw) for _, _, sh, rw in flat]
        out = pr.predict_matrix(probe, pr.project(pca, pr.materialize(reader, addresses)))
        table = {}
        for j, (key, pi, _, _) in enumerate(flat):
            table.setdefault(key, {})[pi] = float(out[j])
        return table

    def attach(chains, table, window):
        for chain in chains:
            key = (chain["qkey"], int(chain["rep"]))
            rows = table.get(key)
            if rows is None:
                if chain["probes"]:
                    raise SystemExit(f"chain {key} has probes but no cached features")
                chain["scores"] = []
                continue
            if len(rows) != len(chain["probes"]):
                raise SystemExit(f"chain {key}: {len(rows)} cached probe rows for "
                                 f"{len(chain['probes'])} probes")
            smoothed = tcl.trailing_mean([rows[i] for i in range(len(rows))], window)
            chain["scores"] = [{"q": float(q), "v": float(args.v_const)} for q in smoothed]

    written = []
    for seed in args.seeds:
        probe_path = out_root / f"probe_{args.target}_s{seed}.npz"
        probe = None if args.no_resume else pr.load_probe(probe_path)
        if probe is not None:
            print(json.dumps({"resumed": "probe", "seed": seed}), flush=True)
        else:
            probe = pr.fit_logistic_matrix(Ztr, train_y, seed=seed, epochs=args.epochs,
                                           batch=args.batch, lr=args.lr, l2=args.l2,
                                           class_weight=args.class_weight)
            pr.save_probe(probe, probe_path)
        prov_probe = pr.probe_record(probe, target=args.target, layer_key=args.layer,
                                     variant=args.variant)
        print(json.dumps({"probe": prov_probe, "seconds": round(time.time() - t0, 1)}), flush=True)
        val_path = out_root / f"scores_val_{args.target}_s{seed}.json"
        sel_path = out_root / f"scores_select_{args.target}_s{seed}.json"
        val_table = None if args.no_resume else pr.load_scores(val_path)
        if val_table is None:
            val_table = raw_scores(fit_reader, val_by_chain, probe)
            pr.save_scores(val_table, val_path)
        else:
            print(json.dumps({"resumed": "val_scores", "seed": seed}), flush=True)
        sel_table = None if args.no_resume else pr.load_scores(sel_path)
        if sel_table is None:
            sel_table = raw_scores(sel_reader, sel_by_chain, probe)
            pr.save_scores(sel_table, sel_path)
        else:
            print(json.dumps({"resumed": "select_scores", "seed": seed}), flush=True)
        ood = {}
        for spec in args.extra_pool:
            name, feat, chains_path = spec.split(":", 2)
            omanifest, oindex = lean.load_lean_index(feat)
            ochains = lean.stream_chains_lite(chains_path)
            oreader = pr.FeatureReader(omanifest, args.layer, args.variant)
            opath = out_root / f"scores_{name}_{args.target}_s{seed}.json"
            otable = None if args.no_resume else pr.load_scores(opath)
            if otable is None:
                otable = raw_scores(oreader, lean.probe_rows_by_chain(oindex), probe)
                pr.save_scores(otable, opath)
            else:
                print(json.dumps({"resumed": f"{name}_scores", "seed": seed}), flush=True)
            ood[name] = (ochains, otable, omanifest)
        for window in args.windows:
            attach(val_chains, val_table, window)
            attach(sel_chains, sel_table, window)
            summary = sio.summary_stub(
                seed=int(seed), targets=["q", "v"], method="tc", knob="lambda",
                smoothing_window=int(window), tc_target=args.target,
                layer=args.layer, variant=args.variant, pca_k=args.pca_k,
                provenance={
                    "commit": args.commit,
                    "chains_sha256": sel_manifest["chains_sha256"],
                    "manifest_sha256": sel_manifest["questions"]["sha256"],
                    "fit_chains_sha256": fit_manifest["chains_sha256"],
                    "features": {"fit": str(args.fit_features),
                                 "select": str(args.select_features),
                                 "layer_semantics": fit_manifest["layer_semantics"]},
                    "model_revision": fit_manifest["model"]["revision"],
                    "head_sha256": prov_probe["sha256"],
                    "answer_format": "boxed", "force_natural_answer": False,
                    "overhead_accounting": "none", "overhead_tokens_per_boundary": 0,
                    "pca": pca_prov, "probe": prov_probe,
                    "label_stats": stats, "convention_disagreement": conventions,
                    "deviation": tcl.DEVIATION, "v_is_constant": args.v_const,
                })
            out_dir = out_root / f"tc-{args.target}-w{window}-s{seed}"
            info = sio.write_run_dir(out_dir, val=val_chains, select=sel_chains, summary=summary)
            written.append(info)
            print(json.dumps({"run_dir": info}), flush=True)
            for name, (ochains, otable, omanifest) in ood.items():
                attach(ochains, otable, window)
                osummary = json.loads(json.dumps(summary))
                osummary["config"]["pool"] = name
                osummary["provenance"]["chains_sha256"] = omanifest["chains_sha256"]
                osummary["provenance"]["manifest_sha256"] = omanifest["questions"]["sha256"]
                osummary["provenance"]["readout_only"] = (
                    "OOD pool scored with the ID-fitted probe; honest members are still "
                    "chosen on the shared ID val split, never here")
                oinfo = sio.write_run_dir(out_root / f"tc-{args.target}-w{window}-s{seed}__{name}",
                                          val=val_chains, select=ochains, summary=osummary)
                print(json.dumps({"ood_run_dir": name, **oinfo}), flush=True)
    print(json.dumps({"done": len(written), "out_root": str(out_root),
                      "seconds": round(time.time() - t0, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
