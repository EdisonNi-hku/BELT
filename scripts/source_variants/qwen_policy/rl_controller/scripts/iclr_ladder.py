"""Belief-to-control evaluation over scored-dump runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Sequence
from pathlib import Path

from rl_controller import s1_depth_margin as dm
from rl_controller.iclr import ladder as ld
from rl_controller.iclr import learning_curve as lc
from rl_controller.iclr.chains import load_chains
from rl_controller.iclr.pi0_bias import consecutive_agreement
from rl_controller.iclr.provenance import parse_and_record, source_record
from rl_controller.single_cot import stop_at_cap, stop_earliest_agreement

DEFAULT_TARGETS = ("q", "v")
LEVELS = ld.SELECT_LEVELS
SPEC_RUNG_PREFIXES = ("L4_", "L6", "L7", "B")
DIAGNOSTIC_RUNGS = ("B_plus_m",)
RUNG_ORDER = ("L0", "L1", "L2", "L3raw", "L3raw_step", "L3", "L3_step", "L4_", "L5_modular",
              "L5_direct", "L6", "L6_pi0", "L7", "L8_hyst", "L8_cusum", "B", "B_plus_m")
BOOTSTRAP_SEMANTICS = ("select-question resampling, paired across seeds and rungs, conditional "
                       "on the seeds (seed-to-seed spread is not in the SE)")
_L3_NAME = re.compile(r"^L3:a([+-]\d+\.\d+),b([+-]\d+\.\d+)$")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run dirs, one per training seed")
    p.add_argument("--out", required=True, help="ladder.json")
    p.add_argument("--md", default=None, help="also write the markdown table here")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--only-families", nargs="+", default=None,
                   help="evaluate only named families plus required internal reference L1")
    p.add_argument("--seed", type=int, default=0, help="bootstrap rng seed")
    p.add_argument("--calib-folds", type=int, default=4, help="val-side cross-fit folds for L2/L3")
    p.add_argument("--no-calibration", action="store_true",
                   help="do not fit the depth-aware q/v calibrators, and so register no "
                        "calibrated rung. For a head whose `q` is not a probability -- G's "
                        "`q = d` is a signed difference (D11.4b) -- the fit refuses its own "
                        "inputs, and every rung such a head may register reads raw q and v. "
                        "Off by default; passing it on a probability head silently costs you "
                        "L2/L3 and is not a shortcut.")
    p.add_argument("--export-row3", default=None,
                   help="write the val-selected L3 member (a, b) + calibrator coefficients "
                        "of the lowest seed for iclr_pi0_fire.py (+ a .meta.json sidecar)")
    p.add_argument("--subset-manifest", nargs="+", default=None, metavar="MANIFEST",
                   help="question manifests giving each qkey's domain (rows without `domain` are math); "
                        "with --subset-domain, val AND select keep only that domain (leave-one-domain-out tables)")
    p.add_argument("--subset-domain", default=None, choices=("code", "math"))
    p.add_argument("--select-manifest", nargs="+", default=None, metavar="MANIFEST",
                   help="restrict the SELECT split to the qkeys of these manifests (per-benchmark tables); val untouched")
    p.add_argument("--validation-reference-table", default=None, metavar="JSON",
                   help="project every run's VAL chains onto exactly this question table's "
                        "(qkey, rep) population BEFORE any fitting, calibration or "
                        "operating-point selection. The reference table is authenticated "
                        "first, the projection is MATERIALIZED as a content-addressed val "
                        "dump (so the export names bytes that hold exactly that population), "
                        "and every projected chain's non-score payload must equal the "
                        "reference's. Use it when this method's val lane holds chains the "
                        "reference method's does not: calibration and selection both happen "
                        "on val, so fitting on the extras selects on a different population "
                        "than the method being compared against")
    p.add_argument("--validation-reference-out", default=None, metavar="DIR",
                   help="where the aligned val dumps are written (default: <dir of --out>"
                        "/aligned_val/<first 16 hex of the reference table's sha256>)")
    p.add_argument("--family-spec", nargs="+", default=None, metavar="JSON",
                   help="external-score family specs (ladder.FamilySpec JSON: a list, or "
                        "{'families': [...]}). A ported baseline's sweep is DATA -- name, "
                        "belief key, grid, direction -- not an edit to ladder.families()")
    p.add_argument("--export-question-tables", default=None, metavar="JSON",
                   help="also write the per-question outcome tables (schema "
                        "iclr-ladder-question-tables/v1) so this run can be merged with OTHER "
                        "methods' tables under one bootstrap resample plan "
                        "(rl_controller.iclr.pareto). Without it a ladder JSON carries only "
                        "aggregate rows and the cross-method paired SE is unrecoverable")
    p.add_argument("--question-tables-scope", default="honest", choices=("honest", "all"),
                   help="honest (default): only the val-selected members + anchors + agreement "
                        "-- what the primary honest-hull ABH needs, a few MB. all: every member "
                        "of every rung, for the descriptive sweep-hull ABH (tens of MB)")
    p.add_argument("--head-tag", default=None, metavar="TAG",
                   help="only with --export-question-tables: name every belief-dependent "
                        "series `<method>@<TAG>`, TAG being the HEAD (scorer) that produced "
                        "the beliefs -- `lean`, `full11`, `rom-L16-R0`, ... -- never the pool "
                        "and never the output file. REQUIRED whenever more than one head's "
                        "tables land in one pool bundle: without it two heads collide on one "
                        "method identity and the merger refuses them as having read different "
                        "chains (the 2026-09-04 G4 assembly refusal). The score-free anchors "
                        "and cap rules are NOT tagged: they are the shared baseline")
    p.add_argument("--pool", default=None, metavar="POOL",
                   help="REQUIRED with --export-question-tables: which POOL (chain bank) this "
                        "ladder read, as an assembly-inventory pool id (`select`, "
                        "`math_ood_report`, `code_ood_pooled`, ...) or a canonical bank id. It "
                        "is written to the table's `pool`/`dataset` comparability fields, "
                        "resolved through `pareto.resolve_pool`. Pool identity is NEVER derived "
                        "from an output path: the 2026-09-02 wet run found a mktemp filename in "
                        "four tables' `pool` field, and the merger then refused the comparison "
                        "as a chain-bank disagreement (contract v1.2 §1.2, v1.7 §2.6)")
    p.add_argument("--answer-format", default="boxed", choices=("boxed", "code"),
                   help="answer format of the scored pool, for the total-cost axis "
                        "(token_accounting.deployment_cost)")
    p.add_argument("--programs", default=None, metavar="JSONL",
                   help="program records for a code pool's natural-answer tokens "
                        "(token_accounting.index_programs); required with --answer-format code")
    p.add_argument("--select-answer-format", default=None, choices=("boxed", "code"),
                   help="override --answer-format for the select split only; when omitted the "
                        "global format still applies to both splits")
    p.add_argument("--select-programs", default=None, metavar="JSONL",
                   help="override --programs for the select split only; required when the "
                        "effective select answer format is code and --programs is absent")
    p.add_argument("--overhead-accounting", default=None,
                   choices=("none", "constant_per_boundary", "per_probe", "unknown"),
                   help="REQUIRED with --export-question-tables: how this method's decision-time "
                        "generation is charged on the primary total-cost axis (contract §4.3). "
                        "'none' is a CLAIM that the method generates nothing to decide")
    p.add_argument("--overhead-tokens-per-boundary", type=float, default=0.0,
                   help="with --overhead-accounting constant_per_boundary")
    p.add_argument("--overhead-key", default=None,
                   help="with --overhead-accounting per_probe: the score key holding this "
                        "method's generated tokens at each boundary")
    p.add_argument("--family-module", nargs="+", default=None, metavar="MODULE",
                   help="import these modules before building families, so they can call "
                        "ladder.register_family / register_rule_factory (e.g. "
                        "rl_controller.iclr.families_ext for custom-kind rules)")
    return p.parse_args(argv)


def load_family_specs(*, spec_paths=None, modules=None) -> list[ld.FamilySpec]:
    import importlib

    before = ld.registry_snapshot()
    for name in modules or ():
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise SystemExit(f"--family-module {name}: {exc}") from None
    specs: dict[str, ld.FamilySpec] = {s.name: s for s in ld.registered_since(before)}
    for path in spec_paths or ():
        try:
            loaded = ld.load_family_specs(path)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"--family-spec {path}: {exc}") from None
        for spec in loaded:
            previous = specs.get(spec.name)
            if previous is not None and previous != spec:
                raise SystemExit(f"--family-spec {path}: family {spec.name!r} conflicts with an "
                                 "earlier spec or a registered one")
            specs[spec.name] = spec
    return [specs[name] for name in sorted(specs)]


def _targets(config: dict) -> list[str]:
    targets = config.get("targets", list(DEFAULT_TARGETS))
    if isinstance(targets, str):
        targets = [t for t in targets.split(",") if t]
    return list(targets)


def load_run(path) -> dict:
    path = Path(path)
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = summary["config"]
    val, val_record = parse_and_record(path / "val_scored.jsonl", load_chains)
    select, select_record = parse_and_record(path / "select_scored.jsonl", load_chains)
    dumps = {"val": {"path": val_record["path"], "sha256": val_record["sha256"]},
             "select": {"path": select_record["path"], "sha256": select_record["sha256"]}}
    for split, chains in (("val", val), ("select", select)):
        if not chains:
            raise SystemExit(f"{path}: {split} split is empty")
        if any("scores" not in c for c in chains):
            raise SystemExit(f"{path}: {split} chains carry no scores")
    if "seed" not in config:
        raise SystemExit(f"{path}: summary.json config has no seed")
    return {"path": str(path), "seed": int(config["seed"]), "targets": _targets(config),
            "val": val, "select": select, "summary_path": str(summary_path),
            "dumps": dumps, "summary": summary}


def check_runs(runs: list[dict]) -> None:
    seen: dict[int, str] = {}
    for run in runs:
        if run["seed"] in seen:
            raise SystemExit(f"duplicate seed {run['seed']}: {seen[run['seed']]} and {run['path']}")
        seen[run["seed"]] = run["path"]
    for split in ("val", "select"):
        sets = {tuple(sorted({c["qkey"] for c in run[split]})) for run in runs}
        if len(sets) != 1:
            raise SystemExit(f"runs disagree on the {split} question set -- refusing to compare them")


def _manifest_records(paths) -> tuple[dict[str, str], list[dict]]:
    from rl_controller.iclr import pareto as pt

    out: dict[str, str] = {}
    records = []
    for path in paths:
        try:
            rows, record = parse_and_record(path, pt.manifest_domains)
        except pt.MergeError as exc:
            raise SystemExit(f"{path}: {exc}") from None
        for qkey, domain in rows.items():
            if out.get(qkey, domain) != domain:
                raise SystemExit(f"{path}: qkey {qkey} has domain {domain!r} here but "
                                 f"{out[qkey]!r} in an earlier manifest")
            out[qkey] = domain
        records.append({**record, "n_keys": len(rows)})
    return out, records


def _manifest_keys(paths) -> dict[str, str]:
    return _manifest_records(paths)[0]


def _applied_manifests(flag: str, paths, records) -> list[dict]:
    if records is None:
        return _manifest_records(paths)[1]
    declared = [str(Path(p).resolve()) for p in paths]
    if [record["path"] for record in records] != declared:
        raise SystemExit(
            f"{flag}: the applied manifest records {[r['path'] for r in records]} are not the "
            f"files the flag names {declared}. The export declares the manifests the run "
            "restricted itself WITH, so the two must be the same read.")
    for record in records:
        try:
            again = source_record(record["path"])
        except OSError as exc:
            raise SystemExit(
                f"{flag} {record['path']} became unreadable while this run was using it "
                f"({exc}). The export declares the manifest the run restricted itself with, "
                "so it must still be there to be re-hashed.") from None
        if again["sha256"] != record["sha256"]:
            raise SystemExit(
                f"{flag} {record['path']} CHANGED while this run was reading it: it was "
                f"applied as sha256 {record['sha256']} and now hashes to {again['sha256']}. "
                "The questions this run kept were derived from the first bytes and the "
                "export would declare the second, which is a table verified against a "
                "manifest it did not use (the 2026-09-03 Codex review of the "
                "validation-reference commit, finding 3). Re-run against an immutable copy "
                "of the manifest.")
    return records


def restriction_block(args, *, pool: str | None = None, applied=None) -> dict:
    if args.select_manifest and args.subset_domain:
        raise SystemExit(
            "--export-question-tables cannot combine --select-manifest with "
            "--subset-manifest/--subset-domain: the table declares ONE restriction so a reader "
            "can re-derive the question set it was entitled to keep (contract v1.11 §2.10), and "
            "two overlapping subsettings have no single honest declaration. Run them as two "
            "exports.")
    if args.select_manifest:
        records = _applied_manifests("--select-manifest", args.select_manifest,
                                     (applied or {}).get("--select-manifest"))
        return {"kind": "select_manifest",
                "manifests": _registered(
                    records, "--select-manifest", restriction_kind="select_manifest",
                    pool=pool or args.pool, splits=("select",)), "domain": None}
    if args.subset_domain:
        records = _applied_manifests("--subset-manifest", args.subset_manifest,
                                     (applied or {}).get("--subset-manifest"))
        return {"kind": "subset_domain",
                "manifests": _registered(
                    records, "--subset-manifest", restriction_kind="subset_domain",
                    pool=pool or args.pool, splits=("val", "select")),
                "domain": args.subset_domain}
    return {"kind": "none", "manifests": [], "domain": None}


def _registered(records, flag: str, *, restriction_kind: str, pool: str,
                splits: tuple[str, ...]) -> list[dict]:
    from rl_controller.iclr import pareto as pt

    for record in records:
        try:
            pt.check_manifest_registered(
                record["path"], record["sha256"], restriction_kind=restriction_kind,
                pool=pool, splits=splits)
        except pt.MergeError as exc:
            raise SystemExit(f"{flag} {record['path']}: {exc}") from None
    return records


def subset_runs(runs: list[dict], *, domain_of: dict[str, str], domain: str) -> dict:
    kept = {"val": 0, "select": 0}
    for run in runs:
        for split in ("val", "select"):
            unknown = sorted({c["qkey"] for c in run[split] if c["qkey"] not in domain_of})
            if unknown:
                raise SystemExit(f"{run['path']}: {len(unknown)} {split} question(s) not in any --subset-manifest "
                                 f"(e.g. {unknown[:3]})")
            run[split] = [c for c in run[split] if domain_of[c["qkey"]] == domain]
            if not run[split]:
                raise SystemExit(f"{run['path']}: {split} split is empty after --subset-domain {domain}")
            kept[split] = len({c["qkey"] for c in run[split]})
    return {"domain": domain, "kept": kept}


FITTED_SCORE_SUFFIXES = ("_cal", "_iso")


def refit_score_keys() -> frozenset:
    class _Stub:
        def calibrate(self, value, depth):
            return 0.0

    score = {"q": 0.0, "v": 0.0}
    chain = {"probes": [{"block": 1, "prefix_len": 1, "forced": 0, "forced_tokens": 0}],
             "scores": [dict(score)]}
    attached = ld.attach_calibrated([chain], {"q": _Stub(), "v": _Stub()})
    return frozenset(attached[0]["scores"][0]) - frozenset(score)


def producer_fitted_score_keys(run: dict) -> list[str]:
    refit = refit_score_keys()
    keys = ld.score_keys(run["val"]) | ld.score_keys(run["select"])
    return sorted(k for k in keys
                  if k.endswith(FITTED_SCORE_SUFFIXES) and k not in refit)


def refuse_producer_fitted_scores(runs: list[dict]) -> None:
    refit = sorted(refit_score_keys())
    for run in runs:
        keys = producer_fitted_score_keys(run)
        if keys:
            raise SystemExit(
                f"--validation-reference-table: {run['path']} carries producer-fitted "
                f"calibrated score key(s) {keys}, which this ladder does not refit (it "
                f"refits only {refit}). Projecting val onto the reference population would "
                "leave those predictions fitted on the chains the projection EXCLUDES, on "
                "val and on select alike, and the exported table would still authenticate "
                "-- it names the projected bytes, not what fitted the scores in them (the "
                "2026-09-03 Codex review of the validation-reference commit, finding 1). "
                "Refit that key on the reference population after projection, the way "
                "results/iclr/trainingfree/greedy_full/reads_greedy_full.py does, or drop "
                "it from the dumps and read the raw belief.")


def project_onto_validation_reference(runs: list[dict], *, table, out_root,
                                      out_dir=None,
                                      keep_questions: set | None = None) -> tuple[list, dict]:
    from rl_controller.iclr import validation_basis as vb

    refuse_producer_fitted_scores(runs)
    try:
        basis = vb.reference_validation_basis(table)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"--validation-reference-table {table}: {exc}") from None
    record = basis["record"]
    root = (Path(out_dir) if out_dir else
            Path(out_root) / "aligned_val" / record["table_sha256"][:16])
    by_seed = basis["chains_by_seed"]
    projected, seen = [], {}
    for run in runs:
        seed = str(run["seed"])
        if seed not in by_seed:
            raise SystemExit(
                f"--validation-reference-table {record['reference_table']}: {run['path']} is "
                f"seed {seed}, which the reference does not cover (its seeds are "
                f"{sorted(by_seed)}). A run is projected onto the reference population of its "
                "OWN seed; export the reference for this seed, or ladder the seeds the "
                "reference covers.")
        reference = by_seed[seed]
        if keep_questions is not None:
            reference = {identity: chain for identity, chain in reference.items()
                         if identity[0] in keep_questions}
        components = (f"seed{seed}", Path(run["path"]).name)
        if components in seen:
            raise SystemExit(
                f"--validation-reference-table {record['reference_table']}: {run['path']} and "
                f"{seen[components]} would write their aligned val views to the same place "
                f"({'/'.join(components)}). Rename one run dir; a derived validation view is "
                "never shared between two runs.")
        seen[components] = run["path"]
        try:
            projected.append(vb.align_validation_run(
                run, reference, output_root=root, components=components,
                reference_record=record))
        except (OSError, ValueError, RuntimeError) as exc:
            raise SystemExit(
                f"--validation-reference-table {record['reference_table']}: {exc}") from None
    return projected, {"reference": record, "out_dir": str(root),
                       "runs": [run["validation_basis"] for run in projected]}


def restrict_select(runs: list[dict], *, keys: set[str]) -> None:
    for run in runs:
        run["select"] = [c for c in run["select"] if c["qkey"] in keys]
        if not run["select"]:
            raise SystemExit(f"{run['path']}: select split is empty after --select-manifest")


def with_anchors(family) -> list[tuple[str, ld.Plan]]:
    return [*family, (ld.NATURAL, ld.natural), (ld.ALWAYS_FIRE, ld.always_fire)]


def split_summary(table: dict) -> dict:
    rows = ld.rows_from_table(table)
    by_name = {r["name"]: r for r in rows}
    natural, always = by_name[ld.NATURAL], by_name[ld.ALWAYS_FIRE]
    rule_rows = [r for r in rows if r["name"] not in (ld.NATURAL, ld.ALWAYS_FIRE)]
    hull = lc.anchor_hull(rule_rows, natural, always)
    out = lc.matched_saving(hull, natural)
    out.update({"hull": hull, "natural": natural, "always_fire": always, "rows": rule_rows,
                "_by_name": by_name})
    return out


def level_key(level: float) -> str:
    return f"{int(round(100 * level))}pp"


def honest_selection(val: dict, select: dict) -> dict:
    members = ld.select_members(val["rows"], val["natural"], levels=LEVELS)
    by_level = {}
    for level in LEVELS:
        picked = ld.select_members(val["rows"], val["natural"], levels=(level,))
        by_level[level_key(level)] = picked[0] if picked else None
    points = []
    for name in members:
        row = select["_by_name"][name]
        levels = [k for k, m in by_level.items() if m == name]
        points.append({**row, "val_accuracy": val["_by_name"][name]["accuracy"],
                       "val_mean_thinking_tokens": val["_by_name"][name]["mean_thinking_tokens"],
                       "levels": levels})
    return {"members": members, "by_level": by_level, "points": points}


def l7_band(chains) -> dict:
    n = 0
    width = 0.0
    inside = 0
    for chain in chains:
        for score in chain["scores"]:
            n += 1
            width += max(0.0, score["r"] - score["vpi0"])
            inside += score["vpi0"] <= score["q"] < score["r"]
    return {"n_probes": n, "band_mean": (width / n) if n else None,
            "in_band_frac": (inside / n) if n else None}


def filter_families(families, only):
    if only is None:
        return families
    wanted = set(only) | {"L1"}
    missing = wanted - families.keys()
    if missing:
        raise ValueError(f"requested families unavailable: {sorted(missing)}")
    return {k: v for k, v in families.items() if k in wanted}


def evaluate_run(run: dict, *, calib_folds: int, specs=None, keep_chains: bool = False,
                 calibrate: bool = True, only_families=None) -> dict:
    if calibrate:
        try:
            models = ld.fit_calibrators(run["val"])
            select_cal = ld.attach_calibrated(run["select"], models)
            val_xf, xf_records = ld.crossfit_calibrated(run["val"], n_folds=calib_folds)
        except dm.InsufficientSupport as exc:
            raise SystemExit(f"{run['path']}: val split too small to calibrate ({exc}); "
                             "the ladder needs a val split with >= 20 positive and 20 negative labels per head")
    else:
        models, xf_records = None, []
        select_cal, val_xf = list(run["select"]), list(run["val"])
    keys = ld.score_keys(select_cal) & ld.score_keys(val_xf)
    fams = ld.families(keys, specs=specs or (), include_registered=False)
    fams = filter_families(fams, only_families)
    rungs = {}
    for rung, family in fams.items():
        family = with_anchors(family)
        table_val = ld.question_table(val_xf, family)
        table_sel = ld.question_table(select_cal, family)
        val, sel = split_summary(table_val), split_summary(table_sel)
        rungs[rung] = {"n_members": len(fams[rung]), "val": val, "select": sel,
                       "honest": honest_selection(val, sel), "_table_select": table_sel}
    free_val = ld.question_table(val_xf, ld.free_family())
    free_sel = ld.question_table(select_cal, ld.free_family())
    agr_family = [(ld.AGREEMENT, stop_earliest_agreement), (ld.NATURAL, ld.natural)]
    agr_val = ld.question_table(val_xf, agr_family)
    agr_sel = ld.question_table(select_cal, agr_family)
    out = {
        "seed": run["seed"], "score_keys": sorted(keys),
        "calibration": ({"q_c": models["q_c"], "v_c": models["v_c"],
                         "q_coefficients": list(models["q"].coefficients),
                         "v_coefficients": list(models["v"].coefficients),
                         "n_questions": models["n_questions"], "crossfit": xf_records}
                        if models is not None else
                        {"skipped": "--no-calibration: this head's q is not a probability "
                                    "(D11.4b), so no calibrated rung is valid for it and none "
                                    "was computed"}),
        "rungs": rungs,
        "free": {"val": split_summary(free_val), "select": split_summary(free_sel),
                 "_table_select": free_sel},
        "agreement": {"val": {r["name"]: r for r in ld.rows_from_table(agr_val)}[ld.AGREEMENT],
                      "select": {r["name"]: r for r in ld.rows_from_table(agr_sel)}[ld.AGREEMENT],
                      "_table_select": agr_sel},
    }
    if {"vpi0", "r"} <= keys:
        out["l7_band"] = {"val": l7_band(val_xf), "select": l7_band(select_cal)}
    if keep_chains:
        out["_chains"] = {"val": val_xf, "select": select_cal}
        out["_families"] = fams
    return out


def rung_sort_key(rung: str):
    for i, prefix in enumerate(RUNG_ORDER):
        if rung == prefix or (prefix.endswith("_") and rung.startswith(prefix)):
            return (i, rung)
    return (len(RUNG_ORDER), rung)


def _public_split(split: dict) -> dict:
    return {k: v for k, v in split.items() if not k.startswith("_")}


def _points(honest: dict) -> list[tuple[float, float]]:
    return [(p["mean_thinking_tokens"], p["accuracy"]) for p in honest["points"]]


def aggregate(evals: list[dict], *, n_boot: int, rng, external: Sequence[str] = ()) -> dict:
    external = set(external)
    by_seed = {e["seed"]: e for e in evals}
    seeds = sorted(by_seed)
    rung_names = sorted({r for e in evals for r in e["rungs"]}, key=rung_sort_key)
    rungs = {}
    for rung in rung_names:
        per_seed = {}
        verdict_in = {}
        for seed in seeds:
            e = by_seed[seed]
            if rung not in e["rungs"]:
                raise SystemExit(f"rung {rung} missing on seed {seed}: runs carry different belief keys")
            entry = e["rungs"][rung]
            natural_sel = entry["select"]["natural"]
            l1_hull = e["rungs"]["L1"]["select"]["hull"]
            per_seed[str(seed)] = {
                "val": _public_split(entry["val"]),
                "select_sweep": _public_split(entry["select"]),
                "select_honest": entry["honest"],
                "vs_L1_op": lc.operating_point_delta(l1_hull, entry["select"]["hull"], natural_sel),
                "vs_L1_op_val": lc.operating_point_delta(e["rungs"]["L1"]["val"]["hull"],
                                                         entry["val"]["hull"], entry["val"]["natural"]),
            }
            verdict_in[seed] = {"points": _points(entry["honest"]),
                                "free_hull": e["free"]["select"]["hull"],
                                "agreement": e["agreement"]["select"]}
        if rung in external:
            verdict = {"kind": "external", "per_seed": None, "pass": None, "n_seeds": len(seeds),
                       "note": "not a §10 rung -- read on the cross-method hull "
                               "(rl_controller.iclr.pareto), not on this verdict"}
        else:
            verdict = ld.go_no_go(verdict_in)
            verdict["per_seed"] = {str(s): v for s, v in verdict["per_seed"].items()}
            verdict["kind"] = "builtin"
        rungs[rung] = {"members": by_seed[seeds[0]]["rungs"][rung]["n_members"],
                       "kind": "external" if rung in external else "builtin",
                       "per_seed": per_seed, "go_no_go": verdict}
    tables_by_seed = {seed: {"rungs": {r: by_seed[seed]["rungs"][r]["_table_select"] for r in rung_names},
                             "free": by_seed[seed]["free"]["_table_select"],
                             "agreement": by_seed[seed]["agreement"]["_table_select"]}
                      for seed in seeds}
    members_by_rung = {r: {seed: by_seed[seed]["rungs"][r]["honest"]["members"] for seed in seeds}
                       for r in rung_names}
    boot = ld.paired_bootstrap(tables_by_seed, members_by_rung=members_by_rung,
                               free_names=[n for n, _ in ld.free_family()], l1_name="L1",
                               n=n_boot, rng=rng)
    for rung in rung_names:
        rungs[rung]["bootstrap"] = boot["rungs"][rung] | {"n": boot["n"]} if boot["n"] else {"n": 0}
    return {
        "rungs": rungs,
        "free_hull": {str(s): {"val": _public_split(by_seed[s]["free"]["val"]),
                               "select": _public_split(by_seed[s]["free"]["select"])} for s in seeds},
        "agreement": {str(s): {"val": by_seed[s]["agreement"]["val"],
                               "select": by_seed[s]["agreement"]["select"]} for s in seeds},
        "calibration": {str(s): by_seed[s]["calibration"] for s in seeds},
        "l7_band": {str(s): by_seed[s]["l7_band"] for s in seeds if "l7_band" in by_seed[s]},
        "bootstrap": {"n": boot["n"], "n_seeds": boot.get("n_seeds"),
                      "n_questions": boot.get("n_questions"), "semantics": BOOTSTRAP_SEMANTICS},
    }


def grids(specs=None) -> dict:
    return {"external_families": {s.name: s.to_dict() for s in (specs or ())},
            "thetas": list(ld.THETAS), "margins": list(ld.MARGINS),
            "linear_a": list(ld.LINEAR_A), "linear_b": list(ld.LINEAR_B),
            "step_margins": [list(m) for m in dm.monotone_candidates()],
            "lambdas": list(ld.LAMBDAS), "hysteresis_k": list(ld.HYSTERESIS_K),
            "cusum_h": list(ld.CUSUM_H), "bracket_mhi": list(ld.BRACKET_MHI),
            "b_plus_m": list(ld.B_PLUS_M), "free_caps": list(ld.FREE_CAPS),
            "select_levels": list(LEVELS)}


def chain_authentication(entries) -> tuple[dict, dict]:
    sources: dict[str, dict] = {}
    identities: dict[str, dict] = {}
    for run, seed, chains in entries:
        for split, split_chains in chains.items():
            dump = (run.get("dumps") or {}).get(split)
            if dump is None:
                raise SystemExit(
                    f"{run['path']}: no recorded {split} scored dump -- the export must name "
                    "the bytes every series was computed from (contract v1.10 §2.10)")
            previous = (sources.get(str(seed)) or {}).get(split)
            if previous is not None and previous["path"] != dump["path"]:
                raise SystemExit(
                    f"{run['path']}: seed {seed} split {split!r} is already bound to "
                    f"{previous['path']}; one (seed, split) names ONE dump, so a table may not "
                    "merge two run dirs under one seed. Export one table per run dir and merge "
                    "them at read time.")
            pairs = sorted({(str(c["qkey"]), int(c["rep"])) for c in split_chains})
            identities.setdefault(str(seed), {})[split] = [[q, r] for q, r in pairs]
            sources.setdefault(str(seed), {})[split] = {
                "path": dump["path"], "sha256": dump["sha256"], "n_chains": len(pairs)}
    return sources, identities


UNTAGGED_ANCHORS = frozenset({ld.NATURAL, ld.ALWAYS_FIRE, ld.AGREEMENT})
_HEAD_TAG_FORBIDDEN = ("@", "/", ":")


def check_head_tag(tag):
    if tag is None:
        return None
    bad = [c for c in _HEAD_TAG_FORBIDDEN if c in tag] + \
          (["whitespace"] if any(c.isspace() for c in tag) else [])
    if not tag or bad:
        raise SystemExit(
            f"--head-tag {tag!r}: a head tag NAMES THE SCORER whose beliefs this ladder read "
            "(`lean`, `full11`, `rom-L16-R0`) and is spelled into every belief-dependent "
            "method as `<method>@<TAG>`, so it must be one non-empty token free of whitespace "
            f"and of {' '.join(_HEAD_TAG_FORBIDDEN)}"
            + (f" -- this one carries {', '.join(map(repr, bad))}" if bad else ""))
    return tag


def tagged_method(method: str, member: str, head_tag) -> str:
    from rl_controller.iclr import pareto as pt

    if not head_tag or method in UNTAGGED_ANCHORS \
            or pt.is_free_rule(method) or pt.is_free_rule(member):
        return method
    return f"{method}@{head_tag}"


def export_question_tables(path, runs: list[dict], evals: list[dict], *, scope: str,
                           costing: dict, overhead: dict, pool: str = "",
                           costing_by_split: dict[str, dict] | None = None,
                           provenance: Sequence[dict] = (),
                           comparability: dict | None = None,
                           restriction: dict | None = None,
                           head_tag: str | None = None) -> dict:
    from rl_controller.iclr import pareto as pt

    block = dict(comparability or {})
    if block and block.get("pool", pool) != pool:
        raise ValueError(f"comparability pool {block['pool']!r} != export pool {pool!r}")
    bundle = pt.Bundle(pool=pool, normal_form="chain")
    split_costing = {split: dict(costing) for split in ("val", "select")}
    for split, override in (costing_by_split or {}).items():
        if split not in split_costing:
            raise ValueError(f"unsupported costing split {split!r}")
        split_costing[split] = dict(override)
    val_method_costing = {**split_costing["val"], **overhead}
    meta = {"pool": pool, **{k: v for k, v in (provenance[0] if provenance else {}).items()
                             if k in pt.COMPARABILITY_FIELDS}, **block}
    sources, identities = chain_authentication(
        (run, entry["seed"], entry["_chains"]) for run, entry in zip(runs, evals))
    for run, entry in zip(runs, evals):
        seed = entry["seed"]
        chains, fams = entry["_chains"], entry["_families"]
        run_meta = {**meta, **_export_identity(run), **block}
        by_split = chain_identity_by_split(run, chains)
        for split, split_chains in chains.items():
            series_meta = {**run_meta, **by_split[split]}
            anchor_costing = split_costing[split]
            method_costing = {**anchor_costing, **overhead}
            wanted: dict[str, list] = {}
            for rung, family in fams.items():
                if scope == "all":
                    keep = list(family)
                else:
                    keep = _honest_union(family, split_chains if split == "val" else None,
                                         entry["rungs"][rung]["honest"]["members"],
                                         run, entry, rung, val_method_costing)
                if keep:
                    wanted[rung] = keep
            wanted[ld.AGREEMENT] = [(ld.AGREEMENT, stop_earliest_agreement)]
            wanted["free"] = [(f"cap{cap}", lambda ch, _b=cap: stop_at_cap(ch, _b))
                              for cap in ld.FREE_CAPS]
            wanted["free"].append(("consecutive_agreement", consecutive_agreement))
            for method, family in wanted.items():
                for member, per_q in pt.question_table_costed(
                        split_chains, family, **method_costing).items():
                    bundle.add(pt.SeriesKey(tagged_method(method, member, head_tag),
                                            member, seed, split), per_q,
                               kind="plan_rule", meta=series_meta)
            for anchor, plan in ((ld.NATURAL, ld.natural), (ld.ALWAYS_FIRE, ld.always_fire)):
                for member, per_q in pt.question_table_costed(
                        split_chains, [(anchor, plan)], **anchor_costing).items():
                    bundle.add(pt.SeriesKey(tagged_method(anchor, member, head_tag),
                                            member, seed, split), per_q,
                               kind="plan_rule", meta=series_meta)
    return pt.dump_question_tables(path, bundle, scope=scope, provenance=list(provenance),
                                   comparability=block, sources=sources,
                                   identities=identities, restriction=restriction,
                                   head_tag=head_tag)


CAMPAIGN_IDENTITY = {"model_id": "Qwen/Qwen3-1.7B",
                     "model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                     "tokenizer_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                     "prompt_contract_id": "qwen3-1.7b-native-thinking-v1"}
GRADER_OF_FORMAT = {"boxed": "archived_math_equal", "code": "archived_code_sandbox"}
_IDENTITY_KEYS = {
    "model_id": ("model_id", "model"),
    "model_revision": ("model_revision", "revision"),
    "tokenizer_revision": ("tokenizer_revision",),
    "prompt_contract_id": ("prompt_contract_id",),
    "grader_id": ("grader_id",),
}


def _declared(run: dict, field_: str):
    summary = run.get("summary") or {}
    for block in (summary.get("provenance") or {}, summary.get("config") or {}):
        for key in _IDENTITY_KEYS[field_]:
            value = block.get(key)
            if value is not None:
                return value
    return None


def comparability_block(runs: list[dict], *, pool: str, answer_format: str) -> tuple[dict, dict]:
    from rl_controller.iclr import pareto as pt

    try:
        resolved = pt.resolve_pool(pool)
    except pt.MergeError as exc:
        raise SystemExit(f"--pool: {exc}") from None
    block = {"pool": resolved, "dataset": resolved}
    sources = {"pool": "--pool", "dataset": "canonical pool family (contract v1.7 §2.6)"}
    for run in runs:
        summary = run.get("summary") or {}
        for scope in (summary.get("provenance") or {}, summary.get("config") or {}):
            for field_ in ("dataset", "pool"):
                value = scope.get(field_)
                if value and pt.canonical_pool(value) != resolved:
                    raise SystemExit(
                        f"--pool {pool} resolves to {resolved!r}, but {run['path']} declares "
                        f"{field_}={value!r}. Pass the pool this ladder actually read; the "
                        "export refuses to stamp a bank id the run dirs contradict.")
    fallback = {**CAMPAIGN_IDENTITY,
                "grader_id": GRADER_OF_FORMAT[answer_format]}
    fallback_source = {f: ("derived from --answer-format " + answer_format
                           if f == "grader_id" else "campaign constant")
                       for f in fallback}
    for field_ in ("grader_id", "model_id", "model_revision", "tokenizer_revision",
                   "prompt_contract_id"):
        resolved_by_run: dict[str, list[str]] = {}
        declared_by = []
        for run in runs:
            value = _declared(run, field_)
            if value is None:
                value, source = fallback[field_], "fallback"
            else:
                value, source = str(value), "declared"
                declared_by.append(run["path"])
            resolved_by_run.setdefault(f"{value} ({source})", []).append(run["path"])
        if len({v.rsplit(" (", 1)[0] for v in resolved_by_run}) > 1:
            raise SystemExit(
                f"run dirs resolve {field_} differently: "
                + json.dumps(resolved_by_run, sort_keys=True)
                + " -- a run that declares nothing resolves to the campaign constant, so a "
                  "declaration by ONE seed does not speak for the other; one export carries "
                  "one identity")
        block[field_] = next(iter(resolved_by_run)).rsplit(" (", 1)[0]
        sources[field_] = ("run summary" if len(declared_by) == len(runs) else
                           fallback_source[field_] if not declared_by else
                           f"run summary ({len(declared_by)}/{len(runs)} runs; the rest resolve "
                           f"to the same value as the {fallback_source[field_]})")
    return block, sources


def _export_identity(run: dict) -> dict:
    summary = run.get("summary") or {}
    prov = summary.get("provenance") or {}
    config = summary.get("config") or {}
    out = {"model_id": prov.get("model_id") or prov.get("model") or config.get("model"),
           "model_revision": prov.get("model_revision") or prov.get("revision"),
           "dataset": prov.get("dataset") or config.get("dataset"),
           "grader_id": prov.get("grader_id"),
           "prompt_contract_id": prov.get("prompt_contract_id")}
    return {k: v for k, v in out.items() if v}


def _chain_identity(prov: dict) -> dict:
    out = {"manifest_sha256": prov.get("manifest_sha256"),
           "chain_bank_sha256": prov.get("chains_sha256") or prov.get("chain_bank_sha256")}
    return {k: v for k, v in out.items() if v}


VAL_SOURCE_KEY = "val_source"


def partition_subset_rule(*, val_count: int, val_salt: str,
                          max_chains_per_question: int) -> dict:
    return {"kind": "partition", "val_count": int(val_count), "val_salt": str(val_salt),
            "max_chains_per_question": int(max_chains_per_question),
            "rule": "lc.partition(trainable fit questions, n_val=val_count, salt=val_salt) "
                    "after lc.cap_chains at max_chains_per_question"}


def reference_dump_subset_rule(*, path, n_val_questions: int) -> dict:
    return {"kind": "reference_dump", "path": str(path),
            "n_val_questions": int(n_val_questions),
            "rule": "the val question set of the reference run's val_scored.jsonl, "
                    "intersected with the fit feature cache"}


def val_source_record(*, derived_from: str, chains_path, chains_sha256: str,
                      manifest_path, manifest_sha256: str, subset_rule: dict) -> dict:
    if not (isinstance(chains_sha256, str) and isinstance(manifest_sha256, str)
            and chains_sha256 and manifest_sha256):
        raise ValueError("val_source needs BOTH digests: half an identity is not a declaration")
    if (subset_rule or {}).get("kind") not in ("partition", "reference_dump"):
        raise ValueError("val_source needs a subset rule that says how the split was picked")
    return {"derived_from": derived_from,
            "chain_bank_sha256": chains_sha256, "manifest_sha256": manifest_sha256,
            "chains_path": str(chains_path), "manifest_path": str(manifest_path),
            "subset_rule": dict(subset_rule)}


def _digests(block: dict) -> dict:
    out = {"chain_bank_sha256": (block or {}).get("chain_bank_sha256"),
           "manifest_sha256": (block or {}).get("manifest_sha256")}
    return {k: v for k, v in out.items() if v}


def declared_fit_bank(prov: dict) -> str | None:
    flat = prov.get("fit_chains_sha256")
    if isinstance(flat, str) and flat:
        return flat
    records = prov.get("fit_chains")
    if not isinstance(records, list):
        return None
    shas = {r.get("sha256") for r in records if isinstance(r, dict) and r.get("sha256")}
    return shas.pop() if len(shas) == 1 else None


def _fit_manifest_from_features(prov: dict, fit_bank: str) -> str | None:
    path = ((prov.get("features") or {}).get("fit") if isinstance(prov.get("features"), dict)
            else None)
    if not isinstance(path, str) or not path:
        return None
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if manifest.get("chains_sha256") != fit_bank:
        return None
    sha = (manifest.get("questions") or {}).get("sha256")
    return sha if isinstance(sha, str) and sha else None


def resolve_validation_source(prov: dict) -> dict:
    fit_bank = declared_fit_bank(prov)
    declared = _digests(prov.get(VAL_SOURCE_KEY))
    flat = _digests({"chain_bank_sha256": prov.get("val_chains_sha256"),
                     "manifest_sha256": prov.get("val_manifest_sha256")})
    if len(declared) == 2:
        if fit_bank and declared["chain_bank_sha256"] != fit_bank:
            return {"val_identity_unresolved":
                    f"val_source names bank {declared['chain_bank_sha256']} but the run "
                    f"declares it fitted on {fit_bank}; a trainer that carves val out of fit "
                    "cannot have validated on a third bank"}
        if len(flat) == 2 and flat != declared:
            return {"val_identity_unresolved":
                    f"val_source says {declared} while val_chains_sha256 / "
                    f"val_manifest_sha256 say {flat}: one summary, two statements about "
                    "which bank the validation rows came from"}
        return {**declared, "val_identity_source": VAL_SOURCE_KEY}
    if len(flat) == 2:
        if fit_bank and flat["chain_bank_sha256"] != fit_bank:
            return {"val_identity_unresolved":
                    f"val_chains_sha256 is {flat['chain_bank_sha256']} while the run declares "
                    f"fit bank {fit_bank}: this run's val split was carved out of the FIT bank, "
                    "so the flat pair names a bank the validation rows did not come from "
                    "(the 2026-09-11 select-into-val copy)"}
        return {**flat, "val_identity_source": "val_chains_sha256/val_manifest_sha256"}
    if not fit_bank:
        return {"val_identity_unresolved": "the run declares no fit bank and no val source"}
    fit_manifest = prov.get("fit_manifest_sha256")
    if isinstance(fit_manifest, str) and fit_manifest:
        return {"chain_bank_sha256": fit_bank, "manifest_sha256": fit_manifest,
                "val_identity_source": "fit_chains_sha256/fit_manifest_sha256"}
    fit_manifest = _fit_manifest_from_features(prov, fit_bank)
    if fit_manifest:
        return {"chain_bank_sha256": fit_bank, "manifest_sha256": fit_manifest,
                "val_identity_source": "provenance.features.fit manifest"}
    return {"val_identity_unresolved":
            f"fit bank {fit_bank} is declared but its question manifest is not: no "
            "fit_manifest_sha256, and provenance.features.fit is unreadable here"}


def _in_distribution_identity(run: dict) -> dict:
    summary = run.get("summary") or {}
    prov, config = summary.get("provenance") or {}, summary.get("config") or {}
    own = _digests(resolve_validation_source(prov))
    if len(own) == 2:
        return own
    source_run = prov.get("source_run") or config.get("source_run")
    if not source_run:
        return {}
    try:
        source = json.loads((Path(source_run) / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    sprov = source.get("provenance") or {}
    resolved = _digests(resolve_validation_source(sprov))
    if len(resolved) == 2:
        return resolved
    bank = declared_fit_bank(sprov)
    questions = [q.get("sha256") for q in (sprov.get("questions") or [])
                 if isinstance(q, dict) and q.get("sha256")]
    if not bank or len(questions) != 1:
        return {}
    return {"chain_bank_sha256": bank, "manifest_sha256": questions[0]}


def chain_identity_by_split(run: dict, splits) -> dict:
    prov = (run.get("summary") or {}).get("provenance") or {}
    read = _chain_identity(prov)
    val = _in_distribution_identity(run)
    if not val:
        fit_bank = declared_fit_bank(prov)
        contradicted = bool(fit_bank) and fit_bank != read.get("chain_bank_sha256")
        val = read if not (_is_scored_elsewhere(run) or contradicted) else {}
    return {split: dict(val if split == "val" else read) for split in splits}


def bank_question_keys(chains_path, *, trainable_only: bool = False) -> set:
    from rl_controller.iclr.chains import is_trainable

    keys = set()
    with open(chains_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if trainable_only and not is_trainable(row):
                continue
            keys.add(row["qkey"])
    return keys


def verify_validation_bank(val_qkeys, *, chains_path, declared_sha256=None,
                           val_count=None, val_salt=None, max_chains_per_question=None,
                           val_chains=None) -> dict:
    val_qkeys = set(val_qkeys)
    bank_all = bank_question_keys(chains_path)
    failures, receipt = [], {"bank_path": str(chains_path),
                             "bank_questions": len(bank_all),
                             "val_questions": len(val_qkeys)}
    missing = sorted(val_qkeys - bank_all)
    receipt["val_questions_in_bank"] = len(val_qkeys) - len(missing)
    receipt["val_questions_missing"] = len(missing)
    receipt["missing_examples"] = missing[:5]
    if missing:
        failures.append(f"{len(missing)} of {len(val_qkeys)} validation questions are NOT in "
                        f"{chains_path} (e.g. {missing[:3]}): the declared bank does not hold "
                        "the rows honest selection is performed on")
    if declared_sha256:
        digest = hashlib.sha256(Path(chains_path).read_bytes()).hexdigest()
        receipt["declared_sha256"], receipt["file_sha256"] = declared_sha256, digest
        if digest != declared_sha256:
            failures.append(f"{chains_path} hashes to {digest}, the declaration says "
                            f"{declared_sha256}")
    if val_count is not None and val_salt is not None:
        trainable = bank_question_keys(chains_path, trainable_only=True)
        recomputed, _train = lc.partition(trainable, n_val=int(val_count), salt=str(val_salt))
        receipt["subset_rule"] = {"val_count": int(val_count), "val_salt": str(val_salt),
                                  "trainable_questions": len(trainable),
                                  "reproduces_val_set": set(recomputed) == val_qkeys}
        if set(recomputed) != val_qkeys:
            only_rule = sorted(set(recomputed) - val_qkeys)[:3]
            only_dump = sorted(val_qkeys - set(recomputed))[:3]
            failures.append(
                f"the subset rule over {chains_path} yields a different val question set "
                f"({len(set(recomputed) - val_qkeys)} only in the rule, e.g. {only_rule}; "
                f"{len(val_qkeys - set(recomputed))} only in the dump, e.g. {only_dump}): the "
                "bank may hold the questions and still not be the bank the split was cut from")
    if val_chains is not None and max_chains_per_question is not None:
        from rl_controller.baselines.tc import lean
        from rl_controller.iclr.chains import is_trainable

        chains = [c for c in lean.stream_chains_lite(chains_path) if is_trainable(c)]
        keep = {(c["qkey"], int(c["rep"])) for c in lc.cap_chains(chains,
                                                                 int(max_chains_per_question))}
        mine = {(q, rep) for q, rep in keep if q in val_qkeys}
        receipt["capped_val_chains"] = {"max_chains_per_question": int(max_chains_per_question),
                                        "from_bank": len(mine), "in_dump": len(set(val_chains)),
                                        "reproduces_val_chains": mine == set(val_chains)}
        if mine != set(val_chains):
            failures.append(
                f"at cap={max_chains_per_question} the bank yields {len(mine)} val chains and "
                f"the dump holds {len(set(val_chains))}: the question identities can match "
                "while the population does not")
    receipt["failures"] = failures
    receipt["holds"] = not failures
    return receipt


SCORED_ELSEWHERE_KEYS = ("source_run", "readout_only")


def _is_scored_elsewhere(run: dict) -> bool:
    summary = run.get("summary") or {}
    prov, config = summary.get("provenance") or {}, summary.get("config") or {}
    return any(prov.get(key) or config.get(key) for key in SCORED_ELSEWHERE_KEYS)


def _honest_union(family, val_chains, thinking_members, run, entry, rung, costing) -> list:
    from rl_controller.iclr import pareto as pt

    keep = set(thinking_members)
    cached = entry.setdefault("_total_honest", {})
    if rung not in cached:
        chains = entry["_chains"]["val"]
        table = pt.question_table_costed(chains, [*family, (ld.NATURAL, ld.natural)], **costing)
        qkeys = sorted(table[ld.NATURAL])
        rows = [pt.mean_row(name, per_q, qkeys, cost="total")
                for name, per_q in table.items() if name != ld.NATURAL]
        natural = pt.mean_row(ld.NATURAL, table[ld.NATURAL], qkeys, cost="total")
        cached[rung] = ld.select_members(rows, natural, levels=LEVELS)
    keep.update(cached[rung])
    return [(name, plan) for name, plan in family if name in keep]


def _load_programs(path):
    if path is None:
        return None
    from rl_controller.iclr import token_accounting as ta

    with open(path, encoding="utf-8") as handle:
        return ta.index_programs(json.loads(line) for line in handle if line.strip())


def _split_costing_args(args, *, load_programs: bool = False) -> dict[str, dict]:
    formats = {
        "val": args.answer_format,
        "select": args.select_answer_format or args.answer_format,
    }
    paths = {
        "val": args.programs,
        "select": args.select_programs or args.programs,
    }
    for split in ("val", "select"):
        if formats[split] == "code" and not paths[split]:
            flag = "--programs" if split == "val" else "--select-programs (or --programs)"
            raise SystemExit(
                f"effective {split} --answer-format code needs {flag}: a code pool's natural "
                "answer tokens live in the program sidecar, not in the chain"
            )
    loaded: dict[str, dict | str | None] = {}
    by_path: dict[str, dict] = {}
    for split in ("val", "select"):
        path = paths[split]
        if load_programs and path is not None and path not in by_path:
            by_path[path] = _load_programs(path)
        loaded[split] = by_path.get(path) if load_programs and path is not None else path
    return {
        split: {"answer_format": formats[split], "programs": loaded[split]}
        for split in ("val", "select")
    }


def parse_l3_member(name: str) -> tuple[float, float]:
    match = _L3_NAME.match(name)
    if not match:
        raise ValueError(f"not an L3 linear member name: {name!r}")
    a, b = float(match.group(1)), float(match.group(2))
    snap = lambda x, grid: min(grid, key=lambda g: abs(g - x))
    return snap(a, dm.LINEAR_A), snap(b, dm.LINEAR_B)


def export_row3(path, evals: list[dict], runs: list[dict]) -> dict:
    e = min(evals, key=lambda e: e["seed"])
    run = next(r for r in runs if r["seed"] == e["seed"])
    honest = e["rungs"]["L3"]["honest"]
    if not honest["members"]:
        raise SystemExit("--export-row3: no L3 member reaches natural accuracy - 3 pp on val")
    level = next(k for k, m in honest["by_level"].items() if m == honest["members"][0])
    member = honest["members"][0]
    a, b = parse_l3_member(member)
    cal = e["calibration"]
    row3 = {"a": a, "b": b, "q_coefficients": cal["q_coefficients"],
            "v_coefficients": cal["v_coefficients"]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row3, indent=2), encoding="utf-8")
    point = next(p for p in honest["points"] if p["name"] == member)
    meta = {"rung": "L3", "member": member, "seed": e["seed"], "level": level,
            "q_c": cal["q_c"], "v_c": cal["v_c"], "n_calibration_questions": cal["n_questions"],
            "val_point": e["rungs"]["L3"]["val"]["_by_name"][member],
            "select_point": {k: v for k, v in point.items() if not k.startswith("val_")},
            "run": run["path"], "provenance": source_record(run["summary_path"])}
    path.with_name(path.stem + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return row3


def _fmt(x, nd=3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _pm(x, se, nd=3) -> str:
    if x is None:
        return "-"
    return f"{x:.{nd}f}" if se is None else f"{x:.{nd}f} ± {se:.{nd}f}"


def _point(honest: dict, level: str) -> str:
    name = honest["by_level"].get(level)
    if name is None:
        return "-"
    p = next(p for p in honest["points"] if p["name"] == name)
    return f"{p['accuracy']:.3f} @ {p['mean_thinking_tokens']:.0f}"


def _row_str(r: dict) -> str:
    return f"{r['accuracy']:.3f} @ {r['mean_thinking_tokens']:.0f}"


def _is_spec_rung(rung: str) -> bool:
    return rung not in DIAGNOSTIC_RUNGS and any(
        rung == p or (p.endswith("_") and rung.startswith(p)) for p in SPEC_RUNG_PREFIXES)


def table(result: dict) -> str:
    lines = ["| rung | seed | members | val save@0pp | val save@1pp | select sweep save@0pp | "
             "select sweep save@1pp | honest select @0pp (acc @ tok) | honest select @1pp | "
             "ΔL1 op dacc_pp ± se | ΔL1 op dsaving_frac ± se | non-dom | beats-agr | pass |",
             "|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|"]
    for rung, entry in result["rungs"].items():
        boot = entry["bootstrap"]
        for seed, per in entry["per_seed"].items():
            per_seed_verdicts = entry["go_no_go"]["per_seed"]
            v = ({"non_dominated": "n/a", "beats_agreement": "n/a", "pass": "n/a"}
                 if per_seed_verdicts is None else per_seed_verdicts[seed])
            op = per["vs_L1_op"]
            lines.append(
                f"| {rung} | {seed} | {entry['members']} | {_fmt(per['val']['saving_at_0pp'])} | "
                f"{_fmt(per['val']['saving_at_1pp'])} | {_fmt(per['select_sweep']['saving_at_0pp'])} | "
                f"{_fmt(per['select_sweep']['saving_at_1pp'])} | {_point(per['select_honest'], '0pp')} | "
                f"{_point(per['select_honest'], '1pp')} | "
                f"{_pm(op['dacc_pp'], boot.get('dacc_pp_se'), 2)} | "
                f"{_pm(op['dsaving_frac'], boot.get('dsaving_frac_se'))} | "
                f"{v['non_dominated']} | {v['beats_agreement']} | {v['pass']} |")
    lines.append("")
    for seed, free in result["free_hull"].items():
        sel = free["select"]
        agr = result["agreement"][seed]["select"]
        free_rows = ", ".join(f"{r['name']} {_row_str(r)}" for r in sel["rows"])
        lines.append(f"seed {seed} select: natural {_row_str(sel['natural'])} | always_fire "
                     f"{_row_str(sel['always_fire'])} | {ld.AGREEMENT} {_row_str(agr)} | free rows: "
                     f"{free_rows} | free hull: "
                     + " ".join(f"({t:.0f}, {a:.3f})" for t, a in sel["hull"]))
    for seed, cal in result["calibration"].items():
        if "skipped" in cal:
            lines.append(f"seed {seed} calibration: not fitted -- {cal['skipped']}")
            continue
        lines.append(f"seed {seed} calibration (fit on val, attached to select): q C={cal['q_c']} "
                     f"v C={cal['v_c']} on {cal['n_questions']} val questions; val-side L2/L3 are "
                     f"{len(cal['crossfit'])}-fold cross-fit inside val")
    for seed, band in result["l7_band"].items():
        lines.append(f"seed {seed} L7 band: " + "; ".join(
            f"{split} mean max(0, r - vpi0)={_fmt(b['band_mean'])} in-band frac={_fmt(b['in_band_frac'])} "
            f"over {b['n_probes']} probes" for split, b in band.items()))
    spec = [r for r in result["rungs"]
            if _is_spec_rung(r) and result["rungs"][r].get("kind") != "external"]
    passing = [r for r in spec if result["rungs"][r]["go_no_go"]["pass"]]
    if spec:
        lines.append(f"go/no-go (§10 rungs L4/L6/L7/B evaluated: {', '.join(spec)}): "
                     f"{', '.join(passing) if passing else 'none pass'}")
    else:
        lines.append("go/no-go (§10 rungs L4/L6/L7/B): none evaluated -- the dumps carry only q,v "
                     "(+ calibrated) beliefs")
    external = result.get("grids", {}).get("external_families", {})
    harness = [r for r in result["rungs"]
               if not _is_spec_rung(r) and r not in DIAGNOSTIC_RUNGS and r not in external
               and result["rungs"][r].get("kind") != "external"]
    lines.append("harness check (L0-L3/L8 are NOT §10 rungs; verdict shown for the estimator only): "
                 f"passing = {[r for r in harness if result['rungs'][r]['go_no_go']['pass']]}")
    if external:
        swept = [r for r in result["rungs"] if r in external]
        lines.append("external families (declared as data, NOT §10 rungs -- each is a ported "
                     "baseline's own sweep; read them on the cross-method hull, not the §10 "
                     "verdict): " + "; ".join(
                         f"{r} n={external[r]['n_members']} on {external[r]['belief_key']!r}"
                         + (f" [{external[r]['belief_source']}]" if external[r]["belief_source"] else "")
                         for r in swept)
                     + (f"; declared but not swept (beliefs absent from the dumps): "
                        f"{sorted(set(external) - set(swept))}" if set(external) - set(swept) else ""))
    b = result["bootstrap"]
    lines.append(f"bootstrap: n={b['n']}; {b['semantics']}; ± se in the ΔL1 columns is the "
                 "seed-mean bootstrap SE (shared by the seed rows); point estimates are per seed")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.calib_folds < 2:
        raise SystemExit("--calib-folds must be >= 2")
    if args.head_tag is not None and not args.export_question_tables:
        raise SystemExit(
            "--head-tag without --export-question-tables: the tag names the head inside the "
            "EXPORTED series names, so there is nothing for it to do here. The ladder JSON's "
            "rungs are this run's own and are never merged across heads -- drop the flag, or "
            "add --export-question-tables")
    check_head_tag(args.head_tag)
    _split_costing_args(args)
    if args.pool:
        from rl_controller.iclr import pareto as pt

        try:
            pt.resolve_pool(args.pool)
        except pt.MergeError as exc:
            raise SystemExit(f"--pool: {exc}") from None
    runs = [load_run(p) for p in args.runs]
    subset: dict = {}
    if args.subset_domain and not args.subset_manifest:
        raise SystemExit("--subset-domain needs --subset-manifest")
    if args.subset_manifest and not args.subset_domain:
        raise SystemExit("--subset-manifest needs --subset-domain")
    kept_questions = None
    applied_manifests: dict[str, list] = {}
    if args.subset_domain:
        domains, subset_records = _manifest_records(args.subset_manifest)
        subset.update(subset_runs(runs, domain_of=domains, domain=args.subset_domain))
        subset["subset_manifests"] = subset_records
        applied_manifests["--subset-manifest"] = subset_records
        kept_questions = {q for q, d in domains.items() if d == args.subset_domain}
    if args.select_manifest:
        keys, select_records = _manifest_records(args.select_manifest)
        restrict_select(runs, keys=set(keys))
        subset["select_manifests"] = select_records
        applied_manifests["--select-manifest"] = select_records
    validation_basis: dict = {}
    if args.validation_reference_out and not args.validation_reference_table:
        raise SystemExit("--validation-reference-out needs --validation-reference-table")
    if args.validation_reference_table:
        runs, validation_basis = project_onto_validation_reference(
            runs, table=args.validation_reference_table,
            out_root=Path(args.out).parent, out_dir=args.validation_reference_out,
            keep_questions=kept_questions)
    check_runs(runs)
    runs.sort(key=lambda r: r["seed"])
    if validation_basis:
        validation_basis["runs"] = [run["validation_basis"] for run in runs]
    specs = load_family_specs(spec_paths=args.family_spec, modules=args.family_module)
    evals = [evaluate_run(run, calib_folds=args.calib_folds, specs=specs,
                          calibrate=not args.no_calibration,
                          only_families=args.only_families,
                          keep_chains=bool(args.export_question_tables)) for run in runs]
    if args.only_families and args.no_calibration:
        for entry in evals:
            entry["calibration"] = {"skipped": "--no-calibration: explicit raw-score family comparison"}
    result = aggregate(evals, n_boot=args.bootstrap, rng=random.Random(args.seed),
                       external=[s.name for s in specs])
    result["runs"] = [{"path": r["path"], "seed": r["seed"], "targets": r["targets"],
                       "n_val_questions": len({c["qkey"] for c in r["val"]}),
                       "n_val_chains": len(r["val"]),
                       "n_select_questions": len({c["qkey"] for c in r["select"]}),
                       "n_select_chains": len(r["select"]), "score_keys": e["score_keys"]}
                      for r, e in zip(runs, evals)]
    result["grids"] = grids(specs)
    result["family_specs"] = [source_record(p) for p in (args.family_spec or ())]
    result["family_modules"] = list(args.family_module or ())
    result["only_families"] = args.only_families
    result["seed"] = args.seed
    result["calib_folds"] = args.calib_folds
    result["select_levels"] = [level_key(l) for l in LEVELS]
    result["provenance"] = [source_record(r["summary_path"]) for r in runs]
    result["subset"] = subset
    result["validation_basis"] = validation_basis
    if args.export_row3:
        result["row3_export"] = {"path": str(args.export_row3),
                                 **export_row3(args.export_row3, evals, runs)}
    if args.export_question_tables:
        split_costing = _split_costing_args(args, load_programs=True)
        if not args.overhead_accounting:
            raise SystemExit(
                "--export-question-tables needs --overhead-accounting: the primary cost axis "
                "charges each method's decision-time generation, and defaulting an undeclared "
                "overhead to 0 would favour exactly the baselines whose probing is expensive "
                "(contract §4.3)")
        if args.overhead_accounting == "unknown":
            raise SystemExit("--overhead-accounting unknown: this method cannot be placed on the "
                             "total-cost axis; report it overhead-unknown instead of exporting")
        if args.overhead_accounting == "per_probe" and not args.overhead_key:
            raise SystemExit("--overhead-accounting per_probe needs --overhead-key")
        if args.overhead_accounting == "constant_per_boundary" \
                and args.overhead_tokens_per_boundary <= 0:
            raise SystemExit("--overhead-accounting constant_per_boundary needs a positive "
                             "--overhead-tokens-per-boundary (use 'none' if it really is 0)")
        overhead = {"overhead_per_boundary": 0.0, "overhead_key": None}
        if args.overhead_accounting == "constant_per_boundary":
            overhead["overhead_per_boundary"] = args.overhead_tokens_per_boundary
        elif args.overhead_accounting == "per_probe":
            overhead["overhead_key"] = args.overhead_key
        if not args.pool:
            raise SystemExit(
                "--export-question-tables needs --pool: the merger asserts every series read "
                "the same chain bank, and a table that does not DECLARE its pool cannot be "
                "merged (it is not agreement, it is silence). Pool identity is the CHAIN BANK "
                "the ladder read -- an inventory pool id such as `select`, `math_ood_report` "
                "or `code_ood_pooled` -- and is never derived from the output path: the "
                "2026-09-02 wet run wrote a mktemp filename into four tables' `pool` field "
                "and lost every honest-hull ABH to it (contract v1.2 §1.2, v1.7 §2.6)")
        block, identity_sources = comparability_block(
            runs, pool=args.pool, answer_format=args.answer_format)
        restriction = restriction_block(args, pool=block["pool"],
                                        applied=applied_manifests)
        result["question_tables"] = export_question_tables(
            args.export_question_tables, runs, evals, scope=args.question_tables_scope,
            costing=split_costing["val"], costing_by_split=split_costing,
            overhead=overhead, pool=block["pool"], comparability=block,
            restriction=restriction, head_tag=args.head_tag,
            provenance=[source_record(r["summary_path"]) for r in runs])
        if args.head_tag:
            result["question_tables"]["head_tag"] = args.head_tag
        result["question_tables"]["restriction"] = restriction
        result["question_tables"]["comparability"] = block
        result["question_tables"]["identity_sources"] = identity_sources
        result["question_tables"]["overhead_accounting"] = args.overhead_accounting
        result["question_tables"]["costing"] = {
            "val": {"answer_format": args.answer_format, "programs": args.programs},
            "select": {"answer_format": args.select_answer_format or args.answer_format,
                       "programs": args.select_programs or args.programs},
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = table(result)
    if args.md:
        md = Path(args.md)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
