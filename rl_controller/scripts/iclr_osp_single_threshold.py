"""Threshold family for a single OS-Pruner checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rl_controller.baselines.ospruner import replay as R
from rl_controller.iclr import osp_policy as osp
from rl_controller.iclr import pareto as pt
from rl_controller.iclr.provenance import source_record
from rl_controller.scripts import iclr_direct_export as dx
from rl_controller.scripts import iclr_osp_export as osx
from rl_controller.scripts import iclr_single_checkpoint_panel as scp

MANIFEST_SCHEMA = "iclr-osp-threshold-manifest/v1"
RECEIPT_SCHEMA = "iclr-osp-single-threshold-receipt/v1"
MANIFEST_TOP = {"schema_version", "export_commit", "source_branch", "single_checkpoint",
                "entries"}
FAMILY = "OSP-single-threshold"
SOURCE_ARM = "OSP"
RESERVED_FAMILY_NAMES = ("OSP", "O-ALIGN", "OSP-pi-threshold", "OSP-bernoulli",
                         scp.SINGLE_THRESHOLD_FAMILY, scp.MULTI_THRESHOLD_FAMILY,
                         "D-FIX", "D-PERL")
FAMILY_NAME_TOKEN = "threshold"
SINGLE_CHECKPOINT_PRICE = scp.SINGLE_CHECKPOINT_PRICE
THRESHOLD_SOURCE_SPLIT = "val"
OSP_V_CONST = R.STRENGTHENED_V
CROSSCHECK_TOL = dx.CROSSCHECK_TOL
CROSSCHECK_KEYS = ("accuracy", "mean_thinking_tokens", "mean_generated_tokens", "n_questions")
KNOB_THRESHOLD = scp.KNOB_THRESHOLD
READ_SPLIT_LABEL = "select"
OSP_ENTRY_EXTRA = ("reference_export",)


class Refused(SystemExit):
    pass


def check_family_name(name) -> str:
    if not isinstance(name, str) or not name.strip():
        raise Refused(f"family name {name!r}: a non-empty string is required")
    folded = "".join(c for c in name.strip().casefold() if c.isalnum())
    for reserved in RESERVED_FAMILY_NAMES:
        if folded == "".join(c for c in reserved.casefold() if c.isalnum()):
            raise Refused(
                f"family name {name!r} collides with {reserved!r}. {reserved!r} already names a "
                "different claim on the same tables -- a multi-training envelope, or another "
                "reader's sweep -- and these rows are shaped identically to those, so a reader "
                "handed this name would spend one arm's numbers on another arm's claim.")
    if FAMILY_NAME_TOKEN not in name.casefold():
        raise Refused(
            f"family name {name!r} does not contain {FAMILY_NAME_TOKEN!r}. Every member of this "
            "family is a THRESHOLD on one set of weights; a name that does not say so reads as "
            "an envelope of trainings, which is exactly the count this family may not lend.")
    return name


check_family_name(FAMILY)
assert scp.SINGLE_CHECKPOINT_PRICE == dx.REFERENCE_LAMBDA == SINGLE_CHECKPOINT_PRICE, (
    "SINGLE_CHECKPOINT_PRICE must stay the campaign's REFERENCE_LAMBDA and the panel's "
    f"SINGLE_CHECKPOINT_PRICE: {scp.SINGLE_CHECKPOINT_PRICE!r} / {dx.REFERENCE_LAMBDA!r}")


def check_manifest(doc, label: str) -> dict:
    if not isinstance(doc, dict) or doc.get("schema_version") != MANIFEST_SCHEMA:
        raise Refused(f"{label}: schema_version must be {MANIFEST_SCHEMA!r}, got "
                      f"{doc.get('schema_version') if isinstance(doc, dict) else doc!r}")
    if set(doc) != MANIFEST_TOP:
        raise Refused(f"{label}: fields must be {sorted(MANIFEST_TOP)}, got {sorted(doc)}")
    dx._hex(doc["export_commit"], 40, f"{label}.export_commit")
    if not isinstance(doc["entries"], list) or not doc["entries"]:
        raise Refused(f"{label}.entries: a finite, non-empty input list is required")
    block = doc["single_checkpoint"]
    if not isinstance(block, dict) or set(block) != {"price", "by_seed"}:
        raise Refused(f"{label}.single_checkpoint: expected exactly ['by_seed', 'price'], got "
                      f"{sorted(block) if isinstance(block, dict) else block!r}")
    if not dx._close(block["price"], SINGLE_CHECKPOINT_PRICE):
        raise Refused(
            f"{label}.single_checkpoint.price is {block['price']!r}, not "
            f"{SINGLE_CHECKPOINT_PRICE:g}. The single-checkpoint arm is the pre-registered "
            "reference price and nothing else; picking a price by how the sweep comes out is "
            "the selection this arm exists to avoid making.")
    by_seed = block["by_seed"]
    if not isinstance(by_seed, dict) or not by_seed:
        raise Refused(f"{label}.single_checkpoint.by_seed: one 64-hex checkpoint digest per "
                      "seed is required; without it nothing ties the pools to one set of weights")
    return {str(seed): dx._hex(digest, 64, f"{label}.single_checkpoint.by_seed[{seed!r}]")
            for seed, digest in by_seed.items()}


def check_read_split_label(entry, label: str) -> None:
    declared, pool = entry["split"], entry["pool"]
    if declared == READ_SPLIT_LABEL:
        return
    sanctioned = osx.sanctioned_split_labels(pool)
    raise Refused(
        f"{label}.split is {declared!r}. Every artifact this producer writes carries the read "
        f"label {READ_SPLIT_LABEL!r}"
        + (f" -- {declared!r} is the other sanctioned label for pool {pool!r} "
           f"({sorted(sanctioned)}), not a third one, which is exactly what makes it dangerous: "
           "every other arm on this pool (the belief question tables, the nine direct records, "
           "`D-FIX-single-threshold`, the O-ALIGN lanes) declares `select`, `read_split` returns "
           "the pool id as soon as any row declares it, and the merger keys series by the "
           "declared split. This arm would be reported ABSENT on this pool -- a missing arm that "
           "reads as a null result, not as an error."
           if declared in sanctioned else
           f" -- {declared!r} is not even a sanctioned label for pool {pool!r} "
           f"({sorted(sanctioned)}), so these rows would be filed on a split no consumer reads.")
        + " The pool identity is unaffected either way: `pool`, `dataset` and the two digests "
          "stay this pool's. Fix the LABEL in the manifest; never relabel at read time.")

def check_osp_entry(entry, index: int) -> dict:
    label = f"entries[{index}]"
    if not isinstance(entry, dict):
        raise Refused(f"{label}: must be an object")
    missing = [f for f in OSP_ENTRY_EXTRA if f not in entry]
    if missing:
        raise Refused(
            f"{label}: missing {missing}. `reference_export` names the lane's ALREADY published "
            "policy-replay sidecar for this pool; the unthresholded replay is checked against it "
            "before any threshold is applied, and without it the 13 members would be emitted off "
            "a policy whose own published numbers were never reproduced.")
    core = {k: v for k, v in entry.items() if k not in OSP_ENTRY_EXTRA}
    checked = dx.check_entry(core, index)
    if checked["mode"] not in dx.FIXED_MODES:
        raise Refused(f"{label}: mode {checked['mode']!r} is not one of {dx.FIXED_MODES}. An "
                      "O-ALIGN lane is a FIXED-price stop head; a conditioned head's members "
                      "are inference prices and are not thresholds of anything.")
    if checked["family"] != SOURCE_ARM:
        raise Refused(
            f"{label}.family is {checked['family']!r}, not {SOURCE_ARM!r}. This producer emits "
            f"under {FAMILY!r}; sweeping another arm's lane here would file that arm's reader "
            "under an OSP-named family, and the rows carry nothing else that says whose reader "
            "it was.")
    check_read_split_label(checked, label)
    if not dx._close(checked["fixed_lambda"], SINGLE_CHECKPOINT_PRICE):
        raise Refused(
            f"{label}.fixed_lambda is {checked['fixed_lambda']!r}, not "
            f"{SINGLE_CHECKPOINT_PRICE:g}. The single-checkpoint arm is the pre-registered "
            "reference-price lane, fixed for every budget and pool; a source-val argmax over "
            "the five trained prices would be a selection and is deliberately not performed.")
    return {**checked, **{k: entry[k] for k in OSP_ENTRY_EXTRA}}


def bind_osp_price(entry, summary, label: str) -> float:
    config = summary.get("config") or {}
    knob = config.get("knob") if isinstance(config.get("knob"), dict) else {}
    recorded = [("config.osp_lambda", config.get("osp_lambda")),
                ("config.knob.value", knob.get("value") if knob.get("name") == "lambda" else None)]
    bad = [f"{name}={value!r}" for name, value in recorded
           if not dx._close(value, entry["fixed_lambda"])]
    if bad:
        raise Refused(f"{label}: the manifest declares fixed_lambda "
                      f"{entry['fixed_lambda']!r}; the run's own summary says {', '.join(bad)}. "
                      "A price read off the wrong record prices the whole family wrongly and "
                      "nothing downstream can see it.")
    return float(entry["fixed_lambda"])


def check_osp_head(entry, summary, label: str) -> None:
    config = summary.get("config") or {}
    if config.get("head_mode") != entry["mode"] or config.get("seed") != entry["seed"]:
        raise Refused(f"{label}: the run is config.head_mode {config.get('head_mode')!r} seed "
                      f"{config.get('seed')!r}; the manifest says {entry['mode']!r} / "
                      f"{entry['seed']!r}")
    if config.get("targets") != ["q", "v"]:
        raise Refused(f"{label}: the run trained targets {config.get('targets')!r}. The OSP "
                      "survival-product replay reads `pi` from a q,v stop head; a head with "
                      "other targets writes something else under the same keys.")


def check_osp_scores(chains, label: str) -> None:
    osx._require_osp_scores(chains, label=label)
    for chain in chains:
        where = f"{label}: chain {(chain.get('qkey'), chain.get('rep'))}"
        if [qp[1] for qp in chain["query_points"]] != [p["prefix_len"] for p in chain["probes"]]:
            raise Refused(f"{where}: query_points prefixes do not align with probes[].prefix_len")
        for i, score in enumerate(chain.get("scores") or []):
            at = f"{where} probe {i}"
            pi = score.get("pi")
            if isinstance(pi, bool) or not isinstance(pi, (int, float)) or not 0.0 <= pi <= 1.0:
                raise Refused(f"{at}: pi {pi!r} is not a probability")
            if score.get("v") != OSP_V_CONST:
                raise Refused(f"{at}: v={score.get('v')!r}, not the O-ALIGN placeholder "
                              f"{OSP_V_CONST!r}. A stop head has no separate natural-end belief "
                              "to put there; a dump that carries one is a two-belief head's, "
                              "read by iclr_ladder and never by a survival-product replay.")


def bind_scoring_checkpoint(entry, pool_config, label: str) -> dict:
    if not pool_config:
        return {"kind": "lane_own_dump", "checkpoint_sha256": entry["checkpoint"]["sha256"],
                "note": ("the training pool's dump is the lane's own file; there is no second "
                         "scoring run to bind")}
    run_dir = Path(entry["pool_summary"]["path"]).parent
    osx._require_report_pool_scores_are_a_stop_probability(pool_config, run_dir=run_dir)
    provenance = json.loads(Path(entry["pool_summary"]["path"]).read_text(
        encoding="utf-8")).get("provenance") or {}
    scored_with = (provenance.get("osp_checkpoint") or {}).get("sha256")
    if scored_with != entry["checkpoint"]["sha256"]:
        raise Refused(
            f"{label}: the scoring run at {run_dir} says it scored this pool with checkpoint "
            f"{scored_with!r}; the manifest declares {entry['checkpoint']['sha256']!r}. A lane "
            "directory holds more than one checkpoint, so naming the run is not naming the "
            "weights -- and a family swept over four pools from four different epochs of one "
            "lane is not a single deployed checkpoint.")
    return {"kind": "score_pool_run", "checkpoint_sha256": scored_with,
            "run_dir": str(run_dir),
            "checkpoint_path": (provenance.get("osp_checkpoint") or {}).get("path"),
            "checkpoint_epoch": provenance.get("osp_checkpoint_epoch")}


def frozen_thresholds(splits, split_inputs, *, read_split: str, label: str) -> list[float]:
    if THRESHOLD_SOURCE_SPLIT not in splits:
        raise Refused(f"{label}: no {THRESHOLD_SOURCE_SPLIT!r} split to read thresholds off. "
                      "They are frozen on source-val and replayed unchanged; there is no "
                      "fallback that is not an estimate on the split being reported.")
    val_id, read_id = split_inputs[THRESHOLD_SOURCE_SPLIT], split_inputs[read_split]
    if all(val_id[k] == read_id[k] for k in dx.VAL_IDENTITY_KEYS):
        raise Refused(
            f"{label}: the split the thresholds are read off and the split they are reported on "
            f"declare the SAME chain bank {val_id['chain_bank_sha256'][:12]}.. / manifest "
            f"{val_id['manifest_sha256'][:12]}... Then the 13 quantiles are quantiles of the "
            "report itself: the operating points are chosen on the numbers they are then "
            "published as, and nothing in the emitted rows would show it.")
    return scp.source_val_thresholds(splits[THRESHOLD_SOURCE_SPLIT][1])


def _compare(label: str, name: str, ours, theirs, problems: list) -> None:
    if not isinstance(theirs, (int, float)) or isinstance(theirs, bool):
        problems.append(f"{name}: the reference export records {theirs!r}")
        return
    if not math.isclose(float(ours), float(theirs), rel_tol=CROSSCHECK_TOL, abs_tol=0.0):
        problems.append(f"{name}: replay {ours!r} vs published {float(theirs)!r}")


def crosscheck_original_decisions(entry, splits, *, natural_fn, label: str) -> dict:
    record = dx.verified_file(entry["reference_export"], f"{label}.reference_export")
    published = json.loads(Path(entry["reference_export"]["path"]).read_text(encoding="utf-8"))
    if published.get("schema_version") != osp.REPLAY_SCHEMA:
        raise Refused(f"{label}.reference_export: declares "
                      f"{published.get('schema_version')!r}, not {osp.REPLAY_SCHEMA!r}")
    declared = {
        "seed": (published.get("seed"), entry["seed"]),
        "grading": (published.get("grading"), entry["cost"]["grade"]),
        "pi_key": (published.get("pi_key"), "pi"),
        "expectation": (published.get("expectation"), "survival_product"),
        "answer_format": (published.get("answer_format"), entry["cost"]["answer_format"]),
        "force_natural_answer": (bool(published.get("force_natural_answer")),
                                 bool(entry["cost"]["force_natural_answer"])),
        "aux_generated_tokens": (published.get("aux_generated_tokens"),
                                 entry["cost"]["aux_generated_tokens"]),
        "pool": (pt.resolve_pool(str(published.get("pool"))), pt.resolve_pool(entry["pool"])),
        "chain_bank_sha256": (published.get("chain_bank_sha256"), entry["chain_bank_sha256"]),
        "manifest_sha256": (published.get("manifest_sha256"), entry["manifest_sha256"]),
    }
    wrong = {k: v for k, v in declared.items() if v[0] != v[1]}
    if wrong:
        raise Refused(f"{label}.reference_export: (published, this entry) disagree on {wrong}. "
                      "A sidecar for a different seed, pool or grading is not a cross-check of "
                      "this one; it would simply never agree, and the refusal would name the "
                      "statistic instead of the mismatch.")
    for name, key in (("pool_scored", "inputs"), ("val_scored", "val_inputs")):
        digests = {i.get("sha256") for i in (published.get(key) or []) if isinstance(i, dict)}
        if entry[name]["sha256"] not in digests:
            raise Refused(
                f"{label}.reference_export: its {key} are {sorted(digests)}, which do not "
                f"include this entry's {name} {entry[name]['sha256']}. The published numbers "
                "were computed from other bytes, so reproducing them would prove nothing about "
                "the dump this family is swept on.")
    header = {k: published[k] for k in osp.POLICY_REPLAY_HEADER_FIELDS}
    problems, observed = [], {}
    for split, (_key, chains) in splits.items():
        is_val = split == THRESHOLD_SOURCE_SPLIT
        want_q = published.get("val_summary" if is_val else "summary")
        want_c = published.get("val_summary_chain_weighted" if is_val
                               else "summary_chain_weighted")
        if not isinstance(want_q, dict) or not isinstance(want_c, dict):
            raise Refused(
                f"{label}.reference_export: publishes no usable {'val ' if is_val else ''}"
                "summary (a partial frontier point is recorded as null). There is nothing to "
                "reproduce, so the original decisions on this split are UNVERIFIED and the "
                "sweep refuses rather than assuming them.")
        rows = R.replay_rows(chains, header={**header, "split": split}, pi_key="pi",
                             grade=entry["cost"]["grade"], natural_answer_tokens=natural_fn,
                             aux_generated_tokens=entry["cost"]["aux_generated_tokens"])
        got_q, got_c = R.question_summary(rows), osp.replay_summary(rows)
        for key in CROSSCHECK_KEYS:
            _compare(label, f"{split}.question.{key}", got_q[key], want_q.get(key), problems)
            _compare(label, f"{split}.chain.{key}", got_c[key], want_c.get(key), problems)
        observed[split] = {"question": got_q, "chain_weighted": got_c, "n_rows": len(rows)}
        if not is_val:
            natural = osp.natural_summary(chains)
            for key in ("accuracy", "mean_thinking_tokens", "n"):
                _compare(label, f"natural.{key}", natural[key],
                         (published.get("natural") or {}).get(key), problems)
            observed["natural"] = natural
    if problems:
        raise Refused(
            f"{label}: replaying the lane's OWN unthresholded pi from these dumps does not "
            f"reproduce what it already published -- {problems}. The thresholded members are "
            "this same replay with pi frozen to 0/1, so a zeroth operating point that cannot be "
            "reproduced makes all 13 unreadable. What is missing is a dump (or a sidecar) whose "
            "bytes the two agree on; nothing is emitted until they do.")
    return {"reference_export": record, "checked": list(CROSSCHECK_KEYS),
            "splits": sorted(splits), "tolerance": CROSSCHECK_TOL, "observed": observed,
            "note": ("the unthresholded replay reproduces the lane's published question-basis "
                     "and chain-weighted summaries on both splits, and its natural row")}


def threshold_header(entry, *, family, config_id, level, threshold, price, split, identity,
                     source, branch, commit) -> dict:
    grade = entry["cost"]["grade"]
    return osp.replay_header(
        run_id=f"{family}-vq{level:.6g}-s{entry['seed']}", method_family="ospruner",
        method_name=family, config_id=config_id,
        knob={"name": KNOB_THRESHOLD, "value": level}, seed=entry["seed"],
        dataset=entry["dataset"], split=split, model_id=dx.MODEL_ID,
        model_revision=dx.MODEL_REVISION, tokenizer_revision=dx.MODEL_REVISION,
        prompt_contract_id=dx.PROMPT_CONTRACT_ID,
        grader_id=dx.GRADER_ID if grade == "loose" else f"{dx.GRADER_ID}+grade_{grade}",
        selection_split=dx.SELECTION_SPLIT, selection_rule=dx.SELECTION_RULE,
        frozen_config={"training_lambda": price, "val_quantile": level,
                       "pi_threshold": threshold, "threshold_source_split": THRESHOLD_SOURCE_SPLIT,
                       "rule": "stop at the first probe with pi >= pi_threshold",
                       "head_mode": entry["mode"], "training_commit": entry["training_commit"],
                       "replay_grade": grade, "expectation": "survival_product",
                       "q_is": "stop probability not belief",
                       "checkpoint_sha256": entry["checkpoint"]["sha256"],
                       "deployment": ("ONE checkpoint, 13 operating points; the checkpoint is "
                                      "fixed across every budget and every pool"),
                       "overhead_accounting": "none",
                       **({"split_input": {
                           "carries_split": split, "from": identity["source"],
                           "pool": identity["pool"],
                           "chain_bank_sha256": identity["chain_bank_sha256"],
                           "manifest_sha256": identity["manifest_sha256"],
                           "note": ("`pool`/`dataset` name the bank this ARTIFACT is filed "
                                    "under; this split's chains came from the bank named "
                                    "here")}}
                          if any(identity[k] != entry[k] for k in
                                 ("pool", "chain_bank_sha256", "manifest_sha256")) else {})},
        chain_bank_sha256=identity["chain_bank_sha256"],
        manifest_sha256=identity["manifest_sha256"],
        pool=pt.resolve_pool(entry["pool"]), source_branch=branch, source_commit=commit,
        source_artifact=source["path"], source_sha256=source["sha256"])


def sweep_entry(entry, index: int, *, family: str, checkpoints: dict, branch: str,
                commit: str) -> dict:
    label = f"entries[{index}] ({entry['family']} seed {entry['seed']}) {family}"
    names = ("summary", "checkpoint", "val_scored", "pool_scored") \
        + (("pool_summary",) if "pool_summary" in entry else ())
    inputs = {n: dx.verified_file(entry[n], f"{label}.{n}") for n in names}
    declared = checkpoints.get(str(entry["seed"]))
    if declared != entry["checkpoint"]["sha256"]:
        raise Refused(
            f"{label}: the manifest's single_checkpoint.by_seed[{str(entry['seed'])!r}] is "
            f"{declared!r} and this entry's checkpoint is {entry['checkpoint']['sha256']!r}. "
            "The arm is ONE set of weights per seed, fixed for every budget and every pool; an "
            "entry on other weights makes the pools incomparable and the claim untrue.")
    summary = json.loads(Path(entry["summary"]["path"]).read_text(encoding="utf-8"))
    check_osp_head(entry, summary, label)
    price = bind_osp_price(entry, summary, label)
    scoring = bind_scoring_checkpoint(entry, dx.bind_pool_summary(entry, label), label)
    split_inputs = dx.check_split_inputs(entry, label)
    split_inputs["val"] = dx.bind_val_split(entry, split_inputs, summary, label)
    natural_fn = dx.natural_answer_fn(entry["cost"]["answer_format"],
                                      entry["cost"]["force_natural_answer"])
    splits = {"val": ("val_scored", dx.load_dump(entry["val_scored"]["path"], f"{label} val")),
              entry["split"]: ("pool_scored",
                               dx.load_dump(entry["pool_scored"]["path"], f"{label} pool"))}
    both = {c["qkey"] for c in splits["val"][1]} & {c["qkey"] for c in splits[entry["split"]][1]}
    if both:
        raise Refused(f"{label}: {len(both)} question(s) are in BOTH splits, e.g. "
                      f"{sorted(both)[:3]} -- the thresholds come from val and are READ on the "
                      "pool, so an overlap tunes the threshold on what it reports")
    for split, (_key, chains) in splits.items():
        if not chains:
            raise Refused(f"{label}: the {split!r} dump has no chains")
        check_osp_scores(chains, f"{label} {split}")
    crosscheck = crosscheck_original_decisions(entry, splits, natural_fn=natural_fn, label=label)
    thresholds = frozen_thresholds(splits, split_inputs, read_split=entry["split"], label=label)
    rows, points, members = [], {}, []
    for index_, (level, tau) in enumerate(zip(scp.THRESHOLD_LEVELS, thresholds)):
        config_id = f"{family}:vq{index_:02d}"
        members.append({"config_id": config_id, "val_quantile": level, "pi_threshold": tau,
                        "training_lambda": price,
                        "checkpoint_sha256": entry["checkpoint"]["sha256"]})
        for split, (key, chains) in splits.items():
            fired = scp.fire_at(chains, tau)
            head = threshold_header(entry, family=family, config_id=config_id, level=level,
                                    threshold=tau, price=price, split=split,
                                    identity=split_inputs[split], source=inputs[key],
                                    branch=branch, commit=commit)
            rows.extend(R.replay_rows(fired, header=head, pi_key="pi",
                                      grade=entry["cost"]["grade"],
                                      natural_answer_tokens=natural_fn,
                                      aux_generated_tokens=entry["cost"]["aux_generated_tokens"]))
            points[f"{config_id}|s{entry['seed']}|{split}"] = osp.expected_point(
                fired, pi_key="pi", grade=entry["cost"]["grade"], natural_answer_tokens=natural_fn)
    return {"rows": rows, "provenance": {
        "label": label, "method_name": family, "inputs": inputs, "cost": entry["cost"],
        **{k: entry[k] for k in ("family", "mode", "seed", "pool", "split", "training_commit")},
        "split_inputs": {split: {**ident, "declared": "split_inputs" in entry,
                                 "source_artifact": inputs[ident["source"]]["path"],
                                 "source_sha256": inputs[ident["source"]]["sha256"]}
                         for split, ident in split_inputs.items()},
        "artifact_identity": {k: entry[k] for k in ("pool", "dataset", "chain_bank_sha256",
                                                    "manifest_sha256")},
        "single_checkpoint": {"price": price, "checkpoint": inputs["checkpoint"],
                              "scored_with": scoring,
                              "note": ("the pre-registered reference-price lane; the same "
                                       "weights serve every budget and every pool")},
        "threshold_source_split": THRESHOLD_SOURCE_SPLIT,
        "threshold_levels": list(scp.THRESHOLD_LEVELS), "members": members, "points": points,
        "crosscheck": crosscheck}}


def sweep(manifest_path, out_dir) -> list[dict]:
    family = check_family_name(FAMILY)
    record = source_record(manifest_path)
    doc = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    checkpoints = check_manifest(doc, "--manifest")
    checked = [check_osp_entry(e, i) for i, e in enumerate(doc["entries"])]
    seen = {}
    for entry in checked:
        cell = (entry["pool"], entry["seed"])
        if cell in seen:
            raise Refused(f"--manifest: two entries for pool {entry['pool']!r} seed "
                          f"{entry['seed']!r}. One checkpoint per seed per pool is the arm; a "
                          "repeated cell would emit two series under one member id.")
        seen[cell] = entry
    done = [(e, sweep_entry(e, i, family=family, checkpoints=checkpoints,
                            branch=doc["source_branch"], commit=doc["export_commit"]))
            for i, e in enumerate(checked)]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for pool in sorted({e["pool"] for e, _ in done}):
        group = [(e, d) for e, d in done if e["pool"] == pool]
        dx.check_group([e for e, _ in group], family, pool)
        written.append(dx.write_group([r for _, d in group for r in d["rows"]],
                                      [d["provenance"] for _, d in group], family=family,
                                      pool=pool, out_dir=out, manifest_record=record))
    return written


def receipt(written: list[dict], manifest_path) -> dict:
    return {
        "schema": RECEIPT_SCHEMA, "family": FAMILY, "source_arm": SOURCE_ARM,
        "single_checkpoint_price": SINGLE_CHECKPOINT_PRICE,
        "threshold_source_split": THRESHOLD_SOURCE_SPLIT,
        "threshold_levels": list(scp.THRESHOLD_LEVELS),
        "selection": ("NONE. The price is the campaign's pre-registered REFERENCE_LAMBDA and "
                      "the panel's SINGLE_CHECKPOINT_PRICE, a constant predating this question; "
                      "a source-val argmax over the five trained prices is not offered."),
        "manifest": source_record(manifest_path),
        "pools": [{"pool": record["pool"], "bank": record["bank"], "n_rows": record["n_rows"],
                   "output": record["output"],
                   "seeds": [{"seed": entry["seed"],
                              "checkpoint_sha256":
                                  entry["single_checkpoint"]["checkpoint"]["sha256"],
                              "scored_with": entry["single_checkpoint"]["scored_with"],
                              "pi_thresholds": [m["pi_threshold"] for m in entry["members"]],
                              "crosscheck": entry["crosscheck"]["note"]}
                             for entry in record["entries"]]}
                  for record in written],
    }


def render(account: dict) -> str:
    lines = [f"### {account['family']}: 13 source-val quantiles, one checkpoint per seed", ""]
    for pool in account["pools"]:
        seeds = pool["seeds"]
        lines += [f"**{pool['pool']}** ({pool['n_rows']} rows) "
                  f"-> `{Path(pool['output']['path']).name}`", "",
                  "| level | " + " | ".join(f"s{s['seed']} pi cutoff" for s in seeds) + " |",
                  "|" + "---|" * (len(seeds) + 1)]
        for index, level in enumerate(account["threshold_levels"]):
            lines.append(f"| vq{index:02d} ({level:.4f}) | "
                         + " | ".join(f"{s['pi_thresholds'][index]:.6g}" for s in seeds) + " |")
        lines += ["", "checkpoints: "
                  + ", ".join(f"s{s['seed']}={s['checkpoint_sha256'][:12]}.." for s in seeds), ""]
    lines.append(account["selection"])
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True,
                   help=f"{MANIFEST_SCHEMA} json: the verified O-ALIGN lam1e-5 lanes for ONE pool")
    p.add_argument("--out-dir", required=True,
                   help="directory the policy-replay artifacts are written to")
    p.add_argument("--receipt", default=None, help="where the frozen-threshold account is written")
    p.add_argument("--md", default=None, help="the frozen-threshold table")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    written = sweep(args.manifest, args.out_dir)
    account = receipt(written, args.manifest)
    text = render(account)
    for path, payload in ((args.receipt, json.dumps(account, indent=2, sort_keys=True)),
                          (args.md, text)):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(payload + "\n", encoding="utf-8")
    print(text)
    print(json.dumps({"family": FAMILY, "artifacts": [r["output"]["path"] for r in written],
                      "n_rows": sum(r["n_rows"] for r in written)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
