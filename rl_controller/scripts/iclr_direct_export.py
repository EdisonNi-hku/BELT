"""Export a direct stop policy dump as policy-replay records."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rl_controller.baselines.ospruner import replay as R
from rl_controller.iclr import osp_policy as osp
from rl_controller.iclr import pareto as pt
from rl_controller.iclr.chains import is_trainable, load_chains, write_jsonl
from rl_controller.iclr.provenance import source_record
from rl_controller.scripts import iclr_ladder as L
from rl_controller.scripts.iclr_osp_replay import (GRADER_ID, MODEL_ID, MODEL_REVISION,
                                                   PROMPT_CONTRACT_ID, SELECTION_RULE,
                                                   SELECTION_SPLIT, natural_answer_fn)

MANIFEST_SCHEMA = "iclr-direct-export-manifest/v1"
EXPORT_SCHEMA = "iclr-direct-export/v1"
MANIFEST_TOP = {"schema_version", "export_commit", "source_branch", "entries"}
CONDITIONED_MODES = ("osp_cond", "osp_aux", "osp_bottleneck")
FIXED_MODES = ("osp",)
V_CONST, REFERENCE_LAMBDA = 0.5, 1e-5
PRICE_REL_TOL, CROSSCHECK_TOL = 1e-12, 1e-6
ENTRY_FIELDS = ("family", "mode", "seed", "pool", "split", "dataset", "training_commit",
                "summary", "checkpoint", "val_scored", "pool_scored",
                "chain_bank_sha256", "manifest_sha256", "cost")
ENTRY_OPTIONAL = ("method_name", "pool_summary", "lambda_columns", "reference_lambda",
                  "fixed_lambda", "split_inputs")
COST_FIELDS = {"answer_format", "force_natural_answer", "grade", "aux_generated_tokens"}
SPLIT_INPUT_FIELDS = {"source", "pool", "chain_bank_sha256", "manifest_sha256"}
SPLIT_SOURCE = {"val": "val_scored"}

class Refused(SystemExit):
    pass

def _hex(value, n: int, label: str) -> str:
    if not isinstance(value, str) or len(value) != n or any(c not in "0123456789abcdef"
                                                            for c in value):
        raise Refused(f"{label}: expected {n} lowercase hex chars, got {value!r}")
    return value

def _close(a, b, tol: float = PRICE_REL_TOL) -> bool:
    return isinstance(a, (int, float)) and not isinstance(a, bool) \
        and math.isclose(float(a), float(b), rel_tol=tol, abs_tol=0.0)

def verified_file(spec, label: str) -> dict:
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        raise Refused(f"{label}: expected exactly path and sha256, got {spec!r}")
    declared = _hex(spec["sha256"], 64, f"{label}.sha256")
    if not Path(spec["path"]).is_file():
        raise Refused(f"{label}: {spec['path']} is not a readable file")
    record = source_record(spec["path"])
    if record["sha256"] != declared:
        raise Refused(f"{label}: {spec['path']} hashes to {record['sha256']}, the manifest "
                      f"declares {declared} -- these are not the verified bytes")
    return record

def check_entry(entry, index: int) -> dict:
    label = f"entries[{index}]"
    if not isinstance(entry, dict):
        raise Refused(f"{label}: must be an object")
    missing = [f for f in ENTRY_FIELDS if f not in entry]
    unknown = [f for f in entry if f not in ENTRY_FIELDS + ENTRY_OPTIONAL]
    if missing or unknown:
        raise Refused(f"{label}: missing {missing}, unknown {unknown}")
    if entry["mode"] not in CONDITIONED_MODES + FIXED_MODES:
        raise Refused(f"{label}: mode {entry['mode']!r} not in "
                      f"{CONDITIONED_MODES + FIXED_MODES}")
    if isinstance(entry["seed"], bool) or not isinstance(entry["seed"], int):
        raise Refused(f"{label}: seed must be an int")
    if entry["split"] == "val":
        raise Refused(f"{label}: split names the POOL split; val comes from val_scored so that "
                      "ONE artifact carries the whole rectangle")
    if pt.INVENTORY_POOL_BANKS.get(str(entry["pool"])) is None:
        raise Refused(f"{label}.pool: {entry['pool']!r} is not an inventory pool id")
    for key, n in (("chain_bank_sha256", 64), ("manifest_sha256", 64), ("training_commit", 40)):
        _hex(entry[key], n, f"{label}.{key}")
    cost = entry["cost"]
    if not isinstance(cost, dict) or set(cost) != COST_FIELDS:
        raise Refused(f"{label}.cost: expected exactly {sorted(COST_FIELDS)}, got {cost!r}")
    if cost["grade"] not in osp.GRADINGS:
        raise Refused(f"{label}.cost.grade: {cost['grade']!r} not in {osp.GRADINGS}")
    natural_answer_fn(cost["answer_format"], cost["force_natural_answer"])
    priced = ("lambda_columns", "reference_lambda") if entry["mode"] in CONDITIONED_MODES \
        else ("fixed_lambda",)
    for name in ("lambda_columns", "reference_lambda", "fixed_lambda"):
        if (name in priced) != (name in entry):
            raise Refused(f"{label}: mode {entry['mode']!r} declares exactly {list(priced)}")
    check_split_inputs(entry, label)
    return entry

def check_split_inputs(entry, label: str) -> dict:
    splits = ("val", entry["split"])
    declared = entry.get("split_inputs")
    if declared is None:
        if "pool_summary" in entry:
            raise Refused(
                f"{label}: this entry's pool dump comes from a separate scoring run, so its two "
                "splits need not come from one bank -- `split_inputs` is REQUIRED. Declare, per "
                f"split {list(splits)}, which file carries it and which bank it came from: "
                '{"val": {"source": "val_scored", "pool": <the in-distribution pool>, '
                '"chain_bank_sha256": ..., "manifest_sha256": ...}, ...}')
        base = {k: entry[k] for k in ("pool", "chain_bank_sha256", "manifest_sha256")}
        return {split: {**base, "source": SPLIT_SOURCE.get(split, "pool_scored")}
                for split in splits}
    if not isinstance(declared, dict) or set(declared) != set(splits):
        raise Refused(f"{label}.split_inputs: expected exactly the splits {sorted(set(splits))}, "
                      f"got {sorted(declared) if isinstance(declared, dict) else declared!r}")
    out = {}
    for split in splits:
        block, at = declared[split], f"{label}.split_inputs[{split!r}]"
        if not isinstance(block, dict) or set(block) != SPLIT_INPUT_FIELDS:
            raise Refused(f"{at}: expected exactly {sorted(SPLIT_INPUT_FIELDS)}, got {block!r}")
        want = SPLIT_SOURCE.get(split, "pool_scored")
        if block["source"] != want:
            raise Refused(f"{at}.source: the {split!r} split is carried by {want!r}, not by "
                          f"{block['source']!r} -- a swapped binding labels one pool's chains "
                          "with the other pool's split and identity")
        if pt.INVENTORY_POOL_BANKS.get(str(block["pool"])) is None:
            raise Refused(f"{at}.pool: {block['pool']!r} is not an inventory pool id")
        for key in ("chain_bank_sha256", "manifest_sha256"):
            _hex(block[key], 64, f"{at}.{key}")
        out[split] = dict(block)
    read = out[entry["split"]]
    if any(read[k] != entry[k] for k in ("pool", "chain_bank_sha256", "manifest_sha256")):
        raise Refused(f"{label}.split_inputs[{entry['split']!r}]: declares "
                      f"{ {k: read[k] for k in ('pool', 'chain_bank_sha256', 'manifest_sha256')} }, "
                      f"the entry declares "
                      f"{ {k: entry[k] for k in ('pool', 'chain_bank_sha256', 'manifest_sha256')} }"
                      " -- the read split IS the artifact's pool; two claims about it is a "
                      "contradiction, not extra provenance")
    val, same_bank = out["val"], all(
        out["val"][k] == read[k] for k in ("chain_bank_sha256", "manifest_sha256"))
    if same_bank and val["pool"] != read["pool"]:
        raise Refused(
            f"{label}.split_inputs: the val split declares pool {val['pool']!r} bank "
            f"{val['chain_bank_sha256'][:12]}.. and the read split pool {read['pool']!r} bank "
            f"{read['chain_bank_sha256'][:12]}.. -- one bank cannot be two pools")
    return out

def bind_prices(entry, summary, label: str) -> list[tuple[int, float]]:
    if entry["mode"] in FIXED_MODES:
        recorded = (summary.get("osp_replay") or {}).get("lambda")
        if not _close(recorded, entry["fixed_lambda"]):
            raise Refused(f"{label}: fixed_lambda {entry['fixed_lambda']!r}, but the run's "
                          f"summary.osp_replay.lambda is {recorded!r}")
        return [(None, float(entry["fixed_lambda"]))]
    declared = entry["lambda_columns"]
    if not isinstance(declared, list) or not declared:
        raise Refused(f"{label}.lambda_columns: expected the recorded price of every column")
    for key in ("val_metrics", "select_metrics"):
        recorded = (summary.get(key) or {}).get("inference_lambdas")
        if not isinstance(recorded, list):
            raise Refused(f"{label}: summary.{key}.inference_lambdas is absent -- the run never "
                          "recorded the prices its columns were scored at")
        if len(recorded) != len(declared) or not all(map(_close, declared, recorded)):
            raise Refused(f"{label}: lambda_columns {declared} do not match the prices the run "
                          f"recorded in summary.{key}.inference_lambdas ({recorded}); a "
                          "reordered or shifted column map prices every config wrongly")
    frozen = list(osp.INFERENCE_LAMBDAS)
    if len(declared) != len(frozen) or not all(map(_close, declared, frozen)):
        raise Refused(f"{label}: the run's inference grid is not the campaign's frozen "
                      f"{len(frozen)}-point grid, so its columns compare with no other cell")
    ref = entry["reference_lambda"]
    if not _close(ref, REFERENCE_LAMBDA):
        raise Refused(f"{label}: reference_lambda {ref!r} is not {REFERENCE_LAMBDA:g}")
    if any(_close(ref, lam) for lam in declared):
        raise Refused(f"{label}: reference_lambda {ref!r} claims to sit ON the inference grid. "
                      "It does not, and `q` must never be read off the nearest column")
    if not any(_close(ref, lam) for lam in (summary.get("lambda_grid") or [])):
        raise Refused(f"{label}: the reference price {ref!r} is not in the run's TRAINING grid "
                      f"{summary.get('lambda_grid')!r}, so `q` is not the price it claims")
    return list(enumerate(float(lam) for lam in declared))

def load_dump(path, label: str) -> list[dict]:
    try:
        return load_chains(path)
    except ValueError as exc:
        raise Refused(f"{label}: {exc}") from None

def check_scores(chains, entry, n_columns: int, label: str) -> None:
    for chain in chains:
        where = f"{label}: chain {(chain['qkey'], chain['rep'])}"
        if [qp[1] for qp in chain["query_points"]] != [p["prefix_len"] for p in chain["probes"]]:
            raise Refused(f"{where}: query_points prefixes do not align with probes[].prefix_len")
        if chain["probes"] and not chain.get("scores"):
            raise Refused(f"{where}: has probes but carries no scores")
        for i, score in enumerate(chain.get("scores") or []):
            at = f"{where} probe {i}"
            if not isinstance(score.get("pi"), (int, float)) or not 0.0 <= score["pi"] <= 1.0:
                raise Refused(f"{at}: pi {score.get('pi')!r} is not a probability")
            if score["q"] != score["pi"] or score["v"] != V_CONST:
                raise Refused(f"{at}: q={score['q']!r} v={score['v']!r} is not the direct "
                              f"encoding (q = pi at the reference price, v = {V_CONST}) -- this "
                              "is not a direct policy dump, or `q` is a belief")
            if not n_columns:
                continue
            column = score.get("pi_by_lambda")
            if not isinstance(column, list) or len(column) != n_columns:
                raise Refused(f"{at}: pi_by_lambda is {column!r}, the manifest binds "
                              f"{n_columns} prices")
            bad = [(j, v) for j, v in enumerate(column) if not 0.0 <= v <= 1.0]
            if bad:
                raise Refused(f"{at}: pi_by_lambda[{bad[0][0]}] = {bad[0][1]!r} is not a "
                              "probability")

def project(chains, column):
    out = []
    for chain in chains:
        scores = chain.get("scores") or []
        if column is not None:
            scores = [dict(s, pi=s["pi_by_lambda"][column]) for s in scores]
        out.append(dict(chain, scores=scores))
    return out

def point(chains, entry, natural_fn) -> dict:
    return osp.expected_point(chains, pi_key="pi", grade=entry["cost"]["grade"],
                              natural_answer_tokens=natural_fn)

def crosscheck(entry, summary, val_chains, natural_fn, label: str) -> dict:
    scored = [c for c in val_chains if entry["mode"] in FIXED_MODES or is_trainable(c)]
    got = point(scored, entry, natural_fn)
    if entry["mode"] in FIXED_MODES:
        theirs = ((summary.get("osp_replay") or {}).get("val") or {}).get("policy") or {}
        pairs = [(k, theirs.get(k), got[k]) for k in ("accuracy", "mean_thinking_tokens")]
    else:
        pairs = [("p_natural", (summary.get("val_metrics") or {}).get("p_natural"),
                  got["mean_p_natural"])]
    for name, recorded, ours in pairs:
        if not isinstance(recorded, (int, float)) or isinstance(recorded, bool):
            raise Refused(f"{label}: the run recorded no {name} to cross-check against")
        if abs(float(recorded) - ours) > CROSSCHECK_TOL * max(1.0, abs(float(recorded))):
            raise Refused(f"{label}: replaying the reference price gives {name} = {ours!r}, the "
                          f"run's own summary says {recorded!r} -- not the same chains")
    return {"price": entry.get("reference_lambda", entry.get("fixed_lambda")), "val_point": got,
            "checked": [n for n, _, _ in pairs], "n_val_chains": len(scored),
            "note": "cross-check only: this price is emitted as NO config"}

VAL_IDENTITY_KEYS = ("chain_bank_sha256", "manifest_sha256")
VAL_FROM_FIT_AND_QUESTIONS = "fit_chains + the run's one provenance.questions manifest"


def validation_source(summary, label: str) -> dict:
    prov = summary.get("provenance") or {}
    resolved = L.resolve_validation_source(prov)
    questions = [q.get("sha256") for q in (prov.get("questions") or [])
                 if isinstance(q, dict) and q.get("sha256")]
    if "val_identity_unresolved" in resolved and len(questions) == 1 \
            and not prov.get("fit_manifest_sha256"):
        resolved = L.resolve_validation_source(dict(prov, fit_manifest_sha256=questions[0]))
        if "val_identity_unresolved" not in resolved:
            resolved = {**resolved,
                        "val_identity_source": VAL_FROM_FIT_AND_QUESTIONS}
    if "val_identity_unresolved" in resolved:
        raise Refused(
            f"{label}: this run does not say which bank its validation rows came from -- "
            f"{resolved['val_identity_unresolved']}. `val_scored.jsonl` is the held-out FIT "
            "split, its bank cannot be read off its chains, and the pool the run SCORED is the "
            "wrong answer, so the export refuses rather than labelling the rows honest "
            "selection is performed on with a bank they did not come from")
    return resolved


def bind_val_split(entry, split_inputs, summary, label: str) -> dict:
    recorded = validation_source(summary, label)
    val = dict(split_inputs["val"])
    if "split_inputs" in entry:
        wrong = {k: (val[k], recorded[k]) for k in VAL_IDENTITY_KEYS if val[k] != recorded[k]}
        if wrong:
            raise Refused(
                f"{label}.split_inputs['val']: binds the validation split to (declared, the "
                f"run's own record) {wrong}. `val_scored.jsonl` is the TRAINING run's held-out "
                f"FIT split (this run's validation source is {recorded['val_identity_source']}); "
                "binding it to the pool the run scored, or to the report pool's bank, is what "
                "labels the rows honest selection is performed on with chains they never came "
                "from")
    val.update({k: recorded[k] for k in VAL_IDENTITY_KEYS})
    val["val_identity_source"] = recorded["val_identity_source"]
    return val

def header_for(entry, *, method, config_id, lam, column, split, identity, source,
               branch, commit) -> dict:
    grade = entry["cost"]["grade"]
    frozen = {"lambda": lam, "lambda_column": column, "head_mode": entry["mode"],
              "training_commit": entry["training_commit"], "replay_grade": grade,
              "reference_lambda": entry.get("reference_lambda"),
              "expectation": "survival_product", "q_is": "stop probability not belief"}
    if any(identity[k] != entry[k] for k in ("pool", "chain_bank_sha256", "manifest_sha256")):
        frozen["split_input"] = {
            "carries_split": split, "from": identity["source"], "pool": identity["pool"],
            "chain_bank_sha256": identity["chain_bank_sha256"],
            "manifest_sha256": identity["manifest_sha256"],
            "note": ("`pool`/`dataset` name the bank this ARTIFACT is filed under; this split's "
                     "chains came from the bank named here")}
    return osp.replay_header(
        run_id=f"{method}-lam{lam:g}-s{entry['seed']}", method_family="two-belief-direct",
        method_name=method, config_id=config_id, knob={"name": "lambda", "value": lam},
        seed=entry["seed"], dataset=entry["dataset"], split=split, model_id=MODEL_ID,
        model_revision=MODEL_REVISION, tokenizer_revision=MODEL_REVISION,
        prompt_contract_id=PROMPT_CONTRACT_ID,
        grader_id=GRADER_ID if grade == "loose" else f"{GRADER_ID}+grade_{grade}",
        selection_split=SELECTION_SPLIT, selection_rule=SELECTION_RULE, frozen_config=frozen,
        chain_bank_sha256=identity["chain_bank_sha256"],
        manifest_sha256=identity["manifest_sha256"],
        pool=pt.resolve_pool(entry["pool"]), source_branch=branch, source_commit=commit,
        source_artifact=source["path"], source_sha256=source["sha256"])

VAL_MEMBERSHIP_UNVERIFIED = (
    "NOT VERIFIED -- no --verify-val-membership was given, so nothing here opened the declared "
    "bank. This artifact's val rows carry the bank the run's own provenance records, checked "
    "against that record; whether that bank CONTAINS these validation questions was not tested. "
    "Do not read this field as a pass.")


def check_val_membership(chains_path, val_chains, identity, label: str, cache: dict) -> dict:
    qkeys = tuple(sorted({c["qkey"] for c in val_chains}))
    key = (str(chains_path), identity["chain_bank_sha256"], qkeys)
    if key not in cache:
        cache[key] = L.verify_validation_bank(
            set(qkeys), chains_path=chains_path,
            declared_sha256=identity["chain_bank_sha256"])
    receipt = cache[key]
    if receipt["failures"]:
        raise Refused(
            f"{label}: --verify-val-membership {chains_path} does not bear out this run's "
            f"validation source {identity['chain_bank_sha256']}: {receipt['failures']}")
    return {"verified": True, "bank_path": str(chains_path),
            "chain_bank_sha256": identity["chain_bank_sha256"],
            "file_sha256": receipt["file_sha256"], "bank_questions": receipt["bank_questions"],
            "val_questions": receipt["val_questions"],
            "val_questions_in_bank": receipt["val_questions_in_bank"],
            "checked": "every validation qkey is in the declared bank, and the bank's bytes "
                       "hash to the digest the run records"}


def bind_pool_summary(entry, label: str) -> dict:
    if "pool_summary" not in entry:
        return {}
    pool_config = json.loads(Path(entry["pool_summary"]["path"]).read_text(
        encoding="utf-8")).get("config") or {}
    if Path(pool_config.get("source_run", "")).resolve() \
            != Path(entry["summary"]["path"]).parent.resolve():
        raise Refused(f"{label}: the scored-pool run says it came from "
                      f"{pool_config.get('source_run')!r}, not from this training run")
    return pool_config

def export_entry(entry, index: int, *, branch: str, commit: str,
                 val_bank=None, membership_cache=None) -> dict:
    label = f"entries[{index}] ({entry['family']} seed {entry['seed']})"
    names = ("summary", "checkpoint", "val_scored", "pool_scored") \
        + (("pool_summary",) if "pool_summary" in entry else ())
    inputs = {n: verified_file(entry[n], f"{label}.{n}") for n in names}
    summary = json.loads(Path(entry["summary"]["path"]).read_text(encoding="utf-8"))
    config = summary.get("config") or {}
    if summary.get("head_mode") != entry["mode"] or config.get("seed") != entry["seed"]:
        raise Refused(f"{label}: the run is head_mode {summary.get('head_mode')!r} seed "
                      f"{config.get('seed')!r}; the manifest says {entry['mode']!r} / "
                      f"{entry['seed']!r}")
    bind_pool_summary(entry, label)
    prices = bind_prices(entry, summary, label)
    split_inputs = check_split_inputs(entry, label)
    split_inputs["val"] = bind_val_split(entry, split_inputs, summary, label)
    natural_fn = natural_answer_fn(entry["cost"]["answer_format"],
                                   entry["cost"]["force_natural_answer"])
    method = entry.get("method_name") or entry["family"]
    splits = {"val": ("val_scored", load_dump(entry["val_scored"]["path"], f"{label} val")),
              entry["split"]: ("pool_scored",
                               load_dump(entry["pool_scored"]["path"], f"{label} pool"))}
    both = {c["qkey"] for c in splits["val"][1]} & {c["qkey"] for c in splits[entry["split"]][1]}
    if both:
        raise Refused(f"{label}: {len(both)} question(s) are in BOTH splits, e.g. "
                      f"{sorted(both)[:3]} -- honest selection would pick on what it reports")
    n_columns = len(prices) if entry["mode"] in CONDITIONED_MODES else 0
    for split, (_, chains) in splits.items():
        if not chains:
            raise Refused(f"{label}: the {split!r} dump has no chains; a config with no val "
                          "rows has no honest operating point at all")
        check_scores(chains, entry, n_columns, f"{label} {split}")
    membership = VAL_MEMBERSHIP_UNVERIFIED if val_bank is None else check_val_membership(
        val_bank, splits["val"][1], split_inputs["val"], label,
        membership_cache if membership_cache is not None else {})
    rows, points = [], {}
    for column, lam in prices:
        config_id = f"{method}:lam{lam:g}"
        for split, (key, chains) in splits.items():
            projected = project(chains, column)
            head = header_for(entry, method=method, config_id=config_id, lam=lam, column=column,
                              split=split, identity=split_inputs[split], source=inputs[key],
                              branch=branch, commit=commit)
            rows.extend(R.replay_rows(projected, header=head, pi_key="pi",
                                      grade=entry["cost"]["grade"],
                                      natural_answer_tokens=natural_fn,
                                      aux_generated_tokens=entry["cost"]["aux_generated_tokens"]))
            points[f"{config_id}|s{entry['seed']}|{split}"] = point(projected, entry, natural_fn)
    return {"rows": rows, "provenance": {
        "label": label, "method_name": method, "inputs": inputs, "cost": entry["cost"],
        **{k: entry[k] for k in ("family", "mode", "seed", "pool", "split", "training_commit")},
        "split_inputs": {split: {**ident, "declared": "split_inputs" in entry,
                                 "source_artifact": inputs[ident["source"]]["path"],
                                 "source_sha256": inputs[ident["source"]]["sha256"]}
                         for split, ident in split_inputs.items()},
        "artifact_identity": {k: entry[k] for k in ("pool", "dataset", "chain_bank_sha256",
                                                    "manifest_sha256")},
        "val_membership": membership,
        "prices": [{"column": c, "lambda": lam} for c, lam in prices], "points": points,
        "crosscheck": crosscheck(entry, summary, splits["val"][1], natural_fn, label)}}

def check_group(entries, family: str, pool: str) -> None:
    identity = {(e["dataset"], e["chain_bank_sha256"], e["manifest_sha256"], e["split"],
                 e.get("method_name") or e["family"], e["mode"],
                 json.dumps(check_split_inputs(e, f"{family}/{pool}"), sort_keys=True))
                for e in entries}
    if len(identity) != 1:
        raise Refused(f"{family}/{pool}: {len(identity)} dataset/bank/split/mode/split-input "
                      f"identities: {sorted(identity)}")

def write_group(rows, provenances, *, family, pool, out_dir: Path, manifest_record) -> dict:
    out = out_dir / f"{family}_{pool}.policy_replay.jsonl"
    write_jsonl(out, sorted(rows, key=lambda r: (r["config_id"], r["seed"], r["split"],
                                                 r["qkey"])))
    try:
        pt.load_policy_replay(out, normalize="chain", seed_semantics="training")
    except pt.MergeError as exc:
        raise Refused(f"{out}: the merger's own loader refuses this artifact: {exc}") from None
    first = provenances[0]
    val_identities = {tuple((p.get("split_inputs") or {}).get("val", {}).get(k)
                            for k in VAL_IDENTITY_KEYS) for p in provenances}
    if len(val_identities) > 1:
        raise Refused(f"{family}/{pool}: the seeds' runs record {len(val_identities)} different "
                      f"validation sources {sorted(val_identities)} -- one artifact's val rows "
                      "cannot come from two banks")
    splits = {split: {k: ident[k] for k in ("source", "pool", "chain_bank_sha256",
                                            "manifest_sha256", "source_artifact",
                                            "source_sha256", "declared")
                      if k in ident}
              for split, ident in (first.get("split_inputs") or {}).items()}
    if "val_identity_source" in (first.get("split_inputs") or {}).get("val", {}):
        splits["val"]["val_identity_source"] = first["split_inputs"]["val"]["val_identity_source"]
    for split, ident in splits.items():
        ident["is_artifact_bank"] = (
            ident["chain_bank_sha256"] == first["artifact_identity"]["chain_bank_sha256"]
            and ident["manifest_sha256"] == first["artifact_identity"]["manifest_sha256"])
    checked = [p.get("val_membership") for p in provenances]
    membership = ({"verified": True,
                   "bank_path": checked[0]["bank_path"],
                   "chain_bank_sha256": checked[0]["chain_bank_sha256"],
                   "file_sha256": checked[0]["file_sha256"],
                   "entries": [{"label": p["label"],
                                "val_questions": p["val_membership"]["val_questions"],
                                "val_questions_in_bank":
                                    p["val_membership"]["val_questions_in_bank"]}
                               for p in provenances],
                   "checked": checked[0]["checked"]}
                  if checked and all(isinstance(c, dict) for c in checked)
                  else VAL_MEMBERSHIP_UNVERIFIED)
    record = {"schema_version": EXPORT_SCHEMA, "replay_schema": osp.REPLAY_SCHEMA,
              "family": family, "pool": pool, "bank": pt.resolve_pool(pool), "n_rows": len(rows),
              "val_membership": membership,
              **({"splits": splits,
                  "selection": {"split": SELECTION_SPLIT, "rule": SELECTION_RULE,
                                "note": ("selection stays on the fixed in-distribution "
                                         "validation set; the report splits are reported, "
                                         "never selected on")}} if splits else {}),
              "expectation": "survival_product", "overhead_accounting": "none", "bootstrap": None,
              "note": ("No bootstrap here: the direct configs share ONE 500-resample question "
                       "bootstrap with the belief tables in the SAME assembly. `q` in the source "
                       "dumps is a stop probability, not a belief, and is emitted as no config."),
              "manifest": manifest_record, "entries": provenances,
              "output": {"path": str(out), "sha256": source_record(out)["sha256"]}}
    out.with_name(out.stem + ".manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True,
                   help=f"{MANIFEST_SCHEMA} json: a finite list of verified direct runs")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--verify-val-membership", default=None, metavar="FIT_CHAINS_JSONL",
                   help="the FIT chain bank the runs declare. Given, every entry's validation "
                        "qkeys must BE in it and the file must hash to the bank the run records "
                        "(iclr_ladder.verify_validation_bank) -- the check that settled the "
                        "2026-09-11 mislabel, since agreeing producers are not evidence. "
                        "Omitted, the artifact says membership was NOT verified.")
    return p.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)
    manifest_record = source_record(args.manifest)
    doc = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema_version") != MANIFEST_SCHEMA:
        raise Refused(f"--manifest: schema_version must be {MANIFEST_SCHEMA!r}")
    if set(doc) != MANIFEST_TOP:
        raise Refused(f"--manifest fields must be {sorted(MANIFEST_TOP)}, got {sorted(doc)}")
    _hex(doc["export_commit"], 40, "--manifest.export_commit")
    if not isinstance(doc["entries"], list) or not doc["entries"]:
        raise Refused("--manifest.entries: a finite, non-empty input list is required")
    val_bank = args.verify_val_membership
    if val_bank is not None and not Path(val_bank).is_file():
        raise Refused(f"--verify-val-membership {val_bank}: not a readable file. The flag names "
                      "the bank to OPEN; a path that is not there cannot be checked, and "
                      "skipping the check because its file is missing is the silent no-op this "
                      "flag exists to avoid. Drop the flag to export with membership declared "
                      "unverified, or give the bank.")
    checked = [check_entry(e, i) for i, e in enumerate(doc["entries"])]
    membership_cache: dict = {}
    done = [export_entry(e, i, branch=doc["source_branch"], commit=doc["export_commit"],
                         val_bank=val_bank, membership_cache=membership_cache)
            for i, e in enumerate(checked)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for family, pool in sorted({(e["family"], e["pool"]) for e in checked}):
        group = [(e, d) for e, d in zip(checked, done)
                 if (e["family"], e["pool"]) == (family, pool)]
        check_group([e for e, _ in group], family, pool)
        written.append(write_group([r for _, d in group for r in d["rows"]],
                                   [d["provenance"] for _, d in group], family=family, pool=pool,
                                   out_dir=out_dir, manifest_record=manifest_record))
    print(json.dumps({"out_dir": str(out_dir),
                      "val_membership": ("verified against " + val_bank if val_bank
                                         else VAL_MEMBERSHIP_UNVERIFIED),
                      "artifacts": [
                          {**{k: r[k] for k in ("family", "pool", "n_rows")}, **r["output"]}
                          for r in written]}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
