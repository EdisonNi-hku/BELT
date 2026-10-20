"""Export OS-Pruner outputs as policy-replay records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from rl_controller.baselines.ospruner import replay as R
from rl_controller.baselines.ospruner import training as T
from rl_controller.baselines.ospruner import layout as L
from rl_controller.iclr import learning_curve as lc
from rl_controller.iclr import osp_policy as osp
from rl_controller.iclr import pareto
from rl_controller.iclr import read_splits as rs
from rl_controller.iclr import scoring_io as sio
from rl_controller.iclr.chains import is_trainable, load_chains
from rl_controller.iclr.provenance import parse_and_record, source_record
from rl_controller.scripts.iclr_osp_replay import (GRADER_ID, MODEL_ID, MODEL_REVISION,
                                                   PROMPT_CONTRACT_ID, SELECTION_RULE,
                                                   SELECTION_SPLIT, load_question_set,
                                                   natural_answer_fn)


EXPORT_MARKER_SCHEMA = "ws-f-osp-export-complete/v2"
QUESTION_SETS_SCHEMA = "ws-f-osp-question-sets/v1"
REPORT_QUESTION_SET_SCHEMA = "ws-f-osp-report-question-set/v2"
REGISTERED_READ_SPLIT_POPULATION = (
    "registered read split, materialized from the pool's read-split question manifest")
LANE_SOURCE_NAMES = ("summary.json", "val_scored.jsonl", "select_scored.jsonl")
LANE_COMPLETION_MARKER = ".complete.sha256"
LANE_IDENTITY_SCHEMA = "iclr-osp-lane-identity/v1"
TRAINING_POPULATION_SCHEMA = "ws-f-training-population/v1"
TRAINING_CONFIG_SCHEMA = "ws-f-osp-training-config/v1"
TRAINING_ARGS_DIGEST_SCHEMA = "ws-f-osp-training-args-digest/v1"

LEGACY_OSP_COST = "thinking"


def _branch_name(value: str) -> str:
    value = str(value)
    components = value.split("/")
    forbidden = ("..", "@{", "\\", " ", "~", "^", ":", "?", "*", "[")
    if (not value or value == "@" or value.startswith("-") or value.endswith((".", "/"))
            or any(not part or part.startswith(".") or part.endswith(".lock")
                   for part in components)
            or any(token in value for token in forbidden)
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise argparse.ArgumentTypeError(f"invalid source branch {value!r}")
    return value


def export_manifest_path(records) -> Path:
    records = Path(records)
    return records.with_name(records.stem + ".manifest.json")


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_write(path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False,
                                         prefix=path.name + ".", suffix=".tmp")
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _jsonl_bytes(rows) -> bytes:
    return b"".join(json.dumps(row, separators=(",", ":"), allow_nan=False).encode() + b"\n"
                    for row in rows)


def _qset_binding(mapping: dict[str, dict]) -> str | None:
    if not mapping:
        return None
    digest = hashlib.sha256()
    for split, record in sorted(mapping.items()):
        digest.update(f"{split}:{record['sha256']}\n".encode())
    return digest.hexdigest()


def _file_binding(path, *, label: str) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing {label} {path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _source_core(record, *, label: str) -> dict[str, str]:
    if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
        raise ValueError(f"lane summary provenance has no complete {label} source record")
    return {"path": str(Path(record["path"]).resolve()), "sha256": record["sha256"]}


def _training_input_bindings(*, expected_fit_chains, expected_select_chains,
                             expected_manifest_path, expected_chain_bank_sha256,
                             expected_manifest_sha256) -> dict:
    fit = [_file_binding(path, label="fit chain bank") for path in expected_fit_chains]
    if not fit:
        raise ValueError("at least one expected fit chain bank is required")
    select = _file_binding(expected_select_chains, label="select chain bank")
    question_manifest = _file_binding(expected_manifest_path,
                                      label="frozen question manifest")
    if select["sha256"] != expected_chain_bank_sha256:
        raise ValueError("live select chain bank sha256 does not match expected chain identity")
    if question_manifest["sha256"] != expected_manifest_sha256:
        raise ValueError("live frozen question manifest sha256 does not match pinned identity")
    return {"fit_chains": fit, "select_chains": select,
            "question_manifest": question_manifest}


def _require_summary_provenance(summary, training_inputs) -> None:
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("lane summary has no training-input provenance")
    observed_fit = provenance.get("fit_chains")
    if not isinstance(observed_fit, list):
        raise ValueError("lane summary provenance fit_chains must be a list")
    observed = [_source_core(item, label="fit_chains") for item in observed_fit]
    if observed != training_inputs["fit_chains"]:
        raise ValueError("lane summary provenance does not match expected fit chain bank(s)")
    if _source_core(provenance.get("select_chains"), label="select_chains") != \
            training_inputs["select_chains"]:
        raise ValueError("lane summary provenance does not match expected select chain bank")
    questions = provenance.get("questions")
    if not isinstance(questions, list) or len(questions) != 1:
        raise ValueError("lane summary provenance must name one frozen question manifest")
    if _source_core(questions[0], label="questions") != training_inputs["question_manifest"]:
        raise ValueError("lane summary provenance does not match expected question manifest")


def _load_fit_population(paths) -> list[dict]:
    rows, seen = [], set()
    for path in paths:
        for row in load_chains(path):
            identity = (row["qkey"], row["rep"])
            if identity in seen:
                raise ValueError(f"duplicate fit chain {identity} across input banks")
            seen.add(identity)
            row["_ends"] = [int(end) for end in row.get("block_ends", [])]
            row["_trainable"] = is_trainable(row)
            rows.append(row)
    return rows


def _require_no_train_drops(over) -> None:
    value = over.get("train_dropped")
    if not isinstance(value, int) or isinstance(value, bool) or value != 0:
        raise ValueError(
            f"lane summary config.over_max_seq_len.train_dropped is {value!r}, expected the "
            "integer 0: a lane that dropped over-context TRAINING chains is unpublishable by "
            "construction (iclr_osp_train.lane_is_publishable), because resume.train_digest is "
            "computed FROM the declared survivors and therefore cannot bind the drop list that "
            "defines them. Raise --max-seq-len until the pool fits and retrain.")
    if "train_dropped_identities" in over:
        identities = over["train_dropped_identities"]
        if not isinstance(identities, list) or identities:
            raise ValueError(
                f"lane summary config.over_max_seq_len.train_dropped_identities is "
                f"{identities!r}; a publishable lane drops no training chain, so it must be "
                "the empty list (a lane predating the field may omit it, but only with "
                "train_dropped exactly 0)")


def _population_candidate(fit, select, *, faithful,
                          expected_val_count, expected_val_salt,
                          expected_max_chains_per_question, expected_max_chain_tokens,
                          expected_max_train_questions) -> dict:
    from rl_controller.scripts import iclr_osp_train as train_cli

    population = R.faithful_subsample(fit) if faithful else list(fit)
    args = SimpleNamespace(
        max_chain_tokens=expected_max_chain_tokens,
        max_chains_per_question=expected_max_chains_per_question,
        val_count=expected_val_count,
        val_salt=expected_val_salt,
        max_train_questions=expected_max_train_questions,
    )
    try:
        train, hold, select_pop, split = train_cli.partition(population, select, args)
    except (SystemExit, ValueError) as exc:
        raise ValueError(f"training population recomputation failed: {exc}") from None
    return {"faithful_subsample": faithful, "train_digest": T.train_order_digest(train),
            "n_train_chains": len(train), "split": split,
            "hold_ids": {(chain["qkey"], chain["rep"]) for chain in hold},
            "select_ids": {(chain["qkey"], chain["rep"]) for chain in select_pop}}


def _verify_training_population(summary, training_inputs, *, expected_faithful,
                                expected_val_count, expected_val_salt,
                                expected_max_chains_per_question, expected_max_chain_tokens,
                                expected_max_train_questions,
                       expected_max_seq_len) -> tuple[dict, dict]:
    config = _mapping(summary.get("config"))
    _require_no_train_drops(_mapping(config.get("over_max_seq_len")))
    stored_digest = _mapping(config.get("resume")).get("train_digest")
    if not isinstance(stored_digest, str) or len(stored_digest) != 64:
        raise ValueError("lane summary has no full resume.train_digest population attestation")
    fit = _load_fit_population([record["path"] for record in training_inputs["fit_chains"]])
    select = _load_fit_population([training_inputs["select_chains"]["path"]])
    candidates = [
        _population_candidate(
            fit, select, faithful=faithful,
            expected_val_count=expected_val_count,
            expected_val_salt=expected_val_salt,
            expected_max_chains_per_question=expected_max_chains_per_question,
            expected_max_chain_tokens=expected_max_chain_tokens,
            expected_max_train_questions=expected_max_train_questions)
        for faithful in (False, True)
    ]
    declared = config.get("training_population")
    if declared is not None:
        actual = next(candidate for candidate in candidates
                      if candidate["faithful_subsample"] is expected_faithful)
        expected_declaration = {
            "schema": TRAINING_POPULATION_SCHEMA,
            "faithful_subsample": expected_faithful,
            "train_digest": actual["train_digest"],
            "n_train_chains": actual["n_train_chains"],
            "population_config": {
                "max_chains_per_question": expected_max_chains_per_question,
                "val_count": expected_val_count,
                "val_salt": expected_val_salt,
                "max_chain_tokens": expected_max_chain_tokens,
                "max_train_questions": expected_max_train_questions,
                "max_seq_len": expected_max_seq_len,
            },
        }
        if declared != expected_declaration:
            raise ValueError("lane training_population declaration does not match recomputation")
        if actual["train_digest"] != stored_digest:
            raise ValueError("lane training_population digest does not match resume.train_digest")
    else:
        matches = [candidate for candidate in candidates
                   if candidate["train_digest"] == stored_digest]
        if len(matches) != 1:
            raise ValueError(
                "training population attestation is not uniquely reproducible as all-chains "
                f"or faithful (matching arms: "
                f"{[m['faithful_subsample'] for m in matches]})")
        actual = matches[0]
        if actual["faithful_subsample"] is not expected_faithful:
            raise ValueError(
                f"faithful training population mismatch: lane proves "
                f"{actual['faithful_subsample']}, export requested {expected_faithful}")
    if config.get("split") != actual["split"]:
        raise ValueError("lane summary training partition does not match recomputation")
    attestation = {"schema": TRAINING_POPULATION_SCHEMA,
                   "faithful_subsample": actual["faithful_subsample"],
                   "train_digest": actual["train_digest"],
                   "n_train_chains": actual["n_train_chains"],
                   "train_dropped": 0,
                   "train_dropped_identities": [],
                   "verification": "recomputed_train_digest"}
    populations = {"hold_ids": actual["hold_ids"], "select_ids": actual["select_ids"],
                   "split": actual["split"]}
    return attestation, populations


def _lane_training_inputs(summary) -> dict:
    provenance = _mapping(summary.get("provenance"))
    fit_records = provenance.get("fit_chains")
    if not isinstance(fit_records, list) or not fit_records:
        raise ValueError("lane summary provenance fit_chains is not a non-empty list")
    fit = [_source_core(record, label="fit_chains") for record in fit_records]
    select = _source_core(provenance.get("select_chains"), label="select_chains")
    questions = provenance.get("questions")
    if not isinstance(questions, list) or len(questions) != 1:
        raise ValueError("lane summary provenance must name one frozen question manifest")
    manifest = _source_core(questions[0], label="questions")
    training_inputs = _training_input_bindings(
        expected_fit_chains=[record["path"] for record in fit],
        expected_select_chains=select["path"],
        expected_manifest_path=manifest["path"],
        expected_chain_bank_sha256=select["sha256"],
        expected_manifest_sha256=manifest["sha256"])
    _require_summary_provenance(summary, training_inputs)
    return training_inputs


def _require_canonical_lane_source(lane_dir, name: str, *, label: str = "lane source") -> Path:
    raw = Path(lane_dir) / name
    if raw.is_symlink():
        raise ValueError(f"{label} {raw} is a symlink; a run dir must own its files")
    if not raw.is_file():
        raise ValueError(f"missing {label} {raw}")
    if raw.resolve() != raw:
        raise ValueError(f"{label} {raw} is not at its canonical path ({raw.resolve()})")
    return raw


def _read_nofollow(path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(path), flags)
    try:
        with os.fdopen(fd, "rb") as handle:
            return handle.read()
    except OSError as exc:
        raise ValueError(f"cannot read lane source {path}: {exc}") from None


def _lane_source_bindings(lane_dir) -> dict[str, dict[str, str]]:
    lane_dir = Path(lane_dir)
    return {name: _file_binding(_require_canonical_lane_source(lane_dir, name),
                                label=f"lane source {name}")
            for name in LANE_SOURCE_NAMES}


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def _require_lane_config(lane_sources, *, expected_seed: int,
                         expected_lambda: float) -> dict:
    summary_path = lane_sources["summary.json"]["path"]
    try:
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid lane source summary.json: {exc}") from None
    return _require_summary_config(summary, expected_seed=expected_seed,
                                   expected_lambda=expected_lambda)


def _require_summary_config(summary, *, expected_seed: int,
                            expected_lambda: float) -> dict:
    if not isinstance(summary, dict):
        raise ValueError("invalid lane source summary.json: not a JSON object")
    config = summary.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"invalid lane source summary.json: config is "
                         f"{type(config).__name__}, expected an object")
    if config.get("seed") != expected_seed:
        raise ValueError(f"lane summary seed {config.get('seed')!r} != {expected_seed!r}")
    if config.get("osp_lambda") != expected_lambda:
        raise ValueError(f"lane summary osp_lambda {config.get('osp_lambda')!r} != "
                         f"{expected_lambda!r}")
    return summary


def _summary_training_inputs(summary, *, head_mode: str = "osp") -> dict:
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("lane summary has no training-input provenance")
    required = ("fit_chains", "select_chains", "questions", "label_dump")
    missing = [key for key in required if key not in provenance]
    if missing:
        raise ValueError(f"lane summary training provenance lacks {missing}")
    if head_mode == "osp":
        if provenance["label_dump"] is not None:
            raise ValueError("an OSP lane must not have a qv label-dump training input")
    elif head_mode == "qv":
        if not isinstance(provenance["label_dump"], dict) or not provenance["label_dump"]:
            raise ValueError("a qv C2 lane must record its certified label-dump training input")
    else:
        raise ValueError(f"unknown head mode {head_mode!r}")
    return {key: provenance[key] for key in required}


def _verify_training_identity(summary, expected_training: dict, *,
                              require_head_mode: str = "osp") -> dict:
    if not isinstance(expected_training, dict):
        raise ValueError("expected training configuration is required")
    expected = dict(expected_training)
    if expected.pop("schema", None) != TRAINING_CONFIG_SCHEMA:
        raise ValueError("expected training configuration has the wrong schema")
    if require_head_mode not in ("osp", "qv"):
        raise ValueError(f"unknown required head mode {require_head_mode!r}")
    head_mode, targets = expected.get("head_mode"), list(expected.get("targets") or [])
    if head_mode != require_head_mode:
        raise ValueError(f"lane training head_mode={head_mode!r} but this path requires "
                         f"head_mode={require_head_mode!r} (OSP survival-product replay "
                         f"reads only head_mode=osp lanes; qv C2 lanes are read by iclr_ladder)")
    expected["inputs"] = _summary_training_inputs(summary, head_mode=require_head_mode)
    config = summary.get("config")
    if not isinstance(config, dict):
        raise ValueError("lane summary has no config")
    resume = _mapping(config.get("resume"))
    stored_digest = resume.get("args_digest")
    if not isinstance(stored_digest, str) or len(stored_digest) != 64:
        raise ValueError("lane summary has no full resume.args_digest")

    declared = config.get("training_config")
    if declared is not None:
        if resume.get("args_digest_schema") != TRAINING_ARGS_DIGEST_SCHEMA:
            raise ValueError("lane summary has no recognized resume args-digest schema")
        canonical = {"schema": TRAINING_CONFIG_SCHEMA, **expected}
        if declared != canonical:
            missing = sorted(set(declared) - set(canonical))
            unknown = sorted(set(canonical) - set(declared))
            differing = sorted(key for key in set(declared) & set(canonical)
                               if declared[key] != canonical[key] and key != "inputs")
            raise ValueError(
                "lane training_config identity (including faithful arm) does not match "
                f"expected training configuration: the lane records {missing} with no "
                f"expectation, the expectation carries {unknown} the lane does not record, "
                f"and {differing} disagree. A field the reconstruction does not have AT ALL "
                "refuses every lane from the commit that added it (2026-09-11: `cost`), so "
                "the three groups are named separately.")
        digest_input = dict(expected)
        digest_input.pop("max_train_questions", None)
        proof = "declared_config_and_args_digest"
    else:
        if resume.get("args_digest_schema") is not None:
            raise ValueError("schema-declared lane is missing training_config; refusing legacy "
                             "downgrade")
        if expected.get("cost") != LEGACY_OSP_COST:
            raise ValueError(
                f"this lane declares no training_config, so it predates `--osp-cost` and was "
                f"charged in the {LEGACY_OSP_COST!r} axis; the export declares "
                f"{expected.get('cost')!r}. A lane trained on another axis records its own "
                "training_config and is verified through that, not reconstructed here.")
        digest_input = dict(expected)
        digest_input.pop("max_train_questions", None)
        digest_input.pop("cost", None)
        proof = "legacy_reconstructed_args_digest"
    if T.run_args_digest(digest_input) != stored_digest:
        raise ValueError("lane training configuration does not match resume.args_digest")

    top_level = {
        "seed": expected["seed"], "osp_lambda": expected["lambda"],
        "head_mode": expected["head_mode"], "targets": expected["targets"],
        "osp_grade": expected["grade"], "osp_anneal": expected["anneal"],
        "epochs": expected["epochs"], "patience": expected["patience"],
        "lr": expected["lr"], "weight_decay": expected["weight_decay"],
    }
    bad = {key: (config.get(key), value) for key, value in top_level.items()
           if config.get(key) != value}
    if bad:
        raise ValueError(f"lane training summary identity mismatch: {bad}")
    if head_mode == "osp" and targets != ["q", "v"]:
        raise ValueError("WS-F OSP export requires head_mode=osp and targets=q,v")
    if head_mode == "qv" and (targets[:2] != ["q", "v"] or len(set(targets)) != len(targets)):
        raise ValueError(f"a qv C2 lane requires distinct targets starting with q,v; got {targets}")
    provenance = summary["provenance"]
    if expected["model"] != MODEL_ID or expected["revision"] != MODEL_REVISION:
        raise ValueError("OSP export requires the pinned model id and revision")
    if provenance.get("model_revision") != expected["revision"]:
        raise ValueError("lane model revision does not match expected training revision")
    reader = config.get("reader")
    if not isinstance(reader, dict):
        raise ValueError("lane summary has no reader capacity attestation")
    head_outputs = 1 if head_mode == "osp" else len(targets)
    exact_count = L.trainable_param_count(
        L.QWEN3_1P7B, n_unfrozen=expected["n_unfrozen"], head_outputs=head_outputs)["trainable"]
    reader_expected = {
        "head_mode": head_mode, "outputs": ["pi"] if head_mode == "osp" else targets,
        "n_layers": L.QWEN3_1P7B.num_hidden_layers,
        "n_unfrozen": expected["n_unfrozen"],
        "n_trainable_params": exact_count,
        "graph_isolation_verified": True, "aux_generated_tokens": 0,
    }
    reader_bad = {key: (reader.get(key), value) for key, value in reader_expected.items()
                  if reader.get(key) != value}
    if reader_bad:
        raise ValueError(f"lane reader attestation mismatch: {reader_bad}")
    if config.get("publishable") is not True:
        raise ValueError("lane summary must explicitly attest publishable=true")
    return {
        "schema": TRAINING_CONFIG_SCHEMA,
        "proof": proof,
        "args_digest": stored_digest,
        "model": expected["model"], "model_path": expected["model_path"],
        "model_revision": expected["revision"],
        "head_mode": head_mode, "targets": targets,
        "osp_grade": expected["grade"], "osp_anneal": expected["anneal"],
        "osp_cost": expected["cost"],
        "max_seq_len": expected["max_seq_len"],
        "reader": reader,
    }


def _require_export_semantics(item, *, expected_commit, expected_source_branch,
                              expected_method, expected_dataset, expected_split,
                              expected_faithful, expected_seed, expected_lambda,
                              row=False) -> None:
    expected = {
        "source_commit": expected_commit,
        "source_branch": expected_source_branch,
        "method_name": expected_method,
        "dataset": expected_dataset,
        "seed": expected_seed,
    }
    bad = {key: (item.get(key), value) for key, value in expected.items()
           if item.get(key) != value}
    observed_split = item.get("split")
    allowed_splits = {"val", expected_split} if row else {expected_split}
    if observed_split not in allowed_splits:
        bad["split"] = (observed_split, sorted(allowed_splits))
    if item.get("knob") != {"name": "lambda", "value": expected_lambda}:
        bad["knob"] = (item.get("knob"), {"name": "lambda", "value": expected_lambda})
    frozen = item.get("frozen_config", {})
    expected_arm = "faithful_1_chain_per_question" if expected_faithful else "all_chains"
    if frozen.get("lambda") != expected_lambda:
        bad["frozen_config.lambda"] = (frozen.get("lambda"), expected_lambda)
    if frozen.get("arm") != expected_arm:
        bad["frozen_config.arm"] = (frozen.get("arm"), expected_arm)
    if not row and item.get("faithful_subsample") is not expected_faithful:
        bad["faithful_subsample"] = (item.get("faithful_subsample"), expected_faithful)
    if bad:
        raise ValueError(f"export semantic identity mismatch (observed, expected): {bad}")


def _require_manifest_inputs(manifest, *, pool_source, val_source) -> None:
    expected = {"inputs": pool_source, "val_inputs": val_source}
    for key, source in expected.items():
        records = manifest.get(key)
        if not isinstance(records, list) or len(records) != 1 \
                or not isinstance(records[0], dict):
            raise ValueError(f"export manifest {key} does not name exactly one bound source")
        observed_path = records[0].get("path")
        if observed_path is None or Path(observed_path).resolve() != Path(source["path"]):
            raise ValueError(f"export manifest {key} bound source path does not match")
        if records[0].get("sha256") != source["sha256"]:
            raise ValueError(f"export manifest {key} bound source sha256 does not match")


def _validation_pool_binding(manifest, *, lane_report, selector, label, expected_seed,
                             expected_chain_bank_sha256, expected_manifest_sha256) -> dict:
    if selector in POOL_SPLIT_LANE_DUMP:
        return _with_read_split_label(_lane_dump_binding(selector, lane_report), label)
    inputs = manifest.get("inputs")
    if (not isinstance(inputs, list) or len(inputs) != 1
            or not isinstance(inputs[0], dict) or not inputs[0].get("path")):
        raise ValueError(
            f"export manifest inputs does not name exactly one scored pool dump, so the "
            f"{REPORT_POOL_BINDING} binding for split {selector!r} cannot be recomputed")
    try:
        return _with_read_split_label(_verify_report_pool_dump(
            [inputs[0]["path"]], lane_report=lane_report, split=selector,
            expected_seed=expected_seed,
            expected_chain_bank_sha256=expected_chain_bank_sha256,
            expected_manifest_sha256=expected_manifest_sha256), label)
    except SystemExit as exc:
        raise ValueError(str(exc)) from None


def _expected_identity(manifest, *, expected_manifest_sha256, expected_chain_bank_sha256,
                       expected_pool, expected_run_id) -> None:
    expected = {
        "manifest_sha256": expected_manifest_sha256,
        "chain_bank_sha256": expected_chain_bank_sha256,
        "pool": expected_pool,
        "run_id": expected_run_id,
    }
    bad = {key: (manifest.get(key), value) for key, value in expected.items()
           if manifest.get(key) != value}
    if bad:
        raise ValueError(f"export identity mismatch (observed, expected): {bad}")


def _require_val_split_identity(manifest, rows, lane_val_identity) -> None:
    declared = (manifest.get("split_chain_identity") or {}).get("val") or {}
    observed = {k: declared.get(k) for k in ("chain_bank_sha256", "manifest_sha256")}
    if observed != lane_val_identity:
        raise ValueError(
            f"export manifest split_chain_identity.val is {observed}, the verified lane's "
            f"validation source is {lane_val_identity}: the val dump is the lane's held-out "
            "FIT split, and an export that names another bank for it labels the rows honest "
            "selection is performed on with chains they did not come from")
    wrong = [row["qkey"] for row in rows if row.get("split") == "val"
             and {k: row.get(k) for k in lane_val_identity} != lane_val_identity]
    if wrong:
        raise ValueError(
            f"{len(wrong)} val rows (e.g. {wrong[:3]}) do not carry the verified lane's "
            f"validation identity {lane_val_identity}")


def validate_export(records, *, lane_dir, expected_commit, expected_lane_commit=None,
                    expected_source_branch,
                    producer_script, expected_method, expected_dataset, expected_split,
                    expected_split_selector=None,
                    expected_faithful, expected_manifest_path, expected_manifest_sha256,
                    expected_fit_chains, expected_select_chains,
                    expected_chain_bank_sha256, expected_pool, expected_val_count,
                    expected_val_salt, expected_max_chains_per_question,
                    expected_max_chain_tokens, expected_max_train_questions,
                    expected_max_seq_len, expected_run_id, expected_seed, expected_lambda,
                    expected_training_config,
                    expected_training_chain_bank_sha256=None,
                    expected_training_manifest_sha256=None,
                    marker=None) -> dict:
    lane_commit = _lane_commit_declaration(expected_commit, expected_lane_commit)
    selector = split_selector_declaration(expected_split, expected_split_selector,
                                          pool=expected_pool)
    lane_bound_pool = selector in POOL_SPLIT_LANE_DUMP
    training_bank = expected_training_chain_bank_sha256
    training_manifest = expected_training_manifest_sha256
    if lane_bound_pool:
        training_bank = training_bank or expected_chain_bank_sha256
        training_manifest = training_manifest or expected_manifest_sha256
    elif not (training_bank and training_manifest):
        raise ValueError(
            f"--expected-split-selector {selector!r} is a report pool, so the LANE's training "
            "select bank and question manifest must be declared explicitly "
            "(--expected-training-chain-bank-sha256 / --expected-training-manifest-sha256). "
            "--expected-chain-bank-sha256 / --expected-manifest-sha256 describe the pool the "
            "rows were replayed from, which is a different bank; inheriting them here would "
            "reconcile the lane's training inputs against chains it never trained on.")
    records = Path(records)
    manifest_path = export_manifest_path(records)
    if not records.is_file() or not manifest_path.is_file():
        raise ValueError(f"incomplete export: need {records} and {manifest_path}")
    records_sha = _sha256(records)
    manifest_sha = _sha256(manifest_path)
    lane_sources = _lane_source_bindings(lane_dir)
    producer = _file_binding(producer_script, label="producer script")
    summary = _require_lane_config(lane_sources, expected_seed=expected_seed,
                                   expected_lambda=expected_lambda)
    training_inputs = _training_input_bindings(
        expected_fit_chains=expected_fit_chains,
        expected_select_chains=expected_select_chains,
        expected_manifest_path=expected_manifest_path,
        expected_chain_bank_sha256=training_bank,
        expected_manifest_sha256=training_manifest)
    _require_summary_provenance(summary, training_inputs)
    if not isinstance(expected_training_config, dict):
        raise ValueError("expected training configuration is required")
    expectation_crosscheck = {
        "lambda": expected_lambda, "seed": expected_seed,
        "faithful": expected_faithful, "val_count": expected_val_count,
        "val_salt": expected_val_salt,
        "max_chains_per_question": expected_max_chains_per_question,
        "max_chain_tokens": expected_max_chain_tokens,
        "max_train_questions": expected_max_train_questions,
        "max_seq_len": expected_max_seq_len,
    }
    inconsistent = {
        key: (expected_training_config.get(key), value)
        for key, value in expectation_crosscheck.items()
        if expected_training_config.get(key) != value
    }
    if inconsistent:
        raise ValueError(f"expected training identity is internally inconsistent: "
                         f"{inconsistent}")
    lane_report = verify_lane_dir(
        lane_dir, expected_commit=lane_commit, expected_seed=expected_seed,
        expected_lambda=expected_lambda, expected_faithful=expected_faithful,
        expected_training=expected_training_config, require_head_mode="osp")
    if lane_report["sources"] != lane_sources:
        raise ValueError("lane sources changed between the export binding and the lane gate: "
                         f"{lane_report['sources']} != {lane_sources}")
    training_identity = lane_report["training_identity"]
    training_population = lane_report["training_population"]
    lane_val_identity = _lane_val_identity(lane_report)
    semantic_config = {
        "source_branch": expected_source_branch,
        "method": expected_method,
        "dataset": expected_dataset,
        "split": expected_split,
        "faithful": expected_faithful,
        "pool": expected_pool,
        "training_chain_bank_sha256": training_bank,
        "training_manifest_sha256": training_manifest,
        "val_count": expected_val_count,
        "val_salt": expected_val_salt,
        "max_chains_per_question": expected_max_chains_per_question,
        "max_chain_tokens": expected_max_chain_tokens,
        "max_train_questions": expected_max_train_questions,
        "max_seq_len": expected_max_seq_len,
        "run_id": expected_run_id,
        "seed": expected_seed,
        "lambda": expected_lambda,
        "training_config": expected_training_config,
    }
    if marker is not None:
        marker_path = Path(marker)
        if not marker_path.is_file():
            raise ValueError(f"missing completion marker {marker_path}")
        marked = json.loads(marker_path.read_text(encoding="utf-8"))
        if marked.get("schema") != EXPORT_MARKER_SCHEMA:
            raise ValueError(f"completion marker schema {marked.get('schema')!r} != "
                             f"{EXPORT_MARKER_SCHEMA!r}")
        marked_records = marked.get("records", {}).get("path")
        marked_manifest = marked.get("export_manifest", {}).get("path")
        if marked_records is None or Path(marked_records).resolve() != records.resolve():
            raise ValueError("completion marker records path does not match export")
        if marked_manifest is None or Path(marked_manifest).resolve() != manifest_path.resolve():
            raise ValueError("completion marker manifest path does not match export")
        if marked.get("records", {}).get("sha256") != records_sha:
            raise ValueError("records sha256 does not match completion marker")
        if marked.get("export_manifest", {}).get("sha256") != manifest_sha:
            raise ValueError("export manifest sha256 does not match completion marker")
        if marked.get("lane_sources") != lane_sources:
            raise ValueError("completion marker lane source paths/hashes do not match")
        if marked.get("producer") != producer:
            raise ValueError("completion marker producer path/hash does not match")
        if marked.get("expected_commit") != expected_commit:
            raise ValueError("completion marker expected commit does not match")
        if marked.get("semantic_config") != semantic_config:
            raise ValueError("completion marker semantic config does not match")
        if marked.get("training_inputs") != training_inputs:
            raise ValueError("completion marker training input paths/hashes do not match")
        if marked.get("training_population") != training_population:
            raise ValueError("completion marker training population does not match")
        if marked.get("training_identity") != training_identity:
            raise ValueError("completion marker training identity does not match")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != osp.REPLAY_SCHEMA:
        raise ValueError(f"export manifest schema {manifest.get('schema_version')!r} != "
                         f"{osp.REPLAY_SCHEMA!r}")
    _expected_identity(manifest, expected_manifest_sha256=expected_manifest_sha256,
                       expected_chain_bank_sha256=expected_chain_bank_sha256,
                       expected_pool=expected_pool, expected_run_id=expected_run_id)
    _require_export_semantics(
        manifest, expected_commit=expected_commit,
        expected_source_branch=expected_source_branch, expected_method=expected_method,
        expected_dataset=expected_dataset, expected_split=expected_split,
        expected_faithful=expected_faithful, expected_seed=expected_seed,
        expected_lambda=expected_lambda)
    pool_binding = _validation_pool_binding(
        manifest, lane_report=lane_report, selector=selector, label=expected_split,
        expected_seed=expected_seed,
        expected_chain_bank_sha256=expected_chain_bank_sha256,
        expected_manifest_sha256=expected_manifest_sha256)
    _require_manifest_inputs(manifest, pool_source=pool_binding["dump"],
                             val_source=lane_sources["val_scored.jsonl"])
    if manifest.get("training_identity") != training_identity:
        raise ValueError("export manifest training identity does not match verified lane")
    if manifest.get("pool_dump_binding") != pool_binding:
        raise ValueError(
            f"export manifest pool_dump_binding does not match the {pool_binding['kind']} "
            "binding this validation recomputed: the published artifact claims a different "
            "proof for the bytes its pool rows came from, and the two proofs certify "
            "different things")
    expected_captions = R.mandatory_captions(
        arm=("faithful_1_chain_per_question" if expected_faithful else "all_chains"),
        faithful=expected_faithful, n_lambda=5, n_seeds=2,
        reader=training_identity["reader"])
    if manifest.get("captions") != expected_captions:
        raise ValueError("export captions do not match verified lane reader")
    for key, expected_value in (
            ("model_id", training_identity["model"]),
            ("model_revision", training_identity["model_revision"]),
            ("tokenizer_revision", training_identity["model_revision"])):
        if manifest.get(key) != expected_value:
            raise ValueError(f"export manifest {key} does not match verified lane training")
    declared_output = Path(manifest.get("output", {}).get("path", ""))
    if not declared_output.is_absolute():
        declared_output = manifest_path.parent / declared_output
    if declared_output.resolve() != records.resolve():
        raise ValueError("export manifest output path does not match records path")
    if manifest.get("output", {}).get("sha256") != records_sha:
        raise ValueError("records sha256 does not match export manifest")
    if not manifest.get("coverage_declared") or set(manifest.get("splits_emitted", ())) != {
            "val", manifest.get("split")}:
        raise ValueError("export is not a declared val+pool coverage rectangle")
    if manifest.get("summary") is None or manifest.get("val_summary") is None:
        raise ValueError("export has unavailable/incomplete question coverage")
    qset_sha = manifest.get("qset_coverage_sha256")
    if not isinstance(qset_sha, str) or len(qset_sha) != 64:
        raise ValueError("export has no separate qset_coverage_sha256")
    question_sets = manifest.get("question_sets")
    expected_splits = {"val", manifest.get("split")}
    if not isinstance(question_sets, dict) or set(question_sets) != expected_splits:
        raise ValueError("export question-set records do not cover the declared splits")
    qkeys_by_split = {}
    for split, record in question_sets.items():
        qset_path = Path(record.get("path", ""))
        if not qset_path.is_file() or _sha256(qset_path) != record.get("sha256"):
            raise ValueError(f"question-set/qset source for {split!r} no longer matches export")
        try:
            qkeys_by_split[split] = load_question_set(qset_path)
        except SystemExit as exc:
            raise ValueError(f"invalid question-set/qset source for {split!r}: {exc}") from None
    if _qset_binding(question_sets) != qset_sha:
        raise ValueError("qset_coverage_sha256 does not bind the declared question sets")

    rows = []
    for line_no, line in enumerate(records.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = pareto.validate_policy_replay_row(json.loads(line))
        except Exception as exc:
            raise ValueError(f"{records}:{line_no}: invalid policy row: {exc}") from None
        row_val = row.get("split") == "val"
        _expected_identity(
            row,
            expected_manifest_sha256=(lane_val_identity["manifest_sha256"] if row_val
                                      else expected_manifest_sha256),
            expected_chain_bank_sha256=(lane_val_identity["chain_bank_sha256"] if row_val
                                        else expected_chain_bank_sha256),
            expected_pool=expected_pool, expected_run_id=expected_run_id)
        _require_export_semantics(
            row, expected_commit=expected_commit,
            expected_source_branch=expected_source_branch, expected_method=expected_method,
            expected_dataset=expected_dataset, expected_split=expected_split,
            expected_faithful=expected_faithful, expected_seed=expected_seed,
            expected_lambda=expected_lambda, row=True)
        for key, expected_value in (
                ("model_id", training_identity["model"]),
                ("model_revision", training_identity["model_revision"]),
                ("tokenizer_revision", training_identity["model_revision"])):
            if row.get(key) != expected_value:
                raise ValueError(f"policy row {key} does not match verified lane training")
        rows.append(row)
    if not rows:
        raise ValueError("export contains no policy rows")
    _require_val_split_identity(manifest, rows, lane_val_identity)
    rows_by_split = {split: {row["qkey"] for row in rows if row["split"] == split}
                     for split in expected_splits}
    if rows_by_split != qkeys_by_split:
        raise ValueError("policy rows do not have exact question-set/qset coverage")
    if len(rows_by_split["val"]) != manifest.get("n_val_rows") or \
            len(rows_by_split[manifest["split"]]) != manifest.get("n_rows"):
        raise ValueError("export row counts do not match its manifest")

    binding = {
        "schema": EXPORT_MARKER_SCHEMA,
        "records": {"path": str(records.resolve()), "sha256": records_sha},
        "export_manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_sha},
        "lane_sources": lane_sources,
        "producer": producer,
        "expected_commit": expected_commit,
        "semantic_config": semantic_config,
        "training_inputs": training_inputs,
        "training_population": training_population,
        "training_identity": training_identity,
        "pool_dump_binding": pool_binding,
        "run_id": expected_run_id,
        "chain_bank_sha256": expected_chain_bank_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "pool": expected_pool,
        "qset_coverage_sha256": qset_sha,
    }
    if marker is not None:
        for key in ("run_id", "chain_bank_sha256", "manifest_sha256", "pool",
                    "qset_coverage_sha256", "pool_dump_binding"):
            if marked.get(key) != binding[key]:
                raise ValueError(f"completion marker {key} does not match export")
    return binding


def seal_export(records, marker, **expected) -> dict:
    binding = validate_export(records, **expected)
    _atomic_write(marker, _json_bytes(binding))
    return binding


def add_common(p):
    p.add_argument("--scored", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--lambda", dest="lam", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--pi-key", default="pi")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("records", help="A1/A3 policy-replay records")
    add_common(rec)
    rec.add_argument("--run-id", required=True)
    rec.add_argument("--config-id", default=None)
    rec.add_argument("--method-family", default="ospruner")
    rec.add_argument("--method-name", default="OSP")
    rec.add_argument("--knob-name", default="lambda")
    rec.add_argument("--dataset", required=True)
    rec.add_argument("--read-split-label", default=None,
                     help="the LABEL the read rows carry in `split`, when it differs from "
                          "--split. --split is the BINDING SELECTOR and must stay the pool the "
                          "dump was scored for; this changes only what the rows are filed "
                          "under, and only `select` or --pool are accepted. Defaults to "
                          "--split, which is every existing caller.")
    rec.add_argument("--split", required=True,
                     help="the POOL split AND the binding selector (e.g. select). `val` is "
                          "emitted automatically from "
                          "--val-scored and may not be named here: contract §2.2 rule 7 wants "
                          "BOTH, in one rectangle")
    rec.add_argument("--val-scored", nargs="+", default=None,
                     help="REQUIRED unless --allow-open-coverage: the shared val split's scored "
                          "dump. Honest selection (§5.2) picks each method's operating point ON "
                          "VAL, so a config with no val rows has no honest operating point at "
                          "all and is unusable for the primary statistic")
    rec.add_argument("--val-question-set", default=None,
                     help="the declared qkeys of the val split (as --question-set is for the "
                          "pool split)")
    rec.add_argument("--pool", required=True,
                     help="the pool identity carried in every row (contract §2.1)")
    rec.add_argument("--chain-bank-sha256", required=True,
                     help="sha256 of the pool chains.jsonl BOTH methods read — this is what "
                          "proves the shared natural/always-fire anchors are auditable")
    rec.add_argument("--manifest-sha256", required=True)
    rec.add_argument("--val-chain-bank-sha256", default=None,
                     help="sha256 of the chain bank the VAL dump came from, when that is not "
                          "the pool's. Defaults to --chain-bank-sha256 for a lane-gated "
                          "(training-pool) export, where the two are the same bank")
    rec.add_argument("--val-manifest-sha256", default=None,
                     help="the val split's question manifest sha256; pass it with "
                          "--val-chain-bank-sha256 or neither")
    rec.add_argument("--model-id", default=MODEL_ID)
    rec.add_argument("--model-revision", default=MODEL_REVISION)
    rec.add_argument("--tokenizer-revision", default=MODEL_REVISION)
    rec.add_argument("--prompt-contract-id", default=PROMPT_CONTRACT_ID)
    rec.add_argument("--grader-id", default=GRADER_ID)
    rec.add_argument("--selection-split", default=SELECTION_SPLIT)
    rec.add_argument("--selection-rule", default=SELECTION_RULE)
    rec.add_argument("--source-branch", required=True, type=_branch_name)
    rec.add_argument("--source-commit", default=None)
    rec.add_argument("--expected-lane-commit", default=None,
                     help="the commit the LANE WAS TRAINED from, checked by the "
                          "--lane-summary gate. Defaults to --source-commit, which is "
                          "stamped into every exported row as this export's own checkout "
                          "and therefore advances independently of the lane (job 12697181)")
    rec.add_argument("--grade", choices=osp.GRADINGS, default="loose")
    rec.add_argument("--answer-format", choices=("boxed", "code"), default="boxed")
    rec.add_argument("--force-natural-answer", action="store_true")
    rec.add_argument("--aux-generated-tokens", type=float, default=0.0)
    rec.add_argument("--question-set", default=None,
                     help="REQUIRED unless --allow-open-coverage: the qkeys the plan-rule "
                          "methods cover on this pool. Contract §2.2 rule 7 wants rectangular "
                          "coverage over val AND the pool; without a declared set this "
                          "producer cannot tell a complete replay from one silently missing "
                          "questions, and the gap would surface at merge time with the sweep "
                          "already spent")
    rec.add_argument("--allow-open-coverage", action="store_true",
                     help="emit without a declared question set. DIAGNOSTIC ONLY: the result "
                          "is not a publishable frontier point")
    rec.add_argument("--faithful", action="store_true",
                     help="A3: subsample to one chain per question before replaying")
    rec.add_argument("--lane-summary", default=None,
                     help="summary.json of the trained lane; required for a publishable "
                          "export. Its directory is put through the SAME gate verify-lane "
                          "and validate/seal run (completion marker, clean commit, training "
                          "identity, dump reconciliation), and --scored / --val-scored must "
                          "then be exactly that lane's own select_scored.jsonl / "
                          "val_scored.jsonl. Needs --source-commit")

    def add_training_args(parser, *, required):
        parser.add_argument("--expected-head-mode", required=required, choices=("osp", "qv"))
        parser.add_argument("--expected-targets", required=required)
        parser.add_argument("--expected-anneal", required=required)
        parser.add_argument("--expected-epochs", required=required, type=int)
        parser.add_argument("--expected-patience", required=required, type=int)
        parser.add_argument("--expected-lr", required=required, type=float)
        parser.add_argument("--expected-weight-decay", required=required, type=float)
        parser.add_argument("--expected-grad-clip", required=required, type=float)
        parser.add_argument("--expected-osp-grade", required=required, choices=osp.GRADINGS)
        parser.add_argument("--expected-osp-cost", required=required, choices=osp.COST_AXES,
                            help="the cost axis the lane's OBJECTIVE was charged in "
                                 "(iclr_osp_train --osp-cost). Recorded as `cost` in the "
                                 "lane's training_config and bound by its resume args digest")
        parser.add_argument("--expected-token-scale", required=required, type=float)
        parser.add_argument("--expected-init-fire-rate", required=required, type=float)
        parser.add_argument("--expected-n-unfrozen", required=required, type=int)
        parser.add_argument("--expected-chains-per-step", required=required, type=int)
        parser.add_argument("--expected-allow-unscored", required=required,
                            choices=("0", "1"))
        parser.add_argument("--expected-dtype", required=required)
        parser.add_argument("--expected-fp32-trainable", required=required,
                            choices=("0", "1"))
        parser.add_argument("--expected-model", required=required)
        parser.add_argument("--expected-model-path", required=required,
                            help="exact trainer value, or the literal NONE")
        parser.add_argument("--expected-model-revision", required=required)

    add_training_args(rec, required=False)
    rec.add_argument("--expected-val-count", type=int)
    rec.add_argument("--expected-val-salt")
    rec.add_argument("--expected-max-chains-per-question", type=int)
    rec.add_argument("--expected-max-chain-tokens", type=int)
    rec.add_argument("--expected-max-train-questions", type=int)
    rec.add_argument("--expected-max-seq-len", type=int)

    run = sub.add_parser("rundir", help="A2/A4 plan-rule run dirs")
    add_common(run)
    run.add_argument("--arm", choices=("strengthened", "bernoulli"), required=True)
    run.add_argument("--val-scored", required=True, nargs="+",
                     help="the shared val split — contract §2.2 rule 7 needs BOTH splits, and "
                          "honest selection picks operating points ON val")
    run.add_argument("--overhead-accounting", default="none",
                     choices=("none", "constant_per_boundary", "per_probe", "unknown"),
                     help="mandatory declaration (contract v1.2 §1.2); `none` is the correct "
                          "claim for a hidden-state head and `unknown` refuses to export")

    def add_validation_args(parser, *, marker_required=True):
        parser.add_argument("--records", required=True)
        parser.add_argument("--marker", required=marker_required)
        parser.add_argument("--lane-dir", required=True)
        parser.add_argument("--expected-commit", required=True)
        parser.add_argument("--expected-lane-commit", default=None,
                            help="the commit the LANE WAS TRAINED from, i.e. the checkout "
                                 "its training inputs were recorded from. Defaults to "
                                 "--expected-commit, which is this EXPORT's own checkout "
                                 "and is what the records must carry as source_commit; "
                                 "declare it separately once the reading code has advanced "
                                 "past the code that trained the lane (job 12697181)")
        parser.add_argument("--expected-source-branch", required=True, type=_branch_name)
        parser.add_argument("--producer-script", required=True)
        parser.add_argument("--expected-method", required=True)
        parser.add_argument("--expected-dataset", required=True)
        parser.add_argument("--expected-split", required=True,
                            help="the LABEL the rows and manifest must CARRY in `split`")
        parser.add_argument("--expected-split-selector", default=None,
                            help="the `--split` the export was PRODUCED with, i.e. which POOL "
                                 "BINDING to recompute: `select` for the lane's own dump, the "
                                 "pool's own id for a separately scored report pool. Defaults "
                                 "to --expected-split, which is right whenever the label was "
                                 "not changed; declare it for a report pool re-exported under "
                                 "the `select` label, or validation re-gates those rows against "
                                 "a lane dump that does not exist (job 14017290)")
        parser.add_argument("--expected-faithful", required=True, choices=("0", "1"))
        parser.add_argument("--expected-manifest-path", required=True)
        parser.add_argument("--expected-manifest-sha256", required=True)
        parser.add_argument("--expected-fit-chains", required=True, nargs="+")
        parser.add_argument("--expected-select-chains", required=True)
        parser.add_argument("--expected-chain-bank-sha256", required=True)
        parser.add_argument("--expected-pool", required=True)
        parser.add_argument("--expected-training-chain-bank-sha256", default=None)
        parser.add_argument("--expected-training-manifest-sha256", default=None)
        parser.add_argument("--expected-val-count", required=True, type=int)
        parser.add_argument("--expected-val-salt", required=True)
        parser.add_argument("--expected-max-chains-per-question", required=True, type=int)
        parser.add_argument("--expected-max-chain-tokens", required=True, type=int)
        parser.add_argument("--expected-max-train-questions", required=True, type=int)
        parser.add_argument("--expected-max-seq-len", required=True, type=int)
        parser.add_argument("--expected-run-id", required=True)
        parser.add_argument("--expected-seed", required=True, type=int)
        parser.add_argument("--expected-lambda", required=True, type=float)
        add_training_args(parser, required=True)

    lane = sub.add_parser("verify-lane", help="prove a finished lane's training identity "
                          "(no export); the only OSP-side reader of a qv C2 lane")
    lane.add_argument("--lane-dir", required=True)
    lane.add_argument("--require-head-mode", required=True, choices=("osp", "qv"))
    lane.add_argument("--expected-commit", required=True)
    lane.add_argument("--expected-seed", required=True, type=int)
    lane.add_argument("--expected-lambda", required=True, type=float)
    lane.add_argument("--expected-faithful", required=True, choices=("0", "1"))
    lane.add_argument("--expected-val-count", required=True, type=int)
    lane.add_argument("--expected-val-salt", required=True)
    lane.add_argument("--expected-max-chains-per-question", required=True, type=int)
    lane.add_argument("--expected-max-chain-tokens", required=True, type=int)
    lane.add_argument("--expected-max-train-questions", required=True, type=int)
    lane.add_argument("--expected-max-seq-len", required=True, type=int)
    add_training_args(lane, required=True)
    lane.add_argument("--out", default=None, help="write the verified identity report here")
    qsets = sub.add_parser("question-sets",
                           help="derive and publish the declared val/pool question sets")
    qsets.add_argument("--fit-chains", required=True, nargs="+",
                       help="the fit bank(s) the lanes were trained on")
    qsets.add_argument("--select-chains", required=True,
                       help="the select bank; its questions are the declared pool set")
    qsets.add_argument("--val-count", required=True, type=int)
    qsets.add_argument("--val-salt", required=True)
    qsets.add_argument("--faithful", action="store_true",
                       help="the lanes' arm: subsample to one chain per question BEFORE "
                            "partitioning, exactly as the trainer does")
    qsets.add_argument("--lane-dir", required=True, nargs="+",
                       help="every lane this batch will export; each one's recorded "
                            "config.split must equal the derived split")
    qsets.add_argument("--out-dir", required=True,
                       help="writes val.txt and pool.txt here, atomically and last")
    rqset = sub.add_parser(
        "report-question-set",
        help="materialize ONE REPORT POOL's declared question set from its REGISTERED "
             "READ SPLIT (the training pool's split comes from question-sets)")
    rqset.add_argument("--pool-manifest", required=True,
                       help="the pool's READ-SPLIT question manifest: a question manifest "
                            "holding exactly the questions the pool consists of, with their "
                            "TEXT. NOT the PARENT manifest the pool was filtered out of "
                            "(which is often the file `iclr_score_pool --questions` was "
                            "handed): the set is checked against the pool's registered read "
                            "split by count AND digest, so a 3,500-question parent handed "
                            "to an 841-question pool is refused naming both numbers. Never "
                            "the scored dump either: a question whose chains all failed to "
                            "score is absent there, so a dump-derived set would drop it "
                            "from the denominator of every rate without saying so")
    rqset.add_argument("--pool-manifest-sha256", required=True,
                       help="the digest of --pool-manifest. REQUIRED: it pins the exact "
                            "bytes the set is materialized from. It equals the digest the "
                            "scored dump records in provenance.questions.sha256 only where "
                            "the pool was SCORED against its own read split; where the pool "
                            "was scored against a larger parent, that digest is bound "
                            "separately by `records --manifest-sha256` and the registry is "
                            "what vouches for the population")
    rqset.add_argument("--pool", required=True,
                       help="the pool id. Its REGISTERED READ SPLIT "
                            "(rl_controller.iclr.read_splits) is the population and the "
                            "authority here; an unregistered pool is refused")
    rqset.add_argument("--scored-dump", action="append", default=None,
                       help="optional, repeatable, REPORT-ONLY: CHECK a scored pool dump "
                            "against the registered read split and publish the gap "
                            "(registered_only / dump_only). This is the opposite of "
                            "deriving the population from a dump and is safe for exactly "
                            "the reason that is not -- the registered digest was frozen "
                            "independently of the dump. It never changes the declared set")
    rqset.add_argument("--out", required=True,
                       help="writes the sorted qkeys here, atomically and last; pass it "
                            "to `records --question-set`. The file IS the frozen "
                            "convention's input, so its own sha256 is the registered "
                            "qkeys_sha256")
    check = sub.add_parser("validate", help="validate a marker-bound published records pair")
    add_validation_args(check)
    seal = sub.add_parser("seal", help="validate records and publish the completion marker")
    add_validation_args(seal)
    return p.parse_args(argv)


def dedup_load(paths):
    chains, provenance, seen = [], [], {}
    for path in paths:
        path = Path(path)
        loaded, record = parse_and_record(path, load_chains)
        for c in loaded:
            ident = (c["qkey"], c["rep"])
            if ident in seen:
                raise SystemExit(f"chain {ident} appears in both {seen[ident]} and {path} — "
                                 "overlapping shards would double-weight the expectation")
            seen[ident] = str(path)
        chains.extend(loaded)
        provenance.append(record)
    if not chains:
        raise SystemExit("no chains loaded")
    return chains, provenance


OSP_SCORE_KEYS = frozenset(("q", "v", "pi"))


def _require_osp_scores(chains, *, label: str = "--scored") -> None:
    for chain in chains:
        ident = (chain.get("qkey"), chain.get("rep"))
        scores, probes = chain.get("scores"), chain.get("probes")
        if not isinstance(scores, list) or not isinstance(probes, list) \
                or len(scores) != len(probes):
            raise SystemExit(
                f"{label}: chain {ident} has {scores if not isinstance(scores, list) else len(scores)} "
                f"scores for {probes if not isinstance(probes, list) else len(probes)} probes — "
                "OSP replay zips them, so a misaligned scores list silently truncates the rule")
        for score in scores:
            if not isinstance(score, dict):
                raise SystemExit(f"{label}: chain {ident} has a non-object score {score!r}")
            keys = set(score)
            if keys != set(OSP_SCORE_KEYS):
                raise SystemExit(
                    f"{label}: chain {ident} scores carry {sorted(keys)}, but OSP replay "
                    f"requires exactly {sorted(OSP_SCORE_KEYS)}. Extra targets (vpi0, r, ...) "
                    "or a missing `pi` mean this is a qv C2 dump whose `q` is a correctness "
                    "belief, not a stop probability; qv C2 lanes are read by iclr_ladder, "
                    "never by OSP survival-product replay")
            if score["q"] != score["pi"]:
                raise SystemExit(
                    f"{label}: chain {ident} has q={score['q']!r} != pi={score['pi']!r}. The "
                    "OSP trainer writes the stop probability under BOTH keys; a dump where "
                    "they differ is not an OSP lane, and qv C2 lanes are read by iclr_ladder")


def _require_pi_key(args) -> None:
    if args.pi_key != "pi":
        raise SystemExit(
            f"--pi-key must be 'pi', got {args.pi_key!r}: the OSP stop probability is written "
            "under `pi`, and pointing this at another target replays a correctness belief as "
            "a stop probability. qv C2 lanes are read by iclr_ladder, not by this exporter.")


def _expected_training_from_args(args, *, records: bool) -> dict:
    required = (
        "expected_head_mode", "expected_targets", "expected_anneal",
        "expected_epochs", "expected_patience", "expected_lr",
        "expected_weight_decay", "expected_grad_clip", "expected_osp_grade",
        "expected_osp_cost",
        "expected_token_scale", "expected_init_fire_rate", "expected_n_unfrozen",
        "expected_chains_per_step", "expected_allow_unscored", "expected_dtype",
        "expected_fp32_trainable", "expected_model", "expected_model_path",
        "expected_model_revision", "expected_val_count", "expected_val_salt",
        "expected_max_chains_per_question", "expected_max_chain_tokens",
        "expected_max_train_questions", "expected_max_seq_len",
    )
    missing = [name for name in required if getattr(args, name, None) is None]
    if missing:
        raise ValueError(f"complete explicit training expectations are required: {missing}")
    model_path = args.expected_model_path
    if model_path == "NONE":
        model_path = None
    lam = args.lam if records else args.expected_lambda
    seed = args.seed if records else args.expected_seed
    faithful = bool(args.faithful) if records else args.expected_faithful == "1"
    return {
        "schema": TRAINING_CONFIG_SCHEMA,
        "lambda": lam,
        "anneal": osp.parse_schedule(args.expected_anneal).as_dict(),
        "seed": seed,
        "head_mode": args.expected_head_mode,
        "targets": [name.strip() for name in args.expected_targets.split(",") if name.strip()],
        "epochs": args.expected_epochs, "patience": args.expected_patience,
        "lr": args.expected_lr, "weight_decay": args.expected_weight_decay,
        "grad_clip": args.expected_grad_clip, "grade": args.expected_osp_grade,
        "token_scale": args.expected_token_scale,
        "cost": args.expected_osp_cost,
        "init_fire_rate": args.expected_init_fire_rate,
        "max_chains_per_question": args.expected_max_chains_per_question,
        "val_count": args.expected_val_count, "val_salt": args.expected_val_salt,
        "max_chain_tokens": args.expected_max_chain_tokens,
        "max_train_questions": args.expected_max_train_questions,
        "max_seq_len": args.expected_max_seq_len,
        "faithful": faithful, "n_unfrozen": args.expected_n_unfrozen,
        "chains_per_step": args.expected_chains_per_step,
        "allow_unscored": args.expected_allow_unscored == "1",
        "dtype": args.expected_dtype,
        "fp32_trainable": args.expected_fp32_trainable == "1",
        "model": args.expected_model, "model_path": model_path,
        "revision": args.expected_model_revision,
    }


def _verify_completion_marker(lane_dir, lane_sources) -> dict[str, str]:
    marker = Path(lane_dir) / LANE_COMPLETION_MARKER
    if not marker.is_file():
        raise ValueError(f"lane has no completion marker {marker}")
    marker = _require_canonical_lane_source(lane_dir, LANE_COMPLETION_MARKER)
    declared: dict[str, str] = {}
    for line in _read_nofollow(marker).decode("utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"malformed completion marker line {line!r}")
        digest, name = parts[0], parts[1].strip().lstrip("*")
        if name in declared:
            raise ValueError(f"completion marker names {name} twice")
        declared[name] = digest
    if set(declared) != set(LANE_SOURCE_NAMES):
        raise ValueError(f"completion marker covers {sorted(declared)}, expected "
                         f"{sorted(LANE_SOURCE_NAMES)}")
    bad = {name: (declared[name], lane_sources[name]["sha256"]) for name in LANE_SOURCE_NAMES
           if declared[name] != lane_sources[name]["sha256"]}
    if bad:
        raise ValueError(f"completion marker does not match the lane sources: {bad}")
    return {"path": str(marker.resolve()), "sha256": _sha256(marker)}


def _lane_input_records(summary) -> list[dict]:
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("lane summary has no training-input provenance")
    records: list[dict] = []
    for key in ("fit_chains", "questions"):
        value = provenance.get(key)
        if not isinstance(value, list) or not value:
            raise ValueError(f"lane summary provenance {key} is not a non-empty list")
        records.extend(value)
    records.append(provenance.get("select_chains"))
    label_dump = provenance.get("label_dump")
    if isinstance(label_dump, dict):
        records.extend(label_dump.values())
    for record in records:
        if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
            raise ValueError("lane summary provenance holds an incomplete source record")
    return records


def _require_lane_commit(summary, expected_commit: str) -> None:
    if not (isinstance(expected_commit, str) and len(expected_commit) == 40
            and all(c in "0123456789abcdef" for c in expected_commit)):
        raise ValueError("expected commit must be a full 40-lowercase-hex sha")
    bad = [(record.get("path"), record.get("commit"), record.get("dirty"))
           for record in _lane_input_records(summary)
           if record.get("commit") != expected_commit or record.get("dirty") is not False]
    if bad:
        raise ValueError(f"lane training inputs were recorded from {sorted(map(str, bad))}, "
                         f"expected clean commit {expected_commit} with dirty=false (an "
                         "unknown/null dirty flag is not clean). That expectation is the "
                         "commit the lane was TRAINED from -- the checkout its training "
                         "inputs were recorded from -- and NOT the commit this export runs "
                         "from, which advances every time the reading code is fixed. If the "
                         "lane is honestly older than the code reading it, declare its own "
                         "commit (--expected-lane-commit, or LANE_COMMIT in the sbatch) and "
                         "leave --expected-commit/--source-commit on the export's checkout; "
                         "retrain only if the lane's provenance is genuinely wrong.")


def _lane_commit_declaration(expected_commit, expected_lane_commit):
    if expected_lane_commit is None:
        return expected_commit
    if not (isinstance(expected_lane_commit, str) and len(expected_lane_commit) == 40
            and all(c in "0123456789abcdef" for c in expected_lane_commit)):
        raise ValueError(
            f"expected lane commit {expected_lane_commit!r} is not a full 40-lowercase-hex "
            "sha. --expected-lane-commit declares the commit the lane was TRAINED from, "
            "which is not the commit this export runs from; omit the flag entirely when "
            "the two are the same rather than passing an empty or partial value.")
    return expected_lane_commit


def _require_canonical_lane_dir(lane_dir, *, label: str = "lane dir") -> Path:
    path = Path(lane_dir)
    if not path.is_absolute():
        raise ValueError(f"{label} {str(lane_dir)!r} must be an absolute path")
    if path.is_symlink():
        raise ValueError(f"{label} {path} is a symlink; verify the real directory")
    resolved = path.resolve()
    if resolved != path:
        raise ValueError(f"{label} {path} is not canonical (resolves to {resolved})")
    if not path.is_dir():
        raise ValueError(f"{label} {path} is not a directory")
    return path


def _identity_score_keys(identity) -> set[str]:
    reader = _mapping(identity.get("reader"))
    outputs = reader.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("verified lane identity has no reader outputs")
    return set(identity["targets"]) | set(outputs)


def _parse_dump(path, *, label: str, wanted: set[str]) -> tuple[set[tuple], dict[str, str]]:
    path = Path(path)
    if path.is_symlink() or path.resolve() != path:
        raise ValueError(f"lane source {label} at {path} is a symlink or not canonical; "
                         f"a lane must own its files")
    chains, record = parse_and_record(path, load_chains)
    if not chains:
        raise ValueError(f"lane source {label} holds no chains")
    identities: set[tuple] = set()
    for chain in chains:
        ident = (chain["qkey"], chain["rep"])
        if ident in identities:
            raise ValueError(f"lane source {label} names chain {ident} twice")
        identities.add(ident)
        scores, probes = chain.get("scores"), chain.get("probes")
        if not isinstance(scores, list) or not isinstance(probes, list) \
                or len(scores) != len(probes):
            raise ValueError(f"lane source {label} chain {ident} is not probe-aligned: "
                             f"{scores!r} scores for {probes!r} probes")
        for score in scores:
            if not isinstance(score, dict) or set(score) != wanted:
                raise ValueError(
                    f"lane source {label} chain {ident} scores carry "
                    f"{sorted(score) if isinstance(score, dict) else score!r}, but the verified "
                    f"training identity wrote {sorted(wanted)}")
    return identities, {"path": str(Path(path).resolve()), "sha256": record["sha256"]}


def _exact_int(value, *, label: str, expected: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"lane summary {label} is {value!r}, expected the integer {expected}")
    if value != expected:
        raise ValueError(f"lane summary {label} is {value}, but the reconstructed trainer "
                         f"population holds {expected}")


def _require_no_unscored_shrink(config) -> None:
    over = config.get("over_max_seq_len")
    if not isinstance(over, dict):
        raise ValueError(f"lane summary config.over_max_seq_len is {over!r}, "
                         "expected an object")
    for key in ("val_unscored", "select_unscored"):
        _exact_int(over.get(key), label=f"config.over_max_seq_len.{key}", expected=0)
    if "unscored_identities" in over:
        identities = over["unscored_identities"]
        if not isinstance(identities, list) or identities:
            raise ValueError(
                f"lane summary config.over_max_seq_len.unscored_identities is "
                f"{identities!r}; a publishable lane drops no chain from its dumps, so it "
                "must be the empty list")
    holders = [("config.over_max_seq_len", over)]
    if config.get("training_config") is not None:
        holders.append(("config.training_config", config["training_config"]))
    for label, holder in holders:
        value = holder.get("allow_unscored") if isinstance(holder, dict) else None
        if value is not False:
            raise ValueError(
                f"lane summary {label}.allow_unscored is {value!r}, expected false: an "
                "--allow-unscored run's dumps have a shrunk chain set and therefore "
                "different anchors, so it is unpublishable by construction")


def _verify_lane_dumps(lane_dir, summary, identity, populations) -> dict[str, dict[str, str]]:
    wanted = _identity_score_keys(identity)
    val_ids, val_binding = _parse_dump(Path(lane_dir) / "val_scored.jsonl",
                                       label="val_scored.jsonl", wanted=wanted)
    select_ids, select_binding = _parse_dump(Path(lane_dir) / "select_scored.jsonl",
                                             label="select_scored.jsonl", wanted=wanted)
    shared = val_ids & select_ids
    if shared:
        raise ValueError(f"{len(shared)} chains appear in BOTH lane dumps (e.g. "
                         f"{sorted(map(str, shared))[:3]}); val and select must be disjoint")

    config = _mapping(summary.get("config"))
    _require_no_unscored_shrink(config)
    for label, dumped, expected in (("val_scored.jsonl", val_ids, populations["hold_ids"]),
                                    ("select_scored.jsonl", select_ids,
                                     populations["select_ids"])):
        extra, missing = dumped - expected, expected - dumped
        if extra or missing:
            raise ValueError(
                f"{label} is not the population the trainer scored: {len(extra)} chains are "
                f"not in it (e.g. {sorted(map(str, extra))[:3]}) and {len(missing)} of its "
                f"chains are absent (e.g. {sorted(map(str, missing))[:3]})")

    declared_dumped = config.get("n_dumped")
    if not isinstance(declared_dumped, dict) or set(declared_dumped) != {"val", "select"}:
        raise ValueError(f"lane summary config.n_dumped is {declared_dumped!r}, expected "
                         "exactly {'val': int, 'select': int}")
    for key, ids in (("val", val_ids), ("select", select_ids)):
        _exact_int(declared_dumped[key], label=f"config.n_dumped.{key}", expected=len(ids))
    declared_split = config.get("split")
    if not isinstance(declared_split, dict):
        raise ValueError(f"lane summary config.split is {declared_split!r}, expected an object")
    _exact_int(declared_split.get("n_val_questions"), label="config.split.n_val_questions",
               expected=populations["split"]["n_val_questions"])
    return {"val_scored.jsonl": val_binding, "select_scored.jsonl": select_binding}


def verify_lane_dir(lane_dir, *, expected_commit, expected_seed, expected_lambda,
                    expected_faithful: bool, expected_training: dict,
                    require_head_mode: str) -> dict:
    if require_head_mode not in ("osp", "qv"):
        raise ValueError(f"unknown required head mode {require_head_mode!r}")
    if not isinstance(expected_training, dict):
        raise ValueError("expected training configuration is required")
    if expected_training.get("faithful") is not expected_faithful:
        raise ValueError(
            f"expected faithful arm {expected_faithful!r} disagrees with the expected training "
            f"configuration's {expected_training.get('faithful')!r}")
    lane_dir = _require_canonical_lane_dir(lane_dir)
    summary_path = lane_dir / "summary.json"
    try:
        summary_bytes = _read_nofollow(_require_canonical_lane_source(lane_dir, "summary.json"))
    except OSError as exc:
        raise ValueError(f"missing lane source summary.json: {exc}") from None
    try:
        summary = json.loads(summary_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid lane source summary.json: {exc}") from None
    summary_binding = {"path": str(summary_path.resolve()),
                       "sha256": hashlib.sha256(summary_bytes).hexdigest()}
    _require_summary_config(summary, expected_seed=expected_seed,
                            expected_lambda=expected_lambda)
    _require_lane_commit(summary, expected_commit)
    identity = _verify_training_identity(summary, expected_training,
                                         require_head_mode=require_head_mode)
    training_inputs = _lane_training_inputs(summary)
    training_population, populations = _verify_training_population(
        summary, training_inputs,
        expected_faithful=expected_faithful,
        expected_val_count=expected_training["val_count"],
        expected_val_salt=expected_training["val_salt"],
        expected_max_chains_per_question=expected_training["max_chains_per_question"],
        expected_max_chain_tokens=expected_training["max_chain_tokens"],
        expected_max_train_questions=expected_training["max_train_questions"],
        expected_max_seq_len=expected_training["max_seq_len"])
    lane_sources = {"summary.json": summary_binding,
                    **_verify_lane_dumps(lane_dir, summary, identity, populations)}
    if set(lane_sources) != set(LANE_SOURCE_NAMES):
        raise ValueError(f"lane sources {sorted(lane_sources)} != "
                         f"{sorted(LANE_SOURCE_NAMES)}")
    marker = _verify_completion_marker(lane_dir, lane_sources)
    return {
        "schema": LANE_IDENTITY_SCHEMA,
        "lane_dir": str(lane_dir),
        "require_head_mode": require_head_mode,
        "expected_commit": expected_commit,
        "seed": expected_seed, "lambda": expected_lambda,
        "faithful": expected_faithful,
        "sources": lane_sources,
        "completion_marker": marker,
        "training_identity": identity,
        "training_population": training_population,
    }


def _lane_declared_split(lane_dir) -> tuple[int, int]:
    lane_dir = _require_canonical_lane_dir(lane_dir)
    try:
        summary = json.loads(_read_nofollow(
            _require_canonical_lane_source(lane_dir, "summary.json")).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{lane_dir}: unreadable lane summary.json: {exc}") from None
    split = _mapping(_mapping(summary.get("config")).get("split"))
    counts = []
    for key in ("n_pool_questions", "n_val_questions"):
        value = split.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{lane_dir}: lane summary config.split.{key} is {value!r}, "
                             "expected a positive integer; every lane records the split it "
                             "was carved on and the declared question sets are checked "
                             "against it")
        counts.append(value)
    return counts[0], counts[1]


def _require_lane_split_agreement(lane_dirs, *, n_pool_questions: int,
                                  n_val_questions: int) -> None:
    observed = {}
    for lane_dir in lane_dirs:
        observed[str(_require_canonical_lane_dir(lane_dir))] = _lane_declared_split(lane_dir)
    distinct = set(observed.values())
    if len(distinct) > 1:
        raise ValueError(
            f"the lanes disagree with each other about their split: {observed}. Lanes "
            "exported together were trained on one pool and must have been carved on one "
            "partition; no single declared question set can be right for both.")
    bad = {lane: counts for lane, counts in observed.items()
           if counts != (n_pool_questions, n_val_questions)}
    if bad:
        lane, (lane_pool, lane_val) = sorted(bad.items())[0]
        raise ValueError(
            f"the declared question sets do not reproduce the lane's split: this derivation "
            f"has n_pool_questions={n_pool_questions} and n_val_questions={n_val_questions}, "
            f"while {lane} was trained with n_pool_questions={lane_pool} and "
            f"n_val_questions={lane_val}. The declared sets are derived from the POOL (never "
            "from what the lane scored, which would be circular), so they must reproduce the "
            "trainer's partition input exactly: the questions carrying a trainable chain, "
            "after --faithful subsampling for a faithful lane. Check --fit-chains points at "
            "the bank the lane trained on, that --faithful matches the lane's arm, and that "
            "--val-count/--val-salt are the lane's own.")


def derive_question_sets(*, fit_chains, select_chains, val_count, val_salt,
                         faithful: bool, lane_dirs, out_dir) -> dict:
    from rl_controller.scripts import iclr_osp_train as train_cli

    fit = _load_fit_population(fit_chains)
    universe = train_cli.partition_universe(fit, faithful=faithful)
    if not universe:
        raise ValueError(f"{list(fit_chains)}: no question in the fit bank carries a "
                         "trainable chain, so there is no partition input")
    try:
        val_keys, _train_order = lc.partition(universe, n_val=val_count, salt=val_salt)
    except ValueError as exc:
        raise ValueError(f"cannot partition {len(universe)} pool questions into "
                         f"{val_count} val questions: {exc}") from None
    pool_keys = {row["qkey"] for row in _load_fit_population([select_chains])}
    if not pool_keys:
        raise ValueError(f"{select_chains}: the select bank names no question")
    _require_lane_split_agreement(lane_dirs, n_pool_questions=len(universe),
                                  n_val_questions=len(val_keys))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    published = {}
    for name, keys in (("val.txt", val_keys), ("pool.txt", pool_keys)):
        path = out_dir / name
        _atomic_write(path, ("\n".join(sorted(keys)) + "\n").encode("utf-8"))
        published[name] = str(path)
    return {"schema": QUESTION_SETS_SCHEMA, "faithful_subsample": faithful,
            "n_pool_questions": len(universe), "n_val_questions": len(val_keys),
            "n_select_questions": len(pool_keys), "val_salt": val_salt,
            "published": published}


def cmd_question_sets(args) -> int:
    try:
        report = derive_question_sets(
            fit_chains=args.fit_chains, select_chains=args.select_chains,
            val_count=args.val_count, val_salt=args.val_salt,
            faithful=bool(args.faithful), lane_dirs=args.lane_dir, out_dir=args.out_dir)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"declared question sets REFUSED: {exc}") from None
    print(json.dumps({"event": "question_sets_declared", **report}), flush=True)
    return 0


def _report_manifest_questions(path) -> list[tuple[str, str]]:
    from rl_controller.splits import question_key

    path = Path(path)
    questions, seen = [], {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: not a JSON object: {exc}. A read-split "
                             "pool question manifest is JSONL, one question per line "
                             "(the same shape iclr_score_pool --questions reads)") from None
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected a question object, got "
                             f"{type(row).__name__}")
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"{path}:{line_no}: row carries no `question` text. The declared population "
                "is derived by recomputing question_key() over the manifest's own question "
                "text, so this file is not a question manifest -- a scored dump or a chain "
                "bank names questions by KEY only, and a population read off one of those "
                "silently drops every question whose chains failed to score, which is "
                "exactly the denominator error this derivation exists to prevent")
        key = question_key(question)
        declared = row.get("qkey") or row.get("question_key")
        if declared is not None and declared != key:
            raise ValueError(
                f"{path}:{line_no}: row declares qkey {declared!r} but its question text "
                f"keys to {key!r}. `iclr_score_pool` keys its prompts by the TEXT, so a row "
                "whose stored key says otherwise names a question that was never scored "
                "under it")
        if key in seen:
            raise ValueError(
                f"{path}: names the same question twice (lines {seen[key]} and {line_no}, "
                f"qkey {key}). A read-split pool manifest is deduplicated by construction, "
                "so a duplicate means this is not that file -- and the pool's own population "
                "count, which is the denominator of every rate the export publishes, would "
                "be ambiguous")
        seen[key] = line_no
        questions.append((key, question))
    if not questions:
        raise ValueError(f"{path}: names no question")
    return questions


REPORT_EMPTY_QUESTION_RULE = (
    "a declared question with no scored chain is emitted as a contract §2.2 rule 5 "
    "unavailable row (missing_reason set, expectations null); the export manifest reports "
    "it under missing_questions/unavailable_reason, its summary is null, and seal/validate "
    "refuse the pair -- such an export is a diagnostic, and the fix is to score the "
    "question, never to shrink the declared set")


READ_SPLIT_DERIVE_VERSUS_CHECK = (
    "the population is the pool's REGISTERED READ SPLIT; a question manifest "
    "materializes it and is checked against it, and a scored dump is only ever "
    "checked against it -- a population derived from a dump silently drops every "
    "question whose chains failed to score")


def _require_registered_read_split(keys, registered, *, pool: str, path: str) -> str:
    n_registered = registered["n_questions"]
    expected = registered["qkeys_sha256"]
    if len(keys) != n_registered:
        if len(keys) > n_registered:
            extra = len(keys) - n_registered
            hint = (f"This manifest is a SUPERSET of the read split: {extra} of its "
                    f"questions are not in the pool. That is what a PARENT pool manifest "
                    f"looks like. Exporting against it would declare {extra} questions "
                    f"the pool never held, emit each as a contract §2.2 rule 5 "
                    f"unavailable row, and produce a diagnostic that seal/validate "
                    f"refuse. ")
        else:
            short = n_registered - len(keys)
            hint = (f"This manifest is {short} questions SHORT of the read split, so the "
                    f"declared set would silently shrink the denominator of every rate "
                    f"the export publishes. ")
        raise ValueError(
            f"{path} names {len(keys)} questions, but the REGISTERED READ SPLIT of pool "
            f"{pool!r} is {n_registered} questions (qkeys_sha256 {expected}). {hint}"
            f"A report pool's population is its registered read split "
            f"({rs.REGISTRY_ID}), never whichever question manifest a caller hands the "
            f"exporter. Point --pool-manifest at this pool's READ-SPLIT manifest; if the "
            f"read split itself changed, it is re-registered there first.")
    observed = rs.qkeys_sha256(keys)
    if observed != expected:
        raise ValueError(
            f"{path} names {len(keys)} questions -- the registered count for pool "
            f"{pool!r} -- but they hash to {observed}, not the REGISTERED {expected} "
            f"({rs.QKEY_SHA256_CONVENTION}). Same size, different questions: this file "
            f"is not this pool's read split, and a set published from it would carry the "
            f"pool's label over another pool's population.")
    return observed


def _check_dump_against_read_split(path, keys, *, pool: str) -> dict:
    path = Path(path)
    row = {"pool": pool, "path": str(path.resolve()), "sha256": _sha256(path),
           "error": None}
    try:
        dump = {chain["qkey"] for chain in load_chains(path)}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {**row, "error": f"{type(exc).__name__}: {exc}", "n_questions": None,
                "registered_only": None, "dump_only": None, "qkeys_sha256": None,
                "matches_registered_read_split": False}
    registered = set(keys)
    return {**row, "n_questions": len(dump),
            "registered_only": len(registered - dump),
            "dump_only": len(dump - registered),
            "qkeys_sha256": rs.qkeys_sha256(sorted(dump)) if dump else None,
            "matches_registered_read_split": dump == registered}


def derive_report_question_set(*, pool_manifest, expected_manifest_sha256: str,
                               pool: str, out, scored_dumps=()) -> dict:
    registered = rs.registered_read_split(pool)
    digest = str(expected_manifest_sha256 or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"--pool-manifest-sha256 must be 64 hex chars, got "
                         f"{expected_manifest_sha256!r}")
    binding = _file_binding(pool_manifest, label="read-split pool question manifest")
    if binding["sha256"] != digest:
        raise ValueError(
            f"{binding['path']} hashes {binding['sha256']}, not the registered "
            f"{digest}. The declared population must be materialized from the manifest "
            "the pool was SCORED against -- the same digest every exported pool row "
            "carries and the scored dump records -- so a set built from another copy "
            "could not be published against it anyway")
    questions = _report_manifest_questions(binding["path"])
    keys = sorted(key for key, _question in questions)
    read_split_sha = _require_registered_read_split(
        keys, registered, pool=pool, path=binding["path"])
    coverage = [_check_dump_against_read_split(dump, keys, pool=pool)
                for dump in (scored_dumps or ())]
    out = Path(out)
    _atomic_write(out, ("\n".join(keys) + "\n").encode("utf-8"))
    published_sha = _sha256(out)
    if published_sha != read_split_sha:
        raise ValueError(f"published question set hashes {published_sha}, not the "
                         f"registered read split {read_split_sha}")
    return {"schema": REPORT_QUESTION_SET_SCHEMA, "pool": pool,
            "manifest": binding, "population": REGISTERED_READ_SPLIT_POPULATION,
            "read_split": {"pool": pool, "n_questions": registered["n_questions"],
                           "qkeys_sha256": registered["qkeys_sha256"],
                           "source": rs.REGISTRY_ID},
            "n_questions": len(keys), "published": str(out.resolve()),
            "question_set_sha256": published_sha,
            "dump_coverage": coverage,
            "derive_versus_check": READ_SPLIT_DERIVE_VERSUS_CHECK,
            "empty_question_rule": REPORT_EMPTY_QUESTION_RULE}


def cmd_report_question_set(args) -> int:
    try:
        report = derive_report_question_set(
            pool_manifest=args.pool_manifest,
            expected_manifest_sha256=args.pool_manifest_sha256,
            pool=args.pool, out=args.out,
            scored_dumps=args.scored_dump or ())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"declared report question set REFUSED: {exc}") from None
    print(json.dumps({"event": "report_question_set_declared", **report}), flush=True)
    return 0


def cmd_verify_lane(args) -> int:
    try:
        report = verify_lane_dir(
            args.lane_dir,
            expected_commit=args.expected_commit,
            expected_seed=args.expected_seed,
            expected_lambda=args.expected_lambda,
            expected_faithful=args.expected_faithful == "1",
            expected_training=_expected_training_from_args(args, records=False),
            require_head_mode=args.require_head_mode)
    except (OSError, ValueError, KeyError, TypeError, AttributeError,
            json.JSONDecodeError) as exc:
        raise SystemExit(f"lane verification FAILED for {args.lane_dir}: {exc}") from None
    identity = report["training_identity"]
    if args.out:
        _atomic_write(Path(args.out), _json_bytes(report))
    print(json.dumps({"event": "lane_verified", "lane_dir": report["lane_dir"],
                      "head_mode": identity["head_mode"], "targets": identity["targets"],
                      "proof": identity["proof"], "out": args.out}), flush=True)
    return 0


def shard_identity(provenance):
    if len(provenance) == 1:
        return provenance[0]["path"], provenance[0]["sha256"]
    digest = hashlib.sha256()
    for line in sorted(f"{pr['path']}:{pr['sha256']}" for pr in provenance):
        digest.update(line.encode() + b"\n")
    return "; ".join(sorted(pr["path"] for pr in provenance)), digest.hexdigest()


POOL_SPLIT_LANE_DUMP = {"select": "select_scored.jsonl"}


def sanctioned_split_labels(pool) -> set:
    return {"select", pool} if isinstance(pool, str) and pool else {"select"}


def read_split_label_declaration(split, label, *, pool):
    if label is None:
        return split
    if not isinstance(label, str) or not label:
        raise ValueError(
            f"--read-split-label {label!r} is not a non-empty string. It declares the LABEL the "
            "read rows carry, which is not the same thing as --split (the binding selector); "
            "omit the flag entirely when the two are the same rather than passing an empty "
            "value, because an empty one is a mis-expanded shell variable.")
    allowed = sanctioned_split_labels(pool)
    if label not in allowed:
        raise ValueError(
            f"--read-split-label {label!r} is none of {sorted(allowed)}. `read_split` accepts a "
            "pool's own rows under `select` or under the pool's own id and nothing else, so a "
            "third label would file these rows on a split no consumer reads -- the export would "
            "seal cleanly and then be invisible to every table built over it.")
    return label


def split_selector_declaration(expected_split, selector, *, pool):
    if selector is None:
        return expected_split
    if not isinstance(selector, str) or not selector:
        raise ValueError(
            f"--expected-split-selector {selector!r} is not a non-empty string. It declares the "
            "`--split` the export was PRODUCED with, which is what decides whether the lane-gated "
            "or the report-pool binding is recomputed; omit the flag when it equals "
            "--expected-split rather than passing an empty value.")
    allowed = sanctioned_split_labels(pool)
    if selector not in allowed:
        raise ValueError(
            f"--expected-split-selector {selector!r} is none of {sorted(allowed)}. The selector "
            "is `select` for the lane's own training pool and the pool's own id for a report "
            "pool; any other value would recompute a binding this export was never produced "
            "under and refuse it for a reason that is not true of it.")
    return selector


def _with_read_split_label(binding, label):
    if label is None or binding.get("split") == label:
        return binding
    return {**binding, "split": label}

LANE_GATED_BINDING = "lane_gated_dump"
REPORT_POOL_BINDING = "report_pool_scored_dump"
POOL_DUMP_BINDING_SCHEMA = "ws-f-osp-pool-dump-binding/v1"

SCORE_POOL_DUMP_NAME = "select_scored.jsonl"
SCORE_POOL_SUMMARY_NAME = "summary.json"
SCORE_POOL_READOUT = "score_pool"
SCORE_POOL_TOP2_KIND = "top2"


def _lane_val_identity(lane_report) -> dict:
    binding = (lane_report.get("sources") or {}).get("summary.json") or {}
    raw = Path(binding["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise SystemExit(f"{binding['path']} changed under the lane gate: its digest is no "
                         "longer the one the gate hashed, so nothing read from it now is "
                         "bound to the verified lane")
    prov = (json.loads(raw.decode("utf-8")).get("provenance") or {})
    banks = {r.get("sha256") for r in (prov.get("fit_chains") or [])
             if isinstance(r, dict) and r.get("sha256")}
    questions = [q.get("sha256") for q in (prov.get("questions") or [])
                 if isinstance(q, dict) and q.get("sha256")]
    if len(banks) != 1 or len(questions) != 1:
        raise SystemExit(
            f"the verified lane names {len(banks)} fit chain bank(s) and {len(questions)} "
            "question manifest(s); its validation split has no single source, so the val "
            "rows' bank must be declared with --val-chain-bank-sha256 / --val-manifest-sha256")
    return {"chain_bank_sha256": banks.pop(), "manifest_sha256": questions[0]}


def _val_identity(args, *, lane_gated: bool, lane_report=None) -> tuple[dict, str]:
    declared = (args.val_chain_bank_sha256, args.val_manifest_sha256)
    if any(declared) and not all(declared):
        raise SystemExit(
            "--val-chain-bank-sha256 and --val-manifest-sha256 go together: half a chain "
            "identity is not a declaration, and the missing half would silently inherit the "
            "POOL's value while the other half named a different bank")
    pool_identity = {"chain_bank_sha256": args.chain_bank_sha256,
                     "manifest_sha256": args.manifest_sha256}
    lane_identity = _lane_val_identity(lane_report) if lane_gated and args.val_scored else None
    if all(declared):
        identity = {"chain_bank_sha256": args.val_chain_bank_sha256,
                    "manifest_sha256": args.val_manifest_sha256}
        if lane_identity is not None and identity != lane_identity:
            raise SystemExit(
                f"--val-chain-bank-sha256 / --val-manifest-sha256 declare {identity}, but the "
                f"verified lane records its validation source as {lane_identity}. The lane's "
                "val_scored.jsonl is the held-out FIT split; a declaration that names another "
                "bank is refused against the lane's bytes rather than believed.")
        return (identity, "--val-chain-bank-sha256 / --val-manifest-sha256")
    if not args.val_scored:
        return (pool_identity, "no val split emitted")
    if not lane_gated:
        raise SystemExit(
            "--val-scored without --lane-summary needs --val-chain-bank-sha256 and "
            "--val-manifest-sha256: the val dump's bank cannot be read off its chains, and it "
            "is NOT the pool's -- neither when `iclr_score_pool` copied an in-distribution val "
            "dump into a report pool's run dir, nor on the training pool itself, where the "
            "val split is held-out FIT and the pool is select. A lane-gated export derives it "
            "from the lane's own provenance; this one must say which bank its validation rows "
            "came from, because honest selection happens on them.")
    return (lane_identity, "derived from the verified lane's fit chain bank and question "
                           "manifest")


def _lane_dir_of_summary(lane_summary) -> Path:
    path = Path(lane_summary)
    if path.name != "summary.json":
        raise ValueError(
            f"--lane-summary must be a lane's own summary.json, got {path.name!r}: the lane "
            "gate runs on the directory containing it")
    return _require_canonical_lane_dir(path.parent)


def _lane_dump_binding(split: str, lane_report) -> dict:
    source = lane_report["sources"][POOL_SPLIT_LANE_DUMP[split]]
    return {
        "schema": POOL_DUMP_BINDING_SCHEMA,
        "kind": LANE_GATED_BINDING,
        "split": split,
        "dump": dict(source),
        "source_run": lane_report["lane_dir"],
        "bound_by": ("the lane gate, which hashed these bytes as one of the lane's own "
                     "three sources and reconciled them against the trainer's own "
                     "partition of the recorded banks"),
    }


def _require_report_pool_scores_are_a_stop_probability(config, *, run_dir) -> None:
    kind, head_mode = config.get("checkpoint_kind"), config.get("head_mode")
    targets = config.get("targets")
    if config.get("readout") != SCORE_POOL_READOUT:
        raise SystemExit(
            f"{run_dir}: config.readout is {config.get('readout')!r}, not "
            f"{SCORE_POOL_READOUT!r}. A report-pool dump is written by "
            "`iclr_score_pool`; a directory that does not say so is not one, and this "
            "path binds nothing it cannot identify.")
    if kind != SCORE_POOL_TOP2_KIND or head_mode != "osp" or targets != ["q", "v"]:
        raise SystemExit(
            f"{run_dir}: this dump declares checkpoint_kind={kind!r}, head_mode="
            f"{head_mode!r}, targets={targets!r}; the OSP survival-product replay reads only "
            f"checkpoint_kind={SCORE_POOL_TOP2_KIND!r}, head_mode='osp', targets=['q','v'], "
            "where `q` is a STOP PROBABILITY. A belief checkpoint scored into the same three "
            "file names writes a CORRECTNESS BELIEF into `q`; those dumps are read by "
            "iclr_ladder, never here.")


def _verify_report_pool_dump(scored, *, lane_report, split: str, expected_seed: int,
                             expected_chain_bank_sha256: str,
                             expected_manifest_sha256: str) -> dict:
    resolved = [Path(path) for path in scored or ()]
    if len(resolved) != 1:
        raise SystemExit(
            f"--split {split!r} is a report pool, so --scored must name exactly ONE "
            f"separately scored dump; got {[str(p) for p in resolved]}. A shard set has no "
            "single `source_run` and therefore nothing to bind to the lane.")
    dump_path = resolved[0].resolve()
    if dump_path.name != SCORE_POOL_DUMP_NAME:
        raise SystemExit(
            f"--scored {dump_path} is not a report-pool dump: `iclr_score_pool` always writes "
            f"{SCORE_POOL_DUMP_NAME!r} (the name is the SCORER's split; the pool is the "
            "directory), and the run dir beside it is what carries the binding.")
    try:
        run_dir = _require_canonical_lane_dir(dump_path.parent,
                                              label="scored pool run dir")
        dump = _file_binding(
            _require_canonical_lane_source(run_dir, SCORE_POOL_DUMP_NAME,
                                           label="scored pool dump"),
            label="scored pool dump")
        summary_path = _require_canonical_lane_source(run_dir, SCORE_POOL_SUMMARY_NAME,
                                                      label="scored pool summary")
        summary_bytes = _read_nofollow(summary_path)
        summary = json.loads(summary_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"--scored is not inside a readable scored pool run dir: {exc}") from None
    config = _mapping(_mapping(summary).get("config"))
    provenance = _mapping(_mapping(summary).get("provenance"))
    _require_report_pool_scores_are_a_stop_probability(config, run_dir=run_dir)

    declared_run = config.get("source_run")
    if not isinstance(declared_run, str) or not declared_run:
        raise SystemExit(
            f"{run_dir}/{SCORE_POOL_SUMMARY_NAME} records no config.source_run. That field is "
            "the ONLY thing that says which lane's checkpoint scored this pool, and the lane "
            "gate never saw these bytes, so there is nothing else to bind them with.")
    if not Path(declared_run).is_absolute():
        raise SystemExit(
            f"{run_dir}: config.source_run {declared_run!r} is a relative path. It would be "
            "resolved against THIS process's working directory, which has nothing to do with "
            "the scoring job's, so it could name a different lane in each reader. Re-score "
            "with an absolute --source-run.")
    if provenance.get("source_run") not in (None, declared_run):
        raise SystemExit(
            f"{run_dir}: config.source_run {declared_run!r} and provenance.source_run "
            f"{provenance.get('source_run')!r} name different runs; a dump that disagrees with "
            "itself about its own producer binds to neither.")
    lane_dir = Path(lane_report["lane_dir"])
    if Path(declared_run).resolve() != lane_dir:
        raise SystemExit(
            f"report-pool binding REFUSED: {run_dir}/{SCORE_POOL_SUMMARY_NAME} records "
            f"source_run={declared_run!r}, but --lane-summary verified {lane_dir}. This dump "
            "was scored from another training run's checkpoint; publishing it under this "
            "lane's verified identity is exactly the substitution this binding exists to "
            "catch, and the lane gate never hashed these bytes so nothing else would have "
            "noticed.")
    if config.get("seed") != expected_seed:
        raise SystemExit(
            f"{run_dir}: config.seed {config.get('seed')!r} != --seed {expected_seed!r}. "
            "`iclr_score_pool` copies the seed out of the source run, so this dump and the "
            "export disagree about which run produced it.")
    chains = _source_core(provenance.get("chains"), label="scored pool chains")
    if chains["sha256"] != expected_chain_bank_sha256:
        raise SystemExit(
            f"report-pool binding REFUSED: {run_dir} scored the chain bank "
            f"{chains['sha256']} ({chains['path']}), but this export stamps "
            f"--chain-bank-sha256 {expected_chain_bank_sha256} on every pool row. The bank "
            "digest is what proves this method read the same chains as every other method on "
            "the pool; a dump of a different bank filed under this one is unmergeable.")
    questions = _source_core(provenance.get("questions"), label="scored pool questions")
    if questions["sha256"] != expected_manifest_sha256:
        raise SystemExit(
            f"report-pool binding REFUSED: {run_dir} scored against question manifest "
            f"{questions['sha256']}, but this export stamps --manifest-sha256 "
            f"{expected_manifest_sha256}.")
    lane_val = lane_report["sources"]["val_scored.jsonl"]
    try:
        copied = _file_binding(
            _require_canonical_lane_source(run_dir, "val_scored.jsonl",
                                           label="scored pool val copy"),
            label="scored pool val copy")
    except ValueError as exc:
        raise SystemExit(
            f"report-pool binding REFUSED: {exc}. `iclr_score_pool` always copies the source "
            "run's val_scored.jsonl in beside the scored pool; a run dir without one was not "
            "written by it, and its claim to have been scored from this lane is unsupported.")
    if copied["sha256"] != lane_val["sha256"]:
        raise SystemExit(
            f"report-pool binding REFUSED: {run_dir}/val_scored.jsonl hashes "
            f"{copied['sha256']} but the verified lane's is {lane_val['sha256']}. That file is "
            f"a COPY of the source run's val split, so a dump claiming source_run="
            f"{declared_run!r} whose copy is not this lane's bytes did not come from this lane "
            "in the state the gate just verified.")
    checkpoint = _source_core(provenance.get("osp_checkpoint"),
                              label="scored pool osp checkpoint")
    if Path(checkpoint["path"]) != lane_dir / "osp_best.pt":
        raise SystemExit(
            f"report-pool binding REFUSED: {run_dir} records osp_checkpoint "
            f"{checkpoint['path']}, not {lane_dir / 'osp_best.pt'}. `iclr_score_pool` always "
            "reads the checkpoint out of its --source-run, so a dump naming another file was "
            "not produced the way this binding assumes.")
    counts = {}
    for key in ("n_select_chains", "n_select_questions"):
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise SystemExit(
                f"{run_dir}: config.{key} is {value!r}, expected a positive integer. The "
                "scorer records what it wrote; without it a dump that lost rows afterwards "
                "would replay a smaller pool and move this pool's natural anchor.")
        counts[key] = value
    return {
        "schema": POOL_DUMP_BINDING_SCHEMA,
        "kind": REPORT_POOL_BINDING,
        "split": split,
        "dump": dump,
        "run_dir": str(run_dir),
        "summary": {"path": str(summary_path.resolve()),
                    "sha256": hashlib.sha256(summary_bytes).hexdigest()},
        "source_run": str(lane_dir),
        "chain_bank_sha256": chains["sha256"],
        "manifest_sha256": questions["sha256"],
        "chains_source": chains,
        "questions_source": questions,
        "osp_checkpoint": checkpoint,
        "lane_val_copy_sha256": copied["sha256"],
        **counts,
        "bound_by": ("the dump's own config.source_run naming the verified lane, plus the "
                     "scored pool's chain-bank and question-manifest digests -- NOT the lane "
                     "gate, which never hashed these separately scored bytes"),
    }


def _require_report_pool_population(chains, binding, *, flag: str) -> None:
    if binding["kind"] != REPORT_POOL_BINDING:
        return
    observed = {"n_select_chains": len(chains),
                "n_select_questions": len({chain.get("qkey") for chain in chains})}
    bad = {key: (observed[key], binding[key]) for key in observed
           if observed[key] != binding[key]}
    if bad:
        raise SystemExit(
            f"{flag} replays {observed}, but {binding['run_dir']} recorded {bad} "
            "(observed, declared): the scored pool dump is not the population the scorer "
            "wrote, and a dump that lost chains moves this pool's natural anchor away from "
            "every other method plotted against it.")


LANE_DUMP_HINTS = {
    "--scored": ("A dump scored SEPARATELY from this lane's checkpoint is published under the "
                 f"{REPORT_POOL_BINDING} binding instead, by naming its own pool in --split; "
                 "it can never stand in for the lane's own."),
    "--val-scored": ("The val split is the lane's held-out FIT split on EVERY pool, including a "
                     "report pool, so pass the LANE's own val_scored.jsonl -- not the copy "
                     "`iclr_score_pool` places beside a scored pool, which is the same bytes at "
                     "a path the lane gate never verified."),
}


def _require_lane_dump_input(paths, lane_source, *, flag: str) -> None:
    resolved = [str(Path(path).resolve()) for path in paths or ()]
    if resolved != [lane_source["path"]]:
        raise SystemExit(
            f"{flag} must be exactly the verified lane's own dump "
            f"{lane_source['path']}, got {resolved}: this is the {LANE_GATED_BINDING} "
            "binding, a publishable export replays the chains THE LANE SCORED, and the lane "
            "gate proves only those bytes. A copy, a shard set or another lane's dump puts "
            f"the export outside everything that was verified. {LANE_DUMP_HINTS.get(flag, '')}")


def _require_lane_dump_bytes(provenance, lane_source, *, flag: str,
                             proof: str = "the lane gate") -> None:
    digests = [record["sha256"] for record in provenance]
    if digests != [lane_source["sha256"]]:
        raise SystemExit(
            f"{flag} hashes {digests}, but {proof} verified "
            f"{[lane_source['sha256']]}: the dump changed between verification and replay, so "
            "the exported rows are not the rows that were checked.")


def _records_lane_gate(args, read_label) -> tuple[dict, dict]:
    try:
        lane_dir = _lane_dir_of_summary(args.lane_summary)
        if not args.source_commit:
            raise ValueError(
                "--lane-summary requires --source-commit: the lane gate refuses any training "
                "input that was not recorded from that exact clean commit, and there is "
                "nothing to check it against otherwise")
        lane_commit = _lane_commit_declaration(args.source_commit, args.expected_lane_commit)
        report = verify_lane_dir(
            lane_dir, expected_commit=lane_commit, expected_seed=args.seed,
            expected_lambda=args.lam, expected_faithful=bool(args.faithful),
            expected_training=_expected_training_from_args(args, records=True),
            require_head_mode="osp")
    except (OSError, ValueError, KeyError, TypeError, AttributeError,
            json.JSONDecodeError) as exc:
        raise SystemExit(f"--lane-summary is not a verified training lane: {exc}") from None
    if args.split in POOL_SPLIT_LANE_DUMP:
        binding = _lane_dump_binding(args.split, report)
        _require_lane_dump_input(args.scored, binding["dump"], flag="--scored")
    else:
        binding = _verify_report_pool_dump(
            args.scored, lane_report=report, split=args.split, expected_seed=args.seed,
            expected_chain_bank_sha256=args.chain_bank_sha256,
            expected_manifest_sha256=args.manifest_sha256)
    if not args.val_scored:
        raise SystemExit(
            "--lane-summary requires --val-scored: a publishable export is the val+pool "
            "rectangle, and the lane's val_scored.jsonl is the only val dump the gate "
            "verified")
    _require_lane_dump_input(args.val_scored, report["sources"]["val_scored.jsonl"],
                             flag="--val-scored")
    return report, _with_read_split_label(binding, read_label)


def cmd_records(args) -> int:
    _require_pi_key(args)
    if args.split == "val":
        raise SystemExit("--split names the POOL split; val is emitted from --val-scored so "
                         "that one artifact carries the whole rectangle")
    try:
        read_label = read_split_label_declaration(args.split, args.read_split_label,
                                                  pool=args.pool)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if not args.allow_open_coverage:
        missing_flags = [f for f, v in (("--val-scored", args.val_scored),
                                        ("--question-set", args.question_set),
                                        ("--val-question-set", args.val_question_set))
                         if not v]
        if missing_flags:
            raise SystemExit(
                f"records needs {', '.join(missing_flags)}: contract §2.2 rule 7 requires "
                "rectangular coverage over BOTH val and the pool, and honest selection (§5.2) "
                "picks each method's operating point ON VAL, so a config with no val rows has "
                "no honest operating point at all. Pass both splits and both declared question "
                "sets, or --allow-open-coverage for a diagnostic that is not published.")
    training_identity, lane_sources, lane_report = None, None, None
    pool_binding = None
    if args.lane_summary:
        lane_report, pool_binding = _records_lane_gate(args, read_label)
        training_identity = lane_report["training_identity"]
        lane_sources = lane_report["sources"]
        args.model_id = training_identity["model"]
        args.model_revision = training_identity["model_revision"]
        args.tokenizer_revision = training_identity["model_revision"]
    chains, provenance = dedup_load(args.scored)
    if pool_binding is not None:
        _require_lane_dump_bytes(provenance, pool_binding["dump"], flag="--scored",
                                 proof=pool_binding["bound_by"])
        _require_report_pool_population(chains, pool_binding, flag="--scored")
    _require_osp_scores(chains, label="--scored")
    if args.faithful:
        before = len(chains)
        chains = R.faithful_subsample(chains)
        print(json.dumps({"event": "faithful_subsample", "before": before,
                          "after": len(chains)}), flush=True)
    source_artifact, source_sha = shard_identity(provenance)
    frozen = {"lambda": args.lam, "replay_grade": args.grade,
              "expectation": "survival_product",
              "arm": "faithful_1_chain_per_question" if args.faithful else "all_chains"}
    grader = args.grader_id if args.grade == "loose" else f"{args.grader_id}+grade_{args.grade}"
    header = osp.replay_header(
        run_id=args.run_id, method_family=args.method_family, method_name=args.method_name,
        config_id=args.config_id or f"{args.method_name}:lam{args.lam:g}",
        knob={"name": args.knob_name, "value": args.lam}, seed=args.seed,
        dataset=args.dataset, split=read_label, model_id=args.model_id,
        model_revision=args.model_revision, tokenizer_revision=args.tokenizer_revision,
        prompt_contract_id=args.prompt_contract_id, grader_id=grader,
        selection_split=args.selection_split, selection_rule=args.selection_rule,
        frozen_config=frozen, source_branch=args.source_branch,
        source_commit=args.source_commit or provenance[0]["commit"],
        source_artifact=source_artifact, source_sha256=source_sha,
        chain_bank_sha256=args.chain_bank_sha256,
        manifest_sha256=args.manifest_sha256, pool=args.pool)

    identity = {"chain_bank_sha256": args.chain_bank_sha256,
                "manifest_sha256": args.manifest_sha256, "pool": args.pool}
    val_identity, val_identity_source = _val_identity(args, lane_gated=lane_sources is not None,
                                                      lane_report=lane_report)
    natural_fn = natural_answer_fn(args.answer_format, args.force_natural_answer)

    def emit(split_chains, split_name, question_set, split_provenance):
        artifact, sha = shard_identity(split_provenance)
        split_header = {**header, "split": split_name,
                        "source_artifact": artifact, "source_sha256": sha}
        if split_name == "val":
            split_header = {**split_header, **val_identity}
            if val_identity != {k: header[k] for k in val_identity}:
                split_header["frozen_config"] = {
                    **header["frozen_config"],
                    "split_input": {"carries_split": "val", "from": "--val-scored",
                                    **val_identity,
                                    "note": ("`pool`/`dataset` name the bank this ARTIFACT is "
                                             "filed under; the val split's chains came from "
                                             "the bank named here")}}
        try:
            rows = R.replay_rows(split_chains, header=split_header, pi_key=args.pi_key,
                                 grade=args.grade, natural_answer_tokens=natural_fn,
                                 aux_generated_tokens=args.aux_generated_tokens)
        except ValueError as exc:
            raise SystemExit(f"{split_name}: {exc}") from None
        gaps = []
        if question_set:
            expected = load_question_set(question_set)
            covered = {r["qkey"] for r in rows}
            extra = covered - expected
            if extra:
                raise SystemExit(f"{split_name}: replay covers {len(extra)} questions outside "
                                 f"the declared set, e.g. {sorted(extra)[:3]} — wrong pool?")
            gaps = sorted(expected - covered)
            rows.extend(R.unavailable_rows(gaps, header=split_header,
                                           reason="no scored chains for this question"))
            rows.sort(key=lambda r: r["qkey"])
        return rows, gaps

    rows, missing = emit(chains, read_label, args.question_set, provenance)
    val_rows, val_missing = [], []
    val_provenance = []
    if args.val_scored:
        val_chains, val_provenance = dedup_load(args.val_scored)
        if lane_sources is not None:
            _require_lane_dump_bytes(val_provenance, lane_sources["val_scored.jsonl"],
                                     flag="--val-scored")
        _require_osp_scores(val_chains, label="--val-scored")
        val_rows, val_missing = emit(val_chains, "val", args.val_question_set, val_provenance)
    expected = load_question_set(args.question_set) if args.question_set else set()
    question_sets = {}
    if args.val_question_set:
        question_sets["val"] = source_record(args.val_question_set)
    if args.question_set:
        question_sets[read_label] = source_record(args.question_set)

    out = Path(args.out)
    rows_payload = _jsonl_bytes(val_rows + rows)
    _atomic_write(out, rows_payload)
    record = {
        "schema_version": osp.REPLAY_SCHEMA, **header, **identity,
        "grading": args.grade, "pi_key": args.pi_key, "expectation": "survival_product",
        "answer_format": args.answer_format,
        "force_natural_answer": bool(args.force_natural_answer),
        "natural_answer_tokens": "archived_math_zero",
        "aux_generated_tokens": args.aux_generated_tokens,
        "overhead_accounting": "none",
        "faithful_subsample": bool(args.faithful),
        "coverage_declared": bool(args.question_set and args.val_question_set
                                  and args.val_scored),
        "question_sets": question_sets,
        "qset_coverage_sha256": _qset_binding(question_sets),
        "splits_emitted": ([read_label] if not val_rows else ["val", read_label]),
        "split_chain_identity": {
            "val": {**val_identity, "source": val_identity_source},
            read_label: {"chain_bank_sha256": args.chain_bank_sha256,
                         "manifest_sha256": args.manifest_sha256,
                         "source": "--chain-bank-sha256 / --manifest-sha256"}},
        "n_val_rows": len(val_rows), "n_val_missing_questions": len(val_missing),
        "val_inputs": val_provenance,
        "pool_dump_binding": pool_binding,
        "n_chains": len(chains), "n_rows": len(rows), "n_missing_questions": len(missing),
        "missing_questions": missing[:20],
        "summary_basis": "question",
        "captions": R.mandatory_captions(
            arm=("faithful_1_chain_per_question" if args.faithful else "all_chains"),
            faithful=bool(args.faithful), n_lambda=5, n_seeds=2,
            reader=(training_identity or {}).get("reader")),
        "training_identity": training_identity,
        "summary": None if missing else R.question_summary(rows),
        "summary_chain_weighted": None if missing else osp.replay_summary(rows),
        "val_summary": (None if (val_missing or not val_rows)
                        else R.question_summary(val_rows)),
        "val_summary_chain_weighted": (None if (val_missing or not val_rows)
                                       else osp.replay_summary(val_rows)),
        "unavailable_reason": (None if not missing else
                               f"{len(missing)} of {len(expected)} declared questions have no "
                               "scored chains; the covered subset is biased"),
        "partial_summary_do_not_report": (None if not missing else
                                          R.question_summary(rows, allow_partial=True)),
        "natural": osp.natural_summary(chains),
        "output": {"path": str(out), "sha256": hashlib.sha256(rows_payload).hexdigest()},
        "inputs": provenance,
    }
    natural_tokens = record["natural"]["mean_thinking_tokens"]
    for key in ("summary", "summary_chain_weighted", "partial_summary_do_not_report"):
        point = record.get(key)
        if point:
            point["thinking_saving"] = (1.0 - point["mean_thinking_tokens"] / natural_tokens
                                        if natural_tokens else 0.0)
    _atomic_write(export_manifest_path(out), _json_bytes(record))
    print(json.dumps(record))
    return 0


def cmd_rundir(args) -> int:
    if args.overhead_accounting == "unknown":
        raise SystemExit("--overhead-accounting unknown: a method that cannot be placed on the "
                         "total-cost axis is reported overhead-unknown, never exported")
    _require_pi_key(args)
    select, select_prov = dedup_load(args.scored)
    _require_osp_scores(select, label="--scored")
    val, val_prov = dedup_load(args.val_scored)
    _require_osp_scores(val, label="--val-scored")

    if args.arm == "strengthened":
        def score_fn(chain):
            return R.strengthened_scores(chain, pi_key=args.pi_key)
    else:
        def score_fn(chain):
            return R.bernoulli_scores(chain, seed=args.seed, pi_key=args.pi_key)

    try:
        val_rows = R.apply_scores(val, score_fn)
        select_rows = R.apply_scores(select, score_fn)
    except ValueError as exc:
        raise SystemExit(f"--scored: {exc}") from None

    config = R.arm_config(args.arm, lam=args.lam, seed=args.seed, pi_key=args.pi_key)
    summary = sio.summary_stub(
        seed=args.seed, targets=["q", "v"], method=config.pop("method"),
        provenance={"select": select_prov, "val": val_prov,
                    "overhead_accounting": args.overhead_accounting,
                    "overhead_tokens_per_boundary": 0.0,
                    "answer_format": "boxed", "force_natural_answer": False},
        **config)
    result = sio.write_run_dir(args.out, val=val_rows, select=select_rows, summary=summary)
    print(json.dumps({"arm": args.arm, **result}))
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.cmd == "records":
        return cmd_records(args)
    if args.cmd == "rundir":
        return cmd_rundir(args)
    if args.cmd == "verify-lane":
        return cmd_verify_lane(args)
    if args.cmd == "question-sets":
        return cmd_question_sets(args)
    if args.cmd == "report-question-set":
        return cmd_report_question_set(args)
    expected_training = _expected_training_from_args(args, records=False)
    expected = {
        "lane_dir": args.lane_dir,
        "expected_commit": args.expected_commit,
        "expected_lane_commit": args.expected_lane_commit,
        "expected_source_branch": args.expected_source_branch,
        "producer_script": args.producer_script,
        "expected_method": args.expected_method,
        "expected_dataset": args.expected_dataset,
        "expected_split": args.expected_split,
        "expected_split_selector": args.expected_split_selector,
        "expected_faithful": args.expected_faithful == "1",
        "expected_manifest_path": args.expected_manifest_path,
        "expected_manifest_sha256": args.expected_manifest_sha256,
        "expected_fit_chains": args.expected_fit_chains,
        "expected_select_chains": args.expected_select_chains,
        "expected_chain_bank_sha256": args.expected_chain_bank_sha256,
        "expected_pool": args.expected_pool,
        "expected_training_chain_bank_sha256": args.expected_training_chain_bank_sha256,
        "expected_training_manifest_sha256": args.expected_training_manifest_sha256,
        "expected_val_count": args.expected_val_count,
        "expected_val_salt": args.expected_val_salt,
        "expected_max_chains_per_question": args.expected_max_chains_per_question,
        "expected_max_chain_tokens": args.expected_max_chain_tokens,
        "expected_max_train_questions": args.expected_max_train_questions,
        "expected_max_seq_len": args.expected_max_seq_len,
        "expected_run_id": args.expected_run_id,
        "expected_seed": args.expected_seed,
        "expected_lambda": args.expected_lambda,
        "expected_training_config": expected_training,
    }
    result = (validate_export(args.records, marker=args.marker, **expected)
              if args.cmd == "validate" else
              seal_export(args.records, args.marker, **expected))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
