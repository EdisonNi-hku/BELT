"""Train the OS-Pruner reader with the top two blocks unfrozen."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from rl_controller.baselines.ospruner import distributed as D
from rl_controller.baselines.ospruner import layout as L
from rl_controller.baselines.ospruner import training as T
from rl_controller.baselines.ospruner.reader import MODEL_ID, MODEL_REVISION, TopTwoReader
from rl_controller.iclr import learning_curve as lc
from rl_controller.iclr import osp_policy as osp
from rl_controller.iclr import scoring_io as sio
from rl_controller.iclr.provenance import source_record

TRAINING_POPULATION_SCHEMA = "ws-f-training-population/v1"
TRAINING_CONFIG_SCHEMA = "ws-f-osp-training-config/v1"
TRAINING_ARGS_DIGEST_SCHEMA = "ws-f-osp-training-args-digest/v1"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit-chains", required=True, nargs="+")
    p.add_argument("--select-chains", required=True)
    p.add_argument("--questions", required=True, nargs="+")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--model-path", default=None)
    p.add_argument("--revision", default=MODEL_REVISION)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--n-unfrozen", type=int, default=L.DEFAULT_N_UNFROZEN,
                   help="trainable top blocks; 2 mirrors the paper")
    p.add_argument("--no-fp32-trainable", action="store_true",
                   help="keep the trainable blocks in bf16 (default: fp32 master weights)")
    p.add_argument("--max-seq-len", type=int, default=16384,
                   help="reader context budget. Chains longer than this are DROPPED from "
                        "train (counted and reported) and are a hard error in the val/select "
                        "dumps, where a missing chain would move the natural anchor. Never "
                        "truncated: truncation shifts every later block end, so the boundary "
                        "gather would read the wrong tokens and still produce a plausible "
                        "loss. The default covers the pool's 12k thinking cap plus a prompt")
    p.add_argument("--head-mode", choices=("osp", "qv"), default="osp")
    p.add_argument("--osp-lambda", type=float, default=1e-4)
    p.add_argument("--osp-lambda-anneal", default="",
                   help="`geometric:<mult>:<epochs>` | `constant` | '' for the registered "
                        "default (geometric:10:4)")
    p.add_argument("--osp-token-scale", type=float, default=osp.DEFAULT_TOKEN_SCALE)
    p.add_argument("--osp-cost", choices=osp.COST_AXES, default="thinking",
                   help="cost axis the OSP objective is charged in; `thinking` (default) is "
                        "what the WS-F lanes trained under, `total` adds the forced answer at "
                        "a probe and the archived-math natural answer at the natural end")
    p.add_argument("--osp-init-fire-rate", type=float, default=0.1)
    p.add_argument("--osp-grade", choices=osp.GRADINGS, default="strict",
                   help="TRAINING grades type-strictly, the notion the q target uses "
                        "(prereg-ws-b §4). The REPLAY grades loose; that is a different flag "
                        "on a different tool")
    p.add_argument("--label-dump", default=None,
                   help="--head-mode qv: the WS-B dump to train on, consumed via "
                        "label_dump.read_dump (which verifies the certified sha256)")
    p.add_argument("--targets", default="q,v", help="--head-mode qv target names")
    p.add_argument("--lr", type=float, default=1e-5, help="their published AdamW lr")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--chains-per-step", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="continue from OUT_DIR/osp_last.pt if it exists (operating rule 5). "
                        "Refuses to resume into a different population or a different set of "
                        "run-defining flags. Safe to pass always: with no checkpoint it is a "
                        "cold start, and on a finished lane it re-enters at scoring")
    p.add_argument("--migrate-from-commit", default="",
                   help="declare the CODE commit that produced OUT_DIR/osp_last.pt, to resume "
                        "a lane across a code change. The declaration is VERIFIED against the "
                        "checkpoint's own args digest: the digest must be reproduced from the "
                        "inputs' current bytes with only the producer's commit/dirty "
                        "substituted, so the population, price, schedule, targets, global "
                        "batch and optimiser settings are all proven identical. The original "
                        "checkpoint is archived and both producers are recorded in "
                        "resume_migration.json. It can never forgive anything but the commit. "
                        f"Pass {NO_COMMIT_DECLARATION!r} to declare a producer whose records "
                        "carry no commit at all (a tree outside a repository); that claim is "
                        "verified the same way. Safe to pass always: it is a no-op when the "
                        "digests already agree")
    p.add_argument("--dist-timeout-s", type=int, default=D.DEFAULT_TIMEOUT_S,
                   help="collective timeout under torchrun. NOT part of the args digest: it "
                        "is an execution knob, and a lane must resume across world sizes")
    p.add_argument("--run-cluster", default="",
                   help="which cluster this lane ran on (euler|clariden). Recorded in the run "
                        "dir's provenance: results are cluster-agnostic, but WHERE a lane ran "
                        "must be auditable when a sweep is split across queues")
    p.add_argument("--val-count", type=int, default=320)
    p.add_argument("--val-salt", default=lc.VAL_SALT)
    p.add_argument("--max-train-questions", type=int, default=0)
    p.add_argument("--max-chains-per-question", type=int, default=0)
    p.add_argument("--max-chain-tokens", type=int, default=0)
    p.add_argument("--allow-unscored", action="store_true",
                   help="permit a run whose val/select splits contain chains longer than "
                        "--max-seq-len, by EXCLUDING those chains from the dumps (they cannot "
                        "be written unscored: the ladder zips scores against probes). That "
                        "shrinks the dump and so moves the natural anchor, which is why the "
                        "run is stamped config.publishable=false. DIAGNOSTIC ONLY")
    p.add_argument("--faithful-subsample", action="store_true",
                   help="A3: keep one chain per question (their 1-trace/problem regime), "
                        "using the registered content-addressed draw")
    return p.parse_args(argv)


_RUNTIME = D.SINGLE


def set_runtime(runtime) -> None:
    global _RUNTIME
    _RUNTIME = runtime


def log(obj, *, every_rank: bool = False) -> None:
    if _RUNTIME.distributed and not _RUNTIME.is_writer and not every_rank:
        return
    if _RUNTIME.distributed and every_rank:
        obj = {"rank": _RUNTIME.rank, **obj}
    print(json.dumps(obj), flush=True)


def validate_args(args) -> None:
    if len(args.questions) != 1:
        raise SystemExit(
            f"--questions takes exactly ONE frozen question manifest, got "
            f"{len(args.questions)} ({list(args.questions)}). The exporter binds a lane to a "
            "single manifest (iclr_osp_export._lane_training_inputs requires "
            "provenance.questions to name one file), so a multi-manifest lane is "
            "unverifiable and unpublishable. Merge the pools into one manifest first.")


def training_population_attestation(args, *, train_digest: str,
                                    n_train_chains: int) -> dict:
    return {
        "schema": TRAINING_POPULATION_SCHEMA,
        "faithful_subsample": bool(args.faithful_subsample),
        "train_digest": train_digest,
        "n_train_chains": n_train_chains,
        "population_config": {
            "max_chains_per_question": args.max_chains_per_question,
            "val_count": args.val_count,
            "val_salt": args.val_salt,
            "max_chain_tokens": args.max_chain_tokens,
            "max_train_questions": args.max_train_questions,
            "max_seq_len": args.max_seq_len,
        },
    }


def chain_identities(chains) -> list[list]:
    return [[qkey, rep] for qkey, rep in
            sorted({(c["qkey"], int(c["rep"])) for c in chains})]


def over_max_seq_len_record(*, train_long, val_long, select_long, allow_unscored) -> dict:
    return {"train_dropped": len(train_long),
            "train_dropped_identities": chain_identities(train_long),
            "val_unscored": len(val_long),
            "select_unscored": len(select_long),
            "allow_unscored": bool(allow_unscored),
            "unscored_identities": chain_identities(val_long + select_long)}


def lane_is_publishable(over: dict) -> bool:
    return (over["val_unscored"] == 0 and over["select_unscored"] == 0
            and over["train_dropped"] == 0 and over["allow_unscored"] is False)


DEVICE_FIELDS = ("device", "gpu", "gpu_mem_gb", "gpu_compute_capability", "gpu_ranks")


def device_record(torch, device) -> dict:
    record = {"device": str(device), "gpu": None, "gpu_mem_gb": None,
              "gpu_compute_capability": None}
    if getattr(device, "type", None) != "cuda":
        return record
    props = torch.cuda.get_device_properties(device)
    return {**record, "gpu": props.name,
            "gpu_mem_gb": round(props.total_memory / 2 ** 30, 1),
            "gpu_compute_capability": f"{props.major}.{props.minor}"}


def gather_device_records(torch, runtime, record: dict) -> list:
    if not runtime.distributed:
        return [{"rank": int(runtime.rank), **record}]
    import torch.distributed as dist

    gathered = [None] * int(runtime.world)
    dist.all_gather_object(gathered, {"rank": int(runtime.rank), **record})
    return [dict(row) for row in sorted(gathered, key=lambda row: row["rank"])]


def training_device_fields(records: list) -> dict:
    distinct = sorted({(r["gpu"], r["gpu_compute_capability"]) for r in records},
                      key=lambda card: tuple("" if v is None else str(v) for v in card))
    if len(distinct) > 1:
        raise SystemExit(
            f"the ranks of this lane are not all on the same card: {distinct}. bf16 kernels are "
            "selected per GPU architecture, so a data-parallel world that mixes architectures "
            "sums two different kernels' gradients into one update -- that is two measurements, "
            "and the lane could only declare one of them. Refusing to train.")
    writer = records[0]
    return {"device": writer["device"], "gpu": writer["gpu"],
            "gpu_mem_gb": writer["gpu_mem_gb"],
            "gpu_compute_capability": writer["gpu_compute_capability"],
            "gpu_ranks": records}


def training_run_config(args, *, names: list[str], schedule,
                        input_provenance: dict) -> dict:
    return {
        "schema": TRAINING_CONFIG_SCHEMA,
        "lambda": args.osp_lambda, "anneal": schedule.as_dict(), "seed": args.seed,
        "head_mode": args.head_mode, "targets": list(names), "epochs": args.epochs,
        "patience": args.patience, "lr": args.lr, "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "grade": args.osp_grade, "token_scale": args.osp_token_scale,
        "cost": args.osp_cost,
        "init_fire_rate": args.osp_init_fire_rate,
        "max_chains_per_question": args.max_chains_per_question,
        "val_count": args.val_count, "val_salt": args.val_salt,
        "max_chain_tokens": args.max_chain_tokens,
        "max_train_questions": args.max_train_questions,
        "max_seq_len": args.max_seq_len,
        "faithful": bool(args.faithful_subsample), "n_unfrozen": args.n_unfrozen,
        "chains_per_step": args.chains_per_step,
        "allow_unscored": bool(args.allow_unscored),
        "dtype": args.dtype, "fp32_trainable": not args.no_fp32_trainable,
        "model": args.model, "model_path": args.model_path, "revision": args.revision,
        "inputs": input_provenance,
    }


def training_summary_identity(args, *, names: list[str], schedule,
                              input_provenance: dict, reader_manifest: dict,
                              training_config: dict) -> dict:
    return {
        "targets": list(names),
        "head_mode": args.head_mode,
        "osp_grade": args.osp_grade,
        "osp_cost": args.osp_cost,
        "osp_anneal": schedule.as_dict(),
        "reader": reader_manifest,
        "training_config": training_config,
    }


def build_training_summary(args, *, names: list[str], schedule,
                           input_provenance: dict, reader_manifest: dict,
                           training_config: dict, execution: dict | None = None,
                           **run_fields) -> dict:
    return sio.summary_stub(
        seed=args.seed,
        method="OSP-full" if args.head_mode == "osp" else "OSP-arch-cell",
        provenance={
            **input_provenance,
            **({"execution": execution} if execution is not None else {}),
            "model_revision": args.revision,
            "run_cluster": args.run_cluster or None,
            "overhead_accounting": "none",
            "overhead_tokens_per_boundary": 0.0,
            "answer_format": "boxed",
            "force_natural_answer": False,
        },
        knob={"name": "lambda", "value": args.osp_lambda},
        **training_summary_identity(
            args, names=names, schedule=schedule,
            input_provenance=input_provenance, reader_manifest=reader_manifest,
            training_config=training_config),
        **run_fields)


def training_args_digest_payload(training_config: dict) -> dict:
    payload = dict(training_config)
    payload.pop("schema", None)
    payload.pop("max_train_questions", None)
    return payload


RESUME_MIGRATION_SCHEMA = "ws-f-osp-resume-migration/v1"
RESUME_MIGRATION_FILE = "resume_migration.json"

MIGRATABLE_PROVENANCE_KEYS = ("commit", "dirty")

MIGRATION_VERIFIED_FIELDS = (
    "lambda", "anneal", "seed", "head_mode", "targets", "epochs", "patience",
    "lr", "weight_decay", "grad_clip", "grade", "token_scale", "cost",
    "init_fire_rate", "max_chains_per_question", "val_count", "val_salt",
    "max_chain_tokens", "max_seq_len", "faithful", "n_unfrozen",
    "chains_per_step", "allow_unscored", "dtype", "fp32_trainable",
    "model", "model_path", "revision",
)


def _json_copy(obj):
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def diff_paths(old, new, prefix: str = "") -> list:
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                out.append(f"{prefix}{key}")
            else:
                out.extend(diff_paths(old[key], new[key], f"{prefix}{key}."))
        return out
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{prefix}[len]"]
        out = []
        for index, (left, right) in enumerate(zip(old, new)):
            out.extend(diff_paths(left, right, f"{prefix}{index}."))
        return out
    return [] if old == new else [prefix.rstrip(".") or "<root>"]


def classify_payload_differences(old_payload: dict, new_payload: dict):
    code, blocking = [], []
    for path in diff_paths(old_payload, new_payload):
        parts = path.split(".")
        if parts[0] == "inputs" and parts[-1] in MIGRATABLE_PROVENANCE_KEYS:
            code.append(path)
        else:
            blocking.append(path)
    return code, blocking


def _walk_source_records(node):
    if isinstance(node, dict):
        if {"path", "sha256", "commit", "dirty"} <= set(node):
            yield node
            return
        for value in node.values():
            yield from _walk_source_records(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_source_records(value)


def producer_commits(payload: dict) -> list:
    seen = []
    for record in _walk_source_records(payload.get("inputs", {})):
        if record["commit"] not in seen:
            seen.append(record["commit"])
    return seen


def with_producer_code(payload: dict, *, commit, dirty) -> dict:
    out = _json_copy(payload)
    for record in _walk_source_records(out.get("inputs", {})):
        record["commit"] = commit
        record["dirty"] = dirty
    return out


def input_sha256s(payload: dict) -> dict:
    return {record["path"]: record["sha256"]
            for record in _walk_source_records(payload.get("inputs", {}))}


NO_COMMIT_DECLARATION = "none"


def declared_producer_commit(value: str):
    return None if str(value).strip().lower() == NO_COMMIT_DECLARATION else value


def _reproduce_old_payload(new_payload: dict, *, declared_commit, stored_digest: str,
                           digest_fn):
    for dirty in (False, True, None):
        candidate = with_producer_code(new_payload, commit=declared_commit, dirty=dirty)
        if digest_fn(candidate) == stored_digest:
            return candidate, dirty
    return None, None


def plan_resume_migration(state, *, new_payload: dict, new_args_digest: str,
                          train_digest: str, declared_commit: str,
                          declared_payload, digest_fn, new_commits: list):
    if state["train_digest"] != train_digest:
        raise SystemExit(
            "resume migration refused: the training population differs from the checkpoint's "
            f"({state['train_digest'][:12]} != {train_digest[:12]}). A migration may only "
            "forgive a CODE commit; a different pool, cap or split is a different experiment "
            "and must not be continued in the same run dir.")
    stored = state["args_digest"]
    if stored == new_args_digest:
        return None
    if declared_commit == "" or declared_commit is False:
        raise SystemExit(
            f"resume refused: the checkpoint's args digest ({stored[:12]}) differs from this "
            f"run's ({new_args_digest[:12]}). If the ONLY difference is the code commit, "
            "re-run with --migrate-from-commit <the producing commit>: it verifies that claim "
            "against the checkpoint's own digest and records both producers. Without it, "
            "nothing is assumed.")
    method, old_payload, summary_note = "declared_config", None, None
    if declared_payload is not None:
        candidate = training_args_digest_payload(declared_payload)
        if digest_fn(candidate) == stored:
            old_payload = candidate
        else:
            summary_note = {"used": False, "summary_args_digest": digest_fn(candidate),
                            "checkpoint_args_digest": stored,
                            "why": "summary.json does not describe this checkpoint"}
    if old_payload is None:
        method = "digest_reproduction"
        old_payload, _dirty = _reproduce_old_payload(
            new_payload, declared_commit=declared_commit, stored_digest=stored,
            digest_fn=digest_fn)
        if old_payload is None:
            raise SystemExit(
                "resume migration refused: substituting the declared producer commit "
                f"{declared_commit} into this run's inputs does NOT reproduce the checkpoint's "
                f"args digest {stored[:12]}. Either the declared commit is not the one that "
                "produced this checkpoint, or something other than the code changed (an "
                "input's bytes or path, the price, the schedule, the targets, the global batch "
                "or an optimiser setting). Nothing is assumed and nothing is written."
                + (f" The run dir's summary.json declares a different config again "
                   f"({summary_note['summary_args_digest'][:12]}), so it is no help here."
                   if summary_note else ""))
    old_commits = producer_commits(old_payload)
    if old_commits != [declared_commit]:
        raise SystemExit(
            f"resume migration refused: the checkpoint was produced at {old_commits}, not at "
            f"the declared --migrate-from-commit {declared_commit!r}. The declaration is "
            "verified against the checkpoint, never trusted.")
    code_paths, blocking = classify_payload_differences(old_payload, new_payload)
    if blocking:
        raise SystemExit(
            "resume migration refused: these payload fields differ and are NOT a code commit: "
            f"{blocking[:8]}{' ...' if len(blocking) > 8 else ''}. A migration forgives the "
            "recorded code revision of the inputs and nothing else.")
    if not code_paths:
        raise SystemExit(
            "resume migration refused: the payloads are identical field by field, yet their "
            f"digests differ ({stored[:12]} != {new_args_digest[:12]}). That means the digest "
            "FUNCTION changed, which a migration must not paper over.")
    verified = {}
    for field in MIGRATION_VERIFIED_FIELDS:
        if old_payload.get(field) != new_payload.get(field):
            raise SystemExit(f"resume migration refused: {field} differs "
                             f"({old_payload.get(field)!r} != {new_payload.get(field)!r})")
        verified[field] = new_payload.get(field)
    return {
        "schema": RESUME_MIGRATION_SCHEMA,
        "method": method,
        "reason": "code_commit_only",
        "changed_paths": code_paths,
        "declared_config": summary_note,
        "original_producer": {
            "commit": declared_commit,
            "commits_recorded": old_commits,
            "args_digest": stored,
            "train_digest": state["train_digest"],
            "epoch": int(state["epoch"]),
            "stopped": bool(state["stopped"]),
            "best": _json_copy(state["best"]),
            "n_history_rows": len(state["history"]),
        },
        "new_executor": {
            "commits_recorded": new_commits,
            "args_digest": new_args_digest,
            "train_digest": train_digest,
        },
        "verified": verified,
        "input_sha256": input_sha256s(new_payload),
        "preserved": {"optimizer": True, "rng_python": True, "rng_torch": True,
                      "best_trainable": True, "history": True, "stopped": True},
    }


def archive_original_checkpoint(resume_path, *, out_dir, args_digest: str) -> dict:
    import hashlib
    import os
    import shutil

    resume_path = Path(resume_path)
    archive_dir = Path(out_dir) / "migration"
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / f"osp_last.{args_digest[:12]}.pt"
    if not target.exists():
        tmp = f"{target}.tmp.{os.getpid()}"
        shutil.copy2(resume_path, tmp)
        os.replace(tmp, target)
    blob = target.read_bytes()
    return {"path": str(target), "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob)}


def append_migration_sidecar(out_dir, record: dict) -> dict:
    path = Path(out_dir) / RESUME_MIGRATION_FILE
    chain = {"schema": RESUME_MIGRATION_SCHEMA, "migrations": []}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("schema") != RESUME_MIGRATION_SCHEMA:
            raise SystemExit(f"{path} has schema {existing.get('schema')!r}, expected "
                             f"{RESUME_MIGRATION_SCHEMA!r}")
        chain["migrations"] = list(existing.get("migrations", []))
    chain["migrations"].append(record)
    path.write_text(json.dumps(chain, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return chain


def declared_training_config(out_dir):
    path = Path(out_dir) / "summary.json"
    if not path.exists():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    config = summary.get("config", {}).get("training_config")
    return config if isinstance(config, dict) and config else None


def load_resume_migrated(resume_path, *, train_digest, args_digest, training_config,
                         declared_commit, out_dir, torch, write: bool = True):
    from pathlib import Path as _P

    if not _P(resume_path).exists():
        return None, None
    state = T.validate_resume_state(torch.load(resume_path, map_location="cpu",
                                               weights_only=False))
    if not declared_commit:
        T.check_resume_compatible(state, train_digest=train_digest, args_digest=args_digest)
        return state, None
    new_payload = training_args_digest_payload(training_config)
    migration = plan_resume_migration(
        state, new_payload=new_payload, new_args_digest=args_digest,
        train_digest=train_digest, declared_commit=declared_producer_commit(declared_commit),
        declared_payload=declared_training_config(out_dir), digest_fn=T.run_args_digest,
        new_commits=producer_commits(new_payload))
    if migration is None:
        T.check_resume_compatible(state, train_digest=train_digest, args_digest=args_digest)
        return state, None
    if write:
        migration["original_producer"]["archived_checkpoint"] = archive_original_checkpoint(
            resume_path, out_dir=out_dir, args_digest=state["args_digest"])
        migration["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        append_migration_sidecar(out_dir, migration)
    return state, migration


def save_resume(path, *, epoch, reader, opt, rng, history, best, best_trainable, stopped,
                train_digest, args_digest, torch):
    import os

    state = {
        "format": T.RESUME_FORMAT,
        "epoch": int(epoch),
        "stopped": bool(stopped),
        "trainable": {n: p_.detach().cpu().clone() for n, p_ in reader.module.named_parameters()
                      if p_.requires_grad},
        "best_trainable": best_trainable,
        "optimizer": opt.state_dict(),
        "rng_python": rng.getstate(),
        "rng_torch": torch.get_rng_state(),
        "history": history,
        "best": best,
        "train_digest": train_digest,
        "args_digest": args_digest,
    }
    T.validate_resume_state(state)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_resume(path, *, train_digest, args_digest, torch):
    from pathlib import Path as _P

    if not _P(path).exists():
        return None
    state = T.validate_resume_state(torch.load(path, map_location="cpu", weights_only=False))
    T.check_resume_compatible(state, train_digest=train_digest, args_digest=args_digest)
    return state


def check_rng_replay(state, replayed_rng) -> None:
    stored = state["rng_python"]
    actual = replayed_rng.getstate()
    if stored is not None and tuple(stored[1]) != tuple(actual[1]):
        raise SystemExit(
            "resume refused: replaying the shuffles did not reproduce the checkpoint's RNG "
            "state. The batch order after resume would differ from the uninterrupted run, so "
            "the lane would not be equivalent. This means the training population or its order "
            "changed in a way the digests did not catch.")


def tg_catalogue():
    from rl_controller.iclr import targets as tg
    return tg.CATALOGUE


def load_questions(paths):
    rows = []
    for path in paths:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("--questions: no rows")
    return rows


def partition_universe(fit, *, faithful: bool = False) -> set[str]:
    if faithful:
        from rl_controller.baselines.ospruner.replay import faithful_subsample
        fit = faithful_subsample(fit)
    return {c["qkey"] for c in fit if c["_trainable"]}


def partition(fit, select, args):
    pool_keys = partition_universe(fit)
    dropped = {"fit": 0, "select": 0}
    if args.max_chain_tokens:
        def within(c):
            return not c["_ends"] or c["_ends"][-1] <= args.max_chain_tokens
        n_fit, n_sel = len(fit), len(select)
        fit = [c for c in fit if within(c)]
        select = [c for c in select if within(c)]
        dropped = {"fit": n_fit - len(fit), "select": n_sel - len(select)}
    trainable = [c for c in fit if c["_trainable"]]
    if args.max_chains_per_question:
        trainable = lc.cap_chains(trainable, args.max_chains_per_question)
    val_keys, train_order = lc.partition(pool_keys, n_val=args.val_count, salt=args.val_salt)
    train_keys = set(lc.nested_subset(train_order, args.max_train_questions)
                     if args.max_train_questions else train_order)
    val_keys = set(val_keys)
    train = [c for c in trainable if c["qkey"] in train_keys]
    hold = [c for c in trainable if c["qkey"] in val_keys]
    if not train or not hold:
        raise SystemExit(f"empty split: {len(train)} train, {len(hold)} val chains")
    return train, hold, select, {"n_pool_questions": len(pool_keys),
                                 "n_val_questions": len(val_keys),
                                 "n_train_questions": len(train_keys),
                                 "dropped_over_max_chain_tokens": dropped}


def split_by_context(chains, prompts, max_seq_len):
    fits, too_long = [], []
    for chain in chains:
        ends = chain["_ends"]
        seq_len = len(prompts[chain["qkey"]]) + (ends[-1] if ends else 0)
        (fits if seq_len <= max_seq_len else too_long).append(chain)
    return fits, too_long


def check_dumpable(chains):
    bad = [(c["qkey"], c["rep"]) for c in chains if not c["_trainable"] and c["probes"]]
    if bad:
        raise SystemExit(f"{len(bad)} chains have probes but are not trainable "
                         f"(e.g. {bad[:3]}) — cannot dump consistently")


def load_label_dump_input(path):
    from rl_controller.iclr import label_dump as ld

    logical = T.logical_dump_path(path)
    manifest, rows = ld.read_dump(logical)
    if manifest.get("schema") != ld.SCHEMA:
        raise ValueError(f"label dump schema {manifest.get('schema')!r} != {ld.SCHEMA!r}")
    data_path = Path(manifest["output"]["path"])
    if not data_path.is_absolute():
        data_path = logical.parent / data_path.name
    expected_data = ld.dump_data_path(logical, manifest["output"]["sha256"])
    if data_path.resolve() != expected_data.resolve():
        raise ValueError(
            f"label dump manifest does not name its content-addressed file: "
            f"{data_path} != {expected_data}")
    manifest_file = ld.manifest_path(logical)
    provenance = {
        "data": source_record(data_path),
        "manifest": source_record(manifest_file),
    }
    return manifest, rows, provenance


def resolve_label_dump_path(path) -> Path:
    from rl_controller.iclr import label_dump as ld

    logical = T.logical_dump_path(path)
    manifest_file = ld.manifest_path(logical)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        output = manifest["output"]
        claimed_sha = output["sha256"]
        data_path = Path(output["path"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid label dump manifest {manifest_file}: {exc}") from None
    if manifest.get("schema") != ld.SCHEMA:
        raise ValueError(f"label dump schema {manifest.get('schema')!r} != {ld.SCHEMA!r}")
    if (not isinstance(claimed_sha, str) or len(claimed_sha) != 64
            or any(char not in "0123456789abcdef" for char in claimed_sha)):
        raise ValueError("label dump manifest output.sha256 is not 64 lowercase hex")
    if not data_path.is_absolute():
        data_path = logical.parent / data_path.name
    expected = ld.dump_data_path(logical, claimed_sha)
    if data_path.resolve() != expected.resolve():
        raise ValueError(f"label dump manifest names {data_path}, expected {expected}")
    if ld.sha256_file(data_path) != claimed_sha:
        raise ValueError(f"label dump addressed bytes do not match manifest sha256 {claimed_sha}")
    return data_path.resolve()


def main(argv=None) -> int:
    args = parse_args(argv)
    validate_args(args)
    import torch

    schedule = osp.parse_schedule(args.osp_lambda_anneal)
    T.require_epoch_budget(args.epochs, schedule, args.patience)
    if args.head_mode == "qv" and not args.label_dump:
        raise SystemExit("--head-mode qv needs --label-dump: C2 cells train on materialized, "
                         "sha-stamped dumps and never re-derive labels (campaign §3, C2 rule)")
    if args.head_mode == "osp" and args.label_dump:
        raise SystemExit("--label-dump is meaningless with --head-mode osp: the OSP objective "
                         "reads A_i/L_i off the pool, not off a label dump")
    names = [n.strip() for n in args.targets.split(",") if n.strip()]
    if args.head_mode == "qv":
        if names[:2] != ["q", "v"]:
            raise SystemExit(f"--head-mode qv requires --targets to start with q,v (the "
                             f"ladder's mandatory beliefs); got {names}")
        unknown = [n for n in names if n not in tg_catalogue()]
        if unknown:
            raise SystemExit(f"--targets names not in targets.CATALOGUE: {unknown}")

    torch.manual_seed(args.seed)
    runtime = D.runtime_from_env()
    set_runtime(runtime)
    if torch.cuda.is_available():
        if runtime.distributed:
            D.require_devices(runtime.world, torch.cuda.device_count())
            device = torch.device(f"cuda:{runtime.local_rank}")
        else:
            device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if runtime.distributed:
        import os as _os

        _os.environ.setdefault("GIT_OPTIONAL_LOCKS", "0")
    backend = D.init_distributed(runtime, timeout_s=args.dist_timeout_s)
    log({"event": "runtime", **runtime.as_dict(), "backend": backend,
         "device": str(device), "chains_per_step_is_global": args.chains_per_step,
         "note": "chains_per_step is the GLOBAL batch; it is never multiplied by world size"})
    device_fields = training_device_fields(
        gather_device_records(torch, runtime, device_record(torch, device)))
    log({"event": "device", **device_fields})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_manifest = label_rows = label_prov = None
    if args.label_dump:
        logical = T.logical_dump_path(args.label_dump)
        if str(logical) != str(args.label_dump):
            log({"event": "label_dump_path_normalised", "given": args.label_dump,
                 "used": str(logical)})
        label_manifest, label_rows, label_prov = load_label_dump_input(args.label_dump)

    input_prov = {
        "fit_chains": [source_record(p) for p in args.fit_chains],
        "select_chains": source_record(args.select_chains),
        "questions": [source_record(p) for p in args.questions],
        "label_dump": label_prov,
    }
    fit = sio.load_chains(args.fit_chains)
    select = sio.load_chains([args.select_chains])
    if args.faithful_subsample:
        from rl_controller.baselines.ospruner.replay import faithful_subsample
        before = len(fit)
        fit = faithful_subsample(fit)
        log({"event": "faithful_subsample", "before": before, "after": len(fit)})
    train, hold, select, split_stats = partition(fit, select, args)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path or args.model,
                                              revision=args.revision, cache_dir=args.hf_cache,
                                              trust_remote_code=True)
    prompts = sio.build_prompt_table(load_questions(args.questions), tokenizer)
    missing = [c["qkey"] for c in train + hold + select if c["qkey"] not in prompts]
    if missing:
        raise SystemExit(f"{len(missing)} chains have no prompt (e.g. {missing[:3]})")
    prompt_lens = {k: len(v) for k, v in prompts.items()}

    val_dump, select_dump = list(hold), list(select)
    check_dumpable(val_dump)
    check_dumpable(select_dump)

    train, train_long = split_by_context(train, prompts, args.max_seq_len)
    val_score, val_long = split_by_context(val_dump, prompts, args.max_seq_len)
    select_score, select_long = split_by_context(
        [c for c in select_dump if c["_trainable"]], prompts, args.max_seq_len)
    if not train or not val_score or not select_score:
        raise SystemExit("a split has no chain inside --max-seq-len")
    if train_long:
        log({"warning": "over_context_training_chains_dropped", "n": len(train_long),
             "do_not_publish": "a lane that dropped training chains is stamped "
                               "publishable=false; raise --max-seq-len and retrain"})
    if val_long or select_long:
        longest = max(len(prompts[c["qkey"]]) + c["_ends"][-1] for c in val_long + select_long)
        if not args.allow_unscored:
            raise SystemExit(
                f"{len(val_long)} val and {len(select_long)} select chains exceed "
                f"--max-seq-len {args.max_seq_len} (longest needs {longest} tokens). They "
                "cannot be scored, and dropping them would move the natural anchor away from "
                "every other method's. Raise --max-seq-len to cover the pool, or pass "
                "--allow-unscored for a DIAGNOSTIC run whose dumps must not be published.")
        unscoreable = {(c["qkey"], c["rep"]) for c in val_long + select_long}
        val_dump = [c for c in val_dump if (c["qkey"], c["rep"]) not in unscoreable]
        select_dump = [c for c in select_dump if (c["qkey"], c["rep"]) not in unscoreable]
        log({"warning": "unscored_chains_excluded_from_dump", "n": len(unscoreable),
             "longest_needed_tokens": longest,
             "do_not_publish": "the dump no longer matches the pool, so its anchors differ"})

    if args.head_mode == "osp":
        T.attach_osp_terms(train + val_score + select_score, grade=args.osp_grade,
                           token_scale=args.osp_token_scale, cost=args.osp_cost)
        log({"event": "terms", "train": T.terms_summary(train),
             "lambda_scale": T.lambda_scale_check(train, [args.osp_lambda])})
        label_stats = None
    else:
        from rl_controller.iclr import targets as tg
        label_stats = T.attach_dump_labels(train + val_score, label_rows, names,
                                           positions_of=lambda n: tg.CATALOGUE[n].positions)
        label_stats["dump_manifest"] = {"output": label_manifest["output"],
                                        "target_set": label_manifest.get("target_set")}
        log({"event": "labels", **{k: v for k, v in label_stats.items() if k != "dump_manifest"}})

    from rl_controller.baselines.ospruner.reader import load_backbone, pad_batch_tensors
    backbone = load_backbone(model_path=args.model_path, model_id=args.model,
                             revision=args.revision, dtype=args.dtype, cache_dir=args.hf_cache)
    reader = TopTwoReader(backbone, head_mode=args.head_mode, n_unfrozen=args.n_unfrozen,
                          init_fire_rate=args.osp_init_fire_rate,
                          fp32_trainable=not args.no_fp32_trainable,
                          output_names=None if args.head_mode == "osp" else names)
    reader.module.to(device)
    reader.train()
    log({"event": "reader", **reader.manifest()})

    probe = pad_batch_tensors([[tokenizer.eos_token_id or 0] * 8], L.plan_batch([(0, 8, [4])]),
                              device=device)
    reader.verify_graph_isolation(*probe)
    log({"event": "graph_isolation_verified", "n_trainable": reader.n_trainable})

    opt = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.999), eps=1e-8)

    sync = D.GradientSync(reader.module.named_parameters(), runtime, device=device)
    step_size = max(1, args.chains_per_step)

    probes_only = T.gathers_probes_only(args.head_mode)

    def forward(group, *, grad: bool):
        plan = L.plan_chain_batch(group, prompt_lens=prompt_lens, probes_only=probes_only)
        seqs = [list(prompts[c["qkey"]]) + list(c["completion_ids"][:c["_ends"][-1]])
                for c in group]
        ids, mask = pad_batch_tensors(seqs, plan, device=device,
                                      pad_id=tokenizer.pad_token_id or 0)
        with torch.set_grad_enabled(grad):
            logits = reader.boundary_logits(ids, mask, plan)
        return reader.split_by_chain(logits, plan), plan

    train_digest = T.train_order_digest(train)
    training_config = training_run_config(
        args, names=names, schedule=schedule, input_provenance=input_prov)
    args_digest = T.run_args_digest(training_args_digest_payload(training_config))
    resume_path = out_dir / "osp_last.pt"
    state, migration = (load_resume_migrated(
        resume_path, train_digest=train_digest, args_digest=args_digest,
        training_config=training_config, declared_commit=args.migrate_from_commit,
        out_dir=out_dir, torch=torch, write=runtime.is_writer)
        if args.resume else (None, None))
    if migration is not None:
        log({"event": "resume_migrated",
             "from_commit": migration["original_producer"]["commit"],
             "from_args_digest": migration["original_producer"]["args_digest"][:12],
             "to_args_digest": migration["new_executor"]["args_digest"][:12],
             "method": migration["method"], "changed_paths": migration["changed_paths"],
             "archived": migration["original_producer"].get("archived_checkpoint", {})
                         .get("path")})
    plan = T.resume_plan(state, epochs=args.epochs)
    parity = D.check_agreement({
        "train_digest": train_digest,
        "args_digest": args_digest,
        "split_sizes": [len(train), len(val_score), len(select_score),
                        len(val_dump), len(select_dump)],
        "param_layout": sync.layout(),
        "resume_plan": plan,
        "resume_epoch": (state["epoch"] if state is not None else -1),
        "resume_migration": (None if migration is None else {
            "method": migration["method"],
            "from_commit": migration["original_producer"]["commit"],
            "from_args_digest": migration["original_producer"]["args_digest"],
            "to_args_digest": migration["new_executor"]["args_digest"],
            "changed_paths": migration["changed_paths"]}),
        "seed_lambda": [args.seed, args.osp_lambda, args.head_mode, list(names)],
    }, runtime, device=device)

    history = []
    val_receipts = []
    best = {"epoch": None, "stat": math.inf}
    best_trainable = None
    if state is not None:
        by_name = dict(reader.module.named_parameters())
        with torch.no_grad():
            for name, value in state["trainable"].items():
                by_name[name].copy_(value.to(by_name[name].device))
        opt.load_state_dict(state["optimizer"])
        history = list(state["history"])
        best = dict(state["best"])
        best_trainable = state["best_trainable"]
        torch.set_rng_state(state["rng_torch"])
        log({"event": "resumed", "from_epoch": state["epoch"], **plan,
             "train_digest": train_digest[:12]})
    rng = T.replay_shuffles(train, args.seed, plan["completed_epochs"])
    if state is not None:
        check_rng_replay(state, rng)
    if plan["complete"]:
        log({"event": "already_trained", **plan, "note": "re-entering at scoring"})

    stopped = plan["early_stopped"]
    for epoch in range(plan["start_epoch"], args.epochs if not stopped else plan["start_epoch"]):
        rng.shuffle(train)
        D.check_agreement({"epoch": epoch,
                           "order": [(c["qkey"], c["rep"]) for c in train]},
                          runtime, device=device)
        lam = T.lambda_at(epoch, args.osp_lambda, schedule)
        started = time.time()
        local_total, n_steps, n_local_chains = 0.0, 0, 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        n_groups = D.n_global_steps(len(train), step_size)
        beat = max(1, n_groups // 20)
        for group, local in D.step_groups(train, step_size, runtime.rank, runtime.world):
            opt.zero_grad(set_to_none=True)
            if local:
                per_chain, _plan = forward(local, grad=True)
                loss = (_loss_terms_or_none(per_chain, local, args, names, torch)
                        if args.head_mode == "qv" else _osp_loss(per_chain, local, lam, torch))
                if loss is not None:
                    loss = loss / len(group)
                    loss.backward()
                    local_total += float(loss.detach())
                n_local_chains += len(local)
            receipt = sync.reduce()
            if receipt["n_global_with_grad"] == 0:
                raise SystemExit("every label in this batch was masked")
            torch.nn.utils.clip_grad_norm_(reader.parameters(), args.grad_clip)
            opt.step()
            n_steps += 1
            if n_steps % beat == 0:
                elapsed = time.time() - started
                log({"event": "progress", "epoch": epoch, "step": n_steps,
                     "of": n_groups, "frac": n_steps / n_groups,
                     "elapsed_s": round(elapsed, 1),
                     "epoch_eta_s": round(elapsed * n_groups / n_steps, 1),
                     "local_train_loss": local_total / n_steps,
                     "world": runtime.world, "n_local_chains": n_local_chains,
                     "gpu_peak_gib": (round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
                                      if torch.cuda.is_available() else None)})
        train_seconds = time.time() - started
        peak_gib = (round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
                    if torch.cuda.is_available() else 0.0)
        global_total, global_chains = D.all_reduce_floats(
            [local_total, float(n_local_chains)], runtime, op="sum", device=device)
        slowest_s, max_peak_gib = D.all_reduce_floats(
            [train_seconds, peak_gib], runtime, op="max", device=device)
        stat, extra = _evaluate(reader, val_score, args, names, forward, torch,
                                args.osp_lambda, runtime=runtime, device=device)
        val_receipts.append({
            "epoch": epoch, "world": runtime.world,
            "n_chains_global": int(extra["n_val_chains"]),
            "n_chains_local": int(extra.get("n_val_chains_local", extra["n_val_chains"])),
            "n_chains_contributing": int(extra["n_val_chains_contributing"]),
            "n_chains_expected": len(val_score),
            "val_stat": stat})
        if int(extra["n_val_chains"]) != len(val_score):
            raise SystemExit(
                f"validation covered {int(extra['n_val_chains'])} chains but the split holds "
                f"{len(val_score)}: the shards do not add up to the split, so the val "
                "statistic is computed over a different population than the lane declares.")
        row = {"epoch": epoch, "lambda_used": lam,
               "train_loss": global_total / max(n_steps, 1),
               "val_stat": stat, "seconds": time.time() - started,
               "gpu_peak_gib": (max_peak_gib if torch.cuda.is_available() else None),
               "eligible": T.checkpoint_eligible(epoch, schedule), **extra}
        if runtime.distributed:
            row.update({"world": runtime.world, "n_global_steps": n_steps,
                        "n_train_chains": int(global_chains),
                        "train_seconds_slowest_rank": round(slowest_s, 1)})
        history.append(row)
        log({"event": "epoch", **row})
        finite, better = D.broadcast_floats(
            [float(math.isfinite(stat)),
             float(bool(row["eligible"] and stat < best["stat"]))],
            runtime, device=device)
        if not finite:
            raise SystemExit(f"non-finite val statistic at epoch {epoch}: training failed")
        if better:
            best = {"epoch": epoch, "stat": stat}
            best_trainable = {n: p.detach().cpu().clone() for n, p in
                              reader.module.named_parameters() if p.requires_grad}
            if runtime.is_writer:
                torch.save({"head": reader.head.state_dict(), "blocks": best_trainable,
                            "epoch": epoch, "lambda": args.osp_lambda,
                            "head_mode": args.head_mode,
                            "targets": names, "metrics": row}, out_dir / "osp_best.pt")
        stopped = T.should_stop(epoch, best["epoch"], schedule=schedule, patience=args.patience)
        stopped = bool(D.broadcast_floats([float(stopped)], runtime, device=device)[0])
        if runtime.is_writer:
            save_resume(resume_path, epoch=epoch, reader=reader, opt=opt, rng=rng,
                        history=history, best=best, best_trainable=best_trainable,
                        stopped=stopped, train_digest=train_digest,
                        args_digest=args_digest, torch=torch)
        if stopped:
            log({"event": "early_stop", "epoch": epoch, "best_epoch": best["epoch"]})
            break

    chosen = T.select_epoch(history, schedule=schedule, key="val_stat", minimise=True,
                            patience=args.patience)
    log({"event": "selected", **chosen})
    if best_trainable is None or best["epoch"] is None:
        raise SystemExit(
            "no eligible epoch produced a checkpoint: every epoch was below the annealing "
            "floor, so there is nothing that may be reported under the nominal lambda.")
    if chosen["epoch"] != best["epoch"]:
        raise SystemExit(
            f"selection disagreement: select_epoch chose {chosen['epoch']} but the tracked best "
            f"is {best['epoch']}. These must agree -- one of them is not the run's operating "
            "point, and scoring the wrong weights is not detectable downstream.")
    by_name = dict(reader.module.named_parameters())
    with torch.no_grad():
        for name, value in best_trainable.items():
            by_name[name].copy_(value.to(by_name[name].device))

    shard_dir = out_dir / ".score_shards"
    attempt = D.attempt_token(D.broadcast_int(time.time_ns(), runtime, device=device))
    if runtime.distributed and runtime.is_writer:
        removed = D.purge_stale_shards(shard_dir, split="val", attempt=attempt)
        removed += D.purge_stale_shards(shard_dir, split="select", attempt=attempt)
        if removed:
            log({"event": "stale_score_shards_removed", "n": len(removed),
                 "names": removed[:4]})
    D.barrier(runtime)

    def score_split(split, chains):
        local = D.shard(chains, runtime.rank, runtime.world)
        _score(reader, local, args, names, forward, torch, split=split, runtime=runtime)
        if not runtime.distributed:
            return {"split": split, "world": 1, "attempt": attempt,
                    "n_chains": len(chains),
                    "n_scores": sum(len(c["scores"]) for c in chains)}
        n_rows = D.write_score_shard(
            shard_dir / D.shard_name(split, runtime.rank, runtime.world, attempt),
            split=split, rank=runtime.rank, world=runtime.world, attempt=attempt,
            rows=local)
        log({"event": "score_shard_written", "split": split, "n_rows": n_rows},
            every_rank=True)
        D.barrier(runtime)
        if not runtime.is_writer:
            return None
        return D.merge_score_shards(shard_dir, split=split, world=runtime.world,
                                    attempt=attempt, chains=chains)

    score_receipts = [score_split("val", val_score), score_split("select", select_score)]
    if not runtime.is_writer:
        log({"event": "rank_done", "world": runtime.world}, every_rank=True)
        D.shutdown()
        return 0

    over_record = over_max_seq_len_record(
        train_long=train_long, val_long=val_long, select_long=select_long,
        allow_unscored=args.allow_unscored)
    execution = D.execution_record(
        runtime, backend=backend, digests=parity,
        chains_per_step_is_global_batch=True,
        n_global_steps_per_epoch=D.n_global_steps(len(train), step_size),
        score_shards=[receipt for receipt in score_receipts if receipt],
        validation=val_receipts,
        resume={"enabled": bool(args.resume),
                "checkpoint_found": state is not None,
                "resumed_from_epoch": (state["epoch"] if state is not None else None),
                "plan": plan,
                "rng_replay_verified": state is not None,
                "train_digest": train_digest,
                "args_digest_loaded": (state["args_digest"] if state is not None else None),
                "args_digest_now": args_digest,
                "migration": migration},
        dist_timeout_s=args.dist_timeout_s)
    summary = build_training_summary(
        args, names=names, schedule=schedule, input_provenance=input_prov,
        reader_manifest=reader.manifest(), training_config=training_config,
        osp_lambda=args.osp_lambda,
        **device_fields,
        q_is=("stop probability pi" if args.head_mode == "osp"
              else "P(forced answer correct at this boundary)"),
        chosen_epoch=chosen, history=history,
        split=split_stats, label_stats=label_stats,
        over_max_seq_len=over_record,
        publishable=lane_is_publishable(over_record),
        n_dumped={"val": len(val_dump), "select": len(select_dump)},
        lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs,
        patience=args.patience,
        training_population=training_population_attestation(
            args, train_digest=train_digest, n_train_chains=len(train)),
        resume={"enabled": bool(args.resume),
                "resumed_from_epoch": (state["epoch"] if state is not None else None),
                "early_stopped": bool(stopped), "selected_epoch": chosen["epoch"],
                "train_digest": train_digest, "args_digest": args_digest,
                "args_digest_schema": TRAINING_ARGS_DIGEST_SCHEMA},
        execution=execution)
    result = sio.write_run_dir(out_dir, val=val_dump, select=select_dump, summary=summary)
    if runtime.distributed:
        for split in ("val", "select"):
            D.remove_shards(shard_dir, split=split)
    published = dict(execution)
    published["export"] = {
        "world_size": runtime.world,
        "writer_rank": runtime.rank,
        **result,
        "n_dumped": {"val": len(val_dump), "select": len(select_dump)},
        "files": {name: source_record(out_dir / name)
                  for name in ("val_scored.jsonl", "select_scored.jsonl", "summary.json")},
    }
    (out_dir / "execution.json").write_text(
        json.dumps(published, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log({"event": "done", **result, "world": runtime.world})
    D.shutdown()
    return 0


def _osp_loss(per_chain, group, lam, torch):
    total = None
    for logits, chain in zip(per_chain, group):
        term = osp.osp_chain_loss(logits[:, 0], chain["_osp_terms"], lam)
        total = term if total is None else total + term
    return total


def _loss_terms_or_none(per_chain, group, args, names, torch):
    total = None
    for logits, chain in zip(per_chain, group):
        for col, name in enumerate(names):
            labels = chain["_dump_labels"][name]
            keep = [(i, y) for i, y in enumerate(labels) if y is not None]
            if not keep:
                continue
            rows = torch.tensor([i for i, _ in keep], dtype=torch.long, device=logits.device)
            target = torch.tensor([float(y) for _, y in keep], dtype=logits.dtype,
                                  device=logits.device)
            term = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[rows, col], target)
            total = term if total is None else total + term
    return total


def _evaluate(reader, chains, args, names, forward, torch, lam, *,
              runtime=D.SINGLE, device=None):
    reader.eval()
    local = D.shard(chains, runtime.rank, runtime.world)
    total, n, n_contrib = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, len(local), max(1, args.chains_per_step)):
            group = local[i:i + max(1, args.chains_per_step)]
            per_chain, _ = forward(group, grad=False)
            if args.head_mode == "osp":
                value = _osp_loss(per_chain, group, lam, torch)
            else:
                value = _loss_terms_or_none(per_chain, group, args, names, torch)
            if value is not None:
                total += float(value)
                n_contrib += len(group)
            n += len(group)
    reader.train()
    total, n, n_contrib = D.all_reduce_floats([total, float(n), float(n_contrib)],
                                              runtime, op="sum", device=device)
    if n_contrib == 0:
        raise SystemExit("every label in the val split was masked")
    extra = {"n_val_chains": int(n), "n_val_chains_contributing": int(n_contrib)}
    if runtime.distributed:
        extra["n_val_chains_local"] = len(local)
    return D.combine_mean(total, n), extra


def _score(reader, chains, args, names, forward, torch, *, split="", runtime=D.SINGLE):
    reader.eval()
    beat = max(1, (len(chains) // max(1, args.chains_per_step)) // 10)
    done = 0
    with torch.no_grad():
        for i in range(0, len(chains), max(1, args.chains_per_step)):
            group = chains[i:i + max(1, args.chains_per_step)]
            per_chain, _ = forward(group, grad=False)
            for logits, chain in zip(per_chain, group):
                probs = torch.sigmoid(logits.float()).cpu().tolist()
                if args.head_mode == "osp":
                    chain["scores"] = [{"q": p[0], "v": 0.0, "pi": p[0]} for p in probs]
                else:
                    chain["scores"] = [{n: probs[i][c] for c, n in enumerate(names)}
                                       for i in chain["_probe_pos"]]
            for chain in group:
                if len(chain["scores"]) != len(chain["probes"]):
                    raise SystemExit(f"chain {(chain['qkey'], chain['rep'])}: scored "
                                     f"{len(chain['scores'])} of {len(chain['probes'])} probes")
            done += len(group)
            if (i // max(1, args.chains_per_step)) % beat == 0:
                log({"event": "scoring", "split": split, "scored": done,
                     "of": len(chains)}, every_rank=runtime.distributed)
    reader.train()


if __name__ == "__main__":
    raise SystemExit(main())
