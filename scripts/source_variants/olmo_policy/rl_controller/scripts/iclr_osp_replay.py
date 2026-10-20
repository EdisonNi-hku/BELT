"""Survival-product replay of a stochastic stop policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rl_controller.iclr import osp_policy as osp
from rl_controller.iclr.chains import load_chains, write_jsonl
from rl_controller.iclr.provenance import source_record

MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
PROMPT_CONTRACT_ID = "qwen3-1.7b-native-thinking-v1"
GRADER_ID = "archived_math_equal"
SELECTION_SPLIT = "val"
SELECTION_RULE = "validation-selected matched-accuracy levels"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scored", required=True, nargs="+",
                   help="one or more scored chain dumps carrying the per-probe pi")
    p.add_argument("--out", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--method-family", default="ospruner")
    p.add_argument("--method-name", default="OSP-port")
    p.add_argument("--config-id", default=None, help="default: <method-name>:lam<lambda>")
    p.add_argument("--knob-name", default="lambda")
    p.add_argument("--lambda", dest="lam", type=float, required=True,
                   help="the curve knob this policy was trained at")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dataset", required=True, help="e.g. math_union")
    p.add_argument("--split", required=True, help="e.g. select")
    p.add_argument("--pool", required=True, help="pool name, e.g. math_union_select")
    p.add_argument("--chain-bank", default=None,
                   help="the pool chains.jsonl the scored dump was produced from (hashed here)")
    p.add_argument("--chain-bank-sha256", default=None,
                   help="its sha256, if the file is not reachable from this host")
    p.add_argument("--manifest", default=None,
                   help="the question manifest (hashed here)")
    p.add_argument("--manifest-sha256", default=None,
                   help="its sha256, if the file is not reachable from this host")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--model-revision", default=MODEL_REVISION)
    p.add_argument("--tokenizer-revision", default=MODEL_REVISION)
    p.add_argument("--prompt-contract-id", default=PROMPT_CONTRACT_ID)
    p.add_argument("--grader-id", default=GRADER_ID)
    p.add_argument("--selection-split", default=SELECTION_SPLIT)
    p.add_argument("--selection-rule", default=SELECTION_RULE)
    p.add_argument("--frozen-config", default=None,
                   help="JSON object; default {\"lambda\": <lambda>}")
    p.add_argument("--source-branch", default="exp/iclr-bl-ws-b")
    p.add_argument("--source-commit", default=None, help="default: the working tree's HEAD")
    p.add_argument("--pi-key", default="pi", help="score key holding the stop probability")
    p.add_argument("--grade", choices=osp.GRADINGS, default="loose",
                   help="`loose` (default) is ladder.stop_outcome's convention, which every "
                        "frontier row in this line uses. A non-default grading is STAMPED into "
                        "grader_id and frozen_config so the rows cannot be compared as though "
                        "they followed the frozen split")
    p.add_argument("--expectation", choices=osp.PRODUCIBLE_EXPECTATIONS,
                   default="survival_product",
                   help="only the exact survival product is computed here; a seeded Bernoulli "
                        "replay is a different computation and ships as a record type 1 run dir")
    p.add_argument("--answer-format", choices=("boxed", "code"), default="boxed",
                   help="the pool's DECLARED answer format (never sniffed from the chains)")
    p.add_argument("--force-natural-answer", action="store_true",
                   help="declare that the pool was generated with --force-natural-answer; "
                        "refused here, since that convention needs the pinned tokenizer")
    p.add_argument("--aux-generated-tokens", type=float, default=0.0,
                   help="tokens the controller generates at decision time (0 for a "
                        "hidden-state head)")
    p.add_argument("--question-set", default=None,
                   help="jsonl/text of the qkeys the plan-rule methods cover on this pool "
                        "(contract §2.2 rule 7); questions this replay misses are emitted with "
                        "a missing_reason instead of being silently dropped")
    return p.parse_args(argv)


def natural_answer_fn(answer_format: str, force_natural_answer: bool):
    if answer_format != "boxed" or force_natural_answer:
        raise SystemExit(
            f"--answer-format {answer_format} --force-natural-answer "
            f"{force_natural_answer}: this producer only implements the archived boxed "
            "convention (natural answer tokens = 0). Charge the other conventions through "
            "rl_controller.iclr.token_accounting (program sidecar / pinned tokenizer).")
    return osp.natural_answer_tokens_archived_math


def load_question_set(path) -> set[str]:
    keys = set()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            row = json.loads(line)
            key = row.get("qkey") or row.get("question_key")
            if not key:
                raise SystemExit(f"--question-set {path}: row without a qkey: {line[:80]}")
            keys.add(key)
        else:
            keys.add(line)
    if not keys:
        raise SystemExit(f"--question-set {path}: no qkeys")
    return keys


def resolve_sha(path, given, label: str) -> str:
    if bool(path) == bool(given):
        raise SystemExit(f"give exactly one of --{label} or --{label}-sha256")
    if given:
        value = given.strip().lower()
    else:
        value = source_record(path)["sha256"]
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise SystemExit(f"--{label}-sha256 must be 64 hex chars, got {value!r}")
    return value


def main(argv=None) -> int:
    args = parse_args(argv)
    chain_bank_sha = resolve_sha(args.chain_bank, args.chain_bank_sha256, "chain-bank")
    manifest_sha = resolve_sha(args.manifest, args.manifest_sha256, "manifest")
    chains, provenance = [], []
    seen: dict[tuple, str] = {}
    for path in args.scored:
        loaded = load_chains(path)
        for c in loaded:
            ident = (c["qkey"], c["rep"])
            if ident in seen:
                raise SystemExit(f"--scored: chain {ident} appears in both {seen[ident]} and "
                                 f"{path} -- overlapping shards would double-weight it")
            seen[ident] = str(path)
        chains.extend(loaded)
        provenance.append(source_record(path))
    if not chains:
        raise SystemExit("no chains in --scored")
    try:
        frozen = json.loads(args.frozen_config) if args.frozen_config else {"lambda": args.lam}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--frozen-config: {exc}") from None
    if not isinstance(frozen, dict):
        raise SystemExit("--frozen-config must be a JSON object")
    frozen = {**frozen, "replay_grade": args.grade, "expectation": args.expectation}
    grader_id = args.grader_id if args.grade == "loose" else f"{args.grader_id}+grade_{args.grade}"
    if len(provenance) == 1:
        source_artifact, source_sha = provenance[0]["path"], provenance[0]["sha256"]
    else:
        source_artifact = "; ".join(sorted(pr["path"] for pr in provenance))
        digest = hashlib.sha256()
        for line in sorted(f"{pr['path']}:{pr['sha256']}" for pr in provenance):
            digest.update(line.encode() + b"\n")
        source_sha = digest.hexdigest()
    header = {
        "run_id": args.run_id, "method_family": args.method_family,
        "method_name": args.method_name,
        "config_id": args.config_id or f"{args.method_name}:lam{args.lam:g}",
        "knob": {"name": args.knob_name, "value": args.lam}, "seed": args.seed,
        "dataset": args.dataset, "split": args.split,
        "model_id": args.model_id, "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "prompt_contract_id": args.prompt_contract_id, "grader_id": grader_id,
        "selection_split": args.selection_split, "selection_rule": args.selection_rule,
        "frozen_config": frozen,
        "chain_bank_sha256": chain_bank_sha, "manifest_sha256": manifest_sha,
        "pool": args.pool,
        "source_branch": args.source_branch,
        "source_commit": args.source_commit or provenance[0]["commit"],
        "source_artifact": source_artifact, "source_sha256": source_sha,
    }
    try:
        rows = osp.replay_question_rows(
            chains, header=header, pi_key=args.pi_key, grade=args.grade,
            expectation=args.expectation,
            natural_answer_tokens=natural_answer_fn(args.answer_format,
                                                    args.force_natural_answer),
            aux_generated_tokens=args.aux_generated_tokens)
    except ValueError as exc:
        raise SystemExit(f"--scored: {exc}") from None

    missing = []
    expected: set[str] = set()
    if args.question_set:
        expected = load_question_set(args.question_set)
        covered = {r["qkey"] for r in rows}
        extra = covered - expected
        if extra:
            raise SystemExit(f"--question-set: replay covers {len(extra)} questions outside the "
                             f"declared set, e.g. {sorted(extra)[:3]} — wrong pool?")
        missing = sorted(expected - covered)
        rows.extend(osp.unavailable_row(qkey, header, "no scored chains for this question",
                                        expectation=args.expectation) for qkey in missing)
        rows.sort(key=lambda r: r["qkey"])

    out = Path(args.out)
    write_jsonl(out, rows)
    record = {
        "schema_version": osp.REPLAY_SCHEMA, **header,
        "grading": args.grade, "pi_key": args.pi_key, "expectation": args.expectation,
        "chain_bank": args.chain_bank, "manifest": args.manifest,
        "answer_format": args.answer_format,
        "force_natural_answer": bool(args.force_natural_answer),
        "natural_answer_tokens": "archived_math_zero",
        "aux_generated_tokens": args.aux_generated_tokens,
        "n_chains": len(chains), "n_rows": len(rows),
        "n_missing_questions": len(missing), "missing_questions": missing[:20],
        "summary": None if missing else osp.replay_summary(rows),
        "unavailable_reason": (None if not missing else
                               f"{len(missing)} of {len(expected)} declared questions have no "
                               "scored chains; the covered subset is biased"),
        "partial_summary_do_not_report": (None if not missing else
                                          osp.replay_summary(rows, allow_partial=True)),
        "natural": osp.natural_summary(chains),
        "output": {"path": str(out), "sha256": source_record(out)["sha256"]},
        "inputs": provenance,
    }
    natural_tokens = record["natural"]["mean_thinking_tokens"]
    for key in ("summary", "partial_summary_do_not_report"):
        point = record.get(key)
        if point:
            point["thinking_saving"] = (1.0 - point["mean_thinking_tokens"] / natural_tokens
                                        if natural_tokens else 0.0)
    out.with_name(out.stem + ".manifest.json").write_text(json.dumps(record, indent=2),
                                                          encoding="utf-8")
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
