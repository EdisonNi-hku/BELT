"""Generate reasoning chains and probe forced answers at query blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from rl_controller.iclr.chains import validate_chain
from rl_controller.iclr.provenance import source_record
from rl_controller.rollout_contract import (
    ANSWER_FORMATS,
    PROMPT_CONTRACTS,
    block_count,
    build_prompt_ids,
    contract_identity,
    code_answer_prompt,
    code_answer_stop,
    extract_program,
    forced_answer_prompt,
    forced_answer_stop,
    grade_forced,
    instruction_for,
    natural_boxed_prompt,
    natural_code_prompt,
    rewrite_user_instruction,
    sampling_config,
    select_prompt_contract,
)
from rl_controller.rollout_runner import GenRequest, VLLMEngine
from rl_controller.single_cot import QUERY_STRIDE, oracle_report, run_chain_batch
from rl_controller.splits import question_key, three_way_split

PROBE_CHUNK = 512

DRAW_SEED_RULE = "d==0: (chain,'probe',block); d>=1: (chain,'probe',block,d)"


def assigned_manifest_work(rows, *, split: str, shard_index: int, shard_count: int,
                           max_chains_per_question: int = 0):
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError(f"invalid shard {shard_index}/{shard_count}")
    if max_chains_per_question < 0:
        raise ValueError("max_chains_per_question must be >= 0")
    work = []
    for row in sorted(rows, key=lambda item: str(item["qkey"])):
        if row["split"] != split:
            continue
        target = int(row["target_chains"])
        if max_chains_per_question:
            target = min(target, max_chains_per_question)
        for rep in range(target):
            identity = f"{row['qkey']}:{rep}".encode()
            owner = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % shard_count
            if owner == shard_index:
                work.append((row, rep))
    return work


def probe_seed_key(chain_seed_key, block, draw: int):
    if draw == 0:
        return (chain_seed_key, "probe", block)
    return (chain_seed_key, "probe", block, draw)


def build_probe_requests(items, *, encode, max_forced_tokens: int, draws: int,
                         first_draw: int = 0, answer_format: str = "boxed"):
    if draws < 1 or not 0 <= first_draw < draws:
        raise ValueError(f"need 0 <= first_draw < draws, got first_draw={first_draw} draws={draws}")
    if answer_format == "boxed":
        continuation, stop = (lambda prefix: forced_answer_prompt(prefix, encode=encode)), forced_answer_stop()
    elif answer_format == "code":
        continuation, stop = (lambda prefix: code_answer_prompt(prefix, encode=encode)), code_answer_stop()
    else:
        raise ValueError(f"unknown answer format {answer_format!r}; choose from {sorted(ANSWER_FORMATS)}")
    requests, meta = [], []
    for item_index, item in enumerate(items):
        comp = item["completion_ids"]
        for block, plen in item["query_points"]:
            prompt = list(item["prompt_ids"]) + continuation(comp[:plen])
            for draw in range(first_draw, draws):
                requests.append(GenRequest(prompt_ids=prompt, max_tokens=max_forced_tokens,
                                           seed_key=probe_seed_key(item["seed_key"], block, draw),
                                           **stop))
                meta.append((item_index, block, plen, draw))
    return requests, meta


def build_natural_requests(rows, items, *, encode, decode, max_forced_tokens: int):
    requests, meta = [], []
    for i, (row, item) in enumerate(zip(rows, items)):
        if row["outcome"] != "natural":
            continue
        prompt = list(item["prompt_ids"]) + natural_code_prompt(item["completion_ids"], encode=encode,
                                                                decode=decode)
        requests.append(GenRequest(prompt_ids=prompt, max_tokens=max_forced_tokens,
                                   seed_key=(item["seed_key"], "natural"), **code_answer_stop()))
        meta.append(i)
    return requests, meta


def build_forced_natural_requests(rows, items, *, encode, decode, max_forced_tokens: int):
    requests, meta = [], []
    for i, (row, item) in enumerate(zip(rows, items)):
        if row["outcome"] != "natural":
            continue
        prompt = list(item["prompt_ids"]) + natural_boxed_prompt(item["completion_ids"], encode=encode,
                                                                 decode=decode)
        requests.append(GenRequest(prompt_ids=prompt, max_tokens=max_forced_tokens,
                                   seed_key=(item["seed_key"], "natural"), **forced_answer_stop()))
        meta.append(i)
    return requests, meta


def program_records(rows, probe_meta, probe_outs, natural_meta, natural_outs):
    records = []
    for (i, block, plen, _draw), out in zip(probe_meta, probe_outs):
        records.append({"qkey": rows[i]["qkey"], "rep": rows[i]["rep"], "kind": "probe",
                        "block": int(block), "prefix_len": int(plen),
                        "program": extract_program(out.text), "tokens": len(out.token_ids),
                        "finish_reason": out.finish_reason})
    for i, out in zip(natural_meta, natural_outs):
        records.append({"qkey": rows[i]["qkey"], "rep": rows[i]["rep"], "kind": "natural",
                        "block": None, "prefix_len": int(rows[i]["thinking_tokens"]),
                        "program": extract_program(out.text), "tokens": len(out.token_ids),
                        "finish_reason": out.finish_reason})
    return records


def attach_probes(rows, meta, outs, *, draws: int, stored_draw0=None, grade=grade_forced):
    if len(meta) != len(outs):
        raise ValueError(f"{len(outs)} outputs for {len(meta)} probe requests")
    if draws == 1:
        for (i, block, plen, _draw), out in zip(meta, outs):
            forced = grade(out.text)
            rows[i]["probes"].append({
                "block": int(block), "prefix_len": int(plen),
                "forced": forced, "forced_tokens": len(out.token_ids),
            })
        for row in rows:
            row["probes"].sort(key=lambda p: p["block"])
        return rows

    collected: dict = {}
    for (i, block, plen, draw), out in zip(meta, outs):
        slot = collected.setdefault((i, int(block)), {"prefix_len": int(plen), "draws": {}})
        if int(draw) in slot["draws"]:
            raise ValueError(f"duplicate draw {draw} for item {i} block {block}")
        slot["draws"][int(draw)] = {"draw": int(draw), "forced": grade(out.text),
                                    "forced_tokens": len(out.token_ids)}
    if stored_draw0 is not None:
        for key, stored in stored_draw0.items():
            slot = collected.setdefault(key, {"prefix_len": int(stored["prefix_len"]),
                                              "draws": {}})
            if 0 in slot["draws"]:
                raise ValueError(f"draw 0 both stored and generated for {key}")
            slot["draws"][0] = {"draw": 0, "forced": stored["forced"],
                                "forced_tokens": stored["forced_tokens"]}
    for (i, block), slot in collected.items():
        if sorted(slot["draws"]) != list(range(draws)):
            raise ValueError(f"item {i} block {block}: draws {sorted(slot['draws'])} "
                             f"are not 0..{draws - 1}")
        ordered = [slot["draws"][d] for d in range(draws)]
        rows[i]["probes"].append({
            "block": int(block), "prefix_len": int(slot["prefix_len"]),
            "forced": ordered[0]["forced"], "forced_tokens": ordered[0]["forced_tokens"],
            "draws": ordered,
        })
    for row in rows:
        row["probes"].sort(key=lambda p: p["block"])
    return rows


def load_reprobe_rows(path, wanted) -> dict:
    wanted = set(wanted)
    found: dict = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = (str(row["qkey"]), int(row["rep"]))
            if identity not in wanted:
                raise SystemExit(f"--reprobe-chains {path}: chain {identity} is not in this "
                                 f"shard's work")
            if identity in found:
                raise SystemExit(f"--reprobe-chains {path}: duplicate chain {identity}")
            try:
                validate_chain(row)
            except ValueError as err:
                raise SystemExit(f"--reprobe-chains {path}: {err}") from err
            if any("draws" in probe for probe in row["probes"]):
                raise SystemExit(f"--reprobe-chains {path}: chain {identity} already carries "
                                 f"draws (re-probe reads single-draw chains only)")
            found[identity] = row
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"--reprobe-chains {path}: {len(missing)} of {len(wanted)} chains "
                         f"missing, e.g. {sorted(missing)[:3]}")
    return found


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--questions")
    source.add_argument("--manifest", help="frozen spec-B JSONL manifest")
    p.add_argument("--split", choices=["fit", "select", "report"], required=True,
                   help="report additionally needs --spend-report: it is the one-shot final split")
    p.add_argument("--spend-report", action="store_true",
                   help="explicit acknowledgement that this run spends the report split "
                        "(2026-08-20: the final-table pass; everything it feeds must be "
                        "frozen choices, no re-tuning downstream)")
    p.add_argument("--split-salt", required=True)
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--chains", type=int, default=8)
    p.add_argument("--max-chains-per-question", type=int, default=0,
                   help="manifest mode only: cap a row's target_chains (0 = the manifest's "
                        "own count). A capped run is a per-shard subset of the uncapped one")
    p.add_argument("--budget", type=int, default=12000)
    p.add_argument("--query-stride", type=int, default=QUERY_STRIDE)
    p.add_argument("--block-cap", type=int, default=128)
    p.add_argument("--caps", default="16,32,64,96,128")
    p.add_argument("--max-forced-tokens", type=int, default=32)
    p.add_argument("--answer-format", choices=sorted(ANSWER_FORMATS), default="boxed",
                   help="code: probes force a ```python program (stop ```), every natural chain gets one "
                        "post-</think> program request, and the outputs are chains.ungraded.jsonl + "
                        "programs.jsonl for iclr_grade_code (verdicts filled in there); the manifest "
                        "rows' answer_format must agree")
    p.add_argument("--force-natural-answer", action="store_true",
                   help="boxed format: take each NATURAL chain's answer from a seeded \\boxed{ "
                        "continuation of its finished thinking instead of from the thinking text "
                        "(the prose answer is kept as natural_answer_prose). For pools whose model "
                        "states the answer in prose -- GPQA-Diamond, where 77-95%% of natural chains "
                        "carry no box and would grade to None")
    p.add_argument("--probe-draws", type=int, default=1,
                   help="K seeded forced-answer draws per probe point (1 = legacy schema; "
                        "K >= 2 adds probes[i].draws, draw 0 keeps the legacy seed)")
    p.add_argument("--reprobe-chains", default=None,
                   help="chains.jsonl of a previous run of THIS shard's work: skip generation, "
                        "copy draw 0 from the stored probes, issue draws 1..K-1 (manifest only). "
                        "--max-questions, --work-shard-index/--work-shard-count, --seed and "
                        "--max-forced-tokens must equal the generation run's")
    p.add_argument("--replay-draw0", action="store_true",
                   help="with --reprobe-chains: also re-issue draw 0 and record "
                        "meta.draw0_replay_agreement (the stored draw 0 is kept)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--work-shard-index", type=int, default=0)
    p.add_argument("--work-shard-count", type=int, default=1)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--model-path", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=40960)
    p.add_argument("--kv-cache-dtype", default=None,
                   help="vLLM KV cache dtype (e.g. fp8). Default None = the engine's own "
                        "default (auto). Halving the KV doubles concurrency on a model with "
                        "no GQA, at the cost of attention numerics, so a run that sets it "
                        "records it and every chain of that run shares the setting")
    p.add_argument("--max-num-seqs", type=int, default=None,
                   help="vLLM max concurrent sequences; None = the engine default")
    p.add_argument("--prompt-contract", choices=sorted(PROMPT_CONTRACTS), default=None,
                   help="which backbone's prompt/sampling contract to build under; the "
                        "default follows $RL_PROMPT_CONTRACT and is the Qwen3-1.7B one")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    if args.probe_draws < 1:
        raise SystemExit(f"--probe-draws must be >= 1, got {args.probe_draws}")
    if args.reprobe_chains is not None:
        if not args.manifest:
            raise SystemExit("--reprobe-chains needs --manifest (the work is resolved from it)")
        if args.probe_draws < 2:
            raise SystemExit("--reprobe-chains needs --probe-draws >= 2 (draw 0 is stored)")
    elif args.replay_draw0:
        raise SystemExit("--replay-draw0 only applies with --reprobe-chains")
    if args.answer_format == "code":
        if args.force_natural_answer:
            raise SystemExit("--force-natural-answer is a boxed-format option; code mode already "
                             "forces a program at the natural end")
        if args.probe_draws != 1:
            raise SystemExit("--answer-format code supports --probe-draws 1 only (A1 on code is out of scope)")
        if args.reprobe_chains is not None:
            raise SystemExit("--answer-format code cannot re-probe stored chains")
    return args


def _load_backend(args):
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path or args.model,
                                              revision=(None if args.model_path else args.revision),
                                              cache_dir=args.hf_cache)
    extra = {}
    if args.kv_cache_dtype:
        extra["kv_cache_dtype"] = args.kv_cache_dtype
    if args.max_num_seqs:
        extra["max_num_seqs"] = args.max_num_seqs
    llm = LLM(model=(args.model_path or args.model),
              revision=(None if args.model_path else args.revision),
              tensor_parallel_size=1, gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len, seed=args.seed,
              enable_prefix_caching=True, download_dir=args.hf_cache, **extra)
    engine = VLLMEngine(llm, sampling=sampling_config(), seed=args.seed)
    return tokenizer, engine


def tokenizer_codec(tokenizer):
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    decode = lambda ids: tokenizer.decode(ids)
    return encode, decode


def chain_prompt_ids(tokenizer, question, *, encode, decode, instruction):
    return rewrite_user_instruction(
        build_prompt_ids(tokenizer, question), encode=encode, decode=decode,
        instruction=instruction,
    )


def _generate_chunked(engine, requests):
    outs = []
    for i in range(0, len(requests), PROBE_CHUNK):
        outs.extend(engine.generate_batch(requests[i: i + PROBE_CHUNK]))
    if len(outs) != len(requests):
        raise RuntimeError(f"engine returned {len(outs)} outputs for {len(requests)} requests")
    return outs


def _check_stored_probe_grid(identity, stored):
    grid = sorted((int(b), int(p)) for b, p in stored["query_points"])
    have = [(int(p["block"]), int(p["prefix_len"])) for p in stored["probes"]]
    if grid != have:
        first = next(((g, h) for g, h in zip(grid, have) if g != h), None)
        detail = (f"first difference: query point {first[0]} vs probe {first[1]}"
                  if first is not None else "one list is a prefix of the other")
        raise SystemExit(f"chain {identity}: {len(have)} probes do not match its "
                         f"{len(grid)} query points ({detail})")


SOURCE_META_REQUIRED = ("max_forced_tokens", "seed")
SOURCE_META_OPTIONAL = ("budget", "block_cap", "query_stride")


def _check_source_summary(reprobe_chains, args) -> dict | None:
    path = Path(reprobe_chains).parent / "summary.json"
    if not path.exists():
        print(f"WARNING: {path} not found; cannot cross-check the source run's "
              f"max_forced_tokens/seed against this run", file=sys.stderr, flush=True)
        return None
    with open(path, encoding="utf-8") as handle:
        meta = json.load(handle).get("meta", {})
    subset = {}
    for field in SOURCE_META_REQUIRED + SOURCE_META_OPTIONAL:
        if field in meta:
            subset[field] = meta[field]
        elif field in SOURCE_META_REQUIRED:
            print(f"warning: {path}: meta lacks {field!r}; relying on the launcher to pin it",
                  file=sys.stderr)
    for field, value in subset.items():
        if value != getattr(args, field):
            raise SystemExit(f"{path}: source run {field}={value!r} but this run has "
                             f"{field}={getattr(args, field)!r}; the re-probe must use the "
                             f"generation run's settings")
    return subset


def _reprobe_source_record(reprobe_chains, source_meta) -> dict:
    record = source_record(reprobe_chains)
    return {"path": record["path"], "sha256": record["sha256"],
            "source_meta": source_meta,
            "code_commit": record["commit"], "code_dirty": record["dirty"]}


def main(argv=None, *, backend=None) -> int:
    args = parse_args(argv)
    select_prompt_contract(args.prompt_contract)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reprobe = args.reprobe_chains is not None

    if args.split == "report" and not args.spend_report:
        raise SystemExit("--split report needs --spend-report (one-shot final split)")
    rows_in = [json.loads(l) for l in open(args.manifest or args.questions) if l.strip()]
    formats = {str(row.get("answer_format") or "boxed") for row in rows_in}
    if formats != {args.answer_format}:
        raise SystemExit(f"--answer-format {args.answer_format} but the rows carry answer_format "
                         f"{sorted(formats)}; the instruction rewrite must match the pool")
    code_mode = args.answer_format == "code"
    instruction = instruction_for(args.answer_format)
    if args.manifest:
        for row in rows_in:
            if question_key(row["question"]) != row["qkey"]:
                raise SystemExit(f"manifest qkey mismatch: {row['qkey']}")
        split_rows = sorted((row for row in rows_in if row["split"] == args.split),
                            key=lambda row: row["qkey"])
        if args.max_questions:
            split_rows = split_rows[:args.max_questions]
        work = assigned_manifest_work(
            split_rows, split=args.split, shard_index=args.work_shard_index,
            shard_count=args.work_shard_count,
            max_chains_per_question=args.max_chains_per_question,
        )
        keyed = {row["qkey"]: row for row in split_rows}
    else:
        keyed = {question_key(r["question"]): r for r in rows_in}
        split_assignment = three_way_split(keyed.keys(), salt=args.split_salt)
        keys = sorted(k for k, value in split_assignment.items() if value == args.split)
        if args.max_questions:
            keys = keys[: args.max_questions]
        work = [(keyed[key], rep) for key in keys for rep in range(args.chains)]
    if not work:
        raise SystemExit(f"no {args.split} work for shard {args.work_shard_index}/{args.work_shard_count}")
    print(json.dumps({"split": args.split,
                      "questions": len({row['qkey'] if args.manifest else question_key(row['question']) for row, _ in work}),
                      "chains": len(work), "work_shard_index": args.work_shard_index,
                      "work_shard_count": args.work_shard_count,
                      "mode": "reprobe" if reprobe else "generate",
                      "probe_draws": args.probe_draws}), flush=True)

    stored: dict = {}
    if reprobe:
        wanted = {(str(row["qkey"]), rep) for row, rep in work}
        stored = load_reprobe_rows(args.reprobe_chains, wanted)
        for row, rep in work:
            identity = (str(row["qkey"]), rep)
            src = stored[identity]
            if int(src["gold"]) != int(row["gold"]):
                raise SystemExit(f"chain {identity}: stored gold {src['gold']} "
                                 f"!= manifest gold {row['gold']}")
            if len(src.get("completion_ids", ())) != int(src["thinking_tokens"]):
                raise SystemExit(f"chain {identity}: completion_ids has "
                                 f"{len(src.get('completion_ids', ()))} tokens but "
                                 f"thinking_tokens is {src['thinking_tokens']}; the stored "
                                 f"prefix is not the one draw 0 was probed on")
            _check_stored_probe_grid(identity, src)
        source_meta = _check_source_summary(args.reprobe_chains, args)
        print(json.dumps({"reprobe_chains_loaded": len(stored),
                          "reprobe_source": args.reprobe_chains,
                          "source_meta": source_meta}), flush=True)

    tokenizer, engine = (backend or _load_backend)(args)
    encode, decode = tokenizer_codec(tokenizer)

    specs = []
    prompts = {}
    for row, rep in work:
        k = str(row["qkey"]) if args.manifest else question_key(row["question"])
        if k not in prompts:
            prompts[k] = chain_prompt_ids(tokenizer, row["question"], encode=encode,
                                          decode=decode, instruction=instruction)
        specs.append({"prompt_ids": prompts[k], "seed_key": (k, rep, "s0chain"),
                      "qkey": k, "rep": rep, "gold": int(row["gold"])})

    chain_rows = []
    items = []
    if reprobe:
        gen_s = 0.0
        for spec in specs:
            src = stored[(spec["qkey"], spec["rep"])]
            row = dict(src)
            row["probes"] = []
            chain_rows.append(row)
            items.append({"prompt_ids": spec["prompt_ids"], "seed_key": spec["seed_key"],
                          "completion_ids": src["completion_ids"],
                          "query_points": [(int(b), int(p)) for b, p in src["query_points"]]})
    else:
        t0 = time.time()
        played = run_chain_batch(engine, specs, encode=encode, budget=args.budget,
                                 query_stride=args.query_stride, block_cap=args.block_cap)
        gen_s = time.time() - t0
        print(json.dumps({"chains_done": len(played), "gen_s": round(gen_s, 1),
                          "mean_tokens": round(sum(len(a["result"].completion_ids)
                                                   for a in played) / len(played), 1)}), flush=True)
        for a in played:
            r = a["result"]
            chain_rows.append({"qkey": a["spec"]["qkey"], "rep": a["spec"]["rep"],
                               "gold": a["spec"]["gold"], "outcome": r.outcome,
                               "answer": r.answer,
                               "correct": (r.answer is not None and r.answer == a["spec"]["gold"]),
                               "blocks": block_count(decode(r.completion_ids)),
                               "thinking_tokens": len(r.completion_ids),
                               "completion_ids": list(r.completion_ids),
                               "block_ends": list(r.block_ends),
                               "query_points": [[b, p] for b, p in a["query_points"]],
                               "probes": []})
            items.append({"prompt_ids": a["spec"]["prompt_ids"], "seed_key": a["spec"]["seed_key"],
                          "completion_ids": r.completion_ids,
                          "query_points": a["query_points"]})

    if code_mode:
        for row in chain_rows:
            row["answer"], row["correct"] = None, False

    first_draw = 1 if reprobe else 0
    probe_reqs, probe_meta = build_probe_requests(
        items, encode=encode, max_forced_tokens=args.max_forced_tokens,
        draws=args.probe_draws, first_draw=first_draw, answer_format=args.answer_format)
    n_probes = sum(len(it["query_points"]) for it in items)
    t0 = time.time()
    probe_outs = _generate_chunked(engine, probe_reqs)
    probe_s = time.time() - t0
    print(json.dumps({"probes": n_probes, "probe_requests": len(probe_outs),
                      "probe_s": round(probe_s, 1)}), flush=True)

    stored_draw0 = None
    if reprobe:
        stored_draw0 = {}
        for i, spec in enumerate(specs):
            for probe in stored[(spec["qkey"], spec["rep"])]["probes"]:
                stored_draw0[(i, int(probe["block"]))] = probe
    attach_probes(chain_rows, probe_meta, probe_outs, draws=args.probe_draws,
                  stored_draw0=stored_draw0, grade=((lambda text: None) if code_mode else grade_forced))

    natural_outs, programs = [], []
    n_prose_answers = n_forced_natural = 0
    if args.force_natural_answer:
        natural_reqs, natural_meta = build_forced_natural_requests(
            chain_rows, items, encode=encode, decode=decode, max_forced_tokens=args.max_forced_tokens)
        t0 = time.time()
        natural_outs = _generate_chunked(engine, natural_reqs)
        natural_s = time.time() - t0
        for row in chain_rows:
            row["natural_answer_prose"] = row["answer"]
        n_prose_answers = sum(r["natural_answer_prose"] is not None for r in chain_rows)
        for i, out in zip(natural_meta, natural_outs):
            answer = grade_forced(out.text)
            chain_rows[i]["answer"] = answer
            chain_rows[i]["correct"] = answer is not None and answer == chain_rows[i]["gold"]
        n_forced_natural = sum(chain_rows[i]["answer"] is not None for i in natural_meta)
        print(json.dumps({"forced_natural_requests": len(natural_outs),
                          "natural_prose_answers": n_prose_answers,
                          "forced_natural_answers": n_forced_natural,
                          "natural_s": round(natural_s, 1)}), flush=True)
    if code_mode:
        natural_reqs, natural_meta = build_natural_requests(
            chain_rows, items, encode=encode, decode=decode, max_forced_tokens=args.max_forced_tokens)
        t0 = time.time()
        natural_outs = _generate_chunked(engine, natural_reqs)
        natural_s = time.time() - t0
        programs = program_records(chain_rows, probe_meta, probe_outs, natural_meta, natural_outs)
        print(json.dumps({"natural_requests": len(natural_outs), "natural_s": round(natural_s, 1),
                          "truncated_programs": sum(p["program"] is None for p in programs)}), flush=True)

    replay = None
    if reprobe and args.replay_draw0:
        replay_reqs, replay_meta = build_probe_requests(
            items, encode=encode, max_forced_tokens=args.max_forced_tokens, draws=1)
        t0 = time.time()
        replay_outs = _generate_chunked(engine, replay_reqs)
        agree = 0
        for (i, block, _plen, _draw), out in zip(replay_meta, replay_outs):
            kept = stored_draw0[(i, int(block))]["forced"]
            got = grade_forced(out.text)
            agree += int(type(got) is type(kept) and got == kept)
        replay = {"draw0_replay_probes": len(replay_meta),
                  "draw0_replay_agreement": (agree / len(replay_meta) if replay_meta else None),
                  "draw0_replay_s": round(time.time() - t0, 1)}
        print(json.dumps(replay), flush=True)

    if reprobe:
        for row in chain_rows:
            try:
                validate_chain(row)
            except ValueError as err:
                raise SystemExit(f"re-probed chain failed the contract: {err}") from err

    if code_mode:
        for row in chain_rows:
            try:
                validate_chain(row)
            except ValueError as err:
                raise SystemExit(f"ungraded code chain failed the contract: {err}") from err
        with open(out_dir / "chains.ungraded.jsonl", "w") as f:
            for row in chain_rows:
                f.write(json.dumps(row) + "\n")
        with open(out_dir / "programs.jsonl", "w") as f:
            for record in programs:
                f.write(json.dumps(record) + "\n")
    else:
        with open(out_dir / "chains.jsonl", "w") as f:
            for row in chain_rows:
                f.write(json.dumps(row) + "\n")
        with open(out_dir / "rows.jsonl", "w") as f:
            for row in chain_rows:
                for probe in row["probes"]:
                    f.write(json.dumps({"qkey": row["qkey"], "rep": row["rep"],
                                        "gold": row["gold"], "natural_answer": row["answer"],
                                        "thinking_tokens": row["thinking_tokens"],
                                        **probe}) + "\n")

    caps = [int(c) for c in args.caps.split(",") if c.strip()]
    report = {"rows": [], "gate_s0": None} if code_mode else oracle_report(chain_rows, caps=caps)
    report["meta"] = {"split": args.split,
                      "questions": len({spec["qkey"] for spec in specs}),
                      "chains": len(chain_rows), "probes": n_probes,
                      "probe_requests": len(probe_outs),
                      "budget": args.budget, "query_stride": args.query_stride,
                      "max_chains_per_question": args.max_chains_per_question,
                      "kv_cache_dtype": args.kv_cache_dtype,
                      "max_num_seqs": args.max_num_seqs,
                      "block_cap": args.block_cap, "caps": caps, "seed": args.seed,
                      "work_shard_index": args.work_shard_index,
                      "work_shard_count": args.work_shard_count,
                      "gen_s": round(gen_s, 1), "probe_s": round(probe_s, 1),
                      "mode": "reprobe" if reprobe else "generate",
                      "probe_draws": args.probe_draws,
                      "draw_seed_rule": DRAW_SEED_RULE,
                      "max_forced_tokens": args.max_forced_tokens,
                      "answer_format": args.answer_format,
                      "prompt_contract": contract_identity(),
                      "graded": not code_mode,
                      "natural_requests": len(natural_outs) if code_mode else 0,
                      "force_natural_answer": args.force_natural_answer,
                      "forced_natural_requests": len(natural_outs) if args.force_natural_answer else 0,
                      "natural_prose_answers": n_prose_answers,
                      "forced_natural_answers": n_forced_natural,
                      "probe_tokens_total": sum(len(o.token_ids) for o in probe_outs),
                      "natural_answer_tokens_total": sum(len(o.token_ids) for o in natural_outs),
                      "truncated_programs": sum(p["program"] is None for p in programs),
                      "programs_over_512": sum(p["tokens"] > 512 for p in programs),
                      "reprobe_source": (_reprobe_source_record(args.reprobe_chains, source_meta)
                                         if reprobe else None)}
    if replay is not None:
        report["meta"].update(replay)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(report, f, indent=2)
    for r in report["rows"]:
        print(json.dumps(r), flush=True)
    print(json.dumps({"gate_s0": report["gate_s0"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
