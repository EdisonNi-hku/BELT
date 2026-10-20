"""Per-boundary confidence, entropy and end-of-thinking rank features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from rl_controller.baselines.trainingfree import probe_spec as ps

DEFAULT_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

DEFAULT_BATCH = 8


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chains", required=True, help="pool chains.jsonl (must carry completion_ids)")
    p.add_argument("--questions", required=True,
                   help="the pool question manifest -- the SAME one the pool was generated "
                        "with. The prompt is part of the state.")
    p.add_argument("--chains-sha256", default=None,
                   help="refuse to run against a chains file whose sha256 differs")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--qkeys", default=None,
                   help="optional jsonl/txt of qkeys to restrict to (one per line, or jsonl "
                        "rows with a 'qkey' field). Used to run only the shared val questions "
                        "of a fit pool.")
    p.add_argument("--mode", default="teacher_forced", choices=list(ps.MODES),
                   help="teacher_forced = the cheap estimator (E_tf); greedy = CoDE-Stop's "
                        "own probe decoded freely to </think> (E_greedy, the fidelity "
                        "reference of prereg 6c)")
    p.add_argument("--greedy-steps", type=int, default=21,
                   help="their cap: method_codestop.py breaks after total_steps > 20")
    p.add_argument("--templates", default=",".join(ps.DEFAULT_TEMPLATES),
                   help=f"comma-separated, from {sorted(ps.TEMPLATES)}")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--max-len", type=int, default=8192,
                   help="skip (and record) any probe sequence longer than this")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--model-path", default=None)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--limit-chains", type=int, default=None,
                   help="smoke only: process at most this many chains of this shard")
    return p.parse_args(argv)


def load_qkey_filter(path):
    from rl_controller.splits import question_key

    keys = set()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            row = json.loads(line)
            keys.add(row["qkey"] if "qkey" in row else question_key(str(row["question"])))
        else:
            keys.add(line)
    if not keys:
        raise SystemExit(f"--qkeys {path} yielded no keys")
    return keys


def shard_of(qkey: str, rep: int, num_shards: int) -> int:
    digest = hashlib.blake2b(f"{qkey}:{int(rep)}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % num_shards


def stream_shard(path, *, shard: int, num_shards: int, qkeys=None, limit=None):
    from rl_controller.iclr.chains import validate_chain

    if not 0 <= shard < num_shards:
        raise SystemExit(f"--shard {shard} outside [0, {num_shards})")
    kept, seen = [], set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ident = (row["qkey"], int(row["rep"]))
            if ident in seen:
                raise SystemExit(f"duplicate chain {ident} in {path}")
            seen.add(ident)
            if qkeys is not None and row["qkey"] not in qkeys:
                continue
            if shard_of(row["qkey"], row["rep"], num_shards) != shard:
                continue
            kept.append(validate_chain(row))
            if limit is not None and len(kept) >= limit:
                break
    kept.sort(key=lambda c: (c["qkey"], int(c["rep"])))
    return kept, len(seen)


def _close_think_id(encode) -> int:
    ids = encode("</think>")
    if not ids:
        raise SystemExit("tokenizer encodes '</think>' to nothing")
    if len(ids) > 1:
        print(json.dumps({"warn": "close_think_multi_token", "ids": ids,
                          "note": "rank is of the FIRST token only, as in method_wheels.py:67"}),
              flush=True)
    return int(ids[0])


def refuse_non_boxed(qrows, chains) -> None:
    from rl_controller.rollout_contract import ContractError, answer_formats

    try:
        formats = answer_formats(qrows)
    except ContractError as exc:
        raise SystemExit(f"--questions: {exc}") from None
    bad = sorted({formats[c["qkey"]] for c in chains
                  if formats.get(c["qkey"], "boxed") != "boxed"})
    if bad:
        raise SystemExit(
            f"this pool's answer_format is {bad} -- the canonical re-rendering "
            "'\\boxed{<forced>}' is only defined for the boxed format. A code pool's "
            "`forced` is a pass/fail VERDICT (1/0), not an answer, so it would pass a "
            "type check and produce a meaningless confidence. Mark the pool N/A for "
            "every confidence-based rule (prereg-ws-g.md section 6); do not force it "
            "through this pass.")


def run_batch(model, torch, batch, *, device, close_id):
    ids, mask, pos = ps.pad_batch([s.ids for s in batch], pad_id=0)
    tail = max(s.tail_len for s in batch)
    width = len(ids[0])
    if tail > min(len(s.ids) for s in batch):
        raise RuntimeError(
            f"kept tail {tail} exceeds the shortest sequence in the batch "
            f"({min(len(s.ids) for s in batch)}); the tail would read padding")
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(mask, dtype=torch.long, device=device)
    position_ids = torch.tensor(pos, dtype=torch.long, device=device)
    with torch.no_grad():
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, logits_to_keep=tail)
        except TypeError:
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, num_logits_to_keep=tail)
        logits = out.logits.float()
        if logits.shape[1] != tail:
            raise RuntimeError(
                f"model returned {logits.shape[1]} logit rows for logits_to_keep={tail}; "
                "this transformers build does not honour the flag, so every readout index "
                "would be wrong -- refusing rather than emitting silently shifted numbers")
        logprobs = torch.log_softmax(logits, dim=-1)
        probs = logprobs.exp()
        entropy = -(probs * logprobs).sum(dim=-1)
        pmax, argmax = probs.max(dim=-1)
        close_logit = logits[:, :, close_id]
        close_rank = (logits > close_logit.unsqueeze(-1)).sum(dim=-1) + 1
        close_prob = probs[:, :, close_id]

    rows = []
    for b, seq in enumerate(batch):
        seq_len = len(seq.ids)
        pi = ps.tail_index(seq_len, tail, seq.n_prefix - 1)
        a_idx = [ps.tail_index(seq_len, tail, seq.n_prefix + seq.n_seed + j - 1)
                 for j in range(seq.n_answer)]
        tokens = ps.answer_token_ids(seq)
        p_forced = [probs[b, t, tok].item() for t, tok in zip(a_idx, tokens)]
        rows.append(ps.make_row(
            seq,
            p_forced=p_forced,
            p_max=[pmax[b, t].item() for t in a_idx],
            argmax_hit=[int(argmax[b, t].item() == tok) for t, tok in zip(a_idx, tokens)],
            h_answer=[entropy[b, t].item() for t in a_idx],
            h_prefix=entropy[b, pi].item(),
            rank_close=int(close_rank[b, pi].item()),
            p_close=close_prob[b, pi].item(),
        ))
    _ = width
    return rows


def stop_ids_for(template: str, encode) -> set:
    ids = set()
    for word in ps.TEMPLATES[template].stop_words:
        got = encode(word)
        if got:
            ids.add(int(got[0]))
    if not ids:
        raise SystemExit(f"template {template!r} declares no usable stop token")
    return ids


def run_greedy_batch(model, torch, batch, *, device, close_id, max_steps, stop_ids_by_template):
    ids, mask, pos = ps.pad_batch([s.ids for s in batch], pad_id=0)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(mask, dtype=torch.long, device=device)
    position_ids = torch.tensor(pos, dtype=torch.long, device=device)
    state = ps.GreedyState([len(s.ids) for s in batch],
                           [stop_ids_by_template[s.template] for s in batch], max_steps)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()
        while not state.done():
            probs = torch.softmax(logits, dim=-1)
            pmax, argmax = probs.max(dim=-1)
            state.step(argmax.tolist(), pmax.tolist())
            if state.done():
                break
            next_pos = torch.tensor([[p] for p in state.positions()],
                                    dtype=torch.long, device=device)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((len(batch), 1), dtype=torch.long, device=device)],
                dim=1)
            step_out = model(input_ids=argmax.unsqueeze(-1), attention_mask=attention_mask,
                             position_ids=next_pos, past_key_values=past, use_cache=True)
            past = step_out.past_key_values
            logits = step_out.logits[:, -1, :].float()

    rows = []
    for b, seq in enumerate(batch):
        tokens, probs = state.tokens[b], state.probs[b]
        if not tokens:
            continue
        generated = ps.ProbeSequence(
            qkey=seq.qkey, rep=seq.rep, probe_index=seq.probe_index, block=seq.block,
            prefix_len=seq.prefix_len, template=seq.template,
            ids=tuple(seq.ids) + tuple(tokens), n_prefix=seq.n_prefix, n_seed=seq.n_seed,
            n_answer=len(tokens), forced_tokens=seq.forced_tokens, forced=seq.forced,
            content_mask=tuple([1] * len(tokens)), mode="greedy")
        row = ps.make_row(
            generated,
            p_forced=probs, p_max=probs, argmax_hit=[1] * len(tokens),
            h_answer=[0.0] * len(tokens),
            h_prefix=0.0, rank_close=1, p_close=0.0)
        row["ended_with_think"] = bool(state.ended_with_stop[b])
        ps.validate_row(row)
        rows.append(row)
    return rows


def main(argv=None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rl_controller.iclr.provenance import source_record

    args = parse_args(argv)
    ps.check_seed_matches_contract()
    templates = tuple(t.strip() for t in args.templates.split(",") if t.strip())
    unknown = [t for t in templates if t not in ps.TEMPLATES]
    if unknown:
        raise SystemExit(f"unknown template(s) {unknown}; choose from {sorted(ps.TEMPLATES)}")

    chains_prov = source_record(args.chains)
    if args.chains_sha256 and chains_prov["sha256"] != args.chains_sha256:
        raise SystemExit(f"--chains sha256 {chains_prov['sha256']} != expected {args.chains_sha256}")

    keep = load_qkey_filter(args.qkeys) if args.qkeys else None
    mine, n_pool = stream_shard(args.chains, shard=args.shard, num_shards=args.num_shards,
                                qkeys=keep, limit=args.limit_chains)
    if not mine:
        raise SystemExit(
            f"shard {args.shard}/{args.num_shards} is empty (pool has {n_pool} chains"
            + (f", qkey filter {args.qkeys}" if keep else "") + ")")
    if "completion_ids" not in mine[0]:
        raise SystemExit(
            "chains carry no completion_ids -- this is a SCORED dump, not a pool. The "
            "confidence pass needs the raw token ids to rebuild each probe prefix.")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model_ref = args.model_path or args.model
    if args.model_path and not str(args.model_path).rstrip("/").endswith(args.revision):
        raise SystemExit(f"--model-path {args.model_path} does not end in --revision {args.revision}")
    revision = None if args.model_path else args.revision
    tokenizer = AutoTokenizer.from_pretrained(model_ref, revision=revision,
                                              cache_dir=args.hf_cache, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref, revision=revision, cache_dir=args.hf_cache,
        torch_dtype=getattr(torch, args.dtype)).to(device).eval()
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    stop_by_template = {name: stop_ids_for(name, encode) for name in ps.TEMPLATES}
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
    close_id = _close_think_id(encode)

    from rl_controller.iclr import scoring_io as sio

    qrows = [json.loads(line) for line in open(args.questions, encoding="utf-8") if line.strip()]
    refuse_non_boxed(qrows, mine)
    prompts = sio.build_prompt_table(qrows, tokenizer)
    missing = sorted({c["qkey"] for c in mine} - set(prompts))
    if missing:
        raise SystemExit(f"{len(missing)} chain questions are not in --questions "
                         f"(first {missing[:3]}) -- wrong manifest for this pool")

    seqs, skipped = [], []
    for chain in mine:
        if args.mode == "greedy":
            chain_seqs, chain_skips = ps.build_greedy_sequences(
                chain, prompts[chain["qkey"]], encode=encode, templates=templates,
                max_len=args.max_len)
        else:
            chain_seqs, chain_skips = ps.build_sequences(
                chain, prompts[chain["qkey"]], encode=encode, decode=decode,
                templates=templates, max_len=args.max_len)
        seqs.extend(chain_seqs)
        skipped.extend(chain_skips)
    if not seqs:
        raise SystemExit("no probe sequences to score (every boundary was skipped)")

    seqs.sort(key=lambda s: len(s.ids))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, done, t0, batch_size = [], 0, time.time(), args.batch
    i = 0
    while i < len(seqs):
        batch = seqs[i: i + batch_size]
        try:
            got = (run_greedy_batch(model, torch, batch, device=device, close_id=close_id,
                                    max_steps=args.greedy_steps,
                                    stop_ids_by_template=stop_by_template)
                   if args.mode == "greedy"
                   else run_batch(model, torch, batch, device=device, close_id=close_id))
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            print(json.dumps({"oom": True, "new_batch": batch_size}), flush=True)
            continue
        rows.extend(got)
        i += len(batch)
        done += len(batch)
        if done % 512 < batch_size:
            rate = done / max(1e-9, time.time() - t0)
            print(json.dumps({"scored": done, "of": len(seqs), "seq_per_s": round(rate, 2)}),
                  flush=True)

    rows.sort(key=ps.row_key)
    shard_path = out_dir / f"conf_{args.shard:04d}.jsonl"
    with open(shard_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    answered = [r for r in rows if r["n_answer"] > 0]
    ewt = [r for r in rows if r.get("ended_with_think") is not None]
    n_hit = sum(1 for r in answered if all(r["argmax_hit"]))
    n_tok_match = sum(1 for r in answered if r["n_answer"] == r["forced_tokens"])
    manifest = ps.write_manifest(
        out_dir / f"conf_{args.shard:04d}-manifest.json.txt",
        rows_written=len(rows), shards=[shard_path],
        extra={"chains": chains_prov, "questions": source_record(args.questions),
               "model": model_ref, "revision": args.revision, "dtype": args.dtype,
               "templates": list(templates), "mode": args.mode,
               "greedy_steps": args.greedy_steps, "shard": args.shard,
               "num_shards": args.num_shards, "qkeys_filter": args.qkeys,
               "n_chains": len(mine), "n_pool_chains": n_pool, "n_sequences": len(seqs),
               "skipped": skipped[:200], "n_skipped": len(skipped),
               "close_think_token_id": close_id,
               "stop_ids_by_template": {k: sorted(v) for k, v in stop_by_template.items()},
               "n_answered": len(answered), "n_unparsed": len(rows) - len(answered),
               "ended_with_think_rate": (round(sum(1 for r in ewt if r["ended_with_think"])
                                               / len(ewt), 4) if ewt else None),
               "greedy_path_agreement": round(n_hit / max(1, len(answered)), 4),
               "forced_tokens_agreement": round(n_tok_match / max(1, len(answered)), 4),
               "commit": os.environ.get("EXPECTED_COMMIT", "")})
    print(json.dumps({"rows": len(rows), "chains": len(mine), "skipped": len(skipped),
                      "n_unparsed": len(rows) - len(answered),
                      "greedy_path_agreement": manifest["greedy_path_agreement"],
                      "forced_tokens_agreement": manifest["forced_tokens_agreement"],
                      "out": str(shard_path)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
