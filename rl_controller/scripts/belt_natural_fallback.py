"""Elicit a final answer for naturally closed chains without a parseable answer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from array import array
from pathlib import Path

from rl_controller.iclr.chains import is_correct
from rl_controller.rollout_contract import (
    OLMO3_CONTRACT,
    QWEN_CONTRACT,
    contract_identity,
    forced_answer_stop,
    grade_forced,
    instruction_for,
    sampling_config,
    select_prompt_contract,
)
from rl_controller.scripts import single_cot_probe as scp
from rl_controller.splits import question_key

CONTRACTS = {"qwen3": QWEN_CONTRACT, "olmo3": OLMO3_CONTRACT}
DEFAULT_MAX_MODEL_LEN = {"qwen3": 40960, "olmo3": 20480}
OUT_NAME = "natural_fallback.jsonl"
REPO_ROOT = Path(__file__).resolve().parents[2]


def is_selected(row: dict) -> bool:
    return row["outcome"] == "natural" and row["answer"] is None


def stream_selected(path, *, limit=None):
    digest = hashlib.sha256()
    selected, seen = [], set()
    n_rows = n_natural = n_selected = 0
    with open(path, "rb") as handle:
        for raw in handle:
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            n_rows += 1
            identity = (str(row["qkey"]), int(row["rep"]))
            if identity in seen:
                raise SystemExit(f"{path}: duplicate chain {identity}")
            seen.add(identity)
            n_natural += row["outcome"] == "natural"
            if not is_selected(row):
                continue
            n_selected += 1
            if limit is not None and len(selected) >= limit:
                continue
            completion = row.get("completion_ids")
            if not completion or len(completion) != int(row["thinking_tokens"]):
                raise SystemExit(f"{path}: chain {identity} completion_ids "
                                 f"({len(completion or ())}) != thinking_tokens "
                                 f"({row['thinking_tokens']})")
            selected.append({"qkey": identity[0], "rep": identity[1], "gold": row["gold"],
                             "outcome": row["outcome"],
                             "thinking_tokens": int(row["thinking_tokens"]),
                             "completion_ids": array("q", completion)})
    stats = {"rows": n_rows, "natural": n_natural, "selected_total": n_selected,
             "selected_kept": len(selected)}
    return selected, digest.hexdigest(), stats


def load_manifest(path):
    data = Path(path).read_bytes()
    rows = {}
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if question_key(row["question"]) != row["qkey"]:
            raise SystemExit(f"manifest {path}: qkey {row['qkey']} is not its question's key")
        rows[row["qkey"]] = row
    return rows, hashlib.sha256(data).hexdigest()


def check_against_manifest(selected, manifest, path):
    missing = sorted({r["qkey"] for r in selected if r["qkey"] not in manifest})
    if missing:
        raise SystemExit(f"manifest {path} lacks {len(missing)} selected qkeys, e.g. {missing[:3]}")
    for r in selected:
        m = manifest[r["qkey"]]
        if m["gold"] != r["gold"] or type(m["gold"]) is not type(r["gold"]):
            raise SystemExit(f"chain {(r['qkey'], r['rep'])}: gold {r['gold']!r} != manifest "
                             f"gold {m['gold']!r}")
        fmt = str(m.get("answer_format") or "boxed")
        if fmt != "boxed":
            raise SystemExit(f"manifest row {r['qkey']} has answer_format {fmt!r}; the "
                             f"fallback is the boxed-format one")


def git_state():
    def run(*args):
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True,
                                 cwd=REPO_ROOT, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout if out.returncode == 0 else None
    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return (head.strip() if head else None), (bool(status.strip()) if status is not None else None)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chains", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--bank-name", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--model-path", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--seed", type=int, default=0,
                   help="the bank's base seed (VLLMEngine seed); every production bank is 0")
    p.add_argument("--prompt-contract", choices=sorted(CONTRACTS), default="qwen3")
    p.add_argument("--max-model-len", type=int, default=None,
                   help="default: the generator's (qwen3 40960, olmo3 20480)")
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--max-forced-tokens", type=int, default=32)
    p.add_argument("--limit", type=int, default=None, help="smoke: keep the first N selected chains")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    if args.max_model_len is None:
        args.max_model_len = DEFAULT_MAX_MODEL_LEN[args.prompt_contract]
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    args.kv_cache_dtype = None
    args.max_num_seqs = None
    return args


def _vllm_version():
    try:
        import vllm
        return getattr(vllm, "__version__", None)
    except Exception:
        return None


def main(argv=None, *, backend=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_path = out_dir / OUT_NAME
    marker = out_dir / (OUT_NAME + ".sha256")
    if out_path.exists() and marker.exists():
        print(json.dumps({"skip": str(out_path), "reason": "output and marker exist"}), flush=True)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    contract = select_prompt_contract(CONTRACTS[args.prompt_contract])
    commit, dirty = git_state()

    selected, chains_sha, stats = stream_selected(args.chains, limit=args.limit)
    manifest, manifest_sha = load_manifest(args.manifest)
    check_against_manifest(selected, manifest, args.manifest)
    print(json.dumps({"bank": args.bank_name, "chains_sha256": chains_sha,
                      "manifest_sha256": manifest_sha, "contract": contract,
                      "commit": commit, "dirty": dirty, **stats}), flush=True)

    rows_out = []
    seconds_gen = 0.0
    engine = None
    if selected:
        tokenizer, engine = (backend or scp._load_backend)(args)
        encode, decode = scp.tokenizer_codec(tokenizer)
        instruction = instruction_for("boxed")
        prompts = {}
        for start in range(0, len(selected), scp.PROBE_CHUNK):
            chunk = selected[start: start + scp.PROBE_CHUNK]
            items = []
            for r in chunk:
                if r["qkey"] not in prompts:
                    prompts[r["qkey"]] = scp.chain_prompt_ids(
                        tokenizer, manifest[r["qkey"]]["question"], encode=encode,
                        decode=decode, instruction=instruction)
                items.append({"prompt_ids": prompts[r["qkey"]],
                              "seed_key": (r["qkey"], r["rep"], "s0chain"),
                              "completion_ids": list(r["completion_ids"])})
            requests, meta = scp.build_forced_natural_requests(
                chunk, items, encode=encode, decode=decode,
                max_forced_tokens=args.max_forced_tokens)
            if meta != list(range(len(chunk))):
                raise RuntimeError(f"expected one request per selected chain, got meta {meta[:5]}...")
            if start == 0:
                first = requests[0]
                print(json.dumps({"audit_first_request": {
                    "qkey": chunk[0]["qkey"], "rep": chunk[0]["rep"],
                    "seed_key": repr(first.seed_key), "max_tokens": first.max_tokens,
                    "stop": first.stop, "include_stop_str_in_output": first.include_stop_str_in_output,
                    "prompt_tokens": len(first.prompt_ids),
                    "prompt_tail": decode(list(first.prompt_ids))[-200:]}}), flush=True)
            t0 = time.time()
            outs = scp._generate_chunked(engine, requests)
            seconds_gen += time.time() - t0
            for r, out in zip(chunk, outs):
                forced = grade_forced(out.text)
                rows_out.append({"qkey": r["qkey"], "rep": r["rep"], "forced": forced,
                                 "forced_text": out.text, "forced_tokens": len(out.token_ids),
                                 "finish_reason": out.finish_reason, "gold": r["gold"],
                                 "correct": is_correct(forced, r["gold"]),
                                 "thinking_tokens": r["thinking_tokens"],
                                 "bank": args.bank_name, "chains_sha256": chains_sha,
                                 "manifest_sha256": manifest_sha, "commit": commit})
            print(json.dumps({"done": len(rows_out), "of": len(selected),
                              "gen_s": round(seconds_gen, 1)}), flush=True)
            for r in chunk:
                r["completion_ids"] = None

    tmp = out_path.with_suffix(".jsonl.partial")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, out_path)
    out_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()

    sampling = dict(getattr(engine, "_sampling", None) or sampling_config())
    summary = {
        "bank": args.bank_name,
        "chains": {"path": str(Path(args.chains).resolve()), "sha256": chains_sha},
        "manifest": {"path": str(Path(args.manifest).resolve()), "sha256": manifest_sha},
        "counts": {**stats, "requests": len(rows_out),
                   "parseable": sum(r["forced"] is not None for r in rows_out),
                   "correct": sum(r["correct"] for r in rows_out),
                   "finish_reasons": {k: sum(r["finish_reason"] == k for r in rows_out)
                                      for k in sorted({str(r["finish_reason"]) for r in rows_out})},
                   "forced_tokens_total": sum(r["forced_tokens"] for r in rows_out)},
        "seconds": {"total": round(time.time() - t_start, 1), "generate": round(seconds_gen, 1)},
        "vllm_version": _vllm_version(),
        "sampling": {"engine_sampling": sampling, "engine_seed": args.seed,
                     "max_tokens": args.max_forced_tokens, **forced_answer_stop(),
                     "seed_key_rule": "((qkey, rep, 's0chain'), 'natural')",
                     "request_seed": "rollout_runner.request_seed(engine_seed, seed_key)"},
        "prompt_contract": args.prompt_contract,
        "contract_identity": contract_identity(),
        "model_path": args.model_path or args.model,
        "max_model_len": args.max_model_len,
        "gpu_mem_util": args.gpu_mem_util,
        "limit": args.limit,
        "commit": commit, "dirty": dirty,
        "argv": sys.argv if argv is None else list(argv),
        "output": {"path": str(out_path.resolve()), "sha256": out_sha},
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(marker, "w", encoding="utf-8") as f:
        f.write(f"{out_sha}  {OUT_NAME}\n")
    print(json.dumps({"complete": str(out_path), "sha256": out_sha, **summary["counts"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
