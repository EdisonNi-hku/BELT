"""Build the shared block-feature cache used by baseline detectors."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from rl_controller.iclr.chains import is_trainable, validate_chain
from rl_controller.raw_tap import (assert_strictly_increasing, assert_within_features,
                                   end_feature_index)
from rl_controller.single_cot import probe_positions_in_ends

SCHEMA = "iclr-block-features/v1"
VARIANTS = ("mean", "last")
POSITIONS = ("all", "probed")
STORE_DTYPE = "float16"


def layer_key(layer) -> str:
    return f"L{int(layer)}"


def resolve_layers(requested) -> list[int]:
    out = sorted({int(x) for x in requested})
    if not out:
        raise ValueError("no layers requested")
    return out


def variant_names(layers) -> list[str]:
    return [f"{layer_key(layer)}_{v}" for layer in layers for v in VARIANTS]


def feature_len(prompt_len: int, block_ends) -> int:
    ends = [int(e) for e in block_ends]
    if not ends:
        raise ValueError("block_ends is empty")
    return int(prompt_len) + ends[-1] - 1


def block_spans(prompt_len: int, block_ends) -> list[tuple[int, int, int]]:
    prompt_len = int(prompt_len)
    if prompt_len < 1:
        raise ValueError(f"prompt_len {prompt_len} must be at least 1")
    ends = [int(e) for e in block_ends]
    if not ends:
        raise ValueError("block_ends is empty")
    if any(e <= 0 for e in ends):
        raise ValueError(f"block_ends must be positive, got {ends[:8]}")
    if ends != sorted(ends) or len(set(ends)) != len(ends):
        raise ValueError(f"block_ends must be strictly increasing, got {ends[:8]}")
    T = feature_len(prompt_len, ends)
    spans = []
    start_offset = 0
    for b in ends:
        end_col = end_feature_index(prompt_len, b - 1)
        span_start = prompt_len + start_offset - 1
        spans.append((span_start, end_col + 1, end_col))
        start_offset = b
    assert_strictly_increasing([s[2] for s in spans])
    assert_within_features([s[2] for s in spans], T)
    for span_start, span_stop, end_col in spans:
        if span_start < 0 or span_start >= span_stop or span_stop > T:
            raise ValueError(f"block span [{span_start}, {span_stop}) outside [0, {T})")
        if span_stop - 1 != end_col:
            raise ValueError(f"span stop {span_stop} does not end at end column {end_col}")
    widths = [stop - start for start, stop, _ in spans]
    expected = [ends[0]] + [ends[i] - ends[i - 1] for i in range(1, len(ends))]
    if widths != expected:
        raise ValueError(f"span widths {widths[:8]} != block token counts {expected[:8]}")
    return spans


def prompt_segments(prompt_len: int, n_segments: int) -> list[tuple[int, int, int]]:
    prompt_len = int(prompt_len)
    n_segments = int(n_segments)
    if n_segments < 1:
        raise ValueError(f"n_segments must be >= 1, got {n_segments}")
    width = prompt_len - 1
    if width < n_segments:
        raise ValueError(f"prompt of {prompt_len} tokens has {width} prompt-only feature "
                         f"columns, fewer than the {n_segments} segments requested")
    out, start = [], 0
    for i in range(n_segments):
        stop = ((i + 1) * width) // n_segments
        if stop <= start:
            raise ValueError(f"empty prompt segment {i} for width {width}")
        out.append((start, stop, stop - 1))
        start = stop
    if out[-1][1] != width:
        raise ValueError(f"prompt segments end at {out[-1][1]}, not {width}")
    return out


def check_positions(positions: str) -> str:
    if positions not in POSITIONS:
        raise ValueError(f"positions must be one of {POSITIONS}, got {positions!r}")
    return positions


def n_rows_for(chain: dict, positions: str) -> int:
    check_positions(positions)
    if not is_trainable(chain):
        return 0
    return len(chain["probes"]) if positions == "probed" else len(chain["block_ends"])


def chain_index_rows(chain: dict, prompt_len: int, *, row_offset: int = 0,
                     positions: str = "all") -> list[dict]:
    check_positions(positions)
    ends = [int(e) for e in chain["block_ends"]]
    if [qp[1] for qp in chain["query_points"]] != [p["prefix_len"] for p in chain["probes"]]:
        raise ValueError(f"chain {(chain['qkey'], chain['rep'])}: query_points prefixes do not "
                         "align with probes[].prefix_len")
    probe_pos = probe_positions_in_ends(ends, chain["query_points"])
    if len(set(probe_pos)) != len(probe_pos):
        raise ValueError(f"chain {(chain['qkey'], chain['rep'])}: two probes share a block end "
                         f"(probe positions {probe_pos})")
    at_probe = {pos: i for i, pos in enumerate(probe_pos)}
    spans = block_spans(prompt_len, ends)
    rows = []
    for end_index, (prefix_len, (span_start, span_stop, end_col)) in enumerate(zip(ends, spans)):
        probe_index = at_probe.get(end_index)
        if positions == "probed" and probe_index is None:
            continue
        probe = chain["probes"][probe_index] if probe_index is not None else None
        rows.append({
            "qkey": chain["qkey"], "rep": chain["rep"],
            "end_index": end_index, "prefix_len": prefix_len,
            "probe_index": probe_index,
            "block": (probe["block"] if probe is not None else None),
            "span_start": span_start, "span_len": span_stop - span_start,
            "end_col": end_col,
            "row": row_offset + len(rows),
        })
    if len(rows) != n_rows_for(chain, positions):
        raise ValueError(f"chain {(chain['qkey'], chain['rep'])}: kept {len(rows)} rows but "
                         f"planned {n_rows_for(chain, positions)} for positions={positions!r}")
    return rows


def kept_spans(chain: dict, prompt_len: int, positions: str) -> list[tuple[int, int, int]]:
    spans = block_spans(prompt_len, [int(e) for e in chain["block_ends"]])
    return [spans[row["end_index"]]
            for row in chain_index_rows(chain, prompt_len, positions=positions)]


def shard_of(index: int, shard_count: int) -> int:
    if shard_count < 1:
        raise ValueError(f"shard_count {shard_count} must be >= 1")
    return int(index) % int(shard_count)


def plan_shard(chains, *, shard_index: int, shard_count: int, positions: str = "all"):
    check_positions(positions)
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index {shard_index} outside [0, {shard_count})")
    picked, offset = [], 0
    for i, chain in enumerate(chains):
        if shard_of(i, shard_count) != shard_index:
            continue
        n = n_rows_for(chain, positions)
        if not n:
            continue
        picked.append((chain, offset))
        offset += n
    return picked, offset


def iter_shard_chains(path, *, shard_index: int, shard_count: int, positions: str = "all",
                      limit: int = 0):
    check_positions(positions)
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index {shard_index} outside [0, {shard_count})")
    picked, offset, index = [], 0, 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            i, index = index, index + 1
            if shard_of(i, shard_count) != shard_index:
                continue
            chain = validate_chain(json.loads(line))
            n = n_rows_for(chain, positions)
            if not n:
                continue
            picked.append((chain, offset))
            offset += n
            if limit and len(picked) >= limit:
                break
    if limit:
        kept = [chain for chain, _ in picked]
        picked = list(zip(kept, _prefix_offsets([n_rows_for(c, positions) for c in kept])))
        offset = sum(n_rows_for(c, positions) for c in kept)
    seen = set()
    for chain, _ in picked:
        ident = (chain["qkey"], chain["rep"])
        if ident in seen:
            raise ValueError(f"duplicate chain {ident} in {path}")
        seen.add(ident)
    return picked, offset


def pool_span(rows, start: int, stop: int) -> list[float]:
    window = rows[start:stop]
    if not window:
        raise ValueError(f"empty span [{start}, {stop})")
    width = len(window[0])
    return [sum(row[j] for row in window) / len(window) for j in range(width)]


def reference_layer_vectors(feature_rows, prompt_len: int, block_ends) -> list[list[list[float]]]:
    out = []
    for span_start, span_stop, end_col in block_spans(prompt_len, block_ends):
        out.append([pool_span(feature_rows, span_start, span_stop),
                    [float(x) for x in feature_rows[end_col]]])
    return out


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


FINGERPRINT_FIELDS = ("chains_sha256", "questions_sha256", "prompt_sha256", "layers",
                      "positions", "dtype_compute", "model_path", "model_revision",
                      "git_commit")


def input_fingerprint(*, chains_sha, questions_sha, layers, positions, dtype, model_path,
                      revision, commit, prompt_sha=None) -> dict:
    return {"chains_sha256": chains_sha, "questions_sha256": questions_sha,
            "prompt_sha256": prompt_sha, "layers": [int(x) for x in layers],
            "positions": positions, "dtype_compute": dtype,
            "model_path": str(model_path), "model_revision": revision,
            "git_commit": commit}


def fingerprint_drift(existing_manifest: dict, wanted: dict) -> dict:
    have = existing_manifest.get("fingerprint") or {}
    out = {}
    for field in FINGERPRINT_FIELDS:
        if field not in wanted or wanted[field] is None:
            continue
        if field not in have:
            out[field] = ("<absent>", wanted[field])
        elif have[field] != wanted[field]:
            out[field] = (have[field], wanted[field])
    return out


def prompt_table_sha256(prompts: dict) -> str:
    digest = hashlib.sha256()
    for qkey in sorted(prompts):
        digest.update(qkey.encode("utf-8"))
        digest.update(b":")
        digest.update(",".join(str(int(t)) for t in prompts[qkey]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def shard_stem(pool: str, positions: str, shard_index: int, shard_count: int) -> str:
    check_positions(positions)
    return f"blockfeat_{pool}_{positions}_s{shard_index:03d}of{shard_count:03d}"


def atomic_write_bytes(target: Path, data: bytes) -> None:
    handle = tempfile.NamedTemporaryFile(dir=str(target.parent), delete=False,
                                         prefix=target.name + ".", suffix=".tmp")
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def git_commit(repo: Path) -> tuple[str, bool]:
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                             capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain",
                                     "--untracked-files=all"], check=True,
                                    capture_output=True, text=True).stdout.strip())
        return sha, dirty
    except Exception:
        return "", True


def shard_manifest(*, pool, positions, shard_index, shard_count, layers, chains_path, chains_sha,
                   questions_path, questions_sha, model_path, revision, payloads, index_path,
                   index_sha, n_rows, n_chains, n_questions, hidden_size, commit, dirty,
                   stats, fingerprint=None, prompt=None) -> dict:
    return {
        "schema_version": SCHEMA,
        "chains_sha256": chains_sha,
        "model_id": "Qwen/Qwen3-1.7B",
        "model_revision": revision,
        "poolings": list(VARIANTS),
        "has_prompt_features": bool(prompt),
        "prompt_segments": (prompt or {}).get("n_segments"),
        "commit": commit,
        "schema": SCHEMA,
        "prompt": prompt,
        "fingerprint": fingerprint or input_fingerprint(
            chains_sha=chains_sha, questions_sha=questions_sha, layers=layers,
            positions=positions, dtype=stats.get("dtype_compute"), model_path=model_path,
            revision=revision, commit=commit),
        "pool": pool,
        "positions": positions,
        "shard": {"index": int(shard_index), "count": int(shard_count)},
        "layers": [int(x) for x in layers],
        "layer_semantics": ("hidden_states[L] = INPUT to model.model.layers[L] = OUTPUT of "
                            "layers[L-1]; layer 16 is the UHead's own input; never L+1"),
        "variants": list(VARIANTS),
        "variant_names": variant_names(layers),
        "dtype": STORE_DTYPE,
        "hidden_size": int(hidden_size),
        "payload_shape": ["n_rows", len(VARIANTS), int(hidden_size)],
        "alignment": ("features = hidden_states[L][0, :P+C-1]; block [a,b) -> columns "
                      "[P+a-1, P+b-1); end column P+b-2 = raw_tap.end_feature_index(P, b-1)"),
        "row_key": ["qkey", "rep", "end_index"],
        "chains": {"path": str(chains_path), "sha256": chains_sha},
        "questions": {"path": str(questions_path), "sha256": questions_sha},
        "model": {"path": str(model_path), "revision": revision},
        "output": {"payloads": payloads, "index": {"path": str(index_path), "sha256": index_sha}},
        "n_rows": int(n_rows),
        "n_chains": int(n_chains),
        "n_questions": int(n_questions),
        "git_commit": commit,
        "git_dirty": dirty,
        "stats": stats,
    }


def pool_manifest_path(out_root, pool: str, positions: str) -> Path:
    return Path(out_root) / pool / f"manifest_{check_positions(positions)}.json"


def merge_manifests(out_root, pool: str, positions: str, shard_count: int) -> dict:
    check_positions(positions)
    root = Path(out_root) / pool
    shards, seen_rows = [], 0
    base = None
    for i in range(shard_count):
        path = root / (shard_stem(pool, positions, i, shard_count) + ".manifest.json")
        if not path.exists():
            raise SystemExit(f"missing shard manifest {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != SCHEMA:
            raise SystemExit(f"{path}: schema {record.get('schema')!r} != {SCHEMA!r}")
        if record.get("pool") != pool:
            raise SystemExit(f"{path}: pool {record.get('pool')!r} != {pool!r}")
        if record.get("positions") != positions:
            raise SystemExit(f"{path}: positions {record.get('positions')!r} != {positions!r}")
        shard_id = record.get("shard") or {}
        if int(shard_id.get("index", -1)) != i:
            raise SystemExit(f"{path}: records shard index {shard_id.get('index')} but was "
                             f"found as shard {i}")
        if int(shard_id.get("count", -1)) != shard_count:
            raise SystemExit(f"{path}: records shard count {shard_id.get('count')} != "
                             f"{shard_count}")
        entries = list(record["output"]["payloads"].values()) + [record["output"]["index"]]
        if record.get("prompt"):
            entries += list(record["prompt"]["payloads"].values())
            entries.append(record["prompt"]["index"])
        for entry in entries:
            got = sha256_file(Path(entry["path"]))
            if got != entry["sha256"]:
                raise SystemExit(f"{entry['path']}: sha256 {got} != manifest {entry['sha256']}")
        fp = record.get("fingerprint") or {}
        key = (record["chains"]["sha256"], record["questions"]["sha256"],
               tuple(record["layers"]), record["dtype"], record["hidden_size"],
               record["git_commit"], record["model"]["revision"], record["positions"],
               fp.get("prompt_sha256"))
        if base is None:
            base = key
        elif key != base:
            raise SystemExit(f"shard {i} disagrees with shard 0 on inputs: {key} != {base}")
        seen_rows += record["n_rows"]
        shards.append(record)
    prompt_owner: dict = {}
    has_prompt = all(bool(rec.get("prompt")) for rec in shards)
    if has_prompt and any(rec.get("prompt") for rec in shards):
        for rec in shards:
            idx_path = Path(rec["prompt"]["index"]["path"])
            for line in idx_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                prompt_owner.setdefault(entry["qkey"], {
                    "shard": rec["shard"]["index"], "row": entry["row"],
                    "n_rows": entry["n_rows"]})
        prompt_index = root / f"prompt_index_{positions}.jsonl"
        atomic_write_bytes(prompt_index, b"".join(
            json.dumps({"qkey": q, **prompt_owner[q]}, separators=(",", ":")).encode("utf-8")
            + b"\n" for q in sorted(prompt_owner)))
    elif any(rec.get("prompt") for rec in shards):
        raise SystemExit("some shards carry prompt features and some do not; a merged manifest "
                         "declaring has_prompt_features would return None for some chains")

    merged = {
        "schema_version": SCHEMA,
        "chains_sha256": shards[0]["chains"]["sha256"],
        "model_id": shards[0].get("model_id", "Qwen/Qwen3-1.7B"),
        "model_revision": shards[0]["model"]["revision"],
        "poolings": list(VARIANTS),
        "hidden_size": shards[0]["hidden_size"],
        "commit": shards[0]["git_commit"],
        "has_prompt_features": bool(has_prompt and prompt_owner),
        "prompt_segments": (shards[0].get("prompt") or {}).get("n_segments"),
        "prompt": ({"n_segments": shards[0]["prompt"]["n_segments"],
                    "n_questions": len(prompt_owner),
                    "span": shards[0]["prompt"]["span"],
                    "index": str(root / f"prompt_index_{positions}.jsonl"),
                    "payloads": {s["shard"]["index"]: s["prompt"]["payloads"] for s in shards}}
                   if (has_prompt and prompt_owner) else None),
        "schema": SCHEMA,
        "pool": pool,
        "positions": positions,
        "shard_count": shard_count,
        "layers": shards[0]["layers"],
        "layer_semantics": shards[0]["layer_semantics"],
        "variants": shards[0]["variants"],
        "variant_names": shards[0]["variant_names"],
        "dtype": shards[0]["dtype"],
        "hidden_size": shards[0]["hidden_size"],
        "payload_shape": shards[0]["payload_shape"],
        "alignment": shards[0]["alignment"],
        "row_key": shards[0]["row_key"],
        "chains": shards[0]["chains"],
        "questions": shards[0]["questions"],
        "model": shards[0]["model"],
        "git_commit": shards[0]["git_commit"],
        "git_dirty": any(s["git_dirty"] for s in shards),
        "fingerprint": shards[0].get("fingerprint"),
        "n_rows": seen_rows,
        "n_chains": sum(s["n_chains"] for s in shards),
        "max_abs": max(s["stats"]["max_abs"] for s in shards),
        "shards": [{"index": s["shard"]["index"], "payloads": s["output"]["payloads"],
                    "index_file": s["output"]["index"], "n_rows": s["n_rows"],
                    "n_chains": s["n_chains"],
                    "prompt_payloads": (s.get("prompt") or {}).get("payloads")}
                   for s in shards],
    }
    target = pool_manifest_path(out_root, pool, positions)
    atomic_write_bytes(target, json.dumps(merged, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return merged


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", required=True, help="pool name (output sub-dir and file stem)")
    p.add_argument("--out", required=True, help="feature root, e.g. $P/iclr/features")
    p.add_argument("--positions", default="all", choices=list(POSITIONS),
                   help="'all' = every block end (required for a TRAINING pool); "
                        "'probed' = probed boundaries only (~4x smaller; enough to SCORE)")
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--merge", action="store_true",
                   help="merge shard manifests into <out>/<pool>/manifest_<positions>.json, exit")
    p.add_argument("--chains", help="pool chains.jsonl (with completion_ids)")
    p.add_argument("--questions", help="pool question manifest jsonl")
    p.add_argument("--chains-sha256", default=None, help="refuse a chains file with another sha")
    p.add_argument("--questions-sha256", default=None,
                   help="refuse a question manifest with another sha. Not cosmetic: qkey comes "
                        "from question text while answer_format is separate metadata, so a "
                        "manifest that lost answer_format keeps every qkey and rebuilds a BOXED "
                        "prompt for a code pool")
    p.add_argument("--layers", type=int, nargs="+", default=[16, 24],
                   help="hidden_states indices; 16 = the UHead's own input (NEVER 17)")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--model-path", default=None, help="local snapshot dir ending in --revision")
    p.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    p.add_argument("--dtype", default="bfloat16", help="backbone compute dtype")
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-batch-tokens", type=int, default=16384,
                   help="B * padded length budget per forward")
    p.add_argument("--max-chains-per-batch", type=int, default=8)
    p.add_argument("--prompt-segments", type=int, default=8,
                   help="prompt feature vectors per QUESTION (WS-D's ROM needs a sequence for "
                        "prompt-to-step attention and query-aware m_0); 0 disables them")
    p.add_argument("--limit-chains", type=int, default=0, help="smoke: cap the shard's chains")
    p.add_argument("--selfcheck", action="store_true",
                   help="check the hooked read == output_hidden_states, and the batched "
                        "(right-padded) read == per-row unpadded reads, on one ragged batch")
    p.add_argument("--selfcheck-tol", type=float, default=0.02,
                   help="max |diff| allowed, as a fraction of the feature scale")
    p.add_argument("--overwrite", action="store_true", help="rebuild a shard whose manifest exists")
    return p.parse_args(argv)


def _load_model(args, torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ref = args.model_path or args.model
    if args.model_path and not str(args.model_path).rstrip("/").endswith(args.revision):
        raise SystemExit(f"--model-path {args.model_path} does not end in the pinned revision "
                         f"{args.revision}; refusing to load an unpinned model by path")
    revision = None if args.model_path else args.revision
    tokenizer = AutoTokenizer.from_pretrained(ref, revision=revision, cache_dir=args.hf_cache,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ref, revision=revision, cache_dir=args.hf_cache,
        torch_dtype=getattr(torch, args.dtype), attn_implementation="sdpa")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return tokenizer, model


class _EarlyExit(Exception):
    pass


def hidden_at_layers(model, layers, ids, am):
    blocks = getattr(getattr(model, "model", None), "layers", None)
    if blocks is None:
        raise RuntimeError("model has no model.layers -- cannot early-exit")
    wanted = sorted({int(x) for x in layers})
    if wanted[0] < 1 or wanted[-1] >= len(blocks):
        raise SystemExit(f"layers {wanted} must satisfy 0 < L < {len(blocks)}; "
                         "L is a hidden_states index, not a decoder-layer index off by one")
    deepest = wanted[-1]
    captured: dict = {}
    handles = []

    def make(layer):
        def hook(_module, _inputs, output):
            captured[layer] = output[0] if isinstance(output, (tuple, list)) else output
            if layer == deepest:
                raise _EarlyExit
        return hook

    for layer in wanted:
        handles.append(blocks[layer - 1].register_forward_hook(make(layer)))
    try:
        try:
            model(input_ids=ids, attention_mask=am, use_cache=False)
        except _EarlyExit:
            pass
    finally:
        for handle in handles:
            handle.remove()
    missing = [layer for layer in wanted if layer not in captured]
    if missing:
        raise RuntimeError(f"early-exit hooks for layers {missing} never fired")
    return captured


def _batches(items, *, max_tokens: int, max_rows: int):
    order = sorted(range(len(items)), key=lambda i: -items[i][1])
    out, group, longest = [], [], 0
    for i in order:
        length = items[i][1]
        nxt = max(longest, length)
        if group and ((len(group) + 1) * nxt > max_tokens or len(group) >= max_rows):
            out.append(group)
            group, longest = [i], length
            continue
        group.append(i)
        longest = nxt
    if group:
        out.append(group)
    return out


def _forward_group(model, layers, seqs, torch, device):
    B = len(seqs)
    S = max(len(s) for s in seqs)
    ids = torch.zeros(B, S, dtype=torch.long)
    am = torch.zeros(B, S, dtype=torch.long)
    for r, seq in enumerate(seqs):
        ids[r, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        am[r, :len(seq)] = 1
    ids = ids.to(device)
    am = am.to(device)
    with torch.no_grad():
        return hidden_at_layers(model, layers, ids, am)


def _selfcheck(model, layers, seqs, got, torch, device, tol: float) -> dict:
    report = {}
    probe = seqs[0][: min(len(seqs[0]), 512)]
    ids = torch.tensor([probe], dtype=torch.long, device=device)
    am = torch.ones_like(ids)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=am, output_hidden_states=True, use_cache=False)
        hooked = hidden_at_layers(model, layers, ids, am)
    for layer in layers:
        ref = out.hidden_states[layer]
        diff = float((ref.float() - hooked[layer].float()).abs().max())
        scale = float(ref.float().abs().max())
        report[f"hook_vs_output_hidden_states_L{layer}"] = {"max_abs_diff": diff, "scale": scale}
        if diff > tol * max(scale, 1.0):
            raise SystemExit(
                f"LAYER INDEXING FAILURE: hooked hidden_states[{layer}] differs from the plain "
                f"output_hidden_states path by {diff:.4g} (scale {scale:.4g}). The hook is on the "
                f"wrong decoder layer -- this is the 16-vs-17 bug.")
    del out, hooked
    worst = {layer: (0.0, 0.0) for layer in layers}
    for r, seq in enumerate(seqs):
        single = _forward_group(model, layers, [seq], torch, device)
        for layer in layers:
            ref = single[layer][0, :len(seq)].float()
            cand = got[layer][r, :len(seq)].float()
            diff = float((ref - cand).abs().max())
            scale = float(ref.abs().max())
            if diff > worst[layer][0]:
                worst[layer] = (diff, scale)
        del single
    for layer in layers:
        diff, scale = worst[layer]
        report[f"padded_vs_single_L{layer}"] = {"max_abs_diff": diff, "scale": scale}
        if diff > tol * max(scale, 1.0):
            raise SystemExit(
                f"PADDING FAILURE: batched right-padded layer {layer} differs from an unpadded "
                f"per-row forward by {diff:.4g} (scale {scale:.4g})")
    return report


def _prefix_offsets(counts):
    out, total = [], 0
    for n in counts:
        out.append(total)
        total += n
    return out


def _run_group_with_oom_split(model, layers, seqs, torch, device):
    try:
        return _forward_group(model, layers, seqs, torch, device)
    except torch.cuda.OutOfMemoryError:
        if len(seqs) == 1:
            raise SystemExit(f"OOM on a single chain of {len(seqs[0])} tokens; lower "
                             "--max-batch-tokens or shard further") from None
        torch.cuda.empty_cache()
    half = len(seqs) // 2
    left = _run_group_with_oom_split(model, layers, seqs[:half], torch, device)
    right = _run_group_with_oom_split(model, layers, seqs[half:], torch, device)
    S = max(len(s) for s in seqs)
    out = {}
    for layer in layers:
        a, b = left[layer], right[layer]
        merged = torch.zeros(len(seqs), S, a.shape[2], dtype=a.dtype, device=a.device)
        merged[:half, :a.shape[1]] = a
        merged[half:, :b.shape[1]] = b
        out[layer] = merged
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    out_root = Path(args.out)
    check_positions(args.positions)
    if args.merge:
        merged = merge_manifests(out_root, args.pool, args.positions, args.shard_count)
        print(json.dumps({"merged": args.pool, "positions": args.positions,
                          "shards": merged["shard_count"], "n_rows": merged["n_rows"],
                          "n_chains": merged["n_chains"], "max_abs": merged["max_abs"],
                          "manifest": str(pool_manifest_path(out_root, args.pool,
                                                             args.positions))}, indent=2))
        return 0
    if not args.chains or not args.questions:
        raise SystemExit("--chains and --questions are required unless --merge")

    import numpy as np
    import torch

    from rl_controller.iclr import scoring_io as sio
    from rl_controller.iclr.provenance import source_record

    t0 = time.time()
    out_dir = out_root / args.pool
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = shard_stem(args.pool, args.positions, args.shard_index, args.shard_count)
    index_path = out_dir / f"{stem}.index.jsonl"
    manifest_path = out_dir / f"{stem}.manifest.json"

    chains_prov = source_record(args.chains)
    if args.chains_sha256 and chains_prov["sha256"] != args.chains_sha256:
        raise SystemExit(f"--chains sha256 {chains_prov['sha256']} != expected {args.chains_sha256}")
    questions_prov = source_record(args.questions)
    if args.questions_sha256 and questions_prov["sha256"] != args.questions_sha256:
        raise SystemExit(f"--questions sha256 {questions_prov['sha256']} != expected "
                         f"{args.questions_sha256}")
    commit, dirty = git_commit(Path(__file__).resolve().parents[2])
    layers = resolve_layers(args.layers)
    fingerprint = input_fingerprint(
        chains_sha=chains_prov["sha256"], questions_sha=questions_prov["sha256"],
        layers=layers, positions=args.positions, dtype=args.dtype,
        model_path=args.model_path or args.model, revision=args.revision, commit=commit)

    if manifest_path.exists() and not args.overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        drift = fingerprint_drift(existing, fingerprint)
        if drift:
            raise SystemExit(
                f"{manifest_path} exists but was built from different inputs: {drift}. "
                "This shard would be skipped and a STALE cache published. Re-run with "
                "--overwrite if the new inputs are intended, or publish to a fresh --out.")
        print(json.dumps({"skip": stem, "reason": "manifest exists and matches these inputs"}))
        return 0

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device: this pass is a full-pool prefill, refusing to run on CPU")
    device = torch.device(f"cuda:{args.gpu}")
    tokenizer, model = _load_model(args, torch)
    model = model.to(device)
    hidden_size = int(model.config.hidden_size)

    picked, n_rows = iter_shard_chains(args.chains, shard_index=args.shard_index,
                                       shard_count=args.shard_count,
                                       positions=args.positions, limit=args.limit_chains)
    if not picked:
        raise SystemExit(f"shard {args.shard_index}/{args.shard_count} of {args.pool} is empty")

    qrows = [json.loads(line) for line in open(args.questions, encoding="utf-8") if line.strip()]
    prompts = sio.build_prompt_table(qrows, tokenizer)
    prompt_sha = prompt_table_sha256(prompts)
    fingerprint["prompt_sha256"] = prompt_sha
    formats = {}
    for row in qrows:
        fmt = str(row.get("answer_format") or "boxed")
        formats[fmt] = formats.get(fmt, 0) + 1
    print(json.dumps({"prompt_table": {"n_questions": len(prompts), "sha256": prompt_sha,
                                       "answer_formats": formats}}), flush=True)
    items, index_rows = [], []
    for chain, offset in picked:
        if chain["qkey"] not in prompts:
            raise SystemExit(f"chain question {chain['qkey']} not in --questions")
        prompt_ids = prompts[chain["qkey"]]
        ends = [int(e) for e in chain["block_ends"]]
        completion = chain.get("completion_ids")
        if completion is None:
            raise SystemExit(f"chain {(chain['qkey'], chain['rep'])} has no completion_ids -- "
                             "this file is a scored dump, not a pool chains.jsonl")
        if ends[-1] > len(completion):
            raise SystemExit(f"chain {(chain['qkey'], chain['rep'])}: last block end {ends[-1]} "
                             f"exceeds completion length {len(completion)}")
        seq = list(prompt_ids) + list(completion[: ends[-1]])
        spans = kept_spans(chain, len(prompt_ids), args.positions)
        items.append((chain, offset, seq, len(prompt_ids), spans))
        index_rows.extend(chain_index_rows(chain, len(prompt_ids), row_offset=offset,
                                           positions=args.positions))
    if len(index_rows) != n_rows:
        raise SystemExit(f"index rows {len(index_rows)} != planned rows {n_rows}")

    if manifest_path.exists():
        os.replace(manifest_path, manifest_path.with_suffix(".json.superseded"))
    unique = f"{os.getpid()}.{int(time.time() * 1000) % 1_000_000}"
    tmp_paths = {layer_key(l): str(out_dir / f"{stem}.{layer_key(l)}.npy.{unique}.tmp")
                 for l in layers}
    n_seg = int(args.prompt_segments)
    prompt_qkeys = sorted({c["qkey"] for c, _ in picked}) if n_seg else []
    prompt_row_of = {q: i for i, q in enumerate(prompt_qkeys)}
    prompt_tmp = {layer_key(l): str(out_dir / f"{stem}.{layer_key(l)}.prompt.npy.{unique}.tmp")
                  for l in layers} if n_seg else {}
    prompt_payloads = {k: np.lib.format.open_memmap(
        pth, mode="w+", dtype=np.float16,
        shape=(len(prompt_qkeys), n_seg, len(VARIANTS), hidden_size))
        for k, pth in prompt_tmp.items()}
    prompt_done: set = set()
    payloads = {k: np.lib.format.open_memmap(p, mode="w+", dtype=np.float16,
                                             shape=(n_rows, len(VARIANTS), hidden_size))
                for k, p in tmp_paths.items()}
    lengths = [(i, len(items[i][2])) for i in range(len(items))]
    groups = _batches(lengths, max_tokens=args.max_batch_tokens,
                      max_rows=args.max_chains_per_batch)
    check_at = None
    if args.selfcheck:
        ragged = [gi for gi, g in enumerate(groups)
                  if len(g) > 1 and len({len(items[i][2]) for i in g}) > 1]
        check_at = ragged[0] if ragged else 0
    checked, done, max_abs = None, 0, 0.0
    for gi, group in enumerate(groups):
        seqs = [items[i][2] for i in group]
        got = _run_group_with_oom_split(model, layers, seqs, torch, device)
        if check_at is not None and gi == check_at:
            checked = _selfcheck(model, layers, seqs, got, torch, device, args.selfcheck_tol)
            checked["batch_rows"] = len(seqs)
            checked["batch_ragged"] = len({len(x) for x in seqs}) > 1
            print(json.dumps({"selfcheck": checked}, indent=2), flush=True)
        for r, i in enumerate(group):
            chain, offset, seq, prompt_len, spans = items[i]
            T = len(seq) - 1
            for layer in layers:
                feats = got[layer][r, :T].float()
                block = np.empty((len(spans), len(VARIANTS), hidden_size), dtype=np.float16)
                for si, (span_start, span_stop, end_col) in enumerate(spans):
                    block[si, 0] = (feats[span_start:span_stop].mean(dim=0)
                                    .to(torch.float16).cpu().numpy())
                    block[si, 1] = feats[end_col].to(torch.float16).cpu().numpy()
                if not np.isfinite(block).all():
                    raise SystemExit(
                        f"chain {(chain['qkey'], chain['rep'])} layer {layer}: "
                        f"{int((~np.isfinite(block)).sum())} non-finite fp16 values -- a hidden "
                        f"state channel overflowed float16 (max |x| seen so far {max_abs:.4g}). "
                        "Widen the store dtype rather than shipping inf into every probe.")
                max_abs = max(max_abs, float(np.abs(block).max()))
                payloads[layer_key(layer)][offset: offset + len(spans)] = block
                if n_seg and chain["qkey"] not in prompt_done:
                    pblock = np.empty((n_seg, len(VARIANTS), hidden_size), dtype=np.float16)
                    for si, (p_start, p_stop, p_last) in enumerate(
                            prompt_segments(prompt_len, n_seg)):
                        pblock[si, 0] = (feats[p_start:p_stop].mean(dim=0)
                                         .to(torch.float16).cpu().numpy())
                        pblock[si, 1] = feats[p_last].to(torch.float16).cpu().numpy()
                    if not np.isfinite(pblock).all():
                        raise SystemExit(f"question {chain['qkey']} layer {layer}: non-finite "
                                         "fp16 prompt vector")
                    prompt_payloads[layer_key(layer)][prompt_row_of[chain["qkey"]]] = pblock
            if n_seg:
                prompt_done.add(chain["qkey"])
            done += 1
        del got
        if done % 256 < len(group):
            print(json.dumps({"chains_done": done, "of": len(items),
                              "seconds": round(time.time() - t0, 1)}), flush=True)
    payload_records = {}
    for layer in layers:
        key = layer_key(layer)
        payloads[key].flush()
        del payloads[key]
        final = out_dir / f"{stem}.{key}.npy"
        os.replace(tmp_paths[key], final)
        payload_records[key] = {"path": str(final), "sha256": sha256_file(final)}
    prompt_record = None
    if n_seg:
        missing = [q for q in prompt_qkeys if q not in prompt_done]
        if missing:
            raise SystemExit(f"{len(missing)} questions have no prompt vector (first "
                             f"{missing[:3]}); the manifest would claim has_prompt_features "
                             "while a consumer got None")
        prompt_payload_records = {}
        for layer in layers:
            key = layer_key(layer)
            prompt_payloads[key].flush()
            del prompt_payloads[key]
            final = out_dir / f"{stem}.{key}.prompt.npy"
            os.replace(prompt_tmp[key], final)
            prompt_payload_records[key] = {"path": str(final), "sha256": sha256_file(final)}
        prompt_index_path = out_dir / f"{stem}.prompt_index.jsonl"
        atomic_write_bytes(prompt_index_path, b"".join(
            json.dumps({"qkey": q, "row": prompt_row_of[q], "n_rows": n_seg},
                       separators=(",", ":")).encode("utf-8") + b"\n" for q in prompt_qkeys))
        prompt_record = {"n_segments": n_seg, "n_questions": len(prompt_qkeys),
                         "span": "feature columns [0, P-1) -- the prompt-only columns",
                         "payloads": prompt_payload_records,
                         "index": {"path": str(prompt_index_path),
                                   "sha256": sha256_file(prompt_index_path)}}

    atomic_write_bytes(index_path, b"".join(
        json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n" for row in index_rows))
    record = shard_manifest(
        fingerprint=fingerprint, prompt=prompt_record,
        pool=args.pool, positions=args.positions, shard_index=args.shard_index,
        shard_count=args.shard_count, layers=layers, chains_path=args.chains,
        chains_sha=chains_prov["sha256"], questions_path=args.questions,
        questions_sha=questions_prov["sha256"], model_path=args.model_path or args.model,
        revision=args.revision, payloads=payload_records,
        index_path=index_path, index_sha=sha256_file(index_path),
        n_rows=n_rows, n_chains=len(items),
        n_questions=len({c["qkey"] for c, _ in picked}), hidden_size=hidden_size,
        commit=commit, dirty=dirty,
        stats={"max_abs": max_abs, "seconds": round(time.time() - t0, 1),
               "selfcheck": checked, "dtype_compute": args.dtype,
               "tokens_forwarded": sum(len(s) for _, _, s, _, _ in items)})
    atomic_write_bytes(manifest_path,
                       json.dumps(record, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps({"shard": stem, "n_rows": n_rows, "n_chains": len(items),
                      "max_abs": max_abs, "seconds": round(time.time() - t0, 1),
                      "payloads": {k: v["path"] for k, v in payload_records.items()}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
