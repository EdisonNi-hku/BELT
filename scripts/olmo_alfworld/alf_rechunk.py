"""Generate thinking in one request per node and cut block boundaries afterwards."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

THINK_CAP = 8192
QUERY_STRIDE = 4
BLOCK_CAP = 128


@dataclass
class _Out:
    token_ids: list
    text: str
    finish_reason: str
    stop_reason: object
    hit_eos: bool


def token_pieces(tokenizer, prompt_ids, ids) -> list[str]:
    from tokenizers.decoders import DecodeStream

    tk = tokenizer._tokenizer
    ds = DecodeStream(skip_special_tokens=True)
    suffix = list(prompt_ids)
    if len(suffix) > 4:
        for i in range(4, min(len(prompt_ids) + 1, 24)):
            cand = prompt_ids[-i:]
            if "�" not in tokenizer.decode(cand):
                suffix = cand
                break
    for t in suffix:
        try:
            ds.step(tk, t)
        except Exception:
            ds = DecodeStream(skip_special_tokens=True)
    out = []
    for t in ids:
        try:
            piece = ds.step(tk, t)
        except Exception:
            ds = DecodeStream(skip_special_tokens=True)
            piece = None
        out.append(piece or "")
    return out


class ReplayEngine:

    def __init__(self, prompt_len: int, ids: list, pieces: list, eos_ids):
        self.prompt_len, self.ids, self.pieces, self.eos = prompt_len, ids, pieces, set(eos_ids)
        self.requests = 0

    def generate_batch(self, reqs):
        return [self._one(r) for r in reqs]

    def _one(self, r):
        self.requests += 1
        o = len(r.prompt_ids) - self.prompt_len
        if o < 0 or o > len(self.ids) or (o and r.prompt_ids[-1] != self.ids[o - 1]):
            raise RuntimeError(f"request offset {o} is not a prefix of the sequence")
        text, out = "", []
        for j in range(o, min(o + int(r.max_tokens), len(self.ids))):
            tid = self.ids[j]
            out.append(tid)
            if tid in self.eos:
                return _Out(out, text, "stop", None, True)
            piece = self.pieces[j]
            if piece and r.stop:
                text += piece
                for s in r.stop:
                    idx = text.find(s, max(0, len(text) - len(piece) - len(s) + 1))
                    if idx != -1:
                        end = idx + len(s) if r.include_stop_str_in_output else idx
                        return _Out(out, text[:end], "stop", s, False)
            else:
                text += piece
        if len(out) == int(r.max_tokens):
            return _Out(out, text, "length", None, False)
        raise RuntimeError(f"sequence of {len(self.ids)} tokens ended at {o + len(out)} without EOS before the draw did")


def rechunk(tokenizer, prompt_ids, ids, eos_ids, *, encode=None, pieces=None):
    from rl_controller.single_cot import run_chain_batch

    encode = encode or (lambda s: tokenizer.encode(s, add_special_tokens=False))
    pieces = pieces if pieces is not None else token_pieces(tokenizer, prompt_ids, ids)
    eng = ReplayEngine(len(prompt_ids), list(ids), pieces, eos_ids)
    ch = run_chain_batch(eng, [{"prompt_ids": list(prompt_ids), "seed_key": ("replay",)}], encode=encode,
                         budget=THINK_CAP, query_stride=QUERY_STRIDE, block_cap=BLOCK_CAP)[0]
    r = ch["result"]
    comp = list(r.completion_ids)
    positions = [{"block": 0, "prefix_len": 0}]
    positions += [{"block": int(b), "prefix_len": int(p)} for b, p in ch["query_points"] if int(p) > 0]
    if r.outcome == "length" and positions[-1]["prefix_len"] != len(comp):
        positions.append({"block": "cap", "prefix_len": len(comp)})
    return {"outcome": r.outcome, "completion_ids": comp, "block_ends": list(r.block_ends),
            "positions": positions, "requests": eng.requests}


_G = {}


def _init(model_path):
    from transformers import AutoTokenizer

    from rl_controller.rollout_contract import sampling_config, select_prompt_contract

    select_prompt_contract("olmo3-7b-think-native-v1")
    _G["tok"] = AutoTokenizer.from_pretrained(model_path)
    _G["eos"] = sampling_config()["eos_token_id"]


def _check(item):
    key, prompt_ids, full = item
    comp = full["completion_ids"]
    try:
        got = rechunk(_G["tok"], prompt_ids, comp, _G["eos"])
    except Exception as exc:
        return {"key": key, "ok": False, "why": f"replay error: {exc}"}
    want_pos = [(p["block"], p["prefix_len"]) for p in full["positions"]]
    got_pos = [(p["block"], p["prefix_len"]) for p in got["positions"]]
    diffs = []
    if got["completion_ids"] != comp:
        diffs.append(f"completion {len(got['completion_ids'])} vs {len(comp)} tokens")
    if got["outcome"] != full["outcome"]:
        diffs.append(f"outcome {got['outcome']} vs {full['outcome']}")
    if got["block_ends"] != full["block_ends"]:
        k = next((i for i, (a, b) in enumerate(zip(got["block_ends"], full["block_ends"])) if a != b),
                 min(len(got["block_ends"]), len(full["block_ends"])))
        diffs.append(f"block_ends differ at {k} ({len(got['block_ends'])} vs {len(full['block_ends'])})")
    if got_pos != want_pos:
        diffs.append(f"positions {len(got_pos)} vs {len(want_pos)}")
    return {"key": key, "ok": not diffs, "why": "; ".join(diffs), "blocks": len(full["block_ends"]),
            "positions": len(want_pos), "tokens": len(comp), "requests": got["requests"]}


def load_tree(run: str) -> dict:
    nodes = {}
    for lane in sorted(glob.glob(f"{run}/lane*/")):
        for line in open(Path(lane) / "nodes.jsonl"):
            if not line.endswith("\n"):
                break
            rec = json.loads(line)
            n = nodes.setdefault(rec["key"], {})
            for k in ("env", "p0", "full"):
                if k in rec and k not in n:
                    n[k] = rec[k]
    return nodes


def tree_index(nodes: dict, split: str, seeds=(0,)) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from alf_envsel import E
    from alf_tree import node_key

    out = {}
    for g, path in enumerate(E.game_list(split)):
        gid = E.game_id(path)
        for seed in seeds:
            todo = [[]]
            while todo:
                h = todo.pop()
                key = node_key(split, gid, seed, h)
                if key not in nodes or key in out:
                    continue
                out[key] = (g, seed, h)
                n = nodes[key]
                acts = set()
                if "p0" in n:
                    acts.add(n["p0"]["action"])
                if "full" in n:
                    acts.update(p["action"] for p in n["full"]["positions"] if "action" in p)
                    if n["full"].get("natural"):
                        acts.add(n["full"]["natural"]["action"])
                todo.extend(h + [a] for a in sorted(acts))
    return out


def tree_histories(nodes: dict, split: str, seeds=(0,)) -> dict:
    return {k: v[2] for k, v in tree_index(nodes, split, seeds).items()}


def load_nodes_with_prompts(run: str, tok, split: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from alf_envsel import E

    nodes = load_tree(run)
    hist = tree_histories(nodes, split)
    items, no_hist = [], 0
    for key, n in nodes.items():
        if "full" not in n:
            continue
        if key not in hist:
            no_hist += 1
            continue
        ids = E.prompt_ids(tok, E.build_user_turn(n["env"], hist[key]))
        items.append((key, ids, n["full"]))
    return items, no_hist


def gate(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dirs as split=dir, e.g. eval_out_of_distribution=$R/runs/oalf/pilot")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    _init(a.model_path)
    report = {"runs": {}, "ok": True}
    for spec in a.runs:
        split, run = spec.split("=", 1)
        items, no_hist = load_nodes_with_prompts(run, _G["tok"], split)
        with get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.model_path,)) as pool:
            res = pool.map(_check, items, chunksize=8)
        bad = [r for r in res if not r["ok"]]
        report["runs"][run] = {"full_nodes_checked": len(res), "full_nodes_without_history": no_hist,
                               "mismatches": len(bad), "first_mismatches": bad[:20],
                               "block_ends_checked": sum(r.get("blocks", 0) for r in res),
                               "positions_checked": sum(r.get("positions", 0) for r in res),
                               "tokens_checked": sum(r.get("tokens", 0) for r in res)}
        report["ok"] = report["ok"] and not bad and no_hist == 0
        print(json.dumps({run: {k: v for k, v in report["runs"][run].items() if k != "first_mismatches"}}),
              flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"gate_ok": report["ok"]}), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gate":
        sys.exit(gate(sys.argv[2:]))
    sys.exit(__doc__)
