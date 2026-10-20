"""Shared agent trajectory tree with on-demand expansion."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import time
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from alf_envsel import E

MAX_STEPS = 30
THINK_CAP = 8192
QUERY_STRIDE = 4
BLOCK_CAP = 128
FORCED_MAX = 32
NATURAL_MAX = 64
PROBE_CHUNK = 512


def node_key(split, gid, seed, history) -> str:
    return hashlib.sha256(json.dumps([split, gid, int(seed), list(history)]).encode()).hexdigest()[:32]


def admissible(env_state: dict, action: str):
    if hasattr(E, "is_admissible"):
        return E.is_admissible(env_state, action)
    adm = env_state.get("admissible")
    return None if adm is None else action in adm


def _env_job(args):
    path, history = args
    return E.state_after(path, history)


def member_needs(name: str) -> str:
    return "p0" if name == "nothink" else "full"


def decide(name: str, node: dict) -> dict:
    if name == "nothink":
        p = node["p0"]
        return {"action": p["action"], "cost": p["tokens"], "stop": 0, "thinking": 0}
    full = node["full"]
    pos = full["positions"]
    if name.startswith("cap"):
        c = int(name[3:])
        for i, p in enumerate(pos):
            if p["prefix_len"] >= c:
                return {"action": p["action"], "cost": p["prefix_len"] + p["tokens"], "stop": i,
                        "thinking": p["prefix_len"]}
    if name in ("natural",) or name.startswith("cap"):
        if full["outcome"] == "length":
            p = pos[-1]
            return {"action": p["action"], "cost": p["prefix_len"] + p["tokens"], "stop": "cap",
                    "thinking": p["prefix_len"]}
        nat = full["natural"]
        return {"action": nat["action"], "cost": full["thinking_tokens"] + nat["tokens"], "stop": "natural",
                "thinking": full["thinking_tokens"]}
    raise SystemExit(f"unknown member {name}")


class Store:

    def __init__(self, path: Path, read_also=(), readonly: bool = False, quarantine=()):
        self.path = path
        self.nodes: dict = {}
        self.skipped_lines = 0
        self.quarantine = set(quarantine)
        self.quarantined_parts = 0
        if not readonly:
            trim_torn_tail(path)
        for f in [*read_also, path]:
            if not Path(f).exists():
                continue
            with open(f) as fh:
                for line in fh:
                    if not line.endswith("\n"):
                        break
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        self.skipped_lines += 1
                        continue
                    node = self.nodes.setdefault(rec["key"], {})
                    held = rec["key"] in self.quarantine and rec.get("v") != 2
                    for part, value in rec.items():
                        if part in ("key", "v"):
                            continue
                        if held and part not in ("env", "p0"):
                            self.quarantined_parts += 1
                            continue
                        if part not in node:
                            node[part] = value
        self._fh = None if readonly else open(path, "a")

    def put(self, key: str, part: str, value: dict):
        node = self.nodes.setdefault(key, {})
        if part in node:
            return
        node[part] = value
        self._fh.write(json.dumps({"key": key, part: value, "v": 2}) + "\n")

    def flush(self):
        self._fh.flush()
        os.fsync(self._fh.fileno())


def trim_torn_tail(path) -> int:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with open(path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 1))
        if f.read(1) == b"\n":
            return 0
        step, pos = 1 << 20, size
        while pos > 0:
            start = max(0, pos - step)
            f.seek(start)
            chunk = f.read(pos - start)
            i = chunk.rfind(b"\n")
            if i >= 0:
                f.truncate(start + i + 1)
                return size - (start + i + 1)
            pos = start
        f.truncate(0)
        return size


def lane_store(lane_dir: Path, tag: str | None, readonly: bool = False) -> Store:
    lane_dir = Path(lane_dir)
    own = lane_dir / ("nodes.jsonl" if not tag else f"nodes.{tag}.jsonl")
    others = sorted(f for f in lane_dir.glob("nodes*.jsonl") if f != own)
    q = lane_dir / "quarantine.json"
    quarantine = json.loads(q.read_text())["keys"] if q.exists() else ()
    return Store(own, read_also=others, readonly=readonly, quarantine=quarantine)


def think_block(engine, encode, full, *, run_chain_batch):
    specs = [{"prompt_ids": n[2], "seed_key": (n[3], "think")} for n in full]
    out = []
    for ch in run_chain_batch(engine, specs, encode=encode, budget=THINK_CAP, query_stride=QUERY_STRIDE,
                              block_cap=BLOCK_CAP):
        r = ch["result"]
        comp = list(r.completion_ids)
        positions = [{"block": 0, "prefix_len": 0}]
        positions += [{"block": int(b), "prefix_len": int(p)} for b, p in ch["query_points"] if int(p) > 0]
        if r.outcome == "length" and positions[-1]["prefix_len"] != len(comp):
            positions.append({"block": "cap", "prefix_len": len(comp)})
        out.append((r.outcome, comp, list(r.block_ends), positions, None))
    return out


def think_single(engine, tokenizer, encode, full, *, GenRequest, eos_ids):
    import alf_rechunk as RC

    reqs = [GenRequest(prompt_ids=list(n[2]), stop=[], include_stop_str_in_output=False,
                       max_tokens=THINK_CAP + NATURAL_MAX, seed_key=(n[3], "think-single")) for n in full]
    out = []
    for n, o in zip(full, engine.generate_batch(reqs)):
        seq = list(o.token_ids)
        rc = RC.rechunk(tokenizer, n[2], seq, eos_ids, encode=encode)
        comp = rc["completion_ids"]
        out.append((rc["outcome"], comp, rc["block_ends"], rc["positions"], seq[len(comp):]))
    return out


def expand(engine, tokenizer, store: Store, need: list, *, run_chain_batch, GenRequest, mode: str = "single",
           eos_ids=()):
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    suffix = encode(E.FORCED_SUFFIX)
    full = [n for n in need if n[1] == "full"]
    chains = []
    stats = {"mode": mode, "think_nodes": len(full)}
    if full:
        t = time.time()
        chains = (think_single(engine, tokenizer, encode, full, GenRequest=GenRequest, eos_ids=eos_ids)
                  if mode == "single" else think_block(engine, encode, full, run_chain_batch=run_chain_batch))
        stats.update(think_s=round(time.time() - t, 1), thinking_tokens=sum(len(c[1]) for c in chains),
                     generated_tokens=sum(len(c[1]) + len(c[4] or []) for c in chains))
    reqs, meta = [], []
    stop_nl = {"stop": ["\n"], "include_stop_str_in_output": False}
    for n in need:
        key, part, prompt, base = n
        if part == "p0" or "p0" not in store.nodes.get(key, {}):
            reqs.append(GenRequest(prompt_ids=list(prompt) + suffix, max_tokens=FORCED_MAX,
                                   seed_key=(base, "forced", 0), **stop_nl))
            meta.append((key, "p0", None))
    full_info = {}
    for n, (outcome, comp, block_ends, positions, rest) in zip(full, chains):
        key, _, prompt, base = n
        text = tokenizer.decode(comp)
        info = {"outcome": outcome, "thinking_tokens": len(comp), "completion_ids": comp,
                "block_ends": block_ends, "positions": positions, "natural": None, "gen_mode": mode}
        full_info[key] = info
        for i, p in enumerate(positions):
            if i == 0:
                continue
            reqs.append(GenRequest(prompt_ids=list(prompt) + comp[:p["prefix_len"]] + suffix, max_tokens=FORCED_MAX,
                                   seed_key=(base, "forced", p["prefix_len"]), **stop_nl))
            meta.append((key, "pos", i))
        if outcome != "length":
            after = text.split(E.CLOSE, 1)[1] if E.CLOSE in text else None
            if after is not None and E.ACTION_RE.match(next((l for l in after.split("\n") if l.strip()), "")):
                action, parsed = E.parse_natural(after)
                info["natural"] = {"action": action, "parsed": parsed, "tokens": 0, "text": "", "in_thinking_chain": True}
            elif after is not None and rest:
                cont = rest[:NATURAL_MAX]
                ntext = tokenizer.decode(cont, skip_special_tokens=True)
                action, parsed = E.parse_natural(ntext)
                info["natural"] = {"action": action, "parsed": parsed, "tokens": len(cont), "text": ntext,
                                   "pad_tokens": 0, "in_thinking_chain": False, "same_request": True}
            else:
                cont = comp + (encode("</think>\n\n") if after is None else
                               ([] if text.endswith("\n\n") else encode("\n" if text.endswith("\n") else "\n\n")))
                reqs.append(GenRequest(prompt_ids=list(prompt) + cont, max_tokens=NATURAL_MAX,
                                       seed_key=(base, "natural"), stop=[], include_stop_str_in_output=False))
                meta.append((key, "natural", len(cont) - len(comp)))
    outs = []
    t = time.time()
    for i in range(0, len(reqs), PROBE_CHUNK):
        outs.extend(engine.generate_batch(reqs[i:i + PROBE_CHUNK]))
    stats.update(action_requests=len(reqs), action_s=round(time.time() - t, 1),
                 action_tokens=sum(len(o.token_ids) for o in outs))
    print(json.dumps(stats), flush=True)
    for (key, kind, extra), out in zip(meta, outs):
        if kind == "p0":
            store.put(key, "p0", {"action": E.parse_forced(out.text), "raw": out.text, "tokens": len(out.token_ids)})
        elif kind == "pos":
            p = full_info[key]["positions"][extra]
            p.update(action=E.parse_forced(out.text), raw=out.text, tokens=len(out.token_ids))
        else:
            action, parsed = E.parse_natural(out.text)
            full_info[key]["natural"] = {"action": action, "parsed": parsed, "tokens": len(out.token_ids),
                                         "text": out.text, "pad_tokens": extra, "in_thinking_chain": False}
    for key, info in full_info.items():
        p0 = store.nodes[key]["p0"]
        info["positions"][0].update(action=p0["action"], raw=p0["raw"], tokens=p0["tokens"])
        store.put(key, "full", info)
    store.flush()
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=sorted(E.SPLIT_DIRS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--members", default="natural,nothink",
                    help="a member-spec JSON file (alf_members) or a list of natural / nothink / cap<c>")
    ap.add_argument("--lane", type=int, default=0)
    ap.add_argument("--lanes", type=int, default=1, help="storage shards: game i lives in lane<i % lanes>/")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes sharing one shard: worker w takes the shard's games with (i // lanes) %% workers == w "
                         "and writes its own files (tag-w<w>)")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--first-game", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--env-workers", type=int, default=16)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=24576)
    ap.add_argument("--engine-seed", type=int, default=0)
    ap.add_argument("--gen-mode", choices=("segments", "single", "block"), default="segments",
                    help="segments (D15.4 + D16, member runs): thinking in fixed segments generated only as far as a "
                         "decision needs (alf_lazy); single (D16): one request per node; block: the pilot's")
    ap.add_argument("--tag", default=None,
                    help="this job's file tag in the lane dirs (nodes.<tag>.jsonl, episodes.<tag>.jsonl); none = the "
                         "pilot's nodes.jsonl / episodes.jsonl")
    ap.add_argument("--belt", default="", help="seed=run dir,... (the members' BELT probes)")
    ap.add_argument("--osp", default="", help="name=run dir,... (the members' OS-Pruner checkpoints)")
    ap.add_argument("--score-pack", type=int, default=1)
    ap.add_argument("--score-sleep", choices=("always", "never"), default="always",
                    help="always: vLLM sleeps (weights to CPU, KV cache released) while the scorers run, so a scorer "
                         "pass has the card (the BELT head attends over every token: ~31 GB at 10k tokens next to "
                         "vLLM ran out, 3656564); never: score next to the awake engine")
    ap.add_argument("--out", required=True, help="the tree dir (lane<k>/ inside)")
    a = ap.parse_args(argv)

    import alf_lazy as AL
    import alf_members as M

    from rl_controller.rollout_contract import contract_identity, sampling_config, select_prompt_contract
    from rl_controller.rollout_runner import GenRequest, VLLMEngine
    from rl_controller.single_cot import run_chain_batch
    from transformers import AutoTokenizer

    select_prompt_contract("olmo3-7b-think-native-v1")
    out = Path(a.out) / f"lane{a.lane}"
    out.mkdir(parents=True, exist_ok=True)
    games = E.game_list(a.split)
    idx = list(range(len(games)))[a.first_game:]
    if a.max_games:
        idx = idx[:a.max_games]
    mine = [i for i in idx if i % a.lanes == a.lane and (i // a.lanes) % a.workers == a.worker]
    if a.workers > 1:
        if not a.tag:
            raise SystemExit("--workers > 1 needs --tag (each worker writes its own files)")
        a.tag = f"{a.tag}-w{a.worker}"
    seeds = [int(s) for s in re.split(r"[,:]", a.seeds)]
    spec = M.load_spec(a.members)
    by_name = {m["name"]: m for m in spec}
    need_scorers = set().union(*[M.scorers_of(m) for m in spec])
    store = Store(out / "nodes.jsonl") if not a.tag else lane_store(out, a.tag)
    ep_name = "episodes.jsonl" if not a.tag else f"episodes.{a.tag}.jsonl"
    print(json.dumps({"lane": a.lane, "games": len(mine), "seeds": seeds, "members": list(by_name),
                      "scorers": sorted(need_scorers), "stored_nodes": len(store.nodes)}), flush=True)

    tok = AutoTokenizer.from_pretrained(a.model_path)
    from vllm import LLM
    llm = LLM(model=a.model_path, tensor_parallel_size=1, gpu_memory_utilization=a.gpu_mem_util,
              max_model_len=a.max_model_len, seed=a.engine_seed, enable_prefix_caching=True,
              enable_sleep_mode=a.score_sleep == "always")
    engine = VLLMEngine(llm, sampling=sampling_config(), seed=a.engine_seed)
    sleep_stats = {"cycles": 0, "sleep_s": 0.0, "wake_s": 0.0}

    @contextlib.contextmanager
    def scoring_room():
        if a.score_sleep != "always":
            yield
            return
        import torch

        t = time.time()
        llm.sleep(level=1)
        torch.cuda.empty_cache()
        sleep_stats["sleep_s"] += time.time() - t
        try:
            yield
        finally:
            torch.cuda.empty_cache()
            t = time.time()
            llm.wake_up()
            sleep_stats["wake_s"] += time.time() - t
            sleep_stats["cycles"] += 1
    scorers = reprs = None
    if need_scorers:
        import alf_scorer as SC
        belt = {int(k): v for k, v in (x.split("=", 1) for x in filter(None, a.belt.split(",")))}
        osp = dict(x.split("=", 1) for x in filter(None, a.osp.split(",")))
        missing_sc = [s for s in need_scorers if not ((s.startswith("belt-s") and int(s[6:]) in belt)
                                                       or (s.startswith("osp:") and s[4:] in osp))]
        if missing_sc:
            raise SystemExit(f"members need scorers {missing_sc} that --belt / --osp do not provide")
        with scoring_room():
            scorers = SC.Scorers(model_path=a.model_path, belt=belt or None, osp=osp or None)
        reprs = SC.ReprStore(out, a.tag or "pilot")
    pool = get_context("spawn").Pool(a.env_workers)
    watchdog = None
    timeout = int(os.environ.get("FIRST_DEPTH_TIMEOUT", "0"))
    if timeout > 0:
        import threading

        def _first_depth_timeout():
            print(json.dumps({"first_depth_timeout_s": timeout, "lane": a.lane}), flush=True)
            os._exit(75)
        watchdog = threading.Timer(timeout, _first_depth_timeout)
        watchdog.daemon = True
        watchdog.start()

    eps = [{"member": m, "game": i, "seed": s, "history": [], "trace": [], "won": False, "done": False, "score": None}
           for m in by_name for i in mine for s in seeds]
    t_lane = time.time()
    score_total = 0.0
    for depth in range(a.max_steps + 1):
        active = [e for e in eps if not e["done"]]
        if not active:
            break
        keys = {}
        for e in active:
            gid = E.game_id(games[e["game"]])
            e["key"] = node_key(a.split, gid, e["seed"], e["history"])
            keys.setdefault(e["key"], (games[e["game"]], list(e["history"])))
        missing = [k for k in keys if "env" not in store.nodes.get(k, {})]
        t = time.time()
        for k, st in zip(missing, pool.map(_env_job, [keys[k] for k in missing], chunksize=1)):
            store.put(k, "env", st)
        store.flush()
        env_s = time.time() - t
        want = {}
        for e in active:
            node = store.nodes[e["key"]]
            env = node["env"]
            if env["won"] or env.get("done"):
                e["won"], e["done"] = bool(env["won"]), True
                e["score"] = env.get("score", 100.0 if env["won"] else 0.0)
                continue
            if depth == a.max_steps:
                e["done"] = True
                e["score"] = env.get("score", 0.0)
                continue
            if len(E.prompt_ids(tok, E.build_user_turn(env, e["history"]))) + THINK_CAP + NATURAL_MAX + 16 \
                    > a.max_model_len:
                e["done"], e["context_exhausted"] = True, True
                e["score"] = env.get("score", 0.0)
                continue
            part = M.needs(by_name[e["member"]])
            if part == "full" or e["key"] not in want:
                want[e["key"]] = (part if want.get(e["key"], ("p0",))[0] != "full" else "full", e)
        if a.gen_mode == "segments":
            undec = [e for e in active if not e["done"]]
            ep_of, prompts = {}, {}
            for e in undec:
                ep_of.setdefault(e["key"], e)

            def prompt_of(k, _ep=ep_of, _pr=prompts):
                if k not in _pr:
                    ids = E.prompt_ids(tok, E.build_user_turn(store.nodes[k]["env"], _ep[k]["history"]))
                    if len(ids) + THINK_CAP + NATURAL_MAX + 16 > a.max_model_len:
                        raise SystemExit(f"prompt of {len(ids)} tokens does not fit max_model_len {a.max_model_len}")
                    _pr[k] = ids
                return _pr[k]

            def base_of(k, _ep=ep_of):
                e = _ep[k]
                return (a.split, E.game_id(games[e["game"]]), e["seed"], len(e["history"]), k)

            t = time.time()
            decisions, st = AL.run_depth(engine=engine, tok=tok, store=store, active=undec, by_name=by_name,
                                         prompt_of=prompt_of, base_of=base_of, scorers=scorers, reprs=reprs,
                                         decide=M.decide, scorers_of=M.scorers_of,
                                         eos_ids=sampling_config()["eos_token_id"], GenRequest=GenRequest,
                                         pack=a.score_pack, score_ctx=scoring_room)
            depth_s = time.time() - t
            score_total += st["score_s"]
            for i, e in enumerate(undec):
                d = decisions[i]
                node = store.nodes[e["key"]]
                e["trace"].append({"key": e["key"], "admissible": admissible(node["env"], d["action"]), **d})
                e["history"].append(d["action"])
            if watchdog is not None:
                watchdog.cancel()
                watchdog = None
            print(json.dumps({"depth": depth, "active": len(active), "decided": len(undec), "env_nodes": len(missing),
                              "env_s": round(env_s, 1), "depth_s": round(depth_s, 1), **st}), flush=True)
            continue
        need = []
        for key, (part, e) in want.items():
            have = store.nodes[key]
            if part in have or (part == "p0" and "full" in have):
                continue
            gid = E.game_id(games[e["game"]])
            ids = E.prompt_ids(tok, E.build_user_turn(have["env"], e["history"]))
            if len(ids) + THINK_CAP + NATURAL_MAX + 16 > a.max_model_len:
                raise SystemExit(f"prompt of {len(ids)} tokens does not fit max_model_len {a.max_model_len}")
            need.append((key, part, ids, (a.split, gid, e["seed"], len(e["history"]), key)))
        t = time.time()
        stats = {}
        if need:
            stats = expand(engine, tok, store, need, run_chain_batch=run_chain_batch, GenRequest=GenRequest,
                           mode=a.gen_mode, eos_ids=sampling_config()["eos_token_id"])
        gen_s = time.time() - t
        t = time.time()
        todo, hist_of = {}, {}
        for e in active:
            if e["done"]:
                continue
            for sc in M.scorers_of(by_name[e["member"]]):
                if f"sc:{sc}" not in store.nodes[e["key"]]:
                    todo.setdefault(sc, set()).add(e["key"])
                    hist_of[e["key"]] = list(e["history"])
        n_scored = 0
        if todo:
            with scoring_room():
                n_scored = score_nodes(scorers, reprs, store, todo, hist_of, tok, pack=a.score_pack)
        score_s = time.time() - t
        score_total += score_s
        for e in active:
            if e["done"]:
                continue
            node = store.nodes[e["key"]]
            d = M.decide(by_name[e["member"]], node, e["key"], reprs=reprs)
            e["trace"].append({"key": e["key"], "admissible": admissible(node["env"], d["action"]), **d})
            e["history"].append(d["action"])
        if watchdog is not None:
            watchdog.cancel()
            watchdog = None
        print(json.dumps({"depth": depth, "active": len(active), "env_nodes": len(missing), "expanded": len(need),
                          "scored": n_scored, "env_s": round(env_s, 1), "gen_s": round(gen_s, 1),
                          "score_s": round(score_s, 1), **{k: v for k, v in stats.items() if k != "mode"}}),
              flush=True)
    pool.close()
    with open(out / (ep_name + ".tmp"), "w") as f:
        for e in eps:
            f.write(json.dumps({"member": e["member"], "split": a.split, "game": e["game"],
                                "game_id": E.game_id(games[e["game"]]), "seed": e["seed"], "won": e["won"],
                                "score": e["score"] if e["score"] is not None else (100.0 if e["won"] else 0.0),
                                "context_exhausted": bool(e.get("context_exhausted")),
                                "steps": len(e["trace"]), "cost": sum(t["cost"] for t in e["trace"]),
                                "thinking": sum(t["thinking"] for t in e["trace"]),
                                "history": e["history"], "trace": e["trace"]}) + "\n")
    os.replace(out / (ep_name + ".tmp"), out / ep_name)
    import alf_envsel
    meta = {"env": alf_envsel.NAME, "lane": a.lane, "lanes": a.lanes, "worker": a.worker, "workers": a.workers, "split": a.split,
            "games": mine, "seeds": seeds,
            "members": spec, "tag": a.tag, "belt": a.belt, "osp": a.osp, "score_pack": a.score_pack,
            "think_cap": THINK_CAP, "query_stride": QUERY_STRIDE, "block_cap": BLOCK_CAP, "forced_max": FORCED_MAX,
            "natural_max": NATURAL_MAX, "max_steps": a.max_steps, "gen_mode": a.gen_mode, "engine_seed": a.engine_seed,
            "gpu_mem_util": a.gpu_mem_util, "prompt_template": E.PROMPT_TEMPLATE, "forced_suffix": E.FORCED_SUFFIX,
            "contract": contract_identity(), "engine_seconds": engine.seconds, "score_seconds": round(score_total, 1),
            "score_sleep": a.score_sleep, "sleep_stats": sleep_stats,
            "lane_seconds": round(time.time() - t_lane, 1), "nodes": len(store.nodes),
            "scorer_identity": scorers.identity if scorers else None, "modules": loaded_modules()}
    (out / ("lane_meta.json" if not a.tag else f"lane_meta.{a.tag}.json")).write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"lane_done": a.lane, "seconds": meta["lane_seconds"]}), flush=True)
    return 0


def score_nodes(scorers, reprs, store, todo: dict, hist_of: dict, tok, *, pack: int) -> int:
    import alf_scorer as SC

    keys = sorted(set().union(*todo.values()))
    chains = {}
    for k in keys:
        node = store.nodes[k]
        chains[k] = SC.node_chain(k, E.prompt_ids(tok, E.build_user_turn(node["env"], hist_of[k])), node["full"])
    belt_seeds = sorted({int(s[6:]) for s in todo if s.startswith("belt-s")})
    osp_names = sorted({s[4:] for s in todo if s.startswith("osp:")})
    res = {}
    if belt_seeds:
        ks = sorted(set().union(*[todo[f"belt-s{s}"] for s in belt_seeds]))
        for k, by_seed in scorers.score_belt([chains[k] for k in ks], seeds=belt_seeds, pack=pack).items():
            for seed, v in by_seed.items():
                ref = reprs.add(k, f"belt-s{seed}", v["reprs"])
                res.setdefault(k, {})[f"sc:belt-s{seed}"] = {"scores": v["scores"], "reprs": ref}
    for name in osp_names:
        ks = sorted(todo[f"osp:{name}"])
        for k, by_name in scorers.score_osp([chains[k] for k in ks], names=[name], pack=pack).items():
            res.setdefault(k, {})[f"sc:osp:{name}"] = {"scores": by_name[name]["scores"]}
    reprs.flush()
    n = 0
    for sc, ks in todo.items():
        for k in ks:
            store.put(k, f"sc:{sc}", res.get(k, {}).get(f"sc:{sc}", {"scores": [], "no_probes": True}))
            n += 1
    store.flush()
    return n


def loaded_modules() -> dict:
    out = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and (name.startswith("rl_controller") or name.startswith("alf_")):
            out[name] = {"file": f, "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest()}
    return out


if __name__ == "__main__":
    sys.exit(main())
