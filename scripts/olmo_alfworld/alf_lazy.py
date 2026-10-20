from __future__ import annotations

import json
import time

from alf_envsel import E

SEGMENTS = (1024, 2048, 4096, 8256)


def seq_of(node: dict) -> tuple[list, bool, int]:
    ids, n, ended = [], 0, False
    while f"seg{n}" in node:
        s = node[f"seg{n}"]
        ids += s["ids"]
        n += 1
        if s["finish"] == "eos":
            ended = True
            break
    if n == len(SEGMENTS):
        ended = True
    return ids, ended, n


def view(tok, prompt: list, node: dict, eos_ids, cache: dict, key: str) -> dict | None:
    import alf_rechunk as RC

    if "full" in node:
        return {"state": "final", **node["full"]}
    ids, ended, n = seq_of(node)
    if n == 0:
        return None
    ck = (key, n)
    if ck in cache:
        return cache[ck]
    if ended:
        rc = RC.rechunk(tok, prompt, ids, eos_ids)
        state = "final"
    else:
        rc = RC.rechunk(tok, prompt, ids + [eos_ids[0]], eos_ids)
        state = "partial" if len(rc["completion_ids"]) > len(ids) else "final"
        if state == "partial":
            rc["completion_ids"] = ids
            rc["block_ends"] = [e for e in rc["block_ends"] if e < len(ids)]
            rc["positions"] = [p for p in rc["positions"] if p["block"] != "cap" and p["prefix_len"] < len(ids)]
    comp = rc["completion_ids"]
    rest = ids[len(comp):] if state == "final" else []
    v = {"state": state, "outcome": "partial" if state == "partial" else rc["outcome"], "completion_ids": comp,
         "thinking_tokens": len(comp), "block_ends": rc["block_ends"], "positions": rc["positions"], "rest": rest,
         "ended": ended, "segments": n}
    if state == "final" and v["outcome"] != "length":
        nat = natural_from(tok, v, node, ended)
        if nat is None:
            v["state"] = "open"
        v["natural"] = nat
    cache[ck] = v
    return v


def natural_from(tok, v: dict, node: dict, ended: bool):
    text = tok.decode(v["completion_ids"])
    after = text.split(E.CLOSE, 1)[1] if E.CLOSE in text else None
    if after is not None and E.ACTION_RE.match(next((l for l in after.split("\n") if l.strip()), "")):
        action, parsed = E.parse_natural(after)
        return {"action": action, "parsed": parsed, "tokens": 0, "text": "", "in_thinking_chain": True}
    rest = v["rest"]
    if after is not None and rest:
        from_eos = any(t in E_EOS for t in rest[:E_NAT])
        if from_eos or len(rest) >= E_NAT or ended:
            cont = rest[:E_NAT]
            ntext = tok.decode(cont, skip_special_tokens=True)
            action, parsed = E.parse_natural(ntext)
            return {"action": action, "parsed": parsed, "tokens": len(cont), "text": ntext, "pad_tokens": 0,
                    "in_thinking_chain": False, "same_request": True}
        return None
    if ended or after is None:
        return node.get("natural") or {"request": True}
    return None


E_EOS: set = set()
E_NAT = 64


def as_node(node: dict, v: dict, merged: dict) -> dict:
    pos = []
    for i, p in enumerate(v["positions"]):
        fa = node.get("p0") if i == 0 else node.get(f"fa:{p['prefix_len']}")
        if fa is None:
            raise RuntimeError(f"position {p} has no forced action yet")
        pos.append({**p, "action": fa["action"], "raw": fa["raw"], "tokens": fa["tokens"]})
    full = {"outcome": v["outcome"], "thinking_tokens": v["thinking_tokens"], "completion_ids": v["completion_ids"],
            "block_ends": v["block_ends"], "positions": pos,
            "natural": v.get("natural") if v["state"] == "final" else None}
    return {"env": node["env"], "p0": node.get("p0"), "full": full, **merged}


def merged_parts(node: dict) -> dict:
    out = {}
    for part in [p for p in node if p.startswith("sc:") and "#" not in p]:
        scorer = part[3:]
        first = node[part]
        scores = list(first["scores"])
        refs = [(first.get("reprs"), len(scores))]
        k = 1
        while f"{part}#{k}" in node:
            nxt = node[f"{part}#{k}"]
            if len(nxt["scores"]) > len(scores):
                refs.append((nxt.get("reprs"), len(nxt["scores"])))
                scores += nxt["scores"][len(scores):]
            k += 1
        out[part] = {"scores": scores, "no_probes": bool(first.get("no_probes")) and len(scores) == 0,
                     "reprs_parts": refs}
    return out


def scored_count(node: dict, scorer: str) -> int:
    m = merged_parts({k: v for k, v in node.items() if k == f"sc:{scorer}" or k.startswith(f"sc:{scorer}#")})
    return len(m.get(f"sc:{scorer}", {}).get("scores", []))


def next_part_name(node: dict, scorer: str) -> str:
    if f"sc:{scorer}" not in node:
        return f"sc:{scorer}"
    k = 1
    while f"sc:{scorer}#{k}" in node:
        k += 1
    return f"sc:{scorer}#{k}"


class MergedReprs:

    def __init__(self, reprs, node_parts: dict):
        self.reprs, self.parts = reprs, node_parts

    def get(self, key, scorer):
        import numpy as np

        refs = self.parts.get(f"sc:{scorer}", {}).get("reprs_parts")
        if refs is None:
            return self.reprs.get(key, scorer)
        rows, have = [], 0
        for ref, upto in refs:
            if upto == 0:
                continue
            if ref is None:
                raise RuntimeError(f"node {key}: a {scorer} score part carries no vector ref")
            r = self.reprs.load(ref)[:upto]
            rows.append(r[have:])
            have = upto
        return np.concatenate(rows, 0)


def run_depth(*, engine, tok, store, active: list, by_name: dict, prompt_of, base_of, scorers, reprs, decide,
              scorers_of, eos_ids, GenRequest, pack: int = 1, log=print, score_ctx=None) -> dict:
    import alf_scorer as SC

    E_EOS.clear()
    E_EOS.update(eos_ids)
    encode = lambda s: tok.encode(s, add_special_tokens=False)
    suffix = encode(E.FORCED_SUFFIX)
    stop_nl = {"stop": ["\n"], "include_stop_str_in_output": False}
    cache = {}
    decisions = {}
    stats = {"rounds": 0, "segments": 0, "seg_tokens": 0, "action_requests": 0, "scored": 0, "gen_s": 0.0,
             "action_s": 0.0, "score_s": 0.0}
    undecided = list(range(len(active)))
    for rnd in range(len(SEGMENTS) + 3):
        stats["rounds"] = rnd + 1
        reqs, meta = [], []
        keys = sorted({active[i]["key"] for i in undecided})
        views = {}
        for k in keys:
            node = store.nodes[k]
            if "p0" not in node:
                reqs.append(GenRequest(prompt_ids=list(prompt_of(k)) + suffix, max_tokens=32,
                                       seed_key=(base_of(k), "forced", 0), **stop_nl))
                meta.append((k, "p0", None))
            v = view(tok, prompt_of(k), node, eos_ids, cache, k)
            views[k] = v
            if v is None or "full" in node:
                continue
            for p in v["positions"][1:]:
                if f"fa:{p['prefix_len']}" not in node:
                    comp = v["completion_ids"]
                    reqs.append(GenRequest(prompt_ids=list(prompt_of(k)) + comp[:p["prefix_len"]] + suffix,
                                           max_tokens=32, seed_key=(base_of(k), "forced", p["prefix_len"]), **stop_nl))
                    meta.append((k, "fa", p["prefix_len"]))
            nat = v.get("natural")
            if isinstance(nat, dict) and nat.get("request") and "natural" not in node:
                text = tok.decode(v["completion_ids"])
                after = text.split(E.CLOSE, 1)[1] if E.CLOSE in text else None
                comp = v["completion_ids"]
                cont = comp + (encode("</think>\n\n") if after is None else
                               ([] if text.endswith("\n\n") else encode("\n" if text.endswith("\n") else "\n\n")))
                reqs.append(GenRequest(prompt_ids=list(prompt_of(k)) + cont, max_tokens=E_NAT,
                                       seed_key=(base_of(k), "natural"), stop=[], include_stop_str_in_output=False))
                meta.append((k, "natural", len(cont) - len(comp)))
        if reqs:
            t = time.time()
            outs = []
            for i in range(0, len(reqs), 512):
                outs.extend(engine.generate_batch(reqs[i:i + 512]))
            stats["action_s"] += time.time() - t
            stats["action_requests"] += len(reqs)
            for (k, kind, extra), o in zip(meta, outs):
                if kind == "p0":
                    store.put(k, "p0", {"action": E.parse_forced(o.text), "raw": o.text, "tokens": len(o.token_ids)})
                elif kind == "fa":
                    store.put(k, f"fa:{extra}", {"action": E.parse_forced(o.text), "raw": o.text,
                                                 "tokens": len(o.token_ids)})
                else:
                    action, parsed = E.parse_natural(o.text)
                    store.put(k, "natural", {"action": action, "parsed": parsed, "tokens": len(o.token_ids),
                                             "text": o.text, "pad_tokens": extra, "in_thinking_chain": False})
            cache.clear()
            views = {k: view(tok, prompt_of(k), store.nodes[k], eos_ids, cache, k) for k in keys}
        for k in keys:
            v, node = views[k], store.nodes[k]
            if v is not None and v["state"] == "final" and "full" not in node:
                if v["outcome"] != "length" and isinstance(v.get("natural"), dict) and v["natural"].get("request"):
                    v["natural"] = node.get("natural")
                full_node = as_node(node, v, {})
                full = dict(full_node["full"], gen_mode="segments", segments=v["segments"])
                store.put(k, "full", full)
        todo = {}
        for i in undecided:
            k = active[i]["key"]
            v = views.get(k)
            if v is None:
                continue
            n_probes = sum(1 for p in v["positions"][1:] if p["block"] != "cap")
            for sc in scorers_of(by_name[active[i]["member"]]):
                node = store.nodes[k]
                have = scored_count(node, sc)
                final_empty = v["state"] != "partial" and n_probes == 0 and f"sc:{sc}" not in node
                if n_probes > have or final_empty:
                    todo.setdefault(sc, {})[k] = v
        if todo:
            t = time.time()
            import contextlib

            with (score_ctx() if score_ctx else contextlib.nullcontext()):
                stats["scored"] += score_views(scorers, reprs, store, todo, prompt_of, pack=pack)
            stats["score_s"] += time.time() - t
        store.flush()
        still = []
        for i in undecided:
            e = active[i]
            k = e["key"]
            node = store.nodes[k]
            m = by_name[e["member"]]
            if m["kind"] == "nothink":
                decisions[i] = decide(m, node, k, reprs=reprs)
                continue
            v = views.get(k)
            if "full" in node:
                merged = merged_parts(node)
                nd = {**node, **merged}
                decisions[i] = decide(m, nd, k, reprs=MergedReprs(reprs, merged))
                continue
            if v is None:
                still.append(i)
                continue
            merged = merged_parts(node)
            d = decide(m, as_node(node, v, merged), k, reprs=MergedReprs(reprs, merged))
            if d is None:
                still.append(i)
            else:
                decisions[i] = d
        undecided = still
        if not undecided:
            break
        reqs, meta = [], []
        for k in sorted({active[i]["key"] for i in undecided}):
            node = store.nodes[k]
            ids, ended, n = seq_of(node)
            if ended:
                raise RuntimeError(f"node {k}: sequence ended but a member is still undecided")
            lo = SEGMENTS[n - 1] if n else 0
            reqs.append(GenRequest(prompt_ids=list(prompt_of(k)) + ids, stop=[], include_stop_str_in_output=False,
                                   max_tokens=SEGMENTS[n] - lo, seed_key=(base_of(k), "think-seg", n)))
            meta.append((k, n))
        t = time.time()
        outs = engine.generate_batch(reqs)
        stats["gen_s"] += time.time() - t
        stats["segments"] += len(reqs)
        for (k, n), o in zip(meta, outs):
            ids = list(o.token_ids)
            stats["seg_tokens"] += len(ids)
            store.put(k, f"seg{n}", {"ids": ids, "finish": "eos" if o.hit_eos else
                                     ("length" if o.finish_reason == "length" else "stop")})
        store.flush()
        cache.clear()
    if undecided:
        raise RuntimeError(f"{len(undecided)} episodes undecided after all segments")
    return decisions, {k: round(v, 1) if isinstance(v, float) else v for k, v in stats.items()}


def score_views(scorers, reprs, store, todo: dict, prompt_of, *, pack: int) -> int:
    import alf_scorer as SC

    n = 0
    for sc, views in todo.items():
        chains, names = [], {}
        for k, v in views.items():
            full_like = {"positions": v["positions"], "completion_ids": v["completion_ids"],
                         "block_ends": v["block_ends"], "outcome": v["outcome"], "thinking_tokens": v["thinking_tokens"]}
            chains.append(SC.node_chain(k, prompt_of(k), full_like))
            names[k] = next_part_name(store.nodes[k], sc)
        if sc.startswith("belt-s"):
            seed = int(sc[6:])
            res = {k: r[seed] for k, r in scorers.score_belt(chains, seeds=[seed], pack=pack).items()}
        else:
            name = sc[4:]
            res = {k: r[name] for k, r in scorers.score_osp(chains, names=[name], pack=pack).items()}
        for k in views:
            part = names[k]
            if k in res:
                rec = {"scores": res[k]["scores"]}
                if "reprs" in res[k]:
                    rec["reprs"] = reprs.add(k, part[3:], res[k]["reprs"])
            else:
                rec = {"scores": [], "no_probes": True}
            store.put(k, part, rec)
            n += 1
    reprs.flush()
    return n
