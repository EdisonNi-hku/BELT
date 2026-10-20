from __future__ import annotations

import hashlib
import json
import os


def load_spec(path_or_names: str) -> list[dict]:
    import re
    from pathlib import Path

    if path_or_names.endswith(".json") and Path(path_or_names).exists():
        spec = json.loads(Path(path_or_names).read_text())
    else:
        spec = []
        for name in re.split(r"[,:]", path_or_names):
            if name in ("natural", "nothink"):
                spec.append({"name": name, "kind": name})
            elif name.startswith("cap"):
                spec.append({"name": name, "kind": "cap", "cap": int(name[3:])})
            else:
                raise SystemExit(f"member {name!r} needs a spec file")
    names = [m["name"] for m in spec]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate member names in the spec")
    return spec


def needs(member: dict) -> str:
    return "p0" if member["kind"] == "nothink" else "full"


def scorers_of(member: dict) -> set:
    f = member.get("feature")
    return {f["scorer"]} if f else set()


CLIP = 1e-4
BELIEFS = ("q", "v", "r", "stability")
_SOURCES, _FEAT = {}, {}


def _logit(p: float) -> float:
    import math

    p = min(max(float(p), CLIP), 1.0 - CLIP)
    return math.log(p / (1.0 - p))


def source_params(src: dict) -> tuple:
    import numpy as np

    path = src["file"]
    remap = os.environ.get("ALF_PATH_REMAP")
    if remap:
        old, new = remap.split("=", 1)
        if path.startswith(old):
            path = new + path[len(old):]
    if path not in _SOURCES:
        _SOURCES[path] = json.loads(open(path).read())
    data = _SOURCES[path]
    m = data["models"][src["model"][0]][int(src["model"][1])] if src.get("model") else data
    st = data["standardization"]
    return (np.asarray(m["w"], dtype=np.float64), float(m["b"]), np.asarray(st["mean"], dtype=np.float64),
            np.asarray(st["std"], dtype=np.float64))


def belief_z(scores: list, src: dict):
    import numpy as np

    w, b, mean, std = source_params(src)
    x = np.array([[_logit(s[c]) for c in BELIEFS] for s in scores], dtype=np.float64).reshape(-1, len(BELIEFS))
    xs = (x - mean[:len(BELIEFS)]) / std[:len(BELIEFS)]
    z = np.full(len(xs), b, dtype=np.float64)
    for c, wt in enumerate(w):
        z += float(wt) * xs[:, c]
    return z


def repr_z(rows, src: dict):
    import numpy as np

    w, b, mean, std = source_params(src)
    xs = (np.asarray(rows, dtype=np.float16).astype(np.float64) - mean) / std
    z = np.full(len(xs), b, dtype=np.float64)
    for j, wt in enumerate(w):
        z += float(wt) * xs[:, j]
    return z


def feature(f: dict, node: dict, key: str = None, reprs=None) -> list[float] | None:
    part = node.get(f"sc:{f['scorer']}")
    if part is None:
        return None
    if part.get("no_probes"):
        return []
    if f["type"] == "field":
        return [float(s[f["field"]]) for s in part["scores"]]
    ck = (key, json.dumps(f, sort_keys=True), len(part["scores"]))
    if ck in _FEAT:
        return _FEAT[ck]
    if f["type"] == "belief_linear":
        out = [float(v) for v in belief_z(part["scores"], f["source"])]
    elif f["type"] == "repr_linear":
        out = [float(v) for v in repr_z(reprs.get(key, f["scorer"]), f["source"])]
    else:
        raise SystemExit(f"unknown feature type {f['type']}")
    if len(_FEAT) > 200000:
        _FEAT.clear()
    _FEAT[ck] = out
    return out


def uniform(salt: str, key: str, j: int) -> float:
    h = hashlib.blake2b(repr((salt, key, int(j))).encode(), digest_size=8).digest()
    return (int.from_bytes(h, "big") >> 11) / float(1 << 53)


def stop_probe(member: dict, node: dict, key: str, reprs=None) -> int | None:
    vals = feature(member["feature"], node, key, reprs)
    if vals is None:
        raise RuntimeError(f"member {member['name']}: node {key} has no {member['feature']['scorer']} scores")
    if member["kind"] == "threshold":
        tau = float(member["tau"])
        return next((j for j, v in enumerate(vals) if v >= tau), None)
    if member["kind"] == "sample":
        return next((j for j, v in enumerate(vals) if uniform(member["salt"], key, j) < v), None)
    raise SystemExit(f"unknown member kind {member['kind']}")


def decide(member: dict, node: dict, key: str, reprs=None) -> dict | None:
    kind = member["kind"]
    if kind == "nothink":
        p = node["p0"]
        return {"action": p["action"], "cost": p["tokens"], "stop": 0, "thinking": 0}
    full = node["full"]
    pos = full["positions"]
    partial = full["outcome"] == "partial"

    def at(i, stop):
        p = pos[i]
        return {"action": p["action"], "cost": p["prefix_len"] + p["tokens"], "stop": stop, "thinking": p["prefix_len"]}

    if kind == "cap":
        for i, p in enumerate(pos):
            if p["prefix_len"] >= member["cap"]:
                return at(i, i)
    elif kind in ("threshold", "sample"):
        n_probes = sum(1 for p in pos[1:] if p["block"] != "cap")
        if n_probes:
            j = stop_probe(member, node, key, reprs)
            if j is not None:
                return at(j + 1, j + 1)
    elif kind != "natural":
        raise SystemExit(f"unknown member kind {kind}")
    if partial:
        return None
    if full["outcome"] == "length":
        return at(len(pos) - 1, "cap")
    nat = full["natural"]
    if nat is None or nat.get("request"):
        return None
    return {"action": nat["action"], "cost": full["thinking_tokens"] + nat["tokens"], "stop": "natural",
            "thinking": full["thinking_tokens"]}
