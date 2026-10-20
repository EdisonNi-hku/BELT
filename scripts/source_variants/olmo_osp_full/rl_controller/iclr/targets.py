from __future__ import annotations

from dataclasses import dataclass

from rl_controller.iclr.chains import is_correct
from rl_controller.s1_depth_margin import GENERATION_CAP

KINDS = ("bce", "huber", "softmax", "mse")
MANDATORY = ("q", "v")
V_SLOT = ("v", "v_probes", "v_sib")


@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: str
    positions: str
    n_out: int
    needs_pi0: bool = False
    needs_siblings: bool = False


CATALOGUE: dict[str, TargetSpec] = {s.name: s for s in (
    TargetSpec("q", "bce", "probes", 1), TargetSpec("v", "bce", "ends", 1),
    TargetSpec("vpi0", "bce", "ends", 1, needs_pi0=True), TargetSpec("r", "bce", "ends", 1),
    TargetSpec("joint", "softmax", "probes", 4), TargetSpec("joint_pi0", "softmax", "probes", 4, needs_pi0=True),
    TargetSpec("cterm", "huber", "probes", 1), TargetSpec("ccont", "huber", "ends", 1),
    TargetSpec("ccont_pi0", "huber", "ends", 1, needs_pi0=True),
    TargetSpec("dy", "softmax", "probes", 3, needs_pi0=True), TargetSpec("dc", "huber", "probes", 1, needs_pi0=True),
    TargetSpec("stability", "bce", "probes", 1), TargetSpec("runaway", "bce", "ends", 1),
    TargetSpec("sibling_agree", "bce", "probes", 1, needs_siblings=True),
    TargetSpec("o", "bce", "probes", 1),
    TargetSpec("v_probes", "bce", "probes", 1),
    TargetSpec("gap", "mse", "probes", 1),
    TargetSpec("v_sib", "bce", "ends", 1, needs_siblings=True),
    TargetSpec("gain", "bce", "probes", 1),
    TargetSpec("harm", "bce", "probes", 1),
)}
assert all(spec.kind in KINDS for spec in CATALOGUE.values()), "unknown loss kind in CATALOGUE"


def parse_targets(spec: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in spec.split(",") if part.strip())
    if names[:1] != MANDATORY[:1] or names[1:2] not in [(v,) for v in V_SLOT]:
        raise ValueError(f"targets must start with 'q' and one of {V_SLOT}, got {spec!r}")
    unknown = [n for n in names if n not in CATALOGUE]
    if unknown:
        raise ValueError(f"unknown targets {unknown}; known: {sorted(CATALOGUE)}")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate targets in {spec!r}")
    return names


def parse_weights(spec: str, names: tuple[str, ...]) -> dict[str, float]:
    weights = {n: 1.0 for n in names}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"weight spec {part!r} must be name=value")
        name, value = part.split("=", 1)
        name = name.strip()
        if name not in weights:
            raise ValueError(f"weight for {name!r} is not among targets {list(names)}")
        try:
            w = float(value)
        except ValueError as exc:
            raise ValueError(f"weight for {name!r} must be a float, got {value!r}") from exc
        if w < 0:
            raise ValueError(f"weight for {name!r} must be >= 0, got {w}")
        weights[name] = w
    return weights


def column_slices(names) -> dict[str, slice]:
    out: dict[str, slice] = {}
    start = 0
    for name in names:
        n = CATALOGUE[name].n_out
        out[name] = slice(start, start + n)
        start += n
    return out


def readout_width(names) -> int:
    return sum(CATALOGUE[name].n_out for name in names)


def _ends(chain) -> list[int]:
    return chain["_ends"] if "_ends" in chain else [int(e) for e in chain["block_ends"]]


def _fire(chain) -> list[bool]:
    fire = chain.get("_pi0_fire")
    if fire is None:
        raise ValueError(f"chain {(chain.get('qkey'), chain.get('rep'))} needs `_pi0_fire` "
                         "(attach the pi0 fire table at load time)")
    if len(fire) != len(chain["probes"]):
        raise ValueError(f"chain {(chain.get('qkey'), chain.get('rep'))} `_pi0_fire` has "
                         f"{len(fire)} entries for {len(chain['probes'])} probes")
    if any(not isinstance(f, bool) for f in fire):
        raise ValueError(f"chain {(chain.get('qkey'), chain.get('rep'))} `_pi0_fire` entries must be bool")
    return list(fire)


ROM_DROP_FLAG = "_rom_drop_never_correct"


def first_correct_probe(chain, y_now=None) -> int | None:
    if y_now is None:
        gold = chain["gold"]
        y_now = [is_correct(p["forced"], gold) for p in chain["probes"]]
    for index, y in enumerate(y_now):
        if y:
            return index
    return None


def _post_fcs(chain, y_now) -> list:
    k = first_correct_probe(chain, y_now)
    if k is None and chain.get(ROM_DROP_FLAG):
        return [None] * len(y_now)
    if k is None:
        return [0.0] * len(y_now)
    return [float(i > k) for i in range(len(y_now))]


def _endpoint_after(chain, min_prefix: int, fire=None) -> tuple[int, int]:
    if fire is None:
        fire = (_fire(chain) if "_pi0_fire" in chain else [False] * len(chain["probes"])) if fire is None else fire
    gold = chain["gold"]
    for probe, fired in zip(chain["probes"], fire):
        if probe["prefix_len"] > min_prefix and fired:
            return is_correct(probe["forced"], gold), int(probe["prefix_len"] + probe["forced_tokens"])
    return int(bool(chain["correct"])), int(chain["thinking_tokens"])


def labels(name: str, chain: dict) -> list:
    spec = CATALOGUE[name]
    probes = chain["probes"]
    gold = chain["gold"]
    y_final = int(bool(chain["correct"]))
    total = int(chain["thinking_tokens"])
    y_now = [is_correct(p["forced"], gold) for p in probes]
    fire = _fire(chain) if spec.needs_pi0 else None
    if spec.needs_siblings and "_sibling_majority" not in chain:
        raise ValueError(f"chain {(chain.get('qkey'), chain.get('rep'))} needs `_sibling_majority` "
                         "(attach labels.sibling_context at load time)")
    if name == "v_sib" and "_sibling_pass_rate" not in chain:
        raise ValueError(f"chain {(chain.get('qkey'), chain.get('rep'))} needs `_sibling_pass_rate` "
                         "(attach labels.sibling_context at load time)")

    if name == "q":
        return [float(y) for y in y_now]
    if name == "v_probes":
        return [float(y_final)] * len(probes)
    if name == "gap":
        return [float(y) - float(y_final) for y in y_now]
    if name == "gain":
        return [float(not y and y_final) for y in y_now]
    if name == "harm":
        return [float(bool(y) and not y_final) for y in y_now]
    if name == "v":
        return [float(y_final)] * len(_ends(chain))
    if name == "v_sib":
        rate = chain["_sibling_pass_rate"]
        return [None if rate is None else float(rate)] * len(_ends(chain))
    if name == "vpi0":
        return [float(_endpoint_after(chain, e, fire)[0]) for e in _ends(chain)]
    if name == "r":
        return [float(int(any(y for p, y in zip(probes, y_now) if p["prefix_len"] > e) or y_final))
                for e in _ends(chain)]
    if name == "joint":
        return [2 * y + y_final for y in y_now]
    if name == "joint_pi0":
        return [2 * y + _endpoint_after(chain, p["prefix_len"], fire)[0]
                for p, y in zip(probes, y_now)]
    if name == "cterm":
        return [p["forced_tokens"] / GENERATION_CAP for p in probes]
    if name == "ccont":
        return [(total - e) / GENERATION_CAP for e in _ends(chain)]
    if name == "ccont_pi0":
        return [(_endpoint_after(chain, e, fire)[1] - e) / GENERATION_CAP for e in _ends(chain)]
    if name == "dy":
        return [(y - _endpoint_after(chain, p["prefix_len"], fire)[0]) + 1
                for p, y in zip(probes, y_now)]
    if name == "dc":
        return [(_endpoint_after(chain, p["prefix_len"], fire)[1] - (p["prefix_len"] + p["forced_tokens"]))
                / GENERATION_CAP for p in probes]
    if name == "stability":
        answer = chain["answer"]
        if answer is None:
            return [None] * len(probes)
        return [float(p["forced"] is not None and p["forced"] == answer) for p in probes]
    if name == "runaway":
        return [float(chain["outcome"] == "length")] * len(_ends(chain))
    if name == "o":
        return _post_fcs(chain, y_now)
    if name == "sibling_agree":
        majority = chain["_sibling_majority"]
        if majority is None:
            return [None] * len(probes)
        return [float(p["forced"] is not None and p["forced"] == majority) for p in probes]
    raise ValueError(f"unknown target {name!r}")
