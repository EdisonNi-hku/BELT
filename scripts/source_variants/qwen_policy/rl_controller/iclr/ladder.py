from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Callable

from rl_controller import s1_depth_margin as dm
from rl_controller.iclr import learning_curve as lc
from rl_controller.iclr.chains import is_correct
from rl_controller.iclr.labels import depth_of
from rl_controller.iclr.pi0_bias import consecutive_agreement
from rl_controller.single_cot import stop_at_cap

Plan = Callable[[dict], "dict | None"]
Predicate = Callable[[dict, dict], bool]

THETAS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
MARGINS = lc.MARGIN_GRID
LINEAR_A = dm.LINEAR_A
LINEAR_B = dm.LINEAR_B
LAMBDAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
HYSTERESIS_K = (2, 3)
CUSUM_H = (0.05, 0.1, 0.2, 0.4)
BRACKET_MHI = (0.0, 0.05, 0.1, 0.2)
B_PLUS_M = (-0.1, 0.0, 0.1)
FREE_CAPS = (16, 32, 64, 96, 128)

NATURAL = "natural"
ALWAYS_FIRE = "always_fire"


def first_probe_where(pred: Predicate) -> Plan:
    def plan(chain: dict) -> dict | None:
        for probe, score in zip(chain["probes"], chain["scores"]):
            if pred(probe, score):
                return probe
        return None
    return plan


def _depth(probe: dict) -> float:
    return depth_of(probe["block"])


def stop_threshold(theta: float, q_key: str = "q") -> Plan:
    return first_probe_where(lambda p, s: s[q_key] >= theta)


def stop_margin(m: float, q_key: str = "q", v_key: str = "v") -> Plan:
    return first_probe_where(lambda p, s: s[q_key] >= s[v_key] - m)


def stop_depth_margin(a: float, b: float, q_key: str, v_key: str) -> Plan:
    return first_probe_where(lambda p, s: s[q_key] >= s[v_key] - (a + b * _depth(p)))


def stop_step_margin(margins4: Sequence[float], q_key: str, v_key: str) -> Plan:
    margins = tuple(margins4)
    if len(margins) != 4:
        raise ValueError(f"step margins need 4 bins, got {margins!r}")
    return first_probe_where(lambda p, s: s[q_key] >= s[v_key] - margins[dm.depth_bin(_depth(p))])


def stop_joint(m: float, key: str = "joint") -> Plan:
    return first_probe_where(lambda p, s: s[key][2] - s[key][1] >= m)


def stop_bracket(m_hi: float, m: float, lower_key: str = "vpi0", upper_key: str = "r",
                 q_key: str = "q", v_key: str = "v") -> Plan:
    def pred(p: dict, s: dict) -> bool:
        q = s[q_key]
        if q < s[lower_key]:
            return False
        if q >= s[upper_key] - m_hi:
            return True
        return q >= s[v_key] - m
    return first_probe_where(pred)


def stop_cost(lam: float, m: float = 0.0, cont_key: str = "v", cterm_key: str = "cterm",
              ccont_key: str = "ccont", q_key: str = "q") -> Plan:
    return first_probe_where(
        lambda p, s: s[q_key] - lam * s[cterm_key] >= s[cont_key] - lam * s[ccont_key] - m)


def stop_direct(lam: float, dy_key: str = "dy", dc_key: str = "dc") -> Plan:
    return first_probe_where(lambda p, s: (s[dy_key][2] - s[dy_key][0]) + lam * s[dc_key] >= 0)


def stop_hysteresis(pred: Predicate, k: int) -> Plan:
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive int, got {k!r}")

    def plan(chain: dict) -> dict | None:
        run = 0
        for probe, score in zip(chain["probes"], chain["scores"]):
            run = run + 1 if pred(probe, score) else 0
            if run >= k:
                return probe
        return None
    return plan


def stop_cusum(m: float, h: float, q_key: str = "q", v_key: str = "v") -> Plan:
    def plan(chain: dict) -> dict | None:
        total = 0.0
        for probe, score in zip(chain["probes"], chain["scores"]):
            total = max(0.0, total + (score[q_key] - score[v_key] + m))
            if total >= h:
                return probe
        return None
    return plan


def natural(chain: dict) -> dict | None:
    return None


def always_fire(chain: dict) -> dict | None:
    probes = chain["probes"]
    return probes[0] if probes else None


def free_family() -> list[tuple[str, Plan]]:
    members: list[tuple[str, Plan]] = [(NATURAL, natural), (ALWAYS_FIRE, always_fire)]
    for cap in FREE_CAPS:
        members.append((f"cap{cap}", lambda c, _b=cap: stop_at_cap(c, _b)))
    members.append(("consecutive_agreement", consecutive_agreement))
    return members


def _fmt_signed(x: float) -> str:
    return f"{x:+.2f}"


def _margin_family(rung: str, q_key: str, v_key: str) -> list[tuple[str, Plan]]:
    return [(f"{rung}:m{_fmt_signed(m)}", stop_margin(m, q_key, v_key)) for m in MARGINS]


def _linear_family(rung: str, q_key: str, v_key: str) -> list[tuple[str, Plan]]:
    return [(f"{rung}:a{_fmt_signed(a)},b{_fmt_signed(b)}", stop_depth_margin(a, b, q_key, v_key))
            for a, b in dm.linear_candidates()]


def _step_family(rung: str, q_key: str, v_key: str) -> list[tuple[str, Plan]]:
    return [(f"{rung}:" + ",".join(f"{v:.2f}" for v in margins), stop_step_margin(margins, q_key, v_key))
            for margins in dm.monotone_candidates()]


KINDS = ("threshold", "margin", "custom")
DIRECTIONS = ("ge", "le")
BUILTIN_RUNGS = ("L0", "L1", "L2", "L3raw", "L3raw_step", "L3", "L3_step", "L5_modular",
                 "L5_direct", "L6", "L6_pi0", "L7", "L8_hyst", "L8_cusum", "B", "B_plus_m")
_BUILTIN_PREFIXES = ("L4_",)


def is_builtin_rung(name: str) -> bool:
    return name in BUILTIN_RUNGS or any(name.startswith(p) for p in _BUILTIN_PREFIXES)


RULE_FACTORIES: dict[str, Callable[..., Plan]] = {}


def register_rule_factory(name: str, factory: Callable[..., Plan]) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError(f"factory name must be a non-empty str, got {name!r}")
    if not callable(factory):
        raise ValueError(f"factory {name!r} is not callable")
    previous = RULE_FACTORIES.get(name)
    if previous is not None and previous is not factory:
        raise ValueError(f"rule factory {name!r} is already registered to a different callable")
    RULE_FACTORIES[name] = factory


GRID_SCALARS = (int, float, str)


@dataclass(frozen=True, eq=False)
class FamilySpec:

    name: str
    grid: tuple = ()
    kind: str = "threshold"
    belief_key: str = "q"
    reference_key: str | None = None
    direction: str = "ge"
    requires: tuple[str, ...] = ()
    params: tuple[str, ...] = ("theta",)
    formats: tuple[str, ...] = ("{:.2f}",)
    factory: str | None = None
    belief_source: str = ""

    def __post_init__(self):
        object.__setattr__(self, "grid", tuple(
            tuple(v) if isinstance(v, (list, tuple)) else v for v in self.grid))
        object.__setattr__(self, "requires", tuple(self.requires))
        object.__setattr__(self, "params", tuple(self.params))
        object.__setattr__(self, "formats", tuple(self.formats))
        if not isinstance(self.name, str) or not self.name or ":" in self.name:
            raise ValueError(f"family name must be a non-empty str without ':', got {self.name!r}")
        if is_builtin_rung(self.name):
            raise ValueError(f"external family {self.name!r} shadows a built-in rung")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {self.direction!r}")
        if not self.grid:
            raise ValueError(f"family {self.name!r} has an empty grid")
        if len(self.params) != len(self.formats):
            raise ValueError(f"family {self.name!r}: {len(self.params)} params but "
                             f"{len(self.formats)} formats")
        if not self.params:
            raise ValueError(f"family {self.name!r} declares no parameters")
        if self.kind == "margin" and not self.reference_key:
            raise ValueError(f"family {self.name!r}: kind 'margin' needs a reference_key")
        if self.kind == "custom" and not self.factory:
            raise ValueError(f"family {self.name!r}: kind 'custom' needs a factory")
        if self.kind in ("threshold", "margin") and len(self.params) != 1:
            raise ValueError(f"family {self.name!r}: kind {self.kind!r} takes exactly one parameter")
        for value in self.grid:
            entries = value if isinstance(value, tuple) else (value,)
            if len(entries) != len(self.params):
                raise ValueError(f"family {self.name!r}: grid entry {value!r} has "
                                 f"{len(entries)} value(s) for {len(self.params)} parameter(s)")
            for entry in entries:
                if isinstance(entry, bool) or not isinstance(entry, GRID_SCALARS):
                    raise ValueError(f"family {self.name!r}: grid values must be int/float/str, "
                                     f"got {type(entry).__name__} ({entry!r})")
        names = [self.member_name(v) for v in self.grid]
        if len(set(names)) != len(names):
            raise ValueError(f"family {self.name!r}: grid produces duplicate member names")

    def identity(self) -> str:
        return json.dumps({"spec": self.to_dict(),
                           "members": [self.member_name(v) for v in self.grid]},
                          sort_keys=True, allow_nan=False)

    def __eq__(self, other) -> bool:
        if not isinstance(other, FamilySpec):
            return NotImplemented
        return self.identity() == other.identity()

    def __hash__(self) -> int:
        return hash(self.identity())

    def required_keys(self) -> set[str]:
        keys = {self.belief_key, *self.requires}
        if self.reference_key:
            keys.add(self.reference_key)
        return keys

    def member_name(self, value) -> str:
        values = value if isinstance(value, tuple) else (value,)
        body = ",".join(label + fmt.format(v)
                        for label, fmt, v in zip(self.params, self.formats, values))
        return f"{self.name}:{body}"

    def plan(self, value) -> Plan:
        values = value if isinstance(value, tuple) else (value,)
        if self.kind == "threshold":
            (theta,) = values
            key, direction = self.belief_key, self.direction
            if direction == "ge":
                return first_probe_where(lambda p, s, _t=theta, _k=key: s[_k] >= _t)
            return first_probe_where(lambda p, s, _t=theta, _k=key: s[_k] <= _t)
        if self.kind == "margin":
            (m,) = values
            q_key, v_key = self.belief_key, self.reference_key
            if self.direction == "ge":
                return stop_margin(m, q_key, v_key)
            return first_probe_where(
                lambda p, s, _m=m, _q=q_key, _v=v_key: s[_q] <= s[_v] + _m)
        factory = RULE_FACTORIES.get(self.factory)
        if factory is None:
            raise KeyError(f"family {self.name!r} needs unregistered rule factory "
                           f"{self.factory!r} (import the module that registers it)")
        return factory(*values, spec=self)

    def members(self) -> list[tuple[str, Plan]]:
        return [(self.member_name(v), self.plan(v)) for v in self.grid]

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "belief_key": self.belief_key,
                "reference_key": self.reference_key, "direction": self.direction,
                "requires": list(self.requires), "params": list(self.params),
                "formats": list(self.formats), "factory": self.factory,
                "belief_source": self.belief_source,
                "grid": [list(v) if isinstance(v, tuple) else v for v in self.grid],
                "n_members": len(self.grid)}

    @classmethod
    def from_dict(cls, row: dict) -> "FamilySpec":
        allowed = {f for f in cls.__dataclass_fields__} | {"n_members"}
        unknown = sorted(set(row) - allowed)
        if unknown:
            raise ValueError(f"family spec has unknown key(s) {unknown}")
        kwargs = {k: v for k, v in row.items() if k != "n_members"}
        if "grid" not in kwargs:
            raise ValueError("family spec needs a 'grid'")
        return cls(**kwargs)


EXTERNAL_FAMILIES: dict[str, FamilySpec] = {}


def register_family(spec: FamilySpec) -> FamilySpec:
    if not isinstance(spec, FamilySpec):
        raise TypeError(f"expected a FamilySpec, got {type(spec).__name__}")
    previous = EXTERNAL_FAMILIES.get(spec.name)
    if previous is not None and previous != spec:
        raise ValueError(f"family {spec.name!r} is already registered with a different spec")
    EXTERNAL_FAMILIES[spec.name] = spec
    return spec


def clear_registry() -> None:
    EXTERNAL_FAMILIES.clear()
    RULE_FACTORIES.clear()


def registry_snapshot() -> dict[str, FamilySpec]:
    return dict(EXTERNAL_FAMILIES)


def registered_since(before: Mapping[str, FamilySpec]) -> list[FamilySpec]:
    return [spec for name, spec in sorted(EXTERNAL_FAMILIES.items())
            if name not in before or before[name] != spec]


def load_family_specs(path) -> list[FamilySpec]:
    payload = json.loads(open(path, encoding="utf-8").read())
    rows = payload.get("families") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list of family specs or {{'families': [...]}}")
    specs = [FamilySpec.from_dict(row) for row in rows]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate family names")
    return specs


def families(available_keys: Iterable[str], *, specs: Iterable["FamilySpec"] | None = None,
             include_registered: bool = True) -> dict[str, list[tuple[str, Plan]]]:
    keys = set(available_keys)
    out: dict[str, list[tuple[str, Plan]]] = {}
    out["L0"] = [(f"L0:theta{t:.2f}", stop_threshold(t)) for t in THETAS]
    out["L1"] = _margin_family("L1", "q", "v")
    if {"q_cal", "v_cal"} <= keys:
        out["L2"] = _margin_family("L2", "q_cal", "v_cal")
    out["L3raw"] = _linear_family("L3raw", "q", "v")
    out["L3raw_step"] = _step_family("L3raw_step", "q", "v")
    if {"q_cal", "v_cal"} <= keys:
        out["L3"] = _linear_family("L3", "q_cal", "v_cal")
        out["L3_step"] = _step_family("L3_step", "q_cal", "v_cal")
    for key in sorted(k for k in keys if k.startswith("vpi0")):
        out[f"L4_{key}"] = _margin_family(f"L4_{key}", "q", key)
    if {"vpi0", "cterm", "ccont_pi0"} <= keys:
        out["L5_modular"] = [(f"L5_modular:lam{lam:.2f}",
                              stop_cost(lam, 0.0, cont_key="vpi0", ccont_key="ccont_pi0"))
                             for lam in LAMBDAS]
    if {"dy", "dc"} <= keys:
        out["L5_direct"] = [(f"L5_direct:lam{lam:.2f}", stop_direct(lam)) for lam in LAMBDAS]
    if "joint" in keys:
        out["L6"] = [(f"L6:m{_fmt_signed(m)}", stop_joint(m, "joint")) for m in MARGINS]
    if "joint_pi0" in keys:
        out["L6_pi0"] = [(f"L6_pi0:m{_fmt_signed(m)}", stop_joint(m, "joint_pi0")) for m in MARGINS]
    if {"vpi0", "r"} <= keys:
        out["L7"] = [(f"L7:mhi{mhi:.2f},m{_fmt_signed(m)}", stop_bracket(mhi, m))
                     for mhi in BRACKET_MHI for m in MARGINS]
    out["L8_hyst"] = [
        (f"L8_hyst:k{k},m{_fmt_signed(m)}",
         stop_hysteresis(lambda p, s, _m=m: s["q"] >= s["v"] - _m, k))
        for k in HYSTERESIS_K for m in MARGINS]
    out["L8_cusum"] = [(f"L8_cusum:h{h:.2f},m{_fmt_signed(m)}", stop_cusum(m, h))
                       for h in CUSUM_H for m in MARGINS]
    if {"cterm", "ccont"} <= keys:
        out["B"] = [(f"B:lam{lam:.2f}", stop_cost(lam)) for lam in LAMBDAS]
        out["B_plus_m"] = [(f"B_plus_m:lam{lam:.2f},m{_fmt_signed(m)}", stop_cost(lam, m))
                           for lam in LAMBDAS for m in B_PLUS_M]
    external = list(EXTERNAL_FAMILIES.values()) if include_registered else []
    external += list(specs or ())
    seen: dict[str, FamilySpec] = {}
    for spec in external:
        previous = seen.get(spec.name)
        if previous is not None and previous != spec:
            raise ValueError(f"two different external families named {spec.name!r}")
        if previous is None and spec.name in out:
            raise ValueError(f"external family {spec.name!r} shadows a built-in rung")
        seen[spec.name] = spec
        if spec.required_keys() <= keys:
            out[spec.name] = spec.members()
    return out


def score_keys(chains: Iterable[dict]) -> set[str]:
    keys: set[str] = set()
    for chain in chains:
        for score in chain.get("scores", ()):
            keys.update(score)
    return keys


def stop_outcome(chain: dict, plan: Plan) -> tuple[int, int]:
    probe = plan(chain)
    if probe is None:
        return int(bool(chain["correct"])), int(chain["thinking_tokens"])
    return int(probe["forced"] is not None and probe["forced"] == chain["gold"]), int(probe["prefix_len"])


def question_table(chains: Sequence[dict], family: Iterable[tuple[str, Plan]]
                   ) -> dict[str, dict[str, tuple[int, int, int]]]:
    members = list(family)
    names = [name for name, _ in members]
    if len(set(names)) != len(names):
        raise ValueError("duplicate member names in family")
    table: dict[str, dict[str, tuple[int, int, int]]] = {}
    for name, plan in members:
        sums: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for chain in chains:
            correct, thinking = stop_outcome(chain, plan)
            entry = sums[chain["qkey"]]
            entry[0] += 1
            entry[1] += correct
            entry[2] += thinking
        table[name] = {q: tuple(v) for q, v in sums.items()}
    return table


def rows_from_table(table: dict[str, dict[str, tuple[int, int, int]]],
                    qkeys: Sequence[str] | None = None) -> list[dict]:
    if NATURAL not in table:
        raise ValueError(f"table needs a {NATURAL!r} member to compute thinking_saving")
    keys = sorted(table[NATURAL]) if qkeys is None else list(qkeys)

    def totals(per_q: dict[str, tuple[int, int, int]]) -> tuple[int, int, int]:
        n = correct = thinking = 0
        for q in keys:
            a, b, c = per_q[q]
            n += a
            correct += b
            thinking += c
        return n, correct, thinking

    natural_thinking = totals(table[NATURAL])[2]
    rows = []
    for name, per_q in table.items():
        n, correct, thinking = totals(per_q)
        if n == 0:
            raise ValueError(f"member {name!r}: no chains over the given questions")
        rows.append({
            "name": name,
            "n": n,
            "accuracy": correct / n,
            "mean_thinking_tokens": thinking / n,
            "thinking_saving": (1.0 - thinking / natural_thinking) if natural_thinking else 0.0,
        })
    return rows


def family_hull(rows: Iterable[dict], natural_row: dict, always_row: dict) -> list[tuple[float, float]]:
    return lc.anchor_hull(list(rows), natural_row, always_row)


CALIB_SALT = "iclr-ladder-cv-2026-08-27"
AGREEMENT = "earliest_agreement"
SELECT_LEVELS = (0.0, 0.01, 0.02, 0.03)
_ACC_EPS = 1e-9


def _by_question(chains: Iterable[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for chain in chains:
        groups[chain["qkey"]].append(chain)
    return {q: sorted(cs, key=lambda c: c["rep"]) for q, cs in sorted(groups.items())}


def calibration_examples(chains: Sequence[dict], kind: str) -> list[dm.WeightedExample]:
    if kind not in ("q", "v"):
        raise ValueError(f"calibration kind must be 'q' or 'v', got {kind!r}")
    rows: list[dm.WeightedExample] = []
    for qkey, qchains in _by_question(chains).items():
        chain_weight = 1.0 / len(qchains)
        for chain in qchains:
            probes, scores = chain["probes"], chain["scores"]
            if not probes:
                continue
            if kind == "q":
                probe_weight = chain_weight / len(probes)
                for probe, score in zip(probes, scores):
                    rows.append(dm.WeightedExample(qkey, float(score["q"]), _depth(probe),
                                                   is_correct(probe["forced"], chain["gold"]),
                                                   probe_weight))
            else:
                rows.append(dm.WeightedExample(qkey, float(scores[-1]["v"]), _depth(probes[-1]),
                                               int(bool(chain["correct"])), chain_weight))
    return rows


def fit_calibrators(chains: Sequence[dict], *, salt: str = CALIB_SALT) -> dict:
    qkeys = sorted({c["qkey"] for c in chains})
    if not qkeys:
        raise ValueError("no chains to calibrate on")
    inner = lc.question_folds(qkeys, n_folds=4, salt=f"{salt}:inner")
    out: dict = {"n_questions": len(qkeys)}
    for kind in ("q", "v"):
        examples = calibration_examples(chains, kind)
        folds = {q: inner[q] for q in {e.qkey for e in examples}}
        c = dm.select_calibration_c(examples, folds)
        out[kind] = dm.fit_calibrator(examples, c)
        out[f"{kind}_c"] = c
    return out


def attach_calibrated(chains: Sequence[dict], models: dict) -> list[dict]:
    out = []
    for chain in chains:
        new = dict(chain)
        new["scores"] = [
            {**score,
             "q_cal": models["q"].calibrate(score["q"], _depth(probe)),
             "v_cal": models["v"].calibrate(score["v"], _depth(probe))}
            for probe, score in zip(chain["probes"], chain["scores"])]
        out.append(new)
    return out


def crossfit_models(chains: Sequence[dict], *, n_folds: int = 4, salt: str = CALIB_SALT
                    ) -> tuple[dict[str, int], dict[int, dict], list[dict]]:
    qkeys = sorted({c["qkey"] for c in chains})
    outer = lc.question_folds(qkeys, n_folds=n_folds, salt=f"{salt}:outer")
    models_by_fold: dict[int, dict] = {}
    records = []
    for fold in range(n_folds):
        fit = [c for c in chains if outer[c["qkey"]] != fold]
        models = fit_calibrators(fit, salt=f"{salt}:outer{fold}")
        models_by_fold[fold] = models
        eval_qkeys = [q for q in qkeys if outer[q] == fold]
        records.append({
            "fold": fold,
            "n_fit_questions": models["n_questions"],
            "n_eval_questions": len(eval_qkeys),
            "q_c": models["q_c"], "v_c": models["v_c"],
            "q_coefficients": list(models["q"].coefficients),
            "v_coefficients": list(models["v"].coefficients),
            "training_qkeys": [q for q in qkeys if outer[q] != fold],
            "eval_qkeys": eval_qkeys,
        })
    return outer, models_by_fold, records


def crossfit_calibrated(chains: Sequence[dict], *, n_folds: int = 4, salt: str = CALIB_SALT
                        ) -> tuple[list[dict], list[dict]]:
    outer, models_by_fold, records = crossfit_models(chains, n_folds=n_folds, salt=salt)
    out = []
    for chain in chains:
        out.extend(attach_calibrated([chain], models_by_fold[outer[chain["qkey"]]]))
    return out, records


def select_members(rows_val: Iterable[dict], natural_val: dict, levels: Sequence[float] = SELECT_LEVELS
                   ) -> list[str]:
    nat_acc = float(natural_val["accuracy"])
    candidates = [r for r in rows_val if r["name"] not in (NATURAL, ALWAYS_FIRE)]
    chosen: list[str] = []
    for level in levels:
        ok = [r for r in candidates if r["accuracy"] >= nat_acc - level - _ACC_EPS]
        if not ok:
            continue
        best = min(ok, key=lambda r: (r["mean_thinking_tokens"], -r["accuracy"], r["name"]))
        if best["name"] not in chosen:
            chosen.append(best["name"])
    return chosen


def weak_pareto_non_dominated(points: Iterable[tuple[float, float]],
                              hull: Sequence[tuple[float, float]]) -> bool:
    if not hull:
        raise ValueError("empty hull")
    for t, a in points:
        if t < hull[0][0]:
            ref = hull[0][1]
        elif t > hull[-1][0]:
            ref = hull[-1][1]
        else:
            ref = lc.acc_at_tokens(hull, t)
        if a > ref:
            return True
    return False


def beats_reference(points: Iterable[tuple[float, float]], ref_row: dict) -> bool:
    return any(a > ref_row["accuracy"] and t < ref_row["mean_thinking_tokens"] for t, a in points)


def go_no_go(per_seed: dict) -> dict:
    if not per_seed:
        raise ValueError("go_no_go needs at least one seed")
    seeds = {}
    for seed, entry in per_seed.items():
        non_dominated = weak_pareto_non_dominated(entry["points"], entry["free_hull"])
        beats = beats_reference(entry["points"], entry["agreement"])
        seeds[seed] = {"non_dominated": non_dominated, "beats_agreement": beats,
                       "pass": non_dominated and beats}
    return {"per_seed": seeds, "pass": all(v["pass"] for v in seeds.values()), "n_seeds": len(seeds)}


def _rows_by_name(table, sample) -> dict[str, dict]:
    return {r["name"]: r for r in rows_from_table(table, sample)}


def paired_bootstrap(tables_by_seed: dict, *, members_by_rung: dict, free_names: Sequence[str],
                     l1_name: str = "L1", n: int, rng) -> dict:
    if n <= 0:
        return {"n": 0}
    seeds = sorted(tables_by_seed)
    qkeys = sorted(tables_by_seed[seeds[0]]["free"][NATURAL])
    for seed in seeds:
        tables = tables_by_seed[seed]
        if sorted(tables["free"][NATURAL]) != qkeys:
            raise ValueError(f"seed {seed}: select question set differs from seed {seeds[0]}")
        if l1_name not in tables["rungs"]:
            raise ValueError(f"seed {seed}: no {l1_name!r} rung table")
        for label, table in [(f"rung {r!r}", t) for r, t in tables["rungs"].items()] + [
                ("agreement", tables["agreement"])]:
            if NATURAL not in table:
                raise ValueError(f"seed {seed}: {label} table has no {NATURAL!r} member")
            if sorted(table[NATURAL]) != qkeys:
                raise ValueError(f"seed {seed}: {label} table's question set differs from the free table")
    rungs = list(members_by_rung)
    counts = {rung: {"pass": 0, "non_dominated": 0, "beats_agreement": 0} for rung in rungs}
    deltas = {rung: {"dacc_pp": [], "dsaving_frac": []} for rung in rungs}
    for _ in range(n):
        sample = [rng.choice(qkeys) for _ in qkeys]
        per_rung: dict[str, dict] = {rung: {} for rung in rungs}
        seed_deltas: dict[str, list[dict]] = {rung: [] for rung in rungs}
        for seed in seeds:
            tables = tables_by_seed[seed]
            free = _rows_by_name(tables["free"], sample)
            natural_row, always_row = free[NATURAL], free[ALWAYS_FIRE]
            free_hull = family_hull([free[name] for name in free_names], natural_row, always_row)
            agreement = _rows_by_name(tables["agreement"], sample)[AGREEMENT]
            l1_hull = family_hull(rows_from_table(tables["rungs"][l1_name], sample),
                                  natural_row, always_row)
            for rung in rungs:
                rows = rows_from_table(tables["rungs"][rung], sample)
                by_name = {r["name"]: r for r in rows}
                selected = members_by_rung[rung]
                names = selected[seed] if isinstance(selected, dict) else selected
                points = [(by_name[m]["mean_thinking_tokens"], by_name[m]["accuracy"]) for m in names]
                per_rung[rung][seed] = {"points": points, "free_hull": free_hull, "agreement": agreement}
                seed_deltas[rung].append(lc.operating_point_delta(
                    l1_hull, family_hull(rows, natural_row, always_row), natural_row))
        for rung in rungs:
            verdict = go_no_go(per_rung[rung])
            counts[rung]["pass"] += verdict["pass"]
            counts[rung]["non_dominated"] += all(v["non_dominated"] for v in verdict["per_seed"].values())
            counts[rung]["beats_agreement"] += all(v["beats_agreement"] for v in verdict["per_seed"].values())
            for key, values in deltas[rung].items():
                readings = [d[key] for d in seed_deltas[rung]]
                values.append(None if any(r is None for r in readings)
                              else sum(readings) / len(readings))
    out: dict = {"n": n, "n_seeds": len(seeds), "n_questions": len(qkeys), "rungs": {}}
    for rung in rungs:
        entry = {f"{k}_frac": counts[rung][k] / n for k in ("pass", "non_dominated", "beats_agreement")}
        for key, values in deltas[rung].items():
            defined = [v for v in values if v is not None]
            entry[f"{key}_mean"] = (sum(defined) / len(defined)) if defined else None
            entry[f"{key}_se"] = statistics.pstdev(defined) if len(defined) > 1 else None
            entry[f"{key}_n_defined"] = len(defined)
        out["rungs"][rung] = entry
    return out


FirePredicate = Callable[[dict, int], bool]
ROW3_LEVEL = 0.01
DEPTH_BIN_LABELS = tuple(
    f"[{lo:g},{hi:g}{']' if i == len(dm.DEPTH_BINS) - 2 else ')'}"
    for i, (lo, hi) in enumerate(zip(dm.DEPTH_BINS, dm.DEPTH_BINS[1:])))


def fire_consecutive_agreement(chain: dict, i: int) -> bool:
    probes = chain["probes"]
    if i < 1:
        return False
    forced = probes[i]["forced"]
    return forced is not None and forced == probes[i - 1]["forced"]


def fire_cap(cap: int) -> FirePredicate:
    if not isinstance(cap, int) or cap < 1:
        raise ValueError(f"cap must be a positive int, got {cap!r}")
    return lambda chain, i: chain["probes"][i]["block"] >= cap


def fire_margin(margin: float, q_key: str = "q", v_key: str = "vpi0") -> FirePredicate:
    return lambda chain, i: chain["scores"][i][q_key] >= chain["scores"][i][v_key] - margin


def fire_depth_margin(a: float, b: float, q_key: str = "q_cal", v_key: str = "v_cal") -> FirePredicate:
    return lambda chain, i: (chain["scores"][i][q_key]
                             >= chain["scores"][i][v_key] - (a + b * _depth(chain["probes"][i])))


def fire_table(chains: Iterable[dict], predicate: FirePredicate) -> list[dict]:
    return [{"qkey": chain["qkey"], "rep": chain["rep"],
             "fire": [bool(predicate(chain, i)) for i in range(len(chain["probes"]))]}
            for chain in chains]


def load_fire_table(path, chains: Iterable[dict] | None = None) -> dict[tuple[str, int], list[bool]]:
    table: dict[tuple[str, int], list[bool]] = {}
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("qkey"), str) or not row["qkey"]:
                raise ValueError(f"{path}:{line_no}: fire row needs a non-empty str qkey")
            rep = row.get("rep")
            if isinstance(rep, bool) or not isinstance(rep, int):
                raise ValueError(f"{path}:{line_no}: fire row needs an int rep")
            fire = row.get("fire")
            if not isinstance(fire, list):
                raise ValueError(f"{path}:{line_no}: fire must be a list")
            if any(type(x) is not bool for x in fire):
                raise ValueError(f"{path}:{line_no}: fire entries must be bools")
            identity = (row["qkey"], rep)
            if identity in table:
                raise ValueError(f"{path}:{line_no}: duplicate fire row for {identity}")
            table[identity] = list(fire)
    if chains is not None:
        for chain in chains:
            identity = (chain["qkey"], chain["rep"])
            if identity not in table:
                raise ValueError(f"{path}: no fire row for chain {identity}")
            if len(table[identity]) != len(chain["probes"]):
                raise ValueError(f"{path}: chain {identity} has {len(chain['probes'])} probes but "
                                 f"{len(table[identity])} fire entries")
    return table


def fire_fraction_by_depth(chains: Iterable[dict], fire: dict[tuple[str, int], list[bool]]) -> dict:
    n = [0] * len(DEPTH_BIN_LABELS)
    fired = [0] * len(DEPTH_BIN_LABELS)
    for chain in chains:
        identity = (chain["qkey"], chain["rep"])
        if identity not in fire:
            raise ValueError(f"fire table has no row for chain {identity}")
        bits = fire[identity]
        if len(bits) != len(chain["probes"]):
            raise ValueError(f"fire table row for chain {identity} has {len(bits)} entries "
                             f"for {len(chain['probes'])} probes")
        for probe, bit in zip(chain["probes"], bits):
            b = dm.depth_bin(_depth(probe))
            n[b] += 1
            fired[b] += bit
    return {label: {"n": n[b], "fired": fired[b], "frac": (fired[b] / n[b]) if n[b] else None}
            for b, label in enumerate(DEPTH_BIN_LABELS)}


_L3_PARAMS = {name: (a, b) for (name, _plan), (a, b) in zip(_linear_family("L3", "q_cal", "v_cal"),
                                                           dm.linear_candidates())}


def select_row3(rows: Iterable[dict], natural_row: dict, level: float = ROW3_LEVEL
                ) -> tuple[str, float, float]:
    linear = [r for r in rows if r["name"] in _L3_PARAMS]
    picked = select_members(linear, natural_row, levels=(level,))
    if not picked:
        raise ValueError(f"no L3 member reaches natural accuracy - {level} on the fitting folds")
    a, b = _L3_PARAMS[picked[0]]
    return picked[0], a, b


def row3_crossfit(chains: Sequence[dict], *, n_folds: int = 4, salt: str = CALIB_SALT,
                  level: float = ROW3_LEVEL) -> tuple[list[dict], list[dict]]:
    outer, models_by_fold, records = crossfit_models(chains, n_folds=n_folds, salt=salt)
    family = [*_linear_family("L3", "q_cal", "v_cal"), (NATURAL, natural)]
    params: dict[int, tuple[float, float]] = {}
    for record in records:
        fold = record["fold"]
        train = attach_calibrated([c for c in chains if outer[c["qkey"]] != fold], models_by_fold[fold])
        rows = rows_from_table(question_table(train, family))
        by_name = {r["name"]: r for r in rows}
        member, a, b = select_row3(rows, by_name[NATURAL], level)
        params[fold] = (a, b)
        record.update({"member": member, "a": a, "b": b, "level": level,
                       "train_point": by_name[member], "train_natural": by_name[NATURAL]})
    out = []
    for chain in chains:
        fold = outer[chain["qkey"]]
        a, b = params[fold]
        (row,) = fire_table(attach_calibrated([chain], models_by_fold[fold]), fire_depth_margin(a, b))
        out.append({**row, "fold": fold, "a": a, "b": b})
    return out, records
