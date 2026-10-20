from __future__ import annotations

import math
from dataclasses import dataclass

from rl_controller.iclr.chains import is_correct

REPLAY_SCHEMA = "iclr-policy-replay/v1"
GRADINGS = ("strict", "loose")
EXPECTATIONS = ("survival_product", "bernoulli_seeded")
PRODUCIBLE_EXPECTATIONS = ("survival_product",)

POLICY_REPLAY_ROW_FIELDS = (
    "schema_version",
    "run_id", "method_family", "method_name", "config_id", "knob", "seed",
    "dataset", "split",
    "model_id", "model_revision", "tokenizer_revision", "prompt_contract_id",
    "grader_id", "selection_split", "selection_rule", "frozen_config",
    "source_branch", "source_commit", "source_artifact", "source_sha256",
    "chain_bank_sha256", "manifest_sha256", "pool",
    "qkey", "n_chains", "e_correct", "e_thinking_tokens", "e_answer_tokens",
    "e_generated_tokens", "aux_generated_tokens", "expectation", "missing_reason",
)
POLICY_REPLAY_HEADER_FIELDS = tuple(
    f for f in POLICY_REPLAY_ROW_FIELDS
    if f not in ("schema_version", "qkey", "n_chains", "e_correct", "e_thinking_tokens",
                 "e_answer_tokens", "e_generated_tokens", "aux_generated_tokens",
                 "expectation", "missing_reason"))


def natural_answer_tokens_archived_math(chain: dict) -> int:
    return 0

DEFAULT_TOKEN_SCALE = 1.0
DEFAULT_LAMBDAS = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3)
COST_AXES = ("thinking", "total")


@dataclass(frozen=True)
class ChainTerms:
    a: tuple[float, ...]
    l: tuple[float, ...]
    a_nat: float
    l_nat: float
    l_total: tuple[float, ...]
    l_total_nat: float
    cost: str = "thinking"

    @property
    def n_probes(self) -> int:
        return len(self.a)


def chain_terms(chain: dict, *, grade: str = "strict",
                token_scale: float = DEFAULT_TOKEN_SCALE,
                cost: str = "thinking") -> ChainTerms:
    if grade not in GRADINGS:
        raise ValueError(f"grade {grade!r} not in {GRADINGS}")
    if cost not in COST_AXES:
        raise ValueError(f"cost {cost!r} not in {COST_AXES}")
    if not token_scale > 0:
        raise ValueError(f"token_scale must be > 0, got {token_scale!r}")
    gold = chain["gold"]
    probes = chain["probes"]
    if grade == "strict":
        a = tuple(float(is_correct(p["forced"], gold)) for p in probes)
    else:
        a = tuple(float(p["forced"] is not None and p["forced"] == gold) for p in probes)
    thinking = tuple(float(p["prefix_len"]) / token_scale for p in probes)
    thinking_nat = float(chain["thinking_tokens"]) / token_scale
    total = tuple(float(p["prefix_len"] + p["forced_tokens"]) / token_scale for p in probes)
    total_nat = thinking_nat + float(natural_answer_tokens_archived_math(chain)) / token_scale
    return ChainTerms(
        a=a,
        l=total if cost == "total" else thinking,
        a_nat=float(int(bool(chain["correct"]))),
        l_nat=total_nat if cost == "total" else thinking_nat,
        l_total=total,
        l_total_nat=total_nat,
        cost=cost,
    )


def survival(pis) -> tuple[list[float], float]:
    out = []
    s = 1.0
    for pi in pis:
        if not 0.0 <= pi <= 1.0:
            raise ValueError(f"pi must be in [0, 1], got {pi!r}")
        out.append(s)
        s *= (1.0 - pi)
    return out, s


_EXPECTATION_TOL = 1e-9


def _clamp_expected_correct(correct: float) -> float:
    if not math.isfinite(correct) or correct > 1.0 + _EXPECTATION_TOL \
            or correct < -_EXPECTATION_TOL:
        raise ValueError(
            f"E[correct] = {correct!r} is outside [0, 1] by more than {_EXPECTATION_TOL}: the "
            "stopping weights sum to 1 and every outcome is in [0, 1], so this is not "
            "floating-point drift -- check the pi values and the per-probe correctness terms")
    return min(1.0, max(0.0, correct))


def expected_outcome(pis, terms: ChainTerms, *, total_cost: bool = False
                     ) -> tuple[float, float]:
    if len(pis) != terms.n_probes:
        raise ValueError(f"{len(pis)} pi values for {terms.n_probes} probes")
    costs = terms.l_total if total_cost else terms.l
    cost_nat = terms.l_total_nat if total_cost else terms.l_nat
    s, tail = survival(pis)
    correct = sum(si * pi * ai for si, pi, ai in zip(s, pis, terms.a)) + tail * terms.a_nat
    tokens = sum(si * pi * li for si, pi, li in zip(s, pis, costs)) + tail * cost_nat
    return _clamp_expected_correct(correct), tokens


def chain_objective(pis, terms: ChainTerms, lam: float) -> float:
    correct, tokens = expected_outcome(pis, terms)
    return correct - lam * tokens


def chain_loss(pis, terms: ChainTerms, lam: float) -> float:
    return -chain_objective(pis, terms, lam)


@dataclass(frozen=True)
class LambdaSchedule:
    kind: str = "geometric"
    start_mult: float = 10.0
    epochs: int = 4

    def __post_init__(self):
        if self.kind not in ("geometric", "constant"):
            raise ValueError(f"unknown lambda schedule kind {self.kind!r}")
        if self.kind == "geometric":
            if not self.start_mult >= 1.0:
                raise ValueError(f"start_mult must be >= 1, got {self.start_mult!r}")
            if self.epochs < 1:
                raise ValueError(f"anneal epochs must be >= 1, got {self.epochs!r}")

    @property
    def anneal_epochs(self) -> int:
        return 0 if self.kind == "constant" else self.epochs

    def at(self, epoch: int, lam: float) -> float:
        if epoch < 0:
            raise ValueError(f"epoch must be >= 0, got {epoch}")
        if self.kind == "constant":
            return lam
        frac = min(epoch, self.epochs) / self.epochs
        return lam * self.start_mult ** (1.0 - frac)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "start_mult": self.start_mult, "epochs": self.epochs,
                "anneal_epochs": self.anneal_epochs}


DEFAULT_ANNEAL = LambdaSchedule()


def parse_schedule(spec: str) -> LambdaSchedule:
    spec = (spec or "").strip()
    if not spec:
        return DEFAULT_ANNEAL
    parts = spec.split(":")
    kind = parts[0]
    if kind == "constant":
        if len(parts) != 1:
            raise ValueError(f"schedule {spec!r}: `constant` takes no parameters")
        return LambdaSchedule(kind="constant")
    if kind != "geometric":
        raise ValueError(f"unknown lambda schedule kind {kind!r} in {spec!r}")
    if len(parts) == 1:
        return LambdaSchedule()
    if len(parts) != 3:
        raise ValueError(f"schedule {spec!r}: expected `geometric:<start_mult>:<epochs>`")
    try:
        return LambdaSchedule(kind="geometric", start_mult=float(parts[1]), epochs=int(parts[2]))
    except ValueError as exc:
        raise ValueError(f"schedule {spec!r}: {exc}") from None


def parse_lambdas(spec: str) -> tuple[float, ...]:
    spec = (spec or "").strip()
    if not spec:
        return DEFAULT_LAMBDAS
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value < 0:
            raise ValueError(f"lambda must be >= 0, got {value}")
        out.append(value)
    if not out:
        raise ValueError(f"no lambdas parsed from {spec!r}")
    return tuple(out)


def osp_chain_loss(logits, terms: ChainTerms, lam: float, *, device=None):
    import torch

    z = logits.reshape(-1)
    if z.shape[0] != terms.n_probes:
        raise ValueError(f"{int(z.shape[0])} logits for {terms.n_probes} probes")
    dev = device if device is not None else z.device
    reward = torch.tensor([a - lam * l for a, l in zip(terms.a, terms.l)],
                          dtype=z.dtype, device=dev)
    reward_nat = terms.a_nat - lam * terms.l_nat
    if terms.n_probes == 0:
        return -torch.as_tensor(reward_nat, dtype=z.dtype, device=dev)
    log_cont = -torch.nn.functional.softplus(z)
    log_stop = -torch.nn.functional.softplus(-z)
    cum = torch.cumsum(log_cont, dim=0)
    log_s = torch.cat([torch.zeros(1, dtype=z.dtype, device=dev), cum[:-1]])
    weights = torch.exp(log_s + log_stop)
    tail = torch.exp(cum[-1])
    return -(torch.dot(weights, reward) + tail * reward_nat)


SUFFIX_BINS = ((0, 3000), (3000, 6000), (6000, 9000), (9000, 12001))


def suffix_terms(terms: ChainTerms, k: int) -> ChainTerms:
    if not 0 <= k < terms.n_probes:
        raise ValueError(f"suffix start {k} outside the chain's {terms.n_probes} probes")
    return ChainTerms(a=terms.a[k:], l=terms.l[k:], a_nat=terms.a_nat, l_nat=terms.l_nat,
                      l_total=terms.l_total[k:], l_total_nat=terms.l_total_nat,
                      cost=terms.cost)


def suffix_bin(prefix_len) -> int | None:
    for index, (lo, hi) in enumerate(SUFFIX_BINS):
        if lo <= prefix_len < hi:
            return index
    return None


def sample_suffix_start(prefix_lens, rng) -> int | None:
    occupied: dict[int, list[int]] = {}
    for index, prefix in enumerate(prefix_lens):
        b = suffix_bin(prefix)
        if b is not None:
            occupied.setdefault(b, []).append(index)
    if not occupied:
        return None
    return rng.choice(occupied[rng.choice(sorted(occupied))])


def suffix_alpha(update: int, total_updates: int) -> float:
    if total_updates < 1:
        raise ValueError(f"total_updates must be >= 1, got {total_updates}")
    return max(0.0, 1.0 - 2.0 * update / total_updates)


def osp_suffix_aux_loss(logits, terms: ChainTerms, lam: float, k: int | None, alpha: float,
                        *, device=None):
    base = osp_chain_loss(logits, terms, lam, device=device)
    if k is None or alpha == 0.0:
        return base
    z = logits.reshape(-1)
    return base + alpha * osp_chain_loss(z[k:], suffix_terms(terms, k), lam, device=device)


def logit_gradients(pis, terms: ChainTerms, lam: float) -> tuple[list[float], list[float]]:
    if len(pis) != terms.n_probes:
        raise ValueError(f"{len(pis)} pi values for {terms.n_probes} probes")
    rewards = [a - lam * l for a, l in zip(terms.a, terms.l)]
    cont = [0.0] * len(pis)
    utility = terms.a_nat - lam * terms.l_nat
    for i in reversed(range(len(pis))):
        cont[i] = utility
        utility = pis[i] * rewards[i] + (1.0 - pis[i]) * utility
    s, _ = survival(pis)
    return ([s[i] * pis[i] * (1.0 - pis[i]) * (rewards[i] - cont[i]) for i in range(len(pis))],
            cont)


def depth_profile(pairs, lam: float, *, saturation: float = 1e-3) -> dict:
    bins = [{"n_probes": 0, "pi": 0.0, "survival": 0.0, "stop_mass": 0.0, "saturated": 0,
             "abs_grad": 0.0, "grad": 0.0, "advantage": 0.0} for _ in SUFFIX_BINS]
    n_chains, natural = 0, 0.0
    for pis, terms, prefix_lens in pairs:
        n_chains += 1
        s, tail = survival(pis)
        natural += tail
        grads, cont = logit_gradients(pis, terms, lam)
        for i, pi in enumerate(pis):
            b = suffix_bin(prefix_lens[i])
            if b is None:
                continue
            row = bins[b]
            row["n_probes"] += 1
            row["pi"] += pi
            row["survival"] += s[i]
            row["stop_mass"] += s[i] * pi
            row["saturated"] += int(pi < saturation or pi > 1.0 - saturation)
            row["abs_grad"] += abs(grads[i])
            row["grad"] += grads[i]
            row["advantage"] += terms.a[i] - lam * terms.l[i] - cont[i]
    out = []
    for (lo, hi), row in zip(SUFFIX_BINS, bins):
        n = row["n_probes"]
        out.append({"bin": [lo, hi], "n_probes": n,
                    **{k: (row[k] / n if n else None)
                       for k in ("pi", "survival", "stop_mass", "abs_grad", "advantage")},
                    "saturated_frac": (row["saturated"] / n if n else None),
                    "grad_sum_per_chain": (row["grad"] / n_chains if n_chains else None)})
    return {"lambda": lam, "n_chains": n_chains,
            "natural_mass": (natural / n_chains if n_chains else None), "bins": out}


def expected_utility(logits, accuracy, total_cost, a_nat, c_nat, lam):
    import torch
    import torch.nn.functional as F

    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    z = logits.to(dtype)
    dev = z.device
    a = torch.as_tensor(accuracy, device=dev, dtype=dtype)
    c = torch.as_tensor(total_cost, device=dev, dtype=dtype)
    price = torch.as_tensor(lam, device=dev, dtype=dtype)
    natural = (torch.as_tensor(a_nat, device=dev, dtype=dtype)
               - price * torch.as_tensor(c_nat, device=dev, dtype=dtype))
    if z.shape[-1] == 0:
        return natural + z.sum() * 0
    price_col = price.unsqueeze(-1) if price.dim() and z.dim() > 1 else price
    log_continue = F.logsigmoid(-z)
    cum = log_continue.cumsum(-1)
    exclusive = torch.cat([torch.zeros_like(cum[..., :1]), cum[..., :-1]], dim=-1)
    mass = exclusive.exp() * torch.sigmoid(z)
    tail = cum[..., -1].exp()
    return (mass * (a - price_col * c)).sum(-1) + tail * natural


def utility_terms(terms: ChainTerms):
    return terms.a, terms.l, terms.a_nat, terms.l_nat


def lambda_grid_features(lambdas):
    return [math.log10(float(l)) + 5.0 for l in lambdas]


INFERENCE_LAMBDAS = tuple(2e-6 * 20.0 ** (k / 12.0) for k in range(13))


def osp_expected(logits, terms: ChainTerms, *, total_cost: bool = False):
    import torch

    z = logits.reshape(-1)
    if z.shape[0] != terms.n_probes:
        raise ValueError(f"{int(z.shape[0])} logits for {terms.n_probes} probes")
    costs = terms.l_total if total_cost else terms.l
    cost_nat = terms.l_total_nat if total_cost else terms.l_nat
    if terms.n_probes == 0:
        return (torch.as_tensor(terms.a_nat, dtype=z.dtype, device=z.device),
                torch.as_tensor(cost_nat, dtype=z.dtype, device=z.device))
    log_cont = -torch.nn.functional.softplus(z)
    log_stop = -torch.nn.functional.softplus(-z)
    cum = torch.cumsum(log_cont, dim=0)
    log_s = torch.cat([torch.zeros(1, dtype=z.dtype, device=z.device), cum[:-1]])
    weights = torch.exp(log_s + log_stop)
    tail = torch.exp(cum[-1])
    acc = torch.tensor(terms.a, dtype=z.dtype, device=z.device)
    cost = torch.tensor(costs, dtype=z.dtype, device=z.device)
    return (torch.dot(weights, acc) + tail * terms.a_nat,
            torch.dot(weights, cost) + tail * cost_nat)


def chain_pis(chain: dict, pi_key: str = "pi") -> list[float]:
    scores = chain.get("scores")
    if scores is None:
        raise ValueError(f"chain {(chain.get('qkey'), chain.get('rep'))} has no scores")
    if len(scores) != len(chain["probes"]):
        raise ValueError(f"chain {(chain['qkey'], chain['rep'])} has {len(scores)} scores for "
                         f"{len(chain['probes'])} probes")
    out = []
    for index, score in enumerate(scores):
        if pi_key not in score:
            raise ValueError(f"chain {(chain['qkey'], chain['rep'])} probe {index}: no "
                             f"{pi_key!r} in scores")
        pi = score[pi_key]
        if isinstance(pi, bool) or not isinstance(pi, (int, float)) or not 0.0 <= pi <= 1.0:
            raise ValueError(f"chain {(chain['qkey'], chain['rep'])} probe {index}: "
                             f"{pi_key} = {pi!r} is not a probability")
        out.append(float(pi))
    return out


def replay_chain(chain: dict, *, pi_key: str = "pi", grade: str = "loose",
                 natural_answer_tokens=natural_answer_tokens_archived_math) -> dict:
    terms = chain_terms(chain, grade=grade)
    pis = chain_pis(chain, pi_key)
    correct, thinking = expected_outcome(pis, terms)
    s, tail = survival(pis)
    nat_answer = float(natural_answer_tokens(chain))
    if nat_answer < 0:
        raise ValueError(f"chain {(chain['qkey'], chain['rep'])}: negative natural answer tokens")
    answer = sum(si * pi * float(p["forced_tokens"])
                 for si, pi, p in zip(s, pis, chain["probes"])) + tail * nat_answer
    return {"qkey": chain["qkey"], "rep": chain["rep"], "e_correct": correct,
            "e_thinking_tokens": thinking, "e_answer_tokens": answer,
            "e_generated_tokens": thinking + answer, "p_natural": tail}


def expected_point(chains, *, pi_key: str = "pi", grade: str = "loose",
                   natural_answer_tokens=natural_answer_tokens_archived_math) -> dict:
    n = len(chains)
    if not n:
        raise ValueError("no chains")
    rows = [replay_chain(c, pi_key=pi_key, grade=grade,
                         natural_answer_tokens=natural_answer_tokens) for c in chains]
    return {"n": n,
            "accuracy": sum(r["e_correct"] for r in rows) / n,
            "mean_thinking_tokens": sum(r["e_thinking_tokens"] for r in rows) / n,
            "mean_answer_tokens": sum(r["e_answer_tokens"] for r in rows) / n,
            "mean_generated_tokens": sum(r["e_generated_tokens"] for r in rows) / n,
            "mean_p_natural": sum(r["p_natural"] for r in rows) / n}


def replay_header(**fields) -> dict:
    missing = set(POLICY_REPLAY_HEADER_FIELDS) - set(fields)
    unknown = set(fields) - set(POLICY_REPLAY_HEADER_FIELDS)
    if missing or unknown:
        raise ValueError(f"policy-replay header: missing {sorted(missing)}, "
                         f"unknown {sorted(unknown)}")
    knob = fields["knob"]
    if not isinstance(knob, dict) or set(knob) != {"name", "value"}:
        raise ValueError(f"knob must be {{'name', 'value'}}, got {knob!r}")
    for name in ("chain_bank_sha256", "manifest_sha256"):
        value = fields[name]
        if not isinstance(value, str) or len(value) != 64 \
                or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{name} must be 64 lowercase hex chars, got {value!r}")
    if not isinstance(fields["pool"], str) or not fields["pool"]:
        raise ValueError(f"pool must be a non-empty str, got {fields['pool']!r}")
    return dict(fields)


def validate_replay_row(row) -> dict:
    missing = set(POLICY_REPLAY_ROW_FIELDS) - set(row)
    unknown = set(row) - set(POLICY_REPLAY_ROW_FIELDS)
    if missing or unknown:
        raise ValueError(f"policy-replay row {row.get('qkey')!r}: missing {sorted(missing)}, "
                         f"unknown {sorted(unknown)}")
    if row["schema_version"] != REPLAY_SCHEMA:
        raise ValueError(f"schema_version must be {REPLAY_SCHEMA!r}, got {row['schema_version']!r}")
    if row["expectation"] not in EXPECTATIONS:
        raise ValueError(f"expectation {row['expectation']!r} not in {EXPECTATIONS}")
    for name in ("chain_bank_sha256", "manifest_sha256"):
        value = row[name]
        if not isinstance(value, str) or len(value) != 64 \
                or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{name} must be 64 lowercase hex chars, got {value!r}")
    if not isinstance(row["pool"], str) or not row["pool"]:
        raise ValueError(f"pool must be a non-empty str, got {row['pool']!r}")
    numeric = ("n_chains", "e_correct", "e_thinking_tokens", "e_answer_tokens",
               "e_generated_tokens", "aux_generated_tokens")
    if row["missing_reason"] is not None:
        if any(row[f] is not None for f in numeric):
            raise ValueError(f"row {row['qkey']!r}: missing_reason set but expectations present")
        return dict(row)
    if not isinstance(row["n_chains"], int) or isinstance(row["n_chains"], bool) \
            or row["n_chains"] < 1:
        raise ValueError(f"row {row['qkey']!r}: n_chains must be an int >= 1")
    for field in numeric[1:]:
        value = row[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value < 0:
            raise ValueError(f"row {row['qkey']!r}: {field} must be finite and >= 0")
    if row["e_correct"] > 1.0:
        raise ValueError(f"row {row['qkey']!r}: e_correct must be in [0, 1]")
    total = row["e_thinking_tokens"] + row["e_answer_tokens"]
    if abs(row["e_generated_tokens"] - total) > 1e-6:
        raise ValueError(f"row {row['qkey']!r}: e_generated_tokens {row['e_generated_tokens']} "
                         f"!= thinking + answer {total}")
    return dict(row)


def unavailable_row(qkey: str, header: dict, reason: str,
                    *, expectation: str = "survival_product") -> dict:
    if not reason:
        raise ValueError("missing_reason must be a non-empty string")
    if expectation not in PRODUCIBLE_EXPECTATIONS:
        raise ValueError(f"this producer computes {PRODUCIBLE_EXPECTATIONS}, "
                         f"refusing to label a row {expectation!r}")
    row = {"schema_version": REPLAY_SCHEMA, **header, "qkey": qkey,
           "n_chains": None, "e_correct": None, "e_thinking_tokens": None,
           "e_answer_tokens": None, "e_generated_tokens": None,
           "aux_generated_tokens": None, "expectation": expectation,
           "missing_reason": reason}
    return validate_replay_row(row)


def replay_question_rows(chains, *, header: dict, pi_key: str = "pi", grade: str = "loose",
                         expectation: str = "survival_product",
                         natural_answer_tokens=natural_answer_tokens_archived_math,
                         aux_generated_tokens: float = 0.0) -> list[dict]:
    if grade not in GRADINGS:
        raise ValueError(f"grade {grade!r} not in {GRADINGS}")
    if expectation not in PRODUCIBLE_EXPECTATIONS:
        raise ValueError(f"this producer computes {PRODUCIBLE_EXPECTATIONS}, refusing to "
                         f"label rows {expectation!r}: a seeded Bernoulli replay is a "
                         "different computation and ships as a record type 1 run dir")
    header = replay_header(**header)
    per_q: dict[str, list[float]] = {}
    for chain in chains:
        row = replay_chain(chain, pi_key=pi_key, grade=grade,
                           natural_answer_tokens=natural_answer_tokens)
        acc = per_q.setdefault(row["qkey"], [0.0, 0.0, 0.0, 0.0])
        acc[0] += 1
        acc[1] += row["e_correct"]
        acc[2] += row["e_thinking_tokens"]
        acc[3] += row["e_answer_tokens"]
    rows = []
    for qkey in sorted(per_q):
        n, correct, thinking, answer = per_q[qkey]
        e_thinking = thinking / n
        e_answer = answer / n
        rows.append(validate_replay_row({
            "schema_version": REPLAY_SCHEMA, **header, "qkey": qkey, "n_chains": int(n),
            "e_correct": correct / n,
            "e_thinking_tokens": e_thinking,
            "e_answer_tokens": e_answer,
            "e_generated_tokens": e_thinking + e_answer,
            "aux_generated_tokens": float(aux_generated_tokens),
            "expectation": expectation, "missing_reason": None,
        }))
    return rows


def replay_summary(rows, *, allow_partial: bool = False) -> dict:
    missing = [r for r in rows if r["missing_reason"] is not None]
    if missing and not allow_partial:
        raise ValueError(
            f"{len(missing)} of {len(rows)} questions are unavailable "
            f"(e.g. {missing[0]['qkey']!r}: {missing[0]['missing_reason']}): this config is "
            "UNAVAILABLE on this pool, not a partial frontier point (contract §2.2 rule 7). "
            "Pass allow_partial=True only for a diagnostic that is not published.")
    usable = [r for r in rows if r["missing_reason"] is None]
    n = sum(r["n_chains"] for r in usable)
    if not n:
        raise ValueError("no usable chains in the replay rows")
    correct = sum(r["e_correct"] * r["n_chains"] for r in usable)
    thinking = sum(r["e_thinking_tokens"] * r["n_chains"] for r in usable)
    generated = sum(r["e_generated_tokens"] * r["n_chains"] for r in usable)
    return {"n_questions": len(usable), "n_missing_questions": len(rows) - len(usable),
            "n": n, "accuracy": correct / n,
            "mean_thinking_tokens": thinking / n, "mean_generated_tokens": generated / n}


def natural_summary(chains) -> dict:
    n = len(chains)
    if not n:
        raise ValueError("no chains")
    correct = sum(int(bool(c["correct"])) for c in chains)
    thinking = sum(int(c["thinking_tokens"]) for c in chains)
    return {"n": n, "accuracy": correct / n, "mean_thinking_tokens": thinking / n}


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
