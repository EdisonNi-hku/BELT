from __future__ import annotations

from rl_controller.iclr.learning_curve import choose_epoch
from rl_controller.iclr.osp_policy import LambdaSchedule, chain_terms


def gathers_probes_only(head_mode: str) -> bool:
    if head_mode not in ("osp", "qv"):
        raise ValueError(f"unknown head_mode {head_mode!r}")
    return head_mode == "osp"


def require_epoch_budget(epochs: int, schedule: LambdaSchedule, patience: int) -> None:
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")
    floor = schedule.anneal_epochs
    needed = floor + patience + 1
    if epochs < needed:
        raise ValueError(
            f"--epochs {epochs} is below the registered floor: annealing finishes at epoch "
            f"{floor} and patience is {patience}, so at least {needed} epochs are needed. "
            "Below it, early stopping would select an epoch trained at a multiple of the "
            "nominal lambda and report it under the nominal label.")


def lambda_at(epoch: int, lam: float, schedule: LambdaSchedule) -> float:
    return schedule.at(epoch, lam)


def checkpoint_eligible(epoch: int, schedule: LambdaSchedule) -> bool:
    return epoch >= schedule.anneal_epochs


def select_epoch(history, *, schedule: LambdaSchedule, key: str, minimise: bool,
                 patience: int) -> dict:
    eligible = [row for row in history if checkpoint_eligible(row["epoch"], schedule)]
    if not eligible:
        raise ValueError(
            f"no epoch reached the checkpoint floor {schedule.anneal_epochs}: every epoch in "
            "this history was trained at an annealed lambda, so there is nothing that may be "
            "reported under the nominal lambda")
    chosen = choose_epoch(eligible, key=key, minimise=minimise, patience=patience)
    return {**chosen, "floor": schedule.anneal_epochs,
            "n_eligible_epochs": len(eligible), "n_history_epochs": len(history)}


def should_stop(epoch: int, best_epoch: int | None, *, schedule: LambdaSchedule,
                patience: int) -> bool:
    floor = schedule.anneal_epochs
    if epoch < floor:
        return False
    reference = best_epoch if (best_epoch is not None and best_epoch >= floor) else floor - 1
    return (epoch - reference) >= patience


def attach_osp_terms(chains, *, grade: str = "strict", token_scale: float = 1.0,
                     cost: str = "thinking") -> int:
    n = 0
    for chain in chains:
        chain["_osp_terms"] = chain_terms(chain, grade=grade, token_scale=token_scale,
                                          cost=cost)
        n += 1
    return n


def terms_summary(chains) -> dict:
    terms = [c["_osp_terms"] for c in chains if c.get("_osp_terms") is not None]
    if not terms:
        raise ValueError("no chains carry _osp_terms; call attach_osp_terms first")
    n_probes = sum(t.n_probes for t in terms)
    if not n_probes:
        raise ValueError("no probes across the pool: the objective has no decision points")
    return {
        "n_chains": len(terms),
        "n_probes": n_probes,
        "mean_probes_per_chain": n_probes / len(terms),
        "mean_a": sum(sum(t.a) for t in terms) / n_probes,
        "mean_l": sum(sum(t.l) for t in terms) / n_probes,
        "mean_a_nat": sum(t.a_nat for t in terms) / len(terms),
        "mean_l_nat": sum(t.l_nat for t in terms) / len(terms),
        "max_l_nat": max(t.l_nat for t in terms),
    }


def lambda_scale_check(chains, lambdas) -> dict:
    summary = terms_summary(chains)
    scale = summary["mean_l_nat"]
    return {
        "mean_l_nat": scale,
        "points": [{"lambda": lam, "cost_of_full_chain": lam * scale,
                    "degenerate": not 0.01 <= lam * scale <= 3.0}
                   for lam in lambdas],
    }


def logical_dump_path(path):
    from pathlib import Path as _P
    import re

    out = _P(path)
    stem = re.sub(r"\.[0-9a-f]{12}$", "", out.stem)
    return out.with_name(stem + out.suffix)


def index_dump_rows(rows) -> dict:
    out = {}
    for row in rows:
        key = (row["qkey"], row["rep"], int(row["end_index"]))
        if key in out:
            raise ValueError(f"label dump has two rows for {key}: a boundary cannot carry "
                             "two label sets")
        out[key] = row["labels"]
    if not out:
        raise ValueError("label dump contains no rows")
    return out


def chain_dump_labels(chain, indexed, names) -> dict:
    n_ends = len(chain["_ends"])
    out = {name: [] for name in names}
    for end_index in range(n_ends):
        key = (chain["qkey"], chain["rep"], end_index)
        if key not in indexed:
            raise ValueError(f"label dump has no row for {key}: the dump and the chain set "
                             "disagree about which boundaries exist")
        labels = indexed[key]
        missing = [n for n in names if n not in labels]
        if missing:
            raise ValueError(f"{key}: dump row lacks target(s) {missing}")
        for name in names:
            out[name].append(labels[name])
    return out


def check_positions(chain, names, positions_of) -> None:
    probe_pos = set(chain["_probe_pos"])
    for name in names:
        if positions_of(name) != "probes":
            continue
        labelled = {i for i, y in enumerate(chain["_dump_labels"][name]) if y is not None}
        stray = labelled - probe_pos
        if stray:
            raise ValueError(
                f"chain {(chain['qkey'], chain['rep'])}: target {name!r} is probes-positioned "
                f"but is labelled at {len(stray)} unprobed boundaries (e.g. {sorted(stray)[:3]})")


def attach_dump_labels(chains, rows, names, positions_of=None) -> dict:
    names = list(names)
    if not names:
        raise ValueError("no target names")
    indexed = index_dump_rows(rows)
    stats = {name: {"n": 0, "labelled": 0} for name in names}
    for chain in chains:
        chain["_dump_labels"] = chain_dump_labels(chain, indexed, names)
        if positions_of is not None:
            check_positions(chain, names, positions_of)
        for name in names:
            values = chain["_dump_labels"][name]
            stats[name]["n"] += len(values)
            stats[name]["labelled"] += sum(v is not None for v in values)
    for name in names:
        row = stats[name]
        row["masked_fraction"] = 1.0 - (row["labelled"] / row["n"] if row["n"] else 0.0)
    return {"targets": names, "n_chains": len(chains), "per_target": stats}


RESUME_STATE_KEYS = (
    "format",
    "epoch",
    "stopped",
    "trainable",
    "best_trainable",
    "optimizer",
    "rng_python",
    "rng_torch",
    "history",
    "best",
    "train_digest",
    "args_digest",
)

RESUME_FORMAT = "ospruner-resume/v1"


def train_order_digest(chains) -> str:
    import hashlib

    digest = hashlib.sha256()
    for chain in chains:
        probes = chain.get("probes") or []
        ends = chain.get("block_ends") or []
        digest.update((
            f"{chain['qkey']}:{chain['rep']}:{chain.get('gold')}:{chain.get('answer')}:"
            f"{int(bool(chain.get('correct')))}:{chain.get('outcome')}:"
            f"{chain.get('thinking_tokens')}:{len(ends)}:{ends[-1] if ends else 0}:"
            f"{len(probes)}:{[p.get('prefix_len') for p in probes]}:"
            f"{[p.get('forced') for p in probes]}\n").encode())
    return digest.hexdigest()


def run_args_digest(mapping) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(mapping, sort_keys=True, default=str).encode()).hexdigest()


def validate_resume_state(state) -> dict:
    if not isinstance(state, dict):
        raise ValueError("resume state must be a dict")
    missing = sorted(set(RESUME_STATE_KEYS) - set(state))
    unknown = sorted(set(state) - set(RESUME_STATE_KEYS))
    if missing or unknown:
        raise ValueError(f"resume state: missing {missing}, unknown {unknown}")
    if state["format"] != RESUME_FORMAT:
        raise ValueError(f"resume format {state['format']!r} != {RESUME_FORMAT!r}")
    if not isinstance(state["epoch"], int) or state["epoch"] < 0:
        raise ValueError(f"resume epoch must be a non-negative int, got {state['epoch']!r}")
    if not state["trainable"]:
        raise ValueError("resume state carries no trainable tensors")
    if not isinstance(state["stopped"], bool):
        raise ValueError(f"resume `stopped` must be a bool, got {state['stopped']!r}")
    if state["best"].get("epoch") is not None and not state["best_trainable"]:
        raise ValueError("resume state names a best epoch but carries no best weights")
    return state


def check_resume_compatible(state, *, train_digest: str, args_digest: str) -> None:
    if state["train_digest"] != train_digest:
        raise ValueError(
            "resume refused: the training population differs from the checkpoint's "
            f"({state['train_digest'][:12]} != {train_digest[:12]}). Same OUT_ROOT, different "
            "pool/cap/split -- resuming would silently continue a different experiment.")
    if state["args_digest"] != args_digest:
        raise ValueError(
            "resume refused: the run-defining flags differ from the checkpoint's "
            f"({state['args_digest'][:12]} != {args_digest[:12]}). lambda, the annealing "
            "schedule, the target set or a population knob changed.")


def replay_shuffles(items, seed: int, n_epochs: int):
    import random

    if n_epochs < 0:
        raise ValueError(f"n_epochs must be >= 0, got {n_epochs}")
    rng = random.Random(seed)
    for _ in range(n_epochs):
        rng.shuffle(items)
    return rng


def resume_plan(state, *, epochs: int) -> dict:
    done = state["epoch"] + 1 if state is not None else 0
    stopped = bool(state["stopped"]) if state is not None else False
    return {"start_epoch": done, "completed_epochs": done,
            "remaining": 0 if stopped else max(0, epochs - done),
            "early_stopped": stopped,
            "complete": stopped or done >= epochs}
