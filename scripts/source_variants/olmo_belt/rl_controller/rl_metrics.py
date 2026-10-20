from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

from rl_controller.budget_episode import (
    BOUNDARY_ACTIONS,
    INNER_ACTIONS,
    dr_grpo_advantages,
)

ACTIONS = INNER_ACTIONS

__all__ = [
    "MetricsError",
    "action_metrics",
    "boundary_action_metrics",
    "draw_outcome_metrics",
    "advantage_metrics",
    "episode_metrics",
    "eval_metrics",
    "group_metrics",
    "head_metrics",
    "policy_entropy",
    "WandbSink",
]

FORBIDDEN_EVAL_SPLITS = ("report",)


class MetricsError(RuntimeError):
    pass


def _mean(values) -> float:
    values = list(values)
    if not values:
        raise MetricsError("mean of an empty sequence")
    return sum(values) / len(values)


def _quantile(values, q: float) -> float:
    values = sorted(values)
    if not values:
        raise MetricsError("quantile of an empty sequence")
    index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[index]


def policy_entropy(probs: Sequence[float]) -> float:
    if not probs:
        raise MetricsError("empty distribution")
    total = sum(probs)
    if total <= 0 or any(p < 0 for p in probs):
        raise MetricsError(f"not a distribution: {probs!r}")
    return -sum((p / total) * math.log(p / total) for p in probs if p > 0)


def action_metrics(steps, per_step_probs=None) -> dict:
    steps = list(steps)
    if not steps:
        raise MetricsError("no steps to summarise")
    counts = Counter(step.action for step in steps)
    unknown = set(counts) - set(ACTIONS)
    if unknown:
        raise MetricsError(f"unknown actions in steps: {sorted(unknown)}")
    out = {f"actions/{a}": counts.get(a, 0) / len(steps) for a in ACTIONS}
    out["actions/decisions"] = len(steps)
    if per_step_probs is not None:
        per_step_probs = list(per_step_probs)
        if len(per_step_probs) != len(steps):
            raise MetricsError(
                f"{len(per_step_probs)} distributions for {len(steps)} steps"
            )
        out["actions/entropy"] = _mean(policy_entropy(p) for p in per_step_probs)
        out["actions/entropy_max"] = math.log(len(ACTIONS))
    return out


def boundary_action_metrics(boundary_steps) -> dict:
    steps = list(boundary_steps)
    if not steps:
        return {"boundary/decisions": 0}
    counts = Counter(step.action for step in steps)
    unknown = set(counts) - set(BOUNDARY_ACTIONS)
    if unknown:
        raise MetricsError(f"unknown boundary actions: {sorted(unknown)}")
    return {
        **{f"boundary/{a}": counts.get(a, 0) / len(steps) for a in BOUNDARY_ACTIONS},
        "boundary/decisions": len(steps),
    }


def draw_outcome_metrics(episodes) -> dict:
    draws = [d for e in episodes for d in e.draws]
    if not draws:
        raise MetricsError("no draws")
    n = len(draws)
    outcomes = Counter(d.outcome for d in draws)
    return {
        "draws/total": n,
        "draws/answered": outcomes.get("answered", 0) / n,
        "draws/abandoned": outcomes.get("abandoned", 0) / n,
        "draws/completed": outcomes.get("completed", 0) / n,
        "draws/cut": outcomes.get("cut", 0) / n,
        "draws/vote_right": sum(1 for d in draws if d.vote_value > 0) / n,
        "draws/vote_wrong": sum(1 for d in draws if d.vote_value < 0) / n,
        "draws/no_vote": sum(1 for d in draws if d.vote_value == 0) / n,
        "draws/tokens_mean": _mean(d.tokens for d in draws),
        **{f"draws/vote_right_given_{o}": _vote_rate(draws, o)
           for o in ("answered", "completed", "cut")},
        **{f"draws/vote_rate_given_{o}": _voted_rate(draws, o)
           for o in ("answered", "completed", "cut")},
    }


def _vote_rate(draws, outcome):
    voted = [d for d in draws if d.outcome == outcome and d.voted]
    if not voted:
        return -1.0
    return sum(1 for d in voted if d.vote_correct) / len(voted)


def _voted_rate(draws, outcome):
    sub = [d for d in draws if d.outcome == outcome]
    if not sub:
        return -1.0
    return sum(1 for d in sub if d.voted) / len(sub)


def episode_metrics(episodes, rewards, *, prefix: str = "train") -> dict:
    episodes, rewards = list(episodes), [float(r) for r in rewards]
    if len(episodes) != len(rewards):
        raise MetricsError(f"{len(episodes)} episodes but {len(rewards)} rewards")
    if not episodes:
        raise MetricsError("no episodes")
    tokens = [e.tokens for e in episodes]
    return {
        f"{prefix}/reward_mean": _mean(rewards),
        f"{prefix}/reward_p10": _quantile(rewards, 0.10),
        f"{prefix}/reward_p90": _quantile(rewards, 0.90),
        f"{prefix}/accuracy": _mean(1.0 if e.correct else 0.0 for e in episodes),
        f"{prefix}/tokens_per_episode": _mean(tokens),
        f"{prefix}/tokens_p90": _quantile(tokens, 0.90),
        f"{prefix}/saved_mean": _mean(e.saved for e in episodes),
        f"{prefix}/draws_per_episode": _mean(e.drawn for e in episodes),
        f"{prefix}/answered_per_episode": _mean(e.answered for e in episodes),
        f"{prefix}/abandoned_per_episode": _mean(e.abandoned for e in episodes),
        f"{prefix}/completed_per_episode": _mean(e.completed for e in episodes),
        f"{prefix}/cut_per_episode": _mean(e.cut for e in episodes),
        f"{prefix}/finalized_fraction": _mean(1.0 if e.finalized else 0.0 for e in episodes),
        f"{prefix}/exhausted_fraction": _mean(1.0 if e.exhausted else 0.0 for e in episodes),
        f"{prefix}/episodes": len(episodes),
    }


def group_metrics(groups: Mapping) -> dict:
    if not groups:
        raise MetricsError("no groups")
    all_correct = all_wrong = mixed = 0
    effective = 0
    for rewards in groups.values():
        rewards = [float(r) for r in rewards]
        if not rewards:
            raise MetricsError("a group has no rewards")
        positives = sum(1 for r in rewards if r > 0)
        if positives == len(rewards):
            all_correct += 1
        elif positives == 0:
            all_wrong += 1
        else:
            mixed += 1
        if any(abs(a) > 1e-12 for a in dr_grpo_advantages(rewards)):
            effective += 1
    n = len(groups)
    return {
        "groups/total": n,
        "groups/all_correct": all_correct / n,
        "groups/all_wrong": all_wrong / n,
        "groups/mixed": mixed / n,
        "groups/effective_fraction": effective / n,
        "groups/dropped": n - effective,
    }


def advantage_metrics(groups: Mapping) -> dict:
    advantages = []
    for rewards in groups.values():
        advantages.extend(dr_grpo_advantages([float(r) for r in rewards]))
    if not advantages:
        raise MetricsError("no advantages")
    return {
        "advantage/abs_mean": _mean(abs(a) for a in advantages),
        "advantage/p90": _quantile(advantages, 0.90),
        "advantage/p10": _quantile(advantages, 0.10),
        "advantage/nonzero_fraction": _mean(1.0 if abs(a) > 1e-12 else 0.0
                                            for a in advantages),
    }


def head_metrics(*, anchor_coef: float, delta_w_norm: float, drift_report: Mapping,
                 aux_loss: float | None = None) -> dict:
    out = {
        "anchor/coef": float(anchor_coef),
        "head/delta_w_norm": float(delta_w_norm),
        "drift/rank_corr": float(drift_report["rank_correlation"]),
        "drift/median_current": float(drift_report["median_current"]),
        "drift/median_pretrained": float(drift_report["median_pretrained"]),
        "drift/ece_current": float(drift_report["ece_current"]),
        "drift/ece_pretrained": float(drift_report["ece_pretrained"]),
        "drift/auroc_current": float(drift_report["auroc_current"]),
        "drift/auroc_pretrained": float(drift_report["auroc_pretrained"]),
    }
    if aux_loss is not None:
        out["head/aux_ready_loss"] = float(aux_loss)
    return out


def eval_metrics(*, split: str, sampled: Mapping, argmax: Mapping,
                 baselines: Mapping) -> dict:
    if split in FORBIDDEN_EVAL_SPLITS:
        raise MetricsError(
            f"split {split!r} is not loggable: the report quarter is read once, "
            "at the end, by a script that is not this one. Logging it here would "
            "make it visible, and visible means selected on"
        )
    required = {"sc", "fixed_cap", "oracle"}
    missing = required - set(baselines)
    if missing:
        raise MetricsError(
            f"eval on {split!r} is missing baselines {sorted(missing)}; a "
            "controller number without its floor, its rival and its ceiling on "
            "the same axis is the comparison this line keeps getting wrong"
        )
    out = {
        f"eval/{split}/accuracy_sampled": float(sampled["accuracy"]),
        f"eval/{split}/tokens_sampled": float(sampled["tokens"]),
        f"eval/{split}/accuracy_argmax": float(argmax["accuracy"]),
        f"eval/{split}/tokens_argmax": float(argmax["tokens"]),
        f"eval/{split}/argmax_minus_sampled": float(argmax["accuracy"]) - float(sampled["accuracy"]),
    }
    for name, value in baselines.items():
        out[f"eval/{split}/{name}_accuracy"] = float(value["accuracy"])
        out[f"eval/{split}/{name}_tokens"] = float(value["tokens"])
    out[f"eval/{split}/delta_vs_sc_sampled"] = (
        float(sampled["accuracy"]) - float(baselines["sc"]["accuracy"])
    )
    out[f"eval/{split}/delta_vs_fixed_cap_sampled"] = (
        float(sampled["accuracy"]) - float(baselines["fixed_cap"]["accuracy"])
    )
    headroom = float(baselines["oracle"]["accuracy"]) - float(baselines["sc"]["accuracy"])
    if headroom > 1e-9:
        out[f"eval/{split}/headroom_captured_sampled"] = (
            float(sampled["accuracy"]) - float(baselines["sc"]["accuracy"])
        ) / headroom
    return out


class WandbSink:

    def __init__(self, *, entity: str, project: str, run_name: str,
                 config: Mapping | None = None, mode: str = "online"):
        if not entity or not project or not run_name:
            raise MetricsError("wandb entity, project and run_name are all required")
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            entity=entity, project=project, name=run_name,
            config=dict(config or {}), mode=mode,
        )

    def log(self, metrics: Mapping, *, step: int) -> None:
        for key, value in metrics.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise MetricsError(f"non-finite metric {key} = {value!r} at step {step}")
        self._wandb.log(dict(metrics), step=step)

    def finish(self) -> None:
        self._run.finish()
