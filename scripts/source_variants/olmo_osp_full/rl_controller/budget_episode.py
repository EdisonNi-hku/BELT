from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rl_controller.redraw import (
    ACTION_HI,
    ACTION_LO,
    Trace,
    Vote,
    vote_key_for,
    weighted_majority,
)

CONTINUE = "continue"
ANSWER = "answer"
ABANDON = "abandon"
INNER_ACTIONS = (CONTINUE, ANSWER, ABANDON)

DRAW = "draw"
FINALIZE = "finalize"
BOUNDARY_ACTIONS = (DRAW, FINALIZE)

ACTIONS = INNER_ACTIONS

VOTE_RIGHT, VOTE_NONE, VOTE_WRONG = 1, 0, -1

__all__ = [
    "ABANDON", "ACTIONS", "ANSWER", "BOUNDARY_ACTIONS", "CONTINUE", "DRAW",
    "FINALIZE", "INNER_ACTIONS", "VOTE_NONE", "VOTE_RIGHT", "VOTE_WRONG",
    "BudgetEpisode", "DrawRecord", "EpisodeError", "Step", "Trace",
    "draw_advantages", "dr_grpo_advantages", "episode_advantages",
    "episode_reward", "group_carries_gradient", "lambda_bites",
    "observe", "observe_boundary", "run_episode", "step_advantages",
    "vote_statistics",
]


class EpisodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Step:

    question: int
    sample_idx: int
    draw_index: int
    block: int
    action: str
    tokens_before: int
    banked_before: int


@dataclass(frozen=True)
class BoundaryStep:

    question: int
    draw_index: int
    action: str
    tokens_before: int
    banked_before: int


@dataclass(frozen=True)
class DrawRecord:

    question: int
    sample_idx: int
    draw_index: int
    outcome: str
    voted: bool
    vote_correct: bool | None
    tokens: int
    end_block: int | None

    @property
    def vote_value(self) -> int:
        if not self.voted:
            return VOTE_NONE
        return VOTE_RIGHT if self.vote_correct else VOTE_WRONG


@dataclass(frozen=True)
class BudgetEpisode:
    correct: bool
    tokens: int
    budget: int
    drawn: int
    answered: int
    abandoned: int
    completed: int
    cut: int
    finalized: bool
    exhausted: bool
    steps: tuple = field(default_factory=tuple)
    boundary_steps: tuple = field(default_factory=tuple)
    draws: tuple = field(default_factory=tuple)

    @property
    def saved(self) -> float:
        if self.budget <= 0:
            raise EpisodeError(f"budget {self.budget} must be positive")
        return max(0.0, 1.0 - self.tokens / self.budget)


def vote_statistics(votes: Sequence[Vote]) -> dict:
    if not votes:
        return {
            "n_banked": 0,
            "top1_share": 0.0,
            "n_distinct_answers": 0,
            "vote_margin": 0.0,
        }
    mass: dict = {}
    for vote in votes:
        mass[vote.key] = mass.get(vote.key, 0.0) + float(vote.weight)
    total = sum(mass.values())
    ordered = sorted(mass.values(), reverse=True)
    top1 = ordered[0] / total
    top2 = ordered[1] / total if len(ordered) > 1 else 0.0
    return {
        "n_banked": len(votes),
        "top1_share": top1,
        "n_distinct_answers": len(mass),
        "vote_margin": top1 - top2,
    }


def observe(trace: Trace, block: int, *, spent: int, budget: int,
            votes: Sequence[Vote], draws_started: int) -> dict:
    if budget <= 0:
        raise EpisodeError(f"budget {budget} must be positive")
    out = {
        "block": block,
        "tokens_so_far": trace.prefix_tokens(block),
        "mean_block_tokens": trace.prefix_tokens(block) / block,
        "belief": trace.belief_at(block),
        "spent": spent,
        "budget": budget,
        "budget_left": budget - spent,
        "budget_fraction_left": max(0.0, 1.0 - spent / budget),
        "draws_started": draws_started,
    }
    out.update(vote_statistics(votes))
    return out


def observe_boundary(*, spent: int, budget: int, votes: Sequence[Vote],
                     draws_started: int) -> dict:
    if budget <= 0:
        raise EpisodeError(f"budget {budget} must be positive")
    out = {
        "spent": spent,
        "budget": budget,
        "budget_left": budget - spent,
        "budget_fraction_left": max(0.0, 1.0 - spent / budget),
        "draws_started": draws_started,
    }
    out.update(vote_statistics(votes))
    return out


def _usable_blocks(trace: Trace) -> range:
    top = min(ACTION_HI, trace.n_blocks - 1)
    return range(ACTION_LO, top + 1)


def _deepest_banked_at_or_before(trace: Trace, block: int):
    for b in range(block, 0, -1):
        banked = trace.banked_at(b)
        if banked is not None:
            return b, banked
    return None, None


def run_episode(pool, order, *, budget: int, policy, boundary_policy=None,
                truncated_weight: float = 1.0) -> BudgetEpisode:
    if budget <= 0:
        raise EpisodeError(f"budget {budget} must be positive")
    spent = 0
    drawn = answered = abandoned = completed = cut = 0
    votes: list = []
    steps: list = []
    boundary_steps: list = []
    draws: list = []
    finalized = False
    exhausted = True

    for draw_index, index in enumerate(order):
        if spent >= budget:
            exhausted = False
            break
        if boundary_policy is not None and draw_index > 0:
            action = boundary_policy(observe_boundary(
                spent=spent, budget=budget, votes=votes, draws_started=drawn,
            ))
            if action not in BOUNDARY_ACTIONS:
                raise EpisodeError(f"unknown boundary action {action!r}")
            boundary_steps.append(BoundaryStep(
                question=pool[index].original_index, draw_index=draw_index,
                action=action, tokens_before=spent, banked_before=len(votes),
            ))
            if action == FINALIZE:
                finalized = True
                exhausted = False
                break

        trace = pool[index]
        drawn += 1
        tokens_before_draw = spent
        decided = False

        for block in _usable_blocks(trace):
            action = policy(observe(
                trace, block, spent=spent, budget=budget,
                votes=votes, draws_started=drawn,
            ))
            if action not in INNER_ACTIONS:
                raise EpisodeError(f"unknown action {action!r}")
            steps.append(Step(
                question=trace.original_index, sample_idx=trace.sample_idx,
                draw_index=draw_index, block=block, action=action,
                tokens_before=spent, banked_before=len(votes),
            ))
            if action == CONTINUE:
                continue

            if action == ABANDON:
                cost = trace.prefix_tokens(block)
                if spent + cost > budget:
                    spent = budget
                    exhausted = False
                    cut += 1
                    draws.append(_cut_draw(trace, draw_index, block, votes,
                                           tokens_before_draw, spent,
                                           truncated_weight))
                    decided = True
                    break
                spent += cost
                abandoned += 1
                draws.append(DrawRecord(
                    question=trace.original_index, sample_idx=trace.sample_idx,
                    draw_index=draw_index, outcome="abandoned", voted=False,
                    vote_correct=None, tokens=cost, end_block=block,
                ))
                decided = True
                break

            banked = trace.banked_at(block)
            if banked is None:
                raise EpisodeError(
                    f"trace {trace.original_index},{trace.sample_idx} answers at "
                    f"block {block} but has no banked answer there; run the "
                    "forced-answer pass for the blocks this policy fires at"
                )
            cost = trace.prefix_tokens(block) + int(banked["tokens"])
            if spent + cost > budget:
                spent = budget
                exhausted = False
                cut += 1
                draws.append(_cut_draw(trace, draw_index, block, votes,
                                       tokens_before_draw, spent,
                                       truncated_weight))
                decided = True
                break
            spent += cost
            answered += 1
            votes.append(Vote(
                key=vote_key_for(banked["answer"], trace.sample_idx),
                correct=bool(banked["correct"]),
                weight=truncated_weight,
                sample_idx=trace.sample_idx,
            ))
            draws.append(DrawRecord(
                question=trace.original_index, sample_idx=trace.sample_idx,
                draw_index=draw_index, outcome="answered", voted=True,
                vote_correct=bool(banked["correct"]), tokens=cost,
                end_block=block,
            ))
            decided = True
            break

        if decided:
            if spent >= budget:
                exhausted = False
                break
            continue

        if spent + trace.completion_tokens > budget:
            spent = budget
            exhausted = False
            cut += 1
            last_block = _usable_blocks(trace)[-1] if _usable_blocks(trace) else 1
            draws.append(_cut_draw(trace, draw_index, last_block, votes,
                                   tokens_before_draw, spent, truncated_weight))
            break
        spent += trace.completion_tokens
        completed += 1
        votes.append(Vote(
            key=trace.vote_key, correct=trace.correct,
            weight=1.0, sample_idx=trace.sample_idx,
        ))
        draws.append(DrawRecord(
            question=trace.original_index, sample_idx=trace.sample_idx,
            draw_index=draw_index, outcome="completed", voted=True,
            vote_correct=bool(trace.correct), tokens=trace.completion_tokens,
            end_block=None,
        ))

    return BudgetEpisode(
        correct=weighted_majority(votes),
        tokens=spent,
        budget=budget,
        drawn=drawn,
        answered=answered,
        abandoned=abandoned,
        completed=completed,
        cut=cut,
        finalized=finalized,
        exhausted=exhausted and spent < budget,
        steps=tuple(steps),
        boundary_steps=tuple(boundary_steps),
        draws=tuple(draws),
    )


def _cut_draw(trace, draw_index, block, votes, tokens_before_draw, spent, weight):
    used_block, banked = _deepest_banked_at_or_before(trace, block)
    if banked is None:
        return DrawRecord(
            question=trace.original_index, sample_idx=trace.sample_idx,
            draw_index=draw_index, outcome="cut", voted=False,
            vote_correct=None, tokens=spent - tokens_before_draw,
            end_block=block,
        )
    votes.append(Vote(
        key=vote_key_for(banked["answer"], trace.sample_idx),
        correct=bool(banked["correct"]),
        weight=weight,
        sample_idx=trace.sample_idx,
    ))
    return DrawRecord(
        question=trace.original_index, sample_idx=trace.sample_idx,
        draw_index=draw_index, outcome="cut", voted=True,
        vote_correct=bool(banked["correct"]), tokens=spent - tokens_before_draw,
        end_block=used_block,
    )


def episode_reward(episode: BudgetEpisode, *, lam: float) -> float:
    if lam < 0:
        raise ValueError(f"lam {lam} must be non-negative")
    return (1.0 + lam * episode.saved) if episode.correct else 0.0


def dr_grpo_advantages(rewards) -> list[float]:
    rewards = [float(r) for r in rewards]
    if not rewards:
        raise ValueError("no rewards in group")
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]


def episode_advantages(episodes: Sequence[BudgetEpisode], *, lam: float) -> list[float]:
    return dr_grpo_advantages(episode_reward(e, lam=lam) for e in episodes)


def _cost_quantile(tokens: int, all_tokens: Sequence[int]) -> float:
    below = sum(1 for t in all_tokens if t < tokens)
    equal = sum(1 for t in all_tokens if t == tokens)
    return (below + 0.5 * (equal - 1)) / max(1, len(all_tokens) - 1) \
        if len(all_tokens) > 1 else 0.5


def draw_advantages(episodes: Sequence[BudgetEpisode], *, lam_d: float) -> list[list[float]]:
    if lam_d < 0:
        raise ValueError(f"lam_d {lam_d} must be non-negative")
    all_draws = [d for e in episodes for d in e.draws]
    if not all_draws:
        raise ValueError("no draws in the group")
    all_tokens = [d.tokens for d in all_draws]

    def reward(draw: DrawRecord) -> float:
        value = draw.vote_value
        if value == VOTE_RIGHT:
            saved = 1.0 - _cost_quantile(draw.tokens, all_tokens)
            return 1.0 + lam_d * saved
        return float(value)

    flat = dr_grpo_advantages(reward(d) for d in all_draws)
    out, cursor = [], 0
    for episode in episodes:
        n = len(episode.draws)
        out.append(flat[cursor:cursor + n])
        cursor += n
    return out


def step_advantages(episodes: Sequence[BudgetEpisode], *, lam: float,
                    lam_d: float, beta: float = 1.0,
                    alpha: float = 0.5) -> list[list[float]]:
    if beta < 0.0:
        raise ValueError(f"beta must be >= 0, got {beta}")
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    a_draw = draw_advantages(episodes, lam_d=lam_d)
    a_ep = episode_advantages(episodes, lam=lam)
    out = []
    for episode, draw_advs, ep_adv in zip(episodes, a_draw, a_ep):
        by_draw = dict(zip((d.draw_index for d in episode.draws), draw_advs))
        steps_per_draw: dict = {}
        for step in episode.steps:
            steps_per_draw[step.draw_index] = steps_per_draw.get(step.draw_index, 0) + 1
        n_ep = len(episode.steps)
        n_draws = max(1, len(episode.draws))
        per_step = []
        for step in episode.steps:
            if step.draw_index not in by_draw:
                raise EpisodeError(
                    f"step at draw {step.draw_index} has no draw record; the "
                    "episode was not closed properly"
                )
            per_step.append(alpha * by_draw[step.draw_index] / (n_draws * steps_per_draw[step.draw_index])
                            + beta * ep_adv / n_ep)
        out.append(per_step)
    return out


def boundary_advantages(episodes: Sequence[BudgetEpisode], *, lam: float) -> list[list[float]]:
    a_ep = episode_advantages(episodes, lam=lam)
    return [[adv] * len(e.boundary_steps) for e, adv in zip(episodes, a_ep)]


def group_carries_gradient(rewards, *, tolerance: float = 1e-12) -> bool:
    advantages = dr_grpo_advantages(rewards)
    return any(abs(a) > tolerance for a in advantages)


LAMBDA_BITE_MIN_RELATIVE_CHANGE = 0.10


def lambda_bites(spend_low: float, spend_high: float, *,
                 min_relative_change: float = LAMBDA_BITE_MIN_RELATIVE_CHANGE) -> bool:
    if spend_low <= 0 or spend_high <= 0:
        raise ValueError("token spends must be positive")
    change = abs(spend_high - spend_low) / max(spend_low, spend_high)
    return change >= min_relative_change
