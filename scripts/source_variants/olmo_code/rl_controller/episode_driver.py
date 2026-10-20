from __future__ import annotations

from collections.abc import Callable, Sequence

from rl_controller.budget_episode import (
    BOUNDARY_ACTIONS,
    DRAW,
    FINALIZE,
    BoundaryStep,
    BudgetEpisode,
    DrawRecord,
    EpisodeError,
    Step,
    observe_boundary,
    vote_statistics,
)
from rl_controller.redraw import Vote, vote_key_for, weighted_majority
from rl_controller.rollout_runner import draw_steps


def _correct(answer, gold) -> bool:
    return answer is not None and gold is not None and int(answer) == int(gold)


class EpisodeMachine:

    def __init__(self, *, prompt_ids, gold, question, budget, policy, state_fn,
                 encode, boundary_policy=None, max_draws=None, max_blocks=128,
                 max_tokens_per_draw=32768, query_stride=1, block_cap=0,
                 seed_key=None):
        if budget <= 0:
            raise EpisodeError(f"budget {budget} must be positive")
        if max_draws is not None and max_draws < 1:
            raise EpisodeError(f"max_draws {max_draws} must be at least 1")
        self.prompt_ids = list(prompt_ids)
        self.gold = gold
        self.question = question
        self.budget = budget
        self.policy = policy
        self.state_fn = state_fn
        self.encode = encode
        self.boundary_policy = boundary_policy
        self.max_draws = max_draws
        self.max_blocks = max_blocks
        self.query_stride = query_stride
        self.block_cap = block_cap
        self.max_tokens_per_draw = max_tokens_per_draw
        self.seed_key = seed_key

        self.spent = 0
        self.votes: list = []
        self.steps: list = []
        self.boundary_steps: list = []
        self.draws: list = []
        self.answered = self.abandoned = self.completed = self.cut = 0
        self.finalized = False
        self.exhausted = True
        self.capped = False
        self.draw_index = -1
        self._closed = False
        self._spent_before_draw = 0

    def next_draw(self):
        if self._closed:
            return None
        self.draw_index += 1
        draw_index = self.draw_index
        if self.max_draws is not None and draw_index >= self.max_draws:
            self.capped = True
            self.exhausted = False
            self._closed = True
            return None
        if self.spent >= self.budget:
            self.exhausted = False
            self._closed = True
            return None
        if self.boundary_policy is not None and draw_index > 0:
            action = self.boundary_policy(observe_boundary(
                spent=self.spent, budget=self.budget, votes=self.votes,
                draws_started=draw_index,
            ))
            if action not in BOUNDARY_ACTIONS:
                raise EpisodeError(f"unknown boundary action {action!r}")
            self.boundary_steps.append(BoundaryStep(
                question=self.question, draw_index=draw_index, action=action,
                tokens_before=self.spent, banked_before=len(self.votes),
            ))
            if action == FINALIZE:
                self.finalized = True
                self.exhausted = False
                self._closed = True
                return None

        draws_started = draw_index + 1
        self._spent_before_draw = self.spent
        stats_before = vote_statistics(self.votes)
        spent = self.spent
        budget = self.budget
        prompt_ids = self.prompt_ids
        state_fn = self.state_fn

        def observe(block, completion_ids, block_ends, _spent=spent, _stats=stats_before,
                    _draws=draws_started):
            obs = {
                "_block_ends": list(block_ends),
                "block": block,
                "tokens_so_far": len(completion_ids),
                "mean_block_tokens": len(completion_ids) / block,
                "spent": _spent,
                "budget": budget,
                "budget_left": budget - _spent - len(completion_ids),
                "budget_fraction_left": max(0.0, 1.0 - (_spent + len(completion_ids)) / budget),
                "draws_started": _draws,
            }
            obs.update(_stats)
            obs.update(state_fn(prompt_ids, completion_ids, block, block_ends))
            return obs

        return draw_steps(
            prompt_ids=prompt_ids, policy=self.policy, observe=observe,
            encode=self.encode, max_blocks=self.max_blocks,
            max_total_tokens=self.max_tokens_per_draw, budget_left=budget - spent,
            query_stride=self.query_stride, block_cap=self.block_cap,
            request_seed_base=(None if self.seed_key is None
                               else (self.seed_key, draw_index)),
        )

    def absorb(self, result) -> None:
        draw_index = self.draw_index
        for block, action in result.decisions:
            self.steps.append(Step(
                question=self.question, sample_idx=draw_index, draw_index=draw_index,
                block=block, action=action, tokens_before=self._spent_before_draw,
                banked_before=len(self.votes),
            ))
        self.spent = min(self.budget, self.spent + result.tokens)
        is_cut = result.outcome == "length" and self.spent >= self.budget

        if result.outcome == "abandoned":
            self.abandoned += 1
            self.draws.append(DrawRecord(
                question=self.question, sample_idx=draw_index, draw_index=draw_index,
                outcome="abandoned", voted=False, vote_correct=None,
                tokens=result.tokens, end_block=result.end_block,
            ))
            return

        if is_cut and result.answer is None:
            self.cut += 1
            self.draws.append(DrawRecord(
                question=self.question, sample_idx=draw_index, draw_index=draw_index,
                outcome="cut", voted=False, vote_correct=None,
                tokens=result.tokens, end_block=result.end_block,
            ))
            self.exhausted = False
            self._closed = True
            return

        vote_correct = _correct(result.answer, self.gold)
        self.votes.append(Vote(
            key=vote_key_for(result.answer, draw_index), correct=vote_correct,
            weight=1.0, sample_idx=draw_index,
        ))
        if result.outcome == "answered":
            self.answered += 1
            outcome = "answered"
        elif is_cut:
            self.cut += 1
            outcome = "cut"
        else:
            self.completed += 1
            outcome = "completed"
        self.draws.append(DrawRecord(
            question=self.question, sample_idx=draw_index, draw_index=draw_index,
            outcome=outcome, voted=True, vote_correct=vote_correct,
            tokens=result.tokens, end_block=result.end_block,
        ))
        if is_cut:
            self.exhausted = False
            self._closed = True

    def finish(self) -> BudgetEpisode:
        return BudgetEpisode(
            correct=weighted_majority(self.votes),
            tokens=self.spent,
            budget=self.budget,
            drawn=len(self.draws),
            answered=self.answered,
            abandoned=self.abandoned,
            completed=self.completed,
            cut=self.cut,
            finalized=self.finalized,
            exhausted=self.exhausted and self.spent < self.budget and not self.capped,
            steps=tuple(self.steps),
            boundary_steps=tuple(self.boundary_steps),
            draws=tuple(self.draws),
        )


def run_online_episode(
    *,
    engine,
    prompt_ids: Sequence[int],
    gold: int,
    question: int,
    budget: int,
    policy: Callable,
    state_fn: Callable,
    encode: Callable,
    boundary_policy: Callable | None = None,
    max_draws: int | None = None,
    max_blocks: int = 128,
    max_tokens_per_draw: int = 32768,
    query_stride: int = 1,
    block_cap: int = 0,
    seed_key=None,
) -> BudgetEpisode:
    machine = EpisodeMachine(
        prompt_ids=prompt_ids, gold=gold, question=question, budget=budget,
        policy=policy, state_fn=state_fn, encode=encode,
        boundary_policy=boundary_policy, max_draws=max_draws,
        max_blocks=max_blocks, max_tokens_per_draw=max_tokens_per_draw,
        query_stride=query_stride, block_cap=block_cap, seed_key=seed_key,
    )
    while True:
        gen = machine.next_draw()
        if gen is None:
            return machine.finish()
        try:
            req = next(gen)
            while True:
                out = engine.generate(req.prompt_ids, stop=req.stop,
                                      include_stop_str_in_output=req.include_stop_str_in_output,
                                      max_tokens=req.max_tokens, seed_key=req.seed_key)
                req = gen.send(out)
        except StopIteration as done:
            machine.absorb(done.value)
