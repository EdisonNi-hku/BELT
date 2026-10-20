from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass


ACTION_LO = 2
ACTION_HI = 128


@dataclass(frozen=True)
class Trace:

    original_index: int
    sample_idx: int
    answer: str | None
    correct: bool
    completion_tokens: int
    block_end_offsets: tuple[int, ...]
    belief_blocks: tuple[int, ...]
    beliefs: tuple[float, ...]
    banked: dict = None

    def __post_init__(self) -> None:
        if self.banked is None:
            object.__setattr__(self, "banked", {})
        if len(self.belief_blocks) != len(self.beliefs):
            raise ValueError("belief_blocks and beliefs must align")
        if list(self.belief_blocks) != sorted(set(self.belief_blocks)):
            raise ValueError("belief_blocks must be strictly increasing")
        if any(not ACTION_LO <= b <= ACTION_HI for b in self.belief_blocks):
            raise ValueError(
                f"belief_blocks must lie in [{ACTION_LO}, {ACTION_HI}]"
            )
        if any(not 0.0 <= p <= 1.0 or not math.isfinite(p) for p in self.beliefs):
            raise ValueError("beliefs must be finite probabilities")
        offsets = list(self.block_end_offsets)
        if offsets != sorted(offsets) or (offsets and offsets[0] < 0):
            raise ValueError("block_end_offsets must be nondecreasing and nonnegative")
        if offsets and offsets[-1] > self.completion_tokens:
            raise ValueError(
                "a prefix may not cost more than the whole completion "
                f"({offsets[-1]} > {self.completion_tokens})"
            )
        if self.answer is None and self.correct:
            raise ValueError("an unparseable trace cannot be graded correct")

    @property
    def n_blocks(self) -> int:
        return len(self.block_end_offsets)

    def prefix_tokens(self, block: int) -> int:
        if not 1 <= block <= self.n_blocks:
            raise ValueError(f"block {block} outside 1..{self.n_blocks}")
        return int(self.block_end_offsets[block - 1])

    def belief_at(self, block: int) -> float | None:
        try:
            position = self.belief_blocks.index(block)
        except ValueError:
            return None
        return float(self.beliefs[position])

    def banked_at(self, block: int):
        return self.banked.get(block)

    @property
    def vote_key(self) -> tuple[str, int]:
        return vote_key_for(self.answer, self.sample_idx)


def vote_key_for(answer, sample_idx: int) -> tuple[str, int]:
    if answer is None:
        return ("\x00unparseable", sample_idx)
    return (str(answer), -1)


@dataclass(frozen=True)
class Vote:

    key: tuple
    correct: bool
    weight: float
    sample_idx: int


def weighted_majority(votes) -> bool:
    if not votes:
        return False
    mass: dict = {}
    lowest: dict = {}
    correct_of: dict = {}
    for vote in votes:
        mass[vote.key] = mass.get(vote.key, 0.0) + vote.weight
        lowest[vote.key] = min(lowest.get(vote.key, vote.sample_idx), vote.sample_idx)
        correct_of[vote.key] = vote.correct
    best = min(mass, key=lambda key: (-mass[key], lowest[key]))
    return bool(correct_of[best])


def load_traces(path: str) -> list[Trace]:
    traces = []
    seen = set()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        key = (int(row["original_index"]), int(row["sample_idx"]))
        if key in seen:
            raise ValueError(f"duplicate trace {key} in {path}")
        seen.add(key)
        answer = row.get("answer")
        traces.append(Trace(
            original_index=key[0],
            sample_idx=key[1],
            answer=None if answer is None else str(answer),
            correct=bool(row["correct"]),
            completion_tokens=int(row["completion_tokens"]),
            block_end_offsets=tuple(int(v) for v in row["block_end_offsets"]),
            belief_blocks=tuple(int(v) for v in row["belief_blocks"]),
            beliefs=tuple(float(v) for v in row["beliefs"]),
            banked={
                int(block): {
                    "answer": None if v.get("answer") is None else str(v["answer"]),
                    "correct": bool(v["correct"]),
                    "tokens": int(v["tokens"]),
                }
                for block, v in (row.get("banked") or {}).items()
            },
        ))
    if not traces:
        raise ValueError(f"no traces in {path}")
    return traces


def _usable_blocks(trace: Trace) -> range:
    if not trace.belief_blocks:
        return range(0)
    top = min(ACTION_HI, trace.n_blocks - 1)
    return range(ACTION_LO, top + 1)


def never_rule(trace: Trace) -> int | None:
    return None


def belief_rule(tau: float):
    def rule(trace: Trace) -> int | None:
        for block in _usable_blocks(trace):
            p = trace.belief_at(block)
            if p is not None and p >= tau:
                return block
        return None
    return rule


def length_rule(limit: int):
    def rule(trace: Trace) -> int | None:
        blocks = _usable_blocks(trace)
        return limit if limit in blocks else None
    return rule


def abort_rate(traces, rule) -> float:
    if not traces:
        raise ValueError("abort_rate needs at least one trace")
    fired = sum(1 for trace in traces if rule(trace) is not None)
    return fired / len(traces)


MATCH_TOLERANCE = 0.01


def match_length_rule(traces, target_rate: float) -> tuple[int, float]:
    best = None
    for limit in range(ACTION_LO, ACTION_HI + 1):
        rate = abort_rate(traces, length_rule(limit))
        key = (abs(rate - target_rate), -limit)
        if best is None or key < best[0]:
            best = (key, limit, rate)
    return best[1], best[2]


def matched_within_tolerance(target_rate: float, achieved_rate: float) -> bool:
    return abs(achieved_rate - target_rate) <= MATCH_TOLERANCE


def renewal_rate(traces, rule) -> dict:
    tokens = 0
    completions = 0
    correct = 0
    aborted = 0
    for trace in traces:
        block = rule(trace)
        if block is None:
            tokens += trace.completion_tokens
            completions += 1
            correct += int(trace.correct)
        else:
            tokens += trace.prefix_tokens(block)
            aborted += 1
    if completions == 0:
        raise ValueError("rule leaves no completion: the rate is undefined")
    if tokens == 0:
        raise ValueError("zero tokens spent")
    return {
        "tokens": tokens,
        "completions": completions,
        "correct": correct,
        "abort_rate": aborted / len(traces),
        "accuracy_of_completions": correct / completions,
        "correct_per_token": correct / tokens,
    }


@dataclass(frozen=True)
class Episode:

    correct: bool
    tokens: int
    drawn: int
    completed: int
    aborted: int
    exhausted: bool


def group_by_question(traces) -> dict:
    grouped: dict = {}
    for trace in traces:
        grouped.setdefault(trace.original_index, []).append(trace)
    return {
        question: sorted(items, key=lambda t: t.sample_idx)
        for question, items in grouped.items()
    }


def _majority_correct(votes) -> bool:
    if not votes:
        return False
    mass: dict = {}
    lowest: dict = {}
    correct_of: dict = {}
    for trace in votes:
        key = trace.vote_key
        mass[key] = mass.get(key, 0) + 1
        lowest[key] = min(lowest.get(key, trace.sample_idx), trace.sample_idx)
        correct_of[key] = trace.correct
    best = min(mass, key=lambda key: (-mass[key], lowest[key]))
    return bool(correct_of[best])


def run_episode(pool, order, *, budget: int, rule, weight: float) -> Episode:
    spent = 0
    drawn = 0
    truncated = 0
    votes = []
    exhausted = True
    for index in order:
        if spent >= budget:
            exhausted = False
            break
        trace = pool[index]
        drawn += 1
        block = rule(trace)
        if block is not None:
            banked = trace.banked_at(block)
            if banked is None:
                raise ValueError(
                    f"trace {trace.original_index},{trace.sample_idx} fires at "
                    f"block {block} but has no banked answer there; run the "
                    "forced-answer pass for this rule's firing blocks"
                )
            cost = trace.prefix_tokens(block) + int(banked["tokens"])
            if spent + cost > budget:
                spent = budget
                exhausted = False
                break
            spent += cost
            truncated += 1
            votes.append(Vote(
                key=vote_key_for(banked["answer"], trace.sample_idx),
                correct=bool(banked["correct"]),
                weight=weight,
                sample_idx=trace.sample_idx,
            ))
            continue
        if spent + trace.completion_tokens > budget:
            spent = budget
            exhausted = False
            break
        spent += trace.completion_tokens
        votes.append(Vote(
            key=trace.vote_key,
            correct=trace.correct,
            weight=1.0,
            sample_idx=trace.sample_idx,
        ))
    return Episode(
        correct=weighted_majority(votes),
        tokens=spent,
        drawn=drawn,
        completed=len(votes) - truncated,
        aborted=truncated,
        exhausted=exhausted and spent < budget,
    )


N_ORDERS = 200
BUDGET_MULTIPLES = (1.0, 2.0, 3.0, 4.0)


def draw_orders(grouped, n_orders: int, seed: int) -> dict:
    if type(n_orders) is not int or n_orders < 1:
        raise ValueError("n_orders must be a positive integer")
    rng = random.Random(seed)
    orders = {}
    for question in sorted(grouped):
        size = len(grouped[question])
        per_question = []
        for _ in range(n_orders):
            order = list(range(size))
            rng.shuffle(order)
            per_question.append(order)
        orders[question] = per_question
    return orders


def evaluate_arm(grouped, orders, *, budget: int, rule, weight: float) -> dict:
    per_question = {}
    exhausted_questions = set()
    exhausted_episodes = 0
    total_episodes = 0
    for question in sorted(grouped):
        pool = grouped[question]
        episodes = [
            run_episode(pool, order, budget=budget, rule=rule, weight=weight)
            for order in orders[question]
        ]
        total_episodes += len(episodes)
        if any(episode.exhausted for episode in episodes):
            exhausted_questions.add(question)
        exhausted_episodes += sum(1 for e in episodes if e.exhausted)
        per_question[question] = {
            "accuracy": statistics.fmean(float(e.correct) for e in episodes),
            "tokens": statistics.fmean(float(e.tokens) for e in episodes),
            "completed": statistics.fmean(float(e.completed) for e in episodes),
            "aborted": statistics.fmean(float(e.aborted) for e in episodes),
        }
    return {
        "per_question": per_question,
        "exhausted_questions": sorted(exhausted_questions),
        "exhausted_rate": exhausted_episodes / max(1, total_episodes),
    }


def drop_exhausted(arms) -> list:
    dropped = set()
    for arm in arms:
        dropped |= set(arm["exhausted_questions"])
    out = []
    for arm in arms:
        kept = {q: v for q, v in arm["per_question"].items() if q not in dropped}
        out.append(dict(arm, per_question=kept, dropped_questions=sorted(dropped)))
    return out


def paired_bootstrap(a_per_question, b_per_question, *, n_boot=2000, seed=0) -> dict:
    questions = sorted(a_per_question)
    if questions != sorted(b_per_question):
        raise ValueError("both arms must cover the same questions")
    if not questions:
        raise ValueError("no questions to compare")
    rng = random.Random(seed)

    def stat(sample):
        acc = statistics.fmean(
            a_per_question[q]["accuracy"] - b_per_question[q]["accuracy"]
            for q in sample
        )
        a_tok = sum(a_per_question[q]["tokens"] for q in sample)
        b_tok = sum(b_per_question[q]["tokens"] for q in sample)
        return acc, (a_tok - b_tok) / b_tok if b_tok else 0.0

    point_acc, point_tok = stat(questions)
    accs, toks = [], []
    for _ in range(n_boot):
        sample = [questions[rng.randrange(len(questions))] for _ in questions]
        acc, tok = stat(sample)
        accs.append(acc)
        toks.append(tok)
    accs.sort()
    toks.sort()

    def summarise(point, draws):
        lo = draws[int(0.025 * len(draws))]
        hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
        worse = sum(1 for value in draws if value <= 0.0)
        p = max((worse + 1) / (len(draws) + 1), 1.0 / (len(draws) + 1))
        return {
            "point": point,
            "ci": [lo, hi],
            "p_one_sided": p,
            "spans_zero": lo <= 0.0 <= hi,
        }

    return {
        "n_questions": len(questions),
        "n_boot": n_boot,
        "d_accuracy": summarise(point_acc, accs),
        "d_tokens_relative": summarise(point_tok, toks),
    }


def holm(p_values, *, alpha=0.05) -> list:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    flags = [False] * len(p_values)
    m = len(p_values)
    for rank, index in enumerate(order):
        if p_values[index] <= alpha / (m - rank):
            flags[index] = True
        else:
            break
    return flags


def fit_banked_weight(traces, rule) -> dict:
    banked_correct = []
    completed_correct = []
    for trace in traces:
        block = rule(trace)
        if block is None:
            completed_correct.append(float(trace.correct))
            continue
        entry = trace.banked_at(block)
        if entry is not None:
            banked_correct.append(float(entry["correct"]))
    if not banked_correct:
        raise ValueError("no banked answers at this rule's firing blocks")
    if not completed_correct:
        raise ValueError("no completions left to normalise against")
    q_banked = statistics.fmean(banked_correct)
    q_complete = statistics.fmean(completed_correct)
    if q_complete <= 0:
        raise ValueError("completion accuracy is zero; w is undefined")
    return {
        "weight": q_banked / q_complete,
        "banked_accuracy": q_banked,
        "completion_accuracy": q_complete,
        "n_banked": len(banked_correct),
        "n_completed": len(completed_correct),
    }


def banked_renewal_rate(traces, rule) -> dict:
    tokens = 0
    answers = 0
    correct = 0
    banked = 0
    for trace in traces:
        block = rule(trace)
        entry = None if block is None else trace.banked_at(block)
        if block is not None and entry is None:
            raise ValueError(
                f"trace {trace.original_index},{trace.sample_idx} fires at "
                f"block {block} with no banked answer; this rate is only "
                "defined once the forced-answer pass has run"
            )
        if block is None:
            tokens += trace.completion_tokens
            correct += int(trace.correct)
        else:
            tokens += trace.prefix_tokens(block) + int(entry["tokens"])
            correct += int(entry["correct"])
            banked += 1
        answers += 1
    if tokens == 0:
        raise ValueError("zero tokens spent")
    return {
        "tokens": tokens,
        "answers": answers,
        "banked": banked,
        "correct": correct,
        "accuracy_of_answers": correct / answers,
        "correct_per_token": correct / tokens,
    }


def has_banked(traces) -> bool:
    return any(trace.banked for trace in traces)


def best_threshold(traces, thresholds, *, match_against=None) -> tuple[float, float]:
    rate_of = banked_renewal_rate if has_banked(traces) else renewal_rate
    best = None
    skipped_unmatchable = []
    for tau in thresholds:
        if match_against is not None:
            fire = abort_rate(match_against, belief_rule(tau))
            _, achieved = match_length_rule(match_against, fire)
            if not matched_within_tolerance(fire, achieved):
                skipped_unmatchable.append(tau)
                continue
        try:
            rate = rate_of(traces, belief_rule(tau))["correct_per_token"]
        except ValueError:
            continue
        if best is None or rate > best[1]:
            best = (tau, rate)
    if best is None:
        if len(skipped_unmatchable) == len(thresholds):
            raise ValueError(
                "no matchable threshold: the length arm cannot reach any of "
                f"{list(thresholds)} inside the action window"
            )
        raise ValueError("no usable threshold: every one abandons every trace")
    return best


def remaining_table(traces) -> dict:
    totals: dict = {}
    counts: dict = {}
    for trace in traces:
        for block in _usable_blocks(trace):
            remaining = trace.completion_tokens - trace.prefix_tokens(block)
            totals[block] = totals.get(block, 0) + max(0, remaining)
            counts[block] = counts.get(block, 0) + 1
    return {block: totals[block] / counts[block] for block in sorted(totals)}


def base_rate_table(traces) -> dict:
    wrong: dict = {}
    counts: dict = {}
    for trace in traces:
        for block in _usable_blocks(trace):
            wrong[block] = wrong.get(block, 0) + int(not trace.correct)
            counts[block] = counts.get(block, 0) + 1
    return {block: wrong[block] / counts[block] for block in sorted(wrong)}


def index_rule(remaining, value: float, base_rates=None):
    def rule(trace: Trace) -> int | None:
        for block in _usable_blocks(trace):
            r = remaining.get(block)
            if not r:
                continue
            p = base_rates.get(block) if base_rates is not None \
                else trace.belief_at(block)
            if p is None:
                continue
            if (1.0 - p) / r < value:
                return block
        return None
    return rule


def solve_value(traces, remaining, *, base_rates=None, max_iterations=10,
                tolerance=1e-4) -> dict:
    value = renewal_rate(traces, never_rule)["correct_per_token"]
    history = [value]
    converged = False
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        rule = index_rule(remaining, value, base_rates=base_rates)
        try:
            nxt = renewal_rate(traces, rule)["correct_per_token"]
        except ValueError:
            break
        history.append(nxt)
        if abs(nxt - value) < tolerance:
            value = nxt
            converged = True
            break
        value = nxt
    return {
        "value": value,
        "converged": converged,
        "iterations": iterations,
        "history": history,
    }
