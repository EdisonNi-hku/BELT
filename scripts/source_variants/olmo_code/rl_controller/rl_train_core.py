from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rl_controller.budget_episode import (
    BudgetEpisode,
    boundary_advantages,
    episode_reward,
    group_carries_gradient,
    step_advantages,
)


class TrainError(RuntimeError):
    pass


@dataclass
class QuestionStat:

    seen: int = 0
    _rate: float | None = None

    @property
    def rate(self) -> float | None:
        return self._rate

    def update(self, episodes: Sequence[BudgetEpisode], *, ema: float) -> None:
        if not episodes:
            return
        batch = sum(1.0 for e in episodes if e.correct) / len(episodes)
        self._rate = batch if self._rate is None else (1 - ema) * self._rate + ema * batch
        self.seen += len(episodes)


BUCKETS = ("unseen", "hopeless", "mixed", "saturated")


def bucket_of(rate: float | None, *, lo: float = 0.15, hi: float = 0.85) -> str:
    if rate is None:
        return "unseen"
    if rate < lo:
        return "hopeless"
    if rate > hi:
        return "saturated"
    return "mixed"


class StratifiedSampler:

    def __init__(self, question_keys: Sequence, *, seed: int,
                 shares: Mapping[str, float] | None = None, ema: float = 0.3):
        keys = list(question_keys)
        if len(set(keys)) != len(keys):
            raise TrainError("duplicate question keys in the fit pool")
        if not keys:
            raise TrainError("empty fit pool")
        self.keys = keys
        self.stats = {k: QuestionStat() for k in keys}
        self.rng = random.Random(seed)
        self.shares = dict(shares or {"mixed": 0.6, "unseen": 0.2,
                                      "hopeless": 0.1, "saturated": 0.1})
        if abs(sum(self.shares.values()) - 1.0) > 1e-9 or set(self.shares) != set(BUCKETS):
            raise TrainError(f"shares must cover {BUCKETS} and sum to 1, got {self.shares}")
        self.ema = ema

    def buckets(self) -> dict[str, list]:
        out = {b: [] for b in BUCKETS}
        for k in self.keys:
            out[bucket_of(self.stats[k].rate)].append(k)
        return out

    def sample(self, batch_size: int) -> list:
        if batch_size < 1:
            raise TrainError("batch_size must be positive")
        if batch_size > len(self.keys):
            raise TrainError(
                f"batch_size {batch_size} exceeds the {len(self.keys)}-question pool"
            )
        pools = self.buckets()
        want = {b: int(round(self.shares[b] * batch_size)) for b in BUCKETS}
        chosen: list = []
        for b in BUCKETS:
            available = pools[b]
            take = min(want[b], len(available))
            chosen.extend(self.rng.sample(available, take))
        if len(chosen) < batch_size:
            remaining = [k for k in self.keys if k not in set(chosen)]
            chosen.extend(self.rng.sample(remaining, batch_size - len(chosen)))
        return chosen[:batch_size]

    def update(self, episodes_by_question: Mapping) -> None:
        for key, episodes in episodes_by_question.items():
            if key not in self.stats:
                raise TrainError(f"update for unknown question {key!r}")
            self.stats[key].update(episodes, ema=self.ema)

    def bucket_counts(self) -> dict[str, int]:
        return {b: len(v) for b, v in self.buckets().items()}

    def state_dict(self) -> dict:
        return {"stats": {str(k): {"seen": st.seen, "rate": st.rate} for k, st in self.stats.items()},
                "rng": self.rng.getstate(), "ema": self.ema, "shares": dict(self.shares)}

    def load_state_dict(self, state: Mapping) -> None:
        stats = state["stats"]
        missing = [k for k in self.keys if str(k) not in stats]
        if missing:
            raise TrainError(f"sampler state lacks {len(missing)} fit questions (e.g. {missing[0]!r}); "
                             "a resumed run must use the same fit pool")
        for k in self.keys:
            st = stats[str(k)]
            self.stats[k].seen = int(st["seen"])
            self.stats[k]._rate = st["rate"]
        rng_state = state["rng"]
        self.rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))


def ppo_clip_loss(logp_new: Sequence[float], logp_old: Sequence[float],
                  advantages: Sequence[float], *, clip: float = 0.2) -> float:
    if not (len(logp_new) == len(logp_old) == len(advantages)):
        raise TrainError("logp_new, logp_old and advantages must align")
    if not logp_new:
        raise TrainError("empty batch")
    if clip <= 0:
        raise TrainError("clip must be positive")
    total = 0.0
    for ln, lo, a in zip(logp_new, logp_old, advantages):
        ratio = math.exp(ln - lo)
        unclipped = ratio * a
        clipped = max(1.0 - clip, min(1.0 + clip, ratio)) * a
        total += -min(unclipped, clipped)
    return total / len(logp_new)


def entropy_of(probs: Sequence[float]) -> float:
    total = sum(probs)
    if total <= 0:
        raise TrainError("not a distribution")
    return -sum((p / total) * math.log(p / total) for p in probs if p > 0)


def entropy_bonus(distributions: Sequence[Sequence[float]]) -> float:
    if not distributions:
        raise TrainError("no distributions")
    return sum(entropy_of(d) for d in distributions) / len(distributions)


def stop_mass_penalty(mass, *, floor: float, multiplier: float):
    if floor <= 0:
        raise TrainError(f"floor must be positive, got {floor}")
    shortfall = floor - mass
    try:
        import torch
        if torch.is_tensor(shortfall):
            return multiplier * torch.clamp(shortfall, min=0.0) / floor
    except ImportError:
        pass
    return multiplier * max(0.0, float(shortfall)) / floor


def stop_mass_multiplier_step(multiplier: float, *, mass: float, floor: float, lr: float) -> float:
    if floor <= 0:
        raise TrainError(f"floor must be positive, got {floor}")
    rel = (floor - float(mass)) / floor
    return max(0.0, float(multiplier) + lr * rel)


@dataclass
class IterationBatch:

    inner_advantages: list = field(default_factory=list)
    inner_steps: list = field(default_factory=list)
    outer_advantages: list = field(default_factory=list)
    outer_steps: list = field(default_factory=list)
    groups_total: int = 0
    groups_dropped: int = 0
    dropped_questions: list = field(default_factory=list)


def assemble_batch(episodes_by_question: Mapping, *, lam: float, lam_d: float,
                   beta: float = 1.0, alpha: float = 0.5) -> IterationBatch:
    batch = IterationBatch()
    for key, episodes in episodes_by_question.items():
        batch.groups_total += 1
        rewards = [episode_reward(e, lam=lam) for e in episodes]
        if not group_carries_gradient(rewards):
            batch.groups_dropped += 1
            batch.dropped_questions.append(key)
            continue
        a_step = step_advantages(episodes, lam=lam, lam_d=lam_d, beta=beta, alpha=alpha)
        a_bound = boundary_advantages(episodes, lam=lam)
        for ei, (episode, advs, badvs) in enumerate(zip(episodes, a_step, a_bound)):
            for step, adv in zip(episode.steps, advs):
                batch.inner_steps.append((key, ei, step))
                batch.inner_advantages.append(adv)
            for bstep, adv in zip(episode.boundary_steps, badvs):
                batch.outer_steps.append((key, ei, bstep))
                batch.outer_advantages.append(adv)
    return batch


def should_run_guard(iteration: int, *, every: int) -> bool:
    if every < 1:
        raise TrainError("guard interval must be at least 1")
    return iteration == 0 or iteration % every == 0


def lambda_bite_pair(lam: float) -> tuple[float, float]:
    hi = min(1.0, lam * 2.0)
    lo = hi / 4.0
    return lo, hi
