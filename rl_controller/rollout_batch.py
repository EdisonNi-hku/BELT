from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rl_controller.budget_episode import BudgetEpisode
from rl_controller.episode_driver import EpisodeMachine
from rl_controller.rollout_runner import GenRequest


@dataclass
class _Live:
    key: object
    machine: EpisodeMachine
    gen: object = None
    pending: GenRequest | None = None
    done: BudgetEpisode | None = None
    _text: str = ""


def run_batched_episodes(
    *,
    engine,
    episodes: Sequence[dict],
    policy_for: Callable,
    boundary_policy_for: Callable | None,
    state_fn: Callable,
    encode: Callable,
    max_blocks: int = 128,
    query_stride: int = 1,
    max_tokens_per_draw: int = 32768,
    prefetch: Callable | None = None,
    block_cap: int = 0,
) -> dict:
    live: list[_Live] = []
    for spec in episodes:
        machine = EpisodeMachine(
            prompt_ids=spec["prompt_ids"], gold=spec["gold"], question=spec["question"],
            budget=spec["budget"], policy=policy_for(spec["key"]), state_fn=state_fn,
            encode=encode,
            boundary_policy=(boundary_policy_for(spec["key"]) if boundary_policy_for else None),
            max_draws=spec.get("max_draws"), max_blocks=max_blocks,
            query_stride=query_stride, block_cap=block_cap,
            max_tokens_per_draw=max_tokens_per_draw,
            seed_key=spec.get("seed_key", spec["key"]),
        )
        live.append(_Live(key=spec["key"], machine=machine))

    for item in live:
        _advance_until_request(item)

    while any(item.done is None for item in live):
        waiting = [item for item in live if item.done is None and item.pending is not None]
        if not waiting:
            raise RuntimeError("scheduler stalled: live episodes with no pending request")
        outs = engine.generate_batch([item.pending for item in waiting])
        if len(outs) != len(waiting):
            raise RuntimeError(f"engine returned {len(outs)} outputs for {len(waiting)} requests")
        if prefetch is not None:
            prefetch(_prefixes_to_query(waiting, outs, max_blocks, query_stride))
        for item, out in zip(waiting, outs):
            _feed(item, out)
            _advance_until_request(item)

    return {item.key: item.done for item in live}


def _prefixes_to_query(waiting, outs, max_blocks, query_stride=1):
    from rl_controller.rollout_contract import block_count, classify_stop, BLOCK_STOP_STRING
    from rl_controller.rollout_runner import is_query_block

    out_prefixes = []
    for item, out in zip(waiting, outs):
        req = item.pending
        if req is None or req.stop != [BLOCK_STOP_STRING]:
            continue
        gen_state = getattr(item, "_text", "")
        text = gen_state + out.text
        item._text = text
        cls = classify_stop(text, finish_reason=out.finish_reason,
                            stop_reason=out.stop_reason, hit_eos=out.hit_eos)
        if cls.kind != "block":
            continue
        block = block_count(text)
        if not is_query_block(block, max_blocks=max_blocks, query_stride=query_stride):
            continue
        completion_ids = list(req.prompt_ids[len(item.machine.prompt_ids):]) + list(out.token_ids)
        out_prefixes.append((item.machine.prompt_ids, completion_ids))
    return out_prefixes


def _advance_until_request(item: _Live) -> None:
    while item.done is None and item.pending is None:
        if item.gen is None:
            nxt = item.machine.next_draw()
            if nxt is None:
                item.done = item.machine.finish()
                return
            item.gen = nxt
            item._text = ""
            try:
                item.pending = next(item.gen)
            except StopIteration as done:
                item.machine.absorb(done.value)
                item.gen = None


def _feed(item: _Live, out) -> None:
    item.pending = None
    try:
        item.pending = item.gen.send(out)
    except StopIteration as done:
        item.machine.absorb(done.value)
        item.gen = None


def run_batched_episodes_in_waves(
    *,
    engine,
    episodes: Sequence[dict],
    policy_for: Callable,
    boundary_policy_for: Callable | None,
    state_fn: Callable,
    encode: Callable,
    max_blocks: int = 128,
    query_stride: int = 1,
    max_tokens_per_draw: int = 32768,
    prefetch: Callable | None = None,
    block_cap: int = 0,
    waves: int = 2,
    thread_setup: Callable | None = None,
) -> dict:
    if waves < 1:
        raise ValueError(f"waves must be >= 1, got {waves}")
    if waves == 1 or len(episodes) < 2:
        return run_batched_episodes(
            engine=engine, episodes=episodes, policy_for=policy_for,
            boundary_policy_for=boundary_policy_for, state_fn=state_fn, encode=encode,
            max_blocks=max_blocks, query_stride=query_stride,
            max_tokens_per_draw=max_tokens_per_draw, prefetch=prefetch,
            block_cap=block_cap,
        )
    import threading

    gen_lock = threading.Lock()
    head_lock = threading.Lock()

    class _LockedEngine:
        def generate_batch(self, requests):
            with gen_lock:
                return engine.generate_batch(requests)

    def locked_state_fn(*a, **k):
        with head_lock:
            return state_fn(*a, **k)

    def locked_prefetch(prefixes):
        with head_lock:
            return prefetch(prefixes)

    groups = [list(episodes[i::waves]) for i in range(waves)]
    results: list = [None] * waves
    errors: list = [None] * waves

    def work(i):
        try:
            if thread_setup is not None:
                ctx = thread_setup()
                if ctx is not None and hasattr(ctx, "__enter__"):
                    ctx.__enter__()
            results[i] = run_batched_episodes(
                engine=_LockedEngine(), episodes=groups[i], policy_for=policy_for,
                boundary_policy_for=boundary_policy_for, state_fn=locked_state_fn,
                encode=encode, max_blocks=max_blocks, query_stride=query_stride,
                max_tokens_per_draw=max_tokens_per_draw, block_cap=block_cap,
                prefetch=(locked_prefetch if prefetch is not None else None),
            )
        except BaseException as exc:
            errors[i] = exc

    threads = [threading.Thread(target=work, args=(i,), name=f"rollout-wave-{i}") for i in range(waves)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for exc in errors:
        if exc is not None:
            raise exc
    out = {}
    for r in results:
        out.update(r)
    if len(out) != len(episodes):
        raise RuntimeError(f"waves returned {len(out)} episodes for {len(episodes)} specs")
    return out
