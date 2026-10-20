from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence

from rl_controller.budget_episode import BudgetEpisode
from rl_controller.episode_driver import EpisodeMachine


class AsyncStats:

    __slots__ = ("requests", "inflight_max", "inflight_integral", "engine_busy_seconds",
                 "head_busy_seconds", "head_batches", "head_items", "head_prefetch_calls")

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        for name in self.__slots__:
            setattr(self, name, 0 if name in ("requests", "inflight_max", "head_batches",
                                              "head_items", "head_prefetch_calls") else 0.0)

    @property
    def inflight_mean(self) -> float:
        return self.inflight_integral / self.engine_busy_seconds if self.engine_busy_seconds > 0 else 0.0

    @property
    def head_batch_mean(self) -> float:
        return self.head_items / self.head_batches if self.head_batches else 0.0

    def snapshot(self) -> dict:
        out = {name: getattr(self, name) for name in self.__slots__}
        out["inflight_mean"] = self.inflight_mean
        out["head_batch_mean"] = self.head_batch_mean
        return out


ASYNC_STATS = AsyncStats()


class SchedulerError(RuntimeError):
    pass


class _Task:
    __slots__ = ("fn", "hint", "payload", "result", "error", "event")

    def __init__(self, fn, hint, payload=None):
        self.fn = fn
        self.hint = hint
        self.payload = payload
        self.result = None
        self.error = None
        self.event = threading.Event()


_STOP = object()


class _HeadWorker:

    def __init__(self, *, prefetch=None, window_s: float = 0.005, max_batch: int = 0,
                 setup: Callable | None = None, candidates: Callable | None = None,
                 state_many: Callable | None = None):
        self._prefetch = prefetch
        self._state_many = state_many
        self._window = max(0.0, float(window_s))
        self._max_batch = max(0, int(max_batch))
        self._setup = setup
        self._candidates = candidates or (lambda: 1)
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name="rollout-head", daemon=True)
        self.error = None
        self.dead = False

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._q.put(_STOP)
        self._thread.join(timeout=60)

    def run(self, fn, hint=None, payload=None):
        if self.dead:
            raise self.error or SchedulerError("head worker is not running")
        task = _Task(fn, hint, payload if self._state_many is not None else None)
        self._q.put(task)
        while not task.event.wait(timeout=1.0):
            if self.dead:
                if not task.event.wait(timeout=1.0):
                    raise self.error or SchedulerError(
                        "head worker stopped before serving a decision")
                break
        if task.error is not None:
            raise task.error
        return task.result

    def _collect(self, first):
        batch = [first]
        stopping = False
        deadline = time.monotonic() + (self._window if first.hint is not None else 0.0)
        while not stopping:
            if self._max_batch and len(batch) >= self._max_batch:
                break
            remaining = deadline - time.monotonic()
            wait = remaining > 0 and len(batch) < self._candidates()
            try:
                nxt = self._q.get(timeout=remaining) if wait else self._q.get_nowait()
            except queue.Empty:
                break
            if nxt is _STOP:
                stopping = True
                break
            batch.append(nxt)
        return batch, stopping

    def _run(self):
        context = None
        try:
            if self._setup is not None:
                context = self._setup()
                if context is not None and hasattr(context, "__enter__"):
                    context.__enter__()
            while True:
                first = self._q.get()
                if first is _STOP:
                    return
                started = time.monotonic()
                batch, stopping = self._collect(first)
                hints = [t.hint for t in batch if t.hint is not None]
                if hints and self._prefetch is not None:
                    try:
                        self._prefetch(hints)
                        ASYNC_STATS.head_prefetch_calls += 1
                    except BaseException as exc:
                        for t in batch:
                            t.error = exc
                            t.event.set()
                        continue
                rest = batch
                states = [t for t in batch if t.payload is not None]
                if states:
                    rest = [t for t in batch if t.payload is None]
                    try:
                        results = self._state_many([t.payload for t in states])
                        if len(results) != len(states):
                            raise SchedulerError(
                                f"the batched head returned {len(results)} results for "
                                f"{len(states)} decisions")
                    except BaseException as exc:
                        for t in states:
                            t.error = exc
                            t.event.set()
                    else:
                        for t, result in zip(states, results):
                            t.result = result
                            t.event.set()
                for t in rest:
                    try:
                        t.result = t.fn()
                    except BaseException as exc:
                        t.error = exc
                    t.event.set()
                ASYNC_STATS.head_busy_seconds += time.monotonic() - started
                ASYNC_STATS.head_batches += 1
                ASYNC_STATS.head_items += len(batch)
                if stopping:
                    return
        except BaseException as exc:
            self.error = exc
            raise
        finally:
            self.dead = True
            self._fail_pending(self.error or SchedulerError("head worker stopped"))
            if context is not None and hasattr(context, "__exit__"):
                context.__exit__(None, None, None)

    def _fail_pending(self, exc):
        while True:
            try:
                task = self._q.get_nowait()
            except queue.Empty:
                return
            if task is _STOP:
                continue
            task.error = exc
            task.event.set()


class _Submitter:

    def __init__(self, engine):
        self.engine = engine
        self._lock = threading.Lock()
        self._own_loop = None
        self._own_thread = None
        self._generate_one = getattr(engine, "generate_one", None)
        self.concurrent = self._generate_one is not None
        if not self.concurrent:
            return
        self._loop = getattr(engine, "loop", None)
        if self._loop is None:
            import asyncio
            self._loop = asyncio.new_event_loop()
            ready = threading.Event()

            def _run():
                asyncio.set_event_loop(self._loop)
                self._loop.call_soon(ready.set)
                self._loop.run_forever()

            self._own_loop = self._loop
            self._own_thread = threading.Thread(target=_run, name="rollout-async-loop",
                                                daemon=True)
            self._own_thread.start()
            ready.wait()

    def submit(self, request):
        if self.concurrent:
            import asyncio
            return asyncio.run_coroutine_threadsafe(
                self._generate_one(request), self._loop).result()
        with self._lock:
            if hasattr(self.engine, "generate_batch"):
                outs = self.engine.generate_batch([request])
                if len(outs) != 1:
                    raise SchedulerError(f"engine returned {len(outs)} outputs for one request")
                return outs[0]
            return self.engine.generate(
                request.prompt_ids, stop=request.stop,
                include_stop_str_in_output=request.include_stop_str_in_output,
                max_tokens=request.max_tokens, seed_key=request.seed_key)

    def close(self):
        if self._own_loop is not None:
            self._own_loop.call_soon_threadsafe(self._own_loop.stop)
            self._own_thread.join(timeout=30)


class _Live:
    __slots__ = ("key", "machine", "done", "error")

    def __init__(self, key, machine):
        self.key = key
        self.machine = machine
        self.done = None
        self.error = None


def run_async_episodes(
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
    head_window_s: float = 0.005,
    head_max_batch: int = 32,
    thread_setup: Callable | None = None,
    state_many: Callable | None = None,
) -> dict:
    ASYNC_STATS.reset()
    if state_many is None:
        state_many = getattr(state_fn, "call_many", None)

    submitter = _Submitter(engine)
    lock = threading.Lock()
    state = {"inflight": 0, "live": len(episodes), "busy_since": None, "last": None}

    def _note(delta: int) -> None:
        now = time.monotonic()
        if state["last"] is not None:
            ASYNC_STATS.inflight_integral += state["inflight"] * (now - state["last"])
        state["last"] = now
        if delta > 0 and state["inflight"] == 0:
            state["busy_since"] = now
        state["inflight"] += delta
        if state["inflight"] == 0 and state["busy_since"] is not None:
            ASYNC_STATS.engine_busy_seconds += now - state["busy_since"]
            state["busy_since"] = None
        ASYNC_STATS.inflight_max = max(ASYNC_STATS.inflight_max, state["inflight"])

    def candidates() -> int:
        return max(1, state["live"] - state["inflight"])

    head = _HeadWorker(prefetch=prefetch, window_s=head_window_s,
                       max_batch=(head_max_batch or max(1, len(episodes))),
                       setup=thread_setup, candidates=candidates,
                       state_many=state_many)
    head.start()

    def on_head(fn, hint=None, payload=None):
        return head.run(fn, hint, payload)

    def wrapped_state_fn(prompt_ids, completion_ids, block, block_ends=None):
        return on_head(
            lambda: state_fn(prompt_ids, completion_ids, block, block_ends),
            (prompt_ids, completion_ids),
            (prompt_ids, completion_ids, block, block_ends))

    def wrap_policy(policy):
        if policy is None:
            return None
        return lambda obs: on_head(lambda: policy(obs))

    live = []
    for spec in episodes:
        machine = EpisodeMachine(
            prompt_ids=spec["prompt_ids"], gold=spec["gold"], question=spec["question"],
            budget=spec["budget"], policy=wrap_policy(policy_for(spec["key"])),
            state_fn=wrapped_state_fn, encode=encode,
            boundary_policy=(wrap_policy(boundary_policy_for(spec["key"]))
                             if boundary_policy_for else None),
            max_draws=spec.get("max_draws"), max_blocks=max_blocks,
            query_stride=query_stride, block_cap=block_cap,
            max_tokens_per_draw=max_tokens_per_draw,
            seed_key=spec.get("seed_key", spec["key"]),
        )
        live.append(_Live(spec["key"], machine))

    def fetch(request):
        with lock:
            _note(+1)
            ASYNC_STATS.requests += 1
        try:
            return submitter.submit(request)
        finally:
            with lock:
                _note(-1)

    def work(item: _Live) -> None:
        try:
            while True:
                gen = item.machine.next_draw()
                if gen is None:
                    item.done = item.machine.finish()
                    return
                try:
                    request = next(gen)
                    while True:
                        request = gen.send(fetch(request))
                except StopIteration as done:
                    item.machine.absorb(done.value)
        except BaseException as exc:
            item.error = exc
        finally:
            with lock:
                state["live"] -= 1

    threads = [threading.Thread(target=work, args=(item,), name=f"episode-{i}")
               for i, item in enumerate(live)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        head.stop()
        submitter.close()

    for item in live:
        if item.error is not None:
            raise item.error
    if head.error is not None:
        raise head.error
    missing = [item.key for item in live if item.done is None]
    if missing:
        raise SchedulerError(f"{len(missing)} episodes did not finish: {missing[:4]}")
    return {item.key: item.done for item in live}
