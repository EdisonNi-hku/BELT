from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from rl_controller.budget_episode import ABANDON, ANSWER, CONTINUE, INNER_ACTIONS
from rl_controller.rollout_contract import (
    BLOCK_SEPARATOR,
    BLOCK_STOP_STRING,
    ContractError,
    block_count,
    classify_stop,
    forced_answer_prompt,
    forced_answer_stop,
    forced_answer_token_cost,
    grade_forced,
    grade_natural,
)


MAX_FORCED_ANSWER_TOKENS = 32
MIN_FORCED_ANSWER_TOKENS = 4


class RolloutError(RuntimeError):
    pass


class RolloutStats:

    __slots__ = ("cap_hits", "straddled_boundaries", "straddled_queries_skipped",
                 "boundary_gaps")

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        for name in self.__slots__:
            setattr(self, name, 0)

    def snapshot(self) -> dict:
        return {name: getattr(self, name) for name in self.__slots__}


STATS = RolloutStats()


def block_ending_separators(text: str):
    out = []
    position = 0
    for match in BLOCK_SEPARATOR.finditer(text):
        if text[position:match.start()].strip():
            out.append((match.start(), match.end()))
        position = match.end()
    return out


def straddled_block_end(accumulated_text: str, junction_chars: int, *,
                        reported: bool, continuation_text: str,
                        continuation_token_ids, encode):
    seps = block_ending_separators(accumulated_text)
    reported_end = seps[-1][1] if (reported and seps) else None
    for index, (start, end) in enumerate(seps):
        if not (start < junction_chars <= end):
            continue
        if reported_end is not None and end == reported_end:
            return None
        tail = end - junction_chars
        extra = 0
        if tail > 0:
            ids = list(encode(continuation_text[:tail]))
            if ids and list(continuation_token_ids[:len(ids)]) == ids:
                extra = len(ids)
        return index + 1, extra
    return None


@dataclass
class DrawResult:
    outcome: str
    voted: bool
    answer: int | None
    completion_ids: list
    tokens: int
    blocks_seen: int
    end_block: int | None
    decisions: list = field(default_factory=list)
    forced_ids: list = field(default_factory=list)
    block_ends: list = field(default_factory=list)


@dataclass
class GenRequest:
    prompt_ids: list
    stop: list
    include_stop_str_in_output: bool
    max_tokens: int
    seed_key: object | None = None


def request_seed(base_seed: int, seed_key) -> int:
    payload = repr((int(base_seed), seed_key)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2 ** 31 - 1)


def is_query_block(block: int, *, max_blocks: int, query_stride: int = 1) -> bool:
    if query_stride < 1:
        raise RolloutError(f"query_stride must be >= 1, got {query_stride}")
    return 2 <= block <= max_blocks and block % query_stride == 0


def draw_steps(*, prompt_ids, policy, observe, encode, max_blocks: int,
               max_total_tokens: int, budget_left: int, query_stride: int = 1,
               block_cap: int = 0, request_seed_base=None):
    completion_ids: list = []
    accumulated_text = ""
    decisions: list = []
    blocks_seen = 0
    block_ends: list = []
    junction = None
    request_index = 0

    while True:
        remaining = min(max_total_tokens - len(completion_ids), budget_left - len(completion_ids))
        if remaining <= 0:
            return _cut(completion_ids, accumulated_text, blocks_seen, decisions, block_ends)

        want = remaining if block_cap <= 0 else min(remaining, block_cap)
        capped_request = want < remaining

        out = yield GenRequest(
            prompt_ids=list(prompt_ids) + completion_ids,
            stop=[BLOCK_STOP_STRING], include_stop_str_in_output=True,
            max_tokens=want,
            seed_key=(None if request_seed_base is None else (request_seed_base, request_index)),
        )
        request_index += 1
        continuation_text = out.text
        continuation_ids = list(out.token_ids)
        completion_ids = completion_ids + continuation_ids
        accumulated_text = accumulated_text + continuation_text

        cls = classify_stop(accumulated_text, finish_reason=out.finish_reason,
                            stop_reason=out.stop_reason, hit_eos=out.hit_eos)

        if junction is not None:
            junction_chars, junction_tokens = junction
            junction = None
            found = straddled_block_end(
                accumulated_text, junction_chars, reported=(cls.kind == "block"),
                continuation_text=continuation_text,
                continuation_token_ids=continuation_ids, encode=encode)
            if found is not None:
                straddled_block, extra = found
                end = min(junction_tokens + extra, len(completion_ids))
                if end > (block_ends[-1] if block_ends else 0):
                    block_ends.append(end)
                    blocks_seen += 1
                    STATS.straddled_boundaries += 1
                    if is_query_block(straddled_block, max_blocks=max_blocks,
                                      query_stride=query_stride):
                        STATS.straddled_queries_skipped += 1

        if cls.kind == "length" and capped_request:
            STATS.cap_hits += 1
            junction = (len(accumulated_text), len(completion_ids))
            continue

        if cls.kind in ("think_closed", "eos"):
            answer = grade_natural(accumulated_text)
            return DrawResult(
                outcome="natural", voted=True, answer=answer,
                completion_ids=completion_ids, tokens=len(completion_ids),
                blocks_seen=blocks_seen, end_block=None, decisions=decisions,
                block_ends=list(block_ends),
            )
        if cls.kind == "length":
            if len(completion_ids) >= budget_left:
                return _cut(completion_ids, accumulated_text, blocks_seen, decisions, block_ends)
            answer = grade_natural(accumulated_text)
            return DrawResult(
                outcome="length", voted=True, answer=answer,
                completion_ids=completion_ids, tokens=len(completion_ids),
                blocks_seen=blocks_seen, end_block=None, decisions=decisions,
                block_ends=list(block_ends),
            )
        if cls.kind == "partial":
            continue

        block = block_count(accumulated_text)
        blocks_seen += 1
        block_ends.append(len(completion_ids))
        if len(block_ends) != block:
            STATS.boundary_gaps += 1
        if not is_query_block(block, max_blocks=max_blocks, query_stride=query_stride):
            continue

        action = policy(observe(block, completion_ids, list(block_ends)))
        if action not in INNER_ACTIONS:
            raise RolloutError(f"policy returned unknown action {action!r}")
        decisions.append((block, action))

        if action == CONTINUE:
            continue
        if action == ABANDON:
            return DrawResult(
                outcome="abandoned", voted=False, answer=None,
                completion_ids=completion_ids, tokens=len(completion_ids),
                blocks_seen=blocks_seen, end_block=block, decisions=decisions,
                block_ends=list(block_ends),
            )
        room = budget_left - len(completion_ids)
        if room < MIN_FORCED_ANSWER_TOKENS:
            return _cut(completion_ids, accumulated_text, blocks_seen, decisions, block_ends)
        forced_prompt = list(prompt_ids) + forced_answer_prompt(completion_ids, encode=encode)
        forced = yield GenRequest(
            prompt_ids=forced_prompt, max_tokens=min(MAX_FORCED_ANSWER_TOKENS, room),
            seed_key=(None if request_seed_base is None else (request_seed_base, request_index)),
            **forced_answer_stop(),
        )
        request_index += 1
        answer = grade_forced(forced.text)
        cost = len(completion_ids) + forced_answer_token_cost(forced.token_ids)
        return DrawResult(
            outcome="answered", voted=True, answer=answer,
            completion_ids=completion_ids, tokens=cost,
            blocks_seen=blocks_seen, end_block=block, decisions=decisions,
            forced_ids=list(forced.token_ids), block_ends=list(block_ends),
        )


def run_draw(*, engine, prompt_ids, policy, observe, encode, max_blocks: int,
             max_total_tokens: int, budget_left: int, query_stride: int = 1,
             block_cap: int = 0, request_seed_base=None) -> DrawResult:
    gen = draw_steps(prompt_ids=prompt_ids, policy=policy, observe=observe,
                     encode=encode, max_blocks=max_blocks,
                     max_total_tokens=max_total_tokens, budget_left=budget_left,
                     query_stride=query_stride, block_cap=block_cap,
                     request_seed_base=request_seed_base)
    try:
        req = next(gen)
        while True:
            out = engine.generate(req.prompt_ids, stop=req.stop,
                                  include_stop_str_in_output=req.include_stop_str_in_output,
                                  max_tokens=req.max_tokens, seed_key=req.seed_key)
            req = gen.send(out)
    except StopIteration as done:
        return done.value


def _cut(completion_ids, accumulated_text, blocks_seen, decisions, block_ends=()) -> DrawResult:
    return DrawResult(
        outcome="length", voted=True, answer=grade_natural(accumulated_text),
        completion_ids=completion_ids, tokens=len(completion_ids),
        blocks_seen=blocks_seen, end_block=None, decisions=decisions,
        block_ends=list(block_ends),
    )


class VLLMEngine:

    def __init__(self, llm, *, sampling: dict, seed: int):
        from vllm import SamplingParams

        self._llm = llm
        self._SamplingParams = SamplingParams
        self._sampling = dict(sampling)
        self._seed = seed
        self._calls = 0
        self.seed_mode = None
        self.seconds = 0.0
        self.batches = 0
        self.max_batch = 0
        self.round_tokens_max_sum = 0
        self.round_tokens_mean_sum = 0.0
        self.round_requests = 0

    def reset_stream(self, seed: int) -> None:
        self._seed = int(seed)
        self._calls = 0

    def _seed_for(self, seed_key):
        self._calls += 1
        if seed_key is None:
            if self.seed_mode is None:
                self.seed_mode = "call-count"
                self._announce_seed_mode()
            return self._seed + self._calls
        if self.seed_mode is None:
            self.seed_mode = "request-key"
            self._announce_seed_mode()
        return request_seed(self._seed, seed_key)

    def _announce_seed_mode(self):
        import json
        print(json.dumps({"sampling_seed_mode": self.seed_mode, "base_seed": self._seed}),
              flush=True)

    extra_sampling_kwargs: dict = {}

    def _params(self, *, stop, include_stop_str_in_output, max_tokens, seed_key=None):
        seed = self._seed_for(seed_key)
        return self._SamplingParams(
            **self.extra_sampling_kwargs,
            temperature=self._sampling["temperature"],
            top_p=self._sampling["top_p"],
            top_k=self._sampling["top_k"],
            min_p=self._sampling["min_p"],
            repetition_penalty=self._sampling["repetition_penalty"],
            max_tokens=int(max_tokens),
            stop=list(stop),
            include_stop_str_in_output=bool(include_stop_str_in_output),
            seed=seed,
        )

    def _wrap(self, out):
        eos_ids = set(self._sampling["eos_token_id"])
        hit_eos = bool(out.token_ids) and out.token_ids[-1] in eos_ids
        return _Out(
            token_ids=list(out.token_ids), text=out.text,
            finish_reason=out.finish_reason, stop_reason=out.stop_reason,
            hit_eos=hit_eos,
        )

    def generate(self, prompt_ids, *, stop, include_stop_str_in_output, max_tokens,
                 seed_key=None):
        from vllm import TokensPrompt

        params = self._params(stop=stop, include_stop_str_in_output=include_stop_str_in_output,
                              max_tokens=max_tokens, seed_key=seed_key)
        outputs = self._llm.generate([TokensPrompt(prompt_token_ids=list(prompt_ids))],
                                     params, use_tqdm=False)
        return self._wrap(outputs[0].outputs[0])

    def generate_batch(self, requests):
        from vllm import TokensPrompt

        if not requests:
            return []
        import time
        t = time.time()
        prompts = [TokensPrompt(prompt_token_ids=list(r.prompt_ids)) for r in requests]
        params = [self._params(stop=r.stop, include_stop_str_in_output=r.include_stop_str_in_output,
                               max_tokens=r.max_tokens, seed_key=r.seed_key) for r in requests]
        outputs = self._llm.generate(prompts, params, use_tqdm=False)
        self.seconds += time.time() - t
        self.batches += 1
        self.max_batch = max(self.max_batch, len(requests))
        if len(outputs) != len(requests):
            raise RolloutError(f"vLLM returned {len(outputs)} outputs for {len(requests)} prompts")
        lens = [len(o.outputs[0].token_ids) for o in outputs]
        self.round_tokens_max_sum += max(lens)
        self.round_tokens_mean_sum += sum(lens) / len(lens)
        self.round_requests += len(lens)
        return [self._wrap(o.outputs[0]) for o in outputs]


@dataclass
class _Out:
    token_ids: list
    text: str
    finish_reason: str
    stop_reason: object
    hit_eos: bool


class AsyncVLLMEngine(VLLMEngine):

    def __init__(self, engine_factory, *, sampling: dict, seed: int,
                 request_prefix: str = "rl"):
        import asyncio
        import threading

        super().__init__(None, sampling=sampling, seed=seed)
        self.extra_sampling_kwargs = _final_only_kwargs()
        self._request_prefix = request_prefix
        self._request_counter = 0
        self._counter_lock = threading.Lock()
        self._inflight = 0
        self._busy_started = None
        self._inflight_integral = 0.0
        self._inflight_last = None
        self.inflight_max = 0
        self.requests = 0

        self.loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.call_soon(ready.set)
            self.loop.run_forever()

        self._loop_thread = threading.Thread(target=_run_loop, name="vllm-async-loop",
                                             daemon=True)
        self._loop_thread.start()
        ready.wait()

        async def _build():
            return engine_factory()

        self._llm = asyncio.run_coroutine_threadsafe(_build(), self.loop).result()


    def _next_request_id(self) -> str:
        with self._counter_lock:
            self._request_counter += 1
            return f"{self._request_prefix}-{self._request_counter}"


    def _note(self, delta: int) -> None:
        import time
        with self._counter_lock:
            now = time.time()
            if self._inflight_last is not None:
                self._inflight_integral += self._inflight * (now - self._inflight_last)
            self._inflight_last = now
            if delta > 0 and self._inflight == 0:
                self._busy_started = now
            self._inflight += delta
            if self._inflight == 0 and self._busy_started is not None:
                self.seconds += now - self._busy_started
                self._busy_started = None
            self.inflight_max = max(self.inflight_max, self._inflight)
            self.max_batch = max(self.max_batch, self._inflight)

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def inflight_mean(self) -> float:
        return self._inflight_integral / self.seconds if self.seconds > 0 else 0.0


    async def generate_one(self, request):
        from vllm import TokensPrompt

        params = self._params(stop=request.stop,
                              include_stop_str_in_output=request.include_stop_str_in_output,
                              max_tokens=request.max_tokens, seed_key=request.seed_key)
        prompt = TokensPrompt(prompt_token_ids=list(request.prompt_ids))
        self._note(+1)
        try:
            final = None
            async for out in self._llm.generate(prompt, params, self._next_request_id()):
                final = out
            if final is None or not final.outputs:
                raise RolloutError("async engine returned no output for a request")
        finally:
            self._note(-1)
        self.requests += 1
        self.batches += 1
        n = len(final.outputs[0].token_ids)
        self.round_tokens_max_sum += n
        self.round_tokens_mean_sum += n
        self.round_requests += 1
        return self._wrap(final.outputs[0])

    def submit(self, request):
        import asyncio
        return asyncio.run_coroutine_threadsafe(self.generate_one(request), self.loop)

    def generate(self, prompt_ids, *, stop, include_stop_str_in_output, max_tokens,
                 seed_key=None):
        req = GenRequest(prompt_ids=list(prompt_ids), stop=list(stop),
                         include_stop_str_in_output=bool(include_stop_str_in_output),
                         max_tokens=int(max_tokens), seed_key=seed_key)
        return self.submit(req).result()

    def generate_batch(self, requests):
        if not requests:
            return []
        futures = [self.submit(r) for r in requests]
        return [f.result() for f in futures]

    def close(self) -> None:
        shutdown = getattr(self._llm, "shutdown", None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception:
                pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._loop_thread.join(timeout=30)


def _final_only_kwargs() -> dict:
    try:
        from vllm.sampling_params import RequestOutputKind
    except Exception:
        return {}
    return {"output_kind": RequestOutputKind.FINAL_ONLY}


def build_async_llm(**engine_kwargs):
    import dataclasses
    import json

    from vllm.engine.arg_utils import AsyncEngineArgs

    try:
        known = {f.name for f in dataclasses.fields(AsyncEngineArgs)}
    except TypeError:
        known = set(engine_kwargs)
    dropped = sorted(k for k in engine_kwargs if k not in known)
    kwargs = {k: v for k, v in engine_kwargs.items() if k in known}
    if "disable_log_requests" in known:
        kwargs.setdefault("disable_log_requests", True)
    elif "enable_log_requests" in known:
        kwargs.setdefault("enable_log_requests", False)
    if dropped:
        print(json.dumps({"async_engine_dropped_kwargs": dropped}), flush=True)
    args = AsyncEngineArgs(**kwargs)

    try:
        from vllm.v1.engine.async_llm import AsyncLLM
    except Exception:
        from vllm import AsyncLLMEngine as AsyncLLM
    if hasattr(AsyncLLM, "from_engine_args"):
        return AsyncLLM.from_engine_args(args)
    return AsyncLLM.from_vllm_config(args.create_engine_config())
