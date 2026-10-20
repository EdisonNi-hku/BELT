from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


OURS_SEED = "</think>\n\n\\boxed{"

CODE_STOP_SEED = "\n**Final Answer**\n\nThe final answer is \\boxed"


@dataclass(frozen=True)
class Template:

    name: str
    seed: str
    seed_opens_brace: bool
    source: str
    stop_words: tuple = ("</think>",)

    def render_answer(self, forced) -> str:
        body = f"{forced}"
        return f"{body}}}" if self.seed_opens_brace else f"{{{body}}}"


TEMPLATES = {
    "ours": Template(
        name="ours",
        seed=OURS_SEED,
        seed_opens_brace=True,
        source="rl_controller.rollout_contract.forced_answer_prompt "
               "(NATURAL_CLOSE + BOXED_SEED) -- the template our pools were probed with",
        stop_words=("<|im_end|>", "</think>"),
    ),
    "code_stop": Template(
        name="code_stop",
        seed=CODE_STOP_SEED,
        seed_opens_brace=False,
        source="vendor/code_stop/method_codestop.py:137 -- the CoDE-Stop/DEER paper's "
               "own trial-answer prompt; our pools were never probed with it",
        stop_words=("</think>",),
    ),
}

DEFAULT_TEMPLATES = ("ours", "code_stop")


def check_seed_matches_contract() -> None:
    from rl_controller.rollout_contract import BOXED_SEED, NATURAL_CLOSE

    expected = NATURAL_CLOSE + BOXED_SEED
    if OURS_SEED != expected:
        raise SystemExit(
            f"probe_spec.OURS_SEED {OURS_SEED!r} != rollout_contract "
            f"{expected!r}. The pools were probed with the contract's string; "
            "either this constant is stale or the contract changed under an "
            "already-generated pool. Do not 'fix' this by copying the new value "
            "without re-checking which pools were generated when.")


@dataclass(frozen=True)
class ProbeSequence:

    qkey: str
    rep: int
    probe_index: int
    block: int
    prefix_len: int
    template: str
    ids: tuple[int, ...]
    n_prefix: int
    n_seed: int
    n_answer: int
    forced_tokens: int
    forced: object = None
    content_mask: tuple = ()
    mode: str = "teacher_forced"

    def __post_init__(self):
        if self.n_answer < 0:
            raise ValueError(f"{self.key()}: negative answer length")
        if self.n_prefix < 1 or self.n_seed < 1:
            raise ValueError(f"{self.key()}: empty prefix or seed")
        if len(self.ids) != self.n_prefix + self.n_seed + self.n_answer:
            raise ValueError(
                f"{self.key()}: {len(self.ids)} ids != {self.n_prefix} prefix + "
                f"{self.n_seed} seed + {self.n_answer} answer")

    def key(self) -> tuple[str, int, int, str]:
        return (self.qkey, self.rep, self.probe_index, self.template)

    @property
    def tail_len(self) -> int:
        return self.n_seed + self.n_answer + 1


def build_greedy_sequences(chain, prompt_ids, *, encode, templates=DEFAULT_TEMPLATES,
                           max_len=None) -> tuple[list[ProbeSequence], list[dict]]:
    seqs, skipped = [], []
    prompt_ids = list(prompt_ids)
    completion = chain["completion_ids"]
    if len(completion) != int(chain["thinking_tokens"]):
        raise ValueError(
            f"chain {(chain.get('qkey'), chain.get('rep'))}: completion_ids has "
            f"{len(completion)} tokens but thinking_tokens is {chain['thinking_tokens']}")
    seed_cache = {name: encode(TEMPLATES[name].seed) for name in templates}
    for probe_index, probe in enumerate(chain["probes"]):
        prefix_len = int(probe["prefix_len"])
        if not 0 < prefix_len <= len(completion):
            raise ValueError(
                f"chain {(chain.get('qkey'), chain.get('rep'))} probe {probe_index}: "
                f"prefix_len {prefix_len} outside (0, {len(completion)}]")
        base = prompt_ids + list(completion[:prefix_len])
        for name in templates:
            seed_ids = seed_cache[name]
            ids = base + seed_ids
            if max_len is not None and len(ids) > max_len:
                skipped.append({"qkey": chain["qkey"], "rep": int(chain["rep"]),
                                "probe_index": probe_index, "template": name,
                                "reason": f"sequence of {len(ids)} tokens exceeds max_len {max_len}"})
                continue
            seqs.append(ProbeSequence(
                qkey=chain["qkey"], rep=int(chain["rep"]), probe_index=probe_index,
                block=int(probe["block"]), prefix_len=prefix_len, template=name,
                ids=tuple(ids), n_prefix=len(base), n_seed=len(seed_ids), n_answer=0,
                forced_tokens=int(probe["forced_tokens"]), forced=probe["forced"],
                content_mask=()))
    return seqs, skipped


class GreedyState:

    def __init__(self, lengths, stop_ids, max_steps: int):
        self.lengths = [int(v) for v in lengths]
        if stop_ids and isinstance(stop_ids[0], (set, frozenset, list, tuple)):
            self.stop_ids = [set(int(v) for v in row) for row in stop_ids]
        else:
            shared = set(int(v) for v in stop_ids)
            self.stop_ids = [set(shared) for _ in self.lengths]
        if len(self.stop_ids) != len(self.lengths):
            raise ValueError("one stop-id set per row required")
        self.max_steps = int(max_steps)
        self.tokens = [[] for _ in lengths]
        self.probs = [[] for _ in lengths]
        self.finished = [False] * len(lengths)
        self.ended_with_stop = [False] * len(lengths)
        self.steps = 0

    def positions(self):
        return [self.lengths[i] + len(self.tokens[i]) for i in range(len(self.lengths))]

    def step(self, token_ids, probabilities) -> None:
        if len(token_ids) != len(self.lengths):
            raise ValueError("one token per row required")
        for i, (token, prob) in enumerate(zip(token_ids, probabilities)):
            if self.finished[i]:
                continue
            self.tokens[i].append(int(token))
            self.probs[i].append(float(prob))
            if int(token) in self.stop_ids[i]:
                self.finished[i] = True
                self.ended_with_stop[i] = True
        self.steps += 1
        if self.steps >= self.max_steps:
            for i in range(len(self.finished)):
                self.finished[i] = True

    def done(self) -> bool:
        return all(self.finished)


def build_sequences(chain, prompt_ids, *, encode, decode=None, templates=DEFAULT_TEMPLATES,
                    max_len=None) -> tuple[list[ProbeSequence], list[dict]]:
    seqs: list[ProbeSequence] = []
    skipped: list[dict] = []
    prompt_ids = list(prompt_ids)
    completion = chain["completion_ids"]
    if len(completion) != int(chain["thinking_tokens"]):
        raise ValueError(
            f"chain {(chain.get('qkey'), chain.get('rep'))}: completion_ids has "
            f"{len(completion)} tokens but thinking_tokens is {chain['thinking_tokens']}; "
            "prefix_len indexes completion_ids, so the two must agree "
            "(single_cot_probe.py enforces this at generation)")
    seed_cache = {name: encode(TEMPLATES[name].seed) for name in templates}
    for probe_index, probe in enumerate(chain["probes"]):
        prefix_len = int(probe["prefix_len"])
        if not 0 < prefix_len <= len(completion):
            raise ValueError(
                f"chain {(chain.get('qkey'), chain.get('rep'))} probe {probe_index}: "
                f"prefix_len {prefix_len} outside (0, {len(completion)}]")
        base = prompt_ids + list(completion[:prefix_len])
        for name in templates:
            tpl = TEMPLATES[name]
            seed_ids = seed_cache[name]
            if probe["forced"] is None:
                answer_ids = []
            else:
                answer_ids = encode(tpl.render_answer(probe["forced"]))
            if probe["forced"] is not None and not answer_ids:
                skipped.append({"qkey": chain["qkey"], "rep": int(chain["rep"]),
                                "probe_index": probe_index, "template": name,
                                "reason": f"rendered answer {tpl.render_answer(probe['forced'])!r} "
                                          "encodes to zero tokens"})
                continue
            ids = base + seed_ids + answer_ids
            if max_len is not None and len(ids) > max_len:
                skipped.append({"qkey": chain["qkey"], "rep": int(chain["rep"]),
                                "probe_index": probe_index, "template": name,
                                "reason": f"sequence of {len(ids)} tokens exceeds max_len {max_len}"})
                continue
            mask = (tuple(content_mask_for(answer_ids, decode)) if decode is not None
                    else tuple([1] * len(answer_ids)))
            seqs.append(ProbeSequence(
                qkey=chain["qkey"], rep=int(chain["rep"]), probe_index=probe_index,
                block=int(probe["block"]), prefix_len=prefix_len, template=name,
                ids=tuple(ids), n_prefix=len(base), n_seed=len(seed_ids),
                n_answer=len(answer_ids), forced_tokens=int(probe["forced_tokens"]),
                forced=probe["forced"], content_mask=mask))
    return seqs, skipped


def tail_index(seq_len: int, tail_len: int, position: int) -> int:
    idx = position - (seq_len - tail_len)
    if not 0 <= idx < tail_len:
        raise IndexError(
            f"position {position} is not in the kept tail of {tail_len} rows of a "
            f"{seq_len}-token sequence (tail covers {seq_len - tail_len}..{seq_len - 1})")
    return idx


def prefix_tail_index(seq: ProbeSequence) -> int:
    return tail_index(len(seq.ids), seq.tail_len, seq.n_prefix - 1)


def answer_tail_indices(seq: ProbeSequence) -> list[int]:
    start = seq.n_prefix + seq.n_seed
    return [tail_index(len(seq.ids), seq.tail_len, start + j - 1)
            for j in range(seq.n_answer)]


def answer_token_ids(seq: ProbeSequence) -> tuple[int, ...]:
    return seq.ids[seq.n_prefix + seq.n_seed:]


def pad_batch(sequences, pad_id: int):
    rows = [list(s) for s in sequences]
    if not rows:
        raise ValueError("pad_batch got an empty batch")
    width = max(len(r) for r in rows)
    input_ids, attention_mask, position_ids = [], [], []
    for row in rows:
        pad = width - len(row)
        input_ids.append([pad_id] * pad + row)
        mask = [0] * pad + [1] * len(row)
        attention_mask.append(mask)
        pos, running = [], 0
        for m in mask:
            running += m
            pos.append(max(running - 1, 0))
        position_ids.append(pos)
    return input_ids, attention_mask, position_ids


SCHEMA_VERSION = "iclr-trainingfree-confidence/v1"

ROW_FIELDS = (
    "schema_version", "qkey", "rep", "probe_index", "template",
    "block", "prefix_len", "forced", "forced_tokens",
    "n_answer", "answer_token_ids", "content_mask",
    "p_forced", "p_max", "argmax_hit", "h_answer",
    "h_prefix", "rank_close", "p_close",
)


def make_row(seq: ProbeSequence, *, p_forced, p_max, argmax_hit, h_answer,
             h_prefix: float, rank_close: int, p_close: float) -> dict:
    n = seq.n_answer
    for name, arr in (("p_forced", p_forced), ("p_max", p_max),
                      ("argmax_hit", argmax_hit), ("h_answer", h_answer),
                      ("content_mask", seq.content_mask)):
        if len(arr) != n:
            raise ValueError(f"{seq.key()}: {name} has {len(arr)} values for {n} answer tokens")
    if seq.mode == "teacher_forced" and (n == 0) != (seq.forced is None):
        raise ValueError(
            f"{seq.key()}: n_answer {n} and forced {seq.forced!r} disagree -- an answerless "
            "row exists only for an unparsed probe, and an unparsed probe has no answer")
    row = {
        "schema_version": SCHEMA_VERSION,
        "qkey": seq.qkey, "rep": seq.rep, "probe_index": seq.probe_index,
        "template": seq.template, "block": seq.block, "prefix_len": seq.prefix_len,
        "forced": seq.forced, "forced_tokens": seq.forced_tokens,
        "n_answer": n, "answer_token_ids": list(seq.ids[seq.n_prefix + seq.n_seed:]),
        "content_mask": [int(bool(m)) for m in seq.content_mask],
        "p_forced": [float(v) for v in p_forced],
        "p_max": [float(v) for v in p_max],
        "argmax_hit": [int(bool(v)) for v in argmax_hit],
        "h_answer": [float(v) for v in h_answer],
        "h_prefix": float(h_prefix), "rank_close": int(rank_close), "p_close": float(p_close),
        "mode": seq.mode,
    }
    validate_row(row)
    return row


def validate_row(row: dict) -> dict:
    missing = [k for k in ROW_FIELDS if k not in row]
    if missing:
        raise ValueError(f"confidence row lacks {missing}")
    unknown = sorted(set(row) - set(ROW_FIELDS) - set(OPTIONAL_ROW_FIELDS))
    if unknown:
        raise ValueError(f"confidence row has unknown key(s) {unknown}")
    if row.get("mode", "teacher_forced") not in MODES:
        raise ValueError(f"unknown mode {row['mode']!r}; choose from {MODES}")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"row schema {row['schema_version']!r} != {SCHEMA_VERSION!r}")
    if row["template"] not in TEMPLATES:
        raise ValueError(f"unknown template {row['template']!r}")
    n = row["n_answer"]
    if n < 0:
        raise ValueError("n_answer must be >= 0")
    if row.get("mode", "teacher_forced") == "teacher_forced" and (n == 0) != (row["forced"] is None):
        raise ValueError(f"n_answer {n} disagrees with forced {row['forced']!r}")
    for name in ("p_forced", "p_max", "argmax_hit", "h_answer", "answer_token_ids",
                 "content_mask"):
        if len(row[name]) != n:
            raise ValueError(f"{name} has {len(row[name])} values for n_answer {n}")
    for name in ("p_forced", "p_max"):
        for v in row[name]:
            if not (0.0 <= v <= 1.0000001):
                raise ValueError(f"{name} value {v} outside [0, 1]")
    for name in ("h_answer",):
        for v in row[name]:
            if not v == v or v < 0.0:
                raise ValueError(f"{name} value {v} is not a finite non-negative entropy")
    h = row["h_prefix"]
    if not h == h or h < 0.0:
        raise ValueError(f"h_prefix {h} is not a finite non-negative entropy")
    if row["rank_close"] < 1:
        raise ValueError(f"rank_close {row['rank_close']} must be >= 1 (1 = argmax)")
    return row


OPTIONAL_ROW_FIELDS = ("mode", "ended_with_think")

MODES = ("teacher_forced", "greedy")


def row_key(row) -> tuple[str, int, int, str]:
    return (row["qkey"], int(row["rep"]), int(row["probe_index"]), row["template"])


def geometric_mean(values) -> float:
    import math

    values = list(values)
    if not values:
        raise ValueError("geometric_mean of an empty sequence")
    total = 0.0
    for v in values:
        if v <= 0.0:
            return 0.0
        total += math.log(v)
    return math.exp(total / len(values))


def code_stop_confidence_verbatim(probs) -> float:
    import math

    probs = list(probs)
    if len(probs) < 2:
        raise ValueError(f"code_stop_confidence_verbatim needs >= 2 tokens, got {len(probs)}")
    body = probs[1:-1]
    if any(p <= 0.0 for p in body):
        return 0.0
    return math.exp(sum(math.log(p) for p in body) / (len(probs) - 1))


def answer_confidence(probs, content_mask=None) -> float:
    probs = list(probs)
    if not probs:
        raise ValueError("answer_confidence of an empty token list")
    if content_mask is not None:
        kept = [p for p, m in zip(probs, content_mask) if m]
        if kept:
            probs = kept
    return geometric_mean(probs)


def content_mask_for(answer_token_ids, decode) -> list[int]:
    mask = []
    for token_id in answer_token_ids:
        text = decode([token_id]).strip()
        mask.append(0 if text and set(text) <= {"{", "}"} else 1)
    if not any(mask):
        return [1] * len(mask)
    return mask


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(path, *, rows_written: int, shards, extra=None) -> dict:
    manifest = {"schema_version": SCHEMA_VERSION, "rows": int(rows_written),
                "shards": [{"path": str(p), "sha256": sha256_file(p)} for p in shards],
                **(extra or {})}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
