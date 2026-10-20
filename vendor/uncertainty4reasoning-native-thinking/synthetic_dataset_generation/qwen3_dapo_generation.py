"""Pure helpers for the pinned Qwen3-1.7B DAPO generation contract."""

import hashlib
import json
import re


MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
BASE_SEED = 20260730
MAX_TOKENS = 32768
MAX_MODEL_LEN = 40960


def official_generation_config():
    return {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "eos_token_id": [151645, 151643],
        "pad_token_id": 151643,
        "bos_token_id": 151643,
        "repetition_penalty": 1.0,
    }


def resolved_generation_config():
    return {
        **official_generation_config(),
        "min_p": 0.0,
        "max_tokens": MAX_TOKENS,
    }


def _exact_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _exact_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def build_messages(question):
    return [
        {
            "role": "user",
            "content": (
                "Answer the following question. Enclose your answer in "
                f"<answer></answer>.\n<Question>: {question}"
            ),
        }
    ]


def build_request_payload(tokenizer, model, question, seed):
    messages = build_messages(question)
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    payload = {
        "model": model,
        "messages": messages,
        "n": 1,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "max_tokens": MAX_TOKENS,
        "seed": seed,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": True},
        "return_token_ids": True,
        "stream": False,
    }
    return payload, list(prompt_ids)


def assert_official_generation_config(config):
    for field, expected in official_generation_config().items():
        actual = config.get(field)
        if not _exact_equal(actual, expected):
            raise ValueError(
                f"official generation config field {field!r} must be "
                f"{expected!r}, got {actual!r}"
            )


def validate_prompt_budget(prompt_ids):
    prompt_tokens = len(prompt_ids)
    if prompt_tokens + MAX_TOKENS > MAX_MODEL_LEN:
        raise ValueError(
            f"{prompt_tokens} prompt tokens plus {MAX_TOKENS} completion tokens "
            f"exceed max_model_len={MAX_MODEL_LEN}"
        )


def _find_subsequence(token_ids, marker_ids, start=0):
    if not marker_ids:
        raise ValueError("thinking marker token sequence must not be empty")
    last_start = len(token_ids) - len(marker_ids)
    for index in range(start, last_start + 1):
        if _exact_equal(token_ids[index:index + len(marker_ids)], marker_ids):
            return index
    return None


def completion_analysis(tokenizer, completion_ids):
    completion_ids = list(completion_ids)
    think_open_ids = list(tokenizer.encode("<think>", add_special_tokens=False))
    think_close_ids = list(tokenizer.encode("</think>", add_special_tokens=False))
    open_index = _find_subsequence(completion_ids, think_open_ids)
    close_anywhere = _find_subsequence(completion_ids, think_close_ids)

    thinking_start = None
    thinking_end = None
    num_thinking_tokens = None
    if open_index is not None:
        thinking_start = open_index + len(think_open_ids)
        close_after_open = _find_subsequence(
            completion_ids, think_close_ids, start=thinking_start
        )
        if close_after_open is not None:
            thinking_end = close_after_open
            num_thinking_tokens = thinking_end - thinking_start

    reply = tokenizer.decode(
        completion_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return {
        "reply": reply,
        "num_completion_tokens": len(completion_ids),
        "has_think_open": open_index is not None,
        "has_think_close": close_anywhere is not None,
        "thinking_start": thinking_start,
        "thinking_end": thinking_end,
        "num_thinking_tokens": num_thinking_tokens,
    }


def _last_boxed_content(text):
    index = text.rfind(r"\boxed")
    if index < 0:
        return None
    start = text.find("{", index)
    if start < 0:
        return None
    depth = 0
    for position in range(start, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : position]
    return None


def exact_integer_from_answer(reply):
    answers = re.findall(r"<answer>(.*?)</answer>", reply, flags=re.DOTALL)
    candidates = answers[-1:] + [_last_boxed_content(reply)]
    for candidate in candidates:
        if candidate is None:
            continue
        match = re.fullmatch(r"\s*([+-]?\d+)\s*", candidate)
        if match is None:
            continue
        try:
            return int(match.group(1))
        except ValueError:
            continue
    return None


def make_example_seed(base_seed, original_index, sample_idx):
    identity = f"{base_seed}:{original_index}:{sample_idx}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big") % (2**31 - 1)


def make_run_fingerprint(
    model,
    revision,
    dataset_fingerprint,
    chat_template,
    extra_contract=None,
):
    fingerprint_data = {
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
        "dataset_fingerprint": dataset_fingerprint,
        "generation": resolved_generation_config(),
        "max_model_len": MAX_MODEL_LEN,
        "model": model,
        "revision": revision,
    }
    if extra_contract is not None:
        fingerprint_data["extra_contract"] = extra_contract
    canonical_json = json.dumps(
        fingerprint_data,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _bounded_repr(value, limit=160):
    text = repr(value)
    if len(text) <= limit:
        return text
    return text[:limit - 3] + "..."


def _sequence_mismatch(actual, expected):
    first_difference = next(
        (
            index
            for index, (actual_item, expected_item) in enumerate(zip(actual, expected))
            if not _exact_equal(actual_item, expected_item)
        ),
        min(len(actual), len(expected)),
    )
    return (
        f"actual length {len(actual)}, expected length {len(expected)}, "
        f"first differing index {first_difference}"
    )


def _require_equal(row, field, expected):
    if field not in row:
        raise ValueError(f"response row is missing {field}")
    actual = row[field]
    if _exact_equal(actual, expected):
        return
    if isinstance(actual, list) and isinstance(expected, list):
        detail = _sequence_mismatch(actual, expected)
    else:
        detail = f"expected {_bounded_repr(expected)}, got {_bounded_repr(actual)}"
    raise ValueError(f"response row {field} mismatch: {detail}")


def _require_integer_token_ids(row, field):
    if field not in row:
        raise ValueError(f"response row is missing {field}")
    token_ids = row[field]
    if not isinstance(token_ids, list):
        raise ValueError(f"response row {field} must be a list")
    for index, token_id in enumerate(token_ids):
        if type(token_id) is not int:
            raise ValueError(
                f"response row {field}[{index}] must be an int, got "
                f"{type(token_id).__name__}"
            )
    return token_ids


def validate_response_row(
    row,
    expected_fingerprint,
    expected_index,
    expected_question,
    expected_gold_answer,
    expected_prompt_ids,
    tokenizer,
    expected_sample_idx=0,
    base_seed=BASE_SEED,
    extra_contract=None,
):
    _require_equal(row, "run_fingerprint", expected_fingerprint)
    _require_equal(row, "original_index", expected_index)
    _require_equal(row, "sample_idx", expected_sample_idx)
    _require_equal(
        row,
        "seed",
        make_example_seed(base_seed, expected_index, expected_sample_idx),
    )
    _require_equal(row, "question", expected_question)
    _require_equal(row, "gold_answer", expected_gold_answer)
    _require_equal(row, "model", MODEL)
    _require_equal(row, "revision", REVISION)

    if not row.get("dataset_fingerprint"):
        raise ValueError("response row dataset_fingerprint must be present")
    expected_template_hash = hashlib.sha256(
        tokenizer.chat_template.encode("utf-8")
    ).hexdigest()
    _require_equal(row, "chat_template_sha256", expected_template_hash)
    _require_equal(row, "generation_config", resolved_generation_config())
    reproduced_fingerprint = make_run_fingerprint(
        model=row["model"],
        revision=row["revision"],
        dataset_fingerprint=row["dataset_fingerprint"],
        chat_template=tokenizer.chat_template,
        extra_contract=extra_contract,
    )
    if reproduced_fingerprint != expected_fingerprint:
        raise ValueError(
            "response row dataset_fingerprint does not reproduce the expected "
            "run_fingerprint"
        )

    _require_integer_token_ids(row, "prompt_token_ids")
    _require_equal(row, "prompt_token_ids", list(expected_prompt_ids))
    completion_ids = _require_integer_token_ids(row, "completion_token_ids")
    _require_integer_token_ids(row, "input_ids")
    _require_equal(
        row,
        "input_ids",
        list(expected_prompt_ids) + completion_ids,
    )

    analysis = completion_analysis(tokenizer, completion_ids)
    for field, expected in analysis.items():
        _require_equal(row, field, expected)

    completion_length = len(completion_ids)
    for field in ("thinking_start", "thinking_end"):
        offset = row[field]
        if offset is not None and not 0 <= offset <= completion_length:
            raise ValueError(
                f"response row {field}={offset} is outside completion length "
                f"{completion_length}"
            )
    if row["thinking_start"] is not None and row["thinking_end"] is not None:
        if row["thinking_start"] > row["thinking_end"]:
            raise ValueError("response row thinking_start exceeds thinking_end")

    if "finish_reason" not in row:
        raise ValueError("response row is missing finish_reason")
    finish_reason = row["finish_reason"]
    if type(finish_reason) is not str or not finish_reason:
        raise ValueError(
            "response row finish_reason must be a nonempty str, got "
            f"{_bounded_repr(finish_reason)}"
        )

    if "stop_reason" not in row:
        raise ValueError("response row is missing stop_reason")
    stop_reason = row["stop_reason"]
    if stop_reason is not None and type(stop_reason) not in (int, str):
        raise ValueError(
            "response row stop_reason must be None, int, or str, got "
            f"{_bounded_repr(stop_reason)}"
        )
