from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


_MODULE_MARKER = ("synthetic_dataset_generation", "qwen3_dapo_generation.py")


def _candidate_roots():
    import os

    env = os.environ.get("U4R_NATIVE_ROOT")
    if env:
        yield Path(env)
    here = Path(__file__).resolve()
    for ancestor in [here.parents[1]] + list(here.parents[2:6]):
        yield ancestor / "uncertainty4reasoning"
        yield ancestor / "vendor" / "uncertainty4reasoning-native-thinking"


def resolve_native_root() -> Path:
    for candidate in _candidate_roots():
        if candidate.joinpath(*_MODULE_MARKER).is_file():
            return candidate
    raise ContractError(
        "cannot find the pinned Qwen3-1.7B generation module "
        "(synthetic_dataset_generation/qwen3_dapo_generation.py). Set "
        "U4R_NATIVE_ROOT to the runtime uncertainty4reasoning clone or to "
        "vendor/uncertainty4reasoning-native-thinking"
    )


def _vendored_generation():
    root = resolve_native_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from synthetic_dataset_generation import qwen3_dapo_generation as g
    return g


QWEN_CONTRACT = "qwen3-1.7b-native-thinking-v1"
OLMO3_CONTRACT = "olmo3-7b-think-native-v1"
PROMPT_CONTRACTS = (QWEN_CONTRACT, OLMO3_CONTRACT)

CONTRACT_ENV = "RL_PROMPT_CONTRACT"

_ACTIVE_CONTRACT = None


def _env_contract():
    import os

    name = os.environ.get(CONTRACT_ENV)
    if name is None or not name.strip():
        return QWEN_CONTRACT
    name = name.strip()
    if name not in PROMPT_CONTRACTS:
        raise ContractError(
            f"{CONTRACT_ENV}={name!r} is not one of {PROMPT_CONTRACTS}"
        )
    return name


def active_prompt_contract() -> str:
    global _ACTIVE_CONTRACT
    if _ACTIVE_CONTRACT is None:
        _ACTIVE_CONTRACT = _env_contract()
    return _ACTIVE_CONTRACT


def reset_prompt_contract() -> None:
    global _ACTIVE_CONTRACT
    _ACTIVE_CONTRACT = None


def select_prompt_contract(name: str | None) -> str:
    global _ACTIVE_CONTRACT
    if name is None:
        reset_prompt_contract()
        return active_prompt_contract()
    if name not in PROMPT_CONTRACTS:
        raise ContractError(f"unknown prompt contract {name!r}; choose from {PROMPT_CONTRACTS}")
    _ACTIVE_CONTRACT = name
    return _ACTIVE_CONTRACT


def _olmo3():
    from rl_controller import olmo3_think_contract as o
    return o


def contract_identity(contract: str | None = None) -> dict:
    name = contract or active_prompt_contract()
    if name == OLMO3_CONTRACT:
        return _olmo3().identity()
    g = _vendored_generation()
    return {
        "prompt_contract_id": QWEN_CONTRACT,
        "model_id": g.MODEL,
        "model_revision": g.REVISION,
        "thinking_mode": "native_on",
        "think_open_in_prompt": False,
        "generation": g.resolved_generation_config(),
        "max_model_len": g.MAX_MODEL_LEN,
    }


BLOCK_SEPARATOR = re.compile(r"\n[ \t]*\n")

BLOCK_STOP_STRING = "\n\n"

CLOSE_TAG = "</think>"

ANSWER_INSTRUCTION_FROM = "Enclose your answer in <answer></answer>."
BOXED_INSTRUCTION = "Reply with the answer only. Answer enclosed with \\boxed{}."
BOXED_SEED = "\\boxed{"
BOXED_STOP = "}"

NATURAL_CLOSE = CLOSE_TAG + "\n\n"

CODE_INSTRUCTION = "Reply with the complete Python 3 program only, inside one ```python code block."
CODE_SEED = "```python\n"
CODE_STOP = "```"
ANSWER_FORMATS = {
    "boxed": (BOXED_INSTRUCTION, BOXED_SEED, BOXED_STOP),
    "code": (CODE_INSTRUCTION, CODE_SEED, CODE_STOP),
}
_INSTRUCTIONS = {triple[0] for triple in ANSWER_FORMATS.values()}


class ContractError(RuntimeError):
    pass


def build_prompt_ids(tokenizer, question: str, *, contract: str | None = None) -> list[int]:
    name = contract or active_prompt_contract()
    if name == OLMO3_CONTRACT:
        o = _olmo3()
        _, prompt_ids = o.build_request_payload(tokenizer, o.MODEL, question, seed=0)
        return list(prompt_ids)
    g = _vendored_generation()
    _, prompt_ids = g.build_request_payload(tokenizer, g.MODEL, question, seed=0)
    return list(prompt_ids)


def rewrite_user_instruction(prompt_ids, *, encode, decode,
                             instruction: str = BOXED_INSTRUCTION) -> list[int]:
    if instruction not in _INSTRUCTIONS:
        raise ContractError(f"instruction {instruction!r} is not one of ANSWER_FORMATS")
    ids = list(prompt_ids)
    text = decode(ids)
    if encode(text) != ids:
        raise ContractError(
            "the tokenizer does not round-trip this prompt, so rewriting the "
            "instruction would shift every downstream offset"
        )
    count = text.count(ANSWER_INSTRUCTION_FROM)
    if count != 1:
        raise ContractError(
            f"expected exactly one {ANSWER_INSTRUCTION_FROM!r} in the prompt, "
            f"found {count}"
        )
    return encode(text.replace(ANSWER_INSTRUCTION_FROM, instruction))


def instruction_for(answer_format: str) -> str:
    try:
        return ANSWER_FORMATS[answer_format][0]
    except KeyError:
        raise ContractError(f"unknown answer format {answer_format!r}; choose from "
                            f"{sorted(ANSWER_FORMATS)}") from None


def answer_formats(rows) -> dict:
    from rl_controller.splits import question_key

    out = {}
    for row in rows:
        fmt = str(row.get("answer_format") or "boxed")
        instruction_for(fmt)
        out[question_key(str(row["question"]))] = fmt
    return out


def prompt_ids_for(tokenizer, question: str, *, answer_format: str = "boxed",
                   encode, decode) -> list[int]:
    return rewrite_user_instruction(build_prompt_ids(tokenizer, question), encode=encode,
                                    decode=decode, instruction=instruction_for(answer_format))


def sampling_config(*, contract: str | None = None) -> dict:
    name = contract or active_prompt_contract()
    if name == OLMO3_CONTRACT:
        o = _olmo3()
        cfg = o.resolved_generation_config()
        o.assert_official_generation_config(cfg)
        return {
            "temperature": cfg["temperature"],
            "top_p": cfg["top_p"],
            "top_k": cfg["top_k"],
            "min_p": cfg["min_p"],
            "repetition_penalty": cfg["repetition_penalty"],
            "max_tokens": cfg["max_tokens"],
            "eos_token_id": list(cfg["eos_token_id"]),
        }
    g = _vendored_generation()
    cfg = g.resolved_generation_config()
    g.assert_official_generation_config(cfg)
    return {
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "top_k": cfg["top_k"],
        "min_p": cfg["min_p"],
        "repetition_penalty": cfg["repetition_penalty"],
        "max_tokens": cfg["max_tokens"],
        "eos_token_id": list(cfg["eos_token_id"]),
    }


@dataclass(frozen=True)
class StopClass:
    kind: str
    boundary_end: int | None


def classify_stop(accumulated_text: str, *, finish_reason: str, stop_reason,
                  hit_eos: bool) -> StopClass:
    if hit_eos:
        return StopClass("eos", None)
    if finish_reason == "length":
        return StopClass("length", None)
    if CLOSE_TAG in accumulated_text:
        return StopClass("think_closed", accumulated_text.index(CLOSE_TAG))
    if finish_reason == "stop":
        matches = list(BLOCK_SEPARATOR.finditer(accumulated_text))
        if matches and not accumulated_text[matches[-1].end():].strip():
            before = accumulated_text[:matches[-1].start()]
            prior = list(BLOCK_SEPARATOR.finditer(before))
            start = prior[-1].end() if prior else 0
            if before[start:].strip():
                return StopClass("block", matches[-1].start())
        return StopClass("partial", None)
    raise ContractError(f"unrecognised finish_reason {finish_reason!r}")


def block_count(accumulated_text: str) -> int:
    spans = 0
    position = 0
    for match in BLOCK_SEPARATOR.finditer(accumulated_text):
        if accumulated_text[position:match.start()].strip():
            spans += 1
        position = match.end()
    if accumulated_text[position:].strip():
        spans += 1
    return spans


def forced_answer_prompt(prefix_completion_ids, *, encode) -> list[int]:
    return list(prefix_completion_ids) + list(encode(NATURAL_CLOSE + BOXED_SEED))


def forced_answer_stop() -> dict:
    return {"stop": [BOXED_STOP], "include_stop_str_in_output": True}


def grade_forced(completion_text: str):
    g = _vendored_generation()
    return g.exact_integer_from_answer(BOXED_SEED + completion_text)


def natural_boxed_prompt(completion_ids, *, encode, decode) -> list[int]:
    ids = list(completion_ids)
    text = decode(ids)
    if CLOSE_TAG not in text:
        return ids + list(encode(NATURAL_CLOSE + BOXED_SEED))
    pad = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    return ids + list(encode(pad + BOXED_SEED))


def grade_natural(reply_text: str):
    g = _vendored_generation()
    return g.exact_integer_from_answer(reply_text)


def forced_answer_token_cost(completion_ids) -> int:
    return len(completion_ids)


def code_answer_prompt(prefix_completion_ids, *, encode) -> list[int]:
    return list(prefix_completion_ids) + list(encode(NATURAL_CLOSE + CODE_SEED))


def natural_code_prompt(completion_ids, *, encode, decode) -> list[int]:
    ids = list(completion_ids)
    text = decode(ids)
    if CLOSE_TAG not in text:
        return ids + list(encode(NATURAL_CLOSE + CODE_SEED))
    pad = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    return ids + list(encode(pad + CODE_SEED))


def code_answer_stop() -> dict:
    return {"stop": [CODE_STOP], "include_stop_str_in_output": True}


def extract_program(completion_text: str) -> str | None:
    if completion_text.endswith(CODE_STOP):
        return completion_text[: -len(CODE_STOP)]
    return None
