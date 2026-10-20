from __future__ import annotations

import hashlib
import json

CONTRACT_ID = "olmo3-7b-think-native-v1"

MODEL = "allenai/Olmo-3-7B-Think"
REVISION = "d97e442d7cc678210054dbcc9b440894d62c89a4"
ARCHITECTURE = "Olmo3ForCausalLM"

CHAT_TEMPLATE_SHA256 = "6d549883b5ed12879e191845c256a30c7dfd4eced0a4f160060a3ea0199d9e3a"
VOCAB_SIZE = 100278

HIDDEN_SIZE = 4096
NUM_HIDDEN_LAYERS = 32
MAX_TOKENS = 32768
MAX_MODEL_LEN = 65536
SLIDING_WINDOW = 4096
EOS_TOKEN_IDS = (100265, 100257)
PAD_TOKEN_ID = 100277

FEATURE_LAYER = 18
QWEN_FEATURE_LAYER = 16
QWEN_NUM_HIDDEN_LAYERS = 28


def official_generation_config() -> dict:
    return {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "eos_token_id": [100265, 100257],
        "max_new_tokens": 32768,
    }


def resolved_generation_config() -> dict:
    return {
        **official_generation_config(),
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": MAX_TOKENS,
    }


def assert_official_generation_config(config) -> None:
    for field, expected in official_generation_config().items():
        actual = config.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"official OLMo generation config field {field!r} must be "
                f"{expected!r}, got {actual!r}"
            )


FORBIDDEN_FIELDS = ("enable_thinking", "chat_template_kwargs")


def build_messages(question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                "Answer the following question. Enclose your answer in "
                f"<answer></answer>.\n<Question>: {question}"
            ),
        }
    ]


def assert_tokenizer(tokenizer) -> None:
    template = getattr(tokenizer, "get_chat_template", None)
    template = template() if callable(template) else getattr(tokenizer, "chat_template", None)
    if not template:
        raise ValueError("the OLMo contract needs a tokenizer that carries a chat template")
    digest = hashlib.sha256(template.encode()).hexdigest()
    if digest != CHAT_TEMPLATE_SHA256:
        raise ValueError(
            f"chat template sha256 {digest} is not the pinned OLMo template "
            f"{CHAT_TEMPLATE_SHA256}: this tokenizer is not {MODEL}@{REVISION}"
        )


def build_request_payload(tokenizer, model, question: str, seed: int):
    assert_tokenizer(tokenizer)
    messages = build_messages(question)
    prompt_ids = list(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    ))
    validate_prompt_budget(prompt_ids)
    payload = {
        "model": model,
        "messages": messages,
        "n": 1,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": MAX_TOKENS,
        "seed": seed,
        "add_generation_prompt": True,
        "return_token_ids": True,
        "stream": False,
    }
    return payload, prompt_ids


def validate_prompt_budget(prompt_ids) -> None:
    if len(prompt_ids) + MAX_TOKENS > MAX_MODEL_LEN:
        raise ValueError(
            f"{len(prompt_ids)} prompt tokens plus {MAX_TOKENS} completion tokens "
            f"exceed max_model_len={MAX_MODEL_LEN}"
        )


def identity() -> dict:
    return {
        "prompt_contract_id": CONTRACT_ID,
        "model_id": MODEL,
        "model_revision": REVISION,
        "architecture": ARCHITECTURE,
        "thinking_mode": "native_on",
        "think_open_in_prompt": True,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "generation": resolved_generation_config(),
        "max_model_len": MAX_MODEL_LEN,
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "feature_layer": FEATURE_LAYER,
    }


def make_run_fingerprint(dataset_fingerprint: str, chat_template: str) -> str:
    payload = {
        "contract": identity(),
        "dataset_fingerprint": dataset_fingerprint,
        "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()
