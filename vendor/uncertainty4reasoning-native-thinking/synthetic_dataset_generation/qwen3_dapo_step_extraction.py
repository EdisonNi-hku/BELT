"""Extract token-exact block or sentence steps from pinned Qwen3 generations."""

import argparse
import json
import re
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from tokenizers.decoders import ByteLevel
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
RUN_FINGERPRINT = "8ad84fdd58e42cfe597a128cda0059f51270adf5e0231b57a879b3d79cab4bf4"
THINK_OPEN_ID = 151667
THINK_CLOSE_ID = 151668
EXPECTED_SOURCE_COUNT = 5000
EXPECTED_CLOSED_COUNT = 4922
GRANULARITIES = ("block", "sent")
MIN_SENTENCE_CHARS = 20
BLOCK_SEPARATOR = re.compile(r"\n[ \t]*\n")
LEADING_MARKER = re.compile(r"^[ \t]*(?:[*\-+]|\d{1,2}[.)]|#{1,6})[ \t]+")
PURE_HEDGE = re.compile(
    r"^[\s*_#>-]*(?:wait|hmm+|hold on|ok|okay|right|so|alright|let'?s see|"
    r"let me see|hold on a moment)[\s.,!?*_:;-]*$",
    re.IGNORECASE,
)
TRAILING_QUESTION = re.compile(r"\?[\s*_`]*$")
ABBREVIATIONS = {"e.g", "i.e", "mr", "mrs", "ms", "dr", "prof", "vs", "no", "etc"}


def bytes_to_unicode() -> dict[int, str]:
    """Return the reversible byte alphabet used by GPT-2/Qwen ByteLevel BPE."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = list(byte_values)
    offset = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + offset)
            offset += 1
    return dict(zip(byte_values, map(chr, unicode_values)))


def _token_bytes(token_id: int, tokenizer, inverse_alphabet: dict[str, int]) -> bytes:
    added_tokens = getattr(tokenizer, "added_tokens_decoder", {})
    added = added_tokens.get(token_id)
    if added is not None:
        return str(added.content).encode("utf-8")

    backend = getattr(tokenizer, "backend_tokenizer", None)
    model = getattr(backend, "model", None)
    token = None if model is None else model.id_to_token(token_id)
    if token is None:
        raise ValueError(f"tokenizer vocabulary has no token for id {token_id}")
    try:
        return bytes(inverse_alphabet[character] for character in token)
    except KeyError as exc:
        raise ValueError(
            f"token id {token_id} contains a character outside the ByteLevel alphabet"
        ) from exc


def token_end_by_char(text: str, token_ids: list[int], tokenizer) -> list[int]:
    """Map each character to the generated token containing its final UTF-8 byte."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    decoder = getattr(backend, "decoder", None)
    if not isinstance(decoder, ByteLevel):
        raise ValueError("tokenizer must use the ByteLevel decoder")

    inverse_alphabet = {character: value for value, character in bytes_to_unicode().items()}
    pieces = [_token_bytes(token_id, tokenizer, inverse_alphabet) for token_id in token_ids]
    reconstructed_bytes = b"".join(pieces)
    try:
        reconstructed = reconstructed_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("ByteLevel token bytes are not valid UTF-8") from exc
    if reconstructed != text:
        raise ValueError(
            "ByteLevel reconstruction mismatch: "
            f"decoded {len(reconstructed)} characters, expected {len(text)}"
        )

    token_byte_ends = []
    total = 0
    for piece in pieces:
        total += len(piece)
        token_byte_ends.append(total)

    result = []
    token_index = 0
    character_byte_end = 0
    for character in text:
        character_byte_end += len(character.encode("utf-8"))
        while (
            token_index < len(token_byte_ends)
            and token_byte_ends[token_index] < character_byte_end
        ):
            token_index += 1
        if token_index >= len(token_byte_ends):
            raise ValueError("ByteLevel character extends past the generated token sequence")
        result.append(token_index)
    return result


def _block_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    position = 0
    for match in BLOCK_SEPARATOR.finditer(text):
        if text[position:match.start()].strip():
            spans.append((position, match.start()))
        position = match.end()
    if text[position:].strip():
        spans.append((position, len(text)))
    return spans


def _closes_list_marker(text: str, index: int) -> bool:
    if text[index] != ".":
        return False
    cursor = index - 1
    run = 0
    while cursor >= 0 and (text[cursor].isdigit() or text[cursor].isalpha()):
        cursor -= 1
        run += 1
    if run == 0 or run > 2:
        return False
    marker = text[cursor + 1:index]
    if not (marker.isdigit() or (marker.isalpha() and len(marker) == 1)):
        return False
    while cursor >= 0 and text[cursor] != "\n":
        if not text[cursor].isspace():
            return False
        cursor -= 1
    return True


def _is_sentence_boundary(text: str, index: int) -> bool:
    character = text[index]
    if character not in ".!?":
        return False
    if _closes_list_marker(text, index):
        return False
    if (
        character == "."
        and index > 0
        and index + 1 < len(text)
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    ):
        return False

    cursor = index - 1
    while cursor >= 0 and text[cursor].isspace():
        cursor -= 1
    word_end = cursor
    while cursor >= 0 and (text[cursor].isalpha() or text[cursor] in "’'"):
        cursor -= 1
    word = text[cursor + 1:word_end + 1].strip("’'").lower()
    return word not in ABBREVIATIONS


def _sentence_spans(text: str, min_chars: int = MIN_SENTENCE_CHARS) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for index in range(len(text)):
        if not _is_sentence_boundary(text, index):
            continue
        end = index + 1
        segment = text[start:end]
        if segment.strip():
            if min_chars > 0 and len(segment.strip()) < min_chars and spans:
                spans[-1] = (spans[-1][0], end)
            else:
                spans.append((start, end))
        start = end
    return spans


def _clean_claim_text(text: str) -> str:
    cleaned = text
    while True:
        stripped = LEADING_MARKER.sub("", cleaned, count=1)
        if stripped == cleaned:
            break
        cleaned = stripped
    return cleaned.strip() or text.strip()


def _is_non_assertive(text: str) -> bool:
    stripped = text.strip()
    return not stripped or bool(TRAILING_QUESTION.search(stripped)) or bool(PURE_HEDGE.match(stripped))


def _split_reasoning(
    text: str,
    token_ids: list[int],
    tokenizer,
    granularity: str = "block",
) -> list[dict]:
    if not text or not token_ids:
        return []
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {GRANULARITIES}, got {granularity!r}")
    end_by_char = token_end_by_char(text, token_ids, tokenizer)
    spans = _block_spans(text) if granularity == "block" else _sentence_spans(text)
    claims = []
    previous_token = 0
    for index, (start, end) in enumerate(spans):
        sentence_end = (
            spans[index + 1][0]
            if granularity == "block" and index + 1 < len(spans)
            else end
        )
        end_token = max(previous_token, end_by_char[end - 1])
        claim_text = _clean_claim_text(text[start:end])
        aligned = list(range(previous_token, min(end_token + 1, len(token_ids))))
        if _is_non_assertive(claim_text):
            aligned = []
        claims.append(
            {
                "claim_text": claim_text,
                "sentence": text[start:sentence_end],
                "aligned_token_ids": aligned,
            }
        )
        previous_token = min(end_token + 1, len(token_ids))
    return claims


def _split_reasoning_blocks(text: str, token_ids: list[int], tokenizer) -> list[dict]:
    """Backward-compatible name for the fixed block segmentation."""
    return _split_reasoning(text, token_ids, tokenizer, granularity="block")


def _require_equal(row: dict, field: str, expected) -> None:
    actual = row.get(field)
    if actual != expected:
        raise ValueError(f"{field} mismatch: expected {expected!r}, got {actual!r}")


def _integer_token_list(row: dict, field: str) -> list[int]:
    value = row.get(field)
    if not isinstance(value, list) or any(type(token_id) is not int for token_id in value):
        raise ValueError(f"{field} must be a list of integer token IDs")
    return value


def validate_source_row(
    row: dict,
    tokenizer,
    expected_run_fingerprint: str = RUN_FINGERPRINT,
    expected_sample_idx: int = 0,
) -> bool:
    """Validate generation provenance; return whether the trace is closed/eligible."""
    if type(expected_sample_idx) is not int or expected_sample_idx < 0:
        raise ValueError("expected_sample_idx must be a nonnegative integer")
    _require_equal(row, "model", MODEL)
    _require_equal(row, "revision", REVISION)
    _require_equal(row, "run_fingerprint", expected_run_fingerprint)
    _require_equal(row, "sample_idx", expected_sample_idx)
    if not row.get("dataset_fingerprint"):
        raise ValueError("dataset_fingerprint must be present")

    prompt_ids = _integer_token_list(row, "prompt_token_ids")
    completion_ids = _integer_token_list(row, "completion_token_ids")
    input_ids = _integer_token_list(row, "input_ids")
    if input_ids != prompt_ids + completion_ids:
        raise ValueError("input_ids must equal prompt_token_ids + completion_token_ids")

    decoded = tokenizer.decode(
        completion_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if row.get("reply") != decoded:
        raise ValueError("reply does not exactly decode from completion_token_ids")

    if row.get("has_think_open") is not True:
        raise ValueError("has_think_open must be true")
    thinking_start = row.get("thinking_start")
    if type(thinking_start) is not int or thinking_start <= 0:
        raise ValueError("thinking_start must be a positive integer")
    if thinking_start > len(completion_ids) or completion_ids[thinking_start - 1] != THINK_OPEN_ID:
        raise ValueError("opening marker is not immediately before thinking_start")

    has_close = row.get("has_think_close")
    thinking_end = row.get("thinking_end")
    if has_close is False and thinking_end is None:
        return False
    if has_close is not True:
        raise ValueError("has_think_close must be a boolean consistent with thinking_end")
    if type(thinking_end) is not int or not (thinking_start < thinking_end < len(completion_ids)):
        raise ValueError("thinking_end must be an integer inside completion_token_ids")
    if completion_ids[thinking_end] != THINK_CLOSE_ID:
        raise ValueError("closing marker is not at thinking_end")
    return True


def extract_row_claims(
    row: dict,
    tokenizer,
    granularity: str = "block",
    expected_run_fingerprint: str = RUN_FINGERPRINT,
    expected_sample_idx: int = 0,
) -> list[dict]:
    """Extract claims aligned to raw positions in the full completion sequence."""
    if not validate_source_row(
        row,
        tokenizer,
        expected_run_fingerprint,
        expected_sample_idx=expected_sample_idx,
    ):
        raise ValueError("row is not an eligible closed trace")

    completion_ids = row["completion_token_ids"]
    thinking_start = row["thinking_start"]
    thinking_end = row["thinking_end"]
    reasoning_ids = completion_ids[thinking_start:thinking_end]
    reasoning_text = tokenizer.decode(
        reasoning_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    claims = _split_reasoning(
        reasoning_text,
        reasoning_ids,
        tokenizer,
        granularity=granularity,
    )

    output = []
    previous_end = thinking_start - 1
    for claim in claims:
        aligned = [thinking_start + position for position in claim["aligned_token_ids"]]
        if aligned:
            if aligned != list(range(aligned[0], aligned[-1] + 1)):
                raise ValueError("claim aligned_token_ids must be contiguous")
            if aligned[0] <= previous_end:
                raise ValueError("claim aligned_token_ids overlap or are out of order")
            if aligned[0] < thinking_start or aligned[-1] >= thinking_end:
                raise ValueError("claim aligned_token_ids fall outside the thinking span")
            previous_end = aligned[-1]
        output.append(
            {
                "claim_text": claim["claim_text"],
                "sentence": claim["sentence"],
                "aligned_token_ids": aligned,
            }
        )
    return output


def _write_json_atomic(value: dict, path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


def run_partition(args: argparse.Namespace) -> dict:
    loaded = load_from_disk(args.dataset_path)
    dataset = loaded["train"] if isinstance(loaded, DatasetDict) else loaded
    if len(dataset) != args.expected_source_count:
        raise ValueError(
            f"expected {args.expected_source_count} source rows, found {len(dataset)}"
        )
    closed_count = sum(value is True for value in dataset["has_think_close"])
    if closed_count != args.expected_closed_count:
        raise ValueError(
            f"expected {args.expected_closed_count} closed rows, found {closed_count}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.hf_cache,
    )
    rows = []
    for row in dataset:
        original_index = row.get("original_index")
        if type(original_index) is not int:
            raise ValueError("original_index must be an integer")
        if original_index % args.partition_count != args.partition_index:
            continue
        eligible = validate_source_row(
            row,
            tokenizer,
            expected_run_fingerprint=args.expected_run_fingerprint,
        )
        if not eligible:
            continue
        row = dict(row)
        row["claims"] = extract_row_claims(
            row,
            tokenizer,
            granularity=args.granularity,
            expected_run_fingerprint=args.expected_run_fingerprint,
        )
        row["step_granularity"] = args.granularity
        row["claim_token_coordinate"] = "completion_token_offset"
        rows.append(row)

    output = DatasetDict({"train": Dataset.from_list(rows)})
    output.save_to_disk(args.output_path)
    summary = {
        "partition_index": args.partition_index,
        "partition_count": args.partition_count,
        "source_rows": len(dataset),
        "closed_rows": closed_count,
        "output_rows": len(rows),
        "claims": sum(len(row["claims"]) for row in rows),
        "annotatable_claims": sum(
            bool(claim["aligned_token_ids"])
            for row in rows
            for claim in row["claims"]
        ),
        "step_granularity": args.granularity,
        "claim_token_coordinate": "completion_token_offset",
        "run_fingerprint": args.expected_run_fingerprint,
    }
    _write_json_atomic(summary, args.summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--hf-cache", default=None)
    parser.add_argument("--partition-index", type=int, required=True)
    parser.add_argument("--partition-count", type=int, default=4)
    parser.add_argument("--expected-source-count", type=int, default=EXPECTED_SOURCE_COUNT)
    parser.add_argument("--expected-closed-count", type=int, default=EXPECTED_CLOSED_COUNT)
    parser.add_argument("--expected-run-fingerprint", default=RUN_FINGERPRINT)
    parser.add_argument("--granularity", choices=GRANULARITIES, default="block")
    args = parser.parse_args()
    if args.model != MODEL or args.revision != REVISION:
        parser.error("model and revision must match the fixed Qwen3 generation contract")
    if args.partition_count <= 0:
        parser.error("--partition-count must be positive")
    if args.partition_index < 0 or args.partition_index >= args.partition_count:
        parser.error("--partition-index must satisfy 0 <= index < count")
    if not args.expected_run_fingerprint:
        parser.error("--expected-run-fingerprint must not be empty")
    return args


if __name__ == "__main__":
    print(json.dumps(run_partition(parse_args()), indent=2, sort_keys=True))
