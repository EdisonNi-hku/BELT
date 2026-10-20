from __future__ import annotations

from collections.abc import Sequence


def end_feature_index(prompt_len: int, last_aligned: int) -> int:
    if prompt_len < 1:
        raise ValueError(f"prompt_len {prompt_len} must be at least 1")
    if last_aligned < 0:
        raise ValueError(f"last_aligned {last_aligned} must be non-negative")
    index = prompt_len + last_aligned - 1
    if index < 0:
        raise ValueError(
            f"feature index {index} is negative for prompt_len={prompt_len}, "
            f"last_aligned={last_aligned}"
        )
    return index


def block_end_feature_indices(
    prompt_len: int, blocks: Sequence[Sequence[int]]
) -> list[int]:
    return [
        end_feature_index(prompt_len, int(tokens[-1]))
        for tokens in blocks
        if len(tokens) > 0
    ]


def assert_strictly_increasing(indices: Sequence[int]) -> None:
    for earlier, later in zip(indices, indices[1:]):
        if later <= earlier:
            raise ValueError(
                f"block end indices are not increasing: {earlier} then {later}. "
                "The claim masks are out of order with the block list, so every "
                "representation would carry a neighbouring block's label"
            )


def assert_within_features(indices: Sequence[int], feature_len: int) -> None:
    for index in indices:
        if not 0 <= index < feature_len:
            raise ValueError(
                f"feature index {index} outside [0, {feature_len}). An index of "
                f"exactly {feature_len} means the mask[1:] collator shift was "
                "not applied"
            )


def assert_tap_parity(*, raw_rows: int, head_rows: int, step_probs: int) -> None:
    if not raw_rows == head_rows == step_probs:
        raise ValueError(
            f"tap parity broken: raw 2048-d rows {raw_rows}, 768-d rows "
            f"{head_rows}, step_probs {step_probs}. P2048 is only a control for "
            "the head if it reads the head's own positions"
        )


def end_positions_from_mask(
    mask_values: Sequence, *, feature_len: int, head_rows: int, step_probs: int
) -> list[int]:
    positions = [index for index, value in enumerate(mask_values) if value]
    assert_strictly_increasing(positions)
    assert_within_features(positions, feature_len)
    assert_tap_parity(
        raw_rows=len(positions), head_rows=head_rows, step_probs=step_probs
    )
    return positions


def assert_tap_alignment(
    *, prompt_len: int, blocks: Sequence[Sequence[int]], feature_len: int,
    head_rows: int, step_probs: int,
) -> list[int]:
    indices = block_end_feature_indices(prompt_len, blocks)
    assert_strictly_increasing(indices)
    assert_within_features(indices, feature_len)
    assert_tap_parity(
        raw_rows=len(indices), head_rows=head_rows, step_probs=step_probs
    )
    return indices
