from __future__ import annotations

import glob
import json

POSITION_FEATURE_NAMES = ("block_index", "tokens_so_far", "mean_block_tokens")


def position_features(block: int, block_end_offsets) -> list[float]:
    if block < 1 or block > len(block_end_offsets):
        raise ValueError(
            f"block {block} outside 1..{len(block_end_offsets)}"
        )
    tokens = float(block_end_offsets[block - 1])
    return [float(block), tokens, tokens / block]


def load_index(index_glob):
    entries = {}
    for path in sorted(glob.glob(index_glob)):
        array_path = path.replace(".index.jsonl", ".npy")
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["original_index"], row["sample_idx"])
            if key in entries:
                raise ValueError(f"duplicate trace {key} in {path}")
            entries[key] = dict(row, array_path=array_path)
    if not entries:
        raise ValueError(f"no index entries matched {index_glob}")
    return entries


def usable_blocks(entry, *, lo: int = 2, hi: int | None = None):
    top = len(entry["blocks"]) if hi is None else min(hi, len(entry["blocks"]))
    top = min(top, entry["n_blocks"] - 1)
    return list(range(lo, top + 1))


def sample_positions(entry, k: int, rng, *, lo: int = 2, hi: int | None = None):
    blocks = usable_blocks(entry, lo=lo, hi=hi)
    if len(blocks) <= k:
        return blocks
    return sorted(rng.sample(blocks, k))


def open_arrays(entries):
    import numpy as np

    return {
        path: np.load(path, mmap_mode="r")
        for path in sorted({entry["array_path"] for entry in entries.values()})
    }


def representation(entry, block: int, arrays):
    if block < 1 or block > len(entry["blocks"]):
        raise ValueError(f"block {block} not in the dump for this trace")
    array = arrays[entry["array_path"]]
    return [float(v) for v in array[entry["row_offset"] + block - 1]]
