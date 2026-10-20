from __future__ import annotations

import zlib
from collections.abc import Sequence


class PackError(RuntimeError):
    pass


TWO_GPU_MEM_UTIL = 0.85
SAME_GPU_MEM_UTIL = 0.5
SAME_GPU_MEM_UTIL_MAX = 0.6

SAME_GPU_RESERVE_GB = {
    "backbone": 3.4, "feature_store": 6.0, "head_optimizer": 1.0,
    "backbone_kv_cache": 24.0,
}
GPU_TOTAL_GB = 96.0


def same_gpu(gpu_gen: int, gpu_head: int) -> bool:
    return int(gpu_gen) == int(gpu_head)


def pack_mode(gpu_gen: int, gpu_head: int) -> str:
    return "same-gpu" if same_gpu(gpu_gen, gpu_head) else "two-gpu"


def headroom_gb(mem_util: float) -> float:
    return GPU_TOTAL_GB * (1.0 - float(mem_util))


def resolve_gpu_mem_util(*, gpu_gen: int, gpu_head: int, requested=None) -> float:
    if int(gpu_gen) != 0:
        raise PackError(
            "--gpu-gen must be 0: vLLM takes the first visible device; "
            "select GPUs with CUDA_VISIBLE_DEVICES"
        )
    if int(gpu_head) < 0:
        raise PackError(f"--gpu-head must be >= 0, got {gpu_head}")
    shared = same_gpu(gpu_gen, gpu_head)
    if requested is None:
        return SAME_GPU_MEM_UTIL if shared else TWO_GPU_MEM_UTIL
    u = float(requested)
    if not 0.0 < u < 1.0:
        raise PackError(f"--gpu-mem-util must be in (0, 1), got {u}")
    if shared and u > SAME_GPU_MEM_UTIL_MAX:
        need = sum(SAME_GPU_RESERVE_GB.values())
        raise PackError(
            f"--gpu-mem-util {u} with the head on the generation GPU leaves "
            f"{headroom_gb(u):.1f} GB for the backbone, the feature store, the "
            f"head/optimizer and the backbone KV cache, which need ~{need:.1f} GB "
            f"(see rl_controller.gpu_pack). Keep it at or below "
            f"{SAME_GPU_MEM_UTIL_MAX} (default {SAME_GPU_MEM_UTIL})."
        )
    return u


def describe_pack(*, gpu_gen: int, gpu_head: int, mem_util: float) -> dict:
    return {
        "mode": pack_mode(gpu_gen, gpu_head), "gpu_gen": int(gpu_gen),
        "gpu_head": int(gpu_head), "gpu_mem_util": float(mem_util),
        "vllm_gb": round(GPU_TOTAL_GB * float(mem_util), 1),
        "headroom_gb": round(headroom_gb(mem_util), 1),
        "reserve_gb": round(sum(SAME_GPU_RESERVE_GB.values()), 1),
    }


def shard_configs(names: Sequence[str], *, index: int, count: int) -> list[str]:
    n = int(count)
    i = int(index)
    if n < 1:
        raise PackError(f"--shard-count must be >= 1, got {count}")
    if not 0 <= i < n:
        raise PackError(f"--shard-index must be in [0, {n}), got {index}")
    return list(names)[i::n]


def config_seed(base_seed: int, name: str) -> int:
    return int(base_seed) + (zlib.crc32(name.encode()) % 65536) * 1_000_000


def assert_distinct_seeds(names: Sequence[str], base_seed: int) -> None:
    seen: dict = {}
    for n in names:
        s = config_seed(base_seed, n)
        if s in seen:
            raise PackError(f"configurations {seen[s]!r} and {n!r} share a request seed")
        seen[s] = n
