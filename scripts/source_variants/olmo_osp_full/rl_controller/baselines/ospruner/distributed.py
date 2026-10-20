from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

SCORE_SHARD_SCHEMA = "ospruner-score-shard/v1"

EXECUTION_SCHEMA = "ospruner-execution/v1"

DEFAULT_TIMEOUT_S = 3 * 3600


@dataclass(frozen=True)
class Runtime:

    rank: int
    world: int
    local_rank: int

    @property
    def is_writer(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world > 1

    def as_dict(self) -> dict:
        return {"rank": self.rank, "world": self.world, "local_rank": self.local_rank}


SINGLE = Runtime(rank=0, world=1, local_rank=0)


def runtime_from_env(env=None, *, single_node: bool = True) -> Runtime:
    env = os.environ if env is None else env

    def _int(name, default):
        raw = env.get(name)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"{name}={raw!r} is not an integer") from None

    world = _int("WORLD_SIZE", 1)
    rank = _int("RANK", 0)
    local_rank = _int("LOCAL_RANK", rank)
    if world < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world}")
    if not 0 <= rank < world:
        raise ValueError(f"RANK {rank} is outside [0, {world})")
    if not 0 <= local_rank < world:
        raise ValueError(f"LOCAL_RANK {local_rank} is outside [0, {world}) for a "
                         f"{world}-process job")
    if single_node and local_rank != rank:
        raise ValueError(
            f"LOCAL_RANK {local_rank} != RANK {rank}: this trainer is launched with "
            "`torchrun --standalone --nnodes=1`, so every rank is local. A multi-node "
            "launch would need a device map and a rendezvous this script does not set up.")
    return Runtime(rank=rank, world=world, local_rank=local_rank)


def require_devices(world: int, n_visible: int) -> None:
    if n_visible < world:
        raise ValueError(
            f"{world} processes but only {n_visible} visible CUDA device(s). Each rank keeps "
            "its own backbone/reader/optimizer copy, so a shared device would oversubscribe "
            "memory and be slower than the single-GPU path. Ask Slurm for --gpus=<type>:"
            f"{world} on ONE node, or lower NPROC.")


def step_groups(items, batch_size: int, rank: int, world: int):
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if world < 1:
        raise ValueError(f"world must be >= 1, got {world}")
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} is outside [0, {world})")
    items = list(items)
    for start in range(0, len(items), batch_size):
        group = items[start:start + batch_size]
        yield group, group[rank::world]


def n_global_steps(n_items: int, batch_size: int) -> int:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if n_items < 0:
        raise ValueError(f"n_items must be >= 0, got {n_items}")
    return (n_items + batch_size - 1) // batch_size


def shard(items, rank: int, world: int) -> list:
    if world < 1:
        raise ValueError(f"world must be >= 1, got {world}")
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} is outside [0, {world})")
    return list(items)[rank::world]


def coverage(n_items: int, batch_size: int, world: int) -> dict:
    per_rank = [0] * world
    for rank in range(world):
        per_rank[rank] = sum(len(local) for _g, local in
                             step_groups(range(n_items), batch_size, rank, world))
    return {"n_items": n_items, "batch_size": batch_size, "world": world,
            "n_global_steps": n_global_steps(n_items, batch_size),
            "chains_per_rank": per_rank,
            "n_covered": sum(per_rank)}


def combine_mean(total_sum: float, count: float) -> float:
    return total_sum / count if count else 0.0


def digest_int(obj) -> int:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") & ((1 << 62) - 1)


def _all_reduce_int64(values, runtime, *, op: str, device=None) -> list:
    values = [int(v) for v in values]
    if not runtime.distributed or not values:
        return values
    import torch

    dist = _dist()
    ops = {"max": dist.ReduceOp.MAX, "min": dist.ReduceOp.MIN}
    buf = torch.tensor(values, dtype=torch.int64, device=device)
    dist.all_reduce(buf, op=ops[op])
    return [int(v) for v in buf.tolist()]


def param_layout(named_params) -> list:
    return [[name, list(p.shape), str(p.dtype), bool(p.requires_grad)]
            for name, p in named_params]


def _dist():
    import torch.distributed as dist

    return dist


def init_distributed(runtime: Runtime, *, backend: str | None = None,
                     timeout_s: int = DEFAULT_TIMEOUT_S):
    if not runtime.distributed:
        return None
    import datetime

    import torch

    dist = _dist()
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend, rank=runtime.rank, world_size=runtime.world,
                                timeout=datetime.timedelta(seconds=timeout_s))
    return backend


def shutdown() -> None:
    dist = _dist()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(runtime: Runtime) -> None:
    if not runtime.distributed:
        return
    dist = _dist()
    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[runtime.local_rank])
    else:
        dist.barrier()


def all_reduce_floats(values, runtime: Runtime, *, op: str = "sum", device=None) -> list:
    values = [float(v) for v in values]
    if not runtime.distributed or not values:
        return values
    import torch

    dist = _dist()
    ops = {"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX, "min": dist.ReduceOp.MIN}
    if op not in ops:
        raise ValueError(f"op {op!r} not in {sorted(ops)}")
    buf = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(buf, op=ops[op])
    return [float(v) for v in buf.tolist()]


def broadcast_floats(values, runtime: Runtime, *, src: int = 0, device=None) -> list:
    values = [float(v) for v in values]
    if not runtime.distributed or not values:
        return values
    import torch

    buf = torch.tensor(values, dtype=torch.float64, device=device)
    _dist().broadcast(buf, src=src)
    return [float(v) for v in buf.tolist()]


def all_gather_objects(value, runtime: Runtime) -> list:
    if not runtime.distributed:
        return [value]
    dist = _dist()
    out = [None] * runtime.world
    dist.all_gather_object(out, value)
    return out


def broadcast_int(value: int, runtime: Runtime, *, src: int = 0, device=None) -> int:
    if not runtime.distributed:
        return int(value)
    import torch

    buf = torch.tensor([int(value)], dtype=torch.int64, device=device)
    _dist().broadcast(buf, src=src)
    return int(buf.item())


def check_agreement(values: dict, runtime: Runtime, *, device=None) -> dict:
    digests = {key: digest_int(value) for key, value in sorted(values.items())}
    if not runtime.distributed:
        return digests
    lo = _all_reduce_int64(list(digests.values()), runtime, op="min", device=device)
    hi = _all_reduce_int64(list(digests.values()), runtime, op="max", device=device)
    bad = [key for key, a, b in zip(digests, lo, hi) if a != b]
    if bad:
        raise SystemExit(
            f"rank {runtime.rank}: the ranks disagree about {bad}. Every rank must load the "
            "same population, the same flags and the same parameter layout, or the summed "
            "gradient mixes different experiments. Refusing to train.")
    return digests


class GradientSync:

    def __init__(self, named_params, runtime: Runtime, *, device=None):
        self.items = [(name, p) for name, p in named_params if p.requires_grad]
        if not self.items:
            raise ValueError("GradientSync got no trainable parameters")
        self.runtime = runtime
        self.device = device
        self.buckets: list = []
        index = {}
        for i, (_name, p) in enumerate(self.items):
            key = (str(p.device), str(p.dtype))
            if key not in index:
                index[key] = len(self.buckets)
                self.buckets.append((key, []))
            self.buckets[index[key]][1].append(i)

    @property
    def params(self) -> list:
        return [p for _name, p in self.items]

    def layout(self) -> list:
        return param_layout(self.items)

    def reduce(self) -> dict:
        n_local = sum(1 for _n, p in self.items if p.grad is not None)
        if not self.runtime.distributed:
            return {"reduced": False, "world": 1, "n_local_with_grad": n_local,
                    "n_global_with_grad": n_local,
                    "n_globally_unused": len(self.items) - n_local}
        import torch

        dist = _dist()
        presence = torch.tensor([1.0 if p.grad is not None else 0.0 for _n, p in self.items],
                                dtype=torch.float64, device=self.device)
        dist.all_reduce(presence, op=dist.ReduceOp.SUM)
        used = [value > 0.0 for value in presence.tolist()]
        for (_key, idxs) in self.buckets:
            sizes = [self.items[i][1].numel() for i in idxs]
            sample = self.items[idxs[0]][1]
            flat = torch.zeros(sum(sizes), dtype=sample.dtype, device=sample.device)
            offset = 0
            for i, size in zip(idxs, sizes):
                grad = self.items[i][1].grad
                if grad is not None:
                    flat[offset:offset + size].copy_(grad.reshape(-1))
                offset += size
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            offset = 0
            for i, size in zip(idxs, sizes):
                param = self.items[i][1]
                if used[i]:
                    chunk = flat[offset:offset + size].view_as(param)
                    if param.grad is None:
                        param.grad = chunk.detach().clone()
                    else:
                        param.grad.copy_(chunk)
                elif param.grad is not None:
                    raise RuntimeError(f"{self.items[i][0]}: gradient presence mask disagrees "
                                       "with the local gradient")
                offset += size
        n_used = sum(used)
        return {"reduced": True, "world": self.runtime.world, "n_local_with_grad": n_local,
                "n_global_with_grad": n_used, "n_globally_unused": len(self.items) - n_used,
                "n_buckets": len(self.buckets)}


def shard_name(split: str, rank: int, world: int, attempt: str) -> str:
    return f"{split}.w{world:02d}r{rank:02d}.{attempt}.jsonl"


def attempt_token(value: int) -> str:
    return f"{int(value) & ((1 << 62) - 1):016x}"


def write_score_shard(path, *, split: str, rank: int, world: int, attempt: str,
                      rows) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"qkey": row["qkey"], "rep": int(row["rep"]), "scores": row["scores"]}
            for row in rows]
    header = {"schema": SCORE_SHARD_SCHEMA, "split": split, "rank": int(rank),
              "world": int(world), "attempt": attempt, "n_rows": len(rows)}
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(tmp, path)
    return len(rows)


def read_score_shard(path, *, split: str, rank: int, world: int, attempt: str) -> list:
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        lines = [line for line in handle.read().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"score shard {path} is empty")
    header = json.loads(lines[0])
    expected = {"schema": SCORE_SHARD_SCHEMA, "split": split, "rank": int(rank),
                "world": int(world), "attempt": attempt}
    bad = {k: (header.get(k), v) for k, v in expected.items() if header.get(k) != v}
    if bad:
        raise ValueError(f"score shard {path} header mismatch {bad} (found, expected)")
    rows = [json.loads(line) for line in lines[1:]]
    if header.get("n_rows") != len(rows):
        raise ValueError(f"score shard {path} declares {header.get('n_rows')} rows but holds "
                         f"{len(rows)} -- truncated write")
    return rows


def purge_stale_shards(shard_dir, *, split: str, attempt: str) -> list:
    shard_dir = Path(shard_dir)
    if not shard_dir.is_dir():
        return []
    removed = []
    for path in sorted(shard_dir.glob(f"{split}.*.jsonl")):
        if f".{attempt}.jsonl" not in path.name:
            path.unlink()
            removed.append(path.name)
    return removed


def remove_shards(shard_dir, *, split: str) -> list:
    shard_dir = Path(shard_dir)
    if not shard_dir.is_dir():
        return []
    removed = []
    for path in sorted(shard_dir.glob(f"{split}.*.jsonl")):
        path.unlink()
        removed.append(path.name)
    return removed


def merge_score_shards(shard_dir, *, split: str, world: int, attempt: str, chains) -> dict:
    shard_dir = Path(shard_dir)
    chains = list(chains)
    foreign = [p.name for p in sorted(shard_dir.glob(f"{split}.*.jsonl"))
               if f".{attempt}.jsonl" not in p.name]
    if foreign:
        raise ValueError(
            f"refusing to publish {split}: shard file(s) from another attempt are present "
            f"({foreign[:3]}). A stale shard merged into this dump would score some chains "
            "with a different checkpoint's weights.")
    merged: dict = {}
    counts = []
    for rank in range(world):
        path = shard_dir / shard_name(split, rank, world, attempt)
        if not path.exists():
            raise ValueError(
                f"refusing to publish {split}: rank {rank}'s score shard is missing "
                f"({path.name}). That rank either died or never finished scoring; the dump "
                "would be short. Nothing is zero-filled.")
        rows = read_score_shard(path, split=split, rank=rank, world=world, attempt=attempt)
        want = [(c["qkey"], int(c["rep"])) for c in shard(chains, rank, world)]
        got = [(row["qkey"], int(row["rep"])) for row in rows]
        if got != want:
            missing = sorted(set(want) - set(got))
            extra = sorted(set(got) - set(want))
            raise ValueError(
                f"refusing to publish {split}: rank {rank}'s shard covers the wrong chains "
                f"({len(missing)} missing e.g. {missing[:2]}, {len(extra)} unexpected e.g. "
                f"{extra[:2]}, {len(got)} rows for {len(want)} expected). A shard written "
                "under a different world size or a different population cannot be merged.")
        for row in rows:
            key = (row["qkey"], int(row["rep"]))
            if key in merged:
                raise ValueError(f"refusing to publish {split}: chain {key} was scored by two "
                                 "ranks -- one of them scored a chain that is not its own")
            merged[key] = row["scores"]
    n_scores = 0
    for chain in chains:
        key = (chain["qkey"], int(chain["rep"]))
        scores = merged.pop(key, None)
        if scores is None:
            raise ValueError(f"refusing to publish {split}: no rank scored chain {key}")
        if len(scores) != len(chain["probes"]):
            raise ValueError(
                f"refusing to publish {split}: chain {key} came back with {len(scores)} scores "
                f"for {len(chain['probes'])} probes -- the ladder zips them, so a short list "
                "would silently truncate the rule")
        chain["scores"] = scores
        n_scores += len(scores)
    if merged:
        raise ValueError(f"refusing to publish {split}: {len(merged)} scored chain(s) are not "
                         f"in the split (e.g. {sorted(merged)[:2]})")
    return {"split": split, "world": world, "attempt": attempt, "n_chains": len(chains),
            "n_scores": n_scores}


def execution_record(runtime: Runtime, *, backend, digests: dict, **extra) -> dict:
    return {"schema": EXECUTION_SCHEMA, "world_size": runtime.world, "backend": backend,
            "parity_digests": {k: f"{v:016x}" for k, v in sorted(digests.items())},
            **extra}
