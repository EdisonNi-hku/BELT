from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from rl_controller.iclr import targets as tg
from rl_controller.iclr.chains import is_correct, is_trainable, probe_draws, validate_chain
from rl_controller.single_cot import probe_positions_in_ends

SCHEMA = "iclr-label-dump/v1"

TARGET_SETS: dict[str, tuple[str, ...]] = {
    "primary": ("q", "v", "vpi0", "r"),
    "secondary": ("q", "v"),
    "rom": ("q", "v", "vpi0", "r", "o"),
    "tc": ("q", "v", "vpi0", "r", "stability"),
}

Q_LABEL_MODES = ("single", "multi_draw_mean")


def resolve_targets(spec: str) -> tuple[str, tuple[str, ...]]:
    if spec in TARGET_SETS:
        return spec, tg.parse_targets(",".join(TARGET_SETS[spec]))
    return "custom", tg.parse_targets(spec)


def _q_labels(chain: dict, *, mode: str = "single", draws: int = 0) -> list[float]:
    if mode not in Q_LABEL_MODES:
        raise ValueError(f"q label mode {mode!r} not in {Q_LABEL_MODES}")
    if draws < 0:
        raise ValueError(f"draws must be >= 0, got {draws}")
    gold = chain["gold"]
    if mode == "single":
        return [float(is_correct(p["forced"], gold)) for p in chain["probes"]]
    out = []
    for p in chain["probes"]:
        ds = probe_draws(p)
        if draws > 0:
            ds = ds[:draws]
        out.append(sum(is_correct(d["forced"], gold) for d in ds) / len(ds))
    return out


def chain_labels(chain: dict, names, *, q_mode: str = "single", q_draws: int = 0) -> dict[str, list]:
    return {name: (_q_labels(chain, mode=q_mode, draws=q_draws) if name == "q"
                   else tg.labels(name, chain))
            for name in names}


def chain_rows(chain: dict, names, *, q_mode: str = "single", q_draws: int = 0) -> list[dict]:
    ends = [int(e) for e in chain["block_ends"]]
    if [qp[1] for qp in chain["query_points"]] != [p["prefix_len"] for p in chain["probes"]]:
        raise ValueError(f"chain {(chain['qkey'], chain['rep'])}: query_points prefixes do not "
                         "align with probes[].prefix_len")
    probe_pos = probe_positions_in_ends(ends, chain["query_points"])
    at_probe = {pos: i for i, pos in enumerate(probe_pos)}
    if len(at_probe) != len(probe_pos):
        raise ValueError(f"chain {(chain['qkey'], chain['rep'])}: two probes share a block end "
                         f"(probe positions {probe_pos}) — one boundary cannot carry two probe "
                         "labels (the trainer refuses this chain too)")
    labs = chain_labels(chain, names, q_mode=q_mode, q_draws=q_draws)
    for name in names:
        spec = tg.CATALOGUE[name]
        want = len(chain["probes"]) if spec.positions == "probes" else len(ends)
        if len(labs[name]) != want:
            raise ValueError(f"chain {(chain['qkey'], chain['rep'])}: target {name!r} produced "
                             f"{len(labs[name])} labels for {want} {spec.positions} positions")
    rows = []
    for end_index, prefix_len in enumerate(ends):
        probe_index = at_probe.get(end_index)
        probe = chain["probes"][probe_index] if probe_index is not None else None
        values = {}
        for name in names:
            spec = tg.CATALOGUE[name]
            if spec.positions == "ends":
                values[name] = labs[name][end_index]
            else:
                values[name] = labs[name][probe_index] if probe_index is not None else None
        rows.append({
            "qkey": chain["qkey"], "rep": chain["rep"],
            "end_index": end_index, "prefix_len": prefix_len,
            "probe_index": probe_index,
            "block": (probe["block"] if probe is not None else None),
            "labels": values,
        })
    return rows


def dump_rows(chains, names, *, q_mode: str = "single", q_draws: int = 0) -> list[dict]:
    rows = []
    for chain in chains:
        if not is_trainable(chain):
            continue
        rows.extend(chain_rows(chain, names, q_mode=q_mode, q_draws=q_draws))
    return rows


def label_stats(rows, names) -> dict:
    out = {}
    for name in names:
        spec = tg.CATALOGUE[name]
        vals = [r["labels"][name] for r in rows if r["labels"][name] is not None]
        entry = {"n_labelled": len(vals), "n_rows": len(rows)}
        if spec.kind == "softmax":
            counts: dict[str, int] = {}
            for v in vals:
                counts[str(int(v))] = counts.get(str(int(v)), 0) + 1
            entry["class_counts"] = counts
        else:
            entry["mean"] = (sum(vals) / len(vals)) if vals else None
        out[name] = entry
    return out


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fire_table_hashed(path) -> tuple[dict, str]:
    from rl_controller.iclr.ladder import load_fire_table

    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    handle = tempfile.NamedTemporaryFile(suffix=".fire.jsonl", delete=False)
    try:
        with handle:
            handle.write(raw)
        return load_fire_table(handle.name), digest
    finally:
        Path(handle.name).unlink(missing_ok=True)


def load_chains_hashed(path, *, keep_completion_ids: bool = False) -> tuple[list[dict], str]:
    digest = hashlib.sha256()
    rows, seen = [], set()
    with open(path, "rb") as handle:
        for raw in handle:
            digest.update(raw)
            if not raw.strip():
                continue
            chain = validate_chain(json.loads(raw.decode("utf-8")))
            identity = (chain["qkey"], chain["rep"])
            if identity in seen:
                raise ValueError(f"duplicate chain identity {identity}")
            seen.add(identity)
            if not keep_completion_ids:
                chain.pop("completion_ids", None)
            rows.append(chain)
    return rows, digest.hexdigest()


def serialise(rows) -> bytes:
    return b"".join(json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n"
                    for row in rows)


def _atomic_write(target: Path, data: bytes) -> None:
    handle = tempfile.NamedTemporaryFile(dir=str(target.parent), delete=False,
                                         prefix=target.name + ".", suffix=".tmp")
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def dump_data_path(out_path, payload_sha: str) -> Path:
    out_path = Path(out_path)
    return out_path.with_name(f"{out_path.stem}.{payload_sha[:12]}{out_path.suffix}")


def publish(out_path, payload: bytes, manifest_builder) -> tuple[Path, Path]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload_sha = hashlib.sha256(payload).hexdigest()
    data_path = dump_data_path(out_path, payload_sha)
    _atomic_write(data_path, payload)
    manifest = manifest_path(out_path)
    _atomic_write(manifest, manifest_builder(data_path))
    return data_path, manifest


def read_dump(out_path) -> tuple[dict, list[dict]]:
    record = json.loads(manifest_path(out_path).read_text(encoding="utf-8"))
    data_path = Path(record["output"]["path"])
    if not data_path.is_absolute():
        data_path = Path(out_path).parent / data_path.name
    raw = data_path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != record["output"]["sha256"]:
        raise ValueError(f"{data_path}: sha256 {got} does not match the manifest's "
                         f"{record['output']['sha256']}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return record, rows


def manifest(*, target_set: str, names, chains_path, chains_sha: str, fire_path, fire_sha,
             out_path, out_sha: str, rows, chains, git_commit, git_dirty,
             options: dict) -> dict:
    n_trainable = sum(1 for c in chains if is_trainable(c))
    return {
        "schema": SCHEMA,
        "target_set": target_set,
        "targets": list(names),
        "chains": {"path": str(chains_path), "sha256": chains_sha},
        "pi0_fire": (None if fire_path is None
                     else {"path": str(fire_path), "sha256": fire_sha}),
        "output": {"path": str(out_path), "sha256": out_sha},
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "n_chains": len(chains),
        "n_trainable_chains": n_trainable,
        "n_skipped_chains": len(chains) - n_trainable,
        "n_questions": len({c["qkey"] for c in chains if is_trainable(c)}),
        "n_rows": len(rows),
        "n_probed_rows": sum(1 for r in rows if r["probe_index"] is not None),
        "options": dict(options),
        "label_stats": label_stats(rows, names),
    }


def crosscheck(rows, chains, names) -> dict:
    by_id = {(c["qkey"], c["rep"]): c for c in chains}
    seen = {}
    for c in chains:
        if not is_trainable(c):
            continue
        ends = [int(e) for e in c["block_ends"]]
        pos = probe_positions_in_ends(ends, c["query_points"])
        seen[(c["qkey"], c["rep"])] = {p: i for i, p in enumerate(pos)}
    mismatches = []
    n_values = 0
    for row in rows:
        ident = (row["qkey"], row["rep"])
        chain = by_id[ident]
        probe_index = seen[ident].get(row["end_index"])
        if probe_index != row["probe_index"]:
            mismatches.append({**{k: row[k] for k in ("qkey", "rep", "end_index")},
                               "target": "_probe_index", "dump": row["probe_index"],
                               "expected": probe_index})
        for name in names:
            spec = tg.CATALOGUE[name]
            table = chain["_labels"][name]
            if spec.positions == "ends":
                expected = table[row["end_index"]]
            else:
                expected = table[probe_index] if probe_index is not None else None
            n_values += 1
            if row["labels"][name] != expected:
                if len(mismatches) < 20:
                    mismatches.append({**{k: row[k] for k in ("qkey", "rep", "end_index")},
                                       "target": name, "dump": row["labels"][name],
                                       "expected": expected})
    return {"n_rows": len(rows), "n_values": n_values,
            "n_mismatch": len(mismatches), "mismatches": mismatches[:20]}


def dump_name(pool: str, target_set: str) -> str:
    return f"labels_{pool}_{target_set}.jsonl"


def manifest_path(out_path) -> Path:
    out = Path(out_path)
    return out.with_name(out.stem + ".manifest.json")
