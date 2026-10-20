from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from rl_controller.iclr.chains import load_chains
from rl_controller.iclr.provenance import parse_and_record


def without_scores(chain: dict) -> dict:
    return {key: value for key, value in chain.items() if key != "scores"}


def non_score_bytes(chain: dict) -> bytes:
    return json.dumps(
        without_scores(chain), separators=(",", ":"), sort_keys=True,
        ensure_ascii=False, allow_nan=False).encode("utf-8")


def jsonl_bytes(rows) -> bytes:
    return b"".join(
        (json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def write_immutable_jsonl(root, *components: str, payload: bytes) -> dict:
    for component in components:
        if not component or component in (".", "..") or Path(component).name != component:
            raise ValueError(
                f"validation-view component {component!r} is not a single path component; "
                "the derived dump's location must be built from names, never from a path "
                "a caller assembled")
    digest = hashlib.sha256(payload).hexdigest()
    directory = Path(root).joinpath(*components)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"val_scored.{digest}.jsonl"
    if destination.exists():
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(
                f"immutable validation view {destination} has sha256 {actual}, expected "
                f"{digest}. A content-addressed view is written once; delete it and re-run "
                "rather than repairing it in place.")
        return {"path": str(destination.resolve()), "sha256": digest}

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".val_scored.", suffix=".tmp",
                dir=directory, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"path": str(destination.resolve()), "sha256": digest}


def align_validation_run(run: dict, reference: dict, *, output_root, components,
                         reference_record: dict) -> dict:
    by_identity: dict[tuple[str, int], dict] = {}
    for chain in run["val"]:
        identity = (str(chain["qkey"]), int(chain["rep"]))
        if identity in by_identity:
            raise RuntimeError(f"{run['path']}: duplicate val chain identity {identity}")
        by_identity[identity] = chain

    aligned = []
    for identity in sorted(reference):
        if identity not in by_identity:
            raise RuntimeError(
                f"{run['path']}: reference validation identity {identity} is absent. The "
                "reference population is not a subset of this run's val split, so these "
                "numbers cannot be computed on it; re-generate the run over the reference "
                "bank rather than projecting onto part of it.")
        candidate = by_identity[identity]
        if non_score_bytes(candidate) != non_score_bytes(reference[identity]):
            raise RuntimeError(
                f"{run['path']}: val chain {identity} has a non-score payload that differs "
                "from the authenticated reference validation chain. Sharing (qkey, rep) with "
                "the reference is not sharing a chain bank -- the two were generated "
                "differently, and their scores are not comparable per question.")
        aligned.append(candidate)
    if not aligned:
        raise RuntimeError(
            f"{run['path']}: the reference validation population is empty for this run, so "
            "there is nothing to fit or select on")

    payload = jsonl_bytes(aligned)
    source = write_immutable_jsonl(output_root, *components, payload=payload)
    parsed, parsed_record = parse_and_record(source["path"], load_chains)
    if parsed_record["sha256"] != source["sha256"]:
        raise RuntimeError(
            f"derived validation view {source['path']} changed before it was consumed")
    aligned = parsed
    source = {"path": parsed_record["path"], "sha256": parsed_record["sha256"]}
    result = dict(run)
    result["val"] = aligned
    result["dumps"] = dict(run["dumps"])
    result["dumps"]["val"] = source
    result["validation_basis"] = {
        "kind": "exact_uhead_chain_identity_projection",
        "reference": dict(reference_record),
        "original_source": dict(run["dumps"]["val"]),
        "aligned_source": dict(source),
        "n_chains": len(aligned),
        "n_questions": len({chain["qkey"] for chain in aligned}),
    }
    return result


def reference_validation_basis(table_path) -> dict:
    from rl_controller.iclr import pareto as pa

    table_path = Path(table_path).resolve(strict=True)
    table_bytes = table_path.read_bytes()
    table_sha256 = hashlib.sha256(table_bytes).hexdigest()
    payload = json.loads(table_bytes)
    pa.load_verified_question_tables(str(table_path))
    if hashlib.sha256(table_path.read_bytes()).hexdigest() != table_sha256:
        raise RuntimeError(
            f"reference validation table changed while it was being verified: {table_path}")

    sources = payload.get("sources") or {}
    identities = payload.get("identities") or {}
    seeds = sorted(set(sources) & set(identities))
    if not seeds or set(sources) != set(identities):
        raise RuntimeError(
            f"{table_path}: reference has missing or inconsistent source/identity seeds")

    reference_by_seed: dict[str, dict] = {}
    source_records: dict[str, dict] = {}
    expected_identities = None
    for seed in seeds:
        try:
            source = sources[seed]["val"]
            declared = identities[seed]["val"]
        except (KeyError, TypeError):
            raise RuntimeError(
                f"{table_path}: reference seed {seed} has no authenticated val source") from None
        chains, record = parse_and_record(source["path"], load_chains)
        if record["sha256"] != source.get("sha256"):
            raise RuntimeError(
                f"{table_path}: reference seed {seed} val source changed after table "
                "verification")
        by_identity = {(str(chain["qkey"]), int(chain["rep"])): chain for chain in chains}
        if len(by_identity) != len(chains):
            raise RuntimeError(
                f"{table_path}: reference seed {seed} has duplicate val identities")
        declared_set = {(str(qkey), int(rep)) for qkey, rep in declared}
        if set(by_identity) != declared_set:
            raise RuntimeError(
                f"{table_path}: reference seed {seed} val identities do not match its source "
                "bytes")
        if expected_identities is None:
            expected_identities = declared_set
        elif declared_set != expected_identities:
            raise RuntimeError(
                f"{table_path}: reference seeds do not share the exact val chain identities, "
                "so there is no one population to project onto")
        reference_by_seed[seed] = by_identity
        source_records[seed] = {"path": record["path"], "sha256": record["sha256"],
                                "n_chains": len(chains)}

    reference_seed = seeds[0]
    reference = reference_by_seed[reference_seed]
    for seed in seeds[1:]:
        for identity in sorted(reference):
            if (non_score_bytes(reference[identity]) !=
                    non_score_bytes(reference_by_seed[seed][identity])):
                raise RuntimeError(
                    f"{table_path}: reference seeds disagree on val chain {identity}'s "
                    "non-score payload")

    record = {
        "reference_table": str(table_path),
        "table_sha256": table_sha256,
        "reference_seed": reference_seed,
        "reference_seeds": seeds,
        "reference_sources": source_records,
    }
    return {"chains": reference, "chains_by_seed": reference_by_seed, "record": record}
