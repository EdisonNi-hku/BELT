from __future__ import annotations

import json
from pathlib import Path

SCHEMA_FEATURES = "iclr-block-features/v1"
SCHEMA_LABELS = "iclr-label-dump/v1"

KEY_FIELDS = ("qkey", "rep", "end_index")


def row_key(row: dict) -> tuple:
    missing = [k for k in KEY_FIELDS if k not in row]
    if missing:
        raise ValueError(f"row lacks join key fields {missing}: {sorted(row)[:8]}")
    return (row["qkey"], int(row["rep"]), int(row["end_index"]))


def assert_same_pool(feature_manifest: dict, label_manifest: dict) -> str:
    if feature_manifest.get("schema") != SCHEMA_FEATURES:
        raise ValueError(f"feature manifest schema {feature_manifest.get('schema')!r} "
                         f"!= {SCHEMA_FEATURES!r}")
    if label_manifest.get("schema") != SCHEMA_LABELS:
        raise ValueError(f"label manifest schema {label_manifest.get('schema')!r} "
                         f"!= {SCHEMA_LABELS!r}")
    fsha = feature_manifest["chains"]["sha256"]
    lsha = label_manifest["chains"]["sha256"]
    if fsha != lsha:
        raise ValueError(
            f"features were built from chains sha256 {fsha} but the labels from {lsha}. "
            "Equal qkeys do not prove equal chains: joining these would pair every feature "
            "with another generation's label and fail silently.")
    return fsha


SCRATCH_PURGE_DAYS = 15


def check_cache_freshness(manifest: dict, *, warn_days: int = 10,
                          require_files: bool = True) -> dict:
    import time

    paths = []
    for shard in manifest.get("shards", ()):
        for entry in (shard.get("payloads") or {}).values():
            paths.append(entry["path"])
        if shard.get("index_file"):
            paths.append(shard["index_file"]["path"])
        for entry in (shard.get("prompt_payloads") or {}).values():
            paths.append(entry["path"])
    missing, ages = [], []
    now = time.time()
    for path in paths:
        handle = Path(path)
        if not handle.exists():
            missing.append(path)
            continue
        ages.append((now - handle.stat().st_mtime) / 86400.0)
    if missing and require_files:
        raise FileNotFoundError(
            f"{len(missing)} of {len(paths)} cache files are missing (first "
            f"{missing[:2]}). Personal scratch purges files older than ~"
            f"{SCRATCH_PURGE_DAYS} days; regenerate with the array spec recorded in "
            "docs/superpowers/contracts/2026-09-01-block-feature-cache-regeneration.md "
            "rather than training on a partial cache.")
    oldest = max(ages) if ages else None
    return {"n_files": len(paths), "n_missing": len(missing),
            "oldest_age_days": oldest,
            "purge_risk": bool(oldest is not None and oldest >= warn_days),
            "purge_days": SCRATCH_PURGE_DAYS, "missing": missing[:10]}


def load_index(manifest_path, *, check_freshness: bool = True) -> tuple[dict, list[dict]]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if check_freshness:
        manifest["_freshness"] = check_cache_freshness(manifest)
    rows = []
    for shard in manifest["shards"]:
        index_path = Path(shard["index_file"]["path"])
        if not index_path.is_absolute():
            index_path = manifest_path.parent / index_path.name
        n = 0
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["shard"] = shard["index"]
            rows.append(row)
            n += 1
        if n != shard["n_rows"]:
            raise ValueError(f"shard {shard['index']} index has {n} rows but its manifest "
                             f"says {shard['n_rows']}")
    if len(rows) != manifest["n_rows"]:
        raise ValueError(f"index has {len(rows)} rows, manifest says {manifest['n_rows']}")
    return manifest, rows


def label_table(label_rows, target: str) -> dict:
    table = {}
    for row in label_rows:
        if target not in row["labels"]:
            raise ValueError(f"label dump has no target {target!r}; it carries "
                             f"{sorted(row['labels'])}")
        key = row_key(row)
        if key in table:
            raise ValueError(f"duplicate label row for {key}")
        table[key] = row["labels"][target]
    return table


def join(index_rows, label_rows, target: str, *, keys=None) -> dict:
    table = label_table(label_rows, target)
    rows, labels, out_keys = [], [], []
    n_candidate = n_masked = 0
    for row in index_rows:
        key = row_key(row)
        if keys is not None and key[0] not in keys:
            continue
        n_candidate += 1
        if key not in table:
            raise ValueError(
                f"feature row {key} has no label row. Features and labels are built from the "
                "same chains over the same trainable population, so a missing key means the "
                "join is misaligned, not that the boundary is unlabelled.")
        value = table[key]
        if value is None:
            n_masked += 1
            continue
        rows.append((row["shard"], row["row"]))
        labels.append(float(value))
        out_keys.append(key)
    return {"rows": rows, "labels": labels, "keys": out_keys,
            "n_candidate": n_candidate, "n_labelled": len(rows), "n_masked": n_masked}


def split_keys(index_rows, val_qkeys) -> tuple[set, set]:
    present = {row["qkey"] for row in index_rows}
    val = {q for q in val_qkeys if q in present}
    missing = set(val_qkeys) - present
    if missing:
        raise ValueError(f"{len(missing)} val questions are absent from this pool "
                         f"(first {sorted(missing)[:3]}); the shared val split must be a "
                         "subset of the fit pool or honest selection is not comparable")
    return present - val, val


def val_qkeys_from_scored(path) -> list[str]:
    seen, out = set(), []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        qkey = json.loads(line)["qkey"]
        if qkey not in seen:
            seen.add(qkey)
            out.append(qkey)
    return out


def coverage_report(joined: dict, target: str) -> dict:
    n = joined["n_candidate"]
    return {
        "target": target,
        "n_candidate_boundaries": n,
        "n_labelled": joined["n_labelled"],
        "n_masked": joined["n_masked"],
        "coverage": (joined["n_labelled"] / n) if n else None,
        "base_rate": (sum(joined["labels"]) / len(joined["labels"]))
                     if joined["labels"] else None,
        "n_questions": len({k[0] for k in joined["keys"]}),
        "n_chains": len({(k[0], k[1]) for k in joined["keys"]}),
    }
