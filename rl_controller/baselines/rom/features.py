from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "iclr-block-features/v1"

POOLINGS = ("mean", "last")


@dataclass(frozen=True)
class FeatureSpec:

    layer: int
    pooling: str = "last"

    def __post_init__(self):
        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {self.pooling!r}")

    @property
    def key(self) -> str:
        return f"layer{self.layer}_{self.pooling}"

    def to_dict(self) -> dict:
        return {"layer": self.layer, "pooling": self.pooling, "key": self.key}

    @classmethod
    def parse(cls, text: str) -> "FeatureSpec":
        parts = str(text).split(":")
        layer = int(parts[0])
        return cls(layer=layer, pooling=parts[1] if len(parts) > 1 else "last")


class FeatureCacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class CacheManifest:

    schema_version: str
    pool: str
    chains_sha256: str
    model_id: str
    model_revision: str
    hidden_size: int
    dtype: str
    layers: tuple
    poolings: tuple
    has_prompt_features: bool
    shards: tuple
    commit: str = ""
    path: str = ""

    @classmethod
    def from_dict(cls, row: dict, *, path: str = "") -> "CacheManifest":
        missing = [k for k in ("schema_version", "pool", "chains_sha256", "hidden_size",
                               "layers") if k not in row]
        if missing:
            raise FeatureCacheError(
                f"block-feature manifest {path or '<dict>'} lacks {missing}. Expected the "
                f"{SCHEMA} header: schema_version, pool, chains_sha256, model_id, "
                "model_revision, hidden_size, dtype, layers, poolings, "
                "has_prompt_features, shards[{path, sha256}].")
        return cls(
            schema_version=str(row["schema_version"]),
            pool=str(row["pool"]),
            chains_sha256=str(row["chains_sha256"]),
            model_id=str(row.get("model_id", "")),
            model_revision=str(row.get("model_revision", "")),
            hidden_size=int(row["hidden_size"]),
            dtype=str(row.get("dtype", "float16")),
            layers=tuple(int(v) for v in row["layers"]),
            poolings=tuple(str(v) for v in row.get("poolings", POOLINGS)),
            has_prompt_features=bool(row.get("has_prompt_features", False)),
            shards=tuple(row.get("shards", ())),
            commit=str(row.get("commit", "")),
            path=path,
        )

    def supports(self, spec: FeatureSpec) -> bool:
        return spec.layer in self.layers and spec.pooling in self.poolings

    def require(self, spec: FeatureSpec) -> None:
        if not self.supports(spec):
            raise FeatureCacheError(
                f"cache {self.path or self.pool} has layers {list(self.layers)} x "
                f"poolings {list(self.poolings)}; it cannot serve {spec.key}")

    def provenance(self) -> dict:
        return {"features_schema": self.schema_version, "features_pool": self.pool,
                "features_chains_sha256": self.chains_sha256,
                "features_model_id": self.model_id,
                "features_model_revision": self.model_revision,
                "features_hidden_size": self.hidden_size,
                "features_dtype": self.dtype,
                "features_commit": self.commit,
                "features_has_prompt": self.has_prompt_features}


class BlockFeatureCache:

    manifest: CacheManifest

    def has(self, qkey: str, rep: int) -> bool:
        raise NotImplementedError

    def _rows(self, qkey: str, rep: int, spec: FeatureSpec):
        raise NotImplementedError

    def _prompt(self, qkey: str, spec: FeatureSpec):
        return None


    def chain_matrix(self, chain: dict, spec: FeatureSpec):
        self.manifest.require(spec)
        qkey, rep = chain["qkey"], chain["rep"]
        if not self.has(qkey, rep):
            raise FeatureCacheError(f"no cached features for chain {(qkey, rep)} "
                                    f"in {self.manifest.path or self.manifest.pool}")
        rows = self._rows(qkey, rep, spec)
        ends = chain["block_ends"]
        if len(rows) != len(ends):
            raise FeatureCacheError(
                f"chain {(qkey, rep)}: cache has {len(rows)} block vectors but the chain has "
                f"{len(ends)} block_ends. These MUST align 1:1 and in order -- an off-by-one "
                "would shift every label by one block.")
        width = self.manifest.hidden_size
        for index, row in enumerate(rows):
            if len(row) != width:
                raise FeatureCacheError(
                    f"chain {(qkey, rep)} block {index}: vector width {len(row)} != "
                    f"manifest hidden_size {width}")
        return rows

    def prompt_matrix(self, chain: dict, spec: FeatureSpec):
        if not self.manifest.has_prompt_features:
            return None
        self.manifest.require(spec)
        rows = self._prompt(chain["qkey"], spec)
        if rows is None:
            raise FeatureCacheError(
                f"manifest declares has_prompt_features but chain {chain['qkey']} has none")
        return rows


class InMemoryBlockFeatures(BlockFeatureCache):

    def __init__(self, manifest: CacheManifest, rows: dict, prompts: dict | None = None):
        self.manifest = manifest
        self._table = rows
        self._prompts = prompts or {}

    def has(self, qkey: str, rep: int) -> bool:
        return (qkey, rep) in self._table

    def _rows(self, qkey: str, rep: int, spec: FeatureSpec):
        table = self._table[(qkey, rep)]
        if spec.key not in table:
            raise FeatureCacheError(f"chain {(qkey, rep)} has no {spec.key} vectors")
        return table[spec.key]

    def _prompt(self, qkey: str, spec: FeatureSpec):
        return self._prompts.get(qkey, {}).get(spec.key)


class ShardedBlockFeatures(BlockFeatureCache):

    VARIANT_AXIS = {"mean": 0, "last": 1}

    def __init__(self, manifest: CacheManifest, root, *, positions: str = "all",
                 index_cache=None, verbose: bool = False):
        self.manifest = manifest
        self.root = Path(root)
        self.positions = positions
        self._payloads: dict = {}
        self._prompt_payloads: dict = {}
        self._chain_index: dict = {}
        self._prompt_index: dict = {}
        self._shard_stems: list = []
        self._load_index(index_cache, verbose)


    def _stems(self) -> list:
        stems = []
        for entry in self.manifest.shards:
            if isinstance(entry, dict):
                for key in ("stem", "shard_stem", "name"):
                    if entry.get(key):
                        stems.append(str(entry[key]))
                        break
                else:
                    path = entry.get("index", {}).get("path") if isinstance(
                        entry.get("index"), dict) else entry.get("path")
                    if path:
                        stems.append(Path(str(path)).name.split(".index.jsonl")[0])
            else:
                stems.append(str(entry))
        if not stems:
            stems = sorted(p.name[: -len(".index.jsonl")]
                           for p in self.root.glob("*.index.jsonl"))
        if not stems:
            raise FeatureCacheError(f"no shard index files under {self.root}")
        return stems

    def _load_index(self, index_cache, verbose: bool) -> None:
        if index_cache and Path(index_cache).exists():
            blob = json.loads(Path(index_cache).read_text(encoding="utf-8"))
            if blob.get("chains_sha256") == self.manifest.chains_sha256:
                self._shard_stems = blob["stems"]
                self._chain_index = {tuple(json.loads(k)): tuple(v)
                                     for k, v in blob["chains"].items()}
                self._prompt_index = {k: tuple(v) for k, v in blob["prompts"].items()}
                return
        self._shard_stems = self._stems()
        for shard, stem in enumerate(self._shard_stems):
            path = self.root / f"{stem}.index.jsonl"
            if not path.exists():
                raise FeatureCacheError(f"missing shard index {path}")
            current = None
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    ident = (row["qkey"], int(row["rep"]))
                    end_index, payload_row = int(row["end_index"]), int(row["row"])
                    if ident != current:
                        if ident in self._chain_index:
                            raise FeatureCacheError(
                                f"chain {ident} appears in two places in the cache index")
                        if end_index != 0:
                            raise FeatureCacheError(
                                f"chain {ident} starts at end_index {end_index}, expected 0")
                        self._chain_index[ident] = (shard, payload_row, 1)
                        current = ident
                    else:
                        prev_shard, row_start, count = self._chain_index[ident]
                        if end_index != count or payload_row != row_start + count:
                            raise FeatureCacheError(
                                f"chain {ident}: index row {end_index}/{payload_row} breaks the "
                                f"contiguity the reader relies on (start {row_start}, seen {count})")
                        self._chain_index[ident] = (prev_shard, row_start, count + 1)
            prompt_path = self.root / f"{stem}.prompt_index.jsonl"
            if self.manifest.has_prompt_features and prompt_path.exists():
                with open(prompt_path, encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            row = json.loads(line)
                            self._prompt_index[row["qkey"]] = (shard, int(row["row"]),
                                                               int(row["n_rows"]))
            if verbose:
                print(f"[rom-features] shard {shard + 1}/{len(self._shard_stems)} indexed "
                      f"({len(self._chain_index)} chains)", flush=True)
        if index_cache:
            Path(index_cache).parent.mkdir(parents=True, exist_ok=True)
            Path(index_cache).write_text(json.dumps({
                "chains_sha256": self.manifest.chains_sha256,
                "stems": self._shard_stems,
                "chains": {json.dumps(list(k)): list(v) for k, v in self._chain_index.items()},
                "prompts": {k: list(v) for k, v in self._prompt_index.items()}}),
                encoding="utf-8")


    def _payload(self, shard: int, layer: int, prompt: bool = False):
        import numpy as np

        table = self._prompt_payloads if prompt else self._payloads
        key = (shard, layer)
        if key not in table:
            suffix = f".L{layer}.prompt.npy" if prompt else f".L{layer}.npy"
            path = self.root / f"{self._shard_stems[shard]}{suffix}"
            if not path.exists():
                raise FeatureCacheError(f"missing payload {path}")
            table[key] = np.load(path, mmap_mode="r")
        return table[key]

    def has(self, qkey: str, rep: int) -> bool:
        return (qkey, int(rep)) in self._chain_index

    def _rows(self, qkey: str, rep: int, spec: FeatureSpec):
        shard, row_start, count = self._chain_index[(qkey, int(rep))]
        payload = self._payload(shard, spec.layer)
        axis = self.VARIANT_AXIS[spec.pooling]
        return payload[row_start:row_start + count, axis, :]

    def _prompt(self, qkey: str, spec: FeatureSpec):
        entry = self._prompt_index.get(qkey)
        if entry is None:
            return None
        shard, row, n_rows = entry
        payload = self._payload(shard, spec.layer, prompt=True)
        axis = self.VARIANT_AXIS[spec.pooling]
        return payload[row, :n_rows, axis, :]

    def stats(self) -> dict:
        return {"n_chains": len(self._chain_index), "n_shards": len(self._shard_stems),
                "n_prompts": len(self._prompt_index), "pool": self.manifest.pool,
                "positions": self.positions}


def load_manifest(path) -> CacheManifest:
    path = Path(path)
    if not path.exists():
        raise FeatureCacheError(f"no block-feature manifest at {path}")
    return CacheManifest.from_dict(json.loads(path.read_text(encoding="utf-8")), path=str(path))


def open_cache(manifest_path, *, positions: str = "all", index_cache=None,
               verbose: bool = False) -> BlockFeatureCache:
    manifest = load_manifest(manifest_path)
    if manifest.schema_version != SCHEMA:
        raise FeatureCacheError(
            f"{manifest_path} declares schema {manifest.schema_version!r}; this reader "
            f"implements {SCHEMA!r}. Update rl_controller/baselines/rom/features.py -- do NOT "
            "reinterpret the bytes.")
    return ShardedBlockFeatures(manifest, Path(manifest_path).parent, positions=positions,
                                index_cache=index_cache, verbose=verbose)
