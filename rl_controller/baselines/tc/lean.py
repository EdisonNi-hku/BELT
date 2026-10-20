from __future__ import annotations

import json
from array import array
from math import isnan, nan
from pathlib import Path


def stream_chains_lite(path, *, keep_untrainable: bool = True):
    from rl_controller.iclr.chains import is_trainable

    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row.pop("completion_ids", None)
            trainable = is_trainable(row)
            if not trainable and not keep_untrainable:
                continue
            row["_trainable"] = trainable
            row["scores"] = []
            out.append(row)
    return out


class LeanIndex:

    __slots__ = ("shard", "row", "end_index", "probe_index", "chain_of", "chains",
                 "_chain_id", "n_rows")

    def __init__(self):
        self.shard = array("i")
        self.row = array("i")
        self.end_index = array("i")
        self.probe_index = array("i")
        self.chain_of = array("i")
        self.chains: list = []
        self._chain_id: dict = {}
        self.n_rows = 0

    def _cid(self, qkey, rep):
        key = (qkey, int(rep))
        cid = self._chain_id.get(key)
        if cid is None:
            cid = len(self.chains)
            self._chain_id[key] = cid
            self.chains.append(key)
        return cid

    def add(self, row: dict, shard: int) -> None:
        self.shard.append(int(shard))
        self.row.append(int(row["row"]))
        self.end_index.append(int(row["end_index"]))
        pi = row.get("probe_index")
        self.probe_index.append(-1 if pi is None else int(pi))
        self.chain_of.append(self._cid(row["qkey"], row["rep"]))
        self.n_rows += 1

    def qkey(self, i: int) -> str:
        return self.chains[self.chain_of[i]][0]

    def address(self, i: int) -> tuple:
        return (self.shard[i], self.row[i])


def load_lean_index(manifest_path, *, check_freshness: bool = True) -> tuple[dict, LeanIndex]:
    from rl_controller.baselines.tc import dataset as ds

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if check_freshness:
        manifest["_freshness"] = ds.check_cache_freshness(manifest)
    index = LeanIndex()
    for shard in manifest["shards"]:
        path = Path(shard["index_file"]["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path.name
        n = 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                index.add(json.loads(line), shard["index"])
                n += 1
        if n != shard["n_rows"]:
            raise ValueError(f"shard {shard['index']} index has {n} rows, manifest says "
                             f"{shard['n_rows']}")
    if index.n_rows != manifest["n_rows"]:
        raise ValueError(f"index has {index.n_rows} rows, manifest says {manifest['n_rows']}")
    return manifest, index


def load_label_arrays(dump_path, targets) -> tuple[dict, dict]:
    import hashlib

    from rl_controller.iclr import label_dump as ld

    manifest = json.loads(ld.manifest_path(dump_path).read_text(encoding="utf-8"))
    missing = [t for t in targets if t not in manifest["targets"]]
    if missing:
        raise SystemExit(f"{dump_path} carries {manifest['targets']}, not {missing}")
    data_path = Path(manifest["output"]["path"])
    if not data_path.is_absolute():
        data_path = Path(dump_path).parent / data_path.name
    digest = hashlib.sha256()
    per_chain: dict = {}
    with open(data_path, "rb") as raw:
        for bline in raw:
            digest.update(bline)
            if not bline.strip():
                continue
            row = json.loads(bline)
            key = (row["qkey"], int(row["rep"]))
            table = per_chain.get(key)
            if table is None:
                table = per_chain[key] = {t: array("d") for t in targets}
            ei = int(row["end_index"])
            for t in targets:
                col = table[t]
                while len(col) <= ei:
                    col.append(nan)
                value = row["labels"][t]
                if value is not None:
                    col[ei] = float(value)
    got = digest.hexdigest()
    if got != manifest["output"]["sha256"]:
        raise SystemExit(f"{data_path}: sha256 {got} != the sha the manifest certifies "
                         f"{manifest['output']['sha256']}")
    return manifest, per_chain


def training_rows(index: LeanIndex, labels: dict, target: str, qkeys: set,
                  *, probed_only: bool = False):
    addresses, y = [], []
    for i in range(index.n_rows):
        if probed_only and index.probe_index[i] < 0:
            continue
        key = index.chains[index.chain_of[i]]
        if key[0] not in qkeys:
            continue
        table = labels.get(key)
        if table is None:
            continue
        col = table[target]
        ei = index.end_index[i]
        if ei >= len(col):
            continue
        value = col[ei]
        if isnan(value):
            continue
        addresses.append((index.shard[i], index.row[i]))
        y.append(value)
    return addresses, y


def probe_rows_by_chain(index: LeanIndex) -> dict:
    out: dict = {}
    for i in range(index.n_rows):
        pi = index.probe_index[i]
        if pi < 0:
            continue
        out.setdefault(index.chains[index.chain_of[i]], []).append(
            (int(pi), int(index.shard[i]), int(index.row[i])))
    for key, rows in out.items():
        rows.sort()
        if [p for p, _, _ in rows] != list(range(len(rows))):
            raise SystemExit(f"chain {key}: cached probe indices {[p for p, _, _ in rows][:8]} "
                             "are not a complete 0..n-1 run")
    return out
