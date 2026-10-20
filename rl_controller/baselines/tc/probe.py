from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

VARIANT_INDEX = {"mean": 0, "last": 1}


def group_by_shard(addresses) -> dict:
    out: dict = {}
    for position, (shard, row) in enumerate(addresses):
        out.setdefault(int(shard), []).append((position, int(row)))
    return out


def batches(n: int, size: int, *, order=None):
    if size < 1:
        raise ValueError(f"batch size must be >= 1, got {size}")
    idx = list(range(n)) if order is None else list(order)
    if len(idx) != n:
        raise ValueError(f"order has {len(idx)} entries for {n} rows")
    for start in range(0, n, size):
        yield idx[start: start + size]


def permutation(n: int, seed: int) -> list:
    import random

    rng = random.Random(f"iclr-ws-e-tc:{seed}")
    idx = list(range(n))
    rng.shuffle(idx)
    return idx


def payload_paths(manifest: dict, layer_key: str) -> dict:
    out = {}
    for shard in manifest["shards"]:
        entry = shard["payloads"].get(layer_key)
        if entry is None:
            raise ValueError(f"shard {shard['index']} has no payload for {layer_key!r}; "
                             f"it carries {sorted(shard['payloads'])}")
        out[int(shard["index"])] = entry["path"]
    return out


def _canonical_signs_reference(components) -> list:
    out = []
    for row in components:
        best = max(range(len(row)), key=lambda j: (abs(row[j]), -j))
        flip = -1.0 if row[best] < 0 else 1.0
        out.append([flip * x for x in row])
    return out


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pca_record(*, k: int, hidden: int, n_rows: int, layer_key: str, variant: str,
               explained: float, path, sha: str, train_questions: int) -> dict:
    return {
        "kind": "pca",
        "k": int(k),
        "hidden": int(hidden),
        "layer": layer_key,
        "variant": variant,
        "n_train_rows": int(n_rows),
        "n_train_questions": int(train_questions),
        "explained_variance_ratio": explained,
        "fitted_on": "train split of the fit pool only (never val/select/OOD)",
        "path": str(path),
        "sha256": sha,
    }


class FeatureReader:

    def __init__(self, manifest: dict, layer_key: str, variant: str = "mean"):
        import numpy as np

        if variant not in VARIANT_INDEX:
            raise ValueError(f"variant must be one of {sorted(VARIANT_INDEX)}, got {variant!r}")
        self._np = np
        self.variant = variant
        self.column = VARIANT_INDEX[variant]
        self.layer_key = layer_key
        self.hidden = int(manifest["hidden_size"])
        self._paths = payload_paths(manifest, layer_key)
        self._open: dict = {}

    def _payload(self, shard: int):
        if shard not in self._open:
            self._open[shard] = self._np.load(self._paths[shard], mmap_mode="r")
        return self._open[shard]

    def gather(self, addresses):
        np = self._np
        out = np.empty((len(addresses), self.hidden), dtype=np.float32)
        for shard, items in group_by_shard(addresses).items():
            payload = self._payload(shard)
            positions = [p for p, _ in items]
            rows = [r for _, r in items]
            out[positions] = payload[rows, self.column, :].astype(np.float32)
        return out


def materialize(reader: FeatureReader, addresses, *, batch: int = 65536, dtype=None):
    import numpy as np

    if dtype is None:
        dtype = np.float16
    order = sorted(range(len(addresses)), key=lambda i: addresses[i])
    X = np.empty((len(addresses), reader.hidden), dtype=dtype)
    for chunk in batches(len(order), batch):
        idx = [order[i] for i in chunk]
        X[idx] = reader.gather([addresses[i] for i in idx]).astype(dtype)
    return X


def project(pca: dict, X, *, batch: int = 65536):
    import numpy as np

    out = np.empty((X.shape[0], pca["k"]), dtype=np.float32)
    for chunk in batches(X.shape[0], batch):
        lo, hi = chunk[0], chunk[-1] + 1
        out[lo:hi] = transform(pca, X[lo:hi].astype(np.float32))
    return out


def fit_pca_matrix(X, *, k: int = 256, batch: int = 65536):
    import numpy as np

    hidden = X.shape[1]
    total = np.zeros(hidden, dtype=np.float64)
    gram = np.zeros((hidden, hidden), dtype=np.float64)
    n = 0
    for chunk in batches(X.shape[0], batch):
        lo, hi = chunk[0], chunk[-1] + 1
        B = X[lo:hi].astype(np.float64)
        total += B.sum(axis=0)
        gram += B.T @ B
        n += B.shape[0]
    if n <= k:
        raise ValueError(f"PCA needs more rows than components: {n} rows, k={k}")
    mean = total / n
    cov = gram / n - np.outer(mean, mean)
    cov = (cov + cov.T) / 2.0
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1][:k]
    components = vectors[:, order].T
    for i in range(components.shape[0]):
        j = int(np.argmax(np.abs(components[i])))
        if components[i, j] < 0:
            components[i] *= -1.0
    return {"mean": mean.astype(np.float32), "components": components.astype(np.float32),
            "k": int(k), "hidden": hidden, "n_rows": int(n),
            "explained_variance_ratio": float(values[order].sum() / max(values.sum(), 1e-12))}


def fit_logistic_matrix(Z, y, *, seed: int, epochs: int = 8, batch: int = 8192,
                        lr: float = 1e-2, l2: float = 1e-4, class_weight: bool = False,
                        device: str = "cpu"):
    import numpy as np
    import torch

    if Z.shape[0] != len(y):
        raise ValueError(f"{Z.shape[0]} rows for {len(y)} labels")
    g = torch.Generator().manual_seed(seed)
    w = torch.zeros(Z.shape[1], 1, dtype=torch.float32)
    torch.nn.init.normal_(w, std=0.01, generator=g)
    w = w.to(device).requires_grad_(True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    y_all = np.asarray(y, dtype=np.float32)
    pos = float(y_all.mean()) if len(y_all) else 0.0
    pos_weight = (torch.tensor([(1.0 - pos) / pos], device=device)
                  if class_weight and 0.0 < pos < 1.0 else None)
    history = []
    for epoch in range(epochs):
        order = permutation(Z.shape[0], seed * 1000 + epoch)
        total, seen = 0.0, 0
        for chunk in batches(Z.shape[0], batch, order=order):
            idx = np.asarray(chunk)
            X = torch.from_numpy(Z[idx]).to(device)
            t = torch.from_numpy(y_all[idx]).unsqueeze(1).to(device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                X @ w + b, t, pos_weight=pos_weight)
            loss = loss + l2 * (w * w).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss) * len(chunk); seen += len(chunk)
        history.append(total / max(seen, 1))
    return {"weight": w.detach().cpu().numpy().reshape(-1), "bias": float(b.detach().cpu()),
            "seed": int(seed), "epochs": epochs, "loss_history": history,
            "base_rate": pos, "class_weighted": pos_weight is not None}


def save_probe(probe: dict, path) -> str:
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp.npz"
    np.savez(tmp, weight=probe["weight"],
             bias=np.asarray([probe["bias"]], dtype=np.float64),
             meta=np.asarray([json.dumps({k: probe[k] for k in
                                          ("seed", "epochs", "loss_history", "base_rate",
                                           "class_weighted")})], dtype=object))
    os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_probe(path) -> dict | None:
    import numpy as np

    path = Path(path)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"][0]))
        return {"weight": data["weight"], "bias": float(data["bias"][0]), **meta}
    except Exception:
        return None


def load_pca_safe(path):
    try:
        return load_pca(path)
    except Exception:
        return None


def save_scores(table: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"{k[0]}|{k[1]}|{pi}": v for k, rows in table.items() for pi, v in rows.items()}
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(flat, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def load_scores(path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        flat = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    out: dict = {}
    for key, value in flat.items():
        qkey, rep, pi = key.rsplit("|", 2)
        out.setdefault((qkey, int(rep)), {})[int(pi)] = float(value)
    return out


def predict_matrix(probe: dict, Z) -> list:
    import numpy as np

    z = Z @ probe["weight"] + probe["bias"]
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float64).tolist()


def fit_pca(reader: FeatureReader, addresses, *, k: int = 256, batch: int = 4096):
    import numpy as np

    hidden = reader.hidden
    total = np.zeros(hidden, dtype=np.float64)
    gram = np.zeros((hidden, hidden), dtype=np.float64)
    n = 0
    for chunk in batches(len(addresses), batch):
        X = reader.gather([addresses[i] for i in chunk]).astype(np.float64)
        total += X.sum(axis=0)
        gram += X.T @ X
        n += X.shape[0]
    if n <= k:
        raise ValueError(f"PCA needs more rows than components: {n} rows, k={k}")
    mean = total / n
    cov = gram / n - np.outer(mean, mean)
    cov = (cov + cov.T) / 2.0
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1][:k]
    components = vectors[:, order].T
    for i in range(components.shape[0]):
        j = int(np.argmax(np.abs(components[i])))
        if components[i, j] < 0:
            components[i] *= -1.0
    kept = float(values[order].sum() / max(values.sum(), 1e-12))
    return {"mean": mean.astype(np.float32), "components": components.astype(np.float32),
            "k": int(k), "hidden": hidden, "n_rows": int(n),
            "explained_variance_ratio": kept}


def save_pca(pca: dict, path) -> str:
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=pca["mean"], components=pca["components"],
             n_rows=np.asarray([int(pca.get("n_rows", 0))]),
             explained=np.asarray([float(pca.get("explained_variance_ratio", float("nan")))]))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pca(path, *, sha256: str | None = None) -> dict:
    import numpy as np

    path = Path(path)
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha256 and got != sha256:
        raise ValueError(f"{path}: sha256 {got} != expected {sha256}")
    data = np.load(path)
    files = set(data.files)
    return {"mean": data["mean"], "components": data["components"],
            "k": int(data["components"].shape[0]), "hidden": int(data["components"].shape[1]),
            "n_rows": int(data["n_rows"][0]) if "n_rows" in files else 0,
            "explained_variance_ratio": (float(data["explained"][0])
                                         if "explained" in files else None),
            "sha256": got}


def transform(pca: dict, X):
    return (X - pca["mean"]) @ pca["components"].T


def fit_logistic(reader: FeatureReader, addresses, labels, pca, *, seed: int,
                 epochs: int = 8, batch: int = 4096, lr: float = 1e-2,
                 l2: float = 1e-4, class_weight: bool = False):
    import numpy as np
    import torch

    if len(addresses) != len(labels):
        raise ValueError(f"{len(addresses)} addresses for {len(labels)} labels")
    g = torch.Generator().manual_seed(seed)
    k = pca["k"]
    w = torch.zeros(k, 1, dtype=torch.float32)
    torch.nn.init.normal_(w, std=0.01, generator=g)
    w.requires_grad_(True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=0.0)
    y_all = np.asarray(labels, dtype=np.float32)
    pos = float(y_all.mean()) if len(y_all) else 0.0
    pos_weight = None
    if class_weight and 0.0 < pos < 1.0:
        pos_weight = torch.tensor([(1.0 - pos) / pos], dtype=torch.float32)
    history = []
    for epoch in range(epochs):
        order = permutation(len(addresses), seed * 1000 + epoch)
        total, seen = 0.0, 0
        for chunk in batches(len(addresses), batch, order=order):
            X = reader.gather([addresses[i] for i in chunk])
            Z = torch.from_numpy(np.ascontiguousarray(transform(pca, X)))
            y = torch.from_numpy(y_all[chunk]).unsqueeze(1)
            logits = Z @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight)
            loss = loss + l2 * (w * w).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(chunk)
            seen += len(chunk)
        history.append(total / max(seen, 1))
    return {"weight": w.detach().numpy().reshape(-1), "bias": float(b.detach()),
            "seed": int(seed), "epochs": epochs, "loss_history": history,
            "base_rate": pos, "class_weighted": pos_weight is not None}


def predict(probe: dict, pca: dict, X):
    import numpy as np

    z = transform(pca, X) @ probe["weight"] + probe["bias"]
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float64).tolist()


def probe_record(probe: dict, *, target: str, layer_key: str, variant: str) -> dict:
    return {"kind": "logistic", "target": target, "layer": layer_key, "variant": variant,
            "seed": probe["seed"], "epochs": probe["epochs"],
            "final_loss": probe["loss_history"][-1] if probe["loss_history"] else None,
            "base_rate": probe["base_rate"], "class_weighted": probe["class_weighted"],
            "n_params": int(len(probe["weight"]) + 1),
            "sha256": sha256_bytes(json.dumps(
                {"w": [float(x) for x in probe["weight"]], "b": probe["bias"]},
                sort_keys=True).encode("utf-8"))}
