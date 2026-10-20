from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
P = Path("/cluster/project/sachan/njingwei/iclr")
E = Path("/cluster/scratch/jingni/iclr")
NF_ROOT = P / "runs/belt_ext/natural_fallback/3537690"
BANK_OF = {"math_t1_report": "math_t1_report", "math_t2_report": "math_t2_report",
           "math_ood_report": "math_ood_report", "ood2_math_report": "ood2_math_report",
           "select": "math_union_select", "val": "math_union_fit",
           "olmo3_report": "olmo3_report", "olmo3_fit": "olmo3_fit"}
POOLS = ("math_t1_report", "math_t2_report", "math_ood_report", "ood2_math_report")

_STATE: dict = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def git_head(root) -> dict:
    def run(*cmd):
        return subprocess.run(cmd, cwd=root, capture_output=True, text=True).stdout.strip()
    mod = [l.split(maxsplit=1)[1] for l in run("git", "status", "--porcelain", "--untracked-files=no").splitlines() if l.strip()]
    return {"root": str(root), "commit": run("git", "rev-parse", "HEAD"),
            "dirty_tracked": bool(mod), "modified": mod,
            "dirty_code": [m for m in mod if not m.startswith("tests/")]}


def setup(producer: str, needs=("deployment_cost",)) -> dict:
    producer = str(Path(producer).resolve())
    sys.path.insert(0, producer)
    import rl_controller
    if not str(Path(rl_controller.__file__).resolve()).startswith(producer):
        raise SystemExit(f"rl_controller resolved to {rl_controller.__file__}, not {producer}")
    import rl_controller.scripts
    fc = _load_by_path("d53_fallback_cost", HERE / "fallback_cost.py")
    patch = fc.install()
    missing = [n for n in needs if not patch["charges"].get(n)]
    if missing:
        raise SystemExit(f"after the patch {missing} still do not charge natural_fallback_tokens "
                         f"at {producer}: {patch['charges']}")
    fbb = _load_by_path("rl_controller.scripts.belt_fallback_bank",
                        REPO / "rl_controller/scripts/belt_fallback_bank.py")
    fbd = _load_by_path("rl_controller.scripts.belt_fallback_dump",
                        REPO / "rl_controller/scripts/belt_fallback_dump.py")
    _STATE.update(fbb=fbb, fbd=fbd, fallback={})
    rec = {"producer": git_head(producer), "d53_code": git_head(REPO), "fallback_cost": patch,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "slurm_array": [os.environ.get("SLURM_ARRAY_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")],
           "host": os.uname().nodename}
    if rec["producer"]["dirty_code"]:
        raise SystemExit(f"producer checkout {producer} has tracked modifications outside tests/: "
                         f"{rec['producer']['dirty_code']}")
    _STATE["setup"] = rec
    return rec


def fallback_for(bank: str, nf_root=None) -> tuple[dict, dict]:
    nf_root = Path(nf_root or os.environ.get("D53_NF_ROOT", NF_ROOT))
    path = nf_root / bank / "natural_fallback.jsonl"
    if bank not in _STATE["fallback"]:
        _STATE["fallback"][bank] = (_STATE["fbb"].load_fallback(path),
                                    {"path": str(path), "sha256": sha_file(path)})
    return _STATE["fallback"][bank]


def read_rows(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def corrected(rows, bank: str) -> tuple[list[dict], dict]:
    coverage = "subset" if bank in ("val", "math_union_fit", "olmo3_fit") else "questions"
    bank = BANK_OF.get(bank, bank)
    fb, fb_rec = fallback_for(bank)
    new, stats = _STATE["fbd"].correct_all(rows, fb, coverage=coverage, accept_precorrected=True)
    h = hashlib.sha256()
    for r in new:
        h.update((json.dumps(r) + "\n").encode())
    return new, {**stats, "fallback": fb_rec, "corrected_sha256": h.hexdigest()}


def load_dump(path, bank: str, loader=None) -> dict:
    if loader is None:
        from rl_controller.iclr.chains import load_chains as loader
    t = time.time()
    rows = loader(str(path))
    fb_rows, stats = corrected(rows, bank)
    if stats["n_precorrected"]:
        drop = ("answer_raw", "correct_raw", "natural_fallback_tokens")
        rows = [dict({k: v for k, v in r.items() if k not in drop}, answer=r["answer_raw"], correct=r["correct_raw"])
                for r in fb_rows]
    rec = {"path": str(path), "sha256": sha_file(path), "bank": BANK_OF.get(bank, bank), "n_rows": len(rows), **stats}
    log(f"loaded {path} ({len(rows)} rows, {stats['n_fallback_applied']} corrected, "
        f"{time.time() - t:.0f}s)")
    return {"orig": rows, "fb": fb_rows, "record": rec}


def write_shard(out_dir, name: str, rows_by_variant: dict, manifest: dict) -> None:
    out_dir = Path(out_dir)
    files = {}
    for variant, rows in rows_by_variant.items():
        d = out_dir / variant
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{name}.json.gz"
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as f:
            json.dump({"columns": ["strict_accuracy", "thinking_tokens", "total_tokens"],
                       "variant": variant, "rows": rows}, f, separators=(",", ":"))
        tmp.rename(p)
        files[variant] = {"path": str(p), "sha256": sha_file(p), "n_rows": len(rows)}
    (out_dir / "manifests").mkdir(parents=True, exist_ok=True)
    doc = {"schema": "d53-shard/v1", "name": name, "setup": _STATE.get("setup"), "files": files,
           **manifest}
    (out_dir / "manifests" / f"{name}.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    log(f"wrote shard {name}: " + ", ".join(f"{v}={f['n_rows']}" for v, f in files.items()))


def load_archived(path) -> dict:
    d = json.load(gzip.open(path, "rt"))
    return {(r["method"], r["member"], r["seed"], r["split"]): r["q"] for r in d["rows"]}


def gate(rows: list[dict], archived: dict, *, keys_filter=None) -> dict:
    got = {(r["method"], r["member"], r["seed"], r["split"]): r["q"] for r in rows}
    want = {k: v for k, v in archived.items() if keys_filter is None or keys_filter(k)}
    missing = sorted(set(want) - set(got), key=str)
    extra = sorted(set(got) - set(want), key=str)
    n_vals = n_exact = 0
    worst = 0.0
    worst_at = None
    qmismatch = []
    for k in set(want) & set(got):
        a, b = got[k], want[k]
        if set(a) != set(b):
            qmismatch.append(str(k))
            continue
        for q in b:
            for x, y in zip(a[q], b[q]):
                n_vals += 1
                if x == y:
                    n_exact += 1
                else:
                    d = abs(float(x) - float(y))
                    if d > worst:
                        worst, worst_at = d, (str(k), q)
    return {"n_series": len(set(want) & set(got)), "n_values": n_vals, "n_exact": n_exact,
            "max_abs_diff": worst, "worst_at": worst_at, "missing": [str(k) for k in missing][:20],
            "n_missing": len(missing), "extra": [str(k) for k in extra][:20], "n_extra": len(extra),
            "qkey_set_mismatch": qmismatch[:20],
            "exact": not missing and not qmismatch and n_vals == n_exact and n_vals > 0}


def table_rows(table: dict, *, method: str, seed, split: str) -> list[dict]:
    out = []
    for member, per_q in table.items():
        out.append({"method": method, "member": member, "seed": seed, "split": split,
                    "q": {q: [x[1] / x[0], x[2] / x[0], (x[3] + x[4]) / x[0]] for q, x in per_q.items()}})
    return out


SPLIT_NAME = {"val": "dev", "select": "pool", "dev": "dev", "pool": "pool"}
