from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rl_controller.iclr import targets as tg
from rl_controller.iclr.chains import is_correct, is_trainable, validate_chain

POPULATIONS = ("all", "trainable")


class BalanceAccumulator:

    def __init__(self, drop_never_correct: bool = False):
        self.drop_never_correct = bool(drop_never_correct)
        self.n_chains = 0
        self.n_chains_seen = 0
        self.n_probes = 0
        self.n_fcs_first = 0
        self.n_fcs_undefined = 0
        self.n_fcs_later = 0
        self.primary_sum = 0.0
        self.n_primary = 0
        self.post_fcs_sum = 0.0
        self.n_labelled = 0
        self.n_pos = 0
        self.qkeys: set[str] = set()

    def add(self, chain: dict) -> None:
        self.n_chains_seen += 1
        probes = chain["probes"]
        if not probes:
            return
        self.n_chains += 1
        self.qkeys.add(chain["qkey"])
        gold = chain["gold"]
        y_now = [is_correct(p["forced"], gold) for p in probes]
        k = tg.first_correct_probe(chain, y_now)
        self.n_probes += len(probes)
        if k is None:
            self.n_fcs_undefined += 1
        elif k == 0:
            self.n_fcs_first += 1
        else:
            self.n_fcs_later += 1
        primary = tg._post_fcs(chain, y_now)
        for value in primary:
            self.primary_sum += float(value)
            self.n_primary += 1
        if self.drop_never_correct:
            marked = dict(chain)
            marked[tg.ROM_DROP_FLAG] = True
            active = tg._post_fcs(marked, y_now)
        else:
            active = primary
        for value in active:
            if value is None:
                continue
            self.n_labelled += 1
            self.post_fcs_sum += float(value)
            self.n_pos += int(value == 1.0)


    @property
    def n_neg(self) -> int:
        return self.n_labelled - self.n_pos

    def pos_weight(self) -> float | None:
        if self.n_pos == 0 or self.n_neg == 0:
            return None
        return self.n_neg / self.n_pos

    def oversample_target(self) -> dict:
        major, minor = self.n_fcs_first, self.n_fcs_later
        if minor == 0:
            return {"applicable": False, "reason": "no chains with fcs_index >= 1 to oversample",
                    "majority": major, "minority": minor, "ratio": None}
        if minor >= major:
            return {"applicable": False, "reason": "minority already >= majority; ratio would be <= 1",
                    "majority": major, "minority": minor, "ratio": major / minor}
        return {"applicable": True, "reason": None, "majority": major, "minority": minor,
                "ratio": major / minor}

    def summary(self) -> dict:
        chains = max(1, self.n_chains)
        return {
            "n_chains_seen": self.n_chains_seen,
            "n_probed_chains": self.n_chains,
            "n_questions": len(self.qkeys),
            "n_probes": self.n_probes,
            "fcs_first_probe_chain_fraction": self.n_fcs_first / chains,
            "fcs_undefined_chain_fraction": self.n_fcs_undefined / chains,
            "post_fcs": (self.primary_sum / self.n_primary) if self.n_primary else None,
            "post_fcs_defined": ((self.post_fcs_sum / self.n_labelled)
                                 if self.n_labelled else None),
            "n_fcs_first": self.n_fcs_first,
            "n_fcs_later": self.n_fcs_later,
            "n_fcs_undefined": self.n_fcs_undefined,
            "n_labelled_boundaries": self.n_labelled,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "pos_weight_R1": self.pos_weight(),
            "oversample_R2": self.oversample_target(),
            "drop_never_correct": self.drop_never_correct,
        }


def oversample_multiplicities(chain_keys_by_group: dict, *, seed: int) -> dict:
    major = list(chain_keys_by_group.get("major", []))
    minor = list(chain_keys_by_group.get("minor", []))
    if not minor or len(minor) >= len(major):
        return {key: 1 for key in major + minor}
    base = len(major) // len(minor)
    remainder = len(major) - base * len(minor)
    ranked = sorted(minor, key=lambda key: hashlib.blake2b(
        f"rom-oversample:{seed}:{key[0]}:{key[1]}".encode(), digest_size=8).digest())
    out = {key: 1 for key in major}
    for index, key in enumerate(ranked):
        out[key] = base + (1 if index < remainder else 0)
    return out


def stream_chains(path):
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not JSON: {exc}") from None
            row.pop("completion_ids", None)
            row.setdefault("scores", [{"q": 0.0, "v": 0.0} for _ in row.get("probes", [])])
            yield validate_chain(row)


def measure(path, *, drop_never_correct: bool = False) -> dict:
    accs = {name: BalanceAccumulator(drop_never_correct) for name in POPULATIONS}
    for chain in stream_chains(path):
        accs["all"].add(chain)
        if is_trainable(chain):
            accs["trainable"].add(chain)
    return {name: acc.summary() for name, acc in accs.items()}


def file_digest(path, *, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def report(path, *, drop_never_correct: bool = False, with_sha: bool = True) -> dict:
    path = Path(path)
    out = {
        "schema_version": "iclr-rom-class-balance/v1",
        "chains": str(path),
        "chains_sha256": file_digest(path) if with_sha else None,
        "populations": measure(path, drop_never_correct=drop_never_correct),
    }
    return out
