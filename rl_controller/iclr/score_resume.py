from __future__ import annotations

import json
import os
from pathlib import Path

SCHEMA = "iclr-score-partial/v1"
PARTIAL_NAME = ".score_pool_partial"

IDENTITY_FIELDS = (
    "run_dir",
    "chains_sha256",
    "questions_sha256",
    "checkpoint_sha256",
    "reader_sha256",
    "targets",
    "head_mode",
    "group_size",
    "chains_per_step",
    "device",
    "commit",
    "dirty",
)
RECORD_FIELDS = frozenset({"qkey", "rep", "scores"})
DEFAULT_EVERY = 512


def groups(chains, size: int) -> list:
    size = max(1, int(size))
    return [chains[i:i + size] for i in range(0, len(chains), size)]


def _norm(value):
    return json.loads(json.dumps(value, sort_keys=True))


def check_identity(identity) -> dict:
    if not isinstance(identity, dict):
        raise ValueError("identity must be a dict")
    missing = [field for field in IDENTITY_FIELDS if field not in identity]
    if missing:
        raise ValueError(f"identity is missing {missing}; an identity that omits a field is an "
                         "identity that cannot refuse on it")
    extra = sorted(set(identity) - set(IDENTITY_FIELDS))
    if extra:
        raise ValueError(f"identity carries unknown field(s) {extra}; add them to "
                         "IDENTITY_FIELDS deliberately or leave them out")
    return {field: _norm(identity[field]) for field in IDENTITY_FIELDS}


def identity_diff(stored, current) -> list:
    if not isinstance(stored, dict):
        return ["the partial declares no identity block"]
    out = []
    for field in IDENTITY_FIELDS:
        if field not in stored:
            out.append(f"{field}: the partial declares none, this run declares "
                       f"{current[field]!r}")
        elif _norm(stored[field]) != current[field]:
            out.append(f"{field}: partial {stored[field]!r} != this run {current[field]!r}")
    return out


def read_partial(path) -> dict:
    data = Path(path).read_bytes()
    identity, records, ends, valid_bytes, torn = None, [], [], 0, None
    header_bytes, start = 0, 0
    while start < len(data):
        end = data.find(b"\n", start)
        if end < 0:
            torn = f"{len(data) - start} trailing byte(s) with no newline"
            break
        line, start = data[start:end], end + 1
        try:
            payload = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            torn = f"record {len(records)} is not JSON ({exc})"
            break
        if identity is None and valid_bytes == 0:
            if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
                return {"identity": None, "records": [], "ends": [], "header_bytes": 0,
                        "valid_bytes": 0,
                        "torn": f"the first line is not an {SCHEMA} header"}
            identity = payload.get("identity")
            header_bytes = valid_bytes = start
            continue
        if not isinstance(payload, dict) or set(payload) != RECORD_FIELDS:
            torn = (f"record {len(records)} has fields "
                    f"{sorted(payload) if isinstance(payload, dict) else type(payload).__name__}"
                    f", not {sorted(RECORD_FIELDS)}")
            break
        records.append(payload)
        ends.append(start)
        valid_bytes = start
    return {"identity": identity, "records": records, "ends": ends,
            "header_bytes": header_bytes, "valid_bytes": valid_bytes, "torn": torn}


def resume_point(records, todo, group_size: int) -> dict:
    if len(records) > len(todo):
        return {"chains": 0, "groups": 0, "matched": 0,
                "refusal": f"the partial holds {len(records)} records for a chain list of "
                           f"{len(todo)}; it was not written for this pass"}
    for position, record in enumerate(records):
        want = (todo[position]["qkey"], todo[position]["rep"])
        got = (record["qkey"], record["rep"])
        if got != want:
            return {"chains": 0, "groups": 0, "matched": position,
                    "refusal": f"record {position} is chain {list(got)}, but this pass scores "
                               f"{list(want)} there; the partial does not describe this "
                               "chain list"}
    size = max(1, int(group_size))
    usable = (len(records) // size) * size
    return {"chains": usable, "groups": usable // size, "matched": len(records), "refusal": None}


class Checkpoint:

    def __init__(self, out_dir, identity, *, every: int = DEFAULT_EVERY,
                 enabled: bool = True, resume: bool = True, fsync: bool = True) -> None:
        self.path = Path(out_dir) / PARTIAL_NAME
        self.identity = check_identity(identity) if enabled else None
        self.every = max(1, int(every))
        self.enabled = bool(enabled)
        self.resume_allowed = bool(resume)
        self.fsync = bool(fsync)
        self.group_size = 1
        self._handle = None
        self._reused = {}
        self._start = 0
        self.report = {"schema": SCHEMA, "started": False, "resumed": False, "one_pass": None,
                       "chains_reused": 0, "chains_scored": 0, "resumed_from_chain": None,
                       "resumed_from_group": None, "group_size": None, "records_discarded": 0,
                       "refusal": None, "torn": None, "persisted": False, "disabled": None,
                       "code_pinned": None}

    @classmethod
    def off(cls, out_dir="."):
        return cls(out_dir, None, enabled=False)


    def start(self, todo, *, group_size: int, extra=None) -> dict:
        self.group_size = max(1, int(group_size))
        self.every = -(-self.every // self.group_size) * self.group_size
        self.report["group_size"] = self.group_size
        self.report["started"] = True
        if not self.enabled:
            self.report["disabled"] = "no identity was supplied to this checkpoint"
            self.report["one_pass"] = True
            return self.report
        self.identity = check_identity({**self.identity, "group_size": self.group_size,
                                        **(extra or {})})
        self.report["code_pinned"] = not self.identity["dirty"]
        found = self._consult(todo)
        self._start = found
        self.report["chains_reused"] = found
        self.report["resumed"] = found > 0
        if found:
            self.report["resumed_from_chain"] = found
            self.report["resumed_from_group"] = found // self.group_size
        self.report["one_pass"] = not self.report["resumed"]
        self._open(truncate_to=self._truncate_to)
        return self.report

    def _consult(self, todo) -> int:
        self._truncate_to = None
        if not self.path.exists():
            return 0
        if not self.resume_allowed:
            self.report["refusal"] = "--no-resume: an existing partial was discarded unread"
            return 0
        try:
            found = read_partial(self.path)
        except OSError as exc:
            self.report["refusal"] = f"the partial could not be read ({exc})"
            return 0
        self.report["torn"] = found["torn"]
        if found["identity"] is None:
            self.report["refusal"] = found["torn"] or "the partial declares no identity block"
            return 0
        differences = identity_diff(found["identity"], self.identity)
        if differences:
            self.report["refusal"] = ("the partial was written by a different pass: "
                                      + "; ".join(differences))
            return 0
        point = resume_point(found["records"], todo, self.group_size)
        if point["refusal"]:
            self.report["refusal"] = point["refusal"]
            return 0
        self.report["records_discarded"] = point["matched"] - point["chains"]
        self._reused = {(r["qkey"], r["rep"]): r["scores"]
                        for r in found["records"][:point["chains"]]}
        usable = point["chains"]
        self._truncate_to = found["ends"][usable - 1] if usable else found["header_bytes"]
        return usable

    def _open(self, *, truncate_to) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if truncate_to is None:
                handle = open(self.path, "wb")
                handle.write(json.dumps({"schema": SCHEMA, "identity": self.identity},
                                        sort_keys=True, separators=(",", ":")).encode() + b"\n")
            else:
                handle = open(self.path, "r+b")
                handle.truncate(truncate_to)
                handle.seek(truncate_to)
            self._flush(handle)
            self._handle = handle
            self.report["persisted"] = True
        except OSError as exc:
            self._disable(f"the partial could not be opened ({exc})")


    def reuse(self, todo) -> int:
        for chain in todo[:self._start]:
            chain["scores"] = self._reused[(chain["qkey"], chain["rep"])]
        return self._start

    def slabs(self, todo) -> list:
        return [todo[i:i + self.every] for i in range(self._start, len(todo), self.every)]

    def record(self, chains) -> None:
        self.report["chains_scored"] += len(chains)
        if self._handle is None:
            return
        try:
            payload = b"".join(
                json.dumps({"qkey": c["qkey"], "rep": c["rep"], "scores": c["scores"]},
                           separators=(",", ":")).encode() + b"\n" for c in chains)
            self._handle.write(payload)
            self._flush(self._handle)
        except (OSError, ValueError) as exc:
            self._disable(f"the partial could not be written ({exc})")

    def _flush(self, handle) -> None:
        handle.flush()
        if self.fsync:
            os.fsync(handle.fileno())

    def _disable(self, why: str) -> None:
        self.report["disabled"] = why
        self.report["persisted"] = False
        try:
            if self._handle is not None:
                self._handle.close()
        except OSError:
            pass
        self._handle = None

    def finish(self) -> dict:
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            self.report["disabled"] = f"the partial could not be removed ({exc})"
        return self.report


def run(checkpoint: Checkpoint, todo, *, group_size: int, score_slab, extra=None,
        after_slab=None) -> dict:
    report = checkpoint.start(todo, group_size=group_size, extra=extra)
    done = checkpoint.reuse(todo)
    for slab in checkpoint.slabs(todo):
        score_slab(slab)
        done += len(slab)
        checkpoint.record(slab)
        if after_slab is not None:
            after_slab(done, len(todo))
    report["chains_total"] = len(todo)
    return report
