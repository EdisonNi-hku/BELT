"""Score fit-generation chain files with a frozen probe runtime."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import secrets
import stat
from pathlib import Path
from typing import Any, Sequence


MANIFEST_SHA256 = "efcacbe076e51c1739be9a1266e6d1176786eaa93f2cc1da08a46dddc9f9e1ae"
PUBLIC_INTERNAL_FIELDS = {"_prompt_ids", "_ends", "_probe_pos", "_trainable"}
CHAIN_FIELDS = {
    "qkey", "rep", "gold", "outcome", "answer", "correct", "blocks",
    "thinking_tokens", "completion_ids", "block_ends", "query_points", "probes",
}
PROBE_FIELDS = {"block", "prefix_len", "forced", "forced_tokens"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_payload(
    payload: bytes, *, label: str,
) -> tuple[list[dict[str, Any]], str]:
    if type(payload) is not bytes or type(label) is not str or not label:
        raise TypeError("JSONL payload and label must be exact bytes/string")
    digest = hashlib.sha256(payload).hexdigest()
    rows: list[dict[str, Any]] = []
    try:
        text = payload.decode("utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("row is not an object")
            rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"invalid JSONL snapshot at {label}:{locals().get('number', 0)}"
        ) from error
    return rows, digest


def read_jsonl_snapshot(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read JSONL snapshot: {path}") from error
    return read_jsonl_payload(payload, label=str(path))


def load_checkpoint_snapshot(path: Path, torch_module) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read S1 checkpoint snapshot: {path}") from error
    try:
        loaded = torch_module.load(
            io.BytesIO(payload), map_location="cpu", weights_only=True
        )
    except TypeError as error:
        raise ValueError("installed torch must support weights_only=True checkpoint loading") from error
    return validate_checkpoint_payload(loaded), hashlib.sha256(payload).hexdigest()


def verify_source_hashes(paths: list[Path], expected_sha256s: list[str]) -> None:
    if len(paths) != len(expected_sha256s):
        raise ValueError("source hash cardinality mismatch")
    for path, expected in zip(paths, expected_sha256s):
        if sha256_file(path) != expected:
            raise ValueError(f"source changed during scoring: {path}")


def _exact_int(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an exact integer >= {minimum}")
    return value


def _validate_manifest_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from rl_controller.splits import question_key

    by_qkey: dict[str, dict[str, Any]] = {}
    for row in rows:
        qkey = row.get("qkey")
        if type(qkey) is not str or not qkey:
            raise ValueError("manifest qkey must be a nonempty exact string")
        if qkey in by_qkey:
            raise ValueError(f"duplicate manifest qkey: {qkey}")
        if row.get("split") not in {"fit", "select", "report"}:
            raise ValueError(f"invalid manifest split for {qkey}")
        _exact_int(row.get("target_chains"), field="manifest target_chains", minimum=1)
        gold = row.get("gold")
        if isinstance(gold, bool) or type(gold) not in (int, str):
            raise ValueError(f"manifest gold must be exact int or string for {qkey}")
        question = row.get("question")
        if type(question) is not str or not question:
            raise ValueError(f"manifest question must be a nonempty exact string for {qkey}")
        if question_key(question) != qkey:
            raise ValueError(f"manifest question/qkey mismatch: {qkey}")
        by_qkey[qkey] = row
    return by_qkey


def load_manifest_payload(
    payload: bytes, *, expected_sha256: str = MANIFEST_SHA256,
) -> dict[str, dict[str, Any]]:
    rows, observed = read_jsonl_payload(payload, label="<held-manifest>")
    if observed != expected_sha256:
        raise ValueError(f"manifest sha256 mismatch: expected={expected_sha256} observed={observed}")
    return _validate_manifest_rows(rows)


def load_manifest(path: Path, *, expected_sha256: str = MANIFEST_SHA256) -> dict[str, dict[str, Any]]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read JSONL snapshot: {path}") from error
    return load_manifest_payload(payload, expected_sha256=expected_sha256)


def _validate_chain(row: dict[str, Any], manifest_row: dict[str, Any]) -> None:
    if set(row) != CHAIN_FIELDS:
        raise ValueError(
            f"chain schema must contain exactly the public generation fields: "
            f"missing={sorted(CHAIN_FIELDS - row.keys())} extra={sorted(row.keys() - CHAIN_FIELDS)}"
        )
    qkey = row.get("qkey")
    if type(qkey) is not str or not qkey:
        raise ValueError("chain qkey must be a nonempty exact string")
    rep = _exact_int(row.get("rep"), field="chain rep")
    if rep >= manifest_row["target_chains"]:
        raise ValueError(f"chain rep outside manifest range: {(qkey, rep)}")
    if type(row.get("gold")) is not type(manifest_row["gold"]) or row["gold"] != manifest_row["gold"]:
        raise ValueError(f"chain gold mismatch: {(qkey, rep)}")
    if type(row.get("outcome")) is not str:
        raise ValueError(f"outcome must be an exact string: {(qkey, rep)}")
    answer = row.get("answer")
    if answer is not None and type(answer) not in (int, str):
        raise ValueError(f"answer must be exact int, string, or null: {(qkey, rep)}")
    if type(row.get("correct")) is not bool:
        raise ValueError(f"correct must be an exact bool: {(qkey, rep)}")
    expected_correct = (
        answer is not None
        and type(answer) is type(manifest_row["gold"])
        and answer == manifest_row["gold"]
    )
    if row["correct"] is not expected_correct:
        raise ValueError(f"correct contradicts exact answer/gold equality: {(qkey, rep)}")
    _exact_int(row.get("blocks"), field="blocks")
    thinking_tokens = _exact_int(row.get("thinking_tokens"), field="thinking_tokens")
    completion_ids = row.get("completion_ids")
    if not isinstance(completion_ids, list) or any(type(token) is not int or token < 0 for token in completion_ids):
        raise ValueError(f"completion_ids must contain exact nonnegative integers: {(qkey, rep)}")
    if thinking_tokens != len(completion_ids):
        raise ValueError(f"thinking_tokens/completion_ids mismatch: {(qkey, rep)}")
    block_ends = row.get("block_ends")
    if not isinstance(block_ends, list) or any(type(end) is not int or end < 1 for end in block_ends):
        raise ValueError(f"block_ends must contain exact positive integers: {(qkey, rep)}")
    if block_ends != sorted(set(block_ends)) or (block_ends and block_ends[-1] > thinking_tokens):
        raise ValueError(f"block_ends must be strictly increasing and in range: {(qkey, rep)}")
    query_points = row.get("query_points")
    if not isinstance(query_points, list):
        raise ValueError(f"query_points must be a list: {(qkey, rep)}")
    points = []
    for point in query_points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"query_points entries must be two-item lists: {(qkey, rep)}")
        points.append((_exact_int(point[0], field="query_points block", minimum=1),
                       _exact_int(point[1], field="query_points prefix_len", minimum=1)))
    if len(set(points)) != len(points):
        raise ValueError(f"duplicate query_points: {(qkey, rep)}")
    if any(prefix not in block_ends for _block, prefix in points):
        raise ValueError(f"query point prefix is not a recorded block end: {(qkey, rep)}")
    probes = row.get("probes")
    if not isinstance(probes, list):
        raise ValueError(f"probes must be a list: {(qkey, rep)}")
    probe_points = []
    for probe in probes:
        if not isinstance(probe, dict) or set(probe) != PROBE_FIELDS:
            raise ValueError(f"probe schema must contain exactly the public fields: {(qkey, rep)}")
        block = _exact_int(probe.get("block"), field="probe block", minimum=1)
        prefix = _exact_int(probe.get("prefix_len"), field="probe prefix_len", minimum=1)
        forced = probe.get("forced")
        if forced is not None and type(forced) not in (int, str):
            raise ValueError(f"probe forced must be exact int, string, or null: {(qkey, rep)}")
        _exact_int(probe.get("forced_tokens"), field="probe forced_tokens")
        if prefix > thinking_tokens:
            raise ValueError(f"probe prefix_len outside completion: {(qkey, rep)}")
        probe_points.append((block, prefix))
    if points != probe_points:
        raise ValueError(f"query_points/probes mismatch: {(qkey, rep)}")


def validate_source_rows(
    rows: list[dict[str, Any]], manifest_by_qkey: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    seen: set[tuple[str, int]] = set()
    for row in rows:
        qkey = row.get("qkey")
        if type(qkey) is not str or not qkey:
            raise ValueError("chain qkey must be a nonempty exact string")
        manifest_row = manifest_by_qkey.get(qkey)
        if manifest_row is None:
            raise ValueError(f"chain qkey absent from manifest: {qkey}")
        if manifest_row.get("split") != "fit":
            raise ValueError(f"source chain is not exact fit: {qkey}")
        _validate_chain(row, manifest_row)
        identity = (qkey, row["rep"])
        if identity in seen:
            raise ValueError(f"duplicate chain identity: {identity}")
        seen.add(identity)
    rows.sort(key=lambda row: (row["qkey"], row["rep"]))
    return rows


def validate_scored_chain(row: dict[str, Any]) -> None:
    probes, scores = row.get("probes"), row.get("scores")
    identity = (row.get("qkey"), row.get("rep"))
    if not isinstance(probes, list) or not isinstance(scores, list) or len(probes) != len(scores):
        raise ValueError(f"probe/score cardinality mismatch: {identity}")
    for score in scores:
        if not isinstance(score, dict) or set(score) != {"q", "v"}:
            raise ValueError(f"score fields must be exactly q/v: {identity}")
        for field in ("q", "v"):
            value = score[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"q/v probability must be numeric: {identity}")
            if not math.isfinite(value):
                raise ValueError(f"q/v must be finite: {identity}")
            if not 0 <= value <= 1:
                raise ValueError(f"q/v must be in [0,1]: {identity}")


def public_scored_chain(row: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in row.items()
              if key not in PUBLIC_INTERNAL_FIELDS and not key.startswith("_")}
    validate_scored_chain(public)
    return public


def validate_checkpoint_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("freeze_head") is not False:
        raise ValueError("S1 checkpoint must be an explicitly unfrozen payload")
    if not isinstance(payload.get("head"), dict) or not isinstance(payload.get("readout"), dict):
        raise ValueError("invalid S1 checkpoint head/readout state")
    if "weight" not in payload["readout"]:
        raise ValueError("invalid S1 checkpoint readout weight")
    return payload


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write while publishing scorer output")
        view = view[written:]


def _publish_bytes_at(dirfd: int, name: str, payload: bytes) -> str:
    temporary = name + ".tmp"
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o666,
        dir_fd=dirfd,
    )
    try:
        _write_all(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, name, src_dir_fd=dirfd, dst_dir_fd=dirfd)
    os.fsync(dirfd)
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl_at(dirfd: int, name: str, rows: list[dict[str, Any]]) -> str:
    temporary = name + ".tmp"
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o666,
        dir_fd=dirfd,
    )
    digest = hashlib.sha256()
    try:
        for row in rows:
            payload = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
            _write_all(fd, payload)
            digest.update(payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, name, src_dir_fd=dirfd, dst_dir_fd=dirfd)
    os.fsync(dirfd)
    return digest.hexdigest()


def _write_json_at(dirfd: int, name: str, payload: dict[str, Any]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return _publish_bytes_at(dirfd, name, encoded)


def _entry_matches(parentfd: int, name: str, expected) -> bool:
    try:
        observed = os.stat(name, dir_fd=parentfd, follow_symlinks=False)
    except OSError:
        return False
    return (
        stat.S_ISDIR(observed.st_mode)
        and (observed.st_dev, observed.st_ino) == (expected.st_dev, expected.st_ino)
    )


def _rename_noreplace_syscall(parentfd: int, source: str, target: str) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable", target) from error
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(parentfd, os.fsencode(source), parentfd, os.fsencode(target), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise ValueError(f"output directory already exists: {target}")
    raise OSError(error_number, os.strerror(error_number), target)


def _rename_noreplace_fallback(parentfd: int, source: str, target: str) -> None:
    lock_name = f".{target}.publish-lock"
    try:
        lockfd = os.open(
            lock_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parentfd,
        )
    except FileExistsError as error:
        raise ValueError(f"output publish lock already exists: {target}") from error

    lock_identity = None
    reserved = False
    try:
        try:
            lock_identity = os.fstat(lockfd)
            payload = (
                f"pid={os.getpid()} nonce={secrets.token_hex(16)}\n"
            ).encode("ascii")
            _write_all(lockfd, payload)
            os.fsync(lockfd)
        finally:
            os.close(lockfd)

        try:
            os.mkdir(target, 0o700, dir_fd=parentfd)
        except FileExistsError as error:
            raise ValueError(f"output directory already exists: {target}")
        reserved = True

        final_path_identity = os.stat(
            target, dir_fd=parentfd, follow_symlinks=False
        )
        finalfd = os.open(
            target,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parentfd,
        )
        try:
            stagefd = os.open(
                source,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parentfd,
            )
        except BaseException:
            os.close(finalfd)
            raise
        try:
            opened_final = os.fstat(finalfd)
            if (
                not stat.S_ISDIR(final_path_identity.st_mode)
                or (final_path_identity.st_dev, final_path_identity.st_ino)
                != (opened_final.st_dev, opened_final.st_ino)
            ):
                raise ValueError("reserved output directory identity changed")
            os.fsync(parentfd)
            for name in (
                "scored_chains.jsonl",
                "summary.json",
                "integrity.json",
            ):
                os.rename(
                    name, name, src_dir_fd=stagefd, dst_dir_fd=finalfd
                )
                os.fsync(finalfd)
            os.fsync(finalfd)
        finally:
            try:
                os.close(stagefd)
            finally:
                os.close(finalfd)

        os.fsync(parentfd)
        os.rmdir(source, dir_fd=parentfd)
        os.fsync(parentfd)
    except Exception as error:
        lock_still_owned = False
        if not reserved and lock_identity is not None:
            try:
                observed = os.stat(
                    lock_name, dir_fd=parentfd, follow_symlinks=False
                )
                lock_still_owned = (
                    stat.S_ISREG(observed.st_mode)
                    and (observed.st_dev, observed.st_ino)
                    == (lock_identity.st_dev, lock_identity.st_ino)
                )
            except OSError:
                pass
        if lock_still_owned:
            try:
                os.unlink(lock_name, dir_fd=parentfd)
                os.fsync(parentfd)
            except OSError as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(
                        "fail-closed publish-lock cleanup was incomplete: "
                        + str(cleanup_error)
                    )
        raise


def _rename_noreplace(parentfd: int, source: str, target: str) -> None:
    try:
        _rename_noreplace_syscall(parentfd, source, target)
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
            raise
        _rename_noreplace_fallback(parentfd, source, target)


def publish_scored_outputs(
    out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any],
    integrity: dict[str, Any],
) -> dict[str, Any]:
    parent = out_dir.parent
    out_name = out_dir.name
    if not out_name or out_name in {".", ".."}:
        raise ValueError(f"invalid output directory name: {out_dir}")
    parentfd = os.open(
        parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    parent_identity = os.fstat(parentfd)
    if (parent_identity.st_uid != os.geteuid()
            or stat.S_IMODE(parent_identity.st_mode) & 0o022):
        os.close(parentfd)
        raise ValueError(
            "trusted parent must be owned by the effective UID and not "
            f"group/world writable: {parent}"
        )
    try:
        os.stat(out_name, dir_fd=parentfd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        os.close(parentfd)
        raise ValueError(f"output directory already exists: {out_dir}")

    staging_name = f".{out_name}.tmp-{secrets.token_hex(16)}"
    try:
        os.mkdir(staging_name, 0o700, dir_fd=parentfd)
    except BaseException:
        os.close(parentfd)
        raise
    staging_path_identity = None
    try:
        staging_path_identity = os.stat(
            staging_name, dir_fd=parentfd, follow_symlinks=False
        )
        dirfd = os.open(
            staging_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parentfd,
        )
    except BaseException:
        if (staging_path_identity is not None
                and _entry_matches(parentfd, staging_name, staging_path_identity)):
            try:
                os.rmdir(staging_name, dir_fd=parentfd)
            except OSError:
                pass
        os.close(parentfd)
        raise
    directory_identity = os.fstat(dirfd)
    staging_owned = (
        stat.S_ISDIR(staging_path_identity.st_mode)
        and (staging_path_identity.st_dev, staging_path_identity.st_ino)
        == (directory_identity.st_dev, directory_identity.st_ino)
    )
    if not staging_owned:
        os.close(dirfd)
        os.close(parentfd)
        raise ValueError("private staging directory identity changed before open")
    known_names = ("scored_chains.jsonl", "summary.json", "integrity.json")
    published = False
    try:
        scored_hash = _write_jsonl_at(dirfd, "scored_chains.jsonl", rows)
        _write_json_at(dirfd, "summary.json", summary)
        integrity = {**integrity, "scored_chains_sha256": scored_hash}
        _write_json_at(dirfd, "integrity.json", integrity)
        _rename_noreplace(parentfd, staging_name, out_name)
        published = True
        os.fsync(parentfd)
        return integrity
    except BaseException as error:
        cleanup_errors = []
        if staging_owned and not published:
            for name in known_names:
                for owned_name in (name + ".tmp", name):
                    try:
                        os.unlink(owned_name, dir_fd=dirfd)
                    except FileNotFoundError:
                        pass
                    except OSError as cleanup_error:
                        cleanup_errors.append(cleanup_error)
        if (staging_owned and not published
                and _entry_matches(parentfd, staging_name, directory_identity)):
            try:
                os.rmdir(staging_name, dir_fd=parentfd)
            except OSError as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors and hasattr(error, "add_note"):
            error.add_note(
                "fail-closed output cleanup was incomplete: "
                + "; ".join(str(item) for item in cleanup_errors)
            )
        raise
    finally:
        os.close(dirfd)
        os.close(parentfd)


def write_outputs(
    out_dir: Path,
    rows: list[dict[str, Any]],
    *,
    manifest_sha256: str,
    source_paths: list[Path],
    s1_checkpoint: Path,
    source_sha256s: list[str] | None = None,
    s1_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    public = [public_scored_chain(row) for row in rows]
    public.sort(key=lambda row: (row["qkey"], row["rep"]))
    identities = [{"qkey": row["qkey"], "rep": row["rep"]} for row in public]
    hashes = source_sha256s or [sha256_file(path) for path in source_paths]
    source_files = [{"path": str(path), "sha256": digest}
                    for path, digest in zip(source_paths, hashes)]
    summary = {
        "split": "fit", "observed_chains": len(public), "identities": identities,
        "source_fit_files": source_files,
    }
    integrity = {
        **summary,
        "manifest_sha256": manifest_sha256,
        "s1_checkpoint_sha256": s1_checkpoint_sha256 or sha256_file(s1_checkpoint),
    }
    return publish_scored_outputs(out_dir, public, summary, integrity)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-fit-chains", action="append", required=True)
    parser.add_argument("--s1-checkpoint", required=True)
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--hs-chunk-size", type=int, default=8192)
    parser.add_argument("--hf-cache", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise ValueError(f"output directory already exists: {out_dir}")
    manifest_path = Path(args.manifest)
    source_paths = [Path(path) for path in args.source_fit_chains]
    checkpoint_path = Path(args.s1_checkpoint)
    manifest = load_manifest(manifest_path)
    raw_rows: list[dict[str, Any]] = []
    source_sha256s: list[str] = []
    for source_path in source_paths:
        source_rows, source_sha256 = read_jsonl_snapshot(source_path)
        if not source_rows:
            raise ValueError(f"empty source fit chains file: {source_path}")
        raw_rows.extend(source_rows)
        source_sha256s.append(source_sha256)
    raw_rows = validate_source_rows(raw_rows, manifest)

    import torch
    from rl_controller.scripts.single_cot_train import collect_scores, prepare_chains

    payload, checkpoint_sha256 = load_checkpoint_snapshot(checkpoint_path, torch)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    from rl_controller.scripts.rl_train import HeadStateFn, _load_head_runtime
    tokenizer, provider, head, frozen, _hidden = _load_head_runtime(args, device)
    state_fn = HeadStateFn(provider=provider, head=head, frozen=frozen,
                           device=device, raw2048=False)
    state_fn.incremental = False
    state_fn.always_frozen = False
    head.load_state_dict(payload["head"])
    head.eval()
    weight = payload["readout"]["weight"]
    readout = torch.nn.Linear(weight.shape[1], weight.shape[0]).to(device)
    readout.load_state_dict(payload["readout"])
    readout.eval()

    chains = prepare_chains(raw_rows)
    from rl_controller.rollout_contract import build_prompt_ids, rewrite_user_instruction
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    decode = lambda ids: tokenizer.decode(ids)
    prompt_cache: dict[str, list[int]] = {}
    for chain in chains:
        qkey = chain["qkey"]
        if qkey not in prompt_cache:
            prompt_cache[qkey] = rewrite_user_instruction(
                build_prompt_ids(tokenizer, manifest[qkey]["question"]),
                encode=encode, decode=decode,
            )
        chain["_prompt_ids"] = prompt_cache[qkey]
    with torch.no_grad():
        collect_scores(state_fn, readout, chains, device)
    for chain in chains:
        validate_scored_chain(chain)
    verify_source_hashes([manifest_path, *source_paths, checkpoint_path],
                         [MANIFEST_SHA256, *source_sha256s, checkpoint_sha256])
    integrity = write_outputs(
        out_dir, chains, manifest_sha256=MANIFEST_SHA256,
        source_paths=source_paths, s1_checkpoint=checkpoint_path,
        source_sha256s=source_sha256s, s1_checkpoint_sha256=checkpoint_sha256,
    )
    print(json.dumps(integrity, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
