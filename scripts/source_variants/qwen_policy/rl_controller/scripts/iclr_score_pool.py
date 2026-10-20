"""Score a chain pool with a trained single-CoT head."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from rl_controller.baselines.ospruner.layout import HEAD_PREFIX
from rl_controller.baselines.ospruner.reader import HEAD_MODES as TOP2_HEAD_MODES
from rl_controller.iclr import osp_policy as osp
from rl_controller.iclr import score_resume as sr
from rl_controller.iclr import targets as tg
from rl_controller.scripts import single_cot_train as sct
from rl_controller.scripts.single_cot_train import build_prompt_table, dump_scored


BELIEF_MODES = ("targets", "targets_probes")

BELIEF_GROUP = 16

BELIEF_KIND, TOP2_KIND = "belief", "top2"
CHECKPOINT_FILES = {BELIEF_KIND: "s1_best.pt", TOP2_KIND: "osp_best.pt"}


def payload_kind(payload) -> str:
    if not isinstance(payload, dict):
        raise SystemExit(f"checkpoint is a {type(payload).__name__}, not a saved state dict")
    has_readout, has_blocks = "readout" in payload, "blocks" in payload
    mode = payload.get("head_mode", "targets")
    if has_readout and not has_blocks:
        if mode not in sct.HEAD_MODES:
            raise SystemExit(f"checkpoint records a belief `readout` but head_mode {mode!r} is "
                             f"not one of single_cot_train's {list(sct.HEAD_MODES)}")
        return BELIEF_KIND
    if has_blocks and not has_readout:
        if mode not in TOP2_HEAD_MODES:
            raise SystemExit(f"checkpoint records top-2 reader `blocks` but head_mode {mode!r} "
                             f"is not one of iclr_osp_train's {sorted(TOP2_HEAD_MODES)}")
        return TOP2_KIND
    found = sorted(k for k in ("readout", "blocks") if k in payload)
    raise SystemExit(
        f"cannot tell what this checkpoint is: a belief checkpoint records `readout`, an "
        f"OS-Pruner checkpoint records `blocks`, and this one records {found or 'neither'}")


def locate_checkpoint(src: Path, requested: str) -> tuple:
    if requested != "auto":
        path = src / CHECKPOINT_FILES[requested]
        if not path.exists():
            raise SystemExit(f"{path} missing")
        return path, requested
    present = [(src / name, kind) for kind, name in CHECKPOINT_FILES.items()
               if (src / name).exists()]
    if not present:
        raise SystemExit(f"{src / CHECKPOINT_FILES[BELIEF_KIND]} missing "
                         f"(and no {CHECKPOINT_FILES[TOP2_KIND]} either)")
    if len(present) > 1:
        raise SystemExit(f"{src} holds both {sorted(CHECKPOINT_FILES.values())}: pass "
                         f"--checkpoint-kind to say which run this is")
    return present[0]


def require_kind(payload, claimed: str, path) -> str:
    actual = payload_kind(payload)
    if actual != claimed:
        raise SystemExit(f"{path} holds a {actual!r} checkpoint, not the {claimed!r} one its "
                         "name/--checkpoint-kind claims: refusing to score it as the wrong arm")
    return actual


def checkpoint_targets(payload) -> tuple:
    names = tuple(payload.get("targets") or tg.MANDATORY)
    unknown = [n for n in names if n not in tg.CATALOGUE]
    if unknown:
        raise SystemExit(f"checkpoint targets {list(names)} contain unknown names {unknown}")
    if payload.get("head_mode", "targets") in ("targets", "osp") \
            and (names[:1] != tg.MANDATORY[:1] or names[1:2] not in [(v,) for v in tg.V_SLOT]):
        raise SystemExit(f"checkpoint targets must start with 'q' and one of {tg.V_SLOT}, "
                         f"got {list(names)}")
    return names


def check_mode(payload, names, width: int) -> str:
    mode = payload.get("head_mode", "targets")
    expected = {"gap": 1, "osp": 1, "osp_cond": 1}.get(
        mode, tg.readout_width(names))
    if width != expected:
        raise SystemExit(f"head_mode {mode!r} with targets {list(names)} expects a readout of "
                         f"width {expected}, checkpoint has {width}")
    if mode in sct.CONDITIONED_MODES and "policy" not in payload:
        raise SystemExit(f"head_mode {mode!r} needs the conditioned policy in the checkpoint; "
                         "this checkpoint has none, so its stop probabilities cannot be scored")
    return mode


class ReprDump:

    def __init__(self, directory, arm, width, source):
        import numpy as np
        self._np = np
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.arm, self.width = arm, width
        self.path = self.dir / f"{arm}.f16"
        self.index = self.dir / f"{arm}.index.jsonl"
        self.rows = 0
        self.chains = 0
        self._blob = open(self.path, "ab")
        self._idx = open(self.index, "a", encoding="utf-8")
        (self.dir / f"{arm}.source.json").write_text(
            json.dumps({"schema": "frozen-repr-dump/v1", "arm": arm, "width": width,
                        "dtype": "float16", **source}, indent=2) + "\n", encoding="utf-8")

    def add(self, chain, matrix):
        rows = matrix.detach().to("cpu", copy=True).float().numpy().astype(self._np.float16)
        if rows.ndim != 2 or rows.shape[1] != self.width:
            raise SystemExit(f"{self.arm}: expected [n, {self.width}], got {tuple(rows.shape)}")
        if rows.shape[0] != len(chain["_probe_pos"]):
            raise SystemExit(f"{self.arm}: {rows.shape[0]} rows for "
                             f"{len(chain['_probe_pos'])} probes of {chain['qkey']}")
        offset = self._blob.tell()
        self._blob.write(rows.tobytes(order="C"))
        self._idx.write(json.dumps({"qkey": chain["qkey"], "rep": chain.get("rep"),
                                    "n_probes": int(rows.shape[0]), "offset": offset}) + "\n")
        self.rows += int(rows.shape[0])
        self.chains += 1

    def flush(self):
        self._blob.flush()
        self._idx.flush()

    def close(self):
        self.flush()
        self._blob.close()
        self._idx.close()

    @staticmethod
    def load(directory, arm):
        import numpy as np
        directory = Path(directory)
        meta = json.loads((directory / f"{arm}.source.json").read_text(encoding="utf-8"))
        width = int(meta["width"])
        blob = np.fromfile(directory / f"{arm}.f16", dtype=np.float16)
        live = {}
        with open(directory / f"{arm}.index.jsonl", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    live[(row["qkey"], row["rep"])] = row
        out = {}
        for key, row in live.items():
            start = row["offset"] // 2
            n = int(row["n_probes"])
            piece = blob[start:start + n * width]
            if piece.size != n * width:
                raise SystemExit(f"{arm}: chain {key} wants {n * width} values, "
                                 f"the blob has {piece.size} from offset {row['offset']}")
            out[key] = piece.reshape(n, width).astype(np.float32)
        return out, meta


def probe_scores(c, vals, names):
    keys = tuple(names) + (("v",) if "v" not in names and "v" in vals else ())
    return [{name: vals[name][p] for name in keys} for p in c["_probe_pos"]]


def one_unit_scores(mode, probe_out):
    import torch
    if mode == "gap":
        return [{"q": v, "v": 0.0, "d": v}
                for v in torch.tanh(probe_out[:, 0]).cpu().tolist()]
    return [{"q": pi, "v": sct.OSP_V_CONST, "pi": pi}
            for pi in torch.sigmoid(probe_out[:, 0]).cpu().tolist()]


def cond_scores(policy, feats, infer_feats, n_inference, ref_index):
    import torch
    pis = torch.sigmoid(sct.policy_logits(policy, feats, infer_feats)).cpu().tolist()
    grid, ref = pis[:n_inference], pis[ref_index]
    return [{"q": ref[i], "v": sct.OSP_V_CONST, "pi": ref[i],
             "pi_by_lambda": [col[i] for col in grid]} for i in range(len(ref))]


def cond_runtime(payload, probe_dim, device):
    import torch
    lambdas = list(payload.get("lambda_grid") or sct.DEFAULT_LAMBDA_GRID)
    if sct.OSP_REFERENCE_LAMBDA not in lambdas:
        raise SystemExit(f"checkpoint lambda_grid {lambdas} lacks the reference price "
                         f"{sct.OSP_REFERENCE_LAMBDA:g}, which is the scored dump's `q` column")
    n_inference = len(osp.INFERENCE_LAMBDAS)
    in_dim = 3 if payload["head_mode"] == "osp_bottleneck" else probe_dim + 1
    policy = sct.LambdaPolicy(in_dim).to(device)
    policy.load_state_dict(payload["policy"])
    policy.eval()
    feats = torch.tensor(osp.lambda_grid_features(list(osp.INFERENCE_LAMBDAS) + lambdas),
                         dtype=torch.float32, device=device)
    return {"policy": policy, "infer_feats": feats, "n_inference": n_inference,
            "ref_index": n_inference + lambdas.index(sct.OSP_REFERENCE_LAMBDA),
            "bottleneck": payload["head_mode"] == "osp_bottleneck",
            "aux": payload["head_mode"] in sct.AUX_MODES}


def mode_scores(mode, rt, reprs, out, probe_pos):
    import torch
    idx = torch.tensor(probe_pos, dtype=torch.long, device=reprs.device)
    if mode in ("gap", "osp"):
        return one_unit_scores(mode, out[idx])
    beliefs = torch.sigmoid(out[idx][:, :2]) if rt["aux"] else None
    feats = beliefs if rt["bottleneck"] else reprs[idx]
    rows = cond_scores(rt["policy"], feats, rt["infer_feats"],
                       rt["n_inference"], rt["ref_index"])
    if rt["aux"]:
        for i, row in enumerate(rows):
            row["q_aux"], row["v_aux"] = beliefs[i, 0].item(), beliefs[i, 1].item()
    return rows


def check_top2_mode(payload, names, cfg) -> str:
    mode = payload["head_mode"]
    if mode == "qv" and tuple(names[:2]) != ("q", "v"):
        raise SystemExit(f"head_mode 'qv' requires targets starting with q,v (the ladder's "
                         f"mandatory beliefs), got {list(names)}")
    head = payload.get("head")
    if not isinstance(head, dict) or "weight" not in head:
        raise SystemExit("checkpoint has no `head.weight`: an OS-Pruner checkpoint records the "
                         "reader head's state dict, and without it there is nothing to score with")
    expected = 1 if mode == "osp" else len(names)
    width = int(head["weight"].shape[0])
    if width != expected:
        raise SystemExit(f"head_mode {mode!r} with targets {list(names)} expects a head of "
                         f"width {expected}, checkpoint has {width}")
    outputs = cfg.get("outputs")
    if outputs is not None and list(outputs) != list(TOP2_HEAD_MODES[mode] if mode == "osp"
                                                     else names):
        raise SystemExit(f"the run's reader manifest declares outputs {list(outputs)}, which is "
                         f"not what head_mode {mode!r} with targets {list(names)} produces")
    return mode


def top2_reader_config(summary, payload, names, args) -> dict:
    config = summary.get("config") or {}
    training = config.get("training_config") or {}
    manifest = config.get("reader") or {}
    mode = payload["head_mode"]
    for label, recorded in (("summary config", config.get("head_mode")),
                            ("training_config", training.get("head_mode")),
                            ("reader manifest", manifest.get("head_mode"))):
        if recorded is not None and recorded != mode:
            raise SystemExit(f"checkpoint head_mode {mode!r} != the {label}'s {recorded!r}: "
                             "the checkpoint and the run dir describe different runs")
    n_unfrozen = manifest.get("n_unfrozen", training.get("n_unfrozen"))
    if n_unfrozen is None:
        raise SystemExit(f"{args.source_run}/summary.json records neither config.reader."
                         "n_unfrozen nor config.training_config.n_unfrozen, so the reader's "
                         "architecture cannot be rebuilt as it was trained")
    revision = training.get("revision") or (summary.get("provenance") or {}).get("model_revision")
    if revision and args.revision and revision != args.revision:
        raise SystemExit(f"--revision {args.revision} != the backbone revision this lane trained "
                         f"against ({revision})")
    return {"head_mode": mode, "targets": list(names),
            "n_unfrozen": int(n_unfrozen),
            "fp32_trainable": bool(training.get("fp32_trainable", True)),
            "dtype": training.get("dtype") or args.dtype,
            "model": training.get("model") or args.model,
            "model_path": args.model_path or training.get("model_path"),
            "revision": revision or args.revision,
            "max_seq_len": training.get("max_seq_len"),
            "outputs": manifest.get("outputs")}


def load_top2_weights(reader, payload, torch) -> None:
    by_name = dict(reader.module.named_parameters())
    blocks = payload["blocks"]
    trainable = {n for n, p in by_name.items() if p.requires_grad}
    missing, extra = sorted(trainable - set(blocks)), sorted(set(blocks) - trainable)
    if missing or extra:
        raise SystemExit(f"checkpoint trainable tensors do not match the rebuilt reader: "
                         f"{len(missing)} missing (e.g. {missing[:3]}), {len(extra)} unknown "
                         f"(e.g. {extra[:3]})")
    for name, value in blocks.items():
        if tuple(value.shape) != tuple(by_name[name].shape):
            raise SystemExit(f"checkpoint tensor {name} has shape {tuple(value.shape)}, the "
                             f"rebuilt reader has {tuple(by_name[name].shape)}")
    for key, value in (payload.get("head") or {}).items():
        stored = blocks.get(f"{HEAD_PREFIX}.{key}")
        if stored is not None and not torch.equal(stored.cpu(), value.cpu()):
            raise SystemExit(f"checkpoint `head`.{key} disagrees with `blocks`.{HEAD_PREFIX}."
                             f"{key}: the two records of one tensor were not written together")
    with torch.no_grad():
        for name, value in blocks.items():
            by_name[name].copy_(value.to(by_name[name].device))


def build_top2_reader(cfg, payload, args, device, torch):
    from rl_controller.baselines.ospruner.reader import TopTwoReader, load_backbone

    backbone = load_backbone(model_path=cfg["model_path"], model_id=cfg["model"],
                             revision=cfg["revision"], dtype=cfg["dtype"],
                             cache_dir=args.hf_cache)
    reader = TopTwoReader(backbone, head_mode=cfg["head_mode"], n_unfrozen=cfg["n_unfrozen"],
                          fp32_trainable=cfg["fp32_trainable"],
                          output_names=None if cfg["head_mode"] == "osp" else cfg["targets"])
    reader.module.to(device)
    load_top2_weights(reader, payload, torch)
    reader.eval()
    return reader


def top2_tokenizer(cfg, args):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg["model_path"] or cfg["model"],
                                         revision=cfg["revision"], cache_dir=args.hf_cache,
                                         trust_remote_code=True)


def top2_forward(reader, prompts, prompt_lens, *, probes_only, pad_id, device, torch):
    from rl_controller.baselines.ospruner import layout as L
    from rl_controller.baselines.ospruner.reader import pad_batch_tensors

    def forward(group, *, grad: bool = False):
        plan = L.plan_chain_batch(group, prompt_lens=prompt_lens, probes_only=probes_only)
        seqs = [list(prompts[c["qkey"]]) + list(c["completion_ids"][:c["_ends"][-1]])
                for c in group]
        ids, mask = pad_batch_tensors(seqs, plan, device=device, pad_id=pad_id)
        with torch.set_grad_enabled(grad):
            logits = reader.boundary_logits(ids, mask, plan)
        return reader.split_by_chain(logits, plan), plan

    return forward


def score_top2(args, payload, names, chains, qrows, summary, device, torch) -> tuple:
    import types

    from rl_controller.baselines.ospruner import training as T
    from rl_controller.scripts import iclr_osp_train as ot

    cfg = top2_reader_config(summary, payload, names, args)
    mode = check_top2_mode(payload, names, cfg)
    reader = build_top2_reader(cfg, payload, args, device, torch)
    tokenizer = top2_tokenizer(cfg, args)
    prompts = build_prompt_table(qrows, tokenizer)
    for c in chains:
        if c["qkey"] not in prompts:
            raise SystemExit(f"chain question {c['qkey']} not in --questions")
    todo = [c for c in chains if c["_trainable"]]
    if cfg["max_seq_len"]:
        _fits, too_long = ot.split_by_context(todo, prompts, cfg["max_seq_len"])
        if too_long:
            longest = max(len(prompts[c["qkey"]]) + c["_ends"][-1] for c in too_long)
            raise SystemExit(
                f"{len(too_long)} chains exceed the lane's --max-seq-len {cfg['max_seq_len']} "
                f"(longest needs {longest} tokens). They cannot be read by this reader, and "
                "dropping them would move this pool's natural anchor away from every other "
                "method's.")
    forward = top2_forward(reader, prompts, {k: len(v) for k, v in prompts.items()},
                           probes_only=T.gathers_probes_only(mode),
                           pad_id=getattr(tokenizer, "pad_token_id", 0) or 0,
                           device=device, torch=torch)
    step = max(1, int(args.chains_per_step))
    ns = types.SimpleNamespace(head_mode=mode, chains_per_step=step)
    sr.run(checkpoint_of(args), todo, group_size=step,
           score_slab=lambda slab: ot._score(reader, slab, ns, list(names), forward, torch,
                                             split="select"),
           extra={"head_mode": mode})
    return mode, {"osp_checkpoint_epoch": int(payload.get("epoch", -1)),
                  "reader": reader.manifest(), "reader_config": cfg}


def check_source_run(source_run, names) -> dict:
    src = Path(source_run)
    summary = src / "summary.json"
    if not summary.exists() or not (src / "val_scored.jsonl").exists():
        raise SystemExit(f"--source-run {src} needs summary.json and val_scored.jsonl")
    config = json.load(open(summary))["config"]
    if "seed" not in config:
        raise SystemExit(f"{summary}: config.seed missing")
    src_names = tuple(config.get("targets") or tg.MANDATORY)
    if isinstance(config.get("targets"), str):
        src_names = tuple(config["targets"].split(","))
    if src_names != tuple(names):
        raise SystemExit(f"source run targets {list(src_names)} != checkpoint targets {list(names)}")
    return config


DEVICE_SCHEMA = "iclr-score-device/v1"

DECLARED_DEVICE_FIELDS = ("gpu", "gpu_mem_gb", "gpu_compute_capability")


def declared_device(config) -> dict:
    ranks = config.get("gpu_ranks") or []
    cards = sorted({(r.get("gpu"), r.get("gpu_compute_capability")) for r in ranks},
                   key=lambda card: tuple("" if v is None else str(v) for v in card))
    if len(cards) > 1:
        raise SystemExit(f"the source run's own record says its ranks were not all the same "
                         f"card ({cards}); its `config.gpu` names one architecture out of two, "
                         "so there is nothing here a re-score could be checked against")
    return {k: config.get(k) for k in DECLARED_DEVICE_FIELDS if config.get(k) is not None}


def scoring_device(torch, device) -> dict:
    record = {"device": str(device), "gpu": None, "gpu_mem_gb": None,
              "gpu_compute_capability": None}
    if getattr(device, "type", None) != "cuda":
        return record
    props = torch.cuda.get_device_properties(device)
    return {**record, "gpu": props.name,
            "gpu_mem_gb": round(props.total_memory / 2 ** 30, 1),
            "gpu_compute_capability": f"{props.major}.{props.minor}"}


def compare_devices(declared: dict, actual: dict) -> tuple:
    if not (declared.get("gpu") or declared.get("gpu_compute_capability")):
        return None, None, "the source run declares no training device"
    if not (actual.get("gpu") or actual.get("gpu_compute_capability")):
        return None, None, (f"scoring on {actual.get('device')}, which has no GPU architecture "
                            "to compare")
    if declared.get("gpu_compute_capability") and actual.get("gpu_compute_capability"):
        return ("gpu_compute_capability",
                declared["gpu_compute_capability"] == actual["gpu_compute_capability"], None)
    return "gpu", declared.get("gpu") == actual.get("gpu"), None


def device_identity(declared: dict, actual: dict, *, allow_mismatch: bool = False) -> dict:
    comparand, match, why = compare_devices(declared, actual)
    record = {"schema": DEVICE_SCHEMA, "scoring": actual, "training": declared or None,
              "comparand": comparand, "checked": comparand is not None, "match": match,
              "override": bool(allow_mismatch), "why": why}
    if comparand is None:
        print(json.dumps({"warning": "device_architecture_unverified", "why": why,
                          "scoring": actual, "training": declared or None}),
              file=sys.stderr, flush=True)
        return record
    if match:
        return record
    trained_on = f"{declared.get('gpu') or 'an unnamed card'} " \
                 f"(compute capability {declared.get('gpu_compute_capability') or 'unrecorded'})"
    scoring_on = f"{actual.get('gpu') or actual.get('device')} " \
                 f"(compute capability {actual.get('gpu_compute_capability') or 'unrecorded'})"
    if not allow_mismatch:
        raise SystemExit(
            f"this lane trained on {trained_on} and this process is on {scoring_on}; they differ "
            f"in {comparand}. bf16 kernel selection is per GPU architecture: the same "
            "checkpoint, the same chains and the same commit reproduce a lane's own scores bit "
            "for bit on the architecture it trained on (48/48, max |dpi| exactly 0) and miss a "
            "1e-4 parity gate on another by up to 9.2e-1 (median 8.4e-3). Whatever such a dump "
            "differed by would be the card, not anything the reader did. Rescore on the "
            "architecture the lane trained on, or pass --allow-device-mismatch to declare a "
            "deliberate cross-architecture comparison, which is recorded in the dump's "
            "provenance.")
    print(json.dumps({"warning": "device_architecture_mismatch_allowed",
                      "comparand": comparand, "scoring": actual, "training": declared,
                      "note": "scores are not comparable bit for bit with this lane's own dump"}),
          file=sys.stderr, flush=True)
    return record


READER_SCHEMA = "iclr-score-reader/v1"

READER_WEIGHTS, READER_CONFIG = "weights.pth", "config.yaml"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def loaded_reader(head_checkpoint) -> dict:
    path = Path(head_checkpoint).expanduser()
    if not path.is_dir():
        raise SystemExit(f"--head-checkpoint {head_checkpoint} is not a local directory: the "
                         "frozen reader a belief head sits on has to be identifiable by its "
                         "bytes, and a hub id resolves to whatever the cache happens to hold")
    path = path.resolve()
    weights, config = path / READER_WEIGHTS, path / READER_CONFIG
    for needed in (weights, config):
        if not needed.is_file():
            raise SystemExit(f"{path} is not a UHead checkpoint: {needed.name} is missing, and "
                             "it is one of the two files the loader reads")
    return {"dir": str(path), "weights_path": str(weights),
            "weights_sha256": _sha256_file(weights),
            "weights_bytes": weights.stat().st_size,
            "config_path": str(config), "config_sha256": _sha256_file(config)}


def declared_reader(provenance) -> dict:
    if not isinstance(provenance, dict):
        return {}
    record = provenance.get("head_checkpoint")
    if not isinstance(record, dict) or not record:
        return {}
    declared = {}
    weights = record.get("weights")
    if isinstance(weights, dict) and weights.get("sha256"):
        declared["weights_sha256"] = weights["sha256"]
        if weights.get("path"):
            declared["weights_path"] = weights["path"]
    config = record.get("config")
    if isinstance(config, dict) and config.get("sha256"):
        declared["config_sha256"] = config["sha256"]
    source = record.get("dir") or record.get("path")
    if source:
        declared["source"] = str(source)
    return declared


def _hash_training_source(declared: dict, training_source) -> dict:
    recorded = declared.get("source")
    source = Path(training_source).expanduser() if training_source else (
        Path(recorded).expanduser() if recorded else None)
    if source is None:
        raise SystemExit(
            "this lane records neither a reader digest nor the source it trained from, so the "
            "frozen feature extractor its head sits on cannot be identified at all")
    weights = source / READER_WEIGHTS if source.is_dir() else source
    if not weights.is_file():
        raise SystemExit(
            f"this lane records no reader digest, only the training source {recorded!r}, and "
            f"{weights} cannot be reached from this machine, so its reader has nothing to be "
            "checked against. Score it where that source lives, or pass "
            "--reader-training-source naming the same source relocated here -- its bytes are "
            "hashed and compared exactly as a recorded digest would be.")
    return {**declared, "weights_sha256": _sha256_file(weights),
            "recorded_source": recorded,
            "source": str((weights.parent if weights.name == READER_WEIGHTS else weights)
                          .resolve())}


def reader_identity(declared: dict, actual: dict, *, training_source=None,
                    allow_mismatch: bool = False) -> dict:
    if not declared:
        raise SystemExit(
            "this lane's summary records no reader at all, so the frozen feature extractor its "
            "belief head sits on cannot be identified. `single_cot_train` has always written it "
            "to provenance.head_checkpoint; unlike the training device, this field is not one "
            "that post-dates the lane, so there is no compatibility case to fall back on.")
    if declared.get("weights_sha256"):
        comparand = "weights_sha256"
    else:
        declared = _hash_training_source(declared, training_source)
        comparand = "training_source_sha256"
    fields = ["weights_sha256"] + (["config_sha256"] if declared.get("config_sha256") else [])
    if len(fields) > 1:
        comparand += "+config_sha256"
    differing = [f for f in fields if declared[f] != actual[f]]
    why = None if not differing else (
        f"the lane's {' and '.join(differing)} is not the reader loaded here")
    record = {"schema": READER_SCHEMA, "loaded": actual, "declared": declared,
              "comparand": comparand, "checked": True, "match": not differing,
              "override": bool(allow_mismatch), "why": why}
    if not differing:
        return record
    detail = "; ".join(f"{f}: lane {declared[f]} vs loaded {actual[f]}" for f in fields)
    if not allow_mismatch:
        raise SystemExit(
            f"the reader at {actual['dir']} is not the reader this lane trained against "
            f"({declared.get('source') or 'source unrecorded'}). {detail}. A belief head is "
            "trained ON this extractor's representation, so scoring it against another one is "
            "a different measurement, not a rounding difference -- and the two readers this "
            "campaign holds are the same byte count with the same 39 tensor keys in the same "
            "order, so agreeing shapes, agreeing paths and a clean load prove nothing. Point "
            "--head-checkpoint at the reader this lane recorded, or pass "
            "--allow-reader-mismatch to declare a deliberate cross-reader comparison, which is "
            "recorded in the dump's provenance.")
    print(json.dumps({"warning": "reader_identity_mismatch_allowed", "comparand": comparand,
                      "loaded": actual, "declared": declared,
                      "note": "this head was scored on a representation it was not trained on"}),
          file=sys.stderr, flush=True)
    return record


def checkpoint_of(args):
    return getattr(args, "resume_checkpoint", None) or sr.Checkpoint.off()


def resume_identity(args, *, chains_prov, questions_prov, checkpoint_prov, reader_prov,
                    device_prov, names) -> dict:
    return {"run_dir": str(Path(args.out_dir).resolve()),
            "chains_sha256": chains_prov["sha256"],
            "questions_sha256": questions_prov["sha256"],
            "checkpoint_sha256": checkpoint_prov["sha256"],
            "reader_sha256": (reader_prov or {}).get("loaded"),
            "targets": list(names),
            "head_mode": None,
            "group_size": None,
            "chains_per_step": int(args.chains_per_step),
            "device": (device_prov or {}).get("scoring"),
            "commit": chains_prov.get("commit"),
            "dirty": chains_prov.get("dirty")}


def write_run_dir(out_dir, chains, *, source_run, names, seed, head_mode, provenance,
                  extra_config=None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(source_run) / "val_scored.jsonl", out / "val_scored.jsonl")
    dump_scored(chains, out / "select_scored.jsonl")
    config = {"seed": int(seed), "targets": list(names), "readout": "score_pool",
              "head_mode": head_mode,
              "source_run": str(source_run), "n_select_chains": len(chains),
              "n_select_questions": len({c["qkey"] for c in chains}),
              **(extra_config or {})}
    with open(out / "summary.json", "w") as f:
        json.dump({"config": config, "provenance": provenance}, f, indent=2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chains", required=True, help="chains.jsonl to score (e.g. the OOD report split)")
    p.add_argument("--questions", required=True, help="jsonl with a 'question' field per row (the pool manifest)")
    p.add_argument("--source-run", required=True,
                   help="training run dir: summary.json, val_scored.jsonl and either "
                        "s1_best.pt (belief) or osp_best.pt (OS-Pruner top-2 reader)")
    p.add_argument("--checkpoint-kind", choices=("auto", BELIEF_KIND, TOP2_KIND), default="auto",
                   help="which checkpoint to read from --source-run; `auto` (default) takes the "
                        "one that is there. Whatever is chosen, the file's own recorded identity "
                        "must agree with it or the run is refused")
    p.add_argument("--head-checkpoint", default=None,
                   help="belief runs only: the frozen UHead dir this lane's head was trained "
                        "ON. Its bytes are verified against the digest the lane recorded "
                        "before anything is loaded. The OS-Pruner path rebuilds its own "
                        "backbone instead and refuses this flag")
    p.add_argument("--reader-training-source", default=None,
                   help="belief runs only, and only for a lane that recorded no reader digest: "
                        "the directory (or weights file) that lane trained from, relocated to "
                        "this machine. Its bytes are hashed and compared like a recorded "
                        "digest; this relocates the check, it does not skip it")
    p.add_argument("--allow-reader-mismatch", action="store_true",
                   help="score a belief lane against a frozen reader it did not train on. The "
                        "head is then evaluated on a representation it never saw; the choice "
                        "is RECORDED in the run dir's provenance "
                        "(head_checkpoint.override) rather than only logged")
    p.add_argument("--chains-per-step", type=int, default=4,
                   help="OS-Pruner path: chains per forward, the trainer's own batch knob")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dump-repr-dir", default=None, metavar="DIR",
                   help="steer 20.3: ALSO write the frozen per-probe representations this scoring "
                        "pass already computes. Off by default and a pure side effect -- no score, "
                        "no receipt and no checkpoint changes when it is set. It exists so that "
                        "LB_H / LB_L16 are fitted on EXACTLY the probe set lane B replays: the rows "
                        "are `reprs[_probe_pos]`, the same selection and the same order that builds "
                        "`chain['scores']` two lines away, so alignment is shared code rather than "
                        "two implementations that agree today.")
    p.add_argument("--dump-repr-raw", action="store_true",
                   help="with --dump-repr-dir, also dump the 2048-d layer-16 input (LB_L16). Turns "
                        "on the state fn's raw2048 capture, which costs memory, so it is separate.")
    p.add_argument("--chains-sha256", default=None, help="refuse to score a chains file whose sha256 differs")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--model-path", default=None)
    p.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--hs-chunk-size", type=int, default=8192)
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--gpu", type=int, default=0,
                   help="CUDA device index; -1 runs on the CPU (for tests, not for a pool)")
    p.add_argument("--allow-device-mismatch", action="store_true",
                   help="score a lane on a GPU architecture it did not train on. bf16 kernel "
                        "selection is per architecture, so the dump cannot be compared bit for "
                        "bit with the lane's own; the choice is RECORDED in the run dir's "
                        "provenance (device.override) rather than only logged")
    p.add_argument("--resume-every", type=int, default=sr.DEFAULT_EVERY, metavar="N",
                   help="flush per-chain progress every N chains, rounded UP to a whole number "
                        "of forward packs. A crash then costs at most N chains instead of the "
                        "whole pass. Not part of the resume identity: the resume point is a "
                        "pack boundary either way")
    p.add_argument("--no-resume", action="store_true",
                   help="start clean: a partial already in --out-dir is discarded UNREAD, and "
                        "the dump's provenance records that it was. This run still writes its "
                        "own progress, so it remains resumable")
    p.add_argument("--store-token-budget", type=int, default=8_000_000,
                   help="clear the GPU feature store after any 16-chain group past this many stored tokens")
    return p.parse_args(argv)


def score_belief(args, payload, names, chains, qrows, device, torch) -> dict:
    from rl_controller.scripts.rl_train import HeadStateFn, _load_head_runtime
    from rl_controller.scripts.single_cot_train import _forward_with_recovery, readout_values

    tokenizer, provider, head, frozen, _hidden = _load_head_runtime(args, device)
    want_raw = bool(args.dump_repr_dir) and bool(args.dump_repr_raw)
    state_fn = HeadStateFn(provider=provider, head=head, frozen=frozen, device=device,
                           raw2048=want_raw)
    state_fn.incremental = False
    state_fn.always_frozen = False
    head.load_state_dict(payload["head"])
    head.eval()
    w = payload["readout"]["weight"]
    readout = torch.nn.Linear(w.shape[1], w.shape[0]).to(device)
    readout.load_state_dict(payload["readout"])
    mode = check_mode(payload, names, w.shape[0])
    rt = cond_runtime(payload, w.shape[1], device) if mode in sct.CONDITIONED_MODES else None

    prompts = build_prompt_table(qrows, tokenizer)
    for c in chains:
        if c["qkey"] not in prompts:
            raise SystemExit(f"chain question {c['qkey']} not in --questions")
        c["_prompt_ids"] = prompts[c["qkey"]]

    todo = [c for c in chains if c["_trainable"]]
    for c in chains:
        c["scores"] = []

    dumps = {}
    if args.dump_repr_dir:
        source = {"chains": str(args.chains), "chains_sha256": args.chains_sha256,
                  "source_run": str(args.source_run), "head_mode": mode,
                  "readout_out_features": int(w.shape[0])}
        dumps["LB_H"] = ReprDump(args.dump_repr_dir, "LB_H", int(w.shape[1]), source)
        if want_raw:
            dumps["LB_L16"] = ReprDump(args.dump_repr_dir, "LB_L16", 2048, source)

    def score_slab(slab):
        for grp in sr.groups(slab, BELIEF_GROUP):
            outs = _forward_with_recovery(state_fn, grp, grad=False)
            for c, o in zip(grp, outs):
                reprs = o["reprs"].float()
                out = readout(reprs)
                if mode in BELIEF_MODES:
                    c["scores"] = probe_scores(c, readout_values(out, names), names)
                else:
                    c["scores"] = mode_scores(mode, rt, reprs, out, c["_probe_pos"])
                if dumps:
                    idx = torch.tensor(c["_probe_pos"], dtype=torch.long, device=reprs.device)
                    dumps["LB_H"].add(c, reprs[idx])
                    if "LB_L16" in dumps:
                        if o.get("raw") is None:
                            raise SystemExit("--dump-repr-raw asked for the 2048-d input but the "
                                             "state fn captured no raw rows")
                        dumps["LB_L16"].add(c, o["raw"].float()[idx])
            if state_fn.store_stats["tokens_stored"] > args.store_token_budget:
                state_fn.clear_store()
        for dump in dumps.values():
            dump.flush()

    with torch.no_grad():
        sr.run(checkpoint_of(args), todo, group_size=BELIEF_GROUP, score_slab=score_slab,
               extra={"head_mode": mode},
               after_slab=lambda done, total: print(json.dumps({"scored": done, "of": total}),
                                                    flush=True))
    result = {"head_mode": mode, "n_trainable": len(todo)}
    if dumps:
        for arm, dump in dumps.items():
            if dump.chains != len(todo):
                raise SystemExit(
                    f"{arm}: dumped {dump.chains} chains but {len(todo)} were scored. A dump must "
                    "cover every scored chain; this run resumed over chains an earlier attempt had "
                    "already scored, so their vectors were never computed. Re-run the dump with "
                    "--no-resume (the scores are unaffected; only the dump is incomplete).")
        result["repr_dump"] = {arm: {"rows": d.rows, "chains": d.chains, "width": d.width,
                                     "path": str(d.path)} for arm, d in dumps.items()}
        for dump in dumps.values():
            dump.close()
    return result


def main(argv=None) -> int:
    import torch
    from rl_controller.iclr.provenance import source_record
    from rl_controller.scripts.single_cot_train import load_chains
    args = parse_args(argv)
    src = Path(args.source_run)
    ckpt, claimed = locate_checkpoint(src, args.checkpoint_kind)
    chains_prov = source_record(args.chains)
    questions_prov = source_record(args.questions)
    if args.chains_sha256 and chains_prov["sha256"] != args.chains_sha256:
        raise SystemExit(f"--chains sha256 {chains_prov['sha256']} != expected {args.chains_sha256}")
    device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 else torch.device("cpu")
    payload = torch.load(ckpt, map_location=device)
    kind = require_kind(payload, claimed, ckpt)
    names = checkpoint_targets(payload)
    config = check_source_run(src, names)
    device_prov = device_identity(declared_device(config), scoring_device(torch, device),
                                  allow_mismatch=args.allow_device_mismatch)
    summary = json.load(open(src / "summary.json"))
    if kind == BELIEF_KIND:
        if not args.head_checkpoint:
            raise SystemExit("--head-checkpoint is required to score a belief checkpoint: it is "
                             "the UHead architecture s1_best.pt's weights are loaded into")
        reader_prov = reader_identity(declared_reader(summary.get("provenance")),
                                      loaded_reader(args.head_checkpoint),
                                      training_source=args.reader_training_source,
                                      allow_mismatch=args.allow_reader_mismatch)
    else:
        if args.head_checkpoint:
            raise SystemExit("--head-checkpoint is a belief-only flag: an OS-Pruner checkpoint "
                             "carries its own reader and never touches a UHead")
        for flag, value in (("--reader-training-source", args.reader_training_source),
                            ("--allow-reader-mismatch", args.allow_reader_mismatch)):
            if value:
                raise SystemExit(f"{flag} is a belief-only flag: an OS-Pruner checkpoint is "
                                 "rebuilt from the lane's own recorded config and names its "
                                 "checkpoint with a digest, so there is no external reader")
        reader_prov = None

    chains = load_chains([args.chains])
    if not chains:
        raise SystemExit("no chains")
    bad = [(c["qkey"], c["rep"]) for c in chains if not c["_trainable"] and c["probes"]]
    if bad:
        raise SystemExit(f"{len(bad)} chains have probes but are not trainable (first {bad[:3]}); "
                         "dump_scored would refuse them after the GPU pass — fix the pool first")
    qrows = [json.loads(line) for line in open(args.questions) if line.strip()]

    checkpoint_prov = source_record(str(ckpt))
    args.resume_checkpoint = sr.Checkpoint(
        args.out_dir,
        resume_identity(args, chains_prov=chains_prov, questions_prov=questions_prov,
                        checkpoint_prov=checkpoint_prov, reader_prov=reader_prov,
                        device_prov=device_prov, names=names),
        every=args.resume_every, resume=not args.no_resume)

    if kind == BELIEF_KIND:
        scored = score_belief(args, payload, names, chains, qrows, device, torch)
        prov_extra = {"s1_checkpoint": checkpoint_prov, "source_run": str(src),
                      "s1_epoch": int(payload.get("epoch", -1)),
                      "head_checkpoint": reader_prov}
        extra_config = None
    else:
        mode, top2_prov = score_top2(args, payload, names, chains, qrows, summary, device, torch)
        scored = {"head_mode": mode, "n_trainable": len([c for c in chains if c["_trainable"]])}
        prov_extra = {"osp_checkpoint": checkpoint_prov, "source_run": str(src),
                      **top2_prov}
        extra_config = {"checkpoint_kind": TOP2_KIND}
    provenance = {"chains": chains_prov, "questions": questions_prov,
                  **prov_extra, "device": device_prov,
                  "resume": args.resume_checkpoint.report}
    write_run_dir(args.out_dir, chains, source_run=src, names=names, seed=config["seed"],
                  head_mode=scored["head_mode"], provenance=provenance,
                  extra_config=extra_config)
    removed = args.resume_checkpoint.finish()
    if removed.get("disabled"):
        print(json.dumps({"warning": "score_partial", "why": removed["disabled"]}),
              file=sys.stderr, flush=True)
    print(json.dumps({"done": len(chains), "trainable": scored["n_trainable"],
                      "targets": list(names), "head_mode": scored["head_mode"],
                      "out": str(args.out_dir)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
