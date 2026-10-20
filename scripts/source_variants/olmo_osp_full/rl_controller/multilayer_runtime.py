from __future__ import annotations

import copy
import hashlib
import json

import json
import os

import torch


ATTENTION_MODES = ("default", "no-cudnn", "no-flash", "math-only", "eager")

DISABLED_BACKENDS = {"default": (), "eager": (),
                     "no-cudnn": ("cudnn",),
                     "no-flash": ("cudnn", "flash"),
                     "math-only": ("cudnn", "flash", "mem_efficient")}


def attention_mode(override=None):
    mode = (override or os.environ.get("OLMO3_ATTENTION") or "default").strip().lower()
    if mode not in ATTENTION_MODES:
        raise ValueError(f"OLMO3_ATTENTION={mode!r}; expected one of {ATTENTION_MODES}")
    return mode


def configure_attention(mode):
    for backend in DISABLED_BACKENDS[mode]:
        switch = getattr(torch.backends.cuda, {"cudnn": "enable_cudnn_sdp",
                                               "flash": "enable_flash_sdp",
                                               "mem_efficient": "enable_mem_efficient_sdp"}[backend])
        switch(False)
    enabled = {}
    for name in ("cudnn", "flash", "mem_efficient", "math"):
        probe = getattr(torch.backends.cuda, f"{name}_sdp_enabled", None)
        if callable(probe):
            enabled[name] = bool(probe())
    print(json.dumps({"attention_mode": mode, "sdp_backends_enabled": enabled}), flush=True)
    return {"attn_implementation": "eager"} if mode == "eager" else {}


def load_runtime(args, device, *, saved_config=None, saved_backbone=None):
    from transformers import AutoModel, AutoTokenizer
    from rl_controller.multilayer_features import MultilayerHiddenStateProvider
    from rl_controller.multilayer_uhead import MultilayerUHead, MultilayerUHeadConfig

    backbone = saved_backbone or {"model": args.model, "revision": args.revision, "dtype": args.dtype}
    model_ref = args.model_path or backbone["model"]
    revision = backbone["revision"]
    if args.model_path and not str(args.model_path).rstrip("/").endswith(revision):
        raise ValueError("--model-path must end in the pinned backbone revision")
    dtype = getattr(torch, backbone["dtype"], None)
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("backbone dtype must be float32, float16 or bfloat16")
    attention = configure_attention(attention_mode(getattr(args, "attention", None)))
    model = AutoModel.from_pretrained(model_ref, revision=None if args.model_path else revision,
                                     cache_dir=args.hf_cache, torch_dtype=dtype,
                                     **attention).to(device)
    config = (MultilayerUHeadConfig.from_dict(saved_config) if saved_config is not None else
              MultilayerUHeadConfig(architecture=args.uhead_arch, layers=tuple(args.uhead_layers),
                                   hidden_size=int(model.config.hidden_size), head_dim=args.uhead_dim,
                                   n_layers=args.uhead_depth, n_heads=args.uhead_heads))
    if config.hidden_size != model.config.hidden_size:
        raise ValueError("checkpoint hidden width disagrees with the backbone")
    head = MultilayerUHead(config).to(device).eval()
    frozen = copy.deepcopy(head).requires_grad_(False).eval()
    provider = MultilayerHiddenStateProvider(
        model, layers=config.layers, model_identity=backbone, chunk_size=args.hs_chunk_size,
        cache_dir=getattr(args, "feature_cache_dir", None))
    tokenizer = AutoTokenizer.from_pretrained(model_ref, revision=None if args.model_path else revision,
                                             cache_dir=args.hf_cache)
    return tokenizer, provider, head, frozen, config.hidden_size


def load_trained_head(payload, device="cpu"):
    from rl_controller.multilayer_uhead import MultilayerUHead, MultilayerUHeadConfig
    if not isinstance(payload.get("uhead_config"), dict):
        raise ValueError("checkpoint has no multilayer architecture metadata")
    head = MultilayerUHead(MultilayerUHeadConfig.from_dict(payload["uhead_config"])).to(device)
    head.load_state_dict(payload["head"], strict=True)
    weight = payload["readout"]["weight"]
    if weight.ndim != 2 or weight.shape[1] != head.head_dim:
        raise ValueError("controller readout is incompatible with the saved head")
    readout = torch.nn.Linear(weight.shape[1], weight.shape[0]).to(device)
    readout.load_state_dict(payload["readout"], strict=True)
    return head.eval(), readout.eval()


CORE_IDENTITY_FIELDS = ("qkey", "rep")
LABEL_IDENTITY_FIELDS = ("_labels", "probes")
IDENTITY_FIELDS = ("qkey", "rep", "gold", "answer", "correct", "outcome", "blocks",
                   "thinking_tokens", "block_ends", "query_points", "probes",
                   "completion_ids", "_ends", "_prompt_ids", "_labels")


def training_identity(train, hold, architecture):
    hashed = set()

    def digest(rows):
        data = []
        for row in rows:
            missing = [f for f in CORE_IDENTITY_FIELDS if f not in row]
            if missing:
                raise KeyError(f"chain row lacks {missing}; refusing to hash a partial identity")
        for row in sorted(rows, key=lambda row: (row["qkey"], row["rep"])):
            if not any(f in row for f in LABEL_IDENTITY_FIELDS):
                raise KeyError(f"chain {(row['qkey'], row['rep'])} carries none of "
                               f"{list(LABEL_IDENTITY_FIELDS)}: its labels would not be in the "
                               "identity at all")
            present = {f: row[f] for f in IDENTITY_FIELDS if f in row}
            hashed.update(present)
            data.append(present)
        raw = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    train_sha, val_sha = digest(train), digest(hold)
    return {"architecture": architecture, "train_sha256": train_sha, "val_sha256": val_sha,
            "identity_fields": sorted(hashed)}
