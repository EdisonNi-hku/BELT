from __future__ import annotations

import copy
import hashlib
import json
import types
from pathlib import Path


def node_chain(key: str, prompt_ids, full: dict) -> dict:
    qp = [(int(p["block"]), int(p["prefix_len"])) for p in full["positions"][1:] if p["block"] != "cap"]
    return {"qkey": key, "rep": 0, "prompt_ids": [int(t) for t in prompt_ids],
            "completion_ids": list(full["completion_ids"]), "block_ends": list(full["block_ends"]),
            "query_points": [list(q) for q in qp],
            "probes": [{"block": b, "prefix_len": p, "forced": None} for b, p in qp],
            "outcome": full["outcome"], "thinking_tokens": full["thinking_tokens"], "gold": None,
            "answer": None, "correct": None, "blocks": len(full["block_ends"])}


class SharedFeatures:

    def __init__(self, provider):
        self._provider = provider
        self._key = None
        self._val = None
        self.hits = 0

    def __getattr__(self, name):
        return getattr(self._provider, name)

    def get_features(self, sequences, device):
        if len(sequences) != 1:
            return self._provider.get_features(sequences, device)
        key = hashlib.sha256(json.dumps([int(t) for t in sequences[0]]).encode()).hexdigest()
        if key != self._key:
            self._val = self._provider.get_features(sequences, device)
            self._key = key
        else:
            self.hits += 1
        feats, mask, keep, skipped = self._val
        return feats, mask, list(keep), list(skipped)

    def forget(self):
        self._key = self._val = None


def _args(model_path, **kw):
    a = types.SimpleNamespace(model="allenai/Olmo-3-7B-Think", model_path=model_path, revision=None,
                              dtype="bfloat16", hf_cache=None, hs_chunk_size=8192, attention=None,
                              feature_cache_dir=None, head_batch_tokens=4096, allow_device_mismatch=True,
                              chains_per_step=1)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


class Scorers:

    def __init__(self, *, model_path: str, belt: dict | None = None, osp: dict | None = None, device="cuda:0",
                 head_batch_tokens: int = 4096, osp_path: str = "trainer"):
        import torch

        from rl_controller.scripts import iclr_score_pool as sp

        self.torch, self.sp = torch, sp
        self.device = torch.device(device)
        self.model_path = model_path
        self.belt, self.osp = {}, {}
        self.identity = {"belt": {}, "osp": {}, "osp_path": osp_path}
        self.osp_path = osp_path
        backbone = None
        if belt:
            backbone = self._load_belt(belt, head_batch_tokens)
        if osp:
            self._load_osp(osp, backbone)


    def _load_belt(self, runs, head_batch_tokens):
        torch, sp = self.torch, self.sp
        from rl_controller.multilayer_runtime import load_runtime
        from rl_controller.multilayer_uhead import MultilayerUHead, MultilayerUHeadConfig
        from rl_controller.scripts.rl_train import HeadStateFn

        payloads = {}
        for seed, run in sorted(runs.items()):
            path, kind = sp.locate_checkpoint(Path(run), sp.BELIEF_KIND)
            payload = torch.load(path, map_location=self.device)
            sp.require_kind(payload, kind, path)
            if "uhead_config" not in payload:
                raise SystemExit(f"{path}: not a self-contained multilayer reader")
            names = sp.checkpoint_targets(payload)
            sp.check_source_run(Path(run), names)
            summary = json.load(open(Path(run) / "summary.json"))
            reader = sp.multilayer_reader_identity(payload, summary, head_batch_tokens=head_batch_tokens)
            payloads[seed] = (payload, names, path, reader)
        first = next(iter(payloads.values()))[0]
        for seed, (payload, *_r) in payloads.items():
            if payload["feature_contract"] != first["feature_contract"] or payload["backbone"] != first["backbone"]:
                raise SystemExit(f"BELT seed {seed}: feature contract / backbone differ from the first seed's; "
                                 "one shared feature pass would feed it a representation it did not train on")
        args = _args(self.model_path, head_batch_tokens=head_batch_tokens)
        args.hs_chunk_size = first["feature_contract"]["chunk_size"]
        tokenizer, provider, head0, frozen0, _hidden = load_runtime(
            args, self.device, saved_config=first["uhead_config"], saved_backbone=first["backbone"])
        ignored = {"device_name", "device_type"}
        actual, expected = provider.contract, first["feature_contract"]
        if {k: v for k, v in actual.items() if k not in ignored} != {k: v for k, v in expected.items() if k not in ignored}:
            raise SystemExit("multilayer feature contract differs from training")
        self.shared = SharedFeatures(provider)
        self.tokenizer = tokenizer
        for i, (seed, (payload, names, path, reader)) in enumerate(payloads.items()):
            if i == 0:
                head, frozen = head0, frozen0
            else:
                head = MultilayerUHead(MultilayerUHeadConfig.from_dict(payload["uhead_config"])).to(self.device).eval()
                frozen = copy.deepcopy(head).requires_grad_(False).eval()
            state_fn = HeadStateFn(provider=self.shared, head=head, frozen=frozen, device=self.device, raw2048=False)
            state_fn.head_batch_tokens = args.head_batch_tokens
            state_fn.incremental = False
            state_fn.always_frozen = False
            head.load_state_dict(payload["head"])
            head.eval()
            w = payload["readout"]["weight"]
            readout = torch.nn.Linear(w.shape[1], w.shape[0]).to(self.device)
            readout.load_state_dict(payload["readout"])
            mode = sp.check_mode(payload, names, w.shape[0])
            if mode not in sp.BELIEF_MODES:
                raise SystemExit(f"BELT seed {seed}: head mode {mode} is not a belief readout")
            self.belt[seed] = {"state_fn": state_fn, "readout": readout, "names": names}
            self.identity["belt"][seed] = {"checkpoint": str(path), "sha256": _sha(path), "names": list(names),
                                           "reader": reader["schema"], "feature_contract": expected}
        return provider.model

    def score_belt(self, chains, *, seeds=None, pack: int = 1) -> dict:
        import numpy as np

        from rl_controller.scripts.single_cot_train import _forward_with_recovery, prepare_chains, readout_values

        torch, sp = self.torch, self.sp
        prepare_chains(chains)
        for c in chains:
            c["_prompt_ids"] = [int(t) for t in c["prompt_ids"]]
        todo = [c for c in chains if c["_trainable"]]
        out = {c["qkey"]: {} for c in todo}
        for i in range(0, len(todo), pack):
            grp = todo[i:i + pack]
            for seed in (seeds if seeds is not None else list(self.belt)):
                b = self.belt[seed]
                with torch.no_grad():
                    outs = _forward_with_recovery(b["state_fn"], grp, grad=False)
                    for c, o in zip(grp, outs):
                        reprs = o["reprs"].float()
                        vals = readout_values(b["readout"](reprs), b["names"])
                        scores = sp.probe_scores(c, vals, b["names"])
                        rows = reprs[c["_probe_pos"]].to(torch.float16).cpu().numpy()
                        if len(scores) != len(c["probes"]) or rows.shape[0] != len(scores):
                            raise SystemExit(f"chain {c['qkey']}: {len(scores)} scores, {rows.shape[0]} reprs, "
                                             f"{len(c['probes'])} probes")
                        out[c["qkey"]][seed] = {"scores": scores, "reprs": np.ascontiguousarray(rows)}
                b["state_fn"].clear_store()
            self.shared.forget()
        return out


    def _load_osp(self, runs, backbone):
        torch, sp = self.torch, self.sp
        from rl_controller.baselines.ospruner.reader import TopTwoReader, load_backbone

        self.osp_payloads, cfg0 = {}, None
        for name, run in runs.items():
            path, kind = sp.locate_checkpoint(Path(run), sp.TOP2_KIND)
            payload = torch.load(path, map_location="cpu")
            sp.require_kind(payload, kind, path)
            names = sp.checkpoint_targets(payload)
            sp.check_source_run(Path(run), names)
            summary = json.load(open(Path(run) / "summary.json"))
            args = _args(self.model_path)
            cfg = sp.top2_reader_config(summary, payload, names, args)
            mode = sp.check_top2_mode(payload, names, cfg)
            cast = bool(((summary.get("config") or {}).get("training_config") or {}).get("fp32_boundary_cast", False))
            cfg["fp32_boundary_cast"] = cast if self.osp_path == "trainer" else False
            arch = {k: cfg[k] for k in ("head_mode", "n_unfrozen", "fp32_trainable", "dtype", "revision", "outputs",
                                        "fp32_boundary_cast")}
            if cfg0 is None:
                cfg0 = (arch, cfg)
            elif arch != cfg0[0]:
                raise SystemExit(f"OSP {name}: reader architecture {arch} differs from {cfg0[0]}; one reader cannot "
                                 "serve both")
            self.osp_payloads[name] = (payload, names, mode, cfg)
            self.identity["osp"][name] = {"checkpoint": str(path), "sha256": _sha(path), "mode": mode,
                                          "max_seq_len": cfg["max_seq_len"], "epoch": int(payload.get("epoch", -1)),
                                          "fp32_boundary_cast": cfg["fp32_boundary_cast"]}
        arch, cfg = cfg0
        if backbone is None:
            backbone = load_backbone(model_path=cfg["model_path"], model_id=cfg["model"], revision=cfg["revision"],
                                     dtype=cfg["dtype"], cache_dir=None)
        self.reader = TopTwoReader(backbone, head_mode=cfg["head_mode"], n_unfrozen=cfg["n_unfrozen"],
                                   fp32_trainable=cfg["fp32_trainable"],
                                   output_names=None if cfg["head_mode"] == "osp" else cfg["targets"],
                                   fp32_boundary_cast=cfg["fp32_boundary_cast"])
        self.reader.module.to(self.device)
        self.reader.eval()
        self._osp_loaded = None
        if not hasattr(self, "tokenizer"):
            self.tokenizer = sp.top2_tokenizer(cfg, _args(self.model_path))

    def _select_osp(self, name):
        if self._osp_loaded != name:
            payload = self.osp_payloads[name][0]
            self.sp.load_top2_weights(self.reader, payload, self.torch)
            self.reader.eval()
            self._osp_loaded = name

    def score_osp(self, chains, *, names=None, pack: int = 1) -> dict:
        from rl_controller.baselines.ospruner import training as T
        from rl_controller.scripts import iclr_osp_train as ot
        from rl_controller.scripts.single_cot_train import prepare_chains

        sp = self.sp
        prepare_chains(chains)
        todo = [c for c in chains if c["_trainable"]]
        prompts = {c["qkey"]: [int(t) for t in c["prompt_ids"]] for c in todo}
        out = {c["qkey"]: {} for c in todo}
        for name in (names if names is not None else list(self.osp_payloads)):
            payload, tnames, mode, cfg = self.osp_payloads[name]
            if cfg["max_seq_len"]:
                _fits, too_long = ot.split_by_context(todo, prompts, cfg["max_seq_len"])
                if too_long:
                    raise SystemExit(f"OSP {name}: {len(too_long)} node chains exceed max_seq_len {cfg['max_seq_len']}")
            self._select_osp(name)
            forward = sp.top2_forward(self.reader, prompts, {k: len(v) for k, v in prompts.items()},
                                      probes_only=T.gathers_probes_only(mode),
                                      pad_id=getattr(self.tokenizer, "pad_token_id", 0) or 0,
                                      device=self.device, torch=self.torch)
            ns = types.SimpleNamespace(head_mode=mode, chains_per_step=max(1, pack))
            for c in todo:
                c["scores"] = []
            ot._score(self.reader, todo, ns, list(tnames), forward, self.torch, split="alfworld")
            for c in todo:
                out[c["qkey"]][name] = {"scores": [dict(s) for s in c["scores"]]}
        return out


def _sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class ReprStore:

    def __init__(self, lane_dir, tag: str):
        self.dir = Path(lane_dir)
        self.tag = tag
        self.pending, self.rows = [], 0
        self.chunk = len(list(self.dir.glob(f"reprs.{tag}.*.npy")))
        self._maps = {}
        self.index = {}
        own = self.dir / f"reprs.{tag}.index.jsonl"
        if own.exists():
            from alf_tree import trim_torn_tail
            trim_torn_tail(own)
        for idx in sorted(self.dir.glob("reprs.*.index.jsonl")):
            for line in open(idx):
                if line.endswith("\n"):
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.index.setdefault((e["key"], e["scorer"]), e)

    def add(self, key: str, scorer: str, rows) -> dict:
        ref = {"key": key, "scorer": scorer, "file": f"reprs.{self.tag}.{self.chunk}.npy", "row": self.rows,
               "n": int(rows.shape[0])}
        self.pending.append((ref, rows))
        self.rows += int(rows.shape[0])
        return ref

    def flush(self):
        import numpy as np

        if not self.pending:
            return
        arr = np.concatenate([r for _, r in self.pending], 0).astype(np.float16)
        path = self.dir / f"reprs.{self.tag}.{self.chunk}.npy"
        tmp = path.with_name(path.name + ".partial")
        with open(tmp, "wb") as f:
            np.save(f, arr)
        tmp.replace(path)
        with open(self.dir / f"reprs.{self.tag}.index.jsonl", "a") as f:
            for ref, _ in self.pending:
                f.write(json.dumps(ref) + "\n")
                self.index.setdefault((ref["key"], ref["scorer"]), ref)
            f.flush()
        self.pending, self.rows, self.chunk = [], 0, self.chunk + 1

    def load(self, ref: dict):
        for r, rows in self.pending:
            if r is ref or (r["file"] == ref["file"] and r["row"] == ref["row"]):
                return rows
        if ref["file"] not in self._maps:
            self._maps[ref["file"]] = __import__("numpy").load(self.dir / ref["file"], mmap_mode="r")
        return self._maps[ref["file"]][ref["row"]:ref["row"] + ref["n"]]

    def get(self, key: str, scorer: str):
        import numpy as np

        for ref, rows in self.pending:
            if ref["key"] == key and ref["scorer"] == scorer:
                return rows
        ref = self.index[(key, scorer)]
        if ref["file"] not in self._maps:
            self._maps[ref["file"]] = np.load(self.dir / ref["file"], mmap_mode="r")
        return self._maps[ref["file"]][ref["row"]:ref["row"] + ref["n"]]
