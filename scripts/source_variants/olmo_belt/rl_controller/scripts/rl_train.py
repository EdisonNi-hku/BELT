"""Training utilities for the budgeted controller."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from rl_controller.budget_episode import (
    BOUNDARY_ACTIONS, INNER_ACTIONS, episode_reward,
)
from rl_controller.drift_guard import DEFAULT_THRESHOLDS, assert_head_healthy, head_drift
from rl_controller.gpu_pack import PackError, describe_pack, resolve_gpu_mem_util
from rl_controller.rollout_async import ASYNC_STATS, run_async_episodes
from rl_controller.rollout_batch import run_batched_episodes_in_waves
from rl_controller.policy_net import (
    PersistentAnchor, boundary_observation_to_vector, build_policy, load_checkpoint,
    observation_to_vector, save_checkpoint,
)
from rl_controller.rl_metrics import (
    WandbSink, action_metrics, advantage_metrics, boundary_action_metrics,
    draw_outcome_metrics, episode_metrics, group_metrics, head_metrics,
)
from rl_controller.rl_train_core import (
    StratifiedSampler, assemble_batch, entropy_bonus, should_run_guard,
    stop_mass_penalty, stop_mass_multiplier_step,
)
from rl_controller.rollout_contract import (
    build_prompt_ids, resolve_native_root, rewrite_user_instruction, sampling_config,
)
from rl_controller.rollout_runner import (
    STATS as ROLLOUT_STATS,
    AsyncVLLMEngine,
    VLLMEngine,
    build_async_llm,
)
from rl_controller.splits import question_key, three_way_split


def _load_head_runtime(args, device):
    if getattr(args, "uhead_arch", "layer16") != "layer16":
        from rl_controller.multilayer_runtime import load_runtime
        return load_runtime(args, device)
    root = resolve_native_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from luh import AutoUncertaintyHead
    from train_luh.hf_hidden_state_provider import HFHiddenStateProvider
    from train_luh.model_config_compat import load_base_config
    from transformers import AutoTokenizer

    model_ref = args.model_path or args.model
    if args.model_path and not str(args.model_path).rstrip("/").endswith(args.revision):
        raise SystemExit(
            f"--model-path {args.model_path} does not end in the pinned revision "
            f"{args.revision}; refusing to load an unpinned model by path"
        )
    base_config = load_base_config(model_ref, revision=None if args.model_path else args.revision,
                                   cache_dir=args.hf_cache)
    base = SimpleNamespace(config=base_config)
    head = AutoUncertaintyHead.from_pretrained(args.head_checkpoint, base).to(device)
    frozen = AutoUncertaintyHead.from_pretrained(args.head_checkpoint, base).to(device).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    head.eval()

    config = SimpleNamespace(
        model=SimpleNamespace(pretrained_model_name_or_path=model_ref,
                              torch_dtype=args.dtype,
                              revision=None if args.model_path else args.revision),
        hs_device=str(device), hs_chunk_size=args.hs_chunk_size,
        hf_cache=args.hf_cache, hf_token=None,
    )
    provider = HFHiddenStateProvider(config, head, layer_nums=[16])
    tokenizer = AutoTokenizer.from_pretrained(model_ref, revision=None if args.model_path else args.revision,
                                              cache_dir=args.hf_cache, trust_remote_code=True)
    return tokenizer, provider, head, frozen, base_config.hidden_size


class HeadStateFn:

    def __init__(self, *, provider, head, frozen, device, raw2048: bool):
        self.provider = provider
        self.head = head
        self.frozen = frozen
        self.device = device
        self.raw2048 = raw2048
        self._captured: dict = {}
        head.classifier.register_forward_pre_hook(self._cap("repr"))
        frozen.classifier.register_forward_pre_hook(self._cap("repr_frozen"))
        if raw2048:
            head.proj.register_forward_pre_hook(self._cap("raw"))
        self.last_frozen_prob = None
        self.last_head_output = None
        self.anchor_pairs: list = []
        self._init_store()

    def _cap(self, name):
        def hook(module, inputs):
            self._captured[name] = inputs[0]
        return hook


    def _init_store(self):
        self._store: dict = {}
        self._head_store: dict = {}
        self.store_stats = {"provider_calls": 0, "provider_prefixes": 0,
                            "hits": 0, "misses": 0, "tokens_stored": 0,
                            "backbone_seconds": 0.0, "head_seconds": 0.0,
                            "backbone_tokens": 0, "incremental_prefixes": 0,
                            "kv_tokens": 0, "kv_tokens_peak": 0,
                            "head_forward_batches": 0, "head_forward_rows": 0,
                            "feature_seconds": 0.0,
                            "head_incremental_prefixes": 0,
                            "head_incremental_tokens": 0,
                            "head_incremental_batches": 0,
                            "head_incremental_initial_checks": 0,
                            "head_incremental_cached_checks": 0,
                            "head_cache_tokens": 0,
                            "head_cache_tokens_peak": 0,
                            "canonical_prefixes": 0,
                            "canonical_max_batch": 0,
                            "canonical_chain_resets": 0}
        self._store_dtype = None
        self._stamp = 0
        self._canonical_chain_identity = None
        self._canonical_last_item = None

    @staticmethod
    def _prompt_key(prompt_ids):
        return (len(prompt_ids), hash(tuple(prompt_ids)))

    def _lookup(self, prompt_ids, completion_ids):
        entries = self._store.get(self._prompt_key(prompt_ids))
        if not entries:
            return None
        L = len(completion_ids)
        c = None
        for e in entries:
            comp, feats = e[0], e[1]
            if comp.numel() < L:
                continue
            if c is None:
                c = torch.tensor(list(completion_ids), dtype=torch.long)
            if torch.equal(comp[:L], c):
                n = len(prompt_ids) + L - 1
                return feats[:n].to(self.device, torch.float32).unsqueeze(0)
        return None

    def _base_entry(self, prompt_ids, completion_ids):
        entries = self._store.get(self._prompt_key(prompt_ids))
        if not entries:
            return None
        L = len(completion_ids)
        c = torch.tensor(list(completion_ids), dtype=torch.long)
        for e in entries:
            comp = e[0]
            if e[2] is None or comp.numel() >= L:
                continue
            if torch.equal(comp, c[:comp.numel()]):
                return e
        return None

    def _put(self, prompt_ids, completion_ids, feats_row, kv=None):
        key = self._prompt_key(prompt_ids)
        entries = self._store.setdefault(key, [])
        c = torch.tensor(list(completion_ids), dtype=torch.long)
        L = c.numel()
        kept = []
        for e in entries:
            comp, feats = e[0], e[1]
            if comp.numel() >= L and torch.equal(comp[:L], c):
                return
            if comp.numel() < L and torch.equal(comp, c[:comp.numel()]):
                self.store_stats["tokens_stored"] -= feats.shape[0]
                self._drop_kv(e)
                continue
            kept.append(e)
        f = feats_row.detach()
        if self._store_dtype is None:
            self._store_dtype = (torch.bfloat16 if torch.equal(f.to(torch.bfloat16).float(), f.float())
                                 else f.dtype)
        self._stamp += 1
        kept.append([c, f.to(self._store_dtype).contiguous(), kv, self._stamp])
        self.store_stats["tokens_stored"] += f.shape[0]
        if kv is not None:
            self.store_stats["kv_tokens"] += int(kv[0][0].shape[-2])
            self.store_stats["kv_tokens_peak"] = max(self.store_stats["kv_tokens_peak"],
                                                     self.store_stats["kv_tokens"])
        self._store[key] = kept


    def _drop_kv(self, entry):
        if entry[2] is not None:
            self.store_stats["kv_tokens"] -= int(entry[2][0][0].shape[-2])
            entry[2] = None

    def _evict_kv(self):
        if self.store_stats["kv_tokens"] <= self.kv_max_tokens:
            return
        live = [e for entries in self._store.values() for e in entries if e[2] is not None]
        live.sort(key=lambda e: e[3])
        for e in live:
            if self.store_stats["kv_tokens"] <= self.kv_max_tokens:
                break
            self._drop_kv(e)

    def release_kv(self):
        for entries in self._store.values():
            for e in entries:
                self._drop_kv(e)

    batch_tokens = 16384
    always_frozen = False
    _batched_check = None
    incremental = True
    kv_max_tokens = 400_000
    incr_batch_kv_tokens = 65536
    _incr_check = None
    _incr_scale = None
    _incr_supported = None

    def _layer_index(self):
        nums = list(self.provider.layer_nums)
        if len(nums) != 1:
            raise RuntimeError(f"expected one layer, provider has {nums}")
        return nums[0]

    def _incremental_supported(self):
        if self._incr_supported is not None:
            return self._incr_supported
        ok, why = True, ""
        model = getattr(self.provider, "model", None)
        layers = getattr(getattr(model, "model", None), "layers", None)
        layer = self._layer_index()
        try:
            from transformers import DynamicCache
        except Exception as exc:
            ok, why = False, f"transformers has no DynamicCache ({exc})"
        if ok and (layers is None or not (0 < layer < len(layers))):
            ok, why = False, f"no early-exit layer for layer={layer}"
        cfg = getattr(model, "config", None)
        types = list(getattr(cfg, "layer_types", None) or [])
        if ok and types and any(t != "full_attention" for t in types[:layer]):
            ok, why = False, f"cached layers are not all full attention: {sorted(set(types[:layer]))}"
        if ok and not types and getattr(cfg, "sliding_window", None) and getattr(cfg, "use_sliding_window", True):
            ok, why = False, "sliding window attention"
        self._incr_supported = ok
        if not ok:
            print(json.dumps({"incremental_backbone": "unavailable", "reason": why}))
        return ok

    @staticmethod
    def _new_cache():
        from transformers import DynamicCache
        return DynamicCache()

    @staticmethod
    def _cache_kv(cache, layer_idx):
        try:
            return cache[layer_idx]
        except (TypeError, KeyError, IndexError):
            layer = cache.layers[layer_idx]
            return layer.keys, layer.values

    def _forward_padded(self, seqs, *, want_kv=False):
        model = self.provider.model
        layer = self._layer_index()
        B = len(seqs)
        T = max(len(x) for x in seqs)
        ids = torch.zeros(B, T, dtype=torch.long)
        am = torch.zeros(B, T, dtype=torch.long)
        for i, x in enumerate(seqs):
            ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
            am[i, :len(x)] = 1
        ids = ids.to(self.provider.device); am = am.to(self.provider.device)
        cache = self._new_cache() if want_kv else None
        with torch.no_grad():
            hs = self._hidden_at_layer(model, layer, ids, am, past_key_values=cache)
            rows = [hs[i, :len(x)].float().to(self.device) for i, x in enumerate(seqs)]
            kvs = self._split_cache(cache, [0] * B, [len(x) for x in seqs]) if want_kv else [None] * B
        self.store_stats["backbone_tokens"] += sum(len(x) for x in seqs)
        del hs, cache
        return rows, kvs

    def _split_cache(self, cache, starts, ends):
        n_layers = len(cache)
        out = []
        for i, (s, e) in enumerate(zip(starts, ends)):
            row = []
            for l in range(n_layers):
                k, v = self._cache_kv(cache, l)
                row.append((k[i:i + 1, :, s:e].contiguous(), v[i:i + 1, :, s:e].contiguous()))
            out.append(row)
        return out

    def _forward_incremental(self, items):
        model = self.provider.model
        layer = self._layer_index()
        device = self.provider.device
        rows: dict = {}
        kvs: dict = {}
        order = sorted(range(len(items)), key=lambda i: items[i][2])
        pos = 0
        while pos < len(order):
            grp, past_max, new_max = [], 0, 0
            while pos + len(grp) < len(order):
                i = order[pos + len(grp)]
                pm = max(past_max, items[i][2])
                nm = max(new_max, len(items[i][0]) - items[i][2])
                if grp and (len(grp) + 1) * (pm + nm) > self.incr_batch_kv_tokens:
                    break
                grp.append(i); past_max, new_max = pm, nm
            B = len(grp)
            n_layers = len(items[grp[0]][1])
            cache = self._new_cache()
            for l in range(n_layers):
                k0, v0 = items[grp[0]][1][l]
                kp = torch.zeros(B, k0.shape[1], past_max, k0.shape[3], dtype=k0.dtype, device=device)
                vp = torch.zeros(B, v0.shape[1], past_max, v0.shape[3], dtype=v0.dtype, device=device)
                for r, i in enumerate(grp):
                    k, v = items[i][1][l]
                    pl = items[i][2]
                    kp[r, :, past_max - pl:] = k[0]
                    vp[r, :, past_max - pl:] = v[0]
                cache.update(kp, vp, l, {})
                del kp, vp
            ids = torch.zeros(B, new_max, dtype=torch.long)
            am = torch.zeros(B, past_max + new_max, dtype=torch.long)
            pids = torch.zeros(B, new_max, dtype=torch.long)
            new_lens = []
            for r, i in enumerate(grp):
                full, _, pl = items[i][0], items[i][1], items[i][2]
                new = full[pl:]
                new_lens.append(len(new))
                ids[r, :len(new)] = torch.tensor(new, dtype=torch.long)
                am[r, past_max - pl:past_max + len(new)] = 1
                pids[r] = torch.arange(pl, pl + new_max, dtype=torch.long)
            ids = ids.to(device); am = am.to(device); pids = pids.to(device)
            cpos = torch.arange(past_max, past_max + new_max, device=device)
            with torch.no_grad():
                hs = self._hidden_at_layer(model, layer, ids, am, past_key_values=cache,
                                           cache_position=cpos, position_ids=pids)
                new_kv = self._split_cache(cache, [past_max - items[i][2] for i in grp],
                                           [past_max + n for n in new_lens])
                for r, i in enumerate(grp):
                    base_rows = items[i][3]
                    rows[i] = torch.cat([base_rows.to(self.device, torch.float32),
                                         hs[r, :new_lens[r]].float().to(self.device)], 0)
                    kvs[i] = new_kv[r]
            self.store_stats["backbone_tokens"] += sum(new_lens)
            check = ([items[i][0] for i in grp], [rows[i] for i in grp]) if self._incr_check is None else None
            del hs, cache, new_kv
            if check is not None:
                self._check_incremental(*check)
            pos += len(grp)
        return [rows[i] for i in range(len(items))], [kvs[i] for i in range(len(items))]

    @staticmethod
    def _hidden_at_layer(model, layer, ids, am, *, past_key_values=None,
                         cache_position=None, position_ids=None):
        use_cache = past_key_values is not None
        extra = {}
        if use_cache:
            extra["past_key_values"] = past_key_values
            if cache_position is not None:
                extra["cache_position"] = cache_position
            if position_ids is not None:
                extra["position_ids"] = position_ids
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None or not (0 < layer < len(layers)):
            out = model(input_ids=ids, attention_mask=am, output_hidden_states=True,
                        use_cache=use_cache, **extra)
            return out.hidden_states[layer]

        class _EarlyExit(Exception):
            pass

        captured = {}

        def hook(_mod, _inp, out):
            captured["hs"] = out[0] if isinstance(out, (tuple, list)) else out
            raise _EarlyExit

        handle = layers[layer - 1].register_forward_hook(hook)
        try:
            try:
                model(input_ids=ids, attention_mask=am, use_cache=use_cache, **extra)
            except _EarlyExit:
                pass
        finally:
            handle.remove()
        if "hs" not in captured:
            raise RuntimeError(f"early-exit hook on layer {layer - 1} never fired")
        return captured["hs"]

    def _check_incremental(self, seqs, got):
        refs, _ = self._forward_padded([list(x) for x in seqs])
        diff = max(float((r - g).abs().max()) for r, g in zip(refs, got))
        scale = max(float(r.abs().max()) for r in refs)
        self._incr_check = diff
        self._incr_scale = scale
        if diff > 0.02 * max(scale, 1.0):
            raise RuntimeError(
                f"incremental backbone forward disagrees with a full forward on the "
                f"same tokens ({len(seqs)} rows, longest {max(len(x) for x in seqs)}): "
                f"max abs diff {diff:.4g} vs feature scale {scale:.4g}")

    def _compute_and_store(self, prefixes):
        todo = [(list(p), list(c)) for p, c in prefixes]
        if not todo:
            return
        inputs = [p + c for p, c in todo]
        t = time.time()
        chunk_limit = int(getattr(self.provider, "chunk_size", 0) or 0)
        batched_ok = hasattr(self.provider, "model") and hasattr(self.provider, "layer_nums")
        rows: dict = {}
        kvs: dict = {}
        incr_items, incr_idx = [], []
        long_idx, full_idx = [], []
        want_kv = bool(self.incremental) and batched_ok and self._incremental_supported()
        for i in range(len(inputs)):
            base = self._base_entry(*todo[i]) if want_kv else None
            past = int(base[0].numel()) + len(todo[i][0]) if base is not None else 0
            if base is not None and base[1].shape[0] >= past and len(inputs[i]) > past:
                incr_idx.append(i)
                incr_items.append((inputs[i], base[2], past, base[1][:past]))
            elif (chunk_limit and len(inputs[i]) > chunk_limit) or not batched_ok:
                long_idx.append(i)
            else:
                full_idx.append(i)
        for i in long_idx:
            with torch.no_grad():
                feats, am, keep, skipped = self.provider.get_features([inputs[i]], self.device)
            if keep != [0] or skipped:
                raise RuntimeError(f"provider skipped a prefix of length {len(inputs[i])}")
            rows[i] = feats[0, :len(inputs[i]) - 1]
            self.store_stats["provider_calls"] += 1
            self.store_stats["backbone_tokens"] += len(inputs[i])
        order = sorted(full_idx, key=lambda i: -len(inputs[i]))
        pos = 0
        while pos < len(order):
            T = len(inputs[order[pos]])
            n = max(1, min(len(order) - pos, self.batch_tokens // T))
            group = order[pos:pos + n]
            outs, outs_kv = self._forward_padded([inputs[i] for i in group], want_kv=want_kv)
            for i, r, kv in zip(group, outs, outs_kv):
                rows[i] = r
                kvs[i] = kv
            self.store_stats["provider_calls"] += 1
            if self._batched_check is None:
                with torch.no_grad():
                    f1, _, keep, skipped = self.provider.get_features([inputs[group[0]]], self.device)
                if keep != [0] or skipped:
                    raise RuntimeError("provider skipped the check prefix")
                ref = f1[0, :len(inputs[group[0]]) - 1]
                diff = float((ref - rows[group[0]][:ref.shape[0]]).abs().max())
                scale = float(ref.abs().max())
                self._batched_check = diff
                if diff > 0.02 * max(scale, 1.0):
                    raise RuntimeError(
                        f"batched backbone forward disagrees with the single-sequence "
                        f"provider: max abs diff {diff:.4g} vs feature scale {scale:.4g}")
            pos += n
        if incr_items:
            outs, outs_kv = self._forward_incremental(incr_items)
            for i, r, kv in zip(incr_idx, outs, outs_kv):
                rows[i] = r
                kvs[i] = kv
            self.store_stats["provider_calls"] += 1
            self.store_stats["incremental_prefixes"] += len(incr_items)
        self.store_stats["backbone_seconds"] += time.time() - t
        self.store_stats["provider_prefixes"] += len(todo)
        for i, (p, c) in enumerate(todo):
            self._put(p, c, rows[i], kvs.get(i))
        del rows, kvs
        self._evict_kv()

    def prefetch(self, prefixes):
        todo = [(p, c) for p, c in prefixes if self._lookup(p, c) is None]
        self._compute_and_store(todo)

    def clear_store(self):
        self.release_kv()
        self._store.clear()
        self.store_stats["tokens_stored"] = 0
        self.store_stats["kv_tokens"] = 0
        self._head_store.clear()
        self.store_stats["head_cache_tokens"] = 0
        self._canonical_chain_identity = None
        self._canonical_last_item = None

    def features_for(self, prompt_ids, completion_ids):
        cached = self._lookup(prompt_ids, completion_ids)
        if cached is not None:
            self.store_stats["hits"] += 1
            T = cached.shape[1]
            return cached, torch.ones(1, T, dtype=torch.long, device=self.device)
        self.store_stats["misses"] += 1
        self._compute_and_store([(prompt_ids, completion_ids)])
        cached = self._lookup(prompt_ids, completion_ids)
        if cached is None:
            raise RuntimeError("feature store cannot serve a prefix it just stored")
        T = cached.shape[1]
        return cached, torch.ones(1, T, dtype=torch.long, device=self.device)

    @staticmethod
    def _end_columns(prompt_len: int, block_ends, T: int):
        from rl_controller.raw_tap import end_feature_index
        ends = [int(e) for e in block_ends]
        if not ends or any(e <= 0 for e in ends) or ends != sorted(ends):
            raise RuntimeError(f"block_ends must be positive and increasing, got {ends[:8]}...")
        cols = [end_feature_index(prompt_len, e - 1) for e in ends]
        if cols[-1] >= T:
            raise RuntimeError(f"block end column {cols[-1]} beyond feature length {T}")
        return cols

    @staticmethod
    def _mask_from_columns(cols, T: int, device):
        mask = torch.zeros(len(cols), T, dtype=torch.long, device=device)
        for j, col in enumerate(cols):
            mask[j, max(0, col - 1):col + 1] = 1
        return mask

    @staticmethod
    def claim_mask(prompt_len: int, block_ends, T: int, device):
        return HeadStateFn._mask_from_columns(
            HeadStateFn._end_columns(prompt_len, block_ends, T), T, device)

    def __call__(self, prompt_ids, completion_ids, block, block_ends=None):
        from rl_controller.raw_tap import end_feature_index

        if block_ends is None:
            raise RuntimeError("state_fn needs block_ends (every block end so far); the driver passes them")
        if int(block_ends[-1]) != len(completion_ids):
            raise RuntimeError(f"last block end {block_ends[-1]} != completion length {len(completion_ids)}")
        features, attention_mask = self.features_for(prompt_ids, completion_ids)
        t_head = time.time()
        T = features.shape[1]
        end_col = end_feature_index(len(prompt_ids), len(completion_ids) - 1)
        mask = self.claim_mask(len(prompt_ids), block_ends, T, self.device)
        logits = self.head.forward_from_features(features=features, attention_mask=attention_mask, claims=[mask])
        self.last_head_output = logits
        repr_ = self._captured.pop("repr")
        if repr_.shape[0] != mask.shape[0]:
            raise RuntimeError(f"head saw {repr_.shape[0]} claims for {mask.shape[0]} block ends")
        last = logits.reshape(-1)[-1]
        belief = torch.sigmoid(last)
        self.last_logit = last
        out = {"repr": repr_[-1], "belief": belief,
               "_prompt_ids": list(prompt_ids), "_completion_ids": list(completion_ids),
               "_block_ends": [int(e) for e in block_ends]}
        if self.raw2048:
            raw = self._captured.pop("raw")
            out["raw"] = raw[0, end_col]
        if torch.is_grad_enabled() or self.always_frozen:
            with torch.no_grad():
                fl = self.frozen.forward_from_features(features=features.detach(), attention_mask=attention_mask, claims=[mask])
                self.last_frozen_prob = float(torch.sigmoid(fl.reshape(-1)[-1]).item())
                frozen_repr = self._captured.pop("repr_frozen")[-1].detach()
            self.anchor_pairs.append((out["repr"], frozen_repr))
        self.store_stats["head_seconds"] += time.time() - t_head
        return out

    def forward_draw(self, prompt_ids, completion_ids, block_ends):
        if not block_ends or int(block_ends[-1]) != len(completion_ids):
            raise RuntimeError("forward_draw needs the draw's full completion and its block ends")
        return self.forward_many([(prompt_ids, completion_ids, block_ends)])[0]


    head_batch_tokens = 32768
    _head_batch_check = None
    _head_batch_scale = None
    _head_batch_checked_b = 0

    head_incremental_batch_tokens = 350_000
    head_incremental_attention_elements = 160_000_000
    _head_incremental_initial_check = None
    _head_incremental_initial_scale = None
    _head_incremental_cached_check = None
    _head_incremental_cached_scale = None

    def _batched_head_available(self) -> bool:
        return all(hasattr(self.head, a) and hasattr(self.frozen, a)
                   for a in ("proj", "step_end_embedding", "transformer_encoder", "classifier",
                             "mask_future_tokens", "_can_stream_causal",
                             "_encode_causal_streaming", "_build_causal_step_mask"))

    def _incremental_head_available(self) -> bool:
        if not self._batched_head_available() or self.head.training:
            return False
        if not self.head.mask_future_tokens or self.always_frozen:
            return False
        encoder = self.head.transformer_encoder
        for layer in encoder.layers:
            mha = layer.self_attn
            if getattr(layer, "norm_first", False):
                return False
            if not mha.batch_first or not mha._qkv_same_embed_dim:
                return False
            if mha.bias_k is not None or mha.bias_v is not None or mha.add_zero_attn:
                return False
        return True

    def _head_entries_for(self, prompt_ids, completion_ids, block_ends):
        entries = self._head_store.get(self._prompt_key(prompt_ids), ())
        current = torch.tensor(list(completion_ids), dtype=torch.long)
        current_ends = tuple(int(end) for end in block_ends)
        exact = None
        best = None
        for entry in entries:
            previous = entry["completion"]
            if previous.numel() > current.numel():
                continue
            if not torch.equal(previous, current[:previous.numel()]):
                continue
            required_ends = tuple(
                end for end in current_ends if end <= previous.numel()
            )
            if entry["block_ends"] != required_ends:
                continue
            if previous.numel() == current.numel():
                exact = entry
            elif best is None or previous.numel() > best["completion"].numel():
                best = entry
        return exact, best

    def _put_head_entry(
        self, prompt_ids, completion_ids, block_ends, feature_len,
        cache, representation, logit,
    ):
        key = self._prompt_key(prompt_ids)
        current = torch.tensor(list(completion_ids), dtype=torch.long)
        current_ends = tuple(int(end) for end in block_ends)
        entries = self._head_store.setdefault(key, [])
        kept = []
        for entry in entries:
            previous = entry["completion"]
            token_prefix = (
                previous.numel() <= current.numel()
                and torch.equal(previous, current[:previous.numel()])
            )
            mark_prefix = entry["block_ends"] == tuple(
                end for end in current_ends if end <= previous.numel()
            )
            if token_prefix and mark_prefix:
                if previous.numel() == current.numel():
                    raise RuntimeError("duplicate prefix UHead entry was recomputed")
                self.store_stats["head_cache_tokens"] -= entry["feature_len"]
                continue
            kept.append(entry)
        kept.append({
            "completion": current,
            "block_ends": current_ends,
            "feature_len": int(feature_len),
            "cache": cache,
            "representation": representation.detach(),
            "logit": logit.detach(),
        })
        self.store_stats["head_cache_tokens"] += int(feature_len)
        self.store_stats["head_cache_tokens_peak"] = max(
            self.store_stats["head_cache_tokens_peak"],
            self.store_stats["head_cache_tokens"],
        )
        self._head_store[key] = kept

    @staticmethod
    def _head_incremental_group(module, records):
        batch = len(records)
        new_max = max(record["new"].shape[0] for record in records)
        past_lens = [record["past_len"] for record in records]
        past_max = max(past_lens)
        raw_dim = records[0]["new"].shape[1]
        raw = records[0]["new"].new_zeros(batch, new_max, raw_dim)
        markers = torch.zeros(batch, new_max, dtype=torch.long, device=raw.device)
        new_lens = []
        for row, record in enumerate(records):
            length = record["new"].shape[0]
            new_lens.append(length)
            raw[row, :length] = record["new"]
            markers[row, record["marker_offsets"]] = 1
        x = module.proj(raw) + module.step_end_embedding(markers)
        new_caches = [[] for _ in records]
        for layer_index, layer in enumerate(module.transformer_encoder.layers):
            mha = layer.self_attn
            dim = x.shape[-1]
            heads = mha.num_heads
            head_dim = dim // heads
            query, new_key, new_value = (
                value.view(batch, new_max, heads, head_dim).transpose(1, 2)
                for value in F.linear(x, mha.in_proj_weight, mha.in_proj_bias).chunk(3, -1)
            )
            total = past_max + new_max
            key = new_key.new_zeros(batch, heads, total, head_dim)
            value = new_value.new_zeros(batch, heads, total, head_dim)
            for row, record in enumerate(records):
                past = record["cache"]
                if past_lens[row]:
                    old_key, old_value = past[layer_index]
                    start = past_max - past_lens[row]
                    key[row, :, start:past_max] = old_key[0]
                    value[row, :, start:past_max] = old_value[0]
                length = new_lens[row]
                key[row, :, past_max:past_max + length] = new_key[row, :, :length]
                value[row, :, past_max:past_max + length] = new_value[row, :, :length]
            if past_max == 0:
                attended = F.scaled_dot_product_attention(
                    query, key, value, is_causal=True, dropout_p=0.0
                )
            else:
                key_position = torch.arange(total, device=x.device).view(1, 1, 1, total)
                query_position = torch.arange(new_max, device=x.device).view(1, 1, new_max, 1)
                starts = (past_max - torch.as_tensor(
                    past_lens, device=x.device
                )).view(batch, 1, 1, 1)
                allowed = ((key_position >= starts)
                           & (key_position <= past_max + query_position))
                attended = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=allowed, dropout_p=0.0
                )
            attended = mha.out_proj(
                attended.transpose(1, 2).reshape(batch, new_max, dim)
            )
            x = layer.norm1(x + layer.dropout1(attended))
            x = layer.norm2(
                x + layer.dropout2(
                    layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                )
            )
            for row, record in enumerate(records):
                length = new_lens[row]
                current_key = new_key[row:row + 1, :, :length].contiguous()
                current_value = new_value[row:row + 1, :, :length].contiguous()
                if past_lens[row]:
                    old_key, old_value = record["cache"][layer_index]
                    current_key = torch.cat((old_key, current_key), dim=2)
                    current_value = torch.cat((old_value, current_value), dim=2)
                new_caches[row].append((current_key, current_value))
        visible = x
        if module.transformer_encoder.norm is not None:
            visible = module.transformer_encoder.norm(visible)
        representations = torch.stack([
            visible[row, length - 1] for row, length in enumerate(new_lens)
        ])
        logits = module.classifier(representations).reshape(-1)
        return representations, logits, tuple(tuple(cache) for cache in new_caches)

    def _pack_head_incremental(self, records):
        order = sorted(
            range(len(records)),
            key=lambda index: -(
                records[index]["new"].shape[0]
                * (records[index]["past_len"] + records[index]["new"].shape[0])
            ),
        )
        groups = []
        position = 0
        while position < len(order):
            group = []
            past_max = new_max = 0
            while position + len(group) < len(order):
                index = order[position + len(group)]
                proposed_past = max(past_max, records[index]["past_len"])
                proposed_new = max(new_max, records[index]["new"].shape[0])
                count = len(group) + 1
                tokens = count * (proposed_past + proposed_new)
                elements = count * proposed_new * (proposed_past + proposed_new)
                if group and (
                    tokens > self.head_incremental_batch_tokens
                    or elements > self.head_incremental_attention_elements
                ):
                    break
                group.append(index)
                past_max, new_max = proposed_past, proposed_new
            groups.append(group)
            position += len(group)
        return groups

    def _check_head_incremental(self, items, outputs, indices, *, cached):
        reference = self.forward_many([items[index] for index in indices])
        difference = 0.0
        scale = 0.0
        for index, expected in zip(indices, reference):
            got_repr = outputs[index]["reprs"][-1]
            got_logit = outputs[index]["logits"][-1]
            ref_repr = expected["reprs"][-1]
            ref_logit = expected["logits"][-1]
            difference = max(
                difference,
                float((got_repr - ref_repr).abs().max()),
                float((got_logit - ref_logit).abs()),
            )
            scale = max(
                scale, float(ref_repr.abs().max()), float(ref_logit.abs())
            )
        label = "cached" if cached else "initial"
        setattr(self, f"_head_incremental_{label}_check", difference)
        setattr(self, f"_head_incremental_{label}_scale", scale)
        self.store_stats[f"head_incremental_{label}_checks"] += 1

    def forward_many_prefixes(self, items):
        if not items:
            return []
        if torch.is_grad_enabled():
            raise RuntimeError("prefix-incremental UHead requires inference/no-grad mode")
        if not self._incremental_head_available():
            raise RuntimeError("head does not satisfy the frozen causal prefix-cache contract")
        records = []
        outputs = [None] * len(items)
        pending = {}
        aliases = {}
        for item_index, (prompt_ids, completion_ids, block_ends) in enumerate(items):
            if not block_ends or int(block_ends[-1]) != len(completion_ids):
                raise RuntimeError("prefix-incremental UHead needs ends through the current prefix")
            signature = (
                tuple(prompt_ids), tuple(completion_ids), tuple(int(end) for end in block_ends)
            )
            if signature in pending:
                aliases[item_index] = pending[signature]
                continue
            features, _attention = self.features_for(prompt_ids, completion_ids)
            row = features[0]
            columns = self._end_columns(len(prompt_ids), block_ends, row.shape[0])
            exact, base = self._head_entries_for(
                prompt_ids, completion_ids, block_ends
            )
            if exact is not None:
                outputs[item_index] = {
                    "reprs": exact["representation"].unsqueeze(0),
                    "logits": exact["logit"].reshape(1),
                    "frozen_reprs": None,
                    "frozen_logits": None,
                    "raw": None,
                }
                pending[signature] = item_index
                continue
            past_len = 0 if base is None else int(base["feature_len"])
            if not 0 <= past_len < row.shape[0]:
                raise RuntimeError("prefix UHead cache does not strictly precede current features")
            marker_offsets = [column - past_len for column in columns if column >= past_len]
            if not marker_offsets or marker_offsets[-1] != row.shape[0] - past_len - 1:
                raise RuntimeError("current probe end is absent from new UHead rows")
            records.append({
                "item_index": item_index,
                "prompt": prompt_ids,
                "completion": completion_ids,
                "block_ends": block_ends,
                "feature_len": int(row.shape[0]),
                "past_len": past_len,
                "new": row[past_len:],
                "marker_offsets": marker_offsets,
                "cache": () if base is None else base["cache"],
            })
            pending[signature] = item_index
        caches = [None] * len(records)
        for group in self._pack_head_incremental(records):
            representations, logits, group_caches = self._head_incremental_group(
                self.head, [records[index] for index in group]
            )
            self.store_stats["head_incremental_batches"] += 1
            for row, index in enumerate(group):
                item_index = records[index]["item_index"]
                outputs[item_index] = {
                    "reprs": representations[row:row + 1],
                    "logits": logits[row:row + 1],
                    "frozen_reprs": None,
                    "frozen_logits": None,
                    "raw": None,
                }
                caches[index] = group_caches[row]
        self.store_stats["head_incremental_prefixes"] += len(records)
        self.store_stats["head_incremental_tokens"] += sum(
            record["new"].shape[0] for record in records
        )
        initial = [record["item_index"] for record in records
                   if record["past_len"] == 0]
        cached = [record["item_index"] for record in records
                  if record["past_len"] > 0]
        if initial and self._head_incremental_initial_check is None:
            self._check_head_incremental(items, outputs, initial, cached=False)
        if cached and self._head_incremental_cached_check is None:
            self._check_head_incremental(items, outputs, cached, cached=True)
        for record, cache in zip(records, caches):
            current = outputs[record["item_index"]]
            self._put_head_entry(
                record["prompt"], record["completion"], record["block_ends"],
                record["feature_len"], cache,
                current["reprs"][0], current["logits"][0],
            )
        for item_index, representative in aliases.items():
            outputs[item_index] = outputs[representative]
        if any(output is None for output in outputs):
            raise RuntimeError("prefix-incremental UHead left an output unassigned")
        return outputs

    def forward_one_prefix_canonical(self, item, *, chain_identity):
        if torch.is_grad_enabled() or not self._incremental_head_available():
            raise RuntimeError("canonical prefix mode requires frozen inference")
        if any(parameter.requires_grad for parameter in self.head.parameters()):
            raise RuntimeError("canonical prefix mode requires frozen inference")
        if (not isinstance(chain_identity, tuple) or len(chain_identity) != 2
                or type(chain_identity[0]) is not str or not chain_identity[0]
                or type(chain_identity[1]) is not int or chain_identity[1] < 0):
            raise RuntimeError("canonical prefix mode requires an exact chain identity")
        try:
            prompt_ids, completion_ids, block_ends = item
        except (TypeError, ValueError) as error:
            raise RuntimeError("canonical prefix item must be prompt/completion/marks") from error
        current = (
            tuple(prompt_ids), tuple(completion_ids),
            tuple(int(end) for end in block_ends),
        )
        if self._canonical_chain_identity != chain_identity:
            if (self._canonical_chain_identity is not None
                    or self._store or self._head_store):
                self.clear_store()
            self._canonical_chain_identity = chain_identity
            self.store_stats["canonical_chain_resets"] += 1
        elif self._canonical_last_item is not None:
            previous_prompt, previous_completion, previous_ends = self._canonical_last_item
            observed_previous_ends = tuple(
                end for end in current[2] if end <= len(previous_completion)
            )
            if (current[0] != previous_prompt
                    or len(current[1]) <= len(previous_completion)
                    or current[1][:len(previous_completion)] != previous_completion
                    or observed_previous_ends != previous_ends):
                raise RuntimeError(
                    "canonical prefix requests must strictly extend one token/mark chain"
                )
        self.prefetch([(prompt_ids, completion_ids)])
        output = self.forward_many_prefixes([item])[0]
        self._canonical_last_item = current
        self.store_stats["canonical_prefixes"] += 1
        self.store_stats["canonical_max_batch"] = max(
            self.store_stats["canonical_max_batch"], 1,
        )
        return output

    def forward_many(self, items):
        if not items:
            return []
        lengths = []
        for prompt_ids, completion_ids, block_ends in items:
            if not block_ends or int(block_ends[-1]) != len(completion_ids):
                raise RuntimeError(
                    f"forward_many needs each draw's full completion and its block ends; "
                    f"last end {list(block_ends)[-1:]} != completion length {len(completion_ids)}")
            lengths.append(len(prompt_ids) + len(completion_ids) - 1)
        out = [None] * len(items)
        batched = self._batched_head_available()
        order = sorted(range(len(items)), key=lambda i: -lengths[i])
        pos = 0
        while pos < len(order):
            T = lengths[order[pos]]
            n = (max(1, min(len(order) - pos, self.head_batch_tokens // max(1, T)))
                 if batched else 1)
            grp = order[pos:pos + n]
            t_feat = time.time()
            rows, cols = [], []
            for i in grp:
                prompt_ids, completion_ids, block_ends = items[i]
                features, _am = self.features_for(prompt_ids, completion_ids)
                if features.shape[1] != lengths[i]:
                    raise RuntimeError(
                        f"the feature store served {features.shape[1]} rows for a prefix of "
                        f"{lengths[i]}; the packing and the store disagree")
                rows.append(features[0])
                cols.append(self._end_columns(len(prompt_ids), block_ends, features.shape[1]))
            self.store_stats["feature_seconds"] += time.time() - t_feat
            t_head = time.time()
            if batched:
                for i, res in zip(grp, self._head_group(rows, cols)):
                    out[i] = res
            else:
                out[grp[0]] = self._forward_one(rows[0], cols[0])
            self.store_stats["head_seconds"] += time.time() - t_head
            del rows
            pos += n
        return out

    def _forward_one(self, row, cols):
        features = row.unsqueeze(0)
        T = features.shape[1]
        attention_mask = torch.ones(1, T, dtype=torch.long, device=features.device)
        mask = self._mask_from_columns(cols, T, self.device)
        logits = self.head.forward_from_features(features=features, attention_mask=attention_mask,
                                                 claims=[mask])
        reprs = self._captured.pop("repr")
        if reprs.shape[0] != mask.shape[0]:
            raise RuntimeError(f"head saw {reprs.shape[0]} claims for {mask.shape[0]} block ends")
        out = {"reprs": reprs, "logits": logits.reshape(-1), "frozen_reprs": None,
               "frozen_logits": None, "raw": None}
        if self.raw2048:
            out["raw"] = self._captured.pop("raw")[0, cols]
        if torch.is_grad_enabled() or self.always_frozen:
            with torch.no_grad():
                fl = self.frozen.forward_from_features(features=features.detach(),
                                                       attention_mask=attention_mask, claims=[mask])
                out["frozen_reprs"] = self._captured.pop("repr_frozen").detach()
                out["frozen_logits"] = fl.reshape(-1).detach()
        return out

    def _head_group(self, rows, cols):
        B = len(rows)
        T = max(r.shape[0] for r in rows)
        X = rows[0].new_zeros(B, T, rows[0].shape[1])
        am = torch.zeros(B, T, dtype=torch.long, device=X.device)
        for i, r in enumerate(rows):
            X[i, :r.shape[0]] = r
            am[i, :r.shape[0]] = 1
        reprs, logits = self._head_math(self.head, X, am, cols)
        self.store_stats["head_forward_batches"] += 1
        self.store_stats["head_forward_rows"] += B
        if self._head_batch_checked_b < min(2, B):
            self._check_batched_head(X, am, cols)
        outs, p = [], 0
        for i, c in enumerate(cols):
            n = len(c)
            o = {"reprs": reprs[p:p + n], "logits": logits[p:p + n],
                 "frozen_reprs": None, "frozen_logits": None, "raw": None}
            if self.raw2048:
                o["raw"] = X[i, c]
            outs.append(o)
            p += n
        if torch.is_grad_enabled() or self.always_frozen:
            with torch.no_grad():
                fr, fl = self._head_math(self.frozen, X.detach(), am, cols)
            p = 0
            for i, c in enumerate(cols):
                n = len(c)
                outs[i]["frozen_reprs"] = fr[p:p + n].detach()
                outs[i]["frozen_logits"] = fl[p:p + n].detach()
                p += n
        return outs

    def _head_math(self, module, X, am, cols):
        B, T = am.shape
        features = module.proj(X)
        end_mask = torch.zeros(B, T, dtype=torch.long, device=X.device)
        for i, c in enumerate(cols):
            end_mask[i, torch.as_tensor(c, dtype=torch.long, device=X.device)] = 1
        seq = features + module.step_end_embedding(end_mask)
        padding = (am == 0)
        if module._can_stream_causal(padding):
            enc = module._encode_causal_streaming(seq)
        else:
            attn = (module._build_causal_step_mask(T, seq.device)
                    if module.mask_future_tokens else None)
            enc = module.transformer_encoder(seq, mask=attn, src_key_padding_mask=padding,
                                             is_causal=bool(module.mask_future_tokens))
        end_reprs = torch.cat([enc[i, c] for i, c in enumerate(cols)], 0)
        logits = module.classifier(end_reprs).reshape(-1)
        for k in ("repr", "repr_frozen", "raw"):
            self._captured.pop(k, None)
        return end_reprs, logits

    def _check_batched_head(self, X, am, cols):
        was_training = self.head.training
        self.head.eval()
        try:
            with torch.no_grad():
                b_reprs, b_logits = self._head_math(self.head, X, am, cols)
                diff, scale, p = 0.0, 0.0, 0
                for i, c in enumerate(cols):
                    Ti = int(am[i].sum().item())
                    mask = self._mask_from_columns(c, Ti, X.device)
                    lg = self.head.forward_from_features(features=X[i:i + 1, :Ti],
                                                         attention_mask=am[i:i + 1, :Ti],
                                                         claims=[mask]).reshape(-1)
                    rp = self._captured.pop("repr")
                    self._captured.pop("raw", None)
                    n = len(c)
                    if rp.shape[0] != n or lg.numel() != n:
                        raise RuntimeError(
                            f"the vendored head returned {rp.shape[0]} representations and "
                            f"{lg.numel()} logits for {n} claims")
                    diff = max(diff, float((rp - b_reprs[p:p + n]).abs().max()),
                               float((lg - b_logits[p:p + n]).abs().max()))
                    scale = max(scale, float(rp.abs().max()), float(lg.abs().max()))
                    p += n
        finally:
            if was_training:
                self.head.train()
        self._head_batch_check = diff
        self._head_batch_scale = scale
        self._head_batch_checked_b = max(self._head_batch_checked_b, len(cols))
        if diff > 0.01 * max(scale, 1.0):
            raise RuntimeError(
                f"the batched head forward disagrees with the vendored per-sample "
                f"forward_from_features ({len(cols)} rows, longest {int(am.shape[1])}): "
                f"max abs diff {diff:.4g} vs representation scale {scale:.4g}")

    def call_many(self, items):
        fwds = self.forward_many([(p, c, e) for p, c, _b, e in items])
        outs = []
        for (prompt_ids, completion_ids, _block, block_ends), fwd in zip(items, fwds):
            last = fwd["logits"].reshape(-1)[-1]
            o = {"repr": fwd["reprs"][-1], "belief": torch.sigmoid(last),
                 "_prompt_ids": list(prompt_ids), "_completion_ids": list(completion_ids),
                 "_block_ends": [int(e) for e in block_ends]}
            if self.raw2048:
                o["raw"] = fwd["raw"][-1]
            self.last_logit = last
            self.last_head_output = fwd["logits"]
            if fwd["frozen_reprs"] is not None:
                self.last_frozen_prob = float(torch.sigmoid(fwd["frozen_logits"][-1]).item())
                self.anchor_pairs.append((o["repr"], fwd["frozen_reprs"][-1]))
            outs.append(o)
        return outs

    def take_anchor_pairs(self):
        pairs, self.anchor_pairs = self.anchor_pairs, []
        return pairs


class TorchPolicy:

    def __init__(self, net, *, raw2048: bool, device, sample: bool = True):
        self.net = net
        self.raw2048 = raw2048
        self.device = device
        self.sample = sample
        self.trace: list = []
        self._ctx = None

    def _vec(self, obs, level):
        if level == "outer":
            return torch.tensor(boundary_observation_to_vector(obs), dtype=torch.float32, device=self.device)
        placeholder = {**obs, "repr": [0.0] * 768}
        if self.raw2048:
            placeholder["raw"] = [0.0] * 2048
        vec = observation_to_vector(placeholder, raw2048=self.raw2048)
        n_scalars = len(vec) - 768 - (2048 if self.raw2048 else 0)
        parts = [obs["repr"].reshape(-1).float()]
        if self.raw2048:
            parts.append(obs["raw"].reshape(-1).float())
        parts.append(torch.tensor(vec[-n_scalars:-1], dtype=torch.float32, device=self.device))
        belief = obs["belief"]
        parts.append((belief if torch.is_tensor(belief)
                      else torch.tensor(float(belief), device=self.device)).reshape(1).float())
        out = torch.cat(parts)
        if out.numel() != len(vec):
            raise RuntimeError(
                f"inner vector is {out.numel()} wide but the contract says {len(vec)}: "
                "the live-tensor substitution and the feature contract disagree"
            )
        return out

    def inner(self, obs):
        with torch.no_grad():
            x = self._vec(obs, "inner")
            logits = self.net["inner"](x.unsqueeze(0), obs["block"])[0]
            probs = torch.softmax(logits, -1)
        idx = int(torch.multinomial(probs, 1)) if self.sample else int(probs.argmax())
        logp = float(torch.log(probs[idx]).item())
        scalars = {k: v for k, v in obs.items() if k not in ("repr", "raw", "belief")}
        scalars["belief"] = float(obs["belief"])
        self.trace.append(("inner", obs["_prompt_ids"], obs["_completion_ids"], obs["block"],
                           scalars, idx, logp, [float(p) for p in probs.cpu()]))
        return INNER_ACTIONS[idx]

    def outer(self, obs):
        with torch.no_grad():
            x = self._vec(obs, "outer")
            logits = self.net["outer"](x.unsqueeze(0))[0]
            probs = torch.softmax(logits, -1)
        idx = int(torch.multinomial(probs, 1)) if self.sample else int(probs.argmax())
        logp = float(torch.log(probs[idx]).item())
        self.trace.append(("outer", None, None, None, x.cpu(), idx, logp, [float(p) for p in probs.cpu()]))
        return BOUNDARY_ACTIONS[idx]

    def recompute_inner_vec(self, state_fn, prompt_ids, completion_ids, block, scalars,
                            collect=None):
        live = state_fn(prompt_ids, completion_ids, block, scalars["_block_ends"])
        if collect is not None:
            collect.append(state_fn.last_logit)
        obs = {**scalars, "repr": live["repr"], "belief": live["belief"]}
        if self.raw2048:
            obs["raw"] = live["raw"]
        return self._vec(obs, "inner")

    def recompute_chunk(self, state_fn, entries, group_keys):
        by_draw: dict = {}
        for i, (t, g) in enumerate(zip(entries, group_keys)):
            by_draw.setdefault(g, []).append(i)
        vecs = [None] * len(entries)
        ready = [None] * len(entries)
        draws = []
        for g, idxs in by_draw.items():
            longest = max(idxs, key=lambda i: len(entries[i][2]))
            t_long = entries[longest]
            draws.append((t_long[1], t_long[2], [int(e) for e in t_long[4]["_block_ends"]]))
        fwds = state_fn.forward_many(draws)
        for (idxs, (prompt_ids, full_completion, full_ends), fwd) in zip(
                by_draw.values(), draws, fwds):
            for i in idxs:
                t = entries[i]
                L = len(t[2])
                if list(t[2]) != list(full_completion[:L]):
                    raise RuntimeError("a decision's prefix is not a prefix of its draw's completion")
                j = full_ends.index(L)
                obs = {**t[4], "repr": fwd["reprs"][j], "belief": torch.sigmoid(fwd["logits"][j])}
                if self.raw2048:
                    obs["raw"] = fwd["raw"][j]
                vecs[i] = self._vec(obs, "inner")
                ready[i] = fwd["logits"][j]
                if fwd["frozen_reprs"] is not None:
                    state_fn.anchor_pairs.append((fwd["reprs"][j], fwd["frozen_reprs"][j]))
        return torch.stack(vecs), ready


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["arm-F", "arm-B"], required=True)
    p.add_argument("--raw2048", action="store_true")
    p.add_argument("--questions", required=True, help="jsonl of {question, gold}")
    p.add_argument("--split-salt", required=True)
    p.add_argument("--head-checkpoint", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--model-path", default=None,
                   help="local snapshot dir of the pinned revision; bypasses hub-cache resolution")
    p.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--hs-chunk-size", type=int, default=8192)
    p.add_argument("--budget", type=int, default=24_000)
    p.add_argument("--replicas", type=int, default=8)
    p.add_argument("--batch-questions", type=int, default=16)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--max-draws", type=int, default=None,
                   help="safety valve only; None (default) is the spec's arm-B, where the BUDGET ends the episode")
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--lam-d", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=1.0,
                   help="weight of the EPISODE advantage on inner steps (A_step = A_draw/n_draw_steps + beta*A_ep/n_ep_steps). 0 trains arm-B on the draw-local objective only, which collapses to abandon (job 3122683 run B, spec 2.1)")
    p.add_argument("--alpha-draw", type=float, default=0.5,
                   help="weight of the draw-level shaping term, in TOTAL per episode relative to beta*A_ep (A_step = alpha*A_draw/(n_draws*n_draw_steps) + beta*A_ep/n_ep_steps)")
    p.add_argument("--anchor", type=float, default=1.0)
    p.add_argument("--target-survival", type=float, default=0.5,
                   help="fraction of horizon-length draws that reach their natural end under the INITIAL policy; the continue bias is derived from it")
    p.add_argument("--horizon-blocks", type=int, default=128,
                   help="max_blocks: the policy is not queried past it, and the survival target is spent over the query blocks up to it. The pool's median trace is 154 blocks; 30 was wrong by 5x")
    p.add_argument("--query-stride", type=int, default=4,
                   help="query the inner policy every k-th block (rollout_runner.is_query_block). k>1 is the exploration knob: k-fold fewer decisions carry the same survival target, so each has k-fold more stopping mass; also k-fold fewer head forwards")
    p.add_argument("--backbone-batch-tokens", type=int, default=16384,
                   help="padded tokens per batched frozen-backbone forward in the feature store")
    p.add_argument("--head-batch-tokens", type=int, default=32768,
                   help="padded tokens (rows x longest) per batched UHead forward. Bounds the padding waste of one forward and, in the update where the forward carries grad, its activation memory")
    p.add_argument("--block-cap", type=int, default=0,
                   help="max tokens per GENERATION request (0 = off, one request per block). A vLLM round's wall clock is its LONGEST output, and the measured shape is mean 42 tokens against a per-round max ~250 (gen/round_maxmean_ratio ~6), so the tail of one block idles every other sequence in the round. A capped block is finished by the next request from the same prefix: the model writes the same thing, the requests are cut differently. Watch gen/cap_hits and gen/straddled_boundaries")
    p.add_argument("--incremental-backbone", dest="incremental_backbone", action="store_true", default=True,
                   help="extend a draw's stored prefix by forwarding only the NEW tokens against a layers-0..15 KV cache (default on)")
    p.add_argument("--no-incremental-backbone", dest="incremental_backbone", action="store_false",
                   help="always re-forward the whole prefix (the pre-2026-08-19 path)")
    p.add_argument("--kv-max-tokens", type=int, default=400_000,
                   help="token budget for the incremental KV cache (64 KiB/token for Qwen3-1.7B layers 0..15, so 24.4 GiB); the least-recently-extended draws are evicted first and simply pay a full forward next time")
    p.add_argument("--incremental-batch-kv-tokens", type=int, default=65536,
                   help="padded (past+new) KV tokens per incremental backbone forward; bounds the transient left-padded batch cache (~4 GB at the default)")
    p.add_argument("--resume-from", default=None,
                   help="checkpoint to continue from: policy, head, optimizer, sampler state and iteration counter are restored; --iterations is the TOTAL")
    p.add_argument("--rollout-mode", choices=["batched", "async"], default="batched",
                   help="batched: one vLLM call per round over every live draw (the reference). "
                        "async: vLLM's AsyncLLM, one request per draw, submitted as soon as the "
                        "previous one returns -- no round barrier (rollout_async.run_async_episodes)")
    p.add_argument("--head-window-ms", type=float, default=5.0,
                   help="async mode only: how long the head worker waits to group decisions "
                        "before running the backbone on them as one batch")
    p.add_argument("--rollout-waves", type=int, default=1,
                   help="split the batch into this many waves run in threads so one wave's head work overlaps another's vLLM generation (rollout_batch.run_batched_episodes_in_waves); 1 = plain")
    p.add_argument("--hazard-power", type=float, default=1.0,
                   help="depth exponent of the initial stopping hazard h(b) = c (b/max_blocks)^power; 1 = linear in depth (agreed 2026-08-19), 0 = constant")
    p.add_argument("--min-completed-fraction", type=float, default=0.15,
                   help="smoke gate: at iteration 0, the share of draws that complete naturally must be at least this, or the run halts -- SC must be REACHABLE, not merely expressible")
    p.add_argument("--aux-ready", type=float, default=0.0)
    p.add_argument("--entropy-coef", type=float, default=0.0,
                   help="entropy bonus weight. Default 0 since 2026-08-19: with a survival-derived prior the policy sits at p(continue)~0.97, where the entropy gradient toward uniform is as large as the advantage signal (job 3124189: entropy 0.15 -> 0.34 and accuracy 0.51 -> 0.26 in three iterations, identically at lambda 0.25 and 1.0)")
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--stop-mass-floor", type=float, default=0.0,
                   help="minimum batch-mean p(stop) = p(answer)+p(abandon) over the queried inner states, "
                        "held by an adaptive multiplier (rl_train_core.stop_mass_penalty). 0 = off. "
                        "The exploration floor: the inner stops are unconditionally worse than continuing "
                        "(forced answers vote right 0.24 vs 0.95 for completed draws), so an unconstrained "
                        "policy drives p(stop) to ~0 in ~10 iterations (lb-armF-ends B: entropy 0.16 -> 0.06) "
                        "before it can learn WHERE stopping pays, and PPO-clip cannot bring a ~0 action back. "
                        "The floor fixes how much stopping mass exists, not where it goes")
    p.add_argument("--stop-mass-lr", type=float, default=0.1,
                   help="step of the stop-mass multiplier per iteration, in units of relative shortfall")
    p.add_argument("--lr-policy", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=5e-5)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--recompute-chunk", type=int, default=16,
                   help="decisions per recompute chunk in the update; peak memory scales with this, not with the batch")
    p.add_argument("--guard-every", type=int, default=5)
    p.add_argument("--guard-set", required=True, help="jsonl of held-out (input_ids, block mask, label) for the drift guard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--wandb-entity", required=True)
    p.add_argument("--wandb-project", required=True)
    p.add_argument("--wandb-run", required=True)
    p.add_argument("--wandb-mode", default="online")
    p.add_argument("--gpu-gen", type=int, default=0,
                   help="MUST be 0: vLLM's engine core takes the first visible device. Choose GPUs with CUDA_VISIBLE_DEVICES (job 3120574: two trainings on one node both put vLLM on GPU 0 and one OOMed)")
    p.add_argument("--gpu-head", type=int, default=1,
                   help="device for the frozen backbone, the head and the feature store. 1 = the two-GPU packing; 0 = SAME GPU as vLLM, which lets four legs share a node (each ~30%% busy on either GPU today). See rl_controller.gpu_pack")
    p.add_argument("--gpu-mem-util", type=float, default=None,
                   help="vLLM gpu_memory_utilization. Default 0.85 with a GPU each, 0.5 when --gpu-head 0 shares the generation GPU (the head, the feature store and the backbone KV cache grow AFTER vLLM profiles the device). Refused above 0.6 in same-GPU mode")
    return p.parse_args(argv)


RUN_CONFIG_KEYS = ("lam", "lam_d", "beta", "alpha_draw", "budget", "replicas",
                   "batch_questions", "max_draws", "block_cap", "entropy_coef",
                   "lr_policy", "ppo_epochs", "anchor", "stop_mass_floor", "stop_mass_lr")


def _run_config(args) -> dict:
    return {k: getattr(args, k, None) for k in RUN_CONFIG_KEYS}


def check_run_config(saved, want) -> str | None:
    if saved is None:
        return "checkpoint has no run_config (pre-2026-08-19); reward/environment equality NOT checked"
    diff = {k: (saved.get(k), want[k]) for k in want if saved.get(k) != want[k]}
    if diff:
        raise SystemExit(f"--resume-from run_config differs from the run's (ckpt, run): {diff}")
    return None


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        mem_util = resolve_gpu_mem_util(gpu_gen=args.gpu_gen, gpu_head=args.gpu_head,
                                        requested=args.gpu_mem_util)
    except PackError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps({"pack": describe_pack(gpu_gen=args.gpu_gen, gpu_head=args.gpu_head,
                                            mem_util=mem_util)}), flush=True)
    if args.rollout_mode == "async" and args.rollout_waves != 1:
        raise SystemExit("--rollout-waves is a batched-scheduler knob; async mode has no rounds to split")
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_head}")

    rows = [json.loads(l) for l in open(args.questions) if l.strip()]
    keyed = {question_key(r["question"]): r for r in rows}
    split = three_way_split(keyed.keys(), salt=args.split_salt)
    fit_keys = [k for k, s in split.items() if s == "fit"]
    if not fit_keys:
        raise SystemExit("no fit questions")

    tokenizer, provider, head, frozen, hidden_size = _load_head_runtime(args, device)
    anchor = PersistentAnchor(args.anchor)
    net = build_policy(raw2048=args.raw2048, arm=args.arm,
                       target_survival=args.target_survival,
                       horizon_blocks=args.horizon_blocks,
                       query_stride=args.query_stride,
                       hazard_power=args.hazard_power).to(device)
    params = [{"params": net.parameters(), "lr": args.lr_policy},
              {"params": head.parameters(), "lr": args.lr_head}]
    opt = torch.optim.AdamW(params)

    engine_kwargs = dict(model=(args.model_path or args.model),
                         revision=(None if args.model_path else args.revision),
                         tensor_parallel_size=1, gpu_memory_utilization=mem_util,
                         max_model_len=40960, seed=args.seed,
                         enable_prefix_caching=True, download_dir=args.hf_cache)
    if args.rollout_mode == "async":
        engine = AsyncVLLMEngine(lambda: build_async_llm(**engine_kwargs),
                                 sampling=sampling_config(), seed=args.seed)
    else:
        from vllm import LLM
        engine = VLLMEngine(LLM(**engine_kwargs), sampling=sampling_config(), seed=args.seed)

    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    decode = lambda ids: tokenizer.decode(ids)
    state_fn = HeadStateFn(provider=provider, head=head, frozen=frozen, device=device, raw2048=args.raw2048)
    state_fn.batch_tokens = args.backbone_batch_tokens
    state_fn.head_batch_tokens = args.head_batch_tokens
    state_fn.incremental = bool(args.incremental_backbone)
    state_fn.kv_max_tokens = args.kv_max_tokens
    state_fn.incr_batch_kv_tokens = args.incremental_batch_kv_tokens
    sampler = StratifiedSampler(fit_keys, seed=args.seed)
    sink = WandbSink(entity=args.wandb_entity, project=args.wandb_project, run_name=args.wandb_run,
                     config=vars(args), mode=args.wandb_mode)
    guard_rows = [json.loads(l) for l in open(args.guard_set) if l.strip()]

    def run_guard(iteration):
        pre, cur, lab = [], [], []
        head.eval()
        with torch.no_grad():
            for r in guard_rows:
                feats, am, keep, _ = provider.get_features([r["input_ids"]], device)
                if len(r["mask"]) != feats.shape[1]:
                    raise SystemExit(
                        f"guard mask has {len(r['mask'])} columns but the feature "
                        f"sequence has {feats.shape[1]}. The mask must be in FEATURE "
                        "space (the collator's mask[1:] shift), which is what "
                        "prepare_rl_inputs builds via raw_tap.end_feature_index. "
                        "Regenerate the guard set."
                    )
                mask = torch.tensor(r["mask"], dtype=torch.long, device=device).unsqueeze(0)
                lp = frozen.forward_from_features(features=feats, attention_mask=am, claims=[mask])
                lc = head.forward_from_features(features=feats, attention_mask=am, claims=[mask])
                pre.append(float(torch.sigmoid(lp.reshape(-1)[-1])))
                cur.append(float(torch.sigmoid(lc.reshape(-1)[-1])))
                lab.append(bool(r["label"]))
        head.eval()
        report = head_drift(pre, cur, lab)
        assert_head_healthy(report, DEFAULT_THRESHOLDS)
        return report

    delta_w = lambda: math.sqrt(sum(((p - q) ** 2).sum().item() for p, q in zip(head.parameters(), frozen.parameters())))

    start_it = 0
    train_state = {"stop_mass_lambda": 0.0}
    if args.resume_from:
        payload = load_checkpoint(args.resume_from, policy=net, head=head, optimizer=opt)
        if payload.get("train_state"):
            train_state.update(payload["train_state"])
        if payload["arm"] != args.arm or bool(payload["raw2048"]) != bool(args.raw2048):
            raise SystemExit(f"--resume-from is {payload['arm']} raw2048={payload['raw2048']}, run is {args.arm} raw2048={args.raw2048}")
        pc = payload.get("policy_config") or {}
        want = {"target_survival": args.target_survival, "horizon_blocks": args.horizon_blocks,
                "query_stride": args.query_stride, "hazard_power": args.hazard_power}
        if {k: pc.get(k) for k in want} != want:
            raise SystemExit(f"--resume-from policy_config {pc} differs from the run's {want}")
        note = check_run_config(payload.get("run_config"), _run_config(args))
        if note:
            print(json.dumps({"resume": note}))
        if payload.get("sampler_state"):
            sampler.load_state_dict(payload["sampler_state"])
        else:
            print(json.dumps({"resume": "checkpoint has no sampler state; sampler restarts fresh"}))
        start_it = int(payload["iteration"]) + 1
        print(json.dumps({"resume": {"from": args.resume_from, "start_iteration": start_it}}))
    for it in range(start_it, args.iterations):
        t0 = time.time()
        drift = run_guard(it) if should_run_guard(it, every=args.guard_every) else None

        batch_keys = sampler.sample(args.batch_questions)
        specs, pols = [], {}
        for qk in batch_keys:
            row = keyed[qk]
            prompt = rewrite_user_instruction(build_prompt_ids(tokenizer, row["question"]), encode=encode, decode=decode)
            for rep in range(args.replicas):
                ek = (qk, rep)
                pols[ek] = TorchPolicy(net, raw2048=args.raw2048, device=device, sample=True)
                specs.append({"key": ek, "prompt_ids": prompt, "gold": int(row["gold"]),
                              "question": hash(qk) & 0xFFFF, "budget": args.budget,
                              "max_draws": args.max_draws,
                              "seed_key": (qk, rep, it)})
        t_roll = time.time()
        vllm_seconds_before, vllm_batches_before = engine.seconds, engine.batches
        rollout_stats_before = ROLLOUT_STATS.snapshot()
        for _a in ("round_tokens_max_sum", "round_tokens_mean_sum", "round_requests"):
            if hasattr(engine, _a):
                setattr(engine, _a, 0)
        with torch.no_grad():
            rollout = (run_async_episodes if args.rollout_mode == "async"
                       else run_batched_episodes_in_waves)
            mode_kwargs = ({"head_window_s": args.head_window_ms / 1000.0}
                           if args.rollout_mode == "async" else {"waves": args.rollout_waves})
            done = rollout(
                engine=engine, episodes=specs,
                policy_for=lambda k: pols[k].inner,
                boundary_policy_for=((lambda k: pols[k].outer) if args.arm == "arm-F" else None),
                state_fn=state_fn, encode=encode, max_blocks=args.horizon_blocks,
                query_stride=args.query_stride, prefetch=state_fn.prefetch,
                block_cap=args.block_cap,
                thread_setup=torch.no_grad, **mode_kwargs,
            )
        t_roll = time.time() - t_roll
        state_fn.release_kv()
        episodes_by_q, traces_by_q = {}, {}
        for qk in batch_keys:
            episodes_by_q[qk] = [done[(qk, rep)] for rep in range(args.replicas)]
            traces_by_q[qk] = [pols[(qk, rep)].trace for rep in range(args.replicas)]
        sampler.update(episodes_by_q)

        if it == 0:
            all_draws0 = [d for eps in episodes_by_q.values() for e in eps for d in e.draws]
            completed_frac = (sum(1 for d in all_draws0 if d.outcome == "completed") / len(all_draws0)
                              if all_draws0 else 0.0)
            print(json.dumps({"reachability": {"completed_fraction": completed_frac,
                                               "min_required": args.min_completed_fraction,
                                               "target_survival": args.target_survival,
                                               "horizon_blocks": args.horizon_blocks,
                                               "query_stride": args.query_stride,
                                               "hazard_power": args.hazard_power}}))
            if completed_frac < args.min_completed_fraction:
                raise SystemExit(
                    f"reachability gate: only {completed_frac:.3f} of iteration-0 draws "
                    f"reached their natural end (need >= {args.min_completed_fraction}). "
                    "The initial policy does not present self-consistency, so GRPO "
                    "would only ever compare ways of stopping early. Raise "
                    "--target-survival or check --horizon-blocks against the pool's "
                    "actual block counts."
                )

        batch = assemble_batch(episodes_by_q, lam=args.lam, lam_d=args.lam_d, beta=args.beta, alpha=args.alpha_draw)

        inner_items, outer_items = [], []
        for qk, eps in episodes_by_q.items():
            if qk in batch.dropped_questions:
                continue
            for ei, tr in enumerate(traces_by_q[qk]):
                inner_dec = [t for t in tr if t[0] == "inner"]
                outer_dec = [t for t in tr if t[0] == "outer"]
                inner_adv = [a for (k, i, s), a in zip(batch.inner_steps, batch.inner_advantages) if k == qk and i == ei]
                outer_adv = [a for (k, i, s), a in zip(batch.outer_steps, batch.outer_advantages) if k == qk and i == ei]
                if len(inner_dec) != len(inner_adv) or len(outer_dec) != len(outer_adv):
                    raise RuntimeError(f"decision/advantage misalignment on {qk}: {len(inner_dec)} vs {len(inner_adv)}, {len(outer_dec)} vs {len(outer_adv)}")
                inner_items += [(t, a, (qk, ei, t[4]["draws_started"])) for t, a in zip(inner_dec, inner_adv)]
                outer_items += [(t, a) for t, a in zip(outer_dec, outer_adv)]

        losses = {}
        t_upd = time.time()
        if head.training:
            raise RuntimeError("the head is in train mode at the update: dropout would make the PPO ratio "
                               "and the anchor noisy (see _load_head_runtime)")
        store_before = dict(state_fn.store_stats)
        state_fn.take_anchor_pairs()
        if inner_items or outer_items:
            idx = torch.tensor([t[5] for t, _, _ in inner_items], device=device, dtype=torch.long)
            lp_old = torch.tensor([t[6] for t, _, _ in inner_items], device=device, dtype=torch.float32)
            adv = torch.tensor([a for _, a, _ in inner_items], dtype=torch.float32, device=device)
            pol_re = TorchPolicy(net, raw2048=args.raw2048, device=device)
            ready_targets = []
            for qk, eps_ in episodes_by_q.items():
                if qk in batch.dropped_questions:
                    continue
                for ep_ in eps_:
                    by_draw_ = {d.draw_index: d for d in ep_.draws}
                    for st_ in ep_.steps:
                        d_ = by_draw_.get(st_.draw_index)
                        ready_targets.append(1.0 if (d_ and d_.vote_correct) else 0.0)
            n_inner = len(inner_items)
            continue_index = INNER_ACTIONS.index("continue")
            for _ in range(args.ppo_epochs):
                opt.zero_grad(set_to_none=True)
                acc = {"loss": 0.0, "anchor": 0.0, "entropy": 0.0, "aux": 0.0, "stop_mass": 0.0}
                CH = args.recompute_chunk
                for c0 in range(0, n_inner, CH):
                    chunk = inner_items[c0:c0 + CH]
                    share = len(chunk) / n_inner
                    X, ready_logits = pol_re.recompute_chunk(state_fn, [t for t, _, _ in chunk],
                                                             [g for _, _, g in chunk])
                    anchor_pairs = state_fn.take_anchor_pairs()
                    idx_c = idx[c0:c0 + CH]; lp_old_c = lp_old[c0:c0 + CH]; adv_c = adv[c0:c0 + CH]
                    blocks_c = torch.tensor([int(t[3]) for t, _, _ in chunk], device=device, dtype=torch.long)
                    logits = net["inner"](X, blocks_c)
                    logp = torch.log_softmax(logits, -1).gather(1, idx_c[:, None])[:, 0]
                    ratio = torch.exp(logp - lp_old_c)
                    surr = torch.min(ratio * adv_c, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_c)
                    ent = -(torch.softmax(logits, -1) * torch.log_softmax(logits, -1)).sum(-1).mean()
                    loss = -surr.mean() - args.entropy_coef * ent
                    stop_mass_c = (1.0 - torch.softmax(logits, -1)[:, continue_index]).mean()
                    if args.stop_mass_floor > 0:
                        loss = loss + stop_mass_penalty(stop_mass_c, floor=args.stop_mass_floor,
                                                        multiplier=train_state["stop_mass_lambda"])
                    acc["stop_mass"] += float(stop_mass_c) * share
                    if anchor_pairs:
                        h_t = torch.stack([a for a, _ in anchor_pairs])
                        h_f = torch.stack([b for _, b in anchor_pairs])
                        anc = anchor.loss(h_t, h_f) / anchor.coef
                    else:
                        anc = torch.zeros((), device=device)
                    loss = loss + anchor.coef * anc
                    aux = torch.zeros((), device=device)
                    if args.aux_ready > 0 and ready_logits:
                        lg = torch.stack(ready_logits)
                        tg = torch.tensor(ready_targets[c0:c0 + CH], dtype=torch.float32, device=device)
                        aux = F.binary_cross_entropy_with_logits(lg, tg)
                        loss = loss + args.aux_ready * aux
                    (loss * share).backward()
                    acc["loss"] += float(loss) * share; acc["anchor"] += float(anc) * share
                    acc["entropy"] += float(ent) * share; acc["aux"] += float(aux) * share
                    del X, logits, loss
                if outer_items and args.arm == "arm-F":
                    Xo = torch.stack([t[4] for t, _ in outer_items]).to(device)
                    io = torch.tensor([t[5] for t, _ in outer_items], device=device)
                    lo = torch.tensor([t[6] for t, _ in outer_items], device=device)
                    ao = torch.tensor([a for _, a in outer_items], dtype=torch.float32, device=device)
                    lg = net["outer"](Xo)
                    lpo = torch.log_softmax(lg, -1).gather(1, io[:, None])[:, 0]
                    ro = torch.exp(lpo - lo)
                    so = torch.min(ro * ao, torch.clamp(ro, 1 - args.clip, 1 + args.clip) * ao)
                    ento = -(torch.softmax(lg, -1) * torch.log_softmax(lg, -1)).sum(-1).mean()
                    (-so.mean() - args.entropy_coef * ento).backward()
                torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(head.parameters()), 1.0)
                opt.step()
            loss, anc, ent, aux = (torch.tensor(acc[k]) for k in ("loss", "anchor", "entropy", "aux"))
            losses = {"loss/total": float(loss), "loss/anchor": float(anc),
                      "loss/entropy": float(ent), "head/aux_ready_loss": float(aux)}
            if n_inner:
                losses["actions/stop_mass_recomputed"] = acc["stop_mass"]
                if args.stop_mass_floor > 0:
                    train_state["stop_mass_lambda"] = stop_mass_multiplier_step(
                        train_state["stop_mass_lambda"], mass=acc["stop_mass"],
                        floor=args.stop_mass_floor, lr=args.stop_mass_lr)
                    losses["loss/stop_mass_lambda"] = train_state["stop_mass_lambda"]
        t_upd = time.time() - t_upd
        update_misses = state_fn.store_stats["misses"] - store_before["misses"]
        store = dict(state_fn.store_stats)
        state_fn.clear_store()

        all_eps = [e for eps in episodes_by_q.values() for e in eps]
        rewards = [episode_reward(e, lam=args.lam) for e in all_eps]
        m = {}
        m.update(episode_metrics(all_eps, rewards, prefix="train"))
        m.update(group_metrics({k: [episode_reward(e, lam=args.lam) for e in v] for k, v in episodes_by_q.items()}))
        m.update(advantage_metrics({k: [episode_reward(e, lam=args.lam) for e in v] for k, v in episodes_by_q.items()}))
        steps = [s for e in all_eps for s in e.steps]
        probs = [t[7] for tr in traces_by_q.values() for tt in tr for t in tt if t[0] == "inner"]
        m.update(action_metrics(steps, probs if len(probs) == len(steps) else None))
        m.update(boundary_action_metrics([b for e in all_eps for b in e.boundary_steps]))
        m.update(draw_outcome_metrics(all_eps))
        if drift is not None:
            m.update(head_metrics(anchor_coef=anchor.coefficient(it), delta_w_norm=delta_w(), drift_report=drift))
        else:
            m["anchor/coef"] = anchor.coefficient(it); m["head/delta_w_norm"] = delta_w()
        m.update(losses)
        m["sampler/mixed"] = sampler.bucket_counts()["mixed"]
        m["sampler/hopeless"] = sampler.bucket_counts()["hopeless"]
        m["sampler/saturated"] = sampler.bucket_counts()["saturated"]
        m["time/iteration_s"] = time.time() - t0
        m["time/rollout_s"] = t_roll
        m["time/update_s"] = t_upd
        m["head/backbone_calls"] = store["provider_prefixes"]
        m["time/backbone_s"] = store["backbone_seconds"]
        m["time/head_forward_s"] = store["head_seconds"]
        m["time/vllm_s"] = engine.seconds - vllm_seconds_before
        m["gen/batches"] = engine.batches - vllm_batches_before
        for _k in ("cap_hits", "straddled_boundaries", "straddled_queries_skipped",
                   "boundary_gaps"):
            m[f"gen/{_k}"] = getattr(ROLLOUT_STATS, _k) - rollout_stats_before[_k]
        m["gen/max_batch"] = engine.max_batch
        if getattr(engine, "round_requests", 0):
            nb = max(1, engine.batches - vllm_batches_before)
            m["gen/round_tokens_max_mean"] = engine.round_tokens_max_sum / nb
            m["gen/round_tokens_mean"] = engine.round_tokens_mean_sum / nb
            m["gen/round_maxmean_ratio"] = (engine.round_tokens_max_sum
                                            / max(1e-9, engine.round_tokens_mean_sum))
        if args.rollout_mode == "async":
            _a = ASYNC_STATS.snapshot()
            m["gen/requests"] = _a["requests"]
            m["gen/inflight_mean"] = _a["inflight_mean"]
            m["gen/inflight_max"] = _a["inflight_max"]
            m["time/head_wait_s"] = _a["head_busy_seconds"]
            m["head/decision_batch_mean"] = _a["head_batch_mean"]
        m["head/store_hits"] = store["hits"]
        m["head/store_misses"] = store["misses"]
        m["head/store_misses_in_update"] = update_misses
        m["head/store_tokens_peak"] = store["tokens_stored"]
        m["head/forward_batches"] = store["head_forward_batches"]
        m["head/forward_rows_mean"] = (store["head_forward_rows"] / store["head_forward_batches"]
                                       if store["head_forward_batches"] else 0.0)
        m["time/head_features_s"] = store["feature_seconds"]
        m["head/batched_head_vs_single_maxdiff"] = (state_fn._head_batch_check
                                                    if state_fn._head_batch_check is not None else -1.0)
        m["head/batched_head_vs_single_reldiff"] = (
            state_fn._head_batch_check / max(state_fn._head_batch_scale, 1e-9)
            if state_fn._head_batch_check is not None else -1.0)
        m["head/batched_vs_single_maxdiff"] = (state_fn._batched_check if state_fn._batched_check is not None else -1.0)
        m["head/incremental_vs_full_maxdiff"] = (state_fn._incr_check if state_fn._incr_check is not None else -1.0)
        m["head/incremental_vs_full_reldiff"] = (state_fn._incr_check / max(state_fn._incr_scale, 1e-9)
                                                 if state_fn._incr_check is not None else -1.0)
        m["head/incremental_prefixes"] = store["incremental_prefixes"]
        m["head/backbone_tokens"] = store["backbone_tokens"]
        m["head/kv_tokens_peak"] = store["kv_tokens_peak"]
        for k_ in ("provider_calls", "provider_prefixes", "hits", "misses",
                   "backbone_seconds", "head_seconds", "backbone_tokens",
                   "incremental_prefixes", "kv_tokens_peak",
                   "head_forward_batches", "head_forward_rows", "feature_seconds"):
            state_fn.store_stats[k_] = 0
        sink.log(m, step=it)
        print(json.dumps({"it": it, **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items() if k.startswith(("train/", "actions/", "groups/", "anchor/", "drift/rank", "loss/", "time/", "head/", "gen/", "draws/", "sampler/", "boundary/"))}}))

        save_checkpoint(out_dir / f"ckpt_{it:04d}.pt", policy=net, head=head, optimizer=opt, anchor=anchor,
                        policy_config={"target_survival": args.target_survival,
                                       "horizon_blocks": args.horizon_blocks,
                                       "query_stride": args.query_stride,
                                       "hazard_power": args.hazard_power},
                        sampler_state=sampler.state_dict(),
                        run_config=_run_config(args), train_state=dict(train_state),
                        raw2048=args.raw2048, arm=args.arm, iteration=it)

    sink.finish()
    if hasattr(engine, "close"):
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
