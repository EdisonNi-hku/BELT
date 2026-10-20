from __future__ import annotations

import json
from dataclasses import dataclass, field

TARGETS = ("q", "v", "vpi0", "r")
SPLITS = ("train", "val", "fold_heldout")


@dataclass
class C2Example:
    qkey: str
    rep: int
    x: object
    dt: list
    labels: list
    mask: list
    probe_pos: list
    prompt: object = None
    split: str = "train"
    meta: dict = field(default_factory=dict)

    @property
    def n_blocks(self) -> int:
        return len(self.labels)


def load_dump(path, *, targets=TARGETS, splits=("train", "val")):
    table: dict = {}
    keep = set(splits) if splits else None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            split = row.get("split", "train")
            if keep is not None and split not in keep:
                continue
            ident = (row["qkey"], int(row["rep"]))
            entry = table.get(ident)
            if entry is None:
                entry = table[ident] = {"ends": [], "probe_ends": [], "split": split}
            labels = row["labels"]
            values, mask = [], []
            for name in targets:
                value = labels.get(name)
                values.append(0.0 if value is None else float(value))
                mask.append(0.0 if value is None else 1.0)
            entry["ends"].append((int(row["end_index"]), values, mask))
            if row.get("probe_index") is not None:
                entry["probe_ends"].append(int(row["end_index"]))
    for ident, entry in table.items():
        entry["ends"].sort(key=lambda r: r[0])
        seen = [r[0] for r in entry["ends"]]
        if seen != list(range(len(seen))):
            raise ValueError(f"chain {ident}: dump end_index values are {seen[:6]}..., "
                             "expected a contiguous 0..n-1 run")
        entry["probe_ends"].sort()
    return table


def build_examples(chains, dump, cache, spec, *, dt_mode="token_gap", with_prompt=True):
    from rl_controller.baselines.rom.data import _check_alignment, block_dt
    from rl_controller.baselines.rom.features import FeatureCacheError
    from rl_controller.single_cot import probe_positions_in_ends

    out, skipped = [], {"not_in_dump": 0, "uncached": 0, "length_mismatch": [], "no_probes": 0}
    for chain in chains:
        ident = (chain["qkey"], int(chain["rep"]))
        entry = dump.get(ident)
        if entry is None:
            skipped["not_in_dump"] += 1
            continue
        if not chain["probes"]:
            skipped["no_probes"] += 1
            continue
        _check_alignment(chain)
        ends = [int(e) for e in chain["block_ends"]]
        if len(entry["ends"]) != len(ends):
            skipped["length_mismatch"].append(
                {"qkey": ident[0], "rep": ident[1], "dump": len(entry["ends"]),
                 "chain": len(ends)})
            continue
        try:
            x = cache.chain_matrix(chain, spec)
            prompt = cache.prompt_matrix(chain, spec) if with_prompt else None
        except FeatureCacheError:
            skipped["uncached"] += 1
            continue
        out.append(C2Example(
            qkey=ident[0], rep=ident[1], x=x,
            dt=block_dt(ends, int(chain["thinking_tokens"]), dt_mode),
            labels=[r[1] for r in entry["ends"]], mask=[r[2] for r in entry["ends"]],
            probe_pos=probe_positions_in_ends(ends, chain["query_points"]),
            prompt=prompt, split=entry["split"],
            meta={"n_probes": len(chain["probes"])}))
    return out, skipped


def collate(examples, *, device=None, dtype=None, n_targets=len(TARGETS)):
    import torch

    if not examples:
        raise ValueError("cannot collate an empty batch")
    dtype = dtype or torch.float32
    batch = len(examples)
    steps = max(e.n_blocks for e in examples)
    width = len(examples[0].x[0])

    x = torch.zeros(batch, steps, width, dtype=dtype)
    dt = torch.zeros(batch, steps, dtype=dtype)
    lengths = torch.zeros(batch, dtype=torch.long)
    label = torch.zeros(batch, steps, n_targets, dtype=dtype)
    mask = torch.zeros(batch, steps, n_targets, dtype=dtype)
    for row, example in enumerate(examples):
        n = example.n_blocks
        x[row, :n] = torch.as_tensor(example.x, dtype=dtype).reshape(n, width)
        dt[row, :n] = torch.as_tensor(example.dt, dtype=dtype)
        lengths[row] = n
        label[row, :n] = torch.as_tensor(example.labels, dtype=dtype)
        mask[row, :n] = torch.as_tensor(example.mask, dtype=dtype)

    prompt = prompt_mask = None
    prompts = [e.prompt for e in examples]
    if any(p is not None for p in prompts):
        if any(p is None for p in prompts):
            raise ValueError("prompt features present for some chains and missing for others; "
                             "m_0 must be initialised the same way for every chain in a run")
        n_prompt = max(len(p) for p in prompts)
        prompt = torch.zeros(batch, n_prompt, width, dtype=dtype)
        prompt_mask = torch.zeros(batch, n_prompt, dtype=torch.bool)
        for row, rows in enumerate(prompts):
            n = len(rows)
            prompt[row, :n] = torch.as_tensor(rows, dtype=dtype).reshape(n, width)
            prompt_mask[row, :n] = True

    out = {"x": x, "dt": dt, "lengths": lengths, "label": label, "mask": mask,
           "prompt": prompt, "prompt_mask": prompt_mask}
    if device is not None:
        out = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in out.items()}
    return out


def multi_bce_sum(logits, batch, *, weights=None):
    import torch

    label, mask = batch["label"], batch["mask"]
    if weights is not None:
        mask = mask * torch.as_tensor(weights, device=mask.device, dtype=mask.dtype)
    denom = float(mask.sum())
    if denom == 0.0:
        return None, 0.0
    per = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, label, reduction="none")
    return (per * mask).sum(), denom


def build_score_examples(chains, cache, spec, *, dt_mode="token_gap", with_prompt=True):
    from rl_controller.baselines.rom.data import _check_alignment, block_dt
    from rl_controller.baselines.rom.features import FeatureCacheError
    from rl_controller.single_cot import probe_positions_in_ends

    out, skipped = [], {"no_probes": 0, "uncached": []}
    for chain in chains:
        if not chain["probes"]:
            skipped["no_probes"] += 1
            continue
        _check_alignment(chain)
        ends = [int(e) for e in chain["block_ends"]]
        try:
            x = cache.chain_matrix(chain, spec)
            prompt = cache.prompt_matrix(chain, spec) if with_prompt else None
        except FeatureCacheError as exc:
            skipped["uncached"].append({"qkey": chain["qkey"], "rep": chain["rep"],
                                        "reason": str(exc)})
            continue
        n = len(ends)
        out.append(C2Example(
            qkey=chain["qkey"], rep=int(chain["rep"]), x=x,
            dt=block_dt(ends, int(chain["thinking_tokens"]), dt_mode),
            labels=[[0.0] * len(TARGETS)] * n, mask=[[0.0] * len(TARGETS)] * n,
            probe_pos=probe_positions_in_ends(ends, chain["query_points"]),
            prompt=prompt))
    return out, skipped


def evaluate_c2(head, examples, *, device, dtype, batch_size=32):
    import torch

    head.eval()
    total = count = 0.0
    per_target = [0.0] * len(TARGETS)
    per_target_n = [0.0] * len(TARGETS)
    with torch.no_grad():
        order = sorted(range(len(examples)), key=lambda i: examples[i].n_blocks)
        for start in range(0, len(order), batch_size):
            chunk = [examples[i] for i in order[start:start + batch_size]]
            batch = collate(chunk, device=device, dtype=dtype)
            logits = head(batch["x"], batch["lengths"], batch["dt"],
                          prompt=batch["prompt"], prompt_mask=batch["prompt_mask"])
            loss_sum, n = multi_bce_sum(logits, batch)
            if loss_sum is None:
                continue
            total += float(loss_sum)
            count += n
            per = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch["label"], reduction="none") * batch["mask"]
            sums = per.sum(dim=(0, 1)).tolist()
            ns = batch["mask"].sum(dim=(0, 1)).tolist()
            for i in range(len(TARGETS)):
                per_target[i] += sums[i]
                per_target_n[i] += ns[i]
    return {"loss": (total / count) if count else float("nan"), "n_labelled": count,
            "per_target": {name: (per_target[i] / per_target_n[i]) if per_target_n[i] else None
                           for i, name in enumerate(TARGETS)}}


def train_c2(train_examples, val_examples, *, input_dim, config, log=None,
             resume_path=None, resume: bool = True):
    import math

    import torch

    from rl_controller.baselines.rom.streaming_head import StreamingHead, set_seed
    from rl_controller.baselines.rom.train import (_batches, _lr_lambda, flushing_log,
                                                    load_resume, save_resume)

    log = log or flushing_log

    set_seed(config.seed)
    dtype = getattr(torch, config.dtype)
    head = StreamingHead(input_dim=input_dim, proj_dim=config.proj_dim,
                         mem_dim=config.mem_dim, num_labels=len(TARGETS),
                         head_mode="bce", cfc=config.cfc)
    head.to(device=config.device, dtype=dtype)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay,
                                  betas=tuple(config.betas), eps=config.eps)
    per_epoch = math.ceil(math.ceil(len(train_examples) / config.batch_size) / config.grad_accum)
    total_steps = max(1, config.epochs * per_epoch)
    warmup = int(total_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_lambda(s, warmup=warmup, total=total_steps))

    def step(window_weight):
        if window_weight <= 0:
            optimizer.zero_grad(set_to_none=True)
            return
        for parameter in head.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(window_weight)
        torch.nn.utils.clip_grad_norm_(head.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    history, best_state, best_loss, best_epoch = [], None, float("inf"), -1
    start_epoch = 0
    state = load_resume(resume_path) if (resume and resume_path) else None
    if state is not None:
        head.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        history = list(state.get("history") or [])
        best_state = state.get("best_state_dict")
        best_loss = state.get("best_val_loss", float("inf"))
        best_epoch = state.get("best_epoch", -1)
        start_epoch = int(state["epoch"]) + 1
        log({"resumed_from_epoch": start_epoch, "best_epoch": best_epoch,
             "best_val_loss": best_loss})
    for epoch in range(start_epoch, config.epochs):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        seen = run_loss = run_n = 0
        window = 0.0
        for chunk in _batches(train_examples, config.batch_size, shuffle=True,
                              seed=config.seed, epoch=epoch):
            batch = collate(chunk, device=config.device, dtype=dtype)
            logits = head(batch["x"], batch["lengths"], batch["dt"],
                          prompt=batch["prompt"], prompt_mask=batch["prompt_mask"],
                          m0_grad=config.m0_grad)
            loss_sum, n = multi_bce_sum(logits, batch)
            if loss_sum is None:
                continue
            loss_sum.backward()
            run_loss += float(loss_sum)
            run_n += n
            window += n
            seen += 1
            if seen % config.grad_accum == 0:
                step(window)
                window = 0.0
        if seen % config.grad_accum:
            step(window)
        val = evaluate_c2(head, val_examples, device=config.device, dtype=dtype,
                          batch_size=config.batch_size * 4)
        row = {"epoch": epoch, "train_loss": (run_loss / run_n) if run_n else float("nan"),
               "val_loss": val["loss"], "val_per_target": val["per_target"],
               "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        log(row)
        if val["loss"] < best_loss:
            best_loss, best_epoch = val["loss"], epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if resume_path:
            save_resume(resume_path, epoch=epoch, head=head, optimizer=optimizer,
                        scheduler=scheduler, history=history, best_state=best_state,
                        best_loss=best_loss, best_epoch=best_epoch,
                        balance={"arm": "n/a (C2)"}, config=config)
    if best_state is not None:
        head.load_state_dict(best_state)
    return head, {"history": history, "best_epoch": best_epoch, "best_val_loss": best_loss,
                  "head": head.config(), "targets": list(TARGETS)}


def score_c2(head, chains, cache, spec, *, config, batch_size=32):
    import torch

    dtype = getattr(torch, config.dtype)
    examples, skipped = build_score_examples(chains, cache, spec, dt_mode=config.dt_mode)
    by_ident = {(e.qkey, e.rep): e for e in examples}
    head.eval()
    with torch.no_grad():
        order = sorted(range(len(examples)), key=lambda i: examples[i].n_blocks)
        for start in range(0, len(order), batch_size):
            chunk = [examples[i] for i in order[start:start + batch_size]]
            batch = collate(chunk, device=config.device, dtype=dtype)
            probs = torch.sigmoid(head(batch["x"], batch["lengths"], batch["dt"],
                                       prompt=batch["prompt"],
                                       prompt_mask=batch["prompt_mask"])).cpu()
            for row, example in enumerate(chunk):
                example.meta["scores"] = [
                    {name: float(probs[row, position, i]) for i, name in enumerate(TARGETS)}
                    for position in example.probe_pos]
    unscored, probeless = [], 0
    for chain in chains:
        if not chain["probes"]:
            chain["scores"] = []
            probeless += 1
            continue
        example = by_ident.get((chain["qkey"], int(chain["rep"])))
        if example is None or "scores" not in example.meta:
            unscored.append({"qkey": chain["qkey"], "rep": chain["rep"]})
            continue
        chain["scores"] = example.meta["scores"]
    skipped["unscored"] = unscored
    skipped["probeless_natural_rows"] = probeless
    return skipped
