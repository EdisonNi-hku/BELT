from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rl_controller.baselines.rom import data as rom_data
from rl_controller.baselines.rom.streaming_head import UPSTREAM_RECIPE, StreamingHead, set_seed

def save_resume(path, *, epoch, head, optimizer, scheduler, history, best_state,
                best_loss, best_epoch, balance, config) -> None:
    import os

    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save({"epoch": epoch, "model_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history, "best_state_dict": best_state,
                "best_val_loss": best_loss, "best_epoch": best_epoch,
                "balance": balance, "config": config.to_dict()}, tmp)
    os.replace(tmp, target)


def load_resume(path):
    import torch

    target = Path(path)
    if not target.exists():
        return None
    try:
        return torch.load(target, map_location="cpu", weights_only=False)
    except Exception as exc:
        print({"warning": "resume_checkpoint_unreadable", "path": str(target),
               "error": str(exc)}, flush=True)
        return None


def flushing_log(row):
    print(row, flush=True)


ARMS = ("R0", "R1", "R2")
EPOCH_SELECTION = ("best_val", "last")


@dataclass
class TrainConfig:

    lr: float = UPSTREAM_RECIPE["lr"]
    weight_decay: float = UPSTREAM_RECIPE["weight_decay"]
    epochs: int = UPSTREAM_RECIPE["num_train_epochs"]
    warmup_ratio: float = UPSTREAM_RECIPE["warmup_ratio"]
    batch_size: int = UPSTREAM_RECIPE["batch_size"]
    grad_accum: int = UPSTREAM_RECIPE["gradient_acc_steps"]
    betas: tuple = UPSTREAM_RECIPE["betas"]
    eps: float = UPSTREAM_RECIPE["eps"]
    max_grad_norm: float = UPSTREAM_RECIPE["max_grad_norm"]
    proj_dim: int = 512
    mem_dim: int = 512
    num_labels: int = 1
    cfc: bool = True
    arm: str = "R0"
    dt_mode: str = "token_gap"
    m0_grad: bool = False
    epoch_selection: str = "best_val"
    seed: int = 0
    device: str = "cuda"
    dtype: str = "float32"

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {self.arm!r}")
        if self.epoch_selection not in EPOCH_SELECTION:
            raise ValueError(f"epoch_selection must be one of {EPOCH_SELECTION}")
        if self.grad_accum < 1 or self.batch_size < 1:
            raise ValueError("batch_size and grad_accum must be >= 1")

    def to_dict(self) -> dict:
        return {**asdict(self), "betas": list(self.betas)}


@dataclass
class TrainResult:
    head: object
    config: TrainConfig
    history: list = field(default_factory=list)
    balance: dict = field(default_factory=dict)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    seconds: float = 0.0

    def summary(self) -> dict:
        return {"config": self.config.to_dict(), "history": self.history,
                "balance": self.balance, "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val_loss, "seconds": self.seconds,
                "params": self.head.parameter_count(), "head": self.head.config()}


def _torch():
    import torch

    return torch


def _batches(examples, batch_size, *, shuffle, seed, epoch):
    import random

    order = sorted(range(len(examples)), key=lambda i: (examples[i].n_blocks, i))
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        random.Random((seed << 16) ^ epoch).shuffle(chunks)
    for chunk in chunks:
        yield [examples[i] for i in chunk]


def masked_bce_sum(logits, batch, *, pos_weight=None, num_labels: int = 1):
    torch = _torch()
    gathered = rom_data.gather_probe_logits(logits, batch["gather"])
    mask, label = batch["mask"], batch["label"]
    denom = float(mask.sum())
    if denom == 0.0:
        return None, 0.0
    if num_labels == 1:
        per = torch.nn.functional.binary_cross_entropy_with_logits(
            gathered.squeeze(-1), label, reduction="none",
            pos_weight=(None if pos_weight is None
                        else torch.as_tensor(pos_weight, device=logits.device,
                                             dtype=logits.dtype)))
    else:
        flat = gathered.reshape(-1, gathered.size(-1))
        target = label.reshape(-1).long()
        weight = None
        if pos_weight is not None:
            weight = torch.tensor([1.0, float(pos_weight)], device=logits.device,
                                  dtype=logits.dtype)
        per = torch.nn.functional.cross_entropy(
            flat, target, weight=weight, reduction="none").reshape(label.shape)
    return (per * mask).sum(), denom


def _lr_lambda(step: int, *, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


def evaluate(head, examples, config: TrainConfig, *, pos_weight=None) -> dict:
    torch = _torch()
    head.eval()
    total_loss = total_n = correct = 0.0
    with torch.no_grad():
        for chunk in _batches(examples, config.batch_size, shuffle=False, seed=0, epoch=0):
            batch = rom_data.collate(chunk, device=config.device,
                                     dtype=getattr(torch, config.dtype))
            logits = head(batch["x"], batch["lengths"], batch["dt"],
                          prompt=batch["prompt"], prompt_mask=batch["prompt_mask"],
                          m0_grad=False)
            loss_sum, n = masked_bce_sum(logits, batch, pos_weight=pos_weight,
                                         num_labels=config.num_labels)
            if loss_sum is None:
                continue
            total_loss += float(loss_sum)
            total_n += n
            probs = head.probabilities(rom_data.gather_probe_logits(logits, batch["gather"]))
            hit = ((probs > 0.5).to(batch["label"].dtype) == batch["label"]).to(probs.dtype)
            correct += float((hit * batch["mask"]).sum())
    return {"loss": (total_loss / total_n) if total_n else float("nan"),
            "boundary_acc": (correct / total_n) if total_n else float("nan"),
            "n_labelled": total_n}


def train_detector(train_examples, val_examples, config: TrainConfig, *,
                   input_dim: int, log=flushing_log, resume_path=None,
                   resume: bool = True, val_extra=None) -> TrainResult:
    torch = _torch()
    if not train_examples or not val_examples:
        raise ValueError(f"need both splits (train {len(train_examples)}, "
                         f"val {len(val_examples)})")
    set_seed(config.seed)
    started = time.time()

    balance = {"arm": config.arm, "applied": None, "pos_weight": None,
               "oversample": None, "counts_train": rom_data.class_counts(train_examples)}
    examples = list(train_examples)
    weight = None
    if config.arm == "R1":
        weight = rom_data.pos_weight(train_examples)
        if weight is None:
            raise ValueError("arm R1 needs both classes present on the train split; "
                             f"got {balance['counts_train']}. Refusing to fall back to R0 "
                             "silently -- report the degenerate split instead.")
        balance.update(applied="R1", pos_weight=weight)
    elif config.arm == "R2":
        examples, plan = rom_data.oversample(train_examples, seed=config.seed)
        balance["oversample"] = plan
        balance["applied"] = "R2" if plan["applicable"] else "R0(R2 inapplicable)"
        if not plan["applicable"]:
            log({"warning": "rom_R2_inapplicable", **plan})
    else:
        balance["applied"] = "R0"

    head = StreamingHead(input_dim=input_dim, proj_dim=config.proj_dim,
                         mem_dim=config.mem_dim, num_labels=config.num_labels,
                         cfc=config.cfc)
    head.to(device=config.device, dtype=getattr(torch, config.dtype))

    optimizer = torch.optim.AdamW(head.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay,
                                 betas=tuple(config.betas), eps=config.eps)
    per_epoch = math.ceil(math.ceil(len(examples) / config.batch_size) / config.grad_accum)
    total_steps = max(1, config.epochs * per_epoch)
    warmup = int(total_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_lambda(s, warmup=warmup, total=total_steps))

    result = TrainResult(head=head, config=config, balance=balance)
    best_state = None
    def _step(window_weight: float) -> None:
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

    start_epoch = 0
    state = load_resume(resume_path) if (resume and resume_path) else None
    if state is not None:
        head.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        result.history = list(state.get("history") or [])
        best_state = state.get("best_state_dict")
        result.best_val_loss = state.get("best_val_loss", float("inf"))
        result.best_epoch = state.get("best_epoch", -1)
        start_epoch = int(state["epoch"]) + 1
        log({"resumed_from_epoch": start_epoch, "best_epoch": result.best_epoch,
             "best_val_loss": result.best_val_loss})
    for epoch in range(start_epoch, config.epochs):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        seen = run_loss = run_n = 0
        window_weight = 0.0
        for step, chunk in enumerate(_batches(examples, config.batch_size, shuffle=True,
                                              seed=config.seed, epoch=epoch)):
            batch = rom_data.collate(chunk, device=config.device,
                                     dtype=getattr(torch, config.dtype))
            logits = head(batch["x"], batch["lengths"], batch["dt"],
                          prompt=batch["prompt"], prompt_mask=batch["prompt_mask"],
                          m0_grad=config.m0_grad)
            loss_sum, n = masked_bce_sum(logits, batch, pos_weight=weight,
                                         num_labels=config.num_labels)
            if loss_sum is None:
                continue
            loss_sum.backward()
            run_loss += float(loss_sum)
            run_n += n
            window_weight += n
            seen += 1
            if seen % config.grad_accum == 0:
                _step(window_weight)
                window_weight = 0.0
        if seen % config.grad_accum:
            _step(window_weight)

        val = evaluate(head, val_examples, config, pos_weight=weight)
        row = {"epoch": epoch, "train_loss": (run_loss / run_n) if run_n else float("nan"),
               "val_loss": val["loss"], "val_boundary_acc": val["boundary_acc"],
               "lr": optimizer.param_groups[0]["lr"]}
        for name, extra_examples in (val_extra or {}).items():
            extra = evaluate(head, extra_examples, config, pos_weight=weight)
            row[f"val_loss_{name}"] = extra["loss"]
            row[f"val_boundary_acc_{name}"] = extra["boundary_acc"]
        result.history.append(row)
        log(row)
        if val["loss"] < result.best_val_loss:
            result.best_val_loss = val["loss"]
            result.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if resume_path:
            save_resume(resume_path, epoch=epoch, head=head, optimizer=optimizer,
                        scheduler=scheduler, history=result.history,
                        best_state=best_state, best_loss=result.best_val_loss,
                        best_epoch=result.best_epoch, balance=balance, config=config)

    if config.epoch_selection == "best_val" and best_state is not None:
        head.load_state_dict(best_state)
    else:
        result.best_epoch = config.epochs - 1
    result.seconds = time.time() - started
    return result
