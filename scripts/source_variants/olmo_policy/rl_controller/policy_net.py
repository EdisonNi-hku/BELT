from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from rl_controller.budget_episode import BOUNDARY_ACTIONS, INNER_ACTIONS

FORBIDDEN_OBSERVATION_KEYS = frozenset({
    "answer", "correct", "gold", "gold_answer", "natural_correct",
    "natural_answer", "n_blocks", "n_blocks_total", "completion_tokens",
})

POSITION_FEATURES = ("block", "tokens_so_far", "mean_block_tokens")
BUDGET_FEATURES = ("spent", "budget", "budget_left", "budget_fraction_left")
VOTE_FEATURES = ("n_banked", "top1_share", "n_distinct_answers", "vote_margin")

HEAD_OUTPUT_FEATURES = ("belief",)

INNER_SCALAR_FEATURES = (POSITION_FEATURES + BUDGET_FEATURES + VOTE_FEATURES
                         + HEAD_OUTPUT_FEATURES)
OUTER_SCALAR_FEATURES = BUDGET_FEATURES + VOTE_FEATURES + ("draws_started",)

HEAD_DIM = 768
RAW_DIM = 2048


class PolicyContractError(RuntimeError):
    pass


def _scale(name: str, value: float) -> float:
    if name in ("block",):
        return math.log1p(float(value)) / math.log1p(128.0)
    if name in ("tokens_so_far", "spent", "budget", "budget_left"):
        return math.log1p(float(value)) / math.log1p(40960.0)
    if name == "mean_block_tokens":
        return math.log1p(float(value)) / math.log1p(2048.0)
    if name in ("n_banked", "n_distinct_answers", "draws_started"):
        return float(value) / 8.0
    return float(value)


def _refuse_leaks(observation: Mapping) -> None:
    leaked = FORBIDDEN_OBSERVATION_KEYS & {k for k in observation if not str(k).startswith("_")}
    if leaked:
        raise PolicyContractError(
            f"observation carries {sorted(leaked)}: an oracle quantity reached the "
            "policy input. Refused at vectorisation so it cannot be trained on"
        )


def inner_feature_dim(*, raw2048: bool) -> int:
    return HEAD_DIM + (RAW_DIM if raw2048 else 0) + len(INNER_SCALAR_FEATURES)


def outer_feature_dim() -> int:
    return len(OUTER_SCALAR_FEATURES)


def observation_to_vector(observation: Mapping, *, raw2048: bool) -> list[float]:
    _refuse_leaks(observation)
    head = list(observation["repr"])
    if len(head) != HEAD_DIM:
        raise PolicyContractError(f"repr has {len(head)} dims, expected {HEAD_DIM}")
    out = [float(v) for v in head]
    if raw2048:
        if "raw" not in observation:
            raise PolicyContractError("raw2048 is enabled but the observation has no 'raw'")
        raw = list(observation["raw"])
        if len(raw) != RAW_DIM:
            raise PolicyContractError(f"raw has {len(raw)} dims, expected {RAW_DIM}")
        out.extend(float(v) for v in raw)
    elif "raw" in observation:
        raise PolicyContractError(
            "raw2048 is disabled but the observation carries 'raw'; the two arms "
            "must differ in the switch, not in what the state_fn happens to emit"
        )
    for name in INNER_SCALAR_FEATURES:
        if name not in observation:
            raise PolicyContractError(f"observation is missing {name!r}")
        out.append(_scale(name, observation[name]))
    return out


def boundary_observation_to_vector(observation: Mapping) -> list[float]:
    _refuse_leaks(observation)
    out = []
    for name in OUTER_SCALAR_FEATURES:
        if name not in observation:
            raise PolicyContractError(f"boundary observation is missing {name!r}")
        out.append(_scale(name, observation[name]))
    return out


class PersistentAnchor:

    def __init__(self, coef: float):
        if not (coef > 0.0 and math.isfinite(coef)):
            raise PolicyContractError(f"anchor coef must be positive and finite, got {coef!r}")
        self.coef = float(coef)

    def coefficient(self, step: int) -> float:
        return self.coef

    def loss(self, h_trainable, h_frozen):
        import torch

        if h_frozen.requires_grad:
            raise PolicyContractError(
                "the frozen reference must be detached; gradient must not flow "
                "into the pretrained head"
            )
        return self.coef * ((h_trainable - h_frozen) ** 2).mean()


REQUIRED_CHECKPOINT_KEYS = ("policy_state", "head_state", "optimizer_state",
                            "anchor_coef", "raw2048", "arm", "iteration")


def validate_checkpoint(payload: Mapping) -> Mapping:
    missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in payload]
    if missing:
        raise PolicyContractError(
            f"checkpoint is missing {missing}: a checkpoint without the trained "
            "head cannot be evaluated on the states it was trained on"
        )
    if not payload["head_state"]:
        raise PolicyContractError("checkpoint head_state is empty")
    if payload["arm"] not in ("arm-F", "arm-B"):
        raise PolicyContractError(f"unknown arm {payload['arm']!r}")
    return payload


def continue_bias_for_survival(target_survival: float, horizon_blocks: int,
                               n_actions: int = 3) -> float:
    if not 0.0 < target_survival < 1.0:
        raise PolicyContractError(f"target_survival must be in (0, 1), got {target_survival}")
    if horizon_blocks < 1:
        raise PolicyContractError(f"horizon_blocks must be >= 1, got {horizon_blocks}")
    if n_actions < 2:
        raise PolicyContractError("need at least two actions")
    p_continue = target_survival ** (1.0 / horizon_blocks)
    return math.log(p_continue * (n_actions - 1) / (1.0 - p_continue))


def survival_under_bias(bias: float, horizon_blocks: int, n_actions: int = 3) -> float:
    p_continue = math.exp(bias) / (math.exp(bias) + (n_actions - 1))
    return p_continue ** horizon_blocks


def query_blocks(*, max_blocks: int, query_stride: int) -> list[int]:
    from rl_controller.rollout_runner import is_query_block
    return [b for b in range(1, max_blocks + 1)
            if is_query_block(b, max_blocks=max_blocks, query_stride=query_stride)]


def hazard_schedule(target_survival: float, *, max_blocks: int, query_stride: int,
                    power: float = 1.0) -> dict:
    if not 0.0 < target_survival < 1.0:
        raise PolicyContractError(f"target_survival must be in (0, 1), got {target_survival}")
    if power < 0:
        raise PolicyContractError(f"power must be >= 0, got {power}")
    blocks = query_blocks(max_blocks=max_blocks, query_stride=query_stride)
    if not blocks:
        raise PolicyContractError(
            f"no query block in 2..{max_blocks} with stride {query_stride}")
    shape = {b: (b / max_blocks) ** power for b in blocks}
    top = max(shape.values())
    log_target = math.log(target_survival)

    def log_survival(c):
        return sum(math.log(1.0 - c * v) for v in shape.values())

    lo, hi = 0.0, (1.0 / top) * (1.0 - 1e-12)
    if log_survival(hi) > log_target:
        raise PolicyContractError(
            f"target_survival {target_survival} is unreachable with a hazard "
            f"that must stay below 1 at the deepest query block")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if log_survival(mid) > log_target:
            lo = mid
        else:
            hi = mid
    c = 0.5 * (lo + hi)
    hazard = {b: c * v for b, v in shape.items()}
    return {"c": c, "hazard": hazard,
            "continue_bias": {b: continue_bias_for_hazard(h) for b, h in hazard.items()}}


def continue_bias_for_hazard(hazard: float, n_actions: int = 3) -> float:
    if not 0.0 < hazard < 1.0:
        raise PolicyContractError(f"hazard must be in (0, 1), got {hazard}")
    p = 1.0 - hazard
    return math.log(p * (n_actions - 1) / hazard)


def survival_under_schedule(continue_bias: Mapping, n_actions: int = 3) -> float:
    total = 0.0
    for bias in continue_bias.values():
        total += math.log(math.exp(bias) / (math.exp(bias) + (n_actions - 1)))
    return math.exp(total)


def build_policy(*, raw2048: bool, arm: str, hidden: int = 512, layers: int = 2,
                 dropout: float = 0.0, target_survival: float = 0.5,
                 horizon_blocks: int = 128, query_stride: int = 1,
                 hazard_power: float = 1.0):
    import torch
    from torch import nn

    if arm not in ("arm-F", "arm-B"):
        raise PolicyContractError(f"unknown arm {arm!r}")

    def mlp(in_dim, out_dim):
        blocks = []
        d = in_dim
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        blocks.append(nn.Linear(d, out_dim))
        return nn.Sequential(*blocks)

    schedule = hazard_schedule(target_survival, max_blocks=horizon_blocks,
                               query_stride=query_stride, power=hazard_power)
    prior = torch.zeros(horizon_blocks + 1)
    for b, bias in schedule["continue_bias"].items():
        prior[b] = bias
    scheduled = sorted(schedule["continue_bias"])
    for b in range(horizon_blocks + 1):
        if b not in schedule["continue_bias"]:
            nxt = next((q for q in scheduled if q >= b), scheduled[-1])
            prior[b] = schedule["continue_bias"][nxt]

    InnerHead = _torch_inner_head()
    modules = {"inner": InnerHead(mlp(inner_feature_dim(raw2048=raw2048), len(INNER_ACTIONS)),
                                  continue_prior=prior)}
    if arm == "arm-F":
        modules["outer"] = mlp(outer_feature_dim(), len(BOUNDARY_ACTIONS))
    policy = nn.ModuleDict(modules)

    with torch.no_grad():
        final = policy["inner"].mlp[-1]
        final.bias.zero_()
        final.weight.mul_(0.01)
        if "outer" in policy:
            outer_final = policy["outer"][-1]
            outer_final.bias.zero_()
            outer_final.bias[BOUNDARY_ACTIONS.index("draw")] = continue_bias_for_survival(
                target_survival, 8, n_actions=len(BOUNDARY_ACTIONS))
            outer_final.weight.mul_(0.01)
    return policy


_INNER_HEAD_CLS = None


def _torch_inner_head():
    global _INNER_HEAD_CLS
    if _INNER_HEAD_CLS is not None:
        return _INNER_HEAD_CLS
    import torch
    from torch import nn

    class InnerHead(nn.Module):

        def __init__(self, mlp, *, continue_prior):
            super().__init__()
            self.mlp = mlp
            self.register_buffer("continue_prior", continue_prior.clone().float())
            self.continue_index = INNER_ACTIONS.index("continue")

        def forward(self, x, block):
            logits = self.mlp(x)
            if not torch.is_tensor(block):
                block = torch.as_tensor(block, device=logits.device)
            block = block.to(device=logits.device, dtype=torch.long).reshape(-1)
            if block.numel() == 1 and logits.shape[0] != 1:
                block = block.expand(logits.shape[0])
            if block.numel() != logits.shape[0]:
                raise PolicyContractError(
                    f"{block.numel()} block indices for {logits.shape[0]} rows")
            block = block.clamp(0, self.continue_prior.numel() - 1)
            prior = torch.zeros_like(logits)
            prior[:, self.continue_index] = self.continue_prior[block]
            return logits + prior

    _INNER_HEAD_CLS = InnerHead
    return InnerHead


def __getattr__(name):
    if name == "InnerHead":
        return _torch_inner_head()
    raise AttributeError(name)


def save_checkpoint(path, *, policy, head, optimizer, anchor: PersistentAnchor,
                    raw2048: bool, arm: str, iteration: int,
                    policy_config: Mapping | None = None,
                    sampler_state: Mapping | None = None,
                    run_config: Mapping | None = None,
                    train_state: Mapping | None = None) -> None:
    import torch

    payload = {
        "policy_config": dict(policy_config or {}),
        "sampler_state": dict(sampler_state) if sampler_state is not None else None,
        "run_config": dict(run_config) if run_config is not None else None,
        "train_state": dict(train_state) if train_state is not None else None,
        "policy_state": policy.state_dict(),
        "head_state": head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "anchor_coef": anchor.coef,
        "raw2048": bool(raw2048),
        "arm": arm,
        "iteration": int(iteration),
    }
    validate_checkpoint(payload)
    torch.save(payload, path)


def load_checkpoint(path, *, policy, head, optimizer=None) -> Mapping:
    import torch

    payload = validate_checkpoint(torch.load(path, map_location="cpu"))
    policy.load_state_dict(payload["policy_state"])
    head.load_state_dict(payload["head_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
