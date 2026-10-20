from __future__ import annotations

import math

from rl_controller.baselines.ospruner.layout import (DEFAULT_N_UNFROZEN, HEAD_PREFIX,
                                                     BatchPlan, assert_frozen_backbone,
                                                     layer_split, partition_parameters)

HEAD_MODES = {"osp": ("pi",), "qv": ("q", "v")}

MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def logit(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"init fire rate must be in (0, 1), got {p}")
    return math.log(p / (1.0 - p))


def load_backbone(*, model_path: str | None = None, model_id: str = MODEL_ID,
                  revision: str = MODEL_REVISION, dtype: str = "bfloat16",
                  cache_dir: str | None = None):
    import torch
    from transformers import AutoModel

    reference = model_path or model_id
    if model_path and not str(model_path).rstrip("/").endswith(revision):
        raise ValueError(f"--model-path {model_path!r} does not end in the pinned revision "
                         f"{revision!r}: refusing to load an unpinned model by path")
    torch_dtype = getattr(torch, dtype)
    return AutoModel.from_pretrained(reference, revision=revision, torch_dtype=torch_dtype,
                                     cache_dir=cache_dir)


class TopTwoReader:

    def __init__(self, backbone, *, head_mode: str = "osp",
                 n_unfrozen: int = DEFAULT_N_UNFROZEN, init_fire_rate: float = 0.1,
                 bias_init=None, fp32_trainable: bool = True, output_names=None):
        import torch

        if head_mode not in HEAD_MODES:
            raise ValueError(f"head_mode {head_mode!r} not in {sorted(HEAD_MODES)}")
        self.head_mode = head_mode
        if output_names is None:
            self.outputs = HEAD_MODES[head_mode]
        else:
            self.outputs = tuple(output_names)
            if head_mode == "osp" and len(self.outputs) != 1:
                raise ValueError(f"the OSP stop unit is a single output, got {self.outputs}")
            if head_mode == "qv" and self.outputs[:2] != ("q", "v"):
                raise ValueError(f"a qv head must start with q, v (the ladder's mandatory "
                                 f"beliefs), got {self.outputs}")
        self.n_unfrozen = n_unfrozen
        config = backbone.config
        self.n_layers = int(config.num_hidden_layers)
        self.hidden_size = int(config.hidden_size)
        if n_unfrozen > self.n_layers:
            raise ValueError(f"n_unfrozen {n_unfrozen} > backbone depth {self.n_layers}")

        head = torch.nn.Linear(self.hidden_size, len(self.outputs))
        torch.nn.init.zeros_(head.weight)
        if bias_init is None:
            bias_init = [logit(init_fire_rate)] * len(self.outputs)
        if len(bias_init) != len(self.outputs):
            raise ValueError(f"{len(bias_init)} bias inits for {len(self.outputs)} outputs")
        with torch.no_grad():
            head.bias.copy_(torch.tensor(bias_init, dtype=head.bias.dtype))

        self.module = torch.nn.Module()
        self.module.add_module("backbone", backbone)
        self.module.add_module(HEAD_PREFIX, head)
        self.backbone = backbone
        self.head = head
        self._freeze(fp32_trainable=fp32_trainable)
        self._isolation_verified = False


    def _freeze(self, *, fp32_trainable: bool) -> None:
        import torch

        names = [n for n, _ in self.module.named_parameters()]
        trainable, frozen = partition_parameters(names, n_layers=self.n_layers,
                                                 n_unfrozen=self.n_unfrozen)
        expected_blocks = self.n_unfrozen * _params_per_block(self.backbone)
        if not trainable:
            raise ValueError("the freeze plan matched no parameters: the backbone's parameter "
                             "names do not look like `...layers.<i>...` and the partition would "
                             "silently train the head alone")
        by_name = dict(self.module.named_parameters())
        for name in frozen:
            by_name[name].requires_grad_(False)
        for name in trainable:
            param = by_name[name]
            param.requires_grad_(True)
            if fp32_trainable and param.dtype != torch.float32:
                param.data = param.data.float()
        assert_frozen_backbone(((n, p.requires_grad) for n, p in self.module.named_parameters()),
                               n_layers=self.n_layers, n_unfrozen=self.n_unfrozen)
        counted = sum(by_name[n].numel() for n in trainable if not n.startswith(HEAD_PREFIX))
        if counted != expected_blocks:
            raise ValueError(f"trainable block parameters {counted} != the {expected_blocks} the "
                             f"layout predicts for {self.n_unfrozen} blocks — the freeze plan "
                             "selected the wrong tensors")
        self.trainable_names = tuple(trainable)
        self.frozen_names = tuple(frozen)
        self.n_trainable = counted + sum(p.numel() for p in self.head.parameters())

    def parameters(self):
        return [p for p in self.module.parameters() if p.requires_grad]

    def train(self) -> None:
        self.module.eval()

    def eval(self) -> None:
        self.module.eval()


    def verify_graph_isolation(self, input_ids, attention_mask) -> dict:
        import torch

        frozen, _trainable = layer_split(self.n_layers, self.n_unfrozen)
        layers = _decoder_layers(self.backbone)
        seen = {}

        def hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            seen["requires_grad"] = bool(tensor.requires_grad)

        handle = layers[frozen[-1]].register_forward_hook(hook)
        try:
            with torch.enable_grad():
                self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                              use_cache=False)
        finally:
            handle.remove()
        if "requires_grad" not in seen:
            raise RuntimeError("graph-isolation probe never fired: the hooked module did not run")
        if seen["requires_grad"]:
            raise RuntimeError(
                f"block {frozen[-1]} (0-based) output requires grad: the frozen prefix is "
                "building an autograd graph, so gradients can reach blocks 1-"
                f"{len(frozen)} and this is not a top-{self.n_unfrozen}-unfrozen reader")
        self._isolation_verified = True
        return {"last_frozen_layer": frozen[-1], "requires_grad": False}

    def boundary_logits(self, input_ids, attention_mask, plan: BatchPlan,
                        *, autocast: bool = True):
        import torch

        if input_ids.shape[0] != plan.batch_size:
            raise ValueError(f"{input_ids.shape[0]} rows for a plan of {plan.batch_size}")
        if input_ids.shape[1] != plan.max_len:
            raise ValueError(f"padded width {input_ids.shape[1]} != plan max_len {plan.max_len}")
        device = input_ids.device
        ctx = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
               if autocast else _nullcontext())
        with ctx:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                use_cache=False)
        hidden = out.last_hidden_state
        rows = torch.tensor([r for r, _ in plan.gather], dtype=torch.long, device=device)
        cols = torch.tensor([c for _, c in plan.gather], dtype=torch.long, device=device)
        states = hidden[rows, cols]
        return self.head(states.float())

    def split_by_chain(self, logits, plan: BatchPlan) -> list:
        return [logits[plan.offsets[r]:plan.offsets[r + 1]] for r in range(plan.batch_size)]


    def manifest(self) -> dict:
        return {
            "head_mode": self.head_mode,
            "outputs": list(self.outputs),
            "n_layers": self.n_layers,
            "n_unfrozen": self.n_unfrozen,
            "frozen_zero_based": list(layer_split(self.n_layers, self.n_unfrozen)[0]),
            "trainable_zero_based": list(layer_split(self.n_layers, self.n_unfrozen)[1]),
            "n_trainable_params": self.n_trainable,
            "n_total_params": sum(p.numel() for p in self.module.parameters()),
            "n_trainable_tensors": len(self.trainable_names),
            "n_frozen_tensors": len(self.frozen_names),
            "graph_isolation_verified": self._isolation_verified,
            "aux_generated_tokens": 0,
        }


def _decoder_layers(backbone):
    layers = getattr(backbone, "layers", None)
    if layers is None:
        layers = getattr(getattr(backbone, "model", None), "layers", None)
    if layers is None:
        raise ValueError("backbone exposes neither `.layers` nor `.model.layers`")
    return layers


def _params_per_block(backbone) -> int:
    return sum(p.numel() for p in _decoder_layers(backbone)[0].parameters())


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def pad_batch_tensors(sequences, plan: BatchPlan, *, device=None, pad_id: int = 0):
    import torch

    if len(sequences) != plan.batch_size:
        raise ValueError(f"{len(sequences)} sequences for a plan of {plan.batch_size}")
    width = plan.max_len
    ids, mask = [], []
    for row, seq in enumerate(sequences):
        n = plan.lengths[row]
        if len(seq) != n:
            raise ValueError(f"row {row}: {len(seq)} tokens but the plan says {n}")
        ids.append(list(seq) + [pad_id] * (width - n))
        mask.append([1] * n + [0] * (width - n))
    return (torch.tensor(ids, dtype=torch.long, device=device),
            torch.tensor(mask, dtype=torch.long, device=device))
