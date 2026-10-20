from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

UPSTREAM_RECIPE = {
    "lr": 5.0e-5,
    "weight_decay": 0.1,
    "num_train_epochs": 20,
    "warmup_ratio": 0.1,
    "batch_size": 8,
    "gradient_acc_steps": 4,
    "betas": (0.9, 0.95),
    "eps": 1e-8,
    "max_grad_norm": 1.0,
    "scheduler": "cosine",
}


class AttentionLayer(nn.Module):

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, valid_mask=None):
        _, seq_len, _ = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / ((k.size(-1) + 1e-6) ** 0.5)

        idx = torch.arange(seq_len, device=x.device)
        allowed = idx.view(1, -1, 1) >= idx.view(1, 1, -1)
        if valid_mask is not None:
            allowed = allowed & valid_mask.unsqueeze(1)
        allowed = allowed | torch.eye(seq_len, dtype=torch.bool, device=x.device).unsqueeze(0)

        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v), weights


class CfcCell(nn.Module):

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        for gate in ("z", "r", "h"):
            self.register_parameter(f"W{gate}", nn.Parameter(torch.empty(input_dim, hidden_dim)))
            self.register_parameter(f"U{gate}", nn.Parameter(torch.empty(hidden_dim, hidden_dim)))
            self.register_parameter(f"b{gate}", nn.Parameter(torch.zeros(hidden_dim)))
        self.reset_parameters()

    def reset_parameters(self):
        for name in ("Wz", "Uz", "Wr", "Ur", "Wh", "Uh"):
            nn.init.orthogonal_(getattr(self, name))
        for name in ("bz", "br", "bh"):
            nn.init.zeros_(getattr(self, name))

    def forward(self, x, h_prev, dt):
        z = torch.sigmoid(x @ self.Wz + h_prev @ self.Uz + self.bz)
        r = torch.sigmoid(x @ self.Wr + h_prev @ self.Ur + self.br)
        h_hat = torch.tanh(x @ self.Wh + (r * h_prev) @ self.Uh + self.bh)
        h_new = (1 - z) * h_prev + z * h_hat
        return h_new + dt * (h_new - h_prev)


class StreamingHead(nn.Module):

    def __init__(self, input_dim: int = 2048, proj_dim: int = 512, mem_dim: int = 512,
                 num_labels: int = 1, cfc: bool = True, attend_prompt: bool = True,
                 head_mode: str | None = None):
        super().__init__()
        if head_mode is None:
            head_mode = "ce2" if num_labels == 2 else "bce"
        if head_mode not in ("bce", "ce2"):
            raise ValueError(f"head_mode must be 'bce' or 'ce2', got {head_mode!r}")
        if head_mode == "ce2" and num_labels != 2:
            raise ValueError("head_mode 'ce2' requires num_labels == 2")
        if num_labels < 1:
            raise ValueError(f"num_labels must be >= 1, got {num_labels}")
        self.head_mode = head_mode
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.mem_dim = mem_dim
        self.num_labels = num_labels
        self.cfc_enabled = bool(cfc)
        self.attend_prompt = bool(attend_prompt)

        self.attention = AttentionLayer(input_dim, proj_dim)
        self.cfc = CfcCell(proj_dim, mem_dim) if cfc else None
        self.mem_head = nn.Linear(mem_dim if cfc else proj_dim, num_labels)
        self.prefix_scorer = nn.Linear(proj_dim, 1, bias=False)
        self.prefix_to_h = nn.Sequential(nn.Linear(proj_dim, mem_dim), nn.Tanh())


    def parameter_count(self, *, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad or not trainable_only)

    def config(self) -> dict:
        return {"input_dim": self.input_dim, "proj_dim": self.proj_dim,
                "mem_dim": self.mem_dim, "num_labels": self.num_labels,
                "head_mode": self.head_mode, "cfc": self.cfc_enabled,
                "attend_prompt": self.attend_prompt, "params": self.parameter_count()}


    def _pool_prefix(self, prefix_feat, prompt_mask, *, grad: bool = False):
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            scores = self.prefix_scorer(prefix_feat).squeeze(-1)
            if prompt_mask is not None:
                scores = scores.masked_fill(~prompt_mask, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1).unsqueeze(1)
            pooled = torch.bmm(weights, prefix_feat).squeeze(1)
            return self.prefix_to_h(pooled)

    def forward(self, x, lengths, dt=None, *, prompt=None, prompt_mask=None,
                m0_grad: bool = False):
        if x.dim() != 3:
            raise ValueError(f"x must be (B, M, D), got {tuple(x.shape)}")
        batch, steps, width = x.shape
        lengths = lengths.to(device=x.device, dtype=torch.long)
        if int(lengths.min()) < 1:
            raise ValueError("every chain needs at least one block end")
        if int(lengths.max()) > steps:
            raise ValueError(f"lengths max {int(lengths.max())} exceeds padded M={steps}")
        valid = torch.arange(steps, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)

        use_prompt = prompt is not None and self.attend_prompt
        if use_prompt:
            if prompt.shape[0] != batch or prompt.shape[-1] != width:
                raise ValueError(f"prompt must be (B, P, {width}), got {tuple(prompt.shape)}")
            n_prompt = prompt.shape[1]
            if prompt_mask is None:
                prompt_mask = torch.ones(batch, n_prompt, dtype=torch.bool, device=x.device)
            combined = torch.cat([prompt.to(x.dtype), x], dim=1)
            combined_valid = torch.cat([prompt_mask, valid], dim=1)
            attended = self.attention(combined, combined_valid)[0]
            prefix_feat, feat = attended[:, :n_prompt], attended[:, n_prompt:]
        else:
            prefix_feat = None
            if prompt is not None and prompt_mask is None:
                prompt_mask = torch.ones(batch, prompt.shape[1], dtype=torch.bool,
                                         device=x.device)
            feat = self.attention(x, valid)[0]
            if prompt is not None:
                prefix_feat = self.attention(prompt.to(x.dtype), prompt_mask)[0]
        feat = feat * valid.unsqueeze(-1).to(feat.dtype)

        if not self.cfc_enabled:
            return self.mem_head(feat)

        if dt is None:
            dt = torch.zeros(batch, steps, device=x.device, dtype=x.dtype)
        dt = dt.to(device=x.device, dtype=feat.dtype)

        if prefix_feat is None:
            h = torch.zeros(batch, self.mem_dim, device=x.device, dtype=feat.dtype)
        else:
            h = self._pool_prefix(prefix_feat, prompt_mask, grad=m0_grad)
        outputs = []
        for t in range(steps):
            active = valid[:, t].unsqueeze(-1)
            h_new = self.cfc(feat[:, t], h, dt[:, t].unsqueeze(-1))
            h = torch.where(active, h_new, h)
            outputs.append(self.mem_head(h))
        return torch.stack(outputs, dim=1)

    def probabilities(self, logits):
        if self.head_mode == "ce2":
            return torch.softmax(logits, dim=-1)[..., 1]
        if self.num_labels == 1:
            return torch.sigmoid(logits.squeeze(-1))
        return torch.sigmoid(logits)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass
