# Vendored: ROM (Real-time Overthinking Mitigation)

Upstream: <https://github.com/SaFo-Lab/ROM>
Commit: `c2e651e0bcfd9bced14a79c844bb5d0f2988003c`
License: **MIT** — Copyright (c) 2025 Xinyan Wang, Xiaogeng Liu, Chaowei Xiao.
The upstream licence is vendored verbatim beside this file as `LICENSE` and
governs the vendored code below.

Paper: *ROM: Real-time Overthinking Mitigation via Streaming Detection and
Intervention*, arXiv 2603.22016. Dossier:
`docs/superpowers/recon/2026-09-01-baseline-campaign/paper_2603_22016.json`.

## What is vendored

| ours | upstream | sha256 of the upstream file |
|---|---|---|
| `streaming_head.py` | `rom/models.py` | `d7c7e2351fe05a7fa041937e1c79542b017580482f69efed018d2e00628fe9c6` |

`AttentionLayer`, `CfcCell` and `StreamingHead` are ported. Nothing else is:
`Qwen3WithHead`, `rom/train.py`, `rom/dataset.py`, `rom/eval.py` and
`rom/env.py` are **not** vendored — see "What is deliberately not vendored".

**No `ncps` dependency.** The dossier flagged the CfC as possibly needing the
`ncps` package. It does not: upstream's `CfcCell` is self-contained, ~40 lines
of explicit GRU-style gates plus a time-scaling term, and is ported as-is.

## Changes made, and why each was necessary

Every change is listed. The recurrence math and the head topology are
unchanged; the adaptations are about granularity, batching correctness and
numerics.

1. **Block granularity (the point of the port).** Upstream consumes one vector
   per *token*; we consume one vector per *block end*, from the WS-E block-feature
   cache. `StreamingHead.forward` therefore takes
   `(x[B, M, D], lengths[B], dt[B, M])` — an explicit padded batch with real
   lengths — instead of `(x[B, T_total, D], assistant_start)`. Campaign plan §4
   WS-D makes block granularity the primary arm; it is also where our labels and
   our firing live, and it removes upstream's `O(T²)` full-prefix attention cost.

2. **Padding is masked in attention.** Upstream pads embeddings with zeros and
   masks *only* causally, so padded positions are attended to as if they were
   real keys — which makes a chain's score depend on which other chains shared
   its batch. We mask padded keys. `tests/test_iclr_rom_baseline.py` pins the
   invariance (same chain, different batch composition and padding ⇒ identical
   logits to 1e-5). This is a **bug fix**, not a design change; without it the
   scored dumps would not be reproducible.

2b. **The prompt is attended over, not just pooled into `m_0`.** Upstream runs
   attention over the FULL prompt+assistant sequence and selects assistant
   positions only inside the recurrence, so every assistant query has a direct
   attention path to the prompt keys. An earlier revision of this port attended
   over block features alone and used the prompt only to build `m_0`, which
   silently deleted that path; the cross-model review caught it. `forward` now
   attends over `[prompt | blocks]` and slices the block half back out — with the
   causal mask and the prompt first, every block query reaches every VALID
   prompt key exactly as upstream, and masked padding removes the need for
   upstream's per-row offset bookkeeping. `attend_prompt=False` keeps the
   pooled-only variant as a **declared** ablation.

3. **Causal mask built on demand.** Upstream registers a `(1, 8192, 8192)` bool
   buffer (67 MB) in every head. At block granularity `M` is tens, so the mask
   is built per forward.

4. **`-1e9` → `torch.finfo(dtype).min`.** `-1e9` overflows fp16. Self-attention
   is always permitted (`allowed[i, i] = True`) so no query row can be fully
   masked, which is what would otherwise produce NaNs on padded rows.

5. **Plain Python recurrence loop instead of `@torch.jit.script`.** Upstream
   JITs the CfC loop to amortise per-token Python overhead over thousands of
   tokens. At tens of blocks that overhead is irrelevant, and the scripted
   function is materially harder to debug. Same arithmetic, verified against a
   direct `CfcCell.forward` rollout in the tests.

6. **BCE with one logit is the default; `num_labels=2` is retained.** ROM's
   *paper* states token-level BCE; ROM's *code* uses
   `CrossEntropyLoss(ignore_index=-100)` over two logits. These are the same
   model up to reparametrisation (softmax over two logits = sigmoid of their
   difference). We default to one logit + `BCEWithLogitsLoss` because the paper
   says so and because it makes `pos_weight` (the pre-registered
   inverse-frequency class-balance arm) expressible. The two-logit path is kept
   so the upstream parametrisation stays runnable. Deviation **D6** in
   `docs/superpowers/contracts/prereg-ws-d.md`.

7. **`dt` is supplied by the caller.** Upstream hardcodes a uniform
   `dt = 1/(T-1)`. Blocks have very unequal token lengths, so uniform steps
   would discard the only continuous-time information the CfC exists to use.
   `dt_mode="token_gap"` (normalised token distance between consecutive block
   ends) is the default; `dt_mode="uniform"` reproduces upstream exactly.
   Pre-registration amendment **A1**.

8. **Query-aware `m_0` is optional and declared.** Upstream always initialises
   the memory from prompt hidden states, under `torch.no_grad()` (so the prefix
   parameters receive no gradient — reproduced by default, and recorded per run
   as `provenance.rom_m0_grad`). Our block-feature cache may not carry prompt
   features; when it does not, `m_0 = 0` and the run records
   `provenance.rom_m0_init = "zeros"`. Deviation **D7**.

9. **fp32 by default.** Upstream trains the head in bf16. The head is ~5M
   parameters; fp32 costs nothing here and removes a class of silent numeric
   differences between the training and scoring passes. bf16 remains selectable.

10. **`reset_parameters` seeding.** Upstream calls `torch.nn.init.orthogonal_`
    on the six recurrent matrices, leaving `nn.Linear` defaults elsewhere. Kept
    as-is; the training seed is set by the caller so the two seeds required by
    the campaign's trained-method standard are genuine init+order seeds.

## What is deliberately not vendored

- **`rom/env.py`** — `setup_hf_cache` mutates `HF_HOME` / `HF_HUB_CACHE`
  process-wide; `setup_single_gpu` sets `CUDA_VISIBLE_DEVICES`,
  `CUDA_DEVICE_ORDER`, `NCCL_P2P_DISABLE`, `NCCL_IB_DISABLE`; `rom/eval.py` sets
  `VLLM_USE_V1=0` at *import* time. Importing any of this would silently
  reconfigure our jobs (we pin `CUDA_VISIBLE_DEVICES` ourselves per subprocess).
  `set_seed` is re-implemented locally without the global cuDNN flag mutation.
- **`rom/eval.py`'s intervention** — its backtrace-to-boundary is a **no-op at
  block granularity** (our probes already sit on blank-line block boundaries, so
  the nearest well-formed boundary to a firing block end *is* that block end),
  and its regenerated close-think conclusion is replaced by our stored
  forced-answer probe. Deviation **D5**. For the record, upstream's cue string
  is `"\nBased on my analysis above, I will now provide the final analysis
  first, and then the final answer in the box. </think>"`; we do not emit it.
- **`rom/dataset.py`'s hidden-state cache** — one `.pt` per sample and a
  standalone re-tokenisation of each attempt chunk to lay out labels. We read
  the shared WS-E block-feature cache instead. The re-tokenisation quirk (a
  chunk tokenised alone does not always tokenise identically in context) is
  reproduced only inside the QwQ label-transfer pilot, where it is part of what
  is being transferred.
- **`Qwen3WithHead`** — an online-generation wrapper. Our evaluation is offline
  replay over cached features.

## Labels come from WS-B

The FCS/post-FCS labels are **not** re-derived here.
`rl_controller/iclr/labels.py` (`fcs_index`, `post_fcs`,
`fcs_first_probe_chain_fraction`) and `rl_controller/iclr/targets.py`
(`first_correct_probe`, `_post_fcs`, `ROM_DROP_FLAG`) are WS-B's, and this
package imports them. `tests/test_iclr_rom_baseline.py` asserts that the
streaming class-balance accumulator here reproduces `labels.label_summary`
exactly, so the two cannot drift apart silently.
