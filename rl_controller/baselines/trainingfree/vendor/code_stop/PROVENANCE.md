# Vendored reference: CoDE-Stop

Upstream: <https://github.com/sudoparsa/CoDE-Stop>
Commit: `b5081e7c2abe23bb1d19649421cc13522fee7c50` ("remove unused code", 2026-04-28T13:32:49-04:00)
Fetched: 2026-09-01, `git clone --depth 1`.
License: **MIT**, Copyright (c) 2026 Parsa Hosseini — `LICENSE` is copied here verbatim
and governs every file in this directory.

Paper: "Early Stopping for Large Reasoning Models via Confidence Dynamics",
Hosseini, Nawathe, Salmani, Razaviyayn, Feizi — arXiv **2604.04930**.
Campaign dossier: `docs/superpowers/recon/2026-09-01-baseline-campaign/paper_2604_04930.json`.

## Status: REFERENCE ONLY — never imported, never executed

These files are copied **unmodified** so the ported rules in
`rl_controller/baselines/trainingfree/rules.py` can be audited line-by-line against the
authors' own implementation. They import `env_vars`, `models`, `nltk`, `math_verify` and
`transformers`, none of which resolve from this directory; that is deliberate. There is no
`__init__.py`, so this directory is not a Python package and nothing here is on any import
path. The port is a **re-implementation against our offline replay**, not a call into this
code.

## Files and sha256 (as fetched)

| file | sha256 | what we harvested |
|---|---|---|
| `LICENSE` | `2e244c45435b1f6ab0342e7ce519775afc1b849fc8c1dea4c16de7c231705f9e` | MIT terms |
| `README.md` | `461bd8628822b21df1416bc634a49ccd63ce6e6cbdff885e544f33c40a25a71e` | usage / registries |
| `cache_utils.py` | `e4bcea4a8c9bb5f76a4e2cacf7070c4c302de4272645d974e1de008636f73d04` | KV-cache clone/move pattern |
| `inference.sh` | `77a8de2e870ab181b779d59aa3463de0b68274f4437734c05cc812aeec5800e6` | the authors' own hyperparameter values |
| `method_codestop.py` | `d020fde481bf1ad2d4ee53ea43a3ae31560e05be1d6cf6f172489b54179cb1ff` | dual exit, ramp, degeneration score |
| `method_deer.py` | `340da5ad17b1be4e97e299ec37e84c711d6ada13fa6b28c018e48d4561a08145` | fixed-threshold confidence exit |
| `method_eat.py` | `9c9d0dda51a7594403d7866279ab8e3d46b4cc85bb309bb1b9b39d7828d53907` | EMA-variance stop rule |
| `method_think_or_not.py` | `848bdc5b91528cc705d5d25b1e4124d7e2eb7ba96c841fd2e922df1a2d1fdfcb` | beam-entropy rule |
| `method_wheels.py` | `939cc8b0b4bf76e50ff73b7343a9db4b1f647ed3bcde0dfec15523007ec5a3b7` | `</think>`-rank cascade |
| `method_ansconv_consistency.py` | `b1a67936182ee2f831de4b42c650e200012aafb46a170e26f2f84086a798bda9` | consecutive-agreement counter |

## What the source actually does (read from the code, not the paper)

These are the facts the port is written against; several differ from the paper's prose and
from our dossier summary, and each one is a place a naive port would silently diverge.

1. **The confidence `c_i` is a GREEDY-DECODED max-probability geometric mean, not the
   probability of a known answer.** `calcu_max_probs_w_kv` (`method_codestop.py:15-93`,
   duplicated in `method_deer.py:12-91`) greedy-decodes from the probe prompt, taking
   `torch.max(probs)` at each step and feeding the argmax back. With `method=0` (Qwen3,
   `ewt=True`) it accumulates `log(max_value)` for every step *except the first*, then
   returns `exp((Σ log p − log p_K) / (K−1))` where `K = total_steps` — i.e. it drops
   **both** the first generated token and the last (stop) token from the numerator while
   still dividing by `K−1`. Decoding stops on `</think>` (method 0) or on a wider newline
   set (method 1), capped at 20 steps (300 for LiveCodeBench).
2. **`ended_with_think` gates the confident exit.** With `ewt=True` the confidence exit
   fires only if the greedy probe actually terminated on `</think>`
   (`method_codestop.py:189`).
3. **The probe prompts differ per method** — four distinct strings, all injected at a step
   boundary:
   - CoDE-Stop / DEER: `"\n**Final Answer**\n\nThe final answer is \\boxed"`
   - EAT: `"\n</think>\n The final answer is \\boxed"` (`method_eat.py:14`)
   - Answer-Convergence: `"\n</think>\n\\boxed{"` (`method_ansconv_consistency.py:43`)
   - Think-or-Not: `"\n</think>\nThe correct answer is: \\boxed{"`
     (`method_think_or_not.py:69`)
   Stop-Spinning-Wheels injects nothing at all.
4. **The degeneration score is not the paper's formula.**
   `compute_degeneration_score` (`method_codestop.py:102-110`) is
   `Σ v(p₁,p₂)·w(t₂,T_K)` over **consecutive pairs**, restricted to pairs with `p₁ > p₂`
   (a downward step), with `v(p₁,p₂) = 1[2p₂ − p₁ < 0.55]` and
   `w(T_k,T_K) = log1p(T_K/T_k − 1) + 1`. Note `log1p(T_K/T_k − 1) = log(T_K/T_k)`, so the
   weight matches the paper's `log(T_K/T_i)+1`, but the `p₁ > p₂` filter and the
   pairwise-consecutive form are code-only. `use_log=True` (the default) first maps the
   confidences through `log(max(p,1e-12))`, so `δ = 0.55` is applied on the **log** scale,
   not on probabilities. `D_k` is forced to `0.0` until at least 3 probes exist
   (`method_codestop.py:171-173`).
5. **The ramp is `ramp_min + (ramp_max − ramp_min)·min(1, i/ramp_steps)`**
   (`compute_ramping_deer_threshold`, `method_codestop.py:96-99`), indexed by the
   0-based boundary index `i`, with `ramp_max = conf_threshold`.
6. **Both exits use `>` (strict), not `≥`** (`method_codestop.py:189`).
7. **Their step boundary is a token match, not a line break.** `stop_token_idxs` are the
   positions of any token of `stop_word` ("Wait" / "Alternatively") plus EOS inside the
   tokenized response (`method_codestop.py:132-134`). `inference.sh` shows the authors'
   own run used `--stop-word Alternatively`.
8. **Authors' hyperparameters, from `inference.sh`** (qwen-4b, gpqad):
   `--conf-threshold 0.95 --degen-threshold 9.5 --ramp-steps 1 --ramp-min 0.9
   --ewt True --stop-word Alternatively`.
9. **EAT is expensive and stochastic**: 10 sampled rollouts × 20 tokens at temperature 0.6
   per boundary, cumulative per-step entropy of the *processed* logits, averaged over
   rollouts (`method_eat.py:67-103`); the stop rule is an exponential moving *variance* of
   that series falling below a threshold after `warmup_steps` (default 50) boundaries
   (`method_eat.py:117-147`).
10. **Stop-Spinning-Wheels generates nothing**: it reads the rank of the `</think>` token
    id in the next-token distribution at each sentence boundary and applies a fixed
    4-condition cascade with no tunable knob (`method_wheels.py:31-56`).
11. **Answer-Convergence never refreshes `last_answer` on agreement** and resets the
    counter to 1 (not 0) on disagreement (`method_ansconv_consistency.py:99-103`);
    equality is `math_verify.verify`, and the probe is greedy (`do_sample` is not passed
    to `generate`), 10 new tokens max.
12. **Think-or-Not's threshold is `alpha / (e·ln 2)`** (`method_think_or_not.py:76`) and
    the statistic is the entropy of the top-5 completed **beam probabilities** (unnormalised
    products, renormalised inside `calculate_entropy`), from a `max_steps=10, top_k=5`
    beam search — deterministic, but ~50 forward passes per boundary.
