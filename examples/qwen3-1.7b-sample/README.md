# Qwen3-1.7B trajectory sample

A small, self-contained slice of the Qwen3-1.7B BELT pipeline. It links, for 16 trajectories, the exact token IDs, the prefix boundaries where the stopping policy is queried, the probe's beliefs and 768-dimensional representation at each of those boundaries, the forced-answer and natural-completion outcomes, the token costs, the split identifiers, and the fitted stopping policies of the matching probe.

Everything here is copied or byte-sliced from the archived files behind the paper. Nothing was regenerated. The one exception is `prompt_ids`, which were rebuilt from the question text with this repository's generation code and the pinned tokenizer. [PROVENANCE.json](PROVENANCE.json) records the input hashes and the checks, and `python3 -m unittest tests.test_qwen_sample -v` (standard library only) re-checks the files and recomputes every policy logit.

## What is in the sample

| File | Contents |
| --- | --- |
| `chains.jsonl` | 16 trajectories, one JSON object per line |
| `belt_h.f16`, `belt_h.index.jsonl`, `belt_h.source.json` | Probe representation H: one float16 row of width 768 per policy query point, in the paper's `frozen-repr-dump/v1` layout |
| `policies/belt_b_r_seed0.json`, `policies/belt_h_r_seed0.json` | Fitted BELT-B-R and BELT-H-R stopping policies at all 17 fitting prices, with their normalization |
| `policies/policy_fit_split.json` | Question keys of the 160 policy-fit and 160 development questions |
| `reference/policy_reference.jsonl` | Per-probe correctness, token costs and policy logits for each sample trajectory, computed with the paper's fitting code |
| `tools/` | The two scripts that produced this directory from the archived files |
| `PROVENANCE.json`, `SHA256SUMS` | Input hashes, code revisions, hardware and checks; file hashes |

The trajectories are all eight sampled chains of two MATH-500 questions from the out-of-distribution report (test) split: `test/number_theory/931.json` and `test/intermediate_algebra/966.json`. They were chosen by a fixed rule, without looking at stopping behaviour. For each of two outcome patterns, the rule takes the MATH-500 question whose chains contain that pattern and whose key has the smallest `sha256("belt-sample-v1:" + qkey)`:

- the first pattern is a natural completion with a parseable answer together with one that needed answer elicitation;
- the second is a natural completion with a parseable answer together with a chain that reached the thinking budget.

The sample therefore covers all three natural-completion cases: 12 parseable natural completions, 2 elicited ones, and 2 budget-capped chains.

**Probe.** All scores and representations come from seed 0 of [BELT-Qwen3-1.7B-ReProbeInit](https://huggingface.co/JingweiNi/BELT-Qwen3-1.7B-ReProbeInit) at revision `664179ee7431db265c6fed041790d66381f1b1d6` (`seed-0/s1_best.pt`, SHA256 `a14b5b5e…0829cd`), read with that package's `reader-template/`. This is the same probe that ThinkBooster loads by default. The scoring run is the one named in the package's seed-0 scoring receipt.

## Trajectory fields (`chains.jsonl`)

| Field | Meaning |
| --- | --- |
| `pool`, `split` | `math_ood_report`, `report`: the out-of-distribution test split |
| `source`, `source_id`, `level` | Benchmark and item identifier (MATH-500 file path) |
| `qkey`, `rep` | Question key used throughout the pipeline, and the chain's replicate index |
| `question`, `question_sha256`, `gold` | Question text, its hash, and the integer reference answer |
| `prompt_ids` | Generation prompt (see below) |
| `completion_ids` | All generated tokens, starting with `<think>` (151667) |
| `outcome` | `natural` (the model closed thinking itself) or `length` (it reached the 12,000-token budget) |
| `thinking_tokens` | Length of `completion_ids` |
| `blocks`, `block_ends` | Number of reasoning blocks and the completion-token offset at the end of each block; a block ends at a blank line |
| `query_points` | `[block, prefix_len]` for each policy query (probe); here every fourth block from block 4 |
| `probe_block_end_index` | Position of each probe in `block_ends` |
| `probes` | Per probe: `block`, `prefix_len`, the parsed forced answer `forced` (integer or `null`), and the number of tokens generated for it, `forced_tokens` |
| `scores` | Per probe: beliefs `q`, `v`, `r`, `stability` (probabilities) = **Now**, **End**, **Later**, **Same** |
| `natural` | Natural-completion outcome: `answer`/`correct` as used by the paper's readers, `answer_before_elicitation`/`correct_before_elicitation`, `elicited`, `elicitation_tokens` |
| `forced_answer_suffix_ids` | Tokens appended after a prefix to request a forced answer |
| `natural_elicitation_suffix_ids` | Present when `elicited` is true: tokens appended after the full completion to elicit its answer |
| `h` | Location of this chain's rows in `belt_h.f16`: byte `offset` and `n_probes` |

## Reconstructing every model input

**Generation prompt.** The prompt is Qwen3's chat template with thinking enabled and the generation prompt added, applied to one user turn:

```
Answer the following question. Reply with the answer only. Answer enclosed with \boxed{}.
<Question>: {question}
```

The rendered prompt ends with `<|im_start|>assistant\n`. Sampling used temperature 0.6, top-p 0.95 and top-k 20.

**Prefix at a probe.** The prefix is `prompt_ids + completion_ids[:prefix_len]`. `prefix_len` counts completion tokens from the leading `<think>` through the token that ends block `block`.

**Reader input.** The probe reads `prompt_ids + completion_ids[:block_ends[-1]]` in one teacher-forced pass and produces a belief vector and an H row at every block end. The sample keeps the rows at the probes. Row `i` of a chain in `belt_h.f16` and `scores[i]` both belong to `probes[i]`.

**Forced answer at a probe.** The request is `prefix + forced_answer_suffix_ids`, where the suffix is `</think>\n\n\boxed{` = `[151668, 271, 59, 79075, 90]`. Generation stops at `}` or after 32 new tokens. Its integer is `forced`, and its token count is `forced_tokens`. The forced-answer token IDs themselves were not stored.

**Natural answer.** For a naturally closed chain, the answer is parsed from the completion. Only when that fails is the answer elicited, with `prompt_ids + completion_ids + natural_elicitation_suffix_ids` (here `\boxed{`). Elicitation charges `elicitation_tokens` and leaves parseable answers unchanged. Budget-capped chains are not elicited. The two capped chains in this sample have no answer and count as incorrect.

## Token costs

| Event | Generated tokens charged |
| --- | --- |
| Stop at probe `i` | `prefix_len + forced_tokens` |
| Natural completion | `thinking_tokens + elicitation_tokens` (`elicitation_tokens` is 0 unless elicited; capped chains: `thinking_tokens`) |

`reference/policy_reference.jsonl` lists both, as `probe_stop_cost_tokens` and `natural_cost_tokens`, together with per-probe correctness. They were computed with the fitting code's `chain_terms(cost="total")` under the elicited-answer cost installer.

## Splits

- **Test.** `split: report` marks the out-of-distribution report split. `source` and `source_id` identify each benchmark item.
- **Policy fitting.** Policies are fitted on the probe's validation bank: the 320 questions of `seed-0/val_scored.jsonl` in the checkpoint package, with natural answers elicited as above. That bank is split by question. The question keys are sorted by the hex digest of `sha256("iclr-policy-split-2026-09-15:" + qkey)`; the first half is the fit split and the second half the development split. `policies/policy_fit_split.json` lists both halves.

## Stopping policies

Both files hold the 17 price-refitted sigmoid-linear policies of seed 0, as fitted for the paper. Each row has its price `lambda`, `w`, `b`, and the selected regularization and initialization.

| | BELT-B-R | BELT-H-R |
| --- | --- | --- |
| Features `x_i` at probe `i` | `logit(q), logit(v), logit(r), logit(stability), prefix_len / 12000`; each probability clipped to `[1e-4, 1 - 1e-4]` before the logit | The 768 float16 H coordinates of row `i`, converted to float64 |
| Normalization | `mean`, `std` (5 values each) | `mean`, `std` (768 values each) |

The decision logit is `z_i = w · ((x_i − mean) / std) + b`, and the stop probability is `π_i = sigmoid(z_i)`. The fitting objective and the paper's readouts treat the policy as stochastic. The run stops at probe `i` with probability `π_i · ∏_{j<i} (1 − π_j)`; otherwise it reaches natural completion. A deterministic deployment that stops at the first probe with `z_i ≥ 0` (π ≥ 0.5) is a different readout. ThinkBooster's `BeltStoppingPolicy` takes the same `w`, `b`, `mean` and `std` with an explicit logit threshold (`hidden_dtype: "float16"` for H-R).

The normalization arrays were not stored with the original fits. They are a deterministic function of the fit split, and were recomputed with the fitting code. With these arrays, re-reading all 17 stored rows of each policy reproduces the archived fit records' development and pool readouts exactly (maximum absolute difference 0.0; see `PROVENANCE.json`). The B-R arrays can also be recomputed from the checkpoint package's `seed-0/val_scored.jsonl`, because they depend only on the fit questions' beliefs and prefix lengths. The H-R arrays need the validation representation dump, which is not public.

`reference/policy_reference.jsonl` gives, for each trajectory, each policy and each price, the per-probe logits `z_i`, as well as the expected correctness and expected generated tokens under the stochastic readout. These are reference values for checking an implementation; they are not evaluation results.

## Comparisons between scoring backends

The existing comparison of ThinkBooster's native HF scorer with its opt-in shared vLLM backend is documented in the [ThinkBooster BELT guide](https://github.com/IINemo/thinkbooster/blob/main/docs/strategies/belt.md#opt-in-shared-vllm-backend). On 2026-10-18 it checked five fixed prefixes on one RTX 4090 with this seed-0 probe. Four passed the predeclared limits. The 1,238-token case exceeded all four limits (layer-16 features, H, beliefs and a fixture policy logit), and HF remains the default backend there. That check used fixed prefixes, not these trajectories. ThinkBooster's CPU tests compare its native reader with BELT's original reader forward on the same input (tolerance 1e-6), and its example prompt with BELT's prompt token for token. No check has yet compared either backend with archived scores such as the ones in this sample. The sample's `prompt_ids`, `completion_ids`, `query_points`, `scores` and H rows are the inputs and reference outputs such a comparison needs.

## Not included

- **Layer-16 backbone hidden states.** The paper's scorer computes them on the fly and never stored them. Only the probe's 768-dimensional H rows were kept.
- **Forced-answer and elicitation token IDs.** Only the parsed answer and the token count were kept.
- **Full trajectory banks and question rosters.** Use the [recipes](../../docs/RECIPES.md) to build them.

## Reading the files

```python
import json
import numpy as np

chains = [json.loads(line) for line in open("chains.jsonl")]
blob = np.fromfile("belt_h.f16", dtype="<f2")
c = chains[0]
h = blob[c["h"]["offset"] // 2:][: c["h"]["n_probes"] * 768].reshape(-1, 768)
assert h.shape[0] == len(c["probes"]) == len(c["scores"])
```
