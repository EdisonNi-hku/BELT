# Reproduction recipes

These commands expose the production entry points used for the paper. They need the real inputs (question manifests, trajectory banks, trained runs); paths below are shell variables to be set to those inputs. See the [asset list](ARTIFACTS.md) for what is downloadable. A fresh stochastic run is not expected to reproduce floating-point results bit for bit.

## Source selection

`provenance/code_sources.json` is the authoritative per-file byte manifest. The original source paths are retained there even when small production scripts originally stored under `results/` are relocated into `scripts/`.

| Recipe | Source revision | Location |
| --- | --- | --- |
| Qwen corrected banks, BELT training/scoring, graders, OSP | `30b7b53b21bdd0f8b5eed38d08b3a3538a4ec04b` | repository root |
| Qwen BELT-B-R/H-R L-BFGS fitting | `521553ad` | `scripts/source_variants/qwen_policy/` |
| OLMo BELT runtime and scorer (agent prompt-ID extension included) | `ea330eddd0a5be03373e904e8ca45ac4fb6504a4` | `scripts/source_variants/olmo_belt/` |
| OLMo full-OSP trainer and scorer | `996adf51` | `scripts/source_variants/olmo_osp_full/` |
| OLMo archived OSP scorer (agent prompt-ID extension included) | `20b001270779dc59fb12235fb23a2af38c5d9185` | `scripts/source_variants/olmo_osp_archived/` |
| OLMo policy fitting/readout/replay | `0647deab` | `scripts/source_variants/olmo_policy/` |
| OLMo single-direction threshold scripts | `14db20ed`, `156ef08d` | `scripts/source_variants/olmo_policy/scripts/x13/` |
| ALFWorld/ScienceWorld tree, calibration, readings | `8bb0aad27747c27416098d82662256d3efd61076` | `scripts/olmo_alfworld/` |
| OLMo code generation and bank assembly | `c75ce840251cba2c1fd0a5ac067cd95eeda62cf0` | `scripts/source_variants/olmo_code/` |

Snapshots retain the local imports of the primary entry points. `scripts/belt_ext/d53/fam_osp.py` is included from revision `30b7b53b21bdd0f8b5eed38d08b3a3538a4ec04b` because the OLMo code row producer imports it. Comments and docstrings were removed from first-party Python files for the release; each file was checked to have an unchanged abstract syntax tree apart from docstrings. Some shared controller modules are included only because the BELT trainer imports `HeadStateFn` from `rl_train.py`.

Set `BELT_ROOT` to this checkout's absolute path. For a snapshot command, change to its directory in a subshell and put that directory first on `PYTHONPATH`:

```bash
export BELT_ROOT="$(pwd)"
export U4R_NATIVE_ROOT="$BELT_ROOT/vendor/uncertainty4reasoning-native-thinking"
(
  cd "$BELT_ROOT/scripts/source_variants/olmo_code"
  PYTHONPATH="$PWD:$BELT_ROOT" python3 -m rl_controller.scripts.single_cot_probe --help
)
```

The `cd` matters: launching `python -m rl_controller...` from the repository root otherwise prefers the root Qwen package. All input/output variables used below should be **absolute paths**. Keep the source manifest with outputs to identify the historical snapshot, since `git rev-parse` inside an extracted snapshot reports this release repository's commit.

## Environments and model pins

The CPU unit tests use the standard library. NumPy-based policy replay and agent readings additionally use `requirements/cpu-analysis.txt`. `requirements/training.txt` lists additional runtime imports; its ranges are not a historical environment lock.

The paper records Qwen RTX 4090 generation with PyTorch `2.7.0+cu126`, Transformers `4.52.4`, and vLLM `0.9.0.1`. In a separate CUDA-capable environment:

```bash
python3 -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
python3 -m pip install -r requirements/qwen-generation.txt -r requirements/training.txt
```

The archived GH200 OLMo stack uses custom container builds, recorded in `requirements/olmo-gh200-environment.json`. Recover that image for numerical reruns; a current PyPI stack is not asserted equivalent. OSP-full also uses per-run A100 device receipts. Agent environments additionally need `requirements/agents.txt`, ALFWorld data release 2.1.1, and JDK 21 for ScienceWorld.

| Backbone | Hugging Face revision | Prompt contract |
| --- | --- | --- |
| `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | `qwen3-1.7b-native-thinking-v1` |
| `allenai/Olmo-3-7B-Think` | `d97e442d7cc678210054dbcc9b440894d62c89a4` | `olmo3-7b-think-native-v1` |

Use snapshot directories ending in the pinned revision for `--model-path`. Both contracts use temperature 0.6, top-p 0.95; Qwen uses top-k 20, OLMo disables top-k. The native chat templates differ. The model, tokenizer, sampler, feature/chunking contract, checkpoint hash, and device recorded by the run must match; retain the original guard checks.

## Mathematics trajectories and corrected natural grading

`MANIFEST` is a frozen question roster with split, identity, reference answer, and requested replicate counts; `SPLIT_SALT` must be the roster's recorded salt. The pool builders (`iclr_build_math_pool`, `iclr_build_ood_pool`, and `iclr_dataset_roster`) preserve admission, overlap screening, and roster construction code, but recreating a roster from a newly downloaded dataset does not establish identity with the archived roster.

Qwen generation uses the root snapshot. This example generates the fit split; report evaluation requires `--split report --spend-report` and frozen choices:

```bash
python3 -m rl_controller.scripts.single_cot_probe \
  --manifest "$MANIFEST" --split fit --split-salt "$SPLIT_SALT" \
  --model Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --model-path "$QWEN_MODEL" --prompt-contract qwen3-1.7b-native-thinking-v1 \
  --chains 8 --max-chains-per-question 8 --budget 12000 --block-cap 128 \
  --query-stride 4 --max-forced-tokens 32 --seed 0 --out-dir "$GEN_OUT"
```

The historical fit archive requests 8 trajectories, or 16 for a subset, then retains at most eight lowest eligible replicate indices for training. `--max-chains-per-question 8` above generates that retained subset; it is not a reconstruction of the larger raw archive. Generation writes under the requested split subdirectory. Sharded runs must retain their recorded `--work-shard-index` and `--work-shard-count`.

The corrected recipe elicits a final answer **only** when a naturally closed mathematics chain has no parseable final answer. It charges the newly generated tokens, preserves already parseable answers, and does not rescue length-capped chains:

```bash
python3 -m rl_controller.scripts.belt_natural_fallback \
  --chains "$RAW_CHAINS" --manifest "$MANIFEST" --bank-name "$BANK_NAME" \
  --model Qwen/Qwen3-1.7B --model-path "$QWEN_MODEL" \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --prompt-contract qwen3 --seed 0 --max-forced-tokens 32 --out-dir "$FALLBACK_OUT"
python3 -m rl_controller.scripts.belt_fallback_bank \
  --chains "$RAW_CHAINS" --fallback "$FALLBACK_JSONL" --out "$CORRECTED_CHAINS"
```

`FALLBACK_JSONL` is the actual `natural_fallback.jsonl` output. Existing scored dumps are corrected using `belt_fallback_dump --dump ... --fallback ... --out ...`; `--coverage subset` is appropriate only for an explicitly declared subset such as validation. Correction precedes label derivation, probe retraining, and policy fitting.

For OLMo generation, select `olmo_code` (the same generator supports mathematics) and replace the model/revision/contract above. Use `--chains 4 --max-chains-per-question 4 --budget 16384`. Natural fallback uses the root tool with `--prompt-contract olmo3` and the OLMo model pin.

## BELT probes and scoring

The trainer's four target names are `q,v,r,stability`, corresponding to **Now, End, Later, Same**. Now uses strict typed correctness at forced-answer positions; End repeats corrected terminal correctness; Later asks about strictly later forced answers or terminal correctness; Same is masked when the terminal answer cannot be parsed. Do not use the default two-target trainer configuration for the paper's full BELT probe.

The historical Qwen recipe uses `--random-init-head` after loading a saved step-head template. `INITIAL_HEAD` must be the compatible 2,048-to-768 layer-16 ReProbe/UHead directory containing its configuration and initialization weights; it is not a finished BELT checkpoint. Its bytes are recorded by the trainer. Run seeds 0, 1, and 2:

```bash
python3 -m rl_controller.scripts.single_cot_train \
  --fit-chains "$FIT_CHAINS" --select-chains "$SELECT_CHAINS" --questions "$QUESTIONS" \
  --head-checkpoint "$INITIAL_HEAD" --random-init-head \
  --model Qwen/Qwen3-1.7B --model-path "$QWEN_MODEL" \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --targets q,v,r,stability --lr-head 5e-5 --lr-readout 1e-3 \
  --epochs 8 --patience 2 --chains-per-step 8 --max-chains-per-question 8 \
  --val-count 320 --seed 0 --out-dir "$TRAIN_OUT"
```

Use the archived `--val-salt` where specified by a run receipt. The paper's main-table Qwen runs stop after three epochs and select epoch 1. Checkpoint selection uses minimum validation loss; the full validation bank is later split into 160 policy-fit and 160 development questions.

For the [recommended ReProbe-initialized Qwen H-R package](CHECKPOINTS.md#recommended-qwen-h-r-checkpoint), omit `--random-init-head` and use the supplied `reader-template/` as `INITIAL_HEAD`. Its original run configurations record `random_init_head: false`; preserve each seed's configuration and selection history when reproducing that initialization analysis. The final trained probes are `seed-{0,1,2}/s1_best.pt`. Scoring uses the same Qwen runtime and `--head-checkpoint` template as below.

OLMo's self-contained random reader is trained through the `olmo_belt` snapshot. Set `RL_PROMPT_CONTRACT=olmo3-7b-think-native-v1`; the key arguments replace the Qwen command as follows:

```bash
(
  cd "$BELT_ROOT/scripts/source_variants/olmo_belt"
  RL_PROMPT_CONTRACT=olmo3-7b-think-native-v1 PYTHONPATH="$PWD:$BELT_ROOT" \
    python3 -m rl_controller.scripts.single_cot_train \
    --fit-chains "$FIT_CHAINS" --select-chains "$SELECT_CHAINS" --questions "$QUESTIONS" \
    --model allenai/Olmo-3-7B-Think --model-path "$OLMO_MODEL" \
    --revision d97e442d7cc678210054dbcc9b440894d62c89a4 \
    --uhead-arch scalar_mix --uhead-layers 18 --uhead-dim 768 --uhead-depth 2 --uhead-heads 8 \
    --targets q,v,r,stability --target-weights q=0.25,v=0.25,r=0.25,stability=0.25 \
    --lr-head 5e-5 --lr-readout 1e-3 --epochs 3 --patience 3 \
    --chains-per-step 1 --max-chains-per-question 4 --val-count 320 \
    --seed 0 --out-dir "$TRAIN_OUT"
)
```

The saved attention mode, feature chunk size, and head batch size remain part of each numerical lane. Both trainers update the reader in evaluation mode (dropout disabled).

Qwen scoring and H-policy representation export:

```bash
python3 -m rl_controller.scripts.iclr_score_pool \
  --chains "$EVAL_CHAINS" --questions "$QUESTIONS" --source-run "$TRAIN_OUT" \
  --head-checkpoint "$INITIAL_HEAD" --model-path "$QWEN_MODEL" \
  --dump-repr-dir "$REPR_OUT" --out-dir "$SCORED_OUT"
```

The Qwen representation format is `.f16` plus `.index.jsonl`/`.source.json`. OLMo uses its `olmo_belt` scorer with `--dump-reprs "$REPR_NPY"` (NumPy matrix plus index), and needs no external head checkpoint for the self-contained reader. These formats are intentionally not interchanged.

## Policy fitting and OSP

Qwen BELT-B-R/H-R uses 17 price-refitted sigmoid-linear policies, float64 L-BFGS, and the original fit160/dev160 split. Preserve fallback token costs when invoking the historical fitting producer:

```bash
python3 scripts/belt_ext/d53/run_patched.py \
  --producer "$BELT_ROOT/scripts/source_variants/qwen_policy" \
  -m rl_controller.scripts.iclr_frozen_campaign -- \
  --arm LB_QVRS --scored-run "$SCORED_OUT" --pool "$POOL" --seed 0 --out "$FIT_JSON"
```

Use `--arm LB_H --val-dump "$VAL_REPR_DIR" --pool-dump "$POOL_REPR_DIR"` for H-R. `LB_QVRS` in this Qwen producer includes the four belief logits and observed prefix length; its name alone does not describe the OLMo feature set. The producer searches both intercept initializations, the documented regularization grid and extensions, and development utility every 25 iterations. The Qwen threshold-expansion producer (`x16_thresholds.py`) is not included.

OLMo preserves the registered 221-member family (17 fitted directions × 13 thresholds), four belief logits with **no prefix length**, and its original loose forced-answer utility labels. Test grading is strict. Its `iclr_learned_linear_campaign` accepts `prepare`, `smoke`, and `fit` stages with `--manifest "$CAMPAIGN_MANIFEST" --out "$CAMPAIGN_OUT" --budget 16384 --self-contained-reader`; the manifest must contain the actual reader-run paths and hashes. `scripts.olmo3_fit_lb_h`, `scripts.olmo3_lb_readout`, and `scripts.olmo3_lb_replay` capture H fitting and unchanged-policy transfer. Run them under the corrected-cost installer with `--producer .../olmo_policy`.

OLMo threshold fitting is captured under `scripts.x13`: `x13_fit --source-val ... --weights ... --seed 0 --feature-set QVRS --price 2e-5 --threshold-split dev --out ...`; `x13_fit_h` adds `--reprs-val` and reads H features. `x13_replay`/`x13_replay_h` write the declared policy replay rows. These scripts are byte-preserved despite retaining historical path strings; select their snapshot through the installer before execution.

The full OSP trainer is `iclr_osp_train`, **not** `single_cot_train --head-mode osp`. For the root Qwen recipe:

```bash
python3 -m rl_controller.scripts.iclr_osp_train \
  --fit-chains "$FIT_CHAINS" --select-chains "$SELECT_CHAINS" --questions "$QUESTIONS" \
  --model Qwen/Qwen3-1.7B --model-path "$QWEN_MODEL" \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --head-mode osp --osp-lambda 1e-5 --osp-cost total --osp-grade strict \
  --epochs 8 --patience 2 --chains-per-step 3 --max-seq-len 16384 \
  --val-count 320 --seed 0 --out-dir "$OSP_OUT"
```

Full OSP uses five prices `{2,5,10,20,40}×10^-6` and three seeds; OSP-1 uses `1e-5` on Qwen and the archived `2e-5` OLMo checkpoint. OLMo OSP training uses the `olmo_osp_full` snapshot, the OLMo model/contract, `--fp32-boundary-cast`, and the per-run sequence/batch settings; archived OSP-1 used seven epochs and a 20,480-token sequence limit. Do not retrain archived OSP-1 and label it the same artifact.

The archived OLMo `iclr_score_pool` top-two-block path does not install the trainer's fp32 boundary cast, while the agent scorer sets it explicitly. Score each checkpoint with the snapshot named in the [checkpoint guide](CHECKPOINTS.md) and keep the device recorded by its run.

## LiveCodeBench transfer

Both models use the same test definitions and unit-test grader. Pre has 549 problems; Post has 416. No probe, policy, or OSP checkpoint is trained on code. Generate with `single_cot_probe --answer-format code --max-forced-tokens 2048`, using the original model's budget and chain count. OLMo selects `olmo_code`, cap 16,384 and four chains; Qwen selects the root source, cap 12,000 and eight chains. Preserve the manifest's starter code or stdin/stdout instruction.

```bash
python3 -m rl_controller.scripts.iclr_grade_code \
  --shard-dir "$CODE_SHARD_DIR" --tests-dir "$TESTS_DIR" --manifest "$CODE_MANIFEST" --workers 16
```

This executes generated programs with the captured subprocess harness; run it in the isolated grading environment used for untrusted generated code. The grader verifies test hashes and writes graded chains. `scripts/olmo_lcb/olcb_bank.py` and `olcb_k1_lane.py`, within the `olmo_code` snapshot, assemble graded shards and attach natural-program token cost for policy replay. Natural code cost includes the generated program; code uses tests rather than integer parsing, so the mathematics fallback correction is not applied.

## ALFWorld and ScienceWorld

The root `scripts/olmo_alfworld` contains the production shared trajectory tree and environment adapters. Run it with a selected OLMo scorer on `PYTHONPATH`; do not let the root Qwen scorer load implicitly. The script uses the model's pinned native template, 8,192 thinking tokens per action, forced actions of at most 32 tokens, natural actions of at most 64, and the stored shared tree for replay. **Set `--max-steps 50` explicitly**; the old pilot default is 30.

```bash
PYTHONPATH="$BELT_ROOT/scripts/source_variants/olmo_belt:$BELT_ROOT" \
  OALF_ENV=alfworld python3 "$BELT_ROOT/scripts/olmo_alfworld/alf_tree.py" \
  --split eval_out_of_distribution --members "$MEMBER_SPEC" --model-path "$OLMO_MODEL" \
  --max-steps 50 --max-model-len 24576 --seeds 0 --gen-mode segments \
  --belt "$BELT_RUN_MAP" --osp "$OSP_RUN_MAP" --score-pack 1 --out "$TREE_OUT"
```

`BELT_RUN_MAP` has `0=/absolute/run0,1=/absolute/run1,2=/absolute/run2`; `OSP_RUN_MAP` has `name=/absolute/osp_run`. `MEMBER_SPEC` is a JSON list in the format documented by `alf_members.py`. It must point to the actual math-fitted parameters; it does not fit agent weights. Set `ALFWORLD_DATA` to the 2.1.1 data root. ALFWorld uses `eval_in_distribution` (seen, 140 games) for outcome-free calibration and `eval_out_of_distribution` (unseen, 134 games) for evaluation.

ScienceWorld uses the same command with `OALF_ENV=sciworld --split test`; its calibration split is `dev`. The adapter fixes JVM identity hashing and canonicalizes simulator item ordering. It uses 30 task types, first three test variations (90 games), and first three development variations (89 games), simplification `easy`, with score as the outcome.

`alf_families.py` builds candidate specifications from math parameters; `alf_calibrate.py --tree ... --families ... --targets ... --out ...` selects members by projected thinking-token savings without reading outcomes. ALFWorld targets are `0.20,0.35,0.50,0.65`; ScienceWorld adds `0.80,0.90,0.95`. Read episode frontiers using `alf_readings.py --episodes '.../lane*/episodes*.jsonl' --methods ... --out ...`; ScienceWorld adds `--outcome score --budgets 0.50,0.60,0.70,0.80`. Costs include generated thinking and actions. Bootstrap readings condition on the existing probe runs and single generation seed.

## CPU checks and remaining inputs

`python3 -m unittest discover -s tests -v` verifies every captured file's SHA256 and byte count, compiles every captured Python module, checks the strict, strictly-later and masked label semantics, invokes the generation and agent-environment parsers, and launches the code-grader subprocess harness on a synthetic program. It does not run a GPU forward pass, training, a live game or a benchmark.

Trained checkpoints and the exact Qwen initialization-head directory are downloadable through [CHECKPOINTS.md](CHECKPOINTS.md); base weights and tokenizers are separate pinned downloads. Question and trajectory banks, representation dumps, fitted policies and unit-test bundles are not included. Third-party notices are in `vendor/NOTICE.md` and `provenance/code_sources.json`.
