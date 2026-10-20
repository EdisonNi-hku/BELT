# Trained checkpoints

The release publishes **42 original trained checkpoints** across six public Hugging Face model repositories. All seeds used by the primary families and the ReProbe-initialized Qwen family are included. The exact Qwen reader template is an additional loading dependency and is not counted as a trained BELT checkpoint.

| Paper family | Hugging Face repository | Trained files | Run directories |
| --- | --- | ---: | --- |
| **Recommended Qwen H-R**: Qwen BELT, ReProbe-initialized | [BELT-Qwen3-1.7B-ReProbeInit](https://huggingface.co/JingweiNi/BELT-Qwen3-1.7B-ReProbeInit) | 3 | `seed-{0,1,2}/` |
| Qwen BELT, random-initialized (main tables) | [BELT-Qwen3-1.7B](https://huggingface.co/JingweiNi/BELT-Qwen3-1.7B) | 3 | `seed-{0,1,2}/` |
| OLMo BELT, layer-18 scalar-mix reader | [BELT-OLMo-3-7B-Think](https://huggingface.co/JingweiNi/BELT-OLMo-3-7B-Think) | 3 | `seed-{0,1,2}/` |
| Qwen full OSP | [OSP-Qwen3-1.7B](https://huggingface.co/JingweiNi/OSP-Qwen3-1.7B) | 15 | `lambda-{price}/seed-{0,1,2}/` |
| OLMo full OSP | [OSP-OLMo-3-7B-Think](https://huggingface.co/JingweiNi/OSP-OLMo-3-7B-Think) | 15 | `lambda-{price}/seed-{0,1,2}/` |
| OLMo archived OSP-1 | [OSP1-OLMo-3-7B-Think](https://huggingface.co/JingweiNi/OSP1-OLMo-3-7B-Think) | 3 | `archived-osp1/seed-{0,1,2}/` |

The full OSP grid uses training prices `2e-6`, `5e-6`, `1e-5`, `2e-5`, and `4e-5`. Qwen OSP-1 uses the full-grid `1e-5` files. OLMo OSP-1 uses the separately archived `2e-5` files, trained on the original labels with evaluation-side natural-completion corrections. It is **a different artifact** from the corrected-label full-grid `2e-5` checkpoint.

## Recommended Qwen H-R checkpoint

Use **[ReProbe-initialized BELT](https://huggingface.co/JingweiNi/BELT-Qwen3-1.7B-ReProbeInit)** for Qwen H-R. These are final four-target BELT probes trained from the layer-16 ReProbe reader with `random_init_head: false`. The package contains all three BELT training seeds, their original validation scores and scoring receipts, and the exact loading template. The three BELT seeds share one ReProbe pretraining run (seed 42).

In the paper's initialization analysis, ReProbe initialization raises H-R's out-of-distribution token saving at a one-point accuracy tolerance from 45.25% to 48.43%. H-R uses the probe's hidden representation with a refitted linear stopping policy; fit the policy parameters separately before deployment. For reproducing the main Qwen tables, use the random-initialized `BELT-Qwen3-1.7B` package.

## Revisions and file identity

[provenance/checkpoints.json](../provenance/checkpoints.json) records each Hub revision, original run path, seed, training price, selected epoch where recorded, byte count, validation-score row count, and SHA256. Each model repository also contains `CHECKPOINTS.json`, per-run `ARTIFACT.json`, and `SHA256SUMS`.

Publication checks compare anonymous Hub LFS metadata with the original file hashes for all checkpoint and validation-score files, and download the small metadata files to check their SHA256. They do not execute model inference or retrain a model.

## Download a complete checkpoint package

Use a separate download environment so that installing the Hub client does not change a recorded training environment:

```bash
python3 -m venv .venv-download
.venv-download/bin/python -m pip install huggingface_hub
.venv-download/bin/hf download JingweiNi/BELT-Qwen3-1.7B-ReProbeInit \
  --revision 1ccc94337a9672dee94f79bc3c3a98a2f994220b \
  --local-dir checkpoints/BELT-Qwen3-1.7B-ReProbeInit
(
  cd checkpoints/BELT-Qwen3-1.7B-ReProbeInit
  sha256sum -c SHA256SUMS
)
```

To download another package at the revision recorded by this release, run from the repository root:

```bash
export BELT_CHECKPOINT_REPO=JingweiNi/OSP-OLMo-3-7B-Think
BELT_CHECKPOINT_REVISION="$(python3 - <<'PY'
import json, os
with open('provenance/checkpoints.json') as f:
    manifest = json.load(f)
repo = os.environ['BELT_CHECKPOINT_REPO']
print(next(r['hub_revision'] for r in manifest['repositories']
           if r['repository'] == repo))
PY
)"
.venv-download/bin/hf download "$BELT_CHECKPOINT_REPO" \
  --revision "$BELT_CHECKPOINT_REVISION" \
  --local-dir checkpoints/OSP-OLMo-3-7B-Think
(
  cd checkpoints/OSP-OLMo-3-7B-Think
  sha256sum -c SHA256SUMS
)
```

Downloading the entire package makes its complete `SHA256SUMS` check applicable. A download filtered to one seed or price will omit files named by that manifest; verify the downloaded subset against its entries or the corresponding `ARTIFACT.json`.

## Files required by the scorer

Each run directory contains the original files:

| File | Role |
| --- | --- |
| `s1_best.pt` or `osp_best.pt` | Selected trained reader/readout or OSP top-two-block checkpoint |
| `summary.json` | Original trainer configuration, target names, selection history and provenance |
| `val_scored.jsonl` | Original validation scores; the historical scorer requires this file and copies it to the output for calibration |
| Execution/migration JSON, when present | Original distributed-execution or resume metadata |
| `scoring_receipt.json`, in the ReProbe-init package | Original scorer provenance recording the evaluated checkpoint SHA256 |
| `ARTIFACT.json` | Publication mapping, hashes and byte counts |

Pass the **run directory**, rather than the `.pt` file, as `--source-run`. Preserve `val_scored.jsonl`; an empty replacement would discard the recorded calibration data.

| Family | Matching released runtime | Additional reader requirement |
| --- | --- | --- |
| Qwen BELT | Root `rl_controller` scorer | `--head-checkpoint` must point to the downloaded `reader-template/` directory |
| Qwen OSP | Root `rl_controller` scorer | No external BELT head/template |
| OLMo BELT | `scripts/source_variants/olmo_belt/` | Saved self-contained scalar-mix reader |
| OLMo full OSP | `scripts/source_variants/olmo_osp_full/` | Run configuration reconstructs the top-two-block reader |
| OLMo archived OSP-1 | `scripts/source_variants/olmo_osp_archived/` | Archived scorer; keep its recorded numerical contract |

Follow [source selection and scoring commands](RECIPES.md). Snapshot commands must run from the selected snapshot directory with that directory first on `PYTHONPATH`. The checkpoint package supplies source-run files; it does not supply evaluation trajectories or questions.

## Base models and remaining inputs

| Backbone | Required revision |
| --- | --- |
| `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| `allenai/Olmo-3-7B-Think` | `d97e442d7cc678210054dbcc9b440894d62c89a4` |

Backbone weights and tokenizers, trajectory and question banks, fitted stopping policies and environment data are separate inputs; see the [asset list](ARTIFACTS.md). BELT's reader/readout checkpoints are not fitted budget-specific stopping policies.

Original model files are preserved without conversion or retraining. No new project license is assigned; see [NOTICE.md](../NOTICE.md).
