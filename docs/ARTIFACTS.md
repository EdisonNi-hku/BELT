# Asset list

| Material | Where | Notes |
| --- | --- | --- |
| Qwen and OLMo generation, belief/OSP training, scoring, grading and policy-fitting source | This repository | Root runtime plus revision snapshots in `scripts/source_variants/`; see [RECIPES.md](RECIPES.md) |
| ALFWorld and ScienceWorld tree, environments, calibration and readings | `scripts/olmo_alfworld/` | Needs `requirements/agents.txt`, ALFWorld data 2.1.1 and JDK 21 for ScienceWorld |
| Native-thinking UHead/ReProbe runtime | `vendor/uncertainty4reasoning-native-thinking/` | Upstream snapshot; provenance in `vendor/uncertainty4reasoning-native-thinking.SOURCE.md` |
| Source revisions and file hashes | `provenance/code_sources.json` | Original repository, revision and path of every captured file |
| Trained BELT and OSP checkpoints, 42 in total | Hugging Face | Seed/price layout, revisions and hashes in [CHECKPOINTS.md](CHECKPOINTS.md) and `provenance/checkpoints.json` |
| Exact Qwen reader template | Included in both Qwen BELT packages | `reader-template/` holds the initialization configuration and weights |
| Backbone weights and tokenizers | Hugging Face | `Qwen/Qwen3-1.7B` and `allenai/Olmo-3-7B-Think` at the revisions pinned in [RECIPES.md](RECIPES.md) |
| Question manifests, trajectory banks, fitted stopping policies, unit-test bundles | Not included | Rebuild them with the recipes |
| Qwen threshold-expansion producer `x16_thresholds.py` | Not included | |
