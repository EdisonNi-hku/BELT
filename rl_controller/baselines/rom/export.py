from __future__ import annotations

import json
from pathlib import Path

from rl_controller.iclr import scoring_io as sio

METHOD = "rom"
KNOB = "tau"
FAMILY = "ROM_tau"

TAU_GRID = (0.02,) + tuple(round(0.05 * i, 2) for i in range(1, 20)) + (0.98,)


CANONICAL_POOLS = {
    "math_union_fit": "math_union",
    "math_union_select": "math_union",
    "math_ood_report": "math_ood_report",
    "code_fit": "code_pool",
    "code_select": "code_pool",
    "code_report": "code_pool",
    "code_ood_pooled": "code_ood_pooled",
    "gpqa_report": "gpqa_diamond_report",
}


def canonical_pool(name: str) -> str:
    return CANONICAL_POOLS.get(str(name), str(name))


def family_spec() -> dict:
    from rl_controller.iclr.ladder import FamilySpec

    spec = FamilySpec(
        name=FAMILY, grid=TAU_GRID, kind="threshold", belief_key="q", direction="ge",
        params=("tau",), formats=("{:.2f}",),
        belief_source="ROM (arXiv 2603.22016) post-FCS streaming detector; "
                      "sigmoid of the block-granularity StreamingHead logit")
    return spec.to_dict()


def write_family_spec(path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([family_spec()], indent=2) + "\n", encoding="utf-8")
    return str(path)


def score_chains(head, chains, cache, spec, *, config, batch_size: int = 32,
                 drop_never_correct: bool = False, device=None, dtype=None):
    import torch

    from rl_controller.baselines.rom import data as rom_data

    device = device or config.device
    dtype = dtype or getattr(torch, config.dtype)
    examples, skipped = rom_data.build_examples(
        chains, cache, spec, dt_mode=config.dt_mode,
        drop_never_correct=drop_never_correct, require_trainable=False)
    by_ident = {(e.qkey, e.rep): e for e in examples}

    head.eval()
    with torch.no_grad():
        order = sorted(range(len(examples)), key=lambda i: examples[i].n_blocks)
        for start in range(0, len(order), batch_size):
            chunk = [examples[i] for i in order[start:start + batch_size]]
            batch = rom_data.collate(chunk, device=device, dtype=dtype)
            logits = head(batch["x"], batch["lengths"], batch["dt"],
                          prompt=batch["prompt"], prompt_mask=batch["prompt_mask"],
                          m0_grad=False)
            gathered = rom_data.gather_probe_logits(logits, batch["gather"])
            probs = head.probabilities(gathered).cpu()
            raw = gathered.detach().cpu()
            for row, example in enumerate(chunk):
                k = len(example.probe_pos)
                example.meta["scores"] = [
                    {"q": float(probs[row, j]), "v": 0.0,
                     "rom_logit": float(raw[row, j, 0] if head.num_labels == 1
                                        else raw[row, j, 1] - raw[row, j, 0])}
                    for j in range(k)]

    unscored, probeless = [], 0
    for chain in chains:
        if not chain["probes"]:
            chain["scores"] = []
            probeless += 1
            continue
        example = by_ident.get((chain["qkey"], int(chain["rep"])))
        if example is None or "scores" not in example.meta:
            unscored.append({"qkey": chain["qkey"], "rep": chain["rep"]})
            continue
        chain["scores"] = example.meta["scores"]
    skipped["unscored"] = unscored
    skipped["probeless_natural_rows"] = probeless
    return skipped


def provenance(*, cache_manifest, chains_sha256: str, manifest_sha256: str, commit: str,
               config, spec, balance: dict, m0_init: str, extra: dict | None = None) -> dict:
    out = {
        "pool": canonical_pool(cache_manifest.pool),
        "chains_sha256": chains_sha256,
        "manifest_sha256": manifest_sha256,
        "commit": commit,
        "model_id": cache_manifest.model_id,
        "model_revision": cache_manifest.model_revision,
        "answer_format": "boxed",
        "force_natural_answer": False,
        "overhead_accounting": "none",
        "overhead_tokens_per_boundary": 0,
        "rom_upstream": "github.com/SaFo-Lab/ROM@c2e651e (MIT)",
        "rom_layer": spec.layer,
        "rom_pooling": spec.pooling,
        "rom_granularity": "block",
        "rom_dt_mode": config.dt_mode,
        "rom_m0_init": m0_init,
        "rom_m0_grad": bool(config.m0_grad),
        "rom_num_labels": config.num_labels,
        "rom_backtrace": "no-op at block granularity (D5)",
        "rom_stop_answer": "stored forced-answer probe, not ROM's regenerated cue (D5)",
        "rom_class_balance": balance,
        "rom_tau_grid": list(TAU_GRID),
        "rom_params": None,
        **(cache_manifest.provenance()),
    }
    if extra:
        out.update(extra)
    return out


def write_run_dir(out_dir, *, val_chains, select_chains, seed: int, provenance_row: dict,
                  extra_config: dict | None = None) -> dict:
    summary = sio.summary_stub(
        seed=seed, targets=["q", "v"], method=METHOD, provenance=provenance_row,
        knob=KNOB, family=FAMILY, **(extra_config or {}))
    return sio.write_run_dir(out_dir, val=val_chains, select=select_chains, summary=summary)
