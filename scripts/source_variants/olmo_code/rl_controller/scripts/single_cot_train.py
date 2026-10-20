"""Train the belief probe on per-block targets."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist

from rl_controller.iclr import labels as lb
from rl_controller.iclr import ladder as ld
from rl_controller.iclr import learning_curve as lc
from rl_controller.iclr import osp_policy as osp
from rl_controller.iclr import targets as tg
from rl_controller.iclr.chains import (draw_count, is_correct, is_trainable, probe_draws,
                                       write_jsonl)
from rl_controller.iclr.learning_curve import bce
from rl_controller.single_cot import (
    average_precision,
    oracle_report,
    probe_positions_in_ends,
    score_frontier,
)

MARGINS = (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2)
THETAS = (0.5, 0.6, 0.7, 0.8, 0.9)
DEPTH_BINS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))
DEFAULT_TARGETS = "q,v"
Q_LABEL_MODES = ("single", "multi_draw_mean")
FIT_FOLD_SALT = "iclr-fitfold-2026-08-27"
HEAD_MODES = ("targets", "osp", "osp_cond", "osp_aux", "osp_bottleneck", "gap",
              "targets_probes")
CONDITIONED_MODES = ("osp_cond", "osp_aux", "osp_bottleneck")
AUX_MODES = ("osp_aux", "osp_bottleneck")
OSP_V_CONST = 0.5
DEFAULT_LAMBDA_GRID = (2e-6, 5e-6, 1e-5, 2e-5, 4e-5)
OSP_REFERENCE_LAMBDA = 1e-5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit-chains", required=True, nargs="+",
                   help="one or more chains.jsonl (e.g. the 16 spec-B fit shards); "
                        "(qkey, rep) must be unique across files")
    p.add_argument("--select-chains", required=True)
    p.add_argument("--questions", required=True, nargs="+",
                   help="one or more jsonl files with a `question` field covering every "
                        "chain (S0 questions.jsonl, the pool-scaleup manifest, ...): "
                        "prompts are REBUILT with the same build_prompt_ids + "
                        "rewrite_user_instruction the probe script used (deterministic), "
                        "since chains.jsonl stores completion ids only")
    p.add_argument("--head-checkpoint", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--model-path", default=None)
    p.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--hs-chunk-size", type=int, default=8192)
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--lr-head", type=float, default=5e-5)
    p.add_argument("--lr-readout", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--resume", action="store_true",
                   help="continue from `s1_last.pt` in --out-dir if it is there, else start "
                        "fresh; makes a wall-clock kill cost one epoch instead of the run "
                        "(W1d). The checkpoint is written after every epoch and carries the "
                        "optimiser, the selection state and both RNG streams, so a resumed "
                        "run sees the same shuffles as an uninterrupted one.")
    p.add_argument("--chains-per-step", type=int, default=8)
    p.add_argument("--holdout-fraction", type=float, default=0.2,
                   help="legacy grouped holdout by salted hash; ignored when --val-count is set")
    p.add_argument("--val-count", type=int, default=0,
                   help="exact number of fit questions held out as val (learning_curve.partition); "
                        "0 = use --holdout-fraction")
    p.add_argument("--val-salt", default=None,
                   help="salt for --val-count (default learning_curve.VAL_SALT)")
    p.add_argument("--max-train-questions", type=int, default=0,
                   help="train on the n lowest-ranked train questions (nested prefix); 0 = all")
    p.add_argument("--max-chains-per-question", type=int, default=0,
                   help="keep the K lowest reps per question for train AND val chains; 0 = all")
    p.add_argument("--select-by", choices=("logloss", "q_ap"), default="logloss",
                   help="early-stopping statistic on val: joint q+v log-loss (spec §3b) or q AP")
    p.add_argument("--targets", default=DEFAULT_TARGETS,
                   help="comma-separated readout targets from iclr.targets.CATALOGUE; must "
                        "start with q,v (default: exactly q,v = today's two heads)")
    p.add_argument("--target-weights", default="",
                   help="per-target loss weights `name=w,...` (default 1.0 each; 1.0 skips "
                        "the multiply so the default run's loss terms are unchanged)")
    p.add_argument("--pi0-fire", default=None,
                   help="pi0 fire table jsonl (scripts.iclr_pi0_fire) required by the *_pi0 / "
                        "dy / dc targets; every fit and select chain with probes needs a row")
    p.add_argument("--fit-folds", type=int, default=0,
                   help="K-fold cross-fit scoring: hold fold --fit-fold of the TRAIN questions "
                        "out of training and dump it scored (fit_fold_scored.jsonl); 0 = off")
    p.add_argument("--fit-fold", type=int, default=-1,
                   help="which fold (0..K-1) to hold out under --fit-folds")
    p.add_argument("--fit-fold-salt", default=FIT_FOLD_SALT,
                   help="salt for learning_curve.question_folds under --fit-folds")
    p.add_argument("--q-label", choices=Q_LABEL_MODES, default="single",
                   help="q training label: the stored probe's single forced-answer draw, or the "
                        "mean over the stored multi-draw re-probes (A1); eval/selection always "
                        "use the single-draw label")
    p.add_argument("--q-draws", type=int, default=0,
                   help="draws per probe used for the soft q label (multi_draw_mean training "
                        "label and the q_soft eval terms); 0 = all stored draws")
    p.add_argument("--store-token-budget", type=int, default=8_000_000,
                   help="clear the GPU feature store once it holds more than this many tokens "
                        "-- after an optimizer step AND after each 16-chain group inside a "
                        "val/select pass, so no pass can exceed it (bf16 ≈ 4 KB/token on "
                        "Qwen3-1.7B; 8 M ≈ 32 GB for 80-96 GB GPUs, ~1.5 M ≈ 6 GB for 24 GB)")
    p.add_argument("--max-chain-tokens", type=int, default=0,
                   help="escape hatch: drop chains whose forwarded completion length "
                        "(block_ends[-1]) exceeds this from train/val/select at load time, "
                        "counted in config[dropped_over_max_chain_tokens]; 0 = no limit. "
                        "The val/train partition is fixed BEFORE this filter (a question "
                        "whose chains are all dropped keeps its side; count in "
                        "config[questions_emptied_by_max_chain_tokens]). Dropping select "
                        "chains makes select_rules / gate_s1 NON-comparable to an "
                        "unfiltered run.")
    p.add_argument("--head-mode", choices=HEAD_MODES, default="targets",
                   help="`targets` (default) = the belief readout selected by --targets; "
                        "`osp` = OS-Pruner's direct stop policy (one sigmoid unit trained on "
                        "the exact-expectation objective, no BCE, one head per --osp-lambda)")
    p.add_argument("--osp-lambda", type=float, default=1e-4,
                   help="the FINAL token price of the OSP objective; the curve knob "
                        f"(pre-registered grid {osp.DEFAULT_LAMBDAS})")
    p.add_argument("--osp-lambda-anneal", default="",
                   help="progressive lambda annealing `geometric:<start_mult>:<epochs>` or "
                        f"`constant`; default = {osp.DEFAULT_ANNEAL.as_dict()}")
    p.add_argument("--osp-token-scale", type=float, default=osp.DEFAULT_TOKEN_SCALE,
                   help="divide every token cost by this before the objective (1.0 = raw "
                        "thinking tokens, the pre-registered convention)")
    p.add_argument("--osp-cost", choices=osp.COST_AXES, default="thinking",
                   help="cost axis the OSP objective is charged in; `thinking` (default) is "
                        "the axis every head in this line was trained under, `total` adds the "
                        "forced answer at a probe and the archived-math natural answer at the "
                        "natural end (the campaign primary axis)")
    p.add_argument("--osp-init-fire-rate", type=float, default=0.1,
                   help="stop unit's bias seed: pi at init, logit-transformed (0.1 = about one "
                        "stop per ten boundaries, well away from the always-continue collapse)")
    p.add_argument("--lambda-grid", default="",
                   help="conditioned modes: comma-separated TRAINING prices the one policy is "
                        "trained across (default = the campaign grid "
                        "2e-6,5e-6,1e-5,2e-5,4e-5). Inference always reads the frozen 13")
    p.add_argument("--dump-reprs", action="store_true",
                   help="write <out-dir>/reprs_<split>.npy: the trunk representation at every "
                        "block end, BEFORE the lambda feature is concatenated (the W3 probe input)")
    p.add_argument("--osp-grade", choices=osp.GRADINGS, default="strict",
                   help="probe-correctness convention for the TRAINING objective; `strict` is "
                        "the type-exact rule the q target uses. The REPLAY grades `loose` "
                        "(ladder.stop_outcome) so its points are comparable with every hull")
    p.add_argument("--rom-drop-never-correct", action="store_true",
                   help="ROM ablation (campaign WS-B item 3): mask the post-FCS target `o` on "
                        "chains with no correct probe anywhere; the pre-registered PRIMARY rule "
                        "(flag off) labels every boundary of such a chain 0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--freeze-head", action="store_true",
                   help="ablation: train only the readout on frozen-head reprs — "
                        "the A2 condition, to isolate what unfreezing buys")
    p.add_argument("--random-init-head", action="store_true",
                   help="C2 init ablation: re-initialise the UHead from scratch instead of the "
                        "--head-checkpoint weights (still unfrozen unless --freeze-head); "
                        "refused together with --freeze-head, which would train a readout on "
                        "random features")
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def prepare_chains(rows):
    for r in rows:
        r["_ends"] = [int(e) for e in r.get("block_ends", [])]
        if r["probes"]:
            if [qp[1] for qp in r["query_points"]] != [p["prefix_len"] for p in r["probes"]]:
                raise SystemExit(f"chain {(r.get('qkey'), r.get('rep'))}: query_points prefixes "
                                 "do not align with probes[].prefix_len")
        trainable = is_trainable(r)
        r["_probe_pos"] = (probe_positions_in_ends(r["_ends"], r["query_points"])
                           if trainable else [])
        if len(set(r["_probe_pos"])) != len(r["_probe_pos"]):
            raise SystemExit(f"chain {(r.get('qkey'), r.get('rep'))}: two probes share a block "
                             f"end (probe positions {r['_probe_pos']}) — one boundary cannot "
                             "carry two probe labels")
        r["_trainable"] = trainable
        r["scores"] = []
    return rows


def load_chains(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rows, seen = [], set()
    for path in paths:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            ident = (r["qkey"], r["rep"])
            if ident in seen:
                raise ValueError(f"duplicate chain {ident} in {path}")
            seen.add(ident)
            rows.append(r)
    return prepare_chains(rows)


def q_labels(chain, *, mode: str = "single", draws: int = 0):
    if mode not in Q_LABEL_MODES:
        raise ValueError(f"q label mode {mode!r} not in {Q_LABEL_MODES}")
    if draws < 0:
        raise ValueError(f"draws must be >= 0, got {draws}")
    gold = chain["gold"]
    if mode == "single":
        return [float(is_correct(p["forced"], gold)) for p in chain["probes"]]
    out = []
    for p in chain["probes"]:
        ds = probe_draws(p)
        if draws > 0:
            ds = ds[:draws]
        out.append(sum(is_correct(d["forced"], gold) for d in ds) / len(ds))
    return out


def check_draws(chains, *, mode: str, draws: int) -> None:
    if mode != "multi_draw_mean":
        return
    if draws == 1:
        raise SystemExit("--q-label multi_draw_mean --q-draws 1 would train on draw 0 only; "
                         "use --q-label single")
    need = max(2, draws)
    for c in chains:
        if not c.get("_trainable", True):
            continue
        k = draw_count(c)
        if k < need:
            raise SystemExit(f"--q-label multi_draw_mean --q-draws {draws}: chain "
                             f"{(c['qkey'], c['rep'])} has {k} draw(s) per probe, need >= {need} "
                             "(re-probe the pool with scripts.single_cot_probe --probe-draws)")


def draws_below(chains, draws: int) -> int:
    if draws <= 0:
        return 0
    return sum(1 for c in chains if c.get("_trainable", True) and 0 < draw_count(c) < draws)


def attach_pi0_fire(chains, table, path) -> None:
    for c in chains:
        ident = (c["qkey"], c["rep"])
        fire = table.get(ident)
        if fire is None:
            if not c["probes"]:
                c["_pi0_fire"] = []
                continue
            raise SystemExit(f"--pi0-fire {path}: no fire row for chain {ident}")
        if len(fire) != len(c["probes"]):
            raise SystemExit(f"--pi0-fire {path}: chain {ident} has {len(c['probes'])} probes "
                             f"but {len(fire)} fire entries")
        c["_pi0_fire"] = list(fire)


def attach_sibling_majority(chains, ctx) -> None:
    for c in chains:
        c["_sibling_majority"] = ctx[(c["qkey"], c["rep"])]["sibling_majority"]
        c["_sibling_pass_rate"] = ctx[(c["qkey"], c["rep"])]["sibling_pass_rate"]


def attach_rom_flag(chains, drop_never_correct: bool) -> None:
    for c in chains:
        c[tg.ROM_DROP_FLAG] = bool(drop_never_correct)


def random_init_head(head, seed: int) -> int:
    torch.manual_seed(seed)
    n = 0
    for module in head.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()
            n += 1
    return n


def attach_side_tables(used_fit, select, *, fire=None, fire_path=None, fit_ctx=None,
                       sel_ctx=None, rom_drop=None) -> None:
    if fire is not None:
        attach_pi0_fire(used_fit, fire, fire_path)
        attach_pi0_fire(select, fire, fire_path)
    if fit_ctx is not None:
        attach_sibling_majority(used_fit, fit_ctx)
    if sel_ctx is not None:
        attach_sibling_majority(select, sel_ctx)
    if rom_drop is not None:
        attach_rom_flag(used_fit, rom_drop)
        attach_rom_flag(select, rom_drop)


def attach_labels(chains, names, *, q_mode: str = "single", q_draws: int = 0):
    for c in chains:
        if not c["_trainable"]:
            c["_labels"], c["_q_single"], c["_q_soft"] = {}, [], None
            continue
        labs = {}
        for name in names:
            labs[name] = (q_labels(c, mode=q_mode, draws=q_draws) if name == "q"
                          else tg.labels(name, c))
        c["_labels"] = labs
        c["_q_single"] = labs["q"] if q_mode == "single" else q_labels(c, mode="single")
        c["_q_soft"] = (q_labels(c, mode="multi_draw_mean", draws=q_draws)
                        if draw_count(c) > 1 else None)
    return chains


def _ensure_labels(chain, names):
    if "_labels" not in chain:
        attach_labels([chain], names)


def fold_slice(train, train_keys, *, n_folds: int, fold: int, salt: str):
    if n_folds < 2:
        raise SystemExit(f"--fit-folds must be >= 2, got {n_folds}")
    if not 0 <= fold < n_folds:
        raise SystemExit(f"--fit-fold must be in 0..{n_folds - 1}, got {fold}")
    fm = lc.question_folds(sorted(train_keys), n_folds=n_folds, salt=salt)
    heldout = [c for c in train if fm[c["qkey"]] == fold]
    kept = [c for c in train if fm[c["qkey"]] != fold]
    keys = {k for k in train_keys if fm[k] != fold}
    if not kept or not heldout:
        raise SystemExit(f"--fit-folds {n_folds} --fit-fold {fold}: train {len(kept)} / "
                         f"fold heldout {len(heldout)} chains -- need both")
    return kept, heldout, keys


def bias_init_values(train, names):
    out = {}
    for name in names:
        spec = tg.CATALOGUE[name]
        k = spec.n_out if spec.kind == "softmax" else 1
        tot = [0.0] * k
        n = 0
        for c in train:
            lab = [y for y in c["_labels"][name] if y is not None]
            if not lab:
                continue
            if spec.positions == "probes":
                if spec.kind == "softmax":
                    for y in lab:
                        tot[y] += 1.0
                else:
                    tot[0] += sum(lab)
                n += len(lab)
            else:
                if spec.kind == "softmax":
                    for y in lab:
                        tot[y] += 1.0 / len(lab)
                else:
                    tot[0] += sum(lab) / len(lab)
                n += 1
        out[name] = [t / max(1, n) for t in tot]
    return out


def chain_losses(logits, c, names, weights, cols, device):
    total = 0.0
    parts = {}
    qp = torch.tensor(c["_probe_pos"], dtype=torch.long, device=device)
    for name in names:
        spec = tg.CATALOGUE[name]
        lab = c["_labels"][name]
        keep = [i for i, y in enumerate(lab) if y is not None]
        if not keep:
            continue
        rows = qp if spec.positions == "probes" else torch.arange(logits.shape[0], device=device)
        if len(keep) != len(lab):
            rows = rows[torch.tensor(keep, dtype=torch.long, device=device)]
        out = logits[rows, cols[name]]
        if spec.kind == "bce":
            y = torch.tensor([lab[i] for i in keep], device=device)
            l = torch.nn.functional.binary_cross_entropy_with_logits(out[:, 0], y)
        elif spec.kind == "huber":
            y = torch.tensor([lab[i] for i in keep], device=device)
            l = torch.nn.functional.smooth_l1_loss(out[:, 0], y)
        elif spec.kind == "mse":
            y = torch.tensor([lab[i] for i in keep], device=device)
            l = torch.nn.functional.mse_loss(torch.tanh(out[:, 0]), y)
        else:
            y = torch.tensor([lab[i] for i in keep], dtype=torch.long, device=device)
            l = torch.nn.functional.cross_entropy(out, y)
        w = weights[name]
        total = total + (l if w == 1.0 else w * l)
        parts[name] = (float(l), len(keep))
    return total, parts


def attach_osp_terms(chains, *, grade: str, token_scale: float,
                     cost: str = "thinking") -> None:
    for c in chains:
        c["_osp_terms"] = (osp.chain_terms(c, grade=grade, token_scale=token_scale, cost=cost)
                           if c["_trainable"] else None)


def osp_probe_logits(readout, reprs, chain, device):
    out = readout(reprs)
    idx = torch.tensor(chain["_probe_pos"], dtype=torch.long, device=device)
    return out[idx, 0]


def osp_chain_record(chain, pis, lam: float) -> dict:
    terms = chain["_osp_terms"]
    e_correct, e_tokens = osp.expected_outcome(pis, terms)
    return {
        "qkey": chain["qkey"], "rep": chain["rep"],
        "scores": [{"q": pi, "v": OSP_V_CONST, "pi": pi} for pi in pis],
        "pis": list(pis), "n_probes": len(pis),
        "q_labels": [float(y) for y in chain["_q_single"]],
        "e_correct": e_correct, "e_tokens": e_tokens,
        "objective": e_correct - lam * e_tokens,
        "p_natural": osp.survival(pis)[1],
    }


def osp_metrics(records, lam: float) -> dict:
    n = max(1, len(records))
    ys, ss = [], []
    for r in records:
        ys.extend(r["q_labels"])
        ss.extend(r["pis"])
    objective = sum(r["objective"] for r in records) / n
    n_probes = sum(r["n_probes"] for r in records)
    return {
        "n_chains": len(records), "n_probes": n_probes,
        "osp_lambda": lam,
        "osp_loss": -objective, "joint_logloss": -objective, "objective": objective,
        "e_accuracy": sum(r["e_correct"] for r in records) / n,
        "e_thinking_tokens": sum(r["e_tokens"] for r in records) / n,
        "mean_pi": (sum(sum(r["pis"]) for r in records) / n_probes) if n_probes else 0.0,
        "p_natural": sum(r["p_natural"] for r in records) / n,
        "q_ap": average_precision(ys, ss) if ys else 0.0,
        "q_base_rate": (sum(ys) / len(ys)) if ys else 0.0,
    }


def osp_eval_split(state_fn, readout, chains, device, *, lam: float,
                   store_token_budget: int | None = None, world: int = 1, rank: int = 0):
    for c in chains:
        c["scores"] = []
    mine = chains[rank::world] if world > 1 else chains
    records = []
    todo = [c for c in mine if c["_trainable"]]
    for i in range(0, len(todo), 16):
        grp = todo[i: i + 16]
        outs = _forward_with_recovery(state_fn, grp, grad=False)
        for c, o in zip(grp, outs):
            with torch.no_grad():
                z = osp_probe_logits(readout, o["reprs"].float(), c, device)
                pis = torch.sigmoid(z).cpu().tolist()
            records.append(osp_chain_record(c, pis, lam))
        if store_token_budget is not None \
                and state_fn.store_stats["tokens_stored"] > store_token_budget:
            state_fn.clear_store()
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, records)
        if rank != 0:
            return None
        records = sorted((r for part in gathered for r in part),
                         key=lambda r: (r["qkey"], r["rep"]))
    by_id = {(c["qkey"], c["rep"]): c for c in chains}
    for r in records:
        by_id[(r["qkey"], r["rep"])]["scores"] = r["scores"]
    return osp_metrics(records, lam)


class LambdaPolicy(torch.nn.Module):

    def __init__(self, in_dim: int, width: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(in_dim, width), torch.nn.SiLU(),
                                       torch.nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x)[..., 0]


def policy_logits(policy, feats, lam_feats):
    k = lam_feats.shape[0]
    rows = feats.unsqueeze(0).expand(k, -1, -1)
    price = lam_feats.view(k, 1, 1).expand(k, feats.shape[0], 1)
    return policy(torch.cat([rows, price], dim=-1))


def cond_chain_loss(policy, feats, terms, lambdas, lam_feats, mult: float = 1.0):
    charged = [lam * mult for lam in lambdas]
    if mult != 1.0:
        lam_feats = torch.as_tensor(osp.lambda_grid_features(charged),
                                    dtype=lam_feats.dtype, device=lam_feats.device)
    z = policy_logits(policy, feats, lam_feats)
    a, c, a_nat, c_nat = osp.utility_terms(terms)
    price = torch.as_tensor(charged, dtype=z.dtype, device=z.device)
    a_t = torch.as_tensor(a, dtype=z.dtype, device=z.device)
    c_t = torch.as_tensor(c, dtype=z.dtype, device=z.device)
    return -osp.expected_utility(z, a_t, c_t, a_nat, c_nat, price).mean()


def cond_chain_record(chain, pis_by_lambda, n_inference, train_lambdas, ref_index,
                     terms, beliefs=None) -> dict:
    grid = pis_by_lambda[:n_inference]
    train = pis_by_lambda[n_inference:]
    ref = pis_by_lambda[ref_index]
    utils = {}
    for lam, pis in zip(train_lambdas, train):
        ec, et = osp.expected_outcome(pis, terms)
        utils[f"{lam:g}"] = ec - lam * et
    rows = [{"q": ref[i], "v": OSP_V_CONST, "pi": ref[i],
             "pi_by_lambda": [col[i] for col in grid]} for i in range(len(ref))]
    for name, values in (beliefs or {}).items():
        for i, row in enumerate(rows):
            row[name] = values[i]
    return {
        "qkey": chain["qkey"], "rep": chain["rep"],
        "scores": rows,
        "pis": list(ref), "n_probes": len(ref),
        "q_labels": [float(y) for y in chain["_q_single"]],
        "utility_by_lambda": utils,
        "mean_utility": sum(utils.values()) / max(1, len(utils)),
        "p_natural": osp.survival(ref)[1],
    }


def cond_metrics(records, lambdas) -> dict:
    n = max(1, len(records))
    ys, ss = [], []
    for r in records:
        ys.extend(r["q_labels"]); ss.extend(r["pis"])
    mean_u = sum(r["mean_utility"] for r in records) / n
    per_lam = {k: sum(r["utility_by_lambda"][k] for r in records) / n
               for k in (records[0]["utility_by_lambda"] if records else {})}
    n_probes = sum(r["n_probes"] for r in records)
    return {"n_chains": len(records), "n_probes": n_probes,
            "lambda_grid_train": list(lambdas),
            "inference_lambdas": list(osp.INFERENCE_LAMBDAS),
            "mean_utility": mean_u, "utility_by_lambda": per_lam,
            "joint_logloss": -mean_u, "objective": mean_u,
            "mean_pi": (sum(sum(r["pis"]) for r in records) / n_probes) if n_probes else 0.0,
            "p_natural": sum(r["p_natural"] for r in records) / n,
            "q_ap": average_precision(ys, ss) if ys else 0.0,
            "q_base_rate": (sum(ys) / len(ys)) if ys else 0.0}


def gap_chain_record(chain, d) -> dict:
    y_now = [float(y) for y in chain["_q_single"]]
    y_final = 1.0 if chain["correct"] else 0.0
    target = [y - y_final for y in y_now]
    return {"qkey": chain["qkey"], "rep": chain["rep"],
            "scores": [{"q": v, "v": 0.0, "d": v} for v in d],
            "d": list(d), "n_probes": len(d),
            "q_labels": y_now, "gap_labels": target,
            "sq_err": [(a - b) ** 2 for a, b in zip(d, target)]}


def gap_metrics(records) -> dict:
    errs = [e for r in records for e in r["sq_err"]]
    ys, ss = [], []
    for r in records:
        ys.extend(r["q_labels"]); ss.extend(r["d"])
    mse = sum(errs) / max(1, len(errs))
    return {"n_chains": len(records), "n_probes": len(errs),
            "gap_mse": mse, "joint_logloss": mse,
            "mean_d": (sum(ss) / len(ss)) if ss else 0.0,
            "mean_gap_label": (sum(l for r in records for l in r["gap_labels"])
                               / max(1, len(errs))),
            "q_ap": average_precision(ys, ss) if ys else 0.0,
            "q_base_rate": (sum(ys) / len(ys)) if ys else 0.0}


def mode_eval_split(state_fn, chains, device, *, record_fn, metrics_fn,
                    store_token_budget=None, world: int = 1, rank: int = 0):
    for c in chains:
        c["scores"] = []
    mine = chains[rank::world] if world > 1 else chains
    records = []
    todo = [c for c in mine if c["_trainable"]]
    for i in range(0, len(todo), 16):
        grp = todo[i: i + 16]
        outs = _forward_with_recovery(state_fn, grp, grad=False)
        for c, o in zip(grp, outs):
            with torch.no_grad():
                records.append(record_fn(c, o["reprs"].float()))
        if store_token_budget is not None \
                and state_fn.store_stats["tokens_stored"] > store_token_budget:
            state_fn.clear_store()
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, records)
        if rank != 0:
            return None
        records = sorted((r for part in gathered for r in part),
                         key=lambda r: (r["qkey"], r["rep"]))
    by_id = {(c["qkey"], c["rep"]): c for c in chains}
    for r in records:
        by_id[(r["qkey"], r["rep"])]["scores"] = r["scores"]
    return metrics_fn(records)


def dump_reprs_npy(state_fn, chains, path, *, store_token_budget=None):
    import numpy as np

    rows, index, n_written = [], [], 0
    todo = sorted((c for c in chains if c["_trainable"]), key=lambda c: (c["qkey"], c["rep"]))
    for i in range(0, len(todo), 16):
        grp = todo[i: i + 16]
        outs = _forward_with_recovery(state_fn, grp, grad=False)
        for c, o in zip(grp, outs):
            with torch.no_grad():
                r = o["reprs"].float().cpu().numpy().astype("float16")
            index.append({"qkey": c["qkey"], "rep": c["rep"], "row": n_written,
                          "n_ends": int(r.shape[0]), "probe_pos": list(c["_probe_pos"])})
            n_written += int(r.shape[0])
            rows.append(r)
        if store_token_budget is not None \
                and state_fn.store_stats["tokens_stored"] > store_token_budget:
            state_fn.clear_store()
    if not rows:
        raise SystemExit(f"no trainable chains to dump reprs for ({path})")
    np.save(path, np.concatenate(rows, axis=0))
    write_jsonl(str(path).replace(".npy", ".index.jsonl"), index)
    return {"path": str(path), "n_chains": len(index), "n_rows": sum(r.shape[0] for r in rows),
            "dim": int(rows[0].shape[1]), "dtype": "float16"}


def _huber(x: float, y: float) -> float:
    d = abs(x - y)
    return 0.5 * d * d if d < 1.0 else d - 0.5


def _nll(probs, y: int, eps: float = 1e-7) -> float:
    return -math.log(max(probs[int(y)], eps))


def eval_term(kind: str, value, y) -> float:
    if kind == "bce":
        return bce(value, y)
    if kind == "huber":
        return _huber(value, y)
    if kind == "mse":
        return (value - y) ** 2
    return _nll(value, y)


def _score_keys(names, vals):
    return tuple(names) + (("v",) if "v" not in names and "v" in vals else ())


def build_eval_record(c, vals, names):
    labels = c["_q_single"]
    y_v = 1.0 if c["correct"] else 0.0
    q_terms = [bce(vals["q"][p], y) for p, y in zip(c["_probe_pos"], labels)]
    v_terms = [bce(vals["v"][j], y_v) for j in range(len(vals["v"]))]
    t_loss = {"q": [sum(q_terms), len(q_terms)], "v": [sum(v_terms), len(v_terms)]}
    t_labels = {}
    for name in names:
        if name in ("q", "v"):
            continue
        spec = tg.CATALOGUE[name]
        lab = c["_labels"][name]
        rows = c["_probe_pos"] if spec.positions == "probes" else range(len(vals["v"]))
        terms = [eval_term(spec.kind, vals[name][row], y) for row, y in zip(rows, lab)
                 if y is not None]
        t_loss[name] = [sum(terms), len(terms)]
        if spec.positions == "probes" and spec.kind == "bce":
            t_labels[name] = list(lab)
    q_soft = c.get("_q_soft")
    return {
        "qkey": c["qkey"], "rep": c["rep"],
        "scores": [{name: vals[name][p] for name in _score_keys(names, vals)}
                   for p in c["_probe_pos"]],
        "q_labels": labels, "q_terms": q_terms,
        "v_terms_sum": sum(v_terms), "v_terms_mean": sum(v_terms) / len(v_terms),
        "n_ends": len(v_terms), "y_v": y_v,
        "v_last": vals["v"][c["_probe_pos"][-1]],
        "depths": [c["_ends"][p] / max(1, c["thinking_tokens"]) for p in c["_probe_pos"]],
        "t_loss": t_loss, "t_labels": t_labels,
        "q_soft": q_soft,
        "q_terms_soft": (None if q_soft is None
                         else [bce(vals["q"][p], y) for p, y in zip(c["_probe_pos"], q_soft)]),
    }


def _forward_chains(state_fn, chains, *, grad: bool):
    items = [(c["_prompt_ids"], c["completion_ids"][: c["_ends"][-1]], c["_ends"])
             for c in chains]
    if grad:
        return state_fn.forward_many(items)
    with torch.no_grad():
        return state_fn.forward_many(items)


_OOM_MARKERS = ("provider skipped a prefix", "out of memory")


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and any(m in str(exc) for m in _OOM_MARKERS)


def _chain_len(c) -> int:
    return len(c["_prompt_ids"]) + c["_ends"][-1]


def _recover(state_fn, chains, epoch):
    state_fn.clear_store()
    torch.cuda.empty_cache()
    print(json.dumps({"oom_recovery": {"epoch": epoch, "n_chains": len(chains),
                                       "max_len": max(_chain_len(c) for c in chains),
                                       "rank": int(os.environ.get("RANK", "0"))}}),
          flush=True)


def _forward_with_recovery(state_fn, chains, *, grad: bool, epoch=None):
    failed = None
    for attempt in (0, 1):
        if attempt == 1:
            _recover(state_fn, chains, epoch)
        try:
            return _forward_chains(state_fn, chains, grad=grad)
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            failed = f"{type(e).__name__}: {e}"
    if len(chains) == 1:
        c = chains[0]
        raise RuntimeError(
            "provider skipped this chain twice (OOM or a feature-extraction error — see "
            "the provider's 'Skipping sample' warning above): "
            f"chain {(c['qkey'], c['rep'])}, prompt {len(c['_prompt_ids'])} + completion "
            f"{c['_ends'][-1]} = {_chain_len(c)} tokens; use --max-chain-tokens or a "
            f"smaller --hs-chunk-size (last error: {failed})"
        )
    outs = []
    for c in chains:
        outs.extend(list(_forward_with_recovery(state_fn, [c], grad=grad, epoch=epoch)))
    return outs


def readout_values(out, names):
    cols = tg.column_slices(names)
    vals = {}
    for name in names:
        spec = tg.CATALOGUE[name]
        block = out[:, cols[name]]
        if spec.kind in ("bce", "mse"):
            vals[name] = torch.sigmoid(block[:, 0]).cpu().tolist()
        elif spec.kind == "huber":
            vals[name] = block[:, 0].cpu().tolist()
        else:
            vals[name] = torch.softmax(block, dim=1).cpu().tolist()
    for alias in ("v_probes", "v_sib"):
        if "v" not in vals and alias in vals:
            vals["v"] = vals[alias]
    return vals


SCORED_KEEP = ("qkey", "rep", "gold", "outcome", "answer", "correct", "blocks",
               "thinking_tokens", "block_ends", "query_points", "probes", "scores")


def dump_scored(chains, path):
    for c in chains:
        if not c["_trainable"] and c["probes"]:
            raise SystemExit(f"chain {(c['qkey'], c['rep'])} has probes but is not "
                             "trainable — cannot dump consistently")
        missing = [k for k in SCORED_KEEP if k not in c]
        if missing:
            raise SystemExit(f"chain {(c['qkey'], c['rep'])} lacks {missing} — cannot dump")
    write_jsonl(path, ({k: c[k] for k in SCORED_KEEP} for c in chains))


def build_prompt_table(qrows, tokenizer) -> dict:
    from rl_controller.rollout_contract import ContractError, answer_formats, prompt_ids_for
    from rl_controller.splits import question_key

    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    decode = lambda ids: tokenizer.decode(ids)
    try:
        formats = answer_formats(qrows)
    except ContractError as exc:
        raise SystemExit(f"--questions: {exc}") from None
    table = {}
    for row in qrows:
        key = question_key(str(row["question"]))
        if key not in table:
            table[key] = prompt_ids_for(tokenizer, str(row["question"]), answer_format=formats[key],
                                        encode=encode, decode=decode)
    return table


def chain_eval_records(state_fn, readout, chains, *, store_token_budget: int | None = None,
                       names=tg.MANDATORY):
    records = []
    todo = [c for c in chains if c["_trainable"]]
    for i in range(0, len(todo), 16):
        grp = todo[i: i + 16]
        outs = _forward_with_recovery(state_fn, grp, grad=False)
        for c, o in zip(grp, outs):
            _ensure_labels(c, names)
            with torch.no_grad():
                vals = readout_values(readout(o["reprs"].float()), names)
            records.append(build_eval_record(c, vals, names))
        if store_token_budget is not None \
                and state_fn.store_stats["tokens_stored"] > store_token_budget:
            state_fn.clear_store()
    return records


def chain_joint(record, names, weights, q_override=None):
    total = 0.0
    for name in names:
        s, n = record["t_loss"][name] if not (name == "q" and q_override) else q_override
        if n == 0:
            continue
        part = s / n
        w = weights[name]
        total = total + (part if w == 1.0 else w * part)
    return total


def metrics_from_records(records, *, names=tg.MANDATORY, weights=None):
    names = tuple(names)
    weights = weights or {n: 1.0 for n in names}
    ys, ss, depths, v_ys, v_ss = [], [], [], [], []
    q_bce_sum = v_bce_sum = joint_sum = 0.0
    n_q = n_v = 0
    t_sum = {n: 0.0 for n in names}
    t_chains = {n: 0 for n in names}
    t_ys = {n: [] for n in names}
    t_ss = {n: [] for n in names}
    soft = all(r.get("q_terms_soft") is not None for r in records)
    q_soft_sum = joint_soft_sum = 0.0
    n_q_soft = 0
    for r in records:
        ys.extend(r["q_labels"])
        ss.extend(sc["q"] for sc in r["scores"])
        depths.extend(r["depths"])
        v_ys.append(r["y_v"]); v_ss.append(r["v_last"])
        q_bce_sum += sum(r["q_terms"]); n_q += len(r["q_terms"])
        v_bce_sum += r["v_terms_sum"]; n_v += r["n_ends"]
        joint_sum += chain_joint(r, names, weights)
        for name in names:
            s, n = r["t_loss"][name]
            if n:
                t_sum[name] += s / n
                t_chains[name] += 1
            if name in r["t_labels"]:
                for y, sc in zip(r["t_labels"][name], r["scores"]):
                    if y is not None:
                        t_ys[name].append(y); t_ss[name].append(sc[name])
        if soft:
            q_soft_sum += sum(r["q_terms_soft"]); n_q_soft += len(r["q_terms_soft"])
            joint_soft_sum += chain_joint(r, names, weights,
                                          q_override=(sum(r["q_terms_soft"]), len(r["q_terms_soft"])))
    m = {"q_ap": average_precision(ys, ss), "q_base_rate": sum(ys) / max(1, len(ys)),
         "v_ap": average_precision(v_ys, v_ss), "n_probes": len(ys), "n_chains": len(records),
         "q_bce": q_bce_sum / max(1, n_q), "v_bce_ends": v_bce_sum / max(1, n_v),
         "joint_logloss": joint_sum / max(1, len(records))}
    for lo, hi in DEPTH_BINS:
        sel = [(y, sc) for y, sc, d in zip(ys, ss, depths) if lo <= d < hi]
        if len(sel) >= 20:
            m[f"q_ap_depth_{lo:.1f}_{hi:.1f}"] = average_precision(*zip(*sel))
            m[f"q_base_depth_{lo:.1f}_{hi:.1f}"] = sum(y for y, _ in sel) / len(sel)
    m["targets"] = list(names)
    m["target_loss"] = {name: (t_sum[name] / t_chains[name] if t_chains[name] else None)
                        for name in names}
    for name in names:
        if t_ys[name]:
            m[f"{name}_ap"] = average_precision(t_ys[name], t_ss[name])
    if soft and records:
        m["q_bce_soft"] = q_soft_sum / max(1, n_q_soft)
        m["joint_logloss_soft"] = joint_soft_sum / len(records)
    return m


def eval_split(state_fn, readout, chains, device, *, store_token_budget: int | None = None,
               world: int = 1, rank: int = 0, names=tg.MANDATORY, weights=None):
    for c in chains:
        c["scores"] = []
    mine = chains[rank::world] if world > 1 else chains
    records = chain_eval_records(state_fn, readout, mine, store_token_budget=store_token_budget,
                                 names=names)
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, records)
        if rank != 0:
            return None
        records = sorted((r for part in gathered for r in part),
                         key=lambda r: (r["qkey"], r["rep"]))
    by_id = {(c["qkey"], c["rep"]): c for c in chains}
    for r in records:
        by_id[(r["qkey"], r["rep"])]["scores"] = r["scores"]
    return metrics_from_records(records, names=names, weights=weights)


holdout_metrics = eval_split


def collect_scores(state_fn, readout, chains, device):
    eval_split(state_fn, readout, chains, device)


def _all_reduce_grads(params, world: int):
    by_dtype: dict = {}
    for p_ in params:
        if p_.grad is not None:
            by_dtype.setdefault(p_.grad.dtype, []).append(p_.grad)
    for grads in by_dtype.values():
        flat = torch.cat([g.reshape(-1) for g in grads])
        dist.all_reduce(flat)
        flat.div_(world)
        for g, synced in zip(grads, torch.split(flat, [g.numel() for g in grads])):
            g.copy_(synced.view_as(g))


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(f"cuda:{args.gpu}")
    main_rank = rank == 0

    def log(obj):
        if main_rank:
            print(json.dumps(obj), flush=True)

    torch.manual_seed(args.seed)

    from rl_controller.scripts.rl_train import HeadStateFn, _load_head_runtime
    tokenizer, provider, head, frozen, _hidden = _load_head_runtime(args, device)
    state_fn = HeadStateFn(provider=provider, head=head, frozen=frozen,
                           device=device, raw2048=False)
    state_fn.incremental = False
    state_fn.always_frozen = False
    n_reset = None
    if args.random_init_head:
        if args.freeze_head:
            raise SystemExit("--random-init-head with --freeze-head would train a readout on "
                             "frozen RANDOM features — not one of the pre-registered cells")
        n_reset = random_init_head(head, args.seed)
        if not n_reset:
            raise SystemExit("--random-init-head reset 0 submodules: this head exposes no "
                             "reset_parameters(), so the pretrained checkpoint would silently "
                             "have been kept")
        log({"random_init_head": {"submodules_reset": n_reset, "seed": args.seed}})
    head.eval()

    fit = load_chains(args.fit_chains)
    select = load_chains(args.select_chains)
    from rl_controller.iclr.provenance import source_record
    try:
        names = tg.parse_targets(args.targets)
        weights = tg.parse_weights(args.target_weights, names)
    except ValueError as exc:
        raise SystemExit(f"--targets/--target-weights: {exc}") from None
    specs = [tg.CATALOGUE[n] for n in names]
    osp_mode = args.head_mode == "osp"
    cond_mode = args.head_mode in CONDITIONED_MODES
    aux_mode = args.head_mode in AUX_MODES
    gap_mode = args.head_mode == "gap"
    probes_mode = args.head_mode == "targets_probes"
    if probes_mode:
        names = ("q", "v_probes")
        weights = {n: 1.0 for n in names}
        specs = [tg.CATALOGUE[n] for n in names]
    elif gap_mode:
        names = ("q", "gap")
        weights = {"q": 0.0, "gap": 1.0}
        specs = [tg.CATALOGUE[n] for n in names]
    lambdas = ()
    lam_feats = None
    if cond_mode:
        grid = args.lambda_grid or ",".join(f"{l:g}" for l in DEFAULT_LAMBDA_GRID)
        try:
            lambdas = osp.parse_lambdas(grid)
        except ValueError as exc:
            raise SystemExit(f"--lambda-grid: {exc}") from None
        if aux_mode:
            if tuple(tg.parse_targets(args.targets)[:2]) != tuple(DEFAULT_TARGETS.split(",")):
                raise SystemExit(
                    f"--head-mode {args.head_mode} reads columns 0 and 1 as q and v, so "
                    f"--targets must START with {DEFAULT_TARGETS!r}; extra targets after "
                    f"that join the auxiliary loss only (got {args.targets!r})")
        elif args.targets != DEFAULT_TARGETS:
            raise SystemExit(f"--head-mode {args.head_mode} trains one stop policy on the "
                             f"q,v readout: --targets must stay {DEFAULT_TARGETS!r}")
    if (gap_mode or probes_mode) and args.targets != DEFAULT_TARGETS:
        raise SystemExit(f"--head-mode {args.head_mode} fixes its own target set: "
                         f"--targets must stay {DEFAULT_TARGETS!r} (got {args.targets!r})")
    if osp_mode or cond_mode:
        if args.targets != DEFAULT_TARGETS and not aux_mode:
            raise SystemExit(f"--head-mode {args.head_mode} trains a stop unit, not a free "
                             f"belief readout: --targets must stay {DEFAULT_TARGETS!r} "
                             f"(got {args.targets!r})")
        try:
            osp_sched = osp.parse_schedule(args.osp_lambda_anneal)
        except ValueError as exc:
            raise SystemExit(f"--osp-lambda-anneal: {exc}") from None
        if args.osp_lambda < 0:
            raise SystemExit(f"--osp-lambda must be >= 0, got {args.osp_lambda}")
        if not 0.0 < args.osp_init_fire_rate < 1.0:
            raise SystemExit(f"--osp-init-fire-rate must be in (0, 1), got "
                             f"{args.osp_init_fire_rate}")
        if not args.osp_token_scale > 0:
            raise SystemExit(f"--osp-token-scale must be > 0, got {args.osp_token_scale}")
        need = osp_sched.anneal_epochs + args.patience + 1
        if args.epochs < need:
            raise SystemExit(f"--head-mode osp with {osp_sched.as_dict()} and --patience "
                             f"{args.patience} needs --epochs >= {need}, got {args.epochs}")
    if args.q_draws < 0:
        raise SystemExit(f"--q-draws must be >= 0, got {args.q_draws}")
    pi0_prov = fire = fit_ctx = sel_ctx = None
    if any(s.needs_pi0 for s in specs):
        if not args.pi0_fire:
            raise SystemExit(f"targets {[s.name for s in specs if s.needs_pi0]} need --pi0-fire")
        try:
            fire = ld.load_fire_table(args.pi0_fire)
        except ValueError as exc:
            raise SystemExit(f"--pi0-fire: {exc}") from None
        pi0_prov = source_record(args.pi0_fire)
    if any(s.needs_siblings for s in specs):
        fit_ctx = lb.sibling_context(fit)
        sel_ctx = lb.sibling_context(select)
    pool_keys = {c["qkey"] for c in fit if c["_trainable"]}
    dropped = {"fit": 0, "select": 0}
    emptied = 0
    if args.max_chain_tokens > 0:
        def within(c):
            return not c["_ends"] or c["_ends"][-1] <= args.max_chain_tokens
        n_fit, n_sel = len(fit), len(select)
        fit = [c for c in fit if within(c)]
        select = [c for c in select if within(c)]
        dropped = {"fit": n_fit - len(fit), "select": n_sel - len(select)}
        emptied = len(pool_keys - {c["qkey"] for c in fit if c["_trainable"]})
        if any(dropped.values()):
            log({"warning": "dropped_over_max_chain_tokens",
                 "max_chain_tokens": args.max_chain_tokens, **dropped,
                 "questions_emptied": emptied})
    if not fit or not select:
        raise SystemExit(f"fit {len(fit)} / select {len(select)} usable chains — need both")

    qrows = [json.loads(l) for path in args.questions
             for l in open(path, encoding="utf-8") if l.strip()]
    prompts = build_prompt_table(qrows, tokenizer)
    for c in fit + select:
        if c["qkey"] not in prompts:
            raise SystemExit(f"chain question {c['qkey']} not in --questions")
        c["_prompt_ids"] = prompts[c["qkey"]]

    trainable = [c for c in fit if c["_trainable"]]
    if args.max_chains_per_question:
        trainable = lc.cap_chains(trainable, args.max_chains_per_question)
    if args.val_count:
        val_keys, train_order = lc.partition(pool_keys, n_val=args.val_count,
                                             salt=args.val_salt or lc.VAL_SALT)
        train_keys = set(lc.nested_subset(train_order, args.max_train_questions))
        val_keys = set(val_keys)
    else:
        if args.max_train_questions:
            raise SystemExit("--max-train-questions needs --val-count")

        def is_holdout(qkey):
            h = int(hashlib.blake2b(f"s1-holdout:{qkey}".encode(), digest_size=8).hexdigest(), 16)
            return (h % 1000) / 1000.0 < args.holdout_fraction
        val_keys = {k for k in pool_keys if is_holdout(k)}
        train_keys = pool_keys - val_keys
    train = [c for c in trainable if c["qkey"] in train_keys]
    hold = [c for c in trainable if c["qkey"] in val_keys]
    heldout = []
    if args.fit_folds or args.fit_fold >= 0:
        if not args.fit_folds:
            raise SystemExit("--fit-fold needs --fit-folds")
        train, heldout, train_keys = fold_slice(train, train_keys, n_folds=args.fit_folds,
                                                fold=args.fit_fold, salt=args.fit_fold_salt)
    if not train or not hold:
        raise SystemExit(f"train {len(train)} / val {len(hold)} trainable chains — need both")
    check_draws(train, mode=args.q_label, draws=args.q_draws)
    used_fit = train + hold + heldout
    attach_side_tables(used_fit, select, fire=fire, fire_path=args.pi0_fire,
                       fit_ctx=fit_ctx, sel_ctx=sel_ctx,
                       rom_drop=(bool(args.rom_drop_never_correct) if "o" in names else None))
    attach_labels(used_fit + select, names, q_mode=args.q_label, q_draws=args.q_draws)
    if osp_mode or cond_mode:
        attach_osp_terms(used_fit + select, grade=args.osp_grade,
                         token_scale=args.osp_token_scale, cost=args.osp_cost)
    if main_rank and args.q_draws > 0:
        short = {"val": draws_below(hold, args.q_draws), "select": draws_below(select, args.q_draws)}
        if any(short.values()):
            print(json.dumps({"warning": "chains_with_fewer_draws_than_q_draws",
                              "q_draws": args.q_draws, **short}), file=sys.stderr, flush=True)
    if world > 1 and args.chains_per_step < world:
        raise SystemExit("--chains-per-step must be ≥ WORLD_SIZE (effective batch = "
                         "world × (chains_per_step // world))")
    per_rank = max(1, args.chains_per_step // world)
    stride = per_rank * world
    if world > 1 and len(train) < stride:
        raise SystemExit(f"{len(train)} train chains < world_size * chains_per_rank = {stride}: "
                         "nothing to train under DDP (lower --chains-per-step or the GPU count)")
    gpu_props = torch.cuda.get_device_properties(device)
    config = {"n_train_questions": len(train_keys), "n_train_chains": len(train),
              "n_val_questions": len(val_keys), "n_val_chains": len(hold),
              "n_select_chains": len(select),
              "max_chains_per_question": args.max_chains_per_question,
              "max_train_questions": args.max_train_questions, "val_count": args.val_count,
              "val_salt": args.val_salt or lc.VAL_SALT, "seed": args.seed,
              "select_by": args.select_by, "patience": args.patience,
              "freeze_head": bool(args.freeze_head), "lr_head": args.lr_head,
              "lr_readout": args.lr_readout, "store_token_budget": args.store_token_budget,
              "model": args.model, "model_path": args.model_path, "revision": args.revision,
              "dtype": args.dtype, "epochs": args.epochs,
              "chains_per_step": args.chains_per_step, "hs_chunk_size": args.hs_chunk_size,
              "world_size": world, "chains_per_step_per_rank": per_rank,
              "effective_chains_per_step": stride,
              "gpu": gpu_props.name, "gpu_mem_gb": round(gpu_props.total_memory / 2 ** 30, 1),
              "max_chain_tokens": args.max_chain_tokens,
              "dropped_over_max_chain_tokens": dropped,
              "questions_emptied_by_max_chain_tokens": emptied,
              "fit_chains": [str(p) for p in args.fit_chains], "select_chains": args.select_chains,
              "targets": list(names), "target_weights": weights, "pi0_fire": pi0_prov,
              "fit_folds": args.fit_folds, "fit_fold": args.fit_fold,
              "fit_fold_salt": args.fit_fold_salt,
              "n_fold_heldout_questions": len({c["qkey"] for c in heldout}),
              "n_fold_heldout_chains": len(heldout),
              "q_label": args.q_label, "q_draws": args.q_draws,
              "rom_drop_never_correct": bool(args.rom_drop_never_correct),
              "random_init_head": bool(args.random_init_head),
              "head_submodules_reset": n_reset,
              "head_mode": args.head_mode,
              "lambda_grid": list(lambdas),
              "osp_cost": args.osp_cost,
              "v_mask": ("probes" if probes_mode else "ends"),
              "v_label": ("sibling_pass_rate" if "v_sib" in names else "own_chain"),
              "aux_belief_names": ([f"{n}_aux" for n in tg.MANDATORY] if aux_mode else None),
              "osp": (None if not (osp_mode or cond_mode) else
                      {"lambda": args.osp_lambda, "anneal": osp_sched.as_dict(),
                       "token_scale": args.osp_token_scale, "grade": args.osp_grade,
                       "init_fire_rate": args.osp_init_fire_rate, "cost": args.osp_cost,
                       "v_const": OSP_V_CONST, "replay_grade": "loose"})}
    log({"config": config})
    chain_counts = {"train": len(train), "holdout": len(hold), "select": len(select)}
    if heldout:
        chain_counts["fold_heldout"] = len(heldout)
    log({"chains": chain_counts,
         "questions": {"train": len(train_keys), "holdout": len(val_keys)},
         "freeze_head": bool(args.freeze_head)})

    with torch.no_grad():
        probe_dim = _forward_with_recovery(state_fn, train[:1], grad=False)[0]["reprs"].shape[1]
    cols = tg.column_slices(names)
    policy = None
    if osp_mode:
        readout = torch.nn.Linear(probe_dim, 1).to(device)
        bias = torch.logit(torch.tensor([args.osp_init_fire_rate]), eps=1e-6)
    elif cond_mode:
        readout = torch.nn.Linear(probe_dim, tg.readout_width(names) if aux_mode else 1).to(device)
        if aux_mode:
            seeds = bias_init_values(train, names)
            bias = torch.zeros(tg.readout_width(names))
            for name in names:
                bias[cols[name]] = torch.logit(
                    torch.tensor([max(1e-3, seeds[name][0])]), eps=1e-6)
        else:
            bias = torch.zeros(1)
        policy = LambdaPolicy(3 if args.head_mode == "osp_bottleneck" else probe_dim + 1).to(device)
    elif gap_mode:
        readout = torch.nn.Linear(probe_dim, 1).to(device)
        bias = torch.zeros(1)
    else:
        readout = torch.nn.Linear(probe_dim, tg.readout_width(names)).to(device)
        seeds = bias_init_values(train, names)
        bias = torch.zeros(tg.readout_width(names))
        for name in names:
            kind = tg.CATALOGUE[name].kind
            if kind == "bce":
                bias[cols[name]] = torch.logit(torch.tensor([max(1e-3, seeds[name][0])]), eps=1e-6)
            elif kind == "huber":
                bias[cols[name]] = torch.tensor(seeds[name])
            else:
                bias[cols[name]] = torch.log(torch.tensor(seeds[name]) + 1e-3)
    with torch.no_grad():
        readout.bias.copy_(bias.to(device))
    if world > 1:
        with torch.no_grad():
            for p_ in readout.parameters():
                dist.broadcast(p_.data, src=0)
    if policy is not None:
        if world > 1:
            with torch.no_grad():
                for p_ in policy.parameters():
                    dist.broadcast(p_.data, src=0)
    params = [{"params": list(readout.parameters()), "lr": args.lr_readout}]
    if policy is not None:
        params.append({"params": list(policy.parameters()), "lr": args.lr_readout})
    if not args.freeze_head:
        params.append({"params": list(head.parameters()), "lr": args.lr_head})
    opt = torch.optim.AdamW(params)
    clip_params = [p_ for g in params for p_ in g["params"]]

    infer_lams = list(osp.INFERENCE_LAMBDAS) + list(lambdas)
    n_inference = len(osp.INFERENCE_LAMBDAS)
    if cond_mode:
        if OSP_REFERENCE_LAMBDA not in lambdas:
            raise SystemExit(f"--lambda-grid must contain the reference price "
                             f"{OSP_REFERENCE_LAMBDA:g} (the scored dump's `q` column); "
                             f"got {list(lambdas)}")
        ref_index = n_inference + list(lambdas).index(OSP_REFERENCE_LAMBDA)
        train_feats = torch.tensor(osp.lambda_grid_features(lambdas),
                                   dtype=torch.float32, device=device)
        infer_feats = torch.tensor(osp.lambda_grid_features(infer_lams),
                                   dtype=torch.float32, device=device)

    def policy_features(reprs, c):
        qp = torch.tensor(c["_probe_pos"], dtype=torch.long, device=device)
        if args.head_mode != "osp_bottleneck":
            return reprs[qp]
        beliefs = torch.sigmoid(readout(reprs)[qp][:, :2])
        return beliefs

    def aux_beliefs(reprs, c):
        if not aux_mode:
            return None
        qp = torch.tensor(c["_probe_pos"], dtype=torch.long, device=device)
        out = readout(reprs)[qp]
        return {f"{n}_aux": torch.sigmoid(out[:, cols[n]][:, 0]).cpu().tolist()
                for n in tg.MANDATORY}

    def cond_record(c, reprs):
        z = policy_logits(policy, policy_features(reprs, c), infer_feats)
        pis = torch.sigmoid(z).cpu().tolist()
        return cond_chain_record(c, pis, n_inference, lambdas, ref_index, c["_osp_terms"],
                                 aux_beliefs(reprs, c))

    def cond_metric(records):
        return cond_metrics(records, lambdas)

    def gap_record(c, reprs):
        qp = torch.tensor(c["_probe_pos"], dtype=torch.long, device=device)
        d = torch.tanh(readout(reprs)[qp][:, 0]).cpu().tolist()
        return gap_chain_record(c, d)

    def _eval(split):
        if osp_mode:
            return osp_eval_split(state_fn, readout, split, device, lam=args.osp_lambda,
                                  store_token_budget=args.store_token_budget,
                                  world=world, rank=rank)
        if cond_mode:
            return mode_eval_split(state_fn, split, device,
                                   record_fn=cond_record, metrics_fn=cond_metric,
                                   store_token_budget=args.store_token_budget,
                                   world=world, rank=rank)
        if gap_mode:
            return mode_eval_split(state_fn, split, device,
                                   record_fn=gap_record, metrics_fn=gap_metrics,
                                   store_token_budget=args.store_token_budget,
                                   world=world, rank=rank)
        return eval_split(state_fn, readout, split, device,
                          store_token_budget=args.store_token_budget, world=world, rank=rank,
                          names=names, weights=weights)

    rng = random.Random(args.seed)
    best = {"epoch": -1, "stat": float("inf") if args.select_by == "logloss" else -1.0}
    history = []
    val_tokens = sum(c["_ends"][-1] for c in hold[rank::world])

    last_path = out_dir / "s1_last.pt"
    ident = {"targets": list(names), "head_mode": args.head_mode, "seed": args.seed,
             "epochs": args.epochs, "select_by": args.select_by, "patience": args.patience,
             "freeze_head": bool(args.freeze_head), "osp_cost": args.osp_cost,
             "osp_lambda": (args.osp_lambda if osp_mode else None),
             "lambda_grid": [float(x) for x in lambdas], "n_train": len(train)}
    start_epoch = 0
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck.get("identity") != ident:
            raise SystemExit(json.dumps({"resume_refused": "identity mismatch",
                                         "checkpoint": ck.get("identity"), "run": ident}))
        head.load_state_dict(ck["head"])
        readout.load_state_dict(ck["readout"])
        if policy is not None:
            policy.load_state_dict(ck["policy"])
        opt.load_state_dict(ck["opt"])
        best, history, start_epoch = ck["best"], ck["history"], ck["epoch"] + 1
        rng.setstate(tuple(tuple(x) if isinstance(x, list) else x for x in ck["py_rng"]))
        torch.set_rng_state(ck["torch_rng"].cpu().to(torch.uint8))
        log({"resumed": {"from_epoch": ck["epoch"], "next_epoch": start_epoch,
                         "best_epoch": best.get("epoch")}})
        if start_epoch >= args.epochs:
            log({"resume_note": "every epoch was already done; going straight to select"})
    elif args.resume:
        log({"resume": "no s1_last.pt; starting from epoch 0"})

    for epoch in range(start_epoch, args.epochs):
        rng.shuffle(train)
        t0 = time.time()
        tot = {name: 0.0 for name in names}
        cnt = {name: 0.0 for name in names}
        lam_epoch = osp_sched.at(epoch, args.osp_lambda) if osp_mode else None
        price_mult = osp_sched.at(epoch, 1.0) if cond_mode else 1.0
        osp_tot, osp_n = 0.0, 0.0
        if world > 1:
            n_steps = len(train) // stride
            if epoch == 0 and len(train) % stride:
                log({"ddp_tail_dropped_per_epoch": len(train) % stride,
                     "world_size": world, "chains_per_step_per_rank": per_rank})
            groups = [train[(s * world + rank) * per_rank: (s * world + rank + 1) * per_rank]
                      for s in range(n_steps)]
        else:
            groups = [train[i: i + args.chains_per_step]
                      for i in range(0, len(train), args.chains_per_step)]
        for grp in groups:
            outs = _forward_with_recovery(state_fn, grp, grad=not args.freeze_head,
                                          epoch=epoch)
            loss = 0.0
            for c, o in zip(grp, outs):
                reprs = o["reprs"].float()
                if args.freeze_head:
                    reprs = reprs.detach()
                if osp_mode:
                    z = osp_probe_logits(readout, reprs, c, device)
                    chain_loss = osp.osp_chain_loss(z, c["_osp_terms"], lam_epoch, device=device)
                    osp_tot += float(chain_loss); osp_n += 1
                    loss = loss + chain_loss
                    continue
                if cond_mode:
                    chain_loss = cond_chain_loss(policy, policy_features(reprs, c),
                                                 c["_osp_terms"], lambdas, train_feats,
                                                 mult=price_mult)
                    if aux_mode:
                        bce_loss, parts = chain_losses(readout(reprs), c, names, weights,
                                                       cols, device)
                        chain_loss = chain_loss + bce_loss
                        for name, (l, n) in parts.items():
                            tot[name] += l * n; cnt[name] += n
                    osp_tot += float(chain_loss); osp_n += 1
                    loss = loss + chain_loss
                    continue
                if gap_mode:
                    qp = torch.tensor(c["_probe_pos"], dtype=torch.long, device=device)
                    d = torch.tanh(readout(reprs)[qp][:, 0])
                    y = torch.tensor(c["_labels"]["gap"], dtype=d.dtype, device=device)
                    chain_loss = torch.nn.functional.mse_loss(d, y)
                    tot["gap"] += float(chain_loss) * len(y); cnt["gap"] += len(y)
                    loss = loss + chain_loss
                    continue
                logits = readout(reprs)
                chain_loss, parts = chain_losses(logits, c, names, weights, cols, device)
                loss = loss + chain_loss
                for name, (l, n) in parts.items():
                    tot[name] += l * n; cnt[name] += n
            loss = loss / len(grp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if world > 1:
                _all_reduce_grads(clip_params, world)
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            opt.step()
            if state_fn.store_stats["tokens_stored"] > args.store_token_budget:
                state_fn.clear_store()
        if world > 1 and osp_mode:
            agg = torch.tensor([osp_tot, osp_n], dtype=torch.float64, device=device)
            dist.all_reduce(agg)
            osp_tot, osp_n = agg.tolist()
        elif world > 1:
            agg = torch.tensor([tot[n] for n in names] + [cnt[n] for n in names],
                               dtype=torch.float64, device=device)
            dist.all_reduce(agg)
            vals = agg.tolist()
            tot = dict(zip(names, vals[:len(names)]))
            cnt = dict(zip(names, vals[len(names):]))
        if state_fn.store_stats["tokens_stored"] + val_tokens > args.store_token_budget:
            state_fn.clear_store()
        m = _eval(hold)
        if state_fn.store_stats["tokens_stored"] > args.store_token_budget:
            state_fn.clear_store()
        better = stop = fail = False
        if main_rank:
            fail = not math.isfinite(m["joint_logloss"]) or not math.isfinite(m["q_ap"])
        if main_rank and not fail:
            val_rules = score_frontier(hold, margins=MARGINS, thetas=())
            if osp_mode:
                train_losses = {"train_osp_loss": osp_tot / max(1.0, osp_n),
                                "osp_lambda_epoch": lam_epoch,
                                "osp_lambda_final": args.osp_lambda}
            elif cond_mode:
                train_losses = {"train_cond_loss": osp_tot / max(1.0, osp_n),
                                "price_multiplier": price_mult,
                                "lambda_grid": list(lambdas)}
                if aux_mode:
                    for name in names:
                        train_losses[f"train_{name}_loss"] = tot[name] / max(1, cnt[name])
            elif gap_mode:
                train_losses = {"train_gap_mse": tot["gap"] / max(1, cnt["gap"])}
            else:
                v_key = next(v for v in tg.V_SLOT if v in names)
                train_losses = {"train_q_bce": tot["q"] / max(1, cnt["q"]),
                                "train_v_bce": tot[v_key] / max(1, cnt[v_key])}
                for name in names:
                    if name in ("q", v_key):
                        continue
                    train_losses[f"train_{name}_loss"] = tot[name] / max(1, cnt[name])
            m.update({"epoch": epoch, **train_losses,
                      "seconds": round(time.time() - t0, 1),
                      "store_tokens": state_fn.store_stats["tokens_stored"],
                      "val_rules": [{k: r[k] for k in ("name", "accuracy", "mean_thinking_tokens",
                                                       "thinking_saving", "fired_fraction")}
                                    for r in val_rules]})
            history.append(m)
            log({k: v for k, v in m.items() if k != "val_rules"})
            stat = m["joint_logloss"] if args.select_by == "logloss" else m["q_ap"]
            better = (stat < best["stat"]) if args.select_by == "logloss" else (stat > best["stat"])
            floor = osp_sched.anneal_epochs if (osp_mode or cond_mode) else 0
            if epoch < floor:
                better = False
            if better:
                best = {"epoch": epoch, "stat": stat, "select_by": args.select_by,
                        "q_ap": m["q_ap"], "joint_logloss": m["joint_logloss"]}
                torch.save({"head": head.state_dict(), "readout": readout.state_dict(),
                            **({"policy": policy.state_dict()} if policy is not None else {}),
                            "epoch": epoch, "metrics": {k: v for k, v in m.items() if k != "val_rules"},
                            "freeze_head": bool(args.freeze_head), "targets": list(names),
                            "head_mode": args.head_mode,
                            "lambda_grid": list(lambdas),
                            "osp_cost": args.osp_cost,
                            "osp_lambda": (args.osp_lambda if osp_mode else None)},
                           out_dir / "s1_best.pt")
            ref = best["epoch"] if best["epoch"] >= floor else floor - 1
            stop = epoch >= floor and (epoch - ref) >= args.patience
            if stop:
                log({"early_stop": epoch, "best": best})
            tmp = out_dir / "s1_last.pt.partial"
            torch.save({"head": head.state_dict(), "readout": readout.state_dict(),
                        **({"policy": policy.state_dict()} if policy is not None else {}),
                        "opt": opt.state_dict(), "epoch": epoch, "best": best,
                        "history": history, "identity": ident,
                        "py_rng": rng.getstate(), "torch_rng": torch.get_rng_state().cpu()},
                       tmp)
            os.replace(tmp, last_path)
        if world > 1:
            flags = [better, stop, fail]
            dist.broadcast_object_list(flags, src=0)
            better, stop, fail = flags
            if better and not main_rank:
                best = {"epoch": epoch}
        if fail:
            if main_rank:
                raise SystemExit(json.dumps({"non_finite_val_metric": {
                    "epoch": epoch, "joint_logloss": m["joint_logloss"], "q_ap": m["q_ap"]}}))
            raise SystemExit(f"aborting: rank 0 reported a non-finite val metric at epoch {epoch}")
        if stop:
            break
    if best["epoch"] < 0:
        raise SystemExit(f"no epoch improved on the initial {args.select_by} statistic "
                         f"(non-finite val loss?) — nothing to evaluate")

    if world > 1:
        dist.barrier()
    payload = torch.load(out_dir / "s1_best.pt", map_location=device)
    head.load_state_dict(payload["head"])
    readout.load_state_dict(payload["readout"])
    if policy is not None:
        policy.load_state_dict(payload["policy"])
    head.eval()
    if state_fn.store_stats["tokens_stored"] + val_tokens > args.store_token_budget:
        state_fn.clear_store()
    val_metrics = _eval(hold)
    state_fn.clear_store()
    select_metrics = _eval(select)
    fold_metrics = None
    if heldout:
        state_fn.clear_store()
        fold_metrics = _eval(heldout)
    if not main_rank:
        dist.barrier()
        dist.destroy_process_group()
        return 0
    val_rules = score_frontier(hold, margins=MARGINS, thetas=THETAS)
    val_oracle = oracle_report(hold)
    rule_rows = score_frontier(select, margins=MARGINS, thetas=THETAS)
    oracle = oracle_report(select)
    natural = oracle["rows"][0]
    caps = [r for r in oracle["rows"] if r["name"].startswith("cap")]

    def gate(row):
        loss_pp = (natural["accuracy"] - row["accuracy"]) * 100.0
        dominated = any(c["thinking_saving"] >= row["thinking_saving"]
                        and c["accuracy"] >= row["accuracy"] for c in caps)
        return {"name": row["name"], "saving": row["thinking_saving"],
                "loss_pp": loss_pp,
                "pass": row["thinking_saving"] >= 0.10 and loss_pp <= 1.0 and not dominated}

    dump_scored(hold, out_dir / "val_scored.jsonl")
    dump_scored(select, out_dir / "select_scored.jsonl")
    if args.dump_reprs:
        repr_manifest = {}
        for split_name, split in (("fit", train), ("val", hold), ("select", select)):
            state_fn.clear_store()
            repr_manifest[split_name] = dump_reprs_npy(
                state_fn, split, out_dir / f"reprs_{split_name}.npy",
                store_token_budget=args.store_token_budget)
        log({"reprs": repr_manifest})
    if heldout:
        dump_scored(heldout, out_dir / "fit_fold_scored.jsonl")
    choice_floor = osp_sched.anneal_epochs if (osp_mode or cond_mode) else 0
    eligible = [row for row in history if row["epoch"] >= choice_floor]
    ap_choice = lc.choose_epoch(eligible, key="q_ap", minimise=False, patience=args.patience)
    ll_choice = lc.choose_epoch(eligible, key="joint_logloss", minimise=True,
                                patience=args.patience)
    gates = [gate(r) for r in rule_rows]
    osp_summary = None
    if osp_mode:
        def _replay(split):
            point = osp.expected_point(split, grade="loose")
            natural = osp.natural_summary(split)
            point["thinking_saving"] = (1.0 - point["mean_thinking_tokens"]
                                        / natural["mean_thinking_tokens"]
                                        if natural["mean_thinking_tokens"] else 0.0)
            return {"policy": point, "natural": natural}

        osp_summary = {"lambda": args.osp_lambda, "seed": args.seed,
                       "anneal": osp_sched.as_dict(), "grade_train": args.osp_grade,
                       "grade_replay": "loose", "token_scale": args.osp_token_scale,
                       "cost": args.osp_cost,
                       "val": _replay(hold), "select": _replay(select)}
        log({"osp_replay": osp_summary})
    head_dir = Path(args.head_checkpoint)
    if head_dir.is_dir():
        weights_path = head_dir / "weights.pth"
        head_prov = {"weights": source_record(weights_path) if weights_path.exists() else None,
                     "dir": str(head_dir), "note": "config.json not hashed"}
    else:
        head_prov = {"path": args.head_checkpoint}
    summary = {"config": config, "best": best, "history": history,
               "choice_by_logloss": ll_choice, "choice_by_q_ap": ap_choice,
               "choice_epoch_floor": choice_floor,
               "val_metrics": val_metrics, "val_rules": val_rules, "val_oracle": val_oracle["rows"],
               "select_rules": rule_rows, "select_oracle": oracle["rows"],
               "select_metrics": select_metrics,
               "fit_fold_metrics": fold_metrics,
               "head_mode": args.head_mode, "osp_replay": osp_summary,
               "lambda_grid": list(lambdas), "osp_cost": args.osp_cost,
               "v_mask": ("probes" if probes_mode else "ends"),
               "v_label": ("sibling_pass_rate" if "v_sib" in names else "own_chain"),
               "aux_belief_names": config["aux_belief_names"],
               "gate_s1": {"any_pass": any(g["pass"] for g in gates), "rules": gates},
               "provenance": {"fit_chains": [source_record(p) for p in args.fit_chains],
                              "select_chains": source_record(args.select_chains),
                              "questions": [source_record(p) for p in args.questions],
                              "head_checkpoint": head_prov}}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for r in rule_rows:
        log(r)
    log({"gate_s1_any_pass": summary["gate_s1"]["any_pass"]})
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
