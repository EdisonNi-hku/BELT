"""Evaluate a single deployed checkpoint across pools."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rl_controller.baselines.ospruner import replay as R
from rl_controller.iclr import learning_curve as lc
from rl_controller.iclr import osp_policy as osp
from rl_controller.iclr import pareto as pt
from rl_controller.iclr.provenance import source_record
from rl_controller.scripts import iclr_direct_export as dx

SCHEMA = "iclr-single-checkpoint-panel/v1"
LEDGER_SCHEMA = "iclr-cost-ledger/v1"
COST_AXIS = "total"
BUDGET_FRACTIONS = (0.9, 0.8, 0.6, 0.4)
K_VALUES = (1, 2, 4)
COST_TOLERANCE, ACCURACY_TOLERANCE_PP = 0.05, 2.0
ACCURACY_LEVELS = tuple(f"{int(round(100 * lv))}pp" for lv in pt.SELECT_LEVELS
                        if lv <= ACCURACY_TOLERANCE_PP / 100 + 1e-12)
assert f"{ACCURACY_TOLERANCE_PP:g}pp" in ACCURACY_LEVELS, pt.SELECT_LEVELS
UNCOVERED = f"no candidate within +/-5% of the cost AND {ACCURACY_TOLERANCE_PP:g} pp of natural"
LEDGER_FIELDS = ("n_trainings", "gpu_hours_allocation", "gpu_hours_payload",
                 "n_checkpoints", "weight_storage_bytes")
_ANCHOR_TOL = 1e-6
CAVEAT = ("Two coverage columns. DENSE asks every val member of a method, which "
          "is the question B1a poses; HONEST asks only the val-honest picks (at most one per "
          "accuracy level) and is therefore a LOWER bound. Candidate counts per seed are printed "
          "so the gap between the columns is visible, not asserted.")
SINGLE_CHECKPOINT_PRICE = 1e-5
THRESHOLD_LEVELS = tuple(k / 12 for k in range(13))
SINGLE_THRESHOLD_FAMILY = "D-FIX-single-threshold"
MULTI_THRESHOLD_FAMILY = "D-FIX-multi-threshold"
THRESHOLD_MODES = ("single", "multi", "both")
KNOB_STOCHASTIC, KNOB_THRESHOLD = "lambda", "val_quantile"

def load_ledger(path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    methods = payload.get("methods")
    if payload.get("schema") != LEDGER_SCHEMA or not isinstance(methods, dict) or not methods:
        raise SystemExit(f"{path}: need schema {LEDGER_SCHEMA!r} and a non-empty `methods` "
                         f"block; got {payload.get('schema')!r} and {methods!r}")
    for method, given in methods.items():
        entry = given if isinstance(given, dict) else {}
        checkpoints, mapping = entry.get("checkpoints"), entry.get("member_checkpoint", {})
        bad = [f for f in LEDGER_FIELDS if not isinstance(entry.get(f), (int, float))
               or isinstance(entry[f], bool) or entry[f] < 0]
        if (bad or not isinstance(checkpoints, list) or not isinstance(mapping, dict)
                or len(checkpoints) != entry["n_checkpoints"]
                or set(mapping.values()) - set(checkpoints)):
            raise SystemExit(f"{path}: method {method!r} is malformed -- {bad} negative or "
                             f"non-numeric, or checkpoints={checkpoints!r} is not n_checkpoints="
                             f"{entry.get('n_checkpoints')!r} names covering {mapping!r}")
    return payload

def deployed_checkpoint(method: str, members, entry: dict) -> tuple[str | None, str | None]:
    mapping = entry.get("member_checkpoint", {})
    if not mapping:
        if len(entry["checkpoints"]) == 1:
            return entry["checkpoints"][0], None
        return None, (f"{method} declares checkpoints {entry['checkpoints']} and no "
                      "member_checkpoint map: which serves which member is undeclared")
    missing = sorted(m for m in members if m not in mapping)
    if missing:
        return None, (f"{method}: member_checkpoint does not declare {missing}, whose "
                      "checkpoint is unknown and cannot count as one deployment")
    used = sorted({mapping[m] for m in members})
    if len(used) > 1:
        return None, (f"{method}: its honestly selected members span checkpoints {used} -- a "
                      "multi-training envelope, not a single deployed checkpoint")
    return used[0], None

def method_hulls(bundle, *, method: str, split: str, val_split: str) -> dict:
    out = {}
    for seed in bundle.seeds(method, split):
        hull, selection = pt.honest_hull(bundle, method=method, seed=seed, split=split,
                                         sample=bundle.qkeys(split), val_split=val_split,
                                         cost=COST_AXIS)
        out[seed] = {"hull": hull, "selection": selection}
    if not out:
        raise pt.MergeError(f"no series for {method!r} on {split!r}")
    return out

def natural_cost(hulls_by_method: dict, split: str) -> float:
    ends = {f"{method}/s{seed}": entry["hull"][-1][0]
            for method, seeds in hulls_by_method.items() for seed, entry in seeds.items()}
    if not ends:
        raise SystemExit(f"split {split!r}: no hulls, so no natural anchor")
    low = min(ends.values())
    if any(abs(v - low) > _ANCHOR_TOL * max(1.0, low) for v in ends.values()):
        raise SystemExit(f"split {split!r}: hulls end at different costs ({ends}). They all "
                         "extend to the natural anchor, so a selected member outspends natural "
                         "here and the natural cost cannot be read off them. Refusing to guess.")
    return low

def matched_budget_accuracy(seeds: dict, budgets: dict) -> dict:
    got = {b: [lc.acc_at_tokens(e["hull"], t) for e in seeds.values()]
           for b, t in budgets.items()}
    return {b: None if any(v is None for v in vs) else sum(vs) / len(vs)
            for b, vs in got.items()}

def choose_member(rows, target_cost: float):
    slack = COST_TOLERANCE * target_cost
    ok = [r for r in rows if abs(r["cost"] - target_cost) <= slack * (1 + 1e-12)]
    return min(ok, key=lambda r: (-r["accuracy"], r["cost"], r["name"])) if ok else None

def dense_rows(bundle, method: str, seed, val_split: str):
    nat = pt.natural_val_row(bundle, split=val_split, seed=seed, cost=COST_AXIS)
    floor = nat["accuracy"] - ACCURACY_TOLERANCE_PP / 100.0
    qkeys = bundle.qkeys(val_split)
    rows = []
    for key in bundle.keys_for(split=val_split, method=method, seed=seed):
        r = pt.mean_row(key.member, bundle.stats[key], qkeys, cost=COST_AXIS)
        if r["accuracy"] >= floor - 1e-12:
            rows.append({"name": r["name"], "accuracy": r["accuracy"],
                         "cost": r["mean_thinking_tokens"]})
    return rows


def coverage(seeds: dict, natural_val: float, dense=None) -> dict:
    requests, n_candidates = [], {}
    for fraction in BUDGET_FRACTIONS:
        target, picks = fraction * natural_val, {}
        for seed, entry in sorted(seeds.items()):
            points, by_level = entry["selection"]["val_points"], entry["selection"]["by_level"]
            rows = (dense(seed) if dense else
                    [{"name": n, "accuracy": points[n]["accuracy"],
                      "cost": points[n]["mean_thinking_tokens"]} for n in dict.fromkeys(
                         n for n in map(by_level.get, ACCURACY_LEVELS) if n)])
            n_candidates[str(seed)] = len(rows)
            picks[str(seed)] = choose_member(rows, target)
        covered = bool(picks) and all(v is not None for v in picks.values())
        names = {v["name"] for v in picks.values() if v}
        requests.append({"fraction": fraction, "requested_cost": target,
                         "by_seed": {s: v and v["name"] for s, v in picks.items()},
                         "status": "covered" if covered else "uncovered",
                         "member": names.pop() if covered and len(names) == 1 else None,
                         "reason": None if covered else UNCOVERED})
    return {"requests": requests, "n_candidates_per_seed": n_candidates, "by_K": {
        str(k): {"n_requests": k, "n_covered": sum(r["status"] == "covered" for r in requests[:k]),
                 "uncovered": [r["fraction"] for r in requests[:k] if r["status"] != "covered"]}
        for k in K_VALUES}}

WORKER_THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                     "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                     "VECLIB_MAXIMUM_THREADS": "1"}

def slurm_cpu_allocation(env=None) -> int | None:
    env = os.environ if env is None else env
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        head = str(env.get(name) or "").strip().split("(")[0].strip()
        if head.isdigit() and int(head) > 0:
            return int(head)
    return None

def check_thread_budget(jobs, env=None) -> dict:
    if not isinstance(jobs, int) or isinstance(jobs, bool) or jobs < 1:
        raise Refused(f"--jobs must be an int >= 1, got {jobs!r}. `--jobs 1` is the serial "
                      "path; there is no value meaning 'decide for me', because a pool sized "
                      "from the node's core count ignores the CPUs this step was given.")
    allocated = slurm_cpu_allocation(env)
    if allocated is not None and jobs > allocated:
        raise Refused(
            f"--jobs {jobs} on a {allocated}-CPU Slurm allocation: {jobs} workers x 1 BLAS/OMP "
            f"thread each is {jobs} threads for {allocated} cores. Oversubscribing does not "
            "make a wrong number, it makes a slow one that looks fine. Raise --cpus-per-task "
            f"to {jobs}, or lower --jobs to {allocated}.")
    return {"jobs": jobs, "threads_per_worker": 1, "threads_requested": jobs,
            "slurm_cpus": allocated, "thread_env": dict(WORKER_THREAD_ENV)}

def _init_worker() -> None:
    os.environ.update(WORKER_THREAD_ENV)

def frontier_row(shared: dict, method: str) -> dict:
    bundle, read, reference = shared["bundle"], shared["read"], shared["reference"]
    seeds = read[method]
    ranges = [pt.abh(entry["hull"], read[reference][s]["hull"])
              for s, entry in seeds.items() if s in read[reference]]
    return {
        **pt.abh_with_ci(bundle, method=method, reference=reference, split=shared["split"],
                         plan=shared["plan"], val_split=shared["val_split"], cost=COST_AXIS),
        "checkpoint": shared["checkpoint_of"][method], "status": "single_checkpoint",
        "knob": knob_kinds(bundle, method, shared["split"]),
        "covered_range": ([max(r["t_lo"] for r in ranges), min(r["t_hi"] for r in ranges)]
                          if ranges else None),
        "matched_budget_accuracy": matched_budget_accuracy(seeds, shared["budgets"])}

def frontier_rows(shared: dict, methods: list, *, jobs: int) -> list:
    if jobs > 1 and methods:
        rows = pt.run_tasks_parallel(
            shared, methods, jobs=jobs, run=frontier_row, initializer=_init_worker,
            failure=lambda m: (f"the frontier row for method {m!r} failed in a worker. A pool "
                               "that swallowed this would return a panel silently missing that "
                               "row, which is the one thing a deployment table may not do:"))
        if rows is not None:
            return rows
    return [frontier_row(shared, method) for method in methods]

def panel(bundle, ledger: dict, *, split: str, val_split: str, reference: str,
          n_boot: int, seed: int, jobs: int = 1) -> dict:
    check_thread_budget(jobs)
    roster, on_split = sorted(ledger["methods"]), set(bundle.methods(split))
    if reference not in on_split:
        raise SystemExit(f"reference {reference!r} not on {split!r}: {sorted(on_split)}")
    read, val, checkpoint_of, multi, refused = {}, {}, {}, {}, {}
    for method in (m for m in roster if m in on_split):
        try:
            hulls, val_hulls = (method_hulls(bundle, method=method, split=s, val_split=val_split)
                                for s in (split, val_split))
        except pt.MergeError as exc:
            refused[method] = {"status": "unavailable", "reason": str(exc)}
            continue
        members = sorted({m for s in hulls.values() for m in s["selection"]["members"]})
        checkpoint, reason = deployed_checkpoint(method, members, ledger["methods"][method])
        if reason is not None:
            multi[method] = {"status": "multi_checkpoint", "reason": reason, "members": members,
                             "knob": knob_kinds(bundle, method, split),
                             "checkpoints": ledger["methods"][method]["checkpoints"]}
            continue
        read[method], val[method], checkpoint_of[method] = hulls, val_hulls, checkpoint
    natural_read, natural_val = natural_cost(read, split), natural_cost(val, val_split)
    budgets = {f"{f:g}x": f * natural_read for f in BUDGET_FRACTIONS}
    plan = pt.bootstrap_plan(bundle.qkeys(split), n=n_boot, seed=seed)
    shared = {"bundle": bundle, "read": read, "reference": reference, "split": split,
              "val_split": val_split, "plan": plan, "budgets": budgets,
              "checkpoint_of": checkpoint_of}
    ordered = sorted(m for m in read if m != reference)
    frontier = dict(zip(ordered, frontier_rows(shared, ordered, jobs=jobs)))
    rows = {m: {f: ledger["methods"][m][f] for f in LEDGER_FIELDS} for m in roster}
    return {
        "config": {"split": split, "val_split": val_split, "reference": reference, "K_values":
                   list(K_VALUES), "cost_axis": COST_AXIS, "bootstrap": n_boot, "seed": seed,
                   "n_questions": len(bundle.qkeys(split)), "cost_tolerance": COST_TOLERANCE,
                   "budget_fractions": list(BUDGET_FRACTIONS),
                   "accuracy_tolerance_pp": ACCURACY_TOLERANCE_PP,
                   "member_scope": f"val-honest members, levels {list(ACCURACY_LEVELS)}"},
        "frontier": {"single_checkpoint": frontier, "multi_checkpoint": multi,
                     "unavailable": refused, "natural_cost": natural_read,
                     "matched_budgets": budgets},
        "coverage_honest_members": {
            "natural_val_cost": natural_val,
            "methods": {m: coverage(s, natural_val) for m, s in sorted(val.items())}},
        "coverage_dense": {
            "natural_val_cost": natural_val,
            "methods": {m: coverage(s, natural_val,
                                    dense=lambda seed, m=m: dense_rows(bundle, m, seed, val_split))
                        for m, s in sorted(val.items())}},
        "ledger": {"rows": rows, "unmatched": {
                       "ledger_only": sorted(set(roster) - on_split),
                       "tables_only": sorted(on_split - set(roster) - set(pt.ANCHORS))},
                   "totals": {f: sum(r[f] for r in rows.values()) for f in LEDGER_FIELDS}},
        "caveat": CAVEAT}

def render(result: dict) -> str:
    def table(title, header, rows):
        return [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header),
                *("| " + " | ".join(map(str, r)) + " |" for r in rows), ""]
    front, led = result["frontier"], result["ledger"]
    cover, dense = result["coverage_honest_members"], result["coverage_dense"]
    rows = [[m, r["checkpoint"], "+".join(r["knob"]),
             "-" if r["abh_pp"] is None else f"{r['abh_pp']:.2f}",
             "-" if r["ci95"][0] is None else "[%.2f, %.2f]" % tuple(r["ci95"]),
             "-" if not r["covered_range"] else "%.0f-%.0f" % tuple(r["covered_range"])]
            for m, r in sorted(front["single_checkpoint"].items())]
    rows += [[m, "+".join(r["checkpoints"]), "+".join(r["knob"]), "MULTI-CHECKPOINT", "-", "-"]
             for m, r in sorted(front["multi_checkpoint"].items())] + [
             [m, "-", "-", "NO HONEST OPERATING POINT", "-", "-"]
             for m in sorted(front["unavailable"])]
    return "\n".join(table(
        f"(a) single-checkpoint frontier vs {result['config']['reference']} ({COST_AXIS} cost on "
        f"{result['config']['split']}, natural {front['natural_cost']:.0f} tok)",
        ["method", "checkpoint", "knob", "ABH (pp)", "95% CI", "covered range"], rows) + table(
        f"(b) budget-request coverage on val (natural {cover['natural_val_cost']:.0f} tok, "
        f"+/-{100 * COST_TOLERANCE:.0f}% cost, {ACCURACY_TOLERANCE_PP:g} pp, every seed). "
        "DENSE = every val member; HONEST = the val-honest picks only, a lower bound",
        ["method", *(f"K={k} dense" for k in K_VALUES), *(f"K={k} honest" for k in K_VALUES),
         "candidates/seed", "uncovered (dense)"],
        [[m, *(f"{dense['methods'][m]['by_K'][str(k)]['n_covered']}/{k}" for k in K_VALUES),
          *(f"{r['by_K'][str(k)]['n_covered']}/{k}" for k in K_VALUES),
          "/".join(str(v) for _, v in sorted(dense["methods"][m]["n_candidates_per_seed"].items())),
          ", ".join(f"{f:g}x" for f in
                    dense["methods"][m]["by_K"][str(max(K_VALUES))]["uncovered"]) or "-"]
         for m, r in sorted(cover["methods"].items())]) + table(
        "(c) cost ledger", ["method", *LEDGER_FIELDS],
        [[m, *(f"{r[f]:g}" for f in LEDGER_FIELDS)]
         for m, r in sorted(led["rows"].items()) + [("**total**", led["totals"])]]
        ) + [result["caveat"]])


class Refused(SystemExit):
    pass

def declared_schema(path) -> str | None:
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                doc = json.loads(line)
                return doc.get("schema_version") if isinstance(doc, dict) else None
    except (OSError, ValueError):
        return None
    return None

def load_inputs(question_tables, policy_replay, *, root=None) -> pt.Bundle:
    bundle = None
    for path in question_tables:
        schema = declared_schema(path)
        if schema == pt.POLICY_REPLAY_SCHEMA:
            raise Refused(f"--question-tables {path}: this file declares {schema!r} -- these are "
                          "POLICY-REPLAY records, one per (config, seed, split, question), not an "
                          f"{pt.QUESTION_TABLE_SCHEMA} export. Pass it to --policy-replay.")
        try:
            bundle = pt.load_verified_question_tables(path, root=root, bundle=bundle,
                                                      usage="final")
        except pt.MergeError as exc:
            raise Refused(f"--question-tables {path}: {exc}") from None
    for path in policy_replay:
        if bundle is None:
            raise Refused("--policy-replay without a question table: the shared `natural` and "
                          "`always_fire` anchors every hull hangs on exist in no policy-replay "
                          "artifact, so there is nothing to anchor these records against")
        schema = declared_schema(path)
        if schema != pt.POLICY_REPLAY_SCHEMA:
            raise Refused(f"--policy-replay {path}: this file declares {schema!r}, not "
                          f"{pt.POLICY_REPLAY_SCHEMA!r}. A question table is read by "
                          "--question-tables, which verifies it against its dumps; feeding it "
                          "here would skip that verification entirely.")
        try:
            bundle = pt.load_policy_replay(path, bundle, normalize=bundle.normal_form,
                                           seed_semantics="training")
        except pt.MergeError as exc:
            raise Refused(f"--policy-replay {path}: {exc}") from None
    return bundle

def knob_kinds(bundle: pt.Bundle, method: str, split: str) -> list[str]:
    names = set()
    for key in bundle.keys_for(split=split, method=method):
        knob = bundle.meta.get(key, {}).get("knob")
        names.add(knob.get("name") if isinstance(knob, dict) else None)
    return sorted(str(n) for n in names if n) or ["undeclared"]


def quantile(values, probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    low = int(position)
    if position == low:
        return ordered[low]
    return ordered[low] + (ordered[low + 1] - ordered[low]) * (position - low)

def source_val_thresholds(val_chains, *, levels=THRESHOLD_LEVELS) -> list[float]:
    pis = [float(s["pi"]) for c in val_chains for s in (c.get("scores") or [])]
    if not pis:
        raise Refused("the val dump carries no probe scores at all, so it has no pi "
                      "distribution to read thresholds off")
    return [quantile(pis, level) for level in levels]

def fire_at(chains, threshold: float):
    return [dict(c, scores=[dict(s, pi=float(float(s["pi"]) >= threshold))
                            for s in (c.get("scores") or [])]) for c in chains]

def threshold_header(entry, *, family, config_id, level, threshold, price, split, identity,
                     source, branch, commit) -> dict:
    grade = entry["cost"]["grade"]
    return osp.replay_header(
        run_id=f"{family}-vq{level:.6g}-s{entry['seed']}", method_family="two-belief-direct",
        method_name=family, config_id=config_id,
        knob={"name": KNOB_THRESHOLD, "value": level}, seed=entry["seed"],
        dataset=entry["dataset"], split=split, model_id=dx.MODEL_ID,
        model_revision=dx.MODEL_REVISION, tokenizer_revision=dx.MODEL_REVISION,
        prompt_contract_id=dx.PROMPT_CONTRACT_ID,
        grader_id=dx.GRADER_ID if grade == "loose" else f"{dx.GRADER_ID}+grade_{grade}",
        selection_split=dx.SELECTION_SPLIT, selection_rule=dx.SELECTION_RULE,
        frozen_config={"training_lambda": price, "val_quantile": level,
                       "pi_threshold": threshold, "threshold_source_split": "val",
                       "rule": "stop at the first probe with pi >= pi_threshold",
                       "head_mode": entry["mode"], "training_commit": entry["training_commit"],
                       "replay_grade": grade, "expectation": "survival_product",
                       "q_is": "stop probability not belief",
                       **({"split_input": {
                           "carries_split": split, "from": identity["source"],
                           "pool": identity["pool"],
                           "chain_bank_sha256": identity["chain_bank_sha256"],
                           "manifest_sha256": identity["manifest_sha256"],
                           "note": ("`pool`/`dataset` name the bank this ARTIFACT is filed "
                                    "under; this split's chains came from the bank named "
                                    "here")}}
                          if any(identity[k] != entry[k] for k in
                                 ("pool", "chain_bank_sha256", "manifest_sha256")) else {})},
        chain_bank_sha256=identity["chain_bank_sha256"],
        manifest_sha256=identity["manifest_sha256"],
        pool=pt.resolve_pool(entry["pool"]), source_branch=branch, source_commit=commit,
        source_artifact=source["path"], source_sha256=source["sha256"])

def sweep_entry(entry, index: int, *, family, many_prices: bool, branch: str, commit: str) -> dict:
    label = f"entries[{index}] ({entry['family']} seed {entry['seed']}) {family}"
    if entry["mode"] not in dx.FIXED_MODES:
        raise Refused(f"{label}: mode {entry['mode']!r} is a CONDITIONED head, whose members are "
                      f"the 13 inference prices. The threshold family is the fixed-price arm's "
                      f"({dx.FIXED_MODES}); the two are not merged.")
    names = ("summary", "checkpoint", "val_scored", "pool_scored") \
        + (("pool_summary",) if "pool_summary" in entry else ())
    inputs = {n: dx.verified_file(entry[n], f"{label}.{n}") for n in names}
    summary = json.loads(Path(entry["summary"]["path"]).read_text(encoding="utf-8"))
    config = summary.get("config") or {}
    if summary.get("head_mode") != entry["mode"] or config.get("seed") != entry["seed"]:
        raise Refused(f"{label}: the run is head_mode {summary.get('head_mode')!r} seed "
                      f"{config.get('seed')!r}; the manifest says {entry['mode']!r} / "
                      f"{entry['seed']!r}")
    dx.bind_pool_summary(entry, label)
    ((_column, price),) = dx.bind_prices(entry, summary, label)
    split_inputs = dx.check_split_inputs(entry, label)
    split_inputs["val"] = dx.bind_val_split(entry, split_inputs, summary, label)
    natural_fn = dx.natural_answer_fn(entry["cost"]["answer_format"],
                                      entry["cost"]["force_natural_answer"])
    splits = {"val": ("val_scored", dx.load_dump(entry["val_scored"]["path"], f"{label} val")),
              entry["split"]: ("pool_scored",
                               dx.load_dump(entry["pool_scored"]["path"], f"{label} pool"))}
    both = {c["qkey"] for c in splits["val"][1]} & {c["qkey"] for c in splits[entry["split"]][1]}
    if both:
        raise Refused(f"{label}: {len(both)} question(s) are in BOTH splits, e.g. "
                      f"{sorted(both)[:3]} -- the thresholds come from val and are READ on the "
                      "pool, so an overlap tunes the threshold on what it reports")
    for split, (_key, chains) in splits.items():
        if not chains:
            raise Refused(f"{label}: the {split!r} dump has no chains")
        dx.check_scores(chains, entry, 0, f"{label} {split}")
    thresholds = source_val_thresholds(splits["val"][1])
    rows, points, members = [], {}, []
    for index_, (level, tau) in enumerate(zip(THRESHOLD_LEVELS, thresholds)):
        config_id = f"{family}:lam{price:g}-vq{index_:02d}" if many_prices \
            else f"{family}:vq{index_:02d}"
        members.append({"config_id": config_id, "val_quantile": level, "pi_threshold": tau,
                        "training_lambda": price})
        for split, (key, chains) in splits.items():
            fired = fire_at(chains, tau)
            head = threshold_header(entry, family=family, config_id=config_id, level=level,
                                    threshold=tau, price=price, split=split,
                                    identity=split_inputs[split], source=inputs[key],
                                    branch=branch, commit=commit)
            rows.extend(R.replay_rows(fired, header=head, pi_key="pi",
                                      grade=entry["cost"]["grade"],
                                      natural_answer_tokens=natural_fn,
                                      aux_generated_tokens=entry["cost"]["aux_generated_tokens"]))
            points[f"{config_id}|s{entry['seed']}|{split}"] = osp.expected_point(
                fired, pi_key="pi", grade=entry["cost"]["grade"], natural_answer_tokens=natural_fn)
    return {"rows": rows, "provenance": {
        "label": label, "method_name": family, "inputs": inputs, "cost": entry["cost"],
        **{k: entry[k] for k in ("family", "mode", "seed", "pool", "split", "training_commit")},
        "split_inputs": {split: {**ident, "declared": "split_inputs" in entry,
                                 "source_artifact": inputs[ident["source"]]["path"],
                                 "source_sha256": inputs[ident["source"]]["sha256"]}
                         for split, ident in split_inputs.items()},
        "artifact_identity": {k: entry[k] for k in ("pool", "dataset", "chain_bank_sha256",
                                                    "manifest_sha256")},
        "training_lambda": price, "threshold_source_split": "val",
        "threshold_levels": list(THRESHOLD_LEVELS), "members": members, "points": points,
        "crosscheck": dx.crosscheck(entry, summary, splits["val"][1], natural_fn, label)}}

def sweep(manifest_path, out_dir, *, mode: str) -> list[dict]:
    if mode not in ("single", "multi"):
        raise Refused(f"--threshold-mode {mode!r}: expected single or multi")
    record = source_record(manifest_path)
    doc = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema_version") != dx.MANIFEST_SCHEMA:
        raise Refused(f"--threshold-sweep: schema_version must be {dx.MANIFEST_SCHEMA!r}")
    if set(doc) != dx.MANIFEST_TOP:
        raise Refused(f"--threshold-sweep fields must be {sorted(dx.MANIFEST_TOP)}, "
                      f"got {sorted(doc)}")
    dx._hex(doc["export_commit"], 40, "--threshold-sweep.export_commit")
    if not isinstance(doc["entries"], list) or not doc["entries"]:
        raise Refused("--threshold-sweep.entries: a finite, non-empty input list is required")
    checked = [dx.check_entry(e, i) for i, e in enumerate(doc["entries"])]
    fixed = [(e, i) for i, e in enumerate(checked) if e["mode"] in dx.FIXED_MODES]
    if not fixed:
        raise Refused("--threshold-sweep: no fixed-price entry in this manifest. The threshold "
                      f"family is swept on a fixed-price checkpoint's own pi ({dx.FIXED_MODES}).")
    prices = sorted({float(e["fixed_lambda"]) for e, _ in fixed})
    if mode == "single":
        family, selected = SINGLE_THRESHOLD_FAMILY, [
            (e, i) for e, i in fixed if dx._close(e["fixed_lambda"], SINGLE_CHECKPOINT_PRICE)]
        if not selected:
            raise Refused(f"--threshold-mode single: no entry trained at "
                          f"{SINGLE_CHECKPOINT_PRICE:g}; this manifest holds {prices}. The "
                          "single-checkpoint family is defined at that price (plan v5 section "
                          "4.3) and no neighbouring price may stand in for it.")
    else:
        family, selected = MULTI_THRESHOLD_FAMILY, fixed
        if len(prices) < 2:
            raise Refused(f"--threshold-mode multi: this manifest holds ONE training price "
                          f"({prices}), so a multi-price family would be the single-checkpoint "
                          "one under a name that claims otherwise")
    many = mode == "multi"
    done = [(e, sweep_entry(e, i, family=family, many_prices=many,
                            branch=doc["source_branch"], commit=doc["export_commit"]))
            for e, i in selected]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for pool in sorted({e["pool"] for e, _ in done}):
        group = [(e, d) for e, d in done if e["pool"] == pool]
        dx.check_group([e for e, _ in group], family, pool)
        written.append(dx.write_group([r for _, d in group for r in d["rows"]],
                                      [d["provenance"] for _, d in group], family=family,
                                      pool=pool, out_dir=out, manifest_record=record))
    return written

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--question-tables", nargs="+", required=True,
                   help="iclr-ladder-question-tables/v1 exports. REQUIRED even with "
                        "--policy-replay: the shared anchors live only here")
    p.add_argument("--policy-replay", nargs="+", default=[],
                   help=f"{pt.POLICY_REPLAY_SCHEMA} records (the direct arms' "
                        "*_<pool>.policy_replay.jsonl), read with the merger's own loader")
    p.add_argument("--threshold-sweep", default=None,
                   help=f"{dx.MANIFEST_SCHEMA} json: produce the POST-HOC threshold family "
                        "from the fixed-price checkpoints it names, and read it in")
    p.add_argument("--threshold-mode", default="single", choices=list(THRESHOLD_MODES),
                   help=f"single = the one checkpoint at {SINGLE_CHECKPOINT_PRICE:g} "
                        f"({SINGLE_THRESHOLD_FAMILY}); multi = every training price "
                        f"({MULTI_THRESHOLD_FAMILY}); both = both families")
    p.add_argument("--threshold-out", default=None,
                   help="directory the produced policy-replay artifacts are written to "
                        "(REQUIRED with --threshold-sweep: the panel reads back what it wrote)")
    p.add_argument("--cost-ledger", required=True, help=f"{LEDGER_SCHEMA} json, built by sacct")
    p.add_argument("--root", default=None, help="root for RELATIVE dump / manifest paths")
    p.add_argument("--split", default="select")
    p.add_argument("--val-split", default="val")
    p.add_argument("--reference", default="L1")
    p.add_argument("--bootstrap", type=int, default=pt.DEFAULT_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=0, help="THE shared resample seed")
    p.add_argument("--jobs", type=int, default=1,
                   help="compute the frontier rows in N forked workers, one METHOD each "
                        "(default 1 = the serial path, not a pool of one). A pure speedup: "
                        "the roster, the natural cost and the ONE resample plan are the "
                        "parent's, so the panel is byte-identical at every N. Each worker "
                        "takes ONE BLAS/OMP thread, so N must fit the Slurm CPU allocation")
    p.add_argument("--out", default=None, help="panel json")
    p.add_argument("--md", default=None, help="markdown tables")
    args = p.parse_args(argv)
    if args.jobs < 1:
        p.error(f"--jobs must be >= 1, got {args.jobs}; 1 is the serial path")
    return args

def main(argv=None) -> int:
    args = parse_args(argv)
    check_thread_budget(args.jobs)
    ledger = load_ledger(args.cost_ledger)
    produced, replays = [], list(args.policy_replay)
    if args.threshold_sweep:
        if not args.threshold_out:
            raise Refused("--threshold-sweep needs --threshold-out: the family is written as "
                          "ordinary policy-replay artifacts and read back through the merger's "
                          "loader, so the assembly reads the same bytes this panel did")
        modes = THRESHOLD_MODES[:2] if args.threshold_mode == "both" else (args.threshold_mode,)
        for mode in modes:
            produced += sweep(args.threshold_sweep, args.threshold_out, mode=mode)
        replays += [record["output"]["path"] for record in produced]
    elif args.threshold_out:
        raise Refused("--threshold-out without --threshold-sweep: nothing would be produced")
    bundle = load_inputs(args.question_tables, replays, root=args.root)
    result = panel(bundle, ledger, split=args.split, val_split=args.val_split,
                   reference=args.reference, n_boot=args.bootstrap, seed=args.seed,
                   jobs=args.jobs)
    result["schema"], result["provenance"] = SCHEMA, {
        "question_tables": [source_record(p) for p in args.question_tables],
        "policy_replay": [source_record(p) for p in replays],
        "threshold_sweep": None if not args.threshold_sweep else {
            "manifest": source_record(args.threshold_sweep), "mode": args.threshold_mode,
            "single_checkpoint_price": SINGLE_CHECKPOINT_PRICE,
            "levels": list(THRESHOLD_LEVELS), "artifacts": produced},
        "cost_ledger": source_record(args.cost_ledger),
        "verified_tables": bundle.verified_tables()}
    text = render(result)
    for path, payload in ((args.out, json.dumps(result, indent=2, sort_keys=True)),
                          (args.md, text)):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(payload + "\n", encoding="utf-8")
    print(text)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
