"""Join confidence features onto pool chains and write rule run directories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rl_controller.baselines.trainingfree import probe_spec as ps
from rl_controller.baselines.trainingfree import rules as tf
from rl_controller.baselines.trainingfree.calibration import attach_isotonic

STAT_KEY = "s"


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _conf_from_entropy(h: float) -> float:
    try:
        return _clamp01(math.exp(-max(0.0, float(h))))
    except OverflowError:
        return 0.0


def _conf_from_rank(rank: float) -> float:
    return _clamp01(1.0 / max(1.0, float(rank)))


CODE_PROBE_CAP = 21
EAT_OVERHEAD = 200
ANSCONV_CAP = 10


def _is_greedy(row) -> bool:
    return row.get("mode", "teacher_forced") == "greedy"


def _conf(row) -> float:
    probs = row["p_forced"]
    if not probs:
        return 0.0
    if _is_greedy(row):
        return (ps.code_stop_confidence_verbatim(probs) if len(probs) >= 2
                else ps.geometric_mean(probs))
    return ps.answer_confidence(probs, row.get("content_mask"))


def _conf_verbatim(row) -> float:
    probs = row["p_forced"]
    if len(probs) < 2:
        return ps.geometric_mean(probs) if probs else 0.0
    return ps.code_stop_confidence_verbatim(probs)


def _q_code_stop(row, prev):
    return _conf(row), False


def _q_eat(row, prev):
    if not row["h_answer"]:
        return (prev if prev is not None else None), True
    return sum(row["h_answer"]), False


def _q_ansconv(row, prev):
    return (1.0 if row["forced"] is not None else 0.0), False


def _q_wheels(row, prev):
    return float(row["rank_close"]), False


METHODS = {
    "code_stop": {
        "to_confidence": _clamp01,
        "needs_answer": True,
        "q": _q_code_stop, "families": ("CoDE", "CoDEnolog"), "knob": "tau,c",
        "ovh": lambda row: float(row["forced_tokens"]),
        "ovh_hi": lambda row: float(CODE_PROBE_CAP),
        "overhead_note":
            "generated probe-answer tokens at every boundary reached. Charged as the "
            "pool's own MEASURED probe length at that boundary (`forced_tokens`) -- a "
            "real generation from the same prefix, and the only per-boundary generated "
            "length we have. It is an ESTIMATE, not a bound in either direction: their "
            "probe uses a different seed and decodes until </think> rather than until "
            f"'}}', capped at {CODE_PROBE_CAP} tokens. What IS known is that their length "
            f"lies in [1, {CODE_PROBE_CAP}]; `ovh_hi` carries that cap and the "
            "upper-bound sensitivity arm is reported beside the point estimate. Fixed "
            "before any curve was seen.",
    },
    "deer": {
        "to_confidence": _clamp01,
        "needs_answer": True,
        "q": _q_code_stop, "families": ("DEER",), "knob": "t",
        "ovh": lambda row: float(row["forced_tokens"]),
        "ovh_hi": lambda row: float(CODE_PROBE_CAP),
        "overhead_note": "identical probe to CoDE-Stop; see the code_stop note.",
    },
    "eat": {
        "to_confidence": _conf_from_entropy,
        "needs_answer": True,
        "q": _q_eat, "families": ("EAT",), "knob": "e,ts,w",
        "ovh": lambda row: float(EAT_OVERHEAD),
        "ovh_hi": lambda row: float(EAT_OVERHEAD),
        "overhead_note":
            f"{EAT_OVERHEAD} generated tokens per boundary: their entropy estimate is 10 "
            "sampled rollouts of 20 tokens (method_eat.py:68-81). We charge what THEIR "
            "method generates even though our deterministic entropy proxy generates "
            "nothing -- the cost axis prices the method, not our replay of it. This is "
            "their NOMINAL configuration and an UPPER bound: a rollout that hits EOS "
            "early generates fewer tokens, and we cannot measure their EOS rate from our "
            "pools. Disclosed because it is the one overhead here large enough to decide "
            "a ranking on its own.",
    },
    "ansconv": {
        "to_confidence": _clamp01,
        "needs_answer": False,
        "q": _q_ansconv, "families": ("AnsConv",), "knob": "k",
        "ovh": lambda row: float(min(row["forced_tokens"], ANSCONV_CAP)),
        "ovh_hi": lambda row: float(ANSCONV_CAP),
        "overhead_note":
            f"their probe generates at most {ANSCONV_CAP} tokens per boundary "
            "(max_answer_tokens). Charged as min(our measured probe length, cap): an "
            "estimate from a different probe procedure, capped at their limit. Their "
            f"true length lies in [1, {ANSCONV_CAP}].",
    },
    "wheels": {
        "to_confidence": _conf_from_rank,
        "needs_answer": False,
        "q": _q_wheels, "families": ("Wheels",), "knob": "s",
        "ovh": lambda row: 0.0,
        "ovh_hi": lambda row: 0.0,
        "overhead_note":
            "ZERO generated tokens: the rule reads the rank of </think> in the "
            "next-token distribution, which is one forward pass over already-computed "
            "state -- the same class as a hidden-state head (contract 4.3). Their own "
            "`work` charges 1 token per boundary (method_wheels.py:128); we declare the "
            "generated count, which is 0, and note the difference is at most one token "
            "per boundary either way.",
    },
}


def load_conf_rows(paths, *, chains_sha256=None):
    files, manifests = [], []
    for path in paths:
        p = Path(path)
        found = sorted(p.glob("conf_*.jsonl")) if p.is_dir() else [p]
        files.extend(found)
        for shard in found:
            manifest = Path(str(shard).replace(".jsonl", "-manifest.json.txt"))
            if manifest.exists():
                manifests.append((shard, json.loads(manifest.read_text(encoding="utf-8"))))
    if not files:
        raise SystemExit(f"no conf_*.jsonl found under {list(paths)}")
    if chains_sha256 is not None:
        if len(manifests) != len(files):
            raise SystemExit(
                f"{len(files) - len(manifests)} of {len(files)} confidence shards have no "
                "sha manifest, so the chain bank they were measured on cannot be proved. "
                "The manifest is written last, so a shard without one is also unfinished.")
        for shard, manifest in manifests:
            got = (manifest.get("chains") or {}).get("sha256")
            if got != chains_sha256:
                raise SystemExit(
                    f"{shard} was measured on chains sha256 {got}, but this join is "
                    f"against {chains_sha256}. Equal (qkey, rep, probe_index) keys do NOT "
                    "prove the same completions: a stale shard would join cleanly and "
                    "supply confidences for chains that no longer exist.")
    rows, seen = {}, set()
    for path in files:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = ps.validate_row(json.loads(line))
            key = ps.row_key(row)
            if key in seen:
                raise SystemExit(f"duplicate confidence row {key} in {path}")
            seen.add(key)
            rows[key] = row
    return rows, [str(f) for f in files]


def attach_scores(chains, conf_rows, *, method: str, template: str) -> dict:
    spec = METHODS[method]
    q_of = spec["q"]
    holes = [(c["qkey"], int(c["rep"]), i, template)
             for c in chains for i in range(len(c["probes"]))
             if (c["qkey"], int(c["rep"]), i, template) not in conf_rows]
    if holes:
        raise SystemExit(
            f"{len(holes)} of {sum(len(c['probes']) for c in chains)} probe boundaries "
            f"have no confidence row for template {template!r} (first {holes[:3]}). Every "
            "boundary must be covered: a rule evaluated on a chain whose boundaries were "
            "silently dropped is reported on data it never saw. Re-run the missing "
            "confidence shards.")
    imputed, unparsed = 0, 0
    for chain in chains:
        scores, prev = [], None
        for probe_index in range(len(chain["probes"])):
            key = (chain["qkey"], int(chain["rep"]), probe_index, template)
            row = conf_rows[key]
            probe = chain["probes"][probe_index]
            for field in ("prefix_len", "block", "forced", "forced_tokens"):
                if row[field] != probe[field]:
                    raise SystemExit(
                        f"{key}: confidence row {field}={row[field]!r} != chain's "
                        f"{probe[field]!r} -- the confidence pass and this pool disagree "
                        "about where the boundary is or what was answered there, so the "
                        "rows describe different realizations of this chain")
            if row["forced"] is None:
                unparsed += 1
            value, was_imputed = q_of(row, prev)
            if value is not None:
                prev = value
            unusable = was_imputed or (not _is_greedy(row)
                                       and row["forced"] is None and spec["needs_answer"])
            entry = {STAT_KEY: value,
                     "q": None if value is None else spec["to_confidence"](value), "v": 0.0,
                     "ovh": spec["ovh"](row), "ovh_hi": spec["ovh_hi"](row),
                     "imputed": float(was_imputed), "nofire": float(unusable)}
            if _is_greedy(row):
                entry["ovh"] = float(row["n_answer"])
                entry["ovh_hi"] = float(row["n_answer"])
                if row.get("ended_with_think") is not None:
                    entry[tf.EWT_KEY] = float(bool(row["ended_with_think"]))
            scores.append({**entry,
                           "q_verbatim": _conf_verbatim(row),
                           "argmax_all": float(bool(row["argmax_hit"])
                                               and all(row["argmax_hit"]))})
        first = next((sc[STAT_KEY] for sc in scores if sc[STAT_KEY] is not None), None)
        for sc in scores:
            if sc[STAT_KEY] is None:
                sc[STAT_KEY] = first if first is not None else _never_fire(method)
                sc["q"] = spec["to_confidence"](sc[STAT_KEY])
                sc["imputed"] = 1.0
                sc["nofire"] = 1.0
        imputed += sum(int(s["imputed"]) for s in scores)
        chain["scores"] = scores
    return {"n_chains": len(chains), "n_probes": sum(len(c["probes"]) for c in chains),
            "n_unparsed_probes": unparsed, "n_imputed_scores": imputed}


FIRES_HIGH = ("code_stop", "deer", "ansconv")
NEVER_FIRE_LOW, NEVER_FIRE_HIGH = 0.0, 1e9


def _never_fire(method: str) -> float:
    return NEVER_FIRE_LOW if method in FIRES_HIGH else NEVER_FIRE_HIGH


def load_qkeys(path):
    from rl_controller.splits import question_key

    keys = set()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            row = json.loads(line)
            keys.add(row["qkey"] if "qkey" in row else question_key(str(row["question"])))
        else:
            keys.add(line)
    if not keys:
        raise SystemExit(f"{path} yielded no qkeys")
    return keys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument("--template", default="ours", choices=sorted(ps.TEMPLATES))
    p.add_argument("--val-chains", required=True)
    p.add_argument("--val-conf", nargs="+", required=True)
    p.add_argument("--val-qkeys", default=None,
                   help="restrict the val chains to these questions. REQUIRED whenever the "
                        "confidence pass ran on a subset (e.g. the 320 shared val questions "
                        "of a 4,241-question fit pool): without it every unscored chain is a "
                        "join hole and the run is refused.")
    p.add_argument("--select-chains", required=True)
    p.add_argument("--select-conf", nargs="+", required=True)
    p.add_argument("--select-qkeys", default=None,
                   help="restrict the read chains to these questions (e.g. select-sub-256 "
                        "for the fidelity arm)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pool", default="", help="pool name for provenance, e.g. math_union_select")
    p.add_argument("--manifest", default=None, help="the question manifest, for its sha256")
    p.add_argument("--commit", default="", help="EXPECTED_COMMIT of the producing job")
    p.add_argument("--model-revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    p.add_argument("--no-isotonic", dest="isotonic", action="store_false",
                   help="skip attaching q_iso (the C2 calibrated cell)")
    p.add_argument("--seed", type=int, default=0,
                   help="training-free replay is deterministic (contract 6): this is a "
                        "run-dir key, NOT a second seed. Never duplicate a dir to fake n=2.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    from rl_controller.iclr import scoring_io as sio
    from rl_controller.iclr.provenance import source_record

    args = parse_args(argv)
    spec = METHODS[args.method]
    out = {}
    for split, chains_path, conf_paths, qkeys_path in (
            ("val", args.val_chains, args.val_conf, args.val_qkeys),
            ("select", args.select_chains, args.select_conf, args.select_qkeys)):
        chains = sio.load_scored_chains(chains_path)
        source = source_record(chains_path)
        if qkeys_path:
            keep = load_qkeys(qkeys_path)
            before = len(chains)
            chains = [c for c in chains if c["qkey"] in keep]
            if not chains:
                raise SystemExit(f"{qkeys_path} selected none of the {before} {split} chains")
        conf_rows, files = load_conf_rows(conf_paths, chains_sha256=source["sha256"])
        stats = attach_scores(chains, conf_rows, method=args.method, template=args.template)
        out[split] = {"chains": chains, "stats": stats, "files": files, "source": source,
                      "qkeys": qkeys_path}

    iso = None
    if args.isotonic:
        iso = attach_isotonic(out["val"]["chains"], out["select"]["chains"])

    provenance = {
        "method": args.method, "template": args.template,
        "isotonic": iso,
        "template_source": ps.TEMPLATES[args.template].source,
        "pool": args.pool,
        "chains_sha256": out["select"]["source"]["sha256"],
        "val_chains_sha256": out["val"]["source"]["sha256"],
        "manifest_sha256": source_record(args.manifest)["sha256"] if args.manifest else "",
        "commit": args.commit,
        "model_revision": args.model_revision,
        "head_sha256": "",
        "answer_format": "boxed",
        "force_natural_answer": False,
        "overhead_accounting": "per_probe",
        "overhead_key": "ovh",
        "overhead_note": spec["overhead_note"],
        "overhead_upper_bound_key": "ovh_hi",
        "families": list(spec["families"]),
        "family_module": "rl_controller.baselines.trainingfree.rules",
        "confidence_files": {k: out[k]["files"] for k in out},
        "qkey_filters": {k: out[k]["qkeys"] for k in out},
        "join_stats": {k: out[k]["stats"] for k in out},
        "seed_convention": "training-free: replay is deterministic, n=1 (contract 6). "
                           "A duplicated run dir is never a second seed.",
        "deviations": DEVIATIONS,
    }
    summary = sio.summary_stub(
        seed=args.seed, targets=["q", "v"], method=args.method, knob=spec["knob"],
        provenance=provenance)
    res = sio.write_run_dir(args.out_dir, val=out["val"]["chains"],
                            select=out["select"]["chains"], summary=summary)
    print(json.dumps({**res, "method": args.method, "template": args.template,
                      "val_stats": out["val"]["stats"],
                      "select_stats": out["select"]["stats"]}), flush=True)
    return 0


DEVIATIONS = [
    "confidence is TEACHER-FORCED on the canonically re-rendered answer, not "
    "greedy-decoded as in method_codestop.py:15-93 (our pools store the parsed probe "
    "answer, never its tokens). The pass records argmax_hit, so the fraction of "
    "boundaries where the two provably coincide is measured.",
    "the `ewt` gate (their confident exit requires the greedy probe to emit </think>) "
    "has no teacher-forced analogue and is NOT applied, so the confident exit fires "
    "slightly more often than theirs would.",
    "boundaries are our pools' blank-line blocks at stride 4, not their 'Wait'/"
    "'Alternatively' token matches (guardrail 1: our boundaries are the primary arm; "
    "the code_stop template is the fidelity arm on the probe string).",
    "K-bar differs from theirs by construction, so their published tau values do not "
    "transfer in; tau is re-tuned on our val split and transferred only BETWEEN OUR "
    "OWN pools, where the boundary convention cancels.",
    "EAT's entropy is a deterministic per-token entropy along the forced answer, not "
    "the mean cumulative entropy of 10 sampled 20-token rollouts; its warmup is swept "
    "from 1 because their default of 50 exceeds our boundary count.",
    "Stop-Spinning-Wheels' scale knob is ours: their rule has no tunable parameter and "
    "would contribute a single point against our swept hull.",
    "Answer-Convergence compares our pools' parsed integers with `==`, not their "
    "re-probed answers with math_verify.verify.",
]


if __name__ == "__main__":
    raise SystemExit(main())
