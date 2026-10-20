"""Freeze the question-to-dataset roster."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rl_controller.iclr import dataset_roster as dr
from rl_controller.iclr.provenance import source_record

SCHEMA = dr.SCHEMA
PROVENANCE_SCHEMA = "iclr-dataset-roster-provenance/v1"

CONVENTION = {
    "basis": "question (one vote per question); the four assemblies are published on it",
    "bootstrap": "500 replicates, seed 0, one shared plan per dataset over that dataset's "
                 "own questions, fixed sorted ordering, plan sha256 recorded per dataset",
    "seeds": "three training seeds, equal weight; per-seed values reported; the CI is "
             "CONDITIONAL ON THE OBSERVED SEEDS and does not contain seed-to-seed spread",
    "selection": "FROZEN: source-val population, checkpoints, the 13 numeric thresholds and "
                 "the members honest selection picked on val are all unchanged. A dataset's "
                 "report rows only EVALUATE those already-selected members.",
    "recomputed_per_subset": "the `natural` and `always_fire` anchors and the ABH geometry "
                             "(hull, common token window). That is evaluation, not re-tuning.",
    "never": ["averaging parent-pool ABH", "averaging CI endpoints",
              "applying a parent pool's CI to a subset",
              "subtracting two same-reference intervals",
              "re-selecting member / checkpoint / lambda per dataset",
              "rebuilding a val split per dataset"],
    "status": "USER-REQUESTED, EXPLORATORY, POINTWISE 95% CIs. Not a pre-registered "
              "confirmatory result: no multiplicity correction is applied, no 'count of "
              "significant wins' is reported, and no dataset is dropped for being small.",
}


def parse_pairs(values, flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for text in values or []:
        pool, sep, path = str(text).partition("=")
        if not sep or not pool.strip() or not path.strip():
            raise SystemExit(f"REFUSED: {flag} {text!r}: expected '<pool>=<path>'")
        if pool.strip() in out:
            raise SystemExit(f"REFUSED: {flag} names pool {pool.strip()!r} twice; one manifest "
                             "per pool, so which file defines the population is never a "
                             "question of argument order")
        out[pool.strip()] = path.strip()
    return out


def build(manifests: dict[str, str], parents: dict[str, str]) -> dr.Roster:
    built = dr.build_roster(manifests, parent_paths=parents)
    return dr.finalize(built)


def provenance_payload(built: dr.Roster, manifests: dict[str, str],
                       parents: dict[str, str], out_dir: Path) -> dict:
    return {
        "schema": PROVENANCE_SCHEMA,
        "roster_schema": SCHEMA,
        "exploratory": True,
        "requested_by": built.roster["requested_by"],
        "generated_by": "rl_controller.scripts.iclr_dataset_roster",
        "inputs": {
            "manifests": {pool: source_record(path) for pool, path in sorted(manifests.items())},
            "parents": {pool: source_record(path) for pool, path in sorted(parents.items())},
            "registered_origin_manifests": dr.ORIGIN_MANIFESTS,
        },
        "outputs": {
            "dataset_roster": str(out_dir / "dataset_roster.json"),
            "dataset_membership": str(out_dir / "dataset_membership.jsonl"),
        },
        "digests": built.digests,
        "checks": {
            "question_sha256_recomputed_from_text": True,
            "population_matched_to_read_split_registry": True,
            "report_split_isolation": {pool: built.roster["pools"][pool]["split_isolation"]
                                       for pool in sorted(built.roster["pools"])},
            "cross_pool_duplicate_qkeys": built.roster["duplicate_qkeys"],
            "every_question_in_exactly_one_dataset": True,
            "source_coverage": built.roster["source_coverage"],
            "counts_vs_frozen_rule_table": "equal on all twelve rows, else this program refuses",
        },
        "not_done": {
            "deduplication": "none. Every pool question is in exactly one dataset row and no "
                             "row lost a question to a lookalike in another pool.",
            "method_intersection": "none. The roster is built from question manifests; no "
                                   "scored dump or method list narrowed it.",
            "dropping_failed_questions": "none. A question whose chains all failed to score is "
                                         "in its dataset row and in its denominator.",
            "year_guessing": "none. The AIME / AMC extra batches keep their batch names; the "
                             "year metadata in the manifests is RECORDED under "
                             "`year_evidence` and applied by nothing.",
        },
        "convention": CONVENTION,
        "pool_level_tables_retained": (
            "results/iclr/two_belief/v5/key_evidence/ keeps the pool-level (t1 / t2 / math_ood "
            "/ ood2) tables for protocol traceability. The by-dataset work is ADDITIVE and "
            "replaces none of it."),
    }


def render_markdown(roster: dict) -> str:
    lines = [
        "# Dataset roster (frozen)", "",
        f"**{roster['n_questions_total']} questions, {roster['n_datasets']} rows.** "
        f"Roster sha256 `{roster['digests']['roster_sha256'][:16]}`, membership sha256 "
        f"`{roster['digests']['membership_sha256'][:16]}`.", "",
        "USER-REQUESTED, EXPLORATORY. Pool and tranche are traceable metadata, not report "
        "rows: t1 and t2 are two report tranches of the SAME three sources.", "",
        f"Every row is a filtered subset -- {roster['answer_filter']}.", "",
        "| dataset | N | origin pool(s) | source(s) | tranche(s) | qkeys sha256 |",
        "|---|---:|---|---|---|---|",
    ]
    for d in roster["datasets"]:
        pools = ", ".join(f"{m['origin_pool']} ({m['n_questions']})" for m in d["members"])
        sources = ", ".join(sorted({m["source"] for m in d["members"]}))
        tranches = ", ".join(str(t) for t in sorted({m["tranche"] for m in d["members"]}))
        lines.append(f"| {d['display_name']} | {d['n_questions']} | {pools} | `{sources}` | "
                     f"{tranches} | `{d['qkeys_sha256'][:12]}` |")
    lines += ["", "| pool | N | registered | tranche |", "|---|---:|---|---|"]
    for pool, info in sorted(roster["pools"].items()):
        lines.append(f"| `{pool}` | {info['n_questions']} | "
                     f"{'yes' if info.get('matches_registry') else 'NO'} | {info['tranche']} |")
    lines += ["", f"Sum over datasets = {roster['n_questions_total']} = "
              + " + ".join(str(n) for _p, n in sorted(roster["pool_totals"].items())) + ".", ""]
    contests = roster.get("contest_cross_reference") or {}
    if contests.get("n_shared_contest_problems"):
        lines += ["## Rows that are not independent samples", "",
                  f"{contests['n_shared_contest_problems']} contest problem(s) appear in two "
                  f"dataset rows under different qkeys and different LaTeX renderings: "
                  f"{contests['pairs']}. Nothing was deduplicated -- both rows keep every "
                  "question -- and the roster refuses any union of the two, so the overlap is "
                  "reported rather than silently halved or silently doubled.", ""]
    evidence = roster.get("year_evidence") or {}
    if evidence:
        lines += ["## Year metadata that was recorded and NOT applied", ""]
        for dataset_id, info in sorted(evidence.items()):
            lines.append(f"- `{dataset_id}`: {info['n_with_metadata']}/{info['n_questions']} "
                         f"questions carry a contest url; years {info['years']}. "
                         f"{info['why_not']}")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", action="append", default=[], metavar="POOL=PATH",
                   help="the manifest defining a pool's population; identified as that pool's "
                        "parent or read split BY DIGEST. Required for all four pools.")
    p.add_argument("--parent", action="append", default=[], metavar="POOL=PATH",
                   help="the PARENT manifest for a pool whose read split was given above, so "
                        "the report questions can be checked against that pool's fit and "
                        "select rows.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--expect-roster-sha256", default=None,
                   help="refuse unless the roster built here hashes to this. Use it to prove a "
                        "later rebuild reproduces the frozen grouping byte for byte.")
    p.add_argument("--dry-run", action="store_true",
                   help="build and check, print the counts, write nothing")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    manifests = parse_pairs(args.manifest, "--manifest")
    parents = parse_pairs(args.parent, "--parent")
    try:
        built = build(manifests, parents)
    except dr.RosterError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    digests = built.digests
    if args.expect_roster_sha256 and args.expect_roster_sha256 != digests["roster_sha256"]:
        print(f"REFUSED: the roster built here hashes to {digests['roster_sha256']}, and "
              f"--expect-roster-sha256 says {args.expect_roster_sha256}. The grouping is frozen "
              "before any result; a rebuild that does not reproduce it is a different grouping, "
              "whatever its counts say.", file=sys.stderr)
        return 2
    counts = {d["display_name"]: d["n_questions"] for d in built.roster["datasets"]}
    print(json.dumps({"total": built.roster["n_questions_total"], "datasets": counts,
                      "roster_sha256": digests["roster_sha256"],
                      "membership_sha256": digests["membership_sha256"]},
                     indent=2, sort_keys=True))
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_membership.jsonl").write_text(
        dr.membership_lines(built.membership), encoding="utf-8")
    (out / "dataset_roster.json").write_text(
        json.dumps(built.roster, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (out / "dataset_roster.md").write_text(render_markdown(built.roster) + "\n",
                                           encoding="utf-8")
    (out / "provenance.json").write_text(
        json.dumps(provenance_payload(built, manifests, parents, out), indent=2,
                   sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        dr.verify_frozen(json.loads((out / "dataset_roster.json").read_text(encoding="utf-8")),
                         (out / "dataset_membership.jsonl").read_bytes())
    except dr.RosterError as exc:
        print(f"REFUSED after writing: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {out}/dataset_roster.json, dataset_membership.jsonl, dataset_roster.md, "
          "provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
