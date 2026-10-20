from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--math-reference", required=True)
    ap.add_argument("--pool", required=True, choices=("ood2_code_report", "lcb_post_f2048"))
    ap.add_argument("--bank-json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = json.loads(Path(a.math_reference).read_text())
    bank = json.loads(Path(a.bank_json).read_text())
    if bank["pool"] != a.pool:
        raise SystemExit(f"bank is for {bank['pool']}")
    if ref.get("schema_version") != "iclr-policy-replay/v1" or ref["answer_format"] != "boxed":
        raise SystemExit("unexpected math reference")
    chains = bank["outputs"]["chains"]
    new = dict(ref)
    new.update(dataset=a.pool, pool=a.pool, chain_bank_sha256=chains["sha256"], manifest_sha256=bank["manifest_sha256"],
               grader_id="iclr_grade_code (euler_code_grade.sbatch @ a86c7045)",
               natural_answer_tokens="k1_code_natural_program_tokens_via_natural_fallback_tokens",
               n_chains=bank["chains"], n_rows=bank["questions"], missing_questions=[], n_missing_questions=0,
               natural=None, summary=None, summary_chain_weighted=None, output=None, inputs=None,
               source_artifact=chains["path"], source_sha256=chains["sha256"],
               question_sets={"select": {"path": chains["path"], "sha256": chains["sha256"], "n": bank["questions"]},
                              "val": (ref.get("question_sets") or {}).get("val") if isinstance(ref.get("question_sets"), dict) else None})
    sci = ref.get("split_chain_identity")
    if not isinstance(sci, dict) or "val" not in sci:
        raise SystemExit("the math reference has no val split identity")
    new["split_chain_identity"] = {"select": {"chain_bank_sha256": chains["sha256"],
                                              "manifest_sha256": bank["manifest_sha256"], "source": chains["path"]},
                                   "val": sci["val"]}
    new["olcb_derivation"] = {"schema": "olcb-reference/v1", "from": str(Path(a.math_reference).resolve()),
                              "from_sha256": sha(a.math_reference), "bank_json": str(Path(a.bank_json).resolve()),
                              "bank_json_sha256": sha(a.bank_json),
                              "kept": {"answer_format": "boxed (producers' only convention; K1 via natural_fallback_tokens)",
                                       "grading": "loose (= strict on 0/1 code answers)"}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(new, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"out": a.out, "pool": a.pool, "chain_bank_sha256": chains["sha256"][:16]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
