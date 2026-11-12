import hashlib
import json
import math
import struct
import unittest
from pathlib import Path

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "qwen3-1.7b-sample"
WIDTH = 768


def rows(name):
    with open(SAMPLE / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def logit(p, clip):
    p = min(max(float(p), clip), 1.0 - clip)
    return math.log(p / (1.0 - p))


class QwenSampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chains = rows("chains.jsonl")
        cls.reference = {(r["qkey"], r["rep"]): r for r in rows("reference/policy_reference.jsonl")}
        cls.blob = (SAMPLE / "belt_h.f16").read_bytes()
        cls.policies = {
            "BELT-B-R": json.loads((SAMPLE / "policies/belt_b_r_seed0.json").read_text()),
            "BELT-H-R": json.loads((SAMPLE / "policies/belt_h_r_seed0.json").read_text()),
        }

    def test_manifest_hashes(self):
        for line in (SAMPLE / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            data = (SAMPLE / name.strip().lstrip("*")).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest, name)

    def test_chain_structure(self):
        index = rows("belt_h.index.jsonl")
        self.assertEqual(len(index), len(self.chains))
        end = 0
        for chain, ix in zip(self.chains, index):
            n = len(chain["probes"])
            self.assertEqual(n, len(chain["scores"]))
            self.assertEqual(n, len(chain["query_points"]))
            self.assertEqual([ix["qkey"], ix["rep"], ix["n_probes"], ix["offset"]],
                             [chain["qkey"], chain["rep"], n, chain["h"]["offset"]])
            self.assertEqual(ix["offset"], end)
            end += n * WIDTH * 2
            for p, qp, pos in zip(chain["probes"], chain["query_points"],
                                  chain["probe_block_end_index"]):
                self.assertEqual([p["block"], p["prefix_len"]], qp)
                self.assertEqual(chain["block_ends"][pos], p["prefix_len"])
                self.assertLessEqual(p["prefix_len"], len(chain["completion_ids"]))
            self.assertEqual(chain["natural"]["elicited"],
                             "natural_elicitation_suffix_ids" in chain)
        self.assertEqual(end, len(self.blob))

    def test_costs(self):
        for chain in self.chains:
            ref = self.reference[(chain["qkey"], chain["rep"])]
            self.assertEqual(ref["probe_stop_cost_tokens"],
                             [p["prefix_len"] + p["forced_tokens"] for p in chain["probes"]])
            extra = chain["natural"]["elicitation_tokens"] if chain["outcome"] == "natural" else 0
            self.assertEqual(ref["natural_cost_tokens"], chain["thinking_tokens"] + extra)

    def test_policy_logits(self):
        for chain in self.chains:
            ref = self.reference[(chain["qkey"], chain["rep"])]
            n = chain["h"]["n_probes"]
            flat = struct.unpack(f"<{n * WIDTH}e",
                                 self.blob[chain["h"]["offset"]: chain["h"]["offset"] + n * WIDTH * 2])
            h = [flat[i * WIDTH:(i + 1) * WIDTH] for i in range(n)]
            pb = self.policies["BELT-B-R"]
            b = [[logit(s["q"], pb["logit_clip"]), logit(s["v"], pb["logit_clip"]),
                  logit(s["r"], pb["logit_clip"]), logit(s["stability"], pb["logit_clip"]),
                  p["prefix_len"] / pb["length_scale"]]
                 for p, s in zip(chain["probes"], chain["scores"])]
            for name, feats in (("BELT-B-R", b), ("BELT-H-R", h)):
                doc = self.policies[name]
                xs = [[(x - m) / s for x, m, s in zip(row, doc["mean"], doc["std"])]
                      for row in feats]
                for policy_row, expected in zip(doc["rows"], ref["policies"][name]):
                    self.assertEqual(policy_row["lambda_index"], expected["lambda_index"])
                    for x, z_ref in zip(xs, expected["logit"]):
                        z = sum(wi * xi for wi, xi in zip(policy_row["w"], x)) + policy_row["b"]
                        self.assertLessEqual(abs(z - z_ref), 1e-6 * max(1.0, abs(z_ref)))


if __name__ == "__main__":
    unittest.main()
