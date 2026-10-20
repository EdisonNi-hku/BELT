import ast
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SourceCaptureTests(unittest.TestCase):
    def test_every_captured_file_matches_its_recorded_bytes(self):
        manifest = json.loads((ROOT / "provenance/code_sources.json").read_text())
        self.assertTrue(manifest["files"])
        seen = set()
        for row in manifest["files"]:
            with self.subTest(path=row["path"]):
                self.assertNotIn(row["path"], seen)
                seen.add(row["path"])
                path = ROOT / row["path"]
                self.assertTrue(path.is_file())
                data = path.read_bytes()
                self.assertEqual(len(data), row["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), row["sha256"])
                self.assertEqual(len(row["source_revision"]), 40)
                if path.suffix == ".py":
                    ast.parse(data, filename=str(path))
                    compile(data, str(path), "exec")

    def test_beliefs_use_strict_now_and_strictly_later_correctness(self):
        from rl_controller.iclr import targets
        from rl_controller.iclr.chains import is_correct

        chain = {
            "qkey": "unit-only",
            "rep": 0,
            "gold": 7,
            "answer": 0,
            "correct": False,
            "outcome": "natural",
            "thinking_tokens": 12,
            "block_ends": [4, 8, 12],
            "probes": [
                {"prefix_len": 4, "forced": "7", "forced_tokens": 1},
                {"prefix_len": 8, "forced": 7, "forced_tokens": 1},
            ],
        }
        self.assertEqual(is_correct("7", 7), 0)
        self.assertEqual(targets.labels("q", chain), [0.0, 1.0])
        self.assertEqual(targets.labels("v", chain), [0.0, 0.0, 0.0])
        self.assertEqual(targets.labels("r", chain), [1.0, 0.0, 0.0])
        chain.update(answer=None, correct=False)
        self.assertEqual(targets.labels("stability", chain), [None, None])
        chain.update(answer=7, correct=True)
        self.assertEqual(targets.labels("v", chain), [1.0, 1.0, 1.0])
        self.assertEqual(targets.labels("r", chain), [1.0, 1.0, 1.0])

    def test_cpu_generation_help_uses_the_real_parser(self):
        result = subprocess.run(
            [sys.executable, "-m", "rl_controller.scripts.single_cot_probe", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in (
            "--prompt-contract",
            "--answer-format",
            "--spend-report",
            "--query-stride",
        ):
            self.assertIn(option, result.stdout)

    def test_code_grader_can_launch_its_captured_subprocess_harness(self):
        from rl_controller.iclr.code_exec import Limits, run_program

        tests = [{"kind": "call", "fn_name": "add", "input": [2, 3], "output": 5}]
        limits = Limits(wall_s=5.0, cpu_s=5)
        good = run_program(
            "def add(a, b):\n    return a + b\n", tests, limits=limits, prefix=[]
        )
        bad = run_program(
            "def add(a, b):\n    return a - b\n", tests, limits=limits, prefix=[]
        )
        self.assertEqual(good.status, "pass", good.as_dict())
        self.assertEqual(bad.status, "fail", bad.as_dict())

    def test_agent_help_uses_each_environment_without_simulator_imports(self):
        for environment in ("alfworld", "sciworld"):
            with self.subTest(environment=environment):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/olmo_alfworld/alf_tree.py"),
                        "--help",
                    ],
                    cwd=ROOT,
                    env={**os.environ, "OALF_ENV": environment},
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--max-steps", result.stdout)
                self.assertIn("--gen-mode", result.stdout)


if __name__ == "__main__":
    unittest.main()
