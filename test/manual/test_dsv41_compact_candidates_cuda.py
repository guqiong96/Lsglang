"""Run the same exact candidate/tail/budget checks on CUDA without a model."""

import importlib.util
import unittest
from pathlib import Path

import torch

source = Path(__file__).parents[1] / "registered/unit/test_dsv41_compact_candidates.py"
spec = importlib.util.spec_from_file_location("compact_candidate_checks", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCompactCandidatesCuda(module.TestCompactCandidates):
    def setUp(self):
        self.device = torch.device("cuda")
        self.device.__enter__()

    def tearDown(self):
        self.device.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
