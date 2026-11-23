"""DeepSeek-V4.1 candidate graph variants are admitted on ROCm and bound every HIP verify position."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.graph_variants import (
    create_dsv41_candidate_graph_variants,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestVerifyCandidateGraph(CustomTestCase):
    def test_hip_verify_shortcut_bounds_all_draft_positions(self):
        """A verify row whose last draft position passes the candidate span must not take the shortcut."""
        from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
            low_ratio_decode_rows_fit_candidate_span,
        )

        backend = SimpleNamespace(
            low_ratio_candidate_span=16384, speculative_num_draft_tokens=6
        )
        for length, expected in ((16378, True), (16379, False)):
            batch = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                seq_lens_cpu=torch.tensor([length]),
            )
            self.assertEqual(
                low_ratio_decode_rows_fit_candidate_span(backend, batch), expected
            )
        batch.seq_lens_cpu = None
        self.assertFalse(low_ratio_decode_rows_fit_candidate_span(backend, batch))

    def test_every_supported_capability_captures_the_same_variants(self):
        """The capability gate is a floor, not an arch match.

        Candidate variants add collectives to the captured graph, so a gate one
        rank in a TP group passes and another fails desyncs the captures and the
        group busy-waits (2026-10-18: gated at capability >= 10, an SM86+SM120
        TP4 pair deadlocked in decode capture; the floor went to 8 because the
        indexer runs on Triton below SM90). What has to hold is stronger than
        "admitted at >= 8": every admitted capability captures the SAME labels,
        so no rank in a mixed group can emit a different graph.
        """
        config = SimpleNamespace(
            model_type="deepseek_v41",
            candidate_source_layer_id=1,
            candidate_topk_blocks=128,
            candidate_block_size=128,
            compress_ratios=[1, 2],
            index_topk=2048,
        )
        runner = SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=config),
            device="cuda",
            gpu_id=0,
            spec_algorithm=SimpleNamespace(is_dspark=lambda: False),
            is_draft_worker=False,
        )

        def labels(cap):
            with (
                patch("torch.cuda.get_device_capability", return_value=cap),
                patch("sglang.srt.utils.is_hip", return_value=False),
            ):
                variants = create_dsv41_candidate_graph_variants(
                    runner, ForwardMode.DECODE
                )
            return None if variants is None else variants.capture_labels

        admitted = {cap: labels(cap) for cap in ((12, 0), (10, 0), (9, 0), (8, 9), (8, 6))}
        for cap, captured in admitted.items():
            self.assertIsNotNone(captured, f"capability {cap} must capture candidate variants")
            self.assertEqual(
                captured,
                next(iter(admitted.values())),
                f"capability {cap} captures a different variant set than its peers",
            )
        # Below the floor the indexer needs graphs this factory cannot build.
        self.assertIsNone(labels((7, 5)))

    def test_factory_is_cuda_only(self):
        """ROCm never builds these variants: the HIP decode graphs come from
        create_attention_graph_variants, so admitting HIP here would capture
        candidate variants on the HIP ranks only and desync a mixed group."""
        config = SimpleNamespace(
            model_type="deepseek_v41",
            candidate_source_layer_id=1,
            candidate_topk_blocks=128,
            candidate_block_size=128,
            compress_ratios=[1, 2],
            index_topk=2048,
        )
        runner = SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=config),
            device="cuda",
            gpu_id=0,
            spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
            is_draft_worker=False,
        )
        for cap in ((9, 5), (12, 0)):
            with (
                patch("torch.cuda.get_device_capability", return_value=cap),
                patch("sglang.srt.utils.is_hip", return_value=True),
                patch("sglang.srt.utils.is_gfx95_supported", return_value=True),
            ):
                self.assertIsNone(
                    create_dsv41_candidate_graph_variants(
                        runner, ForwardMode.TARGET_VERIFY, 6
                    ),
                    f"capability {cap} on HIP",
                )

if __name__ == "__main__":
    unittest.main()
