"""Candidate selection equivalence across block tails, ties and replay slices."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import (
    CompactCandidateMask,
    DeepseekV4AttnBackend,
    candidate_mask_rows,
    compact_candidate_blocks,
)
from sglang.srt.layers.attention.dsv4.indexer import select_candidate_blocks
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestCompactCandidates(unittest.TestCase):
    def test_prefill_publish_consume_and_tail_indices(self):
        torch.manual_seed(777)
        q = torch.randn(21, 2, 3)
        weights = torch.randn(21, 2)
        req = torch.tensor([0] * 9 + [1] * 7 + [2] * 5)
        pos = torch.tensor(
            list(range(75, 84)) + list(range(9, 16)) + list(range(130, 135))
        )
        table = torch.arange(3 * 512).reshape(3, 512)
        keys = torch.randn(1536, 3)

        def context(count, masks=None):
            pages = torch.empty(count, 16, dtype=torch.int32)
            raw = torch.empty_like(pages)
            return (
                NS(
                    candidate_masks=masks,
                    token_to_kv_pool=NS(
                        get_low_ratio_index_k_dequant=lambda layer, slots: keys[slots]
                    ),
                    forward_metadata=NS(
                        core_metadata=NS(
                            sparse_page_indices=lambda ratio: pages,
                            sparse_raw_indices=lambda ratio: raw,
                        )
                    ),
                    req_to_token=table,
                ),
                pages,
                raw,
            )

        score = lambda queries, key, w: (
            torch.einsum("thd,ld->thl", queries, key).relu() * w[:, :, None]
        ).sum(1)
        run = DeepseekV4AttnBackend._low_ratio_index_topk_torch
        for ratio in [1, 2]:
            indexer = NS(
                candidate_block_size=8,
                candidate_topk_blocks=3,
                index_topk=16,
                is_candidate_source=True,
                uses_candidates=False,
                queries=lambda queries, freqs: queries,
                head_weights=lambda w: w,
                scores=score,
            )
            layer = NS(
                compress_ratio=ratio,
                layer_id=0,
                indexer=indexer,
                freqs_cis=torch.zeros(512),
            )
            compact, _, _ = context(21)
            with patch(
                "sglang.srt.layers.attention.deepseek_v4_backend._TORCH_INDEXER_SCORE_BUDGET_BYTES",
                4096,
            ):
                run(compact, layer, weights, q, req, pos)
            dense = []
            for r, mask in enumerate(compact.candidate_masks):
                rows = req == r
                lens = (pos[rows] + 1) // ratio
                width = int(lens.max())
                slots = table[r, torch.arange(width) * ratio] // ratio
                scores = score(q[rows], keys[slots], weights[rows])
                scores.masked_fill_(
                    torch.arange(width)[None, :] >= lens[:, None], -torch.inf
                )
                expected = select_candidate_blocks(scores, lens[:, None], 3, 8)
                self.assertTrue(
                    torch.equal(mask.expanded(slice(None), width), expected)
                )
                dense.append(expected)
            indexer.is_candidate_source = False
            indexer.uses_candidates = True
            for tail in [False, True]:
                selected = (
                    torch.tensor([6, 7, 8, 13, 14, 15, 18, 19, 20])
                    if tail
                    else torch.arange(21)
                )
                a, ap, ar = context(
                    len(selected),
                    [m[-3:] for m in compact.candidate_masks]
                    if tail
                    else compact.candidate_masks,
                )
                b, bp, br = context(
                    len(selected), [m[-3:] for m in dense] if tail else dense
                )
                with patch(
                    "sglang.srt.layers.attention.deepseek_v4_backend._TORCH_INDEXER_SCORE_BUDGET_BYTES",
                    4096,
                ):
                    for ctx in [a, b]:
                        run(
                            ctx,
                            layer,
                            weights[selected] * 0.9,
                            q[selected] + 0.1,
                            req[selected],
                            pos[selected],
                        )
                self.assertTrue(torch.equal(ap, bp))
                self.assertTrue(torch.equal(ar, br))

    def test_exact_blocks_and_tail_replay(self):
        torch.manual_seed(491)
        for width in [1, 7, 32, 65, 129, 1025]:
            for block in [1, 8, 32, 128]:
                for k in [1, 4, 16, 2048]:
                    for tied in [False, True]:
                        with self.subTest(width=width, block=block, k=k, tied=tied):
                            scores = (
                                torch.zeros(9, width) if tied else torch.randn(9, width)
                            )
                            lens = torch.tensor(
                                [
                                    0,
                                    1,
                                    width,
                                    width // 2,
                                    min(3, width),
                                    max(0, width - 1),
                                    width,
                                    0,
                                    width,
                                ]
                            )[:, None]
                            scores.masked_fill_(
                                torch.arange(width)[None, :] >= lens, -torch.inf
                            )
                            expected = select_candidate_blocks(scores, lens, k, block)
                            mask = CompactCandidateMask(
                                compact_candidate_blocks(scores, lens, k, block),
                                width,
                                block,
                            )
                            self.assertTrue(
                                torch.equal(
                                    expected,
                                    candidate_mask_rows(mask, slice(None), width),
                                )
                            )
                            self.assertTrue(
                                torch.equal(
                                    expected[-3:],
                                    mask[-3:].expanded(slice(None), width),
                                )
                            )
                            self.assertEqual(
                                mask.blocks.untyped_storage().nbytes(),
                                mask.blocks.numel(),
                            )
                            actual_scores = scores.masked_fill(
                                ~mask.expanded(slice(None), width), -torch.inf
                            )
                            expected_scores = scores.masked_fill(~expected, -torch.inf)
                            self.assertTrue(
                                torch.equal(
                                    actual_scores.topk(min(5, width)).indices,
                                    expected_scores.topk(min(5, width)).indices,
                                )
                            )

    def test_tensor_decode_mask_unchanged(self):
        mask = torch.tensor([[True, False, True], [False, True, False]])
        self.assertTrue(
            torch.equal(candidate_mask_rows(mask, slice(1, None), 2), mask[1:, :2])
        )

    def test_dense_prefill_respects_budget(self):
        module = "sglang.srt.layers.attention.deepseek_v4_backend"
        batch = NS(
            seq_lens_cpu=[64],
            input_ids=torch.zeros(8, dtype=torch.long),
            extend_seq_lens_cpu=[8],
            forward_mode=NS(is_extend=lambda: True),
        )
        with (
            envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.override(False),
            patch(module + "._has_dense_fp4_indexer", return_value=True),
        ):
            with patch(module + "._TORCH_INDEXER_SCORE_BUDGET_BYTES", 2048):
                self.assertTrue(
                    DeepseekV4AttnBackend._use_dense_fp4_prefill_indexer(batch)
                )
            with patch(module + "._TORCH_INDEXER_SCORE_BUDGET_BYTES", 2047):
                self.assertFalse(
                    DeepseekV4AttnBackend._use_dense_fp4_prefill_indexer(batch)
                )
            batch.seq_lens_cpu = None
            self.assertFalse(
                DeepseekV4AttnBackend._use_dense_fp4_prefill_indexer(batch)
            )


if __name__ == "__main__":
    unittest.main()
