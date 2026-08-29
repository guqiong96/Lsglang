import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.flash_mla_sm120 import (
    _validate_flashinfer_sparse_mla_backend,
    flashinfer_sparse_mla_forward,
)
from sglang.srt.mem_cache.kv_cache_configurator import calculate_mla_kv_cache_dim
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFlashInferSparseMLAAdapter(unittest.TestCase):
    def _mock_flashinfer(self, op):
        flashinfer = ModuleType("flashinfer")
        flashinfer.__path__ = []
        mla = ModuleType("flashinfer.mla")
        mla.trtllm_batch_decode_with_kv_cache_mla = op
        flashinfer.mla = mla
        return patch.dict(
            sys.modules,
            {"flashinfer": flashinfer, "flashinfer.mla": mla},
        )

    def test_maps_sglang_layout_to_public_flashinfer_api(self):
        captured = {}

        def fake_op(**kwargs):
            captured.update(kwargs)
            query = kwargs["query"]
            return query.new_full((*query.shape[:-1], kwargs["kv_lora_rank"]), 2)

        with self._mock_flashinfer(fake_op):
            output = flashinfer_sparse_mla_forward(
                q=torch.zeros((2, 8, 576), dtype=torch.bfloat16),
                kv_cache=torch.zeros((128, 1, 656), dtype=torch.uint8),
                indices=torch.tensor(
                    [[7, 9, -1, -1], [4, 6, 8, -1]], dtype=torch.int32
                ),
                seq_lens=torch.tensor([2, 3], dtype=torch.int32),
                workspace_buffer=torch.zeros(1024, dtype=torch.uint8),
                page_size=64,
                kv_cache_dim=656,
                qk_nope_head_dim=192,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                sparse_mla_top_k=4,
                sm_scale=0.125,
                skip_softmax_threshold_scale_factor=0.25,
            )

        self.assertEqual(tuple(captured["query"].shape), (2, 1, 8, 576))
        self.assertEqual(tuple(captured["kv_cache"].shape), (2, 1, 64, 656))
        self.assertEqual(tuple(captured["block_tables"].shape), (2, 1, 4))
        self.assertEqual(
            captured["block_tables"].tolist(),
            [[[7, 9, -1, -1]], [[4, 6, 8, -1]]],
        )
        self.assertEqual(captured["seq_lens"].tolist(), [2, 3])
        self.assertEqual(captured["max_seq_len"], 4)
        self.assertEqual(captured["sparse_mla_top_k"], 4)
        self.assertEqual(captured["qk_nope_head_dim"], 192)
        self.assertEqual(captured["bmm1_scale"], 0.125)
        self.assertEqual(captured["bmm2_scale"], 1.0)
        self.assertEqual(captured["kv_scale_format"], "arbitrary_fp32")
        self.assertEqual(captured["skip_softmax_threshold_scale_factor"], 0.25)
        self.assertFalse(captured["enable_pdl"])
        self.assertNotIn("backend", captured)
        self.assertIsNone(captured["sparse_mla_top_k_lens"])
        self.assertEqual(tuple(output.shape), (2, 8, 512))
        self.assertTrue(torch.all(output == 2))

    def test_supplies_topk_lengths_for_native_nope_mla(self):
        captured = {}
        topk_lens = torch.tensor([2, 3], dtype=torch.int32)

        def fake_op(**kwargs):
            captured.update(kwargs)
            query = kwargs["query"]
            return query.new_zeros((*query.shape[:-1], kwargs["kv_lora_rank"]))

        with self._mock_flashinfer(fake_op), patch(
            "sglang.kernels.ops.attention.dsa.transform_index.prepare_trtllm_nope_sparse_metadata",
            return_value=topk_lens,
        ):
            flashinfer_sparse_mla_forward(
                q=torch.zeros((2, 8, 512), dtype=torch.bfloat16),
                kv_cache=torch.zeros((128, 1, 656), dtype=torch.uint8),
                indices=torch.tensor([[7, 9, -1], [4, 6, 8]], dtype=torch.int32),
                seq_lens=torch.tensor([2, 3], dtype=torch.int32),
                workspace_buffer=torch.zeros(1024, dtype=torch.uint8),
                page_size=64,
                kv_cache_dim=656,
                qk_nope_head_dim=256,
                kv_lora_rank=512,
                qk_rope_head_dim=0,
                sparse_mla_top_k=3,
                sm_scale=0.125,
                skip_softmax_threshold_scale_factor=None,
            )

        self.assertIsNone(captured["sparse_mla_top_k_lens"])
        self.assertEqual(tuple(captured["query"].shape), (2, 1, 8, 576))
        self.assertTrue(torch.all(captured["query"][..., 512:] == 0))
        self.assertEqual(captured["qk_rope_head_dim"], 64)
        self.assertEqual(captured["seq_lens"].tolist(), [2, 3])
        self.assertEqual(captured["sparse_mla_top_k"], 3)
        self.assertFalse(captured["enable_pdl"])

    def test_clamps_kpool_overflow_to_compiled_topk_capacity(self):
        captured = {}
        topk_lens = torch.tensor([2051], dtype=torch.int32)

        def fake_op(**kwargs):
            captured.update(kwargs)
            query = kwargs["query"]
            return query.new_zeros((*query.shape[:-1], kwargs["kv_lora_rank"]))

        with self._mock_flashinfer(fake_op), patch(
            "sglang.kernels.ops.attention.dsa.transform_index.prepare_trtllm_nope_sparse_metadata",
            return_value=topk_lens,
        ):
            flashinfer_sparse_mla_forward(
                q=torch.zeros((1, 8, 512), dtype=torch.bfloat16),
                kv_cache=torch.zeros((128, 1, 656), dtype=torch.uint8),
                indices=torch.arange(2051, dtype=torch.int32).unsqueeze(0),
                seq_lens=torch.tensor([2051], dtype=torch.int32),
                workspace_buffer=torch.zeros(1024, dtype=torch.uint8),
                page_size=64,
                kv_cache_dim=656,
                qk_nope_head_dim=256,
                kv_lora_rank=512,
                qk_rope_head_dim=0,
                sparse_mla_top_k=2048,
                sm_scale=0.125,
                skip_softmax_threshold_scale_factor=None,
            )

        self.assertEqual(tuple(captured["block_tables"].shape), (1, 1, 2048))
        self.assertIsNone(captured["sparse_mla_top_k_lens"])
        self.assertEqual(captured["seq_lens"].tolist(), [2048])
        self.assertEqual(captured["max_seq_len"], 2048)


class TestFlashInferSparseMLABackendGate(unittest.TestCase):
    def _validate(self, prefill, decode, model_arch="GlmMoeDsaForCausalLM"):
        return _validate_flashinfer_sparse_mla_backend(
            model_arch=model_arch,
            device_sm_major=12,
            kv_cache_dtype=torch.float8_e4m3fn,
            prefill_impl=prefill,
            decode_impl=decode,
        )

    def test_accepts_flashinfer_for_both_phases(self):
        for model_arch in (
            "GlmMoeDsaForCausalLM",
            "GlmMoeDsaForCausalLMNextN",
            "Glm5NextForConditionalGeneration",
            "Glm5NextForConditionalGenerationNextN",
        ):
            with self.subTest(model_arch=model_arch):
                self.assertTrue(
                    self._validate(
                        "flashinfer_sparse_mla",
                        "flashinfer_sparse_mla",
                        model_arch,
                    )
                )

    def test_rejects_other_or_mixed_backends(self):
        for prefill, decode in (
            ("trtllm", "trtllm"),
            ("flashinfer_sparse_mla", "trtllm"),
        ):
            with self.subTest(prefill=prefill, decode=decode):
                with self.assertRaisesRegex(ValueError, "only flashinfer_sparse_mla"):
                    self._validate(prefill, decode)

    def test_reports_unsupported_configuration(self):
        with self.assertRaises(ValueError) as error:
            self._validate(
                "flashinfer_sparse_mla",
                "flashinfer_sparse_mla",
                "DeepseekV3ForCausalLM",
            )

        message = str(error.exception)
        self.assertIn("model_arch='DeepseekV3ForCausalLM'", message)
        self.assertIn("sm_major=12", message)
        self.assertIn("kv_cache_dtype=torch.float8_e4m3fn", message)


class TestFlashInferNoPEPackedKV(unittest.TestCase):
    def test_reserves_deepseek_packed_rope_tail(self):
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(),
            kv_lora_rank=512,
            qk_rope_head_dim=0,
        )
        exec_context = SimpleNamespace(
            kernel=SimpleNamespace(
                dsa_prefill_backend="flashinfer_sparse_mla",
                dsa_decode_backend="flashinfer_sparse_mla",
            )
        )
        with patch(
            "sglang.srt.mem_cache.kv_cache_configurator.is_deepseek_dsa",
            return_value=True,
        ), patch(
            "sglang.srt.mem_cache.kv_cache_configurator.get_disagg",
            return_value=SimpleNamespace(disaggregation_mode=None),
        ), patch(
            "sglang.srt.mem_cache.kv_cache_configurator.get_exec",
            return_value=exec_context,
        ):
            kv_cache_dim = calculate_mla_kv_cache_dim(
                model_config=model_config,
                kv_cache_dtype=torch.float8_e4m3fn,
            )

        self.assertEqual(kv_cache_dim, 656)

    def test_writes_zero_rope_tail_for_native_nope(self):
        pool = object.__new__(MLATokenToKVPool)
        pool.dsa_kv_cache_store_fp8 = True
        pool.kv_lora_rank = 512
        pool.kv_cache_dim = 656
        pool.quant_block_size = 128
        pool.rope_storage_dtype = torch.bfloat16

        captured = {}

        def fake_quantize(k_nope, k_rope):
            captured["rope"] = k_rope
            return (
                torch.zeros((2, 1, 528), dtype=torch.uint8),
                torch.zeros((2, 1, 128), dtype=torch.uint8),
            )

        with patch(
            "sglang.srt.mem_cache.memory_pool.quantize_k_cache_separate",
            side_effect=fake_quantize,
        ), patch("sglang.srt.mem_cache.memory_pool.set_mla_kv_buffer_triton"):
            pool._write_mla_kv_buffer(
                torch.zeros((8, 1, 656), dtype=torch.uint8),
                torch.tensor([1, 2], dtype=torch.int64),
                torch.ones((2, 1, 512), dtype=torch.bfloat16),
                None,
            )

        self.assertEqual(tuple(captured["rope"].shape), (2, 1, 64))
        self.assertEqual(captured["rope"].dtype, torch.bfloat16)
        self.assertTrue(torch.all(captured["rope"] == 0))


if __name__ == "__main__":
    unittest.main()
