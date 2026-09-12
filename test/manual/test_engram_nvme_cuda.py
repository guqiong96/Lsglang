"""Small CUDA integration test, no model checkpoint required.

Run explicitly on a Linux CUDA host with a direct-I/O-capable temp directory:
TMPDIR=/nvme/tmp PYTHONPATH=python python test/manual/test_engram_nvme_cuda.py -v
"""

import gc
import json
import struct
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.engram_nvme import NvmeEngramStore


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestNvmeCuda(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.w = (torch.arange(257 * 256).remainder(112).to(torch.uint8)).view(257, 256)
        self.s = (torch.arange(257 * 8).remainder(3) + 126).to(torch.uint8).view(257, 8)
        header = {
            "layers.1.engram.embed.weight": {
                "dtype": "F8_E4M3",
                "shape": [257, 256],
                "data_offsets": [0, self.w.numel()],
            },
            "layers.1.engram.embed.scale": {
                "dtype": "F8_E8M0",
                "shape": [257, 8],
                "data_offsets": [self.w.numel(), self.w.numel() + self.s.numel()],
            },
        }
        raw = json.dumps(header).encode()
        (self.root / "table.safetensors").write_bytes(
            struct.pack("<Q", len(raw))
            + raw
            + self.w.numpy().tobytes()
            + self.s.numpy().tobytes()
        )
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "table.safetensors" for key in header}})
        )

    def tearDown(self):
        torch.cuda.synchronize()
        self.tmp.cleanup()

    def store(self, staging_bytes=8 << 20):
        return NvmeEngramStore(self.root, 1, 257, 256, 17 * 272, staging_bytes)

    def expected(self, ids):
        weights = (
            self.w[ids.cpu()]
            .view(torch.float8_e4m3fn)
            .float()
            .reshape(*ids.shape, 8, 32)
        )
        scales = torch.pow(2.0, self.s[ids.cpu()].float() - 127)
        return (weights * scales.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)

    def test_eager_empty_cache_and_close(self):
        store = self.store()
        for count in (0, 1, 24, 33, 97, 2048):
            ids = torch.arange(count, device="cuda", dtype=torch.int64).remainder(257)
            torch.testing.assert_close(
                store.lookup(ids).cpu(), self.expected(ids), rtol=0, atol=0
            )
        self.assertGreater(store.stats()["misses"], 0)
        store.close()
        store.close()
        with self.assertRaises(RuntimeError):
            store.lookup(torch.ones(1, device="cuda", dtype=torch.int64))

    def test_eager_cleanup_with_pending_callback(self):
        store = self.store()
        ids = torch.arange(97, device="cuda", dtype=torch.int64)
        out = store.lookup(ids)
        ref = weakref.ref(store)
        del store
        gc.collect()
        self.assertIsNone(ref())
        torch.testing.assert_close(out.cpu(), self.expected(ids), rtol=0, atol=0)

    def test_staging_and_input_guards(self):
        store = self.store(staging_bytes=1)
        with self.assertRaises(ValueError):
            store.lookup(torch.ones(1, dtype=torch.int64))
        with self.assertRaises(RuntimeError):
            store.lookup(torch.ones(24, device="cuda", dtype=torch.int64))
        store.close()

    def test_graph_replay_retains_callback_owners(self):
        store = self.store()
        ids = torch.arange(48, device="cuda", dtype=torch.int64).view(2, 24)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                store.lookup(ids)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = store.lookup(ids)
        with self.assertRaises(RuntimeError):
            store.close()
        ref = weakref.ref(store)
        del store
        gc.collect()
        self.assertIsNotNone(ref())
        for turn in range(24):
            ids.copy_(
                (
                    torch.arange(48, device="cuda", dtype=torch.int64)
                    * (turn + 1)
                    % 257
                ).view(2, 24)
            )
            graph.replay()
            torch.testing.assert_close(out.cpu(), self.expected(ids), rtol=0, atol=0)

    def test_embedding_integration_and_default_path(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers import engram

        parallel = SimpleNamespace(tp_size=1, tp_rank=0, pp_size=1, world_size=1)
        schedule = SimpleNamespace(
            max_running_requests=1, disable_overlap_schedule=True
        )
        with (
            patch.object(engram, "get_parallel", return_value=parallel),
            patch.object(
                engram,
                "get_model",
                return_value=SimpleNamespace(model_path=str(self.root)),
            ),
            patch("sglang.srt.runtime_context.get_schedule", return_value=schedule),
            patch(
                "sglang.srt.runtime_context.get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
            patch(
                "sglang.srt.runtime_context.get_exec",
                return_value=SimpleNamespace(
                    graph=SimpleNamespace(enable_torch_compile=False)
                ),
            ),
            envs.SGLANG_ENABLE_DSV41_ENGRAM_NVME.override(True),
            envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(False),
            envs.SGLANG_DSV41_ENGRAM_NVME_CACHE_BYTES.override(4096),
        ):
            embed = engram.EngramEmbedding(257, 256, 1)
            with self.assertRaises(ValueError):
                embed.finish_load()
            embed._load_rows(embed.weight, self.w.view(torch.float8_e4m3fn))
            embed._load_rows(embed.scale, self.s.view(torch.float8_e8m0fnu))
            embed.finish_load()
            self.assertEqual(embed.weight.numel(), 0)
            self.assertEqual(embed.scale.numel(), 0)
            ids = torch.arange(24, device="cuda", dtype=torch.int64)
            torch.testing.assert_close(
                embed(ids).cpu(), self.expected(ids), rtol=0, atol=0
            )
            with self.assertRaises(RuntimeError):
                embed._load_rows(embed.weight, self.w.view(torch.float8_e4m3fn))
            embed.nvme_table.close()
            with envs.SGLANG_ENABLE_DSV41_ENGRAM_NVME.override(False):
                original = engram.EngramEmbedding(257, 256, 1)
                self.assertIsNone(original.nvme_table)
                self.assertEqual(original.weight.shape, (257, 256))

    def test_reload_rejected_before_consuming_weights(self):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        def weights():
            raise AssertionError("Reload must fail before touching checkpoint weights")
            yield

        with self.assertRaisesRegex(RuntimeError, "immutable"):
            DeepseekV4ForCausalLM.load_weights(
                SimpleNamespace(_nvme_engram_loaded=True), weights()
            )


if __name__ == "__main__":
    unittest.main()
