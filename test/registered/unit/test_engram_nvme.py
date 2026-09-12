"""Exact storage tests; no model weights or GPU required."""

import ctypes
import json
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.engram_nvme import (
    _Work,
    checkpoint_table,
    validate_configuration,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def fixture(root, rows=257, layer=1):
    # Unaligned offsets exercise both row and scale page crossings. The final
    # direct read extends past EOF but must contain every requested table byte.
    weight = bytes((i * 7 + 11) % 256 for i in range(rows * 256))
    scale = bytes((i * 3 + 19) % 256 for i in range(rows * 8))
    prefix = f"layers.{layer}.engram.embed."
    header = {
        prefix + "weight": {
            "dtype": "F8_E4M3",
            "shape": [rows, 256],
            "data_offsets": [4093, 4093 + len(weight)],
        },
        prefix + "scale": {
            "dtype": "F8_E8M0",
            "shape": [rows, 8],
            "data_offsets": [4093 + len(weight), 4093 + len(weight) + len(scale)],
        },
    }
    raw = json.dumps(header).encode()
    shard = root / "table.safetensors"
    shard.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4093 + weight + scale)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard.name for name in header}})
    )
    return shard, 8 + len(raw) + 4093, 8 + len(raw) + 4093 + len(weight), weight, scale


class TestMetadata(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.shard, self.woff, self.soff, _, _ = fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_offsets(self):
        self.assertEqual(
            checkpoint_table(self.root, 1, 257, 256),
            (self.shard, [self.woff, self.soff]),
        )

    def test_missing_or_unsupported_layout(self):
        for layer, rows, dim in [
            (2, 257, 256),
            (1, 258, 256),
            (1, 257, 128),
            (1, 0, 256),
        ]:
            with (
                self.subTest(layer=layer, rows=rows, dim=dim),
                self.assertRaises(ValueError),
            ):
                checkpoint_table(self.root, layer, rows, dim)

    def test_truncated_file(self):
        self.shard.write_bytes(self.shard.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            checkpoint_table(self.root, 1, 257, 256)

    def test_invalid_header(self):
        for data in [b"", struct.pack("<Q", 100_000_001)]:
            self.shard.write_bytes(data)
            with self.assertRaises(ValueError):
                checkpoint_table(self.root, 1, 257, 256)

    def test_invalid_tensor_metadata(self):
        original = self.shard.read_bytes()
        length = struct.unpack("<Q", original[:8])[0]
        for key, value in [
            ("dtype", "U8"),
            ("data_offsets", [0, 3]),
            ("data_offsets", [-1, 257 * 256 - 1]),
            ("data_offsets", [0.0, 257 * 256.0]),
        ]:
            header = json.loads(original[8 : 8 + length])
            header["layers.1.engram.embed.weight"][key] = value
            raw = json.dumps(header).encode()
            self.shard.write_bytes(
                struct.pack("<Q", len(raw)) + raw + original[8 + length :]
            )
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                checkpoint_table(self.root, 1, 257, 256)

    def test_shard_location_and_split(self):
        index = self.root / "model.safetensors.index.json"
        for weight, scale in [
            ("table.safetensors", "other.safetensors"),
            ("../outside", "../outside"),
        ]:
            index.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "layers.1.engram.embed.weight": weight,
                            "layers.1.engram.embed.scale": scale,
                        }
                    }
                )
            )
            with self.assertRaises(ValueError):
                checkpoint_table(self.root, 1, 257, 256)

    def test_configuration_guards(self):
        parallel = SimpleNamespace(tp_size=1, pp_size=1, world_size=1)
        schedule = SimpleNamespace(
            max_running_requests=1, disable_overlap_schedule=True
        )
        spec = SimpleNamespace(speculative_algorithm=None)
        graph = SimpleNamespace(enable_torch_compile=False)
        with patch("torch.version.cuda", "13.0"):
            validate_configuration(parallel, schedule, spec, graph, False)
            for obj, key, value in [
                (parallel, "world_size", 2),
                (parallel, "tp_size", 2),
                (schedule, "max_running_requests", 2),
                (schedule, "disable_overlap_schedule", False),
                (spec, "speculative_algorithm", "DSPARK"),
                (graph, "enable_torch_compile", True),
            ]:
                with (
                    self.subTest(key=key),
                    patch.object(obj, key, value),
                    self.assertRaises(ValueError),
                ):
                    validate_configuration(parallel, schedule, spec, graph, False)
            with self.assertRaises(ValueError):
                validate_configuration(parallel, schedule, spec, graph, True)


class TestNativeReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.shard, cls.woff, cls.soff, cls.weights, cls.scales = fixture(cls.root)
        source = (
            Path(__file__).resolve().parents[3]
            / "python/sglang/srt/layers/engram_nvme.cpp"
        )
        cls.library = cls.root / "rows.so"
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-O2",
                "-shared",
                "-fPIC",
                "-pthread",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(source),
                "-o",
                str(cls.library),
            ],
            check=True,
        )
        cls.lib = ctypes.CDLL(str(cls.library), use_errno=True)
        cls.lib.row_store_open.argtypes = [ctypes.c_char_p] + [ctypes.c_uint64] * 4
        cls.lib.row_store_open.restype = ctypes.c_void_p
        cls.lib.row_store_close.argtypes = [ctypes.c_void_p]
        cls.lib.row_store_lookup.argtypes = [ctypes.c_void_p]
        cls.lib.row_store_range.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        cls.lib.row_store_stats.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
        ]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def lookup(self, store, ids):
        indices = (ctypes.c_int64 * len(ids))(*ids)
        weights = (ctypes.c_uint8 * (len(ids) * 256))()
        scales = (ctypes.c_uint8 * (len(ids) * 8))()
        work = _Work(
            store,
            ctypes.addressof(indices),
            ctypes.addressof(weights),
            ctypes.addressof(scales),
            len(ids),
        )
        self.lib.row_store_lookup(ctypes.byref(work))
        return bytes(weights), bytes(scales)

    def open(self, budget=0):
        store = self.lib.row_store_open(
            os.fsencode(self.shard), 257, self.woff, self.soff, budget
        )
        self.assertTrue(store, os.strerror(ctypes.get_errno()))
        return store

    def test_exact_rows_cache_collisions_ranges_and_concurrency(self):
        rng = random.Random(919)
        for slots in (0, 17, 257):
            store = self.open(slots * 272)
            try:
                for lo, hi in ((0, 257), (11, 233)):
                    self.lib.row_store_range(store, lo, hi)
                    cases = [
                        [rng.randrange(257) for _ in range(n)]
                        for n in (0, 1, 24, 32, 33, 97, 2048)
                    ]
                    cases += [[0, 256, 0, 17, 34, 51] * 12]
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        results = list(
                            pool.map(lambda ids: self.lookup(store, ids), cases * 3)
                        )
                    for ids, (weight, scale) in zip(cases * 3, results):
                        self.assertEqual(
                            weight,
                            b"".join(
                                self.weights[i * 256 : (i + 1) * 256]
                                if lo <= i < hi
                                else bytes(256)
                                for i in ids
                            ),
                        )
                        self.assertEqual(
                            scale,
                            b"".join(
                                self.scales[i * 8 : (i + 1) * 8]
                                if lo <= i < hi
                                else bytes(8)
                                for i in ids
                            ),
                        )
            finally:
                self.lib.row_store_close(store)

    def test_cache_accounting(self):
        store = self.open(17 * 272 + 1)
        try:
            first = self.lookup(store, [2, 3])
            self.assertEqual(first, self.lookup(store, [2, 3]))
            stats = (ctypes.c_uint64 * 4)()
            self.lib.row_store_stats(store, stats)
            self.assertEqual(list(stats)[:2], [2, 2])
            self.assertEqual(stats[3], 17 * 272)
        finally:
            self.lib.row_store_close(store)

    def test_reject_missing_and_invalid_extents(self):
        self.assertFalse(
            self.lib.row_store_open(b"/nonexistent-engram-fixture", 257, 0, 0, 0)
        )
        self.assertFalse(
            self.lib.row_store_open(os.fsencode(self.shard), 2**63, 0, 0, 0)
        )

    def test_invalid_id_and_truncation_fail_closed(self):
        # Isolate deliberate aborts, and avoid writing core dumps in CI.
        script = """
import ctypes as c, json, os, resource, sys
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
library, shard, woff, soff, case = json.loads(sys.argv[1])
lib = c.CDLL(library)
lib.row_store_open.argtypes = [c.c_char_p] + [c.c_uint64] * 4
lib.row_store_open.restype = c.c_void_p
class Work(c.Structure):
    _fields_ = [('store', c.c_void_p), ('ids', c.c_void_p), ('w', c.c_void_p), ('s', c.c_void_p), ('count', c.c_uint64)]
store = lib.row_store_open(os.fsencode(shard), 257, woff, soff, 0)
assert store
if case == 'truncate': os.truncate(shard, soff)
ids = (c.c_int64 * 1)(-1 if case == 'id' else 256)
w, s = (c.c_uint8 * 256)(), (c.c_uint8 * 8)()
work = Work(store, c.addressof(ids), c.addressof(w), c.addressof(s), 1)
lib.row_store_lookup(c.byref(work))
"""
        for case in ("id", "truncate"):
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    json.dumps(
                        [str(self.library), str(self.shard), self.woff, self.soff, case]
                    ),
                ],
                capture_output=True,
            )
            self.assertEqual(result.returncode, -6, result.stderr)
            fixture(self.root)


if __name__ == "__main__":
    unittest.main()
