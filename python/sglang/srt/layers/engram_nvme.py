"""Opt-in exact NVMe Engram lookup for single-worker CUDA inference.

The checkpoint must remain immutable for the worker's lifetime. Captured CUDA
graphs contain raw host pointers, so captured stores are retained until process
exit. They cannot be closed or hot-reloaded independently of their graphs.
"""

from __future__ import annotations

import ctypes
import functools
import json
import os
import struct
import sys
import weakref
from pathlib import Path

import torch


class _Work(ctypes.Structure):
    _fields_ = [
        ("store", ctypes.c_void_p),
        ("ids", ctypes.c_void_p),
        ("weights", ctypes.c_void_p),
        ("scales", ctypes.c_void_p),
        ("count", ctypes.c_uint64),
    ]


# CUDA graph replay bypasses Python; a Python finalizer alone cannot establish
# that all graph references are gone. Process ownership makes that explicit.
_CAPTURED_STORES: set[NvmeEngramStore] = set()


def checkpoint_table(model_path, layer_id, rows, dim):
    """Return validated local shard and offsets without materializing tensors."""
    if dim != 256 or rows < 1:
        raise ValueError("NVMe Engram requires nonempty 256-wide FP8 tables")
    root = Path(model_path).resolve()
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError("NVMe Engram requires a local indexed safetensors checkpoint")
    index = json.loads(index_path.read_text())["weight_map"]
    prefix = f"layers.{layer_id}.engram.embed."
    names = [prefix + "weight", prefix + "scale"]
    if any(name not in index for name in names):
        raise ValueError(f"Missing Engram checkpoint entries for layer {layer_id}")
    if index[names[0]] != index[names[1]]:
        raise ValueError("NVMe Engram weight and scale must share a checkpoint shard")
    shard = (root / index[names[0]]).resolve()
    if not shard.is_relative_to(root):
        raise ValueError("Engram shard must be inside the checkpoint directory")
    with shard.open("rb") as source:
        size = os.fstat(source.fileno()).st_size
        raw = source.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= min(100_000_000, size - 8):
            raise ValueError("Invalid safetensors header length")
        header = json.loads(source.read(length))
    offsets = []
    extents = []
    for name, dtype, width in zip(names, ("F8_E4M3", "F8_E8M0"), (256, 8)):
        tensor = header.get(name, {})
        span = tensor.get("data_offsets", [])
        if (
            tensor.get("dtype") != dtype
            or tensor.get("shape") != [rows, width]
            or len(span) != 2
            or any(type(value) is not int for value in span)
            or not 0 <= span[0] <= span[1] <= size - 8 - length
            or span[1] - span[0] != rows * width
        ):
            raise ValueError(f"Invalid Engram tensor metadata: {name}")
        extents.append(span)
        offsets.append(8 + length + span[0])
    if max(x[0] for x in extents) < min(x[1] for x in extents):
        raise ValueError("Overlapping Engram weight and scale extents")
    return shard, offsets


def validate_configuration(parallel, schedule, spec, graph, host_table):
    if sys.platform != "linux" or torch.version.cuda is None:
        raise ValueError("NVMe Engram requires Linux and CUDA")
    if any(
        getattr(parallel, field) != 1 for field in ("tp_size", "pp_size", "world_size")
    ):
        raise ValueError("NVMe Engram currently supports a single worker (TP=PP=1)")
    if schedule.max_running_requests != 1 or not schedule.disable_overlap_schedule:
        raise ValueError(
            "NVMe Engram requires --max-running-requests 1 --disable-overlap-schedule"
        )
    if spec.speculative_algorithm or graph.enable_torch_compile:
        raise ValueError(
            "NVMe Engram does not yet support speculation or torch.compile"
        )
    if host_table:
        raise ValueError("NVMe Engram and host-table mode are mutually exclusive")


@functools.lru_cache(maxsize=1)
def _native_library():
    from torch.utils.cpp_extension import load

    path = load(
        name="sglang_engram_nvme",
        sources=[str(Path(__file__).with_suffix(".cpp"))],
        extra_cflags=["-O3", "-std=c++17", "-pthread"],
        extra_ldflags=["-pthread"],
        with_cuda=False,
        is_python_module=False,
    )
    lib = ctypes.CDLL(path, use_errno=True)
    lib.row_store_open.argtypes = [ctypes.c_char_p] + [ctypes.c_uint64] * 4
    lib.row_store_open.restype = ctypes.c_void_p
    lib.row_store_close.argtypes = [ctypes.c_void_p]
    lib.row_store_close.restype = None
    lib.row_store_lookup.argtypes = [ctypes.c_void_p]
    lib.row_store_lookup.restype = None
    lib.row_store_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.row_store_stats.restype = None
    return lib


@functools.lru_cache(maxsize=1)
def _cuda_library():
    # Use the runtime already loaded by torch, rather than guessing wheel paths
    # and potentially loading a different CUDA runtime into the process.
    paths = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/libcudart.so" in line and line.split()[-1].startswith("/")
    }
    if len(paths) != 1:
        raise RuntimeError(f"Expected one loaded CUDA runtime; found {len(paths)}")
    lib = ctypes.CDLL(paths.pop())
    lib.cudaLaunchHostFunc.argtypes = [ctypes.c_void_p] * 3
    lib.cudaLaunchHostFunc.restype = ctypes.c_int
    return lib


def _release(lib, store, staging, works, device):
    if staging:
        with torch.cuda.device(device):
            torch.cuda.synchronize()
    lib.row_store_close(store)
    staging.clear()
    works.clear()


class NvmeEngramStore:
    def __init__(self, model_path, layer_id, rows, dim, cache_bytes, staging_bytes):
        if type(cache_bytes) is not int or not 0 <= cache_bytes < 2**63:
            raise ValueError("NVMe Engram cache bytes must be a nonnegative int64")
        if type(staging_bytes) is not int or not 0 < staging_bytes < 2**63:
            raise ValueError("NVMe Engram staging bytes must be a positive int64")
        shard, offsets = checkpoint_table(model_path, layer_id, rows, dim)
        self.rows = rows
        self._loaded = set()
        self._staging = {}
        self._works = {}
        self._staging_bytes = 0
        self._staging_limit = staging_bytes
        self._device = torch.cuda.current_device()
        self._lib = _native_library()
        self._cuda = _cuda_library()
        self._store = self._lib.row_store_open(
            os.fsencode(shard), rows, *offsets, cache_bytes
        )
        if not self._store:
            error = ctypes.get_errno()
            raise OSError(
                error,
                f"Cannot open NVMe Engram shard: {os.strerror(error)}",
                str(shard),
            )
        self._finalizer = weakref.finalize(
            self,
            _release,
            self._lib,
            self._store,
            self._staging,
            self._works,
            self._device,
        )
        # CUDA may already be torn down during interpreter exit. The OS reclaims
        # process-owned mappings and descriptors, including captured stores.
        self._finalizer.atexit = False

    def validate_weight(self, kind, source):
        if kind in self._loaded:
            raise RuntimeError(
                "NVMe Engram checkpoints are immutable; restart to reload"
            )
        width, dtype = (
            (256, torch.float8_e4m3fn)
            if kind == "weight"
            else (8, torch.float8_e8m0fnu)
        )
        if list(source.shape) != [self.rows, width] or source.dtype != dtype:
            raise ValueError(f"Engram {kind} checkpoint shape/dtype mismatch")
        self._loaded.add(kind)

    def close(self):
        if self in _CAPTURED_STORES:
            raise RuntimeError(
                "Captured NVMe Engram storage is owned by the worker until exit"
            )
        self._finalizer()
        self._works.clear()

    def finish_load(self):
        if self._loaded != {"weight", "scale"}:
            raise ValueError(
                "NVMe Engram checkpoint did not load both weight and scale"
            )

    def stats(self):
        if not self._finalizer.alive:
            raise RuntimeError("NVMe Engram store is closed")
        values = (ctypes.c_uint64 * 4)()
        self._lib.row_store_stats(self._store, values)
        return dict(zip(("hits", "misses", "read_pages", "cache_bytes"), values))

    def lookup(self, indices):
        from sglang.kernels.ops.embeddings.engram_gather import engram_gather

        if not self._finalizer.alive:
            raise RuntimeError("NVMe Engram store is closed")
        if (
            not indices.is_cuda
            or indices.device.index != self._device
            or indices.dtype != torch.int64
        ):
            raise ValueError("Engram IDs must be int64 on the store's CUDA device")
        count = indices.numel()
        out = torch.empty(
            (*indices.shape, 256), dtype=torch.bfloat16, device=indices.device
        )
        if not count:
            return out
        capacity = 1 << (count - 1).bit_length()
        stream = torch.cuda.current_stream(self._device)
        key = (stream.cuda_stream, capacity)
        capturing = torch.cuda.is_current_stream_capturing()
        if key not in self._staging:
            if capturing:
                raise RuntimeError(
                    "Warm NVMe Engram staging on the capture stream before capture"
                )
            size = capacity * (8 + 256 + 8 + 256 + 8 + 8)
            if self._staging_bytes + size > self._staging_limit:
                raise RuntimeError(
                    "NVMe Engram staging budget exceeded; reduce prefill chunk size or increase SGLANG_DSV41_ENGRAM_NVME_STAGING_BYTES"
                )
            ids = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
            w = torch.empty((capacity, 256), dtype=torch.uint8, pin_memory=True)
            s = torch.empty((capacity, 8), dtype=torch.uint8, pin_memory=True)
            dw, ds = (
                torch.empty_like(w, device=indices.device),
                torch.empty_like(s, device=indices.device),
            )
            sequential = torch.arange(
                capacity, dtype=torch.int64, device=indices.device
            )
            self._staging[key] = ids, w, s, dw, ds, sequential
            self._staging_bytes += size
        ids, w, s, dw, ds, sequential = self._staging[key]
        work_key = (stream.cuda_stream, count)
        if work_key not in self._works:
            if len(self._works) >= 4096:
                raise RuntimeError(
                    "NVMe Engram work metadata budget exceeded; restart the worker"
                )
            self._works[work_key] = _Work(
                self._store, ids.data_ptr(), w.data_ptr(), s.data_ptr(), count
            )
        if capturing:
            _CAPTURED_STORES.add(self)
        ids[:count].copy_(indices.reshape(-1), non_blocking=True)
        error = self._cuda.cudaLaunchHostFunc(
            stream.cuda_stream,
            ctypes.cast(self._lib.row_store_lookup, ctypes.c_void_p),
            ctypes.addressof(self._works[work_key]),
        )
        if error:
            raise RuntimeError(f"CUDA Engram host callback failed: {error}")
        dw[:count].copy_(w[:count], non_blocking=True)
        ds[:count].copy_(s[:count], non_blocking=True)
        engram_gather(
            dw.data_ptr(), ds.data_ptr(), sequential[:count], out.view(-1, 256), 256, 32
        )
        return out
