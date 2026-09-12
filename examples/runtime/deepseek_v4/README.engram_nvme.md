# Exact NVMe Engram lookup (experimental)

DeepSeek-V4.1 Engram tables can exceed available host memory. This opt-in backend
keeps the original FP8 rows and E8M0 scales in a local safetensors shard and reads
only the rows needed by each forward. Hashing, gating, projections and the
existing `engram_gather` dequantization kernel are unchanged.

This feature does **not** offload inference-time MoE experts to NVMe. CPU expert
weights, checkpoint loading temporaries, KV cache and the rest of the model must
still fit. It does not change NUMA allocation policy or tune FP8 kernels.

## Requirements and supported scope

- Linux, CUDA, a C++17 compiler and Ninja. The small host-only reader is built
  from packaged source through `torch.utils.cpp_extension.load` at initialization.
  No external `libaio` package or CUDA compiler is needed for this reader.
- A local, indexed safetensors checkpoint on a filesystem supporting `O_DIRECT`
  and Linux native AIO. Regular NVMe storage is intended; network filesystems
  and other direct-I/O layouts have not been validated.
- 256-wide `F8_E4M3` weight rows with eight `F8_E8M0` scales per row. A layer's
  weight and scale must be in the same shard. The backing files must remain
  present and immutable for the worker's lifetime.
- One worker (`TP=PP=1`, world size 1), one running request, overlap scheduling
  disabled, no speculative decoding and no `torch.compile`. Unsupported
  configurations fail at initialization. The existing host-table option is
  mutually exclusive with NVMe mode.
- CUDA graph staging must be warmed on the capture stream. Decode graph BS=1
  and eager Prefill are the tested configuration. Concurrent graph replay on
  multiple streams is unsupported.

## Enable

Start from a working DeepSeek-V4.1 model configuration with sufficient memory
for its non-Engram weights. Add the following environment settings and flags:

```bash
export SGLANG_ENABLE_DSV41_ENGRAM_NVME=1
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0
# Each limit is PER ENGRAM LAYER, not per model.
export SGLANG_DSV41_ENGRAM_NVME_CACHE_BYTES=1073741824
export SGLANG_DSV41_ENGRAM_NVME_STAGING_BYTES=134217728

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tensor-parallel-size 1 \
  --max-running-requests 1 --disable-overlap-schedule \
  --chunked-prefill-size 2048 \
  --context-length 32768 --max-total-tokens 32768 \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode 1 --cuda-graph-bs-decode 1
```

`MODEL_PATH` must be a downloaded checkpoint directory, not a Hub model ID.
Retain the appropriate MoE runner, CPU/GPU placement, memory fraction and other
flags from the working configuration. This example is not a promise that the
entire model fits on a particular GPU or a 256GiB host.

NVMe mode defaults to off. Cache bytes may be zero to disable the row cache;
negative budgets are rejected. Staging includes pinned IDs/rows/scales and
device row/scale/sequential-index buffers, allocated lazily by stream and
power-of-two capacity. Its limit must be positive. Output activations and
framework allocations are not included in that staging limit. Each store also
has a fixed 512KiB I/O page buffer and at most 4096 callback descriptors.
The default cache is 1GiB per layer, approximately 2GiB for the two V4.1 tables.

## Execution and lifetime

For a miss, the reader deduplicates the required aligned 4KiB pages in batches
of at most 32 rows / 128 pages. Native AIO submits the page reads and waits for
all completions before publishing exact bytes to pinned staging. The native
CUDA host callback makes no CUDA API calls. The CUDA stream then copies the
rows to the GPU and invokes the existing gather kernel.

Initialization failures report an error. A missing row, truncated backing file
or I/O failure during a callback aborts the worker rather than allowing a
generation to consume stale or missing bytes. The checkpoint is immutable:
model-level weight reload is rejected before consuming new weights once the
initial load has completed.

Eager stores can be closed explicitly; cleanup waits for pending CUDA work.
Captured graphs keep raw host pointers and can outlive Python model references,
so a captured store and its staging are owned by the worker until process exit.
Closing a captured store is rejected. Restart the worker to release captured
storage or replace the checkpoint. This deliberate lifetime policy does not
support model hot-swapping in one worker.

Optional scale residency, service health polling, trigger files, machine-specific
NUMA workarounds and FP4 checkpoint-load staging are not part of this backend.

## Tests

Use a direct-I/O-capable temporary directory. These tests generate small fixture
files; no model checkpoint is required.

```bash
export PYTHONPATH=python
export TMPDIR=/path/to/nvme/test-tmp
mkdir -p "$TMPDIR"
python test/registered/unit/test_engram_nvme.py -v
python test/manual/test_engram_nvme_cuda.py -v
```

The CPU tests check checkpoint metadata, exact bytes, empty and multi-batch
requests, cache collisions, ownership ranges, concurrent native calls, cache
accounting, invalid IDs, short files and unsupported configuration guards.
The CUDA tests check eager lookup, the actual `EngramEmbedding` integration,
unchanged default allocation, pending-callback cleanup, staging limits, reload
rejection, and 24 graph replays after dropping the caller's storage reference.

The backend was extracted from a field adapter tested on SM120. Field timings
of 15.97 to 29.62 tokens/s at 16K input / 2K output included additional scale
and NUMA changes, and are not benchmarks of this isolated PR. Full-model
throughput and model-quality evaluation of this branch remain pending.
