"""Compare the upstream cutlass route with opt-in b12x; no checkpoint required.

CUDA events time captured groups of calls, including activation quantization.
These are individual linear operations, not end-to-end model throughput.
"""

import argparse
import gc
import json
import os
import statistics
from pathlib import Path

import flashinfer
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_blockscaled_linear


def measure(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            fn()
    samples = []
    for _ in range(9):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 20)
    return dict(
        median_ms=statistics.median(samples), min_ms=min(samples), max_ms=max(samples)
    )


def case(m, n, k, seed):
    torch.manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    exponent = torch.randint(
        124, 130, (n // 32, k // 32), device="cuda", dtype=torch.uint8
    )
    scales = flashinfer.block_scale_interleave(exponent.repeat_interleave(32, dim=0))

    def run():
        return flashinfer_mxfp8_blockscaled_linear(
            x, weight, scales, backend="cutlass", pin_tactic=True
        )

    outputs, times = {}, {}
    for enabled in [False, True] if seed == 123 else [True, False]:
        label = "b12x" if enabled else "cutlass"
        with envs.SGLANG_SM120_MXFP8_B12X_SMALL_BATCH.override(enabled):
            outputs[label] = run().clone()
            times[label] = measure(run)
    error = (
        (outputs["b12x"].float() - outputs["cutlass"].float()).norm()
        / outputs["cutlass"].float().norm()
    ).item()
    assert torch.isfinite(outputs["b12x"]).all() and error < 0.005
    return dict(
        M=m,
        N=n,
        K=k,
        seed=seed,
        timings=times,
        relative_l2=error,
        speedup=times["cutlass"]["median_ms"] / times["b12x"]["median_ms"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (12, 0)
    # Run in isolation: lazy compilation can consume substantial host memory.
    # A small CUDA allocator budget alone does not bound compiler processes.
    os.environ.setdefault("MAX_JOBS", "2")
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    free, total = torch.cuda.mem_get_info()
    if available < 16 * 1024**3 or free < total * 0.9:
        raise RuntimeError(
            "Run this benchmark in isolation with >=16 GiB available host memory and >=90% free GPU memory"
        )
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.025)
    report = dict(
        torch=torch.__version__,
        flashinfer=flashinfer.__version__,
        gpu=torch.cuda.get_device_name(),
        scope="captured MXFP8 linear including input quantization; no model throughput claim",
        rows=[],
    )
    for seed in [123, 456]:
        for m in [4, 6]:
            for n, k in [
                (1792, 5120),
                (32768, 1280),
                (5120, 8192),
                (4608, 5120),
                (5120, 2304),
                (25600, 6144),
                (4096, 1280),
            ]:
                row = case(m, n, k, seed)
                report["rows"].append(row)
                args.out.write_text(json.dumps(report, indent=2))
                print(json.dumps(row), flush=True)
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
