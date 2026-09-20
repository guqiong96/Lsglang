# Optional SM120 MXFP8 dispatch

`SGLANG_SM120_MXFP8_B12X_SMALL_BATCH=1` routes the FlashInfer MXFP8 linear
wrapper's `cutlass` calls to `b12x` only on SM120 with flattened row counts
M=1, 4 or 6. The default is off. Other shapes, devices and explicitly selected
backends retain their existing dispatch. This option can also be used without
NVMe Engram. It does not alter DSpark acceptance, CPU MoE, or model weights.

M is the number of rows in the matrix multiplication, not concurrency or
accepted tokens. For a single-request fixed-width DSpark verification pass,
gamma=3 uses M=4 (anchor plus three candidates), and gamma=5 uses M=6.
Accepting fewer candidates does not by itself make a fixed-width pass cheaper.

## Reproducible component benchmark

Baseline: author branch `dsv4.1-lkmoe-sm80plus` at
`7279506ade6c4a25ea3284d812bc6814dafd7113`, using its `cutlass` wrapper route.
Candidate: the same wrapper with the new opt-in enabled. This does not compare
against the older deployment's M=1-only/Triton fallback configuration.

Environment: Linux, RTX PRO 5000 72GB (SM120), PyTorch 2.13.0+cu130,
CUDA 13, FlashInfer 0.6.18. The recorded run used an otherwise idle GPU without
a resident serving model. Results are hardware-specific. Run this benchmark
separately from serving: lazy compilation also consumes host memory. The script
limits compiler parallelism by default and checks available host/GPU memory.
Use an SM120-capable CUDA toolkit (12.9 or newer), configured through
`CUDA_HOME` and `PATH`, and a FlashInfer build supporting the b12x MXFP8 backend.

Synthetic BF16 inputs and FP8 weights use seven model-representative shapes.
Time includes input quantization, the GEMM, and output handling. Weight scale
layout conversion happens before timing. Tactics are pinned for both routes.
For each shape/seed/route: five warmups, a graph containing 20 calls, and nine
CUDA-event samples of graph replay. The table averages the median call time
from seeds 123 and 456; route order is reversed for the second seed.

| M | N | K | cutlass (ms) | b12x (ms) | Speedup |
| --- | --- | --- | --- | --- | --- |
| 4 | 1792 | 5120 | 0.03537 | 0.01227 | 2.88x |
| 4 | 4096 | 1280 | 0.02097 | 0.00508 | 4.13x |
| 4 | 4608 | 5120 | 0.08545 | 0.01247 | 6.85x |
| 4 | 5120 | 2304 | 0.03783 | 0.00714 | 5.30x |
| 4 | 5120 | 8192 | 0.13251 | 0.01854 | 7.15x |
| 4 | 25600 | 6144 | 0.43531 | 0.13795 | 3.16x |
| 4 | 32768 | 1280 | 0.13088 | 0.01547 | 8.46x |
| 6 | 1792 | 5120 | 0.03352 | 0.01230 | 2.73x |
| 6 | 4096 | 1280 | 0.02338 | 0.00522 | 4.48x |
| 6 | 4608 | 5120 | 0.07816 | 0.01248 | 6.26x |
| 6 | 5120 | 2304 | 0.03959 | 0.00731 | 5.42x |
| 6 | 5120 | 8192 | 0.13273 | 0.01854 | 7.16x |
| 6 | 25600 | 6144 | 0.43959 | 0.13806 | 3.18x |
| 6 | 32768 | 1280 | 0.15925 | 0.01639 | 9.71x |

For a synthetic bundle of one call of each of the seven shapes:

- M=4: 0.878 ms -> 0.209 ms, 4.20x, 76.2% less local operation time.
- M=6: 0.906 ms -> 0.210 ms, 4.31x, 76.8% less local operation time.

These sums are **not decode-step latency**: they do not weight operations by
model call counts, and exclude attention, CPU MoE, Engram I/O, synchronization,
draft generation and acceptance effects. They cannot be converted directly
into model tokens/s. A full-model benchmark on this branch remains outstanding.

All 28 shape/seed comparisons produced identical outputs between the two
routes in this sample. The separate 15-shape CUDA correctness test compares
against independently dequantized FP32 references with relative L2 < 0.005.
Neither result guarantees identical generated text for every model/request.

Raw observations: [sm120_mxfp8_benchmark.json](sm120_mxfp8_benchmark.json).

```bash
PYTHONPATH=python python test/registered/unit/test_sm120_mxfp8_routing.py -v
PYTHONPATH=python python test/manual/test_sm120_mxfp8_b12x.py -v
PYTHONPATH=python python test/manual/benchmark_sm120_mxfp8.py --out benchmark.json
```

To opt in after validating the installation:

```bash
export SGLANG_SM120_MXFP8_B12X_SMALL_BATCH=1
```
