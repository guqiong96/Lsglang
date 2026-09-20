# DeepSeek-V4.1 prefill scratch memory

Eager prefill retains candidate selections as block masks rather than repeating
each bit for every position. Consumers expand only the current chunk. Selection,
tie behavior, tail replay and graph-decode tensor masks retain their semantics.

`SGLANG_DSV41_INDEXER_SCORE_BUDGET_BYTES` controls the torch indexer's existing
score chunk budget and prevents the dense FP4 indexer from selecting a path
whose FP32 score allocation exceeds that budget. The default remains 1 GiB.
A memory-constrained configuration may choose a smaller budget, for example:

```bash
export SGLANG_DSV41_INDEXER_SCORE_BUDGET_BYTES=134217728
```

This bounds a scratch allocation, not total GPU memory. Weights, KV storage,
graph buffers and other temporaries still determine whether a request fits.
A smaller budget can slow prefill; this is not a decode throughput claim.

Run the focused checks from the repository root:

```bash
PYTHONPATH=python python test/registered/unit/test_dsv41_compact_candidates.py -v
PYTHONPATH=python python test/manual/test_dsv41_compact_candidates_cuda.py -v
```

Each device tests 192 shape/tie combinations, candidate publication against
the original selector, compact-versus-dense consumption and tail replay for
two compression ratios, unchanged tensor masks and dense-allocation thresholds.
These synthetic checks do not replace a full-model long-context benchmark.
