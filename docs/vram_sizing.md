# VRAM sizing: the GPU pool, the expert cache, and the prefill working set

Audience: whoever edits a `launch/*.sh` recipe or tunes an `LK_*` knob.
Read this before changing `LK_POOL_SLOTS`, `LK_GPU_PREFILL_WAVE_EXPERTS`,
`LK_GPU_PREFILL_D_SLOTS` or `LK_GPU_PREFILL_SUB_M`.

## 1. The concept: one domain, two demands, take the longer

Each weight type owns **one static VRAM domain** (the "pool" / "arena"). Two
consumers live in it and they **overlap** — prefill owns the domain while it
runs, and the decode cache is rebuilt afterwards:

| consumer | sized by | what it is |
|---|---|---|
| **expert cache** | `LK_POOL_SLOTS` x number of layers | the experts kept resident for decode |
| **prefill working set** | wave / `LK_GPU_PREFILL_D_SLOTS` / `LK_GPU_PREFILL_SUB_M` | the staged experts plus the activation window a prefill pass needs |

The domain is the **longer of the two, never their sum**:

```
pool VRAM = max(cache, prefill)
```

Consequences worth knowing before touching anything:

- `LK_POOL_SLOTS` buys **cache only**. It can never make prefill infeasible,
  and `LK_POOL_SLOTS=0` (cache off) still leaves prefill all the VRAM it needs.
- A prefill working set **larger than the cache grows the domain**, and the KV
  cache shrinks one-for-one. This is the only way a cache knob shows up in the
  KV ledger.
- The cache is not reused across a prefill window; budget its hit rate for
  decode only.

## 2. The knobs

### `LK_POOL_SLOTS` — expert cache

Resident expert slots per MoE layer; the value itself is the switch (`0` = off).
Non-zero values must be **`>= top_k`**: one decode token routes to `top_k`
distinct experts, so a narrower cache cannot hold a token's working set.
Raising it raises VRAM by `slots x layers x expert size` — but only until the
cache becomes longer than the prefill working set, after which the domain does
not grow any further.

### `LK_GPU_PREFILL_WAVE_EXPERTS` — how many experts one pass stages at a time

Cutlass engines only; **the triton and flashinfer engines require
`wave == number of experts`** and refuse anything narrower at init.

It is **not a performance knob**: one pass stages every active expert once
whatever the wave width, so the total weight traffic is unchanged. A wide wave
means fewer per-pass synchronisations and a bigger GEMM; a narrow wave means a
smaller prefill working set and therefore a smaller domain. Raising it past what
the cache can absorb is how a booting recipe turns into
`arena cudaMalloc failed`.

Unset means "all experts in one wave", which is the largest possible working
set — that is why recipes whose cache is small must set it explicitly.

### `LK_GPU_PREFILL_D_SLOTS` — activation window

How many slot rows of activations one pass works through. Smaller values shrink
the prefill working set (and therefore the domain) and cost more passes; the
downstream effect is the same as the wave's, but it does not shrink the staged
expert set, so it is the weaker of the two levers.

### `LK_GPU_PREFILL_SUB_M` — super-chunk size

Both the number of tokens one MoE forward carries and the size of the engine's
output buffer. Raising it makes each forward bigger (fewer, longer passes); it
also raises the prefill working set a little through that buffer.

## 3. Procedure

1. Boot once; the server logs the domain plan for every weight type. It is the
   authority — trust it over any arithmetic:

   ```
   arena: 4.75 GiB total = max(cache 4.75 GiB [720 rows], prefill 4.32 GiB [192 map + 464 act rows])
   ```

2. `total == cache` → nothing to tune. Everything else went to KV.
3. `prefill > cache` → the domain grew and took KV with it. Lower
   `LK_GPU_PREFILL_WAVE_EXPERTS` (halve it: `E/2`, `E/4`, ...) and re-boot until
   the logged prefill fits inside the cache.
4. `arena cudaMalloc failed: X needed, Y free` → `max(cache, prefill)` does not
   fit next to the weights. Lower `LK_GPU_PREFILL_D_SLOTS`, `LK_POOL_SLOTS` or
   the wave.
   **Do not raise `--mem-fraction-static` to compensate** — that takes the
   shortfall out of KV and only defers the failure to the first long prefill.
5. Re-boot and confirm `total == cache`.

## 4. Shipped recipes and what they cost

`*` measured on the rig, `~` expected from the sizing rule above.

| deployment | tp | `LK_POOL_SLOTS` | wave | domain | KV |
|---|---|---|---|---|---|
| qwen38 2x3090 | 2 | 128 | unset (=E) | **7.91 GiB*** (= cache) | 262,144* |
| qwen38 2x5060Ti | 2 | 48 | 128 | **2.97 GiB*** (= cache) | 222,400* |
| glm53 2x3090 | 2 | 16 | 96 | **4.75 GiB*** (= cache) | 262,144* |
| dsv41 2x3090 | 2 | 12 | 64 | **4.20 GiB~** (= cache) | capped by `--mem-fraction-static` |
| dsv41 4-way | 4 | 16 | 64 | **2.80 GiB~** (= cache) | — |
| 0731 2x3090 | 2 | 24 | 128 | **6.42 GiB~** (= cache) | — |
| qwen36 2x3090 | 2 | 128 | unset (=E) | **4.22 GiB~** (= cache) | — |

Recipes whose cache is already longer than the prefill working set need no wave
(qwen36, qwen38 2x3090). Everything else sets one so that **the domain equals
the cache** and KV keeps everything else.

Staging engines cannot split waves, so their working set is the full expert set
and the domain is `max(cache, that)`. On dsv41 2x3090 that is 6.90 GiB instead
of 4.20 GiB — 2.7 GiB of KV paid for the structural limit, not for a knob.

## 5. What is NOT in the domain

Budget these separately; they are real device bytes and they do not appear in
the `arena` line:

- the triton engine's own activation block (a torch allocation sized by
  `LK_GPU_PREFILL_D_SLOTS`, created at engine init so the KV profiler sees it);
- the flashinfer engine's expert mirror, plus its per-bucket workspaces;
- framework workspaces: mamba state cache, decode CUDA graphs, multimodal
  reservations, attention scratch, and the prefill transient of the shape you
  actually serve.

## 6. Pitfalls

- **`wave` is not a performance knob.** Raising it only grows the domain.
- **`LK_POOL_SLOTS` must be `>= top_k`** when non-zero; `0` means cache off.
- **A broken env-var chain silently drops knobs.** Every line of the
  `VAR=value \` block must end with a backslash; one missing backslash turns the
  rest into a separate command and the engine boots with defaults. `bash -n`
  does NOT catch this — if a knob looks ignored, check the chain first.
- **`wave == E` is mandatory for triton / flashinfer.** Setting a narrower wave
  there fails at init by design.
- **The server-side `input throughput` line is an interval statistic**, 1.7–2.9x
  away from client wall clock on long prefills. Record both, and compare like
  with like.
- **The logged `arena` line is the authority.** If it disagrees with this
  document, the document is wrong — fix it here.
