from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, List

import numpy as np
import tqdm

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__file__)

_warmup_registry = {}


def warmup(name: str):
    def decorator(fn):
        _warmup_registry[name] = fn
        return fn

    return decorator


# Warmup prompt source: a fixed passage, tokenized once and repeated to the
# requested shape. Real ids, NOT np.random.randint -- a random sweep is the
# worst access pattern a model with off-device embedding tables can be handed.
# The DSV41 engram table is read from NVMe by token id and the N1 path keeps no
# RAM cache ("stores nothing but the pinned stage"), so a random 12K-token
# shape touches tens of thousands of cold rows. Measured on dsv41-1017 TP2/3090
# (2026-11-03): the 3072/6144/12288-token shapes took 76/82/161 s with the GPU
# gates spinning at 155 W while the filler chased the disk, against 10/14 s for
# the neighbouring power-of-2 shapes. Text repeats its n-grams, so the same M
# coverage costs seconds instead of a cold-table sweep.
_WARMUP_TEXT = (
    "The expert cache keeps the rows a decode step is about to read, so the "
    "step itself does not wait for a disk. A prefill batch is different: it "
    "touches every layer at once and the routing table changes shape with the "
    "token count. 1024 tokens, 2048 tokens, 3072 tokens, 4096 tokens, 8192 "
    "tokens, 16384 tokens, 32768 tokens -- each bucket compiles its own "
    "specialisation, and a bucket that is first seen mid-serving pays the "
    "compile while a peer rank spins in the all-reduce.\n"
    "def plan(rows, chunk_max):\n"
    "    bounds, lo = [], 0\n"
    "    while lo < rows:\n"
    "        hi = min(lo + chunk_max, rows)\n"
    "        bounds.append((lo, hi))\n"
    "        lo = hi\n"
    "    return bounds\n"
    "The reduction is handed over exactly as the caller asked: a sub-chunk that "
    "reduces eagerly changes the arithmetic of every layer behind it. Attention "
    "reads a page table of 64 rows, the compressor reads 4, and the narrow "
    "128-row window reads one; the same prompt must score the same logprobs no "
    "matter how the batch was cut. 3.14159 2.71828 1.61803 0.57721 42 1729 "
    "65536 262144.\n"
    "The hit rate follows the residency set, and the residency set follows what "
    "the previous step happened to read: a cache sized for the real tokens, not "
    "for the verify batch, is the difference between 33 and 39 tokens a second."
)

_warmup_ids_cache: List[int] = []


def _warmup_ids(size: int, tokenizer_manager: TokenizerManager) -> List[int]:
    """``size`` token ids of the fixed passage, repeated to cover the shape.

    Falls back to the old random sweep only when there is no tokenizer to
    encode with (``--skip-tokenizer-init``) or the encode yields nothing: a
    warmup arm must never take the server down.
    """
    global _warmup_ids_cache
    if not _warmup_ids_cache:
        tok = getattr(tokenizer_manager, "tokenizer", None)
        ids: List[int] = []
        try:
            if tok is not None:
                ids = [int(i) for i in tok.encode(_WARMUP_TEXT, add_special_tokens=False)]
        except Exception:
            logger.warning(
                "warmup: tokenizer encode failed, keeping random ids", exc_info=True
            )
            ids = []
        _warmup_ids_cache = [i for i in ids if i >= 0]
    pool = _warmup_ids_cache
    if len(pool) < 32:
        return np.random.randint(2**16, size=[size]).tolist()
    reps = (size + len(pool) - 1) // len(pool)
    return (pool * reps)[:size]


async def execute_warmups(
    disaggregation_mode: str,
    warmup_names: List[str],
    tokenizer_manager: TokenizerManager,
):
    for warmup_name in warmup_names:
        if warmup_name not in _warmup_registry:
            logger.warning(f"Could not find custom warmup {warmup_name}")
            continue
        logger.info(f"Running warmup {warmup_name}")
        await _warmup_registry[warmup_name](disaggregation_mode, tokenizer_manager)


@warmup("whisper_autodetect")
async def whisper_autodetect(
    disaggregation_mode: str, tokenizer_manager: TokenizerManager
):
    """Pre-compile the xgrammar FSM for both Whisper auto-detect regexes.

    The first request that uses each structured-generation regex incurs a
    ~15-20s compilation cost. xgrammar caches compiled grammars by the
    exact regex string, so we warm both the notimestamps and timestamps
    variants here — otherwise the first ``language=None +
    timestamp_granularities`` request would still pay the full spike.
    """
    # A short silent audio encoded as base64 WAV (0.1s, 16kHz, mono) —
    # soundfile produces the WAV header + PCM data from a list of floats.
    import base64
    import io

    import soundfile as sf

    from sglang.srt.entrypoints.openai.transcription_adapters.whisper import (
        FUSED_AUTODETECT_FLAG,
        WHISPER_AUTODETECT_REGEX,
        WHISPER_AUTODETECT_TS_REGEX,
    )

    sr, dur = 16000, 0.1
    n = int(sr * dur)
    buf = io.BytesIO()
    sf.write(buf, [0.0] * n, sr, format="WAV")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    audio_data_uri = f"data:audio/wav;base64,{audio_b64}"

    for variant_name, regex in (
        ("notimestamps", WHISPER_AUTODETECT_REGEX),
        ("timestamps", WHISPER_AUTODETECT_TS_REGEX),
    ):
        logger.info(
            "Compiling Whisper auto-detect regex FSM (%s, one-time, ~15-20s)...",
            variant_name,
        )
        req = GenerateReqInput(
            text="",
            audio_data=audio_data_uri,
            sampling_params={
                "max_new_tokens": 4,
                "temperature": 0,
                "regex": regex,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                FUSED_AUTODETECT_FLAG: True,
            },
            modalities=["audio"],
        )
        # PD prefill servers assert req.bootstrap_room is not None in the
        # default follow_bootstrap_room scheduler; the fake values match
        # what the voice_chat warmup uses for the same reason.
        if disaggregation_mode != "null":
            req.bootstrap_room = 0
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        # Drain the generator so the FSM is fully installed and any
        # downstream exception surfaces instead of being swallowed after
        # the first yield.
        async for _ in tokenizer_manager.generate_request(req, None):
            pass
    logger.info("Whisper auto-detect regex FSMs compiled.")


@warmup("voice_chat")
async def voice_chat(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    # this warms up the fused_moe triton kernels and caches them
    # if we don't do this we break real time inference for voice chat
    for i in tqdm.trange(1, 512):
        size = i * 4
        generate_req_input = GenerateReqInput(
            input_ids=_warmup_ids(size, tokenizer_manager),
            sampling_params={
                "max_new_tokens": 30,
                "temperature": 0.8,
                "stop_token_ids": [1],
                "min_p": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


@warmup("prefill_shapes")
async def prefill_shapes(disaggregation_mode: str, tokenizer_manager: TokenizerManager):
    """Warmup Triton kernels across a wide range of prefill seq_lens (up to 32K).

    Uses power-of-2 sizes plus intermediate points to cover the shape space
    that fused_moe, attention extend, and other Triton kernels may encounter.
    """
    page_size = 64
    sizes = set()
    base = 64
    while base <= 32768:
        sizes.add(base)
        mid = base * 3 // 2
        mid = (mid + page_size - 1) // page_size * page_size
        if mid <= 32768:
            sizes.add(mid)
        base *= 2
    # NOTE (2026-10-27): a full _hybrid_buckets() sweep here was tried and
    # REVERTED -- ~21 extra CuTe DSL compiles at boot made the VLM warmup's
    # cudnn handle creation fail (cudaStreamCreateWithFlags) on the 5060Ti.
    # The ladder already compiles every power-of-two bucket; non-power shapes
    # pay one cute.compile on first sight. The mid-serving repeat tax is fixed
    # on the launch side (bucket-only routing above the cap) instead.
    sizes = sorted(sizes)

    for size in tqdm.tqdm(sizes, desc="Warmup prefill shapes (up to 32K)"):
        generate_req_input = GenerateReqInput(
            input_ids=_warmup_ids(size, tokenizer_manager),
            sampling_params={
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
        )
        if disaggregation_mode != "null":
            generate_req_input.bootstrap_room = 0
            generate_req_input.bootstrap_host = FAKE_BOOTSTRAP_HOST

        await tokenizer_manager.generate_request(generate_req_input, None).__anext__()

    # Concurrent-prefill arm: two requests in flight at once is the only
    # path that touches the bs>1 extend alloc, the QSA mixed-batch layout
    # and the multi-layer conv window scatter kernels. Left cold, the
    # first concurrent request JIT-loads them mid-serving; TP ranks reach
    # module loading at unequal times and NCCL turns that stagger into a
    # lockstep deadlock. Cover them here so the cost is boot time, not a
    # hang in the first batched request.
    pair_a = GenerateReqInput(
        input_ids=_warmup_ids(300, tokenizer_manager),
        sampling_params={"max_new_tokens": 16, "temperature": 0.0},
    )
    pair_b = GenerateReqInput(
        input_ids=_warmup_ids(130, tokenizer_manager),
        sampling_params={"max_new_tokens": 16, "temperature": 0.0},
    )
    if disaggregation_mode == "null":
        await asyncio.gather(
            tokenizer_manager.generate_request(pair_a, None).__anext__(),
            tokenizer_manager.generate_request(pair_b, None).__anext__(),
        )

        # Single-request decode arm. max_new_tokens=1 requests above stop after
        # the prefill forward, so the bs=1 decode step never runs and the
        # alloc_decode Triton kernel (keyed on next_power_of_2(bs)) stays cold:
        # the first real decode then JIT-compiles it mid-serving and can hang
        # in _init_handles (seen on GLM-5.3/3090). Drain a 2-token generation
        # so one bs=1 decode step is paid here.
        single = GenerateReqInput(
            input_ids=_warmup_ids(64, tokenizer_manager),
            sampling_params={"max_new_tokens": 2, "temperature": 0.0},
        )
        async for _ in tokenizer_manager.generate_request(single, None):
            pass

    # Prefix-cache chunk arm (2026-11-03, 0731 @3090): everything above warms a
    # *fresh* prefill of `size` tokens, so the extend shape the scheduler emits
    # after a radix-cache hit -- a handful of new tokens behind a long prefix --
    # is never compiled. The extend kernels are keyed on the query-token count,
    # not on seq_len, so a mid-serving request with 6 new tokens behind a 1024
    # token prefix device-loads a fresh cubin while the peer rank spins in
    # all-reduce: py-spy showed both TP mains parked in cuModuleLoadData with the
    # GPUs at 100% and the detokenizer heartbeat 2 minutes stale. Pay the compile
    # here instead of in the first real batch of a prefix-cached conversation.
    if disaggregation_mode == "null":
        prefix_pool = np.asarray(_warmup_ids(4096, tokenizer_manager))

        async def _drain(ids):
            # .tolist() -- a bare list() leaves numpy scalars in place and
            # normalize_batch_and_arguments reads that as a batch request.
            req = GenerateReqInput(
                input_ids=np.asarray(ids).tolist(),
                sampling_params={"max_new_tokens": 1, "temperature": 0.0},
            )
            async for _ in tokenizer_manager.generate_request(req, None):
                pass

        try:
            for prefix_len in (1024, 2048):
                # Fill the prefix once; the following requests then hit the
                # radix cache and extend by a few tokens only.
                await _drain(prefix_pool[:prefix_len])
                for new_tokens in (1, 2, 4, 8, 16, 32, 64):
                    await _drain(
                        np.concatenate(
                            [
                                prefix_pool[:prefix_len],
                                prefix_pool[prefix_len : prefix_len + new_tokens],
                            ]
                        )
                    )
        except Exception:
            # A warmup arm must never take the server down: the shapes it
            # covers are a latency win, not a correctness requirement.
            logger.exception("prefix-cache warmup arm failed; serving anyway")
