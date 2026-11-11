"""Offline unit tests: when does a finished mamba request donate nothing?

`MambaComponent._select_finished_checkpoint` decides the cache length and the
ping-pong slot a finished request hands to the tree. Returning None makes
`prepare_for_caching_req` return 0, which drives `effective_cache_len <= 0` in
`UnifiedRadixCache.checkpoint_into_tree` and skips the insert entirely — the
request then contributes nothing to the radix cache.

Real values: --mamba-track-interval 256, prompts ~9.4K / ~17.9K, decode 16-24,
overlap scheduling on (ping-pong buffer size 2).
"""

from types import SimpleNamespace

import pytest

from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent


def _component(buffer_size=2):
    pool = SimpleNamespace(
        mamba_ping_pong_track_buffer_size=buffer_size,
        get_mamba_ping_pong_keep_idx=lambda req: 0,
        get_mamba_ping_pong_other_idx=lambda keep: 1 - keep,
    )
    return SimpleNamespace(cache=SimpleNamespace(req_to_token_pool=pool))


def _req(last, prev):
    return SimpleNamespace(
        kv=SimpleNamespace(mamba_last_track_seqlen=last, mamba_prev_track_seqlen=prev)
    )


def _select(last, prev, key_len, buffer_size=2):
    return MambaComponent._select_finished_checkpoint(
        _component(buffer_size), _req(last, prev), key_len
    )


class TestSelectFinishedCheckpoint:
    def test_no_overshoot_uses_last_track(self):
        # 9426-token prompt, last track at 9216 -> plain reuse.
        assert _select(9216, 8704, 9426) == (9216, 0)

    def test_overshoot_falls_back_to_previous(self):
        # last track ran past the key (9472), previous (9216) still valid.
        assert _select(9472, 9216, 9426) == (9216, 1)

    def test_overshoot_without_previous_donates_nothing(self):
        # This is the "return 0 -> nothing cached" path.
        assert _select(9472, None, 9426) is None

    def test_overshoot_when_previous_also_overshoots(self):
        assert _select(9728, 9472, 9426) is None

    def test_overshoot_needs_two_slots(self):
        # A single tracking slot cannot name the other half of the state.
        assert _select(9472, 9216, 9426, buffer_size=1) is None

    def test_untracked_request_donates_zero_length(self):
        # No track point at all -> cache_len 0 -> caller skips the insert.
        assert _select(None, None, 9426) == (0, 0)

    def test_exact_boundary_is_not_an_overshoot(self):
        # last track == key length is still a valid checkpoint.
        assert _select(9472, 9216, 9472) == (9472, 0)

    def test_long_prompt_overlap_is_common(self):
        # 17920-token prompt: tracks land at 17920 exactly, decode pushes past.
        assert _select(17920, 17664, 17920) == (17920, 0)
        assert _select(18176, 17920, 17920) == (17920, 1)
        assert _select(18176, None, 17920) is None
