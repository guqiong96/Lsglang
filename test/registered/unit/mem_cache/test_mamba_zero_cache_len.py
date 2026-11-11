"""Offline test: a component that reports cache_len 0 must stop the whole insert.

`checkpoint_into_tree` takes the minimum of every component's cache length; when
that minimum is <= 0 it cleans up and returns before `self.insert(...)`, so the
request contributes nothing to the radix cache. A mamba request whose finished
checkpoint cannot be named (`_select_finished_checkpoint -> None`) lands here.

Needs the unified-cache fixture (CUDA), but no serving process.
"""

import importlib.util
from pathlib import Path
from array import array
from unittest import mock

import pytest
import torch

PATH = str(Path(__file__).with_name("test_unified_radix_cache_unittest.py"))


@pytest.fixture(scope="module")
def uc():
    spec = importlib.util.spec_from_file_location("uc", PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cache fixture needs CUDA")
def test_zero_cache_len_skips_the_insert(uc):
    CT = uc.ComponentType
    cfg = uc.CacheConfig(
        components=(CT.FULL, CT.MAMBA),
        mamba_cache_size=11,
        enable_mamba_extra_buffer=True,
        kv_size=8192,
        max_context_len=4096,
    )
    cache, alloc, rtp = uc.build_fixture(cfg)

    req = uc.Req(
        rid="zero",
        origin_input_text="",
        origin_input_ids=array("q", list(range(1, 257))),
        sampling_params=uc.SamplingParams(temperature=0, max_new_tokens=1),
    )
    rtp.alloc([req])
    tokens = list(range(1, 257))

    # Give the request real KV rows so the insert path would otherwise run.
    value = alloc.alloc(len(tokens))
    rtp.req_to_token[req.kv.req_pool_idx][: len(tokens)] = torch.tensor(
        value[: len(tokens)], dtype=rtp.req_to_token.dtype
    )
    req.full_untruncated_fill_ids = array("q", tokens)
    req.kv.cache_protected_len = 0
    req.kv.cache_inserted_len = 0
    req.priority = 0
    req.session_id = None
    req.kv_rotation_base = None
    req.extra_key = None
    req.cache_salt = None

    # A component that cannot name a finished checkpoint returns 0.
    for comp in cache._components_tuple:
        comp.prepare_for_caching_req = mock.Mock(return_value=0)

    with mock.patch.object(type(cache), "insert", autospec=True) as insert:
        cache.checkpoint_into_tree(req, up_to=len(tokens))

    assert not insert.called, "a zero cache length must not reach cache.insert()"
