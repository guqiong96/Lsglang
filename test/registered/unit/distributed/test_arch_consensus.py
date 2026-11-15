"""The TP-group architecture consensus helper (sglang.srt.distributed.arch_consensus).

A heterogeneous TP group must reach ONE answer for any arch predicate that
changes what a captured forward puts on the wire; a per-rank answer desyncs the
captures and the group busy-waits. These tests pin the mechanism without GPUs:
the group AND, the cache, the prewarm covering the whole table, the refusal to
resolve inside a CUDA graph, and the rule that no model module keeps its own
copy of the helper.
"""

import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.distributed import arch_consensus
from sglang.srt.distributed.arch_consensus import (
    PREDICATES,
    clear_tp_arch_predicates,
    prewarm_tp_arch_predicates,
    resolved_tp_arch_predicates,
    tp_all,
)
from sglang.test.ci.ci_register import register_cpu_ci
import sys

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_tp_arch_predicates()
    yield
    clear_tp_arch_predicates()


def _world(monkeypatch, *, world_size, peer_says_yes=True, capturing=False):
    """A TP group whose OTHER ranks agree (peer_says_yes) or veto."""
    calls = {"all_reduce": 0}

    def fake_all_reduce(tensor, group=None):
        calls["all_reduce"] += 1
        peers = world_size - 1
        yes = int(tensor.item()) + (peers if peer_says_yes else 0)
        tensor.fill_(yes)

    monkeypatch.setattr(arch_consensus.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(
        arch_consensus.torch.distributed, "is_initialized", lambda: world_size > 0
    )
    monkeypatch.setattr(arch_consensus.torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(
        arch_consensus,
        "_flag_device",
        lambda: "cpu",
    )
    monkeypatch.setattr(arch_consensus, "_capturing", lambda: capturing)
    import sglang.srt.runtime_context as runtime_context

    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: SimpleNamespace(
            tp_group=SimpleNamespace(world_size=world_size, device_group=object())
        ),
    )
    return calls


def _local_yes(monkeypatch, name="blackwell"):
    monkeypatch.setitem(PREDICATES, name, lambda: True)


def test_one_rank_veto_turns_a_local_yes_off(monkeypatch):
    """A mixed group takes the common denominator: Blackwell-only paths off."""
    _local_yes(monkeypatch)
    _world(monkeypatch, world_size=4, peer_says_yes=False)
    assert tp_all("blackwell") is False


def test_group_of_yes_keeps_the_local_yes(monkeypatch):
    _local_yes(monkeypatch)
    _world(monkeypatch, world_size=4)
    assert tp_all("blackwell") is True


def test_single_rank_group_returns_the_local_value(monkeypatch):
    """Homogeneous hosts must see no change at all, so no collective and no
    veto from a group that does not exist."""
    _local_yes(monkeypatch)
    calls = _world(monkeypatch, world_size=1)
    assert tp_all("blackwell") is True
    assert calls["all_reduce"] == 0


def test_cache_resolves_each_predicate_once(monkeypatch):
    _local_yes(monkeypatch)
    calls = _world(monkeypatch, world_size=2)
    for _ in range(5):
        tp_all("blackwell")
    assert calls["all_reduce"] == 1


def test_prewarm_resolves_the_whole_table(monkeypatch):
    """The prewarm is the only thing keeping a consensus all-reduce out of
    capture, so it has to walk PREDICATES -- not a hand-written list that can
    fall behind the table."""
    for name in PREDICATES:
        monkeypatch.setitem(PREDICATES, name, lambda: True)
    _world(monkeypatch, world_size=2)
    prewarm_tp_arch_predicates()
    assert set(resolved_tp_arch_predicates()) == set(PREDICATES)


def test_unknown_predicate_names_the_table(monkeypatch):
    _world(monkeypatch, world_size=1)
    with pytest.raises(KeyError, match="PREDICATES"):
        tp_all("sm140")


def test_refuses_to_resolve_while_capturing(monkeypatch):
    """A collective issued inside a capture is the deadlock these tests exist
    to prevent; say so loudly instead of hanging the group."""
    _local_yes(monkeypatch)
    _world(monkeypatch, world_size=2, capturing=True)
    with pytest.raises(RuntimeError, match="prewarm_tp_arch_predicates"):
        tp_all("blackwell")


def test_a_resolved_answer_is_returned_even_while_capturing(monkeypatch):
    _local_yes(monkeypatch)
    _world(monkeypatch, world_size=2)
    assert tp_all("blackwell") is True
    monkeypatch.setattr(arch_consensus, "_capturing", lambda: True)
    assert tp_all("blackwell") is True


@pytest.mark.parametrize(
    "module",
    [
        "sglang.srt.models.deepseek_v4",
        "sglang.srt.models.deepseek_v2",
        "sglang.srt.models.deepseek_v4_dspark",
    ],
)
def test_no_model_module_keeps_its_own_consensus(module):
    """Two copies drifted once already (the v2 table only knew "blackwell"),
    and a per-module cache means per-module orderings of the anonymous
    consensus all-reduce. One helper, imported."""
    source = inspect.getsource(importlib.import_module(module))
    assert "def _tp_all" not in source, f"{module} re-implements the consensus"
    assert "_TP_ARCH_CACHE" not in source, f"{module} caches the decision itself"
    # An arch probe defined in a model module is the same drift one step earlier:
    # it is what PREDICATES entries are for (deepseek_v4 kept an _IS_SM8 like this
    # until the table subsumed it).
    for probe in ("def _IS_SM8", "def _is_sm8", "def _blackwell"):
        assert probe not in source, f"{module} defines its own predicate {probe}(...)"


def test_dspark_stream_gates_use_the_consensus():
    """The draft model decides per batch size whether a captured graph overlaps
    its KV store on a second stream; that is graph structure, so it is the
    group's call, not the rank's."""
    source = inspect.getsource(
        importlib.import_module("sglang.srt.models.deepseek_v4_dspark")
    )
    assert '_multi_stream_bs_limit = 128 if _tp_all("blackwell") else 64' in source

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
