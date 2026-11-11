"""mHC consumer ownership, replicated shared add, and source row contracts."""

import unittest
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
from sglang.srt.layers.layer_boundary.layout import Layout, SumGroup
from sglang.srt.layers.layer_boundary.row_transport import TransportRows
from sglang.srt.models import deepseek_v4_boundary as boundary
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.models.deepseek_common import moe_transport
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHcConsumer(unittest.TestCase):
    def test_dspark_tap_hands_materialized_stream_to_next_layer(self):
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models import deepseek_v4

        class Layer(torch.nn.Module):
            engram = None
            boundary_mhc_fusions = False
            boundary_pre_from_prev = True
            hc_cfg = mhc.HcConfig(2, 4, 1e-5, 1e-6, 4, True, False)

            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, hidden_states, forward_batch, **kwargs):
                stream = residual_batch.stream_of(forward_batch)
                stream.check(hidden_states)
                update = boundary.HcUpdate(self, False)
                update.post = lambda value, state, **kw: mhc.HcState(
                    state.residual + value, torch.ones(3, 2)
                )
                return stream.record(torch.full((3, 2, 4), self.value), update)

        model = deepseek_v4.DeepseekV4Model.__new__(deepseek_v4.DeepseekV4Model)
        torch.nn.Module.__init__(model)
        model.layers = torch.nn.ModuleList([Layer(1.0), Layer(2.0)])
        model.start_layer, model.end_layer = 0, 2
        model.pp_group = SimpleNamespace(world_size=1)
        model.engram_hasher = model.late_layer_start = None
        # The loop reads the config for the pre-engram vision select (8fa2d810f5).
        # This is a text batch, so keep the select off rather than fake a vision
        # config the batch would not match.
        model.config = SimpleNamespace(
            model_type="deepseek_v4", vision_n_layers=0, image_token_id=-1
        )
        model.dspark_layers_to_capture = [1]
        positions = torch.arange(3)
        batch = SimpleNamespace(
            positions=positions,
            input_ids=positions,
            forward_mode=ForwardMode.EXTEND,
            residual_stream=None,
        )
        source = torch.arange(24.0).reshape(3, 2, 4)
        # The collector is what DeepseekV4Model.forward hands in; a bare list has
        # no capture() and the tap goes through it.
        aux = AuxHiddenStateList()
        recorder = SimpleNamespace(with_current_layer=lambda _: nullcontext())
        with (
            patch.object(deepseek_v4, "_is_cuda", True),
            patch.object(deepseek_v4, "_is_hip", False),
            patch.object(deepseek_v4, "is_cp_active", return_value=False),
            patch.object(deepseek_v4, "check_cuda_graph_backend", return_value=False),
            patch.object(
                deepseek_v4,
                "get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
        ):
            output, pre, tail = model._forward_layers_hc_pre_from_prev(
                positions, source, batch, positions, positions, True, aux
            )
        torch.testing.assert_close(output, source + 3)
        torch.testing.assert_close(aux[0], (source + 1).mean(1))
        torch.testing.assert_close(pre, torch.ones(3, 2))
        self.assertIsNone(tail)
        self.assertIsNone(batch.residual_stream)

    def test_replicated_shared_is_added_after_the_sum(self):
        routed = torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16)
        other = torch.tensor([[-1.0, -0.5]], dtype=torch.bfloat16)
        shared = torch.tensor([[0.01, 0.003]], dtype=torch.bfloat16)
        pieces = SimpleNamespace(
            get_merged=lambda: routed, shared=shared, shared_is_replicated=True
        )
        with (
            patch(
                "sglang.srt.layers.moe.post_experts_all_reduce",
                side_effect=lambda x: x + other,
            ) as reduce,
            patch(
                "sglang.srt.layers.moe.utils.should_add_replicated_moe_output",
                return_value=True,
            ),
        ):
            output = boundary.MoeFinalize(pieces).complete()
        reduce.assert_called_once_with(routed)
        torch.testing.assert_close(output, (routed + other) + shared, atol=0, rtol=0)
        self.assertFalse(torch.equal(output, (routed + shared) + other))


class TestAttentionCompletion(unittest.TestCase):
    def test_projection_and_deferred_sum_preserve_full_precision_policy(self):
        from sglang.srt.layers import linear
        from sglang.srt.layers.moe import utils as moe_utils
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models import deepseek_v4
        from sglang.srt.runtime_context import get_forward

        group = SimpleNamespace(all_reduce=Mock(side_effect=lambda value: value * 2))
        parallel = SimpleNamespace(tp_size=4, tp_group=group, attn_tp_group=group)

        class Projection:
            tp_group = group
            input_is_parallel = True
            bias = None
            skip_bias_add = False
            reduce_results = False
            use_decode_attn_tp = False
            use_dp_attention_reduce = False
            parallel_group = "attn_tp"
            quant_method = SimpleNamespace(apply=lambda layer, value, bias: value + 1)

            def __call__(self, value, **kwargs):
                return linear.RowParallelLinear.forward(self, value, **kwargs)

        attention = deepseek_v4.MQALayer.__new__(deepseek_v4.MQALayer)
        torch.nn.Module.__init__(attention)
        attention.attn_tp_size = 2
        attention.wo_b = Projection()
        layer = SimpleNamespace(boundary_mhc_fusions=False)
        value = torch.arange(12.0).view(3, 4)
        rows = TransportRows(
            Layout(frozenset()),
            Layout(frozenset()),
            Layout(frozenset()),
            SumGroup.ATTN_TP,
        )
        for quant in (False, True):
            for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
                batch = SimpleNamespace(forward_mode=mode)
                with (
                    self.subTest(quant=quant, mode=mode),
                    get_forward().scoped(sp_active=False),
                    patch.object(linear, "get_group_rank_size", return_value=(0, 2)),
                    patch.object(linear, "get_parallel", return_value=parallel),
                    patch.object(
                        linear,
                        "get_exec",
                        return_value=SimpleNamespace(
                            comm=SimpleNamespace(enable_quant_communications=quant)
                        ),
                    ),
                    patch.object(
                        linear,
                        "use_symmetric_memory",
                        side_effect=lambda *a, **k: nullcontext(),
                    ),
                    patch.object(linear, "is_allocation_symmetric", return_value=False),
                    patch.object(
                        linear, "should_skip_mlp_all_reduce", return_value=False
                    ),
                    patch.object(
                        linear,
                        "tensor_model_parallel_all_reduce",
                        side_effect=group.all_reduce,
                    ),
                    patch.object(
                        linear,
                        "tensor_model_parallel_quant_all_reduce",
                        side_effect=AssertionError("unexpected quantized sum"),
                    ),
                    patch.object(
                        moe_utils, "should_skip_mlp_all_reduce", return_value=False
                    ),
                    patch.object(deepseek_v4, "get_parallel", return_value=parallel),
                    patch.object(boundary, "_sum_group", return_value=group),
                    patch.object(
                        deepseek_v4, "attn_tp_all_reduce", side_effect=group.all_reduce
                    ),
                    patch.object(
                        boundary,
                        "get_cp_decode_attn_tp_ctx",
                        return_value=SimpleNamespace(use_decode_attn_tp=False),
                    ),
                ):
                    # The production projection never passes forward_batch to
                    # RowParallelLinear, so this flag does not quantize its sum.
                    with patch.object(deepseek_v4, "_is_cuda", False):
                        immediate = attention._project_wo_b(
                            value, defer_all_reduce=True
                        )
                    with patch.object(deepseek_v4, "_is_cuda", True):
                        deferred = attention._project_wo_b(value, defer_all_reduce=True)
                    transport = boundary.AttentionTransport(
                        layer, value, batch, None, rows
                    )
                    completed = transport.prepare_output(deferred).complete()
                torch.testing.assert_close(completed, immediate, atol=0, rtol=0)
                torch.testing.assert_close(completed, (value + 1) * 2, atol=0, rtol=0)


class TestDsparkCollectorChoice(unittest.TestCase):
    """Which collector a DSPARK forward gets, and what a lost plan then means.

    The packer is sized for the whole prefill batch, so a forward that runs its
    late layers over the decoder-SWA bounded-replay tail hands it capture rows it
    cannot take (2026-11-11: the dsv4.1 2x3090 DSPARK recipes run
    --enable-decoder-swa-bounded-replay *and* --speculative-algorithm DSPARK, and
    the first version of the banded capture built a CAP-row packer for them and
    then raised in the tail). The two decisions are kept in step by one
    predicate, so it gets a truth table, and the capture outcomes on either side
    of it get tested rather than asserted in a comment.
    """

    def _fb(self, mode, return_logprob=False, starts=None, lens=None):
        fb = SimpleNamespace(forward_mode=mode)
        if return_logprob:
            fb.return_logprob = True
            fb.extend_logprob_start_lens_cpu = starts
            fb.extend_seq_lens_cpu = lens
        return fb

    def test_only_a_whole_extend_runs_the_tail(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models.deepseek_v4 import _builds_late_layer_tail

        prompt_logprobs = self._fb(ForwardMode.EXTEND, True, [0], [8])
        completion_logprobs_only = self._fb(ForwardMode.EXTEND, True, [8], [8])
        cases = [
            # bounded replay off: nothing gathers to a tail
            (None, self._fb(ForwardMode.EXTEND), False),
            (None, prompt_logprobs, False),
            # on, and this forward takes the tail
            (20, self._fb(ForwardMode.EXTEND), True),
            (20, completion_logprobs_only, True),
            # on, but the rows are not the extend (decode, verify) or some prompt
            # row is read too (start < len), so no tail
            (20, self._fb(ForwardMode.DECODE), False),
            (20, self._fb(ForwardMode.TARGET_VERIFY), False),
            (20, prompt_logprobs, False),
        ]
        for late, fb, want in cases:
            self.assertEqual(
                _builds_late_layer_tail(late, fb),
                want,
                f"late_layer_start={late} mode={fb.forward_mode}",
            )

    def test_tail_capture_through_the_list_path_keeps_its_rows(self):
        from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models import deepseek_v4

        # The tail rows of one request, four of them, three hc lanes.
        values = torch.randn(4, 3, 6)
        collector = AuxHiddenStateList()
        with patch.object(deepseek_v4, "_super_chunk_aux_bounds", return_value=None):
            deepseek_v4._dspark_capture(collector, values, self._fb(ForwardMode.EXTEND))
        torch.testing.assert_close(
            pack_aux_hidden_states(collector), values.mean(1), atol=0, rtol=0
        )

    def test_packer_without_a_plan_refuses_instead_of_short_writing(self):
        from sglang.srt.layers.aux_hidden_states import AuxHiddenStatePacker
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models import deepseek_v4

        # The other half of the pair: with the packer the same capture raises
        # rather than write 4 rows into a buffer sized for the whole batch. That
        # raise is what the predicate above exists to keep unreachable.
        collector = AuxHiddenStatePacker(1, rows=8)
        with patch.object(deepseek_v4, "_super_chunk_aux_bounds", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "lost its super-chunk plan"):
                deepseek_v4._dspark_capture(
                    collector, torch.randn(4, 3, 6), self._fb(ForwardMode.EXTEND)
                )

    def test_packer_with_a_plan_writes_the_same_bytes_as_the_list(self):
        from sglang.srt.layers.aux_hidden_states import (
            AuxHiddenStatePacker,
            pack_aux_hidden_states,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models import deepseek_v4

        values = torch.randn(4, 3, 6)
        packed = AuxHiddenStatePacker(1, rows=4)
        listed = AuxHiddenStateList()
        with patch.object(
            deepseek_v4, "_super_chunk_aux_bounds", return_value=[(0, 2), (2, 4)]
        ):
            deepseek_v4._dspark_capture(packed, values, self._fb(ForwardMode.EXTEND))
        deepseek_v4._dspark_capture(listed, values, self._fb(ForwardMode.EXTEND))
        self.assertTrue(torch.equal(packed.finalize(), pack_aux_hidden_states(listed)))

    def test_both_decisions_share_the_predicate(self):
        # Drift guard: the layer loop and the collector have to agree, so neither
        # is allowed to re-inline the condition.
        import inspect

        from sglang.srt.models import deepseek_v4

        for fn in (
            deepseek_v4.DeepseekV4Model._forward_impl,
            deepseek_v4.DeepseekV4Model._forward_layers_hc_pre_from_prev,
        ):
            self.assertIn(
                "_builds_late_layer_tail(", inspect.getsource(fn), fn.__name__
            )


class TestMoeSourceRows(unittest.TestCase):
    def test_cp_operand_is_released_before_return_and_v41_uses_contiguous_rows(self):
        from sglang.srt.layers.layer_boundary.layout import TokenAxis

        layer = SimpleNamespace(boundary_mhc_fusions=False, mlp=SimpleNamespace())
        rows = TransportRows(
            Layout.sharded_over(TokenAxis.ATTN_CP, axis_sizes={TokenAxis.ATTN_CP: 2}),
            Layout.sharded_over(axis_sizes={TokenAxis.ATTN_CP: 2}),
            Layout.sharded_over(TokenAxis.ATTN_CP, axis_sizes={TokenAxis.ATTN_CP: 2}),
            SumGroup.MOE_OUTPUT,
        )
        backend = SimpleNamespace(is_none=lambda: True)
        group = SimpleNamespace(all_reduce=lambda value: value * 2)
        parallel = SimpleNamespace(attn_cp_size=2, attn_cp_rank=1, tp_group=group)
        batch = SimpleNamespace(dp_padding_mode=None)
        with (
            patch.object(moe_transport, "get_moe_a2a_backend", return_value=backend),
            patch.object(boundary, "get_moe_a2a_backend", return_value=backend),
            patch.object(moe_transport, "get_parallel", return_value=parallel),
            patch.object(
                moe_transport,
                "attn_cp_interleave_gather",
                side_effect=lambda value: torch.cat((value, value)),
            ),
            patch.object(
                moe_transport,
                "attn_cp_interleave_reduce_scatter",
                side_effect=lambda value: value[1::2] * 2,
            ) as interleave,
        ):
            for kind in ("deepseek_v4", "deepseek_v41"):
                layer.config = SimpleNamespace(model_type=kind)
                call = boundary.MoeTransport(
                    layer, torch.ones(2, 3), batch, boundary.LayerCall(), rows
                )
                reference = weakref.ref(call.input)
                value = torch.arange(12.0).reshape(4, 3)
                output = call.prepare_output(value)
                self.assertIsNone(reference())
                result = call.return_output(output)
                expected = value[1::2] if kind == "deepseek_v4" else value[2:]
                torch.testing.assert_close(result, expected * 2)
                call.release()
        interleave.assert_called_once()


if __name__ == "__main__":
    unittest.main()
