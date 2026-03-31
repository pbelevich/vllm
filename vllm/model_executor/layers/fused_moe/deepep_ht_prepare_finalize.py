# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepEP High-Throughput noop implementation.

This replaces the actual DeepEP dispatch/combine communication with no-ops
that return correctly shaped tensors.  Useful for benchmarking the compute
portion of MoE without any inter-rank data movement.
"""
from collections.abc import Callable

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils.math_utils import round_up


# ---------------------------------------------------------------------------
# Lightweight stand-ins for deep_ep types so the rest of the code can
# reference them without importing the real package.
# ---------------------------------------------------------------------------
class _NoopEventOverlap:
    """Mimics deep_ep.EventOverlap with no actual event."""

    def __init__(self):
        self.event = None

    def current_stream_wait(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _NoopBuffer:
    """Mimics the subset of deep_ep.Buffer used by the HT prepare/finalize."""

    def capture(self, *args, **kwargs):
        return _NoopEventOverlap()


class DeepEPHTPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Prepare/Finalize using DeepEP High-Throughput kernels — **noop variant**.

    dispatch() and combine() are replaced with identity operations that
    return tensors of the correct shape/dtype without any communication.
    """

    @staticmethod
    def maybe_roundup_layer_hidden_size(hidden_size: int, dtype: torch.dtype) -> int:
        hidden_size_bytes = hidden_size * dtype.itemsize
        xfer_atom_size = 512  # 32 * 16 (size(int4))
        if hidden_size_bytes % xfer_atom_size == 0:
            return hidden_size
        hidden_size_bytes = round_up(hidden_size_bytes, xfer_atom_size)
        return hidden_size_bytes // dtype.itemsize

    def __init__(
        self,
        buffer,  # accepts real deep_ep.Buffer or _NoopBuffer
        num_dispatchers: int,
        dp_size: int,
        rank_expert_offset: int,
    ):
        super().__init__()
        self.buffer = buffer
        self.num_dispatchers_ = num_dispatchers
        self.dp_size = dp_size
        self.rank_expert_offset = rank_expert_offset
        self.async_prepare = True
        self.handles: list[tuple | None] = [None, None]
        self.available_rank_configs = [2, 4, 8, 16, 24, 32, 64, 128, 144, 160]

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    # ------------------------------------------------------------------
    # Noop dispatch — returns input data with correct shapes, no comms
    # ------------------------------------------------------------------
    def _do_dispatch(
        self,
        tokens: torch.Tensor,
        token_scales: torch.Tensor | None,
        rank_topk_ids: torch.Tensor,
        rank_topk_weights: torch.Tensor,
        num_experts: int,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> Callable:
        num_tokens = tokens.size(0)
        hidden = tokens.size(1)
        num_topk = rank_topk_ids.size(1)
        num_local_experts = num_experts // self.num_dispatchers_

        # ----- Simulate get_dispatch_layout -----
        # In a real run every token goes to num_topk experts.  For the noop
        # we pretend all tokens stay local.
        num_tokens_per_rank = torch.zeros(
            self.num_dispatchers_, dtype=torch.int32, device=tokens.device
        )
        # Put all tokens on rank-0 (self) so the rest of the logic still works.
        num_tokens_per_rank[0] = num_tokens

        # ----- Simulate dispatch -----
        # The real dispatch scatters tokens across ranks and returns the
        # received slice.  In the noop we just pass the input through.
        #
        # recv_x shape: [total_recv_tokens, hidden]
        # Since we pretend everything stays local, total_recv = num_tokens.
        recv_x = tokens  # [num_tokens, hidden]
        recv_x_scales = token_scales  # may be None

        if recv_x_scales is not None:
            token_data = (recv_x, recv_x_scales)
        else:
            token_data = recv_x

        # expert_topk_ids: the original topk_ids offset to local expert space
        expert_topk_ids = rank_topk_ids

        # expert_topk_weights: pass through
        expert_topk_weights = rank_topk_weights

        # expert_num_tokens_per_expert_list: how many tokens each local expert
        # receives.  Compute from topk_ids.
        expert_num_tokens_per_expert_list = [0] * num_local_experts
        if num_tokens > 0:
            flat_ids = rank_topk_ids.view(-1)
            for eid in range(num_local_experts):
                global_eid = eid + self.rank_expert_offset
                expert_num_tokens_per_expert_list[eid] = int(
                    (flat_ids == global_eid).sum().item()
                )

        # handle — combine needs it; store a simple tuple with enough info.
        handle = (num_tokens, hidden, num_topk)

        # Fake event
        event = _NoopEventOverlap()

        # Record handle for combine
        from vllm.v1.worker.ubatching import dbo_current_ubatch_id
        a2a_idx = dbo_current_ubatch_id()
        self.handles[a2a_idx] = handle

        return lambda: self._receiver(
            event,
            recv_x_scales is not None,
            token_data,
            expert_topk_ids,
            num_experts,
            expert_num_tokens_per_expert_list,
            expert_topk_weights,
            a1_scale,
            quant_config,
            defer_input_quant=defer_input_quant,
        )

    def _receiver(
        self,
        event: _NoopEventOverlap,
        has_scales: bool,
        token_data: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
        expert_topk_ids: torch.Tensor | None,
        num_experts: int,
        expert_num_tokens_per_expert_list: list[int],
        expert_topk_weights: torch.Tensor | None,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> mk.PrepareResultType:
        if has_scales:
            expert_x, expert_x_scale = token_data[0], token_data[1]
        else:
            expert_x, expert_x_scale = token_data, None

        assert expert_topk_ids is not None
        expert_topk_ids = torch.where(
            expert_topk_ids == -1,
            num_experts - 1 if self.rank_expert_offset == 0 else 0,
            expert_topk_ids + self.rank_expert_offset,
        )

        expert_tokens_meta = mk.ExpertTokensMetadata.make_from_list(
            expert_num_tokens_per_expert_list, device=expert_x.device
        )

        if not quant_config.is_block_quantized and not defer_input_quant:
            expert_x_scale = None
            if expert_x.numel() != 0:
                expert_x, expert_x_scale = moe_kernel_quantize_input(
                    expert_x,
                    a1_scale,
                    quant_dtype=quant_config.quant_dtype,
                    per_act_token_quant=False,
                    block_shape=quant_config.block_shape,
                    is_fp4_scale_swizzled=quant_config.is_nvfp4_scale_swizzled,
                )

        return (
            expert_x,
            expert_x_scale,
            expert_tokens_meta,
            expert_topk_ids,
            expert_topk_weights,
        )

    def supports_async(self) -> bool:
        return True

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.ReceiverType:
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        if quant_config.is_block_quantized and not defer_input_quant:
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                quant_config.a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
            )
            if a1q_scale is not None and a1q_scale.numel() == 1:
                a1q_scale = a1q_scale.view(1, 1)
            a1_post_scale = None
        else:
            a1q = a1
            a1q_scale = None
            a1_post_scale = (
                quant_config.a1_gscale
                if quant_config.quant_dtype == "nvfp4"
                else quant_config.a1_scale
            )

        return self._do_dispatch(
            tokens=a1q,
            token_scales=a1q_scale,
            rank_topk_ids=topk_ids,
            rank_topk_weights=topk_weights,
            num_experts=num_experts,
            a1_scale=a1_post_scale,
            quant_config=quant_config,
            defer_input_quant=defer_input_quant,
        )

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        receiver = self.prepare_async(
            a1, topk_weights, topk_ids, num_experts, expert_map,
            apply_router_weight_on_input, quant_config, defer_input_quant,
        )
        return receiver()

    # ------------------------------------------------------------------
    # Noop combine — returns expert output directly, no comms
    # ------------------------------------------------------------------
    def _finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> Callable | None:
        # Apply topk weight reduction the same way the real version does.
        if fused_expert_output.numel() != 0:
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            fused_expert_output = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        # Noop combine: the real combine would scatter-reduce across ranks.
        # In the noop we just copy the reduced expert output to the output
        # tensor (same as the real code's final step).
        assert fused_expert_output.dtype == torch.bfloat16, (
            f"Expected fused_expert_output bfloat16, got {fused_expert_output.dtype}"
        )
        combined_x = fused_expert_output

        if do_async:

            def _receiver():
                output.copy_(combined_x, non_blocking=True)

            return _receiver
        else:
            output.copy_(combined_x, non_blocking=True)
            return None

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable:
        receiver = self._finalize(
            output, fused_expert_output, topk_weights, topk_ids,
            apply_router_weight_on_input, weight_and_reduce_impl, True,
        )
        assert receiver is not None
        return receiver

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        self._finalize(
            output, fused_expert_output, topk_weights, topk_ids,
            apply_router_weight_on_input, weight_and_reduce_impl, False,
        )
