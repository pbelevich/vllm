"""
Fake deep_ep.Buffer — noop stand-in for the real DeepEP Buffer.

Every method that vLLM (or its tests) calls is implemented here with
the correct signature and return-type shape.  No actual communication
or CUDA kernels are invoked.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from deep_ep.config import Config
from deep_ep.event import EventOverlap


def _noop_hook():
    """Default recv hook — does nothing."""
    pass


class Buffer:
    """Mock ``deep_ep.Buffer``."""

    # Class-level attribute mirroring the real Buffer
    num_sms: int = 20

    def __init__(
        self,
        group=None,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        low_latency_mode: bool = False,
        num_qps_per_rank: int = 24,
        allow_nvlink_for_low_latency_mode: bool = True,
        allow_mnnvl: bool = False,
        use_fabric: bool = False,
        explicitly_destroy: bool = False,
        enable_shrink: bool = False,
        comm=None,
    ) -> None:
        self.group = group
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.num_qps_per_rank = num_qps_per_rank
        self.allow_nvlink_for_low_latency_mode = allow_nvlink_for_low_latency_mode
        self.allow_mnnvl = allow_mnnvl
        self.use_fabric = use_fabric
        self.explicitly_destroy = explicitly_destroy
        self.enable_shrink = enable_shrink
        self.comm = comm

        # Determine world-size / rank from the group (best-effort).
        if group is not None:
            try:
                import torch.distributed as dist

                self.num_ranks = dist.get_world_size(group)
                self.rank = dist.get_rank(group)
            except Exception:
                self.num_ranks = 1
                self.rank = 0
        else:
            self.num_ranks = 1
            self.rank = 0

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        Buffer.num_sms = new_num_sms

    @staticmethod
    def capture() -> EventOverlap:
        return EventOverlap()

    @staticmethod
    def is_sm90_compiled() -> bool:
        return False

    @staticmethod
    def get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_ranks: int,
        num_experts: int,
    ) -> int:
        # Return a reasonable default — 1 GiB.
        return 1024 * 1024 * 1024

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        return Config()

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        return Config()

    # ------------------------------------------------------------------
    # Instance lifecycle
    # ------------------------------------------------------------------
    def destroy(self) -> None:
        pass

    # ------------------------------------------------------------------
    # High-throughput path
    # ------------------------------------------------------------------
    def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        EventOverlap,
    ]:
        num_tokens = topk_idx.size(0)
        device = topk_idx.device

        # num_tokens_per_rank: [num_ranks]
        num_tokens_per_rank = torch.zeros(
            self.num_ranks, dtype=torch.int32, device=device
        )
        num_tokens_per_rank[self.rank] = num_tokens

        # num_tokens_per_rdma_rank — None for intranode
        num_tokens_per_rdma_rank = None

        # num_tokens_per_expert: [num_experts]
        num_tokens_per_expert = torch.zeros(
            num_experts, dtype=torch.int32, device=device
        )
        flat_ids = topk_idx.view(-1)
        for eid in range(num_experts):
            num_tokens_per_expert[eid] = int((flat_ids == eid).sum().item())

        # is_token_in_rank: [num_tokens, num_ranks]
        is_token_in_rank = torch.zeros(
            num_tokens, self.num_ranks, dtype=torch.bool, device=device
        )
        is_token_in_rank[:, self.rank] = True

        event = EventOverlap()
        return (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        )

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[Tuple] = None,
        num_tokens_per_rank: Optional[torch.Tensor] = None,
        num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
        is_token_in_rank: Optional[torch.Tensor] = None,
        num_tokens_per_expert: Optional[torch.Tensor] = None,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple:
        """
        Returns:
            (recv_x, recv_topk_idx, recv_topk_weights,
             num_recv_tokens_per_expert_list, handle, event)
        """
        # recv_x — pass x through unchanged
        recv_x = x

        # recv_topk_idx / recv_topk_weights — pass through
        recv_topk_idx = topk_idx
        recv_topk_weights = topk_weights

        # Build per-expert token count list
        if num_tokens_per_expert is not None:
            num_recv_tokens_per_expert_list = num_tokens_per_expert.tolist()
        else:
            num_recv_tokens_per_expert_list = []

        # Handle — reuse or create a dummy one
        if handle is None:
            handle = ()

        event = EventOverlap()
        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        )

    def combine(
        self,
        x: torch.Tensor,
        handle: Tuple,
        topk_weights: Optional[torch.Tensor] = None,
        bias=None,
        config: Optional[Config] = None,
        previous_event: Optional[EventOverlap] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Returns:
            (combined_x, combined_topk_weights, event)
        """
        return x, topk_weights, EventOverlap()

    # Internode variants — same signatures as dispatch/combine
    def internode_dispatch(self, *args, **kwargs):
        return self.dispatch(*args, **kwargs)

    def internode_combine(self, *args, **kwargs):
        return self.combine(*args, **kwargs)

    # ------------------------------------------------------------------
    # Low-latency path
    # ------------------------------------------------------------------
    def clean_low_latency_buffer(
        self,
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
    ) -> None:
        pass

    def low_latency_dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
        use_fp8: bool = True,
        round_scale: bool = False,
        use_ue8m0: bool = False,
        use_nvfp4: bool = False,
        x_global_scale: Optional[torch.Tensor] = None,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ) -> Tuple:
        """
        Returns:
            (recv_x, recv_count, handle, event, hook)
        where recv_x is (tensor, scales) when use_fp8=True, else tensor.
        """
        device = x.device
        num_tokens, hidden = x.size()
        num_local_experts = num_experts // max(self.num_ranks, 1)

        if use_nvfp4:
            # nvfp4: recv_x is (packed_half, scales)
            # packed_half has shape [num_local_experts, max_tokens_per_rank, hidden//2]
            packed = torch.zeros(
                num_local_experts,
                num_max_dispatch_tokens_per_rank,
                hidden // 2,
                dtype=torch.uint8,
                device=device,
            )
            # scales shape: depends on blocking
            scales = torch.ones(
                num_local_experts,
                num_max_dispatch_tokens_per_rank,
                (hidden + 15) // 16,
                dtype=torch.float32,
                device=device,
            )
            recv_x = (packed, scales)
        elif use_fp8:
            # FP8: recv_x is (fp8_tensor, scales)
            fp8_tensor = torch.zeros(
                num_local_experts,
                num_max_dispatch_tokens_per_rank,
                hidden,
                dtype=torch.float8_e4m3fn,
                device=device,
            )
            block_size = 128
            num_scale_blocks = (hidden + block_size - 1) // block_size
            scales = torch.ones(
                num_local_experts,
                num_max_dispatch_tokens_per_rank,
                num_scale_blocks,
                dtype=torch.float32,
                device=device,
            )
            recv_x = (fp8_tensor, scales)
        else:
            # BF16: recv_x is just a tensor
            recv_x = torch.zeros(
                num_local_experts,
                num_max_dispatch_tokens_per_rank,
                hidden,
                dtype=x.dtype,
                device=device,
            )

        # recv_count: how many tokens each local expert received
        recv_count = torch.zeros(
            num_local_experts, dtype=torch.int32, device=device
        )

        # Handle — tuple matching the real DeepEP structure
        handle = (None, None, num_max_dispatch_tokens_per_rank, hidden, num_experts)

        event = EventOverlap()
        hook = _noop_hook

        return recv_x, recv_count, handle, event, hook

    def low_latency_combine(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        use_logfmt: bool = False,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        out: Optional[torch.Tensor] = None,
        combine_wait_recv_cost_stats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, EventOverlap, callable]:
        """
        Returns:
            (combined_x, event, hook)
        """
        num_combined_tokens = topk_idx.size(0)
        # Recover hidden from handle or from x
        if handle and len(handle) >= 4:
            hidden = handle[3]
        else:
            hidden = x.size(-1)

        if out is not None:
            # Write zeros into the provided output buffer
            out.zero_()
            combined_x = out
        else:
            combined_x = torch.zeros(
                num_combined_tokens, hidden, dtype=torch.bfloat16, device=x.device
            )

        event = EventOverlap()
        hook = _noop_hook

        return combined_x, event, hook

    # ------------------------------------------------------------------
    # Mask / shrink helpers (not used by vLLM, but stub them out)
    # ------------------------------------------------------------------
    def low_latency_update_mask_buffer(
        self, rank_to_mask: int, mask: bool = False
    ) -> None:
        pass

    def low_latency_query_mask_buffer(self, mask_status: torch.Tensor) -> None:
        pass

    def low_latency_clean_mask_buffer(self) -> None:
        pass

    def get_next_low_latency_combine_buffer(self, handle: object) -> torch.Tensor:
        if handle and len(handle) >= 5:
            num_max_dispatch_tokens_per_rank = handle[2]
            hidden = handle[3]
            num_experts = handle[4]
            num_local_experts = num_experts // max(self.num_ranks, 1)
            return torch.zeros(
                num_local_experts,
                self.num_ranks * num_max_dispatch_tokens_per_rank,
                hidden,
                dtype=torch.bfloat16,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        return torch.empty(0, dtype=torch.bfloat16)

    def get_comm_stream(self) -> torch.cuda.Stream:
        return torch.cuda.current_stream()

    def get_local_buffer_tensor(
        self,
        dtype: torch.dtype,
        size=None,
        offset: int = 0,
        use_rdma_buffer: bool = False,
    ) -> torch.Tensor:
        if size is not None:
            return torch.zeros(size, dtype=dtype)
        return torch.empty(0, dtype=dtype)
