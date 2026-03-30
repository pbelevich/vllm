"""
Fake deep_ep package — mocks the real DeepEP API surface used by vLLM.

When the real ``deep_ep`` C++/CUDA package is not installed this shim
lets vLLM start up and exercise all Python-level MoE code paths without
any actual inter-rank communication.

Every public symbol that vLLM touches is re-exported here so that
``import deep_ep`` followed by ``deep_ep.Buffer``, ``deep_ep.Config``,
``deep_ep.EventOverlap``, etc. all resolve.
"""

from deep_ep.buffer import Buffer  # noqa: F401
from deep_ep.event import EventOverlap  # noqa: F401
from deep_ep.config import Config  # noqa: F401

# Re-export the topk index dtype used by vLLM
import torch
topk_idx_t = torch.int64

__all__ = ["Buffer", "EventOverlap", "Config", "topk_idx_t"]
