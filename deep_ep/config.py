"""Fake deep_ep.Config — lightweight stand-in."""


class Config:
    """Mimics ``deep_ep.Config`` (opaque perf-tuning config object)."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
