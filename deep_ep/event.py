"""Fake deep_ep.EventOverlap — lightweight stand-in."""


class EventOverlap:
    """Mimics ``deep_ep.EventOverlap``."""

    def __init__(self, event=None, extra_tensors=None):
        self.event = event
        self.extra_tensors = extra_tensors

    def current_stream_wait(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
