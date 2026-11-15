"""VidFlow real-time clipper.

Follows a live (or simulated live) stream and emits clips while it is still
running. See ``docs/realtime/DESIGN.md``.

This package must not import the legacy batch pipeline (``VidFlow.pipes``,
``VidFlow.modules``, ``VidFlow.aggregate``, ``VidFlow.cli``). Only the
dependency-free contract is re-exported here; import component modules
(``VidFlow.realtime.media``, ``.buffer``, ...) directly.
"""

from .config import ConfigError, RealtimeConfig
from .types import (
    Clip,
    ClipError,
    Detector,
    LoudnessSample,
    Segment,
    Source,
    Window,
    make_window_id,
)

__all__ = [
    "Clip",
    "ClipError",
    "ConfigError",
    "Detector",
    "LoudnessSample",
    "RealtimeConfig",
    "Segment",
    "Source",
    "Window",
    "make_window_id",
]
