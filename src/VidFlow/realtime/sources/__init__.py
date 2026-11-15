"""Segment sources for the real-time clipper (design section 6.1).

- :class:`ReplaySource` (``replay.py``): a video file replayed as a paced live feed.
- ``HlsSource`` (``hls.py``): an HLS media-playlist poller. Imported lazily so
  this package does not depend on that module being present.
"""

from __future__ import annotations

from typing import Any

from .replay import ReplaySource

__all__ = ["ReplaySource", "HlsSource"]


def __getattr__(name: str) -> Any:
    if name == "HlsSource":
        from .hls import HlsSource  # noqa: PLC0415 - lazy on purpose

        return HlsSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
