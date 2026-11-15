"""Shared dataclasses and protocols for the real-time clipper (the contract).

These definitions are fixed by task T0 and mirror ``docs/realtime/DESIGN.md``
section 5. Do not change them without updating that document.

Media time
    ``Segment.start`` is the running sum of the durations of the segments the
    source has yielded so far. It is contiguous and strictly increasing even
    when content was lost; a loss is signalled only by ``discontinuity=True``.
    Sources should compute it as ``start = previous.start + previous.duration``
    so that ``previous.end == next.start`` holds exactly in floating point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Mapping, Protocol, runtime_checkable

__all__ = [
    "Segment",
    "LoudnessSample",
    "Window",
    "Clip",
    "Source",
    "Detector",
    "ClipError",
    "make_window_id",
]


@dataclass(frozen=True)
class Segment:
    """One short, self-contained chunk of the stream (about 2 s)."""

    channel: str
    seq: int  # strictly increasing within a session
    start: float  # media time: seconds since the session's first segment
    duration: float
    path: Path  # a self-contained MPEG-TS file starting on a keyframe
    discontinuity: bool = False  # content was lost between the previous segment and this one
    arrived_at: float = 0.0  # time.monotonic() when the source released it

    @property
    def end(self) -> float:
        """Media time at which this segment ends (``start + duration``)."""
        return self.start + self.duration


@dataclass(frozen=True)
class LoudnessSample:
    """RMS level of one hop of audio."""

    t: float  # media time of the start of the hop
    db: float  # RMS level in dBFS over [t, t + hop)


def make_window_id(channel: str, start: float) -> str:
    """Return the canonical ``Window.window_id``: ``f"{channel}-{start_ms:09d}"``.

    ``start_ms`` is the window start in whole milliseconds, rounded to nearest.
    """
    start_ms = int(round(start * 1000.0))
    return f"{channel}-{start_ms:09d}"


@dataclass(frozen=True)
class Window:
    """A detected moment, in media time."""

    channel: str
    window_id: str  # f"{channel}-{start_ms:09d}", see make_window_id()
    start: float  # media time of the first loud sample
    end: float  # media time of the end of the last loud sample
    detector: str  # "audio" for now
    stats: Mapping[str, float]  # peak_db, mean_db, threshold_db, prominence_db
    forced: bool = False  # closed by max length, a discontinuity or end of stream

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Clip:
    """A clip file cut from the buffer for one window."""

    window: Window
    path: Path
    file_start: float  # media time of the first frame in the file
    file_duration: float
    content_start: float  # offset in the file where the window (plus padding asked for) begins
    content_end: float
    # Filled in after scoring; empty until then. Use dataclasses.replace() to attach scores.
    scores: Mapping[str, float] = field(default_factory=dict)


@runtime_checkable
class Source(Protocol):
    """Yields segments of one channel in order."""

    channel: str

    def __iter__(self) -> Iterator[Segment]:
        """Block until the next segment; end at end of stream."""
        ...

    def close(self) -> None:
        """Unblock the iterator promptly. Safe to call from any thread."""
        ...


@runtime_checkable
class Detector(Protocol):
    """Turns segments into windows."""

    def feed(self, segment: Segment) -> List[Window]:
        """Return the windows that closed because of this segment."""
        ...

    def flush(self) -> List[Window]:
        """End of stream: return any windows still open."""
        ...


class ClipError(Exception):
    """Cutting a clip failed. Carries the tail of FFmpeg's stderr."""

    def __init__(self, message: str, stderr_tail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.stderr_tail = stderr_tail

    def __str__(self) -> str:
        if self.stderr_tail:
            return f"{self.message}\n--- ffmpeg stderr (tail) ---\n{self.stderr_tail}"
        return self.message
