"""Session log (``events.jsonl``) and latency metrics (design section 6.8).

``SessionLog`` writes one JSON object per line to ``<dir>/events.jsonl``. Lines
are written whole under a lock and flushed one by one, so the log is safe to
append to from several threads and safe to read while it is being written:
:func:`read_events` only returns lines that end in a newline.

Every event has ``type``, ``t_media`` and ``t_wall``:

``t_media``
    Stream time the event refers to (seconds since the session's first segment).
``t_wall``
    When the event happened, on the log's clock. The default clock is
    :func:`time.monotonic`, the same clock as ``Segment.arrived_at``, so wall
    times in the log can be subtracted from arrival times directly. The
    ``session_start`` event also records ``t_epoch`` (``time.time()``) next to
    its ``t_wall``, which converts any ``t_wall`` in the log to calendar time.
    Callers may pass ``t_wall=`` explicitly when the event happened earlier
    than the moment it is written.

Event types and their fields (written by the ``log_*`` helpers; ``emit`` can
write any other type too):

=================  =========================================================
``session_start``  ``t_media`` 0. ``channel``, ``session``, ``config`` (the
                   ``RealtimeConfig.to_dict()`` snapshot), ``t_epoch``, extras.
``segment``        ``t_media`` = segment start. ``seq``, ``start``,
                   ``duration``, ``discontinuity``, ``arrived_at`` (``None``
                   when the source did not stamp it), ``path``, ``due_at``
                   (when the broadcast released it, see :class:`ScheduleClock`;
                   ``None`` or absent when unknown). ``t_wall`` is when the
                   ingest loop finished with the segment.
``audio``          ``t_media`` = segment start. ``seq``, ``discontinuity``,
                   ``hop_s``, ``t`` (list of sample times), ``db`` (list of
                   levels; non-finite levels such as silence's ``-inf`` are
                   written as ``null`` and read back as ``-inf``).
``window``         ``t_media`` = window end. ``window_id``, ``channel``,
                   ``start``, ``end``, ``detector``, ``stats``, ``forced``.
                   ``t_wall`` is when the window was closed.
``clip``           ``t_media`` = window end. ``window_id``, ``path``,
                   ``file_start``, ``file_duration``, ``content_start``,
                   ``content_end``, ``cut_s`` (seconds the cut took, or
                   ``None``). ``t_wall`` is when the clip file was complete.
``score``          ``t_media`` = window end. ``window_id``, ``scores``.
``dropped``        ``what`` (for example ``"score"``), ``reason``,
                   ``window_id`` (or ``None``), extras.
``error``          ``what``, ``error`` (message), ``error_type``,
                   ``window_id`` (or ``None``), extras.
``session_end``    ``t_media`` = the largest ``t_media`` logged (unless
                   given). ``counts`` (events per type), ``latency``
                   (:meth:`LatencyTracker.summary`), extras (totals).
=================  =========================================================

``LatencyTracker`` consumes the same events (a ``SessionLog`` feeds its own
``latency`` tracker as it writes) and reports:

``detect_latency``  window ``t_wall`` minus the ``arrived_at`` of the segment
                    containing the window's end.
``cut_latency``     clip ``t_wall`` minus window ``t_wall``.
``total_latency``   ``detect_latency + cut_latency``.
``ingest_lag``      segment ``t_wall`` minus its ``arrived_at``.
``behind_schedule`` segment ``t_wall`` minus its ``due_at``: how late the
                    ingest loop finished with a segment compared with when the
                    broadcast released it.
``time_to_clip``    clip ``t_wall`` minus the ``due_at`` of the segment
                    containing the window's end: ``total_latency`` measured
                    from the broadcast's schedule instead of from arrival.

Why ``behind_schedule``: ``ingest_lag`` cannot show a backlog with pull-based
sources (``ReplaySource``, ``HlsSource``), because ``arrived_at`` is stamped
when the ingest loop asks for the segment; a loop that falls behind sees every
segment "arrive" just as it is ready for it. ``due_at`` is an optional field of
the ``segment`` event, stamped by the pipeline with a :class:`ScheduleClock`;
the two metrics are empty for segments without it (e.g. unpaced replays).
"""

from __future__ import annotations

import bisect
import dataclasses
import datetime as _dt
import enum
import json
import logging
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from .config import RealtimeConfig
from .types import Clip, LoudnessSample, Segment, Window

__all__ = [
    "EVENTS_FILE",
    "EVENT_TYPES",
    "SessionLog",
    "read_events",
    "to_jsonable",
    "samples_from_event",
    "read_loudness",
    "read_config",
    "window_from_event",
    "LatencyTracker",
    "ClipLatency",
    "SegmentLatency",
    "ScheduleClock",
    "LATENCY_METRICS",
]

log = logging.getLogger(__name__)

PathLike = Union[str, PurePath]

EVENTS_FILE = "events.jsonl"

EVENT_TYPES: Tuple[str, ...] = (
    "session_start",
    "segment",
    "audio",
    "window",
    "clip",
    "score",
    "dropped",
    "error",
    "session_end",
)

_RESERVED = ("type", "t_media", "t_wall")


# --------------------------------------------------------------------------- JSON


def to_jsonable(obj: Any) -> Any:
    """Convert ``obj`` into something :func:`json.dumps` accepts with ``allow_nan=False``.

    Handles paths (as strings), NumPy scalars and arrays, dataclasses, mappings
    (keys become strings), sequences and sets, enums, dates, and non-finite
    floats (written as ``None``). Anything else unknown becomes ``str(obj)``,
    so logging never fails on an odd value.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return to_jsonable(obj.value)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(to_jsonable(k)) if not isinstance(k, str) else k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        items = [to_jsonable(v) for v in obj]
        try:
            return sorted(items)
        except TypeError:
            return items
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        return obj.isoformat()
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj).decode("utf-8", errors="replace")
    return str(obj)


def _dumps(event: Mapping[str, Any]) -> bytes:
    text = json.dumps(to_jsonable(event), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return (text + "\n").encode("utf-8")


# --------------------------------------------------------------------------- writer


class SessionLog:
    """Append-only ``events.jsonl`` for one channel session.

    ``SessionLog(dir)`` creates ``dir`` if needed and appends to
    ``dir/events.jsonl``. Safe to use from several threads. ``clock`` supplies
    ``t_wall`` (default :func:`time.monotonic`, the clock of
    ``Segment.arrived_at``). Every event written is also fed to
    :attr:`latency`, a :class:`LatencyTracker`.

    Events emitted after :meth:`close` are dropped with a warning rather than
    raising, so a late worker job cannot fail because the session ended.
    """

    def __init__(self, dir: PathLike, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / EVENTS_FILE
        self.clock = clock
        self.latency = LatencyTracker()
        self._lock = threading.Lock()
        self._fh = open(self.path, "ab")
        self._closed = False
        self._counts: Counter = Counter()
        self._max_t_media = 0.0

    # ------------------------------------------------------------------ core

    @property
    def closed(self) -> bool:
        return self._closed

    def emit(
        self, type: str, t_media: float, /, *, t_wall: Optional[float] = None, **fields: Any
    ) -> Optional[Dict[str, Any]]:
        """Write one event and return it as written (``None`` if the log is closed).

        ``t_wall`` defaults to ``clock()`` now. ``fields`` may hold paths,
        NumPy values and dataclasses; they are converted by :func:`to_jsonable`.
        """
        if not isinstance(type, str) or not type:
            raise ValueError(f"event type must be a non-empty string, got {type!r}")
        clash = [k for k in fields if k in _RESERVED]
        if clash:
            raise ValueError(f"field names {clash} are reserved")
        if t_wall is None:
            t_wall = self.clock()
        event: Dict[str, Any] = {"type": type, "t_media": t_media, "t_wall": t_wall}
        event.update(fields)
        event = to_jsonable(event)
        line = _dumps(event)
        with self._lock:
            if self._closed:
                log.warning("SessionLog %s is closed; dropping %r event", self.path, type)
                return None
            self._fh.write(line)
            self._fh.flush()
            self._counts[type] += 1
            tm = event.get("t_media")
            if isinstance(tm, (int, float)) and tm > self._max_t_media:
                self._max_t_media = float(tm)
            self.latency.feed(event)
        return event

    def counts(self) -> Dict[str, int]:
        """Number of events written so far, per type."""
        with self._lock:
            return dict(self._counts)

    def close(self) -> None:
        """Close the file. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._fh.close()

    def __enter__(self) -> "SessionLog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # --------------------------------------------------------------- helpers

    def log_session_start(
        self,
        config: Union[RealtimeConfig, Mapping[str, Any], None] = None,
        *,
        channel: Optional[str] = None,
        session: Optional[str] = None,
        **extra: Any,
    ) -> Optional[Dict[str, Any]]:
        cfg: Any = config.to_dict() if isinstance(config, RealtimeConfig) else config
        return self.emit(
            "session_start",
            0.0,
            channel=channel,
            session=session,
            config=cfg,
            t_epoch=time.time(),
            **extra,
        )

    def log_segment(
        self, segment: Segment, *, ingested_at: Optional[float] = None, **extra: Any
    ) -> Optional[Dict[str, Any]]:
        """Log a segment. Call when the ingest loop is done with it, or pass ``ingested_at``."""
        return self.emit(
            "segment",
            segment.start,
            t_wall=ingested_at,
            seq=segment.seq,
            start=segment.start,
            duration=segment.duration,
            discontinuity=segment.discontinuity,
            arrived_at=segment.arrived_at if segment.arrived_at else None,
            path=segment.path,
            **extra,
        )

    def log_audio(
        self,
        segment: Segment,
        samples: Sequence[LoudnessSample],
        *,
        hop_s: Optional[float] = None,
        **extra: Any,
    ) -> Optional[Dict[str, Any]]:
        """Log the loudness samples extracted from one segment (may be empty)."""
        return self.emit(
            "audio",
            segment.start,
            seq=segment.seq,
            discontinuity=segment.discontinuity,
            hop_s=hop_s,
            t=[s.t for s in samples],
            db=[s.db for s in samples],
            **extra,
        )

    def log_window(
        self, window: Window, *, closed_at: Optional[float] = None, **extra: Any
    ) -> Optional[Dict[str, Any]]:
        """Log a closed window. Call when it closes, or pass ``closed_at``."""
        return self.emit(
            "window",
            window.end,
            t_wall=closed_at,
            window_id=window.window_id,
            channel=window.channel,
            start=window.start,
            end=window.end,
            detector=window.detector,
            stats=dict(window.stats),
            forced=window.forced,
            **extra,
        )

    def log_clip(
        self,
        clip: Clip,
        *,
        cut_s: Optional[float] = None,
        done_at: Optional[float] = None,
        **extra: Any,
    ) -> Optional[Dict[str, Any]]:
        """Log a complete clip file. Call when it is on disk, or pass ``done_at``."""
        return self.emit(
            "clip",
            clip.window.end,
            t_wall=done_at,
            window_id=clip.window.window_id,
            path=clip.path,
            file_start=clip.file_start,
            file_duration=clip.file_duration,
            content_start=clip.content_start,
            content_end=clip.content_end,
            cut_s=cut_s,
            **extra,
        )

    def log_score(
        self, window: Window, scores: Mapping[str, float], **extra: Any
    ) -> Optional[Dict[str, Any]]:
        return self.emit(
            "score", window.end, window_id=window.window_id, scores=dict(scores), **extra
        )

    def log_dropped(
        self,
        what: str,
        reason: str,
        *,
        window: Optional[Window] = None,
        t_media: Optional[float] = None,
        **extra: Any,
    ) -> Optional[Dict[str, Any]]:
        """Log dropped work. ``t_media`` defaults to the window's end, else 0."""
        if t_media is None:
            t_media = window.end if window is not None else 0.0
        return self.emit(
            "dropped",
            t_media,
            what=what,
            reason=reason,
            window_id=window.window_id if window is not None else None,
            **extra,
        )

    def log_error(
        self,
        what: str,
        error: Union[BaseException, str],
        *,
        window: Optional[Window] = None,
        t_media: Optional[float] = None,
        **extra: Any,
    ) -> Optional[Dict[str, Any]]:
        """Log a failed job (the job is not retried)."""
        if t_media is None:
            t_media = window.end if window is not None else 0.0
        return self.emit(
            "error",
            t_media,
            what=what,
            error=str(error),
            error_type=type(error).__name__ if isinstance(error, BaseException) else None,
            window_id=window.window_id if window is not None else None,
            **extra,
        )

    def log_session_end(self, *, t_media: Optional[float] = None, **totals: Any) -> Optional[Dict[str, Any]]:
        """Write ``session_end`` with per-type counts and the latency summary."""
        with self._lock:
            counts = dict(self._counts)
            if t_media is None:
                t_media = self._max_t_media
        return self.emit(
            "session_end", t_media, counts=counts, latency=self.latency.summary(), **totals
        )


# --------------------------------------------------------------------------- reading


def _events_path(path: PathLike) -> Path:
    p = Path(path)
    return p / EVENTS_FILE if p.is_dir() else p


def read_events(
    path: PathLike, type: Union[str, Iterable[str], None] = None
) -> Iterator[Dict[str, Any]]:
    """Yield the events in a log, in file order.

    ``path`` is an ``events.jsonl`` file or the directory holding it. ``type``
    filters by one event type or a collection of them. Only complete lines
    (ending in a newline) are returned, so it is safe to read a log that is
    still being written; a complete line that is not valid JSON is skipped
    with a warning.
    """
    if type is None:
        wanted: Optional[frozenset] = None
    elif isinstance(type, str):
        wanted = frozenset((type,))
    else:
        wanted = frozenset(type)
    p = _events_path(path)
    with open(p, "rb") as fh:
        for lineno, raw in enumerate(fh, 1):
            if not raw.endswith(b"\n"):
                break  # a line still being written
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                log.warning("%s:%d: skipping unreadable line (%s)", p, lineno, exc)
                continue
            if not isinstance(event, dict):
                log.warning("%s:%d: skipping non-object line", p, lineno)
                continue
            if wanted is not None and event.get("type") not in wanted:
                continue
            yield event


def samples_from_event(event: Mapping[str, Any]) -> List[LoudnessSample]:
    """Rebuild the ``LoudnessSample``s of one ``audio`` event (``null`` dB -> ``-inf``)."""
    ts = event.get("t") or []
    dbs = event.get("db") or []
    if len(ts) != len(dbs):
        raise ValueError(f"audio event has {len(ts)} times but {len(dbs)} levels")
    return [
        LoudnessSample(t=float(t), db=float(db) if db is not None else float("-inf"))
        for t, db in zip(ts, dbs)
    ]


def read_loudness(path: PathLike) -> List[LoudnessSample]:
    """All loudness samples in a log, in the order they were logged."""
    out: List[LoudnessSample] = []
    for event in read_events(path, "audio"):
        out.extend(samples_from_event(event))
    return out


def read_config(path: PathLike) -> Optional[RealtimeConfig]:
    """The ``RealtimeConfig`` recorded by the first ``session_start`` (``None`` if absent)."""
    for event in read_events(path, "session_start"):
        cfg = event.get("config")
        return RealtimeConfig.from_dict(cfg) if isinstance(cfg, dict) else None
    return None


def window_from_event(event: Mapping[str, Any]) -> Window:
    """Rebuild the ``Window`` of a ``window`` event."""
    return Window(
        channel=event["channel"],
        window_id=event["window_id"],
        start=float(event["start"]),
        end=float(event["end"]),
        detector=event["detector"],
        stats={k: (float(v) if v is not None else float("nan")) for k, v in (event.get("stats") or {}).items()},
        forced=bool(event.get("forced", False)),
    )


# --------------------------------------------------------------------------- latency


#: The metrics of :meth:`LatencyTracker.summary`, in report order.
LATENCY_METRICS: Tuple[str, ...] = (
    "detect_latency",
    "cut_latency",
    "total_latency",
    "time_to_clip",
    "ingest_lag",
    "behind_schedule",
)


class ScheduleClock:
    """Estimates when the broadcast released each segment (its ``due_at``).

    A paced source releases the segment ending at media time ``end`` at wall
    time ``anchor + end / speed``. The anchor is:

    - the source's own playback start (``t0``, e.g. ``ReplaySource.t0``)
      when it exposes one, which is exact;
    - otherwise the smallest ``arrived_at - end / speed`` seen so far: the
      most punctual segment defines the schedule, so a loop that falls
      behind (whose segments "arrive" late because it pulls them late) shows
      up as behind;
    - reset to ``arrived_at - end / speed`` on a discontinuity, because media
      time stays contiguous over lost content while wall time does not; from
      then on the smallest-arrival rule applies even if ``t0`` is known.

    Without ``t0`` the first segments are assumed to arrive on time. A source
    whose clock drifts from the wall clock, or that stalls without flagging a
    discontinuity, also shows up as behind. ``speed=None`` (unpaced) disables
    the clock (:meth:`due_at` returns ``None``). Not thread-safe (one per
    ingest loop).
    """

    def __init__(self, speed: Optional[float] = 1.0) -> None:
        if speed is not None and not speed > 0:
            raise ValueError(f"speed must be > 0 or None, got {speed!r}")
        self.speed = None if speed is None else float(speed)
        self.anchor: Optional[float] = None
        self._use_t0 = True  # until the first discontinuity

    def due_at(
        self,
        end: float,
        arrived_at: Optional[float],
        *,
        discontinuity: bool = False,
        t0: Optional[float] = None,
    ) -> Optional[float]:
        """The ``due_at`` of a segment ending at media time ``end`` (``None`` if unknown)."""
        if self.speed is None or not arrived_at:
            return None
        offset = end / self.speed
        candidate = float(arrived_at) - offset
        if discontinuity and self.anchor is not None:
            self.anchor = candidate
            self._use_t0 = False
        elif t0 is not None and self._use_t0:
            self.anchor = float(t0)
        else:
            self.anchor = candidate if self.anchor is None else min(self.anchor, candidate)
        return self.anchor + offset


@dataclass(frozen=True)
class ClipLatency:
    """Latency of one window/clip, in seconds. ``None`` where an input is missing."""

    window_id: str
    window_end: float
    detect_latency: Optional[float]
    cut_latency: Optional[float]
    time_to_clip: Optional[float] = None  # clip done minus due_at of the segment holding the window end

    @property
    def total_latency(self) -> Optional[float]:
        if self.detect_latency is None or self.cut_latency is None:
            return None
        return self.detect_latency + self.cut_latency


@dataclass(frozen=True)
class SegmentLatency:
    seq: int
    ingest_lag: Optional[float]
    behind_schedule: Optional[float] = None  # t_wall minus due_at


def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "mean": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


class LatencyTracker:
    """Latency metrics computed from ``segment``, ``window`` and ``clip`` events.

    Feed events (as dicts, as written by :class:`SessionLog`) with
    :meth:`feed`, in any order; metrics are computed on demand. Thread-safe.
    Percentiles use NumPy's default linear interpolation.
    """

    _EPS = 1e-6

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # seq -> (start, end, arrived_at, t_wall, due_at)
        self._segments: Dict[int, Tuple[float, float, Optional[float], Optional[float], Optional[float]]] = {}
        # window_id -> (end, t_wall); insertion order = order closed
        self._windows: Dict[str, Tuple[float, Optional[float]]] = {}
        self._clips: Dict[str, Optional[float]] = {}

    @classmethod
    def from_events(cls, events: Iterable[Mapping[str, Any]]) -> "LatencyTracker":
        tracker = cls()
        for event in events:
            tracker.feed(event)
        return tracker

    @classmethod
    def from_log(cls, path: PathLike) -> "LatencyTracker":
        return cls.from_events(read_events(path, ("segment", "window", "clip")))

    def feed(self, event: Mapping[str, Any]) -> None:
        kind = event.get("type")
        if kind not in ("segment", "window", "clip"):
            return
        t_wall = _num(event.get("t_wall"))
        with self._lock:
            if kind == "segment":
                seq = event.get("seq")
                start = _num(event.get("start"))
                duration = _num(event.get("duration"))
                if seq is None or start is None or duration is None:
                    return
                self._segments[int(seq)] = (
                    start, start + duration, _num(event.get("arrived_at")), t_wall, _num(event.get("due_at"))
                )
            elif kind == "window":
                wid = event.get("window_id")
                end = _num(event.get("end"))
                if end is None:
                    end = _num(event.get("t_media"))
                if wid is None or end is None or wid in self._windows:
                    return
                self._windows[str(wid)] = (end, t_wall)
            else:
                wid = event.get("window_id")
                if wid is None or wid in self._clips:
                    return
                self._clips[str(wid)] = t_wall

    # ---------------------------------------------------------------- queries

    def _segment_at(self, t: float, ordered: List[Tuple[float, float, Optional[float], Optional[float]]]):
        """The ``(start, end, arrived_at, due_at)`` of the segment containing media time ``t``
        (``start < t <= end``), or ``None``."""
        starts = [s[0] for s in ordered]
        i = bisect.bisect_left(starts, t - self._EPS) - 1
        if i < 0:
            i = 0 if ordered and t <= ordered[0][1] + self._EPS else -1
        if i < 0:
            return None
        row = ordered[i]
        if t > row[1] + self._EPS:
            return None
        return row

    def per_clip(self) -> List[ClipLatency]:
        """One row per logged window, in the order windows were logged.

        ``cut_latency`` and ``time_to_clip`` are ``None`` for a window with no
        ``clip`` event; ``time_to_clip`` also when the segment has no ``due_at``.
        """
        with self._lock:
            ordered = sorted(((s, e, a, d) for s, e, a, _, d in self._segments.values()), key=lambda r: r[0])
            rows: List[ClipLatency] = []
            for wid, (end, closed) in self._windows.items():
                seg = self._segment_at(end, ordered)
                arrived = seg[2] if seg is not None else None
                due = seg[3] if seg is not None else None
                detect = closed - arrived if closed is not None and arrived is not None else None
                done = self._clips.get(wid)
                cut = done - closed if done is not None and closed is not None else None
                ttc = done - due if done is not None and due is not None else None
                rows.append(ClipLatency(wid, end, detect, cut, ttc))
            return rows

    def per_segment(self) -> List[SegmentLatency]:
        """One row per logged segment, by ``seq``."""
        with self._lock:
            rows = []
            for seq in sorted(self._segments):
                _, _, arrived, done, due = self._segments[seq]
                lag = done - arrived if done is not None and arrived is not None else None
                behind = done - due if done is not None and due is not None else None
                rows.append(SegmentLatency(seq, lag, behind))
            return rows

    def values(self) -> Dict[str, List[float]]:
        """The raw values behind :meth:`summary`, per metric (for pooling over channels)."""
        clips = self.per_clip()
        segs = self.per_segment()
        return {
            "detect_latency": [c.detect_latency for c in clips if c.detect_latency is not None],
            "cut_latency": [c.cut_latency for c in clips if c.cut_latency is not None],
            "total_latency": [c.total_latency for c in clips if c.total_latency is not None],
            "time_to_clip": [c.time_to_clip for c in clips if c.time_to_clip is not None],
            "ingest_lag": [s.ingest_lag for s in segs if s.ingest_lag is not None],
            "behind_schedule": [s.behind_schedule for s in segs if s.behind_schedule is not None],
        }

    def summary(self) -> Dict[str, Dict[str, Optional[float]]]:
        """``{metric: {count, p50, p95, mean, max}}`` for each of :data:`LATENCY_METRICS`
        (``detect_latency``, ``cut_latency``, ``total_latency``, ``time_to_clip``,
        ``ingest_lag``, ``behind_schedule``).

        Missing values are left out of the statistics; a metric with no values
        has ``count`` 0 and ``None`` elsewhere.
        """
        return {k: _stats(v) for k, v in self.values().items()}
