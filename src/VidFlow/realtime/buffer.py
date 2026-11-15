"""Rolling on-disk segment buffer with leases (``docs/realtime/DESIGN.md`` section 6.2).

The buffer keeps an in-memory index of the most recent segments of one channel
and owns their files: it deletes a segment's file once the segment has aged out
of ``retention_s`` *and* no :class:`Lease` still pins it.

Threading: :meth:`SegmentBuffer.add` is called from the ingest thread while
:meth:`SegmentBuffer.lease` and :meth:`Lease.release` are called from workers.
Every public method takes one internal lock; file deletion happens outside it.

Timing uses media time only (``Segment.start`` / ``Segment.end``).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from pathlib import Path
from types import TracebackType
from typing import Deque, Dict, List, Optional, Sequence, Tuple, Type

from .types import Segment

__all__ = ["SegmentBuffer", "Lease"]

log = logging.getLogger(__name__)

# Tolerance for media-time comparisons. Segment boundaries are exact sums of
# durations, but callers' window times (hop multiples, padding) are not.
_EPS = 1e-6


class Lease:
    """A set of contiguous buffered segments pinned against deletion.

    Obtained from :meth:`SegmentBuffer.lease`. Use it as a context manager, or
    call :meth:`release` when done; releasing twice is harmless.

    Attributes
        segments   the leased segments, oldest first (may be empty)
        start      media time of the first leased segment's start
        end        media time of the last leased segment's end
        truncated  True if the requested head (core start minus ``pre``) was
                   cut short because those segments had already been evicted.
                   Stopping the padding at a discontinuity does not count.

    For an empty lease ``start == end`` (the requested core end).
    """

    __slots__ = ("segments", "start", "end", "truncated", "_buffer", "_released", "_lock")

    def __init__(
        self,
        buffer: Optional["SegmentBuffer"],
        segments: Sequence[Segment],
        truncated: bool,
        empty_at: float = 0.0,
    ) -> None:
        self.segments: Tuple[Segment, ...] = tuple(segments)
        if self.segments:
            self.start: float = self.segments[0].start
            self.end: float = self.segments[-1].end
        else:
            self.start = self.end = float(empty_at)
        self.truncated: bool = bool(truncated)
        self._buffer = buffer
        self._released = not self.segments
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ info

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def paths(self) -> List[Path]:
        return [s.path for s in self.segments]

    @property
    def released(self) -> bool:
        return self._released

    def __len__(self) -> int:
        return len(self.segments)

    def __bool__(self) -> bool:
        return bool(self.segments)

    def __repr__(self) -> str:
        seqs = f"{self.segments[0].seq}..{self.segments[-1].seq}" if self.segments else "-"
        return (
            f"Lease(seq={seqs}, start={self.start:.3f}, end={self.end:.3f}, "
            f"truncated={self.truncated}, released={self._released})"
        )

    # --------------------------------------------------------------- release

    def release(self) -> None:
        """Unpin the segments. Idempotent and safe from any thread."""
        with self._lock:
            if self._released:
                return
            self._released = True
        if self._buffer is not None:
            self._buffer._release(self.segments)

    def __enter__(self) -> "Lease":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.release()


class SegmentBuffer:
    """Rolling buffer of one channel's segments, in media time.

    ``retention_s``   keep at least this much media time behind the newest
                      segment's end. A segment is evicted once its end is at or
                      before ``latest_end - retention_s``, so the buffered
                      duration never exceeds ``retention_s`` plus one segment.
    ``delete_files``  set False to never touch files (e.g. shared fixtures);
                      the index and lease bookkeeping behave the same.

    Validate ``retention_s`` against the rest of the settings with
    ``RealtimeConfig.validate()`` at start-up; the buffer only checks it is > 0.
    """

    def __init__(self, retention_s: float, *, delete_files: bool = True) -> None:
        retention_s = float(retention_s)
        if not retention_s > 0:
            raise ValueError(f"retention_s must be > 0, got {retention_s!r}")
        self.retention_s = retention_s
        self.delete_files = delete_files
        self._lock = threading.Lock()
        self._index: Deque[Segment] = deque()
        self._pins: Dict[int, int] = {}  # seq -> number of live leases
        self._evicted_pinned: Dict[int, Segment] = {}  # evicted, waiting for release
        self._undeleted: List[Path] = []  # deletions that failed (e.g. file open on Windows)
        self._last_seq: Optional[int] = None
        self._latest_end: Optional[float] = None
        self._evicted_any = False
        self._channel: Optional[str] = None
        self._closed = False

    # ------------------------------------------------------------- queries

    @property
    def latest_end(self) -> Optional[float]:
        """End of the newest segment ever added (None before the first ``add``)."""
        with self._lock:
            return self._latest_end

    @property
    def oldest_start(self) -> Optional[float]:
        """Start of the oldest segment still leasable (None when empty)."""
        with self._lock:
            return self._index[0].start if self._index else None

    @property
    def buffered_duration(self) -> float:
        """Total duration of the leasable segments."""
        with self._lock:
            return sum(s.duration for s in self._index)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pinned_count(self) -> int:
        """Number of distinct segments held by at least one live lease."""
        with self._lock:
            return len(self._pins)

    def covers(self, t: float) -> bool:
        """True once segments through media time ``t`` have been added."""
        with self._lock:
            return self._latest_end is not None and self._latest_end >= t - _EPS

    def segments(self) -> List[Segment]:
        """Snapshot of the leasable segments, oldest first."""
        with self._lock:
            return list(self._index)

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

    # ---------------------------------------------------------------- add

    def add(self, segment: Segment) -> List[Segment]:
        """Index ``segment`` and evict what has aged out.

        Returns the segments evicted by this call (their files are deleted now
        unless leased, in which case on their last release). Raises
        ``ValueError`` if ``seq`` does not increase or the channel changes, and
        ``RuntimeError`` after :meth:`close`.
        """
        to_delete: List[Path] = []
        evicted: List[Segment] = []
        with self._lock:
            if self._closed:
                raise RuntimeError("SegmentBuffer is closed")
            if self._channel is None:
                self._channel = segment.channel
            elif segment.channel != self._channel:
                raise ValueError(
                    f"segment of channel {segment.channel!r} added to buffer of {self._channel!r}"
                )
            if self._last_seq is not None and segment.seq <= self._last_seq:
                raise ValueError(f"seq must increase: got {segment.seq} after {self._last_seq}")
            self._last_seq = segment.seq
            self._index.append(segment)
            if self._latest_end is None or segment.end > self._latest_end:
                self._latest_end = segment.end

            cutoff = self._latest_end - self.retention_s
            # Never evict the newest segment, whatever the retention.
            while len(self._index) > 1 and self._index[0].end <= cutoff + _EPS:
                old = self._index.popleft()
                evicted.append(old)
                self._evicted_any = True
                if old.seq in self._pins:
                    self._evicted_pinned[old.seq] = old
                else:
                    to_delete.append(old.path)
            retry, self._undeleted = self._undeleted, []
        self._delete(retry + to_delete)
        if evicted:
            log.debug(
                "evicted seq %d..%d (%d pinned)",
                evicted[0].seq,
                evicted[-1].seq,
                len(evicted) - len(to_delete),
            )
        return evicted

    # -------------------------------------------------------------- lease

    def lease(self, core_start: float, core_end: float, pre: float = 0.0, post: float = 0.0) -> Lease:
        """Pin the segments covering ``[core_start, core_end]`` plus padding.

        The core segments are those overlapping the core interval (for a
        zero-length core, the segment containing ``core_start``). The lease
        then extends back over segments ending after ``core_start - pre`` and
        forward over segments starting before ``core_end + post``. Padding
        stops at a discontinuity: it never includes a segment flagged
        ``discontinuity=True`` after the core, and never reaches back past the
        first core-side segment that carries the flag.

        Segments not yet added are simply absent (``lease.end`` is then short
        of what was asked; check :meth:`covers` first). If the head has been
        evicted the remainder is returned with ``truncated=True``; if the whole
        core has been evicted the lease is empty and truncated.
        """
        core_start = float(core_start)
        core_end = float(core_end)
        pre = float(pre)
        post = float(post)
        if core_end < core_start:
            raise ValueError(f"core_end ({core_end}) is before core_start ({core_start})")
        if pre < 0 or post < 0:
            raise ValueError(f"pre and post must be >= 0, got {pre}, {post}")
        want_start = core_start - pre
        want_end = core_end + post

        with self._lock:
            if self._closed:
                raise RuntimeError("SegmentBuffer is closed")
            idx = self._index
            n = len(idx)

            # Core: indices [lo, hi] of segments overlapping the core.
            lo = hi = -1
            for i, s in enumerate(idx):
                if core_end - core_start > _EPS:
                    overlaps = s.end > core_start + _EPS and s.start < core_end - _EPS
                else:
                    overlaps = s.start <= core_start + _EPS and s.end > core_start + _EPS
                if overlaps:
                    if lo < 0:
                        lo = i
                    hi = i
                elif lo >= 0:
                    break

            if lo < 0:
                head_gone = n == 0 or core_end <= idx[0].start + _EPS
                truncated = head_gone and self._evicted_any
                return Lease(None, (), truncated=truncated, empty_at=core_end)

            # Extend back: include idx[lo - 1] unless idx[lo] starts after a gap.
            first = lo
            while (
                first > 0
                and idx[first - 1].end > want_start + _EPS
                and not self._boundary(idx[first - 1], idx[first])
            ):
                first -= 1
            truncated = (
                first == 0
                and self._evicted_any
                and idx[0].start > want_start + _EPS
                and not idx[0].discontinuity
            )

            # Extend forward: include idx[last + 1] unless it starts after a gap.
            last = hi
            while (
                last + 1 < n
                and idx[last + 1].start < want_end - _EPS
                and not self._boundary(idx[last], idx[last + 1])
            ):
                last += 1

            segs = [idx[i] for i in range(first, last + 1)]
            for s in segs:
                self._pins[s.seq] = self._pins.get(s.seq, 0) + 1
        return Lease(self, segs, truncated=truncated)

    @staticmethod
    def _boundary(prev: Segment, nxt: Segment) -> bool:
        """True if content was lost between two adjacent buffered segments."""
        return nxt.discontinuity or abs(nxt.start - prev.end) > _EPS

    def _release(self, segments: Sequence[Segment]) -> None:
        to_delete: List[Path] = []
        with self._lock:
            for s in segments:
                count = self._pins.get(s.seq, 0) - 1
                if count > 0:
                    self._pins[s.seq] = count
                    continue
                self._pins.pop(s.seq, None)
                dead = self._evicted_pinned.pop(s.seq, None)
                if dead is not None:
                    to_delete.append(dead.path)
        self._delete(to_delete)

    # -------------------------------------------------------------- close

    def close(self) -> None:
        """Delete every file not leased; leased ones go on their last release.

        Further ``add``/``lease`` calls raise ``RuntimeError``. Idempotent.
        """
        to_delete: List[Path] = []
        with self._lock:
            if not self._closed:
                self._closed = True
                while self._index:
                    s = self._index.popleft()
                    if s.seq in self._pins:
                        self._evicted_pinned[s.seq] = s
                    else:
                        to_delete.append(s.path)
            to_delete.extend(self._undeleted)
            self._undeleted = []
        self._delete(to_delete)

    def __enter__(self) -> "SegmentBuffer":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    # ------------------------------------------------------------ helpers

    def _delete(self, paths: Sequence[Path]) -> None:
        if not self.delete_files or not paths:
            return
        failed: List[Path] = []
        for p in paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError as e:  # e.g. still open by another process on Windows
                log.warning("could not delete segment %s: %s; will retry", p, e)
                failed.append(Path(p))
        if failed:
            with self._lock:
                self._undeleted.extend(failed)
