"""ReplaySource: a video file replayed as a simulated live feed (design section 6.1).

The file is split into MPEG-TS segments up front (stream copy, or a re-encode
with ``transcode=True``). Iterating then releases each segment at the moment it
would have finished in a real broadcast, ``t0 + end / speed`` on the wall clock,
or immediately with ``speed=None``.

Ownership of segment files: a segment's file belongs to whoever receives it
(normally the ``SegmentBuffer``, which deletes it on eviction). Files the
source never hands out (dropped segments, or the rest of the stream after
:meth:`ReplaySource.close`) are deleted by the source when iteration ends, as
is the muxer's CSV list. So after a full run the work directory holds only
files a consumer was given.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
import uuid
from pathlib import Path
from typing import AbstractSet, Iterable, Iterator, List, Optional, Union

from ..media import SegmentInfo, segment_file
from ..types import Segment

__all__ = ["ReplaySource"]

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _safe_name(text: str) -> str:
    """A file-name-safe version of ``text`` (Windows and POSIX)."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return cleaned or "channel"


class ReplaySource:
    """A :class:`~VidFlow.realtime.types.Source` that replays ``video`` as a live feed.

    Parameters
    ----------
    video:
        The file to replay.
    channel:
        Channel name put on every segment.
    work_dir:
        Directory the segment files are written to (created if needed). File
        names carry the channel and a per-instance token, so several sources
        may share one directory.
    speed:
        Playback rate. Segment ``n`` is released at wall time
        ``t0 + end_n / speed``, where ``end_n`` is the segment's end on the
        file's own timeline (equal to ``Segment.end`` unless segments were
        dropped, see below). ``None`` releases every segment immediately.
    segment_time:
        Target segment length passed to :func:`media.segment_file`.
    transcode:
        Re-encode with a keyframe every ``segment_time`` (for files whose
        keyframes are far apart).
    drop:
        Sequence numbers to withhold, simulating lost segments. The next
        segment yielded after a gap carries ``discontinuity=True``; media time
        stays contiguous (design section 5). ``seq`` is the segment's index in
        the file, so yielded ``seq`` values skip the dropped ones.
    jitter:
        Extra delay per segment, uniform in ``[0, jitter]`` wall seconds,
        drawn from ``random.Random(seed)``. Applied only when paced. Order is
        preserved: a segment is never released before the previous one.
    seed:
        Seed for the jitter generator.

    Splitting happens on the first call to :meth:`prepare` or :meth:`__iter__`
    and is not interrupted by :meth:`close`. ``t0`` is taken after splitting,
    so splitting time does not count against the schedule. A source can be
    iterated once.
    """

    def __init__(
        self,
        video: PathLike,
        channel: str,
        work_dir: PathLike,
        speed: Optional[float] = 1.0,
        segment_time: float = 2.0,
        transcode: bool = False,
        drop: Iterable[int] = (),
        jitter: float = 0.0,
        seed: int = 0,
    ) -> None:
        if speed is not None and not speed > 0:
            raise ValueError(f"speed must be > 0 or None, got {speed!r}")
        if not segment_time > 0:
            raise ValueError(f"segment_time must be > 0, got {segment_time!r}")
        if not jitter >= 0:
            raise ValueError(f"jitter must be >= 0, got {jitter!r}")
        self.video = Path(video)
        self.channel: str = channel
        self.work_dir = Path(work_dir)
        self.speed: Optional[float] = None if speed is None else float(speed)
        self.segment_time = float(segment_time)
        self.transcode = bool(transcode)
        self.drop: AbstractSet[int] = frozenset(int(s) for s in drop)
        self.jitter = float(jitter)
        self.seed = seed

        self._prefix = f"{_safe_name(channel)}_{uuid.uuid4().hex[:8]}_"
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._infos: Optional[List[SegmentInfo]] = None
        self._started = False
        self.t0: Optional[float] = None  # time.monotonic() when playback started

    # ----------------------------------------------------------------- public

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def prepare(self) -> List[SegmentInfo]:
        """Split the file (once) and return the segments on the file's timeline.

        The returned list includes segments that will be dropped.
        """
        with self._lock:
            if self._infos is None:
                t = time.monotonic()
                infos = segment_file(
                    self.video,
                    self.work_dir,
                    segment_time=self.segment_time,
                    transcode=self.transcode,
                    prefix=self._prefix,
                )
                self._list_path().unlink(missing_ok=True)
                kept: List[SegmentInfo] = []
                for info in infos:
                    if info.duration > 0:
                        kept.append(info)
                    else:
                        log.warning("%s: skipping empty segment %s", self.channel, info.path.name)
                        info.path.unlink(missing_ok=True)
                self._infos = kept
                log.info(
                    "%s: split %s into %d segments (%.1f s of media) in %.2f s",
                    self.channel,
                    self.video.name,
                    len(kept),
                    kept[-1].end if kept else 0.0,
                    time.monotonic() - t,
                )
            return list(self._infos)

    def __iter__(self) -> Iterator[Segment]:
        with self._lock:
            if self._started:
                raise RuntimeError("a ReplaySource can only be iterated once")
            self._started = True
        return self._run()

    def close(self) -> None:
        """Make a blocked iterator return promptly. Safe from any thread."""
        self._closed.set()

    # ---------------------------------------------------------------- private

    def _list_path(self) -> Path:
        return self.work_dir / f"{self._prefix}list.csv"

    def _run(self) -> Iterator[Segment]:
        if self._closed.is_set():
            return
        infos = self.prepare()
        rng = random.Random(self.seed)
        self.t0 = time.monotonic()
        handed_out = set()
        media_t = 0.0
        gap = False
        yielded_any = False
        try:
            for info in infos:
                if self._closed.is_set():
                    return
                if self.speed is not None:
                    due = self.t0 + info.end / self.speed
                    if self.jitter > 0:
                        due += rng.uniform(0.0, self.jitter)
                    delay = due - time.monotonic()
                    if delay > 0 and self._closed.wait(delay):
                        return
                elif self._closed.is_set():
                    return
                if info.index in self.drop:
                    log.debug("%s: dropping segment %d", self.channel, info.index)
                    gap = True
                    continue
                seg = Segment(
                    channel=self.channel,
                    seq=info.index,
                    start=media_t,
                    duration=info.duration,
                    path=info.path,
                    discontinuity=gap and yielded_any,
                    arrived_at=time.monotonic(),
                )
                media_t = media_t + info.duration
                gap = False
                yielded_any = True
                handed_out.add(info.index)
                yield seg
        finally:
            # Files never handed out are still ours; remove them.
            for info in infos:
                if info.index not in handed_out:
                    info.path.unlink(missing_ok=True)
