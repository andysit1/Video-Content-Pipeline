"""Fakes and fixture builders for the real-time clipper's tests.

Nothing here is used at run time; it exists so every component can be tested
with only FFmpeg on ``PATH``.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from .media import run_ffmpeg, segment_file
from .types import Segment

__all__ = [
    "SYNTHETIC_DURATION",
    "SYNTHETIC_BURSTS",
    "SYNTHETIC_SIZE",
    "SYNTHETIC_FPS",
    "SYNTHETIC_TONE_HZ",
    "SYNTHETIC_LOUD_AMPLITUDE",
    "SYNTHETIC_QUIET_AMPLITUDE",
    "make_synthetic_video",
    "make_segments",
    "segments_from_file",
    "copy_segments",
    "FakeSource",
]

PathLike = Union[str, Path]

# The default synthetic fixture (design section 9).
SYNTHETIC_DURATION: float = 60.0
SYNTHETIC_BURSTS: Tuple[Tuple[float, float], ...] = ((10.0, 16.0), (40.0, 46.0))
SYNTHETIC_SIZE: Tuple[int, int] = (320, 180)
SYNTHETIC_FPS: int = 30
SYNTHETIC_TONE_HZ: float = 440.0
# Sine peak amplitudes. RMS of a sine is peak / sqrt(2), so:
#   loud  0.8   -> about  -4.9 dBFS RMS
#   quiet 0.008 -> about -44.9 dBFS RMS (below the default floor_db of -35)
SYNTHETIC_LOUD_AMPLITUDE: float = 0.8
SYNTHETIC_QUIET_AMPLITUDE: float = 0.008


def make_synthetic_video(
    path: PathLike,
    duration: float = SYNTHETIC_DURATION,
    bursts: Sequence[Tuple[float, float]] = SYNTHETIC_BURSTS,
    *,
    size: Tuple[int, int] = SYNTHETIC_SIZE,
    fps: int = SYNTHETIC_FPS,
    tone_hz: float = SYNTHETIC_TONE_HZ,
    loud: float = SYNTHETIC_LOUD_AMPLITUDE,
    quiet: float = SYNTHETIC_QUIET_AMPLITUDE,
    sample_rate: int = 48000,
) -> Path:
    """Generate the synthetic test video with FFmpeg and return its path.

    320x180, 30 fps H.264 with a keyframe exactly every second (``testsrc2``
    pattern), plus stereo 48 kHz AAC audio: a quiet ``tone_hz`` sine at
    amplitude ``quiet`` with full-volume (amplitude ``loud``) bursts over each
    ``[start, end)`` in ``bursts``. The file is written to a temporary name in
    the same directory and renamed into place, so a partial file is never seen.
    """
    path = Path(path)
    duration = float(duration)
    if duration <= 0:
        raise ValueError("duration must be > 0")
    for b0, b1 in bursts:
        if not 0 <= b0 < b1 <= duration:
            raise ValueError(f"burst {(b0, b1)} is not inside [0, {duration}]")
    path.parent.mkdir(parents=True, exist_ok=True)

    if bursts:
        # gte/lt instead of between() so the end is exclusive.
        loud_expr = "+".join(f"gte(t,{float(b0)})*lt(t,{float(b1)})" for b0, b1 in bursts)
        amp = f"if({loud_expr},{loud},{quiet})"
    else:
        amp = f"{quiet}"
    tone = f"{amp}*sin(2*PI*{tone_hz}*t)"
    # Single quotes protect the commas in the expression from the filter parser.
    audio_src = f"aevalsrc=exprs='{tone}|{tone}':s={sample_rate}:c=stereo:d={duration}"
    w, h = size
    video_src = f"testsrc2=size={w}x{h}:rate={fps}:duration={duration}"

    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.partial{path.suffix or '.mp4'}")
    try:
        run_ffmpeg(
            [
                "-f", "lavfi", "-i", video_src,
                "-f", "lavfi", "-i", audio_src,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                # No B-frames: keeps dts == pts so segment boundaries land exactly
                # on the keyframes at whole seconds (no encoder-delay shift).
                "-bf", "0",
                "-g", str(fps),
                "-keyint_min", str(fps),
                "-sc_threshold", "0",
                "-force_key_frames", "expr:gte(t,n_forced*1)",
                "-c:a", "aac",
                "-b:a", "128k",
                "-t", f"{duration}",
                "-f", "mp4",
                str(tmp),
            ],
            timeout=300,
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def make_segments(
    channel: str,
    durations: Iterable[float],
    discontinuities: Iterable[int] = (),
    *,
    directory: Optional[PathLike] = None,
    start_seq: int = 0,
    arrived_at: float = 0.0,
    suffix: str = ".ts",
) -> List[Segment]:
    """Build ``Segment`` objects with correct media times.

    Segment ``i`` gets ``seq = start_seq + i`` and ``start`` equal to the
    running sum of the previous durations (so ``prev.end == next.start``
    exactly). ``discontinuities`` lists the *seq* numbers that carry
    ``discontinuity=True``. If ``directory`` is given, a small dummy file is
    created for each segment there (useful for buffer tests that need real
    files to delete); otherwise paths point at non-existent files under the
    current directory.
    """
    disc = set(int(s) for s in discontinuities)
    base = Path(directory) if directory is not None else Path(".")
    if directory is not None:
        base.mkdir(parents=True, exist_ok=True)
    segments: List[Segment] = []
    t = 0.0
    for i, d in enumerate(durations):
        d = float(d)
        if d <= 0:
            raise ValueError(f"segment duration must be > 0, got {d}")
        seq = start_seq + i
        path = base / f"{channel}_{seq:06d}{suffix}"
        if directory is not None:
            path.write_bytes(b"\x47" + bytes(187))  # one (empty) 188-byte TS packet
        segments.append(
            Segment(
                channel=channel,
                seq=seq,
                start=t,
                duration=d,
                path=path,
                discontinuity=seq in disc,
                arrived_at=arrived_at,
            )
        )
        t = t + d
    unknown = disc - {s.seq for s in segments}
    if unknown:
        raise ValueError(f"discontinuities name unknown seq numbers: {sorted(unknown)}")
    return segments


def segments_from_file(
    video: PathLike,
    out_dir: PathLike,
    channel: str = "test",
    segment_time: float = 2.0,
    transcode: bool = False,
) -> List[Segment]:
    """Split ``video`` with :func:`media.segment_file` and wrap the result as ``Segment``s."""
    infos = segment_file(video, out_dir, segment_time=segment_time, transcode=transcode)
    segments: List[Segment] = []
    t = 0.0
    for info in infos:
        segments.append(
            Segment(channel=channel, seq=info.index, start=t, duration=info.duration, path=info.path)
        )
        t = t + info.duration
    return segments


def copy_segments(segments: Sequence[Segment], dest: PathLike) -> List[Segment]:
    """Copy segment files into ``dest`` and return segments pointing at the copies.

    Use this before handing shared (session-scoped) segments to anything that
    deletes files, such as ``SegmentBuffer``.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out: List[Segment] = []
    for seg in segments:
        target = dest / seg.path.name
        shutil.copyfile(seg.path, target)
        out.append(dataclasses.replace(seg, path=target))
    return out


class FakeSource:
    """A :class:`~VidFlow.realtime.types.Source` that yields a fixed list of segments.

    - ``interval``: seconds to wait before releasing each segment (0 = none).
    - ``hold_open``: after the last segment, block (like a live source with no
      new data) until :meth:`close` is called instead of ending.
    - ``stamp_arrival``: set ``arrived_at`` to ``time.monotonic()`` at release.

    :meth:`close` is safe from any thread and makes a blocked iterator return
    promptly. Iterating a second time starts again from the first segment
    unless the source has been closed.
    """

    def __init__(
        self,
        segments: Sequence[Segment],
        channel: Optional[str] = None,
        *,
        interval: float = 0.0,
        hold_open: bool = False,
        stamp_arrival: bool = True,
    ) -> None:
        self.segments: List[Segment] = list(segments)
        if channel is None:
            channel = self.segments[0].channel if self.segments else "fake"
        self.channel: str = channel
        self.interval = float(interval)
        self.hold_open = hold_open
        self.stamp_arrival = stamp_arrival
        self.yielded: List[Segment] = []
        self._closed = threading.Event()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def __iter__(self) -> Iterator[Segment]:
        for seg in self.segments:
            if self._closed.is_set():
                return
            if self.interval > 0 and self._closed.wait(self.interval):
                return
            if self.stamp_arrival:
                seg = dataclasses.replace(seg, arrived_at=time.monotonic())
            self.yielded.append(seg)
            yield seg
        if self.hold_open:
            self._closed.wait()

    def close(self) -> None:
        self._closed.set()
