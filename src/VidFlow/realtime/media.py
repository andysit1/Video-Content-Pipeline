"""FFmpeg / ffprobe helpers: split a file into MPEG-TS segments, probe a file.

Only the standard library is used here; FFmpeg and ffprobe must be on ``PATH``.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

__all__ = [
    "MediaError",
    "SegmentInfo",
    "ProbeInfo",
    "ffmpeg_exe",
    "ffprobe_exe",
    "run_ffmpeg",
    "segment_file",
    "probe",
]

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

_STDERR_TAIL_CHARS = 2000


class MediaError(RuntimeError):
    """An FFmpeg/ffprobe invocation failed. ``stderr_tail`` holds the end of its stderr."""

    def __init__(self, message: str, stderr_tail: str = "") -> None:
        super().__init__(message if not stderr_tail else f"{message}\n{stderr_tail}")
        self.stderr_tail = stderr_tail


@dataclass(frozen=True)
class SegmentInfo:
    """One segment written by :func:`segment_file`.

    ``start`` is relative to the first segment (the first one starts at 0.0)
    and is the running sum of the previous durations, so segments are
    contiguous by construction. ``duration`` is the muxer's ``end - start``.
    """

    index: int
    path: Path
    start: float
    duration: float

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class ProbeInfo:
    duration: float  # seconds; 0.0 if unknown
    has_video: bool
    has_audio: bool
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    audio_rate: Optional[int] = None


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise MediaError("ffmpeg not found on PATH")
    return exe


def ffprobe_exe() -> str:
    exe = shutil.which("ffprobe")
    if exe is None:
        raise MediaError("ffprobe not found on PATH")
    return exe


def _tail(text: str) -> str:
    return text[-_STDERR_TAIL_CHARS:]


def run_ffmpeg(args: Sequence[str], timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    """Run ``ffmpeg -hide_banner -nostdin -loglevel error -y <args>``.

    Raises :class:`MediaError` (with the tail of stderr) on a non-zero exit or
    timeout. Arguments are passed as a list, never through a shell.
    """
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *map(str, args)]
    log.debug("running %s", cmd)
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        err = exc.stderr.decode("utf-8", "replace") if exc.stderr else ""
        raise MediaError(f"ffmpeg timed out after {timeout} s", _tail(err)) from exc
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace")
        raise MediaError(f"ffmpeg exited with code {proc.returncode}", _tail(err))
    return proc


def segment_file(
    src: PathLike,
    out_dir: PathLike,
    segment_time: float = 2.0,
    transcode: bool = False,
    prefix: str = "seg_",
    timeout: Optional[float] = None,
) -> List[SegmentInfo]:
    """Split ``src`` into self-contained MPEG-TS segments in ``out_dir``.

    Uses FFmpeg's segment muxer. With ``transcode=False`` the streams are
    copied, so cuts can only fall on existing keyframes and segments may be
    longer than ``segment_time``. With ``transcode=True`` the video is
    re-encoded (H.264) with a keyframe forced every ``segment_time`` seconds
    and the audio re-encoded to AAC, giving even segments.

    Returns one :class:`SegmentInfo` per file, in order, with start and
    duration read from the muxer's CSV list. Starts are re-based so the first
    segment starts at 0.0 and each start equals the previous end.
    """
    src = Path(src)
    out_dir = Path(out_dir)
    if not src.is_file():
        raise MediaError(f"input file not found: {src}")
    if not segment_time > 0:
        raise ValueError("segment_time must be > 0")
    out_dir.mkdir(parents=True, exist_ok=True)

    list_path = out_dir / f"{prefix}list.csv"
    pattern = out_dir / f"{prefix}%06d.ts"

    args: List[str] = ["-i", str(src), "-map", "0:v:0?", "-map", "0:a:0?"]
    if transcode:
        args += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-bf", "0",
            "-force_key_frames", f"expr:gte(t,n_forced*{segment_time})",
            "-sc_threshold", "0",
            "-c:a", "aac",
            "-b:a", "128k",
        ]
    else:
        args += ["-c", "copy"]
    args += [
        "-f", "segment",
        "-segment_time", f"{segment_time}",
        # Without a small delta, a keyframe exactly on a boundary is missed
        # (timestamp rounding) and the segment runs to the next keyframe.
        # FFmpeg's docs recommend about half a frame; 0.05 s covers >= 10 fps.
        "-segment_time_delta", f"{min(0.05, segment_time / 10.0)}",
        "-segment_format", "mpegts",
        "-segment_list", str(list_path),
        "-segment_list_type", "csv",
        "-reset_timestamps", "0",
        str(pattern),
    ]
    run_ffmpeg(args, timeout=timeout)

    rows: List[tuple] = []
    with open(list_path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            name, start, end = row[0], float(row[1]), float(row[2])
            rows.append((name, start, end))
    if not rows:
        raise MediaError(f"segment muxer produced no segments for {src}")

    infos: List[SegmentInfo] = []
    t = 0.0
    for i, (name, start, end) in enumerate(rows):
        duration = max(0.0, end - start)
        infos.append(SegmentInfo(index=i, path=out_dir / name, start=t, duration=duration))
        t = t + duration
    return infos


def _parse_rate(rate: Optional[str]) -> Optional[float]:
    if not rate or rate in ("0/0", "0"):
        return None
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            d = float(den)
            return float(num) / d if d else None
        return float(rate)
    except ValueError:
        return None


def probe(path: PathLike, timeout: Optional[float] = 30.0) -> ProbeInfo:
    """Return duration and stream presence of ``path`` using ffprobe."""
    path = Path(path)
    cmd = [
        ffprobe_exe(),
        "-v", "error",
        "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,sample_rate,duration",
        "-of", "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"ffprobe timed out on {path}") from exc
    if proc.returncode != 0:
        raise MediaError(
            f"ffprobe failed on {path} (code {proc.returncode})",
            _tail(proc.stderr.decode("utf-8", "replace")),
        )
    data = json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    streams = data.get("streams", []) or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = 0.0
    fmt_dur = (data.get("format") or {}).get("duration")
    if fmt_dur not in (None, "N/A"):
        duration = float(fmt_dur)
    else:
        durs = [float(s["duration"]) for s in streams if s.get("duration") not in (None, "N/A")]
        duration = max(durs) if durs else 0.0

    fps = None
    if video is not None:
        fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    audio_rate = None
    if audio is not None and audio.get("sample_rate"):
        audio_rate = int(audio["sample_rate"])

    return ProbeInfo(
        duration=duration,
        has_video=video is not None,
        has_audio=audio is not None,
        width=int(video["width"]) if video and video.get("width") else None,
        height=int(video["height"]) if video and video.get("height") else None,
        fps=fps,
        video_codec=video.get("codec_name") if video else None,
        audio_codec=audio.get("codec_name") if audio else None,
        audio_rate=audio_rate,
    )
