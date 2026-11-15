"""ClipWriter: cut a clip from buffered segments by stream copy (design section 6.5).

A clip is the leased segments concatenated with FFmpeg's concat demuxer and
stream-copied into an mp4. Nothing is re-encoded, so a cut costs well under a
second; the price is that cuts fall on segment boundaries, and the file can be
up to one segment longer than asked for at each end. The JSON sidecar records
``content_start``/``content_end`` (offsets into the file) so a later render
step can trim exactly.

Output layout::

    <out_dir>/<channel>/<session>/clips/<channel>_<start_ms>.mp4
    <out_dir>/<channel>/<session>/clips/<channel>_<start_ms>.json

Write order: the mp4 is written to ``<name>.mp4.<token>.partial``, checked,
renamed atomically to ``<name>.mp4``, and only then is the sidecar written
(also atomically). A sidecar's existence therefore means the clip is complete.
On any failure no ``.partial``, mp4 or sidecar is left behind and a
:class:`~VidFlow.realtime.types.ClipError` carries the tail of FFmpeg's stderr.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from .media import MediaError, probe, run_ffmpeg
from .types import Clip, ClipError, Segment, Window

__all__ = [
    "ClipWriter",
    "SIDECAR_SCHEMA_VERSION",
    "clip_to_dict",
    "clip_from_dict",
    "read_sidecar",
    "sidecar_path",
    "update_sidecar",
]

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

SIDECAR_SCHEMA_VERSION = 1

# Media-time slack when checking that leased segments are contiguous.
_CONTIGUITY_EPS = 1e-3
# Characters not allowed in a Windows path component (plus control chars).
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


# --------------------------------------------------------------------------- helpers


def _safe_component(name: str) -> str:
    """Make ``name`` usable as one path component on Windows and Linux."""
    cleaned = _UNSAFE_CHARS.sub("_", str(name)).strip().rstrip(".")
    return cleaned or "_"


def _num(value: Any) -> Optional[float]:
    """Plain float for JSON (NumPy scalars included); non-finite becomes ``None``."""
    if value is None:
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _float_map(values: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    return {str(k): _num(v) for k, v in values.items()}


def _concat_line(path: Path) -> str:
    """One concat-demuxer ``file`` line, single-quoted with embedded quotes escaped."""
    text = path.resolve().as_posix().replace("'", "'\\''")
    return f"file '{text}'\n"


def _lease_segments(lease: Any) -> List[Segment]:
    """The ordered segments of a ``Lease`` (anything with ``.segments``) or a plain sequence."""
    segments = getattr(lease, "segments", lease)
    return list(segments)


def sidecar_path(clip_path: PathLike) -> Path:
    """The sidecar of a clip file: same name, ``.json`` extension."""
    return Path(clip_path).with_suffix(".json")


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:12]}.partial")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _unlink_quietly(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:  # pragma: no cover - e.g. a locked file on Windows
        log.warning("could not remove %s: %s", path, exc)


# --------------------------------------------------------------------------- sidecar


def clip_to_dict(clip: Clip, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Serialise every field of ``clip`` and its ``window`` to plain JSON types.

    ``file`` is the clip's file name (relative to the sidecar); ``path`` is
    the absolute path at write time. ``extra`` keys are added at top level and
    may not shadow the clip's own fields.
    """
    w = clip.window
    data: Dict[str, Any] = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "file": clip.path.name,
        "path": str(Path(clip.path).resolve()),
        "file_start": _num(clip.file_start),
        "file_duration": _num(clip.file_duration),
        "content_start": _num(clip.content_start),
        "content_end": _num(clip.content_end),
        "scores": _float_map(clip.scores),
        "window": {
            "channel": w.channel,
            "window_id": w.window_id,
            "start": _num(w.start),
            "end": _num(w.end),
            "detector": w.detector,
            "stats": _float_map(w.stats),
            "forced": bool(w.forced),
        },
    }
    if extra:
        for key, value in extra.items():
            if key in data:
                raise ValueError(f"extra sidecar key {key!r} would overwrite a Clip field")
            data[key] = value
    return data


def clip_from_dict(data: Mapping[str, Any], base_dir: Optional[PathLike] = None) -> Clip:
    """Rebuild a :class:`Clip` from :func:`clip_to_dict` output.

    With ``base_dir`` (normally the sidecar's folder) the path is
    ``base_dir / file``, so a moved output folder still resolves; otherwise
    the stored absolute ``path`` is used. ``None`` values (non-finite numbers
    at write time) come back as ``nan``.
    """

    def f(v: Any) -> float:
        return float("nan") if v is None else float(v)

    wd = data["window"]
    window = Window(
        channel=wd["channel"],
        window_id=wd["window_id"],
        start=f(wd["start"]),
        end=f(wd["end"]),
        detector=wd["detector"],
        stats={k: f(v) for k, v in (wd.get("stats") or {}).items()},
        forced=bool(wd.get("forced", False)),
    )
    path = Path(base_dir) / data["file"] if base_dir is not None else Path(data["path"])
    return Clip(
        window=window,
        path=path,
        file_start=f(data["file_start"]),
        file_duration=f(data["file_duration"]),
        content_start=f(data["content_start"]),
        content_end=f(data["content_end"]),
        scores={k: f(v) for k, v in (data.get("scores") or {}).items()},
    )


def read_sidecar(path: PathLike) -> Clip:
    """Load a sidecar (``.json``, or the clip's ``.mp4`` path) back into a :class:`Clip`."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        path = sidecar_path(path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return clip_from_dict(data, base_dir=path.parent)


def update_sidecar(clip: Clip, extra: Optional[Mapping[str, Any]] = None) -> Path:
    """Rewrite ``clip``'s sidecar atomically, e.g. after scores were attached.

    Keys already in the sidecar that are not ``Clip`` fields (such as
    ``segments`` or ``cut_s``) are kept; ``extra`` keys are merged on top.
    Returns the sidecar path.
    """
    side = sidecar_path(clip.path)
    kept: Dict[str, Any] = {}
    if side.is_file():
        with open(side, encoding="utf-8") as fh:
            kept = json.load(fh)
    base = clip_to_dict(clip)
    merged: Dict[str, Any] = {k: v for k, v in kept.items() if k not in base}
    if extra:
        for key, value in extra.items():
            if key in base:
                raise ValueError(f"extra sidecar key {key!r} would overwrite a Clip field")
            merged[key] = value
    merged.update(base)
    _atomic_write_text(side, json.dumps(merged, indent=2, allow_nan=False) + "\n")
    return side


# --------------------------------------------------------------------------- writer


class ClipWriter:
    """Cuts clips for one session into ``<out_dir>/<channel>/<session>/clips/``.

    Safe to share between worker threads: every call uses its own concat list
    and its own ``.partial`` name.

    ``timeout_s``   FFmpeg timeout per cut (``RealtimeConfig.clip_timeout_s``).
    ``verify``      probe the result and fail if its duration differs from the
                    leased segments' total by more than ``duration_tolerance_s``
                    (catches truncated segments, which FFmpeg copies silently).
    ``tmp_dir``     where concat lists go; default: the system temp directory.
    """

    def __init__(
        self,
        out_dir: PathLike,
        session: str,
        timeout_s: float = 30.0,
        *,
        verify: bool = True,
        duration_tolerance_s: float = 0.5,
        tmp_dir: Optional[PathLike] = None,
    ) -> None:
        if not timeout_s > 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s!r}")
        self.out_dir = Path(out_dir)
        self.session = str(session)
        self.timeout_s = float(timeout_s)
        self.verify = verify
        self.duration_tolerance_s = float(duration_tolerance_s)
        self.tmp_dir = Path(tmp_dir) if tmp_dir is not None else None

    # ----------------------------------------------------------------- paths

    def clips_dir(self, channel: str) -> Path:
        """``<out_dir>/<channel>/<session>/clips`` (not created)."""
        return self.out_dir / _safe_component(channel) / _safe_component(self.session) / "clips"

    def clip_path(self, window: Window) -> Path:
        """Final mp4 path for ``window``: ``<channel>_<start_ms:09d>.mp4``."""
        start_ms = int(round(window.start * 1000.0))
        name = f"{_safe_component(window.channel)}_{start_ms:09d}.mp4"
        return self.clips_dir(window.channel) / name

    # ----------------------------------------------------------------- cutting

    def write(self, window: Window, lease: Any, pre_pad: float = 0.0, post_pad: float = 0.0) -> Clip:
        """Cut ``window`` from ``lease`` and return the finished :class:`Clip`.

        ``lease`` is a ``Lease`` (only its ordered ``segments`` are used; its
        ``truncated`` flag is recorded if present) or a plain sequence of
        segments. ``pre_pad``/``post_pad`` are the padding asked for; the file
        holds whatever part of ``[start - pre_pad, end + post_pad]`` the
        segments cover, and ``content_start``/``content_end`` are that range
        as offsets into the file, clamped to the file.

        Raises :class:`ClipError` on an empty or malformed lease, FFmpeg
        failure or timeout, or a failed check of the output; no files are left
        behind in that case.
        """
        t0 = time.perf_counter()
        segments = _lease_segments(lease)
        self._check_segments(window, segments)

        file_start = segments[0].start
        file_duration = sum(s.duration for s in segments)
        want_start = window.start - pre_pad - file_start
        want_end = window.end + post_pad - file_start
        content_start = min(max(want_start, 0.0), file_duration)
        content_end = min(max(want_end, content_start), file_duration)
        if window.start < file_start - _CONTIGUITY_EPS or window.end > file_start + file_duration + _CONTIGUITY_EPS:
            log.warning(
                "lease [%.3f, %.3f] does not cover window %s [%.3f, %.3f]",
                file_start, file_start + file_duration, window.window_id, window.start, window.end,
            )

        final = self.clip_path(window)
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_name(f"{final.name}.{uuid.uuid4().hex[:12]}.partial")
        side = sidecar_path(final)
        list_path: Optional[Path] = None
        renamed = False
        try:
            list_path = self._write_concat_list(segments)
            self._run_cut(list_path, partial)
            probed = self._verify(partial, file_duration) if self.verify else None
            os.replace(partial, final)
            renamed = True
            cut_s = time.perf_counter() - t0
            clip = Clip(
                window=window,
                path=final,
                file_start=file_start,
                file_duration=file_duration,
                content_start=content_start,
                content_end=content_end,
            )
            extra = {
                "session": self.session,
                "pre_pad": _num(pre_pad),
                "post_pad": _num(post_pad),
                "segments": [s.seq for s in segments],
                "truncated": bool(getattr(lease, "truncated", False)),
                "probed_duration": probed,
                "cut_s": round(cut_s, 4),
                "written_at": time.time(),
            }
            _atomic_write_text(side, json.dumps(clip_to_dict(clip, extra), indent=2, allow_nan=False) + "\n")
        except BaseException:
            _unlink_quietly(partial)
            if renamed:
                _unlink_quietly(final)
                _unlink_quietly(side)
            raise
        finally:
            _unlink_quietly(list_path)

        log.info(
            "clip %s: %d segments, %.2f s, cut in %.3f s",
            final.name, len(segments), file_duration, cut_s,
        )
        return clip

    # ----------------------------------------------------------------- internals

    @staticmethod
    def _check_segments(window: Window, segments: Sequence[Segment]) -> None:
        if not segments:
            raise ClipError(f"lease for window {window.window_id} has no segments")
        for prev, cur in zip(segments, segments[1:]):
            if cur.seq <= prev.seq:
                raise ClipError(f"lease segments out of order: seq {prev.seq} then {cur.seq}")
            if cur.discontinuity:
                raise ClipError(
                    f"lease for window {window.window_id} spans a discontinuity at seq {cur.seq}"
                )
            if abs(cur.start - prev.end) > _CONTIGUITY_EPS:
                raise ClipError(
                    f"lease segments not contiguous: seq {prev.seq} ends at {prev.end:.3f}, "
                    f"seq {cur.seq} starts at {cur.start:.3f}"
                )
        for seg in segments:
            try:
                size = Path(seg.path).stat().st_size
            except OSError as exc:
                raise ClipError(f"segment {seg.seq} is not readable: {seg.path} ({exc})") from exc
            if size == 0:
                raise ClipError(f"segment {seg.seq} is empty: {seg.path}")

    def _write_concat_list(self, segments: Iterable[Segment]) -> Path:
        tmp_dir = str(self.tmp_dir) if self.tmp_dir is not None else None
        if self.tmp_dir is not None:
            self.tmp_dir.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="vidflow-concat-", suffix=".txt", dir=tmp_dir)
        path = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("ffconcat version 1.0\n")
                for seg in segments:
                    fh.write(_concat_line(Path(seg.path)))
        except BaseException:
            _unlink_quietly(path)
            raise
        return path

    def _run_cut(self, list_path: Path, partial: Path) -> None:
        args = [
            "-xerror",  # a segment that cannot be opened or demuxed fails the cut instead of truncating it
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            # Video and audio only: HLS segments can carry ID3/data streams mp4 cannot hold.
            "-map", "0:v?",
            "-map", "0:a?",
            "-dn",
            "-c", "copy",
            "-bsf:a", "aac_adtstoasc",
            "-movflags", "+faststart",
            "-f", "mp4",  # the .partial extension says nothing about the format
            str(partial),
        ]
        try:
            run_ffmpeg(args, timeout=self.timeout_s)
        except MediaError as exc:
            message = str(exc).split("\n", 1)[0]
            raise ClipError(f"cutting {partial.name} failed: {message}", exc.stderr_tail) from exc
        if not partial.is_file() or partial.stat().st_size == 0:
            raise ClipError(f"ffmpeg produced no output for {partial.name}")

    def _verify(self, partial: Path, expected: float) -> float:
        try:
            info = probe(partial, timeout=self.timeout_s)
        except MediaError as exc:
            raise ClipError(f"cut output {partial.name} is not readable", exc.stderr_tail) from exc
        if not (info.has_video or info.has_audio):
            raise ClipError(f"cut output {partial.name} has no audio or video stream")
        if abs(info.duration - expected) > self.duration_tolerance_s:
            raise ClipError(
                f"cut output {partial.name} is {info.duration:.3f} s long but the leased segments "
                f"total {expected:.3f} s (a segment is probably truncated or corrupt)"
            )
        return info.duration
