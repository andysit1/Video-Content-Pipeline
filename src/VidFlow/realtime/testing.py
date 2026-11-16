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
    "SceneLayer",
    "DetectionScene",
    "detection_scenes",
    "render_scene",
    "scene_loudness",
    "make_scene_video",
    "write_truth",
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


# =========================================================================== detection scenes
#
# Synthetic audio that looks more like a stream than the sine-burst fixture:
# noise beds (white/pink/brown) with slow amplitude modulation, speech-like
# gating, transient spikes, and level changes, each scene with the known times
# of its "highlights" (the truth). Used to test the segmenter's threshold rule
# against streams that are loud most of the time. They are still synthetic: a
# pass here says nothing about whether the detected moments are worth clipping.


@dataclasses.dataclass(frozen=True)
class SceneLayer:
    """One sound in a :class:`DetectionScene`, present over ``[start, end)``.

    ``level_db`` is the layer's own RMS level in dBFS before modulation and
    gating. ``color`` is ``"white"``, ``"pink"``, ``"brown"`` (both band-limited
    to 20 Hz and up) or ``"tone"`` (440 Hz sine). ``mod_db``/``mod_hz`` add a
    slow sinusoidal level modulation of +/- ``mod_db``. ``gate`` makes it
    speech-like: alternating on/off runs with durations drawn uniformly from
    ``(on_min, on_max)`` and ``(off_min, off_max)`` seconds; off runs are
    ``gate_off_db`` below the level.
    """

    start: float
    end: float
    level_db: float
    color: str = "pink"
    mod_db: float = 0.0
    mod_hz: float = 0.2
    gate: Optional[Tuple[float, float, float, float]] = None
    gate_off_db: float = -30.0


@dataclasses.dataclass(frozen=True)
class DetectionScene:
    """A synthetic stream: layers summed, plus the highlight times a detector should find."""

    name: str
    duration: float
    layers: Tuple[SceneLayer, ...]
    truth: Tuple[Tuple[float, float], ...]
    description: str = ""
    seed: int = 0


def _scene_noise(color: str, n: int, rate: int, rng) -> "np.ndarray":  # noqa: F821
    import numpy as np

    if color == "tone":
        return np.sqrt(2.0) * np.sin(2.0 * np.pi * SYNTHETIC_TONE_HZ * np.arange(n) / rate)
    white = rng.standard_normal(n)
    if color == "white":
        x = white
    elif color in ("pink", "brown"):
        spec = np.fft.rfft(white)
        f = np.fft.rfftfreq(n, 1.0 / rate)
        shape = np.zeros_like(f)
        band = f >= 20.0  # no sub-audio wander
        shape[band] = 1.0 / (np.sqrt(f[band]) if color == "pink" else f[band])
        x = np.fft.irfft(spec * shape, n)
    else:
        raise ValueError(f"unknown noise color {color!r}")
    rms = float(np.sqrt(np.mean(np.square(x)))) if n else 1.0
    return x / rms if rms > 0 else x


def render_scene(scene: DetectionScene, rate: int = 16000) -> "np.ndarray":  # noqa: F821
    """Render ``scene`` as mono float32 samples at ``rate`` Hz, clipped to [-1, 1].

    Deterministic: the same scene and rate give the same samples.
    """
    import numpy as np

    n = int(round(scene.duration * rate))
    out = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(scene.seed)
    for k, layer in enumerate(scene.layers):
        lrng = np.random.default_rng([scene.seed, k])
        i0 = max(0, int(round(layer.start * rate)))
        i1 = min(n, int(round(layer.end * rate)))
        if i1 <= i0:
            continue
        m = i1 - i0
        x = _scene_noise(layer.color, m, rate, lrng)
        t = layer.start + np.arange(m) / rate
        gain_db = np.full(m, float(layer.level_db))
        if layer.mod_db:
            phase = float(rng.uniform(0, 2 * np.pi))
            gain_db += layer.mod_db * np.sin(2 * np.pi * layer.mod_hz * t + phase)
        if layer.gate is not None:
            on0, on1, off0, off1 = layer.gate
            mask = np.zeros(m, dtype=bool)
            i, on = 0, True
            while i < m:
                d = lrng.uniform(on0, on1) if on else lrng.uniform(off0, off1)
                j = min(m, i + max(1, int(round(d * rate))))
                mask[i:j] = on
                i, on = j, not on
            gain_db[~mask] += layer.gate_off_db
        out[i0:i1] += x * np.power(10.0, gain_db / 20.0)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def scene_loudness(scene: DetectionScene, hop_s: float = 0.1, rate: int = 16000) -> List["LoudnessSample"]:  # noqa: F821
    """The scene's loudness series as the extractor would report it (RMS dBFS per hop), without a codec."""
    from .audio import rms_db
    from .types import LoudnessSample

    audio = render_scene(scene, rate)
    hop = int(round(hop_s * rate))
    n = audio.size // hop
    dbs = rms_db(audio[: n * hop].reshape(n, hop))
    return [LoudnessSample(t=k * hop / rate, db=float(db)) for k, db in enumerate(dbs.tolist())]


def make_scene_video(
    path: PathLike,
    scene: DetectionScene,
    *,
    size: Tuple[int, int] = SYNTHETIC_SIZE,
    fps: int = 10,
    sample_rate: int = 48000,
) -> Path:
    """Encode ``scene`` as a video (``testsrc2`` picture, keyframe every second, stereo AAC).

    Same encoding settings as :func:`make_synthetic_video`; written to a
    temporary name and renamed into place.
    """
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = render_scene(scene, sample_rate)
    stereo = np.repeat(audio[:, None], 2, axis=1)
    token = uuid.uuid4().hex
    raw = path.with_name(f".{path.stem}.{token}.f32")
    tmp = path.with_name(f".{path.stem}.{token}.partial{path.suffix or '.mp4'}")
    w, h = size
    try:
        stereo.astype("<f4").tofile(raw)
        run_ffmpeg(
            [
                "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps}:duration={scene.duration}",
                "-f", "f32le", "-ar", str(sample_rate), "-ac", "2", "-i", str(raw),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-bf", "0", "-g", str(fps), "-keyint_min", str(fps), "-sc_threshold", "0",
                "-force_key_frames", "expr:gte(t,n_forced*1)",
                "-c:a", "aac", "-b:a", "128k",
                "-t", f"{scene.duration}",
                "-f", "mp4", str(tmp),
            ],
            timeout=600,
        )
        os.replace(tmp, path)
    finally:
        for p in (raw, tmp):
            if p.exists():
                p.unlink()
    return path


def write_truth(path: PathLike, windows: Sequence[Tuple[float, float]]) -> Path:
    """Write ``[[start, end], ...]`` as JSON, the format ``eval --truth`` reads."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([[float(s), float(e)] for s, e in windows]), encoding="utf-8")
    return path


def _spikes(seed: int, t0: float, t1: float, gap: Tuple[float, float], dur: Tuple[float, float], level_db: float,
            avoid: Sequence[Tuple[float, float]] = ()) -> List[SceneLayer]:
    """Short white-noise transients (gunshots, clicks) at random times, not inside ``avoid``."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out: List[SceneLayer] = []
    t = t0 + float(rng.uniform(*gap))
    while t < t1:
        d = float(rng.uniform(*dur))
        if not any(a - 1.0 <= t <= b + 1.0 for a, b in avoid):
            out.append(SceneLayer(round(t, 3), round(min(t + d, t1), 3), level_db, "white"))
        t += d + float(rng.uniform(*gap))
    return out


def detection_scenes() -> dict:
    """The named synthetic scenes used to test the detection rule, ``{name: DetectionScene}``.

    Every scene has a highlight inside the first 30 s (the default warm-up).
    Highlights are at least 3 s long and more than ``merge_gap_s`` (4 s) apart.

    ``loud_bed``        pink-noise bed at -20 dBFS (+/-3 dB slow modulation) with
                        highlights 9-14 dB above it (levels differ).
    ``mostly_loud``     a -18 dBFS bed broken by 1.5 s near-silent gaps every
                        ~12 s, highlights 8-13 dB above the bed.
    ``quiet_then_loud`` a quiet first half (-48 dBFS) with two highlights, then
                        a sustained loud bed (-18 dBFS) with three louder ones.
    ``noise_bed``       brown + pink noise with independent slow modulation, white
                        transient spikes every 4.5-9 s, noisy modulated highlights.
    ``speechlike``      gated noise like commentary (-20 dBFS when "talking") over a
                        quiet game bed; highlights are louder gated "shouting".
    ``loud_then_quiet`` the reverse of ``quiet_then_loud``: second-half highlights
                        are quieter than the first half's bed.
    ``dense_peaks``     highlights 14-18 dB over the bed filling ~48 % of the stream.
    ``steady_noise``    modulated noise beds and nothing else (no truth windows).
    ``lull_resume``     commentary, a 2 min silent lull, then commentary again; the
                        resumption is not a highlight.
    ``sparse_speech``   a quiet minute, then utterances separated by >4 s pauses
                        (each its own loud run) with louder shouts.
    """
    scenes: List[DetectionScene] = []

    peaks = ((6.0, 11.0), (35.0, 41.0), (70.0, 74.0), (100.0, 108.0), (140.0, 143.0), (165.0, 171.0))
    scenes.append(DetectionScene(
        "loud_bed", 180.0,
        (SceneLayer(0, 180, -20.0, "pink", mod_db=3.0, mod_hz=0.13),)
        + tuple(SceneLayer(a, b, lv, "pink") for (a, b), lv in zip(peaks, (-8.0, -6.0, -11.0, -7.0, -10.0, -9.0))),
        peaks, "loud bed with louder peaks", seed=1,
    ))

    peaks = ((8.0, 12.0), (40.0, 46.0), (75.0, 79.0), (110.0, 117.0), (150.0, 154.0))
    gaps = [(g, g + 1.5) for g in (3.0, 15.0, 27.0, 33.0, 52.0, 63.0, 86.0, 98.0, 126.0, 138.0, 162.0, 174.0)]
    bed: List[SceneLayer] = []
    t = 0.0
    for g0, g1 in gaps:
        bed.append(SceneLayer(t, g0, -18.0, "pink", mod_db=2.0, mod_hz=0.21))
        bed.append(SceneLayer(g0, g1, -50.0, "pink"))
        t = g1
    bed.append(SceneLayer(t, 180.0, -18.0, "pink", mod_db=2.0, mod_hz=0.21))
    scenes.append(DetectionScene(
        "mostly_loud", 180.0,
        tuple(bed) + tuple(SceneLayer(a, b, lv, "pink") for (a, b), lv in zip(peaks, (-7.0, -5.0, -10.0, -6.0, -9.0))),
        peaks, "mostly loud with short quiet gaps", seed=2,
    ))

    first = ((20.0, 26.0), (70.0, 75.0))
    second = ((150.0, 155.0), (185.0, 192.0), (220.0, 224.0))
    scenes.append(DetectionScene(
        "quiet_then_loud", 240.0,
        (SceneLayer(0, 240, -48.0, "pink"), SceneLayer(120, 240, -18.0, "pink", mod_db=2.0, mod_hz=0.17))
        + tuple(SceneLayer(a, b, -10.0, "pink") for a, b in first)
        + tuple(SceneLayer(a, b, -6.0, "pink") for a, b in second),
        first + second, "quiet first half, sustained loud second half with peaks", seed=3,
    ))

    peaks = ((12.0, 18.0), (50.0, 57.0), (95.0, 99.0), (130.0, 140.0), (160.0, 164.0))
    scenes.append(DetectionScene(
        "noise_bed", 180.0,
        (SceneLayer(0, 180, -26.0, "brown", mod_db=5.0, mod_hz=0.07),
         SceneLayer(0, 180, -30.0, "pink", mod_db=4.0, mod_hz=0.31))
        + tuple(_spikes(40, 0.0, 180.0, (4.5, 9.0), (0.05, 0.3), -12.0, avoid=peaks))
        + tuple(SceneLayer(a, b, lv, "pink", mod_db=3.0, mod_hz=0.5) for (a, b), lv in zip(peaks, (-12.0, -9.0, -15.0, -11.0, -13.0))),
        peaks, "modulated pink/brown noise with transient spikes", seed=4,
    ))

    peaks = ((15.0, 20.0), (60.0, 66.0), (100.0, 104.0), (140.0, 148.0))
    scenes.append(DetectionScene(
        "speechlike", 180.0,
        (SceneLayer(0, 180, -38.0, "brown", mod_db=3.0, mod_hz=0.05),
         SceneLayer(0, 180, -20.0, "pink", gate=(0.1, 0.5, 0.05, 0.6), gate_off_db=-35.0))
        + tuple(SceneLayer(a, b, -9.0, "pink", gate=(0.2, 0.8, 0.05, 0.3), gate_off_db=-30.0) for a, b in peaks),
        peaks, "gated 'commentary' with louder 'shouting'", seed=5,
    ))

    first = ((8.0, 13.0), (80.0, 86.0))
    second = ((150.0, 155.0), (200.0, 206.0))
    scenes.append(DetectionScene(
        "loud_then_quiet", 240.0,
        (SceneLayer(0, 120, -18.0, "pink", mod_db=2.0, mod_hz=0.17), SceneLayer(120, 240, -48.0, "pink"))
        + tuple(SceneLayer(a, b, -6.0, "pink") for a, b in first)
        + tuple(SceneLayer(a, b, -24.0, "pink") for a, b in second),
        first + second, "loud first half, quiet second half with quieter peaks", seed=6,
    ))

    peaks = tuple((float(a), float(a + d)) for a, d in (
        (5, 6), (17, 5), (28, 8), (42, 4), (52, 7), (66, 5), (77, 6), (90, 8), (104, 5), (115, 4),
        (125, 7), (138, 5), (149, 6), (161, 8), (175, 4), (186, 6), (198, 5), (209, 7), (222, 4), (232, 5),
    ))
    levels = (-10.0, -8.0, -12.0, -9.0, -11.0)
    scenes.append(DetectionScene(
        "dense_peaks", 240.0,
        (SceneLayer(0, 240, -26.0, "pink", mod_db=3.0, mod_hz=0.11),)
        + tuple(SceneLayer(a, b, levels[i % len(levels)], "pink", mod_db=2.0, mod_hz=0.4)
                for i, (a, b) in enumerate(peaks)),
        peaks, "highlights fill about 45 % of the stream", seed=7,
    ))

    scenes.append(DetectionScene(
        "steady_noise", 180.0,
        (SceneLayer(0, 180, -30.0, "pink", mod_db=4.0, mod_hz=0.09),
         SceneLayer(0, 180, -36.0, "brown", mod_db=3.0, mod_hz=0.23)),
        (), "modulated noise and nothing else: every window is a false positive", seed=8,
    ))

    peaks = ((10.0, 15.0), (200.0, 206.0), (250.0, 255.0))
    scenes.append(DetectionScene(
        "lull_resume", 300.0,
        (SceneLayer(0, 300, -40.0, "brown", mod_db=3.0, mod_hz=0.05),
         SceneLayer(0, 60, -20.0, "pink", gate=(0.1, 0.5, 0.05, 0.6), gate_off_db=-35.0),
         SceneLayer(180, 300, -20.0, "pink", gate=(0.1, 0.5, 0.05, 0.6), gate_off_db=-35.0))
        + tuple(SceneLayer(a, b, -9.0, "pink", gate=(0.2, 0.8, 0.05, 0.3), gate_off_db=-30.0) for a, b in peaks),
        peaks, "commentary, a 2 min silent lull, commentary resumes: the resumption is not a highlight", seed=9,
    ))

    peaks = ((12.0, 17.0), (130.0, 136.0), (190.0, 194.0), (250.0, 257.0))
    scenes.append(DetectionScene(
        "sparse_speech", 300.0,
        (SceneLayer(0, 300, -40.0, "brown", mod_db=3.0, mod_hz=0.05),
         SceneLayer(60, 300, -20.0, "pink", gate=(2.0, 8.0, 4.5, 9.0), gate_off_db=-40.0))
        + tuple(SceneLayer(a, b, -8.0, "pink", gate=(0.2, 0.8, 0.05, 0.3), gate_off_db=-30.0) for a, b in peaks),
        peaks, "a quiet minute, then utterances separated by >4 s pauses (each its own loud run) "
        "with louder shouts: the utterances must become 'normal'", seed=10,
    ))
    return {s.name: s for s in scenes}
