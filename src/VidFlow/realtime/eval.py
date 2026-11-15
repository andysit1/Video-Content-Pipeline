"""Evaluation, re-tuning and benchmarking of the real-time clipper (design section 9).

Three tools, each a plain function here and a ``vidflow-live`` subcommand in
``cli.py``:

:func:`evaluate`
    Runs the pipeline over a video unpaced and detect-only (no clips are cut),
    then compares its windows with reference windows: either loaded from a
    truth file or built offline from the whole file by :func:`reference_windows`
    (the batch pipeline's ``volumedetect`` + ``silencedetect`` rule, written
    fresh here). Reports precision and recall at an overlap (IoU) threshold
    and the boundary errors of the matched pairs.

:func:`retune`
    Re-runs the :class:`~VidFlow.realtime.segmenter.OnlineSegmenter` over the
    ``audio`` events of a recorded ``events.jsonl`` with changed settings. No
    video is read. With the logged settings it reproduces the live windows
    exactly.

:func:`run_bench`
    N simulated channels (``ReplaySource`` at 1x) under one
    :class:`~VidFlow.realtime.supervisor.Supervisor` for M seconds; reports
    process CPU, memory, ingest lag, how far behind the broadcast schedule
    ingest fell, latency percentiles and clips per minute.

The offline reference
---------------------
The batch pipeline (``pipes/data_cache_stage.py`` and
``pipes/analyze_data_stage.py``, not imported) did this:

1. ``volumedetect`` over the whole file gives a histogram of sample peak
   levels in 1 dB bins, loudest first. The threshold is the first (loudest)
   bin holding more than 20,000 samples.
2. ``silencedetect`` at that threshold with ``d=0.5``: a run of samples all
   below the threshold lasting at least 0.5 s is silence; everything between
   silences is a chunk.
3. Chunks closer than 4 s are merged; chunks shorter than 1.5 s are dropped.

:func:`reference_windows` computes the same thing in one streaming pass with
PyAV and NumPy, with two deliberate corrections:

- The batch code used the bin's *upper* edge as the threshold (bin ``N``,
  levels in ``[-N-1, -N)`` dB, gave ``-N`` dB), so the samples that selected
  the bin are all below it. On a constant-level burst (the test fixture) every
  sample is then "silent" and no chunk is found; bin 0 gave a threshold of 0,
  which the batch code treated as an error. Here the threshold is the bin's
  lower edge, so the selecting samples count as loud.
- The merge loop in the batch code mutated the list it iterated over and so
  skipped chunks; here every gap of at most ``merge_gap_s`` is merged.

Levels are tracked per 10 ms block (the block's sample peak over all
channels), so boundaries are exact to 10 ms and memory stays at about
100 floats per second of audio, whatever the file's length.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import av
import numpy as np

from .config import RealtimeConfig
from .pipeline import ChannelPipeline, ChannelResult
from .segmenter import OnlineSegmenter
from .session import (
    LatencyTracker,
    read_config,
    read_events,
    samples_from_event,
    window_from_event,
)
from .sources.replay import ReplaySource
from .supervisor import Supervisor
from .types import Window

__all__ = [
    "RefWindow",
    "Reference",
    "reference_windows",
    "load_truth",
    "Match",
    "match_windows",
    "EvalReport",
    "evaluate",
    "format_eval",
    "RetuneResult",
    "retune",
    "format_retune",
    "BenchReport",
    "run_bench",
    "format_bench",
    "RETUNE_FIXED",
]

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

# Batch-pipeline constants (pipes/data_cache_stage.py, pipes/analyze_data_stage.py).
BATCH_MIN_COUNT = 20000  # samples a volumedetect bin must exceed to set the threshold
BATCH_SILENCE_S = 0.5  # silencedetect d=
BATCH_MERGE_GAP_S = 4.0  # merge chunks this close
BATCH_MIN_LEN_S = 1.5  # drop chunks shorter than this

_BLOCK_S = 0.01  # resolution of the offline reference
_N_BINS = 200  # volumedetect-style bins: 0 .. 199 dB below full scale


# =========================================================================== reference


@dataclass(frozen=True)
class RefWindow:
    """A reference (ground-truth) window in media time."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Reference:
    """Reference windows and how they were obtained."""

    windows: List[RefWindow]
    method: str  # "silencedetect" or "truth"
    threshold_db: Optional[float] = None
    duration: Optional[float] = None
    source: Optional[str] = None  # truth file path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "threshold_db": self.threshold_db,
            "duration": self.duration,
            "source": self.source,
            "windows": [[w.start, w.end] for w in self.windows],
        }


def _audio_blocks(video: PathLike) -> Tuple[np.ndarray, np.ndarray, int]:
    """Decode the whole audio track once.

    Returns ``(block_peak, histogram, rate)``: the sample peak (absolute value,
    over all channels) of each 10 ms block, and the count of channel samples
    per 1 dB bin below full scale (bin ``N`` holds levels in
    ``[-N-1, -N)`` dBFS, like ``volumedetect``; digital silence goes in the
    last bin).
    """
    hist = np.zeros(_N_BINS, dtype=np.int64)
    peaks: List[np.ndarray] = []
    carry = np.zeros(0, dtype=np.float32)
    rate = 0
    block = 0
    with av.open(str(video), mode="r") as container:
        if not container.streams.audio:
            raise ValueError(f"{video} has no audio stream")
        stream = container.streams.audio[0]
        resampler: Optional[av.AudioResampler] = None
        key: Optional[Tuple[str, int]] = None

        def consume(frames: Iterable[av.AudioFrame]) -> None:
            nonlocal carry
            for out in frames:
                x = out.to_ndarray()  # planar float: (channels, samples)
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                a = np.abs(x.astype(np.float32, copy=False))
                with np.errstate(divide="ignore"):
                    lvl = np.floor(-20.0 * np.log10(a))  # 0 -> +inf (digital silence)
                idx = np.clip(np.nan_to_num(lvl, nan=_N_BINS - 1, posinf=_N_BINS - 1), 0, _N_BINS - 1).astype(np.int64)
                hist[:] += np.bincount(idx.ravel(), minlength=_N_BINS)
                per_t = a.max(axis=0)
                buf = np.concatenate((carry, per_t)) if carry.size else per_t
                n = buf.size // block
                if n:
                    peaks.append(buf[: n * block].reshape(n, block).max(axis=1))
                carry = buf[n * block :].copy()

        for frame in container.decode(stream):
            k = (frame.layout.name, int(frame.sample_rate))
            if resampler is None or k != key:
                if resampler is not None:
                    consume(resampler.resample(None))
                rate = int(frame.sample_rate)
                block = max(1, int(round(rate * _BLOCK_S)))
                # Same rate and layout: a format conversion only, sample values unchanged.
                resampler = av.AudioResampler(format="fltp", layout=frame.layout.name, rate=rate)
                key = k
            consume(resampler.resample(frame))
        if resampler is not None:
            consume(resampler.resample(None))
    if carry.size:
        peaks.append(np.array([carry.max()], dtype=np.float32))
    block_peak = np.concatenate(peaks) if peaks else np.zeros(0, dtype=np.float32)
    return block_peak, hist, rate


def _batch_threshold_db(hist: np.ndarray, min_count: int) -> Optional[float]:
    """The batch rule: the loudest bin holding more than ``min_count`` samples.

    Returns that bin's lower edge in dBFS (see the module docstring), or
    ``None`` when no bin qualifies.
    """
    for n in range(_N_BINS - 1):  # the last bin is digital silence
        if hist[n] > min_count:
            return -float(n + 1)
    return None


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """``[start, end)`` index runs where ``mask`` is True."""
    if not mask.size:
        return []
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def merge_intervals(
    intervals: Sequence[Tuple[float, float]], merge_gap_s: float, min_len_s: float
) -> List[RefWindow]:
    """Merge intervals whose gap is at most ``merge_gap_s``; drop those shorter than ``min_len_s``."""
    merged: List[List[float]] = []
    for s, e in sorted(intervals):
        if merged and s - merged[-1][1] <= merge_gap_s + 1e-9:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [RefWindow(round(s, 3), round(e, 3)) for s, e in merged if e - s >= min_len_s - 1e-9]


def reference_windows(
    video: PathLike,
    *,
    min_count: int = BATCH_MIN_COUNT,
    silence_s: float = BATCH_SILENCE_S,
    merge_gap_s: float = BATCH_MERGE_GAP_S,
    min_len_s: float = BATCH_MIN_LEN_S,
    threshold_db: Optional[float] = None,
) -> Reference:
    """Offline reference windows for ``video`` from its whole audio track.

    The batch pipeline's rule (module docstring): threshold from the
    whole-file level histogram (or ``threshold_db`` if given), non-silent
    chunks between silences of at least ``silence_s``, then a ``merge_gap_s``
    merge and a ``min_len_s`` minimum. Times are seconds from the first audio
    sample.
    """
    peaks, hist, rate = _audio_blocks(video)
    duration = peaks.size * _BLOCK_S
    thr = threshold_db if threshold_db is not None else _batch_threshold_db(hist, min_count)
    if thr is None:
        log.warning("%s: no level bin holds more than %d samples; no reference windows", video, min_count)
        return Reference([], "silencedetect", None, duration)
    loud = peaks >= 10.0 ** (thr / 20.0)
    chunks: List[Tuple[float, float]] = []
    min_quiet = int(round(silence_s / _BLOCK_S))
    for s, e in _runs(loud):
        # A quiet gap shorter than silencedetect's d= is not a silence: bridge it.
        if chunks and s - chunks[-1][1] < min_quiet:
            chunks[-1] = (chunks[-1][0], e)
        else:
            chunks.append((s, e))
    seconds = [(s * _BLOCK_S, e * _BLOCK_S) for s, e in chunks]
    return Reference(merge_intervals(seconds, merge_gap_s, min_len_s), "silencedetect", thr, round(duration, 3))


def load_truth(path: PathLike) -> Reference:
    """Load reference windows from a file.

    JSON: a list of ``[start, end]`` pairs or of objects with ``start`` and
    ``end``, optionally under a ``"windows"`` key (so the ``reference`` block
    of an ``eval`` JSON report can be fed back). Any other file: one
    window per line, ``start end`` or ``start,end``; blank lines and lines
    starting with ``#`` are ignored.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    items: List[Any]
    try:
        data = json.loads(text)
    except ValueError:
        items = []
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                raise ValueError(f"{p}:{lineno}: expected 'start end', got {line!r}")
            items.append((parts[0], parts[1]))
    else:
        if isinstance(data, Mapping):
            data = data.get("windows")
        if not isinstance(data, list):
            raise ValueError(f"{p}: expected a list of windows")
        items = data
    out: List[RefWindow] = []
    for item in items:
        if isinstance(item, Mapping):
            s, e = item["start"], item["end"]
        else:
            s, e = item[0], item[1]
        s, e = float(s), float(e)
        if not e > s:
            raise ValueError(f"{p}: window end {e} is not after start {s}")
        out.append(RefWindow(s, e))
    out.sort(key=lambda w: w.start)
    return Reference(out, "truth", source=str(p))


# =========================================================================== matching


@dataclass(frozen=True)
class Match:
    ref: RefWindow
    det: Window
    iou: float

    @property
    def start_error(self) -> float:
        """Detected start minus reference start (positive: detected late)."""
        return self.det.start - self.ref.start

    @property
    def end_error(self) -> float:
        return self.det.end - self.ref.end


def iou(a0: float, a1: float, b0: float, b1: float) -> float:
    """Intersection over union of two intervals."""
    inter = min(a1, b1) - max(a0, b0)
    if inter <= 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def match_windows(
    refs: Sequence[RefWindow], dets: Sequence[Window], min_iou: float = 0.5
) -> List[Match]:
    """One-to-one matches with IoU at least ``min_iou``, best pairs first."""
    pairs = []
    for i, r in enumerate(refs):
        for j, d in enumerate(dets):
            v = iou(r.start, r.end, d.start, d.end)
            if v >= min_iou - 1e-12 and v > 0:
                pairs.append((v, i, j))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_r, used_d = set(), set()
    out: List[Match] = []
    for v, i, j in pairs:
        if i in used_r or j in used_d:
            continue
        used_r.add(i)
        used_d.add(j)
        out.append(Match(refs[i], dets[j], v))
    out.sort(key=lambda m: m.ref.start)
    return out


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


@dataclass
class EvalReport:
    """Result of :func:`evaluate`."""

    video: str
    reference: Reference
    detected: List[Window]
    matches: List[Match]
    min_iou: float
    config: RealtimeConfig
    session_dir: Optional[Path] = None
    detect_s: float = 0.0  # wall seconds of the online run
    reference_s: float = 0.0  # wall seconds to build the reference

    @property
    def precision(self) -> Optional[float]:
        return len(self.matches) / len(self.detected) if self.detected else None

    @property
    def recall(self) -> Optional[float]:
        return len(self.matches) / len(self.reference.windows) if self.reference.windows else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)

    def to_dict(self) -> Dict[str, Any]:
        se = [m.start_error for m in self.matches]
        ee = [m.end_error for m in self.matches]
        return {
            "video": self.video,
            "session_dir": str(self.session_dir) if self.session_dir else None,
            "min_iou": self.min_iou,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "matched": len(self.matches),
            "n_reference": len(self.reference.windows),
            "n_detected": len(self.detected),
            "mean_start_error": _mean(se),
            "mean_end_error": _mean(ee),
            "mean_abs_start_error": _mean([abs(v) for v in se]),
            "mean_abs_end_error": _mean([abs(v) for v in ee]),
            "reference": self.reference.to_dict(),
            "detected": [
                {"window_id": w.window_id, "start": w.start, "end": w.end, "forced": w.forced, "stats": dict(w.stats)}
                for w in self.detected
            ],
            "matches": [
                {
                    "ref": [m.ref.start, m.ref.end],
                    "window_id": m.det.window_id,
                    "iou": m.iou,
                    "start_error": m.start_error,
                    "end_error": m.end_error,
                }
                for m in self.matches
            ],
            "config": self.config.to_dict(),
            "timing_s": {"detect": self.detect_s, "reference": self.reference_s},
        }


def evaluate(
    video: PathLike,
    config: Optional[RealtimeConfig] = None,
    *,
    truth: Optional[PathLike] = None,
    out_dir: Optional[PathLike] = None,
    session: Optional[str] = None,
    channel: str = "eval",
    min_iou: float = 0.5,
    transcode: bool = False,
) -> EvalReport:
    """Run the online detector over ``video`` and score it against a reference.

    The pipeline runs unpaced and ``detect_only`` (no clips, no job queue),
    logging to ``<out_dir>/<channel>/<session>/events.jsonl`` so the run can
    be fed to :func:`retune`. With ``out_dir=None`` a temporary directory is
    used and removed afterwards.
    """
    cfg = (config if config is not None else RealtimeConfig()).validate()
    video = Path(video)
    tmp: Optional[str] = None
    if out_dir is None:
        tmp = tempfile.mkdtemp(prefix="vidflow-eval-")
        out_dir = tmp
    try:
        t0 = time.perf_counter()
        pipe = ChannelPipeline(
            lambda d: ReplaySource(video, channel, d, speed=None, segment_time=cfg.segment_time_s, transcode=transcode),
            cfg,
            out_dir,
            channel=channel,
            session=session,
            detect_only=True,
        )
        res = pipe.run()
        detect_s = time.perf_counter() - t0
        if not res.ok:
            raise RuntimeError(f"pipeline failed on {video}: {res.error}")
        detected = sorted(pipe.windows, key=lambda w: w.start)

        t0 = time.perf_counter()
        ref = load_truth(truth) if truth is not None else reference_windows(video)
        ref_s = time.perf_counter() - t0
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    return EvalReport(
        video=str(video),
        reference=ref,
        detected=detected,
        matches=match_windows(ref.windows, detected, min_iou),
        min_iou=min_iou,
        config=cfg,
        session_dir=None if tmp is not None else res.session_dir,
        detect_s=round(detect_s, 3),
        reference_s=round(ref_s, 3),
    )


def _f(v: Optional[float], spec: str = ".2f") -> str:
    return "-" if v is None else format(v, spec)


def format_eval(report: EvalReport) -> List[str]:
    """A short text table of an :class:`EvalReport`."""
    ref = report.reference
    lines = [f"video: {report.video}"]
    how = ref.method if ref.method != "truth" else f"truth file {ref.source}"
    if ref.threshold_db is not None:
        how += f", threshold {ref.threshold_db:.0f} dBFS"
    lines.append(f"reference: {len(ref.windows)} windows ({how})")
    lines.append(f"detected:  {len(report.detected)} windows (online, unpaced, detect only)")
    lines.append("")
    lines.append(f"  {'reference':>17}   {'detected':>17}  {'IoU':>5}  {'d_start':>7}  {'d_end':>7}")
    by_ref = {id(m.ref): m for m in report.matches}
    matched_dets = {id(m.det) for m in report.matches}
    rows: List[Tuple[float, str]] = []
    for r in ref.windows:
        m = by_ref.get(id(r))
        rr = f"{r.start:8.2f}-{r.end:8.2f}"
        if m is None:
            rows.append((r.start, f"  {rr}   {'(missed)':>17}"))
        else:
            d = m.det
            rows.append((r.start, f"  {rr}   {d.start:8.2f}-{d.end:8.2f}  {m.iou:5.2f}  {m.start_error:+7.2f}  {m.end_error:+7.2f}"))
    for d in report.detected:
        if id(d) not in matched_dets:
            rows.append((d.start, f"  {'(none)':>17}   {d.start:8.2f}-{d.end:8.2f}"))
    lines.extend(text for _, text in sorted(rows, key=lambda x: x[0]))
    d = report.to_dict()
    lines.append("")
    lines.append(
        f"precision {_f(d['precision'])}  recall {_f(d['recall'])}  f1 {_f(d['f1'])}  "
        f"(IoU >= {report.min_iou:g}; {d['matched']} matched)"
    )
    lines.append(
        f"mean start error {_f(d['mean_start_error'], '+.2f')} s (|.| {_f(d['mean_abs_start_error'])}), "
        f"mean end error {_f(d['mean_end_error'], '+.2f')} s (|.| {_f(d['mean_abs_end_error'])})"
    )
    lines.append(f"time: online {report.detect_s:.2f} s, reference {report.reference_s:.2f} s")
    return lines


# =========================================================================== retune

#: Settings that change the logged samples themselves, so cannot be re-tuned without the video.
RETUNE_FIXED = ("hop_s",)


@dataclass
class RetuneResult:
    """Result of :func:`retune`."""

    events: Path
    config: RealtimeConfig
    windows: List[Window]
    logged: List[Window]  # the windows the live run logged
    changed: Dict[str, Tuple[Any, Any]]  # setting -> (logged, new)
    samples: int = 0
    discontinuities: int = 0

    @property
    def identical(self) -> bool:
        """True when the windows equal the logged ones exactly (ids, times, stats, forced)."""
        return [_wkey(w) for w in self.windows] == [_wkey(w) for w in self.logged]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "events": str(self.events),
            "changed": {k: list(v) for k, v in self.changed.items()},
            "samples": self.samples,
            "discontinuities": self.discontinuities,
            "identical_to_logged": self.identical,
            "windows": [_wdict(w) for w in self.windows],
            "logged": [_wdict(w) for w in self.logged],
            "config": self.config.to_dict(),
        }


def _wkey(w: Window) -> Tuple:
    return (w.window_id, w.start, w.end, w.forced, tuple(sorted(w.stats.items())))


def _wdict(w: Window) -> Dict[str, Any]:
    return {"window_id": w.window_id, "start": w.start, "end": w.end, "forced": w.forced, "stats": dict(w.stats)}


def retune(events: PathLike, config: Optional[RealtimeConfig] = None, **changes: Any) -> RetuneResult:
    """Re-run the segmenter over a session log's ``audio`` events.

    ``events`` is an ``events.jsonl`` or the session directory. The settings
    are ``config`` if given, else the ones logged in ``session_start``, with
    ``changes`` applied on top. A discontinuity flag on an ``audio`` event
    resets the segmenter before that segment's samples, as the live
    ``AudioDetector`` does, and the run ends with a flush as the live
    shutdown does. No media is read.

    Raises ``ValueError`` if a setting in :data:`RETUNE_FIXED` differs from
    the logged one (the samples would have to be re-extracted from the video).
    """
    path = Path(events)
    logged_cfg = read_config(path)
    starts = list(read_events(path, "session_start"))
    channel = (starts[0].get("channel") if starts else None) or "channel"
    base = config if config is not None else (logged_cfg or RealtimeConfig())
    cfg = base.replace(**changes) if changes else base

    audio = list(read_events(path, "audio"))
    hops = {float(e["hop_s"]) for e in audio if e.get("hop_s") is not None}
    ref_hop = logged_cfg.hop_s if logged_cfg is not None else (hops.pop() if len(hops) == 1 else None)
    for name in RETUNE_FIXED:
        logged_value = getattr(logged_cfg, name) if logged_cfg is not None else ref_hop
        if logged_value is not None and getattr(cfg, name) != logged_value:
            raise ValueError(f"{name} cannot be re-tuned from a log (logged {logged_value}); re-run on the video")
    # The buffer is not used here; do not let its retention rule block a re-tune.
    if cfg.retention_s < cfg.min_retention_s():
        cfg = cfg.replace(retention_s=cfg.min_retention_s())
    cfg.validate()

    seg = OnlineSegmenter(channel, cfg)
    windows: List[Window] = []
    n = disc = 0
    for event in audio:
        if event.get("discontinuity"):
            disc += 1
            windows.extend(seg.reset())
        samples = samples_from_event(event)
        n += len(samples)
        windows.extend(seg.feed(samples))
    windows.extend(seg.flush())

    logged = [window_from_event(e) for e in read_events(path, "window")]
    reference = logged_cfg.to_dict() if logged_cfg is not None else {}
    changed = {
        k: (reference.get(k), v) for k, v in cfg.to_dict().items() if k in reference and reference[k] != v
    }
    changed.pop("retention_s", None)
    return RetuneResult(path, cfg, windows, logged, changed, samples=n, discontinuities=disc)


def _window_rows(windows: Sequence[Window]) -> List[str]:
    lines = [f"  {'window':<24}{'start':>8}{'end':>8}{'dur':>6}{'peak':>7}{'thr':>7}{'prom':>6}  forced"]
    for w in windows:
        s = w.stats
        lines.append(
            f"  {w.window_id:<24}{w.start:8.2f}{w.end:8.2f}{w.duration:6.1f}"
            f"{s.get('peak_db', float('nan')):7.1f}{s.get('threshold_db', float('nan')):7.1f}"
            f"{s.get('prominence_db', float('nan')):6.1f}  {'yes' if w.forced else ''}"
        )
    return lines


def format_retune(result: RetuneResult) -> List[str]:
    lines = [f"log: {result.events}"]
    lines.append(
        f"{result.samples} loudness samples ({result.samples * result.config.hop_s:.1f} s), "
        f"{result.discontinuities} discontinuities"
    )
    if result.changed:
        lines.append("changed: " + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in sorted(result.changed.items())))
    else:
        lines.append("changed: nothing (logged settings)")
    lines.append(f"{len(result.windows)} windows:")
    lines.extend(_window_rows(result.windows))
    if result.identical:
        lines.append(f"identical to the {len(result.logged)} logged windows")
    else:
        lines.append(f"differs from the {len(result.logged)} logged windows:")
        lines.extend(_window_rows(result.logged))
    return lines


# =========================================================================== bench


def _cpu_times() -> Tuple[float, float]:
    """(this process, its waited-for children) CPU seconds, user + system."""
    t = os.times()
    return t.user + t.system, t.children_user + t.children_system


def _peak_rss_mb() -> Tuple[Optional[float], Optional[float]]:
    """Peak resident memory of this process and of its largest child, in MB (None if unknown)."""
    try:
        import resource
    except ImportError:  # Windows
        return None, None
    scale = 1.0 / 1024 if sys.platform != "darwin" else 1.0 / (1024 * 1024)  # KB on Linux, bytes on macOS
    me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    return round(me, 1), round(kids, 1)


def _rss_mb() -> Optional[float]:
    try:
        with open("/proc/self/statm", "rb") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1e6
    except (OSError, ValueError, AttributeError):
        return None


def _pstats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    a = np.asarray(values, dtype=float)
    return {
        "count": int(a.size),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


@dataclass
class BenchReport:
    """Result of :func:`run_bench`."""

    video: str
    channels: int
    seconds: float
    speed: float
    cpu_count: int
    wall_s: float  # run() start to all channels finished (includes splitting and draining)
    steady_wall_s: float  # every channel playing, until the run ended (excludes the up-front split)
    cpu_s: float  # process + children over wall_s
    steady_cpu_s: float  # process + children over steady_wall_s
    cpu_children_s: float
    peak_rss_mb: Optional[float]
    peak_child_rss_mb: Optional[float]
    rss_end_mb: Optional[float]
    media_s: float  # seconds of stream ingested, all channels
    windows: int
    clips: int
    scored: int
    dropped: int
    errors: int
    failed: List[str]
    latency: Dict[str, Dict[str, Optional[float]]]  # pooled over channels
    behind: Dict[str, Optional[float]]  # how late each segment finished vs the broadcast schedule
    segment_time_s: float = 2.0
    out_dir: Optional[str] = None
    per_channel: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def cores_used(self) -> float:
        return self.steady_cpu_s / self.steady_wall_s if self.steady_wall_s > 0 else float("nan")

    @property
    def cpu_percent(self) -> float:
        """Steady-state CPU as a percentage of the whole machine."""
        return 100.0 * self.cores_used / max(1, self.cpu_count)

    @property
    def clips_per_minute(self) -> float:
        """Clips per minute of stream, all channels together."""
        return self.clips / (self.media_s / 60.0) if self.media_s > 0 else 0.0

    @property
    def keeping_up(self) -> bool:
        """Every segment was finished within one segment time of its broadcast schedule."""
        mx = self.behind.get("max")
        return not self.failed and mx is not None and mx <= self.segment_time_s

    def to_dict(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d.update(
            cores_used=self.cores_used,
            cpu_percent=self.cpu_percent,
            clips_per_minute=self.clips_per_minute,
            keeping_up=self.keeping_up,
        )
        return d


def _behind_schedule(session_dir: Path, t0: Optional[float], speed: float) -> List[float]:
    """Per segment: when ingest finished with it, minus when the broadcast would have finished it.

    ``ingest_lag`` cannot show a backlog: a pull-based source stamps
    ``arrived_at`` when the ingest loop asks for the segment, so a loop that
    falls behind sees every segment "arrive" just as it is ready for it. This
    measures against the source's own schedule (``t0 + end / speed``) instead.
    """
    if t0 is None:
        return []
    out: List[float] = []
    for e in read_events(session_dir, "segment"):
        end = float(e["start"]) + float(e["duration"])
        if e.get("t_wall") is not None:
            out.append(float(e["t_wall"]) - (t0 + end / speed))
    return out


def run_bench(
    video: PathLike,
    channels: int,
    seconds: float,
    config: Optional[RealtimeConfig] = None,
    *,
    out_dir: Optional[PathLike] = None,
    speed: float = 1.0,
    session: str = "bench",
    transcode: bool = False,
) -> BenchReport:
    """Follow ``channels`` simulated channels of ``video`` for ``seconds`` and measure.

    Every channel replays the same file at ``speed`` (1.0 = real time) with
    its own buffer, detector and log, sharing one worker pool, as in
    production. After ``seconds`` the channels are stopped
    (``Supervisor.run(timeout=seconds)``; pending clips are still cut). With ``out_dir=None`` the output goes to a temporary
    directory that is removed afterwards.

    CPU is ``os.times()`` for this process plus its finished children
    (FFmpeg); the steady-state figure starts once every channel is playing,
    so the up-front file split is excluded, and runs to the end of the drain.
    """
    if channels < 1:
        raise ValueError("channels must be >= 1")
    if not seconds > 0:
        raise ValueError("seconds must be > 0")
    if not speed > 0:
        raise ValueError("speed must be > 0")
    cfg = (config if config is not None else RealtimeConfig()).validate()
    video = Path(video)
    tmp: Optional[str] = None
    if out_dir is None:
        tmp = tempfile.mkdtemp(prefix="vidflow-bench-")
        out_dir = tmp
    names = [f"ch{i:02d}" for i in range(channels)]
    sources: Dict[str, ReplaySource] = {}
    lock = threading.Lock()

    def factory(name: str):
        def make(work_dir: Path) -> ReplaySource:
            src = ReplaySource(video, name, work_dir, speed=speed, segment_time=cfg.segment_time_s, transcode=transcode)
            with lock:
                sources[name] = src
            return src

        return make

    try:
        sup = Supervisor(cfg, out_dir, session=session)
        for name in names:
            sup.add_channel(factory(name), channel=name)

        # Steady state: from the moment every source has started playing until stop.
        marks: Dict[str, Tuple[float, float]] = {}
        stop_watch = threading.Event()

        def watch() -> None:
            while not stop_watch.wait(0.05):
                with lock:
                    started = len(sources) == channels and all(s.t0 is not None for s in sources.values())
                if started and "steady" not in marks:
                    marks["steady"] = (time.monotonic(), sum(_cpu_times()))
                    return

        watcher = threading.Thread(target=watch, name="bench-watch", daemon=True)
        watcher.start()
        wall0 = time.monotonic()
        cpu0, kids0 = _cpu_times()
        results: Dict[str, ChannelResult] = sup.run(timeout=seconds)
        stop_watch.set()
        wall1 = time.monotonic()
        cpu1, kids1 = _cpu_times()
        # Up to the end of the run, so cuts of windows closed just before the stop count.
        steady = marks.get("steady")
        if steady is not None and wall1 > steady[0]:
            steady_wall = wall1 - steady[0]
            steady_cpu = (cpu1 + kids1) - steady[1]
        else:
            steady_wall = wall1 - wall0
            steady_cpu = (cpu1 + kids1) - (cpu0 + kids0)
        peak, peak_child = _peak_rss_mb()
        rss_end = _rss_mb()
        rss_end = None if rss_end is None else round(rss_end, 1)

        clip_lat: Dict[str, List[float]] = {"detect_latency": [], "cut_latency": [], "total_latency": [], "ingest_lag": []}
        behind: List[float] = []
        per_channel: List[Dict[str, Any]] = []
        media_s = 0.0
        for name in names:
            r = results[name]
            tracker = LatencyTracker.from_log(r.session_dir)
            for c in tracker.per_clip():
                if c.detect_latency is not None:
                    clip_lat["detect_latency"].append(c.detect_latency)
                if c.cut_latency is not None:
                    clip_lat["cut_latency"].append(c.cut_latency)
                if c.total_latency is not None:
                    clip_lat["total_latency"].append(c.total_latency)
            clip_lat["ingest_lag"].extend(s.ingest_lag for s in tracker.per_segment() if s.ingest_lag is not None)
            src = sources.get(name)
            b = _behind_schedule(r.session_dir, src.t0 if src is not None else None, speed)
            behind.extend(b)
            m = sum(float(e["duration"]) for e in read_events(r.session_dir, "segment"))
            media_s += m
            per_channel.append(
                {
                    "channel": name,
                    "ok": r.ok,
                    "error": r.error,
                    "segments": r.segments,
                    "media_s": round(m, 3),
                    "windows": r.windows,
                    "clips": r.clips,
                    "scored": r.scored,
                    "dropped": r.dropped,
                    "errors": r.errors,
                    "behind_max_s": max(b) if b else None,
                }
            )
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)

    return BenchReport(
        video=str(video),
        channels=channels,
        seconds=float(seconds),
        speed=float(speed),
        cpu_count=os.cpu_count() or 1,
        wall_s=round(wall1 - wall0, 3),
        steady_wall_s=round(steady_wall, 3),
        cpu_s=round((cpu1 + kids1) - (cpu0 + kids0), 3),
        steady_cpu_s=round(steady_cpu, 3),
        cpu_children_s=round(kids1 - kids0, 3),
        peak_rss_mb=peak,
        peak_child_rss_mb=peak_child,
        rss_end_mb=rss_end,
        media_s=round(media_s / channels, 3),
        windows=sum(r.windows for r in results.values()),
        clips=sum(r.clips for r in results.values()),
        scored=sum(r.scored for r in results.values()),
        dropped=sum(r.dropped for r in results.values()),
        errors=sum(r.errors for r in results.values()),
        failed=[n for n, r in results.items() if not r.ok],
        latency={k: _pstats(v) for k, v in clip_lat.items()},
        behind=_pstats(behind),
        segment_time_s=cfg.segment_time_s,
        out_dir=None if tmp is not None else str(out_dir),
        per_channel=per_channel,
    )


def format_bench(report: BenchReport) -> List[str]:
    r = report
    lines = [
        f"bench: {r.channels} channel(s) of {Path(r.video).name} at {r.speed:g}x for {r.seconds:g} s "
        f"on {r.cpu_count} cores",
        f"  wall {r.wall_s:.1f} s total, {r.steady_wall_s:.1f} s with every channel playing; "
        f"{r.media_s:.1f} s of stream per channel",
        f"  cpu  {r.cores_used:.2f} cores steady ({r.cpu_percent:.1f}% of machine, "
        f"{r.cores_used / r.channels * 100:.1f}% of a core per channel); "
        f"{r.cpu_s:.1f} s total incl. {r.cpu_children_s:.1f} s in ffmpeg",
        f"  mem  peak {_f(r.peak_rss_mb, '.0f')} MB (largest child {_f(r.peak_child_rss_mb, '.0f')} MB)",
        f"  out  {r.windows} windows, {r.clips} clips, {r.scored} scored, {r.dropped} dropped, {r.errors} errors; "
        f"{r.clips_per_minute:.1f} clips per minute ({r.clips_per_minute / r.channels:.2f} per channel)",
        f"    {'metric':<16}{'count':>6}{'p50 s':>9}{'p95 s':>9}{'max s':>9}",
    ]
    rows = dict(r.latency)
    rows["behind_schedule"] = r.behind
    for metric in ("detect_latency", "cut_latency", "total_latency", "ingest_lag", "behind_schedule"):
        s = rows.get(metric) or {}
        lines.append(
            f"    {metric:<16}{int(s.get('count') or 0):>6}"
            f"{_f(s.get('p50'), '.3f'):>9}{_f(s.get('p95'), '.3f'):>9}{_f(s.get('max'), '.3f'):>9}"
        )
    if r.failed:
        lines.append(f"  FAILED channels: {', '.join(r.failed)}")
    lines.append(
        f"  keeping up: {'yes' if r.keeping_up else 'NO'} "
        f"(every segment done within {r.segment_time_s:g} s of its broadcast schedule)"
    )
    return lines
