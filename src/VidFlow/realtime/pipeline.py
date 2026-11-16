"""Per-channel pipeline: AudioDetector glue, WorkerPool, ChannelPipeline (design 6.7, 6.9).

One ingest thread per channel runs::

    for segment in source:
        buffer.add(segment)
        windows = detector.feed(segment)        # audio decode + segmenter
        pending += windows
        for window in pending covered through end + post_pad:
            pool.submit(cut job)                # takes its own lease
            pool.submit(score job)              # takes its own lease
    at end of stream (or stop): detector.flush(), submit everything pending,
    drain this channel's jobs, write session_end, close the buffer.

The ingest thread only does cheap work (audio decode of ~2 s, a few ms) and
never waits on the pool: when the bounded queue is full, a queued scoring job
is dropped first (the clip is still cut, with empty scores) and a ``dropped``
event is logged.

Score jobs wait for their own cut to finish before scoring. That cannot
deadlock: the queue is FIFO and only scoring jobs are ever evicted from it, so
a score job only starts after its cut job has been taken by a worker.

Stats choice. The segmenter keeps its default :class:`HistogramLoudnessStats`
rather than ``audio.RollingLoudnessStats``. The histogram reports the *lower
edge* of the bin holding the rank, so every sample in that bin satisfies
``db >= threshold``: a constant-level burst whose level dominates the history
(the fixture's second burst, judged against a history that holds the first)
is still loud against itself, with up to 1 dB of headroom. The rolling stats
interpolate to the bin mean, which leaves a constant burst on a knife edge
(threshold -4.95 dB against samples at -4.948/-4.949 dB on the fixture), so
codec noise of a hundredth of a dB decides whether a hop counts as loud.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .audio import LoudnessExtractor
from .buffer import SegmentBuffer
from .clip_writer import ClipWriter, sidecar_path, update_sidecar
from .config import RealtimeConfig
from .scoring.visual import VisualScorer
from .segmenter import LoudnessStats, OnlineSegmenter
from .session import ScheduleClock, SessionLog
from .types import Clip, LoudnessSample, Segment, Source, Window

__all__ = [
    "AudioDetector",
    "Job",
    "WorkerPool",
    "RankTracker",
    "ChannelPipeline",
    "ChannelResult",
    "SourceFactory",
    "default_session_name",
    "effective_queue_size",
    "RANK_WEIGHTS",
]

log = logging.getLogger(__name__)

PathLike = Union[str, Path]
#: Builds a source given the directory its segment files must be written to
#: (the buffer's directory, so the buffer can delete them).
SourceFactory = Callable[[Path], Source]

#: Weights of the z-values that make up ``rank_score`` (design 6.7).
RANK_WEIGHTS: Mapping[str, float] = {"audio_prominence": 1.0, "visual_mean": 1.0}


def default_session_name() -> str:
    """A session name from the local time, e.g. ``20261115-142501``."""
    return time.strftime("%Y%m%d-%H%M%S")


def effective_queue_size(config: RealtimeConfig, channels: int) -> int:
    """Job-queue bound of a pool shared by ``channels`` channels.

    ``max(queue_size, queue_per_channel x channels)``: ``queue_size`` is the
    floor (a single channel gets exactly ``queue_size``), and each channel adds
    room for ``queue_per_channel`` jobs, i.e. two windows' cut + score jobs at
    the default of 4, so a burst that closes a window on every channel at once
    does not drop scoring.
    """
    return max(int(config.queue_size), int(config.queue_per_channel) * max(1, int(channels)))


# =========================================================================== detector


class AudioDetector:
    """:class:`~VidFlow.realtime.types.Detector` from a loudness extractor and a segmenter.

    Discontinuity handling (design section 5): on a segment flagged
    ``discontinuity=True`` the segmenter is reset first (closing any open
    window, which is kept) and the extractor drops its carried-over samples
    (it does this itself on seeing the flag).

    ``last_samples`` holds the loudness samples of the latest segment, for
    the session log's ``audio`` event. One detector per channel; not
    thread-safe (the ingest thread only).
    """

    def __init__(
        self,
        channel: str,
        config: Optional[RealtimeConfig] = None,
        *,
        stats: Optional[LoudnessStats] = None,
        extractor: Optional[LoudnessExtractor] = None,
        segmenter: Optional[OnlineSegmenter] = None,
    ) -> None:
        self.channel = channel
        self.config = config if config is not None else RealtimeConfig()
        self.extractor = extractor if extractor is not None else LoudnessExtractor(hop_s=self.config.hop_s)
        self.segmenter = (
            segmenter if segmenter is not None else OnlineSegmenter(channel, self.config, stats)
        )
        self.last_samples: List[LoudnessSample] = []

    def detect(self, segment: Segment) -> Tuple[List[LoudnessSample], List[Window]]:
        """Feed one segment; return its loudness samples and the windows it closed."""
        windows: List[Window] = []
        if segment.discontinuity:
            windows.extend(self.segmenter.reset())
        samples = self.extractor.feed(segment)
        windows.extend(self.segmenter.feed(samples))
        self.last_samples = samples
        return samples, windows

    def feed(self, segment: Segment) -> List[Window]:
        return self.detect(segment)[1]

    def flush(self) -> List[Window]:
        return self.segmenter.flush()


# =========================================================================== pool


@dataclass
class Job:
    """One unit of work for the :class:`WorkerPool`.

    ``kind``     ``"cut"`` or ``"score"``; kinds in ``WorkerPool.droppable`` are
                 dropped first when the queue is full.
    ``fn``       the work; exceptions are caught and logged by the pool.
    ``owner``    anything identifying who submitted it (for :meth:`WorkerPool.cancel`).
    ``on_drop``  called with a reason if the job is dropped instead of run.
    """

    kind: str
    fn: Callable[[], Any]
    owner: Any = None
    on_drop: Optional[Callable[[str], None]] = None
    name: str = ""


class WorkerPool:
    """A fixed set of worker threads behind a bounded FIFO queue.

    :meth:`submit` never blocks. When the queue is full it makes room by
    dropping the oldest queued job of a droppable kind (``"score"`` by
    default); if there is none, the submitted job itself is dropped. Every
    dropped job's ``on_drop`` is called (outside the pool's lock).

    Threads are enough: FFmpeg runs as a subprocess and PyAV, OpenCV and NumPy
    release the interpreter lock. Shareable by several channel pipelines.
    """

    def __init__(
        self,
        workers: int = 4,
        queue_size: int = 64,
        *,
        droppable: Sequence[str] = ("score",),
        name: str = "vidflow-worker",
    ) -> None:
        if int(workers) != workers or workers < 1:
            raise ValueError(f"workers must be an integer >= 1, got {workers!r}")
        if int(queue_size) != queue_size or queue_size < 1:
            raise ValueError(f"queue_size must be an integer >= 1, got {queue_size!r}")
        self.workers = int(workers)
        self.queue_size = int(queue_size)
        self.droppable = tuple(droppable)
        self._cond = threading.Condition()
        self._queue: Deque[Job] = deque()
        self._active = 0
        self._shutdown = False
        self.submitted = 0
        self.completed = 0
        self.failed = 0
        self.dropped = 0
        self._threads = [
            threading.Thread(target=self._worker, name=f"{name}-{i}", daemon=True)
            for i in range(self.workers)
        ]
        for t in self._threads:
            t.start()

    # ---------------------------------------------------------------- public

    def set_queue_size(self, queue_size: int) -> None:
        """Change the queue bound. Shrinking drops nothing already queued; it only
        makes later submissions find the queue full sooner."""
        if int(queue_size) != queue_size or queue_size < 1:
            raise ValueError(f"queue_size must be an integer >= 1, got {queue_size!r}")
        with self._cond:
            self.queue_size = int(queue_size)

    @property
    def pending(self) -> int:
        """Jobs queued and not yet started."""
        with self._cond:
            return len(self._queue)

    @property
    def active(self) -> int:
        """Jobs currently running."""
        with self._cond:
            return self._active

    def submit(self, job: Job) -> bool:
        """Queue ``job`` without blocking; return False if it was dropped."""
        dropped: List[Tuple[Job, str]] = []
        accepted = False
        with self._cond:
            if self._shutdown:
                dropped.append((job, "pool shut down"))
            else:
                if len(self._queue) >= self.queue_size:
                    victim = next((j for j in self._queue if j.kind in self.droppable), None)
                    if victim is not None:
                        self._queue.remove(victim)
                        dropped.append((victim, "queue full"))
                if len(self._queue) < self.queue_size:
                    self._queue.append(job)
                    self.submitted += 1
                    accepted = True
                    self._cond.notify()
                else:
                    dropped.append((job, "queue full"))
            self.dropped += len(dropped)
        for j, reason in dropped:
            self._call_drop(j, reason)
        return accepted

    def cancel(self, owner: Any) -> int:
        """Drop every queued (not yet started) job of ``owner``; return how many."""
        with self._cond:
            victims = [j for j in self._queue if j.owner is owner]
            for j in victims:
                self._queue.remove(j)
            self.dropped += len(victims)
        for j in victims:
            self._call_drop(j, "cancelled")
        return len(victims)

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        """Block until no job is queued or running; False on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._queue or self._active:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Stop accepting jobs, drop the queued ones, and (optionally) join the workers."""
        with self._cond:
            self._shutdown = True
            victims = list(self._queue)
            self._queue.clear()
            self.dropped += len(victims)
            self._cond.notify_all()
        for j in victims:
            self._call_drop(j, "pool shut down")
        if wait:
            deadline = None if timeout is None else time.monotonic() + timeout
            for t in self._threads:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                t.join(remaining)

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    # --------------------------------------------------------------- private

    @staticmethod
    def _call_drop(job: Job, reason: str) -> None:
        log.info("dropped %s job %s: %s", job.kind, job.name, reason)
        if job.on_drop is None:
            return
        try:
            job.on_drop(reason)
        except Exception:  # noqa: BLE001 - a callback must not break the pool
            log.exception("on_drop callback of %s job %s failed", job.kind, job.name)

    def _worker(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._shutdown:
                    self._cond.wait()
                if not self._queue:
                    return  # shut down and nothing left
                job = self._queue.popleft()
                self._active += 1
            ok = True
            try:
                job.fn()
            except BaseException:  # noqa: BLE001 - a failing job never takes down a worker
                ok = False
                log.exception("%s job %s failed", job.kind, job.name)
            with self._cond:
                self._active -= 1
                if ok:
                    self.completed += 1
                else:
                    self.failed += 1
                self._cond.notify_all()


# =========================================================================== ranking


class RankTracker:
    """Running ``rank_score`` / ``rank_percentile`` over one session's clips (design 6.7).

    ``rank_score`` is the weighted sum of each score's z-value against the
    session's running mean and variance (Welford), the new clip included; a
    score with (numerically) zero variance so far contributes 0. ``rank_percentile`` is the
    percentage of the session's clips so far (this one included) whose
    ``rank_score`` is at or below this one's. Thread-safe.
    """

    def __init__(self, weights: Mapping[str, float] = RANK_WEIGHTS) -> None:
        self.weights = dict(weights)
        self._lock = threading.Lock()
        self._moments: Dict[str, List[float]] = {k: [0, 0.0, 0.0] for k in self.weights}  # n, mean, M2
        self._ranks: List[float] = []

    def rank(self, scores: Mapping[str, float]) -> Tuple[float, float]:
        with self._lock:
            total = 0.0
            for key, weight in self.weights.items():
                value = scores.get(key)
                if value is None or not math.isfinite(float(value)):
                    continue
                value = float(value)
                m = self._moments[key]
                m[0] += 1
                delta = value - m[1]
                m[1] += delta / m[0]
                m[2] += delta * (value - m[1])
                std = math.sqrt(m[2] / m[0]) if m[0] > 1 else 0.0
                # Treat float noise (identical scores computed twice) as no spread.
                if std > 1e-6 * max(1.0, abs(m[1])):
                    total += weight * (value - m[1]) / std
            self._ranks.append(total)
            at_or_below = sum(1 for r in self._ranks if r <= total + 1e-12)
            return total, 100.0 * at_or_below / len(self._ranks)


# =========================================================================== pipeline


@dataclass
class _Slot:
    """Hand-over between a window's cut job and its score job."""

    window: Window
    done: threading.Event = field(default_factory=threading.Event)
    clip: Optional[Clip] = None


class _Outstanding:
    """Count of this channel's jobs that are queued or running."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._n = 0

    def inc(self) -> None:
        with self._cond:
            self._n += 1

    def dec(self) -> None:
        with self._cond:
            self._n -= 1
            self._cond.notify_all()

    @property
    def count(self) -> int:
        with self._cond:
            return self._n

    def wait_zero(self, timeout: Optional[float]) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._n > 0:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(remaining)
            return True


@dataclass
class ChannelResult:
    """What a finished channel session produced."""

    channel: str
    session: str
    session_dir: Path
    segments: int = 0
    windows: int = 0
    clips: int = 0
    scored: int = 0
    dropped: int = 0
    errors: int = 0
    stopped: bool = False  # ended by stop() rather than end of stream
    error: Optional[str] = None  # set if the channel itself failed
    latency: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


class ChannelPipeline:
    """Follows one channel: ingest thread, buffer, detector, cut and score jobs.

    ``source``     a :class:`Source`, or a :data:`SourceFactory` called with the
                   buffer's directory (so a ``ReplaySource``/``HlsSource`` writes
                   its files where the buffer will delete them). With a
                   factory, ``channel`` must be given.
    ``out_dir``    clips go to ``<out_dir>/<channel>/<session>/clips/``, the
                   log to ``<out_dir>/<channel>/<session>/events.jsonl``, and
                   buffered segments to ``.../<session>/buffer/``.
    ``pool``       a shared :class:`WorkerPool`; by default the pipeline makes
                   its own (``config.workers``, ``config.queue_size``) and shuts
                   it down when done.
    ``detect_only`` detect and log windows but submit no cut/score jobs.
    ``on_clip`` / ``on_score``  callbacks (from worker threads) when a clip
                   lands / is scored.

    :meth:`run` blocks in the calling thread; :meth:`start` runs it in a new
    thread. :meth:`stop` (any thread) closes the source; the pipeline then
    flushes the detector, cuts what is pending, drains its jobs (up to
    ``drain_timeout_s``) and writes ``session_end``.
    """

    def __init__(
        self,
        source: Union[Source, SourceFactory],
        config: Optional[RealtimeConfig] = None,
        out_dir: PathLike = ".",
        *,
        channel: Optional[str] = None,
        session: Optional[str] = None,
        pool: Optional[WorkerPool] = None,
        detector: Optional[AudioDetector] = None,
        writer: Optional[ClipWriter] = None,
        scorer: Optional[VisualScorer] = None,
        detect_only: bool = False,
        drain_timeout_s: float = 120.0,
        on_clip: Optional[Callable[[Clip], None]] = None,
        on_score: Optional[Callable[[Clip, Mapping[str, Any]], None]] = None,
    ) -> None:
        self.config = (config if config is not None else RealtimeConfig()).validate()
        is_source = isinstance(source, Source)
        if channel is None:
            if not is_source:
                raise ValueError("channel must be given when source is a factory")
            channel = source.channel
        self.channel = channel
        self.session = session or default_session_name()
        self.out_dir = Path(out_dir)
        self.writer = writer if writer is not None else ClipWriter(
            self.out_dir, self.session, timeout_s=self.config.clip_timeout_s
        )
        self.session_dir = self.writer.clips_dir(channel).parent
        self.buffer_dir = self.session_dir / "buffer"
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.source: Source = source if is_source else source(self.buffer_dir)  # type: ignore[operator]
        if self.source.channel != channel:
            raise ValueError(f"source channel {self.source.channel!r} != pipeline channel {channel!r}")

        self.buffer = SegmentBuffer(self.config.retention_s)
        self.detector = detector if detector is not None else AudioDetector(channel, self.config)
        self.scorer = scorer if scorer is not None else VisualScorer.from_config(self.config)
        self.detect_only = detect_only
        self.drain_timeout_s = float(drain_timeout_s)
        self.on_clip = on_clip
        self.on_score = on_score

        self._own_pool = pool is None
        self.pool = pool if pool is not None else WorkerPool(self.config.workers, self.config.queue_size)
        self.log = SessionLog(self.session_dir)
        self.ranker = RankTracker()
        # Broadcast schedule for the ``behind_schedule`` metric; a source that
        # says it is unpaced (``speed`` None) has none. Live sources run at 1x.
        speed = getattr(self.source, "speed", 1.0)
        if speed is not None and not (isinstance(speed, (int, float)) and speed > 0):
            speed = 1.0
        self.schedule = ScheduleClock(speed)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._done = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._outstanding = _Outstanding()
        self._pending: List[Window] = []
        self.windows: List[Window] = []
        self.clips: Dict[str, Clip] = {}  # window_id -> latest Clip (with scores once scored)
        self.result = ChannelResult(channel=channel, session=self.session, session_dir=self.session_dir)

    # ---------------------------------------------------------------- control

    def start(self) -> "ChannelPipeline":
        """Run the ingest loop in a new thread and return at once."""
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("pipeline already started")
            self._thread = threading.Thread(target=self.run, name=f"ingest-{self.channel}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Ask the pipeline to shut down (safe from any thread, idempotent)."""
        self._stop.set()
        try:
            self.source.close()
        except Exception:  # noqa: BLE001
            log.exception("%s: closing the source failed", self.channel)

    def join(self, timeout: Optional[float] = None) -> bool:
        """Wait for the pipeline to finish; True if it has."""
        return self._done.wait(timeout)

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def sorted_clips(self) -> List[Clip]:
        with self._lock:
            return sorted(self.clips.values(), key=lambda c: c.window.start)

    # ---------------------------------------------------------------- ingest

    def run(self) -> ChannelResult:
        """Run the whole session in the calling thread and return its result."""
        res = self.result
        self.log.log_session_start(
            self.config,
            channel=self.channel,
            session=self.session,
            source=type(self.source).__name__,
            detect_only=self.detect_only,
        )
        iterator: Optional[Iterator[Segment]] = None
        try:
            iterator = iter(self.source)
            for segment in iterator:
                self._ingest(segment)
                if self._stop.is_set():
                    break
        except Exception as exc:  # noqa: BLE001 - a failing channel must not take down the supervisor
            log.exception("%s: ingest failed", self.channel)
            res.error = f"{type(exc).__name__}: {exc}"
            self.log.log_error("ingest", exc)
        finally:
            try:
                self._shutdown(iterator)
            except Exception as exc:  # noqa: BLE001
                log.exception("%s: shutdown failed", self.channel)
                if res.error is None:
                    res.error = f"{type(exc).__name__}: {exc}"
            finally:
                self._done.set()
        return res

    def _ingest(self, segment: Segment) -> None:
        self._bump("segments")
        try:
            self.buffer.add(segment)
        except ValueError as exc:
            # Out-of-order or foreign segment: skip it, keep the channel alive.
            self.log.log_error("buffer", exc, t_media=segment.start, seq=segment.seq)
            self._bump("errors")
            return
        try:
            samples, windows = self.detector.detect(segment)
        except Exception as exc:  # noqa: BLE001
            log.exception("%s: detector failed on seq %d", self.channel, segment.seq)
            self.log.log_error("detect", exc, t_media=segment.start, seq=segment.seq)
            self._bump("errors")
            samples, windows = [], []
        self.log.log_audio(segment, samples, hop_s=self.config.hop_s)
        for w in windows:
            self._window_closed(w)
        self._submit_covered(final=False)
        due = self.schedule.due_at(
            segment.end,
            segment.arrived_at,
            discontinuity=segment.discontinuity,
            t0=getattr(self.source, "t0", None),
        )
        self.log.log_segment(segment, due_at=due)

    def _window_closed(self, window: Window) -> None:
        self.log.log_window(window)
        self._bump("windows")
        with self._lock:
            self.windows.append(window)
        if not self.detect_only:
            self._pending.append(window)

    def _submit_covered(self, final: bool) -> None:
        post = self.config.post_pad_s
        keep: List[Window] = []
        for w in self._pending:
            if final or self.buffer.covers(w.end + post):
                self._submit(w)
            else:
                keep.append(w)
        self._pending = keep

    def _submit(self, window: Window) -> None:
        slot = _Slot(window)

        def cut_dropped(reason: str) -> None:
            self.log.log_dropped("clip", reason, window=window)
            self._bump("dropped")
            slot.done.set()
            self._outstanding.dec()

        def score_dropped(reason: str) -> None:
            self.log.log_dropped("score", reason, window=window)
            self._bump("dropped")
            self._outstanding.dec()

        self._outstanding.inc()
        cut_ok = self.pool.submit(
            Job("cut", lambda: self._cut_job(slot), owner=self, on_drop=cut_dropped, name=window.window_id)
        )
        if not cut_ok:
            return
        self._outstanding.inc()
        self.pool.submit(
            Job("score", lambda: self._score_job(slot), owner=self, on_drop=score_dropped, name=window.window_id)
        )

    # ---------------------------------------------------------------- jobs

    def _cut_job(self, slot: _Slot) -> None:
        w = slot.window
        cfg = self.config
        try:
            lease = self.buffer.lease(w.start, w.end, cfg.pre_pad_s, cfg.post_pad_s)
            if not lease:
                log.warning("%s: window %s is no longer buffered; not cut", self.channel, w.window_id)
                self.log.log_dropped("clip", "segments evicted", window=w)
                self._bump("dropped")
                return
            with lease:
                t0 = time.perf_counter()
                clip = self.writer.write(w, lease, cfg.pre_pad_s, cfg.post_pad_s)
                cut_s = time.perf_counter() - t0
            self.log.log_clip(clip, cut_s=round(cut_s, 4), truncated=lease.truncated)
            slot.clip = clip
            with self._lock:
                self.clips[w.window_id] = clip
                self.result.clips += 1
            if self.on_clip is not None:
                self._callback(self.on_clip, clip)
        except Exception as exc:  # noqa: BLE001 - logged; the channel carries on
            log.warning("%s: cut of %s failed: %s", self.channel, w.window_id, exc)
            self.log.log_error("clip", exc, window=w)
            self._bump("errors")
        finally:
            slot.done.set()
            self._outstanding.dec()

    def _score_job(self, slot: _Slot) -> None:
        w = slot.window
        try:
            # FIFO + only score jobs evicted => the cut is running or done.
            if not slot.done.wait(self.config.clip_timeout_s * 2 + 30.0):
                raise TimeoutError(f"cut of {w.window_id} did not finish")
            clip = slot.clip
            if clip is None:
                return  # cut failed or dropped; already logged
            lease = self.buffer.lease(w.start, w.end)
            with lease:
                if not lease:
                    self.log.log_dropped("score", "segments evicted", window=w)
                    self._bump("dropped")
                    return
                scored = self.scorer.score_clip(clip, lease)
            scores: Dict[str, float] = {**dict(scored.scores)}
            prominence = w.stats.get("prominence_db")
            if prominence is not None:
                scores["audio_prominence"] = float(prominence)
            rank_score, rank_pct = self.ranker.rank(scores)
            clip = dataclasses.replace(clip, scores=scores)
            extra = {
                "rank_score": rank_score,
                "rank_percentile": rank_pct,
                "rank_weights": dict(self.ranker.weights),
            }
            update_sidecar(clip, extra)
            self.log.log_score(w, scores, rank_score=rank_score, rank_percentile=rank_pct)
            with self._lock:
                self.clips[w.window_id] = clip
                self.result.scored += 1
            if self.on_score is not None:
                self._callback(self.on_score, clip, extra)
            mp = self.config.min_percentile
            if mp is not None and rank_pct < mp:
                self._cull(clip, rank_pct)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: scoring of %s failed: %s", self.channel, w.window_id, exc)
            self.log.log_error("score", exc, window=w)
            self._bump("errors")
        finally:
            self._outstanding.dec()

    def _cull(self, clip: Clip, rank_pct: float) -> None:
        for p in (clip.path, sidecar_path(clip.path)):
            try:
                Path(p).unlink(missing_ok=True)
            except OSError as exc:
                log.warning("could not remove culled clip file %s: %s", p, exc)
        with self._lock:
            self.clips.pop(clip.window.window_id, None)
        self.log.log_dropped(
            "clip", "below min_percentile", window=clip.window, rank_percentile=rank_pct
        )

    def _bump(self, name: str) -> None:
        with self._lock:
            setattr(self.result, name, getattr(self.result, name) + 1)

    def _callback(self, fn: Callable[..., None], *args: Any) -> None:
        try:
            fn(*args)
        except Exception:  # noqa: BLE001
            log.exception("%s: callback failed", self.channel)

    # ---------------------------------------------------------------- shutdown

    def _shutdown(self, iterator: Optional[Iterator[Segment]]) -> None:
        res = self.result
        res.stopped = self._stop.is_set()
        try:
            self.source.close()
        except Exception:  # noqa: BLE001
            log.exception("%s: closing the source failed", self.channel)
        # Finalise the generator now so the source deletes files it never handed out.
        close = getattr(iterator, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                log.exception("%s: closing the source iterator failed", self.channel)
        try:
            for w in self.detector.flush():
                self._window_closed(w)
        except Exception as exc:  # noqa: BLE001
            self.log.log_error("detect", exc)
            self._bump("errors")
        self._submit_covered(final=True)

        if not self._outstanding.wait_zero(self.drain_timeout_s):
            n = self.pool.cancel(self)
            log.warning(
                "%s: jobs still running after %.0f s; cancelled %d queued",
                self.channel, self.drain_timeout_s, n,
            )
            self._outstanding.wait_zero(5.0)
        if self._own_pool:
            self.pool.shutdown(wait=True, timeout=10.0)

        res.latency = self.log.latency.summary()
        self.log.log_session_end(
            segments=res.segments,
            windows=res.windows,
            clips=res.clips,
            scored=res.scored,
            stopped=res.stopped,
            error=res.error,
        )
        self.log.close()
        self.buffer.close()
        try:
            self.buffer_dir.rmdir()
        except OSError:
            leftovers = list(self.buffer_dir.glob("*"))
            if leftovers:
                log.warning("%s: %d file(s) left in %s", self.channel, len(leftovers), self.buffer_dir)
