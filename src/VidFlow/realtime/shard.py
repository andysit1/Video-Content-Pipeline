"""Multi-process sharding of channels (design section 6.9).

One :class:`~VidFlow.realtime.supervisor.Supervisor` is capped by the
interpreter lock at about 2 cores, which on 1080p60 replays is about 32
channels. :class:`ShardedSupervisor` splits the channels into groups of about
``config.channels_per_process`` (default 20) and runs each group in its own
worker process (``multiprocessing``, spawn context) under an ordinary threaded
``Supervisor`` with its own worker pool.

The parent process:

- distributes channels in balanced contiguous groups (:func:`plan_shards`);
- forwards ``on_clip`` / ``on_score`` callbacks: the workers send each clip
  over a queue and the parent calls the callbacks in its own thread;
- collects each channel's :class:`~VidFlow.realtime.pipeline.ChannelResult`;
- propagates :meth:`stop`, a ``run(timeout=...)`` and Ctrl-C to every worker
  through a shared event (workers ignore SIGINT and sit in their own process
  group, so the terminal's Ctrl-C reaches only the parent, which then shuts
  the workers down gracefully; a second Ctrl-C kills them);
- notices a worker that died (crash, OOM kill): its channels are reported as
  failed, its process group (FFmpeg children included) is killed, ``.partial``
  clip files and buffered segments in its channels' session directories are
  removed, and an ``error`` plus ``session_end`` event are appended to their
  logs. The other workers keep running.
- never leaves workers behind: on the way out, any worker still alive after
  the grace period is killed with its process group. Workers watch their
  parent and shut down gracefully if it disappears.

Sources are built inside the workers, so a channel is given as a picklable
*factory* (a :data:`~VidFlow.realtime.pipeline.SourceFactory` defined at module
level, a ``functools.partial``, or a :class:`ReplayFactory`). The same holds
for pipeline keyword arguments other than ``on_clip``/``on_score``.

:func:`make_supervisor` picks a plain ``Supervisor`` when one process is
enough (so few channels behave exactly as before) and a ``ShardedSupervisor``
otherwise.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import multiprocessing
import os
import queue as queue_mod
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Union

from .config import RealtimeConfig
from .pipeline import ChannelResult, default_session_name
from .session import SessionLog
from .sources.replay import ReplaySource
from .supervisor import Supervisor, is_playing
from .types import Clip, Segment, Source

__all__ = [
    "plan_shards",
    "process_count",
    "ShardedSupervisor",
    "make_supervisor",
    "ChannelSpec",
    "ReplayFactory",
    "PacedReplay",
    "OffsetReplaySource",
    "process_tree_cpu",
]

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


# =========================================================================== planning


def process_count(
    channels: int, *, processes: Optional[int] = None, channels_per_process: int = 20
) -> int:
    """How many worker processes ``channels`` channels get.

    ``processes`` (> 0) forces the count; otherwise
    ``ceil(channels / channels_per_process)``. Never more than ``channels``,
    never less than 1.
    """
    if channels < 1:
        return 1
    if processes is not None and processes > 0:
        n = int(processes)
    else:
        if channels_per_process < 1:
            raise ValueError("channels_per_process must be >= 1")
        n = math.ceil(channels / channels_per_process)
    return max(1, min(n, channels))


def plan_shards(
    channels: int, *, processes: Optional[int] = None, channels_per_process: int = 20
) -> List[List[int]]:
    """Split channel indices ``0 .. channels-1`` into balanced contiguous groups.

    Group sizes differ by at most one, larger groups first.
    """
    if channels < 1:
        return []
    n = process_count(channels, processes=processes, channels_per_process=channels_per_process)
    base, extra = divmod(channels, n)
    out: List[List[int]] = []
    i = 0
    for k in range(n):
        size = base + (1 if k < extra else 0)
        out.append(list(range(i, i + size)))
        i += size
    return out


# =========================================================================== sources


class OffsetReplaySource(ReplaySource):
    """A :class:`ReplaySource` that starts ``offset_s`` into the file.

    Playback starts at the first segment beginning at or after ``offset_s``;
    the segments before it are deleted after the split. Media time, pacing
    and ``seq`` work as for ``ReplaySource`` (media time restarts at 0,
    ``seq`` keeps the file's segment index). An offset past the end yields
    nothing.
    """

    def __init__(self, *args: Any, offset_s: float = 0.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not offset_s >= 0:
            raise ValueError(f"offset_s must be >= 0, got {offset_s!r}")
        self.offset_s = float(offset_s)
        self._shifted: Optional[List[Any]] = None

    def prepare(self):  # type: ignore[override]
        infos = super().prepare()
        with self._lock:
            if self._shifted is None:
                k = next((i for i, s in enumerate(infos) if s.start >= self.offset_s - 1e-6), len(infos))
                for info in infos[:k]:
                    info.path.unlink(missing_ok=True)
                base = infos[k].start if k < len(infos) else 0.0
                self._shifted = [dataclasses.replace(s, start=s.start - base) for s in infos[k:]]
            return list(self._shifted)


class PacedReplay:
    """A source wrapper for benchmarks: start late and stop after a media budget.

    Waits ``start_delay_s`` (interruptible by :meth:`close`) before starting
    ``inner``, then yields its segments up to and including the first one
    that reaches ``media_limit_s`` (``None``: no limit), and ends without
    waiting for the next release. Exposes the inner source's
    ``t0`` and ``speed`` so the pipeline's schedule stays exact.
    """

    def __init__(self, inner: Source, *, start_delay_s: float = 0.0, media_limit_s: Optional[float] = None) -> None:
        self.inner = inner
        self.channel = inner.channel
        self.start_delay_s = max(0.0, float(start_delay_s))
        self.media_limit_s = None if media_limit_s is None else float(media_limit_s)
        self._closed = threading.Event()

    @property
    def t0(self) -> Optional[float]:
        return getattr(self.inner, "t0", None)

    @property
    def speed(self) -> Optional[float]:
        return getattr(self.inner, "speed", 1.0)

    def close(self) -> None:
        self._closed.set()
        self.inner.close()

    def __iter__(self) -> Iterator[Segment]:
        return self._run()

    def _run(self) -> Iterator[Segment]:
        if self.start_delay_s > 0 and self._closed.wait(self.start_delay_s):
            # Never started: let the inner source clean up after itself.
            it = iter(self.inner)
            close = getattr(it, "close", None)
            if callable(close):
                close()
            return
        it = iter(self.inner)
        limit = self.media_limit_s
        try:
            for seg in it:
                if limit is not None and seg.start >= limit - 1e-6:
                    # Not handed out: the file is ours to delete.
                    Path(seg.path).unlink(missing_ok=True)
                    break
                yield seg
                if limit is not None and seg.end >= limit - 1e-6:
                    break  # budget reached: do not wait for the next release
        finally:
            self.inner.close()
            close = getattr(it, "close", None)
            if callable(close):
                close()


@dataclass(frozen=True)
class ReplayFactory:
    """Picklable :data:`SourceFactory` for a (possibly staggered) ``ReplaySource``.

    ``offset_s``       start this far into the file (:class:`OffsetReplaySource`).
    ``start_delay_s``  wait this long before starting (:class:`PacedReplay`).
    ``media_limit_s``  stop after this much media (:class:`PacedReplay`).
    """

    video: Path
    channel: str
    speed: Optional[float] = 1.0
    segment_time: float = 2.0
    transcode: bool = False
    offset_s: float = 0.0
    start_delay_s: float = 0.0
    media_limit_s: Optional[float] = None

    def __call__(self, work_dir: Path) -> Source:
        kw = dict(speed=self.speed, segment_time=self.segment_time, transcode=self.transcode)
        if self.offset_s > 0:
            src: Source = OffsetReplaySource(self.video, self.channel, work_dir, offset_s=self.offset_s, **kw)
        else:
            src = ReplaySource(self.video, self.channel, work_dir, **kw)
        if self.start_delay_s > 0 or self.media_limit_s is not None:
            src = PacedReplay(src, start_delay_s=self.start_delay_s, media_limit_s=self.media_limit_s)
        return src


@dataclass(frozen=True)
class ChannelSpec:
    """One channel for a worker process: its name, picklable source factory and settings."""

    channel: str
    source: Callable[[Path], Source]
    config: Optional[RealtimeConfig] = None


# =========================================================================== CPU accounting


def _proc_cpu(pid: int) -> float:
    """CPU seconds of a live process plus its reaped children, from /proc (0 if unknown)."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read().decode("ascii", "replace")
    except OSError:
        return 0.0
    fields = data[data.rindex(")") + 2:].split()
    # Fields 14-17 of stat (utime, stime, cutime, cstime) are 11-14 after the comm.
    try:
        ticks = sum(int(fields[i]) for i in (11, 12, 13, 14))
        return ticks / os.sysconf("SC_CLK_TCK")
    except (ValueError, IndexError, OSError, AttributeError):
        return 0.0


def process_tree_cpu(pids: Sequence[int] = ()) -> float:
    """CPU seconds of this process, its reaped children, and the live processes ``pids``.

    Each live process counts itself plus *its* reaped children. Used to take
    a CPU reading in the middle of a sharded run, when the worker processes
    have not been reaped yet (``os.times`` only counts reaped children). Live
    readings need ``/proc`` (Linux); elsewhere live workers count 0.
    """
    t = os.times()
    total = t.user + t.system + t.children_user + t.children_system
    return total + sum(_proc_cpu(p) for p in pids)


# =========================================================================== worker process


def _worker_main(
    index: int,
    specs: List[ChannelSpec],
    config: RealtimeConfig,
    out_dir: str,
    session: str,
    kwargs: Dict[str, Any],
    forward_clip: bool,
    forward_score: bool,
    events: Any,
    stop_evt: Any,
    log_level: Optional[int],
) -> None:
    """Entry point of one worker process: a threaded Supervisor for ``specs``."""
    try:
        os.setpgid(0, 0)  # own process group: Ctrl-C goes to the parent only; killpg reaches FFmpeg
    except (AttributeError, OSError):
        pass
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):  # pragma: no cover
        pass
    if log_level is not None:
        logging.basicConfig(
            level=log_level,
            format=f"%(asctime)s %(levelname)s [shard {index}] %(name)s: %(message)s",
            stream=sys.stderr,
        )
    parent = os.getppid()

    def send(msg: Any) -> None:
        try:
            events.put(msg)
        except Exception:  # noqa: BLE001 - the parent is gone or the queue broke
            log.exception("shard %d: could not send %s to the parent", index, msg[0])

    try:
        opts = dict(kwargs)
        if forward_clip:
            opts["on_clip"] = lambda clip: send(("clip", index, clip))
        if forward_score:
            opts["on_score"] = lambda clip, extra: send(("score", index, clip, dict(extra)))
        sup = Supervisor(config, out_dir, session=session, **opts)
        failed: Dict[str, ChannelResult] = {}
        for spec in specs:
            try:
                sup.add_channel(spec.source, channel=spec.channel, config=spec.config)
            except Exception as exc:  # noqa: BLE001 - one bad channel must not sink the shard
                log.exception("shard %d: could not add channel %s", index, spec.channel)
                failed[spec.channel] = ChannelResult(
                    channel=spec.channel,
                    session=session,
                    session_dir=Path(out_dir) / spec.channel / session,
                    error=f"{type(exc).__name__}: {exc}",
                )
                send(("finished", index, spec.channel))
        done = threading.Event()

        def watch() -> None:
            playing: Set[str] = set()
            finished: Set[str] = set()
            stopped = False
            while not done.wait(0.05):
                if not stopped and (stop_evt.is_set() or os.getppid() != parent):
                    stopped = True
                    sup.stop()
                for name in sup.playing():
                    if name not in playing:
                        playing.add(name)
                        send(("playing", index, name))
                for name in sup.finished():
                    if name not in finished:
                        finished.add(name)
                        send(("finished", index, name))

        watcher = threading.Thread(target=watch, name=f"shard-{index}-watch", daemon=True)
        watcher.start()
        if stop_evt.is_set():
            sup.stop()
        results = sup.run()
        done.set()
        watcher.join(1.0)
        results.update(failed)
        send(("results", index, results))
    except BaseException as exc:  # noqa: BLE001
        send(("crash", index, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
        raise


# =========================================================================== parent


def _cleanup_channel(session_dir: Path, reason: str) -> None:
    """Remove what a dead worker left half-done in one channel's session directory."""
    for f in (session_dir / "clips").glob("*.partial"):
        f.unlink(missing_ok=True)
    buf = session_dir / "buffer"
    if buf.is_dir():
        for f in buf.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)
        try:
            buf.rmdir()
        except OSError:
            pass
    events = session_dir / "events.jsonl"
    try:
        if events.is_file():
            with open(events, "rb+") as fh:
                fh.seek(0, os.SEEK_END)
                if fh.tell() > 0:
                    fh.seek(-1, os.SEEK_END)
                    if fh.read(1) != b"\n":
                        fh.write(b"\n")  # end a torn line so the next event starts cleanly
        slog = SessionLog(session_dir)
        try:
            slog.log_error("process", reason)
            slog.log_session_end(error=reason, stopped=False)
        finally:
            slog.close()
    except OSError:
        log.exception("could not finish the log in %s", session_dir)


def _kill_group(pid: Optional[int]) -> None:
    if pid is None:
        return
    killpg = getattr(os, "killpg", None)
    if killpg is None:  # pragma: no cover - Windows
        return
    try:
        killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


class ShardedSupervisor:
    """Runs channels in several worker processes, each under its own :class:`Supervisor`.

    Same interface as ``Supervisor`` (``add_channel``, ``run``, ``stop``,
    ``results``, ``playing``, ``finished``, ``interrupted``, ``session``),
    except that channels must be added before :meth:`run` and given as
    picklable source factories (see the module docstring).

    ``processes``            number of worker processes; ``None``/0: one per
                             ``config.channels_per_process`` channels.
    ``stop_grace_s``         after a stop, how long workers get to drain before
                             being killed (default: ``drain_timeout_s`` + 60).
    ``log_level``            logging level configured in the workers (default:
                             this process's root level).
    """

    def __init__(
        self,
        config: Optional[RealtimeConfig] = None,
        out_dir: PathLike = ".",
        *,
        session: Optional[str] = None,
        processes: Optional[int] = None,
        stop_grace_s: Optional[float] = None,
        log_level: Optional[int] = None,
        start_method: str = "spawn",
        **pipeline_kwargs: Any,
    ) -> None:
        self.config = (config if config is not None else RealtimeConfig()).validate()
        self.out_dir = Path(out_dir)
        self.session = session or default_session_name()
        self.processes = processes
        self.on_clip: Optional[Callable[[Clip], None]] = pipeline_kwargs.pop("on_clip", None)
        self.on_score: Optional[Callable[[Clip, Dict[str, Any]], None]] = pipeline_kwargs.pop("on_score", None)
        self.pipeline_kwargs = pipeline_kwargs
        drain = float(pipeline_kwargs.get("drain_timeout_s", 120.0))
        self.stop_grace_s = float(stop_grace_s) if stop_grace_s is not None else drain + 60.0
        self.log_level = log_level if log_level is not None else logging.getLogger().getEffectiveLevel()
        self._ctx = multiprocessing.get_context(start_method)
        self._stop_evt = self._ctx.Event()
        self._lock = threading.Lock()
        self._specs: List[ChannelSpec] = []
        self._results: Dict[str, ChannelResult] = {}
        self._playing: Set[str] = set()
        self._finished: Set[str] = set()
        self._procs: List[Any] = []
        self._started = False
        self._stop_requested_at: Optional[float] = None
        self.interrupted = False
        self.shards: List[List[str]] = []

    # ----------------------------------------------------------------- channels

    @property
    def channels(self) -> List[str]:
        with self._lock:
            return [s.channel for s in self._specs]

    def add_channel(
        self,
        source: Callable[[Path], Source],
        *,
        channel: Optional[str] = None,
        config: Optional[RealtimeConfig] = None,
    ) -> ChannelSpec:
        """Add a channel (before :meth:`run`). ``source`` is a picklable factory."""
        name = channel if channel is not None else getattr(source, "channel", None)
        if name is None:
            raise ValueError("channel must be given")
        with self._lock:
            if self._started:
                raise RuntimeError("ShardedSupervisor: add channels before run()")
            if any(s.channel == name for s in self._specs):
                raise ValueError(f"channel {name!r} is already added")
            spec = ChannelSpec(name, source, config.validate() if config is not None else None)
            self._specs.append(spec)
        return spec

    @property
    def worker_pids(self) -> List[int]:
        """PIDs of the worker processes still running."""
        return [p.pid for p in self._procs if p.pid is not None and p.is_alive()]

    def cpu_seconds(self) -> float:
        """CPU so far of this process, reaped children and the live workers (see :func:`process_tree_cpu`)."""
        return process_tree_cpu(self.worker_pids)

    def playing(self) -> List[str]:
        with self._lock:
            return sorted(self._playing)

    def finished(self) -> List[str]:
        with self._lock:
            return sorted(self._finished)

    def results(self) -> Dict[str, ChannelResult]:
        with self._lock:
            return {s.channel: self._results[s.channel] for s in self._specs if s.channel in self._results}

    # ----------------------------------------------------------------- running

    def stop(self) -> None:
        """Ask every worker to shut down gracefully. Safe from any thread; returns at once."""
        with self._lock:
            if self._stop_requested_at is None:
                self._stop_requested_at = time.monotonic()
        self._stop_evt.set()

    def run(self, timeout: Optional[float] = None) -> Dict[str, ChannelResult]:
        """Start the workers and block until every channel has a result.

        ``timeout`` stops the channels when it expires (they still drain).
        Ctrl-C stops them gracefully; a second Ctrl-C kills the workers and is
        re-raised.
        """
        with self._lock:
            if self._started:
                raise RuntimeError("ShardedSupervisor.run() can only be called once")
            self._started = True
            specs = list(self._specs)
        if not specs:
            return {}
        groups = plan_shards(
            len(specs), processes=self.processes, channels_per_process=self.config.channels_per_process
        )
        self.shards = [[specs[i].channel for i in g] for g in groups]
        events = self._ctx.Queue()
        log.info("starting %d worker process(es) for %d channel(s)", len(groups), len(specs))
        try:
            for k, g in enumerate(groups):
                p = self._ctx.Process(
                    target=_worker_main,
                    args=(
                        k,
                        [specs[i] for i in g],
                        self.config,
                        str(self.out_dir),
                        self.session,
                        self.pipeline_kwargs,
                        self.on_clip is not None,
                        self.on_score is not None,
                        events,
                        self._stop_evt,
                        self.log_level,
                    ),
                    name=f"vidflow-shard-{k}",
                    daemon=True,
                )
                p.start()
                self._procs.append(p)
            self._wait(events, timeout)
        finally:
            self._reap_all()
            try:
                events.close()
                events.join_thread()
            except Exception:  # noqa: BLE001
                pass
        return self.results()

    # ---------------------------------------------------------------- private

    def _handle(self, msg: Any, have: Set[int]) -> None:
        kind, k = msg[0], msg[1]
        if kind == "clip" and self.on_clip is not None:
            self._callback(self.on_clip, msg[2])
        elif kind == "score" and self.on_score is not None:
            self._callback(self.on_score, msg[2], msg[3])
        elif kind == "playing":
            with self._lock:
                self._playing.add(msg[2])
        elif kind == "finished":
            with self._lock:
                self._finished.add(msg[2])
        elif kind == "results":
            with self._lock:
                self._results.update(msg[2])
                self._finished.update(msg[2])
            have.add(k)
        elif kind == "crash":
            log.error("worker %d crashed: %s", k, msg[2])

    @staticmethod
    def _callback(fn: Callable[..., None], *args: Any) -> None:
        try:
            fn(*args)
        except Exception:  # noqa: BLE001
            log.exception("callback failed")

    def _wait(self, events: Any, timeout: Optional[float]) -> None:
        n = len(self._procs)
        have: Set[int] = set()
        dead_since: Dict[int, float] = {}
        deadline = None if timeout is None else time.monotonic() + timeout
        while len(have) < n:
            try:
                try:
                    msg = events.get(timeout=0.1)
                except queue_mod.Empty:
                    msg = None
                while msg is not None:
                    self._handle(msg, have)
                    try:
                        msg = events.get_nowait()
                    except queue_mod.Empty:
                        msg = None
                now = time.monotonic()
                if deadline is not None and now >= deadline and not self._stop_evt.is_set():
                    log.info("timeout reached; stopping workers")
                    self.stop()
                for k, p in enumerate(self._procs):
                    if k in have or p.is_alive():
                        continue
                    # Exited without results (yet): give the queue a moment to deliver them.
                    first = dead_since.setdefault(k, now)
                    if now - first >= 1.0:
                        self._worker_died(k, p)
                        have.add(k)
                with self._lock:
                    asked = self._stop_requested_at
                if asked is not None and now - asked > self.stop_grace_s:
                    for k, p in enumerate(self._procs):
                        if k not in have:
                            log.error("worker %d did not finish %.0f s after stop; killing it", k, self.stop_grace_s)
                            p.kill()
                            p.join(5.0)
                            self._worker_died(k, p, killed=True)
                            have.add(k)
            except KeyboardInterrupt:
                if self.interrupted:
                    log.error("second interrupt; killing workers")
                    for p in self._procs:
                        p.kill()
                    raise
                log.warning("interrupted; shutting down workers")
                self.interrupted = True
                self.stop()

    def _worker_died(self, k: int, p: Any, killed: bool = False) -> None:
        _kill_group(p.pid)  # FFmpeg children of the dead worker
        how = "was killed after the stop grace period" if killed else f"exited with code {p.exitcode}"
        reason = f"worker process {p.pid} {how}"
        log.error("shard %d: %s; marking its channels failed", k, reason)
        for name in self.shards[k]:
            session_dir = self.out_dir / name / self.session
            with self._lock:
                if name in self._results:
                    continue
            _cleanup_channel(session_dir, reason)
            with self._lock:
                self._results[name] = ChannelResult(
                    channel=name, session=self.session, session_dir=session_dir, error=reason
                )
                self._finished.add(name)

    def _reap_all(self) -> None:
        """Make sure no worker (or its FFmpeg children) outlives :meth:`run`."""
        for p in self._procs:
            p.join(timeout=5.0)
        for k, p in enumerate(self._procs):
            if p.is_alive():
                log.error("worker %d still running at exit; killing it", k)
                p.kill()
                p.join(5.0)
            _kill_group(p.pid)


# =========================================================================== choice


def make_supervisor(
    config: Optional[RealtimeConfig] = None,
    out_dir: PathLike = ".",
    *,
    channels: int,
    processes: Optional[int] = None,
    session: Optional[str] = None,
    **kwargs: Any,
) -> Union[Supervisor, ShardedSupervisor]:
    """A ``Supervisor`` when one process suffices, else a ``ShardedSupervisor``.

    ``processes``: ``None``/0 decides from ``config.channels_per_process``;
    1 forces a single process (exactly the threaded ``Supervisor``); N > 1
    forces N workers. ``kwargs`` go to the supervisor (pipeline options).
    """
    cfg = (config if config is not None else RealtimeConfig()).validate()
    n = process_count(channels, processes=processes, channels_per_process=cfg.channels_per_process)
    if n <= 1:
        return Supervisor(cfg, out_dir, session=session, **kwargs)
    return ShardedSupervisor(cfg, out_dir, session=session, processes=n, **kwargs)
