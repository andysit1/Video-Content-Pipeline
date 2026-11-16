"""Supervisor: runs one :class:`ChannelPipeline` per channel (design section 6.9).

One process, one ingest thread per channel, one shared :class:`WorkerPool`.
The interface (``add_channel``, ``remove_channel``, ``run``, ``stop``) does not
expose threads. Past about 30 channels one process is capped by the
interpreter lock; :mod:`VidFlow.realtime.shard` then runs one ``Supervisor``
per group of channels in separate processes behind the same interface.

The pool's job-queue bound follows the channel count
(:func:`~VidFlow.realtime.pipeline.effective_queue_size`) when the supervisor
owns its pool.

A failing channel never takes down the supervisor: its pipeline records the
error in its result and session log, and the other channels carry on.
Ctrl-C (``KeyboardInterrupt`` in the thread running :meth:`run`) triggers the
shutdown sequence: close sources, flush detectors, drain the pool with a
timeout, write ``session_end``.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import RealtimeConfig
from .pipeline import (
    ChannelPipeline,
    ChannelResult,
    SourceFactory,
    WorkerPool,
    default_session_name,
    effective_queue_size,
)
from .types import Source

__all__ = ["Supervisor", "is_playing"]

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


def is_playing(pipe: ChannelPipeline) -> bool:
    """True once a channel's source has started playing.

    That is when it exposes a playback start (``t0``, e.g. ``ReplaySource``,
    set after the up-front split) or, for other sources, once the first
    segment has been ingested.
    """
    return getattr(pipe.source, "t0", None) is not None or pipe.result.segments > 0


class Supervisor:
    """Owns the channel list and a shared worker pool.

    ``config``   default settings for every channel (``add_channel`` may
                 override per channel); the pool is sized from it.
    ``out_dir``  root of the output tree (``<out_dir>/<channel>/<session>/``).
    ``session``  session name shared by all channels; default: local time.
    ``pipeline_kwargs``  passed to every :class:`ChannelPipeline` (e.g.
                 ``on_clip``, ``scorer``, ``detect_only``).
    """

    def __init__(
        self,
        config: Optional[RealtimeConfig] = None,
        out_dir: PathLike = ".",
        *,
        session: Optional[str] = None,
        pool: Optional[WorkerPool] = None,
        **pipeline_kwargs: Any,
    ) -> None:
        self.config = (config if config is not None else RealtimeConfig()).validate()
        self.out_dir = Path(out_dir)
        self.session = session or default_session_name()
        self._own_pool = pool is None
        self.pool = pool if pool is not None else WorkerPool(self.config.workers, self.config.queue_size)
        self.pipeline_kwargs = pipeline_kwargs
        self._lock = threading.Lock()
        self._pipelines: Dict[str, ChannelPipeline] = {}
        self._running = False
        self._stopping = threading.Event()
        self.interrupted = False

    # ----------------------------------------------------------------- channels

    @property
    def channels(self) -> List[str]:
        with self._lock:
            return list(self._pipelines)

    def pipeline(self, channel: str) -> ChannelPipeline:
        with self._lock:
            return self._pipelines[channel]

    def add_channel(
        self,
        source: Union[Source, SourceFactory],
        *,
        channel: Optional[str] = None,
        config: Optional[RealtimeConfig] = None,
        **kwargs: Any,
    ) -> ChannelPipeline:
        """Add a channel; it starts at once if :meth:`run` is already running.

        ``source`` is a :class:`Source` or a factory called with the channel's
        buffer directory (``channel`` is then required).
        """
        name = channel if channel is not None else getattr(source, "channel", None)
        if name is None:
            raise ValueError("channel must be given when source is a factory")
        with self._lock:
            if name in self._pipelines and not self._pipelines[name].done:
                raise ValueError(f"channel {name!r} is already running")
            if self._stopping.is_set():
                raise RuntimeError("supervisor is stopping")
            opts = {**self.pipeline_kwargs, **kwargs}
            pipe = ChannelPipeline(
                source,
                config if config is not None else self.config,
                self.out_dir,
                channel=name,
                session=self.session,
                pool=self.pool,
                **opts,
            )
            self._pipelines[name] = pipe
            self._resize_queue()
            if self._running:
                pipe.start()
        log.info("added channel %s", name)
        return pipe

    def remove_channel(self, channel: str, timeout: Optional[float] = None) -> Optional[ChannelResult]:
        """Stop one channel, wait for it to finish, and return its result."""
        with self._lock:
            pipe = self._pipelines.pop(channel)
            self._resize_queue()
        pipe.stop()
        if pipe._thread is None:  # never started: nothing to drain
            pipe.run()
        if not pipe.join(timeout):
            log.warning("channel %s did not finish within %s s", channel, timeout)
            return None
        return pipe.result

    # ----------------------------------------------------------------- running

    def run(self, timeout: Optional[float] = None) -> Dict[str, ChannelResult]:
        """Start every channel and block until all have finished.

        Returns each channel's result. ``timeout`` (seconds) stops the
        channels when it expires. Ctrl-C stops them gracefully; a second
        Ctrl-C while shutting down is re-raised.
        """
        with self._lock:
            self._running = True
            for pipe in self._pipelines.values():
                if pipe._thread is None and not pipe.done:
                    pipe.start()
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            while True:
                with self._lock:
                    pipes = list(self._pipelines.values())
                if all(p.done for p in pipes):
                    break
                if deadline is not None and time.monotonic() >= deadline and not self._stopping.is_set():
                    log.info("supervisor timeout reached; stopping channels")
                    self.stop()
                for p in pipes:
                    if not p.done:
                        p.join(0.2)
                        break
        except KeyboardInterrupt:
            log.warning("interrupted; shutting down channels")
            self.interrupted = True
            self.stop()
            with self._lock:
                pipes = list(self._pipelines.values())
            for p in pipes:
                while not p.join(0.2):
                    pass
        finally:
            with self._lock:
                self._running = False
        if self._own_pool:
            self.pool.shutdown(wait=True, timeout=10.0)
        return self.results()

    def stop(self) -> None:
        """Ask every channel to shut down. Safe from any thread; returns at once."""
        self._stopping.set()
        with self._lock:
            pipes = list(self._pipelines.values())
        for p in pipes:
            p.stop()

    def results(self) -> Dict[str, ChannelResult]:
        with self._lock:
            return {name: p.result for name, p in self._pipelines.items()}

    def playing(self) -> List[str]:
        """Channels whose source has started playing (see :func:`is_playing`)."""
        with self._lock:
            pipes = list(self._pipelines.items())
        return [name for name, p in pipes if is_playing(p)]

    def finished(self) -> List[str]:
        """Channels whose pipeline has finished."""
        with self._lock:
            pipes = list(self._pipelines.items())
        return [name for name, p in pipes if p.done]

    def _resize_queue(self) -> None:
        """Scale the queue bound with the channel count (own pool only). Hold ``_lock``."""
        if self._own_pool:
            self.pool.set_queue_size(effective_queue_size(self.config, max(1, len(self._pipelines))))
