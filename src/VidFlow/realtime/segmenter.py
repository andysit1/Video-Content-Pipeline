"""Online segmenter: loudness samples in, windows out (design section 6.4).

Pure logic with no file or media access. Every decision is a function of the
samples alone, processed one at a time, so the result never depends on how the
samples were batched across :meth:`OnlineSegmenter.feed` calls.

Rules
-----
- **Threshold.** A sample is loud if ``db >= threshold`` where
  ``threshold = max(stats.percentile(threshold_percentile), floor_db)`` is
  computed from the samples *before* it (the sample is added to the stats only
  after it has been classified).
- **Hysteresis.** While a window is open, samples are judged against the
  threshold that was in force when it opened (recorded as ``threshold_db``).
  Without this, a sustained loud moment raises the rolling percentile into its
  own level and cuts itself short.
- **Warm-up.** No sample is classified loud until ``warmup_s`` of audio has
  been fed (counted over the whole session; :meth:`reset` does not restart it).
- **Open/close.** The first loud sample opens a window. It closes once
  ``merge_gap_s`` of continuous quiet has passed after the end of the last loud
  sample; the window's end is the end of that last loud sample. The window is
  returned by the ``feed`` call containing the sample that completes the gap.
- **Filters.** Windows shorter than ``min_len_s`` are discarded. A window whose
  loud span reaches ``max_len_s`` is closed with ``forced=True`` and the next
  loud sample may open a new one immediately.
- :meth:`flush` (end of stream) and :meth:`reset` (discontinuity) close an open
  window with ``forced=True`` if it meets ``min_len_s``.

Window stats
------------
``peak_db``       highest sample in the window.
``mean_db``       arithmetic mean of the dB values of every sample in
                  ``[start, end)``, quiet samples inside the window included.
``threshold_db``  the threshold when the window opened.
``prominence_db`` ``peak_db`` minus the rolling median (``percentile(50)``) at
                  the moment the window closed.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional, Protocol, runtime_checkable

from .config import RealtimeConfig
from .types import LoudnessSample, Window, make_window_id

__all__ = ["LoudnessStats", "HistogramLoudnessStats", "OnlineSegmenter"]

log = logging.getLogger(__name__)

# Tolerance for comparisons of media-time durations built from float sums.
_EPS = 1e-6


@runtime_checkable
class LoudnessStats(Protocol):
    """What the segmenter needs from a rolling loudness-statistics object.

    Matches ``audio.RollingLoudnessStats(window_s, hop_s)`` (task T3).
    ``count`` may be an attribute/property or a zero-argument method.
    """

    count: int

    def update(self, db: float) -> None: ...

    def percentile(self, p: float) -> float: ...


class HistogramLoudnessStats:
    """Default rolling stats: a histogram of 1 dB bins over the trailing window.

    O(1) ``update``; ``percentile`` walks the (fixed, small) bin array. A sample
    ``db`` falls in bin ``floor(db)`` (clamped to ``[min_db, max_db)``; ``-inf``
    and NaN go in the lowest bin). ``percentile`` returns the *lower edge* of the
    bin holding the requested rank (``numpy.percentile``'s lower-index element),
    so every sample in that bin satisfies ``db >= percentile(p)``; the result is
    within 1 dB below ``numpy.percentile(..., method="lower")`` on the same
    window.

    Used when no stats object is injected into :class:`OnlineSegmenter`.
    """

    def __init__(
        self,
        window_s: float,
        hop_s: float,
        *,
        min_db: int = -150,
        max_db: int = 20,
    ) -> None:
        if window_s <= 0 or hop_s <= 0:
            raise ValueError("window_s and hop_s must be > 0")
        self.capacity = max(1, int(round(window_s / hop_s)))
        self.min_db = int(min_db)
        self.max_db = int(max_db)
        self._bins = [0] * (self.max_db - self.min_db)
        self._ring: Deque[int] = deque()

    @property
    def count(self) -> int:
        return len(self._ring)

    def _bin(self, db: float) -> int:
        if math.isnan(db) or db == -math.inf:
            return 0
        b = math.floor(db) - self.min_db
        return min(max(b, 0), len(self._bins) - 1)

    def update(self, db: float) -> None:
        b = self._bin(float(db))
        self._ring.append(b)
        self._bins[b] += 1
        if len(self._ring) > self.capacity:
            self._bins[self._ring.popleft()] -= 1

    def percentile(self, p: float) -> float:
        n = len(self._ring)
        if n == 0:
            raise ValueError("percentile of an empty window")
        p = min(max(float(p), 0.0), 100.0)
        k = int(math.floor(p / 100.0 * (n - 1) + 1e-9))  # 0-based rank, lower index
        cum = 0
        for i, c in enumerate(self._bins):
            cum += c
            if cum > k:
                return float(self.min_db + i)
        return float(self.max_db)  # unreachable


@dataclass
class _Open:
    """State of the window currently open."""

    start: float
    loud_end: float  # end of the last loud sample
    threshold_db: float
    peak_db: float
    db_sum: float  # over samples in [start, loud_end)
    db_n: int
    tail_sum: float = 0.0  # quiet samples after loud_end, folded in if the window extends
    tail_n: int = 0


class OnlineSegmenter:
    """Turns a stream of :class:`LoudnessSample` into closed :class:`Window`\\ s.

    ``stats`` is any object satisfying :class:`LoudnessStats` (for example T3's
    ``RollingLoudnessStats(config.stats_window_s, config.hop_s)``); by default a
    :class:`HistogramLoudnessStats` with the same parameters is used. The stats
    object is fed every sample, and is *not* cleared by :meth:`reset`: the
    loudness distribution stays valid across a discontinuity.
    Samples are assumed to be ``config.hop_s`` long and in time order.
    """

    def __init__(
        self,
        channel: str,
        config: Optional[RealtimeConfig] = None,
        stats: Optional[LoudnessStats] = None,
        *,
        detector: str = "audio",
    ) -> None:
        self.channel = channel
        self.config = config if config is not None else RealtimeConfig()
        self.detector = detector
        self.hop_s = float(self.config.hop_s)
        self.stats: LoudnessStats = (
            stats
            if stats is not None
            else HistogramLoudnessStats(self.config.stats_window_s, self.config.hop_s)
        )
        self._seen = 0  # samples fed this session (warm-up counter)
        self._open: Optional[_Open] = None

    # ------------------------------------------------------------------ queries

    @property
    def is_open(self) -> bool:
        """True while a (tentative) window is open."""
        return self._open is not None

    @property
    def samples_seen(self) -> int:
        return self._seen

    def _stats_count(self) -> int:
        c = self.stats.count
        return int(c() if callable(c) else c)

    def current_threshold(self) -> float:
        """Threshold the next sample would be judged against if no window were open."""
        if self._stats_count() == 0:
            return float(self.config.floor_db)
        return max(float(self.stats.percentile(self.config.threshold_percentile)), float(self.config.floor_db))

    def _warmed_up(self) -> bool:
        return self._seen * self.hop_s >= self.config.warmup_s - _EPS

    # ------------------------------------------------------------------ feeding

    def feed(self, samples: Iterable[LoudnessSample]) -> List[Window]:
        """Process ``samples`` in order; return the windows they closed."""
        out: List[Window] = []
        for s in samples:
            w = self._step(float(s.t), float(s.db))
            if w is not None:
                out.append(w)
        return out

    def flush(self) -> List[Window]:
        """End of stream: close any open window (``forced=True``)."""
        return self._force_close()

    def reset(self) -> List[Window]:
        """Discontinuity: close any open window (``forced=True``) and drop it from state.

        The rolling stats and warm-up progress are kept.
        """
        return self._force_close()

    # ------------------------------------------------------------------ internals

    def _step(self, t: float, db: float) -> Optional[Window]:
        cfg = self.config
        hop = self.hop_s
        result: Optional[Window] = None
        w = self._open

        if w is not None:
            loud = db >= w.threshold_db
            if loud:
                w.db_sum += w.tail_sum + db
                w.db_n += w.tail_n + 1
                w.tail_sum, w.tail_n = 0.0, 0
                w.loud_end = t + hop
                if db > w.peak_db:
                    w.peak_db = db
            else:
                w.tail_sum += db
                w.tail_n += 1
        else:
            threshold = self.current_threshold()
            loud = self._warmed_up() and db >= threshold
            if loud:
                self._open = w = _Open(
                    start=t, loud_end=t + hop, threshold_db=threshold,
                    peak_db=db, db_sum=db, db_n=1,
                )

        # The sample is part of the history only after it has been classified.
        self.stats.update(db)
        self._seen += 1

        if w is not None:
            if loud and w.loud_end - w.start >= cfg.max_len_s - _EPS:
                result = self._close(forced=True)
            elif not loud and (t + hop) - w.loud_end >= cfg.merge_gap_s - _EPS:
                result = self._close(forced=False)
        return result

    def _force_close(self) -> List[Window]:
        if self._open is None:
            return []
        w = self._close(forced=True)
        return [w] if w is not None else []

    def _close(self, forced: bool) -> Optional[Window]:
        w = self._open
        self._open = None
        assert w is not None
        length = w.loud_end - w.start
        if length < self.config.min_len_s - _EPS:
            log.debug(
                "%s: discarding %.2f s window at %.2f (min_len_s=%.2f)",
                self.channel, length, w.start, self.config.min_len_s,
            )
            return None
        median = float(self.stats.percentile(50.0)) if self._stats_count() else w.threshold_db
        stats = {
            "peak_db": float(w.peak_db),
            "mean_db": float(w.db_sum / w.db_n),
            "threshold_db": float(w.threshold_db),
            "prominence_db": float(w.peak_db - median),
        }
        return Window(
            channel=self.channel,
            window_id=make_window_id(self.channel, w.start),
            start=w.start,
            end=w.loud_end,
            detector=self.detector,
            stats=stats,
            forced=forced,
        )
