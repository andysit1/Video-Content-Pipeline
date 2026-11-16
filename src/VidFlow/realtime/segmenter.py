"""Online segmenter: loudness samples in, windows out (design section 6.4).

Pure logic with no file or media access. Every decision is a function of the
samples alone, processed one at a time, so the result never depends on how the
samples were batched across :meth:`OnlineSegmenter.feed` calls, and a session
log's ``audio`` events replay it exactly (``vidflow-live retune``).

Rules
-----
- **Threshold.** A sample is loud if ``db >= threshold`` where::

      threshold = max(background.percentile(threshold_percentile) + rise_db, floor_db)

  ``background`` is a weighted histogram of the trailing ``stats_window_s`` of
  samples *before* the one being judged. Samples that ended up inside a
  detected window count with weight ``window_weight`` (default 0.25) instead
  of 1, so detected moments barely raise the bar for the next one; a level
  that keeps being detected still becomes "normal" once it fills roughly
  ``(1 - p) / (window_weight * p + 1 - p)`` of the history (about 31 % at the
  defaults), so the rule cannot lock itself into treating a new normal as a
  highlight. ``rise_db`` makes "loud" mean "clearly above the loud end of what
  this stream usually sounds like", not "in its top 10 %".
- **Regime change.** A window that reaches ``max_len_s`` is evidence that
  the stream may have got louder rather than had a moment: its samples enter
  the background at full weight. If that moves the threshold above the one
  the window opened at, the window is *re-judged*: its span is run through
  the state machine again against the new threshold and whatever stands out
  from the new level is reported instead (often nothing, or the real peaks
  inside it). If the threshold does not move (a long moment against a long,
  quieter history), the window is reported ``forced=True`` as before. The
  re-judged span stays in the background at full weight.
- **Hysteresis.** While a window is open, samples are judged against the
  threshold that was in force when it opened (recorded as ``threshold_db``).
  A window's samples are added to the background only when it closes (it
  is not consulted meanwhile), with the weight its outcome decides.
- **Warm-up (hindsight).** The first ``warmup_s`` of the session are buffered,
  not skipped. When the buffer is full (or at :meth:`flush`), one threshold
  is chosen for it with hindsight, as the fixed point of the rule above over
  the whole buffer (start from ``median + rise_db``, detect, down-weight,
  recompute; at most :data:`WARMUP_ITERATIONS` rounds), and the buffered
  samples are run through the state machine against that threshold. Windows
  found in the warm-up are therefore returned late (by the call that
  completes the warm-up, up to ``warmup_s`` after they ended) but not lost;
  the buffer must still hold them then (``RealtimeConfig.validate`` checks
  ``retention_s``). The same ~31 % budget applies to the short warm-up: a
  level that fills about a third of it is taken as normal. :meth:`reset`
  during warm-up is replayed at the same point. Warm-up counts samples over
  the whole session; :meth:`reset` does not restart it.
- **Open/close.** The first loud sample opens a window. It closes once
  ``merge_gap_s`` of continuous quiet has passed after the end of the last loud
  sample; the window's end is the end of that last loud sample. The window is
  returned by the ``feed`` call containing the sample that completes the gap.
- **Filters.** Windows shorter than ``min_len_s`` are discarded (their samples
  count as background). A window whose loud span reaches ``max_len_s`` is
  closed with ``forced=True`` (unless re-judged, above) and the next loud
  sample may open a new one immediately.
- :meth:`flush` (end of stream) and :meth:`reset` (discontinuity) close an open
  window with ``forced=True`` if it meets ``min_len_s``.

Cost per sample is O(1): one histogram update and one walk of its fixed
number of bins; the warm-up adds a bounded one-off pass over ``warmup_s`` of
samples.

Window stats
------------
``peak_db``       highest sample in the window.
``mean_db``       arithmetic mean of the dB values of every sample in
                  ``[start, end)``, quiet samples inside the window included.
``threshold_db``  the threshold when the window opened.
``prominence_db`` ``peak_db`` minus the background median (``percentile(50)``)
                  at the moment the window closed, before the window's own
                  samples are added.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from .config import RealtimeConfig
from .types import LoudnessSample, Window, make_window_id

__all__ = ["LoudnessStats", "HistogramLoudnessStats", "OnlineSegmenter", "WARMUP_ITERATIONS", "WEIGHT_UNIT"]

log = logging.getLogger(__name__)

# Tolerance for comparisons of media-time durations built from float sums.
_EPS = 1e-6

#: Integer weight of one background sample; window samples weigh
#: ``round(window_weight * WEIGHT_UNIT)``. Integers keep the histogram exact.
WEIGHT_UNIT = 1000

#: Most rounds of the warm-up fixed-point search.
WARMUP_ITERATIONS = 5


@runtime_checkable
class LoudnessStats(Protocol):
    """What the segmenter needs from a rolling loudness-statistics object.

    ``count`` may be an attribute/property or a zero-argument method. An object
    that also has ``update_weighted(db, weight)`` (like
    :class:`HistogramLoudnessStats`) gets the window down-weighting; any other
    gets every sample through plain ``update``.
    """

    count: int

    def update(self, db: float) -> None: ...

    def percentile(self, p: float) -> float: ...


class HistogramLoudnessStats:
    """Default rolling stats: a weighted histogram of 1 dB bins over the trailing window.

    The window holds the last ``window_s / hop_s`` samples, whatever their
    weight. O(1) ``update``; ``percentile`` walks the (fixed, small) bin array.
    A sample ``db`` falls in bin ``floor(db)`` (clamped to ``[min_db, max_db)``;
    ``-inf`` and NaN go in the lowest bin). ``percentile`` returns the *lower
    edge* of the bin holding the requested weighted rank, so every sample in
    that bin satisfies ``db >= percentile(p)``. With every sample added by
    plain ``update`` (weight :data:`WEIGHT_UNIT`) this is
    ``numpy.percentile``'s lower-index element, to within 1 dB below.

    Weights are integers (``update`` uses :data:`WEIGHT_UNIT`), so the sums are
    exact and the result is deterministic.
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
        self._bins = [0] * (self.max_db - self.min_db)  # total weight per bin
        self._ring: Deque[Tuple[int, int]] = deque()  # (bin, weight)
        self._total = 0

    @property
    def count(self) -> int:
        return len(self._ring)

    def _bin(self, db: float) -> int:
        if math.isnan(db) or db == -math.inf:
            return 0
        b = math.floor(db) - self.min_db
        return min(max(b, 0), len(self._bins) - 1)

    def update(self, db: float) -> None:
        self.update_weighted(db, WEIGHT_UNIT)

    def update_weighted(self, db: float, weight: int) -> None:
        """Add one sample with integer ``weight`` (>= 0); the oldest sample ages out when full."""
        w = int(weight)
        if w < 0:
            raise ValueError("weight must be >= 0")
        b = self._bin(float(db))
        self._ring.append((b, w))
        self._bins[b] += w
        self._total += w
        if len(self._ring) > self.capacity:
            ob, ow = self._ring.popleft()
            self._bins[ob] -= ow
            self._total -= ow

    def percentile(self, p: float) -> float:
        n = len(self._ring)
        if n == 0:
            raise ValueError("percentile of an empty window")
        p = min(max(float(p), 0.0), 100.0)
        total = self._total
        if total <= 0:  # every sample has weight 0: fall back to equal weights
            bins = [0] * len(self._bins)
            for b, _ in self._ring:
                bins[b] += 1
            unit, total = 1, n
        else:
            bins, unit = self._bins, min(WEIGHT_UNIT, total)
        # Weighted generalisation of the lower-index rank floor(p/100 * (n - 1)):
        # the first bin whose cumulative weight exceeds p/100 * (total - unit).
        target = p / 100.0 * (total - unit) + 1e-9 * unit
        cum = 0
        for i, c in enumerate(bins):
            cum += c
            if cum > target:
                return float(self.min_db + i)
        return float(self.min_db + len(bins) - 1)  # unreachable


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
    # (t, db) of every sample since the window opened that is not yet in the background.
    pending: List[Tuple[float, float]] = field(default_factory=list)
    n_in: int = 0  # how many of ``pending`` lie in [start, loud_end)
    complete: bool = True  # ``pending`` starts at ``start`` (False if opened during a no-commit replay)


_Buffered = Optional[Tuple[float, float]]  # a (t, db) sample, or None for a reset()


class OnlineSegmenter:
    """Turns a stream of :class:`LoudnessSample` into closed :class:`Window`\\ s.

    ``stats`` is any object satisfying :class:`LoudnessStats`; by default a
    :class:`HistogramLoudnessStats(config.stats_window_s, config.hop_s)`. It
    holds the background history and is *not* cleared by :meth:`reset`: the
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
        self._weighted = callable(getattr(self.stats, "update_weighted", None))
        self._w_bg = WEIGHT_UNIT
        self._w_win = int(round(float(self.config.window_weight) * WEIGHT_UNIT))
        self._seen = 0  # samples fed this session (warm-up counter)
        self._open: Optional[_Open] = None
        # Warm-up: buffered samples and reset markers, until the hindsight pass.
        self._warm = not self.config.warmup_s > _EPS
        self._warmup: List[_Buffered] = []
        # Replay state: a fixed threshold, and whether replayed samples go to the background.
        self._fixed: Optional[float] = None
        self._commit = True
        self.warmup_threshold: Optional[float] = None  # the threshold the warm-up chose
        self.rejudged = 0  # max-length windows re-judged after the background moved

    # ------------------------------------------------------------------ queries

    @property
    def is_open(self) -> bool:
        """True while a (tentative) window is open (never during warm-up buffering)."""
        return self._open is not None

    @property
    def warmed_up(self) -> bool:
        """True once the warm-up buffer has been judged; detection is live from then on."""
        return self._warm

    @property
    def samples_seen(self) -> int:
        return self._seen

    def _stats_count(self) -> int:
        c = self.stats.count
        return int(c() if callable(c) else c)

    def current_threshold(self) -> float:
        """Threshold the next sample would be judged against if no window were open."""
        if self._fixed is not None:
            return self._fixed
        cfg = self.config
        if self._stats_count() == 0:
            return float(cfg.floor_db)
        base = float(self.stats.percentile(cfg.threshold_percentile))
        return max(base + float(cfg.rise_db), float(cfg.floor_db))

    # ------------------------------------------------------------------ feeding

    def feed(self, samples: Iterable[LoudnessSample]) -> List[Window]:
        """Process ``samples`` in order; return the windows they closed."""
        out: List[Window] = []
        for s in samples:
            t, db = float(s.t), float(s.db)
            self._seen += 1
            if self._warm:
                out.extend(self._step(t, db))
                continue
            self._warmup.append((t, db))
            if self._seen * self.hop_s >= self.config.warmup_s - _EPS:
                out.extend(self._finish_warmup())
        return out

    def flush(self) -> List[Window]:
        """End of stream: judge a pending warm-up, then close any open window (``forced=True``)."""
        out = [] if self._warm else self._finish_warmup()
        return out + self._force_close()

    def reset(self) -> List[Window]:
        """Discontinuity: close any open window (``forced=True``) and drop it from state.

        The rolling stats and warm-up progress are kept. During warm-up the
        reset is recorded and applied at the same point when the warm-up is
        judged.
        """
        if not self._warm:
            if self._warmup and self._warmup[-1] is not None:
                self._warmup.append(None)
            return []
        return self._force_close()

    # ------------------------------------------------------------------ hindsight

    def _replay(self, buf: Sequence[_Buffered], threshold: float, commit: bool) -> List[Window]:
        """Run ``buf`` through the state machine with windows opening at ``threshold``.

        ``commit=False`` when the samples are already in the background.
        Windows that reach ``max_len_s`` here are emitted as they are, not
        re-judged again. A window still open at the end stays open.
        """
        out: List[Window] = []
        saved = (self._fixed, self._commit)
        self._fixed, self._commit = threshold, commit
        try:
            for x in buf:
                out.extend(self._force_close() if x is None else self._step(x[0], x[1]))
        finally:
            self._fixed, self._commit = saved
        return out

    def _finish_warmup(self) -> List[Window]:
        buf = self._warmup
        self._warmup = []
        self._warm = True
        if not any(x is not None for x in buf):
            return []
        thr = self._hindsight_threshold(buf)
        self.warmup_threshold = thr
        log.debug("%s: warm-up of %d samples judged at %.1f dB", self.channel, len(buf), thr)
        return self._replay(buf, thr, commit=True)

    def _hindsight_threshold(self, buf: Sequence[_Buffered]) -> float:
        """Fixed point of the threshold rule over the warm-up buffer, judged as a whole.

        Starts low, at ``median + rise_db``, so a highlight that fills a large
        part of the warm-up is found and down-weighted rather than taken as
        the norm; then detect, weight, recompute, until the threshold repeats.
        """
        cfg = self.config
        dbs = sorted(x[1] for x in buf if x is not None)
        median = dbs[(len(dbs) - 1) // 2]
        thr = max(math.floor(median) + float(cfg.rise_db), float(cfg.floor_db))
        trial_cfg = cfg.replace(warmup_s=0.0)
        window_s = max(cfg.stats_window_s, len(buf) * cfg.hop_s)
        for _ in range(WARMUP_ITERATIONS):
            trial = OnlineSegmenter(self.channel, trial_cfg, HistogramLoudnessStats(window_s, cfg.hop_s))
            trial._replay(buf, thr, commit=True)
            trial._force_close()
            new = trial.current_threshold()
            if new == thr:
                break
            thr = new
        return thr

    # ------------------------------------------------------------------ internals

    def _add_background(self, db: float, weight: int) -> None:
        if self._weighted:
            self.stats.update_weighted(db, weight)  # type: ignore[attr-defined]
        else:
            self.stats.update(db)

    def _step(self, t: float, db: float) -> List[Window]:
        cfg = self.config
        hop = self.hop_s
        w = self._open

        if w is not None:
            loud = db >= w.threshold_db
            if self._commit:
                w.pending.append((t, db))
            if loud:
                w.db_sum += w.tail_sum + db
                w.db_n += w.tail_n + 1
                w.tail_sum, w.tail_n = 0.0, 0
                w.loud_end = t + hop
                w.n_in = len(w.pending)
                if db > w.peak_db:
                    w.peak_db = db
            else:
                w.tail_sum += db
                w.tail_n += 1
        else:
            threshold = self.current_threshold()
            loud = db >= threshold
            if loud:
                self._open = w = _Open(
                    start=t, loud_end=t + hop, threshold_db=threshold,
                    peak_db=db, db_sum=db, db_n=1,
                    pending=[(t, db)] if self._commit else [], n_in=1 if self._commit else 0,
                    complete=self._commit,
                )
            elif self._commit:
                # The sample is part of the history only after it has been classified.
                self._add_background(db, self._w_bg)

        if w is not None:
            if loud and w.loud_end - w.start >= cfg.max_len_s - _EPS:
                return self._close(forced=True, at_max_len=True)
            if not loud and (t + hop) - w.loud_end >= cfg.merge_gap_s - _EPS:
                return self._close(forced=False)
        return []

    def _force_close(self) -> List[Window]:
        if self._open is None:
            return []
        return self._close(forced=True)

    def _close(self, forced: bool, at_max_len: bool = False) -> List[Window]:
        w = self._open
        self._open = None
        assert w is not None
        length = w.loud_end - w.start
        keep = length >= self.config.min_len_s - _EPS
        median = (
            float(self.stats.percentile(50.0)) if keep and self._stats_count() else w.threshold_db
        )
        # Commit the window's samples to the background. A discarded window and
        # a window that ran to max_len (the stream got louder) count in full;
        # a kept window's own span is down-weighted, its quiet tail is not.
        in_weight = self._w_bg if (not keep or at_max_len) else self._w_win
        for i, (_, db) in enumerate(w.pending):
            self._add_background(db, in_weight if i < w.n_in else self._w_bg)
        if not keep:
            log.debug(
                "%s: discarding %.2f s window at %.2f (min_len_s=%.2f)",
                self.channel, length, w.start, self.config.min_len_s,
            )
            return []
        if at_max_len and self._fixed is None and w.complete:
            # Regime change? With the window's samples now in the background at
            # full weight, re-judge its span; if the bar moved, what stands out
            # from the new level is what gets reported.
            thr = self.current_threshold()
            if thr > w.threshold_db:
                self.rejudged += 1
                log.debug(
                    "%s: %.0f s window at %.2f re-judged at %.1f dB (was %.1f)",
                    self.channel, length, w.start, thr, w.threshold_db,
                )
                return self._replay(w.pending, thr, commit=False)
        return [self._window(w, forced, median)]

    def _window(self, w: _Open, forced: bool, median: float) -> Window:
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
