"""Tests for the online segmenter (task T4), from plain dB arrays only."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import pytest

from VidFlow.realtime.config import RealtimeConfig
from VidFlow.realtime.segmenter import HistogramLoudnessStats, LoudnessStats, OnlineSegmenter
from VidFlow.realtime.types import LoudnessSample, Window, make_window_id

HOP = 0.1
QUIET = -45.0
LOUD = -5.0
CFG = RealtimeConfig()  # defaults: warmup 30, merge_gap 4, min_len 1.5, max_len 45


def series(
    duration: float,
    bursts: Sequence[Tuple[float, float]] = (),
    quiet: float = QUIET,
    loud: float = LOUD,
    hop: float = HOP,
) -> List[LoudnessSample]:
    """Samples at t = i * hop; loud for hops inside any [b0, b1)."""
    n = int(round(duration / hop))
    idx = [(int(round(b0 / hop)), int(round(b1 / hop))) for b0, b1 in bursts]
    return [
        LoudnessSample(t=i * hop, db=loud if any(a <= i < b for a, b in idx) else quiet)
        for i in range(n)
    ]


def run(samples: Sequence[LoudnessSample], cfg: RealtimeConfig = CFG, chunk: int = 0) -> List[Window]:
    seg = OnlineSegmenter("ch", cfg)
    out: List[Window] = []
    if chunk <= 0:
        out += seg.feed(samples)
    else:
        for i in range(0, len(samples), chunk):
            out += seg.feed(samples[i : i + chunk])
    out += seg.flush()
    return out


def test_one_burst_one_window() -> None:
    windows = run(series(60, [(40, 46)]))
    assert len(windows) == 1
    w = windows[0]
    assert abs(w.start - 40) <= HOP + 1e-9
    assert abs(w.end - 46) <= HOP + 1e-9
    assert not w.forced
    assert w.channel == "ch" and w.detector == "audio"
    assert w.window_id == make_window_id("ch", w.start) == "ch-000040000"
    assert set(w.stats) == {"peak_db", "mean_db", "threshold_db", "prominence_db"}
    assert w.stats["peak_db"] == pytest.approx(LOUD)
    assert w.stats["mean_db"] == pytest.approx(LOUD)
    assert w.stats["threshold_db"] == pytest.approx(CFG.floor_db)  # p90 of quiet is below the floor
    assert w.stats["prominence_db"] == pytest.approx(LOUD - QUIET, abs=1.0)


def test_bursts_3s_apart_merge() -> None:
    windows = run(series(70, [(40, 43), (46, 49)]))
    assert len(windows) == 1
    assert windows[0].start == pytest.approx(40, abs=HOP)
    assert windows[0].end == pytest.approx(49, abs=HOP)
    # The quiet gap inside the window counts towards the mean.
    assert windows[0].stats["mean_db"] == pytest.approx((6 * LOUD + 3 * QUIET) / 9, abs=0.01)


def test_bursts_5s_apart_do_not_merge() -> None:
    windows = run(series(70, [(40, 43), (48, 51)]))
    assert [(round(w.start, 6), round(w.end, 6)) for w in windows] == [(40, 43), (48, 51)]


def test_short_burst_discarded() -> None:
    assert run(series(60, [(40, 41)])) == []


def test_long_burst_split_at_max_len() -> None:
    windows = run(series(200, [(40, 140)]))
    spans = [(round(w.start, 6), round(w.end, 6), w.forced) for w in windows]
    assert spans == [(40, 85, True), (85, 130, True), (130, 140, False)]


def test_nothing_during_warmup() -> None:
    seg = OnlineSegmenter("ch", CFG)
    out = []
    for s in series(30, [(10, 16), (20, 27)]):
        out += seg.feed([s])
        assert not seg.is_open
    assert out == [] and seg.flush() == []


def test_warmup_counts_samples_not_media_time() -> None:
    # A burst right after warm-up is detected; warm-up is not restarted by reset().
    seg = OnlineSegmenter("ch", CFG.replace(warmup_s=5))
    assert seg.feed(series(5)) == []
    assert seg.reset() == []
    samples = [LoudnessSample(5 + s.t, s.db) for s in series(10, [(0, 3)])]
    windows = seg.feed(samples)
    assert len(windows) == 1 and windows[0].start == pytest.approx(5)


def test_window_emitted_by_sample_completing_gap() -> None:
    samples = series(60, [(40, 46)])
    seg = OnlineSegmenter("ch", CFG)
    emitted_at = []
    for i, s in enumerate(samples):
        if seg.feed([s]):
            emitted_at.append(i)
    # Last loud hop is [45.9, 46.0); 4 s of quiet ends at 50.0, i.e. the hop starting at 49.9.
    assert emitted_at == [499]
    assert samples[499].t + HOP == pytest.approx(46 + CFG.merge_gap_s)


def _noisy_series() -> List[LoudnessSample]:
    rng = np.random.default_rng(1234)
    base = series(300, [(35, 38), (40, 47), (60, 61), (80, 140), (150, 152.5), (155, 156), (200, 260)])
    return [LoudnessSample(s.t, s.db + float(rng.normal(0, 0.3))) for s in base]


def test_batching_does_not_change_result() -> None:
    samples = _noisy_series()
    whole = run(samples)
    assert len(whole) >= 4
    assert run(samples, chunk=1) == whole
    assert run(samples, chunk=20) == whole
    assert run(samples, chunk=7) == whole


def test_constant_noise_below_floor_gives_nothing() -> None:
    assert run(series(300, quiet=-40.0)) == []
    rng = np.random.default_rng(0)
    noisy = [LoudnessSample(i * HOP, -42.0 + float(rng.normal(0, 2.0))) for i in range(3000)]
    assert run(noisy) == []


def test_flush_closes_open_window_forced() -> None:
    seg = OnlineSegmenter("ch", CFG)
    assert seg.feed(series(45, [(40, 45)])) == []
    assert seg.is_open
    (w,) = seg.flush()
    assert w.forced and w.start == pytest.approx(40) and w.end == pytest.approx(45)
    assert not seg.is_open and seg.flush() == []


def test_reset_closes_forced_or_discards_short() -> None:
    seg = OnlineSegmenter("ch", CFG)
    seg.feed(series(43, [(40, 43)]))
    (w,) = seg.reset()
    assert w.forced and w.end == pytest.approx(43)

    seg = OnlineSegmenter("ch", CFG)
    seg.feed(series(41, [(40, 41)]))
    assert seg.is_open
    assert seg.reset() == []
    assert not seg.is_open


def test_threshold_from_previous_samples_only() -> None:
    # Stats are fed every sample, after it is classified.
    class Recorder:
        def __init__(self) -> None:
            self.values: List[float] = []
            self.queries: List[int] = []

        def count(self) -> int:  # method form of count is accepted too
            return len(self.values)

        def update(self, db: float) -> None:
            self.values.append(db)

        def percentile(self, p: float) -> float:
            self.queries.append(len(self.values))
            return float(np.percentile(self.values, p))

    rec = Recorder()
    seg = OnlineSegmenter("ch", CFG.replace(warmup_s=1.0), stats=rec)
    windows = seg.feed(series(20, [(5, 9)])) + seg.flush()
    assert len(rec.values) == 200
    assert len(windows) == 1 and windows[0].start == pytest.approx(5)
    # The sample at index 50 opened the window, after 50 samples had been seen.
    assert 50 in rec.queries


def test_injected_stats_raise_threshold() -> None:
    class Fixed:
        count = 1

        def update(self, db: float) -> None:
            pass

        def percentile(self, p: float) -> float:
            return -3.0  # above the burst level

    assert isinstance(Fixed(), LoudnessStats)
    seg = OnlineSegmenter("ch", CFG, stats=Fixed())
    assert seg.feed(series(60, [(40, 46)])) + seg.flush() == []


def test_histogram_stats_match_numpy_within_1db() -> None:
    rng = np.random.default_rng(7)
    values = np.concatenate([rng.normal(-40, 6, 1500), rng.normal(-10, 3, 400), rng.normal(-30, 10, 1500)])
    window_s, hop = 200.0, 0.1  # 2000 samples: the oldest age out
    st = HistogramLoudnessStats(window_s, hop)
    cap = int(round(window_s / hop))
    for i, v in enumerate(values):
        st.update(float(v))
        if i % 97 == 0 or i == len(values) - 1:
            win = values[max(0, i + 1 - cap) : i + 1]
            assert st.count == len(win)
            for p in (10, 50, 90, 99):
                lower = np.percentile(win, p, method="lower")
                got = st.percentile(p)
                assert got <= lower < got + 1.0  # the bin holding numpy's lower-index element


def test_histogram_stats_edge_values() -> None:
    st = HistogramLoudnessStats(10, 0.1)
    with pytest.raises(ValueError):
        st.percentile(50)
    for v in (-math.inf, float("nan"), -500.0, 50.0, -20.0):
        st.update(v)
    assert st.count == 5
    assert st.percentile(0) == st.min_db
    assert st.percentile(50) == st.min_db
    assert st.percentile(100) == st.max_db - 1
    # Every sample in the threshold's bin counts as >= the threshold.
    st2 = HistogramLoudnessStats(10, 0.1)
    for v in [-40.0] * 50 + [-4.9] * 50:
        st2.update(v)
    assert -4.9 >= st2.percentile(90)
