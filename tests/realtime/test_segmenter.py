"""Tests for the online segmenter (task T4), from plain dB arrays only."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import pytest

from VidFlow.realtime.config import RealtimeConfig
from VidFlow.realtime.segmenter import WEIGHT_UNIT, HistogramLoudnessStats, LoudnessStats, OnlineSegmenter
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
    # Against a long quiet history a 100 s burst is a moment: its first max_len
    # is reported (forced). Its samples then enter the background at full
    # weight, and from then on the level is "normal": the re-judged rest is quiet.
    windows = run(series(800, [(640, 740)]))
    spans = [(round(w.start, 6), round(w.end, 6), w.forced) for w in windows]
    assert spans == [(640, 685, True)]


def test_max_len_split_while_background_does_not_move() -> None:
    # With a long enough history the burst stays under 10 % of the background
    # throughout, the bar never moves, and max_len splits the burst as before.
    cfg = CFG.replace(stats_window_s=3000.0, max_len_s=20.0, retention_s=120.0)
    windows = run(series(2600, [(2500, 2555)]), cfg)
    spans = [(round(w.start, 6), round(w.end, 6), w.forced) for w in windows]
    assert spans == [(2500, 2520, True), (2520, 2540, True), (2540, 2555, False)]


def test_level_change_is_rejudged_not_reported() -> None:
    # Quiet, then the stream stays 27 dB louder (a game starts) with one real
    # peak. The rise opens a window that runs to max_len; with its samples in
    # the background the bar moves up, the span is re-judged, and only the peak
    # (judged against the new level) is reported.
    samples = series(240, [(120, 240)], quiet=-45.0, loud=-18.0)
    samples = [LoudnessSample(s.t, -4.0 if 150 <= s.t < 156 - 1e-9 else s.db) for s in samples]
    seg = OnlineSegmenter("ch", CFG)
    windows = seg.feed(samples) + seg.flush()
    assert [(round(w.start, 6), round(w.end, 6)) for w in windows] == [(150, 156)]
    assert seg.rejudged == 1
    assert windows[0].stats["threshold_db"] == pytest.approx(-18.0 + CFG.rise_db, abs=1.0)


def test_warmup_is_judged_with_hindsight_not_skipped() -> None:
    samples = series(60, [(10, 16), (40, 46)])
    seg = OnlineSegmenter("ch", CFG)
    emitted = []
    for i, s in enumerate(samples):
        for w in seg.feed([s]):
            emitted.append((i, round(w.start, 6), round(w.end, 6)))
        if i < 299:
            assert not seg.is_open and not seg.warmed_up
    assert seg.flush() == []
    # The warm-up window comes out with the sample that completes warmup_s (index 299).
    assert emitted == [(299, 10, 16), (499, 40, 46)]
    assert seg.warmup_threshold == CFG.floor_db


def test_flush_judges_a_short_stream() -> None:
    seg = OnlineSegmenter("ch", CFG)
    assert seg.feed(series(20, [(5, 9)])) == []
    (w,) = seg.flush()
    assert (w.start, round(w.end, 6), w.forced) == (5, 9, False)
    assert seg.flush() == []


def test_reset_during_warmup_is_replayed() -> None:
    seg = OnlineSegmenter("ch", CFG)
    seg.feed(series(10, [(6, 10)]))
    assert seg.reset() == []  # recorded, not applied yet
    later = [LoudnessSample(10 + s.t, s.db) for s in series(25, [(0, 2)])]
    windows = seg.feed(later) + seg.flush()
    # The burst 6..12 is cut at the reset (10) as the live path would have done.
    assert [(round(w.start, 6), round(w.end, 6), w.forced) for w in windows] == [(6, 10, True), (10, 12, False)]


def test_warmup_dominated_by_one_level_takes_it_as_normal() -> None:
    # Known limit: the window-weight budget (about 31 %) applies to the short
    # warm-up history too, so a level filling ~43 % of it is the norm.
    seg = OnlineSegmenter("ch", CFG)
    assert seg.feed(series(30, [(10, 16), (20, 27)])) + seg.flush() == []


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
    base = series(
        300, [(12, 15), (35, 38), (40, 47), (60, 61), (80, 100), (150, 152.5), (155, 156), (200, 226), (250, 253)]
    )
    return [LoudnessSample(s.t, s.db + float(rng.normal(0, 0.3))) for s in base]


def test_batching_does_not_change_result() -> None:
    samples = _noisy_series()
    whole = run(samples)
    assert len(whole) >= 6
    assert whole[0].start == pytest.approx(12)  # found in the warm-up
    assert run(samples, chunk=1) == whole
    assert run(samples, chunk=20) == whole
    assert run(samples, chunk=7) == whole
    assert run(samples, chunk=299) == whole  # a batch ending just before the warm-up completes


def test_batching_does_not_change_result_with_rejudge_and_resets() -> None:
    samples = series(240, [(20, 25), (120, 240)], quiet=-45.0, loud=-18.0)
    samples = [LoudnessSample(s.t, -4.0 if 150 <= s.t < 156 - 1e-9 else s.db) for s in samples]

    resets = (220, 1000)  # discontinuities: one inside a warm-up burst, one live

    def go(chunk: int) -> List[Window]:
        seg = OnlineSegmenter("ch", CFG)
        out: List[Window] = []
        cuts = sorted(set(range(0, len(samples), chunk)) | set(resets)) + [len(samples)]
        for a, b in zip(cuts, cuts[1:]):
            if a in resets:
                out += seg.reset()
            out += seg.feed(samples[a:b])
        return out + seg.flush()

    whole = go(len(samples))
    assert [(round(w.start, 6), round(w.end, 6), w.forced) for w in whole] == [
        (20, 22, True), (22, 25, False), (150, 156, False)
    ]
    for chunk in (1, 7, 50, 100):
        assert go(chunk) == whole


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


# --------------------------------------------------------------------------- the threshold rule


def test_weighted_histogram_percentile() -> None:
    st = HistogramLoudnessStats(100, 0.1)
    for _ in range(80):
        st.update(-40.0)
    for _ in range(20):
        st.update_weighted(-5.0, WEIGHT_UNIT // 4)  # 20 samples at a quarter weight = 5 full
    assert st.count == 100
    assert st.percentile(90) == -40.0  # 5 / 85 of the weight is loud: under 10 %
    assert st.percentile(97) == -5.0
    # Plain update() is update_weighted(WEIGHT_UNIT): the unweighted rank rule
    # (checked against numpy in test_histogram_stats_match_numpy_within_1db).
    # All-zero weights fall back to counting samples.
    z = HistogramLoudnessStats(10, 0.1)
    for v in (-30.0, -20.0, -10.0):
        z.update_weighted(v, 0)
    assert z.percentile(50) == -20.0
    with pytest.raises(ValueError):
        z.update_weighted(-1.0, -1)


def test_loud_bed_is_not_a_highlight_but_peaks_above_it_are() -> None:
    # Bed at -20 dBFS (above floor_db) with +/-1 dB jitter; peaks of different levels.
    rng = np.random.default_rng(11)
    peaks = {(50.0, 55.0): -8.0, (90.0, 96.0): -5.0, (130.0, 134.0): -11.0}
    samples = []
    for i in range(1800):
        t = i * HOP
        lvl = next((v for (a, b), v in peaks.items() if a <= t < b - 1e-9), -20.0)
        samples.append(LoudnessSample(t, lvl + float(rng.normal(0, 0.5))))
    windows = run(samples)
    assert [(round(w.start), round(w.end)) for w in windows] == list((round(a), round(b)) for a, b in peaks)
    for w in windows:
        assert w.stats["threshold_db"] == pytest.approx(-20 + CFG.rise_db, abs=1.5)


def test_mostly_loud_stream_still_finds_louder_moments() -> None:
    # Loud (-15) 80 % of the time with quiet gaps; two moments 10 dB louder.
    gaps = [(g, g + 2.0) for g in range(5, 300, 10)]
    samples = []
    for i in range(3000):
        t = i * HOP
        db = -50.0 if any(a <= t < b - 1e-9 for a, b in gaps) else -15.0
        if 100 <= t < 104 - 1e-9 or 201 <= t < 204 - 1e-9:
            db = -5.0
        samples.append(LoudnessSample(t, db))
    windows = run(samples)
    assert [(round(w.start, 6), round(w.end, 6)) for w in windows] == [(100, 104), (201, 204)]


def test_detected_moments_do_not_raise_the_bar_for_quieter_ones() -> None:
    # A long, very loud moment, then a smaller one. With every sample counted
    # in full (the old rule) the first fills 23 % of the history and puts the
    # 90th percentile at its level, hiding the second.
    samples = series(160, [(40, 72)], loud=-5.0)
    samples = [LoudnessSample(s.t, -15.0 if 140 <= s.t < 145 - 1e-9 else s.db) for s in samples]
    spans = [(round(w.start, 6), round(w.end, 6)) for w in run(samples)]
    assert spans == [(40, 72), (140, 145)]
    old_like = CFG.replace(window_weight=1.0, rise_db=0.0)
    assert [(round(w.start, 6), round(w.end, 6)) for w in run(samples, old_like)] == [(40, 72)]


def test_modulated_bed_above_floor_gives_nothing() -> None:
    # The old rule (90th percentile, no rise) flagged the crests of any bed above floor_db.
    t = np.arange(3000) * HOP
    rng = np.random.default_rng(5)
    dbs = -25.0 + 4.0 * np.sin(2 * np.pi * 0.08 * t) + rng.normal(0, 0.7, t.size)
    assert run([LoudnessSample(float(a), float(b)) for a, b in zip(t, dbs)]) == []


def test_frequent_moments_become_normal() -> None:
    # Known limit: a level that keeps being detected becomes the norm once it
    # fills about (1-p)/(w*p + 1-p) of the history (31 % at p=90, w=0.25).
    bursts = [(float(a), float(a + 5)) for a in range(40, 600, 10)]  # 50 % of the time from t=40
    windows = run(series(600, bursts))
    assert 0 < len(windows) < len(bursts) // 2
    assert all(w.end <= 300 for w in windows)  # detected at first, then not


def test_retune_from_logged_samples_is_exact() -> None:
    # Deterministic from the samples alone: two independent runs agree on every field.
    samples = _noisy_series()
    assert run(samples) == run(list(samples))


def test_config_validation_of_rule_fields() -> None:
    from VidFlow.realtime.config import ConfigError

    CFG.replace(window_weight=0.0, rise_db=0.0).validate()
    with pytest.raises(ConfigError):
        CFG.replace(window_weight=1.5).validate()
    with pytest.raises(ConfigError):
        CFG.replace(window_weight=-0.1).validate()
    with pytest.raises(ConfigError):
        CFG.replace(rise_db=float("nan")).validate()
    # Windows found in the warm-up are returned when it ends: retention must still hold their start.
    with pytest.raises(ConfigError, match="warmup_s"):
        CFG.replace(warmup_s=200.0).validate()
    CFG.replace(warmup_s=200.0, retention_s=210.0).validate()
    # A config logged before these fields existed still loads, with the defaults.
    old = {k: v for k, v in CFG.to_dict().items() if k not in ("rise_db", "window_weight")}
    assert RealtimeConfig.from_dict(old) == CFG
