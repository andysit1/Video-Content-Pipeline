"""Tests for LoudnessExtractor and RollingLoudnessStats (task T3, design section 6.3)."""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

from VidFlow.realtime.audio import DB_CEIL, DB_FLOOR, LoudnessExtractor, RollingLoudnessStats, rms_db
from VidFlow.realtime.media import run_ffmpeg
from VidFlow.realtime.testing import SYNTHETIC_BURSTS, make_segments
from VidFlow.realtime.types import LoudnessSample, Segment

HOP = 0.1


def _run(segments: List[Segment], extractor: LoudnessExtractor | None = None) -> List[LoudnessSample]:
    ex = extractor or LoudnessExtractor(hop_s=HOP)
    out: List[LoudnessSample] = []
    for seg in segments:
        out.extend(ex.feed(seg))
    return out


@pytest.fixture(scope="module")
def series(synthetic_segments: List[Segment]) -> List[LoudnessSample]:
    return _run(synthetic_segments)


def _arrays(samples: List[LoudnessSample]):
    return np.array([s.t for s in samples]), np.array([s.db for s in samples])


# ---------------------------------------------------------------- extractor


def test_sample_count_matches_duration(series, synthetic_segments):
    duration = synthetic_segments[-1].end
    assert abs(len(series) - duration / HOP) <= 2


def test_framing_each_segment_alone_loses_audio(synthetic_segments):
    # Documents why carry-over exists: a fresh extractor per segment drops the
    # leftover of every segment.
    alone = sum(len(LoudnessExtractor(hop_s=HOP).feed(s)) for s in synthetic_segments)
    carried = len(_run(synthetic_segments))
    assert carried > alone


def test_times_are_a_regular_grid_from_media_start(series, synthetic_segments):
    t, _ = _arrays(series)
    assert t[0] == synthetic_segments[0].start
    assert np.allclose(np.diff(t), HOP, atol=1e-9)
    # Never runs past the segment that completed it.
    assert t[-1] + HOP <= synthetic_segments[-1].end + 0.05


def test_samples_never_ahead_of_their_segment(synthetic_segments):
    ex = LoudnessExtractor(hop_s=HOP)
    for seg in synthetic_segments:
        for s in ex.feed(seg):
            # A hop is emitted only once complete, by the segment holding its end
            # (allowing for one audio frame of packet/keyframe misalignment).
            assert s.t + HOP <= seg.end + 0.05
            assert s.t + HOP > seg.start - 0.05


def test_bursts_are_25db_above_quiet(series):
    t, db = _arrays(series)
    in_burst = np.zeros(t.shape, bool)
    near_edge = np.zeros(t.shape, bool)
    for b0, b1 in SYNTHETIC_BURSTS:
        in_burst |= (t >= b0 + 0.2) & (t + HOP <= b1 - 0.2)
        near_edge |= (t + HOP > b0 - 0.2) & (t < b1 + 0.2)
    outside = ~near_edge
    assert in_burst.sum() > 80 and outside.sum() > 300
    assert db[in_burst].min() >= db[outside].max() + 25.0
    # Levels match the fixture's design (about -4.9 and -44.9 dBFS RMS).
    assert np.median(db[in_burst]) == pytest.approx(-4.9, abs=0.5)
    assert np.median(db[outside]) == pytest.approx(-44.9, abs=0.5)


def test_burst_onsets_within_150ms(series):
    t, db = _arrays(series)
    for b0, b1 in SYNTHETIC_BURSTS:
        inside = (t >= b0 + 0.5) & (t < b1 - 0.5)
        before = (t >= b0 - 5.0) & (t < b0 - 0.5)
        mid = (np.median(db[inside]) + np.median(db[before])) / 2.0
        cand = np.nonzero((t >= b0 - 3.0) & (db >= mid))[0]
        assert cand.size
        assert abs(t[cand[0]] - b0) <= 0.15


def test_video_only_segment_returns_empty(synthetic_segments, tmp_path, caplog):
    src = synthetic_segments[3].path
    dst = tmp_path / "video_only.ts"
    run_ffmpeg(["-i", str(src), "-map", "0:v", "-c", "copy", "-f", "mpegts", str(dst)])
    seg = dataclasses.replace(synthetic_segments[3], path=dst)
    with caplog.at_level(logging.WARNING, logger="VidFlow.realtime.audio"):
        assert LoudnessExtractor().feed(seg) == []
    assert any("no audio stream" in r.getMessage() for r in caplog.records)


def test_corrupt_or_missing_segment_returns_empty(tmp_path, caplog):
    (seg_bad,) = make_segments("c", [2.0], directory=tmp_path)  # one empty TS packet
    seg_bad.path.write_bytes(b"not a transport stream" * 100)
    seg_missing = dataclasses.replace(seg_bad, seq=1, start=2.0, path=tmp_path / "missing.ts")
    ex = LoudnessExtractor()
    with caplog.at_level(logging.WARNING, logger="VidFlow.realtime.audio"):
        assert ex.feed(seg_bad) == []
        assert ex.feed(seg_missing) == []
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) >= 2


def test_audio_after_a_bad_segment_restarts_at_its_start(synthetic_segments, tmp_path):
    ex = LoudnessExtractor(hop_s=HOP)
    _run(synthetic_segments[:2], ex)
    bad = dataclasses.replace(synthetic_segments[2], path=tmp_path / "missing.ts")
    assert ex.feed(bad) == []
    assert ex.pending_samples == 0
    out = ex.feed(synthetic_segments[3])
    assert out[0].t == synthetic_segments[3].start


def test_discontinuity_drops_leftover_and_reanchors(synthetic_segments):
    ex = LoudnessExtractor(hop_s=HOP)
    _run(synthetic_segments[:3], ex)
    assert ex.pending_samples > 0  # the fixture's segments are not whole hops
    seg = dataclasses.replace(synthetic_segments[5], discontinuity=True)
    out = ex.feed(seg)
    assert out[0].t == seg.start
    # Same result as a fresh extractor fed only that segment.
    fresh = LoudnessExtractor(hop_s=HOP).feed(seg)
    assert out == fresh


def test_reset_drops_leftover(synthetic_segments):
    ex = LoudnessExtractor(hop_s=HOP)
    _run(synthetic_segments[:3], ex)
    ex.reset()
    assert ex.pending_samples == 0
    seg = synthetic_segments[3]
    assert ex.feed(seg) == LoudnessExtractor(hop_s=HOP).feed(seg)


def test_leftover_is_carried_into_next_segment(synthetic_segments):
    ex = LoudnessExtractor(hop_s=HOP)
    first = ex.feed(synthetic_segments[0])
    left = ex.pending_samples
    assert 0 < left < ex.hop_samples
    second = ex.feed(synthetic_segments[1])
    # The first hop of segment 1 starts inside segment 0's audio, contiguous with the last.
    assert second[0].t == pytest.approx(first[-1].t + HOP, abs=1e-9)
    assert second[0].t < synthetic_segments[1].start


def test_large_drift_reanchors_on_segment_start(synthetic_segments, caplog):
    ex = LoudnessExtractor(hop_s=HOP, resync_s=0.25)
    ex.feed(synthetic_segments[0])
    pending = ex.pending_samples
    # Pretend a second of audio went missing without a discontinuity flag.
    shifted = dataclasses.replace(synthetic_segments[1], start=synthetic_segments[1].start + 1.0)
    with caplog.at_level(logging.WARNING, logger="VidFlow.realtime.audio"):
        out = ex.feed(shifted)
    assert any("re-anchoring" in r.getMessage() for r in caplog.records)
    assert out[0].t == pytest.approx(shifted.start - pending / ex.rate, abs=1e-9)


def test_small_jitter_does_not_reanchor(synthetic_segments, caplog):
    with caplog.at_level(logging.WARNING, logger="VidFlow.realtime.audio"):
        _run(synthetic_segments)
    assert not any("re-anchoring" in r.getMessage() for r in caplog.records)


def test_media_time_offset_follows_segment_start(synthetic_segments):
    # A session that starts later in media time shifts every sample by the same amount.
    offset = 100.0
    moved = [dataclasses.replace(s, start=s.start + offset) for s in synthetic_segments[:4]]
    a = _run(synthetic_segments[:4])
    b = _run(moved)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert y.t == pytest.approx(x.t + offset, abs=1e-9)
        assert y.db == x.db


def test_constructor_validation():
    with pytest.raises(ValueError):
        LoudnessExtractor(hop_s=0)
    with pytest.raises(ValueError):
        LoudnessExtractor(rate=0)
    with pytest.raises(ValueError):
        LoudnessExtractor(hop_s=1e-6, rate=16000)
    assert LoudnessExtractor(hop_s=0.1, rate=16000).hop_samples == 1600


def test_rms_db_helper():
    x = np.full((2, 100), 0.5)
    x[1] = 0.0
    out = rms_db(x)
    assert out[0] == pytest.approx(20 * np.log10(0.5))
    assert out[1] == DB_FLOOR


def test_timing_per_segment(synthetic_segments):
    ex = LoudnessExtractor(hop_s=HOP)
    times = []
    for seg in synthetic_segments:
        t0 = time.perf_counter()
        ex.feed(seg)
        times.append(time.perf_counter() - t0)
    # Loose sanity bound only; the measured figure is reported, not asserted.
    assert float(np.median(times)) < 0.25


# -------------------------------------------------------------------- stats


def _mixture(rng: np.random.Generator, n: int) -> np.ndarray:
    quiet = rng.normal(-45.0, 3.0, n)
    loud = rng.normal(-8.0, 2.0, n)
    pick = rng.random(n) < 0.15
    return np.where(pick, loud, quiet)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_percentile_matches_numpy_including_after_aging(seed):
    rng = np.random.default_rng(seed)
    stats = RollingLoudnessStats(window_s=60.0, hop_s=0.1)
    assert stats.capacity == 600
    values = _mixture(rng, 2500)
    checkpoints = {50, 599, 600, 601, 1000, 1777, 2499}
    for i, v in enumerate(values):
        stats.update(v)
        if i in checkpoints:
            window = values[max(0, i + 1 - stats.capacity): i + 1]
            assert stats.count == len(window)
            for p in (0, 1, 10, 25, 50, 75, 90, 95, 99, 100):
                assert stats.percentile(p) == pytest.approx(np.percentile(window, p), abs=1.0), (i, p)


def test_percentile_after_level_change_ages_out_old_samples():
    stats = RollingLoudnessStats(window_s=10.0, hop_s=0.1)
    stats.extend([-60.0] * 100)
    assert stats.percentile(50) == pytest.approx(-60.0)
    stats.extend([-20.0] * 100)  # the -60 dB samples have all aged out
    assert stats.count == 100
    assert stats.percentile(0) == pytest.approx(-20.0)
    assert stats.percentile(100) == pytest.approx(-20.0)


def test_percentile_exact_for_repeated_levels():
    stats = RollingLoudnessStats(window_s=5.0, hop_s=0.1)
    stats.extend([-44.95] * 30 + [-4.95] * 20)
    values = [-44.95] * 30 + [-4.95] * 20
    for p in (0, 50, 59, 60, 61, 90, 100):
        assert stats.percentile(p) == pytest.approx(np.percentile(values, p), abs=1e-9)


def test_percentile_on_real_series(series):
    stats = RollingLoudnessStats(window_s=20.0, hop_s=HOP)
    dbs = np.array([s.db for s in series])
    for i, v in enumerate(dbs):
        stats.update(v)
        if i % 37 == 0:
            window = dbs[max(0, i + 1 - stats.capacity): i + 1]
            for p in (50, 90):
                assert stats.percentile(p) == pytest.approx(np.percentile(window, p), abs=1.0)


def test_stats_clamps_and_validates():
    stats = RollingLoudnessStats(window_s=1.0, hop_s=0.1)
    with pytest.raises(ValueError):
        stats.percentile(50)  # empty
    stats.update(float("-inf"))
    stats.update(1000.0)
    assert stats.percentile(0) == DB_FLOOR
    assert stats.percentile(100) == DB_CEIL
    with pytest.raises(ValueError):
        stats.update(float("nan"))
    with pytest.raises(ValueError):
        stats.percentile(101)
    stats.reset()
    assert stats.count == 0
    with pytest.raises(ValueError):
        RollingLoudnessStats(window_s=0, hop_s=0.1)
