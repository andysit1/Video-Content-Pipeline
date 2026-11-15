"""Tests for ReplaySource (task T1, design section 6.1)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from VidFlow.realtime.media import probe
from VidFlow.realtime.sources import ReplaySource
from VidFlow.realtime.types import Source


def _check_contiguous(segments):
    assert segments[0].start == 0.0
    for prev, cur in zip(segments, segments[1:]):
        assert cur.seq > prev.seq
        assert cur.start == prev.end  # exact, by construction


def test_is_a_source(synthetic_video: Path, tmp_path: Path) -> None:
    src = ReplaySource(synthetic_video, "ch", tmp_path, speed=None)
    assert isinstance(src, Source)
    assert src.channel == "ch"


def test_unpaced_yields_every_segment(synthetic_video: Path, tmp_path: Path) -> None:
    src = ReplaySource(synthetic_video, "ch", tmp_path, speed=None)
    planned = src.prepare()
    t = time.monotonic()
    segments = list(src)
    elapsed = time.monotonic() - t

    assert elapsed < 0.5
    assert len(segments) == len(planned) >= 29
    assert [s.seq for s in segments] == list(range(len(segments)))
    _check_contiguous(segments)
    assert not any(s.discontinuity for s in segments)
    assert all(s.channel == "ch" for s in segments)
    assert all(s.path.is_file() for s in segments)
    assert all(s.arrived_at > 0 for s in segments)
    assert segments[-1].end == pytest.approx(probe(synthetic_video).duration, abs=0.1)
    # Only the yielded segment files are left behind (CSV list removed).
    assert sorted(tmp_path.iterdir()) == sorted(s.path for s in segments)


def test_drop_flags_discontinuity_and_keeps_media_time(synthetic_video: Path, tmp_path: Path) -> None:
    full = list(ReplaySource(synthetic_video, "a", tmp_path / "a", speed=None))
    src = ReplaySource(synthetic_video, "b", tmp_path / "b", speed=None, drop={5})
    segments = list(src)

    assert len(segments) == len(full) - 1
    assert 5 not in [s.seq for s in segments]
    _check_contiguous(segments)
    flagged = [s.seq for s in segments if s.discontinuity]
    assert flagged == [6]
    # The withheld segment's file is cleaned up.
    assert sorted((tmp_path / "b").iterdir()) == sorted(s.path for s in segments)


def test_drop_first_segment_is_not_a_discontinuity(synthetic_video: Path, tmp_path: Path) -> None:
    segments = list(ReplaySource(synthetic_video, "ch", tmp_path, speed=None, drop={0}))
    assert segments[0].seq == 1
    assert segments[0].start == 0.0
    assert not any(s.discontinuity for s in segments)


def test_two_sources_share_a_work_dir(synthetic_video: Path, tmp_path: Path) -> None:
    a = list(ReplaySource(synthetic_video, "ch", tmp_path, speed=None))
    b = list(ReplaySource(synthetic_video, "ch", tmp_path, speed=None))
    assert not {s.path for s in a} & {s.path for s in b}
    assert all(s.path.is_file() for s in a + b)


def test_speed_20_paces_by_media_time(synthetic_video: Path, tmp_path: Path) -> None:
    src = ReplaySource(synthetic_video, "ch", tmp_path, speed=20)
    src.prepare()  # splitting is not part of the schedule
    t = time.monotonic()
    segments = list(src)
    elapsed = time.monotonic() - t

    assert elapsed == pytest.approx(3.0, abs=0.5)
    assert src.t0 is not None
    for s in segments:
        assert s.arrived_at >= src.t0 + s.end / 20
        assert s.arrived_at - (src.t0 + s.end / 20) < 0.1


def test_jitter_delays_but_keeps_order(synthetic_video: Path, tmp_path: Path) -> None:
    src = ReplaySource(synthetic_video, "ch", tmp_path, speed=50, jitter=0.05, seed=1)
    segments = list(src)
    assert src.t0 is not None
    _check_contiguous(segments)
    lateness = [s.arrived_at - (src.t0 + s.end / 50) for s in segments]
    assert min(lateness) >= 0
    assert max(lateness) < 0.15
    assert all(b.arrived_at >= a.arrived_at for a, b in zip(segments, segments[1:]))


def test_close_during_wait_ends_iteration(synthetic_video: Path, tmp_path: Path) -> None:
    src = ReplaySource(synthetic_video, "ch", tmp_path, speed=1.0)
    src.prepare()
    got = []
    done = threading.Event()

    def consume() -> None:
        for seg in src:
            got.append(seg)
        done.set()

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    time.sleep(0.3)  # the first segment is due after about 2 s
    t = time.monotonic()
    src.close()
    assert done.wait(1.0)
    assert time.monotonic() - t < 0.2
    assert got == []
    # Nothing was handed out, so nothing is left behind.
    assert list(tmp_path.iterdir()) == []


def test_close_before_iterating(synthetic_video: Path, tmp_path: Path) -> None:
    src = ReplaySource(synthetic_video, "ch", tmp_path, speed=1.0)
    src.close()
    assert list(src) == []


def test_iterate_once(synthetic_video: Path, tmp_path: Path) -> None:
    src = ReplaySource(synthetic_video, "ch", tmp_path, speed=None)
    list(src)
    with pytest.raises(RuntimeError):
        iter(src)


def test_transcode_gives_even_segments(synthetic_video: Path, tmp_path: Path) -> None:
    segments = list(ReplaySource(synthetic_video, "ch", tmp_path, speed=None, segment_time=3.0, transcode=True))
    _check_contiguous(segments)
    for s in segments[:-1]:
        assert s.duration == pytest.approx(3.0, abs=0.1)


def test_rejects_bad_arguments(synthetic_video: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ReplaySource(synthetic_video, "ch", tmp_path, speed=0)
    with pytest.raises(ValueError):
        ReplaySource(synthetic_video, "ch", tmp_path, jitter=-1)
