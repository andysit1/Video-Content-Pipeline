"""Tests for T2: SegmentBuffer and Lease (design section 6.2)."""

from __future__ import annotations

import random
import threading
import time
from pathlib import Path
from typing import List

import pytest

from VidFlow.realtime.buffer import Lease, SegmentBuffer
from VidFlow.realtime.config import RealtimeConfig
from VidFlow.realtime.testing import make_segments
from VidFlow.realtime.types import Segment


def _files(directory: Path) -> List[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file())


def _seqs(lease: Lease) -> List[int]:
    return [s.seq for s in lease.segments]


# ------------------------------------------------------------------ eviction


def test_eviction_keeps_duration_within_retention_plus_one_segment(tmp_path: Path) -> None:
    durations = [2.0, 1.7, 2.3, 2.0, 0.9, 2.1] * 20
    segs = make_segments("ch", durations, directory=tmp_path)
    buf = SegmentBuffer(retention_s=10.0)
    longest = max(durations)
    for s in segs:
        buf.add(s)
        total = buf.buffered_duration
        assert total <= 10.0 + longest + 1e-9
        assert buf.latest_end == pytest.approx(s.end)
        # The retention window itself is always fully covered.
        assert buf.oldest_start <= max(0.0, s.end - 10.0) + 1e-9
    # Only the files of buffered segments remain.
    kept = {s.path for s in buf.segments()}
    assert set(_files(tmp_path)) == kept
    buf.close()
    assert _files(tmp_path) == []


def test_newest_segment_is_never_evicted(tmp_path: Path) -> None:
    segs = make_segments("ch", [2.0, 2.0, 2.0], directory=tmp_path)
    buf = SegmentBuffer(retention_s=0.5)
    for s in segs:
        buf.add(s)
        assert buf.segments() == [s]
        assert s.path.exists()
    assert len(_files(tmp_path)) == 1


def test_add_rejects_bad_input(tmp_path: Path) -> None:
    segs = make_segments("ch", [2.0, 2.0], directory=tmp_path)
    other = make_segments("other", [2.0], directory=tmp_path, start_seq=5)
    buf = SegmentBuffer(30.0)
    buf.add(segs[1])
    with pytest.raises(ValueError):
        buf.add(segs[0])
    with pytest.raises(ValueError):
        buf.add(other[0])
    with pytest.raises(ValueError):
        SegmentBuffer(0)
    with pytest.raises(ValueError):
        buf.lease(5.0, 4.0)
    with pytest.raises(ValueError):
        buf.lease(0.0, 1.0, pre=-1)


def test_empty_buffer_queries() -> None:
    buf = SegmentBuffer(10.0)
    assert buf.latest_end is None
    assert buf.oldest_start is None
    assert not buf.covers(0.0)
    lease = buf.lease(0.0, 1.0)
    assert not lease and lease.segments == () and not lease.truncated


# --------------------------------------------------------------------- lease


def test_lease_covers_core_and_padding(tmp_path: Path) -> None:
    segs = make_segments("ch", [2.0] * 30, directory=tmp_path)
    buf = SegmentBuffer(120.0)
    for s in segs:
        buf.add(s)
    assert buf.covers(60.0) and not buf.covers(60.5)

    with buf.lease(10.0, 16.0, pre=5.0, post=3.0) as lease:
        # want [5, 19]: segments [4,6) .. [18,20)
        assert _seqs(lease) == list(range(2, 10))
        assert lease.start == 4.0 and lease.end == 20.0
        assert lease.duration == 16.0
        assert not lease.truncated
        assert lease.paths == [s.path for s in segs[2:10]]

    # Core boundaries exactly on segment boundaries do not pull in neighbours.
    with buf.lease(10.0, 16.0) as lease:
        assert _seqs(lease) == [5, 6, 7]
    # Fractional core times.
    with buf.lease(10.05, 15.95, pre=0.04, post=0.04) as lease:
        assert _seqs(lease) == [5, 6, 7]
    # A zero-length core leases the segment containing it.
    with buf.lease(11.0, 11.0) as lease:
        assert _seqs(lease) == [5]
    # Padding past the start of the stream is not truncation.
    with buf.lease(1.0, 3.0, pre=5.0) as lease:
        assert _seqs(lease) == [0, 1]
        assert not lease.truncated
    # Padding past what has arrived returns what is there.
    with buf.lease(56.0, 59.0, post=3.0) as lease:
        assert _seqs(lease) == [28, 29]
        assert lease.end == 60.0
    # A core not yet arrived gives an empty, untruncated lease.
    lease = buf.lease(70.0, 75.0)
    assert not lease and not lease.truncated
    assert buf.pinned_count == 0
    buf.close()


def test_leased_segment_survives_eviction_and_is_removed_on_release(tmp_path: Path) -> None:
    segs = make_segments("ch", [2.0] * 20, directory=tmp_path)
    buf = SegmentBuffer(retention_s=6.0)
    for s in segs[:3]:
        buf.add(s)
    lease = buf.lease(0.0, 4.0)
    assert _seqs(lease) == [0, 1]
    for s in segs[3:]:
        buf.add(s)
    # Evicted from the index but the files are kept.
    assert buf.oldest_start > 4.0
    assert all(s.path.exists() for s in lease.segments)
    assert not segs[2].path.exists()  # not leased, evicted, deleted
    lease.release()
    assert not segs[0].path.exists() and not segs[1].path.exists()
    lease.release()  # idempotent
    assert lease.released
    buf.close()
    assert _files(tmp_path) == []


def test_overlapping_leases_both_keep_segments_alive(tmp_path: Path) -> None:
    segs = make_segments("ch", [2.0] * 20, directory=tmp_path)
    buf = SegmentBuffer(retention_s=10.0)
    for s in segs[:5]:
        buf.add(s)
    a = buf.lease(2.0, 8.0)  # seq 1..3
    b = buf.lease(4.0, 10.0)  # seq 2..4
    assert _seqs(a) == [1, 2, 3] and _seqs(b) == [2, 3, 4]
    for s in segs[5:]:
        buf.add(s)
    a.release()
    assert not segs[1].path.exists()
    assert all(segs[i].path.exists() for i in (2, 3, 4))
    b.release()
    assert not any(segs[i].path.exists() for i in (2, 3, 4))
    assert buf.pinned_count == 0
    buf.close()
    assert _files(tmp_path) == []


def test_truncated_lease_when_head_evicted(tmp_path: Path) -> None:
    segs = make_segments("ch", [2.0] * 20, directory=tmp_path)
    buf = SegmentBuffer(retention_s=10.0)
    for s in segs:
        buf.add(s)
    assert buf.oldest_start == 30.0  # [30, 40] kept
    # Core inside, padding head evicted.
    with buf.lease(32.0, 34.0, pre=5.0) as lease:
        assert _seqs(lease) == [15, 16]
        assert lease.truncated
    # Core head evicted.
    with buf.lease(26.0, 34.0) as lease:
        assert _seqs(lease) == [15, 16]
        assert lease.start == 30.0
        assert lease.truncated
    # Padding fits: not truncated.
    with buf.lease(34.0, 36.0, pre=4.0) as lease:
        assert _seqs(lease) == [15, 16, 17]
        assert not lease.truncated
    # Whole core evicted: empty, truncated, no error.
    lease = buf.lease(10.0, 16.0, pre=5.0, post=3.0)
    assert not lease and lease.truncated
    buf.close()


def test_padding_stops_at_discontinuity(tmp_path: Path) -> None:
    # Content was lost between seq 4 and seq 5 (5 carries the flag).
    segs = make_segments("ch", [2.0] * 12, discontinuities=[5], directory=tmp_path)
    buf = SegmentBuffer(120.0)
    for s in segs:
        buf.add(s)
    # Window right after the gap: pre padding must not reach back across it.
    with buf.lease(10.5, 13.0, pre=5.0, post=3.0) as lease:
        assert _seqs(lease) == [5, 6, 7]
        assert lease.start == 10.0
        assert not lease.truncated
    # Window right before the gap: post padding must not cross it.
    with buf.lease(6.0, 9.5, pre=2.0, post=3.0) as lease:
        assert _seqs(lease) == [2, 3, 4]
        assert lease.end == 10.0
    # A discontinuity on the oldest buffered segment after eviction is a
    # boundary, not truncation.
    buf.close()

    segs2 = make_segments("d", [2.0] * 12, discontinuities=[6], directory=tmp_path / "d")
    buf2 = SegmentBuffer(retention_s=12.0)
    for s in segs2:
        buf2.add(s)
    assert buf2.oldest_start == 12.0  # seq 6, the flagged one
    with buf2.lease(13.0, 15.0, pre=5.0) as lease:
        assert _seqs(lease) == [6, 7]
        assert not lease.truncated
    buf2.close()
    assert _files(tmp_path / "d") == []


def test_close_deletes_unleased_and_defers_leased(tmp_path: Path) -> None:
    segs = make_segments("ch", [2.0] * 10, directory=tmp_path)
    buf = SegmentBuffer(120.0)
    for s in segs:
        buf.add(s)
    lease = buf.lease(4.0, 8.0)
    buf.close()
    buf.close()  # idempotent
    assert set(_files(tmp_path)) == {s.path for s in lease.segments}
    with pytest.raises(RuntimeError):
        buf.add(make_segments("ch", [2.0], start_seq=99)[0])
    with pytest.raises(RuntimeError):
        buf.lease(0.0, 1.0)
    lease.release()
    assert _files(tmp_path) == []


def test_delete_files_false_leaves_files(tmp_path: Path) -> None:
    segs = make_segments("ch", [2.0] * 10, directory=tmp_path)
    with SegmentBuffer(4.0, delete_files=False) as buf:
        for s in segs:
            buf.add(s)
    assert len(_files(tmp_path)) == 10


def test_default_config_retention_is_valid() -> None:
    cfg = RealtimeConfig()
    cfg.validate()
    assert SegmentBuffer(cfg.retention_s).retention_s == cfg.retention_s


def test_real_segments_lease(fresh_synthetic_segments: List[Segment]) -> None:
    segs = fresh_synthetic_segments
    directory = segs[0].path.parent
    with SegmentBuffer(20.0) as buf:
        for s in segs:
            buf.add(s)
        assert buf.latest_end == pytest.approx(segs[-1].end)
        start = buf.oldest_start
        with buf.lease(start + 3.0, start + 9.0, pre=1.0, post=1.0) as lease:
            assert lease.start <= start + 2.0 + 1e-6
            assert lease.end >= start + 10.0 - 1e-6
            assert all(p.exists() for p in lease.paths)
    assert _files(directory) == []


# ------------------------------------------------------------------ threads


def test_threaded_adder_and_leasers_no_errors_no_leaks(tmp_path: Path) -> None:
    n_segments = 400
    segs = make_segments("ch", [0.5] * n_segments, directory=tmp_path)
    buf = SegmentBuffer(retention_s=5.0)
    errors: List[BaseException] = []
    done = threading.Event()
    leases_taken = [0]
    count_lock = threading.Lock()

    def adder() -> None:
        try:
            for s in segs:
                buf.add(s)
                time.sleep(0.0005)
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)
        finally:
            done.set()

    def leaser(seed: int) -> None:
        rng = random.Random(seed)
        try:
            while not done.is_set():
                end = buf.latest_end
                if end is None:
                    continue
                core_end = max(0.0, end - rng.uniform(0.0, 6.0))
                core_start = max(0.0, core_end - rng.uniform(0.0, 3.0))
                with buf.lease(core_start, core_end, pre=rng.uniform(0, 2), post=rng.uniform(0, 1)) as lease:
                    for s in lease.segments:
                        assert s.path.exists(), f"leased file missing: {s.path}"
                    prev = None
                    for s in lease.segments:
                        if prev is not None:
                            assert s.seq == prev.seq + 1 and s.start == prev.end
                        prev = s
                    time.sleep(rng.uniform(0, 0.003))
                    for s in lease.segments:
                        assert s.path.exists(), f"leased file deleted while leased: {s.path}"
                with count_lock:
                    leases_taken[0] += 1
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=adder)] + [
        threading.Thread(target=leaser, args=(i,)) for i in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert not any(t.is_alive() for t in threads)
    assert not errors, errors[0]
    assert leases_taken[0] > 50
    assert buf.pinned_count == 0
    # Only what is still within retention is on disk.
    assert set(_files(tmp_path)) == {s.path for s in buf.segments()}
    assert buf.buffered_duration <= 5.0 + 0.5 + 1e-9
    buf.close()
    assert _files(tmp_path) == []
