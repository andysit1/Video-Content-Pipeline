"""Tests for AudioDetector, WorkerPool, RankTracker, ChannelPipeline and Supervisor (task T8).

Most tests drive the pipeline with ``FakeSource`` over a private copy of the
fixture's segments; end-to-end runs through ``ReplaySource`` and the CLI are
in ``test_e2e.py``.
"""

from __future__ import annotations

import _thread
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

import pytest

from VidFlow.realtime.clip_writer import ClipWriter, read_sidecar
from VidFlow.realtime.config import RealtimeConfig
from VidFlow.realtime.pipeline import AudioDetector, ChannelPipeline, Job, RankTracker, WorkerPool
from VidFlow.realtime.scoring.visual import VisualScorer
from VidFlow.realtime.session import read_events
from VidFlow.realtime.supervisor import Supervisor
from VidFlow.realtime.testing import FakeSource, SYNTHETIC_BURSTS, copy_segments
from VidFlow.realtime.types import ClipError, Segment, Window

# The fixture's first burst (10-16 s) is inside the default 30 s warm-up.
CFG = RealtimeConfig(warmup_s=5.0)


def _windows_match_bursts(windows: List[Window], tol: float = 0.5) -> None:
    assert len(windows) == len(SYNTHETIC_BURSTS), windows
    for w, (b0, b1) in zip(windows, SYNTHETIC_BURSTS):
        assert abs(w.start - b0) <= tol and abs(w.end - b1) <= tol, (w, (b0, b1))


def _events(pipe: ChannelPipeline, kind: Optional[str] = None) -> List[dict]:
    return list(read_events(pipe.session_dir, kind))


# ------------------------------------------------------------------ AudioDetector


def test_audio_detector_finds_the_bursts(fresh_synthetic_segments: List[Segment]) -> None:
    det = AudioDetector("test", CFG)
    windows: List[Window] = []
    for seg in fresh_synthetic_segments:
        samples, closed = det.detect(seg)
        assert samples, "every fixture segment has audio"
        windows += closed
    windows += det.flush()
    _windows_match_bursts(windows)


def test_audio_detector_closes_window_at_discontinuity(fresh_synthetic_segments: List[Segment]) -> None:
    import dataclasses

    segs = list(fresh_synthetic_segments)
    # Mark seq 6 (12-14 s, inside the first burst) as following a gap.
    segs[6] = dataclasses.replace(segs[6], discontinuity=True)
    det = AudioDetector("test", CFG)
    windows: List[Window] = []
    for seg in segs:
        windows += det.feed(seg)
    windows += det.flush()
    first = [w for w in windows if w.start < 20]
    assert len(first) == 2, first
    assert first[0].forced and first[0].end <= segs[6].start + 1e-6
    assert first[1].start >= segs[6].start - 1e-6


# ------------------------------------------------------------------ WorkerPool


def test_pool_runs_jobs_and_survives_failures() -> None:
    done: List[int] = []

    def bad() -> None:
        raise RuntimeError("boom")

    with WorkerPool(2, 10) as pool:
        pool.submit(Job("cut", bad))
        for i in range(5):
            pool.submit(Job("cut", lambda i=i: done.append(i)))
        assert pool.wait_idle(5)
        assert sorted(done) == list(range(5))
        assert pool.failed == 1 and pool.completed == 5


def test_pool_drops_queued_score_jobs_first() -> None:
    gate = threading.Event()
    dropped: List[str] = []
    ran: List[str] = []
    with WorkerPool(1, 2) as pool:
        pool.submit(Job("cut", gate.wait, name="blocker"))
        time.sleep(0.05)  # let the worker take the blocker
        pool.submit(Job("score", lambda: ran.append("s1"), on_drop=lambda r: dropped.append("s1:" + r)))
        pool.submit(Job("cut", lambda: ran.append("c1")))
        # Full: [s1, c1]. A cut evicts the queued score job.
        assert pool.submit(Job("cut", lambda: ran.append("c2")))
        # Full: [c1, c2], nothing droppable: the new score job is dropped.
        assert not pool.submit(Job("score", lambda: ran.append("s2"), on_drop=lambda r: dropped.append("s2:" + r)))
        gate.set()
        assert pool.wait_idle(5)
    assert dropped == ["s1:queue full", "s2:queue full"]
    assert ran == ["c1", "c2"]


def test_pool_cancel_and_shutdown_drop_queued_jobs() -> None:
    gate = threading.Event()
    dropped: List[str] = []
    pool = WorkerPool(1, 10)
    owner = object()
    pool.submit(Job("cut", gate.wait))
    time.sleep(0.05)
    pool.submit(Job("cut", lambda: None, owner=owner, on_drop=dropped.append))
    pool.submit(Job("cut", lambda: None, owner="other", on_drop=dropped.append))
    assert pool.cancel(owner) == 1
    gate.set()
    assert pool.wait_idle(5)
    pool.shutdown(wait=True, timeout=5)
    assert dropped == ["cancelled"]
    assert not pool.submit(Job("cut", lambda: None, on_drop=dropped.append))
    assert dropped[-1] == "pool shut down"


# ------------------------------------------------------------------ RankTracker


def test_rank_tracker() -> None:
    r = RankTracker({"a": 1.0, "b": 1.0})
    score, pct = r.rank({"a": 1.0, "b": 5.0})
    assert score == 0.0 and pct == 100.0
    score, pct = r.rank({"a": 3.0, "b": 5.0 + 1e-14})  # b: float noise, no spread
    assert score == pytest.approx(1.0) and pct == 100.0
    score, pct = r.rank({"a": -10.0})  # missing b is skipped
    assert score < 0 and pct == pytest.approx(100.0 / 3)


# ------------------------------------------------------------------ ChannelPipeline


def test_pipeline_cuts_scores_and_logs(fresh_synthetic_segments: List[Segment], tmp_path: Path) -> None:
    src = FakeSource(fresh_synthetic_segments)
    landed: List[Any] = []
    pipe = ChannelPipeline(src, CFG, tmp_path / "out", session="s", on_clip=landed.append)
    res = pipe.run()
    assert res.ok and res.segments == len(fresh_synthetic_segments)
    assert (res.windows, res.clips, res.scored, res.errors, res.dropped) == (2, 2, 2, 0, 0)
    _windows_match_bursts(pipe.windows)
    assert len(landed) == 2

    clips = pipe.sorted_clips
    for clip in clips:
        assert clip.path.is_file()
        side = read_sidecar(clip.path)
        assert side.scores["frames"] > 0
        assert "visual_mean" in side.scores and "audio_prominence" in side.scores
        assert side.content_start <= side.content_end <= side.file_duration + 1e-6

    kinds = [e["type"] for e in _events(pipe)]
    assert kinds[0] == "session_start" and kinds[-1] == "session_end"
    for k, n in (("segment", 30), ("audio", 30), ("window", 2), ("clip", 2), ("score", 2)):
        assert kinds.count(k) == n, (k, kinds.count(k))
    # Event order per window: window, then clip, then score.
    for w in pipe.windows:
        order = [e["type"] for e in _events(pipe) if e.get("window_id") == w.window_id]
        assert order == ["window", "clip", "score"]
    score_ev = _events(pipe, "score")
    assert all("rank_score" in e and "rank_percentile" in e for e in score_ev)

    lat = pipe.log.latency.summary()
    assert lat["cut_latency"]["count"] == 2 and lat["ingest_lag"]["count"] == 30
    assert not any(pipe.buffer_dir.glob("*")) if pipe.buffer_dir.exists() else True
    # The segment files handed to the buffer are gone.
    assert not any(s.path.exists() for s in fresh_synthetic_segments)


class _FailFirstWriter(ClipWriter):
    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.calls = 0
        self._lock = threading.Lock()

    def write(self, window, lease, pre_pad=0.0, post_pad=0.0):  # type: ignore[override]
        with self._lock:
            self.calls += 1
            first = self.calls == 1
        if first:
            raise ClipError("forced failure", "fake stderr")
        return super().write(window, lease, pre_pad, post_pad)


def test_failing_cut_is_logged_and_pipeline_carries_on(
    fresh_synthetic_segments: List[Segment], tmp_path: Path
) -> None:
    writer = _FailFirstWriter(tmp_path / "out", "s")
    pipe = ChannelPipeline(FakeSource(fresh_synthetic_segments), CFG, tmp_path / "out", session="s", writer=writer)
    res = pipe.run()
    assert res.ok and res.windows == 2 and res.clips == 1 and res.errors == 1
    errors = _events(pipe, "error")
    assert len(errors) == 1 and errors[0]["what"] == "clip" and "forced failure" in errors[0]["error"]
    assert errors[0]["window_id"] == pipe.windows[0].window_id
    # The failed window's score job saw no clip and did nothing; the second was scored.
    assert [e["window_id"] for e in _events(pipe, "score")] == [pipe.windows[1].window_id]
    assert _events(pipe, "session_end")


class _SlowScorer(VisualScorer):
    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay

    def score(self, lease, core_start, core_end):  # type: ignore[override]
        time.sleep(self.delay)
        return super().score(lease, core_start, core_end)


def test_slow_scoring_with_one_worker_drops_scores_not_clips(
    fresh_synthetic_segments: List[Segment], tmp_path: Path
) -> None:
    cfg = CFG.replace(workers=1, queue_size=1)
    src = FakeSource(fresh_synthetic_segments, interval=0.01)
    pipe = ChannelPipeline(src, cfg, tmp_path / "out", session="s", scorer=_SlowScorer(1.0))
    res = pipe.run()
    assert res.ok and res.clips == 2
    assert all(c.path.is_file() for c in pipe.sorted_clips)
    dropped = _events(pipe, "dropped")
    assert dropped and all(e["what"] == "score" for e in dropped)
    unscored = [c for c in pipe.sorted_clips if not c.scores]
    assert len(unscored) == len(dropped)
    lag = pipe.log.latency.summary()["ingest_lag"]
    assert lag["count"] == 30 and lag["p95"] < 0.5


def test_detect_only_writes_no_clips(fresh_synthetic_segments: List[Segment], tmp_path: Path) -> None:
    pipe = ChannelPipeline(
        FakeSource(fresh_synthetic_segments), CFG, tmp_path / "out", session="s", detect_only=True
    )
    res = pipe.run()
    assert res.windows == 2 and res.clips == 0
    assert not (pipe.session_dir / "clips").exists()
    assert len(_events(pipe, "audio")) == 30


def test_pipeline_stop_runs_shutdown_sequence(fresh_synthetic_segments: List[Segment], tmp_path: Path) -> None:
    # Released slowly and held open like a live feed: only stop() ends it.
    src = FakeSource(fresh_synthetic_segments, interval=0.05, hold_open=True)
    pipe = ChannelPipeline(src, CFG, tmp_path / "out", session="s").start()
    time.sleep(1.2)  # about 24 segments: the first burst is closed and cut
    pipe.stop()
    assert pipe.join(10)
    res = pipe.result
    assert res.stopped and res.ok and res.clips >= 1
    assert _events(pipe, "session_end")
    assert not pipe.buffer_dir.exists() or not any(pipe.buffer_dir.iterdir())
    yielded = {s.path for s in src.yielded}
    assert not any(p.exists() for p in yielded)


# ------------------------------------------------------------------ Supervisor


class _BrokenSource:
    """Yields a few segments, then fails like a dead network feed."""

    def __init__(self, segments: List[Segment], channel: str, fail_after: int = 3) -> None:
        self.channel = channel
        self.segments = segments
        self.fail_after = fail_after

    def __iter__(self):
        for i, seg in enumerate(self.segments):
            if i == self.fail_after:
                raise ConnectionError("feed died")
            yield seg

    def close(self) -> None:
        pass


def _channel_segments(segments: List[Segment], channel: str, dest: Path) -> List[Segment]:
    import dataclasses

    return [dataclasses.replace(s, channel=channel) for s in copy_segments(segments, dest)]


def test_supervisor_failing_channel_does_not_stop_others(
    synthetic_segments: List[Segment], tmp_path: Path
) -> None:
    good = _channel_segments(synthetic_segments, "good", tmp_path / "g")
    bad = _channel_segments(synthetic_segments, "bad", tmp_path / "b")
    sup = Supervisor(CFG, tmp_path / "out", session="s")
    sup.add_channel(FakeSource(good))
    sup.add_channel(_BrokenSource(bad, "bad"))
    results = sup.run(timeout=60)
    assert results["good"].ok and results["good"].clips == 2
    assert not results["bad"].ok and "feed died" in results["bad"].error
    bad_errors = list(read_events(results["bad"].session_dir, "error"))
    assert bad_errors and bad_errors[0]["what"] == "ingest"
    assert list(read_events(results["bad"].session_dir, "session_end"))


def test_supervisor_ctrl_c_shuts_down_gracefully(synthetic_segments: List[Segment], tmp_path: Path) -> None:
    sup = Supervisor(CFG, tmp_path / "out", session="s")
    for name in ("a", "b"):
        segs = _channel_segments(synthetic_segments, name, tmp_path / name)
        sup.add_channel(FakeSource(segs, interval=0.02, hold_open=True))
    timer = threading.Timer(1.0, _thread.interrupt_main)
    timer.start()
    try:
        results = sup.run(timeout=30)
    finally:
        timer.cancel()
    assert sup.interrupted
    for name in ("a", "b"):
        r = results[name]
        assert r.stopped and r.ok
        assert list(read_events(r.session_dir, "session_end"))


def test_supervisor_add_and_remove_channel_while_running(
    synthetic_segments: List[Segment], tmp_path: Path
) -> None:
    sup = Supervisor(CFG, tmp_path / "out", session="s")
    first = _channel_segments(synthetic_segments, "one", tmp_path / "1")
    sup.add_channel(FakeSource(first, interval=0.01, hold_open=True))
    runner = threading.Thread(target=sup.run, kwargs={"timeout": 30})
    runner.start()
    try:
        time.sleep(0.2)
        second = _channel_segments(synthetic_segments, "two", tmp_path / "2")
        p2 = sup.add_channel(FakeSource(second))
        assert p2.join(30) and p2.result.clips == 2
        r1 = sup.remove_channel("one", timeout=10)
        assert r1 is not None and r1.stopped
        assert sup.channels == ["two"]
    finally:
        sup.stop()
        runner.join(30)
    assert not runner.is_alive()


# ------------------------------------------------------------------ queue sizing and schedule


def test_effective_queue_size_scales_with_channels() -> None:
    from VidFlow.realtime.pipeline import effective_queue_size

    cfg = RealtimeConfig()  # queue_size 64, queue_per_channel 4
    assert effective_queue_size(cfg, 1) == 64  # floor: one channel behaves as before
    assert effective_queue_size(cfg, 16) == 64
    assert effective_queue_size(cfg, 32) == 128
    assert effective_queue_size(cfg, 64) == 256
    assert effective_queue_size(cfg.replace(queue_per_channel=0), 64) == 64
    assert effective_queue_size(cfg.replace(queue_size=8, queue_per_channel=3), 5) == 15


def test_pool_set_queue_size_takes_effect() -> None:
    gate = threading.Event()
    pool = WorkerPool(1, 1)
    try:
        pool.submit(Job("cut", gate.wait))  # occupies the worker
        time.sleep(0.05)
        assert pool.submit(Job("cut", lambda: None))
        assert not pool.submit(Job("cut", lambda: None))  # bound 1: full
        pool.set_queue_size(3)
        assert pool.submit(Job("cut", lambda: None)) and pool.submit(Job("cut", lambda: None))
        with pytest.raises(ValueError):
            pool.set_queue_size(0)
    finally:
        gate.set()
        pool.shutdown()


def test_supervisor_scales_queue_with_channel_count(synthetic_segments: List[Segment], tmp_path: Path) -> None:
    cfg = CFG.replace(queue_size=10, queue_per_channel=3)
    sup = Supervisor(cfg, tmp_path / "out", session="s")
    try:
        assert sup.pool.queue_size == 10
        for i in range(5):
            sup.add_channel(FakeSource(copy_segments(synthetic_segments[:2], tmp_path / f"c{i}"), f"c{i}"))
        assert sup.pool.queue_size == 15
        sup.remove_channel("c4")
        sup.remove_channel("c3")
        assert sup.pool.queue_size == 10
    finally:
        sup.stop()
        sup.pool.shutdown()
    # A shared pool passed in is left alone.
    pool = WorkerPool(1, 7)
    sup = Supervisor(cfg, tmp_path / "out2", session="s", pool=pool)
    for i in range(5):
        sup.add_channel(FakeSource(copy_segments(synthetic_segments[:2], tmp_path / f"d{i}"), f"d{i}"))
    assert pool.queue_size == 7
    sup.stop()
    pool.shutdown()


class _PacedFake(FakeSource):
    """FakeSource that declares itself paced at 1x with a known playback start."""

    speed = 1.0

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.t0: Optional[float] = None

    def __iter__(self):
        self.t0 = time.monotonic()
        return super().__iter__()


class _UnpacedFake(FakeSource):
    speed = None


def test_pipeline_stamps_due_at_and_reports_behind_schedule(
    fresh_synthetic_segments: List[Segment], tmp_path: Path
) -> None:
    # Released instantly but due at t0 + end: the run is "ahead", behind_schedule is negative.
    pipe = ChannelPipeline(_PacedFake(fresh_synthetic_segments[:5]), CFG, tmp_path / "out", session="s",
                           detect_only=True)
    res = pipe.run()
    segs = _events(pipe, "segment")
    assert all(e["due_at"] is not None for e in segs)
    t0 = pipe.source.t0
    for e in segs:
        assert e["due_at"] == pytest.approx(t0 + e["start"] + e["duration"])
    lat = res.latency
    assert lat["behind_schedule"]["count"] == 5 and lat["behind_schedule"]["max"] < 0


def test_pipeline_unpaced_source_has_no_schedule(fresh_synthetic_segments: List[Segment], tmp_path: Path) -> None:
    pipe = ChannelPipeline(_UnpacedFake(fresh_synthetic_segments[:3]), CFG, tmp_path / "out", session="s",
                           detect_only=True)
    res = pipe.run()
    assert all(e["due_at"] is None for e in _events(pipe, "segment"))
    assert res.latency["behind_schedule"]["count"] == 0


def test_supervisor_playing_and_finished(synthetic_segments: List[Segment], tmp_path: Path) -> None:
    sup = Supervisor(CFG, tmp_path / "out", session="s", detect_only=True)
    sup.add_channel(FakeSource(copy_segments(synthetic_segments[:2], tmp_path / "a"), "a"))
    assert sup.playing() == [] and sup.finished() == []
    sup.run()
    assert sup.playing() == ["a"] and sup.finished() == ["a"]
