"""Tests for SessionLog, read_events and LatencyTracker (task T7)."""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import List

import numpy as np
import pytest

from VidFlow.realtime.config import RealtimeConfig
from VidFlow.realtime.session import (
    EVENTS_FILE,
    LATENCY_METRICS,
    LatencyTracker,
    ScheduleClock,
    SessionLog,
    read_config,
    read_events,
    read_loudness,
    samples_from_event,
    window_from_event,
)
from VidFlow.realtime.testing import make_segments
from VidFlow.realtime.types import Clip, LoudnessSample, Window, make_window_id


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _window(start: float, end: float, channel: str = "ch") -> Window:
    return Window(
        channel=channel,
        window_id=make_window_id(channel, start),
        start=start,
        end=end,
        detector="audio",
        stats={"peak_db": np.float32(-4.5), "mean_db": -6.0, "threshold_db": -20.0, "prominence_db": 30.0},
    )


def _lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


# ------------------------------------------------------------------ writing


def test_emit_writes_one_object_per_line(tmp_path: Path) -> None:
    clock = FakeClock(5.0)
    with SessionLog(tmp_path / "s", clock=clock) as slog:
        ev = slog.emit("segment", 2.0, seq=1)
        clock.t = 6.0
        slog.emit("custom", 3.0, t_wall=4.5, note="x")
    assert ev == {"type": "segment", "t_media": 2.0, "t_wall": 5.0, "seq": 1}
    events = list(read_events(tmp_path / "s"))
    assert events[0] == ev
    assert events[1] == {"type": "custom", "t_media": 3.0, "t_wall": 4.5, "note": "x"}
    assert list(json.loads(_lines(tmp_path / "s" / EVENTS_FILE)[0]).keys())[:3] == ["type", "t_media", "t_wall"]


def test_default_clock_is_monotonic_and_epoch_recorded(tmp_path: Path) -> None:
    import time

    before = time.monotonic()
    with SessionLog(tmp_path) as slog:
        ev = slog.log_session_start(RealtimeConfig(), channel="ch", session="s1")
    assert ev is not None and before <= ev["t_wall"] <= time.monotonic()
    assert abs(ev["t_epoch"] - time.time()) < 60


def test_reserved_field_names_rejected(tmp_path: Path) -> None:
    with SessionLog(tmp_path) as slog:
        with pytest.raises(ValueError):
            slog.emit("x", 0.0, **{"t_media": 1.0})


def test_paths_numpy_and_nonfinite_serialise(tmp_path: Path) -> None:
    with SessionLog(tmp_path) as slog:
        slog.emit(
            "misc",
            np.float64(1.5),
            path=tmp_path / "a b.mp4",
            i=np.int64(7),
            f=np.float32(0.25),
            b=np.bool_(True),
            arr=np.arange(3),
            nested={"p": Path("x") / "y", 3: np.float16(2.0)},
            bad=float("nan"),
            ninf=-math.inf,
            s={2, 1},
        )
    line = _lines(tmp_path / EVENTS_FILE)[0]
    json.loads(line, parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))
    (ev,) = read_events(tmp_path)
    assert ev["t_media"] == 1.5
    assert ev["path"] == str(tmp_path / "a b.mp4")
    assert ev["i"] == 7 and ev["f"] == 0.25 and ev["b"] is True
    assert ev["arr"] == [0, 1, 2]
    assert ev["nested"] == {"p": str(Path("x") / "y"), "3": 2.0}
    assert ev["bad"] is None and ev["ninf"] is None
    assert ev["s"] == [1, 2]


def test_emit_after_close_is_dropped(tmp_path: Path) -> None:
    slog = SessionLog(tmp_path)
    slog.emit("a", 0.0)
    slog.close()
    slog.close()  # idempotent
    assert slog.emit("b", 0.0) is None
    assert [e["type"] for e in read_events(tmp_path)] == ["a"]


def test_reopen_appends(tmp_path: Path) -> None:
    with SessionLog(tmp_path) as slog:
        slog.emit("a", 0.0)
    with SessionLog(tmp_path) as slog:
        slog.emit("b", 0.0)
    assert [e["type"] for e in read_events(tmp_path)] == ["a", "b"]


def test_eight_threads_thousand_events_each(tmp_path: Path) -> None:
    slog = SessionLog(tmp_path)
    n_threads, n_events = 8, 1000
    barrier = threading.Barrier(n_threads)
    errors: List[BaseException] = []

    def worker(k: int) -> None:
        try:
            barrier.wait()
            for i in range(n_events):
                slog.emit("tick", float(i), thread=k, i=i, payload="x" * (i % 300))
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    slog.close()
    assert not errors
    lines = _lines(tmp_path / EVENTS_FILE)
    assert len(lines) == n_threads * n_events
    seen = set()
    for line in lines:
        ev = json.loads(line)
        assert ev["type"] == "tick" and ev["payload"] == "x" * (ev["i"] % 300)
        seen.add((ev["thread"], ev["i"]))
    assert len(seen) == n_threads * n_events
    assert slog.counts() == {"tick": n_threads * n_events}


# ------------------------------------------------------------------ reading


def test_read_events_type_filter(tmp_path: Path) -> None:
    with SessionLog(tmp_path) as slog:
        for kind in ("a", "b", "a", "c"):
            slog.emit(kind, 0.0)
    assert [e["type"] for e in read_events(tmp_path, "a")] == ["a", "a"]
    assert [e["type"] for e in read_events(tmp_path / EVENTS_FILE, type=("b", "c"))] == ["b", "c"]


def test_reader_skips_incomplete_trailing_line(tmp_path: Path) -> None:
    with SessionLog(tmp_path) as slog:
        slog.emit("a", 0.0)
    with open(tmp_path / EVENTS_FILE, "ab") as fh:
        fh.write(b'{"type":"b","t_media":0.0,"t_wa')  # a write in progress
    assert [e["type"] for e in read_events(tmp_path)] == ["a"]


def test_reader_skips_corrupt_complete_line(tmp_path: Path) -> None:
    with open(tmp_path / EVENTS_FILE, "wb") as fh:
        fh.write(b'{"type":"a","t_media":0,"t_wall":0}\nnot json\n{"type":"b","t_media":0,"t_wall":0}\n')
    assert [e["type"] for e in read_events(tmp_path)] == ["a", "b"]


def test_reader_during_writes_sees_only_complete_lines(tmp_path: Path) -> None:
    slog = SessionLog(tmp_path)
    stop = threading.Event()
    big = "y" * 20000  # larger than the file buffer, so lines land in several writes

    n_total = 400

    def writer() -> None:
        for i in range(n_total):
            if stop.is_set():
                return
            slog.emit("big", float(i), i=i, payload=big, tail=True)

    t = threading.Thread(target=writer)
    t.start()
    try:
        passes = 0
        while t.is_alive() and passes < 500:
            events = list(read_events(tmp_path))
            assert all(ev["payload"] == big and ev["tail"] is True for ev in events)
            seen = [ev["i"] for ev in events]
            assert seen == list(range(len(seen)))  # complete lines only, none skipped
            passes += 1
        assert passes > 0
    finally:
        stop.set()
        t.join()
        slog.close()
    assert len(list(read_events(tmp_path))) == slog.counts()["big"]


# ------------------------------------------------------------------ helpers / round trips


def test_audio_round_trip(tmp_path: Path) -> None:
    seg_a, seg_b = make_segments("ch", [2.0, 2.0], discontinuities=[1])
    samples_a = [LoudnessSample(t=seg_a.start + i * 0.1, db=-40.0 + i * 0.37) for i in range(20)]
    samples_b = [
        LoudnessSample(t=seg_b.start + i * 0.1, db=(float("-inf") if i == 3 else float(np.float32(-5.5))))
        for i in range(20)
    ]
    with SessionLog(tmp_path) as slog:
        slog.log_audio(seg_a, samples_a, hop_s=0.1)
        slog.log_audio(seg_b, samples_b, hop_s=0.1)
        slog.log_audio(seg_b, [], hop_s=0.1)
    events = list(read_events(tmp_path, "audio"))
    assert [e["seq"] for e in events] == [0, 1, 1]
    assert [e["discontinuity"] for e in events] == [False, True, True]
    assert events[0]["t_media"] == 0.0 and events[1]["t_media"] == 2.0
    assert samples_from_event(events[0]) == samples_a
    assert samples_from_event(events[1]) == samples_b  # -inf survives as null
    assert samples_from_event(events[2]) == []
    assert read_loudness(tmp_path) == samples_a + samples_b
    assert all(isinstance(s, LoudnessSample) for s in read_loudness(tmp_path))


def test_config_and_window_round_trip(tmp_path: Path) -> None:
    cfg = RealtimeConfig(merge_gap_s=2.5, min_percentile=80.0)
    win = _window(10.0, 16.0)
    with SessionLog(tmp_path) as slog:
        slog.log_session_start(cfg, channel="ch", session="sess")
        slog.log_window(win)
    assert read_config(tmp_path) == cfg
    (ev,) = read_events(tmp_path, "window")
    assert ev["window_id"] == "ch-000010000" and ev["t_media"] == 16.0
    back = window_from_event(ev)
    assert back == Window(**{**win.__dict__, "stats": {k: float(v) for k, v in win.stats.items()}})


def test_helper_event_fields(tmp_path: Path) -> None:
    clock = FakeClock(100.0)
    (seg,) = make_segments("ch", [2.0], arrived_at=0.0)
    win = _window(10.0, 16.0)
    clip = Clip(window=win, path=tmp_path / "clips" / "ch_10000.mp4", file_start=4.0,
                file_duration=16.0, content_start=1.0, content_end=15.0)
    with SessionLog(tmp_path, clock=clock) as slog:
        slog.log_segment(seg)
        slog.log_clip(clip, cut_s=0.3)
        slog.log_score(win, {"visual_mean": np.float64(1.25), "frames": np.int64(6)})
        slog.log_dropped("score", "queue full", window=win)
        slog.log_error("cut", RuntimeError("boom"), window=win)
        end = slog.log_session_end(clips=1)
    s, c, sc, d, e, se = read_events(tmp_path)
    assert s["arrived_at"] is None and s["seq"] == 0 and s["discontinuity"] is False
    assert c["path"] == str(clip.path) and c["cut_s"] == 0.3 and c["content_end"] == 15.0
    assert c["window_id"] == win.window_id and c["t_media"] == 16.0
    assert sc["scores"] == {"visual_mean": 1.25, "frames": 6}
    assert d["what"] == "score" and d["reason"] == "queue full" and d["window_id"] == win.window_id
    assert e["error"] == "boom" and e["error_type"] == "RuntimeError"
    assert se == end and se["t_media"] == 16.0 and se["clips"] == 1
    assert se["counts"] == {"segment": 1, "clip": 1, "score": 1, "dropped": 1, "error": 1}
    assert set(se["latency"]) == set(LATENCY_METRICS)


# ------------------------------------------------------------------ latency


def test_latency_tracker_scripted(tmp_path: Path) -> None:
    clock = FakeClock()
    # Five 2 s segments, arriving at 102, 104, ... (the stream started at 100).
    segs = make_segments("ch", [2.0] * 5)
    lags = [0.1, 0.2, 0.05, 0.4, 0.3]
    w1 = _window(1.0, 4.0)  # end on a boundary -> segment 1 [2, 4], arrived 104
    w2 = _window(5.0, 7.5)  # inside segment 3 [6, 8], arrived 108
    w3 = _window(8.2, 9.0)  # inside segment 4 [8, 10], arrived 110; never cut
    with SessionLog(tmp_path, clock=clock) as slog:
        for seg, lag in zip(segs, lags):
            arrived = 100.0 + seg.end
            seg = seg.__class__(**{**seg.__dict__, "arrived_at": arrived})
            if seg.seq == 3:
                clock.t = 108.0 + 0.04
                slog.log_window(w1)  # closed at 108.04 -> detect 4.04
            if seg.seq == 4:
                # Window logged before its segment's event: order must not matter.
                slog.log_window(w2, closed_at=110.5)  # detect 2.5
                clock.t = 110.9
                slog.log_window(w3)  # detect 0.9
            clock.t = arrived + lag
            slog.log_segment(seg)
        clock.t = 108.5
        slog.log_clip(Clip(w1, Path("a.mp4"), 0, 0, 0, 0))  # cut 0.46
        slog.log_clip(Clip(w2, Path("b.mp4"), 0, 0, 0, 0), done_at=111.0)  # cut 0.5
        slog.log_session_end()

    rows = {r.window_id: r for r in slog.latency.per_clip()}
    assert rows[w1.window_id].detect_latency == pytest.approx(4.04)
    assert rows[w2.window_id].detect_latency == pytest.approx(2.5)
    assert rows[w3.window_id].detect_latency == pytest.approx(0.9)
    assert rows[w1.window_id].cut_latency == pytest.approx(0.46)
    assert rows[w2.window_id].cut_latency == pytest.approx(0.5)
    assert rows[w3.window_id].cut_latency is None and rows[w3.window_id].total_latency is None
    assert [r.ingest_lag for r in slog.latency.per_segment()] == pytest.approx(lags)

    s = slog.latency.summary()
    det = [4.04, 2.5, 0.9]
    assert s["detect_latency"]["count"] == 3
    assert s["detect_latency"]["p50"] == pytest.approx(2.5)
    # Linear interpolation: sorted [0.9, 2.5, 4.04], rank 0.95 * 2 = 1.9.
    assert s["detect_latency"]["p95"] == pytest.approx(2.5 + 0.9 * (4.04 - 2.5))
    assert s["detect_latency"]["p95"] == pytest.approx(float(np.percentile(det, 95)))
    assert s["cut_latency"]["count"] == 2
    assert s["cut_latency"]["p50"] == pytest.approx(0.48)
    assert s["cut_latency"]["p95"] == pytest.approx(0.46 + 0.95 * 0.04)
    assert s["total_latency"]["count"] == 2
    assert s["total_latency"]["p50"] == pytest.approx((4.5 + 3.0) / 2)
    # Lags sorted [0.05, 0.1, 0.2, 0.3, 0.4]; p50 = 0.2; p95 rank 3.8 -> 0.3 + 0.8 * 0.1.
    assert s["ingest_lag"]["count"] == 5
    assert s["ingest_lag"]["p50"] == pytest.approx(0.2)
    assert s["ingest_lag"]["p95"] == pytest.approx(0.38)
    assert s["ingest_lag"]["max"] == pytest.approx(0.4)

    # The same numbers come back from the file alone, and from session_end.
    assert LatencyTracker.from_log(tmp_path).summary() == s
    (end,) = read_events(tmp_path, "session_end")
    assert end["latency"] == s


def test_latency_tracker_empty_and_missing_inputs() -> None:
    tr = LatencyTracker()
    s = tr.summary()
    assert s["detect_latency"] == {"count": 0, "p50": None, "p95": None, "mean": None, "max": None}
    # A window with no segment events and a segment with no arrival stamp.
    tr.feed({"type": "window", "t_media": 5.0, "t_wall": 1.0, "window_id": "w", "end": 5.0})
    tr.feed({"type": "segment", "t_media": 0.0, "t_wall": 1.0, "seq": 0, "start": 0.0,
             "duration": 2.0, "arrived_at": None})
    tr.feed({"type": "audio", "t_media": 0.0, "t_wall": 1.0})  # ignored
    assert tr.per_clip()[0].detect_latency is None
    assert tr.per_segment()[0].ingest_lag is None
    assert tr.summary()["ingest_lag"]["count"] == 0


# ------------------------------------------------------------------ behind schedule


def test_schedule_clock_uses_t0_and_most_punctual_arrival() -> None:
    clock = ScheduleClock(1.0)
    # t0 known (a replay): due = t0 + end, even if the segment was pulled late.
    assert clock.due_at(2.0, 103.0, t0=100.0) == pytest.approx(102.0)
    assert clock.due_at(4.0, 109.0, t0=100.0) == pytest.approx(104.0)
    assert clock.due_at(6.0, 105.0, t0=100.0) == pytest.approx(106.0)  # t0 is exact: early is early
    # After a discontinuity t0 no longer holds (dropped content); re-anchor, then most punctual wins.
    assert clock.due_at(8.0, 120.0, discontinuity=True, t0=100.0) == pytest.approx(120.0)
    assert clock.due_at(10.0, 121.0, t0=100.0) == pytest.approx(121.0)
    assert clock.due_at(12.0, 125.0, t0=100.0) == pytest.approx(123.0)
    # No t0: the first arrival anchors the schedule, a more punctual one moves it earlier.
    clock = ScheduleClock(1.0)
    assert clock.due_at(2.0, 52.5) == pytest.approx(52.5)
    assert clock.due_at(4.0, 54.0) == pytest.approx(54.0)  # anchor 50.0 now
    assert clock.due_at(6.0, 60.0) == pytest.approx(56.0)  # pulled 4 s late
    # A discontinuity re-anchors (lost content: wall time ran on, media time did not).
    assert clock.due_at(8.0, 80.0, discontinuity=True) == pytest.approx(80.0)
    assert clock.due_at(10.0, 82.0) == pytest.approx(82.0)


def test_schedule_clock_speed_and_unpaced() -> None:
    clock = ScheduleClock(4.0)
    assert clock.due_at(8.0, 12.5, t0=10.0) == pytest.approx(12.0)
    assert ScheduleClock(None).due_at(2.0, 1.0) is None
    assert ScheduleClock(1.0).due_at(2.0, None) is None  # no arrival stamp: unknown
    with pytest.raises(ValueError):
        ScheduleClock(0.0)


def test_latency_tracker_behind_schedule_and_time_to_clip(tmp_path: Path) -> None:
    """A loop that falls behind: ingest_lag stays 0 (pull source), behind_schedule grows."""
    clock = FakeClock(0.0)
    sched = ScheduleClock(1.0)
    segs = make_segments("ch", [2.0] * 5)
    w = _window(5.0, 7.5)
    with SessionLog(tmp_path, clock=clock) as slog:
        for seg in segs:
            # Released at 100 + end, but pulled 1 s later each segment; processed instantly.
            arrived = 100.0 + seg.end + seg.seq
            seg = seg.__class__(**{**seg.__dict__, "arrived_at": arrived})
            clock.t = arrived
            due = sched.due_at(seg.end, arrived, t0=100.0)
            slog.log_segment(seg, due_at=due)
        clock.t = 112.0
        slog.log_window(w)
        clock.t = 112.5
        slog.log_clip(Clip(w, Path("a.mp4"), 0, 0, 0, 0))
        slog.log_session_end()
    rows = slog.latency.per_segment()
    assert [r.ingest_lag for r in rows] == pytest.approx([0, 0, 0, 0, 0])
    assert [r.behind_schedule for r in rows] == pytest.approx([0, 1, 2, 3, 4])
    s = slog.latency.summary()
    assert set(s) == set(LATENCY_METRICS)
    assert s["behind_schedule"]["max"] == pytest.approx(4.0)
    # Window end 7.5 lies in segment 3 (6-8], due 108, clip done 112.5.
    (clip,) = slog.latency.per_clip()
    assert clip.time_to_clip == pytest.approx(4.5)
    assert clip.total_latency == pytest.approx(112.5 - 111.0)
    assert LatencyTracker.from_log(tmp_path).summary() == s


def test_latency_tracker_without_due_at_reports_no_schedule() -> None:
    tr = LatencyTracker()
    tr.feed({"type": "segment", "t_wall": 3.0, "seq": 0, "start": 0.0, "duration": 2.0, "arrived_at": 2.0})
    assert tr.per_segment()[0].behind_schedule is None
    s = tr.summary()
    assert s["behind_schedule"]["count"] == 0 and s["time_to_clip"]["count"] == 0
    assert s["ingest_lag"]["count"] == 1
