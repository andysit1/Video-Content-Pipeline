"""Tests for multi-process sharding, staggered replays and the sharded bench (shard.py)."""

from __future__ import annotations

import dataclasses
import os
import threading
import time
from pathlib import Path
from typing import Iterator, List

import pytest

from VidFlow.realtime.config import RealtimeConfig
from VidFlow.realtime.eval import run_bench, stagger_delays
from VidFlow.realtime.session import read_events
from VidFlow.realtime.shard import (
    OffsetReplaySource,
    PacedReplay,
    ReplayFactory,
    ShardedSupervisor,
    make_supervisor,
    plan_shards,
    process_count,
    process_tree_cpu,
)
from VidFlow.realtime.sources.replay import ReplaySource
from VidFlow.realtime.supervisor import Supervisor
from VidFlow.realtime.types import Clip, Segment

CFG = RealtimeConfig(warmup_s=5.0)


# ------------------------------------------------------------------ planning


def test_process_count_and_plan_shards() -> None:
    assert process_count(1) == 1 and process_count(20) == 1 and process_count(21) == 2
    assert process_count(48) == 3 and process_count(64) == 4
    assert process_count(10, channels_per_process=4) == 3
    assert process_count(3, processes=8) == 3  # never more processes than channels
    assert process_count(40, processes=1) == 1
    shards = plan_shards(10, processes=3)
    assert shards == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert sorted(i for s in plan_shards(64) for i in s) == list(range(64))
    assert [len(s) for s in plan_shards(48)] == [16, 16, 16]
    assert plan_shards(0) == []


def test_make_supervisor_picks_single_process_for_few_channels(tmp_path: Path) -> None:
    sup = make_supervisor(CFG, tmp_path, channels=4)
    assert type(sup) is Supervisor
    sup.pool.shutdown()
    sup = make_supervisor(CFG, tmp_path, channels=40, processes=1)
    assert type(sup) is Supervisor
    sup.pool.shutdown()
    sharded = make_supervisor(CFG, tmp_path, channels=40)
    assert isinstance(sharded, ShardedSupervisor) and sharded.processes == 2
    assert isinstance(make_supervisor(CFG, tmp_path, channels=4, processes=2), ShardedSupervisor)
    assert type(make_supervisor(CFG.replace(channels_per_process=2), tmp_path, channels=2)) is Supervisor


def test_stagger_delays_are_spread_and_seeded() -> None:
    d = stagger_delays(8, 16.0, seed=1)
    assert sorted(d) == pytest.approx([0, 2, 4, 6, 8, 10, 12, 14])
    assert d == stagger_delays(8, 16.0, seed=1)
    assert d != stagger_delays(8, 16.0, seed=2)
    assert stagger_delays(4, 0.0) == [0.0] * 4
    assert stagger_delays(0, 5.0) == []


def test_process_tree_cpu_counts_this_process() -> None:
    a = process_tree_cpu()
    sum(i * i for i in range(200000))
    assert process_tree_cpu() > a
    assert process_tree_cpu([os.getpid()]) >= process_tree_cpu()


# ------------------------------------------------------------------ staggered sources


def _drain(src) -> List[Segment]:
    out = list(src)
    return out


def test_offset_replay_source_starts_into_the_file(synthetic_video: Path, tmp_path: Path) -> None:
    src = OffsetReplaySource(synthetic_video, "c", tmp_path, speed=None, offset_s=20.0)
    segs = _drain(src)
    assert segs[0].start == 0.0 and segs[0].seq == 10
    assert sum(s.duration for s in segs) == pytest.approx(40.0, abs=0.1)
    for a, b in zip(segs, segs[1:]):
        assert b.start == pytest.approx(a.end)
    # Only handed-out files remain.
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(s.path.name for s in segs)
    assert _drain(OffsetReplaySource(synthetic_video, "d", tmp_path / "x", speed=None, offset_s=999.0)) == []
    assert not list((tmp_path / "x").glob("*.ts"))


def test_paced_replay_limits_media_and_delays_start(synthetic_video: Path, tmp_path: Path) -> None:
    inner = ReplaySource(synthetic_video, "c", tmp_path, speed=None)
    src = PacedReplay(inner, start_delay_s=0.3, media_limit_s=10.0)
    assert src.channel == "c" and src.t0 is None and src.speed is None
    t = time.monotonic()
    segs = _drain(src)
    assert time.monotonic() - t >= 0.3
    assert len(segs) == 5 and segs[-1].end == pytest.approx(10.0, abs=0.1)
    assert src.t0 is not None
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(s.path.name for s in segs)


def test_paced_replay_close_during_delay_returns_promptly(synthetic_video: Path, tmp_path: Path) -> None:
    src = PacedReplay(ReplaySource(synthetic_video, "c", tmp_path, speed=None), start_delay_s=30.0)
    threading.Timer(0.2, src.close).start()
    t = time.monotonic()
    assert _drain(src) == []
    assert time.monotonic() - t < 5.0
    assert not list(tmp_path.glob("*.ts"))


def test_replay_factory_is_picklable_and_builds_the_right_source(synthetic_video: Path, tmp_path: Path) -> None:
    import pickle

    f = ReplayFactory(synthetic_video, "c", speed=2.0, offset_s=4.0, start_delay_s=1.0, media_limit_s=8.0)
    assert pickle.loads(pickle.dumps(f)) == f
    src = f(tmp_path)
    assert isinstance(src, PacedReplay) and isinstance(src.inner, OffsetReplaySource) and src.speed == 2.0
    assert type(ReplayFactory(synthetic_video, "c")(tmp_path)) is ReplaySource


# ------------------------------------------------------------------ sharded supervisor


@dataclasses.dataclass(frozen=True)
class _DyingFactory:
    """Source factory whose source kills its whole worker process after ``after_s``.

    Before dying it leaves the kind of debris a real crash would: a ``.partial``
    clip and a buffered segment file. Module level, so worker processes can unpickle it.
    """

    channel: str
    after_s: float = 1.0

    def __call__(self, work_dir: Path) -> "_DyingSource":
        return _DyingSource(self.channel, Path(work_dir), self.after_s)


class _DyingSource:
    def __init__(self, channel: str, work_dir: Path, after_s: float) -> None:
        self.channel = channel
        self.work_dir = work_dir
        self.after_s = after_s
        self._closed = threading.Event()

    def __iter__(self) -> Iterator[Segment]:
        clips = self.work_dir.parent / "clips"
        clips.mkdir(parents=True, exist_ok=True)
        (clips / "x.mp4.abc123.partial").write_bytes(b"half")
        (self.work_dir / "seg.ts").write_bytes(b"ts")
        if not self._closed.wait(self.after_s):
            os._exit(7)
        return iter(())

    def close(self) -> None:
        self._closed.set()


def test_sharded_supervisor_matches_single_process(synthetic_video: Path, tmp_path: Path) -> None:
    clips: List[Clip] = []
    scored: List[str] = []
    sup = ShardedSupervisor(
        CFG, tmp_path / "out", session="s", processes=2,
        on_clip=clips.append, on_score=lambda c, extra: scored.append(c.window.window_id),
    )
    names = [f"c{i}" for i in range(3)]
    for n in names:
        sup.add_channel(ReplayFactory(synthetic_video, n, speed=None), channel=n)
    with pytest.raises(ValueError):
        sup.add_channel(ReplayFactory(synthetic_video, "c0", speed=None), channel="c0")
    results = sup.run()
    assert sup.shards == [["c0", "c1"], ["c2"]]
    assert set(results) == set(names) and all(r.ok for r in results.values())
    assert sup.worker_pids == []

    single = Supervisor(CFG, tmp_path / "ref", session="s")
    single.add_channel(ReplayFactory(synthetic_video, "c0", speed=None), channel="c0")
    ref = single.run()["c0"]
    for n in names:
        r = results[n]
        assert (r.segments, r.windows, r.clips, r.scored) == (ref.segments, ref.windows, ref.clips, ref.scored)
        assert r.session_dir == tmp_path / "out" / n / "s"
        kinds = [e["type"] for e in read_events(r.session_dir)]
        assert kinds[0] == "session_start" and kinds[-1] == "session_end"
    # Callbacks were forwarded to this process.
    assert len(clips) == 3 * ref.clips and len(scored) == 3 * ref.scored
    assert all(c.path.is_file() for c in clips)
    assert sorted(sup.playing()) == names and sorted(sup.finished()) == names
    with pytest.raises(RuntimeError):
        sup.run()


def test_sharded_supervisor_timeout_stops_workers(synthetic_video: Path, tmp_path: Path) -> None:
    sup = ShardedSupervisor(CFG, tmp_path / "out", session="s", processes=2)
    for n in ("a", "b"):
        sup.add_channel(ReplayFactory(synthetic_video, n, speed=1.0), channel=n)
    t = time.monotonic()
    results = sup.run(timeout=4.0)
    took = time.monotonic() - t
    assert took < 40.0, took
    assert all(r.ok and r.stopped for r in results.values())
    assert all(0 < r.segments < 30 for r in results.values())
    assert sup.worker_pids == []
    for r in results.values():
        assert not (r.session_dir / "buffer").exists() or not any((r.session_dir / "buffer").iterdir())


def test_dead_worker_fails_its_channels_and_others_carry_on(synthetic_video: Path, tmp_path: Path) -> None:
    sup = ShardedSupervisor(CFG, tmp_path / "out", session="s", processes=2)
    sup.add_channel(_DyingFactory("dies", after_s=1.0), channel="dies")
    sup.add_channel(ReplayFactory(synthetic_video, "lives", speed=10.0), channel="lives")
    results = sup.run(timeout=60.0)
    dead, alive = results["dies"], results["lives"]
    assert not dead.ok and "exited with code 7" in (dead.error or "")
    assert alive.ok and not alive.stopped and alive.segments == 30 and alive.clips >= 1
    # The dead channel's debris is gone and its log is closed.
    assert not list((dead.session_dir / "clips").glob("*.partial"))
    assert not (dead.session_dir / "buffer").exists()
    events = list(read_events(dead.session_dir))
    assert events[-1]["type"] == "session_end" and "exited with code 7" in events[-1]["error"]
    assert any(e["type"] == "error" and e["what"] == "process" for e in events)
    assert sup.worker_pids == []


def test_sharded_stop_before_run_ends_promptly(synthetic_video: Path, tmp_path: Path) -> None:
    sup = ShardedSupervisor(CFG, tmp_path / "out", session="s", processes=2)
    for n in ("a", "b"):
        sup.add_channel(ReplayFactory(synthetic_video, n, speed=1.0), channel=n)
    sup.stop()
    results = sup.run()
    assert all(r.ok and r.segments == 0 for r in results.values())


# ------------------------------------------------------------------ bench


def test_bench_sharded_and_staggered_fast(synthetic_video: Path) -> None:
    rep = run_bench(synthetic_video, 3, 1.5, CFG, speed=20.0, processes=2, stagger_s=0.6, seed=3)
    assert not rep.failed and rep.processes == 2 and [len(s) for s in rep.shards] == [2, 1]
    assert rep.stagger_s == 0.6 and rep.seed == 3
    assert rep.media_s == pytest.approx(30.0, abs=2.1)  # 1.5 s at 20x, every channel
    assert rep.behind["count"] > 0 and rep.latency["ingest_lag"]["count"] > 0
    assert "time_to_clip" in rep.latency and rep.cpu_s > 0 and rep.steady_wall_s > 0
    assert rep.queue_size == CFG.queue_size


def test_bench_synchronized_matches_staggered_clip_counts(synthetic_video: Path) -> None:
    sync = run_bench(synthetic_video, 2, 1.5, CFG, speed=20.0, processes=1, stagger_s=0)
    stag = run_bench(synthetic_video, 2, 1.5, CFG, speed=20.0, processes=1, stagger_s=0.5)
    assert sync.stagger_s == 0 and sync.processes == 1
    assert (sync.windows, sync.clips) == (stag.windows, stag.clips)
    from VidFlow.realtime.eval import format_bench

    text = "\n".join(format_bench(sync))
    assert "starts synchronized" in text and "time_to_clip" in text and "behind_schedule" in text
    assert "staggered over 0.5 s" in "\n".join(format_bench(stag))


@pytest.mark.slow
def test_replay_cli_sharded(synthetic_video: Path, tmp_path: Path) -> None:
    from click.testing import CliRunner

    from VidFlow.realtime.cli import main

    r = CliRunner().invoke(
        main,
        ["replay", str(synthetic_video), str(synthetic_video), "--speed", "max", "--procs", "2",
         "--out", str(tmp_path), "--session", "s", "--set", "warmup_s=5"],
    )
    assert r.exit_code == 0, r.output
    assert "in 2 processes" in r.output and r.output.count("] clip ") >= 2
    assert "behind_schedule" in r.output and "time_to_clip" in r.output
