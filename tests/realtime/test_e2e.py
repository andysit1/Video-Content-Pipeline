"""End-to-end tests: the fixture through ReplaySource, the full pipeline and the CLI (task T8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import pytest
from click.testing import CliRunner

from VidFlow.realtime.cli import channel_names, main, parse_overrides, parse_speed
from VidFlow.realtime.clip_writer import read_sidecar
from VidFlow.realtime.config import RealtimeConfig
from VidFlow.realtime.media import probe
from VidFlow.realtime.pipeline import ChannelPipeline
from VidFlow.realtime.session import read_events
from VidFlow.realtime.sources.replay import ReplaySource
from VidFlow.realtime.supervisor import Supervisor
from VidFlow.realtime.testing import SYNTHETIC_BURSTS
from VidFlow.realtime.types import Window

CFG = RealtimeConfig(warmup_s=5.0)


def _run(video: Path, out: Path, speed=None, drop=(), channel: str = "fx", session: str = "s") -> ChannelPipeline:
    pipe = ChannelPipeline(
        lambda d: ReplaySource(video, channel, d, speed=speed, drop=drop),
        CFG,
        out,
        channel=channel,
        session=session,
    )
    res = pipe.run()
    assert res.ok, res.error
    return pipe


def _key(windows: List[Window]) -> List[Tuple]:
    return [(w.window_id, w.start, w.end, w.forced, tuple(sorted(w.stats.items()))) for w in windows]


def _assert_buffer_clean(pipe: ChannelPipeline) -> None:
    if pipe.buffer_dir.exists():
        assert list(pipe.buffer_dir.iterdir()) == []


def test_fixture_unpaced_gives_the_two_bursts(synthetic_video: Path, tmp_path: Path) -> None:
    pipe = _run(synthetic_video, tmp_path / "out")
    clips = pipe.sorted_clips
    assert len(clips) == 2 and len(pipe.windows) == 2
    for clip, (b0, b1) in zip(clips, SYNTHETIC_BURSTS):
        w = clip.window
        assert abs(w.start - b0) <= 0.5 and abs(w.end - b1) <= 0.5, w
        info = probe(clip.path)
        assert info.has_video and info.has_audio
        assert abs(info.duration - clip.file_duration) < 0.1
        side = read_sidecar(clip.path)
        assert side.window == w
        assert side.scores["frames"] > 0 and side.scores["visual_mean"] > 0
        assert side.scores["audio_prominence"] > 20
        raw = json.loads(clip.path.with_suffix(".json").read_text())
        assert 0 <= raw["rank_percentile"] <= 100 and "rank_score" in raw
        # The padded window lies inside the file.
        assert side.content_start <= w.start - side.file_start + 1e-6
        assert side.content_end >= min(w.end + CFG.post_pad_s - side.file_start, side.file_duration) - 1e-6
    _assert_buffer_clean(pipe)
    clips_dir = pipe.session_dir / "clips"
    assert sorted(p.suffix for p in clips_dir.iterdir()) == [".json", ".json", ".mp4", ".mp4"]


def test_paced_run_gives_identical_windows(synthetic_video: Path, tmp_path: Path) -> None:
    """Principle 2 at a fast pace (the speed=10 variant is marked slow)."""
    unpaced = _run(synthetic_video, tmp_path / "a")
    paced = _run(synthetic_video, tmp_path / "b", speed=30)
    assert _key(paced.windows) == _key(unpaced.windows)


@pytest.mark.slow
def test_speed_10_gives_identical_windows(synthetic_video: Path, tmp_path: Path) -> None:
    unpaced = _run(synthetic_video, tmp_path / "a")
    paced = _run(synthetic_video, tmp_path / "b", speed=10)
    assert _key(paced.windows) == _key(unpaced.windows)
    assert len(paced.sorted_clips) == 2


def test_dropped_segment_inside_burst_gives_clips_without_the_gap(synthetic_video: Path, tmp_path: Path) -> None:
    # Segment 6 is source time 12-14 s, inside the first burst (10-16 s).
    pipe = _run(synthetic_video, tmp_path / "out", drop={6})
    segs = list(read_events(pipe.session_dir, "segment"))
    seqs = [e["seq"] for e in segs]
    assert 6 not in seqs
    assert next(e for e in segs if e["seq"] == 7)["discontinuity"] is True
    first = [w for w in pipe.windows if w.start < 20]
    assert len(first) == 2 and first[0].forced
    clips = pipe.sorted_clips
    assert len(clips) == 3
    for clip in clips:
        used = json.loads(clip.path.with_suffix(".json").read_text())["segments"]
        assert not (min(used) <= 5 and max(used) >= 7), f"{clip.path.name} spans the gap: {used}"
        info = probe(clip.path)
        assert abs(info.duration - clip.file_duration) < 0.1
    _assert_buffer_clean(pipe)


def test_two_channels_at_once(synthetic_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    sup = Supervisor(CFG, out, session="s")
    for name in ("left", "right"):
        sup.add_channel(lambda d, name=name: ReplaySource(synthetic_video, name, d, speed=None), channel=name)
    results = sup.run(timeout=120)
    assert set(results) == {"left", "right"}
    dirs = set()
    for name, r in results.items():
        assert r.ok and r.clips == 2 and r.scored == 2
        clips = sorted((r.session_dir / "clips").glob("*.mp4"))
        assert len(clips) == 2 and all(p.name.startswith(name + "_") for p in clips)
        assert r.session_dir == out / name / "s"
        dirs.add(r.session_dir)
        assert list(read_events(r.session_dir, "session_end"))
        buf = r.session_dir / "buffer"
        assert not buf.exists() or not any(buf.iterdir())
    assert len(dirs) == 2


@pytest.mark.slow
def test_hls_simulator_through_the_pipeline(synthetic_video: Path, tmp_path: Path) -> None:
    from VidFlow.realtime.sim import hls_sim
    from VidFlow.realtime.sources.hls import HlsSource

    with hls_sim.start(synthetic_video, tmp_path / "hls", speed=10, window=6) as sim:
        assert sim.wait_for_playlist(10)
        pipe = ChannelPipeline(
            lambda d: HlsSource(sim.playlist_path, "hls", d, poll_s=0.05, stall_timeout_s=5),
            CFG,
            tmp_path / "out",
            channel="hls",
            session="s",
        )
        res = pipe.run()
    assert res.ok and res.clips == 2
    for w, (b0, b1) in zip(pipe.windows, SYNTHETIC_BURSTS):
        assert abs(w.start - b0) <= 0.5 and abs(w.end - b1) <= 0.5
    _assert_buffer_clean(pipe)


@pytest.mark.slow
def test_real_time_latency_budget(synthetic_video: Path, tmp_path: Path) -> None:
    pipe = _run(synthetic_video, tmp_path / "out", speed=1.0)
    assert len(pipe.sorted_clips) == 2
    total = pipe.log.latency.summary()["total_latency"]
    budget = CFG.merge_gap_s + CFG.segment_time_s + 2.0
    assert total["count"] == 2 and total["p95"] <= budget, total


# ------------------------------------------------------------------ CLI


def test_cli_helpers() -> None:
    assert parse_speed("max") is None and parse_speed("4") == 4.0 and parse_speed("10x") == 10.0
    cfg = parse_overrides(["warmup_s=5", "workers=2", "min_percentile=50", "visual_mode=sampled"])
    assert (cfg.warmup_s, cfg.workers, cfg.min_percentile, cfg.visual_mode) == (5.0, 2, 50.0, "sampled")
    assert parse_overrides(["min_percentile=off"], cfg).min_percentile is None
    assert channel_names([Path("a/x.mp4"), Path("b/x.mp4"), Path("my video.mkv")]) == ["x", "x-2", "my_video"]


def test_cli_replay(synthetic_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = CliRunner().invoke(
        main,
        ["replay", str(synthetic_video), str(synthetic_video), "--speed", "max", "--out", str(out),
         "--session", "cli", "--set", "warmup_s=5"],
    )
    assert result.exit_code == 0, result.output
    names = channel_names([synthetic_video, synthetic_video])
    assert result.output.count(" clip ") == 4
    assert "total_latency" in result.output
    for name in names:
        assert len(list((out / name / "cli" / "clips").glob("*.mp4"))) == 2


def test_cli_rejects_bad_settings(synthetic_video: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    r = runner.invoke(main, ["replay", str(synthetic_video), "--set", "nope=1", "--out", str(tmp_path)])
    assert r.exit_code == 2 and "unknown setting" in r.output
    r = runner.invoke(main, ["replay", str(synthetic_video), "--set", "retention_s=10", "--out", str(tmp_path)])
    assert r.exit_code == 2 and "retention_s" in r.output
    r = runner.invoke(main, ["replay", str(synthetic_video), "--speed", "fast", "--out", str(tmp_path)])
    assert r.exit_code == 2
