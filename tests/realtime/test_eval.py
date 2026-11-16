"""Tests for eval / retune / bench (task T9)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from VidFlow.realtime.cli import main
from VidFlow.realtime.config import RealtimeConfig
from VidFlow.realtime.eval import (
    RULE_FIELDS,
    EvalReport,
    RefWindow,
    evaluate,
    format_eval,
    format_retune,
    load_truth,
    match_windows,
    merge_intervals,
    reference_windows,
    retune,
    run_bench,
)
from VidFlow.realtime.pipeline import ChannelPipeline
from VidFlow.realtime.session import read_events
from VidFlow.realtime.sources.replay import ReplaySource
from VidFlow.realtime.testing import SYNTHETIC_BURSTS, detection_scenes, make_scene_video, write_truth
from VidFlow.realtime.types import Window, make_window_id

CFG = RealtimeConfig(warmup_s=5.0)
DEFAULTS = RealtimeConfig()  # warm-up no longer hides the fixture's first burst


def _w(start: float, end: float, channel: str = "c") -> Window:
    return Window(channel, make_window_id(channel, start), start, end, "audio", {})


@pytest.fixture(scope="module")
def fixture_eval(synthetic_video: Path, tmp_path_factory: pytest.TempPathFactory) -> EvalReport:
    """One detect-only run over the fixture with default settings, its log kept for the retune tests."""
    return evaluate(synthetic_video, DEFAULTS, out_dir=tmp_path_factory.mktemp("eval"), session="s")


# ------------------------------------------------------------------ reference and matching


def test_reference_finds_the_fixture_bursts(synthetic_video: Path) -> None:
    ref = reference_windows(synthetic_video)
    assert ref.threshold_db is not None and ref.threshold_db > -10
    assert len(ref.windows) == len(SYNTHETIC_BURSTS)
    for w, (b0, b1) in zip(ref.windows, SYNTHETIC_BURSTS):
        assert abs(w.start - b0) <= 0.1 and abs(w.end - b1) <= 0.1, w
    assert abs(ref.duration - 60.0) < 0.1


def test_batch_upper_edge_threshold_finds_nothing(synthetic_video: Path) -> None:
    """The literal batch rule puts the threshold above every sample of a constant burst."""
    ref = reference_windows(synthetic_video)
    upper_edge = ref.threshold_db + 1.0
    assert reference_windows(synthetic_video, threshold_db=upper_edge).windows == []


def test_merge_intervals_merges_and_filters() -> None:
    got = merge_intervals([(0, 1), (4.5, 6), (20, 20.5), (30, 31), (35.1, 37)], 4.0, 1.5)
    assert got == [RefWindow(0, 6), RefWindow(35.1, 37)]  # 30-31 is 1 s and 35.1 is 4.1 s after it
    assert merge_intervals([(0, 1), (5, 6)], 4.0, 0) == [RefWindow(0, 6)]


def test_match_is_one_to_one_at_the_iou_threshold() -> None:
    refs = [RefWindow(10, 16), RefWindow(40, 46), RefWindow(70, 72)]
    dets = [_w(10, 16.1), _w(11, 15), _w(40, 50), _w(90, 95)]
    m = match_windows(refs, dets, 0.5)
    assert [(x.ref, x.det.start) for x in m] == [(refs[0], 10), (refs[1], 40)]
    assert m[0].end_error == pytest.approx(0.1) and m[1].iou == pytest.approx(0.6)
    assert match_windows(refs, dets, 0.7)[1:] == []  # 40-50 vs 40-46 is 0.6


def test_load_truth_formats(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps([[40, 46], {"start": 10, "end": 16}]))
    (tmp_path / "b.json").write_text(json.dumps({"windows": [[1, 2]]}))
    (tmp_path / "c.txt").write_text("# start end\n10 16\n\n40,46\n")
    assert load_truth(tmp_path / "a.json").windows == [RefWindow(10, 16), RefWindow(40, 46)]
    assert load_truth(tmp_path / "b.json").windows == [RefWindow(1, 2)]
    assert load_truth(tmp_path / "c.txt").windows == [RefWindow(10, 16), RefWindow(40, 46)]
    (tmp_path / "bad.txt").write_text("5 3\n")
    with pytest.raises(ValueError):
        load_truth(tmp_path / "bad.txt")


# ------------------------------------------------------------------ eval


def test_eval_on_fixture_is_perfect(fixture_eval: EvalReport) -> None:
    r = fixture_eval
    assert r.precision == 1.0 and r.recall == 1.0
    d = r.to_dict()
    assert d["mean_abs_start_error"] <= 0.15 and d["mean_abs_end_error"] <= 0.15
    assert all(abs(m.start_error) <= 0.15 and abs(m.end_error) <= 0.15 for m in r.matches)
    assert json.loads(json.dumps(d))["matched"] == 2
    # detect only: logged, nothing cut
    assert r.session_dir is not None and not list((r.session_dir / "clips").glob("*.mp4"))
    assert len(list(read_events(r.session_dir, "window"))) == 2


def test_eval_cli_with_truth_file(synthetic_video: Path, tmp_path: Path) -> None:
    truth = tmp_path / "truth.txt"
    truth.write_text("10 16\n40 46\n25 28\n")  # one window the detector cannot find
    r = CliRunner().invoke(
        main, ["eval", str(synthetic_video), "--truth", str(truth), "--json", "-"]
    )
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert d["precision"] == 1.0 and d["recall"] == pytest.approx(2 / 3)
    assert d["reference"]["method"] == "truth"


def test_eval_on_a_loud_bed_scene(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Through the real pipeline (AAC, 2 s segments): a -20 dBFS noise bed with louder
    peaks, one of them inside the warm-up, default settings."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg must be on PATH")
    scene = detection_scenes()["loud_bed"]
    d = tmp_path_factory.mktemp("scene")
    video = make_scene_video(d / "loud_bed.mp4", scene)
    r = evaluate(video, DEFAULTS, truth=write_truth(d / "truth.json", scene.truth))
    assert r.precision == 1.0 and r.recall == 1.0, format_eval(r)
    assert all(abs(m.start_error) <= 0.2 and abs(m.end_error) <= 0.2 for m in r.matches), format_eval(r)


# ------------------------------------------------------------------ retune


def test_retune_reproduces_logged_windows(fixture_eval: EvalReport) -> None:
    res = retune(fixture_eval.session_dir)
    assert res.identical and len(res.windows) == 2 and not res.changed
    assert res.samples == 600


def test_retune_merges_bursts_with_a_long_merge_gap(fixture_eval: EvalReport) -> None:
    gap = SYNTHETIC_BURSTS[1][0] - SYNTHETIC_BURSTS[0][1]
    res = retune(fixture_eval.session_dir / "events.jsonl", merge_gap_s=gap + 1)
    assert not res.identical and len(res.windows) == 1
    w = res.windows[0]
    assert abs(w.start - SYNTHETIC_BURSTS[0][0]) <= 0.2 and abs(w.end - SYNTHETIC_BURSTS[1][1]) <= 0.2
    assert res.changed == {"merge_gap_s": (4.0, gap + 1)}
    with pytest.raises(ValueError, match="hop_s"):
        retune(fixture_eval.session_dir, hop_s=0.05)


def test_retune_reproduces_a_run_with_a_discontinuity(synthetic_video: Path, tmp_path: Path) -> None:
    pipe = ChannelPipeline(
        lambda d: ReplaySource(synthetic_video, "gap", d, speed=None, drop={6}),
        CFG, tmp_path, channel="gap", session="s", detect_only=True,
    )
    assert pipe.run().ok
    res = retune(pipe.session_dir)
    assert res.discontinuities == 1 and res.identical
    assert len(res.windows) == 3 and res.windows[0].forced


def test_retune_of_a_log_from_the_older_rule(fixture_eval: EvalReport, tmp_path: Path) -> None:
    """A log whose config predates rise_db/window_weight still loads; the result says the rule changed."""
    lines = (fixture_eval.session_dir / "events.jsonl").read_text().splitlines()
    out = []
    for line in lines:
        e = json.loads(line)
        if e.get("type") == "session_start":
            for k in RULE_FIELDS:
                e["config"].pop(k, None)
        out.append(json.dumps(e))
    old = tmp_path / "events.jsonl"
    old.write_text("\n".join(out) + "\n")
    res = retune(old)
    assert res.changed == {"rise_db": (None, DEFAULTS.rise_db), "window_weight": (None, DEFAULTS.window_weight)}
    assert len(res.windows) == 2
    assert any("predates the current detection rule" in line for line in format_retune(res))


def test_retune_cli(fixture_eval: EvalReport) -> None:
    runner = CliRunner()
    r = runner.invoke(main, ["retune", str(fixture_eval.session_dir / "events.jsonl")])
    assert r.exit_code == 0, r.output
    assert "identical to the 2 logged windows" in r.output
    r = runner.invoke(main, ["retune", str(fixture_eval.session_dir), "--set", "merge_gap_s=30", "--json", "-"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert len(d["windows"]) == 1 and d["identical_to_logged"] is False
    r = runner.invoke(main, ["retune", str(fixture_eval.session_dir), "--set", "hop_s=0.2"])
    assert r.exit_code == 2 and "hop_s" in r.output


# ------------------------------------------------------------------ bench


def test_bench_smoke_fast_paced(synthetic_video: Path) -> None:
    """Two channels at 20x for 1.5 s: the plumbing, not the numbers (the 1x run is slow)."""
    rep = run_bench(synthetic_video, 2, 1.5, CFG, speed=20.0)
    assert not rep.failed and rep.channels == 2
    assert rep.media_s > 10 and rep.cpu_s > 0 and rep.steady_wall_s > 0
    assert rep.latency["ingest_lag"]["count"] > 0 and rep.behind["count"] > 0
    d = rep.to_dict()
    assert {"cores_used", "cpu_percent", "clips_per_minute", "keeping_up"} <= set(d)


@pytest.mark.slow
def test_bench_cli_four_channels_twenty_seconds(synthetic_video: Path, tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    r = CliRunner().invoke(
        main,
        ["bench", str(synthetic_video), "--channels", "4", "--seconds", "20", "--set", "warmup_s=5", "--json", str(out)],
    )
    assert r.exit_code == 0, r.output
    for metric in ("cpu", "mem", "clips per minute", "detect_latency", "cut_latency", "total_latency",
                   "ingest_lag", "behind_schedule", "keeping up: yes"):
        assert metric in r.output, metric
    d = json.loads(out.read_text())
    assert d["channels"] == 4 and d["clips"] == 4 and d["keeping_up"]
    assert d["latency"]["total_latency"]["p95"] <= CFG.merge_gap_s + CFG.segment_time_s + 2.0
