"""Tests for T0: the contract types, config, media helpers and testing fakes."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from VidFlow.realtime import (
    Clip,
    ClipError,
    ConfigError,
    Detector,
    LoudnessSample,
    RealtimeConfig,
    Segment,
    Source,
    Window,
    make_window_id,
)
from VidFlow.realtime.media import MediaError, probe, segment_file
from VidFlow.realtime.testing import (
    SYNTHETIC_BURSTS,
    SYNTHETIC_DURATION,
    FakeSource,
    make_segments,
)

# --------------------------------------------------------------------- package


def test_import_does_not_pull_in_legacy_modules() -> None:
    code = (
        "import sys, VidFlow.realtime, VidFlow.realtime.media, VidFlow.realtime.testing, "
        "VidFlow.realtime.config, VidFlow.realtime.types\n"
        "bad = [m for m in sys.modules if m.split('.')[:2] in "
        "(['VidFlow','pipes'], ['VidFlow','modules'], ['VidFlow','aggregate'], ['VidFlow','cli'])"
        " or m in ('icecream', 'cv2', 'questionary', 'ffmpeg')]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ----------------------------------------------------------------------- types


def test_segment_end_and_frozen() -> None:
    seg = Segment("ch", 3, 6.0, 2.0, Path("x.ts"))
    assert seg.end == 8.0
    assert seg.discontinuity is False and seg.arrived_at == 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        seg.start = 1.0  # type: ignore[misc]


def test_window_id_format() -> None:
    assert make_window_id("chan", 10.0) == "chan-000010000"
    assert make_window_id("chan", 0.0) == "chan-000000000"
    assert make_window_id("a", 12.3456) == "a-000012346"
    w = Window("chan", make_window_id("chan", 10.0), 10.0, 16.0, "audio", {"peak_db": -3.0})
    assert w.forced is False and w.duration == 6.0


def test_clip_scores_default_empty_and_replace() -> None:
    w = Window("c", make_window_id("c", 1.0), 1.0, 3.0, "audio", {})
    clip = Clip(w, Path("c.mp4"), 0.0, 6.0, 1.0, 3.0)
    assert dict(clip.scores) == {}
    scored = dataclasses.replace(clip, scores={"visual_mean": 1.5})
    assert scored.scores["visual_mean"] == 1.5 and clip.scores == {}


def test_loudness_sample() -> None:
    s = LoudnessSample(1.2, -20.0)
    assert (s.t, s.db) == (1.2, -20.0)


def test_clip_error_carries_stderr_tail() -> None:
    err = ClipError("cut failed", "Invalid data found")
    assert err.stderr_tail == "Invalid data found"
    assert "Invalid data found" in str(err)
    assert isinstance(err, Exception)


def test_protocols_are_structural() -> None:
    src = FakeSource(make_segments("c", [2.0]))
    assert isinstance(src, Source)

    class Dummy:
        def feed(self, segment: Segment) -> list:
            return []

        def flush(self) -> list:
            return []

    assert isinstance(Dummy(), Detector)


# ---------------------------------------------------------------------- config


def test_config_defaults_match_design() -> None:
    c = RealtimeConfig()
    assert c.segment_time_s == 2.0
    assert c.retention_s == 120
    assert c.hop_s == 0.1
    assert c.stats_window_s == 600
    assert c.threshold_percentile == 90
    assert c.floor_db == -35
    assert c.warmup_s == 30
    assert c.merge_gap_s == 4.0
    assert c.min_len_s == 1.5
    assert c.max_len_s == 45
    assert (c.pre_pad_s, c.post_pad_s) == (5, 3)
    assert c.visual_mode == "keyframes"
    assert c.sample_fps > 0
    assert c.workers == 4
    assert c.min_percentile is None
    assert c.validate() is c


def test_validate_rejects_small_retention() -> None:
    c = RealtimeConfig()
    needed = c.max_len_s + c.pre_pad_s + c.post_pad_s + c.merge_gap_s + 2 * c.segment_time_s
    assert c.min_retention_s() == pytest.approx(needed)
    RealtimeConfig(retention_s=needed).validate()
    with pytest.raises(ConfigError, match="retention_s"):
        RealtimeConfig(retention_s=needed - 0.5).validate()
    # A longer real segment duration raises the bar.
    with pytest.raises(ConfigError):
        RealtimeConfig(retention_s=needed).validate(segment_duration=6.0)
    assert issubclass(ConfigError, ValueError)


@pytest.mark.parametrize(
    "changes",
    [
        {"hop_s": 0},
        {"threshold_percentile": 120},
        {"visual_mode": "everything"},
        {"workers": 0},
        {"min_len_s": 50, "max_len_s": 45},
        {"min_percentile": -1},
    ],
)
def test_validate_rejects_bad_values(changes: dict) -> None:
    with pytest.raises(ConfigError):
        RealtimeConfig(**changes).validate()


def test_config_replace_and_dict_roundtrip() -> None:
    c = RealtimeConfig().replace(merge_gap_s=2.0)
    assert c.merge_gap_s == 2.0
    assert RealtimeConfig.from_dict(c.to_dict()) == c
    with pytest.raises(TypeError):
        RealtimeConfig().replace(not_a_setting=1)


# --------------------------------------------------------------- make_segments


def test_make_segments_media_time() -> None:
    durs = [2.0, 1.5, 2.5, 0.1, 3.3]
    segs = make_segments("ch", durs, discontinuities={12}, start_seq=10)
    assert [s.seq for s in segs] == [10, 11, 12, 13, 14]
    assert segs[0].start == 0.0
    for prev, nxt in zip(segs, segs[1:]):
        assert nxt.start == prev.end  # exact, by construction
    assert segs[-1].end == pytest.approx(sum(durs))
    assert [s.discontinuity for s in segs] == [False, False, True, False, False]
    assert all(s.channel == "ch" for s in segs)
    # Media time stays contiguous across a discontinuity.
    assert segs[2].start == pytest.approx(3.5)


def test_make_segments_dummy_files(tmp_path: Path) -> None:
    segs = make_segments("ch", [2.0] * 3, directory=tmp_path)
    assert all(s.path.is_file() and s.path.parent == tmp_path for s in segs)
    assert len({s.path for s in segs}) == 3


def test_make_segments_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        make_segments("ch", [2.0, 0.0])
    with pytest.raises(ValueError):
        make_segments("ch", [2.0], discontinuities={5})


# ------------------------------------------------------------------ FakeSource


def test_fake_source_yields_all_and_stamps_arrival() -> None:
    segs = make_segments("ch", [2.0] * 4)
    src = FakeSource(segs)
    before = time.monotonic()
    out = list(src)
    assert [s.seq for s in out] == [0, 1, 2, 3]
    assert all(s.arrived_at >= before for s in out)
    assert src.channel == "ch"


def test_fake_source_close_unblocks_from_other_thread() -> None:
    src = FakeSource(make_segments("ch", [2.0] * 2), hold_open=True)
    got = []
    done = threading.Event()

    def run() -> None:
        for s in src:
            got.append(s)
        done.set()

    th = threading.Thread(target=run, daemon=True)
    th.start()
    time.sleep(0.1)
    assert not done.is_set()  # blocked after the last segment, like a live source
    t0 = time.monotonic()
    src.close()
    assert done.wait(1.0)
    assert time.monotonic() - t0 < 0.2
    assert len(got) == 2


def test_fake_source_close_during_interval() -> None:
    src = FakeSource(make_segments("ch", [2.0] * 5), interval=5.0)
    threading.Timer(0.1, src.close).start()
    t0 = time.monotonic()
    assert list(src) == []
    assert time.monotonic() - t0 < 1.0


# ----------------------------------------------------------------------- media


def test_probe_synthetic_video(synthetic_video: Path) -> None:
    info = probe(synthetic_video)
    assert info.duration == pytest.approx(SYNTHETIC_DURATION, abs=0.1)
    assert info.has_video and info.has_audio
    assert (info.width, info.height) == (320, 180)
    assert info.fps == pytest.approx(30.0)


def test_synthetic_bursts_are_loud(synthetic_video: Path, tmp_path: Path) -> None:
    """The fixture's bursts are much louder than the rest (sanity check with FFmpeg only)."""

    def rms_db(t0: float, t1: float) -> float:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(synthetic_video), "-vn",
             "-af", f"atrim={t0}:{t1},astats=measure_perchannel=none", "-f", "null", "-"],
            capture_output=True, text=True, check=True,
        )
        lines = [ln for ln in proc.stderr.splitlines() if "RMS level dB" in ln]
        return float(lines[-1].rsplit(":", 1)[1])

    (b0, b1) = SYNTHETIC_BURSTS[0]
    loud = rms_db(b0 + 0.5, b1 - 0.5)
    quiet = rms_db(b0 - 4.0, b0 - 0.5)
    assert loud - quiet >= 30
    assert quiet < RealtimeConfig().floor_db


def test_segment_file_contiguous_and_complete(synthetic_video: Path, synthetic_segments: list) -> None:
    total = probe(synthetic_video).duration
    segs = synthetic_segments
    assert len(segs) == pytest.approx(SYNTHETIC_DURATION / 2.0, abs=1)
    assert segs[0].start == 0.0
    for prev, nxt in zip(segs, segs[1:]):
        assert nxt.start == pytest.approx(prev.end, abs=1e-9)
        assert nxt.seq == prev.seq + 1
    assert sum(s.duration for s in segs) == pytest.approx(total, abs=0.1)
    # Keyframes every second and 2 s target: every segment but possibly the
    # first and last is 2 s long (within one frame).
    for s in segs[1:-1]:
        assert s.duration == pytest.approx(2.0, abs=1 / 30 + 1e-6)
    # Each segment is a self-contained, probe-able MPEG-TS file with both streams.
    first = probe(segs[5].path)
    assert first.has_video and first.has_audio
    assert segs[5].path.suffix == ".ts"


def test_segment_file_returns_infos(synthetic_video: Path, tmp_path: Path) -> None:
    infos = segment_file(synthetic_video, tmp_path / "s", segment_time=4.0)
    assert len(infos) == pytest.approx(SYNTHETIC_DURATION / 4.0, abs=1)
    assert [i.index for i in infos] == list(range(len(infos)))
    assert all(i.path.is_file() and i.name == i.path.name for i in infos)
    assert infos[0].start == 0.0
    for prev, nxt in zip(infos, infos[1:]):
        assert nxt.start == pytest.approx(prev.end, abs=1e-9)
    assert sum(i.duration for i in infos) == pytest.approx(SYNTHETIC_DURATION, abs=0.1)


def test_segment_file_transcode(synthetic_video: Path, tmp_path: Path) -> None:
    # Re-encode with a keyframe forced every 3 s (the source has one every 1 s,
    # so 3 s segments show the forced keyframes are honoured).
    infos = segment_file(synthetic_video, tmp_path / "t", segment_time=3.0, transcode=True)
    assert len(infos) == pytest.approx(SYNTHETIC_DURATION / 3.0, abs=1)
    for i in infos[1:-1]:
        assert i.duration == pytest.approx(3.0, abs=1 / 30 + 1e-6)
    assert sum(i.duration for i in infos) == pytest.approx(SYNTHETIC_DURATION, abs=0.1)


def test_media_errors(tmp_path: Path) -> None:
    with pytest.raises(MediaError):
        segment_file(tmp_path / "missing.mp4", tmp_path / "o")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(MediaError):
        probe(bad)
    with pytest.raises(MediaError) as exc:
        segment_file(bad, tmp_path / "o2")
    assert exc.value.stderr_tail


def test_fresh_segments_are_private_copies(synthetic_segments: list, fresh_synthetic_segments: list) -> None:
    assert len(fresh_synthetic_segments) == len(synthetic_segments)
    fresh_synthetic_segments[0].path.unlink()
    assert synthetic_segments[0].path.is_file()
    assert [s.start for s in fresh_synthetic_segments] == [s.start for s in synthetic_segments]
