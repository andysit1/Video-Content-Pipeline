"""Tests for VisualScorer and GameProfile (task T6, design section 6.6)."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

from VidFlow.realtime.config import RealtimeConfig
from VidFlow.realtime.media import run_ffmpeg
from VidFlow.realtime.scoring import (
    DEFAULT_PROFILE,
    SCORE_KEYS,
    GameProfile,
    VisualScorer,
    crop_regions,
    frame_score,
    region_bounds,
)
from VidFlow.realtime.testing import segments_from_file
from VidFlow.realtime.types import Clip, Segment, Window

# ----------------------------------------------------------------- profiles


def test_default_profile_is_150px_centre_box_at_1080p() -> None:
    y0, y1, x0, x1 = region_bounds((1080, 1920), DEFAULT_PROFILE.regions[0])
    assert (x1 - x0, y1 - y0) == (150, 150)
    assert (x0 + x1) / 2 == pytest.approx(960, abs=1)
    assert (y0 + y1) / 2 == pytest.approx(540, abs=1)
    assert DEFAULT_PROFILE.weights == {"binary": 2.0, "canny": 0.5}


def test_crops_cover_same_fraction_at_1080p_and_360p() -> None:
    hd = np.zeros((1080, 1920), np.uint8)
    sd = np.zeros((360, 640), np.uint8)
    (c_hd,) = crop_regions(hd)
    (c_sd,) = crop_regions(sd)
    frac_hd = (c_hd.shape[0] / 1080, c_hd.shape[1] / 1920)
    frac_sd = (c_sd.shape[0] / 360, c_sd.shape[1] / 640)
    # Within one pixel of the smaller frame.
    assert frac_hd[0] == pytest.approx(frac_sd[0], abs=1 / 360)
    assert frac_hd[1] == pytest.approx(frac_sd[1], abs=1 / 640)
    # Same position too.
    b_hd = region_bounds(hd.shape, DEFAULT_PROFILE.regions[0])
    b_sd = region_bounds(sd.shape, DEFAULT_PROFILE.regions[0])
    assert b_hd[0] / 1080 == pytest.approx(b_sd[0] / 360, abs=1 / 360)
    assert b_hd[2] / 1920 == pytest.approx(b_sd[2] / 640, abs=1 / 640)


def test_same_content_scores_alike_at_two_resolutions() -> None:
    def frame(h: int, w: int) -> np.ndarray:
        img = np.zeros((h, w), np.uint8)
        # A white square covering the left half of the centre box.
        y0, y1, x0, x1 = region_bounds((h, w), DEFAULT_PROFILE.regions[0])
        img[y0:y1, x0 : (x0 + x1) // 2] = 255
        return img

    assert frame_score(frame(1080, 1920)) == pytest.approx(frame_score(frame(360, 640)), rel=0.1)


def test_profile_validation() -> None:
    with pytest.raises(ValueError):
        GameProfile("bad", regions=())
    with pytest.raises(ValueError):
        GameProfile("bad", regions=((0.6, 0.0, 0.4, 1.0),))
    with pytest.raises(ValueError):
        GameProfile("bad", regions=((0.0, 0.0, 1.5, 1.0),))
    with pytest.raises(ValueError):
        GameProfile("bad", regions=((0.0, 0.0, 1.0, 1.0),), weights={"sobel": 1.0})
    p = GameProfile("partial", regions=((0, 0, 1, 1),), weights={"canny": 1.0})
    assert p.weights == {"binary": 2.0, "canny": 1.0}


# -------------------------------------------------------------- frame_score


def test_black_frame_scores_zero_and_white_centre_scores_higher() -> None:
    black = np.zeros((180, 320), np.uint8)
    assert frame_score(black) == 0.0
    white_centre = black.copy()
    y0, y1, x0, x1 = region_bounds(black.shape, DEFAULT_PROFILE.regions[0])
    white_centre[y0:y1, x0:x1] = 255
    assert frame_score(white_centre) > frame_score(black)
    # Fully white crop: 100 % threshold x 2.0, no edges inside the crop.
    assert frame_score(white_centre) == pytest.approx(200.0)


def test_edges_add_canny_term() -> None:
    img = np.zeros((1080, 1920), np.uint8)
    img[:, 960:] = 255  # a vertical edge through the centre box
    s = frame_score(img)
    # Half the crop is above threshold (100 points) plus a thin edge line.
    assert 100.0 < s < 110.0


def test_axis_regression_true_centre_vs_lower_left() -> None:
    """The legacy crop swapped axes: on 1920x1080 it looked at rows 885-1035, cols 465-615."""
    h, w = 1080, 1920
    centre = np.zeros((h, w), np.uint8)
    centre[540 - 75 : 540 + 75, 960 - 75 : 960 + 75] = 255
    assert frame_score(centre) >= 150.0

    legacy_patch = np.zeros((h, w), np.uint8)
    legacy_patch[960 - 75 : 960 + 75, 540 - 75 : 540 + 75] = 255  # where the old bug looked
    assert frame_score(legacy_patch) == 0.0

    lower_left = np.zeros((h, w), np.uint8)
    lower_left[h // 2 :, : w // 3] = 255
    assert frame_score(lower_left) == 0.0


def test_multiple_regions_average() -> None:
    profile = GameProfile("two", regions=((0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0)))
    img = np.zeros((100, 200), np.uint8)
    img[:, :100] = 255  # left region all white, right all black
    assert frame_score(img, profile) == pytest.approx(100.0)


def test_rejects_colour_frame() -> None:
    with pytest.raises(ValueError):
        frame_score(np.zeros((10, 10, 3), np.uint8))


# ----------------------------------------------------------- decoding: fixture


def _times(scorer: VisualScorer, segments, a: float, b: float) -> List[float]:
    return [t for t, _ in scorer.frame_scores(segments, a, b)]


def test_keyframes_mode_scores_one_frame_per_second(synthetic_segments: List[Segment]) -> None:
    scorer = VisualScorer(mode="keyframes")
    for a, b in ((10.0, 16.0), (10.5, 15.5), (3.0, 23.0)):
        times = _times(scorer, synthetic_segments, a, b)
        assert len(times) == pytest.approx(b - a, abs=1)
        assert all(a <= t < b for t in times)
        assert times == sorted(times)
        assert np.allclose(np.diff(times), 1.0, atol=0.01)
    result = scorer.score(synthetic_segments, 10.0, 16.0)
    assert set(result) == set(SCORE_KEYS)
    assert result["frames"] == 6
    assert result["visual_max"] >= result["visual_p90"] >= result["visual_mean"] * 0.999


@pytest.mark.parametrize("fps", [4.0, 2.0, 7.5])
def test_sampled_mode_scores_sample_fps(synthetic_segments: List[Segment], fps: float) -> None:
    scorer = VisualScorer(mode="sampled", sample_fps=fps)
    a, b = 10.0, 16.0
    times = _times(scorer, synthetic_segments, a, b)
    assert abs(len(times) - fps * (b - a)) <= 1
    assert all(a <= t < b for t in times)
    # Spacing follows the grid to within one 30 fps frame.
    assert np.all(np.diff(times) > 0)
    assert np.allclose(np.diff(times), 1.0 / fps, atol=1 / 30 + 1e-6)


def test_from_config(synthetic_segments: List[Segment]) -> None:
    scorer = VisualScorer.from_config(RealtimeConfig(visual_mode="sampled", sample_fps=2.0))
    assert (scorer.mode, scorer.sample_fps) == ("sampled", 2.0)
    assert VisualScorer.from_config(RealtimeConfig()).mode == "keyframes"


def test_accepts_lease_like_object_and_skips_segments_outside_core(
    synthetic_segments: List[Segment], tmp_path: Path
) -> None:
    # A segment far outside the core that does not even exist must not be opened.
    ghost = Segment(channel="test", seq=999, start=500.0, duration=2.0, path=tmp_path / "missing.ts")
    lease = SimpleNamespace(segments=list(synthetic_segments[4:9]) + [ghost], start=8.0, end=18.0, truncated=False)
    scorer = VisualScorer()
    assert scorer.score(lease, 10.0, 16.0) == scorer.score(synthetic_segments, 10.0, 16.0)


def test_empty_or_inverted_core_scores_zero(synthetic_segments: List[Segment]) -> None:
    scorer = VisualScorer()
    zero = {"visual_mean": 0.0, "visual_p90": 0.0, "visual_max": 0.0, "frames": 0.0}
    assert scorer.score(synthetic_segments, 16.0, 10.0) == zero
    assert scorer.score(synthetic_segments, 500.0, 510.0) == zero
    assert scorer.score([], 0.0, 10.0) == zero


def test_corrupt_segment_is_skipped_with_warning(
    synthetic_segments: List[Segment], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bad_path = tmp_path / "corrupt.ts"
    bad_path.write_bytes(b"not a transport stream" * 100)
    segs = list(synthetic_segments)
    bad_index = 6  # covers [12, 14)
    segs[bad_index] = dataclasses.replace(segs[bad_index], path=bad_path)
    with caplog.at_level(logging.WARNING, logger="VidFlow.realtime.scoring.visual"):
        result = VisualScorer().score(segs, 10.0, 16.0)
    assert result["frames"] == 4  # 6 keyframes minus the 2 in the corrupt segment
    assert any("could not decode" in r.message or "no video" in r.message for r in caplog.records)


def test_audio_only_segment_is_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "audio.ts"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=d=2", "-c:a", "aac", "-f", "mpegts", str(path)], timeout=60)
    seg = Segment(channel="a", seq=0, start=0.0, duration=2.0, path=path)
    with caplog.at_level(logging.WARNING, logger="VidFlow.realtime.scoring.visual"):
        assert VisualScorer().score([seg], 0.0, 2.0)["frames"] == 0
    assert any("no video" in r.message for r in caplog.records)


def test_score_clip_attaches_scores(synthetic_segments: List[Segment], tmp_path: Path) -> None:
    window = Window(channel="test", window_id="test-000010000", start=10.0, end=16.0, detector="audio", stats={})
    clip = Clip(
        window=window,
        path=tmp_path / "c.mp4",
        file_start=4.0,
        file_duration=16.0,
        content_start=1.0,
        content_end=15.0,
        scores={"audio_prominence": 12.0},
    )
    scored = VisualScorer().score_clip(clip, synthetic_segments)
    assert scored is not clip and clip.scores == {"audio_prominence": 12.0}
    assert scored.scores["audio_prominence"] == 12.0
    assert scored.scores["frames"] == 6
    assert set(SCORE_KEYS) <= set(scored.scores)


def test_bad_mode_rejected() -> None:
    with pytest.raises(ValueError):
        VisualScorer(mode="everything")
    with pytest.raises(ValueError):
        VisualScorer(mode="sampled", sample_fps=0)


# ------------------------------------------------- length independence (static)


@pytest.fixture(scope="module")
def static_segments(tmp_path_factory: pytest.TempPathFactory) -> List[Segment]:
    """24 s of identical frames: black with a white box over part of the centre."""
    d = tmp_path_factory.mktemp("static")
    video = d / "static.mp4"
    run_ffmpeg(
        [
            "-f", "lavfi",
            "-i", "color=c=black:s=320x180:r=30:d=24,drawbox=x=150:y=80:w=20:h=20:color=white:t=fill",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-bf", "0",
            "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
            str(video),
        ],
        timeout=120,
    )
    return segments_from_file(video, d / "segs", channel="static", segment_time=2.0)


@pytest.mark.parametrize("mode", ["keyframes", "sampled"])
def test_long_and_short_clip_of_same_content_score_alike(static_segments: List[Segment], mode: str) -> None:
    scorer = VisualScorer(mode=mode)
    long = scorer.score(static_segments, 0.0, 20.0)
    short = scorer.score(static_segments, 5.0, 10.0)
    assert long["frames"] > 3 * short["frames"] > 0
    assert long["visual_mean"] > 0
    assert long["visual_mean"] == pytest.approx(short["visual_mean"], rel=1e-6)
    assert long["visual_max"] == pytest.approx(short["visual_max"], rel=1e-6)
