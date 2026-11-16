"""The detection rule on synthetic stream-like scenes (``testing.detection_scenes``).

The scenes are rendered to a loudness series in NumPy (RMS per 100 ms hop,
like the extractor, without a codec) and fed straight to the segmenter with
default settings. They are synthetic: passing here says the rule copes with
loud beds, noise, level changes and the warm-up, not that what it finds is
worth clipping on a real stream.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import pytest

from VidFlow.realtime.config import RealtimeConfig
from VidFlow.realtime.eval import RefWindow, match_windows
from VidFlow.realtime.segmenter import OnlineSegmenter
from VidFlow.realtime.testing import detection_scenes, scene_loudness
from VidFlow.realtime.types import LoudnessSample, Window

SCENES = detection_scenes()

#: Scenes the default rule gets exactly right (every highlight, nothing else).
CLEAN = ("loud_bed", "mostly_loud", "quiet_then_loud", "noise_bed", "speechlike", "lull_resume", "steady_noise")

#: Scenes where white-noise spikes within merge_gap_s of a highlight are merged into it
#: (the merge rule working as specified), so its boundaries are not checked.
MERGES_SPIKES = ("noise_bed",)


@lru_cache(maxsize=None)
def _samples(name: str) -> Tuple[LoudnessSample, ...]:
    return tuple(scene_loudness(SCENES[name]))


def _detect(name: str, config: RealtimeConfig = RealtimeConfig()) -> List[Window]:
    seg = OnlineSegmenter("scene", config)
    return seg.feed(_samples(name)) + seg.flush()


def _score(name: str, windows: List[Window]):
    refs = [RefWindow(a, b) for a, b in SCENES[name].truth]
    return refs, match_windows(refs, windows, 0.5)


def test_every_scene_has_a_highlight_in_the_warmup() -> None:
    for name, scene in SCENES.items():
        if scene.truth:
            assert scene.truth[0][0] < RealtimeConfig().warmup_s, name


@pytest.mark.parametrize("name", CLEAN)
def test_clean_scenes(name: str) -> None:
    windows = _detect(name)
    refs, matches = _score(name, windows)
    assert len(matches) == len(refs) == len(windows), [(w.start, w.end) for w in windows]
    if name not in MERGES_SPIKES:
        for m in matches:
            assert abs(m.start_error) <= 0.2 and abs(m.end_error) <= 0.2, (m.ref, m.det.start, m.det.end)


def test_scene_detection_is_deterministic_across_batching() -> None:
    samples = _samples("quiet_then_loud")  # warm-up window and a re-judged level change
    whole = _detect("quiet_then_loud")
    seg = OnlineSegmenter("scene", RealtimeConfig())
    chunked: List[Window] = []
    for i in range(0, len(samples), 20):  # 2 s segments
        chunked += seg.feed(samples[i : i + 20])
    assert chunked + seg.flush() == whole
    assert seg.rejudged == 1


# --------------------------------------------------------------------------- known limits
# strict: if one of these starts passing, the documented limit has changed.


@pytest.mark.xfail(strict=True, reason="history decays over stats_window_s: after a loud stretch, "
                   "quieter highlights are missed for about 0.9 x stats_window_s")
def test_limit_loud_then_quiet() -> None:
    refs, matches = _score("loud_then_quiet", _detect("loud_then_quiet"))
    assert len(matches) == len(refs)


@pytest.mark.xfail(strict=True, reason="a level detected more than ~31 % of the time becomes normal "
                   "(window_weight budget): highlights filling ~half the stream are not found")
def test_limit_dense_peaks() -> None:
    refs, matches = _score("dense_peaks", _detect("dense_peaks"))
    assert len(matches) == len(refs)


@pytest.mark.xfail(strict=True, reason="separate normal-level events (utterances with >4 s pauses) after a "
                   "quiet start are reported until they fill ~31 % of the history (about 2 min here)")
def test_limit_sparse_speech() -> None:
    windows = _detect("sparse_speech")
    refs, matches = _score("sparse_speech", windows)
    assert len(matches) == len(windows)


def test_dense_peaks_found_with_a_lower_window_weight() -> None:
    # The budget is a dial: window_weight 0.1 moves it to ~53 %.
    refs, matches = _score("dense_peaks", _detect("dense_peaks", RealtimeConfig(window_weight=0.1)))
    assert len(matches) == len(refs)
