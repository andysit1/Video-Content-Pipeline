"""
Tier 1 mechanics: RollingBufferRecorder, measure_mean_volume, and the
segment-picking/concat logic in instant_clip_cutter -- exercised against a
real ffmpeg process reading a generated file with `-re` (native frame
rate), which simulates a live source closely enough to validate the whole
"record -> segment -> measure -> cut" mechanism without needing real
Twitch access. InstantClipWatcher's end-to-end run (including the audio
trigger firing) is covered in test_live_watcher.py.
"""
import os
import time

import pytest

from VidFlow.live.buffer_recorder import RollingBufferRecorder
from VidFlow.live.audio_trigger import measure_mean_volume
from VidFlow.live.instant_clip_cutter import cut_clip_from_segments, segments_covering_window


def wait_for(predicate, timeout=20, interval=0.25):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def live_recorder(spiky_source, tmp_path):
    recorder = RollingBufferRecorder(
        input_url=str(spiky_source["path"]),
        output_dir=str(tmp_path / "buffer"),
        segment_seconds=1,
        retention_seconds=90,
        extra_input_args=["-re"],  # paces reads to simulate a live source
    )
    yield recorder
    recorder.stop()


class TestRollingBufferRecorder:
    def test_produces_segments_while_running_and_stops_cleanly(self, live_recorder):
        live_recorder.start()

        assert wait_for(lambda: len(live_recorder.segments()) >= 3, timeout=15)
        assert live_recorder.is_running()

        assert wait_for(lambda: not live_recorder.is_running(), timeout=15)
        final_segments = live_recorder.segments()
        assert len(final_segments) >= 8  # ~10s of content at 1s segments

    def test_segments_are_returned_oldest_first(self, live_recorder):
        live_recorder.start()
        assert wait_for(lambda: len(live_recorder.segments()) >= 4, timeout=15)

        segs = live_recorder.segments()
        indices = [int(os.path.basename(s).split("_")[1].split(".")[0]) for s in segs]
        assert indices == sorted(indices)

    def test_prune_old_segments_bounds_retention(self, tmp_path, spiky_source):
        recorder = RollingBufferRecorder(
            input_url=str(spiky_source["path"]),
            output_dir=str(tmp_path / "buffer"),
            segment_seconds=1,
            retention_seconds=3,  # keep ~3s of segments
            extra_input_args=["-re"],
        )
        recorder.start()
        try:
            assert wait_for(lambda: len(recorder.segments()) >= 8, timeout=15)
            recorder.prune_old_segments()
            # retention_seconds=3 at 1s segments -> max_segments = 3 + 2 = 5
            assert len(recorder.segments()) <= 5
        finally:
            recorder.stop()


def test_measure_mean_volume_distinguishes_loud_from_quiet_segments(live_recorder):
    live_recorder.start()
    assert wait_for(lambda: not live_recorder.is_running(), timeout=15)

    segs = live_recorder.segments()
    # spike_start=4.0s at 1s segments -> segment index 4 or 5 should be the loud one
    quiet_volume = measure_mean_volume(segs[0])
    loud_volume = measure_mean_volume(segs[5])

    assert quiet_volume is not None and loud_volume is not None
    assert loud_volume > quiet_volume + 10  # a real, unambiguous gap


def test_cut_clip_from_segments_produces_a_playable_window(live_recorder):
    import cv2

    live_recorder.start()
    assert wait_for(lambda: not live_recorder.is_running(), timeout=15)

    window = segments_covering_window(live_recorder, event_time=5.0, pre_seconds=2, post_seconds=2)
    assert len(window) >= 3

    out_path = str(live_recorder.output_dir) + "/instant_clip.mp4"
    cut_clip_from_segments(window, out_path)

    cap = cv2.VideoCapture(out_path)
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    ok, _ = cap.read()
    cap.release()
    assert ok
