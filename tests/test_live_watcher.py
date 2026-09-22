"""
InstantClipWatcher end-to-end: runs the full Tier 1 loop (rolling buffer
-> audio-spike trigger -> instant clip cut) against a real simulated live
source with a known loud spike in the middle, and checks a clip actually
gets produced.
"""
import time

import cv2
import pytest

from VidFlow.live.watcher import InstantClipWatcher


def wait_for(predicate, timeout=20, interval=0.5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_watcher_cuts_an_instant_clip_when_the_spike_plays(spiky_source, tmp_path):
    watcher = InstantClipWatcher(
        input_url=str(spiky_source["path"]),
        buffer_dir=str(tmp_path / "buffer"),
        clips_out=str(tmp_path / "clips"),
        segment_seconds=1,
        retention_seconds=90,
        pre_seconds=2,
        post_seconds=2,
        spike_db=8.0,
        extra_input_args=["-re"],
        poll_interval=0.5,
    )

    watcher.start()
    cut_paths = []
    try:
        deadline = time.time() + 15
        while time.time() < deadline and not cut_paths:
            time.sleep(watcher.poll_interval)
            cut_paths.extend(watcher.poll_once())
    finally:
        watcher.stop()

    assert cut_paths, "no instant clip was cut during the spike window"

    cap = cv2.VideoCapture(cut_paths[0])
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    ok, _ = cap.read()
    cap.release()
    assert ok


def test_watcher_poll_once_ignores_the_still_being_written_segment(spiky_source, tmp_path):
    # regression-style check for the "don't touch the tail segment" guard in
    # poll_once(): with only one segment on disk there's nothing finished
    # yet to measure, so it must not raise and must cut nothing.
    watcher = InstantClipWatcher(
        input_url=str(spiky_source["path"]),
        buffer_dir=str(tmp_path / "buffer"),
        clips_out=str(tmp_path / "clips"),
        segment_seconds=5,
        extra_input_args=["-re"],
    )
    watcher.start()
    try:
        assert wait_for(lambda: len(watcher.recorder.segments()) >= 1, timeout=10)
        cut_paths = watcher.poll_once()
        assert cut_paths == []
    finally:
        watcher.stop()
