"""
Shared fixtures for the VidFlow test suite.

The pipeline is built around three kinds of raw material: a recording with an
audio track (silence/volume detection), the video frames pulled out of clips
(OpenCV image analysis), and a set of already-cut clips that get ranked and
stitched back together (clip selection / compiling). Rather than mocking
ffmpeg/OpenCV, this suite generates small real media files with `ffmpeg` and
feeds them through the real pipeline code, so the tests catch the same class
of bugs (off-by-ones in regex parsing, list-mutation-while-iterating, wrong
codec args, etc.) that the pipeline hits on real recordings.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# `src/VidFlow` is a plain (non-installed) package during development, so
# make it importable the same way `poetry run` / an editable install would.
SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from icecream import ic  # noqa: E402
from VidFlow.modules.pipeline_builder import Machine  # noqa: E402

# several pipes call ic.enable()/print() as a side effect of on_run(); keep
# test output readable by default (individual tests can ic.enable() again).
ic.disable()


FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None

try:
    import cv2  # noqa: F401
    import numpy  # noqa: F401
    import ffmpeg  # noqa: F401
    MEDIA_LIBS_AVAILABLE = True
except ImportError:
    MEDIA_LIBS_AVAILABLE = False

requires_media_stack = pytest.mark.skipif(
    not (FFMPEG_AVAILABLE and MEDIA_LIBS_AVAILABLE),
    reason="requires the ffmpeg binary plus cv2/numpy/ffmpeg-python",
)


def run_ffmpeg(args):
    """Run an ffmpeg command, raising with full stderr on failure."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"] + args
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg failed: {}\n{}".format(" ".join(cmd), result.stderr))
    return result


class FakeEngine:
    """
    Minimal stand-in for PipelineEngine. Real Pipe subclasses only touch
    `engine.payload`, `engine.compile` and `engine.machine` -- this gives
    them exactly that, without dragging in cli.py / click / questionary.
    """

    def __init__(self, payload=None, compile_flag=False):
        self.payload = payload or {}
        self.compile = compile_flag
        self.machine = Machine()
        self.output_path = None
        self.running = True

    def load_payload(self, payload: dict):
        self.payload = payload

    def loop(self, max_steps=1000):
        # mirrors PipelineEngine.loop(), bounded so a broken on_done() chain
        # fails the test instead of hanging it.
        steps = 0
        while self.running:
            self.machine.update()
            if self.machine.current:
                self.machine.current.on_run()
            else:
                self.running = False
            steps += 1
            if steps > max_steps:
                raise RuntimeError("FakeEngine.loop() exceeded max_steps -- runaway pipe chain?")

    def run(self, state):
        self.machine.current = state
        self.loop()


@pytest.fixture
def fake_engine(tmp_path):
    cache_dir = tmp_path / "text_cache"
    clips_dir = tmp_path / "clips"
    cache_dir.mkdir()
    clips_dir.mkdir()
    return FakeEngine(payload={
        "is_community": False,
        "is_caster_mode": None,
        "in_filename": None,
        "video_name": "test_video",
        "cache_txt_out": str(cache_dir),
        "clips_out": str(clips_dir),
    })


@pytest.fixture
def fake_video(tmp_path):
    """
    Generates one small .mp4 with a *known* audio and video pattern:

      video: 4s @ 320x240, a solid 100x100 white box centered on black,
             constant for the whole clip.
      audio: 1s tone, 1s silence, 1s tone, 1s silence (44.1kHz mono).

    Returns a dict with the file path plus the ground-truth values tests
    assert against.
    """
    if not (FFMPEG_AVAILABLE and MEDIA_LIBS_AVAILABLE):
        pytest.skip("requires the ffmpeg binary plus cv2/numpy/ffmpeg-python")

    out_path = tmp_path / "fake_source.mp4"
    width, height, fps, duration = 320, 240, 10, 4
    box = 100

    run_ffmpeg([
        "-f", "lavfi", "-i", "color=c=black:s={}x{}:r={}:d={}".format(width, height, fps, duration),
        "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=44100:duration=1",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:duration=1",
        "-filter_complex",
        "[0:v]drawbox=x=(iw-{box})/2:y=(ih-{box})/2:w={box}:h={box}:color=white:t=fill[v];"
        "[1:a][2:a][1:a][2:a]concat=n=4:v=0:a=1[a]".format(box=box),
        "-map", "[v]", "-map", "[a]",
        "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(out_path),
    ])

    return {
        "path": out_path,
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "box_size": box,
        # ground truth: silence lives in the 2nd and 4th one-second blocks
        "silent_intervals": [(1.0, 2.0), (3.0, 4.0)],
    }


@pytest.fixture
def make_color_clip(tmp_path):
    """Factory fixture: generates a short, silent, solid-color .mp4 clip."""

    def _make(name, color, duration=1, width=320, height=240, fps=10):
        out_path = tmp_path / name
        run_ffmpeg([
            "-f", "lavfi", "-i", "color=c={}:s={}x{}:r={}:d={}".format(color, width, height, fps, duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out_path),
        ])
        return out_path

    if not (FFMPEG_AVAILABLE and MEDIA_LIBS_AVAILABLE):
        pytest.skip("requires the ffmpeg binary plus cv2/numpy/ffmpeg-python")

    return _make


@pytest.fixture
def spiky_source(tmp_path):
    """
    A short (10s) file with a known loud spike in the middle: 4s quiet,
    2s loud tone, 4s quiet. Read with `-re` (native frame rate) this
    simulates a live source for the Tier 1 buffer/trigger/cutter tests
    without needing a real Twitch stream.
    """
    if not (FFMPEG_AVAILABLE and MEDIA_LIBS_AVAILABLE):
        pytest.skip("requires the ffmpeg binary plus cv2/numpy/ffmpeg-python")

    out_path = tmp_path / "spiky_source.mp4"
    run_ffmpeg([
        "-f", "lavfi", "-i", "color=c=black:s=320x240:r=10:d=10",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=44100:duration=2",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:duration=4",
        "-filter_complex",
        "[1:a][2:a][3:a]concat=n=3:v=0:a=1[a]",
        "-map", "0:v", "-map", "[a]",
        "-shortest",
        # force a keyframe every second: RollingBufferRecorder segments with
        # `-c copy`, which can only cut on keyframes, and libx264's default
        # GOP is much longer than our 1s test segments would need.
        "-force_key_frames", "expr:gte(t,n_forced*1)",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        str(out_path),
    ])
    return {"path": out_path, "duration": 10, "spike_start": 4.0, "spike_end": 6.0}
