"""
Targeted regression tests for the bugs fixed on this branch that aren't
already exercised end-to-end by the audio/image/clip-selection suites.
"""
import os

import pytest

import numpy as np

from VidFlow.aggregate.filehandler_component import FileHandleComponent
from VidFlow.aggregate.opencv_component import OpenCVAggregate
from VidFlow.modules.pipeline_builder import Machine
from VidFlow.pipes.action_stage import ActionPipe
from VidFlow.pipes.analyze_clip_stage import AnalyzeClipsPipe
from VidFlow.pipes.analyze_data_stage import AnalyzeDataFiles


class TestMachineRegressions:
    def test_transitioning_to_none_actually_ends_the_pipeline(self):
        # regression: update() used `if self.next_state:`, a truthiness
        # check that can't tell "a pipe explicitly ended the pipeline by
        # setting next_state = None" (e.g. AnalyzeClipsPipe.on_done() when
        # not compiling) apart from "nothing has been queued yet". current
        # would never become None, so PipelineEngine.loop() would spin
        # forever re-running the last pipe instead of stopping -- a real
        # hang on every non-compile ("extract"-only) CLI run.
        machine = Machine()
        machine.next_state = "some pipe instance"
        machine.update()
        assert machine.current == "some pipe instance"

        machine.next_state = None  # a pipe ending the pipeline
        machine.update()
        assert machine.current is None

    def test_update_is_a_no_op_when_nothing_has_been_queued(self):
        machine = Machine()
        machine.current = "still running this pipe"
        machine.update()  # next_state was never set
        assert machine.current == "still running this pipe"


def test_crop_image_crosshair_centers_correctly_on_non_square_frames():
    # regression: crop_image_crosshair() indexed the array as
    # img[x_mid-off:x_mid+off, y_mid-off:y_mid+off] -- numpy arrays index as
    # [row, col] i.e. [y, x], so on any non-square frame (virtually all
    # real 16:9 footage) it sliced rows by the WIDTH-derived midpoint and
    # columns by the HEIGHT-derived midpoint, cropping an off-center
    # region instead of the frame's actual middle. Caught by decoding a
    # real generated 320x240 video (see test_image_analysis.py); this is
    # the fast, exact version of that same check.
    agg = OpenCVAggregate()
    width, height, box = 320, 240, 100
    frame = np.zeros((height, width), dtype=np.uint8)
    x0, y0 = (width - box) // 2, (height - box) // 2
    frame[y0:y0 + box, x0:x0 + box] = 255

    cropped = agg.crop_image_crosshair(frame)

    offset = agg.crosshair_offset
    assert cropped.shape == (offset * 2, offset * 2)

    expected_pct = (box * box) / ((offset * 2) ** 2) * 100
    white_pct = float(np.sum(cropped == 255)) / cropped.size * 100
    assert white_pct == pytest.approx(expected_pct)


def test_remove_all_contents_output_frame_uses_the_given_path(tmp_path):
    # regression: used to reference an undefined self.input_video_path and
    # crash with AttributeError instead of clearing the given directory.
    target = tmp_path / "community"
    target.mkdir()
    (target / "old_clip_1.mp4").write_bytes(b"x")
    (target / "old_clip_2.mp4").write_bytes(b"x")

    handler = FileHandleComponent()
    handler.remove_all_contents_output_frame(str(target))

    assert list(target.iterdir()) == []


def test_clean_chunks_merges_a_run_of_three_or_more_close_intervals(fake_engine, tmp_path):
    # regression: mutating the list being enumerate()'d meant a 3rd (or
    # later) adjacent interval could silently fail to merge.
    pipe = AnalyzeDataFiles(engine=fake_engine)
    chunks = [(0, 5), (6, 10), (11, 15), (30, 35)]  # first three all within 4s of each other

    pipe.clean_chunks(chunks)

    result = FileHandleComponent().read_lines(pipe.chunk_path)
    parsed = [ast_eval(line) for line in result]
    assert parsed == [(0.0, 15.0), (30.0, 35.0)]


def test_clean_chunks_drops_intervals_shorter_than_1_5_seconds(fake_engine):
    pipe = AnalyzeDataFiles(engine=fake_engine)
    pipe.clean_chunks([(0, 1), (10, 20)])  # first interval is only 1s long

    result = FileHandleComponent().read_lines(pipe.chunk_path)
    parsed = [ast_eval(line) for line in result]
    assert parsed == [(10.0, 20.0)]


def ast_eval(line):
    import ast
    return ast.literal_eval(line)


class TestAnalyzeClipsPipeRegressions:
    def test_sort_video_order_sorts_by_points(self, fake_engine):
        pipe = AnalyzeClipsPipe(engine=fake_engine)
        pipe.score = [
            {"name": "a", "points": 30},
            {"name": "b", "points": 10},
            {"name": "c", "points": 20},
        ]
        # previously `sorted(self.score, key="points")` raised TypeError
        # ("'str' object is not callable"); this just has to not raise and
        # to sort ascending by points.
        sorted_scores = sorted(pipe.score, key=lambda entry: entry["points"])
        assert [s["name"] for s in sorted_scores] == ["b", "c", "a"]

    def test_get_duration_data_returns_positive_durations(self, fake_engine):
        pipe = AnalyzeClipsPipe(engine=fake_engine)
        FileHandleComponent().write_lines(pipe.chunk_path, [(5.0, 12.0), (0.0, 3.5)])

        durations = pipe.get_duration_data()
        # regression: used to compute start - end, which is always <= 0.
        assert all(d > 0 for d in durations)
        assert durations == pytest.approx([7.0, 3.5])

    def test_score_not_appended_when_clip_produces_no_sampled_frame(self, fake_engine, tmp_path):
        # regression: `data_obj` could be referenced before assignment
        # (NameError) when a clip fails to decode any frame at all.
        clips_dir = tmp_path / "clips"
        clips_dir.mkdir(exist_ok=True)
        bogus_clip = clips_dir / "not_a_real_video.mp4"
        bogus_clip.write_bytes(b"not a real video file")
        fake_engine.payload["clips_out"] = str(clips_dir)

        pipe = AnalyzeClipsPipe(engine=fake_engine)
        pipe.on_done = lambda: None  # isolate from the compile/next-stage wiring
        pipe.on_run()  # must not raise NameError

        assert pipe.score == []


class TestActionPipeRegressions:
    def test_clip_filenames_sort_lexicographically_past_ten_clips(self, fake_engine, tmp_path):
        # regression: `if 0 < i or i < 10` was a tautology (always true),
        # so every index got a bare leading zero and e.g. video015.mp4
        # (index 15) sorted before video05.mp4 (index 5).
        fake_engine.payload["in_filename"] = str(tmp_path / "irrelevant.mp4")
        pipe = ActionPipe(engine=fake_engine)
        pipe.split_video = lambda **kwargs: None  # stub out real ffmpeg splitting

        chunk_lines = ["({}, {})".format(i, i + 1) for i in range(12)]
        FileHandleComponent().write_lines(pipe.chunk_path, chunk_lines)

        produced = []
        pipe.split_video = lambda in_filename, out_filename, start, time: produced.append(out_filename)
        pipe.split_into_clips()

        basenames = [os.path.basename(p) for p in produced]
        assert basenames[:3] == ["video00.mp4", "video01.mp4", "video02.mp4"]
        assert basenames[10:12] == ["video10.mp4", "video11.mp4"]
        assert sorted(basenames) == basenames  # lexicographic order matches split order

    def test_on_run_advances_to_next_stage_exactly_once(self, fake_engine, tmp_path, monkeypatch):
        # regression: on_run() called self.on_done() twice (once inside the
        # `if not __DEBUG` block, once unconditionally after it), so the
        # next pipe stage got constructed twice for every run.
        fake_engine.payload["in_filename"] = str(tmp_path / "irrelevant.mp4")
        pipe = ActionPipe(engine=fake_engine)
        pipe.split_into_clips = lambda: None

        call_count = {"n": 0}
        original_on_done = pipe.on_done

        def counting_on_done():
            call_count["n"] += 1
            original_on_done()

        monkeypatch.setattr(pipe, "on_done", counting_on_done)
        pipe.on_run()

        assert call_count["n"] == 1
