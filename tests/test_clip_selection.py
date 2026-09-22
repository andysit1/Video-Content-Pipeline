"""
Clip-selection tests: CompileVideoPipe reads per-clip scores written by
AnalyzeClipsPipe, drops clips scoring below the average, and stitches the
survivors back together with ffmpeg's concat demuxer. These tests build a
fake analyze_data.txt (and, for the integration test, real tiny generated
clips) to check the ranking/threshold math and the final compiled output.

Note: CompileVideoPipe.get_video_order()/compare_clips() (an alternate,
never-wired-up ordering strategy gated behind the class's own __DEBUG flag)
is left untested here -- compare_clips() indexes its arguments with `[0]`
as if they were strings, but the only caller passes it clip dicts, so it
raises KeyError as soon as two clips are queued. It's unreachable from the
real pipeline (on_run() never sets __DEBUG=True), so it's out of scope for
this suite; flagging it here in case that code is ever revived.
"""
import cv2
import pytest

from VidFlow.aggregate.filehandler_component import FileHandleComponent
from VidFlow.pipes.compile_stage import CompileVideoPipe


def write_analyze_data(cache_dir, clips):
    """clips: list of (name, points) -> writes analyze_data.txt like AnalyzeClipsPipe would."""
    path = cache_dir / "analyze_data.txt"
    rows = [{"name": name, "points": points} for name, points in clips]
    FileHandleComponent().write_lines(str(path), rows)
    return path


class TestRanking:
    def test_average_points_computed_from_all_clips(self, fake_engine, tmp_path):
        clips = [("clipA", 10), ("clipB", 50), ("clipC", 90)]
        write_analyze_data(tmp_path / "text_cache", clips)

        pipe = CompileVideoPipe(engine=fake_engine)
        stats = pipe.data[-1]
        assert stats["total_points"] == 150
        assert stats["average_points"] == pytest.approx(50)

    def test_threshold_drops_only_clips_below_cutoff(self, fake_engine, tmp_path):
        clips = [("clipA", 10), ("clipB", 50), ("clipC", 90)]
        write_analyze_data(tmp_path / "text_cache", clips)

        pipe = CompileVideoPipe(engine=fake_engine)
        pipe.threshold_video_points(40)

        remaining_names = {c["name"] for c in pipe.data[:-1]}
        assert remaining_names == {"clipB", "clipC"}
        # the stats row must survive thresholding -- it's excluded from the
        # scan (self.data[:-1]) specifically so compile_video() can pop it.
        assert pipe.data[-1] == {"total_points": 150, "average_points": pytest.approx(50)}

    def test_sort_videos_in_order_high_ranks_descending_without_mutating_data(self, fake_engine, tmp_path):
        clips = [("clipA", 10), ("clipB", 90), ("clipC", 50)]
        write_analyze_data(tmp_path / "text_cache", clips)

        pipe = CompileVideoPipe(engine=fake_engine)
        original_order = [c["name"] for c in pipe.data[:-1]]

        ranked = pipe.sort_videos_in_order_high()
        assert [c["name"] for c in ranked] == ["clipB", "clipC", "clipA"]
        # sort_videos_in_order_high() must not reorder pipe.data itself
        assert [c["name"] for c in pipe.data[:-1]] == original_order


class TestCompileVideoConcatenation:
    def test_compile_video_keeps_every_surviving_clip_in_the_concat_list(self, fake_engine, tmp_path):
        # regression test for the `lines[1:]` bug: the first surviving clip
        # (in file order) must not be dropped from the ffmpeg concat list.
        clips = [("/videos/clipA.mp4", 80), ("/videos/clipB.mp4", 10), ("/videos/clipC.mp4", 60)]
        write_analyze_data(tmp_path / "text_cache", clips)

        pipe = CompileVideoPipe(engine=fake_engine)
        average = pipe.data[-1]["average_points"]
        pipe.threshold_video_points(average)  # drops clipB (10 < 30)

        # stub out the actual ffmpeg call so this stays a fast unit test
        # focused on the concat-list-building logic; the full real-ffmpeg
        # path is covered separately below.
        pipe.combine_videos_demuxer_method = lambda out_filename, concat_list_path: None
        pipe.compile_video()

        # regression test: compile_video() used to write this list to a bare
        # "tmp_file.txt" wherever the process happened to be launched from
        # (clobbered by concurrent runs); it now lives alongside the clip's
        # other cache files.
        concat_list_path = tmp_path / "text_cache" / "tmp_file.txt"
        written = FileHandleComponent().read_lines(str(concat_list_path))
        names_in_concat_list = [line.strip().removeprefix("file ") for line in written]
        assert names_in_concat_list == ["/videos/clipA.mp4", "/videos/clipC.mp4"]


@pytest.mark.usefixtures("make_color_clip")
def test_compile_video_end_to_end_produces_expected_concatenated_output(
    fake_engine, tmp_path, make_color_clip
):
    clip_a = make_color_clip("clip_a.mp4", "red", duration=1)
    clip_b = make_color_clip("clip_b.mp4", "blue", duration=1)  # should be dropped
    clip_c = make_color_clip("clip_c.mp4", "green", duration=1)

    # points chosen so the survivor order (clipA then clipC, by file order)
    # differs from points-sorted order (clipC=80 > clipA=60) -- this proves
    # compile_video() keeps clips in their original order rather than
    # resorting them by score.
    clips = [(str(clip_a), 60), (str(clip_b), 10), (str(clip_c), 80)]
    write_analyze_data(tmp_path / "text_cache", clips)

    pipe = CompileVideoPipe(engine=fake_engine)
    fake_engine.payload["video_name"] = "compiled_output"
    average = pipe.data[-1]["average_points"]
    pipe.threshold_video_points(average)
    pipe.compile_video()

    out_path = tmp_path / "clips" / "compiled_output.mp4"
    assert out_path.exists()

    cap = cv2.VideoCapture(str(out_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_duration = frame_count / fps
    assert total_duration == pytest.approx(2.0, abs=0.3)  # clipA (1s) + clipC (1s), clipB dropped

    def mean_bgr_at(t_seconds):
        cap.set(cv2.CAP_PROP_POS_MSEC, t_seconds * 1000)
        ok, frame = cap.read()
        assert ok
        return frame.mean(axis=(0, 1))  # B, G, R

    first_half = mean_bgr_at(0.3)
    second_half = mean_bgr_at(1.3)
    cap.release()

    b, g, r = first_half
    assert r > g and r > b, "expected the red clip (clipA) first, got BGR={}".format(first_half)

    b, g, r = second_half
    assert g > r and g > b, "expected the green clip (clipC) second, got BGR={}".format(second_half)
