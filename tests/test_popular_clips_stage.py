"""
PopularClipsStage tests: the context-window math against a fake
TwitchClient (no network), plus a real end-to-end run proving it hands off
to the existing ActionPipe/AnalyzeClipsPipe machinery correctly -- the
whole point of writing chunks.txt in the same format AnalyzeDataFiles does.
"""
import ast

import pytest

from VidFlow.aggregate.filehandler_component import FileHandleComponent
from VidFlow.live.popular_clips_stage import PopularClipsStage, run_popular_clips_pipeline
from VidFlow.pipes.action_stage import ActionPipe
from VidFlow.pipes.analyze_clip_stage import AnalyzeClipsPipe


class FakeTwitchClient:
    def __init__(self, clips):
        self._clips = clips
        self.calls = []

    def get_clips(self, broadcaster_id, started_at=None, ended_at=None, first=100):
        self.calls.append((broadcaster_id, started_at, ended_at, first))
        return self._clips


def make_clip(vod_offset, duration=20.0, view_count=100, clip_id=None):
    return {
        "id": clip_id or "clip_{}".format(vod_offset),
        "vod_offset": vod_offset,
        "duration": duration,
        "view_count": view_count,
    }


class TestClipSelection:
    def test_drops_clips_without_a_vod_offset_or_below_min_views(self, fake_engine):
        clips = [
            make_clip(vod_offset=None, view_count=1000),  # no vod link, e.g. clipped from a rerun/highlight
            make_clip(vod_offset=50, view_count=1),
            make_clip(vod_offset=100, view_count=50),
        ]
        fake_engine.payload["broadcaster_id"] = "123"
        stage = PopularClipsStage(fake_engine, FakeTwitchClient(clips), min_view_count=10)

        kept = stage.fetch_top_clips()
        assert [c["vod_offset"] for c in kept] == [100]

    def test_ranks_by_view_count_descending_and_caps_at_top_n(self, fake_engine):
        clips = [make_clip(vod_offset=i * 100, view_count=v) for i, v in enumerate([50, 500, 200, 10])]
        fake_engine.payload["broadcaster_id"] = "123"
        stage = PopularClipsStage(fake_engine, FakeTwitchClient(clips), top_n=2)

        kept = stage.fetch_top_clips()
        assert [c["view_count"] for c in kept] == [500, 200]

    def test_passes_vod_time_window_through_to_the_api(self, fake_engine):
        fake_engine.payload.update({
            "broadcaster_id": "123",
            "vod_started_at": "2024-01-01T00:00:00Z",
            "vod_ended_at": "2024-01-01T03:00:00Z",
        })
        client = FakeTwitchClient([])
        PopularClipsStage(fake_engine, client).fetch_top_clips()

        assert client.calls[0] == ("123", "2024-01-01T00:00:00Z", "2024-01-01T03:00:00Z", 100)


class TestContextWindows:
    def test_builds_a_padded_window_around_each_clip(self, fake_engine):
        stage = PopularClipsStage(fake_engine, FakeTwitchClient([]),
                                   context_pre_seconds=30, context_post_seconds=45)
        windows = stage.build_context_windows([make_clip(vod_offset=1000, duration=20)])
        assert windows == [(970.0, 1065.0)]

    def test_clamps_window_start_to_zero_near_the_start_of_the_vod(self, fake_engine):
        stage = PopularClipsStage(fake_engine, FakeTwitchClient([]), context_pre_seconds=30)
        windows = stage.build_context_windows([make_clip(vod_offset=10, duration=5)])
        assert windows[0][0] == 0.0

    def test_merges_overlapping_context_windows(self, fake_engine):
        stage = PopularClipsStage(fake_engine, FakeTwitchClient([]),
                                   context_pre_seconds=30, context_post_seconds=45)
        # two popular clips 40s apart -- their padded windows overlap and
        # should become one continuous chunk instead of two.
        windows = stage.build_context_windows([
            make_clip(vod_offset=1000, duration=20),
            make_clip(vod_offset=1040, duration=20),
        ])
        assert windows == [(970.0, 1105.0)]


def test_on_run_writes_chunks_txt_that_action_pipe_can_read(fake_engine):
    clips = [make_clip(vod_offset=500, duration=20, view_count=100)]
    fake_engine.payload["broadcaster_id"] = "123"
    stage = PopularClipsStage(fake_engine, FakeTwitchClient(clips),
                               context_pre_seconds=30, context_post_seconds=45)

    stage.on_run()

    assert isinstance(fake_engine.machine.next_state, ActionPipe)
    action_pipe = fake_engine.machine.next_state
    chunks = action_pipe.get_chunk_data()
    assert chunks == [(470.0, 565.0)]


def test_popular_clips_end_to_end_produces_real_split_clips(fake_engine, fake_video, tmp_path):
    # prove the drop-in claim for real: PopularClipsStage writes chunks.txt,
    # then the *unmodified* ActionPipe/AnalyzeClipsPipe consume it exactly
    # like they would AnalyzeDataFiles's output, against a real generated
    # "VOD".
    fake_engine.payload.update({
        "in_filename": str(fake_video["path"]),
        "broadcaster_id": "123",
    })
    # one popular clip roughly in the middle of our 4s fake VOD
    clips = [make_clip(vod_offset=2.0, duration=1.0, view_count=999)]
    stage = PopularClipsStage(fake_engine, FakeTwitchClient(clips),
                               context_pre_seconds=1.0, context_post_seconds=1.0)

    fake_engine.run(stage)  # drives PopularClipsStage -> ActionPipe -> AnalyzeClipsPipe -> (stop)

    clips_out = fake_engine.payload["clips_out"]
    produced = FileHandleComponent().read_lines(
        __import__("os").path.join(fake_engine.payload["cache_txt_out"], "analyze_data.txt")
    )
    assert len(produced) >= 1  # AnalyzeClipsPipe scored at least the one real split clip

    import glob
    split_clips = glob.glob(clips_out + "/video*.mp4")
    assert len(split_clips) == 1
