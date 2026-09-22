"""
Tier 2 ("go over the most popular clips and build deeper content around
that timestamp"): instead of finding candidate chunk boundaries from
silence detection like AnalyzeDataFiles does, ask Twitch which moments in
a VOD its own viewers already clipped -- that's a stronger, audience-tested
highlight signal than reinventing detection from scratch, and it needs no
live infrastructure at all: it's one API call plus the VOD file, which can
run any time after (or well before) the stream ends.

Each Twitch clip carries `vod_offset` (seconds into the VOD) and
`duration`. This stage turns the top clips (by view count) into wider
"context windows" around each offset -- e.g. 30s of build-up and 45s of
payoff instead of the clip's own ~30s -- merges any that overlap, and
writes them out as chunks.txt in the exact format AnalyzeDataFiles
produces. That makes this a drop-in replacement for it: ActionPipe,
AnalyzeClipsPipe and CompileVideoPipe downstream need no changes at all.
"""
import logging
import os

from VidFlow.aggregate.filehandler_component import FileHandleComponent, FileMaster
from VidFlow.modules.pipeline_builder import Pipe
from VidFlow.pipes.action_stage import ActionPipe

logger = logging.getLogger(__name__)


class PopularClipsStage(Pipe, FileHandleComponent):
    def __init__(self, engine, twitch_client, context_pre_seconds=30, context_post_seconds=45,
                 top_n=10, min_view_count=1):
        super().__init__(engine)
        self.twitch = twitch_client
        self.context_pre_seconds = context_pre_seconds
        self.context_post_seconds = context_post_seconds
        self.top_n = top_n
        self.min_view_count = min_view_count
        self.chunk_path = os.path.join(self.engine.payload['cache_txt_out'], 'chunks.txt')

    def fetch_top_clips(self):
        payload = self.engine.payload
        clips = self.twitch.get_clips(
            broadcaster_id=payload['broadcaster_id'],
            started_at=payload.get('vod_started_at'),
            ended_at=payload.get('vod_ended_at'),
            first=100,
        )
        # only clips Twitch could tie back to this VOD carry a vod_offset
        clips = [c for c in clips
                 if c.get('vod_offset') is not None and c.get('view_count', 0) >= self.min_view_count]
        clips.sort(key=lambda c: c['view_count'], reverse=True)
        return clips[:self.top_n]

    def build_context_windows(self, clips):
        windows = []
        for clip in clips:
            offset = float(clip['vod_offset'])
            duration = float(clip.get('duration') or 0)
            start = max(0.0, offset - self.context_pre_seconds)
            end = offset + duration + self.context_post_seconds
            windows.append((start, end))
        windows.sort(key=lambda w: w[0])
        return self.merge_overlapping_windows(windows)

    @staticmethod
    def merge_overlapping_windows(windows):
        merged = []
        for start, end in windows:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def on_run(self):
        clips = self.fetch_top_clips()
        logger.info("PopularClipsStage: {} popular clips with a usable vod_offset".format(len(clips)))
        windows = self.build_context_windows(clips)
        self.write_lines(self.chunk_path, windows)
        self.on_done()

    def on_done(self):
        self.engine.machine.next_state = ActionPipe(self.engine)

    def on_error(self):
        from VidFlow.pipes.dl_stage import DownloadPipe
        self.engine.machine.current = DownloadPipe(self.engine)


def run_popular_clips_pipeline(engine, twitch_client, **stage_kwargs):
    """
    Entry point for Tier 2: sets up the same cache/clips directory layout
    EntryPipe would (via FileMaster), then runs PopularClipsStage ->
    ActionPipe -> AnalyzeClipsPipe -> (CompileVideoPipe if engine.compile).

    Required in engine.payload: in_filename (the downloaded VOD), video_name,
    is_community, broadcaster_id. Optional: vod_started_at/vod_ended_at
    (RFC3339 -- restricts the clips search to this VOD's time range).
    """
    required = ('in_filename', 'video_name', 'broadcaster_id')
    missing = [key for key in required if not engine.payload.get(key)]
    if missing:
        raise ValueError("Missing required payload keys for popular-clips mode: {}".format(missing))

    FileMaster(engine).setup()
    engine.run(PopularClipsStage(engine, twitch_client, **stage_kwargs))
