"""
Tier 1 MVP orchestrator: watches ONE already-live channel's rolling buffer
for audio spikes and chat-velocity spikes, cutting an instant clip around
whichever fires first. Two independent, general-purpose signals rather
than one: audio alone fires on any loud moment (including ones nobody
cared about -- an explosion, a loud voice line), and chat alone can miss
moments that are exciting but not visually/audibly loud in the stream
itself. Either firing is enough to cut a clip; requiring both would trade
those false positives for missed real moments, which is the wrong
tradeoff for a "don't miss the highlight" instant-clip feature.

Deliberately single-channel and polling-based rather than the full vision
from the design discussion (watching a whole follow list concurrently via
Twitch EventSub push notifications). Scaling this out is mostly "run one
InstantClipWatcher per currently-live channel, in its own
process/thread/asyncio task" plus swapping wait_until_live()'s polling
loop for an EventSub subscription -- but that needs a publicly reachable
endpoint (or the EventSub WebSocket transport) to receive push
notifications, which is a real deployment concern on top of the detection
logic itself, so it's left for later rather than guessed at here.
"""
import logging
import os
import subprocess
import time

from VidFlow.live.audio_trigger import AudioSpikeTrigger, measure_mean_volume
from VidFlow.live.buffer_recorder import RollingBufferRecorder
from VidFlow.live.chat_trigger import ChatVelocityTrigger
from VidFlow.live.instant_clip_cutter import cut_clip_from_segments, segments_covering_window

logger = logging.getLogger(__name__)


def resolve_stream_url(channel, quality="best"):
    """Resolves a live Twitch channel login to a playable stream URL via streamlink."""
    result = subprocess.run(
        ["streamlink", "--stream-url", "https://twitch.tv/{}".format(channel), quality],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("streamlink failed to resolve {}: {}".format(channel, result.stderr))
    return result.stdout.strip()


def wait_until_live(twitch_client, channel_login, poll_interval=30, max_polls=None):
    """Polls Helix `Get Streams` until the channel is live; returns its stream object."""
    polls = 0
    while max_polls is None or polls < max_polls:
        streams = twitch_client.get_streams([channel_login])
        if streams:
            return streams[0]
        polls += 1
        time.sleep(poll_interval)
    return None


class InstantClipWatcher:
    def __init__(self, input_url, buffer_dir, clips_out, segment_seconds=5,
                 retention_seconds=90, pre_seconds=10, post_seconds=15,
                 spike_db=8.0, extra_input_args=None, poll_interval=2.0,
                 chat_monitor=None, chat_spike_ratio=3.0):
        self.recorder = RollingBufferRecorder(
            input_url=input_url, output_dir=buffer_dir, segment_seconds=segment_seconds,
            retention_seconds=retention_seconds, extra_input_args=extra_input_args,
        )
        self.clips_out = clips_out
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.audio_trigger = AudioSpikeTrigger(spike_db=spike_db)
        # chat_monitor is optional (needs a channel to join) -- pass a
        # started ChatMonitor to enable the second signal; anything with a
        # messages_per_second() method works, so tests can inject a fake.
        self.chat_monitor = chat_monitor
        self.chat_trigger = ChatVelocityTrigger(spike_ratio=chat_spike_ratio)
        self.poll_interval = poll_interval
        self._seen_segments = set()
        self._start_time = None
        os.makedirs(clips_out, exist_ok=True)

    def start(self):
        self.recorder.start()
        self._start_time = time.time()

    def stop(self):
        self.recorder.stop()

    def poll_once(self):
        """
        Measures any newly-finished segments and the current chat rate (if
        a chat_monitor is attached), feeding each into its own trigger, and
        cuts a single instant clip if either fires. Returns the list of
        clip paths cut this poll (usually empty, at most one).
        """
        cut_paths = []
        segs = self.recorder.segments()
        # the newest segment file may still be actively written by ffmpeg;
        # only look at ones that are no longer the tail of the list.
        finished = segs[:-1] if len(segs) > 1 else []

        audio_spike = False
        last_volume = None
        for seg in finished:
            if seg in self._seen_segments:
                continue
            self._seen_segments.add(seg)
            last_volume = measure_mean_volume(seg)
            if self.audio_trigger.observe(last_volume):
                audio_spike = True

        chat_spike = False
        chat_rate = None
        if self.chat_monitor is not None:
            chat_rate = self.chat_monitor.messages_per_second()
            chat_spike = self.chat_trigger.observe(chat_rate)

        if audio_spike or chat_spike:
            event_time = time.time() - self._start_time
            window = segments_covering_window(
                self.recorder, event_time, self.pre_seconds, self.post_seconds
            )
            out_path = os.path.join(self.clips_out, "instant_{}.mp4".format(int(event_time)))
            cut_clip_from_segments(window, out_path)
            cut_paths.append(out_path)
            logger.info(
                "Cut instant clip {} (audio_spike={}, chat_spike={}, volume={} dB, chat_rate={} msg/s)".format(
                    out_path, audio_spike, chat_spike, last_volume, chat_rate))

        self.recorder.prune_old_segments()
        return cut_paths

    def run_forever(self):
        self.start()
        try:
            while self.recorder.is_running():
                time.sleep(self.poll_interval)
                self.poll_once()
        finally:
            self.stop()
