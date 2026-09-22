# Live integration (Twitch)

Two-tier clip generation for a clipping channel that follows *other*
streamers, per the design discussion:

1. **Instant clips** (`buffer_recorder.py`, `audio_trigger.py`,
   `instant_clip_cutter.py`, `watcher.py`) -- a rolling local buffer of a
   live channel's stream, with a real-time trigger (currently: audio
   loudness spikes) that cuts a clip out of the buffer the moment something
   exciting happens.
2. **Popular-clip context** (`popular_clips_stage.py`) -- after the fact
   (any time after the VOD exists), ask Twitch which moments its own
   viewers already clipped for a broadcaster, and build a wider "context
   window" around each one so the deep-dive version has real build-up and
   payoff instead of just the ~30s a viewer's own clip captured. This is
   the "easiest" piece: it's one API call plus a VOD file, no live
   infrastructure, and it's a drop-in replacement for `AnalyzeDataFiles`
   -- it writes the same `chunks.txt` format, so `ActionPipe`,
   `AnalyzeClipsPipe` and `CompileVideoPipe` need no changes at all.

## Setup

- A Twitch developer application (https://dev.twitch.tv/console/apps) for
  a Client ID/Secret -- used for the app access token (client-credentials
  flow), no user login needed for the endpoints this uses.
- `streamlink` on PATH, to resolve a live channel to a playable stream URL
  (handles Twitch's playlist auth token and ad-segment quirks -- don't
  hand-roll this): `pip install streamlink`.
- `ffmpeg` on PATH (already a requirement of the rest of this project).

## Usage

### Tier 2 -- popular-clip context (the one to start with)

```python
from VidFlow.live.twitch_api import TwitchClient
from VidFlow.live.popular_clips_stage import run_popular_clips_pipeline
from VidFlow.modules.pipeline_builder import PipelineEngine

twitch = TwitchClient(client_id="...", client_secret="...")
engine = PipelineEngine()
engine.output_path = "/path/to/output"
engine.compile = True  # also concatenate the surviving windows into one video
engine.payload = {
    "is_community": False,
    "video_name": "streamer_vod_2024_01_01",
    "in_filename": "/path/to/downloaded_vod.mp4",
    "broadcaster_id": "123456",             # from twitch.get_users(["login"])
    "vod_started_at": "2024-01-01T00:00:00Z",  # optional: restricts the clip search window
    "vod_ended_at": "2024-01-01T03:00:00Z",
}

run_popular_clips_pipeline(engine, twitch, top_n=10, context_pre_seconds=30, context_post_seconds=45)
```

### Tier 1 -- instant clips (single channel MVP)

```python
from VidFlow.live.twitch_api import TwitchClient
from VidFlow.live.watcher import InstantClipWatcher, resolve_stream_url, wait_until_live

twitch = TwitchClient(client_id="...", client_secret="...")
wait_until_live(twitch, "some_channel")  # blocks, polling, until they go live

watcher = InstantClipWatcher(
    input_url=resolve_stream_url("some_channel"),
    buffer_dir="/path/to/buffer",
    clips_out="/path/to/output/clips",
)
watcher.run_forever()  # runs until the stream ends; Ctrl-C to stop
```

## What's not built yet

- **EventSub instead of polling.** `wait_until_live()` polls Helix
  `Get Streams`; Twitch's EventSub (webhook or WebSocket transport) gives
  near-instant `stream.online` notifications instead, but needs either a
  publicly reachable HTTPS endpoint or a persistent WebSocket connection --
  a real deployment decision, not guessed at here.
- **Multi-channel concurrency.** `InstantClipWatcher` watches one channel.
  Watching a follow list means one watcher instance per currently-live
  channel, each in its own process/thread/asyncio task -- the pieces here
  (recorder, trigger, cutter) don't need to change, only the orchestration
  around them.
- **Additional trigger signals.** Audio spikes are a deliberately general
  first signal. Chat-velocity/hype-emote spikes (Twitch chat/IRC or
  EventSub chat subscriptions) and the crosshair/kill-feed image scoring
  already in this repo (game-specific, so only useful once you know what
  you're watching) are natural additions to combine with it.
- **Auto-posting.** `pipes/upload_stage.py` is still an empty stub --
  clips land in `clips_out` for now, nothing pushes them anywhere.
- **Content/ToS considerations for reposting other streamers' moments**
  are a product question independent of this code, worth having an answer
  to before this runs against real channels.
