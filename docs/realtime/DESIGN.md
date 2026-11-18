# VidFlow real-time clipper: design

Status: proposed. Branch: `real-time`. Task breakdown: [TASKS.md](./TASKS.md).

## 1. Goal and scope

Turn VidFlow from "analyse a finished video file" into "follow a live stream and
emit clips while it is still running", for many channels at once, with the
shortest possible time between a moment ending and its clip existing on disk.

**In scope for this phase** (everything here runs with no Twitch access):

- A simulated live feed built from an existing video file.
- Ingest, rolling buffer, audio detection, clip cutting, visual scoring, ranking
  metadata, session log, latency metrics.
- Several simulated channels running concurrently.
- An evaluation and benchmark harness.
- A generic HLS playlist reader tested against a local simulator. This is the
  piece a Twitch source will later plug into.

**Out of scope for this phase:**

- Anything that needs a Twitch credential or the Twitch network: resolving a
  channel's playlist, chat, go-live detection, the Create Clip API.
- Vertical (9:16) rendering, captions, titles, publishing, approval queues.
- Changing the existing batch pipeline. It stays as it is.

## 2. Measurements behind the design

A throwaway spike was run on a 4-core cloud container against a synthetic
2-minute 1080p60, 6 Mbps stream split into 2-second MPEG-TS segments. Real
gameplay will decode slower than synthetic video; treat these as order of
magnitude.

| Operation | Result |
|---|---|
| Split the 2-minute file into 61 segments (stream copy) | 0.3 s |
| Decode one segment's audio and compute 100 ms loudness (PyAV + NumPy) | 21 ms per 2 s segment, about 1% of one core |
| Cut a 20 s clip from 10 buffered segments (stream copy to mp4) | 0.36 s |
| Decode 20 s of video, every frame, score every 15th | 1.1 s |
| Decode 20 s of video, keyframes only, score each | 0.3 s |

Two things the spike showed that the design depends on:

- **Container timestamps are not trustworthy.** Each `.ts` segment's first audio
  timestamp carried a 1.4 s muxer offset. Timing must come from the segment
  index, never from packet timestamps.
- **Loudness framing must carry over between segments.** Framing each segment
  on its own dropped about 1.5 s of audio over 2 minutes.

Conclusion on language: the per-channel steady-state cost is about 1% of a core,
and the expensive steps are already inside FFmpeg and OpenCV. Python stays.

## 3. Principles

1. **The segment is the unit of flow.** A source yields short compressed
   chunks (about 2 s). Nothing downstream knows where they came from.
2. **Media time, never wall-clock time, drives logic.** Every detection and
   decision is a function of stream time. The same input therefore produces the
   same clips at 1x, at 50x, or unpaced. Wall-clock time is only recorded, for
   latency metrics.
3. **Ingest never waits.** The thread that receives segments does only cheap
   work. Anything slow goes to a worker pool. If the pool is overloaded, work is
   dropped; ingest is never stalled.
4. **Cut first, score second.** Cutting is a stream copy and nearly free, so
   every detected window is cut immediately. Scoring is attached to the clip
   afterwards as metadata. Speed is never held up by ranking.
5. **Decode video only for candidates.** Audio is analysed continuously; video
   frames are decoded only for the few seconds a window covers.
6. **Record everything needed to re-decide.** The session log holds the raw
   loudness series and every decision, so thresholds can be re-tuned in
   milliseconds without the video.

## 4. Architecture

```
                per channel (one ingest thread)                     shared
┌────────┐   ┌──────────────┐   ┌───────────────┐   ┌───────────┐   ┌─────────────┐
│ Source │──▶│ SegmentBuffer│──▶│ Loudness      │──▶│ Online    │──▶│ Worker pool │
│        │   │ (rolling,    │   │ extractor     │   │ segmenter │   │  ClipWriter │
│        │   │  on disk)    │   │ (audio only)  │   │           │   │  VisualScore│
└────────┘   └──────┬───────┘   └───────────────┘   └───────────┘   └──────┬──────┘
                    │ leases (segments pinned while a job uses them)        │
                    └───────────────────────────────────────────────────────┘
                                         │
                                  ┌──────▼──────┐
                                  │ SessionLog  │  events.jsonl, clips + sidecars
                                  └─────────────┘
Supervisor: owns the channel list, starts/stops one ChannelPipeline per channel.
```

### 4.1 Mapping to the existing pipeline

The stage decomposition is kept. The engine is not: `PipelineEngine` runs each
`Pipe` once in order, which cannot express stages that all run continuously.
The real-time code lives in its own package and the batch pipeline is untouched.

| Batch stage | Real-time component |
|---|---|
| `DownloadPipe` | `Source` implementations + `SegmentBuffer` |
| `DataCachePipe` (`volumedetect` histogram, `silencedetect`) | `LoudnessExtractor` + `RollingLoudnessStats` |
| `AnalyzeDataFiles` (merge ≤ 4 s, minimum 1.5 s) | `OnlineSegmenter` |
| `ActionPipe` (seek and re-encode) | `ClipWriter` (stream copy from the buffer) |
| `AnalyzeClipsPipe` (OpenCV scoring) | `VisualScorer`, on demand |
| Average-score cull in `CompileVideoPipe` | Running percentile recorded per clip; culling is left to the consumer |
| txt caches + `ast.literal_eval` | `SessionLog` (JSONL) |

### 4.2 Package layout

```
src/VidFlow/realtime/
  types.py            shared dataclasses and protocols (the contract)
  config.py           RealtimeConfig with all tunables
  media.py            ffmpeg/ffprobe helpers (segment a file, probe)
  testing.py          fakes and fixture builders for tests
  sources/
    replay.py         ReplaySource: file → paced segments
    hls.py            HlsSource: media-playlist poller (local path or URL)
  sim/
    hls_sim.py        file → live HLS playlist on disk
  buffer.py           SegmentBuffer + Lease
  audio.py            LoudnessExtractor, RollingLoudnessStats
  segmenter.py        OnlineSegmenter (pure logic)
  clip_writer.py      ClipWriter
  scoring/visual.py   VisualScorer, GameProfile
  session.py          SessionLog, LatencyTracker
  pipeline.py         ChannelPipeline, WorkerPool, AudioDetector glue
  supervisor.py       Supervisor
  eval.py             oracle comparison, re-tune, benchmark
  cli.py              `vidflow-live` entry point
tests/realtime/       one test module per component + end-to-end
```

## 5. Contracts

These are fixed by task T0 and may not be changed by other tasks without
updating this document.

```python
@dataclass(frozen=True)
class Segment:
    channel: str
    seq: int              # strictly increasing within a session
    start: float          # media time: seconds since the session's first segment
    duration: float
    path: Path            # a self-contained MPEG-TS file starting on a keyframe
    discontinuity: bool = False   # content was lost between the previous segment and this one
    arrived_at: float = 0.0       # time.monotonic() when the source released it
    # end = start + duration

@dataclass(frozen=True)
class LoudnessSample:
    t: float              # media time of the start of the hop
    db: float             # RMS level in dBFS over [t, t + hop)

@dataclass(frozen=True)
class Window:
    channel: str
    window_id: str        # f"{channel}-{start_ms:09d}"
    start: float          # media time of the first loud sample
    end: float            # media time of the end of the last loud sample
    detector: str         # "audio" for now
    stats: Mapping[str, float]    # peak_db, mean_db, threshold_db, prominence_db
    forced: bool = False  # closed by max length, a discontinuity or end of stream

@dataclass(frozen=True)
class Clip:
    window: Window
    path: Path
    file_start: float     # media time of the first frame in the file
    file_duration: float
    content_start: float  # offset in the file where the window (plus padding asked for) begins
    content_end: float
    scores: Mapping[str, float]   # filled in after scoring; empty until then

class Source(Protocol):
    channel: str
    def __iter__(self) -> Iterator[Segment]: ...   # blocks until the next segment; ends at end of stream
    def close(self) -> None: ...                   # unblocks the iterator promptly, safe from any thread

class Detector(Protocol):
    def feed(self, segment: Segment) -> list[Window]: ...   # windows that closed because of this segment
    def flush(self) -> list[Window]: ...                    # end of stream
```

**Media time definition.** `Segment.start` is the running sum of the durations
of the segments the source has yielded so far. It is therefore always
contiguous and strictly increasing, even when content was lost; a loss is
signalled only by `discontinuity=True`.

**Discontinuity rules.** The loudness extractor drops its carried-over samples.
The segmenter force-closes any open window at the boundary. A lease never
extends padding across a boundary. So no clip ever contains a jump.

## 6. Components

### 6.1 Sources

**`ReplaySource(video, channel, work_dir, speed=1.0, ...)`**: the workhorse for
this phase. It splits the file into segments up front (0.3 s for 2 minutes;
stream copy), then releases each segment at the moment it would have finished
in a real broadcast: wall time `t0 + segment.end / speed`. `speed=None`
releases with no waiting, which is how tests and offline evaluation run.
Options: `transcode=True` re-encodes with a keyframe every 2 s for files whose
keyframes are far apart (stream copy can only split on keyframes, so such files
otherwise give long, uneven segments); `drop={seq,...}` and `jitter` simulate
lost and late segments.

**`HlsSource(playlist)`**: polls an HLS media playlist (local path or
http/https URL), yields each new segment once, and flags a discontinuity when
segments scrolled out of the playlist before they were fetched or when the
playlist says so. **`hls_sim`** turns a file into such a playlist, live, using
FFmpeg's HLS muxer. Together they exercise the exact shape of feed Twitch
serves, so a later Twitch source only has to supply a playlist URL and ad
handling.

### 6.2 SegmentBuffer

A per-channel directory plus an in-memory index. The buffer owns segment files
and deletes them when they age out.

- `add(segment)`: index it, then evict the oldest segments beyond `retention_s`.
- `lease(core_start, core_end, pre, post) -> Lease`: returns the segments
  covering the core, extended back by up to `pre` and forward by up to `post`
  seconds, stopping at any discontinuity. Leased segments are reference-counted
  and are not deleted until released, even if they age out meanwhile. If the
  head has already been evicted the lease is returned shorter with
  `truncated=True`; a clip is never failed for a missing head.
- `latest_end`, `oldest_start`.
- Thread-safe: `add` is called from the ingest thread, `lease`/release from
  workers.

Retention is validated at start-up: it must be at least
`max_len_s + pre_pad_s + post_pad_s + merge_gap_s + 2 × segment duration`.
At 6–8 Mbps, the default 120 s is about 90–120 MB of disk per channel.

### 6.3 Loudness extraction

`LoudnessExtractor.feed(segment) -> list[LoudnessSample]` decodes only the audio
stream with PyAV, resamples to 16 kHz mono, and computes RMS dBFS per 100 ms
hop.

- Sample timestamps are derived from `segment.start` plus a running sample
  count, not from packet timestamps (see section 2).
- Samples left over at the end of a segment are carried into the next.
- A segment with no decodable audio yields no samples and logs a warning.

`RollingLoudnessStats` is the streaming replacement for the whole-file
`volumedetect` histogram: a histogram of 1 dB bins over the trailing
`stats_window_s`, with O(1) update and a `percentile(p)` query.

### 6.4 Online segmenter

Pure logic with no I/O: `feed(samples) -> list[Window]`, `flush()`,
`reset()` (discontinuity).

- **Threshold.** A sample is loud if `db >= threshold`, where
  `threshold = max(percentile(threshold_percentile), floor_db)` computed from
  the samples *before* it. This mirrors the batch rule, which picks a level
  near the top of the file's loudness distribution, without needing the whole
  file. Until `warmup_s` of audio has been seen, nothing is detected.
- **State machine.** The first loud sample opens a tentative window. The window
  closes once `merge_gap_s` of continuous quiet has passed; its end is the end
  of the last loud sample. This single rule replaces the batch pipeline's
  0.5 s silence duration plus its later 4 s merge.
- **Filters.** Windows shorter than `min_len_s` are discarded. A window reaching
  `max_len_s` is force-closed and a new one may open immediately.
- **Determinism.** The result must not depend on how samples are batched across
  `feed` calls.

The percentile rule will mark some fraction of any stream as loud, including a
quiet one. `floor_db` and the recorded `prominence_db` (peak minus rolling
median) exist to control that; the right values come out of the evaluation
harness, not this document.

### 6.5 ClipWriter

`write(window, lease) -> Clip`:

- Builds an FFmpeg concat list in a per-job temporary file. (The batch code
  writes `tmp_file.txt` in the working directory, which would collide between
  concurrent jobs.)
- Stream-copies to mp4 with `-bsf:a aac_adtstoasc -movflags +faststart`.
- Writes to `<name>.mp4.partial`, renames atomically when complete, then writes
  the JSON sidecar. Anything watching the output folder therefore never sees a
  half-written clip, and a sidecar's existence means the clip is complete.
- Cuts fall on segment boundaries, so a file is up to one segment longer than
  requested at each end. The sidecar records `content_start`/`content_end`
  offsets so a later render step can trim exactly while it re-encodes anyway.
- On FFmpeg failure or timeout: no files are left behind and a `ClipError`
  carries the tail of stderr.

Output layout: `<out>/<channel>/<session>/clips/<channel>_<start_ms>.mp4` with
`.json` next to it.

### 6.6 VisualScorer

`score(lease, core_start, core_end) -> dict[str, float]`, run in a worker after
the cut has been queued.

- Decodes only frames inside the core window. `mode="keyframes"` (default)
  decodes one frame per keyframe; `mode="sampled"` decodes everything and
  scores `sample_fps` frames per second.
- Per frame: crop the profile's regions, then the two existing measures, binary
  threshold white-pixel percentage and Canny white-pixel percentage, combined
  with the existing weights (2.0 and 0.5).
- Per clip: `visual_mean`, `visual_p90`, `visual_max`. Scores are normalised by
  the number of frames, so a long clip does not outrank a short one by length.
- Regions are stored as fractions of the frame in a `GameProfile`, so they are
  independent of resolution. The default profile is the centre crosshair box.

This is written fresh in `scoring/visual.py`. It does not import
`aggregate/opencv_component.py`, which is left alone (see section 10). Whether
this score actually predicts a good clip is unknown; the harness is how that
gets answered.

### 6.7 Ranking metadata

No clip is withheld in this phase. Each clip gets:

- `audio_prominence` from the window stats, and the visual scores.
- `rank_score`: a weighted sum of each score's z-value against the session's
  running mean and variance.
- `rank_percentile`: where that score sits among the session's clips so far.

An optional `min_percentile` setting deletes clips below a bar once scored. It
is off by default; the consumer decides what to keep.

### 6.8 SessionLog and latency

One `events.jsonl` per channel session, one JSON object per line, appended
thread-safely and flushed per line. Every event has `type`, `t_media` and
`t_wall`.

| Event | Content |
|---|---|
| `session_start` / `session_end` | config snapshot; totals |
| `segment` | seq, start, duration, discontinuity |
| `audio` | the loudness samples for one segment |
| `window` | the closed window and its stats |
| `clip` | path, file timing, cut time |
| `score` | window id and scores |
| `dropped` | what was dropped and why |

`LatencyTracker` reports, per clip and as p50/p95 for the session:

- `detect_latency`: from the arrival of the segment containing the window's end
  to the window being closed.
- `cut_latency`: from window closed to clip file complete.
- `ingest_lag`: from a segment's release to the ingest loop finishing with it.

### 6.9 ChannelPipeline, WorkerPool, Supervisor

One ingest thread per channel runs this loop:

```
for segment in source:
    buffer.add(segment)
    for window in detector.feed(segment):      # audio decode + segmenter
        pending.append(window)
    for window in pending that the buffer now covers through end + post_pad:
        pool.submit(cut job)      # takes its own lease
        pool.submit(score job)    # takes its own lease
at end of stream: detector.flush(), submit everything pending, drain the pool.
```

- The worker pool is a shared thread pool. FFmpeg runs as a subprocess and
  PyAV, OpenCV and NumPy release the interpreter lock, so threads are enough.
- The job queue is bounded. When it is full, scoring jobs are dropped first
  (the clip is still cut, with empty scores) and a `dropped` event is logged.
- A failing job is logged and never takes down the channel. A failing channel
  never takes down the supervisor.
- Shutdown (end of stream or Ctrl-C): close sources, flush detectors, drain the
  pool with a timeout, write `session_end`.

The supervisor is one process with one thread per channel for now. Its
interface (`add_channel`, `remove_channel`, `run`, `stop`) does not expose
that, so it can move to one process per channel if a benchmark says so.

## 7. Time-to-clip budget

With the defaults below, after a moment's last loud sample:

| Step | Time |
|---|---|
| Wait for `merge_gap_s` of quiet to confirm the moment ended | 4 s |
| Wait for the segment holding that quiet to finish and arrive | 0–2 s |
| Cut | under 1 s |
| **Clip on disk** | **about 5–7 s** |

`merge_gap_s` is the dominant term and the main speed/quality dial: lower it
and clips arrive sooner but a moment with a pause in it gets split. Against a
real Twitch feed, add the broadcast delay on top; that part cannot be reduced
from this side.

## 8. Configuration defaults

| Setting | Default | Meaning |
|---|---|---|
| `segment_time_s` | 2.0 | target segment length for simulated feeds |
| `retention_s` | 120 | rolling buffer length |
| `hop_s` | 0.1 | loudness resolution |
| `stats_window_s` | 600 | history used for the adaptive threshold |
| `threshold_percentile` | 90 | loudness percentile that counts as loud |
| `floor_db` | -35 | threshold never goes below this |
| `warmup_s` | 30 | no detection before this much audio |
| `merge_gap_s` | 4.0 | quiet needed to close a window (from the batch code) |
| `min_len_s` | 1.5 | shortest window kept (from the batch code) |
| `max_len_s` | 45 | longest window before a forced close |
| `pre_pad_s` / `post_pad_s` | 5 / 3 | padding around the window |
| `visual_mode` | `keyframes` | or `sampled` with `sample_fps` |
| `workers` | 4 | worker pool size |
| `min_percentile` | off | optional cull |

`threshold_percentile`, `floor_db` and the paddings are starting guesses.

## 9. Testing and evaluation

- **Synthetic fixture.** A 60 s, 320×180 video generated by FFmpeg with a quiet
  tone and loud bursts at known times. The expected windows are known exactly,
  so tests assert on them.
- **Unit tests** per component; the segmenter is tested from plain arrays of dB
  values with no media at all.
- **End-to-end.** The fixture through the full pipeline unpaced must yield
  exactly the expected windows and valid clip files; the same run at 10x must
  yield the identical window list (principle 2).
- **Oracle comparison.** For any real video, run FFmpeg's whole-file
  `silencedetect` with the batch pipeline's rules to get reference windows, and
  report the online detector's precision, recall and boundary error against
  them.
- **Re-tune.** Re-run the segmenter over a recorded `events.jsonl` with
  different settings, no video needed.
- **Benchmark.** N simulated channels at 1x for M seconds; report CPU, ingest
  lag and latency percentiles. This answers how many channels one machine can
  follow.

## 10. Known problems in the existing code

The real-time package does not depend on these, and this phase does not fix
them. Listed so nobody copies them.

- `opencv_component.py`: `crop_image_crosshair` indexes rows with the x centre
  and columns with the y centre, so on a 1920×1080 frame it scores a patch in
  the lower left, not the crosshair.
- `opencv_component.py`: the two `get_*_white_percentage` methods call
  `change_color_to_BRG2GRAY`, which does not exist.
- `analyze_clip_stage.py`: `if not frame_count % 30: continue` skips every 30th
  frame and processes the other 29.
- `analyze_clip_stage.py`: scores are summed, so longer clips always score
  higher.
- `cli.py`: the `-c` path constructs `DownloadPipe` but never runs it.
- `pyproject.toml`: `platformdirs` is imported but not declared, and the exact
  pins (for example `numpy 1.26.4`, `numba 0.60.0`) do not install on current
  Python versions.

## 11. Later phases, for orientation only

1. Twitch source: resolve a channel to a playlist URL, skip ad segments, detect
   go-live and end of stream.
2. Chat as a second `Detector`; windows from several detectors merged before
   cutting.
3. Ground truth from viewer-made clips; a learned ranker fed by post
   performance.
4. Per-game `GameProfile`s chosen from the stream's category.

## Implementation notes (where the built code differs from this document)

The code is authoritative. These points were found while building and supersede the sections above.

- **Threshold hysteresis (6.4).** While a window is open, samples are judged against the threshold frozen when it opened. Without this a sustained loud moment raises the rolling percentile to its own level and cuts itself short.
- **Percentile convention (6.3/6.4).** The segmenter uses its own histogram whose percentile is the lower edge of the 1 dB bin, so a sample at the threshold's own level always counts as loud. `audio.RollingLoudnessStats` (numpy-style interpolation) exists but the pipeline does not use it for detection.
- **Threshold rule (6.4), revised.** `threshold = max(weighted p90 of the background + rise_db, floor_db)` over `stats_window_s`. Samples inside detected windows count `window_weight` (0.25) instead of 1, so highlights barely raise the bar; `rise_db` (5) means "clearly above the loud end of normal". A window that runs to `max_len` enters the background at full weight and is re-judged against the new level. Known limits: highlights filling more than about 31% of the history become "normal" (a scene with peaks ~48% of the time detects nothing); quieter highlights are missed for several minutes after a long loud stretch; the first ~2 min of speech after a silent start is over-reported.
- **Warm-up (6.4), revised.** The first `warmup_s` (30) are buffered, not skipped: when the buffer fills, one threshold is chosen with hindsight and the buffered samples are run through the detector. Warm-up windows therefore come out up to ~30 s late. `validate` requires `retention_s >= warmup_s + pre_pad_s + 2 segments`.
- **Sample timing (6.3).** Loudness samples are stamped from an anchor (the first segment's start) plus a running sample count, giving an exact 0.1 s grid; the anchor resets on discontinuity or when drift exceeds 0.25 s. Levels are RMS over all channels (not a mono downmix, which reads 3 dB hot).
- **Segmenting (6.1).** `-segment_time_delta` is needed or a keyframe exactly on a boundary is missed. AAC priming makes media time run about 21 ms behind source time on the fixture.
- **Clip writing (6.5).** FFmpeg exits 0 on a corrupt or missing segment, so the writer passes `-xerror` and probes the output duration after each cut. It maps video and audio only (`-dn`), because HLS segments can carry ID3 data streams that MP4 cannot hold. Temp files are `<name>.mp4.<token>.partial`.
- **Visual scoring (6.6).** Sampled mode on 1080p60 costs about 2.9 s per 20 s clip (the spike's 1.1 s was optimistic); keyframes mode costs about 0.2 s but yields one frame per source keyframe interval, so short windows on a 2 s keyframe interval get only 2–3 frames.
- **Job queue (6.8).** When full, the oldest queued score job is dropped first; a cut is dropped only if no score job is queued. Ingest never waits. A score job waits for its own cut.
- **Additions.** `RealtimeConfig` gained `sample_fps` (4.0), `queue_size` (64), `clip_timeout_s` (30), `rise_db` (5.0), `window_weight` (0.25), `queue_per_channel` (4) and `channels_per_process` (20). The session log has an `error` event.
- **Measured time-to-clip (7).** About 4.2 s at 1x on the fixture (4.0 s merge gap + ~0.2 s cut), under the 5–7 s estimate.
- **Capacity and sharding (6.9), revised.** One process is capped by the interpreter lock at about 1.8–2.1 cores. `shard.ShardedSupervisor` (chosen by `make_supervisor`, `--procs`) runs `channels_per_process` (20) channels per spawned worker process, each an ordinary `Supervisor`; sources must be given as picklable factories. If a worker dies its channels are reported failed, partial clips are removed, their logs are closed, and the other workers continue. Measured on 4 cores, 1080p60, staggered starts: one process keeps up at 48 channels and falls 22–24 s behind at 64; four processes keep up at 64 with p95 time-to-clip 4.6 s (8.9 s with synchronized starts). `watch` shards the same way (`--procs`); Twitch channels are given as picklable `TwitchFactory`s, and each channel's source statistics come back in `ChannelResult.source` (also in `session_end`). On Windows a dead worker's FFmpeg children are not killed with it (no process groups).
- **Queue sizing.** The bound is `max(queue_size, queue_per_channel x channels in the process)`; `queue_size` is a floor.
- **Behind schedule (6.8).** `ingest_lag` cannot show a backlog with pull-based sources. Each `segment` event now carries `due_at` (`ScheduleClock`: from the source's `t0`, else the earliest arrival, re-anchored on discontinuity); the tracker reports `behind_schedule` and `time_to_clip` from it. A live source that stalls without a discontinuity reads as behind.
- **Benchmark (9).** Channels start staggered over `min(20, seconds/3)` s by default (`--stagger 0` is the synchronized worst case); each plays `seconds x speed` of stream.
- **Batch threshold rule.** The legacy rule (upper edge of the loudest well-populated bin) finds nothing when applied literally; the `eval` reference uses the lower edge and a corrected merge.
- **Twitch source (6.1, 11).** `TwitchSource` resolves a channel or VOD to its HLS media playlist through streamlink (optional `twitch` extra, H.264/MPEG-TS only) and reads segments with `HlsSource`; no official API is used. `HlsSource` skips stitched ad segments (flagging a discontinuity), re-resolves on 403/404/410 or stall (live tokens last ~20 min), ends cleanly when offline, handles complete VOD playlists with start/end offsets and `-muted` segments, polls at half a segment, and rejects fMP4/encrypted/byte-range playlists. Commands: `vidflow-live watch CHANNEL...`, `vidflow-live vod URL --start --end`. Default quality is `best` (source, ~6–9 Mbps per channel). `watch --wait` polls an offline channel every `--offline-poll` (60 s) until it goes live and, after a broadcast ends (offline or stall), waits for the next; broadcasts are chained in one source (`seq` and media time continue, the first segment of each later broadcast is a discontinuity), so a moment open when a broadcast ends is cut when the next one starts, and the command runs until `--duration` or Ctrl-C. Without `--wait` an offline channel ends at once with a hint.
- **Twitch network hosts.** `gql.twitch.tv`, `usher.ttvnw.net`, `*.playlist.ttvnw.net` (live playlists), `*.hls.live-video.net` (live segments), `*.cloudfront.net` (VODs). The edge hosts rotate, so they must be allowed by pattern.
- **Real-footage check.** A 10-minute slice of a real 1080p60 VOD ran end to end in 64 s: 51 ten-second segments, 18 windows, 18 clips, no errors. VOD clips are cut on 10 s segment edges, so files run up to ~10 s long at each end (`content_start`/`content_end` in the sidecar give the exact range). A live run reached the playlist but not the segments (edge host not allowed in the test sandbox); a stream that ends (stall, offline, end of list) while its last 3 or more segment downloads in a row failed now raises `SegmentFetchError`, so the channel is reported FAILED with the last download error and `watch` exits 1.
