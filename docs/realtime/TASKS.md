# Real-time clipper: tasks

Each task is written to be handed to one agent. Read [DESIGN.md](./DESIGN.md)
first; section numbers below refer to it.

## Order

| Wave | Tasks | Notes |
|---|---|---|
| 1 | T0 | Blocks everything. Fixes the shared contract. |
| 2 | T1, T2, T3, T4, T5, T6, T7, T10 | Independent of each other; run in parallel. |
| 3 | T8 | Integration. Needs T1–T7. |
| 4 | T9 | Evaluation and benchmark. Needs T8. |

## Rules for every task

- Work only in the files listed under **Owns**. If the contract in
  `types.py` or `config.py` needs to change, stop and report instead of
  editing it.
- No network access to Twitch and no credentials. Nothing in this phase needs
  them.
- Do not modify the existing batch pipeline (`pipes/`, `modules/`,
  `aggregate/`, `cli.py`).
- Logic uses media time only (design principle 2). Wall-clock time appears only
  in metrics and in source pacing.
- Must run on Windows as well as Linux: `pathlib` for paths, no symlinks or
  hard links, no shell-specific commands, quote paths in FFmpeg list files.
- Standard `logging`, type hints, no `icecream`, no `print` outside the CLI.
- Tests use `pytest`, live under `tests/realtime/`, and need only FFmpeg on the
  `PATH`. A task's own tests should finish in under 30 s; anything paced in
  real time is marked `@pytest.mark.slow`.
- Finish by reporting: what was built, test output, anything in the design that
  turned out wrong, and any measured timings.

---

## T0: Foundation and contract

**Depends on:** nothing.
**Owns:** `pyproject.toml`, `src/VidFlow/realtime/__init__.py`, `types.py`,
`config.py`, `media.py`, `testing.py`, `tests/realtime/conftest.py`,
`tests/realtime/test_foundation.py`.

**Build**

1. `types.py`: `Segment`, `LoudnessSample`, `Window`, `Clip`, `Source`,
   `Detector` exactly as in design section 5, plus `ClipError`.
2. `config.py`: a `RealtimeConfig` dataclass with every setting and default in
   design section 8, and a `validate()` that enforces the retention rule in
   section 6.2.
3. `media.py`:
   - `segment_file(src, out_dir, segment_time, transcode=False) -> list[SegmentInfo]`
     using FFmpeg's segment muxer to MPEG-TS, returning file name, start and
     duration read from the muxer's CSV list. `transcode=True` re-encodes with
     a keyframe forced every `segment_time`.
   - `probe(path)` returning duration and stream presence via `ffprobe`.
4. `testing.py`:
   - `make_synthetic_video(path, duration=60, bursts=((10, 16), (40, 46)))`:
     320×180, 30 fps, keyframe every second, a quiet tone with full-volume
     bursts at the given times.
   - `FakeSource(segments)` implementing `Source`, and
     `make_segments(channel, durations, discontinuities=())` that builds
     `Segment` objects with correct media times.
5. `conftest.py`: session-scoped fixtures `synthetic_video` and
   `synthetic_segments` (the video split into 2 s segments).
6. `pyproject.toml`: reduce dependencies to what the package actually imports
   (`click`, `questionary`, `icecream`, `ffmpeg-python`, `opencv-python`,
   `numpy`, `platformdirs`, `rich`) with lower bounds instead of exact pins;
   add `av`; add `pytest` as a dev dependency; register the `slow` marker.
   The package must install on Python 3.10 through 3.13.

**Done when**

- `pip install -e .` and `pytest tests/realtime` pass on a clean environment.
- Tests cover: media-time arithmetic in `make_segments`; `segment_file` on the
  fixture gives contiguous segments whose durations sum to the video's duration
  within 0.1 s; `validate()` rejects a retention that is too small.

---

## T1: ReplaySource (the simulated live feed)

**Depends on:** T0.
**Owns:** `src/VidFlow/realtime/sources/__init__.py`, `sources/replay.py`,
`tests/realtime/test_replay_source.py`.

**Build** (design section 6.1)

`ReplaySource(video, channel, work_dir, speed=1.0, segment_time=2.0,
transcode=False, drop=(), jitter=0.0, seed=0)` implementing `Source`.

- Splits the file with `media.segment_file`, then yields `Segment`s in order.
- With a numeric `speed`, segment *n* is released at wall time
  `t0 + segment.end / speed`. With `speed=None` there is no waiting.
- Sets `arrived_at` at release.
- `drop`: sequence numbers to withhold; the next yielded segment has
  `discontinuity=True`. Media time stays contiguous as defined in section 5.
- `jitter`: extra random delay per segment, up to this many seconds, seeded.
- `close()` makes a blocked iterator return within 0.2 s, from any thread.

**Done when**

- Unpaced: yields every segment, `seq` increasing, each `start` equal to the
  previous `end`.
- `speed=20` on the 60 s fixture takes 3 s ± 0.5 s, and each segment arrives no
  earlier than its scheduled time.
- `drop={5}` yields one fewer segment, with the discontinuity flag on the
  segment after the gap and contiguous media times.
- `close()` during a wait ends iteration promptly.

---

## T2: SegmentBuffer

**Depends on:** T0.
**Owns:** `src/VidFlow/realtime/buffer.py`,
`tests/realtime/test_buffer.py`.

**Build** (design section 6.2)

`SegmentBuffer(retention_s)` with `add`, `lease(core_start, core_end, pre,
post) -> Lease`, `latest_end`, `oldest_start`, `close()`.

- `Lease` is a context manager exposing `segments`, `start`, `end`,
  `truncated`.
- The buffer deletes a segment's file when it is evicted and not leased, or
  when its last lease is released after eviction. `close()` deletes everything
  not leased.
- Padding never crosses a segment with `discontinuity=True`.
- All methods are safe to call from several threads.

**Done when**

- Eviction keeps total buffered duration within `retention_s` plus one segment.
- A leased segment's file survives eviction and is removed on release.
- Two overlapping leases on the same segments both keep them alive.
- A lease whose head has been evicted returns the remainder with
  `truncated=True`.
- A lease next to a discontinuity stops its padding at the boundary.
- A threaded test (one adder, several leasers) runs without error or leaked
  files.

Tests can use `testing.make_segments` with small dummy files; no media is
needed.

---

## T3: Loudness extraction

**Depends on:** T0.
**Owns:** `src/VidFlow/realtime/audio.py`,
`tests/realtime/test_audio.py`.

**Build** (design section 6.3)

- `LoudnessExtractor(hop_s=0.1, rate=16000)` with
  `feed(segment) -> list[LoudnessSample]` and `reset()`.
  - PyAV, audio stream only, resampled to mono.
  - Timestamps from `segment.start` and a running sample count. Do not use
    packet or frame timestamps.
  - Carry leftover samples across segments; drop them when the segment has
    `discontinuity=True` or on `reset()`.
  - No audio stream or a decode error: return `[]`, log a warning, do not
    raise.
- `RollingLoudnessStats(window_s, hop_s)` with `update(db)`,
  `percentile(p)`, `count`: a histogram of 1 dB bins over the trailing window.

**Done when**

- Over the whole fixture, the number of samples is within 2 of
  `duration / hop_s` (this fails if leftovers are not carried).
- Samples inside the fixture's bursts are at least 25 dB above samples outside.
- The first sample at or above the midpoint level is within 0.15 s of each
  burst's true start.
- Feeding a video-only segment returns `[]`.
- `RollingLoudnessStats.percentile` matches `numpy.percentile` on the same
  window to within 1 dB, including after old samples have aged out.
- Report the measured time per 2 s segment; the spike measured 21 ms on a
  1080p source.

---

## T4: OnlineSegmenter

**Depends on:** T0.
**Owns:** `src/VidFlow/realtime/segmenter.py`,
`tests/realtime/test_segmenter.py`.

**Build** (design section 6.4)

`OnlineSegmenter(channel, config)` with `feed(samples) -> list[Window]`,
`flush() -> list[Window]`, `reset() -> list[Window]`.

- Pure logic, no file or media access. It keeps its own
  `RollingLoudnessStats`-equivalent state; if T3 is not merged yet, implement
  against a minimal local protocol (`update`, `percentile`, `count`) and accept
  the stats object by injection.
- Threshold, warm-up, open/close rule, `min_len_s`, `max_len_s` as specified.
- `Window.stats` holds `peak_db`, `mean_db`, `threshold_db` (at open) and
  `prominence_db` (peak minus the rolling median at close).
- `flush()` and `reset()` close an open window with `forced=True` if it meets
  `min_len_s`.

**Done when** (all tested from plain dB arrays)

- One burst gives one window with start and end within one hop of the truth.
- Two bursts 3 s apart merge; 5 s apart do not.
- A 1 s burst is discarded.
- A 100 s burst is split at `max_len_s` with `forced=True` on the first part.
- Nothing is emitted during warm-up.
- A window is not emitted until `merge_gap_s` of quiet has been fed, and is
  emitted by the sample that completes it.
- Feeding the same series one sample at a time, in chunks of 20, and all at
  once gives identical windows.
- Constant low-level noise below `floor_db` gives no windows.

---

## T5: ClipWriter

**Depends on:** T0. Uses the `Lease` shape from T2; develop against a small
stand-in with `segments`, `start`, `end` if T2 is not merged.
**Owns:** `src/VidFlow/realtime/clip_writer.py`,
`tests/realtime/test_clip_writer.py`.

**Build** (design section 6.5)

`ClipWriter(out_dir, session, timeout_s=30)` with
`write(window, lease, pre_pad, post_pad) -> Clip`.

- Per-job concat list in a unique temporary file, removed afterwards.
- Stream copy to mp4, `.partial` then atomic rename, then the JSON sidecar
  containing every field of `Clip` and `Window`.
- `content_start`/`content_end` computed from the window, the padding and the
  lease's actual start.
- Failure or timeout raises `ClipError` with the tail of FFmpeg's stderr and
  leaves no `.partial`, mp4 or sidecar behind.

**Done when**

- A clip from fixture segments 5–9 probes as valid with one video and one audio
  stream and a duration within 0.1 s of the segments' total.
- Output starts at timestamp 0.
- `content_start`/`content_end` point at the window within one frame.
- Ten concurrent writes to the same output directory all succeed with distinct
  files.
- A lease containing a corrupt segment raises `ClipError` and leaves the
  directory clean.
- Report the cut time; the spike measured 0.36 s for 20 s of 1080p60.

---

## T6: VisualScorer

**Depends on:** T0. Same note on `Lease` as T5.
**Owns:** `src/VidFlow/realtime/scoring/__init__.py`, `scoring/visual.py`,
`tests/realtime/test_visual.py`.

**Build** (design section 6.6)

- `GameProfile(name, regions, weights)` where each region is
  `(x0, y0, x1, y1)` as fractions of the frame; a default profile with a centre
  box equivalent to 150×150 px on a 1080p frame.
- Pure functions `frame_score(gray_frame, profile) -> float` using binary
  threshold (170) white percentage × 2.0 plus Canny (200, 300) white
  percentage × 0.5.
- `VisualScorer(profile, mode="keyframes", sample_fps=4)` with
  `score(lease, core_start, core_end) -> dict` returning `visual_mean`,
  `visual_p90`, `visual_max`, `frames`.
  - Frame times come from the segment's media start plus the frame's offset
    from the first frame in that segment.
  - Only frames inside the core are scored.
- Do not import or modify `aggregate/opencv_component.py`.

**Done when**

- A frame white in the centre scores higher than an all-black frame, which
  scores 0.
- On a non-square frame with white only in the true centre, the score is high;
  with white only in the lower left, it is 0. (This is the regression test for
  the axis bug in design section 10.)
- The same content at 1080p and 360p gives region crops covering the same
  fraction of the frame.
- `keyframes` mode on the fixture scores one frame per second of core;
  `sampled` mode scores `sample_fps` per second within one frame.
- A 20 s clip and a 5 s clip of identical content get the same `visual_mean`.
- Report decode time for both modes.

---

## T7: SessionLog and latency

**Depends on:** T0.
**Owns:** `src/VidFlow/realtime/session.py`,
`tests/realtime/test_session.py`.

**Build** (design section 6.8)

- `SessionLog(dir)` with `emit(type, t_media, **fields)`, `close()`; one JSON
  object per line, flushed per line, safe from several threads, adds `t_wall`.
- `read_events(path, type=None)` iterator.
- `LatencyTracker` fed by the same events, exposing per-clip
  `detect_latency`, `cut_latency`, per-segment `ingest_lag`, and a `summary()`
  with count, p50 and p95 of each.
- Paths and NumPy scalars must serialise without the caller converting them.

**Done when**

- 8 threads emitting 1,000 events each produce 8,000 valid lines.
- A reader opened while the log is still being written sees only complete
  lines.
- `LatencyTracker.summary()` matches hand-computed values on a scripted
  sequence of events with injected clock values.
- `audio` events round-trip: samples written can be read back as
  `LoudnessSample`s.

---

## T8: Integration (pipeline, supervisor, CLI)

**Depends on:** T1–T7.
**Owns:** `src/VidFlow/realtime/pipeline.py`, `supervisor.py`, `cli.py`,
`tests/realtime/test_pipeline.py`, `tests/realtime/test_e2e.py`; may add one
`vidflow-live` line to `[tool.poetry.scripts]` in `pyproject.toml`.

**Build** (design sections 6.7 and 6.9)

- `AudioDetector` implementing `Detector` from `LoudnessExtractor` +
  `OnlineSegmenter`, including the discontinuity handling.
- `WorkerPool` with a bounded queue and the drop policy (scoring first).
- `ChannelPipeline(source, config, out_dir)` running the ingest loop in its own
  thread, holding windows until the buffer covers `end + post_pad`, submitting
  cut then score jobs each with its own lease, computing `rank_score` and
  `rank_percentile`, and emitting every event in the session log.
- `Supervisor` with `add_channel`, `remove_channel`, `run`, `stop`; Ctrl-C
  triggers the shutdown sequence.
- CLI: `vidflow-live replay VIDEO [VIDEO ...] [--speed 1|N|max] [--out DIR]
  [--transcode]`, one simulated channel per file, printing each clip as it
  lands and the latency summary at the end.

**Done when**

- Fixture, unpaced: exactly two clips, windows within 0.5 s of (10, 16) and
  (40, 46), each with a valid mp4, a sidecar and scores.
- Fixture at `speed=10`: the list of windows equals the unpaced run's exactly.
- Two channels at once: both produce their clips, into separate directories.
- A source with a dropped segment inside a burst produces clips that do not
  span the gap.
- A cut job forced to fail is logged and the pipeline carries on.
- With one worker and scoring artificially slowed, clips are still cut, a
  `dropped` event is logged, and `ingest_lag` p95 stays under 0.5 s.
- No segment files remain in the buffer directory after shutdown.
- Slow test at `speed=1`: p95 of detect + cut latency is at most
  `merge_gap_s + segment_time_s + 2 s`.

---

## T9: Evaluation and benchmark

**Depends on:** T8.
**Owns:** `src/VidFlow/realtime/eval.py`, CLI subcommands added to `cli.py`
(coordinate with T8's structure, do not restructure it),
`tests/realtime/test_eval.py`.

**Build** (design section 9)

- `vidflow-live eval VIDEO [--truth FILE]`: runs the pipeline unpaced without
  writing clips; builds reference windows from whole-file `silencedetect`
  using the batch pipeline's threshold rule, 4 s merge and 1.5 s minimum (or
  loads them from `--truth`); reports precision and recall at an overlap of
  0.5, and mean start/end error. JSON and a short text table.
- `vidflow-live retune EVENTS.jsonl [--set key=value ...]`: re-runs the
  segmenter over logged `audio` events with changed settings and prints the
  resulting windows. No video access.
- `vidflow-live bench VIDEO --channels N --seconds M`: N simulated channels at
  1x; reports process CPU, peak memory, ingest lag and latency percentiles,
  clips per minute.

**Done when**

- `eval` on the fixture reports precision and recall of 1.0.
- `retune` on a log from the fixture reproduces the original windows with
  unchanged settings, and merges the two bursts when `merge_gap_s` is set above
  their spacing.
- `bench` with 4 channels for 20 s completes and prints all metrics.
- A short written note: channels per core observed, and which setting most
  affects the result.

---

## T10: HLS source and local simulator (bridge to Twitch)

**Depends on:** T0. Independent of T1–T9.
**Owns:** `src/VidFlow/realtime/sources/hls.py`,
`src/VidFlow/realtime/sim/__init__.py`, `sim/hls_sim.py`,
`tests/realtime/test_hls.py`.

**Build** (design section 6.1)

- `hls_sim.start(video, out_dir, speed=1.0, window=6)`: runs FFmpeg's HLS
  muxer as a subprocess producing a sliding-window live playlist with 2 s
  segments that ends with an end-of-list marker; returns a handle with
  `playlist_path` and `stop()`.
- `HlsSource(playlist, channel, work_dir, poll_s=None)` implementing `Source`:
  - `playlist` is a local path or an http/https URL (standard library only).
  - A small parser for the tags needed: segment duration, media sequence,
    discontinuity, end of list.
  - Yields each segment once, copied or downloaded into `work_dir`.
  - Sets `discontinuity=True` when sequence numbers were skipped or the
    playlist marks one.
  - Ends at end-of-list, or after `stall_timeout_s` with no new segment.
- Nothing Twitch-specific and no requests to Twitch.

**Done when**

- Against the simulator at 10x on the fixture: all segments arrive once, in
  order, with total duration within 0.2 s of the video.
- Starting the source late, after segments have scrolled out, yields the rest
  with no error.
- A reader paused longer than the playlist window resumes with a discontinuity
  flagged.
- The parser is tested on literal playlist text, including a discontinuity tag
  and an end-of-list tag.
- A playlist that stops updating ends the iterator after the stall timeout.
