"""
Audio pipeline tests: run the real `silencedetect`/`volumedetect` ffmpeg
filters against a generated video with a known silence pattern, then verify
the regex parsing (analyze_data_stage.py) and chunk-merging logic
(clean_chunks) recover that pattern correctly.
"""
from VidFlow.aggregate.ffmpeg_component import FFMPEGAggregate
from VidFlow.pipes.analyze_data_stage import (
    AnalyzeDataFiles,
    silence_start_re,
    silence_end_re,
)
from VidFlow.pipes.data_cache_stage import DataCachePipe, histogram_re


def _tolerant_eq(actual, expected, tol=0.35):
    return abs(actual - expected) <= tol


def test_silence_detect_finds_known_silent_intervals(fake_video):
    ffmpeg_agg = FFMPEGAggregate(engine=None, debug=True)
    lines = ffmpeg_agg.silence_detect(
        in_filename=str(fake_video["path"]),
        silence_threshold=-30,
        silence_duration=0.4,
    )

    starts = []
    ends = []
    for line in lines:
        m = silence_start_re.search(line)
        if m:
            starts.append(float(m.group("start")))
        m = silence_end_re.search(line)
        if m:
            ends.append(float(m.group("end")))

    assert len(starts) == 2, "expected two silent_start events, got: {}".format(lines)
    expected = fake_video["silent_intervals"]
    for (exp_start, _), actual_start in zip(expected, starts):
        assert _tolerant_eq(actual_start, exp_start), (actual_start, exp_start)

    # the trailing silence runs to EOF so silencedetect may not emit a
    # matching silence_end for it; only assert on the ones it does report.
    for (_, exp_end), actual_end in zip(expected, ends):
        assert _tolerant_eq(actual_end, exp_end), (actual_end, exp_end)


def test_volumedetect_histogram_regex_matches_real_output(fake_video):
    ffmpeg_agg = FFMPEGAggregate(engine=None, debug=True)
    lines = ffmpeg_agg.get_mean_max(in_filename=str(fake_video["path"]))

    histogram_lines = [l for l in lines if histogram_re.search(l)]
    assert histogram_lines, "volumedetect output should contain histogram_XXdb lines: {}".format(lines)

    # the silent half of the clip means a large fraction of samples land in
    # the deepest histogram bucket (close to digital silence).
    matched = [histogram_re.search(l) for l in histogram_lines]
    counts = {int(m.group("db_level")): int(m.group("count")) for m in matched}
    assert max(counts.values()) > 0


def test_get_silence_db_selects_first_bucket_over_the_sample_count_threshold(fake_engine):
    # get_silence_db() picks a threshold by scanning volumedetect's
    # "histogram_NNdb: count" lines (loudest to quietest) for the first
    # bucket whose sample count clears silence_db_range. It's tuned for
    # full-length recordings, where a real silent/quiet stretch piles tens
    # of thousands of samples into one bucket -- a few seconds of generated
    # tone/silence never reaches that count (verified against a real
    # ffmpeg volumedetect run), so this is tested directly against a
    # crafted volume_detect.txt instead of chasing that scale in a fixture.
    pipe = DataCachePipe(engine=fake_engine)
    pipe.silence_db_range = 20000

    pipe.write_lines(pipe.volume_detect, [
        "[Parsed_volumedetect_0] mean_volume: -20.0 dB",
        "[Parsed_volumedetect_0] histogram_16db: 500",
        "[Parsed_volumedetect_0] histogram_17db: 30000",  # first bucket over threshold
        "[Parsed_volumedetect_0] histogram_40db: 90000",
    ])

    assert pipe.get_silence_db() == -17


def test_get_silence_db_falls_back_to_negative_one_when_nothing_clears_the_threshold(fake_engine):
    pipe = DataCachePipe(engine=fake_engine)
    pipe.silence_db_range = 20000

    pipe.write_lines(pipe.volume_detect, [
        "[Parsed_volumedetect_0] histogram_16db: 28",
        "[Parsed_volumedetect_0] histogram_17db: 1160",
    ])

    assert pipe.get_silence_db() == -1


def test_check_data_exist_runs_real_ffmpeg_and_writes_parseable_cache_files(fake_video, fake_engine):
    # smoke test for the real ffmpeg subprocess plumbing: DataCachePipe
    # should always produce readable volume_detect.txt / silence_detect.txt
    # files, whatever threshold get_silence_db() lands on for this clip.
    fake_engine.payload["in_filename"] = str(fake_video["path"])
    pipe = DataCachePipe(engine=fake_engine)

    assert pipe.check_data_exist() is True
    assert pipe.file_exists(pipe.volume_detect)
    assert pipe.file_exists(pipe.silence_detect)
    assert any(histogram_re.search(l) for l in pipe.read_lines(pipe.volume_detect))


def test_is_caster_mode_activated_does_not_crash_on_missing_key(fake_engine):
    # regression test: payload dicts assembled by cli.py never set
    # 'is_caster_mode', which used to raise a KeyError caught as the wrong
    # exception type. It must behave like "caster mode off".
    fake_engine.payload.pop("is_caster_mode", None)
    pipe = DataCachePipe(engine=fake_engine)
    assert pipe.is_caster_mode_activated() is False


def test_analyze_silence_produces_non_silent_chunks(fake_video, fake_engine):
    fake_engine.payload["in_filename"] = str(fake_video["path"])
    cache_pipe = DataCachePipe(engine=fake_engine)

    # write the cache files directly with an explicit, sane threshold
    # instead of going through check_data_exist()'s get_silence_db(): that
    # auto-threshold is scaled for full-length recordings (see the
    # get_silence_db tests above) and isn't meaningful for a few seconds of
    # generated tone/silence. This test is about the regex-parsing +
    # chunk-merging pipeline downstream of silence_detect.txt, not about
    # threshold auto-selection.
    mean_max_lines = cache_pipe.ffmpeg.get_mean_max(fake_engine.payload["in_filename"])
    cache_pipe.write_lines(cache_pipe.volume_detect, mean_max_lines)
    silence_lines = cache_pipe.ffmpeg.silence_detect(
        fake_engine.payload["in_filename"], silence_threshold=-30, silence_duration=0.5
    )
    cache_pipe.write_lines(cache_pipe.silence_detect, silence_lines)

    analyze_pipe = AnalyzeDataFiles(engine=fake_engine)
    analyze_pipe.analyze_silence()

    assert analyze_pipe.if_chunks_txt()
    chunks = analyze_pipe.get_chunk_data()  # regression: used to return None
    assert chunks is not None
    assert len(chunks) >= 1

    # every kept chunk must be well-formed (start < end).
    for start, end in chunks:
        assert end > start

    # clean_chunks() intentionally bridges gaps up to 4s, and our fixture's
    # silences are only 1s long, so the two tone segments are expected to
    # merge into a single chunk rather than stay split around the silence.
    # What must NOT happen: a chunk reaching into the true trailing silence
    # at the very end of the clip (that used to show up as a bogus
    # start > end phantom chunk -- see the analyze_silence regression fix).
    trailing_silence_start = fake_video["silent_intervals"][-1][0]
    for _, end in chunks:
        assert end <= trailing_silence_start + 0.5, (
            "a chunk reached into the clip's trailing silence: {}".format(chunks)
        )
