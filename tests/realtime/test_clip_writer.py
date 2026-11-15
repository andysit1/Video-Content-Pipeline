"""Tests for ClipWriter (task T5, design section 6.5)."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Sequence

import av
import numpy as np
import pytest

from VidFlow.realtime.clip_writer import (
    ClipWriter,
    clip_to_dict,
    read_sidecar,
    sidecar_path,
    update_sidecar,
)
from VidFlow.realtime.media import probe
from VidFlow.realtime.testing import SYNTHETIC_FPS
from VidFlow.realtime.types import Clip, ClipError, Segment, Window, make_window_id

FRAME = 1.0 / SYNTHETIC_FPS


@dataclasses.dataclass
class StubLease:
    """Stand-in for buffer.Lease: the writer only needs ordered ``segments``."""

    segments: List[Segment]
    truncated: bool = False

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def end(self) -> float:
        return self.segments[-1].end


def make_window(start: float, end: float, channel: str = "test") -> Window:
    return Window(
        channel=channel,
        window_id=make_window_id(channel, start),
        start=start,
        end=end,
        detector="audio",
        stats={"peak_db": -4.9, "mean_db": -5.2, "threshold_db": -30.0, "prominence_db": 40.0},
    )


def files_in(directory: Path) -> List[str]:
    if not directory.exists():
        return []
    return sorted(p.name for p in directory.rglob("*") if p.is_file())


def stream_start_times(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=start_time:stream=codec_type,start_time",
            "-of", "json", str(path),
        ],
        stdout=subprocess.PIPE, check=True,
    ).stdout
    data = json.loads(out)
    result = {"format": float(data["format"]["start_time"])}
    for s in data["streams"]:
        result[s["codec_type"]] = float(s["start_time"])
    return result


def loud_span(path: Path, level: float = 0.3) -> tuple:
    """(first, last) time in the file where the audio amplitude exceeds ``level``.

    Times come from the decoded sample count from the first audio frame, offset
    by that frame's own timestamp, i.e. file time.
    """
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        chunks = []
        first_t = None
        for frame in container.decode(stream):
            if first_t is None:
                first_t = float(frame.pts * frame.time_base)
            chunks.append(np.abs(frame.to_ndarray()).max(axis=0))
        rate = stream.rate
    amp = np.concatenate(chunks)
    loud = np.flatnonzero(amp > level)
    assert loud.size, "no loud audio in clip"
    return first_t + loud[0] / rate, first_t + (loud[-1] + 1) / rate


# --------------------------------------------------------------------------- basic cut


def test_clip_from_segments_5_to_9(synthetic_segments: Sequence[Segment], tmp_path: Path) -> None:
    lease = StubLease(list(synthetic_segments[5:10]))
    tmp_lists = tmp_path / "lists"
    writer = ClipWriter(tmp_path / "out", "sess1", tmp_dir=tmp_lists)
    window = make_window(11.0, 17.0)

    clip = writer.write(window, lease, pre_pad=1.0, post_pad=1.0)

    total = sum(s.duration for s in lease.segments)
    assert clip.path == tmp_path / "out" / "test" / "sess1" / "clips" / "test_000011000.mp4"
    assert clip.path.is_file()
    assert sidecar_path(clip.path).is_file()
    assert clip.file_start == pytest.approx(lease.segments[0].start)
    assert clip.file_duration == pytest.approx(total)
    assert dict(clip.scores) == {}

    info = probe(clip.path)
    assert info.has_video and info.has_audio
    assert abs(info.duration - total) <= 0.1

    with av.open(str(clip.path)) as c:
        assert len(c.streams.video) == 1
        assert len(c.streams.audio) == 1

    # Only the mp4 and its sidecar; the per-job concat list is gone.
    assert files_in(tmp_path / "out") == ["test_000011000.json", "test_000011000.mp4"]
    assert files_in(tmp_lists) == []


def test_output_starts_at_zero(synthetic_segments: Sequence[Segment], tmp_path: Path) -> None:
    # Container timestamps in the segments carry a muxer offset (about 1.4 s);
    # the clip must not.
    writer = ClipWriter(tmp_path, "s")
    clip = writer.write(make_window(21.0, 25.0), synthetic_segments[10:14])
    starts = stream_start_times(clip.path)
    assert starts["format"] == pytest.approx(0.0, abs=1e-3)
    assert starts["video"] == pytest.approx(0.0, abs=1e-3)
    assert 0.0 <= starts["audio"] < FRAME


def test_content_offsets_point_at_window(synthetic_segments: Sequence[Segment], tmp_path: Path) -> None:
    # The fixture's first burst is loud over [10, 16) of media time.
    lease = StubLease(list(synthetic_segments[4:10]))
    window = make_window(10.0, 16.0)
    pre, post = 1.0, 2.0
    clip = ClipWriter(tmp_path, "s").write(window, lease, pre_pad=pre, post_pad=post)

    assert clip.content_start == pytest.approx(window.start - pre - lease.start)
    assert clip.content_end == pytest.approx(window.end + post - lease.start)

    loud_first, loud_last = loud_span(clip.path)
    assert abs((clip.content_start + pre) - loud_first) <= FRAME
    assert abs((clip.content_end - post) - loud_last) <= FRAME


def test_content_offsets_are_clamped_to_file(synthetic_segments: Sequence[Segment], tmp_path: Path) -> None:
    # Head evicted: the lease starts after window.start - pre_pad.
    lease = StubLease(list(synthetic_segments[5:8]), truncated=True)
    window = make_window(10.0, 16.0)
    clip = ClipWriter(tmp_path, "s").write(window, lease, pre_pad=5.0, post_pad=3.0)
    assert clip.content_start == 0.0
    assert clip.content_end == pytest.approx(clip.file_duration)
    side = json.loads(sidecar_path(clip.path).read_text(encoding="utf-8"))
    assert side["truncated"] is True
    assert side["segments"] == [5, 6, 7]


# --------------------------------------------------------------------------- sidecar


def test_sidecar_holds_every_field_and_round_trips(synthetic_segments: Sequence[Segment], tmp_path: Path) -> None:
    window = dataclasses.replace(make_window(41.0, 45.5), forced=True)
    clip = ClipWriter(tmp_path, "s").write(window, StubLease(list(synthetic_segments[19:24])), 2.0, 1.0)
    data = json.loads(sidecar_path(clip.path).read_text(encoding="utf-8"))

    for f in dataclasses.fields(Clip):
        assert f.name in data
    for f in dataclasses.fields(Window):
        assert f.name in data["window"]
    assert data["file"] == clip.path.name
    assert data["window"]["forced"] is True
    assert data["window"]["stats"]["prominence_db"] == pytest.approx(40.0)
    assert data["session"] == "s"
    assert data["pre_pad"] == 2.0 and data["post_pad"] == 1.0
    assert data["segments"] == [19, 20, 21, 22, 23]
    assert data["cut_s"] > 0

    assert read_sidecar(sidecar_path(clip.path)) == clip
    assert read_sidecar(clip.path) == clip  # the mp4 path works too


def test_update_sidecar_attaches_scores_and_keeps_extras(
    synthetic_segments: Sequence[Segment], tmp_path: Path
) -> None:
    clip = ClipWriter(tmp_path, "s").write(make_window(11.0, 15.0), synthetic_segments[5:8])
    scored = dataclasses.replace(clip, scores={"visual_mean": np.float64(1.5), "visual_max": 3.0})
    update_sidecar(scored, extra={"rank_percentile": 80.0})

    data = json.loads(sidecar_path(clip.path).read_text(encoding="utf-8"))
    assert data["scores"] == {"visual_mean": 1.5, "visual_max": 3.0}
    assert data["rank_percentile"] == 80.0
    assert data["segments"] == [5, 6, 7]  # kept from the original write
    assert read_sidecar(clip.path).scores == {"visual_mean": 1.5, "visual_max": 3.0}
    assert files_in(clip.path.parent) == sorted([clip.path.name, sidecar_path(clip.path).name])

    with pytest.raises(ValueError):
        update_sidecar(scored, extra={"content_start": 0.0})


def test_non_finite_stats_serialise_as_null(tmp_path: Path) -> None:
    window = dataclasses.replace(make_window(1.0, 2.0), stats={"peak_db": float("-inf")})
    clip = Clip(window, tmp_path / "x.mp4", 0.0, 2.0, 0.0, 2.0)
    data = clip_to_dict(clip)
    assert data["window"]["stats"]["peak_db"] is None
    json.dumps(data, allow_nan=False)


# --------------------------------------------------------------------------- concurrency


def test_ten_concurrent_writes(synthetic_segments: Sequence[Segment], tmp_path: Path) -> None:
    writer = ClipWriter(tmp_path / "out", "conc", tmp_dir=tmp_path / "lists")

    def job(i: int) -> Clip:
        segs = list(synthetic_segments[i : i + 5])  # overlapping leases on shared files
        window = make_window(segs[1].start + 0.25, segs[3].end - 0.25)
        return writer.write(window, StubLease(segs), pre_pad=1.0, post_pad=1.0)

    with ThreadPoolExecutor(max_workers=10) as pool:
        clips = list(pool.map(job, range(10)))

    paths = {c.path for c in clips}
    assert len(paths) == 10
    for c in clips:
        info = probe(c.path)
        assert info.has_video and info.has_audio
        assert abs(info.duration - c.file_duration) <= 0.1
        assert read_sidecar(c.path) == c
    names = files_in(tmp_path / "out")
    assert len(names) == 20
    assert not any(n.endswith(".partial") for n in names)
    assert files_in(tmp_path / "lists") == []


# --------------------------------------------------------------------------- failures


def _assert_clean(out: Path, lists: Path) -> None:
    assert files_in(out) == []
    assert files_in(lists) == []


@pytest.mark.parametrize("damage", ["random", "garbage", "mid_corrupt", "truncated", "missing"])
def test_corrupt_segment_raises_and_leaves_nothing(
    fresh_synthetic_segments: Sequence[Segment], tmp_path: Path, damage: str
) -> None:
    segs = list(fresh_synthetic_segments[5:10])
    victim = segs[2].path
    data = victim.read_bytes()
    if damage == "random":
        victim.write_bytes(os.urandom(len(data)))
    elif damage == "garbage":
        victim.write_bytes(b"not a transport stream")
    elif damage == "mid_corrupt":
        n = len(data)
        victim.write_bytes(data[: n // 3] + os.urandom(n // 2) + data[n // 3 + n // 2 :])
    elif damage == "truncated":
        victim.write_bytes(data[: len(data) // 3])
    elif damage == "missing":
        victim.unlink()

    out, lists = tmp_path / "out", tmp_path / "lists"
    writer = ClipWriter(out, "s", tmp_dir=lists)
    with pytest.raises(ClipError) as info:
        writer.write(make_window(11.0, 17.0), StubLease(segs), 1.0, 1.0)
    assert info.value.message
    if damage in ("random", "mid_corrupt"):
        assert info.value.stderr_tail  # FFmpeg's own complaint is carried along
    _assert_clean(out, lists)


def test_timeout_raises_and_leaves_nothing(synthetic_segments: Sequence[Segment], tmp_path: Path) -> None:
    out, lists = tmp_path / "out", tmp_path / "lists"
    writer = ClipWriter(out, "s", timeout_s=0.001, tmp_dir=lists)
    with pytest.raises(ClipError, match="timed out"):
        writer.write(make_window(11.0, 17.0), synthetic_segments[5:10])
    _assert_clean(out, lists)


def test_bad_leases_are_rejected(synthetic_segments: Sequence[Segment], tmp_path: Path) -> None:
    out, lists = tmp_path / "out", tmp_path / "lists"
    writer = ClipWriter(out, "s", tmp_dir=lists)
    w = make_window(11.0, 17.0)
    with pytest.raises(ClipError, match="no segments"):
        writer.write(w, StubLease([]))
    segs = list(synthetic_segments[5:9])
    with pytest.raises(ClipError, match="discontinuity"):
        writer.write(w, segs[:2] + [dataclasses.replace(segs[2], discontinuity=True)] + segs[3:])
    with pytest.raises(ClipError, match="out of order"):
        writer.write(w, [segs[1], segs[0]])
    with pytest.raises(ClipError, match="not contiguous"):
        writer.write(w, [segs[0], segs[2]])
    _assert_clean(out, lists)


def test_discontinuity_on_first_segment_is_fine(synthetic_segments: Sequence[Segment], tmp_path: Path) -> None:
    # The lease starts right after a gap; nothing inside the clip jumps.
    segs = [dataclasses.replace(synthetic_segments[5], discontinuity=True)] + list(synthetic_segments[6:9])
    clip = ClipWriter(tmp_path, "s").write(make_window(11.0, 15.0), segs, 5.0, 1.0)
    assert clip.content_start == 0.0
    assert clip.path.is_file()


# --------------------------------------------------------------------------- paths


def test_awkward_paths_are_quoted(synthetic_segments: Sequence[Segment], tmp_path: Path) -> None:
    src = tmp_path / "it's a dir" / "seg files"
    src.mkdir(parents=True)
    segs = []
    for s in synthetic_segments[5:8]:
        target = src / f"o'neil {s.seq}.ts"
        shutil.copyfile(s.path, target)
        segs.append(dataclasses.replace(s, path=target))
    out = tmp_path / "out dir with 'quote'"
    window = make_window(11.0, 15.0, channel="Some:Chan/nel")
    clip = ClipWriter(out, "sess?1").write(window, segs)
    assert clip.path.is_file()
    assert clip.path.parent == out / "Some_Chan_nel" / "sess_1" / "clips"
    assert clip.path.name == "Some_Chan_nel_000011000.mp4"
    assert abs(probe(clip.path).duration - clip.file_duration) <= 0.1


def test_rewriting_the_same_window_replaces_the_clip(
    synthetic_segments: Sequence[Segment], tmp_path: Path
) -> None:
    writer = ClipWriter(tmp_path, "s")
    w = make_window(11.0, 15.0)
    first = writer.write(w, synthetic_segments[5:8])
    second = writer.write(w, synthetic_segments[5:9])
    assert first.path == second.path
    assert read_sidecar(second.path) == second
    assert len(files_in(tmp_path)) == 2


def test_cut_is_fast(synthetic_segments: Sequence[Segment], tmp_path: Path) -> None:
    # 20 s of the 320x180 fixture; the design budget for a cut is under 1 s.
    clip = ClipWriter(tmp_path, "s").write(make_window(12.0, 28.0), synthetic_segments[5:15], 2.0, 2.0)
    cut_s = json.loads(sidecar_path(clip.path).read_text(encoding="utf-8"))["cut_s"]
    assert math.isfinite(cut_s) and cut_s < 1.0
