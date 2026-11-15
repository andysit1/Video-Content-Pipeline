"""Tests for HlsSource and the local live-HLS simulator (task T10)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Sequence

import pytest

from VidFlow.realtime.media import probe
from VidFlow.realtime.sim import hls_sim
from VidFlow.realtime.sources.hls import (
    HlsSource,
    PlaylistError,
    parse_media_playlist,
)
from VidFlow.realtime.testing import copy_segments
from VidFlow.realtime.types import Segment, Source

POLL = 0.05


def _assert_contiguous(segs: Sequence[Segment]) -> None:
    assert segs, "no segments"
    assert segs[0].start == 0.0
    assert [s.seq for s in segs] == list(range(len(segs)))
    for a, b in zip(segs, segs[1:]):
        assert b.start == a.end  # exact: start = prev.start + prev.duration
        assert b.arrived_at >= a.arrived_at
    assert not segs[0].discontinuity


def _write_playlist(path: Path, names: Sequence[str], durations: Sequence[float], *,
                    media_sequence: int = 0, endlist: bool = False,
                    disc_before: Sequence[int] = ()) -> None:
    lines = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-TARGETDURATION:2",
             f"#EXT-X-MEDIA-SEQUENCE:{media_sequence}"]
    for i, (name, d) in enumerate(zip(names, durations)):
        if i in disc_before:
            lines.append("#EXT-X-DISCONTINUITY")
        lines += [f"#EXTINF:{d:.6f},", name]
    if endlist:
        lines.append("#EXT-X-ENDLIST")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------- parser

LIVE_TEXT = """\
#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:2
#EXT-X-MEDIA-SEQUENCE:17
#EXT-X-DISCONTINUITY-SEQUENCE:3
#EXTINF:2.000000,
seg_000017.ts
#EXTINF:1.966667,live
seg_000018.ts
#EXT-X-DISCONTINUITY
#EXTINF:2.033333,
http://127.0.0.1:1/abs/seg_000019.ts
#EXT-X-SOME-UNKNOWN-TAG:foo
#EXTINF:0.5,
seg_000020.ts
#EXT-X-ENDLIST
"""


def test_parser_literal_playlist() -> None:
    pl = parse_media_playlist(LIVE_TEXT)
    assert pl.target_duration == 2.0
    assert pl.media_sequence == 17
    assert pl.discontinuity_sequence == 3
    assert pl.endlist is True
    assert [s.msn for s in pl.segments] == [17, 18, 19, 20]
    assert [s.uri for s in pl.segments] == [
        "seg_000017.ts", "seg_000018.ts", "http://127.0.0.1:1/abs/seg_000019.ts", "seg_000020.ts"]
    assert [s.duration for s in pl.segments] == [2.0, 1.966667, 2.033333, 0.5]
    assert [s.discontinuity for s in pl.segments] == [False, False, True, False]
    assert [s.discontinuity_seq for s in pl.segments] == [3, 3, 4, 4]
    assert pl.segments[1].title == "live"
    assert pl.last_msn == 20


def test_parser_live_without_endlist_and_defaults() -> None:
    pl = parse_media_playlist("#EXTM3U\r\n#EXTINF:2,\r\na.ts\r\n")
    assert pl.endlist is False
    assert pl.media_sequence == 0 and pl.discontinuity_sequence == 0
    assert pl.target_duration is None
    assert [(s.msn, s.uri, s.duration) for s in pl.segments] == [(0, "a.ts", 2.0)]
    assert parse_media_playlist("#EXTM3U\n#EXT-X-TARGETDURATION:2\n").segments == ()


@pytest.mark.parametrize("text", [
    "",
    "#EXTINF:2,\na.ts\n",  # no header
    "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\nmedia.m3u8\n",  # master playlist
    "#EXTM3U\na.ts\n",  # URI without EXTINF
    "#EXTM3U\n#EXTINF:abc,\na.ts\n",
])
def test_parser_rejects_bad_playlists(text: str) -> None:
    with pytest.raises(PlaylistError):
        parse_media_playlist(text)


# ---------------------------------------------------------------------- static playlists


@pytest.fixture
def local_segments(synthetic_segments: List[Segment], tmp_path: Path) -> List[Segment]:
    return copy_segments(synthetic_segments[:4], tmp_path / "pub")


def test_static_playlist_with_discontinuity_tag_and_missing_file(local_segments: List[Segment],
                                                                  tmp_path: Path) -> None:
    pub = local_segments[0].path.parent
    names = [s.path.name for s in local_segments] + ["missing.ts"] + [local_segments[0].path.name]
    durs = [s.duration for s in local_segments] + [2.0, local_segments[0].duration]
    _write_playlist(pub / "p.m3u8", names, durs, media_sequence=100, endlist=True, disc_before=[2])
    src = HlsSource(pub / "p.m3u8", "ch", tmp_path / "work", poll_s=POLL, stall_timeout_s=2)
    assert isinstance(src, Source)
    segs = list(src)
    assert src.end_reason == "endlist"
    assert len(segs) == 5
    _assert_contiguous(segs)
    # Tag on the 3rd entry; the missing 5th entry flags the one after it.
    assert [s.discontinuity for s in segs] == [False, False, True, False, True]
    assert src.segments_missed == 1
    for s, orig in zip(segs, local_segments):
        assert s.path.parent == tmp_path / "work"
        assert s.path.read_bytes() == orig.path.read_bytes()
        assert s.channel == "ch"
    assert not list((tmp_path / "work").glob("*.part"))
    with pytest.raises(RuntimeError):
        iter(src)


def test_stalled_playlist_ends_after_timeout(local_segments: List[Segment], tmp_path: Path) -> None:
    pub = local_segments[0].path.parent
    _write_playlist(pub / "p.m3u8", [s.path.name for s in local_segments[:2]],
                    [s.duration for s in local_segments[:2]])
    src = HlsSource(pub / "p.m3u8", "ch", tmp_path / "work", poll_s=POLL, stall_timeout_s=0.5)
    t0 = time.monotonic()
    segs = list(src)
    elapsed = time.monotonic() - t0
    assert len(segs) == 2
    assert src.end_reason == "stall"
    assert 0.5 <= elapsed < 1.5


def test_missing_playlist_stalls_out(tmp_path: Path) -> None:
    src = HlsSource(tmp_path / "nope.m3u8", "ch", tmp_path / "work", poll_s=POLL, stall_timeout_s=0.3)
    assert list(src) == []
    assert src.end_reason == "stall"


def test_close_from_another_thread_unblocks(local_segments: List[Segment], tmp_path: Path) -> None:
    pub = local_segments[0].path.parent
    _write_playlist(pub / "p.m3u8", [local_segments[0].path.name], [local_segments[0].duration])
    src = HlsSource(pub / "p.m3u8", "ch", tmp_path / "work", poll_s=1.0, stall_timeout_s=60)
    it = iter(src)
    assert next(it).seq == 0
    threading.Timer(0.2, src.close).start()
    t0 = time.monotonic()
    assert list(it) == []
    assert time.monotonic() - t0 < 0.6
    assert src.end_reason == "closed"


def test_playlist_update_with_gap_flags_discontinuity(local_segments: List[Segment],
                                                      tmp_path: Path) -> None:
    """Media sequence jumps between reloads (segments scrolled out unseen)."""
    pub = local_segments[0].path.parent
    names = [s.path.name for s in local_segments]
    durs = [s.duration for s in local_segments]
    pl = pub / "p.m3u8"
    _write_playlist(pl, names[:2], durs[:2], media_sequence=0)
    src = HlsSource(pl, "ch", tmp_path / "work", poll_s=POLL, stall_timeout_s=2)
    it = iter(src)
    first = [next(it), next(it)]
    # msn 2..4 never listed; the playlist now holds msn 5 and 6.
    _write_playlist(pl, names[2:4], durs[2:4], media_sequence=5, endlist=True)
    rest = list(it)
    segs = first + rest
    _assert_contiguous(segs)
    assert [s.discontinuity for s in segs] == [False, False, True, False]
    assert src.segments_missed == 3


# ---------------------------------------------------------------------- against the simulator


def _media_sequence(path: Path) -> int:
    try:
        return parse_media_playlist(path.read_text(encoding="utf-8")).media_sequence
    except (OSError, PlaylistError):
        return -1


def test_simulator_10x_all_segments_once(synthetic_video: Path, tmp_path: Path) -> None:
    duration = probe(synthetic_video).duration
    with hls_sim.start(synthetic_video, tmp_path / "hls", speed=10, window=6) as sim:
        assert sim.wait_for_playlist(10)
        src = HlsSource(sim.playlist_path, "sim", tmp_path / "work", poll_s=POLL, stall_timeout_s=5)
        t0 = time.monotonic()
        segs = list(src)
        elapsed = time.monotonic() - t0
        assert sim.wait(10) == 0, sim.stderr_tail()
    assert src.end_reason == "endlist"
    _assert_contiguous(segs)
    assert not any(s.discontinuity for s in segs)
    assert src.segments_missed == 0
    assert len(segs) == 30
    assert abs(segs[-1].end - duration) <= 0.2
    assert len({s.path for s in segs}) == len(segs)
    assert all(s.path.is_file() and s.path.stat().st_size > 0 for s in segs)
    # Paced: roughly duration / 10, not instantaneous.
    assert 4.0 <= elapsed <= 9.0
    info = probe(segs[5].path)
    assert info.has_video and info.has_audio


def test_simulator_late_start_yields_rest(synthetic_video: Path, tmp_path: Path) -> None:
    duration = probe(synthetic_video).duration
    with hls_sim.start(synthetic_video, tmp_path / "hls", speed=20, window=4) as sim:
        deadline = time.monotonic() + 10
        while _media_sequence(sim.playlist_path) < 5:
            assert time.monotonic() < deadline, "simulator did not advance"
            time.sleep(0.02)
        src = HlsSource(sim.playlist_path, "late", tmp_path / "work", poll_s=POLL, stall_timeout_s=5)
        segs = list(src)
    assert src.end_reason == "endlist"
    _assert_contiguous(segs)
    assert not any(s.discontinuity for s in segs)
    assert 0 < len(segs) < 30
    assert segs[-1].end < duration


def test_paused_reader_resumes_with_discontinuity(synthetic_video: Path, tmp_path: Path) -> None:
    with hls_sim.start(synthetic_video, tmp_path / "hls", speed=20, window=4) as sim:
        assert sim.wait_for_playlist(10)
        src = HlsSource(sim.playlist_path, "pause", tmp_path / "work", poll_s=POLL, stall_timeout_s=5)
        segs: List[Segment] = []
        for seg in src:
            segs.append(seg)
            if len(segs) == 3:
                # The window is 4 x 2 s = 0.4 s of wall time at 20x.
                time.sleep(1.2)
    assert src.end_reason == "endlist"
    _assert_contiguous(segs)
    flagged = [i for i, s in enumerate(segs) if s.discontinuity]
    assert flagged == [3]
    assert src.segments_missed > 0
    assert len(segs) + src.segments_missed == 30


def test_simulator_over_http_loopback(synthetic_video: Path, tmp_path: Path) -> None:
    """The URL path, served from 127.0.0.1 (must bypass any HTTP(S)_PROXY)."""
    duration = probe(synthetic_video).duration
    out = tmp_path / "hls"
    out.mkdir()
    with hls_sim.serve_directory(out) as http, \
            hls_sim.start(synthetic_video, out, speed=20, window=6) as sim:
        assert http.host == "127.0.0.1" and http.port > 0
        assert sim.wait_for_playlist(10)
        src = HlsSource(http.url_for("playlist.m3u8"), "web", tmp_path / "work",
                        poll_s=POLL, stall_timeout_s=5)
        segs = list(src)
    assert src.end_reason == "endlist"
    _assert_contiguous(segs)
    assert not any(s.discontinuity for s in segs)
    assert len(segs) == 30
    assert abs(segs[-1].end - duration) <= 0.2


def test_simulator_killed_stream_stalls_out(synthetic_video: Path, tmp_path: Path) -> None:
    """A stream that dies (no end-of-list marker) ends the source via the stall timeout."""
    sim = hls_sim.start(synthetic_video, tmp_path / "hls", speed=1, window=3)
    try:
        assert sim.wait_for_playlist(10)
        assert sim.running
    finally:
        sim.stop(abrupt=True)
    sim.stop()  # idempotent
    assert not sim.running
    assert "#EXT-X-ENDLIST" not in sim.playlist_path.read_text(encoding="utf-8")
    src = HlsSource(sim.playlist_path, "dead", tmp_path / "work", poll_s=POLL, stall_timeout_s=0.3)
    segs = list(src)
    assert src.end_reason == "stall"
    assert 1 <= len(segs) <= 3
