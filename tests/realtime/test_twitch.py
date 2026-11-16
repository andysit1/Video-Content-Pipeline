"""Tests for TwitchSource, Twitch-flavoured HLS handling and the watch/vod commands.

Nothing here talks to Twitch: the resolver is replaced by ``hls_sim.TwitchLikeServer``,
which serves local segments the way Twitch does (stitched ads, prefetch entries,
expiring tokens, going offline, muted VOD segments).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import List, Sequence

import pytest
from click.testing import CliRunner

from VidFlow.realtime import cli_twitch
from VidFlow.realtime.cli import main
from VidFlow.realtime.config import RealtimeConfig
from VidFlow.realtime.media import probe
from VidFlow.realtime.pipeline import ChannelPipeline
from VidFlow.realtime.session import read_events
from VidFlow.realtime.sim import hls_sim
from VidFlow.realtime.sim.hls_sim import TwitchLikeServer, write_playlist
from VidFlow.realtime.sources import hls as hls_mod
from VidFlow.realtime.sources import twitch as twitch_mod
from VidFlow.realtime.sources.hls import (
    HlsSource,
    ResolveError,
    StreamOffline,
    UnsupportedPlaylistError,
    parse_media_playlist,
)
from VidFlow.realtime.sources.twitch import (
    StreamlinkResolver,
    TwitchSource,
    TwitchTarget,
    parse_target,
    parse_time,
)
from VidFlow.realtime.testing import SYNTHETIC_BURSTS, copy_segments, segments_from_file
from VidFlow.realtime.types import Segment, Source

POLL = 0.05


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hls_mod, "_FIRST_BACKOFF_S", 0.05)


def _assert_contiguous(segs: Sequence[Segment]) -> None:
    assert segs and segs[0].start == 0.0 and not segs[0].discontinuity
    assert [s.seq for s in segs] == list(range(len(segs)))
    for a, b in zip(segs, segs[1:]):
        assert b.start == a.end


@pytest.fixture
def pub(synthetic_segments: List[Segment], tmp_path: Path) -> List[Segment]:
    """The 30 fixture segments copied into a folder to publish."""
    return copy_segments(synthetic_segments, tmp_path / "pub")


def _publish(segs: Sequence[Segment], *, media_sequence: int = 0, endlist: bool = True, **kw) -> Path:
    return write_playlist(segs[0].path.parent / "playlist.m3u8", [s.path.name for s in segs],
                          [s.duration for s in segs], media_sequence=media_sequence, endlist=endlist, **kw)


# ---------------------------------------------------------------------- targets


@pytest.mark.parametrize("text, kind, ident, start", [
    ("SomeStreamer", "live", "somestreamer", 0.0),
    ("twitch.tv/some_streamer", "live", "some_streamer", 0.0),
    ("https://www.twitch.tv/Foo/", "live", "foo", 0.0),
    ("https://m.twitch.tv/foo?referrer=x", "live", "foo", 0.0),
    ("https://www.twitch.tv/videos/123456789", "vod", "123456789", 0.0),
    ("twitch.tv/videos/42?t=1h2m3s", "vod", "42", 3723.0),
    ("https://www.twitch.tv/foo/v/42", "vod", "42", 0.0),
    ("https://player.twitch.tv/?video=v42&parent=x", "vod", "42", 0.0),
    ("https://player.twitch.tv/?channel=foo", "live", "foo", 0.0),
])
def test_parse_target(text: str, kind: str, ident: str, start: float) -> None:
    t = parse_target(text)
    assert (t.kind, t.ident, t.start_s) == (kind, ident, start)
    assert t.default_channel == (ident if kind == "live" else f"vod-{ident}")
    assert t.url.startswith("https://www.twitch.tv/")


@pytest.mark.parametrize("text", ["", "https://youtube.com/foo", "twitch.tv/foo/clip/Abc",
                                  "https://clips.twitch.tv/Abc", "twitch.tv/directory", "bad name!"])
def test_parse_target_rejects(text: str) -> None:
    with pytest.raises(ValueError):
        parse_target(text)


def test_parse_time() -> None:
    assert parse_time("90") == 90.0
    assert parse_time("1:30") == 90.0
    assert parse_time("1:02:03") == 3723.0
    assert parse_time("1h2m3s") == 3723.0
    assert parse_time("10m") == 600.0
    assert parse_time("2.5") == 2.5
    for bad in ("", "x", "1:2:3:4", "-5"):
        with pytest.raises(ValueError):
            parse_time(bad)


# ---------------------------------------------------------------------- playlist parsing

TWITCH_LIVE = """\
#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:6
#EXT-X-MEDIA-SEQUENCE:100
#EXT-X-TWITCH-ELAPSED-SECS:200.000
#EXT-X-DATERANGE:ID="playlist-creation-1",CLASS="timestamp",START-DATE="2024-01-01T00:00:00.000Z",END-ON-NEXT=YES
#EXT-X-DATERANGE:ID="stitched-ad-1-4",CLASS="twitch-stitched-ad",START-DATE="2024-01-01T00:00:04.000Z",DURATION=3.999,X-TV-TWITCH-AD-URL="https://x/y,z"
#EXT-X-PROGRAM-DATE-TIME:2024-01-01T00:00:00.000Z
#EXTINF:2.000,live
https://edge.example/a.ts
#EXT-X-PROGRAM-DATE-TIME:2024-01-01T00:00:02.000Z
#EXTINF:2.000,live
https://edge.example/b.ts
#EXT-X-DISCONTINUITY
#EXT-X-PROGRAM-DATE-TIME:2024-01-01T00:00:04.000Z
#EXTINF:2.000,live
https://edge.example/ad1.ts
#EXTINF:2.000,live
https://edge.example/ad2.ts
#EXT-X-DISCONTINUITY
#EXT-X-PROGRAM-DATE-TIME:2024-01-01T00:00:08.000Z
#EXTINF:2.000,Amazon|123
https://edge.example/ad3.ts
#EXT-X-PROGRAM-DATE-TIME:2024-01-01T00:00:10.12345Z
#EXTINF:2.000,live
https://edge.example/c.ts
#EXT-X-TWITCH-PREFETCH:https://edge.example/d.ts
#EXT-X-TWITCH-PREFETCH:https://edge.example/e.ts
"""


def test_parser_marks_twitch_ads_and_ignores_extras() -> None:
    pl = parse_media_playlist(TWITCH_LIVE)
    assert [s.uri.rsplit("/", 1)[1] for s in pl.segments] == ["a.ts", "b.ts", "ad1.ts", "ad2.ts", "ad3.ts", "c.ts"]
    # ad1/ad2 by the date range (dates carried forward to ad2); ad3 by its title.
    assert [s.ad for s in pl.segments] == [False, False, True, True, True, False]
    assert [s.msn for s in pl.segments] == list(range(100, 106))
    assert pl.segments[3].program_date_time is not None
    assert pl.segments[3].program_date_time.second == 6
    assert pl.segments[5].program_date_time.microsecond == 123450
    assert pl.endlist is False and pl.target_duration == 6.0


@pytest.mark.parametrize("extra", ['#EXT-X-MAP:URI="init.mp4"', "#EXT-X-KEY:METHOD=AES-128,URI=\"k\"",
                                   "#EXT-X-BYTERANGE:100@0"])
def test_parser_refuses_unsupported(extra: str) -> None:
    with pytest.raises(UnsupportedPlaylistError):
        parse_media_playlist(f"#EXTM3U\n{extra}\n#EXTINF:2,\na.ts\n")


def test_parser_accepts_key_none_and_bad_dates() -> None:
    pl = parse_media_playlist("#EXTM3U\n#EXT-X-KEY:METHOD=NONE\n#EXT-X-PROGRAM-DATE-TIME:garbage\n"
                              '#EXT-X-DATERANGE:ID="stitched-ad-x",START-DATE="nope"\n#EXTINF:2,\na.ts\n')
    assert [(s.uri, s.ad, s.program_date_time) for s in pl.segments] == [("a.ts", False, None)]


# ---------------------------------------------------------------------- HlsSource against the Twitch-like server


@pytest.mark.parametrize("by_title, by_range", [(True, True), (False, True), (True, False)])
def test_ads_are_skipped_and_flag_a_discontinuity(pub: List[Segment], tmp_path: Path,
                                                  by_title: bool, by_range: bool) -> None:
    _publish(pub[:10])
    with TwitchLikeServer(pub[0].path.parent, ad_msns={4, 5, 6}, ad_title=by_title, ad_daterange=by_range) as srv:
        src = HlsSource(None, "ch", tmp_path / "work", poll_s=POLL, stall_timeout_s=3, refresh=srv.resolve)
        segs = list(src)
        assert not srv.requested("ad_")
        assert not srv.requested("prefetch_")
    assert src.end_reason == "endlist"
    _assert_contiguous(segs)
    assert len(segs) == 7 and src.segments_ad == 3 and src.segments_missed == 0
    assert [i for i, s in enumerate(segs) if s.discontinuity] == [4]
    for s, orig in zip(segs[4:], pub[7:10]):
        assert s.path.read_bytes() == orig.path.read_bytes()


def test_ads_kept_when_not_skipping(pub: List[Segment], tmp_path: Path) -> None:
    _publish(pub[:4])
    with TwitchLikeServer(pub[0].path.parent, ad_msns={1}, ad_title=True) as srv:
        src = HlsSource(srv.playlist_url, "ch", tmp_path / "work", poll_s=POLL, stall_timeout_s=1, skip_ads=False)
        segs = list(src)
    # The ad's URI is not fetchable on the simulator, so it is "missed" rather than skipped.
    assert src.segments_ad == 0 and src.segments_missed == 1 and len(segs) == 3


def test_expired_token_is_re_resolved_without_a_discontinuity(pub: List[Segment], tmp_path: Path) -> None:
    pl = _publish(pub[:3], endlist=False)
    with TwitchLikeServer(pub[0].path.parent) as srv:
        src = HlsSource(None, "ch", tmp_path / "work", poll_s=POLL, stall_timeout_s=5, refresh=srv.resolve)
        it = iter(src)
        first = [next(it) for _ in range(3)]
        srv.expire_token()
        write_playlist(pl, [s.path.name for s in pub[:6]], [s.duration for s in pub[:6]], endlist=True)
        rest = list(it)
        assert srv.playlist_codes.get(403, 0) >= 1
    segs = first + rest
    _assert_contiguous(segs)
    assert len(segs) == 6 and not any(s.discontinuity for s in segs)
    assert src.reconnects >= 1 and src.end_reason == "endlist"


def test_stream_going_offline_ends_cleanly(pub: List[Segment], tmp_path: Path) -> None:
    _publish(pub[:3], endlist=False)
    with TwitchLikeServer(pub[0].path.parent) as srv:
        src = HlsSource(None, "ch", tmp_path / "work", poll_s=POLL, stall_timeout_s=30, refresh=srv.resolve)
        it = iter(src)
        assert len([next(it) for _ in range(3)]) == 3
        srv.set_offline()
        t0 = time.monotonic()
        assert list(it) == []
        assert time.monotonic() - t0 < 2.0
    assert src.end_reason == "offline"


def test_offline_at_start_yields_nothing(tmp_path: Path) -> None:
    with TwitchLikeServer(tmp_path) as srv:
        srv.set_offline()
        src = TwitchSource("somechannel", tmp_path / "work", resolver=srv)
        assert isinstance(src, Source) and src.channel == "somechannel"
        assert list(src) == []
    assert src.end_reason == "offline"


def test_stall_re_resolves_then_ends(pub: List[Segment], tmp_path: Path) -> None:
    _publish(pub[:2], endlist=False)
    with TwitchLikeServer(pub[0].path.parent) as srv:
        src = HlsSource(None, "ch", tmp_path / "work", poll_s=POLL, stall_timeout_s=0.3, refresh=srv.resolve)
        segs = list(src)
        calls = srv.resolve_calls
    assert len(segs) == 2 and src.end_reason == "stall"
    assert calls == 1 + hls_mod._MAX_STALL_REFRESHES


def test_transient_resolve_failures_are_retried(pub: List[Segment], tmp_path: Path) -> None:
    _publish(pub[:3])
    with TwitchLikeServer(pub[0].path.parent, fail_resolves=3) as srv:
        src = HlsSource(None, "ch", tmp_path / "work", poll_s=POLL, refresh=srv.resolve, reconnect_timeout_s=10)
        assert len(list(src)) == 3
        assert srv.resolve_calls == 4


def test_resolve_failures_give_up_after_timeout(tmp_path: Path) -> None:
    calls = []

    def failing() -> str:
        calls.append(1)
        raise ConnectionError("no route")

    src = HlsSource(None, "ch", tmp_path / "work", refresh=failing, reconnect_timeout_s=0.3)
    with pytest.raises(ResolveError, match="could not resolve"):
        list(src)
    assert len(calls) >= 2


def test_fatal_resolve_error_propagates(tmp_path: Path) -> None:
    def fatal() -> str:
        raise ResolveError("quality 9000p not offered", transient=False)

    src = HlsSource(None, "ch", tmp_path / "work", refresh=fatal)
    with pytest.raises(ResolveError, match="9000p"):
        list(src)
    assert src.end_reason == "error"


def test_new_broadcast_restarts_the_sequence(pub: List[Segment], tmp_path: Path) -> None:
    pl = _publish(pub[:3], media_sequence=500, endlist=False)
    with TwitchLikeServer(pub[0].path.parent) as srv:
        src = HlsSource(None, "ch", tmp_path / "work", poll_s=POLL, stall_timeout_s=5, refresh=srv.resolve)
        it = iter(src)
        first = [next(it) for _ in range(3)]
        write_playlist(pl, [s.path.name for s in pub[3:6]], [s.duration for s in pub[3:6]],
                       media_sequence=0, endlist=True)
        rest = list(it)
    segs = first + rest
    _assert_contiguous(segs)
    assert len(segs) == 6 and [i for i, s in enumerate(segs) if s.discontinuity] == [3]


def test_fmp4_playlist_fails_clearly(pub: List[Segment], tmp_path: Path) -> None:
    _publish(pub[:3])
    with TwitchLikeServer(pub[0].path.parent, fmp4=True) as srv:
        with pytest.raises(UnsupportedPlaylistError, match="MPEG-TS"):
            list(TwitchSource("chan", tmp_path / "work", resolver=srv))
        assert not srv.requested("seg") and not srv.requested("init")


# ---------------------------------------------------------------------- VODs


@pytest.fixture(scope="module")
def vod_segments(synthetic_video: Path, tmp_path_factory: pytest.TempPathFactory) -> List[Segment]:
    """The fixture as ~10 s segments, like a Twitch VOD."""
    return segments_from_file(synthetic_video, tmp_path_factory.mktemp("vod"), segment_time=10.0)


@pytest.fixture
def vod_dir(vod_segments: List[Segment], tmp_path: Path) -> Path:
    segs = copy_segments(vod_segments, tmp_path / "vod")
    _publish(segs, playlist_type="VOD")
    return segs[0].path.parent


def test_vod_whole_with_muted_segment(vod_dir: Path, vod_segments: List[Segment], tmp_path: Path) -> None:
    assert len(vod_segments) == 6 and all(9.0 < s.duration < 11.0 for s in vod_segments)
    with TwitchLikeServer(vod_dir, muted_msns={2}, prefetch=True) as srv:
        src = TwitchSource("twitch.tv/videos/77", tmp_path / "work", resolver=srv)
        assert src.channel == "vod-77"
        t0 = time.monotonic()
        segs = list(src)
        elapsed = time.monotonic() - t0
        assert srv.requested("seg_000002-unmuted") and srv.requested("seg_000002-muted")
    assert src.end_reason == "endlist" and elapsed < 3.0  # unpaced: as fast as it downloads
    _assert_contiguous(segs)
    assert len(segs) == 6 and not any(s.discontinuity for s in segs)
    assert segs[2].path.read_bytes() == vod_segments[2].path.read_bytes()


@pytest.mark.parametrize("start, end, expect", [
    (12.0, 35.0, [1, 2, 3]),  # the segments covering both offsets are included
    (0.0, 10.5, [0, 1]),
    (51.0, None, [5]),
])
def test_vod_slice(vod_dir: Path, vod_segments: List[Segment], tmp_path: Path,
                   start: float, end, expect: List[int]) -> None:
    with TwitchLikeServer(vod_dir) as srv:
        src = TwitchSource(TwitchTarget("vod", "77"), tmp_path / "work", resolver=srv, start_s=start, end_s=end)
        segs = list(src)
        fetched = [r for r in srv.requested("seg_")]
    assert [s.path.read_bytes() for s in segs] == [vod_segments[i].path.read_bytes() for i in expect]
    assert len(fetched) == len(expect)
    _assert_contiguous(segs)
    assert src.end_reason == ("end_offset" if end is not None and expect[-1] < 5 else "endlist")


def test_vod_start_from_url_t(vod_dir: Path, vod_segments: List[Segment], tmp_path: Path) -> None:
    with TwitchLikeServer(vod_dir) as srv:
        segs = list(TwitchSource("twitch.tv/videos/77?t=0h0m41s", tmp_path / "work", resolver=srv))
    assert len(segs) == 2


def test_vod_through_the_pipeline(vod_dir: Path, tmp_path: Path) -> None:
    """10 s segments: detection still finds the two bursts; clips are padded to segment edges."""
    cfg = RealtimeConfig(warmup_s=5.0).validate(segment_duration=10.0)
    with TwitchLikeServer(vod_dir) as srv:
        pipe = ChannelPipeline(lambda d: TwitchSource("twitch.tv/videos/77", d, resolver=srv),
                               cfg, tmp_path / "out", channel="vod-77", session="s")
        res = pipe.run()
    assert res.ok and res.segments == 6 and res.clips == 2, res
    for clip, (b0, b1) in zip(pipe.sorted_clips, SYNTHETIC_BURSTS):
        w = clip.window
        assert abs(w.start - b0) <= 0.5 and abs(w.end - b1) <= 0.5
        info = probe(clip.path)
        assert info.has_audio and info.has_video
        assert clip.file_start <= w.start - cfg.pre_pad_s + 1e-6


# ---------------------------------------------------------------------- live through the pipeline


def test_live_sim_with_ads_and_token_expiry_through_the_pipeline(synthetic_video: Path, tmp_path: Path) -> None:
    """hls_sim live at 10x behind the Twitch-like server: ads replace 50-56 s, the token expires mid-stream."""
    out = tmp_path / "hls"
    out.mkdir()
    with TwitchLikeServer(out, ad_msns={25, 26, 27}) as srv, \
            hls_sim.start(synthetic_video, out, speed=10, window=6) as sim:
        assert sim.wait_for_playlist(10)
        timer = threading.Timer(2.0, srv.expire_token)
        timer.start()
        holder = {}

        def make(d: Path) -> TwitchSource:
            holder["src"] = TwitchSource("simchan", d, resolver=srv, poll_s=POLL, stall_timeout_s=5)
            return holder["src"]

        pipe = ChannelPipeline(make, RealtimeConfig(warmup_s=5.0), tmp_path / "out", channel="simchan", session="s")
        res = pipe.run()
        timer.cancel()
        assert not srv.requested("ad_")
    src = holder["src"]
    assert res.ok and src.end_reason == "endlist", (res, src.end_reason)
    assert src.segments_ad == 3 and src.reconnects >= 1 and src.hls.segments_missed == 0
    assert res.segments == 27 and res.clips == 2
    events = list(read_events(pipe.session_dir, "segment"))
    assert [e["seq"] for e in events if e["discontinuity"]] == [25]


# ---------------------------------------------------------------------- StreamlinkResolver (fake streamlink)


class _FakeStream:
    def __init__(self, url: str) -> None:
        self.url = url


class _FakePlugin:
    streams_result: dict = {}
    broadcast_id = None
    raise_on_streams: Exception = None  # type: ignore[assignment]

    def __init__(self, session, url, options) -> None:
        self.url, self.options = url, options

    def streams(self):
        if self.raise_on_streams is not None:
            raise self.raise_on_streams
        return dict(self.streams_result)

    def get_id(self):
        return self.broadcast_id


class _FakeSession:
    class http:
        headers = {"User-Agent": "UA-test"}
        verify = True

    def resolve_url(self, url):
        return "twitch", _FakePlugin, url


@pytest.fixture
def fake_resolver(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("streamlink")
    monkeypatch.setattr(_FakePlugin, "streams_result", {})
    monkeypatch.setattr(_FakePlugin, "broadcast_id", None)
    monkeypatch.setattr(_FakePlugin, "raise_on_streams", None)

    def make(quality: str = "best", target: str = "somechan") -> StreamlinkResolver:
        r = StreamlinkResolver(parse_target(target), quality)
        monkeypatch.setattr(r, "_get_session", lambda: _FakeSession())
        return r

    return make


def test_resolver_picks_first_available_quality(fake_resolver) -> None:
    _FakePlugin.streams_result = {"best": _FakeStream("u-best"), "720p60": _FakeStream("u-720"),
                                  "480p": _FakeStream("u-480"), "audio_only": _FakeStream("u-a")}
    assert fake_resolver("1080p60, 720p60,best").resolve() == "u-720"
    r = fake_resolver()
    assert r.resolve() == "u-best" and r.chosen_quality == "best"
    assert r.user_agent == "UA-test"
    with pytest.raises(ResolveError) as ei:
        fake_resolver("4k").resolve()
    assert not ei.value.transient and "available: 480p, 720p60, audio_only, best" in str(ei.value)


def test_resolver_offline_vs_unreachable(fake_resolver) -> None:
    with pytest.raises(StreamOffline):
        fake_resolver().resolve()
    _FakePlugin.broadcast_id = "123"
    with pytest.raises(ResolveError) as ei:
        fake_resolver().resolve()
    assert ei.value.transient and "is live" in str(ei.value)


def test_resolver_plugin_error_is_transient(fake_resolver) -> None:
    from streamlink.exceptions import PluginError

    _FakePlugin.raise_on_streams = PluginError("gql said no")
    with pytest.raises(ResolveError) as ei:
        fake_resolver().resolve()
    assert ei.value.transient


def test_missing_streamlink_names_the_extra(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "streamlink", None)
    with pytest.raises(ResolveError) as ei:
        twitch_mod.require_streamlink()
    assert not ei.value.transient and "VidFlow[twitch]" in str(ei.value)
    src = TwitchSource("somechan", tmp_path / "w")  # constructing works; iterating fails clearly
    with pytest.raises(ResolveError, match="streamlink"):
        list(src)
    monkeypatch.setattr(cli_twitch, "RESOLVER_FACTORY", None)
    r = CliRunner().invoke(main, ["watch", "somechan", "--out", str(tmp_path / "o")])
    assert r.exit_code == 1 and "VidFlow[twitch]" in r.output


# ---------------------------------------------------------------------- CLI


def test_cli_vod_slice_and_watch(vod_dir: Path, pub: List[Segment], tmp_path: Path,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    with TwitchLikeServer(vod_dir) as srv:
        monkeypatch.setattr(cli_twitch, "RESOLVER_FACTORY", lambda target, q: srv)
        r = CliRunner().invoke(main, ["vod", "https://www.twitch.tv/videos/77", "--start", "0:05",
                                      "--end", "45", "--out", str(out), "--session", "v", "--set", "warmup_s=5"])
    assert r.exit_code == 0, r.output
    assert r.output.count(" clip ") == 2 and "source ended: end_offset" in r.output
    assert len(list((out / "vod-77" / "v" / "clips").glob("*.mp4"))) == 2
    assert (out / "vod-77" / "v" / "events.jsonl").is_file()

    # watch: two channels at once; one is offline.
    _publish(pub)
    with TwitchLikeServer(pub[0].path.parent, ad_msns={27}) as live, TwitchLikeServer(tmp_path) as dead:
        dead.set_offline()
        monkeypatch.setattr(cli_twitch, "RESOLVER_FACTORY",
                            lambda target, q: live if target.ident == "alive" else dead)
        r = CliRunner().invoke(main, ["watch", "alive", "twitch.tv/gone", "--out", str(out), "--session", "w",
                                      "--set", "warmup_s=5"])
    assert r.exit_code == 0, r.output
    assert "[alive] source ended: endlist" in r.output and "1 ad segments skipped" in r.output
    assert "[gone] source ended: offline" in r.output
    assert len(list((out / "alive" / "w" / "clips").glob("*.mp4"))) == 2
    assert (out / "gone" / "w" / "events.jsonl").is_file()


def test_cli_rejects_bad_input(tmp_path: Path) -> None:
    runner = CliRunner()
    assert runner.invoke(main, ["watch", "https://youtube.com/x"]).exit_code == 2
    assert runner.invoke(main, ["vod", "somechannel"]).exit_code == 2
    assert runner.invoke(main, ["vod", "twitch.tv/videos/1", "--start", "10", "--end", "5"]).exit_code == 2
    assert runner.invoke(main, ["watch", "x", "--duration", "soon"]).exit_code == 2
