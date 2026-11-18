"""Live-following gaps: failed downloads reported as failures, --wait, and sharded watch.

Nothing here talks to Twitch: ``hls_sim.TwitchLikeServer`` stands in for it.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Sequence

import pytest
from click.testing import CliRunner

from VidFlow.realtime import cli_twitch
from VidFlow.realtime.cli import main
from VidFlow.realtime.config import RealtimeConfig
from VidFlow.realtime.pipeline import ChannelPipeline
from VidFlow.realtime.session import read_events
from VidFlow.realtime.sim.hls_sim import FixedUrlFactory, TwitchLikeServer, write_playlist
from VidFlow.realtime.sources import hls as hls_mod
from VidFlow.realtime.sources.hls import HlsSource, SegmentFetchError
from VidFlow.realtime.sources.twitch import TwitchFactory, TwitchSource, parse_target
from VidFlow.realtime.testing import copy_segments
from VidFlow.realtime.types import Segment

POLL = 0.05


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hls_mod, "_FIRST_BACKOFF_S", 0.05)


@pytest.fixture
def pub(synthetic_segments: List[Segment], tmp_path: Path) -> List[Segment]:
    return copy_segments(synthetic_segments, tmp_path / "pub")


def _publish(segs: Sequence[Segment], *, media_sequence: int = 0, endlist: bool = True) -> Path:
    return write_playlist(segs[0].path.parent / "playlist.m3u8", [s.path.name for s in segs],
                          [s.duration for s in segs], media_sequence=media_sequence, endlist=endlist)


# ---------------------------------------------------------------------- failed downloads


def test_blocked_segments_raise_at_the_end(pub: List[Segment], tmp_path: Path) -> None:
    _publish(pub[:6], endlist=False)
    with TwitchLikeServer(pub[0].path.parent) as srv:
        srv.segment_status = 403
        src = HlsSource(None, "ch", tmp_path / "work", poll_s=POLL, stall_timeout_s=0.3, refresh=srv.resolve)
        with pytest.raises(SegmentFetchError, match="no segment was ever downloaded.*HTTP 403"):
            list(src)
    assert src.end_reason == "error" and src.segments_yielded == 0
    assert src.fetch_failures == 6 and src.segments_missed == 6


def test_downloads_failing_after_a_good_start_raise(pub: List[Segment], tmp_path: Path) -> None:
    pl = _publish(pub[:3], endlist=False)
    with TwitchLikeServer(pub[0].path.parent) as srv:
        src = HlsSource(None, "ch", tmp_path / "work", poll_s=POLL, stall_timeout_s=0.3, refresh=srv.resolve)
        it = iter(src)
        assert len([next(it) for _ in range(3)]) == 3
        srv.segment_status = 403
        write_playlist(pl, [s.path.name for s in pub[:8]], [s.duration for s in pub[:8]], endlist=True)
        with pytest.raises(SegmentFetchError, match="the last 5 downloads failed"):
            list(it)


def test_one_lost_segment_is_not_a_failure(pub: List[Segment], tmp_path: Path) -> None:
    _publish(pub[:4])
    (pub[0].path.parent / pub[1].path.name).unlink()
    with TwitchLikeServer(pub[0].path.parent) as srv:
        src = HlsSource(None, "ch", tmp_path / "work", poll_s=POLL, stall_timeout_s=1, refresh=srv.resolve)
        assert len(list(src)) == 3
    assert src.end_reason == "endlist" and src.fetch_failures == 1


def test_pipeline_reports_a_channel_with_no_video_as_failed(pub: List[Segment], tmp_path: Path) -> None:
    _publish(pub[:6], endlist=False)
    with TwitchLikeServer(pub[0].path.parent) as srv:
        srv.segment_status = 403
        pipe = ChannelPipeline(
            lambda d: TwitchSource("chan", d, resolver=srv, stall_timeout_s=0.3, poll_s=POLL),
            RealtimeConfig(warmup_s=5), tmp_path / "out", channel="chan", session="s",
        )
        res = pipe.run()
    assert not res.ok and "SegmentFetchError" in (res.error or "")
    assert res.source["fetch_failures"] == 6 and res.source["end_reason"] == "error"
    end = [e for e in read_events(res.session_dir) if e["type"] == "session_end"][-1]
    assert end["error"] and end["source"]["segments"] == 0


def test_cli_watch_exits_1_when_video_never_arrives(pub: List[Segment], tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    _publish(pub[:6], endlist=False)
    monkeypatch.setattr(hls_mod, "_MAX_STALL_REFRESHES", 0)
    with TwitchLikeServer(pub[0].path.parent) as srv:
        srv.segment_status = 403
        monkeypatch.setattr(cli_twitch, "RESOLVER_FACTORY", lambda target, q: srv)
        r = CliRunner().invoke(main, ["watch", "chan", "--out", str(tmp_path / "out"), "--session", "s",
                                      "--set", "warmup_s=5"])
    assert r.exit_code == 1, r.output
    assert "[chan] FAILED: SegmentFetchError" in r.output
    assert "source ended: error" in r.output and "6 failed downloads" in r.output


# ---------------------------------------------------------------------- --wait


def _wait_for(cond, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def test_wait_follows_each_broadcast_and_chains_them(pub: List[Segment], tmp_path: Path) -> None:
    pl = _publish(pub[:3], endlist=False)
    with TwitchLikeServer(pub[0].path.parent) as srv:
        srv.set_offline()
        src = TwitchSource("chan", tmp_path / "work", resolver=srv, wait=True, offline_poll_s=0.05,
                           poll_s=POLL, stall_timeout_s=30)
        got: List[Segment] = []
        reader = threading.Thread(target=lambda: got.extend(src), daemon=True)
        reader.start()
        time.sleep(0.3)
        assert not got and srv.resolve_calls >= 3  # it kept checking while offline
        srv.set_offline(False)  # first broadcast
        _wait_for(lambda: len(got) == 3)
        srv.set_offline()  # first broadcast ends
        _wait_for(lambda: src.hls.end_reason == "offline")
        write_playlist(pl, [s.path.name for s in pub[3:6]], [s.duration for s in pub[3:6]],
                       media_sequence=0, endlist=True)
        srv.set_offline(False)  # second broadcast
        reader.join(10)
        assert not reader.is_alive()
    assert [s.seq for s in got] == list(range(6))
    assert [i for i, s in enumerate(got) if s.discontinuity] == [3]
    for a, b in zip(got, got[1:]):
        assert b.start == pytest.approx(a.end)
    assert len({s.path for s in got}) == 6
    st = src.stats()
    assert st["broadcasts"] == 2 and st["segments"] == 6 and st["end_reason"] == "endlist"


def test_wait_ends_when_closed_while_offline(tmp_path: Path) -> None:
    with TwitchLikeServer(tmp_path) as srv:
        srv.set_offline()
        src = TwitchSource("chan", tmp_path / "work", resolver=srv, wait=True, offline_poll_s=10)
        threading.Timer(0.2, src.close).start()
        t0 = time.monotonic()
        assert list(src) == []
        assert time.monotonic() - t0 < 2.0
    assert src.end_reason == "closed" and src.stats()["broadcasts"] == 0


def test_wait_is_for_live_channels_only(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="live"):
        TwitchSource("twitch.tv/videos/1", tmp_path, resolver=lambda: "x", wait=True)


def test_cli_watch_wait_until_duration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with TwitchLikeServer(tmp_path) as srv:
        srv.set_offline()
        monkeypatch.setattr(cli_twitch, "RESOLVER_FACTORY", lambda target, q: srv)
        r = CliRunner().invoke(main, ["watch", "chan", "--wait", "--offline-poll", "0.1", "--duration", "1",
                                      "--out", str(tmp_path / "out"), "--session", "s"])
        assert srv.resolve_calls >= 3
    assert r.exit_code == 0, r.output
    assert "waiting for offline channels" in r.output
    assert "[chan] source ended: closed; 0 broadcast(s), 0 segments" in r.output


def test_cli_offline_without_wait_suggests_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with TwitchLikeServer(tmp_path) as srv:
        srv.set_offline()
        monkeypatch.setattr(cli_twitch, "RESOLVER_FACTORY", lambda target, q: srv)
        r = CliRunner().invoke(main, ["watch", "chan", "--out", str(tmp_path / "out"), "--session", "s"])
    assert r.exit_code == 0, r.output
    assert "source ended: offline (not live; pass --wait to wait for it)" in r.output


# ---------------------------------------------------------------------- sharded watch


def test_twitch_factory_is_picklable() -> None:
    import pickle

    f = TwitchFactory(parse_target("chan"), "chan", wait=True, resolver_factory=FixedUrlFactory((("chan", "u"),)))
    assert pickle.loads(pickle.dumps(f)) == f


def test_cli_watch_in_worker_processes(pub: List[Segment], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _publish(pub)
    out = tmp_path / "out"
    with TwitchLikeServer(pub[0].path.parent) as srv:
        url = srv.playlist_url
        factory = FixedUrlFactory((("alpha", url), ("beta", url)))
        monkeypatch.setattr(cli_twitch, "RESOLVER_FACTORY", factory)
        r = CliRunner().invoke(main, ["watch", "alpha", "beta", "gone", "--procs", "2", "--out", str(out),
                                      "--session", "w", "--set", "warmup_s=5"])
    assert r.exit_code == 0, r.output
    assert "in 2 processes" in r.output
    for name in ("alpha", "beta"):
        assert f"[{name}] source ended: endlist" in r.output
        assert len(list((out / name / "w" / "clips").glob("*.mp4"))) == 2
    assert r.output.count(" clip ") == 4  # clips forwarded from the workers
    assert "[gone] source ended: offline" in r.output
