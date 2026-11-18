"""Local live-HLS simulator: a video file played out as a sliding-window playlist.

:func:`start` runs FFmpeg's HLS muxer as a subprocess, reading the input at
``speed`` times real time, so the playlist on disk behaves like a live
stream's: new segments appear as they would be broadcast, old ones scroll out
of the playlist (and are deleted), and ``#EXT-X-ENDLIST`` is written when the
file is finished. :func:`serve_directory` optionally serves the output folder
over HTTP on 127.0.0.1 so the URL code path of ``HlsSource`` can be exercised.
:class:`TwitchLikeServer` serves it the way Twitch does (stitched ads,
low-latency prefetch entries, program date-times, expiring playlist tokens,
going offline, muted VOD segments), for offline tests of ``TwitchSource``.

Nothing here talks to any outside host.
"""

from __future__ import annotations

import dataclasses
import functools
import http.server
import logging
import subprocess
import tempfile
import threading
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ..media import MediaError, ffmpeg_exe

__all__ = ["HlsSim", "start", "HttpServerHandle", "serve_directory", "TwitchLikeServer", "write_playlist"]

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

_STDERR_TAIL_CHARS = 2000


class HlsSim:
    """Handle on a running simulator (returned by :func:`start`).

    Use as a context manager, or call :meth:`stop`.
    """

    def __init__(self, process: subprocess.Popen, playlist_path: Path, stderr_file) -> None:
        self.process = process
        self.playlist_path = playlist_path
        self.out_dir = playlist_path.parent
        self._stderr = stderr_file
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.process.poll() is None

    @property
    def returncode(self) -> Optional[int]:
        return self.process.poll()

    def wait(self, timeout: Optional[float] = None) -> int:
        """Wait for FFmpeg to finish the file; return its exit code.

        Raises ``subprocess.TimeoutExpired`` on timeout.
        """
        return self.process.wait(timeout)

    def wait_for_playlist(self, timeout: float = 10.0) -> bool:
        """Wait until the playlist file exists; ``False`` on timeout or early exit."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.playlist_path.is_file():
                return True
            if not self.running:
                return self.playlist_path.is_file()
            time.sleep(0.02)
        return False

    def stderr_tail(self) -> str:
        with self._lock:
            if self._stderr is None or self._stderr.closed:
                return ""
            self._stderr.seek(0)
            return self._stderr.read().decode("utf-8", "replace")[-_STDERR_TAIL_CHARS:]

    def stop(self, timeout: float = 5.0, *, abrupt: bool = False) -> None:
        """Stop FFmpeg if still running. Idempotent.

        By default FFmpeg is asked to terminate; on POSIX it then finishes the
        playlist cleanly with ``#EXT-X-ENDLIST`` (on Windows terminating is
        always abrupt). ``abrupt=True`` kills it, leaving a playlist without
        ``#EXT-X-ENDLIST`` that never updates again: a stream that died.
        """
        if self.process.poll() is None:
            if abrupt:
                self.process.kill()
            else:
                self.process.terminate()
            try:
                self.process.wait(timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout)
        with self._lock:
            if self._stderr is not None and not self._stderr.closed:
                self._stderr.close()

    def __enter__(self) -> "HlsSim":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def start(
    video: PathLike,
    out_dir: PathLike,
    speed: Optional[float] = 1.0,
    window: int = 6,
    *,
    segment_time: float = 2.0,
    playlist_name: str = "playlist.m3u8",
    transcode: bool = False,
    delete_segments: bool = True,
) -> HlsSim:
    """Start publishing ``video`` as a live HLS playlist in ``out_dir``.

    - ``speed``: input read rate relative to real time (FFmpeg ``-readrate``);
      ``None`` publishes as fast as possible.
    - ``window``: number of segments kept in the playlist (``-hls_list_size``).
    - ``segment_time``: target segment length. With ``transcode=False`` the
      streams are copied, so segments can only start on the input's keyframes;
      ``transcode=True`` re-encodes with a keyframe every ``segment_time``.
    - ``delete_segments``: delete segment files that left the playlist, as a
      live server does.

    Segments and the playlist are written to temporary names and renamed, so a
    reader never sees a half-written file. Returns at once; the playlist file
    appears shortly after (see :meth:`HlsSim.wait_for_playlist`).
    """
    video = Path(video)
    out_dir = Path(out_dir)
    if not video.is_file():
        raise MediaError(f"input file not found: {video}")
    if speed is not None and not speed > 0:
        raise ValueError("speed must be > 0 or None")
    if int(window) != window or window < 1:
        raise ValueError("window must be an integer >= 1")
    if not segment_time > 0:
        raise ValueError("segment_time must be > 0")
    out_dir.mkdir(parents=True, exist_ok=True)
    playlist_path = out_dir / playlist_name
    # A stale playlist from an earlier run must not be mistaken for this one.
    for stale in (playlist_path, playlist_path.with_name(playlist_path.name + ".tmp")):
        if stale.exists():
            stale.unlink()

    cmd: List[str] = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
    if speed is not None:
        cmd += ["-readrate", f"{float(speed)}"]
    cmd += ["-i", str(video), "-map", "0:v:0?", "-map", "0:a:0?"]
    if transcode:
        cmd += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-bf", "0",
            "-force_key_frames", f"expr:gte(t,n_forced*{segment_time})",
            "-sc_threshold", "0",
            "-c:a", "aac",
            "-b:a", "128k",
        ]
    else:
        cmd += ["-c", "copy"]
    flags = "temp_file" + ("+delete_segments" if delete_segments else "")
    cmd += [
        "-f", "hls",
        "-hls_time", f"{segment_time}",
        "-hls_list_size", str(int(window)),
        "-hls_flags", flags,
        "-hls_segment_type", "mpegts",
        "-hls_segment_filename", str(out_dir / "seg_%06d.ts"),
        str(playlist_path),
    ]
    log.debug("starting HLS simulator: %s", cmd)
    stderr_file = tempfile.TemporaryFile()
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
        )
    except BaseException:
        stderr_file.close()
        raise
    return HlsSim(proc, playlist_path, stderr_file)


# ---------------------------------------------------------------------- HTTP


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        log.debug("hls_sim http: " + format, *args)

    def end_headers(self) -> None:
        # Live playlists must never be cached.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


class HttpServerHandle:
    """A static file server on 127.0.0.1 (returned by :func:`serve_directory`)."""

    def __init__(self, server: http.server.ThreadingHTTPServer, thread: threading.Thread) -> None:
        self._server = server
        self._thread = thread
        host, port = server.server_address[:2]
        self.host: str = str(host)
        self.port: int = int(port)
        self.base_url = f"http://{self.host}:{self.port}/"

    def url_for(self, name: str) -> str:
        return self.base_url + name.lstrip("/")

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(5.0)

    def __enter__(self) -> "HttpServerHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def serve_directory(directory: PathLike, host: str = "127.0.0.1", port: int = 0) -> HttpServerHandle:
    """Serve ``directory`` over HTTP on ``host`` (default loopback, OS-assigned port)."""
    handler = functools.partial(_QuietHandler, directory=str(Path(directory)))
    server = http.server.ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="hls-sim-http", daemon=True)
    thread.start()
    return HttpServerHandle(server, thread)


# ---------------------------------------------------------------------- Twitch-like server


def write_playlist(
    path: PathLike,
    names: Sequence[str],
    durations: Sequence[float],
    *,
    media_sequence: int = 0,
    endlist: bool = False,
    playlist_type: Optional[str] = None,
    target_duration: Optional[float] = None,
) -> Path:
    """Write a plain media playlist (atomically) listing ``names`` with ``durations``."""
    path = Path(path)
    target = target_duration if target_duration is not None else max([2.0, *durations])
    lines = ["#EXTM3U", "#EXT-X-VERSION:3", f"#EXT-X-TARGETDURATION:{int(round(target + 0.4999))}"]
    if playlist_type:
        lines.append(f"#EXT-X-PLAYLIST-TYPE:{playlist_type}")
    lines.append(f"#EXT-X-MEDIA-SEQUENCE:{media_sequence}")
    for name, d in zip(names, durations):
        lines += [f"#EXTINF:{d:.3f},", name]
    if endlist:
        lines.append("#EXT-X-ENDLIST")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


_PDT_BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _iso(d: datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // 1000:03d}Z"


class _TwitchHandler(_QuietHandler):
    server: "_TwitchHttpServer"

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        owner = self.server.owner
        parts = urllib.parse.urlsplit(self.path)
        name = urllib.parse.unquote(parts.path.lstrip("/"))
        owner._record(name)
        if name == owner.playlist_name:
            token = urllib.parse.parse_qs(parts.query).get("token", [""])[0]
            code, body = owner._playlist_response(token)
            self.send_response(code)
            self.send_header("Content-Type", "application/vnd.apple.mpegurl")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if "-unmuted." in name:
            self.send_error(403)
            return
        if name.startswith(("ad_", "prefetch_", "init")):
            self.send_error(404)
            return
        blocked = owner.segment_status
        if blocked is not None:
            self.send_error(blocked)
            return
        if "-muted." in name:
            self.path = "/" + name.replace("-muted.", ".")
        super().do_GET()


class _TwitchHttpServer(http.server.ThreadingHTTPServer):
    owner: "TwitchLikeServer"


class TwitchLikeServer:
    """Serve an HLS folder on 127.0.0.1 the way Twitch serves a stream.

    The playlist ``directory/playlist_name`` (written by :func:`start` or
    :func:`write_playlist`) is rewritten on every request:

    - ``ad_msns``: those entries become stitched ad segments: title
      ``"Amazon|..."`` (``ad_title``), covered by an ``#EXT-X-DATERANGE`` of
      class ``twitch-stitched-ad`` (``ad_daterange``), bracketed by
      ``#EXT-X-DISCONTINUITY``, URI ``ad_<msn>.ts`` (404: never fetchable).
      The content they replace is lost, as on Twitch.
    - ``program_date_time``: ``#EXT-X-PROGRAM-DATE-TIME`` on every entry.
    - ``prefetch``: two ``#EXT-X-TWITCH-PREFETCH`` lines at the end (live only).
    - ``extra_tags``: unknown Twitch tags (``#EXT-X-TWITCH-ELAPSED-SECS`` ...).
    - ``fmp4``: an ``#EXT-X-MAP`` line (fragmented MP4).
    - ``muted_msns``: listed as ``<stem>-unmuted.ts`` (answers 403) while
      ``<stem>-muted.ts`` serves the file, as some Twitch VODs do.

    The playlist URL carries a token (:attr:`playlist_url`). After
    :meth:`expire_token` the old URL answers 403; after :meth:`set_offline`
    the playlist answers 404. :meth:`resolve` is a fake resolver: it returns
    the current URL, raises ``StreamOffline`` when offline, and raises
    ``ResolveError`` (transient) ``fail_resolves`` times first. Setting
    :attr:`segment_status` (e.g. 403) makes every segment request answer that
    code while the playlist keeps working, like an edge host that refuses us.
    """

    def __init__(
        self,
        directory: PathLike,
        *,
        playlist_name: str = "playlist.m3u8",
        ad_msns: Iterable[int] = (),
        ad_title: bool = True,
        ad_daterange: bool = True,
        program_date_time: bool = True,
        prefetch: bool = True,
        extra_tags: bool = True,
        fmp4: bool = False,
        muted_msns: Iterable[int] = (),
        fail_resolves: int = 0,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.directory = Path(directory)
        self.playlist_name = playlist_name
        self.ad_msns: Collection[int] = frozenset(ad_msns)
        self.ad_title = ad_title
        self.ad_daterange = ad_daterange
        self.program_date_time = program_date_time
        self.prefetch = prefetch
        self.extra_tags = extra_tags
        self.fmp4 = fmp4
        self.muted_msns: Collection[int] = frozenset(muted_msns)
        self.fail_resolves = int(fail_resolves)
        self._lock = threading.Lock()
        self.token = 1
        self.offline = False
        self.segment_status: Optional[int] = None
        self.requests: List[str] = []
        self.resolve_calls = 0
        self.playlist_codes: Dict[int, int] = {}

        handler = functools.partial(_TwitchHandler, directory=str(self.directory))
        server = _TwitchHttpServer((host, port), handler)
        server.owner = self
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, name="twitch-sim-http", daemon=True)
        thread.start()
        self.http = HttpServerHandle(server, thread)

    # ------------------------------------------------------------- control

    @property
    def playlist_url(self) -> str:
        with self._lock:
            token = self.token
        return self.http.url_for(self.playlist_name) + f"?token={token}"

    def expire_token(self) -> None:
        with self._lock:
            self.token += 1

    def set_offline(self, offline: bool = True) -> None:
        with self._lock:
            self.offline = offline

    def resolve(self) -> str:
        from ..sources.hls import ResolveError, StreamOffline  # noqa: PLC0415 - avoid an import cycle

        with self._lock:
            self.resolve_calls += 1
            if self.offline:
                raise StreamOffline("simulated stream is offline")
            if self.fail_resolves > 0:
                self.fail_resolves -= 1
                raise ResolveError("simulated transient resolve failure")
        return self.playlist_url

    def requested(self, prefix: str = "") -> List[str]:
        with self._lock:
            return [r for r in self.requests if r.startswith(prefix)]

    def stop(self) -> None:
        self.http.stop()

    def __enter__(self) -> "TwitchLikeServer":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------- serving

    def _record(self, name: str) -> None:
        with self._lock:
            self.requests.append(name)

    def _playlist_response(self, token: str):
        with self._lock:
            offline, current = self.offline, self.token
        if offline:
            code, body = 404, b'[{"error":"Can not find channel"}]'
        elif token != str(current):
            code, body = 403, b'[{"error":"token expired"}]'
        else:
            try:
                text = (self.directory / self.playlist_name).read_text(encoding="utf-8")
            except OSError:
                code, body = 404, b""
            else:
                code, body = 200, self.rewrite(text).encode("utf-8")
        with self._lock:
            self.playlist_codes[code] = self.playlist_codes.get(code, 0) + 1
        return code, body

    def rewrite(self, text: str) -> str:
        """Turn a plain media playlist into a Twitch-flavoured one (see the class doc)."""
        header: List[str] = []
        entries: List[tuple] = []  # (msn, duration, uri, disc)
        msn = 0
        pending: Optional[float] = None
        disc = False
        endlist = False
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
                msn = int(line.split(":", 1)[1])
                header.append(line)
            elif line.startswith("#EXTINF:"):
                pending = float(line.split(":", 1)[1].split(",", 1)[0])
            elif line == "#EXT-X-DISCONTINUITY":
                disc = True
            elif line == "#EXT-X-ENDLIST":
                endlist = True
            elif line.startswith("#"):
                if not entries and pending is None:
                    header.append(line)
            elif pending is not None:
                entries.append((msn, pending, line, disc))
                msn += 1
                pending, disc = None, False

        def pdt(m: int) -> datetime:
            return _PDT_BASE + timedelta(seconds=2.0 * m)

        out = list(header)
        if self.fmp4:
            out.append('#EXT-X-MAP:URI="init.mp4"')
        if self.extra_tags:
            out += [
                "#EXT-X-TWITCH-ELAPSED-SECS:1234.567",
                "#EXT-X-TWITCH-TOTAL-SECS:1240.000",
                '#EXT-X-SESSION-DATA:DATA-ID="NODE",VALUE="sim"',
            ]
        if self.ad_daterange:
            run: List[tuple] = []
            for e in entries + [None]:  # type: ignore[list-item]
                if e is not None and e[0] in self.ad_msns:
                    run.append(e)
                    continue
                if run:
                    start = pdt(run[0][0])
                    dur = sum(r[1] for r in run)
                    out.append(
                        f'#EXT-X-DATERANGE:ID="stitched-ad-{run[0][0]}-{int(dur)}",CLASS="twitch-stitched-ad",'
                        f'START-DATE="{_iso(start)}",DURATION={2.0 * len(run) - 0.001:.3f},'
                        f'X-TV-TWITCH-AD-ROLL-TYPE="MIDROLL",X-TV-TWITCH-AD-POD-LENGTH="{len(run)}"'
                    )
                    run = []
        prev_ad: Optional[bool] = None
        for m, dur, uri, d in entries:
            is_ad = m in self.ad_msns
            if d or (prev_ad is not None and is_ad != prev_ad):
                out.append("#EXT-X-DISCONTINUITY")
            if self.program_date_time:
                out.append(f"#EXT-X-PROGRAM-DATE-TIME:{_iso(pdt(m))}")
            if self.extra_tags:
                out.append(f"#EXT-X-TWITCH-LIVE-SEQUENCE:{m}")
            if is_ad:
                title = "Amazon|9876543210" if self.ad_title else "live"
                out += [f"#EXTINF:{dur:.3f},{title}", f"ad_{m}.ts"]
            else:
                if m in self.muted_msns:
                    stem, dot, ext = uri.rpartition(".")
                    uri = f"{stem}-unmuted.{ext}" if dot else uri + "-unmuted"
                out += [f"#EXTINF:{dur:.3f},live", uri]
            prev_ad = is_ad
        if endlist:
            out.append("#EXT-X-ENDLIST")
        elif self.prefetch and entries:
            last = entries[-1][0]
            out += [f"#EXT-X-TWITCH-PREFETCH:prefetch_{last + 1}.ts", f"#EXT-X-TWITCH-PREFETCH:prefetch_{last + 2}.ts"]
        return "\n".join(out) + "\n"


@dataclasses.dataclass(frozen=True)
class FixedUrlResolver:
    """A picklable resolver that always returns ``url`` (for tests across processes).

    ``offline`` makes it raise ``StreamOffline`` instead. See
    :class:`FixedUrlFactory` for a picklable ``resolver_factory``.
    """

    url: str
    offline: bool = False

    def resolve(self) -> str:
        from ..sources.hls import StreamOffline  # noqa: PLC0415 - avoid an import cycle

        if self.offline:
            raise StreamOffline("simulated stream is offline")
        return self.url


@dataclasses.dataclass(frozen=True)
class FixedUrlFactory:
    """Picklable ``resolver_factory(target, quality)``: maps channel ident -> playlist URL (absent: offline)."""

    urls: Tuple[Tuple[str, str], ...]

    def __call__(self, target: Any, quality: str) -> FixedUrlResolver:
        url = dict(self.urls).get(target.ident)
        return FixedUrlResolver(url or "", offline=url is None)
