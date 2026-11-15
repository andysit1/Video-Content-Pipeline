"""Local live-HLS simulator: a video file played out as a sliding-window playlist.

:func:`start` runs FFmpeg's HLS muxer as a subprocess, reading the input at
``speed`` times real time, so the playlist on disk behaves like a live
stream's: new segments appear as they would be broadcast, old ones scroll out
of the playlist (and are deleted), and ``#EXT-X-ENDLIST`` is written when the
file is finished. :func:`serve_directory` optionally serves the output folder
over HTTP on 127.0.0.1 so the URL code path of ``HlsSource`` can be exercised.

Nothing here talks to any outside host.
"""

from __future__ import annotations

import functools
import http.server
import logging
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional, Union

from ..media import MediaError, ffmpeg_exe

__all__ = ["HlsSim", "start", "HttpServerHandle", "serve_directory"]

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
