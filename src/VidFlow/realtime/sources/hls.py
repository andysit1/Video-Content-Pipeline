"""HlsSource: follow an HLS media playlist as a live segment source (design 6.1).

The playlist may be a local file path or an http/https URL. Only the standard
library is used. Nothing here is specific to any streaming service: a later
service-specific source only has to supply a media playlist URL (and handle
its own ads).

Behaviour
    - The playlist is reloaded every ``poll_s`` seconds. Every segment that is
      new since the previous reload is copied/downloaded into ``work_dir`` and
      yielded once, in media-sequence order.
    - ``Segment.seq`` is this source's own counter (0, 1, 2, ...). The
      playlist's media sequence numbers are only used to recognise new
      segments and gaps.
    - ``Segment.start`` is the running sum of the ``#EXTINF`` durations of the
      segments yielded so far (design section 5); packet timestamps are never
      read.
    - ``discontinuity=True`` is set on a yielded segment when segments before
      it were never yielded (they scrolled out of the playlist before they
      were fetched, or a fetch failed) or when the playlist marks one
      (``#EXT-X-DISCONTINUITY`` or a change of discontinuity sequence). The
      first yielded segment never carries the flag: there is nothing before it.
    - Iteration ends at ``#EXT-X-ENDLIST`` (after the remaining segments), when
      no new segment has appeared for ``stall_timeout_s``, or on :meth:`close`.

Files written to ``work_dir`` belong to the consumer afterwards (normally the
``SegmentBuffer``); the source never deletes a file it has yielded.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

from ..types import Segment

__all__ = [
    "PlaylistError",
    "PlaylistSegment",
    "MediaPlaylist",
    "parse_media_playlist",
    "HlsSource",
]

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

_DEFAULT_TARGET_DURATION = 2.0
_MIN_POLL_S = 0.02


class PlaylistError(ValueError):
    """The text is not a usable HLS media playlist."""


@dataclass(frozen=True)
class PlaylistSegment:
    """One ``#EXTINF`` entry of a media playlist."""

    msn: int  # media sequence number
    uri: str  # as written in the playlist (resolve against the playlist's location)
    duration: float  # from #EXTINF
    discontinuity: bool = False  # preceded by #EXT-X-DISCONTINUITY
    discontinuity_seq: int = 0  # EXT-X-DISCONTINUITY-SEQUENCE + tags seen so far
    title: str = ""


@dataclass(frozen=True)
class MediaPlaylist:
    target_duration: Optional[float]
    media_sequence: int
    discontinuity_sequence: int
    segments: Tuple[PlaylistSegment, ...]
    endlist: bool
    playlist_type: Optional[str] = None  # "EVENT", "VOD" or None

    @property
    def last_msn(self) -> Optional[int]:
        return self.segments[-1].msn if self.segments else None


_INT_TAGS = {
    "#EXT-X-MEDIA-SEQUENCE": "media_sequence",
    "#EXT-X-DISCONTINUITY-SEQUENCE": "discontinuity_sequence",
}


def parse_media_playlist(text: str) -> MediaPlaylist:
    """Parse the tags a live reader needs from an HLS media playlist.

    Understood: ``#EXTM3U``, ``#EXT-X-TARGETDURATION``,
    ``#EXT-X-MEDIA-SEQUENCE``, ``#EXT-X-DISCONTINUITY-SEQUENCE``, ``#EXTINF``,
    ``#EXT-X-DISCONTINUITY``, ``#EXT-X-ENDLIST``, ``#EXT-X-PLAYLIST-TYPE``.
    Other tags and comments are ignored. A master playlist
    (``#EXT-X-STREAM-INF``) or a missing ``#EXTM3U`` header raises
    :class:`PlaylistError`.
    """
    lines = [ln.strip() for ln in text.lstrip("﻿").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines or not lines[0].startswith("#EXTM3U"):
        raise PlaylistError("not an HLS playlist (missing #EXTM3U)")

    target: Optional[float] = None
    ints = {"media_sequence": 0, "discontinuity_sequence": 0}
    endlist = False
    playlist_type: Optional[str] = None
    pending_duration: Optional[float] = None
    pending_title = ""
    pending_disc = False
    disc_count = 0
    raw: List[Tuple[str, float, bool, int, str]] = []

    for line in lines[1:]:
        if not line.startswith("#"):
            if pending_duration is None:
                raise PlaylistError(f"segment URI without #EXTINF: {line!r}")
            raw.append((line, pending_duration, pending_disc, disc_count, pending_title))
            pending_duration, pending_title, pending_disc = None, "", False
            continue
        tag, _, value = line.partition(":")
        if tag == "#EXTINF":
            dur, _, title = value.partition(",")
            try:
                pending_duration = float(dur)
            except ValueError as exc:
                raise PlaylistError(f"bad #EXTINF duration: {line!r}") from exc
            if pending_duration < 0:
                raise PlaylistError(f"negative #EXTINF duration: {line!r}")
            pending_title = title
        elif tag == "#EXT-X-TARGETDURATION":
            try:
                target = float(value)
            except ValueError as exc:
                raise PlaylistError(f"bad #EXT-X-TARGETDURATION: {line!r}") from exc
        elif tag in _INT_TAGS:
            try:
                ints[_INT_TAGS[tag]] = int(value)
            except ValueError as exc:
                raise PlaylistError(f"bad {tag}: {line!r}") from exc
        elif tag == "#EXT-X-DISCONTINUITY":
            pending_disc = True
            disc_count += 1
        elif tag == "#EXT-X-ENDLIST":
            endlist = True
        elif tag == "#EXT-X-PLAYLIST-TYPE":
            playlist_type = value.strip().upper() or None
        elif tag == "#EXT-X-STREAM-INF":
            raise PlaylistError("this is a master playlist; a media playlist is required")

    base_msn = ints["media_sequence"]
    base_disc = ints["discontinuity_sequence"]
    segments = tuple(
        PlaylistSegment(
            msn=base_msn + i,
            uri=uri,
            duration=dur,
            discontinuity=disc,
            discontinuity_seq=base_disc + dcount,
            title=title,
        )
        for i, (uri, dur, disc, dcount, title) in enumerate(raw)
    )
    return MediaPlaylist(
        target_duration=target,
        media_sequence=base_msn,
        discontinuity_sequence=base_disc,
        segments=segments,
        endlist=endlist,
        playlist_type=playlist_type,
    )


def _is_url(s: str) -> bool:
    return urllib.parse.urlsplit(s).scheme.lower() in ("http", "https")


def _is_loopback(host: Optional[str]) -> bool:
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class HlsSource:
    """A :class:`~VidFlow.realtime.types.Source` that follows an HLS media playlist.

    ``playlist``          local path or http/https URL of a media playlist.
    ``channel``           channel name put on every segment.
    ``work_dir``          where segment files are copied/downloaded to.
    ``poll_s``            playlist reload interval; ``None`` means half the
                          playlist's target duration (the HLS client rule for
                          an unchanged playlist).
    ``stall_timeout_s``   end iteration when no new segment has appeared for
                          this long (including before the playlist exists);
                          ``None`` means ``max(10, 3 x target duration)``.
    ``request_timeout_s`` timeout of one HTTP request.

    Single-use: iterate once. :meth:`close` is safe from any thread and makes
    a waiting iterator return at once; an HTTP request already in flight is
    not interrupted (it is bounded by ``request_timeout_s``).
    """

    def __init__(
        self,
        playlist: PathLike,
        channel: str,
        work_dir: PathLike,
        poll_s: Optional[float] = None,
        *,
        stall_timeout_s: Optional[float] = None,
        request_timeout_s: float = 10.0,
    ) -> None:
        self.playlist = str(playlist)
        self.channel = channel
        self.work_dir = Path(work_dir)
        if poll_s is not None and not poll_s > 0:
            raise ValueError("poll_s must be > 0 or None")
        if stall_timeout_s is not None and not stall_timeout_s > 0:
            raise ValueError("stall_timeout_s must be > 0 or None")
        self.poll_s = poll_s
        self.stall_timeout_s = stall_timeout_s
        self.request_timeout_s = float(request_timeout_s)

        self._is_url = _is_url(self.playlist)
        self._closed = threading.Event()
        self._started = False
        self._opener = urllib.request.build_opener()  # honours *_PROXY variables
        # Never send loopback traffic through an environment proxy.
        self._direct_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        # Statistics, readable after (or during) iteration.
        self.segments_yielded = 0
        self.segments_missed = 0  # listed or implied segments that were never yielded
        self.end_reason: Optional[str] = None  # "endlist", "stall" or "closed"

    # ------------------------------------------------------------------ Source

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def close(self) -> None:
        self._closed.set()

    def __iter__(self) -> Iterator[Segment]:
        if self._started:
            raise RuntimeError("HlsSource can only be iterated once")
        self._started = True
        return self._run()

    # ------------------------------------------------------------------ main loop

    def _run(self) -> Iterator[Segment]:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        next_msn: Optional[int] = None  # first media sequence number not yet handled
        media_t = 0.0
        prev: Optional[Segment] = None
        prev_disc_seq: Optional[int] = None
        lost = False  # content was lost since the last yielded segment
        target: Optional[float] = None
        last_progress = time.monotonic()

        while not self._closed.is_set():
            pl = self._load_playlist()
            if pl is not None:
                if pl.target_duration:
                    target = pl.target_duration
                for ps in pl.segments:
                    if next_msn is not None and ps.msn < next_msn:
                        continue
                    if self._closed.is_set():
                        self.end_reason = "closed"
                        return
                    if next_msn is not None and ps.msn > next_msn:
                        missed = ps.msn - next_msn
                        self.segments_missed += missed
                        log.warning(
                            "%s: %d segment(s) scrolled out of the playlist before they were fetched",
                            self.channel, missed,
                        )
                        lost = True
                    next_msn = ps.msn + 1

                    seq = self.segments_yielded
                    path = self._fetch(ps, seq)
                    if path is None:
                        self.segments_missed += 1
                        lost = True
                        continue

                    disc = prev is not None and (
                        lost or ps.discontinuity or ps.discontinuity_seq != prev_disc_seq
                    )
                    if prev is not None:
                        media_t = prev.start + prev.duration
                    seg = Segment(
                        channel=self.channel,
                        seq=seq,
                        start=media_t,
                        duration=ps.duration,
                        path=path,
                        discontinuity=disc,
                        arrived_at=time.monotonic(),
                    )
                    prev, prev_disc_seq, lost = seg, ps.discontinuity_seq, False
                    self.segments_yielded += 1
                    yield seg
                    # Time spent by the consumer does not count as a stall.
                    last_progress = time.monotonic()
                if pl.endlist:
                    self.end_reason = "endlist"
                    return

            stall = self.stall_timeout_s
            if stall is None:
                stall = max(10.0, 3.0 * (target or _DEFAULT_TARGET_DURATION))
            if time.monotonic() - last_progress >= stall:
                log.warning("%s: no new segment for %.1f s; ending", self.channel, stall)
                self.end_reason = "stall"
                return
            poll = self.poll_s
            if poll is None:
                poll = max(_MIN_POLL_S, (target or _DEFAULT_TARGET_DURATION) / 2.0)
            if self._closed.wait(poll):
                break
        self.end_reason = "closed"

    # ------------------------------------------------------------------ I/O

    def _open(self, url: str):
        loopback = _is_loopback(urllib.parse.urlsplit(url).hostname)
        opener = self._direct_opener if loopback else self._opener
        return opener.open(url, timeout=self.request_timeout_s)

    def _load_playlist(self) -> Optional[MediaPlaylist]:
        try:
            if self._is_url:
                with self._open(self.playlist) as resp:
                    text = resp.read().decode("utf-8", "replace")
            else:
                text = Path(self.playlist).read_text(encoding="utf-8", errors="replace")
            return parse_media_playlist(text)
        except (OSError, urllib.error.URLError, PlaylistError) as exc:
            # Missing (not written yet), mid-rename on Windows, or a transient
            # network error: try again on the next poll; the stall timeout
            # bounds how long that goes on.
            log.debug("%s: playlist not readable: %s", self.channel, exc)
            return None

    def _resolve(self, uri: str) -> Tuple[bool, str]:
        """Return (is_url, location) of a segment URI."""
        if _is_url(uri):
            return True, uri
        if self._is_url:
            return True, urllib.parse.urljoin(self.playlist, uri)
        if uri.lower().startswith("file:"):
            return False, urllib.request.url2pathname(urllib.parse.urlsplit(uri).path)
        p = Path(urllib.parse.unquote(uri))
        if not p.is_absolute():
            p = Path(self.playlist).parent / p
        return False, str(p)

    def _fetch(self, ps: PlaylistSegment, seq: int) -> Optional[Path]:
        """Copy or download one segment into ``work_dir``; ``None`` if it is gone."""
        is_url, loc = self._resolve(ps.uri)
        dest = self.work_dir / f"{self.channel}_{seq:06d}.ts"
        part = dest.with_name(dest.name + ".part")
        attempts = 2 if is_url else 1
        for attempt in range(attempts):
            try:
                if is_url:
                    with self._open(loc) as resp, open(part, "wb") as fh:
                        shutil.copyfileobj(resp, fh)
                else:
                    shutil.copyfile(loc, part)
                os.replace(part, dest)
                return dest
            except urllib.error.HTTPError as exc:
                self._discard(part)
                if exc.code in (403, 404, 410):
                    log.warning("%s: segment %d gone (%s)", self.channel, ps.msn, exc.code)
                    return None
                log.warning("%s: segment %d fetch failed: %s", self.channel, ps.msn, exc)
            except (OSError, urllib.error.URLError) as exc:
                self._discard(part)
                log.warning("%s: segment %d fetch failed: %s", self.channel, ps.msn, exc)
                if not is_url:
                    return None
            if attempt + 1 < attempts and self._closed.wait(0.1):
                return None
        return None

    @staticmethod
    def _discard(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:  # pragma: no cover - Windows file locks
            log.debug("could not remove %s: %s", path, exc)

