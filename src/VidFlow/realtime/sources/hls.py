"""HlsSource: follow an HLS media playlist as a live segment source (design 6.1).

The playlist may be a local file path or an http/https URL. Only the standard
library is used. Nothing here imports a streaming service's code: a
service-specific source (``twitch.py``) supplies the playlist URL through the
``refresh`` callable and this module does everything else.

Behaviour
    - The playlist is reloaded every ``poll_s`` seconds. Every segment that is
      new since the previous reload is copied/downloaded into ``work_dir`` and
      yielded once, in media-sequence order. A complete playlist
      (``#EXT-X-ENDLIST``, e.g. a VOD) is processed as fast as it downloads.
    - ``Segment.seq`` is this source's own counter (0, 1, 2, ...). The
      playlist's media sequence numbers are only used to recognise new
      segments and gaps.
    - ``Segment.start`` is the running sum of the ``#EXTINF`` durations of the
      segments yielded so far (design section 5); packet timestamps are never
      read.
    - ``discontinuity=True`` is set on a yielded segment when segments before
      it were never yielded (they scrolled out of the playlist before they
      were fetched, a fetch failed, or they were skipped as ads) or when the
      playlist marks one (``#EXT-X-DISCONTINUITY`` or a change of
      discontinuity sequence). The first yielded segment never carries the
      flag: there is nothing before it.
    - Ad segments (Twitch "stitched" ads: an ``#EXT-X-DATERANGE`` of class
      ``twitch-stitched-ad`` covering the segment's
      ``#EXT-X-PROGRAM-DATE-TIME``, or an ``#EXTINF`` title containing
      "Amazon") are never downloaded or yielded; the next content segment is
      flagged as a discontinuity, so no clip spans an ad break.
    - fMP4 (``#EXT-X-MAP``), encrypted (``#EXT-X-KEY``) and byte-range
      playlists raise :class:`UnsupportedPlaylistError` instead of producing
      segments that are not self-contained MPEG-TS. Other unknown tags
      (``#EXT-X-TWITCH-PREFETCH``, ...) are ignored.
    - Iteration ends at ``#EXT-X-ENDLIST`` (after the remaining segments), at
      ``end_offset_s``, when no new segment has appeared for
      ``stall_timeout_s``, when ``refresh`` reports the stream offline, or on
      :meth:`close`.

Reconnecting
    With ``refresh`` (a callable returning a fresh media playlist URL), a
    playlist answer of HTTP 403/404/410 (expired token, moved edge) or a stall
    triggers a re-resolve with exponential backoff. ``refresh`` raising
    :class:`StreamOffline` ends iteration cleanly (``end_reason="offline"``);
    :class:`ResolveError` with ``transient=False`` propagates; any other
    exception is retried until ``reconnect_timeout_s`` has passed, after which
    a :class:`ResolveError` is raised. Without ``refresh`` the old behaviour
    holds: errors are retried on the next poll and a stall ends iteration.

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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Union

from ..types import Segment

__all__ = [
    "PlaylistError",
    "UnsupportedPlaylistError",
    "StreamOffline",
    "ResolveError",
    "PlaylistSegment",
    "MediaPlaylist",
    "parse_media_playlist",
    "parse_attribute_list",
    "HlsSource",
]

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

_DEFAULT_TARGET_DURATION = 2.0
_MIN_POLL_S = 0.02
_FIRST_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 30.0
# Re-resolves in a row without a new segment before a stall is final.
_MAX_STALL_REFRESHES = 2


class PlaylistError(ValueError):
    """The text is not a usable HLS media playlist."""


class UnsupportedPlaylistError(RuntimeError):
    """A valid playlist this reader cannot turn into self-contained MPEG-TS segments.

    Not a :class:`PlaylistError`: it is never retried, it ends the source.
    """


class StreamOffline(Exception):
    """Raised by a ``refresh`` callable: the stream has ended or is offline."""


class ResolveError(RuntimeError):
    """Raised by a ``refresh`` callable when no playlist URL could be obtained.

    ``transient=False`` (bad channel name, quality not offered, missing
    dependency) ends the source with this error at once; transient errors are
    retried with backoff.
    """

    def __init__(self, message: str, *, transient: bool = True) -> None:
        super().__init__(message)
        self.transient = transient


@dataclass(frozen=True)
class PlaylistSegment:
    """One ``#EXTINF`` entry of a media playlist."""

    msn: int  # media sequence number
    uri: str  # as written in the playlist (resolve against the playlist's location)
    duration: float  # from #EXTINF
    discontinuity: bool = False  # preceded by #EXT-X-DISCONTINUITY
    discontinuity_seq: int = 0  # EXT-X-DISCONTINUITY-SEQUENCE + tags seen so far
    title: str = ""
    program_date_time: Optional[datetime] = None  # from #EXT-X-PROGRAM-DATE-TIME, carried forward
    ad: bool = False  # an inserted advertisement, not stream content


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

_ATTR_RE = re.compile(r'\s*([A-Za-z0-9_-]+)\s*=\s*("[^"]*"|[^,]*)\s*(?:,|$)')

# Twitch marks stitched ads this way (same rules as streamlink's Twitch plugin).
_AD_DATERANGE_CLASS = "twitch-stitched-ad"
_AD_DATERANGE_ID_PREFIX = "stitched-ad-"
_AD_TITLE_MARK = "Amazon"


def parse_attribute_list(value: str) -> Dict[str, str]:
    """Parse an HLS attribute list (``A=1,B="x,y"``) into a dict; quotes removed."""
    out: Dict[str, str] = {}
    for m in _ATTR_RE.finditer(value):
        v = m.group(2).strip()
        if len(v) >= 2 and v[0] == v[-1] == '"':
            v = v[1:-1]
        out[m.group(1).upper()] = v
    return out


def _parse_date(text: str) -> Optional[datetime]:
    """ISO 8601 date-time, tolerant of ``Z`` and odd fraction lengths; ``None`` if unparsable."""
    t = text.strip().strip('"')
    if not t:
        return None
    if t.endswith(("Z", "z")):
        t = t[:-1] + "+00:00"
    # Python < 3.11 only takes 3 or 6 fraction digits: normalise to 6.
    m = re.match(r"^(.*T\d\d:\d\d:\d\d)(?:\.(\d+))?(.*)$", t)
    if m:
        frac = (m.group(2) or "")[:6].ljust(6, "0") if m.group(2) else ""
        t = m.group(1) + (f".{frac}" if frac else "") + m.group(3)
    try:
        d = datetime.fromisoformat(t)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d


@dataclass(frozen=True)
class _AdRange:
    start: datetime
    end: Optional[datetime]

    def contains(self, d: datetime) -> bool:
        return self.start <= d and (self.end is None or d < self.end)


def _ad_range(value: str) -> Optional[_AdRange]:
    attrs = parse_attribute_list(value)
    cls = attrs.get("CLASS", "")
    ident = attrs.get("ID", "")
    if cls != _AD_DATERANGE_CLASS and not ident.startswith(_AD_DATERANGE_ID_PREFIX):
        return None
    start = _parse_date(attrs.get("START-DATE", ""))
    if start is None:
        return None
    end = _parse_date(attrs.get("END-DATE", ""))
    if end is None:
        for key in ("DURATION", "PLANNED-DURATION"):
            try:
                dur = float(attrs[key])
            except (KeyError, ValueError):
                continue
            if dur >= 0:
                end = start + timedelta(seconds=dur)
                break
    return _AdRange(start, end)


def parse_media_playlist(text: str) -> MediaPlaylist:
    """Parse the tags a live reader needs from an HLS media playlist.

    Understood: ``#EXTM3U``, ``#EXT-X-TARGETDURATION``,
    ``#EXT-X-MEDIA-SEQUENCE``, ``#EXT-X-DISCONTINUITY-SEQUENCE``, ``#EXTINF``,
    ``#EXT-X-DISCONTINUITY``, ``#EXT-X-ENDLIST``, ``#EXT-X-PLAYLIST-TYPE``,
    ``#EXT-X-PROGRAM-DATE-TIME`` and ``#EXT-X-DATERANGE`` (for ad marking).
    Other tags and comments are ignored. A master playlist
    (``#EXT-X-STREAM-INF``) or a missing ``#EXTM3U`` header raises
    :class:`PlaylistError`; ``#EXT-X-MAP`` (fMP4), ``#EXT-X-KEY`` with a
    method other than ``NONE`` and ``#EXT-X-BYTERANGE`` raise
    :class:`UnsupportedPlaylistError`.
    """
    lines = [ln.strip() for ln in text.lstrip("\ufeff").splitlines()]
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
    pdt: Optional[datetime] = None  # date of the next segment
    ad_ranges: List[_AdRange] = []
    raw: List[Tuple[str, float, bool, int, str, Optional[datetime]]] = []

    for line in lines[1:]:
        if not line.startswith("#"):
            if pending_duration is None:
                raise PlaylistError(f"segment URI without #EXTINF: {line!r}")
            raw.append((line, pending_duration, pending_disc, disc_count, pending_title, pdt))
            if pdt is not None:
                # Carried forward: a later segment without its own tag follows on.
                pdt = pdt + timedelta(seconds=pending_duration)
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
        elif tag == "#EXT-X-PROGRAM-DATE-TIME":
            pdt = _parse_date(value)  # unparsable: ignored (None)
        elif tag == "#EXT-X-DATERANGE":
            rng = _ad_range(value)
            if rng is not None:
                ad_ranges.append(rng)
        elif tag == "#EXT-X-STREAM-INF":
            raise PlaylistError("this is a master playlist; a media playlist is required")
        elif tag == "#EXT-X-MAP":
            raise UnsupportedPlaylistError(
                "fragmented-MP4 playlist (#EXT-X-MAP) is not supported: segments must be "
                "self-contained MPEG-TS (on Twitch, request H.264 renditions)"
            )
        elif tag == "#EXT-X-KEY":
            method = parse_attribute_list(value).get("METHOD", "NONE").upper()
            if method != "NONE":
                raise UnsupportedPlaylistError(f"encrypted playlist (#EXT-X-KEY METHOD={method}) is not supported")
        elif tag == "#EXT-X-BYTERANGE":
            raise UnsupportedPlaylistError("byte-range playlists (#EXT-X-BYTERANGE) are not supported")

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
            program_date_time=date,
            ad=(_AD_TITLE_MARK in title)
            or (date is not None and any(r.contains(date) for r in ad_ranges)),
        )
        for i, (uri, dur, disc, dcount, title, date) in enumerate(raw)
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


def _redact(url: str) -> str:
    """A URL without its query string (Twitch puts access tokens there), for logs."""
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


class HlsSource:
    """A :class:`~VidFlow.realtime.types.Source` that follows an HLS media playlist.

    ``playlist``          local path or http/https URL of a media playlist;
                          may be ``None`` when ``refresh`` is given (it is
                          then called first).
    ``channel``           channel name put on every segment.
    ``work_dir``          where segment files are copied/downloaded to.
    ``poll_s``            playlist reload interval; ``None`` means half the
                          last segment's duration, capped by half the target
                          duration (the HLS client rule for an unchanged
                          playlist).
    ``stall_timeout_s``   no new playlist entry for this long counts as a
                          stall (including before the playlist exists):
                          without ``refresh`` iteration ends, with it the URL
                          is re-resolved; ``None`` means
                          ``max(10, 3 x target duration)``.
    ``request_timeout_s`` timeout of one HTTP request.
    ``refresh``           callable returning a fresh media playlist URL; see
                          the module docstring.
    ``reconnect_timeout_s`` how long transient ``refresh`` failures are
                          retried (with backoff 1, 2, 4 ... 30 s) before a
                          :class:`ResolveError` is raised.
    ``start_offset_s`` / ``end_offset_s``  only yield the part of the stream
                          between these offsets, measured on the playlist's
                          own timeline from the first entry of the first
                          playlist loaded (for a VOD: from its start). Cuts
                          fall on segment boundaries: the segments covering
                          both offsets are included.
    ``skip_ads``          drop ad segments (default); ``False`` treats them as
                          content.
    ``headers``           extra HTTP request headers (e.g. a User-Agent).

    Single-use: iterate once. :meth:`close` is safe from any thread and makes
    a waiting iterator return at once; an HTTP request already in flight is
    not interrupted (it is bounded by ``request_timeout_s``).
    """

    def __init__(
        self,
        playlist: Optional[PathLike],
        channel: str,
        work_dir: PathLike,
        poll_s: Optional[float] = None,
        *,
        stall_timeout_s: Optional[float] = None,
        request_timeout_s: float = 10.0,
        refresh: Optional[Callable[[], str]] = None,
        reconnect_timeout_s: float = 120.0,
        start_offset_s: float = 0.0,
        end_offset_s: Optional[float] = None,
        skip_ads: bool = True,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        if playlist is None and refresh is None:
            raise ValueError("playlist or refresh must be given")
        self.playlist: Optional[str] = None if playlist is None else str(playlist)
        self.channel = channel
        self.work_dir = Path(work_dir)
        if poll_s is not None and not poll_s > 0:
            raise ValueError("poll_s must be > 0 or None")
        if stall_timeout_s is not None and not stall_timeout_s > 0:
            raise ValueError("stall_timeout_s must be > 0 or None")
        if not start_offset_s >= 0:
            raise ValueError("start_offset_s must be >= 0")
        if end_offset_s is not None and not end_offset_s > start_offset_s:
            raise ValueError("end_offset_s must be > start_offset_s")
        if not reconnect_timeout_s >= 0:
            raise ValueError("reconnect_timeout_s must be >= 0")
        self.poll_s = poll_s
        self.stall_timeout_s = stall_timeout_s
        self.request_timeout_s = float(request_timeout_s)
        self.refresh = refresh
        self.reconnect_timeout_s = float(reconnect_timeout_s)
        self.start_offset_s = float(start_offset_s)
        self.end_offset_s = None if end_offset_s is None else float(end_offset_s)
        self.skip_ads = bool(skip_ads)
        self.headers: Dict[str, str] = dict(headers or {})

        self._closed = threading.Event()
        self._started = False
        self._opener = urllib.request.build_opener()  # honours *_PROXY variables
        # Never send loopback traffic through an environment proxy.
        self._direct_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._load_status: Optional[int] = None  # HTTP status of the last failed playlist load

        # Statistics, readable after (or during) iteration.
        self.segments_yielded = 0
        self.segments_missed = 0  # listed or implied segments that were never yielded
        self.segments_ad = 0  # ad segments skipped
        self.reconnects = 0  # successful refresh() calls after the first
        # "endlist", "end_offset", "stall", "offline", "closed" or "error" (raised)
        self.end_reason: Optional[str] = None

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
        return self._guarded()

    def _guarded(self) -> Iterator[Segment]:
        try:
            yield from self._run()
        except Exception:
            if self.end_reason is None:
                self.end_reason = "error"
            raise

    # ------------------------------------------------------------------ main loop

    def _run(self) -> Iterator[Segment]:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if self.playlist is None:
            if not self._reresolve(initial=True):
                return
        next_msn: Optional[int] = None  # first media sequence number not yet handled
        pl_t = 0.0  # playlist time at the start of next_msn
        media_t = 0.0
        prev: Optional[Segment] = None
        prev_disc_seq: Optional[int] = None
        lost = False  # content was lost since the last yielded segment
        target: Optional[float] = None
        last_progress = time.monotonic()
        stall_refreshes = 0
        gone_refreshes = 0
        next_gone_refresh = 0.0

        while not self._closed.is_set():
            pl = self._load_playlist()
            if pl is not None:
                if pl.target_duration:
                    target = pl.target_duration
                if (
                    next_msn is not None
                    and pl.segments
                    and pl.last_msn < next_msn - 1 - max(3, len(pl.segments))  # type: ignore[operator]
                ):
                    # Media sequence went backwards by more than a stale copy
                    # could explain: a new broadcast. Start over from it.
                    log.warning("%s: media sequence restarted (%d -> %d)", self.channel, next_msn, pl.media_sequence)
                    next_msn = None
                    lost = True
                for ps in pl.segments:
                    if next_msn is not None and ps.msn < next_msn:
                        continue
                    if self._closed.is_set():
                        self.end_reason = "closed"
                        return
                    if next_msn is not None and ps.msn > next_msn:
                        missed = ps.msn - next_msn
                        self.segments_missed += missed
                        pl_t += missed * (target or _DEFAULT_TARGET_DURATION)
                        log.warning(
                            "%s: %d segment(s) scrolled out of the playlist before they were fetched",
                            self.channel, missed,
                        )
                        lost = True
                    next_msn = ps.msn + 1
                    entry_start, pl_t = pl_t, pl_t + ps.duration

                    if self.end_offset_s is not None and entry_start >= self.end_offset_s:
                        self.end_reason = "end_offset"
                        return
                    if pl_t <= self.start_offset_s:
                        continue  # before the requested start: not a loss
                    if ps.ad and self.skip_ads:
                        self.segments_ad += 1
                        lost = True
                        last_progress = time.monotonic()
                        stall_refreshes = gone_refreshes = 0
                        log.debug("%s: skipping ad segment %d", self.channel, ps.msn)
                        continue

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
                    stall_refreshes = gone_refreshes = 0
                if pl.endlist:
                    self.end_reason = "endlist"
                    return
            elif (
                self.refresh is not None
                and self._load_status in (403, 404, 410)
                and time.monotonic() >= next_gone_refresh
            ):
                # Expired token or the edge forgot the stream: get a new URL.
                # Backed off, so a resolver that keeps handing out a dead URL
                # is not hammered; the stall rule still ends it eventually.
                gone_refreshes += 1
                next_gone_refresh = time.monotonic() + min(_MAX_BACKOFF_S, _FIRST_BACKOFF_S * 2.0 ** (gone_refreshes - 1))
                log.info("%s: playlist answered %s; re-resolving", self.channel, self._load_status)
                if not self._reresolve():
                    return
                continue

            stall = self.stall_timeout_s
            if stall is None:
                stall = max(10.0, 3.0 * (target or _DEFAULT_TARGET_DURATION))
            if time.monotonic() - last_progress >= stall:
                if self.refresh is not None and stall_refreshes < _MAX_STALL_REFRESHES:
                    stall_refreshes += 1
                    log.info("%s: no new segment for %.1f s; re-resolving", self.channel, stall)
                    if not self._reresolve():
                        return
                    last_progress = time.monotonic()
                    continue
                log.warning("%s: no new segment for %.1f s; ending", self.channel, stall)
                self.end_reason = "stall"
                return
            poll = self.poll_s
            if poll is None:
                # Half a segment: Twitch announces a target duration well above
                # its real segment length, so the target alone would add delay.
                seg_len = target or _DEFAULT_TARGET_DURATION
                if pl is not None and pl.segments and pl.segments[-1].duration > 0:
                    seg_len = min(seg_len, pl.segments[-1].duration)
                poll = max(_MIN_POLL_S, seg_len / 2.0)
            if self._closed.wait(poll):
                break
        self.end_reason = "closed"

    def _reresolve(self, initial: bool = False) -> bool:
        """Get a new playlist URL from ``refresh``. ``False``: stop (``end_reason`` set)."""
        assert self.refresh is not None
        deadline = time.monotonic() + self.reconnect_timeout_s
        delay = _FIRST_BACKOFF_S
        while True:
            if self._closed.is_set():
                self.end_reason = "closed"
                return False
            try:
                url = self.refresh()
            except StreamOffline as exc:
                log.info("%s: stream is offline%s", self.channel, f" ({exc})" if str(exc) else "")
                self.end_reason = "offline"
                return False
            except ResolveError as exc:
                if not exc.transient:
                    raise
                err: Exception = exc
            except Exception as exc:  # noqa: BLE001 - any resolver failure is retried
                err = exc
            else:
                self.playlist = str(url)
                if not initial:
                    self.reconnects += 1
                log.debug("%s: playlist URL %s", self.channel, _redact(self.playlist))
                return True
            now = time.monotonic()
            if now + delay > deadline:
                raise ResolveError(
                    f"{self.channel}: could not resolve the stream for {self.reconnect_timeout_s:g} s: {err}",
                    transient=False,
                ) from err
            log.warning("%s: resolving failed (%s); retrying in %.0f s", self.channel, err, delay)
            if self._closed.wait(delay):
                self.end_reason = "closed"
                return False
            delay = min(_MAX_BACKOFF_S, delay * 2.0)

    # ------------------------------------------------------------------ I/O

    def _open(self, url: str):
        loopback = _is_loopback(urllib.parse.urlsplit(url).hostname)
        opener = self._direct_opener if loopback else self._opener
        req = urllib.request.Request(url, headers=self.headers)
        return opener.open(req, timeout=self.request_timeout_s)

    def _load_playlist(self) -> Optional[MediaPlaylist]:
        self._load_status = None
        assert self.playlist is not None
        try:
            if _is_url(self.playlist):
                with self._open(self.playlist) as resp:
                    text = resp.read().decode("utf-8", "replace")
            else:
                text = Path(self.playlist).read_text(encoding="utf-8", errors="replace")
            return parse_media_playlist(text)
        except urllib.error.HTTPError as exc:
            self._load_status = exc.code
            log.debug("%s: playlist answered HTTP %s", self.channel, exc.code)
            return None
        except (OSError, urllib.error.URLError, PlaylistError) as exc:
            # Missing (not written yet), mid-rename on Windows, or a transient
            # network error: try again on the next poll; the stall timeout
            # bounds how long that goes on.
            log.debug("%s: playlist not readable: %s", self.channel, exc)
            return None

    def _resolve(self, uri: str) -> Tuple[bool, str]:
        """Return (is_url, location) of a segment URI."""
        assert self.playlist is not None
        if _is_url(uri):
            return True, uri
        if _is_url(self.playlist):
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
        attempt = 0
        while attempt < attempts:
            attempt += 1
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
                    alt = _muted_alternative(loc)
                    if alt is not None:
                        # Twitch VODs list "N-unmuted.ts" for muted parts; only "N-muted.ts" exists.
                        log.debug("%s: segment %d: trying the muted copy", self.channel, ps.msn)
                        loc = alt
                        continue
                    log.warning("%s: segment %d gone (%s)", self.channel, ps.msn, exc.code)
                    return None
                log.warning("%s: segment %d fetch failed: %s", self.channel, ps.msn, exc)
            except (OSError, urllib.error.URLError) as exc:
                self._discard(part)
                log.warning("%s: segment %d fetch failed: %s", self.channel, ps.msn, exc)
                if not is_url:
                    return None
            if attempt < attempts and self._closed.wait(0.1):
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


def _muted_alternative(url: str) -> Optional[str]:
    parts = urllib.parse.urlsplit(url)
    if "-unmuted." not in parts.path:
        return None
    path = parts.path.replace("-unmuted.", "-muted.")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))
