"""TwitchSource: follow a Twitch channel (live) or VOD without the Twitch API.

Resolution does what the web player does: ask for a playback access token and
the "usher" master playlist, then pick one rendition's media playlist. That
part is delegated to the optional ``streamlink`` library (install the
``twitch`` extra). Everything after it (polling the media playlist,
downloading the MPEG-TS segments as they are, media time, ads, reconnects) is
our own :class:`~VidFlow.realtime.sources.hls.HlsSource`; nothing is decoded,
rendered or re-encoded on the way in.

No API key, client ID of our own or account is used.

Pieces
    :func:`parse_target`   ``"name"``, ``"twitch.tv/<name>"`` (live) or
                           ``"twitch.tv/videos/<id>"`` (VOD, ``?t=`` honoured)
                           -> :class:`TwitchTarget`.
    :class:`Resolver`      anything with ``resolve() -> media playlist URL``,
                           raising :class:`StreamOffline` when the stream is
                           offline/ended and :class:`ResolveError` otherwise.
                           Tests substitute a fake that points at ``hls_sim``.
    :class:`StreamlinkResolver`  the real one.
    :class:`TwitchSource`  the :class:`~VidFlow.realtime.types.Source`.

Quality
    ``quality`` is a comma-separated preference list of streamlink stream
    names; the first one offered wins (``"720p60,720p,best"``). The default is
    :data:`DEFAULT_QUALITY` (``"best"``, the broadcaster's own source
    rendition): clips are the product and a 9:16 crop of 720p is visibly
    soft, while our CPU cost barely depends on the rendition (audio analysis
    is the same; video is decoded only for candidate windows). What scales is
    download bandwidth and buffer disk: source is about 6-8 Mbps per channel
    (60-110 MB of buffer at the default 120 s retention), 720p60 about
    3-3.5 Mbps, 480p about 1.5 Mbps. When following many channels on a
    limited link, pass ``--quality 720p60,720p,best``. Only H.264 renditions
    are requested, so segments are MPEG-TS (HEVC/AV1 renditions are fMP4).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import threading
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Union

from ..types import Segment
from .hls import HlsSource, ResolveError, SegmentFetchError, StreamOffline

__all__ = [
    "DEFAULT_QUALITY",
    "INSTALL_HINT",
    "TwitchTarget",
    "parse_target",
    "parse_time",
    "Resolver",
    "StreamlinkResolver",
    "TwitchSource",
    "TwitchFactory",
    "StreamOffline",
    "SegmentFetchError",
    "ResolveError",
    "require_streamlink",
]

log = logging.getLogger(__name__)

PathLike = Union[str, Path]

DEFAULT_QUALITY = "best"
INSTALL_HINT = "pip install 'VidFlow[twitch]'  (or: pip install streamlink)"

_LOGIN_RE = re.compile(r"^[A-Za-z0-9_]{1,25}$")
_VOD_PATH_RE = re.compile(r"^/(?:[\w-]+/)?v(?:ideos?)?/(\d+)/?$")
_LIVE_PATH_RE = re.compile(r"^/([A-Za-z0-9_]{1,25})/?$")
_RESERVED = {"videos", "directory", "settings", "search", "p", "downloads", "jobs", "turbo", "subscriptions"}


@dataclass(frozen=True)
class TwitchTarget:
    """What to follow: a live channel or a VOD."""

    kind: str  # "live" or "vod"
    ident: str  # channel login (lower case) or video id
    start_s: float = 0.0  # from a VOD URL's ?t=

    @property
    def is_live(self) -> bool:
        return self.kind == "live"

    @property
    def url(self) -> str:
        if self.is_live:
            return f"https://www.twitch.tv/{self.ident}"
        return f"https://www.twitch.tv/videos/{self.ident}"

    @property
    def default_channel(self) -> str:
        """Name used for the output folder: the login, or ``vod-<id>``."""
        return self.ident if self.is_live else f"vod-{self.ident}"


def parse_time(text: Union[str, float, int, None]) -> float:
    """Seconds from ``"90"``, ``"1:30"``, ``"1:02:03"``, ``"1h2m3s"`` or ``"45s"``."""
    if text is None:
        raise ValueError("no time given")
    if isinstance(text, (int, float)):
        v = float(text)
    else:
        t = text.strip().lower()
        if not t:
            raise ValueError("empty time")
        if ":" in t:
            parts = t.split(":")
            if len(parts) > 3:
                raise ValueError(f"bad time {text!r}")
            try:
                nums = [float(p) for p in parts]
            except ValueError as exc:
                raise ValueError(f"bad time {text!r}") from exc
            v = 0.0
            for n in nums:
                v = v * 60.0 + n
        else:
            m = re.fullmatch(r"(?:(\d+(?:\.\d*)?)h)?(?:(\d+(?:\.\d*)?)m)?(?:(\d+(?:\.\d*)?)s?)?", t)
            if not m or not any(m.groups()):
                raise ValueError(f"bad time {text!r}")
            h, mi, s = (float(g) if g else 0.0 for g in m.groups())
            v = h * 3600.0 + mi * 60.0 + s
    if not v >= 0:
        raise ValueError(f"time must be >= 0, got {text!r}")
    return v


def parse_target(text: str) -> TwitchTarget:
    """Parse a channel name or a twitch.tv live/VOD URL. Raises ``ValueError``."""
    raw = text.strip()
    if not raw:
        raise ValueError("empty channel")
    if _LOGIN_RE.fullmatch(raw):
        return TwitchTarget("live", raw.lower())
    s = raw if "://" in raw else "https://" + raw
    parts = urllib.parse.urlsplit(s)
    host = (parts.hostname or "").lower()
    if not (host == "twitch.tv" or host.endswith(".twitch.tv")):
        raise ValueError(f"not a Twitch channel name or twitch.tv URL: {text!r}")
    if host == "clips.twitch.tv":
        raise ValueError(f"clips are not supported: {text!r}")
    query = urllib.parse.parse_qs(parts.query)
    if host == "player.twitch.tv":
        if query.get("video"):
            vid = query["video"][0].lstrip("v")
            if vid.isdigit():
                return TwitchTarget("vod", vid, _start_from_query(query))
        if query.get("channel") and _LOGIN_RE.fullmatch(query["channel"][0]):
            return TwitchTarget("live", query["channel"][0].lower())
        raise ValueError(f"unsupported player URL: {text!r}")
    m = _VOD_PATH_RE.match(parts.path)
    if m:
        return TwitchTarget("vod", m.group(1), _start_from_query(query))
    m = _LIVE_PATH_RE.match(parts.path)
    if m and m.group(1).lower() not in _RESERVED:
        return TwitchTarget("live", m.group(1).lower())
    raise ValueError(f"not a live channel or VOD URL (clips are not supported): {text!r}")


def _start_from_query(query: Mapping[str, List[str]]) -> float:
    t = (query.get("t") or [""])[0]
    if not t:
        return 0.0
    try:
        return parse_time(t)
    except ValueError:
        log.warning("ignoring unparsable ?t=%s", t)
        return 0.0


# --------------------------------------------------------------------------- resolution


class Resolver(Protocol):
    """Turns a target into a media playlist URL (called again on every reconnect)."""

    def resolve(self) -> str:
        """Return a media playlist URL; raise ``StreamOffline`` or ``ResolveError``."""
        ...


def require_streamlink() -> Any:
    """Import and return the ``streamlink`` module, or raise a non-transient ResolveError."""
    try:
        import streamlink  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError as exc:
        raise ResolveError(
            f"following Twitch needs the optional 'streamlink' package: {INSTALL_HINT}",
            transient=False,
        ) from exc
    return streamlink


def _qualities(spec: Union[str, Sequence[str]]) -> List[str]:
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    out = [q.strip() for q in items if q and q.strip()]
    if not out:
        raise ValueError("quality must name at least one stream")
    return out


class StreamlinkResolver:
    """Resolve a :class:`TwitchTarget` through streamlink's Twitch plugin.

    Each call asks for a fresh access token, so it is also how an expired
    playlist URL is renewed. One streamlink session per resolver (sessions
    are not shared between channel threads).

    ``quality``     preference list (see the module docstring).
    ``low_latency`` ask for the low-latency playlist flavour; the prefetch
                    entries it adds are ignored by ``HlsSource`` anyway.
    ``session_options``  extra streamlink session options (e.g. ``http-proxy``).
    """

    def __init__(
        self,
        target: TwitchTarget,
        quality: Union[str, Sequence[str]] = DEFAULT_QUALITY,
        *,
        low_latency: bool = False,
        session_options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.target = target
        self.qualities = _qualities(quality)
        self.low_latency = bool(low_latency)
        self.session_options = dict(session_options or {})
        self._session: Any = None
        self.chosen_quality: Optional[str] = None
        self.available: List[str] = []

    def _get_session(self) -> Any:
        if self._session is None:
            streamlink = require_streamlink()
            session = streamlink.Streamlink(options=self.session_options or None)
            # Use the environment's CA bundle like the rest of the process does
            # (requests otherwise uses certifi). Never turns verification off.
            ca = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
            if ca and os.path.isfile(ca) and session.http.verify is True:
                session.http.verify = ca
            self._session = session
        return self._session

    @property
    def user_agent(self) -> Optional[str]:
        """The User-Agent streamlink sends (reused for segment downloads)."""
        try:
            return str(self._get_session().http.headers.get("User-Agent") or "") or None
        except ResolveError:
            return None

    def resolve(self) -> str:
        streamlink = require_streamlink()
        from streamlink.exceptions import NoPluginError, NoStreamsError, PluginError  # noqa: PLC0415
        from streamlink.options import Options  # noqa: PLC0415

        session = self._get_session()
        opts = Options({"supported-codecs": ["h264"], "low-latency": self.low_latency})
        name = self.target.default_channel
        try:
            _plugin_name, plugin_cls, resolved_url = session.resolve_url(self.target.url)
            plugin = plugin_cls(session, resolved_url, opts)
            streams = plugin.streams()
            # streamlink answers "no streams" both when the channel is offline
            # and when the stream's media playlists could not be loaded (it
            # probes them). Tell the two apart with the metadata query: a
            # live broadcast (or an existing VOD) has an id.
            broadcast_id = None if streams else plugin.get_id()
        except NoPluginError as exc:  # pragma: no cover - our URLs always match
            raise ResolveError(f"streamlink has no plugin for {self.target.url}", transient=False) from exc
        except NoStreamsError as exc:
            raise StreamOffline(f"{name}: offline or no such channel") from exc
        except PluginError as exc:
            raise ResolveError(f"{name}: {exc}") from exc
        except OSError as exc:  # requests' errors derive from OSError
            raise ResolveError(f"{name}: network error: {exc}") from exc
        except streamlink.exceptions.StreamlinkError as exc:
            raise ResolveError(f"{name}: {exc}") from exc
        if not streams:
            if broadcast_id:
                what = "is live" if self.target.is_live else "exists"
                raise ResolveError(
                    f"{name} {what} (id {broadcast_id}) but none of its media playlists could be "
                    "loaded (playlist host unreachable or access restricted)"
                )
            raise StreamOffline(f"{name}: offline or no such channel")
        self.available = sorted(streams)
        for q in self.qualities:
            stream = streams.get(q)
            if stream is None:
                continue
            url = getattr(stream, "url", None)
            if not url:
                continue
            if self.chosen_quality != q:
                log.info("%s: following quality %s", self.target.default_channel, q)
            self.chosen_quality = q
            return str(url)
        raise ResolveError(
            f"{self.target.default_channel}: none of the qualities {','.join(self.qualities)} is offered; "
            f"available: {', '.join(self.available)}",
            transient=False,
        )


class _CallableResolver:
    def __init__(self, fn: Callable[[], str]) -> None:
        self._fn = fn

    def resolve(self) -> str:
        return self._fn()


# --------------------------------------------------------------------------- source


class TwitchSource:
    """A :class:`~VidFlow.realtime.types.Source` for one Twitch channel or VOD.

    ``target``     a :class:`TwitchTarget` or anything :func:`parse_target` takes.
    ``work_dir``   where segments are downloaded (the pipeline's buffer folder).
    ``quality``    used only when ``resolver`` is not given.
    ``resolver``   a :class:`Resolver` (or a plain callable returning a URL);
                   default :class:`StreamlinkResolver`.
    ``channel``    channel name on segments and output folders; default the
                   login, or ``vod-<id>``.
    ``start_s`` / ``end_s``  slice of a VOD (seconds from its start; ``start_s``
                   defaults to the URL's ``?t=``). For a live channel they
                   count from when following began.
    ``wait``       live channels only: when the channel is offline, check every
                   ``offline_poll_s`` seconds until it goes live, and after each
                   broadcast ends go back to waiting for the next one. The
                   source then ends only when closed (``--duration``, Ctrl-C).
                   Broadcasts are chained: ``seq`` and media time continue,
                   and the first segment of each later broadcast is flagged
                   ``discontinuity``.
    ``hls_options`` passed to :class:`HlsSource` (``poll_s``,
                   ``stall_timeout_s``, ``reconnect_timeout_s`` ...).

    Without ``wait`` a live channel ends when the stream goes offline
    (``end_reason == "offline"``), a VOD at its end (``"endlist"``) or at
    ``end_s`` (``"end_offset"``), and a channel that is offline when following
    starts yields nothing and ends with ``"offline"``. If the stream ends
    while its segments could not be downloaded, iteration raises
    :class:`~VidFlow.realtime.sources.hls.SegmentFetchError`.
    """

    def __init__(
        self,
        target: Union[TwitchTarget, str],
        work_dir: PathLike,
        *,
        quality: Union[str, Sequence[str]] = DEFAULT_QUALITY,
        resolver: Union[Resolver, Callable[[], str], None] = None,
        channel: Optional[str] = None,
        start_s: Optional[float] = None,
        end_s: Optional[float] = None,
        wait: bool = False,
        offline_poll_s: float = 60.0,
        **hls_options: Any,
    ) -> None:
        self.target = target if isinstance(target, TwitchTarget) else parse_target(target)
        self.channel: str = channel or self.target.default_channel
        if resolver is None:
            resolver = StreamlinkResolver(self.target, quality)
        elif not hasattr(resolver, "resolve") and callable(resolver):
            resolver = _CallableResolver(resolver)
        self.resolver: Resolver = resolver  # type: ignore[assignment]
        if wait and not self.target.is_live:
            raise ValueError("wait applies to live channels only")
        if not offline_poll_s > 0:
            raise ValueError("offline_poll_s must be > 0")
        self.wait = bool(wait)
        self.offline_poll_s = float(offline_poll_s)
        self.work_dir = Path(work_dir)
        self._start_s = self.target.start_s if start_s is None else float(start_s)
        self._end_s = end_s
        headers: Dict[str, str] = dict(hls_options.pop("headers", None) or {})
        ua = getattr(self.resolver, "user_agent", None)
        if ua and "User-Agent" not in headers:
            headers["User-Agent"] = ua
        self._headers = headers
        self._hls_options = hls_options
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._started = False
        self._end_reason: Optional[str] = None
        #: One HlsSource per broadcast followed (the current one last).
        self.broadcasts: List[HlsSource] = []
        self.hls = self._new_hls(None, seq_offset=0, time_offset_s=0.0, first=True)

    def _new_hls(self, url: Optional[str], *, seq_offset: int, time_offset_s: float, first: bool) -> HlsSource:
        return HlsSource(
            url,
            self.channel,
            self.work_dir,
            refresh=self.resolver.resolve,
            start_offset_s=self._start_s if first else 0.0,
            end_offset_s=self._end_s if first else None,
            headers=self._headers,
            seq_offset=seq_offset,
            time_offset_s=time_offset_s,
            **self._hls_options,
        )

    # Source protocol
    def __iter__(self) -> Iterator[Segment]:
        if self._started:
            raise RuntimeError("TwitchSource can only be iterated once")
        self._started = True
        if not self.wait:
            with self._lock:
                self.broadcasts.append(self.hls)
            return iter(self.hls)
        return self._follow()

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            hls = self.hls
        hls.close()

    def _follow(self) -> Iterator[Segment]:
        """Wait mode: wait for the channel to go live, follow it, repeat."""
        seq, t = 0, 0.0
        first = True
        while not self._closed.is_set():
            url = self._wait_until_live()
            if url is None:
                break
            hls = self._new_hls(url, seq_offset=seq, time_offset_s=t, first=first)
            with self._lock:
                self.hls = hls
                self.broadcasts.append(hls)
            if self._closed.is_set():
                hls.close()
            log.info("%s: live; following broadcast %d", self.channel, len(self.broadcasts))
            mark = not first
            for seg in hls:
                if mark:
                    seg = dataclasses.replace(seg, discontinuity=True)
                    mark = False
                seq, t = seg.seq + 1, seg.end
                yield seg
            first = False
            if hls.end_reason not in ("offline", "stall"):
                self._end_reason = hls.end_reason
                return
            log.info("%s: broadcast ended (%s); waiting for the next one", self.channel, hls.end_reason)
        self._end_reason = "closed"

    def _wait_until_live(self) -> Optional[str]:
        """Poll the resolver until the channel is live; ``None`` if closed first."""
        announced = False
        while not self._closed.is_set():
            try:
                return self.resolver.resolve()
            except StreamOffline:
                if not announced:
                    log.info("%s: offline; checking every %g s until it goes live", self.channel, self.offline_poll_s)
                    announced = True
            except ResolveError as exc:
                if not exc.transient:
                    raise
                log.warning("%s: could not check whether the channel is live: %s", self.channel, exc)
            except Exception as exc:  # noqa: BLE001 - any resolver failure is retried later
                log.warning("%s: could not check whether the channel is live: %s", self.channel, exc)
            if self._closed.wait(self.offline_poll_s):
                break
        return None

    # Statistics
    def _all(self) -> List[HlsSource]:
        with self._lock:
            return list(self.broadcasts) or [self.hls]

    @property
    def end_reason(self) -> Optional[str]:
        return self._end_reason or self.hls.end_reason

    @property
    def segments_yielded(self) -> int:
        return sum(h.segments_yielded for h in self._all())

    @property
    def segments_missed(self) -> int:
        return sum(h.segments_missed for h in self._all())

    @property
    def segments_ad(self) -> int:
        return sum(h.segments_ad for h in self._all())

    @property
    def reconnects(self) -> int:
        return sum(h.reconnects for h in self._all())

    @property
    def fetch_failures(self) -> int:
        return sum(h.fetch_failures for h in self._all())

    def stats(self) -> Dict[str, Any]:
        """Totals over every broadcast followed (recorded in the channel result)."""
        return {
            "end_reason": self.end_reason,
            "broadcasts": sum(1 for h in self._all() if h.segments_yielded > 0),
            "segments": self.segments_yielded,
            "ad_segments": self.segments_ad,
            "missed": self.segments_missed,
            "fetch_failures": self.fetch_failures,
            "reconnects": self.reconnects,
            "quality": getattr(self.resolver, "chosen_quality", None),
        }


@dataclass(frozen=True)
class TwitchFactory:
    """Picklable source factory for a channel, so channels can run in worker processes.

    Builds a :class:`TwitchSource` with a :class:`StreamlinkResolver`, or with
    ``resolver_factory(target, quality)`` when given (must itself be picklable
    to run in another process).
    """

    target: TwitchTarget
    channel: str
    quality: str = DEFAULT_QUALITY
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    wait: bool = False
    offline_poll_s: float = 60.0
    resolver_factory: Optional[Callable[[TwitchTarget, str], Any]] = None

    def __call__(self, work_dir: Path) -> TwitchSource:
        if self.resolver_factory is not None:
            resolver = self.resolver_factory(self.target, self.quality)
        else:
            resolver = StreamlinkResolver(self.target, self.quality)
        return TwitchSource(
            self.target,
            work_dir,
            resolver=resolver,
            channel=self.channel,
            start_s=self.start_s,
            end_s=self.end_s,
            wait=self.wait,
            offline_poll_s=self.offline_poll_s,
        )
