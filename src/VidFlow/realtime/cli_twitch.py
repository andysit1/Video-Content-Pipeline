"""``vidflow-live watch`` and ``vidflow-live vod``: follow Twitch channels and VODs.

::

    vidflow-live watch CHANNEL_OR_URL [...] [--wait] [--quality best] [--duration 2h] [--procs N]
                       [--out DIR] [--session NAME] [--set KEY=VALUE ...] [-v]
    vidflow-live vod URL [--start 1:00:00] [--end 1:05:00] [--quality best] [--out DIR] ...

``watch`` follows every live channel at once (one ``TwitchSource`` per channel)
until each stream ends, ``--duration`` passes or Ctrl-C. With ``--wait`` an
offline channel is checked every ``--offline-poll`` seconds until it goes live,
and after each broadcast it waits for the next one, so the command runs until
``--duration`` or Ctrl-C. Above ``channels_per_process`` (20) channels, or with
``--procs N``, channels are split over worker processes. It also accepts VOD
URLs (processed from ``?t=`` to the end). ``vod`` processes one VOD, or a
slice of it, as fast as it downloads.

A channel whose stream ends while its segments cannot be downloaded is
reported as FAILED and the command exits with status 1.

Output is the same as ``replay``: ``<out>/<channel>/<session>/{events.jsonl,
clips/*.mp4, clips/*.json}``. The channel folder is the login, or
``vod-<id>``.

Needs the optional ``streamlink`` package (``pip install 'VidFlow[twitch]'``)
to resolve playlists; no Twitch API key is used. Registered on the main
``vidflow-live`` group by :func:`register`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import click

from .config import ConfigError
from .sources.twitch import (
    DEFAULT_QUALITY,
    ResolveError,
    TwitchFactory,
    TwitchTarget,
    parse_target,
    parse_time,
    require_streamlink,
)
from .types import Clip

__all__ = ["register", "watch", "vod", "RESOLVER_FACTORY"]

# Twitch VOD segments are about 10 s long (live ones about 2 s).
_VOD_SEGMENT_S = 10.0

#: Builds the resolver for a target. Tests replace this to point at ``hls_sim``;
#: with more than one process it must be picklable.
RESOLVER_FACTORY: Optional[Callable[[TwitchTarget, str], Any]] = None


def _time_option(value: Optional[str], name: str) -> Optional[float]:
    if value is None:
        return None
    try:
        return parse_time(value)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint=name) from exc


def _unique_names(targets: Sequence[TwitchTarget]) -> List[str]:
    names: List[str] = []
    seen: Dict[str, int] = {}
    for t in targets:
        base = t.default_channel
        n = seen.get(base, 0) + 1
        seen[base] = n
        names.append(base if n == 1 else f"{base}-{n}")
    return names


def _run(
    targets: Sequence[TwitchTarget],
    *,
    quality: str,
    out_dir: Path,
    session: Optional[str],
    overrides: Sequence[str],
    verbose: int,
    duration: Optional[float],
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    wait: bool = False,
    offline_poll_s: float = 60.0,
    processes: int = 0,
) -> None:
    from . import cli as base  # noqa: PLC0415 - the main group module; imported late to avoid a cycle
    from .shard import ShardedSupervisor, make_supervisor  # noqa: PLC0415

    base._setup_logging(verbose)
    config = base._config_from(overrides)
    if any(not t.is_live for t in targets):
        try:
            config.validate(segment_duration=_VOD_SEGMENT_S)
        except ConfigError as exc:
            raise click.BadParameter(
                f"{exc} (VOD segments are about {_VOD_SEGMENT_S:g} s)", param_hint="--set"
            ) from exc

    factory = RESOLVER_FACTORY
    if factory is None:
        try:
            require_streamlink()
        except ResolveError as exc:
            raise click.ClickException(str(exc)) from exc

    def on_clip(clip: Clip) -> None:
        w = clip.window
        base._echo(
            f"[{w.channel}] clip {w.start:8.2f}-{w.end:8.2f} s ({w.duration:4.1f} s"
            f"{', forced' if w.forced else ''}) -> {clip.path}"
        )

    def on_score(clip: Clip, extra: Mapping[str, Any]) -> None:
        s = clip.scores
        base._echo(
            f"[{clip.window.channel}] scored {clip.path.name}: visual_mean={s.get('visual_mean', 0.0):.2f} "
            f"audio_prominence={s.get('audio_prominence', 0.0):.1f} dB "
            f"rank_percentile={extra.get('rank_percentile', 0.0):.0f}"
        )

    sup = make_supervisor(
        config, out_dir, channels=len(targets), processes=processes or None,
        session=session, on_clip=on_clip, on_score=on_score,
    )
    for target, name in zip(targets, _unique_names(targets)):
        sup.add_channel(
            TwitchFactory(
                target,
                name,
                quality=quality,
                start_s=start_s,
                end_s=end_s,
                wait=wait and target.is_live,
                offline_poll_s=offline_poll_s,
                resolver_factory=factory,
            ),
            channel=name,
        )

    live = sum(t.is_live for t in targets)
    kinds = ", ".join(x for x in (f"{live} live" if live else "", f"{len(targets) - live} VOD" if live < len(targets) else "") if x)
    limit = f", for at most {duration:g} s" if duration else ""
    waiting = ", waiting for offline channels" if wait and live else ""
    procs = f" in {sup.processes} processes" if isinstance(sup, ShardedSupervisor) else ""
    base._echo(f"session {sup.session}: {kinds} at quality {quality}{limit}{waiting}{procs} -> {out_dir}")
    results = sup.run(timeout=duration)
    base._report(results)
    for name, r in results.items():
        st = r.source
        if not st:
            continue
        reason = st.get("end_reason") or "unknown"
        hint = " (not live; pass --wait to wait for it)" if reason == "offline" and st.get("segments") == 0 else ""
        broadcasts = f"{st.get('broadcasts', 0)} broadcast(s), " if wait else ""
        base._echo(
            f"[{name}] source ended: {reason}{hint}; {broadcasts}{st.get('segments', 0)} segments, "
            f"{st.get('ad_segments', 0)} ad segments skipped, {st.get('missed', 0)} missed "
            f"({st.get('fetch_failures', 0)} failed downloads), {st.get('reconnects', 0)} reconnects"
        )

    if sup.interrupted:
        sys.exit(130)
    if any(not r.ok for r in results.values()):
        sys.exit(1)


def _targets(items: Sequence[str]) -> List[TwitchTarget]:
    out = []
    for item in items:
        try:
            out.append(parse_target(item))
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="CHANNEL_OR_URL") from exc
    return out


_quality_option = click.option(
    "--quality", "-q", default=DEFAULT_QUALITY, show_default=True,
    help="Rendition preference list, first available wins, e.g. '720p60,720p,best'. "
         "'best' is the broadcaster's source (about 6-8 Mbps per channel); lower it when following many channels.",
)
_out_option = click.option(
    "--out", "out_dir", default="vidflow-live-out", show_default=True, type=click.Path(file_okay=False, path_type=Path),
    help="Output root: <out>/<channel>/<session>/.",
)
_session_option = click.option("--session", default=None, help="Session name (default: local date and time).")
_set_option = click.option(
    "--set", "overrides", multiple=True, metavar="KEY=VALUE", help="Override a RealtimeConfig setting; repeatable."
)
_verbose_option = click.option("-v", "--verbose", count=True, help="-v: info logging, -vv: debug.")
_procs_option = click.option(
    "--procs", "processes", default=0, show_default=True, type=click.IntRange(min=0),
    help="Worker processes; 0 = one per channels_per_process (20) channels, 1 = a single process.",
)


@click.command()
@click.argument("channels", nargs=-1, required=True, metavar="CHANNEL_OR_URL...")
@_quality_option
@click.option("--duration", default=None, metavar="TIME", help="Stop after this long (e.g. 600, 10m, 1:30:00). Default: until the streams end.")
@click.option("--wait", is_flag=True, help="Wait for offline channels to go live, and for the next broadcast after one ends. Runs until --duration or Ctrl-C.")
@click.option("--offline-poll", default="60", show_default=True, metavar="TIME", help="With --wait: how often to check whether an offline channel is live.")
@_procs_option
@_out_option
@_session_option
@_set_option
@_verbose_option
def watch(
    channels: Tuple[str, ...],
    quality: str,
    duration: Optional[str],
    wait: bool,
    offline_poll: str,
    processes: int,
    out_dir: Path,
    session: Optional[str],
    overrides: Tuple[str, ...],
    verbose: int,
) -> None:
    """Follow Twitch channels live (names or twitch.tv URLs) and clip them as they stream.

    Without --wait, a channel that is offline when following starts ends at
    once and each channel ends when its stream ends. Ctrl-C stops all of
    them cleanly.
    """
    targets = _targets(channels)
    poll = _time_option(offline_poll, "--offline-poll")
    if poll is None or poll <= 0:
        raise click.BadParameter("must be > 0", param_hint="--offline-poll")
    _run(
        targets,
        quality=quality,
        out_dir=out_dir,
        session=session,
        overrides=overrides,
        verbose=verbose,
        duration=_time_option(duration, "--duration"),
        wait=wait,
        offline_poll_s=poll,
        processes=processes,
    )


@click.command()
@click.argument("url")
@click.option("--start", default=None, metavar="TIME", help="Start this far into the VOD (default: the URL's ?t= or 0).")
@click.option("--end", default=None, metavar="TIME", help="Stop at this point of the VOD (default: its end).")
@_quality_option
@_out_option
@_session_option
@_set_option
@_verbose_option
def vod(
    url: str,
    start: Optional[str],
    end: Optional[str],
    quality: str,
    out_dir: Path,
    session: Optional[str],
    overrides: Tuple[str, ...],
    verbose: int,
) -> None:
    """Clip a Twitch VOD (twitch.tv/videos/<id>), or a slice of it, as fast as it downloads."""
    (target,) = _targets([url])
    if target.is_live:
        raise click.BadParameter("not a VOD URL (use 'watch' for live channels)", param_hint="URL")
    start_s = _time_option(start, "--start")
    end_s = _time_option(end, "--end")
    if end_s is not None and end_s <= (start_s if start_s is not None else target.start_s):
        raise click.BadParameter("--end must be after --start", param_hint="--end")
    _run(
        [target],
        quality=quality,
        out_dir=out_dir,
        session=session,
        overrides=overrides,
        verbose=verbose,
        duration=None,
        start_s=start_s,
        end_s=end_s,
    )


def register(group: click.Group) -> None:
    """Add ``watch`` and ``vod`` to the ``vidflow-live`` command group."""
    group.add_command(watch)
    group.add_command(vod)
