"""``vidflow-live``: command-line entry point of the real-time clipper.

Subcommands::

    vidflow-live replay VIDEO [VIDEO ...] [--speed 1|N|max] [--out DIR] [--transcode]
                        [--session NAME] [--set KEY=VALUE ...] [-v]

``replay`` follows one simulated channel per file (a ``ReplaySource``), prints
each clip as it lands and the latency summary at the end. ``--set`` overrides
any ``RealtimeConfig`` field, e.g. ``--set warmup_s=5 --set merge_gap_s=3``.

Exit status: 0 when every channel finished cleanly, 1 when a channel failed,
2 on bad arguments, 130 after Ctrl-C (the shutdown sequence still runs).
"""

from __future__ import annotations

import dataclasses
import logging
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import click

from .config import ConfigError, RealtimeConfig
from .pipeline import ChannelResult
from .sources.replay import ReplaySource
from .supervisor import Supervisor
from .types import Clip

__all__ = ["main", "parse_speed", "parse_overrides", "channel_names", "format_latency"]

_print_lock = threading.Lock()


def _echo(text: str = "", err: bool = False) -> None:
    with _print_lock:
        click.echo(text, err=err)


# --------------------------------------------------------------------------- parsing helpers


def parse_speed(text: str) -> Optional[float]:
    """``"max"`` -> ``None`` (unpaced); otherwise a positive number (``"4"``, ``"4x"``, ``"0.5"``)."""
    t = str(text).strip().lower()
    if t in ("max", "none", "unpaced", "inf"):
        return None
    if t.endswith("x"):
        t = t[:-1]
    try:
        v = float(t)
    except ValueError as exc:
        raise click.BadParameter(f"expected a number or 'max', got {text!r}") from exc
    if not v > 0:
        raise click.BadParameter(f"speed must be > 0, got {text!r}")
    return v


def _convert(name: str, raw: str, default: Any) -> Any:
    low = raw.strip().lower()
    if low in ("none", "null", "off") and (default is None or name == "min_percentile"):
        return None
    if isinstance(default, bool):
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise click.BadParameter(f"{name}: expected a boolean, got {raw!r}")
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError:
            raise click.BadParameter(f"{name}: expected an integer, got {raw!r}") from None
    if isinstance(default, float) or default is None:
        try:
            return float(raw)
        except ValueError:
            raise click.BadParameter(f"{name}: expected a number, got {raw!r}") from None
    return raw


def parse_overrides(items: Sequence[str], base: Optional[RealtimeConfig] = None) -> RealtimeConfig:
    """Apply ``KEY=VALUE`` strings to ``base`` (default ``RealtimeConfig()``)."""
    base = base if base is not None else RealtimeConfig()
    defaults = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    changes: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise click.BadParameter(f"expected KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in defaults:
            raise click.BadParameter(f"unknown setting {key!r}; known: {', '.join(sorted(defaults))}")
        changes[key] = _convert(key, raw, defaults[key])
    return base.replace(**changes) if changes else base


def channel_names(videos: Sequence[Path]) -> List[str]:
    """One distinct, file-name-safe channel name per video (from its stem)."""
    names: List[str] = []
    seen: Dict[str, int] = {}
    for v in videos:
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(v).stem).strip("._") or "channel"
        n = seen.get(base, 0) + 1
        seen[base] = n
        names.append(base if n == 1 else f"{base}-{n}")
    return names


def _fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.3f}"


def format_latency(summary: Mapping[str, Mapping[str, Optional[float]]]) -> List[str]:
    """Rows of a small text table of a ``LatencyTracker.summary()``."""
    lines = [f"    {'metric':<16}{'count':>6}{'p50 s':>9}{'p95 s':>9}{'max s':>9}"]
    for metric in ("detect_latency", "cut_latency", "total_latency", "ingest_lag"):
        s = summary.get(metric) or {}
        lines.append(
            f"    {metric:<16}{int(s.get('count') or 0):>6}"
            f"{_fmt(s.get('p50')):>9}{_fmt(s.get('p95')):>9}{_fmt(s.get('max')):>9}"
        )
    return lines


# --------------------------------------------------------------------------- commands


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """VidFlow real-time clipper."""


@main.command()
@click.argument("videos", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--speed", default="1", show_default=True, help="Playback rate: 1 (real time), N (N times faster) or max (unpaced).")
@click.option("--out", "out_dir", default="vidflow-live-out", show_default=True, type=click.Path(file_okay=False, path_type=Path), help="Output root: <out>/<channel>/<session>/.")
@click.option("--transcode", is_flag=True, help="Re-encode with a keyframe every segment (for files with sparse keyframes).")
@click.option("--session", default=None, help="Session name (default: local date and time).")
@click.option("--set", "overrides", multiple=True, metavar="KEY=VALUE", help="Override a RealtimeConfig setting; repeatable.")
@click.option("-v", "--verbose", count=True, help="-v: info logging, -vv: debug.")
def replay(
    videos: Tuple[Path, ...],
    speed: str,
    out_dir: Path,
    transcode: bool,
    session: Optional[str],
    overrides: Tuple[str, ...],
    verbose: int,
) -> None:
    """Replay each VIDEO as a simulated live channel and clip it as it plays."""
    level = logging.WARNING if verbose == 0 else logging.INFO if verbose == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    rate = parse_speed(speed)
    try:
        config = parse_overrides(overrides).validate()
    except ConfigError as exc:
        raise click.BadParameter(str(exc), param_hint="--set") from exc

    def on_clip(clip: Clip) -> None:
        w = clip.window
        _echo(
            f"[{w.channel}] clip {w.start:7.2f}-{w.end:7.2f} s ({w.duration:4.1f} s"
            f"{', forced' if w.forced else ''}) -> {clip.path}"
        )

    def on_score(clip: Clip, extra: Mapping[str, Any]) -> None:
        s = clip.scores
        _echo(
            f"[{clip.window.channel}] scored {clip.path.name}: visual_mean={s.get('visual_mean', 0.0):.2f} "
            f"audio_prominence={s.get('audio_prominence', 0.0):.1f} dB "
            f"rank_percentile={extra.get('rank_percentile', 0.0):.0f}"
        )

    sup = Supervisor(config, out_dir, session=session, on_clip=on_clip, on_score=on_score)
    names = channel_names(videos)
    for video, name in zip(videos, names):

        def factory(work_dir: Path, video: Path = video, name: str = name) -> ReplaySource:
            return ReplaySource(
                video, name, work_dir, speed=rate, segment_time=config.segment_time_s, transcode=transcode
            )

        sup.add_channel(factory, channel=name)

    pace = "unpaced" if rate is None else f"{rate:g}x"
    _echo(f"session {sup.session}: {len(names)} channel(s) at {pace} -> {out_dir}")
    results = sup.run()
    _report(results)

    if sup.interrupted:
        sys.exit(130)
    if any(not r.ok for r in results.values()):
        sys.exit(1)


def _report(results: Mapping[str, ChannelResult]) -> None:
    _echo("")
    for name, r in results.items():
        status = "ok" if r.ok else f"FAILED: {r.error}"
        _echo(
            f"[{name}] {status}{' (stopped)' if r.stopped else ''}: {r.segments} segments, "
            f"{r.windows} windows, {r.clips} clips, {r.scored} scored, {r.dropped} dropped, {r.errors} errors"
        )
        _echo(f"    log: {r.session_dir / 'events.jsonl'}")
        for line in format_latency(r.latency):
            _echo(line)


if __name__ == "__main__":  # pragma: no cover
    main()
