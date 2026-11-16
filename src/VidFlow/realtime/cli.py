"""``vidflow-live``: command-line entry point of the real-time clipper.

Subcommands::

    vidflow-live replay VIDEO [VIDEO ...] [--speed 1|N|max] [--out DIR] [--transcode]
                        [--session NAME] [--set KEY=VALUE ...] [--procs N] [-v]
    vidflow-live eval VIDEO [--truth FILE] [--iou 0.5] [--out DIR] [--set KEY=VALUE ...] [--json FILE|-]
    vidflow-live retune EVENTS.jsonl|SESSION_DIR [--set KEY=VALUE ...] [--json FILE|-]
    vidflow-live bench VIDEO --channels N --seconds M [--out DIR] [--procs N] [--stagger S] [--seed K]
                       [--set KEY=VALUE ...] [--json FILE|-]

``eval``, ``retune`` and ``bench`` are documented in ``eval.py``.

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
    from .session import LATENCY_METRICS

    lines = [f"    {'metric':<16}{'count':>6}{'p50 s':>9}{'p95 s':>9}{'max s':>9}"]
    for metric in LATENCY_METRICS:
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
@click.option("--procs", "processes", default=0, show_default=True, type=click.IntRange(min=0),
              help="Worker processes; 0: one per channels_per_process channels (1 for up to 20).")
@click.option("-v", "--verbose", count=True, help="-v: info logging, -vv: debug.")
def replay(
    videos: Tuple[Path, ...],
    speed: str,
    out_dir: Path,
    transcode: bool,
    session: Optional[str],
    overrides: Tuple[str, ...],
    processes: int,
    verbose: int,
) -> None:
    """Replay each VIDEO as a simulated live channel and clip it as it plays."""
    from .shard import ReplayFactory, ShardedSupervisor, make_supervisor

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

    names = channel_names(videos)
    sup = make_supervisor(
        config, out_dir, channels=len(names), processes=processes, session=session,
        on_clip=on_clip, on_score=on_score,
    )
    for video, name in zip(videos, names):
        # A picklable factory, so channels can be built in worker processes.
        factory = ReplayFactory(video, name, speed=rate, segment_time=config.segment_time_s, transcode=transcode)
        sup.add_channel(factory, channel=name)

    pace = "unpaced" if rate is None else f"{rate:g}x"
    procs = f" in {sup.processes} processes" if isinstance(sup, ShardedSupervisor) else ""
    _echo(f"session {sup.session}: {len(names)} channel(s) at {pace}{procs} -> {out_dir}")
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


# --------------------------------------------------------------------------- eval / retune / bench (T9)


def _setup_logging(verbose: int) -> None:
    level = logging.WARNING if verbose == 0 else logging.INFO if verbose == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)


def _config_from(overrides: Sequence[str], base: Optional[RealtimeConfig] = None) -> RealtimeConfig:
    try:
        return parse_overrides(overrides, base).validate()
    except ConfigError as exc:
        raise click.BadParameter(str(exc), param_hint="--set") from exc


def _emit(lines: Sequence[str], data: Mapping[str, Any], json_out: Optional[str]) -> None:
    """Print the text report, or the JSON to stdout with ``--json -``; ``--json FILE`` writes it too."""
    import json

    from .session import to_jsonable

    text = json.dumps(to_jsonable(data), indent=2)
    if json_out == "-":
        _echo(text)
        return
    for line in lines:
        _echo(line)
    if json_out:
        Path(json_out).write_text(text + "\n", encoding="utf-8")
        _echo(f"json: {json_out}")


_json_option = click.option(
    "--json", "json_out", default=None, metavar="FILE",
    help="Also write the full report as JSON to FILE ('-': print only the JSON).",
)
_set_option = click.option(
    "--set", "overrides", multiple=True, metavar="KEY=VALUE", help="Override a RealtimeConfig setting; repeatable."
)
_verbose_option = click.option("-v", "--verbose", count=True, help="-v: info logging, -vv: debug.")


@main.command("eval")
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--truth", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Reference windows (JSON list of [start, end], or 'start end' lines) instead of the offline silencedetect reference.")
@click.option("--iou", "min_iou", default=0.5, show_default=True, type=click.FloatRange(0.0, 1.0, min_open=True),
              help="Overlap (intersection over union) needed for a match.")
@click.option("--out", "out_dir", default=None, type=click.Path(file_okay=False, path_type=Path),
              help="Keep the run's session log under <out>/eval/<session>/ (for retune). Default: a temporary directory.")
@click.option("--session", default=None, help="Session name (default: local date and time).")
@click.option("--transcode", is_flag=True, help="Re-encode with a keyframe every segment before replaying.")
@_set_option
@_json_option
@_verbose_option
def eval_cmd(
    video: Path,
    truth: Optional[Path],
    min_iou: float,
    out_dir: Optional[Path],
    session: Optional[str],
    transcode: bool,
    overrides: Tuple[str, ...],
    json_out: Optional[str],
    verbose: int,
) -> None:
    """Score the online detector on VIDEO against reference windows (no clips are cut)."""
    from .eval import evaluate, format_eval

    _setup_logging(verbose)
    config = _config_from(overrides)
    report = evaluate(
        video, config, truth=truth, out_dir=out_dir, session=session, min_iou=min_iou, transcode=transcode
    )
    lines = format_eval(report)
    if report.session_dir is not None:
        lines.append(f"log: {report.session_dir / 'events.jsonl'}")
    _emit(lines, report.to_dict(), json_out)


@main.command()
@click.argument("events", type=click.Path(exists=True, path_type=Path))
@_set_option
@_json_option
@_verbose_option
def retune(events: Path, overrides: Tuple[str, ...], json_out: Optional[str], verbose: int) -> None:
    """Re-run the segmenter over a session log (EVENTS.jsonl or its directory) with changed settings.

    Settings start from the ones the session logged; --set changes them. No video is read.
    """
    from .eval import format_retune
    from .eval import retune as do_retune
    from .session import read_config

    _setup_logging(verbose)
    try:
        base = read_config(events) or RealtimeConfig()
        config = parse_overrides(overrides, base)
        result = do_retune(events, config)
    except (ConfigError, ValueError) as exc:
        raise click.BadParameter(str(exc), param_hint="--set") from exc
    except OSError as exc:
        raise click.BadParameter(str(exc), param_hint="EVENTS") from exc
    _emit(format_retune(result), result.to_dict(), json_out)


@main.command()
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--channels", "-n", default=4, show_default=True, type=click.IntRange(min=1), help="Simulated channels.")
@click.option("--seconds", "-m", default=60.0, show_default=True, type=click.FloatRange(min=0, min_open=True),
              help="How long each channel plays (wall seconds at --speed 1; seconds x speed of stream).")
@click.option("--speed", default="1", show_default=True, help="Playback rate of every channel (1 = real time).")
@click.option("--out", "out_dir", default=None, type=click.Path(file_okay=False, path_type=Path),
              help="Keep clips and logs under <out>/<channel>/bench/. Default: a temporary directory.")
@click.option("--transcode", is_flag=True, help="Re-encode with a keyframe every segment before replaying.")
@click.option("--procs", "processes", default=0, show_default=True, type=click.IntRange(min=0),
              help="Worker processes; 0: one per channels_per_process channels (--set channels_per_process=N).")
@click.option("--stagger", "stagger_s", default=None, type=click.FloatRange(min=0),
              help="Spread channel start delays over this many seconds; 0: all start together "
                   "(synchronized worst case). Default: min(20, seconds / 3).")
@click.option("--seed", default=0, show_default=True, type=int, help="Seed of the start-delay order.")
@_set_option
@_json_option
@_verbose_option
def bench(
    video: Path,
    channels: int,
    seconds: float,
    speed: str,
    out_dir: Optional[Path],
    transcode: bool,
    processes: int,
    stagger_s: Optional[float],
    seed: int,
    overrides: Tuple[str, ...],
    json_out: Optional[str],
    verbose: int,
) -> None:
    """Follow N simulated channels of VIDEO at 1x for M seconds; report CPU, memory, lag and latency."""
    from .eval import format_bench, run_bench

    _setup_logging(verbose)
    rate = parse_speed(speed)
    if rate is None:
        raise click.BadParameter("bench needs a paced speed", param_hint="--speed")
    config = _config_from(overrides)
    _echo(f"following {channels} channel(s) for {seconds:g} s ...")
    report = run_bench(
        video, channels, seconds, config, out_dir=out_dir, speed=rate, transcode=transcode,
        processes=processes, stagger_s=stagger_s, seed=seed,
    )
    _emit(format_bench(report), report.to_dict(), json_out)
    if report.failed:
        sys.exit(1)


# Twitch commands (watch, vod) live in their own module.
from .cli_twitch import register as _register_twitch  # noqa: E402

_register_twitch(main)


if __name__ == "__main__":  # pragma: no cover
    main()
