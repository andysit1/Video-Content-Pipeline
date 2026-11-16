"""All tunables of the real-time clipper (``docs/realtime/DESIGN.md`` section 8)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional

__all__ = ["RealtimeConfig", "ConfigError", "VISUAL_MODES"]

VISUAL_MODES = ("keyframes", "sampled")


class ConfigError(ValueError):
    """A ``RealtimeConfig`` failed validation."""


@dataclass(frozen=True)
class RealtimeConfig:
    """Settings for one channel pipeline. Immutable; use :meth:`replace` to derive."""

    # Source / buffer
    segment_time_s: float = 2.0  # target segment length for simulated feeds
    retention_s: float = 120.0  # rolling buffer length

    # Loudness
    hop_s: float = 0.1  # loudness resolution
    stats_window_s: float = 600.0  # history used for the adaptive threshold

    # Segmenter
    threshold_percentile: float = 90.0  # percentile of the background loudness ("loud end of normal")
    rise_db: float = 5.0  # a sample is loud at background percentile + rise_db (or floor_db)
    window_weight: float = 0.25  # weight of detected-window samples in the background stats (0..1)
    floor_db: float = -35.0  # threshold never goes below this
    warmup_s: float = 30.0  # opening span judged with hindsight once complete (not skipped)
    merge_gap_s: float = 4.0  # quiet needed to close a window (from the batch code)
    min_len_s: float = 1.5  # shortest window kept (from the batch code)
    max_len_s: float = 45.0  # longest window before a forced close

    # Clip padding
    pre_pad_s: float = 5.0
    post_pad_s: float = 3.0

    # Visual scoring
    visual_mode: str = "keyframes"  # or "sampled", with sample_fps
    sample_fps: float = 4.0  # frames scored per second in "sampled" mode

    # Workers
    workers: int = 4  # worker pool size
    queue_size: int = 64  # floor of the worker pool's job queue bound (design 6.9: "bounded")
    queue_per_channel: int = 4  # queue bound = max(queue_size, queue_per_channel x channels in the process)
    clip_timeout_s: float = 30.0  # FFmpeg timeout for one cut (design 6.5)
    channels_per_process: int = 20  # above this many channels, shard them over worker processes

    # Ranking
    min_percentile: Optional[float] = None  # optional cull; None = off

    # ------------------------------------------------------------------ helpers

    def min_retention_s(self, segment_duration: Optional[float] = None) -> float:
        """Smallest ``retention_s`` allowed by the rule in design section 6.2.

        ``max_len_s + pre_pad_s + post_pad_s + merge_gap_s + 2 x segment duration``.
        ``segment_duration`` defaults to ``segment_time_s``; pass the real
        (longest observed) segment duration when it is known.
        """
        seg = self.segment_time_s if segment_duration is None else float(segment_duration)
        return self.max_len_s + self.pre_pad_s + self.post_pad_s + self.merge_gap_s + 2.0 * seg

    def validate(self, segment_duration: Optional[float] = None) -> "RealtimeConfig":
        """Check every setting; raise :class:`ConfigError` on the first problem.

        Returns ``self`` so it can be chained.
        """
        positive = (
            "segment_time_s",
            "retention_s",
            "hop_s",
            "stats_window_s",
            "merge_gap_s",
            "max_len_s",
            "sample_fps",
            "clip_timeout_s",
        )
        for name in positive:
            if not getattr(self, name) > 0:
                raise ConfigError(f"{name} must be > 0, got {getattr(self, name)!r}")
        non_negative = ("warmup_s", "min_len_s", "pre_pad_s", "post_pad_s")
        for name in non_negative:
            if not getattr(self, name) >= 0:
                raise ConfigError(f"{name} must be >= 0, got {getattr(self, name)!r}")
        if not 0.0 <= self.threshold_percentile <= 100.0:
            raise ConfigError(
                f"threshold_percentile must be in [0, 100], got {self.threshold_percentile!r}"
            )
        if self.min_len_s > self.max_len_s:
            raise ConfigError(
                f"min_len_s ({self.min_len_s}) must not exceed max_len_s ({self.max_len_s})"
            )
        if self.stats_window_s < self.hop_s:
            raise ConfigError("stats_window_s must be at least one hop")
        if not -100.0 <= self.rise_db <= 100.0:
            raise ConfigError(f"rise_db must be in [-100, 100], got {self.rise_db!r}")
        if not 0.0 <= self.window_weight <= 1.0:
            raise ConfigError(f"window_weight must be in [0, 1], got {self.window_weight!r}")
        if self.visual_mode not in VISUAL_MODES:
            raise ConfigError(f"visual_mode must be one of {VISUAL_MODES}, got {self.visual_mode!r}")
        if int(self.workers) != self.workers or self.workers < 1:
            raise ConfigError(f"workers must be an integer >= 1, got {self.workers!r}")
        if int(self.queue_size) != self.queue_size or self.queue_size < 1:
            raise ConfigError(f"queue_size must be an integer >= 1, got {self.queue_size!r}")
        if int(self.queue_per_channel) != self.queue_per_channel or self.queue_per_channel < 0:
            raise ConfigError(f"queue_per_channel must be an integer >= 0, got {self.queue_per_channel!r}")
        if int(self.channels_per_process) != self.channels_per_process or self.channels_per_process < 1:
            raise ConfigError(
                f"channels_per_process must be an integer >= 1, got {self.channels_per_process!r}"
            )
        if self.min_percentile is not None and not 0.0 <= self.min_percentile <= 100.0:
            raise ConfigError(f"min_percentile must be None or in [0, 100], got {self.min_percentile!r}")
        if segment_duration is not None and not segment_duration > 0:
            raise ConfigError(f"segment_duration must be > 0, got {segment_duration!r}")

        needed = self.min_retention_s(segment_duration)
        if self.retention_s < needed:
            raise ConfigError(
                f"retention_s ({self.retention_s}) is too small: it must be at least "
                f"max_len_s + pre_pad_s + post_pad_s + merge_gap_s + 2 x segment duration "
                f"= {needed}"
            )
        # Windows found in the warm-up are returned when it ends; their padded
        # start must still be buffered then.
        seg = self.segment_time_s if segment_duration is None else float(segment_duration)
        warm_needed = self.warmup_s + self.pre_pad_s + 2.0 * seg
        if self.retention_s < warm_needed:
            raise ConfigError(
                f"retention_s ({self.retention_s}) is too small for warmup_s ({self.warmup_s}): it must be "
                f"at least warmup_s + pre_pad_s + 2 x segment duration = {warm_needed}"
            )
        return self

    def replace(self, **changes: Any) -> "RealtimeConfig":
        """Return a copy with ``changes`` applied (unknown names raise ``TypeError``)."""
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        """Plain-JSON snapshot, e.g. for the ``session_start`` event."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RealtimeConfig":
        """Inverse of :meth:`to_dict`; ignores unknown keys."""
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})
