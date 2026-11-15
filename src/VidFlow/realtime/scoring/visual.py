"""Visual scoring of a candidate window (``docs/realtime/DESIGN.md`` section 6.6).

Written fresh; it does not use ``VidFlow.aggregate.opencv_component`` and fixes
its known problems (section 10):

- Crops index rows with y and columns with x, so the default profile really
  scores the centre of a non-square frame.
- Grayscale conversion happens once, at decode time, with no missing helper.
- Per-clip scores are statistics over scored frames (mean, p90, max), so a long
  clip does not outrank a short one by length.

Frame score
    For each region of the profile: ``binary_weight x`` the percentage of
    pixels above 170 after a binary threshold, plus ``canny_weight x`` the
    percentage of Canny(200, 300) edge pixels. The frame score is the mean
    over regions, so it lies in ``[0, 100 x (binary_weight + canny_weight)]``
    (0 to 250 with the default weights) whatever the number of regions.

Timing
    Frame times are media times: ``segment.start`` plus the frame's offset
    from the first decoded frame of that segment. Packet timestamps are only
    used as differences inside one segment (design section 2: the absolute
    container timestamps carry a muxer offset).
"""

from __future__ import annotations

import dataclasses
import logging
import math
from contextlib import closing
from dataclasses import dataclass, field
from typing import Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import av
import cv2
import numpy as np

from ..config import VISUAL_MODES, RealtimeConfig
from ..types import Clip, Segment

__all__ = [
    "Region",
    "GameProfile",
    "DEFAULT_PROFILE",
    "BINARY_THRESHOLD",
    "CANNY_LOW",
    "CANNY_HIGH",
    "region_bounds",
    "crop_regions",
    "frame_score",
    "VisualScorer",
    "SCORE_KEYS",
]

log = logging.getLogger(__name__)

Region = Tuple[float, float, float, float]  # (x0, y0, x1, y1) as fractions of the frame

BINARY_THRESHOLD = 170
CANNY_LOW = 200
CANNY_HIGH = 300

SCORE_KEYS = ("visual_mean", "visual_p90", "visual_max", "frames")

# Media-time tolerance when deciding whether a frame lies inside the core.
# Far below one frame period at any real frame rate.
_TIME_EPS = 1e-3


def _centre_box(px_w: float, px_h: float, ref_w: float = 1920.0, ref_h: float = 1080.0) -> Region:
    hw = px_w / ref_w / 2.0
    hh = px_h / ref_h / 2.0
    return (0.5 - hw, 0.5 - hh, 0.5 + hw, 0.5 + hh)


@dataclass(frozen=True)
class GameProfile:
    """Where to look in a frame and how to weigh the two measures.

    ``regions`` are ``(x0, y0, x1, y1)`` fractions of the frame width (x) and
    height (y), so a profile is independent of resolution. ``weights`` holds
    the weight of each measure: ``"binary"`` (threshold white percentage) and
    ``"canny"`` (edge white percentage).
    """

    name: str
    regions: Tuple[Region, ...]
    weights: Mapping[str, float] = field(default_factory=lambda: {"binary": 2.0, "canny": 0.5})

    def __post_init__(self) -> None:
        regions = tuple(tuple(float(v) for v in r) for r in self.regions)
        if not regions:
            raise ValueError("a GameProfile needs at least one region")
        for r in regions:
            if len(r) != 4:
                raise ValueError(f"region must be (x0, y0, x1, y1), got {r!r}")
            x0, y0, x1, y1 = r
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ValueError(f"region {r!r} must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
        unknown = set(self.weights) - {"binary", "canny"}
        if unknown:
            raise ValueError(f"unknown weight names {sorted(unknown)}; expected 'binary' and 'canny'")
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "weights", {"binary": 2.0, "canny": 0.5, **dict(self.weights)})

    @property
    def binary_weight(self) -> float:
        return float(self.weights["binary"])

    @property
    def canny_weight(self) -> float:
        return float(self.weights["canny"])


# The centre crosshair box: 150 x 150 px on a 1920 x 1080 frame.
DEFAULT_PROFILE = GameProfile(name="crosshair", regions=(_centre_box(150.0, 150.0),))


def region_bounds(shape: Sequence[int], region: Region) -> Tuple[int, int, int, int]:
    """Pixel bounds ``(y0, y1, x0, x1)`` of ``region`` on a frame of ``shape`` ``(h, w, ...)``.

    Rows come from the y fractions and columns from the x fractions. The box
    is at least one pixel in each direction and never leaves the frame.
    """
    h, w = int(shape[0]), int(shape[1])
    fx0, fy0, fx1, fy1 = region
    x0 = min(max(int(round(fx0 * w)), 0), w - 1)
    y0 = min(max(int(round(fy0 * h)), 0), h - 1)
    x1 = min(max(int(round(fx1 * w)), x0 + 1), w)
    y1 = min(max(int(round(fy1 * h)), y0 + 1), h)
    return y0, y1, x0, x1


def crop_regions(gray: np.ndarray, profile: GameProfile = DEFAULT_PROFILE) -> List[np.ndarray]:
    """Return one (view) crop of ``gray`` per region of ``profile``."""
    if gray.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale frame, got shape {gray.shape}")
    crops = []
    for region in profile.regions:
        y0, y1, x0, x1 = region_bounds(gray.shape, region)
        crops.append(gray[y0:y1, x0:x1])
    return crops


def _white_percentage(binary: np.ndarray) -> float:
    return float(np.count_nonzero(binary)) * 100.0 / float(binary.size)


def frame_score(gray: np.ndarray, profile: GameProfile = DEFAULT_PROFILE) -> float:
    """Score one 2-D uint8 grayscale frame (higher = more bright/edgy content in the regions)."""
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    total = 0.0
    crops = crop_regions(gray, profile)
    for crop in crops:
        crop = np.ascontiguousarray(crop)
        _, thresh = cv2.threshold(crop, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(crop, CANNY_LOW, CANNY_HIGH)
        total += profile.binary_weight * _white_percentage(thresh) + profile.canny_weight * _white_percentage(edges)
    return total / len(crops)


class _LeaseLike(Protocol):
    segments: Sequence[Segment]


LeaseOrSegments = Union[_LeaseLike, Sequence[Segment]]

def _gray(frame: "av.VideoFrame") -> np.ndarray:
    """2-D uint8 full-range grayscale of a decoded frame.

    Converted by swscale, which expands limited-range (16-235) luma to 0-255,
    so the fixed binary threshold means the same as on an RGB-to-gray frame.
    About 0.75 ms at 1080p, small next to decoding.
    """
    return frame.to_ndarray(format="gray")


class VisualScorer:
    """Scores the frames of a lease that fall inside a core window.

    ``mode="keyframes"`` decodes only keyframes (the decoder skips the rest)
    and scores each. ``mode="sampled"`` decodes every frame and scores
    ``sample_fps`` frames per second, on a time grid anchored at the core start.

    Thread-safe: a scorer holds no per-call state, so one instance can be
    shared by all workers.
    """

    def __init__(
        self,
        profile: GameProfile = DEFAULT_PROFILE,
        mode: str = "keyframes",
        sample_fps: float = 4.0,
    ) -> None:
        if mode not in VISUAL_MODES:
            raise ValueError(f"mode must be one of {VISUAL_MODES}, got {mode!r}")
        if not sample_fps > 0:
            raise ValueError(f"sample_fps must be > 0, got {sample_fps!r}")
        self.profile = profile
        self.mode = mode
        self.sample_fps = float(sample_fps)

    @classmethod
    def from_config(cls, config: RealtimeConfig, profile: GameProfile = DEFAULT_PROFILE) -> "VisualScorer":
        return cls(profile=profile, mode=config.visual_mode, sample_fps=config.sample_fps)

    # ------------------------------------------------------------------ public

    def score(self, lease: LeaseOrSegments, core_start: float, core_end: float) -> dict:
        """Score frames in ``[core_start, core_end)`` (media time) from the lease's segments.

        Returns ``visual_mean``, ``visual_p90``, ``visual_max`` (floats) and
        ``frames`` (the number of frames scored, as a float). With no frame
        scored all four are 0.0. Undecodable segments are skipped with a
        warning; this method does not raise for bad media.
        """
        scores = self.frame_scores(lease, core_start, core_end)
        return self.summarise([s for _, s in scores])

    def score_clip(self, clip: Clip, lease: LeaseOrSegments) -> Clip:
        """Score ``clip.window`` from ``lease`` and return a copy of ``clip`` with the scores merged in."""
        result = self.score(lease, clip.window.start, clip.window.end)
        return dataclasses.replace(clip, scores={**dict(clip.scores), **result})

    @staticmethod
    def summarise(values: Sequence[float]) -> dict:
        if len(values) == 0:
            return {"visual_mean": 0.0, "visual_p90": 0.0, "visual_max": 0.0, "frames": 0.0}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "visual_mean": float(arr.mean()),
            "visual_p90": float(np.percentile(arr, 90)),
            "visual_max": float(arr.max()),
            "frames": float(arr.size),
        }

    def frame_scores(self, lease: LeaseOrSegments, core_start: float, core_end: float) -> List[Tuple[float, float]]:
        """``(media_time, score)`` for every frame scored, in order."""
        out: List[Tuple[float, float]] = []
        for t, gray in self.iter_frames(lease, core_start, core_end):
            out.append((t, frame_score(gray, self.profile)))
        return out

    def iter_frames(
        self, lease: LeaseOrSegments, core_start: float, core_end: float
    ) -> Iterator[Tuple[float, np.ndarray]]:
        """Yield ``(media_time, gray_frame)`` for each frame that would be scored."""
        segments: Sequence[Segment] = getattr(lease, "segments", lease)
        core_start = float(core_start)
        core_end = float(core_end)
        if not core_end > core_start:
            return
        period = 1.0 / self.sample_fps
        next_sample = core_start  # sampled-mode grid, carried across segments
        for seg in sorted(segments, key=lambda s: s.start):
            if seg.end <= core_start + _TIME_EPS or seg.start >= core_end - _TIME_EPS:
                continue
            try:
                with closing(self._decode_segment(seg, core_end)) as frames:
                    for t, frame in frames:
                        if t < core_start - _TIME_EPS:
                            continue
                        if t >= core_end - _TIME_EPS:
                            break
                        if self.mode == "sampled":
                            if t < next_sample - _TIME_EPS:
                                continue
                            # Advance the grid past this frame (skips grid points with no frame).
                            steps = math.floor((t - next_sample + _TIME_EPS) / period) + 1
                            next_sample += steps * period
                        yield t, _gray(frame)
            except (av.error.FFmpegError, ValueError, OSError) as exc:
                log.warning("visual scoring: could not decode segment %s (%s): %s", seg.seq, seg.path, exc)

    # ---------------------------------------------------------------- internal

    def _decode_segment(self, seg: Segment, core_end: float) -> Iterator[Tuple[float, "av.VideoFrame"]]:
        with av.open(str(seg.path)) as container:
            if not container.streams.video:
                log.warning("visual scoring: segment %s (%s) has no video stream", seg.seq, seg.path)
                return
            stream = container.streams.video[0]
            if self.mode == "keyframes":
                stream.codec_context.skip_frame = "NONKEY"
            else:
                # Every frame is decoded: frame threading roughly halves the
                # time (measured 4.0 s -> 2.0 s for 20 s of 1080p60).
                stream.thread_type = "AUTO"
            tb = stream.time_base
            first_pts: Optional[int] = None
            for frame in container.decode(stream):
                pts = frame.pts if frame.pts is not None else frame.dts
                if pts is None:
                    continue
                if first_pts is None:
                    first_pts = pts
                t = seg.start + float((pts - first_pts) * tb)
                yield t, frame
                if t >= core_end - _TIME_EPS:
                    return

