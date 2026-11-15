"""Scoring of candidate windows. See ``visual`` (design section 6.6)."""

from .visual import (
    DEFAULT_PROFILE,
    SCORE_KEYS,
    GameProfile,
    VisualScorer,
    crop_regions,
    frame_score,
    region_bounds,
)

__all__ = [
    "DEFAULT_PROFILE",
    "SCORE_KEYS",
    "GameProfile",
    "VisualScorer",
    "crop_regions",
    "frame_score",
    "region_bounds",
]
