"""Loudness extraction and rolling loudness statistics (design section 6.3).

:class:`LoudnessExtractor` turns segments into a regular series of
:class:`~VidFlow.realtime.types.LoudnessSample` (RMS dBFS per hop, 100 ms by
default). :class:`RollingLoudnessStats` is the streaming replacement for the
batch pipeline's whole-file ``volumedetect`` histogram.

Timing
    Sample times never come from packet or frame timestamps (container
    timestamps are not trustworthy, design section 2). The extractor anchors
    the series at a segment's media-time ``start`` (the first segment, or the
    first one after a discontinuity / :meth:`LoudnessExtractor.reset`) and
    from there counts decoded samples: hop ``k`` starts at
    ``anchor + samples_before_it / rate``. Because the audio packets of
    consecutive segments are contiguous, counting is exact; anchoring every
    segment on its own ``start`` instead would add up to one audio frame
    (about 21 ms) of jitter per segment, since segment cuts follow video
    keyframes, not audio frames. If the count ever drifts from a segment's
    ``start`` by more than ``resync_s`` (missing or extra audio inside a
    segment), the series is re-anchored on that segment's ``start``.

Levels
    Audio is resampled to ``rate`` Hz keeping its channels, and each hop's
    level is the RMS over every channel's samples (the mean power across
    channels, as FFmpeg's ``volumedetect`` measures it). This is the
    "mono" series: one level per hop. A plain swresample stereo-to-mono
    downmix was not used because it sums the channels at -3 dB each, which
    reads identical channels 3 dB hot and lets out-of-phase channels cancel.

Carry-over
    Samples that do not fill a whole hop at the end of a segment are kept and
    completed by the next segment, so no audio is lost at boundaries. They are
    dropped on a discontinuity, on :meth:`LoudnessExtractor.reset`, and after
    a segment that yielded no audio.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Deque, Iterable, List, Optional, Tuple

import av
import numpy as np

from .types import LoudnessSample, Segment

__all__ = [
    "DB_FLOOR",
    "DB_CEIL",
    "LoudnessExtractor",
    "RollingLoudnessStats",
    "rms_db",
]

log = logging.getLogger(__name__)

#: Level reported for digital silence, and the lowest level the stats track.
DB_FLOOR: float = -120.0
#: Highest level the stats histogram tracks (float audio can exceed 0 dBFS).
DB_CEIL: float = 20.0

_POWER_FLOOR = 10.0 ** (DB_FLOOR / 10.0)


def rms_db(frames: np.ndarray) -> np.ndarray:
    """RMS level in dBFS of each row of ``frames`` (shape ``(n, hop)``), floored at :data:`DB_FLOOR`."""
    x = np.asarray(frames, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return _power_db(np.mean(np.square(x), axis=1))


def _power_db(mean_square: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(mean_square, _POWER_FLOOR))


class LoudnessExtractor:
    """Decode a segment's audio and return its loudness, one sample per hop.

    ``feed(segment)`` decodes only the audio stream with PyAV, resamples it to
    ``rate`` Hz and returns RMS dBFS (over all channels) for consecutive
    ``hop_s`` hops.
    Not thread-safe; one extractor per channel, fed in segment order.
    """

    def __init__(self, hop_s: float = 0.1, rate: int = 16000, resync_s: float = 0.25) -> None:
        if not hop_s > 0:
            raise ValueError(f"hop_s must be > 0, got {hop_s!r}")
        if int(rate) != rate or rate <= 0:
            raise ValueError(f"rate must be a positive integer, got {rate!r}")
        hop = int(round(hop_s * rate))
        if hop < 1:
            raise ValueError(f"hop_s * rate must be at least one sample, got {hop_s} * {rate}")
        if not resync_s > 0:
            raise ValueError(f"resync_s must be > 0, got {resync_s!r}")
        self.hop_s = float(hop_s)
        self.rate = int(rate)
        self.resync_s = float(resync_s)
        self._hop = hop  # samples per hop
        # Leftover per-sample power (mean square across channels), fewer than one hop.
        self._pending = np.zeros(0, dtype=np.float32)
        self._anchor: Optional[float] = None  # media time of sample 0 of the current run
        self._consumed = 0  # samples turned into hops since the anchor
        self._resampler: Optional[av.AudioResampler] = None
        self._resampler_key: Optional[Tuple[str, str, int]] = None

    # ------------------------------------------------------------------ public

    @property
    def hop_samples(self) -> int:
        """Samples per hop at ``rate``."""
        return self._hop

    @property
    def pending_samples(self) -> int:
        """Samples carried over, waiting for the next segment to complete a hop."""
        return int(self._pending.size)

    def reset(self) -> None:
        """Drop carried-over samples and the time anchor (as at a discontinuity)."""
        self._pending = np.zeros(0, dtype=np.float32)
        self._anchor = None
        self._consumed = 0
        self._resampler = None
        self._resampler_key = None

    def feed(self, segment: Segment) -> List[LoudnessSample]:
        """Return the loudness samples completed by ``segment``.

        Never raises for bad media: a segment with no audio stream or that
        cannot be decoded logs a warning and returns ``[]``.
        """
        if segment.discontinuity:
            self.reset()

        audio = self._decode(segment)
        if audio is None:
            # The audio timeline is broken here: whatever comes next starts a new run.
            self.reset()
            return []

        if self._anchor is None:
            self._anchor = float(segment.start)
            self._consumed = 0
        else:
            expected = self._anchor + (self._consumed + self._pending.size) / self.rate
            drift = expected - float(segment.start)
            if abs(drift) > self.resync_s:
                log.warning(
                    "%s seq %d: audio count is %.3f s off the segment start; re-anchoring",
                    segment.channel,
                    segment.seq,
                    drift,
                )
                # Keep the carried samples: they are still the audio just before this segment.
                self._anchor = float(segment.start) - self._pending.size / self.rate
                self._consumed = 0

        buf = np.concatenate((self._pending, audio)) if self._pending.size else audio
        n_hops = buf.size // self._hop
        used = n_hops * self._hop
        if n_hops:
            dbs = _power_db(buf[:used].reshape(n_hops, self._hop).mean(axis=1, dtype=np.float64))
        else:
            dbs = np.zeros(0)
        # Copy so the (small) leftover does not keep the whole segment's array alive.
        self._pending = buf[used:].copy()

        base = self._consumed
        anchor = self._anchor
        rate = self.rate
        hop = self._hop
        out = [
            LoudnessSample(t=anchor + (base + k * hop) / rate, db=float(db))
            for k, db in enumerate(dbs.tolist())
        ]
        self._consumed = base + used
        return out

    # ----------------------------------------------------------------- private

    @staticmethod
    def _power(out: av.AudioFrame) -> np.ndarray:
        # Planar float: shape (channels, samples). Mean square across channels.
        x = out.to_ndarray()
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.mean(np.square(x, dtype=np.float32), axis=0)

    def _resample(self, frame: av.AudioFrame, chunks: List[np.ndarray]) -> None:
        key = (frame.format.name, frame.layout.name, int(frame.sample_rate))
        if self._resampler is None or key != self._resampler_key:
            if self._resampler is not None:
                self._flush_resampler(chunks)
            self._resampler = av.AudioResampler(format="fltp", layout=frame.layout.name, rate=self.rate)
            self._resampler_key = key
        for out in self._resampler.resample(frame):
            chunks.append(self._power(out))

    def _flush_resampler(self, chunks: List[np.ndarray]) -> None:
        if self._resampler is None:
            return
        try:
            for out in self._resampler.resample(None):
                chunks.append(self._power(out))
        except Exception:  # noqa: BLE001 - a failed flush only loses a few samples
            log.debug("resampler flush failed", exc_info=True)

    def _decode(self, segment: Segment) -> Optional[np.ndarray]:
        """Decode ``segment``'s audio at ``rate`` and return per-sample power
        (mean square across channels, float32); ``None`` if there is none.

        The resampler is carried across segments so its filter state is
        continuous; packets that fail to decode are skipped with a warning.
        """
        chunks: List[np.ndarray] = []
        bad_packets = 0
        try:
            with av.open(str(segment.path), mode="r") as container:
                if not container.streams.audio:
                    log.warning("%s seq %d: no audio stream in %s", segment.channel, segment.seq, segment.path)
                    return None
                stream = container.streams.audio[0]
                for packet in container.demux(stream):
                    try:
                        frames = packet.decode()
                    except av.FFmpegError:
                        bad_packets += 1
                        continue
                    for frame in frames:
                        self._resample(frame, chunks)
        except Exception as exc:  # noqa: BLE001 - bad media must never take down ingest
            log.warning(
                "%s seq %d: could not decode audio from %s: %s",
                segment.channel,
                segment.seq,
                segment.path,
                exc,
            )
            return None
        if bad_packets:
            log.warning("%s seq %d: skipped %d undecodable audio packets", segment.channel, segment.seq, bad_packets)
        if not chunks:
            log.warning("%s seq %d: no decodable audio in %s", segment.channel, segment.seq, segment.path)
            return None
        audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        return np.ascontiguousarray(audio, dtype=np.float32)


class RollingLoudnessStats:
    """Histogram of 1 dB bins over the trailing ``window_s`` of loudness samples.

    ``update`` is O(1); ``percentile`` is O(number of bins), a constant. Each
    bin also keeps the sum of its values, so a percentile is reported as the
    mean of the bin it falls in (exact when a bin holds one repeated level)
    and interpolated between neighbouring ranks like ``numpy.percentile``'s
    default ``linear`` method. The error against ``numpy.percentile`` over
    the same samples is below one bin (1 dB). Levels outside
    ``[DB_FLOOR, DB_CEIL]`` are clamped.
    """

    def __init__(self, window_s: float, hop_s: float) -> None:
        if not window_s > 0 or not hop_s > 0:
            raise ValueError(f"window_s and hop_s must be > 0, got {window_s!r}, {hop_s!r}")
        self.window_s = float(window_s)
        self.hop_s = float(hop_s)
        self.capacity = max(1, int(round(self.window_s / self.hop_s)))
        self._lo = int(math.floor(DB_FLOOR))
        self._nbins = int(math.ceil(DB_CEIL)) - self._lo + 1
        self._counts: List[int] = [0] * self._nbins
        self._sums: List[float] = [0.0] * self._nbins
        self._values: Deque[Tuple[int, float]] = deque()

    @property
    def count(self) -> int:
        """Number of samples currently in the window (at most ``capacity``)."""
        return len(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def reset(self) -> None:
        self._counts = [0] * self._nbins
        self._sums = [0.0] * self._nbins
        self._values.clear()

    def update(self, db: float) -> None:
        """Add one sample; the oldest one ages out once the window is full."""
        v = float(db)
        if math.isnan(v):
            raise ValueError("loudness sample is NaN")
        v = min(max(v, DB_FLOOR), DB_CEIL)
        b = int(math.floor(v)) - self._lo
        self._values.append((b, v))
        self._counts[b] += 1
        self._sums[b] += v
        if len(self._values) > self.capacity:
            ob, ov = self._values.popleft()
            c = self._counts[ob] - 1
            self._counts[ob] = c
            # Reset to exactly zero when a bin empties so float error cannot build up.
            self._sums[ob] = self._sums[ob] - ov if c else 0.0

    def extend(self, dbs: Iterable[float]) -> None:
        for db in dbs:
            self.update(db)

    def percentile(self, p: float) -> float:
        """Approximate ``numpy.percentile(window, p)``; raises ``ValueError`` if empty."""
        if not 0.0 <= p <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {p!r}")
        n = len(self._values)
        if n == 0:
            raise ValueError("percentile of an empty RollingLoudnessStats")
        rank = (p / 100.0) * (n - 1)
        lo_rank = int(math.floor(rank))
        hi_rank = min(lo_rank + 1, n - 1)
        frac = rank - lo_rank
        lo_val = hi_val = None
        seen = 0
        counts = self._counts
        for b in range(self._nbins):
            c = counts[b]
            if not c:
                continue
            seen += c
            if lo_val is None and lo_rank < seen:
                lo_val = self._sums[b] / c
            if hi_rank < seen:
                hi_val = self._sums[b] / c
                break
        assert lo_val is not None and hi_val is not None
        return lo_val + (hi_val - lo_val) * frac
