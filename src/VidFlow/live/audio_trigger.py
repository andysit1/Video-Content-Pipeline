"""
Tier 1's trigger signal: a rolling audio-loudness baseline over the buffer
segments, firing when a new segment is meaningfully louder than recent
history (crowd noise, a reaction, a big play). This is a deliberately
general, game-agnostic proxy -- unlike the crosshair/kill-feed scoring
elsewhere in this pipeline, it works for any streamer/game, which matters
since Tier 1 is meant to watch channels you don't control or tune for.

Chat-velocity and image-based signals are natural additions later (see the
live/README note), but audio alone is a reasonable, low-complexity first
signal to prove the buffer -> trigger -> instant-clip mechanism end to end.
"""
import re
import subprocess

_MEAN_VOLUME_RE = re.compile(r'mean_volume:\s*(-?[0-9.]+)\s*dB')


def measure_mean_volume(segment_path):
    """Runs ffmpeg's volumedetect on one segment file; returns its mean dB, or None if unreadable."""
    cmd = ["ffmpeg", "-i", segment_path, "-af", "volumedetect", "-f", "null", "-"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    match = _MEAN_VOLUME_RE.search(result.stderr)
    return float(match.group(1)) if match else None


class AudioSpikeTrigger:
    def __init__(self, spike_db=8.0, baseline_alpha=0.2, min_baseline_samples=3):
        self.spike_db = spike_db
        self.baseline_alpha = baseline_alpha
        self.min_baseline_samples = min_baseline_samples
        self._baseline = None
        self._sample_count = 0

    @property
    def baseline(self):
        return self._baseline

    def observe(self, mean_volume_db):
        """Feed one segment's mean volume in. Returns True if this segment is a spike."""
        if mean_volume_db is None:
            return False

        is_spike = (
            self._baseline is not None
            and self._sample_count >= self.min_baseline_samples
            and (mean_volume_db - self._baseline) >= self.spike_db
        )

        if self._baseline is None:
            self._baseline = mean_volume_db
        elif not is_spike:
            # exponential moving average; skip updating on a spike itself so
            # one loud segment doesn't drag the baseline up and mask the next
            self._baseline = (
                self.baseline_alpha * mean_volume_db + (1 - self.baseline_alpha) * self._baseline
            )

        self._sample_count += 1
        return is_spike
