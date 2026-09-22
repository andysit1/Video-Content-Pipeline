"""
ChatVelocityTrigger: fires when the current chat rate is a large multiple
of its recent baseline. Ratio-based rather than an additive delta (unlike
AudioSpikeTrigger's dB delta) because baseline chat activity varies wildly
by channel size -- a small streamer's baseline might be 0.2 msg/s, a huge
one 20 msg/s, and a fixed "+N msg/s" threshold tuned for one is useless
for the other. `min_absolute_rate` guards the ratio against firing on
statistical noise in a near-dead chat (e.g. baseline 0.02, current 0.1 is
"5x" but is one extra message, not a reaction).
"""


class ChatVelocityTrigger:
    def __init__(self, spike_ratio=3.0, min_absolute_rate=0.5, baseline_alpha=0.3,
                 min_baseline_samples=3):
        self.spike_ratio = spike_ratio
        self.min_absolute_rate = min_absolute_rate
        self.baseline_alpha = baseline_alpha
        self.min_baseline_samples = min_baseline_samples
        self._baseline = None
        self._sample_count = 0

    @property
    def baseline(self):
        return self._baseline

    def observe(self, messages_per_second):
        """Feed one rate reading in. Returns True if this reading is a spike."""
        if messages_per_second is None:
            return False

        is_spike = (
            self._baseline is not None
            and self._sample_count >= self.min_baseline_samples
            and messages_per_second >= self.min_absolute_rate
            and messages_per_second >= self._baseline * self.spike_ratio
        )

        if self._baseline is None:
            self._baseline = messages_per_second
        elif not is_spike:
            self._baseline = (
                self.baseline_alpha * messages_per_second + (1 - self.baseline_alpha) * self._baseline
            )

        self._sample_count += 1
        return is_spike
