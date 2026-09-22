"""
AudioSpikeTrigger tests: pure baseline/spike-detection logic, no ffmpeg
needed (measure_mean_volume(), which does need ffmpeg, is exercised
separately in test_live_buffer_and_cutter.py against real generated
segments).
"""
from VidFlow.live.audio_trigger import AudioSpikeTrigger


def test_first_sample_sets_baseline_and_never_fires():
    trigger = AudioSpikeTrigger(spike_db=8.0)
    assert trigger.observe(-20.0) is False
    assert trigger.baseline == -20.0


def test_fires_once_enough_baseline_history_and_a_real_spike_arrives():
    trigger = AudioSpikeTrigger(spike_db=8.0, min_baseline_samples=3)
    for volume in [-20.0, -21.0, -20.5, -19.5]:
        assert trigger.observe(volume) is False  # still building baseline / normal volume

    assert trigger.observe(-5.0) is True  # ~15dB above baseline


def test_does_not_fire_for_a_small_fluctuation_under_the_threshold():
    trigger = AudioSpikeTrigger(spike_db=8.0, min_baseline_samples=2)
    for volume in [-20.0, -20.0, -20.0]:
        trigger.observe(volume)

    assert trigger.observe(-15.0) is False  # only +5dB, under the 8dB threshold


def test_spike_does_not_drag_the_baseline_up():
    trigger = AudioSpikeTrigger(spike_db=8.0, min_baseline_samples=2, baseline_alpha=0.5)
    trigger.observe(-20.0)
    trigger.observe(-20.0)
    baseline_before_spike = trigger.baseline

    assert trigger.observe(-5.0) is True
    assert trigger.baseline == baseline_before_spike  # unchanged by the spike itself

    trigger.observe(-20.0)
    assert trigger.baseline == baseline_before_spike  # back to normal, baseline resumes tracking


def test_none_volume_is_ignored_and_never_a_spike():
    trigger = AudioSpikeTrigger()
    assert trigger.observe(None) is False
    assert trigger.baseline is None
