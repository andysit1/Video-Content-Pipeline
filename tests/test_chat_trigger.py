"""
ChatVelocityTrigger tests: pure ratio/baseline logic, no IRC or ffmpeg
needed.
"""
from VidFlow.live.chat_trigger import ChatVelocityTrigger


def test_first_sample_sets_baseline_and_never_fires():
    trigger = ChatVelocityTrigger()
    assert trigger.observe(2.0) is False
    assert trigger.baseline == 2.0


def test_fires_on_a_large_multiple_of_baseline():
    trigger = ChatVelocityTrigger(spike_ratio=3.0, min_baseline_samples=3)
    for rate in [1.0, 1.1, 0.9, 1.0]:
        assert trigger.observe(rate) is False

    assert trigger.observe(5.0) is True  # 5x a ~1msg/s baseline


def test_does_not_fire_on_a_ratio_under_the_threshold():
    trigger = ChatVelocityTrigger(spike_ratio=3.0, min_baseline_samples=2)
    trigger.observe(2.0)
    trigger.observe(2.0)
    assert trigger.observe(4.0) is False  # only 2x, under the 3x threshold


def test_min_absolute_rate_guards_against_dead_chat_noise():
    # baseline near zero -> even a tiny absolute rate is a huge "ratio",
    # but it shouldn't count as a real spike below min_absolute_rate.
    trigger = ChatVelocityTrigger(spike_ratio=3.0, min_absolute_rate=0.5, min_baseline_samples=2)
    trigger.observe(0.01)
    trigger.observe(0.01)
    assert trigger.observe(0.1) is False  # 10x baseline, but only 0.1 msg/s absolute

    assert trigger.observe(0.6) is True  # clears both the ratio and the absolute floor


def test_spike_does_not_drag_the_baseline_up():
    trigger = ChatVelocityTrigger(spike_ratio=3.0, min_baseline_samples=2, baseline_alpha=0.5)
    trigger.observe(1.0)
    trigger.observe(1.0)
    baseline_before_spike = trigger.baseline

    assert trigger.observe(10.0) is True
    assert trigger.baseline == baseline_before_spike


def test_none_rate_is_ignored_and_never_a_spike():
    trigger = ChatVelocityTrigger()
    assert trigger.observe(None) is False
    assert trigger.baseline is None
