from leap_c.utils.logger import GroupWindowTracker
import numpy as np


def test_group_window_tracker_single_value():
    tracker = GroupWindowTracker(interval=2, window_size=3)

    for stats in tracker.update(0, {"a": 1}):
        assert False  # this should not be reached

    for timestamp, stats in tracker.update(1, {"a": 2}):
        assert timestamp == 1
        assert stats == {"a": 1.5}

    for stats in tracker.update(2, {"a": 3}):
        assert False

    for timestamp, stats in tracker.update(3, {"a": 4}):
        assert timestamp == 3
        assert stats == {"a": 3.0}


def test_group_window_tracker_empty():
    tracker = GroupWindowTracker(interval=1, window_size=3)

    for _ in tracker.update(0, {"a": 1, "b": 2}):
        pass

    for _ in tracker.update(1, {"a": 2}):
        pass

    for _, stats in tracker.update(2, {"a": 3}):
        assert stats == {"a": 2.0, "b": 2.0}

    for _, stats in tracker.update(3, {"a": 4}):
        assert stats["a"] == 3.0
        assert np.isnan(stats["b"])


def test_group_multi_report():
    tracker = GroupWindowTracker(interval=3, window_size=6)

    for _ in tracker.update(0, {"a": 1}):
        assert False  # this should not be reached

    timestamps = [2, 5, 8]
    all_stats = [{"a": 1.5}, {"a": 1.5}, {"a": 2}]

    for timestamp, stats in tracker.update(8, {"a": 2}):
        assert timestamp == timestamps.pop(0)
        assert stats == all_stats.pop(0)
