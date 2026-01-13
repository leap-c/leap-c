from csv import DictReader
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from leap_c.utils.logger import GroupCumulativeTracker, GroupWindowTracker, Logger, LoggerConfig


def test_group_cumulative_tracker_accumulates_and_reports_at_intervals() -> None:
    """Tests that cumulative tracker accumulates values and reports at interval boundaries."""
    tracker = GroupCumulativeTracker(interval=3)

    # timestamps 0, 1 - accumulating, no report yet
    assert list(tracker.update(0, {"energy": 10.0})) == []
    assert list(tracker.update(1, {"energy": 20.0})) == []

    # timestamp 2 - crosses first interval boundary, reports cumulative sum
    reports = list(tracker.update(2, {"energy": 30.0}))
    assert reports == [(2, {"energy": 60.0})]

    # timestamps 3, 4 - accumulating
    assert list(tracker.update(3, {"energy": 5.0})) == []
    assert list(tracker.update(4, {"energy": 5.0})) == []

    # timestamp 5 - crosses second interval boundary
    reports = list(tracker.update(5, {"energy": 10.0}))
    assert reports == [(5, {"energy": 80.0})]  # 60 + 5 + 5 + 10 = 80


def test_logger_cumulative_flag(tmp_path: Path) -> None:
    """Tests that logger correctly uses cumulative tracking when cumulative=True."""
    group = "cumulative"

    cfg = LoggerConfig(
        csv_logger=True,
        tensorboard_logger=False,
        wandb_logger=False,
        interval=2,
    )
    with Logger(cfg, tmp_path) as logger:
        logger(group, {"cost": 100.0}, 0, cumulative=True)
        logger(group, {"cost": 50.0}, 1, cumulative=True)
        logger(group, {"cost": 25.0}, 2, cumulative=True)
        logger(group, {"cost": 25.0}, 3, cumulative=True)

    # verify CSV output
    csv_path = tmp_path / f"{group}_log.csv"
    assert csv_path.exists()

    with open(csv_path, "r") as f:
        reader = DictReader(f)
        rows = list(reader)

    # Should have 2 reports: at timestamps 1 and 3
    assert len(rows) == 2
    assert int(rows[0]["timestamp"]) == 1
    assert float(rows[0]["cost"]) == 150.0  # 100 + 50
    assert int(rows[1]["timestamp"]) == 3
    assert float(rows[1]["cost"]) == 200.0  # 100 + 50 + 25 + 25


def test_group_window_tracker_single_value() -> None:
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


def test_group_window_tracker_empty() -> None:
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


def test_group_multi_report() -> None:
    tracker = GroupWindowTracker(interval=3, window_size=6)

    for _ in tracker.update(0, {"a": 1}):
        assert False  # this should not be reached

    timestamps = [2, 5, 8]
    all_stats = [{"a": 1.5}, {"a": 1.5}, {"a": 2}]

    for timestamp, stats in tracker.update(8, {"a": 2}):
        assert timestamp == timestamps.pop(0)
        assert stats == all_stats.pop(0)


def test_logger_fails_if_not_initialized(tmp_path: Path) -> None:
    """Tests that the logger raises an error if called before being initialized."""
    cfg = LoggerConfig(csv_logger=True, tensorboard_logger=False, wandb_logger=False)
    logger = Logger(cfg, tmp_path)
    with pytest.raises(RuntimeError):
        logger("a_group", {}, 0, with_smoothing=False)


def test_logger_writes_to_csv_correctly(tmp_path: Path) -> None:
    """Tests that the logger writes data to CSV correctly."""
    # create dummy data
    groups = ("group1", "group2")
    timestamps = tuple(range(10))
    stats = [{"a": i * 2.0, "b": i / 2.0} for i in timestamps]

    # log data
    cfg = LoggerConfig(csv_logger=True, tensorboard_logger=False, wandb_logger=False)
    with Logger(cfg, tmp_path) as logger:
        for group, timestamp in product(groups, timestamps):
            logger(group, stats[timestamp], timestamp, with_smoothing=False)

    # test CSV files have been created and contain the correct data
    for group in groups:
        csv_path = tmp_path / f"{group}_log.csv"
        assert csv_path.exists(), f"CSV file for group {group} does not exist."

        with open(csv_path, "r") as f:
            reader = DictReader(f)
            rows = list(reader)

        for k, row in enumerate(rows):
            timestamp = int(row.pop("timestamp"))
            row = {key: float(value) for key, value in row.items()}
            assert timestamp == k, f"Timestamp mismatch in group {group} at row {k}."
            assert row == stats[k], f"Data mismatch in group {group} at row {k}."
