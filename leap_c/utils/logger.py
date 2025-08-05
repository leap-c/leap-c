import bisect
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
from torch.utils.tensorboard import SummaryWriter


@dataclass(kw_only=True)
class LoggerConfig:
    """
    Contains the necessary information for logging.

    Args:
        verbose: If True, the logger will collect also verbose statistics.
        interval: The interval at which statistics will be logged.
        window: The moving window size for the statistics.
        csv_logger: If True, the statistics will be logged to a CSV file.
        tensorboard_logger: If True, the statistics will be logged to TensorBoard.
        wandb_logger: If True, the statistics will be logged to Weights & Biases.
        wandb_init_kwargs: The kwargs to pass to wandb.init. If "dir" is not specified, it is set to output path / "wandb".
    """

    verbose: bool = False

    interval: int = 1000
    window: int = 10000

    csv_logger: bool = True
    tensorboard_logger: bool = True
    wandb_logger: bool = False
    wandb_init_kwargs: dict[str, Any] = field(default_factory=dict)


class GroupWindowTracker:
    def __init__(
        self,
        interval: int,
        window_size: int,
    ) -> None:
        """
        Initialize the group window tracker.

        Args:
            interval: The interval at which statistics will be logged.
            window_size: The moving window size for the statistics.
        """
        self._interval = interval
        self._window_size = window_size

        self._timestamps: dict = defaultdict(list)
        self._statistics: dict = defaultdict(list)

    def update(
        self, timestamp: int, stats: dict[str, float]
    ) -> Generator[tuple[int, dict[str, float]], None, None]:
        """
        Add timestamp and statistics to the tracker.

        This method adds the timestamp and statistics to the tracker. If the
        statistics are larger than the window size, the oldest statistics are
        removed. The statistics are smoothed with a moving window, if an
        interval is passed. This method might report multiple statistics at once.

        Args:
            timestamp: The timestamp of the statistics.
            stats: The statistics to be added.

        Returns:
            None if the statistics are not ready to be reported, or a tuple of
            report timestamps and statistics.
        """
        prev_timestamp = -1
        for key, value in stats.items():
            self._timestamps[key].append(timestamp)
            self._statistics[key].append(value)

            if len(self._timestamps[key]) > 1:
                prev_t = self._timestamps[key][-2]
                prev_timestamp = max(prev_timestamp, prev_t)

        if prev_timestamp == -1:
            return None

        interval_idx = (timestamp + 1) // self._interval
        interval_idx_prev = (prev_timestamp + 1) // self._interval

        if interval_idx == interval_idx_prev:
            return None

        def clean_until(t: int) -> None:
            for key in self._statistics.keys():
                while self._timestamps[key] and self._timestamps[key][0] <= t:
                    del self._timestamps[key][0]
                    del self._statistics[key][0]

        report_stamp = (interval_idx_prev + 1) * self._interval - 1
        while report_stamp <= timestamp:
            stats = {}

            clean_until(report_stamp - self._window_size)
            for key in self._statistics.keys():
                border_idx = bisect.bisect_right(
                    self._timestamps[key],
                    report_stamp,
                )
                if border_idx == 0:
                    stats[key] = np.nan
                else:
                    stats[key] = np.mean(
                        self._statistics[key][: border_idx + 1],
                    ).item()

            yield report_stamp, stats
            report_stamp += self._interval

        clean_until(timestamp - self._window_size)


class Logger:
    """
    A simple logger for statistics.

    This logger can write statistics to CSV, TensorBoard, and Weights & Biases.

    # TODO: Logging statistics to the console.

    Attributes:
        cfg: The configuration for the logger.
        output_path: The path to save the logs.
        state: The state of the logger.
        writer: The TensorBoard writer.
    """

    def __init__(self, cfg: LoggerConfig, output_path: str | Path) -> None:
        """
        Initialize the logger.

        Args:
            cfg: The configuration for the logger.
            output_path: The path to save the logs.
        """
        self.cfg = cfg
        self.output_path = Path(output_path)

        self.group_trackers = defaultdict(
            lambda: GroupWindowTracker(cfg.interval, cfg.window)
        )

        # init wandb
        if cfg.wandb_logger:
            if not cfg.wandb_init_kwargs.get("dir", False):  # type:ignore
                wandbdir = self.output_path / "wandb"
                wandbdir.mkdir(exist_ok=True)
                cfg.wandb_init_kwargs["dir"] = str(wandbdir)
            wandb.init(**cfg.wandb_init_kwargs)

        # tensorboard
        if cfg.tensorboard_logger:
            self.writer = SummaryWriter(self.output_path)

    def __call__(
        self,
        group: str,
        stats: dict[str, float | np.ndarray],
        timestamp: int,
        verbose: bool = False,
        with_smoothing: bool = True,
    ):
        """
        Report statistics.

        If the statistics are a numpy array, the array is split into multiple
        statistics of the form `key_{i}`.

        Args:
            group: The group of the statistics is added as a prefix to the log
                entry and determines how to split the statistics.
            stats: The statistics to be reported.
            timestamp: The timestamp of the logging entry.
            verbose: If True, the statistics will only be logged in verbosity mode.
            with_smoothing: If True, the statistics are smoothed with a moving window.
                This also results in the statistics being only reported at specific
                intervals.
        """
        if verbose and not self.cfg.verbose:
            return

        # split numpy arrays
        for key, value in list(stats.items()):
            if not isinstance(value, np.ndarray):
                continue

            if value.size == 1:
                stats[key] = float(value)
                continue

            assert value.ndim == 1, "Only 1D arrays are supported."

            stats.pop(key)
            for i, v in enumerate(value):
                stats[f"{key}_{i}"] = float(v)

        # find correct iterable
        if with_smoothing:
            report_loop = self.group_trackers[group].update(
                timestamp,
                stats,  # type:ignore
            )
        else:
            report_loop = [(timestamp, stats)]

        for report_timestamp, report_stats in report_loop:
            if self.cfg.wandb_logger:
                wandb.log(
                    {f"{group}/{k}": v for k, v in report_stats.items()},
                    step=report_timestamp,
                )

            if self.cfg.tensorboard_logger:
                for key, value in report_stats.items():
                    self.writer.add_scalar(f"{group}/{key}", value, report_timestamp)

            if self.cfg.csv_logger:
                csv_path = self.output_path / f"{group}_log.csv"

                if csv_path.exists():
                    kw = {"mode": "a", "header": False}
                else:
                    kw = {"mode": "w", "header": True}

                df = pd.DataFrame(report_stats, index=[report_timestamp])  # type: ignore
                df.to_csv(csv_path, **kw)

    def close(self) -> None:
        """
        Close the logger.

        This will close the TensorBoard writer and finish the Weights & Biases run.
        """
        if self.cfg.tensorboard_logger:
            self.writer.close()

        if self.cfg.wandb_logger:
            wandb.finish()
