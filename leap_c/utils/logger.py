import bisect
from collections import defaultdict, deque
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(kw_only=True)
class LoggerConfig:
    """Contains the necessary information for logging.

    Args:
        verbose: If `True`, the logger will collect also verbose statistics.
        interval: The interval at which statistics will be logged (in steps).
        window: The moving window size for the statistics (in steps).
        csv_logger: If `True`, the statistics will be logged to a CSV file.
        tensorboard_logger: If `True`, the statistics will be logged to TensorBoard.
        wandb_logger: If `True`, the statistics will be logged to Weights & Biases.
        wandb_init_kwargs: The kwargs to pass to wandb.init. If `"dir"` is not specified, it is set
            to `output_path / "wandb"`.
    """

    verbose: bool = False

    interval: int = 1_000
    window: int = 10_000

    csv_logger: bool = True
    tensorboard_logger: bool = True
    wandb_logger: bool = False
    wandb_init_kwargs: dict[str, Any] = field(default_factory=dict)


class GroupWindowTracker:
    def __init__(self, interval: int, window_size: int) -> None:
        """Initialize the group window tracker.

        Args:
            interval: The interval at which statistics will be logged.
            window_size: The moving window size for the statistics.
        """
        self._interval = interval
        self._window_size = window_size

        self._statistics: dict[str, deque[tuple[int, float]]] = defaultdict(deque)
        self._running_sums: dict[str, float] = defaultdict(float)

    def update(
        self, timestamp: int, stats: dict[str, float]
    ) -> Generator[tuple[int, dict[str, float]], None, None]:
        """Add timestamp and statistics to the tracker.

        This method adds the timestamp and statistics to the tracker. If the
        statistics are larger than the window size, the oldest statistics are
        removed. The statistics are smoothed with a moving window, if an
        interval is passed. This method might report multiple statistics at once.

        Args:
            timestamp: The timestamp of the statistics.
            stats: The statistics to be added.

        Returns:
            `None` if the statistics are not ready to be reported, or a tuple of report timestamps
            and statistics.
        """
        prev_timestamp = -1
        for key, value in stats.items():
            key_statistics = self._statistics[key]
            key_statistics.append((timestamp, value))
            self._running_sums[key] += value
            if len(key_statistics) > 1:
                prev_t, _ = key_statistics[-2]
                prev_timestamp = max(prev_timestamp, prev_t)

        if prev_timestamp == -1:
            return

        interval_idx = (timestamp + 1) // self._interval
        interval_idx_prev = (prev_timestamp + 1) // self._interval

        if interval_idx == interval_idx_prev:
            return

        def clean_until(t: int) -> None:
            for key in self._statistics:
                key_statistics = self._statistics[key]
                key_sum = self._running_sums[key]
                while key_statistics and key_statistics[0][0] <= t:
                    _, value = key_statistics.popleft()
                    key_sum -= value
                self._running_sums[key] = key_sum

        report_stamp = (interval_idx_prev + 1) * self._interval - 1
        while report_stamp <= timestamp:
            stats = {}

            clean_until(report_stamp - self._window_size)
            for key in self._statistics:
                key_statistics = self._statistics[key]
                border_idx = bisect.bisect_right(key_statistics, report_stamp, key=lambda x: x[0])
                if border_idx == 0:
                    stats[key] = float("nan")
                else:
                    length = min(border_idx + 1, len(key_statistics))
                    stats[key] = self._running_sums[key] / length

            yield report_stamp, stats
            report_stamp += self._interval

        clean_until(timestamp - self._window_size)


class Logger:
    """A simple logger for statistics.

    This logger can write statistics to CSV, TensorBoard, and Weights & Biases.

    # TODO: Logging statistics to the console.

    Attributes:
        cfg: The configuration for the logger.
        output_path: The path to save the logs.
        group_trackers: A dictionary of group trackers for smoothing statistics.
    """

    cfg: LoggerConfig
    output_path: Path
    group_trackers: dict[str, GroupWindowTracker]

    def __init__(self, cfg: LoggerConfig, output_path: str | Path) -> None:
        """Initialize the logger, but does not start it.

        Before using the logger, call `__enter__`, e.g., via a `with` statement. This will ensure
        that the logger is properly started and stopped.

        Args:
            cfg: The configuration for the logger.
            output_path: The path to save the logs.
        """
        self.cfg = cfg
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.group_trackers = defaultdict(lambda: GroupWindowTracker(cfg.interval, cfg.window))

    def __enter__(self) -> "Logger":
        """Starts the logger.

        This will initialize the TensorBoard writer, Weights & Biases run, and CSV file.

        Returns:
            Logger: A reference to the logger itself.
        """
        cfg = self.cfg

        if cfg.wandb_logger:
            import wandb

            if not cfg.wandb_init_kwargs.get("dir", False):  # type:ignore
                cfg.wandb_init_kwargs["dir"] = str(self.output_path)
            wandb.init(**cfg.wandb_init_kwargs)
            self._wandb_defined_metrics: dict[str, bool] = {}

        if cfg.tensorboard_logger:
            from torch.utils.tensorboard import SummaryWriter

            self._tensorboard_writer = SummaryWriter(self.output_path)

        if cfg.csv_logger:
            from csv import DictWriter
            from typing import TextIO

            self._csv_files_and_writers: dict[str, tuple[TextIO, DictWriter]] = {}
        return self

    def __exit__(self, *_, **__) -> None:
        """Closes the logger.

        This closes the TensorBoard writer and CSV file, and finishes the Weights & Biases run.
        """
        cfg = self.cfg

        if cfg.tensorboard_logger:
            self._tensorboard_writer.close()

        if cfg.wandb_logger:
            import wandb

            wandb.finish()

        if cfg.csv_logger:
            for csv_file, _ in self._csv_files_and_writers.values():
                if csv_file:
                    csv_file.close()

    def __call__(
        self,
        group: str,
        stats: dict[str, float | np.ndarray],
        timestamp: int,
        verbose: bool = False,
        with_smoothing: bool = True,
    ) -> None:
        """Report statistics.

        If the statistics are a numpy array, the array is split into multiple
        statistics of the form `key_{i}`.

        Args:
            group: The group of the statistics is added as a prefix to the log entry and determines
                how to split the statistics.
            stats: The statistics to be reported.
            timestamp: The timestamp of the logging entry.
            verbose: If `True`, the statistics will only be logged in verbosity mode.
            with_smoothing: If `True`, the statistics are smoothed with a moving window.
                This also results in the statistics being only reported at specific intervals.
        """
        cfg = self.cfg
        if (
            (cfg.tensorboard_logger and not hasattr(self, "_tensorboard_writer"))
            or (cfg.wandb_logger and not hasattr(self, "_wandb_defined_metrics"))
            or (cfg.csv_logger and not hasattr(self, "_csv_files_and_writers"))
        ):
            raise RuntimeError(
                "Logger waws not started before calling it. Must be initialized with `__enter__`, "
                "e.g., via `with Logger(...) as logger:`."
            )

        if verbose and not cfg.verbose:
            return

        if cfg.wandb_logger and not self._wandb_defined_metrics.get(group, False):
            import wandb

            wandb.define_metric(f"{group}/*", f"{group}/step")
            self._wandb_defined_metrics[group] = True

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
            if cfg.wandb_logger:
                import wandb

                wandb.log(
                    {
                        f"{group}/step": report_timestamp,
                        **{f"{group}/{k}": v for k, v in report_stats.items()},
                    }
                )

            if cfg.tensorboard_logger:
                for key, value in report_stats.items():
                    self._tensorboard_writer.add_scalar(f"{group}/{key}", value, report_timestamp)

            if cfg.csv_logger:
                csv_path = self.output_path / f"{group}_log.csv"

                if group in self._csv_files_and_writers:
                    csv_file, csv_writer = self._csv_files_and_writers[group]
                else:
                    from csv import DictWriter

                    csv_file = open(csv_path, mode="a", newline="", buffering=1)
                    csv_writer = DictWriter(csv_file, fieldnames=["timestamp"] + list(report_stats))
                    if csv_file.tell() == 0:
                        csv_writer.writeheader()
                    self._csv_files_and_writers[group] = (csv_file, csv_writer)

                csv_writer.writerow({"timestamp": report_timestamp, **report_stats})
                csv_writer.writer
                csv_file.flush()
