import numpy as np
from scipy.constants import convert_temperature


def set_temperature_limits(
    quarter_hours: np.ndarray,
    night_start_hour: int = 22,
    night_end_hour: int = 8,
    lb_night: float = convert_temperature(12.0, "celsius", "kelvin"),
    lb_day: float = convert_temperature(19.0, "celsius", "kelvin"),
    ub_night: float = convert_temperature(25.0, "celsius", "kelvin"),
    ub_day: float = convert_temperature(22.0, "celsius", "kelvin"),
) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    """Set temperature limits based on the time of day."""
    hours = np.floor(quarter_hours / 4)

    # Vectorized night detection
    night_idx = (hours >= night_start_hour) | (hours < night_end_hour)

    # Initialize and set values
    lb = np.where(night_idx, lb_night, lb_day)
    ub = np.where(night_idx, ub_night, ub_day)
    return lb, ub
