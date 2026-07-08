"""Synthetic day profiles (weather, electricity price) for the heating notebooks."""

import numpy as np


def make_comfort_schedule(
    n_steps: int, dt_hours: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Time-varying thermal comfort band as a function of the time of day.

    Occupancy-based schedule with a night setback, following the residential
    schedule used by the i4b building environment
    (``i4b.gym_interface.env.get_temperature_limits``):

    - night (22:00-08:00): band [12, 21] degC — nobody minds a cool house,
    - day (08:00-22:00): band [17, 21] degC.

    The day repeats every 24 h. A BOPTEST-style alternative would keep a
    21 degC lower bound while occupied (07:00-22:00), set back to 15 degC at
    night, with a looser upper bound (~24 degC); swap the values below to
    experiment.

    Args:
        n_steps: Number of time steps.
        dt_hours: Time step [h].

    Returns:
        t_lower: Lower comfort bound [degC], shape ``(n_steps,)``.
        t_upper: Upper comfort bound [degC], shape ``(n_steps,)``.
    """
    hour = (np.arange(n_steps) * dt_hours) % 24.0
    night = (hour >= 22.0) | (hour < 8.0)
    t_lower = np.where(night, 12.0, 17.0)
    t_upper = np.full(n_steps, 21.0)
    return t_lower, t_upper


def stack_forecast_windows(series: np.ndarray, n_windows: int, N_horizon: int) -> np.ndarray:
    """Stack sliding forecast windows of a signal into a batch.

    Window ``i`` is ``series[i : i + N_horizon + 1]`` — the horizon seen by a
    solve starting at step ``i``. The result has shape
    ``(n_windows, N_horizon + 1)``; append a trailing axis (``[..., None]``)
    to obtain the ``(B, N+1, 1)`` layout of a stagewise scalar parameter.
    """
    if len(series) < n_windows + N_horizon:
        raise ValueError(
            f"series of length {len(series)} is too short for {n_windows} windows "
            f"of length {N_horizon + 1}"
        )
    return np.stack([series[s : s + N_horizon + 1] for s in range(n_windows)])


def make_day_profiles(
    n_steps: int, dt_hours: float = 0.25, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize a deterministic outdoor-temperature and electricity-price profile.

    The day repeats every 24 h, so ``n_steps`` may cover more than one day
    (useful when a forecast window must slide past midnight).

    Args:
        n_steps: Number of time steps.
        dt_hours: Time step [h].
        seed: Seed for the small smooth noise on the temperature.

    Returns:
        t: Time since midnight [h], shape ``(n_steps,)``.
        outdoor_temp: Outdoor temperature [degC], coldest around 3 am,
            warmest around 3 pm.
        price: Electricity price [EUR/kWh], stepping up during the morning
            (7-9 h) and evening (17-20 h) peaks.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) * dt_hours
    hour = t % 24.0

    outdoor_temp = 8.0 + 6.0 * np.sin(2.0 * np.pi * (hour - 9.0) / 24.0)
    # Small smooth perturbation so the profile is not a textbook sinusoid.
    _bumps = rng.normal(0.0, 1.0, n_steps)
    _kernel = np.hanning(max(3, int(2.0 / dt_hours)))
    outdoor_temp = outdoor_temp + np.convolve(_bumps, _kernel / _kernel.sum(), mode="same")

    peak = ((hour >= 7.0) & (hour < 9.0)) | ((hour >= 17.0) & (hour < 20.0))
    price = np.where(peak, 0.35, 0.15)

    return t, outdoor_temp, price
