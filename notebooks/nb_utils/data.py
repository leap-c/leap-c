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


def make_prosumer_profiles(
    n_steps: int,
    dt_hours: float = 0.25,
    peak_height: float = 0.50,
    level_shift: float = 0.0,
    pv_kwp: float = 4.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize a deterministic prosumer day: weather, dynamic tariff, PV.

    The day repeats every 24 h. The buy price mimics a household dynamic
    tariff (EPEX spot shape plus fixed surcharges): a 0.22 EUR/kWh overnight
    base, a morning peak to 0.32 around 08:00, a midday solar dip to 0.18,
    and an evening peak around 18:30 whose height is the main experiment
    knob. The three knobs (``peak_height``, ``level_shift``, ``pv_kwp``) are
    exactly the slider axes of notebook 08.

    Args:
        n_steps: Number of time steps.
        dt_hours: Time step [h].
        peak_height: Absolute buy price at the top of the evening peak
            [EUR/kWh]; 0.25 means "no evening event", 0.50 a dramatic one.
        level_shift: Uniform shift of the whole buy-price profile [EUR/kWh].
        pv_kwp: Peak PV generation at 12:30 on a clear day [kW].
        seed: Seed for the small smooth noise on the temperature.

    Returns:
        t: Time since midnight [h], shape ``(n_steps,)``.
        outdoor_temp: Outdoor temperature [degC], a shoulder-season day —
            coldest ~2 degC around 03:00, so heating is always needed.
        price_buy: Dynamic-tariff buy price [EUR/kWh].
        p_pv: Clear-sky PV generation [kW], zero outside ~6:30-18:30.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) * dt_hours
    hour = t % 24.0

    outdoor_temp = 8.0 + 6.0 * np.sin(2.0 * np.pi * (hour - 9.0) / 24.0)
    _bumps = rng.normal(0.0, 1.0, n_steps)
    _kernel = np.hanning(max(3, int(2.0 / dt_hours)))
    outdoor_temp = outdoor_temp + np.convolve(_bumps, _kernel / _kernel.sum(), mode="same")

    # Smooth dynamic tariff: base + morning peak + midday solar dip + evening peak.
    price_buy = (
        0.22
        + 0.10 * np.exp(-(((hour - 8.0) / 1.0) ** 2))
        - 0.04 * np.exp(-(((hour - 13.0) / 2.5) ** 2))
        + (peak_height - 0.22) * np.exp(-(((hour - 18.5) / 1.5) ** 2))
        + level_shift
    )
    # Defensive floor: keeps price_buy above any plausible feed-in tariff.
    price_buy = np.maximum(price_buy, 0.10)

    # Clear-sky PV bell, peaking at 12:30, zero at night.
    p_pv = np.where(
        np.abs(hour - 12.5) < 6.0,
        pv_kwp * np.cos(np.pi * (hour - 12.5) / 12.0) ** 2,
        0.0,
    )

    return t, outdoor_temp, price_buy, p_pv
