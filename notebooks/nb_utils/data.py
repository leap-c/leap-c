"""Synthetic day profiles (weather, electricity price) for the heating notebooks."""

import numpy as np


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
