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
