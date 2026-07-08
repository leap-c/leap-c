"""A "true" house for the closed-loop notebooks (06-08).

The environment has the same R1C1 *structure* the MPC believes in, plus
everything the MPC does not know:

- a leakier envelope (``R_TRUE = 1.4`` vs. the model's ``R_THERMAL = 2.0``),
- occupancy heat gains in the evening (people, cooking, appliances),
- a small AR(1) thermal disturbance.

``true_step`` and ``step_cost`` are dual-backend pure functions: the identical
code runs on numpy values (closed-loop demos, imitation-learning data) and on
torch tensors (the differentiable rollout of the RL notebook) — they only use
arithmetic and ``.clip(min=...)``.
"""

import numpy as np

from .data import make_comfort_schedule, make_day_profiles
from .heating import C_THERMAL

R_TRUE = 1.4  # the true envelope leaks ~40% faster than the model's R=2.0


def true_step(T, q, outdoor_temp, gain=0.0, dt=0.25, R=R_TRUE, C=C_THERMAL):
    """One Euler step of the true room dynamics (numpy or torch inputs).

    Identical structure to the OCP model, but with the true parameters and an
    additive thermal disturbance ``gain`` [kW] the MPC never sees.
    """
    return T + dt * ((outdoor_temp - T) / (R * C) + (q + gain) / C)


def step_cost(price, q, T_next, t_lower, t_upper, dt=0.25, w_comfort=1.0):
    """Energy cost plus weighted thermal discomfort for one step [EUR].

    BOPTEST-style objective ``cost + w * discomfort``: the energy bill
    ``price * q * dt`` plus ``w_comfort`` [EUR/(K h)] times the band violation
    integrated over the step [K h]. Works on numpy and torch inputs alike
    (pass numpy scalars, not plain floats, for the numpy backend).
    """
    under = (t_lower - T_next).clip(min=0.0)
    over = (T_next - t_upper).clip(min=0.0)
    return price * q * dt + w_comfort * (under + over) * dt


class HouseEnv:
    """The true house as a small ``reset``/``step`` environment (no gymnasium).

    Observations are dicts holding the current room temperature and the
    forecast windows an MPC planner needs, so ``obs`` can be handed to
    ``HeatingPlanner.forward`` directly (after batching):

    ``{"T": float, "k": int, "outdoor": (N+1,), "price": (N+1,),
    "t_lower": (N+1,), "t_upper": (N+1,)}``

    The reward is ``-step_cost(...)``; ``info`` reports the decomposition.
    """

    def __init__(
        self,
        n_days: int = 3,
        dt: float = 0.25,
        N_forecast: int = 32,
        R_true: float = R_TRUE,
        C_true: float = C_THERMAL,
        q_max: float = 12.0,
        w_comfort: float = 1.0,
        gain_kw: float = 1.5,
        noise_std: float = 0.3,
        seed: int = 0,
    ):
        self.dt = dt
        self.N_forecast = N_forecast
        self.R_true = R_true
        self.C_true = C_true
        self.q_max = q_max
        self.w_comfort = w_comfort
        self.n_steps = n_days * round(24.0 / dt)

        # Profiles cover the episode plus one full forecast window.
        n_total = self.n_steps + N_forecast + 1
        self.t_hours, self.outdoor, self.price = make_day_profiles(n_total, dt)
        self.t_lower, self.t_upper = make_comfort_schedule(n_total, dt)

        # What the MPC does not know: evening occupancy gains + AR(1) noise.
        rng = np.random.default_rng(seed)
        hour = self.t_hours % 24.0
        occupancy_gain = np.where((hour >= 18.0) & (hour < 22.0), gain_kw, 0.0)
        phi = 0.9
        noise = np.zeros(n_total)
        for k in range(1, n_total):
            noise[k] = phi * noise[k - 1] + noise_std * np.sqrt(1 - phi**2) * rng.normal()
        self.disturbance_kw = occupancy_gain + noise

        self.k = 0
        self.T = 19.0

    def _obs(self) -> dict:
        s = slice(self.k, self.k + self.N_forecast + 1)
        return {
            "T": self.T,
            "k": self.k,
            "outdoor": self.outdoor[s],
            "price": self.price[s],
            "t_lower": self.t_lower[s],
            "t_upper": self.t_upper[s],
        }

    def reset(self, start: int = 0, T0: float = 19.0) -> dict:
        """Start an episode at profile step ``start`` with room temperature ``T0``."""
        self.k = start
        self.T = T0
        return self._obs()

    def step(self, q: float) -> tuple[dict, float, bool, dict]:
        """Apply heating power ``q`` [kW] for one step of ``dt`` hours."""
        q = float(np.clip(q, 0.0, self.q_max))
        T_next = true_step(
            self.T, q, self.outdoor[self.k], gain=self.disturbance_kw[self.k],
            dt=self.dt, R=self.R_true, C=self.C_true,
        )
        # Comfort is judged where the OCP judges it: at the *next* step.
        lb, ub = self.t_lower[self.k + 1], self.t_upper[self.k + 1]
        cost = step_cost(
            np.float64(self.price[self.k]), q, np.float64(T_next), lb, ub,
            dt=self.dt, w_comfort=self.w_comfort,
        )
        energy_eur = self.price[self.k] * q * self.dt
        self.k += 1
        self.T = float(T_next)
        done = self.k >= self.n_steps
        info = {
            "energy_eur": float(energy_eur),
            "discomfort_kh": float((cost - energy_eur) / self.w_comfort),
            "T": self.T,
        }
        return self._obs(), -float(cost), done, info


def thermostat_expert(
    T: float,
    t_lower: float,
    outdoor_temp: float,
    pref_offset: float = 0.7,
    k_p: float = 3.0,
    R_belief: float = R_TRUE,
    q_max: float = 12.0,
) -> float:
    """A price-blind modulating thermostat — "the user of the house".

    The occupant likes it ``pref_offset`` K above the *scheduled* lower
    comfort bound and has lived here long enough to know the house: they set
    roughly the power that holds their preferred temperature against the
    current weather (feedforward through the **true** ``R``), nudged
    proportionally when the room feels too cold or too warm. Completely
    price-blind.
    """
    t_pref = t_lower + pref_offset
    q = (t_pref - outdoor_temp) / R_belief + k_p * (t_pref - T)
    return float(np.clip(q, 0.0, q_max))


def collect_dataset(env: HouseEnv, n_steps: int, start: int = 0, T0: float = 19.0) -> dict:
    """Roll the thermostat expert in ``env`` and record (observation, action) pairs.

    Returns a dict of stacked numpy arrays ready for batched planner solves:
    ``"T"`` ``(B,)``, the four forecast windows ``(B, N+1)`` and the expert's
    action ``"u_expert"`` ``(B,)``.
    """
    obs = env.reset(start=start, T0=T0)
    records: dict[str, list] = {key: [] for key in
                                ("T", "outdoor", "price", "t_lower", "t_upper", "u_expert")}
    for _ in range(n_steps):
        q = thermostat_expert(obs["T"], obs["t_lower"][0], obs["outdoor"][0])
        for key in ("T", "outdoor", "price", "t_lower", "t_upper"):
            records[key].append(obs[key])
        records["u_expert"].append(q)
        obs, _, done, _ = env.step(q)
        if done:
            break
    return {key: np.asarray(vals) for key, vals in records.items()}
