"""Gymnasium environment for i4b building heat-pump control.

Wraps the i4b simulator (external/i4b/src/simulator.py) and exposes the same
building/HP model objects used in acados_ocp.py, ensuring the environment
dynamics and cost match the OCP exactly.

OCP parameter alignment (p per stage):
    p[0] = T_amb         [degC]
    p[1] = Qdot_gains    [W]
    p[2] = T_set_lower   [degC]
    p[3] = T_set_upper   [degC]
    p[4] = grid_signal   [-]

Available building models (building_params dicts from i4b_data/buildings/):
    sfh_1919_1948 … sfh_2016_now  (0_soc, 1_enev, 2_kfw variants)
    i4c

Available building methods: "2R2C", "4R3C", "5R4C"
Available HP models: Heatpump_AW, Heatpump_Vitocal

Observation space (spaces.Dict):
    "state":          Box(nx,)   – building thermal states [degC]
    "disturbances":   Dict
        "T_amb":      Box(1,)    – current ambient temperature [degC]
        "Qdot_gains": Box(1,)    – current total heat gains [W]
    "setpoints":      Dict
        "T_set_lower": Box(1,)   – lower comfort bound [degC]
        "T_set_upper": Box(1,)   – upper comfort bound [degC]
    "forecast":       Dict       – only if weather_forecast_steps is non-empty
        "T_amb":      Box(nf,)   – ambient temperature forecast [degC]
"""

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from i4b.disturbances import get_int_gains, get_solar_gains, load_weather
from i4b.gym_interface import BUILDING_NAMES2CLASS
from i4b.gym_interface.constant import OBSERVATION_SPACE_LIMIT
from i4b.models.model_buildings import Building
from i4b.models.model_hvac import Heatpump, Heatpump_AW, Heatpump_Vitocal  # noqa: F401
from i4b.simulator import Model_simulator

_I4B_ROOT = Path(__file__).resolve().parents[3] / "external" / "i4b"

# Re-export for convenience
__all__ = [
    "I4bEnvConfig",
    "I4bEnv",
    "BUILDING_NAMES2CLASS",
    "Building",
    "Heatpump_AW",
    "Heatpump_Vitocal",
]

# Internal gains profile path relative to i4b root
_DEFAULT_GAINS_PROFILE = "i4b_data/profiles/InternalGains/ResidentialDetached.csv"


@dataclass(kw_only=True)
class I4bEnvConfig:
    """Configuration for I4bEnv.

    Attributes:
        building_params: Building parameter dict (e.g. sfh_2016_now_0_soc from
            BUILDING_NAMES2CLASS or imported directly from i4b_data/buildings/).
        hp_model: Instantiated heat pump model. Same object can be passed to
            export_parametric_ocp to guarantee dynamics/cost consistency.
        method: Building thermal model type. One of "2R2C", "4R3C", "5R4C".
        mdot_hp: Mass flow rate of the HP system [kg/s].
        delta_t: Simulation timestep [s].
        days: Episode length in days. None uses full weather dataset.
        random_init: Randomise initial state and start time on reset.
        noise_level: Std dev of Gaussian noise added to building state observations.
        T_set_lower: Default lower comfort temperature setpoint [degC]. Overridden
            by time-varying values from get_temperature_limits if using dynamic setpoints.
        T_set_upper: Default upper comfort temperature [degC].
        N_forecast: Number of future steps with available weather/setpoint forecasts
            in the "forecast" observation dict. Should match the MPC horizon for best results,
            but can be set to 0 to disable the "forecast" part of the observation.
        grid_signal: Grid support signal (OCP p[4]). Constant for now.
        gains_profile: Path to internal gains CSV relative to i4b root.
    """

    building_params: dict
    hp_model: Heatpump
    method: str = "4R3C"
    mdot_hp: float = 0.25
    delta_t: int = 900
    days: int | None = None
    random_init: bool = False
    noise_level: float = 0.0
    T_set_lower: float = 20.0
    T_set_upper: float = 26.0
    N_forecast: int = 24 * 4
    grid_signal: float = 1.0
    gains_profile: str = _DEFAULT_GAINS_PROFILE
    apply_heating_logic: bool = False
    """If True, apply the legacy RoomHeatEnv heating logic in step(): add T_offset when
    T_amb < T_amb_lim, fall back to T_hp_ret when warm, and clip via check_hp().
    If False (default), the denormalised MPC action is passed directly to the simulator,
    matching the OCP model."""


class I4bEnv(gym.Env):
    """Gymnasium environment for building heat-pump control using the i4b simulator.

    The observation is a ``spaces.Dict`` — see module docstring for layout.

    The action is a normalised supply temperature in [-1, 1], mapped linearly
    to T_HP in [0, 65] degC (matching the OCP control bounds).

    The reward is the negative electrical energy consumption scaled by the grid
    signal, i.e.  r = -E_el_kWh * grid_signal, which is proportional to the
    OCP stage cost Qth / (COP * 100) * grid_signal up to a positive constant.

    The method `get_ocp_parameters(t)` returns the OCP parameter vector
    p = [T_amb, Qdot_gains, T_set_lower, T_set_upper, grid_signal] at timestep t,
    matching the parameter layout expected by export_parametric_ocp.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        render_mode: str | None = None,
        cfg: I4bEnvConfig | None = None,
        days: int | None = None,
    ):
        """Initialise the environment.

        Args:
            render_mode: Unused; kept for Gymnasium compatibility.
            cfg: Environment configuration. Uses default i4c /
                Heatpump_AW / 4R3C if None.
            days: Episode length in days. Overrides ``cfg.days`` when provided.
                Convenient for use with ``create_env("i4b", days=N)``.
        """
        super().__init__()

        if cfg is None:
            cfg = I4bEnvConfig(
                building_params=BUILDING_NAMES2CLASS["i4c"],
                hp_model=Heatpump_AW(mdot_HP=0.25),
            )
        if days is not None:
            cfg.days = days
        self.cfg = cfg

        # ── Build models ──────────────────────────────────────────────────────
        self.bldg_model = Building(
            params=cfg.building_params,
            mdot_hp=cfg.mdot_hp,
            method=cfg.method,
            T_room_set_lower=cfg.T_set_lower,
            T_room_set_upper=cfg.T_set_upper,
        )
        self.hp_model = cfg.hp_model
        self.simulator = Model_simulator(
            hp_model=self.hp_model,
            bldg_model=self.bldg_model,
            timestep=cfg.delta_t,
        )

        # ── Load weather / disturbances ───────────────────────────────────────
        self._p: pd.DataFrame = self._load_disturbances()

        # ── Spaces ────────────────────────────────────────────────────────────
        self.obs_keys = self.bldg_model.state_keys  # e.g. ("T_room","T_wall","T_hp_ret")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._action_low = 0.0
        self._action_high = 65.0
        self.observation_space = self._make_obs_space()

        # ── Episode state ─────────────────────────────────────────────────────
        self._max_t = self._calc_max_t()
        self._t = 0  # index into weather data
        self._steps = 0  # steps in current episode
        self.state: dict | None = None

        self.reset()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_disturbances(self) -> pd.DataFrame:
        """Load weather and compute total gains; return resampled DataFrame."""
        cfg = self.cfg
        pos = cfg.building_params["position"]
        weather = load_weather(
            pos["lat"],
            pos["long"],
            pos["altitude"],
            tz=pos["timezone"],
            repo_filepath=str(_I4B_ROOT),
        )

        weather = weather.resample(f"{cfg.delta_t}s").interpolate()

        # TODO: Comfort bounds not really a disturbance. Refactor
        T_set_lower, T_set_upper = get_temperature_limits(weather.index)
        comfort_bounds = pd.DataFrame(
            {"T_set_lower": T_set_lower, "T_set_upper": T_set_upper}, index=weather.index
        )

        int_gains = get_int_gains(
            time=weather.index,
            profile_path=str(_I4B_ROOT / cfg.gains_profile),
            bldg_area=cfg.building_params["area_floor"],
        )
        Qdot_sol = get_solar_gains(weather=weather, bldg_params=cfg.building_params)
        Qdot_gains = pd.DataFrame(
            Qdot_sol + int_gains["Qdot_tot"],
            columns=["Qdot_gains"],
        )
        p = pd.concat(
            [weather["T_amb"], Qdot_gains, comfort_bounds],
            axis=1,
        ).astype(np.float32)

        return p.resample(f"{cfg.delta_t}s").interpolate()

    def _make_obs_space(self) -> spaces.Dict:
        state_lows = np.array(
            [OBSERVATION_SPACE_LIMIT[k][0] for k in self.obs_keys], dtype=np.float32
        )
        state_highs = np.array(
            [OBSERVATION_SPACE_LIMIT[k][1] for k in self.obs_keys], dtype=np.float32
        )
        T_amb_lo = np.float32(OBSERVATION_SPACE_LIMIT["T_amb"][0])
        T_amb_hi = np.float32(OBSERVATION_SPACE_LIMIT["T_amb"][1])
        Qdot_lo = np.float32(OBSERVATION_SPACE_LIMIT["Qdot_gains"][0])
        Qdot_hi = np.float32(OBSERVATION_SPACE_LIMIT["Qdot_gains"][1])

        obs_spaces: dict = {
            "state": spaces.Box(low=state_lows, high=state_highs, dtype=np.float32),
            "disturbances": spaces.Dict(
                {
                    "T_amb": spaces.Box(low=T_amb_lo, high=T_amb_hi, shape=(1,), dtype=np.float32),
                    "Qdot_gains": spaces.Box(
                        low=Qdot_lo, high=Qdot_hi, shape=(1,), dtype=np.float32
                    ),
                }
            ),
            "setpoints": spaces.Dict(
                {
                    "T_set_lower": spaces.Box(
                        low=np.float32(15.0), high=np.float32(30.0), shape=(1,), dtype=np.float32
                    ),
                    "T_set_upper": spaces.Box(
                        low=np.float32(20.0), high=np.float32(35.0), shape=(1,), dtype=np.float32
                    ),
                }
            ),
        }
        nf = self.cfg.N_forecast
        obs_spaces["forecast"] = spaces.Dict(
            {
                "T_amb": spaces.Box(low=T_amb_lo, high=T_amb_hi, shape=(nf,), dtype=np.float32),
                "T_set_lower": spaces.Box(
                    low=np.float32(15.0), high=np.float32(30.0), shape=(nf,), dtype=np.float32
                ),
                "T_set_upper": spaces.Box(
                    low=np.float32(20.0), high=np.float32(35.0), shape=(nf,), dtype=np.float32
                ),
            }
        )
        return spaces.Dict(obs_spaces)

    def _calc_max_t(self) -> int:
        if self.cfg.days is None:
            return len(self._p) - 1
        steps_per_day = 24 * int(3600 / self.cfg.delta_t)
        max_t = self.cfg.days * steps_per_day
        if max_t >= len(self._p):
            raise ValueError(
                f"Episode length ({max_t} steps) exceeds available data ({len(self._p)} steps)."
            )
        return max_t

    def _build_obs(self, state_dict: dict) -> dict:
        row = self._p.iloc[self._t]
        obs: dict = {
            "state": np.array([state_dict[k] for k in self.obs_keys], dtype=np.float32),
            "disturbances": {
                "T_amb": np.array([float(row["T_amb"])], dtype=np.float32),
                "Qdot_gains": np.array([float(row["Qdot_gains"])], dtype=np.float32),
            },
            "setpoints": {
                "T_set_lower": np.array([float(row["T_set_lower"])], dtype=np.float32),
                "T_set_upper": np.array([float(row["T_set_upper"])], dtype=np.float32),
            },
        }
        if self.cfg.N_forecast:
            obs["forecast"] = {
                "T_amb": np.array(
                    [
                        float(self._p.iloc[min(self._t + i, len(self._p) - 1)]["T_amb"])
                        for i in range(self.cfg.N_forecast)
                    ],
                    dtype=np.float32,
                ),
                "T_set_lower": np.array(
                    [
                        float(self._p.iloc[min(self._t + i, len(self._p) - 1)]["T_set_lower"])
                        for i in range(self.cfg.N_forecast)
                    ],
                    dtype=np.float32,
                ),
                "T_set_upper": np.array(
                    [
                        float(self._p.iloc[min(self._t + i, len(self._p) - 1)]["T_set_upper"])
                        for i in range(self.cfg.N_forecast)
                    ],
                    dtype=np.float32,
                ),
            }
        return obs

    def _copy_obs(self, obs: dict) -> dict:
        """Return a shallow-copied dict obs (arrays copied, nested dicts recursed)."""
        return {k: self._copy_obs(v) if isinstance(v, dict) else v.copy() for k, v in obs.items()}

    def _denorm_action(self, a: np.ndarray) -> float:
        """Map normalised action in [-1, 1] to T_HP in [0, 65] degC."""
        mid = (self._action_high + self._action_low) / 2
        half = (self._action_high - self._action_low) / 2
        return float(np.clip(a, -1.0, 1.0).flat[0] * half + mid)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_ocp_parameters(self, t: int) -> np.ndarray:
        """Return OCP parameter vector p.

        p = [T_amb, Qdot_gains, T_set_lower, T_set_upper, grid_signal].

        This matches the parameter layout in export_parametric_ocp exactly.

        Args:
            t: Timestep index into the weather data.

        Returns:
            Array of shape (5,).
        """
        row = self._p.iloc[t]
        return np.array(
            [
                float(row["T_amb"]),
                float(row["Qdot_gains"]),
                float(row["T_set_lower"]),
                float(row["T_set_upper"]),
                self.cfg.grid_signal,
            ],
            dtype=np.float64,
        )

    def get_ocp_parameters_horizon(self, t: int, N: int) -> np.ndarray:
        """Return OCP parameters for stages t … t+N (inclusive), shape (N+1, 5).

        Useful to fill all shooting nodes before calling the acados solver.
        """
        n_avail = len(self._p)
        rows = np.stack([self.get_ocp_parameters(min(t + k, n_avail - 1)) for k in range(N + 1)])
        return rows

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        """Advance one timestep.

        Args:
            action: Normalised supply temperature in [-1, 1].

        Returns:
            obs, reward, terminated, truncated, info
        """
        state_dict = dict(zip(self.obs_keys, self.state["state"]))
        pk_dict = self._p.iloc[self._t].to_dict()

        T_hp_sup = self._denorm_action(action)

        if self.cfg.apply_heating_logic:
            # Legacy RoomHeatEnv behaviour: add T_offset when cold outside, fall back to
            # T_hp_ret when ambient is warm, then clip via check_hp.
            if pk_dict["T_amb"] < self.bldg_model.params["T_amb_lim"]:
                T_hp_sup = max(
                    T_hp_sup + self.bldg_model.params["T_offset"],
                    state_dict["T_hp_ret"],
                )
            else:
                T_hp_sup = state_dict["T_hp_ret"]
            T_hp_sup = self.hp_model.check_hp(T_hp_sup, state_dict["T_hp_ret"])

        res = self.simulator.get_next_state(state_dict, T_hp_sup, pk_dict)
        next_state = res["state"]
        costs = res["cost"]
        E_el_kWh = float(costs["E_el"]) / 1000.0

        self._t += 1
        self._steps += 1
        self.state = self._build_obs(next_state)

        reward = -E_el_kWh * self.cfg.grid_signal
        truncated = self._t >= len(self._p) - 1 or self._steps >= self._max_t

        info = {
            "cost": float(costs["dev_neg_max"]),
            "E_el_kWh": E_el_kWh,
            "dev_sum": float(costs["dev_neg_sum"]),
            "dev_max": float(costs["dev_neg_max"]),
            "T_room": float(next_state["T_room"]),
            "T_hp_sup": T_hp_sup,
            "t": self._t,
            # OCP parameter vector at the new timestep (ready for MPC warm-start)
            "p_ocp": self.get_ocp_parameters(min(self._t, len(self._p) - 1)),
        }

        obs = self._copy_obs(self.state)
        if self.cfg.noise_level > 0:
            obs["state"] += self.np_random.normal(
                0, self.cfg.noise_level, obs["state"].shape
            ).astype(np.float32)

        return obs, float(reward), False, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)

        if self.cfg.random_init:
            max_fc = max(self.cfg.weather_forecast_steps) if self.cfg.weather_forecast_steps else 0
            max_start = len(self._p) - self._max_t - 1 - max_fc
            self._t = int(self.np_random.integers(0, max(1, max_start)))
        else:
            self._t = 0

        self._steps = 0

        if self.cfg.random_init:
            state_dict = {
                k: float(
                    self.np_random.uniform(
                        OBSERVATION_SPACE_LIMIT[k][0],
                        OBSERVATION_SPACE_LIMIT[k][1],
                    )
                )
                for k in self.obs_keys
            }
        else:
            state_dict = {k: self.cfg.T_set_lower for k in self.obs_keys}

        self.state = self._build_obs(state_dict)
        return self._copy_obs(self.state), {}


def get_temperature_limits(
    time: pd.DatetimeIndex,
    night_start_hour: int = 22,
    night_end_hour: int = 8,
    lb_night: float = 12.0,
    lb_day: float = 17.0,
    ub_night: float = 21.0,
    ub_day: float = 21.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Get temperature limits based on the time of day."""
    hours = time.hour
    night_idx = (hours >= night_start_hour) | (hours < night_end_hour)
    lb = np.where(night_idx, lb_night, lb_day)
    ub = np.where(night_idx, ub_night, ub_day)
    return lb, ub
