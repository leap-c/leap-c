"""Gymnasium environment for i4b building heat-pump control.

Wraps the i4b simulator (external/i4b/src/simulator.py) and exposes the same
building/HP model objects used in acados_ocp.py, ensuring the environment
dynamics and cost match the OCP exactly.

OCP parameter alignment (p per stage):
    p[0] = T_set_lower   [degC]
    p[1] = T_set_upper   [degC]
    p[2] = grid_signal   [-]
    p[3] = T_amb         [degC]
    p[4] = Qdot_gains    [W]

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

from i4b.disturbances import get_int_gains, get_solar_gains
from i4b.gym_interface import BUILDING_NAMES2CLASS
from i4b.gym_interface.constant import OBSERVATION_SPACE_LIMIT
from i4b.models.model_buildings import Building
from i4b.models.model_hvac import Heatpump, Heatpump_AW, Heatpump_Vitocal  # noqa: F401
from i4b.simulator import Model_simulator
from leap_c.examples.hvac.dataset import DataConfig, HvacDataset, load_and_prepare_data

_I4B_ROOT = Path(__file__).resolve().parents[3] / "external" / "i4b"

# Action bounds for T_HP supply temperature [degC], shared with the planner.
_T_HP_ACT_LOW: float = OBSERVATION_SPACE_LIMIT["T_hp_sup"][0]
_T_HP_ACT_HIGH: float = OBSERVATION_SPACE_LIMIT["T_hp_sup"][1]

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
    days: int = 3
    random_init: bool = False
    noise_level: float = 0.0
    T_set_lower: float = 20.0
    T_set_upper: float = 26.0
    N_forecast: int = 24 * 4
    grid_signal: float = 1.0
    gains_profile: str = _DEFAULT_GAINS_PROFILE
    apply_heating_logic: bool = False
    start_date: str | None = None
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
        dataset: HvacDataset | None = None,
    ):
        """Initialise the environment.

        Args:
            render_mode: Unused; kept for Gymnasium compatibility.
            cfg: Environment configuration. Uses default i4c /
                Heatpump_AW / 4R3C if None.
            dataset: Pre-built HvacDataset with weather and price data. When
                None a default dataset is loaded from local CSV assets.
        """
        super().__init__()

        if cfg is None:
            cfg = I4bEnvConfig(
                building_params=BUILDING_NAMES2CLASS["i4c"],
                hp_model=Heatpump_AW(mdot_HP=0.25),
            )

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

        # ── Dataset ───────────────────────────────────────────────────────────
        self.dataset = dataset if dataset is not None else self._create_default_dataset()
        self._augment_dataset()

        # ── Spaces ────────────────────────────────────────────────────────────
        self.state_keys = self.bldg_model.state_keys  # e.g. ("T_room","T_wall","T_hp_ret")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = self._make_obs_space()

        # ── Episode state ─────────────────────────────────────────────────────
        self._idx = 0  # current index into dataset
        self.step_counter = 0
        self.max_steps = 0
        self.state: dict | None = None

        self.reset()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _create_default_dataset(self) -> HvacDataset:
        """Load a default HvacDataset from local CSV assets."""
        data = load_and_prepare_data(
            price_zone="DE-LU",
            price_data_path=Path(__file__).parent / "assets" / "price.csv",
            weather_data_path=Path(__file__).parent / "assets" / "weather.csv",
        )
        return HvacDataset(data=data, cfg=DataConfig(valid_months=None))

    def _augment_dataset(self) -> None:
        """Compute building-specific disturbances and write them to the dataset.

        Adds ``Qdot_gains``, ``T_set_lower``, and ``T_set_upper`` columns to
        ``self.dataset`` using the full dataset index so that any episode start
        can be served from the pre-computed arrays.
        """
        cfg = self.cfg
        idx = self.dataset.data.index

        weather = self.dataset.data[
            [
                "temperature_2m",
                "diffuse_radiation",
                "shortwave_radiation",
                "direct_normal_irradiance",
            ]
        ].rename(
            columns={
                "temperature_2m": "T_amb",
                "diffuse_radiation": "dhi",
                "shortwave_radiation": "ghi",
                "direct_normal_irradiance": "dni",
            }
        )

        T_set_lower, T_set_upper = get_temperature_limits(idx)
        int_gains = get_int_gains(
            time=idx,
            profile_path=str(_I4B_ROOT / cfg.gains_profile),
            bldg_area=cfg.building_params["area_floor"],
        )
        Qdot_sol: pd.Series = get_solar_gains(weather=weather, bldg_params=cfg.building_params)

        self.dataset.add_columns(
            {
                "Qdot_gains": (Qdot_sol + int_gains["Qdot_tot"]).to_numpy(dtype=np.float32),
                "Qdot_int_oc": int_gains["Qdot_oc"].to_numpy(dtype=np.float32),
                "Qdot_int_app": int_gains["Qdot_app"].to_numpy(dtype=np.float32),
                "Qdot_int_tot": int_gains["Qdot_tot"].to_numpy(dtype=np.float32),
                "Qdot_sol": Qdot_sol.to_numpy(dtype=np.float32),
                "T_set_lower": T_set_lower.astype(np.float32),
                "T_set_upper": T_set_upper.astype(np.float32),
            }
        )

    def _make_obs_space(self) -> spaces.Dict:
        state_lows = np.array(
            [OBSERVATION_SPACE_LIMIT[k][0] for k in self.state_keys], dtype=np.float32
        )
        state_highs = np.array(
            [OBSERVATION_SPACE_LIMIT[k][1] for k in self.state_keys], dtype=np.float32
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
                "quarter_hour": spaces.Box(low=0, high=int(24 * 4 - 1), shape=(nf,), dtype=int),
                "day_of_year": spaces.Box(low=0, high=365, shape=(nf,), dtype=int),
                "day_of_week": spaces.Box(low=0, high=6, shape=(nf,), dtype=int),
                "dhi": spaces.Box(low=0, high=np.float32(2000.0), shape=(nf,), dtype=np.float32),
                "ghi": spaces.Box(low=0, high=np.float32(2000.0), shape=(nf,), dtype=np.float32),
                "dni": spaces.Box(low=0, high=np.float32(2000.0), shape=(nf,), dtype=np.float32),
                # price in €/kWh
                "price": spaces.Box(low=0, high=np.float32(10.0), shape=(nf,), dtype=np.float32),
            }
        )
        return spaces.Dict(obs_spaces)

    def _build_obs(self, state_dict: dict) -> dict:
        # All arrays in self.dataset._arrays are pre-cast to float32 at build
        # time so no dtype conversions are needed here.
        nf = self.cfg.N_forecast
        gcv = self.dataset.get_column_view  # zero-copy view, raises on out-of-bounds
        obs: dict = {
            "state": np.array([state_dict[k] for k in self.state_keys], dtype=np.float32),
            "disturbances": {
                "T_amb": gcv("temperature_2m", self._idx),
                "Qdot_gains": gcv("Qdot_gains", self._idx),
            },
            "setpoints": {
                "T_set_lower": gcv("T_set_lower", self._idx),
                "T_set_upper": gcv("T_set_upper", self._idx),
            },
        }
        if nf:
            obs["forecast"] = {
                "T_amb": gcv("temperature_2m", self._idx, nf),
                "T_set_lower": gcv("T_set_lower", self._idx, nf),
                "T_set_upper": gcv("T_set_upper", self._idx, nf),
                "quarter_hour": gcv("quarter_hour", self._idx, nf),
                "day_of_year": gcv("day_of_year", self._idx, nf),
                "day_of_week": gcv("day_of_week", self._idx, nf),
                "dhi": gcv("diffuse_radiation", self._idx, nf),
                "ghi": gcv("shortwave_radiation", self._idx, nf),
                "dni": gcv("direct_normal_irradiance", self._idx, nf),
                "price": gcv("price", self._idx, nf),
            }
        return obs

    def _copy_obs(self, obs: dict) -> dict:
        """Return a shallow-copied dict obs (arrays copied, nested dicts recursed)."""
        return {k: self._copy_obs(v) if isinstance(v, dict) else v.copy() for k, v in obs.items()}

    def _denorm_action(self, a: np.ndarray) -> float:
        """Map normalised action in [-1, 1] to T_HP in [degC]."""
        mid = (_T_HP_ACT_HIGH + _T_HP_ACT_LOW) / 2
        half = (_T_HP_ACT_HIGH - _T_HP_ACT_LOW) / 2
        return float(np.clip(a, -1.0, 1.0).flat[0] * half + mid)

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        """Advance one timestep.

        Args:
            action: Normalised supply temperature in [-1, 1].

        Returns:
            obs, reward, terminated, truncated, info
        """
        state_dict = dict(zip(self.state_keys, self.state["state"]))
        pk_dict = {
            "T_amb": float(self.dataset.get_column("temperature_2m", self._idx)),
            "Qdot_gains": float(self.dataset.get_column("Qdot_gains", self._idx)),
        }

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

        self._idx += 1
        self.step_counter += 1
        self.state = self._build_obs(next_state)

        reward = -E_el_kWh * self.cfg.grid_signal
        truncated = (
            self._idx + self.cfg.N_forecast >= len(self.dataset)
            or self.step_counter >= self.max_steps
        )

        info = {
            "cost": float(costs["dev_neg_max"]),
            "E_el_kWh": E_el_kWh,
            "dev_sum": float(costs["dev_neg_sum"]),
            "dev_max": float(costs["dev_neg_max"]),
            "T_room": float(next_state["T_room"]),
            "T_hp_sup": T_hp_sup,
            "t": self._idx,
        }

        obs = self._copy_obs(self.state)
        if self.cfg.noise_level > 0:
            obs["state"] += self.np_random.normal(
                0, self.cfg.noise_level, obs["state"].shape
            ).astype(np.float32)

        return obs, float(reward), False, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)

        steps_per_day = 24 * int(3600 / self.cfg.delta_t)
        # Cap max_steps so the forecast window never extends beyond the dataset.
        data_limit = len(self.dataset) - self.cfg.N_forecast - 1
        self.max_steps = min(
            self.cfg.days * steps_per_day if self.cfg.days is not None else data_limit,
            data_limit,
        )

        if self.cfg.random_init:
            max_start = max(0, len(self.dataset) - self.max_steps - self.cfg.N_forecast)
            self._idx = int(self.np_random.integers(0, max(1, max_start)))
        elif self.cfg.start_date is not None:
            self._idx = self.dataset.index.searchsorted(pd.Timestamp(self.cfg.start_date, tz="UTC"))
        else:
            self._idx = 0

        self.step_counter = 0

        if self.cfg.random_init:
            state_dict = {
                k: float(
                    self.np_random.uniform(
                        OBSERVATION_SPACE_LIMIT[k][0],
                        OBSERVATION_SPACE_LIMIT[k][1],
                    )
                )
                for k in self.state_keys
            }
        else:
            state_dict = {k: self.cfg.T_set_lower for k in self.state_keys}

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
