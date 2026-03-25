"""Gymnasium environment for i4b building heat-pump control.

Wraps the i4b simulator (external/i4b/src/simulator.py) and exposes the same
building/HP model objects used in acados_ocp.py, ensuring the environment
dynamics and cost match the OCP exactly.

OCP parameter alignment (p per stage):
    p[0] = T_amb         [degC]
    p[1] = Qdot_gains    [W]
    p[2] = T_set_lower   [degC]
    p[3] = grid_signal   [-]

Available building models (building_params dicts from data/buildings/):
    sfh_1919_1948 … sfh_2016_now  (0_soc, 1_enev, 2_kfw variants)
    i4c

Available building methods: "2R2C", "4R3C", "5R4C"
Available HP models: Heatpump_AW, Heatpump_Vitocal
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

_I4B_ROOT = Path(__file__).resolve().parents[3] / "external" / "i4b"
if str(_I4B_ROOT) not in sys.path:
    sys.path.insert(0, str(_I4B_ROOT))

from src.disturbances import get_int_gains, get_solar_gains, load_weather  # noqa: E402
from src.gym_interface import BUILDING_NAMES2CLASS  # noqa: E402
from src.gym_interface.constant import OBSERVATION_SPACE_LIMIT  # noqa: E402
from src.models.model_buildings import Building  # noqa: E402
from src.models.model_hvac import Heatpump, Heatpump_AW, Heatpump_Vitocal  # noqa: E402, F401
from src.simulator import Model_simulator  # noqa: E402

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
_DEFAULT_GAINS_PROFILE = "data/profiles/InternalGains/ResidentialDetached.csv"


@dataclass(kw_only=True)
class I4bEnvConfig:
    """Configuration for I4bEnv.

    Attributes:
        building_params: Building parameter dict (e.g. sfh_2016_now_0_soc from
            BUILDING_NAMES2CLASS or imported directly from data/buildings/).
        hp_model: Instantiated heat pump model. Same object can be passed to
            export_parametric_ocp to guarantee dynamics/cost consistency.
        method: Building thermal model type. One of "2R2C", "4R3C", "5R4C".
        mdot_hp: Mass flow rate of the HP system [kg/s].
        delta_t: Simulation timestep [s].
        days: Episode length in days. None uses full weather dataset.
        random_init: Randomise initial state and start time on reset.
        noise_level: Std dev of Gaussian noise added to observations.
        T_set_lower: Lower comfort temperature setpoint [degC] (OCP p[2]).
        T_set_upper: Upper comfort temperature [degC].
        weather_forecast_steps: Steps ahead (multiples of delta_t) to include
            as ambient temperature forecasts in the observation.
        grid_signal: Grid support signal (OCP p[3]). Constant for now.
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
    weather_forecast_steps: List[int] = field(default_factory=list)
    grid_signal: float = 1.0
    gains_profile: str = _DEFAULT_GAINS_PROFILE


class I4bEnv(gym.Env):
    """Gymnasium environment for building heat-pump control using the i4b simulator.

    The observation vector is:
        [*building_states, T_amb, Qdot_gains, *T_amb_forecast]

    The action is a normalised supply temperature in [-1, 1], mapped linearly
    to T_HP in [0, 65] degC (matching the OCP control bounds).

    The reward is the negative electrical energy consumption scaled by the grid
    signal, i.e.  r = - E_el_kWh * grid_signal, which is proportional to the
    OCP stage cost Qth / (COP * 100) * grid_signal up to a positive constant.

    The method `get_ocp_parameters(t)` returns the OCP parameter vector
    p = [T_amb, Qdot_gains, T_set_lower, grid_signal] at timestep t, matching
    the parameter layout expected by export_parametric_ocp.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        render_mode: str | None = None,
        cfg: I4bEnvConfig | None = None,
    ):
        """Initialise the environment.

        Args:
            render_mode: Unused; kept for Gymnasium compatibility.
            cfg: Environment configuration. Uses default sfh_2016_now_0_soc /
                Heatpump_AW / 4R3C if None.
        """
        super().__init__()

        if cfg is None:
            cfg = I4bEnvConfig(
                building_params=BUILDING_NAMES2CLASS["sfh_2016_now_0_soc"],
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
        self.state: np.ndarray | None = None

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
        p = pd.concat([weather["T_amb"], Qdot_gains], axis=1).astype(np.float32)
        return p.resample(f"{cfg.delta_t}s").ffill()

    def _make_obs_space(self) -> spaces.Box:
        lows, highs = [], []
        for key in self.obs_keys:
            lo, hi = OBSERVATION_SPACE_LIMIT[key]
            lows.append(lo)
            highs.append(hi)
        # Current disturbances
        lows += [OBSERVATION_SPACE_LIMIT["T_amb"][0], OBSERVATION_SPACE_LIMIT["Qdot_gains"][0]]
        highs += [OBSERVATION_SPACE_LIMIT["T_amb"][1], OBSERVATION_SPACE_LIMIT["Qdot_gains"][1]]
        # Forecast
        for _ in self.cfg.weather_forecast_steps:
            lows.append(OBSERVATION_SPACE_LIMIT["T_amb"][0])
            highs.append(OBSERVATION_SPACE_LIMIT["T_amb"][1])
        return spaces.Box(
            low=np.array(lows, dtype=np.float32),
            high=np.array(highs, dtype=np.float32),
            dtype=np.float32,
        )

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

    def _build_obs(self, state_dict: dict) -> np.ndarray:
        obs = [state_dict[k] for k in self.obs_keys]
        pk = self.get_ocp_parameters(self._t)
        obs += [pk[0], pk[1]]  # T_amb, Qdot_gains
        for i in self.cfg.weather_forecast_steps:
            t_fc = min(self._t + i, len(self._p) - 1)
            obs.append(float(self._p.iloc[t_fc]["T_amb"]))
        return np.array(obs, dtype=np.float32)

    def _denorm_action(self, a: np.ndarray) -> float:
        """Map normalised action in [-1, 1] to T_HP in [0, 65] degC."""
        mid = (self._action_high + self._action_low) / 2
        half = (self._action_high - self._action_low) / 2
        return float(np.clip(a, -1.0, 1.0).flat[0] * half + mid)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_ocp_parameters(self, t: int) -> np.ndarray:
        """Return OCP parameter vector p = [T_amb, Qdot_gains, T_set_lower, grid_signal].

        This matches the parameter layout in export_parametric_ocp exactly.

        Args:
            t: Timestep index into the weather data.

        Returns:
            Array of shape (4,).
        """
        row = self._p.iloc[t]
        return np.array(
            [
                float(row["T_amb"]),
                float(row["Qdot_gains"]),
                self.cfg.T_set_lower,
                self.cfg.grid_signal,
            ],
            dtype=np.float64,
        )

    def get_ocp_parameters_horizon(self, t: int, N: int) -> np.ndarray:
        """Return OCP parameters for stages t … t+N (inclusive), shape (N+1, 4).

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
        state_dict = dict(zip(self.obs_keys, self.state[: len(self.obs_keys)]))
        pk_dict = self._p.iloc[self._t].to_dict()

        T_hp_sup = self._denorm_action(action)

        # Apply heating logic from RoomHeatEnv: if outside is warm enough, no heating
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

        obs = self.state.copy()
        if self.cfg.noise_level > 0:
            obs += self.np_random.normal(0, self.cfg.noise_level, obs.shape).astype(np.float32)

        return obs, float(reward), False, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
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
        return self.state.copy(), {}
