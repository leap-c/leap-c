from dataclasses import asdict, dataclass

import numpy as np
import gymnasium as gym
from scipy.constants import convert_temperature

from leap_c.ocp.acados.parameters import AcadosParameter


@dataclass
class BestestParameters:
    """Base class for hydronic system parameters."""

    # Effective window area [m²]
    gAw: float  # noqa: N815

    # Thermal capacitances [J/K]
    Ch: float  # Heating system thermal capacity
    Ci: float  # Indoor thermal capacity
    Ce: float  # External thermal capacity

    # Noise parameters
    e11: float  # Measurement noise
    sigmai: float
    sigmah: float
    sigmae: float

    # Thermal resistances [K/W]
    Rea: float  # Resistance external-ambient
    Rhi: float  # Resistance heating-indoor
    Rie: float  # Resistance indoor-external

    # Heater parameters
    eta: float  # Efficiency for electric heater

    def to_dict(self) -> dict[str, float]:
        """Convert parameters to a dictionary with string keys and float values."""
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, params_dict: dict[str, float]) -> "BestestParameters":
        """Create an instance from a dictionary."""
        return cls(**params_dict)


@dataclass
class BestestHydronicParameters(BestestParameters):
    """Standard hydronic system parameters."""

    gAw: float = 10.1265729225269  # noqa: N815
    Ch: float = 4015.39425109821
    Ci: float = 1914908.30860716
    Ce: float = 15545663.6743828
    e11: float = -9.49409438095981
    sigmai: float = -37.8538482163307
    sigmah: float = -50.4867241844347
    sigmae: float = -5.57887704511886
    Rea: float = 0.00751396226986365
    Rhi: float = 0.0761996125919563
    Rie: float = 0.00135151763922409
    eta: float = 0.98


@dataclass
class BestestHydronicHeatpumpParameters(BestestParameters):
    """Heat pump system parameters for a hydronic heating system."""

    gAw: float = 40.344131392192  # noqa: N815
    Ch: float = 10447262.2318648
    Ci: float = 14827137.0377258
    Ce: float = 50508258.9032192
    e11: float = -30.0936560706053
    sigmai: float = -23.3175423490014
    sigmah: float = -19.5274067368137
    sigmae: float = -5.07591222090641
    Rea: float = 0.00163027389197229
    Rhi: float = 0.000437603769897038
    Rie: float = 0.000855786902577802
    eta: float = 0.98


def make_default_hvac_params(stagewise: bool = False) -> tuple[AcadosParameter, ...]:
    """Return a tuple of default parameters for the hvac problem."""
    hydronic_params = BestestHydronicParameters().to_dict()

    # NOTE: Only include parameters that are relevant for the parametric OCP.
    params = [
        AcadosParameter(
            name=k,
            default=np.array([v]),
            space=gym.spaces.Box(low=0.95 * np.array([v]), high=1.05 * np.array([v])),
            interface="fix",
        )
        for k, v in hydronic_params.items()
        if k
        in [
            "gAw",  # Effective window area
            "Ch",  # Heating system thermal capacity
            "Ci",  # Indoor thermal capacity
            "Ce",  # External thermal capacity
            "Rea",  # Resistance external-ambient
            "Rhi",  # Resistance heating-indoor
            "Rie",  # Resistance indoor-external]
            "eta",  # Efficiency for electric heater
        ]
    ]

    params.extend(
        [
            AcadosParameter(
                name="Ta",  # Ambient temperature in Kelvin
                default=np.array([convert_temperature(20.0, "celsius", "kelvin")]),
                space=gym.spaces.Box(
                    low=np.array([convert_temperature(-20.0, "celsius", "kelvin")]),
                    high=np.array([convert_temperature(40.0, "celsius", "kelvin")])
                ),
                interface="learnable",
                vary_stages=list(range(48)) if stagewise else [],
            ),
            AcadosParameter(
                name="Phi_s",
                default=np.array([200.0]),  # Solar radiation in W/m²
                space=gym.spaces.Box(low=np.array([0.0]), high=np.array([400.0])),
                interface="learnable",
                vary_stages=list(range(48)) if stagewise else [],
            ),
            AcadosParameter(
                name="price",
                default=np.array([0.15]),  # Electricity price in €/kWh
                space=gym.spaces.Box(low=np.array([0.00]), high=np.array([0.30])),
                interface="learnable",
                vary_stages=list(range(48)) if stagewise else [],
            ),
        ]
    )

    # Comfort constraints for indoor temperature
    params.extend(
        [
            AcadosParameter(
                name="lb_Ti",
                default=np.array([convert_temperature(17.0, "celsius", "kelvin")]),
                space=gym.spaces.Box(
                    low=np.array([convert_temperature(15.0, "celsius", "kelvin")]),
                    high=np.array([convert_temperature(19.0, "celsius", "kelvin")])
                ),
                interface="non-learnable",
                vary_stages=list(range(48)) if stagewise else [],
            ),
            AcadosParameter(
                name="ub_Ti",
                default=np.array([convert_temperature(23.0, "celsius", "kelvin")]),
                space=gym.spaces.Box(
                    low=np.array([convert_temperature(21.0, "celsius", "kelvin")]),
                    high=np.array([convert_temperature(25.0, "celsius", "kelvin")])
                ),
                interface="non-learnable",
                vary_stages=list(range(48)) if stagewise else [],
            ),
            AcadosParameter(
                name="ref_Ti",
                default=np.array([convert_temperature(21.0, "celsius", "kelvin")]),
                space=gym.spaces.Box(
                    low=np.array([convert_temperature(10.0, "celsius", "kelvin")]),
                    high=np.array([convert_temperature(30.0, "celsius", "kelvin")])
                ),
                interface="learnable",
                vary_stages=list(range(48)) if stagewise else [],
            ),
        ]
    )

    params.extend(
        [
            AcadosParameter(
                name="q_Ti",
                default=np.array([0.001]),  # weight on rate of change of heater power
                space=gym.spaces.Box(low=np.array([0.0001]), high=np.array([0.001])),
                interface="learnable",
                vary_stages=list(range(48)) if stagewise else [],
            ),
            AcadosParameter(
                name="q_dqh",
                default=np.array([1.0]),  # weight on rate of change of heater power
                space=gym.spaces.Box(low=np.array([0.5]), high=np.array([1.5])),
                interface="learnable",
                vary_stages=list(range(48)) if stagewise else [],
            ),
            AcadosParameter(
                name="q_ddqh",
                default=np.array([1.0]),  # weight for acceleration of heater power
                space=gym.spaces.Box(low=np.array([0.5]), high=np.array([1.5])),
                interface="learnable",
                vary_stages=list(range(48)) if stagewise else [],
            ),
        ]
    )

    return tuple(params)
