from dataclasses import asdict, dataclass

import numpy as np
from scipy.constants import convert_temperature

from leap_c.ocp.acados.parameters import Parameter


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


def make_default_hvac_params(stagewise: bool = False) -> tuple[Parameter, ...]:
    """Return a tuple of default parameters for the hvac problem."""
    hydronic_params = BestestHydronicParameters().to_dict()

    # NOTE: Only include parameters that are relevant for the parametric OCP.
    params = [
        Parameter(
            name=k,
            value=np.array([v]),
            lower_bound=0.95 * np.array([v]),
            upper_bound=1.05 * np.array([v]),
            fix=True,
            differentiable=False,
            stagewise=False,
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
            Parameter(
                name="Ta",  # Ambient temperature in Kelvin
                value=np.array([convert_temperature(20.0, "celsius", "kelvin")]),
                lower_bound=np.array([convert_temperature(-20.0, "celsius", "kelvin")]),
                upper_bound=np.array([convert_temperature(40.0, "celsius", "kelvin")]),
                fix=False,
                differentiable=True,
                stagewise=stagewise,
            ),
            Parameter(
                name="Phi_s",
                value=np.array([200.0]),  # Solar radiation in W/m²
                lower_bound=np.array([0.0]),
                upper_bound=np.array([400.0]),
                fix=False,
                differentiable=True,
                stagewise=stagewise,
            ),
            Parameter(
                name="price",
                value=np.array([0.15]),  # Electricity price in €/kWh
                lower_bound=np.array([0.00]),
                upper_bound=np.array([0.30]),
                fix=False,
                differentiable=True,
                stagewise=stagewise,
            ),
        ]
    )

    # Comfort constraints for indoor temperature
    params.extend(
        [
            Parameter(
                name="lb_Ti",
                value=np.array([convert_temperature(17.0, "celsius", "kelvin")]),
                lower_bound=np.array([convert_temperature(15.0, "celsius", "kelvin")]),
                upper_bound=np.array([convert_temperature(19.0, "celsius", "kelvin")]),
                fix=False,
                differentiable=False,
                stagewise=stagewise,
            ),
            Parameter(
                name="ub_Ti",
                value=np.array([convert_temperature(23.0, "celsius", "kelvin")]),
                lower_bound=np.array([convert_temperature(21.0, "celsius", "kelvin")]),
                upper_bound=np.array([convert_temperature(25.0, "celsius", "kelvin")]),
                fix=False,
                differentiable=False,
                stagewise=stagewise,
            ),
            Parameter(
                name="ref_Ti",
                value=np.array([convert_temperature(21.0, "celsius", "kelvin")]),
                lower_bound=np.array([convert_temperature(10.0, "celsius", "kelvin")]),
                upper_bound=np.array([convert_temperature(30.0, "celsius", "kelvin")]),
                fix=False,
                differentiable=True,
                stagewise=stagewise,
            ),
        ]
    )

    params.extend(
        [
            Parameter(
                name="q_Ti",
                value=np.array([0.001]),  # weight on rate of change of heater power
                lower_bound=np.array([0.0001]),
                upper_bound=np.array([0.001]),
                fix=False,
                differentiable=True,
                stagewise=stagewise,
            ),
            Parameter(
                name="q_dqh",
                value=np.array([1.0]),  # weight on rate of change of heater power
                lower_bound=np.array([0.5]),
                upper_bound=np.array([1.5]),
                fix=False,
                differentiable=True,
                stagewise=stagewise,
            ),
            Parameter(
                name="q_ddqh",
                value=np.array([1.0]),  # weight for acceleration of heater power
                lower_bound=np.array([0.5]),
                upper_bound=np.array([1.5]),
                fix=False,
                differentiable=True,
                stagewise=stagewise,
            ),
        ]
    )

    return tuple(params)
