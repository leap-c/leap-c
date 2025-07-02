from dataclasses import asdict, dataclass

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


def make_default_hvac_params() -> tuple[Parameter, ...]:
    """Return a tuple of default parameters for the hvac problem."""
    nominal_params = BestestHydronicParameters()

    return (
        Parameter(
            name="gAw",
            value=nominal_params.gAw,
            lower_bound=0.95 * nominal_params.gAw,
            upper_bound=1.05 * nominal_params.gAw,
            fix=False,
            differentiable=True,
            stagewise=False,
        ),
        Parameter(
            name="Ch",
            value=nominal_params.Ch,
            lower_bound=0.95 * nominal_params.Ch,
            upper_bound=1.05 * nominal_params.Ch,
            fix=False,
            differentiable=True,
            stagewise=False,
        ),
        Parameter(
            name="Ci",
            value=nominal_params.Ci,
            lower_bound=0.95 * nominal_params.Ci,
            upper_bound=1.05 * nominal_params.Ci,
            fix=False,
            differentiable=True,
            stagewise=False,
        ),
        Parameter(
            name="Ce",
            value=nominal_params.Ce,
            lower_bound=0.95 * nominal_params.Ce,
            upper_bound=1.05 * nominal_params.Ce,
            fix=False,
            differentiable=True,
            stagewise=False,
        ),
        Parameter(
            name="e11",
            value=nominal_params.e11,
            lower_bound=0.95 * nominal_params.e11,
            upper_bound=1.05 * nominal_params.e11,
            fix=False,
            differentiable=True,
            stagewise=False,
        ),
        Parameter(
            name="sigmai",
            value=nominal_params.sigmai,
            lower_bound=0.95 * nominal_params.sigmai,
            upper_bound=1.05 * nominal_params.sigmai,
            fix=False,
            differentiable=True,
            stagewise=False,
        ),
        Parameter(
            name="sigmah",
            value=nominal_params.sigmah,
            lower_bound=0.95 * nominal_params.sigmah,
            upper_bound=1.05 * nominal_params.sigmah,
            fix=False,
            differentiable=True,
            stagewise=False,
        ),
        Parameter(
            name="sigmae",
            value=nominal_params.sigmae,
            lower_bound=0.95 * nominal_params.sigmae,
            upper_bound=1.05 * nominal_params.sigmae,
            fix=False,
            differentiable=True,
            stagewise=False,
        ),
        Parameter(
            name="Rea",
            value=nominal_params.Rea,
            lower_bound=0.95 * nominal_params.Rea,
            upper_bound=1.05 * nominal_params.Rea,
            fix=False,
            differentiable=True,
            stagewise=False,
        ),
        Parameter(
            name="Rhi",
            value=nominal_params.Rhi,
            lower_bound=0.95 * nominal_params.Rhi,
            upper_bound=1.05 * nominal_params.Rhi,
            fix=False,
            differentiable=True,
            stagewise=False,
        ),
        Parameter(
            name="Rie",
            value=nominal_params.Rie,
            lower_bound=0.95 * nominal_params.Rie,
            upper_bound=1.05 * nominal_params.Rie,
            fix=False,
            differentiable=True,
            stagewise=False,
        ),
        Parameter(
            name="eta",
            value=nominal_params.eta,
            lower_bound=0.95 * nominal_params.eta,
            upper_bound=max(
                1.0, 1.05 * nominal_params.eta
            ),  # efficiency must be <= 1.0
            fix=False,
            differentiable=True,
            stagewise=False,
        ),
    )
