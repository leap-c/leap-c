from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class BestestParameters:
    """Base class for hydronic system parameters.

    Attributes:
        gAw: Effective window area [m²]
        Ch: Heating system thermal capacity [J/K]
        Ci: Indoor thermal capacity [J/K]
        Ce: External thermal capacity [J/K]
        e11: Measurement noise
        sigmai: Indoor temperature process noise
        sigmah: Heating system process noise
        sigmae: External temperature process noise
        Rea: Resistance external-ambient [K/W]
        Rhi: Resistance heating-indoor [K/W]
        Rie: Resistance indoor-external [K/W]
        eta: Efficiency for electric heater
    """

    gAw: float  # noqa: N815

    # Thermal capacitances [J/K]
    Ch: float
    Ci: float
    Ce: float

    # Noise parameters
    e11: float
    sigmai: float
    sigmah: float
    sigmae: float

    # Thermal resistances [K/W]
    Rea: float
    Rhi: float
    Rie: float

    # Heater parameters
    eta: float

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
