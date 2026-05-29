"""Bicycle-model parameter dataclass for the race-car environment."""

from dataclasses import dataclass, fields

import numpy as np


@dataclass(kw_only=True)
class RaceCarDynamicsParameters:
    """Bicycle-model dynamics parameters.

    Defaults match the acados race_cars example.

    Attributes:
        m: Mass [kg].
        C1: Cornering geometry factor (front/rear distribution).
        C2: Cornering geometry factor.
        Cm1: Motor coefficient (acceleration).
        Cm2: Motor coefficient (velocity drag).
        Cr0: Rolling resistance, constant term.
        Cr2: Rolling resistance, quadratic in v.
    """

    m: float = 0.043
    C1: float = 0.5
    C2: float = 15.5
    Cm1: float = 0.28
    Cm2: float = 0.05
    Cr0: float = 0.011
    Cr2: float = 0.006

    def randomize(
        self,
        rng: np.random.Generator,
        noise_scale: float = 0.3,
    ) -> "RaceCarDynamicsParameters":
        """Return a new instance with each field perturbed by Gaussian noise.

        Each field is sampled as ``rng.normal(loc=v, scale=noise_scale * |v|)``.
        Mirrors :py:meth:`leap_c.examples.hvac.dynamics.HydronicParameters.randomize`.
        """
        kwargs = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kwargs[f.name] = float(rng.normal(loc=v, scale=noise_scale * np.sqrt(v * v)))
        return RaceCarDynamicsParameters(**kwargs)

    def to_dict(self) -> dict[str, float]:
        """Return a plain dict view; consumed by the dict-based casadi factories."""
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}
