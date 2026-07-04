"""The heating planner: a comfort-band MPC as an observation-to-action policy.

NOTE: keep in sync with the inline class taught in
``notebooks/getting_started/06_planner_interface.py`` (pedagogical inline
copy).
"""

import numpy as np
import torch

from leap_c.torch import AcadosDiffMpcTorch

from .heating import build_heating_ocp_comfort_band


class HeatingPlanner(torch.nn.Module):
    """Wraps the comfort-band heating MPC into ``forward(obs) -> action``.

    The three differentiable OCP parameters are exposed as ``nn.Parameter``s,
    so a plain torch optimizer can train them (notebooks 07 and 08):

    - ``R``: the planner's belief about the envelope [K/kW],
    - ``price_weight``: how much the occupant cares about the energy bill,
    - ``comfort_margin``: how far above the scheduled lower bound they live.

    ``forward`` takes a batched observation dict (the layout produced by
    ``HouseEnv``/``collect_dataset``) and returns the first heating decision
    ``u0`` of shape ``(B, 1)``, differentiable with respect to the parameters.
    """

    def __init__(
        self,
        N_horizon: int = 32,
        dt: float = 0.25,
        q_max: float = 12.0,
        name: str = "heating_planner",
        n_batch_init: int = 256,
        discount_factor: float | None = None,
    ):
        super().__init__()
        ocp, manager = build_heating_ocp_comfort_band(
            N_horizon, dt=dt, q_max=q_max, name=name
        )
        self.diff_mpc = AcadosDiffMpcTorch(
            ocp,
            manager,
            dtype=torch.float64,
            n_batch_init=n_batch_init,
            discount_factor=discount_factor,
            verbose=False,
        )
        self.R = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        self.price_weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        self.comfort_margin = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
        self.ctx = None  # warm start carried across closed-loop steps

    def forward(self, obs: dict, ctx=None):
        """Solve the MPC for a batch of observations.

        Args:
            obs: Dict with ``"T"`` ``(B,)`` and forecast windows ``"outdoor"``,
                ``"price"``, ``"t_lower"``, ``"t_upper"`` of shape ``(B, N+1)``
                (numpy arrays; ``"T"`` may be a torch tensor, whose autograd
                graph is preserved — the RL notebook differentiates through it).
            ctx: Optional context of a previous solve, used as warm start.

        Returns:
            ``(u0, ctx, (x, u, value))`` — the first heating decision
            ``(B, 1)``, the new solver context, and the full solution.
        """
        x0 = torch.as_tensor(obs["T"], dtype=torch.float64).reshape(-1, 1)
        batch_size = x0.shape[0]
        params = {
            # Forecast windows: non-differentiable stagewise, (B, N+1, 1) numpy.
            "outdoor_temp": np.asarray(obs["outdoor"], dtype=np.float64)[..., None],
            "price": np.asarray(obs["price"], dtype=np.float64)[..., None],
            "t_lower": np.asarray(obs["t_lower"], dtype=np.float64)[..., None],
            "t_upper": np.asarray(obs["t_upper"], dtype=np.float64)[..., None],
            # Learnable knobs: one shared value, expanded over the batch
            # (backward then sums the per-sample gradients — exactly the
            # dataset gradient a training loop wants).
            "R": self.R.expand(batch_size, 1),
            "price_weight": self.price_weight.expand(batch_size, 1),
            "comfort_margin": self.comfort_margin.expand(batch_size, 1),
        }
        ctx, u0, x, u, value = self.diff_mpc(x0, params=params, ctx=ctx)
        return u0, ctx, (x, u, value)

    def act(self, obs: dict) -> float:
        """One warm-started closed-loop step for a single (unbatched) observation.

        A warm-started solve already retries failed samples once from the
        default initializer inside leap-c; if the status is still nonzero the
        stale context is dropped so the next step starts fresh.
        """
        batched = {key: np.asarray(obs[key])[None] for key in
                   ("T", "outdoor", "price", "t_lower", "t_upper")}
        with torch.no_grad():
            u0, ctx, _ = self.forward(batched, ctx=self.ctx)
        self.ctx = ctx if ctx.status[0] == 0 else None
        return float(u0[0, 0])
