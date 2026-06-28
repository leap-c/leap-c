"""Compare two ways of retrieving the sensitivity of the MPC solution w.r.t. ``xref1``.

Path A (current API): ``AcadosPlanner.sensitivity(ctx, name)``.
Path B (simplification): ``torch.autograd.functional.jacobian`` over the
``AcadosDiffMpcTorch`` layer.

Both should yield identical numbers because the autograd backward of the diff-mpc
layer calls the same acados sensitivity routines that ``.sensitivity(...)`` uses.
"""

import torch

from leap_c.examples.cartpole.planner import CartPolePlanner, CartPolePlannerConfig


def main() -> None:
    # 1. Build the planner. float64 gives a clean numerical comparison.
    cfg = CartPolePlannerConfig(param_interface="stagewise", dtype=torch.float64)
    planner = CartPolePlanner(cfg)
    diff_mpc = planner.diff_mpc  # the AcadosDiffMpcTorch == "mpc_layer"

    # 2. Inputs (batch size 1, off-equilibrium so sensitivities are nonzero and the
    #    action/position constraints stay inactive).
    x0 = torch.tensor([[0.0, 0.2, 0.0, 0.0]], dtype=torch.float64)
    # stagewise -> one value per node, shape (B, N_horizon + 1, 1)
    xref1 = 0.1 * torch.ones((1, cfg.N_horizon + 1, 1), dtype=torch.float64)

    # 3. Path A: current API.
    ctx, u0, x, u, value = diff_mpc(x0=x0, params={"xref1": xref1})
    du0_dp_A = torch.as_tensor(planner.sensitivity(ctx, "du0_dp"))  # (B, nu, N+1)
    dvalue_dp_A = torch.as_tensor(planner.sensitivity(ctx, "dvalue_dp"))  # (B, 1, N+1)

    # 4. Path B: standard PyTorch autograd over the diff-mpc layer.
    du0_dp_B = torch.autograd.functional.jacobian(
        lambda p: diff_mpc(x0=x0, params={"xref1": p})[1], xref1
    )  # (B, nu) + xref1.shape
    dvalue_dp_B = torch.autograd.functional.jacobian(
        lambda p: diff_mpc(x0=x0, params={"xref1": p})[4], xref1
    )  # (B, 1) + xref1.shape

    # 5. Compare. With the stagewise interface the sensitivities are vectors (one
    #    entry per node), so print the full squeezed tensors.
    torch.set_printoptions(precision=8, sci_mode=False)
    print("du0/dxref1")
    print("  planner :", du0_dp_A.squeeze())
    print("  autograd:", du0_dp_B.squeeze())
    print("dvalue/dxref1")
    print("  planner :", dvalue_dp_A.squeeze())
    print("  autograd:", dvalue_dp_B.squeeze())

    assert torch.allclose(du0_dp_A.squeeze(), du0_dp_B.squeeze(), atol=1e-6), (
        "du0/dxref1 mismatch between planner.sensitivity and autograd.jacobian"
    )
    assert torch.allclose(dvalue_dp_A.squeeze(), dvalue_dp_B.squeeze(), atol=1e-6), (
        "dvalue/dxref1 mismatch between planner.sensitivity and autograd.jacobian"
    )

    print("OK: AcadosPlanner.sensitivity matches torch.autograd.functional.jacobian.")


if __name__ == "__main__":
    main()
