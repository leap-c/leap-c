"""Compare ways of retrieving the sensitivity of the MPC solution w.r.t. ``xref1``.

Path A: ``diff_mpc.diff_mpc_fun.sensitivity(ctx, field)`` (low-level acados API).
Path B: ``functional.jacobian`` called twice, cold (naive: 2 forward solves).
Path C: ``functional.jacobian`` called twice, warmstarted via ``ctx``.
Path D: ``functional.jacobian`` once, returning both outputs (typical: 1 solve).
Path E: same as D but warmstarted via ``ctx``.
Finite differences: central difference as ground-truth reference.

All paths should yield identical numbers. The script prints a compact correctness
summary, timing summary, and a theoretical cost breakdown.
"""

from timeit import default_timer

import torch

from leap_c.examples.cartpole.planner import CartPolePlanner, CartPolePlannerConfig

WARMUP = 2
REPEATS = 50


def time_call(fn, *, warmup: int = WARMUP, repeats: int = REPEATS) -> tuple[float, float]:
    """Return ``(best, mean)`` wall-clock seconds for ``fn`` over ``repeats`` runs."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = default_timer()
        fn()
        samples.append(default_timer() - t0)
    return min(samples), sum(samples) / len(samples)


def _jacobian_via_grad(output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """Jacobian via a manual ``torch.autograd.grad`` loop (reuses existing graph)."""
    flat = output.reshape(-1)
    rows = []
    for i in range(flat.numel()):
        seed = torch.zeros_like(flat)
        seed[i] = 1.0
        (grad,) = torch.autograd.grad(flat, inp, grad_outputs=seed, retain_graph=True)
        rows.append(grad)
    return torch.stack(rows)


def finite_diff_jacobian(fn, p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Central difference Jacobian of ``fn(p)`` w.r.t. ``p``.

    Perturbs each element of ``p`` by ``±eps`` and evaluates ``fn``.
    Cost: ``2 * p.numel()`` forward solves.
    """
    out0 = fn(p)
    n_out = out0.reshape(-1).numel()
    n_in = p.reshape(-1).numel()
    jac = torch.zeros(n_out, n_in, dtype=p.dtype)
    for i in range(n_in):
        p_plus = p.clone().reshape(-1)
        p_plus[i] += eps
        out_plus = fn(p_plus.reshape_as(p)).reshape(-1)
        p_minus = p.clone().reshape(-1)
        p_minus[i] -= eps
        out_minus = fn(p_minus.reshape_as(p)).reshape(-1)
        jac[:, i] = (out_plus - out_minus) / (2 * eps)
    return jac.reshape(*out0.shape, *p.shape)


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max absolute difference between two tensors (squeezed)."""
    return (a.squeeze() - b.squeeze()).abs().max().item()


def main() -> None:
    cfg = CartPolePlannerConfig(param_interface="stagewise", dtype=torch.float64)
    planner = CartPolePlanner(cfg)
    diff_mpc = planner.diff_mpc

    x0 = torch.tensor([[0.0, 0.2, 0.0, 0.0]], dtype=torch.float64)
    xref1 = 0.1 * torch.ones((1, cfg.N_horizon + 1, 1), dtype=torch.float64)

    # --- compute sensitivities via all paths ---
    ctx, *_ = diff_mpc(x0=x0, params={"xref1": xref1})
    du0_dp_A = torch.as_tensor(diff_mpc.diff_mpc_fun.sensitivity(ctx, "du0_dp_global"))
    dvalue_dp_A = torch.as_tensor(diff_mpc.diff_mpc_fun.sensitivity(ctx, "dvalue_dp_global"))

    du0_dp_B = torch.autograd.functional.jacobian(
        lambda p: diff_mpc(x0=x0, params={"xref1": p})[1], xref1
    )
    dvalue_dp_B = torch.autograd.functional.jacobian(
        lambda p: diff_mpc(x0=x0, params={"xref1": p})[4], xref1
    )

    p = xref1.detach().clone().requires_grad_(True)
    _, u0_p, _, _, value_p = diff_mpc(x0=x0, params={"xref1": p})
    du0_dp_Bfair = _jacobian_via_grad(u0_p, p)
    dvalue_dp_Bfair = _jacobian_via_grad(value_p, p)

    ctx_warm, *_ = diff_mpc(x0=x0, params={"xref1": xref1})
    du0_dp_C = torch.autograd.functional.jacobian(
        lambda p: diff_mpc(x0=x0, params={"xref1": p}, ctx=ctx_warm)[1], xref1
    )
    dvalue_dp_C = torch.autograd.functional.jacobian(
        lambda p: diff_mpc(x0=x0, params={"xref1": p}, ctx=ctx_warm)[4], xref1
    )

    du0_dp_D, dvalue_dp_D = torch.autograd.functional.jacobian(
        lambda p: diff_mpc(x0=x0, params={"xref1": p})[1::3], xref1
    )
    du0_dp_E, dvalue_dp_E = torch.autograd.functional.jacobian(
        lambda p: diff_mpc(x0=x0, params={"xref1": p}, ctx=ctx_warm)[1::3], xref1
    )

    du0_dp_FD = finite_diff_jacobian(lambda p: diff_mpc(x0=x0, params={"xref1": p})[1], xref1)
    dvalue_dp_FD = finite_diff_jacobian(lambda p: diff_mpc(x0=x0, params={"xref1": p})[4], xref1)

    # --- correctness summary ---
    path_descriptions = {
        "pathA": "diff_mpc_fun.sensitivity(ctx, field) — low-level API (1 fwd + 2 adjoint).",
        "pathB": "functional.jacobian ×2, cold — naive (2 cold fwds + 2 adjoint).",
        "pathB_fair": "grad loop on shared graph — 1 cold fwd + 2 grad loops.",
        "pathC": "functional.jacobian ×2, warmstarted via ctx (2 warm fwds + 2 adjoint).",
        "pathD": "functional.jacobian ×1, lambda returns (u0, value) — typical (1 cold fwd).",
        "pathE": "Same as D, warmstarted via ctx (1 warm fwd).",
        "finite_diff": "Central finite differences (2 * p.numel() fwds) — ground truth.",
    }

    paths = {
        "pathB": (du0_dp_B, dvalue_dp_B),
        "pathB_fair": (du0_dp_Bfair, dvalue_dp_Bfair),
        "pathC": (du0_dp_C, dvalue_dp_C),
        "pathD": (du0_dp_D, dvalue_dp_D),
        "pathE": (du0_dp_E, dvalue_dp_E),
        "finite_diff": (du0_dp_FD, dvalue_dp_FD),
    }

    print("=== path descriptions ===")
    for name, desc in path_descriptions.items():
        print(f"  {name:<16} {desc}")

    print("\n=== correctness (max abs diff vs pathA) ===")
    print(f"{'path':<16}{'du0/dp':>14}{'dvalue/dp':>14}")
    for name, (d_u0, d_val) in paths.items():
        err_u0 = max_diff(du0_dp_A, d_u0)
        err_val = max_diff(dvalue_dp_A, d_val)
        print(f"{name:<16}{err_u0:>14.2e}{err_val:>14.2e}")
        tol = 1e-4 if name == "finite_diff" else 1e-6
        assert err_u0 < tol, f"{name} du0/dp mismatch (err={err_u0:.2e}, tol={tol:.0e})"
        assert err_val < tol, f"{name} dvalue/dp mismatch (err={err_val:.2e}, tol={tol:.0e})"
    print("OK: all paths match.")

    # --- timing ---
    def forward_call():
        diff_mpc(x0=x0, params={"xref1": xref1})

    def path_a_call():
        ctx_a, *_ = diff_mpc(x0=x0, params={"xref1": xref1})
        diff_mpc.diff_mpc_fun.sensitivity(ctx_a, "du0_dp_global")
        diff_mpc.diff_mpc_fun.sensitivity(ctx_a, "dvalue_dp_global")

    def path_b_call():
        torch.autograd.functional.jacobian(lambda p: diff_mpc(x0=x0, params={"xref1": p})[1], xref1)
        torch.autograd.functional.jacobian(lambda p: diff_mpc(x0=x0, params={"xref1": p})[4], xref1)

    def path_b_fair_call():
        pp = xref1.detach().clone().requires_grad_(True)
        _, u0_pp, _, _, value_pp = diff_mpc(x0=x0, params={"xref1": pp})
        _jacobian_via_grad(u0_pp, pp)
        _jacobian_via_grad(value_pp, pp)

    def path_c_call():
        torch.autograd.functional.jacobian(
            lambda p: diff_mpc(x0=x0, params={"xref1": p}, ctx=ctx_warm)[1], xref1
        )
        torch.autograd.functional.jacobian(
            lambda p: diff_mpc(x0=x0, params={"xref1": p}, ctx=ctx_warm)[4], xref1
        )

    def path_d_call():
        torch.autograd.functional.jacobian(
            lambda p: diff_mpc(x0=x0, params={"xref1": p})[1::3], xref1
        )

    def path_e_call():
        torch.autograd.functional.jacobian(
            lambda p: diff_mpc(x0=x0, params={"xref1": p}, ctx=ctx_warm)[1::3], xref1
        )

    results = {
        "forward solve": time_call(forward_call),
        "pathA (sensitivity API)": time_call(path_a_call),
        "pathB (jacobian 2x cold)": time_call(path_b_call),
        "pathB_fair (grad loop)": time_call(path_b_fair_call),
        "pathC (jacobian 2x warm)": time_call(path_c_call),
        "pathD (jacobian 1x cold)": time_call(path_d_call),
        "pathE (jacobian 1x warm)": time_call(path_e_call),
    }
    print(f"\n=== timing summary over {REPEATS} repeats (ms) ===")
    print(f"{'path':<28}{'best':>10}{'mean':>10}")
    for name, (best, mean) in results.items():
        print(f"{name:<28}{best * 1e3:>10.3f}{mean * 1e3:>10.3f}")

    print("\n=== theoretical cost breakdown (per call, N=nu) ===")
    print(f"{'path':<28}{'fwd':>8}{'factor':>10}{'adjoint':>10}{'einsum':>10}")
    print(f"{'pathA (sensitivity API)':<28}{'1 cold':>8}{1:>10}{2:>10}{0:>10}")
    print(f"{'pathB (jacobian 2x cold)':<28}{'2 cold':>8}{2:>10}{2:>10}{'2N':>10}")
    print(f"{'pathB_fair (grad loop)':<28}{'1 cold':>8}{1:>10}{2:>10}{'2N':>10}")
    print(f"{'pathC (jacobian 2x warm)':<28}{'2 warm':>8}{2:>10}{2:>10}{'2N':>10}")
    print(f"{'pathD (jacobian 1x cold)':<28}{'1 cold':>8}{1:>10}{2:>10}{'2N':>10}")
    print(f"{'pathE (jacobian 1x warm)':<28}{'1 warm':>8}{1:>10}{2:>10}{'2N':>10}")


if __name__ == "__main__":
    main()
