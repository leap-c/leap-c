"""Compare two ways of retrieving the sensitivity of the MPC solution w.r.t. ``xref1``.

Path A (current API): ``AcadosPlanner.sensitivity(ctx, name)``.
Path B (simplification): ``torch.autograd.functional.jacobian`` over the
``AcadosDiffMpcTorch`` layer.

Both should yield identical numbers because the autograd backward of the diff-mpc
layer calls the same acados sensitivity routines that ``.sensitivity(...)`` uses.

On top of the correctness check, this script times each path and profiles the hot
paths with ``cProfile``. Path A and Path B are *not* apples-to-apples as written:
Path A reuses a populated ``ctx`` (its per-field sensitivity cache plus the
backward-solver factorization cached in ``_PREPARE_BACKWARD_CACHE``), whereas
every ``torch.autograd.functional.jacobian`` call in Path B re-runs a fresh
forward solve with an empty ctx cache. To make the comparison fair we therefore
also time ``pathB_fair`` (a single shared forward solve) and break down the
caching effects (cold vs. cached sensitivity reads).
"""

import cProfile
import io
import pstats
import tempfile
from pathlib import Path
from timeit import default_timer

import torch

from leap_c.examples.cartpole.planner import CartPolePlanner, CartPolePlannerConfig

WARMUP = 2
REPEATS = 10
PROFILE_TOP_N = 25
PROFILE_DIR = Path(tempfile.gettempdir())


def time_call(fn, *, warmup: int = WARMUP, repeats: int = REPEATS) -> tuple[float, float]:
    """Return ``(best, mean)`` wall-clock seconds for ``fn`` over ``repeats`` runs.

    Runs ``warmup`` untimed calls first so one-time costs (acados sanity checks,
    backward-solver factorization, prepare caches) are excluded from the steady
    state numbers.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = default_timer()
        fn()
        samples.append(default_timer() - t0)
    return min(samples), sum(samples) / len(samples)


def _jacobian_via_grad(output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """Jacobian of ``output`` w.r.t. ``inp`` from a single shared forward graph.

    One reverse pass per output element, all reusing the same graph
    (``retain_graph=True``). Mirrors what ``torch.autograd.functional.jacobian``
    does internally, but lets us share one forward solve across several outputs.
    """
    flat = output.reshape(-1)
    rows = []
    for i in range(flat.numel()):
        seed = torch.zeros_like(flat)
        seed[i] = 1.0
        (grad,) = torch.autograd.grad(flat, inp, grad_outputs=seed, retain_graph=True)
        rows.append(grad)
    return torch.stack(rows)


def profile(name: str, fn) -> None:
    """Profile a single call of ``fn`` with cProfile: print top-N + save ``.prof``."""
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()

    out_path = PROFILE_DIR / f"compare_sensitivity_{name}.prof"
    pr.dump_stats(str(out_path))

    stream = io.StringIO()
    pstats.Stats(pr, stream=stream).sort_stats("cumulative").print_stats(PROFILE_TOP_N)
    print(f"\n=== cProfile: {name} (top {PROFILE_TOP_N} by cumulative time) ===")
    print(stream.getvalue())
    print(f"saved: {out_path}  (open with: snakeviz {out_path})")


def main() -> None:
    # 1. Build the planner. float64 gives a clean numerical comparison.
    t0 = default_timer()
    cfg = CartPolePlannerConfig(param_interface="stagewise", dtype=torch.float64)
    planner = CartPolePlanner(cfg)
    t_build = default_timer() - t0
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

    # 4b. Fair Path B: a single shared forward solve, then both jacobians off the
    #     same graph (so it is comparable to Path A's single forward solve).
    p = xref1.detach().clone().requires_grad_(True)
    _, u0_p, _, _, value_p = diff_mpc(x0=x0, params={"xref1": p})
    du0_dp_Bfair = _jacobian_via_grad(u0_p, p)
    dvalue_dp_Bfair = _jacobian_via_grad(value_p, p)

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
    assert torch.allclose(du0_dp_A.squeeze(), du0_dp_Bfair.squeeze(), atol=1e-6), (
        "du0/dxref1 mismatch between planner.sensitivity and the fair autograd path"
    )
    assert torch.allclose(dvalue_dp_A.squeeze(), dvalue_dp_Bfair.squeeze(), atol=1e-6), (
        "dvalue/dxref1 mismatch between planner.sensitivity and the fair autograd path"
    )

    print("OK: AcadosPlanner.sensitivity matches torch.autograd.functional.jacobian.")

    # 6. Timing. Each path callable is self-contained so it can be repeated.
    def forward_call():
        diff_mpc(x0=x0, params={"xref1": xref1})

    def path_a_call():
        ctx_a, *_ = diff_mpc(x0=x0, params={"xref1": xref1})
        planner.sensitivity(ctx_a, "du0_dp")
        planner.sensitivity(ctx_a, "dvalue_dp")

    def path_b_call():
        torch.autograd.functional.jacobian(lambda p: diff_mpc(x0=x0, params={"xref1": p})[1], xref1)
        torch.autograd.functional.jacobian(lambda p: diff_mpc(x0=x0, params={"xref1": p})[4], xref1)

    def path_b_fair_call():
        pp = xref1.detach().clone().requires_grad_(True)
        _, u0_pp, _, _, value_pp = diff_mpc(x0=x0, params={"xref1": pp})
        _jacobian_via_grad(u0_pp, pp)
        _jacobian_via_grad(value_pp, pp)

    results = {
        "build (1x, cold)": (t_build, t_build),
        "forward solve": time_call(forward_call),
        "pathA (solve + 2 sens)": time_call(path_a_call),
        "pathB (2x autograd jac)": time_call(path_b_call),
        "pathB_fair (shared solve)": time_call(path_b_fair_call),
    }
    print(f"\n=== timing summary over {REPEATS} repeats (ms) ===")
    print(f"{'path':<28}{'best':>10}{'mean':>10}")
    for name, (best, mean) in results.items():
        print(f"{name:<28}{best * 1e3:>10.3f}{mean * 1e3:>10.3f}")

    # 7. Caching breakdown: same fresh ctx each pass, time each step in order so the
    #    per-ctx sensitivity cache and _PREPARE_BACKWARD_CACHE effects are visible.
    for _ in range(WARMUP):
        c, *_ = diff_mpc(x0=x0, params={"xref1": xref1})
        planner.sensitivity(c, "du0_dp")
        planner.sensitivity(c, "dvalue_dp")

    acc = [0.0, 0.0, 0.0, 0.0]
    for _ in range(REPEATS):
        t0 = default_timer()
        ctx_c, *_ = diff_mpc(x0=x0, params={"xref1": xref1})
        acc[0] += default_timer() - t0

        t0 = default_timer()
        planner.sensitivity(ctx_c, "du0_dp")  # cold: factorize backward solver + compute
        acc[1] += default_timer() - t0

        t0 = default_timer()
        planner.sensitivity(ctx_c, "dvalue_dp")  # factorization reused (cache hit) + compute
        acc[2] += default_timer() - t0

        t0 = default_timer()
        planner.sensitivity(ctx_c, "du0_dp")  # ctx sensitivity cache hit (no compute)
        acc[3] += default_timer() - t0
    solve_t, du0_cold, dvalue_warm, du0_cached = (a / REPEATS * 1e3 for a in acc)

    print(f"\n=== caching breakdown, mean over {REPEATS} passes (ms) ===")
    print(f"forward solve                 : {solve_t:9.4f}")
    print(f"du0_dp   cold (ctx factorize) : {du0_cold:9.4f}")
    print(f"dvalue_dp warm (factorize     : {dvalue_warm:9.4f}  <- _PREPARE_BACKWARD_CACHE reused")
    print(f"du0_dp   cached (ctx cache)   : {du0_cached:9.4f}  <- per-ctx sensitivity cache hit")

    # 8. Profiling with the standard library (single call each).
    profile("pathA", path_a_call)
    profile("pathB", path_b_call)


if __name__ == "__main__":
    main()
