#!/usr/bin/env python3
"""Verify numerical equivalence of the six HVAC OCP parameterisations.

Six formulations are tested:
  resistance             - original, parameters in physical units (Rhi, Rie, Rea in K/W)
  conductance            - thermal resistances replaced by conductances (ghi, gie, gea in W/K)
  normalized             - all learnable parameters mapped to [-1, 1]
  normalized_conductance - conductance space, then normalised to [-1, 1]
  parameter_linear       - compound params (ahi=1/CiRhi, …) linear in dynamics
  discrete_matrix        - 18 normalised elements of discrete Ad, Bd, Ed

For each of several test cases (default params, perturbed params, extreme initial
states) all three solvers must agree on u₀, the full state/control trajectories,
and the optimal cost to within a configurable tolerance.

Usage
-----
  python scripts/hvac/test_ocp_equivalence.py [--n-horizon 10] [--reuse-code-dir /tmp/hvac]
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from leap_c.examples.hvac.acados_ocp import (
    export_parametric_ocp,
    export_parametric_ocp_conductance,
    export_parametric_ocp_discrete_matrix,
    export_parametric_ocp_normalized,
    export_parametric_ocp_normalized_conductance,
    export_parametric_ocp_parameter_linear,
    make_default_hvac_params,
    make_default_hvac_params_conductance,
    make_default_hvac_params_discrete_matrix,
    make_default_hvac_params_normalized,
    make_default_hvac_params_normalized_conductance,
    make_default_hvac_params_parameter_linear,
)
from leap_c.examples.hvac.dynamics import (
    HydronicDynamicsParameters,
    HydronicParameters,
    transcribe_discrete_state_space,
)
from leap_c.ocp.acados.parameters import AcadosParameterManager
from leap_c.ocp.acados.torch import AcadosDiffMpcTorch

# ── solver setup ──────────────────────────────────────────────────────────────


def _make_solver(
    ocp,
    export_dir: Path | None,
    n_batch: int = 1,
) -> AcadosDiffMpcTorch:
    return AcadosDiffMpcTorch(
        ocp,
        export_directory=export_dir,
        n_batch_init=n_batch,
        num_threads_batch_solver=1,
        verbose=False,
    )


# ── parameter vector helpers ──────────────────────────────────────────────────


def _default_p(pm: AcadosParameterManager) -> torch.Tensor:
    flat = pm.learnable_parameters_default.cat.full().flatten()
    return torch.tensor(flat[None, :], dtype=torch.float64)


def _default_ps(pm: AcadosParameterManager) -> torch.Tensor:
    arr = pm.combine_non_learnable_parameter_values(batch_size=1)
    return torch.tensor(arr, dtype=torch.float64)


_PARAM_LINEAR_KEYS = {"ahi", "aie", "bhi", "cie", "cea", "ch_inv", "gaw_ci"}
_DISCRETE_MATRIX_KEYS = (
    {f"ad{i}{j}" for i in range(3) for j in range(3)}
    | {f"bd{i}" for i in range(3)}
    | {f"ed{i}{j}" for i in range(3) for j in range(2)}
)


def _set_physical_dyn_param(
    pm: AcadosParameterManager,
    p: torch.Tensor,
    phys_name: str,
    phys_value: float,
    nom_dyn: HydronicDynamicsParameters,
) -> torch.Tensor:
    """Return a copy of p with one physical dynamics parameter set to phys_value.

    Handles all six parameterisations:
      - resistance:             key == phys_name,  physical units      (e.g. "Rhi")
      - conductance:            key == g_name,     conductance units   (e.g. "ghi")
      - normalised:             key == phys_name,  bounds in [-1, 1]
      - normalised_conductance: key == g_name,     bounds in [-1, 1]
      - parameter_linear:       recompute all compound params from the full physical state
      - discrete_matrix:        recompute all 18 discrete matrix elements and normalise
    """
    p = p.clone()
    keys = list(pm.learnable_parameters.keys())

    # ── discrete_matrix variant ────────────────────────────────────────────────
    if _DISCRETE_MATRIX_KEYS & set(keys):
        s = 0.8
        phys = {k: s * float(getattr(nom_dyn, k)) for k in asdict(nom_dyn)}
        phys[phys_name] = phys_value
        ref_dyn = HydronicDynamicsParameters(
            **{k: s * float(getattr(nom_dyn, k)) for k in asdict(nom_dyn)}
        )
        pert_dyn = HydronicDynamicsParameters(**phys)
        Ad_ref, Bd_ref, Ed_ref = transcribe_discrete_state_space(dt=900.0, params=ref_dyn)
        Ad_pert, Bd_pert, Ed_pert = transcribe_discrete_state_space(dt=900.0, params=pert_dyn)
        for j_idx, k in enumerate(keys):
            if len(k) == 4 and k[:2] == "ad":
                i, jj = int(k[2]), int(k[3])
                p[0, j_idx] = (float(Ad_pert[i, jj]) / float(Ad_ref[i, jj]) - 1.0) / 0.3
            elif len(k) == 3 and k[:2] == "bd":
                i = int(k[2])
                p[0, j_idx] = (float(Bd_pert[i, 0]) / float(Bd_ref[i, 0]) - 1.0) / 0.3
            elif len(k) == 4 and k[:2] == "ed":
                i, jj = int(k[2]), int(k[3])
                p[0, j_idx] = (float(Ed_pert[i, jj]) / float(Ed_ref[i, jj]) - 1.0) / 0.3
        return p

    # ── parameter_linear variant ───────────────────────────────────────────────
    if _PARAM_LINEAR_KEYS & set(keys):
        # Build the full physical state: all params at 0.8 × nominal, one overridden.
        s = 0.8
        phys = {k: s * float(getattr(nom_dyn, k)) for k in asdict(nom_dyn)}
        phys[phys_name] = phys_value
        compound = {
            "ahi": 1.0 / (phys["Ci"] * phys["Rhi"]),
            "aie": 1.0 / (phys["Ci"] * phys["Rie"]),
            "bhi": 1.0 / (phys["Ch"] * phys["Rhi"]),
            "cie": 1.0 / (phys["Ce"] * phys["Rie"]),
            "cea": 1.0 / (phys["Ce"] * phys["Rea"]),
            "ch_inv": 1.0 / phys["Ch"],
            "gaw_ci": phys["gAw"] / phys["Ci"],
        }
        for j, k in enumerate(keys):
            if k in compound:
                p[0, j] = compound[k]
        return p

    # ── all other variants ─────────────────────────────────────────────────────
    lbs_p = pm.learnable_parameters_lb.cat.full().flatten()
    ubs_p = pm.learnable_parameters_ub.cat.full().flatten()

    _R_TO_G = {"Rhi": "ghi", "Rie": "gie", "Rea": "gea"}
    g_name = _R_TO_G.get(phys_name)

    lb_phys = 0.7 * float(getattr(nom_dyn, phys_name))
    ub_phys = 1.3 * float(getattr(nom_dyn, phys_name))

    for j, k in enumerate(keys):
        lb_p = float(lbs_p[j])
        ub_p = float(ubs_p[j])
        is_normalised = abs(lb_p - (-1.0)) < 1e-9 and abs(ub_p - 1.0) < 1e-9

        if k == phys_name:
            if is_normalised:
                p[0, j] = -1.0 + 2.0 * (phys_value - lb_phys) / (ub_phys - lb_phys)
            else:
                p[0, j] = phys_value
        elif g_name is not None and k == g_name:
            g_value = 1.0 / phys_value
            if is_normalised:
                g_lb = 1.0 / (1.3 * float(getattr(nom_dyn, phys_name)))
                g_ub = 1.0 / (0.7 * float(getattr(nom_dyn, phys_name)))
                p[0, j] = -1.0 + 2.0 * (g_value - g_lb) / (g_ub - g_lb)
            else:
                p[0, j] = g_value

    return p


# ── forward pass ─────────────────────────────────────────────────────────────


@torch.no_grad()
def _solve(solver: AcadosDiffMpcTorch, x0: torch.Tensor, p: torch.Tensor, ps: torch.Tensor):
    ctx, u0, x, u, value = solver(x0, None, p, ps)
    return (
        int(ctx.status.flat[0]),
        u0.cpu().numpy(),
        x.cpu().numpy(),
        u.cpu().numpy(),
        value.cpu().numpy(),
    )


# ── comparison ───────────────────────────────────────────────────────────────


def _compare(label: str, a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> bool:
    ok = np.allclose(a, b, rtol=rtol, atol=atol)
    if not ok:
        diff = np.abs(a - b)
        print(
            f"    FAIL  max|Δ|={diff.max():.3e}  rel={diff.max() / max(np.abs(a).max(), 1e-12):.3e}"
        )
    return ok


# ── test cases ────────────────────────────────────────────────────────────────


def run_tests(
    solvers: dict[str, tuple[AcadosDiffMpcTorch, AcadosParameterManager]],
    nom_dyn: HydronicDynamicsParameters,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> bool:
    """Run all test cases and return True iff all pass."""
    from scipy.constants import convert_temperature as cvt

    Ti_warm = cvt(20.0, "C", "K")
    Ti_cold = cvt(14.0, "C", "K")
    Th_nom = cvt(30.0, "C", "K")
    Te_nom = cvt(10.0, "C", "K")

    def _x0(Ti: float, qh_prev: float = 0.5) -> torch.Tensor:
        return torch.tensor([[Ti, Th_nom, Te_nom, qh_prev]], dtype=torch.float64)

    test_cases: list[tuple[str, torch.Tensor, str | None, float | None]] = [
        # (description, x0, param_to_perturb, physical_value)
        ("default params, Ti=20°C", _x0(Ti_warm), None, None),
        ("default params, Ti=14°C (cold)", _x0(Ti_cold), None, None),
        ("Rhi = 1.1xnominal", _x0(Ti_warm), "Rhi", 1.1 * nom_dyn.Rhi),
        ("Rie = 0.9xnominal", _x0(Ti_warm), "Rie", 0.9 * nom_dyn.Rie),
        ("Rea = 1.2xnominal", _x0(Ti_warm), "Rea", 1.2 * nom_dyn.Rea),
        ("gAw = 0.75xnominal", _x0(Ti_warm), "gAw", 0.75 * nom_dyn.gAw),
        ("Ch  = 1.25xnominal", _x0(Ti_warm), "Ch", 1.25 * nom_dyn.Ch),
    ]

    all_pass = True
    names = list(solvers.keys())

    for desc, x0, pname, pval in test_cases:
        print(f"\n  [{desc}]")

        results: dict[str, tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for sname, (solver, pm) in solvers.items():
            p = _default_p(pm)
            ps = _default_ps(pm)
            if pname is not None:
                p = _set_physical_dyn_param(pm, p, pname, pval, nom_dyn)
            st, u0_sol, x_sol, u_sol, val = _solve(solver, x0, p, ps)
            results[sname] = (st, u0_sol, x_sol, u_sol, val)
            if st != 0:
                print(f"    WARN: {sname} solver status = {st}")

        # Compare all against the first (resistance) solver
        ref_name = names[0]
        _, u0_ref, x_ref, u_ref, val_ref = results[ref_name]

        case_pass = True
        for cname in names[1:]:
            _, u0_c, x_c, u_c, val_c = results[cname]
            fields = [
                ("u0", u0_ref, u0_c),
                ("x", x_ref, x_c),
                ("u", u_ref, u_c),
                ("value", val_ref, val_c),
            ]
            for fname, a, b in fields:
                ok = _compare(f"{ref_name} vs {cname} [{fname}]", a, b, rtol, atol)
                if not ok:
                    print(f"      {ref_name}  {fname} = {a.flat[0]:.6f}")
                    print(f"      {cname}  {fname} = {b.flat[0]:.6f}")
                case_pass = case_pass and ok

        status = "PASS" if case_pass else "FAIL"
        print(f"  → {status}")
        all_pass = all_pass and case_pass

    return all_pass


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify equivalence of HVAC OCP parameterisations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-horizon", type=int, default=10, help="MPC horizon steps (short for fast compilation)."
    )
    parser.add_argument(
        "--reuse-code-dir",
        type=Path,
        default=None,
        help="Reuse compiled acados code from this directory.",
    )
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    N = args.n_horizon
    interface = "reference_dynamics"
    granularity = "global"
    nom = HydronicParameters()

    def _export_dir(suffix: str) -> Path | None:
        if args.reuse_code_dir is None:
            return None
        d = args.reuse_code_dir / suffix
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── build three (param_manager, solver) pairs ─────────────────────────────
    print("Building solvers …")

    pm_r = AcadosParameterManager(make_default_hvac_params(interface, granularity, N, nom), N)
    ocp_r = export_parametric_ocp(pm_r, N, name="hvac_r")
    solver_r = _make_solver(ocp_r, _export_dir("resistance"))
    print("  resistance   ✓")

    pm_c = AcadosParameterManager(
        make_default_hvac_params_conductance(interface, granularity, N, nom), N
    )
    ocp_c = export_parametric_ocp_conductance(pm_c, N, name="hvac_c")
    solver_c = _make_solver(ocp_c, _export_dir("conductance"))
    print("  conductance  ✓")

    pm_n = AcadosParameterManager(
        make_default_hvac_params_normalized(interface, granularity, N, nom), N
    )
    ocp_n = export_parametric_ocp_normalized(pm_n, N, name="hvac_n", hydronic_params=nom)
    solver_n = _make_solver(ocp_n, _export_dir("normalized"))
    print("  normalized   ✓")

    pm_nc = AcadosParameterManager(
        make_default_hvac_params_normalized_conductance(interface, granularity, N, nom), N
    )
    ocp_nc = export_parametric_ocp_normalized_conductance(
        pm_nc, N, name="hvac_nc", hydronic_params=nom
    )
    solver_nc = _make_solver(ocp_nc, _export_dir("normalized_conductance"))
    print("  norm_cond    ✓")

    pm_pl = AcadosParameterManager(
        make_default_hvac_params_parameter_linear(interface, granularity, N, nom), N
    )
    ocp_pl = export_parametric_ocp_parameter_linear(pm_pl, N, name="hvac_pl")
    solver_pl = _make_solver(ocp_pl, _export_dir("parameter_linear"))
    print("  param_linear ✓")

    pm_dm = AcadosParameterManager(
        make_default_hvac_params_discrete_matrix(interface, granularity, N, nom), N
    )
    ocp_dm = export_parametric_ocp_discrete_matrix(pm_dm, N, name="hvac_dm", hydronic_params=nom)
    solver_dm = _make_solver(ocp_dm, _export_dir("discrete_matrix"))
    print("  disc_matrix  ✓")

    # ── print parameter tables ────────────────────────────────────────────────
    print("\nLearnable parameter defaults:")
    header = (
        f"  {'name':<12}  {'resistance':>15}  {'conductance':>15}"
        f"  {'normalized':>12}  {'norm_cond':>12}  {'param_lin':>12}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    def _flat(pm):
        return pm.learnable_parameters_default.cat.full().flatten()

    def _keys(pm):
        return list(pm.learnable_parameters.keys())

    r_dict = dict(zip(_keys(pm_r), _flat(pm_r)))
    c_dict = dict(zip(_keys(pm_c), _flat(pm_c)))
    n_dict = dict(zip(_keys(pm_n), _flat(pm_n)))
    nc_dict = dict(zip(_keys(pm_nc), _flat(pm_nc)))
    pl_dict = dict(zip(_keys(pm_pl), _flat(pm_pl)))
    dm_dict = dict(zip(_keys(pm_dm), _flat(pm_dm)))

    all_names = sorted(
        set(r_dict) | set(c_dict) | set(n_dict) | set(nc_dict) | set(pl_dict) | set(dm_dict)
    )
    for k in all_names:
        rv = f"{r_dict.get(k, float('nan')):>15.5g}"
        cv = f"{c_dict.get(k, float('nan')):>15.5g}"
        nv = f"{n_dict.get(k, float('nan')):>12.5g}"
        ncv = f"{nc_dict.get(k, float('nan')):>12.5g}"
        plv = f"{pl_dict.get(k, float('nan')):>12.5g}"
        dmv = f"{dm_dict.get(k, float('nan')):>12.5g}"
        print(f"  {k:<12}  {rv}  {cv}  {nv}  {ncv}  {plv}  {dmv}")

    # ── run tests ─────────────────────────────────────────────────────────────
    print("\nRunning equivalence tests …")
    solvers = {
        "resistance": (solver_r, pm_r),
        "conductance": (solver_c, pm_c),
        "normalized": (solver_n, pm_n),
        "normalized_conductance": (solver_nc, pm_nc),
        "parameter_linear": (solver_pl, pm_pl),
        "discrete_matrix": (solver_dm, pm_dm),
    }
    passed = run_tests(solvers, nom.dynamics, rtol=args.rtol, atol=args.atol)

    print()
    print("══════════════════════════════════════")
    print(f"  Overall result:  {'ALL PASS ✓' if passed else 'SOME FAILURES ✗'}")
    print("══════════════════════════════════════")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
