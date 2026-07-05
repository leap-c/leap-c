"""Human-readable repr formatting for acados MPC objects."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from acados_template import AcadosOcp

if TYPE_CHECKING:
    from leap_c.parameters import AcadosParameterManager


def _format_array(arr: np.ndarray) -> str:
    return re.sub(
        r"\s+",
        " ",
        np.array2string(arr, max_line_width=np.inf, separator=", ", threshold=10, edgeitems=2),
    )


def format_parameter_sections(parameter_manager: AcadosParameterManager) -> str:
    """Return differentiable/non-differentiable parameter table sections."""
    lines: list[str] = []

    diff_names = parameter_manager.differentiable_parameter_names
    lines.append("  differentiable:")
    if diff_names:
        rows = []
        for name in diff_names:
            param = parameter_manager.parameters[name]
            splits = str(param.splits)
            shape = str(param.overwrite_shape(parameter_manager.N_horizon))
            default = _format_array(param.broadcasted_default(parameter_manager.N_horizon))
            rows.append((name, splits, shape, default))
        w_name = max(len("name"), *(len(r[0]) for r in rows))
        w_splits = max(len("splits"), *(len(r[1]) for r in rows))
        w_shape = max(len("shape"), *(len(r[2]) for r in rows))
        lines.append(
            f"    {'name':<{w_name}}  {'splits':<{w_splits}}  {'shape':<{w_shape}}  default"
        )
        for name, splits, shape, default in rows:
            lines.append(
                f"    {name:<{w_name}}  {splits:<{w_splits}}  {shape:<{w_shape}}  {default}"
            )

    nondiff_names = parameter_manager.non_differentiable_parameter_names
    lines.append("  non-differentiable:")
    if nondiff_names:
        rows = []
        for name in nondiff_names:
            param = parameter_manager.parameters[name]
            shape = (parameter_manager.N_horizon + 1, *param.default.shape)
            tiled = np.tile(
                param.default, (parameter_manager.N_horizon + 1, *([1] * param.default.ndim))
            )
            default = _format_array(tiled)
            rows.append((name, str(shape), default))
        w_name = max(len("name"), *(len(r[0]) for r in rows))
        w_shape = max(len("shape"), *(len(r[1]) for r in rows))
        lines.append(f"    {'name':<{w_name}}  {'shape':<{w_shape}}  default")
        for name, shape, default in rows:
            lines.append(f"    {name:<{w_name}}  {shape:<{w_shape}}  {default}")

    return "\n".join(lines)


def format_parameter_manager_repr(parameter_manager: AcadosParameterManager) -> str:
    """Return the full repr for an acados parameter manager."""
    df_size = parameter_manager.differentiable_default_flat.size
    ndf_size = parameter_manager.non_differentiable_default_flat.size
    header = (
        f"AcadosParameterManager(N_horizon={parameter_manager.N_horizon}, "
        f"casadi_type='{parameter_manager.casadi_type}', "
        f"differentiable_flat={df_size}, "
        f"non_differentiable_flat={ndf_size}),"
    )
    return header + "\n" + format_parameter_sections(parameter_manager)


def format_diff_mpc_module_extra_repr(
    *, ocp: AcadosOcp, parameter_manager: AcadosParameterManager
) -> str:
    """Return the common one-line repr summary for acados DiffMPC modules."""
    N = ocp.solver_options.N_horizon
    nx = ocp.dims.nx
    nu = ocp.dims.nu
    ct = parameter_manager.casadi_type
    return f"N_horizon={N}, nx={nx}, nu={nu}, casadi_type='{ct}'"


def format_diff_mpc_module_repr(
    *, class_name: str, ocp: AcadosOcp, parameter_manager: AcadosParameterManager
) -> str:
    """Return the full repr for a Torch/JAX-style acados DiffMPC module."""
    N = ocp.solver_options.N_horizon
    nx = ocp.dims.nx
    nu = ocp.dims.nu
    sections = format_parameter_sections(parameter_manager)
    sections_indented = "\n".join("  " + line if line else line for line in sections.splitlines())
    return (
        f"{class_name}(\n"
        f"  {format_diff_mpc_module_extra_repr(ocp=ocp, parameter_manager=parameter_manager)}\n"
        "  inputs:\n"
        f"    x0       (B, {nx})     initial states\n"
        f"    u0       (B, {nu})     fixates first-stage control (optional)\n"
        "    params   dict       parameter overrides (see parameters)\n"
        "  outputs:\n"
        f"    u0       (B, {nu})     optimal first-stage control (= u0 if given)\n"
        f"    x        (B, {N + 1}, {nx})  state trajectory\n"
        f"    u        (B, {N}, {nu})    control trajectory\n"
        "    value    (B, 1)      cost (V(x0), or Q(x0, u0) when u0 given)\n"
        "  parameters:\n"
        f"{sections_indented}\n"
        ")"
    )
