from acados_template import AcadosModel, AcadosOcp
import numpy as np
import casadi as ca
from casadi.tools import entry, struct_symSX


def find_param_in_p_or_p_global(
    param_name: list[str], model: AcadosModel
) -> dict[ca.SX]:
    if model.p == []:
        return {key: model.p_global[key] for key in param_name}  # type:ignore
    elif model.p_global is None:
        return {key: model.p[key] for key in param_name}  # type:ignore
    else:
        return {
            key: (model.p[key] if key in model.p.keys() else model.p_global[key])  # type:ignore
            for key in param_name
        }


def _process_params(
    params: list[str], nominal_param: dict[str, np.ndarray]
) -> tuple[list, list]:
    entries = []
    values = []
    for param in params:
        try:
            entries.append(entry(param, shape=nominal_param[param].shape))
            values.append(nominal_param[param].T.reshape(-1, 1))
        except AttributeError:
            entries.append(entry(param, shape=(1, 1)))
            values.append(np.array([nominal_param[param]]).reshape(-1, 1))
    return entries, values


def translate_learnable_param_to_p_global(
    nominal_param: dict[str, np.ndarray],
    learnable_param: list[str],
    ocp: AcadosOcp,
    verbose: bool = False,
) -> AcadosOcp:
    if learnable_param:
        entries, values = _process_params(learnable_param, nominal_param)
        ocp.model.p_global = struct_symSX(entries)
        ocp.p_global_values = np.concatenate(values).flatten()

    non_learnable_params = [key for key in nominal_param if key not in learnable_param]
    if non_learnable_params:
        entries, values = _process_params(non_learnable_params, nominal_param)
        ocp.model.p = struct_symSX(entries)
        ocp.parameter_values = np.concatenate(values).flatten()

    if verbose:
        print("learnable_params", learnable_param)
        print("non_learnable_params", non_learnable_params)
    return ocp
