# OUTDATED
import casadi as ca


def SX_to_labels(SX: ca.SX) -> list[str]:
    return SX.str().strip("[]").split(", ")


def find_idx_for_labels(sub_vars: ca.SX, sub_label: str) -> list[int]:
    """Return a list of indices where sub_label is part of the variable label."""
    return [
        idx
        for idx, label in enumerate(sub_vars.str().strip("[]").split(", "))
        if sub_label in label
    ]
