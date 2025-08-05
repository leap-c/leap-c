import dataclasses
from dataclasses import fields, is_dataclass
from typing import Any


def cfg_as_python(obj: Any, root_name: str = "cfg") -> str:
    """Prints a dataclass as a Python script.

    We assume that each node in the dataclass has a default value.

    Args:
        obj: The dataclass object to print.
        root_name: The name of the root variable in the output.

    Returns:
        A string representing the dataclass as a Python script.
    """
    lines = []

    def _recurse(o: Any, prefix: str, depth: int = 0):
        if dataclasses.is_dataclass(o):
            if depth > 0:
                lines.append("")
                lines.append(f"# ---- Section: {prefix} ----")
            for field in sorted(
                dataclasses.fields(o),
                key=lambda f: dataclasses.is_dataclass(getattr(o, f.name)),
            ):
                value = getattr(o, field.name)
                _recurse(value, f"{prefix}.{field.name}", depth + 1)
        else:
            repr_val = repr(o)
            lines.append(f"{prefix} = {repr_val}")

    lines.append("# ---- Configuration ----")
    lines.append(f"{root_name} = {obj.__class__.__name__}()")
    _recurse(obj, root_name)
    return "\n".join(lines)


def update_dataclass_from_dict(dataclass_instance, update_dict):
    """Recursively update a dataclass instance with values from a dictionary."""
    for field in fields(dataclass_instance):
        # Check if the field is present in the update dictionary
        if field.name in update_dict:
            # If the field is a dataclass itself, recursively update it
            if is_dataclass(getattr(dataclass_instance, field.name)):
                update_dataclass_from_dict(
                    getattr(dataclass_instance, field.name), update_dict[field.name]
                )
            else:
                # Otherwise, directly update the field
                setattr(dataclass_instance, field.name, update_dict[field.name])
