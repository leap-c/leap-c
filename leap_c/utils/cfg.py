import dataclasses
from dataclasses import fields, is_dataclass
from typing import Any


def cfg_as_python(obj: Any, root_name: str = "cfg") -> str:
    lines = []

    def _recurse(o: Any, prefix: str):
        if dataclasses.is_dataclass(o):
            for field in dataclasses.fields(o):
                value = getattr(o, field.name)
                _recurse(value, f"{prefix}.{field.name}")
        else:
            repr_val = repr(o)
            lines.append(f"{prefix} = {repr_val}")

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
                update_dataclass_from_dict(getattr(dataclass_instance, field.name), update_dict[field.name])
            else:
                # Otherwise, directly update the field
                setattr(dataclass_instance, field.name, update_dict[field.name])
