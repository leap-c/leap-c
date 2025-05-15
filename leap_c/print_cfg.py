import dataclasses
from typing import Any


def print_cfg_as_python(obj: Any, root_name: str = "cfg") -> str:
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
