from types import ModuleType

_TORCH_MODULE: ModuleType | None = None
_JAX_MODULE: ModuleType | None = None


def require_torch() -> ModuleType:
    """Import torch lazily and cache it for helpers that require it."""
    global _TORCH_MODULE
    if _TORCH_MODULE is not None:
        return _TORCH_MODULE

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required but not installed. Please install it via 'pip install torch'"
        ) from exc
    _TORCH_MODULE = torch
    return torch


def require_jax() -> ModuleType:
    """Import jax lazily and cache it for helpers that require it."""
    global _JAX_MODULE
    if _JAX_MODULE is not None:
        return _JAX_MODULE

    try:
        import jax
    except ImportError as exc:
        raise ImportError(
            "This function requires JAX. Please install it via 'pip install leap-c[jax]'"
        ) from exc
    _JAX_MODULE = jax
    return jax
