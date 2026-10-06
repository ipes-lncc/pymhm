"""Composable multiscale hybrid mixed methods with lazily loaded public exports.

Basix supplies reusable reference elements. Optional native integrations load
only when their adapter is used; local algebra also accepts external providers.
"""

from importlib import import_module
from typing import Any

from pymhm._registry import PUBLIC_EXPORTS, PUBLIC_NAMES

__version__ = "1.2.0"
__all__ = PUBLIC_NAMES


def __getattr__(name: str) -> Any:
    """Resolve a primary public symbol from its canonical owner on first access."""
    if name in PUBLIC_EXPORTS:
        module, symbol = PUBLIC_EXPORTS[name]
        value = getattr(import_module(module), symbol)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List public exports and already loaded module attributes."""
    return sorted(set(globals()) | set(PUBLIC_NAMES))
