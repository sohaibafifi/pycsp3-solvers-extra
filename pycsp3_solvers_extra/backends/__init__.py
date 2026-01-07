"""
Solver backends for pycsp3-solvers-extra.

Each backend is a Callbacks subclass that translates XCSP3 elements
to solver-specific constraints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pycsp3_solvers_extra.backends.base import BaseCallbacks

# Registry of available backends
_BACKENDS: dict[str, type["BaseCallbacks"] | None] = {}


def register_backend(name: str, backend_class: type["BaseCallbacks"]) -> None:
    """Register a backend class."""
    _BACKENDS[name.lower()] = backend_class


def get_backend(name: str) -> type["BaseCallbacks"] | None:
    """
    Get a backend class by name.

    Returns None if the backend is not available (dependencies not installed).
    """
    name_lower = name.lower()

    # Lazy load backends to avoid import errors when deps missing
    if name_lower not in _BACKENDS:
        _try_load_backend(name_lower)

    return _BACKENDS.get(name_lower)


def _try_load_backend(name: str) -> None:
    """Try to load a backend, catching import errors."""
    if name == "ortools":
        try:
            from pycsp3_solvers_extra.backends.ortools_backend import ORToolsCallbacks
            _BACKENDS["ortools"] = ORToolsCallbacks
        except ImportError:
            _BACKENDS["ortools"] = None

    elif name == "cpo":
        try:
            from pycsp3_solvers_extra.backends.cpo_backend import CPOCallbacks
            _BACKENDS["cpo"] = CPOCallbacks
        except ImportError:
            _BACKENDS["cpo"] = None


def available_backends() -> list[str]:
    """Return list of available backend names."""
    # Try loading all known backends
    for name in ["ortools", "cpo"]:
        if name not in _BACKENDS:
            _try_load_backend(name)

    return [name for name, cls in _BACKENDS.items() if cls is not None]


__all__ = ["get_backend", "register_backend", "available_backends"]
