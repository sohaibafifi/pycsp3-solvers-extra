from __future__ import annotations

from pycsp3.parser.callbacks import Callbacks


class BackendCapabilities:
    def __init__(self, backend: object):
        self._backend = backend
        self._supported_constraints = self._collect_supported_constraints()

    def _collect_supported_constraints(self) -> set[str]:
        supported: set[str] = set()
        for cls in self._backend.__class__.mro():
            if cls is Callbacks:
                break
            for name, value in cls.__dict__.items():
                if name.startswith("ctr_") and callable(value):
                    supported.add(name)
        return supported

    def supports(self, name: str) -> bool:
        return name in self._supported_constraints

    def constraint_names(self) -> set[str]:
        return set(self._supported_constraints)
