from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pycsp3_solvers_extra.transforms.capabilities import BackendCapabilities


@dataclass
class TransformContext:
    backend: Any
    backend_name: str
    capabilities: BackendCapabilities

    def new_aux_int_var(self, lb: int, ub: int, name_hint: str = "aux") -> Any:
        creator = getattr(self.backend, "new_aux_int_var", None)
        if creator is None or not callable(creator):
            backend_label = getattr(self.backend, "__class__", type(self.backend)).__name__
            raise NotImplementedError(
                f"Aux variable creation not supported for backend: {backend_label}"
            )
        return creator(lb, ub, name_hint)
