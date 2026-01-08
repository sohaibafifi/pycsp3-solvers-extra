from __future__ import annotations

from typing import Any

from pycsp3_solvers_extra.transforms.capabilities import BackendCapabilities
from pycsp3_solvers_extra.transforms.context import TransformContext
from pycsp3_solvers_extra.transforms.decompose import decompose_calls
from pycsp3_solvers_extra.transforms.normalize import normalize_calls
from pycsp3_solvers_extra.transforms.rewrite import rewrite_calls
from pycsp3_solvers_extra.transforms.types import ConstraintCall


class TransformingCallbacks:
    def __init__(self, backend: Any, backend_name: str):
        self._backend = backend
        self._backend_name = backend_name
        self._capabilities = BackendCapabilities(backend)
        self._ctx = TransformContext(
            backend=backend,
            backend_name=backend_name,
            capabilities=self._capabilities,
        )

    @property
    def backend(self) -> Any:
        return self._backend

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def __getattr__(self, name: str) -> Any:
        if name.startswith("ctr_"):
            return lambda *args, **kwargs: self._handle_ctr(name, args, kwargs)
        return getattr(self._backend, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._backend, name, value)

    def _handle_ctr(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        calls = [ConstraintCall(name=name, args=args, kwargs=kwargs)]
        calls = normalize_calls(calls, self._ctx)
        calls = rewrite_calls(calls, self._ctx)
        calls = decompose_calls(calls, self._ctx)
        for call in calls:
            self._dispatch(call)

    def _dispatch(self, call: ConstraintCall) -> Any:
        if not self._capabilities.supports(call.name):
            raise NotImplementedError(
                f"{self._backend_name} backend does not implement {call.name}; "
                "add a decomposition or backend support"
            )
        method = getattr(self._backend, call.name)
        return method(*call.args, **call.kwargs)
