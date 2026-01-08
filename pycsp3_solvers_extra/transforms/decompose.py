from __future__ import annotations

from pycsp3_solvers_extra.transforms.context import TransformContext
from pycsp3_solvers_extra.transforms.types import ConstraintCall


def decompose_calls(
    calls: list[ConstraintCall], ctx: TransformContext
) -> list[ConstraintCall]:
    output: list[ConstraintCall] = []
    for call in calls:
        if ctx.capabilities.supports(call.name):
            output.append(call)
            continue
        decomposed = _decompose_call(call, ctx)
        if decomposed is None:
            raise NotImplementedError(
                f"{ctx.backend_name} backend does not support {call.name} "
                "and no decomposition is defined"
            )
        output.extend(decomposed)
    return output


def _decompose_call(
    call: ConstraintCall, ctx: TransformContext
) -> list[ConstraintCall] | None:
    decomposer = getattr(ctx.backend, "decompose_call", None)
    if decomposer is None or not callable(decomposer):
        return None
    return decomposer(call, ctx)
