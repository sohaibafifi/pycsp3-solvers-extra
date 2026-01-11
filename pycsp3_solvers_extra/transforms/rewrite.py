from __future__ import annotations

from pycsp3_solvers_extra.transforms.context import TransformContext
from pycsp3_solvers_extra.transforms.types import ConstraintCall


def rewrite_calls(
    calls: list[ConstraintCall], ctx: TransformContext
) -> list[ConstraintCall]:
    _ = ctx
    output: list[ConstraintCall] = []
    for call in calls:
        if call.name == "ctr_lex_matrix":
            output.append(ConstraintCall("ctr_lex", call.args, call.kwargs))
        else:
            output.append(call)
    return output
