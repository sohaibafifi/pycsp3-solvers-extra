from __future__ import annotations

from pycsp3_solvers_extra.transforms.context import TransformContext
from pycsp3_solvers_extra.transforms.types import ConstraintCall


def rewrite_calls(
    calls: list[ConstraintCall], ctx: TransformContext
) -> list[ConstraintCall]:
    _ = ctx
    return calls
