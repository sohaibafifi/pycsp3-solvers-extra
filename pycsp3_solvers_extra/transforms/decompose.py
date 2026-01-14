from __future__ import annotations

from pycsp3.classes.auxiliary.conditions import Condition
from pycsp3.classes.auxiliary.enums import TypeConditionOperator
from pycsp3.classes.main.variables import VariableInteger, Domain
from pycsp3.classes.nodes import Node, TypeNode

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
    common = _decompose_common(call, ctx)
    if common is not None:
        return common
    decomposer = getattr(ctx.backend, "decompose_call", None)
    if decomposer is None or not callable(decomposer):
        return None
    return decomposer(call, ctx)


def _decompose_common(
    call: ConstraintCall, ctx: TransformContext
) -> list[ConstraintCall] | None:
    if call.name == "ctr_among":
        lst, values, k = call.args
        condition = Condition.build_condition((TypeConditionOperator.EQ, k))
        return [ConstraintCall("ctr_count", (lst, values, condition), {})]

    if call.name == "ctr_cardinality":
        lst, values, occurs, closed = call.args

        def _occ_condition(occ):
            if isinstance(occ, range):
                return Condition.build_condition((TypeConditionOperator.IN, occ))
            if isinstance(occ, (list, tuple, set, frozenset)):
                return Condition.build_condition((TypeConditionOperator.IN, list(occ)))
            if isinstance(occ, str) and ".." in occ:
                parts = occ.split("..", 1)
                if len(parts) == 2 and parts[0].lstrip("-").isdigit() and parts[1].lstrip("-").isdigit():
                    start = int(parts[0])
                    end = int(parts[1])
                    if start <= end:
                        return Condition.build_condition((TypeConditionOperator.IN, range(start, end + 1)))
            return Condition.build_condition((TypeConditionOperator.EQ, occ))

        calls = [
            ConstraintCall("ctr_count", (lst, [value], _occ_condition(occ)), {})
            for value, occ in zip(values, occurs)
        ]
        if closed:
            in_values = Condition.build_condition((TypeConditionOperator.EQ, 1))
            for var in lst:
                calls.append(ConstraintCall("ctr_count", ([var], values, in_values), {}))
        return calls

    if call.name == "ctr_not_all_qual":
        (lst,) = call.args
        condition = Condition.build_condition((TypeConditionOperator.GE, 2))
        return [ConstraintCall("ctr_nvalues", (lst, None, condition), {})]

    if call.name == "ctr_element_matrix":
        matrix, row_index, col_index, condition = call.args

        if not matrix or not matrix[0]:
            raise ValueError("Element matrix must be non-empty")
        cols = len(matrix[0])
        if any(len(row) != cols for row in matrix):
            raise ValueError("Element matrix must be rectangular for decomposition")

        flat = [cell for row in matrix for cell in row]
        max_index = len(flat) - 1
        counter = getattr(ctx.backend, "_aux_counter", 0)
        while True:
            aux_id = f"__aux_idx_{counter}"
            counter += 1
            if aux_id not in ctx.backend.vars:
                break
        ctx.backend._aux_counter = counter
        aux_var = VariableInteger(aux_id, Domain(range(max_index + 1)))
        ctx.backend.vars[aux_id] = ctx.new_aux_int_var(0, max_index, aux_id)

        expr = Node.build(TypeNode.ADD, Node.build(TypeNode.MUL, row_index, cols), col_index)
        tree = Node.build(TypeNode.EQ, aux_var, expr)
        scope = list(tree.scope())

        return [
            ConstraintCall("ctr_intension", (scope, tree), {}),
            ConstraintCall("ctr_element", (flat, aux_var, condition), {}),
        ]

    return None
