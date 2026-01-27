from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

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

    if call.name == "ctr_mdd":
        scope, transitions = call.args
        start_state, final_state, layered_transitions = _prepare_mdd(scope, transitions)
        return [
            ConstraintCall(
                "ctr_regular",
                (scope, layered_transitions, start_state, [final_state]),
                {},
            )
        ]

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


def _domain_values(var) -> list[int]:
    values = var.dom.all_values()
    if values is None:
        raise ValueError("MDD decomposition requires finite integer domains")
    values = list(values)
    if not all(isinstance(v, int) for v in values):
        raise NotImplementedError("MDD decomposition currently supports only integer domains")
    return values


def _expand_label(label, domain_values: list[int], domain_set: set[int]) -> Iterable[int]:
    if isinstance(label, int):
        return (label,) if label in domain_set else ()

    if isinstance(label, Condition):
        candidates = list(label.filtering(domain_values))
    elif isinstance(label, range):
        candidates = list(label)
    elif isinstance(label, (list, tuple, set, frozenset)):
        expanded: list[int] = []
        for item in label:
            expanded.extend(_expand_label(item, domain_values, domain_set))
        candidates = expanded
    else:
        raise NotImplementedError(f"Unsupported MDD transition label type: {type(label).__name__}")

    filtered: list[int] = []
    for value in candidates:
        if not isinstance(value, int):
            raise NotImplementedError("MDD labels must resolve to integer values")
        if value in domain_set:
            filtered.append(value)
    # Remove duplicates while preserving order
    seen: set[int] = set()
    return tuple(v for v in filtered if not (v in seen or seen.add(v)))


def _prepare_mdd(scope: list, transitions: list) -> tuple[str, str, list[tuple[str, int, str]]]:
    if not scope:
        raise ValueError("MDD constraint requires a non-empty scope")
    if not transitions:
        raise ValueError("MDD constraint requires at least one transition")

    outgoing: dict[str, list[tuple[str, object, str]]] = defaultdict(list)
    indegree: dict[str, int] = defaultdict(int)
    outdegree: dict[str, int] = defaultdict(int)
    states: set[str] = set()

    for src, label, dst in transitions:
        if not isinstance(src, str) or not isinstance(dst, str):
            raise TypeError("MDD states must be strings")
        states.add(src)
        states.add(dst)
        outgoing[src].append((src, label, dst))
        indegree[dst] += 1
        outdegree[src] += 1
        # Ensure keys exist for degree lookups
        indegree[src] += 0
        outdegree[dst] += 0

    roots = [s for s in states if indegree[s] == 0]
    if len(roots) != 1:
        raise ValueError(f"MDD must have exactly one root (found {len(roots)})")
    root = roots[0]

    # Restrict to states reachable from the root
    reachable: set[str] = set()
    queue: deque[str] = deque([root])
    while queue:
        state = queue.popleft()
        if state in reachable:
            continue
        reachable.add(state)
        for _, _, dst in outgoing.get(state, ()):
            if dst not in reachable:
                queue.append(dst)

    reachable_transitions = [
        (src, label, dst) for (src, label, dst) in transitions if src in reachable and dst in reachable
    ]

    # Recompute outdegree on reachable subgraph to find the terminal
    reachable_outdegree: dict[str, int] = defaultdict(int)
    for src, _, dst in reachable_transitions:
        reachable_outdegree[src] += 1
        reachable_outdegree[dst] += 0

    terminals = [s for s in reachable if reachable_outdegree[s] == 0]
    if len(terminals) != 1:
        raise ValueError(f"MDD must have exactly one terminal (found {len(terminals)})")
    terminal = terminals[0]

    # Infer unique layer for each state; layered MDDs must be consistent.
    layers: dict[str, int] = {root: 0}
    queue = deque([root])
    while queue:
        state = queue.popleft()
        layer = layers[state]
        for _, _, dst in outgoing.get(state, ()):
            if dst not in reachable:
                continue
            expected = layer + 1
            if dst in layers:
                if layers[dst] != expected:
                    raise ValueError(
                        "MDD is not properly layered: a state is reached at multiple depths"
                    )
                continue
            layers[dst] = expected
            queue.append(dst)

    if terminal not in layers:
        raise ValueError("MDD terminal is not reachable from the root")

    terminal_layer = layers[terminal]
    if terminal_layer != len(scope):
        raise ValueError(
            f"MDD depth ({terminal_layer}) must match scope length ({len(scope)})"
        )

    # Precompute domain values per layer
    domain_info = []
    for var in scope:
        values = _domain_values(var)
        domain_info.append((values, set(values)))

    layered_transitions: list[tuple[str, int, str]] = []
    for src, label, dst in reachable_transitions:
        if src not in layers or dst not in layers:
            continue
        src_layer = layers[src]
        dst_layer = layers[dst]
        if dst_layer != src_layer + 1:
            raise ValueError("MDD transitions must go from layer i to layer i+1")
        if src_layer >= len(scope):
            raise ValueError("MDD contains transitions beyond the scope length")

        domain_values, domain_set = domain_info[src_layer]
        symbols = _expand_label(label, domain_values, domain_set)
        for symbol in symbols:
            layered_transitions.append((src, symbol, dst))

    if not layered_transitions:
        raise ValueError("MDD decomposition produced no valid transitions")

    return root, terminal, layered_transitions
