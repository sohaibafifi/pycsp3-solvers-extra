"""
OR-Tools CP-SAT backend for pycsp3-solvers-extra.

This module implements the ORToolsCallbacks class that translates
XCSP3 constraints to OR-Tools CP-SAT model.
"""

from __future__ import annotations

from typing import Any

from ortools.sat.python import cp_model
import ortools.sat.python.cp_model_helper as cmh

from pycsp3.classes.auxiliary.conditions import Condition
from pycsp3.classes.auxiliary.enums import (
    TypeConditionOperator,
    TypeArithmeticOperator,
    TypeUnaryArithmeticOperator,
    TypeLogicalOperator,
    TypeOrderedOperator,
    TypeRank,
    TypeObj,
    TypeStatus,
    TypeVar,
)
from pycsp3.classes.main.variables import Variable, VariableInteger, Domain
from pycsp3.classes.nodes import Node, TypeNode

from pycsp3_solvers_extra.backends.base import BaseCallbacks


class ORToolsCallbacks(BaseCallbacks):
    """
    OR-Tools CP-SAT backend using callback-based XCSP3 parsing.

    Translates XCSP3 elements to OR-Tools CP-SAT constraints.
    """

    def __init__(
        self,
        time_limit: float | None = None,
        sols: int | str | None = None,
        verbose: int = 0,
        options: str = "",
    ):
        super().__init__(time_limit, sols, verbose, options)

        # OR-Tools model and solver
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()

        # Track interval variables for scheduling constraints
        self.intervals: dict[str, cp_model.IntervalVar] = {}

        # For multi-solution enumeration
        self._all_solutions: list[dict[str, int]] = []
        self._bounds_cache: dict[int, tuple[int, int]] = {}

    # ========== Variable creation ==========

    def var_integer_range(self, x: Variable, min_value: int, max_value: int):
        """Create integer variable with range domain."""
        var = self.model.NewIntVar(min_value, max_value, x.id)
        self.vars[x.id] = var
        self._log(2, f"Created var {x.id} in [{min_value}, {max_value}]")

    def new_aux_int_var(self, lb: int, ub: int, name_hint: str = "aux") -> Any:
        """Create an auxiliary integer variable."""
        return self.model.NewIntVar(lb, ub, name_hint)

    def decompose_call(self, call, ctx):
        if call.name == "ctr_among":
            lst, values, k = call.args
            condition = Condition.build_condition((TypeConditionOperator.EQ, k))
            from pycsp3_solvers_extra.transforms.types import ConstraintCall

            return [ConstraintCall("ctr_count", (lst, values, condition), {})]
        if call.name == "ctr_cardinality":
            lst, values, occurs, closed = call.args
            from pycsp3_solvers_extra.transforms.types import ConstraintCall

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
        if call.name == "ctr_element_matrix":
            matrix, row_index, col_index, condition = call.args
            from pycsp3_solvers_extra.transforms.types import ConstraintCall

            if not matrix or not matrix[0]:
                raise ValueError("Element matrix must be non-empty")
            cols = len(matrix[0])
            if any(len(row) != cols for row in matrix):
                raise ValueError("Element matrix must be rectangular for decomposition")

            flat = [cell for row in matrix for cell in row]
            max_index = len(flat) - 1
            counter = getattr(self, "_aux_counter", 0)
            while True:
                aux_id = f"__aux_idx_{counter}"
                counter += 1
                if aux_id not in self.vars:
                    break
            self._aux_counter = counter
            aux_var = VariableInteger(aux_id, Domain(range(max_index + 1)))
            self.vars[aux_id] = self.new_aux_int_var(0, max_index, aux_id)

            expr = Node.build(TypeNode.ADD, Node.build(TypeNode.MUL, row_index, cols), col_index)
            tree = Node.build(TypeNode.EQ, aux_var, expr)
            scope = list(tree.scope())

            return [
                ConstraintCall("ctr_intension", (scope, tree), {}),
                ConstraintCall("ctr_element", (flat, aux_var, condition), {}),
            ]
        return None

    def var_integer(self, x: Variable, values: list[int]):
        """Create integer variable with enumerated domain."""
        domain = cp_model.Domain.FromValues(values)
        var = self.model.NewIntVarFromDomain(domain, x.id)
        self.vars[x.id] = var
        self._log(2, f"Created var {x.id} with domain {values[:5]}{'...' if len(values) > 5 else ''}")

    def var_symbolic(self, x: Variable, values: list[str]):
        """Create symbolic variable (mapped to integers)."""
        # Map symbolic values to integers 0, 1, 2, ...
        var = self.model.NewIntVar(0, len(values) - 1, x.id)
        self.vars[x.id] = var
        self._log(2, f"Created symbolic var {x.id} with {len(values)} values")

    # ========== Bound helpers ==========

    def _bounds_from_values(self, values) -> tuple[int, int]:
        if isinstance(values, range):
            return values.start, values.stop - 1
        if isinstance(values, (list, tuple, set)):
            values = list(values)
            if not values:
                return 0, 0
            if isinstance(values[0], str):
                return 0, len(values) - 1
            return min(values), max(values)
        if isinstance(values, int):
            return values, values
        return 0, 0

    def _div_bounds(self, lb1: int, ub1: int, lb2: int, ub2: int) -> tuple[int, int]:
        max_num_abs = max(abs(lb1), abs(ub1))
        if lb2 <= 0 <= ub2:
            if lb2 == 0 and ub2 == 0:
                return 0, 0
            min_abs_den = 1
        else:
            min_abs_den = min(abs(lb2), abs(ub2))
            if min_abs_den == 0:
                min_abs_den = 1
        max_q = max_num_abs // min_abs_den
        return -max_q, max_q

    def _mod_bounds(self, lb2: int, ub2: int) -> tuple[int, int]:
        max_abs = max(abs(lb2), abs(ub2))
        if max_abs == 0:
            return 0, 0
        max_mod = max_abs - 1
        return -max_mod, max_mod

    def _node_bounds(self, node: Node) -> tuple[int, int]:
        cache_key = id(node)
        if cache_key in self._bounds_cache:
            return self._bounds_cache[cache_key]

        if node.type.is_predicate_operator():
            bounds = (0, 1)
        elif node.type == TypeNode.VAR:
            dom = node.cnt.dom
            if dom.type == TypeVar.SYMBOLIC:
                values = dom.all_values()
                bounds = (0, len(values) - 1) if values else (0, 0)
            else:
                bounds = (dom.smallest_value(), dom.greatest_value())
        elif node.type == TypeNode.INT:
            bounds = (node.cnt, node.cnt)
        elif node.type == TypeNode.SYMBOL:
            bounds = (node.cnt, node.cnt) if isinstance(node.cnt, int) else (0, 0)
        elif node.type == TypeNode.NEG:
            lb, ub = self._node_bounds(node.cnt[0])
            bounds = (-ub, -lb)
        elif node.type == TypeNode.ABS:
            lb, ub = self._node_bounds(node.cnt[0])
            max_abs = max(abs(lb), abs(ub))
            if lb <= 0 <= ub:
                bounds = (0, max_abs)
            else:
                bounds = (min(abs(lb), abs(ub)), max_abs)
        elif node.type == TypeNode.ADD:
            lbs, ubs = zip(*(self._node_bounds(c) for c in node.cnt))
            bounds = (sum(lbs), sum(ubs))
        elif node.type == TypeNode.SUB:
            if len(node.cnt) == 1:
                lb, ub = self._node_bounds(node.cnt[0])
                bounds = (-ub, -lb)
            else:
                lb1, ub1 = self._node_bounds(node.cnt[0])
                lb2, ub2 = self._node_bounds(node.cnt[1])
                bounds = (lb1 - ub2, ub1 - lb2)
        elif node.type == TypeNode.MUL:
            lb, ub = 1, 1
            for child in node.cnt:
                clb, cub = self._node_bounds(child)
                candidates = (lb * clb, lb * cub, ub * clb, ub * cub)
                lb, ub = min(candidates), max(candidates)
            bounds = (lb, ub)
        elif node.type == TypeNode.DIV:
            lb1, ub1 = self._node_bounds(node.cnt[0])
            lb2, ub2 = self._node_bounds(node.cnt[1])
            bounds = self._div_bounds(lb1, ub1, lb2, ub2)
        elif node.type == TypeNode.MOD:
            lb2, ub2 = self._node_bounds(node.cnt[1])
            bounds = self._mod_bounds(lb2, ub2)
        elif node.type == TypeNode.MIN:
            lbs, ubs = zip(*(self._node_bounds(c) for c in node.cnt))
            bounds = (min(lbs), min(ubs))
        elif node.type == TypeNode.MAX:
            lbs, ubs = zip(*(self._node_bounds(c) for c in node.cnt))
            bounds = (max(lbs), max(ubs))
        elif node.type == TypeNode.DIST:
            lb1, ub1 = self._node_bounds(node.cnt[0])
            lb2, ub2 = self._node_bounds(node.cnt[1])
            if ub1 < lb2:
                min_abs = lb2 - ub1
            elif ub2 < lb1:
                min_abs = lb1 - ub2
            else:
                min_abs = 0
            max_abs = max(abs(lb1 - ub2), abs(ub1 - lb2))
            bounds = (min_abs, max_abs)
        elif node.type == TypeNode.IF:
            lb_then, ub_then = self._node_bounds(node.cnt[1])
            lb_else, ub_else = self._node_bounds(node.cnt[2])
            bounds = (min(lb_then, lb_else), max(ub_then, ub_else))
        else:
            bounds = self._bounds_from_values(node.possible_values())

        self._bounds_cache[cache_key] = bounds
        return bounds

    def _as_int_expr(self, expr: Any) -> Any:
        if isinstance(expr, cmh.BoundedLinearExpression):
            return self._as_bool_var(expr)
        return expr

    def translate_node(self, node: Node) -> Any:
        if node.type == TypeNode.ABS:
            expr = self._as_int_expr(self.translate_node(node.cnt[0]))
            lb, ub = self._node_bounds(node)
            result = self.model.NewIntVar(lb, ub, "")
            self.model.AddAbsEquality(result, expr)
            return result
        if node.type == TypeNode.DIST:
            left = self._as_int_expr(self.translate_node(node.cnt[0]))
            right = self._as_int_expr(self.translate_node(node.cnt[1]))
            lb, ub = self._node_bounds(node)
            result = self.model.NewIntVar(lb, ub, "")
            self.model.AddAbsEquality(result, left - right)
            return result
        if node.type == TypeNode.MIN:
            exprs = [self._as_int_expr(self.translate_node(s)) for s in node.cnt]
            lb, ub = self._node_bounds(node)
            result = self.model.NewIntVar(lb, ub, "")
            self.model.AddMinEquality(result, exprs)
            return result
        if node.type == TypeNode.MAX:
            exprs = [self._as_int_expr(self.translate_node(s)) for s in node.cnt]
            lb, ub = self._node_bounds(node)
            result = self.model.NewIntVar(lb, ub, "")
            self.model.AddMaxEquality(result, exprs)
            return result
        if node.type == TypeNode.DIV:
            left = self._as_int_expr(self.translate_node(node.cnt[0]))
            right = self._as_int_expr(self.translate_node(node.cnt[1]))
            lb, ub = self._node_bounds(node)
            result = self.model.NewIntVar(lb, ub, "")
            self.model.AddDivisionEquality(result, left, right)
            return result
        if node.type == TypeNode.MOD:
            left = self._as_int_expr(self.translate_node(node.cnt[0]))
            right = self._as_int_expr(self.translate_node(node.cnt[1]))
            lb, ub = self._node_bounds(node)
            result = self.model.NewIntVar(lb, ub, "")
            self.model.AddModuloEquality(result, left, right)
            return result
        if node.type == TypeNode.MUL:
            exprs = [self._as_int_expr(self.translate_node(s)) for s in node.cnt]
            lb, ub = self._node_bounds(node)
            result = self.model.NewIntVar(lb, ub, "")
            self.model.AddMultiplicationEquality(result, exprs)
            return result
        if node.type == TypeNode.IF:
            cond = self._as_bool_var(self.translate_node(node.cnt[0]))
            then_val = self._as_int_expr(self.translate_node(node.cnt[1]))
            else_val = self._as_int_expr(self.translate_node(node.cnt[2]))
            lb, ub = self._node_bounds(node)
            result = self.model.NewIntVar(lb, ub, "")
            self.model.Add(result == then_val).OnlyEnforceIf(cond)
            self.model.Add(result == else_val).OnlyEnforceIf(cond.Not())
            return result
        return super().translate_node(node)

    # ========== OR-Tools specific expression operations ==========

    def _abs(self, a: Any) -> Any:
        """Absolute value."""
        if isinstance(a, int):
            return abs(a)
        # Create auxiliary variable for |a|
        # Need to determine bounds
        result = self.model.NewIntVar(0, 10**9, "")
        self.model.AddAbsEquality(result, a)
        return result

    def _min(self, args: list[Any]) -> Any:
        """Minimum of expressions."""
        if all(isinstance(a, int) for a in args):
            return min(args)
        result = self.model.NewIntVar(-10**9, 10**9, "")
        self.model.AddMinEquality(result, args)
        return result

    def _max(self, args: list[Any]) -> Any:
        """Maximum of expressions."""
        if all(isinstance(a, int) for a in args):
            return max(args)
        result = self.model.NewIntVar(-10**9, 10**9, "")
        self.model.AddMaxEquality(result, args)
        return result

    def _div(self, a: Any, b: Any) -> Any:
        """Integer division."""
        if isinstance(a, int) and isinstance(b, int):
            return a // b
        result = self.model.NewIntVar(-10**9, 10**9, "")
        self.model.AddDivisionEquality(result, a, b)
        return result

    def _mod(self, a: Any, b: Any) -> Any:
        """Modulo."""
        if isinstance(a, int) and isinstance(b, int):
            return a % b
        result = self.model.NewIntVar(0, 10**9, "")
        self.model.AddModuloEquality(result, a, b)
        return result

    def _mul(self, a: Any, b: Any) -> Any:
        """Multiplication - handle variable * variable case."""
        if isinstance(a, cmh.BoundedLinearExpression):
            a = self._as_bool_var(a)
        if isinstance(b, cmh.BoundedLinearExpression):
            b = self._as_bool_var(b)
        if isinstance(a, int) and isinstance(b, int):
            return a * b
        if isinstance(a, int):
            return a * b  # OR-Tools handles scalar * var
        if isinstance(b, int):
            return b * a  # OR-Tools handles scalar * var
        # Variable * variable needs auxiliary
        result = self.model.NewIntVar(-10**18, 10**18, "")
        self.model.AddMultiplicationEquality(result, [a, b])
        return result

    def _as_bool_var(self, expr: Any) -> Any:
        """Return a BoolVar equivalent when expr is a comparison expression."""
        if hasattr(expr, "Not"):
            return expr
        if isinstance(expr, cmh.BoundedLinearExpression):
            result = self.model.NewBoolVar("")
            linear_expr = cp_model.LinearExpr.Sum([c * v for c, v in zip(expr.coeffs, expr.vars)])
            if expr.offset:
                linear_expr += expr.offset
            domain = expr.bounds
            if domain.is_empty():
                self.model.Add(result == 0)
                return result
            complement = domain.complement()
            if complement.is_empty():
                self.model.Add(result == 1)
                return result
            self.model.AddLinearExpressionInDomain(linear_expr, domain).OnlyEnforceIf(result)
            self.model.AddLinearExpressionInDomain(linear_expr, complement).OnlyEnforceIf(result.Not())
            return result
        return expr

    def _linear_sum(self, exprs: list[Any]) -> Any:
        """Build a linear sum from expressions, handling boolean expressions safely."""
        exprs = [self._as_bool_var(e) for e in exprs]
        if not exprs:
            return 0
        if len(exprs) == 1:
            return exprs[0]
        return cp_model.LinearExpr.Sum(exprs)

    def _weighted_sum(self, exprs: list[Any], coefficients: list[int]) -> Any:
        """Build a weighted sum from expressions, handling boolean expressions safely."""
        exprs = [self._as_bool_var(e) for e in exprs]
        if not exprs:
            return 0
        if len(exprs) == 1:
            return coefficients[0] * exprs[0]
        return cp_model.LinearExpr.WeightedSum(exprs, coefficients)

    def _and(self, args: list[Any]) -> Any:
        """Logical AND."""
        if len(args) == 1:
            return self._as_bool_var(args[0])
        args = [self._as_bool_var(a) for a in args]
        # Create boolean result
        result = self.model.NewBoolVar("")
        self.model.AddBoolAnd(args).OnlyEnforceIf(result)
        self.model.AddBoolOr([a.Not() for a in args]).OnlyEnforceIf(result.Not())
        return result

    def _or(self, args: list[Any]) -> Any:
        """Logical OR."""
        if len(args) == 1:
            return self._as_bool_var(args[0])
        args = [self._as_bool_var(a) for a in args]
        result = self.model.NewBoolVar("")
        self.model.AddBoolOr(args).OnlyEnforceIf(result)
        self.model.AddBoolAnd([a.Not() for a in args]).OnlyEnforceIf(result.Not())
        return result

    def _not(self, a: Any) -> Any:
        """Logical NOT."""
        return self._as_bool_var(a).Not()

    def _xor(self, args: list[Any]) -> Any:
        """Logical XOR (odd number of true)."""
        args = [self._as_bool_var(a) for a in args]
        if len(args) == 2:
            result = self.model.NewBoolVar("")
            # XOR: exactly one is true
            self.model.Add(args[0] + args[1] == 1).OnlyEnforceIf(result)
            self.model.Add(args[0] + args[1] != 1).OnlyEnforceIf(result.Not())
            return result
        # General case: odd number true
        result = self.model.NewBoolVar("")
        self.model.AddBoolXOr(args).OnlyEnforceIf(result)
        return result

    def _iff(self, a: Any, b: Any) -> Any:
        """Logical equivalence (a <=> b)."""
        a = self._as_bool_var(a)
        b = self._as_bool_var(b)
        result = self.model.NewBoolVar("")
        # a == b
        self.model.Add(a == b).OnlyEnforceIf(result)
        self.model.Add(a != b).OnlyEnforceIf(result.Not())
        return result

    def _imp(self, a: Any, b: Any) -> Any:
        """Logical implication (a => b)."""
        a = self._as_bool_var(a)
        b = self._as_bool_var(b)
        result = self.model.NewBoolVar("")
        # a => b is equivalent to !a or b
        self.model.AddImplication(a, b).OnlyEnforceIf(result)
        # For the negation: a and !b
        self.model.AddBoolAnd([a, b.Not()]).OnlyEnforceIf(result.Not())
        return result

    def _if_then_else(self, cond: Any, then_val: Any, else_val: Any) -> Any:
        """Ternary if-then-else."""
        cond = self._as_bool_var(cond)
        result = self.model.NewIntVar(-10**9, 10**9, "")
        # result == then_val if cond else else_val
        self.model.Add(result == then_val).OnlyEnforceIf(cond)
        self.model.Add(result == else_val).OnlyEnforceIf(cond.Not())
        return result

    def _in_set(self, val: Any, set_vals: list[Any]) -> Any:
        """Set membership."""
        # Use OR of equalities; works for both variables and linear expressions.
        result = self.model.NewBoolVar("")
        or_vars = []
        for v in set_vals:
            eq_var = self.model.NewBoolVar("")
            self.model.Add(val == v).OnlyEnforceIf(eq_var)
            self.model.Add(val != v).OnlyEnforceIf(eq_var.Not())
            or_vars.append(eq_var)
        self.model.AddBoolOr(or_vars).OnlyEnforceIf(result)
        self.model.AddBoolAnd([v.Not() for v in or_vars]).OnlyEnforceIf(result.Not())
        return result

    def _not_in_set(self, val: Any, set_vals: list[Any]) -> Any:
        """Set non-membership."""
        in_set = self._in_set(val, set_vals)
        return in_set.Not()

    # ========== Constraint callbacks ==========

    def ctr_intension(self, scope: list[Variable], tree: Node):
        """Add intension constraint (boolean expression tree)."""
        expr = self.translate_node(tree)
        # The expression should be a boolean - add as constraint
        if hasattr(expr, 'Not'):
            # It's a boolean variable
            self.model.Add(expr == 1)
        else:
            # It's a boolean expression (comparison result)
            self.model.Add(expr)
        self._log(2, f"Added intension constraint on {len(scope)} vars")

    def ctr_extension_unary(self, x: Variable, values: list[int], positive: bool, flags: set[str]):
        """Unary table constraint."""
        var = self.vars[x.id]
        expanded = []
        for v in values:
            if isinstance(v, range):
                expanded.extend(v)
            else:
                expanded.append(v)
        if positive:
            self.model.AddAllowedAssignments([var], [[v] for v in expanded])
        else:
            self.model.AddForbiddenAssignments([var], [[v] for v in expanded])
        self._log(2, f"Added {'positive' if positive else 'negative'} unary extension on {x.id}")

    def ctr_extension(self, scope: list[Variable], tuples: list, positive: bool, flags: set[str]):
        """Table constraint (allowed/forbidden assignments)."""
        vars_list = self._get_var_list(scope)
        if positive:
            self.model.AddAllowedAssignments(vars_list, tuples)
        else:
            self.model.AddForbiddenAssignments(vars_list, tuples)
        self._log(2, f"Added {'positive' if positive else 'negative'} extension on {len(scope)} vars with {len(tuples)} tuples")

    def ctr_all_different(self, scope: list[Variable] | list[Node], excepting: None | list[int]):
        """AllDifferent constraint."""
        exprs = self._get_var_or_node_list(scope)

        if excepting is None or len(excepting) == 0:
            self.model.AddAllDifferent(exprs)
        else:
            # AllDifferent except values - use pairwise constraints
            for i in range(len(exprs)):
                for j in range(i + 1, len(exprs)):
                    # xi != xj OR xi in excepting OR xj in excepting
                    not_equal = self.model.NewBoolVar("")
                    self.model.Add(exprs[i] != exprs[j]).OnlyEnforceIf(not_equal)

                    xi_excepted = self.model.NewBoolVar("")
                    self.model.AddAllowedAssignments([exprs[i]], [[v] for v in excepting]).OnlyEnforceIf(xi_excepted)

                    xj_excepted = self.model.NewBoolVar("")
                    self.model.AddAllowedAssignments([exprs[j]], [[v] for v in excepting]).OnlyEnforceIf(xj_excepted)

                    self.model.AddBoolOr([not_equal, xi_excepted, xj_excepted])

        self._log(2, f"Added AllDifferent on {len(scope)} vars" + (f" except {excepting}" if excepting else ""))

    def ctr_all_different_lists(self, lists: list[list[Variable]], excepting: None | list[list[int]]):
        """AllDifferent on lists (each list differs from others)."""
        # Each pair of lists must differ in at least one position
        for i in range(len(lists)):
            for j in range(i + 1, len(lists)):
                # lists[i] != lists[j]: at least one position differs
                diff_vars = []
                for k in range(len(lists[i])):
                    diff_var = self.model.NewBoolVar("")
                    self.model.Add(self.vars[lists[i][k].id] != self.vars[lists[j][k].id]).OnlyEnforceIf(diff_var)
                    diff_vars.append(diff_var)
                self.model.AddBoolOr(diff_vars)
        self._log(2, f"Added AllDifferent on {len(lists)} lists")

    def ctr_all_equal(self, scope: list[Variable] | list[Node], excepting: None | list[int]):
        """AllEqual constraint."""
        exprs = self._get_var_or_node_list(scope)

        if excepting is None or len(excepting) == 0:
            # All equal: x0 == x1 == x2 == ...
            for i in range(1, len(exprs)):
                self.model.Add(exprs[0] == exprs[i])
        else:
            # All equal except values - more complex
            # If xi not in excepting, then xi == xj for all xj not in excepting
            for i in range(len(exprs)):
                for j in range(i + 1, len(exprs)):
                    xi_excepted = self.model.NewBoolVar("")
                    self.model.AddAllowedAssignments([exprs[i]], [[v] for v in excepting]).OnlyEnforceIf(xi_excepted)

                    xj_excepted = self.model.NewBoolVar("")
                    self.model.AddAllowedAssignments([exprs[j]], [[v] for v in excepting]).OnlyEnforceIf(xj_excepted)

                    # If neither excepted, must be equal
                    both_not_excepted = self.model.NewBoolVar("")
                    self.model.AddBoolAnd([xi_excepted.Not(), xj_excepted.Not()]).OnlyEnforceIf(both_not_excepted)
                    self.model.Add(exprs[i] == exprs[j]).OnlyEnforceIf(both_not_excepted)

        self._log(2, f"Added AllEqual on {len(scope)} vars")

    def ctr_sum(self, lst: list[Variable] | list[Node], coefficients: None | list[int] | list[Variable], condition: Condition):
        """Sum constraint with condition."""
        exprs = [self._as_bool_var(e) for e in self._get_var_or_node_list(lst)]

        # Build sum expression
        if coefficients is None:
            sum_expr = self._linear_sum(exprs)
        else:
            if all(isinstance(c, int) for c in coefficients):
                sum_expr = self._weighted_sum(exprs, coefficients)
            else:
                # Variable coefficients
                terms = []
                for c, e in zip(coefficients, exprs):
                    if isinstance(c, Variable):
                        c = self.vars[c.id]
                    terms.append(self._mul(c, e))
                sum_expr = self._linear_sum(terms)

        # Apply condition
        self._apply_condition_to_model(sum_expr, condition)
        self._log(2, f"Added Sum constraint on {len(lst)} terms with condition {condition.operator}")

    def ctr_count(self, lst: list[Variable] | list[Node], values: list[int] | list[Variable], condition: Condition):
        """Count constraint: count occurrences of values in lst."""
        exprs = self._get_var_or_node_list(lst)

        # Convert values to ortools expressions if needed
        ortools_values = []
        for v in values:
            if isinstance(v, Variable):
                ortools_values.append(self.vars[v.id])
            else:
                ortools_values.append(v)

        # Create boolean for each expr being in values using proper reification
        count_vars = []
        for e in exprs:
            if len(ortools_values) == 1:
                # Single value - use direct equality reification
                val = ortools_values[0]
                in_values = self.model.NewBoolVar("")
                if isinstance(val, int):
                    self.model.Add(e == val).OnlyEnforceIf(in_values)
                    self.model.Add(e != val).OnlyEnforceIf(in_values.Not())
                else:
                    self.model.Add(e == val).OnlyEnforceIf(in_values)
                    self.model.Add(e != val).OnlyEnforceIf(in_values.Not())
            else:
                # Multiple values - OR of equalities
                eq_vars = []
                for val in ortools_values:
                    eq_var = self.model.NewBoolVar("")
                    if isinstance(val, int):
                        self.model.Add(e == val).OnlyEnforceIf(eq_var)
                        self.model.Add(e != val).OnlyEnforceIf(eq_var.Not())
                    else:
                        self.model.Add(e == val).OnlyEnforceIf(eq_var)
                        self.model.Add(e != val).OnlyEnforceIf(eq_var.Not())
                    eq_vars.append(eq_var)

                # in_values iff any eq_var is true (OR)
                in_values = self.model.NewBoolVar("")
                # Forward: if any eq_var is true, then in_values is true
                for eq_var in eq_vars:
                    self.model.AddImplication(eq_var, in_values)
                # Backward: if in_values is true, then at least one eq_var is true
                self.model.AddBoolOr(eq_vars).OnlyEnforceIf(in_values)
                # If not in_values, none of the eq_vars are true
                self.model.AddBoolAnd([v.Not() for v in eq_vars]).OnlyEnforceIf(in_values.Not())

            count_vars.append(in_values)

        count_sum = self._linear_sum(count_vars)
        self._apply_condition_to_model(count_sum, condition)
        self._log(2, f"Added Count constraint")

    def ctr_atleast(self, lst: list[Variable], value: int, k: int):
        """AtLeast constraint: at least k occurrences of value in lst."""
        exprs = self._get_var_list(lst)

        # Count how many variables equal the value
        count_vars = []
        for e in exprs:
            eq_var = self.model.NewBoolVar("")
            self.model.Add(e == value).OnlyEnforceIf(eq_var)
            self.model.Add(e != value).OnlyEnforceIf(eq_var.Not())
            count_vars.append(eq_var)

        # Sum of count_vars >= k
        self.model.Add(self._linear_sum(count_vars) >= k)
        self._log(2, f"Added AtLeast constraint: at least {k} of value {value}")

    def ctr_atmost(self, lst: list[Variable], value: int, k: int):
        """AtMost constraint: at most k occurrences of value in lst."""
        exprs = self._get_var_list(lst)

        # Count how many variables equal the value
        count_vars = []
        for e in exprs:
            eq_var = self.model.NewBoolVar("")
            self.model.Add(e == value).OnlyEnforceIf(eq_var)
            self.model.Add(e != value).OnlyEnforceIf(eq_var.Not())
            count_vars.append(eq_var)

        # Sum of count_vars <= k
        self.model.Add(self._linear_sum(count_vars) <= k)
        self._log(2, f"Added AtMost constraint: at most {k} of value {value}")

    def ctr_exactly(self, lst: list[Variable], value: int, k: int | Variable):
        """Exactly constraint: exactly k occurrences of value in lst."""
        exprs = self._get_var_list(lst)

        # Count how many variables equal the value
        count_vars = []
        for e in exprs:
            eq_var = self.model.NewBoolVar("")
            self.model.Add(e == value).OnlyEnforceIf(eq_var)
            self.model.Add(e != value).OnlyEnforceIf(eq_var.Not())
            count_vars.append(eq_var)

        # Sum of count_vars == k
        if isinstance(k, Variable):
            k_var = self.vars[k.id]
            self.model.Add(self._linear_sum(count_vars) == k_var)
        else:
            self.model.Add(self._linear_sum(count_vars) == k)
        self._log(2, f"Added Exactly constraint: exactly {k} of value {value}")

    def ctr_nvalues(self, lst: list[Variable] | list[Node], excepting: None | list[int], condition: Condition):
        """NValues constraint: number of distinct values."""
        # This is complex - we need to count distinct values
        # For now, use a simple encoding
        exprs = self._get_var_or_node_list(lst)

        # Get all possible values from pycsp3 objects
        all_values: set[int] = set()
        for item in lst:
            if isinstance(item, Variable):
                vals = item.dom.all_values()
            elif isinstance(item, Node):
                vals = item.possible_values()
            else:
                continue
            if isinstance(vals, range):
                all_values.update(vals)
            else:
                all_values.update(vals)

        if excepting:
            all_values -= set(excepting)

        # For each value, create boolean indicating if it appears
        appears = []
        for val in all_values:
            appears_var = self.model.NewBoolVar("")
            # val appears if any expr == val
            eq_vars = []
            for e in exprs:
                eq_var = self.model.NewBoolVar("")
                self.model.Add(e == val).OnlyEnforceIf(eq_var)
                self.model.Add(e != val).OnlyEnforceIf(eq_var.Not())
                eq_vars.append(eq_var)
            self.model.AddBoolOr(eq_vars).OnlyEnforceIf(appears_var)
            self.model.AddBoolAnd([v.Not() for v in eq_vars]).OnlyEnforceIf(appears_var.Not())
            appears.append(appears_var)

        nvalues = self._linear_sum(appears)
        self._apply_condition_to_model(nvalues, condition)
        self._log(2, f"Added NValues constraint")

    def ctr_element(self, lst: list[Variable] | list[int], i: Variable, condition: Condition):
        """Element constraint: lst[i] satisfies condition."""
        index_var = self.vars[i.id]

        if all(isinstance(v, int) for v in lst):
            # Constant array
            target = self.model.NewIntVar(min(lst), max(lst), "")
            self.model.AddElement(index_var, lst, target)
        else:
            # Variable array
            vars_list = [self.vars[v.id] if isinstance(v, Variable) else v for v in lst]
            target = self.model.NewIntVar(-10**9, 10**9, "")
            self.model.AddElement(index_var, vars_list, target)

        self._apply_condition_to_model(target, condition)
        self._log(2, f"Added Element constraint")

    def ctr_minimum(self, lst: list[Variable] | list[Node], condition: Condition):
        """Minimum constraint."""
        exprs = self._get_var_or_node_list(lst)
        min_var = self.model.NewIntVar(-10**9, 10**9, "")
        self.model.AddMinEquality(min_var, exprs)
        self._apply_condition_to_model(min_var, condition)
        self._log(2, f"Added Minimum constraint")

    def ctr_maximum(self, lst: list[Variable] | list[Node], condition: Condition):
        """Maximum constraint."""
        exprs = self._get_var_or_node_list(lst)
        max_var = self.model.NewIntVar(-10**9, 10**9, "")
        self.model.AddMaxEquality(max_var, exprs)
        self._apply_condition_to_model(max_var, condition)
        self._log(2, f"Added Maximum constraint")

    def ctr_channel(self, lst1: list[Variable], lst2: None | list[Variable]):
        """Channel/Inverse constraint."""
        vars1 = self._get_var_list(lst1)
        if lst2 is None:
            # Self-inverse: lst1[lst1[i]] = i
            self.model.AddInverse(vars1, vars1)
        else:
            vars2 = self._get_var_list(lst2)
            self.model.AddInverse(vars1, vars2)
        self._log(2, f"Added Channel constraint")

    def ctr_circuit(self, lst: list[Variable], size: None | int | Variable):
        """Circuit constraint (Hamiltonian cycle)."""
        vars_list = self._get_var_list(lst)
        n = len(vars_list)

        # Build arcs for AddCircuit
        arcs = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Create boolean: vars_list[i] == j
                    arc_var = self.model.NewBoolVar(f"arc_{i}_{j}")
                    self.model.Add(vars_list[i] == j).OnlyEnforceIf(arc_var)
                    self.model.Add(vars_list[i] != j).OnlyEnforceIf(arc_var.Not())
                    arcs.append((i, j, arc_var))

        self.model.AddCircuit(arcs)

        if size is not None:
            # Size constraint on circuit - not standard, handle if needed
            pass

        self._log(2, f"Added Circuit constraint on {n} nodes")

    def ctr_clause(self, pos: list[Variable], neg: list[Variable]):
        """Clause constraint (OR of positive and negated variables)."""
        literals = []
        for v in pos:
            literals.append(self.vars[v.id])
        for v in neg:
            literals.append(self.vars[v.id].Not())
        self.model.AddBoolOr(literals)
        self._log(2, f"Added Clause with {len(pos)} positive and {len(neg)} negative literals")

    def ctr_nooverlap(self, origins: list[Variable], lengths: list[int] | list[Variable], zero_ignored: bool):
        """NoOverlap constraint (1D)."""
        n = len(origins)
        intervals = []

        for i in range(n):
            start = self.vars[origins[i].id]
            if isinstance(lengths[i], int):
                length = lengths[i]
            else:
                length = self.vars[lengths[i].id]

            # Create interval variable (use explicit end var if length is variable)
            if isinstance(lengths[i], int):
                end = start + length
            else:
                end_lb = origins[i].dom.smallest_value() + lengths[i].dom.smallest_value()
                end_ub = origins[i].dom.greatest_value() + lengths[i].dom.greatest_value()
                end = self.model.NewIntVar(end_lb, end_ub, f"interval_{i}_end")
                self.model.Add(end == start + length)
            interval = self.model.NewIntervalVar(start, length, end, f"interval_{i}")
            intervals.append(interval)

        self.model.AddNoOverlap(intervals)
        self._log(2, f"Added NoOverlap constraint on {n} intervals")

    def ctr_cumulative(self, origins: list[Variable], lengths: list[int] | list[Variable],
                       heights: list[int] | list[Variable], condition: Condition):
        """Cumulative constraint."""
        n = len(origins)
        intervals = []
        demands = []

        for i in range(n):
            start = self.vars[origins[i].id]
            if isinstance(lengths[i], int):
                length = lengths[i]
            else:
                length = self.vars[lengths[i].id]

            if isinstance(lengths[i], int):
                end = start + length
            else:
                end_lb = origins[i].dom.smallest_value() + lengths[i].dom.smallest_value()
                end_ub = origins[i].dom.greatest_value() + lengths[i].dom.greatest_value()
                end = self.model.NewIntVar(end_lb, end_ub, f"interval_{i}_end")
                self.model.Add(end == start + length)
            interval = self.model.NewIntervalVar(start, length, end, f"interval_{i}")
            intervals.append(interval)

            if isinstance(heights[i], int):
                demands.append(heights[i])
            else:
                demands.append(self.vars[heights[i].id])

        from pycsp3.classes.auxiliary.conditions import ConditionValue, ConditionVariable

        # Get capacity from condition (supported: LE with constant/variable)
        if not isinstance(condition, (ConditionValue, ConditionVariable)) or condition.operator != TypeConditionOperator.LE:
            raise NotImplementedError("OR-Tools cumulative only supports LE capacity conditions")
        capacity = condition.value if isinstance(condition, ConditionValue) else self.vars[condition.variable.id]

        self.model.AddCumulative(intervals, demands, capacity)
        self._log(2, f"Added Cumulative constraint on {n} tasks")

    def ctr_ordered(self, lst: list[Variable], operator: TypeOrderedOperator, lengths: None | list[int] | list[Variable]):
        """Ordered constraint."""
        vars_list = self._get_var_list(lst)

        for i in range(len(vars_list) - 1):
            offset = 0
            if lengths is not None:
                if isinstance(lengths[i], int):
                    offset = lengths[i]
                else:
                    offset = self.vars[lengths[i].id]

            if operator == TypeOrderedOperator.STRICTLY_INCREASING:
                self.model.Add(vars_list[i] + offset < vars_list[i + 1])
            elif operator == TypeOrderedOperator.INCREASING:
                self.model.Add(vars_list[i] + offset <= vars_list[i + 1])
            elif operator == TypeOrderedOperator.STRICTLY_DECREASING:
                self.model.Add(vars_list[i] + offset > vars_list[i + 1])
            elif operator == TypeOrderedOperator.DECREASING:
                self.model.Add(vars_list[i] + offset >= vars_list[i + 1])

        self._log(2, f"Added Ordered constraint with {operator}")

    def ctr_lex(self, lists: list[list[Variable]], operator: TypeOrderedOperator):
        """Lexicographic constraint on lists."""
        # Encode lex with prefix equality reification; strict adds not-all-equal.
        def add_lex_pair(vars1: list[Any], vars2: list[Any], strict: bool) -> None:
            if len(vars1) != len(vars2):
                raise ValueError("Lex constraint requires lists of equal length")
            if not vars1:
                return

            def reify_eq(x, y):
                eq_var = self.model.NewBoolVar("")
                self.model.Add(x == y).OnlyEnforceIf(eq_var)
                self.model.Add(x != y).OnlyEnforceIf(eq_var.Not())
                return eq_var

            prefix_eq = None
            all_equal = None
            for i, (x, y) in enumerate(zip(vars1, vars2)):
                eq_var = reify_eq(x, y)
                if i == 0:
                    self.model.Add(x <= y)
                    prefix_eq = eq_var
                elif i < len(vars1) - 1:
                    self.model.Add(x <= y).OnlyEnforceIf(prefix_eq)
                    next_prefix = self.model.NewBoolVar("")
                    self.model.AddBoolAnd([prefix_eq, eq_var]).OnlyEnforceIf(next_prefix)
                    self.model.AddBoolOr([prefix_eq.Not(), eq_var.Not()]).OnlyEnforceIf(next_prefix.Not())
                    prefix_eq = next_prefix
                else:
                    self.model.Add(x <= y).OnlyEnforceIf(prefix_eq)
                    all_equal = self.model.NewBoolVar("")
                    self.model.AddBoolAnd([prefix_eq, eq_var]).OnlyEnforceIf(all_equal)
                    self.model.AddBoolOr([prefix_eq.Not(), eq_var.Not()]).OnlyEnforceIf(all_equal.Not())

            if strict:
                if all_equal is None:
                    all_equal = prefix_eq
                self.model.Add(all_equal == 0)

        increasing = operator in (TypeOrderedOperator.INCREASING, TypeOrderedOperator.STRICTLY_INCREASING)
        strict = operator in (TypeOrderedOperator.STRICTLY_INCREASING, TypeOrderedOperator.STRICTLY_DECREASING)

        for i in range(len(lists) - 1):
            vars1 = self._get_var_list(lists[i])
            vars2 = self._get_var_list(lists[i + 1])
            if increasing:
                add_lex_pair(vars1, vars2, strict)
            else:
                add_lex_pair(vars2, vars1, strict)

        self._log(2, f"Added Lex constraint on {len(lists)} lists")

    def ctr_lex_matrix(self, matrix: list[list[Variable]], operator: TypeOrderedOperator):
        """Lexicographic constraint on matrix rows."""
        self.ctr_lex(matrix, operator)

    def ctr_instantiation(self, lst: list[Variable], values: list[int]):
        """Instantiation constraint (fix variables to values)."""
        for var, val in zip(lst, values):
            self.model.Add(self.vars[var.id] == val)
        self._log(2, f"Added Instantiation constraint on {len(lst)} vars")

    # ========== Packing constraints ==========

    def _binpacking_values(self, lst: list[Variable], bin_count: int | None):
        values_set = set()
        for var in lst:
            vals = var.dom.all_values()
            if isinstance(vals, range):
                values_set.update(vals)
            else:
                values_set.update(vals)
        if not values_set:
            raise ValueError("BinPacking requires non-empty bin domains")
        values = sorted(values_set)
        if bin_count is not None and len(values) != bin_count:
            min_val = values[0]
            values = list(range(min_val, min_val + bin_count))
            allowed = [[v] for v in values]
            for var in lst:
                self.model.AddAllowedAssignments([self.vars[var.id]], allowed)
        return values

    def _binpacking_load_exprs(self, bins: list[Any], sizes: list[int], values: list[int]):
        if not all(isinstance(s, int) for s in sizes):
            raise NotImplementedError("OR-Tools binpacking only supports constant sizes")
        load_exprs = []
        for value in values:
            assign_vars = []
            for bin_var in bins:
                bvar = self.model.NewBoolVar("")
                self.model.Add(bin_var == value).OnlyEnforceIf(bvar)
                self.model.Add(bin_var != value).OnlyEnforceIf(bvar.Not())
                assign_vars.append(bvar)
            load_exprs.append(self._weighted_sum(assign_vars, sizes))
        return load_exprs

    def ctr_binpacking(self, lst: list[Variable], sizes: list[int], condition: Condition):
        """BinPacking with a single condition on each bin load."""
        bins = self._get_var_list(lst)
        values = self._binpacking_values(lst, None)
        load_exprs = self._binpacking_load_exprs(bins, sizes, values)
        for load_expr in load_exprs:
            self._apply_condition_to_model(load_expr, condition)
        self._log(2, "Added BinPacking constraint with condition")

    def ctr_binpacking_limits(self, lst: list[Variable], sizes: list[int], limits: list[int] | list[Variable]):
        """BinPacking with per-bin limits."""
        bins = self._get_var_list(lst)
        values = self._binpacking_values(lst, len(limits))
        load_exprs = self._binpacking_load_exprs(bins, sizes, values)
        for load_expr, limit in zip(load_exprs, limits):
            bound = self.vars[limit.id] if isinstance(limit, Variable) else int(limit)
            self.model.Add(load_expr <= bound)
        self._log(2, "Added BinPacking constraint with limits")

    def ctr_binpacking_loads(self, lst: list[Variable], sizes: list[int], loads: list[int] | list[Variable]):
        """BinPacking with explicit load variables."""
        bins = self._get_var_list(lst)
        values = self._binpacking_values(lst, len(loads))
        load_exprs = self._binpacking_load_exprs(bins, sizes, values)
        for load_expr, load in zip(load_exprs, loads):
            target = self.vars[load.id] if isinstance(load, Variable) else int(load)
            self.model.Add(load_expr == target)
        self._log(2, "Added BinPacking constraint with loads")

    def ctr_binpacking_conditions(self, lst: list[Variable], sizes: list[int], conditions: list[Condition]):
        """BinPacking with per-bin conditions."""
        bins = self._get_var_list(lst)
        values = self._binpacking_values(lst, len(conditions))
        load_exprs = self._binpacking_load_exprs(bins, sizes, values)
        for load_expr, condition in zip(load_exprs, conditions):
            self._apply_condition_to_model(load_expr, condition)
        self._log(2, "Added BinPacking constraint with conditions")

    # ========== Primitive constraints (optimized) ==========

    def ctr_primitive1a(self, x: Variable, op: TypeConditionOperator, k: int):
        """x op k"""
        var = self.vars[x.id]
        self._add_simple_comparison(var, op, k)

    def ctr_primitive1b(self, x: Variable, op: TypeConditionOperator, term: list[int] | range):
        """x in/notin set"""
        var = self.vars[x.id]
        values = list(term)
        if op == TypeConditionOperator.IN:
            self.model.AddAllowedAssignments([var], [[v] for v in values])
        else:
            self.model.AddForbiddenAssignments([var], [[v] for v in values])

    def ctr_primitive2b(self, x: Variable, aop: TypeArithmeticOperator, y: Variable, op: TypeConditionOperator, k: int):
        """x aop y op k"""
        vx = self.vars[x.id]
        vy = self.vars[y.id]

        if aop == TypeArithmeticOperator.ADD:
            expr = vx + vy
        elif aop == TypeArithmeticOperator.SUB:
            expr = vx - vy
        elif aop == TypeArithmeticOperator.MUL:
            expr = self._mul(vx, vy)
        elif aop == TypeArithmeticOperator.DIV:
            expr = self._div(vx, vy)
        elif aop == TypeArithmeticOperator.MOD:
            expr = self._mod(vx, vy)
        elif aop == TypeArithmeticOperator.DIST:
            expr = self._abs(vx - vy)
        else:
            raise NotImplementedError(f"Arithmetic operator {aop} not implemented")

        self._add_simple_comparison(expr, op, k)

    def ctr_primitive3(self, x: Variable, aop: TypeArithmeticOperator, y: Variable, op: TypeConditionOperator, z: Variable):
        """x aop y op z"""
        vx = self.vars[x.id]
        vy = self.vars[y.id]
        vz = self.vars[z.id]

        if aop == TypeArithmeticOperator.ADD:
            expr = vx + vy
        elif aop == TypeArithmeticOperator.SUB:
            expr = vx - vy
        elif aop == TypeArithmeticOperator.MUL:
            expr = self._mul(vx, vy)
        else:
            raise NotImplementedError(f"Arithmetic operator {aop} not implemented")

        self._add_simple_comparison(expr, op, vz)

    # ========== Objectives ==========

    def obj_minimize(self, term: Variable | Node):
        """Minimize objective."""
        if isinstance(term, Variable):
            expr = self.vars[term.id]
        else:
            expr = self.translate_node(term)
            expr = self._as_bool_var(expr)
        self.model.Minimize(expr)
        self._log(1, "Set minimization objective")

    def obj_maximize(self, term: Variable | Node):
        """Maximize objective."""
        if isinstance(term, Variable):
            expr = self.vars[term.id]
        else:
            expr = self.translate_node(term)
            expr = self._as_bool_var(expr)
        self.model.Maximize(expr)
        self._log(1, "Set maximization objective")

    def obj_minimize_special(self, obj_type: TypeObj, terms: list[Variable] | list[Node], coefficients: None | list[int]):
        """Minimize special objective (sum, product, etc.)."""
        exprs = [self._as_bool_var(e) for e in self._get_var_or_node_list(terms)]

        if obj_type == TypeObj.SUM:
            if coefficients is None:
                obj_expr = self._linear_sum(exprs)
            else:
                obj_expr = self._weighted_sum(exprs, coefficients)
        elif obj_type == TypeObj.MAXIMUM:
            obj_var = self.model.NewIntVar(-10**9, 10**9, "obj_max")
            self.model.AddMaxEquality(obj_var, exprs)
            obj_expr = obj_var
        elif obj_type == TypeObj.MINIMUM:
            obj_var = self.model.NewIntVar(-10**9, 10**9, "obj_min")
            self.model.AddMinEquality(obj_var, exprs)
            obj_expr = obj_var
        elif obj_type == TypeObj.NVALUES:
            # Number of distinct values - complex
            raise NotImplementedError("NVALUES objective not implemented")
        else:
            raise NotImplementedError(f"Objective type {obj_type} not implemented")

        self.model.Minimize(obj_expr)
        self._log(1, f"Set {obj_type} minimization objective")

    def obj_maximize_special(self, obj_type: TypeObj, terms: list[Variable] | list[Node], coefficients: None | list[int]):
        """Maximize special objective."""
        exprs = [self._as_bool_var(e) for e in self._get_var_or_node_list(terms)]

        if obj_type == TypeObj.SUM:
            if coefficients is None:
                obj_expr = self._linear_sum(exprs)
            else:
                obj_expr = self._weighted_sum(exprs, coefficients)
        elif obj_type == TypeObj.MAXIMUM:
            obj_var = self.model.NewIntVar(-10**9, 10**9, "obj_max")
            self.model.AddMaxEquality(obj_var, exprs)
            obj_expr = obj_var
        elif obj_type == TypeObj.MINIMUM:
            obj_var = self.model.NewIntVar(-10**9, 10**9, "obj_min")
            self.model.AddMinEquality(obj_var, exprs)
            obj_expr = obj_var
        else:
            raise NotImplementedError(f"Objective type {obj_type} not implemented")

        self.model.Maximize(obj_expr)
        self._log(1, f"Set {obj_type} maximization objective")

    # ========== Solving ==========

    def solve(self) -> TypeStatus:
        """Solve the model and return status."""
        # Configure solver
        if self.time_limit is not None:
            self.solver.parameters.max_time_in_seconds = self.time_limit

        if self.verbose >= 2:
            self.solver.parameters.log_search_progress = True

        # Handle multi-solution
        if self.sols == "all" or (isinstance(self.sols, int) and self.sols > 1):
            return self._solve_all_solutions()

        # Single solution
        self._objective_value = None
        self._log(1, "Starting OR-Tools solver...")
        status = self.solver.Solve(self.model)

        if status == cp_model.OPTIMAL:
            self._extract_solution()
            self._set_objective_value()
            self._status = TypeStatus.OPTIMUM
        elif status == cp_model.FEASIBLE:
            self._extract_solution()
            self._set_objective_value()
            self._status = TypeStatus.SAT
        elif status == cp_model.INFEASIBLE:
            self._status = TypeStatus.UNSAT
        else:
            self._status = TypeStatus.UNKNOWN

        self._log(1, f"Solver finished with status: {self._status}")
        return self._status

    def _solve_all_solutions(self) -> TypeStatus:
        """Solve and enumerate all/multiple solutions."""
        self._objective_value = None
        class SolutionCollector(cp_model.CpSolverSolutionCallback):
            def __init__(self, variables, limit, verbose):
                super().__init__()
                self.variables = variables
                self.solutions = []
                self.limit = limit
                self.verbose = verbose

            def on_solution_callback(self):
                sol = {var_id: self.Value(var) for var_id, var in self.variables.items()}
                self.solutions.append(sol)
                if self.verbose >= 1:
                    print(f"Found solution {len(self.solutions)}")
                if self.limit is not None and len(self.solutions) >= self.limit:
                    self.StopSearch()

        limit = None if self.sols == "all" else self.sols
        collector = SolutionCollector(self.vars, limit, self.verbose)

        self._log(1, "Starting OR-Tools solver (enumerating solutions)...")
        status = self.solver.Solve(self.model, collector)

        if len(collector.solutions) > 0:
            self._all_solutions = collector.solutions
            self._solution = collector.solutions[0]
            self._set_objective_value()
            self._status = TypeStatus.SAT
        elif status == cp_model.INFEASIBLE:
            self._status = TypeStatus.UNSAT
        else:
            self._status = TypeStatus.UNKNOWN

        self._log(1, f"Found {len(self._all_solutions)} solutions")
        return self._status

    def _extract_solution(self):
        """Extract solution from solver."""
        self._solution = {}
        for var_id, var in self.vars.items():
            self._solution[var_id] = self.solver.Value(var)

    def _set_objective_value(self):
        """Capture objective value from the solver when present."""
        if not self.model.HasObjective():
            self._objective_value = None
            return
        obj = self.solver.ObjectiveValue()
        if isinstance(obj, float) and obj.is_integer():
            obj = int(obj)
        self._objective_value = obj

    def get_solution(self) -> dict[str, int] | None:
        """Return the solution."""
        return self._solution

    def get_all_solutions(self) -> list[dict[str, int]]:
        """Return all solutions found."""
        return self._all_solutions

    # ========== Helper methods ==========

    def _apply_condition_to_model(self, expr: Any, condition: Condition):
        """Apply a condition to the model as a constraint."""
        from pycsp3.classes.auxiliary.conditions import (
            ConditionValue, ConditionVariable, ConditionInterval, ConditionSet, ConditionNode
        )

        op = condition.operator

        # Extract right operand based on condition type
        if isinstance(condition, ConditionValue):
            right = condition.value
        elif isinstance(condition, ConditionVariable):
            right = self.vars[condition.variable.id]
        elif isinstance(condition, ConditionInterval):
            # Range condition: expr in [min, max]
            if op == TypeConditionOperator.IN:
                self.model.Add(expr >= condition.min)
                self.model.Add(expr <= condition.max)
            else:  # NOTIN
                below = self.model.NewBoolVar("")
                above = self.model.NewBoolVar("")
                self.model.Add(expr <= condition.min - 1).OnlyEnforceIf(below)
                self.model.Add(expr >= condition.max + 1).OnlyEnforceIf(above)
                self.model.AddBoolOr([below, above])
            return
        elif isinstance(condition, ConditionSet):
            # Set condition
            values = list(condition.t)
            in_set = self._in_set(expr, values)
            if op == TypeConditionOperator.IN:
                self.model.Add(in_set == 1)
            else:  # NOTIN
                self.model.Add(in_set == 0)
            return
        elif isinstance(condition, ConditionNode):
            right = self.translate_node(condition.node)
        else:
            raise NotImplementedError(f"Condition type {type(condition)} not implemented")

        self._add_simple_comparison(expr, op, right)

    def _add_simple_comparison(self, expr: Any, op: TypeConditionOperator, right: Any):
        """Add a simple comparison constraint."""
        if op == TypeConditionOperator.EQ:
            self.model.Add(expr == right)
        elif op == TypeConditionOperator.NE:
            self.model.Add(expr != right)
        elif op == TypeConditionOperator.LT:
            self.model.Add(expr < right)
        elif op == TypeConditionOperator.LE:
            self.model.Add(expr <= right)
        elif op == TypeConditionOperator.GT:
            self.model.Add(expr > right)
        elif op == TypeConditionOperator.GE:
            self.model.Add(expr >= right)
        elif op == TypeConditionOperator.IN:
            # right is a set/range
            if isinstance(right, range):
                self.model.Add(expr >= right.start)
                self.model.Add(expr < right.stop)
            else:
                in_set = self._in_set(expr, list(right))
                self.model.Add(in_set == 1)
        elif op == TypeConditionOperator.NOTIN:
            if isinstance(right, range):
                # Not in range: expr < start OR expr >= stop
                below = self.model.NewBoolVar("")
                above = self.model.NewBoolVar("")
                self.model.Add(expr <= right.start - 1).OnlyEnforceIf(below)
                self.model.Add(expr >= right.stop).OnlyEnforceIf(above)
                self.model.AddBoolOr([below, above])
            else:
                in_set = self._in_set(expr, list(right))
                self.model.Add(in_set == 0)
        else:
            raise NotImplementedError(f"Condition operator {op} not implemented")
