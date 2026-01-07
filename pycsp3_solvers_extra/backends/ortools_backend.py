"""
OR-Tools CP-SAT backend for pycsp3-solvers-extra.

This module implements the ORToolsCallbacks class that translates
XCSP3 constraints to OR-Tools CP-SAT model.
"""

from __future__ import annotations

from typing import Any

from ortools.sat.python import cp_model

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
)
from pycsp3.classes.main.variables import Variable
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

    # ========== Variable creation ==========

    def var_integer_range(self, x: Variable, min_value: int, max_value: int):
        """Create integer variable with range domain."""
        var = self.model.NewIntVar(min_value, max_value, x.id)
        self.vars[x.id] = var
        self._log(2, f"Created var {x.id} in [{min_value}, {max_value}]")

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

    def _and(self, args: list[Any]) -> Any:
        """Logical AND."""
        if len(args) == 1:
            return args[0]
        # Create boolean result
        result = self.model.NewBoolVar("")
        self.model.AddBoolAnd(args).OnlyEnforceIf(result)
        self.model.AddBoolOr([a.Not() for a in args]).OnlyEnforceIf(result.Not())
        return result

    def _or(self, args: list[Any]) -> Any:
        """Logical OR."""
        if len(args) == 1:
            return args[0]
        result = self.model.NewBoolVar("")
        self.model.AddBoolOr(args).OnlyEnforceIf(result)
        self.model.AddBoolAnd([a.Not() for a in args]).OnlyEnforceIf(result.Not())
        return result

    def _not(self, a: Any) -> Any:
        """Logical NOT."""
        return a.Not()

    def _xor(self, args: list[Any]) -> Any:
        """Logical XOR (odd number of true)."""
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
        result = self.model.NewBoolVar("")
        # a == b
        self.model.Add(a == b).OnlyEnforceIf(result)
        self.model.Add(a != b).OnlyEnforceIf(result.Not())
        return result

    def _imp(self, a: Any, b: Any) -> Any:
        """Logical implication (a => b)."""
        result = self.model.NewBoolVar("")
        # a => b is equivalent to !a or b
        self.model.AddImplication(a, b).OnlyEnforceIf(result)
        # For the negation: a and !b
        self.model.AddBoolAnd([a, b.Not()]).OnlyEnforceIf(result.Not())
        return result

    def _if_then_else(self, cond: Any, then_val: Any, else_val: Any) -> Any:
        """Ternary if-then-else."""
        result = self.model.NewIntVar(-10**9, 10**9, "")
        # result == then_val if cond else else_val
        self.model.Add(result == then_val).OnlyEnforceIf(cond)
        self.model.Add(result == else_val).OnlyEnforceIf(cond.Not())
        return result

    def _in_set(self, val: Any, set_vals: list[Any]) -> Any:
        """Set membership."""
        if all(isinstance(v, int) for v in set_vals):
            result = self.model.NewBoolVar("")
            self.model.AddAllowedAssignments([val], [[v] for v in set_vals]).OnlyEnforceIf(result)
            self.model.AddForbiddenAssignments([val], [[v] for v in set_vals]).OnlyEnforceIf(result.Not())
            return result
        # Variable set - use OR of equalities
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
        if positive:
            self.model.AddAllowedAssignments([var], [[v] for v in values])
        else:
            self.model.AddForbiddenAssignments([var], [[v] for v in values])
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
        exprs = self._get_var_or_node_list(lst)

        # Build sum expression
        if coefficients is None:
            sum_expr = sum(exprs)
        else:
            if all(isinstance(c, int) for c in coefficients):
                sum_expr = sum(c * e for c, e in zip(coefficients, exprs))
            else:
                # Variable coefficients
                terms = []
                for c, e in zip(coefficients, exprs):
                    if isinstance(c, Variable):
                        c = self.vars[c.id]
                    terms.append(self._mul(c, e))
                sum_expr = sum(terms)

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

        count_sum = sum(count_vars)
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
        self.model.Add(sum(count_vars) >= k)
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
        self.model.Add(sum(count_vars) <= k)
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
            self.model.Add(sum(count_vars) == k_var)
        else:
            self.model.Add(sum(count_vars) == k)
        self._log(2, f"Added Exactly constraint: exactly {k} of value {value}")

    def ctr_nvalues(self, lst: list[Variable] | list[Node], excepting: None | list[int], condition: Condition):
        """NValues constraint: number of distinct values."""
        # This is complex - we need to count distinct values
        # For now, use a simple encoding
        exprs = self._get_var_or_node_list(lst)

        # Get all possible values
        all_values = set()
        for e in exprs:
            if hasattr(e, 'Proto'):
                # It's a variable - get its domain
                for i in range(e.Proto().domain_size() // 2):
                    lb = e.Proto().domain[2 * i]
                    ub = e.Proto().domain[2 * i + 1]
                    all_values.update(range(lb, ub + 1))

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

        nvalues = sum(appears)
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

            # Create interval variable
            interval = self.model.NewIntervalVar(
                start, length, start + length, f"interval_{i}"
            )
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

            interval = self.model.NewIntervalVar(
                start, length, start + length, f"interval_{i}"
            )
            intervals.append(interval)

            if isinstance(heights[i], int):
                demands.append(heights[i])
            else:
                demands.append(self.vars[heights[i].id])

        # Get capacity from condition (should be LE with constant)
        capacity = condition.right_operand
        if isinstance(capacity, Variable):
            capacity = self.vars[capacity.id]

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
        for i in range(len(lists) - 1):
            vars1 = self._get_var_list(lists[i])
            vars2 = self._get_var_list(lists[i + 1])

            if operator in (TypeOrderedOperator.INCREASING, TypeOrderedOperator.STRICTLY_INCREASING):
                self.model.AddLexLess(vars1, vars2) if operator == TypeOrderedOperator.STRICTLY_INCREASING else self.model.AddLexLessOrEqual(vars1, vars2)
            else:
                self.model.AddLexLess(vars2, vars1) if operator == TypeOrderedOperator.STRICTLY_DECREASING else self.model.AddLexLessOrEqual(vars2, vars1)

        self._log(2, f"Added Lex constraint on {len(lists)} lists")

    def ctr_instantiation(self, lst: list[Variable], values: list[int]):
        """Instantiation constraint (fix variables to values)."""
        for var, val in zip(lst, values):
            self.model.Add(self.vars[var.id] == val)
        self._log(2, f"Added Instantiation constraint on {len(lst)} vars")

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
        self.model.Minimize(expr)
        self._log(1, "Set minimization objective")

    def obj_maximize(self, term: Variable | Node):
        """Maximize objective."""
        if isinstance(term, Variable):
            expr = self.vars[term.id]
        else:
            expr = self.translate_node(term)
        self.model.Maximize(expr)
        self._log(1, "Set maximization objective")

    def obj_minimize_special(self, obj_type: TypeObj, terms: list[Variable] | list[Node], coefficients: None | list[int]):
        """Minimize special objective (sum, product, etc.)."""
        exprs = self._get_var_or_node_list(terms)

        if obj_type == TypeObj.SUM:
            if coefficients is None:
                obj_expr = sum(exprs)
            else:
                obj_expr = sum(c * e for c, e in zip(coefficients, exprs))
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
        exprs = self._get_var_or_node_list(terms)

        if obj_type == TypeObj.SUM:
            if coefficients is None:
                obj_expr = sum(exprs)
            else:
                obj_expr = sum(c * e for c, e in zip(coefficients, exprs))
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
        self._log(1, "Starting OR-Tools solver...")
        status = self.solver.Solve(self.model)

        if status == cp_model.OPTIMAL:
            self._extract_solution()
            self._status = TypeStatus.OPTIMUM
        elif status == cp_model.FEASIBLE:
            self._extract_solution()
            self._status = TypeStatus.SAT
        elif status == cp_model.INFEASIBLE:
            self._status = TypeStatus.UNSAT
        else:
            self._status = TypeStatus.UNKNOWN

        self._log(1, f"Solver finished with status: {self._status}")
        return self._status

    def _solve_all_solutions(self) -> TypeStatus:
        """Solve and enumerate all/multiple solutions."""
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
                in_range = self.model.NewBoolVar("")
                self.model.Add(expr >= condition.min).OnlyEnforceIf(in_range)
                self.model.Add(expr <= condition.max).OnlyEnforceIf(in_range)
                self.model.Add(in_range == 0)
            return
        elif isinstance(condition, ConditionSet):
            # Set condition
            values = list(condition.t)
            if op == TypeConditionOperator.IN:
                self.model.AddAllowedAssignments([expr], [[v] for v in values])
            else:  # NOTIN
                self.model.AddForbiddenAssignments([expr], [[v] for v in values])
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
                self.model.AddAllowedAssignments([expr], [[v] for v in right])
        elif op == TypeConditionOperator.NOTIN:
            if isinstance(right, range):
                # Not in range: expr < start OR expr >= stop
                in_range = self.model.NewBoolVar("")
                self.model.Add(expr >= right.start).OnlyEnforceIf(in_range)
                self.model.Add(expr < right.stop).OnlyEnforceIf(in_range)
                self.model.Add(in_range == 0)
            else:
                self.model.AddForbiddenAssignments([expr], [[v] for v in right])
        else:
            raise NotImplementedError(f"Condition operator {op} not implemented")
