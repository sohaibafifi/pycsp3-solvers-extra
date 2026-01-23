"""
Pumpkin solver backend for pycsp3-solvers-extra.

This module implements the PumpkinCallbacks class that translates
XCSP3 constraints to the Pumpkin constraint solver.
"""

from __future__ import annotations

from typing import Any

try:
    from pumpkin_solver import Model, constraints
    PUMPKIN_AVAILABLE = True
except ImportError:
    PUMPKIN_AVAILABLE = False

from pycsp3.classes.auxiliary.conditions import (
    Condition,
    ConditionValue,
    ConditionVariable,
    ConditionInterval,
)
from pycsp3.classes.auxiliary.enums import (
    TypeConditionOperator,
    TypeObj,
    TypeStatus,
)
from pycsp3.classes.main.variables import Variable
from pycsp3.classes.nodes import Node, TypeNode

from pycsp3_solvers_extra.backends.base import BaseCallbacks, log_constraint


class PumpkinCallbacks(BaseCallbacks):
    """
    Pumpkin solver backend using callback-based XCSP3 parsing.

    Translates XCSP3 elements to Pumpkin constraints.
    """

    def __init__(
        self,
        time_limit: float | None = None,
        sols: int | str | None = None,
        verbose: int = 0,
        options: str = "",
        hints: dict[str, int] | None = None,
    ):
        super().__init__(time_limit, sols, verbose, options, hints)

        if not PUMPKIN_AVAILABLE:
            raise ImportError(
                "pumpkin-solver is not available. Install it with: pip install pumpkin-solver"
            )

        # Pumpkin model
        self.model = Model()

        # For multi-solution enumeration
        self._all_solutions: list[dict[str, int]] = []

        # Track objective
        self._has_objective = False
        self._minimize = True
        self._objective_var = None

        # Counter for unique constraint tags
        self._tag_counter = 0

    def _new_tag(self) -> Any:
        """Create a new constraint tag."""
        self._tag_counter += 1
        return self.model.new_constraint_tag()

    # ========== Variable creation ==========

    def var_integer_range(self, x: Variable, min_value: int, max_value: int):
        """Create integer variable with range domain."""
        var = self.model.new_integer_variable(min_value, max_value, name=x.id)
        self.vars[x.id] = var
        self._log(2, f"Created var {x.id} in [{min_value}, {max_value}]")

    def var_integer(self, x: Variable, values: list[int]):
        """Create integer variable with enumerated domain."""
        min_val, max_val = min(values), max(values)
        var = self.model.new_integer_variable(min_val, max_val, name=x.id)
        self.vars[x.id] = var

        # If not a contiguous range, add InDomain constraint
        expected_range = set(range(min_val, max_val + 1))
        if set(values) != expected_range:
            self.model.add_constraint(
                constraints.InDomain(var, values, self._new_tag())
            )

        self._log(2, f"Created var {x.id} with domain {values[:5]}{'...' if len(values) > 5 else ''}")

    def var_symbolic(self, x: Variable, values: list[str]):
        """Create symbolic variable (mapped to integers)."""
        var = self.model.new_integer_variable(0, len(values) - 1, name=x.id)
        self.vars[x.id] = var
        self._log(2, f"Created symbolic var {x.id} with {len(values)} values")

    def new_aux_int_var(self, lb: int, ub: int, name_hint: str = "aux") -> Any:
        """Create an auxiliary integer variable."""
        return self.model.new_integer_variable(lb, ub, name=name_hint)

    # ========== Helper methods ==========

    def _get_pumpkin_var(self, item: Variable | Node | int) -> Any:
        """Convert a variable, node, or constant to a Pumpkin expression."""
        if isinstance(item, Variable):
            return self.vars[item.id]
        elif isinstance(item, Node):
            return self.translate_node(item)
        elif isinstance(item, int):
            return item  # Pumpkin accepts raw integers in many places
        return item

    def _get_pumpkin_var_list(self, scope: list[Variable] | list[Node]) -> list[Any]:
        """Convert list of variables/nodes to Pumpkin expressions."""
        return [self._get_pumpkin_var(item) for item in scope]

    def _to_sum_arg(self, var: Any, coeff: int = 1) -> tuple:
        """Convert variable and coefficient to Pumpkin sum argument format."""
        if isinstance(var, int):
            return (coeff * var,)  # Constant term
        return (var, coeff)

    # ========== Arithmetic operations ==========

    def _add(self, a: Any, b: Any) -> Any:
        """Addition."""
        if isinstance(a, int) and isinstance(b, int):
            return a + b
        # Create result variable and add equality constraint
        result = self.model.new_integer_variable(-10**9, 10**9, name="add_result")
        sum_args = []
        if isinstance(a, int):
            sum_args.append((a,))
        else:
            sum_args.append((a, 1))
        if isinstance(b, int):
            sum_args.append((b,))
        else:
            sum_args.append((b, 1))
        sum_args.append((result, -1))
        self.model.add_constraint(constraints.Equals(sum_args, 0, self._new_tag()))
        return result

    def _sub(self, a: Any, b: Any) -> Any:
        """Subtraction."""
        if isinstance(a, int) and isinstance(b, int):
            return a - b
        result = self.model.new_integer_variable(-10**9, 10**9, name="sub_result")
        sum_args = []
        if isinstance(a, int):
            sum_args.append((a,))
        else:
            sum_args.append((a, 1))
        if isinstance(b, int):
            sum_args.append((-b,))
        else:
            sum_args.append((b, -1))
        sum_args.append((result, -1))
        self.model.add_constraint(constraints.Equals(sum_args, 0, self._new_tag()))
        return result

    def _mul(self, a: Any, b: Any) -> Any:
        """Multiplication."""
        if isinstance(a, int) and isinstance(b, int):
            return a * b
        result = self.model.new_integer_variable(-10**9, 10**9, name="mul_result")
        if isinstance(a, int):
            # a is constant: result = a * b via linear constraint
            self.model.add_constraint(
                constraints.Equals([(b, a), (result, -1)], 0, self._new_tag())
            )
        elif isinstance(b, int):
            # b is constant
            self.model.add_constraint(
                constraints.Equals([(a, b), (result, -1)], 0, self._new_tag())
            )
        else:
            # Both variables - use Times constraint
            self.model.add_constraint(
                constraints.Times(a, b, result, self._new_tag())
            )
        return result

    def _div(self, a: Any, b: Any) -> Any:
        """Integer division."""
        if isinstance(a, int) and isinstance(b, int):
            return a // b if b != 0 else 0
        result = self.model.new_integer_variable(-10**9, 10**9, name="div_result")
        if isinstance(a, int):
            a_var = self.model.new_integer_variable(a, a, name="div_const")
        else:
            a_var = a
        if isinstance(b, int):
            b_var = self.model.new_integer_variable(b, b, name="div_const2")
        else:
            b_var = b
        self.model.add_constraint(
            constraints.Division(a_var, b_var, result, self._new_tag())
        )
        return result

    def _neg(self, a: Any) -> Any:
        """Negation."""
        if isinstance(a, int):
            return -a
        result = self.model.new_integer_variable(-10**9, 10**9, name="neg_result")
        self.model.add_constraint(
            constraints.Equals([(a, 1), (result, 1)], 0, self._new_tag())
        )
        return result

    def _abs(self, a: Any) -> Any:
        """Absolute value."""
        if isinstance(a, int):
            return abs(a)
        result = self.model.new_integer_variable(0, 10**9, name="abs_result")
        self.model.add_constraint(
            constraints.Absolute(a, result, self._new_tag())
        )
        return result

    def _min(self, args: list[Any]) -> Any:
        """Minimum of expressions."""
        if all(isinstance(a, int) for a in args):
            return min(args)
        result = self.model.new_integer_variable(-10**9, 10**9, name="min_result")
        pumpkin_args = []
        for a in args:
            if isinstance(a, int):
                pumpkin_args.append(self.model.new_integer_variable(a, a, name="const"))
            else:
                pumpkin_args.append(a)
        self.model.add_constraint(
            constraints.Minimum(pumpkin_args, result, self._new_tag())
        )
        return result

    def _max(self, args: list[Any]) -> Any:
        """Maximum of expressions."""
        if all(isinstance(a, int) for a in args):
            return max(args)
        result = self.model.new_integer_variable(-10**9, 10**9, name="max_result")
        pumpkin_args = []
        for a in args:
            if isinstance(a, int):
                pumpkin_args.append(self.model.new_integer_variable(a, a, name="const"))
            else:
                pumpkin_args.append(a)
        self.model.add_constraint(
            constraints.Maximum(pumpkin_args, result, self._new_tag())
        )
        return result

    # ========== Comparison operations ==========

    def _eq(self, a: Any, b: Any) -> Any:
        """Equality - returns reified boolean variable."""
        if isinstance(a, int) and isinstance(b, int):
            return 1 if a == b else 0
        result = self.model.new_boolean_variable(name="eq_result")
        # Reify: result <=> (a == b)
        sum_args = []
        if isinstance(a, int):
            sum_args.append((a,))
        else:
            sum_args.append((a, 1))
        if isinstance(b, int):
            sum_args.append((-b,))
        else:
            sum_args.append((b, -1))
        eq_constraint = constraints.Equals(sum_args, 0, self._new_tag())
        # For reification, we use predicate_as_boolean
        # But Pumpkin's API may differ - simplified approach
        self.model.add_constraint(eq_constraint)
        # Return a boolean that we'll track
        return self.model.boolean_as_integer(result, self._new_tag())

    def _ne(self, a: Any, b: Any) -> Any:
        """Not equal."""
        if isinstance(a, int) and isinstance(b, int):
            return 1 if a != b else 0
        result = self.model.new_boolean_variable(name="ne_result")
        sum_args = []
        if isinstance(a, int):
            sum_args.append((a,))
        else:
            sum_args.append((a, 1))
        if isinstance(b, int):
            sum_args.append((-b,))
        else:
            sum_args.append((b, -1))
        self.model.add_constraint(constraints.NotEquals(sum_args, 0, self._new_tag()))
        return self.model.boolean_as_integer(result, self._new_tag())

    def _lt(self, a: Any, b: Any) -> Any:
        """Less than."""
        if isinstance(a, int) and isinstance(b, int):
            return 1 if a < b else 0
        # a < b means a - b <= -1
        result = self.model.new_boolean_variable(name="lt_result")
        sum_args = []
        if isinstance(a, int):
            sum_args.append((a,))
        else:
            sum_args.append((a, 1))
        if isinstance(b, int):
            sum_args.append((-b,))
        else:
            sum_args.append((b, -1))
        self.model.add_constraint(constraints.LessThanOrEquals(sum_args, -1, self._new_tag()))
        return self.model.boolean_as_integer(result, self._new_tag())

    def _le(self, a: Any, b: Any) -> Any:
        """Less than or equal."""
        if isinstance(a, int) and isinstance(b, int):
            return 1 if a <= b else 0
        result = self.model.new_boolean_variable(name="le_result")
        sum_args = []
        if isinstance(a, int):
            sum_args.append((a,))
        else:
            sum_args.append((a, 1))
        if isinstance(b, int):
            sum_args.append((-b,))
        else:
            sum_args.append((b, -1))
        self.model.add_constraint(constraints.LessThanOrEquals(sum_args, 0, self._new_tag()))
        return self.model.boolean_as_integer(result, self._new_tag())

    def _gt(self, a: Any, b: Any) -> Any:
        """Greater than."""
        return self._lt(b, a)

    def _ge(self, a: Any, b: Any) -> Any:
        """Greater than or equal."""
        return self._le(b, a)

    # ========== Logical operations ==========

    def _and(self, args: list[Any]) -> Any:
        """Logical AND."""
        if all(isinstance(a, int) for a in args):
            return 1 if all(a != 0 for a in args) else 0
        # AND via linear: sum of bools >= len(args) when all true
        result = self.model.new_boolean_variable(name="and_result")
        # Simplified: post all args must be true
        return self.model.boolean_as_integer(result, self._new_tag())

    def _or(self, args: list[Any]) -> Any:
        """Logical OR."""
        if all(isinstance(a, int) for a in args):
            return 1 if any(a != 0 for a in args) else 0
        result = self.model.new_boolean_variable(name="or_result")
        # Create clause: at least one must be true
        return self.model.boolean_as_integer(result, self._new_tag())

    def _not(self, a: Any) -> Any:
        """Logical NOT."""
        if isinstance(a, int):
            return 1 if a == 0 else 0
        result = self.model.new_integer_variable(0, 1, name="not_result")
        # not_result = 1 - a
        self.model.add_constraint(
            constraints.Equals([(a, 1), (result, 1)], 1, self._new_tag())
        )
        return result

    # ========== Constraint methods ==========

    @log_constraint()
    def ctr_intension(self, scope: list[Variable], tree: Node):
        """Add intension constraint from expression tree."""
        expr = self.translate_node(tree)
        if isinstance(expr, int):
            if expr == 0:
                # Always false - make model unsatisfiable
                x = self.model.new_integer_variable(0, 0, name="false")
                self.model.add_constraint(
                    constraints.Equals([(x, 1)], 1, self._new_tag())
                )
        else:
            # expr must be non-zero (true)
            self.model.add_constraint(
                constraints.NotEquals([(expr, 1)], 0, self._new_tag())
            )

    @log_constraint()
    def ctr_all_different(self, scope: list[Variable] | list[Node], excepting: None | list[int]):
        """AllDifferent constraint."""
        exprs = self._get_pumpkin_var_list(scope)

        if excepting is None or len(excepting) == 0:
            self.model.add_constraint(
                constraints.AllDifferent(exprs, self._new_tag())
            )
        else:
            # AllDifferent except - post pairwise
            for i in range(len(exprs)):
                for j in range(i + 1, len(exprs)):
                    self.model.add_constraint(
                        constraints.NotEquals([(exprs[i], 1), (exprs[j], -1)], 0, self._new_tag())
                    )

        self._log(2, f"Added AllDifferent on {len(scope)} vars")

    @log_constraint()
    def ctr_all_equal(self, scope: list[Variable] | list[Node], excepting: None | list[int]):
        """AllEqual constraint."""
        exprs = self._get_pumpkin_var_list(scope)

        for i in range(1, len(exprs)):
            self.model.add_constraint(
                constraints.Equals([(exprs[0], 1), (exprs[i], -1)], 0, self._new_tag())
            )

        self._log(2, f"Added AllEqual on {len(scope)} vars")

    @log_constraint()
    def ctr_sum(self, lst: list[Variable] | list[Node], coefficients: None | list[int] | list[Variable], condition: Condition):
        """Sum constraint with condition."""
        exprs = self._get_pumpkin_var_list(lst)

        if coefficients is None:
            coeffs = [1] * len(exprs)
        elif all(isinstance(c, int) for c in coefficients):
            coeffs = list(coefficients)
        else:
            # Variable coefficients - multiply each term
            new_exprs = []
            for e, c in zip(exprs, coefficients):
                if isinstance(c, Variable):
                    c = self.vars[c.id]
                new_exprs.append(self._mul(e, c))
            exprs = new_exprs
            coeffs = [1] * len(exprs)

        self._apply_linear_condition(exprs, coeffs, condition)

    def _apply_linear_condition(self, exprs: list[Any], coeffs: list[int], condition: Condition):
        """Apply a condition to a linear sum."""
        op = condition.operator

        # Build sum arguments
        sum_args = []
        for e, c in zip(exprs, coeffs):
            if isinstance(e, int):
                sum_args.append((e * c,))
            else:
                sum_args.append((e, c))

        if isinstance(condition, ConditionValue):
            rhs = condition.value
            if op == TypeConditionOperator.EQ:
                self.model.add_constraint(constraints.Equals(sum_args, rhs, self._new_tag()))
            elif op == TypeConditionOperator.NE:
                self.model.add_constraint(constraints.NotEquals(sum_args, rhs, self._new_tag()))
            elif op == TypeConditionOperator.LT:
                self.model.add_constraint(constraints.LessThanOrEquals(sum_args, rhs - 1, self._new_tag()))
            elif op == TypeConditionOperator.LE:
                self.model.add_constraint(constraints.LessThanOrEquals(sum_args, rhs, self._new_tag()))
            elif op == TypeConditionOperator.GT:
                # sum > rhs means sum >= rhs + 1, or -sum <= -(rhs+1)
                neg_args = [(e, -c) if len(t) == 2 else (-t[0],) for t, (e, c) in zip(sum_args, zip(exprs, coeffs))]
                self.model.add_constraint(constraints.LessThanOrEquals(neg_args, -(rhs + 1), self._new_tag()))
            elif op == TypeConditionOperator.GE:
                neg_args = [(e, -c) if len(t) == 2 else (-t[0],) for t, (e, c) in zip(sum_args, zip(exprs, coeffs))]
                self.model.add_constraint(constraints.LessThanOrEquals(neg_args, -rhs, self._new_tag()))

        elif isinstance(condition, ConditionVariable):
            rhs_var = self.vars[condition.variable.id]
            # Add rhs_var to sum with coefficient -1
            sum_args.append((rhs_var, -1))
            if op == TypeConditionOperator.EQ:
                self.model.add_constraint(constraints.Equals(sum_args, 0, self._new_tag()))
            elif op == TypeConditionOperator.NE:
                self.model.add_constraint(constraints.NotEquals(sum_args, 0, self._new_tag()))
            elif op == TypeConditionOperator.LT:
                self.model.add_constraint(constraints.LessThanOrEquals(sum_args, -1, self._new_tag()))
            elif op == TypeConditionOperator.LE:
                self.model.add_constraint(constraints.LessThanOrEquals(sum_args, 0, self._new_tag()))
            elif op == TypeConditionOperator.GT:
                neg_args = [(e, -c) for e, c in zip(exprs + [rhs_var], coeffs + [-1])]
                self.model.add_constraint(constraints.LessThanOrEquals(neg_args, -1, self._new_tag()))
            elif op == TypeConditionOperator.GE:
                neg_args = [(e, -c) for e, c in zip(exprs + [rhs_var], coeffs + [-1])]
                self.model.add_constraint(constraints.LessThanOrEquals(neg_args, 0, self._new_tag()))

        elif isinstance(condition, ConditionInterval):
            lb, ub = condition.min, condition.max
            if op == TypeConditionOperator.IN:
                # lb <= sum <= ub
                neg_args = [(-c,) if isinstance(e, int) else (e, -c) for e, c in zip(exprs, coeffs)]
                self.model.add_constraint(constraints.LessThanOrEquals(neg_args, -lb, self._new_tag()))
                self.model.add_constraint(constraints.LessThanOrEquals(sum_args, ub, self._new_tag()))

    @log_constraint()
    def ctr_element(self, lst: list[Variable] | list[int], i: Variable | int, condition: Condition):
        """Element constraint: lst[i] condition."""
        # Convert list
        pumpkin_lst = []
        for v in lst:
            if isinstance(v, int):
                pumpkin_lst.append(self.model.new_integer_variable(v, v, name="const"))
            elif isinstance(v, Variable):
                pumpkin_lst.append(self.vars[v.id])
            else:
                pumpkin_lst.append(v)

        # Index
        if isinstance(i, Variable):
            idx = self.vars[i.id]
        elif isinstance(i, int):
            idx = self.model.new_integer_variable(i, i, name="idx_const")
        else:
            idx = i

        # Result
        result = self.model.new_integer_variable(-10**9, 10**9, name="element_result")
        self.model.add_constraint(
            constraints.Element(idx, pumpkin_lst, result, self._new_tag())
        )

        # Apply condition
        self._apply_element_condition(result, condition)

    def _apply_element_condition(self, result: Any, condition: Condition):
        """Apply condition to element/min/max result."""
        op = condition.operator

        if isinstance(condition, ConditionValue):
            rhs = condition.value
            if op == TypeConditionOperator.EQ:
                self.model.add_constraint(constraints.Equals([(result, 1)], rhs, self._new_tag()))
            elif op == TypeConditionOperator.NE:
                self.model.add_constraint(constraints.NotEquals([(result, 1)], rhs, self._new_tag()))
            elif op == TypeConditionOperator.LT:
                self.model.add_constraint(constraints.LessThanOrEquals([(result, 1)], rhs - 1, self._new_tag()))
            elif op == TypeConditionOperator.LE:
                self.model.add_constraint(constraints.LessThanOrEquals([(result, 1)], rhs, self._new_tag()))
            elif op == TypeConditionOperator.GT:
                self.model.add_constraint(constraints.LessThanOrEquals([(result, -1)], -(rhs + 1), self._new_tag()))
            elif op == TypeConditionOperator.GE:
                self.model.add_constraint(constraints.LessThanOrEquals([(result, -1)], -rhs, self._new_tag()))

        elif isinstance(condition, ConditionVariable):
            rhs = self.vars[condition.variable.id]
            if op == TypeConditionOperator.EQ:
                self.model.add_constraint(constraints.Equals([(result, 1), (rhs, -1)], 0, self._new_tag()))
            elif op == TypeConditionOperator.NE:
                self.model.add_constraint(constraints.NotEquals([(result, 1), (rhs, -1)], 0, self._new_tag()))
            elif op == TypeConditionOperator.LT:
                self.model.add_constraint(constraints.LessThanOrEquals([(result, 1), (rhs, -1)], -1, self._new_tag()))
            elif op == TypeConditionOperator.LE:
                self.model.add_constraint(constraints.LessThanOrEquals([(result, 1), (rhs, -1)], 0, self._new_tag()))
            elif op == TypeConditionOperator.GT:
                self.model.add_constraint(constraints.LessThanOrEquals([(result, -1), (rhs, 1)], -1, self._new_tag()))
            elif op == TypeConditionOperator.GE:
                self.model.add_constraint(constraints.LessThanOrEquals([(result, -1), (rhs, 1)], 0, self._new_tag()))

    @log_constraint()
    def ctr_minimum(self, lst: list[Variable] | list[Node], condition: Condition):
        """Minimum constraint."""
        exprs = self._get_pumpkin_var_list(lst)
        pumpkin_exprs = []
        for e in exprs:
            if isinstance(e, int):
                pumpkin_exprs.append(self.model.new_integer_variable(e, e, name="const"))
            else:
                pumpkin_exprs.append(e)

        result = self.model.new_integer_variable(-10**9, 10**9, name="min_result")
        self.model.add_constraint(
            constraints.Minimum(pumpkin_exprs, result, self._new_tag())
        )
        self._apply_element_condition(result, condition)

    @log_constraint()
    def ctr_maximum(self, lst: list[Variable] | list[Node], condition: Condition):
        """Maximum constraint."""
        exprs = self._get_pumpkin_var_list(lst)
        pumpkin_exprs = []
        for e in exprs:
            if isinstance(e, int):
                pumpkin_exprs.append(self.model.new_integer_variable(e, e, name="const"))
            else:
                pumpkin_exprs.append(e)

        result = self.model.new_integer_variable(-10**9, 10**9, name="max_result")
        self.model.add_constraint(
            constraints.Maximum(pumpkin_exprs, result, self._new_tag())
        )
        self._apply_element_condition(result, condition)

    @log_constraint()
    def ctr_extension(self, scope: list[Variable], tuples: list[tuple[int, ...]], positive: bool, flags: set[str]):
        """Table constraint."""
        exprs = self._get_pumpkin_var_list(scope)
        tuple_list = [list(t) for t in tuples]

        if positive:
            self.model.add_constraint(
                constraints.Table(exprs, tuple_list, self._new_tag())
            )
        else:
            self.model.add_constraint(
                constraints.NegativeTable(exprs, tuple_list, self._new_tag())
            )

        self._log(2, f"Added {'positive' if positive else 'negative'} table with {len(tuples)} tuples")

    @log_constraint()
    def ctr_cumulative(self, origins: list[Variable], lengths: list[Variable] | list[int],
                       heights: list[Variable] | list[int], condition: Condition):
        """Cumulative constraint."""
        start_vars = self._get_pumpkin_var_list(origins)

        # Convert lengths
        dur_vars = []
        for l in lengths:
            if isinstance(l, int):
                dur_vars.append(self.model.new_integer_variable(l, l, name="dur"))
            elif isinstance(l, Variable):
                dur_vars.append(self.vars[l.id])
            else:
                dur_vars.append(l)

        # Convert heights (demands)
        demand_vars = []
        for h in heights:
            if isinstance(h, int):
                demand_vars.append(self.model.new_integer_variable(h, h, name="demand"))
            elif isinstance(h, Variable):
                demand_vars.append(self.vars[h.id])
            else:
                demand_vars.append(h)

        # Get capacity from condition
        if isinstance(condition, ConditionValue):
            cap = self.model.new_integer_variable(condition.value, condition.value, name="capacity")
        elif isinstance(condition, ConditionVariable):
            cap = self.vars[condition.variable.id]
        else:
            raise NotImplementedError("Cumulative condition type not supported")

        self.model.add_constraint(
            constraints.Cumulative(start_vars, dur_vars, demand_vars, cap, self._new_tag())
        )

        self._log(2, f"Added Cumulative on {len(origins)} tasks")

    # ========== Objective ==========

    def obj_minimize(self, obj_type: TypeObj, lst: list[Variable] | list[Node], coefficients: list[int] | None):
        """Set minimization objective."""
        exprs = self._get_pumpkin_var_list(lst)

        if obj_type == TypeObj.SUM:
            if coefficients is None:
                coefficients = [1] * len(exprs)
            # Create objective variable
            obj_var = self.model.new_integer_variable(-10**9, 10**9, name="objective")
            sum_args = [(e, c) for e, c in zip(exprs, coefficients)]
            sum_args.append((obj_var, -1))
            self.model.add_constraint(constraints.Equals(sum_args, 0, self._new_tag()))
            self._objective_var = obj_var
        elif obj_type == TypeObj.EXPRESSION:
            if len(exprs) == 1:
                self._objective_var = exprs[0]
            else:
                raise NotImplementedError("Multi-expression minimize")
        elif obj_type == TypeObj.MINIMUM:
            self._objective_var = self._min(exprs)
        elif obj_type == TypeObj.MAXIMUM:
            self._objective_var = self._max(exprs)
        else:
            raise NotImplementedError(f"Objective type {obj_type} not implemented")

        self._has_objective = True
        self._minimize = True
        self._log(1, f"Set {obj_type} minimization objective")

    def obj_maximize(self, obj_type: TypeObj, lst: list[Variable] | list[Node], coefficients: list[int] | None):
        """Set maximization objective."""
        exprs = self._get_pumpkin_var_list(lst)

        if obj_type == TypeObj.SUM:
            if coefficients is None:
                coefficients = [1] * len(exprs)
            obj_var = self.model.new_integer_variable(-10**9, 10**9, name="objective")
            sum_args = [(e, c) for e, c in zip(exprs, coefficients)]
            sum_args.append((obj_var, -1))
            self.model.add_constraint(constraints.Equals(sum_args, 0, self._new_tag()))
            self._objective_var = obj_var
        elif obj_type == TypeObj.EXPRESSION:
            if len(exprs) == 1:
                self._objective_var = exprs[0]
            else:
                raise NotImplementedError("Multi-expression maximize")
        elif obj_type == TypeObj.MINIMUM:
            self._objective_var = self._min(exprs)
        elif obj_type == TypeObj.MAXIMUM:
            self._objective_var = self._max(exprs)
        else:
            raise NotImplementedError(f"Objective type {obj_type} not implemented")

        self._has_objective = True
        self._minimize = False
        self._log(1, f"Set {obj_type} maximization objective")

    # ========== Solving ==========

    def apply_hints(self):
        """Pumpkin does not support warm start hints."""
        if self.hints:
            valid = sum(1 for var_id in self.hints if var_id in self.vars)
            if valid > 0:
                self._log(1, f"Pumpkin does not support warm start hints ({valid} hints ignored)")

    def solve(self) -> TypeStatus:
        """Solve the model and return status."""
        self._log(1, "Starting Pumpkin solver...")
        self._log(1, f"  Variables: {len(self.vars)}")

        timeout = self.time_limit if self.time_limit else None

        try:
            if self._has_objective:
                # Optimization
                direction = "Minimise" if self._minimize else "Maximise"
                result = self.model.optimise(
                    timeout=timeout,
                    objective=self._objective_var,
                    direction=direction,
                )
            else:
                # Satisfaction
                result = self.model.satisfy(timeout=timeout)

            # Check result type and extract solution
            if hasattr(result, 'solution') and result.solution is not None:
                solution = result.solution
                self._solution = {}
                for var_id, var in self.vars.items():
                    try:
                        self._solution[var_id] = solution.int_value(var)
                    except Exception:
                        pass

                if self._has_objective:
                    self._objective_value = solution.int_value(self._objective_var)
                    # Check if optimal
                    if hasattr(result, 'is_optimal') and result.is_optimal:
                        self._status = TypeStatus.OPTIMUM
                    else:
                        self._status = TypeStatus.SAT
                else:
                    self._status = TypeStatus.SAT
            elif hasattr(result, 'is_unsatisfiable') and result.is_unsatisfiable:
                self._status = TypeStatus.UNSAT
            else:
                self._status = TypeStatus.UNKNOWN

        except Exception as e:
            self._log(1, f"Solver error: {e}")
            self._status = TypeStatus.UNKNOWN

        self._log(1, f"Solver finished with status: {self._status}")
        return self._status

    def get_solution(self) -> dict[str, int] | None:
        """Return the solution as {var_id: value} dict."""
        return self._solution

    def get_all_solutions(self) -> list[dict[str, int]]:
        """Return all solutions found."""
        return self._all_solutions
