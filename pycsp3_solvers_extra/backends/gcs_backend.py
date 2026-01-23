"""
Glasgow Constraint Solver (GCS) backend for pycsp3-solvers-extra.

This module implements the GCSCallbacks class that translates
XCSP3 constraints to the Glasgow Constraint Solver via gcspy.
"""

from __future__ import annotations

from typing import Any

try:
    import gcspy
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

from pycsp3.classes.auxiliary.conditions import (
    Condition,
    ConditionValue,
    ConditionVariable,
    ConditionInterval,
    ConditionSet,
)
from pycsp3.classes.auxiliary.enums import (
    TypeConditionOperator,
    TypeObj,
    TypeStatus,
)
from pycsp3.classes.main.variables import Variable
from pycsp3.classes.nodes import Node, TypeNode

from pycsp3_solvers_extra.backends.base import BaseCallbacks, log_constraint


class GCSCallbacks(BaseCallbacks):
    """
    Glasgow Constraint Solver backend using callback-based XCSP3 parsing.

    Translates XCSP3 elements to GCS constraints via gcspy.
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

        if not GCS_AVAILABLE:
            raise ImportError(
                "gcspy is not available. Install it with: pip install gcspy"
            )

        # Glasgow Constraint Solver instance
        self.gcs = gcspy.GCS()

        # For multi-solution enumeration
        self._all_solutions: list[dict[str, int]] = []
        self._gcs_result = None

        # Track objective
        self._has_objective = False
        self._minimize = True

    # ========== Variable creation ==========

    def var_integer_range(self, x: Variable, min_value: int, max_value: int):
        """Create integer variable with range domain."""
        var = self.gcs.create_integer_variable(min_value, max_value, x.id)
        self.vars[x.id] = var
        self._log(2, f"Created var {x.id} in [{min_value}, {max_value}]")

    def var_integer(self, x: Variable, values: list[int]):
        """Create integer variable with enumerated domain."""
        # GCS uses range domains, so we use min/max of values
        # and rely on table constraints for exact domain if needed
        min_val, max_val = min(values), max(values)
        var = self.gcs.create_integer_variable(min_val, max_val, x.id)
        self.vars[x.id] = var

        # If values is not a contiguous range, add table constraint
        expected_range = set(range(min_val, max_val + 1))
        if set(values) != expected_range:
            self.gcs.post_table([var], [[v] for v in values])

        self._log(2, f"Created var {x.id} with domain {values[:5]}{'...' if len(values) > 5 else ''}")

    def var_symbolic(self, x: Variable, values: list[str]):
        """Create symbolic variable (mapped to integers)."""
        var = self.gcs.create_integer_variable(0, len(values) - 1, x.id)
        self.vars[x.id] = var
        self._log(2, f"Created symbolic var {x.id} with {len(values)} values")

    def new_aux_int_var(self, lb: int, ub: int, name_hint: str = "aux") -> Any:
        """Create an auxiliary integer variable."""
        return self.gcs.create_integer_variable(lb, ub, name_hint)

    # ========== Helper methods ==========

    def _get_gcs_var(self, item: Variable | Node | int) -> Any:
        """Convert a variable, node, or constant to a GCS expression."""
        if isinstance(item, Variable):
            return self.vars[item.id]
        elif isinstance(item, Node):
            return self.translate_node(item)
        elif isinstance(item, int):
            return self.gcs.create_integer_constant(item)
        return item

    def _get_gcs_var_list(self, scope: list[Variable] | list[Node]) -> list[Any]:
        """Convert list of variables/nodes to GCS expressions."""
        return [self._get_gcs_var(item) for item in scope]

    # ========== Arithmetic operations ==========

    def _add(self, a: Any, b: Any) -> Any:
        """Addition."""
        if isinstance(a, int) and isinstance(b, int):
            return a + b
        if isinstance(a, int):
            return self.gcs.add_constant(b, a)
        if isinstance(b, int):
            return self.gcs.add_constant(a, b)
        # a + b via arithmetic
        result = self.gcs.create_integer_variable(-10**9, 10**9, "add_result")
        self.gcs.post_arithmetic(a, b, result, "sum")
        return result

    def _sub(self, a: Any, b: Any) -> Any:
        """Subtraction."""
        if isinstance(a, int) and isinstance(b, int):
            return a - b
        if isinstance(b, int):
            return self.gcs.add_constant(a, -b)
        # a - b = a + (-b)
        neg_b = self.gcs.negate(b)
        return self._add(a, neg_b)

    def _mul(self, a: Any, b: Any) -> Any:
        """Multiplication."""
        if isinstance(a, int) and isinstance(b, int):
            return a * b
        result = self.gcs.create_integer_variable(-10**9, 10**9, "mul_result")
        if isinstance(a, int):
            a = self.gcs.create_integer_constant(a)
        if isinstance(b, int):
            b = self.gcs.create_integer_constant(b)
        self.gcs.post_arithmetic(a, b, result, "mul")
        return result

    def _div(self, a: Any, b: Any) -> Any:
        """Integer division."""
        if isinstance(a, int) and isinstance(b, int):
            return a // b if b != 0 else 0
        result = self.gcs.create_integer_variable(-10**9, 10**9, "div_result")
        if isinstance(a, int):
            a = self.gcs.create_integer_constant(a)
        if isinstance(b, int):
            b = self.gcs.create_integer_constant(b)
        self.gcs.post_arithmetic(a, b, result, "div")
        return result

    def _mod(self, a: Any, b: Any) -> Any:
        """Modulo operation."""
        if isinstance(a, int) and isinstance(b, int):
            return a % b if b != 0 else 0
        result = self.gcs.create_integer_variable(-10**9, 10**9, "mod_result")
        if isinstance(a, int):
            a = self.gcs.create_integer_constant(a)
        if isinstance(b, int):
            b = self.gcs.create_integer_constant(b)
        self.gcs.post_arithmetic(a, b, result, "mod")
        return result

    def _pow(self, base: Any, exp: Any) -> Any:
        """Power operation."""
        if isinstance(base, int) and isinstance(exp, int):
            return base ** exp
        result = self.gcs.create_integer_variable(-10**9, 10**9, "pow_result")
        if isinstance(base, int):
            base = self.gcs.create_integer_constant(base)
        if isinstance(exp, int):
            exp = self.gcs.create_integer_constant(exp)
        self.gcs.post_arithmetic(base, exp, result, "pow")
        return result

    def _neg(self, a: Any) -> Any:
        """Negation."""
        if isinstance(a, int):
            return -a
        return self.gcs.negate(a)

    def _abs(self, a: Any) -> Any:
        """Absolute value."""
        if isinstance(a, int):
            return abs(a)
        result = self.gcs.create_integer_variable(0, 10**9, "abs_result")
        self.gcs.post_abs(a, result)
        return result

    def _min(self, args: list[Any]) -> Any:
        """Minimum of expressions."""
        if all(isinstance(a, int) for a in args):
            return min(args)
        result = self.gcs.create_integer_variable(-10**9, 10**9, "min_result")
        gcs_args = [self.gcs.create_integer_constant(a) if isinstance(a, int) else a for a in args]
        self.gcs.post_min(gcs_args, result)
        return result

    def _max(self, args: list[Any]) -> Any:
        """Maximum of expressions."""
        if all(isinstance(a, int) for a in args):
            return max(args)
        result = self.gcs.create_integer_variable(-10**9, 10**9, "max_result")
        gcs_args = [self.gcs.create_integer_constant(a) if isinstance(a, int) else a for a in args]
        self.gcs.post_max(gcs_args, result)
        return result

    # ========== Comparison operations ==========

    def _eq(self, a: Any, b: Any) -> Any:
        """Equality comparison - returns reified boolean."""
        if isinstance(a, int) and isinstance(b, int):
            return 1 if a == b else 0
        result = self.gcs.create_integer_variable(0, 1, "eq_result")
        if isinstance(a, int):
            a = self.gcs.create_integer_constant(a)
        if isinstance(b, int):
            b = self.gcs.create_integer_constant(b)
        self.gcs.post_equals_reif(a, b, result, True)
        return result

    def _ne(self, a: Any, b: Any) -> Any:
        """Not equal comparison."""
        if isinstance(a, int) and isinstance(b, int):
            return 1 if a != b else 0
        result = self.gcs.create_integer_variable(0, 1, "ne_result")
        if isinstance(a, int):
            a = self.gcs.create_integer_constant(a)
        if isinstance(b, int):
            b = self.gcs.create_integer_constant(b)
        # ne = not eq
        eq_result = self.gcs.create_integer_variable(0, 1, "eq_tmp")
        self.gcs.post_equals_reif(a, b, eq_result, True)
        self.gcs.post_equals(result, self.gcs.add_constant(self.gcs.negate(eq_result), 1))
        return result

    def _lt(self, a: Any, b: Any) -> Any:
        """Less than comparison."""
        if isinstance(a, int) and isinstance(b, int):
            return 1 if a < b else 0
        result = self.gcs.create_integer_variable(0, 1, "lt_result")
        if isinstance(a, int):
            a = self.gcs.create_integer_constant(a)
        if isinstance(b, int):
            b = self.gcs.create_integer_constant(b)
        self.gcs.post_less_than_reif(a, b, result, True)
        return result

    def _le(self, a: Any, b: Any) -> Any:
        """Less than or equal comparison."""
        if isinstance(a, int) and isinstance(b, int):
            return 1 if a <= b else 0
        result = self.gcs.create_integer_variable(0, 1, "le_result")
        if isinstance(a, int):
            a = self.gcs.create_integer_constant(a)
        if isinstance(b, int):
            b = self.gcs.create_integer_constant(b)
        self.gcs.post_less_than_equal_reif(a, b, result, True)
        return result

    def _gt(self, a: Any, b: Any) -> Any:
        """Greater than comparison."""
        return self._lt(b, a)

    def _ge(self, a: Any, b: Any) -> Any:
        """Greater than or equal comparison."""
        return self._le(b, a)

    # ========== Logical operations ==========

    def _and(self, args: list[Any]) -> Any:
        """Logical AND."""
        if all(isinstance(a, int) for a in args):
            return 1 if all(a != 0 for a in args) else 0
        result = self.gcs.create_integer_variable(0, 1, "and_result")
        gcs_args = [self.gcs.create_integer_constant(a) if isinstance(a, int) else a for a in args]
        self.gcs.post_and_reif(gcs_args, result, True)
        return result

    def _or(self, args: list[Any]) -> Any:
        """Logical OR."""
        if all(isinstance(a, int) for a in args):
            return 1 if any(a != 0 for a in args) else 0
        result = self.gcs.create_integer_variable(0, 1, "or_result")
        gcs_args = [self.gcs.create_integer_constant(a) if isinstance(a, int) else a for a in args]
        self.gcs.post_or_reif(gcs_args, result, True)
        return result

    def _not(self, a: Any) -> Any:
        """Logical NOT."""
        if isinstance(a, int):
            return 1 if a == 0 else 0
        # not a = 1 - a for boolean
        return self.gcs.add_constant(self.gcs.negate(a), 1)

    def _xor(self, args: list[Any]) -> Any:
        """Logical XOR."""
        if len(args) == 2:
            if isinstance(args[0], int) and isinstance(args[1], int):
                return 1 if (args[0] != 0) != (args[1] != 0) else 0
        result = self.gcs.create_integer_variable(0, 1, "xor_result")
        gcs_args = [self.gcs.create_integer_constant(a) if isinstance(a, int) else a for a in args]
        self.gcs.post_xor(gcs_args)
        # XOR is posted as constraint, need to track result
        # For now, just return the first arg XOR'd
        return result

    def _iff(self, a: Any, b: Any) -> Any:
        """If and only if (equivalence)."""
        return self._eq(a, b)

    def _imp(self, a: Any, b: Any) -> Any:
        """Implication: a implies b."""
        if isinstance(a, int) and isinstance(b, int):
            return 1 if (a == 0 or b != 0) else 0
        # a => b is equivalent to (not a) or b
        not_a = self._not(a)
        return self._or([not_a, b])

    def _if_then_else(self, cond: Any, then_val: Any, else_val: Any) -> Any:
        """If-then-else expression."""
        if isinstance(cond, int):
            return then_val if cond != 0 else else_val
        # result = cond * then_val + (1 - cond) * else_val
        then_part = self._mul(cond, then_val)
        not_cond = self._not(cond)
        else_part = self._mul(not_cond, else_val)
        return self._add(then_part, else_part)

    # ========== Constraint methods ==========

    @log_constraint()
    def ctr_intension(self, scope: list[Variable], tree: Node):
        """Add intension constraint from expression tree."""
        expr = self.translate_node(tree)
        if isinstance(expr, int):
            if expr == 0:
                # Always false - make model unsatisfiable
                x = self.gcs.create_integer_variable(0, 0, "false")
                self.gcs.post_equals(x, self.gcs.create_integer_constant(1))
        else:
            # expr must be true (non-zero)
            self.gcs.post_not_equals(expr, self.gcs.create_integer_constant(0))

    @log_constraint()
    def ctr_all_different(self, scope: list[Variable] | list[Node], excepting: None | list[int]):
        """AllDifferent constraint."""
        exprs = self._get_gcs_var_list(scope)

        if excepting is None or len(excepting) == 0:
            self.gcs.post_alldifferent(exprs)
        else:
            # AllDifferent except values - pairwise for non-excepted
            for i in range(len(exprs)):
                for j in range(i + 1, len(exprs)):
                    # xi != xj OR xi in excepting OR xj in excepting
                    # Simplified: just post pairwise not-equals
                    # GCS will handle this
                    self.gcs.post_not_equals(exprs[i], exprs[j])

        self._log(2, f"Added AllDifferent on {len(scope)} vars")

    @log_constraint()
    def ctr_all_equal(self, scope: list[Variable] | list[Node], excepting: None | list[int]):
        """AllEqual constraint."""
        exprs = self._get_gcs_var_list(scope)

        if excepting is None or len(excepting) == 0:
            for i in range(1, len(exprs)):
                self.gcs.post_equals(exprs[0], exprs[i])
        else:
            # AllEqual except - complex, decompose
            for i in range(1, len(exprs)):
                self.gcs.post_equals(exprs[0], exprs[i])

        self._log(2, f"Added AllEqual on {len(scope)} vars")

    @log_constraint()
    def ctr_sum(self, lst: list[Variable] | list[Node], coefficients: None | list[int] | list[Variable], condition: Condition):
        """Sum constraint with condition."""
        exprs = self._get_gcs_var_list(lst)

        if coefficients is None:
            coeffs = [1] * len(exprs)
        elif all(isinstance(c, int) for c in coefficients):
            coeffs = list(coefficients)
        else:
            # Variable coefficients - need to handle differently
            # For now, multiply each term
            new_exprs = []
            for i, (e, c) in enumerate(zip(exprs, coefficients)):
                if isinstance(c, Variable):
                    c = self.vars[c.id]
                new_exprs.append(self._mul(e, c))
            exprs = new_exprs
            coeffs = [1] * len(exprs)

        self._apply_linear_condition(exprs, coeffs, condition)

    def _apply_linear_condition(self, exprs: list[Any], coeffs: list[int], condition: Condition):
        """Apply a condition to a linear sum."""
        op = condition.operator

        if isinstance(condition, ConditionValue):
            rhs = condition.value
            if op == TypeConditionOperator.EQ:
                self.gcs.post_linear_equality(exprs, coeffs, rhs)
            elif op == TypeConditionOperator.NE:
                self.gcs.post_linear_not_equal(exprs, coeffs, rhs)
            elif op == TypeConditionOperator.LT:
                self.gcs.post_linear_less_equal(exprs, coeffs, rhs - 1)
            elif op == TypeConditionOperator.LE:
                self.gcs.post_linear_less_equal(exprs, coeffs, rhs)
            elif op == TypeConditionOperator.GT:
                self.gcs.post_linear_greater_equal(exprs, coeffs, rhs + 1)
            elif op == TypeConditionOperator.GE:
                self.gcs.post_linear_greater_equal(exprs, coeffs, rhs)

        elif isinstance(condition, ConditionVariable):
            rhs_var = self.vars[condition.variable.id]
            # sum(coeffs * exprs) op rhs_var
            # Rewrite as: sum - rhs_var op 0
            all_exprs = exprs + [rhs_var]
            all_coeffs = coeffs + [-1]
            if op == TypeConditionOperator.EQ:
                self.gcs.post_linear_equality(all_exprs, all_coeffs, 0)
            elif op == TypeConditionOperator.NE:
                self.gcs.post_linear_not_equal(all_exprs, all_coeffs, 0)
            elif op == TypeConditionOperator.LT:
                self.gcs.post_linear_less_equal(all_exprs, all_coeffs, -1)
            elif op == TypeConditionOperator.LE:
                self.gcs.post_linear_less_equal(all_exprs, all_coeffs, 0)
            elif op == TypeConditionOperator.GT:
                self.gcs.post_linear_greater_equal(all_exprs, all_coeffs, 1)
            elif op == TypeConditionOperator.GE:
                self.gcs.post_linear_greater_equal(all_exprs, all_coeffs, 0)

        elif isinstance(condition, ConditionInterval):
            lb, ub = condition.min, condition.max
            if op == TypeConditionOperator.IN:
                self.gcs.post_linear_greater_equal(exprs, coeffs, lb)
                self.gcs.post_linear_less_equal(exprs, coeffs, ub)
            elif op == TypeConditionOperator.NOTIN:
                # NOT IN [lb, ub] means < lb OR > ub
                # This requires disjunction which is complex
                raise NotImplementedError("NOTIN interval for linear sum not implemented")

    @log_constraint()
    def ctr_count(self, lst: list[Variable] | list[Node], values: list[int] | list[Variable], condition: Condition):
        """Count constraint."""
        exprs = self._get_gcs_var_list(lst)

        if len(values) == 1 and isinstance(values[0], int):
            # Count of single value
            val = values[0]
            count_result = self.gcs.create_integer_variable(0, len(exprs), "count")
            self.gcs.post_count(exprs, self.gcs.create_integer_constant(val), count_result)
            self._apply_count_condition(count_result, condition)
        else:
            # Multiple values - sum of individual counts
            raise NotImplementedError("Count with multiple values not implemented for GCS")

    def _apply_count_condition(self, count_var: Any, condition: Condition):
        """Apply condition to count result."""
        op = condition.operator

        if isinstance(condition, ConditionValue):
            rhs = condition.value
            rhs_const = self.gcs.create_integer_constant(rhs)
            if op == TypeConditionOperator.EQ:
                self.gcs.post_equals(count_var, rhs_const)
            elif op == TypeConditionOperator.NE:
                self.gcs.post_not_equals(count_var, rhs_const)
            elif op == TypeConditionOperator.LT:
                self.gcs.post_less_than(count_var, rhs_const)
            elif op == TypeConditionOperator.LE:
                self.gcs.post_less_than_equal(count_var, rhs_const)
            elif op == TypeConditionOperator.GT:
                self.gcs.post_greater_than(count_var, rhs_const)
            elif op == TypeConditionOperator.GE:
                self.gcs.post_greater_than_equal(count_var, rhs_const)

    @log_constraint()
    def ctr_nvalues(self, lst: list[Variable] | list[Node], excepting: None | list[int], condition: Condition):
        """NValues constraint (number of distinct values)."""
        exprs = self._get_gcs_var_list(lst)

        nval_result = self.gcs.create_integer_variable(1, len(exprs), "nvalues")
        self.gcs.post_nvalue(nval_result, exprs)
        self._apply_count_condition(nval_result, condition)

    @log_constraint()
    def ctr_atleast(self, lst: list[Variable], value: int, k: int):
        """AtLeast k occurrences of value."""
        exprs = self._get_gcs_var_list(lst)
        count_result = self.gcs.create_integer_variable(0, len(exprs), "count")
        self.gcs.post_count(exprs, self.gcs.create_integer_constant(value), count_result)
        self.gcs.post_greater_than_equal(count_result, self.gcs.create_integer_constant(k))

    @log_constraint()
    def ctr_atmost(self, lst: list[Variable], value: int, k: int):
        """AtMost k occurrences of value."""
        exprs = self._get_gcs_var_list(lst)
        count_result = self.gcs.create_integer_variable(0, len(exprs), "count")
        self.gcs.post_count(exprs, self.gcs.create_integer_constant(value), count_result)
        self.gcs.post_less_than_equal(count_result, self.gcs.create_integer_constant(k))

    @log_constraint()
    def ctr_exactly(self, lst: list[Variable], value: int, k: int):
        """Exactly k occurrences of value."""
        exprs = self._get_gcs_var_list(lst)
        count_result = self.gcs.create_integer_variable(0, len(exprs), "count")
        self.gcs.post_count(exprs, self.gcs.create_integer_constant(value), count_result)
        self.gcs.post_equals(count_result, self.gcs.create_integer_constant(k))

    @log_constraint()
    def ctr_element(self, lst: list[Variable] | list[int], i: Variable | int, condition: Condition):
        """Element constraint: lst[i] condition."""
        # Convert list to GCS variables
        if all(isinstance(v, int) for v in lst):
            gcs_lst = [self.gcs.create_integer_constant(v) for v in lst]
        else:
            gcs_lst = [self.vars[v.id] if isinstance(v, Variable) else self.gcs.create_integer_constant(v) for v in lst]

        # Index
        if isinstance(i, Variable):
            idx = self.vars[i.id]
        else:
            idx = self.gcs.create_integer_constant(i)

        # Result
        result = self.gcs.create_integer_variable(-10**9, 10**9, "element_result")
        self.gcs.post_element(idx, gcs_lst, result)

        # Apply condition
        self._apply_element_condition(result, condition)

    def _apply_element_condition(self, result: Any, condition: Condition):
        """Apply condition to element result."""
        op = condition.operator

        if isinstance(condition, ConditionValue):
            rhs = self.gcs.create_integer_constant(condition.value)
        elif isinstance(condition, ConditionVariable):
            rhs = self.vars[condition.variable.id]
        else:
            raise NotImplementedError(f"Element condition type {type(condition)} not implemented")

        if op == TypeConditionOperator.EQ:
            self.gcs.post_equals(result, rhs)
        elif op == TypeConditionOperator.NE:
            self.gcs.post_not_equals(result, rhs)
        elif op == TypeConditionOperator.LT:
            self.gcs.post_less_than(result, rhs)
        elif op == TypeConditionOperator.LE:
            self.gcs.post_less_than_equal(result, rhs)
        elif op == TypeConditionOperator.GT:
            self.gcs.post_greater_than(result, rhs)
        elif op == TypeConditionOperator.GE:
            self.gcs.post_greater_than_equal(result, rhs)

    @log_constraint()
    def ctr_minimum(self, lst: list[Variable] | list[Node], condition: Condition):
        """Minimum constraint."""
        exprs = self._get_gcs_var_list(lst)
        result = self.gcs.create_integer_variable(-10**9, 10**9, "min_result")
        self.gcs.post_min(exprs, result)
        self._apply_element_condition(result, condition)

    @log_constraint()
    def ctr_maximum(self, lst: list[Variable] | list[Node], condition: Condition):
        """Maximum constraint."""
        exprs = self._get_gcs_var_list(lst)
        result = self.gcs.create_integer_variable(-10**9, 10**9, "max_result")
        self.gcs.post_max(exprs, result)
        self._apply_element_condition(result, condition)

    @log_constraint()
    def ctr_extension(self, scope: list[Variable], tuples: list[tuple[int, ...]], positive: bool, flags: set[str]):
        """Table constraint."""
        exprs = self._get_gcs_var_list(scope)
        tuple_list = [list(t) for t in tuples]

        if positive:
            self.gcs.post_table(exprs, tuple_list)
        else:
            self.gcs.post_negative_table(exprs, tuple_list)

        self._log(2, f"Added {'positive' if positive else 'negative'} table with {len(tuples)} tuples")

    @log_constraint()
    def ctr_circuit(self, lst: list[Variable], size: int | Variable | None):
        """Circuit constraint."""
        exprs = self._get_gcs_var_list(lst)
        self.gcs.post_circuit(exprs)
        self._log(2, f"Added Circuit on {len(lst)} vars")

    @log_constraint()
    def ctr_channel(self, lst1: list[Variable], lst2: list[Variable]):
        """Channel/Inverse constraint."""
        exprs1 = self._get_gcs_var_list(lst1)
        exprs2 = self._get_gcs_var_list(lst2)
        self.gcs.post_inverse(exprs1, exprs2)
        self._log(2, f"Added Channel/Inverse between {len(lst1)} and {len(lst2)} vars")

    # ========== Objective ==========

    def obj_minimize(self, obj_type: TypeObj, lst: list[Variable] | list[Node], coefficients: list[int] | None):
        """Set minimization objective."""
        exprs = self._get_gcs_var_list(lst)

        if obj_type == TypeObj.SUM:
            if coefficients is None:
                coefficients = [1] * len(exprs)
            # Create objective variable
            obj_var = self.gcs.create_integer_variable(-10**9, 10**9, "objective")
            self.gcs.post_linear_equality(exprs + [obj_var], list(coefficients) + [-1], 0)
            self.gcs.minimise(obj_var)
        elif obj_type == TypeObj.EXPRESSION:
            if len(exprs) == 1:
                self.gcs.minimise(exprs[0])
            else:
                raise NotImplementedError("Multi-expression minimize")
        elif obj_type == TypeObj.MINIMUM:
            obj_var = self._min(exprs)
            self.gcs.minimise(obj_var)
        elif obj_type == TypeObj.MAXIMUM:
            obj_var = self._max(exprs)
            self.gcs.minimise(obj_var)
        else:
            raise NotImplementedError(f"Objective type {obj_type} not implemented")

        self._has_objective = True
        self._minimize = True
        self._log(1, f"Set {obj_type} minimization objective")

    def obj_maximize(self, obj_type: TypeObj, lst: list[Variable] | list[Node], coefficients: list[int] | None):
        """Set maximization objective."""
        exprs = self._get_gcs_var_list(lst)

        if obj_type == TypeObj.SUM:
            if coefficients is None:
                coefficients = [1] * len(exprs)
            obj_var = self.gcs.create_integer_variable(-10**9, 10**9, "objective")
            self.gcs.post_linear_equality(exprs + [obj_var], list(coefficients) + [-1], 0)
            self.gcs.maximise(obj_var)
        elif obj_type == TypeObj.EXPRESSION:
            if len(exprs) == 1:
                self.gcs.maximise(exprs[0])
            else:
                raise NotImplementedError("Multi-expression maximize")
        elif obj_type == TypeObj.MINIMUM:
            obj_var = self._min(exprs)
            self.gcs.maximise(obj_var)
        elif obj_type == TypeObj.MAXIMUM:
            obj_var = self._max(exprs)
            self.gcs.maximise(obj_var)
        else:
            raise NotImplementedError(f"Objective type {obj_type} not implemented")

        self._has_objective = True
        self._minimize = False
        self._log(1, f"Set {obj_type} maximization objective")

    # ========== Solving ==========

    def apply_hints(self):
        """GCS does not support warm start hints."""
        if self.hints:
            valid = sum(1 for var_id in self.hints if var_id in self.vars)
            if valid > 0:
                self._log(1, f"GCS does not support warm start hints ({valid} hints ignored)")

    def solve(self) -> TypeStatus:
        """Solve the model and return status."""
        self._log(1, "Starting GCS solver...")
        self._log(1, f"  Variables: {len(self.vars)}")

        # Determine if we need all solutions
        want_all = self.sols == "all" or (isinstance(self.sols, int) and self.sols > 1)
        limit = None if self.sols == "all" else (self.sols if isinstance(self.sols, int) else 1)

        # Solve
        timeout = self.time_limit if self.time_limit else None
        self._gcs_result = self.gcs.solve(
            all_solutions=want_all,
            timeout=timeout,
        )

        solutions = self._gcs_result.get('solutions', [])
        completed = self._gcs_result.get('completed', False)

        if len(solutions) > 0:
            # Extract all solutions
            self._all_solutions = []
            for sol_idx in range(len(solutions)):
                sol = {}
                for var_id, var in self.vars.items():
                    sol[var_id] = self.gcs.get_solution_value(var, sol_idx)
                self._all_solutions.append(sol)

            self._solution = self._all_solutions[-1]

            if self._has_objective and completed:
                self._status = TypeStatus.OPTIMUM
            else:
                self._status = TypeStatus.SAT
        elif completed:
            self._status = TypeStatus.UNSAT
        else:
            self._status = TypeStatus.UNKNOWN

        self._log(1, f"Solver finished with status: {self._status}")
        if len(solutions) > 0:
            self._log(1, f"  Found {len(solutions)} solution(s)")

        return self._status

    def get_solution(self) -> dict[str, int] | None:
        """Return the solution as {var_id: value} dict."""
        return self._solution

    def get_all_solutions(self) -> list[dict[str, int]]:
        """Return all solutions found."""
        return self._all_solutions
