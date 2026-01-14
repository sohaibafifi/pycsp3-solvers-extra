"""
Z3 SMT solver backend for pycsp3-solvers-extra.

This module implements the Z3Callbacks class that translates
XCSP3 constraints to Z3 SMT solver constraints.
"""

from __future__ import annotations

from functools import reduce
from typing import Any

from z3 import (
    Int,
    Bool,
    And,
    Or,
    Not,
    Xor,
    If,
    Implies,
    Distinct,
    Sum,
    Solver,
    Optimize,
    sat,
    unsat,
    IntVal,
    is_int,
    is_bool,
    ArithRef,
    BoolRef,
)

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
from pycsp3.classes.main.variables import Variable
from pycsp3.classes.nodes import Node, TypeNode

from pycsp3_solvers_extra.backends.base import BaseCallbacks


class Z3Callbacks(BaseCallbacks):
    """
    Z3 SMT solver backend using callback-based XCSP3 parsing.

    Translates XCSP3 elements to Z3 SMT constraints.
    """

    def __init__(
        self,
        time_limit: float | None = None,
        sols: int | str | None = None,
        verbose: int = 0,
        options: str = "",
    ):
        super().__init__(time_limit, sols, verbose, options)

        # Constraint storage (added to solver in solve())
        self._constraints: list[BoolRef] = []
        self._domain_constraints: list[BoolRef] = []

        # Objective tracking
        self._has_objective = False
        self._objective = None
        self._minimize = True

        # Auxiliary variable counter
        self._aux_counter = 0

        # For multi-solution enumeration
        self._all_solutions: list[dict[str, int]] = []

    # ========== Variable creation ==========

    def var_integer_range(self, x: Variable, min_value: int, max_value: int):
        """Create integer variable with range domain."""
        var = Int(x.id)
        self.vars[x.id] = var
        # Add domain constraint
        self._domain_constraints.append(And(var >= min_value, var <= max_value))
        self._log(2, f"Created var {x.id} in [{min_value}, {max_value}]")

    def new_aux_int_var(self, lb: int, ub: int, name_hint: str = "aux") -> Any:
        """Create an auxiliary integer variable."""
        self._aux_counter += 1
        var = Int(f"{name_hint}_{self._aux_counter}")
        self._domain_constraints.append(And(var >= lb, var <= ub))
        return var

    def var_integer(self, x: Variable, values: list[int]):
        """Create integer variable with enumerated domain."""
        var = Int(x.id)
        self.vars[x.id] = var
        # Add domain constraint as disjunction
        if values:
            self._domain_constraints.append(Or([var == v for v in values]))
        self._log(
            2,
            f"Created var {x.id} with domain {values[:5]}{'...' if len(values) > 5 else ''}",
        )

    def var_symbolic(self, x: Variable, values: list[str]):
        """Create symbolic variable (mapped to integers)."""
        # Map symbolic values to integers 0, 1, 2, ...
        var = Int(x.id)
        self.vars[x.id] = var
        self._domain_constraints.append(And(var >= 0, var <= len(values) - 1))
        self._log(2, f"Created symbolic var {x.id} with {len(values)} values")

    # ========== Arithmetic operations ==========

    def _add(self, a: Any, b: Any) -> Any:
        return a + b

    def _sub(self, a: Any, b: Any) -> Any:
        return a - b

    def _mul(self, a: Any, b: Any) -> Any:
        return a * b

    def _div(self, a: Any, b: Any) -> Any:
        # Z3 integer division
        return a / b

    def _mod(self, a: Any, b: Any) -> Any:
        return a % b

    def _pow(self, base: Any, exp: Any) -> Any:
        # Z3 doesn't have native power, but can handle constant exponents
        if isinstance(exp, int):
            if exp == 0:
                return IntVal(1)
            elif exp == 1:
                return base
            elif exp == 2:
                return base * base
            else:
                result = base
                for _ in range(exp - 1):
                    result = result * base
                return result
        raise NotImplementedError("Z3 backend only supports constant integer exponents")

    def _neg(self, a: Any) -> Any:
        return -a

    def _abs(self, a: Any) -> Any:
        return If(a >= 0, a, -a)

    def _min(self, args: list[Any]) -> Any:
        if len(args) == 1:
            return args[0]
        return reduce(lambda a, b: If(a <= b, a, b), args)

    def _max(self, args: list[Any]) -> Any:
        if len(args) == 1:
            return args[0]
        return reduce(lambda a, b: If(a >= b, a, b), args)

    # ========== Comparison operations ==========

    def _eq(self, a: Any, b: Any) -> Any:
        return a == b

    def _ne(self, a: Any, b: Any) -> Any:
        return a != b

    def _lt(self, a: Any, b: Any) -> Any:
        return a < b

    def _le(self, a: Any, b: Any) -> Any:
        return a <= b

    def _gt(self, a: Any, b: Any) -> Any:
        return a > b

    def _ge(self, a: Any, b: Any) -> Any:
        return a >= b

    # ========== Logical operations ==========

    def _and(self, args: list[Any]) -> Any:
        if len(args) == 0:
            return True
        if len(args) == 1:
            return self._as_bool_expr(args[0])
        return And(*[self._as_bool_expr(a) for a in args])

    def _or(self, args: list[Any]) -> Any:
        if len(args) == 0:
            return False
        if len(args) == 1:
            return self._as_bool_expr(args[0])
        return Or(*[self._as_bool_expr(a) for a in args])

    def _not(self, a: Any) -> Any:
        return Not(self._as_bool_expr(a))

    def _xor(self, args: list[Any]) -> Any:
        if len(args) == 0:
            return False
        if len(args) == 1:
            return self._as_bool_expr(args[0])
        args = [self._as_bool_expr(a) for a in args]
        if len(args) == 2:
            return Xor(args[0], args[1])
        # Chain XOR for multiple args
        return reduce(lambda a, b: Xor(a, b), args)

    def _iff(self, a: Any, b: Any) -> Any:
        return self._as_bool_expr(a) == self._as_bool_expr(b)

    def _imp(self, a: Any, b: Any) -> Any:
        return Implies(self._as_bool_expr(a), self._as_bool_expr(b))

    def _if_then_else(self, cond: Any, then_val: Any, else_val: Any) -> Any:
        return If(self._as_bool_expr(cond), then_val, else_val)

    def _in_set(self, val: Any, set_vals: list[Any]) -> Any:
        """Set membership with optimizations.

        Optimizations applied (based on benchmarks):
        1. Empty set: return False
        2. Consecutive integer range (>=3 values): use range constraint 
        """
        if not set_vals:
            return False

        # Check if all values are constant integers
        const_vals = [v for v in set_vals if isinstance(v, int)]
        if len(const_vals) == len(set_vals) and len(const_vals) >= 3:
            sorted_vals = sorted(const_vals)
            min_val, max_val = sorted_vals[0], sorted_vals[-1]

            # Check if values form a consecutive range
            # Benchmark: 3.49x speedup for 100 values, 1.59x for 20 values
            if max_val - min_val + 1 == len(sorted_vals):
                return And(val >= min_val, val <= max_val)

        # General case
        return Or([val == v for v in set_vals])

    def _not_in_set(self, val: Any, set_vals: list[Any]) -> Any:
        """Set non-membership with optimizations.

        Optimizations applied (based on benchmarks):
        1. Empty set: return True
        2. Consecutive integer range (>=3 values): use Or(val < min, val > max)
        """
        if not set_vals:
            return True
        
        # Check if all values are constant integers
        const_vals = [v for v in set_vals if isinstance(v, int)]
        if len(const_vals) == len(set_vals) and len(const_vals) >= 3:
            sorted_vals = sorted(const_vals)
            min_val, max_val = sorted_vals[0], sorted_vals[-1]

            # Check if values form a consecutive range
            if max_val - min_val + 1 == len(sorted_vals):
                return Or(val < min_val, val > max_val)

        # General case
        return And([val != v for v in set_vals])

    # ========== Helper methods ==========

    def _as_bool_expr(self, expr: Any) -> Any:
        """Coerce arithmetic expressions to boolean expressions."""
        if isinstance(expr, bool) or is_bool(expr):
            return expr
        if is_int(expr) or isinstance(expr, (ArithRef, int)):
            return expr != 0
        return expr

    def _add_constraint(self, constraint: BoolRef) -> None:
        """Add a constraint to the model."""
        self._constraints.append(constraint)

    def _to_z3_value(self, val: Any) -> Any:
        """Convert a value to Z3-compatible format."""
        if isinstance(val, Variable):
            return self.vars[val.id]
        # Convert numpy integers or other numeric types to Python int
        if hasattr(val, 'item'):
            return val.item()
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return int(val)
        return val

    def _apply_condition_to_model(self, expr: Any, condition: Condition) -> None:
        """Apply a condition to an expression and add to model."""
        from pycsp3.classes.auxiliary.conditions import (
            ConditionValue, ConditionVariable, ConditionInterval, ConditionSet, ConditionNode
        )

        op = condition.operator

        # Extract right operand based on condition type
        if isinstance(condition, ConditionValue):
            right = self._to_z3_value(condition.value)
        elif isinstance(condition, ConditionVariable):
            right = self.vars[condition.variable.id]
        elif isinstance(condition, ConditionInterval):
            # Range condition: expr in [min, max]
            min_val = self._to_z3_value(condition.min)
            max_val = self._to_z3_value(condition.max)
            if op == TypeConditionOperator.IN:
                self._add_constraint(And(expr >= min_val, expr <= max_val))
            else:  # NOTIN
                self._add_constraint(Or(expr < min_val, expr > max_val))
            return
        elif isinstance(condition, ConditionSet):
            # Set condition
            values = [self._to_z3_value(v) for v in condition.t]
            if op == TypeConditionOperator.IN:
                self._add_constraint(Or([expr == v for v in values]))
            else:  # NOTIN
                self._add_constraint(And([expr != v for v in values]))
            return
        elif isinstance(condition, ConditionNode):
            right = self.translate_node(condition.node)
        else:
            raise NotImplementedError(f"Condition type {type(condition)} not implemented")

        self._add_simple_comparison(expr, op, right)

    def _add_simple_comparison(self, expr: Any, op: TypeConditionOperator, k: Any) -> None:
        """Add a simple comparison constraint."""
        k = self._to_z3_value(k)
        if op == TypeConditionOperator.EQ:
            self._add_constraint(expr == k)
        elif op == TypeConditionOperator.NE:
            self._add_constraint(expr != k)
        elif op == TypeConditionOperator.LT:
            self._add_constraint(expr < k)
        elif op == TypeConditionOperator.LE:
            self._add_constraint(expr <= k)
        elif op == TypeConditionOperator.GT:
            self._add_constraint(expr > k)
        elif op == TypeConditionOperator.GE:
            self._add_constraint(expr >= k)
        else:
            raise NotImplementedError(f"Comparison operator {op} not implemented")

    # ========== Constraint callbacks ==========

    def ctr_intension(self, scope: list[Variable], tree: Node):
        """Add intension constraint (boolean expression tree)."""
        expr = self.translate_node(tree)
        self._add_constraint(expr)
        self._log(2, f"Added intension constraint on {len(scope)} vars")

    def ctr_extension_unary(
        self, x: Variable, values: list[int], positive: bool, flags: set[str]
    ):
        """Unary table constraint."""
        var = self.vars[x.id]
        expanded = []
        for v in values:
            if isinstance(v, range):
                expanded.extend(v)
            else:
                expanded.append(v)

        if positive:
            self._add_constraint(Or([var == v for v in expanded]))
        else:
            self._add_constraint(And([var != v for v in expanded]))
        self._log(
            2, f"Added {'positive' if positive else 'negative'} unary extension on {x.id}"
        )

    def ctr_extension(
        self, scope: list[Variable], tuples: list, positive: bool, flags: set[str]
    ):
        """Table constraint (allowed/forbidden assignments)."""
        vars_list = self._get_var_list(scope)

        if positive:
            # Allowed tuples: OR of (AND of equalities for each tuple)
            tuple_constraints = []
            for t in tuples:
                tuple_constraint = And([v == val for v, val in zip(vars_list, t)])
                tuple_constraints.append(tuple_constraint)
            if tuple_constraints:
                self._add_constraint(Or(tuple_constraints))
        else:
            # Forbidden tuples: AND of (NOT (AND of equalities for each tuple))
            for t in tuples:
                tuple_constraint = And([v == val for v, val in zip(vars_list, t)])
                self._add_constraint(Not(tuple_constraint))

        self._log(
            2,
            f"Added {'positive' if positive else 'negative'} extension on {len(scope)} vars with {len(tuples)} tuples",
        )

    def ctr_all_different(
        self, scope: list[Variable] | list[Node], excepting: None | list[int]
    ):
        """AllDifferent constraint."""
        exprs = self._get_var_or_node_list(scope)
        n = len(exprs)

        if excepting is None or len(excepting) == 0:
            # Z3's native Distinct is very efficient
            self._add_constraint(Distinct(*exprs))
            self._log(2, f"Added AllDifferent on {n} vars")
            return

        excepting_set = set(excepting)

        # Partition: find which expressions can take excepted values
        can_be_excepted: list[int] = []
        cannot_be_excepted: list[int] = []

        for i, item in enumerate(scope):
            if isinstance(item, Variable):
                dom_vals = item.dom.all_values()
                if isinstance(dom_vals, range):
                    has_excepted = any(v in dom_vals for v in excepting_set)
                else:
                    has_excepted = bool(set(dom_vals) & excepting_set)
            elif isinstance(item, Node):
                poss = item.possible_values()
                if isinstance(poss, range):
                    has_excepted = any(v in poss for v in excepting_set)
                else:
                    has_excepted = bool(set(poss) & excepting_set) if poss else False
            else:
                has_excepted = True

            if has_excepted:
                can_be_excepted.append(i)
            else:
                cannot_be_excepted.append(i)

        # Variables that cannot take excepted values - use native Distinct
        if len(cannot_be_excepted) >= 2:
            self._add_constraint(Distinct(*[exprs[i] for i in cannot_be_excepted]))

        # Variables that cannot be excepted must differ from those that can
        for i in cannot_be_excepted:
            for j in can_be_excepted:
                self._add_constraint(exprs[i] != exprs[j])

        # Among variables that can be excepted, use pairwise conditional constraints
        for idx_a in range(len(can_be_excepted)):
            for idx_b in range(idx_a + 1, len(can_be_excepted)):
                i, j = can_be_excepted[idx_a], can_be_excepted[idx_b]
                xi_excepted = Or([exprs[i] == v for v in excepting])
                xj_excepted = Or([exprs[j] == v for v in excepting])
                self._add_constraint(
                    Or(exprs[i] != exprs[j], xi_excepted, xj_excepted)
                )

        self._log(
            2,
            f"Added AllDifferent on {n} vars except {excepting} "
            f"({len(cannot_be_excepted)} global, {len(can_be_excepted)} conditional)",
        )

    def ctr_all_different_lists(
        self, lists: list[list[Variable]], excepting: None | list[list[int]]
    ):
        """AllDifferent on lists (each list differs from others)."""
        # Each pair of lists must differ in at least one position
        for i in range(len(lists)):
            for j in range(i + 1, len(lists)):
                diff_vars = []
                for k in range(len(lists[i])):
                    diff_vars.append(
                        self.vars[lists[i][k].id] != self.vars[lists[j][k].id]
                    )
                self._add_constraint(Or(diff_vars))
        self._log(2, f"Added AllDifferent on {len(lists)} lists")

    def ctr_all_equal(
        self, scope: list[Variable] | list[Node], excepting: None | list[int]
    ):
        """AllEqual constraint."""
        exprs = self._get_var_or_node_list(scope)

        if excepting is None or len(excepting) == 0:
            # All equal: x0 == x1 == x2 == ...
            for i in range(1, len(exprs)):
                self._add_constraint(exprs[0] == exprs[i])
        else:
            # All equal except values
            for i in range(len(exprs)):
                for j in range(i + 1, len(exprs)):
                    xi_excepted = Or([exprs[i] == v for v in excepting])
                    xj_excepted = Or([exprs[j] == v for v in excepting])
                    # If neither is excepted, must be equal
                    self._add_constraint(
                        Or(xi_excepted, xj_excepted, exprs[i] == exprs[j])
                    )

        self._log(2, f"Added AllEqual on {len(scope)} vars")

    def ctr_sum(
        self,
        lst: list[Variable] | list[Node],
        coefficients: None | list[int] | list[Variable],
        condition: Condition,
    ):
        """Sum constraint with condition."""
        exprs = self._get_var_or_node_list(lst)

        # Build sum expression
        if coefficients is None:
            sum_expr = Sum(exprs)
        else:
            if all(isinstance(c, int) for c in coefficients):
                sum_expr = Sum([c * e for c, e in zip(coefficients, exprs)])
            else:
                # Variable coefficients
                terms = []
                for c, e in zip(coefficients, exprs):
                    if isinstance(c, Variable):
                        c = self.vars[c.id]
                    terms.append(c * e)
                sum_expr = Sum(terms)

        # Apply condition
        self._apply_condition_to_model(sum_expr, condition)
        self._log(
            2, f"Added Sum constraint on {len(lst)} terms with condition {condition.operator}"
        )

    def ctr_count(
        self,
        lst: list[Variable] | list[Node],
        values: list[int] | list[Variable],
        condition: Condition,
    ):
        """Count constraint: count occurrences of values in lst."""
        exprs = self._get_var_or_node_list(lst)

        # Convert values to Z3 expressions if needed
        z3_values = []
        for v in values:
            if isinstance(v, Variable):
                z3_values.append(self.vars[v.id])
            else:
                z3_values.append(v)

        # Create count expression using If
        count_terms = []
        for e in exprs:
            if len(z3_values) == 1:
                count_terms.append(If(e == z3_values[0], 1, 0))
            else:
                in_values = Or([e == v for v in z3_values])
                count_terms.append(If(in_values, 1, 0))

        count_sum = Sum(count_terms)
        self._apply_condition_to_model(count_sum, condition)
        self._log(2, f"Added Count constraint")

    def ctr_atleast(self, lst: list[Variable], value: int, k: int):
        """AtLeast constraint: at least k occurrences of value in lst."""
        exprs = self._get_var_list(lst)
        count_terms = [If(e == value, 1, 0) for e in exprs]
        self._add_constraint(Sum(count_terms) >= k)
        self._log(2, f"Added AtLeast constraint: at least {k} of value {value}")

    def ctr_atmost(self, lst: list[Variable], value: int, k: int):
        """AtMost constraint: at most k occurrences of value in lst."""
        exprs = self._get_var_list(lst)
        count_terms = [If(e == value, 1, 0) for e in exprs]
        self._add_constraint(Sum(count_terms) <= k)
        self._log(2, f"Added AtMost constraint: at most {k} of value {value}")

    def ctr_exactly(self, lst: list[Variable], value: int, k: int | Variable):
        """Exactly constraint: exactly k occurrences of value in lst."""
        exprs = self._get_var_list(lst)
        count_terms = [If(e == value, 1, 0) for e in exprs]

        if isinstance(k, Variable):
            k_var = self.vars[k.id]
            self._add_constraint(Sum(count_terms) == k_var)
        else:
            self._add_constraint(Sum(count_terms) == k)
        self._log(2, f"Added Exactly constraint: exactly {k} of value {value}")

    def ctr_nvalues(
        self,
        lst: list[Variable] | list[Node],
        excepting: None | list[int],
        condition: Condition,
    ):
        """NValues constraint: number of distinct values."""
        exprs = self._get_var_or_node_list(lst)
        n = len(exprs)

        # Build domain info for each expression
        domains: list[set[int]] = []
        all_values: set[int] = set()

        for item in lst:
            if isinstance(item, Variable):
                vals = item.dom.all_values()
            elif isinstance(item, Node):
                vals = item.possible_values()
            else:
                vals = []

            if isinstance(vals, range):
                item_vals = set(vals)
            else:
                item_vals = set(vals) if vals else set()

            domains.append(item_vals)
            all_values.update(item_vals)

        if excepting:
            all_values -= set(excepting)

        # Early exit for trivial cases
        if not all_values or not exprs:
            self._apply_condition_to_model(Int(0), condition)
            self._log(2, "Added NValues constraint (trivial)")
            return

        # For each value, create indicator for whether it appears
        appears = []
        for val in sorted(all_values):
            # Only consider expressions that can take this value
            relevant_indices = [i for i in range(n) if val in domains[i]]

            if not relevant_indices:
                # No expression can take this value - skip
                continue

            if len(relevant_indices) == 1:
                # Only one expression can take this value - direct check
                e = exprs[relevant_indices[0]]
                appears.append(If(e == val, 1, 0))
            else:
                # Multiple expressions - check if any equals val
                eq_terms = [If(exprs[i] == val, 1, 0) for i in relevant_indices]
                appears.append(If(Sum(eq_terms) > 0, 1, 0))

        if not appears:
            self._apply_condition_to_model(Int(0), condition)
        else:
            nvalues = Sum(appears)
            self._apply_condition_to_model(nvalues, condition)

        self._log(2, f"Added NValues constraint ({len(appears)} values)")

    def ctr_element(
        self, lst: list[Variable] | list[int], i: Variable, condition: Condition
    ):
        """Element constraint: lst[i] satisfies condition."""
        index_var = self.vars[i.id]

        # Create target variable for element value
        if all(isinstance(v, int) for v in lst):
            min_val, max_val = min(lst), max(lst)
        else:
            min_val, max_val = -10**9, 10**9

        target = self.new_aux_int_var(min_val, max_val, "element_target")

        # Add element constraint: target == lst[index_var]
        for j, val in enumerate(lst):
            if isinstance(val, Variable):
                val = self.vars[val.id]
            self._add_constraint(Implies(index_var == j, target == val))

        self._apply_condition_to_model(target, condition)
        self._log(2, f"Added Element constraint")

    def ctr_minimum(
        self, lst: list[Variable] | list[Node], condition: Condition
    ):
        """Minimum constraint."""
        exprs = self._get_var_or_node_list(lst)
        min_expr = self._min(exprs)
        self._apply_condition_to_model(min_expr, condition)
        self._log(2, f"Added Minimum constraint")

    def ctr_maximum(
        self, lst: list[Variable] | list[Node], condition: Condition
    ):
        """Maximum constraint."""
        exprs = self._get_var_or_node_list(lst)
        max_expr = self._max(exprs)
        self._apply_condition_to_model(max_expr, condition)
        self._log(2, f"Added Maximum constraint")

    def ctr_channel(self, lst1: list[Variable], lst2: None | list[Variable]):
        """Channel/Inverse constraint."""
        vars1 = self._get_var_list(lst1)
        if lst2 is None:
            # Self-inverse: lst1[lst1[i]] = i
            vars2 = vars1
        else:
            vars2 = self._get_var_list(lst2)

        # x[i] == j <=> y[j] == i
        n1, n2 = len(vars1), len(vars2)
        for i in range(n1):
            for j in range(n2):
                self._add_constraint(
                    (vars1[i] == j) == (vars2[j] == i)
                )
        self._log(2, f"Added Channel constraint")

    def ctr_circuit(self, lst: list[Variable], size: None | int | Variable):
        """Circuit constraint (Hamiltonian cycle)."""
        vars_list = self._get_var_list(lst)
        n = len(vars_list)

        # Each position has exactly one successor
        for i in range(n):
            self._add_constraint(And(vars_list[i] >= 0, vars_list[i] < n))
            self._add_constraint(vars_list[i] != i)  # No self-loops

        # All successors different (forms a permutation)
        self._add_constraint(Distinct(*vars_list))

        # Subcycle elimination using position labels
        # position[i] gives the position of node i in the circuit
        positions = [Int(f"circuit_pos_{i}") for i in range(n)]
        for i in range(n):
            self._domain_constraints.append(And(positions[i] >= 0, positions[i] < n))

        # Position 0 is fixed to the starting node
        self._add_constraint(positions[0] == 0)

        # If successor[i] == j and i != 0, then position[j] == position[i] + 1
        for i in range(n):
            for j in range(n):
                if i != j and j != 0:
                    self._add_constraint(
                        Implies(vars_list[i] == j, positions[j] == positions[i] + 1)
                    )

        self._log(2, f"Added Circuit constraint on {n} nodes")

    def ctr_clause(self, pos: list[Variable], neg: list[Variable]):
        """Clause constraint (OR of positive and negated variables)."""
        literals = []
        for v in pos:
            # Treat as boolean: var != 0
            literals.append(self.vars[v.id] != 0)
        for v in neg:
            # Negated: var == 0
            literals.append(self.vars[v.id] == 0)
        self._add_constraint(Or(literals))
        self._log(
            2, f"Added Clause with {len(pos)} positive and {len(neg)} negative literals"
        )

    def ctr_nooverlap(
        self,
        origins: list[Variable],
        lengths: list[int] | list[Variable],
        zero_ignored: bool,
    ):
        """NoOverlap constraint (1D)."""
        n = len(origins)

        for i in range(n):
            for j in range(i + 1, n):
                start_i = self.vars[origins[i].id]
                start_j = self.vars[origins[j].id]

                if isinstance(lengths[i], int):
                    len_i = lengths[i]
                else:
                    len_i = self.vars[lengths[i].id]

                if isinstance(lengths[j], int):
                    len_j = lengths[j]
                else:
                    len_j = self.vars[lengths[j].id]

                # Either i ends before j starts, or j ends before i starts
                self._add_constraint(
                    Or(start_i + len_i <= start_j, start_j + len_j <= start_i)
                )

        self._log(2, f"Added NoOverlap constraint on {n} intervals")

    def ctr_cumulative(
        self,
        origins: list[Variable],
        lengths: list[int] | list[Variable],
        heights: list[int] | list[Variable],
        condition: Condition,
    ):
        """Cumulative constraint using event-based formulation."""
        from pycsp3.classes.auxiliary.conditions import ConditionValue, ConditionVariable

        # Get capacity from condition (supported: LE with constant/variable)
        if not isinstance(condition, (ConditionValue, ConditionVariable)):
            raise NotImplementedError(
                "Z3 cumulative requires ConditionValue or ConditionVariable"
            )
        if condition.operator != TypeConditionOperator.LE:
            raise NotImplementedError("Z3 cumulative only supports LE capacity conditions")

        capacity = (
            condition.value
            if isinstance(condition, ConditionValue)
            else self.vars[condition.variable.id]
        )

        n = len(origins)

        # Precompute solver expressions for starts, lengths, heights
        starts = [self.vars[origins[i].id] for i in range(n)]
        lens = []
        for i in range(n):
            if isinstance(lengths[i], int):
                lens.append(lengths[i])
            else:
                lens.append(self.vars[lengths[i].id])

        hts = []
        for i in range(n):
            if isinstance(heights[i], int):
                hts.append(heights[i])
            else:
                hts.append(self.vars[heights[i].id])

        # Event-based formulation: at each task's start time, check capacity
        # For each task i, at time start[i], the sum of heights of all tasks
        # that are active at that time must not exceed capacity.
        #
        # Task j is active at time start[i] if:
        #   start[j] <= start[i] < start[j] + length[j]
        #
        # This gives O(n²) constraints instead of O(T*n) for time-indexed.

        for i in range(n):
            # At time start[i], which tasks are active?
            # Task i is always active at its own start time
            overlapping_heights = [hts[i]]

            for j in range(n):
                if i != j:
                    # Task j is active at start[i] if start[j] <= start[i] < start[j] + lens[j]
                    j_active = And(starts[j] <= starts[i], starts[i] < starts[j] + lens[j])
                    overlapping_heights.append(If(j_active, hts[j], 0))

            self._add_constraint(Sum(overlapping_heights) <= capacity)

        # Also check at each task's end time for completeness
        # This handles edge cases where capacity is violated just before a task ends
        for i in range(n):
            end_i = starts[i] + lens[i] - 1  # Last active time point for task i
            overlapping_heights = [hts[i]]  # Task i is still active

            for j in range(n):
                if i != j:
                    # Task j is active at end_i if start[j] <= end_i < start[j] + lens[j]
                    j_active = And(starts[j] <= end_i, end_i < starts[j] + lens[j])
                    overlapping_heights.append(If(j_active, hts[j], 0))

            self._add_constraint(Sum(overlapping_heights) <= capacity)

        self._log(2, f"Added Cumulative constraint on {n} tasks (event-based, {2*n} check points)")

    def ctr_ordered(
        self,
        lst: list[Variable],
        operator: TypeOrderedOperator,
        lengths: None | list[int] | list[Variable],
    ):
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
                self._add_constraint(vars_list[i] + offset < vars_list[i + 1])
            elif operator == TypeOrderedOperator.INCREASING:
                self._add_constraint(vars_list[i] + offset <= vars_list[i + 1])
            elif operator == TypeOrderedOperator.STRICTLY_DECREASING:
                self._add_constraint(vars_list[i] + offset > vars_list[i + 1])
            elif operator == TypeOrderedOperator.DECREASING:
                self._add_constraint(vars_list[i] + offset >= vars_list[i + 1])

        self._log(2, f"Added Ordered constraint with {operator}")

    def ctr_lex(self, lists: list[list[Variable]], operator: TypeOrderedOperator):
        """Lexicographic constraint on lists."""

        def add_lex_pair(vars1: list[Any], vars2: list[Any], strict: bool) -> None:
            if len(vars1) != len(vars2):
                raise ValueError("Lex constraint requires lists of equal length")
            if not vars1:
                return

            n = len(vars1)
            # Build lexicographic constraint recursively
            # lex_le(x, y) = (x[0] < y[0]) OR (x[0] == y[0] AND lex_le(x[1:], y[1:]))

            def lex_constraint(idx: int, strict_required: bool) -> Any:
                if idx >= n:
                    return not strict_required  # If we've checked all, equal is OK unless strict
                if idx == n - 1:
                    if strict_required:
                        return vars1[idx] < vars2[idx]
                    else:
                        return vars1[idx] <= vars2[idx]
                # x[idx] < y[idx] OR (x[idx] == y[idx] AND lex_constraint(idx+1, strict_required))
                return Or(
                    vars1[idx] < vars2[idx],
                    And(vars1[idx] == vars2[idx], lex_constraint(idx + 1, strict_required)),
                )

            self._add_constraint(lex_constraint(0, strict))

        increasing = operator in (
            TypeOrderedOperator.INCREASING,
            TypeOrderedOperator.STRICTLY_INCREASING,
        )
        strict = operator in (
            TypeOrderedOperator.STRICTLY_INCREASING,
            TypeOrderedOperator.STRICTLY_DECREASING,
        )

        for i in range(len(lists) - 1):
            vars1 = self._get_var_list(lists[i])
            vars2 = self._get_var_list(lists[i + 1])
            if increasing:
                add_lex_pair(vars1, vars2, strict)
            else:
                add_lex_pair(vars2, vars1, strict)

        self._log(2, f"Added Lex constraint on {len(lists)} lists")

    def ctr_lex_matrix(
        self, matrix: list[list[Variable]], operator: TypeOrderedOperator
    ):
        """Lexicographic constraint on matrix rows."""
        self.ctr_lex(matrix, operator)

    def ctr_instantiation(self, lst: list[Variable], values: list[int]):
        """Instantiation constraint (fix variables to values)."""
        for var, val in zip(lst, values):
            self._add_constraint(self.vars[var.id] == val)
        self._log(2, f"Added Instantiation constraint on {len(lst)} vars")

    # ========== Packing constraints ==========

    def _binpacking_values(
        self, lst: list[Variable], bin_count: int | None
    ) -> list[int]:
        values_set: set[int] = set()
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
            allowed_values = values
            for var in lst:
                self._add_constraint(
                    Or([self.vars[var.id] == v for v in allowed_values])
                )
        return values

    def _binpacking_load_exprs(
        self, bins: list[Any], sizes: list[int], values: list[int]
    ) -> list[Any]:
        if not all(isinstance(s, int) for s in sizes):
            raise NotImplementedError("Z3 binpacking only supports constant sizes")
        load_exprs = []
        for value in values:
            terms = [
                If(bin_var == value, size, 0) for bin_var, size in zip(bins, sizes)
            ]
            load_exprs.append(Sum(terms))
        return load_exprs

    def ctr_binpacking(
        self, lst: list[Variable], sizes: list[int], condition: Condition
    ):
        """BinPacking with a single condition on each bin load."""
        bins = self._get_var_list(lst)
        values = self._binpacking_values(lst, None)
        load_exprs = self._binpacking_load_exprs(bins, sizes, values)
        for load_expr in load_exprs:
            self._apply_condition_to_model(load_expr, condition)
        self._log(2, "Added BinPacking constraint with condition")

    def ctr_binpacking_limits(
        self,
        lst: list[Variable],
        sizes: list[int],
        limits: list[int] | list[Variable],
    ):
        """BinPacking with per-bin limits."""
        bins = self._get_var_list(lst)
        values = self._binpacking_values(lst, len(limits))
        load_exprs = self._binpacking_load_exprs(bins, sizes, values)
        for load_expr, limit in zip(load_exprs, limits):
            bound = self.vars[limit.id] if isinstance(limit, Variable) else int(limit)
            self._add_constraint(load_expr <= bound)
        self._log(2, "Added BinPacking constraint with limits")

    def ctr_binpacking_loads(
        self,
        lst: list[Variable],
        sizes: list[int],
        loads: list[int] | list[Variable],
    ):
        """BinPacking with explicit load variables."""
        bins = self._get_var_list(lst)
        values = self._binpacking_values(lst, len(loads))
        load_exprs = self._binpacking_load_exprs(bins, sizes, values)
        for load_expr, load in zip(load_exprs, loads):
            target = self.vars[load.id] if isinstance(load, Variable) else int(load)
            self._add_constraint(load_expr == target)
        self._log(2, "Added BinPacking constraint with loads")

    def ctr_binpacking_conditions(
        self,
        lst: list[Variable],
        sizes: list[int],
        conditions: list[Condition],
    ):
        """BinPacking with per-bin conditions."""
        bins = self._get_var_list(lst)
        values = self._binpacking_values(lst, len(conditions))
        load_exprs = self._binpacking_load_exprs(bins, sizes, values)
        for load_expr, condition in zip(load_exprs, conditions):
            self._apply_condition_to_model(load_expr, condition)
        self._log(2, "Added BinPacking constraint with conditions")

    def ctr_knapsack(
        self,
        lst: list[Variable],
        weights: list[int],
        wcondition: Condition,
        profits: list[int],
        pcondition: Condition,
    ):
        """Knapsack constraint: weight and profit conditions on selected items."""
        selection = self._get_var_list(lst)

        # Weight constraint: sum(weights[i] * lst[i]) satisfies wcondition
        weight_sum = Sum([w * s for w, s in zip(weights, selection)])
        self._apply_condition_to_model(weight_sum, wcondition)

        # Profit constraint: sum(profits[i] * lst[i]) satisfies pcondition
        profit_sum = Sum([p * s for p, s in zip(profits, selection)])
        self._apply_condition_to_model(profit_sum, pcondition)

        self._log(2, f"Added Knapsack constraint on {len(lst)} items")

    # ========== Primitive constraints (optimized) ==========

    def ctr_primitive1a(self, x: Variable, op: TypeConditionOperator, k: int):
        """x op k"""
        var = self.vars[x.id]
        self._add_simple_comparison(var, op, k)

    def ctr_primitive1b(
        self, x: Variable, op: TypeConditionOperator, term: list[int] | range
    ):
        """x in/notin set"""
        var = self.vars[x.id]
        values = list(term)
        if op == TypeConditionOperator.IN:
            self._add_constraint(Or([var == v for v in values]))
        else:
            self._add_constraint(And([var != v for v in values]))

    def ctr_primitive2b(
        self,
        x: Variable,
        aop: TypeArithmeticOperator,
        y: Variable,
        op: TypeConditionOperator,
        k: int,
    ):
        """x aop y op k"""
        vx = self.vars[x.id]
        vy = self.vars[y.id]

        if aop == TypeArithmeticOperator.ADD:
            expr = vx + vy
        elif aop == TypeArithmeticOperator.SUB:
            expr = vx - vy
        elif aop == TypeArithmeticOperator.MUL:
            expr = vx * vy
        elif aop == TypeArithmeticOperator.DIV:
            expr = vx / vy
        elif aop == TypeArithmeticOperator.MOD:
            expr = vx % vy
        elif aop == TypeArithmeticOperator.DIST:
            expr = self._abs(vx - vy)
        else:
            raise NotImplementedError(f"Arithmetic operator {aop} not implemented")

        self._add_simple_comparison(expr, op, k)

    def ctr_primitive3(
        self,
        x: Variable,
        aop: TypeArithmeticOperator,
        y: Variable,
        op: TypeConditionOperator,
        z: Variable,
    ):
        """x aop y op z"""
        vx = self.vars[x.id]
        vy = self.vars[y.id]
        vz = self.vars[z.id]

        if aop == TypeArithmeticOperator.ADD:
            expr = vx + vy
        elif aop == TypeArithmeticOperator.SUB:
            expr = vx - vy
        elif aop == TypeArithmeticOperator.MUL:
            expr = vx * vy
        else:
            raise NotImplementedError(f"Arithmetic operator {aop} not implemented")

        self._add_simple_comparison(expr, op, vz)

    # ========== Objectives ==========

    def obj_minimize(self, term: Variable | Node):
        """Minimize objective."""
        self._has_objective = True
        if isinstance(term, Variable):
            self._objective = self.vars[term.id]
        else:
            self._objective = self.translate_node(term)
        self._minimize = True
        self._log(1, "Set minimization objective")

    def obj_maximize(self, term: Variable | Node):
        """Maximize objective."""
        self._has_objective = True
        if isinstance(term, Variable):
            self._objective = self.vars[term.id]
        else:
            self._objective = self.translate_node(term)
        self._minimize = False
        self._log(1, "Set maximization objective")

    def obj_minimize_special(
        self,
        obj_type: TypeObj,
        terms: list[Variable] | list[Node],
        coefficients: None | list[int],
    ):
        """Minimize special objective (sum, product, etc.)."""
        exprs = self._get_var_or_node_list(terms)

        if obj_type == TypeObj.SUM:
            if coefficients is None:
                obj_expr = Sum(exprs)
            else:
                obj_expr = Sum([c * e for c, e in zip(coefficients, exprs)])
        elif obj_type == TypeObj.MAXIMUM:
            obj_expr = self._max(exprs)
        elif obj_type == TypeObj.MINIMUM:
            obj_expr = self._min(exprs)
        elif obj_type == TypeObj.NVALUES:
            raise NotImplementedError("NVALUES objective not implemented")
        else:
            raise NotImplementedError(f"Objective type {obj_type} not implemented")

        self._has_objective = True
        self._objective = obj_expr
        self._minimize = True
        self._log(1, f"Set {obj_type} minimization objective")

    def obj_maximize_special(
        self,
        obj_type: TypeObj,
        terms: list[Variable] | list[Node],
        coefficients: None | list[int],
    ):
        """Maximize special objective."""
        exprs = self._get_var_or_node_list(terms)

        if obj_type == TypeObj.SUM:
            if coefficients is None:
                obj_expr = Sum(exprs)
            else:
                obj_expr = Sum([c * e for c, e in zip(coefficients, exprs)])
        elif obj_type == TypeObj.MAXIMUM:
            obj_expr = self._max(exprs)
        elif obj_type == TypeObj.MINIMUM:
            obj_expr = self._min(exprs)
        else:
            raise NotImplementedError(f"Objective type {obj_type} not implemented")

        self._has_objective = True
        self._objective = obj_expr
        self._minimize = False
        self._log(1, f"Set {obj_type} maximization objective")

    # ========== Solving ==========

    def solve(self) -> TypeStatus:
        """Solve the model and return status."""
        from z3 import set_param

        # Enable Z3 verbose mode if verbosity >= 2
        if self.verbose >= 2:
            set_param('verbose', 10)

        if self._has_objective:
            solver = Optimize()
        else:
            solver = Solver()

        # Add all domain constraints
        for constraint in self._domain_constraints:
            solver.add(constraint)

        # Add all model constraints
        for constraint in self._constraints:
            solver.add(constraint)

        # Set time limit if specified
        if self.time_limit:
            solver.set("timeout", int(self.time_limit * 1000))

        # Add objective if optimization
        if self._has_objective:
            if self._minimize:
                solver.minimize(self._objective)
            else:
                solver.maximize(self._objective)

        want_all = self.sols == "all" or (isinstance(self.sols, int) and self.sols > 1)
        if want_all and not self._has_objective:
            limit = None if self.sols == "all" else int(self.sols)
            return self._solve_all_solutions(solver, limit)
        self._all_solutions = []

        self._log(1, f"Starting Z3 solver...")
        self._log(1, f"  Variables: {len(self.vars)}")
        self._log(1, f"  Constraints: {len(self._constraints)} + {len(self._domain_constraints)} domain")
        if self._has_objective:
            self._log(1, f"  Objective: {'minimize' if self._minimize else 'maximize'}")

        # Solve
        result = solver.check()

        # Show statistics if verbose
        if self.verbose >= 2:
            stats = solver.statistics()
            self._log(2, f"Z3 Statistics:")
            for key in stats.keys():
                self._log(2, f"  {key}: {stats.get_key_value(key)}")

        if result == sat:
            model = solver.model()
            self._solution = {}
            for var_id, var in self.vars.items():
                val = model.eval(var, model_completion=True)
                if val is not None:
                    try:
                        self._solution[var_id] = val.as_long()
                    except AttributeError:
                        # Handle case where value can't be converted
                        self._solution[var_id] = 0

            if self._has_objective:
                obj_val = model.eval(self._objective, model_completion=True)
                if obj_val is not None:
                    try:
                        self._objective_value = obj_val.as_long()
                    except AttributeError:
                        self._objective_value = None
                self._status = TypeStatus.OPTIMUM
                self._log(1, f"Solver finished: OPTIMUM (objective={self._objective_value})")
                return TypeStatus.OPTIMUM

            self._status = TypeStatus.SAT
            self._log(1, f"Solver finished: SAT")
            return TypeStatus.SAT

        elif result == unsat:
            self._status = TypeStatus.UNSAT
            self._log(1, f"Solver finished: UNSAT")
            return TypeStatus.UNSAT

        else:
            self._status = TypeStatus.UNKNOWN
            self._log(1, f"Solver finished: UNKNOWN")
            return TypeStatus.UNKNOWN

    def _solve_all_solutions(self, solver: Solver, limit: int | None) -> TypeStatus:
        """Enumerate all/multiple solutions for satisfaction problems."""
        self._all_solutions = []

        while True:
            result = solver.check()
            if result != sat:
                break

            model = solver.model()
            sol: dict[str, int] = {}
            for var_id, var in self.vars.items():
                val = model.eval(var, model_completion=True)
                if val is not None:
                    try:
                        sol[var_id] = val.as_long()
                    except AttributeError:
                        sol[var_id] = 0
            if sol:
                self._all_solutions.append(sol)

            if limit is not None and len(self._all_solutions) >= limit:
                break

            if not sol:
                break

            block = Or([self.vars[var_id] != val for var_id, val in sol.items()])
            solver.add(block)

        if self._all_solutions:
            self._solution = self._all_solutions[-1]
            self._status = TypeStatus.SAT
            self._log(1, f"Solver finished: SAT ({len(self._all_solutions)} solutions)")
            return TypeStatus.SAT

        if result == unsat:
            self._status = TypeStatus.UNSAT
            self._log(1, "Solver finished: UNSAT")
            return TypeStatus.UNSAT

        self._status = TypeStatus.UNKNOWN
        self._log(1, "Solver finished: UNKNOWN")
        return TypeStatus.UNKNOWN

    def get_solution(self) -> dict[str, int] | None:
        """Return the solution as {var_id: value} dict, or None if no solution."""
        return self._solution

    def get_all_solutions(self) -> list[dict[str, int]]:
        """Return all solutions found."""
        return self._all_solutions
