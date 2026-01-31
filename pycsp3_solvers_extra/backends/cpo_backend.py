"""
CPO (IBM DOcplex CP Optimizer) backend for pycsp3-solvers-extra.

This module provides the CPOCallbacks class that implements the XCSP3 callback
interface using IBM DOcplex CP Optimizer.

Note: Requires IBM CPLEX Optimization Studio to be installed.
"""

from __future__ import annotations

import math
from typing import Any

from pycsp3.classes.auxiliary.enums import TypeStatus, TypeConditionOperator, TypeObj, TypeOrderedOperator
from pycsp3.classes.auxiliary.conditions import Condition
from pycsp3.classes.auxiliary.tables import to_ordinary_table
from pycsp3.classes.nodes import Node
from pycsp3.tools.utilities import ANY

from pycsp3_solvers_extra.backends.base import BaseCallbacks

from pycsp3.classes.main.variables import Variable
from pycsp3.classes.nodes import TypeNode

# Try to import docplex
try:
    from docplex.cp.model import CpoModel
    from docplex.cp.solution import CpoSolveResult
    from docplex.cp.solver.solver_listener import CpoSolverListener
    import docplex.cp.modeler as modeler
    from docplex.cp.catalog import Type_BoolExpr
    from docplex.cp.config import context as cpo_context
    CPO_AVAILABLE = True
except ImportError:
    CPO_AVAILABLE = False
    cpo_context = None
    Type_BoolExpr = None
    CpoSolverListener = None


def _configure_cpoptimizer():
    """Auto-detect and configure cpoptimizer executable path."""
    import os
    import platform
    import glob

    if not CPO_AVAILABLE:
        return False

    # Check if already configured and working
    execfile = cpo_context.solver.local.execfile
    if execfile and os.path.isfile(execfile):
        return True

    # Common installation paths by platform
    system = platform.system()
    machine = platform.machine()

    search_paths = []

    if system == "Darwin":  # macOS
        # Determine architecture
        if machine == "arm64":
            arch_dir = "arm64_osx"
        else:
            arch_dir = "x86-64_osx"

        # Search in common locations
        search_paths = [
            f"/Applications/CPLEX_Studio*/cpoptimizer/bin/{arch_dir}/cpoptimizer",
            f"/opt/ibm/ILOG/CPLEX_Studio*/cpoptimizer/bin/{arch_dir}/cpoptimizer",
            os.path.expanduser(f"~/Applications/CPLEX_Studio*/cpoptimizer/bin/{arch_dir}/cpoptimizer"),
        ]
    elif system == "Linux":
        if machine == "x86_64":
            arch_dir = "x86-64_linux"
        elif machine == "aarch64":
            arch_dir = "arm64_linux"
        else:
            arch_dir = "x86-64_linux"

        search_paths = [
            f"/opt/ibm/ILOG/CPLEX_Studio*/cpoptimizer/bin/{arch_dir}/cpoptimizer",
            f"/opt/CPLEX_Studio*/cpoptimizer/bin/{arch_dir}/cpoptimizer",
            os.path.expanduser(f"~/CPLEX_Studio*/cpoptimizer/bin/{arch_dir}/cpoptimizer"),
        ]
    elif system == "Windows":
        search_paths = [
            r"C:\Program Files\IBM\ILOG\CPLEX_Studio*\cpoptimizer\bin\x64_win64\cpoptimizer.exe",
            r"C:\Program Files (x86)\IBM\ILOG\CPLEX_Studio*\cpoptimizer\bin\x64_win64\cpoptimizer.exe",
        ]

    # Also check CPLEX_STUDIO_DIR environment variable
    cplex_dir = os.environ.get("CPLEX_STUDIO_DIR")
    if cplex_dir:
        if system == "Darwin":
            arch_dir = "arm64_osx" if machine == "arm64" else "x86-64_osx"
            search_paths.insert(0, os.path.join(cplex_dir, "cpoptimizer", "bin", arch_dir, "cpoptimizer"))
        elif system == "Linux":
            arch_dir = "arm64_linux" if machine == "aarch64" else "x86-64_linux"
            search_paths.insert(0, os.path.join(cplex_dir, "cpoptimizer", "bin", arch_dir, "cpoptimizer"))
        elif system == "Windows":
            search_paths.insert(0, os.path.join(cplex_dir, "cpoptimizer", "bin", "x64_win64", "cpoptimizer.exe"))

    # Search for the executable
    for pattern in search_paths:
        matches = glob.glob(pattern)
        if matches:
            # Use the most recent version (sorted by name, last is usually newest)
            matches.sort()
            execfile = matches[-1]
            if os.path.isfile(execfile):
                cpo_context.solver.local.execfile = execfile
                return True

    return False


# Try to configure cpoptimizer on module load
_CPO_CONFIGURED = _configure_cpoptimizer() if CPO_AVAILABLE else False


class CPOCallbacks(BaseCallbacks):
    """
    XCSP3 callback implementation for IBM DOcplex CP Optimizer.

    This class receives callbacks from the XCSP3 parser and builds
    a DOcplex CP model, then solves it.
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

        if not CPO_AVAILABLE:
            raise ImportError(
                "DOcplex is not available. Install it with: pip install docplex\n"
                "Note: CP Optimizer solver requires IBM CPLEX Optimization Studio."
            )

        if not _CPO_CONFIGURED:
            raise ImportError(
                "CP Optimizer executable not found.\n"
                "Please ensure IBM CPLEX Optimization Studio is installed and either:\n"
                "  1. Add cpoptimizer to your PATH, or\n"
                "  2. Set CPLEX_STUDIO_DIR environment variable, or\n"
                "  3. Manually configure: from docplex.cp.config import context; "
                "context.solver.local.execfile = '/path/to/cpoptimizer'"
            )

        self.model = CpoModel()
        self.vars: dict[str, Any] = {}  # var_id -> CPO variable
        self.solution: CpoSolveResult | None = None
        self._objective_expr = None
        self._minimize = True
        self._all_solutions: list[dict[str, int]] = []

    # ========== Variable callbacks ==========

    def var_integer_range(self, x: Variable, min_val: int, max_val: int):
        """Create integer variable with range domain."""
        var = self.model.integer_var(min_val, max_val, x.id)
        self.vars[x.id] = var
        self._log(2, f"Created var {x.id} in [{min_val}, {max_val}]")

    def new_aux_int_var(self, lb: int, ub: int, name_hint: str = "aux") -> Any:
        """Create an auxiliary integer variable."""
        return self.model.integer_var(lb, ub, name_hint)

    def var_integer(self, x: Variable, values: list[int]):
        """Create integer variable with explicit domain."""
        # DOcplex uses domain() for explicit domains
        var = self.model.integer_var(domain=values, name=x.id)
        self.vars[x.id] = var
        self._log(2, f"Created var {x.id} with domain {values}")

    def var_symbolic(self, x: Variable, values: list[str]):
        """Create symbolic variable - map to integers."""
        # Map symbolic values to integers
        value_map = {v: i for i, v in enumerate(values)}
        var = self.model.integer_var(0, len(values) - 1, x.id)
        self.vars[x.id] = var
        # Store the mapping for solution extraction
        if not hasattr(self, '_symbolic_maps'):
            self._symbolic_maps = {}
        self._symbolic_maps[x.id] = {i: v for v, i in value_map.items()}
        self._log(2, f"Created symbolic var {x.id} with values {values}")

    # ========== Helper methods ==========

    def _get_var(self, var: Variable) -> Any:
        """Get CPO variable for a pycsp3 variable."""
        return self.vars[var.id]

    def _get_var_list(self, variables: list[Variable]) -> list[Any]:
        """Get list of CPO variables."""
        return [self.vars[v.id] for v in variables]

    def _get_var_or_node_list(self, items: list[Variable] | list[Node]) -> list[Any]:
        """Get list of CPO expressions from variables or nodes."""
        result = []
        for item in items:
            if isinstance(item, Node):
                result.append(self.translate_node(item))
            else:
                result.append(self.vars[item.id])
        return result

    def _as_bool_expr(self, expr: Any) -> Any:
        """Coerce numeric expressions to boolean CP expressions."""
        if isinstance(expr, bool):
            return expr
        if Type_BoolExpr is not None and hasattr(expr, "is_kind_of") and expr.is_kind_of(Type_BoolExpr):
            return expr
        return expr != 0

    def _translate_node_impl(self, node: Node) -> Any:
        """Translate expression tree node to CPO expression."""
        from pycsp3.classes.nodes import TypeNode

        if node.type == TypeNode.VAR:
            return self.vars[node.cnt.id]
        elif node.type == TypeNode.INT:
            return node.cnt
        elif node.type == TypeNode.ADD:
            return sum(self.translate_node(c) for c in node.cnt)
        elif node.type == TypeNode.SUB:
            sons = [self.translate_node(c) for c in node.cnt]
            result = sons[0]
            for s in sons[1:]:
                result = result - s
            return result
        elif node.type == TypeNode.MUL:
            result = self.translate_node(node.cnt[0])
            for c in node.cnt[1:]:
                result = result * self.translate_node(c)
            return result
        elif node.type == TypeNode.DIV:
            return self.translate_node(node.cnt[0]) // self.translate_node(node.cnt[1])
        elif node.type == TypeNode.MOD:
            return self.translate_node(node.cnt[0]) % self.translate_node(node.cnt[1])
        elif node.type == TypeNode.NEG:
            return -self.translate_node(node.cnt[0])
        elif node.type == TypeNode.ABS:
            return modeler.abs(self.translate_node(node.cnt[0]))
        elif node.type == TypeNode.MIN:
            return modeler.min([self.translate_node(c) for c in node.cnt])
        elif node.type == TypeNode.MAX:
            return modeler.max([self.translate_node(c) for c in node.cnt])
        elif node.type == TypeNode.DIST:
            a, b = self.translate_node(node.cnt[0]), self.translate_node(node.cnt[1])
            return modeler.abs(a - b)
        elif node.type == TypeNode.EQ:
            return self.translate_node(node.cnt[0]) == self.translate_node(node.cnt[1])
        elif node.type == TypeNode.NE:
            return self.translate_node(node.cnt[0]) != self.translate_node(node.cnt[1])
        elif node.type == TypeNode.LT:
            return self.translate_node(node.cnt[0]) < self.translate_node(node.cnt[1])
        elif node.type == TypeNode.LE:
            return self.translate_node(node.cnt[0]) <= self.translate_node(node.cnt[1])
        elif node.type == TypeNode.GT:
            return self.translate_node(node.cnt[0]) > self.translate_node(node.cnt[1])
        elif node.type == TypeNode.GE:
            return self.translate_node(node.cnt[0]) >= self.translate_node(node.cnt[1])
        elif node.type == TypeNode.AND:
            return modeler.logical_and([self._as_bool_expr(self.translate_node(c)) for c in node.cnt])
        elif node.type == TypeNode.OR:
            return modeler.logical_or([self._as_bool_expr(self.translate_node(c)) for c in node.cnt])
        elif node.type == TypeNode.NOT:
            return modeler.logical_not(self._as_bool_expr(self.translate_node(node.cnt[0])))
        elif node.type == TypeNode.IMP:
            a = self._as_bool_expr(self.translate_node(node.cnt[0]))
            b = self._as_bool_expr(self.translate_node(node.cnt[1]))
            return modeler.if_then(a, b)
        elif node.type == TypeNode.IFF:
            a = self._as_bool_expr(self.translate_node(node.cnt[0]))
            b = self._as_bool_expr(self.translate_node(node.cnt[1]))
            return a == b
        elif node.type == TypeNode.IF:
            cond = self._as_bool_expr(self.translate_node(node.cnt[0]))
            then_val = self.translate_node(node.cnt[1])
            else_val = self.translate_node(node.cnt[2])
            return modeler.conditional(cond, then_val, else_val)
        elif node.type == TypeNode.SQR:
            val = self.translate_node(node.cnt[0])
            return val * val
        elif node.type == TypeNode.POW:
            base = self.translate_node(node.cnt[0])
            exp = self.translate_node(node.cnt[1])
            return modeler.power(base, exp)
        else:
            raise NotImplementedError(f"Node type {node.type} not implemented for CPO")

    def _apply_condition(self, expr: Any, condition: Condition):
        """Apply condition to expression and add constraint."""
        from pycsp3.classes.auxiliary.conditions import (
            ConditionValue, ConditionVariable, ConditionInterval, ConditionSet
        )

        op = condition.operator

        if isinstance(condition, ConditionValue):
            right = condition.value
        elif isinstance(condition, ConditionVariable):
            right = self.vars[condition.variable.id]
        elif isinstance(condition, ConditionInterval):
            # Range condition
            if op == TypeConditionOperator.IN:
                self.model.add(expr >= condition.min)
                self.model.add(expr <= condition.max)
            else:  # NOTIN
                self.model.add(modeler.logical_or(expr < condition.min, expr > condition.max))
            return
        elif isinstance(condition, ConditionSet):
            values = list(condition.t)
            if op == TypeConditionOperator.IN:
                self.model.add(modeler.allowed_assignments(expr, values))
            else:
                self.model.add(modeler.forbidden_assignments(expr, values))
            return
        else:
            raise NotImplementedError(f"Condition type {type(condition)} not implemented")

        # Simple comparison
        if op == TypeConditionOperator.EQ:
            self.model.add(expr == right)
        elif op == TypeConditionOperator.NE:
            self.model.add(expr != right)
        elif op == TypeConditionOperator.LT:
            self.model.add(expr < right)
        elif op == TypeConditionOperator.LE:
            self.model.add(expr <= right)
        elif op == TypeConditionOperator.GT:
            self.model.add(expr > right)
        elif op == TypeConditionOperator.GE:
            self.model.add(expr >= right)

    # ========== Set membership operations ==========

    def _in_set(self, val: Any, set_vals: list[Any]) -> Any:
        """Set membership with optimizations.

        Optimizations applied (based on benchmarks):
        1. Empty set: return false constant
        2. Consecutive integer range (>=3 values): use range constraint
        """
        if not set_vals:
            return modeler.false()

        # Check if all values are constant integers
        const_vals = [v for v in set_vals if isinstance(v, int)]
        if len(const_vals) == len(set_vals) and len(const_vals) >= 3:
            sorted_vals = sorted(const_vals)
            min_val, max_val = sorted_vals[0], sorted_vals[-1]

            # Check if values form a consecutive range
            if max_val - min_val + 1 == len(sorted_vals):
                return modeler.logical_and(val >= min_val, val <= max_val)

        # General case: OR of equalities
        return modeler.logical_or([val == v for v in set_vals])

    def _not_in_set(self, val: Any, set_vals: list[Any]) -> Any:
        """Set non-membership with optimizations.

        Optimizations applied (based on benchmarks):
        1. Empty set: return true constant
        2. Consecutive integer range (>=3 values): use Or(val < min, val > max)
        """
        if not set_vals:
            return modeler.true()

        # Note: Single value optimization removed based on Z3 benchmarks

        # Check if all values are constant integers
        const_vals = [v for v in set_vals if isinstance(v, int)]
        if len(const_vals) == len(set_vals) and len(const_vals) >= 3:
            sorted_vals = sorted(const_vals)
            min_val, max_val = sorted_vals[0], sorted_vals[-1]

            # Check if values form a consecutive range
            if max_val - min_val + 1 == len(sorted_vals):
                return modeler.logical_or(val < min_val, val > max_val)

        # General case: AND of inequalities
        return modeler.logical_and([val != v for v in set_vals])

    # ========== Constraint callbacks ==========

    def ctr_intension(self, scope: list[Variable], tree: Node):
        """Add intension constraint."""
        expr = self.translate_node(tree)
        self.model.add(expr)
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
            self.model.add(modeler.allowed_assignments(var, expanded))
        else:
            self.model.add(modeler.forbidden_assignments(var, expanded))
        self._log(2, f"Added {'positive' if positive else 'negative'} unary extension")

    def ctr_extension(self, scope: list[Variable], tuples: list, positive: bool, flags: set[str]):
        """Table constraint."""
        vars_list = self._get_var_list(scope)
        if tuples is None:
            return

        # Expand starred tuples (ANY values)
        needs_expansion = any(
            (v == ANY) or not isinstance(v, int) for t in tuples for v in t
        )
        if needs_expansion:
            doms = [v.dom for v in scope]
            tuples = to_ordinary_table(tuples, doms)

        if positive:
            self.model.add(modeler.allowed_assignments(vars_list, tuples))
        else:
            self.model.add(modeler.forbidden_assignments(vars_list, tuples))
        self._log(2, f"Added {'positive' if positive else 'negative'} extension on {len(scope)} vars")

    def ctr_instantiation(self, lst: list[Variable], values: list[int]):
        """Instantiation constraint (fix variables to values)."""
        for var, val in zip(lst, values):
            self.model.add(self.vars[var.id] == val)
        self._log(2, f"Added Instantiation constraint on {len(lst)} vars")

    def _binpacking_bins(self, lst: list[Variable], bin_count: int | None = None):
        min_val = min(v.dom.smallest_value() for v in lst)
        max_val = max(v.dom.greatest_value() for v in lst)
        count = bin_count if bin_count is not None else (max_val - min_val + 1)
        bins = [self.vars[v.id] - min_val if min_val != 0 else self.vars[v.id] for v in lst]
        return bins, count

    def _binpacking_load_vars(self, sizes: list[int], bin_count: int, limits: None | list[int] | list[Variable]):
        if not all(isinstance(s, int) for s in sizes):
            raise NotImplementedError("CPO binpacking only supports constant sizes")
        total = sum(sizes)
        loads = []
        if limits is None:
            for i in range(bin_count):
                loads.append(self.model.integer_var(0, total, f"_binload_{i}"))
        else:
            for i, limit in enumerate(limits):
                if isinstance(limit, Variable):
                    ub = limit.dom.greatest_value()
                else:
                    ub = int(limit)
                loads.append(self.model.integer_var(0, max(0, ub), f"_binload_{i}"))
        return loads

    def ctr_binpacking(self, lst: list[Variable], sizes: list[int], condition: Condition):
        """BinPacking with a single condition on each bin load."""
        bins, bin_count = self._binpacking_bins(lst)
        loads = self._binpacking_load_vars(sizes, bin_count, None)
        self.model.add(modeler.pack(loads, bins, sizes))
        for load in loads:
            self._apply_condition(load, condition)
        self._log(2, f"Added BinPacking constraint with condition")

    def ctr_binpacking_limits(self, lst: list[Variable], sizes: list[int], limits: list[int] | list[Variable]):
        """BinPacking with per-bin limits."""
        bins, _ = self._binpacking_bins(lst, bin_count=len(limits))
        loads = self._binpacking_load_vars(sizes, len(limits), limits)
        self.model.add(modeler.pack(loads, bins, sizes))
        for load, limit in zip(loads, limits):
            bound = self.vars[limit.id] if isinstance(limit, Variable) else int(limit)
            self.model.add(load <= bound)
        self._log(2, f"Added BinPacking constraint with limits")

    def ctr_binpacking_loads(self, lst: list[Variable], sizes: list[int], loads: list[int] | list[Variable]):
        """BinPacking with explicit load variables."""
        bins, _ = self._binpacking_bins(lst, bin_count=len(loads))
        load_vars = [self.vars[v.id] if isinstance(v, Variable) else int(v) for v in loads]
        self.model.add(modeler.pack(load_vars, bins, sizes))
        self._log(2, f"Added BinPacking constraint with loads")

    def ctr_binpacking_conditions(self, lst: list[Variable], sizes: list[int], conditions: list[Condition]):
        """BinPacking with per-bin conditions."""
        bins, bin_count = self._binpacking_bins(lst, bin_count=len(conditions))
        loads = self._binpacking_load_vars(sizes, bin_count, None)
        self.model.add(modeler.pack(loads, bins, sizes))
        for load, condition in zip(loads, conditions):
            self._apply_condition(load, condition)
        self._log(2, f"Added BinPacking constraint with conditions")

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
        weight_sum = modeler.scal_prod(selection, weights)
        self._apply_condition(weight_sum, wcondition)

        # Profit constraint: sum(profits[i] * lst[i]) satisfies pcondition
        profit_sum = modeler.scal_prod(selection, profits)
        self._apply_condition(profit_sum, pcondition)

        self._log(2, f"Added Knapsack constraint on {len(lst)} items")

    def ctr_all_different(self, scope: list[Variable] | list[Node], excepting: None | list[int]):
        """AllDifferent constraint."""
        exprs = self._get_var_or_node_list(scope)
        n = len(exprs)

        if excepting is None or len(excepting) == 0:
            self.model.add(modeler.all_diff(exprs))
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

        # Variables that cannot take excepted values - use native all_diff
        if len(cannot_be_excepted) >= 2:
            self.model.add(modeler.all_diff([exprs[i] for i in cannot_be_excepted]))

        # Variables that cannot be excepted must differ from those that can
        for i in cannot_be_excepted:
            for j in can_be_excepted:
                self.model.add(exprs[i] != exprs[j])

        # Among variables that can be excepted, use pairwise conditional constraints
        for idx_a in range(len(can_be_excepted)):
            for idx_b in range(idx_a + 1, len(can_be_excepted)):
                i, j = can_be_excepted[idx_a], can_be_excepted[idx_b]
                xi_except = modeler.logical_or([exprs[i] == v for v in excepting])
                xj_except = modeler.logical_or([exprs[j] == v for v in excepting])
                self.model.add(modeler.logical_or(exprs[i] != exprs[j], xi_except, xj_except))

        self._log(2, f"Added AllDifferent on {n} vars except {excepting} "
                     f"({len(cannot_be_excepted)} global, {len(can_be_excepted)} conditional)")

    def ctr_all_equal(self, scope: list[Variable] | list[Node], excepting: None | list[int]):
        """AllEqual constraint."""
        exprs = self._get_var_or_node_list(scope)
        if excepting is None or len(excepting) == 0:
            for i in range(1, len(exprs)):
                self.model.add(exprs[0] == exprs[i])
        else:
            # AllEqual except values
            for i in range(1, len(exprs)):
                except_cond = modeler.logical_or([exprs[0] == v for v in excepting])
                self.model.add(modeler.logical_or(exprs[0] == exprs[i], except_cond))
        self._log(2, f"Added AllEqual on {len(scope)} vars")

    def ctr_sum(self, lst: list[Variable] | list[Node], coefficients: None | list[int] | list[Variable], condition: Condition):
        """Sum constraint."""
        exprs = self._get_var_or_node_list(lst)

        if coefficients is None:
            sum_expr = modeler.sum(exprs)
        else:
            terms = []
            for i, e in enumerate(exprs):
                coef = coefficients[i]
                if isinstance(coef, Variable):
                    coef = self.vars[coef.id]
                terms.append(e * coef)
            sum_expr = modeler.sum(terms)

        self._apply_condition(sum_expr, condition)
        self._log(2, f"Added Sum constraint")

    def ctr_count(self, lst: list[Variable] | list[Node], values: list[int] | list[Variable], condition: Condition):
        """Count constraint."""
        exprs = self._get_var_or_node_list(lst)

        if len(values) == 1 and isinstance(values[0], int):
            count_expr = modeler.count(exprs, values[0])
        else:
            # Multiple values or variable values
            count_exprs = []
            for e in exprs:
                in_values = modeler.logical_or([e == v for v in values])
                count_exprs.append(in_values)
            count_expr = modeler.sum(count_exprs)

        self._apply_condition(count_expr, condition)
        self._log(2, f"Added Count constraint")

    def ctr_nvalues(self, lst: list[Variable] | list[Node], excepting: None | list[int], condition: Condition):
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
                vals = [item] if isinstance(item, int) else []

            if isinstance(vals, range):
                item_vals = set(vals)
            else:
                item_vals = set(vals) if vals else set()

            domains.append(item_vals)
            all_values.update(item_vals)

        if excepting:
            all_values.difference_update(excepting)

        # Early exit for trivial cases
        if not all_values or not exprs:
            self._apply_condition(0, condition)
            self._log(2, "Added NValues constraint (trivial)")
            return

        # For each value, check if it appears
        # Only consider values that are in at least one expression's domain
        present = []
        for val in sorted(all_values):
            # Find which expressions can take this value
            relevant_indices = [i for i in range(n) if val in domains[i]]

            if not relevant_indices:
                # No expression can take this value - skip
                continue

            if len(relevant_indices) == 1:
                # Only one expression can take this value - direct check
                present.append(exprs[relevant_indices[0]] == val)
            else:
                # Multiple expressions - use count on relevant subset
                relevant_exprs = [exprs[i] for i in relevant_indices]
                present.append(modeler.count(relevant_exprs, val) >= 1)

        nvalues_expr = modeler.sum(present) if present else 0
        self._apply_condition(nvalues_expr, condition)
        self._log(2, f"Added NValues constraint ({len(present)} values)")

    def ctr_atleast(self, lst: list[Variable], value: int, k: int):
        """AtLeast constraint."""
        exprs = self._get_var_list(lst)
        self.model.add(modeler.count(exprs, value) >= k)
        self._log(2, f"Added AtLeast constraint")

    def ctr_atmost(self, lst: list[Variable], value: int, k: int):
        """AtMost constraint."""
        exprs = self._get_var_list(lst)
        self.model.add(modeler.count(exprs, value) <= k)
        self._log(2, f"Added AtMost constraint")

    def ctr_exactly(self, lst: list[Variable], value: int, k: int | Variable):
        """Exactly constraint."""
        exprs = self._get_var_list(lst)
        if isinstance(k, Variable):
            k = self.vars[k.id]
        self.model.add(modeler.count(exprs, value) == k)
        self._log(2, f"Added Exactly constraint")

    def ctr_element(self, lst: list[Variable] | list[int], index: Variable, condition: Condition):
        """Element constraint."""
        if all(isinstance(v, int) for v in lst):
            # Constant list
            elem = modeler.element(lst, self.vars[index.id])
        else:
            # Variable list
            vars_list = [self.vars[v.id] if isinstance(v, Variable) else v for v in lst]
            elem = modeler.element(vars_list, self.vars[index.id])

        self._apply_condition(elem, condition)
        self._log(2, f"Added Element constraint")

    def ctr_minimum(self, lst: list[Variable] | list[Node], condition: Condition):
        """Minimum constraint."""
        exprs = self._get_var_or_node_list(lst)
        min_expr = modeler.min(exprs)
        self._apply_condition(min_expr, condition)
        self._log(2, f"Added Minimum constraint")

    def ctr_maximum(self, lst: list[Variable] | list[Node], condition: Condition):
        """Maximum constraint."""
        exprs = self._get_var_or_node_list(lst)
        max_expr = modeler.max(exprs)
        self._apply_condition(max_expr, condition)
        self._log(2, f"Added Maximum constraint")

    def ctr_channel(self, lst1: list[Variable], lst2: list[Variable], start_index1: int = 0, start_index2: int = 0):
        """Channel/Inverse constraint."""
        vars1 = self._get_var_list(lst1)
        vars2 = self._get_var_list(lst2)
        self.model.add(modeler.inverse(vars1, vars2))
        self._log(2, f"Added Channel constraint")

    def ctr_nooverlap(self, origins: list[Variable], lengths: list[int] | list[Variable], zero_ignored: bool):
        """NoOverlap constraint (1D)."""
        intervals = []
        for i, origin in enumerate(origins):
            start_var = self.vars[origin.id]
            start_min = origin.dom.smallest_value()
            start_max = origin.dom.greatest_value()

            length = lengths[i]
            if isinstance(length, int):
                interval = self.model.interval_var(start=(start_min, start_max), length=length, name=f"noov_{i}")
            else:
                len_min = length.dom.smallest_value()
                len_max = length.dom.greatest_value()
                interval = self.model.interval_var(
                    start=(start_min, start_max),
                    length=(len_min, len_max),
                    name=f"noov_{i}",
                )
                self.model.add(modeler.length_of(interval) == self.vars[length.id])

            self.model.add(modeler.start_of(interval) == start_var)
            intervals.append(interval)

        self.model.add(modeler.no_overlap(intervals))
        self._log(2, f"Added NoOverlap constraint on {len(intervals)} intervals")

    def ctr_nooverlap_mixed(
        self,
        origins1: list[Variable],
        origins2: list[Variable],
        lengths1: list[int] | list[Variable],
        lengths2: list[int] | list[Variable],
        zero_ignored: bool,
    ):
        """NoOverlap constraint (2D) with mixed length types."""
        origins = list(zip(origins1, origins2))
        lengths = list(zip(lengths1, lengths2))
        return self.ctr_nooverlap_multi(origins, lengths, zero_ignored)

    def ctr_nooverlap_multi(self, origins: list[list], lengths: list[list], zero_ignored: bool):
        """NoOverlap constraint (multi-dimensional)."""
        n = len(origins)
        if n == 0:
            return
        dim = len(origins[0])

        # CPO doesn't have native multi-dimensional no_overlap, decompose to pairwise
        # For each pair of boxes, at least one dimension must not overlap
        for i in range(n):
            for j in range(i + 1, n):
                disjuncts = []
                for d in range(dim):
                    # Get start and length for both boxes in dimension d
                    start_i = self.vars[origins[i][d].id] if isinstance(origins[i][d], Variable) else origins[i][d]
                    start_j = self.vars[origins[j][d].id] if isinstance(origins[j][d], Variable) else origins[j][d]
                    len_i = self.vars[lengths[i][d].id] if isinstance(lengths[i][d], Variable) else lengths[i][d]
                    len_j = self.vars[lengths[j][d].id] if isinstance(lengths[j][d], Variable) else lengths[j][d]

                    # Check if zero_ignored applies (box i or j has zero length)
                    if zero_ignored:
                        if isinstance(lengths[i][d], int) and lengths[i][d] == 0:
                            disjuncts.append(modeler.true())
                            continue
                        if isinstance(lengths[j][d], int) and lengths[j][d] == 0:
                            disjuncts.append(modeler.true())
                            continue
                        if isinstance(lengths[i][d], Variable):
                            disjuncts.append(len_i <= 0)
                        if isinstance(lengths[j][d], Variable):
                            disjuncts.append(len_j <= 0)

                    # Box i ends before box j starts, or vice versa
                    disjuncts.append(start_i + len_i <= start_j)
                    disjuncts.append(start_j + len_j <= start_i)

                self.model.add(modeler.logical_or(disjuncts))

        self._log(2, f"Added NoOverlap{dim}D constraint on {n} boxes")

    def ctr_circuit(self, lst: list[Variable], start_index: int | None = 0):
        """Circuit constraint (Hamiltonian cycle).

        Uses CPO's native sub_circuit constraint which is much more efficient
        than manual O(n²) subtour elimination decomposition.
        """
        vars_list = self._get_var_list(lst)
        if start_index is None:
            start_index = 0
        n = len(vars_list)

        # CPO's sub_circuit expects 0-indexed successors
        # If start_index != 0, we need to adjust the variables
        if start_index == 0:
            # Direct use of sub_circuit
            self.model.add(modeler.sub_circuit(vars_list))

            # sub_circuit allows partial circuits (nodes with next[i]=i are excluded)
            # For full Hamiltonian circuit, prohibit self-loops
            for i in range(n):
                self.model.add(vars_list[i] != i)
        else:
            # Create 0-indexed auxiliary variables: aux[i] = vars[i] - start_index
            aux_vars = [self.model.integer_var(0, n - 1, f"_circuit_aux_{i}") for i in range(n)]
            for i in range(n):
                self.model.add(aux_vars[i] == vars_list[i] - start_index)

            self.model.add(modeler.sub_circuit(aux_vars))

            # Prohibit self-loops for full circuit
            for i in range(n):
                self.model.add(aux_vars[i] != i)

        self._log(2, f"Added Circuit constraint (native sub_circuit)")

    def ctr_regular(
        self,
        scope: list[Variable],
        transitions: list,
        start_state: str,
        final_states: list[str],
    ):
        """Regular constraint using finite automaton (decomposed)."""
        vars_list = self._get_var_list(scope)
        n = len(vars_list)

        # Map string states to integers
        all_states = set()
        all_states.add(start_state)
        all_states.update(final_states)
        for src, symbol, dst in transitions:
            all_states.add(src)
            all_states.add(dst)

        state_to_int = {state: i for i, state in enumerate(sorted(all_states))}
        num_states = len(all_states)

        int_start = state_to_int[start_state]
        int_finals = [state_to_int[s] for s in final_states]

        # Create state variables q[0..n], q[i] = state after reading vars[0..i-1]
        state_vars = [
            self.model.integer_var(0, num_states - 1, f"_reg_state_{i}")
            for i in range(n + 1)
        ]

        # Initial state
        self.model.add(state_vars[0] == int_start)

        # Final state must be accepting
        if len(int_finals) == 1:
            self.model.add(state_vars[n] == int_finals[0])
        else:
            self.model.add(modeler.allowed_assignments(state_vars[n], int_finals))

        # Build transition table: list of (state, symbol, next_state)
        int_transitions = [
            (state_to_int[src], symbol, state_to_int[dst])
            for src, symbol, dst in transitions
        ]

        # For each position, constrain transitions using table constraint
        for i in range(n):
            self.model.add(
                modeler.allowed_assignments(
                    (state_vars[i], vars_list[i], state_vars[i + 1]),
                    int_transitions
                )
            )

        self._log(2, f"Added Regular constraint with {len(transitions)} transitions (decomposed)")

    def ctr_ordered(self, lst: list[Variable], operator: str, lengths: None | list[int] | list[Variable]):
        """Ordered constraint (increasing/decreasing)."""
        vars_list = self._get_var_list(lst)
        op = operator.upper() if isinstance(operator, str) else str(operator).upper()

        if lengths is None:
            if op in ("INCREASING", "LE"):
                for i in range(len(vars_list) - 1):
                    self.model.add(vars_list[i] <= vars_list[i + 1])
            elif op in ("STRICTLY_INCREASING", "LT"):
                for i in range(len(vars_list) - 1):
                    self.model.add(vars_list[i] < vars_list[i + 1])
            elif op in ("DECREASING", "GE"):
                for i in range(len(vars_list) - 1):
                    self.model.add(vars_list[i] >= vars_list[i + 1])
            elif op in ("STRICTLY_DECREASING", "GT"):
                for i in range(len(vars_list) - 1):
                    self.model.add(vars_list[i] > vars_list[i + 1])
            else:
                raise NotImplementedError(f"Unknown ordering operator: {operator}")
        else:
            # With lengths
            for i in range(len(vars_list) - 1):
                length = lengths[i] if isinstance(lengths[i], int) else self.vars[lengths[i].id]
                if op in ("INCREASING", "LE"):
                    self.model.add(vars_list[i] + length <= vars_list[i + 1])
                elif op in ("DECREASING", "GE"):
                    self.model.add(vars_list[i] >= vars_list[i + 1] + length)

        self._log(2, f"Added Ordered constraint ({operator})")

    def ctr_lex(self, lists: list[list[Variable]], operator: TypeOrderedOperator):
        """Lexicographic constraint on lists."""
        for i in range(len(lists) - 1):
            left = self._get_var_list(lists[i])
            right = self._get_var_list(lists[i + 1])
            if operator == TypeOrderedOperator.STRICTLY_INCREASING:
                self.model.add(modeler.strict_lexicographic(left, right))
            elif operator == TypeOrderedOperator.INCREASING:
                self.model.add(modeler.lexicographic(left, right))
            elif operator == TypeOrderedOperator.STRICTLY_DECREASING:
                self.model.add(modeler.strict_lexicographic(right, left))
            elif operator == TypeOrderedOperator.DECREASING:
                self.model.add(modeler.lexicographic(right, left))
            else:
                raise NotImplementedError(f"Unknown lex operator: {operator}")
        self._log(2, f"Added Lex constraint on {len(lists)} lists")

    def ctr_lex_matrix(self, matrix: list[list[Variable]], operator: TypeOrderedOperator):
        """Lexicographic constraint on matrix rows."""
        self.ctr_lex(matrix, operator)

    def ctr_cumulative(self, origins: list[Variable], lengths: list[int] | list[Variable],
                       heights: list[int] | list[Variable], condition: Condition):
        """Cumulative constraint."""
        from pycsp3.classes.auxiliary.conditions import ConditionValue, ConditionVariable

        intervals = []
        pulses = []
        max_height_sum = 0

        for i, origin in enumerate(origins):
            start_var = self.vars[origin.id]
            start_min = origin.dom.smallest_value()
            start_max = origin.dom.greatest_value()

            length = lengths[i]
            height = heights[i]

            def _build_interval(name_suffix: str, optional: bool):
                if isinstance(length, int):
                    interval = self.model.interval_var(
                        start=(start_min, start_max),
                        length=length,
                        optional=optional,
                        name=f"cum_{i}_{name_suffix}",
                    )
                else:
                    len_min = length.dom.smallest_value()
                    len_max = length.dom.greatest_value()
                    interval = self.model.interval_var(
                        start=(start_min, start_max),
                        length=(len_min, len_max),
                        optional=optional,
                        name=f"cum_{i}_{name_suffix}",
                    )
                    length_expr = modeler.length_of(interval) == self.vars[length.id]
                    if optional:
                        self.model.add(modeler.if_then(modeler.presence_of(interval), length_expr))
                    else:
                        self.model.add(length_expr)

                start_expr = modeler.start_of(interval) == start_var
                if optional:
                    self.model.add(modeler.if_then(modeler.presence_of(interval), start_expr))
                else:
                    self.model.add(start_expr)
                return interval

            if isinstance(height, Variable):
                is_infinite = False
                if hasattr(height.dom, "is_infinite"):
                    is_infinite = height.dom.is_infinite()
                elif hasattr(height.dom, "original_values"):
                    try:
                        is_infinite = (
                            len(height.dom.original_values) == 1
                            and height.dom.original_values[0] == math.inf
                        )
                    except Exception:
                        is_infinite = False
                if is_infinite:
                    raise NotImplementedError(
                        "Cumulative with infinite height domains is not supported in CPO backend"
                    )
                height_values = height.dom.all_values()
                if isinstance(height_values, range):
                    height_values = list(height_values)
                height_values = sorted(height_values)
                if not height_values:
                    raise ValueError("Cumulative height domain is empty")
                if height.dom.smallest_value() < 0:
                    raise NotImplementedError(
                        "Cumulative with negative heights is not supported in CPO backend"
                    )

                intervals_per_height = []
                presences = []
                weighted_presence = []
                for hv in height_values:
                    interval = _build_interval(f"h{hv}", optional=True)
                    intervals_per_height.append(interval)
                    presence = modeler.presence_of(interval)
                    presences.append(presence)
                    weighted_presence.append(hv * presence)
                    pulses.append(modeler.pulse(interval, hv))

                self.model.add(modeler.sum(presences) == 1)
                self.model.add(self.vars[height.id] == modeler.sum(weighted_presence))
                max_height_sum += height.dom.greatest_value()
                intervals.extend(intervals_per_height)
            else:
                interval = _build_interval("h", optional=False)
                intervals.append(interval)
                pulses.append(modeler.pulse(interval, height))
                max_height_sum += height

        cumul = modeler.sum(pulses) if pulses else 0

        if isinstance(condition, ConditionValue):
            capacity = condition.value
        elif isinstance(condition, ConditionVariable):
            capacity = self.vars[condition.variable.id]
        else:
            raise NotImplementedError(
                f"Cumulative condition type {type(condition)} not supported"
            )

        op = condition.operator
        if op == TypeConditionOperator.LE:
            self.model.add(modeler.cumul_range(cumul, 0, capacity))
        elif op == TypeConditionOperator.LT:
            if isinstance(capacity, int):
                self.model.add(modeler.cumul_range(cumul, 0, capacity - 1))
            else:
                raise NotImplementedError("Cumulative LT with variable capacity not supported")
        elif op == TypeConditionOperator.GE:
            self.model.add(modeler.cumul_range(cumul, capacity, max_height_sum))
        elif op == TypeConditionOperator.GT:
            if isinstance(capacity, int):
                self.model.add(modeler.cumul_range(cumul, capacity + 1, max_height_sum))
            else:
                raise NotImplementedError("Cumulative GT with variable capacity not supported")
        elif op == TypeConditionOperator.EQ:
            self.model.add(modeler.cumul_range(cumul, capacity, capacity))
        else:
            raise NotImplementedError(f"Cumulative condition operator {op} not supported")

        self._log(2, f"Added Cumulative constraint on {len(intervals)} tasks")

    # ========== Objective callbacks ==========

    def obj_minimize(self, term: Variable | Node):
        """Minimize a simple objective (single term)."""
        self._mark_objective()
        if isinstance(term, Variable):
            expr = self.vars[term.id]
        else:
            expr = self.translate_node(term)
        self._objective_expr = expr
        self._minimize = True
        self._log(2, "Set minimize objective")

    def obj_maximize(self, term: Variable | Node):
        """Maximize a simple objective (single term)."""
        self._mark_objective()
        if isinstance(term, Variable):
            expr = self.vars[term.id]
        else:
            expr = self.translate_node(term)
        self._objective_expr = expr
        self._minimize = False
        self._log(2, "Set maximize objective")

    def obj_minimize_special(self, obj_type: TypeObj, terms: list[Variable] | list[Node], coefficients: None | list[int]):
        """Minimize special objective (sum, min, max, etc.)."""
        self._set_objective_special(obj_type, terms, coefficients, minimize=True)

    def obj_maximize_special(self, obj_type: TypeObj, terms: list[Variable] | list[Node], coefficients: None | list[int]):
        """Maximize special objective (sum, min, max, etc.)."""
        self._set_objective_special(obj_type, terms, coefficients, minimize=False)

    def _set_objective_special(self, obj_type: TypeObj, terms: list[Variable] | list[Node],
                               coefficients: None | list[int], minimize: bool):
        """Set optimization objective for special types."""
        self._mark_objective()
        exprs = self._get_var_or_node_list(terms)

        if obj_type == TypeObj.SUM:
            if coefficients is None:
                obj_expr = modeler.sum(exprs)
            else:
                obj_expr = modeler.sum(c * e for c, e in zip(coefficients, exprs))
        elif obj_type == TypeObj.MINIMUM:
            obj_expr = modeler.min(exprs)
        elif obj_type == TypeObj.MAXIMUM:
            obj_expr = modeler.max(exprs)
        elif obj_type == TypeObj.NVALUES:
            # Count distinct values
            all_vals = set()
            for e in exprs:
                if hasattr(e, 'domain'):
                    all_vals.update(range(e.domain.min(), e.domain.max() + 1))
            val_used = [modeler.logical_or([e == v for e in exprs]) for v in all_vals]
            obj_expr = modeler.sum(val_used)
        elif obj_type == TypeObj.LEX:
            # Lexicographic - just use first expression
            obj_expr = exprs[0]
        else:
            raise NotImplementedError(f"Objective type {obj_type} not implemented")

        self._objective_expr = obj_expr
        self._minimize = minimize
        self._log(2, f"Set {'minimize' if minimize else 'maximize'} objective ({obj_type})")

    # ========== Solving ==========

    def apply_hints(self):
        """Apply warm start hints using CPO starting point."""
        if not self.hints:
            return
        from docplex.cp.solution import CpoModelSolution
        stp = CpoModelSolution()
        applied = 0
        for var_id, value in self.hints.items():
            if var_id in self.vars:
                stp.add_var_solution(modeler.var_solution(self.vars[var_id], value))
                applied += 1
            else:
                self._log(2, f"Hint for unknown variable '{var_id}' ignored")
        if applied > 0:
            self.model.set_starting_point(stp)
            self._log(1, f"Applied {applied} warm start hints")

    def solve(self) -> TypeStatus:
        """Solve the model and return status."""
        self._log(1, "Starting CPO solver...")
        self._objective_value = None
        use_search_next = False

        # Set objective if any
        if self._objective_expr is not None:
            if self._minimize:
                self.model.minimize(self._objective_expr)
            else:
                self.model.maximize(self._objective_expr)

        # Attach a solver listener for objective progression if requested.
        if self._objective_expr is not None and self.get_competition_progress_printer() is not None and CpoSolverListener is not None:
            progress_printer = self.get_competition_progress_printer()
            use_search_next = True

            class _ObjectiveProgressListener(CpoSolverListener):
                def result_found(self, solver, sres):  # noqa: N802 - docplex naming
                    try:
                        if sres is None or not sres.is_solution():
                            return
                        obj_val = sres.get_objective_value()
                        if obj_val is not None:
                            progress_printer.report(obj_val)
                    except Exception:
                        return

            try:
                self.model.add_solver_listener(_ObjectiveProgressListener())
            except Exception:
                pass

        # Configure solver parameters
        params = {}
        if self.time_limit is not None:
            params['TimeLimit'] = self.time_limit

        if self.sols is not None:
            if self.sols != "all" and isinstance(self.sols, int):
                params['SolutionLimit'] = self.sols
        elif self._objective_expr is None:
            # Ensure at least one solution is searched (avoid SolutionLimit=0 defaults)
            params.setdefault('SolutionLimit', 1)
        if self.verbose <= 0:
            params['LogVerbosity'] = "Quiet"
            params['log_output'] = None
        if use_search_next:
            # Ensure listeners see intermediate solutions for objective progression.
            params["solve_with_search_next"] = True

        try:
            if self.sols is not None and self._objective_expr is None:
                limit = None if self.sols == "all" else int(self.sols)
                return self._solve_all_solutions(params, limit)

            # Solve
            self._all_solutions = []
            self.solution = self.model.solve(**params)

            if self.solution is None:
                self._log(1, "Solver returned None (UNKNOWN)")
                return TypeStatus.UNKNOWN

            solve_status = self.solution.get_solve_status()
            self._log(1, f"Solver finished with status: {solve_status}")

            # Map CPO status to TypeStatus
            if solve_status == "Optimal":
                self._objective_value = self.solution.get_objective_value()
                if self._objective_expr is None:
                    self._status = TypeStatus.SAT
                    return TypeStatus.SAT
                self._status = TypeStatus.OPTIMUM
                return TypeStatus.OPTIMUM
            elif solve_status == "Feasible":
                self._objective_value = self.solution.get_objective_value()
                self._status = TypeStatus.SAT
                return TypeStatus.SAT
            elif solve_status == "Infeasible":
                self._status = TypeStatus.UNSAT
                return TypeStatus.UNSAT
            else:
                self._status = TypeStatus.UNKNOWN
                return TypeStatus.UNKNOWN

        except Exception as e:
            self._log(1, f"Solver error: {e}")
            raise

    def _solve_all_solutions(self, params: dict, limit: int | None) -> TypeStatus:
        """Enumerate all/multiple solutions for satisfaction problems."""
        self._all_solutions = []

        search = self.model.start_search(**params)
        last_solution = None
        try:
            for sol in search:
                last_solution = sol
                sol_dict = {}
                for var_id, cpo_var in self.vars.items():
                    sol_val = sol.get_value(cpo_var)
                    if sol_val is not None:
                        if hasattr(self, '_symbolic_maps') and var_id in self._symbolic_maps:
                            sol_dict[var_id] = self._symbolic_maps[var_id][sol_val]
                        else:
                            sol_dict[var_id] = int(sol_val)
                if sol_dict:
                    self._all_solutions.append(sol_dict)
                if limit is not None and len(self._all_solutions) >= limit:
                    break
        finally:
            search.end()

        if self._all_solutions:
            self.solution = last_solution
            self._status = TypeStatus.SAT
            self._log(1, f"Solver finished: SAT ({len(self._all_solutions)} solutions)")
            return TypeStatus.SAT

        self._status = TypeStatus.UNSAT
        self._log(1, "Solver finished: UNSAT")
        return TypeStatus.UNSAT

    def get_solution(self) -> dict[str, int] | None:
        """Get solution values as dict."""
        if self.solution is None:
            return None

        result = {}
        for var_id, cpo_var in self.vars.items():
            try:
                val = self.solution.get_value(cpo_var)
                if val is not None:
                    # Handle symbolic variables
                    if hasattr(self, '_symbolic_maps') and var_id in self._symbolic_maps:
                        result[var_id] = self._symbolic_maps[var_id][val]
                    else:
                        result[var_id] = int(val)
            except Exception:
                pass

        return result

    def get_all_solutions(self) -> list[dict[str, int]]:
        """Return all solutions found."""
        return self._all_solutions
