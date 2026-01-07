"""
CPO (IBM DOcplex CP Optimizer) backend for pycsp3-solvers-extra.

This module provides the CPOCallbacks class that implements the XCSP3 callback
interface using IBM DOcplex CP Optimizer.

Note: Requires IBM CPLEX Optimization Studio to be installed.
"""

from __future__ import annotations

from typing import Any

from pycsp3.classes.auxiliary.enums import TypeStatus, TypeConditionOperator, TypeObj
from pycsp3.classes.auxiliary.conditions import Condition
from pycsp3.classes.nodes import Node

from pycsp3_solvers_extra.backends.base import BaseCallbacks

from pycsp3.classes.main.variables import Variable

# Try to import docplex
try:
    from docplex.cp.model import CpoModel
    from docplex.cp.solution import CpoSolveResult
    import docplex.cp.modeler as modeler
    from docplex.cp.config import context as cpo_context
    CPO_AVAILABLE = True
except ImportError:
    CPO_AVAILABLE = False
    cpo_context = None


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
    ):
        super().__init__(time_limit, sols, verbose, options)

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

    # ========== Variable callbacks ==========

    def var_integer_range(self, x: Variable, min_val: int, max_val: int):
        """Create integer variable with range domain."""
        var = self.model.integer_var(min_val, max_val, x.id)
        self.vars[x.id] = var
        self._log(2, f"Created var {x.id} in [{min_val}, {max_val}]")

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

    def translate_node(self, node: Node) -> Any:
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
            return modeler.logical_and([self.translate_node(c) for c in node.cnt])
        elif node.type == TypeNode.OR:
            return modeler.logical_or([self.translate_node(c) for c in node.cnt])
        elif node.type == TypeNode.NOT:
            return modeler.logical_not(self.translate_node(node.cnt[0]))
        elif node.type == TypeNode.IMP:
            a, b = self.translate_node(node.cnt[0]), self.translate_node(node.cnt[1])
            return modeler.if_then(a, b)
        elif node.type == TypeNode.IFF:
            a, b = self.translate_node(node.cnt[0]), self.translate_node(node.cnt[1])
            return a == b
        elif node.type == TypeNode.IF:
            cond = self.translate_node(node.cnt[0])
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

    # ========== Constraint callbacks ==========

    def ctr_intension(self, scope: list[Variable], tree: Node):
        """Add intension constraint."""
        expr = self.translate_node(tree)
        self.model.add(expr)
        self._log(2, f"Added intension constraint on {len(scope)} vars")

    def ctr_extension_unary(self, x: Variable, values: list[int], positive: bool, flags: set[str]):
        """Unary table constraint."""
        var = self.vars[x.id]
        if positive:
            self.model.add(modeler.allowed_assignments(var, values))
        else:
            self.model.add(modeler.forbidden_assignments(var, values))
        self._log(2, f"Added {'positive' if positive else 'negative'} unary extension")

    def ctr_extension(self, scope: list[Variable], tuples: list, positive: bool, flags: set[str]):
        """Table constraint."""
        vars_list = self._get_var_list(scope)
        if positive:
            self.model.add(modeler.allowed_assignments(vars_list, tuples))
        else:
            self.model.add(modeler.forbidden_assignments(vars_list, tuples))
        self._log(2, f"Added {'positive' if positive else 'negative'} extension on {len(scope)} vars")

    def ctr_all_different(self, scope: list[Variable] | list[Node], excepting: None | list[int]):
        """AllDifferent constraint."""
        exprs = self._get_var_or_node_list(scope)
        if excepting is None or len(excepting) == 0:
            self.model.add(modeler.all_diff(exprs))
        else:
            # AllDifferent except values - pairwise constraints
            for i in range(len(exprs)):
                for j in range(i + 1, len(exprs)):
                    xi_except = modeler.logical_or([exprs[i] == v for v in excepting])
                    xj_except = modeler.logical_or([exprs[j] == v for v in excepting])
                    self.model.add(modeler.logical_or(exprs[i] != exprs[j], xi_except, xj_except))
        self._log(2, f"Added AllDifferent on {len(scope)} vars")

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

    def ctr_circuit(self, lst: list[Variable], start_index: int = 0):
        """Circuit constraint (Hamiltonian cycle)."""
        # CPO doesn't have direct circuit, use sub_circuit
        vars_list = self._get_var_list(lst)
        # Create successor representation for circuit
        # In pycsp3, circuit means: lst[i] = j means edge from i to j
        # We need to enforce that following successors visits all nodes exactly once
        n = len(vars_list)

        # Use subtour elimination with position variables
        pos = [self.model.integer_var(0, n - 1, f"_pos_{i}") for i in range(n)]

        # All different positions
        self.model.add(modeler.all_diff(pos))

        # Position constraints for subtour elimination
        for i in range(n):
            for j in range(n):
                if i != j:
                    # If lst[i] == j, then pos[j] = pos[i] + 1 (mod n)
                    self.model.add(
                        modeler.if_then(
                            vars_list[i] == j + start_index,
                            (pos[j] == pos[i] + 1) | ((pos[i] == n - 1) & (pos[j] == 0))
                        )
                    )

        self._log(2, f"Added Circuit constraint")

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

    # ========== Objective callbacks ==========

    def obj_minimize(self, term: Variable | Node):
        """Minimize a simple objective (single term)."""
        if isinstance(term, Variable):
            expr = self.vars[term.id]
        else:
            expr = self.translate_node(term)
        self._objective_expr = expr
        self._minimize = True
        self._log(2, "Set minimize objective")

    def obj_maximize(self, term: Variable | Node):
        """Maximize a simple objective (single term)."""
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

    def solve(self) -> TypeStatus:
        """Solve the model and return status."""
        self._log(1, "Starting CPO solver...")

        # Set objective if any
        if self._objective_expr is not None:
            if self._minimize:
                self.model.minimize(self._objective_expr)
            else:
                self.model.maximize(self._objective_expr)

        # Configure solver parameters
        params = {}
        if self.time_limit is not None:
            params['TimeLimit'] = self.time_limit

        if self.sols is not None:
            if self.sols == "all":
                params['SolutionLimit'] = 0  # All solutions
            elif isinstance(self.sols, int):
                params['SolutionLimit'] = self.sols

        try:
            # Solve
            self.solution = self.model.solve(**params)

            if self.solution is None:
                self._log(1, "Solver returned None (UNKNOWN)")
                return TypeStatus.UNKNOWN

            solve_status = self.solution.get_solve_status()
            self._log(1, f"Solver finished with status: {solve_status}")

            # Map CPO status to TypeStatus
            if solve_status == "Optimal":
                return TypeStatus.OPTIMUM
            elif solve_status == "Feasible":
                return TypeStatus.SAT
            elif solve_status == "Infeasible":
                return TypeStatus.UNSAT
            else:
                return TypeStatus.UNKNOWN

        except Exception as e:
            self._log(1, f"Solver error: {e}")
            raise

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
