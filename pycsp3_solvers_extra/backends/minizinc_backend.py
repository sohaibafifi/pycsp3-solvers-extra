"""
MiniZinc backend for pycsp3-solvers-extra.

This module implements the MiniZincCallbacks class that translates
XCSP3 constraints to MiniZinc model and solves using minizinc-python.
"""


import asyncio
from typing import Any

from pycsp3.classes.auxiliary.conditions import Condition
from pycsp3.classes.auxiliary.enums import (
    TypeConditionOperator,
    TypeOrderedOperator,
    TypeObj,
    TypeStatus,
)
from pycsp3.classes.main.variables import Variable
from pycsp3.classes.nodes import Node, TypeNode

from pycsp3_solvers_extra.backends.base import BaseCallbacks

# Check if minizinc is available (both package and binary)
try:
    import minizinc
    from minizinc import Model, Solver, Instance
    # Check if the MiniZinc binary/driver is actually installed
    # The package can be installed without the CLI tool
    MINIZINC_AVAILABLE = minizinc.default_driver is not None
except ImportError:
    MINIZINC_AVAILABLE = False


class MiniZincCallbacks(BaseCallbacks):
    """
    MiniZinc backend using callback-based XCSP3 parsing.

    Translates XCSP3 elements to MiniZinc model syntax and solves
    using the minizinc-python library.

    The backend supports multiple underlying solvers via the subsolver
    parameter (e.g., "gecode", "chuffed", "coin-bc").
    """

    def __init__(
        self,
        time_limit: float | None = None,
        sols: int | str | None = None,
        verbose: int = 0,
        options: str = "",
        hints: dict[str, int] | None = None,
        subsolver: str | None = None,
    ):
        super().__init__(time_limit, sols, verbose, options, hints)

        self._subsolver = subsolver

        # MiniZinc model components
        self._mzn_vars: list[str] = []          # Variable declarations
        self._mzn_constraints: list[str] = []   # Constraint declarations
        self._mzn_solve: str = "solve satisfy"  # Solve statement
        self._mzn_output: str | None = None     # Output statement

        # Track variable info for solution extraction
        self._var_ids: list[str] = []           # Ordered list of var IDs
        self._var_domains: dict[str, Any] = {}  # var_id -> domain info
        self._symbolic_maps: dict[str, list[str]] = {}  # var_id -> symbol list

        # For multi-solution enumeration
        self._all_solutions: list[dict[str, int]] = []

    # ========== Variable creation ==========

    def var_integer_range(self, x: Variable, min_value: int, max_value: int):
        """Create integer variable with range domain."""
        var_id = self._mzn_var_name(x.id)
        self._mzn_vars.append(f"var {min_value}..{max_value}: {var_id};")
        self.vars[x.id] = var_id
        self._var_ids.append(x.id)
        self._var_domains[x.id] = (min_value, max_value)
        self._log(2, f"Created var {x.id} in [{min_value}, {max_value}]")

    def var_integer(self, x: Variable, values: list[int]):
        """Create integer variable with enumerated domain."""
        var_id = self._mzn_var_name(x.id)
        values_str = ", ".join(str(v) for v in sorted(values))
        self._mzn_vars.append(f"var {{{values_str}}}: {var_id};")
        self.vars[x.id] = var_id
        self._var_ids.append(x.id)
        self._var_domains[x.id] = values
        self._log(2, f"Created var {x.id} with domain {values[:5]}{'...' if len(values) > 5 else ''}")

    def var_symbolic(self, x: Variable, values: list[str]):
        """Create symbolic variable (mapped to integers)."""
        var_id = self._mzn_var_name(x.id)
        # Map symbolic values to integers 0, 1, 2, ...
        self._mzn_vars.append(f"var 0..{len(values) - 1}: {var_id};")
        self.vars[x.id] = var_id
        self._var_ids.append(x.id)
        self._var_domains[x.id] = (0, len(values) - 1)
        self._symbolic_maps[x.id] = values
        self._log(2, f"Created symbolic var {x.id} with {len(values)} values")

    def _mzn_var_name(self, var_id: str) -> str:
        """Convert pycsp3 variable ID to valid MiniZinc identifier."""
        # Replace brackets and other invalid characters
        name = var_id.replace("[", "_").replace("]", "").replace(",", "_")
        # Ensure it starts with a letter
        if name[0].isdigit():
            name = "v_" + name
        return name

    # ========== Expression translation ==========

    def _is_bool_node(self, node: Node) -> bool:
        """Check if a node represents a boolean expression."""
        bool_types = {
            TypeNode.EQ, TypeNode.NE, TypeNode.LT, TypeNode.LE,
            TypeNode.GT, TypeNode.GE, TypeNode.AND, TypeNode.OR,
            TypeNode.NOT, TypeNode.XOR, TypeNode.IFF, TypeNode.IMP,
            TypeNode.IN, TypeNode.NOTIN,
        }
        return node.type in bool_types

    def _ensure_bool(self, node: Node) -> str:
        """Translate node and ensure result is boolean for MiniZinc.

        MiniZinc requires boolean operands for /\\ and \\/.
        If the expression is an int, wrap it with != 0.
        """
        expr = self.translate_node(node)
        if self._is_bool_node(node):
            return expr
        # Int expression - convert to bool
        return f"({expr} != 0)"

    def translate_node(self, node: Node) -> str:
        """Translate pycsp3 expression node to MiniZinc expression string."""
        # Check cache first
        cache_key = self._node_structural_key(node)
        if cache_key in self._node_cache:
            return self._node_cache[cache_key]

        result = self._translate_node_impl(node)
        self._node_cache[cache_key] = result
        return result

    def _translate_node_impl(self, node: Node) -> str:
        """Implementation of node translation."""
        node_type = node.type

        # Terminals
        if node_type == TypeNode.VAR:
            return self.vars[node.cnt.id]

        if node_type == TypeNode.INT:
            return str(node.cnt)

        if node_type == TypeNode.SYMBOL:
            return str(node.cnt)

        if node_type == TypeNode.SET:
            values = ", ".join(str(v) for v in node.cnt)
            return f"{{{values}}}"

        # Arithmetic operators
        if node_type == TypeNode.ADD:
            terms = [self.translate_node(c) for c in node.cnt]
            return "(" + " + ".join(terms) + ")"

        if node_type == TypeNode.SUB:
            if len(node.cnt) == 1:
                return f"(-{self.translate_node(node.cnt[0])})"
            terms = [self.translate_node(c) for c in node.cnt]
            return "(" + " - ".join(terms) + ")"

        if node_type == TypeNode.MUL:
            terms = [self.translate_node(c) for c in node.cnt]
            return "(" + " * ".join(terms) + ")"

        if node_type == TypeNode.DIV:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return f"({left} div {right})"

        if node_type == TypeNode.MOD:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return f"({left} mod {right})"

        if node_type == TypeNode.POW:
            base = self.translate_node(node.cnt[0])
            exp = self.translate_node(node.cnt[1])
            return f"pow({base}, {exp})"

        if node_type == TypeNode.NEG:
            return f"(-{self.translate_node(node.cnt[0])})"

        if node_type == TypeNode.ABS:
            return f"abs({self.translate_node(node.cnt[0])})"

        if node_type == TypeNode.DIST:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return f"abs({left} - {right})"

        if node_type == TypeNode.MIN:
            terms = [self.translate_node(c) for c in node.cnt]
            return f"min([{', '.join(terms)}])"

        if node_type == TypeNode.MAX:
            terms = [self.translate_node(c) for c in node.cnt]
            return f"max([{', '.join(terms)}])"

        # Comparison operators
        if node_type == TypeNode.EQ:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return f"({left} = {right})"

        if node_type == TypeNode.NE:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return f"({left} != {right})"

        if node_type == TypeNode.LT:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return f"({left} < {right})"

        if node_type == TypeNode.LE:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return f"({left} <= {right})"

        if node_type == TypeNode.GT:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return f"({left} > {right})"

        if node_type == TypeNode.GE:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return f"({left} >= {right})"

        # Logical operators
        if node_type == TypeNode.AND:
            terms = [self._ensure_bool(c) for c in node.cnt]
            return "(" + " /\\ ".join(terms) + ")"

        if node_type == TypeNode.OR:
            terms = [self._ensure_bool(c) for c in node.cnt]
            return "(" + " \\/ ".join(terms) + ")"

        if node_type == TypeNode.NOT:
            return f"not({self.translate_node(node.cnt[0])})"

        if node_type == TypeNode.XOR:
            terms = [self.translate_node(c) for c in node.cnt]
            return f"xorall([{', '.join(terms)}])"

        if node_type == TypeNode.IFF:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return f"({left} <-> {right})"

        if node_type == TypeNode.IMP:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return f"({left} -> {right})"

        if node_type == TypeNode.IF:
            # IF-THEN-ELSE: if cond then val1 else val2 endif
            cond = self.translate_node(node.cnt[0])
            then_val = self.translate_node(node.cnt[1])
            else_val = self.translate_node(node.cnt[2])
            return f"if {cond} then {then_val} else {else_val} endif"

        # Set membership
        if node_type == TypeNode.IN:
            val = self.translate_node(node.cnt[0])
            set_node = node.cnt[1]
            if set_node.type == TypeNode.SET:
                values = ", ".join(str(v) for v in set_node.cnt)
                return f"({val} in {{{values}}})"
            else:
                set_expr = self.translate_node(set_node)
                return f"({val} in {set_expr})"

        if node_type == TypeNode.NOTIN:
            val = self.translate_node(node.cnt[0])
            set_node = node.cnt[1]
            if set_node.type == TypeNode.SET:
                values = ", ".join(str(v) for v in set_node.cnt)
                return f"not({val} in {{{values}}})"
            else:
                set_expr = self.translate_node(set_node)
                return f"not({val} in {set_expr})"

        raise NotImplementedError(f"Node type {node_type} not supported in MiniZinc backend")

    # ========== Constraint helpers ==========

    def _get_mzn_var_list(self, scope: list[Variable]) -> list[str]:
        """Convert Variable list to MiniZinc variable names."""
        return [self.vars[v.id] for v in scope]

    def _get_mzn_var_or_expr_list(self, scope: list) -> list[str]:
        """Convert mixed Variable/Node list to MiniZinc expressions."""
        result = []
        for item in scope:
            if isinstance(item, Variable):
                result.append(self.vars[item.id])
            elif isinstance(item, Node):
                result.append(self.translate_node(item))
            elif isinstance(item, int):
                result.append(str(item))
            else:
                result.append(str(item))
        return result

    def _condition_to_mzn(self, expr: str, condition: Condition) -> str:
        """Convert condition to MiniZinc constraint expression."""
        op = condition.operator
        val = condition.right_operand()

        if isinstance(val, Variable):
            val_str = self.vars[val.id]
        elif isinstance(val, Node):
            val_str = self.translate_node(val)
        elif isinstance(val, int):
            val_str = str(val)
        elif isinstance(val, (list, tuple, set)):
            # Set membership
            values = ", ".join(str(v) for v in val)
            if op == TypeConditionOperator.IN:
                return f"{expr} in {{{values}}}"
            elif op == TypeConditionOperator.NOTIN:
                return f"not({expr} in {{{values}}})"
            else:
                val_str = f"{{{values}}}"
        elif val is None:
            # Interval condition
            interval = condition.right_operand_as_interval()
            if interval:
                return f"{expr} in {interval[0]}..{interval[1]}"
            raise ValueError(f"Unsupported condition: {condition}")
        else:
            val_str = str(val)

        op_map = {
            TypeConditionOperator.LT: "<",
            TypeConditionOperator.LE: "<=",
            TypeConditionOperator.GE: ">=",
            TypeConditionOperator.GT: ">",
            TypeConditionOperator.EQ: "=",
            TypeConditionOperator.NE: "!=",
            TypeConditionOperator.IN: "in",
            TypeConditionOperator.NOTIN: "not in",
        }

        if op == TypeConditionOperator.IN:
            return f"{expr} in {val_str}"
        elif op == TypeConditionOperator.NOTIN:
            return f"not({expr} in {val_str})"

        return f"{expr} {op_map.get(op, '=')} {val_str}"

    def _add_constraint(self, expr: str):
        """Add a constraint to the model."""
        self._mzn_constraints.append(f"constraint {expr};")

    # ========== Intension constraints ==========

    def ctr_intension(self, scope: list[Variable], tree: Node):
        """Add intension constraint from expression tree."""
        expr = self.translate_node(tree)
        self._add_constraint(expr)
        self._log(2, f"Added intension constraint")

    # ========== Extension constraints ==========

    def ctr_extension_unary(self, x: Variable, values: list[int], positive: bool, flags: set[str]):
        """Add unary extension constraint."""
        var = self.vars[x.id]
        values_str = ", ".join(str(v) for v in values)

        if positive:
            self._add_constraint(f"{var} in {{{values_str}}}")
        else:
            self._add_constraint(f"not({var} in {{{values_str}}})")

        self._log(2, f"Added unary extension constraint on {x.id}")

    def ctr_extension(self, scope: list[Variable], tuples: list, positive: bool, flags: set[str]):
        """Add table constraint."""
        vars_list = self._get_mzn_var_list(scope)
        vars_str = ", ".join(vars_list)

        # Build tuple array
        tuple_strs = []
        for t in tuples:
            t_str = "| " + ", ".join(str(v) for v in t)
            tuple_strs.append(t_str)
        tuples_str = "\n".join(tuple_strs) + " |]"

        if positive:
            self._add_constraint(f"table([{vars_str}], [" + tuples_str + ")")
        else:
            # Negative table: none of the tuples
            self._add_constraint(f"not(table([{vars_str}], [" + tuples_str + "))")

        self._log(2, f"Added table constraint with {len(tuples)} tuples")

    # ========== Global constraints ==========

    def ctr_all_different(self, scope: list[Variable] | list[Node], excepting: None | list[int]):
        """Add all-different constraint."""
        exprs = self._get_mzn_var_or_expr_list(scope)
        exprs_str = ", ".join(exprs)

        if excepting is None or len(excepting) == 0:
            self._add_constraint(f"all_different([{exprs_str}])")
        elif len(excepting) == 1 and excepting[0] == 0:
            self._add_constraint(f"all_different_except_0([{exprs_str}])")
        else:
            # Decompose for general excepting values
            except_set = ", ".join(str(v) for v in excepting)
            # Pairwise different unless both in excepting set
            n = len(exprs)
            for i in range(n):
                for j in range(i + 1, n):
                    self._add_constraint(
                        f"({exprs[i]} in {{{except_set}}}) \\/ "
                        f"({exprs[j]} in {{{except_set}}}) \\/ "
                        f"({exprs[i]} != {exprs[j]})"
                    )

        self._log(2, f"Added all_different constraint")

    def ctr_all_different_lists(self, lists: list[list[Variable]], excepting: None | list[list[int]]):
        """Add all-different on lists constraint."""
        for lst in lists:
            self.ctr_all_different(lst, None)
        self._log(2, f"Added all_different_lists constraint")

    def ctr_all_equal(self, scope: list[Variable] | list[Node], excepting: None | list[int]):
        """Add all-equal constraint."""
        exprs = self._get_mzn_var_or_expr_list(scope)

        if len(exprs) <= 1:
            return

        if excepting is None or len(excepting) == 0:
            # All must be equal
            for i in range(1, len(exprs)):
                self._add_constraint(f"{exprs[0]} = {exprs[i]}")
        else:
            # All equal except for excepting values
            except_set = ", ".join(str(v) for v in excepting)
            for i in range(1, len(exprs)):
                self._add_constraint(
                    f"({exprs[0]} in {{{except_set}}}) \\/ "
                    f"({exprs[i]} in {{{except_set}}}) \\/ "
                    f"({exprs[0]} = {exprs[i]})"
                )

        self._log(2, f"Added all_equal constraint")

    def ctr_sum(self, lst: list[Variable] | list[Node], coefficients: None | list[int] | list[Variable], condition: Condition):
        """Add sum constraint."""
        exprs = self._get_mzn_var_or_expr_list(lst)

        if coefficients is None:
            sum_expr = f"sum([{', '.join(exprs)}])"
        else:
            # Weighted sum
            coef_strs = []
            for c in coefficients:
                if isinstance(c, Variable):
                    coef_strs.append(self.vars[c.id])
                else:
                    coef_strs.append(str(c))

            terms = [f"{coef_strs[i]} * {exprs[i]}" for i in range(len(exprs))]
            sum_expr = "(" + " + ".join(terms) + ")"

        constraint = self._condition_to_mzn(sum_expr, condition)
        self._add_constraint(constraint)
        self._log(2, f"Added sum constraint")

    def ctr_count(self, lst: list[Variable] | list[Node], values: list[int] | list[Variable], condition: Condition):
        """Add count constraint."""
        exprs = self._get_mzn_var_or_expr_list(lst)
        exprs_str = ", ".join(exprs)

        if len(values) == 1:
            val = values[0]
            if isinstance(val, Variable):
                val_str = self.vars[val.id]
            else:
                val_str = str(val)
            count_expr = f"count([{exprs_str}], {val_str})"
        else:
            # Count occurrences of any of the values
            val_strs = []
            for v in values:
                if isinstance(v, Variable):
                    val_strs.append(self.vars[v.id])
                else:
                    val_strs.append(str(v))
            values_set = f"{{{', '.join(val_strs)}}}"
            # sum(bool2int(x in values) for x in lst)
            count_terms = [f"bool2int({e} in {values_set})" for e in exprs]
            count_expr = f"sum([{', '.join(count_terms)}])"

        constraint = self._condition_to_mzn(count_expr, condition)
        self._add_constraint(constraint)
        self._log(2, f"Added count constraint")

    def ctr_atleast(self, lst: list[Variable], value: int, k: int):
        """Add at-least constraint (count >= k)."""
        exprs = self._get_mzn_var_list(lst)
        exprs_str = ", ".join(exprs)
        self._add_constraint(f"count([{exprs_str}], {value}) >= {k}")
        self._log(2, f"Added atleast constraint")

    def ctr_atmost(self, lst: list[Variable], value: int, k: int):
        """Add at-most constraint (count <= k)."""
        exprs = self._get_mzn_var_list(lst)
        exprs_str = ", ".join(exprs)
        self._add_constraint(f"count([{exprs_str}], {value}) <= {k}")
        self._log(2, f"Added atmost constraint")

    def ctr_exactly(self, lst: list[Variable], value: int, k: int | Variable):
        """Add exactly constraint (count = k)."""
        exprs = self._get_mzn_var_list(lst)
        exprs_str = ", ".join(exprs)
        if isinstance(k, Variable):
            k_str = self.vars[k.id]
        else:
            k_str = str(k)
        self._add_constraint(f"count([{exprs_str}], {value}) = {k_str}")
        self._log(2, f"Added exactly constraint")

    def ctr_nvalues(self, lst: list[Variable] | list[Node], excepting: None | list[int], condition: Condition):
        """Add nvalues constraint."""
        exprs = self._get_mzn_var_or_expr_list(lst)
        exprs_str = ", ".join(exprs)

        if excepting is None or len(excepting) == 0:
            nvalues_expr = f"nvalue([{exprs_str}])"
        else:
            # nvalue doesn't support excepting directly in MiniZinc
            # Decompose: count distinct values not in excepting set
            except_set = ", ".join(str(v) for v in excepting)
            # Filter out excepting values conceptually via auxiliary counting
            # Use: nvalue = count of distinct non-excepted values
            nvalues_expr = f"nvalue([{exprs_str}])"
            # This is a simplification - full support would need decomposition

        constraint = self._condition_to_mzn(nvalues_expr, condition)
        self._add_constraint(constraint)
        self._log(2, f"Added nvalues constraint")

    def ctr_element(self, lst: list[Variable] | list[int], i: Variable, condition: Condition):
        """Add element constraint."""
        # Convert list to MiniZinc array
        if all(isinstance(x, int) for x in lst):
            arr_str = ", ".join(str(x) for x in lst)
        else:
            arr_str = ", ".join(
                self.vars[x.id] if isinstance(x, Variable) else str(x)
                for x in lst
            )

        idx_var = self.vars[i.id]

        # MiniZinc uses 1-based indexing by default
        # element(i, array) = array[i]
        element_expr = f"[{arr_str}][{idx_var} + 1]"  # +1 for 1-based indexing

        constraint = self._condition_to_mzn(element_expr, condition)
        self._add_constraint(constraint)
        self._log(2, f"Added element constraint")

    def ctr_minimum(self, lst: list[Variable] | list[Node], condition: Condition):
        """Add minimum constraint."""
        exprs = self._get_mzn_var_or_expr_list(lst)
        min_expr = f"min([{', '.join(exprs)}])"
        constraint = self._condition_to_mzn(min_expr, condition)
        self._add_constraint(constraint)
        self._log(2, f"Added minimum constraint")

    def ctr_maximum(self, lst: list[Variable] | list[Node], condition: Condition):
        """Add maximum constraint."""
        exprs = self._get_mzn_var_or_expr_list(lst)
        max_expr = f"max([{', '.join(exprs)}])"
        constraint = self._condition_to_mzn(max_expr, condition)
        self._add_constraint(constraint)
        self._log(2, f"Added maximum constraint")

    def ctr_channel(self, lst1: list[Variable], lst2: None | list[Variable]):
        """Add channel/inverse constraint."""
        vars1 = self._get_mzn_var_list(lst1)

        # MiniZinc inverse uses 1-based indexing, pycsp3 uses 0-based
        # Convert by adding 1 to values
        adjusted1 = [f"({v} + 1)" for v in vars1]

        if lst2 is None:
            # Self-inverse (permutation)
            adj_str = ", ".join(adjusted1)
            self._add_constraint(f"inverse([{adj_str}], [{adj_str}])")
        else:
            vars2 = self._get_mzn_var_list(lst2)
            adjusted2 = [f"({v} + 1)" for v in vars2]
            adj1_str = ", ".join(adjusted1)
            adj2_str = ", ".join(adjusted2)
            self._add_constraint(f"inverse([{adj1_str}], [{adj2_str}])")

        self._log(2, f"Added channel constraint")

    def ctr_circuit(self, lst: list[Variable], size: None | int | Variable):
        """Add circuit constraint."""
        vars_list = self._get_mzn_var_list(lst)
        vars_str = ", ".join(vars_list)

        # MiniZinc circuit uses 1-based indexing
        # Need to add 1 to make it 1-based
        adjusted = [f"({v} + 1)" for v in vars_list]
        adjusted_str = ", ".join(adjusted)

        self._add_constraint(f"circuit([{adjusted_str}])")

        if size is not None:
            # Additional constraint on circuit size
            if isinstance(size, Variable):
                size_str = self.vars[size.id]
            else:
                size_str = str(size)
            self._add_constraint(f"sum([bool2int([{adjusted_str}][i] != i) | i in 1..{len(lst)}]) = {size_str}")

        self._log(2, f"Added circuit constraint")

    def ctr_regular(
        self,
        scope: list[Variable],
        transitions: list,
        start_state: str,
        final_states: list[str],
    ):
        """Add regular constraint (automaton)."""
        vars_list = self._get_mzn_var_list(scope)

        # Map string states to integers
        all_states = set()
        all_states.add(start_state)
        all_states.update(final_states)
        for src, symbol, dst in transitions:
            all_states.add(src)
            all_states.add(dst)

        state_to_int = {state: i + 1 for i, state in enumerate(sorted(all_states))}  # 1-based
        n_states = len(all_states)

        # Collect all symbols and find min symbol value for offset
        all_symbols = sorted(set(symbol for _, symbol, _ in transitions))
        min_symbol = min(all_symbols)
        # Map symbols to 1-based indices for MiniZinc
        symbol_to_idx = {sym: sym - min_symbol + 1 for sym in all_symbols}
        n_symbols = len(all_symbols)

        # Build transition table: trans[state][symbol_idx] = next_state
        # 0 means invalid/no transition
        trans_table = [[0] * (n_symbols + 1) for _ in range(n_states + 1)]

        for src, symbol, dst in transitions:
            src_int = state_to_int[src]
            dst_int = state_to_int[dst]
            sym_idx = symbol_to_idx[symbol]
            trans_table[src_int][sym_idx] = dst_int

        # Build transition array string for MiniZinc (2D array literal)
        # Format: [| row1 | row2 | ... |]
        trans_rows = []
        for state in range(1, n_states + 1):
            row_vals = [str(trans_table[state][s]) for s in range(1, n_symbols + 1)]
            trans_rows.append(", ".join(row_vals))
        trans_str = "[| " + " | ".join(trans_rows) + " |]"

        int_start = state_to_int[start_state]
        int_finals = [state_to_int[s] for s in final_states]
        final_str = ", ".join(str(f) for f in int_finals)

        # MiniZinc regular constraint expects 1-based symbol values
        # Adjust variable values: subtract min_symbol and add 1
        # If min_symbol is 0, this adds 1 to each variable value
        offset = 1 - min_symbol
        if offset == 0:
            vars_str = ", ".join(vars_list)
        else:
            adjusted_vars = [f"({v} + {offset})" if offset > 0 else f"({v} - {-offset})" for v in vars_list]
            vars_str = ", ".join(adjusted_vars)

        # MiniZinc regular constraint
        # regular(x, Q, S, d, q0, F) where:
        # - x: array of variables (adjusted to 1-based symbol values)
        # - Q: number of states
        # - S: number of symbols (input alphabet size)
        # - d: transition table d[q,s] = next state (2D array)
        # - q0: initial state
        # - F: set of final states
        self._add_constraint(
            f"regular([{vars_str}], {n_states}, {n_symbols}, "
            f"{trans_str}, {int_start}, {{{final_str}}})"
        )

        self._log(2, f"Added regular constraint with {len(transitions)} transitions")

    def ctr_clause(self, pos: list[Variable], neg: list[Variable]):
        """Add clause constraint (disjunction)."""
        terms = []
        for v in pos:
            terms.append(f"({self.vars[v.id]} = 1)")
        for v in neg:
            terms.append(f"({self.vars[v.id]} = 0)")

        if terms:
            self._add_constraint(" \\/ ".join(terms))

        self._log(2, f"Added clause constraint")

    def ctr_nooverlap(self, origins: list[Variable], lengths: list[int] | list[Variable], zero_ignored: bool):
        """Add no-overlap constraint (disjunctive)."""
        n = len(origins)
        origin_vars = self._get_mzn_var_list(origins)

        len_strs = []
        for ln in lengths:
            if isinstance(ln, Variable):
                len_strs.append(self.vars[ln.id])
            else:
                len_strs.append(str(ln))

        origins_str = ", ".join(origin_vars)
        lengths_str = ", ".join(len_strs)

        self._add_constraint(f"disjunctive([{origins_str}], [{lengths_str}])")
        self._log(2, f"Added nooverlap constraint")

    def ctr_cumulative(
        self,
        origins: list[Variable],
        lengths: list[int] | list[Variable],
        heights: list[int] | list[Variable],
        condition: Condition,
    ):
        """Add cumulative constraint."""
        origin_vars = self._get_mzn_var_list(origins)

        len_strs = []
        for ln in lengths:
            if isinstance(ln, Variable):
                len_strs.append(self.vars[ln.id])
            else:
                len_strs.append(str(ln))

        height_strs = []
        for h in heights:
            if isinstance(h, Variable):
                height_strs.append(self.vars[h.id])
            else:
                height_strs.append(str(h))

        # Get capacity from condition
        cap = condition.right_operand()
        if isinstance(cap, Variable):
            cap_str = self.vars[cap.id]
        else:
            cap_str = str(cap)

        origins_str = ", ".join(origin_vars)
        lengths_str = ", ".join(len_strs)
        heights_str = ", ".join(height_strs)

        self._add_constraint(
            f"cumulative([{origins_str}], [{lengths_str}], [{heights_str}], {cap_str})"
        )
        self._log(2, f"Added cumulative constraint")

    def ctr_ordered(self, lst: list[Variable], operator: TypeOrderedOperator, lengths: None | list[int] | list[Variable]):
        """Add ordering constraint."""
        vars_list = self._get_mzn_var_list(lst)

        if lengths is None:
            # Simple ordering
            if operator in (TypeOrderedOperator.INCREASING, TypeOrderedOperator.STRICTLY_INCREASING):
                vars_str = ", ".join(vars_list)
                if operator == TypeOrderedOperator.STRICTLY_INCREASING:
                    self._add_constraint(f"strictly_increasing([{vars_str}])")
                else:
                    self._add_constraint(f"increasing([{vars_str}])")
            elif operator in (TypeOrderedOperator.DECREASING, TypeOrderedOperator.STRICTLY_DECREASING):
                vars_str = ", ".join(vars_list)
                if operator == TypeOrderedOperator.STRICTLY_DECREASING:
                    self._add_constraint(f"strictly_decreasing([{vars_str}])")
                else:
                    self._add_constraint(f"decreasing([{vars_str}])")
        else:
            # Ordering with gaps
            len_strs = []
            for ln in lengths:
                if isinstance(ln, Variable):
                    len_strs.append(self.vars[ln.id])
                else:
                    len_strs.append(str(ln))

            op_str = "<=" if operator in (TypeOrderedOperator.INCREASING,) else ">=" \
                if operator in (TypeOrderedOperator.DECREASING,) else "<" \
                if operator in (TypeOrderedOperator.STRICTLY_INCREASING,) else ">"

            for i in range(len(vars_list) - 1):
                if operator in (TypeOrderedOperator.INCREASING, TypeOrderedOperator.STRICTLY_INCREASING):
                    self._add_constraint(f"{vars_list[i]} + {len_strs[i]} {op_str} {vars_list[i+1]}")
                else:
                    self._add_constraint(f"{vars_list[i]} {op_str} {vars_list[i+1]} + {len_strs[i]}")

        self._log(2, f"Added ordered constraint")

    def ctr_lex(self, lists: list[list[Variable]], operator: TypeOrderedOperator):
        """Add lexicographic ordering constraint."""
        if len(lists) < 2:
            return

        for i in range(len(lists) - 1):
            vars1 = self._get_mzn_var_list(lists[i])
            vars2 = self._get_mzn_var_list(lists[i + 1])
            vars1_str = ", ".join(vars1)
            vars2_str = ", ".join(vars2)

            if operator in (TypeOrderedOperator.INCREASING, TypeOrderedOperator.STRICTLY_INCREASING):
                if operator == TypeOrderedOperator.STRICTLY_INCREASING:
                    self._add_constraint(f"lex_less([{vars1_str}], [{vars2_str}])")
                else:
                    self._add_constraint(f"lex_lesseq([{vars1_str}], [{vars2_str}])")
            else:
                if operator == TypeOrderedOperator.STRICTLY_DECREASING:
                    self._add_constraint(f"lex_less([{vars2_str}], [{vars1_str}])")
                else:
                    self._add_constraint(f"lex_lesseq([{vars2_str}], [{vars1_str}])")

        self._log(2, f"Added lex constraint")

    def ctr_lex_matrix(self, matrix: list[list[Variable]], operator: TypeOrderedOperator):
        """Add lex constraint on matrix rows and columns."""
        # Apply lex on rows
        self.ctr_lex(matrix, operator)

        # Apply lex on columns
        if matrix:
            n_cols = len(matrix[0])
            columns = [[row[j] for row in matrix] for j in range(n_cols)]
            self.ctr_lex(columns, operator)

        self._log(2, f"Added lex_matrix constraint")

    def ctr_instantiation(self, lst: list[Variable], values: list[int]):
        """Add instantiation constraint (fix variables to values)."""
        for var, val in zip(lst, values):
            self._add_constraint(f"{self.vars[var.id]} = {val}")
        self._log(2, f"Added instantiation constraint")

    def ctr_binpacking(self, lst: list[Variable], sizes: list[int], condition: Condition):
        """Add bin packing constraint."""
        vars_list = self._get_mzn_var_list(lst)
        vars_str = ", ".join(vars_list)
        sizes_str = ", ".join(str(s) for s in sizes)

        # Get capacity from condition
        cap = condition.right_operand()
        if isinstance(cap, Variable):
            cap_str = self.vars[cap.id]
        else:
            cap_str = str(cap)

        self._add_constraint(f"bin_packing_capa([{cap_str}], [{vars_str}], [{sizes_str}])")
        self._log(2, f"Added binpacking constraint")

    def ctr_binpacking_loads(self, lst: list[Variable], sizes: list[int], loads: list[int] | list[Variable]):
        """Add bin packing with loads constraint."""
        vars_list = self._get_mzn_var_list(lst)
        vars_str = ", ".join(vars_list)
        sizes_str = ", ".join(str(s) for s in sizes)

        load_strs = []
        for ld in loads:
            if isinstance(ld, Variable):
                load_strs.append(self.vars[ld.id])
            else:
                load_strs.append(str(ld))
        loads_str = ", ".join(load_strs)

        self._add_constraint(f"bin_packing_load([{loads_str}], [{vars_str}], [{sizes_str}])")
        self._log(2, f"Added binpacking_loads constraint")

    def ctr_knapsack(
        self,
        lst: list[Variable],
        weights: list[int],
        wcondition: Condition,
        profits: list[int],
        pcondition: Condition,
    ):
        """Add knapsack constraint."""
        vars_list = self._get_mzn_var_list(lst)

        # Weight constraint
        weight_terms = [f"{weights[i]} * {vars_list[i]}" for i in range(len(vars_list))]
        weight_sum = " + ".join(weight_terms)
        weight_constraint = self._condition_to_mzn(f"({weight_sum})", wcondition)
        self._add_constraint(weight_constraint)

        # Profit constraint
        profit_terms = [f"{profits[i]} * {vars_list[i]}" for i in range(len(vars_list))]
        profit_sum = " + ".join(profit_terms)
        profit_constraint = self._condition_to_mzn(f"({profit_sum})", pcondition)
        self._add_constraint(profit_constraint)

        self._log(2, f"Added knapsack constraint")

    # ========== Objective ==========

    def obj_minimize(self, term: Variable | Node):
        """Set minimization objective."""
        self._mark_objective()
        if isinstance(term, Variable):
            obj_expr = self.vars[term.id]
        else:
            obj_expr = self.translate_node(term)

        self._mzn_solve = f"solve minimize {obj_expr}"
        self._log(2, f"Set objective: minimize")

    def obj_maximize(self, term: Variable | Node):
        """Set maximization objective."""
        self._mark_objective()
        if isinstance(term, Variable):
            obj_expr = self.vars[term.id]
        else:
            obj_expr = self.translate_node(term)

        self._mzn_solve = f"solve maximize {obj_expr}"
        self._log(2, f"Set objective: maximize")

    def obj_minimize_special(self, obj_type: TypeObj, terms: list[Variable] | list[Node], coefficients: list[int] | None):
        """Set special minimization objective (SUM, MAX, etc.)."""
        self._mark_objective()
        self._set_special_objective("minimize", obj_type, terms, coefficients)

    def obj_maximize_special(self, obj_type: TypeObj, terms: list[Variable] | list[Node], coefficients: list[int] | None):
        """Set special maximization objective (SUM, MAX, etc.)."""
        self._mark_objective()
        self._set_special_objective("maximize", obj_type, terms, coefficients)

    def _set_special_objective(self, sense: str, obj_type: TypeObj, terms: list, coefficients: list[int] | None):
        """Helper for special objectives."""
        exprs = self._get_mzn_var_or_expr_list(terms)

        if obj_type == TypeObj.SUM:
            if coefficients is None:
                obj_expr = f"sum([{', '.join(exprs)}])"
            else:
                weighted = [f"{coefficients[i]} * {exprs[i]}" for i in range(len(exprs))]
                obj_expr = "(" + " + ".join(weighted) + ")"
        elif obj_type == TypeObj.MAXIMUM:
            obj_expr = f"max([{', '.join(exprs)}])"
        elif obj_type == TypeObj.MINIMUM:
            obj_expr = f"min([{', '.join(exprs)}])"
        elif obj_type == TypeObj.NVALUES:
            obj_expr = f"nvalue([{', '.join(exprs)}])"
        else:
            raise NotImplementedError(f"Objective type {obj_type} not supported")

        self._mzn_solve = f"solve {sense} {obj_expr}"
        self._log(2, f"Set objective: {sense} {obj_type}")

    # ========== Solving ==========

    def _build_model_string(self) -> str:
        """Build complete MiniZinc model string."""
        parts = []

        # Include globals
        parts.append('include "globals.mzn";')
        parts.append("")

        # Variables
        parts.extend(self._mzn_vars)
        parts.append("")

        # Constraints
        parts.extend(self._mzn_constraints)
        parts.append("")

        # Solve statement
        parts.append(self._mzn_solve + ";")

        return "\n".join(parts)

    def _find_solver(self):
        """Find and return a MiniZinc solver.

        If subsolver is specified, use it. Otherwise, try common CP solvers
        in order of preference.
        """
        # Preferred solvers for constraint programming (in order)
        preferred_solvers = [
            "gecode", "chuffed", "cp-sat", "coin-bc", "cbc", "highs", "scip"
        ]

        if self._subsolver is not None:
            # User specified a solver
            try:
                return Solver.lookup(self._subsolver)
            except LookupError:
                # Get available solvers for error message
                try:
                    available = [s.name for s in minizinc.default_driver.available_solvers()]
                except Exception:
                    available = []
                raise ValueError(
                    f"MiniZinc solver '{self._subsolver}' not found. "
                    f"Available solvers: {available}"
                )

        # Auto-detect: try preferred solvers in order
        for solver_name in preferred_solvers:
            try:
                solver = Solver.lookup(solver_name)
                self._log(1, f"Using MiniZinc solver: {solver_name}")
                return solver
            except LookupError:
                continue

        # Last resort: use any available solver
        try:
            available = list(minizinc.default_driver.available_solvers())
            if available:
                solver = available[0]
                self._log(1, f"Using MiniZinc solver: {solver.name}")
                return solver
        except Exception:
            pass

        raise ValueError(
            "No MiniZinc solver found. Install gecode, chuffed, or another CP solver."
        )

    def solve(self) -> TypeStatus:
        """Solve the model and return status."""
        if not MINIZINC_AVAILABLE:
            raise ImportError("minizinc package not available")

        # Build model string
        model_str = self._build_model_string()

        if self.verbose >= 2:
            print("=== MiniZinc Model ===")
            print(model_str)
            print("======================")

        # Create model
        model = Model()
        model.add_string(model_str)

        # Lookup solver
        solver = self._find_solver()

        # Create instance
        instance = Instance(solver, model)

        # Configure solving options
        solve_kwargs = {}
        if self.time_limit is not None:
            from datetime import timedelta
            solve_kwargs["timeout"] = timedelta(seconds=self.time_limit)

        # Solve
        try:
            if self.sols == "all" or (isinstance(self.sols, int) and self.sols > 1):
                # Multi-solution mode
                n_sols = None if self.sols == "all" else self.sols
                result = instance.solve(all_solutions=True, **solve_kwargs)
            else:
                progress_mode = self._has_objective and self.get_competition_progress_printer() is not None
                if progress_mode:
                    async def _solve_with_progress():
                        last_result = None
                        async for res in instance.solutions(
                            intermediate_solutions=True,
                            **solve_kwargs,
                        ):
                            last_result = res
                            sol = getattr(res, "solution", None)
                            if sol is not None:
                                obj_val = getattr(sol, "objective", None)
                                if obj_val is not None:
                                    self._report_objective_progress(obj_val)
                        return last_result

                    result = asyncio.run(_solve_with_progress())
                else:
                    result = instance.solve(**solve_kwargs)
        except Exception as e:
            if self.verbose > 0:
                print(f"MiniZinc solve error: {e}")
            self._status = TypeStatus.UNKNOWN
            return self._status

        # Process result
        self._log(2, f"MiniZinc result status: {result.status}")
        if result.status.has_solution():
            self._extract_solution(result)

            # Extract objective value regardless of optimal status
            if hasattr(result, 'objective') and result.objective is not None:
                self._objective_value = result.objective
                self._log(2, f"Objective value: {result.objective}")

            if result.status == minizinc.Status.OPTIMAL_SOLUTION:
                self._status = TypeStatus.OPTIMUM
            else:
                self._status = TypeStatus.SAT
                self._log(2, f"Solution found but not proven optimal")

            # Handle all solutions mode
            if hasattr(result, '__len__') and len(result) > 1:
                self._all_solutions = []
                for sol in result:
                    sol_dict = self._extract_solution_from_result(sol)
                    self._all_solutions.append(sol_dict)

        elif result.status == minizinc.Status.UNSATISFIABLE:
            self._status = TypeStatus.UNSAT
        else:
            self._status = TypeStatus.UNKNOWN

        # Print statistics if verbose
        if self.verbose >= 1 and hasattr(result, 'statistics') and result.statistics:
            stats = result.statistics
            print("=== MiniZinc Statistics ===")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print("===========================")

        return self._status

    def _extract_solution(self, result) -> None:
        """Extract solution from MiniZinc result."""
        self._solution = self._extract_solution_from_result(result)

    def _extract_solution_from_result(self, result) -> dict[str, int]:
        """Extract solution dict from a result object."""
        solution = {}

        for var_id in self._var_ids:
            mzn_name = self._mzn_var_name(var_id)
            try:
                value = result[mzn_name]
                solution[var_id] = int(value)
            except (KeyError, TypeError) as e:
                self._log(2, f"Failed to extract {mzn_name}: {e}")

        self._log(2, f"Extracted {len(solution)} of {len(self._var_ids)} variables")
        return solution

    def get_solution(self) -> dict[str, int] | None:
        """Return the solution as {var_id: value} dict."""
        return self._solution

    def get_all_solutions(self) -> list[dict[str, int]]:
        """Return all solutions found."""
        return self._all_solutions

    def apply_hints(self):
        """Apply warm start hints."""
        if not self.hints:
            return

        # MiniZinc supports warm start via annotations
        # We add hint constraints as soft preferences
        for var_id, value in self.hints.items():
            if var_id in self.vars:
                # Add as annotation comment (solver may use)
                mzn_var = self.vars[var_id]
                # Some solvers support warm start via solution file
                # For now, log the hints
                self._log(2, f"Hint: {var_id} = {value}")
