"""
Base callbacks class for solver backends.

Provides common utilities for translating XCSP3 elements to solver models.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

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
from pycsp3.parser.callbacks import Callbacks


class BaseCallbacks(Callbacks):
    """
    Base class for solver backend callbacks.

    Subclasses should:
    1. Initialize their solver model in __init__
    2. Override var_* methods to create solver variables
    3. Override ctr_* methods to add solver constraints
    4. Override obj_* methods to set objectives
    5. Implement solve() to run the solver
    6. Implement get_solution() to return variable assignments

    Attributes:
        vars: Mapping from pycsp3 variable id to solver variable
        time_limit: Time limit in seconds (None for no limit)
        sols: Number of solutions to find
        verbose: Verbosity level
        options: Solver-specific options string
    """

    def __init__(
        self,
        time_limit: float | None = None,
        sols: int | str | None = None,
        verbose: int = 0,
        options: str = "",
    ):
        super().__init__()
        self.print_general_methods = False
        self.print_specific_methods = False

        # Disable pattern recognition so all intension constraints
        # go through ctr_intension for uniform handling
        self.recognize_unary_primitives = False
        self.recognize_binary_primitives = False
        self.recognize_ternary_primitives = False
        self.recognize_logic_intension = False
        self.recognize_sum_intension = False
        self.recognize_extremum_intension = False

        # Configuration
        self.time_limit = time_limit
        self.sols = sols
        self.verbose = verbose
        self.options = options

        # Variable mapping: pycsp3 var id -> solver var
        self.vars: dict[str, Any] = {}

        # Solution storage
        self._solution: dict[str, int] | None = None
        self._status: TypeStatus = TypeStatus.UNKNOWN

    # ========== Abstract methods to implement ==========

    @abstractmethod
    def solve(self) -> TypeStatus:
        """
        Solve the model and return status.

        Should set self._solution with variable assignments if SAT/OPTIMUM.
        """
        raise NotImplementedError

    @abstractmethod
    def get_solution(self) -> dict[str, int] | None:
        """Return the solution as {var_id: value} dict, or None if no solution."""
        return self._solution

    # ========== Expression tree translation ==========

    def translate_node(self, node: Node) -> Any:
        """
        Translate a pycsp3 Node expression tree to solver expression.

        Subclasses should override this if they need solver-specific translation.
        The default implementation handles common patterns.
        """
        if node.type == TypeNode.VAR:
            var_id = node.cnt.id
            if var_id not in self.vars:
                raise ValueError(f"Unknown variable in expression: {var_id}")
            return self.vars[var_id]

        elif node.type == TypeNode.INT:
            return node.cnt

        elif node.type == TypeNode.SYMBOL:
            # Symbolic values - typically mapped to integers
            return node.cnt

        elif node.type == TypeNode.ADD:
            return self._translate_nary(node, self._add)

        elif node.type == TypeNode.SUB:
            if len(node.cnt) == 1:
                return self._neg(self.translate_node(node[0]))
            left = self.translate_node(node[0])
            right = self.translate_node(node[1])
            return self._sub(left, right)

        elif node.type == TypeNode.MUL:
            return self._translate_nary(node, self._mul)

        elif node.type == TypeNode.DIV:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return self._div(left, right)

        elif node.type == TypeNode.MOD:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return self._mod(left, right)

        elif node.type == TypeNode.POW:
            base = self.translate_node(node.cnt[0])
            exp = self.translate_node(node.cnt[1])
            return self._pow(base, exp)

        elif node.type == TypeNode.NEG:
            return self._neg(self.translate_node(node.cnt[0]))

        elif node.type == TypeNode.ABS:
            return self._abs(self.translate_node(node.cnt[0]))

        elif node.type == TypeNode.DIST:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return self._abs(self._sub(left, right))

        elif node.type in (TypeNode.MIN, TypeNode.MAX):
            args = [self.translate_node(s) for s in node.cnt]
            return self._min(args) if node.type == TypeNode.MIN else self._max(args)

        # Comparison operators - return boolean expression
        elif node.type == TypeNode.EQ:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return self._eq(left, right)

        elif node.type == TypeNode.NE:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return self._ne(left, right)

        elif node.type == TypeNode.LT:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return self._lt(left, right)

        elif node.type == TypeNode.LE:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return self._le(left, right)

        elif node.type == TypeNode.GT:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return self._gt(left, right)

        elif node.type == TypeNode.GE:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return self._ge(left, right)

        # Logical operators
        elif node.type == TypeNode.AND:
            args = [self.translate_node(s) for s in node.cnt]
            return self._and(args)

        elif node.type == TypeNode.OR:
            args = [self.translate_node(s) for s in node.cnt]
            return self._or(args)

        elif node.type == TypeNode.NOT:
            return self._not(self.translate_node(node.cnt[0]))

        elif node.type == TypeNode.XOR:
            args = [self.translate_node(s) for s in node.cnt]
            return self._xor(args)

        elif node.type == TypeNode.IFF:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return self._iff(left, right)

        elif node.type == TypeNode.IMP:
            left = self.translate_node(node.cnt[0])
            right = self.translate_node(node.cnt[1])
            return self._imp(left, right)

        elif node.type == TypeNode.IF:
            # Ternary: if(cond, then, else)
            cond = self.translate_node(node.cnt[0])
            then_val = self.translate_node(node.cnt[1])
            else_val = self.translate_node(node.cnt[2])
            return self._if_then_else(cond, then_val, else_val)

        elif node.type == TypeNode.IN:
            val = self.translate_node(node.cnt[0])
            # sons[1] is a SET node
            set_vals = [self.translate_node(s) for s in node.cnt[1].sons]
            return self._in_set(val, set_vals)

        elif node.type == TypeNode.NOTIN:
            val = self.translate_node(node.cnt[0])
            set_vals = [self.translate_node(s) for s in node.cnt[1].sons]
            return self._not_in_set(val, set_vals)

        elif node.type == TypeNode.SET:
            # Return list of values
            return [self.translate_node(s) for s in node.cnt]

        else:
            raise NotImplementedError(
                f"Node type {node.type} not implemented in translate_node"
            )

    def _translate_nary(self, node: Node, op) -> Any:
        """Translate n-ary operation (ADD, MUL)."""
        result = self.translate_node(node.cnt[0])
        for son in node.cnt[1:]:
            result = op(result, self.translate_node(son))
        return result

    # ========== Arithmetic operations (override in subclass) ==========

    def _add(self, a: Any, b: Any) -> Any:
        return a + b

    def _sub(self, a: Any, b: Any) -> Any:
        return a - b

    def _mul(self, a: Any, b: Any) -> Any:
        return a * b

    def _div(self, a: Any, b: Any) -> Any:
        return a // b

    def _mod(self, a: Any, b: Any) -> Any:
        return a % b

    def _pow(self, base: Any, exp: Any) -> Any:
        raise NotImplementedError("Power not supported by this backend")

    def _neg(self, a: Any) -> Any:
        return -a

    def _abs(self, a: Any) -> Any:
        raise NotImplementedError("Abs not supported by this backend")

    def _min(self, args: list[Any]) -> Any:
        raise NotImplementedError("Min not supported by this backend")

    def _max(self, args: list[Any]) -> Any:
        raise NotImplementedError("Max not supported by this backend")

    # ========== Comparison operations (override in subclass) ==========

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

    # ========== Logical operations (override in subclass) ==========

    def _and(self, args: list[Any]) -> Any:
        raise NotImplementedError("And not supported by this backend")

    def _or(self, args: list[Any]) -> Any:
        raise NotImplementedError("Or not supported by this backend")

    def _not(self, a: Any) -> Any:
        raise NotImplementedError("Not not supported by this backend")

    def _xor(self, args: list[Any]) -> Any:
        raise NotImplementedError("Xor not supported by this backend")

    def _iff(self, a: Any, b: Any) -> Any:
        raise NotImplementedError("Iff not supported by this backend")

    def _imp(self, a: Any, b: Any) -> Any:
        raise NotImplementedError("Implication not supported by this backend")

    def _if_then_else(self, cond: Any, then_val: Any, else_val: Any) -> Any:
        raise NotImplementedError("If-then-else not supported by this backend")

    def _in_set(self, val: Any, set_vals: list[Any]) -> Any:
        raise NotImplementedError("Set membership not supported by this backend")

    def _not_in_set(self, val: Any, set_vals: list[Any]) -> Any:
        raise NotImplementedError("Set non-membership not supported by this backend")

    # ========== Condition handling ==========

    def apply_condition(
        self, expr: Any, condition: Condition
    ) -> tuple[TypeConditionOperator, Any, Any]:
        """
        Extract condition components for constraint building.

        Returns:
            (operator, left_expr, right_operand)
        """
        op = condition.operator
        right = condition.right_operand

        # Handle variable right operand
        if isinstance(right, Variable):
            right = self.vars[right.id]

        return (op, expr, right)

    # ========== Utility methods ==========

    def _log(self, level: int, msg: str) -> None:
        """Log message if verbosity is high enough."""
        if self.verbose >= level:
            print(msg)

    def _get_var_list(self, scope: list[Variable]) -> list[Any]:
        """Convert list of pycsp3 Variables to list of solver variables."""
        return [self.vars[v.id] for v in scope]

    def _get_var_or_node_list(
        self, scope: list[Variable] | list[Node]
    ) -> list[Any]:
        """Convert list of Variables or Nodes to solver expressions."""
        result = []
        for item in scope:
            if isinstance(item, Variable):
                result.append(self.vars[item.id])
            elif isinstance(item, Node):
                result.append(self.translate_node(item))
            else:
                result.append(item)
        return result
