"""
Tests for pycsp3_solvers_extra.backends.base module.

These tests verify the BaseCallbacks class including:
- NotImplementedError methods
- Node translation and caching
- Utility methods
"""

import pytest
from unittest.mock import Mock, MagicMock

from pycsp3.classes.auxiliary.enums import TypeStatus
from pycsp3.classes.nodes import Node, TypeNode
from pycsp3.classes.main.variables import Variable

from pycsp3_solvers_extra.backends.base import BaseCallbacks


class StubBackend(BaseCallbacks):
    """Minimal concrete implementation for testing BaseCallbacks."""

    def solve(self) -> TypeStatus:
        return TypeStatus.SAT

    def get_solution(self):
        return self._solution


class TestInitialization:
    """Test BaseCallbacks initialization."""

    def test_default_values(self):
        """Default initialization values."""
        backend = StubBackend()
        assert backend.time_limit is None
        assert backend.sols is None
        assert backend.verbose == 0
        assert backend.options == ""
        assert backend.hints == {}
        assert backend.vars == {}

    def test_custom_values(self):
        """Custom initialization values."""
        backend = StubBackend(
            time_limit=60,
            sols=5,
            verbose=2,
            options="--opt=value",
            hints={"x": 1}
        )
        assert backend.time_limit == 60
        assert backend.sols == 5
        assert backend.verbose == 2
        assert backend.options == "--opt=value"
        assert backend.hints == {"x": 1}


class TestNotImplementedMethods:
    """Test NotImplementedError branches."""

    def test_pow_not_implemented(self):
        """_pow raises NotImplementedError."""
        backend = StubBackend()
        with pytest.raises(NotImplementedError, match="Power not supported"):
            backend._pow(2, 3)

    def test_abs_not_implemented(self):
        """_abs raises NotImplementedError."""
        backend = StubBackend()
        with pytest.raises(NotImplementedError, match="Abs not supported"):
            backend._abs(5)

    def test_min_not_implemented(self):
        """_min raises NotImplementedError."""
        backend = StubBackend()
        with pytest.raises(NotImplementedError, match="Min not supported"):
            backend._min([1, 2])

    def test_max_not_implemented(self):
        """_max raises NotImplementedError."""
        backend = StubBackend()
        with pytest.raises(NotImplementedError, match="Max not supported"):
            backend._max([1, 2])

    def test_and_not_implemented(self):
        """_and raises NotImplementedError."""
        backend = StubBackend()
        with pytest.raises(NotImplementedError, match="And not supported"):
            backend._and([True, False])

    def test_or_not_implemented(self):
        """_or raises NotImplementedError."""
        backend = StubBackend()
        with pytest.raises(NotImplementedError, match="Or not supported"):
            backend._or([True, False])

    def test_not_not_implemented(self):
        """_not raises NotImplementedError."""
        backend = StubBackend()
        with pytest.raises(NotImplementedError, match="Not not supported"):
            backend._not(True)

    def test_xor_not_implemented(self):
        """_xor raises NotImplementedError."""
        backend = StubBackend()
        with pytest.raises(NotImplementedError, match="Xor not supported"):
            backend._xor([True, False])

    def test_iff_not_implemented(self):
        """_iff raises NotImplementedError."""
        backend = StubBackend()
        with pytest.raises(NotImplementedError, match="Iff not supported"):
            backend._iff(True, False)

    def test_imp_not_implemented(self):
        """_imp raises NotImplementedError."""
        backend = StubBackend()
        with pytest.raises(NotImplementedError, match="Implication not supported"):
            backend._imp(True, False)

    def test_if_then_else_not_implemented(self):
        """_if_then_else raises NotImplementedError."""
        backend = StubBackend()
        with pytest.raises(NotImplementedError, match="If-then-else not supported"):
            backend._if_then_else(True, 1, 2)

    def test_in_set_not_implemented(self):
        """_in_set raises NotImplementedError."""
        backend = StubBackend()
        with pytest.raises(NotImplementedError, match="Set membership not supported"):
            backend._in_set(1, [1, 2, 3])

    def test_not_in_set_not_implemented(self):
        """_not_in_set raises NotImplementedError."""
        backend = StubBackend()
        with pytest.raises(NotImplementedError, match="Set non-membership not supported"):
            backend._not_in_set(1, [1, 2, 3])

    def test_new_aux_int_var_not_implemented(self):
        """new_aux_int_var raises NotImplementedError."""
        backend = StubBackend()
        with pytest.raises(NotImplementedError, match="Auxiliary variable creation"):
            backend.new_aux_int_var(0, 10)


class TestArithmeticOperations:
    """Test default arithmetic operation implementations."""

    def test_add(self):
        """_add returns sum."""
        backend = StubBackend()
        assert backend._add(2, 3) == 5

    def test_sub(self):
        """_sub returns difference."""
        backend = StubBackend()
        assert backend._sub(5, 3) == 2

    def test_mul(self):
        """_mul returns product."""
        backend = StubBackend()
        assert backend._mul(4, 3) == 12

    def test_div(self):
        """_div returns integer division."""
        backend = StubBackend()
        assert backend._div(7, 2) == 3

    def test_mod(self):
        """_mod returns modulo."""
        backend = StubBackend()
        assert backend._mod(7, 3) == 1

    def test_neg(self):
        """_neg returns negation."""
        backend = StubBackend()
        assert backend._neg(5) == -5


class TestComparisonOperations:
    """Test default comparison operation implementations."""

    def test_eq(self):
        """_eq returns equality result."""
        backend = StubBackend()
        assert backend._eq(3, 3) is True
        assert backend._eq(3, 4) is False

    def test_ne(self):
        """_ne returns inequality result."""
        backend = StubBackend()
        assert backend._ne(3, 4) is True
        assert backend._ne(3, 3) is False

    def test_lt(self):
        """_lt returns less-than result."""
        backend = StubBackend()
        assert backend._lt(2, 3) is True
        assert backend._lt(3, 3) is False

    def test_le(self):
        """_le returns less-or-equal result."""
        backend = StubBackend()
        assert backend._le(2, 3) is True
        assert backend._le(3, 3) is True
        assert backend._le(4, 3) is False

    def test_gt(self):
        """_gt returns greater-than result."""
        backend = StubBackend()
        assert backend._gt(4, 3) is True
        assert backend._gt(3, 3) is False

    def test_ge(self):
        """_ge returns greater-or-equal result."""
        backend = StubBackend()
        assert backend._ge(4, 3) is True
        assert backend._ge(3, 3) is True
        assert backend._ge(2, 3) is False


class TestUtilityMethods:
    """Test utility methods."""

    def test_log_at_verbose_level(self, capsys):
        """_log prints when verbose >= level."""
        backend = StubBackend(verbose=2)
        backend._log(1, "test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_log_below_verbose_level(self, capsys):
        """_log silent when verbose < level."""
        backend = StubBackend(verbose=0)
        backend._log(1, "test message")
        captured = capsys.readouterr()
        assert "test message" not in captured.out

    def test_log_at_exact_level(self, capsys):
        """_log prints when verbose == level."""
        backend = StubBackend(verbose=1)
        backend._log(1, "test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_clear_expression_cache(self):
        """clear_expression_cache empties _node_cache."""
        backend = StubBackend()
        backend._node_cache["test_key"] = "test_value"
        assert len(backend._node_cache) == 1

        backend.clear_expression_cache()
        assert len(backend._node_cache) == 0

    def test_get_objective_value_default(self):
        """get_objective_value returns None by default."""
        backend = StubBackend()
        assert backend.get_objective_value() is None

    def test_get_objective_value_after_set(self):
        """get_objective_value returns set value."""
        backend = StubBackend()
        backend._objective_value = 42
        assert backend.get_objective_value() == 42

    def test_apply_hints_default_noop(self):
        """apply_hints is a no-op by default."""
        backend = StubBackend(hints={"x": 1})
        # Should not raise
        backend.apply_hints()


class TestDecomposeCall:
    """Test decompose_call default behavior."""

    def test_returns_none_by_default(self):
        """decompose_call returns None by default."""
        backend = StubBackend()
        mock_call = Mock()
        mock_ctx = Mock()
        assert backend.decompose_call(mock_call, mock_ctx) is None


class TestNodeStructuralKey:
    """Test _node_structural_key for cache key generation."""

    def test_int_node_key(self):
        """INT node produces (INT, value) key."""
        backend = StubBackend()
        node = Mock()
        node.type = TypeNode.INT
        node.cnt = 42

        key = backend._node_structural_key(node)
        assert key == ("INT", 42)

    def test_symbol_node_key(self):
        """SYMBOL node produces (SYMBOL, value) key."""
        backend = StubBackend()
        node = Mock()
        node.type = TypeNode.SYMBOL
        node.cnt = "test_symbol"

        key = backend._node_structural_key(node)
        assert key == ("SYMBOL", "test_symbol")

    def test_var_node_key(self):
        """VAR node produces (VAR, id) key."""
        backend = StubBackend()
        var_mock = Mock()
        var_mock.id = "x"

        node = Mock()
        node.type = TypeNode.VAR
        node.cnt = var_mock

        key = backend._node_structural_key(node)
        assert key == ("VAR", "x")


class TestTranslateNodeImpl:
    """Test _translate_node_impl for various node types."""

    def test_int_node(self):
        """INT node returns constant value."""
        backend = StubBackend()
        node = Mock()
        node.type = TypeNode.INT
        node.cnt = 42

        result = backend._translate_node_impl(node)
        assert result == 42

    def test_symbol_node(self):
        """SYMBOL node returns symbolic value."""
        backend = StubBackend()
        node = Mock()
        node.type = TypeNode.SYMBOL
        node.cnt = "test"

        result = backend._translate_node_impl(node)
        assert result == "test"

    def test_var_node_known_variable(self):
        """VAR node returns solver variable."""
        backend = StubBackend()
        backend.vars["x"] = "solver_var_x"

        var_mock = Mock()
        var_mock.id = "x"

        node = Mock()
        node.type = TypeNode.VAR
        node.cnt = var_mock

        result = backend._translate_node_impl(node)
        assert result == "solver_var_x"

    def test_var_node_unknown_raises(self):
        """Unknown variable raises ValueError."""
        backend = StubBackend()
        var_mock = Mock()
        var_mock.id = "unknown"

        node = Mock()
        node.type = TypeNode.VAR
        node.cnt = var_mock

        with pytest.raises(ValueError, match="Unknown variable"):
            backend._translate_node_impl(node)


class TestTranslateNodeCaching:
    """Test translate_node memoization."""

    def test_cache_hit(self):
        """Identical nodes return cached result."""
        backend = StubBackend()

        node = Mock()
        node.type = TypeNode.INT
        node.cnt = 42

        # First call
        result1 = backend.translate_node(node)
        # Second call should use cache
        result2 = backend.translate_node(node)

        assert result1 == result2
        assert result1 == 42

    def test_cache_key_based_on_structure(self):
        """Cache key is based on structural content."""
        backend = StubBackend()

        node1 = Mock()
        node1.type = TypeNode.INT
        node1.cnt = 42

        node2 = Mock()
        node2.type = TypeNode.INT
        node2.cnt = 42

        # Both should produce same key and result
        result1 = backend.translate_node(node1)
        result2 = backend.translate_node(node2)

        assert result1 == result2


class TestApplyCondition:
    """Test apply_condition method."""

    def test_with_int_operand(self):
        """Condition with integer right operand."""
        from pycsp3.classes.auxiliary.conditions import Condition
        from pycsp3.classes.auxiliary.enums import TypeConditionOperator

        backend = StubBackend()

        condition = Mock()
        condition.operator = TypeConditionOperator.EQ
        condition.right_operand = 5

        op, left, right = backend.apply_condition("expr", condition)

        assert op == TypeConditionOperator.EQ
        assert left == "expr"
        assert right == 5

    def test_with_variable_operand(self):
        """Condition with Variable right operand converts to solver var."""
        from pycsp3.classes.auxiliary.enums import TypeConditionOperator

        backend = StubBackend()
        backend.vars["y"] = "solver_var_y"

        var_mock = Mock(spec=Variable)
        var_mock.id = "y"

        condition = Mock()
        condition.operator = TypeConditionOperator.LE
        condition.right_operand = var_mock

        op, left, right = backend.apply_condition("expr", condition)

        assert op == TypeConditionOperator.LE
        assert left == "expr"
        assert right == "solver_var_y"


class TestGetVarList:
    """Test _get_var_list method."""

    def test_converts_variables_to_solver_vars(self):
        """Converts list of pycsp3 Variables to solver variables."""
        backend = StubBackend()
        backend.vars["x"] = "solver_x"
        backend.vars["y"] = "solver_y"

        var_x = Mock(spec=Variable)
        var_x.id = "x"
        var_y = Mock(spec=Variable)
        var_y.id = "y"

        result = backend._get_var_list([var_x, var_y])
        assert result == ["solver_x", "solver_y"]


class TestGetVarOrNodeList:
    """Test _get_var_or_node_list method."""

    def test_handles_variables(self):
        """Handles Variable instances."""
        backend = StubBackend()
        backend.vars["x"] = "solver_x"

        var = Mock(spec=Variable)
        var.id = "x"

        result = backend._get_var_or_node_list([var])
        assert result == ["solver_x"]

    def test_handles_nodes(self):
        """Handles Node instances."""
        backend = StubBackend()

        node = Mock(spec=Node)
        node.type = TypeNode.INT
        node.cnt = 42

        result = backend._get_var_or_node_list([node])
        assert result == [42]

    def test_handles_constants(self):
        """Handles constant values (pass through)."""
        backend = StubBackend()

        result = backend._get_var_or_node_list([1, 2, 3])
        assert result == [1, 2, 3]

    def test_handles_mixed(self):
        """Handles mixed Variable, Node, and constants."""
        backend = StubBackend()
        backend.vars["x"] = "solver_x"

        var = Mock(spec=Variable)
        var.id = "x"

        node = Mock(spec=Node)
        node.type = TypeNode.INT
        node.cnt = 5

        result = backend._get_var_or_node_list([var, node, 10])
        assert result == ["solver_x", 5, 10]
