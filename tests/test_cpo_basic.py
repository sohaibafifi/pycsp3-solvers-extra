"""Basic tests for CPO (IBM DOcplex CP Optimizer) backend.

These tests require IBM CPLEX Optimization Studio to be installed.
They will be skipped if CPO is not available.
"""

import pytest
from pycsp3 import *
from pycsp3.functions import Table
from pycsp3_solvers_extra import solve
from pycsp3_solvers_extra.backends import get_backend

# Skip all tests if CPO is not available or CP Optimizer not installed
pytestmark = pytest.mark.skipif(
    get_backend("cpo") is None,
    reason="CPO backend not available"
)


def cpo_available():
    """Check if CPO solver is actually runnable."""
    try:
        from docplex.cp.model import CpoModel
        mdl = CpoModel()
        x = mdl.integer_var(1, 10)
        mdl.add(x == 5)
        sol = mdl.solve(TimeLimit=5)
        return sol is not None
    except Exception:
        return False


# Additional skip for tests that actually run the solver
requires_cpo_solver = pytest.mark.skipif(
    not cpo_available(),
    reason="CP Optimizer solver not installed"
)


class TestBasicSatisfaction:
    """Tests for basic satisfaction problems."""

    @requires_cpo_solver
    def test_simple_alldifferent(self):
        """Test simple AllDifferent constraint."""
        x = VarArray(size=4, dom=range(1, 5))
        satisfy(AllDifferent(x))

        status = solve(solver="cpo")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert len(set(sol)) == 4, "AllDifferent violated"

    @requires_cpo_solver
    def test_sum_equals(self):
        """Test Sum constraint with equality."""
        x = VarArray(size=3, dom=range(1, 10))
        satisfy(Sum(x) == 15)

        status = solve(solver="cpo")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sum(sol) == 15

    @requires_cpo_solver
    def test_sum_less_than(self):
        """Test Sum constraint with less than."""
        x = VarArray(size=3, dom=range(1, 10))
        satisfy(Sum(x) < 10)

        status = solve(solver="cpo")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sum(sol) < 10

    @requires_cpo_solver
    def test_unsatisfiable(self):
        """Test detection of unsatisfiable problem."""
        x = VarArray(size=5, dom=range(1, 4))  # Only 3 values for 5 vars
        satisfy(AllDifferent(x))

        status = solve(solver="cpo")
        assert status == UNSAT


class TestOptimization:
    """Tests for optimization problems."""

    @requires_cpo_solver
    def test_minimize_sum(self):
        """Test minimizing sum."""
        x = VarArray(size=3, dom=range(1, 10))
        satisfy(AllDifferent(x))
        minimize(Sum(x))

        status = solve(solver="cpo")
        assert status == OPTIMUM

        sol = values(x)
        assert sum(sol) == 6  # 1 + 2 + 3

    @requires_cpo_solver
    def test_maximize_sum(self):
        """Test maximizing sum."""
        x = VarArray(size=3, dom=range(1, 10))
        satisfy(AllDifferent(x))
        maximize(Sum(x))

        status = solve(solver="cpo")
        assert status == OPTIMUM

        sol = values(x)
        assert sum(sol) == 24  # 7 + 8 + 9


class TestTableConstraints:
    """Tests for table/extension constraints."""

    @requires_cpo_solver
    def test_table_positive(self):
        """Test positive table constraint (allowed tuples)."""
        x = Var(dom=range(1, 4))
        y = Var(dom=range(1, 4))

        satisfy(
            Table(scope=[x, y], supports=[(1, 2), (2, 3), (3, 1)])
        )

        status = solve(solver="cpo")
        assert status in (SAT, OPTIMUM)

        vx, vy = value(x), value(y)
        assert (vx, vy) in [(1, 2), (2, 3), (3, 1)]


class TestCountingConstraints:
    """Tests for counting constraints."""

    @requires_cpo_solver
    def test_count_equals(self):
        """Test Count constraint with equality."""
        x = VarArray(size=5, dom=range(1, 4))
        satisfy(
            Count(x, value=1) == 2,
            AllDifferent(x[:3])
        )

        status = solve(solver="cpo")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sol.count(1) == 2

    @requires_cpo_solver
    def test_atleast(self):
        """Test Count with >= (at least)."""
        x = VarArray(size=5, dom=range(1, 4))
        satisfy(Count(x, value=1) >= 2)

        status = solve(solver="cpo")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sol.count(1) >= 2

    @requires_cpo_solver
    def test_atmost(self):
        """Test Count with <= (at most)."""
        x = VarArray(size=5, dom=range(1, 4))
        satisfy(Count(x, value=1) <= 2)

        status = solve(solver="cpo")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sol.count(1) <= 2


class TestElementConstraint:
    """Tests for Element constraint."""

    @requires_cpo_solver
    def test_element_variable_array(self):
        """Test Element with variable array (VarArray[var_index])."""
        x = VarArray(size=5, dom=range(1, 10))
        i = Var(dom=range(5))
        result = Var(dom=range(1, 10))

        satisfy(
            AllDifferent(x),
            x[i] == result,
            result == 5
        )

        status = solve(solver="cpo")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        idx = value(i)
        assert sol[idx] == 5


class TestMinMaxConstraints:
    """Tests for Minimum/Maximum constraints."""

    @requires_cpo_solver
    def test_minimum(self):
        """Test Minimum constraint."""
        x = VarArray(size=4, dom=range(1, 10))
        satisfy(
            AllDifferent(x),
            Minimum(x) == 3
        )

        status = solve(solver="cpo")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert min(sol) == 3

    @requires_cpo_solver
    def test_maximum(self):
        """Test Maximum constraint."""
        x = VarArray(size=4, dom=range(1, 10))
        satisfy(
            AllDifferent(x),
            Maximum(x) == 7
        )

        status = solve(solver="cpo")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert max(sol) == 7


class TestOrderingConstraints:
    """Tests for ordering constraints."""

    @requires_cpo_solver
    def test_increasing(self):
        """Test Increasing constraint."""
        x = VarArray(size=4, dom=range(1, 10))
        satisfy(Increasing(x))

        status = solve(solver="cpo")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        for i in range(len(sol) - 1):
            assert sol[i] <= sol[i + 1]

    @requires_cpo_solver
    def test_strictly_increasing(self):
        """Test Strictly Increasing constraint."""
        x = VarArray(size=4, dom=range(1, 10))
        satisfy(Increasing(x, strict=True))

        status = solve(solver="cpo")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        for i in range(len(sol) - 1):
            assert sol[i] < sol[i + 1]


class TestChannelConstraint:
    """Tests for Channel/Inverse constraint."""

    @requires_cpo_solver
    def test_channel(self):
        """Test Channel constraint."""
        n = 4
        x = VarArray(size=n, dom=range(n))
        y = VarArray(size=n, dom=range(n))
        satisfy(Channel(x, y))

        status = solve(solver="cpo")
        assert status in (SAT, OPTIMUM)

        sol_x = values(x)
        sol_y = values(y)
        # Verify inverse relationship
        for i in range(n):
            assert sol_y[sol_x[i]] == i
            assert sol_x[sol_y[i]] == i


class TestAllEqual:
    """Tests for AllEqual constraint."""

    @requires_cpo_solver
    def test_all_equal(self):
        """Test AllEqual constraint."""
        x = VarArray(size=4, dom=range(1, 5))
        satisfy(AllEqual(x))

        status = solve(solver="cpo")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert len(set(sol)) == 1


class TestIntensionConstraints:
    """Tests for intension (expression) constraints."""

    @requires_cpo_solver
    def test_simple_comparison(self):
        """Test simple comparison constraint."""
        x = Var(dom=range(1, 10))
        y = Var(dom=range(1, 10))
        satisfy(x < y)

        status = solve(solver="cpo")
        assert status in (SAT, OPTIMUM)

        assert value(x) < value(y)

    @requires_cpo_solver
    def test_arithmetic_expression(self):
        """Test arithmetic expression constraint."""
        x = Var(dom=range(1, 10))
        y = Var(dom=range(1, 10))
        z = Var(dom=range(1, 20))
        satisfy(x + y == z)

        status = solve(solver="cpo")
        assert status in (SAT, OPTIMUM)

        assert value(x) + value(y) == value(z)

    @requires_cpo_solver
    def test_multiplication(self):
        """Test multiplication constraint."""
        x = Var(dom=range(1, 10))
        y = Var(dom=range(1, 10))
        satisfy(x * y == 12)

        status = solve(solver="cpo")
        assert status in (SAT, OPTIMUM)

        assert value(x) * value(y) == 12
