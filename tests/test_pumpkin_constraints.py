"""Constraint coverage tests for Pumpkin solver backend.

These tests are skipped if pumpkin-solver is not available.
"""

import pytest
from pycsp3 import *
from pycsp3.functions import Table
from pycsp3_solvers_extra import solve
from pycsp3_solvers_extra.backends import get_backend

PUMPKIN_AVAILABLE = get_backend("pumpkin") is not None

pytestmark = pytest.mark.skipif(
    not PUMPKIN_AVAILABLE,
    reason="Pumpkin backend not available (pumpkin-solver not installed)"
)


class TestBasicSatisfaction:
    """Tests for basic satisfaction problems."""

    def test_simple_alldifferent(self):
        """Test simple AllDifferent constraint."""
        x = VarArray(size=4, dom=range(1, 5))
        satisfy(AllDifferent(x))

        status = solve(solver="pumpkin")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert len(set(sol)) == 4, "AllDifferent violated"

    def test_sum_equals(self):
        """Test Sum constraint with equality."""
        x = VarArray(size=3, dom=range(1, 10))
        satisfy(Sum(x) == 15)

        status = solve(solver="pumpkin")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sum(sol) == 15

    def test_sum_less_than(self):
        """Test Sum constraint with less than."""
        x = VarArray(size=3, dom=range(1, 10))
        satisfy(Sum(x) < 10)

        status = solve(solver="pumpkin")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sum(sol) < 10

    def test_unsatisfiable(self):
        """Test unsatisfiable problem detection."""
        x = VarArray(size=5, dom=range(1, 4))
        satisfy(AllDifferent(x))

        status = solve(solver="pumpkin")
        assert status == UNSAT


class TestOptimization:
    """Tests for optimization problems."""

    def test_minimize_sum(self):
        """Test minimization of sum."""
        x = VarArray(size=3, dom=range(1, 10))
        satisfy(AllDifferent(x))
        minimize(Sum(x))

        status = solve(solver="pumpkin")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sum(sol) == 6  # 1 + 2 + 3

    def test_maximize_sum(self):
        """Test maximization of sum."""
        x = VarArray(size=3, dom=range(1, 10))
        satisfy(AllDifferent(x))
        maximize(Sum(x))

        status = solve(solver="pumpkin")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sum(sol) == 24  # 7 + 8 + 9


class TestTableConstraints:
    """Tests for table constraints."""

    def test_table_positive(self):
        """Test positive table constraint."""
        x = VarArray(size=2, dom=range(1, 4))
        allowed = [(1, 2), (2, 3), (3, 1)]
        satisfy(Table(scope=x, supports=allowed))

        status = solve(solver="pumpkin")
        assert status in (SAT, OPTIMUM)

        sol = tuple(values(x))
        assert sol in allowed


class TestElementConstraint:
    """Tests for element constraint."""

    def test_element_variable_array(self):
        """Test Element with variable array."""
        x = VarArray(size=5, dom=range(10, 20))
        i = Var(dom=range(5))
        satisfy(x[i] == 15)

        status = solve(solver="pumpkin")
        assert status in (SAT, OPTIMUM)

        sol_x = values(x)
        sol_i = value(i)
        assert sol_x[sol_i] == 15


class TestMinMaxConstraints:
    """Tests for minimum/maximum constraints."""

    def test_minimum(self):
        """Test Minimum constraint."""
        x = VarArray(size=4, dom=range(1, 10))
        satisfy(Minimum(x) == 3)

        status = solve(solver="pumpkin")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert min(sol) == 3

    def test_maximum(self):
        """Test Maximum constraint."""
        x = VarArray(size=4, dom=range(1, 10))
        satisfy(Maximum(x) == 7)

        status = solve(solver="pumpkin")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert max(sol) == 7


class TestAllEqual:
    """Tests for AllEqual and NotAllEqual constraints."""

    def test_all_equal(self):
        """Test AllEqual constraint."""
        x = VarArray(size=4, dom=range(1, 10))
        satisfy(AllEqual(x))

        status = solve(solver="pumpkin")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert len(set(sol)) == 1, "AllEqual violated"

    def test_not_all_equal(self):
        """Test NotAllEqual constraint (at least 2 distinct values)."""
        x = VarArray(size=4, dom=range(1, 5))
        satisfy(NotAllEqual(x))

        status = solve(solver="pumpkin")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert len(set(sol)) > 1, "NotAllEqual requires at least 2 distinct values"


class TestIntensionConstraints:
    """Tests for intension (expression) constraints."""

    def test_simple_comparison(self):
        """Test simple comparison constraint."""
        x = Var(dom=range(1, 10))
        y = Var(dom=range(1, 10))
        satisfy(x < y)
        satisfy(x + y == 10)

        status = solve(solver="pumpkin")
        assert status in (SAT, OPTIMUM)

        sol_x, sol_y = value(x), value(y)
        assert sol_x < sol_y
        assert sol_x + sol_y == 10

    def test_arithmetic_expression(self):
        """Test arithmetic expression constraint."""
        x = Var(dom=range(1, 10))
        y = Var(dom=range(1, 10))
        satisfy(x * 2 + y == 15)

        status = solve(solver="pumpkin")
        assert status in (SAT, OPTIMUM)

        sol_x, sol_y = value(x), value(y)
        assert sol_x * 2 + sol_y == 15


class TestCumulativeConstraint:
    """Tests for cumulative constraint (Pumpkin specialty)."""

    def test_cumulative_basic(self):
        """Test basic Cumulative constraint."""
        starts = VarArray(size=3, dom=range(10))
        durations = [2, 3, 2]
        heights = [1, 2, 1]
        capacity = 3

        satisfy(Cumulative(origins=starts, lengths=durations, heights=heights) <= capacity)
        satisfy(AllDifferent(starts))

        status = solve(solver="pumpkin")
        assert status in (SAT, OPTIMUM)

        sol = values(starts)
        # Verify capacity is respected at each time point
        for t in range(15):
            load = sum(
                heights[i]
                for i in range(3)
                if sol[i] <= t < sol[i] + durations[i]
            )
            assert load <= capacity, f"Capacity violated at time {t}"
