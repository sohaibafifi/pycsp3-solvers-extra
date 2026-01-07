"""Basic tests for OR-Tools backend."""

import pytest
from pycsp3 import *
from pycsp3.functions import Table
from pycsp3_solvers_extra import solve


class TestBasicSatisfaction:
    """Tests for basic satisfaction problems."""

    def test_simple_alldifferent(self):
        """Test simple AllDifferent constraint."""
        x = VarArray(size=4, dom=range(1, 5))
        satisfy(AllDifferent(x))

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert len(set(sol)) == 4, "AllDifferent violated"

    def test_sum_equals(self):
        """Test Sum constraint with equality."""
        x = VarArray(size=3, dom=range(1, 10))
        satisfy(Sum(x) == 15)

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sum(sol) == 15

    def test_sum_less_than(self):
        """Test Sum constraint with less than."""
        x = VarArray(size=3, dom=range(1, 10))
        satisfy(Sum(x) < 10)

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sum(sol) < 10

    def test_sum_greater_than(self):
        """Test Sum constraint with greater than."""
        x = VarArray(size=3, dom=range(1, 5))
        satisfy(Sum(x) > 10)

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sum(sol) > 10

    def test_unsatisfiable(self):
        """Test detection of unsatisfiable problem."""
        x = VarArray(size=5, dom=range(1, 4))  # Only 3 values for 5 vars
        satisfy(AllDifferent(x))

        status = solve(solver="ortools")
        assert status == UNSAT


class TestOptimization:
    """Tests for optimization problems."""

    def test_minimize_sum(self):
        """Test minimizing sum."""
        x = VarArray(size=3, dom=range(1, 10))
        satisfy(AllDifferent(x))
        minimize(Sum(x))

        status = solve(solver="ortools")
        assert status == OPTIMUM

        sol = values(x)
        assert sum(sol) == 6  # 1 + 2 + 3


    def test_maximize_sum(self):
        """Test maximizing sum."""
        x = VarArray(size=3, dom=range(1, 10))
        satisfy(AllDifferent(x))
        maximize(Sum(x))

        status = solve(solver="ortools")
        assert status == OPTIMUM

        sol = values(x)
        assert sum(sol) == 24  # 7 + 8 + 9


class TestTableConstraints:
    """Tests for table/extension constraints."""

    def test_table_positive(self):
        """Test positive table constraint (allowed tuples)."""
        x = Var(dom=range(1, 4))
        y = Var(dom=range(1, 4))

        # Only allow (1,2), (2,3), (3,1) using Table function
        satisfy(
            Table(scope=[x, y], supports=[(1, 2), (2, 3), (3, 1)])
        )

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        vx, vy = value(x), value(y)
        assert (vx, vy) in [(1, 2), (2, 3), (3, 1)]


class TestCountingConstraints:
    """Tests for counting constraints."""

    def test_count_equals(self):
        """Test Count constraint with equality."""
        x = VarArray(size=5, dom=range(1, 4))
        satisfy(
            Count(x, value=1) == 2,
            AllDifferent(x[:3])  # Add constraint to make solution non-trivial
        )

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sol.count(1) == 2

    def test_atleast(self):
        """Test Count with >= (at least)."""
        x = VarArray(size=5, dom=range(1, 4))
        satisfy(Count(x, value=1) >= 2)

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sol.count(1) >= 2

    def test_atmost(self):
        """Test Count with <= (at most)."""
        x = VarArray(size=5, dom=range(1, 4))
        satisfy(Count(x, value=1) <= 2)

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sol.count(1) <= 2

    def test_exactly(self):
        """Test Count with == (exactly)."""
        x = VarArray(size=5, dom=range(1, 4))
        satisfy(Count(x, value=1) == 2)

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sol.count(1) == 2


class TestElementConstraint:
    """Tests for Element constraint."""

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

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        idx = value(i)
        assert sol[idx] == 5


class TestMinMaxConstraints:
    """Tests for Minimum/Maximum constraints."""

    def test_minimum(self):
        """Test Minimum constraint."""
        x = VarArray(size=4, dom=range(1, 10))
        satisfy(
            AllDifferent(x),
            Minimum(x) == 3
        )

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert min(sol) == 3

    def test_maximum(self):
        """Test Maximum constraint."""
        x = VarArray(size=4, dom=range(1, 10))
        satisfy(
            AllDifferent(x),
            Maximum(x) == 7
        )

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert max(sol) == 7


class TestOrderingConstraints:
    """Tests for ordering constraints."""

    def test_increasing(self):
        """Test Increasing constraint."""
        x = VarArray(size=4, dom=range(1, 10))
        satisfy(Increasing(x))

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        for i in range(len(sol) - 1):
            assert sol[i] <= sol[i + 1]

    def test_strictly_increasing(self):
        """Test Strictly Increasing constraint."""
        x = VarArray(size=4, dom=range(1, 10))
        satisfy(Increasing(x, strict=True))

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        for i in range(len(sol) - 1):
            assert sol[i] < sol[i + 1]

    def test_decreasing(self):
        """Test Decreasing constraint."""
        x = VarArray(size=4, dom=range(1, 10))
        satisfy(Decreasing(x))

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        for i in range(len(sol) - 1):
            assert sol[i] >= sol[i + 1]


class TestChannelConstraint:
    """Tests for Channel/Inverse constraint."""

    def test_channel(self):
        """Test Channel constraint."""
        n = 4
        x = VarArray(size=n, dom=range(n))
        y = VarArray(size=n, dom=range(n))
        satisfy(Channel(x, y))

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol_x = values(x)
        sol_y = values(y)
        # Verify inverse relationship
        for i in range(n):
            assert sol_y[sol_x[i]] == i
            assert sol_x[sol_y[i]] == i


class TestCircuitConstraint:
    """Tests for Circuit constraint."""

    def test_circuit(self):
        """Test Circuit constraint (Hamiltonian cycle)."""
        n = 5
        x = VarArray(size=n, dom=range(n))
        satisfy(Circuit(x))

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        # Verify it's a valid circuit
        visited = set()
        current = 0
        for _ in range(n):
            assert current not in visited, "Revisited node"
            visited.add(current)
            current = sol[current]
        assert current == 0, "Not a cycle"
        assert len(visited) == n, "Not all nodes visited"


class TestAllEqual:
    """Tests for AllEqual constraint."""

    def test_all_equal(self):
        """Test AllEqual constraint."""
        x = VarArray(size=4, dom=range(1, 5))
        satisfy(AllEqual(x))

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert len(set(sol)) == 1


class TestIntensionConstraints:
    """Tests for intension (expression) constraints."""

    def test_simple_comparison(self):
        """Test simple comparison constraint."""
        x = Var(dom=range(1, 10))
        y = Var(dom=range(1, 10))
        satisfy(x < y)

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        assert value(x) < value(y)

    def test_arithmetic_expression(self):
        """Test arithmetic expression constraint."""
        x = Var(dom=range(1, 10))
        y = Var(dom=range(1, 10))
        z = Var(dom=range(1, 20))
        satisfy(x + y == z)

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        assert value(x) + value(y) == value(z)

    def test_multiplication(self):
        """Test multiplication constraint."""
        x = Var(dom=range(1, 10))
        y = Var(dom=range(1, 10))
        satisfy(x * y == 12)

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        assert value(x) * value(y) == 12
