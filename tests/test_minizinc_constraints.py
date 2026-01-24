"""Constraint coverage tests for MiniZinc backend.

These tests are skipped if MiniZinc is not available.
"""

import pytest
from pycsp3 import *
from pycsp3.functions import Table
from pycsp3_solvers_extra import solve
from pycsp3_solvers_extra.backends import get_backend

MINIZINC_AVAILABLE = get_backend("minizinc") is not None

pytestmark = pytest.mark.skipif(
    not MINIZINC_AVAILABLE,
    reason="MiniZinc backend not available"
)


class TestBasicSatisfaction:
    """Tests for basic satisfaction problems."""

    def test_simple_alldifferent(self):
        """Test simple AllDifferent constraint."""
        x = VarArray(size=4, dom=range(1, 5))
        satisfy(AllDifferent(x))

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert len(set(sol)) == 4, "AllDifferent violated"

    def test_sum_equals(self):
        """Test Sum constraint with equality."""
        x = VarArray(size=3, dom=range(1, 10))
        satisfy(Sum(x) == 15)

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sum(sol) == 15

    def test_sum_less_than(self):
        """Test Sum constraint with less than."""
        x = VarArray(size=3, dom=range(1, 10))
        satisfy(Sum(x) < 10)

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sum(sol) < 10

    def test_sum_greater_than(self):
        """Test Sum constraint with greater than."""
        x = VarArray(size=3, dom=range(1, 5))
        satisfy(Sum(x) > 10)

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sum(sol) > 10

    def test_unsatisfiable(self):
        """Test detection of unsatisfiable problem."""
        x = VarArray(size=5, dom=range(1, 4))  # Only 3 values for 5 vars
        satisfy(AllDifferent(x))

        status = solve(solver="minizinc")
        assert status == UNSAT


class TestSolverVariants:
    """Tests for different MiniZinc solver variants."""

    def test_default_solver(self):
        """Test with default solver (gecode)."""
        x = VarArray(size=3, dom=range(1, 4))
        satisfy(AllDifferent(x))

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

    @pytest.mark.skipif(True, reason="Chuffed may not be installed")
    def test_chuffed_solver(self):
        """Test with chuffed solver."""
        x = VarArray(size=3, dom=range(1, 4))
        satisfy(AllDifferent(x))

        status = solve(solver="minizinc/chuffed")
        assert status in (SAT, OPTIMUM)

    def test_explicit_cpsat(self):
        """Test with explicit cp-sat solver."""
        x = VarArray(size=3, dom=range(1, 4))
        satisfy(AllDifferent(x))

        status = solve(solver="minizinc/cp-sat")
        assert status in (SAT, OPTIMUM)


class TestOptimization:
    """Tests for optimization problems."""

    def test_minimize_sum(self):
        """Test minimizing sum."""
        x = VarArray(size=3, dom=range(1, 10))
        satisfy(AllDifferent(x))
        minimize(Sum(x))

        status = solve(solver="minizinc")
        assert status == OPTIMUM

        sol = values(x)
        assert sum(sol) == 6  # 1 + 2 + 3
        assert bound() == 6

    def test_maximize_sum(self):
        """Test maximizing sum."""
        x = VarArray(size=3, dom=range(1, 10))
        satisfy(AllDifferent(x))
        maximize(Sum(x))

        status = solve(solver="minizinc")
        assert status == OPTIMUM

        sol = values(x)
        assert sum(sol) == 24  # 7 + 8 + 9
        assert bound() == 24

    def test_minimize_single_var(self):
        """Test minimizing a single variable."""
        x = Var(dom=range(1, 10))
        y = Var(dom=range(1, 10))
        satisfy(x + y == 10)
        minimize(x)

        status = solve(solver="minizinc")
        assert status == OPTIMUM
        assert value(x) == 1


class TestTableConstraints:
    """Tests for table/extension constraints."""

    def test_table_positive(self):
        """Test positive table constraint (allowed tuples)."""
        x = Var(dom=range(1, 4))
        y = Var(dom=range(1, 4))

        satisfy(
            Table(scope=[x, y], supports=[(1, 2), (2, 3), (3, 1)])
        )

        status = solve(solver="minizinc")
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
            AllDifferent(x[:3])
        )

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sol.count(1) == 2

    def test_atleast(self):
        """Test Count with >= (at least)."""
        x = VarArray(size=5, dom=range(1, 4))
        satisfy(Count(x, value=1) >= 2)

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sol.count(1) >= 2

    def test_atmost(self):
        """Test Count with <= (at most)."""
        x = VarArray(size=5, dom=range(1, 4))
        satisfy(Count(x, value=1) <= 2)

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sol.count(1) <= 2

    def test_exactly(self):
        """Test Count with == (exactly)."""
        x = VarArray(size=5, dom=range(1, 4))
        satisfy(Count(x, value=1) == 2)

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sol.count(1) == 2


class TestElementConstraint:
    """Tests for element constraint."""

    def test_element_basic(self):
        """Test basic element constraint with VarArray."""
        x = VarArray(size=5, dom=range(1, 10))
        i = Var(dom=range(5))
        result = Var(dom=range(1, 10))

        satisfy(
            AllDifferent(x),
            x[i] == result,
            result == 5
        )

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        idx = value(i)
        assert sol[idx] == 5


class TestMinMaxConstraints:
    """Tests for minimum/maximum constraints."""

    def test_minimum(self):
        """Test Minimum constraint."""
        x = VarArray(size=4, dom=range(1, 10))
        satisfy(
            AllDifferent(x),
            Minimum(x) == 2
        )

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert min(sol) == 2

    def test_maximum(self):
        """Test Maximum constraint."""
        x = VarArray(size=4, dom=range(1, 10))
        satisfy(
            AllDifferent(x),
            Maximum(x) == 8
        )

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert max(sol) == 8


class TestOrderingConstraints:
    """Tests for ordering constraints."""

    def test_increasing(self):
        """Test Increasing constraint."""
        x = VarArray(size=4, dom=range(1, 10))
        satisfy(Increasing(x))

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        for i in range(len(sol) - 1):
            assert sol[i] <= sol[i + 1]

    def test_decreasing(self):
        """Test Decreasing constraint."""
        x = VarArray(size=4, dom=range(1, 10))
        satisfy(Decreasing(x))

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        for i in range(len(sol) - 1):
            assert sol[i] >= sol[i + 1]

    def test_strictly_increasing(self):
        """Test Increasing with strict=True constraint."""
        x = VarArray(size=4, dom=range(1, 10))
        satisfy(Increasing(x, strict=True))

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        for i in range(len(sol) - 1):
            assert sol[i] < sol[i + 1]


class TestChannelConstraint:
    """Tests for channel constraint."""

    def test_channel_basic(self):
        """Test basic channel/inverse constraint."""
        x = VarArray(size=4, dom=range(4))
        y = VarArray(size=4, dom=range(4))
        satisfy(Channel(x, y))

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol_x = values(x)
        sol_y = values(y)

        # Verify inverse relationship: x[y[i]] = i and y[x[i]] = i
        for i in range(4):
            assert sol_x[sol_y[i]] == i
            assert sol_y[sol_x[i]] == i


class TestAllEqual:
    """Tests for AllEqual constraint."""

    def test_all_equal_basic(self):
        """Test basic AllEqual constraint."""
        x = VarArray(size=4, dom=range(1, 10))
        satisfy(AllEqual(x))

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert len(set(sol)) == 1


class TestIntensionConstraints:
    """Tests for intension constraints (expressions)."""

    def test_arithmetic_expression(self):
        """Test arithmetic expression constraint."""
        x = Var(dom=range(1, 10))
        y = Var(dom=range(1, 10))
        satisfy(x * 2 + y == 15)

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        vx, vy = value(x), value(y)
        assert vx * 2 + vy == 15

    def test_abs_expression(self):
        """Test absolute value in constraint."""
        x = Var(dom=range(-5, 6))
        y = Var(dom=range(0, 10))
        satisfy(abs(x) == y, y == 3)

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        vx, vy = value(x), value(y)
        assert abs(vx) == vy == 3

    def test_logical_and(self):
        """Test logical AND constraint."""
        x = Var(dom=range(1, 10))
        y = Var(dom=range(1, 10))
        satisfy((x > 5) & (y < 5))

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        vx, vy = value(x), value(y)
        assert vx > 5 and vy < 5

    def test_logical_or(self):
        """Test logical OR constraint."""
        x = Var(dom=range(1, 10))
        satisfy((x == 1) | (x == 9))

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        vx = value(x)
        assert vx in [1, 9]


class TestNValuesConstraints:
    """Tests for NValues constraint."""

    def test_nvalues_equals(self):
        """Test NValues constraint with equality."""
        x = VarArray(size=5, dom=range(1, 6))
        satisfy(NValues(x) == 3)

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert len(set(sol)) == 3


class TestCircuitConstraint:
    """Tests for circuit constraint."""

    def test_circuit_basic(self):
        """Test basic circuit constraint."""
        x = VarArray(size=4, dom=range(4))
        satisfy(Circuit(x))

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        # Verify it's a valid circuit (Hamiltonian path)
        visited = set()
        current = 0
        for _ in range(4):
            assert current not in visited
            visited.add(current)
            current = sol[current]
        assert len(visited) == 4


class TestRegularConstraint:
    """Tests for regular constraint (automaton)."""

    def test_regular_basic(self):
        """Test basic Regular constraint with simple automaton."""
        # Automaton that accepts sequences ending with 1
        # States: a (start), b (seen 1, accepting)
        x = VarArray(size=5, dom={0, 1})

        a, b = "a", "b"
        transitions = [
            (a, 0, a),  # Stay in a on 0
            (a, 1, b),  # Go to b on 1
            (b, 0, a),  # Back to a on 0
            (b, 1, b),  # Stay in b on 1
        ]
        A = Automaton(start=a, final=b, transitions=transitions)
        satisfy(x in A)

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sol[-1] == 1, "Sequence should end with 1"

    def test_regular_no_consecutive_ones(self):
        """Test Regular constraint: no two consecutive 1s."""
        x = VarArray(size=6, dom={0, 1})

        # States: a (start/after 0), b (after 1)
        a, b = "a", "b"
        transitions = [
            (a, 0, a),  # Stay in a on 0
            (a, 1, b),  # Go to b on 1
            (b, 0, a),  # Back to a on 0 (no transition on 1 from b)
        ]
        A = Automaton(start=a, final=[a, b], transitions=transitions)
        satisfy(x in A)

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        for i in range(len(sol) - 1):
            assert not (sol[i] == 1 and sol[i + 1] == 1), "No consecutive 1s allowed"


class TestCumulativeConstraint:
    """Tests for cumulative constraint."""

    def test_cumulative_basic(self):
        """Test basic cumulative constraint."""
        # 3 tasks with durations [2, 3, 2] and heights [1, 1, 1], capacity 2
        starts = VarArray(size=3, dom=range(10))
        satisfy(
            Cumulative(
                origins=starts,
                lengths=[2, 3, 2],
                heights=[1, 1, 1]
            ) <= 2
        )

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)


class TestLexConstraint:
    """Tests for lexicographic ordering."""

    def test_lex_less(self):
        """Test LexIncreasing constraint."""
        x = VarArray(size=3, dom=range(1, 5))
        y = VarArray(size=3, dom=range(1, 5))
        satisfy(LexIncreasing(x, y))

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol_x = values(x)
        sol_y = values(y)
        # Check lexicographic ordering
        assert sol_x <= sol_y


class TestKnapsackConstraint:
    """Tests for knapsack constraint."""

    def test_knapsack_basic(self):
        """Test basic knapsack constraint."""
        # 5 items with binary selection
        x = VarArray(size=5, dom=range(2))
        weights = [2, 3, 4, 5, 6]
        profits = [3, 4, 5, 6, 7]
        capacity = 10

        satisfy(Knapsack(x, weights=weights, wlimit=capacity, profits=profits) >= 10)

        status = solve(solver="minizinc")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        total_weight = sum(w * s for w, s in zip(weights, sol))
        total_profit = sum(p * s for p, s in zip(profits, sol))

        assert total_weight <= capacity, f"Weight {total_weight} exceeds capacity {capacity}"
        assert total_profit >= 10, f"Profit {total_profit} < 10"
