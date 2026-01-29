"""Constraint coverage tests for OR-Tools backend.

These tests are skipped if OR-Tools is not available.
"""

import pytest
from pycsp3 import *
from pycsp3.functions import Table
from pycsp3_solvers_extra import solve
from pycsp3_solvers_extra.backends import get_backend

ORTOOLS_AVAILABLE = get_backend("ortools") is not None

pytestmark = pytest.mark.skipif(
    not ORTOOLS_AVAILABLE,
    reason="OR-Tools backend not available"
)


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
        assert bound() == 6


    def test_maximize_sum(self):
        """Test maximizing sum."""
        x = VarArray(size=3, dom=range(1, 10))
        satisfy(AllDifferent(x))
        maximize(Sum(x))

        status = solve(solver="ortools")
        assert status == OPTIMUM

        sol = values(x)
        assert sum(sol) == 24  # 7 + 8 + 9
        assert bound() == 24

    def test_maximize_weighted_comparisons(self):
        """Test maximizing a weighted sum of comparisons."""
        x = VarArray(size=3, dom=range(3))
        satisfy(AllDifferent(x))
        weights = [5, 3, 2]
        maximize(Sum((x[i] != i) * weights[i] for i in range(3)))

        status = solve(solver="ortools")
        assert status == OPTIMUM

        sol = values(x)
        assert all(sol[i] != i for i in range(3))
        assert bound() == sum(weights)


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

    def test_among_via_count(self):
        """Test Among decomposition via Count with multiple values."""
        x = VarArray(size=5, dom=range(1, 5))
        satisfy(Count(x, values=[1, 2]) == 3)

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sum(1 for v in sol if v in (1, 2)) == 3

    def test_cardinality_closed(self):
        """Test Cardinality decomposition with closed values."""
        x = VarArray(size=5, dom=range(1, 5))
        satisfy(Cardinality(x, occurrences={1: 2, 2: 3}, closed=True))

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sol.count(1) == 2
        assert sol.count(2) == 3
        assert set(sol) <= {1, 2}

    def test_cardinality_range(self):
        """Test Cardinality decomposition with range occurrences."""
        x = VarArray(size=4, dom=range(1, 4))
        satisfy(Cardinality(x, occurrences={1: range(1, 3)}))

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        count_ones = sol.count(1)
        assert 1 <= count_ones <= 2


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

    def test_element_matrix(self):
        """Test Element with matrix indices (VarArray[row, col])."""
        m = VarArray(size=[2, 3], dom=range(1, 7))
        i = Var(dom=range(2))
        j = Var(dom=range(3))
        satisfy(m[i, j] == 5)

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        row = value(i)
        col = value(j)
        assert value(m[row][col]) == 5


class TestNoOverlapConstraints:
    """Tests for NoOverlap constraints."""

    def test_nooverlap_2d_constant_lengths(self):
        """Test 2D NoOverlap with constant rectangle sizes."""
        x = VarArray(size=2, dom=range(5))
        y = VarArray(size=2, dom=range(5))
        origins = list(zip(x, y))
        lengths = [(2, 2), (2, 2)]

        satisfy(NoOverlap(origins=origins, lengths=lengths))

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol_x = values(x)
        sol_y = values(y)
        assert (
            sol_x[0] + 2 <= sol_x[1]
            or sol_x[1] + 2 <= sol_x[0]
            or sol_y[0] + 2 <= sol_y[1]
            or sol_y[1] + 2 <= sol_y[0]
        ), "Rectangles overlap"

    def test_nooverlap_zero_ignored_variable_length(self):
        """Test zero_ignored with variable length in 2D NoOverlap."""
        x = VarArray(size=2, dom=range(1))  # fixed at 0
        y = VarArray(size=2, dom=range(1))  # fixed at 0
        w1 = Var(dom=[0, 2])
        origins = list(zip(x, y))
        lengths = [(2, 2), (w1, 2)]

        satisfy(NoOverlap(origins=origins, lengths=lengths, zero_ignored=True))

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)
        assert value(w1) == 0


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

    def test_not_all_equal(self):
        """Test NotAllEqual constraint (at least 2 distinct values)."""
        x = VarArray(size=4, dom=range(1, 5))
        satisfy(NotAllEqual(x))

        status = solve(solver="ortools")
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


class TestKnapsackConstraint:
    """Tests for Knapsack constraint."""

    def test_knapsack_basic(self):
        """Test basic Knapsack constraint with weight limit and profit condition."""
        # 5 items with binary selection
        x = VarArray(size=5, dom=range(2))
        weights = [2, 3, 4, 5, 6]
        profits = [3, 4, 5, 6, 7]
        capacity = 10

        satisfy(Knapsack(x, weights=weights, wlimit=capacity, profits=profits) >= 10)

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        total_weight = sum(w * s for w, s in zip(weights, sol))
        total_profit = sum(p * s for p, s in zip(profits, sol))

        assert total_weight <= capacity, f"Weight {total_weight} exceeds capacity {capacity}"
        assert total_profit >= 10, f"Profit {total_profit} < 10"

    def test_knapsack_exact_weight(self):
        """Test Knapsack with exact weight condition."""
        x = VarArray(size=4, dom=range(2))
        weights = [1, 2, 3, 4]
        profits = [1, 2, 3, 4]

        satisfy(Knapsack(x, weights=weights, wcondition=eq(5), profits=profits) >= 1)

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        total_weight = sum(w * s for w, s in zip(weights, sol))
        assert total_weight == 5

    def test_knapsack_maximize_profit(self):
        """Test Knapsack with profit maximization."""
        x = VarArray(size=4, dom=range(2))
        weights = [2, 3, 4, 5]
        profits = [3, 4, 5, 6]
        capacity = 7

        # Use a profit variable and maximize it
        profit = Var(dom=range(100))
        satisfy(
            Knapsack(x, weights=weights, wlimit=capacity, profits=profits) == profit
        )
        maximize(profit)

        status = solve(solver="ortools")
        assert status == OPTIMUM

        sol = values(x)
        total_weight = sum(w * s for w, s in zip(weights, sol))
        total_profit = sum(p * s for p, s in zip(profits, sol))

        assert total_weight <= capacity
        # Optimal: select items 0+3 or 1+2 (weight=7, profit=9)
        assert total_profit == 9

    def test_knapsack_unsatisfiable(self):
        """Test Knapsack with impossible constraints."""
        x = VarArray(size=3, dom=range(2))
        weights = [5, 6, 7]
        profits = [1, 1, 1]

        # Weight limit is 4 but all items weigh more than 4
        # and we require profit >= 1 (need at least one item)
        satisfy(Knapsack(x, weights=weights, wlimit=4, profits=profits) >= 1)

        status = solve(solver="ortools")
        assert status == UNSAT

    def test_knapsack_multi_copy(self):
        """Test Knapsack with multiple copies of items."""
        # Allow up to 3 copies of each item
        x = VarArray(size=3, dom=range(4))
        weights = [2, 3, 4]
        profits = [3, 4, 5]
        capacity = 10

        satisfy(Knapsack(x, weights=weights, wlimit=capacity, profits=profits) >= 12)

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        total_weight = sum(w * s for w, s in zip(weights, sol))
        total_profit = sum(p * s for p, s in zip(profits, sol))

        assert total_weight <= capacity
        assert total_profit >= 12


class TestRegularConstraint:
    """Tests for Regular constraint (finite automaton)."""

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

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sol[-1] == 1, "Sequence should end with 1"

    def test_regular_binary_pattern(self):
        """Test Regular constraint matching binary pattern 11...0."""
        # Must start with two 1s and end with 0
        x = VarArray(size=4, dom={0, 1})

        a, b, c, d = "a", "b", "c", "d"
        transitions = [
            (a, 1, b),  # First 1
            (b, 1, c),  # Second 1
            (c, 0, d),  # Then 0 (accepting)
            (c, 1, c),  # More 1s allowed
            (d, 0, d),  # More 0s allowed
        ]
        A = Automaton(start=a, final=d, transitions=transitions)
        satisfy(x in A)

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sol[0] == 1 and sol[1] == 1, "Must start with 11"
        assert sol[-1] == 0, "Must end with 0"

    def test_regular_no_consecutive_ones(self):
        """Test Regular constraint: no two consecutive 1s."""
        x = VarArray(size=6, dom={0, 1})

        # States: a (start/after 0), b (after 1)
        a, b = "a", "b"
        transitions = [
            (a, 0, a),
            (a, 1, b),
            (b, 0, a),
            # No (b, 1, _) transition - reject consecutive 1s
        ]
        A = Automaton(start=a, final=[a, b], transitions=transitions)
        satisfy(x in A)

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        for i in range(len(sol) - 1):
            assert not (sol[i] == 1 and sol[i + 1] == 1), "No consecutive 1s allowed"

    def test_regular_unsatisfiable(self):
        """Test Regular constraint with impossible automaton."""
        x = VarArray(size=3, dom={0, 1})

        # Automaton requires value 2 which is not in domain
        a, b = "a", "b"
        transitions = [
            (a, 2, b),  # Need 2 to reach accepting state
        ]
        A = Automaton(start=a, final=b, transitions=transitions)
        satisfy(x in A)

        status = solve(solver="ortools")
        assert status == UNSAT


class TestMDDConstraint:
    """Tests for MDD constraint (decomposed to regular)."""

    @staticmethod
    def _exactly_one_one_mdd():
        # Layered MDD enforcing exactly one '1' in a length-3 binary sequence.
        transitions = [
            ("r", 0, "a"),
            ("r", 1, "b"),
            ("a", 0, "c"),
            ("a", 1, "d"),
            ("b", 0, "d"),
            ("b", 1, "e"),
            ("c", 1, "t"),
            ("d", 0, "t"),
            ("e", 2, "t"),  # Dead transition to keep a single terminal
        ]
        return MDD(transitions)

    def test_mdd_exactly_one_one_sat(self):
        x = VarArray(size=3, dom={0, 1})
        satisfy(x in self._exactly_one_one_mdd())

        status = solve(solver="ortools")
        assert status in (SAT, OPTIMUM)

        sol = values(x)
        assert sum(sol) == 1

    def test_mdd_exactly_one_one_unsat(self):
        x = VarArray(size=3, dom={0, 1})
        satisfy(x in self._exactly_one_one_mdd())
        satisfy(Sum(x) >= 2)

        status = solve(solver="ortools")
        assert status == UNSAT
