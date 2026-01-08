"""
Golomb Ruler Problem - Solver Comparison

A Golomb ruler is a set of marks at integer positions along a ruler
such that no two pairs of marks are the same distance apart.
The objective is to find a ruler with a given number of marks
that has the minimum length.
"""

import time
from itertools import combinations
from pycsp3 import *
from pycsp3_solvers_extra import solve


def print_ruler(solution: list[int]) -> None:
    """Print a visual representation of the Golomb ruler."""
    if not solution:
        return

    length = max(solution)
    marks = set(solution)

    # Print scale
    print("Position:", end=" ")
    for i in range(length + 1):
        if i in marks:
            print(f"{i:3}", end="")
        else:
            print("   ", end="")
    print()

    # Print ruler
    print("Ruler:   ", end=" ")
    for i in range(length + 1):
        if i in marks:
            print("  |", end="")
        elif i == 0 or i == length:
            print("  +", end="")
        else:
            print("---", end="")
    print()

    # Print distances
    n = len(solution)
    distances = sorted(abs(solution[i] - solution[j])
                       for i in range(n) for j in range(i+1, n))
    print(f"Distances: {distances}")


def main():
    """Run Golomb Ruler comparison across solvers."""
    import argparse

    parser = argparse.ArgumentParser(description="Golomb Ruler Solver Comparison")
    parser.add_argument("-n", type=int, default=7, help="Number of marks (default: 7)")
    parser.add_argument("-t", "--time-limit", type=float, default=60,
                        help="Time limit in seconds (default: 60)")
    parser.add_argument("-v", "--verbose", type=int, default=0, help="Verbosity level")
    parser.add_argument("--solvers", nargs="+", default=["ortools", "ace", "choco", "cpo"],
                        help="Solvers to compare")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Golomb Ruler Problem (n={args.n} marks)")
    print(f"{'='*60}\n")

    # Known optimal lengths for reference
    optimal = {4: 6, 5: 11, 6: 17, 7: 25, 8: 34, 9: 44, 10: 55, 11: 72}
    if args.n in optimal:
        print(f"Known optimal length: {optimal[args.n]}\n")

    results = {}
    for solver in args.solvers:
        print(f"Solving with {solver}...")

        n = args.n
        ub = n * n + 1
        x = VarArray(size=n, dom=range(ub))

        satisfy(
            x[0] == 0,
            Increasing(x, strict=True),
            AllDifferent(abs(x[i] - x[j]) for i, j in combinations(n, 2))
        )

        minimize(Maximum(x))

        start = time.time()
        try:
            status = solve(solver=solver, time_limit=args.time_limit, verbose=args.verbose)
            elapsed = time.time() - start
            if status in (SAT, OPTIMUM):
                solution = values(x)
                length = max(solution)
            else:
                solution = None
                length = None
            results[solver] = {"status": str(status), "solution": solution, "length": length, "time": elapsed}
        except Exception as e:
            elapsed = time.time() - start
            results[solver] = {"status": f"ERROR: {e}", "solution": None, "length": None, "time": elapsed}

        print(f"  Status: {results[solver]['status']}")
        print(f"  Time:   {results[solver]['time']:.4f}s")
        if results[solver]['solution']:
            print(f"  Length: {results[solver]['length']}")
            print(f"  Marks:  {results[solver]['solution']}")
        print()
        clear()

    # Print comparison table
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print(f"{'='*60}")
    print(f"{'Solver':<12} {'Status':<12} {'Length':<10} {'Time (s)':<12}")
    print("-" * 46)
    for solver, result in results.items():
        status = result['status'][:10] if len(result['status']) > 10 else result['status']
        length = str(result['length']) if result['length'] else "-"
        print(f"{solver:<12} {status:<12} {length:<10} {result['time']:<12.4f}")

    # Show one solution
    for solver, result in results.items():
        if result['solution']:
            print(f"\nRuler from {solver}:")
            print_ruler(result['solution'])
            break


if __name__ == "__main__":
    main()
