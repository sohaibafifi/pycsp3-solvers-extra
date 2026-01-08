"""
N-Queens Problem - Solver Comparison

The N-Queens problem asks to place n queens on an n×n chessboard
so that no two queens attack each other.
"""

import time
from pycsp3 import *
from pycsp3_solvers_extra import solve


def print_board(solution: list[int]) -> None:
    """Print the chess board with queens."""
    n = len(solution)
    print("+" + "---+" * n)
    for row in range(n):
        line = "|"
        for col in range(n):
            if solution[row] == col:
                line += " Q |"
            else:
                line += "   |"
        print(line)
        print("+" + "---+" * n)


def main():
    """Run N-Queens comparison across solvers."""
    import argparse

    parser = argparse.ArgumentParser(description="N-Queens Solver Comparison")
    parser.add_argument("-n", type=int, default=8, help="Board size (default: 8)")
    parser.add_argument("-v", "--verbose", type=int, default=0, help="Verbosity level")
    parser.add_argument("--solvers", nargs="+", default=["ortools", "ace", "choco", "cpo"],
                        help="Solvers to compare")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"N-Queens Problem (n={args.n})")
    print(f"{'='*60}\n")

    results = {}
    for solver in args.solvers:
        print(f"Solving with {solver}...")

        q = VarArray(size=args.n, dom=range(args.n))
        satisfy(
            AllDifferent(q),
            AllDifferent([q[i] + i for i in range(args.n)]),
            AllDifferent([q[i] - i for i in range(args.n)])
        )

        start = time.time()
        try:
            status = solve(solver=solver, verbose=args.verbose)
            elapsed = time.time() - start
            if status in (SAT, OPTIMUM):
                solution = values(q)
            else:
                solution = None
            results[solver] = {"status": str(status), "solution": solution, "time": elapsed}
        except Exception as e:
            elapsed = time.time() - start
            results[solver] = {"status": f"ERROR: {e}", "solution": None, "time": elapsed}

        print(f"  Status: {results[solver]['status']}")
        print(f"  Time:   {results[solver]['time']:.4f}s")
        if results[solver]['solution']:
            print(f"  Solution: {results[solver]['solution']}")
        print()
        clear()

    # Print comparison table
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print(f"{'='*60}")
    print(f"{'Solver':<12} {'Status':<12} {'Time (s)':<12}")
    print("-" * 36)
    for solver, result in results.items():
        status = result['status'][:10] if len(result['status']) > 10 else result['status']
        print(f"{solver:<12} {status:<12} {result['time']:<12.4f}")

    # Show one solution board
    for solver, result in results.items():
        if result['solution']:
            print(f"\nSolution from {solver}:")
            print_board(result['solution'])
            break


if __name__ == "__main__":
    main()
