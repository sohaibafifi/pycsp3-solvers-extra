"""
Magic Sequence Problem - Solver Comparison

A magic sequence of length n is a sequence x[0], x[1], ..., x[n-1]
of integers such that for each i in 0..n-1, x[i] is equal to the
number of occurrences of i in the sequence.

For example, for n=4: [1, 2, 1, 0] is a magic sequence because:
- 0 appears 1 time
- 1 appears 2 times
- 2 appears 1 time
- 3 appears 0 times
"""

import time
from pycsp3 import *
from pycsp3_solvers_extra import solve, supported_solvers


def verify_solution(solution: list[int]) -> bool:
    """Verify that a solution is a valid magic sequence."""
    if not solution:
        return False

    n = len(solution)
    for i in range(n):
        if solution.count(i) != solution[i]:
            return False
    return True


def main():
    """Run Magic Sequence comparison across solvers."""
    import argparse

    parser = argparse.ArgumentParser(description="Magic Sequence Solver Comparison")
    parser.add_argument("-n", type=int, default=10, help="Sequence length (default: 10)")
    parser.add_argument("-v", "--verbose", type=int, default=0, help="Verbosity level")
    parser.add_argument("--solvers", nargs="+", default=supported_solvers(),
                        help="Solvers to compare")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Magic Sequence Problem (n={args.n})")
    print(f"{'='*60}\n")

    results = {}
    for solver in args.solvers:
        print(f"Solving with {solver}...")

        n = args.n
        x = VarArray(size=n, dom=range(n))

        satisfy(
            [Count(x, value=i) == x[i] for i in range(n)],
            Sum(x) == n,
            Sum(i * x[i] for i in range(n)) == n
        )

        start = time.time()
        try:
            status = solve(solver=solver, verbose=args.verbose)
            elapsed = time.time() - start
            if status in (SAT, OPTIMUM):
                solution = values(x)
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
            print(f"  Valid:    {verify_solution(results[solver]['solution'])}")
        print()
        clear()

    # Print comparison table
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print(f"{'='*60}")
    print(f"{'Solver':<12} {'Status':<12} {'Time (s)':<12} {'Valid':<8}")
    print("-" * 44)
    for solver, result in results.items():
        status = result['status'][:10] if len(result['status']) > 10 else result['status']
        valid = "Yes" if verify_solution(result['solution']) else "No" if result['solution'] else "-"
        print(f"{solver:<12} {status:<12} {result['time']:<12.4f} {valid:<8}")

    # Show interpretation of solution
    for solver, result in results.items():
        if result['solution']:
            print(f"\nInterpretation of solution from {solver}:")
            for i, count in enumerate(result['solution']):
                if count > 0:
                    print(f"  Value {i} appears {count} time(s)")
            break


if __name__ == "__main__":
    main()
