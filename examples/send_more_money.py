"""
SEND + MORE = MONEY - Solver Comparison

Classic cryptarithmetic puzzle where each letter represents a unique digit.

    S E N D
  + M O R E
  ---------
  M O N E Y
"""

import time
from pycsp3 import *
from pycsp3_solvers_extra import solve, supported_solvers


def print_solution(solution: dict) -> None:
    """Print the puzzle solution in a nice format."""
    if not solution:
        return

    send = 1000 * solution['S'] + 100 * solution['E'] + 10 * solution['N'] + solution['D']
    more = 1000 * solution['M'] + 100 * solution['O'] + 10 * solution['R'] + solution['E']
    money = 10000 * solution['M'] + 1000 * solution['O'] + 100 * solution['N'] + 10 * solution['E'] + solution['Y']

    print(f"    {send}")
    print(f"  + {more}")
    print(f"  -----")
    print(f"  {money}")
    print()
    print(f"Letter assignments: {solution}")


def main():
    """Run SEND+MORE=MONEY comparison across solvers."""
    import argparse

    parser = argparse.ArgumentParser(description="SEND+MORE=MONEY Solver Comparison")
    parser.add_argument("-v", "--verbose", type=int, default=0, help="Verbosity level")
    parser.add_argument("--solvers", nargs="+", default=supported_solvers(),
                        help="Solvers to compare")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("SEND + MORE = MONEY")
    print(f"{'='*60}\n")

    results = {}
    for solver in args.solvers:
        print(f"Solving with {solver}...")

        # Variables for each letter (use v_ prefix to avoid ACE reserved words like E)
        S, E, N, D = Var(id="v_S", dom=range(1, 10)), Var(id="v_E", dom=range(0, 10)), Var(id="v_N", dom=range(0, 10)), Var(id="v_D", dom=range(0, 10))
        M, O, R, Y = Var(id="v_M", dom=range(1, 10)), Var(id="v_O", dom=range(0, 10)), Var(id="v_R", dom=range(0, 10)), Var(id="v_Y", dom=range(0, 10))
        letters = [S, E, N, D, M, O, R, Y]

        satisfy(
            AllDifferent(letters),
            1000 * S + 100 * E + 10 * N + D +
            1000 * M + 100 * O + 10 * R + E ==
            10000 * M + 1000 * O + 100 * N + 10 * E + Y
        )

        start = time.time()
        try:
            status = solve(solver=solver, verbose=args.verbose)
            elapsed = time.time() - start
            if status in (SAT, OPTIMUM):
                solution = {
                    'S': value(S), 'E': value(E), 'N': value(N), 'D': value(D),
                    'M': value(M), 'O': value(O), 'R': value(R), 'Y': value(Y)
                }
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

    # Show solution
    for solver, result in results.items():
        if result['solution']:
            print(f"\nSolution from {solver}:")
            print_solution(result['solution'])
            break


if __name__ == "__main__":
    main()
