import argparse

from pycsp3 import *
from pycsp3_solvers_extra import solve


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental solving example.")
    parser.add_argument("--solver", default="z3", help="Solver backend (default: z3)")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity level")
    args = parser.parse_args()

    x = VarArray(size=4, dom=range(7))

    satisfy(
        AllDifferent(x),
        Increasing(x),
        Sum(x) == 10
    )

    cnt = 0
    while solve(solver=args.solver, verbose=args.verbose, sols=ALL) in [SAT, OPTIMUM]:
        print(f"{n_solutions()} solutions found.")
        cnt += 1
        print(f"Solution {cnt}: {values(x)}")
        satisfy(x != values(x))


if __name__ == "__main__":
    main()
