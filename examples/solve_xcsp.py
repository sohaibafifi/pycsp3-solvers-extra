import argparse

from pycsp3 import clear, solution
from pycsp3_solvers_extra import load, solve


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and solve an XCSP3 instance.")
    parser.add_argument("filepath", help="Path to .xml or .xml.lzma XCSP3 file")
    parser.add_argument("--solver", default="ortools", help="Solver backend (default: ortools)")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()

    clear()
    load(args.filepath)
    status = solve(solver=args.solver, verbose=args.verbose)
    print(status)
    print(solution())


if __name__ == "__main__":
    main()
