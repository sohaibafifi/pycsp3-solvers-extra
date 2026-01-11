import argparse

from pycsp3 import clear, bound
from pycsp3_solvers_extra import load, solve


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and solve an XCSP3 instance.")
    parser.add_argument("filepath", help="Path to .xml or .xml.lzma XCSP3 file")
    parser.add_argument("--solver", default="ortools", help="Solver backend (default: ortools)")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity level")
    parser.add_argument("--time-limit", type=float, default=None, help="Time limit in seconds")
    parser.add_argument("--options", default="", help="Solver-specific options (e.g. '--options=\"-positive=CT2\"')")
    args = parser.parse_args()

    clear()
    load(args.filepath)
    status = solve(
        solver=args.solver,
        verbose=args.verbose,
        time_limit=args.time_limit,
        options=args.options,
    )
    print(status)
    print(bound())


if __name__ == "__main__":
    main()
