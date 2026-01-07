"""
N-Queens Problem - Solver Comparison

The N-Queens problem asks to place n queens on an n×n chessboard
so that no two queens attack each other.

This example compares the performance of different solvers:
- ortools: Google OR-Tools CP-SAT
- cpo: IBM DOcplex CP Optimizer
- ace: ACE solver (native pycsp3)
- choco: Choco solver (native pycsp3)

Based on: https://github.com/xcsp3team/pycsp3-models/blob/master/academic/Queens/Queens.py
"""

import json
import subprocess
import sys


def solve_queens_subprocess(n: int, solver: str, verbose: int = 0) -> dict:
    """
    Solve N-Queens in a subprocess for clean state isolation.
    """
    import os
    import tempfile
    from pathlib import Path

    parent_dir = Path(__file__).parent.parent

    script = f'''import sys
import time
import json
sys.argv = ['queens.py']  # Override argv before pycsp3 import
sys.path.insert(0, r"{parent_dir}")
from pycsp3 import *
from pycsp3_solvers_extra import solve

n = {n}
solver = "{solver}"

q = VarArray(size=n, dom=range(n))
satisfy(
    AllDifferent(q),
    AllDifferent([q[i] + i for i in range(n)]),
    AllDifferent([q[i] - i for i in range(n)])
)

start = time.time()
try:
    status = solve(solver=solver, verbose={verbose})
    elapsed = time.time() - start
    if status in (SAT, OPTIMUM):
        solution = values(q)
    else:
        solution = None
    result = {{"status": str(status), "solution": solution, "time": elapsed}}
except Exception as e:
    import traceback
    elapsed = time.time() - start
    result = {{"status": f"ERROR: {{e}}", "solution": None, "time": elapsed, "tb": traceback.format_exc()}}

print("RESULT:" + json.dumps(result))
'''
    # Write to temp file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=120
        )
        # Parse result from stdout
        for line in result.stdout.splitlines():
            if line.startswith("RESULT:"):
                data = json.loads(line[7:])
                if "tb" in data and "ERROR" in data["status"]:
                    print(f"  Traceback: {data['tb'][:300]}")
                return data
        # If no result found, return error with more details
        error = result.stderr.strip() if result.stderr else result.stdout.strip() or "No output"
        if result.stderr:
            print(f"  STDERR: {result.stderr[:300]}")
        if result.stdout and "RESULT:" not in result.stdout:
            print(f"  STDOUT: {result.stdout[:300]}")
        return {"status": f"ERROR: {error[:100]}", "solution": None, "time": 0}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "solution": None, "time": 120}
    except Exception as e:
        return {"status": f"ERROR: {e}", "solution": None, "time": 0}
    finally:
        os.unlink(tmp_path)


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
                        help="Solvers to compare (default: ortools ace choco)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"N-Queens Problem (n={args.n})")
    print(f"{'='*60}\n")

    results = {}
    for solver in args.solvers:
        print(f"Solving with {solver}...")
        result = solve_queens_subprocess(args.n, solver, args.verbose)
        results[solver] = result
        print(f"  Status: {result['status']}")
        print(f"  Time:   {result['time']:.4f}s")
        if result['solution']:
            print(f"  Solution: {result['solution']}")
        print()

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
