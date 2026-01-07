"""
Golomb Ruler Problem - Solver Comparison

A Golomb ruler is a set of marks at integer positions along a ruler
such that no two pairs of marks are the same distance apart.
The objective is to find a ruler with a given number of marks
that has the minimum length.

This example compares the performance of different solvers:
- ortools: Google OR-Tools CP-SAT
- cpo: IBM DOcplex CP Optimizer
- ace: ACE solver (native pycsp3)
- choco: Choco solver (native pycsp3)

Based on: https://github.com/xcsp3team/pycsp3-models/blob/master/academic/GolombRuler/GolombRuler.py
"""

import json
import os
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path


def solve_golomb(n: int, solver: str, time_limit: float = 60, verbose: int = 0) -> dict:
    """
    Solve Golomb Ruler problem with specified solver.
    """
    parent_dir = Path(__file__).parent.parent

    script = f'''import sys
import time
import json
sys.argv = ['golomb_ruler.py']
sys.path.insert(0, r"{parent_dir}")
from pycsp3 import *
from pycsp3_solvers_extra import solve

n = {n}
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
    status = solve(solver="{solver}", time_limit={time_limit}, verbose={verbose})
    elapsed = time.time() - start
    if status in (SAT, OPTIMUM):
        solution = values(x)
        length = max(solution)
    else:
        solution = None
        length = None
    result = {{"status": str(status), "solution": solution, "length": length, "time": elapsed}}
except Exception as e:
    import traceback
    elapsed = time.time() - start
    result = {{"status": f"ERROR: {{e}}", "solution": None, "length": None, "time": elapsed, "tb": traceback.format_exc()}}

print("RESULT:" + json.dumps(result))
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=time_limit + 30
        )
        for line in result.stdout.splitlines():
            if line.startswith("RESULT:"):
                data = json.loads(line[7:])
                if "tb" in data and "ERROR" in data["status"]:
                    print(f"  Traceback: {data['tb'][:300]}")
                return data
        error = result.stderr.strip() if result.stderr else result.stdout.strip() or "No output"
        if result.stderr:
            print(f"  STDERR: {result.stderr[:300]}")
        return {"status": f"ERROR: {error[:100]}", "solution": None, "length": None, "time": 0}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "solution": None, "length": None, "time": time_limit}
    except Exception as e:
        return {"status": f"ERROR: {e}", "solution": None, "length": None, "time": 0}
    finally:
        os.unlink(tmp_path)


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
    parser.add_argument("--solvers", nargs="+", default=["ortools", "ace", "choco"],
                        help="Solvers to compare (default: ortools ace choco)")
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
        result = solve_golomb(args.n, solver, args.time_limit, args.verbose)
        results[solver] = result
        print(f"  Status: {result['status']}")
        print(f"  Time:   {result['time']:.4f}s")
        if result['solution']:
            print(f"  Length: {result['length']}")
            print(f"  Marks:  {result['solution']}")
        print()

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
