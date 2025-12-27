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

This example compares the performance of different solvers:
- ortools: Google OR-Tools CP-SAT
- cpo: IBM DOcplex CP Optimizer
- ace: ACE solver (native pycsp3)
- choco: Choco solver (native pycsp3)
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def solve_magic_sequence(n: int, solver: str, verbose: int = 0) -> dict:
    """
    Solve Magic Sequence problem with specified solver.
    """
    parent_dir = Path(__file__).parent.parent

    script = f'''import sys
import time
import json
sys.argv = ['magic_sequence.py']
sys.path.insert(0, r"{parent_dir}")
from pycsp3 import *
from pycsp3_solvers_extra import solve

n = {n}
x = VarArray(size=n, dom=range(n))

satisfy(
    [Count(x, value=i) == x[i] for i in range(n)],
    Sum(x) == n,
    Sum(i * x[i] for i in range(n)) == n
)

start = time.time()
try:
    status = solve(solver="{solver}", verbose={verbose})
    elapsed = time.time() - start
    if status in (SAT, OPTIMUM):
        solution = values(x)
    else:
        solution = None
    result = {{"status": str(status), "solution": solution, "time": elapsed}}
except Exception as e:
    import traceback
    elapsed = time.time() - start
    result = {{"status": f"ERROR: {{e}}", "solution": None, "time": elapsed, "tb": traceback.format_exc()}}

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
            timeout=120
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
        return {"status": f"ERROR: {error[:100]}", "solution": None, "time": 0}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "solution": None, "time": 120}
    except Exception as e:
        return {"status": f"ERROR: {e}", "solution": None, "time": 0}
    finally:
        os.unlink(tmp_path)


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
    parser.add_argument("--solvers", nargs="+", default=["ortools", "ace", "choco"],
                        help="Solvers to compare (default: ortools ace choco)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Magic Sequence Problem (n={args.n})")
    print(f"{'='*60}\n")

    results = {}
    for solver in args.solvers:
        print(f"Solving with {solver}...")
        result = solve_magic_sequence(args.n, solver, args.verbose)
        results[solver] = result
        print(f"  Status: {result['status']}")
        print(f"  Time:   {result['time']:.4f}s")
        if result['solution']:
            print(f"  Solution: {result['solution']}")
            print(f"  Valid:    {verify_solution(result['solution'])}")
        print()

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
