"""
SEND + MORE = MONEY - Solver Comparison

Classic cryptarithmetic puzzle where each letter represents a unique digit.

    S E N D
  + M O R E
  ---------
  M O N E Y

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


def solve_send_more_money(solver: str, verbose: int = 0) -> dict:
    """
    Solve SEND+MORE=MONEY puzzle with specified solver.
    """
    parent_dir = Path(__file__).parent.parent

    script = f'''import sys
import time
import json
sys.argv = ['send_more_money.py']
sys.path.insert(0, r"{parent_dir}")
from pycsp3 import *
from pycsp3_solvers_extra import solve

# Variables for each letter (use v_ prefix to avoid ACE reserved words like E)
S, E, N, D, M, O, R, Y = Var(id="v_S", dom=range(1, 10)), Var(id="v_E", dom=range(0, 10)), Var(id="v_N", dom=range(0, 10)), Var(id="v_D", dom=range(0, 10)), Var(id="v_M", dom=range(1, 10)), Var(id="v_O", dom=range(0, 10)), Var(id="v_R", dom=range(0, 10)), Var(id="v_Y", dom=range(0, 10))

letters = [S, E, N, D, M, O, R, Y]

satisfy(
    AllDifferent(letters),
    1000 * S + 100 * E + 10 * N + D +
    1000 * M + 100 * O + 10 * R + E ==
    10000 * M + 1000 * O + 100 * N + 10 * E + Y
)

start = time.time()
try:
    status = solve(solver="{solver}", verbose={verbose})
    elapsed = time.time() - start
    if status in (SAT, OPTIMUM):
        solution = {{
            'S': value(S), 'E': value(E), 'N': value(N), 'D': value(D),
            'M': value(M), 'O': value(O), 'R': value(R), 'Y': value(Y)
        }}
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
    parser.add_argument("--solvers", nargs="+", default=["ortools", "ace", "choco"],
                        help="Solvers to compare (default: ortools ace choco)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("SEND + MORE = MONEY")
    print(f"{'='*60}\n")

    results = {}
    for solver in args.solvers:
        print(f"Solving with {solver}...")
        result = solve_send_more_money(solver, args.verbose)
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

    # Show solution
    for solver, result in results.items():
        if result['solution']:
            print(f"\nSolution from {solver}:")
            print_solution(result['solution'])
            break


if __name__ == "__main__":
    main()
