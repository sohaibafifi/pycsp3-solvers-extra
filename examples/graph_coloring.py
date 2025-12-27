"""
Graph Coloring Problem - Solver Comparison

Given a graph, assign colors to vertices such that no two adjacent
vertices have the same color. The goal is to minimize the number
of colors used (chromatic number).

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


# Sample graphs
GRAPHS = {
    "petersen": {
        "vertices": 10,
        "edges": [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer pentagon
            (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),  # Inner pentagram
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
        ]
    },
    "wheel6": {
        "vertices": 7,
        "edges": [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),  # Outer cycle
            (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5)   # Hub connections
        ]
    },
    "k4": {
        "vertices": 4,
        "edges": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    },
    "bipartite": {
        "vertices": 6,
        "edges": [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (2, 4), (2, 5)]
    }
}


def solve_graph_coloring(graph_name: str, max_colors: int, solver: str,
                          time_limit: float = 60, verbose: int = 0) -> dict:
    """
    Solve Graph Coloring problem with specified solver.
    """
    parent_dir = Path(__file__).parent.parent
    graph = GRAPHS[graph_name]
    edges_str = str(graph["edges"])

    script = f'''import sys
import time
import json
sys.argv = ['graph_coloring.py']
sys.path.insert(0, r"{parent_dir}")
from pycsp3 import *
from pycsp3_solvers_extra import solve

n = {graph["vertices"]}
edges = {edges_str}
max_colors = {max_colors}

color = VarArray(size=n, dom=range(max_colors))

satisfy(
    [color[i] != color[j] for i, j in edges]
)

minimize(Maximum(color))

start = time.time()
try:
    status = solve(solver="{solver}", time_limit={time_limit}, verbose={verbose})
    elapsed = time.time() - start
    if status in (SAT, OPTIMUM):
        solution = values(color)
        num_colors = max(solution) + 1
    else:
        solution = None
        num_colors = None
    result = {{"status": str(status), "solution": solution, "num_colors": num_colors, "time": elapsed}}
except Exception as e:
    import traceback
    elapsed = time.time() - start
    result = {{"status": f"ERROR: {{e}}", "solution": None, "num_colors": None, "time": elapsed, "tb": traceback.format_exc()}}

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
        return {"status": f"ERROR: {error[:100]}", "solution": None, "num_colors": None, "time": 0}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "solution": None, "num_colors": None, "time": time_limit}
    except Exception as e:
        return {"status": f"ERROR: {e}", "solution": None, "num_colors": None, "time": 0}
    finally:
        os.unlink(tmp_path)


def main():
    """Run Graph Coloring comparison across solvers."""
    import argparse

    parser = argparse.ArgumentParser(description="Graph Coloring Solver Comparison")
    parser.add_argument("-g", "--graph", default="petersen",
                        choices=list(GRAPHS.keys()),
                        help="Graph to color (default: petersen)")
    parser.add_argument("-c", "--max-colors", type=int, default=10,
                        help="Maximum colors to try (default: 10)")
    parser.add_argument("-t", "--time-limit", type=float, default=60,
                        help="Time limit in seconds (default: 60)")
    parser.add_argument("-v", "--verbose", type=int, default=0, help="Verbosity level")
    parser.add_argument("--solvers", nargs="+", default=["ortools", "ace", "choco"],
                        help="Solvers to compare (default: ortools ace choco)")
    args = parser.parse_args()

    graph = GRAPHS[args.graph]
    print(f"\n{'='*60}")
    print(f"Graph Coloring: {args.graph}")
    print(f"Vertices: {graph['vertices']}, Edges: {len(graph['edges'])}")
    print(f"{'='*60}\n")

    # Known chromatic numbers
    chromatic = {"petersen": 3, "wheel6": 4, "k4": 4, "bipartite": 2}
    if args.graph in chromatic:
        print(f"Known chromatic number: {chromatic[args.graph]}\n")

    results = {}
    for solver in args.solvers:
        print(f"Solving with {solver}...")
        result = solve_graph_coloring(args.graph, args.max_colors, solver,
                                       args.time_limit, args.verbose)
        results[solver] = result
        print(f"  Status:     {result['status']}")
        print(f"  Time:       {result['time']:.4f}s")
        if result['solution']:
            print(f"  Colors:     {result['num_colors']}")
            print(f"  Coloring:   {result['solution']}")
        print()

    # Print comparison table
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print(f"{'='*60}")
    print(f"{'Solver':<12} {'Status':<12} {'Colors':<10} {'Time (s)':<12}")
    print("-" * 46)
    for solver, result in results.items():
        status = result['status'][:10] if len(result['status']) > 10 else result['status']
        colors = str(result['num_colors']) if result['num_colors'] else "-"
        print(f"{solver:<12} {status:<12} {colors:<10} {result['time']:<12.4f}")


if __name__ == "__main__":
    main()
