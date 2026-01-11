"""
Graph Coloring Problem - Solver Comparison

Given a graph, assign colors to vertices such that no two adjacent
vertices have the same color. The goal is to minimize the number
of colors used (chromatic number).
"""

import time
from pycsp3 import *
from pycsp3_solvers_extra import solve


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
    parser.add_argument("--solvers", nargs="+", default=["ortools", "ace", "choco", "cpo", "z3"],
                        help="Solvers to compare")
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

        n = graph["vertices"]
        edges = graph["edges"]
        color = VarArray(size=n, dom=range(args.max_colors))

        satisfy(
            [color[i] != color[j] for i, j in edges]
        )

        minimize(Maximum(color))

        start = time.time()
        try:
            status = solve(solver=solver, time_limit=args.time_limit, verbose=args.verbose)
            elapsed = time.time() - start
            if status in (SAT, OPTIMUM):
                solution = values(color)
                num_colors = max(solution) + 1
            else:
                solution = None
                num_colors = None
            results[solver] = {"status": str(status), "solution": solution, "num_colors": num_colors, "time": elapsed}
        except Exception as e:
            elapsed = time.time() - start
            results[solver] = {"status": f"ERROR: {e}", "solution": None, "num_colors": None, "time": elapsed}

        print(f"  Status:     {results[solver]['status']}")
        print(f"  Time:       {results[solver]['time']:.4f}s")
        if results[solver]['solution']:
            print(f"  Colors:     {results[solver]['num_colors']}")
            print(f"  Coloring:   {results[solver]['solution']}")
        print()
        clear()

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
