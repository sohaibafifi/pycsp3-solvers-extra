#!/usr/bin/env python
"""
Run all pycsp3-solvers-extra examples.

This script runs each example with default parameters and
summarizes the results.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_example(script: Path, args: list[str] = None) -> dict:
    """Run an example script and capture output."""
    cmd = [sys.executable, str(script)] + (args or [])
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        elapsed = time.time() - start
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr,
            "time": elapsed
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": "Timeout (120s)",
            "time": 120
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "time": time.time() - start
        }


def main():
    """Run all examples."""
    examples_dir = Path(__file__).parent

    # List of examples with their arguments
    examples = [
        ("queens.py", ["-n", "8"]),
        ("send_more_money.py", []),
        ("magic_sequence.py", ["-n", "10"]),
        ("graph_coloring.py", ["-g", "petersen"]),
        ("golomb_ruler.py", ["-n", "6", "-t", "30"]),
    ]

    print("=" * 70)
    print("Running pycsp3-solvers-extra Examples")
    print("=" * 70)
    print()

    results = []
    for script_name, args in examples:
        script = examples_dir / script_name
        if not script.exists():
            print(f"[SKIP] {script_name}: File not found")
            continue

        print(f"[RUN]  {script_name} {' '.join(args)}")
        result = run_example(script, args)
        results.append((script_name, result))

        if result["success"]:
            print(f"[OK]   Completed in {result['time']:.2f}s")
        else:
            print(f"[FAIL] {result['error'][:50]}")
        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Example':<25} {'Status':<10} {'Time (s)':<10}")
    print("-" * 45)

    passed = 0
    failed = 0
    for name, result in results:
        status = "PASS" if result["success"] else "FAIL"
        if result["success"]:
            passed += 1
        else:
            failed += 1
        print(f"{name:<25} {status:<10} {result['time']:<10.2f}")

    print("-" * 45)
    print(f"Total: {passed} passed, {failed} failed")

    # Show any errors
    if failed > 0:
        print("\nErrors:")
        for name, result in results:
            if not result["success"]:
                print(f"\n{name}:")
                print(f"  {result['error'][:200]}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
