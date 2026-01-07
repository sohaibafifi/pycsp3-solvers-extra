"""
pycsp3-solvers-extra: Extra solver backends for pycsp3.

This package provides additional solver backends (OR-Tools CP-SAT, IBM CPO)
for pycsp3 models using a callback-based XCSP3 parsing approach.

Example usage:
    from pycsp3 import *
    from pycsp3_solvers_extra import solve

    # Build model with pycsp3
    x = VarArray(size=8, dom=range(8))
    satisfy(AllDifferent(x))

    # Solve with OR-Tools instead of ACE/Choco
    status = solve(solver="ortools")
    if status == SAT:
        print(values(x))
"""

from pycsp3_solvers_extra.solver import solve, supported_solvers
from pycsp3_solvers_extra.backends import get_backend

__version__ = "0.1.0"
__all__ = ["solve", "supported_solvers", "get_backend"]
