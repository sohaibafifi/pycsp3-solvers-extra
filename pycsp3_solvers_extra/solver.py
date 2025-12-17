"""
Main solve() function for pycsp3-solvers-extra.

This module provides the entry point for solving pycsp3 models
with alternative solver backends (OR-Tools, CPO).
"""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

from pycsp3 import compile as pycsp3_compile
from pycsp3 import solve as pycsp3_solve
from pycsp3.classes.auxiliary.enums import TypeStatus

from pycsp3_solvers_extra.backends import get_backend

if TYPE_CHECKING:
    from pycsp3.classes.main.variables import Variable

# Solvers that should be delegated to pycsp3's native solve()
NATIVE_SOLVERS = {"ace", "choco"}

# Solvers implemented by this package
EXTRA_SOLVERS = {"ortools", "cpo"}


def supported_solvers() -> list[str]:
    """Return list of all supported solver names."""
    return sorted(NATIVE_SOLVERS | EXTRA_SOLVERS)


def solve(
    *,
    solver: str = "ortools",
    filename: str | None = None,
    time_limit: float | None = None,
    sols: int | str | None = None,
    verbose: int = 0,
    options: str = "",
) -> TypeStatus:
    """
    Solve the current pycsp3 model with the specified solver.

    This function shadows pycsp3.solve() to provide additional solver backends.
    For "ace" and "choco", it delegates to the original pycsp3.solve().
    For "ortools" and "cpo", it uses the callback-based XCSP3 translation.

    Args:
        solver: Solver name - "ortools", "cpo", "ace", or "choco"
        filename: Path to XCSP3 file. If None, compiles current model.
        time_limit: Time limit in seconds (None for no limit)
        sols: Number of solutions to find. Use "all" or pycsp3.ALL for all solutions.
        verbose: Verbosity level (0=quiet, 1=normal, 2=detailed)
        options: Solver-specific options string

    Returns:
        TypeStatus: SAT, UNSAT, OPTIMUM, or UNKNOWN

    Example:
        from pycsp3 import *
        from pycsp3_solvers_extra import solve

        x = VarArray(size=8, dom=range(8))
        satisfy(AllDifferent(x))

        status = solve(solver="ortools", time_limit=60)
        if status == SAT:
            print(values(x))
    """
    solver_lower = solver.lower()

    # Delegate native solvers to pycsp3
    if solver_lower in NATIVE_SOLVERS:
        return _solve_native(solver_lower, filename, time_limit, sols, verbose, options)

    # Handle extra solvers
    if solver_lower not in EXTRA_SOLVERS:
        raise ValueError(
            f"Unknown solver: {solver}. Supported solvers: {supported_solvers()}"
        )

    return _solve_extra(solver_lower, filename, time_limit, sols, verbose, options)


def _solve_native(
    solver: str,
    filename: str | None,
    time_limit: float | None,
    sols: int | str | None,
    verbose: int,
    options: str,
) -> TypeStatus:
    """Delegate to pycsp3's native solve() for ACE/Choco."""
    from pycsp3 import ACE, CHOCO, ALL

    solver_type = ACE if solver == "ace" else CHOCO

    # Build solver string with options
    solver_str = f"[{solver}"
    if verbose > 0:
        solver_str += "," + "v" * min(verbose, 3)
    if time_limit is not None:
        solver_str += f",t={int(time_limit)}s"
    if sols is not None:
        if sols == "all" or sols == ALL:
            solver_str += ",limit=no"
        elif isinstance(sols, int):
            solver_str += f",limit={sols}sols"
    solver_str += "]"

    return pycsp3_solve(
        solver=solver_str,
        options=options,
        filename=filename,
        verbose=verbose,
    )


def _solve_extra(
    solver: str,
    filename: str | None,
    time_limit: float | None,
    sols: int | str | None,
    verbose: int,
    options: str,
) -> TypeStatus:
    """Solve with extra backends (OR-Tools, CPO) using XCSP3 parsing."""
    # Get the appropriate backend
    backend_class = get_backend(solver)
    if backend_class is None:
        raise ImportError(
            f"Backend '{solver}' is not available. "
            f"Install the required package: pip install pycsp3-solvers-extra[{solver}]"
        )

    # Compile to XCSP3 if no filename provided
    if filename is None:
        # Use a temporary file for the XCSP3 instance
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False
        ) as tmp:
            tmp_filename = tmp.name

        try:
            result = pycsp3_compile(tmp_filename, verbose=verbose)
            # compile() returns (filename, is_cop) tuple
            if result is None:
                if verbose > 0:
                    print("Error: Failed to compile model")
                return TypeStatus.UNKNOWN
            xcsp3_file = result[0] if isinstance(result, tuple) else tmp_filename
        except Exception as e:
            if verbose > 0:
                print(f"Error during compilation: {e}")
            return TypeStatus.UNKNOWN
    else:
        xcsp3_file = filename
        tmp_filename = None

    try:
        # Create backend callbacks instance
        callbacks = backend_class(
            time_limit=time_limit,
            sols=sols,
            verbose=verbose,
            options=options,
        )

        # Parse XCSP3 and build solver model
        from pycsp3.parser.xparser import ParserXCSP3, CallbackerXCSP3

        if verbose > 0:
            print(f"Parsing XCSP3 file: {xcsp3_file}")

        # Parse the XML file
        parser = ParserXCSP3(xcsp3_file)

        # Create callbacker to invoke callbacks
        callbacker = CallbackerXCSP3(parser, callbacks)
        callbacker.load_instance()

        if verbose > 0:
            print(f"Solving with {solver}...")

        # Solve and get status
        status = callbacks.solve()

        # Map solution back to pycsp3 variables
        if status in (TypeStatus.SAT, TypeStatus.OPTIMUM):
            _map_solution_to_pycsp3(callbacks)

        return status

    except Exception as e:
        if verbose > 0:
            print(f"Error during solving: {e}")
        raise

    finally:
        # Clean up temporary file
        if tmp_filename is not None and os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def _map_solution_to_pycsp3(callbacks) -> None:
    """Map solution from backend to pycsp3 Variable.value/values attributes."""
    from pycsp3.classes.entities import VarEntities, EVar, EVarArray

    solution = callbacks.get_solution()
    if solution is None:
        return

    def set_var_value(var, value):
        """Set both value and values for a variable."""
        var.value = value
        if not hasattr(var, 'values') or var.values is None:
            var.values = []
        var.values.append(value)

    # Map solution values back to pycsp3 variables
    for item in VarEntities.items:
        if isinstance(item, EVar):
            var = item.variable
            if var.id in solution:
                set_var_value(var, solution[var.id])
        elif isinstance(item, EVarArray):
            for var in item.flatVars:
                if var is not None and var.id in solution:
                    set_var_value(var, solution[var.id])
