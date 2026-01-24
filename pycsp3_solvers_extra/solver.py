"""
Main solve() function for pycsp3-solvers-extra.

This module provides the entry point for solving pycsp3 models
with alternative solver backends (OR-Tools, CPO, Z3, Pumpkin).
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
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
EXTRA_SOLVERS = {"ortools", "cpo", "z3", "pumpkin", "minizinc"}


def supported_solvers() -> list[str]:
    """Return list of all supported solver names."""
    return sorted(NATIVE_SOLVERS | EXTRA_SOLVERS)


def _resolve_output_dir(output_dir: str | os.PathLike | None) -> str:
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    output_dir = os.fspath(output_dir)
    if output_dir == "":
        output_dir = tempfile.gettempdir()
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def solve(
    *,
    solver: str = "ortools",
    filename: str | None = None,
    time_limit: float | None = None,
    sols: int | str | None = None,
    verbose: int = 0,
    options: str = "",
    hints: dict[str, int] | None = None,
    output_dir: str | os.PathLike | None = None,
) -> TypeStatus:
    """
    Solve the current pycsp3 model with the specified solver.

    This function shadows pycsp3.solve() to provide additional solver backends.
    For "ace" and "choco", it delegates to the original pycsp3.solve().
    For "ortools" and "cpo", it uses the callback-based XCSP3 translation.

    Args:
        solver: Solver name - "ortools", "cpo", "z3", "ace", or "choco"
        filename: Path to XCSP3 file. If None, compiles current model.
        time_limit: Time limit in seconds (None for no limit)
        sols: Number of solutions to find. Use "all" or pycsp3.ALL for all solutions.
        verbose: Verbosity level (0=quiet, 1=normal, 2=detailed)
        options: Solver-specific options string
        hints: Warm start hints as {var_id: value} dict. Guides solver search.
        output_dir: Directory for generated XCSP3/log files when compiling models.
            Defaults to the system temp directory. Ignored when filename is provided
            (for extra solvers, filename is treated as an XCSP3 input file).

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

    # Handle minizinc/subsolver pattern (e.g., "minizinc/gecode", "minizinc/chuffed")
    subsolver: str | None = None
    if solver_lower.startswith("minizinc"):
        parts = solver_lower.split("/", 1)
        solver_lower = parts[0]  # "minizinc"
        subsolver = parts[1] if len(parts) > 1 else None  # auto-detect if not specified

    # Delegate native solvers to pycsp3
    if solver_lower in NATIVE_SOLVERS:
        return _solve_native(
            solver_lower,
            filename,
            time_limit,
            sols,
            verbose,
            options,
            hints,
            output_dir,
        )

    # Handle extra solvers
    if solver_lower not in EXTRA_SOLVERS:
        raise ValueError(
            f"Unknown solver: {solver}. Supported solvers: {supported_solvers()}"
        )

    return _solve_extra(
        solver_lower,
        filename,
        time_limit,
        sols,
        verbose,
        options,
        hints,
        output_dir,
        subsolver,
    )


def _solve_native(
    solver: str,
    filename: str | None,
    time_limit: float | None,
    sols: int | str | None,
    verbose: int,
    options: str,
    hints: dict[str, int] | None,
    output_dir: str | os.PathLike | None,
) -> TypeStatus:
    """Delegate to pycsp3's native solve() for ACE/Choco."""
    from pycsp3 import ACE, CHOCO, ALL

    solver_type = ACE if solver == "ace" else CHOCO

    # Build solver string with options
    solver_str = f"[{solver}"
    if verbose > 0:
        solver_str += "," + "v" * min(verbose, 3)
    limit_parts: list[str] = []
    if time_limit is not None:
        limit_parts.append(f"{int(time_limit)}s")
    if sols is not None:
        if sols == "all" or sols == ALL:
            limit_parts.append("no")
        elif isinstance(sols, int):
            limit_parts.append(f"{sols}sols")
    if limit_parts:
        if len(limit_parts) == 1:
            solver_str += f",limit={limit_parts[0]}"
        else:
            solver_str += ",limit=[" + ",".join(limit_parts) + "]"
    solver_str += "]"

    # Add warm start hints for ACE/Choco
    if hints:
        # Format: var=value var=value ...
        warm_str = " ".join(f"{var_id}={value}" for var_id, value in hints.items())
        if options:
            options = f"{options} -warm='{warm_str}'"
        else:
            options = f"-warm='{warm_str}'"

    compile_target = filename
    if compile_target is None:
        compile_target = _resolve_output_dir(output_dir)

    return pycsp3_solve(
        solver=solver_str,
        options=options,
        filename=compile_target,
        verbose=verbose - 1,  # pycsp3 verbose: -1=quiet, 0=normal, 1+=detailed
    )


def _solve_extra(
    solver: str,
    filename: str | None,
    time_limit: float | None,
    sols: int | str | None,
    verbose: int,
    options: str,
    hints: dict[str, int] | None,
    output_dir: str | os.PathLike | None,
    subsolver: str | None = None,
) -> TypeStatus:
    """Solve with extra backends using XCSP3 parsing."""
    # Get the appropriate backend
    backend_class = get_backend(solver)
    if backend_class is None:
        raise ImportError(
            f"Backend '{solver}' is not available. "
            f"Install the required package: pip install pycsp3-solvers-extra[{solver}]"
        )

    # Compile to XCSP3 if no filename provided
    if filename is None:
        tmp_dir = _resolve_output_dir(output_dir)
        # Use a temporary file for the XCSP3 instance
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, dir=tmp_dir
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
        backend_kwargs = {
            "time_limit": time_limit,
            "sols": sols,
            "verbose": verbose,
            "options": options,
            "hints": hints,
        }
        # Pass subsolver for backends that support it (e.g., minizinc)
        if subsolver is not None:
            backend_kwargs["subsolver"] = subsolver
        callbacks = backend_class(**backend_kwargs)

        from pycsp3_solvers_extra.transforms import TransformingCallbacks
        callbacks = TransformingCallbacks(callbacks, backend_name=solver)

        # Parse XCSP3 and build solver model
        from pycsp3.parser.xparser import ParserXCSP3, CallbackerXCSP3

        if verbose > 0:
            print(f"Parsing XCSP3 file: {xcsp3_file}")

        # Parse the XML file
        parser = ParserXCSP3(xcsp3_file)

        # Create callbacker to invoke callbacks
        callbacker = CallbackerXCSP3(parser, callbacks)
        callbacker.load_instance()

        # Apply warm start hints after model is built
        callbacks.apply_hints()

        if verbose > 0:
            print(f"Now the file is parsed, starting solving with {solver}...")

        # Solve and get status
        status = callbacks.solve()

        # Map solution back to pycsp3 variables
        if status in (TypeStatus.SAT, TypeStatus.OPTIMUM):
            _map_solution_to_pycsp3(callbacks)

        _update_pycsp3_solver_state(callbacks, status)

        return status

    except Exception as e:
        if verbose > 0:
            print(f"Error during solving: {e}")
        raise

    finally:
        # Re-enable OpOverrider since xparser.py disables it on import
        # This is needed so that Variable.__add__ etc. work for subsequent models
        from pycsp3.tools.curser import OpOverrider
        OpOverrider.enable()

        # Clean up temporary file
        if tmp_filename is not None and os.path.exists(tmp_filename):
            os.unlink(tmp_filename)


def _map_solution_to_pycsp3(callbacks) -> None:
    """Map solution from backend to pycsp3 Variable.value/values attributes."""
    from pycsp3.classes.entities import VarEntities, EVar, EVarArray

    get_all = getattr(callbacks, "get_all_solutions", None)
    all_solutions = get_all() if callable(get_all) else None

    if all_solutions:
        solutions = [sol for sol in all_solutions if sol]

        def set_var_values(var, values):
            var.values = values
            var.value = values[-1] if values else None

        for item in VarEntities.items:
            if isinstance(item, EVar):
                var = item.variable
                values = [sol[var.id] for sol in solutions if var.id in sol]
                if values:
                    set_var_values(var, values)
            elif isinstance(item, EVarArray):
                for var in item.flatVars:
                    if var is None:
                        continue
                    values = [sol[var.id] for sol in solutions if var.id in sol]
                    if values:
                        set_var_values(var, values)
        return

    solution = callbacks.get_solution()
    if solution is None:
        return

    def set_var_value(var, value):
        """Set both value and values for a variable."""
        var.value = value
        if not hasattr(var, 'values') or var.values is None:
            var.values = []
        var.values = [value]

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


def _update_pycsp3_solver_state(callbacks, status: TypeStatus) -> None:
    """Expose extra-solver results through pycsp3 status/bound helpers."""
    import pycsp3

    solution = callbacks.get_solution() if status in (TypeStatus.SAT, TypeStatus.OPTIMUM) else None
    n_solutions = 0
    if status in (TypeStatus.SAT, TypeStatus.OPTIMUM):
        n_solutions = 1
        get_all = getattr(callbacks, "get_all_solutions", None)
        if callable(get_all):
            all_solutions = get_all()
            if all_solutions:
                n_solutions = len(all_solutions)

    bound = callbacks.get_objective_value() if status in (TypeStatus.SAT, TypeStatus.OPTIMUM) else None

    pycsp3._solver = SimpleNamespace(
        status=status,
        last_solution=solution,
        n_solutions=n_solutions,
        bound=bound,
        core=None,
    )
