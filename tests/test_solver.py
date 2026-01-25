"""
Tests for pycsp3_solvers_extra.solver module.

These tests verify the solve() function and its helpers including:
- Solver routing
- Output directory resolution
- Error handling
- Solution mapping
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from pycsp3.classes.auxiliary.enums import TypeStatus

from pycsp3_solvers_extra.solver import (
    solve,
    supported_solvers,
    _resolve_output_dir,
    NATIVE_SOLVERS,
    EXTRA_SOLVERS,
)


class TestSupportedSolvers:
    """Test supported_solvers function."""

    def test_returns_list(self):
        """Returns a list."""
        result = supported_solvers()
        assert isinstance(result, list)

    def test_returns_sorted_list(self):
        """Returns sorted list."""
        solvers = supported_solvers()
        assert solvers == sorted(solvers)

    def test_includes_native_solvers(self):
        """Includes native solvers (ace, choco)."""
        solvers = supported_solvers()
        for native in NATIVE_SOLVERS:
            assert native in solvers

    def test_includes_extra_solvers(self):
        """Includes extra solvers (ortools, cpo, z3, pumpkin, minizinc)."""
        solvers = supported_solvers()
        for extra in EXTRA_SOLVERS:
            assert extra in solvers


class TestResolveOutputDir:
    """Test _resolve_output_dir function."""

    def test_none_uses_tempdir(self):
        """None input uses system temp directory."""
        result = _resolve_output_dir(None)
        assert result == tempfile.gettempdir()

    def test_empty_string_uses_tempdir(self):
        """Empty string uses system temp directory."""
        result = _resolve_output_dir("")
        assert result == tempfile.gettempdir()

    def test_pathlike_converted(self):
        """PathLike converted to string."""
        with tempfile.TemporaryDirectory() as td:
            result = _resolve_output_dir(Path(td))
            assert result == td

    def test_creates_directory_if_missing(self, tmp_path):
        """Creates directory if it doesn't exist."""
        new_dir = tmp_path / "new_subdir"
        assert not new_dir.exists()

        result = _resolve_output_dir(str(new_dir))

        assert new_dir.exists()
        assert result == str(new_dir)

    def test_existing_directory_ok(self, tmp_path):
        """Existing directory works without error."""
        result = _resolve_output_dir(str(tmp_path))
        assert result == str(tmp_path)


class TestSolve:
    """Test main solve() function."""

    def test_unknown_solver_raises_valueerror(self):
        """Unknown solver raises ValueError."""
        with pytest.raises(ValueError, match="Unknown solver"):
            solve(solver="nonexistent_solver")

    def test_error_message_includes_supported_solvers(self):
        """Error message lists supported solvers."""
        with pytest.raises(ValueError) as exc_info:
            solve(solver="unknown")
        assert "ace" in str(exc_info.value)
        assert "ortools" in str(exc_info.value)

    def test_case_insensitive_solver_name(self):
        """Solver name is case-insensitive."""
        # This should not raise ValueError about unknown solver
        # It may fail for other reasons (no model, backend not available)
        with pytest.raises(Exception) as exc_info:
            solve(solver="ORTOOLS")
        # Should not be "Unknown solver" error
        assert "Unknown solver" not in str(exc_info.value)


class TestMinizincSubsolverParsing:
    """Test minizinc/subsolver pattern parsing."""

    @patch('pycsp3_solvers_extra.solver._solve_extra')
    def test_minizinc_without_subsolver(self, mock_solve_extra):
        """minizinc without subsolver passes None."""
        mock_solve_extra.return_value = TypeStatus.SAT

        solve(solver="minizinc")

        mock_solve_extra.assert_called_once()
        call_args = mock_solve_extra.call_args
        # subsolver should be None (9th positional arg)
        assert call_args[0][8] is None

    @patch('pycsp3_solvers_extra.solver._solve_extra')
    def test_minizinc_with_gecode(self, mock_solve_extra):
        """minizinc/gecode extracts gecode as subsolver."""
        mock_solve_extra.return_value = TypeStatus.SAT

        solve(solver="minizinc/gecode")

        mock_solve_extra.assert_called_once()
        call_args = mock_solve_extra.call_args
        # subsolver should be "gecode"
        assert call_args[0][8] == "gecode"

    @patch('pycsp3_solvers_extra.solver._solve_extra')
    def test_minizinc_with_chuffed(self, mock_solve_extra):
        """minizinc/chuffed extracts chuffed as subsolver."""
        mock_solve_extra.return_value = TypeStatus.SAT

        solve(solver="minizinc/chuffed")

        mock_solve_extra.assert_called_once()
        call_args = mock_solve_extra.call_args
        assert call_args[0][8] == "chuffed"


class TestSolveNativeRouting:
    """Test routing to native solvers."""

    @patch('pycsp3_solvers_extra.solver._solve_native')
    def test_ace_routes_to_native(self, mock_native):
        """ace solver routes to _solve_native."""
        mock_native.return_value = TypeStatus.SAT

        solve(solver="ace")

        mock_native.assert_called_once()

    @patch('pycsp3_solvers_extra.solver._solve_native')
    def test_choco_routes_to_native(self, mock_native):
        """choco solver routes to _solve_native."""
        mock_native.return_value = TypeStatus.SAT

        solve(solver="choco")

        mock_native.assert_called_once()


class TestSolveExtraRouting:
    """Test routing to extra solvers."""

    @patch('pycsp3_solvers_extra.solver._solve_extra')
    def test_ortools_routes_to_extra(self, mock_extra):
        """ortools solver routes to _solve_extra."""
        mock_extra.return_value = TypeStatus.SAT

        solve(solver="ortools")

        mock_extra.assert_called_once()

    @patch('pycsp3_solvers_extra.solver._solve_extra')
    def test_z3_routes_to_extra(self, mock_extra):
        """z3 solver routes to _solve_extra."""
        mock_extra.return_value = TypeStatus.SAT

        solve(solver="z3")

        mock_extra.assert_called_once()

    @patch('pycsp3_solvers_extra.solver._solve_extra')
    def test_pumpkin_routes_to_extra(self, mock_extra):
        """pumpkin solver routes to _solve_extra."""
        mock_extra.return_value = TypeStatus.SAT

        solve(solver="pumpkin")

        mock_extra.assert_called_once()


class TestSolveExtraBackendUnavailable:
    """Test _solve_extra when backend is unavailable."""

    @patch('pycsp3_solvers_extra.solver.get_backend')
    def test_backend_not_available_raises_importerror(self, mock_get_backend):
        """Missing backend raises ImportError."""
        mock_get_backend.return_value = None

        with pytest.raises(ImportError, match="not available"):
            solve(solver="ortools")

    @patch('pycsp3_solvers_extra.solver.get_backend')
    def test_error_message_includes_install_hint(self, mock_get_backend):
        """Error message includes installation hint."""
        mock_get_backend.return_value = None

        with pytest.raises(ImportError) as exc_info:
            solve(solver="ortools")

        assert "pip install" in str(exc_info.value)
        assert "ortools" in str(exc_info.value)


class TestSolveParameters:
    """Test parameter passing."""

    @patch('pycsp3_solvers_extra.solver._solve_extra')
    def test_time_limit_passed(self, mock_extra):
        """time_limit passed to backend."""
        mock_extra.return_value = TypeStatus.SAT

        solve(solver="ortools", time_limit=60)

        call_args = mock_extra.call_args
        assert call_args[0][2] == 60  # time_limit is 3rd positional arg

    @patch('pycsp3_solvers_extra.solver._solve_extra')
    def test_sols_passed(self, mock_extra):
        """sols passed to backend."""
        mock_extra.return_value = TypeStatus.SAT

        solve(solver="ortools", sols=5)

        call_args = mock_extra.call_args
        assert call_args[0][3] == 5  # sols is 4th positional arg

    @patch('pycsp3_solvers_extra.solver._solve_extra')
    def test_verbose_passed(self, mock_extra):
        """verbose passed to backend."""
        mock_extra.return_value = TypeStatus.SAT

        solve(solver="ortools", verbose=2)

        call_args = mock_extra.call_args
        assert call_args[0][4] == 2  # verbose is 5th positional arg

    @patch('pycsp3_solvers_extra.solver._solve_extra')
    def test_options_passed(self, mock_extra):
        """options passed to backend."""
        mock_extra.return_value = TypeStatus.SAT

        solve(solver="ortools", options="--opt=val")

        call_args = mock_extra.call_args
        assert call_args[0][5] == "--opt=val"  # options is 6th positional arg

    @patch('pycsp3_solvers_extra.solver._solve_extra')
    def test_hints_passed(self, mock_extra):
        """hints passed to backend."""
        mock_extra.return_value = TypeStatus.SAT
        hints = {"x": 1, "y": 2}

        solve(solver="ortools", hints=hints)

        call_args = mock_extra.call_args
        assert call_args[0][6] == hints  # hints is 7th positional arg


class TestMapSolutionToPycsp3:
    """Test _map_solution_to_pycsp3 function."""

    def test_none_solution_returns_early(self):
        """None solution returns without error."""
        from pycsp3_solvers_extra.solver import _map_solution_to_pycsp3

        callbacks = Mock()
        callbacks.get_solution.return_value = None
        callbacks.get_all_solutions = None

        # Should not raise
        _map_solution_to_pycsp3(callbacks)


class TestUpdatePycsp3SolverState:
    """Test _update_pycsp3_solver_state function."""

    def test_sets_pycsp3_solver_namespace(self):
        """pycsp3._solver namespace set correctly."""
        import pycsp3
        from pycsp3_solvers_extra.solver import _update_pycsp3_solver_state

        callbacks = Mock()
        callbacks.get_solution.return_value = {"x": 1}
        callbacks.get_objective_value.return_value = 42
        callbacks.get_all_solutions = None

        _update_pycsp3_solver_state(callbacks, TypeStatus.OPTIMUM)

        assert hasattr(pycsp3, '_solver')
        assert pycsp3._solver.status == TypeStatus.OPTIMUM
        assert pycsp3._solver.bound == 42

    def test_unsat_has_no_solution(self):
        """UNSAT status has None solution."""
        import pycsp3
        from pycsp3_solvers_extra.solver import _update_pycsp3_solver_state

        callbacks = Mock()

        _update_pycsp3_solver_state(callbacks, TypeStatus.UNSAT)

        assert pycsp3._solver.status == TypeStatus.UNSAT
        assert pycsp3._solver.last_solution is None
        assert pycsp3._solver.n_solutions == 0

    def test_sat_has_one_solution(self):
        """SAT status with single solution has n_solutions=1."""
        import pycsp3
        from pycsp3_solvers_extra.solver import _update_pycsp3_solver_state

        callbacks = Mock()
        callbacks.get_solution.return_value = {"x": 1}
        callbacks.get_objective_value.return_value = None
        callbacks.get_all_solutions = None

        _update_pycsp3_solver_state(callbacks, TypeStatus.SAT)

        assert pycsp3._solver.n_solutions == 1

    def test_multiple_solutions_counted(self):
        """Multiple solutions counted correctly."""
        import pycsp3
        from pycsp3_solvers_extra.solver import _update_pycsp3_solver_state

        callbacks = Mock()
        callbacks.get_solution.return_value = {"x": 1}
        callbacks.get_objective_value.return_value = None
        callbacks.get_all_solutions.return_value = [{"x": 1}, {"x": 2}, {"x": 3}]

        _update_pycsp3_solver_state(callbacks, TypeStatus.SAT)

        assert pycsp3._solver.n_solutions == 3
