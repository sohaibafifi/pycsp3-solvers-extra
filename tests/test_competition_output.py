"""Tests for XCSP competition (o/s/v) output emission."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

from pycsp3.classes.auxiliary.enums import TypeStatus

from pycsp3_solvers_extra.solver import solve, _native_log_path


class _StubCallbacker:
    """Minimal CallbackerXCSP3 stub."""

    def __init__(self, parser, callbacks):
        self.parser = parser
        self.callbacks = callbacks

    def load_instance(self) -> None:  # pragma: no cover - trivial
        return


def _identity_transform(callbacks, backend_name: str):
    """Replace TransformingCallbacks with a no-op wrapper."""
    _ = backend_name
    return callbacks


def _patch_solver_env(*, parser, backend_class):
    """Patch solver dependencies for controlled competition output tests."""
    return patch.multiple(
        "pycsp3_solvers_extra.solver",
        get_backend=lambda name: backend_class,
        _map_solution_to_pycsp3=lambda callbacks: None,
        _update_pycsp3_solver_state=lambda callbacks, status: None,
    ), patch(
        "pycsp3_solvers_extra.transforms.TransformingCallbacks",
        _identity_transform,
    ), patch(
        "pycsp3.parser.xparser.ParserXCSP3",
        lambda path: parser,
    ), patch(
        "pycsp3.parser.xparser.CallbackerXCSP3",
        _StubCallbacker,
    )


def test_competition_output_csp_sat(capsys):
    """CSP SAT emits s/v lines and no o line."""
    parser = SimpleNamespace(
        vEntries=[
            SimpleNamespace(
                variables=[
                    SimpleNamespace(id="x[0]"),
                    SimpleNamespace(id="x[1]"),
                ]
            ),
            SimpleNamespace(id="y"),
        ],
        oEntries=[],
    )

    class StubBackend:
        def __init__(self, **kwargs):
            _ = kwargs
            self.vars = {"x[0]": 0, "x[1]": 0, "y": 0}

        def apply_hints(self) -> None:
            return

        def solve(self) -> TypeStatus:
            return TypeStatus.SAT

        def get_solution(self):
            return {"x[0]": 1, "x[1]": 2, "y": 3}

        def get_objective_value(self):
            return None

    patches = _patch_solver_env(parser=parser, backend_class=StubBackend)
    with patches[0], patches[1], patches[2], patches[3]:
        status = solve(
            solver="ortools",
            filename="dummy.xml",
            competition_output=True,
            verbose=0,
        )

    out = capsys.readouterr().out
    assert status == TypeStatus.SAT
    assert "s SATISFIABLE" in out
    assert "v <instantiation>" in out
    assert "x[0]" in out and "x[1]" in out and "y" in out
    assert "o " not in out


def test_competition_output_cop_optimum_with_objective_line(capsys):
    """COP OPTIMUM emits o/s/v lines."""
    parser = SimpleNamespace(
        vEntries=[SimpleNamespace(id="x")],
        oEntries=[object()],
    )

    class StubBackend:
        def __init__(self, **kwargs):
            _ = kwargs
            self.vars = {"x": 0}

        def apply_hints(self) -> None:
            return

        def solve(self) -> TypeStatus:
            return TypeStatus.OPTIMUM

        def get_solution(self):
            return {"x": 5}

        def get_objective_value(self):
            return 5

    patches = _patch_solver_env(parser=parser, backend_class=StubBackend)
    with patches[0], patches[1], patches[2], patches[3]:
        status = solve(
            solver="ortools",
            filename="dummy.xml",
            competition_output=True,
            verbose=0,
        )

    out = capsys.readouterr().out
    assert status == TypeStatus.OPTIMUM
    assert "o 5" in out
    assert "s OPTIMUM FOUND" in out
    assert "v <instantiation>" in out


def test_objective_progression_streams_o_lines(capsys):
    """Objective improvements are streamed as multiple o lines."""
    parser = SimpleNamespace(
        vEntries=[SimpleNamespace(id="x")],
        oEntries=[SimpleNamespace(minimize=True)],
    )

    class StubBackend:
        def __init__(self, **kwargs):
            _ = kwargs
            self.vars = {"x": 0}
            self._progress_printer = None

        def set_competition_progress_printer(self, printer) -> None:
            self._progress_printer = printer

        def get_competition_progress_printer(self):
            return self._progress_printer

        def apply_hints(self) -> None:
            return

        def solve(self) -> TypeStatus:
            # Simulate improving solutions during search.
            assert self._progress_printer is not None
            self._progress_printer.report(10)
            self._progress_printer.report(7)
            self._progress_printer.report(7)  # duplicate, should be ignored
            self._progress_printer.report(5)
            return TypeStatus.SAT

        def get_solution(self):
            return {"x": 5}

        def get_objective_value(self):
            return 5

    patches = _patch_solver_env(parser=parser, backend_class=StubBackend)
    with patches[0], patches[1], patches[2], patches[3]:
        status = solve(
            solver="ortools",
            filename="dummy.xml",
            competition_output=True,
            verbose=0,
        )

    out = capsys.readouterr().out
    assert status == TypeStatus.SAT
    o_lines = [line for line in out.splitlines() if line.startswith("o ")]
    assert o_lines == ["o 10", "o 7", "o 5"]
    assert "s SATISFIABLE" in out


def test_optimum_without_objective_reports_satisfiable(capsys):
    """OPTIMUM without objective is mapped to SATISFIABLE (CSP rule)."""
    parser = SimpleNamespace(
        vEntries=[SimpleNamespace(id="x")],
        oEntries=[],
    )

    class StubBackend:
        def __init__(self, **kwargs):
            _ = kwargs
            self.vars = {"x": 0}

        def apply_hints(self) -> None:
            return

        def solve(self) -> TypeStatus:
            return TypeStatus.OPTIMUM

        def get_solution(self):
            return {"x": 1}

        def get_objective_value(self):
            return None

    patches = _patch_solver_env(parser=parser, backend_class=StubBackend)
    with patches[0], patches[1], patches[2], patches[3]:
        status = solve(
            solver="ortools",
            filename="dummy.xml",
            competition_output=True,
            verbose=0,
        )

    out = capsys.readouterr().out
    assert status == TypeStatus.OPTIMUM
    assert "s SATISFIABLE" in out
    assert "s OPTIMUM FOUND" not in out


def test_not_implemented_emits_unsupported_status(capsys):
    """NotImplementedError is mapped to s UNSUPPORTED in competition mode."""
    parser = SimpleNamespace(vEntries=[], oEntries=[])

    class StubBackend:
        def __init__(self, **kwargs):
            _ = kwargs
            self.vars = {}

        def apply_hints(self) -> None:
            return

        def solve(self) -> TypeStatus:
            raise NotImplementedError("unsupported constraint")

        def get_solution(self):
            return None

        def get_objective_value(self):
            return None

    patches = _patch_solver_env(parser=parser, backend_class=StubBackend)
    with patches[0], patches[1], patches[2], patches[3]:
        status = solve(
            solver="ortools",
            filename="dummy.xml",
            competition_output=True,
            verbose=0,
        )

    out = capsys.readouterr().out
    assert status == TypeStatus.UNKNOWN
    assert "s UNSUPPORTED" in out


def test_compile_failure_emits_unknown_status_line(capsys):
    """Compilation failure still emits a valid status line in competition mode."""
    class DummyBackend:
        def __init__(self, **kwargs):
            _ = kwargs

    with patch(
        "pycsp3_solvers_extra.solver.get_backend",
        lambda name: DummyBackend,
    ), patch(
        "pycsp3_solvers_extra.solver.pycsp3_compile",
        lambda filename, verbose=0: None,
    ):
        status = solve(
            solver="ortools",
            filename=None,
            competition_output=True,
            verbose=0,
        )

    out = capsys.readouterr().out
    assert status == TypeStatus.UNKNOWN
    assert "s UNKNOWN" in out


def test_native_solver_streams_o_lines_from_log(capsys, tmp_path):
    """Native solver progress is streamed by tailing its log file."""
    parser = SimpleNamespace(
        vEntries=[SimpleNamespace(id="x")],
        oEntries=[SimpleNamespace(minimize=True)],
    )

    class StubNativeSolver:
        def __init__(self):
            self.log_filename_suffix = None
            self.n_executions = 0
            self.bound = 5
            self.last_solution = SimpleNamespace(
                variables=[SimpleNamespace(id="x")],
                values=[5],
            )

        def setting(self, options: str) -> None:
            _ = options

        def solve(
            self,
            instance,
            solver_str,
            args,
            args_recursive,
            *,
            compiler,
            verbose,
        ):
            _ = (instance, solver_str, args, args_recursive, compiler, verbose)
            from pycsp3.compiler import Compilation

            log_path = _native_log_path(Compilation.pathname, self)
            # Create the log file, then stream improvements.
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("")
                handle.flush()
                time.sleep(0.05)
                for value in (10, 7, 5):
                    handle.write(f"o {value}\n")
                    handle.flush()
                    time.sleep(0.05)
            return TypeStatus.SAT

    stub_solver = StubNativeSolver()

    with patch("pycsp3.parser.xparser.ParserXCSP3", lambda path: parser), patch(
        "pycsp3.solvers.solver.process_options", lambda solving: ("ace", {}, {})
    ), patch("pycsp3.solver", lambda solver_type=None: stub_solver):
        status = solve(
            solver="ace",
            filename="dummy.xml",
            competition_output=True,
            output_dir=str(tmp_path),
        )

    out = capsys.readouterr().out
    assert status == TypeStatus.SAT
    o_lines = [line for line in out.splitlines() if line.startswith("o ")]
    assert o_lines[:3] == ["o 10", "o 7", "o 5"]
    assert "s SATISFIABLE" in out
