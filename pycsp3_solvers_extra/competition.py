"""
XCSP competition output helpers.

Implements the (o, s, v) output format described in
examples/XCSP25/callXCSP25.pdf (Section 7.2 Output Format).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TextIO
import sys

from pycsp3.classes.auxiliary.enums import TypeStatus


@dataclass(frozen=True)
class CompetitionOutput:
    """Structured competition output lines."""

    objective_lines: list[str]
    status_line: str
    values_lines: list[str]


def _objective_sense(parser) -> str | None:
    """Return objective sense ('min' or 'max') when available."""
    entries = getattr(parser, "oEntries", None)
    if not entries:
        return None
    try:
        entry = entries[0]
        minimize = getattr(entry, "minimize", None)
        if minimize is True:
            return "min"
        if minimize is False:
            return "max"
    except Exception:
        return None
    return None


def _has_objective(parser) -> bool:
    """Best-effort detection of whether the parsed instance has an objective."""
    entries = getattr(parser, "oEntries", None)
    try:
        return bool(entries)
    except Exception:
        return False


def _coerce_number(value) -> int | float | None:
    """Best-effort numeric coercion for objective values."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    try:
        return int(value)
    except Exception:
        try:
            return float(value)
        except Exception:
            return None


@dataclass
class ObjectiveProgressPrinter:
    """Stateful printer that emits 'o ' lines on objective improvement."""

    sense: str | None
    stream: TextIO
    best_value: int | float | None = None
    best_value_raw: object | None = None
    printed_any: bool = False

    def _is_improvement(self, value: int | float, raw_value: object) -> bool:
        if self.best_value is None:
            return True
        if self.sense == "min":
            return value < self.best_value
        if self.sense == "max":
            return value > self.best_value
        # Unknown sense: avoid duplicate spam, but allow changes.
        return raw_value != self.best_value_raw

    def report(self, raw_value) -> bool:
        """Report an objective value; prints and returns True on improvement."""
        value = _coerce_number(raw_value)
        # If we cannot compare numerically, fall back to raw equality checks.
        if value is None:
            if raw_value == self.best_value_raw:
                return False
            self.best_value_raw = raw_value
            print(f"o {raw_value}", file=self.stream, flush=True)
            self.printed_any = True
            return True

        if not self._is_improvement(value, raw_value):
            return False

        self.best_value = value
        self.best_value_raw = raw_value
        print(f"o {raw_value}", file=self.stream, flush=True)
        self.printed_any = True
        return True


def make_objective_progress_printer(
    parser,
    *,
    stream: TextIO | None = None,
) -> ObjectiveProgressPrinter | None:
    """Create an objective progress printer when the instance has an objective."""
    if not _has_objective(parser):
        return None
    if stream is None:
        stream = sys.stdout
    return ObjectiveProgressPrinter(sense=_objective_sense(parser), stream=stream)


def _iter_var_ids_from_entries(entries: Iterable) -> Iterable[str]:
    """Yield variable ids from parser variable entries using duck typing."""
    for entry in entries:
        if entry is None:
            continue

        # Arrays expose a flat list via the "variables" attribute.
        variables = getattr(entry, "variables", None)
        if variables is not None:
            for var in variables:
                if var is None:
                    continue
                var_id = getattr(var, "id", None)
                if var_id is not None:
                    yield var_id
            continue

        var_id = getattr(entry, "id", None)
        if var_id is not None:
            yield var_id


def _extract_var_ids(parser, solution: dict[str, int] | None, callbacks) -> list[str]:
    """Extract variable ids in a stable order."""
    entries = getattr(parser, "vEntries", None)
    if entries:
        var_ids = list(_iter_var_ids_from_entries(entries))
        if var_ids:
            return var_ids

    # Fall back to solution or callback variable map if parser entries are missing.
    if solution:
        return sorted(solution.keys())

    vars_map = getattr(callbacks, "vars", None)
    if isinstance(vars_map, dict) and vars_map:
        return sorted(vars_map.keys())

    return []


def _status_to_competition(status: TypeStatus, has_objective: bool) -> str:
    """Map TypeStatus to the XCSP competition status token."""
    if status == TypeStatus.UNSAT:
        return "UNSATISFIABLE"
    if status == TypeStatus.OPTIMUM and has_objective:
        return "OPTIMUM FOUND"
    if status in (TypeStatus.SAT, TypeStatus.OPTIMUM):
        return "SATISFIABLE"
    return "UNKNOWN"


def _chunked(items: list[str], chunk_size: int) -> Iterable[list[str]]:
    """Yield chunks of items to keep lines reasonably sized."""
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def build_competition_output(parser, callbacks, status: TypeStatus) -> CompetitionOutput:
    """Build competition output lines without printing them."""
    has_objective = _has_objective(parser)
    solution = callbacks.get_solution() if status in (TypeStatus.SAT, TypeStatus.OPTIMUM) else None
    objective_value = callbacks.get_objective_value() if solution is not None else None
    progress_printer = None
    get_progress_printer = getattr(callbacks, "get_competition_progress_printer", None)
    if callable(get_progress_printer):
        progress_printer = get_progress_printer()

    var_ids = _extract_var_ids(parser, solution, callbacks)

    objective_lines: list[str] = []
    # Avoid duplicating objective lines if we already streamed progress.
    if (
        has_objective
        and solution is not None
        and objective_value is not None
        and not (progress_printer and getattr(progress_printer, "printed_any", False))
    ):
        objective_lines.append(f"o {objective_value}")

    status_token = _status_to_competition(status, has_objective)
    status_line = f"s {status_token}"

    values_lines: list[str] = []
    if status_token in ("SATISFIABLE", "OPTIMUM FOUND") and solution is not None and var_ids:
        values = [str(solution.get(var_id, "*")) for var_id in var_ids]

        # Emit a multi-line instantiation to keep individual lines manageable.
        chunk_size = 80
        values_lines.append("v <instantiation>")
        values_lines.append("v <list>")
        for chunk in _chunked(var_ids, chunk_size):
            values_lines.append("v " + " ".join(chunk))
        values_lines.append("v </list>")
        values_lines.append("v <values>")
        for chunk in _chunked(values, chunk_size):
            values_lines.append("v " + " ".join(chunk))
        values_lines.append("v </values>")
        values_lines.append("v </instantiation>")

    return CompetitionOutput(
        objective_lines=objective_lines,
        status_line=status_line,
        values_lines=values_lines,
    )


def emit_competition_output(
    parser,
    callbacks,
    status: TypeStatus,
    *,
    stream: TextIO | None = None,
) -> CompetitionOutput:
    """Print competition output lines and return the structured result."""
    if stream is None:
        stream = sys.stdout

    output = build_competition_output(parser, callbacks, status)

    for line in output.objective_lines:
        print(line, file=stream, flush=True)

    print(output.status_line, file=stream, flush=True)

    for line in output.values_lines:
        print(line, file=stream, flush=True)

    return output
