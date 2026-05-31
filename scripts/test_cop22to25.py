#!/usr/bin/env python3
"""Smoke test OR-Tools backend on the COP22-25 instance collection."""

from __future__ import annotations

import atexit
import argparse
import contextlib
import gc
import io
import lzma
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pycsp3.classes.auxiliary.enums import TypeStatus
from pycsp3_solvers_extra.solver import solve

try:
    from pycsp3 import end as pycsp3_end
    atexit.unregister(pycsp3_end)
except (ImportError, AttributeError):
    pass


@dataclass(slots=True)
class InstanceResult:
    instance: str
    elapsed: float
    status_name: str | None = None
    error: str | None = None
    output: str = ""


def discover_instances(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        raise SystemExit(f"Instance directory not found: {dir_path}")
    instances = sorted(dir_path.glob("*.xml")) + sorted(dir_path.glob("*.xml.lzma"))
    return instances


def configure_logging(log_path: Path | None) -> logging.Logger:
    logger = logging.getLogger("cop22to25")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def iter_instances(instances: list[Path], limit: int | None) -> Iterable[Path]:
    if limit is None or limit <= 0:
        yield from instances
        return
    for instance in instances[:limit]:
        yield instance


def decompress_instance(instance: Path) -> tuple[str, Path | None]:
    if instance.suffix.lower() != ".lzma":
        return str(instance), None
    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
    try:
        with lzma.open(instance, "rb") as src:
            shutil.copyfileobj(src, tmp)
    finally:
        tmp.close()
    return tmp.name, Path(tmp.name)


def _captured_output(stdout_buffer: io.StringIO, stderr_buffer: io.StringIO) -> str:
    parts = []
    stdout_text = stdout_buffer.getvalue().strip()
    stderr_text = stderr_buffer.getvalue().strip()
    if stdout_text:
        parts.append(stdout_text)
    if stderr_text:
        parts.append(stderr_text)
    return "\n".join(parts)


def reset_pycsp3_state() -> None:
    try:
        from pycsp3 import clear
        from pycsp3.compiler import Compilation

        clear()
        Compilation.done = False
        Compilation.string_model = None
        Compilation.model = None
        Compilation.string_data = None
    except Exception:
        pass
    gc.collect()


def run_instance(instance_path: str, time_limit: float, verbose: int) -> InstanceResult:
    instance = Path(instance_path)
    start = time.monotonic()
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    cleanup_path: Path | None = None

    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            filename_to_use, cleanup_path = decompress_instance(instance)
            status = solve(
                solver="ortools",
                filename=filename_to_use,
                time_limit=time_limit,
                verbose=verbose,
            )
        status_name = status.name if isinstance(status, TypeStatus) else str(status)
        return InstanceResult(
            instance=instance.name,
            elapsed=time.monotonic() - start,
            status_name=status_name,
            output=_captured_output(stdout_buffer, stderr_buffer),
        )
    except Exception as exc:  # pylint: disable=broad-except
        output = _captured_output(stdout_buffer, stderr_buffer)
        error = "".join(traceback.format_exception(exc)).strip()
        if output:
            error = f"{error}\nCaptured output:\n{output}"
        return InstanceResult(
            instance=instance.name,
            elapsed=time.monotonic() - start,
            error=error,
            output=output,
        )
    finally:
        if cleanup_path is not None and cleanup_path.exists():
            try:
                cleanup_path.unlink()
            except OSError:
                pass
        reset_pycsp3_state()


def run_smoke_test(
    instances: list[Path],
    logger: logging.Logger,
    time_limit: float,
    verbose: int,
    jobs: int,
    max_tasks_per_child: int | None,
) -> tuple[int, dict[str, int], list[InstanceResult]]:
    processed = 0
    status_counts: Counter[str] = Counter()
    errors: list[InstanceResult] = []
    total = len(instances)

    if jobs <= 1:
        for instance in instances:
            processed += 1
            logger.info("[%d/%d] Running OR-Tools on %s", processed, total, instance.name)
            result = run_instance(str(instance), time_limit, verbose)
            if result.error is None:
                status_counts[result.status_name] += 1
                logger.info("[%d/%d] %s -> %s (%.2fs)", processed, total, result.instance, result.status_name, result.elapsed)
            else:
                errors.append(result)
                logger.error("[%d/%d] %s -> Error after %.2fs\n%s", processed, total, result.instance, result.elapsed, result.error)
        return processed, dict(status_counts), errors

    logger.info("Running %d instances with %d worker processes", total, jobs)
    max_in_flight = max(jobs * 2, 1)
    instance_iter = iter(instances)
    with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=max_tasks_per_child) as executor:
        future_to_instance = {}

        while len(future_to_instance) < min(total, max_in_flight):
            instance = next(instance_iter, None)
            if instance is None:
                break
            future = executor.submit(run_instance, str(instance), time_limit, verbose)
            future_to_instance[future] = instance

        while future_to_instance:
            done, _ = wait(future_to_instance, return_when=FIRST_COMPLETED)
            for future in done:
                processed += 1
                instance = future_to_instance.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive path for worker crashes
                    result = InstanceResult(
                        instance=instance.name,
                        elapsed=0.0,
                        error="".join(traceback.format_exception(exc)).strip(),
                    )

                if result.error is None:
                    status_counts[result.status_name] += 1
                    logger.info("[%d/%d] %s -> %s (%.2fs)", processed, total, result.instance, result.status_name, result.elapsed)
                else:
                    errors.append(result)
                    logger.error("[%d/%d] %s -> Error after %.2fs\n%s", processed, total, result.instance, result.elapsed, result.error)

                next_instance = next(instance_iter, None)
                if next_instance is not None:
                    next_future = executor.submit(run_instance, str(next_instance), time_limit, verbose)
                    future_to_instance[next_future] = next_instance

    return processed, dict(status_counts), errors


def summarize(
    discovered: int,
    selected: int,
    processed: int,
    status_counts: dict[str, int],
    errors: list[InstanceResult],
    logger: logging.Logger,
) -> None:
    logger.info("Processed %d/%d selected instances (%d discovered total)", processed, selected, discovered)
    if status_counts:
        logger.info("Status breakdown: %s", status_counts)
    if errors:
        logger.error("Encountered %d errors; see log for details.", len(errors))
        for result in errors:
            logger.error("  %s (%.2fs): %s", result.instance, result.elapsed, result.error.splitlines()[0])


def resolve_jobs(requested_jobs: int) -> int:
    if requested_jobs <= 0:
        return os.cpu_count() or 1
    return requested_jobs


def resolve_max_tasks_per_child(requested_max_tasks: int) -> int | None:
    if requested_max_tasks <= 0:
        return None
    return requested_max_tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test OR-Tools on COP22-25.")
    parser.add_argument("--limit", type=int, default=None, help="Only test the first N instances")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes; use 0 to auto-detect from CPU count",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=1,
        help="Recycle each worker after this many instances; use 0 to disable recycling",
    )
    parser.add_argument("--time-limit", type=float, default=1.0, help="Per-instance time limit in seconds")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity level forwarded to solve()")
    parser.add_argument(
        "--log", type=Path, default=Path("logs/cop22to25-ortools.log"), help="File to append execution logs",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    instances_dir = root / "examples" / "COP22to25"
    instances = discover_instances(instances_dir)
    logger = configure_logging(args.log)
    logger.info("Found %d instances under %s", len(instances), instances_dir)
    subset = list(iter_instances(instances, args.limit))
    jobs = resolve_jobs(args.jobs)
    max_tasks_per_child = resolve_max_tasks_per_child(args.max_tasks_per_child)
    processed, status_counts, errors = run_smoke_test(
        subset,
        logger,
        args.time_limit,
        args.verbose,
        jobs,
        max_tasks_per_child,
    )
    summarize(len(instances), len(subset), processed, status_counts, errors, logger)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
