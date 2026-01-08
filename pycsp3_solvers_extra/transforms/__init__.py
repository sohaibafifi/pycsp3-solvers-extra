"""
Transformation utilities for constraint decomposition.

This module provides a pipeline for normalizing, rewriting, and
decomposing constraints before dispatching to solver backends.
"""

from pycsp3_solvers_extra.transforms.capabilities import BackendCapabilities
from pycsp3_solvers_extra.transforms.context import TransformContext
from pycsp3_solvers_extra.transforms.pipeline import TransformingCallbacks
from pycsp3_solvers_extra.transforms.types import ConstraintCall

__all__ = [
    "BackendCapabilities",
    "ConstraintCall",
    "TransformContext",
    "TransformingCallbacks",
]
