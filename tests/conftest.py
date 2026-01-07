"""
Pytest configuration for pycsp3-solvers-extra tests.

This module sets up the environment to allow PyCSP3 to work with pytest.
PyCSP3 checks sys.argv[0] at import time, which causes issues with pytest.
"""

import sys
import os

# Store original argv
_original_argv = sys.argv.copy()

# Use the stub Python file for PyCSP3's import check
_fake_script = os.path.join(os.path.dirname(__file__), "_pytest_stub.py")

# Override sys.argv[0] before importing pycsp3
sys.argv[0] = _fake_script


import pytest


@pytest.fixture(autouse=True)
def reset_pycsp3_state():
    """Reset PyCSP3 state before each test to ensure isolation."""
    from pycsp3 import clear
    from pycsp3.classes.entities import CtrEntities, VarEntities, ObjEntities, AnnEntities
    from pycsp3.compiler import Compilation

    # Clear all state before test
    clear()

    # Reset compilation state
    Compilation.done = False
    Compilation.string_model = None
    Compilation.model = None

    yield  # Run the test

    # Clear after test as well
    clear()

    # Reset compilation state again
    Compilation.done = False
    Compilation.string_model = None
    Compilation.model = None


def pytest_configure(config):
    """Called after command line options have been parsed."""
    pass


def pytest_unconfigure(config):
    """Called before test process is exited."""
    # Restore original argv
    sys.argv = _original_argv
