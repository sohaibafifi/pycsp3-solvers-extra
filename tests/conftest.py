"""
Pytest configuration for pycsp3-solvers-extra tests.

This module sets up the environment to allow PyCSP3 to work with pytest.
PyCSP3 checks sys.argv[0] at import time, which causes issues with pytest.
"""

import sys
import os
import atexit
import lzma

# Store original argv
_original_argv = sys.argv.copy()

# Use the stub Python file for PyCSP3's import check
_fake_script = os.path.join(os.path.dirname(__file__), "_pytest_stub.py")

# Override sys.argv[0] before importing pycsp3
sys.argv[0] = _fake_script


import pytest

# Unregister pycsp3's atexit callback to prevent errors at exit
try:
    from pycsp3 import end as pycsp3_end
    atexit.unregister(pycsp3_end)
except (ImportError, AttributeError):
    pass


@pytest.fixture(autouse=True)
def reset_pycsp3_state():
    """Reset PyCSP3 state before each test to ensure isolation."""
    from pycsp3 import clear
    from pycsp3.classes.entities import CtrEntities, VarEntities, ObjEntities, AnnEntities
    from pycsp3.compiler import Compilation

    # Clear all state before test
    clear()

    # Reset compilation state fully
    Compilation.done = False
    Compilation.string_model = None
    Compilation.model = None
    Compilation.string_data = None

    yield  # Run the test

    # Clear after test as well
    clear()

    # Reset compilation state again
    Compilation.done = False
    Compilation.string_model = None
    Compilation.model = None
    Compilation.string_data = None


def pytest_configure(config):
    """Called after command line options have been parsed."""
    pass


def pytest_unconfigure(config):
    """Called before test process is exited."""
    # Restore original argv
    sys.argv = _original_argv


# ========== XCSP3 Fixture Generators ==========

@pytest.fixture
def minimal_xcsp_xml():
    """Minimal valid XCSP3 XML with one variable and one constraint."""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<instance format="XCSP3" type="CSP">
  <variables>
    <var id="x"> 1..3 </var>
  </variables>
  <constraints>
    <intension> eq(x,1) </intension>
  </constraints>
</instance>'''


@pytest.fixture
def xcsp_with_array_xml():
    """XCSP3 XML with variable arrays."""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<instance format="XCSP3" type="CSP">
  <variables>
    <array id="x" size="[3]"> 1..5 </array>
  </variables>
  <constraints>
    <allDifferent> x[] </allDifferent>
  </constraints>
</instance>'''


@pytest.fixture
def xcsp_with_2d_array_xml():
    """XCSP3 XML with 2D variable array."""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<instance format="XCSP3" type="CSP">
  <variables>
    <array id="x" size="[2][3]"> 1..6 </array>
  </variables>
  <constraints>
    <allDifferent> x[][] </allDifferent>
  </constraints>
</instance>'''


@pytest.fixture
def xcsp_extension_xml():
    """XCSP3 XML with extension (table) constraint."""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<instance format="XCSP3" type="CSP">
  <variables>
    <var id="x"> 1..3 </var>
    <var id="y"> 1..3 </var>
  </variables>
  <constraints>
    <extension>
      <list> x y </list>
      <supports> (1,2)(2,3)(3,1) </supports>
    </extension>
  </constraints>
</instance>'''


@pytest.fixture
def xcsp_objective_xml():
    """XCSP3 XML with optimization objective."""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<instance format="XCSP3" type="COP">
  <variables>
    <array id="x" size="[3]"> 1..5 </array>
  </variables>
  <constraints>
    <allDifferent> x[] </allDifferent>
  </constraints>
  <objectives>
    <minimize type="sum"> x[] </minimize>
  </objectives>
</instance>'''


@pytest.fixture
def xcsp_fixture(tmp_path, minimal_xcsp_xml):
    """Create a temporary .xml file with minimal XCSP3 content."""
    path = tmp_path / "test.xml"
    path.write_text(minimal_xcsp_xml)
    return path


@pytest.fixture
def xcsp_lzma_fixture(tmp_path, minimal_xcsp_xml):
    """Create a temporary .xml.lzma file with minimal XCSP3 content."""
    path = tmp_path / "test.xml.lzma"
    with lzma.open(path, 'wt') as f:
        f.write(minimal_xcsp_xml)
    return path


@pytest.fixture
def xcsp_array_fixture(tmp_path, xcsp_with_array_xml):
    """Create a temporary .xml file with array variable."""
    path = tmp_path / "test_array.xml"
    path.write_text(xcsp_with_array_xml)
    return path


@pytest.fixture
def xcsp_2d_array_fixture(tmp_path, xcsp_with_2d_array_xml):
    """Create a temporary .xml file with 2D array variable."""
    path = tmp_path / "test_2d_array.xml"
    path.write_text(xcsp_with_2d_array_xml)
    return path


@pytest.fixture
def xcsp_extension_fixture(tmp_path, xcsp_extension_xml):
    """Create a temporary .xml file with extension constraint."""
    path = tmp_path / "test_extension.xml"
    path.write_text(xcsp_extension_xml)
    return path


@pytest.fixture
def xcsp_objective_fixture(tmp_path, xcsp_objective_xml):
    """Create a temporary .xml file with objective."""
    path = tmp_path / "test_objective.xml"
    path.write_text(xcsp_objective_xml)
    return path
