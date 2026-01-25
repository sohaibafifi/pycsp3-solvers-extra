"""
Tests for pycsp3_solvers_extra.loader module.

These tests verify the XCSP3 loading functionality including:
- File parsing (.xml and .xml.lzma)
- Variable and array registration
- Constraint loading
- Reshaping functions
"""

import lzma
import pytest
from pathlib import Path

from pycsp3.classes.main.variables import Variable


class TestReshapeFlatVars:
    """Test the _reshape_flat_vars recursive function."""

    def test_empty_sizes_returns_flat(self):
        """Empty sizes returns original list unchanged."""
        from pycsp3_solvers_extra.loader import _reshape_flat_vars

        flat = [1, 2, 3, 4]
        result = _reshape_flat_vars(flat, [])
        assert result == flat

    def test_1d_reshaping(self):
        """Single dimension reshaping returns flat list."""
        from pycsp3_solvers_extra.loader import _reshape_flat_vars

        flat = [1, 2, 3]
        result = _reshape_flat_vars(flat, [3])
        # With single size, returns list of that size
        assert result == [1, 2, 3]

    def test_2d_reshaping(self):
        """Matrix reshaping [rows, cols]."""
        from pycsp3_solvers_extra.loader import _reshape_flat_vars

        flat = [1, 2, 3, 4, 5, 6]
        result = _reshape_flat_vars(flat, [2, 3])
        # 2 rows, each with 3 elements
        assert result == [[1, 2, 3], [4, 5, 6]]

    def test_3d_reshaping(self):
        """3D tensor reshaping."""
        from pycsp3_solvers_extra.loader import _reshape_flat_vars

        flat = list(range(1, 13))  # 12 elements
        result = _reshape_flat_vars(flat, [2, 2, 3])
        # Expected: 2 matrices of 2x3
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[0][0]) == 3


class TestIsListOfLists:
    """Test the _is_list_of_lists helper."""

    def test_empty_list_returns_false(self):
        """Empty list returns False."""
        from pycsp3_solvers_extra.loader import _is_list_of_lists

        assert _is_list_of_lists([]) is False

    def test_flat_list_returns_false(self):
        """Flat list returns False."""
        from pycsp3_solvers_extra.loader import _is_list_of_lists

        assert _is_list_of_lists([1, 2, 3]) is False

    def test_nested_list_returns_true(self):
        """Nested list returns True."""
        from pycsp3_solvers_extra.loader import _is_list_of_lists

        assert _is_list_of_lists([[1, 2], [3, 4]]) is True

    def test_nested_tuples_returns_true(self):
        """Nested tuples also returns True."""
        from pycsp3_solvers_extra.loader import _is_list_of_lists

        assert _is_list_of_lists([(1, 2), (3, 4)]) is True


class TestNormalizeUnaryTable:
    """Test the _normalize_unary_table helper."""

    def test_flat_values(self):
        """Flat list of values normalized."""
        from pycsp3_solvers_extra.loader import _normalize_unary_table

        result = _normalize_unary_table([1, 2, 3])
        assert result == [1, 2, 3]

    def test_nested_values_flattened(self):
        """Nested tuples flattened."""
        from pycsp3_solvers_extra.loader import _normalize_unary_table

        result = _normalize_unary_table([(1,), (2,), (3,)])
        assert result == [1, 2, 3]

    def test_range_values_expanded(self):
        """Range objects expanded."""
        from pycsp3_solvers_extra.loader import _normalize_unary_table

        result = _normalize_unary_table([range(1, 4)])
        assert result == [1, 2, 3]

    def test_mixed_values(self):
        """Mixed single values and ranges."""
        from pycsp3_solvers_extra.loader import _normalize_unary_table

        result = _normalize_unary_table([1, (2,), range(3, 5)])
        assert result == [1, 2, 3, 4]


class TestParseXcsp3File:
    """Test the _parse_xcsp3_file function."""

    def test_plain_xml_file(self, xcsp_fixture):
        """Load .xml file successfully."""
        from pycsp3_solvers_extra.loader import _parse_xcsp3_file

        parser = _parse_xcsp3_file(xcsp_fixture)
        assert parser is not None
        # Should have parsed at least one variable
        assert len(parser.vEntries) > 0

    def test_lzma_compressed_file(self, xcsp_lzma_fixture):
        """Load .xml.lzma file successfully."""
        from pycsp3_solvers_extra.loader import _parse_xcsp3_file

        parser = _parse_xcsp3_file(xcsp_lzma_fixture)
        assert parser is not None
        assert len(parser.vEntries) > 0

    def test_invalid_extension_raises_error(self, tmp_path):
        """Non-XML file raises ValueError."""
        from pycsp3_solvers_extra.loader import _parse_xcsp3_file

        path = tmp_path / "test.txt"
        path.write_text("not xml")

        with pytest.raises(ValueError, match="Expected an .xml"):
            _parse_xcsp3_file(path)
            
class TestToListVar:
    """Test _to_list_var helper."""

    def test_none_returns_none(self):
        """None input returns None."""
        from pycsp3_solvers_extra.loader import _to_list_var

        assert _to_list_var(None) is None

    def test_empty_list_returns_listvar(self):
        """Empty list converted to ListVar."""
        from pycsp3_solvers_extra.loader import _to_list_var
        from pycsp3.tools.curser import ListVar

        result = _to_list_var([])
        assert isinstance(result, ListVar)


class TestLoadFunction:
    """Test the main load() function."""

    def test_file_not_found(self):
        """FileNotFoundError for missing file."""
        from pycsp3_solvers_extra.loader import load

        with pytest.raises(FileNotFoundError):
            load("/nonexistent/path/file.xml")

    def test_loads_single_variable(self, xcsp_fixture):
        """Single variable loaded into name2obj."""
        from pycsp3_solvers_extra.loader import load
    
        load(xcsp_fixture)
    
        from pycsp3.classes.main.variables import Variable
        assert "x" in Variable.name2obj
        from pycsp3 import clear
        clear()
        Variable.arrays = []

    def test_loads_array_variable(self, xcsp_array_fixture):
        """Array variable loaded and registered."""
        from pycsp3_solvers_extra.loader import load
        from pycsp3 import clear

        load(xcsp_array_fixture)

        # The array should be registered
        assert "x" in Variable.name2obj
        # Array should have 3 elements
        arr = Variable.name2obj["x"]
        assert len(arr) == 3
        clear()
        Variable.arrays = []

    def test_loads_2d_array_variable(self, xcsp_2d_array_fixture):
        """2D array variable loaded with correct shape."""
        from pycsp3_solvers_extra.loader import load

        load(xcsp_2d_array_fixture)
        assert "x" in Variable.name2obj
        arr = Variable.name2obj["x"]
        # Should be 2x3
        assert len(arr) == 2
        assert len(arr[0]) == 3

        from pycsp3 import clear
        clear()
        Variable.arrays = []

    def test_loads_lzma_file(self, xcsp_lzma_fixture):
        """Load .xml.lzma file via load()."""
        from pycsp3_solvers_extra.loader import load

        parser = load(xcsp_lzma_fixture)
        assert parser is not None
        assert "x" in Variable.name2obj

    def test_get_loaded_instance(self, xcsp_fixture):
        """get_loaded_instance returns namespace after load."""
        from pycsp3_solvers_extra.loader import load, get_loaded_instance

        load(xcsp_fixture)
        instance = get_loaded_instance()
        assert instance is not None
        assert instance.parser is not None
        assert instance.source == str(xcsp_fixture)

    def test_clear_model_true(self, xcsp_fixture, xcsp_array_fixture):
        """clear_model=True clears previous state."""
        from pycsp3_solvers_extra.loader import load

        # Load first file
        load(xcsp_fixture)
        assert "x" in Variable.name2obj

        # Load second file with clear_model=True (default)
        load(xcsp_array_fixture)

        # New x should be the array, not the single var
        arr = Variable.name2obj["x"]
        assert isinstance(arr, list)

    def test_clear_model_false(self, xcsp_fixture, tmp_path):
        """clear_model=False preserves previous state."""
        from pycsp3_solvers_extra.loader import load

        # Load first file
        load(xcsp_fixture)

        original_x = Variable.name2obj.get("x")

        # Create a second file with different variable
        xml2 = '''<?xml version="1.0" encoding="UTF-8"?>
<instance format="XCSP3" type="CSP">
  <variables>
    <var id="y"> 1..3 </var>
  </variables>
  <constraints>
    <intension> eq(y,1) </intension>
  </constraints>
</instance>'''
        path2 = tmp_path / "test2.xml"
        path2.write_text(xml2)

        # Load second file without clearing
        load(path2, clear_model=False)


        # Both variables should exist
        assert "x" in Variable.name2obj
        assert "y" in Variable.name2obj


class TestLoadConstraints:
    """Test constraint loading."""

    def test_loads_intension_constraint(self, xcsp_fixture):
        """Intension constraint loaded."""
        from pycsp3_solvers_extra.loader import load
        from pycsp3.classes.entities import CtrEntities

        load(xcsp_fixture)
        # Should have at least one constraint
        assert len(CtrEntities.items) > 0

    def test_loads_extension_constraint(self, xcsp_extension_fixture):
        """Extension (table) constraint loaded."""
        from pycsp3_solvers_extra.loader import load
        from pycsp3.classes.entities import CtrEntities

        load(xcsp_extension_fixture)
        assert len(CtrEntities.items) > 0

    def test_loads_alldifferent_constraint(self, xcsp_array_fixture):
        """AllDifferent constraint loaded."""
        from pycsp3_solvers_extra.loader import load
        from pycsp3.classes.entities import CtrEntities

        load(xcsp_array_fixture)
        assert len(CtrEntities.items) > 0


class TestLoadObjectives:
    """Test objective loading."""

    def test_loads_minimize_objective(self, xcsp_objective_fixture):
        """Minimize objective loaded."""
        from pycsp3_solvers_extra.loader import load
        from pycsp3.classes.entities import ObjEntities

        load(xcsp_objective_fixture)
        # Should have at least one objective
        assert len(ObjEntities.items) > 0


