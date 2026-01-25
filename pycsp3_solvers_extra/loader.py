from __future__ import annotations

import lzma
from pathlib import Path
from types import SimpleNamespace

from pycsp3.classes.auxiliary.enums import TypeCtr, TypeCtrArg, TypeAnn
from pycsp3.classes.entities import EAnnotation, ECtr, EObjective, EToSatisfy, EVar, EVarArray, clear
from pycsp3.classes.main.annotations import TypeAnnArg
from pycsp3.classes.main.constraints import (
    Constraint,
    ConstraintAllDifferent,
    ConstraintAllDifferentList,
    ConstraintAllDifferentMatrix,
    ConstraintExtension,
    ConstraintIntension,
    ConstraintLex,
    ConstraintLexMatrix,
)
from pycsp3.classes.main.variables import Variable
from pycsp3.parser.constants import STATIC
from pycsp3.parser.xentries import XBlock, XCtr, XGroup, XObjExpr, XSlide, XVar, XVarArray
from pycsp3.parser.xparser import ParserXCSP3
from pycsp3.tools.curser import ListVar, OpOverrider
from pycsp3.tools.utilities import matrix_to_string, table_to_string

_loaded_instance = None


def _reshape_flat_vars(flat_vars: list, sizes: list[int]):
    def build(level: int, index: int):
        if level == len(sizes) - 1:
            end = index + sizes[level]
            return list(flat_vars[index:end]), end
        chunk = []
        for _ in range(sizes[level]):
            sub, index = build(level + 1, index)
            chunk.append(sub)
        return chunk, index

    if not sizes:
        return flat_vars
    reshaped, _ = build(0, 0)
    return reshaped


def _to_list_var(tree):
    if tree is None or isinstance(tree, Variable):
        return tree
    return ListVar(_to_list_var(item) for item in tree)


def _parse_xcsp3_file(path: Path):
    from pycsp3.parser.xparser import ParserXCSP3

    suffixes = path.suffixes
    if suffixes[-2:] == [".xml", ".lzma"]:
        with lzma.open(path, "rb") as handle:
            return ParserXCSP3(handle)
    if suffixes[-1:] == [".xml"]:
        return ParserXCSP3(str(path))
    raise ValueError("Expected an .xml or .xml.lzma file")


def _is_list_of_lists(value) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(isinstance(v, (list, tuple)) for v in value)


def _convert_argument(arg):
    value = arg.value
    lifted = False
    content_compressible = True

    if isinstance(value, list):
        if arg.type == TypeCtrArg.MATRIX:
            content_compressible = value
            value = matrix_to_string(value)
        elif arg.type in (TypeCtrArg.SUPPORTS, TypeCtrArg.CONFLICTS):
            if value and not isinstance(value[0], (list, tuple)):
                value = [(v,) for v in value]
            value = table_to_string(value)
        elif arg.type == TypeCtrArg.EXCEPT:
            if _is_list_of_lists(value):
                value = "".join("(" + ",".join(str(v) for v in t) + ")" for t in value)
            else:
                value = "(" + ",".join(str(v) for v in value) + ")"
        elif _is_list_of_lists(value):
            if arg.type == TypeCtrArg.LIST:
                lifted = True
            elif arg.type != TypeCtrArg.MATRIX:
                value = "".join("(" + ",".join(str(v) for v in t) + ")" for t in value)

    content_ordered = isinstance(value, list) and arg.type not in (TypeCtrArg.SET, TypeCtrArg.MSET)
    return value, content_compressible, lifted, content_ordered


def _normalize_unary_table(values: list):
    normalized = []
    for value in values:
        for item in (value if isinstance(value, (list, tuple)) else [value]):
            normalized.extend(item) if isinstance(item, range) else normalized.append(item)

    return normalized


def _collect_template_constraints(template: XCtr, arguments, collected: list[ECtr]) -> None:
    if template.abstraction is None:
        collected.append(ECtr(_constraint_from_xctr(template)))
        return
    for args in arguments:
        template.id = None
        template.abstraction.concretize(args)
        collected.append(ECtr(_constraint_from_xctr(template)))


def _constraint_from_xctr(xctr: XCtr) -> Constraint:
    if xctr.type == TypeCtr.INTENSION:
        tree = next(arg.value for arg in xctr.ctr_args if arg.type == TypeCtrArg.FUNCTION)
        constraint = ConstraintIntension(tree)
    elif xctr.type == TypeCtr.EXTENSION:
        scope = next(arg.value for arg in xctr.ctr_args if arg.type == TypeCtrArg.LIST)
        table_arg = next(arg for arg in xctr.ctr_args if arg.type in (TypeCtrArg.SUPPORTS, TypeCtrArg.CONFLICTS))
        table = table_arg.value
        if len(scope) == 1 and isinstance(table, list) and table:
            table = _normalize_unary_table(table)
        elif isinstance(table, list) and table and isinstance(table[0], list):
            table = [tuple(t) for t in table]
        constraint = ConstraintExtension(scope, table, positive=(table_arg.type == TypeCtrArg.SUPPORTS))
    elif xctr.type == TypeCtr.LEX:
        lists = [arg.value for arg in xctr.ctr_args if arg.type == TypeCtrArg.LIST]
        matrix = next((arg.value for arg in xctr.ctr_args if arg.type == TypeCtrArg.MATRIX), None)
        operator = next((arg.value for arg in xctr.ctr_args if arg.type == TypeCtrArg.OPERATOR), None)
        if matrix is not None:
            constraint = ConstraintLexMatrix(matrix, operator)
        else:
            constraint = ConstraintLex(lists, operator)
    elif xctr.type == TypeCtr.ALL_DIFFERENT:
        lists = [arg.value for arg in xctr.ctr_args if arg.type == TypeCtrArg.LIST]
        matrix = next((arg.value for arg in xctr.ctr_args if arg.type == TypeCtrArg.MATRIX), None)
        excepting = next((arg.value for arg in xctr.ctr_args if arg.type == TypeCtrArg.EXCEPT), None)
        if matrix is not None:
            constraint = ConstraintAllDifferentMatrix(matrix, excepting)
        elif len(lists) > 1:
            constraint = ConstraintAllDifferentList(lists, excepting)
        else:
            constraint = ConstraintAllDifferent(lists[0], excepting) if lists else ConstraintAllDifferent([], excepting)
    else:
        constraint = Constraint(xctr.type)
        for arg in xctr.ctr_args:
            content, compressible, lifted, ordered = _convert_argument(arg)
            constraint.arg(
                arg.type,
                content,
                attributes=list(arg.attributes.items()),
                content_compressible=compressible,
                lifted=lifted,
                content_ordered=ordered,
            )

    constraint.attributes = list(xctr.attributes.items())
    return constraint


def _collect_constraints(entry, collected: list[ECtr]) -> None:
    if isinstance(entry, XCtr):
        collected.append(ECtr(_constraint_from_xctr(entry)))
    elif isinstance(entry, XBlock):
        for sub in entry.subentries:
            _collect_constraints(sub, collected)
    elif isinstance(entry, XGroup):
        _collect_template_constraints(entry.template, entry.all_args, collected)
    elif isinstance(entry, XSlide):
        _collect_template_constraints(entry.template, entry.scopes, collected)


def _load_objectives(entries) -> None:
    for entry in entries:
        way = TypeCtr.MINIMIZE if entry.minimize else TypeCtr.MAXIMIZE
        objective = Constraint(way)
        if isinstance(entry, XObjExpr):
            objective.arg(TypeCtrArg.EXPRESSION, entry.root)
        else:
            objective.attributes.append((TypeCtrArg.TYPE, entry.type))
            objective.arg(TypeCtrArg.LIST, entry.terms)
            if entry.coefficients is not None:
                objective.arg(TypeCtrArg.COEFFS, entry.coefficients)
        EObjective(objective)


def _load_annotations(entries) -> None:
    for entry in entries:
        annotation = Constraint(entry.type)
        if entry.type == TypeAnn.DECISION:
            annotation.arg(TypeAnn.DECISION, entry.value)
        elif entry.type == TypeAnn.VAL_HEURISTIC:
            statics = entry.value.get(STATIC, [])
            annotation.arg(TypeAnnArg.STATICS, statics)
        else:
            continue
        EAnnotation(annotation)


def load(filepath, *, clear_model: bool = True) -> ParserXCSP3:
    """
    Load an XCSP3 instance into pycsp3 global state.

    This registers variables/arrays and rebuilds constraints/objectives
    so that pycsp3 can recompile and solve the instance.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(str(path))

    if clear_model:
        clear()
        Variable.arrays = []

    parser = _parse_xcsp3_file(path)

    array_vars = set()
    for entry in parser.vEntries:
        if isinstance(entry, XVarArray):
            array_vars.update(v for v in entry.variables if v is not None)
            nested = _reshape_flat_vars(entry.variables, entry.size)
            Variable.name2obj[entry.id] = nested
            lv = _to_list_var(nested)
            EVarArray(lv, entry.id)
            Variable.arrays.append(lv)
            for var in entry.variables:
                if var is not None:
                    Variable.name2obj[var.id] = var

    for entry in parser.vEntries:
        if isinstance(entry, XVar) and entry not in array_vars:
            Variable.name2obj[entry.id] = entry
            EVar(entry)

    constraints = []
    for entry in parser.cEntries:
        _collect_constraints(entry, constraints)
    if constraints:
        EToSatisfy(constraints)

    _load_objectives(parser.oEntries)
    _load_annotations(parser.aEntries)

    global _loaded_instance
    _loaded_instance = SimpleNamespace(parser=parser, source=str(path))

    # Parser import disables the OpOverrider; re-enable it for normal modeling.
    OpOverrider.enable()
    return parser


def get_loaded_instance():
    return _loaded_instance
