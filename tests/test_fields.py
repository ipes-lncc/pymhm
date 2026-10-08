"""Named fields preserve executed coefficient, reconstruction and evaluation contracts."""

from __future__ import annotations

import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from pymhm.postprocessing.fields import (
    DiscreteField,
    FieldDefinition,
    evaluate_field,
    solution_field,
)


def polynomial_evaluator(mesh: Any, coefficients: Any, points: Any, *, cells: Any = None) -> Any:
    """Evaluate a declared two-coefficient polynomial, retaining caller side labels."""
    assert mesh == "declared-mesh"
    points = np.asarray(points)
    return coefficients[0] + coefficients[1] * points[:, 0]


def polynomial_gradient(mesh: Any, coefficients: Any, points: Any, *, cells: Any = None) -> Any:
    """Differentiate the declared polynomial independently of the value evaluator."""
    assert mesh == "declared-mesh"
    result = np.zeros_like(np.asarray(points), dtype=float)
    result[:, 0] = coefficients[1]
    return result


def test_named_custom_field_owns_coefficients_and_reconstruction_identity() -> None:
    """Custom spatial evaluation and coefficient views share the same executed basis."""
    matrix = np.array([[1.0, 2.0, 0], [0.0, 1.0, 3.0]])
    definition = FieldDefinition(
        "pressure",
        "declared-mesh",
        evaluator=polynomial_evaluator,
        gradient_evaluator=polynomial_gradient,
        reconstruction=matrix,
        basis_id="monomials-1-x",
    )
    matrix[:] = 0
    assert not definition.reconstruction.flags.writeable
    solution = SimpleNamespace(
        field_data=((definition,), (definition,)), fields=(np.array([1, 2, 3]), np.array([3, 2, 1]))
    )
    fields = solution_field(solution, "pressure")
    assert len(fields) == 2
    assert_array_equal(fields[0].coefficients, [5, 11])
    assert_array_equal(fields[1].coefficients, [7, 5])
    assert not fields[0].coefficients.flags.writeable
    assert fields[0].mesh == "declared-mesh"
    assert fields[0].basis_digest == definition.basis_digest
    assert fields[0].basis_digest != FieldDefinition("pressure", basis_id="different").basis_digest
    assert_allclose(fields[0].evaluate([[0.2, 0], [0.7, 1]], cells=[0, 1]), [7.2, 12.7])
    assert_array_equal(evaluate_field(fields[1], [[0.2, 0]]), [8])
    assert_array_equal(fields[0].gradient([[0.2, 0], [0.7, 1]], cells=[0, 1]), [[11, 0], [11, 0]])
    replay = pickle.loads(pickle.dumps(fields[0]))
    assert not replay.coefficients.flags.writeable
    assert not replay.definition.reconstruction.flags.writeable
    assert_array_equal(replay.coefficients, fields[0].coefficients)
    assert replay.basis_digest == fields[0].basis_digest
    assert_array_equal(replay.gradient([[0.2, 0]]), [[11, 0]])
    value, gradient = replay.values_and_gradient([[0.2, 0], [0.7, 1]], cells=[0, 1])
    assert_allclose(value, [7.2, 12.7])
    assert_array_equal(gradient, [[11, 0], [11, 0]])
    assert_array_equal(
        replay.evaluate([[0.2, 0], [0.7, 1]]), fields[0].evaluate([[0.2, 0], [0.7, 1]])
    )
    restored = pickle.loads(pickle.dumps(definition))
    assert not restored.reconstruction.flags.writeable
    assert restored.basis_digest == definition.basis_digest
    sparse_definition = FieldDefinition(
        "pressure",
        "declared-mesh",
        evaluator=polynomial_evaluator,
        reconstruction=sparse.csr_matrix(definition.reconstruction),
        basis_id="monomials-1-x",
    )
    sparse_fields = solution_field(
        SimpleNamespace(field_data=((sparse_definition,),), fields=(np.array([1, 2, 3]),)),
        "pressure",
    )
    assert_allclose(sparse_fields[0].evaluate([[0.2, 0]]), fields[0].evaluate([[0.2, 0]]))
    sparse_replay = pickle.loads(pickle.dumps(sparse_fields[0]))
    assert sparse_replay.basis_digest == sparse_fields[0].basis_digest
    assert not sparse_replay.definition.reconstruction.data.flags.writeable
    assert not sparse_replay.definition.reconstruction.indices.flags.writeable
    assert not sparse_replay.definition.reconstruction.indptr.flags.writeable
    values = np.array([1.0, 2.0])
    plain = DiscreteField(FieldDefinition("coefficients", mesh="any"), values)
    values[:] = 0
    assert_array_equal(plain.coefficients, [1, 2])
    with pytest.raises(TypeError, match="no declared spatial"):
        plain.evaluate([[0.2]])
    with pytest.raises(TypeError, match="no declared gradient"):
        plain.gradient([[0.2]])
    with pytest.raises(TypeError, match="no declared portable"):
        _ = plain.portable_coefficients


def test_field_contracts_reject_ambiguous_and_incompatible_definitions() -> None:
    """Names, evaluator conventions, vectors and reconstructions fail before misuse."""
    for kwargs, error, message in (
        ({"name": ""}, ValueError, "nonempty string"),
        ({"name": 1}, ValueError, "nonempty string"),
        ({"name": "pressure", "evaluator": 1}, TypeError, "callable"),
        ({"name": "pressure", "gradient_evaluator": 1}, TypeError, "callable"),
        (
            {"name": "pressure", "evaluator": polynomial_evaluator, "descriptor": object()},
            ValueError,
            "one field",
        ),
        ({"name": "pressure", "basis_id": None}, TypeError, "basis_id"),
        ({"name": "pressure", "evaluator": polynomial_evaluator}, ValueError, "basis_id"),
        ({"name": "pressure", "reconstruction": [1, 2]}, ValueError, "matrix"),
    ):
        with pytest.raises(error, match=message):
            FieldDefinition(**kwargs)
    definition = FieldDefinition("pressure")
    with pytest.raises(ValueError, match="vector"):
        DiscreteField(definition, [[1, 2]])
    with pytest.raises(ValueError, match="finite"):
        DiscreteField(definition, [np.nan])
    for solution in (
        SimpleNamespace(field_data=(), fields=()),
        SimpleNamespace(field_data=((),), fields=([1, 2],)),
        SimpleNamespace(field_data=((definition, definition),), fields=([1, 2],)),
    ):
        with pytest.raises(KeyError, match="declared"):
            solution_field(solution, "pressure")
    mapped = FieldDefinition("pressure", reconstruction=[[1, 0, 0]])
    with pytest.raises(ValueError, match="executed local vector"):
        solution_field(SimpleNamespace(field_data=((mapped,),), fields=([1, 2],)), "pressure")
    descriptor = SimpleNamespace(
        mesh="native-mesh", size=3, basis_digest="executed-basis", mapping=np.array([2, 0])
    )
    native = DiscreteField(FieldDefinition("native", descriptor=descriptor), [1, 2, 3])
    assert native.mesh == "native-mesh"
    assert_array_equal(native.portable_coefficients, [3, 1])
    assert not native.portable_coefficients.flags.writeable
    assert native.basis_digest != definition.basis_digest
    with pytest.raises(ValueError, match="descriptor"):
        FieldDefinition("native", descriptor=descriptor, reconstruction=[[1, 2]])


def test_basis_matrix_accessor_preserves_owned_coordinates() -> None:
    """The public archive accessor returns the declared matrix without regeneration."""
    matrix = np.array([[1.0, 2.0], [0.0, 3.0]])
    descriptor = SimpleNamespace(basis_matrix=matrix)
    definition = FieldDefinition("pressure", descriptor=descriptor)
    archived = definition.basis_matrix
    matrix[:] = 0
    assert_array_equal(archived, [[1, 2], [0, 3]])
    assert not archived.flags.writeable
    # A callable object can declare a different basis contract than a native descriptor.
    polynomial_evaluator.__dict__["basis_matrix"] = archived
    try:
        custom = FieldDefinition("pressure", evaluator=polynomial_evaluator, basis_id="declared")
        assert_array_equal(custom.basis_matrix, archived)
    finally:
        delattr(polynomial_evaluator, "basis_matrix")
    with pytest.raises(TypeError, match="single basis matrix"):
        _ = FieldDefinition("coefficients").basis_matrix


def test_recursive_field_views_use_each_child_trace_once() -> None:
    """Missing parent definitions descend, while declared parents retain precedence."""
    definition = FieldDefinition(
        "pressure", reconstruction=[[2]], trace_reconstruction=[[3]], trace_dofs=[0], offset=[4]
    )
    leaf = SimpleNamespace(
        field_data=((definition,),),
        fields=(np.array([1.0]),),
        trace=np.array([5.0]),
        children=(None,),
    )
    parent = SimpleNamespace(
        field_data=((), (definition,)),
        fields=(np.array([9.0]), np.array([2.0])),
        trace=np.array([7.0]),
        children=(leaf, leaf),
    )
    fields = solution_field(parent, "pressure", recursive=True)
    assert [float(field.coefficients[0]) for field in fields] == [21, 29]
    with pytest.raises(KeyError, match="once"):
        solution_field(parent, "pressure")
    with pytest.raises(TypeError, match="boolean"):
        solution_field(parent, "pressure", recursive=1)
    for children in ((), (None,)):
        with pytest.raises(KeyError, match="once"):
            solution_field(
                SimpleNamespace(field_data=((),), fields=([1],), children=children),
                "pressure",
                recursive=True,
            )
    duplicate = SimpleNamespace(field_data=((definition, definition),), fields=([1],))
    with pytest.raises(KeyError, match="once"):
        solution_field(duplicate, "pressure", recursive=True)
