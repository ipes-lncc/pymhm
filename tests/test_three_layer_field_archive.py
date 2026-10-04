"""Verify executed product recipes without substituting the producer's basis."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples import three_layer_field_archive as archive
from pymhm import TriangleMesh, lagrange
from pymhm.elastodynamics import _make_local, _stress_from_gradient
from pymhm.mesh import FaceSpace, SkeletonSpace


@pytest.fixture
def local():
    """A non-axis-aligned macro exercises physical maps and one-sided P3 DOFs."""
    mesh = TriangleMesh([[0.11, 0.23], [0.81, 0.26], [0.17, 0.92]], [[0, 1, 2]])
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2, 2) for _ in mesh.faces), 2)
    return _make_local(0, mesh, skeleton, 3, 2, 5, 1.7, None, 2.0, 3.0)


def _rehash(record, arrays):
    record["arrays_sha256"] = {key: archive.array_digest(value) for key, value in arrays.items()}
    record["recipe_arrays_sha256"] = {
        key: archive.array_digest(arrays[key]) for key in record["recipe_arrays_sha256"]
    }


def test_observer_has_no_shared_mutation_and_exact_original_tables(local):
    """Observing factor derivatives leaves operators, force and owner object bitwise intact."""
    owner = lagrange.Polynomial
    original_mass, original_stiffness = local.mass.data.copy(), local.stiffness.data.copy()
    force = local.load(lambda x: np.column_stack((x[:, 0] ** 3, x[:, 1] ** 2)))
    record, arrays = archive.capture_field_basis(local)
    archive.validate_field_basis(record, arrays)
    assert lagrange.Polynomial is owner
    assert arrays["actual_mass_basis_values"] is local.basis
    assert arrays["actual_mass_quadrature_points"] is local.points
    assert arrays["actual_mass_quadrature_weights"] is local.weights
    np.testing.assert_array_equal(local.mass.data, original_mass)
    np.testing.assert_array_equal(local.stiffness.data, original_stiffness)
    np.testing.assert_array_equal(
        local.load(lambda x: np.column_stack((x[:, 0] ** 3, x[:, 1] ** 2))), force
    )
    bary = np.array([[0.143, 0.278, 0.579], [-0.1, 0.4, 0.7]])
    original = lagrange.reference_basis(3, bary)
    values, gradient = archive.tabulate_product(arrays, bary)
    np.testing.assert_array_equal(values, original[0])
    np.testing.assert_array_equal(gradient, original[1])
    for count in (1, 2):
        with threadpool_limits(limits=count):
            replay = archive.tabulate_product(arrays, bary)
        for actual, expected in zip(replay, (values, gradient), strict=True):
            np.testing.assert_array_equal(actual, expected)


def test_literal_tables_recover_cubic_vector_field_and_raw_stress(local):
    """Saved one-sided physical derivatives evaluate the same Cartesian P3 function."""
    record, arrays = archive.capture_field_basis(local)
    archive.validate_field_basis(record, arrays)
    x, y = local.nodes.T
    coefficients = np.column_stack((x**3 + 2 * y, x * y**2 - x)).ravel()
    for order in (5, 7):
        values, gradient, stress = archive.evaluate_saved_field(
            arrays, local.dofs, coefficients, order=order
        )
        x, y = arrays[f"field_q{order}_points"].transpose(2, 0, 1)
        expected = np.stack((x**3 + 2 * y, x * y**2 - x), axis=-1)
        expected_gradient = np.empty_like(gradient)
        expected_gradient[..., 0, 0] = 3 * x**2
        expected_gradient[..., 0, 1] = 2
        expected_gradient[..., 1, 0] = y**2 - 1
        expected_gradient[..., 1, 1] = 2 * x * y
        np.testing.assert_allclose(values, expected, rtol=0, atol=2e-15)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=0, atol=3e-14)
        np.testing.assert_array_equal(
            stress, _stress_from_gradient(local, arrays[f"field_q{order}_points"], gradient)
        )


@pytest.mark.parametrize(
    "defect",
    ["factor", "derivative", "table", "nodes", "map", "bits", "runtime", "degree", "kelvin"],
)
def test_reject_semantic_basis_tampering_even_with_new_digests(local, defect):
    """Archive hashes alone cannot relabel a different represented basis as selected P3."""
    record, raw = archive.capture_field_basis(local)
    record, arrays = deepcopy(record), {key: value.copy() for key, value in raw.items()}
    if defect == "factor":
        arrays["product_factor_coefficients"][0, 3, 0, 0] += 1e-5
    elif defect == "derivative":
        arrays["product_factor_coefficients"][0, 3, 2, 0] += 1e-5
    elif defect == "table":
        arrays["field_q5_gradients"][0, 0, 0, 0] += 1e-5
    elif defect == "nodes":
        arrays["product_local_nodes"][0, 0] += 1e-5
    elif defect == "map":
        arrays["product_local_dofs"][0, [0, 1]] = arrays["product_local_dofs"][0, [1, 0]]
    elif defect == "bits":
        record["coefficient_precision_bits"] = 64
    elif defect == "runtime":
        record["recipe_runtime"]["polynomial_source_sha256"] = "0" * 64
    elif defect == "degree":
        record["degree"] = 2
    else:
        arrays["product_kelvin_map"][2] *= -1
    _rehash(record, arrays)
    with pytest.raises(ValueError):
        archive.validate_field_basis(record, arrays)


def test_coherent_basis_permutation_preserves_field(local):
    """Reordering basis columns and their actual injection together preserves the field."""
    record, raw = archive.capture_field_basis(local)
    arrays = {key: value.copy() for key, value in raw.items()}
    coefficients = np.arange(2 * len(local.nodes), dtype=float) / 100
    expected = archive.evaluate_saved_field(arrays, local.dofs, coefficients)
    permutation = np.array([3, 7, 2, 1, 9, 5, 0, 6, 8, 4])
    arrays["product_multiindices"] = arrays["product_multiindices"][permutation]
    arrays["product_reference_nodes"] = arrays["product_reference_nodes"][permutation]
    arrays["product_local_dofs"] = arrays["product_local_dofs"][:, permutation]
    arrays["actual_mass_basis_values"] = arrays["actual_mass_basis_values"][..., permutation]
    for order in (5, 7):
        arrays[f"field_q{order}_values"] = arrays[f"field_q{order}_values"][..., permutation]
        arrays[f"field_q{order}_gradients"] = arrays[f"field_q{order}_gradients"][
            ..., permutation, :
        ]
    _rehash(record, arrays)
    archive.validate_field_basis(record, arrays)
    actual = archive.evaluate_saved_field(arrays, arrays["product_local_dofs"], coefficients)
    for result, original in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, original, rtol=0, atol=3e-14)


def test_immediately_excluded_contracts_and_bad_injection(local):
    """The recipe contract applies to planar P3 and sufficient declared field orders."""
    for orders in ((4,), (5, 5), (True,), ()):
        with pytest.raises(ValueError, match="quadrature"):
            archive.capture_field_basis(local, orders=orders)
    with pytest.raises(ValueError, match="P3"):
        archive.capture_field_basis(replace(local, degree=2))
    _, arrays = archive.capture_field_basis(local)
    for dofs in (local.dofs.astype(float), -local.dofs - 1, local.dofs + len(local.nodes)):
        with pytest.raises(ValueError, match="injection"):
            archive.evaluate_saved_field(arrays, dofs, np.zeros(2 * len(local.nodes)))
