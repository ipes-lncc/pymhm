"""Native scalar simplex defaults, canonical derivatives and persisted wave replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_limits

from examples.minimal_wave_elastic_wave import executed_basis_arrays, saved_tabulation
from pymhm import element_backends as providers
from pymhm.elements import p1_geometry
from pymhm.lagrange import (
    element_tabulate,
    reference_basis,
    reference_values,
    scalar_operators,
    tabulate,
)
from pymhm.mesh import TriangleMesh
from pymhm.tetra_lagrange import tetra_indices, tetra_polynomials, tetra_values_gradients
from pymhm.tetrahedral import (
    TetraMesh,
    tetra_barycentric_gradients,
    tetra_basis,
    tetra_element_tabulate,
    tetra_nodal_space,
)


@pytest.mark.parametrize("degree", range(1, 7))
@pytest.mark.parametrize("cell", ["triangle", "tetrahedron"])
def test_reference_extension_retains_native_values_and_physical_derivatives(
    degree: int, cell: str
) -> None:
    """Padded lambda_0 is a declared extension; the field and affine maps are unchanged."""
    dimension = 2 if cell == "triangle" else 3
    bary = np.random.default_rng(83).dirichlet(np.ones(dimension + 1), size=9)
    operation = reference_basis if dimension == 2 else tetra_polynomials
    values, first, second = operation(degree, bary)
    assert_array_equal(first[..., 0], 0)
    assert_array_equal(second[..., 0, :], 0)
    assert_array_equal(second[..., :, 0], 0)
    if dimension == 2:
        triangle = TriangleMesh(
            np.array([[0.1, 0.2], [1.4, 0.3], [0.3, 1.2]]), np.array([[0, 1, 2]])
        )
        actual = tabulate(triangle, degree, bary)
        gradients, _ = p1_geometry(triangle)
        assert_array_equal(reference_values(degree, bary), values)
    else:
        tetrahedron = TetraMesh(
            [[0.1, 0.2, -0.1], [1.2, 0.3, 0], [0.3, 1.1, 0.2], [-0.1, 0.4, 1.0]], [[0, 1, 2, 3]]
        )
        actual = tetra_element_tabulate(tetrahedron, degree, bary)
        gradients = tetra_barycentric_gradients(tetrahedron)
        direct = tetra_values_gradients(degree, bary)
        assert_array_equal(direct[0], values)
        assert_array_equal(direct[1], first)
        assert_array_equal(tetra_basis(degree, bary)[0], values)
    assert_array_equal(actual[2], values)
    assert_allclose(np.einsum("qia,taj->tqij", first, gradients), actual[3], atol=2e-14, rtol=2e-14)
    assert_allclose(
        np.einsum("qiab,taj,tbk->tqijk", second, gradients, gradients),
        actual[4],
        atol=2e-13,
        rtol=2e-14,
    )
    if dimension == 2:
        per_cell = element_tabulate(triangle, degree, bary[None])
        assert_array_equal(per_cell[2], values[None])


@pytest.mark.parametrize(
    "operation,coordinates",
    [
        (reference_values, [[0.1, 0.2, 0.3]]),
        (reference_basis, [[0.1, 0.2, 0.3]]),
        (tetra_polynomials, [[0.1, 0.2, 0.3, 0.1]]),
        (tetra_values_gradients, [[0.1, 0.2, 0.3, 0.1]]),
        (tetra_basis, [[0.1, 0.2, 0.3, 0.1]]),
    ],
)
def test_reference_points_require_unit_sum(operation: Any, coordinates: Any) -> None:
    with pytest.raises(ValueError, match="sum to one"):
        operation(2, np.asarray(coordinates))


@pytest.mark.parametrize(
    "operation,bary",
    [
        (reference_values, [[0.2, 0.3, 0.5]]),
        (reference_basis, [[0.2, 0.3, 0.5]]),
        (tetra_polynomials, [[0.1, 0.2, 0.3, 0.4]]),
        (tetra_values_gradients, [[0.1, 0.2, 0.3, 0.4]]),
    ],
)
def test_no_own_polynomial_fallback_when_native_dependency_is_unavailable(
    operation: Any, bary: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All scalar reference calls enter the same native provider, including defaults."""

    def unavailable(_name: str) -> Any:
        raise ImportError("native dependency absent")

    with providers._ELEMENT_LOCK:
        providers._cached_reference_element.cache_clear()
        providers._cached_nodal_basis.cache_clear()
    monkeypatch.setattr(providers, "import_module", unavailable)
    with pytest.raises(ImportError, match="Basix"):
        operation(2, np.asarray(bary))


def test_default_and_compatibility_spelling_use_the_same_executed_native_basis() -> None:
    triangle = TriangleMesh.unit_square()
    tetrahedron = TetraMesh.unit_cube()
    cases: tuple[tuple[Any, Any, Any], ...] = (
        (tabulate, triangle, np.full((1, 3), 1 / 3)),
        (tetra_element_tabulate, tetrahedron, np.full((1, 4), 1 / 4)),
    )
    for operation, mesh, bary in cases:
        native = operation(mesh, 3, bary, backend="basix")
        for actual in (operation(mesh, 3, bary), operation(mesh, 3, bary, backend="portable")):
            for a, b in zip(actual, native, strict=True):
                assert_array_equal(a, b)


def test_scalar_form_checks_reaction_sign_and_separate_point_backend() -> None:
    """The declared positive reaction and native execution contract reject excluded inputs."""
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="reaction must be nonnegative"):
        scalar_operators(mesh, 2, reaction=-0.1)
    invalid: Any = "unknown"
    per_cell = np.repeat(np.eye(3)[None], len(mesh.cells), axis=0)
    assert per_cell.strides[0] != 0
    with pytest.raises(ValueError, match="backend"):
        element_tabulate(mesh, 2, per_cell, backend=invalid)


def legacy_p3_recipe() -> dict[str, np.ndarray]:
    """Return literal already differentiated coefficients from historical P3 archives."""
    return {
        "actual_multiindices": tetra_indices(3),
        "actual_tetra_cardinal_factor_recipe": np.array(
            [
                [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 3, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]],
                [[0, -1.5, 4.5, 0], [-1.5, 9, 0, 0], [9, 0, 0, 0]],
                [[0, 1, -4.5, 4.5], [1, -9, 13.5, 0], [-9, 27, 0, 0]],
            ],
            dtype=float,
        ),
    }


@pytest.mark.parametrize("native", [True, False])
def test_wave_archive_replays_actual_basis_without_creating_new_element(
    native: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persisted coefficients evaluate the same polynomial under two native thread counts."""
    mesh = TetraMesh(
        [[0.1, 0.2, -0.1], [1.2, 0.3, 0], [0.3, 1.1, 0.2], [-0.1, 0.4, 1.0]], [[0, 1, 2, 3]]
    )
    archive = tmp_path / "basis.npz"
    captured: dict[str, Any] = executed_basis_arrays(3) if native else legacy_p3_recipe()
    np.savez(archive, **captured)
    with np.load(archive, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("archive replay constructed a new finite-element basis")

    monkeypatch.setattr(providers, "create_reference_element", forbidden)
    monkeypatch.setattr(providers, "simplex_lagrange_basis", forbidden)
    bary = np.random.default_rng(21).dirichlet(np.ones(4), size=11)
    with threadpool_limits(1):
        dofs, values, gradient = saved_tabulation(mesh, 3, bary, arrays)
    with threadpool_limits(2):
        replay = saved_tabulation(mesh, 3, bary, arrays)
    for a, b in zip((dofs, values, gradient), replay, strict=True):
        assert_array_equal(a, b)
    _, nodes = tetra_nodal_space(mesh, 3)
    coefficients = 1 + nodes[:, 0] ** 2 + nodes[:, 1] * nodes[:, 2]
    points = bary @ mesh.points[mesh.cells[0]]
    assert_allclose(
        values @ coefficients[dofs[0]],
        1 + points[:, 0] ** 2 + points[:, 1] * points[:, 2],
        atol=3e-14,
    )
    assert_allclose(
        np.einsum("qia,i->qa", gradient[0], coefficients[dofs[0]]),
        np.column_stack((2 * points[:, 0], points[:, 2], points[:, 1])),
        atol=8e-14,
    )


@pytest.mark.parametrize("corruption", ["matrix", "digest", "version", "order", "recipe"])
def test_wave_archive_rejects_changed_basis_contract(corruption: str) -> None:
    mesh = TetraMesh.unit_cube()
    arrays = executed_basis_arrays(3) if corruption != "recipe" else legacy_p3_recipe()
    arrays = {key: value.copy() for key, value in arrays.items()}
    if corruption == "matrix":
        arrays["actual_basix_basis_matrix"][0, 0] += 1e-5
    elif corruption == "digest":
        arrays["actual_basix_basis_sha256"] = np.asarray("invalid")
    elif corruption == "version":
        arrays["actual_basix_version"] = np.asarray("unqualified-version")
    elif corruption == "order":
        arrays["actual_multiindices"] = arrays["actual_multiindices"][::-1]
    else:
        arrays["actual_tetra_cardinal_factor_recipe"] = np.ones((2, 2, 2))
    with pytest.raises(ValueError, match="Archived"):
        saved_tabulation(mesh, 3, np.full((1, 4), 1 / 4), arrays)
