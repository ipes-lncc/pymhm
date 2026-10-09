"""User-defined 3D volume/trace equations preserve executed fields and physical gauges."""

from typing import Any

import numpy as np
import pytest

from examples.formulations import scalar_3d
from examples.formulations.transport_3d import (
    conforming_transport_3d,
    define_transport_3d,
    transport_3d,
)
from pymhm._legacy.models.transport.polyhedral import solve_polyhedral_rad
from pymhm._legacy.models.transport.rad_3d import solve_rad_3d, solve_rad_3d_conforming
from pymhm.fem.scalar.tetrahedron import tetra_boundary_nodes, tetra_nodal_space
from pymhm.fem.traces.polygon_3d import PolygonalSkeleton3D
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.linalg.linear import SolverUnavailableError
from pymhm.meshes.polyhedral import PolyhedralMesh
from pymhm.meshes.tetrahedron import TetraMesh


def affine(points: np.ndarray) -> np.ndarray:
    """Evaluate an independently chosen affine physical field."""
    return 1 + points @ np.array([1.0, 2.0, -1.0])


def conductivity(points: np.ndarray) -> np.ndarray:
    """Return a positive variable diffusion whose divergence is (1,0,0)."""
    return 1 + points[:, 0]


def velocity(points: np.ndarray) -> np.ndarray:
    """Return beta=(x,y,z), independently differentiated to divergence three."""
    return points


def tangent_cube(points: np.ndarray) -> np.ndarray:
    """Return a divergence-free polynomial velocity tangent to the cube boundary."""
    x, y, _ = points.T
    return np.column_stack(
        (x * (1 - x) * (1 - 2 * y), -y * (1 - y) * (1 - 2 * x), np.zeros(len(points)))
    )


def tangent_tetra(points: np.ndarray) -> np.ndarray:
    """Return curl(0,0,x²y²z²(1-x-y-z)²), tangent to all simplex faces."""
    x, y, z = points.T
    w = 1 - x - y - z
    return np.column_stack(
        (2 * x**2 * y * z**2 * w * (w - y), -2 * x * y**2 * z**2 * w * (w - x), 0 * x)
    )


def setup(kind: str) -> tuple[Any, Any, int, Any]:
    """Select original triangular/polygonal faces without changing their volume spaces."""
    if kind == "tetrahedron":
        mesh = TetraMesh.unit_cube(1)
        return mesh, TriangularSkeleton(mesh, degree=1), 2, solve_rad_3d
    mesh = PolyhedralMesh.cubes()
    return mesh, PolygonalSkeleton3D(mesh, degree=1), 1, solve_polyhedral_rad


@pytest.mark.parametrize("kind", ["tetrahedron", "polyhedron"])
@pytest.mark.parametrize("stabilization", ["galerkin", "supg"])
@pytest.mark.parametrize("coarse_space", ["constants", "kernel"])
@pytest.mark.parametrize("boundary", ["dirichlet", "mixed", "natural"])
def test_user_equations_match_implemented_coefficient_fields(
    kind: str, stabilization: str, coarse_space: str, boundary: str
) -> None:
    """Match variable tensor derivatives, all oriented boundaries and retained coordinates."""
    mesh, skeleton, refinement, reference = setup(kind)

    # Independently apply -div(K grad u)+div(beta u)+.5 u to the affine u.
    def source(points: np.ndarray) -> np.ndarray:
        return -1 + points @ np.array([1, 2, -1]) + 3.5 * affine(points)

    natural: dict[int, Any] = {}
    faces = mesh.boundary_faces[:1] if boundary == "mixed" else mesh.boundary_faces
    if boundary != "dirichlet":
        for face in faces:
            normal = mesh.normals[face].copy()
            natural[int(face)] = lambda points, normal=normal: (
                (
                    -conductivity(points)[:, None] * np.array([1, 2, -1])
                    + velocity(points) * affine(points)[:, None] / 2
                )
                @ normal
            )
    options = dict(
        skeleton=skeleton,
        degree=3,
        local_refinement=refinement,
        diffusion=conductivity,
        diffusion_divergence=[1, 0, 0],
        velocity=velocity,
        velocity_divergence=3,
        reaction=0.5,
        source=source,
        dirichlet=affine,
        neumann=natural,
        coarse_space=coarse_space,
        stabilization=stabilization,
    )
    expected = reference(mesh, **options)
    actual = transport_3d(mesh, **options)
    np.testing.assert_allclose(actual.hybrid.trace, expected.hybrid.trace, atol=1e-12, rtol=1e-10)
    for left, right in zip(actual.values, expected.values, strict=True):
        np.testing.assert_allclose(left, right, atol=1e-12, rtol=1e-10)
    assert actual.l2_error(affine) < 1e-10
    assert actual.h1_seminorm_error([1, 2, -1]) < 1e-9
    # Named generic reconstruction is available without the physical convenience record.
    view = actual.hybrid.field("scalar")[0]
    point = view.mesh.points[view.mesh.cells[0]].mean(axis=0)[None]
    np.testing.assert_allclose(
        np.ravel(view.evaluate(point)), affine(point), atol=1e-12, rtol=1e-10
    )


@pytest.mark.parametrize("kind", ["tetrahedron", "polyhedron"])
@pytest.mark.parametrize("coarse_space", ["constants", "kernel"])
def test_pure_diffusion_full_physical_mean_and_mixed_face_spaces(
    kind: str, coarse_space: str
) -> None:
    mesh, skeleton, refinement, reference = setup(kind)
    if kind == "tetrahedron":
        degrees = np.arange(len(mesh.faces)) % 2
        skeleton = TriangularSkeleton(mesh, subdivisions=1 + degrees, degree=degrees)
    prescribed = {
        int(face): float(-mesh.normals[face] @ [1, 2, -1]) for face in mesh.boundary_faces
    }
    options = dict(
        degree=3,
        skeleton=skeleton,
        local_refinement=refinement,
        neumann=prescribed,
        mean_value=2,
        coarse_space=coarse_space,
    )
    expected = reference(mesh, **options)
    actual = transport_3d(mesh, **options)
    for left, right in zip(actual.values, expected.values, strict=True):
        np.testing.assert_allclose(left, right, atol=1e-12, rtol=1e-10)
    assert actual.l2_error(affine) < 1e-10
    definition = define_transport_3d(mesh, **options)
    assert definition.problem.coarse_sizes == (1,) * len(mesh.cells)


@pytest.mark.parametrize("kind", ["tetrahedron", "polyhedron"])
def test_true_tangent_kernel_retains_constant_and_compatible_gauge(kind: str) -> None:
    if kind == "tetrahedron":
        mesh = TetraMesh(
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float), [[0, 1, 2, 3]]
        )
        beta, reference = tangent_tetra, solve_rad_3d
    else:
        mesh = PolyhedralMesh.cubes()
        beta, reference = tangent_cube, solve_polyhedral_rad
    options = dict(
        degree=3,
        local_refinement=2,
        velocity=beta,
        velocity_divergence=0,
        neumann=dict.fromkeys(map(int, mesh.boundary_faces), 0),
        mean_value=2,
        coarse_space="kernel",
    )
    expected = reference(mesh, **options)
    actual = transport_3d(mesh, **options)
    assert len(actual.hybrid.coarse[0]) == 1
    np.testing.assert_allclose(actual.values[0], expected.values[0], atol=1e-12, rtol=1e-10)
    assert actual.l2_error(2) < 1e-10


@pytest.mark.parametrize("kind", ["tetrahedron", "polyhedron"])
@pytest.mark.parametrize("scale", [1e-8, 1e-16])
@pytest.mark.parametrize("precision_capability", ["native", "binary64"])
def test_arbitrarily_small_transport_does_not_change_retained_count(
    monkeypatch: pytest.MonkeyPatch, kind: str, scale: float, precision_capability: str
) -> None:
    """Physical constant retention is independent of extended-precision availability."""
    if precision_capability == "binary64":
        from pymhm.linalg import linear

        monkeypatch.setattr(np, "longdouble", np.float64)
        monkeypatch.setattr(np, "clongdouble", np.complex128)
        monkeypatch.setattr(linear, "_EXTENDED_PRECISION", False)

    wider = np.finfo(np.longdouble).eps < np.finfo(np.float64).eps
    mesh, skeleton, refinement, reference = setup(kind)
    options = dict(
        degree=3,
        skeleton=skeleton,
        local_refinement=refinement,
        velocity=[scale, 0, 0],
        reaction=scale,
        source=lambda p: scale * (1 + affine(p)),
        dirichlet=affine,
        global_refinement_precision="extended" if wider else "double",
    )
    actual = transport_3d(mesh, **options)
    expected = reference(mesh, **options)
    for left, right in zip(actual.values, expected.values, strict=True):
        np.testing.assert_allclose(left, right, atol=1e-12, rtol=1e-10)
    assert all(len(coarse) == 1 for coarse in actual.hybrid.coarse)
    assert actual.l2_error(affine) < 1e-12
    assert actual.hybrid.raw_residual < 1e-12
    if not wider:
        explicit_extended = options | {"global_refinement_precision": "extended"}
        for solve in (transport_3d, reference):
            with pytest.raises(SolverUnavailableError, match="wider long-double"):
                solve(mesh, **explicit_extended)


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_coarse_plan_assembles_each_volume_operator_once(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    mesh = PolyhedralMesh.cubes()
    options = dict(degree=3, velocity=[1, 0, 0], source=1, dirichlet=affine, coarse_space="kernel")
    calls: list[int] = []
    actual_operator = scalar_3d.tetra_transport_operators

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return actual_operator(*args, **kwargs)

    monkeypatch.setattr(scalar_3d, "tetra_transport_operators", counted)
    definition = define_transport_3d(mesh, **options)
    assert definition.problem.coarse_sizes == (0,)
    assert not calls
    serial = transport_3d(mesh, **options)
    assert len(calls) == len(mesh.cells)
    parallel = transport_3d(mesh, **options, backend=backend, workers=2)
    np.testing.assert_allclose(parallel.values[0], serial.values[0], atol=1e-12, rtol=1e-10)
    assert not parallel.hybrid.coarse[0].size


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_original_conforming_matrix_and_exterior_nodal_elimination(degree: int) -> None:
    mesh = TetraMesh.unit_cube(1)
    options = dict(
        degree=degree,
        diffusion=conductivity,
        diffusion_divergence=[1, 0, 0],
        velocity=velocity,
        velocity_divergence=3,
        reaction=0.5,
        source=lambda p: -1 + p @ np.array([1, 2, -1]) + 3.5 * affine(p),
        dirichlet=affine,
        stabilization="supg",
    )
    actual = conforming_transport_3d(mesh, **options)
    expected = solve_rad_3d_conforming(mesh, **options)
    np.testing.assert_allclose(actual.values[0], expected.values[0], atol=1e-12, rtol=1e-10)
    _, nodes = tetra_nodal_space(mesh, degree)
    exterior = np.flatnonzero(np.any((nodes == 0) | (nodes == 1), axis=1))
    np.testing.assert_array_equal(tetra_boundary_nodes(mesh, degree), exterior)
    assert actual.l2_error(affine) < 1e-10


@pytest.mark.parametrize("kind", ["tetrahedron", "polyhedron"])
def test_gauge_excluded_data_remain_explicit(kind: str) -> None:
    mesh, skeleton, refinement, _ = setup(kind)
    options = dict(degree=3, skeleton=skeleton, local_refinement=refinement)
    with pytest.raises(ValueError, match="mean_value"):
        transport_3d(mesh, **options, mean_value=1)
    with pytest.raises(ValueError, match="incompatible|compatibility"):
        transport_3d(
            mesh, **options, neumann=dict.fromkeys(map(int, mesh.boundary_faces), 0), source=1
        )
    for coefficient, message in [
        ({"velocity": velocity}, "velocity_divergence"),
        ({"diffusion": conductivity, "stabilization": "supg"}, "diffusion_divergence"),
        ({"coarse_space": "bad"}, "coarse_space"),
        ({"mean_value": np.nan}, "finite"),
        ({"local_refinement": 3}, "power of two"),
    ]:
        with pytest.raises(ValueError, match=message):
            transport_3d(mesh, **(options | coefficient))


def test_galerkin_first_derivatives_preserve_the_original_volume_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unused second derivative does not change Galerkin coefficient rows or loads."""
    from pymhm.fem.scalar import transport_3d as volume
    from pymhm.fem.scalar.tetrahedron import tetra_element_tabulate

    mesh = TetraMesh.unit_cube(1)
    inputs = dict(degree=3, diffusion=0.1, velocity=[1, 0, 0], source=affine, order=6)
    original = volume.tetra_transport_operators(mesh, **inputs)

    def full_table(*args: Any, **kwargs: Any) -> Any:
        dofs, nodes, basis, gradient, _ = tetra_element_tabulate(*args, **kwargs)
        return dofs, nodes, basis, gradient

    monkeypatch.setattr(volume, "tetra_tabulate", full_table)
    with_full_derivative = volume.tetra_transport_operators(mesh, **inputs)
    np.testing.assert_array_equal(original[0].toarray(), with_full_derivative[0].toarray())
    np.testing.assert_array_equal(original[1], with_full_derivative[1])
