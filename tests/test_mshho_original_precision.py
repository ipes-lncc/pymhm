"""Original Galerkin rows, high-contrast fields and dimension-independent moment arithmetic."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import pymhm.linalg.linear as solvers
from examples.mshho_field_archive import attach_mhm, field_arrays
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm.core.moments import energy_reconstruction as _energy_reconstruction
from pymhm.fem.scalar.tetrahedron import tetra_operators
from pymhm.linalg.linear import SolverUnavailableError
from pymhm.meshes.polyhedral import PolyhedralMesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.methods.hho import MsHHOLocal, _condense_moments, solve_mshho
from pymhm.methods.hho_3d import solve_mshho_3d

HAS_EXTENDED = np.finfo(np.longdouble).nmant > np.finfo(float).nmant


@pytest.mark.parametrize("size", [2, 3])
def test_represented_galerkin_reduction_preserves_antisymmetry(size: int) -> None:
    """The original bilinear operator is retained instead of replacing it with a Hessian."""
    matrix = np.eye(size)
    matrix[0, 1], matrix[1, 0] = 1e-8, -1e-8
    moments = np.eye(size)
    reconstruction, reduced = _energy_reconstruction(sparse.csc_matrix(matrix), moments)
    assert np.linalg.norm(matrix @ reconstruction - moments @ reduced) < 1e-14
    assert np.isclose(reduced[0, 1] - reduced[1, 0], 2e-8, rtol=0, atol=1e-14)


@pytest.mark.skipif(not HAS_EXTENDED, reason="Explicit extended storage is unavailable")
@pytest.mark.parametrize("contrast", [1.0, 1e3, 1e4, 1e6])
@pytest.mark.parametrize("nonzero_boundary", [False, True])
def test_whole_original_saddles_at_high_contrast(contrast: float, nonzero_boundary: bool) -> None:
    """Both complete fields solve the same original rows at the unchanged 1e-10 gate."""
    mesh = TriangleMesh.unit_square(2)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))

    def boundary(points: np.ndarray) -> np.ndarray:
        return points[:, 0] - points[:, 1] if nonzero_boundary else np.zeros(len(points))

    options = dict(
        skeleton=skeleton,
        permeability=np.diag([contrast, 1.0]),
        source=4.0,
        dirichlet=boundary,
        degree=3,
        local_refinement=2,
        quadrature_order=8,
        local_refinement_precision="extended",
    )
    primal = solve_mshho(mesh, cell_degree=1, **options)
    dual = solve_darcy(mesh, hybrid_refinement_steps=2, hybrid_refinement_min_steps=2, **options)
    arrays = field_arrays(primal, 4)
    diagnostics = attach_mhm(arrays, dual, boundary)
    for method in ("mhm", "mshho"):
        assert diagnostics[f"{method}_original_relative_residual"] <= 1e-10
    assert all(value.dtype != np.dtype(np.longdouble) for value in arrays.values())


@pytest.mark.skipif(not HAS_EXTENDED, reason="Explicit extended storage is unavailable")
@pytest.mark.parametrize("polyhedral", [False, True])
@pytest.mark.parametrize("boundary", ["dirichlet", "mixed", "neumann"])
def test_extended_tetrahedral_and_polyhedral_physical_patch(
    polyhedral: bool, boundary: str
) -> None:
    """The shared arithmetic preserves 3D moments, outward data and the physical mean."""
    mesh = PolyhedralMesh.cubes() if polyhedral else TetraMesh.unit_cube()
    material = np.diag([1000.0, 1.0, 3.0])

    def affine(points: np.ndarray) -> np.ndarray:
        return 1 + points[:, 0] + 2 * points[:, 1] - points[:, 2]

    flux = -material @ np.array([1, 2, -1])
    faces = mesh.boundary_faces if boundary == "neumann" else mesh.boundary_faces[:1]
    natural = None if boundary == "dirichlet" else {int(f): flux @ mesh.normals[f] for f in faces}
    result = solve_mshho_3d(
        mesh,
        permeability=material,
        dirichlet=affine,
        neumann=natural,
        mean_pressure=2 if boundary == "neumann" else 0,
        local_refinement_precision="extended",
    )
    assert result.l2_error(affine) < 3e-11
    assert result.flux_l2_error(flux) / np.linalg.norm(flux) < 1e-10
    for local in result.local:
        matrix, _, _ = tetra_operators(local.mesh, result.degree, diffusion=material, order=5)
        defect = matrix @ local.reconstruction - local.moments @ local.energy
        assert np.linalg.norm(defect) < 1e-10
        assert np.allclose(local.moments.T @ local.reconstruction, np.eye(local.moments.shape[1]))


@pytest.mark.parametrize("dimension", [2, 3])
def test_invalid_precision_is_rejected_before_local_assembly(dimension: int) -> None:
    """Dimension wrappers enforce the same admissible numerical precision contract."""
    with pytest.raises(ValueError, match="local_refinement_precision"):
        if dimension == 2:
            solve_mshho(TriangleMesh.unit_square(), local_refinement_precision="unknown")
        else:
            solve_mshho_3d(TetraMesh.unit_cube(), local_refinement_precision="unknown")


def test_extended_precision_has_no_binary64_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable requested representation cannot be silently substituted."""
    monkeypatch.setattr(solvers, "_EXTENDED_PRECISION", False)
    with pytest.raises(SolverUnavailableError, match="wider"):
        solve_mshho(TriangleMesh.unit_square(), local_refinement_precision="extended")


@pytest.mark.skipif(not HAS_EXTENDED, reason="Explicit extended storage is unavailable")
def test_extended_zero_source_and_face_only_field_are_exactly_zero() -> None:
    """Homogeneous data preserve the zero field without artificial mean or load corrections."""
    result = solve_mshho(
        TriangleMesh.unit_square(),
        cell_degree=-1,
        source_variant="reconstructed",
        local_refinement_precision="extended",
    )
    assert all(np.count_nonzero(field) == 0 for field in result.pressure)
    assert np.count_nonzero(result.face_moments) == 0


@pytest.mark.skipif(not HAS_EXTENDED, reason="Explicit extended storage is unavailable")
def test_physical_mean_retains_the_executed_particular_integral() -> None:
    """A gauge fixes the full field's physical integral, including wider source digits."""
    value = np.longdouble(1) + np.longdouble(2) ** -54
    local = MsHHOLocal(
        None,
        np.eye(2, dtype=np.longdouble),
        np.eye(2),
        np.array([[1, -1], [-1, 1]], dtype=np.longdouble),
        np.array([value, -value]),
        1,
        np.array([1, 0], dtype=np.longdouble),
    )
    skeleton = SimpleNamespace(size=1, cell_dofs=lambda _: np.array([0]))
    _, _, _, fields, _ = _condense_moments(
        (local,), skeleton, {}, np.zeros(1), np.ones(1), 0.0, "scipy", "extended"
    )
    assert local.integral @ fields[0] == 0


@pytest.mark.parametrize("minimum,maximum", [(-1, 2), (True, 2), (0.5, 2), (1, 0)])
def test_darcy_correction_bounds_are_checked_before_local_assembly(
    minimum: int, maximum: int
) -> None:
    """The Darcy wrapper preserves the generic correction-count contract."""
    with pytest.raises(ValueError, match="hybrid_refinement"):
        solve_darcy(None, hybrid_refinement_steps=maximum, hybrid_refinement_min_steps=minimum)
