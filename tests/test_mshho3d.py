"""Three-dimensional moment reconstruction, boundaries, gauges and convergence."""

import numpy as np
import pytest

from pymhm.darcy3d import TriangularSkeleton
from pymhm.mshho3d import solve_mshho_3d
from pymhm.polyhedral import PolyhedralMesh
from pymhm.polyhedral_rad import PolygonalSkeleton3D
from pymhm.tetrahedral import TetraMesh


def affine(x):
    """Affine pressure of volume mean two on the unit cube."""
    return 1 + x[:, 0] + 2 * x[:, 1] - x[:, 2]


@pytest.mark.parametrize("polyhedral", [False, True])
@pytest.mark.parametrize("boundary", ["dirichlet", "mixed", "neumann"])
def test_anisotropic_affine_patch(polyhedral, boundary):
    """Original macrofaces and physical conormal orientations reproduce an anisotropic field."""
    mesh = PolyhedralMesh.cubes() if polyhedral else TetraMesh.unit_cube()
    material = np.array([[2, 0.3, 0.1], [0.3, 1, -0.2], [0.1, -0.2, 3]])
    flux = -material @ np.array([1, 2, -1])
    faces = mesh.boundary_faces if boundary == "neumann" else mesh.boundary_faces[:1]
    natural = None if boundary == "dirichlet" else {int(f): flux @ mesh.normals[f] for f in faces}
    result = solve_mshho_3d(
        mesh,
        permeability=material,
        dirichlet=affine,
        neumann=natural,
        mean_pressure=2 if boundary == "neumann" else 0,
    )
    assert result.l2_error(affine) < 2e-11
    assert result.flux_l2_error(flux) < 2e-10
    assert result.residual < 1e-12


@pytest.mark.parametrize("polyhedral", [False, True])
@pytest.mark.parametrize("variant", ["projected", "reconstructed"])
def test_quadratic_patch_with_source_and_nonzero_flux(polyhedral, variant):
    """Volume source and constant planar normal derivatives give exact quadratic energy lifts."""
    mesh = PolyhedralMesh.cubes() if polyhedral else TetraMesh.unit_cube()
    natural = {
        int(f): -2 * mesh.points[mesh.faces[f]].mean(axis=0) @ mesh.normals[f]
        for f in mesh.boundary_faces
    }
    result = solve_mshho_3d(
        mesh,
        source=-6,
        neumann=natural,
        mean_pressure=1,
        source_variant=variant,
    )
    assert result.l2_error(lambda p: np.sum(p * p, axis=1)) < 2e-11
    assert result.flux_l2_error(lambda p: -2 * p) < 2e-10


@pytest.mark.parametrize("polyhedral", [False, True])
def test_face_linear_moments_and_face_only_variant(polyhedral):
    """Independent polynomial face moments retain the affine patch without cell unknowns."""
    mesh = PolyhedralMesh.cubes() if polyhedral else TetraMesh.unit_cube()
    skeleton = (
        PolygonalSkeleton3D(mesh, degree=1) if polyhedral else TriangularSkeleton(mesh, degree=1)
    )
    result = solve_mshho_3d(
        mesh,
        skeleton=skeleton,
        cell_degree=-1,
        degree=3,
        source_variant="reconstructed",
        dirichlet=affine,
    )
    assert result.l2_error(affine) < 3e-11
    assert all(len(m) == 0 for m in result.cell_moments)


@pytest.mark.parametrize("m", [0, 1])
def test_manufactured_sine_refinement(m):
    """Refining macrocells reduces both pressure and raw flux errors with fixed local controls."""

    def exact(p):
        """Evaluate the smooth homogeneous-boundary solution."""
        return np.prod(np.sin(np.pi * p), axis=1)

    def flux(p):
        """Differentiate each tensor-product factor independently."""
        return -np.pi * np.column_stack(
            [
                np.cos(np.pi * p[:, i])
                * np.prod(np.sin(np.pi * p[:, [j for j in range(3) if j != i]]), axis=1)
                for i in range(3)
            ]
        )

    errors = []
    for n in (1, 2):
        result = solve_mshho_3d(
            TetraMesh.unit_cube(n), cell_degree=m, source=lambda p: 3 * np.pi**2 * exact(p)
        )
        errors.append([result.l2_error(exact), result.flux_l2_error(flux)])
    assert np.all(np.array(errors[1]) < 0.85 * np.array(errors[0]))


def test_invalid_spaces_material_and_boundary_contracts():
    """Fail explicitly for incompatible data, insufficient local moments and wrong geometry."""
    mesh = TetraMesh.unit_cube()
    with pytest.raises(TypeError, match="requires"):
        solve_mshho_3d(None)
    for options in (
        {"cell_degree": -1},
        {"source_variant": "bad"},
        {"mean_pressure": np.nan},
        {"mean_pressure": 1},
    ):
        with pytest.raises(ValueError, match="variant|mean_pressure"):
            solve_mshho_3d(mesh, **options)
    with pytest.raises(ValueError, match="skeleton"):
        solve_mshho_3d(mesh, skeleton=TriangularSkeleton(TetraMesh.unit_cube()))
    with pytest.raises(ValueError, match="exterior"):
        solve_mshho_3d(mesh, neumann={len(mesh.faces): 0})
    with pytest.raises(ValueError, match="incompatible"):
        solve_mshho_3d(mesh, source=1, neumann={int(f): 0 for f in mesh.boundary_faces})
    with pytest.raises(ValueError, match="independent"):
        solve_mshho_3d(mesh, local_refinement=1, degree=1)
