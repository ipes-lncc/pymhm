"""General constitutive stiffness, polynomial displacement and physical traction gauges."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.elasticity.primal import _KELVIN, constitutive_values
from pymhm._legacy.models.vector import solve_elasticity
from pymhm.linalg.linear import LinearSolveError

STIFFNESS = np.array([[5.0, 1.0, 0.4], [1.0, 4.0, 0.3], [0.4, 0.3, 2.0]])
STRAIN = np.array([1.0, 3.0, 1 / np.sqrt(2)])
STRESS = np.einsum("a,aij->ij", STIFFNESS @ STRAIN, _KELVIN)


def displacement(points: np.ndarray) -> np.ndarray:
    """Affine displacement with all normal and shear components nonzero."""
    x, y = points.T
    return np.column_stack((x + 2 * y, -x + 3 * y))


@pytest.mark.parametrize("degree,refinement", [(1, 4), (2, 2), (3, 1), (4, 1)])
def test_variable_anisotropic_tensor_affine_patch(degree: int, refinement: int) -> None:
    """Material gradients contribute to body force even for affine displacement."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
    result = solve_elasticity(
        mesh,
        formulation="primal",
        degree=degree,
        local_refinement=refinement,
        skeleton=skeleton,
        dirichlet=displacement,
        constitutive=lambda x: (1 + x[:, 0] + 2 * x[:, 1])[:, None, None] * STIFFNESS,
        source=-STRESS @ np.array([1.0, 2.0]),
    )
    assert result.l2_error(displacement) < 2e-11
    assert (
        result.stress_l2_error(lambda x: (1 + x[:, 0] + 2 * x[:, 1])[:, None, None] * STRESS)
        < 2e-10
    )
    assert_allclose(
        result.gradient(0, np.array([[0.2, 0.3, 0.5]])),
        np.broadcast_to([[1.0, 2.0], [-1.0, 3.0]], (len(result.local_meshes[0].cells), 1, 2, 2)),
        atol=2e-11,
    )


def test_kelvin_and_cartesian_tensor_describe_identical_energy() -> None:
    """The orthonormal shear convention preserves the physical tensor inner product."""
    cartesian = np.einsum("aij,ab,bkl->ijkl", _KELVIN, STIFFNESS, _KELVIN)
    assert_allclose(
        constitutive_values(cartesian, np.zeros((2, 2))),
        np.broadcast_to(STIFFNESS, (2, 3, 3)),
        atol=2e-15,
    )
    result = solve_elasticity(
        TriangleMesh.unit_square(),
        formulation="primal",
        degree=3,
        constitutive=cartesian,
        dirichlet=displacement,
        local_refinement=1,
    )
    assert result.l2_error(displacement) < 1e-11
    assert result.stress_l2_error(STRESS) < 1e-10


def test_pure_traction_prescribes_physical_rigid_moments() -> None:
    """Three rigid modes, rather than a pointwise pin, set the displacement gauge."""
    mesh = TriangleMesh.unit_square()
    traction = {int(face): STRESS @ mesh.normals[face] for face in mesh.boundary_faces}
    result = solve_elasticity(
        mesh,
        formulation="primal",
        constitutive=STIFFNESS,
        neumann=traction,
        rigid_moments=(1.5, 1.0, -0.25),
        local_refinement=4,
    )
    assert result.l2_error(displacement) < 1e-11
    assert result.stress_l2_error(STRESS) < 1e-10


def test_partially_prescribed_traction_matches_affine_field() -> None:
    """The hybrid multiplier remains negative physical Cauchy traction."""
    mesh = TriangleMesh.unit_square()
    face = int(mesh.boundary_faces[0])
    result = solve_elasticity(
        mesh,
        formulation="primal",
        constitutive=STIFFNESS,
        neumann={face: STRESS @ mesh.normals[face]},
        dirichlet=displacement,
    )
    assert result.l2_error(displacement) < 1e-11


def test_underresolved_single_triangle_trace_is_rejected() -> None:
    """Primal P2 and odd P1 trace fail the L17 sufficient compatibility condition."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
    with pytest.raises(LinearSolveError, match="rank"):
        solve_elasticity(
            mesh, formulation="primal", degree=2, local_refinement=1, skeleton=skeleton
        )


@pytest.mark.parametrize(
    "material,match",
    [
        (np.eye(3) * 1j, "real"),
        (np.full((3, 3), np.inf), "finite"),
        (np.ones((2, 2)), "shape"),
        (np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]), "symmetric"),
        (np.diag([1.0, 0.0, 1.0]), "positive definite"),
    ],
)
def test_constitutive_contract(material: np.ndarray, match: str) -> None:
    """Reject tensors violating the elasticity energy assumptions."""
    with pytest.raises(ValueError, match=match):
        constitutive_values(material, np.zeros((1, 2)))


def test_cartesian_minor_symmetry_is_not_silently_projected() -> None:
    """A tensor acting on skew strains is not an elasticity stiffness contract."""
    material = np.einsum("aij,ab,bkl->ijkl", _KELVIN, STIFFNESS, _KELVIN)
    material[0, 1, 0, 0] += 0.1
    with pytest.raises(ValueError, match="symmetries"):
        constitutive_values(material, np.zeros((1, 2)))


@pytest.mark.parametrize("moments", [(0, 0), [np.nan, 0, 0], [0j, 0, 0]])
def test_traction_gauge_requires_three_real_moments(moments) -> None:
    """Reject an incomplete, complex or nonfinite rigid-motion constraint."""
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="rigid_moments"):
        solve_elasticity(
            mesh,
            formulation="primal",
            neumann={int(f): (0, 0) for f in mesh.boundary_faces},
            rigid_moments=moments,
            local_refinement=1,
        )


@pytest.mark.parametrize("options", [dict(mean_pressure=1.0), dict(stabilization_alpha=0.1)])
def test_primal_does_not_accept_pressure_or_gals_controls(options: dict) -> None:
    """Unsupported formulation controls must not be ignored."""
    with pytest.raises(ValueError, match="primal elasticity"):
        solve_elasticity(TriangleMesh.unit_square(), formulation="primal", **options)


def test_general_tensor_is_explicitly_primal_only() -> None:
    """Herrmann isotropic splitting must not silently reinterpret a general stiffness."""
    with pytest.raises(ValueError, match="constitutive"):
        solve_elasticity(TriangleMesh.unit_square(), constitutive=STIFFNESS)


def test_error_tensor_validation() -> None:
    """Nonfinite and complex analytical tensors cannot define physical error norms."""
    result = solve_elasticity(TriangleMesh.unit_square(), formulation="primal", local_refinement=1)
    for target in (np.full((2, 2), np.nan), np.ones((2, 2)) * 1j):
        with pytest.raises(ValueError, match="exact stress"):
            result.stress_l2_error(target)


@pytest.mark.parametrize("degree", [2, 4])
def test_minimal_polynomial_enrichment_restores_trace_injectivity(degree: int) -> None:
    """One added scalar mode makes the odd trace visible without using full P(k+1)."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(degree - 1) for _ in mesh.faces), 2)

    def exact(x: np.ndarray) -> np.ndarray:
        """Return a quadratic solenoidal displacement reproduced by the enriched space."""
        return np.column_stack((x[:, 0] ** 2, -2 * x[:, 0] * x[:, 1]))

    result = solve_elasticity(
        mesh,
        formulation="primal",
        degree=degree,
        minimal_enrichment=True,
        skeleton=skeleton,
        local_refinement=1,
        quadrature_order=degree + 3,
        dirichlet=exact,
        source=(-2.0, 0.0),
    )
    assert result.l2_error(exact, degree + 3) < 2e-10
    assert result.degree == degree + 1
    assert all(len(field) == (degree + 1) * (degree + 2) + 2 for field in result.hybrid.fields)
    assert all(len(values) == (degree + 2) * (degree + 3) // 2 for values in result.values)


@pytest.mark.parametrize("degree,refinement,trace", [(1, 1, 0), (2, 2, 1), (2, 1, 0)])
def test_minimal_enrichment_contract(degree: int, refinement: int, trace: int) -> None:
    """The one-mode construction has the explicitly stated single-cell odd-trace scope."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(trace) for _ in mesh.faces), 2)
    with pytest.raises(ValueError, match="minimal enrichment"):
        solve_elasticity(
            mesh,
            formulation="primal",
            degree=degree,
            minimal_enrichment=True,
            local_refinement=refinement,
            skeleton=skeleton,
        )


def test_minimal_embedding_requires_one_missing_trace_direction() -> None:
    """Do not silently choose one mode when the trace coupling has a larger nullspace."""
    from pymhm._legacy.models.elasticity.primal import _minimal_embedding

    mesh = TriangleMesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
    with pytest.raises(ValueError, match="exactly one"):
        _minimal_embedding(mesh, 2, np.zeros((10, 2)))


def test_minimal_embedding_rejects_multiple_fine_triangles() -> None:
    """The single polynomial enrichment is not silently applied to a composite local grid."""
    from pymhm._legacy.models.elasticity.primal import _minimal_embedding

    with pytest.raises(ValueError, match="one local triangle"):
        _minimal_embedding(TriangleMesh.unit_square(), 2, np.zeros((16, 2)))


def test_minimal_enrichment_requires_boolean() -> None:
    """An integer cannot silently alter the polynomial reconstruction degree."""
    with pytest.raises(ValueError, match="boolean"):
        solve_elasticity(TriangleMesh.unit_square(), formulation="primal", minimal_enrichment=2)


def test_cartesian_tensor_error_norm_integrates_unfitted_interfaces() -> None:
    """Cellwise quadrature preserves the exact stress norm for a layered stiffness."""
    from dataclasses import replace

    from pymhm.materials.cartesian import CartesianCellField

    mesh = TriangleMesh.unit_square()
    result = solve_elasticity(mesh, formulation="primal", local_refinement=1, dirichlet=(0, 0))
    values = np.broadcast_to(np.eye(3), (2, 1, 3, 3)).copy()
    values[0, 0] = np.array([[4, 1, 0], [1, 5, 0], [0, 0, 2]])
    values[1, 0] = np.array([[4, 2, 0], [2, 5, 0], [0, 0, 2]])
    material = CartesianCellField(values, spacing=(0.5, 1.0))
    result = replace(
        result,
        constitutive=material,
        values=tuple(
            np.column_stack((fine.points[:, 0], np.zeros(len(fine.points))))
            for fine in result.local_meshes
        ),
    )
    # epsilon_xx=1 gives sigma=diag(4,1) on half the domain and diag(4,2) on the other half.
    assert_allclose(result.stress_l2_error(np.zeros((2, 2)), order=3), np.sqrt(18.5), rtol=2e-14)
    assert_allclose(
        result.stress(0, np.array([[0.1, 0.2, 0.7]])),
        result.stress(0, np.array([[[0.1, 0.2, 0.7]]])),
        atol=1e-15,
    )


def test_minimal_enrichment_on_a_small_translated_triangle() -> None:
    """Reference polynomial injection preserves the missing trace mode away from the origin."""
    mesh = TriangleMesh([[0.8125, 0.625], [0.875, 0.625], [0.875, 0.6875]], [[0, 1, 2]])
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
    result = solve_elasticity(
        mesh,
        formulation="primal",
        degree=2,
        minimal_enrichment=True,
        local_refinement=1,
        skeleton=skeleton,
        dirichlet=displacement,
        constitutive=STIFFNESS,
    )
    assert result.l2_error(displacement) < 3e-13
    assert result.stress_l2_error(STRESS) < 2e-11
