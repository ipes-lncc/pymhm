"""Published face marking, local closure and actual adaptive Oseen solves."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.flow_adaptive import _aligned_refinement, adapt_flow, mark_flow_faces


def test_adaptive_pressure_problem_reduces_error_and_refines_only_selected_faces() -> None:
    """A quadratic pressure produces nonzero skeletal error and verifies solved adaptive states."""
    mesh = TriangleMesh.unit_square()
    result = adapt_flow(mesh, iterations=2, theta=0.75, drag=1.0, source=lambda x: 2 * x)
    errors = [
        solution.pressure_l2_error(lambda x: np.sum(x * x, axis=1) - 2 / 3)
        for solution in result.solutions
    ]
    assert result.stop_reason == "iterations"
    assert len(result.solutions) == 3
    assert errors[-1] < errors[0]
    assert result.estimators[-1].total < result.estimators[0].total
    assert all(solution.skeleton.mesh is mesh for solution in result.solutions)
    assert max(result.refinements[-1]) > max(result.refinements[0])
    assert result.solutions[-1].skeleton.size > result.solutions[0].skeleton.size
    assert max(solution.hybrid.residual for solution in result.solutions) < 1e-10


def test_exact_zero_stops_at_tolerance() -> None:
    """A zero source and boundary field must not trigger refinement on zero indicators."""
    result = adapt_flow(TriangleMesh.unit_square(), iterations=5, tolerance=1e-12)
    assert result.stop_reason == "tolerance"
    assert len(result.solutions) == 1
    marked, local = mark_flow_faces(result.estimators[0])
    assert not np.any(local)
    assert not any(np.any(value) for value in marked)


def test_resolution_limit_does_not_solve_an_unrequested_larger_grid() -> None:
    """A finite memory cap stops before the next grid, with an explicit reason."""
    result = adapt_flow(
        TriangleMesh.unit_square(), iterations=2, max_local_refinement=1, source=lambda x: 2 * x
    )
    assert result.stop_reason == "resolution_limit"
    assert len(result.solutions) == 1


def test_marking_formula_and_stokes_variant() -> None:
    """Check segment argmax selection and local-error dominance independently of the solve."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1, 2) for _ in mesh.faces), 2)
    result = adapt_flow(mesh, skeleton=skeleton, local_refinement=2, iterations=0)
    estimator = replace(
        result.estimators[0],
        face_squared=tuple(np.array([4.0, 1.0]) for _ in mesh.faces),
        local_squared=np.array([100.0, 0.0]),
    )
    for variant in ("oseen-2021", "stokes-brinkman-2021"):
        marked, local = mark_flow_faces(estimator, theta=0.1, variant=variant)
        assert all(np.array_equal(value, [True, False]) for value in marked)
        assert np.all(local)
    stokes = adapt_flow(
        mesh, iterations=0, formulation="usfem", estimator_variant="stokes-brinkman-2021"
    )
    assert stokes.estimators[0].second_level_scale == 1


def test_alignment_uses_all_skeletal_denominators() -> None:
    """Uniform local closure contains both third and half segments without moving a trace."""
    mesh = TriangleMesh.unit_square()
    faces = tuple(FaceSpace((0.0, 1 / 3, 0.5, 1.0), (1, 1, 1)) for _ in mesh.faces)
    assert _aligned_refinement(SkeletonSpace(mesh, faces, 2), 0, 2) == 6
    faces = tuple(FaceSpace((0.0, np.sqrt(2) / 2, 1.0), (1, 1)) for _ in mesh.faces)
    with pytest.raises(ValueError, match="rational"):
        _aligned_refinement(SkeletonSpace(mesh, faces, 2), 0, 2)


def test_marking_adds_local_norms_outside_the_segment_square_root() -> None:
    """Distinguish Eq.52's sum of norms from a root of their squared sum."""
    estimator = adapt_flow(TriangleMesh.unit_square(), iterations=0).estimators[0]
    mesh = estimator.solution.skeleton.mesh
    interior = int(np.flatnonzero(mesh.face_cells[:, 1] >= 0)[0])
    exterior = int(mesh.boundary_faces[0])
    first = tuple(
        np.array([1.0 if face == interior else (9.0 if face == exterior else 0.0)])
        for face in range(len(mesh.faces))
    )
    estimator = replace(estimator, face_squared=first, local_squared=np.ones(2))
    # exterior:3+1=4; interior:1+1+1=3. At theta=.7 both are marked.
    # The incorrect root-of-squares gives sqrt(10),sqrt(3), excluding interior.
    marked, _ = mark_flow_faces(estimator, theta=0.7)
    assert marked[interior][0] and marked[exterior][0]


def test_initial_local_mesh_must_resolve_skeleton_breakpoints() -> None:
    """A face-adaptive run starts from the stated matching local mesh contract."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1, 2) for _ in mesh.faces), 2)
    with pytest.raises(ValueError, match="skeletal breakpoint"):
        adapt_flow(mesh, skeleton=skeleton, local_refinement=1)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"theta": 0}, "theta"),
        ({"theta": np.nan}, "theta"),
        ({"tolerance": -1}, "tolerance"),
        ({"tolerance": np.inf}, "tolerance"),
        ({"traction": {0: (0, 0)}}, "Dirichlet"),
        ({"local_refinement": 2, "max_local_refinement": 1}, "initial refinement"),
        ({"iterations": -1}, "iterations"),
    ],
)
def test_adaptive_contracts(options: dict, match: str) -> None:
    """Reject invalid marking controls and unsupported boundary contracts."""
    with pytest.raises(ValueError, match=match):
        adapt_flow(TriangleMesh.unit_square(), **options)


def test_invalid_marking_arguments() -> None:
    """Direct marking validates the same bounds as the solve driver."""
    estimator = adapt_flow(TriangleMesh.unit_square(), iterations=0).estimators[0]
    with pytest.raises(ValueError, match="theta"):
        mark_flow_faces(estimator, theta=1)
    with pytest.raises(ValueError, match="variant"):
        mark_flow_faces(estimator, variant="other")
    assert_allclose(estimator.total, 0)
