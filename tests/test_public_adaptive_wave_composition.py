"""Adaptive policies consume actual user-declared states without selecting a prepared solver."""

from typing import Any

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.formulations.application import tetrahedral_darcy, transport
from pymhm.adaptivity.darcy_3d import solve_adaptive_darcy_3d
from pymhm.adaptivity.transport import TransportBounds, solve_adaptive_transport
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.solutions import Darcy3DSolution, ScalarSolution


def test_tetrahedral_adaptation_uses_public_equations_and_transfers_physical_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equivalent coefficients, indicators and inherited Neumann faces survive both states."""
    import pymhm.adaptivity.darcy_3d as owner

    mesh = TetraMesh.unit_cube()
    bottom = {
        int(face): 0.0
        for face in mesh.boundary_faces
        if np.all(mesh.points[mesh.faces[face], 2] == 0)
    }
    options = dict(
        iterations=2,
        degree=3,
        local_refinement=2,
        source=1.0,
        neumann=bottom,
        ellipticity_lower_bound=np.ones(len(mesh.cells)),
    )
    calls = []

    def callback(
        current: TetraMesh, *, skeleton: TriangularSkeleton, **problem: Any
    ) -> Darcy3DSolution:
        """Execute the application's explicit local energy and global conormal equations."""
        calls.append((current, skeleton, dict(problem)))
        return tetrahedral_darcy(current, skeleton=skeleton, **problem)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("adaptive public composition must use the declared solver")

    with threadpool_limits(1):
        expected = solve_adaptive_darcy_3d(mesh, **options)
        monkeypatch.setattr(owner, "solve_darcy_3d", forbidden)
        result = solve_adaptive_darcy_3d(mesh, solve_step=callback, **options)
    assert len(calls) == len(result.solutions) == 2
    np.testing.assert_allclose(result.totals, expected.totals, atol=1e-12, rtol=1e-10)
    for call, actual, reference, estimate in zip(
        calls, result.solutions, expected.solutions, result.estimators, strict=True
    ):
        current, skeleton, problem = call
        assert actual.skeleton is skeleton and skeleton.mesh is current
        assert problem["degree"] == 3 and problem["local_refinement"] == 2
        np.testing.assert_allclose(
            actual.hybrid.trace, reference.hybrid.trace, atol=1e-12, rtol=1e-10
        )
        for a, b in zip(actual.pressure, reference.pressure, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(actual.conservation_residuals(), 0, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(estimate.equilibrium_defect, 0, atol=1e-12, rtol=1e-10)
        natural = problem["neumann"]
        assert set(natural) == {
            int(face)
            for face in current.boundary_faces
            if np.all(current.points[current.faces[face], 2] == 0)
        }
    np.testing.assert_array_equal(result.marked[0], expected.marked[0])
    np.testing.assert_array_equal(
        result.refinements[0].cell_parents, expected.refinements[0].cell_parents
    )


@pytest.mark.parametrize("convention", ["neumann", "diffusive_flux"])
def test_transport_enrichment_uses_public_equations_with_declared_flux_convention(
    convention: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full public strong-boundary states preserve jumps, markings and unsmoothed local fields."""
    import pymhm.adaptivity.transport as owner

    mesh = TriangleMesh.unit_square(2)
    skeleton = SkeletonSpace(mesh)
    velocity = np.array([1.0, 0.2])

    def exact(points: np.ndarray) -> np.ndarray:
        """Declare a nonpolynomial scalar with zero diffusive flux on horizontal walls."""
        return np.sin(np.pi * points[:, 0])

    def source(points: np.ndarray) -> np.ndarray:
        """Differentiate -0.2 Delta(u)+beta.grad(u) independently of finite elements."""
        return 0.2 * np.pi**2 * exact(points) + np.pi * np.cos(np.pi * points[:, 0])

    natural = {}
    for face in mesh.boundary_faces:
        if abs(mesh.normals[face, 1]) == 1:
            normal_velocity = float(velocity @ mesh.normals[face])
            natural[int(face)] = (
                0.0
                if convention == "diffusive_flux"
                else lambda points, coefficient=normal_velocity: coefficient * exact(points) / 2
            )
    options = dict(
        iterations=3,
        diffusion=0.2,
        velocity=velocity,
        source=source,
        dirichlet=exact,
        degree=3,
        local_refinement=4,
        stabilization="supg",
        **{convention: natural},
    )
    bounds = TransportBounds(0.2, float(np.linalg.norm(velocity)), 0)
    calls = []

    def callback(
        current: TriangleMesh, *, skeleton: SkeletonSpace, **problem: Any
    ) -> ScalarSolution:
        """Execute the user's complete conservative SUPG and strong-boundary equations."""
        calls.append((current, skeleton, dict(problem)))
        return transport(current, skeleton=skeleton, **problem)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("adaptive public composition must use the declared solver")

    with threadpool_limits(1):
        expected = solve_adaptive_transport(skeleton, bounds, **options)
        monkeypatch.setattr(owner, "solve_transport", forbidden)
        result = solve_adaptive_transport(skeleton, bounds, solve_step=callback, **options)
    assert len(calls) == len(result.solutions) == 3
    assert result.stop_reason == expected.stop_reason
    assert result.solutions[-1].skeleton.size > skeleton.size
    for call, actual, reference, indicator, other_indicator in zip(
        calls,
        result.solutions,
        expected.solutions,
        result.indicators,
        expected.indicators,
        strict=True,
    ):
        current, trace, problem = call
        assert current is mesh and actual.skeleton is trace
        assert actual.strong_dirichlet and problem["dirichlet_enforcement"] == "strong"
        assert problem[convention] is natural
        for a, b in zip(actual.values, reference.values, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-10)
        for a, b in zip(indicator.values, other_indicator.values, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-10)
        for a, b in zip(indicator.mark(), other_indicator.mark(), strict=True):
            np.testing.assert_array_equal(a, b)
        for face in mesh.boundary_faces:
            assert trace.faces[face] == skeleton.faces[face]
            assert not np.any(indicator.values[face])
        for a, b in zip(actual.local_meshes, result.solutions[0].local_meshes, strict=True):
            np.testing.assert_array_equal(a.cells, b.cells)
            np.testing.assert_array_equal(a.points, b.points)


@pytest.mark.parametrize("callback", [False, 1, "assembled"])
def test_adaptive_callbacks_require_an_actual_callable(callback: Any) -> None:
    """Reject an invalid assembly selection before solving either physical problem."""
    with pytest.raises(TypeError, match="solve_step"):
        solve_adaptive_darcy_3d(TetraMesh.unit_cube(), solve_step=callback)
    with pytest.raises(TypeError, match="solve_step"):
        solve_adaptive_transport(
            SkeletonSpace(TriangleMesh.unit_square()), TransportBounds(1, 0, 0), solve_step=callback
        )
