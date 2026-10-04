"""Independent direct solves for nested nonsymmetric L11 trace restrictions."""

from dataclasses import replace

import numpy as np
import pytest

from examples.pgmhm_campaign import crisscross
from examples.transport_trace_family import TransportTraceFamily, gradient_projection_squared
from pymhm._legacy.models.transport.solver import solve_transport
from pymhm.core.system import HybridSystem
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh


def forcing(points):
    """Use a nonconstant source to exercise the reduced right-hand side."""
    return 1 + points[:, 0] + points[:, 1] ** 2


def boundary(points):
    """Use nonhomogeneous data on both essential boundaries."""
    return 2 + points[:, 0] + points[:, 1]


def test_dg0_gradient_projection_has_analytical_moments():
    """The enlarged DG0 space has zero constant error and known linear variance."""
    mesh = TriangleMesh.unit_square(1)
    for order in (3, 7):
        assert gradient_projection_squared(mesh, (2, 3), order) < 1e-28
        np.testing.assert_allclose(
            gradient_projection_squared(mesh, lambda x: x, order), 1 / 9, rtol=2e-14
        )


@pytest.mark.parametrize("data", [(1.0, 0.0), (forcing, boundary)])
def test_restricted_fields_equal_direct_nonsymmetric_solve(monkeypatch, data):
    """Both Schur couplings, source and essential data survive diag(T,I) restriction."""
    mesh = crisscross(2)
    source, dirichlet = data
    prepared = TransportTraceFamily.prepare(
        mesh, segments=2, local_refinement=4, source=source, dirichlet=dirichlet
    )
    systems = []
    original = HybridSystem.solve

    def capture(self, **kwargs):
        """Record shared-owner equations before the original global solve."""
        systems.append((self.matrix.copy(), self.rhs.copy()))
        return original(self, **kwargs)

    monkeypatch.setattr(HybridSystem, "solve", capture)
    for segments in (2, 1):
        skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, segments) for _ in mesh.faces))
        recovered, checks = prepared.solve(skeleton)
        direct = solve_transport(
            mesh,
            skeleton=skeleton,
            degree=1,
            diffusion=1,
            velocity=(1, 0),
            source=source,
            dirichlet=dirichlet,
            diffusive_flux=dict.fromkeys(prepared.natural_faces, 0.0),
            dirichlet_enforcement="strong",
            local_refinement=4,
            quadrature_order=5,
            coarse_space="constants",
        )
        np.testing.assert_allclose(recovered.values, direct.values, atol=3e-13, rtol=2e-13)
        np.testing.assert_allclose(recovered.hybrid.trace, direct.hybrid.trace, atol=8e-13)
        assert checks["projected_original_residual"] < 1e-12
        assert checks["maximum_original_local_residual"] < 1e-12
    assert (prepared.matrix != systems[0][0]).nnz == 0
    np.testing.assert_array_equal(prepared.rhs, systems[0][1])
    assert (prepared.matrix - prepared.matrix.T).nnz > 0
    assert {c.retained.shape[1] for c in prepared.cells} == {0, 1}


def test_spawn_and_nested_space_contract():
    """Spawn retains source coordinates and rejects foreign or unresolved trace partitions."""
    mesh = crisscross(1)
    parallel = TransportTraceFamily.prepare(mesh, segments=2, local_refinement=2, workers=2)
    serial = TransportTraceFamily.prepare(mesh, segments=2, local_refinement=2)
    a, _ = parallel.solve(SkeletonSpace(mesh))
    b, _ = serial.solve(SkeletonSpace(mesh))
    np.testing.assert_allclose(a.values, b.values, atol=8 * np.finfo(float).eps, rtol=0)
    with pytest.raises(ValueError, match="prepared mesh"):
        parallel.solve(SkeletonSpace(crisscross(1)))
    with pytest.raises(ValueError, match="prepared mesh"):
        parallel.solve(SkeletonSpace(mesh, components=2))
    incompatible = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, 3) for _ in mesh.faces))
    with pytest.raises(ValueError, match="breakpoint"):
        parallel.solve(incompatible)


@pytest.mark.parametrize("epsilon", [0, np.nan])
def test_invalid_diffusion_rejected(epsilon):
    """A nonpositive or nonfinite physical coefficient is not a supported L11 input."""
    with pytest.raises(ValueError, match="epsilon"):
        TransportTraceFamily.prepare(crisscross(1), segments=1, local_refinement=2, epsilon=epsilon)


def test_original_equation_gates_detect_perturbations(monkeypatch):
    """Neither global nor local recovery checks can accept a corrupted solution."""
    import examples.transport_trace_family as owner

    mesh = crisscross(1)
    family = TransportTraceFamily.prepare(mesh, segments=1, local_refinement=2)
    original = owner.solve_linear
    monkeypatch.setattr(owner, "solve_linear", lambda matrix, rhs: np.zeros(len(rhs)))
    with pytest.raises(ValueError, match="restricted original"):
        family.solve(SkeletonSpace(mesh))
    monkeypatch.setattr(owner, "solve_linear", original)
    cell = family.cells[0]
    changed = replace(cell, source=cell.source + np.arange(len(cell.source)))
    corrupt = replace(family, cells=(changed, *family.cells[1:]))
    with pytest.raises(ValueError, match="original local"):
        corrupt.solve(SkeletonSpace(mesh))
