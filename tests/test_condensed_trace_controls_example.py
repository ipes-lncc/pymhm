"""Prepared response restriction changes only the global trace coordinate space."""

import importlib
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from pymhm._legacy.models.transport.rad import _rad_local, solve_rad
from pymhm._legacy.models.transport.stabilization import UnusualParameters
from pymhm.core.contracts import LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh
from pymhm.methods.petrov_galerkin import solve_pgmhm


def helper(monkeypatch):
    """Import original study operations without exporting them as core FEM APIs."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("examples.condensed_trace_controls")


@pytest.mark.parametrize("inhomogeneous", [False, True])
def test_restriction_matches_fresh_pgmhm_without_copying_responses(monkeypatch, inhomogeneous):
    """Nonzero polynomial BCs, penalty and conservative residual enrichment all restrict."""
    module = helper(monkeypatch)
    mesh = TriangleMesh.unit_square()
    coarse = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, 1) for _ in mesh.faces))
    fine = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, 2) for _ in mesh.faces))

    def g(points):
        """Use a quadratic boundary projection exact in the common fine-face quadrature."""
        return 1 + points[:, 0] ** 2 + 2 * points[:, 1] if inhomogeneous else np.zeros(len(points))

    natural = {int(face): 0.0 for face in mesh.boundary_faces if abs(mesh.normals[face, 0]) > 0.5}
    options = dict(
        source=1.0,
        dirichlet=g,
        neumann=natural,
        stabilization_parameter=0.1,
        local_refinement=4,
        permeability=np.array([[2.0, 0.25], [0.25, 1.0]]),
        quadrature_order=6,
    )
    large = solve_pgmhm(mesh, skeleton=fine, **options)
    small = solve_pgmhm(mesh, skeleton=coarse, **options)
    _, fixed = boundary_data(coarse, 0.0, natural, order=6)
    injection = module.p0_injection(coarse, fine)
    narrowed, trace, count = module.restrict_pgmhm(large, injection, fixed=fixed)
    assert narrowed.system is large.system
    assert narrowed.system.responses is large.system.responses
    assert count == len(small.system.rhs)
    assert_allclose(trace, small.hybrid.trace, atol=3e-13, rtol=3e-13)
    assert_allclose(narrowed.hybrid.trace, injection @ trace, atol=0, rtol=0)
    for actual, expected in zip(narrowed.pressure, small.pressure, strict=True):
        assert_allclose(actual, expected, atol=3e-13, rtol=3e-13)
    for actual, expected in zip(narrowed.enriched_pressure, small.enriched_pressure, strict=True):
        assert_allclose(actual, expected, atol=3e-13, rtol=3e-13)
    assert_allclose(narrowed.conservation_residuals(), 0, atol=2e-13, rtol=0)


@pytest.mark.parametrize("inhomogeneous", [False, True])
def test_projected_unusual_matches_fresh_local_and_trace_spaces(monkeypatch, inhomogeneous):
    """Unchanged UNUSUAL response lifts reproduce a fresh smaller trace with mixed BCs."""
    module = helper(monkeypatch)
    mesh = TriangleMesh.unit_square()
    coarse = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, 1) for _ in mesh.faces))
    fine = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, 2) for _ in mesh.faces))

    def material(points):
        """Use a resolved scalar material jump in the identical local grids."""
        return np.where(points[:, 0] < 0.5, 0.01, 2.0)

    def datum(points):
        """Distinguish homogeneous from nonhomogeneous exterior moments."""
        return 1 + points[:, 0] + 2 * points[:, 1] if inhomogeneous else np.zeros(len(points))

    natural = {int(f): 0.125 for f in mesh.boundary_faces if abs(mesh.normals[f, 0]) > 0.5}
    parameters = UnusualParameters(diffusion_lower=material, reaction_upper=1.0)
    boundary, fixed_large = boundary_data(fine, datum, natural, order=6)
    factory = partial(
        _rad_local,
        mesh=mesh,
        skeleton=fine,
        degree=1,
        refinement=4,
        diffusion=material,
        diffusion_divergence=(0.0, 0.0),
        velocity=(0.0, 0.0),
        velocity_divergence=0.0,
        reaction=1.0,
        source=0.75,
        stabilization="unusual",
        unusual_parameters=parameters,
        order=6,
    )
    system = HybridSystem.from_local_factory(
        factory, range(len(mesh.cells)), boundary_load=boundary
    )
    system.solve(fixed=fixed_large)
    _, fixed = boundary_data(coarse, datum, natural, order=6)
    result, trace, _ = module.projected_solve(
        system, module.p0_injection(coarse, fine), fixed=fixed
    )
    direct = solve_rad(
        mesh,
        skeleton=coarse,
        degree=1,
        local_refinement=4,
        diffusion=material,
        diffusion_divergence=(0.0, 0.0),
        reaction=1.0,
        source=0.75,
        dirichlet=datum,
        neumann=natural,
        stabilization="unusual",
        unusual_parameters=parameters,
        quadrature_order=6,
    )
    assert_allclose(trace, direct.hybrid.trace, atol=3e-13, rtol=3e-13)
    for actual, expected in zip(result.fields, direct.values, strict=True):
        assert_allclose(actual, expected, atol=3e-13, rtol=3e-13)


def test_projected_gauge_and_input_contracts(monkeypatch):
    """Constraint rows retain their full-coordinate meaning; inconsistent gauges fail."""
    module = helper(monkeypatch)
    system = HybridSystem(
        [LocalProblem(np.eye(2), np.array([[1.0, -1.0], [0.0, 0.0]]), np.zeros(2), [0, 1])]
    )
    value, trace, count = module.projected_solve(
        system, sparse.eye(2), constraints=[(np.ones(2), 4.0)]
    )
    assert_allclose(trace, [2.0, 2.0], atol=1e-14)
    assert count == 2 and value.residual < 1e-14
    nonsingular = HybridSystem([LocalProblem(np.eye(2), np.eye(2), np.ones(2), [0, 1])])
    with pytest.raises(ValueError, match="physical equations"):
        module.projected_solve(nonsingular, sparse.eye(2), constraints=[(np.ones(2), 4.0)])
    for injection in (
        sparse.eye(3),
        sparse.csr_matrix((2, 0)),
        sparse.csr_matrix([[np.nan], [1.0]]),
    ):
        with pytest.raises(ValueError, match="injection"):
            module.projected_solve(system, injection)
    for fixed in ({-1: 0.0}, {2: 0.0}, {0: np.nan}):
        with pytest.raises(ValueError, match="fixed"):
            module.projected_solve(system, sparse.eye(2), fixed=fixed)
    mesh = TriangleMesh.unit_square()
    small = SkeletonSpace(mesh)
    for other in (SkeletonSpace(TriangleMesh.unit_square()), SkeletonSpace(mesh, components=2)):
        with pytest.raises(ValueError, match="scalar"):
            module.p0_injection(small, other)
    for degree, continuous in ((1, False), (1, True)):
        other = SkeletonSpace(
            mesh, tuple(FaceSpace.uniform(degree, continuous=continuous) for _ in mesh.faces)
        )
        with pytest.raises(ValueError, match="P0"):
            module.p0_injection(small, other)
    nonnested = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, 3) for _ in mesh.faces))
    with pytest.raises(ValueError, match="breakpoint"):
        module.p0_injection(nonnested, small)
