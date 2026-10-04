"""Residual identities, exact integral normalizations and nonmatching local meshes."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.vector import solve_brinkman
from pymhm.estimators.flow import estimate_flow_error
from pymhm.fem.scalar.triangle import nodal_space


def zero_solution(degree: int = 2, subdivisions: int = 1) -> Any:
    """Build a zero field with nonmatching local boundary partitions."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1, subdivisions) for _ in mesh.faces), 2)
    return solve_brinkman(mesh, skeleton=skeleton, degree=degree, local_refinement=(3, 4))


@pytest.mark.parametrize("variant", ["oseen-2021", "stokes-brinkman-2021"])
def test_exact_affine_variable_transport_has_zero_residuals(variant: str) -> None:
    """Volume, interior fine-traction, macro-traction and boundary residuals all vanish."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1, 2) for _ in mesh.faces), 2)

    def exact(points: np.ndarray) -> np.ndarray:
        """Return a solenoidal affine field with nonzero Dirichlet values."""
        return points * [1, -1]

    variable = variant == "oseen-2021"
    beta = (lambda points: points) if variable else (0.0, 0.0)

    def force(points: np.ndarray) -> np.ndarray:
        """Derive the affine exact momentum source without using discrete operators."""
        return (3 if variable else 2) * exact(points) + [1, 2]

    solution = solve_brinkman(
        mesh,
        formulation="oseen" if variable else "usfem",
        degree=2,
        local_refinement=(3, 4),
        skeleton=skeleton,
        drag=2,
        source=force,
        dirichlet=exact,
        advection=beta,
        advection_divergence=2 if variable else None,
        advection_bound=np.sqrt(2) if variable else None,
    )
    estimator = estimate_flow_error(
        solution,
        drag=2,
        source=force,
        dirichlet=exact,
        advection=beta,
        full_dirichlet=True,
        variant=variant,
    )
    assert estimator.total < 1e-10
    assert (
        estimator.mixed_error(exact, np.diag([1, -1]), lambda x: x[:, 0] + 2 * x[:, 1] - 1.5)
        < 1e-10
    )


@pytest.mark.parametrize("subdivisions", [1, 2])
def test_macro_jump_counts_each_incident_cell_and_uses_original_face_length(
    subdivisions: int,
) -> None:
    """A unit jump has eta1²=2 exterior sides + twice(1/4) on the diagonal."""
    solution = zero_solution(subdivisions=subdivisions)
    values = tuple(
        np.tile([1.0, 0.0] if i == 0 else [0.0, 0.0], (len(u), 1))
        for i, u in enumerate(solution.values)
    )
    estimator = estimate_flow_error(replace(solution, values=values), full_dirichlet=True)
    assert_allclose(estimator.eta1**2, 2.5, atol=2e-14)
    assert estimator.eta2 < 2e-14


def test_constant_body_force_integral_and_degree_scaling() -> None:
    """Evaluate h_tau² ||f||² analytically and distinguish L14/L15 normalizations."""
    solution = zero_solution()
    expected = np.array(
        [
            np.sum(fine.areas * fine.lengths[fine.cell_faces].max(axis=1) ** 2) * 5
            for fine in solution.local_meshes
        ]
    )
    oseen = estimate_flow_error(solution, source=(1, 2), full_dirichlet=True)
    stokes = estimate_flow_error(
        solution, source=(1, 2), full_dirichlet=True, variant="stokes-brinkman-2021"
    )
    assert_allclose(oseen.volume_squared, expected, atol=2e-15)
    assert_allclose(oseen.local_squared, expected, atol=2e-15)
    assert_allclose(oseen.eta2, np.sqrt(expected.sum()) / 4, atol=2e-15)
    assert_allclose(stokes.eta2, np.sqrt(expected.sum()), atol=2e-15)
    # A nonzero exact field verifies the V-norm's diameter and physical area.
    assert_allclose(oseen.mixed_error((3, 4), np.zeros((2, 2)), 2), np.sqrt(25 / 2 + 4), rtol=2e-14)


def test_polynomial_volume_residual_includes_laplacian_and_gradient() -> None:
    """An independently prescribed quadratic velocity gives an exact zero momentum residual."""
    solution = zero_solution()
    values, pressures = [], []
    for fine in solution.local_meshes:
        points = nodal_space(fine, solution.degree)[1]
        values.append(np.column_stack((points[:, 0] ** 2, -2 * points[:, 0] * points[:, 1])))
        points = nodal_space(fine, solution.pressure_degree)[1]
        pressures.append(points.sum(axis=1))
    estimator = estimate_flow_error(
        replace(solution, values=tuple(values), pressure=tuple(pressures)),
        source=(-1, 1),
        full_dirichlet=True,
    )
    assert np.linalg.norm(estimator.volume_squared) < 2e-26
    assert np.linalg.norm(estimator.divergence_squared) < 2e-26


@pytest.mark.parametrize(
    "options,match",
    [
        ({"full_dirichlet": False}, "full_dirichlet"),
        ({"viscosity": 0}, "viscosity"),
        ({"variant": "other"}, "variant"),
        ({"variant": "stokes-brinkman-2021", "advection": (1, 0)}, "excludes advection"),
        ({"variant": "stokes-brinkman-2021", "advection": lambda x: x}, "excludes advection"),
        ({"quadrature_order": 1}, "quadrature_order"),
    ],
)
def test_estimator_contracts(options: dict[str, Any], match: str) -> None:
    """Do not silently apply the Dirichlet-only uniform-degree formula outside its API."""
    kwargs = dict(full_dirichlet=True)
    kwargs.update(options)
    with pytest.raises(ValueError, match=match):
        estimate_flow_error(zero_solution(), **kwargs)


def test_mixed_degree_and_absent_pressure_are_rejected() -> None:
    """The published uniform-ell scaling cannot be inferred for mixed face degrees."""
    solution = zero_solution()
    space = solution.skeleton
    faces = (FaceSpace(), *space.faces[1:])
    with pytest.raises(ValueError, match="uniform skeletal"):
        estimate_flow_error(
            replace(solution, skeleton=SkeletonSpace(space.mesh, faces, 2)), full_dirichlet=True
        )
    with pytest.raises(ValueError, match="velocity-pressure"):
        estimate_flow_error(replace(solution, pressure=()), full_dirichlet=True)


def test_invalid_local_refinement_vector() -> None:
    """Nonuniform refinement must specify every macrocell exactly once."""
    with pytest.raises(ValueError, match="one subdivision"):
        solve_brinkman(TriangleMesh.unit_square(), local_refinement=(2,))


def test_scoped_evaluation_tables_preserve_every_indicator_bit(monkeypatch):
    """Repeated edge evaluation reuses topology without changing arithmetic or input fields."""
    import pymhm.estimators.flow as module

    rng = np.random.default_rng(712)
    base = zero_solution()
    solution = replace(
        base,
        values=tuple(rng.normal(size=value.shape) for value in base.values),
        pressure=tuple(rng.normal(size=value.shape) for value in base.pressure),
    )
    evaluate = module._evaluate

    def uncached(solution, macro, points, cells, tables=None):
        """Execute the independently rebuilt geometry/topology path at every edge."""
        return evaluate(solution, macro, points, cells)

    monkeypatch.setattr(module, "_evaluate", uncached)
    expected = module.estimate_flow_error(solution, source=(1, 2), full_dirichlet=True)
    coarse = solution.skeleton.mesh
    for macro, fine in enumerate(solution.local_meshes):
        assert module._boundary_intervals(coarse, macro, fine) == module._boundary_intervals(
            coarse, macro, fine, coarse.lengths
        )

    def checked(solution, macro, points, cells, tables=None):
        """Require immutable call-scoped tables at every fine/macro-edge evaluation."""
        assert tables is not None
        assert all(not table.flags.writeable for table in tables)
        return evaluate(solution, macro, points, cells, tables)

    monkeypatch.setattr(module, "_evaluate", checked)
    nodal = module.nodal_space
    calls = []

    def counted(mesh, degree):
        """Count mesh-wide topological constructions in the estimator module."""
        calls.append((id(mesh), degree))
        return nodal(mesh, degree)

    monkeypatch.setattr(module, "nodal_space", counted)
    lengths, normals = TriangleMesh.lengths.fget, TriangleMesh.normals.fget
    geometry_calls = {"lengths": {}, "normals": {}}

    def counted_lengths(mesh):
        """Count full face-length arrays, including normalization of face normals."""
        counts = geometry_calls["lengths"]
        counts[id(mesh)] = counts.get(id(mesh), 0) + 1
        return lengths(mesh)

    def counted_normals(mesh):
        """Count full normal arrays independently of the number of integrated edges."""
        counts = geometry_calls["normals"]
        counts[id(mesh)] = counts.get(id(mesh), 0) + 1
        return normals(mesh)

    monkeypatch.setattr(TriangleMesh, "lengths", property(counted_lengths))
    monkeypatch.setattr(TriangleMesh, "normals", property(counted_normals))
    actual = module.estimate_flow_error(solution, source=(1, 2), full_dirichlet=True)
    assert len(calls) == 2 * len(solution.local_meshes)
    assert geometry_calls["lengths"] == {
        id(coarse): 1,
        **{id(fine): 2 for fine in solution.local_meshes},
    }
    assert geometry_calls["normals"] == {id(fine): 1 for fine in solution.local_meshes}
    for name in (
        "face_squared",
        "local_squared",
        "volume_squared",
        "divergence_squared",
        "traction_squared",
        "fine_squared",
    ):
        for first, second in zip(getattr(actual, name), getattr(expected, name), strict=True):
            np.testing.assert_array_equal(first, second)
    assert actual.total == expected.total
