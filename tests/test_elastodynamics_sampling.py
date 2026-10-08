"""One-sided constitutive sampling and single-tabulation analytical norm checks."""

import importlib
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm._legacy.models.waves.elastodynamics import (
    ElastodynamicLocal,
    ElastodynamicSolution,
    _stress_from_gradient,
)
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh


def _local(mesh: Any, degree: int, **material: Any) -> ElastodynamicLocal:
    """Build a sampling-only local record without assembling any physical operator."""
    dimension = mesh.points.shape[1]
    if dimension == 3:
        dofs, nodes = tetra_nodal_space(mesh, degree)
    else:
        dofs, nodes = mesh.cells, mesh.points
    return ElastodynamicLocal(
        mesh=mesh,
        degree=degree,
        mass=None,
        stiffness=None,
        coupling=np.empty((0, 0)),
        trace_dofs=np.empty(0, dtype=int),
        dofs=dofs,
        nodes=nodes,
        basis=np.empty((0, 0)),
        points=np.empty((0, 0, dimension)),
        weights=np.empty((0, 0)),
        density_values=np.empty((0, 0)),
        constitutive=material.get("constitutive"),
        lame_lambda=material.get("lame_lambda", 1.0),
        lame_mu=material.get("lame_mu", 1.0),
        quadrature_order=degree + 2,
    )


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("anisotropic", [False, True])
def test_shared_stress_matches_cartesian_and_kelvin_laws(dimension: int, anisotropic: bool) -> None:
    """A variable full Kelvin tensor and isotropic law act on the same physical gradient."""
    mesh = TriangleMesh.unit_square() if dimension == 2 else TetraMesh.unit_cube()
    points = mesh.points[mesh.cells].mean(axis=1)[:, None, :]
    gradient = np.arange(1.0, dimension**2 + 1).reshape(dimension, dimension) / 7
    gradients = np.broadcast_to(gradient, (*points.shape[:-1], dimension, dimension))

    def lam(p: np.ndarray) -> np.ndarray:
        """Return the spatially varying first Lame parameter."""
        return 1 + p[:, 0]

    def mu(p: np.ndarray) -> np.ndarray:
        """Return the positive spatially varying shear modulus."""
        return 2 + p[:, -1]

    pairs = [(0, 1)] if dimension == 2 else [(1, 2), (0, 2), (0, 1)]
    count = dimension + len(pairs)
    matrix = np.diag(np.arange(1.0, count + 1)) + 0.2 * np.ones((count, count))

    def material(p: np.ndarray) -> np.ndarray:
        """Scale the same positive-definite anisotropic Kelvin matrix spatially."""
        return (1 + p[:, 0, None, None]) * matrix

    local = _local(
        mesh, 1, constitutive=material if anisotropic else None, lame_lambda=lam, lame_mu=mu
    )
    stress = _stress_from_gradient(local, points, gradients)
    if anisotropic:
        strain = np.r_[
            np.diag(gradient), [(gradient[i, j] + gradient[j, i]) / np.sqrt(2) for i, j in pairs]
        ]
        components = np.einsum("nab,b->na", material(points.reshape(-1, dimension)), strain)
        expected = np.zeros((len(components), dimension, dimension))
        for axis in range(dimension):
            expected[:, axis, axis] = components[:, axis]
        for index, (i, j) in enumerate(pairs, start=dimension):
            expected[:, i, j] = expected[:, j, i] = components[:, index] / np.sqrt(2)
        expected = expected.reshape(stress.shape)
    else:
        expected = mu(points.reshape(-1, dimension))[:, None, None] * (gradient + gradient.T)
        expected += (
            lam(points.reshape(-1, dimension))[:, None, None]
            * np.trace(gradient)
            * np.eye(dimension)
        )
        expected = expected.reshape(stress.shape)
    assert_allclose(stress, expected, rtol=4e-15, atol=4e-15)


def _campaign() -> ModuleType:
    """Import the installed case owner so patches target its numerical globals."""
    return importlib.import_module("examples.elastodynamics_campaign")


class _QuadraticMotion:
    """Exact u_i=x_i² and v=2u at the sampling time, lambda=mu=0.4."""

    @staticmethod
    def amplitudes(time: float) -> tuple[float, float, float]:
        """Supply the displacement and velocity coefficients at this time."""
        return 1.0, 2.0, 0.0

    @staticmethod
    def spatial(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Differentiate each Cartesian quadratic independently."""
        gradient = np.zeros((len(points), 3, 3))
        hessian = np.zeros((len(points), 3, 3, 3))
        for axis in range(3):
            gradient[:, axis, axis] = 2 * points[:, axis]
            hessian[:, axis, axis, axis] = 2
        return points**2, gradient, hessian

    @staticmethod
    def divergence(hessian: np.ndarray) -> np.ndarray:
        """Apply div(sigma)=0.4 Laplacian(u)+0.8 grad(div(u)) in physical coordinates."""
        return 0.4 * np.einsum("...ijj->...i", hessian) + 0.8 * np.einsum("...jji->...i", hessian)


def test_norms_tabulate_each_macro_once_and_preserve_exact_quadratic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A polynomial motion checks all six physical norms and prohibits redundant evaluators."""
    campaign = _campaign()
    macro = TetraMesh.unit_cube()
    locals_ = tuple(
        _local(macro.submesh(i, 2), 3, lame_lambda=0.4, lame_mu=0.4)
        for i in range(len(macro.cells))
    )
    displacement = tuple((local.nodes**2).ravel() for local in locals_)
    solution = ElastodynamicSolution(
        None, locals_, displacement, tuple(2 * u for u in displacement), np.empty(0), 0.5, 0.0, 0.0
    )
    calls = []
    tabulate = campaign.tetra_element_tabulate

    def counted(mesh: TetraMesh, degree: int, bary: np.ndarray) -> Any:
        """Record each owned mesh before returning the unchanged physical tabulation."""
        calls.append(mesh)
        return tabulate(mesh, degree, bary)

    def prohibited(*args: Any, **kwargs: Any) -> None:
        """Reject a redundant public evaluator in the fused integration path."""
        pytest.fail("norm integration redundantly evaluates the nodal basis")

    monkeypatch.setattr(campaign, "tetra_element_tabulate", counted)
    for name in ["evaluate", "gradient", "stress"]:
        monkeypatch.setattr(ElastodynamicSolution, name, prohibited)
    errors = campaign.norms(solution, _QuadraticMotion(), 5)
    assert len(calls) == len(locals_)
    assert all(mesh is local.mesh for mesh, local in zip(calls, locals_, strict=True))
    assert max(errors.values()) < 2e-12
