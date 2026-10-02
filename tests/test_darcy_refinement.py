"""Public Darcy refinement preserves physical boundary data and evaluated fields."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm import TriangleMesh, solve_darcy
from pymhm.hybrid import HybridSystem
from pymhm.lagrange import tabulate


def pressure(points):
    """Return an affine pressure with nonzero mean and anisotropic flux."""
    return 0.7 + points[:, 0] - 2 * points[:, 1]


@pytest.mark.parametrize("formulation", ["primal", "mixed"])
@pytest.mark.parametrize("boundary", ["DD", "DN", "NN"])
def test_refinement_repairs_original_equations_and_evaluates_corrected_fields(
    monkeypatch, formulation, boundary
):
    """Correct a forced field defect while preserving flux data and the physical mean."""
    mesh = TriangleMesh.unit_square()
    tensor = np.array([[3.0, 0.5], [0.5, 1.0]])
    exact_flux = -tensor @ [1.0, -2.0]
    neumann = {int(f): float(exact_flux @ mesh.normals[f]) for f in mesh.boundary_faces}
    if boundary == "DD":
        neumann = None
    elif boundary == "DN":
        neumann = {next(iter(neumann)): next(iter(neumann.values()))}
    options = dict(
        permeability=tensor,
        dirichlet=pressure,
        neumann=neumann,
        mean_pressure=0.2,
        formulation=formulation,
        local_refinement=2,
    )
    expected = solve_darcy(mesh, **options)
    original = HybridSystem.solve
    calls = []

    def perturbed(system, *args, **kwargs):
        """Introduce a measurable local defect only in the initial global solve."""
        solution = original(system, *args, **kwargs)
        if not calls:
            calls.append(True)
            fields = tuple(u + 0.002 * np.arange(1, len(u) + 1) for u in solution.fields)
            return replace(solution, fields=fields)
        return solution

    monkeypatch.setattr(HybridSystem, "solve", perturbed)
    result = solve_darcy(mesh, hybrid_refinement_steps=2, **options)
    assert calls
    for current, reference in zip(result.hybrid.fields, expected.hybrid.fields, strict=True):
        assert_allclose(current, reference, atol=3e-12, rtol=3e-12)
    assert_allclose(result.hybrid.trace, expected.hybrid.trace, atol=3e-12)
    assert_allclose(result.hybrid.gauge_multipliers, expected.hybrid.gauge_multipliers, atol=3e-12)
    assert result.flux_l2_error(exact_flux) < 3e-12
    assert_allclose(result.conservation_residuals(), 0, atol=3e-12)
    for fine, field, p, flux in zip(
        result.local_meshes, result.hybrid.fields, result.pressure, result.flux, strict=True
    ):
        if formulation == "mixed":
            assert_array_equal(flux, field[: len(fine.faces)])
            assert_array_equal(p, field[len(fine.faces) : len(fine.faces) + len(fine.cells)])
        else:
            assert_array_equal(p, field)
            dofs, _, _, gradients, _ = tabulate(fine, 1, np.full((1, 3), 1 / 3))
            assert_allclose(
                flux,
                -np.einsum("ab,tb->ta", tensor, np.einsum("ti,tia->ta", p[dofs], gradients[:, 0])),
            )


@pytest.mark.parametrize("formulation", ["primal", "mixed"])
def test_zero_refinement_preserves_default_bitwise(monkeypatch, formulation):
    """The default does not call optional defect correction or change coefficients."""
    mesh = TriangleMesh.unit_square()
    expected = solve_darcy(mesh, dirichlet=pressure, formulation=formulation)

    def forbidden(*args, **kwargs):
        """Fail if an explicitly disabled refinement path is entered."""
        raise AssertionError("zero steps must bypass refinement")

    monkeypatch.setattr("pymhm.darcy.refine_hybrid", forbidden)
    current = solve_darcy(
        mesh, dirichlet=pressure, formulation=formulation, hybrid_refinement_steps=0
    )
    assert_array_equal(current.hybrid.trace, expected.hybrid.trace)
    for field, reference in zip(current.hybrid.fields, expected.hybrid.fields, strict=True):
        assert_array_equal(field, reference)


@pytest.mark.parametrize("steps", [-1, True, 1.5])
def test_invalid_refinement_count_is_rejected(steps):
    """Correction counts follow the public nonnegative-integer contract."""
    with pytest.raises(ValueError, match="hybrid_refinement_steps"):
        solve_darcy(TriangleMesh.unit_square(), hybrid_refinement_steps=steps)
