"""Analytical field, gauge and conservation checks of the introductory scalar choices."""

import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.tutorial_scalar_variants import (
    VARIANTS,
    affine_pressure,
    main,
    measure_variant,
    solve_variant,
)
from pymhm.elements import triangle_quadrature
from pymhm.lagrange import tabulate


@pytest.fixture(scope="module", params=tuple(VARIANTS))
def patch(request: pytest.FixtureRequest) -> Iterator[tuple[str, Any, dict[str, float]]]:
    """Solve each declared patch once and retain its independently measured fields."""
    with threadpool_limits(1):
        name = request.param
        solution = solve_variant(name)
        yield name, solution, measure_variant(name, solution)


def test_analytical_fields_and_separate_physical_moments(patch):
    name, solution, metrics = patch
    if name == "rt0":
        # For these two unit-square triangles, the affine pressure variance is 7/18.
        assert metrics["scalar_l2"] == pytest.approx(np.sqrt(7 / 18), abs=3e-14)
        for fine, pressure in zip(solution.local_meshes, solution.pressure, strict=True):
            means = affine_pressure(fine.points[fine.cells].mean(axis=1))
            np.testing.assert_allclose(pressure.reshape(-1), means, rtol=0, atol=2e-13)
    else:
        assert metrics["scalar_l2"] < 1e-10
    if "flux_l2" in metrics:
        assert metrics["flux_l2"] < 1e-10
    for key, value in metrics.items():
        if key != "scalar_l2":
            assert np.isfinite(value) and value < 1e-10, (name, key, value)


def test_nontrivial_neumann_gauge_and_quadratic_source():
    """Check the physical mean and a nonzero source independently of reduced residuals."""
    with threadpool_limits(1):
        neumann = solve_variant("primal-neumann")
        bary, weights = triangle_quadrature(5)
        integral = 0.0
        for fine, pressure in zip(neumann.local_meshes, neumann.pressure, strict=True):
            dofs, _, basis, _, _ = tabulate(fine, neumann.degree, bary)
            integral += float(fine.areas @ ((pressure[dofs] @ basis.T) @ weights))
        assert integral == pytest.approx(2.5, abs=5e-13)

        quadratic = solve_variant("primal-quadratic")
        assert quadratic.source == -6.0
        for cell, fine in enumerate(quadratic.local_meshes):
            # The integrated source is -6 times the physical area on each macrocell.
            outward = 0.0
            for face in quadratic.skeleton.mesh.cell_faces[cell]:
                macro = quadratic.skeleton.mesh
                side = int(np.flatnonzero(macro.cell_faces[cell] == face)[0])
                midpoint = macro.points[macro.faces[face]].mean(axis=0)
                exact_flux = -2 * midpoint * [1.0, 2.0]
                outward += (
                    float(exact_flux @ macro.normals[face])
                    * macro.signs[cell, side]
                    * (macro.lengths[face])
                )
            assert outward == pytest.approx(-6 * fine.areas.sum(), abs=2e-14)
        assert np.max(abs(quadratic.conservation_residuals())) < 1e-10


@pytest.mark.parametrize("name", ["BDM", "invalid", ""])
def test_invalid_local_choice_fails_before_solving(name):
    with pytest.raises(ValueError, match="unknown scalar variant"):
        solve_variant(name)


def test_cli_lists_choices_and_writes_named_field_errors(tmp_path, capsys):
    output = tmp_path / "patch.json"
    main(["--variant", "primal", "--output", str(output)])
    record = json.loads(output.read_text())["records"][0]
    assert json.loads(capsys.readouterr().out)["records"][0] == record
    assert record["metrics"]["scalar_l2"] < 1e-10
    assert record["metrics"]["flux_l2"] < 1e-10
    with pytest.raises(SystemExit):
        main(["--variant", "invalid"])


def test_nonpositive_refinement_is_rejected():
    with pytest.raises(ValueError, match="refinement"):
        solve_variant("primal", refinement=0)
