"""Physical and interoperability checks for the runnable provider tutorial."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from examples.tutorial_local_provider import (
    Boundary,
    Formulation,
    TutorialProvider,
    build_problem,
    diagnostics,
    run_tutorial,
    source,
)
from pymhm._legacy.models.darcy.primal import darcy_local_provider
from pymhm.core.contracts import LocalProblem
from pymhm.core.equations import Equation, LocalEquations, compile_local_equations
from pymhm.core.multiscale import assemble
from pymhm.execution.cpu import ExecutionConfig
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh


@pytest.mark.parametrize("formulation", ["primal", "mixed"])
@pytest.mark.parametrize("boundary", ["dirichlet", "neumann"])
@pytest.mark.parametrize("local_solver", ["scipy", "external"])
def test_manufactured_fields_original_rows_and_physical_gauge(
    formulation: Formulation, boundary: Boundary, local_solver: Any
) -> None:
    """Both operators reproduce the exact admissible flux with physical means."""
    result = run_tutorial(formulation=formulation, boundary=boundary, local_solver=local_solver)
    report = diagnostics(result)
    for name in ("Darcy_flux", "divergence"):
        assert report["field_L2_errors"][name] < 2e-12
    if formulation == "primal":
        assert report["field_L2_errors"]["pressure"] < 2e-12
    else:
        # The quadratic analytical pressure does not belong to cellwise P0.
        assert 0.1 < report["field_L2_errors"]["pressure"] < 0.3
        assert set(report["maximum_local_original_row_l2_by_block"]) == {
            "Darcy_flux",
            "pressure",
            "boundary_flux_constraint",
        }
    assert max(report["maximum_local_original_row_l2_by_block"].values()) < 2e-12
    assert report["global_original_relative_residual"] < 1e-12
    if boundary == "neumann":
        integral = sum(
            float(record[1] @ field)
            for record, field in zip(
                result.system.local_metadata, result.solution.fields, strict=True
            )
        )
        assert_allclose(integral, 5 / 3, atol=2e-13, rtol=0)


@pytest.mark.parametrize("formulation", ["primal", "mixed"])
def test_local_form_compiler_preserves_executed_operator_and_basis(
    formulation: Formulation,
) -> None:
    """Explicit user equations preserve the established executed coefficients."""
    mesh = TriangleMesh.unit_square()
    factory = darcy_local_provider(
        mesh,
        formulation=formulation,
        degree=2 if formulation == "primal" else 1,
        local_refinement=2,
        source=source,
    )
    provider = TutorialProvider(mesh, SkeletonSpace(mesh), formulation)
    restored = pickle.loads(pickle.dumps(provider))
    for cell in range(len(mesh.cells)):
        original = factory(cell)
        compiled = compile_local_equations(restored(cell))
        for name in ("matrix", "coupling", "load", "trace_dofs", "kernel", "constraints"):
            a, b = getattr(original.problem, name), getattr(compiled.problem, name)
            assert_array_equal(
                a.toarray() if name == "matrix" else a, b.toarray() if name == "matrix" else b
            )
        assert_array_equal(original.metadata[0].points, compiled.metadata[0].points)
        assert_array_equal(original.metadata[1], compiled.metadata[1])


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_ordered_batches_preserve_the_mixed_physical_solution(backend: Any) -> None:
    """A spawn-compatible provider preserves orientation, summation and gauge."""
    baseline = run_tutorial(formulation="mixed", boundary="neumann", local_solver="external")
    actual = run_tutorial(
        formulation="mixed",
        boundary="neumann",
        local_solver="external",
        execution=ExecutionConfig(backend, workers=2, native_threads=1, batch_size=1),
    )
    assert_array_equal(actual.system.matrix.toarray(), baseline.system.matrix.toarray())
    assert_array_equal(actual.system.rhs, baseline.system.rhs)
    assert_array_equal(actual.solution.trace, baseline.solution.trace)
    for field, expected in zip(actual.solution.fields, baseline.solution.fields, strict=True):
        assert_array_equal(field, expected)


def test_numeric_compiler_can_request_default_coefficient_moments() -> None:
    """A declared kernel may omit physical moments when no physical gauge is claimed."""
    forms = LocalEquations(
        a=np.array([[1.0, -1.0], [-1.0, 1.0]]),
        L=np.zeros(2),
        b=np.array([[1.0], [0.0]]),
        c=np.array([[-1.0, 0.0]]),
        dofs=np.array([0]),
        kernel=np.ones((2, 1)),
    )
    local = compile_local_equations(forms).problem
    assert isinstance(local, LocalProblem)
    assert_array_equal(local.constraints, local.kernel)


def test_user_global_form_contributes_to_declared_coordinates() -> None:
    """A user global form changes the reduced equation without choosing a model."""
    problem = build_problem()
    baseline = assemble(problem)
    operator = np.diag(np.arange(1, len(baseline.rhs) + 1, dtype=float))
    forcing = np.linspace(0.2, 1.2, len(baseline.rhs))
    equation = Equation(operator, problem.global_equation.L + forcing)
    modified = assemble(replace(problem, global_equation=equation))
    assert_allclose(modified.matrix.toarray(), baseline.matrix.toarray() + operator)
    assert_allclose(modified.rhs, baseline.rhs + forcing)
    actual = modified.solve()
    coefficients = np.r_[actual.trace, *actual.coarse]
    expected = np.linalg.solve(modified.matrix.toarray(), modified.rhs)
    assert_allclose(coefficients, expected, atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"formulation": "unknown"}, "formulation"),
        ({"element_backend": "unknown"}, "element_backend"),
        ({"local_refinement": 0}, "local_refinement"),
    ],
)
def test_application_provider_rejects_invalid_declared_spaces(
    kwargs: dict[str, Any], match: str
) -> None:
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match=match):
        TutorialProvider(mesh, SkeletonSpace(mesh), **kwargs)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"formulation": "unknown"}, "formulation"),
        ({"formulation": "mixed", "degree": 2}, "RT0 uses degree=1"),
        ({"element_backend": "unknown"}, "element_backend"),
        ({"degree": 0}, "degree"),
        ({"local_refinement": 0}, "local_refinement"),
        ({"quadrature_order": 0}, "quadrature_order"),
    ],
)
def test_public_darcy_provider_rejects_invalid_discretizations(
    kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        darcy_local_provider(TriangleMesh.unit_square(), **kwargs)


@pytest.mark.parametrize("wrong_mesh,components", [(True, 1), (False, 2)])
def test_provider_requires_its_own_scalar_skeleton(wrong_mesh: bool, components: int) -> None:
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(
        TriangleMesh.unit_square() if wrong_mesh else mesh, components=components
    )
    with pytest.raises(ValueError, match="scalar skeleton"):
        TutorialProvider(mesh, skeleton)


@pytest.mark.parametrize("face", [FaceSpace.uniform(1), FaceSpace.uniform(0, 3)])
def test_rt0_requires_constant_aligned_trace_segments(face: FaceSpace) -> None:
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(face for _ in mesh.faces))
    provider = TutorialProvider(mesh, skeleton, formulation="mixed", local_refinement=2)
    with pytest.raises(ValueError, match="degree-zero trace segments aligned"):
        provider(0)


def test_example_rejects_unknown_boundary_or_solver() -> None:
    invalid: Any = "unknown"
    with pytest.raises(ValueError, match="boundary"):
        build_problem(boundary=invalid)
    with pytest.raises(ValueError, match="local_solver"):
        run_tutorial(local_solver=invalid)


@pytest.mark.parametrize("formulation", ["primal", "mixed"])
def test_provider_uses_the_native_basis(formulation: Any) -> None:
    pytest.importorskip("basix")
    result = run_tutorial(element_backend="basix", formulation=formulation)
    report = diagnostics(result)
    assert report["element_backend"] == "basix"
    for name in ("Darcy_flux", "divergence"):
        assert report["field_L2_errors"][name] < 3e-12
    if formulation == "primal":
        assert report["field_L2_errors"]["pressure"] < 3e-12
    else:
        # Exact triangle moments give the quadratic pressure's P0 projection.
        # E(lambda_i**2)=1/6 and E(lambda_i*lambda_j)=1/12 for i != j.
        bary, weights = triangle_quadrature(5)
        squared_projection_error = 0.0
        for field, (fine, _) in zip(
            result.solution.fields, result.system.local_metadata, strict=True
        ):
            vertices = fine.points[fine.cells]
            means = (
                1
                + (np.sum(vertices**2, axis=(1, 2)) + np.sum(np.sum(vertices, axis=1) ** 2, axis=1))
                / 12
            )
            start = len(fine.faces)
            assert_allclose(field[start : start + len(fine.cells)], means, atol=3e-12, rtol=0)
            points = bary @ vertices
            exact = 1 + np.sum(points**2, axis=-1)
            squared_projection_error += float(
                fine.areas @ ((exact - means[:, None]) ** 2 @ weights)
            )
        assert_allclose(
            report["field_L2_errors"]["pressure"],
            np.sqrt(squared_projection_error),
            atol=3e-12,
            rtol=0,
        )
    assert max(report["maximum_local_original_row_l2_by_block"].values()) < 2e-12
    assert report["global_original_relative_residual"] < 1e-12


def test_command_line_process_external_neumann_reports_separate_fields() -> None:
    """The guarded entry point is executable under cross-platform spawn."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.tutorial_local_provider",
            "--formulation",
            "mixed",
            "--boundary",
            "neumann",
            "--local-solver",
            "external",
            "--backend",
            "process",
            "--workers",
            "2",
            "--batch-size",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
    )
    report = json.loads(result.stdout)
    assert set(report["field_L2_errors"]) == {"pressure", "Darcy_flux", "divergence"}
    assert report["field_L2_errors"]["Darcy_flux"] < 1e-12
    assert report["global_original_relative_residual"] < 1e-12
