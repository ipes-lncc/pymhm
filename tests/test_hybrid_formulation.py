"""User-written three-field/moment equations recover the same physical discretizations."""

import pytest
from numpy.testing import assert_allclose

from examples.hybrid_formulation import build_problem, recover_pressure, run_patch
from pymhm import ExecutionConfig, assemble
from pymhm.methods.hho import solve_mshho
from pymhm.methods.three_field import solve_mh2m


@pytest.mark.parametrize("method", ["three-field", "moments"])
@pytest.mark.parametrize("backend", ["serial", "process"])
def test_declared_method_blocks_preserve_affine_fields_and_reference_spaces(method, backend):
    problem = build_problem(method)
    system = assemble(problem, execution=ExecutionConfig(backend, workers=2, batch_size=2))
    solution = system.solve()
    fields = recover_pressure(system, solution)
    mesh = problem.local_provider.keywords["mesh"]
    reference = (solve_mh2m if method == "three-field" else solve_mshho)(
        mesh,
        degree=2,
        local_refinement=2,
        dirichlet=lambda x: 1 + x[:, 0] + 2 * x[:, 1],
        quadrature_order=6,
    )
    assert len(fields) == len(reference.pressure)
    for actual, expected in zip(fields, reference.pressure, strict=True):
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    if method == "moments":
        assert_allclose(solution.trace, reference.face_moments, rtol=1e-12, atol=1e-12)
    assert solution.raw_residual < 1e-13


@pytest.mark.parametrize("method", ["three-field", "moments"])
def test_example_checks_manufactured_field_independently(method):
    assert run_patch(method)["pressure_coefficient_error"] < 1e-12


def test_invalid_method_name_cannot_choose_a_hidden_solver():
    with pytest.raises(ValueError, match="method"):
        build_problem("invalid")
