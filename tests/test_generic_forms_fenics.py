"""Native generic UFL block assembly, independent of PDE-specific helper templates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm.backends.forms import assemble_form, assemble_pairing

pytestmark = pytest.mark.fem


@pytest.fixture
def native_spaces() -> tuple[Any, Any, Any, Any]:
    """Create independent scalar P1/P2 spaces and explicit integration on COMM_SELF."""
    pytest.importorskip("dolfinx")
    import ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI

    domain = mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    trial = fem.functionspace(domain, ("Lagrange", 1))
    test = fem.functionspace(domain, ("Lagrange", 2))
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 6})
    return domain, trial, test, dx


def test_native_rectangular_bilinear_form_and_linear_dual_pairing(
    native_spaces: tuple[Any, Any, Any, Any],
) -> None:
    """P2-test/P1-trial mass blocks have the correct row/column spaces and physical moments."""
    import ufl

    _, trial, test, dx = native_spaces
    u, v = ufl.TrialFunction(trial), ufl.TestFunction(test)
    x = ufl.SpatialCoordinate(trial.mesh)
    operator = assemble_form(u * v * dx, shape=(9, 4))
    test_load = assemble_form((1 + x[0] + 2 * x[1]) * v * dx, shape=(9,))
    coordinates = trial.tabulate_dof_coordinates()
    affine_coefficients = 1 + coordinates[:, 0] + 2 * coordinates[:, 1]
    assert_allclose(operator @ affine_coefficients, test_load, atol=2e-14)
    assert_allclose(operator @ np.ones(4), assemble_form(v * dx), atol=2e-14)
    assert_allclose(operator.T @ np.ones(9), assemble_form(u * dx), atol=2e-14)
    assert_allclose(np.ones(9) @ operator @ np.ones(4), 1.0, atol=2e-14)


@pytest.mark.parametrize("axis", ["columns", "rows"])
def test_native_pairings_use_explicit_test_or_trial_arguments(
    axis: str, native_spaces: tuple[Any, Any, Any, Any]
) -> None:
    """Trial-number-one linear forms assemble global rows without symbolic replacement."""
    import ufl

    _, trial, _, dx = native_spaces
    argument = ufl.TestFunction(trial) if axis == "columns" else ufl.TrialFunction(trial)
    x = ufl.SpatialCoordinate(trial.mesh)
    pairings = (argument * dx, -x[0] * argument * dx)
    value = assemble_pairing(pairings, axis=axis, size=4)
    expected = np.column_stack([assemble_form(form) for form in pairings])
    assert_allclose(value, expected if axis == "columns" else expected.T, atol=0)
    integrated = np.ones(4) @ value if axis == "columns" else value @ np.ones(4)
    assert_allclose(integrated, [1.0, -0.5], atol=2e-14)


def test_native_ufl_zero_forms_preserve_both_rectangular_dimensions(
    native_spaces: tuple[Any, Any, Any, Any],
) -> None:
    """Zero blocks keep explicit and native-inferred dimensions without entering the JIT."""
    import ufl

    _, trial, test, dx = native_spaces
    u, v = ufl.TrialFunction(trial), ufl.TestFunction(test)
    assert_array_equal(assemble_form(0 * v * dx, shape=(9,)), np.zeros(9))
    declared = ufl.ZeroBaseForm((v, u))
    assert_array_equal(assemble_form(declared).toarray(), np.zeros((9, 4)))
    assert_array_equal(assemble_form(ufl.ZeroBaseForm((u,))), np.zeros(4))


def test_native_matrix_vector_and_hdiv_spaces_require_no_named_model(
    native_spaces: tuple[Any, Any, Any, Any],
) -> None:
    """Basix vector and H(div) spaces pass directly through the generic assembler."""
    import basix.ufl
    import ufl
    from dolfinx import fem

    domain, _, _, _ = native_spaces
    spaces = [
        fem.functionspace(domain, basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))),
        fem.functionspace(domain, basix.ufl.element("RT", "triangle", 1)),
    ]
    for space in spaces:
        u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
        operator = assemble_form(ufl.inner(u, v) * ufl.dx)
        assert operator.shape[0] == operator.shape[1]
        assert_allclose(operator.toarray(), operator.toarray().T, atol=2e-14)
        assert np.linalg.eigvalsh(operator.toarray()).min() > 0


def test_native_submesh_entity_map_is_forwarded_explicitly(
    native_spaces: tuple[Any, Any, Any, Any],
) -> None:
    """A second mesh can supply an argument through DOLFINx's explicit entity map."""
    import ufl
    from dolfinx import fem, mesh

    domain, _, test, dx = native_spaces
    domain.topology.create_connectivity(2, 2)
    local, local_to_parent, _, _ = mesh.create_submesh(domain, 2, np.array([0, 1], dtype=np.int32))
    space = fem.functionspace(local, ("Lagrange", 1))
    u, v = ufl.TrialFunction(space), ufl.TestFunction(test)
    parent_to_local = np.full(domain.topology.index_map(2).size_local, -1, dtype=np.int32)
    parent_to_local[local_to_parent] = np.arange(len(local_to_parent), dtype=np.int32)
    operator = assemble_form(u * v * dx, entity_maps={local: parent_to_local})
    assert operator.shape == (9, 4)
    assert_allclose(operator @ np.ones(4), assemble_form(v * dx), atol=2e-14)


def test_native_declared_moment_reconstruction_uses_the_generic_operator(
    native_spaces: tuple[Any, Any, Any, Any],
) -> None:
    """A physical integral fixes an operator-independent constant energy lift."""
    import ufl

    from pymhm.core.moments import energy_reconstruction

    _, space, _, dx = native_spaces
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    operator = assemble_form((ufl.inner(ufl.grad(u), ufl.grad(v)) + u * v) * dx)
    moments = assemble_pairing((v * dx,), size=4)
    reconstruction, reduced = energy_reconstruction(operator, moments)
    assert_allclose(reconstruction[:, 0], np.ones(4), atol=2e-14)
    assert_allclose(operator @ reconstruction, moments, atol=2e-14)
    assert_allclose(moments.T @ reconstruction, [[1.0]], atol=2e-14)
    assert_allclose(reduced, [[1.0]], atol=2e-14)


def test_native_global_ufl_form_enters_the_declared_reduced_coordinates(
    native_spaces: tuple[Any, Any, Any, Any],
) -> None:
    """A global UFL mass operator and load agree with an uncondensed eight-DOF system."""
    import ufl

    from pymhm import Equation, LocalEquations, MultiscaleProblem, solve

    _, space, _, dx = native_spaces
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    x = ufl.SpatialCoordinate(space.mesh)
    global_a = u * v * dx
    global_f = (1 + x[0] + 2 * x[1]) * v * dx
    problem = MultiscaleProblem(
        Equation(global_a, global_f),
        lambda _: LocalEquations(
            2 * np.eye(4), [1.0, 2.0, 3.0, 4.0], np.eye(4), -np.eye(4), np.arange(4)
        ),
        [0],
        4,
        (0,),
    )
    result = solve(problem)
    mass = assemble_form(global_a).toarray()
    load = assemble_form(global_f)
    monolithic = np.block([[2 * np.eye(4), np.eye(4)], [-np.eye(4), mass]])
    expected = np.linalg.solve(monolithic, np.r_[[1.0, 2.0, 3.0, 4.0], load])
    assert_allclose(np.r_[result.fields[0], result.trace], expected, atol=2e-14, rtol=2e-14)
    assert result.raw_residual < 1e-13
