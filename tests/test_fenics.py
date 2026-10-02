"""Adapter API contracts and real UFL expression construction without DOLFINx."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import fenics


class SimulatedForm:
    """Explicit assembly-contract fake, not a finite-element implementation."""

    def __init__(self, values: Any, rank: int = 1, space: str = "V") -> None:
        self.values = np.asarray(values)
        self.rank = rank
        self.space = space
        self.mesh = SimpleNamespace(comm=SimpleNamespace(size=1))

    def arguments(self) -> tuple[Any, ...]:
        return tuple(
            SimpleNamespace(ufl_function_space=lambda: self.space) for _ in range(self.rank)
        )

    def integrals(self) -> list[Any]:
        return [SimpleNamespace(integrand=lambda: self.values.item())]


@pytest.fixture
def simulated_fem(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Simulate the documented native DOLFINx CSR/vector assembly interface."""
    fake = SimpleNamespace(
        form=lambda form: form,
        assemble_matrix=lambda form: SimpleNamespace(
            to_scipy=lambda: form.values, scatter_reverse=lambda: None
        ),
        assemble_vector=lambda form: SimpleNamespace(array=form.values),
    )
    monkeypatch.setattr(fenics, "import_module", lambda name: fake)
    return fake


def test_simulated_local_assembly_contract(simulated_fem: Any) -> None:
    a = SimulatedForm([[1.0, -1.0], [-1.0, 1.0]], 2)
    problem = fenics.from_ufl(
        a,
        SimulatedForm([0.0, 0.0]),
        [SimulatedForm([1.0, 0.0]), SimulatedForm([0.0, 1.0])],
        [0, 1],
        kernel=np.ones((2, 1)),
        constraint_forms=[SimulatedForm([0.5, 0.5])],
    )
    assert_allclose(problem.matrix.toarray(), a.values)
    assert_allclose(problem.coupling, np.eye(2))
    assert_allclose(problem.constraints, [[0.5], [0.5]])
    assert_allclose(problem.condense().source, [0.0, 0.0])


def test_simulated_general_coarse_basis_forwarded(simulated_fem: Any) -> None:
    """Retain a non-null mode and its physical moment through the UFL adapter."""
    basis = np.ones((2, 1))
    problem = fenics.from_ufl(
        SimulatedForm([[1.0001, -1.0], [-1.0, 1.0002]], 2),
        SimulatedForm([1.0, -1.0]),
        [SimulatedForm([1.0, 0.0])],
        [0],
        coarse_basis=basis,
        constraint_forms=[SimulatedForm([0.4, 0.6])],
    )
    assert problem.kernel.shape == (2, 0)
    assert_allclose(problem.coarse_basis, basis)
    assert_allclose(problem.constraints, [[0.4], [0.6]])
    response = problem.condense()
    assert response.coarse_vectors is not None
    assert_allclose(problem.constraints.T @ response.source, 0, atol=1e-15)
    assert_allclose(problem.constraints.T @ response.retained_basis, [[1.0]], atol=1e-15)


def test_simulated_zero_forms_and_empty_traces(simulated_fem: Any) -> None:
    problem = fenics.from_ufl(
        SimulatedForm(np.eye(2), 2),
        SimulatedForm(0.0, 0),
        [],
        np.empty(0, dtype=int),
        constraint_forms=[],
    )
    assert_allclose(problem.load, [0.0, 0.0])
    assert problem.coupling.shape == problem.constraints.shape == (2, 0)
    plain = fenics.from_ufl(SimulatedForm(np.eye(2), 2), SimulatedForm([1.0, 2.0]), [], [])
    assert_allclose(plain.load, [1, 2])


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_bilinear_form_required(rank: int, simulated_fem: Any) -> None:
    with pytest.raises(ValueError, match="bilinear"):
        fenics.from_ufl(SimulatedForm(np.eye(2), rank), None, [], [])


def test_matching_trial_space_required(simulated_fem: Any) -> None:
    a = SimulatedForm(np.eye(2), 2)
    a.arguments = lambda: (
        SimpleNamespace(ufl_function_space=lambda: "V"),
        SimpleNamespace(ufl_function_space=lambda: "W"),
    )
    with pytest.raises(ValueError, match="trial and test"):
        fenics.from_ufl(a, None, [], [])


def test_distributed_mesh_rejected(simulated_fem: Any) -> None:
    a = SimulatedForm(np.eye(2), 2)
    a.mesh.comm.size = 2
    with pytest.raises(ValueError, match="single-rank"):
        fenics.from_ufl(a, None, [], [])


@pytest.mark.parametrize("bad_load", [SimulatedForm(1.0, 0), SimulatedForm([1, 2], 1, "W")])
def test_bad_linear_form_rejected(bad_load: SimulatedForm, simulated_fem: Any) -> None:
    with pytest.raises(ValueError, match="same test space"):
        fenics.from_ufl(SimulatedForm(np.eye(2), 2), bad_load, [], [])


@pytest.mark.parametrize("complex_matrix", [False, True])
def test_complex_assembly_rejected(complex_matrix: bool, simulated_fem: Any) -> None:
    a = SimulatedForm(np.eye(2) * (1j if complex_matrix else 1), 2)
    load = SimulatedForm([1j, 1j])
    with pytest.raises(ValueError, match="real-valued"):
        fenics.from_ufl(a, load, [], [])


@pytest.mark.parametrize("error", [ImportError("no module"), OSError("no library")])
def test_optional_import_diagnosis(error: Exception, monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(name: str) -> Any:
        raise error

    monkeypatch.setattr(fenics, "import_module", unavailable)
    with pytest.raises(ImportError, match="Pixi fem"):
        fenics.from_ufl(None, None, [], [])


@pytest.fixture
def ufl_spaces() -> tuple[Any, Any, Any, Any]:
    """Construct real Basix/UFL spaces without importing the DOLFINx runtime."""
    import basix.ufl
    import ufl

    coordinate = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    domain = ufl.Mesh(coordinate)
    scalar = basix.ufl.element("Lagrange", "triangle", 1)
    vector = basix.ufl.element("Lagrange", "triangle", 2, shape=(2,))
    return (
        ufl.FunctionSpace(domain, scalar),
        ufl.FunctionSpace(domain, vector),
        ufl.FunctionSpace(domain, basix.ufl.mixed_element([vector, scalar])),
        ufl.FunctionSpace(
            domain,
            basix.ufl.mixed_element(
                [
                    basix.ufl.element("RT", "triangle", 1),
                    basix.ufl.element("DG", "triangle", 0),
                ]
            ),
        ),
    )


def test_real_ufl_volume_form_helpers(ufl_spaces: tuple[Any, ...]) -> None:
    import ufl

    scalar, vector, mixed, rt = ufl_spaces
    force = ufl.as_vector((1.0, -1.0))
    forms = [
        fenics.primal_darcy_forms(scalar, 2.0, 1.0),
        fenics.primal_darcy_forms(scalar, ufl.as_matrix(((2.0, 0.0), (0.0, 1.0))), 1.0),
        fenics.mixed_darcy_forms(rt, 1.0, 1.0),
        fenics.brinkman_forms(mixed, 1.0, 2.0, force, 1.0),
        fenics.elasticity_forms(vector, 2.0, 1.0, force),
        fenics.usfem_brinkman_forms(mixed, 1.0, 0.0, force, smallest_resistance=0.0),
    ]
    for bilinear, linear in forms:
        assert len(bilinear.arguments()) == 2
        assert len(linear.arguments()) == 1
        assert bilinear.ufl_domains() == linear.ufl_domains()


@pytest.mark.parametrize("value", [0.0, -1.0, 1.0, np.nan])
def test_usfem_inverse_estimate_validation(value: float) -> None:
    with pytest.raises(ValueError, match="inverse_estimate"):
        fenics.usfem_brinkman_forms(None, 1, 1, None, smallest_resistance=1, inverse_estimate=value)
