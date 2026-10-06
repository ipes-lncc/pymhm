"""Adapter API contracts and real UFL expression construction without DOLFINx."""

from __future__ import annotations

import weakref
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from pymhm.backends import fenics, forms


class SimulatedForm:
    """Explicit assembly-contract fake, not a finite-element implementation."""

    def __init__(self, values: Any, rank: int = 1, space: Any = "V") -> None:
        self.values = np.asarray(values)
        self.rank = rank
        self.space = space
        self.mesh = SimpleNamespace(comm=SimpleNamespace(size=1))

    def arguments(self) -> tuple[Any, ...]:
        return tuple(
            SimpleNamespace(ufl_function_space=lambda: self.space) for _ in range(self.rank)
        )

    def integrals(self) -> list[Any]:
        return [SimpleNamespace(integrand=lambda: self.values.item() if self.rank == 0 else 1)]

    def ufl_domains(self) -> tuple[Any, ...]:
        """Expose the integration mesh so it can be checked before native JIT."""
        return (SimpleNamespace(ufl_cargo=lambda: self.mesh),)


@pytest.fixture
def simulated_fem(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Simulate the documented native DOLFINx CSR/vector assembly interface."""
    observed = SimpleNamespace(options=[], references=[], scatters=[])

    class Compiled:
        """Hold native form data for only the duration of assembly."""

        def __init__(self, form: Any, options: Any) -> None:
            """Record precision and options while simulating native option mutation."""
            self.form, self.mesh = form, form.mesh
            observed.options.append(options)
            options["form_compiler_options"]["scalar_type"] = options["dtype"]
            observed.references.append(weakref.ref(self))

    class Buffer:
        """Own native coefficients whose lifetime ends before returning local arrays."""

        def __init__(self, compiled: Any) -> None:
            """Track native coefficient ownership without performing finite elements."""
            self.array = compiled.form.values.copy()
            self.matrix = sparse.csr_matrix(self.array) if self.array.ndim == 2 else None
            observed.references.append(weakref.ref(self))

        def scatter_reverse(self, *mode: str) -> None:
            """Record matrix completion or additive vector accumulation."""
            observed.scatters.append(mode)

        def to_scipy(self) -> Any:
            """Expose a matrix view that must be copied by the shared adapter."""
            return self.matrix

    fem = SimpleNamespace(
        form=lambda form, **options: Compiled(form, options),
        assemble_matrix=Buffer,
        assemble_vector=Buffer,
    )
    actual_import = forms.import_module
    native = {
        "dolfinx.fem": fem,
        "dolfinx.la": SimpleNamespace(InsertMode=SimpleNamespace(add="add")),
    }
    monkeypatch.setattr(
        forms, "import_module", lambda name: native.get(name) or actual_import(name)
    )
    return observed


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
    assert simulated_fem.scatters == [(), ("add",), ("add",), ("add",), ("add",)]
    assert all(reference() is None for reference in simulated_fem.references)


def test_binary64_compile_options_and_independent_assembly_copies(simulated_fem: Any) -> None:
    """Local assembly shares the generic adapter's precision, options and ownership."""
    a = SimulatedForm(np.eye(2, dtype=np.float32), 2)
    load = SimulatedForm(np.array([1.0, 2.0], dtype=np.float32))
    compiler_options = {"quadrature_degree": 7}
    jit_options = {"timeout": 60}
    entity_maps = {"mesh": np.array([0], dtype=np.int32)}
    problem = fenics.from_ufl(
        a,
        load,
        [SimulatedForm([1.0, -1.0])],
        [0],
        coarse_basis=np.ones((2, 1)),
        constraint_forms=[SimulatedForm([0.5, 0.5])],
        form_compiler_options=compiler_options,
        jit_options=jit_options,
        entity_maps=entity_maps,
    )
    assert problem.matrix.dtype == problem.load.dtype == np.dtype(np.float64)
    assert len(simulated_fem.options) == 4
    for options in simulated_fem.options:
        assert options["dtype"] is np.float64
        assert options["form_compiler_options"]["quadrature_degree"] == 7
        assert options["form_compiler_options"] is not compiler_options
        assert options["jit_options"] == jit_options and options["jit_options"] is not jit_options
        assert options["entity_maps"] is not entity_maps
    assert compiler_options == {"quadrature_degree": 7}
    problem.matrix.data[:] = 3
    problem.load[:] = 4
    assert_allclose(a.values, np.eye(2))
    assert_allclose(load.values, [1.0, 2.0])
    assert all(reference() is None for reference in simulated_fem.references)


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
    assert not simulated_fem.options


@pytest.mark.parametrize(
    "bad_load",
    [SimulatedForm(1.0, 0), SimulatedForm([1, 2], 1, "W"), SimulatedForm(np.eye(2), 2)],
)
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

    monkeypatch.setattr(forms, "import_module", unavailable)
    with pytest.raises(ImportError, match="Pixi fem"):
        fenics.from_ufl(SimulatedForm(np.eye(2), 2), SimulatedForm([0.0, 0.0]), [], [])


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


def test_local_adapter_supports_argument_preserving_ufl_zero_forms(
    ufl_spaces: tuple[Any, ...], simulated_fem: Any
) -> None:
    """ZeroBaseForm loads and trace columns use explicit local coefficient dimensions."""
    import ufl

    space = ufl_spaces[0]
    zero = ufl.ZeroBaseForm((ufl.TestFunction(space),))
    problem = fenics.from_ufl(SimulatedForm(np.eye(2), 2, space), zero, [zero], [0])
    assert_allclose(problem.load, [0.0, 0.0])
    assert_allclose(problem.coupling, [[0.0], [0.0]])
    assert len(simulated_fem.options) == 1
    assert all(reference() is None for reference in simulated_fem.references)


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
