"""Generic native assembly contracts independent of any named differential operator."""

from __future__ import annotations

import subprocess
import sys
import weakref
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

from pymhm.backends import forms


class SimulatedForm:
    """A documented assembly-protocol fake, without a finite-element implementation."""

    def __init__(self, values: Any, *, rank: int, zero: bool = False, space: Any = None) -> None:
        """Retain coefficient payloads and independently declared argument spaces."""
        self.values = np.asarray(values)
        self.rank = rank
        self.zero = zero
        self.mesh = SimpleNamespace(comm=SimpleNamespace(size=1))
        self.space = (
            space
            if space is not None
            else SimpleNamespace(
                dofmap=SimpleNamespace(index_map=SimpleNamespace(size_local=3), index_map_bs=1),
                ufl_domain=lambda: SimpleNamespace(ufl_cargo=lambda: self.mesh),
            )
        )

    def arguments(self) -> tuple[Any, ...]:
        """Provide a space for each declared form argument."""
        return tuple(
            SimpleNamespace(ufl_function_space=lambda: self.space) for _ in range(self.rank)
        )

    def integrals(self) -> tuple[Any, ...]:
        """Distinguish a literal zero integral from a nonzero variational form."""
        return (SimpleNamespace(integrand=lambda: 0 if self.zero else 1),)

    def ufl_domains(self) -> tuple[Any, ...]:
        """Expose the single native integration mesh before compilation begins."""
        return (SimpleNamespace(ufl_cargo=lambda: self.mesh),)


@pytest.fixture
def simulated_native(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Simulate native CSR/vector ownership, scatter calls and compiler options."""
    observed = SimpleNamespace(options=[], native_refs=[], scatters=[], compile_count=0)
    actual_import = forms.import_module

    class NativeMatrix:
        """Own one assembled sparse payload until copied by the adapter."""

        def __init__(self, compiled: Any) -> None:
            """Construct the fake native matrix and track its lifetime weakly."""
            self.values = sparse.csr_matrix(compiled.source.values)
            observed.native_refs.append(weakref.ref(self))

        def scatter_reverse(self) -> None:
            """Record finalization of pending native matrix contributions."""
            observed.scatters.append("matrix")

        def to_scipy(self) -> Any:
            """Expose native sparse coefficients without changing their values."""
            return self.values

    class NativeVector:
        """Own one assembled vector until copied by the adapter."""

        def __init__(self, compiled: Any) -> None:
            """Copy the supplied fake payload and track native ownership."""
            self.array = compiled.source.values.copy()
            observed.native_refs.append(weakref.ref(self))

        def scatter_reverse(self, mode: str) -> None:
            """Require additive accumulation of vector ghost contributions."""
            assert mode == "add"
            observed.scatters.append("vector")

    class Compiled:
        """Represent a compiled form whose lifetime must not cross the adapter."""

        def __init__(self, form: Any) -> None:
            """Retain native integration data only during assembly."""
            self.source = form
            self.mesh = form.mesh
            observed.native_refs.append(weakref.ref(self))

    def compile_form(form: Any, **options: Any) -> Any:
        """Mirror DOLFINx's mutation of its private compiler-options dictionary."""
        observed.compile_count += 1
        options["form_compiler_options"]["scalar_type"] = options["dtype"]
        observed.options.append(options)
        return Compiled(form)

    fem = SimpleNamespace(
        form=compile_form, assemble_matrix=NativeMatrix, assemble_vector=NativeVector
    )
    la = SimpleNamespace(InsertMode=SimpleNamespace(add="add"))

    def import_native(module: str) -> Any:
        """Supply the documented fake API while retaining the real UFL package."""
        return {"dolfinx.fem": fem, "dolfinx.la": la}.get(module) or actual_import(module)

    monkeypatch.setattr(forms, "import_module", import_native)
    return observed


@pytest.mark.parametrize("rank", [1, 2])
def test_owned_real_coefficients_rectangular_spaces_and_copied_options(
    rank: int, simulated_native: Any
) -> None:
    """The adapter preserves coefficients and dimensions without retaining native owners."""
    original = np.arange(3 if rank == 1 else 6).reshape((3,) if rank == 1 else (3, 2))
    form = SimulatedForm(original, rank=rank)
    compiler_options = {"quadrature_degree": 5}
    jit_options = {"timeout": 10}
    entity_maps = {"native mesh": np.array([0, 1], dtype=np.int32)}
    value = forms.assemble_form(
        form,
        shape=original.shape,
        form_compiler_options=compiler_options,
        jit_options=jit_options,
        entity_maps=entity_maps,
    )
    assert_array_equal(value.toarray() if sparse.issparse(value) else value, original)
    if sparse.issparse(value):
        value.data[:] = 17
    else:
        value[0] = 17
    assert_array_equal(form.values, original)
    assert compiler_options == {"quadrature_degree": 5}
    assert jit_options == {"timeout": 10}
    assert simulated_native.options[0]["entity_maps"] is not entity_maps
    assert simulated_native.options[0]["jit_options"] is not jit_options
    assert simulated_native.options[0]["form_compiler_options"] is not compiler_options
    assert simulated_native.scatters == (["vector"] if rank == 1 else ["matrix"])
    assert all(reference() is None for reference in simulated_native.native_refs)


@pytest.mark.parametrize("rank", [0, 3])
def test_nonzero_scalar_or_higher_rank_forms_are_rejected(rank: int) -> None:
    """Only vectors and matrices can enter the generic block coefficient contract."""
    with pytest.raises(ValueError, match="linear or bilinear"):
        forms.assemble_form(SimulatedForm(1.0, rank=rank))


@pytest.mark.parametrize("shape", [(), (2, 3, 4), (-1,), (True,), (1.5,), ("2",)])
def test_declared_shape_has_valid_coefficient_dimensions(shape: Any) -> None:
    """Reject rank, sign and integer ambiguities before native compilation."""
    with pytest.raises(ValueError, match="nonnegative integer dimensions"):
        forms.assemble_form(SimulatedForm([1, 2], rank=1), shape=shape)


@pytest.mark.parametrize("zero", [False, True])
def test_shape_rank_must_match_argument_rank(zero: bool) -> None:
    """An explicitly shaped zero retains the same rank contract as a nonzero form."""
    with pytest.raises(ValueError, match="shape rank"):
        forms.assemble_form(SimulatedForm([1, 2], rank=1, zero=zero), shape=(2, 2))


@pytest.mark.parametrize("rank", [1, 2])
def test_native_result_must_match_declared_shape(rank: int, simulated_native: Any) -> None:
    """Detect compiler or layout mismatch instead of silently changing block dimensions."""
    value = [1, 2] if rank == 1 else [[1, 2]]
    with pytest.raises(ValueError, match="assembled shape"):
        forms.assemble_form(SimulatedForm(value, rank=rank), shape=(3,) if rank == 1 else (3, 2))
    assert all(reference() is None for reference in simulated_native.native_refs)


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("invalid", [complex(1, 1), np.nan, np.inf])
def test_assembled_complex_or_nonfinite_coefficients_are_rejected(
    rank: int, invalid: Any, simulated_native: Any
) -> None:
    """A generic adapter preserves the core's real and finite coefficient convention."""
    values = [invalid] if rank == 1 else [[invalid]]
    with pytest.raises(ValueError, match="real-valued|finite"):
        forms.assemble_form(SimulatedForm(values, rank=rank))
    assert all(reference() is None for reference in simulated_native.native_refs)


def test_native_vector_cannot_change_rank(simulated_native: Any) -> None:
    """A rank-one form must not return a matrix-shaped coefficient payload."""
    with pytest.raises(ValueError, match="declared form rank"):
        forms.assemble_form(SimulatedForm([[1, 2]], rank=1))


@pytest.mark.parametrize("rank", [1, 2])
def test_zero_forms_infer_native_owned_dimensions_without_compilation(rank: int) -> None:
    """Argument-preserving zeros reuse the backend's native coefficient count."""
    value = forms.assemble_form(SimulatedForm([], rank=rank, zero=True))
    expected = np.zeros((3,) if rank == 1 else (3, 3))
    assert_array_equal(value.toarray() if sparse.issparse(value) else value, expected)


def test_zero_form_requires_shape_after_ufl_erases_arguments() -> None:
    """A vanished symbolic argument cannot imply a coefficient dimension."""
    form = SimulatedForm(0, rank=0, zero=True)
    with pytest.raises(ValueError, match="shape is required"):
        forms.assemble_form(form)
    assert_array_equal(forms.assemble_form(form, shape=(2,)), [0, 0])
    assert forms.assemble_form(form, shape=(0, 3)).shape == (0, 3)


def test_real_ufl_zero_base_form_and_empty_form_do_not_load_dolfinx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UFL's distinct zero representations produce explicit dimensions portably."""
    import basix.ufl
    import ufl

    coordinate = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    domain = ufl.Mesh(coordinate)
    space = ufl.FunctionSpace(domain, basix.ufl.element("Lagrange", "triangle", 1))
    zero = ufl.ZeroBaseForm((ufl.TestFunction(space),))
    actual_import = forms.import_module

    def forbid_dolfinx(module: str) -> Any:
        """Allow caller-owned UFL while rejecting accidental native FEM imports."""
        assert not module.startswith("dolfinx")
        return actual_import(module)

    monkeypatch.setattr(forms, "import_module", forbid_dolfinx)
    assert_array_equal(forms.assemble_form(zero, shape=(3,)), [0, 0, 0])
    assert forms.assemble_form(ufl.Form([]), shape=(3, 4)).shape == (3, 4)
    with pytest.raises(ValueError, match="cannot be inferred"):
        forms.assemble_form(zero)


def test_unsupported_base_form_is_not_mistaken_for_zero(simulated_native: Any) -> None:
    """A non-integral form object must still pass the ordinary native assembly checks."""
    form = SimulatedForm([1, 2], rank=1)
    form.integrals = None
    assert_array_equal(forms.assemble_form(form), [1, 2])


def test_distributed_domain_is_rejected_before_collective_compilation(
    simulated_native: Any,
) -> None:
    """The serial adapter must not start collective JIT on a distributed domain."""
    form = SimulatedForm([1, 2], rank=1)
    form.mesh.comm.size = 2
    with pytest.raises(ValueError, match="single-rank"):
        forms.assemble_form(form)
    assert simulated_native.compile_count == 0


def test_distributed_compiled_mesh_is_also_rejected(
    simulated_native: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compiler cannot substitute a distributed mesh behind the declared domain."""
    form = SimulatedForm([1, 2], rank=1)
    form.ufl_domains = lambda: ()
    form.mesh.comm.size = 2
    with pytest.raises(ValueError, match="single-rank"):
        forms.assemble_form(form)
    assert all(reference() is None for reference in simulated_native.native_refs)


@pytest.mark.parametrize("error", [ImportError("missing module"), OSError("missing library")])
def test_optional_native_import_has_a_compatible_environment_diagnosis(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Diagnose a missing Python package and a missing native shared library equally."""

    def unavailable(module: str) -> Any:
        """Represent an unavailable optional runtime without assembling substitute data."""
        raise error

    monkeypatch.setattr(forms, "import_module", unavailable)
    with pytest.raises(ImportError, match="Pixi fem environment"):
        forms.assemble_form(SimulatedForm([1, 2], rank=1))


@pytest.mark.parametrize("axis", ["columns", "rows"])
def test_pairing_preserves_signed_order_and_accepts_an_iterator(
    axis: str, simulated_native: Any
) -> None:
    """Independent B and C pairings keep caller-supplied order and orientation signs."""
    space = object()
    columns = np.array([[1.0, -4.0], [2.0, -5.0], [3.0, -6.0]])
    supplied = (SimulatedForm(column, rank=1, space=space) for column in columns.T)
    value = forms.assemble_pairing(supplied, axis=axis)
    assert_array_equal(value, columns if axis == "columns" else columns.T)
    assert all(reference() is None for reference in simulated_native.native_refs)


def test_pairing_zero_and_empty_columns_require_explicit_size() -> None:
    """An empty trace field and an erased initial zero have declared dimensions."""
    with pytest.raises(ValueError, match="empty pairing"):
        forms.assemble_pairing([])
    assert forms.assemble_pairing([], size=3).shape == (3, 0)
    assert forms.assemble_pairing([], axis="rows", size=3).shape == (0, 3)
    zero = SimulatedForm(0, rank=0, zero=True)
    assert_array_equal(forms.assemble_pairing([zero, zero], size=2), np.zeros((2, 2)))


def test_pairing_rejects_unknown_axis_and_incompatible_spaces() -> None:
    """Do not infer a transpose or combine equal-width but distinct native bases."""
    with pytest.raises(ValueError, match="axis"):
        forms.assemble_pairing([], axis="diagonal", size=0)
    with pytest.raises(ValueError, match="same function space"):
        forms.assemble_pairing(
            [
                SimulatedForm([1, 2], rank=1, space="test"),
                SimulatedForm([3, 4], rank=1, space="trial"),
            ]
        )
    with pytest.raises(ValueError, match="must be linear"):
        forms.assemble_pairing([SimulatedForm([[1]], rank=2)])


def test_importing_generic_adapter_keeps_native_packages_optional() -> None:
    """The reusable form adapter imports through the same portable package boundary."""
    command = [
        sys.executable,
        "-c",
        (
            "import sys; from pymhm.backends.forms import assemble_form, assemble_pairing; "
            "assert not any(name.split('.')[0] in {'ufl','dolfinx','mpi4py','petsc4py'} "
            "for name in sys.modules)"
        ),
    ]
    subprocess.run(command, check=True)
