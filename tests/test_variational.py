"""Portable compiler/provider contracts and explicit global hybrid coordinates."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm.backends import fenics
from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.variational import GlobalForm, LocalForm, LocalProvider, compile_local_forms


def array_compiler(forms: LocalForm) -> LocalProblem:
    """Interpret explicit arrays as forms for portable algebraic contract tests."""
    coupling = np.column_stack(forms.trace_forms) if forms.trace_forms else np.empty((2, 0))
    constraints = None if forms.moment_forms is None else np.column_stack(forms.moment_forms)
    return LocalProblem(
        forms.a,
        coupling,
        forms.L,
        forms.trace_dofs,
        kernel=forms.kernel,
        constraints=constraints,
        coarse_basis=forms.coarse_basis,
    )


def interval_form(dofs: Any = (0, 1)) -> LocalForm:
    return LocalForm(
        np.array([[1.0, -1], [-1, 1]]),
        np.zeros(2),
        (np.array([1.0, 0]), np.array([0.0, 1])),
        np.asarray(dofs),
        kernel=np.ones((2, 1)),
        moment_forms=(np.array([0.5, 0.5]),),
    )


def test_portable_import_does_not_load_optional_backends() -> None:
    script = (
        "import sys; import pymhm.core.variational; "
        "assert not any(name.split('.')[0] in "
        "{'dolfinx','ufl','basix','mpi4py','petsc4py','cupy','gmsh'} for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)


def test_structural_provider_compiles_signed_forms_and_reconstructs() -> None:
    forms = (interval_form(), interval_form((1, 2)))
    forms = (forms[0], replace(forms[1], trace_forms=(-forms[1].trace_forms[0], [0.0, 1.0])))

    def provider(cell: int) -> LocalAssembly:
        return LocalAssembly(compile_local_forms(forms[cell], array_compiler), {"cell": cell})

    ordinary_callable: LocalProvider[int] = provider
    system = HybridSystem.from_local_factory(ordinary_callable, range(2), boundary_load=[0, 0, 2])
    result = system.solve()
    assert_allclose(result.fields, [[0, 1], [1, 2]], atol=1e-14)
    assert_allclose(result.trace, [1, -1, -1], atol=1e-14)
    assert result.raw_residual is not None and result.raw_residual < 1e-13
    assert system.local_metadata == ({"cell": 0}, {"cell": 1})


def test_explicit_global_form_applies_physical_integral_gauge() -> None:
    problem = compile_local_forms(interval_form(), array_compiler)
    system = HybridSystem([problem])
    row, target = system.mean_constraint([np.array([0.5, 0.5])], 3.0)
    spec = GlobalForm(2, (1,), fixed_trace={0: -1, 1: 1}, constraints=((row, target),))
    result = system.solve(fixed=dict(spec.fixed_trace or {}), constraints=list(spec.constraints))
    assert_allclose(result.fields[0], [3.5, 2.5], atol=1e-14)
    assert_allclose(result.gauge_multipliers, 0, atol=1e-14)


def test_records_copy_maps_bases_rows_and_preserve_wider_precision() -> None:
    basis = np.ones((2, 1))
    dofs = np.array([0, 1])
    forms = replace(interval_form(), kernel=basis, trace_dofs=dofs)
    basis[:] = 4
    dofs[:] = 5
    assert_array_equal(forms.kernel, np.ones((2, 1)))
    assert_array_equal(forms.trace_dofs, [0, 1])
    with pytest.raises(ValueError, match="read-only"):
        forms.trace_dofs[0] = 3
    boundary = np.array([1, 2], dtype=np.longdouble)
    boundary[0] += np.longdouble("1e-18")
    row = np.array([0, 0, 1], dtype=np.longdouble)
    fixed = {0: np.longdouble("1.000000000000000001")}
    spec = GlobalForm(np.int64(2), [np.int64(1)], boundary, fixed, [(row, boundary[0])])
    boundary[:] = 9
    row[:] = 9
    fixed[0] = 9
    assert spec.coarse_sizes == (1,) and isinstance(spec.trace_size, int)
    assert spec.boundary_load is not None and spec.boundary_load.dtype == np.longdouble
    assert spec.constraints[0][0].dtype == np.longdouble
    assert_array_equal(spec.constraints[0][0], [0, 0, 1])
    assert spec.fixed_trace is not None and spec.fixed_trace[0] != 9
    assert spec.constraints[0][1] == spec.boundary_load[0]
    assert_array_equal(GlobalForm(0, (0,)).constraints, ())
    assert GlobalForm(0, (0,)).fixed_trace == {}


def test_empty_forms_and_general_retained_basis() -> None:
    plain = LocalForm(np.eye(2), [1, 2], (), [])
    assert plain.trace_dofs.dtype == np.int64
    assert_allclose(compile_local_forms(plain, array_compiler).condense().source, [1, 2])
    coarse = replace(plain, coarse_basis=np.ones((2, 1)), moment_forms=([0.4, 0.6],))
    problem = compile_local_forms(coarse, array_compiler)
    assert problem.kernel.shape == (2, 0)
    assert_allclose(problem.constraints, [[0.4], [0.6]])
    assert problem.condense().coarse_vectors is not None
    assert LocalForm(None, None, (), [], moment_forms=[]).moment_forms == ()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"trace_dofs": [[0, 1]]}, "trace_dofs"),
        ({"trace_dofs": [0.0, 1.0]}, "trace_dofs"),
        ({"trace_dofs": [True, False]}, "trace_dofs"),
        ({"trace_dofs": [-1, 1]}, "trace_dofs"),
        ({"trace_dofs": np.array([0, 2**63], dtype=np.uint64)}, "trace_dofs"),
        ({"trace_dofs": [0, 0]}, "trace_dofs"),
        ({"trace_dofs": [0]}, "trace_dofs"),
        ({"coarse_basis": np.ones((2, 1))}, "mutually exclusive"),
        ({"kernel": [1, 1]}, "matrix"),
        ({"kernel": np.empty((0, 1))}, "matrix"),
        ({"kernel": [[1j], [1j]]}, "real"),
        ({"kernel": [[np.nan], [1]]}, "finite"),
        ({"kernel": [["x"], ["x"]]}, "real"),
        ({"moment_forms": ()}, "one form"),
    ],
)
def test_local_record_rejects_invalid_declarations(change: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        replace(interval_form(), **change)


def test_compiler_output_must_preserve_literal_map_and_basis() -> None:
    forms = interval_form()
    with pytest.raises(TypeError, match="LocalForm"):
        compile_local_forms(None, array_compiler)
    with pytest.raises(TypeError, match="LocalProblem"):
        compile_local_forms(forms, lambda _: LocalAssembly(array_compiler(forms)))
    with pytest.raises(ValueError, match="trace_dofs"):
        compile_local_forms(forms, lambda _: array_compiler(replace(forms, trace_dofs=[1, 0])))
    with pytest.raises(ValueError, match="kernel basis"):
        compile_local_forms(forms, lambda _: array_compiler(replace(forms, kernel=-forms.kernel)))
    plain = LocalForm(np.zeros((2, 2)), [0, 0], (), [])
    with pytest.raises(ValueError, match="undeclared kernel"):
        compile_local_forms(plain, lambda _: array_compiler(replace(plain, kernel=np.eye(2))))
    with pytest.raises(ValueError, match="undeclared retained"):
        compile_local_forms(plain, lambda _: array_compiler(replace(plain, coarse_basis=np.eye(2))))
    retained = replace(plain, coarse_basis=np.eye(2))
    rotated = np.array([[0.0, -1], [1, 0]])
    with pytest.raises(ValueError, match="retained basis"):
        compile_local_forms(
            retained, lambda _: array_compiler(replace(retained, coarse_basis=rotated))
        )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"trace_size": True}, "integers"),
        ({"trace_size": np.bool_(False)}, "integers"),
        ({"trace_size": 2.0}, "integers"),
        ({"trace_size": -1}, "integers"),
        ({"coarse_sizes": [False]}, "integers"),
        ({"coarse_sizes": [1.2]}, "integers"),
        ({"coarse_sizes": [-1]}, "integers"),
        ({"boundary_load": [1]}, "shape"),
        ({"boundary_load": [1j, 0]}, "real"),
        ({"boundary_load": [1, np.inf]}, "finite"),
        ({"fixed_trace": {True: 0}}, "keys"),
        ({"fixed_trace": {np.bool_(False): 0}}, "keys"),
        ({"fixed_trace": {0.2: 0}}, "keys"),
        ({"fixed_trace": {-1: 0}}, "keys"),
        ({"fixed_trace": {2: 0}}, "keys"),
        ({"fixed_trace": {0: np.inf}}, "finite"),
        ({"fixed_trace": {0: 1j}}, "real"),
        ({"fixed_trace": {0: [1]}}, "scalar"),
        ({"constraints": [([0, 1], 0)]}, "full reduced"),
        ({"constraints": [([0, 0, 1], [0])]}, "scalar target"),
        ({"constraints": [([0, 0, 1], np.nan)]}, "finite"),
        ({"constraints": [([0, 0, 1j], 0)]}, "real"),
    ],
)
def test_global_record_rejects_invalid_coordinates(change: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        GlobalForm(**({"trace_size": 2, "coarse_sizes": (1,)} | change))


def test_optional_adapter_passes_forms_to_existing_assembly_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forms = interval_form()
    expected = array_compiler(forms)

    def assemble(a: Any, load: Any, traces: Any, dofs: Any, **options: Any) -> LocalProblem:
        assert a is forms.a and load is forms.L
        assert traces is forms.trace_forms and dofs is forms.trace_dofs
        assert options["kernel"] is forms.kernel
        assert options["constraint_forms"] is forms.moment_forms
        assert options["coarse_basis"] is None
        return expected

    monkeypatch.setattr(fenics, "from_ufl", assemble)
    assert fenics.assemble_local_forms(forms) is expected
