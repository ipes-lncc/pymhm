"""Execute native UFL providers inside real spawn workers on the same faces."""

from __future__ import annotations

import os
import pickle
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.variational_darcy import (
    NativeLocalContext,
    affine_pressure,
    build_problem,
    native_local_forms,
)
from pymhm.core.equations import LocalEquations
from pymhm.core.multiscale import assemble, solve
from pymhm.execution.cpu import ExecutionConfig

pytestmark = pytest.mark.fem


def assert_numerical_payload(value: Any) -> None:
    """Reject live optional native objects in returned reconstruction metadata."""
    if isinstance(value, dict):
        for item in value.values():
            assert_numerical_payload(item)
    else:
        assert isinstance(value, (int, np.ndarray))


def double_permeability_forms(context: NativeLocalContext) -> LocalEquations:
    """Supply a distinct user operator while preserving the native signed traces."""
    forms = native_local_forms(context)
    return replace(forms, a=2 * forms.a)


def test_user_native_form_callback_changes_physical_operator() -> None:
    """The generic assembler consumes user A rather than selecting a Darcy solver."""
    pytest.importorskip("dolfinx")
    problem = build_problem(provider="fenics")
    original = assemble(problem)
    callback = replace(problem.local_provider, native_forms=double_permeability_forms)
    modified = assemble(replace(problem, local_provider=callback))
    old, new = original.solve(), modified.solve()
    assert_allclose(new.trace, 2 * old.trace, atol=3e-13, rtol=3e-13)
    for a, b in zip(old.fields, new.fields, strict=True):
        assert_allclose(a, b, atol=3e-13, rtol=3e-13)
    for a, b in zip(original.responses, modified.responses, strict=True):
        assert_allclose(b.problem.matrix.toarray(), 2 * a.problem.matrix.toarray())


@pytest.mark.parametrize("backend,batch_size", [("serial", 1), ("process", 1), ("process", 2)])
def test_native_provider_matches_same_portable_operator_and_fields(
    backend: Any, batch_size: int
) -> None:
    pytest.importorskip("dolfinx")
    portable_problem = build_problem(provider="portable", subdivisions=2)
    native_problem = build_problem(provider="fenics", subdivisions=2)
    execution = ExecutionConfig(backend, workers=2, native_threads=1, batch_size=batch_size)
    portable = assemble(portable_problem)
    native = assemble(native_problem, execution=execution)
    expected, actual = portable.solve(), native.solve()
    assert_allclose(native.matrix.toarray(), portable.matrix.toarray(), atol=3e-14)
    assert_allclose(native.rhs, portable.rhs, atol=3e-14)
    assert_allclose(native.load_scale, portable.load_scale, atol=3e-14)
    assert_allclose(actual.trace, expected.trace, atol=3e-13)
    assert_allclose(actual.coarse, expected.coarse, atol=3e-13)
    assert actual.raw_residual is not None and actual.raw_residual < 1e-12
    for cell, (nresponse, presponse, nrecord, precord) in enumerate(
        zip(
            native.responses,
            portable.responses,
            native.local_metadata,
            portable.local_metadata,
            strict=True,
        )
    ):
        assert_numerical_payload(nrecord)
        pickle.dumps(nresponse)
        pickle.dumps(nrecord)
        assert nrecord["cell"] == precord["cell"] == cell
        assert (nrecord["assembly_pid"] != os.getpid()) == (backend == "process")
        permutation = np.argmin(
            np.linalg.norm(nrecord["points"][:, None] - precord["points"][None, :], axis=2), axis=1
        )
        assert len(np.unique(permutation)) == len(permutation)
        assert_allclose(
            nresponse.problem.matrix.toarray(),
            presponse.problem.matrix.toarray()[permutation][:, permutation],
            atol=3e-14,
        )
        assert_allclose(
            nresponse.problem.coupling, presponse.problem.coupling[permutation], atol=3e-14
        )
        assert_allclose(
            nresponse.problem.constraints, presponse.problem.constraints[permutation], atol=3e-14
        )
        assert_allclose(actual.fields[cell], expected.fields[cell][permutation], atol=3e-13)
        assert_allclose(actual.fields[cell], affine_pressure(nrecord["points"]), atol=3e-13)
    solved = solve(native_problem, execution=execution)
    assert_allclose(solved.trace, actual.trace, atol=3e-13)
    assert_allclose(solved.fields, actual.fields, atol=3e-13)
