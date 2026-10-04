"""Face-only forms assemble shared coordinates without dummy local fields."""

from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm import Equation, ExecutionConfig, LocalEquations, MultiscaleProblem, assemble
from pymhm.core.equations import compile_local_equations


def face_equations(cell, *, rectangular=False):
    """Return an independent face energy/load in explicitly ordered coordinates."""
    ids = [cell, cell + 1]
    if rectangular:
        return LocalEquations(
            np.empty((0, 0)),
            [],
            0,
            0,
            ids,
            test_dofs=ids[::-1],
            d=[[1, 3], [2, 1]],
            g=[2, 1],
        )
    return LocalEquations(np.empty((0, 0)), [], 0, 0, ids, d=[[2, -1], [-1, 2]], g=[1, 2])


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
@pytest.mark.parametrize("rectangular", [False, True])
def test_face_only_shared_faces_match_independent_assembled_system(backend, rectangular):
    problem = MultiscaleProblem(
        Equation(0, 0), partial(face_equations, rectangular=rectangular), range(2), 3, (0, 0)
    )
    system = assemble(problem, execution=ExecutionConfig(backend=backend, workers=2, batch_size=1))
    a = np.array([[2.0, -1, 0], [-1, 4, -1], [0, -1, 2]])
    f = np.array([1.0, 3, 2])
    if rectangular:
        a = np.array([[2.0, 1, 0], [1, 5, 1], [0, 1, 3]])
        f = np.array([1.0, 3, 2])
    assert_array_equal(system.matrix.toarray(), a)
    assert_array_equal(system.rhs, f)
    result = system.solve()
    assert_allclose(result.trace, np.linalg.solve(a, f), atol=1e-14)
    assert result.raw_residual < 1e-14
    assert all(field.shape == (0,) for field in result.fields)
    assert all(response.lifts.shape == (0, 2) for response in system.responses)


def test_empty_local_fields_reconstruct_without_a_factorization(monkeypatch):
    import pymhm.core.condensation as condensation
    import pymhm.core.reconstruction as reconstruction

    def forbidden(*args, **kwargs):
        raise AssertionError("an empty local space must not allocate a factorization")

    monkeypatch.setattr(condensation, "factorize", forbidden)
    monkeypatch.setattr(reconstruction, "solve_linear", forbidden)
    p = compile_local_equations(face_equations(0)).problem
    response = p.condense(refinement_precision="extended")
    assert response.source.shape == (0,)
    assert p.reconstruct([1, 2], []).shape == (0,)
    assert p.reconstruct(np.ones((2, 3)), np.empty((0, 3))).shape == (0, 3)
    assert p.reconstruct([1, 2], [], refinement_precision="extended").dtype == np.longdouble
    with pytest.raises(ValueError, match="precision"):
        p.reconstruct([1, 2], [], refinement_precision="invalid")
    with pytest.raises(ValueError, match="trace"):
        p.reconstruct([1], [])
    with pytest.raises(ValueError, match="nonsingularly"):
        compile_local_equations(
            LocalEquations(np.empty((0, 0)), [], 0, 0, [], kernel=np.empty((0, 1)))
        )
