"""Repeated arbitrary sources preserve original four-block variational equations."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import (
    Equation,
    LocalEquations,
    MultiscaleProblem,
    OfflineMultiscaleSystem,
    assemble,
    solve,
    with_global_equation,
    with_global_load,
)


@pytest.mark.parametrize("retained", ["none", "kernel", "petrov"])
def test_online_volume_and_trace_sources_match_fresh_original_assembly(retained):
    matrix = np.array([[3.0, -1], [0, 2]])
    options = {}
    if retained == "kernel":
        matrix = np.array([[1.0, -1], [-1, 1]])
        options = {"kernel": np.ones((2, 1)), "moments": np.array([[0.5], [0.5]])}
    elif retained == "petrov":
        options = {
            "coarse_basis": [[1.0], [0]],
            "test_basis": [[0.0], [1]],
            "moments": [[1.0], [0]],
            "test_moments": [[0.0], [1]],
        }
    declarations = tuple(
        LocalEquations(
            matrix,
            [1.0, -2],
            [[1.0], [-0.5]],
            [[0.25, 1.0]],
            [cell],
            d=[[4.0]],
            g=[0.3],
            metadata={"cell": cell},
            **options,
        )
        for cell in range(2)
    )
    counts = (0, 0) if retained == "none" else (1, 1)
    size = 2 + sum(counts)
    initial_global = np.linspace(0.2, 0.4, size)
    problem = MultiscaleProblem(
        Equation(np.eye(size) * 0.5, initial_global), declarations.__getitem__, range(2), 2, counts
    )
    system = assemble(problem)
    loads, balances = ([0.5, 3.0], [-1.0, 2.0]), ([0.7], [-0.9])
    global_load = np.linspace(-0.2, 0.3, size)
    fresh_declarations = tuple(
        replace(declaration, L=load, g=balance)
        for declaration, load, balance in zip(declarations, loads, balances, strict=True)
    )
    fresh = assemble(
        replace(
            problem,
            local_provider=fresh_declarations.__getitem__,
            global_equation=Equation(np.eye(size) * 0.5, global_load),
        )
    )
    with OfflineMultiscaleSystem(system) as offline:
        same = offline.with_loads([declaration.L for declaration in declarations])
        assert_allclose(same.rhs, system.rhs, atol=1e-12, rtol=1e-10)
        updated = offline.with_loads(loads, global_load=global_load, balance_loads=balances)
        actual = offline.solve(loads, global_load=global_load, balance_loads=balances)
        assert updated.matrix is system.matrix
        assert updated.local_metadata is system.local_metadata
        for response, old in zip(updated.responses, system.responses, strict=True):
            assert response.lifts is old.lifts
            assert response.coarse_vectors is old.coarse_vectors
        assert_allclose(updated.rhs, fresh.rhs, atol=1e-12, rtol=1e-10)
        expected = solve(fresh)
        assert_allclose(actual.trace, expected.trace, atol=1e-12, rtol=1e-10)
        for field, original in zip(actual.fields, expected.fields, strict=True):
            assert_allclose(field, original, atol=1e-12, rtol=1e-10)
        block = np.zeros((4 + size, 4 + size))
        rhs = np.r_[*loads, global_load]
        for cell, declaration in enumerate(fresh_declarations):
            local = slice(2 * cell, 2 * cell + 2)
            block[local, local] = declaration.a
            block[local, 4 + cell] = np.asarray(declaration.b)[:, 0]
            block[4 + cell, local] = np.asarray(declaration.c)[0]
            block[4 + cell, 4 + cell] = declaration.d[0][0]
            rhs[4 + cell] += declaration.g[0]
        if retained == "none":
            block[4:, 4:] += np.eye(size) * 0.5
            independent = np.linalg.solve(block, rhs)
            assert_allclose(
                np.r_[*actual.fields, actual.trace], independent, atol=1e-12, rtol=1e-10
            )
        # Preserve existing global update helpers and their additional-load contract.
        added = with_global_equation(system, Equation(0, np.ones(size)))
        assert_allclose(added.global_load, initial_global + 1, atol=1e-12)
        changed = with_global_load(system, system.rhs + np.r_[np.ones(2), np.zeros(size - 2)])
        assert_allclose(
            changed.global_load, initial_global + np.r_[np.ones(2), np.zeros(size - 2)], atol=1e-12
        )
    assert_allclose(solve(updated).trace, actual.trace, atol=1e-12, rtol=1e-10)
    with pytest.raises(RuntimeError, match="closed"):
        offline.with_loads(loads)
    with pytest.raises(RuntimeError, match="closed"):
        offline.__enter__()


def test_online_cache_checks_load_counts_and_closes_on_preparation_error(monkeypatch):
    import pymhm.core.online as owner

    with pytest.raises(TypeError, match="MultiscaleSystem"):
        OfflineMultiscaleSystem(None)
    declaration = LocalEquations([[2.0]], [0.0], [[1.0]], [[-1.0]], [0])
    system = assemble(MultiscaleProblem(Equation(0, 0), lambda _: declaration, [0], 1, (0,)))
    with OfflineMultiscaleSystem(system) as offline:
        with pytest.raises(ValueError, match="one volume"):
            offline.with_loads([])
        with pytest.raises(ValueError, match="one volume"):
            offline.with_loads([[1]], balance_loads=[])
        with pytest.raises(ValueError, match="shape"):
            offline.with_loads([[1, 2]])
    modified = replace(system.cells[0], child=system)
    system.cells = (modified,)
    with pytest.raises(ValueError, match="child hierarchy"):
        OfflineMultiscaleSystem(system)
    system.cells = (replace(modified, child=None),)

    def fail(*args, **kwargs):
        raise RuntimeError("failed preparation")

    monkeypatch.setattr(owner, "OfflineLocalProblem", fail)
    with pytest.raises(RuntimeError, match="failed preparation"):
        OfflineMultiscaleSystem(system)
