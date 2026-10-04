"""Portable contract checks and separately marked native MPI integration tests."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.execution import mpi as distributed
from pymhm.execution.mpi import _collective_error, _indices, _values, solve_distributed
from pymhm.linalg.linear import LinearSolveError, SolverUnavailableError


@pytest.fixture
def simulated_petsc(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install an explicitly serial PETSc contract simulation, never MPI evidence."""
    from petsc_contract import PETSC

    monkeypatch.setattr(distributed, "_optional", lambda *args: PETSC)
    return PETSC


def serial_comm() -> Any:
    """Supply only the collectives required by the serial API simulation."""
    return SimpleNamespace(rank=0, allgather=lambda value: [value], allreduce=lambda value: value)


def cell(index: int) -> LocalProblem:
    """Build a two-node conservative cell with a genuine constant nullspace."""
    return LocalProblem(
        [[1.0, -1.0], [-1.0, 1.0]],
        np.diag([1.0, -1.0]),
        [0.0, 0.0],
        np.array([index, index + 1]),
        np.ones((2, 1)),
    )


@pytest.mark.parametrize("neumann", [False, True])
def test_simulated_petsc_assembly_and_physical_fields(simulated_petsc: Any, neumann: bool) -> None:
    """Check native API insertion, elimination and extraction against serial hybrid algebra."""
    boundary = [1.0, 0.0, -2.0]
    kwargs = {"boundary_load": (np.arange(3), boundary)}
    if neumann:
        kwargs = {"fixed": {0: 0.0, 2: 0.0}, "moments": [([np.ones(2) / 2] * 2, 2.0)]}
    actual = solve_distributed(
        lambda i: LocalAssembly(cell(i), i), [0, 1], trace_size=3, comm=serial_comm(), **kwargs
    )
    system = HybridSystem([cell(0), cell(1)], boundary_load=None if neumann else boundary)
    expected = (
        system.solve(
            fixed=kwargs["fixed"], constraints=[system.mean_constraint([np.ones(2) / 2] * 2, 2.0)]
        )
        if neumann
        else system.solve()
    )
    np.testing.assert_allclose(actual.fields, expected.fields, atol=1e-12)
    assert actual.local_metadata == (0, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trace_size": True},
        {"trace_size": -1},
        {"fixed": {3: 0.0}},
        {"boundary_load": ([0], [np.nan])},
        {"moments": [([], 0.0)]},
        {"moments": [([np.ones(1)], 0.0)]},
    ],
)
def test_simulated_collective_invalid_contracts(simulated_petsc: Any, kwargs: Any) -> None:
    """Reject bad local preparation before native collectives are entered."""
    options = dict(trace_size=2, comm=serial_comm()) | kwargs
    with pytest.raises(ValueError, match="preparation failed"):
        solve_distributed(cell, [0], **options)


def test_simulated_distributed_topology_and_solver_failures(
    simulated_petsc: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject missing cells, missing trace owners, factor failures and changed physical data."""
    with pytest.raises(ValueError, match="at least one"):
        solve_distributed(cell, [], trace_size=0, comm=serial_comm())
    with pytest.raises(ValueError, match="preparation failed"):
        solve_distributed(lambda i: None, [0], trace_size=2, comm=serial_comm())
    with pytest.raises(ValueError, match="unowned"):
        solve_distributed(cell, [0], trace_size=3, comm=serial_comm())
    with pytest.raises(ValueError, match="at least one unknown"):
        solve_distributed(
            lambda i: LocalProblem([[1.0]], np.empty((1, 0)), [0.0], np.array([], dtype=int)),
            [0],
            trace_size=0,
            comm=serial_comm(),
        )
    with pytest.raises(LinearSolveError, match="factorization"):
        solve_distributed(
            lambda i: LocalProblem([[1.0]], [[1.0, 1.0]], [1.0], np.array([0, 1])),
            [0],
            trace_size=2,
            comm=serial_comm(),
        )
    with pytest.raises(ValueError, match="gauge changed"):
        solve_distributed(
            lambda i: cell(i).with_load([1.0, 1.0]),
            [0],
            trace_size=2,
            comm=serial_comm(),
            fixed={0: 0.0, 1: 0.0},
            moments=[([np.ones(2)], 0.0)],
        )
    monkeypatch.setattr(simulated_petsc.KSP, "corrupt", True)
    with pytest.raises(LinearSolveError, match="residual"):
        solve_distributed(
            cell, [0], trace_size=2, comm=serial_comm(), boundary_load=([0, 1], [1.0, -2.0])
        )


def test_simulated_inconsistent_metadata_and_missing_mumps(
    simulated_petsc: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject inconsistent replicated metadata and unavailable distributed factorization."""
    comm = serial_comm()
    comm.allgather = lambda value: [value, (999,)] if isinstance(value, tuple) else [value]
    with pytest.raises(ValueError, match="must agree"):
        solve_distributed(cell, [0], trace_size=2, comm=comm)
    monkeypatch.setattr(simulated_petsc.Sys, "hasExternalPackage", lambda name: False)
    with pytest.raises(SolverUnavailableError, match="MUMPS"):
        solve_distributed(cell, [0], trace_size=2, comm=serial_comm())


@pytest.mark.parametrize("indices", [[-1], [2], [1.5], [[0]], [True]])
def test_invalid_distributed_indices(indices: Any) -> None:
    """Reject out-of-range and noninteger additive global contributions."""
    with pytest.raises(ValueError, match="indices|index"):
        _indices(indices, 2, "input")


@pytest.mark.parametrize("values", [[np.nan], [1j], [], [[0]]])
def test_invalid_distributed_values(values: Any) -> None:
    """Require finite real values with exactly one entry per contribution."""
    with pytest.raises(ValueError, match="real finite"):
        _values(values, 1, "input")


def test_collective_error_contains_rank_and_diagnosis() -> None:
    """Propagate a remote failure before other ranks enter native collectives."""
    _collective_error(SimpleNamespace(allgather=lambda error: [error]), None)
    with pytest.raises(ValueError, match="rank 1: RuntimeError: failed"):
        _collective_error(
            SimpleNamespace(allgather=lambda error: [None, error]), RuntimeError("failed")
        )
    assert _indices([], 0, "input").dtype == np.int64
    np.testing.assert_array_equal(_values([1.0], 1, "input"), [1.0])


@pytest.mark.mpi
@pytest.mark.parametrize(
    "case",
    [
        "dirichlet",
        "neumann",
        "empty-rank",
        "invalid",
        "incompatible",
        "cancellation",
        "loads",
        "kernel-roundoff",
    ],
)
def test_native_distributed_mumps(case: str) -> None:
    """Launch two actual ranks with distributed matrices, fields and failure checks."""
    pytest.importorskip("mpi4py")
    pytest.importorskip("petsc4py")
    launcher = Path(sys.executable).parent / "mpiexec"
    command = str(launcher) if launcher.exists() else shutil.which("mpiexec")
    if command is None:
        pytest.skip("MPI launcher unavailable")
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run(
        [command, "-n", "2", sys.executable, str(Path(__file__).with_name("mpi_cases.py")), case],
        capture_output=True,
        text=True,
        timeout=45,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS on 2 ranks" in result.stdout


def test_large_prescribed_reactions_cannot_mask_incompatible_free_equations(simulated_petsc):
    """Normalize physical residuals only by free equations, excluding prescribed rows."""
    with pytest.raises(ValueError, match="gauge changed"):
        solve_distributed(
            lambda i: cell(i).with_load([1.0, 1.0]),
            [0],
            trace_size=2,
            comm=serial_comm(),
            fixed={0: 1e10, 1: 1e10},
            moments=[([np.ones(2)], 0.0)],
        )


def test_prescribed_flux_cancellation_keeps_physical_load_scale(simulated_petsc: Any) -> None:
    """Judge cancellation roundoff against the original coarse balance, not its remainder."""
    actual = solve_distributed(
        lambda index: cell(index).with_load([0.1, 0.1]),
        [0],
        trace_size=2,
        comm=serial_comm(),
        fixed={0: 0.3, 1: 0.1},
        moments=[([np.ones(2)], 1.0)],
    )
    np.testing.assert_allclose(actual.fields[0], [0.4, 0.6], atol=5e-15)
    assert actual.residual < 1e-14


def test_zero_source_prescribed_roundoff(simulated_petsc: Any) -> None:
    """A zero source does not turn one rounding unit of flux cancellation into incompatibility."""
    problem = LocalProblem(
        np.diag([1.0, 0.0]),
        [[1.0, 0.0, 0.0], [0.1, 0.2, -0.3]],
        [0.0, 0.0],
        np.arange(3),
        kernel=np.array([[0.0], [1.0]]),
    )
    actual = solve_distributed(
        lambda _: problem,
        [0],
        trace_size=3,
        comm=serial_comm(),
        fixed={0: 1.0, 1: 1.0, 2: 1.0},
        moments=[([np.array([0.0, 1.0])], 2.0)],
    )
    np.testing.assert_allclose(actual.fields[0], [-1.0, 2.0], atol=1e-14)
    assert actual.residual < 1e-8


def test_local_load_cancellation_uses_preassembly_scale(simulated_petsc: Any) -> None:
    """Keep signed load cancellation distinct from incompatible physical equations."""

    def factory(index: int) -> LocalProblem:
        """Assemble independent source responses on the same trace coefficient."""
        return LocalProblem([[1.0]], [[1.0]], [0.1 * (index + 1)], np.array([0]))

    options = dict(
        trace_size=1,
        comm=serial_comm(),
        moments=[([np.ones(1), np.ones(1)], 0.3)],
    )
    actual = solve_distributed(factory, [0, 1], boundary_load=([0], [0.3]), **options)
    np.testing.assert_allclose(actual.fields, [[0.1], [0.2]], atol=2e-15)
    assert actual.residual < 1e-14
    with pytest.raises(ValueError, match="gauge changed"):
        solve_distributed(factory, [0, 1], boundary_load=([0], [0.31]), **options)
