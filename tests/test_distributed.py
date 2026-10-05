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
from pymhm.core.equations import (
    CompiledLocalEquations,
    Equation,
    LocalEquations,
    compile_form,
    compile_local_equations,
)
from pymhm.core.system import HybridSystem
from pymhm.execution import mpi as distributed
from pymhm.execution.mpi import _binary64, _collective_error, _indices, _values, solve_distributed
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
    for basis, response in zip(actual.local_bases, system.responses, strict=True):
        np.testing.assert_array_equal(basis, response.retained_basis)
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


@pytest.mark.serial
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


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("fixed", [{}, {0: 0.0}, {0: 0.7}])
def test_four_block_forms_against_independent_monolithic_equations(
    simulated_petsc: Any, compiled: bool, fixed: dict[int, float]
) -> None:
    """Preserve nonsymmetric C/D/g and permuted test coordinates under MPI compilation."""
    from scipy import sparse

    equations = LocalEquations(
        [[4.0, 1.0], [0.4, 3.0]],
        [0.3, 1.0],
        [[1.0, 0.2], [-0.3, 1.0]],
        [[-0.6, 0.5], [0.1, -1.2]],
        [0, 1],
        test_dofs=[1, 0],
        d=[[2.0, 0.3], [-0.2, 1.5]],
        g=[0.5, -0.25],
        metadata="four-block",
    )
    definition = compile_local_equations(equations) if compiled else equations
    extra = sparse.csr_matrix([[0.2, 0.1], [0.0, 0.4]])
    extra_load = np.array([0.05, -0.1])
    actual = solve_distributed(
        lambda _: definition,
        [0],
        trace_size=2,
        comm=serial_comm(),
        fixed=fixed,
        global_equation=Equation(extra, extra_load),
    )
    matrix = np.block(
        [
            [np.asarray(equations.a), np.asarray(equations.b)],
            [np.asarray(equations.c)[::-1], np.asarray(equations.d)[::-1] + extra.toarray()],
        ]
    )
    rhs = np.r_[equations.L, np.asarray(equations.g)[::-1] + extra_load]
    for index, value in fixed.items():
        coordinate = index + 2
        rhs -= matrix[:, coordinate] * value
        matrix[:, coordinate] = 0
        matrix[coordinate, :] = 0
        matrix[coordinate, coordinate] = 1
        rhs[coordinate] = value
    expected = np.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(actual.fields[0], expected[:2], atol=2e-14)
    np.testing.assert_allclose(actual.local_trace, expected[2:], atol=2e-14)
    assert actual.local_metadata == ("four-block",)
    assert actual.residual < 1e-14


def test_sparse_global_equation_owns_additional_coordinates(simulated_petsc: Any) -> None:
    """An additive sparse form can own traces absent from every local pairing."""
    from scipy import sparse

    actual = solve_distributed(
        lambda _: LocalProblem([[2.0]], [[1.0]], [1.0], np.array([0])),
        [0],
        trace_size=3,
        comm=serial_comm(),
        global_equation=Equation(sparse.diags([0.0, 2.0, 4.0]), [0.0, 6.0, 20.0]),
    )
    np.testing.assert_allclose(actual.owned_coefficients, [1.0, 3.0, 5.0], atol=1e-14)
    np.testing.assert_allclose(actual.fields, [[0.0]], atol=1e-14)


def test_custom_local_and_global_form_compiler(simulated_petsc: Any) -> None:
    """Delegate symbolic coefficients with declared shapes through one custom compiler."""
    shapes = []

    def compiler(form: Any, shape: tuple[int, ...] | None = None) -> Any:
        """Replace an application coefficient symbol while recording compiler coordinates."""
        shapes.append(shape)
        return compile_form([[2.0]] if isinstance(form, str) else form, shape)

    actual = solve_distributed(
        lambda _: LocalEquations("operator", [1.0], [[1.0]], [[-1.0]], [0], d=[[1.0]], g=[2.0]),
        [0],
        trace_size=1,
        comm=serial_comm(),
        compiler=compiler,
        global_equation=Equation(0, 0),
    )
    np.testing.assert_allclose(actual.local_trace, [5 / 3], atol=1e-14)
    assert None in shapes and (1, 1) in shapes and (1,) in shapes


@pytest.mark.parametrize(
    "equation",
    [
        "invalid",
        Equation([[1.0, 0.0]], [0.0]),
        Equation([[1j]], [0.0]),
        Equation([[1.0]], [np.nan]),
    ],
)
def test_invalid_additive_global_forms_are_collective(simulated_petsc: Any, equation: Any) -> None:
    """Reject malformed global forms before any PETSc insertion is entered."""
    with pytest.raises(ValueError, match="preparation failed"):
        solve_distributed(
            lambda _: LocalProblem([[1.0]], [[1.0]], [0.0], np.array([0])),
            [0],
            trace_size=1,
            comm=serial_comm(),
            global_equation=equation,
        )


def test_custom_global_compiler_must_preserve_declared_shape(simulated_petsc: Any) -> None:
    """A custom compiler cannot insert a matrix outside the physical coordinate map."""
    from scipy import sparse

    def compiler(form: Any, shape: tuple[int, ...] | None = None) -> Any:
        """Deliberately ignore the requested global matrix shape for this failure check."""
        return sparse.eye(2) if isinstance(form, str) else compile_form(form, shape)

    with pytest.raises(ValueError, match="preparation failed.*physical reduced coordinates"):
        solve_distributed(
            lambda _: LocalProblem([[1.0]], [[1.0]], [0.0], np.array([0])),
            [0],
            trace_size=1,
            comm=serial_comm(),
            compiler=compiler,
            global_equation=Equation("bad dimensions", [0.0]),
        )


@pytest.mark.parametrize("kind", ["matrix", "load"])
def test_invalid_compiled_direct_terms_are_collective(simulated_petsc: Any, kind: str) -> None:
    """Externally compiled records still validate their D/g before native assembly."""
    definition = CompiledLocalEquations(
        cell(0), np.zeros((1 if kind == "matrix" else 2, 2)), np.zeros(1 if kind == "load" else 2)
    )
    with pytest.raises(ValueError, match="preparation failed"):
        solve_distributed(lambda _: definition, [0], trace_size=2, comm=serial_comm())


def test_nested_multiscale_definition_is_explicitly_unsupported(simulated_petsc: Any) -> None:
    """Do not infer a distributed recursive partition from a local nested equation."""
    from pymhm.core.multiscale import MultiscaleProblem, NestedEquations

    inner = MultiscaleProblem(
        Equation(0, 0),
        lambda _: LocalEquations([[1.0]], [0.0], [[1.0]], [[-1.0]], [0]),
        [0],
        1,
        [0],
    )
    definitions = [
        NestedEquations(inner, [0], [[1.0]], [0]),
        LocalEquations(inner, [0.0], [[1.0]], [[-1.0]], [0]),
    ]
    for definition in definitions:
        with pytest.raises(ValueError, match="nested MultiscaleProblem.*unsupported"):
            solve_distributed(
                lambda _, definition=definition: definition, [0], trace_size=1, comm=serial_comm()
            )


def test_binary64_precision_contract_is_explicit(simulated_petsc: Any) -> None:
    """Retain exactly representable wider values and reject lost coefficient digits."""
    np.testing.assert_array_equal(_binary64(np.array([1.0], dtype=np.longdouble), "form"), [1.0])
    with pytest.raises(ValueError, match="finite real"):
        _binary64([np.inf], "form")
    with pytest.raises(ValueError, match="finite real"):
        _binary64([1j], "form")
    for value in (np.array([2**53 + 1], dtype=np.int64), np.array([2**64 - 1], dtype=np.uint64)):
        with pytest.raises(ValueError, match="exactly representable"):
            _binary64(value, "form")
    if np.finfo(np.longdouble).eps < np.finfo(float).eps:
        wider = np.ones(1, dtype=np.longdouble) + np.finfo(np.longdouble).eps
        with pytest.raises(ValueError, match="exactly representable"):
            _binary64(wider, "form")
        with pytest.raises(ValueError, match="preparation failed.*binary64"):
            solve_distributed(
                lambda _: LocalProblem([[1.0]], [[1.0]], [0.0], np.array([0])),
                [0],
                trace_size=1,
                comm=serial_comm(),
                global_equation=Equation([[1.0]], wider),
            )


@pytest.mark.parametrize(
    "name",
    [
        "a",
        "L",
        "b",
        "c",
        "d",
        "g",
        "moments",
        "test_moments",
        "kernel",
        "coarse_basis",
        "left_kernel",
        "test_basis",
    ],
)
def test_local_forms_reject_wider_digits_before_contract_conversion(
    simulated_petsc: Any, name: str
) -> None:
    """Check every declared form/basis before LocalProblem normalizes its arrays."""
    from dataclasses import replace

    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("this platform has no wider real coefficient dtype")
    wider = np.ones((1, 1), dtype=np.longdouble) + np.finfo(np.longdouble).eps
    definition = LocalEquations(
        [[1.0]],
        [0.0],
        [[1.0]],
        [[-1.0]],
        [0],
        coarse_basis=[[1.0]],
        moments=[[1.0]],
        test_moments=[[1.0]],
    )
    definition = replace(definition, **{name: wider.ravel() if name in {"L", "g"} else wider})
    with pytest.raises(ValueError, match="preparation failed.*binary64"):
        solve_distributed(lambda _: definition, [0], trace_size=1, comm=serial_comm())


def test_custom_compiler_wider_sparse_data_are_checked(simulated_petsc: Any) -> None:
    """A sparse symbolic compiler cannot lose extra coefficients during local construction."""
    from scipy import sparse

    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("this platform has no wider real coefficient dtype")

    def compiler(form: Any, shape: tuple[int, ...] | None = None) -> Any:
        """Return one nonrepresentable sparse matrix from a symbolic operator."""
        if isinstance(form, str):
            return sparse.csc_matrix(
                np.ones((1, 1), dtype=np.longdouble) + np.finfo(np.longdouble).eps
            )
        return compile_form(form, shape)

    definition = LocalEquations("wide", [0.0], [[1.0]], [[-1.0]], [0])
    with pytest.raises(ValueError, match="preparation failed.*binary64"):
        solve_distributed(
            lambda _: definition, [0], trace_size=1, comm=serial_comm(), compiler=compiler
        )


@pytest.mark.serial
@pytest.mark.mpi
@pytest.mark.parametrize("ranks", [1, 2])
def test_native_variational_distributed_cpu(ranks: int) -> None:
    """Execute the same four-block and gauge driver on one/two actual CPU ranks."""
    _native_variational(ranks, gpu=False)


@pytest.mark.serial
@pytest.mark.mpi
@pytest.mark.gpu
@pytest.mark.parametrize("ranks", [1, 2])
def test_native_variational_distributed_gpu(ranks: int) -> None:
    """Use one CUDA sparse factorization device per rank and distributed MUMPS globally."""
    cupy = pytest.importorskip("cupy")
    pytest.importorskip("nvmath.sparse.advanced")
    if cupy.cuda.runtime.getDeviceCount() < ranks:
        pytest.skip("one visible CUDA device per MPI rank is required")
    _native_variational(ranks, gpu=True)


def _native_variational(ranks: int, *, gpu: bool) -> None:
    """Launch the native CPU/GPU mathematical driver without global matrix gathering."""
    pytest.importorskip("mpi4py")
    pytest.importorskip("petsc4py")
    launcher = Path(sys.executable).parent / "mpiexec"
    command = str(launcher) if launcher.exists() else shutil.which("mpiexec")
    if command is None:
        pytest.skip("MPI launcher unavailable")
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run(
        [
            command,
            "-n",
            str(ranks),
            sys.executable,
            str(Path(__file__).with_name("mpi_gpu_cases.py")),
            *(["--gpu"] if gpu else []),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS on {ranks} ranks" in result.stdout
