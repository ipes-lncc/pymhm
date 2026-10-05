"""Native rank-owned variational assembly checks with optional local CUDA factors.

CPU and GPU invocations use the same mathematical definitions and distributed
PETSc/MUMPS global solve. Dense monolithic operators below are tiny independent
test oracles; the production solve never gathers its matrix or local responses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
from mpi4py import MPI
from numpy.testing import assert_allclose
from scipy import sparse

from pymhm.core.equations import Equation, LocalEquations, compile_local_equations
from pymhm.core.system import HybridSystem
from pymhm.execution.mpi import solve_distributed


def four_block_cell(index: int) -> LocalEquations:
    """Declare sparse nonsymmetric A/B/C/D with reversed test trace coordinates."""
    return LocalEquations(
        sparse.csr_matrix([[4.0 + index, 1.0], [0.4, 3.0 + index]]),
        [0.3, 1.0 + index],
        [[1.0, 0.2], [-0.3, 1.0]],
        [[-0.6, 0.5], [0.1, -1.2]],
        [index, index + 1],
        test_dofs=[index + 1, index],
        d=[[2.0, 0.3], [-0.2, 1.5]],
        g=[0.5 + index, -0.25],
        metadata={"cell": index, "owner": MPI.COMM_WORLD.rank},
    )


def four_block_reference(fixed: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    """Solve independently numbered full physical equations before local elimination."""
    matrix, rhs = np.zeros((7, 7)), np.zeros(7)
    for index in range(2):
        equation = four_block_cell(index)
        field = np.arange(2 * index, 2 * index + 2)
        trial, test = equation.dofs + 4, equation.test_dofs + 4
        matrix[np.ix_(field, field)] += equation.a.toarray()
        matrix[np.ix_(field, trial)] += equation.b
        matrix[np.ix_(test, field)] += equation.c
        matrix[np.ix_(test, trial)] += equation.d
        rhs[field] += equation.L
        rhs[test] += equation.g
    matrix[4:, 4:] += np.diag([0.25, 0.5, 0.75])
    rhs[4:] += [0.2, -0.1, 0.3]
    for index, value in fixed.items():
        coordinate = 4 + index
        rhs -= matrix[:, coordinate] * value
        matrix[:, coordinate] = 0
        matrix[coordinate, :] = 0
        matrix[coordinate, coordinate], rhs[coordinate] = 1, value
    coefficients = np.linalg.solve(matrix, rhs)
    return coefficients[:4].reshape((2, 2)), coefficients[4:]


def check_four_blocks(comm: Any, solver: str) -> list[dict[str, Any]]:
    """Check D/g, custom trial/test maps, sparse global terms and both boundary types."""
    items = list(range(comm.rank, 2, comm.size))
    records = []
    for compiled in (False, True):
        for fixed in ({}, {0: 0.0}, {0: 0.7}):

            def factory(index: int, compiled: bool = compiled) -> Any:
                """Compile coefficients entirely inside their owning MPI rank."""
                equation = four_block_cell(index)
                return compile_local_equations(equation) if compiled else equation

            equation = (
                Equation(sparse.diags([0.25, 0.5, 0.75]), [0.2, -0.1, 0.3])
                if comm.rank == 0
                else Equation(0, 0)
            )
            result = solve_distributed(
                factory,
                items,
                trace_size=3,
                comm=comm,
                fixed=fixed,
                global_equation=equation,
                local_solver=solver,
            )
            fields, trace = four_block_reference(fixed)
            assert_allclose(result.local_trace, trace[result.local_trace_indices], atol=3e-13)
            for index, field, metadata in zip(
                items, result.fields, result.local_metadata, strict=True
            ):
                assert_allclose(field, fields[index], atol=3e-13)
                assert metadata == {"cell": index, "owner": comm.rank}
                definition = four_block_cell(index)
                assert_allclose(
                    definition.a @ field + np.asarray(definition.b) @ trace[definition.dofs],
                    definition.L,
                    atol=3e-13,
                )
            assert result.residual < 1e-12 and result.linear_residual < 1e-12
            records.append(
                {
                    "compiled": compiled,
                    "fixed": fixed,
                    "field_digest": hashlib.sha256(fields.tobytes()).hexdigest(),
                    "residual": result.residual,
                    "local_fields": [field.tolist() for field in result.fields],
                    "trace_indices": result.local_trace_indices.tolist(),
                    "local_trace": result.local_trace.tolist(),
                    "executed_bases": [basis.tolist() for basis in result.local_bases],
                }
            )
    return records


def conservative_cell(index: int) -> LocalEquations:
    """Define a literal constant kernel and a physically normalized mean moment."""
    coupling = np.diag([1.0, -1.0])
    return LocalEquations(
        sparse.csr_matrix([[1.0, -1.0], [-1.0, 1.0]]),
        [0.0, 0.0],
        coupling,
        -coupling.T,
        [index, index + 1],
        kernel=np.ones((2, 1)),
        moments=np.ones((2, 1)) / 2,
    )


def check_gauges(comm: Any, solver: str) -> list[dict[str, Any]]:
    """Compare executed kernel coordinates, physical fields and nonzero global moments."""
    count = 4
    items = list(range(comm.rank, count, comm.size))
    problems = [compile_local_equations(conservative_cell(index)).problem for index in range(count)]
    records = []
    for neumann in (False, True):
        boundary = np.array([1.0, 0.0, 0.0, 0.0, -2.0]) if not neumann else np.zeros(5)
        serial = HybridSystem(problems, boundary_load=boundary)
        weights = [np.ones(2) / 2] * count
        fixed = {0: 0.0, count: 0.0} if neumann else {}
        target = (
            6.0
            if neumann
            else sum(
                float(weight @ field)
                for weight, field in zip(weights, serial.solve().fields, strict=True)
            )
        )
        row, value = serial.mean_constraint(weights, target)
        expected = serial.solve(fixed=fixed, constraints=[(row, value)])
        actual = solve_distributed(
            conservative_cell,
            items,
            trace_size=count + 1,
            comm=comm,
            fixed=fixed,
            boundary_load=(np.arange(count + 1), boundary) if comm.rank == 0 else None,
            moments=[([weights[index] for index in items], target)],
            local_solver=solver,
        )
        assert_allclose(actual.local_trace, expected.trace[actual.local_trace_indices], atol=3e-12)
        for index, field, coarse, basis in zip(
            items, actual.fields, actual.coarse, actual.local_bases, strict=True
        ):
            assert_allclose(field, expected.fields[index], atol=3e-12)
            assert_allclose(coarse, expected.coarse[index], atol=3e-12)
            np.testing.assert_array_equal(basis, serial.responses[index].retained_basis)
            assert_allclose(
                problems[index].matrix @ field
                + problems[index].coupling @ expected.trace[problems[index].trace_dofs],
                0.0,
                atol=3e-12,
            )
        assert (
            abs(
                comm.allreduce(
                    sum(
                        float(weights[index] @ field)
                        for index, field in zip(items, actual.fields, strict=True)
                    )
                )
                - target
            )
            < 3e-12
        )
        counts = comm.allgather(len(actual.owned_coefficients))
        assert sum(counts) == actual.global_size
        if comm.size > 1:
            assert max(counts) < actual.global_size
        records.append(
            {
                "neumann": neumann,
                "basis_digests": [
                    hashlib.sha256(basis.tobytes()).hexdigest() for basis in actual.local_bases
                ],
                "executed_bases": [basis.tolist() for basis in actual.local_bases],
                "local_fields": [field.tolist() for field in actual.fields],
                "local_coarse": [coarse.tolist() for coarse in actual.coarse],
                "residual": actual.residual,
                "owned_counts": counts,
            }
        )
    return records


def check_collective_failure(comm: Any) -> None:
    """Propagate one rank's malformed D before native PETSc matrix insertion."""

    def factory(index: int) -> LocalEquations:
        """Supply one deliberately invalid direct block on rank zero."""
        definition = conservative_cell(index)
        if comm.rank == 0:
            return LocalEquations(
                definition.a, definition.L, definition.b, definition.c, definition.dofs, d=[[1.0]]
            )
        return definition

    try:
        solve_distributed(factory, [comm.rank], trace_size=comm.size + 1, comm=comm)
    except ValueError as error:
        assert "rank 0" in str(error) and "preparation failed" in str(error)
    else:
        raise AssertionError("malformed direct block was accepted")


def run(gpu: bool, record: Path | None) -> None:
    """Select one rank-local CUDA device and verify the same distributed equations."""
    comm = MPI.COMM_WORLD
    with ExitStack() as resources:
        device = None
        if gpu:
            import cupy as cp

            local = comm.Split_type(MPI.COMM_TYPE_SHARED)
            resources.callback(local.Free)
            device = local.rank
            if device >= cp.cuda.runtime.getDeviceCount():
                raise RuntimeError("one visible CUDA device per local MPI rank is required")
            resources.enter_context(cp.cuda.Device(device))
        solver = "cudss" if gpu else "scipy"
        evidence = {
            "rank": comm.rank,
            "ranks": comm.size,
            "device": device,
            "local_solver": solver,
            "four_blocks": check_four_blocks(comm, solver),
            "gauges": check_gauges(comm, solver),
        }
        check_collective_failure(comm)
        if gpu:
            cp.cuda.get_current_stream().synchronize()
        summaries = comm.allgather(evidence)
        if comm.rank == 0:
            if record is not None:
                record.parent.mkdir(parents=True, exist_ok=True)
                sources = [
                    Path(__file__).resolve(),
                    Path(solve_distributed.__code__.co_filename).resolve(),
                ]
                manifest = {
                    path.relative_to(
                        Path(__file__).resolve().parents[1]
                    ).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sources
                }
                record.write_text(
                    json.dumps({"source_sha256": manifest, "ranks": summaries}, indent=2) + "\n"
                )
            print(
                f"MPI variational {'GPU' if gpu else 'CPU'}: PASS on {comm.size} ranks", flush=True
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--record", type=Path)
    options = parser.parse_args()
    run(options.gpu, options.record)
