"""Small native MPI cases executed by the optional integration test subprocess."""

import sys

import numpy as np
from mpi4py import MPI
from numpy.testing import assert_allclose

from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.execution.mpi import solve_distributed


def local_cell(index: int) -> LocalAssembly:
    """Build a conservative two-node diffusion cell with one constant kernel."""
    problem = LocalProblem(
        [[1.0, -1.0], [-1.0, 1.0]],
        np.diag([1.0, -1.0]),
        [0.0, 0.0],
        np.array([index, index + 1]),
        np.ones((2, 1)),
        np.ones((2, 1)) / 2,
    )
    return LocalAssembly(problem, {"cell": index, "owner": MPI.COMM_WORLD.rank})


def kernel_roundoff_case() -> None:
    """Check declared-kernel rounding and represented incompatible physics on two ranks."""
    comm = MPI.COMM_WORLD
    assert comm.size == 2
    scale = (1e-12, 1e12)[comm.rank]
    for kind in ("roundoff", "load", "reaction-kernel", "reaction-general"):
        matrix = scale * np.array([[1.0, -1], [-1, 1]])
        matrix[0, 0] = np.nextafter(matrix[0, 0], np.inf)
        load = np.zeros(2)
        if kind == "load":
            load[0] = 1e-12 * scale
        elif kind.startswith("reaction"):
            matrix += 1e-12 * scale * np.eye(2)
        retained = {"coarse_basis" if kind == "reaction-general" else "kernel": np.ones((2, 1))}
        problem = LocalProblem(
            matrix, np.eye(2), load, np.arange(2 * comm.rank, 2 * comm.rank + 2), **retained
        )
        moments = [
            ([np.ones(2) / 2 if comm.rank == cell else np.zeros(2)], 2.0) for cell in range(2)
        ]
        try:
            result = solve_distributed(
                lambda _, problem=problem: problem,
                [0],
                trace_size=4,
                comm=comm,
                fixed=dict.fromkeys(range(4), 0.0),
                moments=moments,
            )
        except ValueError as exc:
            assert kind != "roundoff" and "gauge changed" in str(exc)
        else:
            assert kind == "roundoff"
            assert_allclose(result.fields[0], 2, rtol=0, atol=3e-15)
            assert result.residual == 0
            assert result.raw_residual > 1e-8 and result.raw_residual_norm > 0
    # A remote large kernel allowance cannot certify another rank's general row.
    if comm.rank == 0:
        matrix = 1e12 * np.array([[1.0, -1], [-1, 1]])
        matrix[0, 0] = np.nextafter(matrix[0, 0], np.inf)
        problem = LocalProblem(
            matrix,
            np.empty((2, 0)),
            np.zeros(2),
            np.empty(0, dtype=int),
            kernel=np.ones((2, 1)),
        )
    else:
        problem = LocalProblem(
            [[1e-3]], np.empty((1, 0)), [0.0], np.empty(0, dtype=int), coarse_basis=[[1.0]]
        )
    moments = [
        (
            [
                np.ones(len(problem.load)) / len(problem.load)
                if comm.rank == cell
                else np.zeros(len(problem.load))
            ],
            2.0 if cell == 0 else 1.0,
        )
        for cell in range(2)
    ]
    try:
        solve_distributed(lambda _: problem, [0], trace_size=0, comm=comm, moments=moments)
    except ValueError as exc:
        assert "gauge changed" in str(exc)
    else:
        raise AssertionError("another rank's kernel allowance masked a physical reaction")
    if comm.rank == 0:
        print(f"MPI case kernel-roundoff: PASS on {comm.size} ranks", flush=True)


def run(case: str) -> None:
    """Verify distributed fields or collective failure on genuine MPI ranks."""
    comm = MPI.COMM_WORLD
    if case == "kernel-roundoff":
        kernel_roundoff_case()
        return
    if case == "loads":
        items = list(range(comm.rank, 2, comm.size))

        def source_cell(index: int) -> LocalProblem:
            """Contribute separately signed source responses to one shared coefficient."""
            return LocalProblem([[1.0]], [[1.0]], [0.1 * (index + 1)], np.array([0]))

        actual = solve_distributed(
            source_cell,
            items,
            trace_size=1,
            comm=comm,
            boundary_load=([0], [0.3]) if comm.rank == 0 else None,
            moments=[([np.ones(1) for _ in items], 0.3)],
        )
        for index, field in zip(items, actual.fields, strict=True):
            assert_allclose(field, [0.1 * (index + 1)], atol=2e-15)
        assert actual.residual < 1e-14

        # Zero volume forcing leaves no nonzero load norm in the compatibility row.
        def zero_source(index: int) -> LocalProblem:
            """Exercise componentwise prescribed-roundoff bounds on an owned macrocell."""
            return LocalProblem(
                np.diag([1.0, 0.0]),
                [[1.0, 0.0, 0.0], [0.1, 0.2, -0.3]],
                [0.0, 0.0],
                np.arange(3),
                kernel=np.array([[0.0], [1.0]]),
            )

        roundoff_items = [0] if comm.rank == 0 else []
        balanced = solve_distributed(
            zero_source,
            roundoff_items,
            trace_size=3,
            comm=comm,
            fixed={0: 1.0, 1: 1.0, 2: 1.0},
            moments=[([np.array([0.0, 1.0]) for _ in roundoff_items], 2.0)],
        )
        for field in balanced.fields:
            assert_allclose(field, [-1.0, 2.0], atol=1e-14)
        assert balanced.residual < 1e-8
        if comm.rank == 0:
            print(f"MPI case {case}: PASS on {comm.size} ranks", flush=True)
        return
    count = 1 if case in {"empty-rank", "cancellation"} else 3
    items = list(range(comm.rank, count, comm.size))
    kwargs = {}
    if case in {"neumann", "empty-rank", "incompatible"}:
        kwargs = {
            "fixed": {0: 0.0, count: 0.0},
            "moments": [([np.ones(2) / 2 for _ in items], float(count))],
        }
    if case == "cancellation":
        kwargs = {
            "fixed": {0: 0.3, 1: 0.1},
            "moments": [([np.ones(2) for _ in items], 1.0)],
        }
    boundary = np.zeros(count + 1)
    if case == "dirichlet":
        boundary[0], boundary[-1] = 1.0, -2.0
        if comm.rank == 0:
            kwargs["boundary_load"] = (np.arange(count + 1), boundary)

    def build(index: int) -> LocalAssembly:
        """Inject one rank-local invalid contract to check collective failure propagation."""
        built = local_cell(index)
        if case == "invalid" and index == 1:
            raise ValueError("injected rank-local failure")
        if case == "incompatible":
            return LocalAssembly(built.problem.with_load([1.0, 1.0]), built.metadata)
        if case == "cancellation":
            return LocalAssembly(built.problem.with_load([0.1, 0.1]), built.metadata)
        return built

    if case in {"invalid", "incompatible"}:
        try:
            solve_distributed(build, items, trace_size=count + 1, comm=comm, **kwargs)
        except ValueError as exc:
            assert ("rank-local" if case == "invalid" else "gauge changed") in str(exc)
        else:
            raise AssertionError("invalid distributed problem was accepted")
    elif case == "cancellation":
        actual = solve_distributed(build, items, trace_size=count + 1, comm=comm, **kwargs)
        for field in actual.fields:
            assert_allclose(field, [0.4, 0.6], atol=5e-15)
        assert actual.residual < 1e-14
    else:
        actual = solve_distributed(build, items, trace_size=count + 1, comm=comm, **kwargs)
        serial = HybridSystem(
            [local_cell(index).problem for index in range(count)], boundary_load=boundary
        )
        if case in {"neumann", "empty-rank"}:
            expected = serial.solve(
                fixed={0: 0.0, count: 0.0},
                constraints=[serial.mean_constraint([np.ones(2) / 2] * count, float(count))],
            )
        else:
            expected = serial.solve()
        assert_allclose(actual.local_trace, expected.trace[actual.local_trace_indices], atol=1e-11)
        for index, field, metadata in zip(items, actual.fields, actual.local_metadata, strict=True):
            assert_allclose(field, expected.fields[index], atol=1e-11)
            assert metadata == {"cell": index, "owner": comm.rank}
        assert sum(comm.allgather(len(actual.owned_coefficients))) == actual.global_size
        assert len(actual.owned_coefficients) < actual.global_size
        assert actual.residual < 1e-10
    if comm.rank == 0:
        print(f"MPI case {case}: PASS on {comm.size} ranks", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
