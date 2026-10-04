"""Native MPI verification of the original conforming HPC4E reference driver."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from mpi4py import MPI
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import hpc4e_parallel as algebra  # noqa: E402
from examples import solve_hpc4e_reference as driver  # noqa: E402


def check_prescribed_coordinates() -> None:
    """Check exact prescribed coordinates against an intentionally perturbed native solve."""
    from contextlib import ExitStack

    from petsc4py import PETSc
    from scipy import sparse

    comm = MPI.COMM_WORLD
    full = sparse.csr_matrix(
        [[4.0, 1.0, 0.0, 1.0], [1.0, 3.0, 1.0, 0.0], [0.0, 1.0, 4.0, 1.0], [1.0, 0.0, 1.0, 5.0]]
    )
    lo, hi = 4 * comm.rank // comm.size, 4 * (comm.rank + 1) // comm.size
    local = full[lo:hi]
    exact = np.array([2.0, -0.3, 0.7, 1.1])
    with ExitStack() as resources:
        a = PETSc.Mat().createAIJ(
            size=((hi - lo, 4), (hi - lo, 4)),
            csr=(local.indptr, local.indices, local.data),
            comm=comm,
        )
        resources.callback(a.destroy)
        a.assemble()
        a.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        bc = a.copy()
        resources.callback(bc.destroy)
        rhs = a.createVecRight()
        resources.callback(rhs.destroy)
        rhs.array[:] = (full @ exact)[lo:hi]
        force = rhs.copy()
        resources.callback(force.destroy)
        values = a.createVecRight()
        resources.callback(values.destroy)
        values.set(0)
        prescribed = np.array([0], dtype=np.int32) if lo == 0 else np.array([], dtype=np.int32)
        if lo == 0:
            values.array[0] = exact[0]
        bc.zeroRowsColumns(prescribed + lo, diag=1.0, x=values, b=rhs)
        ksp = PETSc.KSP().create(comm)
        resources.callback(ksp.destroy)
        ksp.setOperators(bc)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")

        class PerturbedBoundary:
            """Expose a small solver error only in exactly prescribed coordinates."""

            def solve(self, b, x):
                """Solve the true matrix and perturb only the imposed value."""
                ksp.solve(b, x)
                if lo == 0:
                    x.array[0] += 1e-4

            def getConvergedReason(self):
                """Forward the native backend status."""
                return ksp.getConvergedReason()

        result, history = algebra.checked_solve(
            PerturbedBoundary(), a, rhs, force, prescribed, values, np.ones(hi - lo), resources
        )
        np.testing.assert_allclose(result.array, exact[lo:hi], rtol=2e-14, atol=2e-14)
        assert len(history) == 1, history


def run() -> None:
    """Compare all canonical fields, physical gates and five Ruiz sweeps to serial."""
    from contextlib import ExitStack

    from petsc4py import PETSc
    from scipy import sparse

    from pymhm.linalg.linear import _symmetric_equilibration

    comm = MPI.COMM_WORLD
    check_prescribed_coordinates()
    with ExitStack() as resources:
        full = sparse.csr_matrix([[0, 2e3, 0, 1], [2e3, 1e-3, 1, 0], [0, 1, -3, 4], [1, 0, 4, 2]])
        lo, hi = 4 * comm.rank // comm.size, 4 * (comm.rank + 1) // comm.size
        local = full[lo:hi]
        matrix = PETSc.Mat().createAIJ(
            size=((hi - lo, 4), (hi - lo, 4)),
            csr=(local.indptr, local.indices, local.data),
            comm=comm,
        )
        resources.callback(matrix.destroy)
        matrix.assemble()
        original, _ = algebra.compact_matrix(matrix, resources)
        balanced = original.copy()
        resources.callback(balanced.destroy)
        diagonal = algebra.symmetric_equilibration(balanced, resources)
        for column in (1, 3):
            defective = original.copy()
            resources.callback(defective.destroy)
            if lo == 0:
                defective.setValue(0, column, float(full[0, column]) + 1e-4)
            defective.assemble()
            pointer, indices, values = defective.getValuesCSR()
            try:
                algebra.check_symmetry(defective, pointer, indices, values)
            except ValueError as error:
                assert "symmetric" in str(error)
            else:
                raise AssertionError("an asymmetric owned or inter-rank coefficient was accepted")
        expected, reference_diagonal = _symmetric_equilibration(full)
        ptr, ind, values = balanced.getValuesCSR()
        actual = sparse.csr_matrix((values, ind, ptr), shape=(hi - lo, 4))
        assert_allclose(actual.toarray(), expected[lo:hi].toarray(), rtol=3e-15, atol=1e-15)
        assert_allclose(diagonal, reference_diagonal[lo:hi], rtol=3e-15)
        exact = np.array([0.3, 1.2, -2.0, 0.7])
        defect = algebra.original_residual(original, (full @ exact)[lo:hi], exact[lo:hi])
        assert np.linalg.norm(defect) < 3e-13

    data = driver.HPC4EData(
        np.tile([1e8, 2e8], (2, 1)), np.zeros((2, 2)), np.tile([1000.0, 2000.0], (2, 1))
    )
    for degree in (1, 2):
        for material in (None, data):
            field, record = driver.solve(
                4,
                4,
                degree,
                material,
                threads=1,
                factorization="ldlt",
                equilibration="symmetric",
                comm=comm,
                refinement_precision="extended",
            )
            if comm.rank == 0:
                serial, serial_record = driver.solve(
                    4,
                    4,
                    degree,
                    material,
                    threads=1,
                    factorization="ldlt",
                    equilibration="symmetric",
                    refinement_precision="extended",
                )
                for attribute in ("stress", "displacement", "rotation"):
                    assert_allclose(
                        getattr(field, attribute),
                        getattr(serial, attribute),
                        rtol=2e-10,
                        atol=2e-12,
                    )
                assert record["native_dofs"] == serial_record["native_dofs"]
                assert record["free_dofs"] == serial_record["free_dofs"]
                assert record["residual"] < 1e-12
                assert record["equilibrium_relative"] < 1e-12
                assert record["conversion_relative"] < 1e-12
                if material is not None:
                    assert record["energy_work_relative"] < 1e-12
            comm.barrier()
        scratch = comm.bcast(
            tempfile.mkdtemp(prefix="pymhm-hpc4e-ooc-") if comm.rank == 0 else None, root=0
        )
        disk, disk_record = driver.solve(
            4,
            4,
            degree,
            data,
            threads=1,
            factorization="ldlt",
            equilibration="symmetric",
            comm=comm,
            out_of_core_directory=Path(scratch),
        )
        for attribute in ("stress", "displacement", "rotation"):
            assert_allclose(
                getattr(disk, attribute), getattr(field, attribute), rtol=2e-13, atol=2e-13
            )
        assert disk_record["factor_storage"] == "out-of-core"
        assert disk_record["mumps_options"]["mat_mumps_icntl_22"] == 1
        comm.barrier()
        if comm.rank == 0:
            assert not list(Path(scratch).iterdir())
            Path(scratch).rmdir()
    if comm.rank == 0:
        print(f"HPC4E native MPI parity: PASS on {comm.size} ranks", flush=True)


if __name__ == "__main__":
    run()
