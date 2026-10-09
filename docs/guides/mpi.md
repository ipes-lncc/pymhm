# Distributed MPI

MPI distributes both local algebra and the global skeleton matrix. It is a
separate capability from [spawned CPU workers](cpu.md), which reduce the global
matrix on one coordinator. Use the same
[heterogeneous Darcy problem](heterogeneous-darcy.md) to compare them.

## 1. Assign each macrocell once

```python
from mpi4py import MPI
from pymhm.execution.mpi import solve_distributed
from examples.guides.heterogeneous_execution import declared_problem

comm = MPI.COMM_WORLD
problem = declared_problem(n=2, refinement=8)
owned_cells = problem.context.hierarchy.items[comm.rank::comm.size]

solution = solve_distributed(
    problem.local_provider,
    owned_cells,
    trace_size=problem.trace_size,
    comm=comm,
    local_solver="scipy",
)
```

Every rank calls the collective solver, including ranks with no cells.
`problem.local_provider` is the bound mathematical callback: it creates its
local context and native `COMM_SELF` mesh on the requesting rank, compiles the
same UFL equations and releases its native resources. The local numerical
responses remain rank owned.

PETSc adds reduced blocks to the distributed matrix using global trace
indices. It communicates shared rows to their owners, and MUMPS performs the
distributed global LU. Retained coordinates are numbered by rank and local
cell order; compare physical fields by macrocell identity, not by assuming
serial and MPI retained-vector indices match.

## 2. Launch the installed native stack

Save the cell above as `mpi_darcy.py`, then launch:

```bash
pixi run --locked -e introduction mpiexec -n 2 python mpi_darcy.py
```

Set BLAS/OpenMP limits before launching when the scheduler grants one core per
rank. For example, `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` avoid nested
oversubscription. The current distributed solver requires real binary64 data,
MPI, petsc4py and a PETSc build with MUMPS. It does not silently substitute a
sequential solver when MUMPS is unavailable.

## Boundary and gauge ownership

The common guide has zero prescribed pressure, so no nonzero boundary vector
is replicated. For a general problem:

```python
solution = solve_distributed(
    factory,
    owned_cells,
    trace_size=trace_size,
    comm=comm,
    boundary_load=(owned_face_indices, owned_pressure_moments),
    fixed=common_prescribed_trace_coefficients,
    moments=rank_owned_physical_moment_rows,
)
```

The boundary vector and additional global forms are **additive rank-owned**
terms. Supply each physical term once. Prescribed trace coefficients and
physical targets are common metadata and must agree on every rank. A pure
Neumann problem requires its physical compatibility condition and a declared
pressure gauge; a gauge cannot repair an incompatible source. The native
tests cover nonhomogeneous Dirichlet data, Neumann means, empty ranks and
collective failures.

## 3. Measure collective physical errors

```python
from examples.guides.heterogeneous_execution import run_mpi

solution, report = run_mpi(problem, comm)
if comm.rank == 0:
    print(report)
```

Each rank integrates pressure and physical Darcy-flux squared errors on its
own fine meshes; an MPI sum produces the common L2 norms. The complete clock
uses the largest rank time after synchronization. The local fields and
coefficient-basis metadata remain available for rank-local reconstruction.

The guide compares one and two ranks against the same serial problem. This
one-host check establishes its distributed assembly contract, not multi-node
scalability. See the [execution reference](../execution.md#distributed-mpi-assembly)
for additional global forms, explicit gauges and current nesting restrictions.

## References

- Antônio Tadeu A. Gomes, Weslley S. Pereira, Frédéric Valentin and Diego
  Paredes (2017). *On the Implementation of a Scalable Simulator for Multiscale
  Hybrid-Mixed Methods*. [arXiv: 1703.10435v1](https://arxiv.org/abs/1703.10435v1).
