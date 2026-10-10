# Distributed MPI

MPI partitions local work and the global skeleton matrix across ranks. CPU
spawn workers instead contribute to a global matrix owned by one coordinator.
Choose MPI when you need the distributed algebra, then supply your existing
problem definition on each rank.

## Assign cells once

```python
from mpi4py import MPI
from pymhm.execution.mpi import solve_distributed

comm = MPI.COMM_WORLD
owned_cells = problem.context.hierarchy.items[comm.rank::comm.size]
solution = solve_distributed(
    problem.local_provider,
    owned_cells,
    trace_size=problem.trace_size,
    comm=comm,
    local_solver="scipy",
)
```

This fragment assumes a bound problem with zero additional global/boundary
loads. Define `problem` before it, using the [API workflow](../tutorials/overview.md).
Every rank calls the collective solver, including ranks with no cells.
The provider creates native local meshes on `COMM_SELF` in their owning rank.
The application assigns each physical macrocell to exactly one rank; the solver
does not infer a mesh partition.

The skeleton numbering must agree across ranks. PETSc adds reduced blocks by
global index and communicates shared rows to their owners. MUMPS performs the
distributed global LU. Retained modes are numbered by rank and local cell order,
so compare reconstructed fields by macrocell identity rather than assuming
serial and MPI retained-vector indices coincide.

## Supply boundary and gauge data with ownership

```python
solution = solve_distributed(
    problem.local_provider,
    owned_cells,
    trace_size=problem.trace_size,
    comm=comm,
    boundary_load=(owned_face_indices, owned_boundary_moments),
    fixed=common_prescribed_trace_coefficients,
    moments=rank_owned_physical_moments,
    global_equation=rank_owned_global_equation,
)
```

`boundary_load` and `global_equation` are additive rank-owned contributions.
Supply each physical term once; replicating a nonzero term on every rank counts
it repeatedly. Boundary load signs follow the declared hybrid equation.
`fixed` and physical moment targets are common metadata and must agree on every
rank. A pure Neumann problem also requires compatible source/outward-flux data
and a physical pressure gauge.

Only rank-owned local responses and reconstructed fields are retained. For a
field norm, integrate squared contributions on the owned meshes and combine
them with `comm.allreduce(..., op=MPI.SUM)` before taking the square root.
Use collective synchronization and the largest rank duration for a complete
wall-time measurement.

## Launch the native environment

Save the problem definition and collective calls in `run_mpi.py`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pixi run --locked -e introduction mpiexec -n 4 python run_mpi.py
```

Respect the scheduler's CPU allocation and set native thread limits before
imports. This path requires MPI, petsc4py and a real PETSc build with MUMPS.
An unavailable MUMPS backend produces an explicit error.

The distributed interface uses real binary64 data and does not support recursive
`MultiscaleProblem` local operators. Inspect [distributed assembly](../execution.md#distributed-mpi-assembly)
for the numerical data contract and explicit additional forms. For rank-owned
CUDA devices, continue with the [GPU guide](gpu.md#combine-mpi-and-gpus).
