# CUDA and multiple GPUs

Choose a CUDA backend independently of the local variational form. UFL/DOLFINx
assembly remains on the CPU; selecting a device solver moves the numerical
factorization and solves. Distinct materials still have distinct local matrices.

## Solve on one device

```python
from pymhm import SolverConfig, assemble

system = assemble(
    problem,
    solvers=SolverConfig(local_solver="cudss", global_solver="scipy"),
)
solution = system.solve()
```

The cuDSS backend factors general constrained sparse local saddle systems with
pivoting. The global SciPy solve stays on the CPU. Use a supported CUDA device
and driver with CuPy, nvmath and cuDSS; the locked `introduction-gpu` profile
also supplies UFL assembly. See [installation](../installation.md).

## Assign batches to several devices

The low-level numerical interface accepts assembled `LocalProblem` objects:

```python
from pymhm.execution.cuda import condense_multi_gpu

responses = condense_multi_gpu(
    local_problems,
    devices=(0, 1),
    batch_size=8,
    solver="auto",
    dense_size_limit=512,
)
```

One dedicated worker thread owns each visible device. `auto` uses pivoted dense
batched LU for sufficiently small augmented systems and sparse cuDSS above the
size limit. Dense storage grows quadratically with each local system's size;
choose `batch_size` according to device memory. Responses preserve input order
and contain host data after transfer and synchronization complete.

For the basic hybrid equations with no additional direct global `D/g` forms,
construct the global system from those responses:

```python
from pymhm.core.system import HybridSystem

system = HybridSystem.from_responses(
    responses, boundary_load=boundary_moments, metadata=local_metadata
)
solution = system.solve(fixed=prescribed_trace_values)
```

Here boundary moments and prescribed trace coefficients follow your formulation's
convention. If the records also contain direct global terms, assemble those terms
explicitly with the response contributions; `from_responses` alone does not
recover them. The ordinary `assemble(problem)` path handles declared terms
automatically. See [explicit contributions](../execution.md#compact-condensation-and-a-second-reconstruction-pass).

## Combine MPI and GPUs

Give each rank one visible device through the scheduler. Set its device context
before solving the rank-owned cells:

```python
import cupy as cp
from mpi4py import MPI
from pymhm.execution.mpi import solve_distributed

comm = MPI.COMM_WORLD
owned_cells = problem.context.hierarchy.items[comm.rank::comm.size]
with cp.cuda.Device(0):
    solution = solve_distributed(
        problem.local_provider,
        owned_cells,
        trace_size=problem.trace_size,
        comm=comm,
        local_solver="cudss",
    )
```

This fragment assumes zero additional global/boundary loads, as in the basic
[MPI fragment](mpi.md#assign-cells-once). Include owned boundary and global terms
for your actual problem. If each rank sees one assigned GPU, its CUDA ordinal is
zero. If all GPUs are visible, use the MPI **node-local** rank and the scheduler's
allocation to select a device. Global rank modulo device count does not describe
a general multi-node placement policy. PETSc/MUMPS still owns the distributed
global solve.

The Linux `hpc` profile supplies MPI/PETSc/CUDA for assembled numerical operators;
`introduction-gpu` additionally supplies DOLFINx. The separate `gpu` profile
supports device algebra on Linux and Windows.

## Choose an admissible AMG operator

The full constrained Neumann local matrix is indefinite. An SPD-only AMG preset
cannot replace its LU directly. The supported projected local AMG path enforces
kernel compatibility and restores physical moments; alternatively use a valid
saddle block preconditioner. The [solver guide](../solvers.md) specifies these
requirements and the independently checked original-equation residuals.

A device solver is not a speedup guarantee. Measure setup, uploads, hierarchy or
factor construction, repeated RHS work, synchronization, downloads and global
solution. Recorded applications and scaling plots belong to the
[Gallery](../gallery/index.md), separately from these configuration instructions.
