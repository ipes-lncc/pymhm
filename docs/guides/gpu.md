# CUDA and multiple GPUs

Use the [same heterogeneous Darcy problem](heterogeneous-darcy.md). Choosing a
CUDA solver changes the local numerical algebra, while its UFL/DOLFINx forms
are still assembled on the CPU. Local matrices remain distinct when their
materials differ.

## 1. Solve local saddle systems on one GPU

```python
from pymhm import SolverConfig, assemble

system = assemble(
    problem,
    solvers=SolverConfig(local_solver="cudss", global_solver="scipy"),
)
solution = system.solve()
```

The cuDSS backend factors the constrained sparse Neumann saddle systems with
compatible pivoting. The global SciPy factorization remains on the CPU.
Uploads, CUDA setup, factorization, residual checks, downloads and resource
release are part of this workflow. Install the `introduction-gpu` profile and
use a CUDA-capable device/driver; see [CUDA installation](../installation.md).

## 2. Batch small independent local systems

The explicit numerical interface can assemble the same local UFL records
first, then distribute condensation across two devices:

```python
from pymhm.core.equations import compile_local_equations
from pymhm.core.system import HybridSystem
from pymhm.execution.cuda import condense_multi_gpu

records = [
    compile_local_equations(problem.local_provider(cell))
    for cell in problem.context.hierarchy.items
]
responses = condense_multi_gpu(
    (record.problem for record in records),
    devices=(0, 1),
    batch_size=8,
    solver="batched",
)
system = HybridSystem.from_responses(
    responses, metadata=(record.metadata for record in records)
)
solution = system.solve()
```

This example's additional `D/g` and exterior pressure loads are zero, so the
response-only global constructor represents the same problem. A general
`LocalEquations` record can also contain nonzero direct global terms: preserve
and add those terms explicitly when using this low-level interface. The
ordinary `assemble` path handles them automatically.

One dedicated worker thread owns each explicit device. Small size/RHS groups
use pivoted batched LU; dense storage grows quadratically with each augmented
local size. `solver="auto"` selects sparse cuDSS above its configured dense
size limit. Transfers and stream synchronization finish before return.

## 3. Combine MPI and rank-owned GPUs

Inside an MPI launch, give each rank one visible GPU through the scheduler:

```python
import cupy as cp
from mpi4py import MPI
from examples.guides.heterogeneous_execution import declared_problem, run_mpi

# Scheduler allocation: one visible device per rank; its local CUDA ordinal is 0.
with cp.cuda.Device(0):
    solution, report = run_mpi(
        declared_problem(n=2, refinement=8), MPI.COMM_WORLD, local_solver="cudss"
    )
```

Use the Linux `introduction-gpu` profile for this UFL/MPI/PETSc/CUDA example.
The lean `hpc` profile supplies MPI/PETSc/CUDA for already assembled numerical
operators; it does not install DOLFINx. On a single host with both GPUs
visible to every rank, assign distinct devices using the MPI shared-memory
communicator's **node-local** rank; the notebook demonstrates that mapping.
A global-rank modulo device-count rule is not a general multi-node allocation
policy. PETSc/MUMPS still solves the distributed global saddle system.

## AMG has operator requirements

The local Neumann system with its integral constraint is indefinite. An
SPD-only AMG backend is not interchangeable with its sparse LU. A projected
positive-definite complement or a genuinely coercive local operator can use
the compatible AMG workflow, with separate original-equation checks after
reconstruction. The [solver guide](../solvers.md) states each backend's
requirements. The [3D accelerator campaign](../cases/darcy-3d-accelerators.md)
documents its particular projected/iterative local systems and controls.

The guide executes one-device cuDSS, two-device batched condensation and
two-rank/two-device sparse LU on its actual installed hardware. Agreement is
checked against the same CPU coefficients and field norms. This small problem
is a correctness demonstration; it does not establish a speedup. The gallery
retains measured absent gains as well as measured gains.
