# Execution details and repeated solves

The execution APIs operate on the same local and condensed equations as
`HybridSystem`. They provide three distinct capabilities: reuse an unchanged
operator for new loads, distribute local and global algebra across MPI ranks,
and assemble affine P1 operators or condense general local systems on GPUs. Choosing one
does not implicitly enable the others.

## Serial cells and bounded parallel batches

`assemble` consumes bound user-written problems or the explicit forms and
coordinate layout of `MultiscaleProblem`. Both reach the same execution owner.
`assemble_hybrid` uses its fixed `GlobalForm` construction;
both delegate scheduling and reduction to the same owners.
Serial execution constructs, condenses and accumulates one macrocell before
requesting the next. With `pipeline=False`, parallel execution completes a
bounded batch of local jobs, then accumulates each contribution in input order. Only the coordinator
updates shared global face entries. Matrix entries are not pre-summed per batch;
loads and their absolute scales use the same cellwise reduction as ordinary
`HybridSystem` assembly.

The workers compute the local Schur matrices and condensed loads as well as
the local solutions. The coordinator only validates and scatters these small
blocks into the global sparse system. A literal zero global bilinear form
compiles directly to a sparse zero matrix; its storage does not grow as the
square of the number of global unknowns.

```python
from pymhm import (
    Equation, ExecutionConfig, MeshHierarchy, assemble, bind_interface, bind_problem, solve,
)

problem = bind_problem(
    MeshHierarchy(macro_mesh, local_mesh_factory),
    bind_interface(skeleton, convention="normal"), local_provider,
    global_equation=Equation(global_matrix, global_rhs), retained=1,
)
system = assemble(
    problem,
    execution=ExecutionConfig(
        backend="process", workers=10, batch_size=20, pipeline=True,
    ),
)
solution = solve(system)
```

The provider receives `LocalContext` and returns the explicitly written local
equations. The mesh factory runs inside the owning worker when needed. `global_matrix` and
`global_rhs` contain additional global forms in the complete reduced coordinate
order. See the [variational guide](variational.md) for their signs and sizes.

Run process examples inside a script protected by
`if __name__ == "__main__":`. The backend uses `spawn` on every platform.
Providers own FEM, PETSc, MPI and accelerator resources within their worker.
A provider may initialize a reusable workspace on its first item and retain it
for subsequent local problems. Implement `close()` to release that state:
serial and thread execution call it after running jobs finish, including failure
or early iterator closure; process execution calls it on each spawned provider
copy at normal worker shutdown. The caller's process-provider copy remains
caller-owned. Return portable numerical operators and metadata. Thread execution
requires thread-safe native libraries and independent mutable workspaces for
concurrent calls.

UFL/DOLFINx local assembly uses native CSR/vector interfaces and does not
require PETSc. Each local mesh must use a single-rank communicator such as
`MPI.COMM_SELF`; the MPI runtime remains a DOLFINx dependency. These local
operators can be solved with SciPy, PARDISO or another compatible configured
backend in serial, thread or spawn-process execution. Solver choice is
independent of the provider. Distributed PETSc assembly below is a separate
path with its own PETSc/MUMPS requirements; see the
[FEniCS adapter](fenics.md) and [Windows FEM scope](windows.md#native-fem-scope).

`native_threads` defaults to one supported BLAS/OpenMP thread per local job.
`None` leaves native settings unchanged. Thread limits are process wide during
each complete thread batch and are restored before its results are yielded; overlapping
executions must not request conflicting limits. `batch_size=None` uses the
effective worker count in parallel. Serial execution consumes one item at a
time regardless of batch size.

With `pipeline=True`, the bound becomes an ordered rolling window. The
coordinator waits for the next cell in input order, reduces it, and refills one
slot while the remaining workers continue. This avoids a barrier after every
worker-sized batch while preserving the order of every shared-face sum.
Waiting for the next cell can still delay accumulation when that cell is slow;
a window larger than the worker count gives the other workers more pending work.
At most `batch_size` consumed inputs have unyielded outputs. In thread mode,
the process-wide native-thread limit stays active during coordinator work until
the iterator closes and workers join; in process mode the coordinator's pools
remain unchanged. The default `pipeline=False` retains complete-batch failure
semantics. In pipeline mode successful earlier cells can be yielded before a
later failure is observed. Both modes cancel pending tasks and join running
tasks on closure or failure.

`HybridProblem` custom contribution callbacks run on the coordinator by
default. Use `contribution_execution="worker"` only for an independent callback
that uses its response, metadata and supplied coarse indices without writing
shared state. `MultiscaleProblem`'s declared local forms satisfy that contract
and assemble their complete reduced contributions on the workers.

The generic `iter_local` exposes the same ordered scheduling for other local
operations. Close it when stopping early, for example with `contextlib.closing`,
so running tasks finish and workers are joined. `map_local` collects its results
into a list. `assemble_hybrid` closes the iterator on success and on exceptions.
Both interfaces bound submitted jobs; `assemble_hybrid` still retains local
responses for field reconstruction and the assembled sparse global matrix.
This is not a constant-memory global solve.
Process execution transfers complete local responses, including fine-scale
lifts and operators; those transfers remain part of the measured workflow.
Serialization can change array strides and hence floating-point summation in
field reconstruction. Verify the original equations and field agreement at
the declared numerical precision, while recording the executed basis and its
digest for coefficient replay.

For an external local solver, use
`SolverConfig(local_solver=solve_local_columns)`. The callable receives copies
of the constrained operator and RHS and returns coefficients with the same
shape, including local moment multipliers. Verification uses the untouched
original operator, independently for every RHS column. The callable owns its
native precision and resource lifecycle. The augmented local operator also
passes the existing equilibrated sparse-LU numerical rank diagnostic; its
cost is part of this interface's setup. This diagnostic is not a mathematical
inf-sup proof. This extension does not silently
replace a failed solve or change tolerances.

## Compact condensation and a second reconstruction pass

`HybridSystem.from_contributions(contributions, trace_size=..., coarse_sizes=...)`
assembles the tuples produced by `LocalResponse.global_contribution(coarse_dofs)`.
It stores the global matrix and load without retaining local lifts. A caller can
assemble, condense and release one local response at a time, solve the compact
system, then rebuild local responses to reconstruct fields. The returned field
tuple is empty; trace and coarse coefficients retain their ordinary global
equation conventions.

`coarse_sizes` declares the number of retained coordinates in each cell. Their
global slots follow the trace slots, in cell order. Each cell's returned coarse
vector follows ascending global slot order. If `coarse_dofs` permutes slots
within a cell, the caller must apply that mapping before reconstructing in the
local basis order. Passing increasing slots preserves the basis order directly.
Contributions must cover the declared global numbering and cannot use another
cell's coarse slots. Boundary moments have the same sign and additive action as
the ordinary in-memory assembly.

Physical mean rows must be supplied explicitly to `solve`; the compact system
cannot infer them from missing local lifts. Persist the executed local basis,
constraints, orientation maps and operator identity alongside coarse and trace
coefficients. A matching matrix dimension does not identify a reconstruction
basis.

## Repeated sources and boundary values

```python
from pymhm.core.offline import OfflineHybridSystem

# problems contains assembled LocalProblem instances.
with OfflineHybridSystem(problems, boundary_load=boundary_moments) as prepared:
    solutions = prepared.solve_many(source_vectors)
```

Each element of `source_vectors` is a sequence with one new load vector per
local problem. Preparation retains constrained local factorizations, source and
trace lifts, retained-mode responses, and the constrained global factorization.
An online solve computes new source lifts, assembles the changed global load,
applies the existing global factor and reconstructs the fields. It does not
rebuild the global matrix or recompute harmonic trace lifts.

`solve(loads, boundary_load=..., fixed=..., targets=...)` also changes boundary
moments, prescribed trace values and physical integral targets. The locations
of fixed trace coefficients and the physical moment vectors remain fixed.
Changing material, geometry, trial/test spaces, stabilization or boundary type
requires new preparation. Variable time steps require one preparation per
operator, such as one per distinct backward-Euler step length.

`OfflineLocalProblem` exposes the same reuse for one cell. For different cells
with an identical matrix, `LocalFactorCache` reuses a factor only when their
canonical CSC shape, indices and numerical bytes are identical. Loads and trace
couplings may differ. It never treats nearly equal matrices as interchangeable.
Both APIs own their native resources and support context managers.

## Distributed MPI assembly

`pymhm.execution.mpi.solve_distributed` is collective on the supplied mpi4py
communicator. Each rank supplies only its own cell specifications:

```python
from mpi4py import MPI
from pymhm.execution.mpi import solve_distributed

comm = MPI.COMM_WORLD
owned_cells = range(comm.rank, number_of_cells, comm.size)
solution = solve_distributed(
    factory,
    owned_cells,
    trace_size=number_of_skeleton_coefficients,
    comm=comm,
    boundary_load=owned_boundary_moments,
    fixed=prescribed_trace_values,
)
```

The factory returns `LocalEquations`, `CompiledLocalEquations`, `LocalProblem`
or `LocalAssembly(problem, metadata)`. Declared local forms use the same
`compile_local_equations` and contribution owners as `MultiscaleProblem`,
including distinct trial/test trace maps and explicit local `D/g` terms.
The rank compiles, condenses and reduces its own cells. Contributions enter a distributed
PETSc AIJ matrix using global skeleton indices; PETSc communicates entries to
their row owners. Retained coarse modes receive disjoint rank-local numbering.
MUMPS performs a distributed pivoted LU of the resulting saddle system. Only
the trace and coarse coefficients needed for owned cells are communicated back
for reconstruction. There is no gather of every local matrix, response, field
or global matrix onto one rank.

`global_equation=Equation(a, L)` supplies additional **rank-owned additive**
global forms in the physical trace/retained coordinate order, before gauge
rows. Replicating a nonzero form on every rank counts it repeatedly. Sparse
forms stay sparse through compilation and PETSc insertion. A custom `compiler`
can integrate native local forms on `COMM_SELF`; native objects stay on that
rank. Nested `MultiscaleProblem` local operators are currently unsupported by
this distributed interface. The serial/CPU hierarchy uses the ordinary
`MultiscaleProblem` API.

The distributed algebra uses real binary64. The interface checks numerical
literals in local/global forms before compilation, then checks the compiler's
arrays and supplied basis/direct records before storage. Exactly representable
integers and wider floating arrays are accepted; supplied digits cannot be
discarded. A custom compiler retains responsibility for its opaque symbolic
inputs. Previously rounded data, including a previously constructed
`LocalProblem`, cannot reveal or recover digits lost before this interface.

`boundary_load=(indices, values)` contains **additive rank-owned** contributions;
replicating a complete boundary vector would count it once per rank. `fixed`
and global physical targets must agree across ranks. `moments` contains each
rank's local physical weight vectors and the common target. Ranks with no
local cells participate normally. `DistributedHybridSolution` returns owned
global coefficient entries, local trace values, local fields and collective residuals. The
original free physical equations are checked separately from the gauge-augmented
system, so a mean constraint cannot silently balance an incompatible source.

This path requires MPI, petsc4py and PETSc built with MUMPS. It is separate from
`solver="petsc"`, which is a sequential factorization adapter used by ordinary
`HybridSystem`. It does not partition a mesh automatically. The caller assigns
cells and defines globally consistent skeleton numbering. Shared metadata can
remain replicated, as in the benchmark; local algebra and global rows are
partitioned. A MUMPS factorization failure propagates instead of selecting a
sequential fallback.

`local_bases` stores the executed retained matrix for each owned cell.
Archive these matrices and their digests with persisted coefficients; declared
kernel dimensions alone do not identify a replayable numerical basis.
For cell assignment and boundary ownership, use the focused
[MPI guide](guides/mpi.md).

## Resident batches on one GPU

`pymhm.execution.cuda.assemble_p1_batch(points, cells, diffusion=..., source=...)` performs
geometry, affine P1 volume assembly, consistent mass assembly and load assembly
on the device. `points` has shape `(batch, points_per_mesh, dimension)` and
`cells` is shared connectivity, with dimension two or three. Diffusion may be
scalar or symmetric positive-definite tensors; coefficient and source values
are constant on each fine cell. Matrices, mass and loads remain CuPy arrays.
This routine does not evaluate Python coefficient callbacks or assemble the
skeleton boundary forms on the GPU.

`BatchedFactorization(matrices)` stores an equilibrated, pivoted cuBLAS LU for
each equally sized dense matrix. `solve(rhs)` accepts multiple right-hand sides
per matrix and returns device arrays; `host=True` explicitly downloads results.
Factors, matrices, residual evaluation and iterative corrections remain resident
between calls. The ordinary residual criterion applies separately to every
matrix and RHS. Shape groups and storage are explicit: dense storage scales as
`batch * local_size**2`, and this API is intended for sufficiently small local
problems that fit device memory.

`condense_batched(problems)` groups ordinary local problems by augmented size
and RHS count, uploads their constrained systems, factors each batch and returns
host `LocalResponse` objects. This convenience function preserves the general
local algebra, including trial/test distinctions; it does not keep the complete
PDE application resident. For repeated resident solves, use
`BatchedFactorization` directly. The numerical paths use cuBLAS batched LU,
not explicit matrix inverses. GPU sparse direct solvers and GPU AMG remain
separate [solver backends](solvers.md).

## Local work across multiple GPUs

`condense_multi_gpu` accepts ordinary host `LocalProblem` objects from any
compiled local formulation:

```python
from pymhm import HybridSystem, condense_multi_gpu

responses = condense_multi_gpu(
    local_problems, devices=(0, 1), batch_size=8,
    solver="auto", dense_size_limit=512,
)
# This example has no additional D/g or global Equation terms.
system = HybridSystem.from_responses(responses, boundary_load=boundary_load)
solution = system.solve()
```

This example uses the blocks carried by `LocalProblem`. Additional direct
trace terms `D/g` and global `Equation` forms still belong to the declared
contribution assembly; condensation does not include those terms implicitly.
For distributed variational assembly, `solve_distributed` accepts complete
`LocalEquations` and can use `local_solver="cudss"` on each rank's GPU.

One dedicated worker owns each explicitly selected device, with at most one
batch in flight per device. Completed slots are refilled independently;
responses return in input order. `auto` selects dense batched LU for augmented
systems up to the declared size limit and sparse cuDSS above it. The threshold
is a storage policy, not a promised performance crossover. `solver="cudss"`
keeps all nonempty local matrices sparse; `solver="batched"` explicitly selects
dense storage. CPU compilation, transfers, synchronization and shutdown remain
part of a complete workflow. This interface returns all responses to the host
and uses the ordinary host global assembly.

GPU factorizations stay on their construction device even if the caller changes
the active device before a solve or cleanup. Batched resident arrays from a
different device are rejected explicitly; host inputs are uploaded on the owner.
Sparse cuDSS receives all source, trace and retained right-hand sides together,
using one factorization and column-major solve blocks. Its `rhs_columns` hint
sets the native width; subsequent widths reuse those factors through padded
chunks. The original operator and every RHS retain the ordinary residual test.
The adapter serializes cuDSS host API entry within each process, including
analysis, factorization launch and cleanup, following the library's
[thread-safety contract](https://docs.nvidia.com/cuda/cudss/general.html#thread-safety).
Device transfers and asynchronous GPU work remain outside this entry lock.
Separate MPI ranks have independent host analysis; a thread per GPU within
one process does not make the host analysis parallel.

## HPC placement and timing

The MPI path owns local responses and reconstructed fields on their ranks and
keeps the skeleton matrix row-distributed. Assign one MPI rank to each GPU and
enter that device's context before calling `solve_distributed` with
`local_solver="cudss"`. Use the scheduler's node-local rank and visible GPU
mapping; global rank modulo device count is unsuitable across nodes with
different allocations. `CUDA_VISIBLE_DEVICES` can give each rank a single
device, in which case its local CUDA ordinal is zero. Providers construct
native objects inside their owning rank. No live factor, communicator or CUDA
array is sent to another rank.

The locked `hpc` Pixi environment combines MPI, PETSc/MUMPS, CuPy and cuDSS on
Linux CUDA hosts. The separate `gpu` environment supports Linux and Windows
CUDA platforms without importing MPI or PETSc into the portable core. Set
native BLAS/OpenMP thread budgets before launching ranks and respect the batch
memory bound; adding CPU workers around a GPU rank does not automatically
create useful concurrent device work.

Report local-work speedup separately from complete-solve speedup. Complete
timers include setup, worker startup, transfer, sparse assembly, synchronization,
global solve and reconstruction. Strong scaling fixes the entire discretization;
weak scaling fixes local work per worker while reporting global skeleton growth.
Compare classical methods at stated pressure and physical-flux accuracy, with
their own refinement check. One-host two-rank/two-GPU controls cannot establish
multi-node performance.

For configuration snippets, use the [CPU](guides/cpu.md),
[MPI](guides/mpi.md) and [GPU](guides/gpu.md) guides. Measured speedups,
scaling figures and application data are collected in the [Gallery](gallery/index.md)
and [performance report](performance.md).
