# Repeated solves, distributed assembly and resident GPU algebra

The execution APIs operate on the same local and condensed equations as
`HybridSystem`. They provide three distinct capabilities: reuse an unchanged
operator for new loads, distribute local and global algebra across MPI ranks,
and assemble/factor batches of affine P1 local operators on a GPU. Choosing one
does not implicitly enable the others.

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
basis. The [periodic acquisition](https://github.com/volpatto/pymhm/blob/main/docs/cases/periodic.md) implements this procedure
for its declared constant kernel and fixed Q1/P0 spaces.

## Repeated sources and boundary values

```python
from pymhm.offline import OfflineHybridSystem

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

`pymhm.distributed.solve_distributed` is collective on the supplied mpi4py
communicator. Each rank supplies only its own cell specifications:

```python
from mpi4py import MPI
from pymhm.distributed import solve_distributed

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

The factory returns `LocalProblem` or `LocalAssembly(problem, metadata)`. The
rank assembles and condenses its own cells. Contributions enter a distributed
PETSc AIJ matrix using global skeleton indices; PETSc communicates entries to
their row owners. Retained coarse modes receive disjoint rank-local numbering.
MUMPS performs a distributed pivoted LU of the resulting saddle system. Only
the trace and coarse coefficients needed for owned cells are communicated back
for reconstruction. There is no gather of every local matrix, response, field
or global matrix onto one rank.

`boundary_load=(indices, values)` contains **additive rank-owned** contributions;
replicating a complete boundary vector would count it once per rank. `fixed`
and global physical targets must agree across ranks. `moments` contains each
rank's local physical weight vectors and the common target. Ranks with no
local cells participate normally. `DistributedHybridSolution` returns owned
global rows, local trace values, local fields and collective residuals. The
original free physical equations are checked separately from the gauge-augmented
system, so a mean constraint cannot silently balance an incompatible source.

This path requires MPI, petsc4py and PETSc built with MUMPS. It is separate from
`solver="petsc"`, which is a sequential factorization adapter used by ordinary
`HybridSystem`. It does not partition a mesh automatically. The caller assigns
cells and defines globally consistent skeleton numbering. Shared metadata can
remain replicated, as in the benchmark; local algebra and global rows are
partitioned. A MUMPS factorization failure propagates instead of selecting a
sequential fallback.

Native tests launch two MPI ranks and exercise nonhomogeneous Dirichlet data,
Neumann mean constraints, an empty rank and collective error propagation.
The recorded performance study uses one host; it cannot establish inter-node
communication efficiency.

## Resident batches on one GPU

`pymhm.gpu.assemble_p1_batch(points, cells, diffusion=..., source=...)` performs
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

## Recorded measurements

The campaign in `benchmarks/execution_modes.py` records one untimed warmup and
three repetitions, native library threads fixed to one, source hashes, hardware
and numerical checks. MPI times include owned local assembly, condensation,
distributed global assembly/solution and reconstruction. They use the affine
Darcy patch on the unit square; the largest field/flux/residual check was
`4.55e-13`.

| Macrotriangles / fine subdivisions | One MPI rank | Two MPI ranks | Four MPI ranks | Four-rank gain |
| --- | ---: | ---: | ---: | ---: |
| 32 / 16 | 1.202 s | 0.683 s | 0.431 s | 2.786× |
| 128 / 32 | 14.408 s | 7.252 s | 3.870 s | 3.723× |

All ranks ran on the same Linux workstation, with MPICH 4.3.1 and PETSc 3.23.0.
The workloads contain 8,192 and 131,072 fine triangles. The global matrices have
88 and 336 unknowns: these are local-work-dominated cases, not a study of large
coarse-solver scaling. Other project campaigns were paused during acquisition;
the host had no exclusive operating-system reservation or CPU affinity policy.
Individual durations and ownership counts are in
[the MPI reports](https://github.com/volpatto/pymhm/tree/main/benchmarks/results/execution).

For 32 macrotriangles with 16 fine subdivisions, twelve changing source/boundary
queries per preparation gave median preparation `0.08227 s`, median online
query `0.01973 s`, and median independently refactored query `0.10012 s`.
Both query paths exclude their common `0.93095 s` finite-element assembly.
The ratio of query medians is `5.075`; preparation must also be included when
assessing a complete sequence. Fields agreed to `1.78e-15`. The manufactured
family contains a quadratic potential; agreement compares the same P1 discrete
solutions and does not claim that quadratic fields are exactly represented.

On an NVIDIA GeForce RTX 3060 with CuPy 14.2.0, batches of 64 local P1 problems
used 91 or 325 pressure unknowns plus one mean multiplier. With twelve repeated
multi-RHS solves, median complete device-path times were `0.02659 s` and
`0.10565 s`. Timers include initial uploads, resident volume assembly,
constraint/boundary upload, LU, first and repeated solves, synchronization and
final download. Maximum differences from CPU local lifts were `9.77e-15` and
`4.24e-14`. These timings demonstrate the measured resident path; no GPU
speedup is claimed because an equivalent CPU batch campaign was not measured.
The [GPU and offline records](https://github.com/volpatto/pymhm/tree/main/benchmarks/results/execution)
preserve phase times and every repetition.

```bash
pixi run -e fem mpiexec -n 4 python benchmarks/execution_modes.py mpi --mesh 8 --refinement 32 --output mpi.json
pixi run -e test python benchmarks/execution_modes.py offline --mesh 4 --refinement 16 --queries 12 --output offline.json
pixi run -e gpu python benchmarks/execution_modes.py gpu --refinement 24 --batch 64 --queries 12 --output gpu.json
```
