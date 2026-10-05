# Performance and parallel execution

The implementation separates local finite element assembly, local condensation,
global sparse assembly and solution, and field reconstruction. Each local
condensation factors its constrained matrix once and reuses that factorization
for the source lift and all skeleton lifts. A `HybridSystem` retains those lifts
for reconstruction.

Timings include the complete numerical workflow. Numerical agreement and
performance are verified separately: a faster execution must still satisfy the
original equations and reproduce the physical fields at the stated accuracy.

## Oscillatory Darcy with declared local and global forms

The [thread notebook](https://github.com/volpatto/pymhm/blob/main/notebooks/introduction/darcy_parallel_scalability.ipynb)
defines its permeability, manufactured source, local UFL equations, skeletal
couplings and conforming Q1 comparisons in executable cells. The
[process companion](https://github.com/volpatto/pymhm/blob/main/notebooks/introduction/darcy_process_scalability.ipynb)
defines the same physical case and exports its displayed worker definitions
for `spawn`. Both use `LocalEquations`, `Equation` and `MultiscaleProblem`.

Matched fine-element counts specify the amount of fine geometry; the global
spaces differ. Separate pressure and physical-flux errors accompany timings.
Native UFL checks precede timing. Timed volume assembly uses the verified Basix
operators in both methods. Configurations are warmed before three randomized
fresh solves, including setup, pool startup, transfers,
ordered assembly, synchronization, the global solve and reconstruction.
Operators, factors and AMG hierarchies are rebuilt for every solve.

On the two-socket Xeon Silver 4216 workstation, the thread campaign measured:

| Fine quadrilaterals in each method | Classical LU | Classical AMG | Best MHM thread time | Workers |
| ---: | ---: | ---: | ---: | ---: |
| 40,000 | 1.339 s | Not measured | 2.358 s | 1 |
| 250,000 | 8.484 s | Not measured | 9.468 s | 8 |
| 1,000,000 | 48.290 s | 20.813 s | 32.951 s | 8 |

For one million elements, eight threads provide 2.125× speedup over one MHM
thread and 1.466× over classical LU. Classical AMG is faster. Pressure and flux
errors relative to the analytical solution agree within factors 1.003 and
1.001, respectively, between MHM and the conforming reference.
The smaller workloads have no speedup against classical LU.

For one million fine quadrilaterals, the process companion measured:

| Execution | Workers | Median complete time |
| --- | ---: | ---: |
| MHM serial | 1 | 69.531 s |
| MHM processes | 1 | 81.504 s |
| MHM processes | 4 | 24.772 s |
| MHM processes | 8 | 17.140 s |
| MHM processes | 16 | 13.287 s |
| Classical LU | 1 | 48.181 s |
| Classical AMG | 1 | 20.247 s |

Sixteen processes provide 5.233× speedup over serial MHM, 3.626× over classical
LU and 1.524× over classical AMG. Its three complete durations range from
13.243 to 13.428 s. Every configuration includes a fresh process pool where
applicable; child imports, serialization and pool shutdown remain timed.
Warm native initialization removes the common first-use cost from these
medians, and the records also report that initialization separately.
At 500×500 elements, eight processes take 6.153 s, compared with 8.118 s for
classical LU and 4.540 s for classical AMG. Sixteen processes take 7.385 s;
more workers do not improve that workload.

Weak scaling retains 40,000 local fine elements per worker on an expanding
domain. Median complete time rises from 2.326 s with one thread to 292.479 s
with 64 threads, giving 0.795% weak efficiency. These small local problems do
not scale efficiently with threads. The process study measures its own startup
and transfer costs on the physical domains `[0, L] × [0, 1]`, with unchanged
material period, source and boundary conditions:

| Processes / domain length | Total fine elements | MHM processes | Classical LU | Classical AMG |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 40,000 | 3.691 s | 1.043 s | 0.745 s |
| 4 | 160,000 | 5.778 s | 4.616 s | 2.954 s |
| 8 | 320,000 | 9.704 s | 10.318 s | 6.277 s |
| 16 | 640,000 | 14.631 s | 20.074 s | 12.025 s |

Weak efficiency relative to one process is 25.229% at sixteen processes.
Classical AMG remains faster on every weak workload. No inter-node efficiency
follows from either one-host study.

The host has 32 physical and 64 logical cores. Native BLAS/OpenMP budgets are
one thread, including the classical baselines. Project benchmarks run in
separate timing windows; unrelated host applications remain active and there
is no exclusive operating-system allocation or CPU affinity policy. Native
initialization is recorded with the existing compiler cache, rather than as
a cold-cache measurement. The
[thread records and figures](https://github.com/volpatto/pymhm/tree/main/benchmarks/results/execution/introduction-threads-20261004)
and [process records and figures](https://github.com/volpatto/pymhm/tree/main/benchmarks/results/execution/introduction-processes-20261004)
contain every repetition, software versions, source and lockfile hashes,
field checks and archived-basis replay conventions. This is an analytical
application, rather than a matched reproduction of a literature experiment.

## One- and two-GPU local condensation

The 500×500 pilot uses the same physical problem, 100 macroelements and
250,000 fine quadrilaterals with one or two NVIDIA RTX A5000 GPUs. Every MHM
route compiles local equations on eight CPU threads, then condenses them with
one CPU worker, eight CPU workers, one GPU or two GPUs. The CPU route with
serial condensation therefore includes parallel compilation; it is not a
complete serial MHM baseline.

All six routes run in the same locked `hpc` environment. Its SciPy version
differs from the `introduction` environment used for the thread/process
notebooks, so the pilot reruns classical LU and AMG in `hpc`. Each route has
one warmup and three randomized fresh executions. Complete GPU timers include
CPU setup and compilation, transfers, cuDSS analysis/factorization, synchronization,
global assembly/solution and field reconstruction. Factors and responses are
rebuilt for every execution.

| Execution | Median complete time |
| --- | ---: |
| MHM, serial CPU condensation | 12.731 s |
| MHM, eight CPU condensation workers | 9.932 s |
| MHM, one GPU | 9.824 s |
| MHM, two GPUs | 12.020 s |
| Classical LU | 8.330 s |
| Classical AMG | 4.521 s |

The one-GPU and threaded CPU observed ranges overlap; their 1.1% median
difference does not establish a performance advantage. Two GPUs are slower.
Neither GPU route beats classical LU or AMG in this workload. Pressure and
physical-flux errors agree with the classical baseline within factors 1.023
and 1.001. Original equations, physical moments and archived-basis replay are
checked independently of timings.

The [pilot records and figures](https://github.com/volpatto/pymhm/tree/main/benchmarks/results/execution/introduction-multigpu-20261004)
include every sample, phase cost, initialization, actual basis digests and
hardware/software provenance. CPU compilation and host analysis remain
significant parts of this execution model. These measurements establish a
node-local capability and its measured costs, without a multiGPU speedup or
inter-node scalability claim.

## Three-dimensional Darcy with process-local solves

The [3D tutorial](https://github.com/volpatto/pymhm/blob/main/notebooks/introduction/darcy_3d_parallel_scalability.ipynb)
writes its anisotropic multiscale permeability, manufactured source, local UFL
forms, oriented Q1 macroface coupling and global problem explicitly. It also
shows the independent conforming Q1 reference, physical norm integration,
complete timers and literal provider export for spawn workers. The default
small demonstration is separate from the
[resolved campaign](https://github.com/volpatto/pymhm/tree/main/benchmarks/results/execution/introduction-3d-20261004).

A unit-cube mesh of 64 macrohexahedra uses 16, 24 or 32 local fine cells per
direction at global fine counts 64, 96 or 128. These configurations contain
262,144, 884,736 or 2,097,152 fine hexahedra. The 128 setting is optional;
the record identifies the configurations actually measured. Each unsplit
macroface has four
tensor-product Q1 trace modes. The coupled problem has 960 trace coordinates
and 64 retained constants. The classical reference uses the same fine grid
with a different global approximation space. Pressure and physical vector-flux
errors accompany cost measurements; equal fine-element counts do not imply
equal accuracy. Refining local meshes with fixed macro/trace spaces can leave
an interface-error floor.

Local direct solves factor the constrained Neumann matrix for its source and
trace right-hand sides. The optional CPU/GPU AMG route projects the declared
constant, pins one coordinate for an SPD elliptic solve, restores the physical
volume moment and checks the original equations through the shared package
owner. A hierarchy built for one solver call serves its multiple right-hand
sides; refinement remains inside that owner and timer. Numerical data are
rebuilt for each independent sample. Full pressure Dirichlet data require no global
mean-zero gauge. The coupled global solve continues to use sparse direct
algebra.

Complete process times contain setup, child startup/imports, native assembly,
local solves, full response transfer, ordered shared-face accumulation,
synchronization, pool join, global solution and reconstruction. Warmups use
the existing compiler cache; fresh workers remain timed. Strong speedups use
one spawned process as their denominator. True serial execution is a distinct
configuration.
Weak domains expand in x with fixed physical material period, local mesh size
and macro width, including growth of the global problem.

CPU1 classical PyAMG and multi-process MHM use different CPU budgets. The
independent distributed PETSc CG/GAMG baseline uses 32 physical cores; MHM
uses its recorded affinity of 24 or 28 physical cores. Optional GPU components
have a separate local-condensation scope; no accepted large GPU timing is
available in this campaign. Hardware, initialization, transfers, synchronization, raw
samples and numerical controls remain part of the record. No multi-node
performance claim follows from these workstation measurements. The
[case description](cases/darcy-3d-scalability.md) states the physical data,
spaces and relation to the published MHM experiments.

### Recorded 3D complete workflow times

| Fine cells per unit axis | Method | Processes / ranks | Available physical CPU cores | Complete time (s) |
| ---: | --- | ---: | ---: | ---: |
| 64 | Classical PyAMG | 1 | 1 | 26.3190 |
| 64 | Classical PETSc CG/GAMG | 32 | 32 | 3.8283 |
| 64 | MHM PyAMG | 1 | 24 | 123.5853 |
| 64 | MHM PyAMG | 8 | 24 | 19.5680 |
| 64 | MHM PyAMG | 16 | 24 | 11.2846 |
| 64 | MHM PyAMG | 32 | 24 | 8.8042 |
| 96 | Classical PyAMG | 1 | 1 | 87.6479 |
| 96 | Classical PETSc CG/GAMG | 32 | 32 | 9.2348 |
| 96 | MHM PyAMG | 32 | 28 | 23.0012 |

Each row is one completed sample, including startup and initialization. The
64-per-axis MHM process sweep reduces complete time from 123.5853 s with one
process to 8.8042 s with 32 processes: a 14.04× internal speedup. All four
MHM points share an affinity of 24 physical cores, so the 32-process point
oversubscribes that CPU budget. The 96-per-axis MHM point uses 28 available
physical cores.

The classical MPI reference uses 32 physical cores and is faster than the
32-process MHM at both resolutions: 3.8283 versus 8.8042 s at 64, and 9.2348
versus 23.0012 s at 96. These CPU budgets and approximation spaces differ;
this is not an equal-resource or equal-accuracy comparison. The tables expose
those differences instead of inferring a crossover from them.

No completed large weak-scaling or accepted GPU-condensation timing is
available. No complete 128-per-axis workflow is measured. The source notebook
provides configurable procedures for these experiments; the present record
establishes the CPU process sweep only.

The [campaign record](https://github.com/volpatto/pymhm/tree/main/benchmarks/results/execution/introduction-3d-20261004)
states the accompanying field errors, precision controls and unmeasured scopes.

## Choosing and measuring an execution mode

Use `assemble(problem, execution=ExecutionConfig(...))` with declared local and
global forms. Serial execution builds, condenses and accumulates one cell at a
time. Process execution constructs and reduces complete local equations inside
spawned workers; ordered coordinator reduction handles shared faces. A bounded
rolling window can overlap local work with global accumulation. The
[execution guide](execution.md#serial-cells-and-bounded-parallel-batches)
specifies ordering, thread limits, failure handling and cleanup.

Keep the full discretization fixed for strong scaling. For weak scaling, report
both local work per worker and growth of the global skeleton. Retained lifts,
serialized responses and the host global matrix all contribute to memory use.
Separate initialization, local assembly/condensation, transfers, global work
and reconstruction, while retaining a complete timer containing every phase.
Report every repetition and any regression.

Ordinary `MultiscaleProblem` assembly has a host global solve. For distributed
local ownership and global sparse algebra, use
[`solve_distributed`](execution.md#distributed-mpi-assembly), with the caller
assigning cells and globally consistent trace indices to MPI ranks. For
node-local accelerator work,
[`condense_multi_gpu`](execution.md#local-work-across-multiple-gpus)
distributes local systems among explicitly selected devices. Transfers,
synchronization and host work remain part of a complete GPU comparison.
Native correctness controls and one-host measurements do not establish
multi-node efficiency.

[Recorded MPI, resident GPU and offline/online measurements](execution.md#recorded-measurements)
state their own hardware, spaces and timing scopes. Earlier CPU measurements
remain available in the
[benchmark records](https://github.com/volpatto/pymhm/tree/main/benchmarks/results).
Their measured workloads and acquisition hashes are distinct from the
oscillatory application above.
