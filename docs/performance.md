# Performance and parallel execution

The implementation separates local finite element assembly, local condensation,
global sparse assembly and solution, and field reconstruction. Each local
condensation factors its constrained matrix once and reuses that factorization
for the source lift and all skeleton lifts. A `HybridSystem` retains those lifts
for reconstruction.

All timings on this page measure pyMHM's execution paths and solver backends.
The [MSL comparison](https://github.com/volpatto/pymhm/blob/main/docs/cases/reference-comparison.md) and
[NeoPZ comparison](https://github.com/volpatto/pymhm/blob/main/docs/cases/neopz.md) report numerical field agreement separately.
The larger-local measurements compare `solve_darcy` with its default parent-process
assembly against a factory path that assembles complete local problems inside workers.

## Current execution model

`solve_darcy(..., backend="thread", workers=2)` or `backend="process"` distributes
independent **local condensation** tasks. With `parallel_assembly=True`, the same
solver instead builds and condenses each complete primal or RT0 local problem
inside its worker. The default preserves parent-process assembly, including
support for non-picklable material/source closures. Global assembly, the global
solve and final field reconstruction remain sequential. Local independence
alone does not establish an end-to-end speedup.

Process execution uses `spawn` on every platform. Matrices and right-hand sides
are serialized. For `parallel_assembly=True`, material/source callbacks and local
mesh specifications must also be picklable; condensation-only mode evaluates
those callbacks in the parent. Protect executable scripts with
`if __name__ == "__main__":`. Each complete parallel solve creates a new worker pool.

`map_local` limits BLAS/OpenMP threads to one by default to avoid nested worker
and native-thread oversubscription. Its `native_threads` argument can change that
limit for custom local workflows. In thread mode, the limit applies to the whole
process for the duration of the map. Avoid overlapping maps with conflicting
thread settings.

The ordinary `HybridSystem` global algebra is a single-process sparse system.
Selecting `solver="petsc"` invokes its sequential factorization adapter. The
separate [`solve_distributed` API](execution.md#distributed-mpi-assembly) assigns
local construction, condensation and reconstruction to owning MPI ranks and
assembles a distributed PETSc matrix solved by MUMPS. Its recorded one-, two-
and four-rank campaign and the offline/online and resident GPU measurements
are documented in [execution modes](execution.md#recorded-measurements).

## Assemble complete local problems inside workers

`HybridSystem.from_local_factory(factory, items, backend="process", workers=4)`
executes `factory(item)` and one local condensation together inside each worker.
The factory returns `LocalProblem`, or `LocalAssembly(problem, metadata)` when
field reconstruction also needs a mesh or DOF map. The ordered metadata are
available as `system.local_metadata`; they do not require remeshing or a second
factorization in the parent. Global assembly, the skeleton solve and final field
reconstruction remain sequential.

The `HybridSystem(problems, ...)` interface accepts already assembled local
problems. `solve_darcy(..., parallel_assembly=True)` uses the factory interface
for its own finite-element assembly. The same interface accepts application-defined
local operators. Factories,
specifications and metadata must be picklable for spawn workers. Serial and thread execution accept closures, subject to the
thread-safety requirements of the chosen finite-element backend.

Create and release native FEM, PETSc, MPI and GPU resources inside a worker;
transport numerical NumPy/SciPy data rather than native handles. GPU scheduling
across multiple spawned workers is not established by the CPU measurements.
The [architecture](architecture.md) describes the factory contract.
`benchmarks/large_local.py` contains an executable P1 factory using the public
element kernels and returning each fine mesh as metadata. Tests compare its
complete pressure, physical flux and skeleton fields with `solve_darcy` under
all three execution modes, and verify that each local problem is condensed once.

## Reproducible CPU measurement

```bash
pixi run -e test benchmark --output benchmark-results/scaling.json
pixi run -e test benchmark --quick --output benchmark-results/quick.json
pixi run -e test benchmark --parallel-assembly --workers 4 --output benchmark-results/worker-assembly.json
```

Render the recorded reports without rerunning the numerical benchmarks:

```bash
pixi run -e notebooks python examples/plot_performance.py
```

`benchmarks/scaling.py` solves the affine manufactured Darcy problem
\(K=I\), \(f=0\), \(p=1+x+2y\) on the unit square. It compares the entire
pressure, flux and skeleton coefficient arrays against serial execution and
checks pressure/flux errors, macroscopic conservation, and the global residual.
These checks occur outside the timed interval. The complete solve, including
pool startup and data movement, is timed. A serial warmup precedes each workload;
at least three timed repetitions are required.

The report records individual durations, medians, spread, hardware, software,
BLAS thread limits, and SHA-256 hashes of the source files and lockfile. A flag
identifies source changes during measurement. Hashes describe the acquisition
state, which need not match a later checkout or optional-environment update.
These are workstation measurements
without CPU affinity pinning or exclusive access to the machine.

A Linux measurement on 28 September 2026 used an Intel Xeon E5-2698 v4,
Python 3.13.15, NumPy 2.5.3, SciPy 1.18.1 and OpenBLAS 0.3.34. Native libraries
were restricted to one thread, with two Python workers for parallel cases. Both
workloads used eight subdivisions per macrotriangle and 45 local P1 pressure
unknowns. Medians of three repetitions were:

| Macrotriangles | Fine triangles | Serial (s) | Threads (s) | Processes (s) | Thread speedup | Process speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 2,048 | 0.462 | 0.526 | 0.928 | 0.878 | 0.497 |
| 128 | 8,192 | 1.839 | 2.266 | 2.353 | 0.812 | 0.782 |

![Complete CPU solve times and speedup for 32 and 128 macrotriangles](figures/performance/linux-cpu.png)

[Download the CPU figure as SVG](figures/performance/linux-cpu.svg).
Bars show medians and whiskers show the observed minimum–maximum range of three
repetitions. Speedup uses the corresponding serial median as the fixed
numerator; the dashed line marks one. These ranges are not confidence intervals.

The measured parallel executions were slower for these workloads. Serial
assembly, small local factorizations, dispatch, and process startup limit the
benefit available from parallel condensation. This result does not establish
behavior for larger local spaces, expensive constitutive models, or more workers.
The [complete CPU report](https://github.com/volpatto/pymhm/blob/main/benchmarks/results/linux-cpu.json)
contains the actual numerical errors and all individual timings.

## Larger local spaces and complete worker assembly

The larger-local benchmark separates three workloads: the complete
`solve_darcy`, condensation of already assembled local matrices, and the complete
factory solve. All use the same native P1 variational kernels, constant macroface
traces, coefficients and affine manufactured solution. The factory returns the
fine mesh as metadata and reconstructs the same physical pressure and flux. It
does not change the discretization to obtain a faster result.

```bash
pixi run -e test python benchmarks/large_local.py --output benchmarks/results/large-local.json
pixi run -e notebooks python examples/plot_large_local.py
```

Each workload measures serial execution and one, two, four and eight threads or
spawned processes, with three repetitions and one native numerical thread per
worker. The `solve_darcy`, factory and prepared-condensation paths alternate within
each repetition. Complete times include worker-pool creation, serialization,
local and global work, and final reconstruction. Prepared-condensation times
exclude local assembly but include pool creation and input/output serialization.
The manufactured-field and serial-reference comparisons occur outside timers.

The archived high-level runs use `parallel_assembly=False`.
These runs used the same Xeon workstation and software versions as the CPU
measurement above. The machine was not exclusive: unrelated applications
remained active, and CPU affinity was not pinned. Runtime sources, benchmark
sources and the lockfile had identical SHA-256 hashes before and after the run.

The three cases contain 32,768, 73,728 and 131,072 fine triangles, respectively.
Here \(r\) is the number of subdivisions along each macrotriangle edge. The
table reports medians in seconds; the factory process column also includes the
minimum–maximum of its three repetitions.

| Macros | \(r\) | Local pressure DOFs | `solve_darcy` serial | Factory serial | Factory, 8 processes | Gain over factory serial | Gain over `solve_darcy` serial |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 64 | 2,145 | 2.790 | 2.735 | 1.044 (1.021–1.048) | 2.620× | 2.672× |
| 8 | 96 | 4,753 | 6.066 | 6.121 | 1.679 (1.647–1.695) | 3.645× | 3.612× |
| 32 | 64 | 2,145 | 10.936 | 10.867 | 2.406 (2.344–2.496) | 4.516× | 4.545× |

The `solve_darcy` path spent approximately 85%, 80% and 86% of its serial complete
time in local assembly. Its best complete speedups among the measured worker
configurations were only 1.028×, 1.058× and 1.077×, even though prepared local
condensation achieved 1.499×, 1.956× and 1.975×. The factory path distributes both
assembly and condensation among processes. Eight processes gave the lowest
complete factory median in each case. The gains are
sublinear, include startup and communication costs, and establish behavior only
for these problem sizes, operators and implementation.

All complete runs passed the physical-error, conservation and serial-field
comparisons. Across both solution paths and every worker configuration, the
largest pressure and flux \(L^2\) errors were \(8.32\times10^{-14}\) and
\(1.08\times10^{-12}\); the largest macrocell conservation defect and global
relative residual were \(5.55\times10^{-16}\) and \(9.18\times10^{-17}\).
This affine problem checks algebraic equivalence and execution cost; it does
not measure convergence for heterogeneous or nonlinear models.

![Complete solve_darcy time, including its serial local assembly](figures/large-local/complete-solve.png)

[Download the complete-solve figure as SVG](figures/large-local/complete-solve.svg).

![Isolated condensation of prepared local matrices, including pool creation](figures/large-local/prepared-condensation.png)

[Download the prepared-condensation figure as SVG](figures/large-local/prepared-condensation.svg).

![Complete factory solve with local assembly and condensation in workers](figures/large-local/factory-complete-solve.png)

[Download the factory figure as SVG](figures/large-local/factory-complete-solve.svg).
The black horizontal line is the factory serial median; the red line is the
`solve_darcy` serial median. Whiskers show the observed minimum–maximum
range. Three repetitions in a fixed configuration order do not establish
confidence intervals or machine-independent scaling.

![Serial solve_darcy phase costs and the distinction between complete and prepared speedup](figures/large-local/phase-profile.png)

[Download the phase-profile figure as SVG](figures/large-local/phase-profile.svg).
The stacked bars combine independently measured phase medians and omit the
small parent-bookkeeping term. They are not an exact additive decomposition of
the median complete solve. Every individual phase time, bookkeeping time,
verification error and source hash is preserved in the
[complete larger-local report](https://github.com/volpatto/pymhm/blob/main/benchmarks/results/large-local.json).

## GPU and solver boundaries

`solver` selects the global linear solver; `local_solver` selects local
factorizations. The cuDSS adapter can therefore be used for local condensation,
the global system, or both. The public numerical arrays remain NumPy/SciPy arrays.
GPU adapters transfer matrices and right-hand sides to the device and bring
solutions back to the host. For these sparse adapters, local assembly, stored lifts, global assembly and
reconstruction remain on the CPU. The separate
[resident batch API](execution.md#resident-batches-on-one-gpu) assembles affine
P1 volume operators on the device and retains batched LU factors across RHS
queries; it does not make the full PDE pipeline resident.

CuPy sparse QR is a global solve option. cuDSS provides reusable device-side
factorization through `nvmath-python`. Repeated tiny local systems can cost more
to dispatch and transfer than to solve on the CPU. Benchmark transfer and setup
costs, factorization reuse, warmup, and synchronization explicitly before making
an acceleration claim. Simultaneous GPU work from independent Python worker
processes is not a validated scheduling strategy here.

`solve_darcy(..., local_solver="pyamg")` uses CPU algebraic multigrid for the
local primal elliptic problems. The local kernel requires care: source and trace
loads are projected onto the compatible subspace, independent kernel
coordinates are pinned during the positive-definite solve, and the prescribed
physical mean is restored afterward. This differs from applying an SPD solver
directly to the indefinite constrained local matrix.

The explicit general `coarse_basis` condensation path is currently unsupported
by the PyAMG and AmgX adapters and raises an error. It requires retaining the
nonzero action of the operator on the coarse modes, rather than applying the
true-nullspace projection. This restriction matters for the constant modes
retained in reaction–diffusion and heat, and the translations retained in
Brinkman away from zero drag. Use a supported direct local solver for those
retained-basis problems; no backend fallback occurs. The primal Darcy benchmark
uses a true constant kernel and retains AMG support.

With PyAMG local solves, the same CPU workloads gave the following medians.
The coefficient comparison uses relative tolerance \(10^{-9}\) and absolute
tolerance \(10^{-10}\), accommodating the iterative solver's \(10^{-10}\)
relative residual target. The independent manufactured pressure/flux errors,
conservation error, and global residual must each remain below \(10^{-9}\).

| Macrotriangles | Serial (s) | Threads (s) | Processes (s) |
| ---: | ---: | ---: | ---: |
| 32 | 0.666 | 0.943 | 1.039 |
| 128 | 2.550 | 4.132 | 2.732 |

![CPU AMG solve times and speedup, shown separately from direct solvers](figures/performance/linux-cpu-amg.png)

[Download the CPU AMG figure as SVG](figures/performance/linux-cpu-amg.svg).
Here the speedup baseline is serial execution with the same AMG local solver.

For these 45-unknown local problems, AMG setup and iteration cost more than the
direct factorization. This comparison does not establish behavior at larger
local resolutions. The [complete CPU AMG report](https://github.com/volpatto/pymhm/blob/main/benchmarks/results/linux-cpu-amg.json)
records every repetition and verification error.

The hybrid global system contains coarse constraints and is generally
indefinite; use a solver appropriate to that algebraic structure. cuDSS uses
maximum diagonal-product matching and iterative refinement to handle the zero
coarse diagonal block, and every solve still passes the ordinary residual check.
Backend availability, numerical correctness, and favorable timing are three
separate checks.

`HybridSystem.solve` without a prepared factorization first performs an
equilibrated CPU sparse LU rank diagnostic for non-SciPy solvers. This rejects
unsupported trace enrichment even when a compatible right-hand side admits a
small residual. The diagnostic factorization is discarded before the requested
backend runs, so the global GPU timings below include this CPU factorization.
SciPy reuses its factorization for both diagnosis and solution. Prepared
`OfflineHybridSystem` solves reuse their factors; distributed MPI assembly
and solution follow the separate path described in [Execution](execution.md).

```bash
pixi run -e test benchmark --local-solver pyamg --output benchmark-results/cpu-amg.json
pixi run -e gpu benchmark-gpu --output benchmark-results/gpu.json
```

The GPU benchmark compares a CPU baseline, a GPU global solve (CuPy or cuDSS),
GPU local solves (cuDSS), and cuDSS in both positions. Each configuration has an
untimed warmup; all device timings synchronize explicitly. `--amgx` additionally
requires a working native AmgX/PyAMGX installation and measures AMG local solves.
It fails if that requested backend is unavailable.

On an NVIDIA GeForce RTX 3060 (12 GB, compute capability 8.6), the complete solves
gave the following medians in seconds. The GPU environment used CuPy 14.2.0,
CUDA runtime 12.9, nvmath-python 1.0.0 and cuDSS 0.8.0.10. AmgX 2.5.0 was
compiled with CUDA 12.6 and GCC 12.4. Each cell reports three repetitions after
an untimed warmup for that solver placement.

| Macrotriangles | CPU | Global CuPy | Global cuDSS | Local cuDSS | Local + global cuDSS | Local AmgX |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.485 | 0.494 | 0.512 | 0.850 | 0.858 | 2.437 |
| 128 | 1.919 | 1.933 | 1.933 | 3.385 | 3.393 | 9.730 |

![GPU solver placements with elapsed times, observed ranges and CPU baseline speedup](figures/performance/linux-gpu.png)

[Download the GPU figure as SVG](figures/performance/linux-gpu.svg).
Each workload uses its own CPU median as the speedup baseline. Both workload
panels share zero-origin linear axes, preserving absolute time comparisons.

All GPU configurations passed the analytical and coefficient checks; the
largest pressure/flux error was \(1.08\times10^{-13}\). These measurements do
not show GPU acceleration for the selected workloads. Per-local initialization,
hierarchy construction, transfer and synchronization are included; larger
local problems or a device-resident assembly and batching strategy require
separate measurements. These are timings of the recorded implementation, not
comparisons of peak solver performance. The [complete GPU report](https://github.com/volpatto/pymhm/blob/main/benchmarks/results/linux-gpu.json)
records native revisions, build information, individual timings and source hashes.

## Optional native AmgX installation

AmgX is a separately compiled NVIDIA library. Its Python binding is not included
in the root GPU lockfile. A pinned Linux build recipe is provided for Python
3.13, CUDA 12.6 and GCC 12; the installed library was validated alongside the
CUDA 12.9 dependencies in the GPU environment. An NVIDIA driver and sufficient
memory for C++ and CUDA compilation are required.

```bash
pixi install -e gpu --locked
pixi run --manifest-path tools/amgx/pixi.toml install-amgx --cuda-arch 86 --jobs 4
pixi run -e gpu pytest -m gpu
pixi run -e gpu benchmark-gpu --amgx --output benchmark-results/gpu-amgx.json
```

Architecture `86` selects the RTX 3060 used for validation. Omit `--cuda-arch`
to ask CMake to detect the visible GPU, or select the architecture of the target
device. The recipe fetches AmgX revision
`91a8413ef267b1c32aff4014c02820e1c5897ac2` and PyAMGX revision
`6229ff008ee5a264cfc1799eeb2f83d96da0aadc` from their public upstream repositories.
The recipe builds the shared library and binding with optional NVTX profiling
disabled, then verifies a numerical solve.

The native library is installed into the GPU environment's library directory;
the binding uses relative runtime library paths and requires no `PYTHONPATH`
override. The resulting wheel is an installation artifact for that environment,
not a portable wheel for distribution. Recreating the GPU environment requires
running the native installation again. The build recipe currently targets Linux
only; Windows AmgX compilation is not validated.
