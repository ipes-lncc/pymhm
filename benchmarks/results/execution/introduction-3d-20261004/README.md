# Three-dimensional anisotropic Darcy: processes and AMG

These records accompany the
[step-by-step notebook](../../../../notebooks/introduction/darcy_3d_parallel_scalability.ipynb).
It displays the UFL material/source, local provider, face orientation, global
MHM equation, independent conforming Q1 method and complete timers. Its default
small reproduction is separate from the resolved campaign in `results.json`.

## Physical and numerical inputs

The pressure is the product of `sin(pi*x_i)` on `(0,L) × (0,1)²`, for integer
length `L`, with homogeneous exterior pressure. The material is
`exp(prod(sin(2*pi*x_i/0.1)))` times the full symmetric anisotropy tensor shown
in the notebook. The source is the physical `-div(K*grad(p))`, including mixed
pressure derivatives. Physical coordinates and material period remain fixed
as the weak domain expands. With macro width 0.25 and period 0.1, the
scalar multiplier equals one on each macroface, so interface permeability is
the constant tensor D. This alignment and the interior heterogeneity are
part of the benchmark's physical/discretization data.

The unit cube has 64 Cartesian macrohexahedra. Every local volume space is
continuous Q1; each unsplit face has four tensor-product Q1 trace modes.
There are 240 faces, 960 trace coordinates and 64 retained constants, for
1,024 coupled unknowns. Native DOLFINx/UFL assembles the declared volume and
face forms, and Basix supplies tensor tabulation. Physical volume moments
select zero-mean local responses. Full Dirichlet data fix physical pressure;
no global zero-mean gauge is imposed.

Fine counts 64/96/128 correspond to 262,144/884,736/2,097,152 fine hexahedra.
The 128-per-axis setting is optional; the tables identify actual recorded
configurations. The conforming reference uses the same fine cells and physical
data. Its global
space differs from MHM's, so matching fine-element counts does not establish
matching approximation accuracy. Fixed macro/trace spaces can leave an
interface-error floor as local volume meshes are refined.

## Timing conventions

CPU samples include fresh geometry, UFL assembly, fresh local factors or AMG
hierarchies/solves,
all source and trace right-hand sides, complete response transfer, ordered
global accumulation, pool startup/join, global solve and reconstruction.
Warmups populate the existing compiler cache; numerical operators and
hierarchies are rebuilt for every measured trial. Fresh worker imports remain
inside each process time. Integration, plotting and archive writing are
untimed scientific validation. The record preserves every actual sample;
a single timing has no repeated-sample variability estimate.

Strong process speedups use the one-spawned-process time for the same fixed
configuration. True serial MHM is a distinct configuration. Weak cases expand the
x-domain to `L=processes`, with fixed macro width, fine-cell size, trace degree
and coefficient period. Their larger global problems remain timed.

Classical CPU1 PyAMG and multi-process MHM have different CPU budgets. Classical
PETSc CG/GAMG uses 32 physical cores; MHM uses its recorded affinity of 24 or
28 physical cores. Optional GPU procedures are scoped local-condensation components;
their initialization, transfers, synchronization and setup costs cannot be
omitted when interpreting a complete-workflow speedup.

The 96-per-axis MHM configuration with one process and sparse direct local
solves is budget-censored at an invocation lower bound of 900 s. It has no
completed workflow time or accepted field state. It is excluded from medians
and speedup denominators; it is not assigned an estimated completion time.

The optional native component uses
[NVIDIA AmgX 2.5.0](https://github.com/NVIDIA/AMGX/tree/91a8413ef267b1c32aff4014c02820e1c5897ac2)
at revision `91a8413ef267b1c32aff4014c02820e1c5897ac2` and
[PyAMGX](https://github.com/shwina/pyamgx/tree/6229ff008ee5a264cfc1799eeb2f83d96da0aadc)
at revision `6229ff008ee5a264cfc1799eeb2f83d96da0aadc`. Its recorded build targets
CUDA architecture 8.6 with MPI disabled; each GPU has its own spawned process.
CPU components use the same archived operators and component environment.
The complete native-FEM CPU campaign and accelerator component record their
own software versions and timing scopes separately.

Additional physical-residual correction passes can rebuild an AMG hierarchy.
Those passes, moment restoration, setup and transfers remain included in the
component timer; there is no guarantee of one setup for an entire condensation.

## Recorded measurements

The following tables use only actual entries in `results.json`. Every sample
remains in that record; a one-sample row is a measured observation rather than
a robust repeated estimate. Resource counts and solver names are explicit.

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

### Complete fixed-discretization workflows

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

### Complete weak workflows

No complete workflows are recorded for this study.

Weak efficiency compares one process with the same fixed local workload on
its expanding domain. Global growth and process transfer remain included.

### Physical accuracy at the measured local resolutions

| Fine cells per unit axis | Discretization / solver | Pressure L2 error | Physical vector-flux L2 error |
| ---: | --- | ---: | ---: |
| 64 | Classical PETSc CG/GAMG | 8.8900421e-05 | 4.8283419e-02 |
| 64 | Classical pyamg | 8.8900421e-05 | 4.8283419e-02 |
| 64 | MHM pyamg; macro4 / traceQ1 | 1.0714369e-03 | 7.5135753e-02 |
| 96 | Classical PETSc CG/GAMG | 3.9629110e-05 | 3.2451356e-02 |
| 96 | Classical pyamg | 3.9629110e-05 | 3.2451356e-02 |
| 96 | MHM pyamg; macro4 / traceQ1 | 1.0801716e-03 | 6.6169010e-02 |

These rows form a local-resolution performance sweep with fixed MHM macro and
trace spaces. The conforming reference refines its full space. Their pressure
and vector-flux errors must be assessed alongside any cost advantage; equal
fine-element counts alone do not establish equal-accuracy performance.

Source/trace response residuals, reconstructed physical equation residuals and
physical-volume moment defects are recorded separately. The campaign's
physical-field acceptance threshold is `1e-8`; all five recorded MHM states
also meet the stricter `1e-10` diagnostic. The largest reconstructed physical
relative residual is `2.13e-13`. The relative quotient for the retained constant
uses a roundoff-scale kernel-action right-hand side and is recorded separately
from physical-field acceptance. AMG correction and projected-response targets
remain `1e-10` and `1e-12`, respectively.

The conforming reference has observed two-mesh orders 1.993 for pressure and
0.980 for physical vector flux between 64 and 96. MHM's pressure errors remain
near `1.08e-3` while its flux error decreases; this is the fixed macro/trace
resolution floor, rather than full MHM convergence. Assembly/error quadrature
controls compare degrees 6, 8, 10 and 12 on patches at the actual fine-cell
sizes. These integration controls are distinct from complete performance runs.

### Local-condensation components

No local-condensation component timings are recorded.

These component timers start from original assembled operator archives. They
include fresh spawn, imports, native lifecycle, archive reads, solves, all
transfers, synchronization, complete response return and shutdown, while FEM
assembly and the global problem are excluded. Component medians are not added
to independently measured workflow medians to construct a total speedup.

## Physical fields, portability and provenance

Pressure and all components of physical Darcy flux are compared with exact
fields outside the timers. The raw physical flux is `-K*grad(p)`; its broken
reconstruction does not establish H(div) continuity or fine-cell conservation.
Macro conservation concerns the oriented skeleton moments. Norm quadrature,
reference refinement and original-row checks are recorded independently.
Executed retained bases, coordinate maps and coefficient archives preserve the
physical replay contract. Native node order is mapped through integer lattice
addresses, rather than floating coordinate sorting.

The benchmark host has two Xeon Silver 4216 sockets (32 physical, 64 logical
cores) and two RTX A5000 GPUs. Actual affinities, thread limits, software
versions, compiler initialization, source hashes and resource snapshots belong
to `results.json` and associated receipts. Node-local process measurements do
not establish inter-node scalability. These hexahedral manufactured data and
spaces differ from the tetrahedral experiments of
[Gomes et al. (2017)](https://arxiv.org/abs/1703.10435); this is an analytical
application, not a matched literature timing reproduction.
