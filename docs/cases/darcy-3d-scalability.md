# Three-dimensional Darcy scalability

The [introductory notebook](https://github.com/volpatto/pymhm/blob/main/notebooks/introduction/darcy_3d_parallel_scalability.ipynb)
defines the material, exact fields, local UFL equations, oriented face pairings,
global problem, independent conforming method and complete timers in executable
cells. It uses `LocalEquations`, `Equation` and `MultiscaleProblem`; the physical
problem does not require a dedicated Darcy constructor. A small default
reproduction accompanies the larger recorded campaign.

## Physical problem

On the box with integer length,

$$
\Omega_L=(0,L)\times(0,1)^2,
$$

the pressure and physical Darcy flux satisfy

$$
\begin{aligned}
q&=-K\nabla p, & \nabla\cdot q&=f,\\
p&=0 &&\text{on }\partial\Omega_L.
\end{aligned}
$$

The manufactured data are

$$
\begin{aligned}
p_*(x)&=\prod_{i=1}^3\sin(\pi x_i),\\
K(x)&=m(x)D, &
m(x)&=\exp\!\left(\prod_{i=1}^3\sin(2\pi x_i/0.1)\right),\\
D&=\begin{pmatrix}2&0.3&0.2\\0.3&1.5&0.1\\0.2&0.1&1\end{pmatrix}, &
f&=\nabla\cdot(-K\nabla p_*).
\end{aligned}
$$

The selected macro width is 0.25 and the material period is 0.1. Consequently
the scalar multiplier equals one on every macroface: its coordinate factors
there are sine values at integer multiples of five pi. The permeability on
these interfaces is the constant tensor D. Interior volume heterogeneity and
this interface alignment are both part of the benchmark data; the alignment
affects the trace resolution required by this analytical application.

The full symmetric positive-definite tensor includes mixed derivatives in the
source. UFL differentiation and independent analytical pressure/material
formulas specify the same physical operator. Integer lengths preserve the
homogeneous exterior data while keeping the material wavelength fixed.

## Declared discretization

The unit-cube macro mesh has four cells per direction, hence 64 independent
local problems. Each macrocell contains a structured conforming Q1 hexahedral
mesh. A fine count of 64, 96 or 128 per unit direction corresponds to 16, 24 or
32 local cells per direction and 262,144, 884,736 or 2,097,152 total fine cells.
The 128-per-axis setting is optional; the results tables identify the actual
recorded configurations. The independently assembled conforming Q1 reference
uses the same fine grid.

Each unsplit macroface carries a tensor-product Q1 trace with four modes. The
unit-cube skeleton has 240 faces and 960 trace coordinates. One retained
constant per macrocell gives a coupled system of 1,024 coordinates. These
counts do not imply that its global approximation space equals the conforming
reference space. The notebook also exposes the trace degree explicitly;
changing it defines a different discretization.

The local UFL operator and coupling are

$$
\begin{aligned}
a_T(p,v)&=\int_T K\nabla p\cdot\nabla v,\\
b_T(v,\lambda)&=\sum_{F\subset\partial T}s_{T,F}\int_F v\lambda_F,\\
a_T(p_T,v)+b_T(v,\lambda)&=\int_T f v\,dx.
\end{aligned}
$$

The sign maps the canonical face flux to the cell's outward flux. Increasing
physical tangent coordinates give the same Q1 face basis to both incident
cells in this Cartesian application. Basix provides the tensor trace and field
tabulation; DOLFINx/UFL assembles the volume and face forms.

The constant local kernel is declared explicitly. Physical volume moments
select zero-mean local source/trace responses, and retained amplitudes restore
the physical cell means. There is no global zero-mean gauge under these full
Dirichlet data: the exact unit-cube pressure mean is nonzero. Persisted
coefficients include their executed retained basis, trace maps and geometry.

CPU AMG uses the generic Neumann projection and coordinate pinning owner.
Pinning selects an SPD elliptic solve, then the owner restores the declared
physical moment and checks the original rows. A hierarchy built for a solver
call serves its multiple right-hand sides; original-row refinement remains
inside the package owner and timer. Direct local solves instead factor the
constrained matrix once for their right-hand sides. The coupled global problem uses
its generic sparse direct solver. No elliptic AMG assumption is imposed on
the global hybrid matrix.

## Complete timings and accuracy

Every measured workflow constructs fresh geometry, local operators, solver
hierarchies and responses. Complete MHM time includes process startup, child
imports, local assembly and solves, serialization, ordered shared-face
accumulation, synchronization, pool shutdown, the global solve and complete
field reconstruction. Norm integration, plotting and archive writing occur
after that timer.

Warmup populates the existing compiler cache. This separates common first-use
compilation from the warmed measurements without reusing numerical factors
between independent trials. Fresh worker startup remains inside process
timings. A cleared compiler cache is a different experiment.

Strong scaling fixes the complete discretization and compares one process
with several spawned processes. True serial execution is a distinct configuration. Weak scaling
extends the x-domain with its length equal to the process count, preserving
macro width, local cell size, trace degree and coefficient period. Both local
work and the growing global problem remain in the reported total.

The [public record](https://github.com/volpatto/pymhm/tree/main/benchmarks/results/execution/introduction-3d-20261004)
contains the actual timing samples, resource counts, physical errors,
quadrature controls and provenance. CPU1 classical AMG and multi-process MHM
use different CPU budgets. The classical MPI reference uses 32 physical cores,
while MHM has 24 or 28 available physical cores. GPU condensation components
have a separate scope; no accepted large GPU timing is available here.

The local-resolution performance sweep integrates pressure and physical
vector-flux errors separately against the analytical fields. Refining only local volume meshes with fixed macro/trace
spaces can leave an interface-error floor. Equal fine-cell counts therefore
measure a geometry budget, rather than equal accuracy. The conforming reference
has its own refinement controls. The raw local field `-K grad(p)` is broken;
macro conservation does not imply H(div) continuity or fine-cell conservation.
Field plots overlay the actual macro mesh and preserve independent local
values.

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

## Recorded complete times and field errors

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

| Fine cells per unit axis | Discretization / solver | Pressure L2 error | Physical vector-flux L2 error |
| ---: | --- | ---: | ---: |
| 64 | Classical PETSc CG/GAMG | 8.8900421e-05 | 4.8283419e-02 |
| 64 | Classical pyamg | 8.8900421e-05 | 4.8283419e-02 |
| 64 | MHM pyamg; macro4 / traceQ1 | 1.0714369e-03 | 7.5135753e-02 |
| 96 | Classical PETSc CG/GAMG | 3.9629110e-05 | 3.2451356e-02 |
| 96 | Classical pyamg | 3.9629110e-05 | 3.2451356e-02 |
| 96 | MHM pyamg; macro4 / traceQ1 | 1.0801716e-03 | 6.6169010e-02 |

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

Resource counts and actual errors define the scope of these comparisons.
The [full record](https://github.com/volpatto/pymhm/tree/main/benchmarks/results/execution/introduction-3d-20261004)
includes each actual sample, precision controls and unmeasured scopes.

## Relation to literature and HPC execution

[Gomes et al. (2017)](https://arxiv.org/abs/1703.10435) describe independent local
response construction and a separately assembled coupled problem, with MPI
local ownership and distributed sparse algebra. Their 3D performance problem
uses a different coefficient, tetrahedral P2 spaces, face partitions and
cluster resources. The present hexahedral analytical application is not a
matched reproduction of that timing table.

[Penna et al.](https://doi.org/10.1002/cpe.5170) study cost-aware scheduling for
heterogeneous MHM workloads. Its scheduling gains do not transfer directly to
uniform one-shot local meshes. A bounded ordered process pipeline provides
node-local execution here; multi-node efficiency requires a separately
measured distributed campaign. See the [execution guide](../execution.md) and
[performance report](../performance.md) for resource and timing conventions.
