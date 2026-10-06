# Periodic Darcy coefficient and face enrichment

The oscillatory experiment in §4 of
[Paredes, Valentin and Versieux (2017)](https://doi.org/10.1090/mcom/3108)
uses local bilinear fields and constant normal-flux modes on macrofaces.
Current controls verify the complete discretization, original equations and
persisted-field replay through local refinement $r=256$. The conforming
$Q_1/4096$ reference space has been executed. Local and reference refinement
increments remain material; accuracy and Figure 6 reproduction are not established.

## Problem and spaces

On $\Omega=(0,1)^2$,

$$
\begin{aligned}
-\nabla\cdot(K_\varepsilon\nabla p)&=\sin(x)\sin(y),
\qquad p|_{\partial\Omega}=0,\\
K_\varepsilon&=1+100\cos^2(\pi x/\varepsilon)\sin^2(\pi y/\varepsilon),\\
\varepsilon&=\pi/150.
\end{aligned}
$$

The physical Darcy flux is $q=-K_\varepsilon\nabla p$ and
$1\leq K_\varepsilon\leq101$. The coefficient is periodic; the boundary
condition is homogeneous Dirichlet. The source arguments have no extra factor of $\pi$.

The fixed macrogrid contains $8\times8$ squares. Each local space is continuous
$Q_1$ on an $r\times r$ Cartesian submesh. Each macroface has
$s=1,2,4,8,16,32$ independent piecewise constant modes. The retained constant
has a declared unit physical mean. Dirichlet pressure enters the hybrid weak
form, and local assembly uses four Gauss points per coordinate.

![Permeability and source on the actual macrogrid](../figures/periodic/problem.png)

Finite local spaces must detect their skeletal space. On this aligned
weak-Dirichlet grid, one P0 segment per Q1 fine edge admits a nonzero
alternating trace with zero local coupling. In particular, $r=32,s=32$ is
inadmissible and its singular global system is rejected. The $r=32$ storage
pilot uses at most 16 segments; the $r=128$ and $r=256$ controls include all six
partitions. Prescribed exterior Neumann data change the free trace space and
require a separate compatibility check. The exact-local-map analysis does
not certify every finite local/trace pair.

## Original-equation acceptance and replay

Acquisition stores the executed kernel, mass constraints, retained basis,
trace orientation, operator/load digests and source/lock identity.
Original-equation corrections reuse the same local and condensed operators,
physical source and boundary data. The explicit extended mode preserves the
measured defect through the shared solver. The criterion remains $10^{-10}$
relative to the concatenated physical source and weak rows. A condensed
residual has its own normalization and does not replace this criterion.

All six partitions at $r=128$ and $r=256$ pass the original criterion.
The largest final relative residual is $2.800\times10^{-14}$ at $r=128$ and
$1.166\times10^{-13}$ at $r=256$. At $r=256$, the weak-row norm is at most
$7.022\times10^{-22}$, the energy–source-work defect is at most
$4.207\times10^{-20}$, and the macro skeletal outflow–source imbalance is at
most $1.903\times10^{-14}$. The last check concerns macro conservation;
it does not establish fine-cell conservation of the raw constitutive flux.

Replay uses the executed initial fields and ordered physical increments,
their trace/coarse increments and basis digests. At both refinements, fresh
local assembly with equivalent sign rotations of the one-dimensional
constant kernel and two BLAS threads reproduces the numeric field, trace,
coarse and kernel digests exactly. The archived kernel, constraints and
executed retained basis are restored. This control covers equivalent sign
rotations of this one-dimensional kernel.

| Acquisition | Native threads | Full process time | Peak resident memory |
|---|---:|---:|---:|
| $r=128$, all six partitions | 1 | 25 min 34 s | 643 MiB |
| $r=256$, all six partitions | 1 | 1 h 50 min 12 s | 2.52 GiB |
| $r=256$ executed-history replay | 2 | 22 min 28 s | 1.93 GiB |

These measurements include setup, factors, reconstruction, original checks
and persistence. Shared-host contention is uncontrolled, and the workloads
differ; no speedup is claimed. The compact record
`examples/results/periodic/streamed-original-verification.json` identifies
the acquisitions and executed basis/operator digests. Each acquisition
retains its executed immutable source generation.

## Independent complete-system comparison

A separate DOLFINx 0.9.0/UFL application assembles the complete uncondensed
Q1/segmented-P0 saddle system on all 64 macros with the same geometry,
coefficient, source and weak exterior pressure convention. Volume/load
matrices and geometric face coupling are assembled independently. SciPy
SuperLU solves the original full system, with wider accumulation of original
defects and corrections under the unchanged physical-source criterion.

| Control | Total saddle unknowns | Final original relative residual | Relative pressure difference | Relative gradient difference | Relative Darcy-flux difference |
|---|---:|---:|---:|---:|---:|
| $r=128,s=32$ | 1,069,632 | $2.705\times10^{-14}$ | $3.883\times10^{-13}$ | $1.369\times10^{-12}$ | $1.429\times10^{-12}$ |
| $r=256,s=32$ | 4,231,744 | $1.130\times10^{-13}$ | $2.962\times10^{-11}$ | $2.803\times10^{-11}$ | $2.804\times10^{-11}$ |

The r256 full matrix has 37,994,560 nonzero entries. All 64 oriented couplings
match byte for byte. Maximum relative volume-matrix and load differences are
$1.966\times10^{-14}$ and $5.427\times10^{-15}$. Independent integration with
eight and ten Gauss points per coordinate gives the stated physical differences.
The full control took 8 min 16 s with one native thread and a 13.59 GiB peak.

Executed Basix cardinal matrices and nodal permutations identify the native
field basis. This is an instrumented independent verifier, separate from the
article authors' original application. The verified
[DOLFINx release revision](https://github.com/FEniCS/dolfinx/commit/6443e3b27d29aec04ce83d8424b57c339e35f865)
and [Basix release revision](https://github.com/FEniCS/basix/commit/19555f5b629b4090b14014f9db5f2c9ac80984f9)
are distinguished from the installed DOLFINx build's reported revision,
which is not verified as an accessible upstream commit. Recorded binary
digests identify the actual executed builds.

Discrete agreement and small residuals do not certify an inf-sup bound or
approximation accuracy.

## Conforming reference and its own resolution

The material is the exact separated sum $K(x,y)=1+a(x)b(y)$.
Weighted one-dimensional matrices give the same tensor-Gauss bilinear form
as ordinary cell assembly:

$$
A=M_y\otimes S_x+S_y\otimes M_x
  +M_y^b\otimes S_x^a+S_y^b\otimes M_x^a.
$$

The matrix-free solver applies these original terms. Conjugate gradients
uses a symmetric AMG V-cycle of the refined-grid Q1 operator as a
preconditioner; the approximation remains the requested conforming Qk space.
Positive sampled permeability and full strong Dirichlet elimination give
positive-definite Galerkin operators. No degree- or contrast-independent
convergence bound is asserted. The original residual must satisfy
$\lVert b-Au\rVert_2\leq10^{-10}\lVert b\rVert_2$.

Current conforming Q1 controls use tensor Gauss order 4, one worker and one
native thread, explicit extended accumulation and before/after source and lock guards.

| Cells per coordinate | Nodal coefficients | Original relative residual | Full process time | Peak resident memory |
|---:|---:|---:|---:|---:|
| 512 | 263,169 | $6.254\times10^{-11}$ | 4.93 s | 288 MiB |
| 1024 | 1,050,625 | $3.220\times10^{-14}$ | 21.63 s | 853 MiB |
| 2048 | 4,198,401 | $1.341\times10^{-13}$ | 1 min 30 s | 2.99 GiB |
| 4096 | 16,785,409 | $5.408\times10^{-13}$ | 7 min 9 s | 11.58 GiB |

Own refinement increments use the finer numerical field as denominator:

| Q1 refinement | Relative pressure $L^2$ increment | Relative full $H^1$ increment |
|---|---:|---:|
| $256^2\to512^2$ | 26.12299% | 53.17389% |
| $512^2\to1024^2$ | 11.73957% | 34.48692% |
| $1024^2\to2048^2$ | 3.70154% | 19.12405% |
| $2048^2\to4096^2$ | 0.98654% | 9.76864% |

An increment is not an upper bound on the finer field's unknown continuum
error. The Q1/4096 space stated for the article's reference is executed here
with the $\varepsilon=\pi/150$ coefficient; its own resolution is not demonstrated.
The original Figure 6 reference field remains unavailable.

At Q1/512, increasing assembly quadrature from 4 to 6 changes full H1 by
$9.231\times10^{-10}$ relatively; this comparison also includes the solvers'
algebraic errors. An independently assembled full conforming DOLFINx/UFL
control agrees with PyMHM in pressure, gradient and physical Darcy flux within
$2.031\times10^{-12}$. Its native original residual is $2.569\times10^{-11}$;
matrix and load audits differ relatively by at most $1.970\times10^{-14}$ and
$1.841\times10^{-14}$. The record
`examples/results/periodic/q1-reference-verification.json` identifies the
actual fields, sources, precision conventions and checks.

A sufficiently resolved higher-order classical denominator remains a separate
acquisition and refinement requirement. No current Q5/1024 denominator or MHM
r512 face curve is certified by these controls.

## Norms and literature comparison

The reported norm is the full broken H1 norm:

$$
\frac{
\left(\sum_T
\left[
\lVert p_{\rm ref}-p_h\rVert_{L^2(T)}^2+
\lVert\nabla p_{\rm ref}-\nabla p_h\rVert_{L^2(T)}^2
\right]\right)^{1/2}
}{
\left(
\lVert p_{\rm ref}\rVert_{L^2(\Omega)}^2+
\lVert\nabla p_{\rm ref}\rVert_{L^2(\Omega)}^2
\right)^{1/2}
}.
$$

Polynomial products are integrated exactly on the common nested Cartesian
partition, with positive direct quadrature near cancellation. Independent
values on opposite sides of a macrointerface are retained. Full H1,
pressure L2 and the gradient seminorm have separate data fields.

For fixed $s=1,2,4,8,16,32$, the local $r=128\to256$ increment is
18.952–19.134% in full broken H1, using the finer MHM field as denominator.
Local error remains material. This is separate from MHM/reference error and
classical-reference refinement.

Figure 6 has four printed, five-decimal annotations:
$s=1:0.17922$, $s=4:0.12410$, $s=16:0.05614$ and $s=32:0.03969$.
The s2 and s8 ordinates retain vector-digitization uncertainty.
The manuscript has 31 PDF pages; Figure 6 is on page 28.
`examples/results/published/paredes2017_figure6.json` records the PDF identity,
printed annotations and extraction convention without replacing the digitized
ordinates. The digitization allowance is not an uncertainty estimate for the published solver.

The article does not identify the local subdivision count or provide the
original reference vector for this figure. Matching physical data and nominal
spaces alone does not establish identical historical numerical reproduction.
A final curve requires measured local and classical resolution, matching
discretizations and the stated full norm.

## Selected figures

The selected convergence and spatial panels remain for regeneration from
verified archives and fresh norm comparisons. Their final accuracy
comparison is pending the local/reference controls above.

![Selected face, local and classical-reference refinement](../figures/periodic/convergence.png)

![Selected pressure and signed Darcy-flux panels](../figures/periodic/fields.png)

Spatial panels highlight the actual macrogrid and reserve separate color scales
for pressure and each signed Darcy-flux component. Profiles retain independent
one-sided values at macrofaces and mark those intersections. Pixel samples
illustrate fields; they do not define integrated norms. The raw constitutive
flux of a primal Q1 field is not an $H(\mathrm{div})$ reconstruction.

## Acquisition and refinement procedures

For the conforming Q1 reference space:

~~~bash
pixi run --locked -e test-core python examples/periodic_reference.py \
  --degree 1 --order 4 --sizes 512 1024 2048 4096 --native-threads 1 \
  --artifacts build/results/periodic-q1-reference \
  --records build/results/periodic-q1-reference-records
~~~

This command defaults to Q5 and order 10 when `--degree` and `--order` are
omitted. It preserves the shared Galerkin operator and the $10^{-10}$ original
equation criterion. The executed tensor-factor basis, ordered nodal cardinal
matrix and their digests are archived. Assembly and coefficient precision are
recorded separately; native extended reference coefficients require a compatible
effective significand on the reading host. Consumers verify the canonical
archived basis and the shared owners used for physical evaluation.

For a phased Q1/P0 acquisition at an accepted local/trace pair:

~~~bash
pixi run --locked -e test-core python examples/verify_periodic.py \
  --macro 8 --local-refinement 256 --segments 1 2 4 8 16 32 \
  --stage all --workers 1 --native-threads 1 \
  --refinement-precision extended --original-refinement-steps 2 \
  --artifacts build/results/periodic-r256
~~~

The stages `condense`, `solve`, `reconstruct` and `norms` can run separately.
Condensation retains compact skeletal contributions from one macrocell at a
time. Reconstruction rebuilds local responses and streams fields and correction
histories to files. Archives are published atomically after original-equation
checks. Source generations remain immutable for replay, while new acquisitions
record the sources they execute.

Classical fields impose zero pressure strongly; MHM imposes the same physical
data weakly. Consumers validate material, source, boundary, coefficient precision
and basis/orientation contracts. Legacy nodal archives without case provenance remain unverified.

Before increasing local refinement to r512, measure a maximum-trace single-cell
assembly and factorization with the same Q1/P0 spaces, order 4 and original
criterion. A full campaign, replay and r256-to-r512 increment follow the measured
resource preflight. This refines the local resolution while retaining the published skeletal spaces.

## References

- Diego Paredes, Frédéric Valentin, and Henrique M. Versieux (2017). *On the robustness of multiscale hybrid-mixed methods*, Mathematics of Computation 86(304), 525–548. [DOI: 10.1090/mcom/3108](https://doi.org/10.1090/mcom/3108).
