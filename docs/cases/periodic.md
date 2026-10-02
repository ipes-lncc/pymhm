# Periodic Darcy coefficient and face enrichment

The experiment follows the oscillatory example in §4 of
[Paredes, Valentin and Versieux (2017)](https://doi.org/10.1090/mcom/3108).
It compares increasing the number of constant normal-flux modes on each
macroface with refining the local bilinear finite-element space. The classical
reference is refined independently; its discretization error is measured
separately from the MHM differences.

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

The physical flux is $q=-K_\varepsilon\nabla p$. The source has no extra factor
of $\pi$ in its arguments. The scalar coefficient ranges from 1 to 101.
The fixed macrogrid has $8\times8$ squares. Each local field is continuous $Q_1$
on an $r\times r$ Cartesian submesh, and the skeleton has $s=1,2,4,8,16,32$
independent piecewise constant modes per macroface. The local constant is
retained through its physical mean. Dirichlet data enter the hybrid weak form.

![Permeability and source on the fixed macrogrid](../figures/periodic/problem.png)

Local assembly uses four Gauss points per coordinate. The classical conforming
$Q_k$ calculation assembles one global continuous space and imposes the same
homogeneous Dirichlet data strongly. Its polynomial degree, cell count and
assembly quadrature are recorded for every run. This is an independently
refined discretization using PyMHM's original Cartesian finite-element
implementation; it is not an execution of an external reference program.

## Independent full-system comparison

A separate **DOLFINx 0.9.0/UFL** application assembles the complete broken
$Q_1$/piecewise-$P_0$ hybrid problem on the same $8\times8$ macrogrid,
with the same coefficient, source and weak boundary condition. It uses
$r=16,32$ and $s=1,2,4,8$: eight complete physical cases. Volume and load
matrices, canonical face orientation and coupling moments are assembled
independently. SciPy SuperLU solves the uncondensed saddle system with two
corrections evaluated in the original equations.

The largest case has 69,696 pressure coefficients and 1,152 trace coefficients.
Across the eight cases, relative differences in pressure, broken gradient and
physical flux are below $5.07\times10^{-13}$. Independent integration with
eight and ten Gauss points per coordinate verifies the physical norms.
The [comparison records](../figures/periodic/native-discrete-verification.json)
identify the source revision, installed native build, executed solver,
coefficient archives and nodal permutations.

This agreement verifies the stated finite hybrid discretizations. The
separately refined classical reference below measures their approximation
error; agreement between implementations alone does not make these coarse
discretizations accurate.

## Exact separated assembly and iterative reference solver

The declared coefficient is the exact finite sum
$K(x,y)=1+a(x)b(y)$, rather than a low-rank approximation. Weighted
one-dimensional matrices give the same tensor-Gauss bilinear form as ordinary
cell assembly:

$$
A=M_y\otimes S_x+S_y\otimes M_x
  +M_y^{b}\otimes S_x^{a}+S_y^{b}\otimes M_x^{a}.
$$

Tests compare every matrix and load entry against elementwise assembly for
$Q_1$ through $Q_5$, including variable coefficients, translated rectangles,
nonzero sources and inhomogeneous Dirichlet data. At $256^2$ cells, the $Q_5$
separated and elementwise solves differ by $4.08\times10^{-11}$ in relative
$H^1$.

The matrix-free reference solver applies these same Kronecker terms. Conjugate
gradients uses a symmetric AMG V-cycle for the conforming $Q_1$ problem on the
same equidistant nodal grid. This lower-order operator is solely a
preconditioner; the solved field remains $Q_5$. Positive sampled permeability
and full strong Dirichlet elimination give positive-definite Galerkin operators.
No contrast-independent convergence rate is assumed. The true residual of the
original operator must satisfy $\|b-Au\|_2\leq10^{-10}\|b\|_2$.
The explicit extended-precision mode accumulates correction digits while keeping
operator storage and Krylov corrections in double precision.

On the $64^2$ and $128^2$ controls, this solver agrees with direct PARDISO fields
to relative $H^1$ differences $7.22\times10^{-13}$ and
$5.50\times10^{-12}$, respectively. Their measured true residuals are
$1.36\times10^{-14}$ and $7.92\times10^{-14}$. These are algebraic controls,
not estimates of discretization error. Acquisitions can run concurrently with
other campaigns; recorded wall times are not controlled performance benchmarks.

An independent physical weak-form check differentiates the archived cardinal
polynomials through nodal differences and integrates their element contributions
in extended precision. Each diagnostic identifies the actual archived field:

| $Q_5$ field | Relative physical equation residual | Relative energy–source-work defect |
|---|---:|---:|
| $512^2$ refinement control | $2.19\times10^{-9}$ | $5.56\times10^{-11}$ |
| $1024^2$ comparison denominator | $3.38\times10^{-7}$ | $5.68\times10^{-10}$ |
| $1024^2$ repeat acquisition | $8.75\times10^{-9}$ | $1.86\times10^{-10}$ |

These physical residuals are distinct from the Kronecker solver's residual.
The two $1024^2$ acquisitions differ by $8.53\times10^{-10}$ in relative full
$H^1$ and $8.97\times10^{-10}$ in relative $L^2$. The
[field-equivalence record](../figures/periodic/periodic-q5-1024-equivalence.json)
preserves their separate archive digests, iteration counts and physical checks;
it does not identify them as the same coefficient vector.

With $K\geq1$, $C_P=1/(\pi\sqrt2)$ and the smallest eigenvalue
$\lambda_M=0.0232642487261$ of the reference one-dimensional $Q_5$ mass matrix,
the assembled mass lower bound and the corresponding discrete correction
estimate are

$$
\begin{aligned}
M_{\rm global}&\succeq h^2\lambda_M^2 I,\\
\|\delta p\|_{H^1(\Omega)}
&\leq \sqrt{1+C_P^2}\,C_P\,
\frac{\|r\|_2}{h\lambda_M}.
\end{aligned}
$$

Dividing by each archived field's full $H^1$ norm gives relative estimates
$1.005\times10^{-7}$, $1.551\times10^{-5}$ and $4.018\times10^{-7}$,
respectively, for the three fields in the table. All are well below the
measured $512^2\to1024^2$ change of 1.85609%. This controls the algebraic
perturbation within the same quadrature-based discrete problem, excluding
spatial and quadrature error. Constants are evaluated in floating arithmetic,
so this is not an interval-certified bound. The
[physical-form record](../figures/periodic/periodic-physical-form.json) identifies
the field hashes, formulas and independently evaluated quantities.

## Norms and comparison with the publication

The plotted relative norm is the full broken $H^1$ norm,

$$
\frac{\bigl(\sum_T\left[\|p_{\rm ref}-p_h\|_{L^2(T)}^2+
\|\nabla p_{\rm ref}-\nabla p_h\|_{L^2(T)}^2\right]\bigr)^{1/2}}
{\bigl(\|p_{\rm ref}\|_{L^2(\Omega)}^2+
\|\nabla p_{\rm ref}\|_{L^2(\Omega)}^2\bigr)^{1/2}}.
$$

Products are integrated exactly on the common nested Cartesian partition.
One-dimensional polynomial mass and derivative cross-moments give equivalent
tensor-product integrals. Independent positive Gauss quadrature tests these
products, including different polynomial degrees and independent values on
either side of macrointerfaces. Near cancellation, direct positive quadrature
of the field difference preserves the small norm. Interface values are never
averaged. Both the full norm and the gradient seminorm are retained in the data.

The published face-enrichment curve is independently extracted from the vector
PDF of Figure 6. Its error bars represent a vertical allowance of one PDF
point after logarithmic calibration; they are not uncertainty bounds on the
published numerical solver. The numerical data and spaces above match the
problem and face-enrichment configuration. The article does not report the
local subdivision count, so agreement with the historical curve cannot be
inferred from matching the nominal polynomial spaces alone.

Section 4 states a classical reference with 16,777,216 bilinear cells for its
initial example at $\varepsilon=1/64$. The later face-enrichment example uses
$\varepsilon=\pi/150$. Our $4096\times4096$ conforming $Q_1$ control uses that
stated resolution with the latter coefficient. The original reference field
for Figure 6 is unavailable here. Higher-order, independently refined
classical fields quantify sensitivity to the chosen reference.

## Recorded reference and local sensitivity

The primary numerical denominator is the continuous $Q_5$ field on
$1024\times1024$ cells: **26,224,641 nodal unknowns**, tensor Gauss order 10,
and the matrix-free operator with its low-order-refined AMG preconditioner.
It required 414 CG iterations, including the true-residual correction solves,
and satisfied the original relative equation criterion with $6.11\times10^{-12}$.
Its acquisition took 4064.96 seconds with eight native threads; this concurrent
workstation run is not a controlled timing comparison.

| Classical family | Last refinement | Relative full $H^1$ change |
|---|---:|---:|
| $Q_1$, quadrature 4 | $2048^2\to4096^2$ | 9.76864% |
| $Q_3$, quadrature 4 | $512^2\to1024^2$ | 7.53776% |
| $Q_5$, quadrature 10 | $256^2\to512^2$ | 10.36417% |
| $Q_5$, quadrature 10 | $512^2\to1024^2$ | **1.85609%** |

The last $Q_5$ relative $L^2$ change is 0.0269437%. Increasing assembly
quadrature from 6 to 10 at $512^2$ changes relative $H^1$ by
$1.0871\times10^{-5}$; at $256^2$, order 8 versus 10 differs by
$1.7187\times10^{-9}$. Quadrature and spatial refinement are separate controls.
The last reference change is smaller than the smallest MHM/reference difference,
but is still material: it is a measured sensitivity, **not an upper bound on
the unknown exact-solution error**.

At local $r=512$, face enrichment gives:

| Segments per macroface | MHM / $Q_5$ 1024: relative $H^1$ | MHM / $Q_1$ 4096: relative $H^1$ |
|---:|---:|---:|
| 1 | 18.32730% | 17.51719% |
| 2 | 13.68952% | 12.52390% |
| 4 | 13.04947% | 11.81293% |
| 8 | 12.78897% | 11.52118% |
| 16 | 7.07969% | 4.24357% |
| 32 | 5.90524% | 1.60526% |

The $Q_1$ 4096 reference and the $r=512$ local fields have the same cell width.
Their mutual difference shares part of the local approximation error. The
$Q_1$ reference itself differs from $Q_5$ 1024 by 5.68522% in $H^1$; its much
smaller MHM distance at 32 segments does not establish a correspondingly small
exact-solution error. Against the independent $Q_5$ denominator, local refinement
at that fixed skeleton gives 22.1242%, 11.3982% and 5.90524% for
$r=128,256,512$. The local error is therefore not negligible at these settings.

The finest recorded MHM distance and the published Figure 6 ordinate
(approximately 3.9690%) differ. Since the historical local refinement and
reference field are unspecified, these data verify the stated spaces and
separate refinement sensitivities; they do not establish identical historical
numerical reproduction.

![Face, local and classical-reference refinement](../figures/periodic/convergence.png)

## Physical fields

![Pressure and signed flux components](../figures/periodic/fields.png)

Each map samples the finite-element field directly at $512\times512$ pixel
centers. The displayed flux is the physical constitutive flux, with one common,
zero-centered scale per signed component. These sample maps illustrate the
fields; all reported differences use integrated polynomial norms. Macrogrid
contours are the actual square boundaries. The raw constitutive flux of a
primal $Q_1$ field is generally discontinuous across fine-cell boundaries and
is not an $H(\mathrm{div})$ reconstruction.

## Reproduction

```bash
pixi run -e notebooks python examples/verify_periodic.py --reference-only \
  --reference-levels 256 512 1024 2048 4096
pixi run -e notebooks python examples/verify_periodic.py --reference-only \
  --reference-degree 3 --reference-levels 128 256 512 1024
pixi run -e notebooks python examples/verify_periodic.py --reference-only \
  --reference-degree 5 --reference-levels 128 256 512
pixi run -e notebooks python examples/verify_periodic.py --reference-levels 4096 \
  --local-refinement 128 --workers 2
pixi run -e notebooks python examples/verify_periodic.py --reference-levels 4096 \
  --local-refinement 256 --workers 2
pixi run -e notebooks python examples/verify_periodic.py --reference-levels 4096 \
  --local-refinement 512
pixi run -e intel python examples/verify_periodic.py --reference-only \
  --reference-degree 5 --reference-quadrature 10 --reference-assembly separable \
  --reference-solver pypardiso --reference-levels 64 128 256 512 --native-threads 8
pixi run -e test python examples/verify_periodic.py --reference-only \
  --reference-degree 5 --reference-quadrature 10 --reference-assembly lor \
  --reference-levels 1024 --native-threads 8
pixi run -e test python examples/compare_periodic.py --reference 5:1024:10:lor \
  --compare-references 5:512:10:separable 3:1024 1:4096 \
  --refinements 128 256 512 --primary
pixi run -e notebooks python examples/plot_periodic.py
```

Each MHM run prepares the largest face space once and restricts its already
computed local responses to the coarser face partitions. This changes neither
the local Galerkin operator nor the prescribed face space. Numerical archives
are kept under `build/results/periodic`; result summaries and extraction
provenance are in `examples/results`. The largest local preparation stores
all local responses and requires substantially more memory than the final
skeletal solve. Source hashes attached to new acquisitions and separate
comparison hashes distinguish operator acquisition from later norm evaluation.
