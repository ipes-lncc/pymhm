# PGMHM on the first SPE10 layer

This case evaluates the Petrov–Galerkin MHM of
[Fernando et al. (2023), §6.3](https://doi.org/10.1007/s40314-023-02304-y)
on the layer and boundary conditions shown in Figures 11–13. It distinguishes
material identification, discretization verification, and agreement with the
historical plots. The paper does not supply a complete executable specification
of this heterogeneous experiment; the choices below therefore remain explicit.

## Physical data and identification of the coefficient

The domain is $(0,1200)\times(0,2200)$, with the original numerical coordinates
in feet. Pressure is one on the bottom boundary and zero on the top boundary;
the two vertical sides have zero outward flux. The adopted volume source is
zero. No conversion to SI or rescaling of the material is applied:

$$
\begin{aligned}
 -\nabla\cdot(K\nabla p)&=0, & q&=-K\nabla p,\\
 p\big|_{y=0}&=1, & p\big|_{y=2200}&=0,\\
 q\cdot n\big|_{x=0,1200}&=0.&&
\end{aligned}
$$

The pressure/flux campaign uses the **$K_x=K_y$ component of SPE10 layer one**,
applied isotropically. The printed material image and pressure profile require
separate identification: Figure 11 matches **$K_z$**, whereas the reference
profile in Figure 13 supports **$K_x$**. The article does not name a component
in the heterogeneous subsection. The material image’s printed limits, approximately $3.6\times10^{-7}$ and $1.4\times10^3$,
agree with the dataset's $K_z$ limits $3.636\times10^{-7}$ and $1394.25$.
The $K_x=K_y$ limits are instead $0.003033$ and $4647.5$.

The spatial identification also uses all 13,200 geological cell centers.
Ranking their colors against the figure's own color scale gives a correlation
of **0.99577 for $K_z$**, compared with **0.76981 for $K_x$ or $K_y$**. Neither
axis reversal improves the match. The JPEG compression and printed color scale
limit color precision, so this evidence identifies a component and orientation;
it does not recover historical finite-element coefficient interpolation.
[The material record](../figures/pgmhm-spe10/material-identification.json)
preserves the image digest, sampling boxes, component comparisons and dataset
digest. Dataset provenance and units follow the [reservoir data page](spe10.md).

The pressure profile provides an independent observable. At eight visible
plateau samples between $y=450$ and $1050$, the classical $K_x$ solution on
$60\times220$ pixel-aligned rectangles differs from the printed reference by
at most **0.00131** in pressure. The graphical half-widths are approximately
0.0027–0.0034. The $K_z$ solution on $240\times880$ rectangles differs by as
much as **0.7312** over the same samples. This comparison uses the same
source, pressure datum and boundary conditions; neither the coefficient nor
the stabilization parameter is fitted to the curve. The transition region is
less resolved and partially occluded, so these plateau values do not establish
complete historical finite-element equivalence.
[The digitized values](../figures/pgmhm-spe10/profile-digitization.json) and
[the component comparison](../figures/pgmhm-spe10/profile-component-comparison.json)
preserve all selected points and their individual graphical uncertainty.

The results below therefore identify $K_x$ as the **profile-supported solve
configuration**, and retain $K_z$ as the **image-identified material control**.
A single component cannot reproduce both published observables under the
declared zero-source diffusion problem. The historical input linking those
two figures is not available.


## Spaces and integration

The macro mesh has two triangles, separated by the **northwest–southeast**
diagonal visible in Figure 13. Each macrotriangle has 1,024 uniformly refined
fine triangles, for 2,048 local triangles in total. The local pressure space is
continuous $P_2$ on each local triangulation; it is broken across the macroface.
The multiplier space is discontinuous $P_0$ on each independently partitioned
macroface. Thus “P2/P0” refers to **local pressure / normal-flux multiplier**,
not to a mixed RT pressure degree or a global conforming pressure pair.

One, two and four equal trace segments produce five, eight and fourteen free
global unknowns after side-Neumann data are eliminated, including the two
retained local constant modes. Eight and sixteen segments are additional trace
resolution controls, with the same local pressure space and physical data.
These additional configurations are not attributed to Figure 13.

The coefficient is constant on the original $60\times220$ geological pixels.
Every fine-cell integral is split at pixel interfaces. The local bilinear form
is

$$
 a_K(p,v)=\int_K K\nabla p\cdot\nabla v.
$$

It contains $K$, not $1/K$, and requires no derivative of the discontinuous
coefficient. The Petrov–Galerkin residual is a **pressure-jump residual on the
macro skeleton**, not a strong volume residual involving $\nabla K$. Exact
integration of material cuts does not, however, introduce additional pressure
DOFs at a cut: a polynomial crossing a pixel boundary still cannot represent
two independent normal derivatives there.

The declared stabilization parameter is $\alpha=0.1$, with the global lower
ellipticity bound in the paper's formula:

$$
 \tau_E=\frac{\alpha K_{\min}}{2H_E}.
$$

$H_E$ is the full macroface length. It does not shrink when that face is
partitioned. The historical value of $\alpha$, local polynomial degree,
local connectivity and coefficient representation are not stated in §6.3.
Consequently, matching two macros, 2,048 local cells and fourteen unknowns
does not establish complete historical discretization equivalence.

## Independent variational verification

A complete DOLFINx/UFL comparison uses the full layer-1 $K_x$ reservoir,
the same two macrotriangles, unfitted P2/r32 local space and P0 traces with
one, two and four segments. An independent material-intersection construction
integrates the UFL forms on 60,800 cut triangles and restricts them to the
original P2 polynomials. PETSc/MUMPS solves the complete auxiliary PGMHM
system, including residual enrichment, without PyMHM condensation.
Across the three configurations, relative physical L2 differences are at
most $1.46\times10^{-11}$ for pressure and $7.40\times10^{-11}$ for Darcy
flux; quadrature orders four and six agree. The largest original-system
residual is $6.36\times10^{-15}$.
The [full-case numerical record](../figures/pgmhm-spe10/native-discrete-verification.json)
identifies the input fields, comparison-module digests and verified
DOLFINx 0.9.0 source provenance. This verifies the declared discrete reservoir
problem while the historical input discrepancy remains unresolved.

A native DOLFINx/UFL check uses a four-region coefficient with the same
$K_z$ contrast, $3.8346\times10^9$, nonzero Dirichlet values and zero lateral
flux. UFL assembles on a mesh fitted to the material and then restricts the
operator to the **same unfitted coarse P2 polynomials**. The global comparison
uses the complete uncondensed PGMHM equations and both base and enriched
pressures. It does not reuse PyMHM's condensed matrix or its cut quadrature.

The relative operator difference is $4.80\times10^{-16}$. The relative base
and enriched pressure differences are below $6.6\times10^{-15}$, and the
multiplier difference is $8.57\times10^{-15}$. Both implementations exhibit
the same lack of a discrete maximum principle for that coarse space.
[The numerical record](../figures/pgmhm-spe10/ufl-verification.json) and
`tests/test_pgmhm_heterogeneous_fenics.py` make this operator and field check
explicit. Agreement of two discrete implementations establishes neither
accuracy on the full reservoir nor agreement with the historical figure.

For the $K_z$ material control's four-segment configuration, increasing the material-cut
quadrature from order five to order eight changes the pressure coefficients
by $1.25\times10^{-10}$ relatively. This is a consistency check of the stated
operator integration, separate from spatial approximation error.



## Refined baseline and measured spatial differences

The separately refined PyMHM RT2 sequence contains 60×220, 120×440 and
240×880 pixel-aligned rectangles, each split into two triangles. The last
reference has **6,972,960 mixed unknowns**. Its 120→240 refinement increment
is **0.00985% in pressure**, **1.9965% in flux L²** and **2.0511% in weighted
flux**. These are differences between two numerical solutions, not bounds
on the error from the exact reservoir solution. The last algebraic residual
is 3.84×10⁻¹⁶ and the maximum fine-cell flux balance is 1.89×10⁻¹².

The following table uses the 240×880 reference as the denominator in each
physical norm. Integrals use the exact intersections of the local triangles,
reference triangles and material pixels; orders four and five independently
check the same integrals. The weighted flux norm is

$$
 \lVert q-q_{\mathrm{ref}}\rVert_{K^{-1}}^2
 =\int_\Omega K^{-1}\lvert q-q_{\mathrm{ref}}\rvert^2.
$$

| P0 segments per macroface | Free global DOFs | Pressure L² | Raw flux L² | Weighted raw flux | Nodal minimum p | Nodal maximum p |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 26.9223% | 66.8792% | 66.1353% | -0.13784 | 1.03741 |
| 2 | 8 | 10.9814% | 82.4278% | 56.1914% | -0.25443 | 1.02554 |
| 4 | 14 | 8.0615% | 60.9722% | 53.4655% | -0.31840 | 1.01550 |
| 8 | 26 | 2.0391% | 61.6871% | 47.7395% | -0.00122 | 1.00760 |
| 16 | 50 | 1.8921% | 60.8803% | 47.3751% | -0.00104 | 1.00138 |

The pressure profile improves substantially when more macroface moments are
retained. The raw flux difference remains about 61% at sixteen segments,
well above the classical reference's own 2% increment. That result identifies
the stated local polynomial resolution as inadequate for a quantitatively
accurate flux, even though material integration is resolved. It does not
establish that further trace enrichment alone resolves the interior material
response. The base and residual-enriched fields are both archived; their
relative differences from RT2 are almost equal for this parameter and
coefficient. The enrichment nevertheless enforces the prescribed macro
balance: its maximum residual is below 1.3×10⁻¹⁵ over these five cases.

![Material, published profile samples and independent macro-local pressures](../figures/pgmhm-spe10/material-profiles.png)

The graphical reference samples retain their digitization uncertainty. The
computed profiles preserve both values at the macroface intersection; no
averaging or clipping is applied.

![Pressure and signed horizontal flux compared with the refined classical field](../figures/pgmhm-spe10/fields.png)

Pressure is evaluated inside the actual broken P2 cells. Flux panels show
geological cell-center samples for visualization; the table uses physical
intersection integrals, not the RMS of those samples.

![Reference refinement and PGMHM field differences](../figures/pgmhm-spe10/norms.png)

The five-point trace study preserves the same 2,048 local triangles. It is
a resolution control of the declared method, not a reproduction of unspecified
historical local spaces.

## Separating material approximation from skeleton resolution

A further control fits each local triangulation to the exact permeability
interfaces, while preserving continuous P2, the two macrotriangles, sixteen
P0 segments per macroface, $\alpha=0.1$ and the original physical data.
The trace space and its 50 free global unknowns are identical to the last
nominal row. Fitting introduces 112,896 fine triangles. It changes the local
approximation space, whereas the nominal material-cut quadrature only changes
how that earlier polynomial space is integrated.

| Local material representation | Fine triangles | Pressure $L^2$ | Raw flux $L^2$ | Weighted raw flux |
|---|---:|---:|---:|---:|
| P2 polynomials crossing pixels, exact material cuts | 2,048 | 1.89208% | 60.88034% | 47.37512% |
| Material-fitted continuous P2 | 112,896 | 0.66371% | 12.22491% | 8.53321% |

Both rows use the same independently refined RT2 reference and the same
physical norm denominators. Norm orders four and five agree to rounding
precision for the fitted case. Its global residual is $1.75\times10^{-16}$;
the enriched macro balance is below $1.8\times10^{-16}$.
The base pressure range is $[-0.001962,1.001590]$, without clipping.

This controlled reduction demonstrates a significant local-space contribution
to the nominal flux discrepancy. It does not prove that all remaining error
comes from the skeleton, nor that the material-fitted field has reached its
asymptotic regime. The remaining **12.2249%** raw-flux difference is still
larger than the reference's own **1.9965%** increment. The fitted configuration
is an explicitly enriched control, not the historical 2,048-cell discretization.

A second control holds those **same 112,896 local triangles and P2 spaces**
fixed and enriches only the P0 skeleton. It starts with 64 uniform subdivisions
per macroface and includes every intersection with a geological pixel boundary.
The resulting five macrofaces contain 120, 320, 280, 280 and 120 segments;
the material interfaces themselves and the two macrotriangles are unchanged.
There are 1,122 total global unknowns and 562 free unknowns, including the
two retained constant modes.
The [geometry replay record](../figures/pgmhm-spe10/geometry-replay-verification.json)
checks every archived local coordinate and cell connectivity against the
material-fitting implementation; all arrays agree bit for bit.

| Fixed material-fitted P2 local spaces | Free global DOFs | Pressure $L^2$ | Raw flux $L^2$ | Weighted raw flux |
|---|---:|---:|---:|---:|
| Uniform P0 trace, 16 segments per macroface | 50 | 0.66371% | 12.22491% | 8.53321% |
| 64 uniform subdivisions plus material-interface cuts | 562 | 0.03564% | 5.24665% | 5.15711% |

The second comparison isolates a skeletal-space change from the preceding
local-space change. Its order-four and order-five physical integrals agree
within $7\times10^{-15}$ relatively. The global residual is
$5.96\times10^{-15}$ and the maximum enriched macro balance is
$6.26\times10^{-15}$. The nodal pressure range is
$[-3.1534\times10^{-5},1.00001025]$, with no clipping. The base and enriched
flux differences agree to eight significant digits.

At this local resolution, the **5.24665%** raw-flux distance uses a numerical reference whose
own last increment is **1.99652%**. These quantities do not provide an error
bound or an orthogonal error decomposition. They show that both local material
resolution and the retained macroface moments affect this experiment. Neither
enriched configuration is attributed to the historical 2,048-cell experiment.

A second pair of controls uses **182,976 material-fitted fine triangles**,
starting from local refinement $r=64$. The first retains exactly the preceding
562 free skeleton/kernel coordinates; the second enriches only the trace on
those same local meshes. All rows retain P2 local pressure, P0 multipliers,
$\alpha=0.1$ and the same coefficient, source and boundary data.

| Local refinement | Fine triangles | Uniform trace subdivisions plus material cuts | Free global DOFs | Pressure $L^2$ | Raw flux $L^2$ | Weighted raw flux |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 112,896 | 64 | 562 | 0.03564% | 5.24665% | 5.15711% |
| 64 | 182,976 | 64 | 562 | 0.02064% | 4.19206% | 4.04196% |
| 64 | 182,976 | 128 | 754 | 0.02061% | 4.18370% | 4.02654% |

The $r=64$ pair reuses the same local operators and harmonic responses. The
smaller skeleton is obtained by exact restriction of the assembled global
form and its load; reconstruction uses the original local lifts. This preserves
the stated discrete problems. The larger trace has 184, 384, 344, 344 and
184 segments on the five macrofaces. Its global residual is
$2.38\times10^{-14}$ and its maximum enriched macro balance is
$1.78\times10^{-14}$. The nodal pressure range is
$[-5.4831\times10^{-6},1.00000517]$. Both norm quadratures agree within
$7\times10^{-15}$ relatively.

At fixed trace, the additional local resolution reduces the raw-flux distance
from 5.24665% to 4.19206%; subsequent trace enrichment changes it to 4.18370%.
These differences measure the two specified space changes separately. They do
not assign an exact-continuum error to either space: the reference's own
1.99652% increment is approximately 48% of the final measured distance. A
finer classical reference is therefore needed to quantify the remaining flux
discrepancy with substantially smaller reference uncertainty.

![Signed flux components for the nominal space, refined local and skeletal spaces, and classical reference](../figures/pgmhm-spe10/control-flux-components.png)

Both signed components are evaluated at the actual fitted fine-cell centroids;
the same sample locations and display triangles are used in all three columns.
The right column uses the final material-fitted local and skeletal spaces;
the middle column retains the nominal unfitted P2/r32/s16 space.
The reference is H(div); both PGMHM fields are raw gradients. Colors use a
common symmetric-logarithmic normalization in each row, with a linear central
interval of one percent of that row's largest component magnitude. This
normalization makes smaller channels visible while retaining every extremum.
The macro diagonal is highlighted in all panels. These samples visualize the
fields; the table continues to use exact physical intersection integrals.

![Fixed-method local and skeletal resolution controls](../figures/pgmhm-spe10/resolution-controls.png)

## Reproducible acquisitions

The original driver saves both pressures, independent macro-local coefficient
vectors, actual geometry, trace coefficients and one-sided profiles at
$x=600$. Wider-precision coefficients use portable float64 high/remainder/tail
arrays. Local residual refinement is explicitly requested without changing the
operator or its residual tolerance.

```bash
pixi run -e intel python -m examples.solve_pgmhm_spe10 mhm --component kx --segments 1 2 4 8 16
pixi run -e intel python -m examples.solve_pgmhm_spe10 reference --component kx \
  --nx 60 120 240 --solver pypardiso --native-threads 4
pixi run -e fem python -m pytest -q tests/test_pgmhm_heterogeneous_fenics.py
pixi run -e intel python -m examples.solve_pgmhm_spe10 mhm --component kx \
  --refinement 32 --segments 64 --material-fitted --trace-fitted \
  --output examples/results/pgmhm-spe10/kx/controls
pixi run --locked -e test-core python -m examples.solve_pgmhm_spe10 compare --component kx \
  --mhm examples/results/pgmhm-spe10/kx/controls/pgmhm-fitted-tracefit-r32-s64-q5.npz \
  --reference examples/results/pgmhm-spe10/kx/classical-rt2-240x880.npz \
  --output examples/results/pgmhm-spe10/kx/controls --workers 8
```

The classical reference is the global conforming RT2/P2 mixed method with no
macro restriction. The meshes resolve every material pixel and use the same
operator, boundary conditions and pressure datum. Its own refinement increment
must be assessed before using it as a numerical baseline. Raw PGMHM fluxes
remain $-K\nabla p$; they are not $H(\operatorname{div})$ reconstructions.

## References

- Honório Fernando, Larissa Martins, Weslley Pereira, and Frédéric Valentin (2023). *A Petrov–Galerkin multiscale hybrid-mixed method for the Darcy equation on polytopes*. Computational and Applied Mathematics 42, article 173. [DOI: 10.1007/s40314-023-02304-y](https://doi.org/10.1007/s40314-023-02304-y).
