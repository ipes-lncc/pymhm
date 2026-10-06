# Petrov–Galerkin MHM on polytopes

`solve_pgmhm` implements the residual enrichment in equations (27)–(34) of
Fernando, Martins, Pereira and Valentin, [*A Petrov–Galerkin multiscale
hybrid-mixed method for the Darcy equation on polytopes*, Computational and
Applied Mathematics 42, 173 (2023)](https://doi.org/10.1007/s40314-023-02304-y).
The method changes the global test equations and reconstructs an additional
local pressure correction. It is distinct from merely supplying unequal
trial and test kernels to a generic condensation interface.

The implementation supports triangular and simple polygonal macroelements
in two dimensions, including nonconvex polygons. Local spaces are continuous
Lagrange elements. Trace degree, trace partition and local mesh can vary
independently, subject to the compatibility conditions below. Materials may
be scalar or symmetric positive-definite tensors; Dirichlet, mixed and
compatible pure Neumann data are supported.

The [SPE10 layer-one case](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/pgmhm-spe10.md) specifies the permeability
component identified in the original figure, the mixed boundary conditions,
and separate local/trace and classical-reference resolution checks.

## Formulation and signs

The physical Darcy flux is $q=-K\nabla p$. The stored multiplier $\lambda$
uses physical normal flux, opposite to the article's conormal convention.
Let $R_K$ be the inverse of the local diffusion operator constrained to zero
volume mean, $B_K$ its normal-trace coupling, and $Z_K=1$. The base field is

$$
 p_K=R_Kf_K-R_KB_K\lambda+Z_Kc_K.
$$

Write $p=P x+p_f$ with $x=(\lambda,c)$. On an interior macroface,
$J_Ep$ is the signed difference between the incident pressures, using the
same normal orientation as the skeleton. On an exterior Dirichlet face it
is the outward trace. Define

$$
 \tau_E=\frac{\alpha K_{\min}}{2H_E},\qquad \alpha>0.
$$

$H_E$ is the **full macroface length**, including when its multiplier space
is segmented. It is not the length of an individual segment. The global
equations are the usual MHM equations plus

$$
\begin{aligned}
 A_{\rm PG}&=A_{\rm MHM}
 +\sum_E (J_EP)^T W_E(J_EP),\\
 b_{\rm PG}&=b_{\rm MHM}
 +\sum_E (J_EP)^T W_E(g_E-J_Ep_f),
\end{aligned}
$$

where $W_E$ integrates $\tau_E$, and $g_E=0$ on interior faces. On exterior
Dirichlet faces, the prescribed trace contributes to both global equations.
The matrix is symmetric; the positive jump contribution does not make every
choice of $\alpha$ a positive-definite global formulation.

The residual enrichment and postprocessed pressure are

$$
\begin{aligned}
 \lambda_R\big|_E&=-\tau_E(J_Ep-g_E),\\
 p_{R,K}&=-R_K\langle\lambda_R,\cdot\rangle_{\partial K},\\
 \widetilde p_K&=p_K+p_{R,K}.
\end{aligned}
$$

The sign in $\lambda_R$ follows the physical-flux convention. The local
correction has zero volume mean. It requires additional local solves but no
additional global unknowns. Exterior pressure is projected in the $L^2$
sense onto piecewise $P_k$ on the common incident fine-face partition; its
moments therefore agree with the original Dirichlet functional even when
the data are nonpolynomial. This is the fine-trace realization of equation
(32), whose single-face polynomial notation also covers a single local
element.

## Conservation and physical boundary data

Only the **enriched** multiplier satisfies the macro balance:

$$
 \int_{\partial K}(\lambda+\lambda_R)=\int_K f.
$$

`conservation_residuals(enriched=True)` and
`normal_flux(cell, face, parameter, enriched=True)` use this quantity.
The latter returns the outward value on the selected macrocell; values from
opposite sides have opposite signs. The stored `hybrid.trace` contains only
$\lambda$. Its integral generally does not balance the source.

`pressure` is the base approximation in equation (31);
`enriched_pressure` is equation (34). Their corresponding volume fluxes are
the raw gradients $-K\nabla p$ and $-K\nabla\widetilde p$. Macro conservation
of the enriched multiplier does not establish fine-cell conservation or
global $H(\mathrm{div})$ conformity of either raw gradient.

Neumann faces prescribe the physical outward flux essentially, following
the treatment stated in section 6.3 of the article. They receive no pressure
jump penalty and no flux enrichment. A compatible all-Neumann boundary uses
the physical volume mean of pressure as a gauge. A nonzero `mean_pressure`
is rejected when Dirichlet faces are present.

## Stability hypotheses and parameter choice

The local degree is required to satisfy $k\ge\ell+2$ in two dimensions.
Lemma 2 and Theorem 4 of the published article require $k\ge\ell+d$
and $\ell\ge1$; Theorem 4 also requires sufficiently small $\alpha$.
The article includes $\ell=0$ numerical experiments separately from these
stated stability hypotheses.
Fine triangulations must be shape regular and compatible with the skeleton
moments.

The approximation estimate in Theorem 6 assumes $k\ge\ell+d$,
$u\in H^{s+1}(\mathcal P)$ and $K\nabla u\in[H^m(\mathcal P)]^d$, with
$1\le s\le k$ and $1\le m\le\ell+1$. Although its statement allows
$\ell\ge0$, its proof uses the stability estimate (50) from Theorem 4.
The $\ell=0$ results below remain numerical observations under their declared
discretizations.

`local_refinement_precision="extended"` explicitly retains additional correction
digits in both local responses and enrichment solves where the platform provides
a wider real type. It preserves the original operator and residual threshold;
it is not an accuracy or stability criterion. A numerical rank check does not establish a mesh-independent
inf-sup constant on arbitrary fine and skeletal meshes.

`stabilization_parameter` is an explicit required argument. The article's
numerical section does not specify its numerical value, and its theorem
does not provide a universal computable threshold. The campaign below uses
$\alpha=0.1$, stated as a selected numerical parameter. It is not inferred
from fitting a convergence curve. Constant and Cartesian coefficients
provide $K_{\min}$ directly. Other coefficient callbacks require a certified
`ellipticity_lower_bound`; evaluation at sample points can reject an
inconsistent declaration but cannot certify it.

## Published smooth problem and two refinement mechanisms

Section 6.1 prescribes $K=I$, homogeneous Dirichlet data on the unit square,
and

$$
\begin{aligned}
 p(x,y)&=\sin(2\pi x)\sin(2\pi y),\\
 f(x,y)&=8\pi^2p(x,y),\qquad q=-\nabla p.
\end{aligned}
$$

The macro-refinement study uses $n=2,4,8,16,32$ and degree pairs
$(\ell,k)=(0,2),(1,3)$. Each square is split into two triangles; each
triangular macrocell has a single local element, as explicitly stated in
the article for this experiment. The second family uses a declared tiling
by complementary L-shaped polygons, with local ear triangulations. Every
macroface has one multiplier segment.

Assembly and error integration orders are eight and ten. Norms are genuine
physical integrals. The figures distinguish base and enriched pressures,
their physical flux errors, the enriched broken divergence error
$\|\nabla\cdot(-\nabla\widetilde p)-f\|_{L^2}$, and macro balances.
The archived ordinary broken $H(\mathrm{div})$ norm has divergence weight
one, explicitly separate from the weighted convention printed in the
article. No claim of identical plotted values or historical L-mesh
connectivity is made.

The last macro-refinement interval gives the following base-field errors
and observed rates. These are rates for the stated finite sequences,
not a verification of every hypothesis of the error theorem.

| Macro family | Trace degree | Final pressure L² error | Final flux L² error | Pressure rate | Flux rate |
|---|---:|---:|---:|---:|---:|
| Triangles | 0 | 2.32255e−3 | 2.51626e−1 | 1.990 | 0.997 |
| Triangles | 1 | 1.40164e−5 | 3.30568e−3 | 2.989 | 2.020 |
| L-polygons | 0 | 2.16371e−4 | 6.51252e−2 | 1.988 | 0.996 |
| L-polygons | 1 | 7.38332e−7 | 4.74969e−4 | 3.083 | 2.045 |

For L-polygons with $\ell=0$, the enriched divergence error changes only
from 13.7161 to 13.3944 over the last interval, an observed rate of 0.034.
This sequence does **not** demonstrate convergence in the broken
$H(\mathrm{div})$ norm. The stronger-degree sequence has final divergence
error 0.238420 and rate 1.082. The largest enriched macro-balance residual
over all forty cases is $4.47\times10^{-12}$; that conservation result does
not remove the divergence-error limitation. The $\ell=0$ experiments also
fall outside the stated $\ell\ge1$ theorem.

![Triangular macro refinement, base and enriched fields](../figures/pgmhm/triangles-convergence.png)

![Nonconvex polygon macro refinement](../figures/pgmhm/L-polygons-convergence.png)

The space-refinement experiment holds the macro meshes fixed at sixteen
crisscross triangles or thirty-two L-shaped polygons. Each macroface has
$1,2,4,8,16$ segments, with the same number of subdivisions per original
local edge. This changes trace resolution while preserving macro geometry;
it is distinct from increasing the number of macrocells. The counts match
those reported for this experiment, but its historical connectivity and
local resolution are not available.

![Fixed-macro skeletal refinement](../figures/pgmhm/skeleton-convergence.png)

## Base and enriched fields

Every panel retains the actual macro mesh and independent one-sided field
values. Exact, base and enriched fields use a shared signed scale within
each row; the enriched error has its own labeled scale.

![Triangular pressure and signed physical flux](../figures/pgmhm/triangles-fields.png)

![Polygonal pressure and signed physical flux](../figures/pgmhm/L-polygons-fields.png)

## Verification and execution

Lightweight tests cover exact anisotropic quadratic pressures, nonconvex
geometry, mixed and pure Neumann boundaries, physical mean constraints,
nonpolynomial Dirichlet moments, unequal incident local trace partitions,
coefficient scaling, and serial/thread/spawn agreement. A fitted local
interface also verifies a permeability jump inside macroelements. Independent
DOLFINx/UFL assembly verifies a complete uncondensed system with variable
permeability, cubic source and exponential boundary data for P2 and P3.
It compares both pressure fields and the physical multiplier.

```bash
pixi run -e notebooks python -m examples.pgmhm_campaign --workers 8
pixi run -e notebooks python -m examples.plot_pgmhm
pixi run -e fem pytest tests/test_pgmhm_fenics.py
```

The [numerical record](../figures/pgmhm/comparison.json) includes every
resolution, the two pressures' norms, enriched and unenriched balances,
global dimensions and source digests. The repository notebook `notebooks/darcy/59_pgmhm.ipynb`
combines a small executable patch with these archived studies. These
experiments address the analytical case; the article's 27-by-27 inclusions
and first-layer SPE10 comparisons require their own heterogeneous reference
campaigns and are not asserted to be reproduced by this page.
