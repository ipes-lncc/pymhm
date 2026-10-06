# Three-field multiscale Darcy: MH²M

`solve_mh2m` implements the two-dimensional three-field method of
[de Barros, Madureira and Valentin (2026)](https://arxiv.org/abs/2404.16978v3), using its local operators and pressure-trace
system in equations (25)–(29)
(version 3, 5 August 2026). It has two independently specified interface spaces:
a globally continuous pressure trace $\Gamma_{H_\Gamma}$ and a broken conormal
space $\Lambda_{H_\Lambda}$. Refining $\Lambda$ changes local operators without
adding global pressure-trace unknowns.

The implementation uses triangular or simple polygonal macroelements, including
nonconvex polygons, continuous local Lagrange elements, independently partitioned
polynomial face spaces, and scalar or
symmetric positive-definite tensor permeability. Cartesian coefficient jumps
use the shared integration by material cuts. Nonhomogeneous Dirichlet, mixed
and compatible pure Neumann conditions are supported. These boundary conditions
extend the homogeneous Dirichlet numerical cases in the article.

Polygons retain their actual boundary sides and vertices in both interface
spaces. Local ear triangulations supply continuous finite-element spaces;
internal diagonals of this triangulation do not become macrofaces.

`solve_mh2m_3d` supplies a separately verified
[tetrahedral implementation](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mh3d.md) with a continuous triangular
pressure skeleton and independently resolved conormal faces. Its original
three-dimensional tests are distinct from the two-dimensional published
experiments below.

## Fields, signs and local averages

The physical fields and the article's conormal multiplier satisfy

$$
\begin{aligned}
 -\nabla\cdot(A\nabla p)&=f,& q&=-A\nabla p,\\
 \rho&=p\big\vert_{\partial\mathcal T_H},&
 \lambda_K&=A\nabla p\cdot n_K=-q\cdot n_K.
\end{aligned}
$$

There is one copy of $\lambda_K$ for each incident macroelement. In contrast,
pressure-trace values share their macrovertex degrees of freedom. This is a
different global unknown from the flux trace of the two-field MHM.
The local complement uses **boundary mean zero**:

$$
 \widetilde V_h(K)
 =\left\{v_h\in V_h(K):\int_{\partial K}v_h=0\right\},
 \qquad
 \widetilde\Lambda_H(K)
 =\left\{\mu_h\in\Lambda_H(K):\int_{\partial K}\mu_h=0\right\}.
$$

For $a_K(u,v)=\int_K A\nabla u\cdot\nabla v$, the discrete Neumann maps are

$$
\begin{aligned}
 a_K(T_h\mu,v_h)&=\langle\mu,v_h\rangle_{\partial K},\\
 a_K(\widetilde T_h f,v_h)&=(f,v_h)_K,
 \qquad v_h\in\widetilde V_h(K),\\
 \langle\mu,T_hG_h\phi\rangle_{\partial K}
 &=\langle\mu,\phi\rangle_{\partial K},
 \qquad\mu\in\widetilde\Lambda_H(K).
\end{aligned}
$$

The local mean and compatible constant conormal are

$$
 p_K^0=\frac{1}{\lvert\partial K\rvert}\int_{\partial K}\rho,
 \qquad
 \lambda_K^0=-\frac{1}{\lvert\partial K\rvert}\int_K f.
$$

Equation (29) then gives

$$
\begin{aligned}
 p_h\big\vert_K
 &=p_K^0+T_hG_h\rho+
       (I-T_hG_h)\widetilde T_h f,\\
 \lambda_h\big\vert_{\partial K}
 &=\lambda_K^0+G_h\rho-G_h\widetilde T_h f.
\end{aligned}
$$

The source term in this reconstruction is retained. The reported pressure and
raw flux use the full reconstructed $p_h$, including this source contribution.

## Discrete inverse and stability contract

Let $R_K$ be the stiffness inverse on the boundary-mean-zero complement,
$B_K$ the local unsigned conormal coupling and $D_K$ the pairing of the two
face spaces. A deterministic basis $Z_K$ spans the zero-integral conormal
coordinates. The matrices are

$$
\begin{aligned}
 N_K&=(B_KZ_K)^T R_K(B_KZ_K),& M_K&=Z_K^T D_K,\\
 S_K&=M_K^T N_K^{-1}M_K,& \eta_K&=R_Kf_K,\\
 b_K&=-D_K^T\lambda_K^0
       +M_K^TN_K^{-1}(B_KZ_K)^T\eta_K.
\end{aligned}
$$

The assembled pressure-trace matrix is symmetric positive definite after
Dirichlet elimination when the discrete stability conditions hold. A pure
Neumann problem instead has its physical constant mode, fixed by the prescribed
**volume mean of the full pressure**. The compatibility check uses
$\int_\Omega f=\int_{\partial\Omega}q\cdot n$. Natural data are outward
physical fluxes; their contribution to the pressure-trace right-hand side is
$-\langle q\cdot n,\xi\rangle$.

Section 5 of the article states the two discrete stability assumptions.
Section 6 gives sufficient mesh conditions: each conormal segment contains
at least two local boundary edges (M1), and each pressure-trace segment
contains at least two conormal segments (M2), for the corresponding polynomial
families. Remark 16 permits particular configurations outside these sufficient
conditions, including the smooth study below. The implementation rejects a
rank-deficient local Neumann map and a rank-deficient global system. This
numerical check does not replace a mesh-uniform inf-sup estimate. In particular,
a choice admissible with Dirichlet conditions can have additional modes under
pure Neumann conditions; enrichment of $\Lambda$ must then resolve them.

`conservation_residuals()` checks each macroelement's integrated flux balance.
`trace_moment_residuals()` checks the pressure matching tested by $\Lambda$.
The global equation gives flux continuity tested by $\Gamma$. These weak
conditions do not assert pointwise normal continuity, an $H(\mathrm{div})$
raw gradient, or conservation on every fine triangle.

## Smooth case from Section 8.1

On the unit square, with $A=I$ and homogeneous Dirichlet conditions,

$$
\begin{aligned}
 p(x,y)&=x(x-1)y(y-1),\\
 f(x,y)&=-2x(x-1)-2y(y-1),\\
 \lVert p\rVert_{L^2(\Omega)}&=\frac1{30},&
 \lVert\nabla p\rVert_{L^2(\Omega)}&=\frac1{\sqrt{45}}.
\end{aligned}
$$

The three families are $\Gamma=P_{k+1}$, $\Lambda=P_k$ and local
$V_h=P_{k+1}$ for $k=0,1,2$. Uniform square grids split along the southwest–northeast
diagonal have $n=2,4,8,16,32$ and macro diameter $H=\sqrt2/n$.
The local size is $h=H$ for $k=0,2$ and $h=H/2$ for $k=1$, matching the
space and size choices associated with Figure 2. The article does not provide
the original mesh connectivity or a numerical table for that figure; the
comparison verifies the stated rates rather than a digitized equality of curves.
Assembly uses order 6 and independent error integration uses order 8.

| $n$ | Relative gradient error, $k=0$ | $k=1$ | $k=2$ |
|---:|---:|---:|---:|
| 2 | 7.15345e−1 | 1.64636e−1 | 3.28378e−2 |
| 4 | 3.94289e−1 | 4.33882e−2 | 3.98319e−3 |
| 8 | 2.02327e−1 | 1.09753e−2 | 4.88523e−4 |
| 16 | 1.01836e−1 | 2.75159e−3 | 6.04203e−5 |
| 32 | 5.10027e−2 | 6.88376e−4 | 7.51030e−6 |

The measured gradient orders on the last three refinement intervals approach
the order $k+1$ reported in the article's Figure 2. The pressure $L^2$ errors
additionally approach order $k+2$; that pressure rate is a measured result
here, rather than a separate rate stated in the publication:

| Family | Error norm | Published order | $4\to8$ | $8\to16$ | $16\to32$ |
|---|---|---:|---:|---:|---:|
| $k=0$ | Pressure $L^2$ | — | 1.919 | 1.979 | 1.995 |
| $k=0$ | Broken gradient $L^2$ | 1 | 0.963 | 0.990 | 0.998 |
| $k=1$ | Pressure $L^2$ | — | 3.040 | 3.014 | 3.004 |
| $k=1$ | Broken gradient $L^2$ | 2 | 1.983 | 1.996 | 1.999 |
| $k=2$ | Pressure $L^2$ | — | 4.071 | 4.040 | 4.021 |
| $k=2$ | Broken gradient $L^2$ | 3 | 3.027 | 3.015 | 3.008 |

Orders are computed from consecutive error ratios and macro diameters;
the [numerical record](../figures/mh2m/convergence-verification.json) retains
all intervals and the input digest. These rates concern this smooth exact
solution and the stated spaces. They do not establish the same orders for
the oscillatory material study. The pressure errors at $n=32$, relative to
$\lVert p\rVert_{L^2}$, are 2.75169e−3, 9.47498e−6 and 9.18941e−8.

![Smooth relative gradient and pressure errors](../figures/mh2m/convergence.png)

The maps below use $n=8$. Exact and numerical pressures share a scale; each
signed-error panel has its own symmetric scale. Every panel retains the actual
macro mesh, and the numerical samples preserve separate incident values.

![Exact, MH2M and signed-error pressures for three degrees](../figures/mh2m/smooth-fields.png)

## Oscillatory material and independent interface resolutions

The coefficient follows equation (61), with the explicitly selected $\gamma=1$:

$$
\begin{aligned}
 A(x,y)&=a(x,y)I, &\varepsilon&=1/14,\\
 a(x,y)&=\frac{2+\sin(2\pi x/\varepsilon)}
                  {2+\cos(2\pi y/\varepsilon)}
       +\frac{2+\sin(2\pi y/\varepsilon)}
                  {2+\sin(2\pi x/\varepsilon)}.
\end{aligned}
$$

This experiment retains the smooth case's source and homogeneous Dirichlet data.
Section 8.2 does not specify the numerical value of $\gamma$ or restate the
source. Consequently this is a declared-data test of the published coefficient
family and the independent interface spaces, not a numerical reproduction of
Figures 5–8.

The separate [oscillatory-material study](mh2m-heterogeneous.md) uses
$\gamma=1.8$ and the source specified by [de Barros (2022)](https://www.lncc.br/~alm/students/frankdissert.pdf), retaining the
article's $\varepsilon=1/14$. It checks a longer classical-reference series,
includes Figure 5's stated mesh parameters, and tests the local injectivity
requirement when the conormal partition is refined independently. Its data
convention is distinct from the $\gamma=1$ study below.

The classical baseline is a separately assembled conforming P1 solution on
$64^2$, $128^2$ and $256^2$ squares, each split into two triangles. It shares
the package's finite-element kernels and is not an independent reference code.
Its own consecutive refinement differences are:

| Refinement | Pressure difference / finer pressure norm | Flux difference / finer flux norm |
|---|---:|---:|
| 64 → 128 | 0.660744% | 7.82272% |
| 128 → 256 | 0.187542% | 4.16967% |

All comparisons integrate $p_h-p_{\mathrm{ref}}$ and the physical raw flux
$-A\nabla p_h+A\nabla p_{\mathrm{ref}}$ over a common nested fine partition,
using the corresponding finest-reference $L^2$ norm as denominator.
The material remains inside the flux norm. The last refinement difference is
a resolution indicator, not a rigorous reference-error bound.

First, fix 32 macrotriangles, a single P1 pressure-trace segment per face
(nine free global unknowns), and a P1 local mesh with 32 subdivisions per
macroedge. Only the P0 conormal partition changes:

| $\Lambda$ segments / macroface | Free global unknowns | Relative pressure difference | Relative flux difference |
|---:|---:|---:|---:|
| 1 | 9 | 12.5803% | 35.6390% |
| 2 | 9 | 13.2779% | 36.5699% |
| 4 | 9 | 14.3710% | 37.9219% |
| 8 | 9 | 15.0269% | 37.9736% |
| 16 | 9 | 15.3626% | 38.0548% |

This series confirms the independence of the global dimension, but **does not
show monotone improvement from conormal enrichment**. The coarse pressure trace
remains unresolved. Increasing $\Lambda$ changes the local constraints and is
not an error-minimizing nested global trial-space enrichment.

![Fixed-Gamma conormal enrichment](../figures/mh2m/enrichment.png)

The separate pressure-trace refinement keeps the 16 conormal segments and the
same local mesh. Its archived series distinguishes pressure-trace resolution
from changes to the local pressure space or the permeability quadrature.

| $\Gamma$ segments / macroface | Free global unknowns | Relative pressure difference | Relative flux difference |
|---:|---:|---:|---:|
| 1 | 9 | 15.3626% | 38.0548% |
| 2 | 49 | 2.17445% | 14.0337% |
| 4 | 129 | 1.09511% | 10.1094% |
| 8 | 289 | 0.451260% | 6.47142% |
| 16 | 609 | 0.176569% | 4.33429% |

The finest flux difference is close to the 4.16967% change of the last classical
refinement. The fixed local P1 resolution is also coarser than the finest
reference. These results establish a decreasing measured difference in this
series, without identifying its value with an exact-solution error.

![Pressure-trace refinement at fixed local spaces](../figures/mh2m/pressure-trace-enrichment.png)

The pressure maps compare the classical reference, one and sixteen pressure-trace
segments, and the signed difference for the latter. Both MH²M fields use sixteen
conormal segments. Flux maps show signed components for the finest pressure trace;
they display raw physical fluxes, without interface averaging or an $H(\mathrm{div})$
postprocessing step.

![Oscillatory pressure and signed difference](../figures/mh2m/oscillatory-pressure.png)

![Signed physical flux components and differences](../figures/mh2m/oscillatory-flux.png)

## Nonconvex polygons and physical boundary data

The [shared analytical boundary campaign](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mh.md#nonhomogeneous-boundaries-on-nonconvex-polygons)
uses two complementary L polygons per grid rectangle, with
$n=1,2,4,8,16$ and $2n^2$ macroelements. The anisotropic tensor has
off-diagonal entries; the exact nonpolynomial pressure provides nonzero
Dirichlet values and physical Neumann fluxes. Every local space is continuous
P2 with four edge subdivisions; $\Gamma$ is continuous P2 on one face segment
and $\Lambda$ is broken P1 on two segments. Thus each pressure-trace segment
contains two conormal segments, and each conormal segment contains two local
boundary edges. The rank and physical conservation checks remain necessary
for the resulting polygonal assembly.

At 512 macroelements, the relative pressure errors are
$3.13188\times10^{-7}$, $3.09876\times10^{-7}$ and
$3.06707\times10^{-7}$ for Dirichlet, mixed and pure Neumann conditions,
respectively. The corresponding physical-flux errors are
$1.14649\times10^{-4}$, $1.14199\times10^{-4}$ and
$1.13727\times10^{-4}$. Final pressure rates are 2.970–3.000 and flux rates
1.987–1.999. Relative errors use exact physical $L^2$ norms; the Neumann gauge
uses the volume mean of the complete pressure.

![Polygonal boundary convergence for MH and MH²M](../figures/mh/boundary-convergence.png)

These exact-solution results verify nonconvex geometry, independent interface
spaces and boundary conditions. They do not reproduce a table or a
heterogeneous-material figure from the article. The
[complete record](../figures/mh/boundary-comparison.json) contains all 25
MH/MH²M runs and their source hashes; notebook `65_mh_boundary.ipynb` executes
a small pure-Neumann patch.

## Verification and rerunning

Light tests compare the condensed solution against an independently assembled
uncondensed three-field system. They cover anisotropic quadratic patches,
nonzero Dirichlet and Neumann data, incompatible loads at small physical scales,
the pressure gauge, independently partitioned faces, and rank-deficient choices.
Native DOLFINx tests independently assemble the variable-tensor stiffness,
source and boundary-mean functional for P1–P3, then compare the constrained
Neumann source lift, including a nonconvex polygon. The smooth campaign checks
the original local equations,
pressure-trace moments and macro conservation in addition to error norms.

Run the separate numerical campaigns and render their archived fields:

```bash
pixi run -e notebooks python -m examples.mh2m_campaign
pixi run -e notebooks python -m examples.mh2m_trace_campaign
pixi run -e notebooks python -m examples.plot_mh2m
pixi run --locked -e test-core pytest -q tests/test_mh2m.py tests/test_mh2m_campaign.py
pixi run -e fem pytest -q tests/test_mh2m_fenics.py
```

The [smooth/reference/conormal record](../figures/mh2m/comparison.json) and
[pressure-trace refinement record](../figures/mh2m/pressure-enrichment.json)
contain dimensions, quadrature conventions, original-equation diagnostics and
executed source hashes. The source acquisition is separate from rendering.
Notebook `52_mh2m.ipynb` executes a small solve and displays these archived studies.
The polygonal driver and the published-case studies on this page are
two-dimensional. Their verification does not establish uniform stability for
arbitrary independent face partitions. The tetrahedral implementation and its
separate dimensional verification are described on the [MH/MH²M 3D page](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mh3d.md).

## References

- Franklin de Barros, Alexandre L. Madureira, and Frédéric Valentin (2026). *A three-field Multiscale Method*. arXiv preprint, version 3, 5 August 2026; first submitted 25 April 2024. [arXiv: 2404.16978v3](https://arxiv.org/abs/2404.16978v3).

- Franklin da Conceição de Barros (2022). *The Multiscale Hybrid-Hybrid-Mixed Method*. Master’s dissertation in Computational Modeling, Laboratório Nacional de Computação Científica, Petrópolis, Brazil, 77 pages. [Institutional dissertation](https://www.lncc.br/~alm/students/frankdissert.pdf).
