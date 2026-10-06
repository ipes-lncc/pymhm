# Multiscale Hybrid diffusion with Robin local problems

`solve_mh` implements the Multiscale Hybrid method of
[Barrenechea, Gomes and Paredes (2024)](https://doi.org/10.1137/22M1542556). Its local problems and condensed
global Dirichlet problem are elliptic. This formulation differs from both the
kernel-constrained MHM and the [three-field MH²M method](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mh2m.md).

The current implementation uses triangular or simple polygonal macroelements
in two dimensions, including nonconvex polygons, continuous local Lagrange
elements, and independently partitioned polynomial skeleton spaces. Scalar
and symmetric positive-definite tensor permeability are supported. Exterior
faces prescribe pressure weakly or the outward physical flux. Compatible pure
Neumann data use a prescribed volume-mean pressure. The physical Neumann
extension below retains the local Robin operators; its augmented global
system is symmetric indefinite.

The separately verified [tetrahedral driver](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mh3d.md), `solve_mh_3d`, uses
the dimensionally consistent Robin field $\sigma=\nu(x-a)/3$ and the same
physical boundary equations. The numerical studies on this page are
two-dimensional.

Dirichlet moments are integrated on the union of the incident fine-edge
partition and skeletal breakpoints. This integration partition resolves data
that are piecewise polynomial on the local boundary without enriching the
skeletal approximation. A callback with additional discontinuities requires
the integration partition to resolve those discontinuities.

## Operator, multiplier and conservation

For $q=-K\nabla p$, define the vector field

$$
 \sigma(x)=\frac{\nu}{2}(x-a),\qquad \nabla\cdot\sigma=\nu>0.
$$

The multiplier and local bilinear form are

$$
\begin{aligned}
 \lambda_K&=(q-p\sigma)\cdot n_K,\\
 a_K(u,v)&=\int_K K\nabla u\cdot\nabla v
          +\int_{\partial K}(\sigma\cdot n_K)uv,\\
 &=\int_K K\nabla u\cdot\nabla v
   +\int_K (\sigma\cdot\nabla u)v
   +\int_K u(\sigma\cdot\nabla v)+\nu\int_K uv.
\end{aligned}
$$

The boundary term may have either sign on individual faces. Coercivity is a
property of the complete local operator. The discrete equations are

$$
\begin{aligned}
 A_Kp_K+B_K\lambda&=f_K,\\
 S&=\sum_K B_K^T A_K^{-1}B_K,\\
 S\lambda&=\sum_K B_K^T A_K^{-1}f_K-g_D,\\
 p_K&=A_K^{-1}f_K-A_K^{-1}B_K\lambda.
\end{aligned}
$$

Here $g_D$ is the weak Dirichlet functional. Local inverses have no mean-zero
constraint and no retained constant kernel. The global matrix is symmetric
positive definite when the skeleton coupling is injective and the local
operators are coercive, for the Dirichlet problem. Global unknowns include
exterior multiplier moments.

The Robin multiplier is **not** the physical Darcy normal flux.
`normal_flux(cell, face, parameter)` evaluates the one-sided outward quantity
$\lambda_K+p_K\sigma\cdot n_K$. Testing the local equation with one gives

$$
 \int_{\partial K}(\lambda_K+p_K\sigma\cdot n_K)=\int_K f.
$$

This is the macro balance checked by `conservation_residuals()`. The raw
volume flux is $-K\nabla p_K$. Neither pointwise continuity of that raw flux
nor fine-cell conservation follows from this macro identity. Pressure traces
remain broken; the Robin correction is evaluated separately on each incident
macroelement. Passing the multiplier directly to a Darcy normal-flux
reconstruction would change the mathematical data.

## Physical Neumann and mixed boundaries

`neumann` maps exterior face indices to prescribed $h_N=q\cdot n$.
`dirichlet` applies to the other exterior faces. Prescribing $\lambda=h_N$
would impose a different boundary condition because of the Robin pressure
term.

On a straight exterior face, $r=\sigma\cdot n$ is constant. Introduce a
pressure trace $\rho$ in that face's multiplier space, with mass matrix $M$.
The discrete boundary conditions are

$$
\begin{aligned}
 B_N^Tp&=M\rho,\\
 M\lambda_N+rM\rho&=h_N.
\end{aligned}
$$

The last $h_N$ denotes the integrated functional against the face basis.
Writing $J$ for the scattering of the Neumann face mass matrices and
$R=\operatorname{diag}_F(r_FM_F)$ gives

$$
\begin{bmatrix}
 S&J\\ J^T&R
\end{bmatrix}
\begin{bmatrix}\lambda\\ \rho\end{bmatrix}
=
\begin{bmatrix}
 \sum_KB_K^TA_K^{-1}f_K-g_D\\ h_N
\end{bmatrix}.
$$

This formulation also covers faces with $r=0$; it never divides by the Robin
coefficient. Local reconstruction is unchanged. `boundary_pressure` exposes
the face coefficients of $\rho$; `global_matrix`, `global_rhs` and
`global_coefficients` expose these boundary-extended equations, before adding
a pure-Neumann gauge. Coefficients are ordered as the Robin multiplier followed
by pressure coefficients on the Neumann faces in increasing face-index order.

For pure Neumann data, the source and outward-flux integrals must satisfy

$$
 \int_\Omega f=\int_{\partial\Omega}h_N,
 \qquad
 \frac1{\lvert\Omega\rvert}\int_\Omega p=\texttt{mean\_pressure}.
$$

The gauge uses the full reconstructed pressure, including its source lift.
Incompatible data are rejected at their physical scale. A nonzero
`mean_pressure` is rejected when any Dirichlet face remains. This is a
consistent boundary extension of the article's Dirichlet formulation; the
article's positive-definiteness result for its global Dirichlet operator is
not asserted for this indefinite system.

## Coercivity and compatible spaces

Let $K_{\min}>0$ be a certified lower eigenvalue bound and
$C_\sigma=\max_{x\in\Omega}\lvert x-a\rvert/2$. The implementation requires

$$
 0<\nu\le\frac{K_{\min}}{4C_\sigma^2}.
$$

Indeed, Young's inequality gives

$$
\begin{aligned}
 a_K(v,v)&\ge\frac{K_{\min}}2\|\nabla v\|_K^2
 +\left(\nu-\frac{2\nu^2C_\sigma^2}{K_{\min}}\right)\|v\|_K^2\\
 &\ge\frac{K_{\min}}2\|\nabla v\|_K^2+\frac\nu2\|v\|_K^2.
\end{aligned}
$$

The bound uses the actual norm of the prescribed vector field; the paper's
equation (2.14) uses a more conservative bounding-domain diameter. This
changes the sufficient parameter bound, not the Robin bilinear form.
The default origin is the lower corner of the bounding box, and the default
parameter is half the permitted upper bound. For the unit square and $K=I$,
these choices give $\nu=1/4$, as in the article's numerical experiments.

Constant and Cartesian materials provide their eigenvalue bound directly.
Other material callbacks require `ellipticity_lower_bound`; point samples
can reject an inconsistent declaration but do not certify it. Local and
skeletal meshes must satisfy the article's discrete trace compatibility
conditions. The studies below use $k=\ell+2$ and two local boundary edges per
skeletal edge. An algebraic rank check does not establish a uniform inf-sup
constant for arbitrary user-selected spaces.

At $\nu=0$, the local Robin inverse is undefined. `solve_darcy` supplies the
kernel-constrained MHM limit. Decreasing $\nu$ can preserve approximation
accuracy while worsening both local and global conditioning.

## Published analytical problem on declared meshes

Section 4.1 uses the unit square, homogeneous Dirichlet conditions, $K=I$ and

$$
\begin{aligned}
 p(x,y)&=\sin(6\pi x)\sin(14\pi y),\\
 f(x,y)&=232\pi^2 p(x,y),\qquad q=-\nabla p,\\
 \|p\|_{L^2(\Omega)}&=\frac12,\qquad
 \|q\|_{L^2(\Omega)}=\pi\sqrt{58}.
\end{aligned}
$$

The campaign uses trace/local degree pairs $(\ell,k)=(1,3)$ and $(2,4)$,
$\nu=1/4$, and five resolutions for each mesh family. Triangular macro meshes
have $n=4,8,16,32,64$ square subdivisions, split along one diagonal. The
nonconvex family tiles $n\times n$ rectangles with two complementary L-shaped
polygons, for $n=2,4,8,16,32$. Collinear polygon vertices retain matching
macroface partitions. The respective macro diameters are $\sqrt2/n$ and
$\sqrt{13}/(3n)$.

Local ear triangulations are refined by two in each edge; assembly and error
quadratures have orders eight and ten. The article instead fixes a much finer
local resolution in its convergence figures. These results verify its
analytical equation and degree pairs on explicitly defined meshes; they do
not reproduce the historical mesh connectivity or plotted numerical values.

The campaign explicitly selects extended accumulation for local and global
iterative refinement, retaining double-precision factors and the shared
residual thresholds. It requires a platform whose NumPy `longdouble` has more
precision than `float64`. The portable solver's default remains double
precision; the precision options neither change the equations nor guarantee
that an arbitrarily ill-conditioned system satisfies its residual criterion.

The final relative errors and rates between the last two resolutions are:

| Macro mesh | Trace/local degrees | Pressure relative error | Flux relative error | Pressure rate | Flux rate |
|---|---:|---:|---:|---:|---:|
| Triangles | P1/P3 | $1.296\times10^{-3}$ | $1.004\times10^{-2}$ | 2.767 | 1.893 |
| Triangles | P2/P4 | $4.739\times10^{-5}$ | $5.748\times10^{-4}$ | 3.941 | 2.957 |
| L polygons | P1/P3 | $1.304\times10^{-4}$ | $1.836\times10^{-3}$ | 3.008 | 2.010 |
| L polygons | P2/P4 | $2.250\times10^{-6}$ | $4.737\times10^{-5}$ | 3.982 | 2.997 |

The maximum absolute macro balance over these twenty cases is
$3.92\times10^{-13}$. The triangular P1/P3 series has not reached the
same final rate as its polygonal counterpart; both curves retain all five
measured resolutions.

![Five-level convergence of physical flux and pressure](../figures/mh/convergence.png)

The field panels preserve independent values at fine and macro interfaces.
Exact and numerical fields share each row's signed scale; the error has its
own scale. Every panel includes the actual macro boundaries.

![Triangular MH pressure and signed flux components](../figures/mh/triangles-fields.png)

![Nonconvex polygonal MH pressure and signed flux components](../figures/mh/L-polygons-fields.png)

## Same-space limit to MHM

The comparison uses a fixed triangular macro mesh with $n=4$, trace P1,
local P3 and refinement four. MH and MHM use identical fine partitions,
source quadrature and boundary data. The measured norm is

$$
 \left(\sum_K\int_K\lvert\nabla(p_{\rm MH}-p_{\rm MHM})\rvert^2\right)^{1/2}.
$$

It is a physical broken energy difference for $K=I$, not a coefficient norm
or a comparison with an exact solution. The five values of $\nu$ halve from
$1/4$ to $1/64$. The right panel reports spectral condition numbers of the
condensed matrix and the first local Robin matrix in their declared bases.
Condition numbers change under basis scaling and are not timing measurements.

The energy difference decreases from $0.04125565$ to $0.002583777$,
approximately by a factor of two per halving of $\nu$. Over the same sequence,
the global spectral condition number increases from $2.82\times10^4$ to
$4.30\times10^5$. Spectral diagnostics use the double-precision representation
of the assembled matrices; solution and residual verification retain the
selected extended accumulation.

![Measured MH-to-MHM limit and spectral conditioning](../figures/mh/vanishing-robin.png)

## Verification and reproduction

### Nonhomogeneous boundaries on nonconvex polygons

The boundary-extension campaign uses the unit square and the original exact data

$$
\begin{aligned}
 K&=\begin{bmatrix}3&0.4\\0.4&2\end{bmatrix},\\
 p&=1+x+2y+\sin(\pi x)\sin(\pi y),\\
 q&=-K\nabla p,\\
 f&=\pi^2\left[5\sin(\pi x)\sin(\pi y)
                  -0.8\cos(\pi x)\cos(\pi y)\right].
\end{aligned}
$$

For mixed conditions, the right side prescribes $q\cdot n$ and the other
sides prescribe $p$. Pure Neumann conditions use the exact volume mean
$5/2+4/\pi^2$. Each of $n^2$ grid rectangles contains two complementary L
polygons, with $n=1,2,4,8,16$. Local P2 ear triangulations have four
subdivisions per edge. MH uses P2 Robin multipliers on two face segments and
$\nu=1/4$. The MH²M control uses continuous P2 pressure traces on one segment
and broken P1 conormals on two segments. These are distinct discrete methods,
not a same-space comparison.

The relative errors divide by the analytical physical norms

$$
\begin{aligned}
 \|p\|_{L^2}^2&=\frac{20}{3}+\frac14+\frac{20}{\pi^2},\\
 \|q\|_{L^2}^2&=33.8+3.33\pi^2.
\end{aligned}
$$

The final level has 512 macroelements. Rates use the last two resolutions.

| Method and boundary data | Pressure relative error | Flux relative error | Pressure rate | Flux rate |
|---|---:|---:|---:|---:|
| MH, mixed | 6.17004e−8 | 6.06408e−5 | 3.000 | 2.000 |
| MH, pure Neumann | 6.20011e−8 | 6.06031e−5 | 3.007 | 1.999 |
| MH²M, Dirichlet | 3.13188e−7 | 1.14649e−4 | 3.000 | 1.999 |
| MH²M, mixed | 3.09876e−7 | 1.14199e−4 | 2.985 | 1.993 |
| MH²M, pure Neumann | 3.06707e−7 | 1.13727e−4 | 2.970 | 1.987 |

Across all 25 cases, the maximum absolute macro balance is
$2.05\times10^{-12}$. Changing independent error quadrature from order eight
to ten changes the absolute norms by at most $2.50\times10^{-15}$. Assembly
uses order eight. MH explicitly uses extended refinement accumulation with
the shared residual thresholds and double-precision factors; MH²M uses double
precision. This is an exact-solution verification of the boundary and geometry
extensions, not a reproduction of a published numerical table.

![Physical boundary and polygonal convergence](../figures/mh/boundary-convergence.png)

The [boundary record](../figures/mh/boundary-comparison.json) preserves all five
levels, physical norms, dimensions, source hashes and original-equation
diagnostics. Notebook `65_mh_boundary.ipynb` runs a small compatible
pure-Neumann patch and displays the archived campaign.

Lightweight tests cover nonzero Dirichlet data, anisotropic nonharmonic
quadratic patches on triangles and a nonconvex polygon, the uncondensed
saddle equations, physical normal-flux orientation, coercivity, and serial,
thread and spawn-process agreement. Independent DOLFINx/UFL assembly checks
variable-coefficient local matrices, loads and source responses with P1 and
P3. The multilevel studies run separately from these tests.

The Neumann tests additionally compare all blocks of a full uncondensed
mixed-boundary and pure-Neumann system against independent UFL assembly,
including the auxiliary boundary pressure and physical volume gauge.

```bash
pixi run -e notebooks python -m examples.mh_campaign --workers 8
pixi run -e notebooks python -m examples.plot_mh
pixi run -e notebooks python -m examples.mh_boundary_campaign
pixi run -e notebooks python -m examples.plot_mh_boundary
```

The [numerical record](../figures/mh/comparison.json) includes dimensions,
absolute and relative norms, balances, residuals and executed source hashes.
The repository notebook `notebooks/darcy/56_mh.ipynb` combines a small executable patch
with the archived multilevel evidence. The portable solver does not require
FEniCS; DOLFINx is an optional independent verification backend.

## References

- Gabriel R. Barrenechea, Antonio Tadeu A. Gomes, and Diego Paredes (2024). *A Multiscale Hybrid Method*. SIAM Journal on Scientific Computing 46(3), A1628–A1657. [DOI: 10.1137/22M1542556](https://doi.org/10.1137/22M1542556).
