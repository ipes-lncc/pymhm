# Tetrahedral MH and three-field MH²M

`solve_mh_3d` and `solve_mh2m_3d` assemble local volume and triangular-face
operators on tetrahedral macroelements. Both accept nonhomogeneous pressure,
physical outward flux, mixed boundaries and compatible pure Neumann data.
The local pressure degree, fine tetrahedral refinement and interface spaces
are explicit inputs.

The [Barrenechea, Gomes and Paredes (2024)](https://doi.org/10.1137/22M1542556) presents its analysis and
numerical examples in two dimensions and indicates an extension to three
dimensions. The Robin formulation here is that dimensional extension, with
its coercivity bound derived below. The
[de Barros, Madureira and Valentin (2026)](https://arxiv.org/abs/2404.16978v3) states the model and local-map
construction for dimensions two and three; its sufficient polynomial-space
constructions and numerical examples are two-dimensional. The computations
on this page are original three-dimensional verification cases, not numerical
tables reproduced from those articles.

## Geometry and interface spaces

`TetraMesh` supplies affine tetrahedra, physical normals and local tetrahedral
partitions. The broken multiplier uses `TriangularSkeleton`: each macroface
has its own polynomial degree and conforming triangular subdivision. Its
Bernstein coefficients represent a scalar density with respect to the
physical unit normal, not integrated fluxes.

For MH²M, `PressureTraceSpace3D` constructs continuous nodal $P_k$ functions
on uniformly subdivided triangular macrofaces. Shared macroedge and vertex
nodes have the same global index, determined by integer barycentric weights.
The pressure trace is therefore continuous across the complete skeleton.
Its degree and subdivision are independent of the broken conormal space.
The local boundary triangulation must resolve both partitions.

This driver supports tetrahedral macroelements. The
[two-dimensional polygonal extension](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mh.md#physical-neumann-and-mixed-boundaries)
does not imply an implementation on general three-dimensional polyhedra.
No mesh-uniform stability claim is made for arbitrary degree/subdivision
pairs. Local inverse and global rank checks reject insufficient spaces, but
do not replace the approximation and inf-sup assumptions of the papers.

## Robin MH in three dimensions

For $q=-K\nabla p$, use

$$
\begin{aligned}
 \sigma(x)&=\frac{\nu}{3}(x-a),&
 \nabla\cdot\sigma&=\nu,\\
 \lambda_K&=(q-p\sigma)\cdot n_K,\\
 a_K(u,v)&=(K\nabla u,\nabla v)_K
             +\langle\sigma\cdot n_K,u v\rangle_{\partial K}.
\end{aligned}
$$

If $K_{\min}$ is a certified lower eigenvalue bound and
$C_\sigma=\max_{x\in\Omega}\lvert x-a\rvert/3$, then

$$
\begin{aligned}
0<\nu&\le\frac{K_{\min}}{4C_\sigma^2},\\
a_K(v,v)&\ge
 \frac{K_{\min}}2\lVert\nabla v\rVert_K^2+
 \frac{\nu}{2}\lVert v\rVert_K^2.
\end{aligned}
$$

This follows by expanding the boundary term with the divergence theorem and
applying Young's inequality to $2(\sigma v,\nabla v)_K$. No zero-mean local
constraint is required. The local reconstruction and condensed Dirichlet
operator use the [same Robin algebra](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mh.md#operator-multiplier-and-conservation)
as in two dimensions.

On each exterior Neumann face, the auxiliary pressure and mass matrix impose
$M\lambda+rM\rho=h_N$, where $r=\sigma\cdot n$ and $h_N$ is the integrated
physical outward-flux functional. Faces with $r=0$ remain admissible.
The shared boundary operator is symmetric indefinite. Pure Neumann data
satisfy $\int_\Omega f=\int_{\partial\Omega}h_N$, and `mean_pressure`
fixes the volume mean of the full reconstructed pressure.

`normal_flux_moments(cell, face)` returns moments of
$\lambda_K+p_K\sigma\cdot n_K$. These moments satisfy macro balance.
The raw field $-K\nabla p_K$ is distinct from the multiplier and need not be
$H(\mathrm{div})$-conforming or fine-cell conservative.

## Three-field MH²M

The local conormal has the opposite sign from the physical outward flux:

$$
 \lambda_K=K\nabla p_K\cdot n_K=-q\cdot n_K.
$$

Local Neumann maps use a **boundary-mean-zero** pressure complement.
The constant conormal is
$\lambda_K^0=-\int_K f/\lvert\partial K\rvert$. The implementation reuses the
same dimension-independent map algebra as the two-dimensional solver, after
assembling all volume, conormal and continuous-pressure pairings on physical
tetrahedra and triangular facets.

The global unknown is $\rho\in\Gamma$, and the complete reconstruction is

$$
\begin{aligned}
p_h\big\vert_K
 &=\frac{1}{\lvert\partial K\rvert}\int_{\partial K}\rho
   +T_hG_h\rho+(I-T_hG_h)\widetilde T_h f,\\
\lambda_h\big\vert_{\partial K}
 &=\lambda_K^0+G_h\rho-G_h\widetilde T_h f.
\end{aligned}
$$

Physical Neumann data contribute $-\langle h_N,\xi\rangle$ to the global
pressure-trace equations. Dirichlet data are interpolated at the pressure
trace nodes. A pure Neumann gauge fixes the full volume mean, including the
source contribution. The original trace equations are checked after applying
that gauge.

The diagnostics distinguish macro flux balance, pressure matching tested by
$\Lambda$, and nodal volume equations. Global flux continuity is tested by
$\Gamma$; it does not assert pointwise continuity of the raw gradient flux.

## Independent verification

Small portable checks use a full anisotropic constant tensor and the
quadratic pressure $p=1+x^2+y^2+z^2$. Dirichlet, mixed and pure Neumann
problems reproduce pressure, physical flux and the prescribed volume mean.
They also verify conormal moments, source balance and the uncondensed equations.

Three native DOLFINx/UFL checks assemble the complete volume/interface
systems independently on an oblique tetrahedron with variable
$K(x)=(2+x)I$. Two checks cover the MH mixed/pure-Neumann boundary equations;
the third covers all three MH²M fields and its pressure gauge. Their matrices
include independently integrated face masses, physical Neumann loads and
volume moments. These checks compare coefficient vectors before any plotting
or numerical norm reduction.

## Smooth nonaffine refinement

The unit cube has $6n^3$ tetrahedral macroelements. The exact data are

$$
\begin{aligned}
p(x,y,z)&=1+x+2y+3z+\sin(\pi x)\sin(\pi y)\sin(\pi z),\\
K&=I,\qquad q=-\nabla p,\\
f(x,y,z)&=3\pi^2\sin(\pi x)\sin(\pi y)\sin(\pi z).
\end{aligned}
$$

Nonhomogeneous pressure is prescribed on every exterior face. Relative errors
use exact physical denominators:

$$
\begin{aligned}
\lVert p\rVert_{L^2(\Omega)}^2
 &=\frac{103}{6}+\frac18+\frac{64}{\pi^3},\\
\lVert q\rVert_{L^2(\Omega)}^2
 &=14+\frac{3\pi^2}{8}.
\end{aligned}
$$

The MH series uses local $P_3$, two fine subdivisions per macroedge,
an undivided $P_1$ Robin multiplier and $\nu=1/4$. The MH²M series uses local
$P_2$ with four subdivisions per macroedge, continuous $\Gamma=P_2$ on
undivided macrofaces and broken $\Lambda=P_1$ on four subtriangles per
macroface. Thus neither local nor global dimensions are artificially equated
between the methods. Both use assembly order 6 and independent error orders
6 and 8.

| Method | n | Macro tetrahedra | Globally solved unknowns | Relative pressure L2 | Relative flux L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MH | 1 | 6 | 54 | 2.75145% | 21.2522% |
| MH | 2 | 48 | 360 | 0.217256% | 4.06973% |
| MH | 3 | 162 | 1134 | 0.0620152% | 1.83186% |
| MH | 4 | 384 | 2592 | 0.0257875% | 1.03576% |
| MH | 5 | 750 | 4950 | 0.0131498% | 0.665219% |
| MH²M | 1 | 6 | 1 | 2.40841% | 22.6116% |
| MH²M | 2 | 48 | 27 | 0.889328% | 12.6659% |
| MH²M | 3 | 162 | 125 | 0.276020% | 6.36592% |
| MH²M | 4 | 384 | 343 | 0.117706% | 3.80846% |
| MH²M | 5 | 750 | 729 | 0.0605284% | 2.52552% |

MH counts the Robin unknowns; MH²M counts the free pressure-trace unknowns
after imposing Dirichlet data. These distinct counts do not measure the local
work or establish a performance comparison. Between the last two levels,
pressure/flux rates are 3.018/1.984 for MH and 2.980/1.841 for MH²M.
The fields converge toward the exact solution with the stated fixed spaces;
these observations are not a stability proof for other space combinations.

Changing the error quadrature from order 6 to 8 changes the coarsest MH norm
by at most $5.83\times10^{-6}$ in absolute value; its other levels change by at
most $5.07\times10^{-10}$. The MH²M maximum is $8.44\times10^{-9}$, also at the
coarsest level. The table uses order 8. This comparison concerns norm
integration; assembly remains at order 6. Maximum integrated macro-balance
defects are $6.05\times10^{-14}$ for MH and $5.33\times10^{-15}$ for MH²M.
All ten acquisitions retain unchanged executed numerical-source hashes.

The field figures retain independent one-sided samples on fine-cell cuts.
The numerical and analytical panels share a scale; differences have their
own signed scale. All panels show the actual macroface intersections.

![Tetrahedral pressure and physical-flux convergence](../figures/mh3d/convergence.png)

![Robin MH pressure and signed flux component](../figures/mh3d/mh-fields.png)

![MH²M pressure and signed flux component](../figures/mh3d/mh2m-fields.png)

## Reproduce

The [numerical record](../figures/mh3d/comparison.json) identifies every
configuration, executed source digest, field archive, norm and quadrature check.
The archive stores the full local pressure coefficients and physical geometry.

```bash
pixi run --locked -e notebooks python -m examples.mh3d_campaign
pixi run --locked -e notebooks python -m examples.plot_mh3d
jupyter lab notebooks/darcy/69_mh3d.ipynb
pixi run --locked -e fem pytest -q tests/test_mh3d_fenics.py tests/test_mh2m3d_fenics.py
```

These studies verify the stated tetrahedral configurations. A general
three-dimensional polynomial-space stability theorem, arbitrary polyhedral
macroelements and historical three-dimensional numerical reproduction are
not inferred from these results.

## References

- Gabriel R. Barrenechea, Antonio Tadeu A. Gomes, and Diego Paredes (2024). *A Multiscale Hybrid Method*. SIAM Journal on Scientific Computing 46(3), A1628–A1657. [DOI: 10.1137/22M1542556](https://doi.org/10.1137/22M1542556).

- Franklin de Barros, Alexandre L. Madureira, and Frédéric Valentin (2026). *A three-field Multiscale Method*. arXiv preprint, version 3, 5 August 2026; first submitted 25 April 2024. [arXiv: 2404.16978v3](https://arxiv.org/abs/2404.16978v3).
