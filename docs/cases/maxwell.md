# Time-domain Maxwell with tangential hybridization

`MaxwellStepper` implements the MHM time-domain formulation of
[Lanteri, Paredes, Scheid and Valentin (2018)](https://doi.org/10.1137/16M110037X).
The local approximation is discontinuous Galerkin with central fluxes.
Mass elimination produces a positive-definite global tangential-trace
problem at each electric update. The time-independent local and global
factorizations are prepared once and reused.

The implementation supports scalar transverse-magnetic (TM) fields on
triangular and Cartesian quadrilateral meshes in two dimensions and full vector
fields on tetrahedra in three dimensions. Local polynomial degree, local refinement and skeletal
degree are independent. In three dimensions the fine boundary triangles must
resolve the triangular skeletal partition. Incompatible partitions or
rank-deficient trace choices are rejected.

## Fields, signs and boundary data

With positive scalar or symmetric positive-definite material tensors, the
physical equations are

$$
\begin{aligned}
\varepsilon\,\partial_t E-\nabla\times H&=f,\\
\mu\,\partial_t H+\nabla\times E&=0.
\end{aligned}
$$

In TM, $E=E_z$ and $H=(H_x,H_y)$; electric permittivity is scalar and magnetic
permeability may be a positive-definite $2\times2$ tensor. In three dimensions
both fields have three Cartesian components and both material tensors may be
anisotropic. The skeletal multiplier is

$$
\lambda=H\times n=-n\times H.
$$

Its two coordinates in three dimensions use a deterministic orthonormal
tangent frame on each canonical macroface. Opposite incident normals enter
the local coupling with opposite signs.

Perfectly conducting faces prescribe zero tangential electric field.
Absorbing faces use the physical impedance condition

$$
E_{\mathrm{tan}}-\alpha(H\times n)=g,\qquad \alpha>0.
$$

Here $\alpha$ is a supplied scalar impedance. This does not implement a
general anisotropic transparent boundary map. The callback
`boundary_data(time, points, normals)` returns the scalar TM or Cartesian
vector datum. Boundary power is evaluated with this same sign convention;
homogeneous absorbing data dissipate energy.

Cartesian material interfaces in two dimensions and planar material
interfaces in two or three dimensions split volume integration. Splitting
quadrature does not enrich the local DG polynomial space: resolving a
material-induced derivative jump may require a fitted fine partition.

## Central DG operator and leapfrog

Let $M_\varepsilon,M_\mu$ be the physical mass matrices, $C$ the negative DG
curl, and $B$ the tangential coupling. The semidiscrete equations are

$$
\begin{aligned}
M_\varepsilon\dot e&=-C^Th-B\lambda+f,\\
M_\mu\dot h&=Ce.
\end{aligned}
$$

The derivative uses the central average on interior fine facets and the
incident field on a macro boundary. The electric operator is exactly the
negative transpose of the magnetic operator. This adjoint pairing underlies
the energy balance.

The magnetic update advances $h^n$ to $h^{n+1}$. The electric update advances
$e^{n+1/2}$ to $e^{n+3/2}$. Initialization uses a material-weighted projection,
the discrete interior/PEC constraint, and a half electric step.
`MaxwellSolution.electric_time` and `magnetic_time` explicitly record these
different physical times.

For an electric step of duration $\Delta t$, let $e_f$ be the unconstrained
electric update and $W$ the positive absorbing impedance mass. The trace solves

$$
\begin{aligned}
\left(\frac{\Delta t}{2}B^TM_\varepsilon^{-1}B+W\right)\lambda
&=B^T\frac{e_{\mathrm{old}}+e_f}{2}-g,\\
e_{\mathrm{new}}&=e_f-\Delta t\,M_\varepsilon^{-1}B\lambda.
\end{aligned}
$$

For PEC, $W=0$ and the old constrained field eliminates the midpoint factor,
recovering Equation (5.17). The implementation checks the original
tangential equation after the solve.

The discrete cross-time energy from Equation (5.11) is

$$
\begin{aligned}
2\mathcal E^n={}&
(e^{n+1/2})^TM_\varepsilon e^{n+1/2}\\
&+(h^n)^TM_\mu h^n
+\Delta t\,(h^n)^TCe^{n+1/2}.
\end{aligned}
$$

It is conserved for unforced PEC evolution and includes the physical
source/absorbing work for the midpoint boundary treatment. Conservation of
this quantity is distinct from small field error or magnetic
divergence-freeness.

A conservative local bound uses the mass-scaled curl and positive weighted
row sums of its Gram matrix:

$$
\begin{aligned}
S_K&=L_{\mu,K}^{-1}C_KL_{\varepsilon,K}^{-T},\\
\widehat\Omega_K^2&=\min\left\{
\|S_K\|_1\|S_K\|_\infty,
\max_i\frac{(G_Kx)_i}{x_i}\right\},\qquad x_i>0,\\
G_K&\mathrel{\geq}|S_K^TS_K|\quad\text{entrywise},\\
M_{\delta,K}&=L_{\delta,K}L_{\delta,K}^T,\qquad
\Delta t\,\max_K\widehat\Omega_K<2.
\end{aligned}
$$

Sixteen positive iterations tighten the weighted row bound. The Gram matrix
includes a componentwise floating-point product bound, and a final padding
covers sparse row sums. Every retained iterate supplies an upper bound;
an unconverged eigenvalue estimate is not used as a stability certificate.
Steps violating the resulting sufficient CFL condition are rejected.
The bound can remain more restrictive than the constrained spectral radius.

## Published TM cavity data

Section 6.2 uses $\varepsilon=\mu=1$, zero source, PEC walls, and
$\omega=2\sqrt2\pi$:

$$
\begin{aligned}
E_z(t,x,y)&=\cos(\omega t)\sin(2\pi x)\sin(2\pi y),\\
H_x(t,x,y)&=-\frac{\sin(\omega t)}{\sqrt2}
                 \sin(2\pi x)\cos(2\pi y),\\
H_y(t,x,y)&=\frac{\sin(\omega t)}{\sqrt2}
                 \cos(2\pi x)\sin(2\pi y).
\end{aligned}
$$

The campaign preserves these data, linear or quadratic face traces
($\ell=1,2$), and one local P$_{\ell+2}$ element per triangular macrocell.
The five grid widths are $1/n$ for $n=2,4,8,12,16$. Time stepping uses
$\Delta t=0.0005$ for 100 steps: the final magnetic time is $0.05$ and the
final electric time is $0.05025$. These declared meshes and final times do
not identify every historical marker in the article.

The maximum-in-time norms include every computed step, with analytical
electric and magnetic fields evaluated at their respective staggered
times. `combined_hcurl` includes the broken elementwise curls; it does not
include jump penalties or certify an H(curl)-conforming field.

The combined L2 norm is the square root of the sum of the electric and
magnetic squared errors. The broken H(curl) norm additionally includes both
squared curl errors. The maximum is taken after combining the fields at
each step, rather than combining maxima attained at different steps.
The finest recorded values are:

| Study | Finest grid | Trace / local degree | Maximum combined L2 error | Maximum combined broken H(curl) error |
|---|---:|---|---:|---:|
| Published TM cavity data | $n=16$ | P1 / P3 | $6.81708\times10^{-4}$ | $5.08578\times10^{-2}$ |
| Published TM cavity data | $n=16$ | P2 / P4 | $1.78115\times10^{-5}$ | $3.30319\times10^{-3}$ |
| Vector 3D verification below | $n=5$ | P1 / P3 | $2.34501\times10^{-2}$ | $8.09239\times10^{-1}$ |

The last two levels give L2 rates 2.29, 3.34 and 1.30, respectively.
The three-dimensional sequence still has appreciable field and curl errors;
five decreasing values do not establish its asymptotic rate. These are
absolute physical norms with assembly and error quadrature order 8.

![Spatial Maxwell convergence](../figures/maxwell/convergence.png)

![TM cavity with linear traces](../figures/maxwell/cavity-2d-ell1-n8-fields.png)

![TM cavity with quadratic traces](../figures/maxwell/cavity-2d-ell2-n8-fields.png)

## Full vector three-dimensional cavity

An original analytical verification superposes three nonzero PEC modes:

$$
\begin{aligned}
\widehat E(x,y,z)&=
\begin{pmatrix}
\sin(2\pi y)\sin(2\pi z)\\
0.7\sin(2\pi z)\sin(2\pi x)\\
0.4\sin(2\pi x)\sin(2\pi y)
\end{pmatrix},\\
E(t)&=\cos(\omega t)\widehat E,\qquad
H(t)=-\frac{\sin(\omega t)}{\omega}\nabla\times\widehat E.
\end{aligned}
$$

Both fields are divergence-free analytically. The campaign uses tetrahedral
grids with $n=1,2,3,4,5$, linear tangential traces and local P3 fields.
All three electric and magnetic components are retained. The sections below
evaluate the one-sided DG fields at $z=0.37$ and show the actual intersections
of the macrofaces.

![Three-dimensional electric components](../figures/maxwell/cavity-3d-ell1-n4-electric.png)

![Three-dimensional magnetic components](../figures/maxwell/cavity-3d-ell1-n4-magnetic.png)

## Independent time and operator checks

The temporal study fixes a two-triangle P3/linear-trace discretization and
compares leapfrog with the exact matrix exponential of its constrained
semidiscrete ODE. Five time steps range from $0.004$ to $0.00025$.
This comparison isolates temporal consistency from continuum spatial error.
The combined L2 error decreases from $7.85214\times10^{-5}$ at
$\Delta t=0.004$ to $3.10602\times10^{-7}$ at $\Delta t=0.00025$.
The last halving gives order 2.00. Across all twenty spatial and temporal
records, the largest relative drift of the modified PEC energy is
$2.665\times10^{-15}$. This invariant check does not replace the field-error
measurements.

![Temporal convergence and cross-time energy](../figures/maxwell/time-energy.png)

Portable tests cover nonzero constant fields, outgoing-boundary orientation,
an independently assembled electric/trace saddle, CFL rejection, material
masses, trace compatibility, resource cleanup and PEC energy. Four native
DOLFINx/UFL checks independently assemble central DG derivatives and
anisotropic mass blocks for P1/P2 in two and three dimensions. Analytical
curl identities, divergence and PEC traces are checked by complex-step
differentiation.

```bash
pixi run -e notebooks python -m examples.maxwell_campaign
pixi run -e notebooks python -m examples.plot_maxwell
```

The [numerical records](../figures/maxwell/comparison.json) identify every
degree, norm, time and executed source digest. Notebook `62_maxwell.ipynb`
executes a small three-dimensional energy check and displays the archived
studies.

The [quadrilateral nano-waveguide case](maxwell-nanoguide.md) uses the Section 6.3
geometry, circular material inclusions, local Q2 fields and the two published
skeletal dimensions. Its heterogeneous acquisition has separate material
quadrature, time and classical DG refinement controls. The implementation does not infer an H(div)-conforming
magnetic field or a divergence-cleaning property from energy conservation.
