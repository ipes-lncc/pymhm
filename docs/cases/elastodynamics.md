# Elastodynamic MHM with local time propagation

`ElastodynamicStepper` implements the space-time MHM construction of
[Gomes, Paredes, Pereira, Souto and Valentin (2017)](https://doi.org/10.20906/CPS/CILAMCE2017-0399),
Equations (42)–(48). Independent local Newmark responses determine a global
traction problem at each macro time. The implementation supports triangles in
two dimensions and tetrahedra in three dimensions, continuous local Pk fields,
independent vector face spaces, positive heterogeneous density and symmetric
positive-definite elasticity tensors in Kelvin coordinates.

## Equations and time discretization

The displacement, velocity and physical stress satisfy

$$
\begin{aligned}
\rho\,\partial_{tt}u-\nabla\cdot\sigma(u)&=f,\qquad v=\partial_tu,\\
\sigma(u)&=C\varepsilon(u),\qquad
\varepsilon(u)=\tfrac12(\nabla u+\nabla u^T).
\end{aligned}
$$

The multiplier is the negative physical traction, $\lambda=-\sigma n$.
Each macrocell may use its own positive number of local substeps; every
substep sequence ends at the same macro time. The multiplier is constant
over that time slab. Newmark parameters are $\beta=1/4$ and $\gamma=1/2$.
For a local time step $\delta t$, mass $M$ and stiffness $A$, the effective
matrix is

$$
M+\frac{\delta t^2}{4}A.
$$

Its factorization and the global trace factorization are reused throughout
the time march. Inertia retains the rigid motions; a static mean or rotation
gauge is not added to this positive local operator. Initial displacement and
velocity use constrained physical mass projections. Time-independent
Dirichlet displacements and prescribed tractions are supported, together with
time-dependent body forces. Spatial assembly supports serial, thread and
spawn-based process execution.

With one local step per macro step, zero forcing and homogeneous displacement
data, the method conserves the discrete physical energy

$$
\mathcal E^n=\frac12\sum_K\left[
(\rho v^n,v^n)_K+(C\varepsilon(u^n),\varepsilon(u^n))_K\right].
$$

The corresponding invariant is checked numerically. This statement does not
extend automatically to arbitrary asynchronous local step counts. The spatial
displacement formulation is not uniformly locking-free in the incompressible
limit; the static mixed and GaLS elasticity families address that different
requirement.

## Published analytical data

Section 5.1, Equation (53), prescribes the unit cube, $\lambda=\mu=0.4$,
$\rho=1$, zero initial displacement and velocity, and

$$
\begin{aligned}
u(t,x,y,z)&=\frac{1-\cos(\omega t)}{2}
\begin{pmatrix}
\sin(2\pi x)\sin(2\pi y)\sin(2\pi z)\\
-\sin(2\pi x)\sin(2\pi y)\sin(2\pi z)\\
\sin(\pi x)\sin(\pi y)\sin(\pi z)
\end{pmatrix},\\
\omega&=\pi\sqrt{0.8},\qquad
f=\rho\,\partial_{tt}u-\nabla\cdot\sigma(u).
\end{aligned}
$$

The body force is derived from analytical spatial Hessians and time
derivatives, independently of the finite-element assembly. The Dirichlet
data vanish on the cube boundary. Velocity and stress comparisons use the
derivatives of this same exact displacement.

The recorded study uses linear vector face traces, local P3 elements on two
subdivisions per macro direction, and $6n^3$ macrotetrahedra. The seven macro
resolutions are $n=1,2,3,4,5,6,8$, with $T=0.5$ and $\Delta t=0.005$.
The article's Figure 3 specifies linear traces and reports spatial rates
$H^3$ for displacement/velocity L2, $H^2$ for broken H1 and $H$ for the
broken stress H(div) norm. Its text prints $\ell=k+2$, which does not identify
a compatible positive local degree for the linear-trace figure. The local
P3 choice and the tetrahedral connectivity are therefore explicit choices
of this study, rather than an assertion of identical historical matrices.

![Published elastodynamic convergence](../figures/elastodynamics/published-convergence.png)

*Gomes et al. (2017), Figure 3. The spatial and temporal reference slopes
concern different refinement sequences.*

The stress norm includes the divergence inside every fine tetrahedron:

$$
\|\sigma-\sigma_h\|_{\mathrm{div},h}^2=
\sum_{K}\sum_{\tau\subset K}
\left(\|\sigma-\sigma_h\|_{L^2(\tau)}^2+
\|\nabla\cdot(\sigma-\sigma_h)\|_{L^2(\tau)}^2\right).
$$

Raw stress is not an H(div)-conforming reconstruction. The temporal study
fixes the spatial space and compares eight time steps against a separately
computed finer time sequence; it isolates time error from the spatial floor.

## Spatial and temporal convergence

All reported errors are physical absolute norms at $T=0.5$. The spatial
sequence keeps $\Delta t=0.005$ fixed and uses the exact solution above.
The macro diameter is $H=\sqrt{3}/n$.

| $n$ | $\lVert u-u_h\rVert_0$ | $\lVert v-v_h\rVert_0$ | $\lVert u-u_h\rVert_{1,h}$ | $\lVert v-v_h\rVert_{1,h}$ | $\lVert\sigma-\sigma_h\rVert_{\mathrm{div},h}$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.165342 | 1.05576 | 2.27829 | 13.0409 | 11.4577 |
| 2 | 0.105892 | 0.642821 | 1.88806 | 10.6966 | 6.17255 |
| 3 | 0.0490952 | 0.266965 | 1.51492 | 7.54504 | 6.70681 |
| 4 | 0.0265844 | 0.123447 | 1.07173 | 4.59022 | 5.63246 |
| 5 | 0.015206 | 0.0599513 | 0.758453 | 2.78195 | 4.70529 |
| 6 | 0.0091762 | 0.0323375 | 0.546213 | 1.82147 | 3.94695 |
| 8 | 0.00400298 | 0.0137415 | 0.317497 | 1.07124 | 2.98365 |

The last two levels give orders **2.884, 2.975, 1.886, 1.845 and 0.973** in
the five columns, respectively. These approach the published rates $3,3,2,2,1$;
the coarsest levels are not in that asymptotic regime. Increasing the error
quadrature from 12 to 16 changes the finest norms by at most $1.25\times10^{-15}$
relative. The finest stress L2 error is $0.105584$. These are broken derivative
norms: interelement jumps are not included as distributional derivatives.

The assembly rule uses order 12 on the coarsest mesh and order 8 on the other
reported spatial levels. An independent order-14 acquisition at $n=1$ changes
the fields by at most $2.89\times10^{-12}$ of their physical error norms.
At $n=2$, increasing the assembly order from 8 to 12 at four time steps changes
the fields by at most $4.81\times10^{-10}$ of their error norms. The
[assembly-quadrature record](../figures/elastodynamics/assembly-quadrature.json)
retains all six physical increments. These checks concern integration of the
source and operators, separately from the error-norm quadrature above.

The temporal sequence fixes $n=2$ and compares with
$\Delta t_{\mathrm{ref}}=0.00009765625$ on the identical spatial space.
The values below are physical differences to that numerical time reference,
not errors relative to the continuous exact solution.

| $\Delta t$ | Displacement L2 | Velocity L2 | Displacement H1 | Velocity H1 | Stress H(div) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.00248663 | 0.0238645 | 0.045666 | 0.628849 | 0.351221 |
| 0.05 | 0.000722904 | 0.0095129 | 0.016475 | 0.353289 | 0.167898 |
| 0.025 | 0.000197545 | 0.00358144 | 0.00564637 | 0.170268 | 0.0847293 |
| 0.0125 | 5.20472e-05 | 0.00131175 | 0.00173116 | 0.0799423 | 0.0315893 |
| 0.00625 | 1.33426e-05 | 0.000358633 | 0.000484022 | 0.0230522 | 0.0121941 |
| 0.003125 | 3.34991e-06 | 9.26956e-05 | 0.000122859 | 0.00619241 | 0.00314729 |
| 0.0015625 | 8.35725e-07 | 2.33195e-05 | 3.06688e-05 | 0.00157328 | 0.000780334 |
| 0.00078125 | 2.06515e-07 | 5.77257e-06 | 7.57966e-06 | 0.000390216 | 0.000192569 |

The final temporal orders range from **2.011 to 2.019**. The reference has
its own refinement check: halving its predecessor's step from $0.0001953125$
to $0.00009765625$ changes displacement L2 by $9.835\times10^{-9}$,
velocity L2 by $2.750\times10^{-7}$ and stress H(div) by
$9.167\times10^{-6}$. These increments are about **4.76%** of the smallest
reported comparison in each norm. The reference is therefore a numerical
trajectory with a measured time-resolution uncertainty. For a leading
second-order error, comparison with a finite reference step has a term
proportional to $\Delta t^2-\Delta t_{\mathrm{ref}}^2$; here the smallest
comparison step is eight times the reference step.


## Displacement and velocity fields

The sections at $z=0.37$ show each signed component on the finest spatial
level. Exact and numerical panels share their full color limits; difference
panels have separate symmetric scales. Actual macro intersections are drawn
on every panel. Evaluation preserves separate values on either side of a
macroface, including visible trace errors.

![Displacement components and signed errors](../figures/elastodynamics/displacement-components.png)


The profiles use $y=0.413$, $z=0.37$. Dotted vertical lines mark macroface
crossings. Each polynomial piece is plotted independently, so continuity is
not imposed by the rendering.

![Displacement and velocity profiles](../figures/elastodynamics/profiles.png)

The [numerical records](../figures/elastodynamics/comparison.json) retain all
seven spatial levels, all eight temporal comparisons, acquisition source
hashes and field archive hashes. The
[elastodynamic notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/waves/elastodynamics/71_elastodynamics.ipynb) replays these
results.

## Independent verification

[DOLFINx/UFL 0.9.0](https://github.com/FEniCS/dolfinx/tree/6443e3b27d29aec04ce83d8424b57c339e35f865)
independently assembles the complete displacement/traction saddle system for
Equation (53), without using PyMHM local matrices or condensation. Both codes
use the same P3/P1 spaces, two local subdivisions, zero initial/boundary
displacement, and the same Newmark rule through $T=0.5$. Five full trajectories
cover $n=1$, $\Delta t=0.005$, and $n=2$ with
$\Delta t=0.1,0.05,0.025,0.005$. The native systems have 1,674 and 13,176
unknowns, respectively.

The maximum relative field/gradient difference over all acquired times is
$1.103\times10^{-11}$, and the largest normalized original-equation residual
is $7.62\times10^{-15}$. UFL independently integrates displacement and velocity
L2/full broken H1, stress Frobenius L2, and broken stress H(div) errors.
The largest relative difference between the codes' error norms is
$6.07\times10^{-13}$. Raising native error quadrature from degree 24 to 28
changes a norm by at most $8.41\times10^{-11}$ relatively; independently
replayed native energy agrees within $1.13\times10^{-14}$.


These panels compare the $n=2$, $\Delta t=0.005$ discrete fields at $z=0.37$.
The differences use their own indicated roundoff-scale color bars. The
coarser cross-code comparison verifies the time propagation and assembled
spaces; the preceding seven-level exact-solution study measures approximation
accuracy. Actual macro intersections and separate polynomial traces are
preserved in both views.

The [independent record](../figures/elastodynamics/native-discrete-verification.json)
identifies the native release/build, source hashes, every acquired time step,
physical norms, and archive digests. Native nodal coordinates and their
permutation into the declared polynomial ordering are saved with the fields.

## Heterogeneous published example

The heterogeneous example in Section 5.2 uses homogeneous exterior traction.
Its Equation (57) specifies a finite-radius volumetric force,

$$
f(t,x)=
\begin{cases}
f_{\mathrm{shot}}(t)\dfrac{x-p_{\mathrm{shot}}}
{\lVert x-p_{\mathrm{shot}}\rVert},
&0<\lVert x-p_{\mathrm{shot}}\rVert<r_{\mathrm{shot}},
\quad t<t_{\mathrm{shot}},\\
0,&\text{otherwise}.
\end{cases}
$$

The direction is radial and the magnitude is bounded; this is not a Dirac
load. It can be supplied through the existing time-dependent body-force
interface, with spatial quadrature resolving the disk. The article gives
$p_{\mathrm{shot}}=(500,50)\,\mathrm m$ and names an Ormsby wavelet, but does
not state the radius, duration, four frequencies, amplitude or time shift.
Those parameters and the layer interfaces are required for a quantitative
reproduction of Figures 6–9. Absorbing elastic boundaries are mentioned in
Section 2 as a possible extension and are not used in these examples.

The [three-layer verification record](../figures/elastodynamics/three-layer-2017-whole301-verification.json)
reports complete original-equation and coefficient-coordinate comparisons for a specified
interpretation of this problem. Its domain is $1000\times450\,\mathrm m$,
with depth increasing downward and the radial source centered at
$(500,225)\,\mathrm m$. Digitized interfaces and material-table layers 1, 7
and 15 define the three materials. The force has radius $20\,\mathrm m$,
unit amplitude in $\mathrm{N/m^3}$, selected Ormsby corners
$(5,10,15,20)\,\mathrm{Hz}$, time shift $0.2\,\mathrm s$ and duration
$0.4\,\mathrm s$. These declared inputs define a reproducible physical
case; the unavailable original mesh, horizons and source parameters prevent
identifying it as a literal reproduction of Figures 6–9.

The physical plane-strain convention preserves the tabulated wave speeds:

$$
\mu=\rho V_s^2,\qquad
\lambda=\rho(V_p^2-2V_s^2),\qquad
\sigma=2\mu\epsilon(u)+\lambda\operatorname{tr}(\epsilon(u))I.
$$

The article prints $\rho(V_p^2-V_s^2)$ for its second coefficient.
Substituting that value into the stated stress law changes the compressional
speed to $\sqrt{V_p^2+V_s^2}$; the record explicitly distinguishes the
executed physical convention from that printed expression.

Both implementations use the same 341 triangular macros, continuous local
vector P3 spaces on 64 fine triangles per macro, vector P2 traction traces
on eight segments per original face, zero initial data and homogeneous
exterior traction. Newmark parameters $(\beta,\gamma)=(1/4,1/2)$ and
$\Delta t=0.001\,\mathrm s$ give 301 saved states through
$T=0.3\,\mathrm s$. No rigid-motion gauge is added: the transient mass
operator and initial data determine that component. The independent
reference uses native FEniCS Basix/UFL/DOLFINx operators; its verified
revisions and executed binary, basis, operator and state digests appear in
the record. It is an instrumented comparison implementation.

All 301 states pass the unchanged relative $10^{-10}$ coordinate comparison
criterion. Volume differences use native Basix Gram matrices after a nodal
coefficient bijection. These values therefore describe a common Basix
interpretation; evaluation with each implementation's own executed volume
basis remains a separate required comparison. The maximum differences in
that common interpretation and the declared P2 traction convention are:

| Common-basis volume norm or P2 traction norm | Maximum relative difference |
| --- | ---: |
| Displacement, volume L2 | $2.170\times10^{-13}$ |
| Velocity, volume L2 | $2.874\times10^{-13}$ |
| Raw displacement gradient, volume L2 | $8.580\times10^{-13}$ |
| Raw Cauchy stress, volume L2 | $4.800\times10^{-13}$ |
| Negative mean physical traction, skeleton L2 | $4.404\times10^{-11}$ |
| Full broken H1 in the declared normalized variables | $8.580\times10^{-13}$ |

The multiplier is the negative physical traction averaged over the preceding
time slab. Every original macroface is integrated once. Raw Cauchy stress
is not an H(div) stress reconstruction. The source remains active at the
final time, so the energy balance includes its work. Original momentum
residuals are at most $2.318\times10^{-13}$ in PyMHM and
$2.101\times10^{-13}$ in the native reference. Original-equation and
coordinate agreement do not establish physical resolution. A classical conforming
reference on several finer meshes and space, time and quadrature refinement
remain required; such a numerical reference is not an exact solution.

Portable checks compare a complete displacement/traction saddle with local
condensation, verify physical energy, independent local step counts, material
mass integration, boundary signs and serial/process trajectories. Native
DOLFINx/UFL comparisons assemble the inertia and stiffness independently in
two and three dimensions. A constrained semidiscrete reference isolates the
second-order Newmark error. The source also passes an independent
finite-difference check of acceleration and stress divergence.

```bash
pixi run --locked -e test-core python -m examples.elastodynamics_campaign --n 1 --dt 0.005 --order 12
for n in 2 3 4 5 6 8; do
  pixi run --locked -e test-core python -m examples.elastodynamics_campaign --n "$n" --dt 0.005 --order 8 --workers 4
done
for dt in 0.1 0.05 0.025 0.0125 0.00625 0.003125 0.0015625 0.00078125 0.0001953125 0.00009765625; do
  pixi run --locked -e test-core python -m examples.elastodynamics_campaign --n 2 --dt "$dt" --order 8
done
pixi run -e notebooks python -m examples.elastodynamics_results
pixi run -e notebooks python -m examples.plot_elastodynamics_native
pixi run -e notebooks notebooks-run notebooks/waves/elastodynamics/71_elastodynamics.ipynb
```
