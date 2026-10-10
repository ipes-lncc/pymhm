# MHM for elasticity and elastodynamics

Elasticity requires all rigid motions as retained physical modes. Choose primal displacement, displacement–pressure GaLS, or stress–displacement–rotation local spaces according to the material regime and the fields required. The [primal](../tutorials/methods/primal-elasticity.md), [GaLS](../tutorials/methods/gals-elasticity.md) and [mixed stress](../tutorials/methods/mixed-elasticity.md) tutorials declare these choices explicitly.

## Linear elasticity and nearly incompressible materials

For displacement $u$, strain
$\varepsilon(u)=(\nabla u+\nabla u^T)/2$, and stiffness tensor
$\mathsf C$, the model is

$$
-\nabla\cdot\tau=f,\qquad\tau=\mathsf C:\varepsilon(u).
$$

The local kernel is the space of rigid motions,

$$
\mathcal R(K)=\{a+Rx:\ R^T=-R\},\qquad
\dim\mathcal R(K)=d(d+1)/2.
$$

Orthogonality to all these modes is required. Translation constraints alone leave
rotation singularities. Testing against rigid motions imposes both resultant
force and moment balance. With a positive boundary term in the local equation,
the multiplier is $\lambda_K=-\tau n_K$. Some elasticity papers use
$+\tau n_K$ and reverse the boundary signs; both conventions are valid when
used consistently.

### Primal local problems and global equilibrium

Set $a_K(u,v)=(\mathsf C:\varepsilon(u),\varepsilon(v))_K$.
**Locally**, find source and negative-traction responses in

$$
\widetilde V_h(K)=
 \{v\in V_h(K)^d:(v,r)_K=0\text{ for every }r\in\mathcal R(K)\},
$$

by solving

$$
a_K(\widetilde u_K,v)+\langle\lambda_K,v\rangle_{\partial K}
 =(f,v)_K,\qquad v\in\widetilde V_h(K).
$$

The full displacement is
$u_K=\widetilde u_K+\sum_jc_{K,j}r_{K,j}$.
**Globally**, determine the vector traction field $\lambda$ and all
rigid-motion coefficients $c_{K,j}$ from

$$
\begin{aligned}
\sum_K\langle\mu_K,u_K\rangle_{\partial K}
 &=\langle\mu,u_D\rangle_{\Gamma_D},\\
\langle\lambda_K,r\rangle_{\partial K}&=(f,r)_K,
 &&r\in\mathcal R(K).
\end{aligned}
$$

Translations give force balance and rotations give moment balance.
Prescribed physical traction $t_N=\tau n$ fixes $\lambda=-t_N$ on
$\Gamma_N$; free trace tests vanish there. A pure-traction problem needs
global force/moment compatibility and one set of global rigid-displacement
moments. Fixing every cell's rigid displacement would remove the coarse
physical unknowns.

The mixed weak-symmetry formulation of [Devloo et al. (2021)](https://doi.org/10.1051/m2an/2021013) uses stress in a tensor H(div) space,
displacement in a discontinuous space, and a rotation multiplier. In two dimensions,
the weak symmetry equation tests $\tau_{12}-\tau_{21}$ against the rotation
space. It enforces stress symmetry in moments, not necessarily pointwise.

For isotropic nearly incompressible elasticity, let $G$ be the shear modulus and
$\lambda_L$ the first Lamé coefficient. The Herrmann-pressure version is

$$
\tau=2G\varepsilon(u)-pI,\qquad
\nabla\cdot u+\lambda_L^{-1}p=0.
$$

The method of [Gomes, Pereira and Valentin (2024, preprint v1)](https://arxiv.org/abs/2403.16890v1) combines this local mixed form with consistent least-squares stabilization and
rigid-motion constraints. A primal displacement-only local method can lock as
$\lambda_L/G\to\infty$. Accurate affine patches at one Poisson ratio cannot
establish a locking-free claim; displacement, pressure, stress, and conditioning
must be tested over the nearly incompressible limit.

### Implemented GaLS operator

Let $\epsilon_L=1/\lambda_L$ and $R(u,p)=\nabla\cdot(2G\varepsilon(u)-pI)$.
The local bilinear and source forms are

$$
\begin{aligned}
B_K((u,p),(v,q))={}&(2G\varepsilon(u),\varepsilon(v))_K
-(p,\operatorname{div}v)_K-(q,\operatorname{div}u)_K
-(\epsilon_Lp,q)_K\\
&-\alpha_K\sum_{\tau\subset K}h_\tau^2(R(u,p),R(v,q))_\tau,\\
F_K(v,q)={}&(f,v)_K+\alpha_K\sum_{\tau\subset K}h_\tau^2(f,R(v,q))_\tau.
\end{aligned}
$$

**Local unknowns** are the displacement–Herrmann-pressure pair $(u_K,p_K)$.
The equation on the rigid-displacement complement is

$$
B_K((u_K,p_K),(v,q))+\langle\lambda_K,v\rangle_{\partial K}
 =F_K(v,q).
$$

The pressure is left free.
**Global unknowns** are the same negative tractions and rigid
displacements as in primal elasticity. Their coupling uses the
displacement-moment and rigid-equilibrium equations above, now with the
GaLS pair reconstructed locally. Testing a rigid displacement introduces
no strain or residual stabilization, so its force/moment balance is
unchanged. Compressibility or the global incompressible pressure gauge
supplies the pressure identity described below.

Equal-order Pk/Pk fields are restricted only by physical displacement moments
against the three rigid modes; there is no local pressure gauge. For variable
shear modulus, $R$ includes $2\varepsilon(u)\nabla G$. The API requires its
derivative and material bounds used in the stabilization inequality. The
inverse constant is computed without an exact solution or measured error.
In two dimensions, a linear traction segment needs four fine intervals for
P1, two for P2 and one for P3; the implemented sufficient conditions follow
[Gomes, Pereira and Valentin (2024, preprint v1)](https://arxiv.org/abs/2403.16890v1) Lemma 4.5 and reject incompatible trace/local choices before elimination.

A full displacement boundary determines the physical integral identity

$$
\int_\Omega \frac{p}{\lambda_L}
=-\int_{\partial\Omega}g\cdot n.
$$

This identity is enforced consistently with the assembled boundary moments,
including when $\lambda_L$ varies spatially. If the entire material is exactly
incompressible, compatibility requires zero boundary volume flux and one global
pressure mean instead. The unconstrained physical equations are checked after
this augmentation; a nonzero imposed mean does not replace equilibrium.
The [elasticity cases](../cases/elasticity.md) and
[MSL comparison](../cases/elasticity-reference.md) provide non-affine verification.

### Implemented weak-symmetry stress method

Write $\mathsf A$ for the material compliance and $r_K$ for a
skew-symmetric rotation field. **Locally**, prescribe
$\tau_hn_K=-\lambda_K$ and find stress, displacement and rotation from

$$
\begin{aligned}
(\mathsf A\tau_h,S)_K+(u_h,\operatorname{div}S)_K+(r_h,S)_K&=0,
 &&Sn_K=0,\\
(\operatorname{div}\tau_h,v)_K&=-(f,v)_K,\\
(\tau_h,s)_K&=0,
 &&s\text{ in the skew-symmetric test space}.
\end{aligned}
$$

Rigid displacement and its associated rotation form a coupled kernel.
During a source or trace response, the displacement/rotation test pair
belongs to the complement of this coupled kernel. The omitted kernel
tests remain global equilibrium equations; the complete reconstructed
solution satisfies the displayed divergence and symmetry rows for the
full test space. The last equation enforces weak stress symmetry.

**Globally**, negative-traction coefficients and the coupled rigid
coefficients determine displacement continuity. The local displacement
trace is the variational functional $\widehat u_K$ defined by

$$
\langle\widehat u_K,Sn_K\rangle_{\partial K}
 =(\mathsf A\tau_h,S)_K+(u_h,\operatorname{div}S)_K+(r_h,S)_K.
$$

Its skeletal moments obey

$$
\sum_K\langle\mu_K,\widehat u_K\rangle_{\partial K}
 =\langle\mu,u_D\rangle_{\Gamma_D},
$$

together with the same force/moment balances. This functional is determined by the mixed
equations; a discontinuous volume displacement is not assigned an
arbitrary pointwise boundary value.

The two-dimensional stress solver uses two BDM2 rows, discontinuous P1
vector displacement and discontinuous P1 scalar rotation. All local rigid
motions are represented exactly; the rotation component of a rigid mode is
retained together with its displacement. Physical traction moments restrict
the macro boundary. Exterior faces may carry higher-resolution tractions than
interior faces, following [Devloo et al. (2021)](https://doi.org/10.1051/m2an/2021013)'s construction. Compliance is evaluated through
spherical and deviatoric parts to avoid subtracting nearly equal material
coefficients. It supports heterogeneous isotropic Lamé fields and zero
spherical compliance at infinite first Lamé modulus. For a fully prescribed
displacement boundary, the integrated constitutive identity fixes the finite
hydrostatic stress; at infinite modulus, one global mean of negative half
the stress trace fixes its pressure gauge.

Weak symmetry, divergence moments and normal-traction agreement are measured
separately. P1 force projection is enforced in every fine cell, whereas the
pointwise divergence error against a non-polynomial force remains nonzero.
See [mixed elasticity](../cases/mixed-elasticity.md) for the oscillatory-modulus
example, convergence, material sweep and independent finite-element comparison.


## Primal and mixed degree requirements

[Harder, Madureira and Valentin (2016)](https://doi.org/10.1051/m2an/2015046)
require injectivity of the skeletal traction lifting. For one triangular
local element, Lemma 6.6 gives the sufficient conditions
$k\ge\ell+1$ for even $\ell$ and $k\ge\ell+2$ for odd $\ell$.
Compatible bubble enrichment or local mesh refinement offers other
constructions. Smooth primal energy/stress rates and displacement gains
require the theorem's local and dual regularity; the primal material
constants are not uniform at incompressibility.

[Devloo et al. (2021)](https://doi.org/10.1051/m2an/2021013) analyze
row-wise mixed stress spaces with independent interior enrichment in two
dimensions. BDM2/P1/P1 contains all three rigid displacements and their
coupled rotations, unlike BDM1/P0/P0. The three-dimensional construction
uses the AFW tetrahedral family of
[Arnold, Falk and Winther (2007)](https://doi.org/10.1090/S0025-5718-07-01998-9):
BDM$_k$ stress rows and discontinuous P$_{k-1}$ displacement/rotation, with
$k\ge2$ so every rigid displacement is retained. Six rigid modes are
required. Smooth $L^2$ stress, displacement and rotation targets are order
$k$ for this stated family, with sufficiently accurate macro traction.
The 2D polygonal enrichment theorem is not used as a proof for arbitrary
3D mixed polyhedra.

GaLS requires its own strain-energy inverse inequality. The two-dimensional
linear-trace sufficient pairs $(k,r)=(1,4),(2,2),(3,1)$ come from
[Gomes, Pereira and Valentin (2024)](https://arxiv.org/abs/2403.16890v1),
Lemma 4.5. The tetrahedral implementation recomputes inverse bounds on the
physical cell and retains six rigid modes. A sampled Lamé sweep verifies
the stated discretization; it does not establish contrast-independent
accuracy for arbitrary anisotropic elasticity.

## Elastodynamics and time accuracy

[Gomes et al. (2017)](https://doi.org/10.20906/CPS/CILAMCE2017-0399) combine
local dynamics with slabwise negative-traction coupling. For positive
material density $\rho$,

$$
\rho\,\partial_{tt}u-\operatorname{div}(C\varepsilon(u))=f,
\qquad v=\partial_tu.
$$

**Local problem on a macro time slab.** For a supplied negative traction
$\lambda_K$ and the known displacement/velocity at the slab's start,
propagate

$$
(\rho\,\partial_{tt}u_K,v)_K
 +(C\varepsilon(u_K),\varepsilon(v))_K
 +\langle\lambda_K,v\rangle_{\partial K}=(f,v)_K.
$$

Each cell computes its free response and responses to the slabwise
constant traction coefficients. **The global problem** chooses those
coefficients so that the end-of-slab displacement satisfies

$$
\sum_K\langle\mu_K,u_K(t_{n+1})\rangle_{\partial K}
 =\langle\mu,u_D(t_{n+1})\rangle_{\Gamma_D}.
$$

Physical traction data fix the exterior negative-traction coefficients.
Initial conditions determine the rigid displacement and velocity;
static force/moment gauges are not added to the inertial local solve.

The local Newmark parameters $\beta=1/4$, $\gamma=1/2$ give the effective
operator $M+\delta t^2A/4$ and second-order time accuracy for sufficiently
smooth solutions. Local substeps finish at the common macro time. Inertia
controls rigid modes, so a static displacement gauge must not be added to
this positive local operator.

For one local time step per macro step, zero forcing and homogeneous
prescribed displacement, the scheme conserves its discrete physical
energy. This does not extend automatically to asynchronous local substep
counts. Section 5.1 of Gomes et al. reports displacement and velocity $L^2$
order three, broken $H^1$ order two and broken stress $H(\mathrm{div})$ order
one for its smooth linear-traction experiment. These are published numerical
targets; that paper does not prove a dynamic error estimate. Time error must
be reduced independently.

The [elastodynamics tutorial](../tutorials/methods/elastodynamics.md) evaluates
a smooth two-dimensional analogue with P3 local displacement and P1 negative
traction. It measures displacement, velocity and physical stress separately,
with a time-step control and an independently refined classical reference.
The tetrahedral applications are separate cases, rather than a claim of
reproducing the paper's three-dimensional discretization in this tutorial.
Primal dynamics is not a uniform locking-free extension of mixed static
elasticity.

## References

- Christopher Harder, Alexandre L. Madureira, and Frédéric Valentin (2016). *A hybrid-mixed method for elasticity*, ESAIM: M2AN 50, 311–336. [DOI: 10.1051/m2an/2015046](https://doi.org/10.1051/m2an/2015046).

- Philippe R. B. Devloo, Agnaldo M. Farias, Sônia M. Gomes, Weslley Pereira, Antonio J. B. dos Santos, and Frédéric Valentin (2021). *New H(div)-conforming multiscale hybrid-mixed methods for the elasticity problem on polygonal meshes*, ESAIM: M2AN 55, 1005–1037. [DOI: 10.1051/m2an/2021013](https://doi.org/10.1051/m2an/2021013).

- Antônio Tadeu Azevedo Gomes, Weslley da Silva Pereira, and Frédéric Valentin (2024). *A low-order locking-free multiscale finite element method for isotropic elasticity*, arXiv preprint, version 1, 25 March 2024. [arXiv: 2403.16890v1](https://arxiv.org/abs/2403.16890v1).

- Antonio Tadeu Gomes, Diego Paredes, Weslley Pereira, Roberto Souto, and Frederic Valentin (2017). *A Multiscale Hybrid-Mixed Method for the Elastodynamic Model with Rough Coefficients*. Proceedings of the XXXVIII Iberian Latin American Congress on Computational Methods in Engineering. [DOI: 10.20906/CPS/CILAMCE2017-0399](https://doi.org/10.20906/CPS/CILAMCE2017-0399).

- Douglas N. Arnold, Richard S. Falk, and Ragnar Winther (2007). *Mixed finite element methods for linear elasticity with weakly imposed symmetry*. Mathematics of Computation 76, 1699–1723. [DOI: 10.1090/S0025-5718-07-01998-9](https://doi.org/10.1090/S0025-5718-07-01998-9). [Preprint: arXiv:math/0701506v1](https://arxiv.org/abs/math/0701506v1).
