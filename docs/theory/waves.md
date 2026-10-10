# Hybrid multiscale wave methods

Frequency-domain acoustics and time-domain electromagnetics use different
volume operators and interface fields. The local/global decomposition is
common, but coercive elliptic assumptions cannot be transferred to an
indefinite Helmholtz operator or an explicit Maxwell time step.

## Helmholtz: complex fields and normal traces

[Chaumont-Frelet and Valentin (2020)](https://doi.org/10.1137/19M1255616)
develop MHM for heterogeneous Helmholtz problems. With positive density $\rho$,
bulk modulus $\kappa$ and frequency $\omega$, the pressure equation is

$$
-\nabla\cdot(\rho^{-1}\nabla p)-\omega^2\kappa^{-1}p=f.
$$

Define $q=-\rho^{-1}\nabla p$ and outward normal trace
$\lambda_K=q\cdot n_K$. **Locally**, source and trace responses solve

$$
\begin{aligned}
a_K^\omega(p_K,v)+\langle\lambda_K,v\rangle_{\partial K}
 &=(f,v)_K,\\
a_K^\omega(p,v)
 &=(\rho^{-1}\nabla p,\nabla v)_K
   -\omega^2(\kappa^{-1}p,v)_K.
\end{aligned}
$$

The inner products conjugate the test field for complex coefficients.
**The global unknown** is the complex normal-trace vector, determined by

$$
\sum_K\langle\mu_K,p_K(\lambda)\rangle_{\partial K}
 =\langle\mu,p_D\rangle_{\Gamma_D}.
$$

Prescribed natural traces are fixed; impedance boundaries contribute
their pressure/normal-trace relation to the same global equations.
There is no elliptic pressure-mean kernel at a resolved nonzero
frequency, and an arbitrary mean constraint cannot repair a resonant
operator. Polynomial and oscillatory face spaces are
different trial bases; absorbing boundary forms and PML introduce complex
coefficients and additional boundary/material conventions.

Local Neumann operators must avoid resonance. For a convex interior
macrocell with constant wave speed $c_K$, $\omega H_K/c_K<\pi$ is a
sufficient resolved-cell condition; it is not a certificate for arbitrary
heterogeneous or nonconvex cells. Numerical rank checks remain necessary.
The global wave problem additionally needs the physical uniqueness
conditions of its boundary value problem.

In the published smooth plane-wave family, trace degree $\ell$ gives
pressure $L^2$ order $\ell+2$ and broken-gradient order $\ell+1$ once
frequency and local approximation are resolved. Current Q$_{\ell+2}$
controls independently assemble the same discrete problem with UFL. A
frequency sweep, a PML study or a heterogeneous Marmousi calculation is a
different qualification from this fixed-frequency rate. Follow the
[Helmholtz tutorial](../tutorials/methods/helmholtz.md).

Implemented local meshes include triangles/polygons, Cartesian Qk cells
and tetrahedra. An oscillatory trace basis can improve a selected angular
case without being uniformly superior to a polynomial basis.

## Maxwell: tangential coupling and staggered dynamics

[Lanteri, Paredes, Scheid and Valentin (2018)](https://doi.org/10.1137/16M110037X)
use tangential hybridization for Maxwell equations in heterogeneous media:

$$
\begin{aligned}
\varepsilon\,\partial_t e-\nabla\times h&=f,\\
\mu\,\partial_t h+\nabla\times e&=0.
\end{aligned}
$$

The local fields evolve on discontinuous fine spaces with central numerical
fluxes; the global multiplier imposes tangential coupling between macrocells.
The implemented transverse-magnetic 2D fields have scalar $e_z$ and vector
$(h_x,h_y)$; the tetrahedral 3D path has full electric and magnetic vectors.
This implementation uses discontinuous polynomial local fields, not a
conforming Nédélec discretization.

**Local semidiscrete problem.** Let $M_{e,K}$ and $M_{h,K}$ be the
positive material mass matrices, $C_K$ the central-DG magnetic evolution
operator (the discrete $-\operatorname{curl}$), and $B_K$ the oriented
tangential coupling. The cell fields satisfy

$$
\begin{aligned}
M_{h,K}\dot h_K&=C_Ke_K,\\
M_{e,K}\dot e_K&=-C_K^Th_K-B_K\lambda_K+f_K.
\end{aligned}
$$

Paired operators $C_K$ and $-C_K^T$ encode the discrete energy exchange.
The global multiplier supplies magnetic tangential loading to the
electric update. **The global problem** enforces the electric tangential
moments:

$$
\sum_K B_K^Te_K-Z\lambda=g.
$$

Interior faces and perfect-electric-conductor (PEC) faces have $Z=0$.
On an absorbing exterior face, $Z$ is the positive impedance trace mass
for $e_{\rm tan}-\alpha(h\times n)=g$, with $\alpha>0$.
The outward orientation and tangential basis are part of $B_K$.

For an electric kick of duration $\Delta t$, compute the free local
prediction $e_K^*$ from magnetic fields and forcing. The midpoint
constraint gives the global trace system

$$
\begin{aligned}
\left(\frac{\Delta t}{2}S+Z\right)\lambda
 &=\sum_KB_K^T\frac{e_K^{\rm old}+e_K^*}{2}-g,\\
S&=\sum_KB_K^TM_{e,K}^{-1}B_K,\\
e_K^{\rm new}&=e_K^*-\Delta t\,M_{e,K}^{-1}B_K\lambda_K.
\end{aligned}
$$

Thus only tangential coefficients are solved globally at the kick;
cell mass solves and the magnetic update are local. Boundary forcing is
evaluated at the corresponding midpoint. This positive mass trace
system is different from the indefinite Helmholtz trace system.

Leapfrog stores $e^{n+1/2}$ and $h^n$ at different times. Compare each field
with the exact field at its own recorded time. The conserved discrete
quadratic energy uses this staggered pair and differs from simply adding
two fields evaluated at an invented common time. Stability requires a
CFL bound based on the actual condensed spatial operator and material
masses; unstable time steps are rejected.

For the published smooth TM cavity with trace degree $\ell=1,2$, combined
$L^2$ orders are $\ell+1$, combined broken-curl orders are $\ell$, and
temporal order is two. The electric-field superconvergence observed for
one published configuration is separate from these combined-field targets.
Local spaces and time steps must make their errors smaller than the
skeletal spatial error during a spatial study. Recorded 2D cavity sequences
approach these targets. The current 3D cavity sequence remains
preasymptotic; the 2D rates are not assigned to it without further evidence.
The [Maxwell tutorial](../tutorials/methods/maxwell.md) declares the
tangential space, time updates and separate field comparisons.

## References

- Stéphane Lanteri, Diego Paredes, Claire Scheid, and Frédéric Valentin (2018). *The Multiscale Hybrid-Mixed method for the Maxwell Equations in Heterogeneous Media*. Multiscale Modeling & Simulation 16(4) 1648-1683. [DOI: 10.1137/16M110037X](https://doi.org/10.1137/16M110037X).

- Théophile Chaumont-Frelet, and Frédéric Valentin (2020). *A Multiscale Hybrid-Mixed Method for the Helmholtz Equation in Heterogeneous Domains*. SIAM Journal on Numerical Analysis 58(2) 1029-1067. [DOI: 10.1137/19M1255616](https://doi.org/10.1137/19M1255616).
