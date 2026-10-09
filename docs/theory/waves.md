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

The complex local bilinear form contains gradient and negative mass terms.
Local responses to source and oriented normal traces enter the global
pressure-continuity equations. Polynomial and oscillatory face spaces are
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
