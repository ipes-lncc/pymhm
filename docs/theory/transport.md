# Conservative transport and implicit time steps

The conservative scalar operator distinguishes its physical transport flux from the half-advection Robin multiplier. [MHM-USFEM](../tutorials/methods/mhm-usfem.md) changes the local reaction–diffusion approximation; [transient transport](../tutorials/methods/transient-transport.md) adds mass and a time discretization.

## Reactive–advective–diffusive equations

The conservative model in [Harder, Paredes and Valentin (2015)](https://doi.org/10.1137/130938499) and [Araya et al. (2024)](https://doi.org/10.1016/j.cma.2024.117089) is

$$
\nabla\cdot(-\mathsf K\nabla u+\alpha u)+\sigma u=f,
\qquad\sigma+\tfrac12\nabla\cdot\alpha\ge0.
$$

Its local skew form is

$$
\begin{aligned}
a_K(u,v)={}&(\mathsf K\nabla u,\nabla v)_K
+\tfrac12(\alpha\cdot\nabla u,v)_K
-\tfrac12(u,\alpha\cdot\nabla v)_K\\
&+((\sigma+\tfrac12\nabla\cdot\alpha)u,v)_K.
\end{aligned}
$$

The local Robin trace is
$\lambda_K=(-\mathsf K\nabla u+\alpha u/2)\cdot n_K$.
The physical transport flux is $j=-\mathsf K\nabla u+\alpha u$.
These fields have different boundary balances, so the half-advection multiplier
must not be inserted into a transport conservation diagnostic as if it were $j$.

[Araya et al. (2024)](https://doi.org/10.1016/j.cma.2024.117089) defines the coarse constant space from the kernel of $a_K$ on each cell.
Positive effective reaction removes it; pure diffusion retains it. Divergence-free
advection tangent to the cell boundary can also retain it. This explains the
transition between skeletal systems with and without coarse constants. It also
explains why kernel detection based solely on whether a coefficient is named
“reaction” is insufficient.

The native `solve_transport` uses this form with Pk local fields and variable
coefficients. A variable velocity requires its divergence explicitly. The
optional SUPG term uses the full conservative residual

$$
L_h u=-\mathsf K:\nabla^2u-(\nabla\cdot\mathsf K)\cdot\nabla u
+\alpha\cdot\nabla u+(\sigma+\nabla\cdot\alpha)u,
$$

and adds $(\tau L_hu,\alpha\cdot\nabla v)$ to the operator and
$(\tau f,\alpha\cdot\nabla v)$ to the load. Both coefficient derivatives
are supplied by the problem definition; they are not silently set to zero for
variable coefficients. This consistent option does not imply a discrete maximum
principle. The constant mode is retained even when it is not a kernel, using the
general decomposition above to preserve the diffusion limit. Mixed boundary
data prescribe the Robin multiplier; a pure-Neumann diffusion problem imposes
one global scalar mean.

## Time-dependent diffusion

For $\partial_tu-\nabla\cdot(\mathsf K\nabla u)=f$, backward Euler
uses the same scalar hybrid spatial operator. A step of length $\Delta t$ replaces each local operator by
$A_K+M_K/\Delta t$ and its source by
$f_K^{n+1}+M_Ku_K^n/\Delta t$. Positive mass removes the scalar local constant
kernel. The skeletal trace still represents outward diffusive flux.
The constant is retained as a coarse mode, so large time steps do not require
inverting an almost singular Neumann operator.

Spatial approximation and first-order temporal error must be assessed separately.
Time-independent diffusion/mass matrices can be assembled once. A fixed time
step permits reuse of the complete local and global step operators; a changed
time step changes the discrete operator. Time-dependent Dirichlet and source
data enter the new-time load. The transient extension must be qualified
separately from a steady benchmark.

The conservative transient study in the
[tutorial](../tutorials/methods/transient-transport.md) also includes convection
and reaction. It uses the smooth spatial estimate of
[Araya et al. (2024), Theorems 2–3](https://doi.org/10.1016/j.cma.2024.117089),
with the local approximation condition A2. For the stated two-dimensional
P2/r2 volume and P1 trace family, the broken gradient target is order two.
The measured concentration order three is additional observed accuracy.
The manufactured concentration is linear in time; an independent half-step
control checks that time integration does not set the spatial error floor.


## MHM-USFEM: reaction–diffusion stabilization

[Santiago, Valentin and Martins (CILAMCE 2025)](https://doi.org/10.55592/cilamce2025.v5i.14270)
use UNUSUAL local stabilization for the scalar reaction–diffusion equation
$Lu=-\operatorname{div}(A\nabla u)+\sigma u=f$. The residual construction
follows [Franca and Valentin (2000)](https://doi.org/10.1016/S0045-7825(00)00190-0).
The local stabilized equations are

$$
\begin{aligned}
a_{\mathrm{UN}}(u,v)&=(A\nabla u,\nabla v)_K+(\sigma u,v)_K
 -\sum_{\tau\subset K}\delta_\tau(Lu,Lv)_\tau,\\
F_{\mathrm{UN}}(v)&=(f,v)_K
 -\sum_{\tau\subset K}\delta_\tau(f,Lv)_\tau.
\end{aligned}
$$

The negative residual sign and the matching load are part of the method.
Inverse constants and certified cellwise coefficient bounds limit
$\delta_\tau$; higher-order fields include their actual Hessians and
variable coefficients include their derivatives. Derivative-based local
stabilization requires its material alignment/regularity conditions.
This scalar method is distinct from mixed velocity/pressure USFEM for flow.

The smooth diffusion-dominated P1 local family has first-order broken
energy/gradient and second-order scalar $L^2$ targets with adequate trace
resolution. These targets do not imply a singular-perturbation-uniform
bound or layer accuracy on a fixed coarse local mesh. The
[MHM-USFEM tutorial](../tutorials/methods/mhm-usfem.md) declares the residual
before adding it to operator and load, and keeps layer and smooth-rate
studies separate.

## Conservative transient transport

With capacity $\rho>0$, backward Euler adds
$(\rho u^{n+1},v)_K/\Delta t$ to the local operator and
$(\rho u^n,v)_K/\Delta t$ to the load. Conservative advection includes
physical source and boundary contributions at the new time. Operator
reuse is valid while geometry, coefficient fields, stabilization and time
step remain unchanged; changing a source need not rebuild the kernel.

The implemented Darcy/transport coupling can use an equilibrated RT0 Darcy
flux as its advective velocity, with the stated hydrodynamic dispersion
law. Pressure-gradient sampling is a different advective field. The
[transient tutorial](../tutorials/methods/transient-transport.md) identifies
capacity-weighted mass and physical boundary transport fluxes. Backward
Euler's temporal order is one; spatial and material errors must not hide
that slope. No monotonicity or discrete maximum principle is implied by
SUPG or MHM assembly alone.

## References

- Christopher Harder, Diego Paredes, and Frédéric Valentin (2015). *On a Multiscale Hybrid-Mixed Method for Advective-Reactive Dominated Problems with Heterogeneous Coefficients*, Multiscale Modeling & Simulation 13(2), 491–518. [DOI: 10.1137/130938499](https://doi.org/10.1137/130938499).

- Rodolfo Araya, Fabrice Jaillet, Diego Paredes, and Frédéric Valentin (2024). *Generalizing the multiscale hybrid-mixed method for reactive-advective-diffusive equations*, Computer Methods in Applied Mechanics and Engineering 428, 117089. [DOI: 10.1016/j.cma.2024.117089](https://doi.org/10.1016/j.cma.2024.117089).

- Juan Felipe Pacazuca Santiago, Frédéric Valentin, and Larissa Martins (2025). *A Multiscale Hybrid-Mixed Method with Local Stabilization*. Proceedings of the Ibero-Latin American Congress on Computational Methods in Engineering, CILAMCE 2025, volume 5, article 14270; published online 18 March 2026. [DOI: 10.55592/cilamce2025.v5i.14270](https://doi.org/10.55592/cilamce2025.v5i.14270).

- L. P. Franca and Frédéric Valentin (2000). *On an improved unusual stabilized finite element method for the advective–reactive–diffusive equation*. Computer Methods in Applied Mechanics and Engineering 190(13–14), 1785–1800. [DOI: 10.1016/S0045-7825(00)00190-0](https://doi.org/10.1016/S0045-7825(00)00190-0).
