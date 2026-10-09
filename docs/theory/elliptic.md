# Elliptic MHM: primal and mixed local spaces

The original MHM couples normal-flux traces and macrocell pressure means. A primal or mixed local finite element method changes the local approximation and the meaning of the reconstructed flux; the global hybrid structure remains explicit. See the [primal MHM tutorial](../tutorials/methods/primal-mhm.md) and [mixed MHM tutorial](../tutorials/methods/mixed-mhm.md).

## Darcy: primal local problems

The primal hybridization follows [Harder, Paredes and Valentin (2013)](https://doi.org/10.1016/j.jcp.2013.03.019); its elliptic
analysis and estimator are developed by [Araya et al. (2013)](https://doi.org/10.1137/120888223).

For a symmetric uniformly positive definite tensor $\mathsf K$, consider

$$
q=-\mathsf K\nabla p,\qquad \nabla\cdot q=f.
$$

The hybrid unknown is $\lambda_K=q\cdot n_K$. With
$a_K(p,v)=(\mathsf K\nabla p,\nabla v)_K$, the local equation is

$$
a_K(p,v)+\langle\lambda_K,v\rangle_{\partial K}=(f,v)_K.
$$

The global trace equation imposes weak continuity of pressure and Dirichlet data:

$$
\sum_K\langle\mu_K,p_K\rangle_{\partial K}
=\langle\mu,p_D\rangle_{\Gamma_D}.
$$

For diffusion with pure local Neumann conditions, constants form the local kernel.
Split $p_K=c_K+\widetilde p_K$, with
$\int_K\widetilde p_K=0$, and define

$$
\begin{aligned}
a_K(T_K\lambda,v)&=-\langle\lambda_K,v\rangle_{\partial K},\\
a_K(\widehat T_K f,v)&=(f,v)_K,
\end{aligned}
\qquad \int_Kv=0.
$$

Then $p_K=c_K+T_K\lambda+\widehat T_K f$. Testing with a constant gives
the coarse conservation equation

$$
\int_{\partial K}\lambda_K\,ds=\int_Kf\,dx.
$$

The source lifting is generally necessary. Omitting it solves a different
discretization even if the global flux balance still holds. Local face basis
functions with nonzero net normal flux are well-defined on the zero-mean test
space: their complementary constant residual is accounted for by the coarse
balance equation. They should not be rejected individually as incompatible pure
Neumann data.

For a globally pure Neumann problem, compatibility requires
$\int_\Omega f=\int_{\partial\Omega}q_N$. A single global pressure
normalization, such as $\int_\Omega p=0$, then fixes uniqueness. Prescribing
all macroelement means to zero would remove physical coarse unknowns.

## Mixed H(div) local Darcy problems

Choose $V_h(K)\subset H(\mathrm{div};K)$ and
$Q_h(K)\subset L^2(K)$ with a stable pairing and matching divergence space.
For prescribed normal trace, solve

$$
\begin{aligned}
(\mathsf K^{-1}q_h,v_h)_K-(p_h,\nabla\cdot v_h)_K&=0,
&&v_h\cdot n_K=0,\\
(\nabla\cdot q_h,r_h)_K&=(f,r_h)_K.
\end{aligned}
$$

The normal traces on the macro boundary are constrained to the selected skeletal
space, while internal flux degrees of freedom can be refined or enriched. A
mean-zero pressure complement and a coarse pressure constant separate local and
global work. If the pressure space contains each microelement constant and
quadrature is consistent, the divergence equation enforces microelement mass
balance. The matching normal trace across every face gives global H(div)
conformity. The lowest-order Raviart–Thomas pair uses a discontinuous constant
pressure and is only the first member of the mixed family in [Durán et al. (2019)](https://doi.org/10.1016/j.cma.2019.05.013).

A primal local solve gives a broken raw flux
$q_h^{\rm raw}=-\mathsf K\nabla p_h$. Its normal component generally jumps
across fine faces. The multiplier can satisfy coarse balance even when this raw
flux is not H(div). Reporting “conservative flux” therefore requires identifying
the field and the spatial scale. A reconstruction and a mixed local solve are
distinct ways of obtaining an H(div) field.

The implemented `equilibrate_flux` instead solves, independently in each
macroelement, the constrained minimization

$$
\min_{q_h\in RT_0}\frac12\|q_h+\mathsf K\nabla p_h\|^2_{\mathsf K^{-1},K},
\qquad
\nabla\cdot q_h|_\tau=\frac1{|\tau|}\int_\tau f,
\qquad q_h\cdot n_K=\lambda_K.
$$

The fine-cell balance and prescribed trace must be compatible. The current
operator requires piecewise constant trace segments aligned with fine boundary
edges. It is not the face-moment reconstruction analyzed in [Barrenechea et al. (2026)](https://doi.org/10.1137/24M1673073): its defining
constraints enforce fine-cell source moments explicitly.


## Degrees, rates and geometry

For the smooth original family, skeletal degree $\ell$ targets broken
energy/physical-flux order $\ell+1$. Pressure $L^2$ can gain one order
when the dual problem is regular. These are two-level statements: the
conforming local finite element error must be made smaller independently.
The original construction and analysis are
[Harder, Paredes and Valentin (2013)](https://doi.org/10.1016/j.jcp.2013.03.019)
and [Araya et al. (2013)](https://doi.org/10.1137/120888223).
An algebraic kernel/trace rank check does not establish the uniform
lifting condition required by those estimates.

Mixed locals must satisfy $\operatorname{div}V_h=Q_h$. The triangular RT$_r$
pair uses discontinuous $P_r$ pressure; BDM$_r$ uses $P_{r-1}$. Rectangular
RT spaces have tensor-product divergence spaces. The 3D tetrahedral,
prismatic and mapped-hexahedral families use their declared divergence
and normal-trace spaces rather than transferring these triangular degree
labels. Skeletal normal degree cannot exceed the local normal degree,
and partitions must align. Interior enrichment can increase pressure or
flux accuracy without increasing boundary normal degree.

## Face-based and unfitted approximation

[Paredes, Valentin and Versieux (2024)](https://doi.org/10.1016/j.cam.2023.115415)
analyze refinement of independently partitioned faces, with continuous
interpolation within each macroface. The macro topology can remain fixed;
local approximation and the face approximation error are different terms.
Periodic homogenization robustness in
[Paredes, Valentin and Versieux (2017)](https://doi.org/10.1090/mcom/3108)
requires periodic coefficients and the stated scale/regularity relations;
it is not a contrast-independent theorem for arbitrary materials.

[Chaumont-Frelet, Paredes and Valentin (2026)](https://doi.org/10.1016/j.camwa.2026.01.016)
allow the macro mesh to cross material regions. Every skeletal subface must
remain inside one physical region. Theorem 2 assumes piecewise
$H^{\ell+3}$ regularity of pressure, piecewise $W^{\ell+1,\infty}$
regularity of diffusion and an $H(\mathrm{div})$ exact flux. Its bound has
skeletal factor $h_\Gamma^{\ell+3/2}$ for a fixed macro partition. The
macro diameter enters separately; this half-order gain is not a statement
for an unresolved cut or an arbitrary macro-refinement sequence.
See the [unfitted tutorial](../tutorials/methods/unfitted.md).

## Recovery and estimation

The [recovery and adaptivity theory](recovery.md) distinguishes RT0
minimum-energy equilibration from moment-based H(div) reconstruction,
conforming potential recovery and complete field-specific estimators.
The degree condition $k\ge\ell+d$, source/trace compatibility and the
estimator's independent subface tests belong to the stated theorem;
existence of a numerical reconstruction alone does not establish its bound.

## References

- Christopher Harder, Diego Paredes, and Frédéric Valentin (2013). *A family of Multiscale Hybrid-Mixed finite element methods for the Darcy equation with rough coefficients*, Journal of Computational Physics 245, 107–130. [DOI: 10.1016/j.jcp.2013.03.019](https://doi.org/10.1016/j.jcp.2013.03.019).

- Rodolfo Araya, Christopher Harder, Diego Paredes, and Frédéric Valentin (2013). *Multiscale Hybrid-Mixed Method*, SIAM Journal on Numerical Analysis 51(6), 3505–3531. [DOI: 10.1137/120888223](https://doi.org/10.1137/120888223).

- Diego Paredes, Frédéric Valentin, and Henrique M. Versieux (2017). *On the robustness of multiscale hybrid-mixed methods*, Mathematics of Computation 86(304), 525–548. [DOI: 10.1090/mcom/3108](https://doi.org/10.1090/mcom/3108).

- Omar Durán, Philippe R. B. Devloo, Sônia M. Gomes, and Frédéric Valentin (2019). *A multiscale hybrid method for Darcy’s problems using mixed finite element local solvers*, Computer Methods in Applied Mechanics and Engineering 354, 213–244. [DOI: 10.1016/j.cma.2019.05.013](https://doi.org/10.1016/j.cma.2019.05.013).

- Diego Paredes, Frédéric Valentin, and Henrique M. Versieux (2024). *Revisiting the robustness of the multiscale hybrid-mixed method: The face-based strategy*, Journal of Computational and Applied Mathematics 436, 115415. [DOI: 10.1016/j.cam.2023.115415](https://doi.org/10.1016/j.cam.2023.115415).

- Gabriel R. Barrenechea, Larissa Martins, Weslley Pereira, and Frédéric Valentin (2026). *An H(div; Ω)-Conforming Flux Reconstruction for the Multiscale Hybrid-Mixed Method*, Multiscale Modeling & Simulation 24(2), 399–428. [DOI: 10.1137/24M1673073](https://doi.org/10.1137/24M1673073).

- Théophile Chaumont-Frelet, Diego Paredes, and Frédéric Valentin (2026). *Flux approximation on unfitted meshes and application to multiscale hybrid-mixed methods*, Computers & Mathematics with Applications 209, 16–27. [DOI: 10.1016/j.camwa.2026.01.016](https://doi.org/10.1016/j.camwa.2026.01.016).
