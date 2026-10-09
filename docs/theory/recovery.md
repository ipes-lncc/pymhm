# Flux recovery, error indicators and adaptive approximation

Recovery changes a derived field; estimation measures a solution defect;
adaptation chooses the next approximation space. These operations complement
a discretization and do not carry one universal convergence power of their
own. Their applicable rates and bounds depend on the underlying operator,
spaces, material regularity, boundaries and quadrature.

## RT0 minimum-energy equilibration

For primal Darcy pressure $p_h$, `equilibrate_flux` seeks an RT0 flux nearest
to $-A\nabla p_h$ in the $A^{-1}$-weighted norm. It constrains divergence to
the fine-cell source average and fixes the macro boundary normal flux from
the MHM multiplier. These constraints require compatible source/trace
balance and piecewise constant trace segments aligned with fine boundary
edges. The result satisfies the stated fine-cell balances; the raw gradient
and original pressure remain independent fields.

This implementation is a constrained energy minimization. Its defining
constraints differ from the moment reconstruction of
[Barrenechea et al. (2026)](https://doi.org/10.1137/24M1673073), whose theorem
must not be assigned to it simply because both return an RT field.
The [flux-recovery tutorial](../tutorials/methods/flux-recovery.md) derives
both constructions before using their convenience APIs.

## Moment-based H(div) reconstruction

The reconstruction of
[Barrenechea et al. (2026)](https://doi.org/10.1137/24M1673073) assigns
macro-boundary RT moments from the skeletal multiplier, interior fine-face
moments from averaged raw fluxes, and interior moments from the pressure
field. Oriented normal moments produce a globally $H(\mathrm{div})$ field
and preserve its projected source divergence under the stated conditions.

Let $k$ be primal local degree, $\ell$ skeletal normal degree and $m$ RT
reconstruction order in spatial dimension $d$. The theorem's sufficient
polynomial requirements are

$$
k\ge\ell+d,\qquad \ell\le m\le k.
$$

Constant traces therefore need local P2 in 2D and P3 in 3D; linear traces
need P3 and P4. Trace subdivisions align with fine boundary facets. An
algebraic reconstruction can exist under weaker conditions, but that fact
does not establish the cited flux estimate or estimator efficiency.

The estimator analysis also assumes independent polynomial tests on each
skeletal subface. Continuous interpolation across several subfaces removes
those independent constant tests. Reconstruction can still be meaningful
on that space, but the implemented theorem-based estimator rejects it.
A single-subface continuous face spans the same polynomial space as its
discontinuous counterpart.

## Conforming potential and energy estimates

On a globally conforming union of local fine meshes, the Oswald potential
$s_h$ averages the broken pressure at shared Lagrange nodes, counting each
incident fine element. Its boundary trace represents prescribed Dirichlet
data. It is distinct from the original MHM pressure and from a smoothed
plot of that pressure.

With reconstructed flux $q_h^R$, a material-weighted estimator includes
the flux mismatch and nonconformity terms

$$
\begin{aligned}
\eta_{1,K}&=\lVert A^{-1/2}(A\nabla p_h+q_h^R)\rVert_K,\\
\eta_{2,K}&=\lVert A^{1/2}\nabla(p_h-s_h)\rVert_K.
\end{aligned}
$$

Projected-divergence defect and source oscillation are separate required
terms. On convex cells their Poincaré factor is
$H_K/(\pi\sqrt{\alpha_K})$, for a certified lower diffusion eigenvalue
$\alpha_K$. Mixed-boundary versions must represent both the prescribed
potential and natural flux; unmeasured boundary errors cannot be omitted.
Point wells do not satisfy the $L^2$-source assumption of this estimate.

For general material, the printed unweighted indicator and the physical
energy-normalized estimator have different coefficient scalings. The
[error-indicator tutorial](../tutorials/methods/error-indicators.md)
distinguishes these conventions. The code integrates floating-point
quadrature estimates; it does not supply interval-certified integrals or
an unconditional coefficient-independent bound.

## Residual indicators for other equations

[Araya et al. (2013)](https://doi.org/10.1137/120888223) analyze elliptic
face residuals. [Araya, Rebolledo and Valentin (2021)](https://doi.org/10.1093/imanum/drz053)
combine coarse trace jumps with fine momentum, divergence and pseudotraction
residuals for flow. [Araya et al. (2021)](https://doi.org/10.1007/s10444-020-09833-8)
include Oseen convection; [Harder, Paredes and Valentin (2015)](https://doi.org/10.1137/130938499)
provide the conservative RAD face construction. Primal elasticity retains
traction and displacement residual conventions from
[Harder, Madureira and Valentin (2016)](https://doi.org/10.1051/m2an/2015046).

These are separate equation-specific estimators. The selected variant must
match its physical operator, boundary data, material weights, local residual
terms and geometry. An experimental jump indicator is useful without
being the full reliable estimator of one of these papers.

## Adaptation: macro, face, local mesh and polynomial degree

[Dörfler (1996)](https://doi.org/10.1137/0733054) introduces bulk marking:
choose cells or faces whose squared indicators account for a prescribed
fraction of the total. Marking does not determine which scale to refine.
MHM can change macro topology, subdivide faces, increase polynomial degree,
or refine local meshes while leaving the macro partition fixed.

The two-level flow algorithms of
[Araya, Rebolledo and Valentin (2021)](https://doi.org/10.1093/imanum/drz053)
separate face and local defects. Darcy energy-based policies likewise
compare local and macro contributions. Increasing face resolution without
enlarging its local trace lifting can introduce undetected modes; every
new space must satisfy the formulation's compatibility conditions.

Conforming closure must preserve face/boundary ancestry, material tags and
physical gauges. Longest-edge refinement follows
[Rivara (1984)](https://doi.org/10.1002/nme.1620200412); red/green closure
has its own mesh-quality contract. A useful adaptive experiment reports
error against degrees of freedom, estimator components, effectivity and
mesh quality. A monotone error reduction or optimal-complexity theorem is
not asserted for every implemented marking/refinement policy.
The [adaptivity tutorial](../tutorials/methods/adaptivity.md) follows
solve → estimate → mark → refine → rebuild compatible spaces → solve.

## References

- Gabriel R. Barrenechea, Larissa Martins, Weslley Pereira, and Frédéric Valentin (2026). *An H(div; Ω)-Conforming Flux Reconstruction for the Multiscale Hybrid-Mixed Method*. Multiscale Modeling & Simulation 24(2), 399–428. [DOI](https://doi.org/10.1137/24M1673073).
- Rodolfo Araya, Christopher Harder, Diego Paredes, and Frédéric Valentin (2013). *Multiscale Hybrid-Mixed Method*. SIAM Journal on Numerical Analysis 51(6), 3505–3531. [DOI](https://doi.org/10.1137/120888223).
- Rodolfo Araya, Ramiro Rebolledo, and Frédéric Valentin (2021). *On a multiscale a posteriori error estimator for the Stokes and Brinkman equations*. IMA Journal of Numerical Analysis 41(1), 344–380. [DOI](https://doi.org/10.1093/imanum/drz053).
- Rodolfo Araya, Cristian Cárcamo, Abner H. Poza, and Frédéric Valentin (2021). *An adaptive multiscale hybrid-mixed method for the Oseen equations*. Advances in Computational Mathematics 47, 15. [DOI](https://doi.org/10.1007/s10444-020-09833-8).
- Christopher Harder, Diego Paredes, and Frédéric Valentin (2015). *On a Multiscale Hybrid-Mixed Method for Advective-Reactive Dominated Problems with Heterogeneous Coefficients*. Multiscale Modeling & Simulation 13(2), 491–518. [DOI](https://doi.org/10.1137/130938499).
- Christopher Harder, Alexandre L. Madureira, and Frédéric Valentin (2016). *A hybrid-mixed method for elasticity*. ESAIM: M2AN 50, 311–336. [DOI](https://doi.org/10.1051/m2an/2015046).
- Willy Dörfler (1996). *A Convergent Adaptive Algorithm for Poisson’s Equation*. SIAM Journal on Numerical Analysis 33(3), 1106–1124. [DOI](https://doi.org/10.1137/0733054).
- M. Cecilia Rivara (1984). *Algorithms for refining triangular grids suitable for adaptive and multigrid techniques*. International Journal for Numerical Methods in Engineering 20(4), 745–756. [DOI](https://doi.org/10.1002/nme.1620200412).
