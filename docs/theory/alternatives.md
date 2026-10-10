# Robin, three-field, moment and Petrov–Galerkin methods

These families share local elimination and global assembly, but use different
unknowns and equations. An interchangeable local solver can assemble each
declared variational operator; it cannot convert one family into another by
changing a backend name.

## MH: coercive Robin local problems

[Barrenechea, Gomes and Paredes (2024)](https://doi.org/10.1137/22M1542556)
introduce a Multiscale Hybrid method with elliptic local problems and an
elliptic global Dirichlet operator. For Darcy flux $q=-A\nabla p$, use

$$
\begin{aligned}
\sigma(x)&=\frac{\nu}{d}(x-a),&\operatorname{div}\sigma&=\nu,\\
\lambda_K&=(q-p\sigma)\cdot n_K,\\
a_K(p,v)&=(A\nabla p,\nabla v)_K
 +\langle(\sigma\cdot n_K)p,v\rangle_{\partial K}.
\end{aligned}
$$

**Local problem.** For each source or Robin trace basis, find
$p_K\in V_h(K)$ from

$$
a_K(p_K,v)+\langle\lambda_K,v\rangle_{\partial K}=(f,v)_K,
\qquad v\in V_h(K).
$$

**Global problem.** The only interior hybrid unknown is $\lambda$.
Determine it by the pressure moment equations

$$
\sum_K\langle\mu_K,p_K(\lambda)\rangle_{\partial K}
 =\langle\mu,p_D\rangle_{\Gamma_D}.
$$

The Robin correction has opposite normal signs on the two sides of a
shared face; pressure continuity therefore also gives physical normal
flux continuity in the represented moments. On $\Gamma_N$ the physical
condition is $\lambda_K+p_K\sigma\cdot n_K=q_N$, rather than
$\lambda_K=q_N$. Its boundary pressure moments form the additional
unknowns of the natural-boundary extension.

Unlike kernel-constrained MHM, this coercive Robin problem does not remove a
constant mean locally. The global unknown is the Robin multiplier. The
physical normal flux is $q\cdot n_K=\lambda_K+p\sigma\cdot n_K$, so its
macro balance includes the pressure correction. The raw gradient is still a
broken field, without automatic fine-cell conservation.

The positive parameter must obey a sufficient coercivity bound. In the
implemented geometric convention, with
$C_\sigma=\max_{x\in\Omega}\lvert x-a\rvert/d$ and certified lower
eigenvalue $A_{\min}$, use

$$
0<\nu\le\frac{A_{\min}}{4C_\sigma^2}.
$$

Local/skeletal trace compatibility remains necessary. Corollary 2.7 gives
broken energy and multiplier errors of order
$h_\Gamma^{\ell+1}+h^k$ under the stated piecewise regularity. The paper's
smooth experiments use $k=\ell+2$ and display the additional pressure order;
this pressure order requires the appropriate dual regularity. The
[MH tutorial](../tutorials/methods/robin-mh.md) follows both equations.
Two-dimensional triangles and polygons, and a separate tetrahedral path,
are implemented. Mixed and pure-Neumann boundary extensions use physical
fluxes and a global pressure gauge; their augmented global matrix is not the
positive-definite Dirichlet matrix analyzed in the original construction.

## MH²M: independent pressure and conormal traces

[de Barros, Madureira and Valentin (2026, version 3)](https://arxiv.org/abs/2404.16978v3)
define a three-field method with a globally continuous pressure trace
$\rho\in\Gamma$ and a separate outward conormal trace
$\lambda_K=A\nabla p\cdot n_K=-q\cdot n_K\in\Lambda$ on each macrocell.
The local complement has zero **boundary** mean. The two Neumann maps satisfy

$$
\begin{aligned}
a_K(T_h\mu,v)&=\langle\mu,v\rangle_{\partial K},\\
a_K(\widetilde T_hf,v)&=(f,v)_K,\\
\langle\mu,T_hG_h\rho\rangle_{\partial K}
 &=\langle\mu,\rho\rangle_{\partial K}.
\end{aligned}
$$

**Local problems.** The cell field is reconstructed from these Neumann
maps and the selected pressure-trace values. The conormal coefficients
on the two sides of a face are independent local variables before the
global coupling; this differs from the signed single normal-flux field
of primal MHM.

**Global problem.** Before elimination, the three fields satisfy

$$
\begin{aligned}
a_K(p_K,v)-\langle\lambda_K,v\rangle_{\partial K}
 &=(f,v)_K,\\
\langle\mu_K,p_K-\rho\rangle_{\partial K}&=0,\\
\sum_K\langle\lambda_K,\xi\rangle_{\partial K}&=0.
\end{aligned}
$$

The first equation is local equilibrium. The second matches the cell
pressure to the global pressure trace in conormal moments. The third
imposes conormal continuity against globally continuous trace tests
$\xi$ vanishing on the Dirichlet boundary. Thus $\rho$ is the global
unknown after local elimination. Dirichlet data prescribe the pressure
trace interpolation on $\Gamma_D$; natural data prescribe conormal
moments of $-q_N$ on $\Gamma_N$ and enter the third equation's
right-hand side.

The pressure reconstruction includes its source lifting:

$$
p_h\big\vert_K=p_K^0+T_hG_h\rho
 +(I-T_hG_h)\widetilde T_hf.
$$

Eliminating conormal coefficients produces a pressure-trace system. It is
positive definite after Dirichlet elimination under the discrete stability
assumptions. A pure-Neumann system retains the physical constant mode and
uses a volume mean of the full reconstructed pressure.

For $\Gamma=P_{r+1}$, $\Lambda=P_r$ and $V_h=P_{r+1}$, Theorem 19 bounds
volume energy and pressure/conormal trace errors by contributions of order
$h_\Gamma^{r+1}$, $h_\Lambda^{r+1}$ and $h^{r+1}$. It assumes piecewise
$H^{r+2}$ regularity of pressure and the source lifting, piecewise
$H^{r+1}$ conormal regularity, and the Fortin conditions. Sufficient mesh
conditions put at least two local boundary edges inside every conormal
segment and at least two conormal segments inside every pressure-trace
segment. The exceptions in Remark 16 concern specified families, not every
choice passing a numerical rank check. See the
[MH²M tutorial](../tutorials/methods/mh2m.md). Triangular/polygonal 2D and
tetrahedral 3D implementations have separate evidence.

## MsHHO: cell and face moments

[Cicuttin, Ern and Lemaire (2019)](https://doi.org/10.1515/cmam-2018-0013) introduced the multiscale Hybrid High-Order construction for highly oscillatory elliptic operators.
[Chaumont-Frelet, Ern, Lemaire and Valentin (2022)](https://doi.org/10.1051/m2an/2021082)
connect MHM to the Multiscale Hybrid High-Order method. Let cell moments
belong to $P_m(K)$ and face moments to $P_\ell(F)$. The reconstructed local
function minimizes

$$
\frac12(A\nabla v,\nabla v)_K
$$

subject to those moments. Its Euler–Lagrange equations define a constrained
local energy reconstruction; the reconstructed basis is generally
nonpolynomial for heterogeneous $A$. Condensing cell coefficients gives a
global system of face **pressure** moments, distinct from the normal-flux
coordinates of MHM.

**Local problem.** Collect the volume and face moment functionals in
$M_K$. If $\boldsymbol u_K$ contains their values, the reconstruction
$R_K\boldsymbol u_K$ and its multiplier $\zeta_K$ solve

$$
\begin{aligned}
a_K(R_K\boldsymbol u_K,v)
 +\zeta_K^TM_Kv&=0,\qquad v\in V_h(K),\\
M_K(R_K\boldsymbol u_K)&=\boldsymbol u_K.
\end{aligned}
$$

Here $a_K(v,w)=(A\nabla v,\nabla w)_K$. The moment map must be
surjective and fix the constant mode; independent cell/face data cannot
be represented by an inadequate local space. Moment coordinates are
integrals in PyMHM, so converting polynomial coefficients to moments
uses the corresponding volume or face Gram matrix.

**Global problem.** Share one pressure-moment vector on every macroface
and find cell and face moments satisfying

$$
\sum_K a_K(R_K\boldsymbol u_K,R_K\boldsymbol v_K)
 =\sum_K F_K(\boldsymbol v_K)
  -\langle q_N,v_F\rangle_{\Gamma_N}.
$$

The test moments vanish on fixed Dirichlet faces; $v_F$ is the face
polynomial represented by those moments. For the reconstructed-source
variant, $F_K(\boldsymbol v_K)=(f,R_K\boldsymbol v_K)_K$;
for the projected-source variant, it loads the cell polynomial moments.
Cell moments are condensed locally, leaving the shared face system.
Dirichlet moments integrate $p_D$ against the face basis. Pure Neumann
data require mass compatibility and one mean of the reconstructed
pressure, rather than a mean of the face vector alone.

The reconstructed-source variant loads the energy-reconstructed test
functions. The projected-source variant uses the cellwise $L^2$ projection
onto $P_m$. Theorem 5.1's MHM equivalence requires the stated polynomial
source condition for the semi-explicit construction; the projected variant
extends it to square-integrable sources. The face-only choice $m=-1$ is an
explicit exception.

For regular data and admissible degree choices, the $m=\ell=0$ family has
energy/physical-flux order one and pressure order two; $m=\ell=1$ has orders
two and three. Local discretization adds its own error. The
[MsHHO tutorial](../tutorials/methods/mshho.md) declares moments and the
constrained reconstruction before assembly. Triangular/polygonal 2D and
tetrahedral/polyhedral 3D evidence is stated separately.

## PGMHM: residual enrichment of global equations

[Fernando, Martins, Pereira and Valentin (2023)](https://doi.org/10.1007/s40314-023-02304-y)
enrich skeletal trial functions with face residuals and alter the global
test equations. With the physical-flux convention, pressure jump $J_Fp$,
Dirichlet datum $g_F$ and full macroface diameter $H_F$, define

$$
\tau_F=\frac{\alpha A_{\min}}{2H_F},\qquad
\lambda_R=-\tau_F(J_Fp-g_F).
$$

**Local problems.** First compute the ordinary mean-zero Darcy source
and trace responses. After coupling, reconstruct an additional
mean-zero pressure $p_K^R$ from

$$
a_K(p_K^R,v)=-\langle\lambda_{R,K},v\rangle_{\partial K},
\qquad \int_Kv=0.
$$

**Global problem.** Let $z=(\lambda,\{c_K\})$ contain the base trace
and pressure means, and write the reconstructed base-pressure jump as
$J_Fp^0(z)=D_Fz+j_F^f$. If $\mathcal K_0z=b_0$ denotes the original
MHM equations, the residual-enriched equations are

$$
\mathcal K_0z
 +\sum_F D_F^T\tau_F(D_Fz+j_F^f-g_F)=b_0.
$$

Face integrals and quadrature weights are understood in each contraction.
The sum contains interior and Dirichlet faces; natural-flux faces have
fixed normal data and no residual enrichment. On Dirichlet faces $g_F$
uses the local polynomial projection of the boundary datum. Both its
base boundary functional and its residual term use the same physical
partition. The final pressure is $p^0+p^R$ and the final multiplier is
$\lambda+\lambda_R$. A pure-Neumann problem additionally fixes the
physical mean pressure.

The global equations add the associated positive jump form, and the local
pressure correction solves a zero-mean Neumann problem driven by
$\lambda_R$. The enriched multiplier $\lambda+\lambda_R$ satisfies macro
balance. Neither the base nor enriched raw gradient becomes an
$H(\mathrm{div})$ field automatically.

Published stability requires $k\ge\ell+d$, $\ell\ge1$ and sufficiently
small positive $\alpha$. Its approximation theorem also requires regular
pressure and conormal fields. The numerical $\ell=0$ cases lie outside
that stated stability proof. Smooth pressure/flux rates approach
$\ell+2$/$\ell+1$ in the recorded families, while the enriched divergence
error has a separate order and can stagnate for the lowest family. The
[PGMHM tutorial](../tutorials/methods/pgmhm.md) keeps these field norms
separate. The implemented PGMHM path is two-dimensional on triangles and
polygons; polytopal analysis does not imply an implemented 3D solver.

## References

- Matteo Cicuttin, Alexandre Ern and Simon Lemaire (2019). *A Hybrid High-Order Method for Highly Oscillatory Elliptic Problems*, Computational Methods in Applied Mathematics 19(4), 723–748. [DOI: 10.1515/cmam-2018-0013](https://doi.org/10.1515/cmam-2018-0013).
- Théophile Chaumont-Frelet, Alexandre Ern, Simon Lemaire, and Frédéric Valentin (2022). *Bridging the Multiscale Hybrid-Mixed and Multiscale Hybrid High-Order Methods*, ESAIM: M2AN 56, 261–285. [DOI: 10.1051/m2an/2021082](https://doi.org/10.1051/m2an/2021082).

- Honório Fernando, Larissa Martins, Weslley Pereira, and Frédéric Valentin (2023). *A Petrov–Galerkin multiscale hybrid-mixed method for the Darcy equation on polytopes*. Computational and Applied Mathematics 42, article 173. [DOI: 10.1007/s40314-023-02304-y](https://doi.org/10.1007/s40314-023-02304-y).

- Gabriel R. Barrenechea, Antonio Tadeu A. Gomes, and Diego Paredes (2024). *A Multiscale Hybrid Method*. SIAM Journal on Scientific Computing 46(3), A1628–A1657. [DOI: 10.1137/22M1542556](https://doi.org/10.1137/22M1542556).

- Franklin de Barros, Alexandre L. Madureira, and Frédéric Valentin (2026). *A three-field Multiscale Method*. arXiv preprint, version 3, 5 August 2026; first submitted 25 April 2024. [arXiv: 2404.16978v3](https://arxiv.org/abs/2404.16978v3).
