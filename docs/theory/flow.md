# MHM for Stokes, Brinkman and Oseen

Vector velocity and scalar pressure use local mixed equations and a vector skeleton. The [Stokes–Brinkman tutorial](../tutorials/methods/stokes-brinkman.md) compares stable Taylor–Hood and residual-stabilized USFEM local spaces; the [Oseen tutorial](../tutorials/methods/oseen.md) adds prescribed convection.

## Stokes and Brinkman

The convention used by [Araya et al. (2017)](https://doi.org/10.1016/j.cma.2017.05.027) and [Araya et al. (2025)](https://doi.org/10.1137/24M1649368) is

$$
-\nu\Delta u+\Theta u+\nabla p=f,\qquad\nabla\cdot u=0.
$$

With $a_K(u,v)=\nu(\nabla u,\nabla v)_K+(\Theta u,v)_K$, the
local unknowns are velocity $u_K\in V_h(K)^d$ and pressure
$p_K\in Q_h(K)$. For a supplied vector pseudotraction, their equations
are

$$
\begin{aligned}
a_K(u,v)-(p,\nabla\cdot v)_K+
\langle\lambda_K,v\rangle_{\partial K}&=(f,v)_K,\\
-(q,\nabla\cdot u)_K&=0,\\
\lambda_K&=(-\nu\nabla u+pI)n_K.
\end{aligned}
$$

Velocity is prescribed on $\Gamma_D$ and pseudotraction on $\Gamma_N$.
The pressure test has been negated to match the symmetric saddle form
used below; this does not change incompressibility.

This is a pseudotraction associated with the vector Laplacian. Replacing the
gradient term by $2\nu(\varepsilon(u),\varepsilon(v))$ changes the natural
traction and the local kernel, even where the interior incompressible PDE can be
related. The two conventions should be named explicitly.

For $\Theta=0$, the grad-grad velocity kernel consists of $d$ translations.
For uniformly positive definite drag it is trivial. A semidefinite anisotropic
drag can retain some translations and must be analyzed through the actual local
operator. A constant pressure is **not** an independent local kernel of the
natural-traction mixed problem: it acts on velocity boundary traces. The global
pressure shift is represented jointly by $p\mapsto p+c$ and
$\lambda_F\mapsto\lambda_F+c n_F$.

When a pressure normalization is required, it is global. If $m_K^Tw_K$
integrates the local pressure, the
[hybrid reconstruction](foundations.md#local-matrices-and-condensation) induces

$$
\sum_Km_K^T\left(w_K^f-W_K\lambda_K+Z_Kc_K\right)=0.
$$

It is one scalar constraint on the assembled variables. It can be imposed in a
saddle system, with a compatibility multiplier that should vanish for compatible
data. Zero pressure mean imposed on every local problem without additional global
pressure unknowns changes the formulation.

### Global coupling and velocity modes

The global unknown is the vector pseudotraction $\lambda$, together
with $d$ translation coefficients per Stokes macrocell. Each local
Stokes response is computed with zero velocity moments against those
translations; its retained translation is reconstructed from the global
coefficients. Positive Brinkman drag determines the translations locally;
retaining them in the equivalent global decomposition is useful near the
Stokes limit and does not declare them to be exact kernel vectors.

The global coupling equations are

$$
\begin{aligned}
\sum_K\langle\mu_K,u_K\rangle_{\partial K}
 &=\langle\mu,u_D\rangle_{\Gamma_D},\\
a_K(u_K,r)-(p_K,\nabla\cdot r)_K
 +\langle\lambda_K,r\rangle_{\partial K}
 &=(f,r)_K,\qquad r\text{ retained on }K.
\end{aligned}
$$

The first condition matches velocity moments across macrofaces.
For Stokes translations, the second becomes force balance
$\int_{\partial K}\lambda_K=\int_Kf$ componentwise. For a stabilized
local operator, this retained-mode equation uses the full stabilized
bilinear form and load given below. Fixed natural coefficients are
excluded from the free trace tests. A constant pressure test gives
$\int_K\operatorname{div}u_K=0$ even in the residual-stabilized form.
Fully prescribed velocity requires
$\int_{\partial\Omega}u_D\cdot n=0$; when the boundary conditions leave
the pressure shift undetermined, the single global normalization fixes it.

### Local USFEM stabilization

Stable mixed local spaces, such as Taylor–Hood under the relevant mesh conditions,
and stabilized equal-order spaces are both possible. The USFEM construction in
[Araya et al. (2017)](https://doi.org/10.1016/j.cma.2017.05.027) uses the momentum residual
$R(u,p)=-\nu\Delta u+\Theta u+\nabla p$ and a matching test residual.
Its load stabilization must be included as well. Keeping only a pressure-gradient
penalty when Brinkman drag is present does not reproduce that formulation.
For piecewise linear fields, cellwise second derivatives vanish, but drag terms
in the residual remain.

With the pressure test negated, the symmetric USFEM form used by the reference
backend is

$$
\begin{aligned}
B_K((u,p),(v,q))={}&a_K(u,v)-(p,\nabla\cdot v)_K
-(q,\nabla\cdot u)_K\\
&-\sum_{\tau\subset K}\kappa_\tau(R(u,p),R(v,q))_\tau,\\
F_K(v,q)={}&(f,v)_K-\sum_{\tau\subset K}\kappa_\tau(f,R(v,q))_\tau.
\end{aligned}
$$

The source and trace responses therefore solve

$$
B_K((u_K,p_K),(v,q))+\langle\lambda_K,v\rangle_{\partial K}
 =F_K(v,q),
$$

with the same retained velocity moments and global coupling as above.

For constant scalar drag $\gamma\ge0$ and P1/P1 fields, the selected parameter is

$$
\kappa_\tau=\frac{h_\tau^2}
{\max(\gamma h_\tau^2,12\nu)+12\nu}.
$$

This is the stabilization parameter of [Araya et al. (2017)](https://doi.org/10.1016/j.cma.2017.05.027) with $m_\tau=1/3$, written without division by drag;
its Stokes limit is $h_\tau^2/(24\nu)$. The minus sign in the residual term
and the stabilized force are essential. The formula by itself is not a proof of
uniform Darcy-limit accuracy for every local/skeletal space combination.

For higher-order local fields the implementation evaluates their exact
Laplacians and computes the element inverse constant from a generalized
eigenproblem modulo constants. With $m_\tau=\min(1/3,C_\tau)$ satisfying
$C_\tau h_\tau^2\|\Delta v_h\|_0^2\le\|\nabla v_h\|_0^2$, it uses

$$
\kappa_\tau=\frac{h_\tau^2}
{\max(\theta_{\max,\tau}h_\tau^2,4\nu/m_\tau)+4\nu/m_\tau}.
$$

The largest resistance eigenvalue is the tensor bound in [Araya et al. (2025)](https://doi.org/10.1137/24M1649368). For a scalar
constant resistance this reduces to the parameter in [Araya et al. (2017)](https://doi.org/10.1016/j.cma.2017.05.027). The two papers state their inverse
inequalities using different squared/unsquared constant conventions; the
computed constant above is defined by the displayed squared inequality.
The local residual retains every component of $\Theta u$, including
cross-component coupling. Taylor–Hood uses Pk/P(k−1); USFEM uses Pk/Pk.

The API makes the material parameter convention explicit:

- `stabilization="tensor-2025"` uses the largest eigenvalue over the fine cell
  and is the default.
- `stabilization="minimum-2017"` uses the minimum-eigenvalue expression printed
  in [Araya et al. (2017)](https://doi.org/10.1016/j.cma.2017.05.027) with a declared global lower bound `gamma_min`.
- `stabilization="pointwise-2017"` evaluates that expression using the smallest
  eigenvalue at each integration point.

The global-bound and pointwise versions of the 2017 minimum-eigenvalue
expression are distinct for heterogeneous resistance. A global lower bound does not control
the negative squared-residual term in a much more resistant cell. For example,
the reaction part for a constant vector is
$\gamma_\tau(1-\kappa_\tau\gamma_\tau)|u|^2$, which becomes negative when
$\kappa_\tau\gamma_\tau>1$. An algebraic residual or macro balance alone cannot
verify stability in this situation.

`CartesianCellField` coefficients are integrated on the geometric
intersections of material pixels and finite elements. The inverse-inequality
constant is computed independently using polynomial-exact quadrature. Resolving
material integration does not enrich the polynomial approximation space.

The published macro mass property is $\int_K\nabla\cdot u_h=0$.
It does not imply pointwise incompressibility or an H(div)-conforming velocity
across macrofaces. Likewise, raw approximate pseudostress need not possess the
normal continuity of the skeletal multiplier.

## Oseen

For prescribed convection $\alpha$, [Araya et al. (2021)](https://doi.org/10.1007/s10444-020-09833-8) considers

$$
-\nu\Delta u+(\alpha\cdot\nabla)u+\gamma u+\nabla p=f,
\qquad\nabla\cdot u=0.
$$

The effective reaction in the skew form is
$\gamma_0=\gamma-\tfrac12\nabla\cdot\alpha$, with a strictly positive
lower bound in the stated coercivity hypothesis. This minus sign differs from
the conservative scalar RAD model because the strong convection operators are
written differently. The skeletal quantity is

$$
\lambda_K=\left(-\nu\nabla u+pI+
\tfrac12u\otimes\alpha\right)n_K.
$$

**Local problems** use the mixed momentum/divergence equations above,
replacing the velocity bilinear form by

$$
\begin{aligned}
a_K^{\mathrm O}(u,v)={}&\nu(\nabla u,\nabla v)_K
 +\tfrac12((\alpha\cdot\nabla)u,v)_K\\
&-\tfrac12(u,(\alpha\cdot\nabla)v)_K+(\gamma_0u,v)_K.
\end{aligned}
$$

Consistent stabilized locals add the full Oseen momentum residual to
both the operator and the source. **The global problem** couples the
half-advection pseudotraction through the same velocity-moment equation,
retained-mode equations and applicable pressure gauge. Thus the extra
convection changes the local operator and physical boundary quantity,
while the global test still measures velocity continuity. Prescribing
$(-\nu\nabla u+pI)n$ as though it were the Oseen multiplier would omit
the half-advection boundary term.

The pressure gauge and mixed-space compatibility remain necessary. A prescribed
convection solve may be a component of a time or nonlinear iteration, but a
time integrator or Navier–Stokes iteration is an additional numerical method.

The smooth degree-one trace comparison in
[Araya et al. (2021), Section 5.1](https://doi.org/10.1007/s10444-020-09833-8)
uses the product norm defined in Section 2.2:

$$
\begin{aligned}
\lVert(e_u,e_p)\rVert_{V\times Q}^2
={}&d_\Omega^{-2}\lVert e_u\rVert_{L^2(\Omega)}^2\\
&+\sum_K\lVert\nabla e_u\rVert_{L^2(K)}^2
+\lVert e_p\rVert_{L^2(\Omega)}^2.
\end{aligned}
$$

Here $d_\Omega$ is the domain diameter. The
[Oseen tutorial](../tutorials/methods/oseen.md) compares order two in this
observable on the same smooth space family. Its additional third-order
velocity $L^2$ measurement is reported separately; the product-norm estimate
alone does not establish that stronger velocity estimate.


## Space compatibility and rates

In [Araya et al. (2025)](https://doi.org/10.1137/24M1649368), Theorem 4.2
for equal-order stabilized locals gives a sufficient condition
$k-\ell\ge d$, with $\ell\ge0$ for discontinuous traces and
$\ell\ge1$ for continuous traces. Refined compatible local partitions
can relax this condition as described in Remark 4.4; the requirement is
still a trace lifting condition, not a choice of method name. The
Taylor–Hood analysis uses its separate local mesh/inf-sup hypotheses.

For local regularity $u\in H^{s+1}$ and $p\in H^s$, with
$1\le s\le\min(k,\ell+1)$, velocity energy/pressure errors contain
$h_\Gamma^s+h^s$. With one extra derivative, the face contribution gains
one half-order for a fixed macro partition, with macro-diameter dependence
stated separately in the theorem. Smooth velocity $L^2$ studies can gain
an additional order under the relevant dual regularity. A boundary-layer
study must first resolve the layer; an underresolved local mesh need not
achieve these targets or remove overshoots solely through stabilization.

The two-dimensional and tetrahedral three-dimensional operators have
separate material bounds, inverse inequalities and trace alignment checks.
For 3D P1 velocity and a P1 triangular trace, the implemented sufficient
choice uses local refinement four; higher local degrees use refinement two.
These finite configurations are not a general stability theorem.

[Araya, Rebolledo and Valentin (2021)](https://doi.org/10.1093/imanum/drz053)
combine coarse trace jumps with fine momentum, divergence and pseudotraction
residuals. [Araya et al. (2021)](https://doi.org/10.1007/s10444-020-09833-8)
use the corresponding Oseen residual and convection assumptions. Mesh
adaptation must include the actual estimator terms and stated boundary
scope; raw multiplier magnitude alone is not that estimator. See
[error indicators](../tutorials/methods/error-indicators.md) and
[adaptivity](../tutorials/methods/adaptivity.md).

## References

- Rodolfo Araya, Christopher Harder, Abner H. Poza, and Frédéric Valentin (2017). *Multiscale hybrid-mixed method for the Stokes and Brinkman equations—The method*, Computer Methods in Applied Mechanics and Engineering 324, 29–53. [DOI: 10.1016/j.cma.2017.05.027](https://doi.org/10.1016/j.cma.2017.05.027).

- Rodolfo Araya, Ramiro Rebolledo, and Frédéric Valentin (2021). *On a multiscale a posteriori error estimator for the Stokes and Brinkman equations*, IMA Journal of Numerical Analysis 41(1), 344–380. [DOI: 10.1093/imanum/drz053](https://doi.org/10.1093/imanum/drz053). An earlier version is [HAL: hal-01945934v1](https://hal.science/hal-01945934v1).

- Rodolfo Araya, Cristian Cárcamo, Abner H. Poza, and Frédéric Valentin (2021). *An adaptive multiscale hybrid-mixed method for the Oseen equations*, Advances in Computational Mathematics 47, article 15. [DOI: 10.1007/s10444-020-09833-8](https://doi.org/10.1007/s10444-020-09833-8).

- Rodolfo Araya, Christopher Harder, Abner H. Poza, and Frédéric Valentin (2025). *Multiscale Hybrid-Mixed Methods for the Stokes and Brinkman Equations—A Priori Analysis*, SIAM Journal on Numerical Analysis 63(2), 588–618. [DOI: 10.1137/24M1649368](https://doi.org/10.1137/24M1649368).
