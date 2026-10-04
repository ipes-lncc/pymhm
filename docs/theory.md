# Mathematical structure

This page states conventions and the mathematical obligations behind an MHM
implementation. Results from the [literature catalog](literature.md) apply under
their stated hypotheses; the existence of an assembly interface does not transfer
those results to an arbitrary local discretization.

## Three independent discretization scales

Let a bounded domain be partitioned into macroelements $K\in\mathcal P_H$.
Each macroface has its own subdivision and polynomial degree. Each macroelement
also has a local finite element mesh. We distinguish:

| Symbol | Meaning | What changing it does |
| --- | --- | --- |
| $H_K$ | Macroelement diameter | Changes the domain partition and global topology |
| $h_{\Gamma,F}$ | Subface diameter | Enlarges the skeletal space while preserving macro topology |
| $p_{\Gamma,F}$ | Subface polynomial degree | Changes the moments exchanged by adjacent macroelements |
| $h_K$ | Local finite element diameter | Resolves coefficients and local basis functions |
| $p_K$ | Local polynomial degree | Controls local approximation and compatibility |

The same letter is used differently in different articles. Numerical reports should
record all these quantities explicitly. A first-level face refinement is not a
global mesh refinement, and a fine local mesh does not repair an inadequate
skeletal space automatically.

Give each face $F$ one fixed unit normal $n_F$. Its orientation relative to the
outward normal of $K$ is $s_{KF}=n_K\cdot n_F\in\{-1,1\}$. If
$\lambda_F$ is the coefficient field in that orientation, the local outward
trace is $s_{KF}\lambda_F$. Scalar coefficients multiplying a polynomial basis
are not integrated fluxes unless that normalization is part of the basis
definition. Reversing the face parametrization also transforms odd polynomial
modes, independently of the normal sign.

## Darcy: primal local problems

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

## Local matrices and condensation

Write a local discrete hybrid equation as

$$
A_Kw_K+B_K\lambda_K=f_K.
$$

Here $B_K$ already contains geometry, quadrature, basis normalization, and the
orientation map. Let the columns of $Z_K$ span the kernel of $A_K$. A complementary
constraint $C_K^Tw_K=0$ defines the local inverse on a subspace. For a symmetric
problem, a constrained solve has the form

$$
\begin{bmatrix}A_K&C_K\\C_K^T&0\end{bmatrix}
\begin{bmatrix}w_K\\\eta_K\end{bmatrix}
=\begin{bmatrix}r_K\\0\end{bmatrix}.
$$

The constraints must remove exactly the local kernel; in particular
$C_K^TZ_K$ must be invertible. Write $L_Kr_K$ for the first block of this
solution and set $W_K=L_KB_K$, $w_K^f=L_Kf_K$. Reconstruction is

$$
w_K=w_K^f-W_K\lambda_K+Z_Kc_K.
$$

Assembling the trace equation gives

$$
\begin{bmatrix}S&-G\\-G^T&0\end{bmatrix}
\begin{bmatrix}\lambda\\c\end{bmatrix}
=\begin{bmatrix}b-g\\-F_0\end{bmatrix},
$$

where the assembled contributions are
$S_K=B_K^TW_K$, $G_K=B_K^TZ_K$, $b_K=B_K^Tw_K^f$, and
$F_{0,K}=Z_K^Tf_K$. The vector $g$ carries prescribed primal boundary moments.
This sign convention agrees with the Darcy equations above. Other literature
conventions may negate the multiplier; comparing matrices requires translating
that convention first.

For a nonsymmetric local operator, $S$ need not be symmetric and left and right
kernels need not coincide. The compatibility equation uses the **left** kernel.
The symmetric saddle form is valid only when its assumptions hold. Neither a
Cholesky factorization nor conjugate gradients is a general solver for the coupled
MHM saddle system. Reuse a factorization for all trace columns and source right-hand
sides; do not form an explicit inverse.

### Retaining nearly null local modes

A small positive reaction or Brinkman drag removes an exact local kernel, but
directly inverting its weakly controlled modes is numerically unsafe. Source and
trace responses can each grow as the reciprocal coefficient, then lose digits
when subtracted. Retaining these modes globally avoids that cancellation without
changing the variational operator.

Let $Z$ now contain retained modes, not necessarily null vectors. Choose $C$ so
that $C^TZ$ is invertible and the augmented local matrix is nonsingular. Let $R$
be the first block of the constrained solve with that matrix, and define

$$
F=Rf,\qquad L=RB,\qquad E=Z-RAZ.
$$

The reconstruction and cell contributions to the global equations are

$$
u=F-L\lambda+Ec,
$$

$$
\begin{bmatrix}
B^TL&-B^TE\\
-Z^T(B-AL)&-Z^TAE
\end{bmatrix}
\begin{bmatrix}\lambda\\c\end{bmatrix}
=
\begin{bmatrix}B^TF-g\\Z^TAF-Z^Tf\end{bmatrix}.
$$

The lower block is the original local equation tested against $Z$ with an
overall minus sign. The complementary equation leaves a residual in the range
of $C$; nonsingularity of $Z^TC$ then makes that residual zero. This establishes
equivalence to the uncondensed equations. For a common left/right kernel,
$AZ=A^TZ=0$, these expressions reduce to the preceding kernel formulation.
For nonsymmetric operators the lower-left block must be computed as written;
it is generally not the transpose of the upper-right block.

Since $C^TE=C^TZ$, the coefficients $c$ still encode the retained physical
moments. An integral constraint with weights $m$ must nevertheless use the
reconstructed field, namely $m^TF-m^TL\lambda+m^TEc$. Replacing $E$ by $Z$ here
would impose the wrong pressure mean when retained velocity responses contain
pressure components. In a nonsymmetric global system, the gauge augmentation
also requires constraints that pair with both its left and right nullspaces.

The implementation exposes this decomposition through `coarse_basis`, separately
from the checked `kernel` declaration. Scalar reaction/advection and heat retain
constants; Brinkman and Oseen retain velocity translations. Tests compare
symmetric positive-definite, indefinite and nonsymmetric local operators with
independent full saddle solves, and exercise reaction/drag down to $10^{-16}$
and heat steps up to $10^{16}$.

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
pressure and is only the first member of the mixed family in L05.

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
edges. It is not the face-moment reconstruction analyzed in L09: its defining
constraints enforce fine-cell source moments explicitly.

## Stokes and Brinkman

The convention in L13 and L16 is

$$
-\nu\Delta u+\Theta u+\nabla p=f,\qquad\nabla\cdot u=0.
$$

With $a_K(u,v)=\nu(\nabla u,\nabla v)_K+(\Theta u,v)_K$, the
momentum equation is

$$
a_K(u,v)-(p,\nabla\cdot v)_K+
\langle\lambda_K,v\rangle_{\partial K}=(f,v)_K,
\qquad\lambda_K=(-\nu\nabla u+pI)n_K.
$$

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

Pressure normalization is global. If $m_K^Tw_K$ integrates the local pressure,
the reconstruction above induces

$$
\sum_Km_K^T\left(w_K^f-W_K\lambda_K+Z_Kc_K\right)=0.
$$

It is one scalar constraint on the assembled variables. It can be imposed in a
saddle system, with a compatibility multiplier that should vanish for compatible
data. Zero pressure mean imposed on every local problem without additional global
pressure unknowns changes the formulation.

Stable mixed local spaces, such as Taylor–Hood under the relevant mesh conditions,
and stabilized equal-order spaces are both possible. The USFEM construction in
L13 uses the momentum residual
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

For constant scalar drag $\gamma\ge0$ and P1/P1 fields, the selected parameter is

$$
\kappa_\tau=\frac{h_\tau^2}
{\max(\gamma h_\tau^2,12\nu)+12\nu}.
$$

This is the L13 parameter with $m_\tau=1/3$, written without division by drag;
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

The largest resistance eigenvalue is the tensor bound in L16. For a scalar
constant resistance this reduces to L13. The two papers state their inverse
inequalities using different squared/unsquared constant conventions; the
computed constant above is defined by the displayed squared inequality.
The local residual retains every component of $\Theta u$, including
cross-component coupling. Taylor–Hood uses Pk/P(k−1); USFEM uses Pk/Pk.

The API makes the material parameter convention explicit:

- `stabilization="tensor-2025"` uses the largest eigenvalue over the fine cell
  and is the default.
- `stabilization="minimum-2017"` uses the minimum-eigenvalue expression printed
  in L13 with a declared global lower bound `gamma_min`.
- `stabilization="pointwise-2017"` evaluates that expression using the smallest
  eigenvalue at each integration point.

L13 does not specify how its minimum is evaluated spatially for a heterogeneous
coefficient. These options expose that distinction; their names do not establish
which was used in the historical program. A global lower bound does not control
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

## Reactive–advective–diffusive equations

The conservative model in L11–L12 is

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

L12 defines the coarse constant space from the kernel of $a_K$ on each cell.
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

For $\partial_tu-\nabla\cdot(\mathsf K\nabla u)=f$, `solve_heat` applies
backward Euler. A step of length $\Delta t$ replaces each local operator by
$A_K+M_K/\Delta t$ and its source by
$f_K^{n+1}+M_Ku_K^n/\Delta t$. Positive mass removes the scalar local constant
kernel. The skeletal trace still represents outward diffusive flux.
The constant is retained as a coarse mode, so large time steps do not require
inverting an almost singular Neumann operator.

Spatial approximation and first-order temporal error must be assessed separately.
The current reference implementation reuses assembled diffusion/mass matrices,
but refactorizes for each step. It supports time-independent diffusion and
time-dependent full Dirichlet/source data; this is a separate extension of the
steady diffusion building blocks, not a reproduction of an unlisted transient
benchmark.

## Oseen

For prescribed convection $\alpha$, L15 considers

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

The pressure gauge and mixed-space compatibility remain necessary. A prescribed
convection solve may be a component of a time or nonlinear iteration, but a
time integrator or Navier–Stokes iteration is an additional numerical method.

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
force and moment balance. With the generic positive boundary term used above,
the multiplier is $\lambda_K=-\tau n_K$. Some elasticity papers use
$+\tau n_K$ and reverse the boundary signs; both conventions are valid when
used consistently.

The mixed weak-symmetry formulation of L18 uses stress in a tensor H(div) space,
displacement in a discontinuous space, and a rotation multiplier. In two dimensions,
the weak symmetry equation tests $\tau_{12}-\tau_{21}$ against the rotation
space. It enforces stress symmetry in moments, not necessarily pointwise.

For isotropic nearly incompressible elasticity, let $G$ be the shear modulus and
$\lambda_L$ the first Lamé coefficient. The Herrmann-pressure version is

$$
\tau=2G\varepsilon(u)-pI,\qquad
\nabla\cdot u+\lambda_L^{-1}p=0.
$$

L19 combines this local mixed form with consistent least-squares stabilization and
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

Equal-order Pk/Pk fields are restricted only by physical displacement moments
against the three rigid modes; there is no local pressure gauge. For variable
shear modulus, $R$ includes $2\varepsilon(u)\nabla G$. The API requires its
derivative and material bounds used in the stabilization inequality. The
inverse constant is computed without an exact solution or measured error.
In two dimensions, a linear traction segment needs four fine intervals for
P1, two for P2 and one for P3; the implemented sufficient conditions follow
L19 Lemma 4.5 and reject incompatible trace/local choices before elimination.

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
The [elasticity cases](https://github.com/volpatto/pymhm/blob/main/docs/cases/elasticity.md) and
[MSL comparison](https://github.com/volpatto/pymhm/blob/main/docs/cases/elasticity-reference.md) provide non-affine verification.

### Implemented weak-symmetry stress method

The two-dimensional stress solver uses two BDM2 rows, discontinuous P1
vector displacement and discontinuous P1 scalar rotation. All local rigid
motions are represented exactly; the rotation component of a rigid mode is
retained together with its displacement. Physical traction moments restrict
the macro boundary. Exterior faces may carry higher-resolution tractions than
interior faces, following L18's construction. Compliance is evaluated through
spherical and deviatoric parts to avoid subtracting nearly equal material
coefficients. It supports heterogeneous isotropic Lamé fields and zero
spherical compliance at infinite first Lamé modulus. For a fully prescribed
displacement boundary, the integrated constitutive identity fixes the finite
hydrostatic stress; at infinite modulus, one global mean of negative half
the stress trace fixes its pressure gauge.

Weak symmetry, divergence moments and normal-traction agreement are measured
separately. P1 force projection is enforced in every fine cell, whereas the
pointwise divergence error against a non-polynomial force remains nonzero.
See [mixed elasticity](https://github.com/volpatto/pymhm/blob/main/docs/cases/mixed-elasticity.md) for the oscillatory-modulus
example, convergence, material sweep and independent finite-element comparison.

## Stability and error assessment

There are at least three distinct compatibility requirements:

1. The local operator must be invertible on the constrained complement; mixed
   local spaces must satisfy their inf-sup or stabilization assumptions.
2. The local test traces must detect the skeletal space. A trace mode invisible
   to every local test function is an unstable global degree of freedom.
3. The skeletal space must couple the remaining coarse kernel modes.

Increasing face degree or adding subfaces can violate the second requirement if
the local space is unchanged. A small numerical residual of a singular or
regularized system does not establish stability. Rank checks on coupling blocks,
nullspace tests, and convergence under independent refinements address different
parts of this issue.

The implementation tests each declared kernel mode with relative operator
scaling. Before a global solve, it also checks the equilibrated free/gauged system
for small LU pivots at working precision. This catches unresolved trace modes
even when a compatible manufactured load gives a small residual. The check
diagnoses numerical rank loss or severe ill-conditioning; it is not a proof of a
mesh-uniform inf-sup constant. Optional global solver backends incur a separate
CPU diagnostic factorization, while the SciPy path reuses its factorization.

For smooth problems, two-level error bounds have the schematic structure

$$
\text{error}\ \lesssim\
\text{trace approximation error}+
\text{local discretization error}+
\text{algebraic error}.
$$

The powers, norms, and constants depend on the problem and hypotheses. They may
also depend on contrast, viscosity, regularity, or cell geometry. A fixed fine
mesh eventually creates an error floor during a skeletal convergence study.
Likewise, a coefficient sampled once per cell can change the PDE independently
of finite element interpolation error.

For the unfitted analysis in L10, the macro mesh may cross material interfaces,
but every skeletal subface must belong to a single material region. Its extra
half-order is a face-refinement statement with suitable physical-region
regularity and sufficiently accurate local solves. It is not a blanket guarantee
for arbitrary unresolved cuts.

## Adaptivity and parallel execution

The multilevel estimators in L14–L15 separate a skeletal contribution from local
residual contributions. A practical adaptive step marks subfaces, assesses
whether neighboring local errors dominate, updates affected local bases, and
reassembles their contributions. The macro topology can remain fixed. A generic
jump indicator is useful experimentally, but should not be called the published
reliable estimator without all required terms and scaling.

Local factorizations, multiple trace/source right-hand sides, and reconstruction
are independent across macroelements. Assembly and the global solve remain
coupled. Effective implementations distinguish local assembly, factorization,
triangular solves, global assembly, global solution, and reconstruction in their
timings. Reuse is valid while geometry, coefficients, spaces, and stabilization
are unchanged; changing only the source can reuse trace bases.

CPU workers, MPI ranks, and GPU batches exploit different levels of this
independence. The mathematical decomposition permits these strategies but does
not guarantee speedup. Scheduling, thread oversubscription, data transfer,
factorization memory, and the global system determine observed performance.
