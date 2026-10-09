# Hybrid formulation and local/global decomposition

The common structure follows [Harder and Valentin (2016)](https://doi.org/10.1007/978-3-319-41640-3_13). Define the mesh hierarchy, write the local variational equations, declare their interface pairing and retained physical moments, then assemble and solve the global problem. The [API overview](../tutorials/overview.md) follows this order in code.

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

## Local matrices and condensation

The decomposition into local complements and operator kernels follows
[Harder and Valentin (2016)](https://doi.org/10.1007/978-3-319-41640-3_13).

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

For the unfitted analysis in [Chaumont-Frelet, Paredes and Valentin (2026)](https://doi.org/10.1016/j.camwa.2026.01.016), the macro mesh may cross material interfaces,
but every skeletal subface must belong to a single material region. Its extra
half-order is a face-refinement statement with suitable physical-region
regularity and sufficiently accurate local solves. It is not a blanket guarantee
for arbitrary unresolved cuts.

## Adaptivity and parallel execution

The multilevel estimators in [Araya, Rebolledo and Valentin (2021)](https://doi.org/10.1093/imanum/drz053) and [Araya et al. (2021)](https://doi.org/10.1007/s10444-020-09833-8) separate a skeletal contribution from local
residual contributions. A practical adaptive step marks subfaces, assesses
whether neighboring local errors dominate, updates affected local bases, and
reassembles their contributions. The macro topology can remain fixed. A generic
jump indicator is useful experimentally, but should not be called the published
reliable estimator without all required terms and scaling.

The implementation study of [Gomes et al. (2017, preprint v1)](https://arxiv.org/abs/1703.10435v1) separates local and
global work. Local factorizations, multiple trace/source right-hand sides, and
reconstruction are independent across macroelements. Assembly and the global solve remain
coupled. Effective implementations distinguish local assembly, factorization,
triangular solves, global assembly, global solution, and reconstruction in their
timings. Reuse is valid while geometry, coefficients, spaces, and stabilization
are unchanged; changing only the source can reuse trace bases.

CPU workers, MPI ranks, and GPU batches exploit different levels of this
independence. The mathematical decomposition permits these strategies but does
not guarantee speedup. Scheduling, thread oversubscription, data transfer,
factorization memory, and the global system determine observed performance.


## Recursive levels and reusable operators

The same equations can define a local solver at another scale. A local
problem in the outer hierarchy then contains an inner global trace system
and its own leaf local problems. Its returned response must represent the
same source, trial/test pairings, retained moments and physical gauge as
an ordinary local solve. Elimination of an inner level must preserve the
outer original equations; solving independent smaller systems without the
inner coupling would define a different method.

[Harder and Valentin (2016)](https://doi.org/10.1007/978-3-319-41640-3_13)
provide the abstract kernel/complement framework. The
[recursive MHM tutorial](../tutorials/methods/recursive-mhm.md) compares
hierarchical reconstruction against the complete leaf system. A recursive
level has no additional universal convergence power: its error includes
the approximation errors of all levels, each under its own trace/local
compatibility conditions.

Offline/online preparation can retain geometry, local factorizations and
harmonic lifts while changing source or boundary data. Reuse requires
identity of the discrete operator, spaces, coefficient fields and
stabilization. Sharing a compiled UFL kernel across structurally identical
cells does not justify sharing numerical matrices across different
materials. The [execution guide](../execution.md) distinguishes scheduling,
operator reuse and native backend choices.

Persisted coefficient vectors include their executed basis matrices and
identities. Replaying a field uses those matrices consistently in
orientation and evaluation maps; a source digest or a matching basis
size alone does not identify a numerical basis.

## References

- Christopher Harder and Frédéric Valentin (2016). *Foundations of the MHM Method*, in *Building Bridges: Connections and Challenges in Modern Approaches to Numerical Partial Differential Equations*, Lecture Notes in Computational Science and Engineering 114, Springer. [DOI: 10.1007/978-3-319-41640-3_13](https://doi.org/10.1007/978-3-319-41640-3_13).

- Théophile Chaumont-Frelet, Diego Paredes, and Frédéric Valentin (2026). *Flux approximation on unfitted meshes and application to multiscale hybrid-mixed methods*, Computers & Mathematics with Applications 209, 16–27. [DOI: 10.1016/j.camwa.2026.01.016](https://doi.org/10.1016/j.camwa.2026.01.016).

- Rodolfo Araya, Ramiro Rebolledo, and Frédéric Valentin (2021). *On a multiscale a posteriori error estimator for the Stokes and Brinkman equations*, IMA Journal of Numerical Analysis 41(1), 344–380. [DOI: 10.1093/imanum/drz053](https://doi.org/10.1093/imanum/drz053). An earlier version is [HAL: hal-01945934v1](https://hal.science/hal-01945934v1).

- Rodolfo Araya, Cristian Cárcamo, Abner H. Poza, and Frédéric Valentin (2021). *An adaptive multiscale hybrid-mixed method for the Oseen equations*, Advances in Computational Mathematics 47, article 15. [DOI: 10.1007/s10444-020-09833-8](https://doi.org/10.1007/s10444-020-09833-8).

- Antônio Tadeu A. Gomes, Weslley S. Pereira, Frédéric Valentin, and Diego Paredes (2017). *On the Implementation of a Scalable Simulator for Multiscale Hybrid-Mixed Methods*, arXiv preprint, version 1, 30 March 2017. [arXiv: 1703.10435v1](https://arxiv.org/abs/1703.10435v1).
