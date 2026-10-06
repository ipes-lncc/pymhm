# Literature and reproducibility

The MHM literature describes a family of discretizations, not one interchangeable
algorithm. The local finite element spaces, skeletal space, treatment of local
kernels, and reconstruction determine which stability and conservation statements
apply. This catalog distinguishes those mathematical results from capabilities
validated in this package. A reference in this page is not a claim that its numerical
tables have been reproduced.

Use the [case evidence guide](cases/index.md) to select a representative numerical
comparison and its public regeneration procedure.

The core collection contains 20 documents representing 19 distinct works: the 2022
preprint on unfitted meshes and its 2026 journal publication describe the same line
of research. Each citation identifies its publication or preprint version. Numerical
results from a preprint must be identified by version when compared with a later
journal article.

## Software provenance

Article data, executed reference programs, and independently written verification
forms are different sources of evidence. The comparisons identify their code by
name and revision rather than treating every external result as one reference
implementation.

| Code or source | Verified role in this repository |
|---|---|
| MSL: `msl_mhm` at `4cb8cf81518284313b680b13fd586ee619f08b99`, `msl_cg` at `afb76d14c1baf50f0b9e69f7bcac675749ef4458`, `msl_core` at `7f15f455717173d29080d411a7e732c72c1e87f8` | Executed primal MHM Darcy reference: MSL global coupling, continuous Galerkin local solves, and crisscross geometry. The [field comparison](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/reference-comparison.md) uses the same discrete spaces and weak Dirichlet moments as pyMHM. |
| MSL_MHM + MSL_CG (GaLS), at the same pinned MSL revisions above | [Independent displacement–pressure comparison](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/elasticity-reference.md) with a recorded mixed-field MHM adapter, five P1/P1 macro meshes and P2/P2/P3/P3 checks. Native GaLS element assembly is unchanged. |
| MSL_CG + MSL_Core, at the same pinned revisions above | [SPE10 Darcy flux comparison](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/spe10-flux.md) using native global conforming triangular P1 assembly and Eigen SparseLU on five pixel-aligned meshes. This classical reference is distinct from MSL's MHM coupling. |
| `msl_mfem` at `b9a67e7079c7e487e4ab1679c1bb3c880cc1909a`, with MFEM 4.9 | Executed as an auxiliary strong-Dirichlet reference. Its boundary enforcement differs from the five-mesh MSL comparison above; its fields are not substituted for that comparison. |
| `mhm-mfem` at `fb535acec1be87c19b3aa538ac73a265570e9fdc`, with MFEM 4.9 | A separate MFEM-based flow implementation. No validated Stokes field comparison is available for this revision. It is distinct from the historical 2017 equal-order implementation. |
| [DOLFINx/UFL](https://docs.fenicsproject.org/dolfinx/) | Independently written finite element assemblies for [Darcy](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/darcy-audit.md), [Stokes](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/flow-audit.md), [tetrahedral full-saddle Darcy](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/reconstruction3d.md), [3D flow](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/flow3d.md), [GaLS3D](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/gals3d.md) and the dedicated wave operators, plus local assembly through pyMHM's adapter. The recorded native comparisons identify DOLFINx 0.9.0 and their actual solvers. |
| [NeoPZ at `4c6b6d2`](https://github.com/labmec/neopz/tree/4c6b6d277ce097b97bfc8dea1b6725860f4fe05a) | Executed native RT0/P0 assembly, including restricted macro traces: [mixed comparison](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/neopz.md) and [classical SPE10 reference](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/spe10-flux.md). The [tetrahedral/prismatic study](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mixed-well-geometries.md) also executes native mixed spaces and `TPZMHMixedMeshControl`; its records distinguish NeoPZ assembly/restriction/field evaluation from the linear solver used in each comparison. |
| [Labmec/MHM at `f978f29`](https://github.com/labmec/MHM/tree/f978f29d657d28fe58bcea20fabee68953093482) | Source-level description of the mixed MHM controller and application settings. The executed NeoPZ RT0 driver is distinct from this positive-order application. |
| The Darcy 2013 and Stokes 2017 articles | Published curves digitized and compared with new pyMHM calculations in the [paper comparison](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/reproduction.md). The historical coefficient arrays and diagnostic programs for those particular figures have not been recovered. |

The DOLFINx assemblies are original verification code using a separate finite
element library. A monolithic solve that reuses pyMHM's local matrices checks
condensation, whereas an independently assembled UFL operator also checks
element assembly. The case pages state which of these checks was performed.
The analytical scalar, heat and displacement-elasticity galleries use pyMHM
and exact formulas; they are not executions of the reference packages above.
The MSL repositories are not public. Their names and revisions identify the
executed references; the documented results do not imply distribution of their
source code or comparison drivers with this package.

The [coarse cosine comparison](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/coarse-cosine.md) also executes the diagonal
mesh with constant traces. Its MSL adapter shares the assembly of prescribed
weak boundary moments with the sine comparison. The numerical libraries remain
unchanged; the reports identify the reference revisions and adapter digests.

## Darcy and elliptic diffusion

### Harder, Paredes and Valentin (2013): The original Darcy family {#harder-paredes-valentin-2013-darcy}

Christopher Harder, Diego Paredes, and Frédéric Valentin (2013). *A family of Multiscale Hybrid-Mixed finite element methods for the Darcy equation with rough coefficients*, Journal of Computational Physics 245, 107–130. [DOI: 10.1016/j.jcp.2013.03.019](https://doi.org/10.1016/j.jcp.2013.03.019).

Primal hybridization produces global normal-flux degrees of freedom and one
pressure constant per macroelement. Independent zero-mean Neumann problems
produce flux-driven basis functions and a source lifting. Higher polynomial degrees
and subdivisions of a macroface enlarge the skeletal space independently of the
macro partition. Local mixed solvers are already discussed as an alternative to
primal solvers.

Section 5 supplies a smooth cosine solution, constant and layered quarter five-spot
problems, oscillatory permeability, and a lognormal permeability field. The layered
example compares interfaces on coarse faces and inside macroelements. The
oscillatory example uses a fine finite element solution as a **reference**, rather
than an exact analytical solution. The random field requires the actual realization
for an exact reproduction; a new random seed defines a related benchmark.

### Araya et al. (2013): A priori and a posteriori analysis {#araya-et-al-2013-mhm}

Rodolfo Araya, Christopher Harder, Diego Paredes, and Frédéric Valentin (2013). *Multiscale Hybrid-Mixed Method*, SIAM Journal on Numerical Analysis 51(6), 3505–3531. [DOI: 10.1137/120888223](https://doi.org/10.1137/120888223).

This work establishes well-posedness, approximation estimates, and a face-residual
estimator for the elliptic formulation. The analysis distinguishes the ideal local
solution operator from its numerical realization. A compatible trace/local-space
pair is essential; choosing unrelated polynomial degrees does not produce a stable
method automatically.

Section 6 evaluates an analytical cosine solution, a low-permeability square
inclusion, and a quarter five-spot problem. The authors explicitly place the
Dirac-well example outside the regularity assumptions of the theory. Constants used
to scale the estimator in the numerical section must be retained when comparing
effectivity indices.

### Harder and Valentin (2016): Abstract foundations {#harder-valentin-2016-foundations}

Christopher Harder and Frédéric Valentin (2016). *Foundations of the MHM Method*, in *Building Bridges: Connections and Challenges in Modern Approaches to Numerical Partial Differential Equations*, Lecture Notes in Computational Science and Engineering 114, Springer. [DOI: 10.1007/978-3-319-41640-3_13](https://doi.org/10.1007/978-3-319-41640-3_13).

The abstract construction separates an operator kernel from a complementary local
space and uses generalized inverses to connect hybrid and global/local
formulations. This supports an implementation with arbitrary local nullspaces,
multiple source liftings, and independently defined coupling operators. General
nonsymmetric problems require attention to both the operator and its adjoint
kernel. The framework also explains relations to primal hybrid methods and special
Raviart–Thomas cases; these relations are conditional equivalences, not a statement
that every MHM discretization equals a standard mixed element.

`LocalProblem` provides independent trial/test couplings, left/right kernels,
retained bases and physical complement constraints. Independent full
Petrov–Galerkin systems verify its condensation and reconstruction. The
[recursive adapter](cases/nested.md) supplies an MHM discretization as a local
problem at another scale, with field equivalence checked on five refinements.
[Offline/online preparation](execution.md) reuses unchanged local/global factors
and harmonic lifts for changing source and boundary values. These algebraic
checks implement the construction; they do not establish the analytical
hypotheses for every user-defined operator. The chapter supplies no numerical
benchmark table requiring historical curve matching.

### Paredes, Valentin and Versieux (2017): Periodic-coefficient robustness {#paredes-valentin-versieux-2017-robustness}

Diego Paredes, Frédéric Valentin, and Henrique M. Versieux (2017). *On the robustness of multiscale hybrid-mixed methods*, Mathematics of Computation 86(304), 525–548. [DOI: 10.1090/mcom/3108](https://doi.org/10.1090/mcom/3108).

Homogenization estimates for periodic coefficients establish convergence in
specified relations between mesh size and physical wavelength without oversampling.
The hypotheses include regularity of the homogenized solution and periodic
correctors. They do not establish a parameter-independent error bound for every
heterogeneous tensor and every mesh.

Section 4 compares macro-mesh refinement with face-space enrichment for
`A = [1 + 100 cos²(πx/ε) sin²(πy/ε)] I`. It exposes an intermediate regime where
macro refinement can increase the error; face enrichment addresses that behavior
in the experiments. The fine reference uses 16,777,216 quadrilateral bilinear
elements, so matching only the coarse mesh does not reproduce the experiment.

### Durán et al. (2019): Mixed local Darcy solvers {#duran-et-al-2019-mixed-darcy}

Omar Durán, Philippe R. B. Devloo, Sônia M. Gomes, and Frédéric Valentin (2019). *A multiscale hybrid method for Darcy’s problems using mixed finite element local solvers*, Computer Methods in Applied Mechanics and Engineering 354, 213–244. [DOI: 10.1016/j.cma.2019.05.013](https://doi.org/10.1016/j.cma.2019.05.013).

MHM–H(div) constrains local normal traces to the skeletal space while retaining
finer interior flux and pressure spaces. The key compatibility is
`div V_h = Q_h`, with compatible normal-trace restrictions. Interior mesh
refinement, interior polynomial enrichment, and their combination are separate
options. The scheme can be implemented through multiscale basis construction or
two stages of static condensation. It preserves conservation on microelements
when the divergence equation is tested in the corresponding discontinuous pressure
space.

Sections 6–7 explicitly identify [NeoPZ](https://github.com/labmec/neopz) as
the implementation framework for both MHM-H1 and MHM-H(div). This attributes
the paper's numerical experiments to NeoPZ, but does not identify the exact
repository revision that generated each figure. The present
[NeoPZ RT0 comparison](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/neopz.md) states its own revision and spaces.
The native [triangular RT0/RT1/RT2](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/darcy-rt.md) and
[BDM2/P1](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/darcy-bdm.md) implementations preserve their discontinuous
pressure equilibrium moments and independent aligned skeletal restrictions.
Independent DOLFINx operators check the full-trace spaces. The
[rectangular RT family](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/tensor-rt.md) additionally separates normal degree
from zero-normal interior enrichment. Its Figure 3 comparison uses the
reference-source frequency and records all 50 distinguishable markers within
the stated three-pixel digitization uncertainty.
This is a published-curve comparison, not execution of the NeoPZ MHM controller.

Section 7 contains five problem families: smooth cosine pressure; an oscillatory
radial transition; random heterogeneous permeability; the three-dimensional
Dupuit–Thiem well problem; and radial flow with oscillatory anisotropic
permeability. The three-dimensional cases use hexahedral, tetrahedral, or prismatic
elements. Comparing the primal gradient with an H(div) flux requires separate
normal-continuity and divergence diagnostics, not just pressure errors.

The [affine tetrahedral and prismatic mixed implementation](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mixed-well-geometries.md)
supplies explicit normal-degree-one families: 18 tetrahedral flux modes with P1
pressure, 32 enriched modes with P2 pressure, and 27 prismatic modes with W11
pressure. Native NeoPZ basis evaluations and independent Basix tabulations
check these spaces; their physical Piola, complete divergence moments and full-trace
classical limit are tested separately. The well study partitions the same octagonal
domain as the hexahedral case, with exact pressure data on its polygonal walls.
The tetrahedral BDFM/interior-degree wording in §7.4 is stated explicitly rather
than identifying one implemented family with an undocumented historical configuration.
The prism's 27-dimensional tensor space excludes an additional solenoidal bubble
that the maximal W22 trace/divergence constraints alone would permit.
Independent NeoPZ classical mixed solves and its native `TPZMHMixedMeshControl`
add physical-field comparisons on matched cells and skeletal spaces. The
executed controller configuration and linear solver are recorded separately
from polynomial-space checks. [Durán et al. (2019)](https://doi.org/10.1016/j.cma.2019.05.013) Figure 15 normalizes by the classical
fine-mesh approximation error, whereas the gallery also reports errors relative
to the exact field norm. Historical mesh reproduction and same-space agreement
between codes are distinct comparisons.

### Chaumont-Frelet et al. (2022): The MHM–MsHHO connection {#chaumont-frelet-et-al-2022-mhm-mshho}

Théophile Chaumont-Frelet, Alexandre Ern, Simon Lemaire, and Frédéric Valentin (2022). *Bridging the Multiscale Hybrid-Mixed and Multiscale Hybrid High-Order Methods*, ESAIM: M2AN 56, 261–285. [DOI: 10.1051/m2an/2021082](https://doi.org/10.1051/m2an/2021082).

Theorem 5.1 proves equivalence on general polytopal meshes under exact local
solves. For the original semi-explicit MHM scheme, the source must belong to the
specified piecewise polynomial space. A fully explicit variant with a projected
source extends the equivalence to arbitrary square-integrable sources. The
`m = -1` variant is an explicit exception to equivalence. Section 7 develops
primal/dual bases and an offline/online organization for repeated source terms.
This is a theoretical and algorithmic comparison, not a numerical timing study.

The [MsHHO implementation](cases/mshho.md) constructs cell/face moment spaces,
constrained energy reconstruction and cell condensation for both source variants,
including polygonal cells and the face-only exception. Five-level analytical
campaigns complement conditional MHM field equivalence. The comparison is
measured through anisotropy contrast 1e6 with explicit extended local accumulation
and unchanged residual criteria. The maximum relative nodal pressure discrepancy
is 4.357956e-13. The separate native assembly and cross-operator original-equation
qualification are stated on the case page. Separate
[operator-reuse measurements](execution.md) are original engineering evidence,
not a timing reproduction of this article.

### Paredes, Valentin and Versieux (2024): Face-based robustness {#paredes-valentin-versieux-2024-face-robustness}

Diego Paredes, Frédéric Valentin, and Henrique M. Versieux (2024). *Revisiting the robustness of the multiscale hybrid-mixed method: The face-based strategy*, Journal of Computational and Applied Mathematics 436, 115415. [DOI: 10.1016/j.cam.2023.115415](https://doi.org/10.1016/j.cam.2023.115415).

Continuous piecewise polynomial spaces on independently refined faces complement
the original discontinuous multiplier construction. Convergence with a fixed macro
partition is analyzed under local regularity assumptions. Skeletal refinement and
local discretization must be tracked separately; a local error floor can hide the
predicted improvement from face enrichment.

Section 5 revisits the periodic benchmark and compares macro refinement with face
refinement, then studies layer 36 of SPE10 Model 2 on a 66-square macro partition.
Its SPE10 reference has 9,738,625 degrees of freedom. Reduced global dimension
must not be reported as reduced total computational work without accounting for
local solves.

`FaceSpace(..., continuous=True)` implements continuous polynomial interpolation
within a subdivided macroface. Continuity is not imposed between distinct
macrofaces. Both continuous and discontinuous bases are available independently
of the local refinement. The [layer-36 campaign](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/spe10.md) records Q1/C0-P1
MHM fields and the [flux comparison](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/spe10-flux.md) adds independently
refined Q3, MSL conforming P1 and NeoPZ global RT0 references. The historical
local/material quadrature choices and original coefficient arrays are not
identified solely by their agreement with sampled published pressure curves.

### Chaumont-Frelet, Paredes and Valentin (2022, preprint v1): Unfitted flux approximation {#chaumont-frelet-paredes-valentin-2022-unfitted}

Théophile Chaumont-Frelet, Diego Paredes, and Frédéric Valentin (2022). *Flux approximation on unfitted meshes and application to multiscale hybrid-mixed methods*, preprint, HAL version 1, 31 October 2022. [HAL: hal-03834748v1](https://inria.hal.science/hal-03834748v1).

The preprint develops flux projection in negative trace norms using regularity on
physical material regions. It is an earlier version of [Chaumont-Frelet, Paredes and Valentin (2026)](https://doi.org/10.1016/j.camwa.2026.01.016), not an independent
method to count twice. The journal publication adds numerical examples and gives
the authoritative published statement for a new verification campaign.

### Barrenechea et al. (2026): H(div) flux reconstruction {#barrenechea-et-al-2026-flux-reconstruction}

Gabriel R. Barrenechea, Larissa Martins, Weslley Pereira, and Frédéric Valentin (2026). *An H(div; Ω)-Conforming Flux Reconstruction for the Multiscale Hybrid-Mixed Method*, Multiscale Modeling & Simulation 24(2), 399–428. [DOI: 10.1137/24M1673073](https://doi.org/10.1137/24M1673073).

The accepted manuscript circulated before journal publication. It addresses the
loss of H(div) conformity when local problems are approximated by primal finite
elements. A local Raviart–Thomas reconstruction matches the skeletal multiplier
on macro boundaries, averaged primal-flux moments on interior microfaces, and
interior moments. It also supports a computable a posteriori estimator.

The conservation identity in Proposition 4.3 of the manuscript is tested against
**continuous** piecewise polynomials on each macroelement. For order zero it gives
macroelement balance; it must not be silently strengthened to conservation against
every discontinuous microcell constant. An independently equilibrated
reconstruction imposing all microcell balances is a different algorithm requiring
its own validation.

`pymhm.recovery.moments` implements the boundary, averaged interior-face
and volume moments for nonnegative RT orders. It exposes raw divergence and its
continuous macro-local projection separately. Normal orientation, polynomial
moments, projected convergence and independent Basix/DOLFINx interpolation are
checked. The primal wrapper requires `m <= k`; the optimal-estimate hypothesis
is dimension-dependent: \(k\geq\ell+d\) and \(\ell\leq m\leq k\).
Theorems 3.1, 4.4, 4.7 and 5.2 retain this dimensional hypothesis.
In three dimensions the [tetrahedral reconstruction and estimator](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/reconstruction3d.md)
therefore require local \(P_3\) for a constant skeletal space when those
estimates are invoked. Merely constructing RT moments is a separate algebraic
operation and need not satisfy the stronger error-estimate hypothesis.
Material jumps inside a macrocell require explicit one-sided flux evaluation.
The [oblique three-dimensional transmission checks](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/planar3d.md) combine
material-fitted local tetrahedra with unequal-area triangular partitions of
the original macrofaces. Their nine polynomial patches include Dirichlet,
mixed and pure-Neumann data. Since the analytical pressure belongs to the
material-wise local space, these are exact-representation and interface checks,
not a measured convergence rate or a historical [Barrenechea et al. (2026)](https://doi.org/10.1137/24M1673073) experiment.
The separate [unit-diffusion estimator](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/estimator.md) adds Oswald potential
recovery and all four terms of equations (5.3)–(5.7). It requires globally
conforming fine meshes, convex macrotriangles and homogeneous Dirichlet data.
Five-level energy-error and effectivity measurements accompany the mathematical
identities. For heterogeneous materials, `estimate_darcy_indicator` explicitly
distinguishes `convention="published"`, the unweighted terms printed in equations
(5.3)–(5.7), from `convention="energy"`, the coefficient-weighted energy version.
These choices can produce different adaptive rankings; the energy variant is
not substituted for the printed indicator in a publication comparison.
The [material-weighted estimator](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/weighted-estimator.md) states
its coefficient and boundary assumptions separately. [Macro adaptation](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/adaptive-darcy.md)
supports Dörfler marking, conforming red/green and longest-edge refinement.
The [SPE10 adaptive campaign](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/spe10-adaptive.md) instead uses
`PublishedDarcyIndicator` and the documented isotropic residual metric with
native FreeFEM/BAMG 4.13, coefficient one, P2/P0 spaces and four local triangles.
Seven solved states reach 4,783 macrotriangles / 12,001 global unknowns, compared
with the article's 3,786 / 9,559. A separately refined classical RT2/P2 hierarchy
has six levels and a terminal flux variation of 3.87% in L2.

The final pressure difference is 1.897%, whereas raw and reconstructed flux
L2 differences are 72.95% and 65.91%. The pressure profile is compared directly
with the digitized published initial and adapted curves. Nine additional
controls hold those 4,783 macrotriangles fixed while varying local resolution,
skeletal segmentation and material alignment. They separate local-only and
trace-only changes through six Galerkin energy identities. A final joint
control uses 869,028 material-fitted fine triangles and 62,527 global unknowns;
its pressure, raw-flux and reconstructed-flux L2 differences are 0.11371%,
11.70349% and 13.29238%, respectively. The corresponding inverse-material
flux differences are 9.06770% and 14.69486%. All percentages use the final
classical RT2 field as denominator; its own terminal flux increments remain
3.87% in L2 and 3.02% in the weighted norm.

The joint control changes both local and skeletal spaces, so neither its
improvement nor its remaining error is attributed to one change alone. It is
not the paper's four-triangle, one-P0-per-face discretization. Material fitting
also creates thin local triangles and uses explicit extended-precision residual
refinement; these solves do not establish uniform shape regularity. The
article and public implementation note leave the global permeability
normalization and modified remeshing wrapper unspecified. The present
calculation retains raw mD/ft conventions explicitly; pressure agreement alone
cannot identify a global material scale, which also affects the printed
indicator's relative terms. These records do not establish a historical flux
reproduction, optimal adaptive complexity or arbitrary-contrast effectivity.

### Chaumont-Frelet, Paredes and Valentin (2026): Unfitted flux approximation {#chaumont-frelet-paredes-valentin-2026-unfitted}

Théophile Chaumont-Frelet, Diego Paredes, and Frédéric Valentin (2026). *Flux approximation on unfitted meshes and application to multiscale hybrid-mixed methods*, Computers & Mathematics with Applications 209, 16–27. [DOI: 10.1016/j.camwa.2026.01.016](https://doi.org/10.1016/j.camwa.2026.01.016).

The macro mesh need not fit material interfaces, but the **skeletal subface
partition does fit the physical partition**: each subface lies within one material
region. The estimates involve material-region regularity, geometric constants,
coefficient bounds, and quasi-uniformity of the macro mesh. With fixed macro mesh
and adequately solved local problems, an extra half-order in skeletal refinement
appears. Arbitrary interface cuts through an unresolved subface are not covered.

Section 6 first uses `u = sin(2πx) sin(2πy)` on 16 coarse triangles, then compares
three configurations for a horizontal coefficient jump. In the unfitted
configuration with unresolved skeletal cuts, errors and oscillations increase;
placing skeletal subdivision points at the material intersections recovers the
favorable behavior. Observed exponential degree convergence for an analytic
solution is explicitly distinguished from the theorem’s super-algebraic result.

The [implemented interface study](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/unfitted.md) separates exact material
integration, fitted local triangular approximation and fitted skeletal
subfaces. Cartesian clipping and explicit one-sided evaluations preserve the
physical material regions. Recorded perturbation and local-refinement studies
include independent Fourier and refined conforming references; their stated
connectivity and local refinement define the comparison. The historical
connectivity is not completely specified, so identical historical curves are
not asserted.

## Reactive, advective, and diffusive transport

### Harder, Paredes and Valentin (2015): Advective/reactive domination {#harder-paredes-valentin-2015-rad}

Christopher Harder, Diego Paredes, and Frédéric Valentin (2015). *On a Multiscale Hybrid-Mixed Method for Advective-Reactive Dominated Problems with Heterogeneous Coefficients*, Multiscale Modeling & Simulation 13(2), 491–518. [DOI: 10.1137/130938499](https://doi.org/10.1137/130938499).

The conservative equation is `div(-K grad u + αu) + σu = f`. Its local
skew-symmetric weak form leads to a Robin multiplier
`(-K grad u + αu/2)·n`, rather than the total physical flux. A local reaction or
advection term can remove the constant kernel. The paper develops a face-adaptive
algorithm and permits stabilized local approximation.

Section 5 studies an analytical case, a source problem, skew advection, and a
heterogeneous problem. Positivity preservation and a discrete maximum principle
are not automatic consequences of these experiments. Boundary-layer comparisons
must record both trace resolution and local resolution.

The native Pk implementation includes variable SPD diffusion, velocity and
reaction, with Galerkin or consistent SUPG local forms. The conservative strong
residual includes both `div(alpha) * u` and the spatial divergence of the
diffusion tensor; coefficient derivatives are explicit inputs. Stabilization
changes the source as well as the operator. The [face estimator and adaptive
loop](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/adaptive-transport.md) implement equations (4.3)–(4.4), with declared
material bounds, strong Dirichlet conditions on essential faces and prescribed
Robin or physical diffusive-flux data on natural faces. Natural exterior
indicators are zero; local-discretization error and approximation of boundary
data are not measured by this jump indicator. The Figure-12 physical data are
used in an eight-state P1/Galerkin/P0 mixed-wall campaign, with independent
macro and local refinements. Its nominal \(\epsilon=0.1\) errors disagree with
the Figure 12 mesh curve. In a separate \(\epsilon=1\) regime, explicitly
published in Figure 7, all ten error values from five macro refinements lie
within the independent Figure 12 graphical intervals. Complete DOLFINx/UFL
saddle assemblies verify both discrete regimes. This comparison does not
identify the historical inputs or the cause of the incompatible caption.
Separate uniform and maximum-marked trace studies quantify the finite P1
local error, including its DG0 gradient lower bound. At local subdivision
256, seven of ten uniform-space error values lie inside the graphical
intervals; the final gradient error remains 59.28% above its marker.
The full-Dirichlet P3
study remains a separate control; historical local connectivity is not inferred.
[Transient transport](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/transient-transport.md) uses backward Euler,
operator reuse, RT0 Darcy velocity and the hydrodynamic dispersion law. Its
heterogeneous exact test is an original realization, not the article's
unavailable random coefficient field. None of these methods implies a discrete
maximum principle or monotone adaptive error reduction.

### Araya et al. (2024): Generalized RAD on polytopes {#araya-et-al-2024-generalized-rad}

Rodolfo Araya, Fabrice Jaillet, Diego Paredes, and Frédéric Valentin (2024). *Generalizing the multiscale hybrid-mixed method for reactive-advective-diffusive equations*, Computer Methods in Applied Mechanics and Engineering 428, 117089. [DOI: 10.1016/j.cma.2024.117089](https://doi.org/10.1016/j.cma.2024.117089).

This unifies diffusion-dominated and reaction/advection cases by identifying the
local constant kernel from the operator. Different macroelements can contribute
different numbers of coarse constants. The global system can therefore contain
both invertible local contributions and kernel-constrained contributions. The
analysis covers stability, two-level errors, and local/global condition numbers.

Section 5 uses oscillatory manufactured solutions on triangles, squares, rhombi,
hexagons, L-shaped cells, and tetrahedra; an analytical boundary layer; and five
coefficient configurations for condition-number studies. The reported
condition-number estimates are bounds, sometimes pessimistic relative to the
experiments. A scalar reaction threshold alone does not detect the local kernel:
divergence-free advection tangent to a cell boundary is a relevant counterexample.

The implementation retains local constants through exact complementary
elimination and checks physical global gauges. A zero-reaction, divergence-free
velocity tangent to the exterior boundary admits a constant global mode only
if the discrete half-advection Robin trace also represents that mode. The
verified polynomial test uses local P4 and a cubic skeletal trace; a coarse
constant trace cannot be substituted in that test without changing the discrete
kernel. [Simple polygonal macrocells](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/polygons.md), including nonconvex
cells, support the common local operators through conforming local
triangulation. Five polygon families have five-level oscillatory studies.
The explicit `coarse_space="kernel"` option selects primal or mixed local
problems cell by cell; reconstructed fields agree with complementary elimination.
The [tetrahedral RAD path](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/rad3d.md) uses general positive-degree local
Pk fields and independent Bernstein face polynomials. The
[general-degree verification](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/tetra-pk.md) includes native P5/P6 basis,
derivative and operator comparisons, and an admissible P5/P2 estimator study.
Its five-level P4/P1 RAD study uses the published
three-dimensional sine solution and a separately assembled classical P2
comparison. Both studies identify their original meshes instead of claiming
the article's historical connectivity. The [conditioning study](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/rad-conditioning.md)
uses all five published coefficient cases, separate parameter/local/face sweeps
and directly extracted figure markers. Global ordinates agree closely at unit
parameters; local ordinates and floating-point limits are reported separately.
The [hexagonal boundary layer](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/rad-layer.md) uses the published P3/P1
spaces, an exact profile, classical P2 and independent macro/skeleton/local
refinement histories in the correctly normalized $V$ norm.
Independent DOLFINx/UFL complete saddle solves also cover three boundary-layer
configurations and all five conditioning coefficient fields at two diffusion
values. They agree with the corresponding PyMHM physical fields to within
$2.96\times10^{-13}$ relatively. This verifies the specified discrete problems;
it does not resolve the unavailable historical mesh or the local spectral
ordinates reported separately in the conditioning study.
The [polyhedral scalar path](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/polyhedral-rad.md) retains P0/P1 traces on
original polygonal faces and Pk local tetrahedral fields. Its convex studies
cover cubes, triangular prisms and Voronoi prisms. The
[nonconvex extension](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/star-polyhedra.md) certifies an interior kernel ball
for each cell, as required by §4.1(iii), and preserves reentrant faces and
notches through validated tetrahedral cones. It rejects empty kernels,
disconnected cavity shells and intersecting cones. Curved faces and arbitrary
non-star-shaped cells are outside this geometry contract; the scalar
construction does not provide polyhedral flow or elasticity operators.

## Stokes, Brinkman, and Oseen

### Araya et al. (2017): Stokes and Brinkman construction {#araya-et-al-2017-stokes-brinkman}

Rodolfo Araya, Christopher Harder, Abner H. Poza, and Frédéric Valentin (2017). *Multiscale hybrid-mixed method for the Stokes and Brinkman equations—The method*, Computer Methods in Applied Mechanics and Engineering 324, 29–53. [DOI: 10.1016/j.cma.2017.05.027](https://doi.org/10.1016/j.cma.2017.05.027).

The model uses the vector Laplacian, `-ν Δu + Θu + grad p = f`, and
`div u = 0`. Its multiplier is the pseudotraction
`(-ν grad u + pI)n`. Local Stokes velocity kernels are translations; positive
definite Brinkman drag removes them. The pressure normalization is global, with
one additional scalar in the construction. The two-level realization uses the
unusual stabilized finite element method (USFEM), including consistent residual
terms on both sides of the local equations.

Section 3 contains a polynomial manufactured Stokes solution, a Brinkman boundary
layer, Stokes and high-drag lid-driven cavities, a heterogeneous channel, and an
SPE10 flow. Tables 1–2 measure **macroelement** mass balance; they do not show
pointwise incompressibility of every fine element. The exact local stress is
H(div), while raw derivatives of an approximate primal local solution need not be.

The [published-curve comparison](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/reproduction.md#stress-norms-and-published-values)
also checks the full Frobenius pseudostress against its pressure/gradient
identity and the USFEM pressure equation. For the declared right-isosceles
P2/P2 geometry, no admissible stabilization constant reconciles all four
digitized Figure 3 norms, even with independent 1% ordinate intervals.
That conclusion is conditional on the stated stress and broken-norm
conventions; it does not identify the historical diagnostic or connectivity.

The [SPE10 case](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/spe10.md) identifies layer 1 by the reported extrema,
retains the 264-macrotriangle P3/P3 configuration and separates the 2017
stabilization convention from [Araya et al. (2025)](https://doi.org/10.1137/24M1649368). The slip-wall component conditions and
the material-pixel intersections are explicit in the implementation.
The independently refined classical Taylor–Hood comparison uses the same PDE
and componentwise boundary conditions, but a different global finite element
space. Its own refinement differences quantify the numerical reference limit.

### Araya, Rebolledo and Valentin (2021): Multilevel Stokes/Brinkman estimator {#araya-rebolledo-valentin-2021-estimator}

Rodolfo Araya, Ramiro Rebolledo, and Frédéric Valentin (2021). *On a multiscale a posteriori error estimator for the Stokes and Brinkman equations*, IMA Journal of Numerical Analysis 41(1), 344–380. [DOI: 10.1093/imanum/drz053](https://doi.org/10.1093/imanum/drz053). An earlier version is [HAL: hal-01945934v1](https://hal.science/hal-01945934v1).

The estimator combines coarse-skeleton residuals and fine local residuals.
Efficiency and reliability apply to that complete quantity. The face-adaptive
algorithm keeps the macro topology fixed but can also refine neighboring local
meshes when their errors dominate. It is therefore more than marking faces from
the magnitude of their flux coefficients.

The numerical study compares an analytical Stokes case, lid-driven cavities, and
an SPE10 heterogeneous case. The analytical velocity printed in Section 5.3 of the
2018 preprint has a coordinate placement that does not satisfy `div u = 0` when
read literally. Reproduction must resolve that discrepancy, check the published
version, and differentiate the adopted analytical fields before generating data.

`estimate_flow_error(..., variant="stokes-brinkman-2021")` implements the
coarse jump and fine momentum, divergence and pseudotraction terms of
equations (4.1)–(4.7). It requires full Dirichlet data, constant positive
viscosity, uniform skeletal degree and material jumps aligned with fine cells.
Its source residual is integrated explicitly; there is no separate certified
oscillation bound or constant-one error guarantee. `adapt_flow_macros` implements
the macrocell marking of Algorithm 1, and `adapt_flow` implements the face and
local marking of Algorithm 2. The [Stokes–Brinkman campaigns](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/stokes-adaptive.md)
state the mesh-closure choices, estimator contributions and stopping criteria;
the historical mesh connectivity and numerical reliability constants are not
identified by these comparisons. The [Oseen campaigns](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/oseen.md) use [Araya et al. (2021)](https://doi.org/10.1007/s10444-020-09833-8)'s
estimator and identify that variant separately.

### Araya et al. (2021): Adaptive Oseen {#araya-et-al-2021-oseen}

Rodolfo Araya, Cristian Cárcamo, Abner H. Poza, and Frédéric Valentin (2021). *An adaptive multiscale hybrid-mixed method for the Oseen equations*, Advances in Computational Mathematics 47, article 15. [DOI: 10.1007/s10444-020-09833-8](https://doi.org/10.1007/s10444-020-09833-8).

The Oseen operator extends the flow model with a prescribed convection field.
The coercivity assumption in the analysis is
`γ - div α/2 ≥ γ_min > 0`. The local multiplier includes the corresponding
half-advection correction. A multilevel residual estimator drives skeletal and
local refinement without changing the macro partition.

Section 5 supplies smooth, boundary-layer, and internal-layer solutions. The
internal-layer velocity is built from a streamfunction with a hyperbolic tangent
transition. Its `γ = 0` setting lies beyond the strictly positive reaction
assumption stated for the analysis. The numerical study also shows estimator
effectivity changing with viscosity; parameter-robust constants must not be
claimed. Solving a prescribed-convection Oseen problem is not a Navier–Stokes
nonlinear or transient solver.

The [Oseen implementation](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/oseen.md) includes variable convection,
consistent equal-order stabilization, the two-level estimator and a closed
face/local refinement loop. Eight recorded analytical/adaptive studies contain
five states each, with independent velocity/pressure errors and effectivity.
The declared meshes and marking parameters are original; historical adaptive
connectivity and all numerical tables are not claimed to be reproduced.

### Araya et al. (2025): Stokes/Brinkman a priori analysis {#araya-et-al-2025-stokes-brinkman-analysis}

Rodolfo Araya, Christopher Harder, Abner H. Poza, and Frédéric Valentin (2025). *Multiscale Hybrid-Mixed Methods for the Stokes and Brinkman Equations—A Priori Analysis*, SIAM Journal on Numerical Analysis 63(2), 588–618. [DOI: 10.1137/24M1649368](https://doi.org/10.1137/24M1649368).

This analyzes both one- and two-level formulations using an abstract hybrid
framework. Stable Taylor–Hood local spaces and stabilized equal-order spaces are
treated, along with continuous and discontinuous skeletal interpolation.
Conditions on local inf-sup stability, trace compatibility, and kernel coupling
are separate obligations. A local solver with a familiar finite element name is
not sufficient by itself to establish global stability.

The resistance bound used in stabilization differs between [Araya et al. (2017)](https://doi.org/10.1016/j.cma.2017.05.027) and [Araya et al. (2025)](https://doi.org/10.1137/24M1649368):
[Araya et al. (2017)](https://doi.org/10.1016/j.cma.2017.05.027), equations (41)–(42), names the minimum eigenvalue, whereas [Araya et al. (2025)](https://doi.org/10.1137/24M1649368), section 4.1,
uses the maximum eigenvalue over each fine cell. For anisotropic resistance the
latter controls the negative squared-residual term. The inverse constants are
also written with different conventions: [Araya et al. (2017)](https://doi.org/10.1016/j.cma.2017.05.027) bounds squared norms, while [Araya et al. (2025)](https://doi.org/10.1137/24M1649368)
writes an unsquared inequality. A computed constant and its convention must be
recorded; copying the same symbol does not identify the numerical parameter.

Section 5 evaluates USFEM local solves, compares continuous and discontinuous
skeletal spaces, and studies a heterogeneous Brinkman case. The stable local
family is analyzed separately in Section 4.2. A new comparison between local
formulations should hold the skeletal space fixed and reduce the local
discretization error independently before attributing differences to a formulation.

## Elasticity

### Harder, Madureira and Valentin (2016): Primal hybrid elasticity {#harder-madureira-valentin-2016-elasticity}

Christopher Harder, Alexandre L. Madureira, and Frédéric Valentin (2016). *A hybrid-mixed method for elasticity*, ESAIM: M2AN 50, 311–336. [DOI: 10.1051/m2an/2015046](https://doi.org/10.1051/m2an/2015046).

The coarse unknowns include all rigid-body motions: three per two-dimensional
cell and six per three-dimensional cell. Face multipliers represent signed
tractions. Local Neumann solves take place in the orthogonal complement of rigid
motions, while coarse equations impose force and moment equilibrium. The work
analyzes one- and two-level methods, high-order estimates, and an a posteriori
estimator. Symmetric equilibrium stress from an exact local solve must be
distinguished from a raw stress computed with finite-dimensional primal bases.

For a single triangular local element, Lemma 6.6 gives sufficient polynomial
compatibility: local degree at least `ell+1` for even skeletal degree `ell`, and
at least `ell+2` for odd `ell`. Bubble enrichment provides additional options.
These are sufficient conditions for the specified spaces, not a universal rule
for every refined local mesh. The essential condition is injectivity of the
skeletal-to-local lifting in equation (6.6).

This is primarily an analysis paper. Its constants depend on the elasticity
tensor; the conclusion leaves the incompressible limit for further work. Passing
an affine displacement patch test does not establish uniform accuracy as the
Poisson ratio approaches one half. In particular, the P1 displacement method
and the mixed locking-free methods below are distinct discretizations.

The [two-dimensional primal implementation](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/primal-elasticity.md) includes
general Kelvin material tensors, P1–P4 and enriched local spaces, an explicit
lifting-injectivity check and the one-level face indicator. The
[three-dimensional implementation](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/elasticity3d.md) retains all six rigid
motions and accepts general Kelvin 6-by-6 tensors. Independent UFL operators,
patches and five-level manufactured campaigns verify these paths. They do not
turn raw primal stress into an H(div) field or establish uniform accuracy at
infinite Lamé modulus.

### Devloo et al. (2021): Mixed elasticity with weak stress symmetry {#devloo-et-al-2021-mixed-elasticity}

Philippe R. B. Devloo, Agnaldo M. Farias, Sônia M. Gomes, Weslley Pereira, Antonio J. B. dos Santos, and Frédéric Valentin (2021). *New H(div)-conforming multiscale hybrid-mixed methods for the elasticity problem on polygonal meshes*, ESAIM: M2AN 55, 1005–1037. [DOI: 10.1051/m2an/2021013](https://doi.org/10.1051/m2an/2021013).

The two-dimensional local mixed problem approximates stress, displacement, and
rotation. Normal stress traces are restricted to the skeletal space; the rotation
multiplier enforces stress symmetry weakly. The global stress is H(div), and its
divergence matches the discrete equilibrium space. Suitable Poisson and Stokes
finite element pairs underpin the stable mixed elasticity construction.

The paper's table and numerical experiments are two-dimensional. A separate
[three-dimensional mixed implementation](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mixed-elasticity3d.md) uses
the classical AFW family $[\mathrm{BDM}_k]^3/[P_{k-1}]^3/[P_{k-1}]^3$
on tetrahedra, with $k\ge2$ to contain all six rigid displacements exactly.
Its justification refers to [Arnold, Falk and Winther (2007)](https://arxiv.org/abs/math/0701506v1), Eq. (7.1) and
Theorems 7.1–7.2, rather than transferring the paper's two-dimensional
enrichment theorem. Native UFL checks verify local operators and complete
classical systems; the original MHM convergence and bulk-modulus studies
report displacement, full stress and axial rotation separately.

Table 1 includes row-wise BDM stress spaces with displacement and rotation of
one lower degree, together with enriched and quadrilateral alternatives. The
MHM decomposition also requires the displacement space to contain every rigid
motion: BDM2/P1/P1 in two dimensions satisfies that requirement; the lowest-order
BDM1/P0/P0 pair does not contain the rotational displacement. With
`asym(tau)=tau12-tau21`, the exact rotation is
`q=(du1/dy-du2/dx)/2`. A counterclockwise rigid displacement therefore has
rotation `q=-1`. Both fields belong to the same local null mode; fixing the
rotation independently would change the problem.

The compliance tensor can be evaluated stably in plane strain as
`C tau = dev(tau)/(2 mu) + tr(tau) I/[4(mu+lambda)]`. Section 5, remark (iv),
states independence of the error constants from Poisson ratio under the stated
space and material hypotheses. This does not establish uniform robustness for
arbitrary heterogeneous material contrast. Finite values tending to infinity
and an exactly infinite first Lamé modulus also require distinct gauge handling.

Section 6 considers an oscillatory Young modulus and a layered geomechanical
cross-section loaded by gravity, including nearly incompressible clay layers.
Comparisons target stress profiles as well as displacement. Weak symmetry must be
reported as a moment condition; it is not pointwise symmetric stress.
The [HPC4e case](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/hpc4e.md) uses the original 512 × 256 material arrays
published in `labmec/MHM`, with the 16 × 8 macro partition, RT1/Q1/P1 local
spaces and four P1 skeleton resolutions of section 6.2. The source file revision,
units, depth orientation and checksums are recorded with the results.
Independent DOLFINx/UFL classical RT1 and RT2 fields quantify reference
sensitivity. Refining RT2 from 512 × 256 to 1024 × 512 cells changes stress
L2 by 1.119% and displacement L2 by 0.01059%. The finest recorded MHM field
differs from the latter reference by 4.159% and 0.6252%, respectively.
These measured increments are not continuum error bounds. The
published-profile raster comparison is reported separately.
For section 6.1 the exact displacement is
`u1=x² y² cos(6 pi x) sin(7 pi y)/27`,
`u2=exp(y) sin(4 pi x)/5`, with
`E=100[1+0.3 sin(10 pi(x-0.5)) cos(10 pi y)]` and `nu=0.3`.
The printed factor in the first component is `(1/3)(x/3)²`, hence the denominator
is 27. These formulas have nonzero boundary values, which must be prescribed
from their actual trace. Table 3 fixes 32 macro triangles, uses skeletal sizes
`2^-j`, `j=2,...,5`, and local size half the skeletal size. Its exterior trace
resolution follows the fine space; changing it changes the comparison.

The numerical section explicitly states that these simulations were implemented
in [NeoPZ](https://github.com/labmec/neopz). This provenance belongs to the
paper's mixed stress/displacement/rotation formulation. The native pyMHM
displacement gallery is a different method and does not execute that NeoPZ
elasticity program.

The native [triangular mixed families](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mixed-families.md) implement BDMk
with zero, one or two additional interior degrees while retaining the prescribed
normal degree. The [rectangular RT families](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/elasticity-tensor-rt.md) use
displacement Qs and total-degree Ps rotation. Independent UFL compliance,
divergence and weak-symmetry operators, five-level field studies and finite/
infinite modulus sweeps accompany both paths. The oscillatory section-6.1 data
are evaluated with explicitly recorded trace and local spaces; incomplete
historical mesh and size conventions prevent an unqualified Table 3 claim.

### Gomes, Pereira and Valentin (2024, preprint v1): Low-order locking-free elasticity {#gomes-pereira-valentin-2024-locking-free}

Antônio Tadeu Azevedo Gomes, Weslley da Silva Pereira, and Frédéric Valentin (2024). *A low-order locking-free multiscale finite element method for isotropic elasticity*, arXiv preprint, version 1, 25 March 2024. [arXiv: 2403.16890v1](https://arxiv.org/abs/2403.16890v1).

Local displacement–Herrmann-pressure problems use consistent Galerkin least-squares
terms. The pressure is `p = -λ_L div u`; the stress is
`2G ε(u) - pI`. Bounds on stabilization and sufficient local refinement support
the well-posedness and locking-free claims. Kernel removal still concerns rigid
motions, rather than pressure or displacement means chosen arbitrarily.

For constant shear modulus, the local bilinear form is

$$
\begin{aligned}
B_K((u,p),(v,q))
&=(2G\varepsilon(u),\varepsilon(v))_K\\
&\quad -(p,\nabla\!\cdot v)_K-(q,\nabla\!\cdot u)_K
 -(\lambda_L^{-1}p,q)_K\\
&\quad -\alpha\sum_{\tau\subset K}h_\tau^2
 (\nabla\!\cdot\sigma(u,p),\nabla\!\cdot\sigma(v,q))_\tau.
\end{aligned}
$$

The matching load adds
`+alpha sum h_tau² (f, div sigma(v,q))`. The positive sign follows from
`-div sigma=f`. P1 with constant `G` has zero displacement Hessians, but P2/P3
requires the full displacement and pressure residual. Equation (4.9) bounds
`alpha` using the inverse constant in (4.4); no numerical value of `alpha` is
specified in section 6. A stable default must come from that inequality, not
from fitting an error curve. For skeletal degree one, Lemma 4.5 supplies the
local-refinement choices `(k,r)=(1,4),(2,2),(3,1)`.

In two dimensions, all three rigid displacement modes are removed locally. At the exact
incompressible limit with displacement prescribed on the full boundary, pressure
requires one global gauge and compatible boundary volume. For finite positive
`lambda_L`, its mean is constrained by the integrated constitutive equation.

Section 6 varies Poisson ratio from 0.2 to 0.49999. Tables 1–6 compare face
refinement for the stabilized and primal Galerkin variants. The low-order primal
variant loses accuracy near incompressibility; simply wrapping it in MHM does not
remove locking. Tables 1–3 use equal-order GaLS locals of degrees one, two and
three, respectively, and tables 4–6 use primal Galerkin locals. The fixed macro
mesh has 32 triangles and diameter `2^(-3/2)`; the six skeletal diameters divide
that value by `2^j`, `j=0,...,5`. The local diameter is `2^(k-3)` times the
skeletal diameter. These are face-refinement tables, not a macro-refinement
sequence.

The trigonometric displacement, pressure and force printed in section 6 require
a consistency check before numerical comparison. In the family
`u=u0+a(1-2nu)sin(pi x)sin(pi y)(1,1)`, the displacement printed there has
`a=1/2`, whereas the printed pressure has `a=1`. The package's analytical data
provide both displacement amplitudes and derive the corresponding pressure and
force consistently. The MSL manufactured input uses the consistent `a=1`
family. Matching that input validates a code comparison; it does not by itself
identify which data generated the published tables.

The formulation and analysis permit dimensions two and three; the paper's
numerical section uses two dimensions. The original [tetrahedral GaLS and
Taylor–Hood studies](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/gals3d.md) retain all six rigid modes and evaluate
the strain inverse bound on the physical tetrahedron. Independent UFL checks
cover the variable-shear residual, including its coefficient derivative,
and finite/infinite Lamé limits. These checks do not substitute triangular
inverse constants into a tetrahedral problem or identify historical 3D tables.

## Computational architecture

### Gomes et al. (2017, preprint v1): Scalable implementation {#gomes-et-al-2017-scalable-implementation}

Antônio Tadeu A. Gomes, Weslley S. Pereira, Frédéric Valentin, and Diego Paredes (2017). *On the Implementation of a Scalable Simulator for Multiscale Hybrid-Mixed Methods*, arXiv preprint, version 1, 30 March 2017. [arXiv: 1703.10435v1](https://arxiv.org/abs/1703.10435v1).

Independent local work, reduction/assembly, the global solve, and reconstruction
are separate computational stages. The study compares MPI and Erlang coordination
around numerical kernels and reports strong and weak scaling on a CPU cluster.
GPU acceleration is identified as future work, not demonstrated performance.

The described prototypes combine C++ numerical kernels with MPI or Erlang,
using libraries including Eigen, PaStiX and MKL/PARDISO. The paper does not
provide a pinned source revision establishing identity with the MSL revisions
executed in the present Darcy comparison. Its historical timings therefore
remain article data, not timings reproduced by that comparison.

Section 5 uses a three-dimensional sine solution, up to 74,293,248 local
tetrahedra, and 24–768 CPU cores. Table 2 shows that small tasks can scale poorly
while larger local problems amortize communication. Its best-of-three timings and
historical hardware configuration cannot predict Python thread, MPI, or GPU
speedup on another machine.

The [distributed API](execution.md) owns local problems by rank, assembles a
distributed PETSc matrix and solves it with MUMPS without gathering all local
responses or fields. Native tests include empty ranks and incompatible physical
data. The measured one-host strong-scaling studies have modest global matrices;
they do not reproduce the historical multi-node cluster campaign.
The [3D Darcy study](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/darcy3d.md) uses the same analytical sine PDE on
recorded tetrahedral meshes. Resident P1 GPU assembly and batched local LU are
separate original engineering capabilities, with transfers, synchronization and
resource limits stated explicitly. The GPU campaign does not claim acceleration
without an equivalent CPU-batch timing baseline.

## Published targets and numerical evidence

The [case evidence guide](cases/index.md) is the canonical map from this catalog
to representative calculations, public acquisition programs and comparison limits.
It distinguishes published comparisons, matched problem data, independent
references and analytical verification. The case pages retain the numerical tables,
reference refinement and approximation settings; the [scientific scope](https://github.com/ipes-lncc/pymhm/blob/main/ROADMAP.md#scientific-scope-and-acceptance-criteria)
identifies historical targets that remain open.

The collection is not fully reproduced. In particular, unresolved historical
inputs or norm conventions in the periodic, adaptive SPE10, transport, Stokes,
elasticity and wave experiments remain explicit in their cases. An analytical
verification or a separate reference calculation does not close those differences.

## Evidence needed for a reproduction claim

A reproducible comparison records the exact mathematical formulation, coefficient
data and units, boundary conditions, mesh geometry, macro/face/local resolutions,
polynomial degrees, quadrature, stabilization, gauges, solver tolerances, and error
norms. It reports numerical values and convergence rates with tolerances justified
by discretization and reference errors. Visual similarity is useful diagnostic
evidence, but does not establish a reproduced table or an asymptotic result.

Matching the published physical and discrete problem is the primary acceptance
criterion. When investigation cannot recover an input or reconcile a reported
value, the unresolved choice remains explicit and the same physical case must
be checked against an independently assembled implementation. Discrete agreement
requires matched spaces and norms. Accuracy without an exact solution additionally
requires a separately refined classical conforming reference with its own
refinement check. Small patches, another analytical problem and a solver residual
do not replace these comparisons. Published smooth-problem rates are not presumed
for singular forcing or discontinuous materials without the required hypotheses.

## Additional dedicated formulations

These methods use their own local/global equations. The current
[initial convergence catalogue](cases/minimal-convergence.md) states their
executed data, refinement directions and limitations. They are distinct from
the twenty-document catalog above.

| Reference | Implemented method and current numerical scope |
| --- | --- |
| [de Barros, Madureira and Valentin (2026, preprint v3)](https://arxiv.org/abs/2404.16978v3) | MH²M has independent pressure and conormal traces and the complete source lifting. The initial catalogue contains three polynomial-family series for the analytical quartic pressure on explicit triangular meshes. Oscillatory media, recovered historical meshes and independently refined heterogeneous references are separate comparisons. |
| [Barrenechea, Gomes and Paredes (2024)](https://doi.org/10.1137/22M1542556) | MH uses positive Robin local problems and distinguishes the Robin multiplier from physical Darcy flux. Two three-level analytical series use triangles and nonconvex polygons. Mixed and pure-Neumann inputs require physical compatibility and the declared pressure gauge. |
| [Fernando et al. (2023)](https://doi.org/10.1007/s40314-023-02304-y) | PGMHM uses residual local enrichment and its own global test equations. The current initial series measures enriched pressure and raw-gradient flux on three triangular smooth-problem meshes. Inclusion and SPE10 reproductions require their own material fitting, stabilization and resolved classical baselines. |
| [Santiago, Valentin and Martins (CILAMCE 2025)](https://doi.org/10.55592/cilamce2025.v5i.14270) | Scalar MHM-UNUSUAL uses the negative strong-residual reaction–diffusion form with its stated inverse constants and boundary convention. Three smooth epsilon=1 meshes provide the current initial convergence control; no singular-perturbation or heterogeneous rate follows from it. |
| [Chaumont-Frelet and Valentin (2020)](https://doi.org/10.1137/19M1255616) | The initial Helmholtz study uses the published analytical plane-wave data with declared Cartesian Q4/P2 spaces. Angular, resonance, local-resolution and PML studies are separate targets. The Marmousi pilot states its 160-by-80-metre crop and fixed fine spacing; it does not reproduce the full historical domain or establish a resolved reference. |
| [Lanteri et al. (2018)](https://doi.org/10.1137/16M110037X) | Maxwell has tangential coupling and central-DG local dynamics. The current nanoguide study refines an independently assembled DG Q2 discretization of the complete selected device at a shortened observation horizon. Its finest field is a numerical comparison level; MHM agreement and resolved-reference accuracy remain separate requirements. |
| [Gomes et al. (2017)](https://doi.org/10.20906/CPS/CILAMCE2017-0399) | Elastodynamics uses Newmark local responses and slabwise traction coupling. Equation (53) supplies the exact analytical data for three spatial levels at the explicitly shortened time 0.025s. The three-layer study varies the time step on one fixed conforming spatial mesh; it does not verify spatial resolution or a complete MHM/reference comparison. |

These entries establish specific implementations and evidence, not universal
coverage of every mesh, coefficient regime or experiment in the wider literature.

## References

- Christopher Harder, Diego Paredes, and Frédéric Valentin (2013). *A family of Multiscale Hybrid-Mixed finite element methods for the Darcy equation with rough coefficients*, Journal of Computational Physics 245, 107–130. [DOI: 10.1016/j.jcp.2013.03.019](https://doi.org/10.1016/j.jcp.2013.03.019).

- Rodolfo Araya, Christopher Harder, Diego Paredes, and Frédéric Valentin (2013). *Multiscale Hybrid-Mixed Method*, SIAM Journal on Numerical Analysis 51(6), 3505–3531. [DOI: 10.1137/120888223](https://doi.org/10.1137/120888223).

- Christopher Harder and Frédéric Valentin (2016). *Foundations of the MHM Method*, in *Building Bridges: Connections and Challenges in Modern Approaches to Numerical Partial Differential Equations*, Lecture Notes in Computational Science and Engineering 114, Springer. [DOI: 10.1007/978-3-319-41640-3_13](https://doi.org/10.1007/978-3-319-41640-3_13).

- Diego Paredes, Frédéric Valentin, and Henrique M. Versieux (2017). *On the robustness of multiscale hybrid-mixed methods*, Mathematics of Computation 86(304), 525–548. [DOI: 10.1090/mcom/3108](https://doi.org/10.1090/mcom/3108).

- Omar Durán, Philippe R. B. Devloo, Sônia M. Gomes, and Frédéric Valentin (2019). *A multiscale hybrid method for Darcy’s problems using mixed finite element local solvers*, Computer Methods in Applied Mechanics and Engineering 354, 213–244. [DOI: 10.1016/j.cma.2019.05.013](https://doi.org/10.1016/j.cma.2019.05.013).

- Théophile Chaumont-Frelet, Alexandre Ern, Simon Lemaire, and Frédéric Valentin (2022). *Bridging the Multiscale Hybrid-Mixed and Multiscale Hybrid High-Order Methods*, ESAIM: M2AN 56, 261–285. [DOI: 10.1051/m2an/2021082](https://doi.org/10.1051/m2an/2021082).

- Diego Paredes, Frédéric Valentin, and Henrique M. Versieux (2024). *Revisiting the robustness of the multiscale hybrid-mixed method: The face-based strategy*, Journal of Computational and Applied Mathematics 436, 115415. [DOI: 10.1016/j.cam.2023.115415](https://doi.org/10.1016/j.cam.2023.115415).

- Théophile Chaumont-Frelet, Diego Paredes, and Frédéric Valentin (2022). *Flux approximation on unfitted meshes and application to multiscale hybrid-mixed methods*, preprint, HAL version 1, 31 October 2022. [HAL: hal-03834748v1](https://inria.hal.science/hal-03834748v1).

- Gabriel R. Barrenechea, Larissa Martins, Weslley Pereira, and Frédéric Valentin (2026). *An H(div; Ω)-Conforming Flux Reconstruction for the Multiscale Hybrid-Mixed Method*, Multiscale Modeling & Simulation 24(2), 399–428. [DOI: 10.1137/24M1673073](https://doi.org/10.1137/24M1673073).

- Théophile Chaumont-Frelet, Diego Paredes, and Frédéric Valentin (2026). *Flux approximation on unfitted meshes and application to multiscale hybrid-mixed methods*, Computers & Mathematics with Applications 209, 16–27. [DOI: 10.1016/j.camwa.2026.01.016](https://doi.org/10.1016/j.camwa.2026.01.016).

- Christopher Harder, Diego Paredes, and Frédéric Valentin (2015). *On a Multiscale Hybrid-Mixed Method for Advective-Reactive Dominated Problems with Heterogeneous Coefficients*, Multiscale Modeling & Simulation 13(2), 491–518. [DOI: 10.1137/130938499](https://doi.org/10.1137/130938499).

- Rodolfo Araya, Fabrice Jaillet, Diego Paredes, and Frédéric Valentin (2024). *Generalizing the multiscale hybrid-mixed method for reactive-advective-diffusive equations*, Computer Methods in Applied Mechanics and Engineering 428, 117089. [DOI: 10.1016/j.cma.2024.117089](https://doi.org/10.1016/j.cma.2024.117089).

- Rodolfo Araya, Christopher Harder, Abner H. Poza, and Frédéric Valentin (2017). *Multiscale hybrid-mixed method for the Stokes and Brinkman equations—The method*, Computer Methods in Applied Mechanics and Engineering 324, 29–53. [DOI: 10.1016/j.cma.2017.05.027](https://doi.org/10.1016/j.cma.2017.05.027).

- Rodolfo Araya, Ramiro Rebolledo, and Frédéric Valentin (2021). *On a multiscale a posteriori error estimator for the Stokes and Brinkman equations*, IMA Journal of Numerical Analysis 41(1), 344–380. [DOI: 10.1093/imanum/drz053](https://doi.org/10.1093/imanum/drz053). An earlier version is [HAL: hal-01945934v1](https://hal.science/hal-01945934v1).

- Rodolfo Araya, Cristian Cárcamo, Abner H. Poza, and Frédéric Valentin (2021). *An adaptive multiscale hybrid-mixed method for the Oseen equations*, Advances in Computational Mathematics 47, article 15. [DOI: 10.1007/s10444-020-09833-8](https://doi.org/10.1007/s10444-020-09833-8).

- Rodolfo Araya, Christopher Harder, Abner H. Poza, and Frédéric Valentin (2025). *Multiscale Hybrid-Mixed Methods for the Stokes and Brinkman Equations—A Priori Analysis*, SIAM Journal on Numerical Analysis 63(2), 588–618. [DOI: 10.1137/24M1649368](https://doi.org/10.1137/24M1649368).

- Christopher Harder, Alexandre L. Madureira, and Frédéric Valentin (2016). *A hybrid-mixed method for elasticity*, ESAIM: M2AN 50, 311–336. [DOI: 10.1051/m2an/2015046](https://doi.org/10.1051/m2an/2015046).

- Philippe R. B. Devloo, Agnaldo M. Farias, Sônia M. Gomes, Weslley Pereira, Antonio J. B. dos Santos, and Frédéric Valentin (2021). *New H(div)-conforming multiscale hybrid-mixed methods for the elasticity problem on polygonal meshes*, ESAIM: M2AN 55, 1005–1037. [DOI: 10.1051/m2an/2021013](https://doi.org/10.1051/m2an/2021013).

- Antônio Tadeu Azevedo Gomes, Weslley da Silva Pereira, and Frédéric Valentin (2024). *A low-order locking-free multiscale finite element method for isotropic elasticity*, arXiv preprint, version 1, 25 March 2024. [arXiv: 2403.16890v1](https://arxiv.org/abs/2403.16890v1).

- Antônio Tadeu A. Gomes, Weslley S. Pereira, Frédéric Valentin, and Diego Paredes (2017). *On the Implementation of a Scalable Simulator for Multiscale Hybrid-Mixed Methods*, arXiv preprint, version 1, 30 March 2017. [arXiv: 1703.10435v1](https://arxiv.org/abs/1703.10435v1).

- Stéphane Lanteri, Diego Paredes, Claire Scheid, and Frédéric Valentin (2018). *The Multiscale Hybrid-Mixed method for the Maxwell Equations in Heterogeneous Media*. Multiscale Modeling & Simulation 16(4) 1648-1683. [DOI: 10.1137/16M110037X](https://doi.org/10.1137/16M110037X).

- Théophile Chaumont-Frelet, and Frédéric Valentin (2020). *A Multiscale Hybrid-Mixed Method for the Helmholtz Equation in Heterogeneous Domains*. SIAM Journal on Numerical Analysis 58(2) 1029-1067. [DOI: 10.1137/19M1255616](https://doi.org/10.1137/19M1255616).

- Antonio Tadeu Gomes, Diego Paredes, Weslley Pereira, Roberto Souto, and Frederic Valentin (2017). *A Multiscale Hybrid-Mixed Method for the Elastodynamic Model with Rough Coefficients*. Proceedings of the XXXVIII Iberian Latin American Congress on Computational Methods in Engineering. [DOI: 10.20906/CPS/CILAMCE2017-0399](https://doi.org/10.20906/CPS/CILAMCE2017-0399).

- Honório Fernando, Larissa Martins, Weslley Pereira, and Frédéric Valentin (2023). *A Petrov–Galerkin multiscale hybrid-mixed method for the Darcy equation on polytopes*. Computational and Applied Mathematics 42, article 173. [DOI: 10.1007/s40314-023-02304-y](https://doi.org/10.1007/s40314-023-02304-y).

- Gabriel R. Barrenechea, Antonio Tadeu A. Gomes, and Diego Paredes (2024). *A Multiscale Hybrid Method*. SIAM Journal on Scientific Computing 46(3), A1628–A1657. [DOI: 10.1137/22M1542556](https://doi.org/10.1137/22M1542556).

- Juan Felipe Pacazuca Santiago, Frédéric Valentin, and Larissa Martins (2025). *A Multiscale Hybrid-Mixed Method with Local Stabilization*. Proceedings of the Ibero-Latin American Congress on Computational Methods in Engineering, CILAMCE 2025, volume 5, article 14270; published online 18 March 2026. [DOI: 10.55592/cilamce2025.v5i.14270](https://doi.org/10.55592/cilamce2025.v5i.14270).

- Franklin de Barros, Alexandre L. Madureira, and Frédéric Valentin (2026). *A three-field Multiscale Method*. arXiv preprint, version 3, 5 August 2026; first submitted 25 April 2024. [arXiv: 2404.16978v3](https://arxiv.org/abs/2404.16978v3).

- Douglas N. Arnold, Richard S. Falk, and Ragnar Winther (2007). *Mixed finite element methods for linear elasticity with weakly imposed symmetry*. Mathematics of Computation 76, 1699–1723. [DOI: 10.1090/S0025-5718-07-01998-9](https://doi.org/10.1090/S0025-5718-07-01998-9). [Preprint: arXiv:math/0701506v1](https://arxiv.org/abs/math/0701506v1).
