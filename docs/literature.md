# Bibliography

Cite the original publications for the methods, reconstructions and error
estimators used in a calculation, together with the PyMHM version. The
[Available Methods](index.md#available-methods) table identifies these works
at a glance; [Theoretical Background](theory.md) explains their assumptions.
Each tutorial and case also gives its relevant references locally.

Journal articles and versioned preprints are identified separately. A later
publication of the same work is not an additional implemented method. A
citation describes an intellectual source; the [Gallery](gallery/index.md)
states which problems, spaces and results have actually been evaluated.

## Original MHM formulations and analyses

### Harder, Paredes and Valentin (2013): The original Darcy family {#harder-paredes-valentin-2013-darcy}

Christopher Harder, Diego Paredes, and Frédéric Valentin (2013). *A family of Multiscale Hybrid-Mixed finite element methods for the Darcy equation with rough coefficients*, Journal of Computational Physics 245, 107–130. [DOI: 10.1016/j.jcp.2013.03.019](https://doi.org/10.1016/j.jcp.2013.03.019).

Primal hybridization produces global normal-flux degrees of freedom and one
pressure constant per macroelement. Independent zero-mean Neumann problems
produce flux-driven basis functions and a source lifting. Higher polynomial degrees
and subdivisions of a macroface enlarge the skeletal space independently of the
macro partition. Local mixed solvers are already discussed as an alternative to
primal solvers.

### Araya et al. (2013): A priori and a posteriori analysis {#araya-et-al-2013-mhm}

Rodolfo Araya, Christopher Harder, Diego Paredes, and Frédéric Valentin (2013). *Multiscale Hybrid-Mixed Method*, SIAM Journal on Numerical Analysis 51(6), 3505–3531. [DOI: 10.1137/120888223](https://doi.org/10.1137/120888223).

This work establishes well-posedness, approximation estimates, and a face-residual
estimator for the elliptic formulation. The analysis distinguishes the ideal local
solution operator from its numerical realization. A compatible trace/local-space
pair is essential; choosing unrelated polynomial degrees does not produce a stable
method automatically.

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

### Paredes, Valentin and Versieux (2017): Periodic-coefficient robustness {#paredes-valentin-versieux-2017-robustness}

Diego Paredes, Frédéric Valentin, and Henrique M. Versieux (2017). *On the robustness of multiscale hybrid-mixed methods*, Mathematics of Computation 86(304), 525–548. [DOI: 10.1090/mcom/3108](https://doi.org/10.1090/mcom/3108).

Homogenization estimates for periodic coefficients establish convergence in
specified relations between mesh size and physical wavelength without oversampling.
The hypotheses include regularity of the homogenized solution and periodic
correctors. They do not establish a parameter-independent error bound for every
heterogeneous tensor and every mesh.

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

### Cicuttin, Ern and Lemaire (2019): Original MsHHO construction {#cicuttin-ern-lemaire-2019-mshho}

Matteo Cicuttin, Alexandre Ern and Simon Lemaire (2019). *A Hybrid High-Order Method for Highly Oscillatory Elliptic Problems*, Computational Methods in Applied Mathematics 19(4), 723–748. [DOI: 10.1515/cmam-2018-0013](https://doi.org/10.1515/cmam-2018-0013).

This work introduces multiscale HHO local reconstructions and analyzes highly oscillatory elliptic problems on general meshes. The following bridge paper specifies the cell/face moment variants and equivalence with MHM used in PyMHM.

### Chaumont-Frelet et al. (2022): The MHM–MsHHO connection {#chaumont-frelet-et-al-2022-mhm-mshho}

Théophile Chaumont-Frelet, Alexandre Ern, Simon Lemaire, and Frédéric Valentin (2022). *Bridging the Multiscale Hybrid-Mixed and Multiscale Hybrid High-Order Methods*, ESAIM: M2AN 56, 261–285. [DOI: 10.1051/m2an/2021082](https://doi.org/10.1051/m2an/2021082).

Theorem 5.1 proves equivalence on general polytopal meshes under exact local
solves. For the original semi-explicit MHM scheme, the source must belong to the
specified piecewise polynomial space. A fully explicit variant with a projected
source extends the equivalence to arbitrary square-integrable sources. The
`m = -1` variant is an explicit exception to equivalence. Section 7 develops
primal/dual bases and an offline/online organization for repeated source terms.
This is a theoretical and algorithmic comparison, not a numerical timing study.

### Paredes, Valentin and Versieux (2024): Face-based robustness {#paredes-valentin-versieux-2024-face-robustness}

Diego Paredes, Frédéric Valentin, and Henrique M. Versieux (2024). *Revisiting the robustness of the multiscale hybrid-mixed method: The face-based strategy*, Journal of Computational and Applied Mathematics 436, 115415. [DOI: 10.1016/j.cam.2023.115415](https://doi.org/10.1016/j.cam.2023.115415).

Continuous piecewise polynomial spaces on independently refined faces complement
the original discontinuous multiplier construction. Convergence with a fixed macro
partition is analyzed under local regularity assumptions. Skeletal refinement and
local discretization must be tracked separately; a local error floor can hide the
predicted improvement from face enrichment.

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

### Chaumont-Frelet, Paredes and Valentin (2026): Unfitted flux approximation {#chaumont-frelet-paredes-valentin-2026-unfitted}

Théophile Chaumont-Frelet, Diego Paredes, and Frédéric Valentin (2026). *Flux approximation on unfitted meshes and application to multiscale hybrid-mixed methods*, Computers & Mathematics with Applications 209, 16–27. [DOI: 10.1016/j.camwa.2026.01.016](https://doi.org/10.1016/j.camwa.2026.01.016).

The macro mesh need not fit material interfaces, but the **skeletal subface
partition does fit the physical partition**: each subface lies within one material
region. The estimates involve material-region regularity, geometric constants,
coefficient bounds, and quasi-uniformity of the macro mesh. With fixed macro mesh
and adequately solved local problems, an extra half-order in skeletal refinement
appears. Arbitrary interface cuts through an unresolved subface are not covered.

### Harder, Paredes and Valentin (2015): Advective/reactive domination {#harder-paredes-valentin-2015-rad}

Christopher Harder, Diego Paredes, and Frédéric Valentin (2015). *On a Multiscale Hybrid-Mixed Method for Advective-Reactive Dominated Problems with Heterogeneous Coefficients*, Multiscale Modeling & Simulation 13(2), 491–518. [DOI: 10.1137/130938499](https://doi.org/10.1137/130938499).

The conservative equation is `div(-K grad u + αu) + σu = f`. Its local
skew-symmetric weak form leads to a Robin multiplier
`(-K grad u + αu/2)·n`, rather than the total physical flux. A local reaction or
advection term can remove the constant kernel. The paper develops a face-adaptive
algorithm and permits stabilized local approximation.

### Araya et al. (2024): Generalized RAD on polytopes {#araya-et-al-2024-generalized-rad}

Rodolfo Araya, Fabrice Jaillet, Diego Paredes, and Frédéric Valentin (2024). *Generalizing the multiscale hybrid-mixed method for reactive-advective-diffusive equations*, Computer Methods in Applied Mechanics and Engineering 428, 117089. [DOI: 10.1016/j.cma.2024.117089](https://doi.org/10.1016/j.cma.2024.117089).

This unifies diffusion-dominated and reaction/advection cases by identifying the
local constant kernel from the operator. Different macroelements can contribute
different numbers of coarse constants. The global system can therefore contain
both invertible local contributions and kernel-constrained contributions. The
analysis covers stability, two-level errors, and local/global condition numbers.

### Araya et al. (2017): Stokes and Brinkman construction {#araya-et-al-2017-stokes-brinkman}

Rodolfo Araya, Christopher Harder, Abner H. Poza, and Frédéric Valentin (2017). *Multiscale hybrid-mixed method for the Stokes and Brinkman equations—The method*, Computer Methods in Applied Mechanics and Engineering 324, 29–53. [DOI: 10.1016/j.cma.2017.05.027](https://doi.org/10.1016/j.cma.2017.05.027).

The model uses the vector Laplacian, `-ν Δu + Θu + grad p = f`, and
`div u = 0`. Its multiplier is the pseudotraction
`(-ν grad u + pI)n`. Local Stokes velocity kernels are translations; positive
definite Brinkman drag removes them. The pressure normalization is global, with
one additional scalar in the construction. The two-level realization uses the
unusual stabilized finite element method (USFEM), including consistent residual
terms on both sides of the local equations.

### Araya, Rebolledo and Valentin (2021): Multilevel Stokes/Brinkman estimator {#araya-rebolledo-valentin-2021-estimator}

Rodolfo Araya, Ramiro Rebolledo, and Frédéric Valentin (2021). *On a multiscale a posteriori error estimator for the Stokes and Brinkman equations*, IMA Journal of Numerical Analysis 41(1), 344–380. [DOI: 10.1093/imanum/drz053](https://doi.org/10.1093/imanum/drz053). An earlier version is [HAL: hal-01945934v1](https://hal.science/hal-01945934v1).

The estimator combines coarse-skeleton residuals and fine local residuals.
Efficiency and reliability apply to that complete quantity. The face-adaptive
algorithm keeps the macro topology fixed but can also refine neighboring local
meshes when their errors dominate. It is therefore more than marking faces from
the magnitude of their flux coefficients.

### Araya et al. (2021): Adaptive Oseen {#araya-et-al-2021-oseen}

Rodolfo Araya, Cristian Cárcamo, Abner H. Poza, and Frédéric Valentin (2021). *An adaptive multiscale hybrid-mixed method for the Oseen equations*, Advances in Computational Mathematics 47, article 15. [DOI: 10.1007/s10444-020-09833-8](https://doi.org/10.1007/s10444-020-09833-8).

The Oseen operator extends the flow model with a prescribed convection field.
The coercivity assumption in the analysis is
`γ - div α/2 ≥ γ_min > 0`. The local multiplier includes the corresponding
half-advection correction. A multilevel residual estimator drives skeletal and
local refinement without changing the macro partition.

### Araya et al. (2025): Stokes/Brinkman a priori analysis {#araya-et-al-2025-stokes-brinkman-analysis}

Rodolfo Araya, Christopher Harder, Abner H. Poza, and Frédéric Valentin (2025). *Multiscale Hybrid-Mixed Methods for the Stokes and Brinkman Equations—A Priori Analysis*, SIAM Journal on Numerical Analysis 63(2), 588–618. [DOI: 10.1137/24M1649368](https://doi.org/10.1137/24M1649368).

This analyzes both one- and two-level formulations using an abstract hybrid
framework. Stable Taylor–Hood local spaces and stabilized equal-order spaces are
treated, along with continuous and discontinuous skeletal interpolation.
Conditions on local inf-sup stability, trace compatibility, and kernel coupling
are separate obligations. A local solver with a familiar finite element name is
not sufficient by itself to establish global stability.

### Harder, Madureira and Valentin (2016): Primal hybrid elasticity {#harder-madureira-valentin-2016-elasticity}

Christopher Harder, Alexandre L. Madureira, and Frédéric Valentin (2016). *A hybrid-mixed method for elasticity*, ESAIM: M2AN 50, 311–336. [DOI: 10.1051/m2an/2015046](https://doi.org/10.1051/m2an/2015046).

The coarse unknowns include all rigid-body motions: three per two-dimensional
cell and six per three-dimensional cell. Face multipliers represent signed
tractions. Local Neumann solves take place in the orthogonal complement of rigid
motions, while coarse equations impose force and moment equilibrium. The work
analyzes one- and two-level methods, high-order estimates, and an a posteriori
estimator. Symmetric equilibrium stress from an exact local solve must be
distinguished from a raw stress computed with finite-dimensional primal bases.

### Devloo et al. (2021): Mixed elasticity with weak stress symmetry {#devloo-et-al-2021-mixed-elasticity}

Philippe R. B. Devloo, Agnaldo M. Farias, Sônia M. Gomes, Weslley Pereira, Antonio J. B. dos Santos, and Frédéric Valentin (2021). *New H(div)-conforming multiscale hybrid-mixed methods for the elasticity problem on polygonal meshes*, ESAIM: M2AN 55, 1005–1037. [DOI: 10.1051/m2an/2021013](https://doi.org/10.1051/m2an/2021013).

The two-dimensional local mixed problem approximates stress, displacement, and
rotation. Normal stress traces are restricted to the skeletal space; the rotation
multiplier enforces stress symmetry weakly. The global stress is H(div), and its
divergence matches the discrete equilibrium space. Suitable Poisson and Stokes
finite element pairs underpin the stable mixed elasticity construction.

### Gomes, Pereira and Valentin (2024, preprint v1): Low-order locking-free elasticity {#gomes-pereira-valentin-2024-locking-free}

Antônio Tadeu Azevedo Gomes, Weslley da Silva Pereira, and Frédéric Valentin (2024). *A low-order locking-free multiscale finite element method for isotropic elasticity*, arXiv preprint, version 1, 25 March 2024. [arXiv: 2403.16890v1](https://arxiv.org/abs/2403.16890v1).

Local displacement–Herrmann-pressure problems use consistent Galerkin least-squares
terms. The pressure is `p = -λ_L div u`; the stress is
`2G ε(u) - pI`. Bounds on stabilization and sufficient local refinement support
the well-posedness and locking-free claims. Kernel removal still concerns rigid
motions, rather than pressure or displacement means chosen arbitrarily.

### Gomes et al. (2017, preprint v1): Scalable implementation {#gomes-et-al-2017-scalable-implementation}

Antônio Tadeu A. Gomes, Weslley S. Pereira, Frédéric Valentin, and Diego Paredes (2017). *On the Implementation of a Scalable Simulator for Multiscale Hybrid-Mixed Methods*, arXiv preprint, version 1, 30 March 2017. [arXiv: 1703.10435v1](https://arxiv.org/abs/1703.10435v1).

Independent local work, reduction/assembly, the global solve, and reconstruction
are separate computational stages. The study compares MPI and Erlang coordination
around numerical kernels and reports strong and weak scaling on a CPU cluster.
GPU acceleration is identified as future work, not demonstrated performance.

## Other implemented multiscale methods

- Stéphane Lanteri, Diego Paredes, Claire Scheid, and Frédéric Valentin (2018). *The Multiscale Hybrid-Mixed method for the Maxwell Equations in Heterogeneous Media*. Multiscale Modeling & Simulation 16(4) 1648-1683. [DOI: 10.1137/16M110037X](https://doi.org/10.1137/16M110037X).

- Théophile Chaumont-Frelet, and Frédéric Valentin (2020). *A Multiscale Hybrid-Mixed Method for the Helmholtz Equation in Heterogeneous Domains*. SIAM Journal on Numerical Analysis 58(2) 1029-1067. [DOI: 10.1137/19M1255616](https://doi.org/10.1137/19M1255616).

- Antonio Tadeu Gomes, Diego Paredes, Weslley Pereira, Roberto Souto, and Frederic Valentin (2017). *A Multiscale Hybrid-Mixed Method for the Elastodynamic Model with Rough Coefficients*. Proceedings of the XXXVIII Iberian Latin American Congress on Computational Methods in Engineering. [DOI: 10.20906/CPS/CILAMCE2017-0399](https://doi.org/10.20906/CPS/CILAMCE2017-0399).

- Honório Fernando, Larissa Martins, Weslley Pereira, and Frédéric Valentin (2023). *A Petrov–Galerkin multiscale hybrid-mixed method for the Darcy equation on polytopes*. Computational and Applied Mathematics 42, article 173. [DOI: 10.1007/s40314-023-02304-y](https://doi.org/10.1007/s40314-023-02304-y).

- Gabriel R. Barrenechea, Antonio Tadeu A. Gomes, and Diego Paredes (2024). *A Multiscale Hybrid Method*. SIAM Journal on Scientific Computing 46(3), A1628–A1657. [DOI: 10.1137/22M1542556](https://doi.org/10.1137/22M1542556).

- Juan Felipe Pacazuca Santiago, Frédéric Valentin, and Larissa Martins (2025). *A Multiscale Hybrid-Mixed Method with Local Stabilization*. Proceedings of the Ibero-Latin American Congress on Computational Methods in Engineering, CILAMCE 2025, volume 5, article 14270; published online 18 March 2026. [DOI: 10.55592/cilamce2025.v5i.14270](https://doi.org/10.55592/cilamce2025.v5i.14270).

- Franklin de Barros, Alexandre L. Madureira, and Frédéric Valentin (2026). *A three-field Multiscale Method*. arXiv preprint, version 3, 5 August 2026; first submitted 25 April 2024. [arXiv: 2404.16978v3](https://arxiv.org/abs/2404.16978v3).

- Douglas N. Arnold, Richard S. Falk, and Ragnar Winther (2007). *Mixed finite element methods for linear elasticity with weakly imposed symmetry*. Mathematics of Computation 76, 1699–1723. [DOI: 10.1090/S0025-5718-07-01998-9](https://doi.org/10.1090/S0025-5718-07-01998-9). [Preprint: arXiv:math/0701506v1](https://arxiv.org/abs/math/0701506v1).

## Finite elements, stabilization, meshing, software and data

- M. Cecilia Rivara (1984). *Algorithms for refining triangular grids suitable for adaptive and multigrid techniques*. International Journal for Numerical Methods in Engineering 20(4), 745–756. [DOI: 10.1002/nme.1620200412](https://doi.org/10.1002/nme.1620200412).

- Pedro Henrique Penna, Antônio Tadeu A. Gomes, Márcio Castro, Patricia D.M. Plentz, Henrique C. Freitas, François Broquedis, and Jean‐François Méhaut (2019). *A comprehensive performance evaluation of the BinLPT workload‐aware loop scheduler*. Concurrency and Computation: Practice and Experience 31(18) e5170. [DOI: 10.1002/cpe.5170](https://doi.org/10.1002/cpe.5170).

- Gary S. Martin, Robert Wiley, and Kurt J. Marfurt (2006). *Marmousi2: An elastic upgrade for Marmousi*. The Leading Edge 25(2) 156-166. [DOI: 10.1190/1.2172306](https://doi.org/10.1190/1.2172306).

- Douglas A. Castro, Philippe R.B. Devloo, Agnaldo M. Farias, Sônia M. Gomes, Denise de Siqueira, and Omar Durán (2016). *Three dimensional hierarchical mixed finite element approximations with enhanced primal variable accuracy*. Computer Methods in Applied Mechanics and Engineering 306 479-502. [DOI: 10.1016/j.cma.2016.03.050](https://doi.org/10.1016/j.cma.2016.03.050).

- Gabriel R. Barrenechea, Fabrice Jaillet, Diego Paredes, and Frédéric Valentin (2020). *The multiscale hybrid mixed method in general polygonal meshes*. Numerische Mathematik 145(1), 197–237. [DOI: 10.1007/s00211-020-01103-5](https://doi.org/10.1007/s00211-020-01103-5).

- L. P. Franca and Frédéric Valentin (2000). *On an improved unusual stabilized finite element method for the advective–reactive–diffusive equation*. Computer Methods in Applied Mechanics and Engineering 190(13–14), 1785–1800. [DOI: 10.1016/S0045-7825(00)00190-0](https://doi.org/10.1016/S0045-7825(00)00190-0).

- Martin S. Alnæs, Anders Logg, Kristian B. Ølgaard, Marie E. Rognes, and Garth N. Wells (2014). *Unified Form Language: A domain-specific language for weak formulations of partial differential equations*. ACM Transactions on Mathematical Software 40(2), article 9, 1–37. [DOI: 10.1145/2566630](https://doi.org/10.1145/2566630). [Author preprint: arXiv:1211.4047v2](https://arxiv.org/abs/1211.4047v2).

- M. A. Christie and M. J. Blunt (2001). *Tenth SPE Comparative Solution Project: A Comparison of Upscaling Techniques*. SPE Reservoir Evaluation & Engineering 4(4), 308–317. [SPE10 dataset and benchmark description](https://www.spe.org/web/csp/datasets/set02.htm).

- [Basix documentation](https://docs.fenicsproject.org/basix/main/): reference finite elements, basis tabulation, degree-of-freedom transformations and mappings.
- [DOLFINx documentation](https://docs.fenicsproject.org/dolfinx/): native finite element assembly and distributed mesh/function interfaces.
- [PETSc documentation](https://petsc.org/release/): distributed matrices, solvers and preconditioners.
- [Gmsh documentation](https://gmsh.info/doc/texinfo/): geometry, mesh generation and physical groups.
- [meshio project](https://github.com/nschloe/meshio): supported mesh exchange formats and cell/point metadata.

- Willy Dörfler (1996). *A Convergent Adaptive Algorithm for Poisson’s Equation*. SIAM Journal on Numerical Analysis 33(3), 1106–1124. [DOI](https://doi.org/10.1137/0733054).

## Software provenance

Article data, executed reference programs, and independently written verification
forms are different sources of evidence. The comparisons identify their code by
name and revision rather than treating every external result as one reference
implementation.

| Code or source | Verified role in this repository |
|---|---|
| MSL: `msl_mhm` at `4cb8cf81518284313b680b13fd586ee619f08b99`, `msl_cg` at `afb76d14c1baf50f0b9e69f7bcac675749ef4458`, `msl_core` at `7f15f455717173d29080d411a7e732c72c1e87f8` | Executed primal MHM Darcy reference: MSL global coupling, continuous Galerkin local solves, and crisscross geometry. The [field comparison](cases/reference-comparison.md) uses the same discrete spaces and weak Dirichlet moments as pyMHM. |
| MSL_MHM + MSL_CG (GaLS), at the same pinned MSL revisions above | [Independent displacement–pressure comparison](cases/elasticity-reference.md) with a recorded mixed-field MHM adapter, five P1/P1 macro meshes and P2/P2/P3/P3 checks. Native GaLS element assembly is unchanged. |
| MSL_CG + MSL_Core, at the same pinned revisions above | [SPE10 Darcy flux comparison](cases/spe10-flux.md) using native global conforming triangular P1 assembly and Eigen SparseLU on five pixel-aligned meshes. This classical reference is distinct from MSL's MHM coupling. |
| `msl_mfem` at `b9a67e7079c7e487e4ab1679c1bb3c880cc1909a`, with MFEM 4.9 | Executed as an auxiliary strong-Dirichlet reference. Its boundary enforcement differs from the five-mesh MSL comparison above; its fields are not substituted for that comparison. |
| `mhm-mfem` at `fb535acec1be87c19b3aa538ac73a265570e9fdc`, with MFEM 4.9 | A separate MFEM-based flow implementation. No validated Stokes field comparison is available for this revision. It is distinct from the historical 2017 equal-order implementation. |
| [DOLFINx/UFL](https://docs.fenicsproject.org/dolfinx/) | Independently written finite element assemblies for [Darcy](cases/darcy-audit.md), [Stokes](cases/flow-audit.md), [tetrahedral full-saddle Darcy](cases/reconstruction3d.md), [3D flow](cases/flow3d.md), [GaLS3D](cases/gals3d.md) and the dedicated wave operators, plus local assembly through pyMHM's adapter. The recorded native comparisons identify DOLFINx 0.9.0 and their actual solvers. |
| [NeoPZ at `4c6b6d2`](https://github.com/labmec/neopz/tree/4c6b6d277ce097b97bfc8dea1b6725860f4fe05a) | Executed native RT0/P0 assembly, including restricted macro traces: [mixed comparison](cases/neopz.md) and [classical SPE10 reference](cases/spe10-flux.md). The [tetrahedral/prismatic study](cases/mixed-well-geometries.md) also executes native mixed spaces and `TPZMHMixedMeshControl`; its records distinguish NeoPZ assembly/restriction/field evaluation from the linear solver used in each comparison. |
| [Labmec/MHM at `f978f29`](https://github.com/labmec/MHM/tree/f978f29d657d28fe58bcea20fabee68953093482) | Source-level description of the mixed MHM controller and application settings. The executed NeoPZ RT0 driver is distinct from this positive-order application. |
| The Darcy 2013 and Stokes 2017 articles | Published curves digitized and compared with new pyMHM calculations in the [paper comparison](cases/reproduction.md). The historical coefficient arrays and diagnostic programs for those particular figures have not been recovered. |

The DOLFINx assemblies are original verification code using a separate finite
element library. A monolithic solve that reuses pyMHM's local matrices checks
condensation, whereas an independently assembled UFL operator also checks
element assembly. The case pages state which of these checks was performed.
The analytical scalar, heat and displacement-elasticity galleries use pyMHM
and exact formulas; they are not executions of the reference packages above.
The MSL repositories are not public. Their names and revisions identify the
executed references; the documented results do not imply distribution of their
source code or comparison drivers with this package.

The [coarse cosine comparison](cases/coarse-cosine.md) also executes the diagonal
mesh with constant traces. Its MSL adapter shares the assembly of prescribed
weak boundary moments with the sine comparison. The numerical libraries remain
unchanged; the reports identify the reference revisions and adapter digests.
