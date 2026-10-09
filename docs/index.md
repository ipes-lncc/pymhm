---
title: PyMHM
hide:
  - toc
---

# Composable multiscale methods

<div class="pymhm-hero">
  <div class="pymhm-signature-mobile">
    <img src="assets/branding/pymhm-wordmark.svg" width="938" height="335" alt="PyMHM">
    <p>Composable Multiscale Hybrid Mixed<br>finite element methods in Python</p>
  </div>
  <img class="pymhm-signature pymhm-signature--light" src="assets/branding/pymhm-logo.svg" width="938" height="444" alt="PyMHM — Composable Multiscale Hybrid Mixed finite element methods in Python">
  <img class="pymhm-signature pymhm-signature--dark" src="assets/branding/pymhm-logo-dark.svg" width="938" height="444" alt="PyMHM — Composable Multiscale Hybrid Mixed finite element methods in Python">
</div>

`pymhm` is a research Python package for Multiscale Hybrid Mixed finite element
methods. User-defined local and global equations declare their independent
variational blocks, trace pairings and retained modes. `assemble` and `solve`
reuse the shared elimination and reconstruction operations. A local operator
can itself be another multiscale problem. Local discretization, skeleton
approximation and linear solvers are separate choices.

<div class="pymhm-start-grid">
  <a href="installation/">
    <strong>Install PyMHM</strong>
    <span>Set up the portable core and choose optional native backends.</span>
  </a>
  <a class="pymhm-tutorial-card" href="tutorials/overview/">
    <strong>Start the tutorials</strong>
    <span>New to PyMHM? Learn the API step by step with introductory cases, rendered code and field plots.</span>
  </a>
  <a href="gallery/">
    <strong>Explore the Gallery</strong>
    <span>Browse problems and dimensions, field plots, convergence and scientific limits.</span>
  </a>
</div>

The reference implementation uses NumPy, SciPy and Basix. The optional
[FEniCS interface](fenics.md) assembles user-defined scalar, vector, mixed and
H(div) local forms with UFL/DOLFINx. [Meshing](meshing.md) connects Gmsh, Netgen
and meshio.

The [Windows guide](windows.md) covers the portable core, SciPy/PyPardiso,
spawn workers and installed-wheel verification, with the native FEM scope
stated separately.

Start with the [API overview tutorial](tutorials/overview.md) and the
[rendered introductory course](tutorials/notebooks.md) for scalar and vector
formulations and interchangeable local providers. Use the
[visual case gallery](gallery/index.md) to compare numerical fields
with exact references, inspect profiles and errors, and read what each case is
expected to demonstrate.

The [overview](tutorials/overview.md) introduces UFL from its weak form.
The portable coefficient interface follows the same mesh-first workflow;
mesh/space binding owns shared numbering and orientation maps:

```python
import numpy as np
from pymhm import (
    CartesianMacroMesh, Equation, FaceSpace, LocalContext, LocalEquations,
    MeshHierarchy, SkeletonSpace, assemble, bind_interface, bind_problem, solve,
)

macro = CartesianMacroMesh(1, 1)
hierarchy = MeshHierarchy(macro, (macro.submesh(0, 2),))
skeleton = SkeletonSpace(macro, tuple(FaceSpace.uniform(0) for _ in macro.faces))
interface = bind_interface(skeleton, convention="value")

def local(context: LocalContext) -> LocalEquations:
    """Declare independent volume and interface equations."""
    return context.equations(
        a=[[2.0]], L=[1.0], b=np.ones((1, 4)), c=-np.ones((4, 1)),
        d=np.eye(4),
    )

problem = bind_problem(hierarchy, interface, local, global_equation=Equation(0, 0))
system = assemble(problem)
solution = solve(system)
print(solution.trace, solution.fields)  # every coordinate is 1/6
```

This small algebraic example declares $2u+\sum_F\lambda_F=1$ and
$-u+\lambda_F=0$ on each face. The binding owns the shared numbering and geometric
maps; the user supplies both equations. The
[variational guide](variational.md) describes UFL forms, retained modes,
physical constraints and recursive problems, and states the supported limits.
For complete control, see the [custom-space tutorial](tutorials/custom-interface.md),
which declares a nonorthogonal basis and its independent trial/test maps.

The [vector UFL notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/vector_ufl.ipynb)
declares a coercive two-component reaction-diffusion operator; it is separate
from the mixed Brinkman formulations. The
[three-level hierarchy](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/variational_hierarchy.ipynb)
checks recursive coefficients against an independently written full system.

## Available Methods

Choose the variational method and its admissible approximation spaces independently
of the linear solver and execution backend. **When publishing results obtained
with PyMHM, cite the original publications of every method, reconstruction or
estimator used**, together with the PyMHM version. The [Theoretical Background](theory.md) states degree, geometry and regularity
conditions; each linked tutorial follows the mathematical formulation in code.

| Method | Global/local construction | Original publication |
| --- | --- | --- |
| [Primal MHM](tutorials/methods/primal-mhm.md) | Normal-flux skeleton; conforming scalar local fields | [Harder et al. (2013)](https://doi.org/10.1016/j.jcp.2013.03.019) |
| [Mixed H(div) MHM](tutorials/methods/mixed-mhm.md) | Restricted normal traces; RT/BDM local flux and pressure | [Durán et al. (2019)](https://doi.org/10.1016/j.cma.2019.05.013) |
| [Robin MH](tutorials/methods/robin-mh.md) | Coercive Robin locals and a face multiplier system | [Barrenechea et al. (2024)](https://doi.org/10.1137/22M1542556) |
| [MH²M](tutorials/methods/mh2m.md) | Independent continuous pressure and broken conormal traces | [de Barros et al. (2026, v3)](https://arxiv.org/abs/2404.16978v3) |
| [MsHHO](tutorials/methods/mshho.md) | Cell/face pressure moments and constrained energy reconstruction | [Cicuttin, Ern and Lemaire (2019)](https://doi.org/10.1515/cmam-2018-0013); [MHM connection (2022)](https://doi.org/10.1051/m2an/2021082) |
| [Petrov–Galerkin MHM](tutorials/methods/pgmhm.md) | Face residual enrichment and independent global test equations | [Fernando et al. (2023)](https://doi.org/10.1007/s40314-023-02304-y) |
| [MHM-USFEM (MHM-UNUSUAL)](tutorials/methods/mhm-usfem.md) | Residual-stabilized reaction–diffusion local problems | [Santiago et al. (CILAMCE 2025)](https://doi.org/10.55592/cilamce2025.v5i.14270) |
| [Stokes–Brinkman MHM](tutorials/methods/stokes-brinkman.md) | Taylor–Hood or consistent USFEM velocity/pressure locals | [Araya et al. (2017)](https://doi.org/10.1016/j.cma.2017.05.027) |
| [Oseen MHM](tutorials/methods/oseen.md) | Prescribed convection, mixed locals and Robin pseudotraction | [Araya et al. (2021)](https://doi.org/10.1007/s10444-020-09833-8) |
| [Primal elasticity MHM](tutorials/methods/primal-elasticity.md) | Displacement locals, traction skeleton and rigid modes | [Harder et al. (2016)](https://doi.org/10.1051/m2an/2015046) |
| [GaLS elasticity MHM](tutorials/methods/gals-elasticity.md) | Displacement–pressure locals for nearly incompressible materials | [Gomes et al. (2024, v1)](https://arxiv.org/abs/2403.16890v1) |
| [Mixed stress elasticity MHM](tutorials/methods/mixed-elasticity.md) | H(div) stress, displacement and weak-symmetry rotation | [Devloo et al. (2021)](https://doi.org/10.1051/m2an/2021013) |
| [Conservative transport MHM](tutorials/methods/transient-transport.md) | Galerkin/SUPG RAD and backward-Euler transient extension | [Harder et al. (2015)](https://doi.org/10.1137/130938499) |
| [Elastodynamic MHM](tutorials/methods/elastodynamics.md) | Local Newmark dynamics and slabwise traction coupling | [Gomes et al. (2017)](https://doi.org/10.20906/CPS/CILAMCE2017-0399) |
| [Helmholtz MHM](tutorials/methods/helmholtz.md) | Complex local wave operators and polynomial/oscillatory traces | [Chaumont-Frelet and Valentin (2020)](https://doi.org/10.1137/19M1255616) |
| [Maxwell MHM](tutorials/methods/maxwell.md) | Tangential coupling, central-DG locals and leapfrog dynamics | [Lanteri et al. (2018)](https://doi.org/10.1137/16M110037X) |
| [Unfitted/face-refined MHM](tutorials/methods/unfitted.md) | Material-fitted subfaces on an unfitted macro mesh | [Chaumont-Frelet et al. (2026)](https://doi.org/10.1016/j.camwa.2026.01.016) |

Recovery and adaptive strategies complement those discretizations:

| Strategy | Purpose | Original analysis |
| --- | --- | --- |
| [Flux and potential recovery](tutorials/methods/flux-recovery.md) | RT0 equilibration, moment H(div) reconstruction and conforming potentials | [Barrenechea et al. (2026)](https://doi.org/10.1137/24M1673073) |
| [Error indicators](tutorials/methods/error-indicators.md) | Field-specific residual and reconstructed-energy diagnostics | [Araya et al. (2013)](https://doi.org/10.1137/120888223) |
| [Adaptive approximation](tutorials/methods/adaptivity.md) | Independent macro, face, degree and local controls | [Araya et al. (2021)](https://doi.org/10.1093/imanum/drz053) |
| [Recursive MHM](tutorials/methods/recursive-mhm.md) | A multiscale local operator with its own global/local hierarchy | [Harder and Valentin (2016)](https://doi.org/10.1007/978-3-319-41640-3_13) |

The table identifies implemented families, not every theorem or historical
experiment in their papers. The [Gallery](gallery/index.md) records verified
spaces and remaining limitations. RT0 equilibration and moment reconstruction
are distinct operators; a generic indicator does not inherit the reliability
proof of a complete published estimator.

## Executable formulations

The following predefined physical formulations have numerical records for
their stated discretizations. They are separate from the ability to
declare a new variational problem; their case evidence does not qualify
arbitrary user-supplied forms.

| Problem | Local approximation | Skeleton / normalization |
| --- | --- | --- |
| Darcy | Primal Pk, triangular RT0–RT2/BDM2 and enriched rectangular RT | Scalar normal flux; physical local means; pure Neumann gauge |
| Three-dimensional primal Darcy | Conforming local tetrahedral Pk | Continuous or discontinuous polynomials within independently partitioned triangular macrofaces |
| Three-dimensional mixed Darcy | Tetrahedral 18/P1 and 32/P2 families, prismatic 27/W11 and mapped hexahedral RT | Oriented Piola normal traces; physical cell moments and production balances |
| Cartesian Darcy | Conforming local Qk on rectangular grids | Continuous or discontinuous face traces; geometric material-pixel integration |
| Stokes–Brinkman | Pk/P(k−1) Taylor–Hood or Pk/Pk USFEM | Vector pseudotraction; tensor resistance; global pressure mean for full velocity data |
| Oseen with variable advection | Taylor–Hood or consistent equal-order stabilization | Robin pseudotraction, prescribed convection derivatives and global pressure mean |
| Nearly incompressible elasticity | GaLS Pk/Pk or Taylor–Hood | Three rigid modes; negative Cauchy traction; physical compressibility constraint |
| Weakly symmetric elasticity | Row-wise BDM/RT stress with independent interior enrichment | Restricted interior tractions; force, moment and weak-symmetry equations |
| Primal tensor elasticity | Two- and three-dimensional Pk with physical rigid modes | Negative Cauchy traction and displacement moments; no general locking-free claim |
| Three-dimensional weakly symmetric elasticity | Row-wise BDMk stress, discontinuous displacement and rotation | AFW spaces with k≥2, six rigid modes and the incompressible limit |
| Elastodynamics | Two- and three-dimensional Pk with Newmark time integration | Slabwise traction, local subcycling and heterogeneous inertia |
| Conservative scalar RAD | Pk Galerkin or SUPG with variable coefficients | Robin multiplier with half the advective flux |
| Heat equation | Pk and backward Euler | Flux skeleton, prescribed scalar boundary values |
| Equilibrated Darcy flux | Constrained minimum-energy RT0 reconstruction | Fine-cell balances and preserved macro fluxes |
| MsHHO | Cell and face pressure moments with constrained local energy reconstruction | Condensed face-pressure moment system; conditional field equivalence with MHM |
| Adaptive methods | RT moment recovery, potential estimators and equation-specific residual indicators | Independent macro, face and local controls with stated marking rules |

Built-in geometries include triangles, rectangular cells, simple planar polygons,
tetrahedra, affine prisms, star-shaped polyhedra with planar faces and trilinearly
mapped hexahedra.
Nonconvex polyhedra require a certified positive-volume kernel and a conforming
tetrahedral decomposition, as described in the [polyhedral case](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/star-polyhedra.md).
The [scope matrix](https://github.com/ipes-lncc/pymhm/blob/main/ROADMAP.md#scientific-scope-and-acceptance-criteria) identifies the formulations available on each
geometry; they are not interchangeable backends for every equation.
Face partitions and polynomial degrees are independent of local refinement.
Planar edge bases can be discontinuous
Legendre polynomials or continuous nodal polynomials within each macroface.
Triangular macrofaces likewise support discontinuous Bernstein polynomials or
continuous piecewise polynomials, with independent degrees and continuity
choices per face.
Normal-trace restrictions and local refinement must satisfy each formulation's
compatibility conditions.

[Scientific scope](https://github.com/ipes-lncc/pymhm/blob/main/ROADMAP.md#scientific-scope-and-acceptance-criteria) maps the literature to the implemented paths and
remaining mathematical requirements. The [Bibliography](literature.md) identifies the original formulations, analyses,
reconstruction strategies and supporting numerical methods; a citation is not
a claim of complete paper reproduction.

## Evidence and reproducibility

Use [verification](verification.md) for measured errors, conservation tests,
convergence studies and backend integration results. Coverage measures executable
branches; it does not prove stability, validate an unavailable backend or
reproduce a paper. [Performance](https://github.com/ipes-lncc/pymhm/blob/main/docs/performance.md) records timings, including cases
where parallel execution is slower.

Executed comparisons identify their reference implementations explicitly:
[MSL_MHM with MSL_CG and MSL_Core](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/reference-comparison.md) for primal
Darcy, and [NeoPZ](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/neopz.md) for conforming and restricted-trace RT0/P0
Darcy. Their case pages record the source revisions, discrete spaces, boundary
conditions and complete-field comparisons. Labmec/MHM's positive-order
controller has a separate source-level description from the executed RT0 NeoPZ driver.

This is version 1.3.0, an official release of PyMHM. The
[installation guide](installation.md) explains package installation through pip,
optional backends and the locked environments used for reproducible studies.

The [MSL GaLS comparison](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/elasticity-reference.md) checks displacement,
pressure, gradients and full stress for P1/P1, P2/P2 and P3/P3 elasticity.
The [near-incompressibility study](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/elasticity.md) includes finite material
ratios through $10^8$, the exact incompressible limit and six refinement points.
[BDM2 Darcy](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/darcy-bdm.md) and [mixed elasticity](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/mixed-elasticity.md)
include independent DOLFINx/Basix assembly checks and analytical convergence cases.

<section class="institutional-support" markdown="1">

Original figures and diagrams are © IPES Research Group and licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Commercial reuse and
adaptations require attribution; see [figure reuse and attribution](licensing.md).

## Institutional Support

PyMHM is developed by the
[Innovative Parallel numErical Solvers (IPES)](https://ipes.lncc.br/)
research group and receives institutional support from the
[Laboratório Nacional de Computação Científica (LNCC)](https://www.gov.br/lncc/pt-br),
a research unit of the
[Ministério da Ciência, Tecnologia e Inovação (MCTI)](https://www.gov.br/mcti/pt-br),
Brazil.

<div class="institutional-logos">
  <a href="https://ipes.lncc.br/">
    <img src="assets/institutions/ipes.png" width="1671" height="631" loading="lazy" alt="IPES — Innovative Parallel numErical Solvers">
  </a>
  <a href="https://www.gov.br/lncc/pt-br">
    <img src="assets/institutions/lncc.svg" loading="lazy" alt="LNCC — Laboratório Nacional de Computação Científica">
  </a>
  <a href="https://www.gov.br/mcti/pt-br">
    <img src="assets/institutions/mcti.svg" width="460" height="120" loading="lazy" alt="MCTI — Ministério da Ciência, Tecnologia e Inovação">
  </a>
</div>

</section>
