# Scientific scope and acceptance criteria

PyMHM separates local finite element problems from an explicit skeletal coupling.
A supported family needs a concrete operator, compatible spaces and numerical
evidence. An optional dependency or a generic matrix adapter alone does not
supply a finite element discretization.

The [case evidence guide](cases/index.md) maps representative calculations to the
[literature](literature.md). Its labels distinguish published comparisons,
matched data, independent references and analytical verification. The selected
cases do not establish complete reproduction of the MHM literature.

## Implemented constructions

The table identifies implemented paths; applicable degrees, boundaries, material
assumptions and executed evidence are specified on their linked pages.

| Construction | Concrete scope and limitations |
| --- | --- |
| Local/global hybrid algebra | Independent trial/test couplings and adjoint kernels, retained physical modes, constrained local solves, source lifts, offline/online reuse and [recursive MHM](cases/nested.md). Supplied operators require their own kernel and stability checks. |
| Primal Darcy | Triangular Pk, rectangular Qk and tetrahedral Pk local pressure, physical pressure means and scalar/SPD permeability; [2D](api/darcy.md), [3D](api/darcy3d.md) and [Cartesian](quadrilateral.md) conventions are explicit. |
| Mixed Darcy | Triangular RT/BDM, enriched rectangular RT, affine tetrahedral/prismatic families and mapped hexahedral RT. [Normal traces, pressure moments and interior enrichment](api/darcy3d.md) remain independent choices. Nonaffine prisms and pyramidal mixed elements are outside this path. |
| Material interfaces | Exact integration over Cartesian/planar intersections, material-fitted local meshes and explicit macroface partitions. [Unfitted](cases/unfitted.md) integration alone does not resolve a gradient jump in an uncut polynomial cell. |
| Reconstruction and estimation | RT moment recovery, Oswald potentials and distinct published, energy-weighted and face-jump indicators. [Estimator hypotheses](cases/reconstruction3d.md) require `k >= ell + d`; algebraic RT construction is weaker. Continuous-test equilibrium and fine-cell DG balance are different conditions. |
| Alternative multiscale formulations | [MsHHO](cases/mshho.md), [Robin MH](https://github.com/volpatto/pymhm/blob/main/docs/cases/mh.md), [MH²M](https://github.com/volpatto/pymhm/blob/main/docs/cases/mh2m.md) and [PGMHM](https://github.com/volpatto/pymhm/blob/main/docs/cases/pgmhm.md), with their own face/cell/source conventions. Equivalence and injectivity depend on the stated spaces and source assumptions. |
| RAD and transport | Conservative Pk Galerkin/SUPG, tensor diffusion, reaction and explicit coefficient derivatives; [MHM-USFEM](cases/unusual.md) uses the full unusual residual form. Stabilization does not imply a maximum principle. |
| Transient scalar problems | Backward Euler, positive capacity, changing loads/boundaries and prepared spatial operators. [Darcy coupling](cases/transient-transport.md) distinguishes volume flux from the numerical normal trace. Manufactured time convergence does not reproduce an unavailable random realization. |
| Stokes–Brinkman/Oseen | Taylor–Hood and full-residual equal-order USFEM in 2D/3D, tensor resistance, prescribed convection, component slip and physical pressure gauges. [Flow adaptation](cases/stokes-adaptive.md) assumes constant viscosity, full Dirichlet data, uniform trace degree and resolved jumps. No nonlinear Navier–Stokes iteration is claimed. |
| Primal and displacement–pressure elasticity | General material tensors, physical traction and three/six rigid modes; GaLS/Taylor–Hood and finite/infinite bulk limits. [Primal displacement](https://github.com/volpatto/pymhm/blob/main/docs/cases/primal-elasticity.md) alone is not uniformly locking-free. |
| Mixed elasticity | Row-wise BDM/enriched/rectangular RT stress, weak rotation, anisotropic compliance and [tetrahedral AFW](cases/mixed-elasticity3d.md). Displacement must represent rigid modes; weak symmetry is a moment condition. Classical AFW stability is not a theorem for arbitrary MHM traces. |
| Polygonal/polyhedral geometry | Straight-sided simple polygons, including nonconvex cells, and [certified star-shaped polyhedra](https://github.com/volpatto/pymhm/blob/main/docs/cases/star-polyhedra.md) with original polygonal face spaces. Empty-kernel cells, cavity shells and curved faces are outside the polyhedral path. |
| Helmholtz | Complex triangular/polygonal Pk and Cartesian Qk locals, polynomial/oscillatory traces, absorbing boundaries, diagonal PML and local resonance checks. [Wave](cases/helmholtz.md) and [Marmousi](cases/marmousi.md) comparisons retain their distinct data contracts. |
| Maxwell and elastodynamics | Tangentially coupled central-DG dynamics with mass-scaled CFL, and Newmark local responses with slabwise traction/substeps. [Maxwell](https://github.com/volpatto/pymhm/blob/main/docs/cases/maxwell.md) and [elastodynamics](cases/elastodynamics.md) state their analytical and heterogeneous comparison scopes separately. |

See the [API](api.md), [FEniCS local forms](fenics.md), [meshing](meshing.md)
and [linear solvers](solvers.md) pages for construction and backend details.

## Historical targets still requiring evidence

Scientific completion is tied to a particular experiment, including its input
fields, mesh, spaces, quadrature, boundary convention, gauge and norm. The following
limits remain material to claims about the literature:

- The [periodic](cases/periodic.md), [oscillatory well](cases/mapped-well-oscillatory.md)
  and [SPE10](cases/spe10-adaptive.md) comparisons retain nonzero classical-reference
  refinement increments and historical input/mesh qualifications. Local or face
  enrichment controls are different discretizations from the published setup.
- The literal L11 mixed-wall curve remains quantitatively different; its ε=1
  control is separate. The [transient random-field example](cases/transient-transport.md)
  requires a complete same-case acquisition and independent reference.
- The [Stokes stress diagnostic](cases/reproduction.md), [GaLS elasticity
  amplitudes/stabilization](cases/elasticity.md) and historical exterior traces in
  [mixed elasticity](cases/mixed-families.md) prevent unqualified reproduction of
  every printed ordinate or table.
- The [Helmholtz angular ordinates](cases/helmholtz.md), [Marmousi material crop
  and 15-case family](cases/marmousi.md), [Maxwell incident-wave conventions and
  complete comparison](cases/maxwell-nanoguide.md), and heterogeneous [2017
  elastodynamic source case](cases/elastodynamics.md) have their own unresolved
  data or comparison requirements. Analytical wave tests do not replace them.
- [Historical cluster scaling](execution.md) on 24–768 cores and the largest 3D
  meshes is not reproduced. Available MPI/GPU measurements concern their declared
  hardware and workloads; they do not establish an unmeasured speedup.

The bibliographic catalog and detailed case pages retain the theorem hypotheses
and numerical evidence behind each statement. Historical discrepancies are
reported with independently assembled same-case comparisons where available;
missing reference acquisition programs must be supplied before a clean execution
can reproduce those comparisons.

## Numerical acceptance

1. Verify geometry, quadrature, basis orientation, local/adjoint kernels,
   constrained solves, trace signs, physical gauges and admissible spaces.
2. Derive manufactured sources and boundary data from the actual operator.
   Verify original physical equations and saved-field replay, rather than only
   a scaled CSR residual or a patch solution.
3. Refine macro, local and skeletal spaces independently. Measure physical
   pressure/displacement and flux/stress separately, preserving broken traces.
4. Compare matched fields and norms with an independent assembly. Without an
   exact solution, also refine a classical reference and report its own increments.
5. Claim a published reproduction only with matching data and discretization,
   stated extraction uncertainty and compatible rate hypotheses. Singular forcing
   or heterogeneous materials do not inherit smooth-problem rates automatically.

Line and branch coverage of at least 99% is a software quality gate, not a proof of
accuracy, inf-sup stability or literature completion. Optional API contract tests
are accompanied by native integrations on available platforms.

## Execution and distribution

[Execution](execution.md) and [performance](https://github.com/volpatto/pymhm/blob/main/docs/performance.md) record setup, assembly,
factorization, repeated loads, global solve, reconstruction, transfers,
synchronization and peak memory. Compare backends at equal accuracy and reproducible
thread settings. Portable spawn semantics, MPI partitions, operator reuse and
resident GPU batches retain their explicit numerical and platform restrictions.
AMG on admissible positive systems and block preconditioning of saddle systems are
distinct paths; backend availability does not establish acceleration.

The portable core imports without optional FEM, CAD, MPI or accelerator runtimes.
Platform claims require native tests. PyPI readiness requires checked wheel/sdist,
clean-install verification and synchronized metadata; release automation does not
mean an upload has occurred. Conda-forge additionally requires an accepted recipe
and available dependencies on its target platforms.
