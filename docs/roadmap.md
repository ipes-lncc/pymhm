# Scientific scope and acceptance criteria

PyMHM separates local finite element problems from an explicit skeletal coupling.
A supported family needs a concrete operator, compatible spaces and numerical
evidence. An optional dependency or a generic matrix adapter alone does not
supply a finite element discretization.

The [initial convergence catalogue](cases/minimal-convergence.md) reports the
current short studies, including unresolved numerical increments and incomplete
levels. The extended historical campaigns and performance sweeps are separate
from that evidence. The [case evidence guide](cases/index.md) maps representative calculations to the
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
| Material interfaces | Exact integration over Cartesian/planar intersections, material-fitted local meshes and explicit macroface partitions. [Unfitted](https://github.com/volpatto/pymhm/blob/main/docs/cases/unfitted.md) integration alone does not resolve a gradient jump in an uncut polynomial cell. |
| Reconstruction and estimation | RT moment recovery, Oswald potentials and distinct published, energy-weighted and face-jump indicators. [Estimator hypotheses](https://github.com/volpatto/pymhm/blob/main/docs/cases/reconstruction3d.md) require `k >= ell + d`; algebraic RT construction is weaker. Continuous-test equilibrium and fine-cell DG balance are different conditions. |
| Alternative multiscale formulations | [MsHHO](cases/mshho.md), [Robin MH](https://github.com/volpatto/pymhm/blob/main/docs/cases/mh.md), [MH²M](https://github.com/volpatto/pymhm/blob/main/docs/cases/mh2m.md) and [PGMHM](https://github.com/volpatto/pymhm/blob/main/docs/cases/pgmhm.md), with their own face/cell/source conventions. Equivalence and injectivity depend on the stated spaces and source assumptions. |
| RAD and transport | Conservative Pk Galerkin/SUPG, tensor diffusion, reaction and explicit coefficient derivatives; [MHM-USFEM](https://github.com/volpatto/pymhm/blob/main/docs/cases/unusual.md) uses the full unusual residual form. Stabilization does not imply a maximum principle. |
| Transient scalar problems | Backward Euler, positive capacity, changing loads/boundaries and prepared spatial operators. [Darcy coupling](https://github.com/volpatto/pymhm/blob/main/docs/cases/transient-transport.md) distinguishes volume flux from the numerical normal trace. Manufactured time convergence does not reproduce an unavailable random realization. |
| Stokes–Brinkman/Oseen | Taylor–Hood and full-residual equal-order USFEM in 2D/3D, tensor resistance, prescribed convection, component slip and physical pressure gauges. [Flow adaptation](https://github.com/volpatto/pymhm/blob/main/docs/cases/stokes-adaptive.md) assumes constant viscosity, full Dirichlet data, uniform trace degree and resolved jumps. No nonlinear Navier–Stokes iteration is claimed. |
| Primal and displacement–pressure elasticity | General material tensors, physical traction and three/six rigid modes; GaLS/Taylor–Hood and finite/infinite bulk limits. [Primal displacement](https://github.com/volpatto/pymhm/blob/main/docs/cases/primal-elasticity.md) alone is not uniformly locking-free. |
| Mixed elasticity | Row-wise BDM/enriched/rectangular RT stress, weak rotation, anisotropic compliance and [tetrahedral AFW](https://github.com/volpatto/pymhm/blob/main/docs/cases/mixed-elasticity3d.md). Displacement must represent rigid modes; weak symmetry is a moment condition. Classical AFW stability is not a theorem for arbitrary MHM traces. |
| Polygonal/polyhedral geometry | Straight-sided simple polygons, including nonconvex cells, and [certified star-shaped polyhedra](https://github.com/volpatto/pymhm/blob/main/docs/cases/star-polyhedra.md) with original polygonal face spaces. Empty-kernel cells, cavity shells and curved faces are outside the polyhedral path. |
| Helmholtz | Complex triangular/polygonal Pk and Cartesian Qk locals, polynomial/oscillatory traces, absorbing boundaries, diagonal PML and local resonance checks. [Wave](https://github.com/volpatto/pymhm/blob/main/docs/cases/helmholtz.md) and [Marmousi](https://github.com/volpatto/pymhm/blob/main/docs/cases/marmousi.md) comparisons retain their distinct data contracts. |
| Maxwell and elastodynamics | Tangentially coupled central-DG dynamics with mass-scaled CFL, and Newmark local responses with slabwise traction/substeps. [Maxwell](https://github.com/volpatto/pymhm/blob/main/docs/cases/maxwell.md) and [elastodynamics](https://github.com/volpatto/pymhm/blob/main/docs/cases/elastodynamics.md) state their analytical and heterogeneous comparison scopes separately. |

See the [API](api.md), [FEniCS local forms](fenics.md), [meshing](meshing.md)
and [linear solvers](solvers.md) pages for construction and backend details.

## Historical targets still requiring evidence

Scientific completion is tied to a particular experiment, including its input
fields, mesh, spaces, quadrature, boundary convention, gauge and norm. The following
limits remain material to claims about the literature:

- The [periodic](https://github.com/volpatto/pymhm/blob/main/docs/cases/periodic.md), [oscillatory well](https://github.com/volpatto/pymhm/blob/main/docs/cases/mapped-well-oscillatory.md)
  and [SPE10](https://github.com/volpatto/pymhm/blob/main/docs/cases/spe10-adaptive.md) comparisons retain nonzero classical-reference
  refinement increments and historical input/mesh qualifications. Local or face
  enrichment controls are different discretizations from the published setup.
- The literal L11 mixed-wall curve remains quantitatively different; its ε=1
  control is separate. The [transient random-field example](https://github.com/volpatto/pymhm/blob/main/docs/cases/transient-transport.md)
  requires a complete same-case acquisition and independent reference.
- The [Stokes stress diagnostic](https://github.com/volpatto/pymhm/blob/main/docs/cases/reproduction.md), [GaLS elasticity
  amplitudes/stabilization](https://github.com/volpatto/pymhm/blob/main/docs/cases/elasticity.md) and historical exterior traces in
  [mixed elasticity](https://github.com/volpatto/pymhm/blob/main/docs/cases/mixed-families.md) prevent unqualified reproduction of
  every printed ordinate or table.
- The [Helmholtz angular ordinates](https://github.com/volpatto/pymhm/blob/main/docs/cases/helmholtz.md), [Marmousi material crop
  and 15-case family](https://github.com/volpatto/pymhm/blob/main/docs/cases/marmousi.md), [Maxwell incident-wave conventions and
  complete comparison](https://github.com/volpatto/pymhm/blob/main/docs/cases/maxwell-nanoguide.md), and heterogeneous [2017
  elastodynamic source case](https://github.com/volpatto/pymhm/blob/main/docs/cases/elastodynamics.md) have their own unresolved
  data or comparison requirements. Analytical wave tests do not replace them.
- [Historical cluster scaling](execution.md) on 24–768 cores and the largest 3D
  meshes is not reproduced. Available MPI/GPU measurements concern their declared
  hardware and workloads; they do not establish an unmeasured speedup.

The bibliographic catalog and detailed case pages retain the theorem hypotheses
and numerical evidence behind each statement. Historical discrepancies are
reported with independently assembled same-case comparisons where available;
missing reference acquisition programs must be supplied before a clean execution
can reproduce those comparisons.

## Acceptance by literature target

Each row identifies a formulation, its physical case, the available comparison
and the remaining acceptance condition. The [catalog](literature.md) identifies
the publications and reference-code revisions. Numerical summaries retain their
acquisition provenance; reproducing their results requires the corresponding
fields and reference programs. A checked-in summary alone does not verify a new
execution. Analytical tests and local operator comparisons remain separate from
complete published-case comparisons.

| Target and formulation | Case | Comparison and recorded evidence | Remaining acceptance condition |
| --- | --- | --- | --- |
| L01: primal Darcy MHM | Cosine, quarter five-spot, square obstacle and rough coefficients | Analytical fields, published curves, and complete square-obstacle Basix/P1 and native NeoPZ/RT0 comparisons on stated spaces | Resolve classical obstacle refinement and the independent point-well comparison; identify historical random data before literal reproduction. |
| L02: elliptic error estimation | Cosine, inclusion and Dirac wells | [Reconstruction and estimators](https://github.com/volpatto/pymhm/blob/main/docs/cases/reconstruction3d.md), analytical errors and indicator controls | Repeat the same-case estimator comparison and reference refinement; retain the Dirac regularity limitation. |
| L03: abstract hybrid algebra | Local/global and recursive systems | Independent full Petrov–Galerkin algebra, retained kernels and source lifts | User-defined operators still require kernel, compatibility and stability hypotheses; no published numerical table is supplied. |
| L04: periodic Darcy robustness | [Periodic permeability](https://github.com/volpatto/pymhm/blob/main/docs/cases/periodic.md), fixed macrogrid and face enrichment | Phased basis replay, complete 64-macro independent Q1/P0 assembly and current five-level Q1 controls; Q5 accuracy is outside the current acceptance | Resolve local and classical-reference refinement; finite local/trace pairs require injectivity. The historical reference is Q1 on 4096² elements, with unidentified historical local refinement. |
| L05: mixed local Darcy | Rectangular, tetrahedral and prismatic wells | [Mixed wells](https://github.com/volpatto/pymhm/blob/main/docs/cases/mixed-well-geometries.md), analytical fields and native NeoPZ spaces | Complete matched whole-case reference acquisition and interior/face/quadrature controls with compatible divergence spaces. |
| L06: MHM–MsHHO connection | Elliptic macro and skeletal refinement | Analytical MsHHO/MHM comparisons with declared source spaces | Repeat convergence and same-case independent assembly under the equivalence hypotheses. |
| L07: face-based robustness | Periodic medium and SPE10 layer 36 with continuous face interpolation | [Darcy flux](https://github.com/volpatto/pymhm/blob/main/docs/cases/spe10-flux.md), Q1/C0-P1 MHM and Q3/MSL/NeoPZ references | Resolve local/face refinement and the classical-reference increment while retaining the published 66-square macro partition and material. |
| L08: unfitted preprint | Two-layer interface problem | [Unfitted study](https://github.com/volpatto/pymhm/blob/main/docs/cases/unfitted.md), analytical series and independent UFL | Identify preprint-specific data and hypotheses separately from final L10; the final regularity endpoint differs. |
| L09: H(div) recovery and adaptivity | [Adaptive SPE10](https://github.com/volpatto/pymhm/blob/main/docs/cases/spe10-adaptive.md) | Published P2/r2/P0 spaces, RT2/Oswald recovery and classical-reference controls | Repeat BAMG acquisition and full independent initial/final systems; estimator estimates require `k >= ell + d`. |
| L10: unfitted flux approximation | Smooth h/p sweeps and two-layer contrast | [Unfitted study](https://github.com/volpatto/pymhm/blob/main/docs/cases/unfitted.md), analytical norms, native assemblies and full 16-macro q11/q13 control on admissible P8/r16–P2/s32 | Resolve local r32→r64 on admissible spaces; P8/r16–P3/s32 has an exact multiplier kernel. Reconcile S2 or compare the complete selected case independently. |
| L11: advective/reactive MHM | Mixed walls and random Darcy–transport | [Transport](https://github.com/volpatto/pymhm/blob/main/docs/cases/transient-transport.md), analytical and coefficient controls | Acquire the complete §5.4 realization, coupled trajectory and independently refined space/time/quadrature reference. |
| L12: generalized RAD | Polygonal/polyhedral diffusion and reaction layers | [Polytopes](https://github.com/volpatto/pymhm/blob/main/docs/cases/polygons.md), conditioning, boundary and layer controls | Repeat conditioning/layer/star-polyhedron studies and independent fields for the matched cases. |
| L13: equal-order Stokes–Brinkman | Smooth flow and SPE10 layer one | [SPE10](https://github.com/volpatto/pymhm/blob/main/docs/cases/spe10.md), published-space fields and Taylor–Hood baseline | Refine the same-BC baseline, independently integrate field differences and qualify missing historical stabilization constants. |
| L14: multilevel flow estimator | [Stokes adaptation and cavity](https://github.com/volpatto/pymhm/blob/main/docs/cases/stokes-adaptive.md) | Analytical indicators and classical cavity comparisons | Repeat adaptation and reference refinement with the same constant lid, physical pressure gauge and corner cutout. |
| L15: adaptive Oseen | Smooth and boundary-layer flow | [Oseen](https://github.com/volpatto/pymhm/blob/main/docs/cases/oseen.md), analytical refinement and independent operators | Repeat layer/adaptive whole-case controls, quadrature and physical norms. |
| L16: flow a priori analysis | Admissible Stokes–Brinkman spaces | Analytical/native flow verification and stated degree conditions | Verify the theorem's discretization and regularity conditions for each rate claim. |
| L17: primal elasticity | Analytical displacement and HPC4E | [HPC4E](https://github.com/volpatto/pymhm/blob/main/docs/cases/hpc4e.md), analytical fields and independent classical stress baseline | Complete same-case stress/compliance/work comparisons and refine the classical reference. |
| L18: weakly symmetric mixed elasticity | Tensor families and oscillatory Table 3 case | [Mixed elasticity](https://github.com/volpatto/pymhm/blob/main/docs/cases/mixed-families.md), native fields and analytical norms | Repeat BDM/RT/enrichment and tensor controls; retain the printed rotation-column inconsistency explicitly. |
| L19: locking-free elasticity | Finite/infinite bulk limits | [GaLS elasticity](https://github.com/volpatto/pymhm/blob/main/docs/cases/elasticity.md), mixed-field and native comparisons | Execute matched Lamé sweeps and refinement; an affine primal patch does not establish uniform locking freedom. |
| L20: scalable implementation | Distributed local/global execution | [Execution](execution.md), spawn/MPI contracts and workload-specific measurements | Reproduce setup, transfers, synchronization and scaling at matched accuracy; historical cluster scaling remains unverified. |
| Additional MH | Robin hybrid diffusion | Analytical boundary/parameter sweeps and independent UFL | Repeat complete refinement; distinguish the Robin multiplier from physical flux. |
| Additional MH²M | [Oscillatory medium](https://github.com/volpatto/pymhm/blob/main/docs/cases/mh2m-heterogeneous.md) | Crisscross/diagonal cases and refined independent CG3 fields | Repeat same-case acquisition, overlay quadrature and baseline refinement; historical curve differences remain. |
| Additional PGMHM | Inclusions and SPE10 | Analytical/native equations and separately refined classical fields | Acquire matched whole-case comparisons with published enrichment and heterogeneous stabilization. |
| Additional Unusual | [Reaction–diffusion](https://github.com/volpatto/pymhm/blob/main/docs/cases/unusual.md) and SPE10 layers | Analytical/native fields and recorded FreeFem comparisons | Reacquire the full external comparison and verify material/reaction-length/trace resolution. |
| Additional Helmholtz | [Angular and stability studies](https://github.com/volpatto/pymhm/blob/main/docs/cases/helmholtz.md), [Marmousi](https://github.com/volpatto/pymhm/blob/main/docs/cases/marmousi.md) | Plane/Hankel fields, native saddle systems and classical material-crop references | Repeat published angular/stability sequences; validate Marmousi storage phases and reference refinement while retaining historical-input limits. |
| Additional Maxwell | [Nanoguide](https://github.com/volpatto/pymhm/blob/main/docs/cases/maxwell-nanoguide.md) | Analytical dynamics, native operators and recorded DG field comparisons | Supply independent DG Q2 acquisition and complete trajectory, staggered-time, CFL and space/time/quadrature controls. |
| Additional elastodynamics | [Equation (53) and three layers](https://github.com/volpatto/pymhm/blob/main/docs/cases/elastodynamics.md) | Analytical Newmark trajectories, complete independent 341-macro original equations and 301-state common-basis coordinate comparisons for the selected three-layer case | Evaluate both executed field bases separately, verify the conforming reference on several finer meshes and complete space, time and quadrature refinement. Historical heterogeneous inputs remain unresolved. |

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
