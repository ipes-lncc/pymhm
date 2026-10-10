# PyMHM implementation, verification and validation roadmap

Updated **October 10, 2026**. This is the project's canonical roadmap.
The [scientific scope](#scientific-scope-and-acceptance-criteria) and
[literature acceptance matrix](#acceptance-by-literature-target) identify
implemented capabilities and remaining evidence. Detailed results belong on the
[Gallery](docs/gallery/index.md). This roadmap does not certify complete
reproduction of the literature.

## Using this roadmap

Choose a bounded delivery, inspect its existing evidence and define acceptance
criteria before implementation or acquisition. Missing reference data, programs
or hardware do not necessarily imply that a method needs implementation.

Use these states when updating milestones and case records:

- **Implemented:** the capability exists with explicit contracts and restrictions.
- **Verified:** an identified execution verifies the stated scope.
- **Pending:** a specified implementation, comparison or control remains.
- **External dependency:** historical inputs, reference access or resources are missing.

Record the accepted evidence and next action here; keep numerical records on the
case page. Retain accepted acquisitions unless a changed source, input or unresolved
control requires a new execution. Keep this document focused on current scope,
dependencies and acceptance conditions.

## Current baseline

The contextual API follows **meshes → spaces → local/global equations →
assembly → solve → named fields and errors**. `MeshHierarchy`, `bind_interface`
and `bind_problem` supply coordinate maps through `LocalContext` and
`GlobalContext`. Built-in and custom `InterfaceSpace`/`TraceBinding` providers
lower to the existing `Equation`, `LocalEquations` and `MultiscaleProblem`
owners. The fully explicit route remains available. See the
[overview](docs/tutorials/overview.md), [architecture](docs/architecture.md) and
[custom-interface guide](docs/guides/custom-interface.md).

Implemented geometry, physics and backend scopes appear in the table below.
The [Gallery](docs/gallery/index.md) retains numerical results and the
[performance reports](docs/performance.md) identify measured workloads, hardware
and accuracy. The current source includes the contextual interface and portable
native-form assembly; package metadata is synchronized at version 1.3.0.

Local qualification includes **6,517 passing tests**, line coverage of
**99.80%** and branch coverage of **99.24%**, together with lint, formatting,
type checking, native FEM, MPI and two-GPU integrations. The installed wheel
passes its eight portable execution and PARDISO checks outside the checkout.
The standalone wheel-ownership test requires that isolated installation;
its skip in the checkout suite is not a missing dependency. The
[API qualification record](benchmarks/results/api-binding-20261006/README.md)
identifies the earlier numerical controls and their scope. Verification applies
to the executed cases; it does not establish stability or resolution in other
regimes. The documentation is organized into Home, Getting Started, Theoretical
Background, Guides, Tutorials, Gallery, Development, API and Bibliography.
Seventeen formulation tutorials and four recovery/adaptation/hierarchy lessons
pair the equations with their implementation. Attributed refinement evidence
distinguishes smooth spatial approximation, temporal integration, independent
trace refinement and differences to qualified numerical references. Physical
conservation, recovery errors, estimator effectivity and adaptive error against
work accompany the applicable convergence estimates.
The [theory](docs/theory.md) and individual tutorials distinguish spatial rates,
temporal rates, trace refinement and observations without a theorem claim.
The twenty-one formulation refinement series retain every measured level,
successive orders and target-normalized errors. Smooth Stokes qualification
includes both Taylor–Hood and stabilized local spaces; difficult Brinkman layers
remain a separate resolution study. Matched conforming references include the
homogeneous anisotropic primal-elasticity family. The recovery lessons verify
RT moment errors, projected divergence, distinct fine-cell RT0 conservation,
adaptive error against work and equivalence of physical recursive condensation.
Historical scaling measurements retain their original source revisions.
All 114 downloadable notebooks have checksum-pinned support archives. Existing
tutorial receipts record complete executions of 25 notebooks at their identified
source checksums. The CPU-kernel performance notebook defines and executes its
own 2D/3D workflow. Numerical execution receipts and download archive integrity
are verified separately; they do not certify complete execution of the entire
notebook catalogue. Public
API pages render every canonical module directly from its source docstrings.
Original documentation figures use CC BY 4.0 with attribution to IPES Research
Group; third-party assets retain their own terms. Mathematical rendering and
figure layout are inspected in native Chrome as well as through `docs-check`.
Native Windows and macOS qualification of this changed revision remains a CI
acceptance gate.
Private `_legacy` implementations support comparisons, not the primary API.

## Scientific scope and acceptance criteria

PyMHM separates local finite element problems from explicit skeletal coupling.
A supported family requires a concrete operator, compatible spaces and numerical
evidence. A backend dependency or generic matrix adapter alone does not supply
a finite element discretization.

The [initial convergence catalogue](docs/cases/minimal-convergence.md) records
short studies, unresolved increments and incomplete levels. Extended historical
campaigns are separate evidence. The [case evidence guide](docs/cases/index.md)
and [literature catalogue](docs/literature.md) distinguish analytical verification,
independent references, matched data and published comparisons.

### Implemented constructions

Applicable degrees, boundaries, material assumptions and executed evidence are
specified on the linked pages.

| Construction | Scope and limitations |
| --- | --- |
| Local/global hybrid algebra | Independent trial/test couplings and adjoint kernels, retained physical modes, constrained local solves, source lifts, offline/online reuse and [recursive MHM](docs/cases/nested.md). Supplied operators require kernel, compatibility and stability checks. |
| Primal Darcy | Triangular Pk, rectangular Qk and tetrahedral Pk pressure, physical pressure means and scalar/SPD permeability. [2D](docs/api/darcy.md), [3D](docs/api/darcy3d.md) and [Cartesian](docs/quadrilateral.md) conventions are explicit. |
| Mixed Darcy | Triangular RT/BDM, enriched rectangular RT, affine tetrahedral/prismatic families and mapped hexahedral RT. [Normal traces, pressure moments and interior enrichment](docs/api/darcy3d.md) are independent choices. Nonaffine prisms and pyramidal mixed elements are outside this path. |
| Material interfaces | Exact integration over Cartesian/planar intersections, material-fitted local meshes and explicit macroface partitions. [Unfitted integration](docs/cases/unfitted.md) alone does not resolve a gradient jump in an uncut polynomial cell. |
| Reconstruction and estimation | RT moment recovery, Oswald potentials and distinct published, energy-weighted and face-jump indicators. [Estimator estimates](docs/cases/reconstruction3d.md) require `k >= ell + d`; algebraic reconstruction has weaker conditions, including `ell <= m <= k`. Continuous-test equilibrium differs from fine-cell DG balance. |
| Alternative multiscale formulations | [MsHHO](docs/cases/mshho.md), [Robin MH](docs/cases/mh.md), [MH²M](docs/cases/mh2m.md) and [PGMHM](docs/cases/pgmhm.md), each with stated face/cell/source conventions. Equivalence and injectivity depend on those spaces and assumptions. |
| RAD and transport | Conservative Pk Galerkin/SUPG, tensor diffusion, reaction and explicit coefficient derivatives. [MHM-USFEM](docs/cases/unusual.md) uses the full unusual residual form; stabilization does not imply a maximum principle. |
| Transient scalar problems | Backward Euler, positive capacity, changing loads/boundaries and prepared spatial operators. [Darcy coupling](docs/cases/transient-transport.md) distinguishes volume flux from numerical normal trace. Manufactured convergence does not reproduce an unavailable random realization. |
| Stokes–Brinkman/Oseen | Taylor–Hood and full-residual equal-order USFEM in 2D/3D, tensor resistance, prescribed convection, component slip and pressure gauges. [Flow adaptation](docs/cases/stokes-adaptive.md) assumes constant viscosity, full Dirichlet data, uniform trace degree and resolved jumps. Nonlinear Navier–Stokes iteration is outside this scope. |
| Primal and displacement–pressure elasticity | General material tensors, physical traction, three/six rigid modes, GaLS/Taylor–Hood and finite/infinite bulk limits. [Primal displacement](docs/cases/primal-elasticity.md) alone is not uniformly locking-free. |
| Mixed elasticity | Row-wise BDM/enriched/rectangular RT stress, weak rotation, anisotropic compliance and [tetrahedral AFW](docs/cases/mixed-elasticity3d.md). Displacement must represent rigid modes; weak symmetry is a moment condition. Classical AFW stability does not cover arbitrary MHM traces. |
| Polygonal/polyhedral geometry | Straight-sided simple polygons, including nonconvex cells, and [certified star-shaped polyhedra](docs/cases/star-polyhedra.md) with original polygonal face spaces. Empty-kernel cells, cavity shells and curved faces are outside the polyhedral path. |
| Helmholtz | Complex triangular/polygonal Pk and Cartesian Qk locals, polynomial/oscillatory traces, absorbing boundaries, diagonal PML and local resonance checks. [Wave](docs/cases/helmholtz.md) and [Marmousi](docs/cases/marmousi.md) comparisons have distinct data contracts. |
| Maxwell and elastodynamics | Tangentially coupled central-DG dynamics with mass-scaled CFL, and Newmark local responses with slabwise traction/substeps. [Maxwell](docs/cases/maxwell.md) and [elastodynamics](docs/cases/elastodynamics.md) distinguish analytical and heterogeneous comparisons. |

See the [API](docs/api.md), [FEniCS local forms](docs/fenics.md),
[meshing](docs/meshing.md) and [linear solvers](docs/solvers.md) for construction
and backend details.

## Priorities and dependencies

| Milestone | Priority | Deliverable | Dependency |
| --- | --- | --- | --- |
| R1 — Brinkman | P0 | 2D/3D regime and space matrix with demonstrated accuracy limits | Existing variational operators and independent references |
| R2 — General performance | P1 | Scaling and cost at matched accuracy beyond current benchmarks | Physical controls and profiling of the complete execution |
| R3 — Scientific acceptance | P1 | Progressive completion of literature targets and resolution controls | Identified inputs and refined references for each case |
| R4 — API and architecture | P1 | Less manual infrastructure and duplication, preserving expressiveness | Numerical equivalence before/after each change |
| R5 — Windows | P1 | Identified native execution of the core, PARDISO and installed artifacts | Windows runner and the target dependency profile |
| R6 — External providers | P2 | Interchangeable providers using the local contracts | R4 contracts and physical controls |
| R7 — Distribution | P2 | Release with demonstrated scientific and platform scope | Required gates and evidence for advertised capabilities |

The contextual API is implemented; R4 now lists its remaining extensions. R1 remains
the highest-priority scientific qualification. Work through R3 in bounded
family-specific deliveries; R5 may proceed independently when it does not change
the source of an active acquisition. Performance acquisitions require exclusive
hardware; concurrent tests or campaigns must not contaminate timing.

## R1 — Qualify Stokes–Brinkman in 2D/3D

Analytical cases, polynomial-family layer studies, refined Taylor–Hood references
and native comparisons already exist. Extend their robustness and resolution
evidence using [layers](docs/cases/introduction-layers.md),
[3D flow](docs/cases/flow3d.md), [Stokes comparisons](docs/cases/reproduction.md)
and [SPE10](docs/cases/spe10.md).

- [ ] Select a minimal matrix covering Stokes, intermediate Brinkman, small/large
  resistance, small viscosity, admissible tensor resistance and high contrast
  in both dimensions. Define per-field criteria, reference type and estimated cost.
- [ ] Qualify analytical 2D layers first, then 3D and extreme regimes after
  operator and reference controls pass. Treat MHM–Taylor–Hood and MHM-USFEM
  as distinct discretizations.
- [ ] Check local degrees/meshes, kernels, traces, pressure gauge, stabilization
  and source. Declare gradient versus symmetric-strain operators and the theorem's
  resistance/inverse-inequality conventions: L13 and L16 use different parameters.
- [ ] Distinguish pseudo-traction, the multiplier and physical stress. Verify
  zero resistance, homogeneous/nonhomogeneous boundaries and compatibility.
- [ ] Refine macro mesh, local mesh and trace space independently, using at least
  three levels where feasible. Resolve plateaus and underresolved layers before
  assigning an observed order.
- [ ] Measure velocity, pressure and gradient/stress errors separately, plus
  divergence, macro balance and physical residuals per field. Control reference
  refinement and assembly/error quadrature independently.
- [ ] Inspect one-sided layer profiles at macrofaces and quantify oscillations.
  Publish resolution/error tables, fields, actual macro meshes and measured rates
  in the case pages and executed introductory notebook.
- [ ] Record the next unqualified configuration after accepting each delivery.

**Acceptance:** each configuration has admissible spaces, operator/kernel checks
and a physical gauge. Field norms and refinement increments demonstrate resolution.
Rate claims meet the applicable regularity and asymptotic hypotheses. Unresolved
or pre-asymptotic regimes retain explicit limits; analytical agreement does not
establish reproduction of a historical figure.

## R2 — Generalize performance and prepare HPC execution

The recorded CPU/multi-GPU strong/weak campaign is complete within its stated
scope. Classical CG/GAMG is faster on the larger grids in that campaign.
Extend [execution strategies](docs/execution.md) and
[performance evidence](docs/performance.md) without assuming universal MHM speedups.

Cached Numba kernels supply scalar diffusion integration, boundary moments,
ordered sparse contribution reduction and planar field ownership/coordinates.
The formulation, providers and native linear solvers retain their Python API.
Ordinary quadrature uses compensated binary64; exceptional exponent ranges
retain native wider accumulation when available and the portable accumulation
contract otherwise. Explicit extended solver/refinement data remain supported.
The bounded [kernel performance notebook](notebooks/darcy/numba_kernel_performance.ipynb)
and performance report separate first-use compilation, warm kernel costs and
complete 2D/3D execution, with identical-discretization field and equation checks.

- [ ] Compare cost at common pressure and physical-flux error targets, refining
  macro meshes and traces. Keep equal-element-budget comparisons separate:
  they use different global approximation spaces and may have different accuracy.
- [ ] Include nonperiodic materials, SPE10 and uneven local workloads. Share
  compatible kernels/infrastructure while assembling each required material operator.
- [ ] Profile assembly, factors/hierarchies, all right-hand sides, communication,
  serialization, reduction, global solve, reconstruction and peak memory.
  Identify local/global bottlenecks and crossover size at each accuracy target.
- [ ] Reduce physical-error integration costs in the shared field and reference
  tabulation owners beyond the compiled planar sampling path: reuse affine
  Jacobians and compute only requested derivatives.
  Qualify batched and spawn-parallel evaluation of archived fields with importable
  exact solutions, preserved basis digests and ordered per-field quadrature sums.
  Compare scalar and vector norms against the existing independent controls;
  measure complete notebook time separately from assembly and solver speedups.
- [ ] Compare SciPy/PARDISO/MUMPS LU and CPU/GPU AMG with admissible operators
  and declared CPU/GPU/thread budgets. Define iterative tolerances before trials
  using field and discretization error; retain existing acquisition criteria.
- [ ] Qualify heterogeneous scheduling, resident workspaces and CPU/GPU transfers
  against the original operator. Reuse factors only for proven equivalent operators.
- [ ] Distribute the global skeleton without gathering all local matrices at
  one rank. Distinguish AMG for positive blocks from saddle-system preconditioning
  and treatment of physical kernels.
- [ ] Extend to multiple nodes and more than two GPUs when resources permit,
  preserving worker/rank ownership, spawn semantics, thread limits, resource cleanup
  and correct shared-face reduction.

**Acceptance:** measured time, speedup, strong/weak efficiency, memory and
cost-versus-error curves include repetitions and reproducible provenance. Separate
JIT warmup and state what it warms. Complete time includes setup, transfers,
synchronization and teardown; phase timings come from the same acquisition.
Report absent gains and regressions. Gomes et al. and Penna et al. motivate these
studies, but their historical 24–768-core and largest-mesh cluster results remain
unreproduced on the current hardware.

## R3 — Complete scientific acceptance by family

Use the following matrix as the single backlog for literature acceptance.
Check each case's current evidence before scheduling work; existing complete
systems and references are not missing implementations. Prioritize unresolved
controls capable of changing the physical conclusion. Historical inputs that
cannot be identified remain external dependencies.

### Acceptance by literature target

The [literature catalogue](docs/literature.md) identifies publications,
reference projects/modules, revisions and source URLs. A summary does not verify
a new execution: reproduction requires its input fields and reference programs.
Keep external reference solver sources and private comparison tools outside the
versioned repository and release artifacts.

| Target and formulation | Case | Recorded evidence | Remaining acceptance condition |
| --- | --- | --- | --- |
| L01: primal Darcy MHM | Cosine, quarter five-spot, square obstacle and rough coefficients | Analytical fields, published curves, complete square-obstacle Basix/P1 and native NeoPZ/RT0 comparisons, and six full point-well Basix/P2 and RT0 comparisons on stated spaces | Resolve classical obstacle-reference refinement and singular-well controls; identify historical random inputs before literal reproduction. |
| L02: elliptic error estimation | Cosine, inclusion and Dirac wells | [Reconstruction and estimators](docs/cases/reconstruction3d.md), analytical errors and indicator controls | Complete matched estimator/reference refinement while retaining Dirac regularity limits. Distinguish algebraic reconstruction from estimate hypotheses. |
| L03: abstract hybrid algebra | Local/global and recursive systems | Independent full Petrov–Galerkin algebra, retained kernels and source lifts | User operators require kernel, compatibility and stability checks. Qualify more general recursion without bypassing child boundary, injectivity or execution restrictions; no published numerical table is supplied. |
| L04: periodic Darcy robustness | [Periodic permeability](docs/cases/periodic.md), fixed macrogrid and face enrichment | Phased basis replay, complete 64-macro independent Q1/P0 assembly and five-level Q1 controls; Q5 accuracy is outside current acceptance | Resolve local/trace and classical-reference increments with injective finite space pairs. Preserve published material and macro spaces. The historical reference is Q1 on 4096² elements; its historical local refinement is unidentified. |
| L05: mixed local Darcy | [Rectangular, tetrahedral and prismatic wells](docs/cases/mixed-well-geometries.md), [mapped oscillatory well](docs/cases/mapped-well-oscillatory.md) | Analytical fields and native NeoPZ space comparisons | Complete matched whole-case references and interior/face/quadrature refinements. Preserve common geometry, Piola conventions, pressure spaces and `div(V)=Q`; control singular forcing. |
| L06: MHM–MsHHO connection | [Elliptic macro and skeletal refinement](docs/cases/mshho.md) | Analytical comparisons with declared source spaces and complete independent assembly; field differences meet their declared criterion | Complete convergence under the equivalence hypotheses. At contrast 10⁶, cross-insertion into the other rounded operator gives about `5.19e-9`, above its `1e-10` criterion; joint certification remains unresolved. |
| L07: face-based robustness | Periodic medium and SPE10 layer 36 with continuous face interpolation | [Darcy flux](docs/cases/spe10-flux.md), Q1/C0-P1 MHM and Q3/MSL/NeoPZ references | Resolve local/face/material and classical-reference increments, preserving layer, units, orientation and the published 66-square macro partition. Q3/RT0 energy control is `1.80e-6` against `1e-7`; this is separate from solver residual acceptance. |
| L08: unfitted preprint | [Two-layer interface](docs/cases/unfitted.md) | Analytical series and independent UFL systems | Identify preprint-specific inputs and hypotheses separately from final L10; their regularity endpoints differ. |
| L09: H(div) recovery and adaptivity | [Adaptive SPE10](docs/cases/spe10-adaptive.md) | Published P2/r2/P0 spaces, RT2/Oswald recovery, full independent initial/final UFL systems and refined RT2 references | Reduce reference sensitivity and qualify historical BAMG connectivity/marking. Centralize `k >= ell + d` for estimates and `ell <= m <= k` for reconstruction; test admissible boundaries and immediately excluded cases in each dimension. |
| L10: unfitted flux approximation | [Smooth h/p sweeps and two-layer contrast](docs/cases/unfitted.md) | Analytical norms, native assemblies, P8 r32→r64 endpoints for ell0/ell1 and twelve complete independent UFL S0/S2 contrast systems | Resolve local error for higher admissible traces. P8/r16–P3/s32 has an exact multiplier kernel that solver or quadrature changes cannot remove. Historical Figure-7 S2 inputs remain unspecified despite same-discretization agreement for declared cases. |
| L11: advective/reactive MHM | [Mixed walls and random Darcy–transport](docs/cases/transient-transport.md) | Analytical and coefficient controls | Acquire the full §5.4 realization with 512 macros through T=7, coupled trajectory and independent space/time/quadrature refinements. Distinguish volume Darcy flux from numerical normal transfer. The literal mixed-wall curve remains quantitatively different; its epsilon=1 control is separate. |
| L12: generalized RAD | [Polygonal/polyhedral diffusion and reaction layers](docs/cases/polygons.md) | Conditioning, boundary and layer controls | Complete matched conditioning/layer/star-polyhedron studies and independent fields, preserving full residuals, coefficient derivatives and boundary conventions. |
| L13: equal-order Stokes–Brinkman | Smooth flow, layers and SPE10 layer one | [SPE10](docs/cases/spe10.md), published-space fields, analytical polynomial-family layer convergence and a five-level independent Taylor–Hood baseline with physical integration | Close local/trace/reference sensitivity and R1's 2D/3D extreme-regime matrix. Qualify historical mesh, stabilization and Stokes stress conventions before literal figure reproduction. |
| L14: multilevel flow estimator | [Stokes adaptation and cavity](docs/cases/stokes-adaptive.md) | Analytical indicators and classical cavity comparisons | Complete adaptation/reference refinement with the same constant lid, pressure gauge and corner cutout. A regularized lid is a different physical case. |
| L15: adaptive Oseen | [Smooth and boundary-layer flow](docs/cases/oseen.md) | Analytical refinement and independent operators | Complete layer/adaptive whole-case controls, quadrature and physical norms. Prescribed Oseen convection does not qualify nonlinear Navier–Stokes. |
| L16: flow a priori analysis | Admissible Stokes–Brinkman spaces | Analytical/native verification and stated degree conditions | Verify dimension-dependent discretization, regularity and inverse inequalities for each rate claim. Preserve L13's minimum-resistance versus L16's cellwise maximum-resistance convention and squared/unsquared inverse constants. |
| L17: primal elasticity | Analytical displacement and [HPC4E](docs/cases/hpc4e.md) | Analytical fields, independent DOLFINx/UFL RT1/RT2 stress–displacement–rotation references, equilibrium/work checks and RT2 spatial refinement | Reduce or quantify rotation/compliance reference sensitivity; preserve published skeletal spaces and full-stress norms. |
| L18: weakly symmetric mixed elasticity | [Tensor families and oscillatory Table 3](docs/cases/mixed-families.md), [3D mixed spaces](docs/cases/mixed-elasticity3d.md) | Native fields and analytical norms | Complete BDM/RT/enrichment, geometry and tensor controls with physical rigid modes and weak rotation. The printed rotation-column inconsistency and unidentified historical connectivity/exterior traces limit literal table reproduction. |
| L19: locking-free elasticity | [Finite/infinite bulk limits](docs/cases/elasticity.md) | GaLS, mixed-field and native comparisons | Complete matched Lamé sweeps and refinement for GaLS, Taylor–Hood and mixed methods. Qualify historical amplitude/stabilization choices; an affine primal patch does not prove uniform locking freedom. |
| L20: scalable implementation | [Distributed local/global execution](docs/execution.md) | Spawn/MPI contracts and workload-specific CPU/GPU strong/weak measurements | Complete R2's material diversity and matched-accuracy comparisons; historical cluster scaling remains unverified. |
| Additional MH | [Robin hybrid diffusion](docs/cases/mh.md) | Analytical boundary/parameter sweeps and independent UFL | Complete whole-case refinement; distinguish Robin multipliers from physical flux. |
| Additional MH²M | [Oscillatory medium](docs/cases/mh2m-heterogeneous.md) | Crisscross/diagonal cases and refined independent CG3 fields | Complete overlay quadrature and baseline refinement under face/source hypotheses. Historical curve differences remain; do not fit material, source or method parameters to them. |
| Additional PGMHM | [Inclusions](docs/cases/pgmhm.md) and [SPE10](docs/cases/pgmhm-spe10.md) | Analytical/native equations and independently refined classical fields | Complete matched whole-case comparisons with published enrichment and heterogeneous stabilization, resolving material interfaces and reactive lengths. |
| Additional Unusual (MHM-USFEM) | [Reaction–diffusion](docs/cases/unusual.md) and [SPE10 layers](docs/cases/unusual-spe10.md) | Analytical/native fields and recorded FreeFem comparisons | Acquire the complete external comparison and qualify material, reaction-length and trace resolution. Distinguish published spaces from enriched controls. |
| Additional Helmholtz | [Angular/stability studies](docs/cases/helmholtz.md) and [Marmousi](docs/cases/marmousi.md) | Plane/Hankel fields, native saddle systems and classical material-crop P1–P4 references | Complete angular/stability sequences and the 15-case Marmousi family, including phase-wise memory/storage, admissible resonances and reference refinement on the same physical crop. Retain historical input limits; 2D point-source derivative norms use a fixed physical cutout, without claiming finite global H1 norm. |
| Additional Maxwell | [Nanoguide](docs/cases/maxwell-nanoguide.md) | Analytical dynamics, native operators, independent central-DG Q2 through 1024², matched staggered-time fields, CPU/GPU equivalence and space/time/material controls | Quantify reference and MHM temporal/spatial sensitivity over the trajectory, preserving staggering, CFL and balance. Identify historical incident phase, amplitude and turn-on before literal image reproduction. |
| Additional elastodynamics | [Equation (53) and three layers](docs/cases/elastodynamics.md) | Analytical Newmark trajectories, complete independent 341-macro original equations and 301-state common-basis coordinate comparisons | Evaluate both executed field bases separately; common coordinates do not establish field agreement. Verify a conforming reference on several finer meshes and complete space/time/quadrature controls. Historical heterogeneous 2017 inputs remain unresolved. |

**Acceptance:** close each target's specific condition, or identify the unresolved
input/discrepancy and provide an independently assembled complete comparison of
the same physical case. A manufactured solution, small patch or coverage result
does not replace matched published-case acceptance. Missing private reference
programs remain acquisition dependencies, not evidence of a new execution.

## R4 — Simplify the API without specializing it by physics

The contextual workflow and automatic/custom binding paths are described in
[the current API](docs/tutorials/overview.md). They use the existing numerical
owners, including independent trial/test couplings, retained modes, physical
moments, local boundary spaces, shared-face accumulation and named field views.
The remaining extensions are:

- [ ] Add automatic native 3D trace pairings and global interface UFL integration
  for explicitly supported face families. Reuse Basix and existing topology,
  orientation and quadrature owners; audit dimension-dependent hypotheses and
  independently verify physical fields before extending capability claims.
- [ ] Extend portable field descriptors to selected moment-based H(div)/H(curl)
  and non-equispaced families. Preserve executed basis matrices, transformations,
  Piola maps and their digests; verify replay against native fields and physical
  moments. Native assembly support alone does not imply portable nodal conversion.
- [ ] Extend exact recursive trace restrictions beyond compatible planar normal
  spaces. Declare parent/child support and any projection explicitly; preserve
  child boundary, gauge, injectivity and execution restrictions. Recursive MPI
  requires a separate qualified implementation.
- [ ] Qualify reusable native contexts and form workspaces for larger general
  user-defined operators. Share compiled kernels across compatible spaces;
  reuse material matrices or factors only after proving operator equivalence.
  Record setup, compilation, transfers, reduction and field reconstruction costs.
  The [bounded route-cost control](benchmarks/results/api-binding-20261006/route-cost.json)
  measures 6.043 s for automatic UFL trace pairings versus 0.979 s for explicit
  Basix boundary blocks on a small identical discretization; repeated native
  form lookups dominate. Optimize general workspace reuse and remeasure,
  preserving independent pairings and original-equation/field agreement.
- [ ] Audit remaining advanced notebooks and legacy comparison consumers for
  opportunities to replace duplicated infrastructure with the contextual owners.
  Preserve their executed coefficients, spaces, boundary data and numerical
  evidence; retain explicit operations when they explain mathematical choices.

Automatic native local/global trace integration currently supports planar
polynomial `SkeletonSpace` bases with fine-facet-aligned partitions. Separate
local and global boundary spaces can use unsigned UFL pairings and the generic
boundary mass owner. Other geometries and custom bases provide their capabilities
or explicit numerical blocks. Tangential basis rotations remain explicit custom
maps; unsupported capabilities fail before numerical assembly.

**Acceptance for each extension:** users can express its local/global forms
without infrastructure indexing, advanced users can supply those same maps, and
both routes agree with independent original-equation and physical-field controls.
Run native integrations, archive/replay controls, executed tutorials and the
engineering gates below. No physical coupling sign, kernel, gauge or stability
property is inferred from a PDE or method name.

## R5 — Verify native Windows execution

- [ ] Execute an identified revision on Windows x86-64: `test-core`, Basix,
  native DOLFINx/UFL, SciPy/PyAMG, PARDISO, shared faces and
  serial/thread/spawn-process paths. Publish the actual native FEM reports.
- [ ] Build and install a wheel outside the checkout; verify import ownership,
  MKL runtime, factor cleanup and fields against independent assembly.
- [ ] Execute selected portable notebooks and publish the measured platform/backend
  matrix. Notebook sections requiring PETSc/MUMPS, including distributed
  conforming references, require the Unix stack. Keep WSL2 and native Windows
  qualification distinct.

**Acceptance:** a native Windows receipt identifies the revision, locked
dependencies and supported options. The core has no mandatory optional-runtime
imports; unsupported choices fail explicitly. Extended precision and worker limits
follow the platform. CI configuration or lock resolution alone does not prove
native execution. See [Windows support](docs/windows.md).

## R6 — Qualify external local providers

- [ ] Demonstrate an independent provider delivering the maps, moments,
  operators and source/trace responses required by the global formulation.
- [ ] Qualify a learned provider against independent FEM on held-out data,
  measuring field error and physical residuals. Define rejection/fallback outside
  the contract without tying the interface to a particular ML architecture.
- [ ] Evaluate optional preCICE integration for a justified coupling use case,
  including transfers, synchronization, licensing and platform availability.
  Keep it outside core dependencies; native Windows support requires evidence.
- [ ] Evaluate scikit-fem as an optional local assembly backend, outside the
  core dependencies and alongside Basix and DOLFINx. Begin with primal Darcy
  P1/P2 in 2D, matching geometry, forms, spaces, boundary data, physical moments
  and oriented traces; compare fields and conservation for homogeneous and
  nonhomogeneous data. Extend to 3D and mixed problems only after checking
  dimension-dependent element degrees, derivative support, trace compatibility,
  kernels, gauges and stability hypotheses. Qualify installation and native
  execution separately on Linux, macOS and Windows; measure performance rather
  than assuming a speedup.

**Acceptance:** providers are interchangeable without changing global equations,
worker data support spawn transfer and native resources have explicit owners.
Real integration demonstrates accuracy, stability and cost; mocks and API
contracts do not qualify external software. See [providers](docs/guides/providers.md).

## R7 — Release the demonstrated scope

- [ ] Select capabilities and cases for the release using evidence valid for
  the delivered source; synchronize version, dependencies, metadata and release notes.
- [ ] Keep notebooks, catalogues, literature labels and publication assets current.
  Version only selected figures under an explicit allowlist. Large field archives
  and intermediate outputs remain outside Git.
- [ ] Publish the next validated `v*` release with the contextual API through the
  existing workflow sequence below. Submit the conda-forge recipe separately
  when target dependencies are available.

**Acceptance:** installed artifacts contain all runtime and typing files, license
and required metadata. The sdist includes only `src/pymhm`, `pyproject.toml`,
`README.md`, `LICENSE`, backend-required `.gitignore` and generated metadata.
Docs, scripts, examples, tests, benchmarks, notebooks, recipes, roadmap and Pixi
environments remain repository resources, outside Python/Conda installation
artifacts. A checked build or configured workflow does not establish publication
or conda-forge acceptance.

## Delivery acceptance protocol

1. **Define the experiment:** formulation, geometry, material, source, boundaries,
   gauge, spaces/degrees, refinements, quadrature, tolerances and accuracy targets.
   Independently derive manufactured data from the actual operator. Recheck
   dimension-dependent regularity, degree, stability and injectivity hypotheses.
2. **Verify the operator:** original physical equations, left/right kernels,
   orientation, moments and condensation against independent full assembly.
   A small reduced/scaled residual does not establish uniqueness, inf-sup stability
   or field accuracy.
3. **Validate physical fields:** errors and residuals per field/block, macro and
   fine-cell conservation as appropriate, energy/work and separate macro/local/trace
   refinements. Distinguish raw gradients, H(div) fluxes and multipliers. Without
   an exact solution, refine a classical conforming reference on several meshes
   with the same operator, material and boundaries; report its own increments
   and quadrature controls.
4. **Preserve provenance and replay:** archive source revision, lockfile, versions,
   input/mesh/material hashes, actual executed basis matrices with digests, dtype,
   solver, resources and timing scope. Fix nullspace orientation by declared
   moments and use archived bases consistently in evaluation and orientation maps.
   Check replay across BLAS thread counts and equivalent kernel rotations.
   Source/input changes require identified reacquisition; replotting is not a PDE run.
5. **Apply shared corrections:** fix the operation's owner, test the invariant,
   audit consumers and reacquire affected cases, including homogeneous/nonhomogeneous
   boundaries where relevant. Avoid case-specific formulas, coverage exclusions
   or relaxed tolerances that conceal numerical failures.
6. **Publish current evidence:** updated case page, lightweight records, executed
   notebooks and inspected figures. Show actual macro meshes on analytical,
   numerical and error panels, and intersections with independent one-sided
   values on profiles. Label velocity and Darcy flux correctly; check layout,
   colorbars and readability at publication size. Identify reference project,
   module, revision and URL; distinguish inspection, execution and comparison drivers.

Matched literature reproduction requires matching inputs, discretization and
norms, with extraction uncertainty stated. Rate claims require the source theorem's
hypotheses; singular or heterogeneous cases do not inherit smooth-problem rates.
Software coverage is an engineering gate rather than scientific certification.

## Engineering and release gates

Use **Pixi 0.76.2** and the checked-in workspace and AmgX integration lockfiles.
Check `pixi --version` and `pixi list --locked --no-install -e test-core`
before environment preparation. Update manifests and locks together when
dependencies change. Initial full Linux/two-GPU setup:

```bash
pixi install --locked -e test
pixi run --locked -e test test-setup-amgx
pixi run --locked -e test test-dependencies
```

Required engineering gates:

```bash
pixi run --locked -e test lint
pixi run --locked -e test format-check
pixi run --locked -e test typecheck
pixi run --locked -e test test-cov
pixi run --locked -e docs docs-check
pixi run --locked -e packaging lock-check
pixi run --locked -e packaging ci-check
pixi run --locked -e packaging metadata-check
pixi run --locked -e test build
pixi run --locked -e test check-dist
pixi run --locked -e packaging conda-build
```

The test runner uses all available CPUs, one numerical thread per worker and an
exclusive `@pytest.mark.serial` phase. Reserve that marker for tests requiring
exclusive execution, not merely expensive tests. Maintain independent **99% line
and branch coverage gates**. Test package behavior, meaningful numerical
invariants and relevant script algorithms with native integrations alongside
optional contracts. Reduce redundant combinations without losing distinct
numerical cases; missing dependencies do not count as backend validation.

### CI and release ordering

[Tests](https://github.com/ipes-lncc/pymhm/actions/workflows/tests.yml),
[Lint and Quality](https://github.com/ipes-lncc/pymhm/actions/workflows/lint-and-quality.yml)
and [Docs](https://github.com/ipes-lncc/pymhm/actions/workflows/docs.yml) have
dedicated responsibilities and run independently on pull requests and main pushes.
Within Tests, the Integration matrix starts only after **all Core matrix jobs
succeed**. Its native jobs cover DOLFINx assembly with SciPy on Linux, Windows
and macOS, FEM/PARDISO on Linux and Windows, and PETSc/MUMPS and distributed MPI
on Linux. PETSc-blocked processes verify the assembly dependency boundary.
Optional native mesh-generation, remeshing
and visualization checks run in the manually dispatched full Linux/two-GPU
suite after Core succeeds. Reusable checks use both locked workspaces.

Core and Linux FEM runs collect branch coverage. After Core and Integration
succeed, the Coverage job combines the Linux core and native FEM measurements
from the same revision and enforces independent 99% line and branch gates.
Codecov receives the combined XML report after those gates pass.

Jobs that install environments explicitly enable `setup-pixi` caching, keyed
by platform, requested environments, Pixi binary, lockfile and paths. The
workspace-only validation job installs no environments and disables caching.

The active matrix targets Linux x86-64, Windows x86-64 and Apple Silicon
macOS ARM64 with `macos-latest`. macOS x86-64 resolution remains available
for local use, without an active CI target. Use `test-core`, `test-py311`
and `test-py312` on their configured platforms; these do not replace native
backend qualification. Full two-GPU Tests requires manual dispatch with
`full_native` enabled.

The [release workflow](https://github.com/ipes-lncc/pymhm/actions/workflows/publish-pypi.yml)
runs **Version/tag validation → Tests, Quality and Docs checks in parallel →
PyPI publication of checked artifacts → GitHub Release → Docs deployment**.
Use `release-fetch`, `changelog-preview`, `release-prepare VERSION` and
`version-check` in the locked `release` environment; initial preparation requires
`--initial`. The [development guide](docs/development/contributing.md#prepare-versions-and-release-notes)
describes the first-parent main history and version synchronization rules.
Release administration is described in the [development guide](docs/development/contributing.md).
Docs also supports manual publication with `publish=true` from `main` or a `v*`
tag, including an initial documentation deployment before the next release.
Keep platform/release claims tied to successful identified runs.

### Scientific documentation and notebooks

The documentation uses method-based tutorials, a flat visual Gallery of
applications, task-specific configuration Guides, source-generated API coverage
and an architecture diagram. Application notebooks are separate from method
lessons; duplicated scalar lesson copies and problem/dimension Gallery indexes
are consolidated. Theoretical pages identify local and global variational
problems, their unknowns and approximation conditions. Keep these pages and
their original-publication citations synchronized when adding a method or
extending its admissible spaces.
The two-dimensional TM Maxwell tutorial demonstrates combined L2 order two
and broken H(curl) order one over its final spatial levels, with a separately
refined conforming scalar-wave reference for that exact physical reduction.
Its time-step sensitivity control applies to the stated resolution and retains
the larger sensitivity of the individual electric error. This evidence does not
qualify general three-dimensional propagation: the 3D stationary patch and
earlier preasymptotic measurements remain separate. An independently refined
3D conforming H(curl) reference and spatial qualification are still required
when extending that scope. The constrained semidiscrete ODE qualifies temporal
integration, rather than a separate physical spatial discretization.

The smooth two-dimensional elastodynamic tutorial retains all nine spatial
levels and separate endpoint and trajectory-maximum norms. Displacement and
physical Cauchy-stress errors approach the published numerical targets of
orders three and two; velocity follows an order-three envelope with visible
interval variation. Named time-step controls at resolutions 96 and 128 quantify
their own sensitivity, without claiming a control at resolution 192. These
targets are numerical observations from the cited work, not a proved dynamic
error estimate or reproduction of its historical three-dimensional data.

Use `introduction` for UFL introductory notebooks and the case's documented
profile for other acquisitions. Run affected notebooks with `notebooks-run`
and update their catalogues. Unaffected heavy campaigns are separate from routine
checks.

Run `docs-check` and inspect MathJax in the browser: a successful Markdown/site
build does not compile TeX. Use standalone `$$` display blocks with blank
lines, aligned long equations and protected TeX table delimiters. Inspect final
figures at their intended viewing size. Detailed maintained procedures are in
[development](docs/development/contributing.md) and [verification](docs/verification.md).
