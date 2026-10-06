# Changelog

## 0.1.0

<!-- pymhm:generated:start -->
<!-- pymhm:generated:end -->

- Make user-declared `Equation` and `LocalEquations` the primary API: independent
  local A/B and global C/D blocks, rectangular trial/test trace maps, explicit
  moments, additional global forms and recursive multiscale local operators.
- Compile scalar, vector and mixed UFL forms through a generic native adapter;
  preserve declared global coefficient ordering and support external compilers.
- Recover physical finest-scale moments through recursive source offsets and
  boundary reactions; retain numerical bases during global load updates.
- Import predefined physical solvers from their implementation owners; the
  primary namespace exposes formulation and numerical tools.
- Present instructional examples as problem-oriented notebooks, with an indexed
  catalogue of methods and introductory scalar, vector, provider and UFL cases.
- Discover nested notebooks and select them by group, path or historical ID;
  preserve problem folders in executed outputs and verify source notebooks in
  distribution artifacts.
- Organize implementation into responsibility packages, separate geometry,
  reference spaces and predefined formulations, and share mixed Darcy assembly
  and coefficient evaluation.
- Validate private documentation and annotations recursively, and compare every
  packaged source and typing file with its current implementation bytes.
- Expose free functions for local condensation, reconstruction and global
  assembly, with explicit object delegations and unchanged numerical results.
- Add local/global form descriptions, callable local providers, verified external
  local solvers and ordered serial/thread/spawn execution in bounded batches.
- Use Basix as a runtime dependency for nodal, RT/BDM and polynomial trace
  tabulation, preserving declared node/moment coordinates and archived bases.
- Add introductory scalar, vector and provider tutorials.
- Add compact hybrid assembly and staged periodic acquisition with atomic
  archives, bounded local-response lifetime and executed-basis replay checks.
- Restore oscillatory Helmholtz face spaces from executed segment matrices,
  preserving their orientation and constant moments.
- Expose independent assembly and error quadrature controls for unfitted studies;
  verify admissible and excluded finite local/trace pairs.
- Scope notebook archive inventories to the selected notebooks before reading
  their scientific manifests.
- Add arbitrary-degree tetrahedral scalar elements and certified star-shaped
  nonconvex polyhedra with original polygonal faces.
- Add three-dimensional BDM mixed-stress elasticity with weak symmetry,
  physical rigid modes and finite/infinite bulk-modulus verification.
- Add Newmark elastodynamic local responses, slabwise traction coupling,
  local substeps and independent two/three-dimensional inertia operators.
- Add Cartesian Maxwell nanowaveguide spaces, circular material integration
  and independent space/time/quadrature reference controls.
- Add all 256 Helmholtz propagation directions per published configuration,
  physical point-source loads and exact-flux interpolation diagnostics.
- Add MH²M, Robin-local MH, residual Petrov–Galerkin MHM and unusual
  reaction–diffusion formulations, with analytical refinement and native UFL checks.
- Add tetrahedral Taylor–Hood/USFEM flow and GaLS elasticity, three-dimensional
  MsHHO, general H(div) normal/interior degrees and anisotropic polygonal stress.
- Add arbitrary planar material cuts, incident-side reconstruction and volume
  mesh exchange through meshio, Gmsh and Netgen.
- Implement tetrahedral RT moments, Oswald recovery and conforming adaptation;
  validate the dimension-dependent local degree condition across all Darcy estimators.
- Add Helmholtz polynomial/oscillatory traces, absorbing boundaries and PML;
  add Maxwell DG local dynamics, skeletal constraints and explicit energy/CFL checks.
- Evaluate broken polynomial fields and one-sided profiles using persisted
  executed bases; distinguish physical approximation errors from algebraic agreement.

- Add Cartesian Qk Darcy locals, continuous face traces and exact integration
  across Cartesian permeability interfaces; compare the stated SPE10 Model 2
  layer-36 configuration with a 9,738,625-unknown conforming Q3 reference.
- Add checksum-verified SPE10 loading with explicit units, one-based slices,
  PyVista/VTK visualization and unaveraged broken-field rendering.
- Add angularly allocated discrete point wells and quarter-five-spot series
  verification; compare a central square low-permeability obstacle covering 25%
  of the domain with MSL and NeoPZ, including interfaces cutting macroelements.
- Preserve supplied local meshes in conservative transport, heat evolution and
  Darcy–transport coupling; verify the resulting physical operators independently.
- Support face-adaptive transport with mixed essential and diffusive-flux
  boundary conditions, retaining the published natural-boundary indicator.
- Support componentwise pseudotraction data for Brinkman slip walls and preserve
  the corresponding physical pressure and velocity gauges.
- Add a classical conforming P2/P1 Taylor–Hood reference for SPE10 Brinkman,
  independently assembled with DOLFINx/UFL and solved by PETSc/MUMPS on
  five pixel-aligned meshes up to 30,435,203 unknowns. Verify constant and layered
  channel patches, and compare physical fields with the fixed MHM discretization.
- Introduce backend-independent local Neumann problems, physical kernel constraints,
  sparse global condensation, global gauges and deterministic local execution.
- Add oriented triangular meshes, independent polynomial face partitions and
  Gmsh/Netgen/meshio adapters with physical-marker preservation.
- Implement primal and RT0/P0 Darcy, Taylor–Hood and USFEM Stokes–Brinkman,
  variable-advection Oseen, plane-strain elasticity and implicit heat evolution.
- Add native Pk primal Darcy, BDM2/P1 mixed Darcy, high-order flow locals,
  variable-coefficient conservative RAD with consistent SUPG, and Pk heat spaces.
- Add continuous polynomial interpolation within independently subdivided
  macrofaces, alongside the discontinuous Legendre trace basis.
- Implement displacement–pressure GaLS and Taylor–Hood elasticity, including
  the incompressible limit, variable material residuals and computed stability
  bounds; add BDM2/P1/P1 weak-symmetry mixed elasticity with physical rigid gauges.
- Add constrained RT0 energy equilibration with fine-cell conservation.
- Add RT0/RT1/RT2 face-and-volume moment recovery with continuous-test
  conservation and separate raw/projected divergence diagnostics.
- Add conforming Oswald potential recovery and the complete four-term Darcy
  estimator for identity diffusion, with mesh/boundary/space checks and measured
  effectivity on five macro meshes.
- Integrate real UFL/DOLFINx assembly, direct CPU/GPU solvers and local elliptic AMG.
- Detect numerical rank deficiency, incompatible gauges and invalid coefficients;
  verify kernels and compatibility without dimensional absolute thresholds.
- Add analytical verification, native backend integrations, executable
  notebooks, MkDocs theory/reference documentation and measured CPU/GPU benchmarks.
- Add a visual case gallery with SVG/PNG figures, analytical reference fields,
  error maps, conservation diagnostics, profiles and expected-result discussion.
- Add worker-local factories that assemble and condense once, retain ordered
  mesh/DOF metadata, and preserve serial, thread and portable spawn semantics.
- Retain nearly null local modes through exact coarse elimination, avoiding
  cancellation as Brinkman drag or scalar reaction vanishes and heat steps grow.
  Preserve reconstructed pressure gauges and nonsymmetric coupling blocks.
- Normalize compatible tangential-advection scalar kernels and unresisted
  Brinkman translation directions; reject means in resisted directions without
  replacing small positive material coefficients by zero.
- Match four published Darcy error curves at five resolutions, retaining the
  discrepant primal interpretation and recording digitization uncertainty.
- Compare primal Darcy fields with MSL_MHM using MSL_CG and MSL_Core; add
  Stokes pressure/local/trace refinement studies, including both pressure traces
  at macro interfaces.
- Compare GaLS displacement, pressure and full stress with native MSL local
  assembly; check BDM2 Darcy and weak-symmetry elasticity against independent
  DOLFINx mixed operators and polynomial projection identities.
- Verify the original coarse cosine flux against MSL primal and restricted
  NeoPZ RT0 fields, using shared weak boundary assembly in the MSL comparisons.
- Compare native RT0/P0 fields with independent NeoPZ assembly and explicit
  macroface restrictions; reject inadmissible nearly aligned RT0 trace partitions.
  Distinguish the RT0 driver from Labmec/MHM's positive-order controller.
- Reproduce higher-order local Stokes velocity and pressure curves through UFL,
  recording unresolved differences in published stress norms explicitly.
- Package with Hatchling and Pixi, portable CI, independent 99% line/branch gates,
  trusted PyPI release automation and a tested Conda recipe.
- Add triangular RT0–RT2, independently enriched rectangular and mapped
  hexahedral RT spaces, tensor elasticity and weak-symmetry BDM/RT families.
- Add affine tetrahedral mixed Darcy families with P1/P2 divergence spaces
  and the 27-mode prismatic family, with native Basix and NeoPZ basis checks.
- Add tetrahedral Pk Darcy, conservative transport and elasticity, polygonal
  macro partitions and original polygonal trace faces on certified star-shaped polyhedra.
- Implement MHM–MsHHO equivalence, recursive local condensation, reusable
  offline/online factors and exact-matrix caching.
- Add conforming macro refinement, local and skeletal refinement, material
  intersections and two-level Darcy, Stokes–Brinkman and Oseen indicators.
- Separate the published and energy-normalized Darcy indicators; add native
  FreeFEM/BAMG residual-metric remeshing and quantitative SPE10 adaptation studies.
- Add rank-owned distributed PETSc/MUMPS assembly, resident GPU local batches,
  CPU/GPU block AMG, symmetric-indefinite PARDISO/MUMPS factorizations and
  optional Ruiz congruence with checks against the original equations.
- Add periodic permeability, mapped wells, geological mixed elasticity and
  adaptive flow campaigns with independently refined classical references,
  signed physical fields and explicit publication comparisons.

The release is pre-alpha. Supported equations, geometry classes, approximation
spaces and estimator hypotheses are specified in the scientific scope. Material
limits and selected published comparisons do not establish robustness for
arbitrary meshes, contrasts or trace choices, or reproduce every historical table.
