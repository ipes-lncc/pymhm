# API reference

The primary interface is `Equation`, `LocalEquations`, `MultiscaleProblem`,
`assemble` and `solve`. User-supplied forms define the operator and its trace
pairings; the package supplies compilation, elimination, ordered assembly and
reconstruction. See the [variational guide](variational.md).

The reference is organized by mathematical formulation and numerical infrastructure.
Each family page contains the complete signatures and docstrings of its documented
objects. Optional dependencies are identified by the relevant adapters; the native
multiscale formulations remain part of the portable package.

The API pages render documentation directly from the canonical Python modules
with `mkdocstrings`; signatures and numerical conventions come from the source.
The reference includes the shared local operators, validation contracts and
physical field containers, alongside the high-level problem interfaces.

Physical family pages document reusable element, trace, material and recovery
operations together with the predefined formulations used by the verified
gallery. User implementations compose the public operations with their own
equations; they require no private solver imports. The root namespace presents
the generic variational and infrastructure API. The
[formulation inventory](variational.md#formulations-composed-from-the-same-api)
links the mathematical ingredients to importable, editable providers.

| Family | Scope |
| --- | --- |
| [Meshes and geometric refinement](api/geometry.md) | Geometric partitions, skeletal topology and conforming refinement. |
| [Materials and reservoir data](api/materials.md) | Material fields, physical loads and integration across interfaces. |
| [Mesh exchange and visualization](api/meshing.md) | Optional mesh generators, mesh-file exchange and visualization adapters. |
| [Hybrid operators and multiscale constructions](api/hybrid.md) | Local condensation, operator reuse, recursive problems and distinct multiscale formulations. |
| [Robin MH and three-field MH²M](api/mh.md) | Local Robin and Neumann maps, independent pressure/conormal traces, and tetrahedral assembly. |
| [Darcy in two dimensions](api/darcy.md) | Primal, mixed, analytical and residual-enriched Darcy discretizations. |
| [Darcy in three dimensions](api/darcy3d.md) | Tetrahedral, prismatic and mapped mixed formulations with their explicit geometry and degree contracts. |
| [Flux reconstruction and error estimation](api/reconstruction.md) | Moment reconstruction, conforming potentials and dimension-dependent estimator hypotheses. |
| [Adaptive scalar and elasticity indicators](api/adaptivity.md) | Darcy refinement policies, local error controls, metric remeshing and the primal elasticity indicator. |
| [Stokes, Brinkman and Oseen](api/flow.md) | Incompressible-flow formulations, residual estimators and adaptive policies. |
| [Elasticity](api/elasticity.md) | Displacement, displacement–pressure and weakly symmetric stress formulations. |
| [Transport, reaction and diffusion](api/transport.md) | Stationary and transient scalar problems, conservative transport and stabilization. |
| [Finite element bases](api/elements.md) | Lagrange, BDM and RT bases, curl operators and tangential traces with their declared local degrees of freedom. |
| [Local finite-element operators](api/local-operators.md) | Volume/trace forms, quadrature orders and reusable scalar and vector element operations. |
| [Numerical contracts](api/numerical-contracts.md) | Admissible spaces, physical constraints and numerical validation. |
| [Physical fields and post-processing](api/postprocessing.md) | Executed field coordinates, component evaluation and physical norms. |
| [Helmholtz and Maxwell](api/waves.md) | Time-harmonic acoustic and transient electromagnetic formulations. |
| [Solvers and execution backends](api/backends.md) | Finite-element adapters, sparse solvers, parallel execution and separable operators. |

## Common entry points

Use `from pymhm import ...` for high-level problem descriptions, meshes and
generic numerical operations. The package resolves those exports on demand;
importing its root does not import Basix or optional native backends. Import
element, material, trace and field operations from the canonical submodules
documented below. The editable providers in `examples/formulations/` compose
these operations into each mathematical formulation.

| Task | API owner |
| --- | --- |
| Bind meshes, interface spaces and user forms | `MeshHierarchy`, `bind_interface`, `bind_problem`, `LocalContext`, `GlobalContext` in [hybrid API](api/hybrid.md) |
| Declare custom interface representations | `InterfaceSpace`, `TraceBinding` in [hybrid API](api/hybrid.md) |
| Declare Cartesian vector traces | `ComponentTraceSpace` in [hybrid API](api/hybrid.md) |
| Integrate, project and prescribe interface values | `trace_quadrature`, `trace_linear_form`, `trace_bilinear_form`, `project_trace`, `trace_boundary_data` in [hybrid API](api/hybrid.md) |
| Evaluate named physical fields | `DiscreteField`, `solution_field`, `evaluate_field` in [hybrid API](api/hybrid.md) |
| Declare executed nodal, modal or Piola field coordinates | `nodal_field`, `modal_field`, `piola_field`, `hdiv_field` in [hybrid API](api/hybrid.md) |
| Declare local and global variational blocks | [`Equation`, `LocalEquations`, `columns`, `rows`](api/hybrid.md#pymhm.core.equations) |
| Assemble, solve and reconstruct a hierarchy | [`MultiscaleProblem`, `NestedEquations`, `assemble`, `solve`](api/hybrid.md#pymhm.core.multiscale) |
| Declare a fully global form without local elimination | `MultiscaleProblem.from_global(Equation(A, L), size, ...)` in [hybrid API](api/hybrid.md) |
| Reuse an assembled hierarchy and recover coefficients | [`with_global_load`, `with_global_equation`, `solve_multiscale_system`, `reconstruct_multiscale`](api/hybrid.md#pymhm.core.multiscale) |
| Change volume and interface sources with reusable factors | [`OfflineMultiscaleSystem`](api/hybrid.md#pymhm.core.online) |
| Represent a physical finest-scale integral | [`leaf_moment`](api/hybrid.md#pymhm.core.multiscale) |
| Reconstruct from prescribed cell/face moments | [`energy_reconstruction`](api/hybrid.md#pymhm.core.moments) |
| Describe fixed hybrid forms | [`LocalForm`, `GlobalForm`, `HybridProblem`](api/hybrid.md#pymhm.core.variational) |
| Store local equations and reconstructed coefficients | [`LocalProblem`, `LocalResponse`, `HybridSolution`](api/hybrid.md#pymhm.core.contracts) |
| Condense each local operator | [Local condensation](api/hybrid.md#pymhm.core.condensation) |
| Accumulate shared macroface contributions | [Global contributions](api/hybrid.md#pymhm.core.contributions) |
| Solve, constrain and reconstruct the global system | [`HybridSystem`](api/hybrid.md#pymhm.core.system) |
| Select serial, threaded or spawned local execution | [`ExecutionConfig`](api/backends.md#pymhm.execution.cpu) |
| Define and tabulate native reference elements | [Basix adapter](api/elements.md#pymhm.fem.reference) |
| Compile user-written local and global UFL forms | [Generic DOLFINx compiler](api/backends.md#pymhm.backends.forms) |
| Scatter arbitrary rectangular element blocks | [`assemble_element_blocks`](api/backends.md#pymhm.fem.assembly) |
| Represent arbitrary complex blocks in real coordinates | [Complex coordinate operations](api/backends.md#pymhm.linalg.complex) |
| Reuse local UFL kernels and worker-owned assembly buffers | [Native form workspaces](api/backends.md#pymhm.backends.workspace) |

The family pages generate signatures and documentation directly from their
owners. They contain the supported spaces, array conventions, orientations,
physical gauges and failure conditions. See the [architecture](architecture.md)
for package responsibilities and [tutorials](tutorials/notebooks.md) for complete scalar,
vector and provider examples.
