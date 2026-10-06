# API reference

The primary interface is `Equation`, `LocalEquations`, `MultiscaleProblem`,
`assemble` and `solve`. User-supplied forms define the operator and its trace
pairings; the package supplies compilation, elimination, ordered assembly and
reconstruction. See the [variational guide](variational.md).

The reference is organized by mathematical formulation and numerical infrastructure.
Each family page contains the complete signatures and docstrings of its documented
objects. Optional dependencies are identified by the relevant adapters; the native
multiscale formulations remain part of the portable package.

Physical family pages also document the predefined formulations used by the
verified case gallery, including their coefficient contracts. Import them
directly from the private `_legacy.models` owners shown on those pages.
The root namespace presents the generic variational and infrastructure API.

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
| [Helmholtz and Maxwell](api/waves.md) | Time-harmonic acoustic and transient electromagnetic formulations. |
| [Solvers and execution backends](api/backends.md) | Finite-element adapters, sparse solvers, parallel execution and separable operators. |

## Common entry points

Use `from pymhm import ...` for high-level problem descriptions, meshes and
generic numerical operations. The package resolves those exports on demand;
importing its root does not import Basix or optional native backends. Import
predefined physical solvers and method-specific operations from the canonical
submodules documented below.

| Task | API owner |
| --- | --- |
| Bind meshes, interface spaces and user forms | `MeshHierarchy`, `bind_interface`, `bind_problem`, `LocalContext`, `GlobalContext` in [hybrid API](api/hybrid.md) |
| Declare custom interface representations | `InterfaceSpace`, `TraceBinding` in [hybrid API](api/hybrid.md) |
| Evaluate named physical fields | `DiscreteField`, `solution_field`, `evaluate_field` in [hybrid API](api/hybrid.md) |
| Declare local and global variational blocks | [`Equation`, `LocalEquations`, `columns`, `rows`](api/hybrid.md#pymhm.core.equations) |
| Assemble, solve and reconstruct a hierarchy | [`MultiscaleProblem`, `NestedEquations`, `assemble`, `solve`](api/hybrid.md#pymhm.core.multiscale) |
| Reuse an assembled hierarchy and recover coefficients | [`with_global_load`, `with_global_equation`, `solve_multiscale_system`, `reconstruct_multiscale`](api/hybrid.md#pymhm.core.multiscale) |
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
| Reuse local UFL kernels and worker-owned assembly buffers | [Native form workspaces](api/backends.md#pymhm.backends.workspace) |

The family pages generate signatures and documentation directly from their
owners. They contain the supported spaces, array conventions, orientations,
physical gauges and failure conditions. See the [architecture](architecture.md)
for package responsibilities and [tutorials](tutorials.md) for complete scalar,
vector and provider examples.
