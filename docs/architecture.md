# Architecture

## Data and operations

Problem descriptions, execution settings, local responses and solutions are
separate objects. Numerical operations are free functions; the existing
`LocalProblem`, `LocalResponse` and `HybridSystem` methods delegate to the same
implementations. Both interfaces use the same represented operators, oriented
maps, executed retained bases and residual checks.

| Responsibility | Objects | Functions |
| --- | --- | --- |
| Describe local variational equations | `LocalForm`, `LocalProblem` | `compile_local_forms`, `fenics.assemble_local_forms` |
| Describe the global problem | `GlobalForm`, `HybridProblem` | `assemble_hybrid`, `solve_hybrid` |
| Execute independent cells | `ExecutionConfig`, callable provider | `iter_local`, `map_local` |
| Eliminate and reconstruct | `LocalResponse`, `SolverConfig` | `condense_local`, `local_condensation_system`, `reconstruct_local`, `reconstruct_response` |
| Reduce and solve global equations | `HybridSystem`, `HybridSolution` | `local_global_contribution`, `assemble_hybrid_contributions`, `solve_hybrid_system`, `hybrid_mean_constraint` |

Each formula has one owner. Offline factors, GPU batching, refinement and
existing PDE drivers reach that owner through the compatible methods or the
free functions. A factorization remains an object with an explicit lifetime;
it owns native resources rather than defining the physical problem.

## Variational descriptions and providers

`LocalForm(a, L, trace_forms, trace_dofs, ...)` describes

$$
a_K(u_K,v_K)+\sum_j\lambda_j b_{K,j}(v_K)=L_K(v_K).
$$

Expressions can be UFL forms or another compiler's input. The portable core
stores them without importing a FEM backend. `compile_local_forms` accepts an
ordinary callable compiler and checks that its assembled `LocalProblem`
preserves the declared trace map and literal kernel or retained basis.
`pymhm.fenics.assemble_local_forms` supplies the DOLFINx compiler. Physical
moment forms, signed trace forms and integration choices remain explicit.
`LocalForm` does not describe independent trial/test trace couplings or left
retained bases. Such Petrov–Galerkin blocks can be supplied directly as a
`LocalProblem` by the same provider interface; the DOLFINx adapter requires
matching trial and test spaces.

`GlobalForm` declares the skeleton dimension, the retained dimension per cell,
boundary moments, prescribed trace coefficients and physical constraint rows.
If `P_K` gathers a cell's trace and retained coordinates, and `S_K`, `g_K` are
its condensed block and load, the global form is

$$
\begin{aligned}
\sum_K(P_Ky)^T S_K(P_Kx)
&=\sum_K(P_Ky)^Tg_K-y_\Lambda^Tg_D,\\
x&=(\lambda,c),\qquad Q^Tx=d.
\end{aligned}
$$

This interface composes the condensed hybrid form in declared coordinates.
It does not compile an arbitrary UFL form on an independent skeleton mesh.
The boundary convention is the same as `HybridSystem`; physical gauges must
describe the complete reconstructed field. Local physical weights can be
converted to a constraint with `hybrid_mean_constraint` after assembly.
Declared arrays preserve their stored precision. Local operators and native
element tabulations use binary64; retaining wider correction digits through a
solve requires the explicit `extended` refinement setting.

A local provider is a callable `provider(item) -> LocalProblem | LocalAssembly`.
It can use portable kernels, UFL/FEniCS or another simulation package. No
framework superclass is required. `HybridProblem` combines that provider, its
ordered items and the global form. `LocalAssembly.metadata` carries portable
evaluation data such as local coordinates, never a live native mesh or factor.

`SolverConfig.local_solver` also accepts a callable on the constrained matrix
and all source/trace/retained-mode right-hand sides. This permits an external
linear solver or response model to supply local coefficients. Every returned
column must satisfy the unchanged original residual criterion before its
response is decoded. Machine-learning accuracy, preCICE coupling and backend
specific discretization stability require their own scientific qualification;
no such qualification follows from implementing a callable.

## Local operators

`LocalProblem(A, B, f, trace_dofs, kernel=Z, constraints=C)` represents

$$
A_K u_K+B_K\lambda_K=f_K,\qquad u_K=w_K+Z_Kc_K,\quad C_K^T w_K=0.
$$

The coupling already contains face orientation signs. The kernel is checked
against `A @ Z` and `A.T @ W`, where `left_kernel=W` defaults to `Z`.
Distinct left and right nullspaces are allowed when their respective constraint
pairings are nonsingular. For L2 orthogonality use `C=M @ Z`.
The represented floating-point actions of declared null modes are retained:
nonzero `A @ Z` or `A.T @ W` uses the complementary response below. Only exactly
vanishing represented actions permit the shorter kernel formula. Accumulation
uses wider precision where available and compensated sums otherwise.

`coarse_basis=Z` instead retains modes that need not be null vectors, with the
same nonsingular moment pairing. This prevents cancellation between large local
responses when reaction or Brinkman drag approaches zero. The operator is not
perturbed: the complementary solves include `A @ Z`, reconstruction uses
`E=Z-R @ A @ Z`, and the global system retains the equations tested against `Z`.
Here `R` denotes the constrained local solve. `kernel` and `coarse_basis` are
mutually exclusive. See the [elimination equations](theory.md#retaining-nearly-null-local-modes)
for the nonsymmetric blocks and reconstruction of integral constraints.

A constrained factorization solves all source and trace right-hand sides together.
`LocalResponse` stores the lifts. `HybridSystem` assembles sparse contributions,
solves the global system and reconstructs every local field. The independent
uncondensed block system is checked in tests to verify the signs of this process.

## Geometry and trace spaces

`TriangleMesh` copies and orients its cells, rejects degenerate and nonmanifold
connectivity, and records signed face incidence. `SkeletonSpace` accepts one
`FaceSpace` for each face. Each `FaceSpace` has arbitrary increasing breaks in
`[0,1]` and one degree per segment; bases are discontinuous Legendre polynomials.
With `continuous=True`, a nodal polynomial basis shares values at segment
endpoints within the same macroface. It does not identify endpoints belonging
to different macrofaces. Vector components are interleaved within each basis
mode. Constant traces are represented through `constant_coefficients()`, which
accounts for the different basis conventions.

Boundary integration splits at both local mesh vertices and skeleton breaks.
Thus local and skeletal partitions can differ for primal local spaces. Increasing
trace dimension without enriching local spaces can violate the discrete inf-sup
condition. Singular systems are errors; no diagonal shift or least-squares solve
is used to conceal them.

`PolygonMesh` retains original macroedges while triangulating local domains,
including simple nonconvex polygons. `PolyhedralMesh` represents convex cells
with planar polygonal faces: shared face triangulations and cell-center cones
define local tetrahedra, while `PolygonalSkeleton3D` keeps one P0 or P1 space
on each **original** face. Triangulating a quadrilateral face therefore does
not introduce independent multipliers on its two triangles. The
[polyhedral RAD cases](https://github.com/volpatto/pymhm/blob/main/docs/cases/polyhedral-rad.md) exercise cubes and two prism
families; [polygonal cases](https://github.com/volpatto/pymhm/blob/main/docs/cases/polygons.md) cover five planar families.

Mapped hexahedral RT fields use a trilinear geometry and contravariant Piola
transformation, with surface Jacobians in normal moments and the physical
Jacobian in divergence. Independent normal and interior degrees distinguish
the rectangular enriched RT spaces from a uniformly raised polynomial degree.
The [well studies](https://github.com/volpatto/pymhm/blob/main/docs/cases/mapped-well.md) report geometric, normal-continuity
and physical-rate checks separately from pressure and flux approximation.

## Local backends

### Reference-element libraries

`pymhm.element_backends` delegates reference-element creation, values,
Cartesian derivatives and entity transformations to
[Basix](https://docs.fenicsproject.org/basix/v0.11.0/python/index.html).
`ReferenceElementSpec` uses the library's own family, cell, degree and variant
names. `create_reference_element` records the native coefficient matrix after
dualization, its digest, mapping, polynomial set and backend version.
`tabulate_reference` supports the installed library's derivative orders and
element shapes; it applies no physical Piola map or global face orientation.
An explicit `dof_ordering` reorders both the archived coefficient matrix and
the full orientation maps into the executed coefficient order. Entity-local
maps keep their intrinsic order; `reference_entity_dofs` identifies their
global slots in that executed order.

```python
from pymhm.element_backends import (
    ReferenceElementSpec, create_reference_element, tabulate_reference,
)

element = create_reference_element(
    ReferenceElementSpec("P", "triangle", 4, lagrange_variant="equispaced")
)
table = tabulate_reference(element, [[0.2, 0.3]], nderiv=2)
```

Basix supplies the built-in interval, triangular and tetrahedral Pk tabulations.
Nodal permutations preserve PyMHM's topological numbering; affine Jacobians map
Cartesian reference derivatives into physical coordinates. The public
barycentric derivative representation uses the extension
`p(lambda_1,...,lambda_d)`, with zero lambda_0 derivatives, on the unit-sum
hyperplane. Cartesian Qk bases use native interval factors in declared tensor
order. The historical `backend="portable"` keyword is a compatibility spelling
for the same Basix implementation.

At literally declared interpolation nodes, nodal values equal their Kronecker
rows. This applies the interpolation functional without a proximity threshold;
nearby points and derivatives use the executed native tables. Tetrahedral face
values use the exact topological support of the nodal trace. Scalar diffusion
contractions accumulate in NumPy's widest real type before returning binary64
matrix entries; quadrature, coefficients and physical operators retain their
declared conventions. On platforms with binary64 `longdouble`, the accumulation
uses that precision.

RT and BDM tabulations use native elements, transformed into the declared
normal/interior moment coordinates. Basix RT degree one denotes mathematical
RT0; simplicial BDM uses its vector polynomial degree. Restricted/enriched MHM
spaces retain the stated normal restriction and all prescribed bubble modes.
Their subspace maps, physical gauges, Piola transformations and face orientations
remain explicit PyMHM responsibilities. Prism spaces use native simplex and
interval polynomial factors. The persisted candidate-coordinate matrices retain
their meaning during field replay.

Conventional Legendre moment tests and declared monomial coordinates delegate
repeated value and derivative evaluation to Basix's orthogonal polynomial sets.
Their normalization and representation maps preserve existing coefficient
conventions. Exports of ascending-power coefficients are representation adapters
for archives; they do not provide a second finite-element tabulator. New
three-layer and elastic-wave archives record the executed native basis matrices;
legacy archives retain consumers for their original persisted representations.

Basix is a runtime dependency loaded at element creation or polynomial
tabulation. DOLFINx, UFL, PETSc and MPI remain separate optional integrations.
The algebraic local-provider and global-assembly contracts do not depend on a
native element handle. The lockfile includes Windows packages; native execution
checks in this workspace run on Linux. FIAT/FInAT can participate through a local
provider or form compiler; a built-in FIAT adapter is not supplied.

### Local finite-element assembly

The built-in FEM assembly uses Basix reference bases. FEniCS
assembles the same `LocalProblem` from UFL forms. Arbitrary forms do not imply
that their discretizations have been analyzed or benchmarked. In particular,
H(div) normal boundary conditions are essential and require a correct lifting,
restriction or augmented formulation.

The native RT0 implementation augments flux and pressure with local boundary
pressure multipliers to prescribe normal flux from the MHM skeleton. It retains
one joint constant-pressure/boundary-pressure null mode. It is a flux-based MHM
local Neumann solve, not a pressure-trace hybridization relabeled as MHM.

The BDM2/P1 Darcy path uses the same flux-prescription principle with quadratic
normal moments and a discontinuous linear divergence space. Its trace breaks
must align with fine boundary edges. The mixed elasticity path uses a BDM2
stress row for each physical component, vector P1 displacement and scalar P1
rotation. Its retained rigid modes include the corresponding rotation
multiplier, and its constraints are physical displacement moments.
The enriched BDM and rectangular RT variants preserve their boundary normal
degree while adding zero-normal interior fields; their displacement and rotation
spaces follow the resulting divergence degree. See the
[mixed-family definitions](https://github.com/volpatto/pymhm/blob/main/docs/cases/mixed-families.md) and
[rectangular weak-symmetry construction](https://github.com/volpatto/pymhm/blob/main/docs/cases/elasticity-tensor-rt.md).

Primal Darcy, conservative RAD and backward Euler use a shared nodal Pk
tabulation, including physical gradients and Hessians. Displacement–pressure
GaLS and equal-order flow use their complete strong residuals; prescribed
coefficient derivatives enter the variable-material terms. A higher polynomial
degree does not automatically validate an arbitrarily enriched trace space.

`reconstruct_darcy_moments` in `pymhm.reconstruction_moments` recovers RT0, RT1
or RT2 fields from skeletal, averaged interior-face and volume moments. Its
source conservation is tested against continuous macro-local polynomials.
`equilibrate_flux` instead adds fine-cell balance constraints to a minimum-energy
RT0 correction. These reconstructions solve different mathematical problems.
`pymhm.estimator` builds a separate conforming Oswald potential and evaluates
the complete four-term energy estimator for identity diffusion. The fine
meshes must join conformingly, and boundary and continuous-test moments are
checked. Its explicit coefficient and boundary restrictions are part of the
API; the recovery is never silently used to replace a plotted original field.

Independent checks use [MSL_MHM with MSL_CG and MSL_Core](https://github.com/volpatto/pymhm/blob/main/docs/cases/reference-comparison.md)
for primal P1 local problems and [NeoPZ](https://github.com/volpatto/pymhm/blob/main/docs/cases/neopz.md) for RT0/P0 mixed
assembly. These comparisons match physical fields and trace spaces rather than
assuming that internal basis coefficients or retained pressure coordinates
have the same meaning in each implementation.
The [GaLS comparison](https://github.com/volpatto/pymhm/blob/main/docs/cases/elasticity-reference.md) also checks displacement,
pressure, gradient and Cauchy stress against native MSL GaLS assembly.

## Physical global gauges

Local constraints select complements during elimination; they do not replace
physical global normalization. Pure-Neumann Darcy fixes one pressure mean.
Pure-traction elasticity fixes all three planar rigid-motion moments. Flow
pressure normalization depends on the boundary conditions and acts on the
complete reconstructed pressure, including contributions from retained modes.

For flow with pure traction and no advection, a semidefinite resistance can
leave only some translations unresisted. Automatic detection uses structurally
zero material columns. A rotated nullspace can be supplied with
`translation_kernel`; its independent columns are orthonormalized and checked
against the material using a componentwise residual scale. `mean_velocity`
cannot prescribe a resisted direction. A small positive coefficient is not
silently replaced by zero.

For conservative RAD with zero reaction and divergence-free velocity tangent
to the exterior boundary, a constant solution has an undetermined mean.
Its discrete representation also requires the half-advection Robin multiplier
to belong to the trace space. A continuous kernel alone does not imply a
discrete kernel for an incompatible trace choice. Gauges are checked against
the original assembled equations after solving.

## Solvers and parallel execution

Global and local solvers are selected independently. All backends check achieved
residuals, including multiple right-hand sides. SciPy and native direct backends
reuse local factorizations. CPU workers preserve result order; process execution
uses `spawn` on all platforms and limits native thread pools.

For SPD elliptic Neumann locals, AMG first projects each load onto the compatible
range, pins independent kernel coordinates, solves the nonsingular principal
system, and restores the physical mean. This preserves sparse matrices without a
dense rank update. AMG is not blindly applied to the indefinite global saddle or
to the local mixed Stokes/Darcy saddle matrices.
The current AMG adapters reject a general `coarse_basis`; its augmented local
system requires a supported direct solver. This restriction does not affect the
AMG path for elliptic locals with a true constant kernel.

PDE drivers can pass assembled local operators to workers for **condensation**,
or use a local factory to distribute assembly as well. The native primal Darcy
convenience path uses the first arrangement; flow, displacement–pressure
elasticity and conservative RAD use local factories.
`HybridSystem.from_local_factory(factory, items, ...)` additionally distributes
**local construction and condensation together**. Each worker calls the factory
once and factors that local problem once for its source and all trace lifts.
The parent assembles the global system from those completed responses.

A factory can return `LocalProblem` alone, or `LocalAssembly(problem, metadata)`.
The latter carries ordinary application data such as a local mesh or DOF map;
`system.local_metadata` preserves input order. Returning the mesh built in the
worker avoids rebuilding it sequentially for field reconstruction. This metadata
does not change the variational operator or the condensation algebra.

Process execution uses portable `spawn`, so the factory, inputs, and metadata
must be picklable. Construct and release native FEM, MPI, PETSc or CUDA resources
inside the worker, returning numerical arrays and ordinary metadata. The serial
and thread variants also accept closures. Exceptions propagate to the caller;
there is no automatic serial fallback. The global assembly and solve remain in
the parent process. See the [measured workloads](https://github.com/volpatto/pymhm/blob/main/docs/performance.md) for costs of
startup, data transfer, local work, and the complete solve.

`solve_distributed` provides a separate MPI path: ranks own local cells, PETSc
owns global matrix rows, and MUMPS solves the distributed saddle matrix.
`OfflineHybridSystem` retains local/global factorizations for changing sources
and boundary values at a fixed operator. `LocalFactorCache` also reuses factors
across exactly identical constrained matrices.

GPU sparse direct solvers upload CPU matrices. The explicit batched GPU API
also provides resident affine P1 volume assembly and dense batched LU with
multiple RHS. It does not distribute work across GPUs or keep skeleton assembly
on the device. See [execution contracts and measurements](execution.md).

## Extension contract

A new local formulation supplies its matrix, signed trace forms, load, kernel and
physical moments. Test its kernel and compatibility independently; then compare
condensed and uncondensed solutions. Add PDE patches, independently derived
sources, convergence and conservation diagnostics before adding a support claim.
