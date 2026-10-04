# Architecture

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

The portable FEM kernels provide auditable reference implementations. FEniCS
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
