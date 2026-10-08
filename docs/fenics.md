# Defining local problems with FEniCSx

The introductory path binds the macro/local meshes and interface space first.
A provider then receives `LocalContext` and declares its volume and boundary
forms directly with UFL:

The example helpers below are repository sources, supplied separately from the
installed library. Open the corresponding notebook to acquire its verified
companion, as described in the [notebook guide](tutorials.md#execute-downloaded-notebooks).

```python
def local_equations(local):
    binding = local.native_space(element)
    u, v = ufl.TrialFunction(binding.space), ufl.TestFunction(binding.space)
    dx = ufl.Measure("dx", domain=binding.mesh)
    b = local.trace_pairings(lambda phi, ds: phi * v * ds)
    c = local.trace_pairings(lambda phi, ds: -phi * u * ds, axis="rows")
    local.field("pressure", binding)
    return local.equations(
        a=ufl.inner(ufl.grad(u), ufl.grad(v)) * dx, L=f * v * dx,
        b=b, c=c, kernel=constant, moments=columns(v * dx),
    )
```

`element`, `f`, `constant` and the physical kernel moments belong to the user's
formulation. The [complete UFL overview](tutorials/overview.md) defines each
input, the global boundary equation, `bind_problem`, `assemble` and `solve`.
`native_space` owns geometry and coefficient order; the interface binding applies
outward-normal incidence signs once. Physical signs in `b` and `c` remain
independent, explicit mathematical choices.

Register `local.field(...)` before returning `local.equations(...)`, which
snapshots those definitions. `solution.field(name)` returns views in selected
`MeshHierarchy.items` order. Named fields carry their executed basis and mesh. `field.evaluate`,
`field.gradient` and `field.values_and_gradient` support physical values and raw
spatial derivatives, including explicit one-sided fine-cell owners. Scalar
gradients have axes `(point, derivative)`; vector gradients have axes
`(point, component, derivative)`. A raw gradient is not an H(div) reconstruction.

Native mesh bindings cover existing triangles, tetrahedra, Cartesian
quadrilaterals and hexahedra. Automatic portable coefficient maps and named-field
descriptors currently cover supported equispaced nodal Lagrange fields and mixed
components on triangles, tetrahedra and quadrilaterals. Hexahedral native assembly
and direct evaluation remain available, but automatic hex nodal maps and
`local.field(name, hex_space)` descriptors are not provided. Use an explicit
custom named-field evaluator and executed basis contract instead.
Moment-based H(div)/H(curl) elements retain their native conventions unless an
adapter declares its evaluation and basis contract. Automatic local/global
interface UFL capabilities currently use supported planar polynomial face spaces;
local trace breakpoints must align with boundary facets. Arbitrary cross-mesh UFL
forms and custom or three-dimensional interfaces require their declared adapter
capabilities or explicit assembled blocks. See [custom spaces](tutorials/custom-interface.md)
and the [variational guide](variational.md) for exact limits.

Providers construct and compile native resources in their owning worker. No live
DOLFINx object crosses a spawn boundary. Prepared blocks use
`coordinates="global"` when their canonical orientation has already been applied;
raw local UFL pairings use the default local convention.

## Explicit assembled-form interfaces

`LocalEquations` and `MultiscaleProblem` remain available for complete manual
control. `pymhm.backends.forms.assemble_form` compiles real linear and bilinear
UFL forms into owned numerical arrays. `columns` and `rows` supply independent
rectangular pairings on supported native spaces. The local pivot and full global
operator must be square. These interfaces require the explicit basis maps that
the bound introductory path derives from declared spaces.
Explicit cross-mesh `entity_maps` follow the installed DOLFINx version:
mesh-to-entity-index mappings for 0.9, or sequences of native `EntityMap` objects
for 0.10. See [native compilation limits](variational.md#native-compilation-limits).

## Fixed hybrid local-form adapter

`LocalForm` records local variational expressions and explicit trace maps;
`pymhm.backends.fenics.assemble_local_forms` compiles them using DOLFINx and returns an
ordinary `LocalProblem`. The lower-level `pymhm.backends.fenics.from_ufl` accepts the same
assembly inputs directly. Global skeleton numbering and condensation remain
independent of DOLFINx. The same adapter accepts scalar, vector, mixed and
H(div) spaces, provided the supplied local equations and constraints are
well posed.

Here FEniCSx means [DOLFINx](https://docs.fenicsproject.org/dolfinx/) assembly
of forms expressed in [UFL](https://docs.fenicsproject.org/ufl/). The
[Darcy reference comparison](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/darcy-audit.md) records DOLFINx 0.9.0. The
examples assemble their local forms directly through UFL and DOLFINx.

Use the optional environment:

```bash
pixi run --locked -e fem pytest -m fem
```

The adapter uses the native DOLFINx CSR/vector assembly interface and requires
DOLFINx 0.9 or 0.10. It does not need PETSc for assembly. Meshes must have a
single-rank communicator; `MPI.COMM_SELF` makes that choice explicit. Global
MPI-distributed assembly of a local mesh is not implemented by this adapter.
The assembled arrays can be condensed using the serial, thread or spawn-process
execution modes and any compatible linear-solver backend. GPU solving does not
move finite-element assembly to the GPU.

The locked `fem` profile resolves this assembly stack on Linux, macOS and
Windows. DOLFINx still needs an MPI runtime, and UFL form compilation needs
a working native C compiler; on Windows, use Visual Studio's C/C++ compiler
and Windows SDK in a developer terminal. See
[installation](installation.md#native-ufl-assembly) and
[Windows FEM requirements](windows.md#native-fem-scope).
PETSc/MUMPS is an optional numerical solver capability supplied by the Unix
FEM profiles, rather than a requirement of either UFL adapter.

Select solvers independently of the provider, for example:

```python
from pymhm import SolverConfig, solve

solution = solve(
    problem,
    solvers=SolverConfig(local_solver="scipy", global_solver="scipy"),
)
```

SciPy is the portable default. The locked `fem-intel` profile combines native
assembly and PARDISO on Linux and Windows. Select
`SolverConfig(local_solver="pypardiso", global_solver="scipy")` for local
PARDISO factors, or choose `global_solver="pypardiso"` independently.
Symmetric-indefinite presets, Krylov and AMG choices retain the operator
restrictions in the [solver guide](solvers.md). Solver selection does not change
the UFL form, local kernel, physical moments or trace orientation. Requesting
`petsc` requires PETSc/MUMPS explicitly and does not silently select another solver.

The [DOLFINx/UFL sparse-solver notebook](https://github.com/ipes-lncc/pymhm/blob/main/notebooks/foundations/operators/dolfinx_sparse_solvers.ipynb)
demonstrates primal P1 Darcy with independent local/global SciPy and PARDISO
choices, signed normal-flux traces, integral pressure moments and spawn workers.
Run it with `pixi run --locked -e introduction-intel notebooks-run foundations/operators/dolfinx_sparse_solvers.ipynb`.

## Local form contract

The adapter implements

$$
A_K u_K+B_K\lambda_K=f_K.
$$

`a` defines \(A_K\), `load` defines \(f_K\), and every entry of `trace_forms`
defines one column of \(B_K\). `trace_dofs` maps those columns to the global
skeleton. For this explicit `LocalForm`/`from_ufl` interface, a trace form must contain its orientation sign already. The bound `LocalContext` path above applies its declared geometric map instead. For primal
Darcy, a face basis \(\psi_j\) produces
\(B_{K,j}(v)=s_{K,F}\int_F\psi_jv\), where \(s_{K,F}\) relates the outward
normal of the local cell to the fixed global face normal.

`kernel` is an explicit coefficient array with one nullspace vector per column.
It is checked against both sides of the local matrix. `constraint_forms` defines
physical moments of the constrained local lifts. For scalar Neumann diffusion,
`kernel=np.ones((V.dofmap.index_map.size_local, 1))` and
`constraint_forms=[v * dx]` enforce zero integral. The default constraint uses
coefficient-space orthogonality, which generally differs from a zero physical
mean. Mixed-space kernels must follow DOLFINx's mixed coefficient ordering;
use collapsed-subspace maps when constructing them.

For small positive reaction or drag, use `coarse_basis=Z` in place of `kernel=Z`
to retain nearly null modes without declaring them exact null vectors. The same
`constraint_forms` defines one physical moment per retained column. For example,
scalar reaction–diffusion retains the constant coefficient vector with `[v * dx]`.
The adapter passes these choices to the exact
[coarse elimination](theory.md#retaining-nearly-null-local-modes); it does not
infer which modes are needed. Native tests compare a nonsymmetric UFL problem
near the Poisson limit with an independent uncondensed saddle solve.

The following assembles a local scalar Neumann response with four separately
marked boundary traces. The local field remains free on its boundary.

```python
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from pymhm.backends.fenics import assemble_local_forms
from pymhm.core.variational import LocalForm

local_mesh = mesh.create_unit_square(MPI.COMM_SELF, 4, 4)
V = fem.functionspace(local_mesh, ("Lagrange", 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
dx = ufl.dx(domain=local_mesh)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
load = 0.0 * v * dx

predicates = [
    lambda x: np.isclose(x[0], 0.0),
    lambda x: np.isclose(x[0], 1.0),
    lambda x: np.isclose(x[1], 0.0),
    lambda x: np.isclose(x[1], 1.0),
]
facets, values = [], []
for marker, predicate in enumerate(predicates, start=1):
    found = mesh.locate_entities_boundary(local_mesh, 1, predicate)
    facets.extend(found)
    values.extend([marker] * len(found))
order = np.argsort(facets)
tags = mesh.meshtags(
    local_mesh, 1,
    np.asarray(facets, dtype=np.int32)[order],
    np.asarray(values, dtype=np.int32)[order],
)
ds = ufl.Measure("ds", domain=local_mesh, subdomain_data=tags)

# These signs assume every global face normal is locally outward.
# Interior macrofaces must use opposite signs in their two adjacent cells.
trace_forms = [v * ds(marker) for marker in range(1, 5)]
local = assemble_local_forms(
    LocalForm(
        a, load, tuple(trace_forms), np.arange(4),
        kernel=np.ones((V.dofmap.index_map.size_local, 1)),
        moment_forms=(v * dx,),
    )
)
response = local.condense()
```

The integration test `test_two_subdomain_darcy_affine_reconstruction` provides a
complete two-macrotriangle problem: it creates independent fine meshes, assembles
signed interface traces, compares local matrices and moments with the native P1
implementation, solves the common skeleton system, and recovers the exact affine
pressure and globally oriented flux.

## Local provider and global hybrid form

An ordinary callable `provider(cell)` supplies a `LocalProblem` or
`LocalAssembly(problem, metadata)`. No provider base class is required.
`compile_local_forms(forms, compiler)` also accepts an ordinary callable
compiler and checks that it returns a validated local problem with the exact
declared trace map and retained coefficient basis. A rotated basis is a
different coefficient contract, even when it spans the same space. Compilation
does not certify quadrature accuracy or stability of the selected spaces.

`GlobalForm` declares the global trace size, the retained width of each cell,
boundary loads, prescribed trace coefficients and physical gauge rows. Its
bilinear and linear forms are the sum of the condensed local records. Writing
the local reduced trial/test vectors as `x_K` and `y_K`, their blocks and loads
as `S_K` and `h_K`, and the boundary trace load as `g_D`, this means

$$
\begin{aligned}
\mathcal B(x,y) &= \sum_K y_K^\mathsf{T} S_K x_K,\\
\mathcal L(y) &= \sum_K y_K^\mathsf{T} h_K-\mu^\mathsf{T}g_D,\\
r_j^\mathsf{T}x &= m_j.
\end{aligned}
$$

Here `x=(lambda,c)` contains the trace and ordered cell-retained coefficients,
`mu` is the trace part of the test vector, and `(r_j,m_j)` are declared physical
constraints. The blocks come from `LocalResponse.global_contribution`, including
its signed hybrid saddle convention. This scoped algebraic form uses the shared
hybrid assembler; it does not compile arbitrary UFL expressions on the global
skeleton. `boundary_load` follows the pressure-boundary sign convention above,
and `fixed_trace` prescribes oriented flux coefficients. Physical local moments
can be converted to global rows with `system.mean_constraint` after assembly.

The complete affine Darcy example uses continuous local P1 pressure, one
constant moment per macrotriangle and constant signed face traces. It recovers
`p=1+x+2y` and Darcy flux `(-1,-2)` on the unit square. For its native provider,
the same local expressions shown above are constructed inside each worker.

```python
from examples.variational_darcy import DarcyProvider, affine_pressure
from pymhm import Equation, MultiscaleProblem, solve
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh
from pymhm.execution.cpu import ExecutionConfig
import numpy as np

macro_mesh = TriangleMesh.unit_square()
skeleton = SkeletonSpace(macro_mesh)
provider = DarcyProvider(macro_mesh, skeleton, kind="fenics", subdivisions=2)
boundary, _ = boundary_data(skeleton, affine_pressure)
problem = MultiscaleProblem(
    Equation(0, -np.r_[boundary, np.zeros(2)]), provider, range(2),
    trace_size=skeleton.size, coarse_sizes=(1, 1),
)
solution = solve(problem, execution=ExecutionConfig())
```

Run the provider notebook and select its assembly provider:

```bash
pixi run --locked -e notebooks notebooks-run foundations/operators/ufl_provider.ipynb
```

The `backend` choices are `serial`, `thread` and `process`; `batch_size`
bounds work submitted together in parallel, while serial execution solves one
cell at a time. Shared faces are accumulated by the coordinator in cell order.
The Schur contribution is computed with its local solve on the worker.
`pipeline=True` selects a bounded rolling window rather than complete batches;
the order of global sums stays fixed. See [execution policies](execution.md)
for failure delivery, thread limits and resource ownership.
Responses remain available for reconstruction, so this bound applies to
in-flight work rather than the total stored response and global matrix size.
Spawn execution requires a top-level picklable provider and a guarded script
entry point, as in this example. Its DOLFINx provider creates a `COMM_SELF`
mesh inside the invocation and returns only numerical arrays and point
metadata. Live UFL, DOLFINx, MPI and factorization objects do not cross workers.
Native thread safety remains the provider's responsibility in thread mode;
separate processes give each invocation its own native state.

Native provider tests execute serial and actual spawn-process assembly with
batch sizes one and two. They compare the same local operators, signed
couplings, integral moments, global records and affine fields against the
portable P1 implementation, with the DOLFINx coefficient permutation stated
explicitly. This verifies the common discrete problem and scheduling contract.

## Predefined volume-form helpers

These helpers retain the established physical conventions and native
integration tests. New user-defined equations can write their UFL forms
directly and use the generic compiler above.

| Helper | Unknowns and equation | Boundary/nullspace responsibility |
|---|---|---|
| `primal_darcy_forms` | Pressure, \(-\nabla\cdot(K\nabla p)=f\) | Normal Darcy flux multiplies the pressure test; one constant local mode for pure diffusion. |
| `mixed_darcy_forms` | H(div) Darcy flux and L2 pressure, \(K^{-1}u+\nabla p=0\), \(\nabla\cdot u=f\) | Pressure is natural data, normal Darcy flux is essential; RT/DG stability and essential constraints must be supplied. |
| `brinkman_forms` | H1 velocity and pressure, \(-\nabla\cdot(2\mu\varepsilon(u))+Ru+\nabla p=f\) | Select a stable pair and supply traction, the actual local kernel and any required global pressure gauge. Rigid motions form the kernel in the zero-drag Stokes limit; positive-definite drag removes them. |
| `elasticity_forms` | Displacement, stress \(2\mu\varepsilon(u)+\lambda\operatorname{div}(u)I\) | Supply rigid motions and signed traction forms. The 2D default interpretation is plane strain. |
| `usfem_brinkman_forms` | Equal-order velocity/pressure with vector-Laplacian pseudostress and residual stabilization | Use compatible pseudotractions, the stated inverse estimate and the appropriate global pressure constraint. |

The mixed Darcy and incompressible helpers negate the pressure test equation to
produce a symmetric saddle operator. Accordingly, a divergence source \(g\)
contributes \(-\int gq\). For mixed Darcy, imposed pressure contributes
\(-\int_{\partial K}p_D(v\cdot n_K)\). The native integration test checks this
sign by recovering constant pressure and zero RT Darcy flux.

The adapter does not infer essential normal-flux constraints for H(div)
spaces. An arbitrary mixed volume form plus a boundary pressure coupling is not
a replacement for the flux-multiplier MHM construction: choose the intended
hybrid formulation and enforce its essential conditions through elimination or
an appropriate augmented local system.

## USFEM Stokes–Brinkman formulation

`usfem_brinkman_forms` implements equations (41)–(42) of
[Araya, Harder, Poza and Valentin (2017)](https://doi.org/10.1016/j.cma.2017.05.027),
with the pressure test sign changed consistently. Define

$$
R(u,p)=-\nu\Delta u+\Theta u+\nabla p.
$$

The bilinear form and source are

$$
\begin{aligned}
B((u,p),(v,q))={}&(\nu\nabla u,\nabla v)+(\Theta u,v)
\\
&-(p,\nabla\cdot v)-(q,\nabla\cdot u)\\
&-\sum_\tau\kappa_\tau(R(u,p),R(v,q))_\tau,\\
L(v,q)={}&(f,v)-(g,q)-\sum_\tau\kappa_\tau(f,R(v,q))_\tau.
\end{aligned}
$$

The stabilization has a **negative** sign. With
\(m_\tau=\min(1/3,C_\tau)\), the implementation uses the equivalent expression

$$
\kappa_\tau=
\frac{h_\tau^2}{\max(\gamma_{\min}h_\tau^2,4\nu/m_\tau)+4\nu/m_\tau}.
$$

Here \(\gamma_{\min}\) is the smallest eigenvalue of the resistance tensor.
For Stokes and a P1 pair this reduces to \(h_\tau^2/(24\nu)\); an integration
test checks the assembled pressure block against that exact coefficient.
Viscosity must be constant per cell, and a higher-order pair requires a justified
inverse estimate. Both viscosity and resistance cannot vanish simultaneously.
The helper does not validate coercivity of user-provided coefficient fields.

`pymhm.fem.inequalities.laplacian_inverse_bound` computes the scalar polynomial
inverse quotient from physical gradients, Hessians, cell diameters and exact
Gram-matrix quadrature. The same owner serves two- and three-dimensional flow
assembly and the introductory polynomial-family comparison. The quotient
requires one resolved constant kernel; it does not certify the separate
velocity–pressure or skeletal approximation hypotheses.
The a priori analysis of
[Araya, Harder, Poza and Valentin (2025)](https://doi.org/10.1137/24M1649368)
requires its local and skeletal compatibility conditions and justified inverse
inequalities; assembling the displayed form alone does not verify them.

This form uses \(\nu\nabla u:\nabla v\), so its natural boundary quantity is
pseudostress. The physical symmetric-stress form in `brinkman_forms` has different
boundary terms and, in the Stokes limit, different rigid-motion modes. They must
not be interchanged while keeping the same boundary and kernel definitions.
See [Araya, Harder, Poza and Valentin (2017)](https://doi.org/10.1016/j.cma.2017.05.027) and
[the literature map](literature.md).
The [high-order reproduction](https://github.com/ipes-lncc/pymhm/blob/main/docs/cases/reproduction.md#stokes-2017-equal-order-local-spaces-and-direct-figure-comparison)
used P2/P2 and P3/P3 local spaces through DOLFINx with a computed inverse
estimate, then compared their numerical errors with digitized article curves.
These calculations use pyMHM with DOLFINx local assembly; they do not execute
the authors' Stokes program.

## Validation boundary

Core tests exercise real UFL expression construction and explicitly simulated
DOLFINx API contracts. The `fem` test suite separately executes actual DOLFINx
assembly: scalar MHM affine reconstruction, RT/DG pressure-boundary signs,
Taylor–Hood and elasticity symmetry, and the signed USFEM pressure block.
An additional comparison maps DOLFINx mixed DOFs to the portable ordering and
checks the complete USFEM matrices and loads for scalar drag 0, 3 and 1000,
including both branches of the stabilization parameter and the drag cross-terms.
These checks establish the implemented local form and coupling contracts. They
do not establish all stability estimates or reproduce every benchmark in the
literature. DOLFINx's assembly APIs are documented in the
[official finite-element reference](https://docs.fenicsproject.org/dolfinx/v0.9.0/python/generated/dolfinx.fem.html).
Native tests also run assembly and local/global SciPy and optional PARDISO
solves in a fresh process that rejects imports of `petsc4py` and
`dolfinx.fem.petsc`. This verifies the assembly dependency boundary using the
installed DOLFINx runtime. Platform qualification requires that platform's
actual test reports for the identified revision; a locked Windows resolution
or a Linux run alone does not establish native Windows execution.

## References

- Rodolfo Araya, Christopher Harder, Abner H. Poza, and Frédéric Valentin (2017). *Multiscale hybrid-mixed method for the Stokes and Brinkman equations—The method*, Computer Methods in Applied Mechanics and Engineering 324, 29–53. [DOI: 10.1016/j.cma.2017.05.027](https://doi.org/10.1016/j.cma.2017.05.027).

- Rodolfo Araya, Christopher Harder, Abner H. Poza, and Frédéric Valentin (2025). *Multiscale Hybrid-Mixed Methods for the Stokes and Brinkman Equations—A Priori Analysis*, SIAM Journal on Numerical Analysis 63(2), 588–618. [DOI: 10.1137/24M1649368](https://doi.org/10.1137/24M1649368).
