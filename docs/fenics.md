# Defining local problems with FEniCSx

`LocalForm` records local variational expressions and explicit trace maps;
`pymhm.fenics.assemble_local_forms` compiles them using DOLFINx and returns an
ordinary `LocalProblem`. The lower-level `pymhm.fenics.from_ufl` accepts the same
assembly inputs directly. Global skeleton numbering and condensation remain
independent of DOLFINx. The same adapter accepts scalar, vector, mixed and
H(div) spaces, provided the supplied local equations and constraints are
well posed.

Here FEniCSx means [DOLFINx](https://docs.fenicsproject.org/dolfinx/) assembly
of forms expressed in [UFL](https://docs.fenicsproject.org/ufl/). The
[Darcy reference comparison](https://github.com/volpatto/pymhm/blob/main/docs/cases/darcy-audit.md) records DOLFINx 0.9.0. The
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

## Local form contract

The adapter implements

$$
A_K u_K+B_K\lambda_K=f_K.
$$

`a` defines \(A_K\), `load` defines \(f_K\), and every entry of `trace_forms`
defines one column of \(B_K\). `trace_dofs` maps those columns to the global
skeleton. A trace form must contain its orientation sign already. For primal
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
from pymhm.fenics import assemble_local_forms
from pymhm.variational import LocalForm

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
from pymhm.assembly import HybridProblem, solve_hybrid
from pymhm.elements import boundary_data
from pymhm.mesh import SkeletonSpace, TriangleMesh
from pymhm.parallel import ExecutionConfig
from pymhm.variational import GlobalForm

macro_mesh = TriangleMesh.unit_square()
skeleton = SkeletonSpace(macro_mesh)
provider = DarcyProvider(macro_mesh, skeleton, kind="fenics", subdivisions=2)
boundary, _ = boundary_data(skeleton, affine_pressure)
global_form = GlobalForm(
    skeleton.size, coarse_sizes=(1, 1), boundary_load=boundary,
)
problem = HybridProblem(global_form, provider, range(2))
solution = solve_hybrid(problem, execution=ExecutionConfig())
```

Run the complete script with either assembly provider:

```bash
pixi run --locked -e test python -m examples.variational_darcy --provider portable
pixi run --locked -e fem python -m examples.variational_darcy --provider fenics --backend process --workers 2 --batch-size 1
```

The `--backend` choices are `serial`, `thread` and `process`; `--batch-size`
bounds work submitted together in parallel, while serial execution solves one
cell at a time. Shared faces are accumulated by the coordinator in cell order.
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

## Volume-form helpers

| Helper | Unknowns and equation | Boundary/nullspace responsibility |
|---|---|---|
| `primal_darcy_forms` | Pressure, \(-\nabla\cdot(K\nabla p)=f\) | Normal Darcy flux multiplies the pressure test; one constant local mode for pure diffusion. |
| `mixed_darcy_forms` | H(div) velocity and L2 pressure, \(K^{-1}u+\nabla p=0\), \(\nabla\cdot u=f\) | Pressure is natural data, normal velocity is essential; RT/DG stability and essential constraints must be supplied. |
| `brinkman_forms` | H1 velocity and pressure, \(-\nabla\cdot(2\mu\varepsilon(u))+Ru+\nabla p=f\) | Select a stable pair and supply traction, the actual local kernel and any required global pressure gauge. Rigid motions form the kernel in the zero-drag Stokes limit; positive-definite drag removes them. |
| `elasticity_forms` | Displacement, stress \(2\mu\varepsilon(u)+\lambda\operatorname{div}(u)I\) | Supply rigid motions and signed traction forms. The 2D default interpretation is plane strain. |
| `usfem_brinkman_forms` | Equal-order velocity/pressure with vector-Laplacian pseudostress and residual stabilization | Use compatible pseudotractions, the stated inverse estimate and the appropriate global pressure constraint. |

The mixed Darcy and incompressible helpers negate the pressure test equation to
produce a symmetric saddle operator. Accordingly, a divergence source \(g\)
contributes \(-\int gq\). For mixed Darcy, imposed pressure contributes
\(-\int_{\partial K}p_D(v\cdot n_K)\). The native integration test checks this
sign by recovering constant pressure and zero RT velocity.

The adapter does not infer essential normal-velocity constraints for H(div)
spaces. An arbitrary mixed volume form plus a boundary pressure coupling is not
a replacement for the flux-multiplier MHM construction: choose the intended
hybrid formulation and enforce its essential conditions through elimination or
an appropriate augmented local system.

## USFEM Stokes–Brinkman formulation

`usfem_brinkman_forms` implements equations (41)–(42) of the 2017 construction
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

This form uses \(\nu\nabla u:\nabla v\), so its natural boundary quantity is
pseudostress. The physical symmetric-stress form in `brinkman_forms` has different
boundary terms and, in the Stokes limit, different rigid-motion modes. They must
not be interchanged while keeping the same boundary and kernel definitions.
See [the published construction](https://doi.org/10.1016/j.cma.2017.05.027) and
[the literature map](literature.md).
The [high-order reproduction](https://github.com/volpatto/pymhm/blob/main/docs/cases/reproduction.md#stokes-2017-equal-order-local-spaces-and-direct-figure-comparison)
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
